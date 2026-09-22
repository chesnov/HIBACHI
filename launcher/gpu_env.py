"""
gpu_env -- choose how HIBACHI's viewer renders, before the app starts.

Windows only. On every other platform `prepare()` does nothing and returns a
report saying so, so the launcher can call it unconditionally.

Why this exists
---------------
napari draws through OpenGL (via vispy). On Windows that can quietly go wrong in
ways that look like bugs in HIBACHI:

* Remote Desktop, a virtual machine, or a missing vendor driver ("Microsoft Basic
  Display Adapter") leave only Windows' legacy OpenGL 1.1, and the viewer crashes
  on open ("glBindFramebuffer not found"). The launcher has long had a software
  fallback for this, but it was opt-in (HIBACHI_SOFTWARE_OPENGL=1), so a user had
  to already know the answer.
* On laptops with integrated AND discrete graphics, Python may be put on the
  integrated GPU, so large volumes render slowly and exhaust shared memory.

What it does
------------
1. Lists the graphics adapters and their drivers (a WMI query via PowerShell).
2. On a hybrid machine, asks Windows to run HIBACHI's interpreter on the
   high-performance GPU -- the per-app setting Settings > Display > Graphics
   writes. Per user, no admin rights, and an existing choice is never changed.
3. Opens an OpenGL context in a THROWAWAY process, started with the same
   interpreter the app uses, so it gets the same GPU choice, and reads back what
   OpenGL really delivers. A separate process because a broken driver can abort
   the process that asks it for a context; that must not be the launcher.
4. Decides: hardware rendering, or the bundled software renderer.
5. Lists problems worth telling the user about, each with the fix.

The decision comes from step 3 ALONE. Remote Desktop and virtual machines only
change the wording of a message, because some of them do provide a GPU. A probe
that fails for reasons unrelated to graphics -- a module that will not import --
is "inconclusive" and changes nothing, so this can never make a working machine
worse than it was before this module existed.

Results are cached per machine state (adapters, driver versions, interpreter,
remote session), so the probe's few seconds are paid only when something
changed.

Overrides
---------
HIBACHI_SOFTWARE_OPENGL=1   always software rendering (as before)
HIBACHI_SOFTWARE_OPENGL=0   always hardware rendering; skip the probe
HIBACHI_GPU_PROBE=0         skip all of this and launch exactly as before
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_CREATE_NO_WINDOW = 0x08000000 if sys.platform.startswith("win") else 0

#: Bump when the cache's meaning changes, so old entries are ignored.
_CACHE_VERSION = 1
_CACHE_FILE = "gpu_probe.json"

#: The probe imports Qt and vispy and opens a context: a few seconds normally.
#: A driver that hangs on context creation would otherwise hang the launch.
_GL_PROBE_TIMEOUT_S = 45
_WMI_TIMEOUT_S = 20

#: Minimum OpenGL napari/vispy need.
_MIN_GL = (2, 1)

#: Renderer strings that mean "no GPU is doing this".
_SOFTWARE_RENDERERS = ("gdi generic", "llvmpipe", "softpipe", "swrast",
                       "microsoft basic render", "swiftshader", "software rasterizer")

#: Adapters that are not graphics hardware HIBACHI could render on.
_VIRTUAL_ADAPTERS = ("remote display", "hyper-v", "vmware svga", "virtualbox",
                     "parsec", "displaylink", "citrix", "idd", "virtual display",
                     "spacedesk")

_DRIVER_PAGES = {
    "nvidia": "https://www.nvidia.com/Download/index.aspx",
    "amd": "https://www.amd.com/en/support/download/drivers.html",
    "intel": "https://www.intel.com/content/www/us/en/support/detect.html",
}
_WINDOWS_UPDATE = "ms-settings:windowsupdate-optionalupdates"

_PCI_VENDORS = {"10DE": "nvidia", "1002": "amd", "1022": "amd", "8086": "intel"}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class Adapter:
    name: str
    vendor: str            # nvidia | amd | intel | other
    kind: str              # discrete | integrated | basic | virtual
    driver_version: str = ""
    driver_date: str = ""  # YYYY-MM-DD, or ""
    has_driver: bool = True


@dataclass
class GLInfo:
    status: str            # ok | no_context | crashed | timeout | inconclusive
    vendor: str = ""
    renderer: str = ""
    version: str = ""
    detail: str = ""

    def version_tuple(self):
        m = re.match(r"\s*(\d+)\.(\d+)", self.version or "")
        return (int(m.group(1)), int(m.group(2))) if m else None

    @property
    def is_software(self) -> bool:
        low = (self.renderer or "").lower()
        return any(s in low for s in _SOFTWARE_RENDERERS)


@dataclass
class Issue:
    id: str
    title: str
    message: str
    link: str = ""
    link_label: str = ""


@dataclass
class Report:
    mode: str = "unchanged"        # hardware | software | unchanged
    reason: str = ""
    adapters: List[Adapter] = field(default_factory=list)
    gl: Optional[GLInfo] = None
    remote_session: bool = False
    virtual_machine: bool = False
    gpu_preference_set: List[str] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    from_cache: bool = False

    def summary(self) -> str:
        gl = self.gl
        where = (f"{gl.renderer} (OpenGL {gl.version})" if gl and gl.renderer
                 else (gl.status if gl else "not probed"))
        return f"rendering={self.mode}; {where}; reason: {self.reason or '-'}"


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
_WMI_SCRIPT = r"""
$ErrorActionPreference = 'SilentlyContinue'
$v = @(Get-CimInstance Win32_VideoController | ForEach-Object {
    [pscustomobject]@{
        Name = $_.Name
        AdapterCompatibility = $_.AdapterCompatibility
        DriverVersion = $_.DriverVersion
        DriverDate = $(if ($_.DriverDate) { $_.DriverDate.ToString('yyyy-MM-dd') } else { '' })
        ConfigManagerErrorCode = $_.ConfigManagerErrorCode
        PNPDeviceID = $_.PNPDeviceID
    }
})
$c = Get-CimInstance Win32_ComputerSystem
[pscustomobject]@{
    adapters = $v
    manufacturer = $c.Manufacturer
    model = $c.Model
} | ConvertTo-Json -Depth 4 -Compress
"""


def classify_adapter(raw: Dict[str, Any]) -> Adapter:
    """One Win32_VideoController record as an Adapter. Pure, so it is testable."""
    name = str(raw.get("Name") or "").strip()
    low = name.lower()
    compat = str(raw.get("AdapterCompatibility") or "").lower()
    pnp = str(raw.get("PNPDeviceID") or "").upper()

    m = re.search(r"VEN_([0-9A-F]{4})", pnp)
    vendor = _PCI_VENDORS.get(m.group(1), "other") if m else "other"
    if vendor == "other":
        for key in ("nvidia", "amd", "intel"):
            if key in low or key in compat or (key == "amd" and "radeon" in low):
                vendor = key
                break

    err = raw.get("ConfigManagerErrorCode")
    has_driver = err in (0, None, "0", "")

    if ("microsoft basic" in low or "standard display" in compat
            or "standard vga" in low):
        kind, has_driver = "basic", False
    elif any(v in low for v in _VIRTUAL_ADAPTERS):
        kind = "virtual"
    elif vendor == "nvidia":
        kind = "discrete"
    elif vendor == "intel":
        kind = "discrete" if re.search(r"\barc\b", low) else "integrated"
    elif vendor == "amd":
        # AMD APUs report a bare "Radeon(TM) Graphics" / "Radeon Vega 8 Graphics"
        # / "Radeon 780M"; discrete cards carry a model family (RX, Pro, FirePro).
        integrated = (re.search(r"radeon\s*(\(tm\))?\s*(vega\s*\d*\s*)?graphics", low)
                      or re.search(r"radeon\s*\d{3,4}m\b", low))
        kind = "integrated" if integrated else "discrete"
    else:
        kind = "integrated"

    return Adapter(name=name or "unknown adapter", vendor=vendor, kind=kind,
                   driver_version=str(raw.get("DriverVersion") or ""),
                   driver_date=str(raw.get("DriverDate") or ""),
                   has_driver=bool(has_driver))


def parse_wmi_output(text: str):
    """(adapters, is_virtual_machine) from the PowerShell JSON. Pure."""
    try:
        data = json.loads(text or "{}")
    except ValueError:
        return [], False
    raw = data.get("adapters") or []
    if isinstance(raw, dict):              # PowerShell collapses a 1-item array
        raw = [raw]
    adapters = [classify_adapter(r) for r in raw if isinstance(r, dict)]
    system = f"{data.get('manufacturer') or ''} {data.get('model') or ''}".lower()
    vm = any(s in system for s in ("vmware", "virtualbox", "qemu", "kvm",
                                   "virtual machine", "xen", "parallels",
                                   "bochs"))
    return adapters, vm


def _query_adapters():
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-Command", _WMI_SCRIPT],
            capture_output=True, text=True, timeout=_WMI_TIMEOUT_S,
            creationflags=_CREATE_NO_WINDOW)
        return parse_wmi_output(out.stdout)
    except Exception:
        return [], False


def _is_remote_session() -> bool:
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetSystemMetrics(0x1000))  # SM_REMOTESESSION
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# GPU preference (hybrid laptops)
# --------------------------------------------------------------------------- #
_GPU_PREF_KEY = r"Software\Microsoft\DirectX\UserGpuPreferences"
_HIGH_PERFORMANCE = "GpuPreference=2;"


def interpreter_paths(executable: Optional[str] = None) -> List[str]:
    """The interpreter the app runs under, and its console/windowless twin.

    Both, because the shortcut starts pythonw.exe and a console launch starts
    python.exe, and the preference is recorded per executable path.
    """
    exe = os.path.abspath(executable or sys.executable)
    folder = os.path.dirname(exe)
    out = [exe]
    for twin in ("python.exe", "pythonw.exe"):
        path = os.path.join(folder, twin)
        if os.path.isfile(path) and os.path.normcase(path) not in \
                {os.path.normcase(p) for p in out}:
            out.append(path)
    return out


def ensure_high_performance_gpu(adapters: List[Adapter], exes: List[str],
                                winreg_module=None) -> List[str]:
    """Ask Windows to run `exes` on the discrete GPU. Returns the paths set.

    Only on a machine that actually has a choice (a discrete AND an integrated
    GPU). A preference already recorded for a path -- by the user in Settings,
    or by an earlier run -- is left exactly as it is.
    """
    kinds = {a.kind for a in adapters if a.has_driver}
    if not ("discrete" in kinds and "integrated" in kinds):
        return []
    try:
        winreg = winreg_module
        if winreg is None:
            import winreg  # type: ignore
    except ImportError:
        return []
    written: List[str] = []
    try:
        key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, _GPU_PREF_KEY, 0,
                                 winreg.KEY_READ | winreg.KEY_WRITE)
    except OSError:
        return []
    try:
        for exe in exes:
            try:
                winreg.QueryValueEx(key, exe)
                continue                    # a choice exists: respect it
            except OSError:
                pass
            try:
                winreg.SetValueEx(key, exe, 0, winreg.REG_SZ, _HIGH_PERFORMANCE)
                written.append(exe)
            except OSError:
                pass
    finally:
        try:
            winreg.CloseKey(key)
        except Exception:
            pass
    return written


# --------------------------------------------------------------------------- #
# OpenGL probe (runs in a child process: `python gpu_env.py --gl-probe`)
# --------------------------------------------------------------------------- #
def _gl_probe_child() -> int:
    """Open an OpenGL context the way the viewer will, and print what it is.

    Prints one JSON line. Exit status is irrelevant: a crash is detected by the
    parent as missing output, which is the point of running it separately.
    """
    def emit(**kw):
        print("GLPROBE " + json.dumps(kw), flush=True)

    # Same DLL setup as segment.py, or PyQt5 cannot load on a clean machine and
    # the probe would be "inconclusive" for a reason that has nothing to do
    # with graphics.
    if sys.platform == "win32":
        for d in (sys.prefix, os.path.join(sys.prefix, "Library", "bin")):
            if os.path.isdir(d):
                try:
                    os.add_dll_directory(d)
                except OSError:
                    pass
    try:
        from PyQt5.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext
        from vispy.gloo import gl
    except Exception as exc:
        emit(status="inconclusive", detail=f"import failed: {exc}")
        return 0

    try:
        app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
        surface = QOffscreenSurface()
        surface.create()
        ctx = QOpenGLContext()
        if not ctx.create() or not ctx.makeCurrent(surface):
            emit(status="no_context", detail="Qt could not create an OpenGL context")
            return 0
        # vispy's own function table, loaded from the same library napari
        # will use, so this sees exactly what the viewer will see.
        emit(status="ok",
             vendor=str(gl.glGetParameter(gl.GL_VENDOR) or ""),
             renderer=str(gl.glGetParameter(gl.GL_RENDERER) or ""),
             version=str(gl.glGetParameter(gl.GL_VERSION) or ""))
        ctx.doneCurrent()
        del app
    except Exception as exc:
        emit(status="no_context", detail=f"{type(exc).__name__}: {exc}")
    return 0


def probe_opengl(executable: Optional[str] = None,
                 env: Optional[Dict[str, str]] = None) -> GLInfo:
    """Run the probe child and interpret its output."""
    child_env = dict(env if env is not None else os.environ)
    # Probe what the HARDWARE path delivers: a software override inherited from
    # the environment would make every machine look capable.
    for var in ("QT_OPENGL", "VISPY_GL_LIB"):
        child_env.pop(var, None)
    try:
        out = subprocess.run(
            [executable or sys.executable, os.path.abspath(__file__), "--gl-probe"],
            capture_output=True, text=True, timeout=_GL_PROBE_TIMEOUT_S,
            env=child_env, creationflags=_CREATE_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return GLInfo(status="timeout",
                      detail=f"no answer from the graphics driver in {_GL_PROBE_TIMEOUT_S}s")
    except Exception as exc:
        return GLInfo(status="inconclusive", detail=f"could not start probe: {exc}")

    for line in (out.stdout or "").splitlines():
        if line.startswith("GLPROBE "):
            try:
                data = json.loads(line[len("GLPROBE "):])
            except ValueError:
                break
            return GLInfo(status=str(data.get("status") or "inconclusive"),
                          vendor=str(data.get("vendor") or ""),
                          renderer=str(data.get("renderer") or ""),
                          version=str(data.get("version") or ""),
                          detail=str(data.get("detail") or ""))
    # No answer at all: the process died while talking to the driver.
    return GLInfo(status="crashed",
                  detail=f"probe exited with code {out.returncode} without an answer")


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #
def _gl_vendor(gl: GLInfo) -> str:
    text = f"{gl.vendor} {gl.renderer}".lower()
    if "nvidia" in text:
        return "nvidia"
    if "amd" in text or "ati " in text or "radeon" in text:
        return "amd"
    if "intel" in text:
        return "intel"
    return "other"


def _driver_link(adapters: List[Adapter]):
    """(url, label) for the adapter most in need of a driver."""
    for a in adapters:
        if a.vendor in _DRIVER_PAGES and a.kind in ("discrete", "integrated", "basic"):
            return _DRIVER_PAGES[a.vendor], f"Open the {a.vendor.upper() if a.vendor != 'intel' else 'Intel'} driver page"
    return _WINDOWS_UPDATE, "Open Windows Update"


def assess(adapters: List[Adapter], gl: GLInfo, remote: bool, vm: bool,
           exe_paths: List[str]) -> Report:
    """Turn the observations into a decision and a list of issues. Pure."""
    rep = Report(adapters=adapters, gl=gl, remote_session=remote,
                 virtual_machine=vm)

    if gl.status == "inconclusive":
        rep.mode, rep.reason = "unchanged", f"probe inconclusive ({gl.detail})"
        return rep

    usable = (gl.status == "ok" and not gl.is_software
              and (gl.version_tuple() or (0, 0)) >= _MIN_GL)
    real = [a for a in adapters if a.kind in ("discrete", "integrated")]

    if usable:
        rep.mode, rep.reason = "hardware", f"{gl.renderer}"
        discrete = [a for a in real if a.kind == "discrete" and a.has_driver]
        integrated = [a for a in real if a.kind == "integrated"]
        on = _gl_vendor(gl)
        wrong = (discrete and integrated
                 and on in {a.vendor for a in integrated}
                 and on not in {a.vendor for a in discrete})
        if wrong:
            d = discrete[0]
            exe = exe_paths[0] if exe_paths else "pythonw.exe"
            if d.vendor == "nvidia":
                how = ("NVIDIA Control Panel \u203a Manage 3D settings \u203a "
                       "Program settings \u203a Add, choose\n"
                       f"    {exe}\nand select \u201cHigh-performance NVIDIA "
                       "processor\u201d.")
            elif d.vendor == "amd":
                how = ("AMD Software \u203a Settings \u203a Graphics \u203a "
                       "Switchable Graphics, and set\n"
                       f"    {exe}\nto High Performance.")
            else:
                how = ("Settings \u203a System \u203a Display \u203a Graphics, "
                       f"add\n    {exe}\nand choose High performance.")
            rep.issues.append(Issue(
                id="wrong_gpu",
                title="HIBACHI is using the slower graphics chip",
                message=(f"This computer has a {d.name}, but HIBACHI's viewer is "
                         f"drawing with {gl.renderer}. Large 3D images will be "
                         "slow and use more memory.\n\n"
                         "Windows was asked to use the faster chip for HIBACHI, "
                         f"but the graphics driver overrides that. To fix it:\n{how}\n\n"
                         "Then restart HIBACHI.")))
        return rep

    # Not usable: software rendering.
    rep.mode = "software"
    if gl.status == "ok":
        what = (f"its graphics driver provides only {gl.renderer or 'an unknown renderer'}"
                f" (OpenGL {gl.version or '?'})")
    elif gl.status == "timeout":
        what = "its graphics driver stopped responding when HIBACHI asked for OpenGL"
    elif gl.status == "crashed":
        what = "its graphics driver crashed when HIBACHI asked for OpenGL"
    else:
        what = "no OpenGL context could be created"
    rep.reason = what

    missing = [a for a in adapters if not a.has_driver or a.kind == "basic"]
    if remote:
        cause = ("You are connected through Remote Desktop, which usually gives "
                 "programs no access to the graphics card.")
        fix = ("Use HIBACHI on the computer itself for full speed. An "
               "administrator can also allow graphics hardware in Remote "
               "Desktop sessions through Windows group policy.")
        link = ("", "")
    elif vm:
        cause = "HIBACHI is running in a virtual machine, which has no real graphics card."
        fix = ("Enable 3D acceleration in the virtual machine's settings, or use "
               "HIBACHI on a physical computer for full speed.")
        link = ("", "")
    elif missing:
        names = ", ".join(a.name for a in missing)
        cause = (f"The graphics adapter ({names}) has no driver from its "
                 "manufacturer installed.")
        fix = "Install the driver for your graphics card, then restart HIBACHI."
        link = _driver_link(adapters)
    else:
        oldest = sorted((a.driver_date for a in real if a.driver_date))
        age = f" (driver dated {oldest[0]})" if oldest else ""
        cause = f"This computer's graphics driver does not provide what the viewer needs{age}."
        fix = "Updating the graphics driver usually fixes this. Then restart HIBACHI."
        link = _driver_link(adapters)

    rep.issues.append(Issue(
        id="software_rendering",
        title="Using software rendering",
        message=(f"HIBACHI's viewer will draw without the graphics card, because "
                 f"{what}. Everything works and results are unaffected, but large "
                 f"images will open and move slowly.\n\n{cause}\n\n{fix}"),
        link=link[0], link_label=link[1]))
    return rep


# --------------------------------------------------------------------------- #
# Cache and entry point
# --------------------------------------------------------------------------- #
def _fingerprint(adapters: List[Adapter], exe: str, remote: bool) -> str:
    parts = sorted(f"{a.name}|{a.driver_version}|{a.has_driver}" for a in adapters)
    return json.dumps({"v": _CACHE_VERSION, "exe": os.path.normcase(exe),
                       "remote": remote, "adapters": parts}, sort_keys=True)


def _load_cache(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_cache(path: str, data: Dict[str, Any]) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except OSError:
        pass


def invalidate(state_dir: str) -> None:
    """Forget the cached decision, e.g. after the viewer crashed the driver."""
    try:
        os.remove(os.path.join(state_dir, _CACHE_FILE))
    except OSError:
        pass


def issues_to_announce(state_dir: str, report: Report) -> List[Issue]:
    """The issues not yet shown for this machine state; records them as shown."""
    if not report.issues:
        return []
    path = os.path.join(state_dir, _CACHE_FILE)
    cache = _load_cache(path)
    shown = set(cache.get("shown") or [])
    fresh = [i for i in report.issues
             if f"{cache.get('fingerprint')}::{i.id}" not in shown]
    for i in fresh:
        shown.add(f"{cache.get('fingerprint')}::{i.id}")
    if fresh:
        cache["shown"] = sorted(shown)
        _save_cache(path, cache)
    return fresh


def _report_from_dict(d: Dict[str, Any]) -> Report:
    rep = Report(**{k: v for k, v in d.items()
                    if k not in ("adapters", "gl", "issues")})
    rep.adapters = [Adapter(**a) for a in d.get("adapters") or []]
    rep.gl = GLInfo(**d["gl"]) if d.get("gl") else None
    rep.issues = [Issue(**i) for i in d.get("issues") or []]
    return rep


def prepare(state_dir: str, executable: Optional[str] = None,
            log=lambda msg: None) -> Report:
    """Probe (or reuse the cached probe) and return the decision.

    Does not change the environment; the caller applies `report.mode`.
    """
    if not sys.platform.startswith("win"):
        return Report(mode="unchanged", reason="not Windows")
    if os.environ.get("HIBACHI_GPU_PROBE") == "0":
        return Report(mode="unchanged", reason="HIBACHI_GPU_PROBE=0")
    forced = os.environ.get("HIBACHI_SOFTWARE_OPENGL")
    if forced == "1":
        return Report(mode="software", reason="HIBACHI_SOFTWARE_OPENGL=1")
    if forced == "0":
        return Report(mode="hardware", reason="HIBACHI_SOFTWARE_OPENGL=0")

    exe = os.path.abspath(executable or sys.executable)
    adapters, vm = _query_adapters()
    remote = _is_remote_session()

    # Before the probe, which then reports what the preference achieved.
    written = ensure_high_performance_gpu(adapters, interpreter_paths(exe))
    if written:
        log(f"asked Windows to use the high-performance GPU for {', '.join(written)}")

    path = os.path.join(state_dir, _CACHE_FILE)
    fp = _fingerprint(adapters, exe, remote)
    cache = _load_cache(path)
    if cache.get("fingerprint") == fp and cache.get("report") and not written:
        rep = _report_from_dict(cache["report"])
        rep.from_cache = True
        return rep

    started = time.time()
    gl = probe_opengl(exe)
    log(f"OpenGL probe: {gl.status} {gl.renderer!r} {gl.version!r} "
        f"in {time.time() - started:.1f}s {gl.detail}")
    rep = assess(adapters, gl, remote, vm, interpreter_paths(exe))
    rep.gpu_preference_set = written
    if rep.mode != "unchanged":           # inconclusive results are not cached
        _save_cache(path, {"fingerprint": fp, "report": asdict(rep),
                           "shown": [s for s in (cache.get("shown") or [])
                                     if s.startswith(fp + "::")]})
    return rep


if __name__ == "__main__" and "--gl-probe" in sys.argv[1:]:
    sys.exit(_gl_probe_child())
