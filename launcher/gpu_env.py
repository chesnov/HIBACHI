"""
gpu_env -- choose how HIBACHI's viewer renders, before the app starts.

Windows: the full treatment described below. Linux: one job only -- on a laptop
with an NVIDIA GPU next to an integrated one, request NVIDIA render offload when
HIBACHI would otherwise run on the integrated GPU (see `_prepare_linux`). macOS:
nothing. `prepare()` is safe to call unconditionally.

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

This module is also the ONE place HIBACHI takes inventory of its GPUs.
`resource_budget` (the processing budget and the Settings tab) asks
`nvidia_gpus()` here rather than running its own query, so the launcher's
decision and the Settings tab can never describe two different machines. The
app imports this file from ``<repo>/launcher`` the way `version_manager`
imports `updater`; nothing here may import from the app, so a broken app update
can never stop the launcher from starting (or from offering a rollback).
Standard library only, for the same reason and because `resource_budget` is
imported inside processing workers.

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
#: 3: Intel Arc 1xxT/1xxV/"Arc Graphics" are integrated (earlier verdicts
#:    treated them as discrete and must not be reused).
_CACHE_VERSION = 3
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
    vram_gb: float = 0.0   # dedicated memory where known (NVIDIA), else 0


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
    #: Environment variables the app should be started with (Linux offload).
    env: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        gl = self.gl
        where = (f"{gl.renderer} (OpenGL {gl.version})" if gl and gl.renderer
                 else (gl.status if gl else "not probed"))
        return f"rendering={self.mode}; {where}; reason: {self.reason or '-'}"


# --------------------------------------------------------------------------- #
# NVIDIA inventory (every platform) -- the one nvidia-smi query in HIBACHI
# --------------------------------------------------------------------------- #
_NVIDIA_CACHE: Optional[List[Adapter]] = None


def parse_nvidia_smi(text: str) -> List[Adapter]:
    """`name, memory.total (MiB), driver_version` lines as Adapters. Pure."""
    found: List[Adapter] = []
    for line in (text or "").strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            mib = float(parts[1])
        except ValueError:
            mib = 0.0
        # MiB to GB (1024^3), so a 24576 MiB card reads as 24.0, not 25.8.
        found.append(Adapter(name=parts[0], vendor="nvidia", kind="discrete",
                             driver_version=parts[2] if len(parts) > 2 else "",
                             vram_gb=mib / 1024.0))
    return found


def nvidia_gpus(refresh: bool = False) -> List[Adapter]:
    """NVIDIA GPUs with a working driver, with their VRAM. [] if none.

    Asks the driver's own `nvidia-smi`, which ships with it on Windows and
    Linux alike, rather than a CUDA library: there is none in the environment,
    and adding one to answer a question the driver already answers would be a
    dependency for nothing. A missing binary, a driver error, a timeout or
    unparseable output all mean "no NVIDIA GPU"; this is never allowed to be
    the reason anything fails to start. Cached per process, because it spawns
    one.
    """
    global _NVIDIA_CACHE
    if _NVIDIA_CACHE is not None and not refresh:
        return list(_NVIDIA_CACHE)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5.0, check=False,
            creationflags=_CREATE_NO_WINDOW)
        found = parse_nvidia_smi(out.stdout) if out.returncode == 0 else []
    except (OSError, subprocess.SubprocessError):
        found = []
    _NVIDIA_CACHE = found
    return list(found)


# --------------------------------------------------------------------------- #
# Adapters (Windows)
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
        # Only the Arc A- and B-series CARDS are discrete (A770, A370M, B580,
        # Pro A60M, ...). "Arc 140T", "Arc 130V", "Arc Graphics" are the
        # graphics built into Core Ultra processors: integrated. Treating every
        # "Arc" as discrete hid the second GPU on such laptops, so no
        # preference was set and no notice shown while the viewer sat on the
        # integrated chip.
        discrete_arc = re.search(
            r"\barc\b\W*(?:\(tm\))?\s*(?:pro\s*)?[ab]\d{2,3}m?\b", low)
        kind = "discrete" if discrete_arc else "integrated"
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
        adapters, vm = parse_wmi_output(out.stdout)
    except Exception:
        return [], False
    # WMI caps reported memory at 4 GB, so NVIDIA VRAM comes from the driver.
    by_name = {a.name.lower(): a for a in nvidia_gpus()}
    for a in adapters:
        if a.vendor == "nvidia":
            match = by_name.get(a.name.lower()) or next(
                (n for key, n in by_name.items()
                 if key in a.name.lower() or a.name.lower() in key), None)
            if match is not None:
                a.vram_gb = match.vram_gb
    return adapters, vm


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
        on = _gl_vendor(gl)
        # Evidence, not the adapter list: a discrete GPU exists, and OpenGL is
        # being served by a DIFFERENT vendor's chip. That is only possible on
        # a machine with a second GPU, whether or not the adapter query
        # managed to list it (PowerShell can be blocked or slow).
        wrong = bool(discrete and on != "other"
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
                         "HIBACHI asks Windows to use the faster chip, but a "
                         "setting in the graphics driver (or one made earlier in "
                         f"Windows) takes precedence. To fix it:\n{how}\n\n"
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


# --------------------------------------------------------------------------- #
# Linux: NVIDIA render offload on hybrid laptops
# --------------------------------------------------------------------------- #
#: What `prime-run` sets. With the proprietary driver in on-demand mode these
#: route a program's OpenGL to the NVIDIA GPU; without them it stays on the
#: integrated one, by design, to save power.
NV_OFFLOAD_ENV = {"__NV_PRIME_RENDER_OFFLOAD": "1",
                  "__GLX_VENDOR_LIBRARY_NAME": "nvidia"}

#: Any of these set means the user (or the desktop, e.g. GNOME's "Launch using
#: Discrete Graphics Card") already chose a GPU. Never overridden.
_LINUX_GPU_CHOICE_VARS = ("__NV_PRIME_RENDER_OFFLOAD", "__GLX_VENDOR_LIBRARY_NAME",
                          "DRI_PRIME", "__VK_LAYER_NV_optimus")


def _drm_devices_linux() -> List[str]:
    """`vendor:device` for each GPU the kernel knows, for the cache key."""
    found = []
    base = "/sys/class/drm"
    try:
        for name in sorted(os.listdir(base)):
            if not re.fullmatch(r"card\d+", name):
                continue
            ids = []
            for part in ("vendor", "device"):
                try:
                    with open(os.path.join(base, name, "device", part)) as fh:
                        ids.append(fh.read().strip())
                except OSError:
                    ids.append("?")
            found.append(":".join(ids))
    except OSError:
        pass
    return found


def _prepare_linux(state_dir: str, exe: str, log) -> Report:
    """Request NVIDIA render offload when that is what makes the viewer use it.

    Three probes' worth of evidence, each cheap next to guessing wrong:
    a working NVIDIA driver (nvidia-smi answers), the default renderer NOT
    being NVIDIA already, and a second probe confirming the offload variables
    really deliver the NVIDIA renderer. Anything else leaves the launch as it
    was. Mesa already falls back to software rendering by itself on Linux, so
    there is no software decision to make here.
    """
    already = [v for v in _LINUX_GPU_CHOICE_VARS if v in os.environ]
    if already:
        return Report(mode="unchanged",
                      reason=f"GPU already chosen in the environment ({', '.join(already)})")
    nvidia = nvidia_gpus()
    if not nvidia:
        return Report(mode="unchanged", reason="no working NVIDIA driver")

    path = os.path.join(state_dir, _CACHE_FILE)
    fp = json.dumps({"v": _CACHE_VERSION, "os": "linux", "exe": exe,
                     "nvidia": sorted(f"{a.name}|{a.driver_version}" for a in nvidia),
                     "drm": _drm_devices_linux()},
                    sort_keys=True)
    cache = _load_cache(path)
    if cache.get("fingerprint") == fp and cache.get("report"):
        rep = _report_from_dict(cache["report"])
        rep.from_cache = True
        return rep

    default = probe_opengl(exe)
    log(f"OpenGL probe (default): {default.status} {default.renderer!r}")
    rep_adapters = list(nvidia)
    if default.status == "ok" and _gl_vendor(default) == "nvidia":
        rep = Report(mode="hardware", gl=default,
                     reason=f"already on {default.renderer}")
    else:
        env = dict(os.environ)
        env.update(NV_OFFLOAD_ENV)
        offload = probe_opengl(exe, env=env)
        log(f"OpenGL probe (NVIDIA offload): {offload.status} {offload.renderer!r}")
        if (offload.status == "ok" and _gl_vendor(offload) == "nvidia"
                and not offload.is_software):
            rep = Report(mode="hardware", gl=offload, env=dict(NV_OFFLOAD_ENV),
                         reason=f"NVIDIA render offload: {offload.renderer}")
        else:
            rep = Report(mode="unchanged", gl=default,
                         reason=f"NVIDIA offload did not take effect "
                                f"({offload.status} {offload.renderer!r})")
            if offload.status == "inconclusive" or default.status == "inconclusive":
                return rep                  # not cached: nothing was learned
    rep.adapters = rep_adapters
    _save_cache(path, {"fingerprint": fp, "report": asdict(rep), "shown": []})
    return rep


def last_report(state_dir: str) -> Optional[Report]:
    """The launcher's most recent recorded decision, or None if there is none."""
    data = _load_cache(os.path.join(state_dir, _CACHE_FILE)).get("report")
    if not data:
        return None
    try:
        return _report_from_dict(data)
    except Exception:
        return None


def short_renderer(renderer: str) -> str:
    """The card's name from an OpenGL renderer string, without driver detail.

    "NVIDIA GeForce RTX 4060 Laptop GPU/PCIe/SSE2" -> "NVIDIA GeForce RTX 4060
    Laptop GPU"; "AMD Radeon 890M Graphics (radeonsi, strix1, ACO, ...)" ->
    "AMD Radeon 890M Graphics". The full string stays in the launcher log.
    """
    text = (renderer or "").strip()
    if "/" in text and "nvidia" in text.lower():
        text = text.split("/", 1)[0]
    text = re.sub(r"\s*\([^()]*\)\s*$", "", text)
    return text.strip() or (renderer or "").strip()


def rendering_summary(state_dir: str) -> str:
    """What the viewer draws with, in one line.

    Prefers what an open viewer measured in its own OpenGL context; otherwise
    the launcher's decision, read, never probed: the launcher is the only place that asks the graphics
    driver. Which record applies is decided by the environment the launcher
    gives the app (HIBACHI_GL_MODE / HIBACHI_GL_REASON), because that describes
    THIS launch:

    * absent -> the app was not started by the launcher; "" (unknown). The
      cache is not used then, since nothing ties it to this process.
    * "unchanged" -> the launcher decided to change nothing, and says why
      (no NVIDIA driver, a GPU already chosen in the environment, probe
      skipped or inconclusive). The renderer was not asked for, so it is not
      claimed.
    * "hardware" / "software" -> the renderer recorded for this launch.
    """
    # Ground truth first: the renderer an open viewer's own context reported
    # (recorded by app_launch). Driver profiles can move the viewer to a
    # different GPU from the one the launcher's pre-start probe saw.
    live = os.environ.get("HIBACHI_VIEWER_RENDERER")
    if live:
        return f"{short_renderer(live)} (measured in the viewer)"
    mode = os.environ.get("HIBACHI_GL_MODE")
    reason = os.environ.get("HIBACHI_GL_REASON") or ""
    if not mode:
        return ""
    if mode == "unchanged":
        return f"system default \u2014 {reason}" if reason else "system default"
    rep = last_report(state_dir)
    if rep is None or rep.gl is None or not rep.gl.renderer or rep.mode != mode:
        if mode == "software":
            return "software rendering (no graphics-card acceleration)"
        return f"hardware rendering \u2014 {reason}" if reason else "hardware rendering"
    how = ""
    if mode == "software":
        how = " \u2014 software rendering, no graphics-card acceleration"
    elif rep.env.get("__NV_PRIME_RENDER_OFFLOAD") == "1":
        how = " (NVIDIA render offload)"
    return f"{short_renderer(rep.gl.renderer)}{how}"


def prepare(state_dir: str, executable: Optional[str] = None,
            log=lambda msg: None) -> Report:
    """Probe (or reuse the cached probe) and return the decision.

    Does not change the environment; the caller applies `report.mode` and
    `report.env`.
    """
    if os.environ.get("HIBACHI_GPU_PROBE") == "0":
        return Report(mode="unchanged", reason="HIBACHI_GPU_PROBE=0")
    forced = os.environ.get("HIBACHI_SOFTWARE_OPENGL")
    if forced == "1":
        return Report(mode="software", reason="HIBACHI_SOFTWARE_OPENGL=1")
    if sys.platform.startswith("linux"):
        return _prepare_linux(state_dir, os.path.abspath(executable or sys.executable), log)
    if sys.platform == "darwin":
        # One GPU on Apple Silicon; on dual-GPU Intel Macs macOS itself moves
        # OpenGL programs to the discrete GPU (the app bundle's Info.plist does
        # not opt out). OpenGL always exists, so there is no fallback to make.
        return Report(mode="unchanged",
                      reason="macOS chooses the graphics processor automatically")
    if not sys.platform.startswith("win"):
        return Report(mode="unchanged", reason="not Windows or Linux")
    if forced == "0":
        return Report(mode="hardware", reason="HIBACHI_SOFTWARE_OPENGL=0")

    exe = os.path.abspath(executable or sys.executable)
    adapters, vm = _query_adapters()
    remote = _is_remote_session()
    if not adapters:
        log("adapter query returned nothing (PowerShell blocked or slow?); "
            "using the NVIDIA driver's own inventory")
    # The adapter list must not be the only way to know a discrete GPU exists:
    # the NVIDIA driver answers for itself, and the Settings tab already
    # trusts that answer.
    known = {a.name.lower() for a in adapters}
    adapters = adapters + [a for a in nvidia_gpus() if a.name.lower() not in known]

    # Before the probe, which then reports what the preference achieved.
    written = ensure_high_performance_gpu(adapters, interpreter_paths(exe))
    if written:
        log(f"asked Windows to use the high-performance GPU for {', '.join(written)}")

    path = os.path.join(state_dir, _CACHE_FILE)
    fp = _fingerprint(adapters, exe, remote)
    cache = _load_cache(path)
    # A verdict of "on the slower chip" is never reused. What fixes it -- a
    # program setting in NVIDIA Control Panel or AMD Software -- changes none
    # of what the fingerprint sees, so a cached copy would keep reporting the
    # old GPU after the user had already followed the notice's instructions.
    # Re-probing costs a few seconds per start, only while the problem lasts.
    cached = cache.get("report") if cache.get("fingerprint") == fp else None
    stale = bool(cached) and any(
        i.get("id") == "wrong_gpu" for i in (cached.get("issues") or []))
    if cached and not written and not stale:
        rep = _report_from_dict(cached)
        rep.from_cache = True
        return rep

    started = time.time()
    gl = probe_opengl(exe)
    log(f"OpenGL probe: {gl.status} {gl.renderer!r} {gl.version!r} "
        f"in {time.time() - started:.1f}s {gl.detail}")

    # A discrete GPU exists but a different vendor's chip answered: this is a
    # two-GPU machine even if the adapter query could not show it. Ask for the
    # high-performance GPU now, and probe again to see whether it took.
    discrete = [a for a in adapters if a.kind == "discrete" and a.has_driver]
    on = _gl_vendor(gl)
    if (not written and gl.status == "ok" and discrete and on != "other"
            and on not in {a.vendor for a in discrete}):
        seen = adapters + [Adapter(name=short_renderer(gl.renderer) or "integrated GPU",
                                   vendor=on, kind="integrated")]
        written = ensure_high_performance_gpu(seen, interpreter_paths(exe))
        if written:
            log(f"OpenGL is on {gl.renderer!r} although {discrete[0].name} is "
                f"present; asked Windows to use the high-performance GPU for "
                f"{', '.join(written)}")
            gl = probe_opengl(exe)
            log(f"OpenGL probe after the preference: {gl.status} {gl.renderer!r}")
    rep = assess(adapters, gl, remote, vm, interpreter_paths(exe))
    rep.gpu_preference_set = written
    if rep.mode != "unchanged":           # inconclusive results are not cached
        _save_cache(path, {"fingerprint": fp, "report": asdict(rep),
                           "shown": [s for s in (cache.get("shown") or [])
                                     if s.startswith(fp + "::")]})
    return rep


if __name__ == "__main__" and "--gl-probe" in sys.argv[1:]:
    sys.exit(_gl_probe_child())