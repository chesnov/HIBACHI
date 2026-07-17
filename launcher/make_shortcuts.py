"""
make_shortcuts.py -- create a double-click launcher for the current OS.

Run once by the installer (inside the conda env), e.g.:
    micromamba run -n hibachi python <repo>/launcher/make_shortcuts.py

It figures out the command that launches run_app.py inside this environment and
writes the appropriate native shortcut:

    Linux    ~/.local/share/applications/hibachi.desktop  (+ a copy on the Desktop)
    macOS    ~/Applications/HIBACHI.app
    Windows  %USERPROFILE%\\Desktop\\HIBACHI.bat  (+ Start Menu; + a .lnk if possible)

The `platform` argument is injectable so the logic can be unit-tested for all
three targets from any host.

IMPORTANT: the shortcut embeds `sys.prefix` (the prefix of whatever Python runs
this script) into the launch command. If you run this from the wrong env (e.g.
'base'), the shortcut will launch the app in that env, show the splash, and then
die when it fails to import napari/PyQt5. To catch that early, make_shortcut()
verifies the current interpreter can import the GUI stack before writing
anything (skippable with HIBACHI_SKIP_ENV_CHECK=1, and skipped when `platform`
is passed explicitly for tests / cross-platform generation).
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
from typing import List, Optional

APP_NAME = "HIBACHI"


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)  # launcher/ -> repo root


def _current_env_can_run_app() -> bool:
    """
    True if the interpreter generating this shortcut can import the app's GUI
    stack. The shortcut launches run_app.py with *this same* interpreter
    (sys.prefix is baked into the Exec line), so if these are missing the app
    would start the splash and then fail on import -- exactly the "quick splash,
    then nothing" failure. Uses find_spec so we don't pay to import the heavy
    stack just to check.
    """
    import importlib.util

    # Require the WHOLE stack: PyQt5 alone shows up in plenty of envs (incl.
    # base), so `any` would wave those through even though napari is missing.
    return all(
        importlib.util.find_spec(mod) is not None
        for mod in ("napari", "PyQt5")
    )


def _env_manager_exe() -> str:
    """Absolute path to micromamba/mamba/conda, for embedding in the shortcut."""
    for cand in (os.environ.get("MAMBA_EXE"), "micromamba", "mamba", "conda"):
        if not cand:
            continue
        exe = cand if os.path.isabs(cand) else shutil.which(cand)
        if exe:
            return exe
    return "micromamba"  # last-resort literal; user PATH may still resolve it


def launch_command(repo_root: str, windowless: bool = False) -> List[str]:
    """
    Build the command that runs the launcher inside this environment.

    Uses `<mgr> run -p <prefix> python run_app.py` so it works no matter how the
    env was created (named or prefix-based). On Windows we can swap in pythonw
    to avoid a console window.
    """
    mgr = _env_manager_exe()
    prefix = sys.prefix
    py = "pythonw" if windowless else "python"
    run_app = os.path.join(repo_root, "launcher", "run_app.py")
    return [mgr, "run", "--prefix", prefix, py, run_app]


def _quote(parts: List[str]) -> str:
    out = []
    for p in parts:
        out.append(f'"{p}"' if (" " in p or "\\" in p) else p)
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Linux
# --------------------------------------------------------------------------- #
def _make_linux(repo_root: str, home: str) -> List[str]:
    exec_cmd = _quote(launch_command(repo_root))
    icon = os.path.join(repo_root, "launcher", "assets", "hibachi.png")
    desktop = (
        "[Desktop Entry]\n"
        "Type=Application\n"
        f"Name={APP_NAME}\n"
        "Comment=Heuristic image segmentation for microscopy\n"
        f"Exec={exec_cmd}\n"
        f"Icon={icon}\n"
        "Terminal=false\n"
        "StartupWMClass=HIBACHI\n"
        "Categories=Science;Biology;ImageProcessing;\n"
    )
    written = []
    apps_dir = os.path.join(home, ".local", "share", "applications")
    os.makedirs(apps_dir, exist_ok=True)
    app_path = os.path.join(apps_dir, "hibachi.desktop")
    with open(app_path, "w") as fh:
        fh.write(desktop)
    os.chmod(app_path, 0o755)
    written.append(app_path)

    desktop_dir = os.path.join(home, "Desktop")
    if os.path.isdir(desktop_dir):
        dt_path = os.path.join(desktop_dir, "hibachi.desktop")
        with open(dt_path, "w") as fh:
            fh.write(desktop)
        os.chmod(dt_path, 0o755)
        written.append(dt_path)
    return written


# --------------------------------------------------------------------------- #
# macOS
# --------------------------------------------------------------------------- #
def _make_macos(repo_root: str, home: str) -> List[str]:
    app_dir = os.path.join(home, "Applications", f"{APP_NAME}.app")
    macos_dir = os.path.join(app_dir, "Contents", "MacOS")
    os.makedirs(macos_dir, exist_ok=True)

    info_plist = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n<dict>\n'
        "  <key>CFBundleName</key>\n"
        f"  <string>{APP_NAME}</string>\n"
        "  <key>CFBundleExecutable</key>\n"
        f"  <string>{APP_NAME}</string>\n"
        "  <key>CFBundleIdentifier</key>\n"
        f"  <string>org.hibachi.{APP_NAME.lower()}</string>\n"
        "  <key>CFBundlePackageType</key>\n"
        "  <string>APPL</string>\n"
        "  <key>CFBundleIconFile</key>\n"
        "  <string>hibachi.icns</string>\n"
        "  <key>LSMinimumSystemVersion</key>\n"
        "  <string>10.13</string>\n"
        "</dict>\n</plist>\n"
    )
    with open(os.path.join(app_dir, "Contents", "Info.plist"), "w") as fh:
        fh.write(info_plist)

    # Copy the app icon into the bundle if it exists.
    icns_src = os.path.join(repo_root, "packaging", "macos", "hibachi.icns")
    res_dir = os.path.join(app_dir, "Contents", "Resources")
    if os.path.isfile(icns_src):
        os.makedirs(res_dir, exist_ok=True)
        shutil.copy(icns_src, os.path.join(res_dir, "hibachi.icns"))

    launcher_script = os.path.join(macos_dir, APP_NAME)
    script = "#!/bin/bash\n" f"exec {_quote(launch_command(repo_root))}\n"
    with open(launcher_script, "w") as fh:
        fh.write(script)
    st = os.stat(launcher_script)
    os.chmod(launcher_script, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return [app_dir]


# --------------------------------------------------------------------------- #
# Windows
# --------------------------------------------------------------------------- #
def _cleanup_windows_shortcuts(dirs: List[str]) -> None:
    """
    Remove HIBACHI shortcuts left by earlier installs.

    Historically we wrote BOTH a HIBACHI.bat (generic/no-logo icon) and a
    HIBACHI.lnk (logo) to the Desktop, so users saw two icons -- and a stale
    .lnk from an older build could launch the wrong interpreter and fail with
    "MSVCP140.dll is missing". We now write a single shortcut, so first delete
    any of the old ones (both extensions, Desktop + Start Menu) before writing.
    """
    for d in dirs:
        for name in (f"{APP_NAME}.bat", f"{APP_NAME}.lnk"):
            p = os.path.join(d, name)
            try:
                if os.path.exists(p):
                    os.remove(p)
                    print(f"[shortcut] removed stale shortcut: {p}")
            except OSError as exc:
                print(f"[shortcut] could not remove {p}: {exc}")


def _make_windows(repo_root: str, home: str) -> List[str]:
    written = []
    desktop_dir = os.path.join(home, "Desktop")
    start_menu = os.path.join(
        os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming")),
        "Microsoft", "Windows", "Start Menu", "Programs",
    )

    # Clear out anything a previous install left behind so we don't end up with
    # a broken logo .lnk sitting next to a working .bat (the "two icons" bug).
    _cleanup_windows_shortcuts([desktop_dir, start_menu])

    # Preferred launcher: ONE .lnk with the real icon, launched through the env
    # manager (`micromamba run --prefix <prefix> pythonw ...`) so the app runs
    # under the env's own pythonw.exe -- the only place the conda-provided MSVC
    # runtime (MSVCP140.dll, needed by the pip Qt wheels) is on the DLL search
    # path. A .lnk that launches Python any other way is what produces the
    # "MSVCP140.dll is missing" error.
    lnk_ok = False
    try:
        os.makedirs(desktop_dir, exist_ok=True)
        _make_windows_lnk(repo_root, desktop_dir)
        written.append(os.path.join(desktop_dir, f"{APP_NAME}.lnk"))
        lnk_ok = True
    except Exception as exc:  # pragma: no cover
        print(f"[shortcut] .lnk creation failed, falling back to .bat: {exc}")

    # Fallback ONLY if the .lnk could not be created (e.g. PowerShell blocked):
    # a .bat that still routes through the env manager.
    if not lnk_ok:
        cmd = _quote(launch_command(repo_root, windowless=True))
        bat = "@echo off\r\n" f"start \"\" {cmd}\r\n"
        try:
            os.makedirs(desktop_dir, exist_ok=True)
            bat_path = os.path.join(desktop_dir, f"{APP_NAME}.bat")
            with open(bat_path, "w", newline="") as fh:
                fh.write(bat)
            written.append(bat_path)
        except Exception as exc:  # pragma: no cover
            print(f"[shortcut] could not write to {desktop_dir}: {exc}")

    # Start Menu entry (one .lnk; not on the Desktop, so it doesn't double the icon).
    try:
        os.makedirs(start_menu, exist_ok=True)
        _make_windows_lnk(repo_root, start_menu)
        written.append(os.path.join(start_menu, f"{APP_NAME}.lnk"))
    except Exception as exc:  # pragma: no cover
        print(f"[shortcut] Start Menu .lnk skipped: {exc}")

    return written


def _make_windows_lnk(repo_root: str, dest_dir: str) -> None:
    import subprocess

    mgr = _env_manager_exe()
    prefix = sys.prefix
    run_app = os.path.join(repo_root, "launcher", "run_app.py")
    icon = os.path.join(repo_root, "packaging", "windows", "hibachi.ico")
    lnk = os.path.join(dest_dir, f"{APP_NAME}.lnk")
    args = f'run --prefix "{prefix}" pythonw "{run_app}"'
    icon_line = f"$S.IconLocation = '{icon}'; " if os.path.isfile(icon) else ""
    ps = (
        "$W = New-Object -ComObject WScript.Shell; "
        f"$S = $W.CreateShortcut('{lnk}'); "
        f"$S.TargetPath = '{mgr}'; "
        f"$S.Arguments = '{args}'; "
        f"$S.WorkingDirectory = '{repo_root}'; "
        f"{icon_line}"
        "$S.Save()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def make_shortcut(platform: Optional[str] = None, home: Optional[str] = None) -> List[str]:
    plat = (platform or sys.platform).lower()
    home = home or os.path.expanduser("~")
    repo_root = _repo_root()

    # Guard: the shortcut we write launches run_app.py under *this* interpreter
    # (sys.prefix is baked into the launch command). If we're in the wrong env
    # (e.g. 'base'), the app would show the splash and then fail to import its
    # GUI stack. Fail loudly with instructions instead of writing a dead
    # shortcut. `platform` is only passed explicitly by tests / cross-platform
    # generation, so skip the check in that case (and via an env override).
    if platform is None and os.environ.get("HIBACHI_SKIP_ENV_CHECK") != "1":
        if not _current_env_can_run_app():
            raise RuntimeError(
                f"Running under a Python without HIBACHI's deps (sys.prefix={sys.prefix!r}).\n"
                "The shortcut would launch the app in this same env and fail right "
                "after the splash.\n"
                "Re-run inside the app environment, e.g.:\n"
                "    micromamba run -n hibachi python launcher/make_shortcuts.py\n"
                "(set HIBACHI_SKIP_ENV_CHECK=1 to bypass this check)"
            )

    if plat.startswith("linux"):
        written = _make_linux(repo_root, home)
    elif plat == "darwin" or plat.startswith("mac"):
        written = _make_macos(repo_root, home)
    elif plat.startswith("win"):
        written = _make_windows(repo_root, home)
    else:
        raise RuntimeError(f"Unsupported platform: {plat}")

    for path in written:
        print(f"[shortcut] created: {path}")
    return written


if __name__ == "__main__":
    try:
        make_shortcut()
    except RuntimeError as exc:
        print(f"[shortcut] {exc}", file=sys.stderr)
        sys.exit(1)