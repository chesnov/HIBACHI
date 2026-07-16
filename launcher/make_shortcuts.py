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
def _make_windows(repo_root: str, home: str) -> List[str]:
    # pythonw avoids a lingering console window.
    cmd = _quote(launch_command(repo_root, windowless=True))
    bat = "@echo off\r\n" f"start \"\" {cmd}\r\n"
    written = []

    desktop_dir = os.path.join(home, "Desktop")
    for target_dir in (
        desktop_dir,
        os.path.join(
            os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming")),
            "Microsoft", "Windows", "Start Menu", "Programs",
        ),
    ):
        try:
            os.makedirs(target_dir, exist_ok=True)
            bat_path = os.path.join(target_dir, f"{APP_NAME}.bat")
            with open(bat_path, "w", newline="") as fh:
                fh.write(bat)
            written.append(bat_path)
        except Exception as exc:  # pragma: no cover
            print(f"[shortcut] could not write to {target_dir}: {exc}")

    # Best-effort: also drop a nicer .lnk on the Desktop via PowerShell.
    try:
        _make_windows_lnk(repo_root, desktop_dir)
        written.append(os.path.join(desktop_dir, f"{APP_NAME}.lnk"))
    except Exception as exc:  # pragma: no cover
        print(f"[shortcut] .lnk creation skipped: {exc}")
    return written


def _make_windows_lnk(repo_root: str, desktop_dir: str) -> None:
    import subprocess

    mgr = _env_manager_exe()
    prefix = sys.prefix
    run_app = os.path.join(repo_root, "launcher", "run_app.py")
    icon = os.path.join(repo_root, "packaging", "windows", "hibachi.ico")
    lnk = os.path.join(desktop_dir, f"{APP_NAME}.lnk")
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
    make_shortcut()
