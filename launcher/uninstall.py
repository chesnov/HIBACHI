"""Complete removal of a HIBACHI installation.

Reached from the version menu (dialogs.choose_rollback -> "Uninstall..."), so the
same entry point that rolls back also removes.

Deletion runs in a DETACHED SHELL SCRIPT, not here: the targets include the
interpreter executing this module (<install>/micromamba/envs/hibachi/bin/python)
and this file itself. Windows cannot delete a running executable at all, and on
POSIX a lazily-imported module can vanish mid-delete. So we write a small script,
spawn it, and exit -- it waits for this PID to disappear, then deletes.

Nothing is backed up and nothing is recoverable; a reinstall starts from scratch.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from typing import List

_APP_BUNDLE = "/Applications/HIBACHI.app"


def _state_dirs() -> List[str]:
    """Log/state dirs, honouring the same env vars logging_setup/updater use."""
    out = []
    for var, default in (
        ("HIBACHI_STATE_DIR", os.path.join(os.path.expanduser("~"), ".hibachi")),
        ("HIBACHI_LOG_DIR", None),
    ):
        val = os.environ.get(var) or default
        if val:
            out.append(os.path.abspath(val))
    return out


def targets(repo_root: str) -> List[str]:
    """Every path an uninstall should remove, for this platform.

    install_dir comes from the running checkout (parent of repo_root), not from
    $HIBACHI_HOME, so a non-default install removes its own files rather than the
    default location.
    """
    install_dir = os.path.abspath(os.path.dirname(repo_root))
    home = os.path.expanduser("~")
    paths = [install_dir] + _state_dirs()

    if sys.platform == "darwin":
        paths += [_APP_BUNDLE, os.path.join(home, "Applications", "HIBACHI.app")]
    elif sys.platform.startswith("win"):
        appdata = os.environ.get("APPDATA", os.path.join(home, "AppData", "Roaming"))
        programs = os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs")
        for folder in (os.path.join(home, "Desktop"), programs):
            paths += [os.path.join(folder, "HIBACHI.lnk"),
                      os.path.join(folder, "HIBACHI.bat")]
    else:
        paths += [
            os.path.join(home, ".local", "share", "applications", "hibachi.desktop"),
            os.path.join(home, "Desktop", "hibachi.desktop"),
        ]

    # Deduplicate, keep order, drop anything that isn't actually there.
    seen, out = set(), []
    for p in paths:
        p = os.path.abspath(p)
        if p not in seen and os.path.exists(p) and _is_safe(p):
            seen.add(p)
            out.append(p)
    return out


def _looks_like_install(path: str) -> bool:
    """True if *path* is a HIBACHI install root, judged by contents not by name.

    A custom $HIBACHI_HOME need not have 'hibachi' anywhere in its name, so
    requiring that would silently skip the main tree and leave a half-removed
    install. Both markers must be present.
    """
    return (os.path.isdir(os.path.join(path, "app", ".git"))
            and os.path.isdir(os.path.join(path, "micromamba")))


def _is_safe(path: str) -> bool:
    """Guard against deleting a root, a home directory, or anything unrelated.

    Every target must sit below the user's home (or be the known /Applications
    bundle) AND either be named for HIBACHI or verifiably be an install root.
    Without this a surprising repo_root -- a checkout at ~, say -- could expand
    to $HOME.
    """
    path = os.path.abspath(path)
    home = os.path.abspath(os.path.expanduser("~"))
    if path in ("/", home) or len(path.strip("/\\")) < 4:
        return False
    if not (path.startswith(home + os.sep) or path == _APP_BUNDLE):
        return False
    return "hibachi" in os.path.basename(path).lower() or _looks_like_install(path)


def _posix_script(paths: List[str], pid: int) -> str:
    quoted = " ".join(shlex.quote(p) for p in paths)
    notify = ""
    if sys.platform == "darwin":
        notify = (
            "osascript -e 'display notification \"HIBACHI has been removed.\" "
            "with title \"HIBACHI\"' >/dev/null 2>&1 || true\n"
        )
    return (
        "#!/bin/sh\n"
        "# Wait for the app to exit before deleting the files it is running from.\n"
        f"while kill -0 {pid} 2>/dev/null; do sleep 0.5; done\n"
        f"rm -rf {quoted}\n"
        + notify +
        'rm -f "$0"\n'
    )


def _windows_script(paths: List[str], pid: int) -> str:
    lines = [
        "@echo off",
        "rem Wait for the app to exit before deleting the files it is running from.",
        ":wait",
        f'tasklist /FI "PID eq {pid}" 2>nul | find "{pid}" >nul',
        'if not errorlevel 1 (timeout /t 1 /nobreak >nul & goto wait)',
    ]
    # Decide dir-vs-file at run time rather than trusting the filesystem state
    # when this script was generated.
    for p in paths:
        lines.append(f'if exist "{p}\\" (rmdir /s /q "{p}") else (del /q /f "{p}")')
    lines.append('(goto) 2>nul & del "%~f0"')   # self-delete
    return "\r\n".join(lines) + "\r\n"


def spawn(paths: List[str]) -> str:
    """Write the deletion script and start it detached. Returns its path."""
    import tempfile

    pid = os.getpid()
    tmp = tempfile.gettempdir()
    if sys.platform.startswith("win"):
        script = os.path.join(tmp, "hibachi_uninstall.cmd")
        body = _windows_script(paths, pid)
    else:
        script = os.path.join(tmp, "hibachi_uninstall.sh")
        body = _posix_script(paths, pid)

    with open(script, "w", encoding="utf-8", newline="") as fh:
        fh.write(body)
    os.chmod(script, 0o755)

    devnull = subprocess.DEVNULL
    if sys.platform.startswith("win"):
        DETACHED_PROCESS = 0x00000008
        CREATE_NO_WINDOW = 0x08000000
        subprocess.Popen(["cmd.exe", "/c", script], stdout=devnull, stderr=devnull,
                         stdin=devnull, close_fds=True,
                         creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW)
    else:
        subprocess.Popen(["/bin/sh", script], stdout=devnull, stderr=devnull,
                         stdin=devnull, close_fds=True, start_new_session=True)
    return script


def run(repo_root: str) -> bool:
    """Confirm, then schedule the uninstall. True if the user went ahead."""
    import dialogs

    paths = targets(repo_root)
    if not paths:
        dialogs.notify("HIBACHI", "Nothing to uninstall: no HIBACHI files were found.")
        return False

    if not dialogs.confirm_uninstall(paths):
        return False

    try:
        spawn(paths)
    except Exception as exc:  # noqa: BLE001 - report anything and stay put
        dialogs.notify("HIBACHI", f"Uninstall could not start:\n{exc}")
        return False

    dialogs.notify(
        "HIBACHI",
        "HIBACHI will finish removing itself once this window closes.\n\n"
        "You can close this dialog now.",
    )
    return True
