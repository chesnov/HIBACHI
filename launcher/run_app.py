"""
run_app.py -- the double-click entry point for HIBACHI.

Order of operations on every launch:

    1. Show a small splash so the user gets immediate feedback.
    2. Self-update the git checkout (safe + offline-tolerant; see updater.py).
    3. If (and only if) the dependency spec changed, update the conda env and
       re-exec this launcher once (so the app runs under the new packages).
    4. Launch the real application (segment.py) as a subprocess and wait.

Everything is defensive: if updating fails for any reason, we still launch the
version already on disk. The heavy GUI stack (PyQt/napari) is only touched by
the child process, never here -- so a mid-update package change is safe.

Environment knobs (all optional):
    HIBACHI_BRANCH      branch to track (default: current branch, else 'main')
    HIBACHI_NO_UPDATE   set to '1' to skip the self-update (offline/dev use)
    HIBACHI_NO_SPLASH   set to '1' to disable the splash window
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

# Make sibling modules importable whether run as a script or a module.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import updater  # noqa: E402
from splash import get_splash  # noqa: E402

APP_ENTRY = "segment.py"  # relative to repo root
_REEXEC_GUARD = "HIBACHI_ENV_UPDATED"


def _find_env_manager() -> tuple[str, list[str]] | tuple[None, None]:
    """
    Locate a conda-family tool to run `env update`. Prefers micromamba, then
    mamba, then conda. Returns (exe, base_args) or (None, None).
    """
    # micromamba usually exposes itself via MAMBA_EXE.
    for candidate in (os.environ.get("MAMBA_EXE"), "micromamba", "mamba", "conda"):
        if not candidate:
            continue
        exe = candidate if os.path.isabs(candidate) else shutil.which(candidate)
        if exe:
            return exe, []
    return None, None


def _update_environment(repo_root: str, splash) -> bool:
    """Run `<mgr> env update` for the *currently active* prefix. Best-effort."""
    exe, _ = _find_env_manager()
    env_file = os.path.join(repo_root, "install", "environment.yml")
    if not exe or not os.path.isfile(env_file):
        _msg(splash, "Skipping dependency update (no environment manager found).")
        return False

    prefix = sys.prefix  # the active env we are running inside
    cmd = [exe, "env", "update", "--prefix", prefix, "--file", env_file, "--yes"]
    _msg(splash, "Updating dependencies (this may take a few minutes)...")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        _msg(splash, f"Dependency update failed ({exc.returncode}); using existing packages.")
        return False
    except Exception as exc:  # pragma: no cover - defensive
        _msg(splash, f"Dependency update error: {exc}; using existing packages.")
        return False


def _msg(splash, text: str) -> None:
    if splash is not None:
        splash.set_status(text)
    else:
        print(f"[startup] {text}")


def _refresh_shortcut(splash) -> None:
    """Regenerate the desktop launcher (best-effort) so shortcut changes apply."""
    try:
        import make_shortcuts  # sibling module in launcher/

        make_shortcuts.make_shortcut()
        _msg(splash, "Refreshed application shortcut.")
    except Exception as exc:  # never block launch on this
        _msg(splash, f"Could not refresh shortcut: {exc}")


def _launch_app(repo_root: str) -> int:
    """Launch the real application as a child process and return its exit code."""
    entry = os.path.join(repo_root, APP_ENTRY)
    if not os.path.isfile(entry):
        print(f"[startup] ERROR: cannot find {entry}")
        return 1
    # Run with the env's own interpreter, from the repo root so `import utils` works.
    return subprocess.run([sys.executable, entry], cwd=repo_root).returncode


def main() -> int:
    no_update = os.environ.get("HIBACHI_NO_UPDATE") == "1"
    no_splash = os.environ.get("HIBACHI_NO_SPLASH") == "1"
    already_reexeced = os.environ.get(_REEXEC_GUARD) == "1"

    repo_root = updater.find_repo_root(_HERE) or os.path.dirname(_HERE)
    splash = get_splash(enabled=not no_splash)

    try:
        if no_update:
            _msg(splash, "Update check skipped.")
        else:
            result = updater.check_and_update(repo_root=repo_root, logger=lambda m: _msg(splash, m))

            # Apply a dependency update once, then re-exec so the app sees it.
            if result.env_changed and not already_reexeced:
                _update_environment(repo_root, splash)
                _msg(splash, "Restarting with updated dependencies...")
                if splash is not None:
                    splash.close()
                new_env = dict(os.environ)
                new_env[_REEXEC_GUARD] = "1"
                os.execve(sys.executable, [sys.executable, __file__], new_env)
                # os.execve does not return on success.

            # When an update lands, regenerate the desktop launcher so changes to
            # the shortcut (icon, launch command, window-class) take effect. The
            # macOS .app is managed by its own bundle, so skip it there.
            if result.status == updater.UPDATED and not sys.platform.startswith("darwin"):
                _refresh_shortcut(splash)

        _msg(splash, "Launching HIBACHI...")
    finally:
        # Always tear the splash down before the Qt app grabs the screen.
        if splash is not None:
            splash.close()

    return _launch_app(repo_root)


if __name__ == "__main__":
    sys.exit(main())
