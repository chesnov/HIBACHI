"""
run_app.py -- the double-click entry point for HIBACHI.

Order of operations on every launch:

    1. Show a small splash so the user gets immediate feedback.
    2. Check for a newer version (safe + offline-tolerant; see updater.py).
    3. If one is available, ASK the user whether to install it. Install only on
       consent. If the dependency spec changed, update the conda env and
       re-exec this launcher once (so the app runs under the new packages).
    4. Refresh the desktop launcher so a changed icon / launch command / moved
       checkout self-heals (best-effort; skipped on macOS).
    5. Launch the real application (segment.py) as a subprocess and wait.
    6. If it exits with an error, offer to roll back to a previous version.

Everything is defensive: if updating fails for any reason, we still launch the
version already on disk. The heavy GUI stack (PyQt/napari) is only touched by
the child process, never here -- so a mid-update package change is safe.

Environment knobs (all optional):
    HIBACHI_BRANCH        branch to track (default: current branch, else 'main')
    HIBACHI_NO_UPDATE     '1' to skip the update check entirely (offline/dev use)
    HIBACHI_AUTO_UPDATE   '1' to install updates without asking (headless/kiosk)
    HIBACHI_NO_SPLASH     '1' to disable the splash window
    HIBACHI_ROLLBACK      '1' to open the rollback chooser instead of launching
    HIBACHI_STATE_DIR     where to keep launcher state (default: ~/.hibachi)

Command line:
    --rollback            open the rollback chooser instead of launching
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


def _ask_update(check) -> str:
    """Prompt the user about an available update. Assumes no splash is open."""
    import dialogs

    current = (check.old_rev or "")[:8]
    latest = (check.new_rev or "")[:8]
    return dialogs.ask_update(current, latest, list(check.changelog or []), bool(check.env_changed))


def _run_rollback(repo_root: str) -> None:
    """Let the user pick a previous version and switch the checkout to it."""
    import dialogs

    versions = updater.list_versions(repo_root, limit=15)
    if not versions:
        dialogs.notify("HIBACHI", "No version history is available to roll back to.")
        return

    current = updater.current_rev(repo_root) or ""
    chosen = dialogs.choose_rollback(versions, current)
    if not chosen:
        return

    ok, msg = updater.rollback_to(repo_root, chosen, logger=lambda m: print(f"[startup] {m}"))
    if ok:
        # Don't immediately re-prompt to update back up to the tip: remember the
        # current remote tip as "skipped" so the next launch stays put unless
        # the user opts in.
        tip = updater.remote_tip(repo_root)
        if tip:
            updater.set_skipped_rev(tip)
        dialogs.notify("HIBACHI", f"{msg}\n\nPlease start HIBACHI again to use this version.")
    else:
        dialogs.notify("HIBACHI", f"Rollback failed:\n{msg}")


def _offer_rollback_after_crash(repo_root: str, code: int) -> None:
    """After a failed launch, offer to roll back to a previous version."""
    import dialogs

    versions = updater.list_versions(repo_root, limit=15)
    if len(versions) < 2:
        return  # nothing earlier to go to
    if not dialogs.ask_yes_no(
        "HIBACHI stopped unexpectedly",
        f"HIBACHI exited with an error (code {code}).\n\n"
        "Would you like to roll back to a previous version?",
    ):
        return
    _run_rollback(repo_root)


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
    auto_update = os.environ.get("HIBACHI_AUTO_UPDATE") == "1"
    already_reexeced = os.environ.get(_REEXEC_GUARD) == "1"
    want_rollback = ("--rollback" in sys.argv[1:]) or os.environ.get("HIBACHI_ROLLBACK") == "1"

    repo_root = updater.find_repo_root(_HERE) or os.path.dirname(_HERE)

    # Explicit rollback entry point (recovery shortcut / power users).
    if want_rollback:
        _run_rollback(repo_root)
        return 0

    splash = get_splash(enabled=not no_splash)
    try:
        if no_update:
            _msg(splash, "Update check skipped.")
        else:
            check = updater.check_for_update(repo_root=repo_root, logger=lambda m: _msg(splash, m))

            if check.status == updater.UPDATE_AVAILABLE:
                if auto_update:
                    decision = "update"
                elif updater.get_skipped_rev() == check.new_rev:
                    decision = "later"
                    _msg(splash, "Update available (previously skipped).")
                else:
                    # Only one Tk window at a time: tear the splash down to ask.
                    if splash is not None:
                        splash.close()
                        splash = None
                    decision = _ask_update(check)

                if decision == "update":
                    if splash is None and not no_splash:
                        splash = get_splash(enabled=True)
                    applied = updater.apply_update(
                        repo_root, check, logger=lambda m: _msg(splash, m)
                    )

                    # Apply a dependency update once, then re-exec so the app
                    # sees it -- only if the update actually landed.
                    if (
                        applied.status == updater.UPDATED
                        and applied.env_changed
                        and not already_reexeced
                    ):
                        _update_environment(repo_root, splash)
                        _msg(splash, "Restarting with updated dependencies...")
                        if splash is not None:
                            splash.close()
                            splash = None
                        new_env = dict(os.environ)
                        new_env[_REEXEC_GUARD] = "1"
                        os.execve(sys.executable, [sys.executable, __file__], new_env)
                        # os.execve does not return on success.
                elif decision == "skip":
                    updater.set_skipped_rev(check.new_rev)
                    _msg(splash, "Skipping this version.")
                else:  # "later"
                    _msg(splash, "Update postponed.")

        # If the splash was torn down for a prompt and we didn't re-exec, bring
        # one back for the final status lines.
        if splash is None and not no_splash:
            splash = get_splash(enabled=True)

        # Keep the desktop launcher in sync on every start so a changed icon, a
        # moved checkout, or an updated launch command self-heals. Guarded:
        # make_shortcuts refuses to run from an env that can't launch the app,
        # and _refresh_shortcut swallows errors so it never blocks startup.
        # macOS is skipped -- its .app bundle owns the launch.
        if not sys.platform.startswith("darwin"):
            _refresh_shortcut(splash)

        _msg(splash, "Launching HIBACHI...")
    finally:
        # Always tear the splash down before the Qt app grabs the screen.
        if splash is not None:
            splash.close()

    code = _launch_app(repo_root)

    # If the app failed to start / crashed, offer a way back to a good version.
    if code != 0:
        _offer_rollback_after_crash(repo_root, code)
    return code


if __name__ == "__main__":
    sys.exit(main())