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
    HIBACHI_SOFTWARE_OPENGL '1' to force Qt's bundled software OpenGL. Use this
                          in virtual machines (VirtualBox/VMware), over remote
                          desktop, or on any host whose GPU/driver can't provide
                          modern OpenGL -- symptom is a vispy/OpenGL crash on
                          opening a project ("Using glBindFramebuffer with no
                          OpenGL context" / "glBindFramebuffer not found").

Command line:
    --rollback            open the rollback chooser instead of launching
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import sys
import time

# --------------------------------------------------------------------------- #
# Launcher-side logging (stdlib only -- the launcher deliberately avoids the
# heavy GUI stack). We write a small log of our own AND capture the child's
# stdout/stderr, because the native "Aborted" message that accompanies a
# "code -6" crash is printed by the C runtime to the *child's* stderr; teeing it
# here preserves it even when the in-process faulthandler can't (e.g. the macOS
# exec path). The child (segment.py) writes its own richer logs via
# logging_setup; we point it at the same directory with HIBACHI_LOG_DIR.
# --------------------------------------------------------------------------- #
def _log_dir() -> str:
    """Mirror logging_setup.log_dir()/updater state-dir convention (stdlib only)."""
    explicit = os.environ.get("HIBACHI_LOG_DIR")
    if explicit:
        base = explicit
    else:
        state = os.environ.get("HIBACHI_STATE_DIR")
        base = os.path.join(state, "logs") if state else \
            os.path.join(os.path.expanduser("~"), ".hibachi", "logs")
    try:
        os.makedirs(base, exist_ok=True)
    except Exception:
        import tempfile
        base = os.path.join(tempfile.gettempdir(), "hibachi-logs")
        os.makedirs(base, exist_ok=True)
    return base


def _get_launcher_logger() -> logging.Logger:
    lg = logging.getLogger("hibachi.launcher")
    if lg.handlers:  # already configured
        return lg
    lg.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s [launcher] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    try:
        fh = logging.FileHandler(os.path.join(_log_dir(), "hibachi-launcher.log"),
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    except Exception:
        pass
    if sys.stderr is not None:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        lg.addHandler(sh)
    return lg


def _describe_exit(code: int) -> str:
    """Turn a subprocess return code into something a human can act on.

    On POSIX a negative code means "killed by signal N"; e.g. -6 is SIGABRT,
    which is the mysterious "code -6" the user reported. Name the signal so the
    log says WHAT killed it, not just a number.
    """
    if code is None:
        return "unknown"
    if code < 0:
        signo = -code
        try:
            name = signal.Signals(signo).name
        except (ValueError, AttributeError):
            name = f"signal {signo}"
        note = " (native abort -- see faulthandler.log)" if signo == signal.SIGABRT else ""
        return f"killed by {name}{note}"
    if code == 0:
        return "clean exit"
    return f"exit code {code}"


_LAUNCHER_LOG = _get_launcher_logger()


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


# Windows: suppress the console window that pops when this GUI (pythonw) process
# spawns console programs (micromamba, python). 0 on other platforms.
_CREATE_NO_WINDOW = 0x08000000 if sys.platform.startswith("win") else 0


def _pip_requirements_from_env(env_file: str) -> list[str]:
    """Return the packages listed under the `pip:` subsection of environment.yml.

    These are pip-managed and must be applied with pip explicitly (see
    _update_environment): `micromamba env update` does not reliably re-run the
    pip subsection when the conda-level solve finds nothing to do, so pip-only
    additions would otherwise be silently skipped on update.
    """
    try:
        import yaml
        with open(env_file) as fh:
            spec = yaml.safe_load(fh) or {}
    except Exception:
        return []
    reqs: list[str] = []
    for dep in (spec.get("dependencies") or []):
        if isinstance(dep, dict) and dep.get("pip"):
            reqs.extend(str(x) for x in dep["pip"])
    return reqs


def _update_environment(repo_root: str, splash) -> bool:
    """Bring the *currently active* env in line with environment.yml. Best-effort.

    Two passes, because they cover different dependency classes:
      1. `<mgr> env update` applies the conda-level packages.
      2. an explicit `pip install` of the env file's `pip:` subsection, run with
         the env's own interpreter. This second pass is essential: micromamba's
         `env update` may skip the pip subsection entirely when the conda solve
         finds nothing to do, so pip-only additions (e.g. imageio-ffmpeg) never
         land on update without it. Version specifiers in the file mean pip
         no-ops on already-satisfied packages, so this is safe to run every time.
    """
    env_file = os.path.join(repo_root, "install", "environment.yml")
    if not os.path.isfile(env_file):
        _msg(splash, "Skipping dependency update (environment.yml not found).")
        return False

    did_something = False

    # --- Pass 1: conda-level packages via the environment manager -------- #
    exe, _ = _find_env_manager()
    prefix = sys.prefix  # the active env we are running inside
    if exe:
        cmd = [exe, "env", "update", "--prefix", prefix, "--file", env_file, "--yes"]
        _msg(splash, "Updating dependencies (this may take a few minutes)...")
        try:
            subprocess.run(cmd, check=True, creationflags=_CREATE_NO_WINDOW)
            did_something = True
        except subprocess.CalledProcessError as exc:
            _msg(splash, f"conda-level update failed ({exc.returncode}); continuing with pip.")
        except Exception as exc:  # pragma: no cover - defensive
            _msg(splash, f"conda-level update error: {exc}; continuing with pip.")
    else:
        _msg(splash, "No environment manager found; applying pip dependencies only.")

    # --- Pass 2: pip subsection via the env's own interpreter ------------ #
    reqs = _pip_requirements_from_env(env_file)
    if reqs:
        pip_cmd = [sys.executable, "-m", "pip", "install", "--no-input", *reqs]
        _msg(splash, "Installing/verifying pip dependencies...")
        try:
            subprocess.run(pip_cmd, check=True, creationflags=_CREATE_NO_WINDOW)
            did_something = True
        except subprocess.CalledProcessError as exc:
            _msg(splash, f"pip dependency install failed ({exc.returncode}); using existing packages.")
        except Exception as exc:  # pragma: no cover - defensive
            _msg(splash, f"pip dependency error: {exc}; using existing packages.")

    if not did_something:
        _msg(splash, "Dependency update did nothing; using existing packages.")
    return did_something


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
    if chosen == dialogs.UNINSTALL:
        import uninstall

        uninstall.run(repo_root)
        return
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


def _tail(path: str, max_lines: int, max_chars: int = 20000) -> str:
    """Return the last *max_lines* lines of a file (also capped at *max_chars*).

    The interesting part of every HIBACHI log is at the END: faulthandler
    appends its crash dump last, and the lifecycle breadcrumbs run chronologically
    -- so the tail is exactly what a debugger wants to see.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return "(not present)"
    except Exception as exc:
        return f"(could not read: {exc})"
    text = "".join(lines[-max_lines:])
    if len(text) > max_chars:
        text = "\u2026(truncated)\u2026\n" + text[-max_chars:]
    return text.rstrip() or "(empty)"


#: Signatures a GPU/driver reset leaves in the child's output. A reset aborts the
#: process from native code, so faulthandler often writes nothing and the crash
#: report's most useful section is empty -- the giveaway is here instead.
_GPU_RESET_SIGNATURES = (
    "context is lost",              # amdgpu: "The CS has cancelled because the context is lost"
    "guilty of a hard recovery",    # amdgpu, same event
    "GPU hang",
    "gpu hung",
    "Xid",                          # nvidia driver error line
    "DEVICE_LOST",                  # vulkan/GL device loss
    "ring gfx timeout",             # amdgpu ring timeout
    "GL_INVALID_OPERATION out of memory",
)


def _detect_gpu_reset(child_output: str) -> str | None:
    """Return the matching signature if the child output shows a GPU reset."""
    haystack = (child_output or "").lower()
    for sig in _GPU_RESET_SIGNATURES:
        if sig.lower() in haystack:
            return sig
    return None


def _gpu_reset_advice(signature: str) -> str:
    """Actionable text for a graphics-driver crash.

    Worth calling out specifically: the workaround already exists
    (HIBACHI_SOFTWARE_OPENGL=1) but a user hitting this has no way to know that
    from a bare SIGABRT, and the native traceback is usually empty for this
    crash class, so the report looks uninformative.
    """
    return (
        "\n" + "!" * 72 + "\n"
        "LIKELY A GRAPHICS DRIVER CRASH, NOT AN ANALYSIS ERROR\n\n"
        f"The app's output contains {signature!r}, which means the graphics\n"
        "driver reset the GPU and aborted the process. Your data and results on\n"
        "disk are unaffected: this happens in the display layer, after results\n"
        "are written.\n\n"
        "Try this first -- render with software OpenGL:\n"
        "    HIBACHI_SOFTWARE_OPENGL=1\n"
        "Set it in the environment before launching. It is slower to draw but\n"
        "bypasses the GPU driver entirely, and is the supported workaround for\n"
        "driverless, virtual-machine and remote-desktop setups.\n\n"
        "Also worth doing:\n"
        "  - update your graphics driver;\n"
        "  - reduce how many viewer windows are open at once;\n"
        "  - if it recurs on one dataset, note what you were doing and report it.\n"
        "\nNote: the NATIVE CRASH TRACEBACK below is often EMPTY for this kind of\n"
        "crash, because the abort comes from the driver rather than from Python.\n"
        "That is expected and does not mean diagnostics are broken.\n"
        + "!" * 72 + "\n"
    )


def _collect_crash_report(code: int) -> tuple[str, str | None]:
    """Assemble a single diagnostics blob from the log files and save a copy.

    Returns (text_to_show, saved_report_path_or_None). The saved file is the
    reliable artefact the user can attach even after the window closes.
    """
    directory = _log_dir()
    files = [
        # (filename, human label, how many trailing lines to include)
        ("faulthandler.log", "NATIVE CRASH TRACEBACK (faulthandler)", 250),
        ("hibachi-child.log", "APP CONSOLE OUTPUT (captured by launcher)", 150),
        ("hibachi-app.log", "APP LOG / LIFECYCLE BREADCRUMBS", 250),
        ("hibachi-launcher.log", "LAUNCHER LOG", 80),
    ]
    header = (
        "HIBACHI crash report\n"
        f"  when      : {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"  platform  : {sys.platform}, Python {sys.version.split()[0]}\n"
        f"  exit      : {_describe_exit(code)} (raw code {code})\n"
        f"  logs dir  : {directory}\n"
        "\nPlease send this whole report to the developers. The most useful part is\n"
        "the NATIVE CRASH TRACEBACK below, which names the thread and line the app\n"
        "was on when it aborted.\n"
    )
    sections = [header]
    app_side_present = False  # did the child (segment.py/logging_setup) write anything?
    child_output = ""
    for fname, label, n in files:
        body = _tail(os.path.join(directory, fname), n)
        if fname in ("faulthandler.log", "hibachi-app.log") and body not in ("(not present)", "(empty)"):
            app_side_present = True
        if fname == "hibachi-child.log":
            child_output = body
        sections.append(f"\n{'=' * 72}\n{label}  [{fname}]\n{'=' * 72}\n{body}\n")

    # A graphics-driver reset explains both the abort AND the empty native
    # traceback, so this goes first when detected.
    gpu_signature = _detect_gpu_reset(child_output)
    if gpu_signature:
        sections.insert(1, _gpu_reset_advice(gpu_signature))
        _LAUNCHER_LOG.warning("crash looks like a GPU driver reset (%r)", gpu_signature)

    # If the app-side logs are absent, the crash window would otherwise look
    # empty of anything useful. Say so plainly and point at the likely cause so
    # the report is self-explanatory rather than mysteriously blank.
    if not app_side_present and not gpu_signature:
        note = (
            "\n" + "!" * 72 + "\n"
            "NOTE: No app-side diagnostics were found (faulthandler.log / "
            "hibachi-app.log\nare missing or empty in the logs directory above).\n\n"
            "That means the in-app logging isn't active. Check that the child-side\n"
            "files are installed and being run:\n"
            "  - segment.py                       (should import logging_setup)\n"
            "  - utils/high_level_gui/logging_setup.py\n"
            "  - utils/high_level_gui/app_launch.py, gui_manager.py\n"
            "The LAUNCHER LOG and APP CONSOLE OUTPUT below still apply.\n"
            + "!" * 72 + "\n"
        )
        sections.insert(1, note)  # right after the header, before the log sections

    text = "".join(sections)

    saved_path = None
    try:
        saved_path = os.path.join(directory, "crash-report.txt")
        with open(saved_path, "w", encoding="utf-8") as fh:
            fh.write(text)
    except Exception as exc:
        _LAUNCHER_LOG.warning("could not save consolidated crash report: %s", exc)
        saved_path = None
    return text, saved_path


def _offer_rollback_after_crash(repo_root: str, code: int) -> None:
    """After a failed launch, show the crash window with copyable diagnostics and
    (if there is an earlier version) offer to roll back to it."""
    import dialogs

    try:
        versions = updater.list_versions(repo_root, limit=15)
    except Exception:
        versions = []
    can_rollback = len(versions) >= 2

    details, report_path = _collect_crash_report(code)
    summary = f"HIBACHI stopped unexpectedly \u2014 {_describe_exit(code)} (raw exit code {code})."

    # Always surface the diagnostics, even when rollback isn't possible, so the
    # user can copy the logs from the crash window itself.
    want_rollback = dialogs.crash_report(
        summary=summary,
        details=details,
        log_dir=_log_dir(),
        report_path=report_path,
        offer_rollback=can_rollback,
    )
    if want_rollback and can_rollback:
        _run_rollback(repo_root)


def _launch_app(repo_root: str) -> int:
    """Launch the real application and return its exit code (POSIX/Windows)."""
    entry = os.path.join(repo_root, APP_ENTRY)
    if not os.path.isfile(entry):
        _LAUNCHER_LOG.error("cannot find %s", entry)
        return 1

    # Tell the child where to put its logs so parent and child agree, and so the
    # user only has one folder to send us. The child reads this in logging_setup.
    os.environ.setdefault("HIBACHI_LOG_DIR", _log_dir())
    _LAUNCHER_LOG.info("Launching %s (logs -> %s)", entry, os.environ["HIBACHI_LOG_DIR"])

    # macOS: REPLACE this process with the GUI instead of spawning a child. The
    # .app bundle already owns a Dock tile ("HIBACHI"); if we launch segment.py
    # as a separate process it becomes its own foreground app and macOS adds a
    # SECOND tile labelled "python". exec keeps a single process (same PID) under
    # the bundle, so only one icon shows. The splash + update check have already
    # finished by now, so nothing is lost -- though note the post-launch
    # rollback-on-crash offer (below, in main) can't run after an exec; macOS
    # users reach rollback via `--rollback` instead.
    if sys.platform == "darwin":
        os.chdir(repo_root)  # so `import utils` resolves (mirrors cwd= below)
        # exec REPLACES this process, so we can't observe the child's exit code
        # or tee its output from here. That's fine: the child arms faulthandler
        # itself (logging_setup), so a native abort still lands in
        # faulthandler.log. Record the handoff before we vanish.
        _LAUNCHER_LOG.info("macOS: exec-ing into the app (single-process mode); "
                           "crash diagnostics handled in-process by the app.")
        for h in list(_LAUNCHER_LOG.handlers):
            try:
                h.flush()
            except Exception:
                pass
        os.execv(sys.executable, [sys.executable, entry])
        # os.execv does not return on success.

    # All other platforms (incl. Windows): run the real entry point as a plain,
    # named script with the env's own interpreter, from the repo root so
    # `import utils` resolves. We deliberately do NOT use `python -c "exec(...)"`:
    # executing a code string is a behavioural pattern endpoint-security tools
    # (EDR) flag as loader-like, and it is unnecessary here -- segment.py already
    # registers the env's DLL directories itself, at its very top, before Qt is
    # imported (see the os.add_dll_directory block in segment.py). Running a real
    # file on disk is both cleaner and far less likely to trip a false positive.
    # Capture the child's combined stdout+stderr and tee it to a log while still
    # echoing to the console. This preserves the C-runtime "Aborted" line and any
    # last-gasp output that precedes a native crash -- output that would be lost
    # if we simply inherited the parent's streams and the process then vanished.
    child_log_path = os.path.join(_log_dir(), "hibachi-child.log")
    try:
        child_log = open(child_log_path, "a", buffering=1, encoding="utf-8", errors="replace")
        child_log.write(f"\n===== child launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    except Exception:
        child_log = None

    try:
        proc = subprocess.Popen(
            [sys.executable, entry],
            cwd=repo_root,
            creationflags=_CREATE_NO_WINDOW,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as exc:
        _LAUNCHER_LOG.error("failed to start child process: %s", exc)
        if child_log:
            child_log.close()
        return 1

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if sys.stdout is not None:
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except Exception:
                    pass
            if child_log:
                try:
                    child_log.write(line)
                except Exception:
                    pass
    except Exception as exc:  # never let a tee error mask the real exit code
        _LAUNCHER_LOG.warning("error while streaming child output: %s", exc)
    finally:
        code = proc.wait()
        if child_log:
            try:
                child_log.write(f"===== child exited: {_describe_exit(code)} (raw={code}) =====\n")
                child_log.close()
            except Exception:
                pass
    return code


def _activate_env_path() -> None:
    """
    Put the active env's binary directories on PATH.

    We launch the app with the env's interpreter directly (not `micromamba run`),
    which is what fixed the DLL-not-found crash -- but it also skips conda
    activation, so PATH no longer contains the env's bin dirs. Tools the app
    shells out to via subprocess (notably `git`, used by the self-updater) then
    fail with "git executable not found on PATH", even though git ships inside
    the env. This restores just the PATH part of activation. subprocess and the
    launched child both inherit os.environ, so this covers both.
    """
    prefix = sys.prefix
    if sys.platform.startswith("win"):
        dirs = [
            prefix,
            os.path.join(prefix, "Library", "bin"),
            os.path.join(prefix, "Library", "cmd"),
            os.path.join(prefix, "Library", "mingw-w64", "bin"),
            os.path.join(prefix, "Library", "mingw64", "bin"),
            os.path.join(prefix, "Library", "usr", "bin"),
            os.path.join(prefix, "Scripts"),
        ]
    else:
        dirs = [os.path.join(prefix, "bin")]
    dirs = [d for d in dirs if os.path.isdir(d)]
    current = os.environ.get("PATH", "")
    # Prepend, skipping any already present, so env tools win but we don't bloat PATH.
    have = set(current.split(os.pathsep))
    new = [d for d in dirs if d not in have]
    if new:
        os.environ["PATH"] = os.pathsep.join(new + ([current] if current else []))


def main() -> int:
    no_update = os.environ.get("HIBACHI_NO_UPDATE") == "1"
    no_splash = os.environ.get("HIBACHI_NO_SPLASH") == "1"
    auto_update = os.environ.get("HIBACHI_AUTO_UPDATE") == "1"
    already_reexeced = os.environ.get(_REEXEC_GUARD) == "1"
    want_rollback = ("--rollback" in sys.argv[1:]) or os.environ.get("HIBACHI_ROLLBACK") == "1"

    repo_root = updater.find_repo_root(_HERE) or os.path.dirname(_HERE)

    # Restore the env's PATH first: the update check below shells out to `git`,
    # which lives inside the env but isn't on PATH when we launch the interpreter
    # directly. Do this before check_for_update or it reports "git not found".
    _activate_env_path()

    # Software-OpenGL fallback for VMs / remote desktop / driverless hosts. Must
    # be set before the child creates its QApplication and before vispy imports
    # its GL backend, so we set it on our own environment (the launch below
    # inherits it). Two pieces are needed and BOTH matter:
    #   * QT_OPENGL=software  -> Qt loads its bundled opengl32sw.dll (Mesa
    #     llvmpipe, OpenGL 3.3) instead of the host's legacy OpenGL 1.1.
    #   * VISPY_GL_LIB=<...>/opengl32sw.dll  -> vispy loads its GL functions from
    #     the SAME Mesa library. vispy otherwise loads the host opengl32.dll on
    #     its own and never sees Qt's software context, which is why QT_OPENGL
    #     alone leaves "glBindFramebuffer not found" / "no OpenGL context".
    # Opt-in only, so machines with real GPUs keep hardware acceleration.
    if os.environ.get("HIBACHI_SOFTWARE_OPENGL") == "1":
        os.environ.setdefault("QT_OPENGL", "software")
        if sys.platform.startswith("win") and not os.environ.get("VISPY_GL_LIB"):
            _sp = os.path.join(sys.prefix, "Lib", "site-packages")
            for _rel in (
                os.path.join("PyQt5", "Qt5", "bin", "opengl32sw.dll"),
                os.path.join("PyQt5", "Qt", "bin", "opengl32sw.dll"),
                os.path.join("PySide2", "opengl32sw.dll"),
                os.path.join("PySide6", "opengl32sw.dll"),
            ):
                _cand = os.path.join(_sp, _rel)
                if os.path.isfile(_cand):
                    os.environ["VISPY_GL_LIB"] = _cand
                    break

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

    # Record how the app exited. A negative code is a signal death (e.g. -6 =
    # SIGABRT); _describe_exit names it and points at faulthandler.log.
    if code == 0:
        _LAUNCHER_LOG.info("HIBACHI exited cleanly.")
    else:
        _LAUNCHER_LOG.error(
            "HIBACHI exited abnormally: %s. Diagnostics in %s "
            "(hibachi-app.log, faulthandler.log, hibachi-child.log).",
            _describe_exit(code), _log_dir(),
        )

    # If the app failed to start / crashed, offer a way back to a good version.
    if code != 0:
        _offer_rollback_after_crash(repo_root, code)
    return code


if __name__ == "__main__":
    sys.exit(main())