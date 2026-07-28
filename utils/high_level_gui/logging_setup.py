"""logging_setup: one place to configure diagnostics for HIBACHI.

Why this module exists
----------------------
The app can die with a bare "exited with an error (code -6)". On POSIX, a
negative subprocess return code means the child was *killed by a signal*, and
signal 6 is SIGABRT -- a native (C/C++) abort coming from Qt / napari / vispy /
OpenGL or a memmap being pulled out from under a worker. A SIGABRT terminates
the process immediately: Python's ``sys.excepthook`` never runs, and any
``print()`` diagnostics that were only on stdout are lost the moment the process
is gone. So to debug it we need diagnostics that survive a native abort and land
in a file on disk.

This module wires up four complementary layers:

  1. **File + console logging** (stdlib ``logging``, rotating file). Everything
     the app logs goes to ``<logdir>/hibachi-<role>.log`` and is flushed.
  2. **faulthandler** -> ``<logdir>/faulthandler.log``. On SIGABRT/SIGSEGV/etc.
     it dumps a Python traceback for *every* thread. This is the single most
     useful artefact for a "code -6" crash: it tells you which thread and which
     line the process was on at the instant it aborted.
  3. **Qt message handler**. Qt's own warnings/criticals (e.g. "QThread:
     Destroyed while thread is still running", "QObject::~QObject: Timers cannot
     be stopped from another thread") almost always print immediately before one
     of these aborts. Routing them through logging captures that breadcrumb.
  4. **Exception hooks** for the main thread and for ``QThread``/``threading``
     workers, so any *Python* exception is logged (with full traceback) instead
     of vanishing.

Everything here is best-effort and import-light: it deliberately does NOT import
the heavy GUI stack, so it is safe to call at the very top of ``segment.py``
before Qt/vispy are touched. The Qt message handler is installed lazily and only
if PyQt5 is importable.

Public API
----------
    configure_logging(role="app")        -> pathlib.Path (the log directory)
    get_logger(name)                     -> logging.Logger
    lifecycle(event, **fields)           -> None      (structured breadcrumb)
    dump_now(label="manual")             -> None      (force a traceback dump)
    install_qt_message_handler()         -> None      (called by configure_logging)
"""

from __future__ import annotations

import atexit
import faulthandler
import logging
import logging.handlers
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Location
# --------------------------------------------------------------------------- #
def log_dir() -> Path:
    """Directory for all HIBACHI logs.

    Resolution order (first that is set wins):
        $HIBACHI_LOG_DIR
        $HIBACHI_STATE_DIR/logs
        ~/.hibachi/logs
    Kept in step with the launcher's state dir (see updater._state_path) so logs
    live next to the rest of the app's on-disk state. Always returns a directory
    that exists (best-effort: falls back to the temp dir if it cannot be made).
    """
    explicit = os.environ.get("HIBACHI_LOG_DIR")
    if explicit:
        base = Path(explicit)
    else:
        state = os.environ.get("HIBACHI_STATE_DIR")
        base = (Path(state) if state else Path.home() / ".hibachi") / "logs"
    try:
        base.mkdir(parents=True, exist_ok=True)
        return base
    except Exception:
        import tempfile
        fallback = Path(tempfile.gettempdir()) / "hibachi-logs"
        try:
            fallback.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return fallback


# --------------------------------------------------------------------------- #
# Module state (so configure_logging is idempotent and file handles survive)
# --------------------------------------------------------------------------- #
_configured = False
_fault_file = None          # keep the faulthandler file object alive for the
                            # whole process; faulthandler writes to its fd.
_LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(processName)s/%(threadName)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(role: str = "app", level: int = logging.DEBUG) -> Path:
    """Set up logging + crash diagnostics. Safe to call more than once.

    Parameters
    ----------
    role : str
        A short tag distinguishing writers of the same log directory, e.g.
        ``"launcher"`` (run_app.py) vs ``"app"`` (segment.py). It becomes part
        of the log filename so a parent and its child don't clobber each other.
    level : int
        Root logging level. DEBUG by default -- we want everything while a crash
        is being chased; noisy third-party loggers are turned down below.

    Returns
    -------
    pathlib.Path
        The directory the logs were written to (handy to show the user).
    """
    global _configured, _fault_file

    directory = log_dir()

    if _configured:
        # Already set up in this process; just hand back the location.
        return directory

    # --- stdlib logging: rotating file + console -------------------------- #
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    log_path = directory / f"hibachi-{role}.log"
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except Exception as exc:  # never let logging setup crash the app
        print(f"[logging] could not open log file {log_path}: {exc}", file=sys.stderr)

    # Console handler -- mirrors to the terminal when there is one. When the app
    # is launched from a GUI (pythonw / .app bundle) stdout may be None; guard.
    stream = sys.stderr if sys.stderr is not None else sys.stdout
    if stream is not None:
        console = logging.StreamHandler(stream)
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        root.addHandler(console)

    # Third-party chatter: keep the file readable.
    for noisy in ("numba", "matplotlib", "PIL", "vispy", "OpenGL", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.captureWarnings(True)  # route warnings.warn(...) into logging

    # --- faulthandler: dumps every thread's traceback on a fatal signal --- #
    # This is what makes a "code -6" (SIGABRT) debuggable: the dump names the
    # thread and line the process died on. We keep our own always-open file so
    # the dump is captured even though the process is being torn down.
    try:
        fault_path = directory / "faulthandler.log"
        _fault_file = open(fault_path, "a", buffering=1, encoding="utf-8")
        _fault_file.write(
            f"\n===== faulthandler armed {time.strftime(_DATE_FORMAT)} "
            f"(role={role}, pid={os.getpid()}) =====\n"
        )
        _fault_file.flush()
        # enable() installs handlers for the fatal signals (SIGSEGV, SIGFPE,
        # SIGABRT, SIGBUS, SIGILL) and dumps all threads when one fires.
        faulthandler.enable(file=_fault_file, all_threads=True)
    except Exception as exc:
        logging.getLogger("hibachi.logging").warning("faulthandler unavailable: %s", exc)

    # --- Python exception hooks (main thread + workers) ------------------- #
    _install_excepthooks()

    # --- Qt fatal/warning messages (best-effort; needs PyQt5) ------------- #
    install_qt_message_handler()

    _configured = True

    log = logging.getLogger("hibachi")
    log.info("=" * 70)
    log.info("HIBACHI logging started (role=%s, pid=%s)", role, os.getpid())
    log.info("Python %s on %s", sys.version.split()[0], sys.platform)
    log.info("Log directory: %s", directory)
    log.info("Crash traceback (native aborts) -> %s", directory / "faulthandler.log")
    log.info("=" * 70)

    atexit.register(lambda: logging.getLogger("hibachi").info("Process exiting (pid=%s).", os.getpid()))
    return directory


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger (e.g. get_logger('gui_manager'))."""
    return logging.getLogger(f"hibachi.{name}" if not name.startswith("hibachi") else name)


def lifecycle(event: str, **fields: Any) -> None:
    """Log a structured lifecycle breadcrumb.

    Use this around open/close/cleanup so the last lines before a native abort
    tell you exactly which step was in flight, e.g.::

        lifecycle("viewer.open", folder=name)
        lifecycle("cleanup.start", worker_running=True, layers=3)

    The message is greppable and the fields are shown as key=value.
    """
    extra = " ".join(f"{k}={v!r}" for k, v in fields.items())
    logging.getLogger("hibachi.lifecycle").info("%-22s %s", event, extra)


def dump_now(label: str = "manual") -> None:
    """Force a full all-threads traceback dump into faulthandler.log right now.

    Useful for diagnosing a *hang* (the app is frozen but not aborting) or for
    marking the exact state at a suspicious point in the close path.
    """
    if _fault_file is not None:
        try:
            _fault_file.write(f"\n----- manual dump ({label}) {time.strftime(_DATE_FORMAT)} -----\n")
            _fault_file.flush()
            faulthandler.dump_traceback(file=_fault_file, all_threads=True)
            _fault_file.flush()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _install_excepthooks() -> None:
    """Route uncaught Python exceptions (main + worker threads) into logging."""
    log = logging.getLogger("hibachi.uncaught")

    prev_hook = sys.excepthook

    def _hook(exctype, value, tb):
        # KeyboardInterrupt is a normal Ctrl-C; don't shout about it.
        if issubclass(exctype, KeyboardInterrupt):
            prev_hook(exctype, value, tb)
            return
        log.critical("Uncaught exception on main thread", exc_info=(exctype, value, tb))
        try:
            prev_hook(exctype, value, tb)  # preserve existing behaviour (exit code, popup)
        except Exception:
            pass

    sys.excepthook = _hook

    # threading.excepthook exists on Python 3.8+. QThread exceptions that reach
    # the C++ layer are handled by faulthandler; this catches pure-Python worker
    # threads and anything that surfaces through the threading machinery.
    if hasattr(threading, "excepthook"):
        def _thread_hook(args):
            if issubclass(args.exc_type, SystemExit):
                return
            log.critical(
                "Uncaught exception on thread %r",
                getattr(args.thread, "name", "?"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )

        threading.excepthook = _thread_hook


def install_qt_message_handler() -> None:
    """Install a Qt message handler that forwards Qt's own logs into logging.

    No-op if PyQt5 isn't importable yet (e.g. called from the launcher). Safe to
    call again after Qt is available.
    """
    try:
        from PyQt5.QtCore import QtMsgType, qInstallMessageHandler  # type: ignore
    except Exception:
        return

    qt_log = logging.getLogger("hibachi.qt")
    _level_for = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def _handler(mode, context, message):
        level = _level_for.get(mode, logging.INFO)
        where = ""
        try:
            if context is not None and context.file:
                where = f" ({context.file}:{context.line})"
        except Exception:
            pass
        qt_log.log(level, "%s%s", message, where)
        # A Qt *fatal* message is about to call abort(): capture all threads now,
        # while we still can, before the SIGABRT lands.
        if mode == QtMsgType.QtFatalMsg:
            dump_now("qt-fatal")

    try:
        qInstallMessageHandler(_handler)
    except Exception:
        pass