# Diagnostics, Logs and Crash Reports

**Corresponding modules:**
*   `utils/high_level_gui/logging_setup.py` — the four diagnostic layers
*   `launcher/run_app.py` — child-output capture, crash report, rollback offer
*   `launcher/dialogs.py` — the crash window

## The problem this exists for

HIBACHI can exit with **code -6**. On POSIX a negative exit code means the
process was killed by a signal, and signal 6 is `SIGABRT` — a native abort from
Qt, napari, vispy, OpenGL, or a memmap freed under a worker.

A `SIGABRT` terminates the process immediately. Python's `sys.excepthook` never
runs, and anything printed to stdout but not yet flushed is gone. So diagnostics
have to survive a native abort and land in a file.

## Four layers

Set up by `configure_logging()`, which `segment.py` calls before any Qt, vispy or
napari import — so a crash *during* startup is captured too.

1.  **File and console logging.** Everything the app logs goes to
    `hibachi-<role>.log`, rotating at 5 MB with 5 backups.
2.  **`faulthandler`** → `faulthandler.log`. On `SIGABRT`, `SIGSEGV` and similar,
    dumps a Python traceback for **every thread**. For a code -6 crash this is
    the one artefact that says which thread and which line the process was on.
3.  **Qt message handler.** Qt's own warnings — *"QThread: Destroyed while thread
    is still running"*, *"Timers cannot be stopped from another thread"* — go to
    Qt's own output rather than through `logging`, so this routes them into the
    log file where they sit next to the surrounding events.
4.  **Exception hooks** for the main thread and for `QThread` / `threading`
    workers, so a Python exception is logged with its traceback rather than
    vanishing.

All of it is best-effort: a diagnostics failure prints a warning and the app
continues.

## Where the logs are

Resolution order, first one set wins:

```
$HIBACHI_LOG_DIR
$HIBACHI_STATE_DIR/logs
~/.hibachi/logs
```

If none of those can be created, a temp directory is used, so logging never
becomes the reason a launch fails.

| File | Written by | Holds |
| :--- | :--- | :--- |
| `hibachi-app.log` | the app | Everything the app logs, including lifecycle breadcrumbs |
| `faulthandler.log` | the app | Per-thread tracebacks at the moment of a native abort |
| `hibachi-launcher.log` | the launcher | Launch, update, rollback and exit codes |
| `hibachi-child.log` | the launcher | The app's combined stdout and stderr, teed |
| `crash-report.txt` | the launcher | The consolidated report from the last crash |

`hibachi-child.log` exists because the C runtime prints its abort message to the
child's stderr, where Python's logging cannot see it. The launcher tees the
child's output so that message is on disk.

> **On macOS there is no child log.** The launcher `exec`s into the app so the
> Dock shows one tile instead of two, which replaces the launcher process. It
> cannot then observe the exit code or tee the output. `faulthandler.log` still
> works, because the app arms it itself. The post-crash rollback offer cannot
> run either, so macOS reaches rollback through `--rollback`.

## Lifecycle breadcrumbs

`lifecycle(event, **fields)` writes structured one-line events —
`viewer.open`, `worker.start`, `cleanup.begin`, `napari.destroyed`, `app.quit`
and similar. Reading them backwards from a crash shows what the app was doing.

`dump_now(label)` forces a traceback dump on demand, for a hang rather than a
crash.

## The crash report

When the app exits abnormally, the launcher assembles `crash-report.txt` from the
four logs, with a header giving the time, platform, Python version and how the
process died. The crash window shows it with a **Copy to clipboard** button and,
where an earlier version exists, offers to roll back.

Two cases the report handles explicitly:

*   **A graphics driver reset.** If the child's output contains a signature such
    as `context is lost`, `guilty of a hard recovery`, `Xid` or `DEVICE_LOST`,
    the report says so at the top: the crash is in the display layer rather than
    the analysis, results already on disk are unaffected, and
    `HIBACHI_SOFTWARE_OPENGL=1` is the workaround. It also notes that an empty
    native traceback is expected for this class of crash, because the abort comes
    from the driver rather than from Python.
*   **No app-side logs at all.** If `faulthandler.log` and `hibachi-app.log` are
    both absent, the report says which files are missing and what that implies,
    rather than presenting an empty section.

## Environment variables

| Variable | Effect |
| :--- | :--- |
| `HIBACHI_LOG_DIR` | Put logs here |
| `HIBACHI_STATE_DIR` | Move all app state, logs included |
| `HIBACHI_SOFTWARE_OPENGL=1` | Render with software OpenGL — the fix for a driver reset, and for VMs, remote desktops and machines without a usable GPU driver |

See [Installation](../INSTALL.md) for the full set.

## When reporting a problem

Send `crash-report.txt` from the log directory. It already contains the four logs
and the exit description. If the app is still running and misbehaving rather than
crashing, send `hibachi-app.log`.

---

## Where to go next

*   [`segment.py`](segment.md) — the order diagnostics are armed in, and why.
*   [Installation](../INSTALL.md) — environment variables, rollback, uninstalling.
