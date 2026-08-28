# `segment.py`

**Location:** repository root

## Overview

`segment.py` is the application entry point. It performs no image analysis — it
bootstraps the environment, arms diagnostics, and launches the GUI.

**The desktop shortcut does not run this file.** It runs `launcher/run_app.py`,
which handles the splash, the update check and crash reporting, and launches
`segment.py` as a child. Running `segment.py` by hand skips all of that.

## Key responsibilities, in execution order

Several of these must happen before the GUI stack is imported.

### 1. Diagnostics, before anything heavy

`logging_setup.configure_logging(role="app")` is called **first**, ahead of any
Qt / vispy / napari import.

This is first because of one specific failure. The app can exit with code -6,
which is `SIGABRT` — a native abort raised inside Qt, vispy or OpenGL, or by a
memmap freed under a worker. It terminates the process without unwinding Python,
so `sys.excepthook` never runs and buffered stdout is lost. `faulthandler`
installs a signal handler that writes every thread's traceback to
`faulthandler.log` on such a signal. Installing it before the GUI imports covers
a crash during startup as well.

This sets up four things: rotating file logs, `faulthandler`, a Qt message
handler, and exception hooks for the main thread and QThreads. The import is
best-effort — a diagnostics failure prints a warning and the app continues.

See [Diagnostics](diagnostics.md) for log locations.

### 2. Windows DLL registration

On Windows, `os.add_dll_directory()` is called for `sys.prefix` and
`sys.prefix/Library/bin`.

pip-built extensions (PyQt5, vispy, numba) are compiled against the MSVC runtime.
conda-forge ships that runtime inside the environment, but since Python 3.8 the
loader does not search `PATH` for an extension's dependent DLLs. On a clean
machine without the system VC++ Redistributable this surfaces as
**"MSVCP140.dll was not found"** the moment PyQt5 imports. Registering the
directories explicitly fixes it with no admin rights required.

### 3. The vispy arcball monkeypatch

Patches `vispy.scene.cameras.arcball._arcball` to truncate its `xy` argument to
two values. Some input devices deliver a third component (pressure or z-depth),
and the unpatched function assumes exactly two — producing a
*"too many values to unpack"* crash when switching between panning (Shift+Click)
and rotating.

### 4. High-DPI scaling

Sets `QT_AUTO_SCREEN_SCALE_FACTOR=1`, so the interface scales on Retina and 4K
displays.

### 5. Global exception handling

`global_exception_hook()` replaces `sys.excepthook`.

*   Logs the full traceback at CRITICAL to the file log.
*   Shows a `QMessageBox` with the traceback in its details pane, the log
    directory, and a **Copy to clipboard** button — the same affordance the
    launcher's crash dialog gives for native crashes.

This catches exceptions that reach the top of the interpreter. A native abort
does not, which is what `faulthandler` above is for.

### 6. Lifecycle

`main()`, in order:

1.  Calls `multiprocessing.freeze_support()`. On Windows and in a frozen build,
    a worker process re-imports the entry module; without this guard each one
    would launch another copy of the GUI.
2.  Calls `launch_image_segmentation_tool()` from
    `utils.high_level_gui.helper_funcs`, which creates the `QApplication` and the
    first window. A failure here is logged at CRITICAL and shown in a message box
    built from a temporary `QApplication` if none exists yet, then exits with 1.
3.  Exits with 1 if that returned `None`.
4.  Enters `app.exec_()` and blocks there until the last window closes, then
    exits with the loop's return code.

## Developer usage

```bash
python segment.py
```

Useful when you want to skip the launcher — no update check, no splash, no crash
reporting or rollback. For normal use, launch via the shortcut or `run_app.py`.
