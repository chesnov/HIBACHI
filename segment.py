# --- START OF FILE segment.py ---
"""
Entry Point for the 3D Segmentation Tool.

This script acts as the 'Controller' in the application architecture.
It is responsible for:
1. Setting up the execution environment (High-DPI scaling, Multiprocessing support).
2. Initializing the Qt Application (The GUI Framework).
3. Launching the main logic located in 'utils'.
4. Handling the Event Loop and graceful shutdowns.
5. catching global exceptions to prevent silent crashes.
"""

import sys
import os
import traceback
import multiprocessing

# -----------------------------------------------------------------------------
# 0. Diagnostics FIRST -- before anything heavy (Qt / vispy / napari) imports.
# -----------------------------------------------------------------------------
# The app can die with a bare "exited with an error (code -6)". Code -6 is a
# SIGABRT: a native (C/C++) abort from Qt/vispy/OpenGL or a memmap being freed
# under a worker. It kills the process instantly, so sys.excepthook never runs
# and stdout prints are lost. logging_setup arms faulthandler (which dumps every
# thread's traceback on such an abort, to <logdir>/faulthandler.log), plus file
# logging, a Qt message handler, and exception hooks. Arming it here -- before
# the GUI stack is even imported -- means a crash *during* startup is captured
# too. Import is best-effort so a diagnostics problem can never stop the app.
try:
    from utils.high_level_gui.logging_setup import configure_logging, get_logger
    _LOG_DIR = configure_logging(role="app")
    log = get_logger("segment")
    log.info("segment.py starting; diagnostics active.")
except Exception as _diag_exc:  # pragma: no cover - diagnostics must never block launch
    print(f"[segment] WARNING: could not initialise logging ({_diag_exc}); "
          "continuing without file diagnostics.")

    def get_logger(_name):  # type: ignore
        import logging
        return logging.getLogger(_name)

    log = get_logger("segment")

# -----------------------------------------------------------------------------
# Windows: make the conda env's MSVC runtime (MSVCP140.dll & friends) findable
# -----------------------------------------------------------------------------
# pip-built extensions (PyQt5, vispy, numba, ...) are compiled against the
# Microsoft Visual C++ runtime. conda-forge ships that runtime *inside* the env
# (via the vc14_runtime dependency of python), but since Python 3.8 the loader
# no longer searches PATH for an extension module's dependent DLLs. On a clean
# machine without the system VC++ Redistributable that surfaces as
# "MSVCP140.dll was not found" the moment PyQt5/vispy import. Registering the
# env's DLL directories explicitly resolves it, with no admin rights needed.
if sys.platform == "win32":
    for _dll_dir in (sys.prefix, os.path.join(sys.prefix, "Library", "bin")):
        if os.path.isdir(_dll_dir):
            try:
                os.add_dll_directory(_dll_dir)
            except OSError:
                pass

# This monkeypatch fixes the "too many values to unpack" crash in Vispy/Napari
# when switching between Panning (Shift+Click) and Rotating.
try:
    import vispy.scene.cameras.arcball as arcball_module
    
    # Save the original function
    _original_arcball = arcball_module._arcball

    def _patched_arcball(xy, wh):
        # The crash happens because 'xy' sometimes contains 3 or more values 
        # (e.g. pressure or z-depth) but the function assumes exactly 2 (x, y).
        if len(xy) > 2:
            xy = xy[:2]
        return _original_arcball(xy, wh)

    # Apply the patch
    arcball_module._arcball = _patched_arcball
    log.info("Applied Vispy Arcball camera patch.")
except Exception as e:
    log.warning("Could not apply Vispy patch: %s", e)

# PyQt5 is the GUI framework used. 
# QApplication is the singleton that manages the GUI control flow and main settings.
# QMessageBox is used for reporting critical errors to the user.
from PyQt5.QtWidgets import QApplication, QMessageBox # type: ignore

# -----------------------------------------------------------------------------
# 1. Environment Configuration
# -----------------------------------------------------------------------------
# Enable High-DPI scaling. This is crucial for viewing large microscopy images
# on modern laptop screens (Retina/4K) without them looking tiny or blurry.
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

# -----------------------------------------------------------------------------
# 2. Dependency Import
# -----------------------------------------------------------------------------
try:
    # We assume the directory structure is:
    # ./segment.py
    # ./utils/high_level_gui/helper_funcs.py
    from utils.high_level_gui.helper_funcs import launch_image_segmentation_tool
except ImportError:
    # If the user runs this script from the wrong directory, Python won't find 'utils'.
    # We catch this early to print a helpful debugging message.
    print("CRITICAL IMPORT ERROR: Could not import 'launch_image_segmentation_tool'.")
    print("-----------------------------------------------------------------------")
    print("Troubleshooting:")
    print("1. Ensure you are running this script from the root 'HIBACHI' directory.")
    print("   Current Working Directory:", os.getcwd())
    print("2. Ensure the 'utils' folder contains an '__init__.py' file.")
    print("3. Check your PYTHONPATH environment variable.")
    print("-----------------------------------------------------------------------")
    sys.exit(1)

# -----------------------------------------------------------------------------
# 3. Global Exception Handling
# -----------------------------------------------------------------------------
def global_exception_hook(exctype, value, tb):
    """
    Catches unhandled exceptions that would otherwise crash the application silently.
    This is critical for memory-heavy apps where OOM (Out of Memory) errors might occur.
    """
    error_msg = "".join(traceback.format_exception(exctype, value, tb))
    log.critical("Uncaught exception reached the top level:\n%s", error_msg)
    
    # Try to show a GUI popup so the user knows what happened
    app = QApplication.instance()
    if app:
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle("Critical Application Error")
        error_box.setText("An unexpected error occurred.")
        error_box.setDetailedText(error_msg)
        error_box.exec_()

    # Call the default handler to ensure proper exit code
    sys.__excepthook__(exctype, value, tb)

# Register the hook
sys.excepthook = global_exception_hook


# -----------------------------------------------------------------------------
# 4. Main Execution Routine
# -----------------------------------------------------------------------------
def main():
    """
    Main application function. 
    """
    log.info("--- Starting HIBACHI ---")

    # Guard for Windows Multiprocessing
    # If this app is ever frozen (PyInstaller) or run on Windows, this prevents 
    # the subprocesses from infinitely spawning new instances of the GUI.
    multiprocessing.freeze_support()

    # A. Launch the application setup
    # launch_image_segmentation_tool() creates the QApplication and the Main Window.
    try:
        app = launch_image_segmentation_tool()
    except Exception as e:
        # If we fail here, the app didn't even start (e.g., config error, missing lib).
        error_msg = f"Critical error during application launch:\n{str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"
        log.critical(error_msg)
        
        # Attempt to create a temporary app just to show the error message box
        try:
            temp_app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "Launch Error", error_msg)
        except Exception:
            pass # Use standard print if GUI completely fails
        sys.exit(1)

    # B. Validation
    if app is None:
        log.error("Application instance is None. Launch failed gracefully.")
        sys.exit(1)

    # C. Start the Event Loop
    # The event loop waits for user input (clicks, keypresses).
    # Processing stops here until the user closes the window.
    log.info("GUI initialized. Starting Qt event loop...")
    try:
        exit_code = app.exec_()
        log.info("Application closed normally with exit code: %s", exit_code)
        sys.exit(exit_code)
    except Exception as e:
        log.critical("Error during Qt event loop: %s\n%s", e, traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
# --- END OF FILE segment.py ---