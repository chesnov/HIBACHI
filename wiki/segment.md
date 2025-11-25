# Segment.py

## Overview
`segment.py` serves as the **Entry Point** (Controller) for the 3D Microglia Segmentation Tool. It is the script that users execute to start the application. It does not perform image analysis itself but is responsible for bootstrapping the environment and launching the GUI.

## Key Responsibilities
1.  **Environment Configuration:** Sets operating system variables (like High-DPI scaling for Windows/Linux) to ensure the GUI renders correctly on modern screens.
2.  **Global Exception Handling:** Implements a `sys.excepthook` to catch unhandled errors. This prevents the application from silently closing (crashing) without feedback, which is critical for debugging memory issues.
3.  **Dependency Check:** Verifies that the `utils` package is accessible before attempting to load heavy modules.
4.  **Lifecycle Management:** Initializes the `QApplication`, launches the main window, and manages the Qt Event Loop.

## Functions

### `global_exception_hook(exctype, value, tb)`
*   **Purpose:** Catches any exception that isn't caught by a `try...except` block elsewhere in the code.
*   **Behavior:** Prints the traceback to the console and attempts to launch a `QMessageBox` to display the error to the user.
*   **Why it matters:** In memory-intensive applications, crashes can occur deep within C++ bindings or threads. This hook ensures the user is notified.

### `main()`
*   **Purpose:** The main execution routine.
*   **Process Flow:**
    1.  Sets `multiprocessing.freeze_support()` (essential for Windows executables).
    2.  Calls `launch_image_segmentation_tool()` from `utils.high_level_gui.helper_funcs`.
    3.  Validates that the application instance started correctly.
    4.  Starts the `app.exec_()` event loop, blocking execution until the window is closed.

## Usage
Run from the terminal in the project root:
```bash
python segment.py
```