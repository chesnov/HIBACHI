# --- START OF FILE segment.py ---

import sys
import os
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox

# Import the main launcher function from the utils subpackage
try:
    # Assuming 'segment.py' is in the '3D_microglia_segment' root folder
    from utils.high_level_gui.helper_funcs import launch_image_segmentation_tool
except ImportError:
    # Handle case where script might be run from a different CWD
    # or if the package structure isn't standard python path
    print("Error: Could not import from utils.helper_funcs.")
    print("Ensure you are running this script from the '3D_microglia_segment' directory,")
    print("or that the directory containing 'utils' is in your PYTHONPATH.")
    sys.exit(1)

def main():
    """
    Main application function. Checks prerequisites, launches the GUI,
    and starts the event loop.
    """
    print("Starting 3D Microglia Segmentation Tool...")

    # 1. Launch the application setup (creates windows, connects signals)
    # This function now returns the QApplication instance
    try:
        app = launch_image_segmentation_tool()
    except Exception as e:
        # Catch errors specifically from the launch function itself
        error_msg = f"Critical error during application launch:\n{str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"
        print(error_msg)
        try:
            # Try showing error message using a temporary app if needed
            temp_app = QApplication.instance() or QApplication(sys.argv)
            QMessageBox.critical(None, "Launch Error", error_msg)
        except Exception:
            pass # Ignore if GUI cannot be shown
        sys.exit(1)


    # 2. Check if launch was successful (app instance should be returned)
    if app is None:
        print("Application launch failed (app instance is None). Exiting.")
        # An error message should have already been shown by launch_image_segmentation_tool
        sys.exit(1)

    # 3. Start the Qt event loop
    print("Starting Qt Application event loop...")
    try:
        # Use sys.exit(app.exec_()) for proper exit code propagation
        exit_code = app.exec_()
        print(f"Qt Application event loop finished with exit code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        # Catch potential errors during the event loop execution or exit
        print(f"Error during or after Qt event loop: {e}")
        print(traceback.format_exc())
        sys.exit(1) # Exit with error code

# Standard Python entry point guard
if __name__ == "__main__":
    main()

# --- END OF FILE segment.py ---