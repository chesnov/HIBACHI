"""app_launch: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
import sys
import gc
import yaml  # type: ignore
import tifffile as tiff  # type: ignore
import napari  # type: ignore
from magicgui import magicgui  # type: ignore
from typing import Any
from PyQt5.QtCore import QTimer  # type: ignore
from PyQt5.QtGui import QIcon  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QMessageBox, QVBoxLayout, QPushButton, QWidget, QLabel
)

from .gui_text_utils import app_icon_path
from .project_manager import ProjectManager, app_state
from .project_view_window import ProjectViewWindow



def _check_if_last_window() -> None:
    """Checks if the project window is closed; if so, quits the app."""
    app = QApplication.instance()
    if not app:
        return
    pv = app_state.project_view_window
    valid = False
    try:
        if pv:
            valid = pv.isVisible()
    except Exception:
        pass
    
    if not valid:
        app.quit()

def _handle_napari_close() -> None:
    """Callback when Napari closes."""
    QTimer.singleShot(100, _check_if_last_window)

def interactive_segmentation_with_config(selected_folder: str = None, project_manager=None) -> None:
    """Launches Napari with the DynamicGUIManager for a single sample."""
    try:
        from .gui_manager import DynamicGUIManager
    except ImportError:
        return

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = None
    
    try:
        if not selected_folder:
            raise ValueError("No folder selected.")
            
        contents = os.listdir(selected_folder)
        tif = next(
            (f for f in contents if f.lower().endswith(('.tif', '.tiff'))), None
        )
        yml = next(
            (f for f in contents if f.lower().endswith(('.yaml', '.yml'))), None
        )
        
        if not tif or not yml:
            raise FileNotFoundError("Missing TIF/YAML files in folder.")

        file_loc = os.path.join(selected_folder, tif)
        with open(os.path.join(selected_folder, yml), 'r') as f:
            config = yaml.safe_load(f)
        mode = config.get('mode')

        image_stack = tiff.memmap(file_loc, mode='r') 
        viewer = napari.Viewer(title=f"Segmentation: {os.path.basename(selected_folder)}")

        qt_window = viewer.window._qt_window
        qt_window.destroyed.connect(_handle_napari_close)

        gui_manager = DynamicGUIManager(viewer, config, image_stack, file_loc, mode,
                                        project_manager=project_manager)
        viewer.window.add_dock_widget(
            create_back_to_project_button(viewer, gui_manager), area="left", name="Navigation"
        )

        @magicgui(call_button="▶ Next Step / Run Current")
        def continue_processing():
            gui_manager.execute_processing_step()

        @magicgui(call_button="◀ Previous Step")
        def go_to_previous_step():
            idx = gui_manager.current_step["value"]
            if idx > 0:
                gui_manager.cleanup_step(idx)
                gui_manager.current_step["value"] -= 1
                gui_manager.create_step_widgets(
                    gui_manager.processing_steps[gui_manager.current_step["value"]]
                )
                update_navigation_buttons()

        def update_navigation_buttons():
            idx = gui_manager.current_step["value"]
            total = len(gui_manager.processing_steps)
            go_to_previous_step.enabled = (idx > 0)
            continue_processing.enabled = (idx < total)
            
            if idx < total:
                step_name = gui_manager.processing_steps[idx]
                display = gui_manager.step_display_names.get(step_name, step_name)
                continue_processing.label = f"Run Step {idx + 1}: {display}"
            else:
                continue_processing.label = "Processing Complete"

        def disable_buttons_during_process():
            continue_processing.enabled = False
            go_to_previous_step.enabled = False
            continue_processing.label = "Processing... (Please Wait)"

        gui_manager.process_started.connect(disable_buttons_during_process)
        gui_manager.process_finished.connect(update_navigation_buttons)

        container = QWidget()
        l = QVBoxLayout()
        container.setLayout(l)
        l.addWidget(continue_processing.native)
        l.addWidget(go_to_previous_step.native)
        l.setContentsMargins(5, 5, 5, 5)
        
        viewer.window.add_dock_widget(
            container, area="left", name="Processing Control"
        )
        update_navigation_buttons()

        # --- ROI / Sub-region panel ---
        roi_container = QWidget()
        roi_layout = QVBoxLayout()
        roi_container.setLayout(roi_layout)
        roi_layout.setContentsMargins(5, 5, 5, 5)
        roi_layout.setSpacing(4)

        roi_header = QLabel("Sub-region (ROI)")
        roi_header.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #aaa; padding-top: 4px;"
        )
        roi_layout.addWidget(roi_header)

        btn_draw = QPushButton("✏  Draw ROI")
        btn_draw.setToolTip(
            "Add a polygon layer to draw a sub-region.\n"
            "Click to add vertices, double-click to close."
        )
        btn_draw.clicked.connect(gui_manager.draw_roi)
        roi_layout.addWidget(btn_draw)

        btn_confirm = QPushButton("✓  Confirm ROI")
        btn_confirm.setToolTip(
            "Crop to the drawn polygon and process only that region.\n"
            "Parameters tuned here can be transferred to the full image\n"
            "via 'Set New Channel Config' in the Project View."
        )
        btn_confirm.clicked.connect(gui_manager.confirm_roi)
        roi_layout.addWidget(btn_confirm)

        btn_clear = QPushButton("✗  Clear ROI")
        btn_clear.setToolTip(
            "Remove the ROI layer.\n"
            "If an ROI session is active, offers to return to full-image mode.\n"
            "Existing ROI outputs are kept on disk."
        )
        btn_clear.clicked.connect(gui_manager.clear_roi)
        roi_layout.addWidget(btn_clear)

        viewer.window.add_dock_widget(
            roi_container, area="left", name="ROI / Sub-region"
        )

    except Exception as e:
        QMessageBox.critical(None, "Error", str(e))
        app_state.show_project_view_signal.emit()

def _apply_app_identity(app: QApplication) -> None:
    """
    Give the running application its name and icon.

    The window/taskbar/dock icon comes from QApplication.setWindowIcon -- NOT
    from the .desktop file -- so this is what makes the icon appear while the
    app is running, on every platform. setDesktopFileName links the running
    window to the installed `hibachi.desktop` entry on Linux (GNOME/Wayland).
    """
    app.setApplicationName("HIBACHI")
    app.setApplicationDisplayName("HIBACHI")
    app.setOrganizationName("HIBACHI")
    try:
        # Qt >= 5.7; ties the running window to hibachi.desktop on Linux.
        app.setDesktopFileName("hibachi")
    except Exception:
        pass
    icon = app_icon_path()
    if icon:
        app.setWindowIcon(QIcon(icon))


def launch_image_segmentation_tool() -> QApplication:
    """Main entry point for the GUI application."""
    app = QApplication.instance() or QApplication(sys.argv)
    _apply_app_identity(app)
    
    # Prevent Napari from attempting to shut down the global app lifecycle 
    # when a viewer closes. This prevents the `_close_app` TypeError on macOS.
    app.setQuitOnLastWindowClosed(False)

    def show_pv():
        if not app_state.project_view_window:
            app_state.project_view_window = ProjectViewWindow(ProjectManager())
        app_state.project_view_window.show()
        app_state.project_view_window.raise_()

    app_state.show_project_view_signal.connect(show_pv)
    show_pv()
    return app

def create_back_to_project_button(viewer: napari.Viewer, gui_manager: Any) -> QWidget:
    """Creates the 'Back to Project List' button widget."""
    def _do():
        nonlocal viewer, gui_manager
        
        # 1. Clean up background tasks and data
        if gui_manager:
            gui_manager.shutdown_and_cleanup()
            gui_manager = None
            
        # 2. Show project view first
        app_state.show_project_view_signal.emit()
        
        # 3. Safely close Napari
        if viewer:
            v = viewer
            viewer = None  # Break Python reference cycle
            
            try:
                # Hide immediately for responsiveness
                if hasattr(v.window, '_qt_window'):
                    v.window._qt_window.hide()
                
                # CRITICAL FIX: Use Napari's native close() instead of qt_win.close().
                # This ensures the internal app_model deregisters its actions properly
                # and prevents the 'in_n_out missing window' TypeError on macOS.
                QTimer.singleShot(50, v.close)
            except Exception:
                pass
            
            # Force GC to collect the Python Napari viewer object
            QTimer.singleShot(200, gc.collect)

    btn = QPushButton("Back to Project List")
    btn.clicked.connect(_do)
    w = QWidget()
    l = QVBoxLayout()
    w.setLayout(l)
    l.addWidget(btn)
    l.setContentsMargins(5, 5, 5, 5)
    return w
