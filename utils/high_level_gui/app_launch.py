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

# Diagnostics. Best-effort: if logging_setup is unavailable for any reason we
# fall back to a plain logger so this module still imports and runs.
try:
    from .logging_setup import get_logger, lifecycle
    log = get_logger("app_launch")
except Exception:  # pragma: no cover
    import logging
    log = logging.getLogger("hibachi.app_launch")

    def lifecycle(event, **fields):
        log.info("%s %s", event, " ".join(f"{k}={v!r}" for k, v in fields.items()))



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
        lifecycle("app.quit", reason="project window closed")
        app.quit()

def _layer_list_dock(viewer):
    """Locate napari's layer-list dock widget across versions, so we can place
    the toggle directly beneath it. Returns the QDockWidget or None."""
    for accessor in (lambda: viewer.window._qt_viewer.dockLayerList,
                     lambda: viewer.window.qt_viewer.dockLayerList):
        try:
            d = accessor()
            if d is not None:
                return d
        except Exception:
            pass
    try:
        from PyQt5.QtWidgets import QDockWidget
        for d in viewer.window._qt_window.findChildren(QDockWidget):
            if "layer list" in d.windowTitle().lower():
                return d
    except Exception:
        pass
    return None


def add_channel_visibility_toggle(viewer):
    """Add one button, docked immediately under the channel (layer) list, that
    toggles every layer between all-hidden and all-shown. The label reflects the
    action it will perform next."""
    btn = QPushButton()
    btn.setToolTip("Hide or show every channel at once, instead of toggling each "
                   "layer's eye icon individually.")

    def _any_visible():
        return any(getattr(l, "visible", False) for l in viewer.layers)

    def _refresh():
        btn.setText("🙈 Hide All Channels" if _any_visible()
                    else "👁️ Show All Channels")

    def _toggle():
        hide = _any_visible()          # something is visible -> hide everything
        for layer in list(viewer.layers):
            try:
                layer.visible = not hide
            except Exception:
                pass
        _refresh()

    btn.clicked.connect(_toggle)
    _refresh()
    # Keep the label correct as layers are added/removed (e.g. after processing).
    try:
        viewer.layers.events.inserted.connect(lambda e: _refresh())
        viewer.layers.events.removed.connect(lambda e: _refresh())
    except Exception:
        pass

    container = QWidget()
    lay = QVBoxLayout(container)
    lay.setContentsMargins(5, 3, 5, 3)
    lay.addWidget(btn)

    dock = viewer.window.add_dock_widget(container, area="left", name="Channels")
    # Move it to sit directly below the layer list.
    ll = _layer_list_dock(viewer)
    if ll is not None:
        try:
            from PyQt5.QtCore import Qt
            viewer.window._qt_window.splitDockWidget(ll, dock, Qt.Vertical)
        except Exception:
            pass
    return dock


def _handle_napari_close() -> None:
    """Callback when Napari closes."""
    lifecycle("napari.destroyed", note="qt window destroyed signal fired")
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
        lifecycle("viewer.open",
                  folder=os.path.basename(selected_folder),
                  mode=mode,
                  shape=getattr(image_stack, "shape", None),
                  dtype=str(getattr(image_stack, "dtype", "?")))
        viewer = napari.Viewer(title=f"Segmentation: {os.path.basename(selected_folder)}")

        qt_window = viewer.window._qt_window
        qt_window.destroyed.connect(_handle_napari_close)

        # Open maximized and bring to the foreground so the user doesn't have to
        # click the app icon to surface it.
        try:
            qt_window.showMaximized()
            qt_window.raise_()
            qt_window.activateWindow()
        except Exception:
            pass

        gui_manager = DynamicGUIManager(viewer, config, image_stack, file_loc, mode,
                                        project_manager=project_manager)
        viewer.window.add_dock_widget(
            create_back_to_project_button(viewer, gui_manager), area="left", name="Navigation"
        )

        # One-click toggle to hide/show every channel, docked under the layer list.
        add_channel_visibility_toggle(viewer)

        # --- Processing control -------------------------------------------
        # Navigation (Back / Forward) is non-destructive: it never deletes
        # results. Computing is a separate, explicit action (Process), which is
        # the only thing that clears downstream results. Forward is allowed only
        # into already-processed steps and disables at the first unprocessed one;
        # all the enable/frontier logic lives in the gui_manager.
        btn_process = QPushButton("\u2699 Process Current Step")
        btn_process.setToolTip(
            "Process this step with the current parameters.\n"
            "Re-processing a step clears the results of every step after it."
        )
        btn_forward = QPushButton("Forward \u25b6")
        btn_forward.setToolTip(
            "Go to the next step (no computation). Enabled only for steps that are\n"
            "already processed; disabled once you reach the first unprocessed step."
        )
        btn_back = QPushButton("\u25c0 Back")
        btn_back.setToolTip(
            "Go to the previous step. Non-destructive \u2014 results are kept.\n"
            "If you changed parameters here, you'll be asked to discard them or\n"
            "process this step first."
        )

        def refresh_nav():
            # While a step is processing, leave the buttons as the process-start
            # handler set them; this also avoids a race when "Process now" is
            # chosen from the Back prompt (which starts processing synchronously).
            worker = getattr(gui_manager, "worker", None)
            if worker is not None and worker.isRunning():
                return
            idx = gui_manager.current_step["value"]
            total = gui_manager.num_steps
            btn_back.setEnabled(gui_manager.can_go_back())
            btn_forward.setEnabled(gui_manager.can_go_forward())
            can_proc = gui_manager.can_process()
            btn_process.setEnabled(can_proc)
            if can_proc and idx < total:
                step_name = gui_manager.processing_steps[idx]
                display = gui_manager.step_display_names.get(step_name, step_name)
                if gui_manager.is_current_step_dirty():
                    btn_process.setText(
                        f"\u2699 Re-process Step {idx + 1}: {display}  (clears later results)"
                    )
                else:
                    btn_process.setText(f"\u2699 Process Step {idx + 1}: {display}")
            elif gui_manager.valid_frontier() >= total:
                btn_process.setText("\u2713 All Steps Complete")
            else:
                btn_process.setText("\u2699 Process Current Step")

        def on_back():
            gui_manager.go_back()
            refresh_nav()

        def on_forward():
            gui_manager.go_forward()
            refresh_nav()

        def on_process():
            # Asynchronous: process_started disables the buttons, process_finished
            # re-runs refresh_nav once the step (and any advance) has completed.
            gui_manager.execute_processing_step()

        btn_back.clicked.connect(on_back)
        btn_forward.clicked.connect(on_forward)
        btn_process.clicked.connect(on_process)

        def disable_buttons_during_process():
            btn_back.setEnabled(False)
            btn_forward.setEnabled(False)
            btn_process.setEnabled(False)
            btn_process.setText("Processing\u2026 (please wait)")

        gui_manager.process_started.connect(disable_buttons_during_process)
        gui_manager.process_finished.connect(refresh_nav)
        # Editing a parameter can change the frontier (this step becomes dirty),
        # so re-evaluate the buttons whenever a value changes.
        gui_manager.params_edited.connect(refresh_nav)

        container = QWidget()
        l = QVBoxLayout()
        container.setLayout(l)
        l.addWidget(btn_process)
        l.addWidget(btn_forward)
        l.addWidget(btn_back)
        l.setContentsMargins(5, 5, 5, 5)

        viewer.window.add_dock_widget(
            container, area="left", name="Processing Control"
        )
        refresh_nav()

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
        log.exception("Failed to open segmentation viewer for %r", selected_folder)
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
        pv = app_state.project_view_window
        pv.showNormal()          # de-minimise if needed
        pv.show()
        pv.raise_()
        pv.activateWindow()

    app_state.show_project_view_signal.connect(show_pv)
    show_pv()
    return app

def create_back_to_project_button(viewer: napari.Viewer, gui_manager: Any) -> QWidget:
    """Creates the 'Back to Project List' button widget."""
    def _do():
        nonlocal viewer, gui_manager

        # Each numbered step below is logged so that if the process aborts
        # (SIGABRT / "code -6") during teardown, the last breadcrumb in the log
        # tells us exactly which step it died on -- the faulthandler dump then
        # names the thread/line.
        lifecycle("back_to_project.click")

        # 1. Clean up background tasks and data
        if gui_manager:
            lifecycle("cleanup.begin")
            gui_manager.shutdown_and_cleanup()
            gui_manager = None
            lifecycle("cleanup.end")

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
                lifecycle("napari.close.scheduled", delay_ms=50)
                QTimer.singleShot(50, lambda: (lifecycle("napari.close.invoke"), v.close()))
            except Exception:
                log.exception("Error while closing napari viewer")

            # Force GC to collect the Python Napari viewer object
            QTimer.singleShot(200, lambda: (lifecycle("gc.collect.post_close"), gc.collect()))

    btn = QPushButton("Back to Project List")
    btn.clicked.connect(_do)
    w = QWidget()
    l = QVBoxLayout()
    w.setLayout(l)
    l.addWidget(btn)
    l.setContentsMargins(5, 5, 5, 5)
    return w