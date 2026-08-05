"""app_launch: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
import sys
import gc
import yaml  # type: ignore
import tifffile as tiff  # type: ignore
import napari  # type: ignore
from magicgui import magicgui  # type: ignore
from typing import Any
from PyQt5.QtCore import QTimer, Qt  # type: ignore
from PyQt5.QtGui import QIcon  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QMessageBox, QVBoxLayout, QHBoxLayout, QPushButton, QWidget,
    QLabel, QFrame, QSizePolicy
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


_shutdown_hook_connected = False


def _stop_running_qthreads_before_teardown() -> None:
    """Stop every still-running QThread before Qt's C++ objects are destroyed.

    This is the fix for the "exited with an error (code -6)" crash. Code -6 is a
    SIGABRT, and the diagnostics traced it to this exact sequence at exit:

        Application closed normally (exit 0)
        QThread: Destroyed while thread is still running   <- Qt fatal -> abort()

    When the app quits while a napari viewer is still open (e.g. the user closes
    the project window rather than clicking "Back to Project"), that viewer's
    background StatusChecker QThread is still running. When the QThread's C++
    object is later destroyed during interpreter teardown while it is still
    running, Qt calls abort(). We pre-empt that here, in aboutToQuit -- which
    fires the instant app.quit() is called, while every object is still alive --
    by stopping each running QThread so nothing is running when it is destroyed.

    napari parents its StatusChecker to the viewer's main window (not to the
    QApplication), so we search under every top-level window, not just the app.
    """
    from PyQt5.QtCore import QThread  # local import: keep module top light

    app = QApplication.instance()
    if app is None:
        return

    threads = set()
    try:
        for w in app.topLevelWidgets():
            try:
                threads.update(w.findChildren(QThread))
            except Exception:
                pass
    except Exception:
        return

    for th in threads:
        try:
            if not th.isRunning():
                continue
        except Exception:
            continue

        lifecycle("shutdown.qthread.stop", thread=type(th).__name__)

        # 1) Cooperative stop. napari's StatusChecker loops until interruption is
        #    requested but blocks on a Python Event, so requestInterruption()
        #    alone won't wake it -- we also set that Event. Older variants use a
        #    private _terminate flag; set it too. All best-effort.
        try:
            th.requestInterruption()
        except Exception:
            pass
        ev = getattr(th, "_need_status_update", None)
        if ev is not None and hasattr(ev, "set"):
            try:
                ev.set()
            except Exception:
                pass
        if hasattr(th, "_terminate"):
            try:
                th._terminate = True
            except Exception:
                pass
        try:
            th.quit()  # no-op for non-event-loop threads, harmless
        except Exception:
            pass

        # 2) Wait for a clean exit.
        try:
            if th.wait(1500):
                continue
        except Exception:
            pass

        # 3) Last resort: hard-terminate so the C++ destructor won't see a
        #    running thread and abort(). Acceptable only because we are exiting.
        try:
            log.warning("Force-terminating unresponsive %s thread at shutdown.",
                        type(th).__name__)
            th.terminate()
            th.wait(500)
        except Exception:
            pass


def _install_shutdown_hook(app) -> None:
    """Connect the QThread-stopping cleanup to aboutToQuit (exactly once)."""
    global _shutdown_hook_connected
    if _shutdown_hook_connected or app is None:
        return
    try:
        app.aboutToQuit.connect(_stop_running_qthreads_before_teardown)
        _shutdown_hook_connected = True
    except Exception:
        pass


def _has_open_napari_viewer() -> bool:
    """True if any napari viewer window is currently open/visible.

    napari's main window is a top-level widget whose class lives in a napari
    module, so we look for a visible top-level widget from the napari package.
    """
    app = QApplication.instance()
    if app is None:
        return False
    try:
        for w in app.topLevelWidgets():
            try:
                if not w.isVisible():
                    continue
                if "napari" in (type(w).__module__ or ""):
                    return True
            except Exception:
                pass
    except Exception:
        pass
    return False


def _check_if_last_window() -> None:
    """Quit the app only when the project window is gone AND no viewer is open.

    This runs 100 ms after any napari window is destroyed (see
    _handle_napari_close). It must NOT quit just because the project window
    isn't the visible window at that instant: when the user closes one sample
    and opens another, the freshly-opened napari viewer is what's visible and
    the project window is not -- quitting there would tear the app down right as
    the user opens a sample (the "code -6 on open" crash). So we also require
    that no napari viewer is open before quitting.
    """
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

    if valid:
        return  # project window is up; nothing to do

    if _has_open_napari_viewer():
        # A viewer is open (e.g. the user just opened a sample). Do not quit.
        lifecycle("app.quit.skipped", reason="napari viewer still open")
        return

    lifecycle("app.quit", reason="no project window and no open viewer")
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


# --------------------------------------------------------------------------- #
# Compact left-panel building blocks
#
# The left column is shared with napari's own layer-controls and layer-list
# docks. Every dock we add there competes with the layer list for vertical
# space, and each one costs a title bar (~24 px) plus its own margins on top of
# its actual content. So: one dock, compact rows, and a hard height cap so the
# surplus height goes to the layer list instead of to empty panel padding.
# --------------------------------------------------------------------------- #

_PANEL_BTN_HEIGHT = 26        # standard row height inside the control panel
_PANEL_PRIMARY_HEIGHT = 30    # slightly taller for the primary Process action
_LAYER_LIST_MIN_HEIGHT = 160  # floor for napari's layer list, in pixels


def _compactify(btn: QPushButton, height: int = _PANEL_BTN_HEIGHT) -> QPushButton:
    """Force a button to a fixed, compact row height.

    Fixed *height* (not width) is deliberate: it makes the panel's total height
    deterministic, which is what lets _lock_panel_height cap the dock exactly.
    """
    btn.setFixedHeight(height)
    btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return btn


def _compact_button(text: str, tooltip: str = "",
                    height: int = _PANEL_BTN_HEIGHT) -> QPushButton:
    """A panel button: compact fixed height, full tooltip retained."""
    btn = QPushButton(text)
    if tooltip:
        btn.setToolTip(tooltip)
    return _compactify(btn, height)


def _hline() -> QFrame:
    """A 1 px separator used instead of a second dock title bar."""
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    line.setFixedHeight(1)
    line.setStyleSheet("color: #3a3a3a; background-color: #3a3a3a; border: none;")
    return line


def _section_header(text: str) -> QWidget:
    """A small caption + hairline: the visual grouping a dock title used to give
    us, for about a fifth of the vertical cost."""
    holder = QWidget()
    lay = QVBoxLayout(holder)
    lay.setContentsMargins(0, 3, 0, 1)
    lay.setSpacing(1)

    label = QLabel(text.upper())
    font = label.font()
    font.setPointSize(max(7, font.pointSize() - 2))
    font.setBold(True)
    label.setFont(font)
    label.setStyleSheet("color: #8a8a8a; letter-spacing: 1px;")
    lay.addWidget(label)
    lay.addWidget(_hline())

    holder.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
    return holder


def _lock_panel_height(container: QWidget, dock=None, slack: int = 6) -> None:
    """Cap the panel (and its dock) at exactly the height of its contents.

    Without this, QMainWindow is free to hand spare vertical space to our dock,
    which then sits as dead padding under the buttons while the layer list stays
    squeezed. Capping the dock's maximum height forces that surplus to the only
    remaining expandable dock in the column -- the layer list.

    Applied immediately and again on the next two event-loop turns, because the
    real title-bar chrome height is only measurable once the window is shown.
    """
    def _apply(measure_chrome: bool) -> None:
        try:
            layout = container.layout()
            if layout is not None:
                layout.activate()
            content_h = container.sizeHint().height() + slack
            container.setMaximumHeight(content_h)
            if dock is not None:
                chrome = 30  # fallback: typical dock title bar + frame
                if measure_chrome:
                    measured = dock.height() - container.height()
                    if 0 < measured <= 80:
                        chrome = measured
                dock.setMaximumHeight(content_h + chrome)
        except Exception:
            log.debug("could not lock control panel height", exc_info=True)

    _apply(measure_chrome=False)
    QTimer.singleShot(0, lambda: _apply(True))
    QTimer.singleShot(300, lambda: _apply(True))


def _give_layer_list_room(viewer, panel_dock=None) -> None:
    """Give napari's layer list a floor on its height, and dock our panel
    directly beneath it so the grouping reads top-to-bottom."""
    layer_list = _layer_list_dock(viewer)
    if layer_list is None:
        return
    try:
        # Only enforce the floor when the screen can actually afford it, so this
        # never makes the viewer unusable on a small laptop display.
        screen_h = QApplication.primaryScreen().availableGeometry().height()
        if screen_h > 700:
            layer_list.setMinimumHeight(_LAYER_LIST_MIN_HEIGHT)
    except Exception:
        pass

    if panel_dock is not None:
        try:
            viewer.window._qt_window.splitDockWidget(
                layer_list, panel_dock, Qt.Vertical
            )
        except Exception:
            log.debug("could not place control panel under the layer list",
                      exc_info=True)


def make_channel_visibility_button(viewer) -> QPushButton:
    """Build (and fully wire) the all-channels hide/show toggle button.

    Returned bare so it can either get its own dock (add_channel_visibility_toggle,
    used by the cross-channel viewer) or be folded into a larger panel.
    """
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

    return btn


def add_channel_visibility_toggle(viewer):
    """Add one button, docked immediately under the channel (layer) list, that
    toggles every layer between all-hidden and all-shown. The label reflects the
    action it will perform next.

    Kept as a standalone dock for viewers that have no other controls (the
    cross-channel viewer). The segmentation viewer folds the same button into its
    single merged control panel instead -- see build_segmentation_control_panel.
    """
    btn = _compactify(make_channel_visibility_button(viewer))

    container = QWidget()
    lay = QVBoxLayout(container)
    lay.setContentsMargins(5, 3, 5, 3)
    lay.addWidget(btn)

    dock = viewer.window.add_dock_widget(container, area="left", name="Channels")
    # Cap it and move it to sit directly below the layer list.
    _lock_panel_height(container, dock)
    _give_layer_list_room(viewer, dock)
    return dock


def build_segmentation_control_panel(viewer, gui_manager):
    """Build the single merged left-hand control panel for the segmentation viewer.

    Replaces what used to be four separate left docks (Navigation, Channels,
    Processing Control, ROI / Sub-region -- four title bars and eight stacked
    full-width rows) with one dock, section hairlines, and paired button rows.

    Returns (container, refresh_nav). The caller docks the container and is free
    to call refresh_nav(); all signal wiring is already done here.
    """
    container = QWidget()
    outer = QVBoxLayout(container)
    outer.setContentsMargins(6, 4, 6, 6)
    outer.setSpacing(4)

    # ---- Processing ------------------------------------------------------- #
    # Navigation (Back / Forward) is non-destructive: it never deletes results.
    # Computing is a separate, explicit action (Process), which is the only thing
    # that clears downstream results. Forward is allowed only into already-
    # processed steps and disables at the first unprocessed one; all the
    # enable/frontier logic lives in the gui_manager.
    #
    # No "Processing" section header here: it is the first thing in the panel and
    # the status line below already names the step, so the header would only cost
    # height.

    # The step name lives here rather than in the button label. It used to be
    # interpolated into the button text, which made the button demand enough
    # width for "Re-process Step 4: Calculate Features (clears later results)"
    # and dragged the whole left column wide. Two lines are reserved so the
    # panel's height never changes as the text does.
    status = QLabel("")
    status.setWordWrap(True)
    status.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    _sf = status.font()
    _sf.setPointSize(max(7, _sf.pointSize() - 1))
    status.setFont(_sf)
    status.setStyleSheet("color: #a8a8a8;")
    status.setFixedHeight(status.fontMetrics().height() * 2 + 4)
    outer.addWidget(status)

    btn_process = _compact_button(
        "\u2699 Process Current Step",
        "Process this step with the current parameters.\n"
        "Re-processing a step clears the results of every step after it.",
        height=_PANEL_PRIMARY_HEIGHT,
    )
    outer.addWidget(btn_process)

    # Back and Forward are a natural pair -> one row instead of two.
    nav_row = QHBoxLayout()
    nav_row.setContentsMargins(0, 0, 0, 0)
    nav_row.setSpacing(4)
    btn_back = _compact_button(
        "\u25c0 Back",
        "Go to the previous step. Non-destructive \u2014 results are kept.\n"
        "If you changed parameters here, you'll be asked to discard them or\n"
        "process this step first.",
    )
    btn_forward = _compact_button(
        "Forward \u25b6",
        "Go to the next step (no computation). Enabled only for steps that are\n"
        "already processed; disabled once you reach the first unprocessed step.",
    )
    nav_row.addWidget(btn_back)
    nav_row.addWidget(btn_forward)
    outer.addLayout(nav_row)

    # ---- Sub-region (ROI) ------------------------------------------------- #
    # The section header carries the name, so the old redundant in-panel
    # "Sub-region (ROI)" QLabel is gone, and the three buttons share one row.
    outer.addWidget(_section_header("Sub-region (ROI)"))

    roi_row = QHBoxLayout()
    roi_row.setContentsMargins(0, 0, 0, 0)
    roi_row.setSpacing(4)
    btn_draw = _compact_button(
        "\u270f Draw",
        "Add a polygon layer to draw a sub-region.\n"
        "Click to add vertices, double-click to close.",
    )
    btn_confirm = _compact_button(
        "\u2713 Apply",
        "Crop to the drawn polygon and process only that region.\n"
        "Parameters tuned here can be transferred to the full image\n"
        "via 'Set New Channel Config' in the Project View.",
    )
    btn_clear = _compact_button(
        "\u2717 Clear",
        "Remove the ROI layer.\n"
        "If an ROI session is active, offers to return to full-image mode.\n"
        "Existing ROI outputs are kept on disk.",
    )
    btn_draw.clicked.connect(gui_manager.draw_roi)
    btn_confirm.clicked.connect(gui_manager.confirm_roi)
    btn_clear.clicked.connect(gui_manager.clear_roi)
    for _b in (btn_draw, btn_confirm, btn_clear):
        roi_row.addWidget(_b)
    outer.addLayout(roi_row)

    # ---- Channels --------------------------------------------------------- #
    # A hairline rather than a section header: the button labels itself.
    outer.addWidget(_hline())
    outer.addWidget(_compactify(make_channel_visibility_button(viewer)))

    # ---- Leave ------------------------------------------------------------ #
    outer.addWidget(_hline())
    outer.addWidget(
        _compactify(make_back_to_project_button(viewer, gui_manager))
    )

    # ---- Behaviour (unchanged semantics) ---------------------------------- #
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
                btn_process.setText(f"\u2699 Re-process Step {idx + 1}")
                status.setText(
                    f"Step {idx + 1}/{total} \u00b7 {display}\n"
                    "Re-processing clears later results."
                )
            else:
                btn_process.setText(f"\u2699 Process Step {idx + 1}")
                status.setText(f"Step {idx + 1}/{total} \u00b7 {display}")
        elif gui_manager.valid_frontier() >= total:
            btn_process.setText("\u2713 All Steps Complete")
            status.setText(f"All {total} steps complete.")
        else:
            btn_process.setText("\u2699 Process Current Step")
            status.setText(f"Step {idx + 1}/{total}")

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
        btn_process.setText("Processing\u2026")
        status.setText("Processing \u2014 please wait.")

    gui_manager.process_started.connect(disable_buttons_during_process)
    gui_manager.process_finished.connect(refresh_nav)
    # Editing a parameter can change the frontier (this step becomes dirty),
    # so re-evaluate the buttons whenever a value changes.
    gui_manager.params_edited.connect(refresh_nav)

    return container, refresh_nav


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
        control_panel, refresh_nav = build_segmentation_control_panel(
            viewer, gui_manager
        )
        control_dock = viewer.window.add_dock_widget(
            control_panel, area="left", name="Controls"
        )
        # Cap the panel at its content height and sit it under the layer list, so
        # the spare vertical space in the left column goes to the layer list.
        _lock_panel_height(control_panel, control_dock)
        _give_layer_list_room(viewer, control_dock)
        refresh_nav()

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

    # Stop any still-running QThreads (notably napari's StatusChecker) the moment
    # we quit, so they aren't destroyed while running -> prevents the SIGABRT
    # (code -6) crash on exit. See _stop_running_qthreads_before_teardown.
    _install_shutdown_hook(app)

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

def make_back_to_project_button(viewer: napari.Viewer, gui_manager: Any) -> QPushButton:
    """Creates the bare 'Back to Project List' button, fully wired.

    Returned without its own container/dock so it can be folded into the merged
    control panel. create_back_to_project_button still wraps it in a QWidget for
    any caller that wants a standalone dockable widget.
    """
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

    btn = QPushButton("\u2190 Back to Project List")
    btn.clicked.connect(_do)
    return btn


def create_back_to_project_button(viewer: napari.Viewer, gui_manager: Any) -> QWidget:
    """Standalone dockable wrapper around make_back_to_project_button.

    Retained for backwards compatibility -- it is re-exported from helper_funcs.
    The segmentation viewer no longer uses it; it embeds the bare button in the
    merged control panel instead.
    """
    w = QWidget()
    l = QVBoxLayout()
    w.setLayout(l)
    l.addWidget(_compactify(make_back_to_project_button(viewer, gui_manager)))
    l.setContentsMargins(5, 5, 5, 5)
    return w