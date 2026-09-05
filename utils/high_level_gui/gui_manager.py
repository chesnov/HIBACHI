import os
import sys
import gc
import copy
import json
import shutil
import time
import traceback
import yaml  # type: ignore
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
from skimage.draw import polygon as skimage_polygon  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QMessageBox, QWidget, QVBoxLayout, QScrollArea, QLabel,
    QTextEdit, QProgressBar, QApplication, QPushButton, QFileDialog, QDockWidget, QLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer  # type: ignore
from PyQt5.QtGui import QTextCursor  # type: ignore
import napari  # type: ignore

# --- Relative Imports ---
try:
    from ..fluorescence_module.fluorescence_strategy import FluorescenceStrategy
    from ..fluorescence_module.config_migration import (
        UNIFIED_MODE, config_basename, find_config_path, find_processed_dir,
        normalise_mode)
    from .processing_strategies import ProcessingStrategy
    from .helper_funcs import create_parameter_widget
except ImportError as e:
    print(f"Error importing dependencies in gui_manager.py: {e}")
    raise

# Diagnostics. Best-effort with a plain-logger fallback so this module always
# imports even if logging_setup is somehow absent.
try:
    from .logging_setup import get_logger, lifecycle
    log = get_logger("gui_manager")
except Exception:  # pragma: no cover
    import logging as _logging
    log = _logging.getLogger("hibachi.gui_manager")

    def lifecycle(event, **fields):
        log.info("%s %s", event, " ".join(f"{k}={v!r}" for k, v in fields.items()))


# =============================================================================
# 0. Processing-mode registry
# =============================================================================
# Single source of truth for which modes this build can open. It lives at module
# level (rather than inline in DynamicGUIManager.__init__) so callers can check a
# mode BEFORE building a napari viewer -- constructing the viewer first and
# discovering the mode is unsupported afterwards leaked a live GL context on
# every attempt.

#: mode string -> strategy class. The one place this mapping is defined; both
#: DynamicGUIManager and the pre-viewer validation in app_launch read it.
STRATEGY_CLASSES: Dict[str, Any] = {
    UNIFIED_MODE: FluorescenceStrategy,
}

SUPPORTED_MODES: Tuple[str, ...] = tuple(STRATEGY_CLASSES)

#: Modes that earlier versions wrote into configs and this build no longer has.
#: Both former ramified modes are the fluorescence pipeline under an old name,
#: so both point at the single surviving mode.
RETIRED_MODES: Dict[str, str] = {
    'ramified': UNIFIED_MODE,
    'ramified_2d': UNIFIED_MODE,
}


def strategy_for(mode: Any):
    """
    Strategy class for `mode`, or None if this build cannot open it.

    Every lookup goes through `normalise_mode` because saved projects still
    carry 'fluorescence_2d' (and older ones 'ramified'), and the registry now
    has one entry. Looking up the raw string would return None for those
    projects and report them as unsupported -- the strategy is the same class
    either way, since rank comes from the data.
    """
    return STRATEGY_CLASSES.get(normalise_mode(mode))


def is_supported_mode(mode: Any) -> bool:
    """True if this build can open a config with this mode."""
    return strategy_for(mode) is not None


def unsupported_mode_message(mode: Any, folder: str = "") -> str:
    """Explain an unopenable mode and how to fix it.

    A config carrying a retired mode is not corrupt -- it was written by an older
    HIBACHI whose pipeline has since been replaced. The fix is to apply a current
    config, which preserves the image's dimensions, so the message says so
    rather than leaving the user with a dead project.
    """
    mode_str = str(mode) if mode else "(none)"
    where = f"\n\nFolder:\n{folder}" if folder else ""

    if mode_str in RETIRED_MODES:
        replacement = RETIRED_MODES[mode_str]
        return (
            f"This image is configured for '{mode_str}', which was removed from "
            "HIBACHI and is no longer available.\n\n"
            "To use it again, select it in the project window and choose "
            f"'Set New Channel Config', picking a '{replacement}' config. Your "
            "image dimensions are preserved; only the processing parameters "
            f"change.{where}"
        )

    supported = ", ".join(SUPPORTED_MODES)
    return (
        f"This image's config specifies mode '{mode_str}', which this version of "
        f"HIBACHI cannot open.\n\nSupported modes are: {supported}.\n\n"
        "Use 'Set New Channel Config' in the project window to apply a current "
        f"config (your image dimensions are preserved).{where}"
    )


class UnsupportedModeError(ValueError):
    """A config's mode has no strategy in this build. Carries a usable message."""

    def __init__(self, mode: Any, folder: str = ""):
        self.mode = mode
        super().__init__(unsupported_mode_message(mode, folder))


# =============================================================================
# 1. Output Redirector
# =============================================================================

class OutputStream(QObject):
    """
    Redirects stdout/stderr to a Qt Signal for display in the GUI log widget.
    """
    text_written = pyqtSignal(str)

    def write(self, text: str) -> None:
        self.text_written.emit(str(text))

    def flush(self) -> None:
        pass


# =============================================================================
# 2. Background Worker Thread
# =============================================================================

import atexit

_orphan_threads = []
_quit_hook_connected = False

def _cleanup_all_orphans():
    """Forcefully terminate any lingering background threads on app exit to prevent C++ aborts."""
    pending = [w for w in _orphan_threads if w is not None]
    if pending:
        lifecycle("orphan_threads.cleanup", count=len(pending))
    for worker in list(_orphan_threads):
        try:
            if worker is not None and worker.isRunning():
                log.warning("Terminating still-running orphan worker thread at exit: %r", worker)
                worker.terminate()
                worker.wait(200) # Give it time to cleanly exit C++ scope
        except Exception:
            log.exception("Error terminating orphan thread")
    _orphan_threads.clear()

def _register_quit_hook():
    global _quit_hook_connected
    if not _quit_hook_connected:
        app = QApplication.instance()
        if app:
            app.aboutToQuit.connect(_cleanup_all_orphans)
        atexit.register(_cleanup_all_orphans)
        _quit_hook_connected = True

def _cleanup_orphan_thread(worker):
    """Safely cleans up an orphaned worker thread after it finishes."""
    try:
        if _orphan_threads is not None and worker in _orphan_threads:
            _orphan_threads.remove(worker)
        if worker is not None:
            worker.deleteLater()
    except Exception:
        pass


def _snapshot_child_pids() -> set:
    """PIDs of every child process this (GUI) process currently owns.

    Captured just before a step starts so we can tell that step's newly-spawned
    workers apart from anything that was already running."""
    try:
        import psutil  # already a project dependency
        return {c.pid for c in psutil.Process().children(recursive=True)}
    except Exception:
        return set()


def _terminate_new_children(baseline_pids: set) -> None:
    """Kill every worker process spawned since ``baseline_pids`` was taken.

    Interactive processing runs in a worker *thread*, but the heavy lifting is
    fanned out to multiprocessing.Pool workers that are child processes of the
    GUI. Killing the thread mid-call is unsafe (it can strand the GIL), so when
    the user leaves the image we instead kill the *processes* it spawned — that
    is what actually stops the computation. Anything already alive at baseline
    (i.e. not part of this run) is left untouched. Mirrors the batch path, which
    likewise cancels by killing processes rather than threads."""
    try:
        import psutil
        me = psutil.Process()
    except Exception:
        return

    victims = []
    try:
        for c in me.children(recursive=True):
            if c.pid not in baseline_pids:
                victims.append(c)
    except Exception:
        return
    if not victims:
        return

    for c in victims:
        try:
            c.terminate()  # SIGTERM first for a clean shutdown
        except Exception:
            pass
    try:
        _, alive = psutil.wait_procs(victims, timeout=1.5)
    except Exception:
        alive = victims
    for c in alive:
        try:
            c.kill()  # hard-kill anything that ignored SIGTERM
        except Exception:
            pass


class StepWorker(QThread):
    """
    Executes a processing step in a separate thread to keep the GUI responsive.
    """
    finished_signal = pyqtSignal(bool)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        strategy: ProcessingStrategy,
        step_index: int,
        image_stack: Optional[np.ndarray],
        params: Dict[str, Any]
    ):
        super().__init__()
        self.strategy = strategy
        self.step_index = step_index
        self.image_stack = image_stack
        self.params = params

    def run(self) -> None:
        lifecycle("worker.run.start", step=self.step_index)
        try:
            success = self.strategy.execute_step(
                step_index=self.step_index,
                viewer=None,  # Viewer is handled by main thread, not worker
                image_stack_or_none=self.image_stack,
                params=self.params
            )
            lifecycle("worker.run.finish", step=self.step_index, success=bool(success))
            self.finished_signal.emit(success)
        except Exception as e:
            log.exception("Worker step %s failed", self.step_index)
            self.error_signal.emit(str(e))
            self.finished_signal.emit(False)


# =============================================================================
# 3. Main GUI Manager
# =============================================================================

class DynamicGUIManager(QObject):
    """
    Manages the Napari GUI state, step navigation, and widget generation.

    It acts as the Controller between the View (Napari/Qt) and the Model
    (ProcessingStrategy). It dynamically builds parameter widgets based on
    the YAML configuration of the current strategy.
    """
    process_started = pyqtSignal()
    process_finished = pyqtSignal()
    # Emitted when a parameter widget value changes, so the navigation buttons
    # (Back / Forward / Process) can re-evaluate the "valid frontier".
    params_edited = pyqtSignal()

    def __init__(
        self,
        viewer: napari.Viewer,
        config: Dict[str, Any],
        image_stack: np.ndarray,
        file_loc: str,
        processing_mode: str,
        roi_name: Optional[str] = None,
        project_manager: Any = None,
    ):
        super().__init__()
        self.viewer = viewer
        # Migrated once, here, before anything reads it. The widgets are built
        # from `self.config` by step key, and a saved project still carries
        # `execute_<step>_fluorescence_2d`; `get_config_key` looks for the
        # unified-suffixed then the bare spelling and finds neither, so the
        # step rendered with NO parameter fields and could not be edited or
        # re-processed. `initial_config` is migrated too, or the dirty check
        # would compare two different schemas and read as permanently dirty.
        config = self._migrate_config(config)
        self.initial_config = config.copy()
        self.config = config.copy()
        self.image_stack = image_stack
        self.file_loc = file_loc
        self.processing_mode = processing_mode
        self.project_manager = project_manager

        # UI State
        self.current_widgets: Dict[QDockWidget, QScrollArea] = {}
        self.current_step = {"value": 0}
        self.parameter_values: Dict[str, Any] = {}
        # Baseline for the currently-shown step: the parameter values the widgets
        # were built from (i.e. the last-committed / last-run values). Compared
        # against live values to tell whether the current step has unsaved edits.
        self._active_config_key: Optional[str] = None
        self._active_baseline: Dict[str, Any] = {}
        self.worker: Optional[StepWorker] = None
        # Snapshot of child PIDs taken right before each step starts, so leaving
        # the image can kill exactly the worker processes this run spawned.
        self._worker_child_baseline: set = set()

        # ROI / sub-region state
        # These are set when the user confirms a polygon crop.
        # _full_* refs allow returning to the full-image session at any time.
        self.roi_active: bool = False
        # Which named ROI session is loaded. None means the full image. Auto-named
        # ("ROI 1", "ROI 2", ...) so several regions can coexist per channel.
        self.active_roi_name: Optional[str] = None
        self._full_contrast_limits: Optional[list] = None
        self._full_image_stack: Optional[np.ndarray] = None
        self._full_processed_dir: Optional[str] = None
        self._full_config: Optional[Dict[str, Any]] = None
        
        # Console Redirection
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.output_stream: Optional[OutputStream] = None
        
        # Initialize Persistent Log Widget
        self.log_widget: Optional[QTextEdit] = None
        self._init_persistent_log()

        # Project Paths
        self.inputdir = os.path.dirname(self.file_loc)
        basename = os.path.basename(self.file_loc)
        self.basename = os.path.splitext(basename)[0]
        # Prefers `<basename>_processed_fluorescence`, but uses an existing
        # legacy directory when one is there: a project processed before the
        # modes merged has its results under `..._fluorescence_2d`, and building
        # only the new name would orphan them.
        self.processed_dir = find_processed_dir(
            self.inputdir, self.basename,
            log=lambda m: print(m))

        # Spacing
        self.spacing: Union[Tuple[float, float, float], Tuple[float, float]] = (1.0, 1.0, 1.0)
        self.z_scale_factor: float = 1.0
        self._calculate_spacing()

        # Initialize Strategy
        try:
            strategy_class = strategy_for(self.processing_mode)

            if not strategy_class:
                # Callers should validate with is_supported_mode() *before*
                # constructing a viewer -- reaching here means a viewer already
                # exists and its caller must close it (see app_launch).
                raise UnsupportedModeError(self.processing_mode, self.inputdir)

            self.strategy = strategy_class(
                self.config,
                self.processed_dir,
                self.image_stack.shape,
                self.spacing,
                self.z_scale_factor
            )

            self.processing_steps = self.strategy.get_step_names()
            self.num_steps = self.strategy.num_steps
            
            # Prettify step names for display
            self.step_display_names = {
                name: name.replace('execute_', '').replace('_', ' ').title()
                for name in self.processing_steps
            }
            print(f"Initialized strategy '{self.processing_mode}' with "
                  f"{self.num_steps} steps.")

        except Exception as e:
            print(f"FATAL ERROR: {e}")
            traceback.print_exc()
            raise

        self._initialize_layers()
        # Check for a previously confirmed ROI session first.  If found and
        # the user accepts, _try_load_existing_roi_session() handles the
        # checkpoint restore itself; otherwise we fall through to the normal path.
        if not self._try_load_existing_roi_session(roi_name=roi_name):
            self.restore_from_checkpoint()

    def _init_persistent_log(self) -> None:
        """Creates a permanent log widget that isn't destroyed between steps."""
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(150)
        self.log_widget.setMaximumHeight(200)
        self.log_widget.setStyleSheet("font-family: 'Courier New', Courier, monospace; font-size: 11px;")

        self.viewer.window.add_dock_widget(
            self.log_widget, area="right", name="Process Log"
        )

    # =========================================================================
    # ROI / SUB-REGION SELECTION
    # =========================================================================

    # --- Shared helpers ---

    def _get_strategy_class(self):
        """Returns the strategy class for the current processing mode."""
        return strategy_for(self.processing_mode)

    def _rebuild_strategy(self) -> None:
        """
        Tears down the current strategy and rebuilds it from the current
        self.config / self.processed_dir / self.image_stack.  Used by both
        the ROI switch-in and switch-out code paths.
        """
        if hasattr(self, 'strategy') and self.strategy is not None:
            try:
                self.strategy.intermediate_state.clear()
            except Exception:
                pass
            self.strategy = None

        self._calculate_spacing()

        strategy_class = self._get_strategy_class()
        if not strategy_class:
            raise ValueError(f"Unsupported mode: {self.processing_mode}")

        self.strategy = strategy_class(
            self.config,
            self.processed_dir,
            self.image_stack.shape,
            self.spacing,
            self.z_scale_factor,
        )
        self.processing_steps = self.strategy.get_step_names()
        self.num_steps = self.strategy.num_steps
        self.step_display_names = {
            name: name.replace('execute_', '').replace('_', ' ').title()
            for name in self.processing_steps
        }

    @staticmethod
    def _build_crop_memmap(
        src: np.ndarray,
        y0: int, x0: int, y1: int, x1: int,
        z_polygons: Dict[int, np.ndarray],
        out_path: str,
        z0_crop: int = 0,
        z1_crop: Optional[int] = None,
    ) -> np.memmap:
        """Thin wrapper over roi_sharing.build_crop_memmap.

        The implementation moved out of this class so batch processing -- which
        runs in a child process with no GUI object -- can build an ROI crop. This
        wrapper stays so existing call sites and any saved user scripts keep
        working, and so there is exactly one implementation.
        """
        from .roi_sharing import build_crop_memmap
        return build_crop_memmap(src, y0, x0, y1, x1, z_polygons, out_path,
                                 z0_crop=z0_crop, z1_crop=z1_crop)

    def _build_roi_config(
        self,
        y0: int, x0: int, y1: int, x1: int,
        base_config: Dict[str, Any],
        z0: int = 0,
        z1: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Thin wrapper over roi_sharing.build_roi_config.

        The implementation moved out of this class for the same reason as the crop
        builder. It previously read the full image shape and the mode off `self`;
        those are now passed explicitly so a batch worker can derive an ROI config
        with no GUI object in existence.
        """
        from .roi_sharing import build_roi_config
        return build_roi_config(
            y0, x0, y1, x1, base_config,
            full_shape=self._full_image_stack.shape,
            mode=self.processing_mode,
            z0=z0, z1=z1,
        )

    # --- Startup: detect an existing ROI session ---

    def _try_load_existing_roi_session(self, roi_name: Optional[str] = None) -> bool:
        """
        Loads a saved ROI (sub-region) session for this image.

        `roi_name` selects one explicitly -- which is how the project view opens a
        particular ROI row, and how a batch worker targets one. When it is None the
        behaviour depends on how many sessions exist: none means fall through to
        the full image, one means ask as before, and several means offer a picker,
        because a channel can now hold "ROI 1", "ROI 2", ... and silently loading
        the first would be a coin toss.

        Handles both the v1 JSON format (single polygon) and the current v2 format
        (dict of Z->polygon entries).

        Returns True if a session was loaded (caller must NOT call
        restore_from_checkpoint again), False otherwise.
        """
        from .roi_sharing import list_roi_sessions, roi_session_dir

        sample_dir = os.path.dirname(self.processed_dir)
        sessions = [se for se in list_roi_sessions(sample_dir)
                    if se["has_polygon"]]

        if roi_name is not None:
            roi_dir = roi_session_dir(sample_dir, roi_name)
            if not roi_dir or not os.path.isfile(
                    os.path.join(roi_dir, "roi_polygon.json")):
                print(f"[ROI] no session named {roi_name!r} for this image")
                return False
        elif not sessions:
            return False
        elif len(sessions) == 1:
            roi_name = sessions[0]["name"]
            roi_dir = sessions[0]["roi_dir"]
            reply = QMessageBox.question(
                None,
                "ROI Session Found",
                "A previous ROI (sub-region) session was found for this image.\n\n"
                "Load the ROI session?\n"
                "(Choose 'No' to work on the full image instead.)",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return False
        else:
            choice = self._pick_roi_session(sessions)
            if choice is None or choice is self.FULL_IMAGE:
                return False        # both mean "work on the full image"
            roi_name, roi_dir = choice

        roi_json = os.path.join(roi_dir, "roi_polygon.json")
        if not os.path.exists(roi_json):
            return False

        try:
            with open(roi_json, 'r') as fh:
                roi_data = json.load(fh)

            bbox = roi_data['bbox']
            y0, x0, y1, x1 = bbox['y0'], bbox['x0'], bbox['y1'], bbox['x1']
            z0_crop = bbox.get('z0', 0)
            z1_crop = bbox.get('z1', None)

            # v1: single 'polygon_yx' key → convert to z_polygons dict
            # v2: 'z_polygons' list of {z, polygon_yx} dicts
            if 'z_polygons' in roi_data:
                z_polygons = {
                    int(entry['z']): np.array(entry['polygon_yx'])
                    for entry in roi_data['z_polygons']
                }
            else:
                z_polygons = {0: np.array(roi_data['polygon_yx'])}

            # Save full-image references, including the display range.
            self._remember_full_image_state()

            # Derive crop shape from saved bbox
            src = self._full_image_stack
            is_3d = src.ndim == 3
            crop_h, crop_w = y1 - y0, x1 - x0
            effective_z1 = z1_crop if z1_crop is not None else (src.shape[0] if is_3d else None)
            if is_3d:
                crop_shape = (effective_z1 - z0_crop, crop_h, crop_w)
            else:
                crop_shape = (crop_h, crop_w)

            crop_path = os.path.join(roi_dir, "roi_image_crop.dat")
            if os.path.exists(crop_path):
                crop_mm = np.memmap(crop_path, dtype=src.dtype, mode='r+',
                                    shape=crop_shape)
            else:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    crop_mm = self._build_crop_memmap(
                        src, y0, x0, y1, x1, z_polygons, crop_path,
                        z0_crop=z0_crop, z1_crop=z1_crop
                    )
                finally:
                    QApplication.restoreOverrideCursor()

            # Load persisted ROI config, or rebuild it fresh
            # A legacy ROI keeps its own filename; anything new is written
            # under the unified one.
            roi_cfg_path = find_config_path(roi_dir) or os.path.join(
                roi_dir, config_basename()
            )
            if os.path.exists(roi_cfg_path):
                with open(roi_cfg_path, 'r') as fh:
                    roi_config = yaml.safe_load(fh) or {}
            else:
                roi_config = self._build_roi_config(
                    y0, x0, y1, x1, self._full_config,
                    z0=z0_crop, z1=z1_crop
                )

            self._switch_to_roi_mode(crop_mm, roi_dir, roi_config,
                                     call_restore=True, roi_name=roi_name)
            return True

        except Exception as exc:
            print(f"[ROI] Failed to load existing session: {exc}")
            traceback.print_exc()
            return False

    SAVED_REGION_SUFFIX = " (region)"

    def _show_saved_region_layers(self) -> int:
        """Outline every saved region on the full image.

        Opening a channel on the full image used to give no sign that regions
        existed, so they were invisible unless you happened to answer a prompt.
        Each region gets its own read-only layer, named after it, so you can see
        where they are and step into one with 'Open region'.
        """
        if self.roi_active:
            return 0
        from .roi_sharing import list_roi_sessions, load_roi_record, record_polygons

        for layer in [l for l in list(self.viewer.layers)
                      if str(l.name).endswith(self.SAVED_REGION_SUFFIX)]:
            self.viewer.layers.remove(layer.name)

        sample_dir = os.path.dirname(self.processed_dir)
        sessions = [se for se in list_roi_sessions(sample_dir)
                    if se["has_polygon"]]
        if not sessions:
            return 0

        colours = ("cyan", "magenta", "yellow", "lime", "orange", "white")
        is_3d = self.image_stack.ndim == 3
        scale = self._layer_scale()

        for index, session in enumerate(sessions):
            record = load_roi_record(session["roi_dir"])
            if not record:
                continue
            try:
                z_polys = record_polygons(record)
            except Exception:
                continue
            verts = []
            for z, poly in sorted(z_polys.items()):
                arr = np.asarray(poly, dtype=float)
                if is_3d:
                    verts.append(np.hstack(
                        [np.full((arr.shape[0], 1), float(z)), arr]))
                else:
                    verts.append(arr)
            if not verts:
                continue
            try:
                layer = self.viewer.add_shapes(
                    verts, name=f"{session['name']}{self.SAVED_REGION_SUFFIX}",
                    shape_type="polygon", edge_color=colours[index % len(colours)],
                    face_color=[0, 0, 0, 0.0], edge_width=2, scale=scale,
                )
                # Read-only: this shows what is committed to disk, and editing it
                # would imply the change is saved, which it is not.
                layer.mode = "pan_zoom"
                layer.editable = False
            except Exception as exc:
                print(f"  [ROI] could not outline {session['name']}: {exc}")

        print(f"  [ROI] outlined {len(sessions)} saved region(s) on the full image")
        return len(sessions)

    def open_roi_session(self) -> None:
        """Switch from the full image into one of its saved regions, in place.

        Avoids closing and reopening the viewer just to reach a region, which is
        what you had to do before if you dismissed the prompt on open.
        """
        from .roi_sharing import list_roi_sessions

        sample_dir = os.path.dirname(
            self._full_processed_dir if self.roi_active else self.processed_dir)
        sessions = [se for se in list_roi_sessions(sample_dir)
                    if se["has_polygon"]]
        if not sessions:
            QMessageBox.information(
                None, "No regions",
                "This image has no saved regions. Draw one and click Apply.")
            return

        choice = self._pick_roi_session(sessions)
        if choice is None:
            return                                  # cancelled
        if choice is self.FULL_IMAGE:
            if self.roi_active:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    self._switch_to_full_image_mode()
                finally:
                    QApplication.restoreOverrideCursor()
            return
        roi_name, _roi_dir = choice
        if self.roi_active and roi_name == self.active_roi_name:
            return                                  # already there

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if self.roi_active:
                # Back to the full image first, so the saved full-image state is
                # restored before a different crop replaces it. call_restore=False
                # because we are only passing through: the full image's restore
                # prompt would be about an image the user is not opening.
                self._switch_to_full_image_mode(call_restore=False)
            if not self._try_load_existing_roi_session(roi_name=roi_name):
                QMessageBox.warning(
                    None, "Could not open region",
                    f"'{roi_name}' could not be loaded. It may be missing its "
                    "polygon file.")
        except Exception as exc:
            print(f"[ROI] open_roi_session failed: {exc}")
            traceback.print_exc()
            QMessageBox.critical(None, "ROI Error", str(exc))
        finally:
            QApplication.restoreOverrideCursor()

    def delete_roi_session(self) -> None:
        """Delete a saved region from disk, with its results.

        Distinct from Clear, which only leaves ROI mode and deliberately keeps the
        region. There was previously no way to remove a region at all from here:
        Clear looked like a delete but wasn't.
        """
        from .roi_sharing import list_roi_sessions, roi_session_dir

        sample_dir = os.path.dirname(
            self._full_processed_dir if self.roi_active else self.processed_dir)
        sessions = [se for se in list_roi_sessions(sample_dir)]
        if not sessions:
            QMessageBox.information(None, "No regions",
                                    "This image has no saved regions to delete.")
            return

        choice = self._pick_roi_session(sessions, prompt="Delete which region?",
                                       include_full=False)
        # include_full=False means the sentinel cannot come back, but unpacking it
        # as a pair would raise, so it is handled rather than assumed away.
        if choice is None or choice is self.FULL_IMAGE:
            return
        roi_name, roi_dir = choice

        try:
            n_files = len([f for f in os.listdir(roi_dir)
                           if f != "roi_polygon.json"])
        except OSError:
            n_files = 0
        reply = QMessageBox.question(
            None, "Delete region",
            f"Delete '{roi_name}' and its {n_files} result file(s)?\n\n"
            "The region outline and anything computed on it are removed from "
            "disk. Full-image results are not affected.\n\nThis cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        was_active = self.roi_active and self.active_roi_name == roi_name
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if was_active:
                # Leave the session before deleting it: the crop memmap is open
                # and the processed_dir points inside the folder being removed.
                self._switch_to_full_image_mode()
            shutil.rmtree(roi_dir)
            print(f"  [ROI] deleted session '{roi_name}'")
            self._show_saved_region_layers()
        except Exception as exc:
            print(f"[ROI] delete_roi_session failed: {exc}")
            traceback.print_exc()
            QMessageBox.critical(None, "ROI Error",
                                 f"Could not delete '{roi_name}':\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    # Sentinel for "the user chose the full image", as opposed to None for
    # "the user cancelled".
    FULL_IMAGE = object()

    def _pick_roi_session(self, sessions, prompt: str = "",
                          include_full: bool = True):
        """Ask which saved region to act on. Returns (name, dir), or None.

        "Full image" is a first-class choice rather than a Cancel, because working
        on the whole image is a normal thing to want and should not require
        dismissing a dialog. It is omitted for destructive actions, where "full
        image" would be a meaningless answer.
        """
        from PyQt5.QtWidgets import QInputDialog  # type: ignore

        full_label = "Full image (no ROI)"
        labels = ([full_label] if include_full else []) + [
            se["name"] for se in sessions]
        if not labels:
            return None
        label, ok = QInputDialog.getItem(
            None, "Choose a region",
            prompt or (f"This image has {len(sessions)} saved regions.\n"
                       "Which would you like to open?"),
            labels, 0, False,
        )
        if not ok:
            return None                      # cancelled: do nothing
        if label == full_label:
            # Distinct from cancelling. Collapsing the two meant picking
            # "Full image" from inside a region did nothing at all, because the
            # caller could not tell the two apart.
            return self.FULL_IMAGE
        for se in sessions:
            if se["name"] == label:
                return se["name"], se["roi_dir"]
        return None

    # --- Three user-facing buttons ---

    def draw_roi(self) -> None:
        """
        Adds an empty Shapes layer in polygon-draw mode and shows instructions.

        Forces Napari into 2D slice view so that:
        - The user can scroll to any Z slice and draw a polygon there.
        - Each completed polygon is tagged with the exact Z slice index
          (read from viewer.dims.current_step[0]) via the shapes data event.

        This is necessary because in 3D perspective mode (ndisplay=3) Napari
        stamps all polygon vertices with the camera's focal Z, making it
        impossible to reliably distinguish polygons drawn on different slices.

        2D:  draw one polygon, confirm.
        3D:  scroll to a slice → draw polygon → scroll to next slice → draw
             polygon → repeat → confirm.  Each polygon is automatically
             assigned the Z slice it was drawn on.  A single polygon is
             extruded through the full Z range.
        """
        layer_name = "ROI Selection"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)

        is_3d = self.image_stack.ndim == 3

        # Force 2D slice mode — the only reliable way to get per-slice Z tags.
        if is_3d:
            self.viewer.dims.ndisplay = 2

        # MUST share the image layer's scale. Without it the layer sits in world
        # space, so for a 2D image (scaled by microns/pixel) napari returns
        # vertices in microns; confirm_roi treats them as pixel indices, and the
        # crop lands offset toward the origin and shrunk by 1/spacing. 3D was
        # unaffected because its y/x scale is 1, which is why this only showed up
        # on 2D data -- and only once real pixel dimensions were imported, since
        # an uncalibrated image has spacing exactly 1.0.
        self.viewer.add_shapes(
            name=layer_name,
            ndim=3 if is_3d else 2,
            shape_type='polygon',
            edge_color='yellow',
            face_color=[1, 1, 0, 0.08],
            edge_width=3,
            scale=self._layer_scale(),
        )
        self.viewer.layers[layer_name].mode = 'add_polygon'

        # Reset the reliable Z→polygon map and connect the data event.
        # The event fires each time the shapes data changes (polygon added/edited).
        # We record the current Z slice and the polygon count so we can detect
        # additions vs edits and avoid double-counting.
        # A LIST of (z, polygon), not a dict keyed by z. Keying by z meant a
        # second polygon drawn on the same slice replaced the first, so in 2D --
        # where everything is z=0 -- only the last polygon drawn ever survived.
        self._roi_drawn: List[Tuple[int, np.ndarray]] = []
        self._roi_last_polygon_count: int = 0

        def _on_shapes_data_changed(event=None):
            """Called whenever the shapes layer data changes."""
            shapes_layer = self.viewer.layers[layer_name] if layer_name in self.viewer.layers else None
            if shapes_layer is None:
                return
            current_count = len(shapes_layer.data)
            if current_count <= self._roi_last_polygon_count:
                # Edit or deletion — update the existing entry in place
                # by re-reading all shapes with their stored Z tags.
                return
            # A new polygon was just completed — record the current Z slice.
            self._roi_last_polygon_count = current_count
            z_slice = int(self.viewer.dims.current_step[0]) if is_3d else 0
            poly_raw = np.array(shapes_layer.data[-1], dtype=float)
            # Strip Z column if present (ndisplay=2 still gives (N,3) in 3D)
            poly_yx = poly_raw[:, 1:] if poly_raw.shape[1] > 2 else poly_raw
            self._roi_drawn.append((z_slice, poly_yx))
            print(f"  [ROI] Polygon recorded at Z={z_slice} "
                  f"({len(self._roi_drawn)} total)")

        self.viewer.layers[layer_name].events.data.connect(_on_shapes_data_changed)

        if is_3d:
            msg = (
                "Draw one or more regions. Each polygon is tagged to the Z\n"
                "slice it was drawn on.\n\n"
                "  1. Scroll to a Z slice\n"
                "  2. Click to add vertices, press Escape to close the polygon\n"
                "  3. Repeat for as many regions as you like\n\n"
                "Several polygons on the SAME slice become separate regions.\n"
                "Polygons on DIFFERENT slices are ambiguous, so you will be\n"
                "asked whether they are one region spanning those slices or\n"
                "separate regions. A single polygon extrudes through all Z.\n\n"
                "When finished, click  \u2713 Apply."
            )
        else:
            msg = (
                "Draw one or more regions on the image.\n\n"
                "  \u2022 Click to add vertices\n"
                "  \u2022 Press Escape to close the polygon\n"
                "  \u2022 Repeat for as many regions as you like\n\n"
                "Every polygon becomes its own region, each with its own\n"
                "config and results.\n\n"
                "When finished, click  \u2713 Apply."
            )

        QMessageBox.information(None, "Draw ROI", msg)

    def _roi_spec(self, z_polygons, is_3d: bool, img_h: int, img_w: int):
        """Validate one region's polygons and return its geometry, or None.

        Everything is validated for every region BEFORE any of them is written,
        so a bad third polygon cannot leave two half-created sessions on disk.
        """
        all_yx = np.vstack(list(z_polygons.values()))
        y0 = max(0, int(np.floor(all_yx[:, 0].min())))
        x0 = max(0, int(np.floor(all_yx[:, 1].min())))
        y1 = min(img_h, int(np.ceil(all_yx[:, 0].max())) + 1)
        x1 = min(img_w, int(np.ceil(all_yx[:, 1].max())) + 1)
        crop_h, crop_w = y1 - y0, x1 - x0

        if crop_h < 10 or crop_w < 10:
            QMessageBox.warning(
                None, "ROI Too Small",
                f"One region is too small ({crop_h} \u00d7 {crop_w} px; the "
                "minimum is 10 px per side). Nothing was created.")
            return None

        # Coordinate-space sanity check. Vertices are expected in image PIXEL
        # indices; an extent far outside the image means the Shapes layer and the
        # image layer are in different spaces, and cropping on those numbers would
        # silently sample the wrong part of the image.
        if (float(all_yx[:, 0].max()) > img_h * 1.5
                or float(all_yx[:, 1].max()) > img_w * 1.5):
            QMessageBox.critical(
                None, "ROI Coordinate Error",
                "A drawn region lies outside the image "
                f"({all_yx[:, 0].max():.0f}, {all_yx[:, 1].max():.0f} vs image "
                f"{img_h} x {img_w}).\n\nThis indicates the ROI layer is not in "
                "the image's coordinate space. Nothing was created. Please "
                "report this.")
            return None

        if is_3d:
            sorted_zs = sorted(z_polygons.keys())
            if len(sorted_zs) == 1:
                z0_crop, z1_crop = 0, self.image_stack.shape[0]
                z_desc = "extruded through all Z"
            else:
                z0_crop = max(0, sorted_zs[0])
                z1_crop = min(self.image_stack.shape[0], sorted_zs[-1] + 1)
                z_desc = (f"Z {z0_crop}\u2013{z1_crop} "
                          f"({len(sorted_zs)} defined levels)")
        else:
            z0_crop, z1_crop = 0, None
            z_desc = "2D"

        return {"z_polygons": z_polygons, "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                "crop_h": crop_h, "crop_w": crop_w,
                "z0": z0_crop, "z1": z1_crop, "z_desc": z_desc}

    def _write_roi_session(self, spec):
        """Create one region on disk. Returns (name, dir, crop memmap, config)."""
        from .roi_sharing import next_roi_name, roi_session_dir

        sample_dir = os.path.dirname(self._full_processed_dir)
        roi_name = next_roi_name(sample_dir)
        roi_dir = roi_session_dir(sample_dir, roi_name)
        if not roi_dir:
            # describe_channel could not read the folder; fall back to the legacy
            # path rather than refusing to create a region at all.
            roi_dir = self._full_processed_dir + "_roi"
            roi_name = "ROI 1"
        os.makedirs(roi_dir, exist_ok=True)
        print(f"  [ROI] creating session '{roi_name}' in "
              f"{os.path.basename(roi_dir)}")

        y0, x0, y1, x1 = spec["y0"], spec["x0"], spec["y1"], spec["x1"]
        z0_crop, z1_crop = spec["z0"], spec["z1"]
        z_polygons = spec["z_polygons"]

        roi_data = {
            "format": "v2",
            "z_polygons": [
                {"z": int(z), "polygon_yx": np.asarray(poly, dtype=float).tolist()}
                for z, poly in sorted(z_polygons.items())
            ],
            "bbox": {"y0": y0, "x0": x0, "y1": y1, "x1": x1,
                     "z0": z0_crop, "z1": z1_crop},
            "full_image_shape": list(self._full_image_stack.shape),
        }
        with open(os.path.join(roi_dir, "roi_polygon.json"), 'w') as fh:
            json.dump(roi_data, fh, indent=2)

        crop_mm = self._build_crop_memmap(
            self._full_image_stack, y0, x0, y1, x1, z_polygons,
            os.path.join(roi_dir, "roi_image_crop.dat"),
            z0_crop=z0_crop, z1_crop=z1_crop,
        )
        roi_config = self._build_roi_config(
            y0, x0, y1, x1, self._full_config, z0=z0_crop, z1=z1_crop)
        with open(os.path.join(roi_dir, config_basename()), 'w') as fh:
            yaml.safe_dump(roi_config, fh, default_flow_style=False,
                           sort_keys=False)
        return roi_name, roi_dir, crop_mm, roi_config

    def _group_drawn_polygons(self, drawn, is_3d: bool):
        """Split drawn polygons into one entry per region.

        The rule is chosen to be predictable rather than clever:

          * one polygon                 -> one region
          * several, all on one Z slice -> one region EACH. This covers all of 2D
            and is what "draw several regions in one go" means.
          * several across >=2 Z slices -> ambiguous, so ask. The default stays
            "one region spanning Z", because that is the documented 3D workflow
            (draw the same structure on several slices) and silently changing it
            would split existing users' regions apart.

        Returns a list of {z: polygon} dicts, or [] if the user cancelled.
        """
        if len(drawn) <= 1:
            return [{z: poly} for z, poly in drawn]

        distinct_z = sorted({z for z, _ in drawn})
        if len(distinct_z) <= 1:
            return [{z: poly} for z, poly in drawn]

        reply = QMessageBox.question(
            None, "Several polygons drawn",
            f"You drew {len(drawn)} polygons across {len(distinct_z)} Z slices.\n\n"
            "Save them as ONE region spanning those slices, or as "
            f"{len(drawn)} SEPARATE regions?\n\n"
            "Yes  -  one region spanning Z\n"
            f"No   -  {len(drawn)} separate regions",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if reply == QMessageBox.Cancel:
            return []
        if reply == QMessageBox.Yes:
            # One region. Where two polygons share a slice only one mask can be
            # applied to it, so say so rather than dropping one silently.
            merged = {}
            collisions = 0
            for z, poly in drawn:
                if z in merged:
                    collisions += 1
                merged[z] = poly
            if collisions:
                QMessageBox.information(
                    None, "Overlapping slices",
                    f"{collisions} polygon(s) shared a Z slice with another. A "
                    "single region uses one outline per slice, so the later "
                    "polygon was kept for those slices.\n\n"
                    "Choose 'separate regions' instead if you meant them to be "
                    "distinct.")
            return [merged]
        return [{z: poly} for z, poly in drawn]

    def confirm_roi(self) -> None:
        """
        Reads all drawn polygons (one per Z level or a single extruded one),
        builds the 3D crop, persists the ROI to disk, and reinitialises the
        pipeline on the sub-region.

        For 3D images: each shape in the layer carries the Z slice it was drawn
        on.  Multiple polygons at different Z levels define a true 3D ROI.
        Slices between defined levels use the nearest polygon (nearest-neighbour
        interpolation).  A single polygon is extruded through the full Z range.
        For 2D images: only the first/only polygon is used.
        """
        layer_name = "ROI Selection"
        if layer_name not in self.viewer.layers:
            QMessageBox.warning(None, "No ROI",
                                "Please draw an ROI first using '✏ Draw ROI'.")
            return

        shapes_layer = self.viewer.layers[layer_name]
        if not shapes_layer.data:
            QMessageBox.warning(None, "Empty ROI",
                                "The ROI layer contains no polygon. "
                                "Please draw one first.")
            return

        is_3d = self.image_stack.ndim == 3
        img_h = self.image_stack.shape[-2]
        img_w = self.image_stack.shape[-1]

        # --- Use the event-tracked Z→polygon map built during draw_roi ---
        # Fall back to parsing from vertex coordinates only if the map is
        # empty (e.g. confirm clicked without using draw_roi first).
        drawn: List[Tuple[int, np.ndarray]] = list(getattr(self, '_roi_drawn', []))
        if drawn:
            print(f"  [ROI] {len(drawn)} polygon(s) drawn at "
                  f"Z={sorted({z for z, _ in drawn})}")
        else:
            # Fallback: parse Z from the vertex arrays. Works in 2D; in 3D
            # perspective mode napari stamps every vertex with the camera focal
            # plane, so per-slice tagging needs the Draw button.
            for raw in shapes_layer.data:
                arr = np.array(raw, dtype=float)
                if arr.shape[1] == 3:
                    drawn.append((int(round(float(arr[:, 0].mean()))), arr[:, 1:]))
                else:
                    drawn.append((0, arr))

        if not drawn:
            QMessageBox.warning(None, "Empty ROI", "No valid polygons found.")
            return

        groups = self._group_drawn_polygons(drawn, is_3d)
        if not groups:
            return  # user cancelled the grouping question

        # --- Validate and describe every region before writing anything ---
        specs = []
        for z_polygons in groups:
            spec = self._roi_spec(z_polygons, is_3d, img_h, img_w)
            if spec is None:
                return          # a problem was reported; nothing written
            specs.append(spec)

        full_shape = self.image_stack.shape
        if len(specs) == 1:
            sp = specs[0]
            detail = (f"YX bounding box: rows {sp['y0']}\u2013{sp['y1']}, "
                      f"cols {sp['x0']}\u2013{sp['x1']}\n"
                      f"Crop YX size: {sp['crop_h']} \u00d7 {sp['crop_w']} px  "
                      f"(full image: {full_shape[-2]} \u00d7 {full_shape[-1]})\n"
                      f"Z range: {sp['z_desc']}\n")
        else:
            detail = f"{len(specs)} regions will be created:\n" + "\n".join(
                f"  \u2022 {sp['crop_h']} \u00d7 {sp['crop_w']} px, {sp['z_desc']}"
                for sp in specs) + "\n"

        reply = QMessageBox.question(
            None,
            "Confirm regions" if len(specs) > 1 else "Confirm ROI",
            detail + "\nEach region gets its own config and is processed "
            "independently, starting from Step 1.\n\nContinue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        created = []
        try:
            # Save full-image references (idempotent if called again)
            self._remember_full_image_state()

            for spec in specs:
                created.append(self._write_roi_session(spec))

            # --- Remove the draw layer before reinitializing ---
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)
            self._roi_drawn = []

            if len(created) == 1:
                # One region: step into it, as before.
                name, roi_dir, crop_mm, roi_config = created[0]
                self._switch_to_roi_mode(crop_mm, roi_dir, roi_config,
                                         call_restore=False, roi_name=name)
            else:
                # Several regions: stepping into an arbitrary one would be a
                # guess, so stay on the full image and show them all. Use
                # "Open region" to enter one.
                for _n, _d, crop_mm, _c in created:
                    del crop_mm
                self._show_saved_region_layers()
                QApplication.restoreOverrideCursor()
                QMessageBox.information(
                    None, "Regions created",
                    f"{len(created)} regions were created:\n\n"
                    + "\n".join(f"  \u2022 {n}" for n, _d, _m, _c in created)
                    + "\n\nThey are outlined on the full image. Use "
                      "'Open region' to work on one, or process them from the "
                      "Project View.")

        except Exception as exc:
            print(f"[ROI] confirm_roi failed: {exc}")
            traceback.print_exc()
            QMessageBox.critical(None, "ROI Error",
                                 f"Failed to set up ROI session:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def clear_roi(self) -> None:
        """
        Removes the ROI draw layer.  If an ROI session is active, offers to
        return to full-image processing mode.  ROI outputs are kept on disk.
        """
        if "ROI Selection" in self.viewer.layers:
            self.viewer.layers.remove("ROI Selection")

        if not self.roi_active:
            return

        # Name the session being left, now that a channel can hold several.
        which = self.active_roi_name or "this region"
        reply = QMessageBox.question(
            None,
            "Return to Full Image",
            f"Return to full-image processing mode?\n\n"
            f"'{which}' and its outputs are KEPT on disk and can be reopened\n"
            "later. Use 'Delete region' to remove it permanently.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._switch_to_full_image_mode()
        except Exception as exc:
            print(f"[ROI] clear_roi failed: {exc}")
            traceback.print_exc()
            QMessageBox.critical(None, "ROI Error",
                                 f"Failed to return to full-image mode:\n{exc}")
        finally:
            QApplication.restoreOverrideCursor()

    # --- Engine: mode switching ---

    def _switch_to_roi_mode(
        self,
        cropped_image: np.ndarray,
        roi_processed_dir: str,
        roi_config: Dict[str, Any],
        call_restore: bool = False,
        roi_name: Optional[str] = None,
    ) -> None:
        """
        Reinitialises the pipeline to operate on *cropped_image*.

        Args:
            cropped_image:     Cropped (and masked) array or memmap.
            roi_processed_dir: The *_roi processed directory path.
            roi_config:        Config with rescaled physical dimensions.
            call_restore:      If True, calls restore_from_checkpoint() at the
                               end (used when loading an existing ROI session).
                               If False (fresh confirm), starts from Step 1.
        """
        self._stop_worker_safely()

        self.roi_active = True
        self.active_roi_name = roi_name
        self.image_stack = cropped_image
        self.processed_dir = roi_processed_dir
        # Same migration as the full-image path: a region config saved before
        # the modes merged carries the legacy step keys too.
        self.config = self._migrate_config(roi_config)
        self.initial_config = copy.deepcopy(roi_config)

        self._rebuild_strategy()
        self.clear_current_widgets()
        self._initialize_layers()

        if self.log_widget:
            self.log_widget.clear()
            shape = self.image_stack.shape
            self.log_widget.append(
                f"[ROI MODE]  crop shape: {shape}\n"
                f"            dir: {os.path.basename(roi_processed_dir)}"
            )

        self.current_step["value"] = 0
        self.strategy.intermediate_state = {}

        if call_restore:
            self.restore_from_checkpoint()
        else:
            # Fresh start: wipe any stale roi outputs and go to Step 1
            self.delete_all_checkpoint_files()
            self.create_step_widgets(self.processing_steps[0])

        # Notify connected slots (refresh_nav in app_launch) so the Back /
        # Forward / Process buttons reflect the current step index.
        self.process_finished.emit()

        print(f"[ROI] Now in ROI mode — shape {self.image_stack.shape}, "
              f"dir: {os.path.basename(roi_processed_dir)}")

    def _switch_to_full_image_mode(self, call_restore: bool = True) -> None:
        """Tears down the ROI session and reinstates full-image processing.

        `call_restore=False` is for passing THROUGH full-image mode on the way to
        another region. The restore prompt ("all steps complete - view results or
        restart?") describes the full image, so raising it when the user asked for
        a different region was answering a question they had not been asked.
        """
        if self._full_image_stack is None:
            return

        self._stop_worker_safely()

        self.roi_active = False
        self.active_roi_name = None
        # Full image gets its own auto-contrast again.
        self._full_contrast_limits = None
        self.image_stack = self._full_image_stack
        self.processed_dir = self._full_processed_dir
        self.config = copy.deepcopy(self._full_config)
        self.initial_config = copy.deepcopy(self._full_config)

        self._rebuild_strategy()
        self.clear_current_widgets()
        self._initialize_layers()

        if self.log_widget:
            self.log_widget.clear()
            self.log_widget.append("[FULL IMAGE MODE]")

        self.current_step["value"] = 0
        self.strategy.intermediate_state = {}
        if call_restore:
            self.restore_from_checkpoint()

        # Notify connected slots (refresh_nav in app_launch) so the Back /
        # Forward / Process buttons reflect the resumed step index.
        self.process_finished.emit()

        print("[ROI] Returned to full-image mode.")

    def _stop_worker_safely(self) -> None:
        """Stops a running step: kills the worker processes it spawned, then
        detaches the (now quickly-unwinding) thread so it can't crash the app on
        destruction."""
        if getattr(self, 'worker', None) and self.worker.isRunning():
            lifecycle("worker.stop.begin",
                      baseline_pids=len(getattr(self, '_worker_child_baseline', set())))
            _register_quit_hook()  # Ensure cleanup happens at shutdown

            # Kill the multiprocessing.Pool workers this step spawned. Once they
            # die, the imap loop in the worker thread raises and unwinds on its
            # own — no computation keeps running in the background.
            try:
                _terminate_new_children(getattr(self, '_worker_child_baseline', set()))
            except Exception:
                pass

            try:
                self.worker.finished_signal.disconnect()
                self.worker.error_signal.disconnect()
            except Exception:
                pass

            # _on_step_finished won't run now (signals are detached), so restore
            # the streams here or stdout stays pointed at a dead log widget.
            try:
                sys.stdout = self.original_stdout
                sys.stderr = self.original_stderr
            except Exception:
                pass

            # PRESERVE data references so the unwinding worker doesn't segfault
            # reading a memmap we might otherwise free.
            self.worker._preserved_strategy = self.strategy
            self.worker._preserved_stack = self.image_stack

            self.worker.setParent(None)
            _orphan_threads.append(self.worker)
            self.worker.finished.connect(lambda w=self.worker: _cleanup_orphan_thread(w))
            self.worker = None
            lifecycle("worker.stop.detached", orphan_count=len(_orphan_threads))

    def shutdown_and_cleanup(self) -> None:
        """Forcefully clears all data references and Napari internal buffers."""
        # Check if worker is running BEFORE we stop it
        is_worker_running = getattr(self, 'worker', None) and self.worker.isRunning()
        try:
            n_layers = len(self.viewer.layers) if self.viewer else 0
        except Exception:
            n_layers = "?"
        lifecycle("cleanup.shutdown.begin",
                  worker_running=bool(is_worker_running), layers=n_layers)
        self._stop_worker_safely()

        # 1. Clear Napari layers and buffers first
        if self.viewer:
            lifecycle("cleanup.layers.clear")
            try:
                self.viewer.layers.clear() 
            except Exception:
                log.exception("Error clearing napari layers")
        
        # 2. Clear strategy and large data references
        if hasattr(self, 'strategy') and self.strategy is not None:
            if hasattr(self.strategy, 'intermediate_state'):
                # CRITICAL FIX: DO NOT clear the dictionary in-place if a worker is running!
                # Doing so rips the memory-mapped file out from under the Loky process pool, causing a crash.
                if not is_worker_running:
                    lifecycle("cleanup.intermediate_state.clear")
                    self.strategy.intermediate_state.clear() 
                else:
                    lifecycle("cleanup.intermediate_state.skip", reason="worker still running")
            self.strategy = None
        
        lifecycle("cleanup.refs.release")  # dropping image_stack / viewer refs
        self.image_stack = None 
        self.viewer = None 
        
        gc.collect()
        gc.collect()
        lifecycle("cleanup.shutdown.end")
        log.info("Deep cleanup complete. All heavy references released.")

    def _calculate_spacing(self) -> None:
        """Parses spacing from config or defaults to 1.0."""
        from .metadata import require_dimensions

        # Rank comes from the loaded image, not from the mode string, which is
        # the same for every project now. Passing it also lets
        # require_dimensions reject a config whose block disagrees with the
        # image -- the protection the old mode-based key choice provided.
        ndim = len(self.image_stack.shape)
        # No fallback. Defaulting a TOTAL extent to 1.0 made a whole 2916 px axis
        # one micron across, which silently rescaled every physical parameter --
        # a 0.7 um smoothing sigma became a 2084 px blur. An unset extent stops
        # the run instead.
        label = (self.active_roi_name
                 or os.path.basename(os.path.dirname(self.processed_dir))
                 or "this image")
        dim = require_dimensions(self.config, source=label, ndim=ndim)
        tx = dim['x']
        ty = dim['y']
        tz = dim['z'] if 'z' in dim else 1.0

        shape = self.image_stack.shape
        
        if len(shape) == 2:
            # 2D Case
            ys = ty / shape[0] if shape[0] > 0 else 1.0
            xs = tx / shape[1] if shape[1] > 0 else 1.0
            self.spacing = (1.0, ys, xs)  # Z=1.0 for compatibility
            self.z_scale_factor = 1.0
            
        elif len(shape) == 3:
            # 3D Case
            zs = tz / shape[0] if shape[0] > 0 else 1.0
            ys = ty / shape[1] if shape[1] > 0 else 1.0
            xs = tx / shape[2] if shape[2] > 0 else 1.0
            self.spacing = (zs, ys, xs)
            self.z_scale_factor = zs / xs if xs > 1e-9 else 1.0
            
        else:
            self.spacing = (1.0, 1.0, 1.0)
            self.z_scale_factor = 1.0

    def _remember_full_image_state(self) -> None:
        """Snapshot everything needed to return to the full image later.

        ONE place, called by every path that enters a region. There used to be two
        independent copies of this: confirm_roi captured the display range, and the
        session loader did not. Entering a region by drawing it therefore kept the
        full image's brightness while opening a SAVED region did not, so the crop
        came up auto-contrasted against its own mostly-zero array and looked
        thresholded. Adding the fourth line to only one copy is what caused that,
        so there is no second copy to forget.

        Idempotent: does nothing once a region is already active, so passing
        through does not snapshot a crop as if it were the full image.
        """
        if self.roi_active:
            return
        self._full_image_stack = self.image_stack
        self._full_processed_dir = self.processed_dir
        self._full_config = copy.deepcopy(self.config)
        self._full_contrast_limits = self._current_contrast_limits()

    def _current_contrast_limits(self):
        """Contrast limits of the image layer currently on screen, or None."""
        name = f"Original stack ({self.processing_mode} mode)"
        try:
            if name in self.viewer.layers:
                return list(self.viewer.layers[name].contrast_limits)
        except Exception:
            pass
        return None

    def _layer_scale(self) -> tuple:
        """Napari `scale` for layers over the current image.

        Single source of truth so an overlaid layer cannot end up in a different
        world space from the image. That mattered: the ROI Shapes layer used to be
        added unscaled, and because the 2D branch below scales by MICRONS PER
        PIXEL, napari then reported polygon vertices in microns while confirm_roi
        read them as pixel indices -- cropping the wrong part of the image.
        """
        return (
            (self.z_scale_factor, 1, 1) if self.image_stack.ndim == 3
            else (self.spacing[1], self.spacing[2])
        )

    def _display_levels(self):
        """Pyramid levels for the image layer, or None to show it directly.

        Only for the FULL image: an ROI crop is a small `.dat` of its own and
        needs no preview, and `self.image_stack` is then the crop rather than
        the file `self.file_loc` names, so a pyramid built from the latter
        would draw the wrong pixels.

        `self.image_stack` is left untouched either way. Every crop, shape and
        rank check in this class reads it, and the layer is the only thing that
        should ever see a reduced level -- the pyramid exists to draw pixels,
        not to measure them.
        """
        if self.roi_active:
            return None
        try:
            from .display_pyramid import open_levels
            return open_levels(self.file_loc)
        except Exception as exc:
            print(f"  [display] preview unavailable ({exc}); "
                  "showing full resolution")
            return None

    def _initialize_layers(self) -> None:
        """Adds the original image to Napari."""
        self.viewer.layers.clear() 
        layer_name = f"Original stack ({self.processing_mode} mode)"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)

        levels = self._display_levels()
        source = levels if levels else self.image_stack
        # Explicit limits so napari does not scan pixel data to invent them.
        # That scan is a large part of the time to open one of these images,
        # and it reads a different array once a pyramid is present, which is
        # why the slider handles moved as soon as one appeared.
        limits: Dict[str, Any] = {}
        try:
            from .display_pyramid import contrast_limits_for
            computed = contrast_limits_for(source)
            if computed:
                limits = {"contrast_limits": list(computed)}
        except Exception:
            limits = {}
        layer = self.viewer.add_image(
            source, name=layer_name, scale=self._layer_scale(),
            multiscale=bool(levels), **limits
        )
        # Carry the full image's display range into an ROI session. A crop is
        # bounding-box shaped with everything outside the polygon zeroed, so
        # napari's auto-contrast stretches to that mostly-empty array and the
        # sub-region looks thresholded rather than like the image it came from.
        if getattr(self, "_full_contrast_limits", None):
            try:
                layer.contrast_limits = self._full_contrast_limits
            except Exception:
                pass

        # Outline any saved regions so they are visible from the full image, and
        # reachable via 'Open region', instead of being invisible unless a prompt
        # happened to be answered on open.
        if not self.roi_active:
            try:
                self._show_saved_region_layers()
            except Exception as exc:
                print(f"  [ROI] could not outline saved regions: {exc}")

    def restore_from_checkpoint(self) -> None:
        """
        Checks for existing outputs and prompts the user to Resume/View or Restart.

        The choice is offered in an explicit loop rather than by mutual recursion
        with _confirm_restart(). Declining the restart confirmation comes back
        here to re-offer the choice, but a dismissal that is NOT an explicit
        button press -- Escape, the window's close box, or a stray queued event
        closing the modal -- is treated as the safe, non-destructive "just show
        what's on disk" and ends the loop.

        This matters because QMessageBox.clickedButton() returns None on such a
        dismissal. The previous code routed None into the `else` (Restart) path,
        and _confirm_restart() then recursed back into this method, so a repeated
        or automatic dismissal (a held Escape key, or an event delivered while
        the dialog is shown right after the viewer window is raised/activated)
        bounced between the two methods, stacking a new modal dialog and two
        nested exec_() event loops each round until the machine became
        unresponsive. Terminating on dismissal and looping instead of recursing
        makes that runaway impossible.
        """
        checkpoint_step = self.strategy.get_last_completed_step()

        if checkpoint_step <= 0:
            self.create_step_widgets(self.processing_steps[0])
            return

        # Load saved config (once, before prompting).
        files = self.strategy.get_checkpoint_files()
        if files.get("config") and os.path.exists(files["config"]):
            try:
                with open(files["config"], 'r') as f:
                    saved = yaml.safe_load(f)
                    if saved:
                        self.config.update(saved)
                        self.strategy.config = self.config
                        # Restore intermediate state (e.g. threshold)
                        if 'saved_state' in self.config:
                            s = self.config['saved_state']
                            if 'segmentation_threshold' in s:
                                self.strategy.intermediate_state['segmentation_threshold'] = \
                                    float(s['segmentation_threshold'])
            except Exception:
                pass

        is_complete = (checkpoint_step == self.num_steps)

        def _accept_existing() -> None:
            """Non-destructive outcome: load what's on disk and settle there."""
            if not is_complete:
                self.strategy.intermediate_state['original_volume_ref'] = self.image_stack
            self.load_checkpoint_data(checkpoint_step)
            self.current_step["value"] = checkpoint_step
            if not is_complete and checkpoint_step < self.num_steps:
                self.create_step_widgets(self.processing_steps[checkpoint_step])

        while True:
            msg = QMessageBox()
            if is_complete:
                msg.setText("All steps complete.")
                msg.setInformativeText("View results or restart from beginning?")
                accept_btn = msg.addButton("View Results", QMessageBox.YesRole)
            else:
                msg.setText("Resume previous session?")
                msg.setInformativeText(f"Found data up to Step {checkpoint_step}.\n"
                                       f"Resume from Step {checkpoint_step + 1}?")
                accept_btn = msg.addButton("Resume", QMessageBox.YesRole)
            restart_btn = msg.addButton("Restart", QMessageBox.NoRole)
            # Escape / closing the dialog maps to the non-destructive choice, so
            # a dismissal can never fall through to the destructive Restart path
            # or re-open this prompt.
            msg.setDefaultButton(accept_btn)
            msg.setEscapeButton(accept_btn)
            msg.exec_()

            if msg.clickedButton() is restart_btn:
                # Explicit Restart request: confirm it. If the reset goes ahead
                # we're done; if the user declines (or dismisses the confirm),
                # loop once to re-offer the choice. _confirm_restart() no longer
                # calls back into this method, so no recursive dialog stack can
                # build up.
                if self._confirm_restart():
                    return
                continue

            # "View Results" / "Resume", or any dismissal (clickedButton() is
            # None): settle on the existing results without re-prompting.
            _accept_existing()
            return

    def _confirm_restart(self) -> bool:
        """Confirm, then delete old files and restart from Step 1.

        Returns True if the pipeline was actually reset, False if the user
        declined or dismissed the confirmation. This method deliberately does
        NOT re-open the resume/restart prompt itself: the caller decides what to
        do next. Recursing back into restore_from_checkpoint() here is what let a
        repeated/automatic dismissal spawn an unbounded chain of dialogs.
        """
        reply = QMessageBox.question(
            self.viewer.window._qt_window,
            "Confirm Restart",
            "This will delete all existing processing files for this mode.\nAre you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,   # default, and Escape/close, map to No
        )

        if reply != QMessageBox.Yes:
            return False

        self.delete_all_checkpoint_files()
        self.current_step["value"] = 0
        self.strategy.intermediate_state = {}
        self.config = self.initial_config.copy()
        self.strategy.config = self.config
        self._initialize_layers()
        self.create_step_widgets(self.processing_steps[0])
        return True

    def delete_all_checkpoint_files(self) -> None:
        """Helper to clear disk artifacts."""
        for _, path in self.strategy.get_checkpoint_files().items():
            self.strategy._remove_file_safely(path)

    def load_checkpoint_data(self, checkpoint_step: int) -> None:
        """Loads visualization data."""
        self.strategy.load_checkpoint_data(self.viewer, checkpoint_step)

    def cleanup_step(self, step_number: int, keep_layers: bool = False) -> None:
        """Cleans artifacts for a specific step.

        `keep_layers=True` deletes the step's files but leaves its napari layers
        alone. Passing viewer=None is what suppresses layer removal, since
        `_remove_layer_safely` no-ops on a None viewer -- the same path the batch
        processor already uses.

        This exists for the re-process case: destroying the layer for a step that
        is about to re-add it forces napari to free and reallocate the GPU
        texture. Repeating that on every re-run churned the driver, and on at
        least one AMD setup ended in a GPU context loss and a SIGABRT. Updating
        the layer in place (which `_add_layer_safely` does when the layer still
        exists) avoids the churn entirely.
        """
        self.strategy.cleanup_step_artifacts(
            None if keep_layers else self.viewer, step_number)

    # ------------------------------------------------------------------ #
    # Valid frontier + non-destructive navigation
    # ------------------------------------------------------------------ #
    # Model: steps [0, frontier) are "processed and current" (their artifact is
    # on disk and their parameters are unchanged). `frontier` is the first step
    # that is unprocessed OR (for the step currently on screen) has unsaved edits.
    # Because edits are never allowed to survive navigation, only the current
    # step can ever be dirty, so the frontier is fully determined by on-disk
    # artifacts plus the current step's edit state. Navigation (Back/Forward) is
    # non-destructive; only Process deletes downstream results.

    def _params_snapshot(self, config_key: str) -> Dict[str, Any]:
        """Deep-copied {param_name: value} for a step's parameter block."""
        block = self.config.get(config_key, {}) or {}
        params = block.get("parameters", {}) or {}
        return {
            name: copy.deepcopy(pconf.get("value"))
            for name, pconf in params.items()
            if isinstance(pconf, dict)
        }

    def is_current_step_dirty(self) -> bool:
        """True if the on-screen step has parameter values differing from the
        ones its widgets were built from (i.e. unsaved edits)."""
        # No editable step is shown when we're past the last step (e.g. right
        # after processing the final step, which clears the widgets). The stale
        # baseline from the step we just committed must not read as "dirty".
        cur = self.current_step["value"]
        if cur < 0 or cur >= self.num_steps:
            return False
        key = getattr(self, "_active_config_key", None)
        if not key:
            return False
        # The interaction-analysis step is repeatable and not part of the
        # processed segmentation chain, so it never gates navigation.
        if getattr(self, "current_step_method", None) == "execute_interaction_analysis":
            return False
        return self._params_snapshot(key) != getattr(self, "_active_baseline", {})

    def _revert_current_edits(self) -> None:
        """Restore the current step's parameters to their built-from baseline."""
        key = getattr(self, "_active_config_key", None)
        baseline = getattr(self, "_active_baseline", None)
        if not key or baseline is None:
            return
        params = (self.config.get(key, {}) or {}).get("parameters", {}) or {}
        for name, val in baseline.items():
            if isinstance(params.get(name), dict):
                params[name]["value"] = copy.deepcopy(val)

    def valid_frontier(self) -> int:
        """0-based index of the first step that is not processed-and-current."""
        completed = self.strategy.get_last_completed_step()  # count of leading done
        frontier = completed
        cur = self.current_step["value"]
        # Editing an already-processed step collapses the frontier to it: its
        # downstream results are now provisional until it is re-processed.
        if cur < frontier and self.is_current_step_dirty():
            frontier = cur
        return frontier

    def can_go_back(self) -> bool:
        return self.current_step["value"] > 0

    def can_go_forward(self) -> bool:
        """Forward moves into any already-valid position, up to and including the
        terminal 'all steps complete' state (current == num_steps) once every
        step is processed. The valid frontier is the only cap: a config that
        isn't fully processed, or an edited (dirty) current step, stops Forward
        at the right place.
        """
        cur = self.current_step["value"]
        return (cur + 1) <= self.valid_frontier()

    def can_process(self) -> bool:
        """Process is available on the first non-valid step, i.e. the current
        step is either unprocessed or has unsaved edits (which collapse the
        frontier to it). Clean, already-processed steps have nothing to compute."""
        cur = self.current_step["value"]
        if cur >= self.num_steps:
            return False
        return cur == self.valid_frontier()

    def go_forward(self) -> None:
        """Navigate forward without computing anything.

        Advancing past the last step lands on the terminal 'complete' state
        (current == num_steps): no parameter widgets, just the final results
        (the viewer layers persist). This mirrors where processing the last step
        leaves you, so Back/Forward are symmetric at the end.
        """
        if not self.can_go_forward():
            return
        self.current_step["value"] += 1
        idx = self.current_step["value"]
        if idx >= self.num_steps:
            self.clear_current_widgets()
        else:
            self.create_step_widgets(self.processing_steps[idx])

    def go_back(self) -> None:
        """Non-destructive back navigation.

        Results are never deleted by going back. If the current step has unsaved
        edits, the user must resolve them first: discard (revert to the last
        committed values) or process now (compute this step, which clears later
        results). Cancelling leaves everything as-is.
        """
        if not self.can_go_back():
            return

        if self.is_current_step_dirty():
            parent = None
            try:
                parent = self.viewer.window._qt_window
            except Exception:
                parent = None
            box = QMessageBox(parent)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Unsaved parameter changes")
            box.setText("You changed parameters on this step but haven't processed them.")
            box.setInformativeText(
                "Going back won't keep un-processed edits. Discard them, or "
                "process this step now (which clears any later results and "
                "recomputes)?"
            )
            discard_btn = box.addButton("Discard changes", QMessageBox.DestructiveRole)
            process_btn = box.addButton("Process now", QMessageBox.AcceptRole)
            cancel_btn = box.addButton("Cancel", QMessageBox.RejectRole)
            box.setDefaultButton(cancel_btn)
            box.exec_()
            clicked = box.clickedButton()

            if clicked == cancel_btn:
                return
            if clicked == process_btn:
                # Commit the edits by processing; stay on this step (processing
                # is asynchronous, so we don't also navigate). The user can go
                # back again once it completes.
                self.execute_processing_step()
                return
            # Discard: revert to the committed values, then navigate.
            self._revert_current_edits()

        self.current_step["value"] -= 1
        self.create_step_widgets(self.processing_steps[self.current_step["value"]])

    @staticmethod
    def _migrate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Bring a config to the current schema, reporting what changed.

        Logged (unlike the silent migration inside ProcessingStrategy) because
        this runs once per opened image, so the report is useful rather than
        noise -- and because a user who opens an old project should be able to
        see in the log that its keys were translated.
        """
        try:
            from ..fluorescence_module.config_migration import normalise_config
            return normalise_config(config)
        except Exception as exc:
            print(f"[gui] config migration unavailable: {exc}")
            return config

    def _persist_project_config(self) -> bool:
        """Write `self.config` back to the sample's own YAML. True if written.

        `strategy.save_config` writes the RUN PROVENANCE record inside the
        results directory. `app_launch` opens a sample by reading the first
        .yaml in the sample FOLDER, so anything adopted here that is not written
        there is lost the moment the viewer closes: the widened bounds reached
        the provenance file and the editor rebuilt from a config that had never
        heard of the parameter.

        The same gap applies to an accepted reconcile with real parameter
        changes -- the merged config was adopted in memory and recorded in the
        provenance, and the next open reconciled all over again from the stale
        sample config.

        Written to a temporary file and renamed, so an interrupted write cannot
        leave a sample folder with a truncated config -- the one file that makes
        the folder recognisable as a sample.
        """
        try:
            names = [f for f in sorted(os.listdir(self.inputdir))
                     if f.lower().endswith((".yaml", ".yml"))]
        except OSError as exc:
            print(f"[config] could not list {self.inputdir}: {exc}")
            return False
        if not names:
            print(f"[config] no config found in {self.inputdir}; nothing to update")
            return False

        target = os.path.join(self.inputdir, names[0])
        partial = target + ".part"
        try:
            with open(partial, "w") as handle:
                yaml.safe_dump(self.config, handle, default_flow_style=False,
                               sort_keys=False)
            os.replace(partial, target)
            return True
        except Exception as exc:
            try:
                if os.path.isfile(partial):
                    os.remove(partial)
            except OSError:
                pass
            print(f"[config] could not update {os.path.basename(target)}: {exc}")
            return False

    def _ensure_config_canonical(self, step_index: int) -> bool:
        """Make the config match the current pipeline before (re)processing.

        Returns True if processing may proceed now (the config was already
        current). Returns False if it must not proceed right now — because the
        user cancelled, a config problem was surfaced, or the config was just
        canonized (in which case the widgets have been rebuilt to the current
        pipeline and the user is asked to review and press Process again).

        This is the ONLY place staleness is enforced. Viewing results and
        cross-channel analysis never reach here, so an old run can always be
        inspected or analysed without reconciling; only actual computation is
        gated. A config that merely holds different-but-valid tuned values
        reconciles clean and is left untouched, so normal use has no friction.
        """
        try:
            from .config_library import reconcile, ConfigLibraryError
            from .reconcile_dialog import confirm_reconcile
        except Exception as exc:
            # Don't silently skip: say so, but don't block processing either.
            print(f"[reconcile] unavailable, proceeding without canonize: {exc}")
            return True

        parent = None
        try:
            parent = self.viewer.window._qt_window
        except Exception:
            parent = None

        try:
            recon = reconcile(self.config)
        except ConfigLibraryError as exc:
            QMessageBox.critical(
                parent, "Config error",
                "This run's config can't be reconciled against the current "
                f"pipeline, so it can't be re-processed:\n\n{exc}"
            )
            return False
        except Exception as exc:
            QMessageBox.critical(parent, "Config error", str(exc))
            return False

        if recon.is_clean:
            return True  # already current — nothing to do

        if not recon.affects_results:
            # Only the DEFINITIONS drifted: bounds, labels, descriptions. The
            # values are untouched, so a re-run produces exactly what it would
            # have produced -- there is nothing to warn about and nothing to
            # invalidate. Adopt it silently.
            #
            # This is what a widened maximum needs. `merged` is a deepcopy of
            # the reference with the source's values carried over, so it has
            # been correct all along; `is_clean` used to report True whenever
            # nothing value-affecting had changed, and the caller discarded it.
            # A project set up before a bound moved could therefore never see
            # the new range.
            self.config = recon.merged
            self.initial_config = copy.deepcopy(recon.merged)
            self.strategy.config = self.config
            # BOTH files: the provenance record, and the sample's own config,
            # which is what app_launch reads when the sample is next opened.
            try:
                self.strategy.save_config(self.config)
            except Exception as exc:
                print(f"[config] could not record refreshed definitions: {exc}")
            self._persist_project_config()
            for line in recon.summary_lines():
                print(f"[config] {line}")
            if self.current_step_method:
                self.create_step_widgets(self.current_step_method)
            return True

        # Did the user have un-processed edits on the current step when they hit
        # Process? If so, once we canonize we can run those edits straight away
        # instead of forcing a second Process click (the reported friction: the
        # rebuild below resets the step's baseline, so the pending edit would
        # otherwise stop reading as "dirty" and the Process button would grey
        # out until the user re-typed the same change).
        was_dirty = self.is_current_step_dirty()

        # Work out which *already-computed* results this reconcile invalidates.
        # A step is stale if its parameters changed (clamped / reset / type
        # changed / a param added or removed) or the step itself was added by the
        # current pipeline. Everything from the earliest such step onward depends
        # on it, so those results must go — otherwise old outputs would look like
        # they came from the new parameters. (Steps the reference dropped aren't
        # current-pipeline steps, so they don't map to a result to invalidate.)
        changed_keys = {c.step for c in recon.param_changes} | set(recon.added_steps)
        step_keys = [self.strategy.get_config_key(s["method"])
                     for s in self.strategy.steps]
        changed_indices = [i for i, k in enumerate(step_keys) if k in changed_keys]
        earliest = min(changed_indices) if changed_indices else None

        completed = self.strategy.get_last_completed_step()  # count of leading done
        # Only completed steps have results to lose; the affected, data-bearing
        # steps run from `earliest` up to the last completed step.
        affected = []
        if earliest is not None and earliest < completed:
            for i in range(earliest, completed):
                method = self.processing_steps[i]
                affected.append(
                    f"Step {i + 1}: {self.step_display_names.get(method, method)}"
                )

        # Stale: show the diff (and, if results will be lost, warn explicitly).
        if not confirm_reconcile(parent, recon, context="Re-processing this run",
                                 impact_lines=affected or None):
            return False  # user cancelled — leave the config and results untouched

        # Adopt the reconciled config and persist it.
        self.config = recon.merged
        self.initial_config = copy.deepcopy(recon.merged)
        self.strategy.config = self.config
        # Write it back to the sample's own config as well, not only to the run
        # provenance: `app_launch` reads the sample folder's YAML, so a config
        # adopted here but not written there means the next open reconciles the
        # same stale file again and re-asks a question already answered.
        self._persist_project_config()

        # Delete the now-invalid results (the earliest changed step and every
        # step after it), so nothing on disk claims to come from the new params.
        if affected:
            for i in range(earliest, self.num_steps):
                self.cleanup_step(i + 1)  # cleanup_step_artifacts is 1-based
            # Land on the first step that must be recomputed so Process is
            # enabled there (frontier == cursor == earliest after the cleanup).
            self.current_step["value"] = earliest

        try:
            self.strategy.save_config(self.config)
        except Exception as exc:
            print(f"[reconcile] could not persist canonized config: {exc}")

        idx = self.current_step["value"]
        if idx < self.num_steps:
            self.create_step_widgets(self.processing_steps[idx])
        else:
            self.clear_current_widgets()
        self.params_edited.emit()  # let the nav buttons re-evaluate

        if affected:
            QMessageBox.information(
                parent, "Config updated \u2014 stale results cleared",
                "This run was tuned on a different pipeline version, so its config "
                "was updated to match the current one and the now-outdated results "
                f"for {len(affected)} step(s) were cleared. Your tuned values were "
                "kept wherever they still apply. Review the parameters, then press "
                "Process to recompute from the first cleared step."
            )
            return False  # results were lost — make the user review + click again

        if was_dirty:
            # The user was mid-tune and pressed Process; they've now approved the
            # config diff, and their edits survived into the canonical config.
            # Run them now rather than forcing another click.
            return True

        QMessageBox.information(
            parent, "Config updated to current pipeline",
            "This run was tuned on a different pipeline version, so its config "
            "was updated to match the current one. Your tuned values were kept "
            "wherever they still apply. No completed results were affected. "
            "Review the parameters, then press Process to run."
        )
        return False

    # --- Step Execution ---

    def execute_processing_step(self) -> None:
        """
        Triggers execution of the current step in a background thread.
        Handles UI locking, log redirection, and parameter validation.
        """
        # ---> NEW: Prevent overwriting an active thread (Double-click protection)
        if getattr(self, 'worker', None) and self.worker.isRunning():
            print("A processing step is already running. Ignoring click.")
            return

        step_index = self.current_step["value"]
        if step_index >= self.num_steps:
            return

        logical_step = self.processing_steps[step_index]
        step_display = self.step_display_names.get(
            logical_step, f"Step {step_index + 1}"
        )

        # Before computing, ensure a stale config is brought up to the current
        # pipeline (interaction analysis is exempt — repeatable, not in the
        # canonical chain). If this canonizes or is cancelled, don't process now.
        if logical_step != "execute_interaction_analysis":
            if not self._ensure_config_canonical(step_index):
                return

        current_values = self.get_current_values()
        
        # Validation for Interaction Step
        if logical_step == "execute_interaction_analysis":
            if not current_values.get("target_channel_folder"):
                QMessageBox.warning(
                    None, "Missing Input",
                    "Please select a Reference Channel folder first."
                )
                return

        # Prepare Execution
        self.strategy.save_config(self.config)
        
        # Clear results this run invalidates.
        #
        # `step_index` is 0-based and cleanup_step is 1-based, so `step_index + 1`
        # is THIS step, not the next one -- the old loop started there and so
        # removed the current step's own layer moments before re-adding it. Its
        # files are still deleted (a step that fails partway must not leave a
        # previous run's artifact behind, which would make resume think it
        # succeeded), but its layers are kept so the re-add updates in place.
        self.cleanup_step(step_index + 1, keep_layers=True)
        for i in range(step_index + 2, self.num_steps + 1):
            self.cleanup_step(i)
        
        # Ensure state has image reference
        if 'original_volume_ref' not in self.strategy.intermediate_state:
            self.strategy.intermediate_state['original_volume_ref'] = self.image_stack

        # Setup UI
        # ONLY clear log here, when starting a new run explicitly
        if self.log_widget:
            self.log_widget.clear()
            self.log_widget.append(f"--- Starting {step_display} ---\n")
        
        self._set_ui_busy(True)
        self.process_started.emit()

        # Redirect Stdout
        self.output_stream = OutputStream()
        self.output_stream.text_written.connect(self._append_log)
        sys.stdout = self.output_stream
        sys.stderr = self.output_stream

        # Start Worker
        # Record existing children first so _stop_worker_safely can later kill
        # only the pool workers *this* step spawns.
        self._worker_child_baseline = _snapshot_child_pids()
        lifecycle("worker.start", step=step_index,
                  baseline_children=len(self._worker_child_baseline))
        self.worker = StepWorker(
            self.strategy, step_index, self.image_stack, current_values
        )
        self.worker.setParent(self)
        self.worker.finished_signal.connect(self._on_step_finished)
        self.worker.start()

    def _append_log(self, text: str) -> None:
        """Appends text to the GUI log widget."""
        if not self.log_widget:
            return
        
        try:
            # C++ check: might raise RuntimeError if wrapped object deleted
            cursor = self.log_widget.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(text)
            self.log_widget.setTextCursor(cursor)
            self.log_widget.ensureCursorVisible()
        except RuntimeError:
            self.log_widget = None

    def _on_step_finished(self, success: bool) -> None:
        """Callback when the worker thread finishes."""
        # Restore Stdout immediately
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        
        try:
            self._set_ui_busy(False)
        except RuntimeError:
            return
        
        step_index = self.current_step["value"]
        logical_step = self.processing_steps[step_index]
        step_display = self.step_display_names.get(
            logical_step, f"Step {step_index + 1}"
        )

        if success:
            try:
                if self.log_widget:
                    self.log_widget.append(f"\n--- {step_display} COMPLETED ---")
            except RuntimeError:
                pass
            
            # Visualization Phase (Main Thread)
            try: 
                self.strategy.load_checkpoint_data(self.viewer, step_index + 1)
            except Exception as e:
                err_msg = f"\n!!! Visualization Failed !!!\n{str(e)}"
                try:
                    if self.log_widget:
                        self.log_widget.append(err_msg)
                except RuntimeError:
                    pass
                print(f"Viz Error: {e}")
                traceback.print_exc()

            self.strategy.save_config(self.config)
            
            if logical_step == "execute_interaction_analysis":
                try:
                    QMessageBox.information(
                        None, "Analysis Complete",
                        "Interaction analysis finished.\n"
                        "You can select another channel to compare against."
                    )
                except Exception:
                    pass
            else:
                self.current_step["value"] += 1
                if self.current_step["value"] < self.num_steps:
                    next_step = self.processing_steps[self.current_step["value"]]
                    self.create_step_widgets(next_step)
                else:
                    self.clear_current_widgets()
                    try:
                        QMessageBox.information(None, "Complete", "All steps finished.")
                    except Exception:
                        pass
        else:
            try:
                if self.log_widget:
                    self.log_widget.append(f"\n!!! {step_display} FAILED !!!")
                QMessageBox.warning(None, "Step Failed", "Check log for details.")
            except Exception:
                pass
        
        self.process_finished.emit()

    def _set_ui_busy(self, is_busy: bool) -> None:
        """Disables/Enables parameter widgets during processing."""
        if not self.current_widgets:
            return
        
        for dock in list(self.current_widgets.keys()):
            try:
                if dock.widget():
                    dock.widget().setEnabled(not is_busy)
            except RuntimeError:
                continue

    def clear_current_widgets(self) -> None:
        """Removes current parameter widgets from the viewer."""
        for dock in list(self.current_widgets.keys()):
            try:
                self.viewer.window.remove_dock_widget(dock)
            except Exception:
                pass
        self.current_widgets.clear()
        # No step is on screen now, so there is no baseline to compare against.
        # (create_step_widgets calls this first, then sets a fresh baseline; the
        # end-of-processing path calls it alone, and this keeps dirty == False.)
        self._active_config_key = None
        self._active_baseline = {}

    # --- Widget Creation ---

    def _reference_config(self) -> Dict[str, Any]:
        """The built-in reference config, loaded once per session.

        Cached because `_param_applies` is called for every parameter of every
        step and this reads a file from disk.
        """
        if getattr(self, "_ref_cache", None) is None:
            try:
                from .config_library import MODE, builtin_reference
                self._ref_cache = builtin_reference(MODE) or {}
            except Exception as exc:
                print(f"[gui] reference config unavailable: {exc}")
                self._ref_cache = {}
        return self._ref_cache

    def _reference_param(self, config_key: str, pname: str):
        """A parameter's DEFINITION from the built-in reference, or None.

        Definitions belong to the reference and values to the project. A
        project's config is a snapshot of the reference as it stood when the
        project was set up, so anything added since is absent from it -- and
        `_ensure_config_canonical` only reconciles when something is
        processed, deliberately, so viewing an old run stays frictionless.
        That leaves the reference as the only place to ask whether a parameter
        exists at all.
        """
        ref = self._reference_config()
        block = (ref.get(config_key) or {}).get("parameters") or {}
        pconf = block.get(pname)
        return pconf if isinstance(pconf, dict) else None

    def _param_defined(self, config_key: str, pname: str,
                       parameters: Any = None) -> bool:
        """Whether this parameter exists in the project's config or the reference."""
        if isinstance(parameters, dict) and isinstance(parameters.get(pname), dict):
            return True
        return self._reference_param(config_key, pname) is not None

    def _param_applies(self, config_key: str, pname: str, pconf: Any) -> bool:
        """Whether a parameter is meaningful at this image's rank.

        Rank comes from the ARRAY, not the config's mode or dimension block:
        the array is what the step will actually be handed.

        The `ndim` annotation is looked up in the PROJECT's config first and
        then in the built-in reference. The reference matters because that is
        where the annotation lives: a saved project's config was written before
        `ndim` existed, so consulting only `pconf` meant every existing project
        still showed the control -- the annotation was on disk with nothing able
        to see it. Definitions belong to the reference and values to the
        project, and this is the one piece of definition read at render time.

        A parameter annotated in neither applies at every rank, so this stays
        opt-in and an un-annotated parameter behaves exactly as before.
        """
        want = pconf.get("ndim") if isinstance(pconf, dict) else None
        if want is None:
            ref = self._reference_config()
            ref_block = (ref.get(config_key) or {}).get("parameters") or {}
            ref_pconf = ref_block.get(pname)
            if isinstance(ref_pconf, dict):
                want = ref_pconf.get("ndim")
        if want is None:
            return True
        try:
            return int(want) == int(getattr(self.image_stack, "ndim", 0))
        except (TypeError, ValueError):
            return True     # unreadable annotation: show it rather than hide it

    def _add_soma_source_widget(self, layout, config_key: str,
                                parameters: dict) -> bool:
        """Add the soma-source dropdown. True if a source is currently set.

        A dropdown of real candidates rather than free text: the only valid
        answers are channels whose matching segment has reached soma
        extraction, and typing a channel name that has not been processed
        would produce a run that fails at step 3 with nothing to show for the
        wait.

        Failing to build the control leaves the parameter unset and every
        normal parameter visible, so a project whose layout this cannot read
        behaves exactly as it did before.
        """
        from PyQt5.QtWidgets import QComboBox, QLabel  # type: ignore

        pconf = parameters.get("soma_source_channel")
        if not isinstance(pconf, dict):
            pconf = self._reference_param(config_key, "soma_source_channel") or {}
        current = str(pconf.get("value") or "").strip()

        try:
            from .soma_source import candidate_channels
            candidates = [name for name, _path
                          in candidate_channels(self.processed_dir)]
        except Exception as exc:
            print(f"[gui] could not list soma source channels: {exc}")
            candidates = []

        if not candidates and not current:
            return False

        label = QLabel(str(pconf.get("label") or "Take somas from"))
        label.setStyleSheet("font-weight: bold;")
        layout.addWidget(label)

        box = QComboBox()
        box.addItem("Find somas in this channel", "")
        for name in candidates:
            box.addItem(name, name)
        # A channel recorded in the config but no longer offering results is
        # kept in the list and selected, so the run's provenance stays visible
        # instead of silently reverting to local extraction.
        if current and current not in candidates:
            box.addItem(f"{current}  (no results found)", current)
        index = box.findData(current)
        box.setCurrentIndex(index if index >= 0 else 0)
        layout.addWidget(box)

        note = QLabel(
            "Cell bodies come from the chosen channel's segmentation of this "
            "same image, keeping only those inside this channel's cells. The "
            "parameters below are unused while a channel is chosen."
            if candidates else
            "No other channel has reached soma extraction for this image yet."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(note)

        def _changed(_index: int, key=config_key) -> None:
            self.parameter_changed(key, "soma_source_channel",
                                   box.currentData() or "")
            # Rebuild so the parameters below appear or disappear with the
            # choice, matching how the absolute-threshold switch behaves.
            if self.current_step_method:
                self.create_step_widgets(self.current_step_method)

        box.currentIndexChanged.connect(_changed)
        return bool(current)

    def create_step_widgets(self, step_method_name: str) -> None:
        """Generates parameter widgets for the given step."""
        # Temporarily lock the window size to prevent macOS from shrinking 
        # the window when the old parameter dock is removed.
        try:
            qt_win = self.viewer.window._qt_window
            old_min = qt_win.minimumSize()
            qt_win.setMinimumSize(qt_win.size())
        except Exception:
            qt_win = None

        self.clear_current_widgets()
        self.parameter_values = {}
        self.current_step_method = step_method_name # Store for dynamic refresh
        
        config_key = self.strategy.get_config_key(step_method_name)
        step_display = self.step_display_names.get(step_method_name, step_method_name)

        # Record the values these widgets are built from. Because edits are never
        # allowed to survive navigation (see go_back), this baseline is always the
        # last-committed state of the step, so "live vs baseline" is a reliable
        # dirty check.
        self._active_config_key = config_key
        self._active_baseline = self._params_snapshot(config_key)
        
        # Special Case: Interaction Analysis
        if step_method_name == "execute_interaction_analysis":
            self.create_interaction_widgets(step_display, config_key)
            return

        # Generic Case: Params from Config
        step_config = self.config.get(config_key, {})
        parameters = step_config.get("parameters", {}) if isinstance(step_config, dict) else {}

        scroll_w = QWidget()
        scroll_l = QVBoxLayout(scroll_w)
        lbl = QLabel(f"Parameters: {step_display}")
        lbl.setStyleSheet("font-weight: bold;")
        scroll_l.addWidget(lbl)

        # Somas can be taken from another channel instead of found in this
        # one's signal. The control is a dropdown of channels whose MATCHING
        # segment -- this same full image, or this same region -- has reached
        # soma extraction, because a channel that has not been processed
        # cannot seed one that is being processed now.
        # VALUES come from the config, not from whichever widgets are rendered.
        # `get_current_values()` is what a step is handed, and it used to be
        # populated inside the render loop -- so every display filter below
        # silently removed a parameter from the run. Hiding the soma parameters
        # while seeding from another channel made step 3 fail with
        # "required parameter 'min_fragment_size' is missing from the step
        # config", and the rank filter has the same latent flaw for any hidden
        # parameter a step requires. Showing and supplying are separate
        # decisions; only the first is made below.
        if isinstance(parameters, dict):
            for _pname, _pconf in parameters.items():
                if isinstance(_pconf, dict):
                    self.parameter_values[_pname] = _pconf.get("value")

        #
        # Whether the control belongs here is answered by the REFERENCE as well
        # as the project's own config. A definition lives in the reference and a
        # value in the project, so a project set up before this parameter
        # existed has no trace of it -- and its config is only reconciled when
        # something is processed, which is far too late for a control that
        # decides how step 3 runs. Same shape as the `ndim` annotation, which
        # had to be read from the reference for the same reason.
        seeded_externally = False
        if self._param_defined(config_key, "soma_source_channel", parameters):
            seeded_externally = self._add_soma_source_widget(
                scroll_l, config_key, parameters)

        if isinstance(parameters, dict):
            # Check if Absolute mode is enabled
            is_absolute = False
            if "use_absolute_thresholds" in parameters:
                is_absolute = bool(parameters["use_absolute_thresholds"].get("value", False))

            for pname, pconf in parameters.items():
                # The dropdown above is this parameter's control.
                if pname == "soma_source_channel":
                    continue
                # Seeding from another channel makes every soma-finding
                # parameter here unused: nothing in this step reads them. They
                # are hidden rather than left editable and inert, for the same
                # reason rank-inapplicable parameters are -- an editable
                # control that cannot affect the result invites tuning it.
                if seeded_externally:
                    continue
                # Mutually exclusive parameter filtering
                if pname in["scale_profiles", "scale_profiles_percentile"] and is_absolute:
                    continue
                if pname == "scale_profiles_absolute" and not is_absolute:
                    continue
                # Rank filtering: a parameter declaring `ndim:` in the config is
                # shown only at that rank. There is one default config for both
                # ranks, so it necessarily contains parameters that mean nothing
                # at the other one -- Z-anisotropy erosion on a single plane, for
                # example. They stay IN the file (the 3D path still needs their
                # values, and convention 7 keeps rank-varying settings in the
                # YAML where they are visible) but showing them invites tuning a
                # control that cannot affect the result.
                if not self._param_applies(config_key, pname, pconf):
                    continue

                try:
                    cb = lambda val, k=config_key, p=pname: self.parameter_changed(k, p, val)
                    w = create_parameter_widget(pname, pconf, cb)
                    if w: 
                        scroll_l.addWidget(w.native)
                        self.parameter_values[pname] = pconf.get('value')
                except Exception:
                    pass

        self._dock_widget(scroll_w, step_display)
        
        # Release the minimum size lock on the next event loop tick
        if qt_win:
            QTimer.singleShot(0, lambda: qt_win.setMinimumSize(old_min))

    def create_interaction_widgets(self, step_display: str, config_key: str) -> None:
        """Creates specialized widgets for the Interaction Analysis step."""
        scroll_w = QWidget()
        layout = QVBoxLayout(scroll_w)
        
        lbl = QLabel(f"{step_display}")
        lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(lbl)
        
        desc = QLabel(
            "Select a processed project folder for another channel (e.g. Plaques/Vessels)."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # File Selection Button
        self.btn_select_ref = QPushButton("Analyze with Other Channels...")
        self.btn_select_ref.setStyleSheet("padding: 8px; font-weight: bold;")
        self.btn_select_ref.clicked.connect(self.select_reference_channel)
        layout.addWidget(self.btn_select_ref)
        
        self.lbl_ref_path = QLabel("No reference selected")
        self.lbl_ref_path.setStyleSheet("color: #666; font-style: italic; margin-bottom: 10px;")
        self.lbl_ref_path.setWordWrap(True)
        layout.addWidget(self.lbl_ref_path)

        # Other parameters (bools)
        step_config = self.config.get(config_key, {})
        parameters = step_config.get("parameters", {})
        
        if isinstance(parameters, dict):
            for pname, pconf in parameters.items():
                if pname == "target_channel_folder":
                    continue
                try:
                    cb = lambda val, k=config_key, p=pname: self.parameter_changed(k, p, val)
                    w = create_parameter_widget(pname, pconf, cb)
                    if w: 
                        layout.addWidget(w.native)
                        self.parameter_values[pname] = pconf.get('value')
                except Exception:
                    pass

        self._dock_widget(scroll_w, step_display)

    def select_reference_channel(self) -> None:
        """Opens dialog to select reference project folder."""
        start_dir = os.path.dirname(self.inputdir)
        folder = QFileDialog.getExistingDirectory(
            None, "Select Reference Channel Project", start_dir
        )
        
        if folder:
            self.parameter_values['target_channel_folder'] = folder
            display_name = os.path.basename(folder)
            self.lbl_ref_path.setText(f"Selected: {display_name}")
            self.lbl_ref_path.setStyleSheet("color: #2E8B57; font-weight: bold;")

    def _dock_widget(self, widget: QWidget, name: str) -> None:
        """Docks the given widget into the Napari window."""
        # Keep the parameter panel at its natural height and scroll when the
        # dock is short, instead of compressing the controls. SetMinimumSize
        # ties the content's minimum to its contents so the scroll area shows a
        # scrollbar rather than squishing; the trailing stretch soaks up extra
        # space when the dock is tall so the widgets don't splay out either.
        lay = widget.layout()
        if isinstance(lay, QVBoxLayout):
            lay.addStretch(1)
            lay.setSizeConstraint(QLayout.SetMinimumSize)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        # Open the dock wide enough to show the widest content (e.g. the
        # multi-column scale table) without the user having to drag it wider.
        # Use the *minimum* size hint (SetMinimumSize + the table's explicit
        # minimum width make it reflect the real column widths, unlike
        # QTableView.sizeHint). Capped so simple steps stay compact.
        try:
            want = max(widget.minimumSizeHint().width(), widget.sizeHint().width())
        except Exception:
            want = 0
        scroll.setMinimumWidth(max(360, min(want + 36, 720)))
        dock = self.viewer.window.add_dock_widget(
            scroll, area="right", name=f"Step: {name}"
        )
        self.current_widgets[dock] = scroll

    def parameter_changed(
        self, config_key: str, param_name: str, value: Any
    ) -> None:
        """Updates internal config when UI widgets change."""
        try:
            # A parameter added to the reference after this project was set up
            # is not in its config, and the assignment below would raise into
            # the bare `except` -- the control would look live and change
            # nothing. Seed the definition from the reference first. An empty
            # default means the previous behaviour, so seeding decides nothing.
            block = (self.config.setdefault(config_key, {})
                     .setdefault("parameters", {}))
            if not isinstance(block.get(param_name), dict):
                seed = self._reference_param(config_key, param_name)
                block[param_name] = dict(seed) if seed else {"value": value}
            self.config[config_key]["parameters"][param_name]["value"] = value
            self.parameter_values[param_name] = value

            # If the threshold mode toggle was flipped, trigger a deferred redraw 
            # to swap the tables without crashing the PyQT event loop
            if param_name == "use_absolute_thresholds":
                QTimer.singleShot(0, lambda: self.create_step_widgets(self.current_step_method))
        except Exception:
            pass
        # Let the navigation controls re-evaluate (this step may now be "dirty").
        self.params_edited.emit()

    def get_current_values(self) -> Dict[str, Any]:
        """Returns the current state of parameters."""
        return self.parameter_values.copy()