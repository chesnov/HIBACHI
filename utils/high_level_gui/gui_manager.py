import os
import sys
import gc
import copy
import json
import time
import traceback
import yaml  # type: ignore
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
from skimage.draw import polygon as skimage_polygon  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QMessageBox, QWidget, QVBoxLayout, QScrollArea, QLabel,
    QTextEdit, QProgressBar, QApplication, QPushButton, QFileDialog, QDockWidget,
    QInputDialog
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt, QTimer  # type: ignore
from PyQt5.QtGui import QTextCursor  # type: ignore
import napari  # type: ignore

# --- Relative Imports ---
try:
    from ..module_3d._3D_strategy import FluorescenceStrategy
    from ..module_2d._2D_strategy import Fluorescence2DStrategy
    from .processing_strategies import ProcessingStrategy
    from .helper_funcs import create_parameter_widget
except ImportError as e:
    print(f"Error importing dependencies in gui_manager.py: {e}")
    raise


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
    for worker in list(_orphan_threads):
        try:
            if worker is not None and worker.isRunning():
                worker.terminate()
                worker.wait(200) # Give it time to cleanly exit C++ scope
        except Exception:
            pass
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
        try:
            success = self.strategy.execute_step(
                step_index=self.step_index,
                viewer=None,  # Viewer is handled by main thread, not worker
                image_stack_or_none=self.image_stack,
                params=self.params
            )
            self.finished_signal.emit(success)
        except Exception as e:
            traceback.print_exc()
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

    def __init__(
        self,
        viewer: napari.Viewer,
        config: Dict[str, Any],
        image_stack: np.ndarray,
        file_loc: str,
        processing_mode: str,
        project_manager: Any = None,
    ):
        super().__init__()
        self.viewer = viewer
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
        self.worker: Optional[StepWorker] = None

        # ROI / sub-region state
        # These are set when the user confirms a polygon crop.
        # _full_* refs allow returning to the full-image session at any time.
        self.roi_active: bool = False
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
        # "Save to Library…" — the primary way the library grows from a config
        # dialled in here in the tuning view.
        self._init_library_dock()

        # Project Paths
        self.inputdir = os.path.dirname(self.file_loc)
        basename = os.path.basename(self.file_loc)
        self.basename = os.path.splitext(basename)[0]
        self.processed_dir = os.path.join(
            self.inputdir, f"{self.basename}_processed_{self.processing_mode}"
        )

        # Spacing
        self.spacing: Union[Tuple[float, float, float], Tuple[float, float]] = (1.0, 1.0, 1.0)
        self.z_scale_factor: float = 1.0
        self._calculate_spacing()

        # Initialize Strategy
        try:
            strategy_class = {
                'fluorescence': FluorescenceStrategy,
                'fluorescence_2d': Fluorescence2DStrategy
            }.get(self.processing_mode)

            if not strategy_class:
                raise ValueError(f"Unsupported mode: {self.processing_mode}")

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
        if not self._try_load_existing_roi_session():
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

    def _init_library_dock(self) -> None:
        """Add a small dock with a 'Save to Library…' button.

        The library is the cross-project home for dialled-in configs; saving from
        the tuning view is the primary way it grows. The button is best-effort:
        if anything about the dock fails it must never block the segmentation UI
        from opening.
        """
        try:
            container = QWidget()
            lay = QVBoxLayout(container)
            lay.setContentsMargins(5, 5, 5, 5)
            btn = QPushButton("\U0001f4be  Save to Library\u2026")
            btn.setToolTip(
                "Save the current parameters as a reusable preset in your config "
                "library (shared across projects). Image- and run-specific data "
                "(dimensions, saved thresholds) are stripped; the pipeline version "
                "is kept as provenance."
            )
            btn.clicked.connect(self.save_current_to_library)
            lay.addWidget(btn)
            self.viewer.window.add_dock_widget(
                container, area="right", name="Config Library"
            )
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[library] could not add Save-to-Library dock: {exc}")

    def save_current_to_library(self) -> None:
        """Save the live config as a library preset (§5.4).

        ``self.config`` is kept in lockstep with the widgets by
        ``parameter_changed`` (which writes each edited value straight back into
        ``self.config[step]['parameters'][param]['value']``), so it already
        reflects the current widget state. ``sanitize_for_library`` strips
        image/run-specific keys and keeps ``mode`` + ``hibachi_version``.
        """
        from . import config_library as cl
        from .config_library import ConfigLibraryError, ConfigModeError

        # Stamp provenance if the working config doesn't already carry it, so the
        # saved preset records which pipeline version tuned it.
        config = self.config
        if not config.get("hibachi_version"):
            try:
                from .processing_strategies import _hibachi_version_stamp
                config = dict(config)
                config["hibachi_version"] = _hibachi_version_stamp()
            except Exception as exc:  # provenance is best-effort, never fatal
                print(f"[library] could not stamp version: {exc}")
                config = self.config

        default_name = f"{self.basename} ({self.processing_mode})"
        name, ok = QInputDialog.getText(
            None, "Save to Library", "Preset name:", text=default_name
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        try:
            entry = cl.save_to_library(config, name)
        except FileExistsError:
            reply = QMessageBox.question(
                None, "Already exists",
                f"A library config named '{name}' already exists.\n\nOverwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            try:
                entry = cl.save_to_library(config, name, overwrite=True)
            except (ConfigLibraryError, OSError) as exc:
                QMessageBox.critical(None, "Config error", str(exc))
                return
        except ConfigModeError as exc:
            QMessageBox.critical(
                None, "Config error",
                f"This config has no valid 'mode', so it can't be saved:\n\n{exc}"
            )
            return
        except (ConfigLibraryError, OSError) as exc:
            QMessageBox.critical(None, "Config error", str(exc))
            return

        QMessageBox.information(
            None, "Saved to Library",
            f"Saved '{entry.name}' to your config library:\n\n{entry.path}"
        )

    # =========================================================================
    # ROI / SUB-REGION SELECTION
    # =========================================================================

    # --- Shared helpers ---

    def _get_strategy_class(self):
        """Returns the strategy class for the current processing mode."""
        return {
            'fluorescence': FluorescenceStrategy,
            'fluorescence_2d': Fluorescence2DStrategy,
        }.get(self.processing_mode)

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
        """
        Writes a cropped, polygon-masked copy of *src* to *out_path* and
        returns an 'r+' memmap handle to it.

        *z_polygons* maps global Z indices to YX polygon arrays (full-image
        coordinates).  For each Z slice in the crop the nearest defined polygon
        is used — nearest-neighbour interpolation between defined levels.  This
        supports three cases uniformly:

          • 2D image          — single entry {0: polygon_yx}.
          • 3D full-Z extrude — single entry {any_z: polygon_yx}; the same
                                 mask is applied to every slice.
          • 3D multi-polygon  — one entry per drawn Z level; slices between
                                 defined levels get the nearest polygon.

        Slices before the first defined Z use the first polygon; slices after
        the last defined Z use the last polygon (no extrapolation to zeros).

        Args:
            src:           Full-resolution image (2-D or 3-D array/memmap).
            y0,x0,y1,x1:  YX bounding box in full-image pixel coordinates.
            z_polygons:    Dict {global_z: polygon_yx (N×2, full-image coords)}.
            out_path:      Destination path for the output .dat memmap.
            z0_crop:       First Z slice index to include (3D only).
            z1_crop:       One-past-last Z slice index (3D only; None = end).

        Returns:
            np.memmap opened in 'r+' mode at *out_path*.
        """
        is_3d = src.ndim == 3
        crop_h, crop_w = y1 - y0, x1 - x0

        if is_3d:
            if z1_crop is None:
                z1_crop = src.shape[0]
            crop_depth = z1_crop - z0_crop
            crop_shape = (crop_depth, crop_h, crop_w)
        else:
            crop_shape = (crop_h, crop_w)

        crop_mm = np.memmap(out_path, dtype=src.dtype, mode='w+', shape=crop_shape)

        sorted_zs = sorted(z_polygons.keys())

        def _mask_for_z(global_z: int) -> np.ndarray:
            """Returns a boolean crop-local mask for the nearest polygon."""
            nearest_z = min(sorted_zs, key=lambda z: abs(z - global_z))
            poly = z_polygons[nearest_z] - np.array([y0, x0], dtype=float)
            rr, cc = skimage_polygon(poly[:, 0], poly[:, 1],
                                     shape=(crop_h, crop_w))
            m = np.zeros((crop_h, crop_w), dtype=bool)
            m[rr, cc] = True
            return m

        if is_3d:
            print(f"  [ROI] Building 3D crop "
                  f"({crop_depth} slices × {crop_h} × {crop_w})…")
            # Cache masks: if there is only one polygon defined all slices share
            # the same mask — avoid rebuilding it 192 times.
            mask_cache: Dict[int, np.ndarray] = {}
            for local_z in range(crop_depth):
                global_z = z0_crop + local_z
                nearest_z = min(sorted_zs, key=lambda z: abs(z - global_z))
                if nearest_z not in mask_cache:
                    mask_cache[nearest_z] = _mask_for_z(global_z)
                mask2d = mask_cache[nearest_z]
                slice_data = np.array(src[global_z, y0:y1, x0:x1])
                slice_data[~mask2d] = 0
                crop_mm[local_z] = slice_data
        else:
            mask2d = _mask_for_z(0)
            crop_mm[:] = src[y0:y1, x0:x1]
            crop_mm[~mask2d] = 0

        crop_mm.flush()
        return crop_mm

    def _build_roi_config(
        self,
        y0: int, x0: int, y1: int, x1: int,
        base_config: Dict[str, Any],
        z0: int = 0,
        z1: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Produces a deep-copied config with the physical dimensions rescaled to
        the crop extent.  All execute_* parameter blocks are taken verbatim.

        The YAMLs store *total* physical extent (not per-voxel size), so the
        crop dimensions scale linearly with pixel count:
            new_x_um = original_x_um × (crop_w / full_w)
        This leaves per-voxel spacing identical while making the config
        self-consistent for the smaller array.

        Args:
            y0,x0,y1,x1: YX bounding box in full-image pixel coordinates.
            base_config:  Original full-image config to copy from.
            z0, z1:       Z crop range (3D only).  z1=None means full Z range.
        """
        roi_config = copy.deepcopy(base_config)
        is_2d_mode = self.processing_mode.endswith('_2d')
        dim_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'

        orig_dims = base_config.get(dim_key, {'x': 1.0, 'y': 1.0, 'z': 1.0})
        orig_x = float(orig_dims.get('x', 1.0))
        orig_y = float(orig_dims.get('y', 1.0))

        full_h = self._full_image_stack.shape[-2]
        full_w = self._full_image_stack.shape[-1]

        new_dims = dict(orig_dims)
        new_dims['x'] = orig_x * ((x1 - x0) / full_w)
        new_dims['y'] = orig_y * ((y1 - y0) / full_h)

        if not is_2d_mode and 'z' in orig_dims:
            orig_z = float(orig_dims.get('z', 1.0))
            full_z = self._full_image_stack.shape[0]
            effective_z1 = z1 if z1 is not None else full_z
            new_dims['z'] = orig_z * ((effective_z1 - z0) / full_z)

        roi_config[dim_key] = new_dims
        return roi_config

    # --- Startup: detect an existing ROI session ---

    def _try_load_existing_roi_session(self) -> bool:
        """
        Checks whether a completed ROI session already exists for this image.
        Handles both the v1 JSON format (single polygon) and the current v2
        format (dict of Z→polygon entries).

        Returns True if the ROI session was loaded (caller must NOT call
        restore_from_checkpoint again), False otherwise.
        """
        roi_dir = self.processed_dir + "_roi"
        roi_json = os.path.join(roi_dir, "roi_polygon.json")
        if not os.path.exists(roi_json):
            return False

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

            # Save full-image references
            self._full_image_stack = self.image_stack
            self._full_processed_dir = self.processed_dir
            self._full_config = copy.deepcopy(self.config)

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
            roi_cfg_path = os.path.join(
                roi_dir, f"processing_config_{self.processing_mode}.yaml"
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
                                     call_restore=True)
            return True

        except Exception as exc:
            print(f"[ROI] Failed to load existing session: {exc}")
            traceback.print_exc()
            return False

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

        self.viewer.add_shapes(
            name=layer_name,
            shape_type='polygon',
            edge_color='yellow',
            face_color=[1, 1, 0, 0.08],
            edge_width=3,
        )
        self.viewer.layers[layer_name].mode = 'add_polygon'

        # Reset the reliable Z→polygon map and connect the data event.
        # The event fires each time the shapes data changes (polygon added/edited).
        # We record the current Z slice and the polygon count so we can detect
        # additions vs edits and avoid double-counting.
        self._roi_z_polygon_map: Dict[int, np.ndarray] = {}
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
            self._roi_z_polygon_map[z_slice] = poly_yx
            print(f"  [ROI] Polygon recorded at Z={z_slice} "
                  f"({len(self._roi_z_polygon_map)} total)")

        self.viewer.layers[layer_name].events.data.connect(_on_shapes_data_changed)

        if is_3d:
            msg = (
                "Draw polygons on any Z slices to define the 3D sub-region.\n\n"
                "  1. Scroll to a Z slice\n"
                "  2. Click to add vertices, press Escape to close the polygon\n"
                "  3. Scroll to the next relevant slice and repeat\n\n"
                "Each polygon is automatically tagged to the slice it was\n"
                "drawn on.  Drawing on only ONE slice extrudes that shape\n"
                "through the entire Z stack.\n\n"
                "When finished, click  ✓ Confirm ROI."
            )
        else:
            msg = (
                "Draw a polygon on the image to define the sub-region.\n\n"
                "  • Click to add vertices\n"
                "  • Press Escape to close the polygon\n\n"
                "When finished, click  ✓ Confirm ROI."
            )

        QMessageBox.information(None, "Draw ROI", msg)

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
        z_polygons: Dict[int, np.ndarray] = {}

        tracked = getattr(self, '_roi_z_polygon_map', {})
        if tracked:
            z_polygons = dict(tracked)
            print(f"  [ROI] Using tracked map: {len(z_polygons)} polygon(s) "
                  f"at Z={sorted(z_polygons.keys())}")
        else:
            # Fallback: parse Z from vertex arrays (works in 2D, unreliable
            # in 3D perspective mode — warn the user).
            for raw in shapes_layer.data:
                arr = np.array(raw, dtype=float)
                if arr.shape[1] == 3:
                    z_val = int(round(float(arr[:, 0].mean())))
                    poly_yx = arr[:, 1:]
                else:
                    z_val = 0
                    poly_yx = arr
                z_polygons[z_val] = poly_yx
            if is_3d and len(z_polygons) < len(shapes_layer.data):
                print("  [ROI] Warning: some polygons may share the same Z "
                      "index. Use '✏ Draw ROI' button to ensure reliable "
                      "per-slice tagging.")

        if not z_polygons:
            QMessageBox.warning(None, "Empty ROI", "No valid polygons found.")
            return

        # --- Union YX bounding box across all polygons ---
        all_yx = np.vstack(list(z_polygons.values()))
        y0 = max(0, int(np.floor(all_yx[:, 0].min())))
        x0 = max(0, int(np.floor(all_yx[:, 1].min())))
        y1 = min(img_h, int(np.ceil(all_yx[:, 0].max())) + 1)
        x1 = min(img_w, int(np.ceil(all_yx[:, 1].max())) + 1)
        crop_h, crop_w = y1 - y0, x1 - x0

        if crop_h < 10 or crop_w < 10:
            QMessageBox.warning(None, "ROI Too Small",
                                "The selected region is too small (< 10 px). "
                                "Please draw a larger polygon.")
            return

        # --- Z range ---
        # Single polygon → extrude through full Z stack.
        # Multiple polygons → crop to the Z range they span.
        if is_3d:
            sorted_zs = sorted(z_polygons.keys())
            if len(sorted_zs) == 1:
                z0_crop, z1_crop = 0, self.image_stack.shape[0]
                z_desc = "extruded through all Z"
            else:
                z0_crop = max(0, sorted_zs[0])
                z1_crop = min(self.image_stack.shape[0], sorted_zs[-1] + 1)
                z_desc = f"Z {z0_crop}–{z1_crop}  ({len(sorted_zs)} defined levels)"
        else:
            z0_crop, z1_crop = 0, None
            z_desc = "2D"

        # --- Confirmation dialog ---
        full_shape = self.image_stack.shape
        n_poly = len(z_polygons)
        poly_note = (f"{n_poly} polygon(s) defined" if n_poly > 1
                     else "1 polygon")
        reply = QMessageBox.question(
            None,
            "Confirm ROI",
            f"YX bounding box: rows {y0}–{y1}, cols {x0}–{x1}\n"
            f"Crop YX size: {crop_h} × {crop_w} px  "
            f"(full image: {full_shape[-2]} × {full_shape[-1]})\n"
            f"Z range: {z_desc}\n"
            f"Polygons: {poly_note}\n\n"
            f"This will clear any existing ROI session outputs and\n"
            f"restart from Step 1 on the cropped region.\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Save full-image references (idempotent if called again)
            if not self.roi_active:
                self._full_image_stack = self.image_stack
                self._full_processed_dir = self.processed_dir
                self._full_config = copy.deepcopy(self.config)

            roi_dir = self._full_processed_dir + "_roi"
            os.makedirs(roi_dir, exist_ok=True)

            # --- Persist polygon metadata (v2 format) ---
            roi_data = {
                "format": "v2",
                "z_polygons": [
                    {"z": int(z), "polygon_yx": poly.tolist()}
                    for z, poly in sorted(z_polygons.items())
                ],
                "bbox": {
                    "y0": y0, "x0": x0, "y1": y1, "x1": x1,
                    "z0": z0_crop, "z1": z1_crop,
                },
                "full_image_shape": list(full_shape),
            }
            with open(os.path.join(roi_dir, "roi_polygon.json"), 'w') as fh:
                json.dump(roi_data, fh, indent=2)

            # --- Build cropped + masked image memmap ---
            crop_path = os.path.join(roi_dir, "roi_image_crop.dat")
            crop_mm = self._build_crop_memmap(
                self._full_image_stack,
                y0, x0, y1, x1,
                z_polygons, crop_path,
                z0_crop=z0_crop, z1_crop=z1_crop,
            )

            # --- Build rescaled config ---
            roi_config = self._build_roi_config(
                y0, x0, y1, x1, self._full_config,
                z0=z0_crop, z1=z1_crop,
            )

            roi_cfg_path = os.path.join(
                roi_dir, f"processing_config_{self.processing_mode}.yaml"
            )
            with open(roi_cfg_path, 'w') as fh:
                yaml.safe_dump(roi_config, fh, default_flow_style=False,
                                sort_keys=False)

            # --- Remove the draw layer before reinitializing ---
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)

            # --- Switch pipeline to ROI mode (always restart from Step 1) ---
            self._switch_to_roi_mode(crop_mm, roi_dir, roi_config,
                                     call_restore=False)

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

        reply = QMessageBox.question(
            None,
            "Return to Full Image",
            "Return to full-image processing mode?\n\n"
            "ROI session outputs are preserved on disk and can be\n"
            "reloaded the next time you open this image.",
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
        self.image_stack = cropped_image
        self.processed_dir = roi_processed_dir
        self.config = roi_config
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

        # Notify connected slots (update_navigation_buttons in helper_funcs.py)
        # so the ◀ Previous Step button reflects the current step index.
        self.process_finished.emit()

        print(f"[ROI] Now in ROI mode — shape {self.image_stack.shape}, "
              f"dir: {os.path.basename(roi_processed_dir)}")

    def _switch_to_full_image_mode(self) -> None:
        """Tears down the ROI session and reinstates full-image processing."""
        if self._full_image_stack is None:
            return

        self._stop_worker_safely()

        self.roi_active = False
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
        self.restore_from_checkpoint()

        # Notify connected slots (update_navigation_buttons in helper_funcs.py)
        # so the ◀ Previous Step button reflects the resumed step index.
        self.process_finished.emit()

        print("[ROI] Returned to full-image mode.")

    def _stop_worker_safely(self) -> None:
        """Safely detaches a running worker thread so it doesn't crash the app on destruction."""
        if getattr(self, 'worker', None) and self.worker.isRunning():
            print("    [Thread] Detaching running background worker to prevent crash...")
            _register_quit_hook()  # Ensure cleanup happens at shutdown
            try:
                self.worker.finished_signal.disconnect()
                self.worker.error_signal.disconnect()
            except Exception:
                pass
            
            # PRESERVE data references so Loky background processes don't segfault!
            self.worker._preserved_strategy = self.strategy
            self.worker._preserved_stack = self.image_stack

            self.worker.setParent(None)
            _orphan_threads.append(self.worker)
            self.worker.finished.connect(lambda w=self.worker: _cleanup_orphan_thread(w))
            self.worker = None

    def shutdown_and_cleanup(self) -> None:
        """Forcefully clears all data references and Napari internal buffers."""
        # Check if worker is running BEFORE we stop it
        is_worker_running = getattr(self, 'worker', None) and self.worker.isRunning()
        self._stop_worker_safely()

        # 1. Clear Napari layers and buffers first
        if self.viewer:
            try:
                self.viewer.layers.clear() 
            except Exception:
                pass
        
        # 2. Clear strategy and large data references
        if hasattr(self, 'strategy') and self.strategy is not None:
            if hasattr(self.strategy, 'intermediate_state'):
                # CRITICAL FIX: DO NOT clear the dictionary in-place if a worker is running!
                # Doing so rips the memory-mapped file out from under the Loky process pool, causing a crash.
                if not is_worker_running:
                    self.strategy.intermediate_state.clear() 
            self.strategy = None
        
        self.image_stack = None 
        self.viewer = None 
        
        gc.collect()
        gc.collect()
        print("    [RAM] Deep cleanup complete. All heavy references released.")

    def _calculate_spacing(self) -> None:
        """Parses spacing from config or defaults to 1.0."""
        is_2d_mode = self.processing_mode.endswith("_2d")
        dim_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'
        dim = self.config.get(dim_key, {})
        
        try:
            tx = float(dim.get('x', 1.0))
            ty = float(dim.get('y', 1.0))
            tz = float(dim.get('z', 1.0))
        except (ValueError, TypeError):
            tx, ty, tz = 1.0, 1.0, 1.0

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

    def _initialize_layers(self) -> None:
        """Adds the original image to Napari."""
        self.viewer.layers.clear() 
        layer_name = f"Original stack ({self.processing_mode} mode)"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)
            
        scale = (
            (self.z_scale_factor, 1, 1) if self.image_stack.ndim == 3
            else (self.spacing[1], self.spacing[2])
        )
        self.viewer.add_image(
            self.image_stack, name=layer_name, scale=scale
        )

    def restore_from_checkpoint(self) -> None:
        """
        Checks for existing outputs and prompts user to Resume or Restart.
        """
        checkpoint_step = self.strategy.get_last_completed_step()
        
        if checkpoint_step > 0:
            # Load saved config
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

            msg = QMessageBox()
            if checkpoint_step == self.num_steps:
                msg.setText("All steps complete.")
                msg.setInformativeText("View results or restart from beginning?")
                view = msg.addButton("View Results", QMessageBox.YesRole)
                msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()
                
                if msg.clickedButton() == view:
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                else:
                    self._confirm_restart()
            else:
                msg.setText("Resume previous session?")
                msg.setInformativeText(f"Found data up to Step {checkpoint_step}.\n"
                                       f"Resume from Step {checkpoint_step + 1}?")
                res = msg.addButton("Resume", QMessageBox.YesRole)
                msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()
                
                if msg.clickedButton() == res:
                    # Restore state
                    self.strategy.intermediate_state['original_volume_ref'] = self.image_stack
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                    
                    if checkpoint_step < self.num_steps:
                        self.create_step_widgets(
                            self.processing_steps[checkpoint_step]
                        )
                else:
                    self._confirm_restart()
        else:
            self.create_step_widgets(self.processing_steps[0])

    def _confirm_restart(self) -> None:
        """Deletes old files and restarts from Step 1."""
        reply = QMessageBox.question(
            self.viewer.window._qt_window,
            "Confirm Restart",
            "This will delete all existing processing files for this mode.\nAre you sure?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.delete_all_checkpoint_files()
            self.current_step["value"] = 0
            self.strategy.intermediate_state = {}
            self.config = self.initial_config.copy()
            self.strategy.config = self.config
            self._initialize_layers()
            self.create_step_widgets(self.processing_steps[0])
        else:
            self.restore_from_checkpoint()

    def delete_all_checkpoint_files(self) -> None:
        """Helper to clear disk artifacts."""
        for _, path in self.strategy.get_checkpoint_files().items():
            self.strategy._remove_file_safely(path)

    def load_checkpoint_data(self, checkpoint_step: int) -> None:
        """Loads visualization data."""
        self.strategy.load_checkpoint_data(self.viewer, checkpoint_step)

    def cleanup_step(self, step_number: int) -> None:
        """Cleans artifacts for a specific step."""
        self.strategy.cleanup_step_artifacts(self.viewer, step_number)

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
        
        # If not repeating interaction analysis, clean subsequent steps
        if logical_step != "execute_interaction_analysis":
            for i in range(step_index + 1, self.num_steps + 1):
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

    # --- Widget Creation ---

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

        if isinstance(parameters, dict):
            # Check if Absolute mode is enabled
            is_absolute = False
            if "use_absolute_thresholds" in parameters:
                is_absolute = bool(parameters["use_absolute_thresholds"].get("value", False))

            for pname, pconf in parameters.items():
                # Mutually exclusive parameter filtering
                if pname in["scale_profiles", "scale_profiles_percentile"] and is_absolute:
                    continue
                if pname == "scale_profiles_absolute" and not is_absolute:
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setMinimumWidth(350) 
        dock = self.viewer.window.add_dock_widget(
            scroll, area="right", name=f"Step: {name}"
        )
        self.current_widgets[dock] = scroll

    def parameter_changed(
        self, config_key: str, param_name: str, value: Any
    ) -> None:
        """Updates internal config when UI widgets change."""
        try:
            self.config[config_key]["parameters"][param_name]["value"] = value
            self.parameter_values[param_name] = value

            # If the threshold mode toggle was flipped, trigger a deferred redraw 
            # to swap the tables without crashing the PyQT event loop
            if param_name == "use_absolute_thresholds":
                QTimer.singleShot(0, lambda: self.create_step_widgets(self.current_step_method))
        except Exception:
            pass

    def get_current_values(self) -> Dict[str, Any]:
        """Returns the current state of parameters."""
        return self.parameter_values.copy()