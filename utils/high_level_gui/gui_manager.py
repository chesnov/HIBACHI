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
    QTextEdit, QProgressBar, QApplication, QPushButton, QFileDialog, QDockWidget
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt  # type: ignore
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
        self.log_widget.setStyleSheet("font-family: monospace; font-size: 11px;")

        self.viewer.window.add_dock_widget(
            self.log_widget, area="right", name="Process Log"
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
        polygon_yx: np.ndarray,
        out_path: str,
    ) -> np.memmap:
        """
        Writes a cropped, polygon-masked copy of *src* to *out_path* and
        returns an 'r+' memmap handle to it.

        For 3D sources (Z, Y, X) the polygon is extruded through all Z slices.
        Pixels outside the polygon bounding-box crop are zero; pixels inside
        the bounding box but outside the polygon itself are also zeroed.

        For 3D data the copy and mask are applied one Z slice at a time to
        avoid loading the full crop into RAM (e.g. a 192 × 5000 × 5000
        float32 volume would be ~4 GB in one shot).

        Args:
            src:         Full-resolution image (2-D or 3-D numpy array /
                         memmap).
            y0,x0,y1,x1: Bounding box in full-image pixel coordinates.
            polygon_yx:  (N, 2) array of polygon vertices in full-image YX
                         coordinates.
            out_path:    Destination path for the .dat memmap file.

        Returns:
            np.memmap opened in 'r+' mode at *out_path*.
        """
        is_3d = src.ndim == 3
        crop_h, crop_w = y1 - y0, x1 - x0

        crop_shape = (src.shape[0], crop_h, crop_w) if is_3d else (crop_h, crop_w)
        crop_mm = np.memmap(out_path, dtype=src.dtype, mode='w+', shape=crop_shape)

        # Build the 2-D polygon mask once (crop-local coordinates)
        local_yx = polygon_yx - np.array([y0, x0])
        rr, cc = skimage_polygon(local_yx[:, 0], local_yx[:, 1],
                                 shape=(crop_h, crop_w))
        mask2d = np.zeros((crop_h, crop_w), dtype=bool)
        mask2d[rr, cc] = True
        outside = ~mask2d

        if is_3d:
            # Copy and mask one Z slice at a time — O(1 slice) peak RAM.
            print(f"  [ROI] Building 3D crop ({src.shape[0]} slices × "
                  f"{crop_h} × {crop_w})…")
            for z in range(src.shape[0]):
                slice_data = np.array(src[z, y0:y1, x0:x1])
                slice_data[outside] = 0
                crop_mm[z] = slice_data
        else:
            crop_mm[:] = src[y0:y1, x0:x1]
            crop_mm[outside] = 0

        crop_mm.flush()
        return crop_mm

    def _build_roi_config(
        self,
        y0: int, x0: int, y1: int, x1: int,
        base_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Produces a deep-copied config with the physical dimensions rescaled to
        the crop extent.  All execute_* parameter blocks are taken verbatim.

        The YAMLs store *total* physical extent (not per-voxel size), so the
        crop dimensions scale linearly with pixel count:
            new_x_um = original_x_um × (crop_w / full_w)
        This leaves per-voxel spacing identical while making the config
        self-consistent for the smaller array.
        """
        roi_config = copy.deepcopy(base_config)
        is_2d_mode = self.processing_mode.endswith('_2d')
        dim_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'

        orig_dims = base_config.get(dim_key, {'x': 1.0, 'y': 1.0, 'z': 1.0})
        orig_x = float(orig_dims.get('x', 1.0))
        orig_y = float(orig_dims.get('y', 1.0))

        full_h = self._full_image_stack.shape[-2]
        full_w = self._full_image_stack.shape[-1]
        crop_h = y1 - y0
        crop_w = x1 - x0

        new_dims = dict(orig_dims)
        new_dims['x'] = orig_x * (crop_w / full_w)
        new_dims['y'] = orig_y * (crop_h / full_h)
        # Z extent is unchanged — the full Z range is always kept.
        roi_config[dim_key] = new_dims
        return roi_config

    # --- Startup: detect an existing ROI session ---

    def _try_load_existing_roi_session(self) -> bool:
        """
        Checks whether a completed ROI session already exists for this image.
        If it does, offers the user a choice:
          • Load ROI session  → switches to ROI mode and calls restore_from_checkpoint
          • Process full image → returns False so the caller runs restore_from_checkpoint

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
            polygon_yx = np.array(roi_data['polygon_yx'])

            # Save full-image references
            self._full_image_stack = self.image_stack
            self._full_processed_dir = self.processed_dir
            self._full_config = copy.deepcopy(self.config)

            # Reuse existing crop dat if present, otherwise rebuild it
            crop_path = os.path.join(roi_dir, "roi_image_crop.dat")
            src = self._full_image_stack
            is_3d = src.ndim == 3
            crop_h, crop_w = y1 - y0, x1 - x0
            crop_shape = (src.shape[0], crop_h, crop_w) if is_3d else (crop_h, crop_w)

            if os.path.exists(crop_path):
                crop_mm = np.memmap(crop_path, dtype=src.dtype, mode='r+',
                                    shape=crop_shape)
            else:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                try:
                    crop_mm = self._build_crop_memmap(
                        src, y0, x0, y1, x1, polygon_yx, crop_path
                    )
                finally:
                    QApplication.restoreOverrideCursor()

            # Load persisted ROI config, or build it fresh
            roi_cfg_path = os.path.join(
                roi_dir, f"processing_config_{self.processing_mode}.yaml"
            )
            if os.path.exists(roi_cfg_path):
                with open(roi_cfg_path, 'r') as fh:
                    roi_config = yaml.safe_load(fh) or {}
            else:
                roi_config = self._build_roi_config(y0, x0, y1, x1,
                                                    self._full_config)

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
        The user draws a single polygon on any Z slice; for 3D data it is
        automatically extruded through the entire Z range on confirmation.
        """
        layer_name = "ROI Selection"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)

        # For 3D viewers force 2D display mode before adding the shape layer.
        # This ensures Napari presents a flat YX plane for drawing and that the
        # polygon coordinates come back as (Z, Y, X) with a constant Z — which
        # our extrusion logic expects.  The user does NOT need to draw on every
        # Z slice; one polygon on any slice is enough.
        is_3d = self.image_stack.ndim == 3
        if is_3d and self.viewer.dims.ndisplay != 2:
            self.viewer.dims.ndisplay = 2

        self.viewer.add_shapes(
            name=layer_name,
            shape_type='polygon',
            edge_color='yellow',
            face_color=[1, 1, 0, 0.08],
            edge_width=3,
        )
        self.viewer.layers[layer_name].mode = 'add_polygon'

        if is_3d:
            msg = (
                "Draw a polygon on the current Z slice to define the sub-region.\n\n"
                "  • Click to add vertices\n"
                "  • Double-click or press Enter to close the polygon\n\n"
                "You only need to draw on ONE Z slice.\n"
                "On confirmation the polygon is automatically extruded\n"
                "through the entire Z stack.\n\n"
                "When finished, click  ✓ Confirm ROI."
            )
        else:
            msg = (
                "Draw a polygon on the image to define the sub-region.\n\n"
                "  • Click to add vertices\n"
                "  • Double-click or press Enter to close the polygon\n\n"
                "When finished, click  ✓ Confirm ROI."
            )

        QMessageBox.information(None, "Draw ROI", msg)

    def confirm_roi(self) -> None:
        """
        Reads the drawn polygon, crops the image, persists the ROI to disk,
        and reinitialises the pipeline on the cropped sub-region.
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

        # Use the last drawn polygon (most recently confirmed by the user)
        raw_polygon = np.array(shapes_layer.data[-1])

        # Napari includes a Z column for 3D viewers — strip it
        is_3d = self.image_stack.ndim == 3
        polygon_yx = raw_polygon[:, 1:] if (raw_polygon.ndim == 2 and
                                             raw_polygon.shape[1] > 2) else raw_polygon
        polygon_yx = np.array(polygon_yx, dtype=float)

        # Bounding box, clamped to image extent
        img_h = self.image_stack.shape[-2]
        img_w = self.image_stack.shape[-1]
        y0 = max(0, int(np.floor(polygon_yx[:, 0].min())))
        x0 = max(0, int(np.floor(polygon_yx[:, 1].min())))
        y1 = min(img_h, int(np.ceil(polygon_yx[:, 0].max())) + 1)
        x1 = min(img_w, int(np.ceil(polygon_yx[:, 1].max())) + 1)
        crop_h, crop_w = y1 - y0, x1 - x0

        if crop_h < 10 or crop_w < 10:
            QMessageBox.warning(None, "ROI Too Small",
                                "The selected region is too small (< 10 px). "
                                "Please draw a larger polygon.")
            return

        full_shape = self.image_stack.shape
        reply = QMessageBox.question(
            None,
            "Confirm ROI",
            f"Bounding box: row {y0}–{y1}, col {x0}–{x1}\n"
            f"Crop size: {crop_h} × {crop_w} px  "
            f"(full image: {full_shape[-2]} × {full_shape[-1]})\n\n"
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

            # --- Persist polygon metadata ---
            roi_data = {
                "polygon_yx": polygon_yx.tolist(),
                "bbox": {"y0": y0, "x0": x0, "y1": y1, "x1": x1},
                "full_image_shape": list(full_shape),
            }
            with open(os.path.join(roi_dir, "roi_polygon.json"), 'w') as fh:
                json.dump(roi_data, fh, indent=2)

            # --- Build cropped + masked image memmap ---
            crop_path = os.path.join(roi_dir, "roi_image_crop.dat")
            crop_mm = self._build_crop_memmap(
                self._full_image_stack, y0, x0, y1, x1, polygon_yx, crop_path
            )

            # --- Build rescaled config ---
            roi_config = self._build_roi_config(y0, x0, y1, x1,
                                                self._full_config)

            # Persist the roi config so it can be reloaded on resume
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
        # Stop any running background worker
        if getattr(self, 'worker', None) and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
            self.worker = None

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

        print(f"[ROI] Now in ROI mode — shape {self.image_stack.shape}, "
              f"dir: {os.path.basename(roi_processed_dir)}")

    def _switch_to_full_image_mode(self) -> None:
        """Tears down the ROI session and reinstates full-image processing."""
        if self._full_image_stack is None:
            return

        if getattr(self, 'worker', None) and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
            self.worker = None

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

        print("[ROI] Returned to full-image mode.")

    def shutdown_and_cleanup(self) -> None:
        """Forcefully clears all data references and Napari internal buffers."""
        # Safely terminate the background worker if it's running
        if getattr(self, 'worker', None) and self.worker.isRunning():
            print("    [Thread] Stopping background worker...")
            self.worker.quit()
            self.worker.wait()
            self.worker = None

        # 1. Clear Napari layers and buffers first
        if self.viewer:
            try:
                self.viewer.layers.clear() # This drops the actual NumPy/Memmap references in Napari
            except Exception:
                pass
        
        # 2. Clear strategy and large data references
        if hasattr(self, 'strategy'):
            if hasattr(self.strategy, 'intermediate_state'):
                # Crucial: This dictionary often holds the 'original_volume_ref'
                self.strategy.intermediate_state.clear() 
            self.strategy = None
        
        self.image_stack = None # Release the memmap object
        self.viewer = None # Release the Napari viewer object
        
        # 3. Double-pass Garbage Collection (often needed for circular Qt references)
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
        self.clear_current_widgets()
        self.parameter_values = {}
        
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
            for pname, pconf in parameters.items():
                try:
                    # Create callback closure
                    cb = lambda val, k=config_key, p=pname: self.parameter_changed(k, p, val)
                    
                    # Create widget using helper
                    w = create_parameter_widget(pname, pconf, cb)
                    if w: 
                        scroll_l.addWidget(w.native)
                        self.parameter_values[pname] = pconf.get('value')
                except Exception:
                    pass
        
        # Log is persistent, so we don't add it here
        self._dock_widget(scroll_w, step_display)

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
        except Exception:
            pass

    def get_current_values(self) -> Dict[str, Any]:
        """Returns the current state of parameters."""
        return self.parameter_values.copy()