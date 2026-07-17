import abc
import os
import sys
import gc
import time
import traceback
import warnings
from typing import Dict, List, Any, Tuple, TypedDict, Optional, Union

import numpy as np
import yaml  # type: ignore

# Suppress the Napari/vispy GL_MAX_TEXTURE_SIZE downsampling notice.
# This fires asynchronously from the Qt event loop after layer data is uploaded
# to the GPU, so a catch_warnings() context manager around add_*() calls cannot
# intercept it.  The filter is intentionally narrow: UserWarning only, matching
# the exact message text, and scoped to the napari._vispy module.
warnings.filterwarnings(
    "ignore",
    message=r".*GL_MAX_TEXTURE_SIZE.*",
    category=UserWarning,
    module=r"napari\._vispy",
)


def _hibachi_version_stamp() -> Dict[str, Any]:
    """
    Build the `hibachi_version` block written into each processed config.

    Records the git commit (so the exact code can be checked out to reproduce
    the run), the branch, whether the working tree was dirty, and a UTC
    timestamp of when the config was saved. Reuses the launcher's stdlib-only
    `updater` module (no Qt import). Fully best-effort: on any failure it returns
    a minimal stamp rather than raising, so saving the config never breaks.
    """
    import datetime

    stamp: Dict[str, Any] = {
        "commit": None,
        "short": None,
        "date": None,
        "branch": None,
        "dirty": None,
        "processed_at": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }
    try:
        # repo layout: <repo>/utils/high_level_gui/processing_strategies.py
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        launcher = os.path.join(repo_root, "launcher")
        if launcher not in sys.path:
            sys.path.insert(0, launcher)
        import updater  # type: ignore

        info = updater.describe_version(repo_root)
        for key in ("commit", "short", "date", "branch", "dirty"):
            if key in info:
                stamp[key] = info[key]
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[version] could not determine HIBACHI version: {exc}")
    return stamp


class StepDefinition(TypedDict):
    """
    Defines a single processing step.

    Attributes:
        method: The exact method name in the Strategy class to execute (e.g., 'execute_raw_segmentation').
        artifact: The key in `get_checkpoint_files()` that proves this step is complete.
                  If None, the step is considered repeatable/optional and not skipped on resume.
    """
    method: str
    artifact: Optional[str]


class ProcessingStrategy(abc.ABC):
    """
    Abstract Base Class for segmentation processing strategies.

    This class encapsulates the workflow definition, ensuring that external managers
    (like the GUI or BatchProcessor) do not need to know specific step names,
    file paths, or internal logic. It handles the "Template Method" pattern for
    executing steps, saving state, and managing Napari viewer layers.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        processed_dir: str,
        image_shape: Tuple[int, ...],
        spacing: Union[Tuple[float, float], Tuple[float, float, float]],
        scale_factor: float
    ):
        """
        Initialize the Processing Strategy.

        Args:
            config: Dictionary containing configuration parameters.
            processed_dir: Path to the directory where outputs will be saved.
            image_shape: Shape of the input image (Z, Y, X) or (Y, X).
            spacing: Voxel/Pixel spacing (e.g., (1.0, 0.5, 0.5)).
            scale_factor: Z-scale factor for 3D visualization (anisotropy).
        """
        self.config = config
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)
        self.temp_dir = os.path.join(self.processed_dir, "temp_artifacts")
        os.makedirs(self.temp_dir, exist_ok=True)

        self.image_shape = image_shape
        self.spacing = spacing
        self.z_scale_factor = scale_factor
        self.mode_name = self._get_mode_name()
        self._visibility_cache: Dict[str, bool] = {}
        
        # Dictionary to store runtime state (e.g., calculated thresholds)
        # that needs to pass between steps but isn't strictly config.
        self.intermediate_state: Dict[str, Any] = {}

        # Auto-detect number of steps from the definitions
        self.steps = self.get_step_definitions()
        self.num_steps = len(self.steps)

        if self.num_steps == 0:
            print(f"Warning: Strategy '{self.__class__.__name__}' defines 0 steps.")

    @abc.abstractmethod
    def _get_mode_name(self) -> str:
        """
        Returns the unique string identifier for this strategy (e.g., 'fluorescence').
        Used for file naming and config lookups.
        """
        pass

    @abc.abstractmethod
    def get_step_definitions(self) -> List[StepDefinition]:
        """
        The Single Source of Truth for the workflow.

        Returns:
            List[StepDefinition]: An ordered list of steps, defining the method
            to call and the artifact to check for completion.
        """
        pass

    def get_step_names(self) -> List[str]:
        """Helper: returns just the method names for iteration."""
        return [s['method'] for s in self.steps]

    def get_config_key(self, step_name: str) -> str:
        """
        Generates the configuration key for a specific step.
        Example: 'execute_raw_segmentation' -> 'execute_raw_segmentation_fluorescence'

        Args:
            step_name: The base name of the step method.

        Returns:
            str: The key used in the YAML config file.
        """
        mode_specific_key = f"{step_name}_{self.mode_name}"
        if mode_specific_key in self.config:
            return mode_specific_key
        elif step_name in self.config:
            # Fallback for backward compatibility
            return step_name
        return mode_specific_key

    def get_checkpoint_files(self) -> Dict[str, str]:
        """
        Defines default checkpoint files. Subclasses should call super()...
        and update the returned dictionary with their specific artifacts.

        Returns:
            Dict[str, str]: Map of artifact keys to file paths.
        """
        return {
            "config": os.path.join(
                self.processed_dir, f"processing_config_{self.mode_name}.yaml"
            ),
            "metrics_df": os.path.join(
                self.processed_dir, f"metrics_df_{self.mode_name}.csv"
            ),
        }

    def get_last_completed_step(self) -> int:
        """
        Determines which steps are already finished by checking for their
        output artifacts on disk.

        Returns:
            int: The 1-based index of the last successful step.
                 (e.g., if step 1 and 2 are done, returns 2).
        """
        checkpoints = self.get_checkpoint_files()
        last_step = 0

        for i, step_def in enumerate(self.steps):
            step_num = i + 1
            artifact_key = step_def['artifact']

            # If a step has no artifact defined, it's likely an optional/
            # analysis step that doesn't block progress, or we can't verify it.
            # We assume the chain breaks here for auto-resume purposes.
            if not artifact_key:
                break

            file_path = checkpoints.get(artifact_key)

            if file_path and os.path.exists(file_path):
                last_step = step_num
            else:
                # Chain broken: this step is missing, so subsequent steps are invalid
                break

        return last_step

    def execute_step(
        self,
        step_index: int,
        viewer: Optional[Any],
        image_stack_or_none: Optional[np.ndarray],
        params: Dict[str, Any]
    ) -> bool:
        """
        Executes a single processing step by its 0-based index.

        Args:
            step_index: Index of the step in `get_step_definitions`.
            viewer: Napari viewer instance (can be None for batch mode).
            image_stack_or_none: The input image array (or None if not needed).
            params: Dictionary of parameters for this step.

        Returns:
            bool: True if the step completed successfully, False otherwise.
        """
        if step_index < 0 or step_index >= self.num_steps:
            print(f"Error: Invalid step index {step_index}.")
            return False

        step_def = self.steps[step_index]
        method_name = step_def['method']
        step_display_number = step_index + 1

        print(f"\n--- Attempting Step {step_display_number}/{self.num_steps}: "
              f"'{method_name}' ---")

        try:
            step_method = getattr(self, method_name)
        except AttributeError:
            print(f"FATAL ERROR: Method '{method_name}' defined in steps but "
                  f"not implemented in class {self.__class__.__name__}.")
            return False

        try:
            success = step_method(
                viewer=viewer,
                image_stack=image_stack_or_none,
                params=params
            )

            if not isinstance(success, bool):
                print(f"Warning: Step method '{method_name}' return value "
                      f"was not boolean ({type(success)}). Assuming failure.")
                success = False

            print(f"--- Step {step_display_number} finished (Success: {success}) ---")
            return success

        except Exception as e:
            print(f"!!! ERROR during execution of step method '{method_name}': {e} !!!")
            traceback.print_exc()
            return False

    @abc.abstractmethod
    def load_checkpoint_data(self, viewer: Any, checkpoint_step: int) -> None:
        """
        Loads result data into the viewer for steps completed up to checkpoint_step.

        Args:
            viewer: Napari viewer instance.
            checkpoint_step: The 1-based index of the last completed step.
        """
        pass

    @abc.abstractmethod
    def cleanup_step_artifacts(self, viewer: Optional[Any], step_number: int) -> None:
        """
        Cleans up files and viewer layers created by a specific step.
        Used when re-running a step to ensure a clean state.

        Args:
            viewer: Napari viewer instance (or None).
            step_number: The 1-based index of the step to clean.
        """
        pass

    def save_config(self, current_config: Dict[str, Any]) -> None:
        """
        Saves the current configuration state to the processed directory.
        Includes parameter settings and intermediate state (like thresholds).

        Args:
            current_config: The full configuration dictionary.
        """
        files = self.get_checkpoint_files()
        config_save_path = files.get("config")
        if not config_save_path:
            return

        config_to_save = {}
        
        # Copy relevant keys, filtering out internal UI state if any
        for step_key, step_data in current_config.items():
            if step_key in ['voxel_dimensions', 'pixel_dimensions', 'mode', 'saved_state']:
                config_to_save[step_key] = step_data
            elif step_key.startswith("execute_") and isinstance(step_data, dict):
                config_to_save[step_key] = step_data

        if 'mode' not in config_to_save:
            config_to_save['mode'] = self.mode_name

        # Save intermediate python state (e.g., calculated thresholds)
        saved_state_dict = {}
        if 'segmentation_threshold' in self.intermediate_state:
            try:
                val = float(self.intermediate_state['segmentation_threshold'])
                saved_state_dict['segmentation_threshold'] = val
            except (TypeError, ValueError):
                pass

        if saved_state_dict:
            config_to_save['saved_state'] = saved_state_dict

        # Stamp the running HIBACHI version so this processed run can be
        # reproduced later (git checkout the recorded commit). Best-effort and
        # non-fatal: if the version can't be determined, we still save the config.
        try:
            config_to_save['hibachi_version'] = _hibachi_version_stamp()
        except Exception as e:
            print(f"Warning: could not stamp HIBACHI version: {e}")

        try:
            with open(config_save_path, 'w') as file:
                yaml.safe_dump(
                    config_to_save, file, default_flow_style=False, sort_keys=False
                )
        except Exception as e:
            print(f"Error saving config: {e}")

    def _add_layer_safely(
        self,
        viewer: Any,
        data: Union[np.ndarray, Any],
        name: str,
        layer_type: str = 'labels',
        **kwargs: Any
    ) -> None:
        """
        Adds or updates a layer in the Napari viewer safely.
        Handles scaling for 2D vs 3D and different layer types.

        Args:
            viewer: Napari viewer instance.
            data: Data to display (array, points, etc.).
            name: Base name for the layer (mode suffix will be appended).
            layer_type: 'labels', 'image', 'shapes', or 'points'.
            **kwargs: Additional arguments for the viewer.add_* method.
        """
        if viewer is None: 
            return
        layer_name = f"{name}_{self.mode_name}"

        # Handle Visibility: 
        # We must avoid initializing layers as invisible (visible=False) because 
        # it can cause Napari to skip slice initialization, leading to 
        # dimensionality mismatch errors in _update_thumbnail (ndi.zoom).
        
        target_visibility = True # Default to visible
        if 'visible' in kwargs:
            target_visibility = kwargs.pop('visible') # Extract and remove from kwargs

        if layer_name in viewer.layers:
            # If the layer exists, remember its current visibility
            target_visibility = viewer.layers[layer_name].visible
            self._visibility_cache[layer_name] = target_visibility
        elif layer_name in self._visibility_cache:
            # If the layer is new but we have a cached state, restore it
            target_visibility = self._visibility_cache[layer_name]

        # Determine dimensionality of data.
        # For multiscale input (list of arrays), use the finest level.
        spatial_ndim = 2
        _ref = data[0] if isinstance(data, list) else data
        if hasattr(_ref, 'ndim'):
            spatial_ndim = _ref.ndim
        elif hasattr(_ref, 'shape'):
            spatial_ndim = len(_ref.shape)
        if layer_type == 'shapes':
            spatial_ndim = 2  # Shapes usually handled as overlay

        # Calculate display scale
        display_scale = (1.0,) * spatial_ndim
        if spatial_ndim == 2:
            if hasattr(self, 'spacing') and len(self.spacing) >= 3:
                # Use Y, X spacing
                display_scale = (self.spacing[1], self.spacing[2])
            elif hasattr(self, 'spacing') and len(self.spacing) == 2:
                display_scale = self.spacing
        elif spatial_ndim == 3:
            if hasattr(self, 'z_scale_factor'):
                display_scale = (self.z_scale_factor, 1.0, 1.0)
            elif hasattr(self, 'spacing') and len(self.spacing) == 3:
                display_scale = tuple(self.spacing)

        kwargs['scale'] = display_scale
        kwargs['translate'] = kwargs.get('translate', (0.0,) * spatial_ndim)

        # Update existing layer or add new one
        if layer_name not in viewer.layers:
            try:
                # Add layer (defaulting to visible=True implicitly by removing kwarg)
                new_layer = None
                if layer_type == 'labels':
                    new_layer = viewer.add_labels(data, name=layer_name, **kwargs)
                elif layer_type == 'image':
                    new_layer = viewer.add_image(data, name=layer_name, **kwargs)
                elif layer_type == 'shapes':
                    if len(data) > 0:
                        new_layer = viewer.add_shapes(data, name=layer_name, **kwargs)
                elif layer_type == 'points':
                    new_layer = viewer.add_points(data, name=layer_name, **kwargs)

                # Apply visibility AFTER creation to ensure initialization logic runs
                if new_layer is not None:
                    new_layer.visible = target_visibility

            except Exception as e:
                print(f"Error adding layer {layer_name}: {e}")
        else:
            try:
                layer = viewer.layers[layer_name]
                # Ensure data is in memory if it was a memmap (for stability on update).
                # For multiscale lists, resolve any memmaps in the finest level only;
                # coarser levels are already plain ndarrays from block_reduce.
                if isinstance(data, list):
                    if isinstance(data[0], np.memmap):
                        data[0] = np.array(data[0])
                elif isinstance(data, np.memmap):
                    data = np.array(data)
                layer.data = data
                if 'scale' in kwargs:
                    layer.scale = kwargs['scale']
                layer.refresh()
            except Exception as e:
                print(f"Error updating layer {layer_name}: {e}")

    @staticmethod
    def _build_label_pyramid(
        data: np.ndarray,
        gl_max: int = 16384
    ) -> list:
        """
        Builds a max-pooling multiscale pyramid for a labels array.

        Napari accepts a list of arrays as multiscale data and selects the
        appropriate level based on the current zoom, mirroring how QuPath /
        OpenSlide handle whole-slide images.  Max-pooling is used (rather than
        averaging) so that 1-pixel-wide structures such as skeletons are never
        lost: a label survives into a coarser level as long as it was present
        in *any* pixel of the corresponding block at the finer level.

        The pyramid is only built when the array actually exceeds the GPU
        texture limit; otherwise the original array is returned as a
        single-element list (no overhead).

        Args:
            data:   Integer label array (2-D or 3-D).
            gl_max: GPU texture size limit to build down to (default 16384).

        Returns:
            List of arrays [full_res, half_res, quarter_res, ...].
            Single-element list when no downsampling is needed.
        """
        if max(data.shape) <= gl_max:
            return [data]

        # Strided slicing is used instead of block_reduce/max-pooling.
        # On a memmap, data[::2, ::2] adjusts stride metadata only — no data
        # is read or copied until Napari requests a tile — making the pyramid
        # build essentially instantaneous regardless of array size.
        # The trade-off vs max-pooling is that a skeleton pixel that happens to
        # fall between sampled positions could be missed at a coarser level, but
        # at a 1.22× ratio (20038 → 16384) this is imperceptible in practice.
        ndim = data.ndim
        sl_2d = (slice(None, None, 2), slice(None, None, 2))
        stride = (slice(None),) + sl_2d if ndim == 3 else sl_2d  # Keep Z intact in 3D

        pyramid = [data]
        current = data
        while max(current.shape[-2:]) > gl_max:
            current = current[stride]
            pyramid.append(current)

        print(f"  [Pyramid] Built {len(pyramid)}-level pyramid: "
              f"{' → '.join(str(a.shape) for a in pyramid)}")
        return pyramid

    def _remove_layer_safely(self, viewer: Any, name: str) -> None:
        """
        Removes a layer from the viewer if it exists.

        Args:
            viewer: Napari viewer instance.
            name: Base name of the layer.
        """
        if viewer is None:
            return
        layer_name = f"{name}_{self.mode_name}"
        if layer_name in viewer.layers:
            # Save visibility to cache before removing the layer
            self._visibility_cache[layer_name] = viewer.layers[layer_name].visible
            try:
                viewer.layers.remove(layer_name)
            except Exception as e:
                print(f"Error removing layer {layer_name}: {e}")

    def _remove_file_safely(self, file_path: Optional[str]) -> bool:
        """
        Safely removes a file from disk, handling garbage collection delays.

        Args:
            file_path: Path to the file to remove.

        Returns:
            bool: True if removed, False otherwise.
        """
        if file_path and isinstance(file_path, str) and os.path.exists(file_path):
            try:
                gc.collect()
                time.sleep(0.1)  # Allow file handles to release
                os.remove(file_path)
                return True
            except Exception as e:
                print(f"  Failed to delete {file_path}: {e}")
        return False
    
    def cleanup_temporary_files(self) -> None:
        """Removes the localized temp directory and all its non-checkpoint contents."""
        if os.path.exists(self.temp_dir):
            import shutil
            try:
                gc.collect() # Release handles
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                print(f"Warning: Could not fully clean temp directory: {e}")