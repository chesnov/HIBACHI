# --- START OF FILE utils/high_level_gui/processing_strategies.py ---
import abc
import os
import yaml # type: ignore
from typing import Dict, List, Any, TypedDict
import numpy as np
import traceback
import gc
import time

class StepDefinition(TypedDict):
    method: str      # The exact method name in the class (e.g., 'execute_raw_segmentation')
    artifact: str    # The key in get_checkpoint_files() that proves this step is done

class ProcessingStrategy(abc.ABC):
    """
    Abstract base class for processing strategies.
    
    Encapsulates the entire workflow definition so external managers (GUI, BatchProcessor)
    don't need to know specific step names or file paths.
    """

    def __init__(self, config, processed_dir, image_shape, spacing, scale_factor):
        self.config = config
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

        self.image_shape = image_shape
        self.spacing = spacing # [z, x, y]
        self.z_scale_factor = scale_factor
        self.mode_name = self._get_mode_name()
        self.intermediate_state = {} 

        # Auto-detect number of steps from the definitions
        self.steps = self.get_step_definitions()
        self.num_steps = len(self.steps)
        
        if self.num_steps == 0:
            print(f"Warning: Strategy '{self.__class__.__name__}' defines 0 processing steps.")

    @abc.abstractmethod
    def _get_mode_name(self) -> str:
        """Unique string identifier for this strategy (e.g., 'ramified')."""
        pass

    @abc.abstractmethod
    def get_step_definitions(self) -> List[StepDefinition]:
        """
        The Single Source of Truth for the workflow.
        Returns an ordered list of steps and their completion criteria.
        """
        pass

    def get_step_names(self) -> List[str]:
        """Helper: returns just the method names for iteration."""
        return [s['method'] for s in self.steps]

    def get_config_key(self, step_name: str) -> str:
        """Generates the configuration key (e.g. 'execute_raw_segmentation_ramified')."""
        mode_specific_key = f"{step_name}_{self.mode_name}"
        if mode_specific_key in self.config:
            return mode_specific_key
        elif step_name in self.config:
            # Fallback for backward compatibility
            return step_name
        return mode_specific_key

    def get_checkpoint_files(self) -> Dict[str, str]:
        """
        Default checkpoints. Subclasses should call super()... and add their own.
        """
        return {
            "config": os.path.join(self.processed_dir, f"processing_config_{self.mode_name}.yaml"),
            # Add generic metric path if applicable to all strategies
            "metrics_df": os.path.join(self.processed_dir, f"metrics_df_{self.mode_name}.csv"),
        }

    def get_last_completed_step(self) -> int:
        """
        Determines which steps are already finished by checking for their output artifacts.
        Returns the 1-based index of the last successful step.
        """
        checkpoints = self.get_checkpoint_files()
        last_step = 0
        
        for i, step_def in enumerate(self.steps):
            step_num = i + 1
            artifact_key = step_def['artifact']
            
            # If a step has no artifact defined, we assume it's not skippable/checkable
            if not artifact_key:
                break
                
            file_path = checkpoints.get(artifact_key)
            
            if file_path and os.path.exists(file_path):
                last_step = step_num
            else:
                # Chain broken: this step is missing, so subsequent steps are invalid
                break
                
        return last_step

    def execute_step(self, step_index: int, viewer, image_stack_or_none: Any, params: Dict) -> bool:
        """Executes a single processing step by 0-based index."""
        if step_index < 0 or step_index >= self.num_steps:
            print(f"Error: Invalid step index {step_index}.")
            return False

        step_def = self.steps[step_index]
        method_name = step_def['method']
        step_display_number = step_index + 1

        print(f"\n--- Attempting Step {step_display_number}/{self.num_steps}: '{method_name}' ---")

        try:
            step_method = getattr(self, method_name)
        except AttributeError:
            print(f"FATAL ERROR: Method '{method_name}' defined in steps but not implemented in class.")
            return False

        try:
            success = step_method(viewer=viewer, image_stack=image_stack_or_none, params=params)
            
            if not isinstance(success, bool):
                 print(f"Warning: Step method '{method_name}' return value was not boolean. Assuming failure.")
                 success = False

            print(f"--- Step {step_display_number} finished (Success: {success}) ---")
            return success

        except Exception as e:
            print(f"!!! ERROR during execution of step method '{method_name}': {e} !!!")
            traceback.print_exc()
            return False

    @abc.abstractmethod
    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        """Loads data into viewer for steps completed up to checkpoint_step."""
        pass

    @abc.abstractmethod
    def cleanup_step_artifacts(self, viewer, step_number: int):
        """Cleans up files created by a specific step (and subsequent steps)."""
        pass

    def save_config(self, current_config):
        """Saves configuration state."""
        files = self.get_checkpoint_files()
        config_save_path = files.get("config")
        if not config_save_path: return

        config_to_save = {}
        # Copy relevant keys
        for step_key, step_data in current_config.items():
            if step_key in ['voxel_dimensions', 'pixel_dimensions', 'mode', 'saved_state']:
                 config_to_save[step_key] = step_data
            elif step_key.startswith("execute_") and isinstance(step_data, dict):
                 config_to_save[step_key] = step_data

        if 'mode' not in config_to_save:
             config_to_save['mode'] = self.mode_name

        # Save intermediate python state (thresholds)
        saved_state_dict = {}
        if 'segmentation_threshold' in self.intermediate_state:
            try:
                 saved_state_dict['segmentation_threshold'] = float(self.intermediate_state['segmentation_threshold'])
            except (TypeError, ValueError): pass

        if saved_state_dict:
            config_to_save['saved_state'] = saved_state_dict

        try:
             with open(config_save_path, 'w') as file:
                 yaml.safe_dump(config_to_save, file, default_flow_style=False, sort_keys=False)
        except Exception as e:
             print(f"Error saving config: {e}")

    # ... (keep _add_layer_safely, _remove_layer_safely, _remove_file_safely as is) ...
    def _add_layer_safely(self, viewer, data, name, layer_type='labels', **kwargs):
        if viewer is None: return
        layer_name = f"{name}_{self.mode_name}"
        # (Rest of logic identical to previous versions for brevity)
        spatial_ndim = 2
        if hasattr(data, 'ndim'): spatial_ndim = data.ndim
        elif hasattr(data, 'shape'): spatial_ndim = len(data.shape)
        if layer_type == 'shapes': spatial_ndim = 2 

        display_scale = (1.0,) * spatial_ndim
        if spatial_ndim == 2:
            if hasattr(self, 'spacing') and len(self.spacing) >= 3:
                display_scale = (self.spacing[1], self.spacing[2])
        elif spatial_ndim == 3:
            if hasattr(self, 'z_scale_factor'):
                display_scale = (self.z_scale_factor, 1.0, 1.0)
            elif hasattr(self, 'spacing') and len(self.spacing) == 3:
                display_scale = tuple(self.spacing)

        kwargs['scale'] = display_scale
        kwargs['translate'] = kwargs.get('translate', (0.0,) * spatial_ndim)

        if layer_name not in viewer.layers:
            try:
                if layer_type == 'labels': viewer.add_labels(data, name=layer_name, **kwargs)
                elif layer_type == 'image': viewer.add_image(data, name=layer_name, **kwargs)
                elif layer_type == 'shapes': 
                    if len(data) > 0: viewer.add_shapes(data, name=layer_name, **kwargs)
                elif layer_type == 'points': viewer.add_points(data, name=layer_name, **kwargs)
            except Exception as e: print(f"Error adding layer {layer_name}: {e}")
        else:
            try:
                layer = viewer.layers[layer_name]
                if isinstance(data, np.memmap): data = np.array(data)
                layer.data = data
                if 'scale' in kwargs: layer.scale = kwargs['scale']
                layer.refresh()
            except Exception as e: print(f"Error updating layer {layer_name}: {e}")

    def _remove_layer_safely(self, viewer, name):
        if viewer is None: return
        layer_name = f"{name}_{self.mode_name}"
        if layer_name in viewer.layers:
            try: viewer.layers.remove(layer_name)
            except Exception as e: print(f"Error removing layer {layer_name}: {e}")

    def _remove_file_safely(self, file_path):
        if file_path and isinstance(file_path, str) and os.path.exists(file_path):
            try:
                gc.collect(); time.sleep(0.1)
                os.remove(file_path)
                return True
            except Exception as e: print(f"  Failed to delete {file_path}: {e}")
        return False
# --- END OF FILE utils/high_level_gui/processing_strategies.py ---