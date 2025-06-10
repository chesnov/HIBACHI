# --- START OF FILE utils/high_level_gui/processing_strategies.py ---
import abc
import os
import yaml # type: ignore
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import traceback
import gc
import time # Added for potential use in _remove_file_safely

class ProcessingStrategy(abc.ABC):
    """
    Abstract base class for processing strategies.
    Defines the interface for different segmentation/analysis workflows.
    Subclasses must implement methods to define their steps, load/save data,
    and clean up artifacts.
    """

    def __init__(self, config, processed_dir, image_shape, spacing, scale_factor):
        """
        Initializes the strategy.

        Args:
            config (dict): The configuration dictionary.
            processed_dir (str): The directory for saving processed outputs.
            image_shape (tuple): The shape of the input image stack (z, y, x).
            spacing (list/tuple): Voxel spacing [z, x, y].
            scale_factor (float): Z-axis scale factor for visualization (z_spacing / x_spacing).
        """
        self.config = config
        # Ensure processed_dir exists
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True) # Create dir if it doesn't exist

        self.image_shape = image_shape
        self.spacing = spacing # Store as [z, x, y]
        self.z_scale_factor = scale_factor
        self.mode_name = self._get_mode_name() # Get mode name from subclass
        self.intermediate_state = {} # To store temporary data between steps

        # Store step names defined by the subclass upon initialization
        self._step_names = self.get_step_names() # Call implementation from subclass
        self.num_steps = len(self._step_names)
        if self.num_steps == 0:
            print(f"Warning: Strategy '{self.__class__.__name__}' defines 0 processing steps.")

    @abc.abstractmethod
    def _get_mode_name(self) -> str:
        """
        Abstract method: Subclasses must implement this to return their
        unique mode identifier (e.g., 'nuclei', 'ramified').
        """
        pass

    @abc.abstractmethod
    def get_step_names(self) -> List[str]:
        """
        Abstract method: Subclasses must implement this to return an ordered
        list of strings. Each string must correspond exactly to the name of a
        method within the subclass that performs a processing step.
        Example: ["execute_step_one", "execute_step_two", ...]
        """
        pass

    def get_config_key(self, step_name: str) -> str:
        """
        Generates the configuration key for a given step, usually by appending
        the mode name. Checks if the mode-specific key exists in the config,
        otherwise returns the constructed key.

        Args:
            step_name (str): The internal method name of the step (e.g., "execute_raw_segmentation").

        Returns:
            str: The configuration key (e.g., "raw_segmentation_ramified").
        """
        mode_specific_key = f"{step_name}_{self.mode_name}"
        # Prioritize mode-specific key
        if mode_specific_key in self.config:
            return mode_specific_key
        # Fallback to generic name (less common, might indicate config issue)
        elif step_name in self.config:
            print(f"Warning: Using generic config key '{step_name}' for mode '{self.mode_name}'. Mode-specific key '{mode_specific_key}' not found.")
            return step_name
        else:
            # Return the expected mode-specific key even if absent from config,
            # allowing the caller (e.g., widget creation) to handle the missing key.
            return mode_specific_key


    def get_checkpoint_files(self) -> Dict[str, str]:
        """
        Return a dictionary of *default* checkpoint filenames.
        Subclasses MUST override this to define paths for ALL their outputs,
        potentially calling super().get_checkpoint_files() first to get defaults.
        Keys should ideally correspond logically to the data (e.g., "raw_segmentation",
        "final_segmentation", "metrics_df").

        Returns:
            Dict[str, str]: A dictionary mapping logical file keys to absolute paths.
        """
        # Base implementation only provides config and generic metrics path.
        # Warning removed as it's expected to be called by super()
        return {
            "config": os.path.join(self.processed_dir, f"processing_config_{self.mode_name}.yaml"),
            "metrics_df": os.path.join(self.processed_dir, f"metrics_df_{self.mode_name}.csv"),
            # Add other common/default files if applicable, but subclasses will likely define most needed paths.
        }

    def execute_step(self, step_index: int, viewer, image_stack_or_none: Any, params: Dict) -> bool:
        """
        Executes a processing step based on its 0-based index using the list
        from get_step_names().

        Args:
            step_index (int): The 0-based index of the step to execute.
            viewer: The Napari viewer instance.
            image_stack_or_none: The input image stack (or None). Step methods
                                 must handle receiving None if they don't need the stack.
            params (Dict): Dictionary of parameters for the step.

        Returns:
            bool: True if the step executed successfully, False otherwise.
        """
        if step_index < 0 or step_index >= self.num_steps:
            print(f"Error: Invalid step index {step_index} (must be 0 to {self.num_steps - 1}).")
            return False

        # Get the method name for this step index from the subclass's list
        step_method_name = self._step_names[step_index]
        step_display_number = step_index + 1 # For user messages (1-based)

        print(f"\n--- Attempting Step {step_display_number}/{self.num_steps}: '{step_method_name}' ---")

        try:
            # Get the actual method object from the subclass instance
            step_method = getattr(self, step_method_name)
        except AttributeError:
            print(f"FATAL ERROR: Method '{step_method_name}' required by get_step_names() not found in strategy class '{self.__class__.__name__}'.")
            return False # Cannot proceed

        try:
            # Call the step method. Assumes step methods accept these arguments.
            # Methods should handle image_stack_or_none being None if they don't need it.
            success = step_method(viewer=viewer, image_stack=image_stack_or_none, params=params)

            # Validate return type
            if not isinstance(success, bool):
                 print(f"Warning: Step method '{step_method_name}' did not return bool. Assuming failure.")
                 success = False

            print(f"--- Step {step_display_number} ('{step_method_name}') finished (Success: {success}) ---")
            return success

        except Exception as e:
            print(f"!!! ERROR during execution of step method '{step_method_name}': {e} !!!")
            traceback.print_exc()
            return False # Step failed due to exception

    @abc.abstractmethod
    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        """
        Abstract method: Subclasses must implement this to load data from
        checkpoint files relevant up to the end of the completed step number
        (1-based). For example, if checkpoint_step=2, load outputs from step 1 and 2.

        Args:
            viewer: The Napari viewer.
            checkpoint_step (int): The number of the last successfully completed step (1 to num_steps).
        """
        pass

    @abc.abstractmethod
    def cleanup_step_artifacts(self, viewer, step_number: int):
        """
        Abstract method: Subclasses must implement this to remove all layers
        and files generated by a specific step number (1-based). This is used
        when restarting or before re-running a step.

        Args:
            viewer: The Napari viewer.
            step_number (int): The 1-based number of the step whose artifacts should be cleaned.
        """
        pass

    # --- save_config (Complete Function with State Saving) ---
    def save_config(self, current_config):
        """
        Saves the provided configuration dictionary and serializable intermediate state
        to the designated YAML file. Excludes large numpy arrays like object references.

        Args:
            current_config (dict): The dictionary holding the current parameter values,
                                   likely self.config from the GUIManager.
        """
        files = self.get_checkpoint_files()
        config_save_path = files.get("config")
        if not config_save_path:
            print("Error: Cannot save config, 'config' path not defined in get_checkpoint_files().")
            return

        # --- Create a dictionary containing ONLY what should be saved ---
        config_to_save = {}

        # 1. Copy step parameters from the current config state
        #    Include known top-level keys explicitly if needed.
        print("DEBUG save_config: Processing current_config keys...")
        for step_key, step_data in current_config.items():
            if step_key == 'voxel_dimensions' and isinstance(step_data, dict):
                 config_to_save[step_key] = step_data
                 print(f"  Included key: {step_key}")
            # Include step configurations (assuming they start with 'execute_')
            elif step_key.startswith("execute_") and isinstance(step_data, dict):
                 config_to_save[step_key] = step_data
                 print(f"  Included key: {step_key}")
            elif step_key == 'mode': # Include mode if present
                config_to_save[step_key] = step_data
                print(f"  Included key: {step_key}")
            elif step_key == 'saved_state':
                 # Avoid copying old state if present in input config, we regenerate it below
                 print(f"  Ignoring existing 'saved_state' key in input config.")
            else:
                 print(f"  Skipping key (not recognized step/voxel_dims/mode): {step_key}")


        # Ensure mode is included if not already present
        if 'mode' not in config_to_save:
             config_to_save['mode'] = self.mode_name
             print(f"  Added missing 'mode' key: {self.mode_name}")

        # 2. Explicitly add *serializable* intermediate state under 'saved_state'
        #    Check the IN-MEMORY self.intermediate_state of the strategy object.
        #    DO NOT add large numpy arrays like 'original_volume_ref'
        print(f"DEBUG save_config: Checking intermediate_state keys: {list(self.intermediate_state.keys())}")
        saved_state_dict = {}
        if 'segmentation_threshold' in self.intermediate_state:
            try:
                 # Convert to standard Python float for YAML
                 threshold_value = float(self.intermediate_state['segmentation_threshold'])
                 saved_state_dict['segmentation_threshold'] = threshold_value
                 print(f"  Adding segmentation_threshold ({threshold_value}) to saved_state.")
            except (TypeError, ValueError) as e:
                 print(f"Warning: Could not convert segmentation_threshold '{self.intermediate_state['segmentation_threshold']}' to float for saving: {e}")
        else:
             print("  'segmentation_threshold' not found in intermediate_state.")

        # Add other simple, serializable state variables here if needed
        # e.g., if first_pass_params contained simple values you wanted to persist:
        # simple_params = {k:v for k,v in self.intermediate_state.get('first_pass_params', {}).items()
        #                  if isinstance(v, (int, float, str, bool, list, dict))} # Filter serializable
        # if simple_params: saved_state_dict['first_pass_params'] = simple_params

        # Only add the saved_state key to the final dictionary if it contains something
        if saved_state_dict:
            config_to_save['saved_state'] = saved_state_dict
            print(f"DEBUG: Final 'saved_state' content: {saved_state_dict}")
        else:
            print("DEBUG: No serializable intermediate state found to save.")

        # --- End creating clean dictionary ---

        try:
             print(f"Saving final configuration keys to {config_save_path}: {list(config_to_save.keys())}")
             with open(config_save_path, 'w') as file:
                 # Use safe_dump which is generally preferred. It handles basic types well.
                 # Avoid numpy types in config_to_save to prevent complex representations.
                 yaml.safe_dump(config_to_save, file, default_flow_style=False, sort_keys=False)
             print(f"Configuration successfully saved.")
        except Exception as e:
             print(f"Error saving configuration YAML to {config_save_path}: {e}")
             traceback.print_exc()


    # In utils/high_level_gui/processing_strategies.py

    def _add_layer_safely(self, viewer, data, name, layer_type='labels', **kwargs):
        """
        Helper to add/update a layer in Napari with mode suffix.
        Ensures consistent scale and translation for display, aligning with the
        original image layer's display transformation.
        """
        if viewer is None:
            # print(f"  BATCH MODE: Viewer is None. Skipping layer addition for '{name}_{self.mode_name}'.")
            return # Do nothing if no viewer
        
        layer_name = f"{name}_{self.mode_name}"

        # Determine the spatial dimensionality of the data
        spatial_ndim = -1
        if layer_type in ['image', 'labels']:
            if hasattr(data, 'ndim'):
                spatial_ndim = data.ndim
            else:
                print(f"Warning: Data for layer '{layer_name}' (type: {layer_type}) has no 'ndim'. Assuming 2D.")
                spatial_ndim = 2
        elif layer_type == 'shapes':
            if hasattr(data, 'ndim') and data.ndim >= 1:
                spatial_ndim = data.shape[-1] # Last dimension is usually spatial for shapes
            else:
                print(f"Warning: Shapes data for '{layer_name}' has unexpected ndim or no ndim. Assuming 2 spatial dims.")
                spatial_ndim = 2
        # Add handling for other layer types (e.g., points) if necessary
        # elif layer_type == 'points':
        #     if hasattr(data, 'ndim') and data.ndim == 2: # Points are usually (N, D_spatial)
        #         spatial_ndim = data.shape[1]
        #     else:
        #         print(f"Warning: Points data for '{layer_name}' has unexpected shape. Assuming 2D.")
        #         spatial_ndim = 2
        else:
            print(f"Warning: Unknown layer_type '{layer_type}' for '{layer_name}'. Attempting to use data.ndim.")
            try:
                spatial_ndim = data.ndim
            except AttributeError:
                print(f"Error: Cannot determine dimensionality for '{layer_name}'. Using default 2D.")
                spatial_ndim = 2

        # --- Determine the scale and translate for display ---
        # The goal is to use the same display scaling transform that was applied
        # to the original image when it was first added to the viewer by DynamicGUIManager.
        
        # These attributes (z_scale_factor, spacing) are set on the Strategy instance
        # by the DynamicGUIManager during initialization.
        # z_scale_factor = (actual_Z_voxel_size / actual_X_voxel_size)
        # spacing = (actual_Z_voxel_size, actual_Y_voxel_size, actual_X_voxel_size)

        display_scale_to_use = None
        display_translate_to_use = (0.0,) * spatial_ndim # Default translation is usually (0,0,...)

        if spatial_ndim == 2:
            # Original 2D image in DynamicGUIManager is added with scale=(self.spacing[1], self.spacing[2])
            # which are (Y_voxel_size, X_voxel_size)
            if hasattr(self, 'spacing') and len(self.spacing) >= 3:
                display_scale_to_use = (self.spacing[1], self.spacing[2])
            else:
                print(f"Warning: Strategy 'spacing' attribute ill-defined for 2D layer '{layer_name}'. Using (1,1) scale.")
                display_scale_to_use = (1.0, 1.0)
        
        elif spatial_ndim == 3:
            # Original 3D image in DynamicGUIManager is added with scale=(self.z_scale_factor, 1, 1)
            # This scale is designed for visual aspect ratio correction.
            # Any subsequent layer (like a segmentation) that is voxel-aligned with the original image
            # should use this SAME display scale to align correctly in the viewer's world space.
            if hasattr(self, 'z_scale_factor'):
                display_scale_to_use = (self.z_scale_factor, 1.0, 1.0)
            else:
                print(f"Warning: Strategy 'z_scale_factor' not found for 3D layer '{layer_name}'. "
                      "This implies an issue with strategy initialization or mode. "
                      "Falling back to raw voxel spacing, which might cause visual misalignment or squashing "
                      "if the original image was aspect-ratio corrected for display.")
                if hasattr(self, 'spacing') and len(self.spacing) == 3:
                    display_scale_to_use = tuple(self.spacing) # (abs_Z, abs_Y, abs_X)
                else:
                    display_scale_to_use = (1.0, 1.0, 1.0)
        
        elif spatial_ndim > 0 : # Generic case for other dimensions (e.g., 4D image)
            print(f"Warning: Generic spatial dimensionality {spatial_ndim} for layer '{layer_name}'. "
                  "Using default scale (1.0, ...) . Ensure this is intended.")
            display_scale_to_use = (1.0,) * spatial_ndim
        
        else: # Fallback if spatial_ndim determination failed or is 0/-1
             print(f"Error: Could not determine valid spatial dimensionality for layer '{layer_name}' (is {spatial_ndim}). "
                   "Using default 2D scale (1.0, 1.0).")
             display_scale_to_use = (1.0, 1.0)
             # Update spatial_ndim for translate if it was invalid
             if not isinstance(display_translate_to_use, tuple) or len(display_translate_to_use) !=2 :
                 display_translate_to_use = (0.0, 0.0)


        # Ensure kwargs for scale and translate are set
        kwargs['scale'] = display_scale_to_use
        kwargs['translate'] = display_translate_to_use

        # --- Debug prints ---
        print(f"  DEBUG [_add_layer_safely] Preparing layer '{layer_name}' (Type: {layer_type})")
        if hasattr(data, 'shape') and hasattr(data, 'dtype'):
             print(f"    Input data shape: {data.shape}, dtype: {data.dtype}")
        else:
             print(f"    Input data type: {type(data)}")
        print(f"    Determined spatial_ndim: {spatial_ndim}")
        print(f"    Applied DISPLAY scale: {kwargs['scale']}")
        print(f"    Applied DISPLAY translate: {kwargs['translate']}")

        # --- Add or Update Layer ---
        if layer_name not in viewer.layers:
            print(f"    Attempting ADD layer '{layer_name}'")
            try:
                if layer_type == 'labels':
                    viewer.add_labels(data, name=layer_name, **kwargs)
                elif layer_type == 'image':
                    viewer.add_image(data, name=layer_name, **kwargs)
                elif layer_type == 'shapes':
                    # Ensure data is not empty for shapes, as Napari might error
                    if (isinstance(data, np.ndarray) and data.size == 0) or \
                       (isinstance(data, list) and not data):
                        print(f"    Skipping addition of empty shapes layer '{layer_name}'.")
                        return # Or return the layer if add_shapes can handle empty data gracefully
                    viewer.add_shapes(data, name=layer_name, **kwargs)
                # Add elif for 'points' if you use it
                # elif layer_type == 'points':
                #     viewer.add_points(data, name=layer_name, **kwargs)
                else:
                    print(f"    Warning: Unknown layer type '{layer_type}' for '{layer_name}'. Cannot add.")
            except Exception as e:
                print(f"ERROR adding layer '{layer_name}': {e}")
                traceback.print_exc()
        else:
            print(f"    Layer '{layer_name}' exists. Attempting UPDATE.")
            try:
                layer = viewer.layers[layer_name]
                
                # Handle memmap data by converting to in-memory array for update
                # as direct assignment of memmap can sometimes be problematic for refresh
                if isinstance(data, np.memmap):
                    print(f"      Converting memmap data to in-memory array for update of layer '{layer_name}'.")
                    data_for_update = np.array(data) # Create a copy
                    # No need to close original memmap here if it's managed elsewhere or read-only for this operation
                else:
                    data_for_update = data

                layer.data = data_for_update
                
                # Explicitly set scale and translate on update as well,
                # in case they could have changed or need re-assertion.
                if hasattr(layer, 'scale'):
                    layer.scale = kwargs['scale']
                if hasattr(layer, 'translate'):
                    layer.translate = kwargs['translate']
                
                # Update other properties from kwargs if necessary
                for key, value in kwargs.items():
                    if key not in ['scale', 'translate', 'name'] and hasattr(layer, key):
                        try:
                            setattr(layer, key, value)
                        except Exception as e_setattr:
                            print(f"      Note: Could not set attribute '{key}' on layer '{layer_name}': {e_setattr}")
                
                layer.refresh() # Ensure the viewer updates
                print(f"    Layer '{layer_name}' updated.")
            except Exception as e:
                print(f"Error updating layer '{layer_name}': {e}")
                traceback.print_exc()

    def _remove_layer_safely(self, viewer, name):
        """Helper to remove a layer by its base name + mode suffix."""
        if viewer is None:
            # print(f"  BATCH MODE: Viewer is None. Skipping layer removal for '{name}_{self.mode_name}'.")
            return # Do nothing if no viewer
        
        layer_name = f"{name}_{self.mode_name}"
        if layer_name in viewer.layers:
            try:
                layer = viewer.layers[layer_name]
                # --- Remove memmap check - layer.data should be array now ---
                # if isinstance(layer.data, np.memmap):
                #     print(f"Closing memmap handle for layer '{layer_name}' before removal.")
                #     memmap_obj = layer.data
                #     if hasattr(memmap_obj, '_mmap') and memmap_obj._mmap is not None:
                #         try: memmap_obj._mmap.close()
                #         except Exception as e_close: print(f"  Warn: Error closing memmap: {e_close}")
                #     del memmap_obj; gc.collect()
                # --- End Remove memmap check ---

                viewer.layers.remove(layer) # Remove by layer object
                print(f"Removed layer: {layer_name}")
            except KeyError: print(f"Layer '{layer_name}' not found during removal.")
            except Exception as e: print(f"Error removing layer '{layer_name}': {e}"); traceback.print_exc()


    def _remove_file_safely(self, file_path):
        """Helper to remove a file if it exists, handling memmap cases."""
        if file_path and isinstance(file_path, str) and os.path.exists(file_path):
            print(f"Attempting to delete file: {file_path}")
            try:
                # Attempt GC before deleting .dat files
                if file_path.endswith('.dat'):
                    print("  Running GC before deleting .dat file...")
                    gc.collect()
                    time.sleep(0.2) # Increased pause

                os.remove(file_path)
                print(f"  Deleted file successfully: {file_path}")
                return True # Indicate success
            except PermissionError as pe:
                 print(f"  PermissionError deleting {file_path}. File in use? ({str(pe)})")
            except Exception as e:
                print(f"  Failed to delete {file_path}: {str(e)}")
        # Return False if path invalid, file not found, or deletion failed
        return False

# --- END OF FILE utils/high_level_gui/processing_strategies.py ---