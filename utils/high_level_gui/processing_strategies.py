# --- START OF FILE gui/processing_strategies.py ---
import abc
import os
import yaml
from typing import Dict

# --- REMOVED imports for segmentation functions ---
# from ramified_segmenter import *
# from nuclear_segmenter import *
# from calculate_features import *
# from initial_3d_segmentation import segment_microglia_first_pass
# These will be imported directly by the concrete strategies that use them.


class ProcessingStrategy(abc.ABC):
    """Abstract base class for processing strategies."""

    def __init__(self, config, processed_dir, image_shape, spacing, scale_factor):
        self.config = config
        self.processed_dir = processed_dir
        self.image_shape = image_shape
        self.spacing = spacing # Store as [z, x, y]
        self.z_scale_factor = scale_factor
        self.mode_name = self._get_mode_name()
        self.intermediate_state = {} # To store things like first_pass_params

    @abc.abstractmethod
    def _get_mode_name(self) -> str:
        """Return the name of the mode (e.g., 'nuclei', 'ramified')."""
        pass

    def get_config_key(self, step_name: str) -> str:
        """Get the specific config key for a step, potentially mode-specific."""
        mode_specific_key = f"{step_name}_{self.mode_name}"
        return mode_specific_key if mode_specific_key in self.config else step_name

    def get_checkpoint_files(self) -> Dict[str, str]:
        """Return a dictionary of checkpoint filenames for this mode."""
        # This default implementation is fine, specific strategies can override if needed
        return {
            "initial_segmentation": os.path.join(self.processed_dir, f"initial_segmentation_{self.mode_name}.npy"),
            "final_segmentation": os.path.join(self.processed_dir, f"final_segmentation_{self.mode_name}.dat"),
            "cell_bodies": os.path.join(self.processed_dir, f"cell_bodies_{self.mode_name}.npy"),
            "config": os.path.join(self.processed_dir, f"processing_config_{self.mode_name}.yaml"),
            "distances_matrix": os.path.join(self.processed_dir, f"distances_matrix_{self.mode_name}.csv"),
            "points_matrix": os.path.join(self.processed_dir, f"points_matrix_{self.mode_name}.csv"),
            "lines": os.path.join(self.processed_dir, f"lines_{self.mode_name}.csv"),
            "metrics_df": os.path.join(self.processed_dir, f"metrics_df_{self.mode_name}.csv"),
            "ramification_summary": os.path.join(self.processed_dir, f"ramification_summary_{self.mode_name}.csv"),
            "branch_data": os.path.join(self.processed_dir, f"branch_data_{self.mode_name}.csv"),
            "endpoint_data": os.path.join(self.processed_dir, f"endpoint_data_{self.mode_name}.csv"),
            "skeleton_array": os.path.join(self.processed_dir, f"skeleton_array_{self.mode_name}.npy"),
            "depth_df": os.path.join(self.processed_dir, f"depth_df_{self.mode_name}.csv")
        }

    @abc.abstractmethod
    def execute_initial_segmentation(self, viewer, image_stack, params: Dict):
        """Execute step 1: Initial Segmentation."""
        pass

    @abc.abstractmethod
    def execute_refine_rois(self, viewer, image_stack, params: Dict):
        """Execute step 2: Refine ROIs."""
        pass

    @abc.abstractmethod
    def execute_calculate_features(self, viewer, image_stack, params: Dict):
        """Execute step 3: Calculate Features."""
        pass

    @abc.abstractmethod
    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        """Load data from checkpoint files for this mode."""
        pass

    @abc.abstractmethod
    def cleanup_step_artifacts(self, viewer, step_number: int):
        """Remove files and layers specific to a step for this mode."""
        pass

    # --- save_config and helper methods (_add_layer_safely, etc.) remain the same ---
    def save_config(self, current_config):
        """Save the configuration specific to this mode."""
        config_save_path = self.get_checkpoint_files()["config"]
        updated_config = current_config.copy()
        if 'mode' not in updated_config:
             updated_config['mode'] = self.mode_name
        updated_config.update(self.intermediate_state) # Include intermediate state if any

        try:
             with open(config_save_path, 'w') as file:
                 yaml.dump(updated_config, file, default_flow_style=False, sort_keys=False)
             print(f"Configuration saved to {config_save_path}")
        except Exception as e:
             print(f"Error saving configuration to {config_save_path}: {e}")


    def _add_layer_safely(self, viewer, data, name, layer_type='labels', **kwargs):
        """Helper to add layer if it doesn't exist."""
        if name not in viewer.layers:
            try:
                 if layer_type == 'labels':
                     viewer.add_labels(data, name=name, scale=(self.z_scale_factor, 1, 1), **kwargs)
                 elif layer_type == 'image':
                     viewer.add_image(data, name=name, scale=(self.z_scale_factor, 1, 1), **kwargs)
                 elif layer_type == 'shapes':
                     viewer.add_shapes(data, name=name, scale=(self.z_scale_factor, 1, 1), **kwargs)
                 else:
                     print(f"Warning: Unknown layer type '{layer_type}' for layer '{name}'")
            except Exception as e:
                 print(f"Error adding layer '{name}': {e}")
        else:
            print(f"Layer '{name}' already exists. Skipping addition.")
            # Optionally update data if layer exists
            # try:
            #    viewer.layers[name].data = data
            # except Exception as e:
            #    print(f"Error updating data for existing layer '{name}': {e}")


    def _remove_layer_safely(self, viewer, name):
        """Helper to remove layer if it exists."""
        if name in viewer.layers:
            try:
                viewer.layers.remove(name)
            except Exception as e:
                 print(f"Error removing layer '{name}': {e}")


    def _remove_file_safely(self, file_path):
        """Helper to remove file if it exists."""
        if file_path and os.path.exists(file_path): # Add check for None/empty path
            try:
                if file_path.endswith('.dat'):
                    import gc
                    gc.collect() # Attempt to release file handles for memmaps
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {str(e)}")
        # else:
        #      print(f"File not found or path invalid, skipping deletion: {file_path}")


# --- END OF FILE gui/processing_strategies.py ---