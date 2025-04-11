# --- START OF FILE gui/_3D_nuclear_strategy.py ---
import os
import numpy as np
import pandas as pd
import yaml
import sys
from typing import Dict

# Import base class using relative import
from .processing_strategies import ProcessingStrategy

# Add parent directory (project root) to path to find segmentation modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import necessary functions from sibling modules
try:
    from nuclear_segmenter import segment_nuclei
    from calculate_features import shortest_distance, analyze_segmentation, calculate_depth_df
except ImportError as e:
    print(f"Error importing segmentation functions in _3D_nuclear_strategy.py: {e}")
    print(f"Ensure nuclear_segmenter.py and calculate_features.py are in the directory: {parent_dir}")
    raise


class NuclearStrategy(ProcessingStrategy):

    def _get_mode_name(self) -> str:
        return "nuclei"

    # --- Paste the entire execute_initial_segmentation method here ---
    def execute_initial_segmentation(self, viewer, image_stack, params: Dict):
        print(f"Running initial {self.mode_name} segmentation...")
        labeled_cells, first_pass_params = segment_nuclei(
            image_stack,
            first_pass=None,
            smooth_sigma=params.get("smooth_sigma", [0, 0.5, 1.0, 2.0]), # Example defaults
            min_distance=params.get("min_distance"),
            min_size=params.get("min_size"),
            contrast_threshold_factor=params.get("contrast_threshold_factor"),
            spacing=self.spacing,
            anisotropy_normalization_degree=params.get("anisotropy_normalization_degree")
        )
        self.intermediate_state['first_pass_params'] = first_pass_params

        initial_seg_path = self.get_checkpoint_files()["initial_segmentation"]
        try:
             np.save(initial_seg_path, labeled_cells)
             self._add_layer_safely(viewer, labeled_cells, "Intermediate segmentation")
             print(f"Saved initial segmentation to {initial_seg_path}")
             return True
        except Exception as e:
             print(f"Error saving or adding layer for initial segmentation: {e}")
             return False


    # --- Paste the entire execute_refine_rois method here ---
    def execute_refine_rois(self, viewer, image_stack, params: Dict):
        print(f"Refining {self.mode_name} ROIs...")
        initial_seg_path = self.get_checkpoint_files()["initial_segmentation"]
        if not os.path.exists(initial_seg_path):
             print("Error: Initial segmentation file not found.")
             return False

        try:
             labeled_cells = np.load(initial_seg_path)
        except Exception as e:
             print(f"Error loading initial segmentation file {initial_seg_path}: {e}")
             return False

        # Retrieve first_pass_params saved from the previous step or config
        first_pass_params = self.intermediate_state.get('first_pass_params')
        if first_pass_params is None:
             print("Warning: first_pass_params not in intermediate state. Trying config...")
             config_file = self.get_checkpoint_files()["config"]
             if os.path.exists(config_file):
                  try:
                       with open(config_file, 'r') as file:
                            saved_config = yaml.safe_load(file)
                            first_pass_params = saved_config.get('first_pass_params')
                       if first_pass_params:
                            print("Loaded first_pass_params from saved config.")
                            self.intermediate_state['first_pass_params'] = first_pass_params # Store again
                  except Exception as e:
                       print(f"Error loading first_pass_params from config {config_file}: {e}")

        if first_pass_params is None:
             print("Error: Could not find first_pass_params needed for nuclear refinement.")
             return False

        try:
             merged_roi_array = segment_nuclei(
                 image_stack,
                 first_pass=labeled_cells,
                 first_pass_params=first_pass_params,
                 smooth_sigma=params.get("smooth_sigma", [0, 0.5, 1.0, 2.0]),
                 min_distance=params.get("min_distance", 10),
                 min_size=params.get("min_size", 100),
                 contrast_threshold_factor=params.get("contrast_threshold_factor", 1.5),
                 spacing=self.spacing,
                 anisotropy_normalization_degree=params.get("anisotropy_normalization_degree", 1.0)
             )
        except Exception as e:
             print(f"Error during segment_nuclei (refinement): {e}")
             return False

        final_seg_path = self.get_checkpoint_files()["final_segmentation"]
        try:
             # Use numpy save for .dat? No, tofile was used. Keep it.
             merged_roi_array.tofile(final_seg_path)
             # Add as layer, need to memmap to view? Or just pass array? Pass array for now.
             self._add_layer_safely(viewer, merged_roi_array, "Final segmentation")
             print(f"Saved final segmentation to {final_seg_path}")
             return True
        except Exception as e:
             print(f"Error saving or adding layer for final segmentation: {e}")
             return False


    # --- Paste the entire execute_calculate_features method here ---
    def execute_calculate_features(self, viewer, image_stack, params: Dict):
        print(f"Calculating {self.mode_name} features...")
        final_seg_path = self.get_checkpoint_files()["final_segmentation"]
        if not os.path.exists(final_seg_path):
             print("Error: Final segmentation file not found.")
             return False

        try:
            merged_roi_array = np.memmap(
                final_seg_path, dtype=np.int32, mode='r', shape=self.image_shape
            )
        except Exception as e:
            print(f"Error memory mapping final segmentation {final_seg_path}: {e}")
            return False

        try:
             distances_matrix, points_matrix, lines = shortest_distance(merged_roi_array, spacing=self.spacing)
             metrics_df, _ = analyze_segmentation( # No ramification for nuclei
                  merged_roi_array, spacing=self.spacing, calculate_skeletons=False
             )
        except Exception as e:
             print(f"Error calculating features (shortest_distance/analyze_segmentation): {e}")
             return False

        # Save results
        files = self.get_checkpoint_files()
        try:
            distances_matrix.to_csv(files["distances_matrix"])
            points_matrix.to_csv(files["points_matrix"])
            metrics_df.to_csv(files["metrics_df"])
            # Save lines (check shape and handle potential errors)
            if lines is not None and lines.size > 0:
                 # Reshape safely
                 lines_reshaped = lines.reshape(-1, lines.shape[-1] * lines.shape[-2]) # Flatten inner dims
                 pd.DataFrame(lines_reshaped).to_csv(files["lines"], header=False, index=False)
            else:
                 print("Warning: No lines generated or lines array is empty. Skipping lines save.")
                 # Create empty file? Or just skip? Skip for now.

            print(f"Saved feature matrices and metrics for {self.mode_name}.")
        except Exception as e:
             print(f"Error saving feature results to CSV: {e}")
             # Continue to add layers if possible

        # Add visualization
        if lines is not None and lines.size > 0:
             self._add_layer_safely(viewer, lines, "Closest Points Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)

        # Calculate and save depth df
        depth_df_loc = files["depth_df"]
        if not os.path.exists(depth_df_loc):
             try:
                  print("Calculating depth dataframe...")
                  depth_df = calculate_depth_df(merged_roi_array, self.spacing)
                  depth_df.to_csv(depth_df_loc)
                  print(f"Saved depth dataframe to {depth_df_loc}")
             except Exception as e:
                  print(f"Error calculating or saving depth dataframe: {e}")
                  # Don't fail the whole step for this

        # Close memmap
        if hasattr(merged_roi_array, '_mmap'):
            merged_roi_array._mmap.close()

        return True

    # --- Paste the entire load_checkpoint_data method here ---
    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        files = self.get_checkpoint_files()
        print(f"Loading checkpoint data for step {checkpoint_step} ({self.mode_name})...")
        if checkpoint_step >= 1:
            initial_seg_path = files["initial_segmentation"]
            if os.path.exists(initial_seg_path):
                try:
                    labeled_cells = np.load(initial_seg_path)
                    self._add_layer_safely(viewer, labeled_cells, "Intermediate segmentation")
                    print(f"Loaded initial segmentation ({self.mode_name})")
                    # Try to load first_pass_params for potential resume at step 2
                    config_file = files["config"]
                    if os.path.exists(config_file):
                         try:
                              with open(config_file, 'r') as file:
                                   saved_config = yaml.safe_load(file)
                                   if 'first_pass_params' in saved_config:
                                       self.intermediate_state['first_pass_params'] = saved_config['first_pass_params']
                                       print("Loaded first_pass_params from saved config.")
                         except Exception as e:
                              print(f"Warning: Error loading first_pass_params from config {config_file}: {e}")
                except Exception as e:
                     print(f"Error loading initial segmentation file {initial_seg_path}: {e}")
            else:
                print(f"Checkpoint file not found: {initial_seg_path}")

        if checkpoint_step >= 2:
            final_seg_path = files["final_segmentation"]
            if os.path.exists(final_seg_path):
                 try:
                      # Need to memmap for viewing potentially large files
                      merged_roi_array = np.memmap(final_seg_path, dtype=np.int32, mode='r', shape=self.image_shape)
                      self._add_layer_safely(viewer, merged_roi_array, "Final segmentation")
                      print(f"Loaded final segmentation ({self.mode_name})")
                      # Keep memmap open? Napari should handle it.
                 except Exception as e:
                      print(f"Error loading final segmentation file {final_seg_path}: {e}")
            else:
                print(f"Checkpoint file not found: {final_seg_path}")

        if checkpoint_step >= 3:
            lines_path = files["lines"]
            if os.path.exists(lines_path):
                 try:
                      lines_df = pd.read_csv(lines_path, header=None)
                      if not lines_df.empty:
                           # Expect 6 columns: x1,y1,z1,x2,y2,z2 (adjust if order differs)
                           # Assuming order is start_coords(3), end_coords(3) per row
                           lines = lines_df.values.reshape(-1, 2, 3) # Reshape into (N, 2 points, 3 coords)
                           self._add_layer_safely(viewer, lines, "Closest Points Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)
                           print(f"Loaded connections ({self.mode_name})")
                      else:
                           print(f"Lines file is empty: {lines_path}")
                 except Exception as e:
                      print(f"Error loading or reshaping lines file {lines_path}: {e}")
            else:
                print(f"Checkpoint file not found: {lines_path}")

            metrics_path = files["metrics_df"]
            if os.path.exists(metrics_path):
                 try:
                      metrics_df = pd.read_csv(metrics_path)
                      print(f"Metrics file loaded. Analysis complete: {len(metrics_df)} {self.mode_name} cells found.")
                 except Exception as e:
                      print(f"Error loading metrics file {metrics_path}: {e}")
            else:
                 print(f"Checkpoint file not found: {metrics_path}")


    # --- Paste the entire cleanup_step_artifacts method here ---
    def cleanup_step_artifacts(self, viewer, step_number: int):
        files = self.get_checkpoint_files()
        print(f"Cleaning artifacts for step {step_number} ({self.mode_name})")
        if step_number == 1:
            self._remove_layer_safely(viewer, "Intermediate segmentation")
            self._remove_file_safely(files.get("initial_segmentation")) # Use .get for safety
        elif step_number == 2:
            self._remove_layer_safely(viewer, "Final segmentation")
            self._remove_file_safely(files.get("final_segmentation"))
        elif step_number == 3:
            self._remove_layer_safely(viewer, "Closest Points Connections")
            self._remove_file_safely(files.get("distances_matrix"))
            self._remove_file_safely(files.get("points_matrix"))
            self._remove_file_safely(files.get("lines"))
            self._remove_file_safely(files.get("metrics_df"))
            # No ramification files for nuclei
            self._remove_file_safely(files.get("depth_df")) # Remove depth df too

# --- END OF FILE gui/_3D_nuclear_strategy.py ---