# --- START OF FILE gui/_3D_ramified_strategy.py ---
import os
import numpy as np
import pandas as pd
from typing import Dict
import sys
import traceback

# Import base class using relative import
from .processing_strategies import ProcessingStrategy

# Add parent directory (project root) to path to find segmentation modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import necessary functions from sibling modules
try:
    from initial_3d_segmentation import segment_microglia_first_pass
    from ramified_segmenter import extract_soma_masks, separate_multi_soma_cells
    from calculate_features import shortest_distance, analyze_segmentation, calculate_depth_df
except ImportError as e:
    print(f"Error importing segmentation functions in _3D_ramified_strategy.py: {e}")
    print(f"Ensure initial_3d_segmentation.py, ramified_segmenter.py, and calculate_features.py are in the directory: {parent_dir}")
    raise


class RamifiedStrategy(ProcessingStrategy):

    def _get_mode_name(self) -> str:
        return "ramified"

    def execute_initial_segmentation(self, viewer, image_stack, params: Dict):
        print(f"Running initial {self.mode_name} segmentation with params: {params}")
        try:
            # --- Get tubular_scales, expecting a list ---
            tubular_scales_list = params.get("tubular_scales") # Get the value from the params dict

            # --- Add validation ---
            if tubular_scales_list is None:
                print("Warning: 'tubular_scales' not found in parameters. Using default list.")
                tubular_scales_list = [0.8, 1.0, 1.5, 2.0] # Default if missing
            elif not isinstance(tubular_scales_list, list):
                print(f"Warning: 'tubular_scales' parameter is not a list (type: {type(tubular_scales_list)}). Attempting fallback or using default.")
                # Optionally try to parse if it's a string that wasn't handled upstream? Unlikely here.
                tubular_scales_list = [0.8, 1.0, 1.5, 2.0] # Use default as fallback
            elif not all(isinstance(x, (float, int)) for x in tubular_scales_list):
                 print(f"Warning: Not all elements in 'tubular_scales' list are numbers: {tubular_scales_list}. Using default.")
                 tubular_scales_list = [0.8, 1.0, 1.5, 2.0] # Use default if elements are wrong type
            elif not tubular_scales_list: # Handle empty list case if needed
                 print(f"Warning: 'tubular_scales' list is empty. Using default.")
                 tubular_scales_list = [0.8, 1.0, 1.5, 2.0]

            # --- Get other parameters, ensuring correct types ---
            # Use .get with defaults and potential type casting/checking
            min_size_voxels = int(params.get("min_size", 2000)) # Expecting int based on updated YAML
            smooth_sigma = float(params.get("smooth_sigma", 1.3))
            low_threshold_percentile = float(params.get("low_threshold_percentile", 98.0))
            high_threshold_percentile = float(params.get("high_threshold_percentile", 100.0))
            brightness_cutoff_factor = float(params.get("brightness_cutoff_factor", 10000.0))
            edge_trim_distance_threshold = float(params.get("edge_trim_distance_threshold", 4.5))
            connect_max_gap_physical = float(params.get("connect_max_gap_physical", 1.0)) # Added based on function call signature


            print(f"Using tubular_scales: {tubular_scales_list}") # Confirm value being used

            labeled_cells, first_pass_params, edge_mask = segment_microglia_first_pass(
                image_stack,
                spacing=self.spacing,
                tubular_scales=tubular_scales_list, # Pass the validated list
                smooth_sigma=smooth_sigma,
                connect_max_gap_physical=connect_max_gap_physical, # Ensure this param exists in config or has default
                min_size_voxels=min_size_voxels,
                low_threshold_percentile=low_threshold_percentile,
                high_threshold_percentile=high_threshold_percentile,
                brightness_cutoff_factor=brightness_cutoff_factor,
                edge_trim_distance_threshold=edge_trim_distance_threshold,
            )
            # ... (rest of saving logic) ...
            initial_seg_path = self.get_checkpoint_files()["initial_segmentation"]
            try:
                np.save(initial_seg_path, labeled_cells)
                self._add_layer_safely(viewer, labeled_cells, "Intermediate segmentation")
                self._add_layer_safely(viewer, edge_mask, "Edge Mask", layer_type='image')
                print(f"Saved initial segmentation to {initial_seg_path}")
                return True
            except Exception as e:
                 print(f"Error saving or adding layers for initial segmentation: {e}")
                 return False

        except Exception as e:
             print(f"Error during segment_microglia_first_pass execution: {e}")
             print(traceback.format_exc())
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
             print(f"Error loading initial segmentation {initial_seg_path}: {e}")
             return False

        try:
            cell_bodies = extract_soma_masks(
                labeled_cells,
                params.get("small_object_percentile"),
                params.get("thickness_percentile")
            )
            merged_roi_array = separate_multi_soma_cells(
                labeled_cells,
                image_stack,
                cell_bodies,
                params.get('min_size_threshold')
            )
        except Exception as e:
             print(f"Error during ROI refinement (extract_soma/separate_multi_soma): {e}")
             return False

        # Save cell bodies and final segmentation
        files = self.get_checkpoint_files()
        cell_bodies_path = files.get("cell_bodies")
        final_seg_path = files.get("final_segmentation")
        try:
             if cell_bodies_path:
                  cell_bodies.tofile(cell_bodies_path)
             if final_seg_path:
                  merged_roi_array.tofile(final_seg_path)

             # Add visualization
             self._add_layer_safely(viewer, cell_bodies, "Cell bodies")
             # Add final segmentation layer (memmap for viewing?)
             self._add_layer_safely(viewer, merged_roi_array, "Final segmentation")
             print(f"Saved cell bodies to {cell_bodies_path} and final segmentation to {final_seg_path}")
             return True
        except Exception as e:
             print(f"Error saving or adding layers for refined ROIs: {e}")
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
            metrics_df, ramification_metrics = analyze_segmentation(
                merged_roi_array, spacing=self.spacing, calculate_skeletons=True # Calculate skeletons for ramified
            )
        except Exception as e:
             print(f"Error calculating features (shortest_distance/analyze_segmentation): {e}")
             # Close memmap before returning
             if hasattr(merged_roi_array, '_mmap'): merged_roi_array._mmap.close()
             return False


        # Save results
        files = self.get_checkpoint_files()
        skeleton_array = None # Initialize
        try:
            distances_matrix.to_csv(files["distances_matrix"])
            points_matrix.to_csv(files["points_matrix"])
            metrics_df.to_csv(files["metrics_df"])
            if lines is not None and lines.size > 0:
                lines_reshaped = lines.reshape(-1, lines.shape[-1] * lines.shape[-2])
                pd.DataFrame(lines_reshaped).to_csv(files["lines"], header=False, index=False)
            else:
                 print("Warning: No lines generated. Skipping save.")


            if ramification_metrics is not None:
                # Unpack carefully, check existence of components
                ramification_summary_df, branch_data_df, endpoint_data_df, skeleton_array = ramification_metrics

                if ramification_summary_df is not None:
                     ramification_summary_df.to_csv(files["ramification_summary"])
                if branch_data_df is not None:
                     branch_data_df.to_csv(files["branch_data"])
                if endpoint_data_df is not None:
                     endpoint_data_df.to_csv(files["endpoint_data"])

                if skeleton_array is not None:
                    np.save(files["skeleton_array"], skeleton_array)
                    self._add_layer_safely(viewer, skeleton_array, "Skeletons")
                print(f"Ramification metrics saved for {len(ramification_summary_df) if ramification_summary_df is not None else 'N/A'} cells.")
            else:
                print("No ramification metrics were generated.")

        except Exception as e:
             print(f"Error saving feature results: {e}")
             # Continue to visualization if possible


        # Add visualization for connections
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
                      # Edge mask is transient, probably don't need to reload unless debugging step 1
                 except Exception as e:
                      print(f"Error loading initial segmentation {initial_seg_path}: {e}")
            else:
                 print(f"Checkpoint file not found: {initial_seg_path}")

        if checkpoint_step >= 2:
            final_seg_path = files["final_segmentation"]
            if os.path.exists(final_seg_path):
                 try:
                      merged_roi_array = np.memmap(final_seg_path, dtype=np.int32, mode='r', shape=self.image_shape)
                      self._add_layer_safely(viewer, merged_roi_array, "Final segmentation")
                      print(f"Loaded final segmentation ({self.mode_name})")
                 except Exception as e:
                      print(f"Error loading final segmentation {final_seg_path}: {e}")
            else:
                 print(f"Checkpoint file not found: {final_seg_path}")

            cell_bodies_path = files["cell_bodies"]
            if os.path.exists(cell_bodies_path):
                 try:
                      # Reshape cell bodies array from file
                      cell_bodies = np.fromfile(cell_bodies_path, dtype=np.int32).reshape(self.image_shape)
                      self._add_layer_safely(viewer, cell_bodies, "Cell bodies")
                      print(f"Loaded cell bodies ({self.mode_name})")
                 except Exception as e:
                      print(f"Error loading cell bodies {cell_bodies_path}: {e}")
            # else: # Cell bodies might not exist if only step 1 was done before crash
            #      print(f"Checkpoint file not found or skipped: {cell_bodies_path}")


        if checkpoint_step >= 3:
            lines_path = files["lines"]
            if os.path.exists(lines_path):
                 try:
                      lines_df = pd.read_csv(lines_path, header=None)
                      if not lines_df.empty:
                           lines = lines_df.values.reshape(-1, 2, 3)
                           self._add_layer_safely(viewer, lines, "Closest Points Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)
                           print(f"Loaded connections ({self.mode_name})")
                      else:
                           print(f"Lines file empty: {lines_path}")
                 except Exception as e:
                      print(f"Error loading lines file {lines_path}: {e}")
            else:
                print(f"Checkpoint file not found: {lines_path}")

            skeleton_path = files["skeleton_array"]
            if os.path.exists(skeleton_path):
                 try:
                      skeleton_array = np.load(skeleton_path)
                      self._add_layer_safely(viewer, skeleton_array, "Skeletons")
                      print(f"Loaded skeletons ({self.mode_name})")
                 except Exception as e:
                      print(f"Error loading skeleton file {skeleton_path}: {e}")
            else:
                print(f"Checkpoint file not found: {skeleton_path}")

            metrics_path = files["metrics_df"]
            if os.path.exists(metrics_path):
                 try:
                      metrics_df = pd.read_csv(metrics_path)
                      print(f"Metrics loaded. Analysis complete: {len(metrics_df)} {self.mode_name} cells found.")
                 except Exception as e:
                      print(f"Error loading metrics file {metrics_path}: {e}")
            else:
                 print(f"Checkpoint file not found: {metrics_path}")

            ram_summary_path = files["ramification_summary"]
            if os.path.exists(ram_summary_path):
                 try:
                      ram_df = pd.read_csv(ram_summary_path)
                      print(f"Ramification metrics available for {len(ram_df)} cells.")
                 except Exception as e:
                      print(f"Error loading ramification summary file {ram_summary_path}: {e}")
            # else: Ramification might not have been calculated


    # --- Paste the entire cleanup_step_artifacts method here ---
    def cleanup_step_artifacts(self, viewer, step_number: int):
        files = self.get_checkpoint_files()
        print(f"Cleaning artifacts for step {step_number} ({self.mode_name})")
        if step_number == 1:
            self._remove_layer_safely(viewer, "Intermediate segmentation")
            self._remove_layer_safely(viewer, "Edge Mask")
            self._remove_file_safely(files.get("initial_segmentation"))
        elif step_number == 2:
            self._remove_layer_safely(viewer, "Final segmentation")
            self._remove_layer_safely(viewer, "Cell bodies")
            self._remove_file_safely(files.get("final_segmentation"))
            self._remove_file_safely(files.get("cell_bodies"))
        elif step_number == 3:
            self._remove_layer_safely(viewer, "Closest Points Connections")
            self._remove_layer_safely(viewer, "Skeletons")
            self._remove_file_safely(files.get("distances_matrix"))
            self._remove_file_safely(files.get("points_matrix"))
            self._remove_file_safely(files.get("lines"))
            self._remove_file_safely(files.get("metrics_df"))
            self._remove_file_safely(files.get("ramification_summary"))
            self._remove_file_safely(files.get("branch_data"))
            self._remove_file_safely(files.get("endpoint_data"))
            self._remove_file_safely(files.get("skeleton_array"))
            self._remove_file_safely(files.get("depth_df"))


# --- END OF FILE gui/_3D_ramified_strategy.py ---