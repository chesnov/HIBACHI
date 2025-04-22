# --- START OF FILE gui/_3D_ramified_strategy.py ---
import os
import numpy as np
import pandas as pd
from typing import Dict
import sys
import traceback
import gc # Import garbage collector for cleanup

# Import base class using relative import
from ..high_level_gui.processing_strategies import ProcessingStrategy

# Add parent directory (project root) to path to find segmentation modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import necessary functions from sibling modules
try:
    from .initial_3d_segmentation import segment_microglia_first_pass
    from .ramified_segmenter import extract_soma_masks, refine_seeds_pca, separate_multi_soma_cells
    # Only import the main analysis function now
    from ..calculate_features import analyze_segmentation # Keep this import
except ImportError as e:
    print(f"Error importing segmentation functions in _3D_ramified_strategy.py: {e}")
    print(f"Ensure initial_3d_segmentation.py, ramified_segmenter.py, and calculate_features.py are in the directory: {parent_dir}")
    raise


class RamifiedStrategy(ProcessingStrategy):

    # --- NO CHANGE: Assume __init__ is handled by base class ---

    # --- NO CHANGE: Assume _get_mode_name is correct ---
    def _get_mode_name(self) -> str:
        return "ramified"

    # --- NO CHANGE: Assume get_checkpoint_files is defined in base/elsewhere correctly ---

    # --- execute_initial_segmentation --- (No Changes Needed Here)
    def execute_initial_segmentation(self, viewer, image_stack, params: Dict):
        print(f"Running initial {self.mode_name} segmentation with params: {params}")
        try:
            # --- Parameter Handling ---
            tubular_scales_list = params.get("tubular_scales", [0.8, 1.0, 1.5, 2.0])
            if not isinstance(tubular_scales_list, list) or not all(isinstance(x, (float, int)) for x in tubular_scales_list) or not tubular_scales_list:
                print(f"Warning: Invalid 'tubular_scales'. Using default: {[0.8, 1.0, 1.5, 2.0]}")
                tubular_scales_list = [0.8, 1.0, 1.5, 2.0]

            min_size_voxels = int(params.get("min_size", 2000))
            smooth_sigma = float(params.get("smooth_sigma", 1.3))
            low_threshold_percentile = float(params.get("low_threshold_percentile", 98.0))
            high_threshold_percentile = float(params.get("high_threshold_percentile", 100.0))
            brightness_cutoff_factor = float(params.get("brightness_cutoff_factor", 10000.0))
            edge_trim_distance_threshold = float(params.get("edge_trim_distance_threshold", 4.5))
            connect_max_gap_physical = float(params.get("connect_max_gap_physical", 1.0))

            print(f"Using tubular_scales: {tubular_scales_list}") # Confirm value being used

            labeled_cells, first_pass_params, edge_mask = segment_microglia_first_pass(
                image_stack,
                spacing=self.spacing, # Assumes self.spacing is set
                tubular_scales=tubular_scales_list,
                smooth_sigma=smooth_sigma,
                connect_max_gap_physical=connect_max_gap_physical,
                min_size_voxels=min_size_voxels,
                low_threshold_percentile=low_threshold_percentile,
                high_threshold_percentile=high_threshold_percentile,
                brightness_cutoff_factor=brightness_cutoff_factor,
                edge_trim_distance_threshold=edge_trim_distance_threshold,
            )
            # --- Saving logic ---
            initial_seg_path = self.get_checkpoint_files()["initial_segmentation"]
            try:
                np.save(initial_seg_path, labeled_cells)
                self._add_layer_safely(viewer, labeled_cells, "Intermediate segmentation")
                self._add_layer_safely(viewer, edge_mask, "Edge Mask", layer_type='image', colormap='gray', blending='additive')
                print(f"Saved initial segmentation to {initial_seg_path}")
                return True
            except Exception as e:
                 print(f"Error saving or adding layers for initial segmentation: {e}")
                 return False

        except Exception as e:
             print(f"Error during segment_microglia_first_pass execution: {e}")
             print(traceback.format_exc())
             return False


    # --- execute_refine_rois (RESTORED tofile Saving for cell_bodies) ---
    def execute_refine_rois(self, viewer, image_stack, params: Dict):
        # ... (Keep original implementation) ...
        print(f"Refining {self.mode_name} ROIs...")
        files = self.get_checkpoint_files()
        initial_seg_path = files["initial_segmentation"]
        if not os.path.exists(initial_seg_path):
             print("Error: Initial segmentation file not found.")
             return False
        try:
             labeled_cells = np.load(initial_seg_path)
        except Exception as e:
             print(f"Error loading initial segmentation {initial_seg_path}: {e}")
             return False

        try:
            # --- Parameter Handling ---
            smallest_quantile = float(params.get("smallest_quantile", 0.05))
            min_samples_for_median = int(params.get("min_samples_for_median", 5))
            soma_thresholds = params.get("soma_extraction_thresholds", [0.3, 0.4, 0.5, 0.6])
            min_fragment_size = int(params.get("min_fragment_size", 50))
            target_aspect_ratio = float(params.get("target_aspect_ratio", 1.1))
            projection_percentile_crop = int(params.get("projection_percentile_crop", 10))
            intensity_weight = float(params.get("intensity_weight", 0.5))

            cell_bodies = extract_soma_masks(
                labeled_cells,
                self.spacing, # Assumes self.spacing is set
                smallest_quantile=smallest_quantile,
                min_samples_for_median=min_samples_for_median,
                thresholds=soma_thresholds,
                min_fragment_size=min_fragment_size,
            )

            refined_mask = refine_seeds_pca(cell_bodies,
                     self.spacing, # Assumes self.spacing is set
                     target_aspect_ratio=target_aspect_ratio,
                     projection_percentile_crop=projection_percentile_crop,
                     min_fragment_size=min_fragment_size)

            merged_roi_array = separate_multi_soma_cells(
                labeled_cells,
                image_stack,
                refined_mask,
                self.spacing, # Assumes self.spacing is set
                min_fragment_size=min_fragment_size,
                intensity_weight=intensity_weight
            )
        except Exception as e:
             print(f"Error during ROI refinement (extract_soma/refine_pca/separate_multi_soma): {e}")
             print(traceback.format_exc())
             return False

        # --- Saving logic ---
        cell_bodies_path = files.get("cell_bodies")
        # Assuming refined_rois path is needed/wanted
        refined_rois_path = os.path.join(os.path.dirname(files["final_segmentation"]), f"{self.mode_name}_refined_rois.npy") # Construct path if needed
        final_seg_path = files.get("final_segmentation")
        try:
             if cell_bodies_path:
                 # --- RESTORED ORIGINAL SAVING METHOD ---
                 cell_bodies.astype(np.int32).tofile(cell_bodies_path) # Ensure dtype and use tofile
                 print(f"Saved cell bodies using tofile to {cell_bodies_path}")
                 # --- END RESTORATION ---
             if refined_rois_path: np.save(refined_rois_path, refined_mask) # Keep npy
             if final_seg_path: np.save(final_seg_path, merged_roi_array) # Keep npy

             # --- Add visualization ---
             self._add_layer_safely(viewer, cell_bodies, "Cell bodies")
             self._add_layer_safely(viewer, refined_mask, "Refined ROIs")
             self._add_layer_safely(viewer, merged_roi_array, "Final segmentation")

             print(f"Saved cell bodies, refined ROIs, and final segmentation.")
             return True
        except Exception as e:
             print(f"Error saving or adding layers for refined ROIs: {e}")
             return False


    # --- execute_calculate_features method (MODIFIED FOR SKELETON VIZ FIX) ---
    def execute_calculate_features(self, viewer, image_stack, params: Dict):
        print(f"Calculating {self.mode_name} features using updated backend...")
        files = self.get_checkpoint_files()
        final_seg_path = files["final_segmentation"]

        if not os.path.exists(final_seg_path):
             print(f"Error: Final segmentation file not found at {final_seg_path}.")
             return False

        merged_roi_array = None
        try:
            if not hasattr(self, 'image_shape') or not self.image_shape:
                raise ValueError("Strategy instance is missing 'image_shape' attribute.")
            print(f"Memory mapping final segmentation: {final_seg_path} with shape {self.image_shape}")
            merged_roi_array = np.memmap(final_seg_path, dtype=np.int32, mode='r', shape=self.image_shape)
        except Exception as e:
             print(f"Error loading/memory mapping final segmentation {final_seg_path}: {e}")
             if merged_roi_array is not None and isinstance(merged_roi_array, np.memmap): merged_roi_array._mmap.close()
             return False

        metrics_df = None
        detailed_outputs = {}
        try:
            if not hasattr(self, 'spacing') or not self.spacing:
                raise ValueError("Strategy instance is missing 'spacing' attribute.")
            print("Running analyze_segmentation...")
            metrics_df, detailed_outputs = analyze_segmentation(
                segmented_array=merged_roi_array,
                spacing=self.spacing,
                calculate_distances=True,
                calculate_skeletons=True,
                skeleton_export_path=None,
                return_detailed=True,
                n_jobs=params.get('n_jobs', None)
            )
            print("analyze_segmentation finished.")

        except Exception as e:
             print(f"Error during feature calculation (analyze_segmentation): {e}")
             print(traceback.format_exc())
             if merged_roi_array is not None and isinstance(merged_roi_array, np.memmap):
                 print("Closing segmentation memmap due to error.")
                 merged_roi_array._mmap.close()
             return False

        print("Saving calculated features...")
        skeleton_array_result = None
        try:
            if metrics_df is not None and not metrics_df.empty:
                metrics_df_path = files["metrics_df"]
                metrics_df.to_csv(metrics_df_path, index=False)
                print(f"Saved NEW comprehensive metrics to {metrics_df_path}")
            else: print("Warning: Main metrics DataFrame is empty or None.")

            if detailed_outputs:
                dist_matrix = detailed_outputs.get('distance_matrix')
                if dist_matrix is not None and not dist_matrix.empty:
                    dist_matrix.to_csv(files["distances_matrix"])
                    print(f"Saved distance matrix to {files['distances_matrix']}")

                all_points = detailed_outputs.get('all_pairs_points')
                if all_points is not None and not all_points.empty:
                    all_points.to_csv(files["points_matrix"], index=False)
                    print(f"Saved all pairs points to {files['points_matrix']}")

                branch_data = detailed_outputs.get('detailed_branches')
                if branch_data is not None and not branch_data.empty:
                    branch_data.to_csv(files["branch_data"], index=False)
                    print(f"Saved detailed branch data to {files['branch_data']}")

                skeleton_array_result = detailed_outputs.get('skeleton_array')
                if skeleton_array_result is not None:
                    np.save(files["skeleton_array"], skeleton_array_result)
                    print(f"Saved skeleton array to {files['skeleton_array']}")

        except Exception as e:
             print(f"Error saving feature results: {e}")
             print(traceback.format_exc())

        print("Preparing visualizations...")
        if skeleton_array_result is not None:
             # --- FIX: Remove colormap argument ---
             self._add_layer_safely(viewer, skeleton_array_result, "Skeletons", layer_type='labels')
             # --- END FIX ---
             print("Attempted to add Skeletons layer.")

        if metrics_df is not None and not metrics_df.empty and 'shortest_distance_um' in metrics_df.columns:
            connections_df = metrics_df[
                metrics_df['shortest_distance_um'].notna() &
                np.isfinite(metrics_df['shortest_distance_um']) &
                metrics_df['closest_neighbor_label'].notna()
            ].copy()
            if not connections_df.empty:
                coord_cols_self = ['point_on_self_z', 'point_on_self_x', 'point_on_self_y']
                coord_cols_neigh = ['point_on_neighbor_z', 'point_on_neighbor_x', 'point_on_neighbor_y']
                if all(c in connections_df for c in coord_cols_self) and all(c in connections_df for c in coord_cols_neigh):
                    points1 = connections_df[coord_cols_self].values
                    points2 = connections_df[coord_cols_neigh].values
                    lines_to_draw = np.stack((points1, points2), axis=1)
                    lines_to_draw = lines_to_draw[~np.isnan(lines_to_draw).any(axis=(1,2))]
                    if lines_to_draw.shape[0] > 0:
                        self._add_layer_safely(viewer, lines_to_draw, "Closest Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)
                        print(f"Visualizing {lines_to_draw.shape[0]} closest connections.")

        print("Closing segmentation memmap.")
        if merged_roi_array is not None and isinstance(merged_roi_array, np.memmap):
            merged_roi_array._mmap.close()
            del merged_roi_array
            gc.collect()

        print("Feature calculation step complete.")
        return True


    # --- load_checkpoint_data method (RESTORED cell_bodies loading & SKELETON VIZ FIX) ---
    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        files = self.get_checkpoint_files()
        print(f"Loading checkpoint data for step {checkpoint_step} ({self.mode_name})...")

        # Step 1 & 2 Data
        if checkpoint_step >= 1:
            initial_seg_path = files["initial_segmentation"]
            if os.path.exists(initial_seg_path):
                 try:
                      labeled_cells = np.load(initial_seg_path)
                      self._add_layer_safely(viewer, labeled_cells, "Intermediate segmentation")
                      print(f"Loaded initial segmentation ({self.mode_name})")
                 except Exception as e: print(f"Error loading initial segmentation {initial_seg_path}: {e}")

        if checkpoint_step >= 2:
            final_seg_path = files["final_segmentation"]
            if os.path.exists(final_seg_path):
                 try:
                      merged_roi_array = np.memmap(final_seg_path, dtype=np.int32, mode='r', shape=self.image_shape)
                      self._add_layer_safely(viewer, merged_roi_array, "Final segmentation")
                      print(f"Loaded final segmentation ({self.mode_name})")
                 except Exception as e: print(f"Error loading final segmentation {final_seg_path}: {e}")

            cell_bodies_path = files["cell_bodies"]
            if os.path.exists(cell_bodies_path):
                 try:
                      # --- FIX: RESTORED ORIGINAL LOADING METHOD ---
                      if not hasattr(self, 'image_shape') or not self.image_shape:
                          raise ValueError("Cannot load cell bodies from file without image_shape.")
                      # Ensure dtype matches what was saved with tofile (likely int32)
                      cell_bodies = np.fromfile(cell_bodies_path, dtype=np.int32).reshape(self.image_shape)
                      # --- END FIX ---
                      self._add_layer_safely(viewer, cell_bodies, "Cell bodies")
                      print(f"Loaded cell bodies ({self.mode_name}) using fromfile")
                 except Exception as e:
                      print(f"Error loading cell bodies {cell_bodies_path}: {e}") # Original error will appear here

            # Try loading refined ROIs (keep using np.load for this one)
            base_dir = os.path.dirname(files["final_segmentation"])
            refined_rois_path = files.get("refined_rois", os.path.join(base_dir, f"{self.mode_name}_refined_rois.npy"))
            if os.path.exists(refined_rois_path):
                 try:
                      refined_mask = np.load(refined_rois_path, allow_pickle=True) # Keep allow_pickle just in case
                      self._add_layer_safely(viewer, refined_mask, "Refined ROIs")
                      print(f"Loaded refined ROIs ({self.mode_name})")
                 except Exception as e: print(f"Error loading refined ROIs {refined_rois_path}: {e}")

        # Step 3 Data
        if checkpoint_step >= 3:
            metrics_path = files["metrics_df"]
            metrics_df = None
            if os.path.exists(metrics_path):
                 try:
                      metrics_df = pd.read_csv(metrics_path)
                      print(f"Loaded comprehensive metrics ({len(metrics_df)} rows).")
                 except Exception as e: print(f"Error loading metrics file {metrics_path}: {e}")

            skeleton_path = files["skeleton_array"]
            if os.path.exists(skeleton_path):
                 try:
                      skeleton_array = np.load(skeleton_path)
                      # --- FIX: Remove colormap argument ---
                      self._add_layer_safely(viewer, skeleton_array, "Skeletons", layer_type='labels')
                      # --- END FIX ---
                      print(f"Loaded skeletons ({self.mode_name})")
                 except Exception as e: print(f"Error loading skeleton file {skeleton_path}: {e}")

            if metrics_df is not None and not metrics_df.empty and 'shortest_distance_um' in metrics_df.columns:
                connections_df = metrics_df[
                    metrics_df['shortest_distance_um'].notna() &
                    np.isfinite(metrics_df['shortest_distance_um']) &
                    metrics_df['closest_neighbor_label'].notna()
                ].copy()
                if not connections_df.empty:
                    coord_cols_self = ['point_on_self_z', 'point_on_self_x', 'point_on_self_y']
                    coord_cols_neigh = ['point_on_neighbor_z', 'point_on_neighbor_x', 'point_on_neighbor_y']
                    if all(c in connections_df for c in coord_cols_self) and all(c in connections_df for c in coord_cols_neigh):
                        points1 = connections_df[coord_cols_self].values
                        points2 = connections_df[coord_cols_neigh].values
                        lines_to_draw = np.stack((points1, points2), axis=1)
                        lines_to_draw = lines_to_draw[~np.isnan(lines_to_draw).any(axis=(1,2))]
                        if lines_to_draw.shape[0] > 0:
                           self._add_layer_safely(viewer, lines_to_draw, "Closest Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)
                           print(f"Loaded/visualized {lines_to_draw.shape[0]} connections.")


    # --- cleanup_step_artifacts method (Keep previous corrected version) ---
    def cleanup_step_artifacts(self, viewer, step_number: int):
        files = self.get_checkpoint_files()
        print(f"Cleaning artifacts for step {step_number} ({self.mode_name})...")
        if step_number == 1:
            self._remove_layer_safely(viewer, "Intermediate segmentation")
            self._remove_layer_safely(viewer, "Edge Mask")
            self._remove_file_safely(files.get("initial_segmentation"))
            print("Cleaned initial segmentation artifacts.")
        elif step_number == 2:
            self._remove_layer_safely(viewer, "Final segmentation")
            self._remove_layer_safely(viewer, "Cell bodies")
            self._remove_layer_safely(viewer, "Refined ROIs")
            self._remove_file_safely(files.get("final_segmentation"))
            self._remove_file_safely(files.get("cell_bodies"))
            base_dir = os.path.dirname(files["final_segmentation"])
            refined_rois_path = files.get("refined_rois", os.path.join(base_dir, f"{self.mode_name}_refined_rois.npy"))
            self._remove_file_safely(refined_rois_path)
            print("Cleaned ROI refinement artifacts.")
        elif step_number == 3:
            self._remove_layer_safely(viewer, "Closest Connections")
            self._remove_layer_safely(viewer, "Skeletons")
            self._remove_file_safely(files.get("metrics_df"))
            self._remove_file_safely(files.get("distances_matrix"))
            self._remove_file_safely(files.get("points_matrix"))
            self._remove_file_safely(files.get("branch_data"))
            self._remove_file_safely(files.get("skeleton_array"))
            self._remove_file_safely(files.get("lines"))
            self._remove_file_safely(files.get("ramification_summary"))
            self._remove_file_safely(files.get("endpoint_data"))
            self._remove_file_safely(files.get("depth_df"))
            print("Cleaned feature calculation artifacts (including deprecated files).")


    # --- Helper methods (Assume these exist in ProcessingStrategy base class) ---
    # def _add_layer_safely(...): ...
    # def _remove_layer_safely(...): ...
    # def _remove_file_safely(...): ...

# --- END OF FILE gui/_3D_ramified_strategy.py ---