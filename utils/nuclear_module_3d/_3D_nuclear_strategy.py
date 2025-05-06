# --- START OF MODIFIED FILE utils/nuclear_module_3d/_3D_nuclear_strategy.py ---
import os
import numpy as np
import pandas as pd
import yaml # type: ignore
import sys
from typing import Dict, List, Any # Keep List
import traceback
import gc # For garbage collection

# --- Corrected Import Logic ---
try:
    from .nuclear_segmenter import segment_nuclei
    print("Successfully imported nuclear_segmenter using relative import.")
except ImportError as e_ns:
    print(f"Error importing nuclear_segmenter relatively: {e_ns}")
    raise

utils_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if utils_dir not in sys.path:
    sys.path.append(utils_dir)
    print(f"Appended to sys.path for calculate_features: {utils_dir}")

try:
    # Assuming analyze_segmentation can return detailed outputs including distances and points
    from calculate_features import analyze_segmentation, calculate_depth_df
    # If shortest_distance is still a separate utility you might want to call for lines:
    # from calculate_features import shortest_distance
    print("Successfully imported calculate_features (analyze_segmentation, calculate_depth_df).")
except ImportError as e_cf:
    print(f"Error importing from calculate_features from {utils_dir}: {e_cf}")
    print(f"  Current sys.path: {sys.path}")
    raise
# --- End Corrected Import Logic ---

from ..high_level_gui.processing_strategies import ProcessingStrategy


class NuclearStrategy(ProcessingStrategy):

    def __init__(self, config, processed_dir, image_shape, spacing, scale_factor):
        super().__init__(config, processed_dir, image_shape, spacing, scale_factor)
        print(f"NuclearStrategy initialized for mode: {self.mode_name}")
        print(f"  Defined steps: {self._step_names}")

    def _get_mode_name(self) -> str:
        return "nuclei"

    def get_step_names(self) -> List[str]:
        return [
            "execute_initial_segmentation",
            "execute_refine_rois",
            "execute_calculate_features"
        ]

    def get_checkpoint_files(self) -> Dict[str, str]:
        base_files = super().get_checkpoint_files()
        mode_files = {
            "initial_segmentation": os.path.join(self.processed_dir, f"initial_segmentation_{self.mode_name}.npy"),
            "final_segmentation": os.path.join(self.processed_dir, f"final_segmentation_{self.mode_name}.npy"),
            "metrics_df": os.path.join(self.processed_dir, f"metrics_df_{self.mode_name}.csv"),
            "distances_matrix": os.path.join(self.processed_dir, f"distances_matrix_{self.mode_name}.csv"),
            "points_matrix": os.path.join(self.processed_dir, f"points_matrix_{self.mode_name}.csv"),
            "lines_for_viz": os.path.join(self.processed_dir, f"lines_for_viz_{self.mode_name}.npy"), # For saving reconstructed lines
            "depth_df": os.path.join(self.processed_dir, f"depth_df_{self.mode_name}.csv"),
        }
        base_files.update(mode_files)
        # Remove the old "lines" key if it was different from "lines_for_viz"
        base_files.pop("lines", None)
        return base_files

    def save_config(self, current_config: Dict[str, Any]):
        files = self.get_checkpoint_files()
        config_save_path = files.get("config")
        if not config_save_path:
            print("Error: Cannot save config, 'config' path not defined.")
            return

        config_to_save = {}
        for step_key, step_data in current_config.items():
            if step_key in ['voxel_dimensions', 'mode'] or (step_key.startswith("execute_") and isinstance(step_data, dict)):
                config_to_save[step_key] = step_data
            elif step_key == 'saved_state': pass

        if 'mode' not in config_to_save:
            config_to_save['mode'] = self.mode_name

        saved_state_dict = {}
        if 'segmentation_threshold' in self.intermediate_state:
            try:
                saved_state_dict['segmentation_threshold'] = float(self.intermediate_state['segmentation_threshold'])
            except (TypeError, ValueError): pass
        
        if 'first_pass_params' in self.intermediate_state:
            fpp = self.intermediate_state['first_pass_params']
            if isinstance(fpp, dict):
                 serializable_fpp = {}
                 for k, v in fpp.items():
                     if isinstance(v, np.ndarray): serializable_fpp[k] = v.tolist()
                     elif isinstance(v, (np.float32, np.float64, np.int32, np.int64)):
                         serializable_fpp[k] = float(v) if isinstance(v, (np.float32, np.float64)) else int(v)
                     elif isinstance(v, (int, float, str, bool, list, tuple, dict)): serializable_fpp[k] = v
                     else: print(f"Warning: Skipping non-serializable item '{k}' type {type(v)} in first_pass_params.")
                 saved_state_dict['first_pass_params'] = serializable_fpp
            else: print(f"Warning: 'first_pass_params' not dict, cannot save to config.")

        if saved_state_dict: config_to_save['saved_state'] = saved_state_dict
        
        try:
            with open(config_save_path, 'w') as file:
                yaml.safe_dump(config_to_save, file, default_flow_style=False, sort_keys=False)
            print(f"NuclearStrategy: Config saved to {config_save_path}.")
        except Exception as e:
            print(f"Error saving config YAML in NuclearStrategy: {e}"); traceback.print_exc()

    def execute_initial_segmentation(self, viewer, image_stack, params: Dict) -> bool:
        print(f"\n--- Executing Step: Initial Segmentation ({self.mode_name}) ---")
        try:
            if image_stack is None: print("  Error: Input image_stack is None."); return False
            labeled_cells, first_pass_params_dict = segment_nuclei(
                image_stack, first_pass=None,
                smooth_sigma=params.get("smooth_sigma", [0.5, 1.0, 2.0]),
                min_distance=params.get("min_distance", 10),
                min_size=int(params.get("min_size", 100)),
                spacing=self.spacing,
                anisotropy_normalization_degree=params.get("anisotropy_normalization_degree", 1.0)
            )
            self.intermediate_state['first_pass_params'] = first_pass_params_dict
            output_path = self.get_checkpoint_files()["initial_segmentation"]
            np.save(output_path, labeled_cells.astype(np.int32))
            self._add_layer_safely(viewer, labeled_cells.astype(np.int32), name="Initial Segmentation", layer_type='labels', opacity=0.7)
            print("--- Initial Segmentation: Success ---")
            return True
        except Exception as e:
            print(f"!!! ERROR in Initial Segmentation: {e} !!!"); traceback.print_exc(); return False

    def execute_refine_rois(self, viewer, image_stack, params: Dict) -> bool:
        print(f"\n--- Executing Step: Refine ROIs ({self.mode_name}) ---")
        try:
            if image_stack is None: print("  Error: Input image_stack is None."); return False
            initial_seg_path = self.get_checkpoint_files()["initial_segmentation"]
            if not os.path.exists(initial_seg_path): print(f"  Error: Initial segmentation file not found: {initial_seg_path}"); return False
            initial_labeled_cells = np.load(initial_seg_path)
            fpp_from_state = self.intermediate_state.get('first_pass_params')
            if fpp_from_state is None:
                print("  Attempting to load 'first_pass_params' from saved config as fallback...")
                config_file_path = self.get_checkpoint_files().get("config")
                if config_file_path and os.path.exists(config_file_path):
                    try:
                        with open(config_file_path, 'r') as f_cfg: loaded_cfg = yaml.safe_load(f_cfg)
                        fpp_from_state = loaded_cfg.get('saved_state', {}).get('first_pass_params')
                        if fpp_from_state: self.intermediate_state['first_pass_params'] = fpp_from_state; print("  Loaded 'first_pass_params' from config.")
                        else: print("  'first_pass_params' not in config's saved_state."); return False
                    except Exception as e_cfg_load: print(f"  Error loading first_pass_params from config: {e_cfg_load}"); return False
                else: print("  Config file for 'first_pass_params' not found."); return False
            
            final_segmentation = segment_nuclei(
                image_stack, first_pass=initial_labeled_cells, first_pass_params=fpp_from_state,
                smooth_sigma=params.get("smooth_sigma", [0.5, 1.0, 2.0]),
                min_distance=params.get("min_distance", 2.5),
                min_size=int(params.get("min_size", 200)),
                contrast_threshold_factor=params.get("contrast_threshold_factor", 1.5),
                spacing=self.spacing,
                anisotropy_normalization_degree=params.get("anisotropy_normalization_degree", 1.0),
                use_advanced_splitting=params.get("use_advanced_splitting", True)
            )
            output_path = self.get_checkpoint_files()["final_segmentation"]
            np.save(output_path, final_segmentation.astype(np.int32))
            self._add_layer_safely(viewer, final_segmentation.astype(np.int32), name="Final Segmentation", layer_type='labels', opacity=0.7)
            self.intermediate_state['final_segmentation_labels'] = final_segmentation.astype(np.int32)
            print("--- Refine ROIs: Success ---")
            return True
        except Exception as e:
            print(f"!!! ERROR in Refine ROIs: {e} !!!"); traceback.print_exc(); return False

    # --- Step 3: Calculate Features (Modified to align with Ramified Strategy pattern) ---
    def execute_calculate_features(self, viewer, image_stack, params: Dict) -> bool:
        print(f"\n--- Executing Step: Calculate Features ({self.mode_name}) ---")
        try:
            final_seg = None
            if 'final_segmentation_labels' in self.intermediate_state:
                final_seg = self.intermediate_state['final_segmentation_labels']
                print("  Using final segmentation from intermediate state.")
            else:
                final_seg_path = self.get_checkpoint_files()["final_segmentation"]
                if not os.path.exists(final_seg_path):
                    print(f"  Error: Final segmentation file not found: {final_seg_path}")
                    return False
                final_seg = np.load(final_seg_path)
                print(f"  Loaded final segmentation from: {final_seg_path}")
                self.intermediate_state['final_segmentation_labels'] = final_seg

            if final_seg is None: print("  Error: Final segmentation data is not available."); return False

            print("  Calling analyze_segmentation for metrics and detailed outputs...")
            # Assuming analyze_segmentation for nuclei does not produce skeletons
            # but can produce distance_matrix and all_pairs_points in detailed_outputs.
            metrics_df, detailed_outputs = analyze_segmentation(
                 segmented_array=final_seg,
                 spacing=self.spacing,
                 calculate_distances=True, # Request distances
                 calculate_skeletons=False, # Nuclei don't have skeletons in this context
                 return_detailed=True # Crucial to get detailed_outputs dictionary
            )
            
            print("  Calculating depth dataframe...")
            depth_df = calculate_depth_df(final_seg, self.spacing)

            # Save results
            files = self.get_checkpoint_files()
            if metrics_df is not None: metrics_df.to_csv(files["metrics_df"], index=False)
            else: print("Warning: metrics_df is None after analyze_segmentation.")
            
            if depth_df is not None: depth_df.to_csv(files["depth_df"], index=False)
            else: print("Warning: depth_df is None.")

            distances_matrix = None
            points_matrix = None
            lines_for_viz = None # This will be reconstructed for visualization

            if detailed_outputs:
                distances_matrix = detailed_outputs.get('distance_matrix')
                if distances_matrix is not None:
                    distances_matrix.to_csv(files["distances_matrix"], index=False)
                    print("  Saved distances_matrix.")
                else: print("  distance_matrix not found in detailed_outputs.")

                points_matrix = detailed_outputs.get('all_pairs_points') # Or a similar key
                if points_matrix is not None:
                    points_matrix.to_csv(files["points_matrix"], index=False)
                    print("  Saved points_matrix.")
                else: print("  points_matrix not found in detailed_outputs (key might be different, e.g., 'closest_points_coordinates').")

            # Reconstruct lines for visualization from metrics_df (similar to ramified)
            # This assumes metrics_df contains columns like 'point_on_self_z', 'point_on_neighbor_x' etc.
            # which would be populated by analyze_segmentation if calculate_distances=True
            if metrics_df is not None and not metrics_df.empty and 'shortest_distance_um' in metrics_df.columns:
                connections_df = metrics_df[
                    metrics_df['shortest_distance_um'].notna() &
                    np.isfinite(metrics_df['shortest_distance_um']) &
                    metrics_df['closest_neighbor_label'].notna() # Ensure a neighbor was found
                ].copy()
                
                # Define ZYX column names as expected by analyze_segmentation output for points
                # These might be like: 'centroid_z', 'centroid_y', 'centroid_x' for self,
                # and 'closest_neighbor_centroid_z' etc. for neighbor.
                # OR, if analyze_segmentation populates specific 'point_on_self/neighbor' columns:
                coord_cols_self = ['point_on_self_z', 'point_on_self_x', 'point_on_self_y'] # ZYX order
                coord_cols_neigh = ['point_on_neighbor_z', 'point_on_neighbor_x', 'point_on_neighbor_y'] # ZYX order

                if all(c in connections_df for c in coord_cols_self) and all(c in connections_df for c in coord_cols_neigh):
                    points1 = connections_df[coord_cols_self].values
                    points2 = connections_df[coord_cols_neigh].values
                    lines_for_viz = np.stack((points1, points2), axis=1)
                    lines_for_viz = lines_for_viz[~np.isnan(lines_for_viz).any(axis=(1,2))]
                    
                    if lines_for_viz.shape[0] > 0:
                        np.save(files["lines_for_viz"], lines_for_viz) # Save the reconstructed lines
                        print(f"  Saved {lines_for_viz.shape[0]} lines for visualization.")
                        self._add_layer_safely(viewer, lines_for_viz, "Closest Connections",
                                               layer_type='shapes', shape_type='line',
                                               edge_color='red', edge_width=1.5) # Changed color/width
                    else: print("  No valid connection lines to draw after NaN filter from metrics_df.")
                else:
                    print(f"  Warning: Required coordinate columns for line visualization not found in metrics_df.")
                    print(f"    Needed self: {coord_cols_self}, Needed neighbor: {coord_cols_neigh}")
                    print(f"    Available columns in metrics_df: {metrics_df.columns.tolist()}")
            else:
                print("  Not enough data in metrics_df to reconstruct connection lines.")


            self.intermediate_state['metrics_df'] = metrics_df
            if distances_matrix is not None: self.intermediate_state['distances_matrix'] = distances_matrix
            if points_matrix is not None: self.intermediate_state['points_matrix'] = points_matrix
            if lines_for_viz is not None: self.intermediate_state['lines_for_viz'] = lines_for_viz


            print(f"  Saved feature data.")
            print("--- Calculate Features: Success ---")
            return True
        except Exception as e:
            print(f"!!! ERROR in Calculate Features: {e} !!!"); traceback.print_exc(); return False
        finally:
            if 'final_seg' in locals() and final_seg is not None: del final_seg; gc.collect()


    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        print(f"Loading Nuclear checkpoint data for step {checkpoint_step}")
        files = self.get_checkpoint_files()
        layer_base_names_to_manage = ["Initial Segmentation", "Final Segmentation", "Closest Connections"]
        print("Pre-removing potentially existing layers...")
        for base_name in layer_base_names_to_manage: self._remove_layer_safely(viewer, base_name)
        gc.collect()

        if checkpoint_step >= 1:
            initial_seg_path = files.get("initial_segmentation")
            if initial_seg_path and os.path.exists(initial_seg_path):
                try:
                    data = np.load(initial_seg_path); self._add_layer_safely(viewer, data, "Initial Segmentation", layer_type='labels', opacity=0.7)
                    config_file = files.get("config")
                    if config_file and os.path.exists(config_file):
                        with open(config_file, 'r') as f_cfg: loaded_cfg = yaml.safe_load(f_cfg)
                        fpp_from_config = loaded_cfg.get('saved_state', {}).get('first_pass_params')
                        if fpp_from_config:
                            for key, val in fpp_from_config.items():
                                if isinstance(val, list) and key in ['spacing', 'isotropic_spacing', 'downsampled_shape']:
                                    fpp_from_config[key] = tuple(val)
                            self.intermediate_state['first_pass_params'] = fpp_from_config
                    print(f"  Loaded: {os.path.basename(initial_seg_path)}")
                except Exception as e: print(f"  Error loading initial_segmentation: {e}; {traceback.format_exc()}")

        if checkpoint_step >= 2:
            final_seg_path = files.get("final_segmentation")
            if final_seg_path and os.path.exists(final_seg_path):
                try:
                    data = np.load(final_seg_path, mmap_mode='r'); self._add_layer_safely(viewer, data, "Final Segmentation", layer_type='labels', opacity=0.7)
                    self.intermediate_state['final_segmentation_labels'] = data
                    print(f"  Loaded: {os.path.basename(final_seg_path)}")
                except Exception as e: print(f"  Error loading final_segmentation: {e}; {traceback.format_exc()}")

        if checkpoint_step >= 3:
            metrics_path = files.get("metrics_df")
            if metrics_path and os.path.exists(metrics_path):
                try: self.intermediate_state['metrics_df'] = pd.read_csv(metrics_path); print(f"  Loaded metrics_df")
                except Exception as e: print(f"  Error loading metrics_df: {e}")
            
            # Load reconstructed lines for visualization if they were saved
            lines_viz_path = files.get("lines_for_viz")
            if lines_viz_path and os.path.exists(lines_viz_path):
                try:
                    lines_data = np.load(lines_viz_path)
                    if lines_data.size > 0:
                        self._add_layer_safely(viewer, lines_data, "Closest Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1.5)
                        self.intermediate_state['lines_for_viz'] = lines_data
                        print(f"  Loaded and displayed lines_for_viz ({lines_data.shape[0]} lines)")
                except Exception as e: print(f"  Error loading lines_for_viz: {e}; {traceback.format_exc()}")
        
        print("--- Nuclear Checkpoint data loading complete ---")


    def cleanup_step_artifacts(self, viewer, step_number: int):
        print(f"Cleaning Nuclear artifacts for step {step_number}")
        files = self.get_checkpoint_files()
        if step_number == 1:
            self._remove_layer_safely(viewer, "Initial Segmentation")
            self._remove_file_safely(files.get("initial_segmentation"))
            if 'first_pass_params' in self.intermediate_state: del self.intermediate_state['first_pass_params']
        elif step_number == 2:
            self._remove_layer_safely(viewer, "Final Segmentation")
            self._remove_file_safely(files.get("final_segmentation"))
            if 'final_segmentation_labels' in self.intermediate_state: del self.intermediate_state['final_segmentation_labels']
        elif step_number == 3:
            self._remove_layer_safely(viewer, "Closest Connections")
            self._remove_file_safely(files.get("metrics_df"))
            self._remove_file_safely(files.get("distances_matrix"))
            self._remove_file_safely(files.get("points_matrix"))
            self._remove_file_safely(files.get("lines_for_viz"))
            self._remove_file_safely(files.get("depth_df"))
            # Clear relevant intermediate state
            for key in ['metrics_df', 'distances_matrix', 'points_matrix', 'lines_for_viz']:
                if key in self.intermediate_state: del self.intermediate_state[key]
        print(f"--- Nuclear Cleanup for step {step_number} complete ---")

# --- END OF MODIFIED FILE utils/nuclear_module_3d/_3D_nuclear_strategy.py ---