# --- START OF FILE utils/ramified_module_2d/_2D_ramified_strategy.py ---
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sys
import traceback
import gc
from shutil import rmtree
import time
import tempfile

#Import QMessageBox
from PyQt5.QtWidgets import QMessageBox # type: ignore

# Correct relative imports based on structure
from ..high_level_gui.processing_strategies import ProcessingStrategy
try:
    # Import the 2D segmentation functions
    from .initial_2d_segmentation import segment_microglia_first_pass_raw_2d, apply_hull_trimming_2d
    # Placeholder imports for 2D refinement and features (needs implementation later)
    from .ramified_segmenter_2d import extract_soma_masks_2d, refine_seeds_pca_2d, separate_multi_soma_cells_2d
    from .calculate_features_2d import analyze_segmentation_2d
except ImportError as e:
    expected_ramified_dir = os.path.dirname(os.path.abspath(__file__))
    expected_utils_dir = os.path.dirname(expected_ramified_dir)
    print(f"Error importing 2D segmentation functions in _2D_ramified_strategy.py: {e}")
    print(f"Ensure initial_2d_segmentation.py (with segment_microglia_first_pass_raw_2d, apply_hull_trimming_2d) exists.")
    print(f"Ensure ramified_segmenter_2d.py (placeholder) exists in: {expected_ramified_dir}")
    print(f"Ensure calculate_features.py exists in: {expected_utils_dir}")
    raise


class Ramified2DStrategy(ProcessingStrategy):
    """
    Processing strategy for ramified microglia in 2D.
    4-step workflow: Raw Seg, Trim Edges, Refine ROIs, Features.
    Uses NPY for intermediate segmentations.
    """

    def _get_mode_name(self) -> str:
        # --- Use a distinct mode name ---
        return "ramified_2d"

    def get_step_names(self) -> List[str]:
        """Returns the ordered list of LOGICAL method names for 2D processing steps."""
        # --- MODIFICATION: Remove _2d suffix from names returned here ---
        return [
            "execute_raw_segmentation",     # Step 1 (was execute_raw_segmentation_2d)
            "execute_trim_edges",           # Step 2 (was execute_trim_edges_2d)
            "execute_refine_rois",          # Step 3 (was execute_refine_rois_2d)
            "execute_calculate_features"    # Step 4 (was execute_calculate_features_2d)
        ]

    def get_checkpoint_files(self) -> Dict[str, str]:
        """Defines checkpoint file paths specific to the Ramified 2D strategy."""
        files = super().get_checkpoint_files() # Get defaults (config, metrics_df)
        mode_prefix = self.mode_name # Will be "ramified_2d"
        # --- Add/Override paths with the correct mode prefix ---
        files["raw_segmentation"] = os.path.join(self.processed_dir, f"raw_segmentation_{mode_prefix}.npy") # Step 1
        files["edge_mask"] = os.path.join(self.processed_dir, f"{mode_prefix}_edge_mask.npy")            # Step 2
        files["trimmed_segmentation"] = os.path.join(self.processed_dir, f"trimmed_segmentation_{mode_prefix}.npy") # Step 2
        files["cell_bodies"] = os.path.join(self.processed_dir, f"{mode_prefix}_cell_bodies.npy") # Step 3
        files["refined_rois"] = os.path.join(self.processed_dir, f"{mode_prefix}_refined_rois.npy") # Step 3
        files["final_segmentation"] = os.path.join(self.processed_dir, f"final_segmentation_{mode_prefix}.npy") # Step 3
        files["skeleton_array"] = os.path.join(self.processed_dir, f"skeleton_array_{mode_prefix}.npy") # Step 4 (if applicable)
        # CSV files
        files["distances_matrix"]=os.path.join(self.processed_dir, f"distances_matrix_{mode_prefix}.csv")
        files["points_matrix"]=os.path.join(self.processed_dir, f"points_matrix_{mode_prefix}.csv")
        files["branch_data"]=os.path.join(self.processed_dir, f"branch_data_{mode_prefix}.csv")
        # Ensure base keys are updated if overridden
        files["config"] = os.path.join(self.processed_dir, f"processing_config_{mode_prefix}.yaml")
        files["metrics_df"] = os.path.join(self.processed_dir, f"metrics_df_{mode_prefix}.csv")
        # Remove legacy/unused keys if any
        # ...

        return files

    # --- Step 1: Raw Segmentation (2D) ---
    def execute_raw_segmentation_2d(self, viewer, image_stack: Any, params: Dict) -> bool:
        """ Executes Step 1: Raw 2D Segmentation. Calls 2D processing function. """
        # --- Adapt for 2D: Expect image_stack to be 2D ---
        if image_stack is None or image_stack.ndim != 2:
            print(f"Error: Input image is required and must be 2D for {self.mode_name} mode. Got shape {image_stack.shape if image_stack is not None else 'None'}.")
            return False
        image_2d = image_stack # Rename for clarity

        print(f"Executing Step 1: Raw {self.mode_name} segmentation...")
        files = self.get_checkpoint_files()
        persistent_raw_npy_path = files.get("raw_segmentation")

        if not persistent_raw_npy_path: print("Error: Path 'raw_segmentation' (.npy) missing."); return False
        if not persistent_raw_npy_path.endswith('.npy'): persistent_raw_npy_path = os.path.splitext(persistent_raw_npy_path)[0] + ".npy"

        temp_raw_labels_dat_path = None
        temp_raw_labels_dir = None
        segmentation_threshold = 0.0
        first_pass_params = {}
        success_raw_func = False
        raw_labels_memmap_handle = None

        try:
            # --- Get parameters (Adjust names if needed, e.g., min_size_pixels) ---
            tubular_scales_list = params.get("tubular_scales", [0.8, 1.0, 1.5, 2.0]) # Keep scales
            min_size_pixels = int(params.get("min_size", 100)) # Use 'min_size' from config, assume pixels
            smooth_sigma = float(params.get("smooth_sigma", 1.0)) # Adjust default if needed
            low_threshold_percentile = float(params.get("low_threshold_percentile", 98.0))
            high_threshold_percentile = float(params.get("high_threshold_percentile", 100.0))
            connect_max_gap_physical = float(params.get("connect_max_gap_physical", 1.0))

            # --- Call 2D raw segmentation function ---
            # It returns path to temp .dat, its dir, threshold, params
            temp_raw_labels_dat_path, temp_raw_labels_dir, segmentation_threshold, first_pass_params = \
                segment_microglia_first_pass_raw_2d(
                    image=image_2d, spacing=self.spacing, # Pass 2D image and 2D spacing
                    tubular_scales=tubular_scales_list, smooth_sigma=smooth_sigma,
                    connect_max_gap_physical=connect_max_gap_physical, min_size_pixels=min_size_pixels, # Use pixels
                    low_threshold_percentile=low_threshold_percentile,
                    high_threshold_percentile=high_threshold_percentile
                )

            # --- Check result and STORE intermediate state IMMEDIATELY ---
            if temp_raw_labels_dat_path is None or not os.path.exists(temp_raw_labels_dat_path):
                print("Error: Raw 2D segmentation function failed or did not produce output temp path.")
                if temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir): rmtree(temp_raw_labels_dir, ignore_errors=True)
                return False
            else:
                success_raw_func = True
                self.intermediate_state['segmentation_threshold'] = segmentation_threshold
                self.intermediate_state['original_volume_ref'] = image_2d # Store 2D image ref
                print(f"  Raw 2D function success. Stored intermediate seg_threshold: {self.intermediate_state.get('segmentation_threshold')}")

            # --- Load temp DAT, Save persistent NPY, Visualize with np.load ---
            try:
                print(f"  Loading raw 2D labels from temp DAT {temp_raw_labels_dat_path}...")
                # Use self.image_shape which should be 2D (y, x)
                if not hasattr(self, 'image_shape') or len(self.image_shape) != 2: raise ValueError("Invalid 2D 'image_shape'.")
                raw_labels_memmap_handle = np.memmap(temp_raw_labels_dat_path, dtype=np.int32, mode='r', shape=self.image_shape)
                print("  LOADED temp DAT successfully.")

                print(f"  Saving to persistent NPY: {persistent_raw_npy_path}")
                np.save(persistent_raw_npy_path, raw_labels_memmap_handle)
                print("  SAVED persistent NPY successfully.")

                # --- Cleanup temporary DAT file and dir AFTER saving NPY ---
                print("  Cleaning up temp DAT resources...")
                if hasattr(raw_labels_memmap_handle, '_mmap') and raw_labels_memmap_handle._mmap is not None: raw_labels_memmap_handle._mmap.close()
                del raw_labels_memmap_handle; gc.collect(); raw_labels_memmap_handle = None
                if temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir): rmtree(temp_raw_labels_dir, ignore_errors=True)
                temp_raw_labels_dir = None; temp_raw_labels_dat_path = None

                print(f"  Loading persistent NPY for display: {persistent_raw_npy_path}...")
                if not os.path.exists(persistent_raw_npy_path): raise FileNotFoundError("Persistent NPY missing.")
                raw_labels_npy = np.load(persistent_raw_npy_path)
                print("  LOADED persistent NPY successfully.")

                # --- Add layer (uses mode name automatically) ---
                print("  Adding layer 'Raw Intermediate Segmentation'...")
                # _add_layer_safely expects ZYX scale, but for 2D input (shape YX),
                # Napari handles scale=(1,1) correctly. Let _add_layer_safely handle it.
                # It will use self.z_scale_factor=1.0 if spacing[0]/spacing[1] calculation is adapted.
                # Ensure spacing calculation in GUI Manager handles 2D gracefully.
                self._add_layer_safely(viewer, raw_labels_npy, "Raw Intermediate Segmentation")
                print("  ADDED layer successfully.")
                del raw_labels_npy; gc.collect()

                return True

            except Exception as e_save_disp:
                 print(f"ERROR inside 2D save/display block: {e_save_disp}")
                 traceback.print_exc()
                 if 'raw_labels_memmap_handle' in locals() and raw_labels_memmap_handle is not None and hasattr(raw_labels_memmap_handle, '_mmap') and raw_labels_memmap_handle._mmap is not None:
                     try: raw_labels_memmap_handle._mmap.close()
                     except: pass
                     del raw_labels_memmap_handle; gc.collect()
                 return False

        except Exception as e:
             print(f"Error during execute_raw_segmentation_2d main block: {e}")
             traceback.print_exc()
             self.intermediate_state.pop('segmentation_threshold', None)
             self.intermediate_state.pop('original_volume_ref', None)
             return False
        finally:
             if 'temp_raw_labels_dir' in locals() and temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir): rmtree(temp_raw_labels_dir, ignore_errors=True)
             if 'raw_labels_memmap_handle' in locals() and raw_labels_memmap_handle is not None and hasattr(raw_labels_memmap_handle, '_mmap'):
                 try: raw_labels_memmap_handle._mmap.close()
                 except: pass
                 del raw_labels_memmap_handle; gc.collect()


    # --- Step 2: Trim Edges (2D) ---
    def execute_trim_edges_2d(self, viewer, image_stack: Any, params: Dict) -> bool:
        """ Reads raw 2D NPY, performs 2D trimming, saves trimmed NPY and mask NPY. """
        print(f"Executing Step 2: Edge Trimming (2D)...")
        files = self.get_checkpoint_files()
        raw_seg_input_path = files.get("raw_segmentation")       # NPY input path
        trimmed_seg_output_path = files.get("trimmed_segmentation") # NPY output path
        edge_mask_output_path = files.get("edge_mask")             # NPY output path

        # --- Input validation ---
        if not raw_seg_input_path or not os.path.exists(raw_seg_input_path): print(f"Error: Raw 2D NPY missing: {raw_seg_input_path}."); return False
        if not trimmed_seg_output_path or not edge_mask_output_path: print("Error: Output paths missing for 2D trim step."); return False
        if 'segmentation_threshold' not in self.intermediate_state: print("Error: Seg threshold missing."); return False
        if 'original_volume_ref' not in self.intermediate_state: print("Error: Original 2D image missing."); return False
        # Ensure original image is 2D
        original_image = self.intermediate_state['original_volume_ref']
        if original_image is None or original_image.ndim != 2: print(f"Error: Original image ref is not 2D (shape {original_image.shape})."); return False

        segmentation_threshold = self.intermediate_state['segmentation_threshold']
        hull_boundary_mask = None
        temp_trimmed_dat_path = None
        temp_trimmed_dir = None
        temp_writable_dir = None

        try:
            # --- Get parameters ---
            edge_trim_distance_threshold=float(params.get("edge_trim_distance_threshold",4.5))
            hull_boundary_thickness_phys=float(params.get("hull_boundary_thickness_phys",2.0))
            brightness_cutoff_factor=float(params.get("brightness_cutoff_factor",1.5))
            min_size_pixels=int(params.get("min_size_pixels",50)) # Use pixel param name
            smoothing_iterations_param=int(params.get("smoothing_iterations",1))
            heal_iterations_param=int(params.get("heal_iterations",1))

            # --- Prepare writable DAT input for apply_hull_trimming_2d ---
            print(f"  Loading raw 2D NPY {raw_seg_input_path} into temp writable DAT...")
            if not hasattr(self, 'image_shape') or len(self.image_shape) != 2: raise ValueError("Invalid 2D image_shape.")
            raw_labels_npy = np.load(raw_seg_input_path); temp_writable_dir = tempfile.mkdtemp(prefix="trim_input_2d_")
            temp_writable_dat_path = os.path.join(temp_writable_dir, 'writable_raw_2d.dat')
            writable_raw_memmap = np.memmap(temp_writable_dat_path, dtype=np.int32, mode='w+', shape=self.image_shape)
            writable_raw_memmap[:] = raw_labels_npy[:]; writable_raw_memmap.flush(); del raw_labels_npy; gc.collect()
            print(f"  Created temporary writable 2D DAT: {temp_writable_dat_path}")
            # Close handle as apply_hull_trimming_2d takes path and outputs new file
            del writable_raw_memmap; gc.collect()

            # --- Call apply_hull_trimming_2d (reads temp DAT, returns new temp DAT path) ---
            print(f"  Calling apply_hull_trimming_2d (Input temp DAT: {temp_writable_dat_path})...")
            temp_trimmed_dat_path, temp_trimmed_dir, hull_boundary_mask = apply_hull_trimming_2d(
                raw_labels_path=temp_writable_dat_path,
                original_image=original_image, spacing=self.spacing, # Pass 2D image and spacing
                hull_boundary_thickness_phys=hull_boundary_thickness_phys, edge_trim_distance_threshold=edge_trim_distance_threshold,
                brightness_cutoff_factor=brightness_cutoff_factor, segmentation_threshold=segmentation_threshold,
                min_size_pixels=min_size_pixels, smoothing_iterations=smoothing_iterations_param, heal_iterations=heal_iterations_param)

            # --- Cleanup temporary writable input DAT ---
            if temp_writable_dir and os.path.exists(temp_writable_dir): rmtree(temp_writable_dir, ignore_errors=True); temp_writable_dir = None

            # Check result
            if temp_trimmed_dat_path is None or not os.path.exists(temp_trimmed_dat_path) or hull_boundary_mask is None:
                 print("Error: apply_hull_trimming_2d failed.");
                 if temp_trimmed_dir and os.path.exists(temp_trimmed_dir): rmtree(temp_trimmed_dir, ignore_errors=True); return False
            else: print(f"  apply_hull_trimming_2d created temp trimmed DAT: {temp_trimmed_dat_path}")

            # --- Save persistent NPY files FIRST ---
            trimmed_labels_memmap = None
            try:
                print(f"  Loading temp trimmed 2D DAT {temp_trimmed_dat_path} to save persistent NPYs...")
                if not hasattr(self, 'image_shape') or len(self.image_shape) != 2: raise ValueError("Invalid 2D image_shape.")
                trimmed_labels_memmap = np.memmap(temp_trimmed_dat_path, dtype=np.int32, mode='r', shape=self.image_shape)

                print(f"  Saving trimmed 2D result to NPY: {trimmed_seg_output_path}")
                np.save(trimmed_seg_output_path, trimmed_labels_memmap)
                print(f"    SAVED: {os.path.basename(trimmed_seg_output_path)}")

                print(f"  Saving 2D edge mask to NPY: {edge_mask_output_path}")
                np.save(edge_mask_output_path, hull_boundary_mask)
                print(f"    SAVED: {os.path.basename(edge_mask_output_path)}")

            except Exception as e_save: print(f"ERROR saving persistent 2D NPY files: {e_save}"); traceback.print_exc(); return False
            finally:
                # --- Cleanup temporary trimmed DAT resources AFTER saving NPYs ---
                print("  Cleaning up temp trimmed 2D DAT resources...")
                if trimmed_labels_memmap is not None and hasattr(trimmed_labels_memmap, '_mmap') and trimmed_labels_memmap._mmap is not None:
                    try: trimmed_labels_memmap._mmap.close()
                    except: pass
                if 'trimmed_labels_memmap' in locals() and trimmed_labels_memmap is not None: del trimmed_labels_memmap; gc.collect()
                if temp_trimmed_dir and os.path.exists(temp_trimmed_dir): rmtree(temp_trimmed_dir, ignore_errors=True)
                temp_trimmed_dir = None; temp_trimmed_dat_path = None

            # --- Visualize results using np.load AFTER saving ---
            try:
                print(f"  Loading persistent trimmed 2D NPY {trimmed_seg_output_path} for display...")
                if not os.path.exists(trimmed_seg_output_path): raise FileNotFoundError("Persistent trimmed NPY missing.")
                trimmed_labels_npy = np.load(trimmed_seg_output_path)
                self._add_layer_safely(viewer, trimmed_labels_npy, "Trimmed Intermediate Segmentation")
                del trimmed_labels_npy; gc.collect()
                print("    Added trimmed layer.")

                print(f"  Loading persistent 2D edge mask NPY {edge_mask_output_path} for display...")
                if not os.path.exists(edge_mask_output_path): raise FileNotFoundError("Persistent edge mask NPY missing.")
                edge_mask_npy = np.load(edge_mask_output_path)
                self._add_layer_safely(viewer, edge_mask_npy, "Edge Mask", layer_type='image', colormap='gray', blending='additive')
                del edge_mask_npy; gc.collect()
                print("    Added edge mask layer.")

                return True

            except Exception as e_viz: print(f"ERROR during 2D visualization: {e_viz}"); traceback.print_exc(); return True # Save succeeded

        except Exception as e: print(f"Error during 2D trim_edges setup/execution: {e}"); traceback.print_exc(); return False
        finally:
             if 'temp_writable_dir' in locals() and temp_writable_dir and os.path.exists(temp_writable_dir): rmtree(temp_writable_dir, ignore_errors=True)
             if 'temp_trimmed_dir' in locals() and temp_trimmed_dir and os.path.exists(temp_trimmed_dir): rmtree(temp_trimmed_dir, ignore_errors=True)


     # --- Step 3: Refine ROIs (2D - Placeholder Functions) ---
    def execute_refine_rois_2d(self, viewer, image_stack: Any, params: Dict):
        """ Executes Step 3: Refine 2D ROIs using placeholder 2D functions. """
        print(f"Refining {self.mode_name} ROIs...")
        files = self.get_checkpoint_files()
        initial_seg_path = files.get("trimmed_segmentation") # Input is trimmed 2D NPY
        if not os.path.exists(initial_seg_path): print("Error: Trimmed 2D segmentation file not found."); return False

        original_image = self.intermediate_state.get('original_volume_ref')
        if original_image is None or original_image.ndim != 2: print(f"Error: Original image ref is not 2D (shape {getattr(original_image, 'shape', 'None')})."); return False

        # --- Ensure self.spacing exists and has expected ZYX format ---
        if not hasattr(self, 'spacing') or len(self.spacing) != 3:
             print(f"Error: Strategy spacing attribute invalid or missing. Found: {getattr(self, 'spacing', 'None')}")
             return False
        # --- Extract the 2D YX spacing ---
        spacing_yx = tuple(self.spacing[1:])
        print(f"  Using extracted YX spacing for 2D refinement functions: {spacing_yx}")
        # --- End spacing extraction ---

        try:
             labeled_cells = np.load(initial_seg_path)
        except Exception as e: print(f"Error loading trimmed 2D segmentation {initial_seg_path}: {e}"); return False

        try:
            # --- Parameter Handling ---
            smallest_quantile = float(params.get("smallest_quantile", 0.05))
            min_fragment_size = int(params.get("min_fragment_size", 50)) # Pixels?
            target_aspect_ratio = float(params.get("target_aspect_ratio", 1.05))
            projection_percentile_crop = int(params.get("projection_percentile_crop", 10)) # Keep param, maybe PCA func uses it?
            intensity_weight = float(params.get("intensity_weight", 1.0)) # Default adjusted earlier
            # Get separation-specific params
            max_seed_centroid_dist = float(params.get("max_seed_centroid_dist", 15.0))
            min_path_intensity_ratio = float(params.get("min_path_intensity_ratio", 0.6))
            # Get post-merge param if added to config later
            post_merge_min_interface_pixels = int(params.get("post_merge_min_interface_pixels", 10)) # Default if not in config

            # --- Call 2D Functions with CORRECT spacing_yx ---
            print("  Calling extract_soma_masks_2d...")
            cell_bodies = extract_soma_masks_2d(
                    segmentation_mask=labeled_cells,
                    intensity_image=original_image, # Pass the intensity image
                    spacing=spacing_yx, # Pass 2D spacing
                    smallest_quantile=smallest_quantile, min_fragment_size=min_fragment_size,
            )
            print("  Calling refine_seeds_pca_2d...")
            refined_mask = refine_seeds_pca_2d(
                 cell_bodies, spacing_yx, # Pass 2D spacing
                 target_aspect_ratio=target_aspect_ratio,
                 projection_percentile_crop=projection_percentile_crop,
                 min_fragment_size=min_fragment_size
                 )
            print("  Calling separate_multi_soma_cells_2d...")
            merged_roi_array = separate_multi_soma_cells_2d(
                labeled_cells, original_image, # Pass 2D image
                refined_mask, spacing_yx, # Pass 2D spacing
                min_size_threshold=min_fragment_size, intensity_weight=intensity_weight,
                # Pass heuristics params
                max_seed_centroid_dist=max_seed_centroid_dist,
                min_path_intensity_ratio=min_path_intensity_ratio,
                post_merge_min_interface_pixels=post_merge_min_interface_pixels
            )
            print("  Placeholder functions executed.")
        except NotImplementedError:
            print("ERROR: 2D Refinement functions (extract_soma_masks_2d, etc.) not implemented yet.")
            msg = QMessageBox(); msg.setIcon(QMessageBox.Warning); msg.setText("Step Not Implemented"); msg.setInformativeText("The functions for 2D ROI refinement haven't been created yet."); msg.exec_(); return False
        except Exception as e:
             print(f"Error during 2D ROI refinement: {e}")
             print(traceback.format_exc())
             return False

        # --- Saving logic (Uses 2D NPY paths) ---
        cell_bodies_path = files.get("cell_bodies")
        refined_rois_path = files.get("refined_rois")
        final_seg_path = files.get("final_segmentation")
        try:
             if cell_bodies_path: np.save(cell_bodies_path, cell_bodies)
             if refined_rois_path: np.save(refined_rois_path, refined_mask)
             if final_seg_path: np.save(final_seg_path, merged_roi_array)

             # --- Add visualization ---
             self._add_layer_safely(viewer, cell_bodies, "Cell bodies")
             self._add_layer_safely(viewer, refined_mask, "Refined ROIs")
             self._add_layer_safely(viewer, merged_roi_array, "Final segmentation")

             print(f"Saved 2D cell bodies, refined ROIs, and final segmentation.")
             return True
        except Exception as e: print(f"Error saving or adding layers for 2D refined ROIs: {e}"); return False

    # --- Step 4: Calculate Features (2D - Assume backend handles 2D) ---
    def execute_calculate_features_2d(self, viewer, image_stack: Any, params: Dict):
        """ Executes Step 4: Calculate 2D features. Assumes analyze_segmentation handles 2D. """
        print(f"Calculating {self.mode_name} features...")
        files = self.get_checkpoint_files()
        final_seg_path = files.get("final_segmentation") # 2D NPY input

        if not os.path.exists(final_seg_path): print(f"Error: Final 2D segmentation file not found: {final_seg_path}."); return False

        # --- Extract 2D YX spacing ---
        if not hasattr(self, 'spacing') or len(self.spacing) != 3:
             print(f"Error: Strategy spacing attribute invalid or missing. Found: {getattr(self, 'spacing', 'None')}")
             return False
        spacing_yx = tuple(self.spacing[1:])
        print(f"  Using extracted YX spacing for feature calculation: {spacing_yx}")
        # --- End spacing extraction ---

        merged_roi_array = None
        try:
            print(f"Loading final 2D segmentation with np.load: {final_seg_path}")
            merged_roi_array = np.load(final_seg_path)
            # ... (Optional 2D shape check) ...
        except Exception as e: print(f"Error loading final 2D segmentation {final_seg_path}: {e}"); return False

        metrics_df = None
        detailed_outputs = {}
        try:
            # --- Pass CORRECT 2D spacing ---
            print("Running analyze_segmentation (assuming 2D support)...")
            metrics_df, detailed_outputs = analyze_segmentation_2d(
                segmented_array=merged_roi_array,
                spacing_yx=spacing_yx, # Pass 2D spacing
                calculate_distances=True,
                calculate_skeletons=True,
                skeleton_export_path=None,
                return_detailed=True)
            print("analyze_segmentation finished.")
        except Exception as e: print(f"Error during feature calculation (analyze_segmentation on 2D): {e}"); traceback.print_exc(); return False

        # --- Saving calculated features (Uses 2D NPY/CSV paths) ---
        print("Saving calculated 2D features...")
        skeleton_array_result = None
        try:
            if metrics_df is not None and not metrics_df.empty: metrics_df.to_csv(files["metrics_df"], index=False)
            if detailed_outputs:
                dist_matrix = detailed_outputs.get('distance_matrix')
                if dist_matrix is not None and not dist_matrix.empty: dist_matrix.to_csv(files["distances_matrix"])
                all_points = detailed_outputs.get('all_pairs_points')
                if all_points is not None and not all_points.empty: all_points.to_csv(files["points_matrix"], index=False)
                branch_data = detailed_outputs.get('detailed_branches')
                if branch_data is not None and not branch_data.empty: branch_data.to_csv(files["branch_data"], index=False)
                skeleton_array_result = detailed_outputs.get('skeleton_array')
                if skeleton_array_result is not None: np.save(files["skeleton_array"], skeleton_array_result) # Save 2D skeleton NPY

        except Exception as e: print(f"Error saving 2D feature results: {e}"); traceback.print_exc()

        # --- Visualization (Adapt for 2D) ---
        print("Preparing 2D visualizations...")
        if skeleton_array_result is not None:
             self._add_layer_safely(viewer, skeleton_array_result, "Skeletons", layer_type='labels')

        # Closest connections visualization MIGHT work if analyze_segmentation returns 2D points
        if metrics_df is not None and not metrics_df.empty and 'shortest_distance_um' in metrics_df.columns:
            connections_df = metrics_df[
                metrics_df['shortest_distance_um'].notna() & np.isfinite(metrics_df['shortest_distance_um']) & metrics_df['closest_neighbor_label'].notna()
            ].copy()
            if not connections_df.empty:
                # --- Use Y, X order for 2D shapes layer ---
                coord_cols_self = ['point_on_self_y', 'point_on_self_x'] # Assume these columns exist
                coord_cols_neigh = ['point_on_neighbor_y', 'point_on_neighbor_x']
                if all(c in connections_df for c in coord_cols_self) and all(c in connections_df for c in coord_cols_neigh):
                    try:
                        points1 = connections_df[coord_cols_self].values
                        points2 = connections_df[coord_cols_neigh].values
                        lines_to_draw = np.stack((points1, points2), axis=1)
                        lines_to_draw = lines_to_draw[~np.isnan(lines_to_draw).any(axis=(1,2))]
                        if lines_to_draw.shape[0] > 0:
                            self._add_layer_safely(viewer, lines_to_draw, "Closest Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)
                            print(f"Visualizing {lines_to_draw.shape[0]} closest 2D connections.")
                    except KeyError as e: print(f"ERROR: Missing YX coordinate columns for 2D shapes: {e}")
                    except Exception as e_lines: print(f"ERROR creating 2D connection lines: {e_lines}")
                else: print("Warning: Missing required YX coordinate columns for 2D connection visualization.")

        # --- Cleanup loaded array ---
        if merged_roi_array is not None: del merged_roi_array; gc.collect()
        print("2D Feature calculation step complete.")
        return True

    # --- Load Checkpoint Data (2D) ---
    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        """Loads 2D data artifacts using np.load."""
        files = self.get_checkpoint_files()
        print(f"Loading Ramified 2D checkpoint data artifacts up to step {checkpoint_step}...")
        layer_base_names_to_manage = [ # Same base names
            "Raw Intermediate Segmentation", "Trimmed Intermediate Segmentation", "Edge Mask",
            "Cell bodies", "Refined ROIs", "Final segmentation",
            "Skeletons", "Closest Connections"
        ]
        print("Pre-removing potentially existing layers...")
        for base_name in layer_base_names_to_manage: self._remove_layer_safely(viewer, base_name)
        gc.collect()

        # --- Load NPY data using np.load, similar logic to 3D but uses 2D paths ---
        if checkpoint_step >= 1: # Raw 2D NPY
            raw_seg_path = files.get("raw_segmentation")
            if raw_seg_path and os.path.exists(raw_seg_path):
                 try: data = np.load(raw_seg_path); self._add_layer_safely(viewer, data, "Raw Intermediate Segmentation")
                 except Exception as e: print(f"Error loading raw 2D NPY {raw_seg_path}: {e}")
        if checkpoint_step >= 2: # Trimmed 2D NPY, Edge Mask 2D NPY
            trimmed_seg_path = files.get("trimmed_segmentation")
            if trimmed_seg_path and os.path.exists(trimmed_seg_path):
                 try: data = np.load(trimmed_seg_path); self._add_layer_safely(viewer, data, "Trimmed Intermediate Segmentation")
                 except Exception as e: print(f"Error loading trimmed 2D NPY {trimmed_seg_path}: {e}")
            edge_mask_path = files.get("edge_mask")
            if edge_mask_path and os.path.exists(edge_mask_path):
                 try: data = np.load(edge_mask_path); self._add_layer_safely(viewer, data, "Edge Mask", layer_type='image', colormap='gray', blending='additive')
                 except Exception as e: print(f"Error loading edge mask 2D NPY {edge_mask_path}: {e}")
        if checkpoint_step >= 3: # Final 2D NPY, Cell Bodies 2D NPY, Refined 2D NPY
            final_seg_path = files.get("final_segmentation")
            if final_seg_path and os.path.exists(final_seg_path):
                 try: data = np.load(final_seg_path); self._add_layer_safely(viewer, data, "Final segmentation")
                 except Exception as e: print(f"Error loading final 2D NPY {final_seg_path}: {e}")
            cell_bodies_path = files.get("cell_bodies")
            if cell_bodies_path and os.path.exists(cell_bodies_path):
                 try: data = np.load(cell_bodies_path); self._add_layer_safely(viewer, data, "Cell bodies")
                 except Exception as e: print(f"Error loading cell bodies 2D NPY {cell_bodies_path}: {e}")
            refined_rois_path = files.get("refined_rois")
            if refined_rois_path and os.path.exists(refined_rois_path):
                 try: data = np.load(refined_rois_path); self._add_layer_safely(viewer, data, "Refined ROIs")
                 except Exception as e: print(f"Error loading refined ROIs 2D NPY {refined_rois_path}: {e}")
        if checkpoint_step >= 4: # Features (CSVs, 2D Skeleton NPY)
             metrics_path = files.get("metrics_df"); metrics_df = None
             if metrics_path and os.path.exists(metrics_path):
                  try: metrics_df = pd.read_csv(metrics_path); print(f" Loaded 2D Metrics CSV ({len(metrics_df)} rows)")
                  except Exception as e: print(f"Error loading 2D metrics CSV {metrics_path}: {e}")
             skeleton_path = files.get("skeleton_array")
             if skeleton_path and os.path.exists(skeleton_path):
                  try: data = np.load(skeleton_path); self._add_layer_safely(viewer, data, "Skeletons", layer_type='labels')
                  except Exception as e: print(f"Error loading 2D skeletons NPY {skeleton_path}: {e}")
             # Visualize 2D connections
             if metrics_df is not None and not metrics_df.empty and 'shortest_distance_um' in metrics_df.columns:
                # (Same logic as in execute_calculate_features_2d)
                connections_df = metrics_df[metrics_df['shortest_distance_um'].notna() & np.isfinite(metrics_df['shortest_distance_um']) & metrics_df['closest_neighbor_label'].notna()].copy()
                if not connections_df.empty:
                    coord_cols_self = ['point_on_self_y', 'point_on_self_x']; coord_cols_neigh = ['point_on_neighbor_y', 'point_on_neighbor_x']
                    if all(c in connections_df for c in coord_cols_self) and all(c in connections_df for c in coord_cols_neigh):
                        try:
                            points1 = connections_df[coord_cols_self].values; points2 = connections_df[coord_cols_neigh].values
                            lines_to_draw = np.stack((points1, points2), axis=1)
                            lines_to_draw = lines_to_draw[~np.isnan(lines_to_draw).any(axis=(1,2))]
                            if lines_to_draw.shape[0] > 0: self._add_layer_safely(viewer, lines_to_draw, "Closest Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)
                        except Exception as e_lines: print(f"ERROR loading 2D connection lines: {e_lines}")
                    else: print("Warning: Missing YX coordinate columns for loading 2D connections.")


    # --- Cleanup Artifacts (2D) ---
    def cleanup_step_artifacts(self, viewer, step_number: int):
        """Cleans 2D artifacts for Ramified 2D step_number (1-based)."""
        files = self.get_checkpoint_files()
        print(f"Cleaning Ramified 2D artifacts for step {step_number}...")

        # Remove layers and NPY/CSV files based on step number
        if step_number == 1:
            self._remove_layer_safely(viewer, "Raw Intermediate Segmentation")
            self._remove_file_safely(files.get("raw_segmentation"))
        elif step_number == 2:
            self._remove_layer_safely(viewer, "Trimmed Intermediate Segmentation")
            self._remove_layer_safely(viewer, "Edge Mask")
            self._remove_file_safely(files.get("trimmed_segmentation"))
            self._remove_file_safely(files.get("edge_mask"))
        elif step_number == 3:
             self._remove_layer_safely(viewer, "Final segmentation")
             self._remove_layer_safely(viewer, "Cell bodies")
             self._remove_layer_safely(viewer, "Refined ROIs")
             self._remove_file_safely(files.get("final_segmentation"))
             self._remove_file_safely(files.get("cell_bodies"))
             self._remove_file_safely(files.get("refined_rois"))
        elif step_number == 4:
            self._remove_layer_safely(viewer, "Closest Connections")
            self._remove_layer_safely(viewer, "Skeletons")
            self._remove_file_safely(files.get("metrics_df")); self._remove_file_safely(files.get("distances_matrix")); self._remove_file_safely(files.get("points_matrix")); self._remove_file_safely(files.get("branch_data")); self._remove_file_safely(files.get("skeleton_array"))
        print(f"Cleaned step {step_number} (2D) artifacts.")

# --- END OF FILE utils/ramified_module_2d/_2D_ramified_strategy.py ---