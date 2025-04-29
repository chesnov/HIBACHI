# --- START OF FILE utils/ramified_module_3d/_3D_ramified_strategy.py ---
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sys
import traceback
import gc
from shutil import rmtree # Keep for cleanup if needed by sub-functions
import time
import tempfile

# Correct relative imports based on structure
from ..high_level_gui.processing_strategies import ProcessingStrategy
try:
    # Import the refactored functions for raw seg and trimming
    from .initial_3d_segmentation import segment_microglia_first_pass_raw, apply_hull_trimming
    from .ramified_segmenter import extract_soma_masks, refine_seeds_pca, separate_multi_soma_cells
    from ..calculate_features import analyze_segmentation
except ImportError as e:
    expected_ramified_dir = os.path.dirname(os.path.abspath(__file__))
    expected_utils_dir = os.path.dirname(expected_ramified_dir)
    print(f"Error importing segmentation functions in _3D_ramified_strategy.py: {e}")
    print(f"Ensure initial_3d_segmentation.py (with segment_microglia_first_pass_raw, apply_hull_trimming), "
          f"ramified_segmenter.py are in: {expected_ramified_dir}")
    print(f"Ensure calculate_features.py is in: {expected_utils_dir}")
    raise


class RamifiedStrategy(ProcessingStrategy):
    """
    Processing strategy for ramified microglia.
    4-step workflow: Raw Seg, Trim Edges, Refine ROIs, Features.
    Uses NPY for intermediate segmentations.
    """

    def _get_mode_name(self) -> str:
        return "ramified"

    # --- IMPLEMENT New Abstract Method ---
    def get_step_names(self) -> List[str]:
        """Returns the ordered list of method names for processing steps."""
        return [
            "execute_raw_segmentation",     # Step 1
            "execute_trim_edges",           # Step 2
            "execute_refine_rois",          # Step 3
            "execute_calculate_features"    # Step 4
        ]

    # --- UPDATED get_checkpoint_files (New keys, all NPY for seg) ---
    def get_checkpoint_files(self) -> Dict[str, str]:
        """Defines checkpoint file paths specific to the Ramified strategy."""
        files = super().get_checkpoint_files() # Get defaults (config, metrics_df)
        mode_prefix = self.mode_name
        # Add/Override paths specific to Ramified steps and outputs
        files["raw_segmentation"] = os.path.join(self.processed_dir, f"raw_segmentation_{mode_prefix}.npy") # Step 1 output
        files["edge_mask"] = os.path.join(self.processed_dir, f"{mode_prefix}_edge_mask.npy")            # Step 2 output
        files["trimmed_segmentation"] = os.path.join(self.processed_dir, f"trimmed_segmentation_{mode_prefix}.npy") # Step 2 output
        files["cell_bodies"] = os.path.join(self.processed_dir, f"{mode_prefix}_cell_bodies.npy") # Step 3 output
        files["refined_rois"] = os.path.join(self.processed_dir, f"{mode_prefix}_refined_rois.npy") # Step 3 output
        files["final_segmentation"] = os.path.join(self.processed_dir, f"final_segmentation_{mode_prefix}.npy") # Step 3 output (main labeled result)
        files["skeleton_array"] = os.path.join(self.processed_dir, f"skeleton_array_{mode_prefix}.npy") # Step 4 output
        # CSV files
        files["distances_matrix"]=os.path.join(self.processed_dir, f"distances_matrix_{mode_prefix}.csv")
        files["points_matrix"]=os.path.join(self.processed_dir, f"points_matrix_{mode_prefix}.csv")
        files["branch_data"]=os.path.join(self.processed_dir, f"branch_data_{mode_prefix}.csv")
        # Deprecated keys (for cleanup)
        files["lines"]=os.path.join(self.processed_dir, f"lines_{mode_prefix}.csv")
        files["ramification_summary"]=os.path.join(self.processed_dir, f"ramification_summary_{mode_prefix}.csv")
        files["endpoint_data"]=os.path.join(self.processed_dir, f"endpoint_data_{mode_prefix}.csv")
        files["depth_df"]=os.path.join(self.processed_dir, f"depth_df_{mode_prefix}.csv")
        # Ensure base keys
        if "config" not in files: files["config"] = os.path.join(self.processed_dir, f"processing_config_{mode_prefix}.yaml")
        if "metrics_df" not in files: files["metrics_df"] = os.path.join(self.processed_dir, f"metrics_df_{mode_prefix}.csv")

        # Remove old key if present from base class (unlikely now)
        files.pop("initial_segmentation", None)

        # print(f"DEBUG: RamifiedStrategy get_checkpoint_files returning keys: {list(files.keys())}") # Optional Debug
        return files

    # --- Step 1: execute_raw_segmentation (Saves raw NPY from temp DAT) ---
    def execute_raw_segmentation(self, viewer, image_stack: Any, params: Dict) -> bool:
        """
        Executes Step 1: Raw Segmentation.
        Calls the processing function (which uses temp .dat files), stores
        intermediate state, loads the temporary .dat result, saves it as a
        persistent .npy file, and loads/displays the .npy via np.load.

        Args:
            viewer: The Napari viewer instance.
            image_stack (np.ndarray or None): The input image stack. Required.
            params (Dict): Parameters for the raw segmentation step.

        Returns:
            bool: True if the step completed successfully (including saving),
                  False otherwise.
        """
        if image_stack is None:
            print("Error: Image stack is required for raw segmentation.")
            return False
        print(f"Executing Step 1: Raw {self.mode_name} segmentation...")
        files = self.get_checkpoint_files()
        persistent_raw_npy_path = files.get("raw_segmentation") # Expecting NPY path now

        if not persistent_raw_npy_path:
            print("Error: Path 'raw_segmentation' (.npy) missing in checkpoint files.")
            return False
        # Ensure the target path uses .npy (based on the NPY standardization fix)
        if not persistent_raw_npy_path.endswith('.npy'):
            print(f"Warning: Expected .npy path for raw_segmentation, got {persistent_raw_npy_path}. Adjusting.")
            persistent_raw_npy_path = os.path.splitext(persistent_raw_npy_path)[0] + ".npy"
            # It might be better to fix this in get_checkpoint_files if this warning appears


        # Variables to hold results from the segmentation function
        temp_raw_labels_dat_path = None # Path to temporary .dat file created by func
        temp_raw_labels_dir = None      # Directory containing the temp file
        segmentation_threshold = 0.0
        first_pass_params = {}
        success_raw_func = False # Flag to track if the core function succeeded
        raw_labels_memmap_handle = None # Define variable for memmap handle

        try:
            # --- Get parameters ---
            tubular_scales_list = params.get("tubular_scales", [0.8, 1.0, 1.5, 2.0])
            if not isinstance(tubular_scales_list, list) or not all(isinstance(x, (float, int)) for x in tubular_scales_list) or not tubular_scales_list:
                 print(f"Warning: Invalid 'tubular_scales'. Using default."); tubular_scales_list = [0.8, 1.0, 1.5, 2.0]
            min_size_voxels = int(params.get("min_size", 2000))
            smooth_sigma = float(params.get("smooth_sigma", 1.3))
            low_threshold_percentile = float(params.get("low_threshold_percentile", 98.0))
            high_threshold_percentile = float(params.get("high_threshold_percentile", 100.0))
            connect_max_gap_physical = float(params.get("connect_max_gap_physical", 1.0))

            # --- Call raw segmentation function ---
            # It returns path to temp .dat, its dir, threshold, params
            temp_raw_labels_dat_path, temp_raw_labels_dir, segmentation_threshold, first_pass_params = \
                segment_microglia_first_pass_raw(
                    volume=image_stack, spacing=self.spacing,
                    tubular_scales=tubular_scales_list, smooth_sigma=smooth_sigma,
                    connect_max_gap_physical=connect_max_gap_physical, min_size_voxels=min_size_voxels,
                    low_threshold_percentile=low_threshold_percentile,
                    high_threshold_percentile=high_threshold_percentile
                )

            # --- Check result and STORE intermediate state IMMEDIATELY ---
            if temp_raw_labels_dat_path is None or not os.path.exists(temp_raw_labels_dat_path):
                print("Error: Raw segmentation function failed or did not produce output temp path.")
                if temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir): rmtree(temp_raw_labels_dir, ignore_errors=True)
                return False # Exit before storing state
            else:
                success_raw_func = True # Mark that the function call succeeded
                self.intermediate_state['segmentation_threshold'] = segmentation_threshold
                self.intermediate_state['original_volume_ref'] = image_stack
                print(f"  Raw function success. Stored intermediate seg_threshold: {self.intermediate_state.get('segmentation_threshold')}") # Debug print
            # --- End immediate storage ---

            # --- Load temp DAT, Save persistent NPY, Visualize with np.load ---
            # This block handles the core conversion and display
            try:
                print(f"  Attempting to load raw labels from temp DAT {temp_raw_labels_dat_path}...") # DEBUG
                if not hasattr(self, 'image_shape'): raise ValueError("Missing 'image_shape' attribute for display.")
                # Load the temporary .dat file using memmap (read-only)
                raw_labels_memmap_handle = np.memmap(temp_raw_labels_dat_path, dtype=np.int32, mode='r', shape=self.image_shape)
                print(f"  LOADED temp DAT successfully.") # DEBUG

                print(f"  Attempting to save to persistent NPY: {persistent_raw_npy_path}") # DEBUG
                np.save(persistent_raw_npy_path, raw_labels_memmap_handle) # Save as NPY
                print(f"  SAVED persistent NPY successfully.") # DEBUG

                # --- Cleanup temporary DAT file and dir AFTER saving NPY ---
                print(f"  Attempting cleanup of temp DAT resources...") # DEBUG
                if hasattr(raw_labels_memmap_handle, '_mmap') and raw_labels_memmap_handle._mmap is not None:
                    raw_labels_memmap_handle._mmap.close(); print(f"    Closed handle to temp DAT.") # DEBUG
                else: print("    Temp DAT handle already closed or invalid.") # DEBUG
                del raw_labels_memmap_handle; gc.collect(); raw_labels_memmap_handle = None # Clear handle
                if temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir):
                    rmtree(temp_raw_labels_dir, ignore_errors=True); print(f"    Removed temp dir: {temp_raw_labels_dir}") # DEBUG
                temp_raw_labels_dir = None # Mark as cleaned
                temp_raw_labels_dat_path = None # Clear path too
                # --- End Temp Cleanup ---

                print(f"  Attempting to load persistent NPY for display: {persistent_raw_npy_path}...") # DEBUG
                # --- Visualize using np.load ---
                if not os.path.exists(persistent_raw_npy_path): raise FileNotFoundError("Persistent NPY missing before display.")
                raw_labels_npy = np.load(persistent_raw_npy_path) # Load into memory
                print(f"  LOADED persistent NPY successfully.") # DEBUG

                print(f"  Attempting to add layer 'Raw Intermediate Segmentation'...") # DEBUG
                self._add_layer_safely(viewer, raw_labels_npy, "Raw Intermediate Segmentation")
                print(f"  ADDED layer successfully.") # DEBUG
                print("  Raw segmentation saved (.npy) and displayed.")
                del raw_labels_npy; gc.collect() # Clean up in-memory copy

                print(f"  DEBUG execute_raw_seg: Final intermediate_state before return True = {self.intermediate_state}") # DEBUG
                return True # Success

            except Exception as e_save_disp:
                 print(f"ERROR inside save/display block: {e_save_disp}") # Capture error here
                 traceback.print_exc()
                 # Ensure handle is closed if error occurred after opening it
                 if 'raw_labels_memmap_handle' in locals() and raw_labels_memmap_handle is not None and hasattr(raw_labels_memmap_handle, '_mmap') and raw_labels_memmap_handle._mmap is not None:
                     print(f"  Closing temp DAT handle after error in save/display block.")
                     try: raw_labels_memmap_handle._mmap.close()
                     except: pass
                     del raw_labels_memmap_handle; gc.collect()

                 return False # Fail step if final save/display fails


        except Exception as e:
             # Catch errors during the main try block (param getting, raw func call itself)
             print(f"Error during execute_raw_segmentation main block: {e}")
             traceback.print_exc()
             # Clear potentially partially stored state on error
             self.intermediate_state.pop('segmentation_threshold', None)
             self.intermediate_state.pop('original_volume_ref', None)
             return False # Return False if any outer exception occurred
        finally:
             # Final cleanup: Ensure temp directory is removed if it somehow still exists
             # and wasn't cleaned up after the successful NPY save.
             if 'temp_raw_labels_dir' in locals() and temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir):
                 print(f"Final cleanup check: Removing temp dir {temp_raw_labels_dir}")
                 rmtree(temp_raw_labels_dir, ignore_errors=True)
             # Ensure handle is cleaned up if outer exception happened after it was opened
             if 'raw_labels_memmap_handle' in locals() and raw_labels_memmap_handle is not None and hasattr(raw_labels_memmap_handle, '_mmap'):
                 print(f"  Closing temp DAT handle in outer finally block.")
                 try: raw_labels_memmap_handle._mmap.close()
                 except Exception as e_close: print(f"    Warn: Error closing handle in outer finally: {e_close}")
                 del raw_labels_memmap_handle; gc.collect()

    # Step 2: execute_trim_edges (Reads raw NPY, saves trimmed NPY and mask NPY)
    def execute_trim_edges(self, viewer, image_stack: Any, params: Dict) -> bool:
        """ Reads raw NPY, performs trimming, saves trimmed NPY and mask NPY. """
        print(f"Executing Step 2: Edge Trimming...")
        files = self.get_checkpoint_files()
        raw_seg_input_path = files.get("raw_segmentation")       # NPY input path
        trimmed_seg_output_path = files.get("trimmed_segmentation") # NPY output path
        edge_mask_output_path = files.get("edge_mask")             # NPY output path

        # --- Input validation ---
        if not raw_seg_input_path or not os.path.exists(raw_seg_input_path): print(f"Error: Raw NPY missing: {raw_seg_input_path}."); return False
        if not trimmed_seg_output_path or not edge_mask_output_path: print("Error: Output paths missing for trim step."); return False
        if 'segmentation_threshold' not in self.intermediate_state: print("Error: Seg threshold missing."); return False
        if 'original_volume_ref' not in self.intermediate_state: print("Error: Original volume missing."); return False

        segmentation_threshold = self.intermediate_state['segmentation_threshold']
        original_volume = self.intermediate_state['original_volume_ref']
        hull_boundary_mask = None
        temp_trimmed_dat_path = None # Path from apply_hull_trimming
        temp_trimmed_dir = None      # Dir from apply_hull_trimming
        temp_writable_dir = None # For try...finally cleanup

        try:
            # --- Get parameters ---
            edge_trim_distance_threshold=float(params.get("edge_trim_distance_threshold",4.5)); hull_boundary_thickness_phys=float(params.get("hull_boundary_thickness_phys",2.0)); brightness_cutoff_factor=float(params.get("brightness_cutoff_factor",1.5)); min_size_voxels=int(params.get("min_size_voxels",50)); smoothing_iterations_param=int(params.get("smoothing_iterations",1)); heal_iterations_param=int(params.get("heal_iterations",1))

            # --- Prepare writable DAT input for apply_hull_trimming ---
            print(f"  Loading raw NPY {raw_seg_input_path} into temp writable DAT...")
            if not hasattr(self, 'image_shape'): raise ValueError("Missing image_shape.")
            raw_labels_npy = np.load(raw_seg_input_path); temp_writable_dir = tempfile.mkdtemp(prefix="trim_input_")
            temp_writable_dat_path = os.path.join(temp_writable_dir, 'writable_raw.dat')
            writable_raw_memmap = np.memmap(temp_writable_dat_path, dtype=np.int32, mode='w+', shape=self.image_shape)
            writable_raw_memmap[:] = raw_labels_npy[:]; writable_raw_memmap.flush(); del raw_labels_npy; gc.collect()
            print(f"  Created temporary writable DAT: {temp_writable_dat_path}")
            # Keep handle open for apply_hull_trimming if it modifies in place,
            # OR close now if apply_hull_trimming reads by path and outputs new file.
            # Based on current apply_hull_trimming, it expects path and outputs new. Close handle.
            del writable_raw_memmap; gc.collect()

            # --- Call apply_hull_trimming (reads temp DAT, returns new temp DAT path) ---
            print(f"  Calling apply_hull_trimming (Input temp DAT: {temp_writable_dat_path})...")
            temp_trimmed_dat_path, temp_trimmed_dir, hull_boundary_mask = apply_hull_trimming(
                raw_labels_path=temp_writable_dat_path, # Pass path to temp writable DAT
                original_volume=original_volume, spacing=self.spacing,
                hull_boundary_thickness_phys=hull_boundary_thickness_phys, edge_trim_distance_threshold=edge_trim_distance_threshold,
                brightness_cutoff_factor=brightness_cutoff_factor, segmentation_threshold=segmentation_threshold,
                min_size_voxels=min_size_voxels, smoothing_iterations=smoothing_iterations_param, heal_iterations=heal_iterations_param)

            # --- Cleanup temporary writable input DAT ---
            if temp_writable_dir and os.path.exists(temp_writable_dir): print(f"  Cleaning up temp writable input dir: {temp_writable_dir}"); rmtree(temp_writable_dir, ignore_errors=True); temp_writable_dir = None

            # Check result of apply_hull_trimming
            if temp_trimmed_dat_path is None or not os.path.exists(temp_trimmed_dat_path) or hull_boundary_mask is None:
                 print("Error: apply_hull_trimming failed or did not produce output."); 
                 if temp_trimmed_dir and os.path.exists(temp_trimmed_dir): rmtree(temp_trimmed_dir, ignore_errors=True); return False
            else: print(f"  apply_hull_trimming created temp trimmed DAT: {temp_trimmed_dat_path}")

            # --- Save persistent NPY files FIRST ---
            trimmed_labels_memmap = None # Define handle here for finally block
            try:
                print(f"  Loading temp trimmed DAT {temp_trimmed_dat_path} to save persistent NPYs...")
                if not hasattr(self, 'image_shape'): raise ValueError("Missing image_shape.")
                trimmed_labels_memmap = np.memmap(temp_trimmed_dat_path, dtype=np.int32, mode='r', shape=self.image_shape)

                print(f"  Saving trimmed result to persistent NPY: {trimmed_seg_output_path}")
                np.save(trimmed_seg_output_path, trimmed_labels_memmap)
                print(f"    SAVED: {os.path.basename(trimmed_seg_output_path)}")

                print(f"  Saving edge mask to persistent NPY: {edge_mask_output_path}")
                np.save(edge_mask_output_path, hull_boundary_mask)
                print(f"    SAVED: {os.path.basename(edge_mask_output_path)}")

            except Exception as e_save:
                 print(f"ERROR saving persistent NPY files: {e_save}")
                 traceback.print_exc()
                 return False # Critical failure if we can't save results
            finally:
                # --- Cleanup temporary trimmed DAT resources AFTER saving NPYs ---
                print(f"  Attempting cleanup of temp trimmed DAT resources...")
                if trimmed_labels_memmap is not None and hasattr(trimmed_labels_memmap, '_mmap') and trimmed_labels_memmap._mmap is not None:
                    try: trimmed_labels_memmap._mmap.close(); print(f"    Closed handle to temp trimmed DAT.")
                    except: print("    Warn: Error closing temp trimmed DAT handle.")
                if 'trimmed_labels_memmap' in locals() and trimmed_labels_memmap is not None: del trimmed_labels_memmap; gc.collect()
                if temp_trimmed_dir and os.path.exists(temp_trimmed_dir):
                    try: rmtree(temp_trimmed_dir, ignore_errors=True); print(f"    Removed temp trimmed dir: {temp_trimmed_dir}")
                    except Exception as e_rm: print(f"    Warn: Error removing temp dir {temp_trimmed_dir}: {e_rm}")
                temp_trimmed_dir = None; temp_trimmed_dat_path = None # Mark as cleaned
                # --- End Temp Cleanup ---


            # --- Visualize results using np.load AFTER saving is complete ---
            try:
                print(f"  Loading persistent trimmed NPY {trimmed_seg_output_path} for display...")
                if not os.path.exists(trimmed_seg_output_path): raise FileNotFoundError("Persistent trimmed NPY missing after save.")
                trimmed_labels_npy = np.load(trimmed_seg_output_path)
                self._add_layer_safely(viewer, trimmed_labels_npy, "Trimmed Intermediate Segmentation")
                del trimmed_labels_npy; gc.collect()
                print("    Added trimmed layer.")

                print(f"  Loading persistent edge mask NPY {edge_mask_output_path} for display...")
                if not os.path.exists(edge_mask_output_path): raise FileNotFoundError("Persistent edge mask NPY missing after save.")
                edge_mask_npy = np.load(edge_mask_output_path)
                self._add_layer_safely(viewer, edge_mask_npy, "Edge Mask", layer_type='image', colormap='gray', blending='additive')
                del edge_mask_npy; gc.collect()
                print("    Added edge mask layer.")

                print("  Trimmed segmentation (.npy) and mask (.npy) saved and displayed.")
                return True # Success

            except Exception as e_viz:
                 print(f"ERROR during visualization block: {e_viz}")
                 traceback.print_exc()
                 # Saving succeeded, but visualization failed. Return True but log error.
                 # Depending on requirements, could return False here.
                 return True

        except Exception as e:
             # Catch errors from parameter getting, prep, or apply_hull_trimming call
             print(f"Error during trim_edges step setup or execution: {e}")
             traceback.print_exc()
             return False
        finally:
             # Final safety cleanup for temp dirs
             if 'temp_writable_dir' in locals() and temp_writable_dir and os.path.exists(temp_writable_dir): print(f"Final cleanup check: Removing temp writable dir {temp_writable_dir}"); rmtree(temp_writable_dir, ignore_errors=True)
             if 'temp_trimmed_dir' in locals() and temp_trimmed_dir and os.path.exists(temp_trimmed_dir): print(f"Final cleanup check: Removing trimmed labels temp dir {temp_trimmed_dir}"); rmtree(temp_trimmed_dir, ignore_errors=True)
    
    # --- execute_refine_rois (RESTORED tofile Saving for cell_bodies) ---
    def execute_refine_rois(self, viewer, image_stack, params: Dict):
        # ... (Keep original implementation) ...
        print(f"Refining {self.mode_name} ROIs...")
        files = self.get_checkpoint_files()
        initial_seg_path = files["trimmed_segmentation"]
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
                relative_core_definition_ratios=soma_thresholds,
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
                min_size_threshold=min_fragment_size,
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
                 np.save(cell_bodies_path, cell_bodies) # Save as NPY
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
        
    # --- Step 4: execute_calculate_features (Loads final NPY with np.load, saves features) ---
    def execute_calculate_features(self, viewer, image_stack, params: Dict):
        print(f"Calculating {self.mode_name} features using updated backend...")
        files = self.get_checkpoint_files()
        final_seg_path = files["final_segmentation"]

        if not os.path.exists(final_seg_path):
             print(f"Error: Final segmentation file not found at {final_seg_path}.")
             return False

        merged_roi_array = None
        try:
            # --- MODIFICATION START: Use np.load instead of np.memmap ---
            print(f"Loading final segmentation with np.load: {final_seg_path}")
            merged_roi_array = np.load(final_seg_path)
            # --- MODIFICATION END ---

            # --- Optional but recommended: Check loaded shape ---
            if not hasattr(self, 'image_shape') or not self.image_shape:
                 print("Warning: Strategy instance is missing 'image_shape' attribute for comparison.")
            elif merged_roi_array.shape != self.image_shape:
                 print(f"CRITICAL WARNING: Shape mismatch after np.load!")
                 print(f"  Expected shape (from strategy): {self.image_shape}")
                 print(f"  Loaded shape (from file): {merged_roi_array.shape}")
                 print(f"  This may indicate an upstream error or file corruption.")
                 # Depending on severity, you might want to return False here
                 # return False
            else:
                 print(f"  Loaded shape matches expected shape: {merged_roi_array.shape}")
            # --- End shape check ---

        except Exception as e:
             print(f"Error loading final segmentation {final_seg_path} with np.load: {e}")
             # No memmap handle to close here
             return False

        metrics_df = None
        detailed_outputs = {}
        try:
            if not hasattr(self, 'spacing') or not self.spacing:
                raise ValueError("Strategy instance is missing 'spacing' attribute.")
            print("Running analyze_segmentation...")
            metrics_df, detailed_outputs = analyze_segmentation(
                segmented_array=merged_roi_array, # Pass the np.array loaded into memory
                spacing=self.spacing,
                calculate_distances=True,
                calculate_skeletons=True,
                skeleton_export_path=None, # Keep as None unless memmap specifically needed for skeleton output
                return_detailed=True,
                n_jobs=params.get('n_jobs', None)
            )
            print("analyze_segmentation finished.")

        except Exception as e:
             print(f"Error during feature calculation (analyze_segmentation): {e}")
             print(traceback.format_exc())
             # No memmap handle to close here
             return False

        # --- Saving calculated features (Unchanged) ---
        print("Saving calculated features...")
        skeleton_array_result = None
        try:
            # ... (saving logic for metrics_df, distance_matrix, points_matrix, branch_data, skeleton_array remains the same) ...
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
                # Note: analyze_segmentation might still return a memmap for the skeleton if skeleton_export_path was used internally somehow,
                # but the main input `merged_roi_array` is now guaranteed to be a standard numpy array.
                if skeleton_array_result is not None:
                    np.save(files["skeleton_array"], skeleton_array_result)
                    print(f"Saved skeleton array to {files['skeleton_array']}")

        except Exception as e:
             print(f"Error saving feature results: {e}")
             print(traceback.format_exc())
        # --- End Saving Logic ---


        # --- Visualization (Unchanged logic, uses potentially corrected coords from previous debug) ---
        print("Preparing visualizations...")
        if skeleton_array_result is not None:
             # Use potentially corrected coordinate order if that fix was applied
             self._add_layer_safely(viewer, skeleton_array_result, "Skeletons", layer_type='labels')
             print("Attempted to add Skeletons layer.")

        if metrics_df is not None and not metrics_df.empty and 'shortest_distance_um' in metrics_df.columns:
            connections_df = metrics_df[
                metrics_df['shortest_distance_um'].notna() &
                np.isfinite(metrics_df['shortest_distance_um']) &
                metrics_df['closest_neighbor_label'].notna()
            ].copy()
            if not connections_df.empty:
                # --- Ensure you are using the Z,Y,X order here if you applied that fix ---
                coord_cols_self = ['point_on_self_z', 'point_on_self_x', 'point_on_self_y']
                coord_cols_neigh = ['point_on_neighbor_z', 'point_on_neighbor_x', 'point_on_neighbor_y']
                # --- End coordinate order check ---

                if all(c in connections_df for c in coord_cols_self) and all(c in connections_df for c in coord_cols_neigh):
                    points1 = connections_df[coord_cols_self].values
                    points2 = connections_df[coord_cols_neigh].values
                    lines_to_draw = np.stack((points1, points2), axis=1)
                    lines_to_draw = lines_to_draw[~np.isnan(lines_to_draw).any(axis=(1,2))]
                    if lines_to_draw.shape[0] > 0:
                        self._add_layer_safely(viewer, lines_to_draw, "Closest Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)
                        print(f"Visualizing {lines_to_draw.shape[0]} closest connections.")
        # --- End Visualization Logic ---


        # --- MODIFICATION START: Remove memmap closing logic ---
        # print("Closing segmentation memmap.") # No longer applicable
        # if merged_roi_array is not None and isinstance(merged_roi_array, np.memmap): # This check is no longer needed
        #     merged_roi_array._mmap.close()
        # Explicitly delete the large array and suggest garbage collection if memory is tight
        if merged_roi_array is not None:
            del merged_roi_array
            gc.collect()
            print("Cleaned up loaded segmentation array from memory.")
        # --- MODIFICATION END ---

        print("Feature calculation step complete.")
        return True


    # --- load_checkpoint_data (Loads all NPY segmentations into memory for display) ---
    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        """Loads data artifacts using np.load for display consistency."""
        files = self.get_checkpoint_files()
        print(f"Loading Ramified checkpoint data artifacts up to step {checkpoint_step}...")

        # --- Force remove all potentially relevant layers before loading ---
        layer_base_names_to_manage = [
            "Raw Intermediate Segmentation", "Trimmed Intermediate Segmentation", "Edge Mask",
            "Cell bodies", "Refined ROIs", "Final segmentation",
            "Skeletons", "Closest Connections"
        ]
        print("Pre-removing potentially existing layers before loading checkpoint...")
        for base_name in layer_base_names_to_manage:
            self._remove_layer_safely(viewer, base_name) # Handles check and potential memmap closing
        gc.collect()

        # --- Load Data Based on Completed Step using np.load ---

        # Step 1 Data (Raw Seg NPY)
        if checkpoint_step >= 1:
            raw_seg_path = files.get("raw_segmentation") # NPY path
            if raw_seg_path and os.path.exists(raw_seg_path):
                 print(f"Attempt loading Raw NPY: {raw_seg_path}")
                 try: data = np.load(raw_seg_path); self._add_layer_safely(viewer, data, "Raw Intermediate Segmentation"); print(f" Added: {os.path.basename(raw_seg_path)}")
                 except Exception as e: print(f"Error loading raw NPY {raw_seg_path}: {e}"); traceback.print_exc()

        # Step 2 Data (Trimmed NPY + Edge Mask NPY)
        if checkpoint_step >= 2:
            trimmed_seg_path = files.get("trimmed_segmentation") # NPY path
            if trimmed_seg_path and os.path.exists(trimmed_seg_path):
                 print(f"Attempt loading Trimmed NPY: {trimmed_seg_path}")
                 try: data = np.load(trimmed_seg_path); self._add_layer_safely(viewer, data, "Trimmed Intermediate Segmentation"); print(f" Added: {os.path.basename(trimmed_seg_path)}")
                 except Exception as e: print(f"Error loading trimmed NPY {trimmed_seg_path}: {e}"); traceback.print_exc()
            edge_mask_path = files.get("edge_mask") # NPY
            if edge_mask_path and os.path.exists(edge_mask_path):
                 print(f"Attempt loading Edge Mask NPY: {edge_mask_path}")
                 try: data = np.load(edge_mask_path); self._add_layer_safely(viewer, data, "Edge Mask", layer_type='image', colormap='gray', blending='additive'); print(f" Added: {os.path.basename(edge_mask_path)}")
                 except Exception as e: print(f"Error loading edge mask NPY {edge_mask_path}: {e}")

        # Step 3 Data (Final NPY + Cell Bodies NPY + Refined NPY)
        if checkpoint_step >= 3:
            final_seg_path = files.get("final_segmentation") # NPY path
            if final_seg_path and os.path.exists(final_seg_path):
                 print(f"Attempt loading Final Seg NPY: {final_seg_path}")
                 try: data = np.load(final_seg_path); self._add_layer_safely(viewer, data, "Final segmentation"); print(f" Added: {os.path.basename(final_seg_path)}")
                 except Exception as e: print(f"Error loading final NPY {final_seg_path}: {e}"); traceback.print_exc()

            cell_bodies_path = files.get("cell_bodies") # NPY path
            if cell_bodies_path and os.path.exists(cell_bodies_path):
                 print(f"Attempt loading Cell Bodies NPY: {cell_bodies_path}")
                 try: data = np.load(cell_bodies_path); self._add_layer_safely(viewer, data, "Cell bodies"); print(f" Added: {os.path.basename(cell_bodies_path)}")
                 except Exception as e: print(f"Error loading cell bodies NPY {cell_bodies_path}: {e}")

            refined_rois_path = files.get("refined_rois") # NPY path
            if refined_rois_path and os.path.exists(refined_rois_path):
                 print(f"Attempt loading Refined ROIs NPY: {refined_rois_path}")
                 try: data = np.load(refined_rois_path); self._add_layer_safely(viewer, data, "Refined ROIs"); print(f" Added: {os.path.basename(refined_rois_path)}")
                 except Exception as e: print(f"Error loading refined ROIs NPY {refined_rois_path}: {e}")

        # Step 4 Data (Features)
        if checkpoint_step >= 4:
            # ... (loading logic for features - unchanged, loads CSVs and skeleton NPY) ...
             metrics_path = files.get("metrics_df"); metrics_df = None
             if metrics_path and os.path.exists(metrics_path):
                  print(f"Attempt loading Metrics CSV: {metrics_path}")
                  try: metrics_df = pd.read_csv(metrics_path); print(f" Loaded: {os.path.basename(metrics_path)} ({len(metrics_df)} rows)")
                  except Exception as e: print(f"Error loading metrics CSV {metrics_path}: {e}")
             skeleton_path = files.get("skeleton_array") # NPY
             if skeleton_path and os.path.exists(skeleton_path):
                  print(f"Attempt loading Skeletons NPY: {skeleton_path}")
                  try: data = np.load(skeleton_path); self._add_layer_safely(viewer, data, "Skeletons", layer_type='labels'); print(f" Loaded: {os.path.basename(skeleton_path)}")
                  except Exception as e: print(f"Error loading skeletons NPY {skeleton_path}: {e}")
             # Visualize connections
             if metrics_df is not None and not metrics_df.empty and 'shortest_distance_um' in metrics_df.columns:
                connections_df = metrics_df[
                    metrics_df['shortest_distance_um'].notna() &
                    np.isfinite(metrics_df['shortest_distance_um']) &
                    metrics_df['closest_neighbor_label'].notna()
                ].copy()
                if not connections_df.empty:
                    # --- !!! DOUBLE-CHECK THIS BLOCK !!! ---
                    # Use Z, Y, X order for Napari shapes layer
                    coord_cols_self = ['point_on_self_z', 'point_on_self_x', 'point_on_self_y'] # Swapped Y and X
                    coord_cols_neigh = ['point_on_neighbor_z', 'point_on_neighbor_x', 'point_on_neighbor_y'] # Swapped Y and X
                    print(f"DEBUG: Using coordinate columns for shapes: Self={coord_cols_self}, Neigh={coord_cols_neigh}") # Add this print
                    # --- !!! END DOUBLE-CHECK !!! ---

                    if all(c in connections_df for c in coord_cols_self) and all(c in connections_df for c in coord_cols_neigh):
                        try:
                            points1 = connections_df[coord_cols_self].values
                            points2 = connections_df[coord_cols_neigh].values
                            lines_to_draw = np.stack((points1, points2), axis=1)

                            # --- DEBUG: Print coordinates being used ---
                            if lines_to_draw.shape[0] > 0:
                                print(f"DEBUG: First few lines_to_draw coords (Z,Y,X):\n{lines_to_draw[:3]}")
                            # --- END DEBUG ---

                            lines_to_draw = lines_to_draw[~np.isnan(lines_to_draw).any(axis=(1,2))] # Check for NaNs

                            if lines_to_draw.shape[0] > 0:
                                # Pass these ZYX coordinates to _add_layer_safely
                                self._add_layer_safely(viewer, lines_to_draw, "Closest Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)
                                print(f"Visualizing {lines_to_draw.shape[0]} closest connections.")
                            else:
                                print("No valid connection lines to draw after NaN filter.")
                        except KeyError as e:
                            print(f"ERROR: Could not find required coordinate columns (Z,Y,X) in DataFrame: {e}")
                            print(f"Available columns: {connections_df.columns.tolist()}")
                        except Exception as e_lines:
                            print(f"ERROR creating or adding connection lines layer: {e_lines}")
                            traceback.print_exc()

                    else:
                        print(f"Warning: Could not find all required coordinate columns in metrics_df for connections visualization.")
                        print(f"  Need ZYX: {coord_cols_self} and {coord_cols_neigh}")
                        print(f"  Available: {connections_df.columns.tolist()}")


    # --- cleanup_step_artifacts (Handles all NPY segmentations) ---
    def cleanup_step_artifacts(self, viewer, step_number: int):
        """Cleans artifacts for Ramified step_number (1-based)."""
        files = self.get_checkpoint_files()
        print(f"Cleaning Ramified artifacts for step {step_number}...")

        # --- Close memmap handles before deleting files (memmap layers shouldn't exist now, but safe check) ---
        # Layers added by load_checkpoint_data are now standard arrays.
        # Layers added during processing *might* be memmap if processing outputted memmap directly,
        # but current logic loads intermediates for display. Let's remove the memmap check here.
        # gc.collect(); time.sleep(0.1) # Keep GC

        # --- Remove layers and files based on step number ---
        if step_number == 1: # Raw Seg NPY
            self._remove_layer_safely(viewer, "Raw Intermediate Segmentation")
            self._remove_file_safely(files.get("raw_segmentation")) # NPY
            print("Cleaned step 1 artifacts.")
        elif step_number == 2: # Trimmed Seg NPY, Edge Mask NPY
            self._remove_layer_safely(viewer, "Trimmed Intermediate Segmentation")
            self._remove_layer_safely(viewer, "Edge Mask")
            self._remove_file_safely(files.get("trimmed_segmentation")) # NPY
            self._remove_file_safely(files.get("edge_mask")) # NPY
            print("Cleaned step 2 artifacts.")
        elif step_number == 3: # Refine ROIs: final npy, cell bodies npy, refined npy
             # No memmap closing needed here as layers hold numpy arrays
             self._remove_layer_safely(viewer, "Final segmentation")
             self._remove_layer_safely(viewer, "Cell bodies")
             self._remove_layer_safely(viewer, "Refined ROIs")
             self._remove_file_safely(files.get("final_segmentation")) # NPY
             self._remove_file_safely(files.get("cell_bodies")) # NPY
             self._remove_file_safely(files.get("refined_rois")) # NPY
             print("Cleaned step 3 artifacts.")
        elif step_number == 4: # Features: metrics csv, skeleton npy, other csvs
            self._remove_layer_safely(viewer, "Closest Connections")
            self._remove_layer_safely(viewer, "Skeletons")
            # Remove files
            self._remove_file_safely(files.get("metrics_df")); self._remove_file_safely(files.get("distances_matrix")); self._remove_file_safely(files.get("points_matrix")); self._remove_file_safely(files.get("branch_data")); self._remove_file_safely(files.get("skeleton_array"))
            # Legacy
            self._remove_file_safely(files.get("lines")); self._remove_file_safely(files.get("ramification_summary")); self._remove_file_safely(files.get("endpoint_data")); self._remove_file_safely(files.get("depth_df"))
            print("Cleaned step 4 artifacts.")