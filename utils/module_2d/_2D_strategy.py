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
    from .initial_2d_segmentation import segment_cells_first_pass_raw_2d
    from .remove_artifacts_2d import apply_hull_trimming_2d
    from .ramified_segmenter_2d import extract_soma_masks_2d, separate_multi_soma_cells_2d # Assuming refine_seeds_pca_2d might be here or not used
    from .calculate_features_2d import analyze_segmentation_2d
except ImportError as e:
    expected_ramified_dir = os.path.dirname(os.path.abspath(__file__))
    expected_utils_dir = os.path.dirname(expected_ramified_dir)
    print(f"Error importing 2D segmentation functions in _2D_ramified_strategy.py: {e}")
    print(f"Ensure initial_2d_segmentation.py (with segment_cells_first_pass_raw_2d, apply_hull_trimming_2d) exists.")
    print(f"Ensure ramified_segmenter_2d.py (placeholder) exists in: {expected_ramified_dir}")
    print(f"Ensure calculate_features.py exists in: {expected_utils_dir}") # This should be calculate_features_2d.py
    raise


class Ramified2DStrategy(ProcessingStrategy):
    """
    Processing strategy for ramified microglia in 2D.
    4-step workflow: Raw Seg, Trim Edges, Refine ROIs, Features.
    Uses NPY for intermediate segmentations.
    """

    def _get_mode_name(self) -> str:
        return "ramified_2d"

    def get_step_names(self) -> List[str]:
        """Returns the ordered list of LOGICAL method names for 2D processing steps."""
        return [
            "execute_raw_segmentation",     # Corresponds to execute_raw_segmentation_2d
            "execute_trim_edges",           # Corresponds to execute_trim_edges_2d
            "execute_refine_rois",          # Corresponds to execute_refine_rois_2d
            "execute_calculate_features"    # Corresponds to execute_calculate_features_2d
        ]

    def get_checkpoint_files(self) -> Dict[str, str]:
        """Defines checkpoint file paths specific to the Ramified 2D strategy."""
        files = super().get_checkpoint_files()
        mode_prefix = self.mode_name
        files["raw_segmentation"] = os.path.join(self.processed_dir, f"raw_segmentation_{mode_prefix}.npy")
        files["edge_mask"] = os.path.join(self.processed_dir, f"{mode_prefix}_edge_mask.npy")
        files["trimmed_segmentation"] = os.path.join(self.processed_dir, f"trimmed_segmentation_{mode_prefix}.npy")
        files["cell_bodies"] = os.path.join(self.processed_dir, f"{mode_prefix}_cell_bodies.npy")
        files["refined_rois"] = os.path.join(self.processed_dir, f"{mode_prefix}_refined_rois.npy") # Keep if used
        files["final_segmentation"] = os.path.join(self.processed_dir, f"final_segmentation_{mode_prefix}.npy")
        files["skeleton_array"] = os.path.join(self.processed_dir, f"skeleton_array_{mode_prefix}.npy")
        files["distances_matrix"]=os.path.join(self.processed_dir, f"distances_matrix_{mode_prefix}.csv")
        files["points_matrix"]=os.path.join(self.processed_dir, f"points_matrix_{mode_prefix}.csv")
        files["branch_data"]=os.path.join(self.processed_dir, f"branch_data_{mode_prefix}.csv")
        # Ensure base keys are updated
        files["config"] = os.path.join(self.processed_dir, f"processing_config_{mode_prefix}.yaml")
        files["metrics_df"] = os.path.join(self.processed_dir, f"metrics_df_{mode_prefix}.csv")
        return files

    # --- Step 1: Raw Segmentation (2D) ---
    def execute_raw_segmentation_2d(self, viewer, image_stack: Any, params: Dict) -> bool:
        if image_stack is None or image_stack.ndim != 2:
            print(f"Error: Input image is required and must be 2D for {self.mode_name} mode. Got shape {image_stack.shape if image_stack is not None else 'None'}.")
            return False
        image_2d = image_stack

        print(f"Executing Step 1: Raw {self.mode_name} segmentation...")
        files = self.get_checkpoint_files()
        persistent_raw_npy_path = files.get("raw_segmentation")

        if not persistent_raw_npy_path:
            print("Error: Path 'raw_segmentation' (.npy) missing in checkpoint files.")
            return False
        if not persistent_raw_npy_path.endswith('.npy'):
            print(f"Warning: Expected .npy path for raw_segmentation, got {persistent_raw_npy_path}. Adjusting.")
            persistent_raw_npy_path = os.path.splitext(persistent_raw_npy_path)[0] + ".npy"

        temp_raw_labels_dat_path = None
        temp_raw_labels_dir = None
        segmentation_threshold = 0.0
        first_pass_params = {}
        raw_labels_memmap_handle = None

        try:
            tubular_scales_list = params.get("tubular_scales", [0.8, 1.0, 1.5, 2.0])
            min_size_pixels = int(params.get("min_size", 100)) # Default for 2D, might be 'min_size_pixels' in config
            smooth_sigma = float(params.get("smooth_sigma", 1.0))
            low_threshold_percentile = float(params.get("low_threshold_percentile", 98.0))
            high_threshold_percentile = float(params.get("high_threshold_percentile", 100.0))
            connect_max_gap_physical = float(params.get("connect_max_gap_physical", 1.0))

            # CRITICAL: Verify how segment_cells_first_pass_raw_2d handles spacing.
            # If it expects (dy, dx), then pass tuple(self.spacing[1:]).
            # If it expects [Z, Y, X] and internally uses Y, X, then self.spacing is fine.
            # For this example, let's assume it handles the full ZYX spacing and picks YX.
            print(f"  Passing spacing {self.spacing} to segment_cells_first_pass_raw_2d.")
            temp_raw_labels_dat_path, temp_raw_labels_dir, segmentation_threshold, first_pass_params = \
                segment_cells_first_pass_raw_2d(
                    image=image_2d, spacing=self.spacing,
                    tubular_scales=tubular_scales_list, smooth_sigma=smooth_sigma,
                    connect_max_gap_physical=connect_max_gap_physical, min_size_pixels=min_size_pixels,
                    low_threshold_percentile=low_threshold_percentile,
                    high_threshold_percentile=high_threshold_percentile
                )

            if temp_raw_labels_dat_path is None or not os.path.exists(temp_raw_labels_dat_path):
                print("Error: Raw 2D segmentation function failed or did not produce output temp path.")
                if temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir): rmtree(temp_raw_labels_dir, ignore_errors=True)
                return False
            else:
                self.intermediate_state['segmentation_threshold'] = segmentation_threshold
                self.intermediate_state['original_volume_ref'] = image_2d
                print(f"  Raw 2D function success. Stored intermediate seg_threshold: {segmentation_threshold:.4f}")

            try:
                print(f"  Attempting to load raw 2D labels from temp DAT {temp_raw_labels_dat_path}...")
                if not hasattr(self, 'image_shape') or len(self.image_shape) != 2:
                    raise ValueError(f"Invalid 2D 'image_shape' for strategy. Expected 2 dims, got {self.image_shape if hasattr(self, 'image_shape') else 'None'}.")
                raw_labels_memmap_handle = np.memmap(temp_raw_labels_dat_path, dtype=np.int32, mode='r', shape=self.image_shape)
                print(f"  LOADED temp DAT successfully.")

                print(f"  Attempting to save to persistent NPY: {persistent_raw_npy_path}")
                np.save(persistent_raw_npy_path, raw_labels_memmap_handle)
                print(f"  SAVED persistent NPY successfully: {os.path.basename(persistent_raw_npy_path)}")

                print(f"  Attempting cleanup of temp DAT resources...")
                if hasattr(raw_labels_memmap_handle, '_mmap') and raw_labels_memmap_handle._mmap is not None:
                    raw_labels_memmap_handle._mmap.close(); print(f"    Closed handle to temp DAT.")
                del raw_labels_memmap_handle; gc.collect(); raw_labels_memmap_handle = None
                if temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir):
                    rmtree(temp_raw_labels_dir, ignore_errors=True); print(f"    Removed temp dir: {temp_raw_labels_dir}")
                temp_raw_labels_dir = None; temp_raw_labels_dat_path = None

                if viewer is not None:
                    print(f"  Attempting to load persistent 2D NPY for display: {persistent_raw_npy_path}...")
                    if not os.path.exists(persistent_raw_npy_path):
                        print("    Error: Persistent NPY missing before display attempt with viewer.")
                    else:
                        raw_labels_npy_for_display = np.load(persistent_raw_npy_path)
                        self._add_layer_safely(viewer, raw_labels_npy_for_display, "Raw Intermediate Segmentation")
                        del raw_labels_npy_for_display; gc.collect()
                        print(f"    Added layer 'Raw Intermediate Segmentation' to viewer.")
                else:
                    print("  Viewer is None, skipping display of raw 2D segmentation.")
                print("  Raw 2D segmentation saved (.npy).")
                return True

            except Exception as e_save_disp:
                 print(f"ERROR inside 2D save/display block: {e_save_disp}")
                 traceback.print_exc()
                 if 'raw_labels_memmap_handle' in locals() and raw_labels_memmap_handle is not None and hasattr(raw_labels_memmap_handle, '_mmap') and raw_labels_memmap_handle._mmap is not None:
                     print(f"  Closing temp DAT handle after error in 2D save/display block.")
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
             if 'temp_raw_labels_dir' in locals() and temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir):
                 print(f"Final cleanup check: Removing temp dir {temp_raw_labels_dir}")
                 rmtree(temp_raw_labels_dir, ignore_errors=True)
             if 'raw_labels_memmap_handle' in locals() and raw_labels_memmap_handle is not None and hasattr(raw_labels_memmap_handle, '_mmap'):
                 print(f"  Closing temp DAT handle in outer finally block (raw_seg_2d).")
                 try: raw_labels_memmap_handle._mmap.close()
                 except Exception as e_close: print(f"    Warn: Error closing handle in outer finally: {e_close}")
                 del raw_labels_memmap_handle; gc.collect()


    def execute_trim_edges_2d(self, viewer, image_stack: Any, params: Dict) -> bool:
        print(f"Executing Step 2: Edge Trimming ({self.mode_name})...")
        files = self.get_checkpoint_files()
        raw_seg_input_path = files.get("raw_segmentation")
        trimmed_seg_output_path = files.get("trimmed_segmentation")
        edge_mask_output_path = files.get("edge_mask")

        if not raw_seg_input_path or not os.path.exists(raw_seg_input_path): print(f"Error: Raw 2D NPY missing: {raw_seg_input_path}."); return False
        if not trimmed_seg_output_path or not edge_mask_output_path: print("Error: Output paths missing for 2D trim step."); return False
        if 'segmentation_threshold' not in self.intermediate_state: print("Error: Seg threshold missing."); return False
        if 'original_volume_ref' not in self.intermediate_state: print("Error: Original 2D image missing."); return False
        original_image = self.intermediate_state['original_volume_ref']
        if original_image is None or original_image.ndim != 2: print(f"Error: Original image ref is not 2D (shape {original_image.shape if original_image is not None else 'None'})."); return False
        segmentation_threshold = self.intermediate_state['segmentation_threshold']

        hull_boundary_mask = None
        temp_trimmed_dat_path, temp_trimmed_dir, temp_writable_dir = None, None, None
        trimmed_labels_memmap = None

        try:
            edge_trim_distance_threshold=float(params.get("edge_trim_distance_threshold",2.5))
            hull_boundary_thickness_phys=float(params.get("hull_boundary_thickness_phys",1.0))
            brightness_cutoff_factor=float(params.get("brightness_cutoff_factor",1.5))
            min_size_pixels=int(params.get("min_size_pixels",20))
            smoothing_iterations_param=int(params.get("smoothing_iterations",1))
            heal_iterations_param=int(params.get("heal_iterations",1))

            print(f"  Loading raw 2D NPY {raw_seg_input_path} into temp writable DAT...")
            if not hasattr(self, 'image_shape') or len(self.image_shape) != 2: raise ValueError("Invalid 2D image_shape for strategy.")
            raw_labels_npy = np.load(raw_seg_input_path); temp_writable_dir = tempfile.mkdtemp(prefix="trim_input_2d_")
            temp_writable_dat_path = os.path.join(temp_writable_dir, 'writable_raw_2d.dat')
            writable_raw_memmap = np.memmap(temp_writable_dat_path, dtype=np.int32, mode='w+', shape=self.image_shape)
            writable_raw_memmap[:] = raw_labels_npy[:]; writable_raw_memmap.flush(); del raw_labels_npy; gc.collect()
            print(f"  Created temporary writable 2D DAT: {temp_writable_dat_path}")
            del writable_raw_memmap; gc.collect()

            # CRITICAL: Verify how apply_hull_trimming_2d handles spacing.
            # If it expects (dy, dx), then pass tuple(self.spacing[1:]).
            print(f"  Passing spacing {self.spacing} to apply_hull_trimming_2d.")
            temp_trimmed_dat_path, temp_trimmed_dir, hull_boundary_mask = apply_hull_trimming_2d(
                raw_labels_path=temp_writable_dat_path,
                original_image=original_image, spacing=self.spacing, # Verify this function's spacing expectation
                hull_boundary_thickness_phys=hull_boundary_thickness_phys, edge_trim_distance_threshold=edge_trim_distance_threshold,
                brightness_cutoff_factor=brightness_cutoff_factor, segmentation_threshold=segmentation_threshold,
                min_size_pixels=min_size_pixels, smoothing_iterations=smoothing_iterations_param, heal_iterations=heal_iterations_param)

            if temp_writable_dir and os.path.exists(temp_writable_dir): rmtree(temp_writable_dir, ignore_errors=True); temp_writable_dir = None

            if temp_trimmed_dat_path is None or not os.path.exists(temp_trimmed_dat_path) or hull_boundary_mask is None:
                 print("Error: apply_hull_trimming_2d failed or did not produce output.");
                 if temp_trimmed_dir and os.path.exists(temp_trimmed_dir): rmtree(temp_trimmed_dir, ignore_errors=True); return False
            else: print(f"  apply_hull_trimming_2d created temp trimmed DAT: {temp_trimmed_dat_path}")

            try:
                print(f"  Loading temp trimmed 2D DAT {temp_trimmed_dat_path} to save persistent NPYs...")
                trimmed_labels_memmap = np.memmap(temp_trimmed_dat_path, dtype=np.int32, mode='r', shape=self.image_shape)
                np.save(trimmed_seg_output_path, trimmed_labels_memmap)
                print(f"    SAVED trimmed 2D NPY: {os.path.basename(trimmed_seg_output_path)}")
                np.save(edge_mask_output_path, hull_boundary_mask)
                print(f"    SAVED 2D edge mask NPY: {os.path.basename(edge_mask_output_path)}")
            except Exception as e_save: print(f"ERROR saving persistent 2D NPY files: {e_save}"); traceback.print_exc(); return False
            finally:
                print(f"  Attempting cleanup of temp trimmed 2D DAT resources...")
                if trimmed_labels_memmap is not None and hasattr(trimmed_labels_memmap, '_mmap') and trimmed_labels_memmap._mmap is not None:
                    try: trimmed_labels_memmap._mmap.close(); print(f"    Closed handle to temp trimmed DAT.")
                    except: print("    Warn: Error closing temp trimmed DAT handle.")
                if 'trimmed_labels_memmap' in locals() and trimmed_labels_memmap is not None: del trimmed_labels_memmap; gc.collect()
                if temp_trimmed_dir and os.path.exists(temp_trimmed_dir):
                    try: rmtree(temp_trimmed_dir, ignore_errors=True); print(f"    Removed temp trimmed dir: {temp_trimmed_dir}")
                    except Exception as e_rm: print(f"    Warn: Error removing temp dir {temp_trimmed_dir}: {e_rm}")
                temp_trimmed_dir = None; temp_trimmed_dat_path = None

            if viewer is not None:
                print(f"  Attempting to display trimmed 2D NPY and edge mask in viewer...")
                try:
                    if not os.path.exists(trimmed_seg_output_path): raise FileNotFoundError("Persistent trimmed NPY missing for display.")
                    trimmed_labels_npy = np.load(trimmed_seg_output_path)
                    self._add_layer_safely(viewer, trimmed_labels_npy, "Trimmed Intermediate Segmentation")
                    del trimmed_labels_npy; gc.collect()

                    if not os.path.exists(edge_mask_output_path): raise FileNotFoundError("Persistent edge mask NPY missing for display.")
                    edge_mask_npy = np.load(edge_mask_output_path)
                    self._add_layer_safely(viewer, edge_mask_npy, "Edge Mask", layer_type='image', colormap='gray', blending='additive')
                    del edge_mask_npy; gc.collect()
                    print("  Trimmed 2D segmentation and mask displayed in viewer.")
                except Exception as e_viz_trim:
                    print(f"    WARNING during 2D trim visualization block: {e_viz_trim}")
            else:
                print("  Viewer is None, skipping display of trimmed 2D segmentation and edge mask.")
            print("  Trimmed 2D segmentation (.npy) and mask (.npy) saved.")
            return True

        except Exception as e: print(f"Error during 2D trim_edges step setup or execution: {e}"); traceback.print_exc(); return False
        finally:
             if 'temp_writable_dir' in locals() and temp_writable_dir and os.path.exists(temp_writable_dir): print(f"Final cleanup check: Removing temp writable dir {temp_writable_dir}"); rmtree(temp_writable_dir, ignore_errors=True)
             if 'temp_trimmed_dir' in locals() and temp_trimmed_dir and os.path.exists(temp_trimmed_dir): print(f"Final cleanup check: Removing trimmed labels temp dir {temp_trimmed_dir}"); rmtree(temp_trimmed_dir, ignore_errors=True)


    def execute_refine_rois_2d(self, viewer, image_stack: Any, params: Dict): # image_stack is original 2D image here
        print(f"Refining {self.mode_name} ROIs...")
        files = self.get_checkpoint_files()
        initial_seg_path = files.get("trimmed_segmentation")
        if not os.path.exists(initial_seg_path): print("Error: Trimmed 2D segmentation file not found."); return False

        # original_image is passed as image_stack to this method for step 3 if image_stack is image_2d
        # If image_stack refers to the first input image_stack of the whole pipeline, use intermediate_state
        original_image = self.intermediate_state.get('original_volume_ref')
        if original_image is None or original_image.ndim != 2:
            print(f"Error: Original 2D image ref is missing or not 2D (shape {getattr(original_image, 'shape', 'None')}). Attempting to use provided image_stack.");
            if image_stack is not None and image_stack.ndim == 2:
                original_image = image_stack # Fallback to image_stack if it's 2D
            else:
                print("  Fallback image_stack is also not suitable. Cannot proceed.")
                return False

        if not hasattr(self, 'spacing') or len(self.spacing) != 3: print(f"Error: Strategy spacing attribute invalid or missing."); return False
        spacing_yx = tuple(self.spacing[1:])
        print(f"  Using extracted YX spacing for 2D refinement functions: {spacing_yx}")

        labeled_cells = np.load(initial_seg_path)
        cell_bodies, merged_roi_array = None, None # refined_mask might not be generated/saved explicitly

        try:
            min_fragment_size = int(params.get("min_fragment_size", 20))
            default_percentiles = [100 - i for i in range(0, 99, 5)] + [99, 1]
            ratios_to_process = params.get("ratios_to_process", [0.3, 0.4, 0.5, 0.6])
            intensity_percentiles_to_process = params.get("intensity_percentiles_to_process", default_percentiles)
            smallest_quantile = float(params.get("smallest_quantile", 0.05))
            core_volume_target_factor_lower = float(params.get("core_volume_target_factor_lower", 0.1))
            core_volume_target_factor_upper = float(params.get("core_volume_target_factor_upper", 10.0))
            erosion_iterations = int(params.get("erosion_iterations", 0))
            min_physical_peak_separation = float(params.get("min_physical_peak_separation", 3.0))
            max_allowed_core_aspect_ratio = float(params.get("max_allowed_core_aspect_ratio", 5.0))
            ref_vol_percentile_lower = int(params.get("ref_vol_percentile_lower", 30))
            ref_vol_percentile_upper = int(params.get("ref_vol_percentile_upper", 70))
            ref_thickness_percentile_lower = int(params.get("ref_thickness_percentile_lower", 1))
            absolute_min_thickness_um = float(params.get("absolute_min_thickness_um", 0.5))
            absolute_max_thickness_um = float(params.get("absolute_max_thickness_um", 5.0))
            min_size_threshold = int(params.get("min_size_threshold", 100)) # Area in pixels for 2D
            max_seed_centroid_dist = float(params.get("max_seed_centroid_dist", 20.0))
            min_path_intensity_ratio = float(params.get("min_path_intensity_ratio", 0.8))
            min_local_intensity_difference = float(params.get("min_local_intensity_difference", 0.05))
            local_analysis_radius = float(params.get("local_analysis_radius", 2.0))
            max_hole_size = int(params.get("max_hole_size", 0)) # 0 means not filling any holes

            print("  Calling extract_soma_masks_2d...")
            cell_bodies = extract_soma_masks_2d(
                segmentation_mask=labeled_cells, intensity_image=original_image, spacing=spacing_yx,
                smallest_quantile=smallest_quantile, min_fragment_size=min_fragment_size,
                core_volume_target_factor_lower=core_volume_target_factor_lower,
                core_volume_target_factor_upper=core_volume_target_factor_upper,
                erosion_iterations=erosion_iterations, ratios_to_process=ratios_to_process,
                intensity_percentiles_to_process=intensity_percentiles_to_process,
                min_physical_peak_separation=min_physical_peak_separation,
                max_allowed_core_aspect_ratio=max_allowed_core_aspect_ratio,
                ref_vol_percentile_lower=ref_vol_percentile_lower,
                ref_vol_percentile_upper=ref_vol_percentile_upper,
                ref_thickness_percentile_lower=ref_thickness_percentile_lower,
                absolute_min_thickness_um=absolute_min_thickness_um,
                absolute_max_thickness_um=absolute_max_thickness_um
            )
            # Assuming refine_seeds_pca_2d is not part of this simplified workflow or is integrated
            # refined_mask = refine_seeds_pca_2d(...) # If used, save it to files.get("refined_rois")

            print("  Calling separate_multi_soma_cells_2d...")
            merged_roi_array = separate_multi_soma_cells_2d(
                segmentation_mask=labeled_cells, intensity_volume=original_image, soma_mask=cell_bodies,
                spacing=spacing_yx, min_size_threshold=min_size_threshold,
                max_seed_centroid_dist=max_seed_centroid_dist,
                min_path_intensity_ratio=min_path_intensity_ratio,
                min_local_intensity_difference=min_local_intensity_difference,
                local_analysis_radius=local_analysis_radius,
                max_hole_size=max_hole_size
            )
        except NotImplementedError:
            print("ERROR: 2D Refinement functions (extract_soma_masks_2d, etc.) not implemented yet.")
            if viewer is not None:
                msg = QMessageBox(); msg.setIcon(QMessageBox.Warning); msg.setText("Step Not Implemented"); msg.setInformativeText("The functions for 2D ROI refinement haven't been created yet."); msg.exec_();
            return False
        except Exception as e:
             print(f"Error during 2D ROI refinement: {e}")
             print(traceback.format_exc())
             return False

        cell_bodies_path = files.get("cell_bodies")
        # refined_rois_path = files.get("refined_rois") # If you save a refined_mask separately
        final_seg_path = files.get("final_segmentation")
        try:
             if cell_bodies_path and cell_bodies is not None: np.save(cell_bodies_path, cell_bodies)
             # if refined_rois_path and refined_mask is not None: np.save(refined_rois_path, refined_mask)
             if final_seg_path and merged_roi_array is not None: np.save(final_seg_path, merged_roi_array)
             print(f"Saved 2D cell bodies and final segmentation.")

             if viewer is not None:
                 print("  Attempting to display 2D refined ROI layers in viewer...")
                 try:
                     if cell_bodies is not None: self._add_layer_safely(viewer, cell_bodies, "Cell bodies")
                     # if refined_mask is not None: self._add_layer_safely(viewer, refined_mask, "Refined ROIs")
                     if merged_roi_array is not None: self._add_layer_safely(viewer, merged_roi_array, "Final segmentation")
                     print("  Added 2D refined ROI layers to viewer.")
                 except Exception as e_disp_refine:
                     print(f"    Warning: Failed to display 2D refined ROI layers in viewer: {e_disp_refine}")
             else:
                 print("  Viewer is None, skipping display of 2D refined ROI layers.")
             return True
        except Exception as e: print(f"Error saving or adding layers for 2D refined ROIs: {e}"); return False


    def execute_calculate_features_2d(self, viewer, image_stack: Any, params: Dict):
        print(f"Calculating {self.mode_name} features...")
        files = self.get_checkpoint_files()
        final_seg_path = files.get("final_segmentation")

        if not os.path.exists(final_seg_path): print(f"Error: Final 2D segmentation file not found: {final_seg_path}."); return False

        if not hasattr(self, 'spacing') or len(self.spacing) != 3: print(f"Error: Strategy spacing attribute invalid or missing."); return False
        spacing_yx = tuple(self.spacing[1:])
        print(f"  Using extracted YX spacing for 2D feature calculation: {spacing_yx}")

        merged_roi_array = None
        metrics_df, detailed_outputs, skeleton_array_result = None, {}, None

        try:
            print(f"Loading final 2D segmentation with np.load: {final_seg_path}")
            merged_roi_array = np.load(final_seg_path)
            if not hasattr(self, 'image_shape') or merged_roi_array.shape != self.image_shape:
                 print(f"Warning: Shape mismatch for 2D final_seg. Expected {self.image_shape}, got {merged_roi_array.shape}")
        except Exception as e: print(f"Error loading final 2D segmentation {final_seg_path}: {e}"); return False

        try:
            prune_spurs_le_um = params.get("prune_spurs_le_um", 0)

            print("Running analyze_segmentation_2d...")
            metrics_df, detailed_outputs = analyze_segmentation_2d(
                segmented_array=merged_roi_array,
                spacing_yx=spacing_yx,
                calculate_distances=params.get("calculate_distances", True),
                calculate_skeletons=params.get("calculate_skeletons", True), # Ensure analyze_segmentation_2d supports this
                skeleton_export_path=None, # Or specific path if needed by backend
                return_detailed=True,
                prune_spurs_le_um=prune_spurs_le_um
            )
            print("analyze_segmentation_2d finished.")
            skeleton_array_result = detailed_outputs.get('skeleton_array') # Capture for display/save
        except Exception as e: print(f"Error during 2D feature calculation (analyze_segmentation_2d): {e}"); traceback.print_exc(); return False

        print("Saving calculated 2D features...")
        try:
            if metrics_df is not None and not metrics_df.empty: metrics_df.to_csv(files["metrics_df"], index=False); print(f"  Saved metrics to {files['metrics_df']}")
            if detailed_outputs:
                dist_matrix = detailed_outputs.get('distance_matrix')
                if dist_matrix is not None and not dist_matrix.empty: dist_matrix.to_csv(files["distances_matrix"]); print(f"  Saved distance matrix to {files['distances_matrix']}")
                all_points = detailed_outputs.get('all_pairs_points')
                if all_points is not None and not all_points.empty: all_points.to_csv(files["points_matrix"], index=False); print(f"  Saved points matrix to {files['points_matrix']}")
                branch_data = detailed_outputs.get('detailed_branches')
                if branch_data is not None and not branch_data.empty: branch_data.to_csv(files["branch_data"], index=False); print(f"  Saved branch data to {files['branch_data']}")
                if skeleton_array_result is not None: np.save(files["skeleton_array"], skeleton_array_result); print(f"  Saved skeleton array to {files['skeleton_array']}")
        except Exception as e: print(f"Error saving 2D feature results: {e}"); traceback.print_exc()

        if viewer is not None:
            print("  Preparing 2D visualizations for viewer...")
            try:
                if skeleton_array_result is not None:
                     self._add_layer_safely(viewer, skeleton_array_result, "Skeletons", layer_type='labels')
                if metrics_df is not None and not metrics_df.empty and 'shortest_distance_um' in metrics_df.columns:
                    connections_df = metrics_df[metrics_df['shortest_distance_um'].notna() & np.isfinite(metrics_df['shortest_distance_um']) & metrics_df['closest_neighbor_label'].notna()].copy()
                    if not connections_df.empty:
                        coord_cols_self = ['point_on_self_y', 'point_on_self_x']
                        coord_cols_neigh = ['point_on_neighbor_y', 'point_on_neighbor_x']
                        if all(c in connections_df for c in coord_cols_self) and all(c in connections_df for c in coord_cols_neigh):
                            points1 = connections_df[coord_cols_self].values; points2 = connections_df[coord_cols_neigh].values
                            lines_to_draw = np.stack((points1, points2), axis=1)
                            lines_to_draw = lines_to_draw[~np.isnan(lines_to_draw).any(axis=(1,2))]
                            if lines_to_draw.shape[0] > 0:
                                self._add_layer_safely(viewer, lines_to_draw, "Closest Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)
                                print(f"    Visualizing {lines_to_draw.shape[0]} closest 2D connections.")
                        else: print(f"    Warning: Missing YX coordinate columns in metrics_df for 2D connection viz: needed {coord_cols_self}, {coord_cols_neigh}")
            except Exception as e_viz_feat: print(f"    Warning: Error during 2D feature visualization: {e_viz_feat}")
        else:
            print("  Viewer is None, skipping display of 2D feature visualizations.")

        if merged_roi_array is not None: del merged_roi_array; gc.collect(); print("  Cleaned up loaded 2D segmentation array.")
        print("2D Feature calculation step complete.")
        return True


    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        if viewer is None:
            # print(f"BATCH MODE ({self.mode_name}): Viewer is None in load_checkpoint_data. Skipping display.")
            return

        files = self.get_checkpoint_files()
        print(f"Loading {self.mode_name} checkpoint data artifacts up to step {checkpoint_step} for viewer...")
        layer_base_names_to_manage = [
            "Raw Intermediate Segmentation", "Trimmed Intermediate Segmentation", "Edge Mask",
            "Cell bodies", "Refined ROIs", "Final segmentation",
            "Skeletons", "Closest Connections"
        ]
        # print("  Pre-removing potentially existing layers...") # Less verbose
        for base_name in layer_base_names_to_manage: self._remove_layer_safely(viewer, base_name)
        gc.collect()

        if checkpoint_step >= 1:
            path = files.get("raw_segmentation")
            if path and os.path.exists(path):
                 try: data = np.load(path); self._add_layer_safely(viewer, data, "Raw Intermediate Segmentation"); print(f"    Added: {os.path.basename(path)}")
                 except Exception as e: print(f"    Error loading raw 2D NPY {path}: {e}")
        if checkpoint_step >= 2:
            path = files.get("trimmed_segmentation")
            if path and os.path.exists(path):
                 try: data = np.load(path); self._add_layer_safely(viewer, data, "Trimmed Intermediate Segmentation"); print(f"    Added: {os.path.basename(path)}")
                 except Exception as e: print(f"    Error loading trimmed 2D NPY {path}: {e}")
            path = files.get("edge_mask")
            if path and os.path.exists(path):
                 try: data = np.load(path); self._add_layer_safely(viewer, data, "Edge Mask", layer_type='image', colormap='gray', blending='additive'); print(f"    Added: {os.path.basename(path)}")
                 except Exception as e: print(f"    Error loading edge mask 2D NPY {path}: {e}")
        if checkpoint_step >= 3:
            path = files.get("final_segmentation")
            if path and os.path.exists(path):
                 try: data = np.load(path); self._add_layer_safely(viewer, data, "Final segmentation"); print(f"    Added: {os.path.basename(path)}")
                 except Exception as e: print(f"    Error loading final 2D NPY {path}: {e}")
            path = files.get("cell_bodies")
            if path and os.path.exists(path):
                 try: data = np.load(path); self._add_layer_safely(viewer, data, "Cell bodies"); print(f"    Added: {os.path.basename(path)}")
                 except Exception as e: print(f"    Error loading cell bodies 2D NPY {path}: {e}")
            path = files.get("refined_rois") # If you have this output
            if path and os.path.exists(path):
                 try: data = np.load(path); self._add_layer_safely(viewer, data, "Refined ROIs"); print(f"    Added: {os.path.basename(path)}")
                 except Exception as e: print(f"    Error loading refined ROIs 2D NPY {path}: {e}")
        if checkpoint_step >= 4:
             metrics_path = files.get("metrics_df"); metrics_df = None
             if metrics_path and os.path.exists(metrics_path):
                  try: metrics_df = pd.read_csv(metrics_path); print(f"    Loaded 2D Metrics CSV ({len(metrics_df)} rows)")
                  except Exception as e: print(f"    Error loading 2D metrics CSV {metrics_path}: {e}")
             path = files.get("skeleton_array")
             if path and os.path.exists(path):
                  try: data = np.load(path); self._add_layer_safely(viewer, data, "Skeletons", layer_type='labels'); print(f"    Added: {os.path.basename(path)}")
                  except Exception as e: print(f"    Error loading 2D skeletons NPY {path}: {e}")
             if metrics_df is not None and not metrics_df.empty and 'shortest_distance_um' in metrics_df.columns:
                connections_df = metrics_df[metrics_df['shortest_distance_um'].notna() & np.isfinite(metrics_df['shortest_distance_um']) & metrics_df['closest_neighbor_label'].notna()].copy()
                if not connections_df.empty:
                    coord_cols_self = ['point_on_self_y', 'point_on_self_x']; coord_cols_neigh = ['point_on_neighbor_y', 'point_on_neighbor_x']
                    if all(c in connections_df for c in coord_cols_self) and all(c in connections_df for c in coord_cols_neigh):
                        try:
                            points1 = connections_df[coord_cols_self].values; points2 = connections_df[coord_cols_neigh].values
                            lines_to_draw = np.stack((points1, points2), axis=1)
                            lines_to_draw = lines_to_draw[~np.isnan(lines_to_draw).any(axis=(1,2))]
                            if lines_to_draw.shape[0] > 0:
                                self._add_layer_safely(viewer, lines_to_draw, "Closest Connections", layer_type='shapes', shape_type='line', edge_color='red', edge_width=1)
                                print(f"    Visualizing {lines_to_draw.shape[0]} closest 2D connections.")
                        except Exception as e_lines: print(f"    ERROR loading 2D connection lines: {e_lines}")
                    else: print(f"    Warning: Missing YX coordinate columns for loading 2D connections from {metrics_path}.")


    def cleanup_step_artifacts(self, viewer, step_number: int):
        """Cleans 2D artifacts. Viewer can be None."""
        files = self.get_checkpoint_files()
        # print(f"Cleaning {self.mode_name} artifacts for step {step_number} (Viewer: {'Present' if viewer else 'None'})...") # Less verbose

        if step_number == 1:
            if viewer is not None: self._remove_layer_safely(viewer, "Raw Intermediate Segmentation")
            self._remove_file_safely(files.get("raw_segmentation"))
        elif step_number == 2:
            if viewer is not None:
                self._remove_layer_safely(viewer, "Trimmed Intermediate Segmentation")
                self._remove_layer_safely(viewer, "Edge Mask")
            self._remove_file_safely(files.get("trimmed_segmentation"))
            self._remove_file_safely(files.get("edge_mask"))
        elif step_number == 3:
            if viewer is not None:
                 self._remove_layer_safely(viewer, "Final segmentation")
                 self._remove_layer_safely(viewer, "Cell bodies")
                 self._remove_layer_safely(viewer, "Refined ROIs") # If you have this layer
            self._remove_file_safely(files.get("final_segmentation"))
            self._remove_file_safely(files.get("cell_bodies"))
            self._remove_file_safely(files.get("refined_rois")) # If you have this file
        elif step_number == 4:
            if viewer is not None:
                self._remove_layer_safely(viewer, "Closest Connections")
                self._remove_layer_safely(viewer, "Skeletons")
            self._remove_file_safely(files.get("metrics_df"))
            self._remove_file_safely(files.get("distances_matrix"))
            self._remove_file_safely(files.get("points_matrix"))
            self._remove_file_safely(files.get("branch_data"))
            self._remove_file_safely(files.get("skeleton_array"))
        # print(f"  Cleaned step {step_number} ({self.mode_name}) artifacts.") # Less verbose

# --- END OF FILE utils/ramified_module_2d/_2D_ramified_strategy.py ---