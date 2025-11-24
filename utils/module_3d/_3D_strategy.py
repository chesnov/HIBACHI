# --- START OF FILE utils/ramified_module_3d/_3D_ramified_strategy.py ---
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sys
import traceback
import gc
from shutil import rmtree, copyfile
import time
import tempfile

# Correct relative imports based on structure
from ..high_level_gui.processing_strategies import ProcessingStrategy
try:
    from .initial_3d_segmentation import segment_cells_first_pass_raw
    from .remove_artifacts import apply_hull_trimming
    from .ramified_segmenter import extract_soma_masks, separate_multi_soma_cells
    from .calculate_features_3d import analyze_segmentation
except ImportError as e:
    expected_ramified_dir = os.path.dirname(os.path.abspath(__file__))
    expected_utils_dir = os.path.dirname(expected_ramified_dir)
    print(f"Error importing segmentation functions in _3D_ramified_strategy.py: {e}")
    print(f"Ensure initial_3d_segmentation.py (with segment_cells_first_pass_raw), "
          f"ramified_segmenter.py are in: {expected_ramified_dir}")
    print(f"Ensure calculate_features.py is in: {expected_utils_dir}")
    raise


class RamifiedStrategy(ProcessingStrategy):
    """
    Processing strategy for ramified microglia.
    4-step workflow: Raw Seg, Trim Edges, Refine ROIs, Features.
    Uses DAT files for memory-mapped intermediate segmentations.
    """

    def _get_mode_name(self) -> str:
        return "ramified"

    def get_step_names(self) -> List[str]:
        """Returns the ordered list of method names for processing steps."""
        return [
            "execute_raw_segmentation",     # Step 1
            "execute_trim_edges",           # Step 2
            "execute_refine_rois",          # Step 3
            "execute_calculate_features"    # Step 4
        ]

    def get_checkpoint_files(self) -> Dict[str, str]:
        """Defines checkpoint file paths specific to the Ramified strategy."""
        files = super().get_checkpoint_files()
        mode_prefix = self.mode_name
        files["raw_segmentation"] = os.path.join(self.processed_dir, f"raw_segmentation_{mode_prefix}.dat")
        files["edge_mask"] = os.path.join(self.processed_dir, f"{mode_prefix}_edge_mask.dat")
        files["trimmed_segmentation"] = os.path.join(self.processed_dir, f"trimmed_segmentation_{mode_prefix}.dat")
        files["cell_bodies"] = os.path.join(self.processed_dir, f"cell_bodies.dat")
        files["final_segmentation"] = os.path.join(self.processed_dir, f"final_segmentation_{mode_prefix}.dat")
        files["skeleton_array"] = os.path.join(self.processed_dir, f"skeleton_array_{mode_prefix}.dat")
        files["distances_matrix"]=os.path.join(self.processed_dir, f"distances_matrix_{mode_prefix}.csv")
        files["points_matrix"]=os.path.join(self.processed_dir, f"points_matrix_{mode_prefix}.csv")
        files["branch_data"]=os.path.join(self.processed_dir, f"branch_data_{mode_prefix}.csv")
        return files

    def execute_raw_segmentation(self, viewer, image_stack: Any, params: Dict) -> bool:
        if image_stack is None: return False
        print(f"Executing Step 1: Raw {self.mode_name} segmentation...")
        files = self.get_checkpoint_files()
        persistent_raw_dat_path = files.get("raw_segmentation")

        temp_raw_labels_dat_path = temp_raw_labels_dir = None
        try:
            tubular_scales_list = params.get("tubular_scales", [0.8, 1.0, 1.5, 2.0])
            skip_tubular_enhancement = len(tubular_scales_list) == 1 and tubular_scales_list[0] == 0.0
            
            temp_raw_labels_dat_path, temp_raw_labels_dir, segmentation_threshold, _ = \
                segment_cells_first_pass_raw(
                    volume=image_stack, spacing=self.spacing,
                    tubular_scales=tubular_scales_list, 
                    smooth_sigma=float(params.get("smooth_sigma", 1.3)),
                    connect_max_gap_physical=float(params.get("connect_max_gap_physical", 1.0)), 
                    min_size_voxels=int(params.get("min_size", 2000)),
                    low_threshold_percentile=float(params.get("low_threshold_percentile", 25.0)),
                    high_threshold_percentile=float(params.get("high_threshold_percentile", 95.0)),
                    skip_tubular_enhancement=skip_tubular_enhancement
                )

            if not temp_raw_labels_dat_path or not os.path.exists(temp_raw_labels_dat_path):
                raise RuntimeError("Raw segmentation function failed to produce an output file.")

            self.intermediate_state['segmentation_threshold'] = segmentation_threshold
            self.intermediate_state['original_volume_ref'] = image_stack

            print(f"  Copying temporary result to persistent DAT: {persistent_raw_dat_path}")
            copyfile(temp_raw_labels_dat_path, persistent_raw_dat_path)
            print(f"  SAVED persistent DAT successfully.")

            if viewer is not None:
                print(f"  Attempting to load persistent DAT for display...")
                raw_labels_for_display = np.memmap(persistent_raw_dat_path, dtype=np.int32, mode='r', shape=self.image_shape)
                self._add_layer_safely(viewer, raw_labels_for_display, "Raw Intermediate Segmentation")
            
            return True

        except Exception as e:
             print(f"Error during execute_raw_segmentation: {e}"); traceback.print_exc()
             return False
        finally:
             if temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir):
                 rmtree(temp_raw_labels_dir, ignore_errors=True)

    def execute_trim_edges(self, viewer, image_stack: Any, params: Dict) -> bool:
        print(f"Executing Step 2: Edge Trimming...")
        files = self.get_checkpoint_files()
        raw_seg_input_path = files.get("raw_segmentation")
        trimmed_seg_output_path = files.get("trimmed_segmentation")
        edge_mask_output_path = files.get("edge_mask")

        if not raw_seg_input_path or not os.path.exists(raw_seg_input_path): return False
        if 'segmentation_threshold' not in self.intermediate_state: 
            print("Error: segmentation_threshold not found in intermediate state. Please run Step 1 first."); return False
        
        temp_trimmed_dat_path = temp_trimmed_dir = None
        
        try:
            temp_trimmed_dat_path, temp_trimmed_dir, hull_boundary_mask = apply_hull_trimming(
                raw_labels_path=raw_seg_input_path,
                original_volume=self.intermediate_state['original_volume_ref'], 
                spacing=self.spacing,
                segmentation_threshold=self.intermediate_state['segmentation_threshold'],
                hull_boundary_thickness_phys=float(params.get("hull_boundary_thickness_phys", 2.0)),
                edge_trim_distance_threshold=float(params.get("edge_trim_distance_threshold", 4.5)),
                brightness_cutoff_factor=float(params.get("brightness_cutoff_factor", 1.5)),
                min_size_voxels=int(params.get("min_size_voxels", 50)),
                smoothing_iterations=int(params.get("smoothing_iterations", 1)),
                heal_iterations=int(params.get("heal_iterations", 1)),
                edge_distance_chunk_size_z=int(params.get("edge_distance_chunk_size_z", 32))
            )

            if not temp_trimmed_dat_path or not os.path.exists(temp_trimmed_dat_path) or hull_boundary_mask is None:
                raise RuntimeError("apply_hull_trimming failed to produce an output file.")

            print(f"  Copying trimmed result to persistent DAT: {trimmed_seg_output_path}")
            copyfile(temp_trimmed_dat_path, trimmed_seg_output_path)
            
            print(f"  Saving edge mask to persistent DAT: {edge_mask_output_path}")
            edge_mask_memmap = np.memmap(edge_mask_output_path, dtype=bool, mode='w+', shape=self.image_shape)
            edge_mask_memmap[:] = hull_boundary_mask[:]
            edge_mask_memmap.flush()
            del edge_mask_memmap

            if viewer is not None:
                print(f"  Loading persistent DAT files for display...")
                trimmed_labels_for_display = np.memmap(trimmed_seg_output_path, dtype=np.int32, mode='r', shape=self.image_shape)
                self._add_layer_safely(viewer, trimmed_labels_for_display, "Trimmed Intermediate Segmentation")
                
                edge_mask_for_display = np.memmap(edge_mask_output_path, dtype=bool, mode='r', shape=self.image_shape)
                self._add_layer_safely(viewer, edge_mask_for_display, "Edge Mask", layer_type='image', colormap='gray', blending='additive')

            return True

        except Exception as e:
             print(f"Error during trim_edges: {e}"); traceback.print_exc()
             return False
        finally:
             if temp_trimmed_dir and os.path.exists(temp_trimmed_dir):
                 rmtree(temp_trimmed_dir, ignore_errors=True)

    def execute_refine_rois(self, viewer, image_stack, params: Dict):
        print(f"Refining {self.mode_name} ROIs...")
        files = self.get_checkpoint_files()
        trimmed_seg_path = files["trimmed_segmentation"]
        if not os.path.exists(trimmed_seg_path): return False
        
        try:
            # Open inputs as read-only memmaps
            trimmed_labels_memmap = np.memmap(trimmed_seg_path, dtype=np.int32, mode='r', shape=self.image_shape)
            # NOTE: For this to be fully effective, 'image_stack' should also be a memmap object
            # when passed into this function.

            soma_extraction_params = {
                "smallest_quantile": float(params.get("smallest_quantile", 25)),
                "min_fragment_size": int(params.get("min_fragment_size", 30)),
                "core_volume_target_factor_lower": float(params.get("core_volume_target_factor_lower", 0.1)),
                "core_volume_target_factor_upper": float(params.get("core_volume_target_factor_upper", 10.0)),
                "erosion_iterations": int(params.get("erosion_iterations", 0)),
                "ratios_to_process": params.get("ratios_to_process", [0.3, 0.4, 0.5, 0.6]),
                "intensity_percentiles_to_process": params.get("intensity_percentiles_to_process", [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]),
                "min_physical_peak_separation": float(params.get("min_physical_peak_separation", 7.0)),
                "max_allowed_core_aspect_ratio": float(params.get("max_allowed_core_aspect_ratio", 10.0)),
                "ref_vol_percentile_lower": int(params.get("ref_vol_percentile_lower", 30)),
                "ref_vol_percentile_upper": int(params.get("ref_vol_percentile_upper", 70)),
                "ref_thickness_percentile_lower": int(params.get("ref_thickness_percentile_lower", 1)),
                "absolute_min_thickness_um": float(params.get("absolute_min_thickness_um", 1.5)),
                "absolute_max_thickness_um": float(params.get("absolute_max_thickness_um", 10.0)),
                "memmap_final_mask": True # Ensure extract_soma_masks uses a memmap for its output
            }

            separation_params = {
                "min_size_threshold": int(params.get("min_size_threshold", 100)),
                "intensity_weight": float(params.get("intensity_weight", 0.0)),
                "max_seed_centroid_dist": float(params.get("max_seed_centroid_dist", 40.0)),
                "min_path_intensity_ratio": float(params.get("min_path_intensity_ratio", 0.8)),
                "min_local_intensity_difference": float(params.get("min_local_intensity_difference", 0.05)),
                "local_analysis_radius": int(params.get("local_analysis_radius", 10)),
                # Pass memmap control params to the chunked wrapper
                "memmap_dir": os.path.join(self.processed_dir, "sep_multi_soma_temp"),
                "memmap_voxel_threshold": int(params.get("memmap_voxel_threshold", 25_000_000))
            }
            
            # This function now returns EITHER a memmap object OR a numpy array
            cell_bodies = extract_soma_masks(trimmed_labels_memmap, image_stack, self.spacing, **soma_extraction_params)
            
            # Pass the object directly (whether memmap or array).
            final_separated_cells = separate_multi_soma_cells(trimmed_labels_memmap, image_stack, cell_bodies, self.spacing, **separation_params)
            
            cell_bodies_path = files.get("cell_bodies")
            
            # --- [START] ROBUST, CORRECTED SAVING LOGIC ---
            # Check the type to handle saving correctly
            if isinstance(cell_bodies, np.memmap):
                print("Cell bodies is a memmap object. Copying file to persistent location.")
                temp_cb_path = cell_bodies.filename
                # Ensure the object is deleted to release the file lock before copying
                del cell_bodies
                gc.collect()
                copyfile(temp_cb_path, cell_bodies_path)
            else:
                print("Cell bodies is a NumPy array. Saving to persistent memmap location.")
                # Save the in-memory array to the persistent memmap file
                cb_memmap = np.memmap(cell_bodies_path, dtype=cell_bodies.dtype, mode='w+', shape=cell_bodies.shape)
                cb_memmap[:] = cell_bodies[:]
                cb_memmap.flush()
                del cb_memmap # Release the handle to the new persistent file
            # --- [END] ROBUST, CORRECTED SAVING LOGIC ---
            
            final_seg_path = files.get("final_segmentation")
            final_memmap = np.memmap(final_seg_path, dtype=np.int32, mode='w+', shape=self.image_shape)
            final_memmap[:] = final_separated_cells[:]; final_memmap.flush()
            
            if viewer is not None:
                # Open the persistent file for viewing
                cb_display = np.memmap(cell_bodies_path, dtype=np.int32, mode='r', shape=self.image_shape)
                self._add_layer_safely(viewer, cb_display, "Cell bodies")
                self._add_layer_safely(viewer, final_memmap, "Final segmentation")
            
            return True
        except Exception as e:
             print(f"Error during ROI refinement: {e}"); traceback.print_exc()
             return False

    def execute_calculate_features(self, viewer, image_stack, params: Dict):
        print(f"Calculating {self.mode_name} features...")
        files = self.get_checkpoint_files()
        final_seg_path = files["final_segmentation"]
        if not os.path.exists(final_seg_path): return False

        try:
            final_seg_memmap = np.memmap(final_seg_path, dtype=np.int32, mode='r', shape=self.image_shape)

            metrics_df, detailed_outputs = analyze_segmentation(
                segmented_array=final_seg_memmap, spacing=self.spacing,
                **params, return_detailed=True
            )
            
            if metrics_df is not None: metrics_df.to_csv(files["metrics_df"], index=False)
            skeleton_array_result = detailed_outputs.get('skeleton_array')
            if skeleton_array_result is not None: 
                skeleton_path = files.get("skeleton_array")
                skel_memmap = np.memmap(skeleton_path, dtype=skeleton_array_result.dtype, mode='w+', shape=skeleton_array_result.shape)
                skel_memmap[:] = skeleton_array_result[:]; skel_memmap.flush()
            
            if viewer is not None and skeleton_array_result is not None:
                skeleton_path = files.get("skeleton_array")
                skel_display_memmap = np.memmap(skeleton_path, dtype=skeleton_array_result.dtype, mode='r', shape=skeleton_array_result.shape)
                self._add_layer_safely(viewer, skel_display_memmap, "Skeletons", layer_type='labels')
            
            return True
        except Exception as e:
            print(f"Error during feature calculation: {e}"); traceback.print_exc()
            return False

    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        if viewer is None: return
        files = self.get_checkpoint_files()
        print(f"Loading checkpoint data up to step {checkpoint_step} using memmap...")
        
        layer_base_names = ["Raw Intermediate Segmentation", "Trimmed Intermediate Segmentation", 
                            "Edge Mask", "Final segmentation", "Cell bodies", "Skeletons"]
        for name in layer_base_names:
            self._remove_layer_safely(viewer, name)
        
        if checkpoint_step >= 1:
            path = files.get("raw_segmentation")
            if path and os.path.exists(path):
                self._add_layer_safely(viewer, np.memmap(path, dtype=np.int32, mode='r', shape=self.image_shape), "Raw Intermediate Segmentation")
        if checkpoint_step >= 2:
            path = files.get("trimmed_segmentation")
            if path and os.path.exists(path):
                self._add_layer_safely(viewer, np.memmap(path, dtype=np.int32, mode='r', shape=self.image_shape), "Trimmed Intermediate Segmentation")
            path = files.get("edge_mask")
            if path and os.path.exists(path):
                self._add_layer_safely(viewer, np.memmap(path, dtype=bool, mode='r', shape=self.image_shape), "Edge Mask", layer_type='image', colormap='gray', blending='additive')
        if checkpoint_step >= 3:
            path = files.get("final_segmentation")
            if path and os.path.exists(path):
                self._add_layer_safely(viewer, np.memmap(path, dtype=np.int32, mode='r', shape=self.image_shape), "Final segmentation")
            path = files.get("cell_bodies")
            if path and os.path.exists(path):
                # We need to know the shape of the cell_bodies array to load it.
                # A robust way is to save shape metadata, but for now we assume it's the same as the image.
                self._add_layer_safely(viewer, np.memmap(path, dtype=np.int32, mode='r', shape=self.image_shape), "Cell bodies")
        if checkpoint_step >= 4:
            path = files.get("skeleton_array")
            if path and os.path.exists(path):
                self._add_layer_safely(viewer, np.memmap(path, dtype=np.int32, mode='r', shape=self.image_shape), "Skeletons", layer_type='labels')

    def cleanup_step_artifacts(self, viewer, step_number: int):
        files = self.get_checkpoint_files()
        print(f"Cleaning artifacts for step {step_number}...")

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
            self._remove_file_safely(files.get("final_segmentation"))
            self._remove_file_safely(files.get("cell_bodies"))
        elif step_number == 4:
            self._remove_layer_safely(viewer, "Skeletons")
            self._remove_file_safely(files.get("skeleton_array"))
            self._remove_file_safely(files.get("metrics_df"))