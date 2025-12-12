# --- START OF FILE utils/module_3d/_3D_strategy.py ---
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import sys
import traceback
import gc
from shutil import rmtree, copyfile
import time
import tempfile

from ..high_level_gui.processing_strategies import ProcessingStrategy, StepDefinition

try:
    from .initial_3d_segmentation import segment_cells_first_pass_raw
    from .remove_artifacts import apply_hull_trimming
    from .ramified_segmenter import extract_soma_masks, separate_multi_soma_cells
    from .calculate_features_3d import analyze_segmentation
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 3D segmentation modules: {e}")
    raise


class RamifiedStrategy(ProcessingStrategy):
    """
    Orchestrates the 5-step segmentation workflow for Ramified Microglia in 3D.
    """

    def _get_mode_name(self) -> str:
        return "ramified"

    def get_step_definitions(self) -> List[StepDefinition]:
        """
        Defines the workflow.
        Format: {'method': 'function_name', 'artifact': 'checkpoint_file_key'}
        
        The 'artifact' key corresponds to keys in get_checkpoint_files().
        If the file exists, the step is considered complete.
        """
        return [
            # Step 1
            {
                "method": "execute_raw_segmentation",
                "artifact": "raw_segmentation" 
            },
            # Step 2
            {
                "method": "execute_trim_edges",
                "artifact": "trimmed_segmentation"
            },
            # Step 3
            {
                "method": "execute_soma_extraction",
                "artifact": "cell_bodies"
            },
            # Step 4
            {
                "method": "execute_cell_separation",
                "artifact": "final_segmentation"
            },
            # Step 5
            {
                "method": "execute_calculate_features",
                "artifact": "metrics_df" 
            }
        ]

    def get_checkpoint_files(self) -> Dict[str, str]:
        files = super().get_checkpoint_files()
        mode_prefix = self.mode_name
        
        files["raw_segmentation"] = os.path.join(self.processed_dir, f"raw_segmentation_{mode_prefix}.dat")
        files["edge_mask"] = os.path.join(self.processed_dir, f"{mode_prefix}_edge_mask.dat")
        files["trimmed_segmentation"] = os.path.join(self.processed_dir, f"trimmed_segmentation_{mode_prefix}.dat")
        files["cell_bodies"] = os.path.join(self.processed_dir, f"cell_bodies.dat")
        files["final_segmentation"] = os.path.join(self.processed_dir, f"final_segmentation_{mode_prefix}.dat")
        files["skeleton_array"] = os.path.join(self.processed_dir, f"skeleton_array_{mode_prefix}.dat")
        
        files["distances_matrix"] = os.path.join(self.processed_dir, f"distances_matrix_{mode_prefix}.csv")
        files["points_matrix"] = os.path.join(self.processed_dir, f"points_matrix_{mode_prefix}.csv")
        files["branch_data"] = os.path.join(self.processed_dir, f"branch_data_{mode_prefix}.csv")
        
        return files

    def _close_memmap(self, memmap_obj: Union[np.memmap, np.ndarray, None]):
        if memmap_obj is None: return
        if isinstance(memmap_obj, np.memmap):
            try:
                memmap_obj.flush()
                if hasattr(memmap_obj, '_mmap') and memmap_obj._mmap is not None:
                    memmap_obj._mmap.close()
            except Exception as e:
                print(f"Warning: Error closing memmap: {e}")
        del memmap_obj

    # --- EXECUTION STEPS (Logic remains unchanged) ---

    def execute_raw_segmentation(self, viewer, image_stack: Any, params: Dict) -> bool:
        if image_stack is None: return False
        print(f"Executing Step 1: Raw {self.mode_name} segmentation...")
        files = self.get_checkpoint_files()
        persistent_raw_dat_path = files.get("raw_segmentation")
        temp_raw_labels_dir = None
        
        try:
            tubular_scales_list = params.get("tubular_scales", [0.8, 1.0, 1.5, 2.0])
            skip_tubular_enhancement = len(tubular_scales_list) == 1 and tubular_scales_list[0] == 0.0
            
            temp_raw_labels_dat_path, temp_raw_labels_dir, segmentation_threshold, _ = \
                segment_cells_first_pass_raw(
                    volume=image_stack, 
                    spacing=self.spacing,
                    tubular_scales=tubular_scales_list, 
                    smooth_sigma=float(params.get("smooth_sigma", 1.3)),
                    connect_max_gap_physical=float(params.get("connect_max_gap_physical", 1.0)), 
                    min_size_voxels=int(params.get("min_size", 2000)),
                    low_threshold_percentile=float(params.get("low_threshold_percentile", 25.0)),
                    high_threshold_percentile=float(params.get("high_threshold_percentile", 95.0)),
                    skip_tubular_enhancement=skip_tubular_enhancement
                )

            if not temp_raw_labels_dat_path or not os.path.exists(temp_raw_labels_dat_path):
                raise RuntimeError("Raw segmentation function failed.")

            self.intermediate_state['segmentation_threshold'] = segmentation_threshold
            self.intermediate_state['original_volume_ref'] = image_stack

            copyfile(temp_raw_labels_dat_path, persistent_raw_dat_path)
            
            if viewer is not None:
                raw_labels_for_display = np.memmap(persistent_raw_dat_path, dtype=np.int32, mode='r', shape=self.image_shape)
                self._add_layer_safely(viewer, raw_labels_for_display, "Raw Intermediate Segmentation")
            return True

        except Exception as e:
             print(f"Error during execute_raw_segmentation: {e}")
             traceback.print_exc()
             return False
        finally:
             if temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir):
                 rmtree(temp_raw_labels_dir, ignore_errors=True)
             gc.collect()

    def execute_trim_edges(self, viewer, image_stack: Any, params: Dict) -> bool:
        print(f"Executing Step 2: Edge Trimming...")
        files = self.get_checkpoint_files()
        raw_seg_input_path = files.get("raw_segmentation")
        trimmed_seg_output_path = files.get("trimmed_segmentation")
        edge_mask_output_path = files.get("edge_mask")

        if not os.path.exists(raw_seg_input_path): return False
        if 'segmentation_threshold' not in self.intermediate_state: return False
        
        temp_trimmed_dir = None
        hull_boundary_mask = None
        
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

            if not temp_trimmed_dat_path or not os.path.exists(temp_trimmed_dat_path):
                raise RuntimeError("apply_hull_trimming failed.")

            copyfile(temp_trimmed_dat_path, trimmed_seg_output_path)
            edge_mask_memmap = np.memmap(edge_mask_output_path, dtype=bool, mode='w+', shape=self.image_shape)
            if hull_boundary_mask is not None:
                edge_mask_memmap[:] = hull_boundary_mask[:]
            self._close_memmap(edge_mask_memmap)

            if viewer is not None:
                trimmed_labels_for_display = np.memmap(trimmed_seg_output_path, dtype=np.int32, mode='r', shape=self.image_shape)
                self._add_layer_safely(viewer, trimmed_labels_for_display, "Trimmed Intermediate Segmentation")
                edge_mask_for_display = np.memmap(edge_mask_output_path, dtype=bool, mode='r', shape=self.image_shape)
                self._add_layer_safely(viewer, edge_mask_for_display, "Edge Mask", layer_type='image', colormap='gray', blending='additive')

            return True

        except Exception as e:
             print(f"Error during trim_edges: {e}")
             traceback.print_exc()
             return False
        finally:
             if temp_trimmed_dir and os.path.exists(temp_trimmed_dir):
                 rmtree(temp_trimmed_dir, ignore_errors=True)
             if 'hull_boundary_mask' in locals(): del hull_boundary_mask
             gc.collect()

    def execute_soma_extraction(self, viewer, image_stack, params: Dict) -> bool:
        print(f"Executing Step 3: Soma Extraction...")
        files = self.get_checkpoint_files()
        trimmed_seg_path = files["trimmed_segmentation"]
        cell_bodies_path = files["cell_bodies"]
        
        if not os.path.exists(trimmed_seg_path): return False
        
        trimmed_labels_memmap = None
        try:
            trimmed_labels_memmap = np.memmap(trimmed_seg_path, dtype=np.int32, mode='r', shape=self.image_shape)
            
            # --- UPDATED PARAMETER DICTIONARY ---
            soma_extraction_params = {
                "smallest_quantile": float(params.get("smallest_quantile", 25)),
                "min_fragment_size": int(params.get("min_fragment_size", 30)),
                "core_volume_target_factor_lower": float(params.get("core_volume_target_factor_lower", 0.1)),
                "core_volume_target_factor_upper": float(params.get("core_volume_target_factor_upper", 10.0)),
                "erosion_iterations": int(params.get("erosion_iterations", 0)),
                "ratios_to_process": params.get("ratios_to_process", [0.3, 0.4, 0.5, 0.6]),
                "intensity_percentiles_to_process": params.get("intensity_percentiles_to_process", [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]),
                "min_physical_peak_separation": float(params.get("min_physical_peak_separation", 7.0)),
                
                # --- NEW PARAMETER HERE ---
                # We use .get() without a default, so it returns None if missing.
                # The extract function handles None by falling back to legacy behavior.
                "seeding_min_distance_um": params.get("seeding_min_distance_um"),
                # --------------------------

                "max_allowed_core_aspect_ratio": float(params.get("max_allowed_core_aspect_ratio", 10.0)),
                "ref_vol_percentile_lower": int(params.get("ref_vol_percentile_lower", 30)),
                "ref_vol_percentile_upper": int(params.get("ref_vol_percentile_upper", 70)),
                "ref_thickness_percentile_lower": int(params.get("ref_thickness_percentile_lower", 1)),
                "absolute_min_thickness_um": float(params.get("absolute_min_thickness_um", 1.5)),
                "absolute_max_thickness_um": float(params.get("absolute_max_thickness_um", 10.0)),
                "memmap_final_mask": True 
            }
            
            # Now **soma_extraction_params will contain the new key
            cell_bodies = extract_soma_masks(trimmed_labels_memmap, image_stack, self.spacing, **soma_extraction_params)
            
            if isinstance(cell_bodies, np.memmap):
                temp_cb_path = cell_bodies.filename
                self._close_memmap(cell_bodies)
                copyfile(temp_cb_path, cell_bodies_path)
                if os.path.exists(temp_cb_path): os.remove(temp_cb_path)
            else:
                cb_memmap = np.memmap(cell_bodies_path, dtype=cell_bodies.dtype, mode='w+', shape=cell_bodies.shape)
                cb_memmap[:] = cell_bodies[:]
                self._close_memmap(cb_memmap)
                del cell_bodies

            if viewer is not None:
                cb_display = np.memmap(cell_bodies_path, dtype=np.int32, mode='r', shape=self.image_shape)
                self._add_layer_safely(viewer, cb_display, "Cell bodies")
            return True

        except Exception as e:
             print(f"Error during Soma Extraction: {e}")
             traceback.print_exc()
             return False
        finally:
             self._close_memmap(trimmed_labels_memmap)
             temp_soma_dir = os.path.join(self.processed_dir, "ramiseg_temp_memmap") 
             if os.path.exists(temp_soma_dir): rmtree(temp_soma_dir, ignore_errors=True)
             gc.collect()

    def execute_cell_separation(self, viewer, image_stack, params: Dict) -> bool:
        print(f"Executing Step 4: Cell Separation...")
        files = self.get_checkpoint_files()
        trimmed_seg_path = files["trimmed_segmentation"]
        cell_bodies_path = files["cell_bodies"]
        final_seg_path = files["final_segmentation"]
        
        if not os.path.exists(trimmed_seg_path) or not os.path.exists(cell_bodies_path): return False
        
        trimmed_labels_memmap = None
        cell_bodies_ref = None
        
        try:
            trimmed_labels_memmap = np.memmap(trimmed_seg_path, dtype=np.int32, mode='r', shape=self.image_shape)
            cell_bodies_ref = np.memmap(cell_bodies_path, dtype=np.int32, mode='r', shape=self.image_shape)
            
            separation_params = {
                "min_size_threshold": int(params.get("min_size_threshold", 100)),
                "intensity_weight": float(params.get("intensity_weight", 0.0)),
                "max_seed_centroid_dist": float(params.get("max_seed_centroid_dist", 40.0)),
                "min_path_intensity_ratio": float(params.get("min_path_intensity_ratio", 0.8)),
                "min_local_intensity_difference": float(params.get("min_local_intensity_difference", 0.05)),
                "local_analysis_radius": int(params.get("local_analysis_radius", 10)),
                "memmap_dir": os.path.join(self.processed_dir, "sep_multi_soma_temp"),
                "memmap_voxel_threshold": int(params.get("memmap_voxel_threshold", 25_000_000))
            }
            
            final_separated_cells = separate_multi_soma_cells(
                trimmed_labels_memmap, image_stack, cell_bodies_ref, self.spacing, **separation_params
            )
            
            final_memmap = np.memmap(final_seg_path, dtype=np.int32, mode='w+', shape=self.image_shape)
            final_memmap[:] = final_separated_cells[:]
            self._close_memmap(final_memmap)
            
            if viewer is not None:
                final_display = np.memmap(final_seg_path, dtype=np.int32, mode='r', shape=self.image_shape)
                self._add_layer_safely(viewer, final_display, "Final segmentation")
            return True

        except Exception as e:
             print(f"Error during Cell Separation: {e}")
             traceback.print_exc()
             return False
        finally:
             self._close_memmap(trimmed_labels_memmap)
             self._close_memmap(cell_bodies_ref)
             if 'final_separated_cells' in locals(): del final_separated_cells
             
             temp_chunk_dir = os.path.join(self.processed_dir, "sep_multi_soma_temp")
             if os.path.exists(temp_chunk_dir): rmtree(temp_chunk_dir, ignore_errors=True)
             gc.collect()

    def execute_calculate_features(self, viewer, image_stack, params: Dict) -> bool:
        print(f"Executing Step 5: Feature Calculation...")
        files = self.get_checkpoint_files()
        final_seg_path = files["final_segmentation"]
        
        if not os.path.exists(final_seg_path): return False

        final_seg_memmap = None
        try:
            final_seg_memmap = np.memmap(final_seg_path, dtype=np.int32, mode='r', shape=self.image_shape)

            metrics_df, detailed_outputs = analyze_segmentation(
                segmented_array=final_seg_memmap, spacing=self.spacing, **params, return_detailed=True
            )
            
            if metrics_df is not None: 
                metrics_df.to_csv(files["metrics_df"], index=False)

            skeleton_array_result = detailed_outputs.get('skeleton_array')
            if skeleton_array_result is not None: 
                skeleton_path = files.get("skeleton_array")
                skel_memmap = np.memmap(skeleton_path, dtype=skeleton_array_result.dtype, mode='w+', shape=skeleton_array_result.shape)
                skel_memmap[:] = skeleton_array_result[:]
                self._close_memmap(skel_memmap)
                
                del skeleton_array_result
                if 'skeleton_array' in detailed_outputs: del detailed_outputs['skeleton_array']
                gc.collect()
            
            points_df = detailed_outputs.get('all_pairs_points')
            if points_df is not None and not points_df.empty:
                points_df.to_csv(files["points_matrix"], index=False)

            if viewer is not None:
                skeleton_path = files.get("skeleton_array")
                if os.path.exists(skeleton_path):
                    skel_display_memmap = np.memmap(skeleton_path, dtype=np.int32, mode='r', shape=self.image_shape)
                    self._add_layer_safely(viewer, skel_display_memmap, "Skeletons", layer_type='labels')
                if points_df is not None and not points_df.empty:
                    self._add_neighbor_lines(viewer, points_df)
            return True
        except Exception as e:
            print(f"Error during feature calculation: {e}")
            traceback.print_exc()
            return False
        finally:
            self._close_memmap(final_seg_memmap)
            gc.collect()

    def _add_neighbor_lines(self, viewer, points_df):
        if points_df is None or points_df.empty: return
        lines = []
        for _, row in points_df.iterrows():
            p1 = [row['mask1_z'], row['mask1_y'], row['mask1_x']]
            p2 = [row['mask2_z'], row['mask2_y'], row['mask2_x']]
            lines.append([p1, p2])
            
        layer_name = f"Neighbor Connections_{self.mode_name}"
        if layer_name in viewer.layers: viewer.layers.remove(layer_name)
        viewer.add_shapes(
            lines, shape_type='line', edge_color='red', edge_width=1, name=layer_name,
            scale=tuple(self.spacing) if hasattr(self, 'z_scale_factor') and self.z_scale_factor == 1.0 else (self.z_scale_factor, 1, 1)
        )

    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        if viewer is None: return
        files = self.get_checkpoint_files()
        print(f"Loading checkpoint data up to step {checkpoint_step}...")
        
        layer_base_names = ["Raw Intermediate Segmentation", "Trimmed Intermediate Segmentation", 
                            "Edge Mask", "Final segmentation", "Cell bodies", "Skeletons", "Neighbor Connections"]
        for name in layer_base_names:
            self._remove_layer_safely(viewer, name)
        
        def load_and_add(path_key, layer_name, dtype=np.int32, **kwargs):
            path = files.get(path_key)
            if path and os.path.exists(path):
                data = np.memmap(path, dtype=dtype, mode='r', shape=self.image_shape)
                self._add_layer_safely(viewer, data, layer_name, **kwargs)

        if checkpoint_step >= 1:
            load_and_add("raw_segmentation", "Raw Intermediate Segmentation")
        if checkpoint_step >= 2:
            load_and_add("trimmed_segmentation", "Trimmed Intermediate Segmentation")
            load_and_add("edge_mask", "Edge Mask", dtype=bool, layer_type='image', colormap='gray', blending='additive')
        if checkpoint_step >= 3:
            load_and_add("cell_bodies", "Cell bodies")
        if checkpoint_step >= 4:
            load_and_add("final_segmentation", "Final segmentation")
        if checkpoint_step >= 5:
            load_and_add("skeleton_array", "Skeletons", layer_type='labels')
            pts_path = files.get("points_matrix")
            if pts_path and os.path.exists(pts_path):
                try:
                    df = pd.read_csv(pts_path)
                    self._add_neighbor_lines(viewer, df)
                except Exception as e:
                    print(f"Error loading neighbor lines: {e}")

    def cleanup_step_artifacts(self, viewer, step_number: int):
        files = self.get_checkpoint_files()
        if step_number == 1:
            self._remove_layer_safely(viewer, "Raw Intermediate Segmentation")
            self._remove_file_safely(files.get("raw_segmentation"))
        elif step_number == 2:
            self._remove_layer_safely(viewer, "Trimmed Intermediate Segmentation")
            self._remove_layer_safely(viewer, "Edge Mask")
            self._remove_file_safely(files.get("trimmed_segmentation"))
            self._remove_file_safely(files.get("edge_mask"))
        elif step_number == 3:
            self._remove_layer_safely(viewer, "Cell bodies")
            self._remove_file_safely(files.get("cell_bodies"))
        elif step_number == 4:
            self._remove_layer_safely(viewer, "Final segmentation")
            self._remove_file_safely(files.get("final_segmentation"))
        elif step_number == 5:
            self._remove_layer_safely(viewer, "Skeletons")
            self._remove_layer_safely(viewer, "Neighbor Connections")
            self._remove_file_safely(files.get("skeleton_array"))
            self._remove_file_safely(files.get("metrics_df"))
            self._remove_file_safely(files.get("branch_data"))
            self._remove_file_safely(files.get("distances_matrix"))
            self._remove_file_safely(files.get("points_matrix"))
# --- END OF FILE utils/module_3d/_3D_strategy.py ---