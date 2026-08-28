import os
import sys
import gc
import time
import traceback
import random
import colorsys
import shutil
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import yaml  # type: ignore
import tifffile as tiff

from ..high_level_gui.processing_strategies import ProcessingStrategy, StepDefinition

# Attempt imports of specific 3D segmentation modules
try:
    from .initial_3d_segmentation import segment_cells_first_pass_raw
    from .remove_artifacts import apply_hull_trimming
    from .soma_extraction import extract_soma_masks
    from .cell_splitting import separate_multi_soma_cells
    from .calculate_features_3d import analyze_segmentation, export_to_fcs
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 3D segmentation modules: {e}")
    raise


class FluorescenceStrategy(ProcessingStrategy):
    """
    Orchestrates the 3D segmentation workflow for Fluorescent cells.

    This strategy manages a 5-step pipeline:
    1. Raw Segmentation (Hessian/Frangi + Thresholding)
    2. Edge Trimming (Hull-based artifact removal)
    3. Soma Extraction (Core detection)
    4. Cell Separation (Splitting multi-soma objects)
    5. Feature Calculation (Morphometrics, Skeletonization)

    Cross-channel work is not part of this pipeline: it operates on the finished
    segmentations of several channels at once and lives in the Cross-Channel
    Analyzer (`cross_channel_window` / `relational_engine`).
    """

    def _get_mode_name(self) -> str:
        """Returns the unique identifier for this strategy."""
        return "fluorescence"

    def get_step_definitions(self) -> List[StepDefinition]:
        """
        Defines the sequential steps and their completion artifacts.

        Returns:
            List[StepDefinition]: Ordered list of steps.
        """
        return [
            {
                "method": "execute_raw_segmentation",
                "artifact": "raw_segmentation"
            },
            {
                "method": "execute_trim_edges",
                "artifact": "trimmed_segmentation"
            },
            {
                "method": "execute_soma_extraction",
                "artifact": "cell_bodies"
            },
            {
                "method": "execute_cell_separation",
                "artifact": "final_segmentation"
            },
            {
                "method": "execute_calculate_features",
                "artifact": "metrics_df"
            }
        ]

    def get_checkpoint_files(self) -> Dict[str, str]:
        """
        Defines file paths for all intermediate and final outputs.

        Returns:
            Dict[str, str]: Map of artifact keys to file paths.
        """
        files = super().get_checkpoint_files()
        p = self.mode_name
        files.update({
            "raw_segmentation": os.path.join(
                self.processed_dir, f"raw_segmentation_{p}.dat"
            ),
            "edge_mask": os.path.join(
                self.processed_dir, f"{p}_edge_mask.dat"
            ),
            "trimmed_segmentation": os.path.join(
                self.processed_dir, f"trimmed_segmentation_{p}.dat"
            ),
            "cell_bodies": os.path.join(
                self.processed_dir, "cell_bodies.dat"
            ),
            "final_segmentation": os.path.join(
                self.processed_dir, f"final_segmentation_{p}.dat"
            ),
            "skeleton_array": os.path.join(
                self.processed_dir, f"skeleton_array_{p}.dat"
            ),
            "distances_matrix": os.path.join(
                self.processed_dir, f"distances_matrix_{p}.csv"
            ),
            "points_matrix": os.path.join(
                self.processed_dir, f"points_matrix_{p}.csv"
            ),
            "branch_data": os.path.join(
                self.processed_dir, f"branch_data_{p}.csv"
            ),
            "metrics_fcs": os.path.join(
                self.processed_dir, f"metrics_{p}.fcs"
            ),
        })
        return files

    def _close_memmap(self, memmap_obj: Any):
        """Safely closes a numpy memmap object to release file locks."""
        if memmap_obj is None:
            return
        try:
            if isinstance(memmap_obj, np.memmap):
                memmap_obj.flush()
                if hasattr(memmap_obj, '_mmap') and memmap_obj._mmap:
                    memmap_obj._mmap.close()
        except Exception:
            pass
        del memmap_obj

    # =========================================================================
    # EXECUTION STEPS
    # =========================================================================

    def execute_raw_segmentation(
        self, viewer, image_stack: Any, params: Dict
    ) -> bool:
        """
        Step 1: Performs initial raw segmentation.

        Args:
            viewer: Napari viewer instance (or None).
            image_stack: Input 3D image array.
            params: Dictionary of parameters from config.

        Returns:
            bool: True if successful, False otherwise.
        """
        if image_stack is None:
            return False
        print(f"Executing Step 1: Raw {self.mode_name} segmentation...")
        
        files = self.get_checkpoint_files()
        persistent_raw_dat_path = files.get("raw_segmentation")
        temp_raw_labels_dir = None

        try:
            # --- 1. PARAMETER PARSING ---
            # Extract toggle state
            is_absolute = params.get("use_absolute_thresholds", False)
            threshold_mode = "Absolute" if is_absolute else "Percentile"

            # Route to the appropriate table
            if is_absolute and "scale_profiles_absolute" in params:
                profiles = params["scale_profiles_absolute"]
            elif not is_absolute and "scale_profiles_percentile" in params:
                profiles = params["scale_profiles_percentile"]
            elif "scale_profiles" in params:
                # Fallback to legacy behavior
                profiles = params["scale_profiles"]
            else:
                profiles =[{"scale": 1.0, "low": 95.0, "high": 100.0}]

            tubular_scales = [p['scale'] for p in profiles]
            low_thresh_input = [p['low'] for p in profiles]
            high_thresh_input = [p['high'] for p in profiles]

            # Per-scale smoothing / gap-closing. Base value comes from the
            # step's top-level scalar param; a profile row may override it
            # per-scale by carrying its own key. (min_size stays global below.)
            base_smooth_sigma = float(params.get("smooth_sigma", 1.3))
            base_connect_gap = float(params.get("connect_max_gap_physical", 1.0))
            smooth_sigma_input = [
                float(p.get("smooth_sigma", base_smooth_sigma)) for p in profiles
            ]
            connect_max_gap_input = [
                float(p.get("connect_max_gap_physical", base_connect_gap))
                for p in profiles
            ]

            # Check for special 'skip' signal in scales
            skip_enhancement = (
                len(tubular_scales) == 1 and tubular_scales[0] == 0.0
            )

            # --- 2. CALL LOGIC ---
            result = segment_cells_first_pass_raw(
                volume=image_stack,
                spacing=self.spacing,
                tubular_scales=tubular_scales,
                smooth_sigma=smooth_sigma_input,
                connect_max_gap_physical=connect_max_gap_input,
                min_size_voxels=int(params.get("min_size", 2000)),
                # Use the variables prepared above
                low_threshold_percentile=low_thresh_input,
                high_threshold_percentile=high_thresh_input,
                threshold_mode=threshold_mode,
                skip_tubular_enhancement=skip_enhancement,
                trace_max_gap=float(params.get("trace_max_gap", 0.0)),
                temp_root_path=self.temp_dir
            )

            # Unpack results
            temp_dat_path, temp_raw_labels_dir, seg_threshold, _ = result

            if not temp_dat_path or not os.path.exists(temp_dat_path):
                raise RuntimeError("Raw segmentation function failed.")

            # Store state and persist result
            self.intermediate_state['segmentation_threshold'] = seg_threshold
            self.intermediate_state['original_volume_ref'] = image_stack
            shutil.copyfile(temp_dat_path, persistent_raw_dat_path)

            if viewer is not None:
                display_data = np.memmap(
                    persistent_raw_dat_path, dtype=np.int32, mode='r',
                    shape=self.image_shape
                )
                self._add_layer_safely(
                    viewer, display_data, "Raw Intermediate Segmentation"
                )
            return True

        except Exception as e:
            print(f"Error during execute_raw_segmentation: {e}")
            traceback.print_exc()
            return False
        finally:
            if temp_raw_labels_dir and os.path.exists(temp_raw_labels_dir):
                shutil.rmtree(temp_raw_labels_dir, ignore_errors=True)
            gc.collect()

    def execute_trim_edges(
        self, viewer, image_stack: Any, params: Dict
    ) -> bool:
        """
        Step 2: Removes artifacts near the tissue block edges (Hull Trimming).
        """
        print(f"Executing Step 2: Edge Trimming...")
        files = self.get_checkpoint_files()
        raw_seg_path = files.get("raw_segmentation")
        trimmed_seg_path = files.get("trimmed_segmentation")
        edge_mask_path = files.get("edge_mask")

        if not os.path.exists(raw_seg_path):
            return False
        if 'segmentation_threshold' not in self.intermediate_state:
            return False

        temp_trimmed_dir = None
        hull_boundary_mask = None

        try:
            temp_dat_path, temp_trimmed_dir, hull_boundary_mask = apply_hull_trimming(
                raw_labels_path=raw_seg_path,
                original_volume=self.intermediate_state['original_volume_ref'],
                spacing=self.spacing,
                segmentation_threshold=self.intermediate_state['segmentation_threshold'],
                edge_trim_distance_threshold=float(
                    params.get("edge_trim_distance_threshold", 4.5)
                ),
                brightness_cutoff_factor=float(
                    params.get("brightness_cutoff_factor", 1.5)
                ),
                min_size_voxels=int(params.get("min_size_voxels", 50)),
                hull_closing_radius=int(params.get("hull_closing_radius", 1)),
                z_erosion_iterations=int(params.get("z_erosion_iterations", 0)),
                otsu_scale_factor=float(params.get("otsu_scale_factor", 0.8)),
                temp_root_path=self.temp_dir
            )

            if not temp_dat_path or not os.path.exists(temp_dat_path):
                raise RuntimeError("apply_hull_trimming failed.")

            # Persist Trimmed Segmentation
            shutil.copyfile(temp_dat_path, trimmed_seg_path)

            # Persist Edge Mask
            edge_memmap = np.memmap(
                edge_mask_path, dtype=bool, mode='w+', shape=self.image_shape
            )
            if hull_boundary_mask is not None:
                edge_memmap[:] = hull_boundary_mask[:]
            self._close_memmap(edge_memmap)

            if viewer is not None:
                trimmed_display = np.memmap(
                    trimmed_seg_path, dtype=np.int32, mode='r',
                    shape=self.image_shape
                )
                self._add_layer_safely(
                    viewer, trimmed_display, "Trimmed Intermediate Segmentation"
                )
                edge_display = np.memmap(
                    edge_mask_path, dtype=bool, mode='r',
                    shape=self.image_shape
                )
                self._add_layer_safely(
                    viewer, edge_display, "Edge Mask",
                    layer_type='image', colormap='gray', blending='additive'
                )

            return True

        except Exception as e:
            print(f"Error during trim_edges: {e}")
            traceback.print_exc()
            return False
        finally:
            if temp_trimmed_dir and os.path.exists(temp_trimmed_dir):
                shutil.rmtree(temp_trimmed_dir, ignore_errors=True)
            if 'hull_boundary_mask' in locals():
                del hull_boundary_mask
            gc.collect()

    def execute_soma_extraction(
        self, viewer, image_stack: Any, params: Dict
    ) -> bool:
        """
        Step 3: Extracts cell bodies (somas) from the segmented volume.
        """
        print(f"Executing Step 3: Soma Extraction...")
        files = self.get_checkpoint_files()
        trimmed_seg_path = files["trimmed_segmentation"]
        cell_bodies_path = files["cell_bodies"]

        if not os.path.exists(trimmed_seg_path):
            return False

        trimmed_labels_memmap = None
        try:
            trimmed_labels_memmap = np.memmap(
                trimmed_seg_path, dtype=np.int32, mode='r',
                shape=self.image_shape
            )

            soma_extraction_params = {
                "min_fragment_size": int(params.get("min_fragment_size", 100)),
                "erosion_iterations": int(params.get("erosion_iterations", 0)),
                "ratios_to_process": params.get(
                    "ratios_to_process", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                ),
                "intensity_percentiles_to_process": params.get(
                    "intensity_percentiles_to_process",
                    [99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 1]
                ),
                "min_physical_peak_separation": float(
                    params.get("min_physical_peak_separation", 5.0)
                ),
                "max_allowed_core_aspect_ratio": float(
                    params.get("max_allowed_core_aspect_ratio", 5.0)
                ),
                "absolute_min_thickness_um": float(
                    params.get("absolute_min_thickness_um", 1.5)
                ),
                "absolute_max_thickness_um": float(
                    params.get("absolute_max_thickness_um", 10.0)
                ),
                "memmap_final_mask": True,
                "temp_root_path": self.temp_dir
            }

            cell_bodies = extract_soma_masks(
                trimmed_labels_memmap, image_stack, self.spacing,
                **soma_extraction_params
            )

            # Persist results
            if isinstance(cell_bodies, np.memmap):
                temp_cb_path = cell_bodies.filename
                self._close_memmap(cell_bodies)
                shutil.copyfile(temp_cb_path, cell_bodies_path)
                if os.path.exists(temp_cb_path):
                    os.remove(temp_cb_path)
            else:
                cb_memmap = np.memmap(
                    cell_bodies_path, dtype=cell_bodies.dtype, mode='w+',
                    shape=cell_bodies.shape
                )
                cb_memmap[:] = cell_bodies[:]
                self._close_memmap(cb_memmap)
                del cell_bodies

            if viewer is not None:
                cb_display = np.memmap(
                    cell_bodies_path, dtype=np.int32, mode='r',
                    shape=self.image_shape
                )
                self._add_layer_safely(viewer, cb_display, "Cell bodies")
            return True

        except Exception as e:
            print(f"Error during Soma Extraction: {e}")
            traceback.print_exc()
            return False
        finally:
            self._close_memmap(trimmed_labels_memmap)
            # Cleanup internal temporary dir of soma extractor if known
            temp_soma_dir = os.path.join(
                self.processed_dir, "ramiseg_temp_memmap"
            )
            if os.path.exists(temp_soma_dir):
                shutil.rmtree(temp_soma_dir, ignore_errors=True)
            gc.collect()

    def execute_cell_separation(
        self, viewer, image_stack: Any, params: Dict
    ) -> bool:
        """
        Step 4: Separates merged cells using extracted somas as seeds.
        """
        print(f"Executing Step 4: Cell Separation...")
        files = self.get_checkpoint_files()
        trimmed_seg_path = files["trimmed_segmentation"]
        cell_bodies_path = files["cell_bodies"]
        final_seg_path = files["final_segmentation"]

        # Dedicated, isolated temp dir for this step's on-disk chunk arrays. It is
        # defined ONCE here and reused for cleanup in `finally` so the write dir
        # and cleanup dir can never drift apart. It must NOT be the shared
        # `self.temp_dir` (temp_artifacts): other steps and the pipeline
        # create/clear that directory, which can delete these chunks mid-stitch
        # and cause a FileNotFoundError when the stitch phase reloads them.
        temp_chunk_dir = os.path.join(self.processed_dir, "sep_multi_soma_temp")

        if not os.path.exists(trimmed_seg_path) or not os.path.exists(cell_bodies_path):
            return False

        trimmed_labels_memmap = None
        cell_bodies_ref = None
        final_separated_cells = None

        try:
            trimmed_labels_memmap = np.memmap(
                trimmed_seg_path, dtype=np.int32, mode='r',
                shape=self.image_shape
            )
            cell_bodies_ref = np.memmap(
                cell_bodies_path, dtype=np.int32, mode='r',
                shape=self.image_shape
            )

            separation_params = {
                "min_size_threshold": int(params.get("min_size_threshold", 100)),
                "intensity_weight": float(params.get("intensity_weight", 0.0)),
                "max_seed_centroid_dist": float(
                    params.get("max_seed_centroid_dist", 40.0)
                ),
                "min_path_intensity_ratio": float(
                    params.get("min_path_intensity_ratio", 0.8)
                ),
                "min_local_intensity_difference": float(
                    params.get("min_local_intensity_difference", 0.05)
                ),
                "local_analysis_radius": int(
                    params.get("local_analysis_radius", 10)
                ),
                "memmap_dir": temp_chunk_dir,
                "memmap_voxel_threshold": int(
                    params.get("memmap_voxel_threshold", 25_000_000)
                )
            }

            final_separated_cells = separate_multi_soma_cells(
                trimmed_labels_memmap, image_stack, cell_bodies_ref,
                self.spacing, **separation_params
            )

            final_memmap = np.memmap(
                final_seg_path, dtype=np.int32, mode='w+',
                shape=self.image_shape
            )
            final_memmap[:] = final_separated_cells[:]
            self._close_memmap(final_memmap)

            if viewer is not None:
                final_display = np.memmap(
                    final_seg_path, dtype=np.int32, mode='r',
                    shape=self.image_shape
                )
                self._add_layer_safely(
                    viewer, final_display, "Final segmentation"
                )
            return True

        except Exception as e:
            print(f"Error during Cell Separation: {e}")
            traceback.print_exc()
            return False
        finally:
            self._close_memmap(trimmed_labels_memmap)
            self._close_memmap(cell_bodies_ref)
            if 'final_separated_cells' in locals():
                del final_separated_cells

            if os.path.exists(temp_chunk_dir):
                shutil.rmtree(temp_chunk_dir, ignore_errors=True)
            gc.collect()

    def execute_calculate_features(
        self, viewer, image_stack: Any, params: Dict
    ) -> bool:
        """
        Step 5: Calculates morphometrics and skeletonizes the cells.
        """
        print(f"Executing Step 5: Feature Calculation...")
        files = self.get_checkpoint_files()
        final_seg_path = files["final_segmentation"]
        
        # 1. SETUP PATHS
        skel_dat_path = files.get("skeleton_array")
        metrics_csv_path = files.get("metrics_df")
        fcs_path = os.path.join(self.processed_dir, f"metrics_{self.mode_name}.fcs")
        pts_csv_path = files.get("points_matrix")
        dist_csv_path = files.get("distances_matrix")
        branch_csv_path = files.get("branch_data")

        if not os.path.exists(final_seg_path):
            return False

        final_seg_memmap = None
        try:
            final_seg_memmap = np.memmap(
                final_seg_path, dtype=np.int32, mode='r',
                shape=self.image_shape
            )

            intensity_vol = self.intermediate_state.get(
                'original_volume_ref', image_stack
            )

            # 2. CALL ANALYSIS
            metrics_df, detailed_outputs = analyze_segmentation(
                segmented_array=final_seg_memmap,
                intensity_image=intensity_vol,
                spacing=self.spacing,
                temp_dir=self.temp_dir,
                calculate_distances=params.get("calculate_distances", True),
                calculate_skeletons=params.get("calculate_skeletons", True),
                calculate_solidity=params.get("calculate_solidity", False),
                skeleton_export_path=skel_dat_path,
                fcs_export_path=fcs_path,
                return_detailed=True,
                prune_spurs_le_um=params.get("prune_spurs_le_um", 0.0)
            )

            # Record the region these measurements came from, so counts can be
            # normalised: the full image extent, or an ROI's polygon volume.
            metrics_df = self.stamp_analyzed_extent(metrics_df)
            if metrics_df is not None:
                metrics_df.to_csv(metrics_csv_path, index=False)
            # Written unconditionally: an image with zero detections still has a
            # meaningful analysed volume, and that is a result worth keeping.
            self.write_analysis_summary(metrics_df)

            # Persist full N×N pairwise distance matrix
            dist_df = detailed_outputs.get('distance_matrix')
            if dist_df is not None and not dist_df.empty:
                print(f"  Saving pairwise distance matrix to: {os.path.basename(dist_csv_path)}")
                dist_df.to_csv(dist_csv_path, index=True)

            # Persist nearest-neighbour connection coordinates
            points_df = detailed_outputs.get('all_pairs_points')
            if points_df is not None and not points_df.empty:
                points_df.to_csv(pts_csv_path, index=False)

            # Persist per-branch skan statistics
            branch_df = detailed_outputs.get('detailed_branches')
            if branch_df is not None and not branch_df.empty:
                print(f"  Saving branch data to: {os.path.basename(branch_csv_path)}")
                branch_df.to_csv(branch_csv_path, index=False)

            if viewer is not None:
                if skel_dat_path and os.path.exists(skel_dat_path):
                    skel_display = np.memmap(
                        skel_dat_path, dtype=np.int32, mode='r',
                        shape=self.image_shape
                    )
                    self._add_layer_safely(
                        viewer, self._build_label_pyramid(skel_display),
                        "Skeletons", layer_type='labels'
                    )
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

    # =========================================================================
    # VISUALIZATION & HELPERS
    # =========================================================================

    def _add_neighbor_lines(self, viewer, points_df):
        """Helper to draw lines between neighbor centroids in Napari."""
        if points_df is None or points_df.empty:
            return
        lines = []
        for _, row in points_df.iterrows():
            p1 = [row['mask1_z'], row['mask1_y'], row['mask1_x']]
            p2 = [row['mask2_z'], row['mask2_y'], row['mask2_x']]
            lines.append([p1, p2])

        layer_name = f"Neighbor Connections_{self.mode_name}"
        if layer_name in viewer.layers:
            viewer.layers.remove(layer_name)

        display_scale = (self.z_scale_factor, 1, 1)

        viewer.add_shapes(
            lines, shape_type='line', edge_color='red', edge_width=1,
            name=layer_name, scale=display_scale
        )

    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        """Loads results into Napari for the given completion state."""
        if viewer is None:
            return
        files = self.get_checkpoint_files()
        print(f"Loading checkpoint data up to step {checkpoint_step}...")

        # 1. Clean up old specific layers
        for layer in viewer.layers:
            if "Ref:" in layer.name or "Overlap:" in layer.name:
                viewer.layers.remove(layer.name)

        layer_base_names = [
            "Raw Intermediate Segmentation", "Trimmed Intermediate Segmentation",
            "Edge Mask", "Final segmentation", "Cell bodies", "Skeletons",
            "Neighbor Connections"
        ]
        for name in layer_base_names:
            self._remove_layer_safely(viewer, name)

        def load_and_add(path_key, layer_name, dtype=np.int32, **kwargs):
            path = files.get(path_key)
            if path and os.path.exists(path):
                data = np.memmap(
                    path, dtype=dtype, mode='r', shape=self.image_shape
                )
                self._add_layer_safely(viewer, data, layer_name, **kwargs)

        # 2. Load layers based on progress
        if checkpoint_step >= 1:
            load_and_add("raw_segmentation", "Raw Intermediate Segmentation")
        if checkpoint_step >= 2:
            load_and_add("trimmed_segmentation", "Trimmed Intermediate Segmentation")
            load_and_add(
                "edge_mask", "Edge Mask", dtype=bool, layer_type='image',
                colormap='gray', blending='additive'
            )
        if checkpoint_step >= 3:
            load_and_add("cell_bodies", "Cell bodies")
        if checkpoint_step >= 4:
            load_and_add("final_segmentation", "Final segmentation")
        if checkpoint_step >= 5:
            path = files.get("skeleton_array")
            if path and os.path.exists(path):
                try:
                    skel_data = np.memmap(path, dtype=np.int32, mode='r', shape=self.image_shape)
                    self._add_layer_safely(
                        viewer, self._build_label_pyramid(skel_data),
                        "Skeletons", layer_type='labels'
                    )
                except Exception as e:
                    print(f"Error loading skeleton: {e}")
            pts_path = files.get("points_matrix")
            if pts_path and os.path.exists(pts_path):
                try:
                    df = pd.read_csv(pts_path)
                    self._add_neighbor_lines(viewer, df)
                except Exception as e:
                    print(f"Error loading neighbor lines: {e}")


    def cleanup_step_artifacts(self, viewer, step_number: int):
        """
        Removes temporary files and layers for a specific step to allow restart.
        """
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
            self._remove_file_safely(files.get("metrics_fcs"))