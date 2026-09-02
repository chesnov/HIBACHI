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
    from .initial_segmentation import segment_cells_first_pass_raw
    from .remove_artifacts import apply_hull_trimming
    from .soma_extraction import extract_soma_masks
    from .cell_splitting import separate_multi_soma_cells
    from .calculate_features import analyze_segmentation, export_to_fcs
    from .dim_utils import normalise_spacing
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import fluorescence pipeline modules: {e}")
    raise


def _require(params: Dict[str, Any], key: str, cast=None):
    """
    Fetch a REQUIRED parameter from the step's config block.

    No fallback. Every parameter this strategy needs is defined in the config, so
    a default here would be a second, invisible place to configure the pipeline.
    That is not hypothetical: the two former strategies carried 31 such
    fallbacks and 15 of them DISAGREED -- `min_fragment_size` 100 against 20,
    `min_physical_peak_separation` 5.0 against 6.0, `local_analysis_radius` 10
    against 2.0, `max_seed_centroid_dist` 40 against 20, and different percentile
    and ratio lists. All unreachable in the app, all silently divergent, and all
    a trap for anyone calling a step without a full config.

    A missing key now raises with the key named, rather than substituting a value
    that nothing in the config asked for.
    """
    if key not in params or params[key] is None:
        raise KeyError(
            f"required parameter '{key}' is missing from the step config. "
            f"Parameters come from the processing config, not from code "
            f"defaults; add it there rather than relying on a fallback."
        )
    v = params[key]
    return cast(v) if cast is not None else v


class FluorescenceStrategy(ProcessingStrategy):
    """
    Orchestrates the fluorescence segmentation workflow, in 2D or 3D.

    Rank is inferred from the data: `self.image_shape` decides, and every
    pipeline module reads `arr.ndim` and adapts. There is one strategy and one
    `mode` string ("fluorescence"); the legacy "fluorescence_2d" is accepted by
    the config loader and rewritten.

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
        """
        The single identifier for this strategy, at either rank.

        Deliberately NOT suffixed by rank. Checkpoint filenames are built from
        this, so a rank-dependent value would give 2D and 3D outputs different
        names for the same artifact -- which is what the two former strategies
        did, and why `cell_bodies.dat` was unsuffixed in one and
        `cell_bodies_fluorescence_2d.dat` in the other.
        """
        return "fluorescence"

    @property
    def spacing_checked(self):
        """
        Physical spacing, validated and reduced to the data's rank.

        Raises rather than substituting: every micron-valued parameter in the
        pipeline is interpreted against this, so a wrong or invented spacing
        makes every distance, size and density wrong by a constant factor with
        nothing in the output to reveal it. The 2D strategy previously reduced a
        3D spacing inline with `spacing[1:] if len(spacing) == 3 else spacing`;
        `normalise_spacing` is that rule, once.
        """
        return normalise_spacing(self.spacing, self.ndim)

    @property
    def ndim(self) -> int:
        """Rank of the data being processed, from the image shape."""
        return len(self.image_shape)

    @property
    def is_2d(self) -> bool:
        return self.ndim == 2

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
            is_absolute = _require(params, "use_absolute_thresholds", bool)
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

            # Per-scale smoothing and gap-closing, read from the profile ROWS.
            #
            # There is no top-level `smooth_sigma` / `connect_max_gap_physical`
            # parameter in the config and there never was: the former code read
            # `params.get("smooth_sigma", 1.3)` (3D) and `..., 0.1)` (2D), so the
            # hardcoded number was ALWAYS authoritative and the per-row values in
            # the config were only consulted where a row happened to carry them.
            # That is why the two tracks smoothed differently for identical
            # profiles. Both keys are now required per row -- which is what the
            # unified config guarantees, and what the 2D config always had.
            def _row(prof, key, idx):
                if key not in prof or prof[key] is None:
                    raise KeyError(
                        f"scale profile row {idx} is missing '{key}'. Every row "
                        f"must carry it; the 3D percentile table historically "
                        f"omitted these two keys while its own absolute table "
                        f"and both 2D tables had them."
                    )
                return float(prof[key])

            smooth_sigma_input = [
                _row(p, "smooth_sigma", i) for i, p in enumerate(profiles)
            ]
            connect_max_gap_input = [
                _row(p, "connect_max_gap_physical", i)
                for i, p in enumerate(profiles)
            ]

            # Check for special 'skip' signal in scales
            skip_enhancement = (
                len(tubular_scales) == 1 and tubular_scales[0] == 0.0
            )

            # --- 2. CALL LOGIC ---
            result = segment_cells_first_pass_raw(
                volume=image_stack,
                spacing=self.spacing_checked,
                tubular_scales=tubular_scales,
                smooth_sigma=smooth_sigma_input,
                connect_max_gap_physical=connect_max_gap_input,
                min_size_voxels=_require(params, "min_size", int),
                # Use the variables prepared above
                low_threshold_percentile=low_thresh_input,
                high_threshold_percentile=high_thresh_input,
                threshold_mode=threshold_mode,
                skip_tubular_enhancement=skip_enhancement,
                trace_max_gap=float(_require(params, "trace_max_gap", float)),
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
                spacing=self.spacing_checked,
                segmentation_threshold=self.intermediate_state['segmentation_threshold'],
                edge_trim_distance_threshold=float(
                    _require(params, "edge_trim_distance_threshold", float)
                ),
                brightness_cutoff_factor=float(
                    _require(params, "brightness_cutoff_factor", float)
                ),
                # The unified config key is `min_size` -- one key whose unit
                # follows the data's rank. The former tracks read
                # `min_size_voxels` and `min_size_pixels`.
                min_size_voxels=_require(params, "min_size", int),
                hull_closing_radius=int(_require(params, "hull_closing_radius", int)),
                z_erosion_iterations=int((_require(params, "z_erosion_iterations", int)
                                      if self.ndim == 3 else 0)),
                otsu_scale_factor=float(_require(params, "otsu_scale_factor", float)),
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
                "min_fragment_size": int(_require(params, "min_fragment_size", int)),
                "intensity_smooth_um": float(_require(params, "intensity_smooth_um", float)),
                "intensity_weight": float(_require(params, "intensity_weight", float)),
                "ratios_to_process": _require(params, "ratios_to_process"),
                "intensity_percentiles_to_process": _require(params, "intensity_percentiles_to_process"),
                "min_physical_peak_separation": float(
                    _require(params, "min_physical_peak_separation", float)
                ),
                "max_allowed_core_aspect_ratio": float(
                    _require(params, "max_allowed_core_aspect_ratio", float)
                ),
                "absolute_min_thickness_um": float(
                    _require(params, "absolute_min_thickness_um", float)
                ),
                "absolute_max_thickness_um": float(
                    _require(params, "absolute_max_thickness_um", float)
                ),
                "memmap_final_mask": True,
                "temp_root_path": self.temp_dir
            }

            cell_bodies = extract_soma_masks(
                trimmed_labels_memmap, image_stack, self.spacing_checked,
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
                "min_size_threshold": int(_require(params, "min_size_threshold", int)),
                "intensity_weight": float(_require(params, "intensity_weight", float)),
                "max_seed_centroid_dist": float(
                    _require(params, "max_seed_centroid_dist", float)
                ),
                "min_path_intensity_ratio": float(
                    _require(params, "min_path_intensity_ratio", float)
                ),
                "min_local_intensity_difference": float(
                    _require(params, "min_local_intensity_difference", float)
                ),
                "local_analysis_radius": int(
                    _require(params, "local_analysis_radius", int)
                ),
                "memmap_dir": temp_chunk_dir,
                "memmap_voxel_threshold": int(
                    params.get("memmap_voxel_threshold", 25_000_000)
                )
            }

            final_separated_cells = separate_multi_soma_cells(
                trimmed_labels_memmap, image_stack, cell_bodies_ref,
                self.spacing_checked, **separation_params
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
                spacing=self.spacing_checked,
                temp_dir=self.temp_dir,
                calculate_distances=_require(params, "calculate_distances", bool),
                calculate_skeletons=_require(params, "calculate_skeletons", bool),
                calculate_solidity=_require(params, "calculate_solidity", bool),
                skeleton_export_path=skel_dat_path,
                fcs_export_path=fcs_path,
                return_detailed=True,
                prune_spurs_le_um=_require(params, "prune_spurs_le_um", float)
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
                    self._add_neighbor_lines_dispatch(viewer, points_df)
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

    def _add_neighbor_lines_dispatch(self, viewer, points_df):
        """
        Draw nearest-neighbour connection lines, whichever schema produced them.

        Feature calculation stays rank-split (see `calculate_features`), and the
        two implementations name their bridge coordinates differently: 3D emits
        `mask1_z/y/x` and `mask2_z/y/x`, 2D emits `point_on_self_y/x` and
        `point_on_neighbor_y/x`. Rather than assume, the columns present decide
        which pair to read -- so this keeps working if either implementation is
        changed independently, and says so if neither schema is recognised.

        Viewer-only: nothing here affects a saved artifact.
        """
        if points_df is None or points_df.empty:
            return

        cols = set(points_df.columns)
        if {"mask1_y", "mask1_x", "mask2_y", "mask2_x"} <= cols:
            axes = ["z", "y", "x"] if "mask1_z" in cols else ["y", "x"]
            a = [f"mask1_{k}" for k in axes]
            b = [f"mask2_{k}" for k in axes]
        elif {"point_on_self_y", "point_on_self_x"} <= cols:
            a = ["point_on_self_y", "point_on_self_x"]
            b = ["point_on_neighbor_y", "point_on_neighbor_x"]
        else:
            print(f"  [Warn] Connection lines: unrecognised point schema "
                  f"{sorted(cols)}; skipping.")
            return

        lines = []
        try:
            for _, row in points_df.iterrows():
                lines.append([[float(row[c]) for c in a],
                              [float(row[c]) for c in b]])
        except Exception as e:
            print(f"  [Warn] Failed to parse connection lines: {e}")
            return
        if not lines:
            return

        layer_name = f"Neighbor Connections_{self.mode_name}"
        # `_remove_layer_safely` rather than `viewer.layers.remove`, which the 3D
        # track used and which raises when the layer is absent.
        self._remove_layer_safely(viewer, layer_name)

        if self.ndim == 3:
            display_scale = (self.z_scale_factor, 1, 1)
        else:
            display_scale = tuple(self.spacing_checked)

        viewer.add_shapes(
            lines, shape_type='line', edge_color='red',
            edge_width=1 if self.ndim == 3 else 2,
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
                    self._add_neighbor_lines_dispatch(viewer, df)
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


# --------------------------------------------------------------------------
# Legacy aliases
# --------------------------------------------------------------------------
#: The former per-rank classes. `FluorescenceStrategy` now handles both, so both
#: names resolve to it and the mode-keyed registries keep working while they are
#: migrated to a single "fluorescence" entry.
Fluorescence2DStrategy = FluorescenceStrategy

