import os
import sys
import gc
import traceback
import shutil
import random
import colorsys
import tempfile
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import yaml  # type: ignore
import tifffile as tiff
from PyQt5.QtWidgets import QMessageBox  # type: ignore

from ..high_level_gui.processing_strategies import ProcessingStrategy, StepDefinition

# Attempt imports of specific 2D segmentation modules
try:
    from .initial_2d_segmentation import segment_cells_first_pass_raw_2d
    from .remove_artifacts_2d import apply_hull_trimming_2d
    from .soma_extraction_2d import extract_soma_masks_2d
    from .cell_splitting_2d import separate_multi_soma_cells_2d
    from .calculate_features_2d import analyze_segmentation_2d
    from .interaction_analysis_2d import calculate_interaction_metrics_2d
    from .calculate_features_2d import export_to_fcs # Ensure this is imported for FCS updates
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import 2D segmentation modules: {e}")
    raise


class Ramified2DStrategy(ProcessingStrategy):
    """
    Orchestrates the 2D segmentation workflow for Ramified Microglia.

    This strategy manages a 6-step pipeline (mirroring the 3D workflow):
    1. Raw Segmentation (Frangi/Sato 2D + Thresholding)
    2. Edge Trimming (Hull-based artifact removal)
    3. Soma Extraction (Core detection)
    4. Cell Separation (Splitting multi-soma objects)
    5. Feature Calculation (Morphometrics, Skeletonization)
    6. Interaction Analysis (Overlap with other channels)
    """

    def _get_mode_name(self) -> str:
        """Returns the unique identifier for this strategy."""
        return "ramified_2d"

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
            },
            {
                "method": "execute_interaction_analysis",
                "artifact": None
            }
        ]

    def get_checkpoint_files(self) -> Dict[str, str]:
        """
        Defines file paths for all intermediate and final outputs.
        Uses .dat memmaps for consistency with 3D module.

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
                self.processed_dir, f"cell_bodies_{p}.dat"
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
            "last_interaction_meta": os.path.join(
                self.processed_dir, "last_interaction_meta.yaml"
            )
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
        Step 1: Performs initial raw 2D segmentation.

        Args:
            viewer: Napari viewer instance (or None).
            image_stack: Input 2D image array.
            params: Dictionary of parameters from config.

        Returns:
            bool: True if successful, False otherwise.
        """
        if image_stack is None or image_stack.ndim != 2:
            print(f"Error: Input must be 2D. Got {getattr(image_stack, 'shape', None)}")
            return False
        
        print(f"Executing Step 1: Raw {self.mode_name} segmentation...")
        files = self.get_checkpoint_files()
        persistent_raw_dat_path = files.get("raw_segmentation")
        temp_raw_labels_dir = None

        try:
            # Parse parameters
            tubular_scales = params.get("tubular_scales", [0.8, 1.0, 1.5, 2.0])
            min_size_pixels = int(params.get("min_size", 100))
            smooth_sigma = float(params.get("smooth_sigma", 1.0))
            low_threshold_percentile = float(
                params.get("low_threshold_percentile", 98.0)
            )
            high_threshold_percentile = float(
                params.get("high_threshold_percentile", 100.0)
            )
            connect_max_gap_physical = float(
                params.get("connect_max_gap_physical", 1.0)
            )

            # Call logic
            result = segment_cells_first_pass_raw_2d(
                image=image_stack,
                spacing=self.spacing,
                tubular_scales=tubular_scales,
                smooth_sigma=smooth_sigma,
                connect_max_gap_physical=connect_max_gap_physical,
                min_size_pixels=min_size_pixels,
                low_threshold_percentile=low_threshold_percentile,
                high_threshold_percentile=high_threshold_percentile,
                temp_root_path=self.temp_dir
            )

            # Unpack results
            temp_dat_path, temp_raw_labels_dir, seg_threshold, _ = result

            if not temp_dat_path or not os.path.exists(temp_dat_path):
                raise RuntimeError("Raw 2D segmentation function failed.")

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
                del display_data
            return True

        except Exception as e:
            print(f"Error during execute_raw_segmentation_2d: {e}")
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
        Step 2: Removes artifacts near edges (Hull Trimming 2D).
        """
        print(f"Executing Step 2: Edge Trimming ({self.mode_name})...")
        files = self.get_checkpoint_files()
        raw_seg_path = files.get("raw_segmentation")
        trimmed_seg_path = files.get("trimmed_segmentation")
        edge_mask_path = files.get("edge_mask")

        if not os.path.exists(raw_seg_path):
            return False
        if 'original_volume_ref' not in self.intermediate_state:
            return False
        
        original_image = self.intermediate_state['original_volume_ref']
        temp_trimmed_dir = None
        temp_writable_dir = None
        hull_boundary_mask = None

        try:
            # 2D hull trimming often expects a path, so we use the persisted file
            temp_dat_path, temp_trimmed_dir, hull_boundary_mask = apply_hull_trimming_2d(
                raw_labels_path=raw_seg_path,
                original_image=original_image,
                spacing=self.spacing,
                segmentation_threshold=self.intermediate_state.get('segmentation_threshold', 0.0),
                hull_boundary_thickness_phys=float(params.get("hull_boundary_thickness_phys", 1.0)),
                edge_trim_distance_threshold=float(params.get("edge_trim_distance_threshold", 2.5)),
                brightness_cutoff_factor=float(params.get("brightness_cutoff_factor", 1.5)),
                min_size_pixels=int(params.get("min_size_pixels", 20)),
                smoothing_iterations=int(params.get("smoothing_iterations", 1)),
                heal_iterations=int(params.get("heal_iterations", 1))
            )

            if not temp_dat_path or not os.path.exists(temp_dat_path):
                raise RuntimeError("apply_hull_trimming_2d failed.")

            # Persist Trimmed Segmentation
            shutil.copyfile(temp_dat_path, trimmed_seg_path)

            # Persist Edge Mask
            if hull_boundary_mask is not None:
                edge_memmap = np.memmap(
                    edge_mask_path, dtype=bool, mode='w+', shape=self.image_shape
                )
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
            print(f"Error during trim_edges_2d: {e}")
            traceback.print_exc()
            return False
        finally:
            if temp_trimmed_dir and os.path.exists(temp_trimmed_dir):
                shutil.rmtree(temp_trimmed_dir, ignore_errors=True)
            gc.collect()

    def execute_soma_extraction(
        self, viewer, image_stack: Any, params: Dict
    ) -> bool:
        """
        Step 3: Extracts cell bodies (somas) from 2D segmentation.
        """
        print(f"Executing Step 3: Soma Extraction ({self.mode_name})...")
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
            
            # Retrieve original image
            original_image = self.intermediate_state.get('original_volume_ref', image_stack)
            if original_image is None or original_image.ndim != 2:
                print("Error: Invalid 2D reference image for soma extraction.")
                return False

            # Extract YX spacing
            spacing_yx = tuple(self.spacing[1:]) if len(self.spacing) == 3 else tuple(self.spacing)

            cell_bodies = extract_soma_masks_2d(
                segmentation_mask=trimmed_labels_memmap,
                intensity_image=original_image,
                spacing=spacing_yx,
                smallest_quantile=float(params.get("smallest_quantile", 0.05)),
                min_fragment_size=int(params.get("min_fragment_size", 20)),
                core_volume_target_factor_lower=float(params.get("core_volume_target_factor_lower", 0.1)),
                core_volume_target_factor_upper=float(params.get("core_volume_target_factor_upper", 10.0)),
                erosion_iterations=int(params.get("erosion_iterations", 0)),
                ratios_to_process=params.get("ratios_to_process", [0.3, 0.4, 0.5, 0.6]),
                intensity_percentiles_to_process=params.get("intensity_percentiles_to_process", [100, 95, 90, 85, 80]),
                min_physical_peak_separation=float(params.get("min_physical_peak_separation", 3.0)),
                max_allowed_core_aspect_ratio=float(params.get("max_allowed_core_aspect_ratio", 5.0)),
                ref_vol_percentile_lower=int(params.get("ref_vol_percentile_lower", 30)),
                ref_vol_percentile_upper=int(params.get("ref_vol_percentile_upper", 70)),
                ref_thickness_percentile_lower=int(params.get("ref_thickness_percentile_lower", 1)),
                absolute_min_thickness_um=float(params.get("absolute_min_thickness_um", 0.5)),
                absolute_max_thickness_um=float(params.get("absolute_max_thickness_um", 5.0))
            )

            # Persist
            cb_memmap = np.memmap(
                cell_bodies_path, dtype=np.int32, mode='w+', shape=self.image_shape
            )
            if cell_bodies is not None:
                cb_memmap[:] = cell_bodies[:]
            self._close_memmap(cb_memmap)

            if viewer is not None:
                cb_display = np.memmap(
                    cell_bodies_path, dtype=np.int32, mode='r',
                    shape=self.image_shape
                )
                self._add_layer_safely(viewer, cb_display, "Cell bodies")
            return True

        except Exception as e:
            print(f"Error during Soma Extraction 2D: {e}")
            traceback.print_exc()
            return False
        finally:
            self._close_memmap(trimmed_labels_memmap)
            gc.collect()

    def execute_cell_separation(
        self, viewer, image_stack: Any, params: Dict
    ) -> bool:
        """
        Step 4: Separates merged cells in 2D using soma seeds.
        """
        print(f"Executing Step 4: Cell Separation ({self.mode_name})...")
        files = self.get_checkpoint_files()
        trimmed_seg_path = files["trimmed_segmentation"]
        cell_bodies_path = files["cell_bodies"]
        final_seg_path = files["final_segmentation"]

        if not os.path.exists(trimmed_seg_path) or not os.path.exists(cell_bodies_path):
            return False

        trimmed_labels_memmap = None
        cell_bodies_ref = None

        try:
            trimmed_labels_memmap = np.memmap(
                trimmed_seg_path, dtype=np.int32, mode='r', shape=self.image_shape
            )
            cell_bodies_ref = np.memmap(
                cell_bodies_path, dtype=np.int32, mode='r', shape=self.image_shape
            )
            
            original_image = self.intermediate_state.get('original_volume_ref', image_stack)
            spacing_yx = tuple(self.spacing[1:]) if len(self.spacing) == 3 else tuple(self.spacing)

            final_separated_cells = separate_multi_soma_cells_2d(
                segmentation_mask=trimmed_labels_memmap,
                intensity_volume=original_image,
                soma_mask=cell_bodies_ref,
                spacing=spacing_yx,
                memmap_dir=self.temp_dir,
                min_size_threshold=int(params.get("min_size_threshold", 100)),
                max_seed_centroid_dist=float(params.get("max_seed_centroid_dist", 20.0)),
                min_path_intensity_ratio=float(params.get("min_path_intensity_ratio", 0.8)),
                min_local_intensity_difference=float(params.get("min_local_intensity_difference", 0.05)),
                local_analysis_radius=float(params.get("local_analysis_radius", 2.0)),
                max_hole_size=int(params.get("max_hole_size", 0))
            )

            # Persist
            final_memmap = np.memmap(
                final_seg_path, dtype=np.int32, mode='w+', shape=self.image_shape
            )
            if final_separated_cells is not None:
                final_memmap[:] = final_separated_cells[:]
            self._close_memmap(final_memmap)

            if viewer is not None:
                final_display = np.memmap(
                    final_seg_path, dtype=np.int32, mode='r',
                    shape=self.image_shape
                )
                self._add_layer_safely(viewer, final_display, "Final segmentation")
            return True

        except Exception as e:
            print(f"Error during Cell Separation 2D: {e}")
            traceback.print_exc()
            return False
        finally:
            self._close_memmap(trimmed_labels_memmap)
            self._close_memmap(cell_bodies_ref)
            gc.collect()

    def execute_calculate_features(
        self, viewer, image_stack: Any, params: Dict
    ) -> bool:
        """
        Step 5: Calculates comprehensive morphometrics and skeletonizes 2D cells.
        
        This method acts as the high-level orchestrator for the feature 
        extraction module. It handles:
        1. Memory-mapped I/O for the final segmentation masks.
        2. Coordination of physical paths for skeletons and FCS files.
        3. Invocation of the 2D analysis engine with full precision.
        4. Persistence of tabular data (CSV) and spatial data (memmaps).
        5. Napari visualization of the resulting skeletons and connections.

        Args:
            viewer: Napari viewer instance for visualization.
            image_stack: The raw input 2D image (not strictly needed here if 
                         using intermediate state, but kept for signature parity).
            params: Dictionary containing user-defined parameters:
                - calculate_distances: bool, whether to run N x N distance matrix.
                - calculate_skeletons: bool, whether to run topological analysis.
                - prune_spurs_le_um: float, length threshold for cleaning skeletons.

        Returns:
            bool: True if the analysis completed and results were saved, False otherwise.
        """
        print(f"\n--- Executing Step 5: Feature Calculation ({self.mode_name}) ---")
        
        # 1. PATH INITIALIZATION
        # Retrieve the central checkpoint registry for file management
        files = self.get_checkpoint_files()
        final_seg_path = files.get("final_segmentation")
        metrics_csv_path = files.get("metrics_df")
        
        # Define specific output paths for the new persistent artifacts
        # We ensure these files land in the correct processed directory
        skel_dat_path = files.get("skeleton_array")
        fcs_out_path = os.path.join(self.processed_dir, f"metrics_{self.mode_name}.fcs")
        dist_matrix_path = files.get("distances_matrix")
        
        # Validation: Ensure the previous segmentation step actually exists
        if not final_seg_path or not os.path.exists(final_seg_path):
            print(f"  [Error] Final segmentation file not found at: {final_seg_path}")
            return False

        final_seg_memmap = None
        try:
            # 2. DATA PREPARATION
            # Open the segmentation as a read-only memmap to keep RAM usage low
            print(f"  [Features] Opening segmentation memmap: {os.path.basename(final_seg_path)}")
            final_seg_memmap = np.memmap(
                final_seg_path, dtype=np.int32, mode='r', shape=self.image_shape
            )
            
            # Extract YX spacing (Handling cases where 3D spacing might be stored)
            spacing_yx = tuple(self.spacing[1:]) if len(self.spacing) == 3 else tuple(self.spacing)
            print(f"  [Features] Using physical spacing: {spacing_yx} um/px")

            # Retrieve the normalized intensity image from the pipeline state
            # This is essential for calculating Mean/Max intensity metrics
            intensity_ref = self.intermediate_state.get('original_volume_ref', image_stack)
            if intensity_ref is None:
                print("  [Warn] Normalized intensity image not found in state. Intensity metrics will be skipped.")

            # 3. CORE ANALYSIS EXECUTION
            # We call the module-level analysis function with all optimization flags enabled
            print(f"  [Features] Starting analysis engine for {np.max(final_seg_memmap)} objects...")
            metrics_df, detailed_outputs = analyze_segmentation_2d(
                segmented_array=final_seg_memmap,
                intensity_image=intensity_ref,
                spacing_yx=spacing_yx,
                temp_dir=self.temp_dir,
                calculate_distances=params.get("calculate_distances", True),
                calculate_skeletons=params.get("calculate_skeletons", True),
                skeleton_export_path=skel_dat_path,  # Write skeletons to disk to prevent RAM crash
                fcs_export_path=fcs_out_path,       # Triggers the FCS writer logic
                return_detailed=True,               # Required to populate the GUI layers
                prune_spurs_le_um=params.get("prune_spurs_le_um", 0)
            )

            # 4. TABULAR DATA PERSISTENCE
            if metrics_df is not None and not metrics_df.empty:
                metrics_df.to_csv(metrics_csv_path, index=False)

            # 5. DETAILED DATA PERSISTENCE (Coordinate Persistence)
            if detailed_outputs:
                # Retrieve the red lines table
                nn_points = detailed_outputs.get('all_pairs_points')
                if nn_points is not None and not nn_points.empty:
                    pts_csv_path = files.get("points_matrix")
                    print(f"  [Features] Saving nearest neighbor lines to: {os.path.basename(pts_csv_path)}")
                    nn_points.to_csv(pts_csv_path, index=False)

            # 6. VISUALIZATION (NAPARI)
            if viewer is not None:
                # Load the skeleton back from the persistent disk file we just created
                # We use 'r' mode to ensure Napari only reads what it needs to display
                if skel_dat_path and os.path.exists(skel_dat_path):
                    skel_display = np.memmap(
                        skel_dat_path, dtype=np.int32, mode='r', shape=self.image_shape
                    )
                    self._add_layer_safely(
                        viewer, skel_display, "Skeletons", layer_type='labels'
                    )
                
                # If distances were calculated, visualize the nearest neighbor connections
                # We retrieve the filtered points from the detailed output dictionary
                nn_points = detailed_outputs.get('all_pairs_points')
                if nn_points is not None and not nn_points.empty:
                    print(f"  [Features] Plotting {len(nn_points)} neighbor connections in viewer.")
                    self._add_neighbor_lines_2d(viewer, nn_points)

            print(f"--- Feature Calculation for {self.mode_name} COMPLETE ---")
            return True

        except Exception as e:
            print(f"  [Critical Error] Feature calculation failed: {str(e)}")
            traceback.print_exc()
            return False
            
        finally:
            # Explicitly close the segmentation memmap to release the file lock
            if final_seg_memmap is not None:
                self._close_memmap(final_seg_memmap)
            
            # Force garbage collection to free up memory before the next strategy step
            gc.collect()

    def execute_interaction_analysis(
        self, viewer, image_stack: Any, params: Dict
    ) -> bool:
        """
        Step 6: Analyses spatial overlap with a secondary 2D channel (e.g. Plaques/Vessels).
        """
        print("\n--- Executing Step 6: Multi-Channel Interaction (2D) ---")
        
        target_root_dir = params.get("target_channel_folder")
        if not target_root_dir or not os.path.isdir(target_root_dir):
            print("  [Error] Invalid reference directory selected.")
            return False

        # 1. LOCATE REFERENCE DATA
        # Identify the sample name (the folder containing the images)
        sample_folder_name = os.path.basename(os.path.dirname(self.processed_dir))
        ref_sample_dir = os.path.join(target_root_dir, sample_folder_name)
        
        if not os.path.exists(ref_sample_dir):
            print(f"  [Error] Reference sample '{sample_folder_name}' not found in target channel.")
            return False

        # Find the processed folder inside the reference sample
        ref_processed_dir = None
        for item in os.listdir(ref_sample_dir):
            if "_processed_" in item and os.path.isdir(os.path.join(ref_sample_dir, item)):
                ref_processed_dir = os.path.join(ref_sample_dir, item)
                break
        
        if not ref_processed_dir:
            print(f"  [Error] Reference sample exists, but has not been processed yet.")
            return False

        # Find the final segmentation mask in the reference channel
        ref_seg_path = None
        for f in os.listdir(ref_processed_dir):
            if f.startswith("final_segmentation") and f.endswith(".dat"):
                ref_seg_path = os.path.join(ref_processed_dir, f)
                break
        
        if not ref_seg_path:
            print(f"  [Error] final_segmentation.dat not found in {ref_processed_dir}")
            return False

        ref_name = os.path.basename(target_root_dir)
        print(f"  [Info] Found Reference Mask: {ref_seg_path}")

        # 2. RUN ANALYSIS
        final_seg_path = self.get_checkpoint_files()["final_segmentation"]
        # Extract YX spacing
        spacing_yx = tuple(self.spacing[1:]) if len(self.spacing) == 3 else tuple(self.spacing)

        primary_df, ref_df, intersection_path = calculate_interaction_metrics_2d(
            primary_mask_path=final_seg_path,
            reference_mask_path=ref_seg_path,
            output_dir=self.processed_dir,
            shape=self.image_shape,
            spacing_yx=spacing_yx,
            reference_name=ref_name,
            calculate_distance=params.get("calculate_distance", True),
            calculate_overlap=params.get("calculate_overlap", True)
        )

        # 3. MERGE RESULTS INTO MAIN TABLE
        if not primary_df.empty:
            metrics_path = self.get_checkpoint_files()["metrics_df"]
            if os.path.exists(metrics_path):
                main_df = pd.read_csv(metrics_path)
                # Remove previous interaction columns for this specific reference to allow re-runs
                cols_to_drop = [c for c in main_df.columns if c in primary_df.columns and c != 'label']
                if cols_to_drop:
                    main_df.drop(columns=cols_to_drop, inplace=True)
                
                merged_df = pd.merge(main_df, primary_df, on='label', how='left')
                merged_df.to_csv(metrics_path, index=False)
                print(f"  [Success] Integrated 2D interaction metrics into metrics_df.csv")

                # Update FCS file
                fcs_path = os.path.join(self.processed_dir, f"metrics_{self.mode_name}.fcs")
                export_to_fcs(merged_df, fcs_path)
            
            # Save the reference-specific coverage stats
            if not ref_df.empty:
                ref_csv = os.path.join(self.processed_dir, f"interaction_{ref_name}_coverage.csv")
                ref_df.to_csv(ref_csv, index=False)
        else:
            print("  [Warning] No spatial interactions detected.")

        # 4. PERSIST METADATA FOR VISUALIZATION
        meta_path = self.get_checkpoint_files()["last_interaction_meta"]
        try:
            with open(meta_path, 'w') as f:
                yaml.dump({
                    'ref_seg_path': ref_seg_path,
                    'ref_raw_path': os.path.join(ref_sample_dir, f"{sample_folder_name}.tif"),
                    'ref_name': ref_name,
                    'intersection_path': intersection_path
                }, f)
        except Exception as e:
            print(f"  [Warn] Could not save visualization metadata: {e}")

        return True

    # =========================================================================
    # VISUALIZATION & HELPERS
    # =========================================================================

    def _add_neighbor_lines_2d(self, viewer, points_df):
        """Standardized 2D Line Visualization for Nearest Neighbors."""
        if points_df is None or points_df.empty:
            return
        
        lines = []
        try:
            for _, row in points_df.iterrows():
                # Extract and explicitly structure as [Y, X] coordinate pairs
                p1 = [float(row['point_on_self_y']), float(row['point_on_self_x'])]
                p2 = [float(row['point_on_neighbor_y']), float(row['point_on_neighbor_x'])]
                lines.append([p1, p2])
        except Exception as e:
            print(f"  [Warn] Failed to parse connection lines: {e}")
            return

        if lines:
            layer_name = f"Neighbor Connections_{self.mode_name}"
            self._remove_layer_safely(viewer, layer_name)
            
            # Match the display scale of the current project spacing
            display_scale = (self.spacing[1], self.spacing[2]) if len(self.spacing) == 3 else self.spacing
            
            viewer.add_shapes(
                lines, 
                shape_type='line', 
                edge_color='red', 
                edge_width=2, # Increased for better visibility on whole-slide scans
                name=layer_name, 
                scale=display_scale
            )

    def load_checkpoint_data(self, viewer, checkpoint_step: int):
        """Loads 2D results into Napari."""
        if viewer is None:
            return
        files = self.get_checkpoint_files()
        print(f"Loading 2D checkpoint data up to step {checkpoint_step}...")

        # Clean layers
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
                # Handle both .dat and .npy for backward compatibility during dev
                try:
                    if path.endswith('.npy'):
                        data = np.load(path)
                    else:
                        data = np.memmap(path, dtype=dtype, mode='r', shape=self.image_shape)
                    self._add_layer_safely(viewer, data, layer_name, **kwargs)
                except Exception as e:
                    print(f"Error loading {path}: {e}")

        if checkpoint_step >= 1:
            load_and_add("raw_segmentation", "Raw Intermediate Segmentation")
        if checkpoint_step >= 2:
            load_and_add("trimmed_segmentation", "Trimmed Intermediate Segmentation")
            load_and_add("edge_mask", "Edge Mask", dtype=bool, layer_type='image',
                         colormap='gray', blending='additive')
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
                    self._add_neighbor_lines_2d(viewer, df)
                except Exception:
                    pass
        
        # Step 6 Viz (Interaction Analysis)
        if checkpoint_step >= 6:
            meta_path = files.get("last_interaction_meta")
            if meta_path and os.path.exists(meta_path):
                try:
                    with open(meta_path, 'r') as f:
                        meta = yaml.safe_load(f)
                    
                    ref_path = meta.get('ref_seg_path')
                    ref_raw = meta.get('ref_raw_path')
                    ref_name = meta.get('ref_name', 'Ref')
                    inter_path = meta.get('intersection_path')
                    
                    # 2D Scaling Logic (Y, X)
                    display_scale = (self.spacing[1], self.spacing[2]) if len(self.spacing) == 3 else self.spacing

                    # A. Load Reference Mask
                    if ref_path and os.path.exists(ref_path):
                        ref_data = np.memmap(ref_path, dtype=np.int32, mode='r', shape=self.image_shape)
                        
                        # Apply distinct random colors for the reference objects
                        unique_lbls = np.unique(ref_data)
                        unique_lbls = unique_lbls[unique_lbls > 0]
                        cmap = {lbl: (*colorsys.hsv_to_rgb(random.random(), 0.6, 0.9), 1.0) for lbl in unique_lbls}
                        
                        l = viewer.add_labels(ref_data, name=f"Ref: {ref_name}", scale=display_scale)
                        l.color = cmap
                    
                    # B. Load Intersection Mask (Where the overlap happens)
                    if inter_path and os.path.exists(inter_path):
                        int_data = np.memmap(inter_path, dtype=np.int32, mode='r', shape=self.image_shape)
                        viewer.add_labels(int_data, name=f"Overlap ({ref_name})", scale=display_scale, opacity=0.7)

                    # C. Load Reference Raw Intensity (for context)
                    if ref_raw and os.path.exists(ref_raw):
                        try:
                            ref_img = tiff.imread(ref_raw)
                            viewer.add_image(ref_img, name=f"Ref Intensity ({ref_name})", 
                                             colormap='magenta', blending='additive', scale=display_scale)
                        except Exception: pass

                except Exception as e:
                    print(f"  [Error] Failed to load 2D interaction visualization: {e}")

    def cleanup_step_artifacts(self, viewer, step_number: int):
        """Cleans artifacts for 2D steps."""
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
            self._remove_file_safely(files.get("distances_matrix"))
            self._remove_file_safely(files.get("points_matrix"))
            self._remove_file_safely(files.get("branch_data"))