# --- START OF FILE utils/ramified_module_2d/ramified_segmenter_2d.py ---

import numpy as np
from scipy import ndimage
from tqdm import tqdm
from shutil import rmtree
import gc
from functools import partial
from skimage.measure import regionprops, label as skimage_label # type: ignore
from skimage.segmentation import watershed # type: ignore
from skimage.graph import route_through_array # type: ignore
from skimage.morphology import binary_dilation, disk # type: ignore # Use disk for 2D
import math
from sklearn.decomposition import PCA # type: ignore
from skimage.feature import peak_local_max # type: ignore
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from skimage.segmentation import relabel_sequential # type: ignore
import traceback

seed = 42
np.random.seed(seed)         # For NumPy


# Helper function for physical size based min_distance (2D version)
def get_min_distance_pixels_2d(spacing, physical_distance):
    """Calculates minimum distance in pixels for peak_local_max based on physical distance (2D)."""
    # Use the minimum spacing in Y or X
    min_spacing_yx = min(spacing) if spacing else 1.0
    if min_spacing_yx <= 1e-6: return 3 # Avoid division by zero, return small default
    pixels = int(round(physical_distance / min_spacing_yx))
    return max(3, pixels) # Ensure minimum separation of at least 3 pixels


# --- extract_soma_masks_2d ---
def extract_soma_masks_2d(
    segmentation_mask: np.ndarray, # 2D mask
    spacing: Optional[Tuple[float, float]], # 2D spacing (y, x)
    smallest_quantile: int = 25,
    min_fragment_size: int = 30, # Now interpreted as pixels
    core_area_target_factor_lower: float = 0.4, # Renamed from volume
    core_area_target_factor_upper: float = 10 # Renamed from volume
) -> np.ndarray:
    """
    Generates candidate seed mask using hybrid approach + mini-watershed (2D).
    Aggregates results: keeps all distinct splits and the smallest valid version
    of persistent seeds. Fallbacks fill remaining gaps. Adapted for 2D.
    Uses area and 2D shape metrics instead of volume/thickness.
    """
    print("--- Starting 2D Soma Extraction (Smallest Persistent + Splits Aggregation) ---")

    # --- Handle input ratios ---
    ratios_to_process = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6] # Keep same ratios for core detection
    print(f"Processing large cells with ratios: {ratios_to_process}")

    # --- Internal Parameters & Setup (2D Adapted) ---
    min_physical_peak_separation = 7.0 # Keep physical distance separation
    max_allowed_core_aspect_ratio = 10.0 # Keep aspect ratio limit
    min_seed_fragment_area = max(10, min_fragment_size) # Use area, smaller default min?
    ref_area_percentile_lower, ref_area_percentile_upper = 30, 70 # Use area percentiles
    # Thickness filter replaced by aspect ratio from regionprops
    absolute_min_minor_axis_um = 1.0 # Minimum feature size in um

    # Spacing setup (2D)
    if spacing is None: spacing = (1.0, 1.0); print("Warn: No 2D spacing, assume isotropic.")
    else:
        try: spacing = tuple(float(s) for s in spacing); assert len(spacing) == 2
        except: print(f"Warn: Invalid 2D spacing {spacing}. Using default."); spacing = (1.0, 1.0)
    print(f"Using 2D spacing (y,x): {spacing}")

    # ==============================================================
    # --- 1. Perform Setup Calculations (2D Adapted) ---
    # ==============================================================
    print("Calculating initial object properties (2D)...")
    unique_labels_all, counts = np.unique(segmentation_mask[segmentation_mask > 0], return_counts=True)
    if len(unique_labels_all) == 0: return np.zeros_like(segmentation_mask, dtype=np.int32)
    initial_areas = dict(zip(unique_labels_all, counts)) # Use area

    print("Finding object bounding boxes and validating labels (2D)...")
    label_to_slice: Dict[int, Tuple[slice, ...]] = {}
    valid_unique_labels_list: List[int] = []
    try:
        object_slices = ndimage.find_objects(segmentation_mask) # Works for 2D
        if object_slices is not None:
            num_slices_found = len(object_slices)
            for label in unique_labels_all:
                idx = label - 1
                if 0 <= idx < num_slices_found and object_slices[idx] is not None:
                    s = object_slices[idx]
                    # Check 2D slice validity
                    if len(s) == 2 and all(si.start < si.stop for si in s):
                        label_to_slice[label] = s; valid_unique_labels_list.append(label)
        if not valid_unique_labels_list: print("Error: No valid bounding boxes."); return np.zeros_like(segmentation_mask, dtype=np.int32)
        unique_labels = np.array(valid_unique_labels_list)
    except Exception as e: print(f"Error finding boxes: {e}"); return np.zeros_like(segmentation_mask, dtype=np.int32)

    areas = {label: initial_areas[label] for label in unique_labels}; all_areas_list = list(areas.values()) # Use area
    if not all_areas_list: print("Error: No valid areas."); return np.zeros_like(segmentation_mask, dtype=np.int32)

    min_samples_for_median = 5
    # Heuristic Target Area & Filter Ranges (2D)
    smallest_thresh_area = np.percentile(all_areas_list, smallest_quantile) if len(all_areas_list) > 1 else (all_areas_list[0] + 1 if all_areas_list else 1)
    smallest_object_labels_set = {label for label, area in areas.items() if area <= smallest_thresh_area}
    target_soma_area = np.median(all_areas_list) if len(all_areas_list) < min_samples_for_median*2 else np.median([areas[l] for l in smallest_object_labels_set if l in areas] or all_areas_list)
    target_soma_area = max(target_soma_area, 1.0)
    min_accepted_core_area = max(min_seed_fragment_area, target_soma_area * core_area_target_factor_lower) # Use area
    max_accepted_core_area = target_soma_area * core_area_target_factor_upper # Use area
    print(f"Core Area filter range: [{min_accepted_core_area:.2f} - {max_accepted_core_area:.2f}] pixels (Abs min: {min_seed_fragment_area})")

    # Reference Minor Axis Calculation (Replaces Thickness)
    area_thresh_lower, area_thresh_upper = (min(all_areas_list)-1, max(all_areas_list)+1) if len(all_areas_list) <= 1 else (np.percentile(all_areas_list, ref_area_percentile_lower), np.percentile(all_areas_list, ref_area_percentile_upper))
    reference_object_labels = {label for label in unique_labels if areas[label] > area_thresh_lower and areas[label] <= area_thresh_upper}
    reference_object_labels_for_ref = list(unique_labels) if len(reference_object_labels) < min_samples_for_median else list(reference_object_labels)
    print(f"Calculating reference minor axis from {len(reference_object_labels_for_ref)} objects...")
    minor_axes_reference_objs_px = [] # Store in pixels initially
    avg_spacing = np.mean(spacing)
    for label_ref in tqdm(reference_object_labels_for_ref, desc="Calc Ref Minor Axis", disable=len(reference_object_labels_for_ref) < 10):
        bbox_slice = label_to_slice.get(label_ref)
        if bbox_slice is None: continue
        sub_segmentation, object_mask_sub, props = None, None, None
        try:
            sub_segmentation = segmentation_mask[bbox_slice]
            object_mask_sub = (sub_segmentation == label_ref)
            if object_mask_sub.size == 0 or not np.any(object_mask_sub): continue
            # Use regionprops for 2D shape analysis
            props = regionprops(object_mask_sub.astype(np.uint8))
            if props:
                 minor_axis_px = props[0].minor_axis_length
                 if minor_axis_px > 0: minor_axes_reference_objs_px.append(minor_axis_px)
        except Exception as e: print(f"Warn: Ref Minor Axis Err L{label_ref}: {e}"); continue
        finally:
            del sub_segmentation, object_mask_sub, props

    # Final Minor Axis Range (Convert pixels to physical units for comparison)
    min_accepted_minor_axis_um = absolute_min_minor_axis_um
    if minor_axes_reference_objs_px:
        minor_axes_reference_objs_um = [axis * avg_spacing for axis in minor_axes_reference_objs_px]
        if len(minor_axes_reference_objs_um) < min_samples_for_median:
            median_ref_minor_um = np.median(minor_axes_reference_objs_um)
            range_estimate_um = 1.0 # Smaller default range in um?
            min_accepted_minor_axis_um = max(absolute_min_minor_axis_um, median_ref_minor_um - range_estimate_um)
        else:
            # Using a low percentile might be too strict, maybe use 10th or 15th?
            ref_minor_axis_perc_lower = 15
            q_low_val_um = np.percentile(minor_axes_reference_objs_um, ref_minor_axis_perc_lower)
            min_accepted_minor_axis_um = max(absolute_min_minor_axis_um, q_low_val_um)

    print(f"Final Core Minor Axis filter minimum: {min_accepted_minor_axis_um:.2f} um")
    min_peak_sep_pixels = get_min_distance_pixels_2d(spacing, min_physical_peak_separation) # Call 2D helper

    # ==============================================================
    # --- 2. Process Small Cells (Directly to Final Mask - 2D) ---
    # ==============================================================
    print("\nProcessing objects (2D)...")
    final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32) # 2D output mask
    next_final_label = 1
    added_small_labels: Set[int] = set()

    small_cell_labels_to_process = [l for l in unique_labels if l in smallest_object_labels_set]
    print(f"Processing {len(small_cell_labels_to_process)} small objects (2D)...")
    for label in tqdm(small_cell_labels_to_process, desc="Small Cells", disable=len(small_cell_labels_to_process) < 10):
        bbox_slice = label_to_slice.get(label)
        if bbox_slice is None: continue
        slice_obj = bbox_slice
        sub_segmentation, object_mask_sub, obj_coords = None, None, None
        try:
            sub_segmentation = segmentation_mask[slice_obj]
            object_mask_sub = (sub_segmentation == label)
            obj_area = np.sum(object_mask_sub)
            # Use area check
            if object_mask_sub.size > 0 and obj_area >= min_seed_fragment_area:
                offset = (bbox_slice[0].start, bbox_slice[1].start) # 2D offset
                obj_coords = np.argwhere(object_mask_sub)
                # 2D global coords
                global_coords_y = obj_coords[:, 0] + offset[0]; global_coords_x = obj_coords[:, 1] + offset[1]
                # 2D boundary checks
                valid_indices = ((global_coords_y >= 0) & (global_coords_y < final_seed_mask.shape[0]) &
                                 (global_coords_x >= 0) & (global_coords_x < final_seed_mask.shape[1]))
                if isinstance(valid_indices, np.ndarray) and valid_indices.ndim == 1 and len(valid_indices) == len(global_coords_y):
                     coords_to_write = (global_coords_y[valid_indices], global_coords_x[valid_indices]) # 2D tuple
                     if np.all(final_seed_mask[coords_to_write] == 0):
                          final_seed_mask[coords_to_write] = next_final_label
                          next_final_label += 1; added_small_labels.add(label)
        except MemoryError: print(f"Warn: MemError small L{label}."); gc.collect()
        except IndexError as ie: print(f"Warn: IndexErr small L{label}: {ie}")
        except Exception as e: print(f"Warn: Error small L{label}: {e}")
        finally: del obj_coords, object_mask_sub, sub_segmentation

    print(f"Added {len(added_small_labels)} initial seeds from small cells.")
    gc.collect()

    # ==============================================================
    # --- 3. Generate All Candidate Seeds from Large Cells (2D) ---
    # ==============================================================
    large_cell_labels_to_process = [l for l in unique_labels if l not in added_small_labels]
    print(f"Generating candidates from {len(large_cell_labels_to_process)} large objects (2D)...")

    valid_candidates = [] # List of {'mask': mask, 'area': a, 'offset': offset}
    fallback_candidates = []

    for label in tqdm(large_cell_labels_to_process, desc="Large Cell Candidates"):
        bbox_slice = label_to_slice.get(label)
        if bbox_slice is None: continue

        for current_ratio in ratios_to_process:
            pad = 1
            y_min = max(0, bbox_slice[0].start - pad); x_min = max(0, bbox_slice[1].start - pad) # 2D slices
            y_max = min(segmentation_mask.shape[0], bbox_slice[0].stop + pad); x_max = min(segmentation_mask.shape[1], bbox_slice[1].stop + pad)
            if y_min >= y_max or x_min >= x_max: continue
            slice_obj = np.s_[y_min:y_max, x_min:x_max]; offset = (y_min, x_min) # 2D slice/offset

            passed_filters_list_this_ratio = []
            min_size_only_list_this_ratio = []
            sub_segmentation, object_mask_sub, dist_transform_obj = None, None, None

            try:
                try: sub_segmentation = segmentation_mask[slice_obj]
                except IndexError: continue
                object_mask_sub = (sub_segmentation == label)
                if not np.any(object_mask_sub): continue

                dist_transform_obj = ndimage.distance_transform_edt(object_mask_sub, sampling=spacing) # 2D EDT
                max_dist_in_obj = np.max(dist_transform_obj)
                if max_dist_in_obj <= 1e-9: continue

                core_thresh = max_dist_in_obj * current_ratio
                initial_core_region_mask_sub = (dist_transform_obj >= core_thresh) & object_mask_sub
                if not np.any(initial_core_region_mask_sub): continue

                labeled_cores, num_cores = ndimage.label(initial_core_region_mask_sub) # 2D label
                if num_cores == 0: continue

                # Process Core Components
                for core_label in range(1, num_cores + 1):
                    core_component_mask_sub = (labeled_cores == core_label)
                    if not np.any(core_component_mask_sub): continue

                    # Mini-Watershed -> candidate_fragment_masks (2D)
                    candidate_fragment_masks = []
                    dt_core, peaks, markers_core, ws_core = None, None, None, None
                    try:
                        dt_core = ndimage.distance_transform_edt(core_component_mask_sub, sampling=spacing) # 2D EDT
                        if np.max(dt_core) > 1e-9:
                            peaks = peak_local_max(dt_core, min_distance=min_peak_sep_pixels, labels=core_component_mask_sub, num_peaks_per_label=0, exclude_border=False) # Works in 2D
                            if peaks.shape[0] > 1:
                                markers_core = np.zeros(dt_core.shape, dtype=np.int32); markers_core[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                                ws_core = watershed(-dt_core, markers_core, mask=core_component_mask_sub, watershed_line=False) # Works in 2D
                                unique_ws_labels = np.unique(ws_core)
                                for ws_label in unique_ws_labels:
                                     if ws_label == 0: continue
                                     fragment_mask = (ws_core == ws_label)
                                     if np.any(fragment_mask): candidate_fragment_masks.append(fragment_mask)
                            else: candidate_fragment_masks.append(core_component_mask_sub)
                        else: candidate_fragment_masks.append(core_component_mask_sub)
                    except Exception: candidate_fragment_masks.append(core_component_mask_sub) # Fallback
                    finally: del dt_core, peaks, markers_core, ws_core

                    # Filter Candidate Fragments (2D Adapted)
                    for i, fragment_mask in enumerate(candidate_fragment_masks):
                        fragment_passed_all = False; fragment_mask_copy = None
                        labeled_fragment, props_fragment_labeled = None, None
                        try:
                            fragment_area = np.sum(fragment_mask) # Use area
                            if fragment_area < min_seed_fragment_area: continue # 1. Min Area Check

                            fragment_mask_copy = fragment_mask.copy()
                            temp_min_size_info = {'mask': fragment_mask_copy, 'area': fragment_area}

                            # 2. Minor Axis Check (replaces thickness)
                            passes_minor_axis = False
                            try:
                                props_fragment = regionprops(fragment_mask.astype(np.uint8))
                                if props_fragment:
                                    minor_axis_px = props_fragment[0].minor_axis_length
                                    minor_axis_um = minor_axis_px * avg_spacing # Convert to physical
                                    if minor_axis_um >= min_accepted_minor_axis_um:
                                        passes_minor_axis = True
                            except Exception: passes_minor_axis = True # Assume pass on error

                            if not passes_minor_axis: del fragment_mask_copy; temp_min_size_info = None; continue

                            # 3. Area Range Check
                            passes_area = (fragment_area >= min_accepted_core_area and fragment_area <= max_accepted_core_area)
                            if not passes_area: del fragment_mask_copy; temp_min_size_info = None; continue

                            # 4. Aspect Ratio Check (use regionprops for 2D)
                            passes_aspect = True; aspect_ratio = -1
                            try:
                                if props_fragment: # Use props calculated earlier
                                    maj_ax = props_fragment[0].major_axis_length
                                    min_ax = props_fragment[0].minor_axis_length
                                    if min_ax > 1e-7: aspect_ratio = maj_ax / min_ax
                                    if aspect_ratio != -1 and aspect_ratio > max_allowed_core_aspect_ratio: passes_aspect = False
                            except Exception: passes_aspect = True # Assume pass

                            if not passes_aspect: del fragment_mask_copy; temp_min_size_info = None; continue

                            # --- Passed ALL Filters ---
                            fragment_passed_all = True
                            passed_filters_list_this_ratio.append({'mask': fragment_mask_copy, 'area': fragment_area})
                            temp_min_size_info = None

                        except MemoryError: print(f"Warn: Filter MemError L{label} R{current_ratio} F{i}."); gc.collect(); temp_min_size_info=None
                        finally:
                             del labeled_fragment, props_fragment_labeled
                             if temp_min_size_info is not None and not fragment_passed_all:
                                 min_size_only_list_this_ratio.append(temp_min_size_info)
                             elif fragment_passed_all and 'fragment_mask_copy' in locals() and fragment_mask_copy is not None:
                                 del fragment_mask_copy

                # --- Decide what to add to global candidate lists ---
                if passed_filters_list_this_ratio:
                    for item in passed_filters_list_this_ratio:
                        valid_candidates.append({'mask': item['mask'], 'area': item['area'], 'offset': offset})
                elif min_size_only_list_this_ratio:
                    largest_fallback = max(min_size_only_list_this_ratio, key=lambda x: x['area'])
                    fallback_candidates.append({'mask': largest_fallback['mask'], 'area': largest_fallback['area'], 'offset': offset})

            except MemoryError: print(f"Warn: MemError L{label} R{current_ratio}."); gc.collect()
            except IndexError as ie: print(f"Warn: IndexError L{label} R{current_ratio}: {ie}.")
            except Exception as e: print(f"Warn: Error L{label} R{current_ratio}: {e}")
            finally:
                 del sub_segmentation, object_mask_sub, dist_transform_obj
                 for item in passed_filters_list_this_ratio:
                      if 'mask' in item: del item['mask']
                 for item in min_size_only_list_this_ratio:
                      if 'mask' in item: del item['mask']
                 del passed_filters_list_this_ratio, min_size_only_list_this_ratio
                 gc.collect()
        # --- End Ratio Loop ---
    # --- End Label Loop ---

    print(f"Generated {len(valid_candidates)} valid 2D candidates and {len(fallback_candidates)} fallback candidates.")

    # ==============================================================
    # --- 4. Process Valid Candidates (Smallest Area First) (2D) ---
    # ==============================================================
    print("Placing smallest valid 2D seeds first...")
    valid_candidates.sort(key=lambda x: x['area']) # Sort by area

    processed_count = 0
    for candidate in tqdm(valid_candidates, desc="Placing Valid Seeds"):
        mask_sub = candidate['mask']; offset = candidate['offset']
        if mask_sub is None or mask_sub.size == 0: continue
        try:
            coords_sub = np.argwhere(mask_sub)
            if coords_sub.size == 0: continue
            global_coords_y = coords_sub[:, 0] + offset[0]; global_coords_x = coords_sub[:, 1] + offset[1] # 2D global
            valid_indices = ((global_coords_y >= 0) & (global_coords_y < final_seed_mask.shape[0]) & (global_coords_x >= 0) & (global_coords_x < final_seed_mask.shape[1])) # 2D bounds
            if not np.any(valid_indices): continue
            coords_global = (global_coords_y[valid_indices], global_coords_x[valid_indices]) # 2D tuple
            if np.all(final_seed_mask[coords_global] == 0):
                final_seed_mask[coords_global] = next_final_label
                next_final_label += 1; processed_count += 1
        except Exception as e: print(f"Warn: Error processing valid 2D candidate: {e}.")
        finally:
             if 'mask' in candidate: del candidate['mask']

    print(f"Placed {processed_count} seeds from valid 2D candidates.")
    del valid_candidates; gc.collect()

    # ==============================================================
    # --- 5. Process Fallback Candidates (Fill Gaps) (2D) ---
    # ==============================================================
    print("Placing fallback 2D seeds in empty regions...")
    fallback_processed_count = 0
    for fallback in tqdm(fallback_candidates, desc="Placing Fallbacks"):
        mask_sub = fallback['mask']; offset = fallback['offset']
        if mask_sub is None or mask_sub.size == 0: continue
        try:
            coords_sub = np.argwhere(mask_sub)
            if coords_sub.size == 0: continue
            global_coords_y = coords_sub[:, 0] + offset[0]; global_coords_x = coords_sub[:, 1] + offset[1]
            valid_indices = ((global_coords_y >= 0) & (global_coords_y < final_seed_mask.shape[0]) & (global_coords_x >= 0) & (global_coords_x < final_seed_mask.shape[1]))
            if not np.any(valid_indices): continue
            coords_global = (global_coords_y[valid_indices], global_coords_x[valid_indices])
            if np.all(final_seed_mask[coords_global] == 0):
                final_seed_mask[coords_global] = next_final_label
                next_final_label += 1; fallback_processed_count += 1
        except Exception as e: print(f"Warn: Error processing 2D fallback: {e}.")
        finally:
             if 'mask' in fallback: del fallback['mask']

    print(f"Placed {fallback_processed_count} seeds from fallback 2D candidates.")
    del fallback_candidates; gc.collect()

    total_final_seeds = next_final_label - 1
    print(f"\nGenerated a total of {total_final_seeds} final aggregated 2D seeds.")
    print("--- Finished 2D Intermediate Seed Extraction ---")

    return final_seed_mask


# --- refine_seeds_pca_2d ---
def refine_seeds_pca_2d(
    intermediate_seed_mask: np.ndarray, # 2D mask
    spacing: Optional[Tuple[float, float]], # 2D spacing
    target_aspect_ratio: float = 1.1,
    projection_percentile_crop: int = 10,
    min_fragment_size: int = 30 # Pixels
) -> np.ndarray:
    """
    Refines preliminary 2D seeds using PCA to make them more compact. Adapted for 2D.
    """
    print("--- Starting 2D PCA Seed Refinement ---")
    if spacing is None: spacing = (1.0, 1.0); print("Warn: No 2D spacing, assume isotropic.")
    else: print(f"Using 2D spacing (y,x): {spacing}")
    print(f"Parameters: target_aspect_ratio={target_aspect_ratio}, projection_crop%={projection_percentile_crop}, min_area={min_fragment_size}")

    final_refined_mask = np.zeros_like(intermediate_seed_mask, dtype=np.int32) # 2D output
    next_final_label = 1; kept_refined_count = 0
    seed_labels = np.unique(intermediate_seed_mask); seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels) == 0: return final_refined_mask

    seed_slices = ndimage.find_objects(intermediate_seed_mask) # Works for 2D
    label_to_slice = {label: seed_slices[label-1] for label in seed_labels if label-1 < len(seed_slices) and seed_slices[label-1] is not None and len(seed_slices[label-1])==2}

    for label in tqdm(seed_labels, desc="Refining Seeds PCA (2D)"):
        bbox_slice = label_to_slice.get(label);
        if bbox_slice is None: continue
        offset = (bbox_slice[0].start, bbox_slice[1].start); slice_obj = bbox_slice # 2D offset/slice
        seed_mask_sub, coords_vox, coords_phys, pca = None, None, None, None; refined_mask_sub = None
        try:
            seed_mask_sub = (intermediate_seed_mask[slice_obj] == label); current_area = np.sum(seed_mask_sub)
            if seed_mask_sub.size == 0 or not np.any(seed_mask_sub) or current_area < min_fragment_size: continue

            coords_vox = np.argwhere(seed_mask_sub)
            if coords_vox.shape[0] <= 2: refined_mask_sub = seed_mask_sub # Need > 2 points for PCA
            else:
                coords_phys = coords_vox * np.array(spacing) # Apply 2D spacing
                pca = PCA(n_components=2); pca.fit(coords_phys); eigenvalues = pca.explained_variance_; eigenvectors = pca.components_ # 2 components
                sorted_indices = np.argsort(eigenvalues)[::-1]; eigenvalues = eigenvalues[sorted_indices]; eigenvectors = eigenvectors[sorted_indices]
                if eigenvalues[1] < 1e-9: eigenvalues[1] = 1e-9 # Avoid division by zero for aspect ratio
                aspect_ratio = eigenvalues[0] / eigenvalues[1] # 2D aspect ratio

                if aspect_ratio >= target_aspect_ratio:
                    coords_centered_phys = coords_phys - pca.mean_; proj1 = coords_centered_phys @ eigenvectors[0] # Project onto longest axis
                    min_p = np.percentile(proj1, projection_percentile_crop); max_p = np.percentile(proj1, 100 - projection_percentile_crop)
                    voxel_indices_to_keep = (proj1 >= min_p) & (proj1 <= max_p)
                    refined_mask_sub = np.zeros_like(seed_mask_sub, dtype=bool)
                    kept_coords_vox = coords_vox[voxel_indices_to_keep]
                    if kept_coords_vox.shape[0] > 0: refined_mask_sub[tuple(kept_coords_vox.T)] = True
                    refined_area = np.sum(refined_mask_sub)
                    if refined_area < min_fragment_size: refined_mask_sub = None # Discard if too small
                else: refined_mask_sub = seed_mask_sub # Keep original

            if refined_mask_sub is not None and np.any(refined_mask_sub):
                 final_area = np.sum(refined_mask_sub)
                 if final_area >= min_fragment_size:
                     refined_coords = np.argwhere(refined_mask_sub)
                     if refined_coords.size > 0:
                         # 2D global coords
                         global_coords_y=refined_coords[:,0]+offset[0]; global_coords_x=refined_coords[:,1]+offset[1]
                         # 2D bounds check
                         valid_indices=((global_coords_y>=0)&(global_coords_y<final_refined_mask.shape[0])& (global_coords_x>=0)&(global_coords_x<final_refined_mask.shape[1]))
                         # Write to 2D final mask
                         final_refined_mask[global_coords_y[valid_indices], global_coords_x[valid_indices]] = next_final_label
                         next_final_label += 1; kept_refined_count += 1
        except Exception as e: print(f"Warning: Error during 2D PCA refinement for seed label {label}: {e}")
        finally: # Cleanup vars
            del seed_mask_sub, coords_vox, coords_phys, pca, refined_mask_sub
            if 'coords_centered_phys' in locals(): del coords_centered_phys
            if 'proj1' in locals(): del proj1
            if 'kept_coords_vox' in locals(): del kept_coords_vox
            if 'refined_coords' in locals(): del refined_coords
    print(f"Kept {kept_refined_count} refined 2D seeds after PCA.")
    print("--- Finished 2D PCA Seed Refinement ---"); gc.collect(); return final_refined_mask


# --- separate_multi_soma_cells_2d ---
def separate_multi_soma_cells_2d(segmentation_mask, # 2D
                              intensity_image, # 2D
                              soma_mask, # 2D
                              spacing, # 2D
                              min_size_threshold=100, # Pixels
                              intensity_weight=0.0,
                              max_seed_centroid_dist=15.0,
                              min_path_intensity_ratio=0.6,
                              post_merge_min_interface_pixels=10 # Replaces thickness check
                             ):
    """
    Separates 2D cell segmentations containing multiple seeds. Adapted for 2D.
    Uses path intensity heuristic, post-watershed merging based on interface size.
    RELABELED OUTPUT V2: Ensures final mask labels are sequential 1..N.
    """
    print(f"--- Starting 2D Multi-Soma Separation (Path Heuristics + Interface Merge) ---")

    # --- Parameter & Spacing Checks (2D) ---
    print(f"Post-merge check: min_interface_pixels={post_merge_min_interface_pixels}")
    if spacing is None: spacing = (1.0, 1.0); print("Warn: No 2D spacing, assume isotropic.")
    else:
        try: spacing = tuple(float(s) for s in spacing); assert len(spacing) == 2
        except: print(f"Error: Invalid 2D spacing ({spacing}). Using default."); spacing = (1.0, 1.0)
    print(f"Using 2D spacing (y,x): {spacing}")
    print(f"Seed merging heuristics: max_dist={max_seed_centroid_dist}, min_path_intensity_ratio={min_path_intensity_ratio}")

    # --- Input Validation & Type Checks (2D) ---
    if segmentation_mask.ndim != 2 or intensity_image.ndim != 2 or soma_mask.ndim != 2:
        raise ValueError("All input masks and image must be 2D.")
    if not np.issubdtype(intensity_image.dtype, np.floating):
        intensity_image = intensity_image.astype(np.float32, copy=False)
    if not np.issubdtype(segmentation_mask.dtype, np.integer):
        segmentation_mask = segmentation_mask.astype(np.int32)
    if not np.issubdtype(soma_mask.dtype, np.integer):
        soma_mask = soma_mask.astype(np.int32)

    separated_mask = np.copy(segmentation_mask).astype(np.int32) # 2D copy

    # --- Mapping (2D) ---
    print("Mapping 2D seeds to original cell segments...")
    unique_initial_labels = np.unique(segmentation_mask); unique_initial_labels = unique_initial_labels[unique_initial_labels > 0]
    if unique_initial_labels.size == 0: return separated_mask
    cell_to_somas = {cell_label: set() for cell_label in unique_initial_labels}
    present_soma_labels = np.unique(soma_mask); present_soma_labels = present_soma_labels[present_soma_labels > 0]
    if present_soma_labels.size == 0: return separated_mask

    for soma_label in tqdm(present_soma_labels, desc="Mapping Seeds (2D)", disable=len(present_soma_labels) < 10):
        soma_loc_mask = (soma_mask == soma_label)
        cell_labels_under_soma = np.unique(segmentation_mask[soma_loc_mask])
        cell_labels_under_soma = cell_labels_under_soma[cell_labels_under_soma > 0]
        for cell_label in cell_labels_under_soma:
            if cell_label in cell_to_somas: cell_to_somas[cell_label].add(soma_label)

    # --- Identify candidates & Label Management (2D) ---
    multi_soma_cell_labels = [lbl for lbl, somas in cell_to_somas.items() if len(somas) > 1]
    print(f"Found {len(multi_soma_cell_labels)} initial 2D segments with multiple seeds...")
    max_orig_label = np.max(unique_initial_labels) if unique_initial_labels.size > 0 else 0
    max_soma_label = np.max(present_soma_labels) if present_soma_labels.size > 0 else 0
    next_label = max(max_orig_label, max_soma_label) + 1
    print(f"Tentative starting label for new 2D segments: {next_label}")

    # --- Separation Loop (2D) ---
    skipped_count = 0; processed_count = 0
    current_max_label_in_use = next_label - 1

    for cell_label in tqdm(multi_soma_cell_labels, desc="Separating Segments (2D)"):
        processed_count += 1
        cell_mask_sub, soma_sub, cell_soma_sub_mask = None, None, None
        seed_props, adj, intensity_sub, cost_array = None, None, None, None
        distance_transform, landscape, watershed_result = None, None, None
        temp_mask = None; watershed_markers = None; marker_id_reverse_map = None
        used_new_labels = set()

        try:
            # --- Get Sub-Image (2D) ---
            cell_mask = segmentation_mask == cell_label
            obj_slice = ndimage.find_objects(cell_mask)
            if not obj_slice or obj_slice[0] is None: continue
            bbox_slice = obj_slice[0]
            pad = 5 # Keep padding
            y_min=max(0,bbox_slice[0].start-pad); x_min=max(0,bbox_slice[1].start-pad)
            y_max=min(segmentation_mask.shape[0],bbox_slice[0].stop+pad); x_max=min(segmentation_mask.shape[1],bbox_slice[1].stop+pad)
            slice_obj=np.s_[y_min:y_max, x_min:x_max]; offset=(y_min,x_min) # 2D slice/offset

            cell_mask_sub = cell_mask[slice_obj]
            if cell_mask_sub.size == 0 or not np.any(cell_mask_sub): continue

            soma_sub = soma_mask[slice_obj]
            cell_soma_sub_mask = np.zeros_like(soma_sub, dtype=np.int32); cell_soma_sub_mask[cell_mask_sub] = soma_sub[cell_mask_sub]
            soma_labels_in_cell = np.unique(cell_soma_sub_mask); soma_labels_in_cell = soma_labels_in_cell[soma_labels_in_cell > 0]
            if len(soma_labels_in_cell) <= 1: continue

            # --- Seed Merging Heuristic (Pathfinding - 2D) ---
            intensity_sub = intensity_image[slice_obj] # Use intensity_image
            seed_props = regionprops(cell_soma_sub_mask, intensity_image=intensity_sub) # regionprops works on 2D
            seed_prop_dict = {prop.label: prop for prop in seed_props}

            num_seeds = len(soma_labels_in_cell); seed_indices = {lbl: i for i, lbl in enumerate(soma_labels_in_cell)}
            adj = np.zeros((num_seeds, num_seeds), dtype=bool); merge_candidates = False

            max_I_sub = np.max(intensity_sub[cell_mask_sub]) if np.any(cell_mask_sub) else 1.0
            cost_array = np.full(intensity_sub.shape, np.inf, dtype=np.float32); cost_array[cell_mask_sub] = (max_I_sub - intensity_sub[cell_mask_sub]) + 1e-6

            for i in range(num_seeds):
                for j in range(i + 1, num_seeds):
                    label1 = soma_labels_in_cell[i]; label2 = soma_labels_in_cell[j]
                    prop1 = seed_prop_dict.get(label1); prop2 = seed_prop_dict.get(label2)
                    if prop1 is None or prop2 is None: continue
                    # Use 2D centroids and spacing
                    cent1_phys = np.array(prop1.centroid) * np.array(spacing); cent2_phys = np.array(prop2.centroid) * np.array(spacing)
                    dist = np.linalg.norm(cent1_phys - cent2_phys)
                    if dist > max_seed_centroid_dist: continue

                    coords1_bounded = tuple(np.clip(int(round(c)), 0, s-1) for c, s in zip(prop1.centroid, intensity_sub.shape))
                    coords2_bounded = tuple(np.clip(int(round(c)), 0, s-1) for c, s in zip(prop2.centroid, intensity_sub.shape))
                    median_intensity_on_path = 0
                    try: # route_through_array works in N-D
                        indices, weight = route_through_array(cost_array, coords1_bounded, coords2_bounded, fully_connected=True)
                        # Validate 2D indices (shape should be (2, N))
                        if isinstance(indices, np.ndarray) and indices.ndim == 2 and indices.shape[0] == 2 and indices.shape[1] > 0:
                            path_intensities = intensity_sub[tuple(indices)]
                            if path_intensities.size > 0: median_intensity_on_path = np.median(path_intensities)
                        # else: print("Warn: Invalid path indices format.") # Optional debug
                    except ValueError: median_intensity_on_path = 0

                    max_intensity_in_seeds = max(prop1.max_intensity, prop2.max_intensity) if prop1.max_intensity > 0 and prop2.max_intensity > 0 else 1.0
                    intensity_ratio = median_intensity_on_path / (max_intensity_in_seeds if max_intensity_in_seeds > 1e-6 else 1.0)
                    if intensity_ratio < min_path_intensity_ratio: continue

                    adj[seed_indices[label1], seed_indices[label2]] = adj[seed_indices[label2], seed_indices[label1]] = True; merge_candidates = True

            # --- Make Decision & Prepare Watershed Markers (2D) ---
            watershed_markers = np.zeros_like(cell_mask_sub, dtype=np.int32); marker_id_reverse_map = {}
            largest_soma_label = -1; max_soma_size = -1; num_markers_final = 0; current_marker_id = 1

            if merge_candidates:
                n_comp, comp_labels = ndimage.label(adj) # Works on adj matrix
                if n_comp == 1: print(f"    Cell {cell_label}: All {num_seeds} seeds connected. Skip split."); skipped_count += 1; continue
                print(f"    Cell {cell_label}: Found {n_comp} groups. Merging markers...")
                for group_idx in range(1, n_comp + 1):
                    group_marker_id = current_marker_id; seed_indices_in_group = np.where(comp_labels == group_idx)[0]
                    group_mask = np.zeros_like(cell_mask_sub, dtype=bool); original_labels_in_group = set()
                    for seed_idx in seed_indices_in_group:
                         original_label = soma_labels_in_cell[seed_idx]; original_labels_in_group.add(original_label)
                         group_mask |= (cell_soma_sub_mask == original_label)
                         prop = seed_prop_dict.get(original_label);
                         if prop and prop.area > max_soma_size: max_soma_size = prop.area; largest_soma_label = original_label
                    watershed_markers[group_mask] = group_marker_id; marker_id_reverse_map[group_marker_id] = tuple(sorted(list(original_labels_in_group))); current_marker_id += 1
                num_markers_final = n_comp
            else: # Individual seeds
                 for soma_label in soma_labels_in_cell:
                     watershed_markers[cell_soma_sub_mask == soma_label] = current_marker_id; marker_id_reverse_map[current_marker_id] = (soma_label,)
                     prop = seed_prop_dict.get(soma_label);
                     if prop and prop.area > max_soma_size: max_soma_size = prop.area; largest_soma_label = soma_label
                     current_marker_id += 1
                 num_markers_final = num_seeds

            # --- Landscape and Watershed (2D) ---
            distance_transform = ndimage.distance_transform_edt(cell_mask_sub, sampling=spacing) # 2D EDT
            landscape = -distance_transform
            if intensity_weight > 1e-6:
                 intensity_values_in_cell = intensity_sub[cell_mask_sub]
                 if intensity_values_in_cell.size > 0:
                     min_I=np.min(intensity_values_in_cell); max_I=np.max(intensity_values_in_cell); range_I = max_I - min_I
                     if range_I > 1e-9:
                          inverted_intensity_term = np.zeros_like(distance_transform, dtype=np.float32)
                          norm_inv_I=(max_I - intensity_sub[cell_mask_sub])/range_I; inverted_intensity_term[cell_mask_sub]=norm_inv_I
                          landscape += intensity_weight * inverted_intensity_term

            watershed_result = watershed(landscape, watershed_markers, mask=cell_mask_sub, watershed_line=True) # Works in 2D

            # --- Initial Post-processing (2D) ---
            temp_mask = np.zeros_like(cell_mask_sub, dtype=np.int32); used_new_labels.clear()
            current_temp_next_label = current_max_label_in_use + 1

            for marker_id in range(1, num_markers_final + 1):
                original_labels_tuple = marker_id_reverse_map.get(marker_id, tuple())
                final_label = cell_label if largest_soma_label in original_labels_tuple else current_temp_next_label
                if final_label == current_temp_next_label: used_new_labels.add(final_label); current_temp_next_label += 1
                temp_mask[watershed_result == marker_id] = final_label
                current_max_label_in_use = max(current_max_label_in_use, final_label)

            # Line filling (2D)
            watershed_lines_mask = (watershed_result == 0) & cell_mask_sub
            if np.any(watershed_lines_mask):
                 non_line_mask = temp_mask != 0
                 if np.any(non_line_mask):
                      _, nearest_label_indices = ndimage.distance_transform_edt(~non_line_mask, return_indices=True, sampling=spacing) # 2D EDT
                      nearest_labels = temp_mask[tuple(nearest_label_indices)]; temp_mask[watershed_lines_mask] = nearest_labels[watershed_lines_mask]
                 else: temp_mask[cell_mask_sub] = cell_label # Assign original if lines are all that's left

            # Small fragment merging (2D)
            fragments_merged = True
            while fragments_merged:
                 fragments_merged = False; labels_to_check = np.unique(temp_mask); labels_to_check = labels_to_check[labels_to_check > 0]
                 for lbl in labels_to_check:
                      lbl_mask = temp_mask == lbl; size = np.sum(lbl_mask)
                      if 0 < size < min_size_threshold:
                           struct = ndimage.generate_binary_structure(temp_mask.ndim, 1) # 2D structure
                           dilated_mask = ndimage.binary_dilation(lbl_mask, structure=struct)
                           neighbor_region = dilated_mask & (~lbl_mask) & (temp_mask != 0)
                           neighbor_labels = np.unique(temp_mask[neighbor_region])
                           if len(neighbor_labels) == 0: continue
                           neighbor_sizes = [(n_lbl, np.sum(temp_mask == n_lbl)) for n_lbl in neighbor_labels]
                           if not neighbor_sizes: continue
                           largest_neighbor_label = max(neighbor_sizes, key=lambda item: item[1])[0]
                           temp_mask[lbl_mask] = largest_neighbor_label; fragments_merged = True
                           if lbl in used_new_labels: used_new_labels.discard(lbl)

            # --- Post-Watershed Merging (2D - Interface Size) ---
            struct_dilate = ndimage.generate_binary_structure(temp_mask.ndim, 1) # 2D structure
            labels_after_frag_merge = np.unique(temp_mask); labels_after_frag_merge = labels_after_frag_merge[labels_after_frag_merge > 0]
            main_label = cell_label if cell_label in labels_after_frag_merge else -1
            new_labels_remaining = used_new_labels.intersection(labels_after_frag_merge)

            if main_label != -1 and new_labels_remaining:
                 main_mask = (temp_mask == main_label)
                 for new_lbl in list(new_labels_remaining):
                     if new_lbl not in np.unique(temp_mask): continue
                     new_lbl_mask = (temp_mask == new_lbl)
                     dilated_new_mask = ndimage.binary_dilation(new_lbl_mask, structure=struct_dilate)
                     interface_mask = main_mask & dilated_new_mask
                     if np.any(interface_mask):
                          interface_pixels_count = np.sum(interface_mask)
                          # Use interface pixel count instead of neck thickness
                          if interface_pixels_count >= post_merge_min_interface_pixels:
                              temp_mask[new_lbl_mask] = main_label
                              used_new_labels.discard(new_lbl)

            # Update global max label tracker
            final_labels_in_temp = np.unique(temp_mask); final_labels_in_temp = final_labels_in_temp[final_labels_in_temp > 0]
            if final_labels_in_temp.size > 0: current_max_label_in_use = max(current_max_label_in_use, np.max(final_labels_in_temp))

            # --- Map Back Sub-Image to Full Mask (2D) ---
            full_subvol = separated_mask[slice_obj]
            original_cell_pixels_in_sub = (segmentation_mask[slice_obj] == cell_label) # Renamed voxel->pixel
            full_subvol[original_cell_pixels_in_sub] = temp_mask[original_cell_pixels_in_sub]
            separated_mask[slice_obj] = full_subvol

        except Exception as e_outer: print(f"ERROR processing cell {cell_label} (2D): {e_outer}"); traceback.print_exc(); continue
        finally: # Cleanup loop vars
            del cell_mask_sub, soma_sub, cell_soma_sub_mask, watershed_markers, marker_id_reverse_map
            del distance_transform, landscape, watershed_result, temp_mask, used_new_labels
            if 'seed_props' in locals(): del seed_props
            if 'adj' in locals(): del adj
            if 'intensity_sub' in locals(): del intensity_sub
            if 'cost_array' in locals(): del cost_array
            if 'full_subvol' in locals(): del full_subvol
            if 'cell_mask' in locals(): del cell_mask
            if processed_count % 50 == 0: gc.collect()

    # --- End Separation Loop ---

    print(f"Finished processing {processed_count} 2D multi-soma cells. Skipped splitting {skipped_count}.")
    gc.collect()

    # --- Sequential Relabeling (Same logic, works on 2D final mask) ---
    print("\nRelabeling final 2D mask sequentially...")
    try:
        final_sequential_mask, fw_map, inv_map = relabel_sequential(separated_mask)
        final_max_label = len(inv_map) -1 # inv_map includes background 0
        print(f"Relabeled 2D mask contains {final_max_label} objects sequentially labeled from 1 to {final_max_label}.")
        return final_sequential_mask.astype(np.int32) # Ensure correct dtype
    except Exception as e_relabel:
        print(f"Error during sequential relabeling: {e_relabel}. Returning non-sequential mask.")
        return separated_mask # Fallback

# --- END OF FILE utils/ramified_module_2d/ramified_segmenter_2d.py ---