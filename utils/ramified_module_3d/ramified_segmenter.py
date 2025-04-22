# --- START OF FILE ramified_segmenter.py ---

import numpy as np
from scipy import ndimage
from tqdm import tqdm
from shutil import rmtree
import gc
from functools import partial
from skimage.measure import regionprops, label as skimage_label # Added skimage_label
from skimage.segmentation import watershed
from skimage.graph import route_through_array # Added for pathfinding
from skimage.morphology import binary_dilation # Added for interface finding
import math
from sklearn.decomposition import PCA # Keep PCA import
from skimage.feature import peak_local_max
from typing import List, Dict, Optional, Tuple, Union, Any, Set

seed = 42
np.random.seed(seed)         # For NumPy


# Helper function for physical size based min_distance (keep as is)
def get_min_distance_pixels(spacing, physical_distance):
    """Calculates minimum distance in pixels for peak_local_max based on physical distance."""
    # Use resolution in Y and X for separation within a slice primarily
    min_spacing_yx = min(spacing[1:])
    if min_spacing_yx <= 1e-6: return 3 # Avoid division by zero, return small default
    pixels = int(round(physical_distance / min_spacing_yx))
    return max(3, pixels) # Ensure minimum separation of at least 3 pixels

def extract_soma_masks(
    segmentation_mask: np.ndarray,
    spacing: Optional[Tuple[float, float, float]],
    smallest_quantile: int = 25,
    min_samples_for_median: int = 5,
    relative_core_definition_ratios: Union[List[float], float] = 0.3,
    min_fragment_size: int = 30,
    core_volume_target_factor_lower: float = 0.4,
    core_volume_target_factor_upper: float = 10
) -> np.ndarray:
    """
    Generates candidate seed mask using hybrid approach + mini-watershed.
    Aggregates results from multiple ratios: keeps all distinct splits and the
    smallest valid version of persistent seeds. Fallbacks fill remaining gaps.

    REVISED V7: Implements complex aggregation - smallest persistent seeds + all splits.
    """
    print("--- Starting Soma Extraction (Smallest Persistent + Splits Aggregation, V7) ---")

    # --- Handle input ratios (same as V5/V6) ---
    if isinstance(relative_core_definition_ratios, (float, int)): ratios_to_process = [float(relative_core_definition_ratios)]
    elif isinstance(relative_core_definition_ratios, list):
        if not relative_core_definition_ratios: ratios_to_process = [0.3]
        else:
             try: ratios_to_process = sorted(list(set(float(r) for r in relative_core_definition_ratios if r > 0 and r < 1)));
             except (ValueError, TypeError): ratios_to_process = [0.3]
             if not ratios_to_process: ratios_to_process = [0.3] # Handle empty list after filtering
    else: ratios_to_process = [0.3]
    print(f"Processing large cells with ratios: {ratios_to_process} (aggregating splits + smallest persistent)")

    # --- Internal Parameters & Setup (Identical to V6) ---
    min_physical_peak_separation = 7.0
    max_allowed_core_aspect_ratio = 10.0
    min_seed_fragment_volume = max(30, min_fragment_size)
    ref_vol_percentile_lower, ref_vol_percentile_upper = 30, 70
    ref_thickness_percentile_lower = 15
    absolute_min_thickness_um, absolute_max_thickness_um = 1.5, 10.0

    # Spacing setup
    if spacing is None: spacing = (1.0, 1.0, 1.0)
    else:
        try: spacing = tuple(float(s) for s in spacing); assert len(spacing) == 3
        except: print(f"Warn: Invalid spacing {spacing}. Using default."); spacing = (1.0, 1.0, 1.0)
    print(f"Using spacing (z,y,x): {spacing}")

    # ==============================================================
    # --- 1. Perform Setup Calculations (Only Once) ---
    #    (Finding valid labels, slices, volumes, thickness refs - Identical to V5/V6)
    # ==============================================================
    print("Calculating initial object properties...")
    unique_labels_all, counts = np.unique(segmentation_mask[segmentation_mask > 0], return_counts=True)
    if len(unique_labels_all) == 0: return np.zeros_like(segmentation_mask, dtype=np.int32)
    initial_volumes = dict(zip(unique_labels_all, counts))

    print("Finding object bounding boxes and validating labels...")
    label_to_slice: Dict[int, Tuple[slice, ...]] = {}
    valid_unique_labels_list: List[int] = []
    try: # Robust BBox finding
        object_slices = ndimage.find_objects(segmentation_mask)
        if object_slices is not None:
            num_slices_found = len(object_slices)
            for label in unique_labels_all:
                idx = label - 1
                if 0 <= idx < num_slices_found and object_slices[idx] is not None:
                    s = object_slices[idx]
                    if all(si.start < si.stop for si in s):
                        label_to_slice[label] = s; valid_unique_labels_list.append(label)
        if not valid_unique_labels_list: print("Error: No valid bounding boxes."); return np.zeros_like(segmentation_mask, dtype=np.int32)
        unique_labels = np.array(valid_unique_labels_list)
    except Exception as e: print(f"Error finding boxes: {e}"); return np.zeros_like(segmentation_mask, dtype=np.int32)

    volumes = {label: initial_volumes[label] for label in unique_labels}; all_volumes_list = list(volumes.values())
    if not all_volumes_list: print("Error: No valid volumes."); return np.zeros_like(segmentation_mask, dtype=np.int32)

    # Heuristic Target Volume & Filter Ranges
    smallest_thresh_volume = np.percentile(all_volumes_list, smallest_quantile) if len(all_volumes_list) > 1 else (all_volumes_list[0] + 1 if all_volumes_list else 1)
    smallest_object_labels_set = {label for label, vol in volumes.items() if vol <= smallest_thresh_volume}
    target_soma_volume = np.median(all_volumes_list) if len(all_volumes_list) < min_samples_for_median*2 else np.median([volumes[l] for l in smallest_object_labels_set if l in volumes] or all_volumes_list) # Simplified logic
    target_soma_volume = max(target_soma_volume, 1.0)
    min_accepted_core_volume = max(min_seed_fragment_volume, target_soma_volume * core_volume_target_factor_lower)
    max_accepted_core_volume = target_soma_volume * core_volume_target_factor_upper
    print(f"Core Volume filter range: [{min_accepted_core_volume:.2f} - {max_accepted_core_volume:.2f}] (Abs min: {min_seed_fragment_volume})")

    # Reference Thickness Calculation
    if len(all_volumes_list) <= 1: vol_thresh_lower, vol_thresh_upper = min(all_volumes_list)-1 if all_volumes_list else 0, max(all_volumes_list)+1 if all_volumes_list else 1
    else: vol_thresh_lower, vol_thresh_upper = np.percentile(all_volumes_list, ref_vol_percentile_lower), np.percentile(all_volumes_list, ref_vol_percentile_upper)
    reference_object_labels = {label for label in unique_labels if volumes[label] > vol_thresh_lower and volumes[label] <= vol_thresh_upper}
    reference_object_labels_for_ref = list(unique_labels) if len(reference_object_labels) < min_samples_for_median else list(reference_object_labels)
    print(f"Calculating reference thickness from {len(reference_object_labels_for_ref)} objects...")
    max_thicknesses_reference_objs = []
    for label_ref in tqdm(reference_object_labels_for_ref, desc="Calc Ref Thickness", disable=len(reference_object_labels_for_ref) < 10):
        # ... (thickness calculation loop - identical to V5/V6) ...
        bbox_slice = label_to_slice[label_ref]
        sub_segmentation, object_mask_sub, dist_transform_obj_ref = None, None, None
        try:
            sub_segmentation = segmentation_mask[bbox_slice]
            object_mask_sub = (sub_segmentation == label_ref)
            if object_mask_sub.size == 0 or not np.any(object_mask_sub): continue
            dist_transform_obj_ref = ndimage.distance_transform_edt(object_mask_sub, sampling=spacing)
            max_dist = np.max(dist_transform_obj_ref)
            if max_dist > 0: max_thicknesses_reference_objs.append(max_dist)
        except MemoryError: print(f"Warn: Ref thick MemError L{label_ref}."); gc.collect(); continue
        except Exception as e: print(f"Warn: Ref thick Error L{label_ref}: {e}"); continue
        finally:
             if 'dist_transform_obj_ref' in locals() and dist_transform_obj_ref is not None: del dist_transform_obj_ref
             if 'object_mask_sub' in locals() and object_mask_sub is not None: del object_mask_sub
             if 'sub_segmentation' in locals() and sub_segmentation is not None: del sub_segmentation

    # Final Thickness Range
    min_accepted_thickness = absolute_min_thickness_um; max_accepted_thickness = absolute_max_thickness_um
    if max_thicknesses_reference_objs:
        if len(max_thicknesses_reference_objs) < min_samples_for_median:
            median_ref_thick = np.median(max_thicknesses_reference_objs); range_estimate = 2.0
            min_accepted_thickness = max(absolute_min_thickness_um, median_ref_thick - range_estimate)
        else:
            q_low_val = np.percentile(max_thicknesses_reference_objs, ref_thickness_percentile_lower)
            min_accepted_thickness = max(absolute_min_thickness_um, q_low_val)
    if min_accepted_thickness >= max_accepted_thickness: min_accepted_thickness, max_accepted_thickness = absolute_min_thickness_um, absolute_max_thickness_um
    print(f"Final Core Thickness filter range: [{min_accepted_thickness:.2f} - {max_accepted_thickness:.2f}] um")
    min_peak_sep_pixels = get_min_distance_pixels(spacing, min_physical_peak_separation)

    # ==============================================================
    # --- 2. Process Small Cells (Directly to Final Mask) ---
    # ==============================================================
    print("\nProcessing objects...")
    final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
    next_final_label = 1
    added_small_labels: Set[int] = set()

    small_cell_labels_to_process = [l for l in unique_labels if l in smallest_object_labels_set]
    print(f"Processing {len(small_cell_labels_to_process)} small objects...")
    for label in tqdm(small_cell_labels_to_process, desc="Small Cells", disable=len(small_cell_labels_to_process) < 10):
        # ... (identical small cell processing loop from V5/V6, adding directly to final_seed_mask) ...
        bbox_slice = label_to_slice[label]
        slice_obj = bbox_slice
        sub_segmentation, object_mask_sub, obj_coords = None, None, None
        try:
            sub_segmentation = segmentation_mask[slice_obj]
            object_mask_sub = (sub_segmentation == label)
            obj_volume = np.sum(object_mask_sub)
            if object_mask_sub.size > 0 and obj_volume >= min_seed_fragment_volume:
                offset = (bbox_slice[0].start, bbox_slice[1].start, bbox_slice[2].start)
                obj_coords = np.argwhere(object_mask_sub)
                global_coords_z = obj_coords[:, 0] + offset[0]; global_coords_y = obj_coords[:, 1] + offset[1]; global_coords_x = obj_coords[:, 2] + offset[2]
                valid_indices = ((global_coords_z >= 0) & (global_coords_z < final_seed_mask.shape[0]) & (global_coords_y >= 0) & (global_coords_y < final_seed_mask.shape[1]) & (global_coords_x >= 0) & (global_coords_x < final_seed_mask.shape[2]))
                if isinstance(valid_indices, np.ndarray) and valid_indices.ndim == 1 and len(valid_indices) == len(global_coords_z):
                     # Check if region is already occupied before writing
                     coords_to_write = (global_coords_z[valid_indices], global_coords_y[valid_indices], global_coords_x[valid_indices])
                     if np.all(final_seed_mask[coords_to_write] == 0):
                          final_seed_mask[coords_to_write] = next_final_label
                          next_final_label += 1; added_small_labels.add(label)
                     # else: Region already occupied (e.g., by another small cell's overlap), skip.
        except MemoryError: print(f"Warn: MemError small L{label}."); gc.collect()
        except IndexError as ie: print(f"Warn: IndexErr small L{label}: {ie}")
        except Exception as e: print(f"Warn: Error small L{label}: {e}")
        finally: # Cleanup vars
            if 'obj_coords' in locals() and obj_coords is not None: del obj_coords
            if 'object_mask_sub' in locals() and object_mask_sub is not None: del object_mask_sub
            if 'sub_segmentation' in locals() and sub_segmentation is not None: del sub_segmentation

    print(f"Added {len(added_small_labels)} initial seeds from small cells.")
    gc.collect()

    # ==============================================================
    # --- 3. Generate All Candidate Seeds from Large Cells ---
    # ==============================================================
    large_cell_labels_to_process = [l for l in unique_labels if l not in added_small_labels]
    print(f"Generating candidates from {len(large_cell_labels_to_process)} large objects across {len(ratios_to_process)} ratios...")

    valid_candidates = [] # List of {'mask': mask, 'volume': v, 'offset': offset}
    fallback_candidates = [] # List of {'mask': mask, 'volume': v, 'offset': offset}

    for label in tqdm(large_cell_labels_to_process, desc="Large Cell Candidates"):
        bbox_slice = label_to_slice[label]

        for current_ratio in ratios_to_process:
            # --- Setup for this ratio/label ---
            pad = 1
            z_min = max(0, bbox_slice[0].start - pad); y_min = max(0, bbox_slice[1].start - pad); x_min = max(0, bbox_slice[2].start - pad)
            z_max = min(segmentation_mask.shape[0], bbox_slice[0].stop + pad); y_max = min(segmentation_mask.shape[1], bbox_slice[1].stop + pad); x_max = min(segmentation_mask.shape[2], bbox_slice[2].stop + pad)
            if z_min >= z_max or y_min >= y_max or x_min >= x_max: continue
            slice_obj = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]; offset = (z_min, y_min, x_min)

            # Temp lists for this specific label/ratio run
            passed_filters_list_this_ratio = []
            min_size_only_list_this_ratio = []
            sub_segmentation, object_mask_sub, dist_transform_obj = None, None, None # Init vars for finally

            try:
                # --- Core Detection & Watershed (mostly same as V6) ---
                try: sub_segmentation = segmentation_mask[slice_obj]
                except IndexError: continue
                object_mask_sub = (sub_segmentation == label)
                if not np.any(object_mask_sub): continue

                dist_transform_obj = ndimage.distance_transform_edt(object_mask_sub, sampling=spacing)
                max_dist_in_obj = np.max(dist_transform_obj)
                if max_dist_in_obj <= 1e-9: continue

                core_thresh = max_dist_in_obj * current_ratio
                initial_core_region_mask_sub = (dist_transform_obj >= core_thresh) & object_mask_sub
                if not np.any(initial_core_region_mask_sub): continue

                labeled_cores, num_cores = ndimage.label(initial_core_region_mask_sub)
                if num_cores == 0: continue

                # Process Core Components
                for core_label in range(1, num_cores + 1):
                    core_component_mask_sub = (labeled_cores == core_label)
                    if not np.any(core_component_mask_sub): continue

                    # Mini-Watershed -> candidate_fragment_masks
                    candidate_fragment_masks = []
                    # ... (watershed logic from V6/V7) ...
                    dt_core, peaks, markers_core, ws_core = None, None, None, None
                    try:
                        dt_core = ndimage.distance_transform_edt(core_component_mask_sub, sampling=spacing)
                        if np.max(dt_core) > 1e-9:
                            peaks = peak_local_max(dt_core, min_distance=min_peak_sep_pixels, labels=core_component_mask_sub, num_peaks_per_label=0, exclude_border=False)
                            if peaks.shape[0] > 1:
                                markers_core = np.zeros(dt_core.shape, dtype=np.int32); markers_core[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                                ws_core = watershed(-dt_core, markers_core, mask=core_component_mask_sub, watershed_line=False)
                                unique_ws_labels = np.unique(ws_core)
                                for ws_label in unique_ws_labels:
                                     if ws_label == 0: continue
                                     fragment_mask = (ws_core == ws_label)
                                     if np.any(fragment_mask): candidate_fragment_masks.append(fragment_mask)
                            else: candidate_fragment_masks.append(core_component_mask_sub)
                        else: candidate_fragment_masks.append(core_component_mask_sub)
                    except MemoryError: candidate_fragment_masks.append(core_component_mask_sub); gc.collect() # Fallback on mem error
                    except Exception: candidate_fragment_masks.append(core_component_mask_sub) # Fallback on other errors
                    finally: del dt_core, peaks, markers_core, ws_core # Cleanup

                    # Filter Candidate Fragments
                    for i, fragment_mask in enumerate(candidate_fragment_masks):
                        # ... (filtering logic checking min size, thickness, volume, aspect ratio) ...
                        fragment_passed_all = False; fragment_mask_copy = None
                        labeled_fragment, props_fragment_labeled = None, None # Init for cleanup
                        try:
                            fragment_volume = np.sum(fragment_mask)
                            if fragment_volume < min_seed_fragment_volume: continue # 1. Min Size Check

                            # Passed min size, create copy for potential storage
                            fragment_mask_copy = fragment_mask.copy()
                            temp_min_size_info = {'mask': fragment_mask_copy, 'volume': fragment_volume} # Store copy info

                            fragment_distances = dist_transform_obj[fragment_mask]; max_dist_in_fragment = np.max(fragment_distances) if fragment_distances.size > 0 else 0
                            passes_thickness = (max_dist_in_fragment >= min_accepted_thickness and max_dist_in_fragment <= max_accepted_thickness) # 2. Thickness
                            if not passes_thickness: del fragment_mask_copy; temp_min_size_info = None; continue # Failed, discard copy info

                            passes_volume = (fragment_volume >= min_accepted_core_volume and fragment_volume <= max_accepted_core_volume) # 3. Volume Range
                            if not passes_volume: del fragment_mask_copy; temp_min_size_info = None; continue

                            passes_aspect = True; aspect_ratio = -1 # 4. Aspect Ratio
                            try: # Robust aspect ratio calc
                                fragment_mask_int = fragment_mask.astype(np.uint8, copy=False) # Avoid copy if possible
                                labeled_fragment, num_feat = ndimage.label(fragment_mask_int)
                                if num_feat > 0:
                                    props_fragment_labeled = regionprops(labeled_fragment)
                                    if props_fragment_labeled:
                                        frag_prop = props_fragment_labeled[0]; eigvals_sq = frag_prop.inertia_tensor_eigvals
                                        if eigvals_sq is not None and np.all(np.isfinite(eigvals_sq)):
                                             eigvals_sq_sorted = np.sort(np.abs(eigvals_sq))[::-1]
                                             if np.min(eigvals_sq_sorted) > 1e-12:
                                                  lengths = np.sqrt(eigvals_sq_sorted)
                                                  if lengths.size >= 3 and lengths[2] > 1e-7: aspect_ratio = lengths[0] / lengths[2]
                                                  elif lengths.size == 2 and lengths[1] > 1e-7: aspect_ratio = lengths[0] / lengths[1]
                                                  if aspect_ratio != -1 and aspect_ratio > max_allowed_core_aspect_ratio: passes_aspect = False
                            except MemoryError: passes_aspect = True; gc.collect() # Assume pass
                            except Exception: passes_aspect = True # Assume pass

                            if not passes_aspect: del fragment_mask_copy; temp_min_size_info = None; continue

                            # --- Passed ALL Filters ---
                            fragment_passed_all = True
                            # Add the copy we made earlier to the list for this ratio
                            passed_filters_list_this_ratio.append({'mask': fragment_mask_copy, 'volume': fragment_volume})
                            temp_min_size_info = None # Signal it passed, don't add to fallback list

                        except MemoryError: print(f"Warn: Filter MemError L{label} R{current_ratio} F{i}. Skip."); gc.collect(); temp_min_size_info=None
                        finally: # Cleanup filter vars
                             if 'labeled_fragment' in locals() and labeled_fragment is not None: del labeled_fragment
                             if 'props_fragment_labeled' in locals() and props_fragment_labeled is not None: del props_fragment_labeled
                             # If it only passed min size, add its info to the ratio's fallback list
                             if temp_min_size_info is not None and not fragment_passed_all:
                                 min_size_only_list_this_ratio.append(temp_min_size_info)
                             elif fragment_passed_all and 'fragment_mask_copy' in locals() and fragment_mask_copy is not None:
                                 # Mask copy was added to passed_filters_list_this_ratio
                                 del fragment_mask_copy # Remove local reference


                # --- End Core Component Loop ---

                # --- Decide what to add to global candidate lists for this Label/Ratio ---
                if passed_filters_list_this_ratio:
                    # Add all valid fragments found for this ratio
                    for item in passed_filters_list_this_ratio:
                        valid_candidates.append({'mask': item['mask'], 'volume': item['volume'], 'offset': offset})
                elif min_size_only_list_this_ratio:
                    # Only add the largest fallback IF NO valid fragments were found this ratio
                    largest_fallback = max(min_size_only_list_this_ratio, key=lambda x: x['volume'])
                    fallback_candidates.append({'mask': largest_fallback['mask'], 'volume': largest_fallback['volume'], 'offset': offset})

            except MemoryError: print(f"Warn: MemError L{label} R{current_ratio}. Skip ratio."); gc.collect()
            except IndexError as ie: print(f"Warn: IndexError L{label} R{current_ratio}: {ie}. Skip ratio.")
            except Exception as e: print(f"Warn: Error L{label} R{current_ratio}: {e}"); import traceback; traceback.print_exc()
            finally: # Cleanup ratio vars
                if 'sub_segmentation' in locals() and sub_segmentation is not None: del sub_segmentation
                if 'object_mask_sub' in locals() and object_mask_sub is not None: del object_mask_sub
                if 'dist_transform_obj' in locals() and dist_transform_obj is not None: del dist_transform_obj
                # Clean up mask copies in temporary lists for this ratio
                for item in passed_filters_list_this_ratio:
                     if 'mask' in item: del item['mask']
                for item in min_size_only_list_this_ratio:
                     if 'mask' in item: del item['mask']
                del passed_filters_list_this_ratio, min_size_only_list_this_ratio
                gc.collect()
        # --- End Ratio Loop ---
    # --- End Label Loop ---

    print(f"Generated {len(valid_candidates)} valid candidates and {len(fallback_candidates)} fallback candidates.")

    # ==============================================================
    # --- 4. Process Valid Candidates (Smallest First) ---
    # ==============================================================
    print("Placing smallest valid seeds first...")
    valid_candidates.sort(key=lambda x: x['volume']) # Sort by volume ascending

    processed_count = 0
    for candidate in tqdm(valid_candidates, desc="Placing Valid Seeds"):
        mask_sub = candidate['mask']
        offset = candidate['offset']
        # Calculate global coordinates only if mask is not empty
        if mask_sub is None or mask_sub.size == 0: continue
        try:
            coords_sub = np.argwhere(mask_sub)
            if coords_sub.size == 0: continue # Skip if no True values

            global_coords_z = coords_sub[:, 0] + offset[0]
            global_coords_y = coords_sub[:, 1] + offset[1]
            global_coords_x = coords_sub[:, 2] + offset[2]

            # Boundary checks
            valid_indices = ((global_coords_z >= 0) & (global_coords_z < final_seed_mask.shape[0]) &
                             (global_coords_y >= 0) & (global_coords_y < final_seed_mask.shape[1]) &
                             (global_coords_x >= 0) & (global_coords_x < final_seed_mask.shape[2]))

            if not np.any(valid_indices): continue # Skip if all coords outside bounds

            # Get valid global coordinates as tuple for indexing
            coords_global = (global_coords_z[valid_indices],
                             global_coords_y[valid_indices],
                             global_coords_x[valid_indices])

            # Check if ALL target voxels in final_mask are currently empty (0)
            if np.all(final_seed_mask[coords_global] == 0):
                final_seed_mask[coords_global] = next_final_label
                next_final_label += 1
                processed_count += 1

        except MemoryError: print("Warn: MemError processing valid candidate. Skipping."); gc.collect()
        except IndexError as ie: print(f"Warn: IndexError processing valid candidate: {ie}. Skipping.")
        except Exception as e: print(f"Warn: Error processing valid candidate: {e}. Skipping.")
        finally:
             # Explicitly delete the mask copy after processing
             if 'mask' in candidate: del candidate['mask']


    print(f"Placed {processed_count} seeds from valid candidates.")
    del valid_candidates # Free memory
    gc.collect()

    # ==============================================================
    # --- 5. Process Fallback Candidates (Fill Gaps) ---
    # ==============================================================
    print("Placing fallback seeds in empty regions...")
    # Optional: Sort fallbacks, e.g., largest first? May not be critical.
    # fallback_candidates.sort(key=lambda x: x['volume'], reverse=True)

    fallback_processed_count = 0
    for fallback in tqdm(fallback_candidates, desc="Placing Fallbacks"):
        mask_sub = fallback['mask']
        offset = fallback['offset']
        if mask_sub is None or mask_sub.size == 0: continue
        try:
            coords_sub = np.argwhere(mask_sub)
            if coords_sub.size == 0: continue

            global_coords_z = coords_sub[:, 0] + offset[0]; global_coords_y = coords_sub[:, 1] + offset[1]; global_coords_x = coords_sub[:, 2] + offset[2]
            valid_indices = ((global_coords_z >= 0) & (global_coords_z < final_seed_mask.shape[0]) & (global_coords_y >= 0) & (global_coords_y < final_seed_mask.shape[1]) & (global_coords_x >= 0) & (global_coords_x < final_seed_mask.shape[2]))
            if not np.any(valid_indices): continue

            coords_global = (global_coords_z[valid_indices], global_coords_y[valid_indices], global_coords_x[valid_indices])

            # Check if ALL target voxels are currently empty (0)
            if np.all(final_seed_mask[coords_global] == 0):
                final_seed_mask[coords_global] = next_final_label
                next_final_label += 1
                fallback_processed_count += 1

        except MemoryError: print("Warn: MemError processing fallback. Skipping."); gc.collect()
        except IndexError as ie: print(f"Warn: IndexError processing fallback: {ie}. Skipping.")
        except Exception as e: print(f"Warn: Error processing fallback: {e}. Skipping.")
        finally:
             if 'mask' in fallback: del fallback['mask']

    print(f"Placed {fallback_processed_count} seeds from fallback candidates.")
    del fallback_candidates
    gc.collect()

    total_final_seeds = next_final_label - 1
    print(f"\nGenerated a total of {total_final_seeds} final aggregated seeds.")
    print("--- Finished Intermediate Seed Extraction ---")

    return final_seed_mask


# --- refine_seeds_pca function remains the same ---
def refine_seeds_pca(intermediate_seed_mask,
                     spacing,
                     target_aspect_ratio=1.1,
                     projection_percentile_crop=10,
                     min_fragment_size=30):
    # ... (Code identical to previous version - no changes needed here based on request) ...
    print("--- Starting PCA Seed Refinement ---")
    if spacing is None: spacing = (1.0, 1.0, 1.0); print("Warn: No spacing, assume isotropic.")
    else: print(f"Using spacing (z,y,x): {spacing}")
    print(f"Parameters: target_aspect_ratio={target_aspect_ratio}, projection_crop%={projection_percentile_crop}, min_vol={min_fragment_size}")
    final_refined_mask = np.zeros_like(intermediate_seed_mask, dtype=np.int32)
    next_final_label = 1; kept_refined_count = 0
    seed_labels = np.unique(intermediate_seed_mask); seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels) == 0: print("Input intermediate seed mask is empty."); return final_refined_mask
    seed_slices = ndimage.find_objects(intermediate_seed_mask)
    label_to_slice = {label: seed_slices[label-1] for label in seed_labels if label-1 < len(seed_slices) and seed_slices[label-1] is not None}
    for label in tqdm(seed_labels, desc="Refining Seeds PCA"):
        bbox_slice = label_to_slice.get(label);
        if bbox_slice is None: continue
        offset = (bbox_slice[0].start, bbox_slice[1].start, bbox_slice[2].start); slice_obj = bbox_slice
        seed_mask_sub, coords_vox, coords_phys, pca = None, None, None, None; refined_mask_sub = None
        try:
            seed_mask_sub = (intermediate_seed_mask[slice_obj] == label); current_volume = np.sum(seed_mask_sub)
            if seed_mask_sub.size == 0 or not np.any(seed_mask_sub) or current_volume < min_fragment_size: continue
            coords_vox = np.argwhere(seed_mask_sub)
            if coords_vox.shape[0] <= 3: refined_mask_sub = seed_mask_sub
            else:
                coords_phys = coords_vox * np.array(spacing)
                pca = PCA(n_components=3); pca.fit(coords_phys); eigenvalues = pca.explained_variance_; eigenvectors = pca.components_
                sorted_indices = np.argsort(eigenvalues)[::-1]; eigenvalues = eigenvalues[sorted_indices]; eigenvectors = eigenvectors[sorted_indices]
                if eigenvalues[2] < 1e-9: eigenvalues[2] = 1e-9 # Avoid division by zero
                aspect_ratio = eigenvalues[0] / eigenvalues[2]
                if aspect_ratio >= target_aspect_ratio:
                    coords_centered_phys = coords_phys - pca.mean_; proj1 = coords_centered_phys @ eigenvectors[0]
                    min_p = np.percentile(proj1, projection_percentile_crop); max_p = np.percentile(proj1, 100 - projection_percentile_crop)
                    voxel_indices_to_keep = (proj1 >= min_p) & (proj1 <= max_p)
                    refined_mask_sub = np.zeros_like(seed_mask_sub, dtype=bool)
                    kept_coords_vox = coords_vox[voxel_indices_to_keep]
                    if kept_coords_vox.shape[0] > 0: refined_mask_sub[tuple(kept_coords_vox.T)] = True
                    refined_volume = np.sum(refined_mask_sub)
                    if refined_volume < min_fragment_size: refined_mask_sub = None # Discard if refinement makes it too small
                else: refined_mask_sub = seed_mask_sub # Keep original if not elongated enough
            if refined_mask_sub is not None and np.any(refined_mask_sub):
                 final_volume = np.sum(refined_mask_sub)
                 if final_volume >= min_fragment_size:
                     refined_coords = np.argwhere(refined_mask_sub)
                     if refined_coords.size > 0:
                         global_coords_z=refined_coords[:,0]+offset[0]; global_coords_y=refined_coords[:,1]+offset[1]; global_coords_x=refined_coords[:,2]+offset[2]
                         valid_indices=((global_coords_z>=0)&(global_coords_z<final_refined_mask.shape[0])& (global_coords_y>=0)&(global_coords_y<final_refined_mask.shape[1])& (global_coords_x>=0)&(global_coords_x<final_refined_mask.shape[2]))
                         final_refined_mask[global_coords_z[valid_indices], global_coords_y[valid_indices], global_coords_x[valid_indices]] = next_final_label
                         next_final_label += 1; kept_refined_count += 1
        except Exception as e: print(f"Warning: Error during PCA refinement for seed label {label}: {e}")
        finally:
            if seed_mask_sub is not None: del seed_mask_sub;
            if coords_vox is not None: del coords_vox;
            if coords_phys is not None: del coords_phys;
            if pca is not None: del pca;
            if refined_mask_sub is not None: del refined_mask_sub;
            if 'coords_centered_phys' in locals(): del coords_centered_phys;
            if 'proj1' in locals(): del proj1;
            if 'kept_coords_vox' in locals(): del kept_coords_vox;
            if 'refined_coords' in locals(): del refined_coords;
    print(f"Kept {kept_refined_count} refined seeds after PCA.")
    print("--- Finished PCA Seed Refinement ---"); gc.collect(); return final_refined_mask


def separate_multi_soma_cells(segmentation_mask,
                              intensity_volume,
                              soma_mask, # Output from refine_seeds_pca
                              spacing,
                              min_size_threshold=100,
                              intensity_weight=0.0,
                              # Heuristic parameters passed in:
                              max_seed_centroid_dist=15.0,
                              min_path_intensity_ratio=0.6
                             ):
    """
    Separates cell segmentations containing multiple seeds. Includes heuristics
    to merge seeds based on path intensity and post-watershed merging based
    on neck thickness. RELABELED OUTPUT V2: Ensures final mask labels are sequential 1..N.
    """
    print(f"--- Starting Multi-Soma Separation (Path Heuristics + Post-Merge + Sequential Relabel) ---")

    # --- Internal Parameters ---
    post_merge_min_neck_thickness = 2.0
    post_merge_min_interface_voxels = 20
    print(f"Internal Params: post_merge_min_neck_thickness={post_merge_min_neck_thickness}, post_merge_min_interface_voxels={post_merge_min_interface_voxels}")

    if spacing is None: spacing = (1.0, 1.0, 1.0); print("Warn: No spacing, assume isotropic.")
    else:
        try: # Validate spacing input
            spacing = tuple(float(s) for s in spacing)
            if len(spacing) != 3: raise ValueError("Spacing must have 3 dimensions.")
            print(f"Using spacing (z,y,x): {spacing}")
        except (ValueError, TypeError) as e:
            print(f"Error: Invalid spacing provided ({spacing}). Using default (1.0, 1.0, 1.0). Error: {e}")
            spacing = (1.0, 1.0, 1.0)
    print(f"Seed merging heuristics: max_dist={max_seed_centroid_dist}, min_path_intensity_ratio={min_path_intensity_ratio}")

    # Use float32 for intensity if performing calculations, ensure input matches
    if not np.issubdtype(intensity_volume.dtype, np.floating):
        print("Converting intensity volume to float32 for calculations.")
        intensity_volume = intensity_volume.astype(np.float32, copy=False) # Avoid copy if already float

    # Ensure segmentation and soma masks are integer types
    if not np.issubdtype(segmentation_mask.dtype, np.integer):
        segmentation_mask = segmentation_mask.astype(np.int32)
    if not np.issubdtype(soma_mask.dtype, np.integer):
        soma_mask = soma_mask.astype(np.int32)

    separated_mask = np.copy(segmentation_mask).astype(np.int32) # Work on a copy

    # --- Mapping ---
    print("Mapping seeds to original cell segments...")
    unique_initial_labels = np.unique(segmentation_mask); unique_initial_labels = unique_initial_labels[unique_initial_labels > 0]
    if unique_initial_labels.size == 0: print("Original segmentation mask is empty."); return separated_mask
    cell_to_somas = {cell_label: set() for cell_label in unique_initial_labels}
    present_soma_labels = np.unique(soma_mask); present_soma_labels = present_soma_labels[present_soma_labels > 0]
    if present_soma_labels.size == 0: print("Seed mask is empty. No separation possible."); return separated_mask

    # Efficient mapping using unique labels and masks
    for soma_label in tqdm(present_soma_labels, desc="Mapping Seeds", disable=len(present_soma_labels) < 10):
        soma_loc_mask = (soma_mask == soma_label)
        # Find original cell labels under this soma mask
        cell_labels_under_soma = np.unique(segmentation_mask[soma_loc_mask])
        cell_labels_under_soma = cell_labels_under_soma[cell_labels_under_soma > 0]
        for cell_label in cell_labels_under_soma:
            if cell_label in cell_to_somas: # Check if cell_label is valid
                cell_to_somas[cell_label].add(soma_label)

    # --- Identify candidates AND Original Label Management ---
    multi_soma_cell_labels = [lbl for lbl, somas in cell_to_somas.items() if len(somas) > 1]
    print(f"Found {len(multi_soma_cell_labels)} initial segments with multiple seeds - applying heuristics...")
    # Determine starting label for NEW segments safely
    max_orig_label = np.max(unique_initial_labels) # Already filtered > 0
    max_soma_label = np.max(present_soma_labels) # Already filtered > 0
    next_label = max(max_orig_label, max_soma_label) + 1
    print(f"Tentative starting label for new segments: {next_label}")


    # --- Separation Loop ---
    skipped_count = 0
    processed_count = 0
    current_max_label_in_use = next_label - 1 # Track highest label assigned so far

    for cell_label in tqdm(multi_soma_cell_labels, desc="Separating Segments"):
        processed_count += 1
        # Define variables needed in finally block outside try
        cell_mask_sub, soma_sub, cell_soma_sub_mask = None, None, None
        seed_props, adj, intensity_sub, cost_array = None, None, None, None
        distance_transform, landscape, watershed_result = None, None, None
        temp_mask = None; watershed_markers = None; marker_id_reverse_map = None
        used_new_labels = set() # Track labels introduced *for this cell*

        try:
            # --- Get Subvolume ---
            cell_mask = segmentation_mask == cell_label
            # Use find_objects for potentially more efficient bounding box
            obj_slice = ndimage.find_objects(cell_mask)
            if not obj_slice or obj_slice[0] is None: continue # Skip if object not found
            bbox_slice = obj_slice[0]
            pad = 5
            z_min=max(0,bbox_slice[0].start-pad); y_min=max(0,bbox_slice[1].start-pad); x_min=max(0,bbox_slice[2].start-pad)
            z_max=min(segmentation_mask.shape[0],bbox_slice[0].stop+pad); y_max=min(segmentation_mask.shape[1],bbox_slice[1].stop+pad); x_max=min(segmentation_mask.shape[2],bbox_slice[2].stop+pad)
            slice_obj=np.s_[z_min:z_max, y_min:y_max, x_min:x_max]; offset=(z_min,y_min,x_min)

            cell_mask_sub = cell_mask[slice_obj]
            if cell_mask_sub.size == 0 or not np.any(cell_mask_sub): continue

            # Get seeds within this subvolume
            soma_sub = soma_mask[slice_obj]
            cell_soma_sub_mask = np.zeros_like(soma_sub, dtype=np.int32)
            cell_soma_sub_mask[cell_mask_sub] = soma_sub[cell_mask_sub] # Apply cell mask
            soma_labels_in_cell = np.unique(cell_soma_sub_mask); soma_labels_in_cell = soma_labels_in_cell[soma_labels_in_cell > 0]
            if len(soma_labels_in_cell) <= 1: continue

            # --- Seed Merging Heuristic (Pathfinding) ---
            intensity_sub = intensity_volume[slice_obj] # Already float32
            seed_props = regionprops(cell_soma_sub_mask, intensity_image=intensity_sub)
            seed_prop_dict = {prop.label: prop for prop in seed_props}

            num_seeds = len(soma_labels_in_cell); seed_indices = {lbl: i for i, lbl in enumerate(soma_labels_in_cell)}
            adj = np.zeros((num_seeds, num_seeds), dtype=bool); merge_candidates = False

            # Prepare cost array for pathfinding
            max_I_sub = np.max(intensity_sub[cell_mask_sub]) if np.any(cell_mask_sub) else 1.0
            cost_array = np.full(intensity_sub.shape, np.inf, dtype=np.float32)
            cost_array[cell_mask_sub] = (max_I_sub - intensity_sub[cell_mask_sub]) + 1e-6

            # Check pairs for merging
            for i in range(num_seeds):
                for j in range(i + 1, num_seeds):
                    # ... (distance check) ...
                    label1 = soma_labels_in_cell[i]; label2 = soma_labels_in_cell[j]
                    prop1 = seed_prop_dict.get(label1); prop2 = seed_prop_dict.get(label2)
                    if prop1 is None or prop2 is None: continue
                    cent1_phys = np.array(prop1.centroid) * np.array(spacing); cent2_phys = np.array(prop2.centroid) * np.array(spacing)
                    dist = np.linalg.norm(cent1_phys - cent2_phys)
                    if dist > max_seed_centroid_dist: continue

                    # ... (path intensity check using route_through_array) ...
                    coords1_bounded = tuple(np.clip(int(round(c)), 0, s-1) for c, s in zip(prop1.centroid, intensity_sub.shape))
                    coords2_bounded = tuple(np.clip(int(round(c)), 0, s-1) for c, s in zip(prop2.centroid, intensity_sub.shape))
                    median_intensity_on_path = 0
                    try:
                        indices, weight = route_through_array(cost_array, coords1_bounded, coords2_bounded, fully_connected=True)
                        if indices.shape[1] > 0: # Check if path found (shape is (ndim, npoints))
                            path_intensities = intensity_sub[tuple(indices)] # Direct indexing works for (ndim, npoints)
                            if path_intensities.size > 0: median_intensity_on_path = np.median(path_intensities)
                    except ValueError: median_intensity_on_path = 0 # Path not found

                    max_intensity_in_seeds = max(prop1.max_intensity, prop2.max_intensity) if prop1.max_intensity > 0 and prop2.max_intensity > 0 else 1.0
                    if max_intensity_in_seeds < 1e-6: max_intensity_in_seeds = 1.0
                    intensity_ratio = median_intensity_on_path / max_intensity_in_seeds
                    if intensity_ratio < min_path_intensity_ratio: continue

                    # Passed checks -> mark for merging
                    adj[seed_indices[label1], seed_indices[label2]] = adj[seed_indices[label2], seed_indices[label1]] = True
                    merge_candidates = True

            # --- Make Decision & Prepare Watershed Markers ---
            watershed_markers = np.zeros_like(cell_mask_sub, dtype=np.int32)
            marker_id_reverse_map = {}
            largest_soma_label = -1; max_soma_size = -1
            num_markers_final = 0; current_marker_id = 1

            if merge_candidates:
                n_comp, comp_labels = ndimage.label(adj)
                if n_comp == 1: # All seeds connected -> skip watershed
                    print(f"    Cell {cell_label}: All {num_seeds} seeds connected. Skipping split.")
                    skipped_count += 1; continue # Skip to next cell_label

                print(f"    Cell {cell_label}: Found {n_comp} groups of seeds. Merging markers...")
                for group_idx in range(1, n_comp + 1): # comp_labels are 1-based
                    group_marker_id = current_marker_id
                    seed_indices_in_group = np.where(comp_labels == group_idx)[0]
                    group_mask = np.zeros_like(cell_mask_sub, dtype=bool)
                    original_labels_in_group = set()
                    for seed_idx in seed_indices_in_group:
                         original_label = soma_labels_in_cell[seed_idx]; original_labels_in_group.add(original_label)
                         group_mask |= (cell_soma_sub_mask == original_label)
                         prop = seed_prop_dict.get(original_label)
                         if prop and prop.area > max_soma_size: max_soma_size = prop.area; largest_soma_label = original_label
                    watershed_markers[group_mask] = group_marker_id
                    marker_id_reverse_map[group_marker_id] = tuple(sorted(list(original_labels_in_group)))
                    current_marker_id += 1
                num_markers_final = n_comp
            else: # No merges -> use individual seeds
                 for soma_label in soma_labels_in_cell:
                     seed_region_mask = cell_soma_sub_mask == soma_label
                     watershed_markers[seed_region_mask] = current_marker_id
                     marker_id_reverse_map[current_marker_id] = (soma_label,)
                     prop = seed_prop_dict.get(soma_label)
                     if prop and prop.area > max_soma_size: max_soma_size = prop.area; largest_soma_label = soma_label
                     current_marker_id += 1
                 num_markers_final = num_seeds

            # --- Landscape and Watershed ---
            distance_transform = ndimage.distance_transform_edt(cell_mask_sub, sampling=spacing)
            landscape = -distance_transform
            if intensity_weight > 1e-6: # Add intensity term if needed
                 intensity_values_in_cell = intensity_sub[cell_mask_sub]
                 if intensity_values_in_cell.size > 0:
                     min_I=np.min(intensity_values_in_cell); max_I=np.max(intensity_values_in_cell); range_I = max_I - min_I
                     if range_I > 1e-9:
                          inverted_intensity_term = np.zeros_like(distance_transform, dtype=np.float32)
                          norm_inv_I=(max_I - intensity_sub[cell_mask_sub])/range_I; inverted_intensity_term[cell_mask_sub]=norm_inv_I
                          landscape += intensity_weight * inverted_intensity_term

            watershed_result = watershed(landscape, watershed_markers, mask=cell_mask_sub, watershed_line=True)

            # --- Initial Post-processing (Label assignment, line filling, small fragment removal) ---
            temp_mask = np.zeros_like(cell_mask_sub, dtype=np.int32);
            # Reset used_new_labels for this cell
            used_new_labels.clear()
            # Start assigning labels from the current global max + 1
            current_temp_next_label = current_max_label_in_use + 1

            for marker_id in range(1, num_markers_final + 1):
                original_labels_tuple = marker_id_reverse_map.get(marker_id, tuple())
                if largest_soma_label in original_labels_tuple:
                    final_label = cell_label # Assign original ID to main segment
                else:
                    final_label = current_temp_next_label # Assign new unique label
                    used_new_labels.add(final_label)
                    current_temp_next_label += 1
                temp_mask[watershed_result == marker_id] = final_label
                # Update the global max label tracker
                current_max_label_in_use = max(current_max_label_in_use, final_label)

            # Line filling
            watershed_lines_mask = (watershed_result == 0) & cell_mask_sub
            if np.any(watershed_lines_mask):
                 non_line_mask = temp_mask != 0
                 if np.any(non_line_mask):
                      _, nearest_label_indices = ndimage.distance_transform_edt(~non_line_mask, return_indices=True, sampling=spacing)
                      nearest_labels = temp_mask[tuple(nearest_label_indices)]; temp_mask[watershed_lines_mask] = nearest_labels[watershed_lines_mask]
                 else: temp_mask[cell_mask_sub] = cell_label

            # Small fragment merging
            fragments_merged = True
            while fragments_merged:
                 fragments_merged = False; labels_to_check = np.unique(temp_mask); labels_to_check = labels_to_check[labels_to_check > 0]
                 for lbl in labels_to_check:
                      lbl_mask = temp_mask == lbl; size = np.sum(lbl_mask)
                      if 0 < size < min_size_threshold:
                           struct = ndimage.generate_binary_structure(temp_mask.ndim, 1)
                           dilated_mask = ndimage.binary_dilation(lbl_mask, structure=struct)
                           neighbor_region = dilated_mask & (~lbl_mask) & (temp_mask != 0)
                           neighbor_labels = np.unique(temp_mask[neighbor_region])
                           if len(neighbor_labels) == 0: continue
                           neighbor_sizes = [(n_lbl, np.sum(temp_mask == n_lbl)) for n_lbl in neighbor_labels]
                           if not neighbor_sizes: continue
                           largest_neighbor_label = max(neighbor_sizes, key=lambda item: item[1])[0]
                           temp_mask[lbl_mask] = largest_neighbor_label; fragments_merged = True
                           # If a new label was merged, remove it from tracking
                           if lbl in used_new_labels: used_new_labels.discard(lbl)

            # --- Post-Watershed Merging (Neck Thickness) ---
            # print(f"    Cell {cell_label}: Applying post-watershed merge check...") # Keep less verbose
            struct_dilate = ndimage.generate_binary_structure(temp_mask.ndim, 1)
            labels_after_frag_merge = np.unique(temp_mask); labels_after_frag_merge = labels_after_frag_merge[labels_after_frag_merge > 0]
            main_label = cell_label if cell_label in labels_after_frag_merge else -1
            new_labels_remaining = used_new_labels.intersection(labels_after_frag_merge) # Check against current labels

            if main_label != -1 and new_labels_remaining:
                 main_mask = (temp_mask == main_label)
                 # Iterate over a copy in case set changes during iteration
                 for new_lbl in list(new_labels_remaining):
                     if new_lbl not in np.unique(temp_mask): continue # Already merged away earlier
                     new_lbl_mask = (temp_mask == new_lbl)
                     dilated_new_mask = ndimage.binary_dilation(new_lbl_mask, structure=struct_dilate)
                     interface_mask = main_mask & dilated_new_mask
                     if np.any(interface_mask): # Check if interface exists
                          interface_voxels_count = np.sum(interface_mask)
                          interface_dt_values = distance_transform[interface_mask]
                          min_neck_thickness = np.min(interface_dt_values) if interface_dt_values.size > 0 else 0
                          if min_neck_thickness >= post_merge_min_neck_thickness and interface_voxels_count >= post_merge_min_interface_voxels:
                              # print(f"      Merging fragment {new_lbl} into {main_label}") # Optional debug
                              temp_mask[new_lbl_mask] = main_label
                              # Remove from used_new_labels as it's gone
                              used_new_labels.discard(new_lbl)

            # --- Update global max label tracker after all processing for this cell ---
            final_labels_in_temp = np.unique(temp_mask); final_labels_in_temp = final_labels_in_temp[final_labels_in_temp > 0]
            if final_labels_in_temp.size > 0:
                current_max_label_in_use = max(current_max_label_in_use, np.max(final_labels_in_temp))


            # --- Map Back Subvolume to Full Mask ---
            # Get current state of the full mask in the subvolume
            full_subvol = separated_mask[slice_obj]
            # Identify voxels belonging to the original cell within the subvolume
            original_cell_voxels_in_sub = (segmentation_mask[slice_obj] == cell_label)
            # Update ONLY those voxels in the full_subvol that belonged to the original cell
            full_subvol[original_cell_voxels_in_sub] = temp_mask[original_cell_voxels_in_sub]
            # Write the modified subvolume back
            separated_mask[slice_obj] = full_subvol

        except Exception as e_outer:
            print(f"ERROR processing cell {cell_label}: {e_outer}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            # Skip to next cell if error occurs during processing of one cell
            continue
        finally:
            # --- Explicit Cleanup for loop variables ---
            # Using 'del' can help release memory sooner, especially in loops with large arrays
            del cell_mask_sub, soma_sub, cell_soma_sub_mask, watershed_markers, marker_id_reverse_map
            del distance_transform, landscape, watershed_result, temp_mask, used_new_labels
            # Check existence before deleting potentially unassigned variables from try block
            if 'seed_props' in locals() and seed_props is not None: del seed_props
            if 'adj' in locals() and adj is not None: del adj
            if 'intensity_sub' in locals() and intensity_sub is not None: del intensity_sub
            if 'cost_array' in locals() and cost_array is not None: del cost_array
            if 'full_subvol' in locals() and full_subvol is not None: del full_subvol
            if 'cell_mask' in locals() and cell_mask is not None: del cell_mask # Defined outside try, but good practice
            # Call garbage collector periodically if memory is an issue
            if processed_count % 50 == 0: gc.collect() # e.g., every 50 cells

    # --- End Separation Loop ---

    print(f"Finished processing {processed_count} multi-soma cells. Skipped splitting {skipped_count}.")
    # Final GC before relabeling
    gc.collect()


    # ==============================================================
    # --- V2 MODIFICATION: Sequential Relabeling Step ---
    # ==============================================================
    print("\nRelabeling final mask sequentially...")
    final_labels_present = np.unique(separated_mask)
    # Filter out background label 0
    old_labels = final_labels_present[final_labels_present > 0]

    if old_labels.size == 0:
        print("No objects found in the final mask. Returning.")
        return separated_mask # Nothing to relabel

    print(f"Found {old_labels.size} unique objects with current labels (min: {np.min(old_labels)}, max: {np.max(old_labels)}).")

    # Create the mapping from old labels to new sequential labels (1 to N)
    # np.unique sorts the labels, so the mapping order is consistent
    label_map = {old_label: new_label for new_label, old_label in enumerate(old_labels, start=1)}

    # Build the lookup table for efficient mapping
    # The table index corresponds to the old label, the value is the new label
    max_old_label_val = np.max(old_labels)
    # Size needs to accommodate the maximum old label value + 1 (for 0-based index)
    lookup_table_size = int(max_old_label_val) + 1
    # Initialize lookup table with zeros (background maps to background)
    # Use the same dtype as the mask being relabeled
    lookup_table = np.zeros(lookup_table_size, dtype=separated_mask.dtype)

    # Populate the lookup table
    for old_label, new_label in label_map.items():
         # Basic check for index validity, although max_old_label_val should ensure this
         if 0 <= old_label < lookup_table_size:
              lookup_table[old_label] = new_label
         else:
              # This should ideally not happen if max_old_label_val calculation is correct
              print(f"Warning: Old label {old_label} is out of bounds for lookup table (size {lookup_table_size}). This might indicate an issue.")

    # Apply the lookup table to the entire mask using advanced indexing
    # This operation efficiently replaces each old label value with its corresponding new label value from the table
    print("Applying sequential mapping...")
    try:
        final_sequential_mask = lookup_table[separated_mask]
        final_max_label = len(label_map) # The highest new label assigned
        print(f"Relabeled mask contains {final_max_label} objects sequentially labeled from 1 to {final_max_label}.")
        return final_sequential_mask
    except IndexError:
         print("Error: An index error occurred during lookup table application. This might happen if labels in separated_mask exceed max_old_label_val.")
         print(f"Max label found by unique: {max_old_label_val}. Max label in mask: {np.max(separated_mask)}.")
         print("Returning the non-sequentially labeled mask as a fallback.")
         return separated_mask # Fallback to non-sequential mask on error
    except MemoryError:
         print("Error: MemoryError occurred during final relabeling. System might be out of memory.")
         print("Returning the non-sequentially labeled mask as a fallback.")
         return separated_mask # Fallback