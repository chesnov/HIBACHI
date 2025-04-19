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

def extract_soma_masks(segmentation_mask,
                       spacing,
                       smallest_quantile=25,
                       min_samples_for_median=5,
                       relative_core_definition_ratio=0.3, # Use the value found useful
                       min_fragment_size=30,
                       core_volume_target_factor_lower=0.4,
                       core_volume_target_factor_upper=10 # Increased based on previous finding
                       ):
    """
    Generates candidate seed mask using the hybrid approach + mini-watershed.
    REVISED V3: Incorporates debugging prints and refined fallback logic management.
                Uses data-driven lower thickness bound, fixed upper bound.
    """
    print("--- Starting Soma Extraction (Hybrid + Mini-WS + Hybrid Thickness Filter V3) ---")

    # --- Internal Parameters ---
    # relative_core_definition_ratio_internal = 0.3 # Using arg directly now
    min_physical_peak_separation = 7.0 # Tunable peak separation
    max_allowed_core_aspect_ratio = 10.0 # Keep relaxed initially
    min_seed_fragment_volume = max(30, min_fragment_size) # Min voxel size for ANY seed fragment
    # --- Thickness Calculation Parameters ---
    ref_vol_percentile_lower = 30
    ref_vol_percentile_upper = 70
    ref_thickness_percentile_lower = 15 # Use Q15 for lower bound
    absolute_min_thickness_um = 1.5  # TUNABLE: Min plausible core thickness
    absolute_max_thickness_um = 10.0 # TUNABLE: Max plausible core thickness (Set based on tests)
    # --- DEBUGGING: Set your problem cell label here ---
    YOUR_PROBLEM_CELL_LABEL = 10551 # Replace with the actual label, or None to disable debug prints
    # ---

    print(f"Params: core_ratio={relative_core_definition_ratio}, peak_sep={min_physical_peak_separation}um, aspect_ratio={max_allowed_core_aspect_ratio}, min_frag_vol={min_seed_fragment_volume}")
    print(f"Thickness Ref: Vol=[{ref_vol_percentile_lower}-{ref_vol_percentile_upper}]%, Thick=[Q{ref_thickness_percentile_lower} Lower, Fixed Upper], Absolute Bounds=[{absolute_min_thickness_um:.1f}-{absolute_max_thickness_um:.1f}]um")

    if spacing is None: spacing = (1.0, 1.0, 1.0); print("Warn: No spacing, assume isotropic.")
    else: print(f"Using spacing (z,y,x): {spacing}")

    # --- Calculate reference stats (Volume first) ---
    print("Calculating initial object volumes..."); unique_labels, counts = np.unique(segmentation_mask[segmentation_mask > 0], return_counts=True)
    if len(unique_labels) == 0: return np.zeros_like(segmentation_mask, dtype=np.int32)
    volumes = dict(zip(unique_labels, counts)); all_volumes_list = list(volumes.values())
    if not all_volumes_list: return np.zeros_like(segmentation_mask, dtype=np.int32)

    # --- Identify Reference Objects for Thickness (Mid-Range Volume) ---
    print(f"Identifying reference objects ({ref_vol_percentile_lower}-{ref_vol_percentile_upper}% volume)...")
    if len(all_volumes_list) <= 1:
        vol_thresh_lower = np.min(all_volumes_list) -1
        vol_thresh_upper = np.max(all_volumes_list) +1
    else:
        vol_thresh_lower = np.percentile(all_volumes_list, ref_vol_percentile_lower)
        vol_thresh_upper = np.percentile(all_volumes_list, ref_vol_percentile_upper)
    reference_object_labels = {label for label, vol in volumes.items() if vol > vol_thresh_lower and vol <= vol_thresh_upper}
    print(f"Found {len(reference_object_labels)} objects in reference volume range [{vol_thresh_lower:.0f} - {vol_thresh_upper:.0f}]")
    if len(reference_object_labels) < min_samples_for_median:
        print(f"Warning: Only {len(reference_object_labels)} mid-range objects found. Falling back to using ALL objects for ref thickness stats.")
        reference_object_labels_for_ref = unique_labels
    else:
        reference_object_labels_for_ref = list(reference_object_labels)

    print("Finding object bounding boxes..."); object_slices = ndimage.find_objects(segmentation_mask)
    label_to_slice = {label: object_slices[label-1] for label in unique_labels if label-1 < len(object_slices) and object_slices[label-1] is not None}

    # --- Calculate Reference Volume (Heuristic using smallest quantile - optional) ---
    smallest_thresh_volume = np.percentile(all_volumes_list, smallest_quantile) if len(all_volumes_list)>1 else all_volumes_list[0]+1
    smallest_object_labels_set = {label for label, vol in volumes.items() if vol <= smallest_thresh_volume}
    target_soma_volume = 1.0; smallest_volumes_for_vol_ref = [volumes[l] for l in smallest_object_labels_set if l in volumes]
    if len(smallest_volumes_for_vol_ref) >= min_samples_for_median: target_soma_volume = np.median(smallest_volumes_for_vol_ref); print(f"Heuristic Target Volume (Median of {len(smallest_volumes_for_vol_ref)} smallest): {target_soma_volume:.2f}")
    elif smallest_volumes_for_vol_ref: target_soma_volume = np.median(smallest_volumes_for_vol_ref); print(f"Heuristic Target Volume (Median of {len(smallest_volumes_for_vol_ref)} smallest - low N): {target_soma_volume:.2f}")
    else: target_soma_volume = np.median(all_volumes_list) if all_volumes_list else 1.0; print(f"Heuristic Target Volume (Overall Median - Fallback): {target_soma_volume:.2f}")
    target_soma_volume = max(target_soma_volume, 1.0); min_accepted_core_volume = max(min_seed_fragment_volume, target_soma_volume * core_volume_target_factor_lower); max_accepted_core_volume = target_soma_volume * core_volume_target_factor_upper
    print(f"Core Volume filter range: [{min_accepted_core_volume:.2f} - {max_accepted_core_volume:.2f}] (Absolute min: {min_seed_fragment_volume})")

    # --- Calculate Reference Thickness from selected reference population ---
    print(f"Calculating reference thickness distribution from {len(reference_object_labels_for_ref)} objects..."); max_thicknesses_reference_objs = []
    for label_ref in tqdm(reference_object_labels_for_ref, desc="Calc Ref Thickness"): # Renamed loop var
        bbox_slice = label_to_slice.get(label_ref);
        if bbox_slice is None: continue
        sub_segmentation, object_mask_sub, dist_transform_obj_ref = None, None, None # Use _ref suffix
        try:
            sub_segmentation = segmentation_mask[bbox_slice]; object_mask_sub = (sub_segmentation == label_ref)
            if object_mask_sub.size == 0 or not np.any(object_mask_sub): continue
            dist_transform_obj_ref = ndimage.distance_transform_edt(object_mask_sub, sampling=spacing)
            max_dist = np.max(dist_transform_obj_ref)
            if max_dist > 0: max_thicknesses_reference_objs.append(max_dist)
        except Exception as e: print(f"Warn: Ref thick {label_ref} (Slice: {bbox_slice}): Error occurred: {e}"); continue
        finally: # Cleanup ref thickness inner loop
             if dist_transform_obj_ref is not None: del dist_transform_obj_ref
             if object_mask_sub is not None: del object_mask_sub
             if sub_segmentation is not None: del sub_segmentation

    # --- Define Final Thickness Range using Hybrid Strategy ---
    min_accepted_thickness = absolute_min_thickness_um
    max_accepted_thickness = absolute_max_thickness_um

    if not max_thicknesses_reference_objs:
        print(f"ERROR: Could not get any thickness values from reference objects. Using absolute bounds [{absolute_min_thickness_um:.2f} - {absolute_max_thickness_um:.2f}].")
    elif len(max_thicknesses_reference_objs) < min_samples_for_median:
        print(f"Warning: Only {len(max_thicknesses_reference_objs)} thickness values found. Using Median +/- range clamped by absolute.")
        median_ref_thick = np.median(max_thicknesses_reference_objs)
        estimated_lower = median_ref_thick - 2.0
        min_accepted_thickness = max(absolute_min_thickness_um, estimated_lower)
    else:
        q_low_val = np.percentile(max_thicknesses_reference_objs, ref_thickness_percentile_lower)
        median_target_max_thickness = np.median(max_thicknesses_reference_objs)
        q_high_ref_val = np.percentile(max_thicknesses_reference_objs, 85) # For printing
        print(f"Reference thickness distribution (n={len(max_thicknesses_reference_objs)}): Median={median_target_max_thickness:.2f}, Q{ref_thickness_percentile_lower}={q_low_val:.2f}, Q85={q_high_ref_val:.2f} (physical units)")
        min_accepted_thickness = max(absolute_min_thickness_um, q_low_val)
        # max_accepted_thickness remains absolute_max_thickness_um

    if min_accepted_thickness > max_accepted_thickness:
        print(f"Warning: Calculated min thickness ({min_accepted_thickness:.2f}) > max ({max_accepted_thickness:.2f}). Using absolute bounds only.")
        min_accepted_thickness = absolute_min_thickness_um
        max_accepted_thickness = absolute_max_thickness_um

    print(f"Final Core Max Thickness filter range (Q{ref_thickness_percentile_lower} Lower Clamped, Fixed Upper): [{min_accepted_thickness:.2f} - {max_accepted_thickness:.2f}] (physical units)")

    # --- Process ALL Objects ---
    print("Processing objects: Keeping small, applying mini-watershed to cores of large...");
    final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
    next_final_label = 1; kept_seed_count = 0
    added_small_labels = set()

    for label in tqdm(unique_labels, desc="Generating Seeds"):
        bbox_slice = label_to_slice.get(label);
        if bbox_slice is None: continue

        # --- Handle Small Cells ---
        if label in smallest_object_labels_set:
            # (Identical logic as previous version)
            slice_obj = bbox_slice; sub_segmentation, object_mask_sub, obj_coords = None, None, None
            try:
                sub_segmentation = segmentation_mask[slice_obj]; object_mask_sub = (sub_segmentation == label)
                if object_mask_sub.size > 0 and np.any(object_mask_sub) and np.sum(object_mask_sub) >= min_seed_fragment_volume:
                     offset = (bbox_slice[0].start, bbox_slice[1].start, bbox_slice[2].start); obj_coords = np.argwhere(object_mask_sub)
                     global_coords_z = obj_coords[:, 0] + offset[0]; global_coords_y = obj_coords[:, 1] + offset[1]; global_coords_x = obj_coords[:, 2] + offset[2]
                     valid_indices = ((global_coords_z >= 0) & (global_coords_z < final_seed_mask.shape[0]) & (global_coords_y >= 0) & (global_coords_y < final_seed_mask.shape[1]) & (global_coords_x >= 0) & (global_coords_x < final_seed_mask.shape[2]))
                     final_seed_mask[global_coords_z[valid_indices], global_coords_y[valid_indices], global_coords_x[valid_indices]] = next_final_label
                     next_final_label += 1; kept_seed_count += 1
                     added_small_labels.add(label)
            except Exception as e: print(f"Warn: Error adding small cell {label}: {e}")
            finally: # Cleanup small cell vars
                if sub_segmentation is not None: del sub_segmentation
                if object_mask_sub is not None: del object_mask_sub
                if obj_coords is not None: del obj_coords
            continue

        # --- Handle Large Cells (Mini-Watershed Core Processing) ---
        else:
            if label in added_small_labels: continue

            pad = 1
            z_min = max(0, bbox_slice[0].start - pad); y_min = max(0, bbox_slice[1].start - pad); x_min = max(0, bbox_slice[2].start - pad)
            z_max = min(segmentation_mask.shape[0], bbox_slice[0].stop + pad); y_max = min(segmentation_mask.shape[1], bbox_slice[1].stop + pad); x_max = min(segmentation_mask.shape[2], bbox_slice[2].stop + pad)
            slice_obj = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]; offset = (z_min, y_min, x_min)

            sub_segmentation, object_mask_sub, dist_transform_obj = None, None, None
            initial_core_region_mask_sub, labeled_cores = None, None
            processed_fragments_passing_filters = [] # Holds masks that passed ALL filters for this cell
            fallback_candidates_data = [] # Holds {'mask': copy, 'volume': vol, ...} for fragments passing MIN size ONLY

            try:
                 sub_segmentation = segmentation_mask[slice_obj]; object_mask_sub = (sub_segmentation == label)
                 if object_mask_sub.size == 0 or not np.any(object_mask_sub): continue
                 dist_transform_obj = ndimage.distance_transform_edt(object_mask_sub, sampling=spacing); max_dist_in_obj = np.max(dist_transform_obj)
                 if max_dist_in_obj <= 0: continue

                 # Use relative_core_definition_ratio argument directly
                 core_thresh = max_dist_in_obj * relative_core_definition_ratio
                 initial_core_region_mask_sub = (dist_transform_obj >= core_thresh) & object_mask_sub
                 if not np.any(initial_core_region_mask_sub): continue
                 labeled_cores, num_cores = ndimage.label(initial_core_region_mask_sub)
                 min_peak_sep_pixels = get_min_distance_pixels(spacing, min_physical_peak_separation)

                 # Debug print before processing cores
                 if label == YOUR_PROBLEM_CELL_LABEL:
                      print(f"\n--- Processing Problem Cell {label} ---")
                      print(f"  Num initial core components: {num_cores}")

                 for core_label in range(1, num_cores + 1):
                     core_component_mask_sub = (labeled_cores == core_label)
                     if not np.any(core_component_mask_sub): continue

                     if label == YOUR_PROBLEM_CELL_LABEL:
                         print(f"---> Core Comp. {core_label} (Vol: {np.sum(core_component_mask_sub)})")

                     candidate_fragment_masks = []
                     dt_core, peaks, markers_core, ws_core = None, None, None, None
                     try: # Mini-Watershed
                         dt_core = ndimage.distance_transform_edt(core_component_mask_sub, sampling=spacing)
                         peaks = peak_local_max(dt_core, min_distance=min_peak_sep_pixels, labels=core_component_mask_sub, num_peaks_per_label=0, exclude_border=False)
                         if label == YOUR_PROBLEM_CELL_LABEL:
                             print(f"    Mini-WS: Found {peaks.shape[0]} peaks for Core {core_label}")
                         if peaks.shape[0] > 1:
                             markers_core = np.zeros(dt_core.shape, dtype=np.int32)
                             markers_core[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                             ws_core = watershed(-dt_core, markers_core, mask=core_component_mask_sub, watershed_line=False)
                             for ws_label in range(1, peaks.shape[0] + 1):
                                 fragment_mask = (ws_core == ws_label)
                                 if np.any(fragment_mask): candidate_fragment_masks.append(fragment_mask)
                         else: candidate_fragment_masks.append(core_component_mask_sub) # Treat as one fragment
                     except Exception as e_miniws:
                         print(f"    Warn: Mini-watershed failed Label {label}, Core {core_label}: {e_miniws}. Treating as single.")
                         candidate_fragment_masks.append(core_component_mask_sub) # Fallback
                     finally: # Mini-WS cleanup
                          if dt_core is not None: del dt_core
                          if peaks is not None: del peaks
                          if markers_core is not None: del markers_core
                          if ws_core is not None: del ws_core

                     # Filter Each Candidate Fragment
                     if label == YOUR_PROBLEM_CELL_LABEL:
                         print(f"    Filtering {len(candidate_fragment_masks)} candidate fragments from Core {core_label}")

                     props_fragment_labeled, labeled_fragment = None, None
                     fragment_filter_temp_fallback_list = [] # Temp list for this core component's fallbacks

                     for i, fragment_mask in enumerate(candidate_fragment_masks):
                         fragment_passed_all = False; fragment_mask_copy = None
                         # Unique identifier for fallback dicts within this core component
                         fallback_id = (core_label, i)

                         try: # Fragment filtering try block
                             fragment_volume = np.sum(fragment_mask)

                             if label == YOUR_PROBLEM_CELL_LABEL:
                                 print(f"    --> Frag {i} (from Core {core_label}): Vol={fragment_volume:.0f} (MinSeedVol={min_seed_fragment_volume})")

                             if fragment_volume < min_seed_fragment_volume:
                                 if label == YOUR_PROBLEM_CELL_LABEL: print(f"        -> FAILED: Too Small (Min Seed Vol)")
                                 continue # Min size check

                             # Passed min size, potentially add to fallback
                             fragment_mask_copy = fragment_mask.copy()
                             # Add to a *temporary* list first
                             fragment_filter_temp_fallback_list.append({'mask': fragment_mask_copy, 'volume': fragment_volume, 'id': fallback_id})
                             if label == YOUR_PROBLEM_CELL_LABEL: print(f"        -> Passed Min Seed Vol Check. (Added temp fallback)")


                             fragment_distances = dist_transform_obj[fragment_mask]
                             max_dist_in_fragment = np.max(fragment_distances) if fragment_distances.size > 0 else 0

                             if label == YOUR_PROBLEM_CELL_LABEL:
                                  print(f"        Vol Filter Range: [{min_accepted_core_volume:.0f}-{max_accepted_core_volume:.0f}]")
                                  print(f"        Thickness Filter Range: [{min_accepted_thickness:.2f}-{max_accepted_thickness:.2f}]")
                                  print(f"        Actual Vol = {fragment_volume:.0f}")
                                  print(f"        Actual Max Thick = {max_dist_in_fragment:.2f}")

                             # 1. Thickness filter
                             passes_thickness = (max_dist_in_fragment >= min_accepted_thickness and max_dist_in_fragment <= max_accepted_thickness)
                             if label == YOUR_PROBLEM_CELL_LABEL: print(f"        Passes Thickness? {passes_thickness}")
                             if not passes_thickness:
                                  # Remove from temp fallback list if fails subsequent filters
                                  fragment_filter_temp_fallback_list.pop(-1)
                                  del fragment_mask_copy
                                  continue

                             # 2. Volume filter
                             passes_volume = (fragment_volume >= min_accepted_core_volume and fragment_volume <= max_accepted_core_volume)
                             if label == YOUR_PROBLEM_CELL_LABEL: print(f"        Passes Volume? {passes_volume}")
                             if not passes_volume:
                                  fragment_filter_temp_fallback_list.pop(-1)
                                  del fragment_mask_copy
                                  continue

                             # 3. Aspect Ratio Filter
                             passes_aspect = True; aspect_ratio = -1
                             try:
                                 labeled_fragment, _ = ndimage.label(fragment_mask)
                                 props_fragment_labeled = regionprops(labeled_fragment)
                                 if props_fragment_labeled:
                                      frag_prop = props_fragment_labeled[0]
                                      if frag_prop.inertia_tensor_eigvals is not None:
                                          eigvals = np.sqrt(np.abs(frag_prop.inertia_tensor_eigvals))
                                          eigvals = np.sort(eigvals)[::-1]
                                          if eigvals.size > 2 and eigvals[2] > 1e-6:
                                              aspect_ratio = eigvals[0] / eigvals[2]
                                              if aspect_ratio > max_allowed_core_aspect_ratio: passes_aspect = False
                             except Exception as e_aspect: print(f"    Warn: Aspect ratio calc failed frag {i}: {e_aspect}")

                             if label == YOUR_PROBLEM_CELL_LABEL:
                                 print(f"        Actual Aspect = {aspect_ratio:.2f} (Max: {max_allowed_core_aspect_ratio})")
                                 print(f"        Passes Aspect? {passes_aspect}")
                             if not passes_aspect:
                                 fragment_filter_temp_fallback_list.pop(-1)
                                 del fragment_mask_copy
                                 continue

                             # Passed ALL filters
                             fragment_passed_all = True
                             processed_fragments_passing_filters.append(fragment_mask)
                             if label == YOUR_PROBLEM_CELL_LABEL: print(f"        -> PASSED ALL FILTERS")
                             # Remove from temp fallback list as it passed all
                             fragment_filter_temp_fallback_list.pop(-1)
                             del fragment_mask_copy

                         finally: # Fragment filter cleanup
                              if props_fragment_labeled is not None: del props_fragment_labeled
                              if labeled_fragment is not None: del labeled_fragment
                              # fragment_mask_copy is now managed explicitly above

                     # After checking all fragments from a core component, add remaining temp fallbacks to main list
                     fallback_candidates_data.extend(fragment_filter_temp_fallback_list)
                     # We keep the masks in fallback_candidates_data until the very end of processing the cell ('label')

                 # --- End of loop through core components ---

                 # --- Add Seeds for this Large Cell (label) ---
                 if label == YOUR_PROBLEM_CELL_LABEL:
                     print(f"--- Summary for Cell {label} ---")
                     print(f"  Fragments Passing Filters: {len(processed_fragments_passing_filters)}")
                     print(f"  Fallback Candidates (passed min size only): {len(fallback_candidates_data)}")
                     if fallback_candidates_data:
                          fb_vols = [item['volume'] for item in fallback_candidates_data]
                          print(f"  Fallback Volumes: {fb_vols}")

                 # --- Fallback Logic Check ---
                 if processed_fragments_passing_filters: # If ANY fragment passed filters for this 'label'
                     if label == YOUR_PROBLEM_CELL_LABEL: print(f"  -> Adding {len(processed_fragments_passing_filters)} fragments that passed filters.")
                     for frag_mask in processed_fragments_passing_filters: # Add all that passed
                         comp_coords = np.argwhere(frag_mask)
                         if comp_coords.size > 0:
                             global_coords_z=comp_coords[:,0]+offset[0]; global_coords_y=comp_coords[:,1]+offset[1]; global_coords_x=comp_coords[:,2]+offset[2]
                             valid_indices=((global_coords_z>=0)&(global_coords_z<final_seed_mask.shape[0])& (global_coords_y>=0)&(global_coords_y<final_seed_mask.shape[1])& (global_coords_x>=0)&(global_coords_x<final_seed_mask.shape[2]))
                             final_seed_mask[global_coords_z[valid_indices], global_coords_y[valid_indices], global_coords_x[valid_indices]] = next_final_label
                             next_final_label += 1; kept_seed_count += 1
                     # Fallback is NOT used if any fragment passed primary filters

                 elif fallback_candidates_data: # If NO fragments passed filters, but some passed min size
                     largest_fallback_candidate = max(fallback_candidates_data, key=lambda x: x['volume'])
                     fallback_mask = largest_fallback_candidate['mask'] # Use the stored mask
                     if label == YOUR_PROBLEM_CELL_LABEL:
                         print(f"  -> Primary filters failed for ALL fragments. Adding LARGEST fallback fragment (Vol: {largest_fallback_candidate['volume']:.0f}).")
                     else: # Print for other cells hitting fallback
                         print(f"  Label {label}: Primary filters failed. Adding LARGEST fallback fragment (Vol: {largest_fallback_candidate['volume']:.0f}).")

                     fb_coords = np.argwhere(fallback_mask)
                     if fb_coords.size > 0:
                          global_coords_z=fb_coords[:,0]+offset[0]; global_coords_y=fb_coords[:,1]+offset[1]; global_coords_x=fb_coords[:,2]+offset[2]
                          valid_indices=((global_coords_z>=0)&(global_coords_z<final_seed_mask.shape[0])& (global_coords_y>=0)&(global_coords_y<final_seed_mask.shape[1])& (global_coords_x>=0)&(global_coords_x<final_seed_mask.shape[2]))
                          final_seed_mask[global_coords_z[valid_indices], global_coords_y[valid_indices], global_coords_x[valid_indices]] = next_final_label
                          next_final_label += 1; kept_seed_count += 1

                 else: # No fragments passed filters, and no fragments passed min size either
                     if label == YOUR_PROBLEM_CELL_LABEL: print(f"  -> No fragments passed filters OR min size.")

            except Exception as e: print(f"Warning: Error processing object label {label}: {e}")
            finally: # Large cell block cleanup
                 if sub_segmentation is not None: del sub_segmentation
                 if object_mask_sub is not None: del object_mask_sub
                 if dist_transform_obj is not None: del dist_transform_obj
                 if initial_core_region_mask_sub is not None: del initial_core_region_mask_sub
                 if labeled_cores is not None: del labeled_cores
                 # Clean up mask copies stored in fallback_candidates_data for this 'label'
                 for item in fallback_candidates_data:
                     if 'mask' in item and item['mask'] is not None: del item['mask']
                 del fallback_candidates_data # Clear the list of dicts
                 del processed_fragments_passing_filters # Clear the list of masks
                 gc.collect() # Explicit garbage collect for large objects

    print(f"Generated {kept_seed_count} intermediate seeds before PCA refinement.")
    print("--- Finished Intermediate Seed Extraction ---")
    del added_small_labels # Clean up the tracking set
    gc.collect()
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
                              # landscape_smoothing_sigma=0.0, # Removed smoothing
                              # Heuristic parameters passed in:
                              max_seed_centroid_dist=15.0,
                              min_path_intensity_ratio=0.6
                             ):
    """
    Separates cell segmentations containing multiple seeds. Includes heuristics
    to merge seeds within the same original label if they are close and connected
    by bright pixels along a path within the mask, aiming to prevent splitting processes.
    ADDED: Post-watershed merging based on interface thickness.
    """
    print(f"--- Starting Multi-Soma Separation (Path Heuristics + Post-Merge) ---")

    # --- Internal Parameters ---
    # Post-watershed merging: Merge split-off segment B into main segment A if
    # the connection is thick enough and the interface is large enough.
    # Physical distance threshold based on distance transform at interface
    post_merge_min_neck_thickness = 2.0  # e.g., in um if spacing is in um. Tune this heavily.
    # Voxel count threshold for the interface between the two segments
    post_merge_min_interface_voxels = 20 # Tune this.
    print(f"Internal Params: post_merge_min_neck_thickness={post_merge_min_neck_thickness}, post_merge_min_interface_voxels={post_merge_min_interface_voxels}")
    # ---

    if spacing is None: spacing = (1.0, 1.0, 1.0); print("Warn: No spacing, assume isotropic.")
    else: print(f"Using spacing (z,y,x): {spacing}")
    print(f"Seed merging heuristics: max_dist={max_seed_centroid_dist}, min_path_intensity_ratio={min_path_intensity_ratio}")

    separated_mask = np.copy(segmentation_mask).astype(np.int32)

    # --- Mapping (No change) ---
    print("Mapping seeds to original cell segments...")
    unique_initial_labels = np.unique(segmentation_mask); unique_initial_labels = unique_initial_labels[unique_initial_labels > 0]
    cell_to_somas = {cell_label: set() for cell_label in unique_initial_labels}
    present_soma_labels = np.unique(soma_mask); present_soma_labels = present_soma_labels[present_soma_labels > 0]
    if present_soma_labels.size == 0: print("Seed mask is empty."); return separated_mask
    soma_indices = np.argwhere(soma_mask > 0)
    print(f"Checking {len(soma_indices)} seed voxels for mapping...")
    for idx_tuple in tqdm(soma_indices, desc="Mapping Seeds"):
        idx = tuple(idx_tuple); cell_label = segmentation_mask[idx]
        if cell_label > 0: soma_label = soma_mask[idx]; cell_to_somas[cell_label].add(soma_label)

    # --- Identify potential candidates AND Original Label Management ---
    multi_soma_cell_labels = [lbl for lbl, somas in cell_to_somas.items() if len(somas) > 1]
    print(f"Found {len(multi_soma_cell_labels)} initial segments with multiple seeds - applying heuristics...")
    max_orig_label = np.max(segmentation_mask) if unique_initial_labels.size > 0 else 0
    max_seed_label = np.max(soma_mask) if present_soma_labels.size > 0 else 0
    next_label = max(max_orig_label, max_seed_label) + 1
    print(f"Starting label for new segments: {next_label}")

    # --- Separation Loop with Heuristics & Post-Merge ---
    skipped_count = 0
    processed_count = 0
    for cell_label in tqdm(multi_soma_cell_labels, desc="Separating Segments"):
        processed_count += 1
        cell_mask = segmentation_mask == cell_label
        props = regionprops(cell_mask.astype(np.uint8));
        if not props: continue
        prop = props[0]; bbox = prop.bbox; pad = 5
        z_min=max(0,bbox[0]-pad); y_min=max(0,bbox[1]-pad); x_min=max(0,bbox[2]-pad)
        z_max=min(segmentation_mask.shape[0],bbox[3]+pad); y_max=min(segmentation_mask.shape[1],bbox[4]+pad); x_max=min(segmentation_mask.shape[2],bbox[5]+pad)
        slice_obj=np.s_[z_min:z_max, y_min:y_max, x_min:x_max]; offset=(z_min,y_min,x_min)
        cell_mask_sub = cell_mask[slice_obj]
        if cell_mask_sub.size == 0 or not np.any(cell_mask_sub): continue

        # Get seeds within this subvolume
        soma_sub = soma_mask[slice_obj]
        cell_soma_sub_mask = np.zeros_like(soma_sub, dtype=np.int32)
        cell_soma_sub_mask[cell_mask_sub] = soma_sub[cell_mask_sub]
        soma_labels_in_cell = np.unique(cell_soma_sub_mask); soma_labels_in_cell = soma_labels_in_cell[soma_labels_in_cell > 0]

        if len(soma_labels_in_cell) <= 1: continue # Safety check

        # --- Seed Merging Heuristic (Pathfinding) ---
        # Declare variables used across try/finally
        seed_props = None; adj = None; intensity_sub = None; cost_array = None
        distance_transform = None # Needed for post-merge as well
        try:
            seed_props = regionprops(cell_soma_sub_mask, intensity_image=intensity_volume[slice_obj])
            seed_prop_dict = {prop.label: prop for prop in seed_props}

            num_seeds = len(soma_labels_in_cell)
            seed_indices = {lbl: i for i, lbl in enumerate(soma_labels_in_cell)}
            adj = np.zeros((num_seeds, num_seeds), dtype=bool)
            merge_candidates = False

            # Prepare intensity subvolume and cost array for pathfinding
            intensity_sub = intensity_volume[slice_obj].astype(np.float32)
            if np.any(cell_mask_sub):
                # Inverse intensity cost: lower cost for brighter voxels. Add small epsilon.
                max_I_sub = np.max(intensity_sub[cell_mask_sub]) if np.any(intensity_sub[cell_mask_sub]) else 1.0
                cost_array = np.full(intensity_sub.shape, np.inf, dtype=np.float32) # Initialize with infinity
                # Calculate cost only within the cell mask to guide path finding
                cost_array[cell_mask_sub] = (max_I_sub - intensity_sub[cell_mask_sub]) + 1e-6
            else:
                 cost_array = np.full(intensity_sub.shape, np.inf, dtype=np.float32) # All inf if no cell mask


            # Check pairs
            for i in range(num_seeds):
                for j in range(i + 1, num_seeds):
                    label1 = soma_labels_in_cell[i]
                    label2 = soma_labels_in_cell[j]
                    prop1 = seed_prop_dict.get(label1)
                    prop2 = seed_prop_dict.get(label2)
                    if prop1 is None or prop2 is None: continue

                    # 1. Check centroid distance (physical units)
                    cent1_phys = np.array(prop1.centroid) * np.array(spacing)
                    cent2_phys = np.array(prop2.centroid) * np.array(spacing)
                    dist = np.linalg.norm(cent1_phys - cent2_phys) # Use linalg.norm

                    if dist > max_seed_centroid_dist: continue

                    # 2. Check intensity along PATH (relative median)
                    coords1 = tuple(int(round(c)) for c in prop1.centroid)
                    coords2 = tuple(int(round(c)) for c in prop2.centroid)
                    coords1_bounded = tuple(np.clip(c, 0, s-1) for c, s in zip(coords1, intensity_sub.shape))
                    coords2_bounded = tuple(np.clip(c, 0, s-1) for c, s in zip(coords2, intensity_sub.shape))

                    median_intensity_on_path = 0 # Default to low value
                    try:
                        # Find path on cost array (favors bright intensity)
                        indices, weight = route_through_array(cost_array, coords1_bounded, coords2_bounded, fully_connected=True)
                        if indices.shape[0] > 0:
                             path_indices_tuple = tuple(indices.T)
                             path_intensities = intensity_sub[path_indices_tuple]
                             if path_intensities.size > 0:
                                 median_intensity_on_path = np.median(path_intensities)
                    except ValueError as e_path: # Handles "No Path Found"
                         #print(f"    Warn: Pathfinding failed for {label1}-{label2} in cell {cell_label}: {e_path}")
                         median_intensity_on_path = 0 # Assume bad path if pathfinding fails

                    # Reference intensity: max within the two seeds
                    max_intensity_in_seeds = max(prop1.max_intensity, prop2.max_intensity) if prop1.max_intensity > 0 and prop2.max_intensity > 0 else 1.0
                    if max_intensity_in_seeds < 1e-6 : max_intensity_in_seeds = 1.0

                    intensity_ratio = median_intensity_on_path / max_intensity_in_seeds

                    if intensity_ratio < min_path_intensity_ratio: continue

                    # If passed both distance and path intensity checks
                    adj[i, j] = adj[j, i] = True
                    merge_candidates = True
                    # print(f"    Cell {cell_label}: Seeds {label1} & {label2} candidates (Dist: {dist:.2f}, PathIntRatio: {intensity_ratio:.2f})")

            # --- Make Decision based on Connectivity ---
            watershed_markers = np.zeros_like(cell_mask_sub, dtype=np.int32) # Initialize markers
            marker_id_reverse_map = {} # WS ID -> Original Seed Label(s) tuple
            largest_soma_label = -1 # Track largest *original* seed label across all groups/seeds
            max_soma_size = -1      # Track volume of largest *original* seed

            if merge_candidates:
                n_comp, comp_labels = ndimage.label(adj) # Use label on adjacency matrix directly

                if n_comp == 1: # All seeds connected
                    print(f"    Cell {cell_label}: All {num_seeds} seeds deemed connected. Skipping split.")
                    skipped_count += 1
                    # Clean up before skipping to next cell_label
                    del cell_mask, cell_mask_sub, soma_sub, cell_soma_sub_mask, seed_props, seed_prop_dict, adj, intensity_sub, cost_array
                    gc.collect()
                    continue # Skip watershed for this cell

                # Generate Merged Markers for Watershed
                print(f"    Cell {cell_label}: Found {n_comp} groups of seeds. Merging markers...")
                current_marker_id = 1
                for group_idx in range(n_comp):
                    group_marker_id = current_marker_id
                    seed_indices_in_group = np.where(comp_labels == group_idx)[0]
                    group_mask = np.zeros_like(cell_mask_sub, dtype=bool)
                    original_labels_in_group = set()
                    for seed_idx in seed_indices_in_group:
                         original_label = soma_labels_in_cell[seed_idx]
                         original_labels_in_group.add(original_label)
                         group_mask |= (cell_soma_sub_mask == original_label)
                         # Track overall largest original seed label
                         prop = seed_prop_dict.get(original_label)
                         if prop and prop.area > max_soma_size:
                             max_soma_size = prop.area
                             largest_soma_label = original_label

                    watershed_markers[group_mask] = group_marker_id
                    marker_id_reverse_map[group_marker_id] = tuple(sorted(list(original_labels_in_group)))
                    current_marker_id += 1
                num_markers_final = n_comp

            else: # No pairs met merging criteria - use individual seeds
                 current_marker_id = 1
                 for soma_label in soma_labels_in_cell:
                     seed_region_mask = cell_soma_sub_mask == soma_label
                     watershed_markers[seed_region_mask] = current_marker_id
                     marker_id_reverse_map[current_marker_id] = (soma_label,) # Store as tuple
                     # Track overall largest original seed label
                     prop = seed_prop_dict.get(soma_label)
                     if prop and prop.area > max_soma_size:
                         max_soma_size = prop.area
                         largest_soma_label = soma_label
                     current_marker_id += 1
                 num_markers_final = num_seeds

            # --- Landscape Calculation ---
            # Calculate distance transform here, needed for watershed AND post-merge
            distance_transform = ndimage.distance_transform_edt(cell_mask_sub, sampling=spacing)
            landscape = -distance_transform
            if intensity_weight > 1e-6:
                try:
                    # Intensity term calculation (same as before)
                    intensity_values_in_cell = intensity_sub[cell_mask_sub]
                    if intensity_values_in_cell.size > 0:
                        min_I=np.min(intensity_values_in_cell); max_I=np.max(intensity_values_in_cell); range_I = max_I - min_I
                        inverted_intensity_term = np.zeros_like(distance_transform, dtype=np.float32)
                        if range_I > 1e-9: norm_inv_I=(max_I - intensity_sub[cell_mask_sub])/range_I; inverted_intensity_term[cell_mask_sub]=norm_inv_I
                        landscape += intensity_weight * inverted_intensity_term
                except Exception as e: print(f"  Warn: Failed intensity for {cell_label}: {e}"); landscape = -distance_transform # Revert if error

            landscape_input = landscape # No smoothing applied

            # --- Watershed ---
            watershed_result = watershed(landscape_input, watershed_markers, mask=cell_mask_sub, watershed_line=True)

            # --- Initial Post-processing (Label assignment, line filling, small fragment removal) ---
            temp_mask = np.zeros_like(cell_mask_sub, dtype=np.int32); used_new_labels = set()
            for marker_id in range(1, num_markers_final + 1):
                original_labels_tuple = marker_id_reverse_map.get(marker_id, tuple()) # Safe get
                # Assign original cell ID to the segment containing the largest original seed
                if largest_soma_label in original_labels_tuple:
                    final_label = cell_label
                else:
                    final_label = next_label; used_new_labels.add(final_label); next_label += 1
                temp_mask[watershed_result == marker_id] = final_label

            # Line filling (same as before)
            watershed_lines_mask = (watershed_result == 0) & cell_mask_sub
            if np.any(watershed_lines_mask):
                non_line_mask = temp_mask != 0
                if np.any(non_line_mask):
                     dist_to_labels, nearest_label_indices = ndimage.distance_transform_edt(~non_line_mask, return_indices=True, sampling=spacing)
                     nearest_labels = temp_mask[tuple(nearest_label_indices)]; temp_mask[watershed_lines_mask] = nearest_labels[watershed_lines_mask]
                else: temp_mask[cell_mask_sub] = cell_label # Assign original if only lines left

            # Small fragment merging (same as before)
            fragments_merged = True
            while fragments_merged:
                fragments_merged = False; labels_to_check = np.unique(temp_mask); labels_to_check = labels_to_check[labels_to_check > 0]
                for lbl in labels_to_check:
                    lbl_mask = temp_mask == lbl; size = np.sum(lbl_mask)
                    if 0 < size < min_size_threshold:
                        struct = ndimage.generate_binary_structure(temp_mask.ndim, 1) # Simple connectivity
                        dilated_mask = ndimage.binary_dilation(lbl_mask, structure=struct)
                        neighbor_region = dilated_mask & (~lbl_mask) & (temp_mask != 0) # Valid neighbors
                        neighbor_labels = np.unique(temp_mask[neighbor_region])
                        if len(neighbor_labels) == 0: continue
                        neighbor_sizes = [(n_lbl, np.sum(temp_mask == n_lbl)) for n_lbl in neighbor_labels]
                        if not neighbor_sizes: continue
                        largest_neighbor_label = max(neighbor_sizes, key=lambda item: item[1])[0]
                        temp_mask[lbl_mask] = largest_neighbor_label; fragments_merged = True
                        if lbl in used_new_labels: used_new_labels.discard(lbl) # Track which new labels are gone

            # --- NEW: Post-Watershed Merging based on Neck Thickness ---
            print(f"    Cell {cell_label}: Applying post-watershed merge check...")
            post_merge_occurred = False
            struct_dilate = ndimage.generate_binary_structure(temp_mask.ndim, 1) # For finding interface
            labels_after_frag_merge = np.unique(temp_mask)
            labels_after_frag_merge = labels_after_frag_merge[labels_after_frag_merge > 0]

            # Identify the main segment (retained original label) and potential fragments
            main_label = cell_label if cell_label in labels_after_frag_merge else -1
            new_labels_remaining = used_new_labels.intersection(labels_after_frag_merge)

            if main_label != -1 and new_labels_remaining:
                main_mask = (temp_mask == main_label)
                for new_lbl in new_labels_remaining:
                    new_lbl_mask = (temp_mask == new_lbl)
                    # Find interface: Dilate new label and check overlap with main label
                    dilated_new_mask = ndimage.binary_dilation(new_lbl_mask, structure=struct_dilate)
                    interface_mask = main_mask & dilated_new_mask # Voxels in main adjacent to new
                    interface_voxels_count = np.sum(interface_mask)

                    if interface_voxels_count > 0:
                        # Get distance transform values at the interface voxels within the main mask
                        # (Reflects distance from main mask edge = thickness at interface)
                        interface_dt_values = distance_transform[interface_mask]
                        min_neck_thickness = np.min(interface_dt_values) if interface_dt_values.size > 0 else 0

                        # Check criteria
                        if min_neck_thickness >= post_merge_min_neck_thickness and interface_voxels_count >= post_merge_min_interface_voxels:
                            print(f"      Merging fragment {new_lbl} into {main_label} (Neck: {min_neck_thickness:.2f} >= {post_merge_min_neck_thickness}, Interface: {interface_voxels_count} >= {post_merge_min_interface_voxels})")
                            temp_mask[new_lbl_mask] = main_label # Perform merge
                            post_merge_occurred = True
                        # else:
                        #     print(f"      NOT Merging {new_lbl} into {main_label} (Neck: {min_neck_thickness:.2f}, Interface: {interface_voxels_count})")
                    # else: print(f"      Fragment {new_lbl} not adjacent to {main_label}?")


            # Update next_label if any merges happened or just generally
            final_labels_in_mask = np.unique(temp_mask); final_labels_in_mask = final_labels_in_mask[final_labels_in_mask > 0]
            if final_labels_in_mask.size > 0: next_label = max(next_label, np.max(final_labels_in_mask) + 1)

            # --- Map Back (No change) ---
            full_subvol = separated_mask[slice_obj]; original_cell_voxels_in_sub = cell_mask_sub
            # Ensure we only write back where the original cell was
            full_subvol[original_cell_voxels_in_sub] = temp_mask[original_cell_voxels_in_sub]
            separated_mask[slice_obj] = full_subvol

            # Explicit cleanup for loop variables
            del cell_mask, cell_mask_sub, soma_sub, cell_soma_sub_mask, watershed_markers
            del distance_transform, landscape_input, watershed_result, temp_mask, full_subvol
            del marker_id_reverse_map, used_new_labels
            if seed_props is not None: del seed_props
            if adj is not None: del adj
            if intensity_sub is not None: del intensity_sub
            if cost_array is not None: del cost_array
            if 'inverted_intensity_term' in locals(): del inverted_intensity_term
            gc.collect()


        except Exception as e_outer:
             print(f"ERROR during heuristic/watershed/post-merge processing for cell {cell_label}: {e_outer}")
             # Ensure basic cleanup even on outer error
             if 'cell_mask' in locals(): del cell_mask
             if 'cell_mask_sub' in locals(): del cell_mask_sub
             # Clean up others if they exist
             if distance_transform is not None: del distance_transform
             if seed_props is not None: del seed_props
             if adj is not None: del adj
             if intensity_sub is not None: del intensity_sub
             if cost_array is not None: del cost_array
             gc.collect()
             continue # Skip this cell

    print(f"Skipped splitting {skipped_count} cells due to seed merging heuristics.")
    print("--- Finished Multi-Soma Separation ---")
    del cell_to_somas; gc.collect()
    return separated_mask

# --- END OF FILE ramified_segmenter.py ---