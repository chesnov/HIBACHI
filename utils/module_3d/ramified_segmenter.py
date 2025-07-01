import numpy as np # type: ignore
from scipy import ndimage, sparse # type: ignore
from tqdm import tqdm # type: ignore
from shutil import rmtree
import gc
import time # For profiling
from functools import partial
from skimage.measure import regionprops, label as skimage_label # type: ignore
from skimage.segmentation import watershed # type: ignore
from skimage.graph import route_through_array # type: ignore
from skimage.morphology import binary_erosion, binary_dilation, ball # type: ignore
import math
from sklearn.decomposition import PCA # type: ignore
from skimage.feature import peak_local_max # type: ignore
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from skimage.segmentation import relabel_sequential # type: ignore
import traceback # For detailed error logging
from skimage.measure import label, regionprops # type: ignore
from collections import deque
from scipy.ndimage import binary_fill_holes, find_objects # type: ignore
from skimage.morphology import binary_dilation, ball, footprint_rectangle # type: ignore
from scipy.ndimage import binary_fill_holes, find_objects, gaussian_filter, distance_transform_edt # type: ignore
import os
from shutil import rmtree

seed = 42
np.random.seed(seed)         # For NumPy


# Helper function for physical size based min_distance (keep as is)
def get_min_distance_pixels(spacing, physical_distance):
    """Calculates minimum distance in pixels for peak_local_max based on physical distance."""
    min_spacing_yx = min(spacing[1:]) # Use YX for in-plane separation
    if min_spacing_yx <= 1e-6: return 3
    pixels = int(round(physical_distance / min_spacing_yx))
    return max(3, pixels)


# --- Helper function for filtering candidate fragments (3D) ---
def _filter_candidate_fragment_3d(
    fragment_mask: np.ndarray, # sub-mask (boolean)
    parent_dt: np.ndarray, # OPTIMIZED: DT of the core this fragment came from
    spacing: Tuple[float, float, float],
    min_seed_fragment_volume: int,
    min_accepted_core_volume: float,
    max_accepted_core_volume: float,
    min_accepted_thickness_um: float, # From fragment's own DT, in um
    max_accepted_thickness_um: float, # From fragment's own DT, in um
    max_allowed_core_aspect_ratio: float
) -> Tuple[str, Optional[str], Optional[np.ndarray], Optional[float], float]:
    """
    Filters a single 3D candidate fragment and returns a detailed status.
    Returns: (status, reason, mask_copy, volume, duration)
    - status: 'valid', 'fallback', or 'discard'.
    - reason: The specific reason for fallback/discard (e.g., 'thickness', 'aspect').
    """
    t_start = time.time()
    
    try:
        fragment_volume = np.sum(fragment_mask)
        if fragment_volume < min_seed_fragment_volume:
            return 'discard', 'too_small', None, None, time.time() - t_start

        # --- Thickness Check (Already Spacing-Aware) ---
        max_dist_in_fragment_um = np.max(parent_dt[fragment_mask])
        passes_thickness = (min_accepted_thickness_um <= max_dist_in_fragment_um <= max_accepted_thickness_um)
        
        # --- Volume Check ---
        passes_volume_range = (min_accepted_core_volume <= fragment_volume <= max_accepted_core_volume)

        # --- Aspect Ratio Check (NOW SPACING-AWARE) ---
        passes_aspect = True
        if fragment_volume > 3: # Need more than 3 points for PCA
            try:
                # Get voxel coordinates and convert to physical coordinates
                coords_vox = np.argwhere(fragment_mask)
                coords_phys = coords_vox * np.array(spacing)

                # Use PCA on physical coordinates to get axis lengths
                pca = PCA(n_components=3)
                pca.fit(coords_phys)
                eigenvalues = pca.explained_variance_ # These are variances along principal axes
                eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
                
                if eigenvalues_sorted[2] > 1e-12: # Avoid division by zero
                    aspect_ratio = math.sqrt(eigenvalues_sorted[0]) / math.sqrt(eigenvalues_sorted[2])
                    if aspect_ratio > max_allowed_core_aspect_ratio:
                        passes_aspect = False
            except Exception:
                passes_aspect = True
        
        fragment_mask_copy = fragment_mask.copy()
        duration = time.time() - t_start

        if passes_thickness and passes_volume_range and passes_aspect:
            return 'valid', None, fragment_mask_copy, fragment_volume, duration
        else:
            # Determine the first reason for failure
            reason = 'unknown'
            if not passes_thickness: reason = 'thickness'
            elif not passes_volume_range: reason = 'volume'
            elif not passes_aspect: reason = 'aspect'
            return 'fallback', reason, fragment_mask_copy, fragment_volume, duration
            
    except Exception as e_filt:
        print(f"Warn: Unexpected error during 3D fragment filtering: {e_filt}")
        return 'discard', 'error', None, None, time.time() - t_start


def extract_soma_masks(
    segmentation_mask: np.ndarray, # 3D mask
    intensity_image: np.ndarray, # 3D intensity image
    spacing: Optional[Tuple[float, float, float]],
    smallest_quantile: int = 25,
    min_fragment_size: int = 30, # Voxels
    core_volume_target_factor_lower: float = 0.1,
    core_volume_target_factor_upper: float = 10,
    erosion_iterations: int = 0,
    ratios_to_process = [0.3, 0.4, 0.5, 0.6],
    intensity_percentiles_to_process: List[int] = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
    min_physical_peak_separation: float = 7.0, # um
    max_allowed_core_aspect_ratio: float = 10.0,
    ref_vol_percentile_lower: int = 30,
    ref_vol_percentile_upper: int = 70,
    ref_thickness_percentile_lower: int = 1,
    absolute_min_thickness_um: float = 1.5,
    absolute_max_thickness_um: float = 10.0
) -> np.ndarray:
    """
    Generates candidate seed mask using a memory-efficient strategy by storing
    candidate coordinates and provides detailed performance and rejection profiling.
    """
    print("--- Starting 3D Soma Extraction (Coordinate Storage Strategy with Rejection Profiling) ---")

    print(f"DT ratios: {ratios_to_process}")
    intensity_percentiles_to_process = sorted([p for p in intensity_percentiles_to_process if 0 < p < 100], reverse=True)
    print(f"Intensity Percentiles: {intensity_percentiles_to_process}")

    # --- Internal Parameters & Setup ---
    if min_fragment_size is None or min_fragment_size < 1:
        min_seed_fragment_volume = 30
    else:
        min_seed_fragment_volume = min_fragment_size
    MAX_CORE_VOXELS_FOR_WS = 500_000 # SAFETY VALVE
    print(f"Watershed safety valve: cores > {MAX_CORE_VOXELS_FOR_WS} voxels will not be split.")

    # --- Setup block ---
    if spacing is None: spacing = (1.0, 1.0, 1.0)
    else:
        try: spacing = tuple(float(s) for s in spacing); assert len(spacing) == 3
        except: spacing = (1.0, 1.0, 1.0)
    print(f"Using 3D spacing (z,y,x): {spacing}")
    unique_labels_all, counts = np.unique(segmentation_mask[segmentation_mask > 0], return_counts=True)
    if len(unique_labels_all) == 0: return np.zeros_like(segmentation_mask, dtype=np.int32)
    initial_volumes = dict(zip(unique_labels_all, counts))
    label_to_slice: Dict[int, Tuple[slice, ...]] = {}
    valid_unique_labels_list: List[int] = []
    object_slices = ndimage.find_objects(segmentation_mask)
    if object_slices:
        for label in unique_labels_all:
            idx = label - 1
            if 0 <= idx < len(object_slices) and object_slices[idx]:
                s = object_slices[idx]
                if len(s) == 3 and all(si.start < si.stop for si in s):
                    label_to_slice[label] = s; valid_unique_labels_list.append(label)
    if not valid_unique_labels_list: return np.zeros_like(segmentation_mask, dtype=np.int32)
    unique_labels = np.array(valid_unique_labels_list)
    volumes = {label: initial_volumes[label] for label in unique_labels}; all_volumes_list = list(volumes.values())
    if not all_volumes_list: return np.zeros_like(segmentation_mask, dtype=np.int32)
    smallest_thresh_volume = np.percentile(all_volumes_list, smallest_quantile) if len(all_volumes_list) > 1 else (all_volumes_list[0] + 1 if all_volumes_list else 1)
    smallest_object_labels_set = {label for label, vol in volumes.items() if vol <= smallest_thresh_volume}
    target_soma_volume = np.median([v for l,v in volumes.items() if l in smallest_object_labels_set] or all_volumes_list)
    min_accepted_core_volume = max(min_seed_fragment_volume, target_soma_volume * core_volume_target_factor_lower)
    max_accepted_core_volume = target_soma_volume * core_volume_target_factor_upper
    print(f"Core Volume filter range: [{min_accepted_core_volume:.2f} - {max_accepted_core_volume:.2f}] voxels")
    vol_thresh_lower, vol_thresh_upper = np.percentile(all_volumes_list, ref_vol_percentile_lower), np.percentile(all_volumes_list, ref_vol_percentile_upper)
    reference_object_labels = [l for l in unique_labels if vol_thresh_lower < volumes[l] <= vol_thresh_upper]
    if len(reference_object_labels) < 5: reference_object_labels = list(unique_labels)
    max_thicknesses_um = []
    for label_ref in tqdm(reference_object_labels, desc="Calc Ref Thickness", disable=len(reference_object_labels) < 10):
        bbox_slice = label_to_slice.get(label_ref);
        if bbox_slice:
            mask_sub = (segmentation_mask[bbox_slice] == label_ref)
            if np.any(mask_sub):
                dt = ndimage.distance_transform_edt(mask_sub, sampling=spacing)
                max_thicknesses_um.append(np.max(dt))
    min_accepted_thickness_um = max(absolute_min_thickness_um, np.percentile(max_thicknesses_um, ref_thickness_percentile_lower)) if max_thicknesses_um else absolute_min_thickness_um
    min_accepted_thickness_um = min(min_accepted_thickness_um, absolute_max_thickness_um - 1e-6)
    print(f"Final Core Thickness filter range: [{min_accepted_thickness_um:.2f} - {absolute_max_thickness_um:.2f}] um")
    min_peak_sep_pixels = get_min_distance_pixels(spacing, min_physical_peak_separation)
    gc.collect()

    # --- 2. Process Small Cells & Initialize Final Mask ---
    final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32); next_final_label = 1; added_small_labels: Set[int] = set()
    small_cell_labels = [l for l in unique_labels if l in smallest_object_labels_set]
    print(f"\nProcessing {len(small_cell_labels)} small objects (3D)...")
    for label in tqdm(small_cell_labels, desc="Small Cells", disable=len(small_cell_labels) < 10):
        if volumes.get(label, 0) >= min_seed_fragment_volume:
            coords = np.argwhere(segmentation_mask == label)
            if coords.size > 0:
                final_seed_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = next_final_label
                next_final_label += 1; added_small_labels.add(label)
    print(f"Added {len(added_small_labels)} initial seeds from small cells."); gc.collect()

    # --- 3. Generate Candidates from Large Cells ---
    large_cell_labels = [l for l in unique_labels if l not in added_small_labels]
    print(f"Generating and placing candidates from {len(large_cell_labels)} large objects (3D)...")
    struct_el_erosion = ndimage.generate_binary_structure(3, 3) if erosion_iterations > 0 else None

    for label in tqdm(large_cell_labels, desc="Large Cell Candidates"):
        bbox_slice = label_to_slice.get(label)
        if not bbox_slice: continue
        
        valid_candidates_for_this_label: List[Dict[str, Any]] = []
        fallback_candidates_for_this_label: List[Dict[str, Any]] = []
        seg_sub, mask_sub, int_sub, dist_transform_obj = None, None, None, None

        try:
            pad = 1
            slice_obj_tuple = (slice(max(0, s.start-pad), min(sh, s.stop+pad)) for s, sh in zip(bbox_slice, segmentation_mask.shape))
            slice_obj = tuple(slice_obj_tuple)
            offset = (slice_obj[0].start, slice_obj[1].start, slice_obj[2].start)

            seg_sub = segmentation_mask[slice_obj]
            mask_sub = (seg_sub == label)
            if not np.any(mask_sub): continue
            
            def process_core_mask(core_mask_for_labeling: np.ndarray, time_log: Dict[str, float]) -> Tuple[int, int, Dict[str, int]]:
                rejection_stats = {'too_small': 0, 'thickness': 0, 'volume': 0, 'aspect': 0, 'error': 0}
                t_start = time.time(); labeled_cores, num_cores = ndimage.label(core_mask_for_labeling); time_log['labeling'] += time.time() - t_start
                if num_cores == 0: return 0, 0, rejection_stats

                t_start = time.time(); core_volumes = ndimage.sum_labels(core_mask_for_labeling, labeled_cores, range(1, num_cores + 1)); time_log['summing'] += time.time() - t_start
                
                kept_fragments = 0
                for core_idx in np.where(core_volumes >= min_seed_fragment_volume)[0]:
                    core_lbl = core_idx + 1
                    core_component_mask = (labeled_cores == core_lbl)
                    core_volume = core_volumes[core_idx]
                    
                    t_ws_start = time.time()
                    frags_masks, dt_core = [], None
                    if core_volume > MAX_CORE_VOXELS_FOR_WS:
                        frags_masks.append(core_component_mask)
                        dt_core = ndimage.distance_transform_edt(core_component_mask, sampling=spacing)
                    else:
                        dt_core = ndimage.distance_transform_edt(core_component_mask, sampling=spacing)
                        peaks = peak_local_max(dt_core, min_distance=min_peak_sep_pixels, labels=core_component_mask, exclude_border=False)
                        if peaks.shape[0] > 1:
                            markers = np.zeros(dt_core.shape, dtype=np.int32); markers[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                            ws_core = watershed(-dt_core, markers, mask=core_component_mask, watershed_line=False)
                            for l in np.unique(ws_core):
                                if l > 0: frags_masks.append(ws_core == l)
                        else: frags_masks.append(core_component_mask)
                    time_log['mini_ws'] += time.time() - t_ws_start
                    
                    for f_mask in frags_masks:
                        status, reason, m_copy, vol, dur = _filter_candidate_fragment_3d(f_mask, dt_core, spacing, min_seed_fragment_volume, min_accepted_core_volume, max_accepted_core_volume, min_accepted_thickness_um, absolute_max_thickness_um, max_allowed_core_aspect_ratio)
                        time_log['filtering'] += dur
                        
                        if status == 'valid':
                            coords = np.argwhere(m_copy)
                            valid_candidates_for_this_label.append({'coords': coords, 'volume': vol, 'offset': offset})
                            kept_fragments += 1
                        elif status == 'fallback':
                            coords = np.argwhere(m_copy)
                            fallback_candidates_for_this_label.append({'coords': coords, 'volume': vol, 'offset': offset})
                            if reason: rejection_stats[reason] += 1
                        elif status == 'discard' and reason:
                            rejection_stats[reason] += 1
                        
                        if m_copy is not None: del m_copy

                return num_cores, kept_fragments, rejection_stats
            
            # --- A) DT Path ---
            dist_transform_obj = ndimage.distance_transform_edt(mask_sub, sampling=spacing)
            max_dist = np.max(dist_transform_obj)
            if max_dist > 1e-9:
                for ratio in ratios_to_process:
                    t_iter_start = time.time(); time_log = {'labeling': 0, 'summing': 0, 'mini_ws': 0, 'filtering': 0}
                    initial_core = (dist_transform_obj >= max_dist * ratio) & mask_sub
                    if not np.any(initial_core): continue
                    eroded_core = ndimage.binary_erosion(initial_core, footprint=struct_el_erosion, iterations=erosion_iterations) if erosion_iterations > 0 else initial_core
                    if np.any(eroded_core):
                        num_found, num_kept, stats = process_core_mask(eroded_core, time_log)
                        total_time = time.time() - t_iter_start
                        rejection_str = ", ".join([f"rej {k}: {v}" for k, v in stats.items() if v > 0])
                        print(f"    L{label} DT R{ratio:.1f}: Found {num_found} cores, kept {num_kept} frags. "
                              f"{rejection_str}. Total time: {total_time:.2f}s "
                              f"[label:{time_log['labeling']:.2f}s, sum:{time_log['summing']:.2f}s, "
                              f"ws:{time_log['mini_ws']:.2f}s, filter:{time_log['filtering']:.2f}s]")

            # --- B) Intensity Path ---
            int_sub = intensity_image[slice_obj]
            ints_obj_vals = int_sub[mask_sub]
            if ints_obj_vals.size > 0:
                for perc in intensity_percentiles_to_process:
                    t_iter_start = time.time(); time_log = {'labeling': 0, 'summing': 0, 'mini_ws': 0, 'filtering': 0}
                    thresh = np.percentile(ints_obj_vals, perc)
                    initial_core = (int_sub >= thresh) & mask_sub
                    if not np.any(initial_core): continue
                    eroded_core = ndimage.binary_erosion(initial_core, footprint=struct_el_erosion, iterations=erosion_iterations) if erosion_iterations > 0 else initial_core
                    if np.any(eroded_core):
                        num_found, num_kept, stats = process_core_mask(eroded_core, time_log)
                        total_time = time.time() - t_iter_start
                        rejection_str = ", ".join([f"rej {k}: {v}" for k, v in stats.items() if v > 0])
                        print(f"    L{label} INT P{perc:02d}: Found {num_found} cores, kept {num_kept} frags. "
                              f"{rejection_str}. Total time: {total_time:.2f}s "
                              f"[label:{time_log['labeling']:.2f}s, sum:{time_log['summing']:.2f}s, "
                              f"ws:{time_log['mini_ws']:.2f}s, filter:{time_log['filtering']:.2f}s]")

            # --- Placement logic (run once per parent cell) ---
            candidates_to_place = []
            if valid_candidates_for_this_label: candidates_to_place.extend(valid_candidates_for_this_label)
            elif fallback_candidates_for_this_label: candidates_to_place.append(max(fallback_candidates_for_this_label, key=lambda x: x['volume']))
            
            if candidates_to_place:
                candidates_to_place.sort(key=lambda x: x['volume'])
                for cand in candidates_to_place:
                    coords_sub, offset_cand = cand['coords'], cand['offset']
                    if coords_sub.size == 0: continue
                    gz, gy, gx = coords_sub[:, 0] + offset_cand[0], coords_sub[:, 1] + offset_cand[1], coords_sub[:, 2] + offset_cand[2]
                    vg = (gz, gy, gx)
                    if np.all(final_seed_mask[vg] == 0):
                        final_seed_mask[vg] = next_final_label
                        next_final_label += 1

        except MemoryError: print(f"CRITICAL: MemoryError processing L{label}. Skipping this cell."); gc.collect()
        except Exception as e: print(f"Warn: Uncaught Error L{label}: {e}"); traceback.print_exc()
        finally:
            del valid_candidates_for_this_label, fallback_candidates_for_this_label
            if seg_sub is not None: del seg_sub
            if mask_sub is not None: del mask_sub
            if int_sub is not None: del int_sub
            if dist_transform_obj is not None: del dist_transform_obj
            gc.collect()

    total_final_seeds = next_final_label - 1
    print(f"\nGenerated {total_final_seeds} final aggregated seeds (3D)."); print("--- Finished 3D Intermediate Seed Extraction ---")
    return final_seed_mask

# --- Helper: Analyze local intensity difference ---
def _analyze_local_intensity_difference_hybrid(
    region1_mask: np.ndarray, region2_mask: np.ndarray, parent_cell_mask: np.ndarray,
    intensity_vol_local: np.ndarray, local_analysis_radius: int,
    min_local_intensity_difference_threshold: float, log_prefix: str = "    [LID]"
) -> bool:
    t_start_lid = time.time()
    print(f"{log_prefix} Start LID. R1:{np.sum(region1_mask)}, R2:{np.sum(region2_mask)}, Parent:{np.sum(parent_cell_mask)}, Radius:{local_analysis_radius}, Thresh:{min_local_intensity_difference_threshold:.3f}")
    footprint_elem = ball(local_analysis_radius) if local_analysis_radius > 1 else footprint_rectangle((3,3,3))

    r1m, r2m, pcm = region1_mask.astype(bool), region2_mask.astype(bool), parent_cell_mask.astype(bool)
    
    dr1 = binary_dilation(r1m, footprint=footprint_elem)
    dr2 = binary_dilation(r2m, footprint=footprint_elem)

    # Define the zone where dilated regions overlap within the parent mask
    interface_overlap_zone = dr1 & dr2 & pcm
    
    # Local region 1: Part of the overlap zone, but strictly outside r1 itself.
    # These are voxels effectively "on r2's side" of the r1/r2 interface.
    la_r1 = interface_overlap_zone & ~r1m 
    
    # Local region 2: Part of the overlap zone, but strictly outside r2 itself.
    # These are voxels "on r1's side" of the r1/r2 interface.
    la_r2 = interface_overlap_zone & ~r2m
    la_r1 = dr1 & dr2 & pcm & ~r1m & dr2 # Voxels near r1's boundary but in r2's dilation, within parent, outside r1
    la_r2 = dr1 & dr2 & pcm & ~r2m & dr1 # Voxels near r2's boundary but in r1's dilation, within parent, outside r2

    sa1, sa2 = np.sum(la_r1), np.sum(la_r2)
    print(f"{log_prefix} Adj. region sizes: LA_R1={sa1}, LA_R2={sa2}")
    MIN_ADJ_SIZE = 20
    if sa1 < MIN_ADJ_SIZE or sa2 < MIN_ADJ_SIZE:
        print(f"{log_prefix} Small adj. regions. LID returns False. Time:{time.time()-t_start_lid:.3f}s")
        return False
    i1, i2 = intensity_vol_local[la_r1], intensity_vol_local[la_r2]
    m1, m2 = (np.mean(i1) if i1.size > 0 else 0), (np.mean(i2) if i2.size > 0 else 0)
    print(f"{log_prefix} Adj. means: M1={m1:.2f}, M2={m2:.2f}")
    ref_i = max(m1, m2)
    if ref_i < 1e-6: 
        print(f"{log_prefix} Ref intensity near zero. LID True. Time:{time.time()-t_start_lid:.3f}s"); 
        return True
    rel_d = abs(m1 - m2) / ref_i
    is_valid = rel_d >= min_local_intensity_difference_threshold
    print(f"{log_prefix} Rel.Diff={rel_d:.3f} (Thresh {min_local_intensity_difference_threshold:.3f}). Valid={is_valid}. Time:{time.time()-t_start_lid:.3f}s")
    return is_valid

def _calculate_interface_metrics(
    mask_A_local: np.ndarray, # Boolean mask of segment A in its local crop
    mask_B_local: np.ndarray, # Boolean mask of segment B in its local crop
    parent_mask_local: np.ndarray, # Boolean mask of the original cell these segments belong to
    intensity_local: np.ndarray, # Intensity image for the crop
    avg_soma_intensity_for_interface: float, # Avg intensity of somas seeding A & B
    spacing_tuple: Tuple[float, float, float],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float, # Pass this for comparison printout
    log_prefix: str = "      [IfaceMetrics]"
) -> Dict[str, Any]:
    """
    Calculates metrics for the interface between two adjacent segments A and B.
    Now includes mean_interface_intensity and ratio checks.
    """
    metrics = {
        'interface_voxel_count': 0,
        'mean_interface_intensity': 0.0,
        'interface_intensity_ratio': float('inf'), # High if no interface or low soma intensity
        'ratio_threshold_passed': False, # Did it pass the min_path_intensity_ratio check? (i.e., dark enough)
        'lid_passed_separation': True, # Assume distinct unless LID fails (i.e., True if different enough)
        'should_merge_decision': False # Final decision based on heuristics
    }
    if not np.any(mask_A_local) or not np.any(mask_B_local):
        print(f"{log_prefix} One or both masks empty, cannot calculate metrics.")
        return metrics

    footprint_dilation = footprint_rectangle((3,3,3))

    dilated_A = binary_dilation(mask_A_local, footprint=footprint_dilation)
    interface_A_B = dilated_A & mask_B_local 

    dilated_B = binary_dilation(mask_B_local, footprint=footprint_dilation)
    interface_B_A = dilated_B & mask_A_local 

    combined_interface_mask = (interface_A_B | interface_B_A) & parent_mask_local
    metrics['interface_voxel_count'] = np.sum(combined_interface_mask)
    print(f"{log_prefix} Interface Voxel Count: {metrics['interface_voxel_count']}")

    if metrics['interface_voxel_count'] > 0:
        interface_intensity_values = intensity_local[combined_interface_mask]
        if interface_intensity_values.size > 0:
            metrics['mean_interface_intensity'] = np.mean(interface_intensity_values)
        
        if avg_soma_intensity_for_interface > 1e-6:
            metrics['interface_intensity_ratio'] = metrics['mean_interface_intensity'] / avg_soma_intensity_for_interface
        
        # Check against min_path_intensity_ratio heuristic
        # A "good" separation has a LOW ratio (dark interface compared to somas)
        metrics['ratio_threshold_passed'] = metrics['interface_intensity_ratio'] < min_path_intensity_ratio_heuristic
        
        print(f"{log_prefix} Mean Interface Intensity: {metrics['mean_interface_intensity']:.2f}")
        print(f"{log_prefix} Avg Soma Intensity Ref: {avg_soma_intensity_for_interface:.2f}")
        print(f"{log_prefix} Interface/Soma Intensity Ratio: {metrics['interface_intensity_ratio']:.3f} (Threshold for merge if >= {min_path_intensity_ratio_heuristic:.3f})")
        if metrics['ratio_threshold_passed']:
            print(f"{log_prefix} --> Path Ratio Check: PASSED (interface is dark enough relative to somas)")
        else:
            print(f"{log_prefix} --> Path Ratio Check: FAILED (interface is too bright, suggests merge)")


        metrics['lid_passed_separation'] = _analyze_local_intensity_difference_hybrid(
            mask_A_local, mask_B_local, parent_mask_local, intensity_local,
            local_analysis_radius, min_local_intensity_difference,
            log_prefix=log_prefix + " LID"
        )
        if metrics['lid_passed_separation']:
            print(f"{log_prefix} --> LID Check: PASSED (segments are locally distinct)")
        else:
            print(f"{log_prefix} --> LID Check: FAILED (segments are too similar locally, suggests merge)")
        
        # Decision to merge:
        # Merge if EITHER the interface is too bright (FAILS ratio_threshold_passed)
        # OR if the local regions are too similar (FAILS lid_passed_separation)
        if not metrics['ratio_threshold_passed'] or not metrics['lid_passed_separation']:
            metrics['should_merge_decision'] = True
            print(f"{log_prefix} ===> Merge Decision: YES (RatioFail: {not metrics['ratio_threshold_passed']}, LIDFail: {not metrics['lid_passed_separation']})")
        else:
            metrics['should_merge_decision'] = False
            print(f"{log_prefix} ===> Merge Decision: NO (Separation seems valid by both heuristics)")

    else: # No direct interface voxels found
        print(f"{log_prefix} No interface voxels found between segments. Defaulting to NO MERGE.")
        metrics['ratio_threshold_passed'] = True # No interface to be "too bright"
        metrics['lid_passed_separation'] = True  # No interface for LID to fail on
        metrics['should_merge_decision'] = False

    return metrics

def _build_adjacency_graph_for_cell(
    current_cell_segments_mask_local: np.ndarray, 
    original_cell_mask_local: np.ndarray,
    soma_mask_local: np.ndarray, # Original full soma mask, in local crop
    soma_props_for_cell: Dict[int, Dict[str, Any]], # Pre-calculated props for ORIGINAL somas in this cell's crop
    intensity_local: np.ndarray, 
    spacing_tuple: Tuple[float, float, float],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float, # Pass this for comparison printout in metrics
    log_prefix: str = "    [GraphBuild]"
) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]]]:
    print(f"{log_prefix} Building graph. SegMaskShape:{current_cell_segments_mask_local.shape}")
    nodes: Dict[int, Dict[str, Any]] = {}
    edges: Dict[Tuple[int, int], Dict[str, Any]] = {}
    seg_lbls = np.unique(current_cell_segments_mask_local[current_cell_segments_mask_local > 0])
    print(f"{log_prefix} Found {len(seg_lbls)} P1 segments: {seg_lbls}")

    if len(seg_lbls) <= 1:
        # ... (single node creation logic - same as before) ...
        if len(seg_lbls) == 1:
            lbl = seg_lbls[0]; m_node = (current_cell_segments_mask_local == lbl)
            s_in_node_vals = soma_mask_local[m_node] # Values from soma_mask_local
            s_in_node_unique = np.unique(s_in_node_vals[s_in_node_vals > 0]) # Unique positive labels
            s_in_node = sorted([s_lbl for s_lbl in s_in_node_unique if s_lbl in soma_props_for_cell]) # Filter by those in props

            obj_sl = find_objects(m_node.astype(np.int32))
            nodes[lbl] = {'mask_bbox_local': obj_sl[0] if obj_sl and obj_sl[0] else None, 
                          'orig_somas': s_in_node, 'volume': np.sum(m_node)}
        print(f"{log_prefix} <=1 segment. Graph: {len(nodes)} nodes, {len(edges)} edges.")
        return nodes, edges
    
    footprint_d_graph = footprint_rectangle((3,3,3)) # Renamed

    for lbl_node in seg_lbls: # Renamed lbl to lbl_node
        m_node_build = (current_cell_segments_mask_local == lbl_node) # Renamed
        if not np.any(m_node_build): continue
        
        s_in_node_vals_build = soma_mask_local[m_node_build] # Renamed
        s_in_node_unique_build = np.unique(s_in_node_vals_build[s_in_node_vals_build > 0]) # Renamed
        # Filter somas to only those for which we have properties (valid original somas)
        orig_somas_list_build = sorted([s_lbl for s_lbl in s_in_node_unique_build if s_lbl in soma_props_for_cell]) # Renamed
        
        obj_sl_build = find_objects(m_node_build.astype(np.int32)) # Renamed
        nodes[lbl_node] = {
            'mask_bbox_local': obj_sl_build[0] if obj_sl_build and obj_sl_build[0] else None,
            'orig_somas': orig_somas_list_build, 
            'volume': np.sum(m_node_build)
        }
        print(f"{log_prefix}   Node {lbl_node}: Vol={nodes[lbl_node]['volume']}, OrigSomas={nodes[lbl_node]['orig_somas']}")

    for i_idx_graph in range(len(seg_lbls)): # Renamed i_idx
        lbl_A_graph = seg_lbls[i_idx_graph] # Renamed
        mask_A_graph = (current_cell_segments_mask_local == lbl_A_graph) # Renamed
        if not np.any(mask_A_graph) or lbl_A_graph not in nodes: continue # Ensure node_A exists

        dil_A_graph = binary_dilation(mask_A_graph, footprint=footprint_d_graph) # Renamed
        pot_neigh_mask_graph = dil_A_graph & original_cell_mask_local & \
                               (current_cell_segments_mask_local != lbl_A_graph) & \
                               (current_cell_segments_mask_local > 0) # Renamed
        if not np.any(pot_neigh_mask_graph): continue
        
        neigh_lbls_graph = np.unique(current_cell_segments_mask_local[pot_neigh_mask_graph]) # Renamed
        for lbl_B_graph in neigh_lbls_graph: # Renamed
            if lbl_B_graph <= lbl_A_graph or lbl_B_graph not in nodes: continue # Avoid duplicates and ensure node_B exists
            
            mask_B_graph = (current_cell_segments_mask_local == lbl_B_graph) # Renamed
            if not np.any(mask_B_graph): continue

            edge_key_graph = tuple(sorted((lbl_A_graph, lbl_B_graph))) # Renamed
            if edge_key_graph in edges: continue

            print(f"{log_prefix}   Checking interface: {edge_key_graph}")
            
            # Calculate avg_soma_intensity for this specific interface
            somas_A = nodes[lbl_A_graph].get('orig_somas', [])
            somas_B = nodes[lbl_B_graph].get('orig_somas', [])
            combined_interface_somas = list(set(somas_A + somas_B))
            
            interface_soma_intensities = [soma_props_for_cell[s_lbl]['mean_intensity'] 
                                          for s_lbl in combined_interface_somas 
                                          if s_lbl in soma_props_for_cell and 'mean_intensity' in soma_props_for_cell[s_lbl]]
            
            avg_soma_intensity_for_this_interface = np.mean(interface_soma_intensities) if interface_soma_intensities else 1.0 # Default to 1.0 to avoid div by zero
            if avg_soma_intensity_for_this_interface < 1e-6 : avg_soma_intensity_for_this_interface = 1.0
            
            print(f"{log_prefix}     Edge{edge_key_graph}: OrigSomasA={somas_A}, OrigSomasB={somas_B}, AvgSomaIntensityForIface={avg_soma_intensity_for_this_interface:.2f}")

            if_metrics_graph = _calculate_interface_metrics( # Renamed
                mask_A_graph, mask_B_graph, original_cell_mask_local, intensity_local, 
                avg_soma_intensity_for_this_interface, # Pass the calculated average
                spacing_tuple, local_analysis_radius, min_local_intensity_difference,
                min_path_intensity_ratio_heuristic, # For comparison printout
                log_prefix=f"{log_prefix}     Edge{edge_key_graph}")
            edges[edge_key_graph] = if_metrics_graph
            
    print(f"{log_prefix} Graph built: {len(nodes)} nodes, {len(edges)} edges.")
    return nodes, edges


def separate_multi_soma_cells(
    segmentation_mask: np.ndarray, intensity_volume: np.ndarray, soma_mask: np.ndarray,
    spacing: Optional[Tuple[float, float, float]], min_size_threshold: int = 100,
    intensity_weight: float = 0.0, max_seed_centroid_dist: float = 40,
    min_path_intensity_ratio: float = 0.8, # If the metric for two tentative segments is >= this, they are considered "too bright" and merged
    min_local_intensity_difference: float = 0.05, local_analysis_radius: int = 10, #If the local intensity difference between two segments is less than this, they are merged
    # --- MODIFIED parameters for memory optimization ---
    memmap_dir: Optional[str] = "ramiseg_temp_memmap",
    memmap_voxel_threshold: int = 50_000_000 # ~50 million voxels
) -> np.ndarray:
    """
    Separates multi-soma cells using a hybrid watershed and graph-based merging approach.
    
    This version includes memory optimization for processing extremely large cells. If a
    cell's bounding box voxel count exceeds `memmap_voxel_threshold`, its large 
    intermediate arrays (like distance transforms, cost arrays, etc.) are created
    on disk as memory-mapped files in the `memmap_dir` to prevent RAM overload.

    Args:
        ... (existing args) ...
        memmap_dir (Optional[str]): Directory to store temporary memory-mapped files. 
            If None, this feature is disabled. The directory is created and cleaned up automatically.
        memmap_voxel_threshold (int): If the number of voxels in a cell's local bounding box
            exceeds this, memory-mapping is triggered to avoid RAM overload.
    """
    
    log_main_prefix = "[SepMultiSomaSerialV7_MemOpt_V2]"
    overall_start_time = time.time()
    print(f"{log_main_prefix} Starting Serialized Hybrid Separation (Voxel Threshold Trigger)")
    
    # --- Setup for Memory-Mapping ---
    use_memmap_feature = memmap_dir is not None
    if use_memmap_feature:
        if os.path.exists(memmap_dir):
            print(f"{log_main_prefix} INFO: Cleaning pre-existing memmap directory: {memmap_dir}")
            try:
                rmtree(memmap_dir)
            except OSError as e:
                print(f"{log_main_prefix} WARNING: Could not remove memmap directory: {e}. Files may be locked.")
        os.makedirs(memmap_dir, exist_ok=True)
        print(f"{log_main_prefix} INFO: Using temporary directory for large arrays: {memmap_dir}")

    try:
        # --- Initial parameter prints, spacing setup, type checks - same as before) ...
        if spacing is None: spacing_arr = np.array([1.0, 1.0, 1.0]); spacing_tuple = (1.0,1.0,1.0)
        else:
            try: spacing_tuple = tuple(float(s) for s in spacing); assert len(spacing_tuple) == 3; spacing_arr = np.array(spacing_tuple)
            except: spacing_tuple = (1.0,1.0,1.0); spacing_arr = np.array([1.0,1.0,1.0])
        print(f"{log_main_prefix} Using 3D spacing (z,y,x): {spacing_tuple}")
        if not np.issubdtype(intensity_volume.dtype, np.floating): intensity_volume = intensity_volume.astype(np.float32, copy=False)
        if not np.issubdtype(segmentation_mask.dtype, np.integer): segmentation_mask = segmentation_mask.astype(np.int32)
        if not np.issubdtype(soma_mask.dtype, np.integer): soma_mask = soma_mask.astype(np.int32)

        final_output_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
        unique_initial_labels = np.unique(segmentation_mask[segmentation_mask > 0])
        if unique_initial_labels.size == 0: print(f"{log_main_prefix} Seg mask empty."); return final_output_mask
        
        cell_to_somas_orig: Dict[int, Set[int]] = {lbl: set() for lbl in unique_initial_labels}
        present_soma_labels_orig = np.unique(soma_mask[soma_mask > 0])
        if present_soma_labels_orig.size == 0: 
            print(f"{log_main_prefix} Soma mask empty. Returning original seg (relabelled).")
            final_output_mask = segmentation_mask.copy()
            if np.any(final_output_mask): return relabel_sequential(final_output_mask.astype(np.int32), offset=1)[0]
            return final_output_mask

        for soma_lbl_init in present_soma_labels_orig:
            soma_loc_mask_init = (soma_mask == soma_lbl_init)
            cell_lbls_under_soma_init = np.unique(segmentation_mask[soma_loc_mask_init])
            for cell_lbl_init in cell_lbls_under_soma_init:
                if cell_lbl_init > 0 and cell_lbl_init in cell_to_somas_orig:
                    cell_to_somas_orig[cell_lbl_init].add(soma_lbl_init)

        multi_soma_cell_labels_list = [lbl for lbl, somas in cell_to_somas_orig.items() if len(somas) > 1]
        
        for lbl_init in unique_initial_labels:
            if lbl_init not in multi_soma_cell_labels_list:
                final_output_mask[segmentation_mask == lbl_init] = lbl_init
        
        if not multi_soma_cell_labels_list:
            print(f"{log_main_prefix} No multi-soma cells. Relabeling initial output. Time:{time.time()-overall_start_time:.2f}s")
            if np.any(final_output_mask): return relabel_sequential(final_output_mask.astype(np.int32), offset=1)[0]
            return final_output_mask
        
        print(f"{log_main_prefix} Found {len(multi_soma_cell_labels_list)} multi-soma cells to process.")
        current_max_overall_label_val = np.max(final_output_mask) if np.any(final_output_mask) else 0
        if present_soma_labels_orig.size > 0 : current_max_overall_label_val = max(current_max_overall_label_val, np.max(present_soma_labels_orig))
        next_global_label_offset_val = current_max_overall_label_val + 1
        print(f"{log_main_prefix} Initial next_global_label_offset: {next_global_label_offset_val}")

        phase1_combined_time = time.time()
        print(f"\n{log_main_prefix} Phase 1 (GWS + Graph-Merge per cell).")
        footprint_p1_dilation_val = footprint_rectangle((3,3,3)) 

        for cell_idx_p1, cell_label_p1 in enumerate(tqdm(multi_soma_cell_labels_list, desc=f"{log_main_prefix} P1:CellProc")):
            # --- More aggressive garbage collection ---
            gc.collect()

            print(f"\n{log_main_prefix}   P1 Processing cell L{cell_label_p1} ({cell_idx_p1+1}/{len(multi_soma_cell_labels_list)})")
            
            # --- GWS LOGIC FOR ONE CELL ---
            original_cell_mask_full_p1gws = (segmentation_mask == cell_label_p1)
            obj_slices_p1gws = find_objects(original_cell_mask_full_p1gws)
            if not obj_slices_p1gws or obj_slices_p1gws[0] is None: print(f"{log_main_prefix}     L{cell_label_p1} No object slices. Skip."); continue
            bbox_p1gws = obj_slices_p1gws[0]
            pad_p1gws = 3 
            local_bbox_slices_p1gws = tuple(slice(max(0, s.start - pad_p1gws), min(dim_size, s.stop + pad_p1gws))
                               for s, dim_size in zip(bbox_p1gws, segmentation_mask.shape))

            # --- Voxel-based Memory Activation ---
            local_shape = tuple(s.stop - s.start for s in local_bbox_slices_p1gws)
            num_voxels = np.prod(local_shape)
            use_memmap_for_this_cell = use_memmap_feature and (num_voxels > memmap_voxel_threshold)

            if use_memmap_for_this_cell:
                print(f"{log_main_prefix}     L{cell_label_p1} is large ({num_voxels:,} voxels > {memmap_voxel_threshold:,}). SWITCHING TO MEMORY-MAPPED ARRAYS.")
            else:
                print(f"{log_main_prefix}     L{cell_label_p1} processing in RAM ({num_voxels:,} voxels).")

            original_cell_mask_local_for_gws_p1gws = original_cell_mask_full_p1gws[local_bbox_slices_p1gws]
            if not np.any(original_cell_mask_local_for_gws_p1gws): print(f"{log_main_prefix}     L{cell_label_p1} Empty local cell mask. Skip."); continue
            soma_mask_local_orig_labels_gws_p1gws = soma_mask[local_bbox_slices_p1gws]
            intensity_local_gws_p1gws = intensity_volume[local_bbox_slices_p1gws]
            active_soma_labels_in_cell_gws_p1gws = sorted(list(cell_to_somas_orig[cell_label_p1]))
            
            soma_props_local_gws_p1gws: Dict[int, Dict[str, Any]] = {} 
            temp_soma_labeled_for_props_gws_p1gws = np.zeros_like(soma_mask_local_orig_labels_gws_p1gws, dtype=np.int32)
            for sl_orig_gws_p1gws in active_soma_labels_in_cell_gws_p1gws:
                temp_soma_labeled_for_props_gws_p1gws[(soma_mask_local_orig_labels_gws_p1gws == sl_orig_gws_p1gws) & original_cell_mask_local_for_gws_p1gws] = sl_orig_gws_p1gws
            if np.any(temp_soma_labeled_for_props_gws_p1gws):
                try:
                    props_gws_p1gws = regionprops(temp_soma_labeled_for_props_gws_p1gws, intensity_image=intensity_local_gws_p1gws)
                    for p_item_gws_p1gws in props_gws_p1gws:
                        if p_item_gws_p1gws.area > 0 : 
                             soma_props_local_gws_p1gws[p_item_gws_p1gws.label] = {'centroid': p_item_gws_p1gws.centroid, 'area': p_item_gws_p1gws.area, 'mean_intensity': p_item_gws_p1gws.mean_intensity if hasattr(p_item_gws_p1gws, 'mean_intensity') and p_item_gws_p1gws.mean_intensity is not None else 0.0}
                except Exception as e_rp_gws:
                    print(f"{log_main_prefix}     L{cell_label_p1} regionprops error: {e_rp_gws}. Fallback.")
                    for sl_orig_gws_p1gws in active_soma_labels_in_cell_gws_p1gws:
                        coords_gws_p1gws = np.argwhere((soma_mask_local_orig_labels_gws_p1gws == sl_orig_gws_p1gws) & original_cell_mask_local_for_gws_p1gws)
                        if coords_gws_p1gws.shape[0] > 0:
                            soma_mean_int = np.mean(intensity_local_gws_p1gws[coords_gws_p1gws[:,0], coords_gws_p1gws[:,1], coords_gws_p1gws[:,2]]) if coords_gws_p1gws.size > 0 else 0.0
                            soma_props_local_gws_p1gws[sl_orig_gws_p1gws] = {'centroid': np.mean(coords_gws_p1gws, axis=0), 'area': coords_gws_p1gws.shape[0], 'mean_intensity': soma_mean_int}
            
            valid_soma_labels_for_ws_gws_p1gws = [lbl for lbl in active_soma_labels_in_cell_gws_p1gws if lbl in soma_props_local_gws_p1gws and soma_props_local_gws_p1gws[lbl]['area'] > 0]
            if len(valid_soma_labels_for_ws_gws_p1gws) <= 1: print(f"{log_main_prefix}     L{cell_label_p1} <=1 valid GWS soma. Skip GWS."); continue
            
            num_seeds_gws_p1gws = len(valid_soma_labels_for_ws_gws_p1gws); soma_idx_map_gws_p1gws = {lbl: i for i, lbl in enumerate(valid_soma_labels_for_ws_gws_p1gws)}; adj_matrix_gws_p1gws = np.zeros((num_seeds_gws_p1gws, num_seeds_gws_p1gws), dtype=bool)
            max_intensity_val_local_gws_p1gws = np.max(intensity_local_gws_p1gws[original_cell_mask_local_for_gws_p1gws]) if np.any(original_cell_mask_local_for_gws_p1gws) else 1.0
            
            print(f"{log_main_prefix}     L{cell_label_p1} Creating cost array ({'memmap' if use_memmap_for_this_cell else 'RAM'})...")
            if use_memmap_for_this_cell:
                cost_array_path = os.path.join(memmap_dir, f"cell_{cell_label_p1}_cost.mmp")
                cost_array_local_gws_p1gws = np.memmap(cost_array_path, dtype=np.float32, mode='w+', shape=local_shape)
                cost_array_local_gws_p1gws[:] = np.inf
            else:
                cost_array_local_gws_p1gws = np.full(local_shape, np.inf, dtype=np.float32)

            if np.any(original_cell_mask_local_for_gws_p1gws): cost_array_local_gws_p1gws[original_cell_mask_local_for_gws_p1gws] = np.maximum(1e-6, max_intensity_val_local_gws_p1gws - intensity_local_gws_p1gws[original_cell_mask_local_for_gws_p1gws])
            
            for i_gws_p1gws in range(num_seeds_gws_p1gws):
                for j_gws_p1gws in range(i_gws_p1gws + 1, num_seeds_gws_p1gws):
                    lbl1_gws_p1gws, lbl2_gws_p1gws = valid_soma_labels_for_ws_gws_p1gws[i_gws_p1gws], valid_soma_labels_for_ws_gws_p1gws[j_gws_p1gws]; prop1_gws_p1gws, prop2_gws_p1gws = soma_props_local_gws_p1gws[lbl1_gws_p1gws], soma_props_local_gws_p1gws[lbl2_gws_p1gws]
                    c1_phys_gws_p1gws, c2_phys_gws_p1gws = np.array(prop1_gws_p1gws['centroid']) * spacing_arr, np.array(prop2_gws_p1gws['centroid']) * spacing_arr; dist_um_gws_p1gws = np.linalg.norm(c1_phys_gws_p1gws - c2_phys_gws_p1gws)
                    if dist_um_gws_p1gws > max_seed_centroid_dist: continue
                    c1_vox_gws_p1gws, c2_vox_gws_p1gws = tuple(np.round(prop1_gws_p1gws['centroid']).astype(int)), tuple(np.round(prop2_gws_p1gws['centroid']).astype(int))
                    c1_vox_c_gws_p1gws, c2_vox_c_gws_p1gws = tuple(np.clip(c1_vox_gws_p1gws[d_idx],0,s-1)for d_idx,s in enumerate(cost_array_local_gws_p1gws.shape)), tuple(np.clip(c2_vox_gws_p1gws[d_idx],0,s-1)for d_idx,s in enumerate(cost_array_local_gws_p1gws.shape))
                    if not original_cell_mask_local_for_gws_p1gws[c1_vox_c_gws_p1gws] or not original_cell_mask_local_for_gws_p1gws[c2_vox_c_gws_p1gws]: continue
                    path_median_intensity_gws_p1gws = 0.0
                    try:
                        path_indices_tup_gws_p1gws, _ = route_through_array(cost_array_local_gws_p1gws, c1_vox_c_gws_p1gws, c2_vox_c_gws_p1gws, fully_connected=True, geometric=False)
                        if isinstance(path_indices_tup_gws_p1gws, np.ndarray) and path_indices_tup_gws_p1gws.ndim==2 and path_indices_tup_gws_p1gws.shape[1]>0:
                            path_intensities_vals_gws_p1gws = intensity_local_gws_p1gws[path_indices_tup_gws_p1gws[0], path_indices_tup_gws_p1gws[1], path_indices_tup_gws_p1gws[2]]
                            if path_intensities_vals_gws_p1gws.size > 0: path_median_intensity_gws_p1gws = np.median(path_intensities_vals_gws_p1gws)
                    except: pass
                    ref_intensity_val_gws_p1gws = max(prop1_gws_p1gws.get('mean_intensity',1.0), prop2_gws_p1gws.get('mean_intensity',1.0), 1e-6)
                    ratio_val_gws_p1gws = path_median_intensity_gws_p1gws / ref_intensity_val_gws_p1gws if ref_intensity_val_gws_p1gws > 1e-6 else float('inf')
                    if ratio_val_gws_p1gws >= min_path_intensity_ratio:
                        adj_matrix_gws_p1gws[soma_idx_map_gws_p1gws[lbl1_gws_p1gws], soma_idx_map_gws_p1gws[lbl2_gws_p1gws]] = adj_matrix_gws_p1gws[soma_idx_map_gws_p1gws[lbl2_gws_p1gws], soma_idx_map_gws_p1gws[lbl1_gws_p1gws]] = True
            
            print(f"{log_main_prefix}     L{cell_label_p1} Creating watershed markers ({'memmap' if use_memmap_for_this_cell else 'RAM'})...")
            if use_memmap_for_this_cell:
                markers_path = os.path.join(memmap_dir, f"cell_{cell_label_p1}_markers.mmp")
                ws_markers_local_gws_p1gws = np.memmap(markers_path, dtype=np.int32, mode='w+', shape=local_shape)
                ws_markers_local_gws_p1gws[:] = 0
            else:
                ws_markers_local_gws_p1gws = np.zeros_like(original_cell_mask_local_for_gws_p1gws, dtype=np.int32)
            
            current_ws_marker_id_gws_p1gws = 1
            ws_marker_to_orig_somas_gws_p1gws: Dict[int, List[int]] = {}; orig_soma_to_ws_marker_gws_p1gws: Dict[int, int] = {}
            if np.any(adj_matrix_gws_p1gws):
                n_comps_adj_gws, comp_lbls_adj_gws = ndimage.label(adj_matrix_gws_p1gws)
                for k_comp_adj in range(1, n_comps_adj_gws + 1):
                    soma_indices_in_group_gws = np.where(comp_lbls_adj_gws == k_comp_adj)[0]
                    if not soma_indices_in_group_gws.size: continue
                    group_marker_id_gws = current_ws_marker_id_gws_p1gws
                    current_group_somas_gws = []
                    for seed_idx_gws in soma_indices_in_group_gws:
                        orig_s_lbl_gws = valid_soma_labels_for_ws_gws_p1gws[seed_idx_gws]
                        current_group_somas_gws.append(orig_s_lbl_gws)
                        ws_markers_local_gws_p1gws[(soma_mask_local_orig_labels_gws_p1gws == orig_s_lbl_gws) & original_cell_mask_local_for_gws_p1gws] = group_marker_id_gws
                        orig_soma_to_ws_marker_gws_p1gws[orig_s_lbl_gws] = group_marker_id_gws
                    if current_group_somas_gws: ws_marker_to_orig_somas_gws_p1gws[group_marker_id_gws] = sorted(current_group_somas_gws); current_ws_marker_id_gws_p1gws +=1
            for sl_orig_gws in valid_soma_labels_for_ws_gws_p1gws:
                if sl_orig_gws not in orig_soma_to_ws_marker_gws_p1gws:
                    group_marker_id_gws = current_ws_marker_id_gws_p1gws
                    ws_markers_local_gws_p1gws[(soma_mask_local_orig_labels_gws_p1gws == sl_orig_gws) & original_cell_mask_local_for_gws_p1gws] = group_marker_id_gws
                    orig_soma_to_ws_marker_gws_p1gws[sl_orig_gws] = group_marker_id_gws
                    ws_marker_to_orig_somas_gws_p1gws[group_marker_id_gws] = [sl_orig_gws]; current_ws_marker_id_gws_p1gws +=1
            num_final_ws_markers_gws_p1gws = current_ws_marker_id_gws_p1gws - 1
            if num_final_ws_markers_gws_p1gws <= 1: print(f"{log_main_prefix}     L{cell_label_p1} <=1 final WS marker. Skip WS."); continue
            
            print(f"{log_main_prefix}     L{cell_label_p1} Calculating distance transform ({'memmap' if use_memmap_for_this_cell else 'RAM'})...")
            if use_memmap_for_this_cell:
                dt_path = os.path.join(memmap_dir, f"cell_{cell_label_p1}_dt.mmp")
                dt_local_gws = np.memmap(dt_path, dtype=np.float32, mode='w+', shape=local_shape)
                distance_transform_edt(original_cell_mask_local_for_gws_p1gws, sampling=spacing_tuple, output=dt_local_gws)
            else:
                dt_local_gws = distance_transform_edt(original_cell_mask_local_for_gws_p1gws, sampling=spacing_tuple)
            
            print(f"{log_main_prefix}     L{cell_label_p1} Creating watershed landscape ({'memmap' if use_memmap_for_this_cell else 'RAM'})...")
            if use_memmap_for_this_cell:
                ws_landscape_path = os.path.join(memmap_dir, f"cell_{cell_label_p1}_landscape.mmp")
                ws_landscape_gws = np.memmap(ws_landscape_path, dtype=np.float32, mode='w+', shape=local_shape)
                np.negative(dt_local_gws, out=ws_landscape_gws)
            else:
                ws_landscape_gws = -dt_local_gws.astype(np.float32)

            if intensity_weight > 1e-6:
                print(f"{log_main_prefix}     L{cell_label_p1} Adding intensity term to landscape...")
                icell_gws = intensity_local_gws_p1gws[original_cell_mask_local_for_gws_p1gws]
                if icell_gws.size > 0:
                    min_ic_gws, max_ic_gws = np.min(icell_gws), np.max(icell_gws)
                    if (max_ic_gws - min_ic_gws) > 1e-6:
                        norm_int_term_gws = np.zeros_like(ws_landscape_gws); norm_int_term_gws[original_cell_mask_local_for_gws_p1gws] = (max_ic_gws - icell_gws) / (max_ic_gws - min_ic_gws)
                        max_dt_gws = np.max(dt_local_gws); ws_landscape_gws += intensity_weight * norm_int_term_gws * (max_dt_gws if max_dt_gws > 1e-6 else 1.0)
            
            ws_markers_local_gws_p1gws[~original_cell_mask_local_for_gws_p1gws] = 0
            
            # --- Enhanced Pre-Watershed Diagnostics ---
            print(f"{log_main_prefix}     L{cell_label_p1} PRE-WATERSHED DIAGNOSTICS:")
            print(f"{log_main_prefix}       - Landscape shape: {ws_landscape_gws.shape}, dtype: {ws_landscape_gws.dtype}, is_memmap: {isinstance(ws_landscape_gws, np.memmap)}")
            print(f"{log_main_prefix}       - Markers shape:   {ws_markers_local_gws_p1gws.shape}, dtype: {ws_markers_local_gws_p1gws.dtype}, is_memmap: {isinstance(ws_markers_local_gws_p1gws, np.memmap)}")
            print(f"{log_main_prefix}       - Mask shape:      {original_cell_mask_local_for_gws_p1gws.shape}, dtype: {original_cell_mask_local_for_gws_p1gws.dtype}")
            unique_markers, counts = np.unique(ws_markers_local_gws_p1gws, return_counts=True)
            print(f"{log_main_prefix}       - Unique markers ({len(unique_markers)}): {unique_markers}")
            if np.any(np.isinf(ws_landscape_gws)): print(f"{log_main_prefix}       - WARNING: Landscape contains Inf values!")
            if np.any(np.isnan(ws_landscape_gws)): print(f"{log_main_prefix}       - WARNING: Landscape contains NaN values!")
            
            print(f"{log_main_prefix}     L{cell_label_p1} EXECUTING WATERSHED (this step will use RAM for its result)...")
            ws_start_time = time.time()
            global_ws_result_local_p1gws = watershed(ws_landscape_gws, ws_markers_local_gws_p1gws, mask=original_cell_mask_local_for_gws_p1gws, watershed_line=True)
            print(f"{log_main_prefix}     L{cell_label_p1} Watershed finished. Time: {time.time() - ws_start_time:.2f}s. Result is in RAM.")

            print(f"{log_main_prefix}     L{cell_label_p1} Creating temp mask for post-processing ({'memmap' if use_memmap_for_this_cell else 'RAM'})...")
            if use_memmap_for_this_cell:
                temp_gws_path = os.path.join(memmap_dir, f"cell_{cell_label_p1}_temp_gws.mmp")
                temp_gws_mask_local_p1gws = np.memmap(temp_gws_path, dtype=np.int32, mode='w+', shape=local_shape)
                temp_gws_mask_local_p1gws[:] = 0
            else:
                temp_gws_mask_local_p1gws = np.zeros_like(global_ws_result_local_p1gws, dtype=np.int32)
            
            largest_soma_lbl_gws, max_area_s_gws = -1, -1
            for sl_prop_gws in valid_soma_labels_for_ws_gws_p1gws:
                if soma_props_local_gws_p1gws[sl_prop_gws]['area'] > max_area_s_gws: max_area_s_gws = soma_props_local_gws_p1gws[sl_prop_gws]['area']; largest_soma_lbl_gws = sl_prop_gws
            main_marker_id_gws = orig_soma_to_ws_marker_gws_p1gws.get(largest_soma_lbl_gws, -1)
            unique_ws_res_lbls_gws = np.unique(global_ws_result_local_p1gws[global_ws_result_local_p1gws > 0])
            current_next_lbl_cell_gws = next_global_label_offset_val
            for res_lbl_gws in unique_ws_res_lbls_gws:
                final_lbl_gws = cell_label_p1 if res_lbl_gws == main_marker_id_gws else current_next_lbl_cell_gws
                if res_lbl_gws != main_marker_id_gws: current_next_lbl_cell_gws +=1
                temp_gws_mask_local_p1gws[global_ws_result_local_p1gws == res_lbl_gws] = final_lbl_gws
            next_global_label_offset_val = max(next_global_label_offset_val, current_next_lbl_cell_gws)

            print(f"{log_main_prefix}     L{cell_label_p1} Freeing in-memory watershed result and intermediate arrays...")
            del global_ws_result_local_p1gws, cost_array_local_gws_p1gws, dt_local_gws, ws_landscape_gws, ws_markers_local_gws_p1gws
            gc.collect()

            ws_lines_gws = (temp_gws_mask_local_p1gws == 0) & original_cell_mask_local_for_gws_p1gws
            if np.any(ws_lines_gws):
                line_coords_gws = np.argwhere(ws_lines_gws)
                for zlg,ylg,xlg in line_coords_gws:
                    nh_slice_gws = tuple(slice(max(0,c-1),min(s,c+2))for c,s in zip((zlg,ylg,xlg),temp_gws_mask_local_p1gws.shape))
                    nh_vals_gws = temp_gws_mask_local_p1gws[nh_slice_gws]; un_nh_gws,cts_nh_gws=np.unique(nh_vals_gws[nh_vals_gws>0],return_counts=True)
                    if un_nh_gws.size > 0: temp_gws_mask_local_p1gws[zlg,ylg,xlg] = un_nh_gws[np.argmax(cts_nh_gws)]
            
            print(f"{log_main_prefix}     L{cell_label_p1} Merging small fragments...")
            frag_merged_gws,sfm_iters_gws=True,0
            while frag_merged_gws and sfm_iters_gws < 10:
                sfm_iters_gws+=1; frag_merged_gws=False
                curr_lbls_tgws = np.unique(temp_gws_mask_local_p1gws[temp_gws_mask_local_p1gws>0])
                if len(curr_lbls_tgws)<=1:break
                for lbl_sfm_gws in curr_lbls_tgws:
                    curr_frag_gws=(temp_gws_mask_local_p1gws==lbl_sfm_gws);size_sfm_gws=np.sum(curr_frag_gws)
                    if size_sfm_gws > 0 and size_sfm_gws < min_size_threshold:
                        dil_frag_gws=binary_dilation(curr_frag_gws,footprint=footprint_p1_dilation_val)
                        neigh_reg_gws = dil_frag_gws & (~curr_frag_gws) & original_cell_mask_local_for_gws_p1gws & (temp_gws_mask_local_p1gws!=0) & (temp_gws_mask_local_p1gws!=lbl_sfm_gws)
                        if not np.any(neigh_reg_gws): continue
                        neigh_lbls_gws,neigh_cts_gws=np.unique(temp_gws_mask_local_p1gws[neigh_reg_gws],return_counts=True)
                        if neigh_lbls_gws.size==0:continue
                        largest_neigh_lbl_gws=neigh_lbls_gws[np.argmax(neigh_cts_gws)];temp_gws_mask_local_p1gws[curr_frag_gws]=largest_neigh_lbl_gws
                        frag_merged_gws=True;break

            # --- Phase 1.5: Build Local Graph & Merge Weak Interfaces for THIS CELL ---
            print(f"{log_main_prefix}     P1 L{cell_label_p1}: Building local graph for P1.5 merges on {np.sum(np.unique(temp_gws_mask_local_p1gws)>0)} GWS segments.")
            local_nodes_p15, local_edges_p15 = _build_adjacency_graph_for_cell(
                temp_gws_mask_local_p1gws,
                original_cell_mask_local_for_gws_p1gws,
                soma_mask_local_orig_labels_gws_p1gws,
                soma_props_local_gws_p1gws,
                intensity_local_gws_p1gws,
                spacing_tuple,
                local_analysis_radius, min_local_intensity_difference,
                min_path_intensity_ratio,
                log_prefix=f"{log_main_prefix}       GraphBuild_P1.5 L{cell_label_p1}"
            )

            if local_edges_p15: 
                print(f"{log_main_prefix}     P1 L{cell_label_p1}: P1.5 - Checking {len(local_edges_p15)} interfaces for merging.")
                p1_5_local_merge_passes = 0; MAX_P1_5_LOCAL_PASSES = 5
                p1_5_merged_in_pass_loc = True
                while p1_5_merged_in_pass_loc and p1_5_local_merge_passes < MAX_P1_5_LOCAL_PASSES:
                    p1_5_local_merge_passes += 1; p1_5_merged_in_pass_loc = False
                    print(f"{log_main_prefix}       P1.5 L{cell_label_p1} Merge Pass {p1_5_local_merge_passes}")
                    
                    current_local_labels_in_temp_gws = np.unique(temp_gws_mask_local_p1gws[temp_gws_mask_local_p1gws > 0])
                    if len(current_local_labels_in_temp_gws) <=1: print(f"{log_main_prefix}         P1.5 L{cell_label_p1}: Only {len(current_local_labels_in_temp_gws)} segment(s) left. No more P1.5 merges."); break

                    sorted_local_edges_to_check_p15 = sorted(list(local_edges_p15.keys()))

                    for edge_key_local_p1_5_val in sorted_local_edges_to_check_p15:
                        lbl_A_loc_p15, lbl_B_loc_p15 = edge_key_local_p1_5_val
                        edge_metrics_local_p1_5_val = local_edges_p15.get(edge_key_local_p1_5_val)

                        if not edge_metrics_local_p1_5_val: continue 

                        mask_A_exists_loc_p15 = np.any(temp_gws_mask_local_p1gws == lbl_A_loc_p15)
                        mask_B_exists_loc_p15 = np.any(temp_gws_mask_local_p1gws == lbl_B_loc_p15)
                        if not mask_A_exists_loc_p15 or not mask_B_exists_loc_p15 or lbl_A_loc_p15 == lbl_B_loc_p15: 
                            if edge_key_local_p1_5_val in local_edges_p15: del local_edges_p15[edge_key_local_p1_5_val] 
                            continue
                        
                        should_merge_local_p1_5_val = edge_metrics_local_p1_5_val.get('should_merge_decision', False)
                        
                        if should_merge_local_p1_5_val:
                            print(f"{log_main_prefix}       P1.5 L{cell_label_p1} Edge ({lbl_A_loc_p15},{lbl_B_loc_p15}): Metrics indicate MERGE.")
                            label_to_keep_loc_p15 = lbl_A_loc_p15; label_to_remove_loc_p15 = lbl_B_loc_p15
                            vol_A_loc_p15 = local_nodes_p15.get(lbl_A_loc_p15,{}).get('volume',0)
                            vol_B_loc_p15 = local_nodes_p15.get(lbl_B_loc_p15,{}).get('volume',0)

                            if lbl_A_loc_p15 != cell_label_p1 and lbl_B_loc_p15 == cell_label_p1:
                                label_to_keep_loc_p15, label_to_remove_loc_p15 = lbl_B_loc_p15, lbl_A_loc_p15
                            elif lbl_A_loc_p15 == cell_label_p1 and lbl_B_loc_p15 != cell_label_p1:
                                pass 
                            elif vol_A_loc_p15 < vol_B_loc_p15:
                                label_to_keep_loc_p15, label_to_remove_loc_p15 = lbl_B_loc_p15, lbl_A_loc_p15
                            
                            print(f"{log_main_prefix}         Merging L{label_to_remove_loc_p15} into L{label_to_keep_loc_p15} in temp_gws_mask_local_p1gws.")
                            temp_gws_mask_local_p1gws[temp_gws_mask_local_p1gws == label_to_remove_loc_p15] = label_to_keep_loc_p15
                            p1_5_merged_in_pass_loc = True
                            
                            if label_to_remove_loc_p15 in local_nodes_p15: 
                                local_nodes_p15[label_to_keep_loc_p15]['volume'] += local_nodes_p15[label_to_remove_loc_p15]['volume']
                                local_nodes_p15[label_to_keep_loc_p15]['orig_somas'] = sorted(list(set(local_nodes_p15[label_to_keep_loc_p15]['orig_somas'] + local_nodes_p15[label_to_remove_loc_p15]['orig_somas'])))
                                del local_nodes_p15[label_to_remove_loc_p15]
                            
                            stale_edges_p15 = [ek for ek in local_edges_p15 if label_to_remove_loc_p15 in ek]
                            for sek_p15 in stale_edges_p15:
                                if sek_p15 in local_edges_p15: del local_edges_p15[sek_p15]
                            break 
                    if p1_5_merged_in_pass_loc: continue 
                    if not p1_5_merged_in_pass_loc: break 
            else:
                print(f"{log_main_prefix}     P1 L{cell_label_p1}: P1.5 - No interfaces or no merges needed based on graph.")
            
            print(f"{log_main_prefix}     P1 L{cell_label_p1} Writing final segments to global mask...")
            output_mask_local_view_final_p1_val = final_output_mask[local_bbox_slices_p1gws]
            output_mask_local_view_final_p1_val[original_cell_mask_local_for_gws_p1gws] = \
                temp_gws_mask_local_p1gws[original_cell_mask_local_for_gws_p1gws]
            final_output_mask[local_bbox_slices_p1gws] = output_mask_local_view_final_p1_val
            
            max_lbl_in_cell_after_p15_val = np.max(temp_gws_mask_local_p1gws) if np.any(temp_gws_mask_local_p1gws) else 0
            if max_lbl_in_cell_after_p15_val >= cell_label_p1 : 
                 next_global_label_offset_val = max(next_global_label_offset_val, max_lbl_in_cell_after_p15_val + 1)
            
            print(f"{log_main_prefix}     P1 L{cell_label_p1}: Finished P1 & P1.5. Max label in cell: {max_lbl_in_cell_after_p15_val}. Next global offset: {next_global_label_offset_val}.")
            del temp_gws_mask_local_p1gws, local_nodes_p15, local_edges_p15
            
        print(f"{log_main_prefix} Phase 1 & 1.5 (GWS + Per-Cell Graph Merging) completed. Time: {time.time()-phase1_combined_time:.2f}s. Final next_global_label_offset after P1.5: {next_global_label_offset_val}")

        print(f"\n{log_main_prefix} Phase 2 (Local Splitting) has been REMOVED as per request.")

        phase3_start_time = time.time()
        print(f"\n{log_main_prefix} Phase 3: Finalizing mask.")
        try:
            print(f"{log_main_prefix} Filling internal voids...")
            final_mask_filled = fill_internal_voids(final_output_mask) 
        except NameError:
            print(f"{log_main_prefix} Warning: fill_internal_voids not defined. Skipping this step.")
            final_mask_filled = final_output_mask.copy()

        if np.any(final_mask_filled):
            print(f"{log_main_prefix} Relabeling final mask sequentially...")
            relabeled_array, forward_map, inverse_map = relabel_sequential(
                final_mask_filled.astype(np.int32), offset=1 
            )
            unique_labels_in_relabeled = np.unique(relabeled_array)
            num_objects_final = len(unique_labels_in_relabeled[unique_labels_in_relabeled > 0])
            max_label_val = num_objects_final if num_objects_final > 0 else 0
            
            print(f"{log_main_prefix} Relabeled final mask contains {num_objects_final} objects (max label: {max_label_val}).")
            output_to_return = relabeled_array.astype(np.int32)
        else:
            print(f"{log_main_prefix} Final mask is empty.")
            output_to_return = final_mask_filled.astype(np.int32)

        print(f"{log_main_prefix} Phase 3 (Finalization) completed. Time: {time.time()-phase3_start_time:.2f}s")
        print(f"{log_main_prefix} Total processing time: {time.time()-overall_start_time:.2f}s")
        
        return output_to_return

    finally:
        # --- Final Cleanup of Memory-Mapped Files ---
        if use_memmap_feature and memmap_dir and os.path.exists(memmap_dir):
            print(f"{log_main_prefix} Final cleanup of memmap directory: {memmap_dir}")
            # Force garbage collection to release file handles before attempting to delete
            gc.collect()
            rmtree(memmap_dir, ignore_errors=True)

# --- fill_internal_voids (ensure it's defined as before) ---
def fill_internal_voids(segmentation_mask_input: np.ndarray) -> np.ndarray:
    log_fill_prefix = "[FillVoids]"
    print(f"{log_fill_prefix} Starting Internal Void Filling...")
    t_start_fill = time.time()
    filled_mask_output = segmentation_mask_input.copy()
    
    # find_objects requires positive integer labels. Ensure input is appropriate.
    # If mask contains 0s, find_objects will skip them.
    # It returns a list where index i corresponds to label i+1.
    input_for_find_objects = segmentation_mask_input.astype(np.int32)
    if np.max(input_for_find_objects) == 0 : # All zeros or empty
        print(f"{log_fill_prefix} Input mask is empty or all zeros. No voids to fill.")
        print(f"{log_fill_prefix} Void Filling Complete. Time: {time.time()-t_start_fill:.2f}s")
        return filled_mask_output

    bboxes = find_objects(input_for_find_objects)
    
    voids_filled_count = 0
    voxels_added_total = 0

    if bboxes is None: # Should not happen if max > 0, but defensive.
        print(f"{log_fill_prefix} No objects found by find_objects. No voids to fill.")
        print(f"{log_fill_prefix} Void Filling Complete. Time: {time.time()-t_start_fill:.2f}s")
        return filled_mask_output

    for i, bbox_slices in enumerate(bboxes):
        label_val_fill = i + 1 # find_objects list is indexed by label-1
        
        if bbox_slices is None: continue # This label doesn't exist (e.g., if labels are not sequential from 1)

        if not (isinstance(bbox_slices, tuple) and len(bbox_slices) == segmentation_mask_input.ndim and \
                all(isinstance(s, slice) for s in bbox_slices)):
            print(f"{log_fill_prefix}   Skipping label {label_val_fill} due to invalid bbox: {bbox_slices}")
            continue

        original_roi_fill = segmentation_mask_input[bbox_slices] # Use original mask for ROI
        cell_mask_roi_fill = (original_roi_fill == label_val_fill) # Create binary mask for current label in ROI
        
        if not np.any(cell_mask_roi_fill):
            print(f"{log_fill_prefix}   ROI for label {label_val_fill} is empty for this label. Skipping fill.")
            continue

        original_voxels_fill = np.sum(cell_mask_roi_fill)
        try:
            # binary_fill_holes expects a 2D or 3D binary image.
            filled_cell_roi_fill = binary_fill_holes(cell_mask_roi_fill)
        except Exception as e_fill:
            print(f"{log_fill_prefix}   Error filling holes for label {label_val_fill} in ROI: {e_fill}. Skipping.")
            continue

        filled_voxels_fill = np.sum(filled_cell_roi_fill)
        if filled_voxels_fill > original_voxels_fill:
            voxels_added_fill = filled_voxels_fill - original_voxels_fill
            voxels_added_total += voxels_added_fill
            voids_filled_count += 1
            print(f"{log_fill_prefix}   Filled void in label {label_val_fill}: added {voxels_added_fill} voxels.")
            
            # Update the corresponding ROI in our result mask (filled_mask_output)
            # We use the filled_cell_roi_fill (a boolean mask) to specify which
            # voxels *within the bounding box* should be set to the current label.
            result_roi_view_fill = filled_mask_output[bbox_slices] # Get a view of the output mask's ROI
            result_roi_view_fill[filled_cell_roi_fill] = label_val_fill # Modify the view (updates filled_mask_output)
    
    print(f"{log_fill_prefix} --- Void Filling Summary ---")
    if voids_filled_count > 0:
        print(f"{log_fill_prefix}   Filled voids in {voids_filled_count} objects.")
        print(f"{log_fill_prefix}   Total voxels added: {voxels_added_total}")
    else:
        print(f"{log_fill_prefix}   No internal voids were found to fill.")
    print(f"{log_fill_prefix} Void Filling Complete. Time: {time.time()-t_start_fill:.2f}s")
    return filled_mask_output