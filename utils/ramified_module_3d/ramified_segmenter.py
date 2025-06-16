import numpy as np
from scipy import ndimage, sparse
from tqdm import tqdm
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
from scipy.sparse.csgraph import connected_components, dijkstra
from skimage.measure import label, regionprops # type: ignore
from collections import deque

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
                    eroded_core = ndimage.binary_erosion(initial_core, structure=struct_el_erosion, iterations=erosion_iterations) if erosion_iterations > 0 else initial_core
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
                    eroded_core = ndimage.binary_erosion(initial_core, structure=struct_el_erosion, iterations=erosion_iterations) if erosion_iterations > 0 else initial_core
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

def separate_multi_soma_cells(
    segmentation_mask: np.ndarray,
    intensity_volume: np.ndarray,
    soma_mask: np.ndarray,
    spacing: Tuple[float, float, float],
    min_size_threshold: int = 100,
    max_seed_centroid_dist: float = 40.0,
    min_path_intensity_ratio: float = 0.8,
    min_local_intensity_difference: float = 0.05,
    local_analysis_radius: int = 10
) -> np.ndarray:
    """
    Memory-efficient recursive separation of cells with multiple somas.
    Continues separating until all somas are properly isolated.
    
    Parameters:
    -----------
    segmentation_mask : np.ndarray
        3D array with integer labels for cell regions
    intensity_volume : np.ndarray
        3D fluorescence intensity volume
    soma_mask : np.ndarray
        3D binary or labeled mask indicating soma locations
    spacing : Tuple[float, float, float]
        Voxel spacing in (z, y, x) dimensions
    min_size_threshold : int
        Minimum size for a valid cell region
    max_seed_centroid_dist : float
        Maximum distance between soma centroids to consider separation
    min_path_intensity_ratio : float
        Minimum ratio of path intensity to mean cell intensity
    min_local_intensity_difference : float
        Minimum relative intensity difference between regions adjacent to cut site
    local_analysis_radius : int
        Radius (in voxels) for local intensity analysis around cut site
        
    Returns:
    --------
    np.ndarray
        Refined segmentation mask with separated cells
    """
    
    def calculate_physical_distance(coord1, coord2, spacing):
        """Calculate physical distance between two coordinates"""
        diff = np.array(coord1) - np.array(coord2)
        physical_diff = diff * np.array(spacing)
        return np.linalg.norm(physical_diff)
    
    def get_cell_bounding_box(cell_mask, padding=5):
        """Get tight bounding box around cell with padding"""
        coords = np.where(cell_mask)
        if len(coords[0]) == 0:
            return None
        
        min_coords = [max(0, np.min(coords[i]) - padding) for i in range(3)]
        max_coords = [min(cell_mask.shape[i], np.max(coords[i]) + padding + 1) for i in range(3)]
        
        return tuple(slice(min_coords[i], max_coords[i]) for i in range(3))
    
    def get_local_regions_around_interface(region1, region2, cell_region, radius=3):
        """Get voxels that are within 'radius' distance from the interface between two regions."""
        from scipy.ndimage import binary_dilation
        from skimage.morphology import ball # type: ignore
        
        if radius <= 1:
            struct_elem = np.ones((3, 3, 3), dtype=bool)
        else:
            struct_elem = ball(radius)
        
        dilated_region1 = binary_dilation(region1, struct_elem)
        dilated_region2 = binary_dilation(region2, struct_elem)
        
        interface_zone = dilated_region1 & dilated_region2 & cell_region
        local_region1 = interface_zone & dilated_region1 & ~region1
        local_region2 = interface_zone & dilated_region2 & ~region2
        
        return local_region1, local_region2, interface_zone
    
    def analyze_local_intensity_difference(region1, region2, cell_region, intensity_vol, radius=3):
        """Analyze intensity differences in regions immediately adjacent to the separation interface."""
        print(f"    Analyzing local intensity differences around interface (radius: {radius})")
        
        local_region1, local_region2, interface_zone = get_local_regions_around_interface(
            region1, region2, cell_region, radius
        )
        
        print(f"      Interface zone size: {np.sum(interface_zone)} voxels")
        print(f"      Local region 1 size: {np.sum(local_region1)} voxels")
        print(f"      Local region 2 size: {np.sum(local_region2)} voxels")
        
        if np.sum(local_region1) == 0 or np.sum(local_region2) == 0:
            print(f"      WARNING: One or both local regions are empty - cannot analyze")
            return None, False
        
        local1_intensities = intensity_vol[local_region1]
        local2_intensities = intensity_vol[local_region2]
        
        local1_mean = np.mean(local1_intensities)
        local1_std = np.std(local1_intensities)
        local2_mean = np.mean(local2_intensities)
        local2_std = np.std(local2_intensities)
        
        print(f"      Local region 1: mean={local1_mean:.2f}, std={local1_std:.2f}")
        print(f"      Local region 2: mean={local2_mean:.2f}, std={local2_std:.2f}")
        
        reference_intensity = max(local1_mean, local2_mean)
        if reference_intensity == 0:
            print(f"      ERROR: Reference intensity is zero")
            return None, False
        
        intensity_difference = abs(local1_mean - local2_mean)
        relative_difference = intensity_difference / reference_intensity
        
        print(f"      Absolute intensity difference: {intensity_difference:.2f}")
        print(f"      Relative intensity difference: {relative_difference:.3f}")
        print(f"      Threshold: {min_local_intensity_difference:.3f}")
        
        is_valid = relative_difference >= min_local_intensity_difference
        
        if is_valid:
            print(f"      PASSED: Local intensity difference is sufficient ({relative_difference:.3f} >= {min_local_intensity_difference:.3f})")
        else:
            print(f"      FAILED: Local intensity difference too small ({relative_difference:.3f} < {min_local_intensity_difference:.3f})")
        
        return relative_difference, is_valid
    
    def bresenham_line_3d(start, end):
        """Generate 3D line coordinates between two points"""
        start = np.array(start, dtype=float)
        end = np.array(end, dtype=float)
        
        diff = end - start
        steps = int(np.max(np.abs(diff))) + 1
        
        if steps <= 1:
            return [start]
        
        coords = []
        for i in range(steps):
            t = i / (steps - 1)
            coord = start + t * diff
            coords.append(coord)
        
        return coords
    
    def find_connecting_path_and_cut_plane(cell_region, soma1_region, soma2_region, intensity_vol):
        """Find the connecting path between somas using watershed separation."""
        print(f"    Analyzing connection between somas (sizes: {np.sum(soma1_region)}, {np.sum(soma2_region)})")
        
        soma1_coords = np.where(soma1_region)
        soma1_centroid = [np.mean(soma1_coords[i]) for i in range(3)]
        
        soma2_coords = np.where(soma2_region)
        soma2_centroid = [np.mean(soma2_coords[i]) for i in range(3)]
        
        print(f"    Soma centroids: {[f'{c:.1f}' for c in soma1_centroid]} -> {[f'{c:.1f}' for c in soma2_centroid]}")
        
        soma1_intensity = np.mean(intensity_vol[soma1_region])
        soma2_intensity = np.mean(intensity_vol[soma2_region])
        mean_soma_intensity = (soma1_intensity + soma2_intensity) / 2
        
        print(f"    Soma intensities: {soma1_intensity:.2f}, {soma2_intensity:.2f} (mean: {mean_soma_intensity:.2f})")
        print(f"    Using watershed-based separation following intensity valleys")
        
        from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation
        from skimage.segmentation import watershed # type: ignore
        from skimage.morphology import cube # type: ignore
        
        cell_intensities = intensity_vol * cell_region.astype(float)
        cell_intensities[~cell_region] = 0
        
        cell_intensity_values = intensity_vol[cell_region]
        min_intensity = np.min(cell_intensity_values)
        max_intensity = np.max(cell_intensity_values)
        intensity_range = max_intensity - min_intensity
        
        print(f"    Cell intensity range: {min_intensity:.1f} to {max_intensity:.1f}")
        
        if intensity_range < mean_soma_intensity * 0.1:
            print(f"    Insufficient intensity variation for watershed separation ({intensity_range:.1f} < {mean_soma_intensity * 0.1:.1f})")
            return None
        
        normalized_intensities = (cell_intensities - min_intensity) / intensity_range
        inverted_intensities = 1.0 - normalized_intensities
        inverted_intensities[~cell_region] = 0
        
        smoothed_inverted = gaussian_filter(inverted_intensities, sigma=0.8)
        
        markers = np.zeros_like(cell_region, dtype=int)
        markers[soma1_region] = 1
        markers[soma2_region] = 2
        
        watershed_labels = watershed(smoothed_inverted, markers, mask=cell_region)
        
        print(f"    Watershed segmentation complete")
        
        region1 = watershed_labels == 1
        region2 = watershed_labels == 2
        
        if np.sum(region1) == 0 or np.sum(region2) == 0:
            print(f"    Watershed failed - empty regions (sizes: {np.sum(region1)}, {np.sum(region2)})")
            return None
        
        print(f"    Watershed regions: {np.sum(region1)} and {np.sum(region2)} voxels")
        
        boundary_mask = cell_region & (watershed_labels == 0)
        
        if np.sum(boundary_mask) > 0:
            boundary_intensities = intensity_vol[boundary_mask]
            separation_intensity = np.mean(boundary_intensities)
            
            print(f"    Watershed boundary analysis:")
            print(f"      Boundary voxels: {np.sum(boundary_mask)}")
            print(f"      Boundary mean intensity: {separation_intensity:.2f}")
            print(f"      Boundary/Soma intensity ratio: {separation_intensity/mean_soma_intensity:.3f}")
            
            boundary_coords = np.where(boundary_mask)
            region1_mean_intensity = np.mean(intensity_vol[region1])
            region2_mean_intensity = np.mean(intensity_vol[region2])
            
            for i in range(len(boundary_coords[0])):
                coord = (boundary_coords[0][i], boundary_coords[1][i], boundary_coords[2][i])
                voxel_intensity = intensity_vol[coord]
                
                dist_to_region1 = abs(voxel_intensity - region1_mean_intensity)
                dist_to_region2 = abs(voxel_intensity - region2_mean_intensity)
                
                if dist_to_region1 < dist_to_region2:
                    region1[coord] = True
                else:
                    region2[coord] = True
        else:
            print(f"    No explicit watershed boundary found - analyzing region interface")
            
            struct_elem = cube(3)
            dilated_region1 = binary_dilation(region1, struct_elem)
            dilated_region2 = binary_dilation(region2, struct_elem)
            
            interface_mask = (dilated_region1 & dilated_region2 & cell_region & 
                            ~region1 & ~region2)
            
            if np.sum(interface_mask) > 0:
                interface_intensities = intensity_vol[interface_mask]
                separation_intensity = np.mean(interface_intensities)
                
                print(f"      Found interface region: {np.sum(interface_mask)} voxels")
                print(f"      Interface mean intensity: {separation_intensity:.2f}")
                print(f"      Interface/Soma intensity ratio: {separation_intensity/mean_soma_intensity:.3f}")
                
                interface_coords = np.where(interface_mask)
                region1_mean_intensity = np.mean(intensity_vol[region1])
                region2_mean_intensity = np.mean(intensity_vol[region2])
                
                for i in range(len(interface_coords[0])):
                    coord = (interface_coords[0][i], interface_coords[1][i], interface_coords[2][i])
                    voxel_intensity = intensity_vol[coord]
                    
                    dist_to_region1 = abs(voxel_intensity - region1_mean_intensity)
                    dist_to_region2 = abs(voxel_intensity - region2_mean_intensity)
                    
                    if dist_to_region1 < dist_to_region2:
                        region1[coord] = True
                    else:
                        region2[coord] = True
            else:
                print(f"      No clear interface found - sampling direct path between somas")
                
                path_coords = bresenham_line_3d(soma1_centroid, soma2_centroid)
                
                path_intensities = []
                for coord in path_coords:
                    int_coord = tuple(int(round(c)) for c in coord)
                    
                    if all(0 <= int_coord[i] < cell_region.shape[i] for i in range(3)):
                        if (cell_region[int_coord] and 
                            not soma1_region[int_coord] and 
                            not soma2_region[int_coord]):
                            path_intensities.append(intensity_vol[int_coord])
                
                if len(path_intensities) > 0:
                    separation_intensity = np.mean(path_intensities)
                    print(f"      Path sampling: {len(path_intensities)} voxels, mean intensity: {separation_intensity:.2f}")
                else:
                    non_soma_mask = cell_region & ~soma1_region & ~soma2_region
                    if np.sum(non_soma_mask) > 0:
                        separation_intensity = np.min(intensity_vol[non_soma_mask])
                        print(f"      Using minimum non-soma intensity: {separation_intensity:.2f}")
                    else:
                        print(f"      ERROR: Cannot determine separation intensity")
                        return None
        
        intensity_ratio = separation_intensity / mean_soma_intensity
        print(f"    Final separation analysis:")
        print(f"      Separation intensity: {separation_intensity:.2f}")
        print(f"      Mean soma intensity: {mean_soma_intensity:.2f}")
        print(f"      Intensity ratio: {intensity_ratio:.3f} (threshold: {min_path_intensity_ratio})")
        
        if intensity_ratio >= min_path_intensity_ratio:
            print(f"    Separation region too bright for valid separation ({intensity_ratio:.3f} >= {min_path_intensity_ratio})")
            return None
        
        print(f"    \n    === LOCAL INTENSITY ANALYSIS ===")
        relative_diff, local_validation_passed = analyze_local_intensity_difference(
            region1, region2, cell_region, intensity_vol, local_analysis_radius
        )
        
        if not local_validation_passed:
            print(f"    FAILED: Local intensity analysis indicates regions are too similar")
            return None
        
        total_original = np.sum(cell_region)
        total_separated = np.sum(region1) + np.sum(region2)
        
        print(f"    Region conservation check: {total_original} -> {total_separated} voxels")
        
        if abs(total_separated - total_original) > total_original * 0.01:
            print(f"    WARNING: Significant voxel count mismatch!")
        
        if np.any(region1 & region2):
            print(f"    ERROR: Regions overlap!")
            return None
        
        print(f"    SUCCESSFUL WATERSHED SEPARATION with local validation")
        print(f"      Final regions: {np.sum(region1)} and {np.sum(region2)} voxels")
        print(f"      Local intensity difference: {relative_diff:.3f}")
        print(f"      Separation intensity ratio: {intensity_ratio:.3f}")
        
        return region1, region2, separation_intensity, mean_soma_intensity

    def validate_separation(region1, region2, bridge_intensity, soma_intensity):
        """Validate if separation results in valid cell regions"""
        print(f"    Validating separation:")
        print(f"      Region sizes: {np.sum(region1)}, {np.sum(region2)} (min threshold: {min_size_threshold})")
        
        if np.sum(region1) < min_size_threshold:
            print(f"      FAILED: Region 1 too small ({np.sum(region1)} < {min_size_threshold})")
            return False
        if np.sum(region2) < min_size_threshold:
            print(f"      FAILED: Region 2 too small ({np.sum(region2)} < {min_size_threshold})")
            return False
            
        intensity_ratio = bridge_intensity / soma_intensity
        print(f"      Bridge/Soma intensity ratio: {intensity_ratio:.3f} (threshold: {min_path_intensity_ratio})")
        
        if intensity_ratio >= min_path_intensity_ratio:
            print(f"      FAILED: Bridge too bright ({intensity_ratio:.3f} >= {min_path_intensity_ratio})")
            return False
            
        print(f"      PASSED: All validation criteria met")
        return True

    # --- NEW HELPER FUNCTION FOR A SINGLE SEPARATION ATTEMPT ---
    def attempt_one_separation(
        cell_region: np.ndarray,
        soma_regions_in_cell: List[np.ndarray]
    ) -> Optional[Tuple[np.ndarray, List[np.ndarray], np.ndarray, List[np.ndarray]]]:
        """
        Tries to find and perform one valid separation on the given region.
        If successful, returns the two new regions and their respective soma lists.
        If not, returns None.
        """
        bbox = get_cell_bounding_box(cell_region)
        if bbox is None: return None

        local_cell = cell_region[bbox]
        local_intensity = intensity_volume[bbox]

        soma_centroids, local_soma_regions = [], []
        for i, soma_region in enumerate(soma_regions_in_cell):
            local_soma = soma_region[bbox]
            if np.any(local_soma):
                coords = np.where(local_soma)
                centroid_local = [np.mean(coords[j]) for j in range(3)]
                centroid_global = [centroid_local[j] + bbox[j].start for j in range(3)]
                soma_centroids.append(centroid_global)
                local_soma_regions.append((i, local_soma))

        if len(local_soma_regions) < 2: return None

        valid_pairs = []
        for i in range(len(soma_centroids)):
            for j in range(i + 1, len(soma_centroids)):
                dist = calculate_physical_distance(soma_centroids[i], soma_centroids[j], spacing)
                if dist > max_seed_centroid_dist:
                    valid_pairs.append((i, j, dist))
        
        if not valid_pairs: return None
        
        valid_pairs.sort(key=lambda x: x[2], reverse=True)

        for soma_i, soma_j, _ in valid_pairs:
            local_soma1 = local_soma_regions[soma_i][1]
            local_soma2 = local_soma_regions[soma_j][1]
            
            separation_result = find_connecting_path_and_cut_plane(local_cell, local_soma1, local_soma2, local_intensity)
            if separation_result is None: continue

            region1_local, region2_local, bridge_intensity, soma_intensity = separation_result
            if not validate_separation(region1_local, region2_local, bridge_intensity, soma_intensity): continue

            # Success! Now process the result.
            global_region1 = np.zeros_like(cell_region, dtype=bool)
            global_region2 = np.zeros_like(cell_region, dtype=bool)
            global_region1[bbox] = region1_local
            global_region2[bbox] = region2_local

            somas_for_region1, somas_for_region2 = [], []
            for soma_region in soma_regions_in_cell:
                if np.any(soma_region & global_region1):
                    somas_for_region1.append(soma_region)
                else:
                    somas_for_region2.append(soma_region)
            
            return global_region1, somas_for_region1, global_region2, somas_for_region2

        return None # No valid separation was found among all pairs

    # Main processing
    print("=== Starting Recursive Multi-Soma Cell Separation ===")
    print(f"Input parameters:")
    print(f"  Segmentation mask shape: {segmentation_mask.shape}")
    print(f"  Intensity volume shape: {intensity_volume.shape}")
    print(f"  Soma mask shape: {soma_mask.shape}")
    print(f"  Spacing: {spacing}")
    print(f"  Min size threshold: {min_size_threshold}")
    print(f"  Max soma distance: {max_seed_centroid_dist}")
    print(f"  Min intensity ratio: {min_path_intensity_ratio}")
    print(f"  Min local intensity difference: {min_local_intensity_difference}")
    print(f"  Local analysis radius: {local_analysis_radius}")

    result_mask = segmentation_mask.copy()
    from skimage.measure import label # type: ignore
    
    cell_labels = np.unique(segmentation_mask[segmentation_mask > 0])
    print(f"\nFound {len(cell_labels)} cells to process")
    
    cells_processed = 0
    total_separations = 0
    next_available_label = np.max(result_mask) + 1
    
    for cell_idx, cell_label in enumerate(cell_labels):
        print(f"\n{'='*80}")
        print(f"Processing cell {cell_idx + 1}/{len(cell_labels)}: Label {cell_label}")
        
        cell_region = (segmentation_mask == cell_label)
        cell_soma_mask = soma_mask * cell_region
        if not np.any(cell_soma_mask):
            print("  No somas found - skipping.")
            continue
            
        labeled_somas, num_somas = label(cell_soma_mask > 0, return_num=True)
        if num_somas < 2:
            print(f"  Only {num_somas} soma region(s) - no separation needed.")
            continue
        
        cells_processed += 1
        print(f"  Found {num_somas} soma regions. Starting iterative separation.")

        soma_regions_in_cell = [labeled_somas == i for i in range(1, num_somas + 1)]

        # --- THE NEW ITERATIVE LOOP ---
        final_regions = []
        regions_to_process = deque([(cell_region, soma_regions_in_cell)])
        
        # Safety break to prevent infinite loops
        max_iterations = num_somas * 2 
        
        while regions_to_process and max_iterations > 0:
            max_iterations -= 1
            current_region, current_somas = regions_to_process.popleft()
            
            # Try to perform one separation on the current region
            separation_result = attempt_one_separation(current_region, current_somas)
            
            if separation_result:
                # Separation was successful, add the two new sub-regions back to the queue
                region1, somas1, region2, somas2 = separation_result
                regions_to_process.append((region1, somas1))
                regions_to_process.append((region2, somas2))
                total_separations += 1
                print(f"    Separation successful. Queue size: {len(regions_to_process)}")
            else:
                # No more separations possible for this region, it's considered final
                final_regions.append(current_region)
                print(f"    Region finalized. Total final regions: {len(final_regions)}")
        
        if max_iterations <= 0:
            print("  WARNING: Max iterations reached. Moving remaining regions to final.")
            # Add any remaining unprocessed regions to the final list
            while regions_to_process:
                final_regions.append(regions_to_process.popleft()[0])

        # --- END OF THE ITERATIVE LOOP ---

        print(f"\n  SEPARATION COMPLETE for cell {cell_label}:")
        print(f"    Original cell -> {len(final_regions)} final regions")

        result_mask[cell_region] = 0 # Clear the original cell
        
        for region_idx, region_mask in enumerate(final_regions):
            label_to_use = cell_label if region_idx == 0 else next_available_label
            if region_idx > 0: next_available_label += 1
            
            result_mask[region_mask] = label_to_use
            print(f"      Region {region_idx + 1}: label {label_to_use}, size {np.sum(region_mask)}")

        del cell_region, final_regions, regions_to_process
        gc.collect()

    # ... (Your summary and post-processing code remains the same) ...
    print("\n--- SUMMARY ---")
    # ...
    
    return result_mask