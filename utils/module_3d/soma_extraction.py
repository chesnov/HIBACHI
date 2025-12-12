# --- START OF FILE utils/module_3d/soma_extraction.py ---
import numpy as np
import os
from scipy import ndimage
from skimage.measure import regionprops # type: ignore
from skimage.feature import peak_local_max # type: ignore
from skimage.segmentation import watershed # type: ignore
from sklearn.decomposition import PCA # type: ignore
import math
import time
import gc
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from tqdm import tqdm
import traceback
from shutil import rmtree

# Import shared helpers
try:
    from .segmentation_helpers import (
        flush_print, 
        log_memory_usage, 
        distance_transform_edt, 
        _watershed_with_simpleitk
    )
except ImportError:
    # Fallback for running script directly
    from segmentation_helpers import (
        flush_print, 
        log_memory_usage, 
        distance_transform_edt, 
        _watershed_with_simpleitk
    )

def get_min_distance_pixels(spacing, physical_distance):
    """Calculates minimum distance in pixels for peak_local_max based on physical distance."""
    min_spacing_yx = min(spacing[1:]) # Use YX for in-plane separation
    if min_spacing_yx <= 1e-6: return 3
    pixels = int(round(physical_distance / min_spacing_yx))
    return max(3, pixels)

def _filter_candidate_fragment_3d(
    fragment_mask: np.ndarray, 
    parent_dt: np.ndarray,
    spacing: Tuple[float, float, float],
    min_seed_fragment_volume: int,
    min_accepted_core_volume: float,
    max_accepted_core_volume: float,
    min_accepted_thickness_um: float, 
    max_accepted_thickness_um: float, 
    max_allowed_core_aspect_ratio: float
) -> Tuple[str, Optional[str], Optional[np.ndarray], Optional[float], float]:
    """
    Filters a single 3D candidate fragment based on shape, size, and thickness rules.
    """
    t_start = time.time()
    try:
        fragment_volume = np.sum(fragment_mask)
        if fragment_volume < min_seed_fragment_volume:
            return 'discard', 'too_small', None, None, time.time() - t_start

        # Thickness Check
        max_dist_in_fragment_um = np.max(parent_dt[fragment_mask])
        passes_thickness = (min_accepted_thickness_um <= max_dist_in_fragment_um <= max_accepted_thickness_um)
        
        # Volume Check
        passes_volume_range = (min_accepted_core_volume <= fragment_volume <= max_accepted_core_volume)

        # Aspect Ratio Check (PCA)
        passes_aspect = True
        if fragment_volume > 3: 
            try:
                coords_vox = np.argwhere(fragment_mask)
                coords_phys = coords_vox * np.array(spacing)
                pca = PCA(n_components=3)
                pca.fit(coords_phys)
                eigenvalues = pca.explained_variance_ 
                eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
                
                if eigenvalues_sorted[2] > 1e-12: 
                    aspect_ratio = math.sqrt(eigenvalues_sorted[0]) / math.sqrt(eigenvalues_sorted[2])
                    if aspect_ratio > max_allowed_core_aspect_ratio:
                        passes_aspect = False
            except Exception:
                passes_aspect = True # Permissive on PCA error
        
        fragment_mask_copy = fragment_mask.copy()
        duration = time.time() - t_start

        if passes_thickness and passes_volume_range and passes_aspect:
            return 'valid', None, fragment_mask_copy, fragment_volume, duration
        else:
            reason = 'unknown'
            if not passes_thickness: reason = 'thickness'
            elif not passes_volume_range: reason = 'volume'
            elif not passes_aspect: reason = 'aspect'
            return 'fallback', reason, fragment_mask_copy, fragment_volume, duration
            
    except Exception as e_filt:
        flush_print(f"Warn: Error during 3D fragment filtering: {e_filt}")
        return 'discard', 'error', None, None, time.time() - t_start


def extract_soma_masks(
    segmentation_mask: np.ndarray, 
    intensity_image: np.ndarray, 
    spacing: Optional[Tuple[float, float, float]],
    smallest_quantile: int = 25,
    min_fragment_size: int = 30, 
    core_volume_target_factor_lower: float = 0.1,
    core_volume_target_factor_upper: float = 10,
    erosion_iterations: int = 0,
    ratios_to_process = [0.3, 0.4, 0.5, 0.6],
    intensity_percentiles_to_process: List[int] = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
    min_physical_peak_separation: float = 7.0, 
    # NEW PARAMETER: Decouples detection from filtering
    seeding_min_distance_um: Optional[float] = None,
    max_allowed_core_aspect_ratio: float = 10.0,
    ref_vol_percentile_lower: int = 30,
    ref_vol_percentile_upper: int = 70,
    ref_thickness_percentile_lower: int = 1,
    absolute_min_thickness_um: float = 1.5,
    absolute_max_thickness_um: float = 10.0,
    memmap_dir: Optional[str] = "ramiseg_temp_memmap",
    memmap_voxel_threshold: int = 25_000_000,
    memmap_final_mask: bool = True
) -> np.ndarray:
    """
    The Main Soma Extraction Logic.
    Iterates through objects, identifies potential multi-soma clumps, and attempts
    to find 'seeds' inside them using distance and intensity cues.
    """
    flush_print("--- Starting 3D Soma Extraction ---")
    
    # Setup
    use_memmap_feature = memmap_dir is not None
    if use_memmap_feature:
        os.makedirs(memmap_dir, exist_ok=True)

    min_seed_fragment_volume = max(1, min_fragment_size)
    MAX_CORE_VOXELS_FOR_WS = 500_000
    
    if spacing is None: spacing = (1.0, 1.0, 1.0)
    else: spacing = tuple(float(s) for s in spacing)

    # 1. Analyze Distribution of Object Sizes
    unique_labels_all, counts = np.unique(segmentation_mask[segmentation_mask > 0], return_counts=True)
    if len(unique_labels_all) == 0: 
        return np.zeros_like(segmentation_mask, dtype=np.int32)
    
    initial_volumes = dict(zip(unique_labels_all, counts))
    
    # Find slices for all objects
    label_to_slice = {}
    valid_unique_labels_list = []
    object_slices = ndimage.find_objects(segmentation_mask)
    for label in unique_labels_all:
        idx = label - 1
        if 0 <= idx < len(object_slices) and object_slices[idx]:
            label_to_slice[label] = object_slices[idx]
            valid_unique_labels_list.append(label)
            
    unique_labels = np.array(valid_unique_labels_list)
    all_volumes_list = list(initial_volumes.values())
    
    # 2. Determine Parameters relative to population
    smallest_thresh_volume = np.percentile(all_volumes_list, smallest_quantile) if len(all_volumes_list) > 1 else (all_volumes_list[0] if all_volumes_list else 1)
    
    smallest_object_labels_set = {l for l in unique_labels if initial_volumes[l] <= smallest_thresh_volume}
    
    target_soma_volume = np.median([v for l,v in initial_volumes.items() if l in smallest_object_labels_set] or all_volumes_list)
    min_accepted_core_volume = max(min_seed_fragment_volume, target_soma_volume * core_volume_target_factor_lower)
    max_accepted_core_volume = target_soma_volume * core_volume_target_factor_upper
    
    # Calculate Ref Thickness
    vol_thresh_lower = np.percentile(all_volumes_list, ref_vol_percentile_lower)
    vol_thresh_upper = np.percentile(all_volumes_list, ref_vol_percentile_upper)
    reference_labels = [l for l in unique_labels if vol_thresh_lower < initial_volumes[l] <= vol_thresh_upper]
    if len(reference_labels) < 5: reference_labels = list(unique_labels)
    
    # Sample thickness
    max_thicknesses = []
    sample_refs = reference_labels[:min(20, len(reference_labels))]
    for l in sample_refs:
        sl = label_to_slice[l]
        m = (segmentation_mask[sl] == l)
        if np.any(m):
            dt = ndimage.distance_transform_edt(m, sampling=spacing)
            max_thicknesses.append(np.max(dt))
    
    if max_thicknesses:
        calc_min_thick = np.percentile(max_thicknesses, ref_thickness_percentile_lower)
        if absolute_min_thickness_um < 0.1: # Heuristic: User set a manual "catch-all"
             min_accepted_thickness_um = absolute_min_thickness_um
        else:
             min_accepted_thickness_um = max(absolute_min_thickness_um, calc_min_thick)
    else:
        min_accepted_thickness_um = absolute_min_thickness_um
        
    min_accepted_thickness_um = min(min_accepted_thickness_um, absolute_max_thickness_um - 0.1)
    
    # --- DECOUPLED DISTANCE LOGIC ---
    # 1. Internal Seeding Distance: Used for peak_local_max to detect seeds
    if seeding_min_distance_um is not None:
        internal_seed_pixels = get_min_distance_pixels(spacing, seeding_min_distance_um)
    else:
        # Fallback: If user didn't specify, we default to the physical sep,
        # but this mimics the old behavior which might merge things if set too high.
        internal_seed_pixels = get_min_distance_pixels(spacing, min_physical_peak_separation)

    # 2. Global Filter Distance: Used to delete duplicates at the end
    # (No pixel conversion needed here, we do it in physical space later)
    
    gc.collect()

    # 3. Initialize Result Mask
    final_seed_mask_path = None
    final_seed_mask = None
    
    if use_memmap_feature and memmap_final_mask:
        final_seed_mask_path = os.path.join(memmap_dir, 'final_seed_mask.mmp')
        final_seed_mask = np.memmap(final_seed_mask_path, dtype=np.int32, mode='w+', shape=segmentation_mask.shape)
    else:
        final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
        
    next_final_label = 1
    added_small_labels = set()

    try:
        # 4. Process Small Cells (Pass-through)
        small_cell_labels = [l for l in unique_labels if l in smallest_object_labels_set]
        flush_print(f"Processing {len(small_cell_labels)} small objects (Assumed Single)...")
        
        for label in tqdm(small_cell_labels, desc="Small Cells"):
            coords = np.argwhere(segmentation_mask == label) 
            if coords.size > 0:
                final_seed_mask[coords[:,0], coords[:,1], coords[:,2]] = next_final_label
                next_final_label += 1
                added_small_labels.add(label)
        
        # 5. Process Large Cells (The Hunt for Somas)
        large_cell_labels = [l for l in unique_labels if l not in added_small_labels]
        flush_print(f"Processing {len(large_cell_labels)} large objects (Finding Cores)...")
        
        struct_el_erosion = ndimage.generate_binary_structure(3, 3) if erosion_iterations > 0 else None

        for label in tqdm(large_cell_labels, desc="Large Cells"):
            bbox_slice = label_to_slice.get(label)
            if not bbox_slice: continue
            
            # Setup Local Crop
            pad = 1
            slice_obj = tuple(slice(max(0, s.start-pad), min(sh, s.stop+pad)) for s, sh in zip(bbox_slice, segmentation_mask.shape))
            local_shape = tuple(s.stop - s.start for s in slice_obj)
            offset = (slice_obj[0].start, slice_obj[1].start, slice_obj[2].start)
            num_voxels = np.prod(local_shape)
            
            # Mask within crop
            seg_sub = segmentation_mask[slice_obj]
            mask_sub = (seg_sub == label)
            if not np.any(mask_sub): continue
            
            # Use Memmap for DT if object is huge
            use_memmap_local = use_memmap_feature and (num_voxels > memmap_voxel_threshold)
            dt_obj = None
            dt_path = None
            
            if use_memmap_local:
                dt_path = os.path.join(memmap_dir, f"cell_{label}_dt.mmp")
                dt_memmap = np.memmap(dt_path, dtype=np.float64, mode='w+', shape=(3,)+local_shape) 
                dt_obj = ndimage.distance_transform_edt(mask_sub, sampling=spacing)
                del dt_memmap
            else:
                dt_obj = ndimage.distance_transform_edt(mask_sub, sampling=spacing)
            
            max_dist = np.max(dt_obj)
            
            # List to store candidates: (priority_score, volume, global_coords)
            candidates_with_score = []
            
            # Inner function to process a "Core" mask
            def process_core(core_mask, dt_map, priority_score):
                lbl_cores, n_cores = ndimage.label(core_mask)
                if n_cores == 0: return
                
                for c_idx in range(1, n_cores + 1):
                    c_mask = (lbl_cores == c_idx)
                    c_vol = np.sum(c_mask)
                    if c_vol < min_seed_fragment_volume: continue
                    
                    ws_labels = np.zeros_like(dt_map, dtype=np.int32)
                    
                    if c_vol < MAX_CORE_VOXELS_FOR_WS:
                        core_dt = dt_map.copy()
                        core_dt[~c_mask] = 0
                        
                        # USE THE DECOUPLED INTERNAL DISTANCE
                        peaks = peak_local_max(core_dt, min_distance=internal_seed_pixels, labels=c_mask, exclude_border=False)
                        
                        if peaks.shape[0] > 1:
                            markers = np.zeros_like(ws_labels)
                            markers[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                            ws_labels = watershed(-core_dt, markers, mask=c_mask)
                        else:
                            ws_labels[c_mask] = 1
                    else:
                        ws_labels[c_mask] = 1
                        
                    for f_id in np.unique(ws_labels[ws_labels > 0]):
                        f_mask = (ws_labels == f_id)
                        res, _, mask_final, vol, _ = _filter_candidate_fragment_3d(
                            f_mask, dt_map, spacing,
                            min_seed_fragment_volume, min_accepted_core_volume, max_accepted_core_volume,
                            min_accepted_thickness_um, absolute_max_thickness_um, max_allowed_core_aspect_ratio
                        )
                        if res == 'valid':
                            locs = np.argwhere(mask_final)
                            locs += np.array(offset)
                            # Store with priority
                            candidates_with_score.append((priority_score, vol, locs))

            # --- A. Distance Transform Strategy ---
            if max_dist > 1e-9:
                local_ratios = list(ratios_to_process)
                needed_min_ratio = min_accepted_thickness_um / max_dist
                
                # Check for small nucleus ratio rescue
                if needed_min_ratio < min(local_ratios) and needed_min_ratio > 0.05:
                    local_ratios.append(needed_min_ratio)
                
                local_ratios = sorted(local_ratios, reverse=True)

                for ratio in local_ratios:
                    thresh = max_dist * ratio
                    initial_core = (dt_obj >= thresh) & mask_sub
                    if erosion_iterations > 0:
                        initial_core = ndimage.binary_erosion(initial_core, structure=struct_el_erosion, iterations=erosion_iterations)
                    
                    process_core(initial_core, dt_obj, priority_score=ratio)

            # --- B. Intensity Strategy ---
            int_sub = intensity_image[slice_obj]
            vals = int_sub[mask_sub]
            
            if vals.size > 0:
                for perc in intensity_percentiles_to_process:
                    p_thresh = np.percentile(vals, perc)
                    int_core = (int_sub >= p_thresh) & mask_sub
                    if erosion_iterations > 0:
                        int_core = ndimage.binary_erosion(int_core, structure=struct_el_erosion, iterations=erosion_iterations)
                    
                    if np.any(int_core):
                        core_dt = ndimage.distance_transform_edt(int_core, sampling=spacing)
                        process_core(int_core, core_dt, priority_score=2.0)

            # 6. Place Best Candidates (With Global Proximity Check)
            if candidates_with_score:
                # Sort primarily by Priority (High to Low)
                candidates_with_score.sort(key=lambda x: (x[0], x[1]), reverse=True)
                
                # List to track physical centroids of placed seeds
                placed_centroids_phys = []

                for _, _, glob_coords in candidates_with_score:
                    z, y, x = glob_coords[:,0], glob_coords[:,1], glob_coords[:,2]
                    
                    # 1. Check for pixel overlap
                    existing = final_seed_mask[z, y, x]
                    if np.any(existing > 0):
                        continue
                    
                    # 2. Check for physical proximity (Global Filter)
                    # We compute the centroid of the new candidate
                    current_cent_z = np.mean(z) * spacing[0]
                    current_cent_y = np.mean(y) * spacing[1]
                    current_cent_x = np.mean(x) * spacing[2]
                    current_cent_phys = np.array([current_cent_z, current_cent_y, current_cent_x])
                    
                    too_close = False
                    # Compare against all previously placed seeds for this object
                    for placed_c in placed_centroids_phys:
                        dist = np.linalg.norm(current_cent_phys - placed_c)
                        # USE THE STRICT GLOBAL PARAMETER HERE
                        if dist < min_physical_peak_separation:
                            too_close = True
                            break
                    
                    if too_close:
                        continue

                    # Place the seed
                    final_seed_mask[z, y, x] = next_final_label
                    next_final_label += 1
                    placed_centroids_phys.append(current_cent_phys)
            
            del seg_sub, mask_sub, dt_obj, int_sub
            gc.collect()

        if isinstance(final_seed_mask, np.memmap):
            final_seed_mask.flush()
            return final_seed_mask
        else:
            return final_seed_mask

    except Exception as e:
        flush_print(f"CRITICAL ERROR in extract_soma_masks: {e}")
        traceback.print_exc()
        return np.zeros_like(segmentation_mask, dtype=np.int32)
    finally:
        if 'final_seed_mask' in locals(): del final_seed_mask
        gc.collect()
# --- END OF FILE utils/module_3d/soma_extraction.py ---