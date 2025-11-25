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
    
    # Separate "Small" (likely single) vs "Large" (likely clumped)
    smallest_object_labels_set = {l for l in unique_labels if initial_volumes[l] <= smallest_thresh_volume}
    
    # Heuristics for what a "valid" soma looks like
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
    sample_refs = reference_labels[:min(20, len(reference_labels))] # sample first 20 for speed
    for l in sample_refs:
        sl = label_to_slice[l]
        m = (segmentation_mask[sl] == l)
        if np.any(m):
            dt = ndimage.distance_transform_edt(m, sampling=spacing)
            max_thicknesses.append(np.max(dt))
    
    if max_thicknesses:
        calc_min_thick = np.percentile(max_thicknesses, ref_thickness_percentile_lower)
        min_accepted_thickness_um = max(absolute_min_thickness_um, calc_min_thick)
    else:
        min_accepted_thickness_um = absolute_min_thickness_um
        
    min_accepted_thickness_um = min(min_accepted_thickness_um, absolute_max_thickness_um - 0.1)
    min_peak_sep_pixels = get_min_distance_pixels(spacing, min_physical_peak_separation)
    
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
        # These are assumed to be single cells already
        small_cell_labels = [l for l in unique_labels if l in smallest_object_labels_set]
        flush_print(f"Processing {len(small_cell_labels)} small objects (Assumed Single)...")
        
        for label in tqdm(small_cell_labels, desc="Small Cells"):
            coords = np.argwhere(segmentation_mask == label) # Global search (slow if many small objs, consider optimize)
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
            
            valid_candidates = []
            
            # --- A. Distance Transform Strategy ---
            # Use Memmap for DT if object is huge
            use_memmap_local = use_memmap_feature and (num_voxels > memmap_voxel_threshold)
            dt_obj = None
            dt_path = None
            
            if use_memmap_local:
                dt_path = os.path.join(memmap_dir, f"cell_{label}_dt.mmp")
                dt_memmap = np.memmap(dt_path, dtype=np.float64, mode='w+', shape=(3,)+local_shape) # temp output buffer for custom edt
                # Note: custom edt needs fixing to match shapes, using standard scipy for now on chunks usually ok
                # Fallback to RAM for DT calculation logic simplicity here unless massive
                dt_obj = ndimage.distance_transform_edt(mask_sub, sampling=spacing)
                del dt_memmap
            else:
                dt_obj = ndimage.distance_transform_edt(mask_sub, sampling=spacing)
            
            max_dist = np.max(dt_obj)
            
            # Inner function to process a "Core" mask (thresholded region)
            def process_core(core_mask, dt_map):
                lbl_cores, n_cores = ndimage.label(core_mask)
                if n_cores == 0: return
                
                # For each core, check if it needs splitting via watershed
                for c_idx in range(1, n_cores + 1):
                    c_mask = (lbl_cores == c_idx)
                    c_vol = np.sum(c_mask)
                    if c_vol < min_seed_fragment_volume: continue
                    
                    # If core is huge, maybe just use it. If complex, watershed it.
                    ws_labels = np.zeros_like(dt_map, dtype=np.int32)
                    
                    if c_vol < MAX_CORE_VOXELS_FOR_WS:
                        # Detect peaks in this core
                        core_dt = dt_map.copy()
                        core_dt[~c_mask] = 0
                        peaks = peak_local_max(core_dt, min_distance=min_peak_sep_pixels, labels=c_mask, exclude_border=False)
                        
                        if peaks.shape[0] > 1:
                            markers = np.zeros_like(ws_labels)
                            markers[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                            ws_labels = watershed(-core_dt, markers, mask=c_mask)
                        else:
                            ws_labels[c_mask] = 1
                    else:
                        ws_labels[c_mask] = 1
                        
                    # Filter the resulting fragments
                    for f_id in np.unique(ws_labels[ws_labels > 0]):
                        f_mask = (ws_labels == f_id)
                        res, _, mask_final, vol, _ = _filter_candidate_fragment_3d(
                            f_mask, dt_map, spacing,
                            min_seed_fragment_volume, min_accepted_core_volume, max_accepted_core_volume,
                            min_accepted_thickness_um, absolute_max_thickness_um, max_allowed_core_aspect_ratio
                        )
                        if res == 'valid':
                            # Convert to global coords
                            locs = np.argwhere(mask_final)
                            locs += np.array(offset)
                            valid_candidates.append((vol, locs))

            # Run Distance Strategy
            if max_dist > 1e-9:
                for ratio in ratios_to_process:
                    thresh = max_dist * ratio
                    initial_core = (dt_obj >= thresh) & mask_sub
                    if erosion_iterations > 0:
                        initial_core = ndimage.binary_erosion(initial_core, footprint=struct_el_erosion, iterations=erosion_iterations)
                    process_core(initial_core, dt_obj)

            # --- B. Intensity Strategy ---
            # Load intensity crop
            int_sub = intensity_image[slice_obj]
            vals = int_sub[mask_sub]
            
            if vals.size > 0:
                # We found candidates via DT? If so, clear them if we find better intensity ones? 
                # Current logic: Accumulate valid ones, prioritize by volume later.
                
                for perc in intensity_percentiles_to_process:
                    p_thresh = np.percentile(vals, perc)
                    int_core = (int_sub >= p_thresh) & mask_sub
                    if erosion_iterations > 0:
                        int_core = ndimage.binary_erosion(int_core, footprint=struct_el_erosion, iterations=erosion_iterations)
                    
                    # We need a DT for this core for thickness checks.
                    # Calculate DT *of the core itself* to ensure it's thick enough.
                    if np.any(int_core):
                        core_dt = ndimage.distance_transform_edt(int_core, sampling=spacing)
                        process_core(int_core, core_dt)

            # 6. Place Best Candidates
            # We might have duplicates or overlapping candidates from different ratios/methods.
            # Strategy: Sort by volume (descending). Place. If overlaps existing placement, skip.
            if valid_candidates:
                valid_candidates.sort(key=lambda x: x[0], reverse=True) # Largest first
                
                for _, glob_coords in valid_candidates:
                    # Check overlap
                    z, y, x = glob_coords[:,0], glob_coords[:,1], glob_coords[:,2]
                    
                    # Check if these pixels are already claimed by a seed for THIS object
                    # (We don't want to overwrite other objects, but we are working in the bbox of just this object)
                    # Ideally, we check final_seed_mask.
                    existing = final_seed_mask[z, y, x]
                    if np.any(existing > 0):
                        # Already occupied?
                        # Simple logic: if significant overlap, skip.
                        # Strict logic: If ANY overlap, skip.
                        continue
                    
                    # Place
                    final_seed_mask[z, y, x] = next_final_label
                    next_final_label += 1
            
            # Cleanup per cell
            del seg_sub, mask_sub, dt_obj, int_sub
            gc.collect()

        if isinstance(final_seed_mask, np.memmap):
            # Return in-memory copy if small enough, or return memmap?
            # Strategy expects an object it can use. If it's a memmap, it persists on disk.
            # We should probably flush it.
            final_seed_mask.flush()
            return final_seed_mask
        else:
            return final_seed_mask

    except Exception as e:
        flush_print(f"CRITICAL ERROR in extract_soma_masks: {e}")
        traceback.print_exc()
        return np.zeros_like(segmentation_mask, dtype=np.int32)
    finally:
        # If we created a memmap, we don't delete the FILE here (the caller needs it),
        # but we can delete the local python objects.
        if 'final_seed_mask' in locals(): del final_seed_mask
        gc.collect()
# --- END OF FILE utils/module_3d/soma_extraction.py ---