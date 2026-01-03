import time
import math
import gc
import os
import sys
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from scipy import ndimage
from scipy.spatial import KDTree 
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from sklearn.decomposition import PCA
from tqdm import tqdm

# Attempt to get psutil for RAM profiling, fallback if not installed
try:
    import psutil
    def get_ram_usage():
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
except ImportError:
    def get_ram_usage():
        return 0.0

def get_min_distance_pixels_2d(spacing: Tuple[float, float], physical_distance: float) -> int:
    min_spacing = min(spacing)
    if min_spacing <= 1e-6: return 3
    return max(3, int(round(physical_distance / min_spacing)))

def _generate_2d_tiles(bbox: Tuple[slice, slice], tile_size: int = 2048, padding: int = 40):
    y_range, x_range = bbox[0], bbox[1]
    y0, y1 = y_range.start, y_range.stop
    x0, x1 = x_range.start, x_range.stop
    tiles = []
    for y in range(y0, y1, tile_size):
        for x in range(x0, x1, tile_size):
            ty1, tx1 = min(y + tile_size, y1), min(x + tile_size, x1)
            py0, px0 = max(y0, y - padding), max(x0, x - padding)
            py1, px1 = min(y1, ty1 + padding), min(x1, tx1 + padding)
            tiles.append({'target': (y, x, ty1, tx1), 'pad': (py0, px0, py1, px1)})
    return tiles

def extract_soma_masks_2d(
    segmentation_mask: np.ndarray,
    intensity_image: np.ndarray,
    spacing: Tuple[float, float],
    smallest_quantile: float = 0.25,
    min_fragment_size: int = 15,
    core_volume_target_factor_lower: float = 0.1,
    core_volume_target_factor_upper: float = 10.0,
    erosion_iterations: int = 0,
    ratios_to_process: List[float] = [0.3, 0.4, 0.5, 0.6],
    intensity_percentiles_to_process: List[int] = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
    min_physical_peak_separation: float = 5.0,
    max_allowed_core_aspect_ratio: float = 10.0,
    ref_vol_percentile_lower: int = 30,
    ref_vol_percentile_upper: int = 70,
    ref_thickness_percentile_lower: int = 1,
    absolute_min_thickness_um: float = 1.0,
    absolute_max_thickness_um: float = 7.0,
    tile_size_threshold: int = 2048,
    memmap_output_path: Optional[str] = None
) -> np.ndarray:
    """
    Refactored Memory-Efficient 2D Soma Extraction with Label-First Processing.
    STRICT LOGIC PRESERVATION: Higher percentiles/ratios always take precedence.
    """
    t_start_global = time.time()
    print("\n" + "="*60)
    print("SOMA EXTRACTION: STARTING")
    print("="*60)
    
    # 1. Parameter Validation & Setup
    if spacing is None:
        spacing = (1.0, 1.0)
    else:
        spacing = tuple(float(s) for s in spacing)

    min_seed_fragment_area = max(1, min_fragment_size)
    min_peak_sep_pixels = get_min_distance_pixels_2d(spacing, min_physical_peak_separation)
    
    # Identify objects (find_objects is much more RAM efficient than regionprops)
    slices = ndimage.find_objects(segmentation_mask)
    valid_labels = [i+1 for i, s in enumerate(slices) if s is not None]
    if not valid_labels: 
        return np.zeros_like(segmentation_mask, dtype=np.int32)

    # 2. Population Analysis
    print(f"Analyzing {len(valid_labels)} objects for population statistics...")
    areas = []
    # Sample objects for area statistics
    sample_indices = valid_labels if len(valid_labels) < 500 else valid_labels[::5]
    for lbl in sample_indices:
        sl = slices[lbl-1]
        areas.append(np.sum(segmentation_mask[sl] == lbl))
    
    areas = np.array(areas)
    vol_p_low = np.percentile(areas, ref_vol_percentile_lower)
    vol_p_high = np.percentile(areas, ref_vol_percentile_upper)
    
    target_median_area = np.median(areas[areas <= np.percentile(areas, smallest_quantile*100)])
    min_accepted_core_area = max(min_seed_fragment_area, target_median_area * core_volume_target_factor_lower)
    max_accepted_core_area = target_median_area * core_volume_target_factor_upper

    # Thickness Analysis on reference population
    max_thicknesses_um = []
    ref_labels = [lbl for i, lbl in enumerate(sample_indices) if vol_p_low < areas[i] <= vol_p_high]
    if len(ref_labels) < 5: ref_labels = valid_labels[:50]
    
    for lbl in ref_labels[:50]:
        sl = slices[lbl-1]
        m = (segmentation_mask[sl] == lbl)
        dt = ndimage.distance_transform_edt(m, sampling=spacing)
        max_thicknesses_um.append(np.max(dt))
    
    if max_thicknesses_um:
        calc_min = np.percentile(max_thicknesses_um, ref_thickness_percentile_lower)
        min_accepted_thick = max(absolute_min_thickness_um, calc_min)
    else:
        min_accepted_thick = absolute_min_thickness_um
    
    min_accepted_thick = min(min_accepted_thick, absolute_max_thickness_um - 0.1)
    
    print(f"  Thresh: Area [{min_accepted_core_area:.1f}-{max_accepted_core_area:.1f}]")
    print(f"  Thresh: Thick [{min_accepted_thick:.2f}-{absolute_max_thickness_um:.2f}]")

    # 3. Output Mask Initialization
    if memmap_output_path:
        final_seed_mask = np.memmap(memmap_output_path, dtype='int32', mode='w+', shape=segmentation_mask.shape)
        final_seed_mask[:] = 0
    else:
        final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32)

    next_label_id = 1
    all_placed_centroids = []
    spatial_index: Optional[KDTree] = None

    # Define Strategies with STRICT priority scores
    # We add (val / 1000) to the score to ensure higher percentiles/ratios always rank higher
    strategies = []
    for p in sorted(intensity_percentiles_to_process, reverse=True): 
        strategies.append({'type': 'Int', 'val': p, 'score': 2.0 + (p / 1000.0)})
    for r in sorted(ratios_to_process, reverse=True): 
        # For ratios, higher ratio = smaller core = more specific. 
        # Note: ratios_to_process is usually [0.3, 0.4, 0.5, 0.6]
        strategies.append({'type': 'DT', 'val': r, 'score': r + (r / 1000.0)})
    
    # Sort strategies globally so that when we process a label, we check them in order
    strategies.sort(key=lambda x: x['score'], reverse=True)

    # 4. Main Processing (Label-by-Label)
    main_pbar = tqdm(valid_labels, desc="Total Labels", unit="label", dynamic_ncols=True)
    
    for lbl_idx, lbl in enumerate(main_pbar):
        sl = slices[lbl-1]
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        is_huge = (h > tile_size_threshold or w > tile_size_threshold)
        
        tiles = _generate_2d_tiles(sl, tile_size_threshold)
        label_candidates = []

        # Sub-progress bar for giant microglia clumps
        tile_iter = tqdm(tiles, desc=f"  ↳ Clump {lbl}", leave=False, unit="tile", disable=not is_huge)
        
        for t_idx, t in enumerate(tile_iter):
            p0, p1 = t['pad'][0:2], t['pad'][2:4]
            t_mask = (segmentation_mask[p0[0]:p1[0], p0[1]:p1[1]] == lbl)
            if not np.any(t_mask): continue
            
            t_int = intensity_image[p0[0]:p1[0], p0[1]:p1[1]]
            offset = np.array([p0[0], p0[1]])
            
            # Context-wide distance transform (computed once per tile)
            dt_full_obj = ndimage.distance_transform_edt(t_mask, sampling=spacing)
            max_dt_val = np.max(dt_full_obj)

            # Strategy loop with Early Stopping
            for strat in strategies:
                if is_huge:
                    tile_iter.set_postfix({
                        "Strat": f"{strat['type']}{strat['val']}",
                        "Cands": len(label_candidates),
                        "RAM": f"{get_ram_usage():.1f}GB"
                    })

                if strat['type'] == 'DT':
                    thresh = max_dt_val * strat['val']
                    if thresh <= 0: continue
                    core_mask = (dt_full_obj >= thresh) & t_mask
                    dt_ref = dt_full_obj
                else:
                    # Intensity Percentile Strategy
                    vals = t_int[t_mask]
                    if vals.size == 0: continue
                    core_mask = (t_int >= np.percentile(vals, strat['val'])) & t_mask
                    # Recalculate local DT for intensity peaks
                    dt_ref = ndimage.distance_transform_edt(core_mask, sampling=spacing)

                # Early Stopping: If core is already too small for this priority, it won't yield somas
                if np.sum(core_mask) < min_seed_fragment_area:
                    continue 

                if erosion_iterations > 0:
                    core_mask = ndimage.binary_erosion(core_mask, iterations=erosion_iterations)
                    if not np.any(core_mask): continue

                # Fragment Extraction using vectorized regionprops
                labeled_core, num_cores = ndimage.label(core_mask)
                for region in regionprops(labeled_core):
                    if region.area < min_seed_fragment_area: continue
                    
                    # Local Watershed Splitting for fused somas
                    frag_crop = region.image
                    frag_dt = ndimage.distance_transform_edt(frag_crop, sampling=spacing)
                    peaks = peak_local_max(frag_dt, min_distance=min_peak_sep_pixels, labels=frag_crop)

                    def process_frag_logic(m, sub_off):
                        local_coords = np.argwhere(m)
                        tile_local_coords = local_coords + sub_off
                        g_coords = tile_local_coords + offset
                        
                        # Use parent DT for thickness validation
                        max_thick = np.max(dt_ref[tuple(tile_local_coords.T)])
                        if not (min_accepted_thick <= max_thick <= absolute_max_thickness_um): 
                            return

                        # Aspect Ratio Check (PCA)
                        if m.sum() > 5:
                            pca = PCA(n_components=2).fit(local_coords * np.array(spacing))
                            ev = np.sort(np.abs(pca.explained_variance_))[::-1]
                            if ev[1] > 1e-12 and (math.sqrt(ev[0]) / math.sqrt(ev[1])) > max_allowed_core_aspect_ratio: 
                                return

                        # Tile Boundary Logic: Centroid must be in target box to avoid duplicates
                        cent = np.mean(g_coords, axis=0)
                        if t['target'][0] <= cent[0] < t['target'][2] and t['target'][1] <= cent[1] < t['target'][3]:
                            # SCORE PRECENDENCE: We store the strat score. 
                            # Sorting later will ensure strat score wins over area.
                            label_candidates.append({
                                'coords': g_coords, 
                                'area': m.sum(), 
                                'score': strat['score']
                            })

                    if len(peaks) > 1:
                        markers = np.zeros(frag_crop.shape, dtype=np.int32)
                        for idx, pk in enumerate(peaks): markers[pk[0], pk[1]] = idx + 1
                        ws = watershed(-frag_dt, markers, mask=frag_crop)
                        for ws_id in range(1, len(peaks) + 1):
                            m_ws = (ws == ws_id)
                            if m_ws.sum() >= min_seed_fragment_area: 
                                process_frag_logic(m_ws, region.bbox[:2])
                    else:
                        process_frag_logic(region.image, region.bbox[:2])

                del core_mask
            
            del t_mask, t_int, dt_full_obj
            if t_idx % 10 == 0: gc.collect()

    # 5. Global Greedy Placement
    # In this Label-First version, we can place per label to save RAM, 
    # but we must ensure candidate priority within the label.
        if label_candidates:
            # SORTING IS CRITICAL HERE:
            # 1. Primary: Strategy Score (Highest percentile wins)
            # 2. Secondary: Area (If percentile is same, larger soma wins - though usually percentiles are unique now)
            label_candidates.sort(key=lambda x: (x['score'], x['area']), reverse=True)
            
            for cand in label_candidates:
                coords = cand['coords']
                cent_phys = np.mean(coords, axis=0) * np.array(spacing)
                
                # Check Global Conflict (Physical Proximity)
                if spatial_index is not None:
                    dist, _ = spatial_index.query(cent_phys, k=1)
                    if dist < min_physical_peak_separation: 
                        continue

                # Check Global Conflict (Pixel overlap)
                idx_tuple = tuple(coords.T)
                if np.any(final_seed_mask[idx_tuple] > 0): 
                    continue

                # Placement
                final_seed_mask[idx_tuple] = next_label_id
                next_label_id += 1
                all_placed_centroids.append(cent_phys)
                
                # Rebuild spatial index periodically
                if len(all_placed_centroids) % 500 == 0:
                    spatial_index = KDTree(all_placed_centroids)
            
            if all_placed_centroids: 
                spatial_index = KDTree(all_placed_centroids)

        # Update main status
        main_pbar.set_postfix({"Seeds": next_label_id - 1, "RAM": f"{get_ram_usage():.1f}GB"})
        del label_candidates
        if lbl_idx % 100 == 0: gc.collect()

    t_total = time.time() - t_start_global
    print("\n" + "="*60)
    print(f"EXTRACTION COMPLETE")
    print(f"  Total Somas: {next_label_id - 1}")
    print(f"  Execution Time: {t_total/60:.2f} mins")
    print("="*60 + "\n")
    
    if isinstance(final_seed_mask, np.memmap): 
        final_seed_mask.flush()
    
    return final_seed_mask