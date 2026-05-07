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
    """
    Calculates minimum distance in pixels for peak detection.

    Uses the minimum in-plane resolution (YX) to determine the pixel separation
    required to satisfy a physical distance requirement.

    Args:
        spacing: Pixel spacing (Y, X).
        physical_distance: Desired minimum separation in physical units (um).

    Returns:
        int: Minimum distance in pixels (minimum of 3).
    """
    min_spacing = min(spacing)
    if min_spacing <= 1e-6:
        return 3
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
    min_fragment_size: int = 30,
    erosion_iterations: int = 0,
    ratios_to_process: List[float] = [0.3, 0.4, 0.5, 0.6],
    intensity_percentiles_to_process: List[int] = [100, 90, 80, 70, 60, 50, 40, 30],
    min_physical_peak_separation: float = 7.0,
    max_allowed_core_aspect_ratio: float = 10.0,
    absolute_min_thickness_um: float = 1.5,
    absolute_max_thickness_um: float = 10.0,
    tile_size_threshold: int = 2048,
    pixel_area_threshold: int = 4_000_000,
    memmap_output_path: Optional[str] = None,
    **kwargs,
) -> np.ndarray:
    """
    Memory-Efficient 2D Soma Extraction with Label-First Processing.

    Logically equivalent to the 3D implementation (extract_soma_masks). All
    morphological checks, recovery paths, and placement logic mirror the 3D
    version exactly; only dimensionality-specific calls differ (2D DT, 2D PCA,
    2D tile generation).

    STRICT LOGIC PRESERVATION: Higher percentiles/ratios always take precedence.
    """
    t_start_global = time.time()
    print("\n" + "="*60)
    print("2D SOMA EXTRACTION: STARTING")
    print("="*60)
    
    # 1. Parameter Validation & Setup
    if spacing is None:
        spacing = (1.0, 1.0)
    else:
        spacing = tuple(float(s) for s in spacing)

    min_seed_vol = max(1, min_fragment_size)

    # Consolidated peak separation used for both global deduplication and internal splitting
    int_peak_sep = get_min_distance_pixels_2d(spacing, min_physical_peak_separation)

    # Dynamic tile padding: ensure context covers at least one soma radius + buffer
    tile_padding = int(absolute_max_thickness_um / min(spacing) + 2)

    # Identify objects (find_objects is much more RAM efficient than regionprops)
    slices = ndimage.find_objects(segmentation_mask)
    valid_labels = [i+1 for i, s in enumerate(slices) if s is not None]
    if not valid_labels: 
        return np.zeros_like(segmentation_mask, dtype=np.int32)

    # 2. Absolute Mode Initialization & Profiling
    print(f"  Absolute Mode Enforced: Processing {len(valid_labels)} labels...")
    print(f"  Thresh: Min Volume = {min_seed_vol} pixels")
    print(f"  Thresh: Thickness = [{absolute_min_thickness_um:.2f} - {absolute_max_thickness_um:.2f}] µm")
    print(f"  Thresh: Peak Separation = {min_physical_peak_separation:.2f} µm")

    diag_stats = {
        "cores_evaluated": 0,
        "cores_too_small": 0,
        "thickness_rejected": 0,
        "aspect_ratio_rejected": 0,
        "spatial_overlap_rejected": 0
    }

    # 3. Output Mask Initialization
    if memmap_output_path:
        final_seed_mask = np.memmap(memmap_output_path, dtype='int32', mode='w+', shape=segmentation_mask.shape)
        final_seed_mask[:] = 0
    else:
        final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32)

    next_label_id = 1
    all_placed_centroids = []
    spatial_index: Optional[KDTree] = None

    # 4. Strategy Definitions
    # Strategies are ordered by Strict Priority Score (higher score = wins overlap).
    # We add (val / 1000) to the score to ensure higher percentiles/ratios always rank higher.
    strategies = []
    for p in sorted(intensity_percentiles_to_process, reverse=True):
        strategies.append({'type': 'Int', 'val': p, 'score': 2.0 + (p / 1000.0)})
    for r in sorted(ratios_to_process, reverse=True):
        strategies.append({'type': 'DT', 'val': r, 'score': r + (r / 1000.0)})

    strategies.sort(key=lambda x: x['score'], reverse=True)

    # 5. Main Processing (Label-by-Label)
    main_pbar = tqdm(valid_labels, desc="Total Labels", unit="label", dynamic_ncols=True)

    for lbl_idx, lbl in enumerate(main_pbar):
        sl = slices[lbl-1]
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        is_huge = (h * w) > pixel_area_threshold

        tiles = _generate_2d_tiles(sl, tile_size_threshold, padding=tile_padding)
        label_candidates = []

        # Per-label deduplication list: prevents two seeds from the same merged
        # object being placed too close. Cross-label deduplication is handled
        # by the pixel-overlap check below.
        label_placed_peaks: List = []

        # Sub-progress bar for giant cell clumps
        tile_iter = tqdm(tiles, desc=f"  ↳ Clump {lbl}", leave=False, unit="tile", disable=not is_huge)
        
        for t_idx, t in enumerate(tile_iter):
            p0, p1 = t['pad'][0:2], t['pad'][2:4]
            t_mask = (segmentation_mask[p0[0]:p1[0], p0[1]:p1[1]] == lbl)
            if not np.any(t_mask):
                continue
            
            t_int = intensity_image[p0[0]:p1[0], p0[1]:p1[1]]
            offset = np.array([p0[0], p0[1]])
            
            # Context-wide distance transform (computed once per tile)
            dt_obj = ndimage.distance_transform_edt(t_mask, sampling=spacing)
            max_dt_val = np.max(dt_obj)

            def process_frag_logic(mask_arr, sub_off):
                """Checks morphological validity and converts to global coords."""
                local_coords = np.argwhere(mask_arr)
                tile_coords = local_coords + sub_off
                g_coords = tile_coords + offset

                # Thickness: max inscribed radius sampled from the full-object DT.
                # Always use dt_obj (the full label DT) regardless of which strategy
                # generated this fragment — matches 3D logic exactly.
                dt_vals = dt_obj[tuple(tile_coords.T)]
                max_thick = np.max(dt_vals)

                # Lower bound: hard rejection — fragment is too thin regardless of
                # how it is sub-sampled.
                if max_thick < absolute_min_thickness_um:
                    diag_stats["thickness_rejected"] += 1
                    return

                # Upper bound: rather than discarding the whole fragment, attempt to
                # recover a sub-kernel — the pixels whose inscribed-circle radius is
                # within the accepted thickness window.  This preserves somas that
                # have already been selected from a neighbouring strategy while still
                # honouring the morphological constraint at the kernel level.
                if max_thick > absolute_max_thickness_um:
                    min_allowed_dt = max_thick - absolute_max_thickness_um
                    within_upper = dt_vals >= min_allowed_dt

                    if not np.any(within_upper):
                        diag_stats["thickness_rejected"] += 1
                        return

                    sub_dt_vals = dt_vals[within_upper]

                    # Effective thickness is the internal radius from the new
                    # boundary to the peak.
                    effective_thickness = np.max(sub_dt_vals) - np.min(sub_dt_vals)
                    if effective_thickness < absolute_min_thickness_um:
                        diag_stats["thickness_rejected"] += 1
                        return

                    # Narrow all coordinate arrays to the valid sub-kernel pixels.
                    local_coords = local_coords[within_upper]
                    tile_coords  = tile_coords[within_upper]
                    g_coords     = g_coords[within_upper]
                    dt_vals      = sub_dt_vals
                    max_thick    = np.max(sub_dt_vals)
                    sub_min      = local_coords.min(axis=0)
                    sub_shape    = local_coords.max(axis=0) - sub_min + 1
                    mask_arr     = np.zeros(sub_shape, dtype=bool)
                    mask_arr[tuple((local_coords - sub_min).T)] = True
                    local_coords = local_coords - sub_min

                # DT peak pixel — nucleus geometric centre regardless of strategy.
                peak_idx = int(np.argmax(dt_vals))
                peak_coord_g = g_coords[peak_idx]

                # 2D PCA elongation check
                if mask_arr.sum() > 10:
                    try:
                        coords_phys = local_coords * np.array(spacing)
                        pca = PCA(n_components=2).fit(coords_phys)
                        ev = np.sort(np.abs(pca.explained_variance_))[::-1]
                        if (
                            ev[1] > 1e-12
                            and (math.sqrt(ev[0]) / math.sqrt(ev[1]))
                            > max_allowed_core_aspect_ratio
                        ):
                            # Recovery: shave off elongated tails (lower DT pixels)
                            # to isolate the roughly-circular peak region.
                            core_threshold = max_thick - (absolute_min_thickness_um * 0.5)
                            valid_core = dt_vals >= core_threshold

                            if np.sum(valid_core) < min_seed_vol:
                                diag_stats["aspect_ratio_rejected"] += 1
                                return

                            # Update coordinates to represent only the recovered core.
                            g_coords = g_coords[valid_core]
                            mask_arr = valid_core  # 1D bool; mask_arr.sum() = new count
                    except Exception:
                        pass

                # Tiling check: centroid must fall inside this tile's target box
                # to prevent duplicate detections across overlapping tiles.
                cent = np.mean(g_coords, axis=0)
                if (
                    t['target'][0] <= cent[0] < t['target'][2]
                    and t['target'][1] <= cent[1] < t['target'][3]
                ):
                    label_candidates.append({
                        'coords': g_coords.astype(np.int32),
                        'peak_coord': peak_coord_g,
                        'vol': mask_arr.sum(),
                        'score': strat['score'],
                        'strat_name': f"{strat['type']}_{strat['val']}",
                        'frag_max_thick': max_thick,
                    })

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
                    if thresh <= 0:
                        continue
                    core = (dt_obj >= thresh) & t_mask
                    dt_ref = dt_obj
                else:
                    # Intensity Percentile Strategy
                    vals = t_int[t_mask]
                    if vals.size == 0:
                        continue
                    core = (t_int >= np.percentile(vals, strat['val'])) & t_mask
                    # Recalculate local DT for intensity peaks
                    dt_ref = ndimage.distance_transform_edt(core, sampling=spacing)

                # Early Stopping: skip if core is already too small
                if np.sum(core) < min_seed_vol:
                    continue

                if erosion_iterations > 0:
                    core = ndimage.binary_erosion(core, iterations=erosion_iterations)
                    if not np.any(core):
                        continue

                # Fragment Extraction using vectorized regionprops
                labeled_core, num_cores = ndimage.label(core)
                for region in regionprops(labeled_core):
                    diag_stats["cores_evaluated"] += 1
                    if region.area < min_seed_vol:
                        diag_stats["cores_too_small"] += 1
                        continue

                    # Local Watershed Splitting for fused somas
                    frag_crop = region.image
                    frag_dt = ndimage.distance_transform_edt(frag_crop, sampling=spacing)
                    peaks = peak_local_max(frag_dt, min_distance=int_peak_sep, labels=frag_crop)

                    if len(peaks) > 1:
                        markers = np.zeros(frag_crop.shape, dtype=np.int32)
                        for idx, pk in enumerate(peaks):
                            markers[pk[0], pk[1]] = idx + 1
                        ws = watershed(-frag_dt, markers, mask=frag_crop)
                        for ws_id in range(1, len(peaks) + 1):
                            m_ws = (ws == ws_id)
                            if m_ws.sum() >= min_seed_vol:
                                process_frag_logic(m_ws, region.bbox[:2])
                    else:
                        process_frag_logic(region.image, region.bbox[:2])

                del core
                if 'dt_ref' in locals() and dt_ref is not dt_obj:
                    del dt_ref

            del t_mask, t_int, dt_obj
            if t_idx % 5 == 0:
                gc.collect()

        # 6. Placement (Greedy based on Priority and Spatial Separation)
        if label_candidates:
            # SORTING IS CRITICAL:
            # 1. Primary:   Strategy Score (highest percentile wins)
            # 2. Secondary: Volume (if score ties, larger soma wins)
            label_candidates.sort(key=lambda x: (x['score'], x['vol']), reverse=True)

            for cand in label_candidates:
                coords = cand['coords']
                peak_phys = cand['peak_coord'] * np.array(spacing)

                # Within-label proximity gate: prevents two seeds from the same
                # merged clump being placed too close together.
                if label_placed_peaks:
                    dists = np.linalg.norm(
                        np.array(label_placed_peaks) - peak_phys, axis=1
                    )
                    min_dist = np.min(dists)
                    if min_dist < min_physical_peak_separation:
                        diag_stats["spatial_overlap_rejected"] += 1
                        continue
                    else:
                        # --- THE TRAP: multiple somas placed within one label ---
                        print(
                            f"\n  [TRAP] Label {lbl} got MULTIPLE somas!"
                            f"\n    -> New Edge/Extra Soma: Strategy {cand.get('strat_name', 'Unknown')}, "
                            f"Vol {cand['vol']}, Thick {cand.get('frag_max_thick', 0):.1f}"
                            f"\n    -> Distance to nearest existing soma: {min_dist:.1f} µm "
                            f"(Limit is {min_physical_peak_separation:.1f} µm)"
                        )

                # Pixel Overlap Check (cross-label deduplication)
                idx_tuple = tuple(coords.T)
                if np.any(final_seed_mask[idx_tuple] > 0):
                    diag_stats["spatial_overlap_rejected"] += 1
                    continue

                # Place Seed
                final_seed_mask[idx_tuple] = next_label_id
                next_label_id += 1
                label_placed_peaks.append(peak_phys)
                all_placed_centroids.append(peak_phys)

        # Update main status
        main_pbar.set_postfix({"Seeds": next_label_id - 1, "RAM": f"{get_ram_usage():.1f}GB"})

        if 'label_candidates' in locals():
            label_candidates.clear()
            del label_candidates

        if lbl_idx % 20 == 0:
            gc.collect()

    t_total = time.time() - t_start_global
    print("\n" + "="*60)
    print("2D EXTRACTION COMPLETE")
    print(f"  Total Somas Placed: {next_label_id - 1}")
    print(f"  Execution Time: {t_total/60:.2f} mins")
    print("-" * 60)
    print("  DIAGNOSTICS (Absolute Mode Tracking):")
    print(f"    Total Core Fragments Evaluated: {diag_stats['cores_evaluated']}")
    print(f"    Rejected -> Too Small:          {diag_stats['cores_too_small']}")
    print(f"    Rejected -> Thickness Bound:    {diag_stats['thickness_rejected']}")
    print(f"    Rejected -> Aspect Ratio:       {diag_stats['aspect_ratio_rejected']}")
    print(f"    Rejected -> Spatial Overlap:    {diag_stats['spatial_overlap_rejected']}")
    print("="*60 + "\n")

    if isinstance(final_seed_mask, np.memmap):
        final_seed_mask.flush()

    return final_seed_mask