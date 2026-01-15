"""
2D Feature Calculation Module
=============================

This module provides comprehensive morphometric, topological, and spatial 
analysis for 2D segmented objects. It is designed to mirror the 3D analysis 
logic exactly, ensuring that results from different dimensions are 
statistically comparable.

Key Features:
- Disk-backed N x N distance matrix calculation.
- Topology-preserving skeletonization with global connectivity safety.
- Anisotropy-aware perimeter and morphology metrics.
- Multi-threaded processing with Copy-on-Write memory optimization.
"""

import os
import sys
import tempfile
import multiprocessing as mp
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pandas as pd
import gc
import math
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, remove_small_holes
from skimage.measure import regionprops, find_contours
from skan import Skeleton, summarize
from tqdm.auto import tqdm

# --- Standardized FCS Export Logic ---
try:
    import fcswrite  # type: ignore
except ImportError:
    fcswrite = None
    print("Warning: 'fcswrite' library not found. FCS export will be disabled.")


def flush_print(*args: Any, **kwargs: Any) -> None:
    """
    Standardized print wrapper for real-time logging.
    Forces buffer flushing to ensure logs are visible during heavy CPU tasks.
    """
    print(*args, **kwargs)
    sys.stdout.flush()


# --- Multiprocessing Shared Cache ---
# Used to share contour coordinates with worker processes without pickling.
_ALL_CONTOURS: List[np.ndarray] = []


# =============================================================================
# 1. DISTANCE QUANTIFICATION (Two-Pass High-Precision System)
# =============================================================================

def _calculate_row_distances_worker_2d(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Worker Pass 1: Computes Euclidean distances for a triangular matrix row.

    Args:
        args: (row_index, total_labels, physical_spacing_array)

    Returns:
        Tuple: (row_index, numpy_array_of_float32_distances)
    """
    i, n_proc, spacing_arr = args
    row_dists = np.full(n_proc - (i + 1), np.inf, dtype=np.float32)

    p1 = _ALL_CONTOURS[i]
    if p1.shape[0] == 0:
        return i, row_dists

    # Build cKDTree in physical (micron) space
    tree = cKDTree(p1 * spacing_arr)

    for idx, j in enumerate(range(i + 1, n_proc)):
        p2 = _ALL_CONTOURS[j]
        if p2.shape[0] > 0:
            # Query point cloud of object J against the tree of object I
            dists, _ = tree.query(p2 * spacing_arr, k=1)
            row_dists[idx] = np.min(dists)

    return i, row_dists


def _extract_winning_points_worker_2d(args: Tuple) -> Dict[str, Any]:
    """
    Worker Pass 2: Extracts exact coordinates for identified nearest neighbors.
    Only triggered for the N object pairs that represent closest contacts.
    """
    label_i, label_j, idx_i, idx_j, spacing_arr = args
    p1, p2 = _ALL_CONTOURS[idx_i], _ALL_CONTOURS[idx_j]

    tree = cKDTree(p1 * spacing_arr)
    dists, indices = tree.query(p2 * spacing_arr, k=1)

    # Locate absolute minimum contact point
    min_idx_in_p2 = np.argmin(dists)
    p1_pt = p1[indices[min_idx_in_p2]]
    p2_pt = p2[min_idx_in_p2]

    return {
        'mask1': int(label_i),
        'mask2': int(label_j),
        'point_on_self_y': float(p1_pt[0]),
        'point_on_self_x': float(p1_pt[1]),
        'point_on_neighbor_y': float(p2_pt[0]),
        'point_on_neighbor_x': float(p2_pt[1])
    }


def shortest_distance_2d(
    segmented_array: np.ndarray,
    spacing: Tuple[float, float] = (1.0, 1.0),
    temp_dir: Optional[str] = None,
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coordinates the N x N Distance Matrix and coordinate extraction.
    Ensures O(N) memory for coordinates by using a two-pass architecture.
    """
    global _ALL_CONTOURS
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    spacing_arr = np.array(spacing)

    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    n_labels = len(labels)

    flush_print(f"\n[PROFILE DIST] Starting distance calculation for {n_labels} labels.")
    if n_labels <= 1:
        return pd.DataFrame(), pd.DataFrame()

    # --- Stage 1: Contour Cache ---
    _ALL_CONTOURS = []
    locations = ndi.find_objects(segmented_array)
    actual_labels = []

    for lbl in tqdm(labels, desc="    Contour Extraction"):
        sl = locations[lbl - 1]
        if sl is None:
            continue
        mask = (segmented_array[sl] == lbl)
        eroded = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(2, 1))
        y, x = np.where(mask ^ eroded)
        if len(y) > 0:
            _ALL_CONTOURS.append(np.column_stack((y + sl[0].start, x + sl[1].start)))
            actual_labels.append(lbl)

    n_valid = len(actual_labels)
    flush_print(f"[PROFILE DIST] Found {n_valid} valid masks with contours.")

    # --- Stage 2: Pass 1 (Memory-Mapped Distance Matrix) ---
    # Use the project-specific temp_dir if provided, otherwise system temp
    target_dir = temp_dir if (temp_dir and os.path.isdir(temp_dir)) else tempfile.gettempdir()
    mmap_path = os.path.join(target_dir, f"dist_mat_2d_{os.getpid()}.dat")
    
    dist_mat_mm = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n_valid, n_valid))
    dist_mat_mm[:] = np.inf
    np.fill_diagonal(dist_mat_mm, 0)

    tasks = [(i, n_valid, spacing_arr) for i in range(n_valid)]
    with mp.Pool(n_jobs) as pool:
        for i, row_results in tqdm(pool.imap_unordered(_calculate_row_distances_worker_2d, tasks),
                                  total=n_valid, desc="    Distance Pass 1/2"):
            dist_mat_mm[i, i + 1:] = row_results
            dist_mat_mm[i + 1:, i] = row_results

    # --- Stage 3: Pass 2 (Coordinate Recovery) ---
    winning_pairs = []
    for i in range(n_valid):
        row = dist_mat_mm[i].copy()
        row[i] = np.inf
        j = np.argmin(row)
        if not np.isinf(row[j]):
            winning_pairs.append((actual_labels[i], actual_labels[j], i, j, spacing_arr))

    with mp.Pool(n_jobs) as pool:
        points_list = list(tqdm(pool.imap_unordered(_extract_winning_points_worker_2d, winning_pairs),
                               total=len(winning_pairs), desc="    Distance Pass 2/2"))

    # Convert binary matrix to Pandas and cleanup
    dist_df = pd.DataFrame(np.array(dist_mat_mm), index=actual_labels, columns=actual_labels)
    points_df = pd.DataFrame(points_list)

    _ALL_CONTOURS = []
    del dist_mat_mm
    if os.path.exists(mmap_path):
        os.remove(mmap_path)

    return dist_df, points_df


# =============================================================================
# 2. SKELETONIZATION (Process-Preservation Logic)
# =============================================================================

def _prune_internal_artifacts_2d(skeleton_binary: np.ndarray) -> np.ndarray:
    """
    Refined Artifact Pruner: Ensures 1-pixel width without eating branches.
    Targets diagonal 2x2 elbow blocks that cause spur regrowth.
    """
    pruned = skeleton_binary.copy().astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    for _ in range(3):
        changed = False
        candidates = np.argwhere(pruned == 1)
        for y, x in candidates:
            if ndi.convolve(pruned, kernel, mode='constant', cval=0)[y, x] < 2:
                continue

            y0, y1, x0, x1 = max(0, y-1), min(pruned.shape[0], y+2), max(0, x-1), min(pruned.shape[1], x+2)
            roi = pruned[y0:y1, x0:x1].copy()
            cy, cx = y - y0, x - x0

            # Connectivity Test: Removal is safe only if component count stays constant
            _, before = ndi.label(roi, structure=np.ones((3, 3)))
            roi[cy, cx] = 0
            _, after = ndi.label(roi, structure=np.ones((3, 3)))

            if before == after and before > 0:
                pruned[y, x] = 0
                changed = True
        if not changed:
            break
    return pruned.astype(bool)


def _break_skeleton_cycles_2d(skeleton_binary: np.ndarray, mask_dt: np.ndarray) -> np.ndarray:
    """
    Global-Safety Cycle Breaker:
    Trial-snaps loop pixels and only accepts the snap if global connectivity 
    is preserved. Breaks loops at the thinnest biological points.
    """
    if not np.any(skeleton_binary):
        return skeleton_binary

    working_skel = skeleton_binary.copy().astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    for _ in range(15):
        # PEELING: Strip terminal branches to isolate cycle cores
        temp_peel = working_skel.copy()
        while True:
            neighs = ndi.convolve(temp_peel, kernel, mode='constant', cval=0)
            endpoints = (temp_peel == 1) & (neighs == 1)
            if not np.any(endpoints): break
            temp_peel[endpoints] = 0

        if not np.any(temp_peel): break

        # SNAP TRIAL: Resolve cycles via global safety check
        labeled_loops, num_loops = ndi.label(temp_peel, structure=np.ones((3, 3)))
        _, initial_b0 = ndi.label(working_skel, structure=np.ones((3, 3)))

        for i in range(1, num_loops + 1):
            candidates = np.argwhere(labeled_loops == i)
            dt_vals = [mask_dt[tuple(c)] for c in candidates]
            for idx in np.argsort(dt_vals):
                y, x = candidates[idx]
                working_skel[y, x] = 0
                _, post_snap_b0 = ndi.label(working_skel, structure=np.ones((3, 3)))
                if post_snap_b0 == initial_b0:
                    break  # Success
                else:
                    working_skel[y, x] = 1 # Restore
    return working_skel.astype(bool)


def _trace_spur_length_and_path_2d(
    skel: np.ndarray,
    start_y: int,
    start_x: int,
    spacing: Tuple[float, float],
    sl: Tuple[slice, slice],
    image_shape: Tuple[int, int]
) -> Tuple[float, List[Tuple[int, int]], bool, bool]:
    """
    Traces a terminal branch from an endpoint to its parent junction.

    This function follows a single-pixel path starting from a degree-1 voxel.
    It calculates the total physical length and compiles a list of coordinates 
    targeted for removal. Crucially, it excludes the junction pixel from the 
    path to ensure that pruning can be performed without disconnecting the 
    remainder of the skeleton.

    Args:
        skel: Binary skeleton mask (expected to be padded by 10 pixels).
        start_y: Local Y coordinate of the starting endpoint.
        start_x: Local X coordinate of the starting endpoint.
        spacing: Physical pixel dimensions (Y, X) in microns.
        sl: Bounding box slices of the label within the global image.
        image_shape: Global dimensions of the full image (Y, X).

    Returns:
        Tuple containing:
        - total_len (float): Physical Euclidean length of the branch in microns.
        - path (list): List of (y, x) coordinates representing the branch pixels 
                       to be deleted (excludes the junction pixel).
        - is_spur (bool): True if the branch connects to a junction, False 
                          if it is a standalone isolated segment.
        - hits_boundary (bool): True if the starting endpoint touches the 
                                Field of View (FOV) edge.
    """
    visited = set()
    current = (start_y, start_x)
    path = []
    total_len = 0.0

    # 8-connectivity neighborhood offsets
    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    # --- 1. FOV BOUNDARY ANALYSIS ---
    # Map local coordinates to global space using the 10-pixel padding offset
    global_y = (start_y - 10) + sl[0].start
    global_x = (start_x - 10) + sl[1].start

    # Determine if the endpoint is truncated by the edge of the FOV
    hits_boundary = (
        global_y <= 0 or global_y >= image_shape[0] - 1 or
        global_x <= 0 or global_x >= image_shape[1] - 1
    )

    # --- 2. BRANCH TRAVERSAL ---
    while True:
        visited.add(current)
        cy, cx = current
        
        # Identify valid neighbors for the next step in the path
        neighs = []
        for dy, dx in offsets:
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < skel.shape[0] and 
                0 <= nx < skel.shape[1] and 
                skel[ny, nx] and 
                (ny, nx) not in visited):
                neighs.append((ny, nx))

        # --- CASE A: TERMINAL END OR STANDALONE ---
        # If no neighbors exist, we have reached a dead end or finished a segment.
        if len(neighs) == 0:
            path.append(current)
            return total_len, path, False, hits_boundary

        # --- CASE B: JUNCTION ENCOUNTERED ---
        # If multiple neighbors exist, we have hit a junction. 
        if len(neighs) > 1:
            # VITAL: We stop tracing HERE. 
            # We do NOT add the junction pixel to the 'path' list, 
            # effectively protecting it from deletion.
            return total_len, path, True, hits_boundary

        # --- CASE C: PATH CONTINUATION ---
        # Add current pixel to deletion set and move to the only neighbor.
        path.append(current)
        nxt = neighs[0]

        # Sum physical distance scaled by anisotropic spacing
        dy_um = (nxt[0] - cy) * spacing[0]
        dx_um = (nxt[1] - cx) * spacing[1]
        total_len += math.sqrt(dy_um**2 + dx_um**2)

        current = nxt

def _prune_skeleton_spurs_2d(skel, spacing, max_len, sl, image_shape):
    """Sequential, real-time connectivity-safe spur pruning logic."""
    if max_len <= 0: return skel
    working_skel = skel.copy().astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    for _ in range(10):
        # Identify endpoints
        neigh_counts = ndi.convolve(working_skel, kernel, mode='constant', cval=0)
        endpoints = (working_skel == 1) & (neigh_counts == 1)
        coords = np.argwhere(endpoints)
        if len(coords) == 0: break

        branch_candidates = []
        for y, x in coords:
            L, P, is_att, hits = _trace_spur_length_and_path_2d(working_skel, y, x, spacing, sl, image_shape)
            branch_candidates.append({'len': L, 'path': P, 'is_spur': is_att, 'hits_edge': hits, 'y': y, 'x': x})

        # Process shortest first to maintain topological stability
        branch_candidates.sort(key=lambda x: x['len'])
        changed = False

        for cand in branch_candidates:
            if working_skel[cand['y'], cand['x']] == 0: continue
            if cand['hits_edge'] or not cand['is_spur'] or cand['len'] > max_len: continue

            _, b0_prev = ndi.label(working_skel, structure=np.ones((3, 3)))
            test_skel = working_skel.copy()
            for py, px in cand['path']: test_skel[py, px] = 0
            _, b0_post = ndi.label(test_skel, structure=np.ones((3, 3)))

            if b0_post <= b0_prev:
                working_skel = test_skel
                changed = True
        if not changed: break
    return working_skel.astype(bool)

def calculate_ramification_with_skan_2d(
    segmented_array: np.ndarray,
    spacing: Tuple[float, float],
    skeleton_export_path: Optional[str],
    prune_spurs_le_um: float
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Coordinates 2D skeletonization with Logic for Ring-shaped cells.
    - Selective hole filling: prevents skeletons floating in vacuoles.
    - Global-safety cycle breaking: turns rings into lines on the mask 'flesh'.
    - Sequential real-time pruning: removes length-1 spurs without fragmentation.
    """
    original_shape = segmented_array.shape
    use_memmap = (skeleton_export_path is not None)
    if use_memmap:
        os.makedirs(os.path.dirname(skeleton_export_path), exist_ok=True)
        skel_out = np.memmap(skeleton_export_path, dtype=np.int32, mode='w+', shape=original_shape)
    else: skel_out = np.zeros(original_shape, dtype=np.int32)
        
    labels = np.unique(segmented_array); labels = labels[labels > 0]
    locations = ndi.find_objects(segmented_array)
    stats_list, detailed_dfs = [], []

    def get_topology_stats(mask):
        if not np.any(mask): return 0, 0
        _, b0 = ndi.label(mask, structure=np.ones((3, 3)))
        _, b1 = ndi.label(~mask, structure=ndi.generate_binary_structure(2, 1))
        return b0, b1 - 1

    for lbl in tqdm(labels, desc="    Skeletonizing"):
        idx = lbl - 1
        if idx >= len(locations) or locations[idx] is None: continue
        sl = locations[idx]; offset = np.array([s.start for s in sl])
        mask = (segmented_array[sl] == lbl).astype(bool)
        if not np.any(mask): continue
        
        # 1. Mask Prep (Selective filling for vacuoles)
        mask_padded = np.pad(mask, pad_width=10, mode='constant', constant_values=0)
        area_thresh_px = int(10.0 / (spacing[0] * spacing[1])) 
        mask_padded = remove_small_holes(mask_padded, area_threshold=area_thresh_px)
        mask_dt = ndi.distance_transform_edt(mask_padded)
        
        # 2. Skeletonize (Lee method is used for best reach into bulbs)
        try: skel = skeletonize(mask_padded, method='lee')
        except: skel = skeletonize(mask_padded)
        
        raw_b0, raw_b1 = get_topology_stats(skel)

        # 3. Refinement: break artifacts and cycles
        skel = _prune_internal_artifacts_2d(skel)
        if raw_b1 > 0:
            skel = _break_skeleton_cycles_2d(skel, mask_dt)
        post_b0, post_b1 = get_topology_stats(skel)

        # 4. User-parameter Pruning
        if prune_spurs_le_um > 0:
            skel = _prune_skeleton_spurs_2d(skel, spacing, prune_spurs_le_um, sl, original_shape)
            
        # 5. Final Shave: Thin again and remove sub-pixel 'elbow' pixels
        skel = skeletonize(skel.astype(bool))
        skel = _prune_internal_artifacts_2d(skel)
        skel = _prune_skeleton_spurs_2d(skel, spacing, min(spacing) * 1.1, sl, original_shape)
        
        final_b0, final_b1 = get_topology_stats(skel)

        # Verbose Profiling
        if raw_b1 > 0 or final_b0 > 1:
            flush_print(f"\n      [DEBUG TOPOLOGY] Label {lbl}: Initial Loops={raw_b1}, Final Comp={final_b0}, Final Loops={final_b1}")

        skel_crop = skel[10:-10, 10:-10]
        skel_out[sl][skel_crop.astype(bool)] = lbl

        # Detailed Skan Analysis
        skan_len, skan_branches, avg_len = 0.0, 0, 0.0
        if np.any(skel_crop):
            try:
                skel_obj = Skeleton(skel_crop, spacing=spacing)
                summ = summarize(skel_obj, separator='-')
                if not summ.empty:
                    summ['label'] = int(lbl)
                    for c in summ.columns:
                        if 'coord' in c: summ[c] += offset[int(c.split('-')[-1])]
                    detailed_dfs.append(summ)
                    skan_len, skan_branches, avg_len = summ['branch-distance'].sum(), len(summ), summ['branch-distance'].mean()
            except: pass

        # Primary Table Metrics
        kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
        neighbors = ndi.convolve(skel_crop.astype(np.uint8), kernel, mode='constant', cval=0)
        n_end = np.count_nonzero((skel_crop > 0) & (neighbors == 1))
        n_junc = np.count_nonzero((skel_crop > 0) & (neighbors >= 3))
        
        stats_list.append({
            'label': int(lbl), 'true_num_branches': max(0, n_end - 1 + n_junc),
            'skan_total_length_um': skan_len, 'skan_avg_branch_length_um': avg_len, 
            'true_num_junctions': n_junc, 'true_num_endpoints': n_end, 
            'skan_num_skeleton_pixels': np.count_nonzero(skel_crop)
        })
        del mask, mask_padded, mask_dt, skel, skel_crop
        if lbl % 100 == 0:
            gc.collect()

    if use_memmap: skel_out.flush()
    return pd.DataFrame(stats_list), pd.concat(detailed_dfs, ignore_index=True) if detailed_dfs else pd.DataFrame(), skel_out

# =============================================================================
# 3. MORPHOLOGY & EXPORT (Numerical Parity with 3D)
# =============================================================================

def calculate_morphology_2d(segmented_array, spacing):
    """Calculates precisely integrated morphology metrics."""
    props = regionprops(segmented_array, spacing=spacing)
    if not props: return pd.DataFrame()
    res = []
    px_area = spacing[0] * spacing[1]
    for p in tqdm(props, desc="    Morphology"):
        perimeter_um = 0.0
        try:
            padded = np.pad(p.image, 1, mode='constant')
            for c in find_contours(padded, 0.5):
                delta = np.diff(c, axis=0) * np.array(spacing)
                perimeter_um += np.sum(np.sqrt(np.sum(delta**2, axis=1)))
        except: perimeter_um = np.nan

        area = p.area * px_area
        circ = (4 * np.pi * area) / (perimeter_um**2) if perimeter_um > 1e-6 else np.nan
        res.append({'label': int(p.label), 'area_um2': area, 'perimeter_um': perimeter_um,
                    'pixel_count': p.area, 'circularity': min(1.0, circ) if not np.isnan(circ) else np.nan,
                    'solidity': p.solidity, 'eccentricity': p.eccentricity})
    return pd.DataFrame(res)


def calculate_intensity_2d(segmented_array, intensity_image):
    """Calculates intensity statistics including Median for 3D Parity."""
    labels = np.unique(segmented_array); labels = labels[labels > 0]
    locs = ndi.find_objects(segmented_array); res = []
    for lbl in tqdm(labels, desc="    Intensity"):
        idx = int(lbl) - 1
        if idx >= len(locs) or locs[idx] is None: continue
        sl = locs[idx]; mask = (segmented_array[sl] == lbl)
        vals = intensity_image[sl][mask]
        if vals.size > 0:
            res.append({'label': int(lbl), 'mean_intensity': np.mean(vals), 'median_intensity': np.median(vals),
                        'std_intensity': np.std(vals), 'integrated_density': np.sum(vals), 'max_intensity': np.max(vals)})
    return pd.DataFrame(res)


def export_to_fcs(metrics_df, fcs_path):
    """Exports metrics to FCS (Logical Parity with 3D)."""
    if not fcs_path or fcswrite is None or metrics_df is None or metrics_df.empty: return
    try:
        flush_print(f"  [Export] Writing FCS: {os.path.basename(fcs_path)}")
        num_df = metrics_df.select_dtypes(include=[np.number]).copy()
        num_df.replace([np.inf, -np.inf], np.nan, inplace=True); num_df.fillna(0, inplace=True)
        if 'label' in metrics_df.columns: num_df['label'] = metrics_df['label']
        fcswrite.write_fcs(filename=fcs_path, chn_names=list(num_df.columns), data=num_df.values)
    except Exception as e: flush_print(f"  [Export] Error: {e}")


# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================

def analyze_segmentation_2d(
    segmented_array: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    spacing_yx: Tuple[float, float] = (1.0, 1.0),
    calculate_distances: bool = True,
    calculate_skeletons: bool = True,
    skeleton_export_path: Optional[str] = None,
    fcs_export_path: Optional[str] = None,
    temp_dir: Optional[str] = None,
    n_jobs: Optional[int] = None,
    return_detailed: bool = False,
    prune_spurs_le_um: float = 0.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Main 2D analysis suite. 
    Achieves full numerical parity with the 3D pipeline.
    """
    flush_print("\n--- Starting Feature Calculation (2D) ---")
    
    # 1. MORPHOLOGY: metrics_df initialization
    metrics_df = calculate_morphology_2d(segmented_array, spacing=spacing_yx)
    if metrics_df.empty: return pd.DataFrame(), {}
    
    # Parity: Depth placeholder for table width consistency
    metrics_df['depth_um'] = 0.0
    detailed_outputs = {}

    # 2. INTENSITY: Median parity
    if intensity_image is not None:
        int_df = calculate_intensity_2d(segmented_array, intensity_image)
        if not int_df.empty:
            metrics_df = pd.merge(metrics_df, int_df, on='label', how='outer')

    # 3. DISTANCES: High-precision join logic
    if calculate_distances:
        dist_df, pts_df = shortest_distance_2d(segmented_array, spacing_yx, temp_dir, n_jobs)
        if not dist_df.empty:
            temp_mat = dist_df.values.copy(); np.fill_diagonal(temp_mat, np.inf)
            min_dist = np.nanmin(temp_mat, axis=1)
            neighbor_indices = np.nanargmin(temp_mat, axis=1)
            closest_neighs = dist_df.columns[neighbor_indices]
            
            dist_metrics = pd.DataFrame({
                'label': dist_df.index.astype(int), 
                'shortest_distance_um': min_dist, 
                'closest_neighbor_label': closest_neighs.astype(int)
            })
            metrics_df['label'] = metrics_df['label'].astype(int)
            metrics_df = pd.merge(metrics_df, dist_metrics, on='label', how='left')
            
            if return_detailed and not pts_df.empty:
                detailed_outputs['distance_matrix'] = dist_df
                # Type-safe join to prepare Red Lines for Orchestrator
                pts_df['m1_k'], pts_df['m2_k'] = pts_df['mask1'].astype(int), pts_df['mask2'].astype(int)
                f_keys = dist_metrics[['label', 'closest_neighbor_label']].copy()
                f_keys.columns = ['m1_k', 'm2_k']
                nearest_pts = pd.merge(pts_df, f_keys, on=['m1_k', 'm2_k'], how='inner')
                detailed_outputs['all_pairs_points'] = nearest_pts.drop(columns=['m1_k', 'm2_k'])
                flush_print(f"      [DEBUG DIST] Red lines data prepared for Orchestrator: {len(detailed_outputs['all_pairs_points'])} rows.")

    # 4. SKELETONS
    if calculate_skeletons:
        summ_skel, detail_skel, skel_arr = calculate_ramification_with_skan_2d(segmented_array, spacing_yx, skeleton_export_path, prune_spurs_le_um)
        if not summ_skel.empty:
            summ_skel['label'] = summ_skel['label'].astype(int)
            metrics_df = pd.merge(metrics_df, summ_skel, on='label', how='left')
        if return_detailed:
            detailed_outputs['detailed_branches'] = detail_skel
            detailed_outputs['skeleton_array'] = skel_arr

    # 5. FCS EXPORT
    if fcs_export_path: export_to_fcs(metrics_df, fcs_export_path)
    
    flush_print("--- Analysis Complete (2D) ---")
    return metrics_df, detailed_outputs if return_detailed else {}