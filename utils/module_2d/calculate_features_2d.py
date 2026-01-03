import os
import time
import gc
import math
import tempfile
import traceback
import multiprocessing as mp
from typing import Tuple, List, Dict, Optional, Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, binary_closing, disk
from skimage.measure import regionprops, find_contours  # type: ignore
from skimage.feature import peak_local_max  # type: ignore
from skan import Skeleton, summarize  # type: ignore
from tqdm.auto import tqdm

# --- Optional Import for FCS ---
try:
    import fcswrite  # type: ignore
except ImportError:
    fcswrite = None
    print("Warning: 'fcswrite' not installed. FCS export will be disabled.")


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    import sys
    sys.stdout.flush()


# =============================================================================
# 1. DISTANCE CALCULATION (Map-Reduce 2D)
# =============================================================================

def _extract_contour_worker(args):
    """
    Worker: Extracts contour coordinates for 2D objects.
    Returns numpy array with columns: [Y, X]
    """
    label_idx, label, region, offset, temp_dir = args

    try:
        mask = (region == label)
        if not np.any(mask):
            np.save(os.path.join(temp_dir, f"contour_{label}.npy"), np.empty((0, 2), dtype=int))
            return label_idx, 0

        # Find boundary
        eroded = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(2, 1))
        contour_mask = mask ^ eroded

        # np.where returns (y, x)
        y_local, x_local = np.where(contour_mask)

        # Global coordinates
        contour_points_global = np.column_stack((
            y_local + offset[0],
            x_local + offset[1]
        ))

        np.save(os.path.join(temp_dir, f"contour_{label}.npy"), contour_points_global)
        return label_idx, len(contour_points_global)

    except Exception:
        np.save(os.path.join(temp_dir, f"contour_{label}.npy"), np.empty((0, 2), dtype=int))
        return label_idx, 0


def _calculate_pair_distance_worker_2d(args):
    """
    Worker: Calculates shortest distance between two 2D contours.
    """
    i, j, label1, label2, temp_dir, spacing_arr = args

    try:
        # p1, p2 columns are [Y, X]
        p1 = np.load(os.path.join(temp_dir, f"contour_{label1}.npy"))
        p2 = np.load(os.path.join(temp_dir, f"contour_{label2}.npy"))

        if p1.shape[0] == 0 or p2.shape[0] == 0:
            return i, j, np.inf, np.full(4, np.nan)

        # spacing_arr is [Y_um, X_um]
        p1_um = p1 * spacing_arr
        p2_um = p2 * spacing_arr

        tree = cKDTree(p2_um)
        dists, indices = tree.query(p1_um, k=1)

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        # Closest points in voxel coordinates
        point_on_1 = p1[min_idx]          # [y1, x1]
        point_on_2 = p2[indices[min_idx]] # [y2, x2]

        # Result: [y1, x1, y2, x2]
        result_pts = np.concatenate([point_on_1, point_on_2])

        return i, j, min_dist, result_pts

    except Exception:
        return i, j, np.inf, np.full(4, np.nan)


def shortest_distance_2d(
    segmented_array: np.ndarray,
    spacing: Tuple[float, float] = (1.0, 1.0),
    temp_dir: Optional[str] = None,
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coordinator for Pairwise Distance Calculation (2D).
    Uses a file-based Map-Reduce approach to handle memory efficiently.
    """
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    spacing_arr = np.array(spacing)  # [Y, X]

    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    n_labels = len(labels)

    if n_labels <= 1:
        return pd.DataFrame(), pd.DataFrame()

    temp_dir_managed = False
    if temp_dir is None:
        try:
            # Try local dir first for speed/perms
            local_tmp = os.path.join(os.getcwd(), 'temp_dist_2d')
            os.makedirs(local_tmp, exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix="dist_2d_", dir=local_tmp)
        except OSError:
            temp_dir = tempfile.mkdtemp(prefix="dist_2d_")
        temp_dir_managed = True
    else:
        os.makedirs(temp_dir, exist_ok=True)

    try:
        # 1. Extract Contours
        flush_print(f"  [Dist] Extracting contours for {n_labels} objects...")
        locations = ndi.find_objects(segmented_array)

        extract_tasks = []
        labels_processed = []

        for i, lbl in enumerate(labels):
            idx = lbl - 1
            if idx < len(locations) and locations[idx] is not None:
                sl = locations[idx]
                # Pad slices
                padded_sl = []
                offset = []
                for dim_s, dim_len in zip(sl, segmented_array.shape):
                    start = max(0, dim_s.start - 1)
                    stop = min(dim_len, dim_s.stop + 1)
                    padded_sl.append(slice(start, stop))
                    offset.append(start)

                region = segmented_array[tuple(padded_sl)]
                # offset is [y_start, x_start]
                extract_tasks.append((i, lbl, region, np.array(offset), temp_dir))
                labels_processed.append(lbl)

        with mp.Pool(n_jobs) as pool:
            list(tqdm(pool.imap_unordered(_extract_contour_worker, extract_tasks),
                      total=len(extract_tasks), desc="    Contour Extraction"))

        # 2. Calculate Distances
        flush_print(f"  [Dist] Calculating pairwise distances...")
        pairs = []
        n_proc = len(labels_processed)
        for i in range(n_proc):
            for j in range(i + 1, n_proc):
                pairs.append((i, j, labels_processed[i], labels_processed[j], temp_dir, spacing_arr))

        if not pairs:
            return pd.DataFrame(), pd.DataFrame()

        with mp.Pool(n_jobs) as pool:
            results = list(tqdm(pool.imap_unordered(_calculate_pair_distance_worker_2d, pairs),
                                total=len(pairs), desc="    Distance Matrix"))

        # 3. Assemble Results
        dist_mat = np.full((n_proc, n_proc), np.inf, dtype=np.float32)
        np.fill_diagonal(dist_mat, 0)

        points_list = []

        for i, j, d, pts in results:
            dist_mat[i, j] = d
            dist_mat[j, i] = d

            if not np.isinf(d):
                l1 = labels_processed[i]
                l2 = labels_processed[j]
                # pts is [y1, x1, y2, x2]

                # Store connection L1 -> L2
                points_list.append({
                    'mask1': l1, 'mask2': l2,
                    'point_on_self_y': pts[0], 'point_on_self_x': pts[1],
                    'point_on_neighbor_y': pts[2], 'point_on_neighbor_x': pts[3]
                })
                # Store connection L2 -> L1
                points_list.append({
                    'mask1': l2, 'mask2': l1,
                    'point_on_self_y': pts[2], 'point_on_self_x': pts[3],
                    'point_on_neighbor_y': pts[0], 'point_on_neighbor_x': pts[1]
                })

        dist_df = pd.DataFrame(dist_mat, index=labels_processed, columns=labels_processed)
        points_df = pd.DataFrame(points_list)

        return dist_df, points_df

    finally:
        if temp_dir_managed and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# 2. BASIC METRICS
# =============================================================================

def calculate_morphology_2d(segmented_array, spacing=(1.0, 1.0)):
    """
    Calculates Area, Perimeter, and Shape metrics.
    Includes CORRECTED perimeter calculation for anisotropic spacing.
    """
    flush_print("  [Metrics] Calculating Morphology...")
    
    if segmented_array.ndim != 2:
        raise ValueError("Input must be 2D")

    # regionprops handles area correctly with spacing
    props = regionprops(segmented_array.astype(np.int32), spacing=spacing)
    
    results = []
    pixel_area_um2 = spacing[0] * spacing[1]

    for prop in tqdm(props, desc="    Morphology"):
        label = prop.label
        
        # Basic props
        area_um2 = prop.area * pixel_area_um2
        pixel_count = prop.area
        
        min_r, min_c, max_r, max_c = prop.bbox
        bbox_h_um = (max_r - min_r) * spacing[0]
        bbox_w_um = (max_c - min_c) * spacing[1]
        bbox_area_um2 = bbox_h_um * bbox_w_um
        
        # Corrected Perimeter Calculation
        perimeter_um = 0.0
        try:
            # Pad mask to ensure closed contours
            padded_mask = np.pad(prop.image, pad_width=1, mode='constant', constant_values=0)
            contours = find_contours(padded_mask, level=0.5)
            
            for contour in contours:
                # Difference between points
                delta = np.diff(contour, axis=0)
                # Scale by spacing (y, x)
                delta_scaled = delta * np.array(spacing)
                # Euclidean distance
                perimeter_um += np.sum(np.sqrt(np.sum(delta_scaled**2, axis=1)))
        except Exception:
            perimeter_um = np.nan

        # Circularity
        circularity = np.nan
        if perimeter_um > 1e-6:
            circularity = (4 * math.pi * area_um2) / (perimeter_um**2)
            circularity = min(1.0, max(0.0, circularity))

        results.append({
            'label': label,
            'area_um2': area_um2,
            'perimeter_um': perimeter_um,
            'pixel_count': pixel_count,
            'circularity': circularity,
            'eccentricity': prop.eccentricity,
            'solidity': prop.solidity,
            'major_axis_length_um': prop.major_axis_length,
            'minor_axis_length_um': prop.minor_axis_length,
            'bbox_area_um2': bbox_area_um2
        })

    return pd.DataFrame(results)


def calculate_intensity_2d(segmented_array, intensity_image):
    """Calculates intensity statistics."""
    flush_print("  [Metrics] Calculating Intensity...")
    
    if segmented_array.shape != intensity_image.shape:
        print("    Warning: Shape mismatch for intensity calc. Skipping.")
        return pd.DataFrame()

    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    
    locations = ndi.find_objects(segmented_array)
    results = []

    for lbl in tqdm(labels, desc="    Intensity"):
        idx = lbl - 1
        if idx >= len(locations) or locations[idx] is None: continue
        
        sl = locations[idx]
        mask = (segmented_array[sl] == lbl)
        vals = intensity_image[sl][mask]
        
        if vals.size > 0:
            results.append({
                'label': lbl,
                'mean_intensity': np.mean(vals),
                'median_intensity': np.median(vals),
                'std_intensity': np.std(vals),
                'integrated_density': np.sum(vals),
                'max_intensity': np.max(vals)
            })
        else:
            results.append({'label': lbl}) # Fill defaults later

    return pd.DataFrame(results)


# =============================================================================
# 3. SKELETONIZATION (2D)
# =============================================================================

def _break_skeleton_cycles_2d(skeleton_binary):
    """
    Robustly breaks cycles by ensuring the skeleton is thinned 
    before breaking, preventing 'thick' loops from surviving.
    """
    # 1. Ensure the skeleton is strictly 1-pixel thick first
    pruned = _prune_internal_artifacts_2d(skeleton_binary.astype(np.uint8))
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    
    for _ in range(20):
        # 2. Isolate the loop 'cores' by peeling endpoints
        core = pruned.copy()
        while True:
            neighbors = ndi.convolve(core, kernel, mode='constant', cval=0)
            endpoints = (core == 1) & (neighbors <= 1)
            if not np.any(endpoints):
                break
            core[endpoints] = 0
        
        if not np.any(core):
            break
            
        # 3. Identify junctions to avoid breaking them
        all_neighbors = ndi.convolve(pruned, kernel, mode='constant', cval=0)
        junctions = (pruned == 1) & (all_neighbors >= 3)
        
        # 4. Find one break-point per loop
        labeled_core, n = ndi.label(core, structure=np.ones((3,3)))
        for i in range(1, n + 1):
            component = (labeled_core == i)
            # Prefer breaking away from junctions
            candidates = np.argwhere(component & ~junctions)
            if len(candidates) == 0:
                candidates = np.argwhere(component)
            
            if len(candidates) > 0:
                by, bx = candidates[0]
                pruned[by, bx] = 0
                
    return pruned.astype(bool)

def _trace_spur_length_and_path_2d(skeleton, start_y, start_x, spacing):
    """Traces 2D spur from endpoint. Stops exactly BEFORE a junction."""
    if skeleton[start_y, start_x] == 0: return 0.0, []
    
    visited = set()
    current = (start_y, start_x)
    path = []
    total_len = 0.0
    
    offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    while True:
        visited.add(current)
        cy, cx = current
        
        neighbors = []
        for dy, dx in offsets:
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < skeleton.shape[0] and 
                0 <= nx < skeleton.shape[1] and 
                skeleton[ny, nx]):
                if (ny, nx) not in visited:
                    neighbors.append((ny, nx))
        
        if len(neighbors) == 1:
            path.append(current) 
            next_node = neighbors[0]
            dy_um = (next_node[0] - cy) * spacing[0]
            dx_um = (next_node[1] - cx) * spacing[1]
            total_len += np.sqrt(dy_um**2 + dx_um**2)
            current = next_node
        else:
            # Junction or isolated end reached
            break
        
    return total_len, path


def _prune_skeleton_spurs_2d(skeleton_binary, spacing, max_spur_length_um):
    """Prunes short spurs while protecting junctions."""
    if max_spur_length_um <= 0: return skeleton_binary
    
    pruned = skeleton_binary.copy().astype(np.uint8)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    
    for _ in range(5): 
        changed = False
        neighbors = ndi.convolve(pruned, kernel, mode='constant', cval=0)
        endpoints = (pruned == 1) & (neighbors == 1)
        
        if not np.any(endpoints): break
        
        coords = np.argwhere(endpoints)
        for y, x in coords:
            if pruned[y, x] == 0: continue
            
            length, path = _trace_spur_length_and_path_2d(pruned, y, x, spacing)
            if length <= max_spur_length_um and len(path) > 0:
                for py, px in path:
                    pruned[py, px] = 0
                changed = True
        if not changed: break
                
    return pruned


def _prune_internal_artifacts_2d(skeleton_binary):
    """Removes 'ladder' artifacts while preserving connectivity."""
    pruned = skeleton_binary.copy().astype(np.uint8)
    for _ in range(3):
        changed = False
        candidates = np.argwhere(pruned == 1)
        for r, c in candidates:
            y_min, y_max = max(0, r-1), min(pruned.shape[0], r+2)
            x_min, x_max = max(0, c-1), min(pruned.shape[1], c+2)
            roi = pruned[y_min:y_max, x_min:x_max].copy()
            cr, cc = r - y_min, c - x_min
            _, before = ndi.label(roi, structure=np.ones((3,3)))
            roi[cr, cc] = 0 
            _, after = ndi.label(roi, structure=np.ones((3,3)))
            if before == after and before > 0:
                if np.sum(roi) > 1:
                    pruned[r, c] = 0
                    changed = True
        if not changed: break
    return pruned


def _analyze_skeleton_topology_2d(skeleton_binary, spacing):
    """
    Counts true biological branches/junctions.
    """
    if not np.any(skeleton_binary): return 0, 0, 0, 0.0
    
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    neighbors = ndi.convolve(skeleton_binary.astype(np.uint8), kernel, mode='constant', cval=0)
    
    skel_mask = skeleton_binary > 0
    endpoints = skel_mask & (neighbors == 1)
    junctions = skel_mask & (neighbors >= 3)
    
    n_end = np.sum(endpoints)
    n_junc = np.sum(junctions)
    
    # Calculate branches (Euler characteristic for planar graph)
    # Approx: Branches = Endpoints + Junction_Excess
    # Exact calculation is hard without traversing graph.
    # We use: Branches = (Sum(Degree) - 2) / 2 + 1 ?? No.
    
    # Heuristic for trees:
    # If 0 junctions: 1 branch (if 2 endpoints) or 0 (loop)
    if n_junc == 0:
        true_branches = 1 if n_end >= 2 else 0
    else:
        # Each junction splits 1 incoming into N outgoing. 
        # Total segments = Endpoints + Sum(Degree_J - 2)?
        # Let's count degree at junctions
        j_coords = np.argwhere(junctions)
        excess = 0
        for y, x in j_coords:
            deg = neighbors[y, x]
            excess += max(0, deg - 2)
        true_branches = excess + (n_end if n_end > 0 else 0) # Rough approx
        
        # Better heuristic: Count skan branches if possible, otherwise use this.
        # Actually, let's stick to the heuristic used in the old file or 3D one.
        # 3D one uses: n_end - 1 + n_junc. Let's use that for consistency.
        true_branches = max(0, n_end - 1 + n_junc)

    # Total Length
    try:
        skel_obj = Skeleton(skeleton_binary, spacing=spacing)
        summary = summarize(skel_obj)
        total_len = summary['branch-distance'].sum()
    except Exception:
        total_len = np.sum(skeleton_binary) * np.mean(spacing)
        
    return true_branches, n_junc, n_end, total_len


def calculate_ramification_with_skan_2d(
    segmented_array, spacing, skeleton_export_path, prune_spurs_le_um
):
    flush_print(f"  [Skel] Skeletonizing (Pruning <= {prune_spurs_le_um} um)...")
    original_shape = segmented_array.shape
    
    use_memmap = (skeleton_export_path is not None)
    if use_memmap:
        os.makedirs(os.path.dirname(skeleton_export_path), exist_ok=True)
        skel_out = np.memmap(skeleton_export_path, dtype=np.int32, mode='w+', shape=original_shape)
    else:
        skel_out = np.zeros(original_shape, dtype=np.int32)
        
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    locations = ndi.find_objects(segmented_array)
    
    stats_list = []
    detailed_dfs = []
    
    for lbl in tqdm(labels, desc="    Skeletonizing"):
        idx = lbl - 1
        if idx >= len(locations) or locations[idx] is None: continue
        
        sl = locations[idx]
        offset = np.array([s.start for s in sl])
        
        mask = (segmented_array[sl] == lbl).astype(bool)
        if not np.any(mask): continue
        
        # --- FIX: ROBUST HOLE FILLING ---
        # 1. Pad the mask so holes touching the crop edge are correctly filled
        mask = np.pad(mask, pad_width=2, mode='constant', constant_values=0)
        
        # 2. Morphological Closing (Seals 1-2 pixel 'leaks' to the background)
        # Using a disk(1) or a 3x3 square is usually enough.
        mask = binary_closing(mask, np.ones((3, 3)))
        
        # 3. Fill holes
        mask = ndi.binary_fill_holes(mask)
        
        # 4. Remove padding
        mask = mask[2:-2, 2:-2]
        
        # --- SKELETONIZATION ---
        try:
            skel = skeletonize(mask, method='lee')
        except:
            skel = skeletonize(mask)
        
        if np.any(skel):
            # Clean artifacts (laddering) before breaking cycles
            skel = _prune_internal_artifacts_2d(skel)
            # Break any surviving topological loops
            skel = _break_skeleton_cycles_2d(skel)

        # Prune small spurs
        if prune_spurs_le_um > 0 and np.any(skel):
            skel = _prune_skeleton_spurs_2d(skel, spacing, prune_spurs_le_um)
            
        # Final cleanup
        if np.any(skel):
            skel = _prune_internal_artifacts_2d(skel)
            
        # Skan / Analysis (logic remains the same)
        skan_len, skan_branches, avg_len = 0.0, 0, 0.0
        if np.any(skel):
            try:
                skel_obj = Skeleton(skel, spacing=spacing)
                summ = summarize(skel_obj, separator='-')
                if not summ.empty:
                    summ['label'] = lbl
                    for c in summ.columns:
                        if 'coord' in c:
                            axis = int(c.split('-')[-1])
                            summ[c] += offset[axis]
                    detailed_dfs.append(summ)
                    skan_len = summ['branch-distance'].sum()
                    skan_branches = len(summ)
                    avg_len = summ['branch-distance'].mean()
            except: pass
        
        true_branches, n_junc, n_end, _ = _analyze_skeleton_topology_2d(skel, spacing)
        
        if np.any(skel):
            skel_out[sl][skel > 0] = lbl

        stats_list.append({
            'label': lbl, 
            'true_num_branches': true_branches, 
            'skan_total_length_um': skan_len,
            'skan_avg_branch_length_um': avg_len, 
            'true_num_junctions': n_junc,
            'true_num_endpoints': n_end, 
            'skan_num_skeleton_pixels': np.count_nonzero(skel)
        })

    if use_memmap: skel_out.flush()
    return pd.DataFrame(stats_list), pd.concat(detailed_dfs, ignore_index=True) if detailed_dfs else pd.DataFrame(), skel_out


def export_to_fcs(metrics_df, fcs_path):
    """Exports metrics to FCS format."""
    if fcswrite is None:
        print("  [Export] 'fcswrite' not installed. Skipping.")
        return
        
    if metrics_df is None or metrics_df.empty:
        return

    try:
        flush_print(f"  [Export] Writing FCS: {os.path.basename(fcs_path)}")
        numeric_df = metrics_df.select_dtypes(include=[np.number]).copy()
        numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_df.fillna(0, inplace=True)
        
        if 'label' not in numeric_df.columns and 'label' in metrics_df.columns:
            numeric_df['label'] = metrics_df['label']
            
        fcswrite.write_fcs(filename=fcs_path, chn_names=list(numeric_df.columns), data=numeric_df.values)
    except Exception as e:
        print(f"  [Export] Error: {e}")


# =============================================================================
# 4. MAIN ENTRY POINT (2D)
# =============================================================================

def analyze_segmentation_2d(
    segmented_array: np.ndarray,
    spacing_yx: Tuple[float, float] = (1.0, 1.0),
    intensity_image: Optional[np.ndarray] = None,
    calculate_distances: bool = True,
    calculate_skeletons: bool = True,
    skeleton_export_path: Optional[str] = None,
    fcs_export_path: Optional[str] = None,
    temp_dir: Optional[str] = None,
    n_jobs: Optional[int] = None,
    return_detailed: bool = False,
    prune_spurs_le_um: float = 0.0
):
    """
    Comprehensive 2D Analysis.
    """
    flush_print("\n--- Starting Feature Calculation (2D) ---")
    
    if segmented_array.ndim != 2:
        raise ValueError("Input must be 2D")

    # 1. Morphology
    metrics_df = calculate_morphology_2d(segmented_array, spacing=spacing_yx)
    if metrics_df.empty:
        return pd.DataFrame(), {}

    detailed_outputs = {}

    # 2. Intensity
    if intensity_image is not None:
        int_df = calculate_intensity_2d(segmented_array, intensity_image)
        if not int_df.empty:
            metrics_df = pd.merge(metrics_df, int_df, on='label', how='outer')

    # 3. Distances
    if calculate_distances:
        dist_df, pts_df = shortest_distance_2d(segmented_array, spacing_yx, temp_dir, n_jobs)
        
        if not dist_df.empty:
            np.fill_diagonal(dist_df.values, np.inf)
            min_dists = dist_df.min(axis=1)
            closest_neighs = dist_df.idxmin(axis=1)
            
            dist_metrics = pd.DataFrame({
                'label': dist_df.index,
                'shortest_distance_um': min_dists.values,
                'closest_neighbor_label': closest_neighs.values
            })
            metrics_df = pd.merge(metrics_df, dist_metrics, on='label', how='left')
            
            if return_detailed:
                detailed_outputs['distance_matrix'] = dist_df
                # Filter nearest neighbor lines
                if not pts_df.empty:
                    neigh_map = dist_metrics.set_index('label')['closest_neighbor_label'].to_dict()
                    def is_nearest(r): return neigh_map.get(r['mask1']) == r['mask2']
                    nearest_pts = pts_df[pts_df.apply(is_nearest, axis=1)].reset_index(drop=True)
                    detailed_outputs['all_pairs_points'] = nearest_pts

    # 4. Skeletons
    if calculate_skeletons:
        # Fix: Redefine calculate_ramification to init int32 array correctly
        # We need to handle the skeleton array creation carefully inside the function.
        # Here we just pass the path.
        summ_skel, detail_skel, skel_arr = calculate_ramification_with_skan_2d(
            segmented_array, spacing_yx, skeleton_export_path, prune_spurs_le_um
        )
        
        if not summ_skel.empty:
            metrics_df = pd.merge(metrics_df, summ_skel, on='label', how='left')
            
        if return_detailed:
            detailed_outputs['detailed_branches'] = detail_skel
            detailed_outputs['skeleton_array'] = skel_arr

    # 5. FCS Export
    if fcs_export_path:
        export_to_fcs(metrics_df, fcs_export_path)

    flush_print("--- Analysis Complete (2D) ---")
    return metrics_df, detailed_outputs if return_detailed else {}


# --- Patch for calculate_ramification_with_skan_2d to handle int32 ---
# (Overwriting the previous definition in this file context for clarity)
def calculate_ramification_with_skan_2d(
    segmented_array, spacing, skeleton_export_path, prune_spurs_le_um
):
    flush_print(f"  [Skel] Skeletonizing (Pruning <= {prune_spurs_le_um} um)...")
    original_shape = segmented_array.shape
    
    use_memmap = (skeleton_export_path is not None)
    if use_memmap:
        os.makedirs(os.path.dirname(skeleton_export_path), exist_ok=True)
        skel_out = np.memmap(skeleton_export_path, dtype=np.int32, mode='w+', shape=original_shape)
    else:
        skel_out = np.zeros(original_shape, dtype=np.int32)
        
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    locations = ndi.find_objects(segmented_array)
    
    stats_list = []
    detailed_dfs = []
    
    for lbl in tqdm(labels, desc="    Skeletonizing"):
        idx = lbl - 1
        if idx >= len(locations) or locations[idx] is None: continue
        
        sl = locations[idx]
        offset = np.array([s.start for s in sl])
        
        crop = segmented_array[sl]
        mask = (crop == lbl)
        if not np.any(mask): continue
        
        # 1. Skeletonize using 'lee' (more consistent for microscopy)
        try:
            skel = skeletonize(mask, method='lee')
        except:
            skel = skeletonize(mask)
        
        # 2. Prune spurs
        if prune_spurs_le_um > 0:
            skel = _prune_skeleton_spurs_2d(skel, spacing, prune_spurs_le_um)
            
        # 3. Clean Artifacts (Safe connectivity)
        if np.any(skel):
            skel = _prune_internal_artifacts_2d(skel)
            
        # 4. Skan Analysis
        skan_len, skan_branches, avg_len = 0.0, 0, 0.0
        if np.any(skel):
            try:
                skel_obj = Skeleton(skel, spacing=spacing)
                summ = summarize(skel_obj, separator='-')
                if not summ.empty:
                    summ['label'] = lbl
                    for c in summ.columns:
                        if 'coord' in c:
                            axis = int(c.split('-')[-1])
                            summ[c] += offset[axis]
                    detailed_dfs.append(summ)
                    skan_len = summ['branch-distance'].sum()
                    skan_branches = len(summ)
                    avg_len = summ['branch-distance'].mean()
            except: pass
        
        # True Topology metrics
        true_branches, n_junc, n_end, _ = _analyze_skeleton_topology_2d(skel, spacing)
        
        # 5. Save Visual Labels
        if np.any(skel):
            skel_out[sl][skel > 0] = lbl

        stats_list.append({
            'label': lbl, 
            'true_num_branches': true_branches, 
            'skan_total_length_um': skan_len,
            'skan_avg_branch_length_um': avg_len, 
            'true_num_junctions': n_junc,
            'true_num_endpoints': n_end, 
            'skan_num_skeleton_pixels': np.count_nonzero(skel)
        })

    if use_memmap: skel_out.flush()
    return pd.DataFrame(stats_list), pd.concat(detailed_dfs, ignore_index=True) if detailed_dfs else pd.DataFrame(), skel_out