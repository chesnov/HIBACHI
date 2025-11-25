# --- START OF FILE utils/module_3d/calculate_features_3d.py ---
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree # type: ignore
from scipy import ndimage as ndi # type: ignore
import multiprocessing as mp
import os
import tempfile
import time
import gc
import traceback
from skimage.morphology import skeletonize # type: ignore
from skan import Skeleton, summarize # type: ignore
from tqdm.auto import tqdm # type: ignore

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    import sys
    sys.stdout.flush()

# =============================================================================
# 1. DISTANCE CALCULATION (Map-Reduce)
# =============================================================================

def _extract_surface_worker(args):
    """
    Worker: Extracts surface coordinates.
    Returns numpy array with columns: [Z, Y, X]
    """
    label_idx, label, region, offset, temp_dir = args
    
    try:
        mask = (region == label)
        if not np.any(mask):
            np.save(os.path.join(temp_dir, f"surface_{label}.npy"), np.empty((0, 3), dtype=int))
            return label_idx, 0

        eroded = ndi.binary_erosion(mask, structure=ndi.generate_binary_structure(3, 1))
        surface_mask = mask ^ eroded
        
        # np.where returns (dim0, dim1, dim2) -> (Z, Y, X)
        z_local, y_local, x_local = np.where(surface_mask)
        
        # Add offsets to get global coordinates
        surface_points_global = np.column_stack((
            z_local + offset[0], 
            y_local + offset[1], 
            x_local + offset[2]
        ))
        
        np.save(os.path.join(temp_dir, f"surface_{label}.npy"), surface_points_global)
        return label_idx, len(surface_points_global)

    except Exception as e:
        np.save(os.path.join(temp_dir, f"surface_{label}.npy"), np.empty((0, 3), dtype=int))
        return label_idx, 0

def _calculate_pair_distance_worker(args):
    """
    Worker: Calculates shortest distance between two surfaces.
    """
    i, j, label1, label2, temp_dir, spacing_arr = args
    
    try:
        # p1, p2 columns are [Z, Y, X]
        p1 = np.load(os.path.join(temp_dir, f"surface_{label1}.npy"))
        p2 = np.load(os.path.join(temp_dir, f"surface_{label2}.npy"))
        
        if p1.shape[0] == 0 or p2.shape[0] == 0:
            return i, j, np.inf, np.full(6, np.nan)
            
        # spacing_arr is [Z_um, Y_um, X_um]
        p1_um = p1 * spacing_arr
        p2_um = p2 * spacing_arr
        
        tree = cKDTree(p2_um)
        dists, indices = tree.query(p1_um, k=1)
        
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        
        # Closest points in voxel coordinates
        point_on_1 = p1[min_idx]       # [z1, y1, x1]
        point_on_2 = p2[indices[min_idx]] # [z2, y2, x2]
        
        # Result: [z1, y1, x1, z2, y2, x2]
        result_pts = np.concatenate([point_on_1, point_on_2])
        
        return i, j, min_dist, result_pts
        
    except Exception:
        return i, j, np.inf, np.full(6, np.nan)

def shortest_distance(segmented_array, spacing=(1.0, 1.0, 1.0), temp_dir=None, n_jobs=None):
    """
    Coordinator for Pairwise Distance Calculation.
    """
    start_time = time.time()
    if n_jobs is None: n_jobs = max(1, mp.cpu_count() - 1)
    spacing_arr = np.array(spacing) # Expect [Z, Y, X]

    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    n_labels = len(labels)
    
    if n_labels <= 1:
        return pd.DataFrame(), pd.DataFrame()

    temp_dir_managed = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="dist_calc_")
        temp_dir_managed = True
    else:
        os.makedirs(temp_dir, exist_ok=True)

    try:
        # 1. Extract Surfaces
        flush_print(f"  [Dist] Extracting surfaces for {n_labels} objects...")
        locations = ndi.find_objects(segmented_array)
        
        extract_tasks = []
        labels_processed = []
        
        for i, lbl in enumerate(labels):
            idx = lbl - 1
            if idx < len(locations) and locations[idx] is not None:
                sl = locations[idx]
                # Extract crop with 1px padding
                padded_sl = []
                offset = []
                for dim_s, dim_len in zip(sl, segmented_array.shape):
                    start = max(0, dim_s.start - 1)
                    stop = min(dim_len, dim_s.stop + 1)
                    padded_sl.append(slice(start, stop))
                    offset.append(start)
                
                region = segmented_array[tuple(padded_sl)]
                # Pass offset as [z_start, y_start, x_start]
                extract_tasks.append((i, lbl, region, np.array(offset), temp_dir))
                labels_processed.append(lbl)
        
        with mp.Pool(n_jobs) as pool:
            list(tqdm(pool.imap_unordered(_extract_surface_worker, extract_tasks), 
                      total=len(extract_tasks), desc="    Surface Extraction"))
            
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
            results = list(tqdm(pool.imap_unordered(_calculate_pair_distance_worker, pairs), 
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
                # pts is [z1, y1, x1, z2, y2, x2]
                
                # Store connection L1 -> L2
                points_list.append({
                    'mask1': l1, 'mask2': l2,
                    'mask1_z': pts[0], 'mask1_y': pts[1], 'mask1_x': pts[2],
                    'mask2_z': pts[3], 'mask2_y': pts[4], 'mask2_x': pts[5]
                })
                # Store connection L2 -> L1 (symmetric)
                points_list.append({
                    'mask1': l2, 'mask2': l1,
                    'mask1_z': pts[3], 'mask1_y': pts[4], 'mask1_x': pts[5],
                    'mask2_z': pts[0], 'mask2_y': pts[1], 'mask2_x': pts[2]
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

def calculate_volume(segmented_array, spacing=(1.0, 1.0, 1.0)):
    """Calculates Volume, Surface Area, Sphericity."""
    flush_print("  [Metrics] Calculating Volume and Shape...")
    voxel_vol = np.prod(spacing)
    az = spacing[1] * spacing[2] 
    ay = spacing[0] * spacing[2] 
    ax = spacing[0] * spacing[1] 
    
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    if len(labels) == 0: return pd.DataFrame()
    
    locations = ndi.find_objects(segmented_array)
    results = []
    
    for lbl in tqdm(labels, desc="    Volume/Shape"):
        idx = lbl - 1
        if idx >= len(locations) or locations[idx] is None: continue
        
        sl = locations[idx]
        crop = segmented_array[sl]
        mask = (crop == lbl)
        
        n_vox = np.count_nonzero(mask)
        vol_um = n_vox * voxel_vol
        
        # Surface Area
        sa = 0.0
        try:
            diff_z = np.diff(mask.astype(np.int8), axis=0)
            sa += np.count_nonzero(diff_z) * ax
            sa += (np.count_nonzero(mask[0,:,:]) + np.count_nonzero(mask[-1,:,:])) * ax
            
            diff_y = np.diff(mask.astype(np.int8), axis=1)
            sa += np.count_nonzero(diff_y) * ay
            sa += (np.count_nonzero(mask[:,0,:]) + np.count_nonzero(mask[:,-1,:])) * ay
            
            diff_x = np.diff(mask.astype(np.int8), axis=2)
            sa += np.count_nonzero(diff_x) * az
            sa += (np.count_nonzero(mask[:,:,0]) + np.count_nonzero(mask[:,:,-1])) * az
        except Exception:
            sa = np.nan
            
        sphericity = np.nan
        if sa > 1e-6:
            sphericity = (np.pi**(1/3) * (6 * vol_um)**(2/3)) / sa
            
        results.append({
            'label': lbl, 'volume_um3': vol_um, 'surface_area_um2': sa,
            'sphericity': sphericity, 'voxel_count': n_vox
        })
    return pd.DataFrame(results)

def calculate_depth(segmented_array, spacing=(1.0, 1.0, 1.0)):
    flush_print("  [Metrics] Calculating Depths...")
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    results = []
    for lbl in tqdm(labels, desc="    Depth"):
        loc = ndi.find_objects(segmented_array == lbl)
        if loc and loc[0]:
            z_local, _, _ = np.where(segmented_array[loc[0]] == lbl)
            depth = np.median(z_local + loc[0][0].start) * spacing[0]
            results.append({'label': lbl, 'depth_um': depth})
        else:
            results.append({'label': lbl, 'depth_um': np.nan})
    return pd.DataFrame(results)

# =============================================================================
# 3. SKELETONIZATION
# =============================================================================

def _trace_spur_length_and_path_3d(skeleton, start_z, start_y, start_x, spacing):
    """Traces spur. Order is (z, y, x)."""
    if skeleton[start_z, start_y, start_x] == 0: return 0.0, []
    visited = set()
    current = (start_z, start_y, start_x)
    path = [current]
    total_len = 0.0
    
    offsets = [(dz, dy, dx) for dz in (-1,0,1) for dy in (-1,0,1) for dx in (-1,0,1) if not (dz==0 and dy==0 and dx==0)]
    
    while True:
        visited.add(current)
        cz, cy, cx = current
        neighbors = []
        for dz, dy, dx in offsets:
            nz, ny, nx = cz+dz, cy+dy, cx+dx
            if (0<=nz<skeleton.shape[0] and 0<=ny<skeleton.shape[1] and 0<=nx<skeleton.shape[2]):
                if skeleton[nz, ny, nx] and (nz, ny, nx) not in visited:
                    neighbors.append((nz, ny, nx))
        
        if len(neighbors) != 1: break
        next_node = neighbors[0]
        dist = np.sqrt(((next_node[0]-cz)*spacing[0])**2 + ((next_node[1]-cy)*spacing[1])**2 + ((next_node[2]-cx)*spacing[2])**2)
        total_len += dist
        current = next_node
        path.append(current)
    return total_len, path

def _prune_skeleton_spurs_3d(skeleton_binary, spacing, max_spur_length_um):
    if max_spur_length_um <= 0: return skeleton_binary
    pruned = skeleton_binary.copy().astype(np.uint8)
    kernel = ndi.generate_binary_structure(3, 3).astype(np.uint8)
    kernel[1,1,1] = 0
    
    changed = True
    iters = 0
    while changed and iters < 20:
        changed = False
        iters += 1
        neighbors = ndi.convolve(pruned, kernel, mode='constant', cval=0)
        endpoints = (pruned == 1) & (neighbors == 1)
        coords = np.argwhere(endpoints)
        for z, y, x in coords:
            if pruned[z, y, x] == 0: continue
            length, path = _trace_spur_length_and_path_3d(pruned, z, y, x, spacing)
            if length <= max_spur_length_um:
                for pz, py, px in path: pruned[pz, py, px] = 0
                changed = True
    return pruned

def _analyze_skeleton_topology(skeleton_binary, spacing):
    if not np.any(skeleton_binary): return 0, 0, 0, 0.0
    kernel = ndi.generate_binary_structure(3, 3).astype(np.uint8)
    kernel[1,1,1] = 0
    neighbors = ndi.convolve(skeleton_binary.astype(np.uint8), kernel, mode='constant', cval=0)
    skel_mask = skeleton_binary > 0
    endpoints = skel_mask & (neighbors == 1)
    junctions = skel_mask & (neighbors >= 3)
    
    n_end = np.count_nonzero(endpoints)
    n_junc = np.count_nonzero(junctions)
    total_len = 0.0
    try:
        graph = Skeleton(skeleton_binary, spacing=spacing)
        if graph.paths.indices.shape[0] > 0:
            summary = summarize(graph, separator='-')
            total_len = summary['branch-distance'].sum()
    except Exception:
        total_len = np.count_nonzero(skeleton_binary) * np.mean(spacing)
    n_branch = max(0, n_end - 1 + (n_junc if n_junc > 0 else 0))
    return n_branch, n_junc, n_end, total_len

def calculate_ramification_with_skan(segmented_array, spacing, skeleton_export_path, prune_spurs_le_um):
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
        
        skel = skeletonize(mask)
        if prune_spurs_le_um > 0:
            skel = _prune_skeleton_spurs_3d(skel, spacing, prune_spurs_le_um)
            
        skan_branches, skan_len, avg_len = 0, 0.0, 0.0
        try:
            if np.any(skel):
                skel_obj = Skeleton(skel, spacing=spacing)
                summ = summarize(skel_obj, separator='-')
                if not summ.empty:
                    summ['label'] = lbl
                    # Fix: Coordinates are usually returned as 'coord-src-0' (Z), 'coord-src-1' (Y), 'coord-src-2' (X)
                    # depending on skan version, but usually axis order matches input.
                    # We offset them to global.
                    for c in summ.columns:
                        if 'coord' in c:
                            axis = int(c.split('-')[-1])
                            summ[c] += offset[axis]
                    detailed_dfs.append(summ)
                    skan_branches = len(summ)
                    skan_len = summ['branch-distance'].sum()
                    avg_len = summ['branch-distance'].mean()
        except Exception: pass
        
        _, n_junc, n_end, _ = _analyze_skeleton_topology(skel, spacing)
        
        if np.any(skel):
            out_view = skel_out[sl]
            out_view[skel > 0] = lbl
            skel_out[sl] = out_view
            
        stats_list.append({
            'label': lbl, 'true_num_branches': skan_branches, 'skan_total_length_um': skan_len,
            'skan_avg_branch_length_um': avg_len, 'true_num_junctions': n_junc,
            'true_num_endpoints': n_end, 'skan_num_skeleton_voxels': np.count_nonzero(skel)
        })
        
    if use_memmap: skel_out.flush()
    return pd.DataFrame(stats_list), pd.concat(detailed_dfs, ignore_index=True) if detailed_dfs else pd.DataFrame(), skel_out

# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================

def analyze_segmentation(segmented_array, spacing=(1.0, 1.0, 1.0), 
                         calculate_distances=True, calculate_skeletons=True,
                         skeleton_export_path=None, temp_dir=None, n_jobs=None,
                         return_detailed=False, prune_spurs_le_um=0.0):
    flush_print("\n--- Starting Feature Calculation ---")
    
    # 1. Volume & Depth
    vol_df = calculate_volume(segmented_array, spacing)
    depth_df = calculate_depth(segmented_array, spacing)
    
    if vol_df.empty and depth_df.empty: return pd.DataFrame(), {}
    metrics_df = pd.merge(vol_df, depth_df, on='label', how='outer')
    detailed_outputs = {}
    
    # 2. Distances
    if calculate_distances:
        dist_df, pts_df = shortest_distance(segmented_array, spacing, temp_dir, n_jobs)
        
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
            
            if return_detailed and not pts_df.empty:
                detailed_outputs['distance_matrix'] = dist_df
                
                # --- FILTER POINTS FOR VISUALIZATION ---
                # We only want the lines connecting a cell to its CLOSEST neighbor.
                # pts_df has 'mask1', 'mask2'.
                # We want rows where mask2 == closest_neighbor_label[mask1]
                
                # Create a map of label -> closest_neigh
                neighbor_map = dist_metrics.set_index('label')['closest_neighbor_label'].to_dict()
                
                # Filter function
                def is_nearest(row):
                    m1 = row['mask1']
                    m2 = row['mask2']
                    # Check if m2 is the nearest neighbor of m1
                    return neighbor_map.get(m1) == m2

                # Apply filter
                nearest_pts_df = pts_df[pts_df.apply(is_nearest, axis=1)].reset_index(drop=True)
                detailed_outputs['all_pairs_points'] = nearest_pts_df
                print(f"  [Dist] Filtered visualization: {len(nearest_pts_df)} lines (nearest neighbors only).")

    # 3. Skeletons
    if calculate_skeletons:
        summ_skel, detail_skel, skel_arr = calculate_ramification_with_skan(
            segmented_array, spacing, skeleton_export_path, prune_spurs_le_um
        )
        if not summ_skel.empty:
            metrics_df = pd.merge(metrics_df, summ_skel, on='label', how='left')
        if return_detailed:
            detailed_outputs['detailed_branches'] = detail_skel
            detailed_outputs['skeleton_array'] = skel_arr

    flush_print("--- Analysis Complete ---")
    return metrics_df, detailed_outputs if return_detailed else {}
# --- END OF FILE utils/module_3d/calculate_features_3d.py ---