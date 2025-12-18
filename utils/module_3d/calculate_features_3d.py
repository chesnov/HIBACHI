import os
import gc
import sys
import time
import shutil
import tempfile
import traceback
import multiprocessing as mp
from typing import Tuple, List, Dict, Optional, Any, Union

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from tqdm.auto import tqdm

# Optional Import
try:
    import fcswrite  # type: ignore
except ImportError:
    fcswrite = None
    print("Warning: 'fcswrite' not installed. FCS export will be disabled.")


def flush_print(*args: Any, **kwargs: Any) -> None:
    """Wrapper for print that forces immediate flushing of stdout."""
    print(*args, **kwargs)
    sys.stdout.flush()


# =============================================================================
# 1. DISTANCE CALCULATION (Map-Reduce)
# =============================================================================

def _extract_surface_worker(args: Tuple) -> Tuple[int, int]:
    """
    Worker: Extracts surface coordinates from a label region.
    Saves [Z, Y, X] coordinates to a temporary .npy file.

    Args:
        args: Tuple containing (label_idx, label, region, offset, temp_dir).

    Returns:
        Tuple: (label_idx, number_of_points).
    """
    label_idx, label, region, offset, temp_dir = args

    try:
        mask = (region == label)
        if not np.any(mask):
            np.save(
                os.path.join(temp_dir, f"surface_{label}.npy"),
                np.empty((0, 3), dtype=int)
            )
            return label_idx, 0

        # Create 1-pixel border to find surface
        struct = ndi.generate_binary_structure(3, 1)
        eroded = ndi.binary_erosion(mask, structure=struct)
        surface_mask = mask ^ eroded

        # np.where returns (dim0, dim1, dim2) -> (Z, Y, X)
        z_local, y_local, x_local = np.where(surface_mask)

        # Add offsets to get global coordinates
        surface_points_global = np.column_stack((
            z_local + offset[0],
            y_local + offset[1],
            x_local + offset[2]
        ))

        np.save(
            os.path.join(temp_dir, f"surface_{label}.npy"),
            surface_points_global
        )
        return label_idx, len(surface_points_global)

    except Exception:
        np.save(
            os.path.join(temp_dir, f"surface_{label}.npy"),
            np.empty((0, 3), dtype=int)
        )
        return label_idx, 0


def _calculate_pair_distance_worker(args: Tuple) -> Tuple[int, int, float, np.ndarray]:
    """
    Worker: Calculates shortest Euclidean distance between two point clouds.

    Args:
        args: Tuple containing (i, j, label1, label2, temp_dir, spacing_arr).

    Returns:
        Tuple: (i, j, min_dist, result_pts[z1, y1, x1, z2, y2, x2]).
    """
    i, j, label1, label2, temp_dir, spacing_arr = args

    try:
        # Load surface points [Z, Y, X]
        p1 = np.load(os.path.join(temp_dir, f"surface_{label1}.npy"))
        p2 = np.load(os.path.join(temp_dir, f"surface_{label2}.npy"))

        if p1.shape[0] == 0 or p2.shape[0] == 0:
            return i, j, np.inf, np.full(6, np.nan)

        # Apply spacing to get physical coordinates [Z_um, Y_um, X_um]
        p1_um = p1 * spacing_arr
        p2_um = p2 * spacing_arr

        # Use KDTree for efficient nearest neighbor search
        tree = cKDTree(p2_um)
        dists, indices = tree.query(p1_um, k=1)

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        # Get the specific points in Voxel coordinates
        point_on_1 = p1[min_idx]           # [z1, y1, x1]
        point_on_2 = p2[indices[min_idx]]  # [z2, y2, x2]

        # Combine into result vector
        result_pts = np.concatenate([point_on_1, point_on_2])

        return i, j, min_dist, result_pts

    except Exception:
        return i, j, np.inf, np.full(6, np.nan)


def shortest_distance(
    segmented_array: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    temp_dir: Optional[str] = None,
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the shortest distance between all pairs of objects in 3D.
    Uses Map-Reduce with temporary files to handle memory for large surfaces.

    Args:
        segmented_array: 3D labeled array.
        spacing: Voxel spacing (Z, Y, X).
        temp_dir: Directory for temporary surface files.
        n_jobs: Number of parallel workers.

    Returns:
        Tuple: (Distance Matrix DataFrame, Points Info DataFrame).
    """
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    spacing_arr = np.array(spacing)

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
        # 1. Map: Extract Surfaces
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
                extract_tasks.append(
                    (i, lbl, region, np.array(offset), temp_dir)
                )
                labels_processed.append(lbl)

        with mp.Pool(n_jobs) as pool:
            list(tqdm(pool.imap_unordered(_extract_surface_worker, extract_tasks),
                      total=len(extract_tasks), desc="    Surface Extraction"))

        # 2. Reduce: Calculate Pairwise Distances
        flush_print(f"  [Dist] Calculating pairwise distances...")
        pairs = []
        n_proc = len(labels_processed)
        for i in range(n_proc):
            for j in range(i + 1, n_proc):
                pairs.append((
                    i, j,
                    labels_processed[i], labels_processed[j],
                    temp_dir, spacing_arr
                ))

        if not pairs:
            return pd.DataFrame(), pd.DataFrame()

        with mp.Pool(n_jobs) as pool:
            results = list(tqdm(
                pool.imap_unordered(_calculate_pair_distance_worker, pairs),
                total=len(pairs), desc="    Distance Matrix"
            ))

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

        dist_df = pd.DataFrame(
            dist_mat, index=labels_processed, columns=labels_processed
        )
        points_df = pd.DataFrame(points_list)

        return dist_df, points_df

    finally:
        if temp_dir_managed and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# 2. BASIC METRICS
# =============================================================================

def calculate_volume(
    segmented_array: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> pd.DataFrame:
    """Calculates Volume, Surface Area, and Sphericity for each label."""
    flush_print("  [Metrics] Calculating Volume and Shape...")
    voxel_vol = np.prod(spacing)
    az = spacing[1] * spacing[2]
    ay = spacing[0] * spacing[2]
    ax = spacing[0] * spacing[1]

    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return pd.DataFrame()

    locations = ndi.find_objects(segmented_array)
    results = []

    for lbl in tqdm(labels, desc="    Volume/Shape"):
        idx = lbl - 1
        if idx >= len(locations) or locations[idx] is None:
            continue

        sl = locations[idx]
        crop = segmented_array[sl]
        mask = (crop == lbl)

        n_vox = np.count_nonzero(mask)
        vol_um = n_vox * voxel_vol

        # Approximate Surface Area via Crofton formula concept on grid
        sa = 0.0
        try:
            diff_z = np.diff(mask.astype(np.int8), axis=0)
            sa += np.count_nonzero(diff_z) * ax
            sa += (np.count_nonzero(mask[0, :, :]) +
                   np.count_nonzero(mask[-1, :, :])) * ax

            diff_y = np.diff(mask.astype(np.int8), axis=1)
            sa += np.count_nonzero(diff_y) * ay
            sa += (np.count_nonzero(mask[:, 0, :]) +
                   np.count_nonzero(mask[:, -1, :])) * ay

            diff_x = np.diff(mask.astype(np.int8), axis=2)
            sa += np.count_nonzero(diff_x) * az
            sa += (np.count_nonzero(mask[:, :, 0]) +
                   np.count_nonzero(mask[:, :, -1])) * az
        except Exception:
            sa = np.nan

        sphericity = np.nan
        if sa > 1e-6:
            # Sphericity = (pi^(1/3) * (6 * Volume)^(2/3)) / SurfaceArea
            sphericity = (np.pi**(1/3) * (6 * vol_um)**(2/3)) / sa

        results.append({
            'label': lbl,
            'volume_um3': vol_um,
            'surface_area_um2': sa,
            'sphericity': sphericity,
            'voxel_count': n_vox
        })
    return pd.DataFrame(results)


def calculate_depth(
    segmented_array: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> pd.DataFrame:
    """Calculates the median depth (Z-coordinate) for each object."""
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


def calculate_intensity(
    segmented_array: np.ndarray,
    intensity_image: np.ndarray
) -> pd.DataFrame:
    """Calculates fluorescence intensity statistics for each object."""
    flush_print("  [Metrics] Calculating Intensity Stats...")

    if segmented_array.shape != intensity_image.shape:
        print(f"    Warning: Shape mismatch. Skipping intensity.")
        return pd.DataFrame()

    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return pd.DataFrame()

    locations = ndi.find_objects(segmented_array)
    results = []

    for lbl in tqdm(labels, desc="    Intensity"):
        idx = lbl - 1
        if idx >= len(locations) or locations[idx] is None:
            continue

        sl = locations[idx]
        mask_crop = (segmented_array[sl] == lbl)
        int_crop = intensity_image[sl]

        values = int_crop[mask_crop]

        if values.size > 0:
            results.append({
                'label': lbl,
                'mean_intensity': np.mean(values),
                'median_intensity': np.median(values),
                'std_intensity': np.std(values),
                'integrated_density': np.sum(values),
                'max_intensity': np.max(values)
            })
        else:
            results.append({
                'label': lbl, 'mean_intensity': 0, 'median_intensity': 0,
                'std_intensity': 0, 'integrated_density': 0, 'max_intensity': 0
            })

    return pd.DataFrame(results)


# =============================================================================
# 3. SKELETONIZATION
# =============================================================================

def _trace_spur_length_and_path_3d(
    skeleton: np.ndarray,
    start_z: int,
    start_y: int,
    start_x: int,
    spacing: Tuple[float, float, float]
) -> Tuple[float, List[Tuple[int, int, int]]]:
    """Traces a skeleton branch from an endpoint to a junction or end."""
    if skeleton[start_z, start_y, start_x] == 0:
        return 0.0, []

    visited = set()
    current = (start_z, start_y, start_x)
    path = [current]
    total_len = 0.0

    # 26-connectivity offsets excluding (0,0,0)
    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    while True:
        visited.add(current)
        cz, cy, cx = current
        neighbors = []
        for dz, dy, dx in offsets:
            nz, ny, nx = cz + dz, cy + dy, cx + dx
            if (0 <= nz < skeleton.shape[0] and
                0 <= ny < skeleton.shape[1] and
                0 <= nx < skeleton.shape[2]):
                if skeleton[nz, ny, nx] and (nz, ny, nx) not in visited:
                    neighbors.append((nz, ny, nx))

        if len(neighbors) != 1:
            break  # Junction or End

        next_node = neighbors[0]
        dist = np.sqrt(
            ((next_node[0] - cz) * spacing[0])**2 +
            ((next_node[1] - cy) * spacing[1])**2 +
            ((next_node[2] - cx) * spacing[2])**2
        )
        total_len += dist
        current = next_node
        path.append(current)

    return total_len, path


def _prune_skeleton_spurs_3d(
    skeleton_binary: np.ndarray,
    spacing: Tuple[float, float, float],
    max_spur_length_um: float
) -> np.ndarray:
    """Iteratively removes terminal branches shorter than threshold."""
    if max_spur_length_um <= 0:
        return skeleton_binary

    pruned = skeleton_binary.copy().astype(np.uint8)
    kernel = ndi.generate_binary_structure(3, 3).astype(np.uint8)
    kernel[1, 1, 1] = 0

    changed = True
    iters = 0
    while changed and iters < 20:
        changed = False
        iters += 1
        neighbors = ndi.convolve(pruned, kernel, mode='constant', cval=0)
        # Endpoints have exactly 1 neighbor
        endpoints = (pruned == 1) & (neighbors == 1)
        coords = np.argwhere(endpoints)

        for z, y, x in coords:
            if pruned[z, y, x] == 0:
                continue
            length, path = _trace_spur_length_and_path_3d(pruned, z, y, x, spacing)
            if length <= max_spur_length_um:
                for pz, py, px in path:
                    pruned[pz, py, px] = 0
                changed = True
    return pruned


def _analyze_skeleton_topology(
    skeleton_binary: np.ndarray,
    spacing: Tuple[float, float, float]
) -> Tuple[int, int, int, float]:
    """Returns (n_branches, n_junctions, n_endpoints, total_length_um)."""
    if not np.any(skeleton_binary):
        return 0, 0, 0, 0.0

    kernel = ndi.generate_binary_structure(3, 3).astype(np.uint8)
    kernel[1, 1, 1] = 0
    neighbors = ndi.convolve(
        skeleton_binary.astype(np.uint8), kernel, mode='constant', cval=0
    )
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

    # Topological estimation of branches
    n_branch = max(0, n_end - 1 + (n_junc if n_junc > 0 else 0))
    return n_branch, n_junc, n_end, total_len


def calculate_ramification_with_skan(
    segmented_array: np.ndarray,
    spacing: Tuple[float, float, float],
    skeleton_export_path: Optional[str],
    prune_spurs_le_um: float
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Performs skeletonization and topological analysis on each cell.
    """
    flush_print(f"  [Skel] Skeletonizing (Pruning <= {prune_spurs_le_um} um)...")
    original_shape = segmented_array.shape
    use_memmap = (skeleton_export_path is not None)

    if use_memmap and skeleton_export_path:
        os.makedirs(os.path.dirname(skeleton_export_path), exist_ok=True)
        skel_out = np.memmap(
            skeleton_export_path, dtype=np.int32, mode='w+', shape=original_shape
        )
    else:
        skel_out = np.zeros(original_shape, dtype=np.int32)

    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    locations = ndi.find_objects(segmented_array)
    stats_list = []
    detailed_dfs = []

    for lbl in tqdm(labels, desc="    Skeletonizing"):
        idx = lbl - 1
        if idx >= len(locations) or locations[idx] is None:
            continue
        sl = locations[idx]
        offset = np.array([s.start for s in sl])
        crop = segmented_array[sl]
        mask = (crop == lbl)
        if not np.any(mask):
            continue

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
                    # Correct local coordinates to global
                    for c in summ.columns:
                        if 'coord' in c:
                            axis = int(c.split('-')[-1])
                            summ[c] += offset[axis]
                    detailed_dfs.append(summ)
                    skan_branches = len(summ)
                    skan_len = summ['branch-distance'].sum()
                    avg_len = summ['branch-distance'].mean()
        except Exception:
            pass

        _, n_junc, n_end, _ = _analyze_skeleton_topology(skel, spacing)

        if np.any(skel):
            out_view = skel_out[sl]
            out_view[skel > 0] = lbl
            skel_out[sl] = out_view

        stats_list.append({
            'label': lbl,
            'true_num_branches': skan_branches,
            'skan_total_length_um': skan_len,
            'skan_avg_branch_length_um': avg_len,
            'true_num_junctions': n_junc,
            'true_num_endpoints': n_end,
            'skan_num_skeleton_voxels': np.count_nonzero(skel)
        })

    if use_memmap:
        skel_out.flush()

    return (
        pd.DataFrame(stats_list),
        pd.concat(detailed_dfs, ignore_index=True) if detailed_dfs else pd.DataFrame(),
        skel_out
    )


def export_to_fcs(metrics_df: pd.DataFrame, fcs_path: str) -> None:
    """
    Exports metrics DataFrame to FCS file format.
    Requires 'fcswrite' library.
    """
    if fcswrite is None:
        print("  [Export] 'fcswrite' not installed. Skipping FCS export.")
        return

    if metrics_df is None or metrics_df.empty:
        print("  [Export] No metrics to export.")
        return

    try:
        flush_print(f"  [Export] Writing FCS file: {os.path.basename(fcs_path)}")

        # 1. Filter numeric columns
        numeric_df = metrics_df.select_dtypes(include=[np.number]).copy()

        # 2. Handle NaN/Inf
        numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_df.fillna(0, inplace=True)

        # 3. Ensure 'label' is present
        if 'label' not in numeric_df.columns and 'label' in metrics_df.columns:
            numeric_df['label'] = metrics_df['label']

        # 4. Write
        columns = list(numeric_df.columns)
        data = numeric_df.values
        fcswrite.write_fcs(filename=fcs_path, chn_names=columns, data=data)

    except Exception as e:
        print(f"  [Export] Error writing FCS: {e}")
        traceback.print_exc()


# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================

def analyze_segmentation(
    segmented_array: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
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
    Performs comprehensive feature analysis on a 3D segmentation.

    Metrics calculated:
    1. Volume, Surface Area, Sphericity, Depth.
    2. Fluorescence Intensity (Mean, Max, Integrated, etc.).
    3. Pairwise shortest distances (Nearest Neighbors).
    4. Skeletonization (Branch length, junctions, endpoints).

    Args:
        segmented_array: 3D labeled array.
        intensity_image: 3D intensity array (optional).
        spacing: Voxel dimensions.
        calculate_distances: Whether to compute N-N distances.
        calculate_skeletons: Whether to compute skeleton stats.
        skeleton_export_path: Path to save the labeled skeleton.
        fcs_export_path: Path to export metrics as FCS.
        temp_dir: Directory for temporary files.
        n_jobs: Number of parallel workers.
        return_detailed: If True, returns detailed skan/points data.
        prune_spurs_le_um: Length threshold for skeleton pruning.

    Returns:
        Tuple: (Metrics DataFrame, Dictionary of Detailed Outputs)
    """
    flush_print("\n--- Starting Feature Calculation ---")

    # 1. Volume & Depth
    vol_df = calculate_volume(segmented_array, spacing)
    depth_df = calculate_depth(segmented_array, spacing)

    if vol_df.empty and depth_df.empty:
        return pd.DataFrame(), {}

    metrics_df = pd.merge(vol_df, depth_df, on='label', how='outer')
    detailed_outputs = {}

    # 2. Intensity
    if intensity_image is not None:
        int_df = calculate_intensity(segmented_array, intensity_image)
        if not int_df.empty:
            metrics_df = pd.merge(metrics_df, int_df, on='label', how='outer')

    # 3. Distances
    if calculate_distances:
        dist_df, pts_df = shortest_distance(
            segmented_array, spacing, temp_dir, n_jobs
        )

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

                # Filter for Nearest Neighbors only
                neighbor_map = dist_metrics.set_index('label')['closest_neighbor_label'].to_dict()

                def is_nearest(row):
                    return neighbor_map.get(row['mask1']) == row['mask2']

                nearest_pts_df = pts_df[pts_df.apply(is_nearest, axis=1)].reset_index(drop=True)
                detailed_outputs['all_pairs_points'] = nearest_pts_df
                print(f"  [Dist] Filtered visualization: {len(nearest_pts_df)} lines.")

    # 4. Skeletons
    if calculate_skeletons:
        summ_skel, detail_skel, skel_arr = calculate_ramification_with_skan(
            segmented_array, spacing, skeleton_export_path, prune_spurs_le_um
        )
        if not summ_skel.empty:
            metrics_df = pd.merge(metrics_df, summ_skel, on='label', how='left')
        if return_detailed:
            detailed_outputs['detailed_branches'] = detail_skel
            detailed_outputs['skeleton_array'] = skel_arr

    # 5. FCS Export
    if fcs_export_path:
        export_to_fcs(metrics_df, fcs_export_path)

    flush_print("--- Analysis Complete ---")
    return metrics_df, detailed_outputs if return_detailed else {}