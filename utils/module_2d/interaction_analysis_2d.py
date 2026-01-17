"""
2D Interaction Analysis Module
==============================

This module quantifies spatial relationships between two 2D segmentation layers.
It is logically identical to the 3D version, allowing for direct statistical 
comparison of proximity and overlap metrics across dimensions.
"""

import os
import sys
import gc
from typing import Tuple, Optional, Dict, List, Any, Union

import numpy as np
import pandas as pd
import dask.array as da
import dask_image.ndmeasure
from dask.diagnostics import ProgressBar
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from scipy.spatial import cKDTree
from tqdm import tqdm
import multiprocessing as mp

def _extract_contours_2d(mask_memmap, labels, shape):
    """Helper to extract boundary coordinates for a set of labels."""
    contours = []
    locs = ndimage.find_objects(mask_memmap)
    struct = generate_binary_structure(2, 1)
    for lbl in tqdm(labels, desc="    Extracting Boundaries", leave=False):
        sl = locs[lbl-1]
        if sl is None: continue
        mask = (mask_memmap[sl] == lbl)
        eroded = ndimage.binary_erosion(mask, structure=struct)
        y, x = np.where(mask ^ eroded)
        if len(y) > 0:
            contours.append(np.column_stack((y + sl[0].start, x + sl[1].start)))
        else: contours.append(np.array([]))
    return contours

def _inter_channel_dist_worker_2d(args):
    """Worker: Calculates distances from one primary object to all reference objects."""
    prim_idx, prim_contour, ref_contours, spacing_arr = args
    row = np.full(len(ref_contours), np.inf, dtype=np.float32)
    if prim_contour.size == 0: return prim_idx, row
    
    # Build tree for the primary cell
    tree = cKDTree(prim_contour * spacing_arr)
    for j, r_cnt in enumerate(ref_contours):
        if r_cnt.size > 0:
            dists, _ = tree.query(r_cnt * spacing_arr, k=1)
            row[j] = np.min(dists)
    return prim_idx, row

def calculate_pairwise_distances_2d(primary_memmap, reference_memmap, spacing_yx):
    """Calculates all pairwise distances and returns a long-format DataFrame."""
    p_labels = np.unique(primary_memmap[primary_memmap > 0])
    r_labels = np.unique(reference_memmap[reference_memmap > 0])
    spacing_arr = np.array(spacing_yx)
    
    
    # 1. Extract Boundaries
    p_contours = _extract_contours_2d(primary_memmap, p_labels, primary_memmap.shape)
    r_contours = _extract_contours_2d(reference_memmap, r_labels, reference_memmap.shape)
    
    # 2. Parallel Execution
    n_jobs = max(1, mp.cpu_count() - 1)
    tasks = [(i, p_contours[i], r_contours, spacing_arr) for i in range(len(p_labels))]
    
    all_rows = []
    with mp.Pool(n_jobs) as pool:
        for prim_idx, distances in tqdm(pool.imap_unordered(_inter_channel_dist_worker_2d, tasks), 
                                       total=len(tasks), desc="    Pairwise Distance"):
            # 'distances' is an array of M distances for Primary Object 'prim_idx'
            p_label = p_labels[prim_idx]
            
            # Create a small temporary dataframe for this specific object
            df_chunk = pd.DataFrame({
                'channel_a_label': p_label,
                'channel_b_label': r_labels,
                'distance_um': distances
            })
            
            all_rows.append(df_chunk)
            
    # Combine all chunks into one large master table
    if not all_rows:
        return pd.DataFrame()
        
    final_df = pd.concat(all_rows, ignore_index=True)
    
    # Explicit Cleanup
    del p_contours, r_contours, all_rows
    gc.collect()
    
    return final_df

def flush_print(*args: Any, **kwargs: Any) -> None:
    """Standardized wrapper for immediate log flushing."""
    print(*args, **kwargs)
    sys.stdout.flush()


def _safe_load_memmap(
    path: str,
    shape: Tuple[int, int],
    dtype: type = np.int32,
    mode: str = 'r'
) -> Optional[np.memmap]:
    """Safely loads a 2D numpy memmap file."""
    if not os.path.exists(path):
        return None
    try:
        return np.memmap(path, dtype=dtype, mode=mode, shape=shape)
    except Exception as e:
        print(f"Error loading memmap {path}: {e}")
        return None


def calculate_interaction_metrics_2d(
    primary_mask_path: str,
    reference_mask_path: str,
    output_dir: str,
    shape: Tuple[int, int],
    spacing_yx: Tuple[float, float],
    reference_name: str,
    calculate_distance: bool = True,
    calculate_overlap: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Calculates proximity and overlap between Primary and Reference 2D objects.

    Args:
        primary_mask_path: Path to .dat file for primary segmentation.
        reference_mask_path: Path to .dat file for reference segmentation.
        output_dir: Directory for results.
        shape: (Y, X) dimensions.
        spacing_yx: (dy, dx) in microns.
        reference_name: Name of partner (e.g. 'Plaque' or 'Vessel').
        
    Returns:
        - primary_df: Per-cell interaction stats.
        - ref_df: Per-reference object coverage stats.
        - intersection_path: Path to the labeled intersection mask.
    """
    flush_print(f"--- Starting 2D Interaction Analysis vs '{reference_name}' ---")

    # 1. Load Data
    primary_memmap = _safe_load_memmap(primary_mask_path, shape)
    reference_memmap = _safe_load_memmap(reference_mask_path, shape)

    if primary_memmap is None or reference_memmap is None:
        raise FileNotFoundError("Could not load 2D segmentation masks.")

    # 2. Reference Accumulators
    ref_areas = {}
    ref_interactions = []
    unit_area = spacing_yx[0] * spacing_yx[1]

    if calculate_overlap:
        flush_print("  Calculating areas of reference objects...")
        u, c = np.unique(reference_memmap, return_counts=True)
        ref_areas = dict(zip(u, c))
        if 0 in ref_areas: del ref_areas[0]

    # 3. Intersection Mask
    intersection_path = None
    intersection_memmap = None
    if calculate_overlap:
        intersection_path = os.path.join(output_dir, f"intersection_{reference_name}.dat")
        intersection_memmap = np.memmap(intersection_path, dtype=np.int32, mode='w+', shape=shape)

    # 4. Distance Mapping (EDT)
    dist_map = None
    indices = None
    if calculate_distance:
        flush_print(f"  Calculating 2D Distance Transform to nearest {reference_name}...")
        try:
            ref_binary_inverted = (reference_memmap == 0)
            # spacing_yx handles anisotropy (e.g. non-square pixels)
            dist_map, indices = distance_transform_edt(
                ref_binary_inverted, sampling=spacing_yx, return_indices=True
            )
            dist_map = dist_map.astype(np.float32)
        except MemoryError:
            flush_print("    Error: Not enough RAM for 2D EDT.")
            calculate_distance = False

    # 5. Iterate Primary Objects
    object_slices = ndimage.find_objects(primary_memmap)
    labels = np.unique(primary_memmap)
    labels = labels[labels > 0]
    primary_results = []

    for lbl in tqdm(labels, desc=f"    Analyzing {reference_name} proximity"):
        idx = lbl - 1
        if idx >= len(object_slices) or object_slices[idx] is None: continue

        sl = object_slices[idx]
        mask_p = (primary_memmap[sl] == lbl)
        crop_r = reference_memmap[sl]
        row = {'label': lbl}

        # --- Overlap Logic ---
        if calculate_overlap:
            intersect_mask = mask_p & (crop_r > 0)
            overlap_px = np.count_nonzero(intersect_mask)
            
            if overlap_px > 0 and intersection_memmap is not None:
                intersection_memmap[sl][intersect_mask] = 1

            total_px_p = np.count_nonzero(mask_p)
            row[f'overlap_area_um2_{reference_name}'] = overlap_px * unit_area
            row[f'overlap_fraction_{reference_name}'] = overlap_px / total_px_p if total_px_p > 0 else 0
            row[f'overlaps_any_{reference_name}'] = (overlap_px > 0)

            dom_id = 0
            dom_px_intersect = 0
            if overlap_px > 0:
                overlap_ids, overlap_counts = np.unique(crop_r[intersect_mask], return_counts=True)
                for o_id, o_count in zip(overlap_ids, overlap_counts):
                    if o_id == 0: continue
                    ref_interactions.append({'ref_label': o_id, 'overlap_area': o_count * unit_area, 'primary_label': lbl})
                
                valid = np.where(overlap_ids > 0)[0]
                if valid.size > 0:
                    dom_idx = valid[np.argmax(overlap_counts[valid])]
                    dom_id = overlap_ids[dom_idx]
                    dom_px_intersect = overlap_counts[dom_idx]

            row[f'dominant_overlap_id_{reference_name}'] = dom_id
            if dom_id > 0 and dom_id in ref_areas:
                row[f'overlap_fraction_of_partner_{reference_name}'] = dom_px_intersect / ref_areas[dom_id]
            else:
                row[f'overlap_fraction_of_partner_{reference_name}'] = 0.0

        # --- Distance Logic ---
        if calculate_distance and dist_map is not None:
            dist_crop = dist_map[sl]
            if np.any(mask_p):
                min_dist = np.min(dist_crop[mask_p])
                row[f'dist_to_nearest_{reference_name}_um'] = min_dist
                
                # Find ID of the nearest neighbor
                local_mins = np.argwhere((dist_crop == min_dist) & mask_p)
                if local_mins.size > 0:
                    y_g, x_g = (local_mins[0][0] + sl[0].start, local_mins[0][1] + sl[1].start)
                    nearest_y = int(indices[0, y_g, x_g])
                    nearest_x = int(indices[1, y_g, x_g])
                    row[f'nearest_{reference_name}_id'] = reference_memmap[nearest_y, nearest_x]
            else:
                row[f'dist_to_nearest_{reference_name}_um'] = np.nan
                row[f'nearest_{reference_name}_id'] = 0

        primary_results.append(row)

    # 6. Post-Process Intersection (Unique Labeling)
    if calculate_overlap and intersection_memmap is not None:
        flush_print("  Unique labeling of 2D overlap regions...")
        intersection_memmap.flush()
        d_int = da.from_array(intersection_memmap, chunks=(4096, 4096))
        # 8-connectivity for 2D
        labeled_int, _ = dask_image.ndmeasure.label(d_int, structure=np.ones((3, 3)))
        with ProgressBar(dt=2):
            da.store(labeled_int, intersection_memmap, lock=True)
        intersection_memmap.flush()

    # 7. Aggregate Reference Stats
    ref_df = pd.DataFrame()
    if ref_interactions:
        inter_df = pd.DataFrame(ref_interactions)
        ref_df = inter_df.groupby('ref_label').agg(
            total_overlap_area_um2=('overlap_area', 'sum'),
            interacting_cell_count=('primary_label', 'nunique'),
            interacting_labels=('primary_label', lambda x: list(x))
        ).reset_index()
        ref_df['ref_total_area_um2'] = ref_df['ref_label'].map(ref_areas).fillna(0) * unit_area
        ref_df['percent_covered'] = (ref_df['total_overlap_area_um2'] / ref_df['ref_total_area_um2'].replace(0, 1)) * 100.0

    # 8. Pairwise Distance Calculation (Long Format)
    pairwise_df = calculate_pairwise_distances_2d(primary_memmap, reference_memmap, spacing_yx)
    
    if not pairwise_df.empty:
        pairwise_out_path = os.path.join(output_dir, f"pairwise_distances_{reference_name}.csv")
        # Use compression if the table is very large (thousands of objects)
        pairwise_df.to_csv(pairwise_out_path, index=False)
        flush_print(f"  [Success] Pairwise distances saved to: {os.path.basename(pairwise_out_path)}")

    # Cleanup
    if intersection_memmap is not None:
        intersection_memmap.flush()
        if hasattr(intersection_memmap, '_mmap') and intersection_memmap._mmap:
            intersection_memmap._mmap.close()

    del dist_map, indices, primary_memmap, reference_memmap
    gc.collect()

    return pd.DataFrame(primary_results), ref_df, intersection_path