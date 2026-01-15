import os
import sys
import gc
import traceback
from typing import Tuple, Optional, Dict, List, Any, Union

import numpy as np
import pandas as pd
import dask.array as da
import dask_image.ndmeasure
from dask.diagnostics import ProgressBar
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, generate_binary_structure
from tqdm import tqdm


def flush_print(*args: Any, **kwargs: Any) -> None:
    """Wrapper for print that forces immediate flushing of stdout."""
    print(*args, **kwargs)
    sys.stdout.flush()


def _safe_load_memmap(
    path: str,
    shape: Tuple[int, ...],
    dtype: type = np.int32,
    mode: str = 'r'
) -> Optional[np.memmap]:
    """
    Safely loads a numpy memmap file.

    Args:
        path: File path.
        shape: Shape of the array.
        dtype: Data type.
        mode: Open mode ('r', 'r+', 'w+').

    Returns:
        np.memmap object or None if file missing/error.
    """
    if not os.path.exists(path):
        return None
    try:
        return np.memmap(path, dtype=dtype, mode=mode, shape=shape)
    except Exception as e:
        print(f"Error loading memmap {path}: {e}")
        return None


def calculate_interaction_metrics(
    primary_mask_path: str,
    reference_mask_path: str,
    output_dir: str,
    shape: Tuple[int, ...],
    spacing: Tuple[float, ...],
    reference_name: str,
    calculate_distance: bool = True,
    calculate_overlap: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Calculates spatial relationships between Primary cells and Reference objects.
    Supports both 2D and 3D data.

    Metrics:
    1. Overlap Volume/Area (intersection).
    2. Overlap Fraction (relative to primary and partner).
    3. Nearest Neighbor Distance (Primary edge to Reference edge).

    Args:
        primary_mask_path: Path to .dat file for primary segmentation.
        reference_mask_path: Path to .dat file for reference segmentation.
        output_dir: Directory to save the intersection mask.
        shape: Dimensions of the arrays (e.g. (Z,Y,X) or (Y,X)).
        spacing: Physical spacing (e.g. (1.0, 0.5, 0.5)).
        reference_name: Label string for the reference (e.g., 'Plaque').
        calculate_distance: Whether to compute distance transforms.
        calculate_overlap: Whether to compute intersection metrics.

    Returns:
        Tuple containing:
        - primary_df (pd.DataFrame): Interaction stats for each primary cell.
        - ref_df (pd.DataFrame): Coverage stats for reference objects.
        - intersection_path (str): Path to the labeled intersection mask file.
    """
    flush_print(f"--- Starting Interaction Analysis vs '{reference_name}' ---")

    ndim = len(shape)
    
    # 0. Spacing Adaptation (Handle 2D shape with 3D spacing inputs)
    # Some 2D workflows pass spacing as (1.0, dy, dx) for (Y, X) shape.
    edt_spacing = spacing
    if len(spacing) != ndim:
        if len(spacing) > ndim:
            # Take last N dimensions (e.g. [1, y, x] -> [y, x])
            edt_spacing = spacing[-ndim:]
            print(f"  Adapted spacing for {ndim}D: {edt_spacing}")
        else:
            # Fallback
            edt_spacing = tuple(1.0 for _ in range(ndim))

    # 1. Load Data
    primary_memmap = _safe_load_memmap(primary_mask_path, shape)
    reference_memmap = _safe_load_memmap(reference_mask_path, shape)

    if primary_memmap is None or reference_memmap is None:
        raise FileNotFoundError("Could not load segmentation masks.")

    # 2. Setup Reference Volumes & Accumulators
    ref_volumes = {}
    ref_interactions = []  # List of dicts for reference stats

    if calculate_overlap:
        flush_print("  Calculating volumes/areas of reference objects...")
        try:
            u, c = np.unique(reference_memmap, return_counts=True)
            ref_volumes = dict(zip(u, c))
            if 0 in ref_volumes:
                del ref_volumes[0]
        except MemoryError:
            flush_print("    Warning: Reference mask too large for global stats.")

    # 3. Setup Intersection Mask
    intersection_path = None
    intersection_memmap = None

    if calculate_overlap:
        intersection_path = os.path.join(
            output_dir, f"intersection_{reference_name}.dat"
        )
        intersection_memmap = np.memmap(
            intersection_path, dtype=np.int32, mode='w+', shape=shape
        )

    # 4. Setup Distance Map
    dist_map = None
    indices = None

    if calculate_distance:
        flush_print("  Calculating Distance Transform of Reference Channel...")
        try:
            # Create binary mask of reference (inverted for EDT)
            # True = Background (calculate distance to nearest foreground)
            # False = Foreground (distance 0)
            ref_binary_inverted = (reference_memmap == 0)
            
            dt_tuple = distance_transform_edt(
                ref_binary_inverted,
                sampling=edt_spacing,
                return_indices=True
            )
            dist_map = dt_tuple[0].astype(np.float32)
            indices = dt_tuple[1]  # Shape: (ndim, *shape)
            
            del ref_binary_inverted
        except MemoryError:
            flush_print("    Error: Not enough RAM for Distance Transform.")
            calculate_distance = False

    # 5. Iterate Primary Objects
    flush_print("  Analyzing object interactions...")

    object_slices = ndimage.find_objects(primary_memmap)
    labels = np.unique(primary_memmap)
    labels = labels[labels > 0]

    primary_results = []
    
    # Unit volume/area based on full spacing vector
    # (Use original spacing to preserve e.g. Z-depth if 2D was passed as flat 3D)
    unit_vol = np.prod(spacing)

    for lbl in tqdm(labels, desc=f"    Scanning {reference_name}"):
        idx = lbl - 1
        if idx >= len(object_slices) or object_slices[idx] is None:
            continue

        sl = object_slices[idx]
        mask_p = (primary_memmap[sl] == lbl)
        crop_r = reference_memmap[sl]

        row = {'label': lbl}

        # --- A. Overlap Metrics ---
        if calculate_overlap:
            # Intersection in local crop
            intersect_mask = mask_p & (crop_r > 0)
            overlap_vox = np.count_nonzero(intersect_mask)

            # Write to Intersection Map
            if overlap_vox > 0 and intersection_memmap is not None:
                current_int_view = intersection_memmap[sl]
                # Mark intersection (binary 1 for now)
                current_int_view[intersect_mask] = 1
                intersection_memmap[sl] = current_int_view

            total_vox_p = np.count_nonzero(mask_p)
            overlap_frac_p = (overlap_vox / total_vox_p) if total_vox_p > 0 else 0.0

            row[f'overlap_vol_um3_{reference_name}'] = overlap_vox * unit_vol
            row[f'overlap_fraction_{reference_name}'] = overlap_frac_p
            row[f'overlaps_any_{reference_name}'] = (overlap_vox > 0)

            # Identify partners
            dom_id = 0
            dom_vox_intersect = 0
            
            if overlap_vox > 0:
                overlap_ids, overlap_counts = np.unique(
                    crop_r[intersect_mask], return_counts=True
                )
                
                # Record detailed interactions
                for o_id, o_count in zip(overlap_ids, overlap_counts):
                    if o_id == 0: continue
                    ref_interactions.append({
                        'ref_label': o_id,
                        'overlap_vol': o_count * unit_vol,
                        'primary_label': lbl
                    })

                # Find dominant partner
                valid_indices = np.where(overlap_ids > 0)[0]
                if valid_indices.size > 0:
                    dom_idx = valid_indices[np.argmax(overlap_counts[valid_indices])]
                    dom_id = overlap_ids[dom_idx]
                    dom_vox_intersect = overlap_counts[dom_idx]

            row[f'dominant_overlap_id_{reference_name}'] = dom_id

            # Fraction of Partner Covered by this Cell
            if dom_id > 0 and dom_id in ref_volumes:
                total_vox_r = ref_volumes[dom_id]
                row[f'overlap_fraction_of_partner_{reference_name}'] = \
                    dom_vox_intersect / total_vox_r
            else:
                row[f'overlap_fraction_of_partner_{reference_name}'] = 0.0

        # --- B. Distance Metrics ---
        if calculate_distance and dist_map is not None and indices is not None:
            dist_crop = dist_map[sl]

            if np.any(mask_p):
                # 1. Minimum Distance
                min_dist = np.min(dist_crop[mask_p])
                row[f'dist_to_nearest_{reference_name}_um'] = min_dist

                # 2. Identify Nearest Neighbor ID
                # Find local coordinates of minimum
                local_mins = np.argwhere((dist_crop == min_dist) & mask_p)
                
                if local_mins.size > 0:
                    local_min = local_mins[0]
                    # Convert to global coordinates
                    global_coords = tuple(
                        local_c + s.start for local_c, s in zip(local_min, sl)
                    )
                    
                    try:
                        # indices array shape: (ndim, dim1, dim2...)
                        # Use tuple indexing to fetch nearest point coords
                        indexer = (slice(None),) + global_coords
                        nearest_point_flat = indices[indexer]
                        nearest_point = tuple(int(c) for c in nearest_point_flat)
                        
                        nearest_id = reference_memmap[nearest_point]
                        row[f'nearest_{reference_name}_id'] = nearest_id
                    except Exception:
                        row[f'nearest_{reference_name}_id'] = 0
                else:
                    row[f'nearest_{reference_name}_id'] = 0
            else:
                row[f'dist_to_nearest_{reference_name}_um'] = np.nan
                row[f'nearest_{reference_name}_id'] = 0

        primary_results.append(row)

    # 6. Post-Process Intersection Mask (Unique Labeling)
    if calculate_overlap and intersection_memmap is not None:
        flush_print("  Labeling overlap regions...")
        intersection_memmap.flush()

        # Chunk size heuristic: smaller chunks for 3D, larger for 2D
        dask_chunks = (64, 256, 256) if ndim == 3 else (4096, 4096)
        
        d_int = da.from_array(intersection_memmap, chunks=dask_chunks)
        
        # Structure for full connectivity (8-conn for 2D, 26-conn for 3D)
        s = generate_binary_structure(ndim, ndim)

        labeled_int, num_features = dask_image.ndmeasure.label(d_int, structure=s)

        flush_print("  Writing unique overlap IDs to disk...")
        with ProgressBar(dt=2):
            da.store(labeled_int, intersection_memmap, lock=True)

        intersection_memmap.flush()

    # 7. Aggregate Reference Stats
    ref_df = pd.DataFrame()
    if ref_interactions:
        flush_print("  Aggregating Reference Coverage Stats...")
        inter_df = pd.DataFrame(ref_interactions)

        grp = inter_df.groupby('ref_label')

        ref_stats = grp.agg(
            total_overlap_vol_um3=('overlap_vol', 'sum'),
            interacting_cell_count=('primary_label', 'nunique'),
            interacting_labels=('primary_label', lambda x: list(x))
        ).reset_index()

        # Calculate % Covered
        ref_stats['ref_total_vol_um3'] = \
            ref_stats['ref_label'].map(ref_volumes).fillna(0) * unit_vol
            
        ref_stats['percent_covered'] = (
            ref_stats['total_overlap_vol_um3'] / 
            ref_stats['ref_total_vol_um3'].replace(0, 1) # Avoid div/0
        ) * 100.0
        
        ref_stats['percent_covered'] = ref_stats['percent_covered'].clip(upper=100.0)

        ref_df = ref_stats

    # Cleanup
    if intersection_memmap is not None:
        intersection_memmap.flush()
        if hasattr(intersection_memmap, '_mmap') and intersection_memmap._mmap:
            intersection_memmap._mmap.close()
            
    # Explicitly clear large arrays (Indices can be 3x the size of the volume)
    del dist_map, indices, primary_memmap, reference_memmap
    gc.collect()

    return pd.DataFrame(primary_results), ref_df, intersection_path