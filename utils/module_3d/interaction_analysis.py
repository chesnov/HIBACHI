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
    primary_name: str,    # Descriptive name of the masks being analyzed (e.g. Agg_in_Neur)
    partner_name: str,    # Descriptive name of the reference channel (e.g. Microglia)
    calculate_distance: bool = True,
    calculate_overlap: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Calculates spatial relationships between Primary cells and Partner objects.
    Supports both 2D and 3D data, but optimized for 3D volumes.

    Metrics:
    1. Overlap Volume (intersection).
    2. Overlap Fraction (Bi-directional: Primary-in-Partner and Partner-occupied-by-Primary).
    3. Nearest Neighbor Distance (Primary edge to Partner edge).
    4. Bridge Coordinates (Source and Target points for visual 'red lines').

    Args:
        primary_mask_path: Path to .dat file for primary segmentation.
        reference_mask_path: Path to .dat file for partner segmentation.
        output_dir: Directory to save the intersection mask and csvs.
        shape: Dimensions of the arrays (Z, Y, X).
        spacing: Physical spacing (dz, dy, dx).
        primary_name: Biological name of the primary objects.
        partner_name: Biological name of the partner objects.
        calculate_distance: Whether to compute distance transforms and bridge lines.
        calculate_overlap: Whether to compute intersection metrics.

    Returns:
        Tuple containing:
        - primary_df (pd.DataFrame): Interaction stats for each primary object.
        - ref_df (pd.DataFrame): Coverage stats for partner objects.
        - intersection_path (str): Path to the labeled intersection mask file.
    """
    flush_print(f"--- Starting Interaction Analysis: '{primary_name}' vs '{partner_name}' ---")

    ndim = len(shape)
    
    # 0. Spacing Adaptation
    edt_spacing = spacing
    if len(spacing) != ndim:
        if len(spacing) > ndim:
            edt_spacing = spacing[-ndim:]
            print(f"  Adapted spacing for {ndim}D: {edt_spacing}")
        else:
            edt_spacing = tuple(1.0 for _ in range(ndim))

    # 1. Load Data
    primary_memmap = _safe_load_memmap(primary_mask_path, shape)
    reference_memmap = _safe_load_memmap(reference_mask_path, shape)

    if primary_memmap is None or reference_memmap is None:
        raise FileNotFoundError("Could not load segmentation masks.")

    # 2. Setup Partner Volumes & Accumulators
    ref_volumes = {}
    ref_interactions = []  # List of dicts for partner-view stats

    if calculate_overlap:
        flush_print(f"  Calculating volumes of {partner_name} objects...")
        try:
            u, c = np.unique(reference_memmap, return_counts=True)
            ref_volumes = dict(zip(u, c))
            if 0 in ref_volumes:
                del ref_volumes[0]
        except MemoryError:
            flush_print(f"    Warning: {partner_name} mask too large for global stats.")

    # 3. Setup Intersection Mask
    intersection_path = None
    intersection_memmap = None

    if calculate_overlap:
        intersection_path = os.path.join(
            output_dir, f"intersection_{partner_name}.dat"
        )
        intersection_memmap = np.memmap(
            intersection_path, dtype=np.int32, mode='w+', shape=shape
        )

    # 4. Setup Distance Map
    dist_map = None
    indices = None

    if calculate_distance:
        flush_print(f"  Calculating Distance Transform to nearest {partner_name}...")
        try:
            # Create binary mask of partner (inverted for EDT)
            ref_binary_inverted = (reference_memmap == 0)
            
            dt_tuple = distance_transform_edt(
                ref_binary_inverted,
                sampling=edt_spacing,
                return_indices=True
            )
            dist_map = dt_tuple[0].astype(np.float32)
            indices = dt_tuple[1]  # Shape: (ndim, Z, Y, X)
            
            del ref_binary_inverted
        except MemoryError:
            flush_print("    Error: Not enough RAM for Distance Transform.")
            calculate_distance = False

    # 5. Iterate Primary Objects
    flush_print(f"  Analyzing {primary_name} interactions...")

    object_slices = ndimage.find_objects(primary_memmap)
    labels = np.unique(primary_memmap)
    labels = labels[labels > 0]

    primary_results = []
    unit_vol = np.prod(spacing)

    for lbl in tqdm(labels, desc=f"    Scanning {partner_name}"):
        idx = lbl - 1
        if idx >= len(object_slices) or object_slices[idx] is None:
            continue

        sl = object_slices[idx]
        mask_p = (primary_memmap[sl] == lbl)
        crop_r = reference_memmap[sl]

        row = {'label': lbl}

        # --- A. Overlap Metrics (3D) ---
        if calculate_overlap:
            intersect_mask = mask_p & (crop_r > 0)
            overlap_vox = np.count_nonzero(intersect_mask)

            # Mark intersection for Napari
            if overlap_vox > 0 and intersection_memmap is not None:
                current_int_view = intersection_memmap[sl]
                current_int_view[intersect_mask] = 1
                intersection_memmap[sl] = current_int_view

            total_vox_p = np.count_nonzero(mask_p)
            
            # Bi-directional Stat: Volume of Primary inside Partner
            row[f'overlap_vol_with_{partner_name}_um3'] = overlap_vox * unit_vol
            
            # Bi-directional Stat: % of this Primary contained by Partner
            row[f'pct_of_this_{primary_name}_inside_{partner_name}'] = \
                (overlap_vox / total_vox_p) * 100.0 if total_vox_p > 0 else 0.0
            
            row[f'is_touching_{partner_name}'] = (overlap_vox > 0)

            # Identify partner IDs for aggregation
            dom_id = 0
            if overlap_vox > 0:
                overlap_ids, overlap_counts = np.unique(
                    crop_r[intersect_mask], return_counts=True
                )
                
                # Store interactions for coverage_stats.csv
                for o_id, o_count in zip(overlap_ids, overlap_counts):
                    if o_id == 0: continue
                    ref_interactions.append({
                        'ref_label': o_id, 
                        'overlap_vol': o_count * unit_vol, 
                        'primary_label': lbl
                    })

                # Find dominant partner ID
                valid_indices = np.where(overlap_ids > 0)[0]
                if valid_indices.size > 0:
                    dom_idx = valid_indices[np.argmax(overlap_counts[valid_indices])]
                    dom_id = overlap_ids[dom_idx]

            row[f'dominant_partner_id_{partner_name}'] = dom_id

        # --- B. Distance Metrics (3D) ---
        if calculate_distance and dist_map is not None and indices is not None:
            dist_crop = dist_map[sl]
            if np.any(mask_p):
                # 1. Minimum Distance
                min_dist = np.min(dist_crop[mask_p])
                row[f'dist_um_{partner_name}'] = min_dist

                # 2. Identify Bridge Coordinates (Source and Target)
                local_mins = np.argwhere((dist_crop == min_dist) & mask_p)
                if local_mins.size > 0:
                    local_src = local_mins[0]
                    global_src = tuple(l_c + s.start for l_c, s in zip(local_src, sl))
                    
                    # Target: Exact pixel on the partner mask
                    indexer = (slice(None),) + global_src
                    global_target = tuple(int(c) for c in indices[indexer])
                    
                    row[f'nearest_id_{partner_name}'] = reference_memmap[global_target]
                    
                    # Coordinates for Napari Bridge Lines
                    row[f'src_z_{partner_name}'] = global_src[0]
                    row[f'src_y_{partner_name}'] = global_src[1]
                    row[f'src_x_{partner_name}'] = global_src[2]
                    row[f'tgt_z_{partner_name}'] = global_target[0]
                    row[f'tgt_y_{partner_name}'] = global_target[1]
                    row[f'tgt_x_{partner_name}'] = global_target[2]
                else:
                    row[f'nearest_id_{partner_name}'] = 0
            else:
                row[f'dist_um_{partner_name}'] = np.nan
                row[f'nearest_id_{partner_name}'] = 0

        primary_results.append(row)

    # 6. Post-Process Intersection Mask (Unique Labeling)
    if calculate_overlap and intersection_memmap is not None:
        flush_print(f"  Labeling unique overlap regions between {primary_name} and {partner_name}...")
        intersection_memmap.flush()

        dask_chunks = (64, 256, 256) if ndim == 3 else (4096, 4096)
        d_int = da.from_array(intersection_memmap, chunks=dask_chunks)
        s = generate_binary_structure(ndim, ndim)

        labeled_int, num_features = dask_image.ndmeasure.label(d_int, structure=s)

        with ProgressBar(dt=2):
            da.store(labeled_int.astype(np.int32), intersection_memmap, lock=True)

        intersection_memmap.flush()

    # 7. Aggregate Partner Coverage Stats (Partner View)
    ref_df = pd.DataFrame()
    if ref_interactions:
        flush_print(f"  Aggregating coverage stats for {partner_name}...")
        inter_df = pd.DataFrame(ref_interactions)
        grp = inter_df.groupby('ref_label')

        ref_stats = grp.agg(
            overlap_vol_um3=('overlap_vol', 'sum'),
            touching_count=('primary_label', 'nunique'),
            touching_ids=('primary_label', lambda x: list(x))
        ).reset_index()

        # Biologically Explicit Naming
        ref_stats = ref_stats.rename(columns={
            'ref_label': f'id_{partner_name}',
            'overlap_vol_um3': f'total_vol_of_{primary_name}_inside_this_{partner_name}',
            'touching_count': f'count_of_unique_{primary_name}_touching_this_{partner_name}',
            'touching_ids': f'list_of_{primary_name}_ids_touching_this_{partner_name}'
        })

        # Calculate Percentages
        partner_total_vol = ref_stats[f'id_{partner_name}'].map(ref_volumes).fillna(0) * unit_vol
        ref_stats[f'total_vol_of_this_{partner_name}'] = partner_total_vol
        
        # Calculation: How much of the Partner (e.g. Neuron) is filled with Primary (e.g. Aggregates)
        ref_stats[f'pct_of_{partner_name}_occupied_by_{primary_name}'] = \
            (ref_stats[f'total_vol_of_{primary_name}_inside_this_{partner_name}'] / partner_total_vol.replace(0, 1)) * 100.0

        ref_df = ref_stats

    # Cleanup
    if intersection_memmap is not None:
        intersection_memmap.flush()
        if hasattr(intersection_memmap, '_mmap') and intersection_memmap._mmap:
            intersection_memmap._mmap.close()
            
    del dist_map, indices, primary_memmap, reference_memmap
    gc.collect()

    return pd.DataFrame(primary_results), ref_df, intersection_path