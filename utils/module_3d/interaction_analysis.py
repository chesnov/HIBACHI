# --- START OF FILE utils/module_3d/interaction_analysis.py ---
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import os
import gc
from tqdm import tqdm
from typing import Tuple, Optional, Dict, List

# Dask for memory-safe labeling of the intersection mask
import dask.array as da
import dask_image.ndmeasure
from dask.diagnostics import ProgressBar

def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    import sys
    sys.stdout.flush()

def _safe_load_memmap(path: str, shape: Tuple, dtype=np.int32, mode='r'):
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
    output_dir: str, # Needed to save intersection mask
    shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    reference_name: str,
    calculate_distance: bool = True,
    calculate_overlap: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Calculates relationships between Primary and Reference objects.
    
    Returns:
        primary_df: Stats per Primary Cell.
        ref_df: Stats per Reference Object (Coverage).
        intersection_mask_path: Path to the labeled intersection mask.
    """
    flush_print(f"--- Starting Interaction Analysis vs '{reference_name}' ---")
    
    # 1. Load Data
    primary_memmap = _safe_load_memmap(primary_mask_path, shape)
    reference_memmap = _safe_load_memmap(reference_mask_path, shape)
    
    if primary_memmap is None or reference_memmap is None:
        raise FileNotFoundError("Could not load segmentation masks.")

    # 2. Setup Reference Volumes & Accumulators
    ref_volumes = {}
    ref_interactions = [] # List of (ref_id, overlap_vol, primary_id)
    
    if calculate_overlap:
        flush_print("  Calculating volumes of reference objects...")
        try:
            u, c = np.unique(reference_memmap, return_counts=True)
            ref_volumes = dict(zip(u, c))
            if 0 in ref_volumes: del ref_volumes[0]
        except MemoryError:
            flush_print("    Warning: Reference mask too large for global stats.")

    # 3. Setup Intersection Mask (Binary first)
    intersection_path = None
    intersection_memmap = None
    
    if calculate_overlap:
        intersection_path = os.path.join(output_dir, f"intersection_{reference_name}.dat")
        intersection_memmap = np.memmap(intersection_path, dtype=np.int32, mode='w+', shape=shape)
        # We start with 0. We will set to 1 where overlap exists.
        # Later we re-label it to unique IDs.

    # 4. Setup Distance Map
    dist_map = None
    indices = None
    
    if calculate_distance:
        flush_print("  Calculating Distance Transform of Reference Channel...")
        # Note: Inverting a huge memmap to bool might be heavy.
        # Chunking EDT is safer for massive data but requires overlap handling.
        # For typical 16GB RAM, standard scipy EDT on <2GB mask is fine.
        
        # Optimization: Process boolean conversion in chunks if needed?
        # Standard approach for now:
        try:
            ref_binary = (reference_memmap > 0)
            dt_tuple = distance_transform_edt(
                np.logical_not(ref_binary), 
                sampling=spacing, 
                return_indices=True
            )
            dist_map = dt_tuple[0].astype(np.float32)
            indices = dt_tuple[1]
            del ref_binary
        except MemoryError:
            flush_print("    Error: Not enough RAM for Distance Transform. Skipping distance metrics.")
            calculate_distance = False

    # 5. Iterate Primary Objects
    flush_print("  Analyzing object interactions...")
    
    object_slices = ndimage.find_objects(primary_memmap)
    labels = np.unique(primary_memmap)
    labels = labels[labels > 0]
    
    primary_results = []
    voxel_vol = np.prod(spacing)
    
    for lbl in tqdm(labels, desc=f"    Scanning {reference_name}"):
        idx = lbl - 1
        if idx >= len(object_slices) or object_slices[idx] is None: continue
        
        sl = object_slices[idx]
        mask_p = (primary_memmap[sl] == lbl)
        crop_r = reference_memmap[sl]
        
        row = {'label': lbl}
        
        # --- A. Overlap Metrics ---
        if calculate_overlap:
            # Calculate Intersection Logic
            intersect_mask = mask_p & (crop_r > 0)
            overlap_vox = np.count_nonzero(intersect_mask)
            
            # Write to Intersection Map (Binary marking)
            if overlap_vox > 0:
                # We use the existing 0/1 logic. 
                # Later we relabel.
                # Writing to memmap slice:
                current_int_view = intersection_memmap[sl]
                # Logic: Set to 1 where intersection is True. 
                # (OR operation to handle potential overlap from other cells if labels touch, 
                # though primary labels shouldn't overlap).
                current_int_view[intersect_mask] = 1
                intersection_memmap[sl] = current_int_view

            total_vox_p = np.count_nonzero(mask_p)
            overlap_frac_p = (overlap_vox / total_vox_p) if total_vox_p > 0 else 0.0
            
            row[f'overlap_vol_um3_{reference_name}'] = overlap_vox * voxel_vol
            row[f'overlap_fraction_{reference_name}'] = overlap_frac_p 
            row[f'overlaps_any_{reference_name}'] = (overlap_vox > 0)
            
            # Identify partners
            dom_id = 0
            if overlap_vox > 0:
                overlap_ids, overlap_counts = np.unique(crop_r[intersect_mask], return_counts=True)
                valid = overlap_ids > 0
                overlap_ids = overlap_ids[valid]
                overlap_counts = overlap_counts[valid]
                
                # Accumulate for Reference Stats
                for o_id, o_count in zip(overlap_ids, overlap_counts):
                    ref_interactions.append({
                        'ref_label': o_id,
                        'overlap_vol': o_count * voxel_vol,
                        'primary_label': lbl
                    })

                if overlap_ids.size > 0:
                    dom_idx = np.argmax(overlap_counts)
                    dom_id = overlap_ids[dom_idx]
                    dom_vox_intersect = overlap_counts[dom_idx]
            
            row[f'dominant_overlap_id_{reference_name}'] = dom_id
            
            # Fraction of Partner (Dominant)
            if dom_id > 0 and dom_id in ref_volumes:
                total_vox_r = ref_volumes[dom_id]
                # Note: This is partial coverage by THIS cell.
                row[f'overlap_fraction_of_partner_{reference_name}'] = dom_vox_intersect / total_vox_r
            else:
                row[f'overlap_fraction_of_partner_{reference_name}'] = 0.0

        # --- B. Distance Metrics ---
        if calculate_distance and dist_map is not None:
            dist_crop = dist_map[sl]
            
            if np.any(mask_p):
                min_dist = np.min(dist_crop[mask_p])
                row[f'dist_to_nearest_{reference_name}_um'] = min_dist
                
                # Nearest ID
                local_mins = np.argwhere((dist_crop == min_dist) & mask_p)
                if local_mins.size > 0:
                    lz, ly, lx = local_mins[0]
                    gz, gy, gx = lz + sl[0].start, ly + sl[1].start, lx + sl[2].start
                    try:
                        sz, sy, sx = indices[:, gz, gy, gx]
                        nearest_id = reference_memmap[sz, sy, sx]
                        row[f'nearest_{reference_name}_id'] = nearest_id
                    except:
                         row[f'nearest_{reference_name}_id'] = 0
            else:
                row[f'dist_to_nearest_{reference_name}_um'] = np.nan
                row[f'nearest_{reference_name}_id'] = 0
        
        primary_results.append(row)

    # 6. Post-Process Intersection Mask (Unique Labeling)
    if calculate_overlap and intersection_memmap is not None:
        flush_print("  Labeling overlap regions...")
        intersection_memmap.flush()
        
        # Use Dask for memory-safe connected components
        d_int = da.from_array(intersection_memmap, chunks=(64, 256, 256))
        # Structure for 26-connectivity
        s = ndimage.generate_binary_structure(3, 3) 
        
        labeled_int, num_features = dask_image.ndmeasure.label(d_int, structure=s)
        
        # Overwrite the binary 1s with unique IDs
        flush_print("  Writing unique overlap IDs to disk...")
        with ProgressBar(dt=2):
            da.store(labeled_int, intersection_memmap, lock=True)
            
        intersection_memmap.flush()
        # intersection_path now contains unique integer IDs for every blob

    # 7. Aggregate Reference Stats
    ref_df = pd.DataFrame()
    if ref_interactions:
        flush_print("  Aggregating Reference Coverage Stats...")
        inter_df = pd.DataFrame(ref_interactions)
        
        # Group by Reference Label
        # Sum overlap volume
        # Count unique primary partners
        grp = inter_df.groupby('ref_label')
        
        ref_stats = grp.agg(
            total_overlap_vol_um3=('overlap_vol', 'sum'),
            interacting_cell_count=('primary_label', 'nunique'),
            interacting_labels=('primary_label', lambda x: list(x))
        ).reset_index()
        
        # Calculate % Covered
        ref_stats['ref_total_vol_um3'] = ref_stats['ref_label'].map(ref_volumes).fillna(0) * voxel_vol
        ref_stats['percent_covered'] = (ref_stats['total_overlap_vol_um3'] / ref_stats['ref_total_vol_um3']) * 100.0
        ref_stats['percent_covered'] = ref_stats['percent_covered'].clip(upper=100.0)
        
        ref_df = ref_stats

    # Cleanup
    del dist_map, indices
    gc.collect()
    
    return pd.DataFrame(primary_results), ref_df, intersection_path

# --- END OF FILE utils/module_3d/interaction_analysis.py ---