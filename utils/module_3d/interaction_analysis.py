# --- START OF FILE utils/module_3d/interaction_analysis.py ---
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
import os
import gc
from tqdm import tqdm
from typing import Tuple, Optional, Dict

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
    shape: Tuple[int, int, int],
    spacing: Tuple[float, float, float],
    reference_name: str,
    calculate_distance: bool = True,
    calculate_overlap: bool = True
) -> pd.DataFrame:
    """
    Calculates spatial relationships between objects in the Primary Mask 
    and objects in the Reference Mask.
    """
    flush_print(f"--- Starting Interaction Analysis vs '{reference_name}' ---")
    
    # 1. Load Data
    primary_memmap = _safe_load_memmap(primary_mask_path, shape)
    reference_memmap = _safe_load_memmap(reference_mask_path, shape)
    
    if primary_memmap is None or reference_memmap is None:
        raise FileNotFoundError("Could not load segmentation masks.")

    # 2. Pre-calculate Reference Volumes
    # We need the TOTAL volume of each reference object to calculate 
    # "Overlap Fraction of Partner".
    # Since reference_memmap can be large, we scan it once.
    ref_volumes = {}
    if calculate_overlap:
        flush_print("  Calculating volumes of reference objects...")
        # np.unique on memmap reads the whole file. 
        # This is generally fast enough for < 10GB files on SSD.
        try:
            u, c = np.unique(reference_memmap, return_counts=True)
            ref_volumes = dict(zip(u, c))
            if 0 in ref_volumes: del ref_volumes[0] # Remove background
        except MemoryError:
            flush_print("    Warning: Reference mask too large for global stats. 'Fraction of Partner' will be skipped.")

    # 3. Distance Map Setup
    dist_map = None
    indices = None
    
    if calculate_distance:
        flush_print("  Calculating Distance Transform of Reference Channel...")
        ref_binary = (reference_memmap > 0)
        
        # Calculate EDT
        dt_tuple = distance_transform_edt(
            np.logical_not(ref_binary), 
            sampling=spacing, 
            return_indices=True
        )
        
        dist_map = dt_tuple[0].astype(np.float32)
        indices = dt_tuple[1]

    # 4. Iterate Primary Objects
    flush_print("  Analyzing object interactions...")
    
    object_slices = ndimage.find_objects(primary_memmap)
    labels = np.unique(primary_memmap)
    labels = labels[labels > 0]
    
    results = []
    voxel_vol = np.prod(spacing)
    
    for lbl in tqdm(labels, desc=f"    Comparing to {reference_name}"):
        idx = lbl - 1
        if idx >= len(object_slices) or object_slices[idx] is None: continue
        
        sl = object_slices[idx]
        mask_p = (primary_memmap[sl] == lbl)
        crop_r = reference_memmap[sl]
        
        row = {'label': lbl}
        
        # --- A. Overlap Metrics ---
        if calculate_overlap:
            intersection = mask_p & (crop_r > 0)
            overlap_vox = np.count_nonzero(intersection)
            
            total_vox_p = np.count_nonzero(mask_p)
            overlap_frac_p = (overlap_vox / total_vox_p) if total_vox_p > 0 else 0.0
            
            row[f'overlap_vol_um3_{reference_name}'] = overlap_vox * voxel_vol
            row[f'overlap_fraction_{reference_name}'] = overlap_frac_p # Fraction of SELF
            row[f'overlaps_any_{reference_name}'] = (overlap_vox > 0)
            
            # Identify dominant partner
            dom_id = 0
            if overlap_vox > 0:
                overlap_ids, overlap_counts = np.unique(crop_r[intersection], return_counts=True)
                valid = overlap_ids > 0
                overlap_ids = overlap_ids[valid]
                overlap_counts = overlap_counts[valid]
                
                if overlap_ids.size > 0:
                    dom_idx = np.argmax(overlap_counts)
                    dom_id = overlap_ids[dom_idx]
                    dom_vox_intersect = overlap_counts[dom_idx]
            
            row[f'dominant_overlap_id_{reference_name}'] = dom_id
            
            # Calculate Fraction of PARTNER
            # (Intersection / Total Volume of Reference Object)
            if dom_id > 0 and dom_id in ref_volumes:
                total_vox_r = ref_volumes[dom_id]
                row[f'overlap_fraction_of_partner_{reference_name}'] = dom_vox_intersect / total_vox_r
            else:
                row[f'overlap_fraction_of_partner_{reference_name}'] = 0.0

        # --- B. Distance Metrics ---
        if calculate_distance and dist_map is not None:
            dist_crop = dist_map[sl]
            
            if np.any(mask_p):
                min_dist = np.min(dist_crop[mask_p])
                row[f'dist_to_nearest_{reference_name}_um'] = min_dist
                
                # Identify Nearest Neighbor via Indices
                local_mins = np.argwhere((dist_crop == min_dist) & mask_p)
                if local_mins.size > 0:
                    lz, ly, lx = local_mins[0]
                    gz = lz + sl[0].start
                    gy = ly + sl[1].start
                    gx = lx + sl[2].start
                    
                    try:
                        sz = indices[0, gz, gy, gx]
                        sy = indices[1, gz, gy, gx]
                        sx = indices[2, gz, gy, gx]
                        nearest_id = reference_memmap[sz, sy, sx]
                        row[f'nearest_{reference_name}_id'] = nearest_id
                    except Exception:
                         row[f'nearest_{reference_name}_id'] = 0
            else:
                row[f'dist_to_nearest_{reference_name}_um'] = np.nan
                row[f'nearest_{reference_name}_id'] = 0
        
        results.append(row)

    del dist_map, indices
    gc.collect()
    
    if not results: return pd.DataFrame()
    return pd.DataFrame(results)
# --- END OF FILE utils/module_3d/interaction_analysis.py ---