"""
Cross-channel interaction analysis, 2D and 3D.

Quantifies spatial relationships between two segmentation layers: how much they
overlap, in both directions, and how far apart they are edge-to-edge.

Provenance
----------
This is the 2D implementation generalised to either rank. 2D was the tested,
authoritative version for cross-channel work, and where the two tracks disagreed
it wins. Two disagreements were real:

*   **Pairwise distances.** 2D extracts surface voxels per object, builds a
    cKDTree per primary and queries the partner contours, parallelised across
    processes. The 3D track instead computed one FULL-VOLUME distance transform
    per primary object -- O(objects x volume). Measured on an 11x1024x1024 stack
    that was 4.0 s per object, about 6 minutes for 87 objects, and it degrades
    with both cell count and image size. The contour/KDTree approach is the
    authoritative one and also the only tractable one, so it is used at both
    ranks.

*   **Pairwise table schema.** 2D emits ``primary_id`` / ``id_<partner>`` /
    ``dist_um_<partner>``; the 3D track emitted ``src_id`` / ``tgt_id`` /
    ``dist``. 2D's names are used at both ranks.

What stays rank-dependent, deliberately
---------------------------------------
Overlap is reported as an AREA in um2 in 2D and a VOLUME in um3 in 3D, and
bridge coordinates carry a Z component only in 3D. Those are not drift: an area
is not a volume, and there is no Z to report in a plane. Column names therefore
differ by rank. Nothing downstream is coupled to them -- ``relational_engine``
requires only a ``label`` column and otherwise passes the frames through.

Features:
- Informative, biologically explicit column naming.
- Bridge coordinate discovery for visual connection lines.
- Bi-directional coverage stats (Primary-in-Partner vs Partner-occupied-by-Primary).
- Memory-efficient processing via memmaps and Dask.
"""

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
from scipy.spatial import cKDTree
from tqdm import tqdm
import multiprocessing as mp

try:
    from .dim_utils import adjacency_footprint, binary_structure, normalise_spacing
except ImportError:  # pragma: no cover - direct script execution
    from dim_utils import adjacency_footprint, binary_structure, normalise_spacing


#: Axis names for bridge-coordinate columns, per rank. 3D gains a Z component;
#: there is no Z in a plane, so 2D has none to report.
_AXIS_NAMES = {2: ("y", "x"), 3: ("z", "y", "x")}


def _extent_terms(ndim: int):
    """
    (noun, unit) for an n-dimensional extent.

    An overlap is an area in 2D and a volume in 3D. Reporting either as the
    other would be wrong, so the column names differ by rank -- see the module
    docstring.
    """
    return ("area", "um2") if ndim == 2 else ("vol", "um3")

def flush_print(*args: Any, **kwargs: Any) -> None:
    """Standardized wrapper for immediate log flushing to console."""
    print(*args, **kwargs)
    sys.stdout.flush()

def _safe_load_memmap(
    path: str,
    shape: Tuple[int, ...],
    dtype: type = np.int32,
    mode: str = 'r'
) -> Optional[np.memmap]:
    """Loads a memmap of any rank, returning None rather than raising."""
    if not os.path.exists(path):
        return None
    try:
        return np.memmap(path, dtype=dtype, mode=mode, shape=shape)
    except Exception as e:
        print(f"Error loading memmap {path}: {e}")
        return None

def _extract_contours(mask_memmap, labels, shape=None):
    """
    Boundary coordinates per label, in global index space, at either rank.

    The boundary is the object minus its face-connected erosion: perimeter
    pixels in 2D, surface voxels in 3D. Only these are needed for edge-to-edge
    distance, which is what keeps the pairwise calculation tractable -- the
    interior never enters a KD-tree.
    """
    ndim = int(np.asarray(mask_memmap).ndim)
    contours = []
    locs = ndimage.find_objects(mask_memmap)
    struct = binary_structure(ndim, 1)
    for lbl in tqdm(labels, desc="    Extracting Boundaries", leave=False):
        sl = locs[lbl - 1]
        if sl is None:
            continue
        mask = (mask_memmap[sl] == lbl)
        eroded = ndimage.binary_erosion(mask, structure=struct)
        coords = np.where(mask ^ eroded)
        if len(coords[0]) > 0:
            contours.append(np.column_stack([
                coords[k] + sl[k].start for k in range(ndim)
            ]))
        else:
            contours.append(np.array([]))
    return contours

def _inter_channel_dist_worker(args):
    """Worker: Calculates distances from one primary object to all partner objects."""
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

def calculate_pairwise_distances(primary_memmap, reference_memmap, spacing,
                                 partner_name):
    """
    All pairwise edge-to-edge distances, as a long-format DataFrame.

    One KD-tree per primary object over its boundary coordinates, queried
    against every partner boundary. Physical distance comes from scaling
    coordinates by `spacing` before the query, so the result is in microns at
    either rank.
    """
    p_labels = np.unique(primary_memmap[primary_memmap > 0])
    r_labels = np.unique(reference_memmap[reference_memmap > 0])
    spacing_arr = np.array(spacing, dtype=np.float64)

    # 1. Extract Boundaries
    p_contours = _extract_contours(primary_memmap, p_labels)
    r_contours = _extract_contours(reference_memmap, r_labels)
    
    # 2. Parallel Execution
    n_jobs = max(1, mp.cpu_count() - 1)
    tasks = [(i, p_contours[i], r_contours, spacing_arr) for i in range(len(p_labels))]
    
    all_rows = []
    with mp.Pool(n_jobs) as pool:
        for prim_idx, distances in tqdm(pool.imap_unordered(_inter_channel_dist_worker, tasks), 
                                       total=len(tasks), desc="    Pairwise Distance"):
            p_label = p_labels[prim_idx]
            df_chunk = pd.DataFrame({
                'primary_id': p_label,
                f'id_{partner_name}': r_labels,
                f'dist_um_{partner_name}': distances
            })
            all_rows.append(df_chunk)
            
    if not all_rows:
        return pd.DataFrame()
        
    final_df = pd.concat(all_rows, ignore_index=True)
    del p_contours, r_contours, all_rows
    gc.collect()
    return final_df

def calculate_interaction_metrics(
    primary_mask_path: str,
    reference_mask_path: str,
    output_dir: str,
    shape: Tuple[int, ...],
    spacing: Tuple[float, ...],
    primary_name: str,    # Descriptive name of derived mask (e.g. Neurons_in_Aggregates)
    partner_name: str,    # Descriptive name of reference channel (e.g. Microglia)
    calculate_distance: bool = True,
    calculate_overlap: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Calculates spatial relationships between Primary and Partner 2D objects.
    
    Metrics include: 
    - Area of overlap
    - % of Primary inside Partner
    - % of Partner occupied by Primary
    - Edge-to-edge Euclidean distances
    - Source/Target Bridge coordinates for visualization
    """
    ndim = len(tuple(shape))
    if ndim not in (2, 3):
        raise ValueError(
            f"interaction analysis handles 2D and 3D data; got a {ndim}D shape"
        )
    # Required, not defaulted. Every metric below is physical -- overlap extent,
    # edge-to-edge distance -- so a substituted spacing makes all of them wrong
    # by a constant factor with nothing in the output to reveal it. The 3D track
    # fell back to `tuple(1.0 for _ in range(ndim))` when the spacing was the
    # wrong length; that is exactly the silent corruption this refuses.
    spacing = normalise_spacing(spacing, ndim)
    extent, unit = _extent_terms(ndim)
    axes = _AXIS_NAMES[ndim]

    flush_print(f"--- Starting {ndim}D Interaction Analysis: "
                f"'{primary_name}' vs '{partner_name}' ---")

    # 1. Load Data
    primary_memmap = _safe_load_memmap(primary_mask_path, shape)
    reference_memmap = _safe_load_memmap(reference_mask_path, shape)

    if primary_memmap is None or reference_memmap is None:
        raise FileNotFoundError("Could not load 2D segmentation masks.")

    # 2. Partner Accumulators
    ref_areas = {}
    ref_interactions = []
    # Physical extent of one voxel: an area in 2D, a volume in 3D.
    unit_extent = float(np.prod(spacing))

    if calculate_overlap:
        flush_print(f"  Calculating areas of {partner_name} objects...")
        u, c = np.unique(reference_memmap, return_counts=True)
        ref_areas = dict(zip(u, c))
        if 0 in ref_areas: del ref_areas[0]

    # 3. Setup Intersection Mask
    intersection_path = None
    intersection_memmap = None
    if calculate_overlap:
        intersection_path = os.path.join(output_dir, f"intersection_{partner_name}.dat")
        intersection_memmap = np.memmap(intersection_path, dtype=np.int32, mode='w+', shape=shape)

    # 4. Setup Distance Map (EDT)
    dist_map = None
    indices = None
    if calculate_distance:
        flush_print(f"  Calculating {ndim}D Distance Transform to nearest {partner_name}...")
        try:
            ref_binary_inverted = (reference_memmap == 0)
            dist_map, indices = distance_transform_edt(
                ref_binary_inverted, sampling=spacing, return_indices=True
            )
            dist_map = dist_map.astype(np.float32)
        except MemoryError:
            flush_print(f"    Error: Not enough RAM for the {ndim}D EDT.")
            calculate_distance = False

    # 5. Iterate Primary Objects
    flush_print(f"  Analyzing {primary_name} interactions...")
    object_slices = ndimage.find_objects(primary_memmap)
    labels = np.unique(primary_memmap)
    labels = labels[labels > 0]
    primary_results = []

    for lbl in tqdm(labels, desc=f"    Scanning {partner_name}"):
        idx = lbl - 1
        if idx >= len(object_slices) or object_slices[idx] is None: continue

        sl = object_slices[idx]
        mask_p = (primary_memmap[sl] == lbl)
        crop_r = reference_memmap[sl]
        row = {'label': lbl}

        # --- Overlap Logic (2D) ---
        if calculate_overlap:
            intersect_mask = mask_p & (crop_r > 0)
            overlap_px = np.count_nonzero(intersect_mask)
            
            if overlap_px > 0 and intersection_memmap is not None:
                current_int_view = intersection_memmap[sl]
                current_int_view[intersect_mask] = 1
                intersection_memmap[sl] = current_int_view

            total_px_p = np.count_nonzero(mask_p)
            
            # Bi-directional Stat: Area of Primary inside Partner
            row[f'overlap_{extent}_with_{partner_name}_{unit}'] = overlap_px * unit_extent
            
            # Bi-directional Stat: % of this Primary contained by Partner
            row[f'pct_of_this_{primary_name}_inside_{partner_name}'] = \
                (overlap_px / total_px_p) * 100.0 if total_px_p > 0 else 0.0
            
            row[f'is_touching_{partner_name}'] = (overlap_px > 0)

            dom_id = 0
            if overlap_px > 0:
                overlap_ids, overlap_counts = np.unique(crop_r[intersect_mask], return_counts=True)
                for o_id, o_count in zip(overlap_ids, overlap_counts):
                    if o_id == 0: continue
                    ref_interactions.append({
                        'ref_label': o_id, 
                        'overlap_extent': o_count * unit_extent,
                        'primary_label': lbl
                    })
                
                valid = np.where(overlap_ids > 0)[0]
                if valid.size > 0:
                    dom_idx = valid[np.argmax(overlap_counts[valid])]
                    dom_id = overlap_ids[dom_idx]

            row[f'dominant_partner_id_{partner_name}'] = dom_id

        # --- Distance Logic (2D) ---
        if calculate_distance and dist_map is not None and indices is not None:
            dist_crop = dist_map[sl]
            if np.any(mask_p):
                min_dist = np.min(dist_crop[mask_p])
                row[f'dist_um_{partner_name}'] = min_dist
                
                local_mins = np.argwhere((dist_crop == min_dist) & mask_p)
                if local_mins.size > 0:
                    local_src = local_mins[0]
                    global_src = tuple(l_c + s.start for l_c, s in zip(local_src, sl))
                    
                    indexer = (slice(None),) + global_src
                    global_target = tuple(int(c) for c in indices[indexer])
                    
                    row[f'nearest_id_{partner_name}'] = reference_memmap[global_target]
                    
                    # Bridge coordinates for the connection lines in the
                    # viewer. One column per axis, so 3D carries a Z component
                    # and 2D does not -- there is no Z in a plane to report.
                    for _k, _ax in enumerate(axes):
                        row[f'src_{_ax}_{partner_name}'] = global_src[_k]
                        row[f'tgt_{_ax}_{partner_name}'] = global_target[_k]
                else:
                    row[f'nearest_id_{partner_name}'] = 0
            else:
                row[f'dist_um_{partner_name}'] = np.nan
                row[f'nearest_id_{partner_name}'] = 0

        primary_results.append(row)

    # 6. Post-Process Intersection (Unique Labeling)
    if calculate_overlap and intersection_memmap is not None:
        flush_print(f"  Unique labeling of {ndim}D overlap regions between "
                    f"{primary_name} and {partner_name}...")
        intersection_memmap.flush()
        # Chunk shapes preserved from each track: (4096, 4096) in 2D and
        # (64, 256, 256) in 3D. Memory-shape choices, not parameters.
        d_int = da.from_array(intersection_memmap,
                              chunks=(4096, 4096) if ndim == 2 else (64, 256, 256))
        labeled_int, _ = dask_image.ndmeasure.label(
            d_int, structure=adjacency_footprint(ndim)
        )
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
            overlap_extent_total=('overlap_extent', 'sum'),
            touching_count=('primary_label', 'nunique'),
            touching_ids=('primary_label', lambda x: list(x))
        ).reset_index()

        ref_stats = ref_stats.rename(columns={
            'ref_label': f'id_{partner_name}',
            'overlap_extent_total':
                f'total_{extent}_of_{primary_name}_inside_this_{partner_name}',
            'touching_count': f'count_of_unique_{primary_name}_touching_this_{partner_name}',
            'touching_ids': f'list_of_{primary_name}_ids_touching_this_{partner_name}'
        })

        partner_total = (ref_stats[f'id_{partner_name}'].map(ref_areas).fillna(0)
                         * unit_extent)
        ref_stats[f'total_{extent}_of_this_{partner_name}'] = partner_total

        ref_stats[f'pct_of_{partner_name}_occupied_by_{primary_name}'] = (
            ref_stats[f'total_{extent}_of_{primary_name}_inside_this_{partner_name}']
            / partner_total.replace(0, 1)
        ) * 100.0

        ref_df = ref_stats

    # 8. Pairwise Distance Calculation
    pairwise_df = calculate_pairwise_distances(
        primary_memmap, reference_memmap, spacing, partner_name
    )
    if not pairwise_df.empty:
        pairwise_out_path = os.path.join(output_dir, f"pairwise_distances_{partner_name}.csv")
        pairwise_df.to_csv(pairwise_out_path, index=False)

    # Cleanup
    if intersection_memmap is not None:
        intersection_memmap.flush()
        if hasattr(intersection_memmap, '_mmap') and intersection_memmap._mmap:
            intersection_memmap._mmap.close()

    del dist_map, indices, primary_memmap, reference_memmap
    gc.collect()

    return pd.DataFrame(primary_results), ref_df, intersection_path


# --------------------------------------------------------------------------
# 2D entry points
# --------------------------------------------------------------------------
def calculate_interaction_metrics_2d(primary_mask_path, reference_mask_path,
                                     output_dir, shape, spacing_yx, primary_name,
                                     partner_name, calculate_distance=True,
                                     calculate_overlap=True):
    """
    2D entry point, kept so existing callers keep working.

    `calculate_interaction_metrics` handles both ranks; this only translates the
    `spacing_yx` argument name. New code should call the rank-agnostic function.
    """
    return calculate_interaction_metrics(
        primary_mask_path=primary_mask_path,
        reference_mask_path=reference_mask_path,
        output_dir=output_dir,
        shape=shape,
        spacing=spacing_yx,
        primary_name=primary_name,
        partner_name=partner_name,
        calculate_distance=calculate_distance,
        calculate_overlap=calculate_overlap,
    )


def calculate_pairwise_distances_2d(primary_memmap, reference_memmap, spacing_yx,
                                    partner_name):
    """2D entry point; forwards to the rank-agnostic implementation."""
    return calculate_pairwise_distances(primary_memmap, reference_memmap,
                                        spacing_yx, partner_name)


def _extract_contours_2d(mask_memmap, labels, shape=None):
    """2D alias; the implementation is rank-agnostic."""
    return _extract_contours(mask_memmap, labels, shape)


def _inter_channel_dist_worker_2d(args):
    """2D alias; the implementation is rank-agnostic."""
    return _inter_channel_dist_worker(args)

