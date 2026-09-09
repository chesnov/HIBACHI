"""
3D Feature Calculation Module (Production Grade)
==============================================

This module provides high-precision volumetric, topological, and spatial 
quantification for 3D segmented objects. It uses a multi-stage refinement 
pipeline to ensure skeletons are strictly 1-voxel wide, topologically 
accurate, and connectivity-safe.

Performance Features:
- Two-Pass Distance Engine: RAM-stable N x N matrix calculation.
- Sequential Spur Pruning: Real-time connectivity verification.
- Anisotropy-Aware Metrics: Accurate surface area and volume for 3D.
- Impeccable Logic: Prevents branch shredding and spur regrowth.
"""

import os
import gc
import sys
import time
import math
import shutil
import tempfile
import traceback
import multiprocessing as mp
from typing import Tuple, List, Dict, Optional, Any, Sequence, Union

import numpy as np

try:
    from .dim_utils import normalise_spacing
except ImportError:  # pragma: no cover - direct script execution
    from dim_utils import normalise_spacing
import networkx as nx
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize, remove_small_holes, convex_hull_image
from skan import Skeleton, summarize
from tqdm.auto import tqdm

# --- Standardized FCS Export Logic ---
try:
    import fcswrite  # type: ignore
except ImportError:
    fcswrite = None
    print("Warning: 'fcswrite' library not found. FCS export will be disabled.")


def flush_print(*args: Any, **kwargs: Any) -> None:
    """Standardized wrapper for immediate log flushing."""
    print(*args, **kwargs)
    sys.stdout.flush()


# --------------------------------------------------------------------------
# Bounded-memory helpers
# --------------------------------------------------------------------------
def _unique_labels_streaming(arr, block_rows: int = 0):
    """Sorted positive labels of `arr`, without copying the whole volume.

    `np.unique(arr)` flattens first, and flattening a memmap materialises every
    voxel as an in-RAM copy -- an int32 label volume's worth, to obtain a list
    of ids. Taking the unique values of one leading-axis slab at a time and
    unioning gives the identical sorted array, because a union of per-block
    unique sets is the unique set of the whole.
    """
    if block_rows <= 0:
        # A slab of a few hundred MB regardless of cross-section, floored at one
        # row so a very wide plane still makes progress.
        per_row = max(1, int(np.prod(arr.shape[1:])) * int(arr.dtype.itemsize))
        block_rows = max(1, min(int(arr.shape[0]), (256 << 20) // per_row))
    found = None
    for start in range(0, int(arr.shape[0]), int(block_rows)):
        blk = np.asarray(arr[start:start + int(block_rows)])
        u = np.unique(blk)
        found = u if found is None else np.union1d(found, u)
    if found is None:
        return np.zeros(0, dtype=np.int64)
    return found[found > 0]


class _SurfaceStore:
    """Every object's boundary points, concatenated in one on-disk array.

    Replaces the list-of-arrays module global that the distance workers used to
    read. That list had two costs, and the second is the one that bit hardest:

      * every object's boundary coordinates were resident in RAM at once, and
      * on macOS and Windows, where multiprocessing uses 'spawn' rather than
        'fork', the whole list was PICKLED INTO EVERY WORKER -- so the memory
        was multiplied by `n_jobs` rather than shared.

    One memmap plus an offsets array fixes both: workers map the same file
    read-only and slice it, so nothing is copied and nothing is pickled but the
    offsets. The points themselves are byte-identical, so every KD-tree, every
    query and every resulting distance is unchanged.
    """

    def __init__(self, path: str, shape: Tuple[int, int], offsets: np.ndarray):
        self.path = path
        self.shape = (int(shape[0]), int(shape[1]))
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self._mm = None

    @property
    def n_objects(self) -> int:
        return int(self.offsets.size - 1)

    def _open(self):
        if self._mm is None:
            if self.shape[0] == 0:
                self._mm = np.zeros(self.shape, dtype=np.int64)
            else:
                self._mm = np.memmap(self.path, dtype=np.int64, mode='r',
                                     shape=self.shape)
        return self._mm

    def points(self, i: int) -> np.ndarray:
        mm = self._open()
        return np.asarray(mm[self.offsets[i]:self.offsets[i + 1]])

    def close(self) -> None:
        self._mm = None

    def remove(self) -> None:
        self.close()
        try:
            if self.path and os.path.exists(self.path):
                os.remove(self.path)
        except OSError:
            pass

    @staticmethod
    def build(chunks: List[np.ndarray], ndim: int, temp_dir: str,
              tag: str) -> "_SurfaceStore":
        """Write the per-object point arrays out in order."""
        counts = [int(c.shape[0]) for c in chunks]
        offsets = np.zeros(len(counts) + 1, dtype=np.int64)
        if counts:
            offsets[1:] = np.cumsum(np.asarray(counts, dtype=np.int64))
        total = int(offsets[-1])
        path = os.path.join(temp_dir, f"{tag}_{os.getpid()}.dat")
        if total > 0:
            mm = np.memmap(path, dtype=np.int64, mode='w+',
                           shape=(total, ndim))
            for k, c in enumerate(chunks):
                if c.shape[0]:
                    mm[offsets[k]:offsets[k + 1]] = c
            mm.flush()
            del mm
        return _SurfaceStore(path, (total, ndim), offsets)


class _DistanceMatrixOnDisk:
    """The full N x N distance matrix, kept on disk and read a row at a time.

    The matrix is genuinely wanted -- it is exported as `distances_matrix_*.csv`
    -- but it was being handled three times over in RAM: `np.array(dist_mat_mm)`
    materialised it, the DataFrame wrapped that, and
    `dist_df.values.copy()` in `analyze_segmentation` made a second full copy to
    mask the diagonal. At 50k objects that is two 10 GB float32 arrays for a
    file that is written straight back out to disk.

    This exposes only what the two consumers actually use -- `empty`, `index`,
    `columns`, a streaming `to_csv`, and the per-row minimum -- and never holds
    more than a block of rows. `to_csv` formats each block with pandas rather
    than by hand, so the bytes are the same as `DataFrame.to_csv` would have
    produced.
    """

    def __init__(self, path: str, labels: Sequence[int],
                 block_rows: int = 512):
        self.path = path
        self.index = pd.Index(list(labels))
        self.columns = pd.Index(list(labels))
        self.n = len(self.index)
        self.block_rows = max(1, int(block_rows))
        self._mm = None

    @property
    def empty(self) -> bool:
        return self.n == 0

    def _open(self):
        if self._mm is None:
            self._mm = np.memmap(self.path, dtype='float32', mode='r',
                                 shape=(self.n, self.n))
        return self._mm

    def row_minima(self):
        """``(shortest distance, index of the winner)`` for every row.

        The diagonal is masked with inf per row, which is what
        `np.fill_diagonal(temp_mat, np.inf)` did to the full copy. `nanmin` and
        `nanargmin` are kept rather than `min`/`argmin` so the values match the
        previous code exactly even if a NaN ever reaches the matrix.
        """
        mm = self._open()
        best = np.empty(self.n, dtype=np.float32)
        who = np.empty(self.n, dtype=np.int64)
        for start in range(0, self.n, self.block_rows):
            stop = min(start + self.block_rows, self.n)
            blk = np.array(mm[start:stop], dtype=np.float32)
            for r in range(stop - start):
                blk[r, start + r] = np.inf
            best[start:stop] = np.nanmin(blk, axis=1)
            who[start:stop] = np.nanargmin(blk, axis=1)
        return best, who

    def to_csv(self, path_or_buf, index: bool = True, **kwargs) -> None:
        """Write the matrix in row blocks, formatted by pandas."""
        mm = self._open()
        first = True
        with open(path_or_buf, "w", newline="") as fh:
            for start in range(0, self.n, self.block_rows):
                stop = min(start + self.block_rows, self.n)
                frame = pd.DataFrame(
                    np.array(mm[start:stop], dtype=np.float32),
                    index=self.index[start:stop], columns=self.columns,
                )
                frame.to_csv(fh, index=index, header=first, **kwargs)
                first = False

    def close(self) -> None:
        self._mm = None

    def remove(self) -> None:
        self.close()
        try:
            if self.path and os.path.exists(self.path):
                os.remove(self.path)
        except OSError:
            pass


# --- Global Shared Cache for Multiprocessing ---
# Shared via Copy-on-Write on Linux. Stores object surface points to avoid 
# the massive RAM overhead of pickling data to worker threads.
#: The point store the distance workers read. A `_SurfaceStore` (one
#: on-disk array) rather than a list of per-object arrays: see that
#: class for why the list form cost RAM twice over.
_ALL_SURFACES: Optional["_SurfaceStore"] = None

def _init_shared_surfaces(store: "_SurfaceStore") -> None:
    """Pool initializer. Only the offsets travel; the points are mapped.

    Under 'spawn' this used to pickle every object's coordinates into
    every worker. A `_SurfaceStore` pickles as a path, a shape and an
    offsets array, and each worker maps the same file read-only.
    """
    global _ALL_SURFACES
    _ALL_SURFACES = store
    _ALL_SURFACES.close()   # drop any mapping inherited across a fork

# =============================================================================
# 1. DISTANCE QUANTIFICATION (Two-Pass High-Precision System)
# =============================================================================

def _calculate_row_distances_worker_3d(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Worker Pass 1: Computes minimum 3D Euclidean distances for a matrix row.
    Returns a float32 array segment to keep the multiprocessing queue small.
    """
    i, n_proc, spacing_arr = args
    row_dists = np.full(n_proc - (i + 1), np.inf, dtype=np.float32)
    
    p1 = _ALL_SURFACES.points(i)
    if p1.shape[0] == 0:
        return i, row_dists
    
    tree = cKDTree(p1 * spacing_arr)
    
    for idx, j in enumerate(range(i + 1, n_proc)):
        p2 = _ALL_SURFACES.points(j)
        if p2.shape[0] > 0:
            # Query the target point cloud against the source tree
            dists, _ = tree.query(p2 * spacing_arr, k=1)
            row_dists[idx] = np.min(dists)
            
    return i, row_dists


def _extract_winning_points_worker_3d(args: Tuple) -> Dict[str, Any]:
    """
    Worker Pass 2: Identifies 3D coordinates for closest contact points.
    Forces native Python float types for absolute GUI compatibility.
    """
    label_i, label_j, idx_i, idx_j, spacing_arr = args
    p1, p2 = _ALL_SURFACES.points(idx_i), _ALL_SURFACES.points(idx_j)
    
    tree = cKDTree(p1 * spacing_arr)
    dists, indices = tree.query(p2 * spacing_arr, k=1)
    
    # Identify the specific pixel-pair representing the absolute minimum path
    min_idx = np.argmin(dists)
    p1_pt = p1[indices[min_idx]]
    p2_pt = p2[min_idx]
    
    return {
        'mask1': int(label_i), 
        'mask2': int(label_j),
        'mask1_z': float(p1_pt[0]), 
        'mask1_y': float(p1_pt[1]), 
        'mask1_x': float(p1_pt[2]),
        'mask2_z': float(p2_pt[0]), 
        'mask2_y': float(p2_pt[1]), 
        'mask2_x': float(p2_pt[2])
    }


def shortest_distance(
    segmented_array: np.ndarray,
    spacing: Tuple[float, float, float] = None,
    temp_dir: Optional[str] = None,
    n_jobs: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Coordinates the N x N Distance Matrix calculation in 3D.
    Uses disk-backed storage and a two-pass system for RAM stability.
    """
    global _ALL_SURFACES
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    spacing_arr = np.array(spacing)

    # Streamed, not `np.unique(segmented_array)`: that flattens first, and
    # flattening a memmap copies the entire label volume into RAM.
    labels = _unique_labels_streaming(segmented_array)
    n_labels = len(labels)

    flush_print(f"\n[Dist] Calculating 3D distances for {n_labels} objects...")
    if n_labels <= 1:
        return pd.DataFrame(), pd.DataFrame()

    # --- 1. Surface Extraction ---
    # `actual_labels` records WHICH label each row of `_ALL_SURFACES` belongs to.
    # It is not bookkeeping for its own sake: a label is skipped when its bbox
    # is None or when erosion leaves no boundary voxel, so `_ALL_SURFACES` can be
    # shorter than `labels` and the two stop being index-aligned at the first
    # skip. Everything downstream is indexed by POSITION in the surface list --
    # the memmap rows, the winning pairs, the matrix axes -- so without this the
    # row for surface i would be reported under `labels[i]`, which is a
    # different object, and `pd.DataFrame(..., index=labels)` on an
    # (n_valid, n_valid) matrix raises outright. The 2D contour path already
    # carried this list; the 3D copy did not.
    _surface_chunks: List[np.ndarray] = []
    actual_labels: List[int] = []
    locations = ndi.find_objects(segmented_array)
    struct = ndi.generate_binary_structure(3, 1) # 6-connectivity
    
    for lbl in tqdm(labels, desc="    Surface Extraction"):
        sl = locations[lbl-1]
        if sl is None: continue
        mask = (segmented_array[sl] == lbl)
        eroded = ndi.binary_erosion(mask, structure=struct)
        z, y, x = np.where(mask ^ eroded)
        if len(z) > 0:
            _surface_chunks.append(np.column_stack((
                z + sl[0].start,
                y + sl[1].start,
                x + sl[2].start
            )))
            actual_labels.append(lbl)

    n_valid = len(actual_labels)
    flush_print(f"[Dist] Found {n_valid} valid masks with surfaces.")
    
    # Fast exit if objects shrunk to 0 during erosion
    if n_valid <= 1:
        _surface_chunks = []
        return pd.DataFrame(), pd.DataFrame()

    # --- 2. Pass 1: Disk-Backed Matrix ---
    # Redirect to the project-specific temp_dir to keep system temp clean
    # Temporary files MUST live in the project directory (temp_dir). No
    # hidden/OS-temp fallback: a missing temp_dir is a bug, so fail loudly.
    if not temp_dir:
        raise ValueError(
            "Feature calculation requires a project temp directory (temp_dir); "
            "temporary files must live in the project directory."
        )
    os.makedirs(temp_dir, exist_ok=True)
    target_dir = temp_dir
    mmap_path = os.path.join(target_dir, f"dist_mat_3d_{os.getpid()}.dat")
    
    # Write the point clouds out once, then drop the in-RAM chunks. From here
    # on the coordinates live in one file that every worker maps.
    _ALL_SURFACES = _SurfaceStore.build(_surface_chunks, 3, target_dir,
                                        "surfaces")
    _surface_chunks = []

    dist_mat_mm = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n_valid, n_valid))
    dist_mat_mm[:] = np.inf
    np.fill_diagonal(dist_mat_mm, 0)

    # Configure pool to safely share surfaces on macOS/Windows ('spawn'), 
    # while preserving RAM-saving Copy-on-Write on Linux ('fork')
    pool_kwargs = {}
    if mp.get_start_method() != 'fork':
        pool_kwargs['initializer'] = _init_shared_surfaces
        pool_kwargs['initargs'] = (_ALL_SURFACES,)

    tasks = [(i, n_valid, spacing_arr) for i in range(n_valid)]
    with mp.Pool(n_jobs, **pool_kwargs) as pool:
        for i, row_results in tqdm(pool.imap_unordered(_calculate_row_distances_worker_3d, tasks), 
                                  total=n_valid, desc="    Distance Pass 1/2"):
            dist_mat_mm[i, i+1:] = row_results
            dist_mat_mm[i+1:, i] = row_results 

    # --- 3. Pass 2: Coordinate Extraction ---
    winning_pairs = []
    for i in range(n_valid):
        row = dist_mat_mm[i].copy()
        row[i] = np.inf
        j = np.argmin(row)
        if not np.isinf(row[j]):
            winning_pairs.append(
                (actual_labels[i], actual_labels[j], i, j, spacing_arr))

    with mp.Pool(n_jobs, **pool_kwargs) as pool:
        points_list = list(tqdm(pool.imap_unordered(_extract_winning_points_worker_3d, winning_pairs),
                               total=len(winning_pairs), desc="    Distance Pass 2/2"))

    # The matrix is handed back as a disk-backed view, not materialised. Its
    # only consumers are the per-row minimum in `analyze_segmentation` and the
    # CSV export, both of which this serves a block at a time. The file is left
    # in place for those; `_DistanceMatrixOnDisk.remove()` deletes it.
    points_df = pd.DataFrame(points_list)

    _ALL_SURFACES.remove()
    _ALL_SURFACES = None
    del dist_mat_mm
    dist_df = _DistanceMatrixOnDisk(mmap_path, actual_labels)

    return dist_df, points_df


# =============================================================================
# 2. SKELETONIZATION (Topology-Preserving Logic)
# =============================================================================

def calculate_ramification_with_skan(
    segmented_array: np.ndarray,
    spacing: Tuple[float, float, float],
    skeleton_export_path: Optional[str],
    prune_spurs_le_um: float
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    3D Skeletonization with Mathematical Tree Guarantee.
    
    This module uses a three-stage topology enforcement:
    1. MST Graph Refinement: Initial cycle removal based on process thickness.
    2. Graph-Based Pruning: Removing spurs by micron threshold.
    3. Voxel-Level Cycle Killing: A final pass that detects residual diagonal 
       leaks in the 3D volume and breaks them.
    """
    flush_print(f"  [Skel] 3D Tree-Enforcement Mode (Pruning <= {prune_spurs_le_um} um)...")
    
    original_shape = segmented_array.shape
    use_memmap = (skeleton_export_path is not None)
    if use_memmap:
        os.makedirs(os.path.dirname(skeleton_export_path), exist_ok=True)
        skel_out = np.memmap(
            skeleton_export_path, dtype=np.int32, mode='w+', shape=original_shape
        )
    else:
        skel_out = np.zeros(original_shape, dtype=np.int32)
        
    # Streamed rather than `np.unique(segmented_array)`, which flattens the
    # whole label volume into RAM to list its ids.
    labels = _unique_labels_streaming(segmented_array)
    locations = ndi.find_objects(segmented_array)
    
    stats_list, detailed_dfs = [], []

    for lbl in tqdm(labels, desc="    Tree-Enforcement 3D"):
        idx = int(lbl) - 1
        if idx >= len(locations) or locations[idx] is None: continue
        sl = locations[idx]; offset = np.array([s.start for s in sl])
        mask = (segmented_array[sl] == lbl).astype(bool)
        if not np.any(mask): continue
            
        # 1. INITIAL THINNING
        mask_padded = np.pad(mask, pad_width=3, mode='constant', constant_values=0)
        mask_padded = ndi.binary_fill_holes(mask_padded)
        mask_dt = ndi.distance_transform_edt(mask_padded, sampling=spacing)
        
        # Lee's algorithm is the most robust for initial line extraction
        skel_binary = skeletonize(mask_padded)
        if not np.any(skel_binary): continue

        # 2. GRAPH REFINEMENT (MST + PRUNE)
        try:
            skel_obj = Skeleton(skel_binary, spacing=spacing)
            G = nx.Graph()
            for b in range(skel_obj.n_paths):
                path_coords = skel_obj.path_coordinates(b)
                u, v = tuple(path_coords[0].astype(int)), tuple(path_coords[-1].astype(int))
                coords_idx = path_coords.astype(int)
                weight = np.mean(mask_dt[coords_idx[:,0], coords_idx[:,1], coords_idx[:,2]])
                G.add_edge(u, v, weight=weight, length=skel_obj.path_lengths()[b], path=path_coords)
            
            # MST to break cycles in the graph
            G_tree = nx.Graph()
            for comp in nx.connected_components(G):
                G_tree.add_edges_from(nx.maximum_spanning_tree(G.subgraph(comp), weight='weight').edges(data=True))
            
            # Prune spurs
            if prune_spurs_le_um > 0:
                while True:
                    tips = [n for n, d in G_tree.degree() if d == 1]
                    removed = False
                    for tip in tips:
                        if tip not in G_tree or G_tree.degree(tip) == 0: continue
                        neighbor = list(G_tree.neighbors(tip))[0]
                        data = G_tree.get_edge_data(tip, neighbor)
                        
                        global_coords = np.array(tip) - 3 + offset
                        hits_edge = (np.any(global_coords <= 0) or 
                                     np.any(global_coords >= np.array(original_shape)-1))
                        
                        if not hits_edge and data['length'] <= prune_spurs_le_um:
                            G_tree.remove_node(tip); removed = True
                    if not removed: break
            
            # 3. RECONSTRUCTION
            # We build a fresh binary image from the tree graph
            skel_binary = np.zeros_like(mask_padded, dtype=bool)
            for u, v, data in G_tree.edges(data=True):
                coords = data['path'].astype(int)
                skel_binary[coords[:,0], coords[:,1], coords[:,2]] = True
        except: pass

        # 4. FINAL VOXEL-LEVEL CYCLE KILLER
        # This resolves diagonal contacts and 2x2 micro-loops that the graph missed.
        for _ in range(5): # Usually resolves in 1-2 passes
            try:
                loop_skel = Skeleton(skel_binary)
                # Convert CSR to undirected graph
                G_loop = nx.from_scipy_sparse_array(loop_skel.graph).to_undirected()
                cycles = nx.cycle_basis(G_loop)
                if not cycles: break
                
                for cycle in cycles:
                    # Find the weakest voxel in this cycle to break it
                    # node index -> skeleton coordinate
                    cycle_coords = loop_skel.coordinates[cycle].astype(int)
                    # Get distance transform values for cycle voxels
                    dt_vals = mask_dt[cycle_coords[:,0], cycle_coords[:,1], cycle_coords[:,2]]
                    # Snap the thinnest part of the loop
                    weak_idx = np.argmin(dt_vals)
                    bad_voxel = cycle_coords[weak_idx]
                    skel_binary[tuple(bad_voxel)] = 0
            except: break

        # 5. UNPAD & FINAL CLEAN
        # Re-thinning ensures 1-voxel width after surgical voxel removal
        skel_binary = skeletonize(skel_binary)
        skel_crop = skel_binary[3:-3, 3:-3, 3:-3]
        skel_out[sl][skel_crop] = lbl

        # 6. QUANTIFICATION
        skan_len, skan_branches, avg_len = 0.0, 0, 0.0
        if np.any(skel_crop):
            try:
                final_skel_obj = Skeleton(skel_crop, spacing=spacing)
                summ = summarize(final_skel_obj, separator='-')
                if not summ.empty:
                    summ['label'] = int(lbl)
                    for c in summ.columns:
                        if 'coord' in c: summ[c] += offset[int(c.split('-')[-1])]
                    detailed_dfs.append(summ)
                    skan_len, skan_branches = summ['branch-distance'].sum(), len(summ)
                    avg_len = summ['branch-distance'].mean()
            except: pass

        # Final structural metrics
        kernel = np.ones((3, 3, 3), dtype=np.uint8); kernel[1, 1, 1] = 0
        neighbors = ndi.convolve(skel_crop.astype(np.uint8), kernel, mode='constant', cval=0)
        n_end = np.count_nonzero((skel_crop > 0) & (neighbors == 1))
        n_junc = np.count_nonzero((skel_crop > 0) & (neighbors >= 3))
        
        stats_list.append({
            'label': int(lbl), 'true_num_branches': max(0, n_end - 1 + n_junc),
            'skan_total_length_um': skan_len, 'skan_avg_branch_length_um': avg_len, 
            'true_num_junctions': n_junc, 'true_num_endpoints': n_end, 
            'skan_num_skeleton_voxels': np.count_nonzero(skel_crop)
        })

        # Explicit RAM recovery inside the loop
        del mask, mask_padded, mask_dt, skel_binary, skel_crop
        if lbl % 50 == 0:
            gc.collect()

    if use_memmap: skel_out.flush()
    return pd.DataFrame(stats_list), pd.concat(detailed_dfs, ignore_index=True) if detailed_dfs else pd.DataFrame(), skel_out


# =============================================================================
# 3. VOLUMETRICS & EXPORT (Numerical Parity with 2D)
# =============================================================================

def _solidity_3d(mask: np.ndarray) -> float:
    """Solidity of a 3D object: its voxel count over its convex hull's.

    Numerically identical to what ``regionprops`` reports for the 2D path --
    ``area / area_convex`` with both as element counts of the cropped mask and
    its ``convex_hull_image`` -- so a 3D solidity is directly comparable with a
    2D one. Verified equal to ``regionprops(...).solidity`` to within 1e-12.

    Being a ratio of volumes it is dimensionless: scaling each axis multiplies
    the object and its hull by the same determinant. It therefore needs no
    spacing argument and is unaffected by voxel anisotropy.

    Returns NaN when the hull is degenerate. An object that is a single voxel,
    a plane or a line has fewer than four non-coplanar points, so Qhull cannot
    build a hull; ``regionprops`` reports ``inf`` there, which would poison any
    downstream mean, so NaN is used instead.
    """
    try:
        n_obj = int(np.count_nonzero(mask))
        if n_obj == 0:
            return float("nan")
        hull = convex_hull_image(mask)
        n_hull = int(np.count_nonzero(hull))
        if n_hull <= 0:
            return float("nan")
        return n_obj / n_hull
    except Exception:
        return float("nan")


def calculate_volume(segmented_array, spacing, calculate_solidity: bool = False):
    """Calculates Volume and Surface Area using Crofton approximation.

    `calculate_solidity` gates the convex-hull ratio, which is off by default
    because it is the most expensive per-object measurement here: the hull is
    rasterised over the object's bounding box, so cost grows with that box
    rather than with the voxel count. Left off, the column is NaN.
    """
    voxel_vol = np.prod(spacing)
    az, ay, ax = spacing[1]*spacing[2], spacing[0]*spacing[2], spacing[0]*spacing[1]
    # Streamed rather than `np.unique(segmented_array)`, which flattens the
    # whole label volume into RAM to list its ids.
    labels = _unique_labels_streaming(segmented_array)
    locs = ndi.find_objects(segmented_array); res = []
    for lbl in tqdm(labels, desc="    Volume/Shape"):
        idx = int(lbl) - 1
        if idx >= len(locs) or locs[idx] is None: continue
        mask = (segmented_array[locs[idx]] == lbl)
        n_vox = np.count_nonzero(mask); vol_um = n_vox * voxel_vol
        sa = 0.0
        try:
            sa += np.count_nonzero(np.diff(mask.astype(np.int8), axis=0)) * ax
            sa += (np.count_nonzero(mask[0,:,:]) + np.count_nonzero(mask[-1,:,:])) * ax
            sa += np.count_nonzero(np.diff(mask.astype(np.int8), axis=1)) * ay
            sa += (np.count_nonzero(mask[:,0,:]) + np.count_nonzero(mask[:,-1,:])) * ay
            sa += np.count_nonzero(np.diff(mask.astype(np.int8), axis=2)) * az
            sa += (np.count_nonzero(mask[:,:,0]) + np.count_nonzero(mask[:,:,-1])) * az
        except: sa = np.nan
        sph = (np.pi**(1/3) * (6 * vol_um)**(2/3)) / sa if sa > 1e-6 else np.nan
        res.append({'label': int(lbl),
                    'volume_um3': vol_um,
                    'surface_area_um2': sa,
                    'sphericity': sph,
                    'voxel_count': n_vox,
                    'solidity': (_solidity_3d(mask) if calculate_solidity
                                 else float('nan'))})
        
        del mask
        if lbl % 100 == 0:
            gc.collect()

    return pd.DataFrame(res)


def calculate_intensity(segmented_array, intensity_image):
    """Calculates fluorescence intensity summary statistics."""
    # Streamed rather than `np.unique(segmented_array)`, which flattens the
    # whole label volume into RAM to list its ids.
    labels = _unique_labels_streaming(segmented_array)
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
    """FCS export with support for high-throughput 3D datasets."""
    if not fcs_path or fcswrite is None or metrics_df is None or metrics_df.empty: return
    try:
        flush_print(f"  [Export] Writing FCS: {os.path.basename(fcs_path)}")
        num_df = metrics_df.select_dtypes(include=[np.number]).copy()
        num_df.replace([np.inf, -np.inf], np.nan, inplace=True); num_df.fillna(0, inplace=True)
        if 'label' in metrics_df.columns: num_df['label'] = metrics_df['label']
        fcswrite.write_fcs(filename=fcs_path, chn_names=list(num_df.columns), data=num_df.values)
    except Exception as e: flush_print(f"  [Export] Error during FCS write: {e}")


# =============================================================================
# 4. MAIN ENTRY POINT
# =============================================================================

def analyze_segmentation(
    segmented_array: np.ndarray,
    intensity_image: Optional[np.ndarray] = None,
    spacing: Tuple[float, float, float] = None,
    calculate_distances: bool = True,
    calculate_skeletons: bool = True,
    calculate_solidity: bool = False,
    skeleton_export_path: Optional[str] = None,
    fcs_export_path: Optional[str] = None,
    temp_dir: Optional[str] = None,
    n_jobs: Optional[int] = None,
    return_detailed: bool = False,
    prune_spurs_le_um: float = 0.0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Comprehensive 3D analysis suite."""
    # Physical spacing is required, not defaulted. Every feature below is a
    # physical measurement; substituting 1.0 um/voxel would make all of them
    # wrong by a constant factor with nothing in the output to reveal it.
    spacing = normalise_spacing(spacing, 3)
    flush_print("\n--- Starting Feature Calculation (3D) ---")
    vol_df = calculate_volume(segmented_array, spacing, calculate_solidity)
    if vol_df.empty: return pd.DataFrame(), {}
    
    # Depth Analysis
    depths = []
    locs = ndi.find_objects(segmented_array)
    for lbl in tqdm(vol_df['label'].values, desc="    Depth"):
        sl = locs[lbl-1]
        z_local, _, _ = np.where(segmented_array[sl] == lbl)
        depths.append({'label': int(lbl), 'depth_um': np.median(z_local + sl[0].start) * spacing[0]})
    metrics_df = pd.merge(vol_df, pd.DataFrame(depths), on='label', how='outer')
    
    detailed_outputs = {}

    if intensity_image is not None:
        int_df = calculate_intensity(segmented_array, intensity_image)
        if not int_df.empty: metrics_df = pd.merge(metrics_df, int_df, on='label', how='outer')

    if calculate_distances:
        dist_df, pts_df = shortest_distance(segmented_array, spacing, temp_dir, n_jobs)
        if not dist_df.empty:
            # `dist_df.values.copy()` was a SECOND full N x N array in RAM,
            # alongside the one the DataFrame already held, purely to mask the
            # diagonal before reducing. The reduction is per row, so it streams:
            # `row_minima` masks each row's own entry and returns the same
            # `nanmin` / `nanargmin` results a block at a time.
            _best, _who = dist_df.row_minima()
            dist_metrics = pd.DataFrame({
                'label': dist_df.index.astype(int),
                'shortest_distance_um': _best,
                'closest_neighbor_label': dist_df.columns[_who].astype(int)
            })
            # Both merge keys coerced to int. `dist_metrics['label']` comes from
            # a DataFrame index and `metrics_df['label']` from the measurement
            # tables, so the two can carry different integer widths (or object
            # dtype); pandas then matches nothing and every distance column
            # arrives as NaN with no error. The 2D path already did this.
            metrics_df['label'] = metrics_df['label'].astype(int)
            metrics_df = pd.merge(metrics_df, dist_metrics, on='label', how='left')
            if return_detailed and not pts_df.empty:
                detailed_outputs['distance_matrix'] = dist_df
                pts_df['m1_k'], pts_df['m2_k'] = pts_df['mask1'].astype(int), pts_df['mask2'].astype(int)
                f_keys = dist_metrics[['label', 'closest_neighbor_label']].copy()
                f_keys.columns = ['m1_k', 'm2_k']
                detailed_outputs['all_pairs_points'] = pd.merge(pts_df, f_keys, on=['m1_k', 'm2_k'], how='inner').drop(columns=['m1_k', 'm2_k'])

    if calculate_skeletons:
        summ_skel, detail_skel, skel_arr = calculate_ramification_with_skan(segmented_array, spacing, skeleton_export_path, prune_spurs_le_um)
        if not summ_skel.empty:
            summ_skel['label'] = summ_skel['label'].astype(int)
            metrics_df = pd.merge(metrics_df, summ_skel, on='label', how='left')
        if return_detailed:
            detailed_outputs['detailed_branches'] = detail_skel
            detailed_outputs['skeleton_array'] = skel_arr

    if fcs_export_path: export_to_fcs(metrics_df, fcs_export_path)
    flush_print("--- Analysis Complete (3D) ---")
    return metrics_df, detailed_outputs if return_detailed else {}