import os
import gc
import math
import shutil
import tempfile
import traceback
from typing import Tuple, Optional, Any, Union

import numpy as np
import dask.array as da
import dask_image.ndmeasure
from dask.diagnostics import ProgressBar
from scipy import ndimage
from scipy.ndimage import (
    distance_transform_edt,
    binary_fill_holes,
    binary_dilation,
    generate_binary_structure
)
from skimage.filters import threshold_otsu  # type: ignore
from skimage.transform import resize  # type: ignore
from skimage.morphology import disk, binary_closing, binary_dilation as sk_dilation, remove_small_objects  # type: ignore
from tqdm import tqdm

# Dask Config
DASK_SCHEDULER = 'threads'


def _get_safe_temp_dir(base_path: str, suffix: str = "") -> str:
    """Creates a temporary directory strictly inside the provided base_path."""
    # Temporary files MUST live in the project directory. No hidden/OS-temp
    # fallback: a missing base_path is a bug, so fail loudly.
    if not base_path:
        raise ValueError(
            "_get_safe_temp_dir requires a project temp directory (temp_root_path); "
            "temporary files must live in the project directory."
        )
    os.makedirs(base_path, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"step2_{suffix}_", dir=base_path)


def _safe_close_memmap(memmap_obj: Any) -> None:
    """Safely closes a numpy memmap object to release file locks."""
    if memmap_obj is None:
        return
    try:
        if hasattr(memmap_obj, 'flush'):
            memmap_obj.flush()
        if hasattr(memmap_obj, '_mmap') and memmap_obj._mmap is not None:
            memmap_obj._mmap.close()
    except Exception:
        pass


def relabel_and_filter_fragments(
    labels_memmap: np.memmap,
    min_size_voxels: int
) -> None:
    """
    Refined 3D fragment filter.
    Uses dask-image to perform GLOBAL connected component labeling and size filtering
    in an out-of-core manner (scalable to RAM).
    """
    if min_size_voxels <= 0:
        return

    print(f"  [Refine] Global Labeling & Size Filtering (Min: {min_size_voxels} voxels)...")

    # 1. Setup Dask Array (Out-of-Core)
    # Use reasonable chunks to fit in laptop RAM (e.g. ~100MB chunks)
    # (128, 256, 256) of int32 is roughly 32MB per chunk, very safe for laptops.
    chunk_size = (128, 256, 256)
    d_seg = da.from_array(labels_memmap, chunks=chunk_size)
    
    # 2. Binarize (Virtual)
    binary_mask = (d_seg > 0)

    # 3. Global Connected Components
    # dask_image handles the stitching of chunks to resolve global labels without
    # loading the full brain into RAM.
    print("    Resolving global connectivity (this may take a while)...")
    # Full (26-)connectivity, matching step 1's labeling, so relabeling does not
    # re-split objects merged across a diagonal contact or a thin link bridge.
    structure_3d = generate_binary_structure(3, 3)
    
    # This builds the graph but doesn't compute yet
    labeled_dask, num_features_dask = dask_image.ndmeasure.label(
        binary_mask, structure=structure_3d
    )

    # Compute total features to set up histogram
    # This triggers the first pass of the graph
    num_features = num_features_dask.compute()
    
    if num_features == 0:
        print("    No objects found.")
        labels_memmap[:] = 0
        labels_memmap.flush()
        return

    print(f"    Found {num_features} unique objects. Analyzing sizes...")

    # 4. Global Size Histogram
    # dask.histogram computes volume of every label index globally
    # Integer-aligned bins so every label id (including the highest) is counted.
    counts, _ = da.histogram(
        labeled_dask, bins=num_features + 1, range=[-0.5, num_features + 0.5]
    )
    counts_val = counts.compute()

    # 5. Identify Valid Labels
    # Create a boolean mask of IDs to keep based on total global volume
    valid_mask = (counts_val >= min_size_voxels)
    valid_mask[0] = False # Background is 0
    
    # Convert to array of IDs for isin()
    ids_to_keep = np.where(valid_mask)[0]
    
    removed_count = num_features - len(ids_to_keep)
    print(f"    Removing {removed_count} small artifacts. Keeping {len(ids_to_keep)} objects.")

    if len(ids_to_keep) == 0:
         labels_memmap[:] = 0
         labels_memmap.flush()
         return

    # 6. Apply Filter & Write Back
    # dask.isin creates a boolean mask where pixels belong to valid IDs
    # da.where preserves the Label ID if valid, else 0
    mask_keep = da.isin(labeled_dask, ids_to_keep)
    final_dask = da.where(mask_keep, labeled_dask, 0)

    print("    Writing filtered segmentation to disk...")
    with ProgressBar(dt=5):
        # lock=True ensures thread safety when writing to the memmap
        da.store(
            final_dask.astype(np.int32), 
            labels_memmap, 
            lock=True, 
            scheduler=DASK_SCHEDULER
        )

    labels_memmap.flush()
    gc.collect()
    print("    Refinement complete.")


def apply_clamped_z_erosion(
    labels_path: str,
    shape: Tuple[int, ...],
    iterations: int
) -> None:
    """
    Performs 'Clamped' Z-Erosion to correct for Z-anisotropy smearing.
    Erodes in Z, but prevents disjointing objects by restoring connectivity
    if a column is completely eroded.
    """
    if iterations <= 0:
        return
    print(f"  [Z-Correct] Clamped Erosion (iter={iterations})...")

    fp = np.memmap(labels_path, dtype=np.int32, mode='r+', shape=shape)

    chunk_size = 64
    overlap = iterations + 2
    total_z = shape[0]

    structure = np.zeros((3, 1, 1), dtype=bool)
    structure[:, 0, 0] = 1

    for start_z in tqdm(range(0, total_z, chunk_size), desc="    Z-Correction"):
        end_z = min(start_z + chunk_size, total_z)
        r_start = max(0, start_z - overlap)
        r_end = min(total_z, end_z + overlap)

        chunk_data = fp[r_start:r_end].copy()
        mask = (chunk_data > 0)

        if not np.any(mask):
            continue

        eroded_mask = ndimage.binary_erosion(
            mask, structure=structure, iterations=iterations
        )

        footprint_orig = np.max(mask, axis=0)
        footprint_erod = np.max(eroded_mask, axis=0)
        lost_map = footprint_orig & (~footprint_erod)

        if np.any(lost_map):
            top_idx = np.argmax(mask, axis=0)
            mask_flipped = mask[::-1, :, :]
            bottom_idx_flipped = np.argmax(mask_flipped, axis=0)
            bottom_idx = mask.shape[0] - 1 - bottom_idx_flipped
            mid_idx = (top_idx + bottom_idx) // 2

            ys, xs = np.where(lost_map)
            zs = mid_idx[ys, xs]
            eroded_mask[zs, ys, xs] = True

        chunk_data[~eroded_mask] = 0

        w_start_rel = start_z - r_start
        w_end_rel = w_start_rel + (end_z - start_z)
        fp[start_z:end_z] = chunk_data[w_start_rel:w_end_rel]

    fp.flush()
    _safe_close_memmap(fp)


def _find_largest_hull_component_slice_graph(hull_memmap: np.memmap) -> None:
    """
    Finds and keeps only the largest connected 3D component of a boolean hull
    memmap using a Z-slice connectivity graph.

    Replaces the full-volume dask_image CCA which hangs on data that exceeds
    RAM, as dask-image's label algorithm must stitch chunk boundaries
    sequentially and scales poorly with both volume size and chunk count.

    This implementation is exact (no approximation) and requires at most two
    adjacent Z-slices in memory at any time — O(Z) I/O regardless of XY size.

    Algorithm:
      Pass 1 — Iterate Z slices.  For each slice run scipy.ndimage.label (fast
               single-thread C code on one 2D array).  Feed overlapping
               component pairs from adjacent slices into a Union-Find
               structure, accumulating voxel counts at each root.
      Identify the root with the highest total voxel count.
      Pass 2 — Re-derive 2D labels per slice (identical assignment because
               ndimage.label is deterministic / raster-scan).  Zero out any
               component whose Union-Find root is not the largest.

    Logical equivalence with 2D pipeline:
      generate_tight_hull_2d already uses scipy.ndimage.label on the
      downsampled 2D hull — i.e. a single-slice degenerate case of this exact
      algorithm.  Both pipelines keep the largest connected hull component and
      remove stray artifacts; only the dimensionality and working resolution
      differ.
    """
    total_z = hull_memmap.shape[0]

    # 8-connectivity in 2D: more permissive than 4-connectivity, avoids
    # artificially splitting diagonal hull features into separate components.
    structure_2d = np.ones((3, 3), dtype=bool)

    # ── Union-Find ────────────────────────────────────────────────────────────
    # Nodes are (z, local_label_id) tuples — globally unique per component.
    parent: dict = {}   # node → parent node (itself when root)
    uf_size: dict = {}  # root node → accumulated voxel count (valid at roots only)

    def _find(x):
        """Path-compressing find with halving."""
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        curr = x
        while curr != root:
            nxt = parent[curr]
            parent[curr] = root
            curr = nxt
        return root

    def _union(x, y):
        """Union by size: smaller root is merged into larger."""
        rx, ry = _find(x), _find(y)
        if rx == ry:
            return
        if uf_size.get(rx, 0) < uf_size.get(ry, 0):
            rx, ry = ry, rx
        parent[ry] = rx
        uf_size[rx] = uf_size.get(rx, 0) + uf_size.get(ry, 0)

    def _register(node, count):
        if node not in parent:
            parent[node] = node
            uf_size[node] = count

    # ── Pass 1: Build inter-slice connectivity graph ──────────────────────────
    print("    [SliceGraph] Pass 1: building inter-slice connectivity...")
    prev_labeled = None

    for z in tqdm(range(total_z), desc="    SliceGraph Pass 1"):
        hull_slice = hull_memmap[z]
        if not np.any(hull_slice):
            prev_labeled = None
            continue

        labeled, n = ndimage.label(hull_slice, structure=structure_2d)

        for lid in range(1, n + 1):
            _register((z, lid), int(np.count_nonzero(labeled == lid)))

        if prev_labeled is not None:
            # Pixels where both slices are foreground share a Z-face — the
            # corresponding component IDs must be connected in 3D.
            overlap = (prev_labeled > 0) & (labeled > 0)
            if np.any(overlap):
                for pid, cid in set(zip(prev_labeled[overlap].tolist(),
                                        labeled[overlap].tolist())):
                    _union((z - 1, pid), (z, cid))

        prev_labeled = labeled

    if not parent:
        return  # Hull is entirely empty

    # ── Find the largest 3D root ──────────────────────────────────────────────
    distinct_roots = {_find(n) for n in parent}

    if len(distinct_roots) == 1:
        print("    [SliceGraph] Hull is already a single component — nothing to remove.")
        return

    # uf_size is only reliable at roots: non-roots had their counts absorbed
    # into the root during union and their uf_size entries are stale.
    root_sizes = {r: uf_size.get(r, 0) for r in distinct_roots}
    largest_root = max(root_sizes, key=root_sizes.__getitem__)
    n_components = len(distinct_roots)
    print(f"    [SliceGraph] {n_components} 3D components found. "
          f"Largest: {root_sizes[largest_root]} voxels. "
          f"Removing {n_components - 1} disjoint artifact(s).")

    # ── Pass 2: Zero out non-largest components ───────────────────────────────
    print("    [SliceGraph] Pass 2: removing disjoint artifacts...")
    removed = 0

    for z in tqdm(range(total_z), desc="    SliceGraph Pass 2"):
        hull_slice = hull_memmap[z].copy()
        if not np.any(hull_slice):
            continue

        labeled, n = ndimage.label(hull_slice, structure=structure_2d)
        modified = False

        for lid in range(1, n + 1):
            if _find((z, lid)) != largest_root:
                mask = (labeled == lid)
                hull_slice[mask] = False
                removed += int(np.count_nonzero(mask))
                modified = True

        if modified:
            hull_memmap[z] = hull_slice

    print(f"    [SliceGraph] Done. Removed {removed} artifact voxels.")


def generate_tight_hull_stack(
    volume: np.ndarray,
    cell_mask: np.ndarray,
    temp_dir: str,
    hull_closing_radius: int = 10,
    downsample_factor: int = 4,
    otsu_scale_factor: float = 0.8
) -> np.memmap:
    """
    Generates a solid 'Shrink-Wrap' hull around the tissue block.
    Uses Log-Space Otsu for detection and Morphological Closing for shape.
    """
    print(f"\n  [HullGen] Generating Concave Hull (Radius {hull_closing_radius}, DS {downsample_factor}x)...")
    original_shape = volume.shape
    hull_path = os.path.join(temp_dir, 'tight_hull.dat')
    hull_memmap = np.memmap(hull_path, dtype=bool, mode='w+', shape=original_shape)

    # 1. Robust Global Threshold Calculation (Log-Space Otsu)
    print("    Calculating Tissue Threshold (Log-Space Otsu)...")
    sample_stride = 50
    pixels = volume[::sample_stride, ::sample_stride, ::sample_stride].ravel()
    valid_pixels = pixels[pixels > 0]
    
    if valid_pixels.size > 0:
        log_pixels = np.log1p(valid_pixels.astype(np.float32))
        try:
            log_thresh = threshold_otsu(log_pixels)
        except:
            log_thresh = 0
        tissue_thresh = (np.expm1(log_thresh)) * otsu_scale_factor
    else:
        tissue_thresh = 0
        
    print(f"    Tissue Threshold: {tissue_thresh:.2f}")

    small_h = original_shape[1] // downsample_factor
    small_w = original_shape[2] // downsample_factor
    
    # Structuring element for closing (Concavity Control)
    struct_elem = disk(hull_closing_radius)

    for z in tqdm(range(original_shape[0]), desc="    Hull Computation"):
        vol_slice = volume[z]
        seg_slice = cell_mask[z]
        
        if not np.any(vol_slice > tissue_thresh) and not np.any(seg_slice):
            continue

        mask_raw = (vol_slice > tissue_thresh) | (seg_slice > 0)
        
        # Downsample
        small_mask = resize(
            mask_raw, (small_h, small_w),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(bool)
        
        # Filter Noise BEFORE hull generation
        small_mask = remove_small_objects(small_mask, min_size=100)
        
        if not np.any(small_mask):
            continue

        # Morphological Operations (Bridging & Filling)
        # Dilation connects sparse cells
        bridged = sk_dilation(small_mask, footprint=disk(3))
        # Closing bridges gaps defined by radius
        closed = binary_closing(bridged, footprint=struct_elem)
        # Fill holes solves the "Swiss Cheese" problem
        filled = binary_fill_holes(closed)
        
        # Upsample
        final_mask = resize(
            filled, (original_shape[1], original_shape[2]),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(bool)
        
        hull_memmap[z] = final_mask
    
    hull_memmap.flush()

    # 3. Filter: Keep ONLY Largest 3D Component
    # Essential for removing floating artifacts (tile corners, dust).
    # Uses Z-slice graph instead of full-volume dask CCA — see docstring of
    # _find_largest_hull_component_slice_graph for rationale.
    print("    Filtering disjoint hull artifacts (Keeping Largest Component)...")
    _find_largest_hull_component_slice_graph(hull_memmap)

    hull_memmap.flush()
    return hull_memmap


def _trim_zero_data_edges_3d(
    labels_memmap: np.memmap,
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    distance_threshold: float
) -> None:
    """
    Removes artifacts at the boundary of Missing Tiles (Pixel Value 0) in 3D.
    Detects 'True Zero' regions and hard-deletes segmentations near them.
    """
    print("  [ZeroTrim] Removing Missing Tile Artifacts (3D)...")
    total_z = labels_memmap.shape[0]
    
    margin = int(distance_threshold / spacing[0]) + 5
    chunk_size = 32
    deleted_voxels = 0
    
    for start_z in tqdm(range(0, total_z, chunk_size), desc="    Zero-Edge Trim"):
        end_z = min(start_z + chunk_size, total_z)
        r_start = max(0, start_z - margin)
        r_end = min(total_z, end_z + margin)
        
        vol_chunk = volume[r_start:r_end]
        lbl_chunk = labels_memmap[r_start:r_end]
        
        # 1. Identify 'True Zero' (with epsilon)
        is_zero = (vol_chunk < 1e-4)
        if not np.any(is_zero):
            continue
        
        # 2. EDT from Void
        dist_from_void = distance_transform_edt(~is_zero, sampling=spacing)
        
        # 3. Hard Delete
        rel_start = start_z - r_start
        rel_end = rel_start + (end_z - start_z)
        
        center_dist = dist_from_void[rel_start:rel_end]
        center_lbl = lbl_chunk[rel_start:rel_end]
        
        mask_distance = (center_dist < distance_threshold)
        
        # Ensure immediate boundary is caught if threshold is small
        if distance_threshold > 0:
            avg_px = np.mean(spacing)
            if distance_threshold < avg_px:
                mask_distance |= (center_dist <= (avg_px * 1.5))

        to_delete = (center_lbl > 0) & mask_distance
        
        count = np.sum(to_delete)
        if count > 0:
            center_lbl[to_delete] = 0
            labels_memmap[start_z:end_z] = center_lbl
            deleted_voxels += count

    labels_memmap.flush()
    print(f"    Deleted {deleted_voxels} pixels at tile boundaries.")


def trim_edges_with_core_protection(
    labels_memmap: np.memmap,
    volume_memmap: np.ndarray,
    hull_memmap: np.memmap,
    spacing: Tuple[float, float, float],
    distance_threshold: float,
    global_brightness_cutoff: float
) -> None:
    """
    Trims objects near the tissue boundary unless they are bright (Cores).
    """
    print("  [EdgeTrim] Trimming with Core Protection...")

    total_z = labels_memmap.shape[0]
    spacing_yx = (spacing[1], spacing[2])  # YX only — see note below

    print("    Calculating distance map (2D per slice)...")
    dist_map_path = os.path.join(
        os.path.dirname(labels_memmap.filename), 'dist_map.dat'
    )
    dist_memmap = np.memmap(
        dist_map_path, dtype=np.float32, mode='w+', shape=labels_memmap.shape
    )

    # Compute EDT slice-by-slice in 2D rather than as a chunked 3D operation.
    #
    # Rationale: this function trims lateral XY edge artifacts — cells near
    # the tissue boundary in the XY plane.  For those cells the nearest hull
    # boundary point is always in the same Z slice, so 2D and 3D EDT produce
    # identical distances.  Cells near the top/bottom Z surfaces are handled
    # separately by apply_clamped_z_erosion and are not the target here.
    #
    # The old approach computed a 3D EDT over overlapping Z chunks.  The overlap
    # was int(30 / spacing[0]) + 5, which for fine Z spacings (e.g. 0.5 µm)
    # gave 65 slices of overlap.  For a 20038×20038 XY image (1.6 GB per slice)
    # this forced each EDT call to allocate >200 GB — causing the hang.
    #
    # 2D per-slice EDT uses at most two slices of memory at any time (~3 GB
    # for a 20038×20038 image), matching the memory profile of the 2D pipeline
    # which calls distance_transform_edt(hull_mask, sampling=spacing) on the
    # full 2D hull in one shot.
    for z in tqdm(range(total_z), desc="    Distance Transform"):
        dist_memmap[z] = distance_transform_edt(
            hull_memmap[z], sampling=spacing_yx
        ).astype(np.float32)

    dist_memmap.flush()

    # 2. Protection & Filter
    protection_radius_um = 6.0
    protection_iter = max(2, min(10, int(protection_radius_um / spacing[0])))

    print(f"    Applying Protection (Dilate Bright Cores {protection_iter}x)...")

    scan_chunk_size = 64
    scan_overlap = protection_iter + 2
    deleted_voxels = 0
    struct_protect = generate_binary_structure(3, 1)

    for z in tqdm(range(0, total_z, scan_chunk_size), desc="    Processing"):
        end_z = min(z + scan_chunk_size, total_z)
        r_start = max(0, z - scan_overlap)
        r_end = min(total_z, end_z + scan_overlap)

        lbl_chunk = labels_memmap[r_start:r_end]
        vol_chunk = volume_memmap[r_start:r_end]
        dist_chunk = dist_memmap[r_start:r_end]

        if not np.any(lbl_chunk):
            continue

        core_mask = (lbl_chunk > 0) & (vol_chunk > global_brightness_cutoff)

        if np.any(core_mask):
            protected_mask = binary_dilation(
                core_mask, structure=struct_protect, iterations=protection_iter
            )
        else:
            protected_mask = np.zeros_like(core_mask, dtype=bool)

        # Incorporate absolute Z-distance from the top and bottom of the stack.
        # The 2D per-slice EDT ignores the Z-faces. By calculating physical Z-distance 
        # and taking the minimum against the 2D XY distance, we mathematically restore 
        # the exact 3D distance to the bounding box without crashing the RAM.
        z_indices = np.arange(r_start, r_end)
        z_dist_top = z_indices * spacing[0]
        z_dist_bot = (total_z - 1 - z_indices) * spacing[0]
        z_dist = np.minimum(z_dist_top, z_dist_bot)
        
        # Broadcast the 1D Z-distance to match the 3D chunk (Z, Y, X)
        z_dist = z_dist[:, None, None]
        
        # The true distance is whichever is closer: the XY edge or the Z caps
        effective_dist = np.minimum(dist_chunk, z_dist)

        to_delete = (lbl_chunk > 0) & \
                    (effective_dist < distance_threshold) & \
                    (~protected_mask)

        w_start = z - r_start
        w_end = w_start + (end_z - z)
        center_delete = to_delete[w_start:w_end]
        center_lbls = labels_memmap[z:end_z]

        count = np.count_nonzero(center_delete)
        if count > 0:
            deleted_voxels += count
            center_lbls[center_delete] = 0
            labels_memmap[z:end_z] = center_lbls

    labels_memmap.flush()
    print(f"    Deleted {deleted_voxels} artifact voxels.")

    _safe_close_memmap(dist_memmap)
    if os.path.exists(dist_map_path):
        os.remove(dist_map_path)


def apply_hull_trimming(
    raw_labels_path: str,
    original_volume: np.ndarray,
    spacing: Tuple[float, float, float],
    edge_trim_distance_threshold: float,
    brightness_cutoff_factor: float,
    segmentation_threshold: float,
    min_size_voxels: int,
    hull_closing_radius: int = 10,
    z_erosion_iterations: int = 0,
    otsu_scale_factor: float = 0.8,
    *,
    temp_root_path: str,
) -> Tuple[Optional[str], Optional[str], Optional[np.ndarray]]:
    """
    Main Entry Point for Step 2.
    """
    print(f"\n--- Applying Hull Generation and Edge Trimming ---")
    original_shape = original_volume.shape
    
    workflow_temp_dir = _get_safe_temp_dir(temp_root_path, "hull_trim")
    final_output_temp_dir = _get_safe_temp_dir(temp_root_path, "trimmed_final")
    
    trimmed_labels_memmap = None
    hull_memmap = None
    hull_boundary_for_return = None

    print(f"  Hull Closing Radius: {hull_closing_radius}")

    try:
        # 1. Output Setup
        final_output_path = os.path.join(final_output_temp_dir, 'trimmed_labels.dat')
        shutil.copyfile(raw_labels_path, final_output_path)
        trimmed_labels_memmap = np.memmap(
            final_output_path, dtype=np.int32, mode='r+', shape=original_shape
        )

        # 2. Z-Correction
        if z_erosion_iterations > 0:
            apply_clamped_z_erosion(
                final_output_path, original_shape, z_erosion_iterations
            )

        # 3. Edge Trimming
        if edge_trim_distance_threshold > 0:
            
            # A. Zero Edge Trim
            _trim_zero_data_edges_3d(
                trimmed_labels_memmap, original_volume, spacing,
                edge_trim_distance_threshold
            )
            
            # B. Generate Hull
            hull_memmap = generate_tight_hull_stack(
                original_volume, trimmed_labels_memmap,
                workflow_temp_dir, 
                hull_closing_radius=hull_closing_radius, 
                downsample_factor=4,
                otsu_scale_factor=otsu_scale_factor
            )

            # C. Threshold Recalc
            print("  [Filter] Checking reference intensity...")
            raw_pixels = original_volume[::50, ::50, ::50].ravel()
            valid = raw_pixels[raw_pixels > 0]
            
            raw_otsu = threshold_otsu(valid) if valid.size > 0 else 0
            vol_max = np.max(raw_pixels) if raw_pixels.size > 0 else 0
            
            ref_thresh = raw_otsu if (vol_max > 5 and segmentation_threshold < 2.0) else segmentation_threshold
            global_brightness_cutoff = ref_thresh * brightness_cutoff_factor

            print(f"  Edge Trim Active: Dist<{edge_trim_distance_threshold}um, "
                  f"CoreBrightness>{int(global_brightness_cutoff)}")

            # D. Trim
            trim_edges_with_core_protection(
                labels_memmap=trimmed_labels_memmap,
                volume_memmap=original_volume,
                hull_memmap=hull_memmap,
                spacing=spacing,
                distance_threshold=edge_trim_distance_threshold,
                global_brightness_cutoff=global_brightness_cutoff
            )

            # E. Generate Boundary for Viz
            eroded_hull = np.zeros_like(hull_memmap, dtype=bool)
            struct = np.ones((3, 3, 3), dtype=bool)
            
            for z in range(0, original_shape[0], 32):
                end_z = min(z + 32, original_shape[0])
                r0, r1 = max(0, z - 1), min(original_shape[0], end_z + 1)
                h_c = hull_memmap[r0:r1]
                e_c = ndimage.binary_erosion(h_c, structure=struct, iterations=1)
                eroded_hull[z:end_z] = e_c[(z - r0):(z - r0) + (end_z - z)]
            
            hull_boundary_for_return = (hull_memmap ^ eroded_hull)

        else:
            print("  Edge Trim Disabled (Dist=0).")
            hull_memmap = None
            hull_boundary_for_return = np.zeros(original_shape, dtype=bool)

        # 4. FINAL CLEANUP
        if min_size_voxels > 0:
            relabel_and_filter_fragments(trimmed_labels_memmap, min_size_voxels)

        trimmed_labels_memmap = _safe_close_memmap(trimmed_labels_memmap)
        hull_memmap = _safe_close_memmap(hull_memmap)

        return final_output_path, final_output_temp_dir, hull_boundary_for_return

    except Exception as e:
        print(f"\n!!! ERROR during Hull Trimming Workflow: {e} !!!")
        traceback.print_exc()
        if final_output_temp_dir and os.path.exists(final_output_temp_dir):
            shutil.rmtree(final_output_temp_dir, ignore_errors=True)
        return None, None, None
    finally:
        if 'trimmed_labels_memmap' in locals():
            trimmed_labels_memmap = _safe_close_memmap(trimmed_labels_memmap)
        if 'hull_memmap' in locals():
            hull_memmap = _safe_close_memmap(hull_memmap)
        # Targeted deletion of large 3D arrays
        for var in ['hull_boundary_for_return', 'eroded_hull', 'core_mask', 'protected_mask']:
            if var in locals():
                del locals()[var]
        gc.collect()
        if workflow_temp_dir and os.path.exists(workflow_temp_dir):
            try:
                shutil.rmtree(workflow_temp_dir, ignore_errors=True)
            except Exception:
                pass