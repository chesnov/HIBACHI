import os
import gc
import math
import shutil
import tempfile
import traceback
from typing import Tuple, Optional, Any, Union, List

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
from skimage.morphology import disk, binary_closing  # type: ignore
from tqdm import tqdm

# Dask Config
DASK_SCHEDULER = 'threads'


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
    Re-calculates connected components (Labels) and filters small fragments in 3D.
    Solves the issue where disjoint fragments share an ID and bypass size checks.

    Args:
        labels_memmap: Memory-mapped 3D array of labels (modified in-place).
        min_size_voxels: Minimum size in voxels to retain.
    """
    if min_size_voxels <= 0:
        return

    print(f"  [Refine] Re-labeling and filtering fragments < {min_size_voxels} voxels...")

    # 1. Setup Dask Array
    # Chunking Z is critical for 3D labeling efficiency
    chunk_size = (64, 256, 256)

    # We treat the input as a binary mask first
    d_seg = da.from_array(labels_memmap, chunks=chunk_size)
    binary_mask = (d_seg > 0)

    # 2. Re-Label Connected Components
    # This separates fragments into unique IDs
    structure = generate_binary_structure(3, 1)
    labeled_dask, num_features_dask = dask_image.ndmeasure.label(
        binary_mask, structure=structure
    )

    # Trigger compute to get total count
    num_features = num_features_dask.compute()

    if num_features == 0:
        print("    No objects remaining.")
        return

    # 3. Calculate Sizes (Histogram)
    print(f"    Analyzing {num_features} fragments...")
    # bins=num_features+1 because range is [0, num_features]
    counts, _ = da.histogram(
        labeled_dask, bins=num_features + 1, range=[0, num_features]
    )
    counts_val = counts.compute()

    # 4. Identify Keepers
    # counts_val[i] is size of label i. Index 0 is background.
    valid_labels_mask = (counts_val >= min_size_voxels)
    valid_labels_mask[0] = False  # Ensure background remains 0

    # Get the IDs to keep
    ids_to_keep = np.where(valid_labels_mask)[0]

    print(f"    Fragments: {num_features}. Kept: {len(ids_to_keep)}. "
          f"Removed: {num_features - len(ids_to_keep)}.")

    # 5. Apply Filter and Save
    # We effectively map: ID -> ID (if valid) else 0.
    # isin is memory efficient in dask for this
    mask_keep = da.isin(labeled_dask, ids_to_keep)

    # Apply to the NEW labels so they are contiguous and clean
    final_dask = da.where(mask_keep, labeled_dask, 0)

    print("    Writing filtered result...")
    with ProgressBar(dt=2):
        da.store(final_dask, labels_memmap, lock=True, scheduler=DASK_SCHEDULER)

    labels_memmap.flush()


def apply_clamped_z_erosion(
    labels_path: str,
    shape: Tuple[int, ...],
    iterations: int
) -> None:
    """
    Performs 'Clamped' Z-Erosion to correct for Z-anisotropy smearing.
    Erodes in Z, but prevents disjointing objects by restoring connectivity
    if a column is completely eroded.

    Args:
        labels_path: Path to the .dat memmap file.
        shape: Shape of the array.
        iterations: Number of Z-erosion iterations.
    """
    if iterations <= 0:
        return
    print(f"  [Z-Correct] Clamped Erosion (iter={iterations})...")

    fp = np.memmap(labels_path, dtype=np.int32, mode='r+', shape=shape)

    chunk_size = 64
    overlap = iterations + 2
    total_z = shape[0]

    # Structure for Z-only erosion
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

        # "Clamp" Logic: Don't lose the object entirely in Z
        footprint_orig = np.max(mask, axis=0)
        footprint_erod = np.max(eroded_mask, axis=0)
        lost_map = footprint_orig & (~footprint_erod)

        if np.any(lost_map):
            # Restore the middle pixel for lost columns
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


def generate_tight_hull_stack(
    volume: np.ndarray,
    cell_mask: np.ndarray,
    temp_dir: str,
    downsample_factor: int = 4
) -> np.memmap:
    """
    Generates a tight hull around the tissue using slice-by-slice 2D morphology
    on downsampled data. Used to define the tissue boundary.

    Args:
        volume: Original 3D intensity volume.
        cell_mask: Current segmentation mask.
        temp_dir: Directory to store the temporary hull memmap.
        downsample_factor: Factor to speed up computation.

    Returns:
        np.memmap: Boolean 3D mask of the hull.
    """
    print(f"\n  [HullGen] Generating Tight Hull (Downsample {downsample_factor}x)...")
    original_shape = volume.shape
    hull_path = os.path.join(temp_dir, 'tight_hull.dat')
    hull_memmap = np.memmap(hull_path, dtype=bool, mode='w+', shape=original_shape)

    # Global threshold estimation
    pixels = volume[::50, ::50, ::50].ravel()
    valid = pixels[pixels > 0]
    thresh = (threshold_otsu(valid) * 0.5) if valid.size > 0 else 0

    small_h = original_shape[1] // downsample_factor
    small_w = original_shape[2] // downsample_factor
    struct_elem = disk(10)

    for z in tqdm(range(original_shape[0]), desc="    Hull Computation"):
        vol_slice = volume[z]
        seg_slice = cell_mask[z]
        if not np.any(vol_slice > thresh) and not np.any(seg_slice):
            continue

        mask_raw = (vol_slice > thresh) | (seg_slice > 0)
        
        # Downsample -> Close -> Fill -> Upsample
        small_mask = resize(
            mask_raw, (small_h, small_w),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(bool)
        
        closed = binary_closing(small_mask, footprint=struct_elem)
        filled = binary_fill_holes(closed)
        
        final_mask = resize(
            filled, (original_shape[1], original_shape[2]),
            order=0, preserve_range=True, anti_aliasing=False
        ).astype(bool)
        
        hull_memmap[z] = final_mask

    return hull_memmap


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
    Prevents deleting real somas at the edge while removing noise.

    Args:
        labels_memmap: 3D Labels (modified in-place).
        volume_memmap: 3D Intensity.
        hull_memmap: 3D Hull mask.
        spacing: Voxel spacing.
        distance_threshold: Physical distance from hull to trim.
        global_brightness_cutoff: Intensity threshold for core protection.
    """
    print("  [EdgeTrim] Trimming with Core Protection...")

    total_z = labels_memmap.shape[0]

    # 1. Distance Map (Chunked Calculation)
    print("    Calculating global distance map...")
    dist_map_path = os.path.join(
        os.path.dirname(labels_memmap.filename), 'dist_map.dat'
    )
    dist_memmap = np.memmap(
        dist_map_path, dtype=np.float32, mode='w+', shape=labels_memmap.shape
    )

    rows, cols = labels_memmap.shape[1], labels_memmap.shape[2]
    slice_bytes = rows * cols * 4
    target_ram = 1024**3
    edt_chunk_z = max(10, min(100, int(target_ram / slice_bytes)))
    edt_overlap = int(30 / spacing[0]) + 5

    for z in tqdm(range(0, total_z, edt_chunk_z), desc="    Distance Transform"):
        z0 = max(0, z - edt_overlap)
        z1 = min(total_z, z + edt_chunk_z + edt_overlap)
        hull_chunk = hull_memmap[z0:z1]
        
        # Calculate distance inside the hull (distance to background)
        dt_chunk = distance_transform_edt(hull_chunk, sampling=spacing)
        
        w_start = z - z0
        w_end = w_start + min(z + edt_chunk_z, total_z) - z
        dist_memmap[z:min(z + edt_chunk_z, total_z)] = \
            dt_chunk[w_start:w_end].astype(np.float32)
            
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

        # A. Identify Cores (Bright & Labeled)
        core_mask = (lbl_chunk > 0) & (vol_chunk > global_brightness_cutoff)

        # B. Create Protection Zone (Dilated Cores)
        if np.any(core_mask):
            protected_mask = binary_dilation(
                core_mask, structure=struct_protect, iterations=protection_iter
            )
        else:
            protected_mask = np.zeros_like(core_mask, dtype=bool)

        # C. Kill Logic: Labeled AND Near Edge AND Not Protected
        # Near Edge = dist_chunk < threshold
        to_delete = (lbl_chunk > 0) & \
                    (dist_chunk < distance_threshold) & \
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
    hull_boundary_thickness_phys: float,
    edge_trim_distance_threshold: float,
    brightness_cutoff_factor: float,
    segmentation_threshold: float,
    min_size_voxels: int,
    smoothing_iterations: int = 1,
    heal_iterations: int = 1,
    edge_distance_chunk_size_z: int = 32,
    z_erosion_iterations: int = 0,
    post_smoothing_iter: int = 0
) -> Tuple[Optional[str], Optional[str], Optional[np.ndarray]]:
    """
    Main Entry Point for Step 2.
    Applies Z-Correction, Hull Generation, Edge Trimming, and Size Filtering.

    Returns:
        Tuple: (output_path, temp_dir, hull_boundary_mask_for_viz)
    """
    print(f"\n--- Applying Hull Generation and Edge Trimming ---")
    original_shape = original_volume.shape
    workflow_temp_dir = tempfile.mkdtemp(prefix="hull_trim_")
    final_output_temp_dir = tempfile.mkdtemp(prefix="trimmed_final_")
    trimmed_labels_memmap = None
    hull_memmap = None
    hull_boundary_for_return = None

    try:
        # 1. Output Setup
        final_output_path = os.path.join(final_output_temp_dir, 'trimmed_labels.dat')
        shutil.copyfile(raw_labels_path, final_output_path)
        trimmed_labels_memmap = np.memmap(
            final_output_path, dtype=np.int32, mode='r+', shape=original_shape
        )

        # 2. Z-Correction (Clamped)
        if z_erosion_iterations > 0:
            apply_clamped_z_erosion(
                final_output_path, original_shape, z_erosion_iterations
            )

        # 3. Edge Trimming
        if edge_trim_distance_threshold > 0:
            # A. Generate Hull
            hull_memmap = generate_tight_hull_stack(
                original_volume, trimmed_labels_memmap,
                workflow_temp_dir, downsample_factor=4
            )

            # B. Recalculate Threshold (Robustness)
            print("  [Filter] Checking reference intensity...")
            # Sample volume to check if threshold needs adjustment
            raw_pixels = original_volume[::50, ::50, ::50].ravel()
            valid = raw_pixels[raw_pixels > 0]
            raw_otsu = threshold_otsu(valid) if valid.size > 0 else 0

            vol_max = np.max(raw_pixels) if raw_pixels.size > 0 else 0
            
            # Heuristic fallback if provided threshold seems low for bright image
            ref_thresh = raw_otsu if (vol_max > 5 and segmentation_threshold < 2.0) else segmentation_threshold
            global_brightness_cutoff = ref_thresh * brightness_cutoff_factor

            print(f"  Edge Trim Active: Dist<{edge_trim_distance_threshold}um, "
                  f"CoreBrightness>{int(global_brightness_cutoff)}")

            # C. Trim
            trim_edges_with_core_protection(
                labels_memmap=trimmed_labels_memmap,
                volume_memmap=original_volume,
                hull_memmap=hull_memmap,
                spacing=spacing,
                distance_threshold=edge_trim_distance_threshold,
                global_brightness_cutoff=global_brightness_cutoff
            )

            # D. Generate Boundary for Viz
            eroded_hull = np.zeros_like(hull_memmap, dtype=bool)
            struct = np.ones((3, 3, 3), dtype=bool)
            
            # Process in chunks to save RAM
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

        # 4. FINAL CLEANUP: Re-label and Filter Size
        # This fixes fragmentation caused by Z-Erosion and Trimming
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
        gc.collect()
        if workflow_temp_dir and os.path.exists(workflow_temp_dir):
            try:
                shutil.rmtree(workflow_temp_dir, ignore_errors=True)
            except Exception:
                pass