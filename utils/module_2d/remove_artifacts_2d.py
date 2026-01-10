import os
import gc
import math
import shutil
import tempfile
import traceback
from typing import Tuple, Optional, Any, Union

import numpy as np
import psutil
from scipy import ndimage
from scipy.ndimage import (
    distance_transform_edt,
    binary_dilation,
    generate_binary_structure,
    binary_fill_holes
)
from skimage.filters import threshold_otsu  # type: ignore
from skimage.transform import resize  # type: ignore
from skimage.morphology import disk, binary_closing, binary_dilation as sk_dilation, remove_small_objects  # type: ignore

import dask.array as da
import dask_image.ndmeasure
from dask.diagnostics import ProgressBar
import dask_image

def _get_safe_temp_dir(base_path: Optional[str] = None, suffix: str = "") -> str:
    """
    Creates a temporary directory.
    If base_path is provided, creates 'hibachi_scratch' inside it.
    Otherwise, creates it in the current working directory.
    """
    if base_path and os.path.isdir(base_path):
        scratch_root = os.path.join(base_path, "hibachi_scratch")
    else:
        scratch_root = os.path.abspath("hibachi_scratch")
    
    os.makedirs(scratch_root, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"step2_2d_{suffix}_", dir=scratch_root)


def _safe_close_memmap(memmap_obj: Any) -> None:
    """Safely closes a numpy memmap object."""
    if memmap_obj is None:
        return
    try:
        if hasattr(memmap_obj, 'flush'):
            memmap_obj.flush()
        if hasattr(memmap_obj, '_mmap') and memmap_obj._mmap is not None:
            memmap_obj._mmap.close()
    except Exception:
        pass


def relabel_and_filter_fragments_2d(
    labels_memmap: np.memmap,
    min_size_pixels: int
) -> None:
    """
    Re-calculates connected components and filters small fragments in 2D.
    Matches the 3D logic in 'remove_artifacts.py' exactly.

    This function identifies global connected components across the entire 
    2D image, calculates their areas, and removes those below the threshold.
    It does NOT perform morphological thinning; it is a pure size filter.

    Args:
        labels_memmap: Memory-mapped 2D array of labels (modified in-place).
        min_size_pixels: Minimum size in pixels to retain.
    """
    if min_size_pixels <= 0:
        return

    print(f"  [Refine] Re-labeling and filtering fragments < {min_size_pixels} pixels...")

    # 1. Setup Dask Array
    # Logic Parity: Matches the chunked Dask setup in the 3D version
    chunk_size = (2048, 2048)
    d_seg = da.from_array(labels_memmap, chunks=chunk_size)
    binary_mask = (d_seg > 0)

    # 2. Re-Label Connected Components
    # Logic Parity: Matches the use of dask_image.ndmeasure.label
    # 2D 4-connectivity corresponds to 3D 6-connectivity (rank 1)
    structure = generate_binary_structure(2, 1)
    labeled_dask, num_features_dask = dask_image.ndmeasure.label(
        binary_mask, structure=structure
    )

    # Trigger compute to get total count (Matches 3D logic)
    num_features = num_features_dask.compute()

    if num_features == 0:
        print("    No objects remaining.")
        labels_memmap[:] = 0
        labels_memmap.flush()
        return

    # 3. Calculate Sizes (Histogram)
    # Logic Parity: Matches the use of da.histogram for area calculation
    print(f"    Analyzing {num_features} fragments...")
    counts, _ = da.histogram(
        labeled_dask, bins=num_features + 1, range=[0, num_features]
    )
    counts_val = counts.compute()

    # 4. Identify Keepers
    # Logic Parity: Identify IDs that meet the area requirement
    valid_labels_mask = (counts_val >= min_size_pixels)
    valid_labels_mask[0] = False  # Ensure background remains 0
    ids_to_keep = np.where(valid_labels_mask)[0]

    print(f"    Fragments: {num_features}. Kept: {len(ids_to_keep)}. "
          f"Removed: {num_features - len(ids_to_keep)}.")

    # 5. Apply Filter and Save
    # Logic Parity: Matches the use of da.isin and da.where
    mask_keep = da.isin(labeled_dask, ids_to_keep)
    final_dask = da.where(mask_keep, labeled_dask, 0)

    print("    Writing filtered result...")
    with ProgressBar(dt=2):
        # DASK_SCHEDULER is assumed to be defined globally as in your 3D file
        da.store(final_dask.astype(np.int32), labels_memmap, lock=True, scheduler='threads')

    labels_memmap.flush()


def generate_tight_hull_2d(
    image: np.ndarray,
    cell_mask: np.ndarray,
    hull_closing_radius: int = 10,
    downsample_factor: int = 4
) -> np.ndarray:
    """
    Generates a solid 'Shrink-Wrap' hull around the tissue slice.
    Uses Log-Space Otsu and Morphological Closing (Concave Hull).
    """
    print(f"  [HullGen] Generating Concave Hull (Radius {hull_closing_radius}, DS {downsample_factor}x)...")
    
    # 1. Robust Global Threshold Calculation (Log-Space Otsu)
    valid = image[image > 0]
    if valid.size > 0:
        # Sample if huge
        if valid.size > 200000:
            valid = np.random.choice(valid, 200000, replace=False)
            
        log_pixels = np.log1p(valid.astype(np.float32))
        try:
            log_thresh = threshold_otsu(log_pixels)
        except:
            log_thresh = 0
        # Use 0.8 factor to match 3D logic
        tissue_thresh = (np.expm1(log_thresh)) * 0.8
    else:
        tissue_thresh = 0

    print(f"    Tissue Threshold: {tissue_thresh:.2f}")

    h, w = image.shape
    small_h = h // downsample_factor
    small_w = w // downsample_factor
    
    mask_raw = (image > tissue_thresh) | (cell_mask > 0)
    
    if not np.any(mask_raw):
        return np.zeros_like(image, dtype=bool)

    # Downsample
    small_mask = resize(
        mask_raw, (small_h, small_w),
        order=0, preserve_range=True, anti_aliasing=False
    ).astype(bool)

    # Filter Noise BEFORE hull generation
    small_mask = remove_small_objects(small_mask, min_size=100)

    if not np.any(small_mask):
        return np.zeros_like(image, dtype=bool)

    # Morphological Operations
    # 1. Bridge sparse cells (Dilation)
    bridged = sk_dilation(small_mask, footprint=disk(3))
    
    # 2. Define Hull Shape (Closing)
    struct_elem = disk(hull_closing_radius)
    closed = binary_closing(bridged, footprint=struct_elem)
    
    # 3. Filter: Keep ONLY Largest Component
    labeled_hull, num_features = ndimage.label(closed)
    if num_features > 1:
        counts = np.bincount(labeled_hull.ravel())
        counts[0] = 0 # Ignore background
        largest_label = np.argmax(counts)
        print(f"    Keeping largest tissue component (Label {largest_label}). Removing {num_features-1} artifacts.")
        closed = (labeled_hull == largest_label)

    # 4. Fill Holes
    filled = binary_fill_holes(closed)

    # Upsample
    final_mask = resize(
        filled, (h, w),
        order=0, preserve_range=True, anti_aliasing=False
    ).astype(bool)

    return final_mask


def trim_edges_with_core_protection_2d(
    labels_memmap: np.memmap,
    image: np.ndarray,
    hull_mask: np.ndarray,
    spacing: Tuple[float, float],
    distance_threshold: float,
    global_brightness_cutoff: float
) -> None:
    """
    Trims edges based on distance from hull, protecting bright cores.
    """
    print("  [EdgeTrim] Trimming with Core Protection (2D)...")

    # 1. Distance Map (from background/hull edge)
    print("    Calculating distance map...")
    dist_map = distance_transform_edt(hull_mask, sampling=spacing)

    # 2. Protection (Bright Cores)
    protection_radius_um = 6.0
    avg_spacing = np.mean(spacing)
    protection_iter = max(2, min(10, int(protection_radius_um / avg_spacing)))
    
    print(f"    Applying Protection (Dilate Bright Cores {protection_iter}x)...")
    
    mask_labels = labels_memmap[:] > 0
    mask_bright = image > global_brightness_cutoff
    core_mask = mask_labels & mask_bright
    
    if np.any(core_mask):
        struct = generate_binary_structure(2, 1)
        protected_mask = binary_dilation(
            core_mask, structure=struct, iterations=protection_iter
        )
    else:
        protected_mask = np.zeros_like(core_mask)

    # 3. Kill Logic
    to_delete = (mask_labels) & (dist_map < distance_threshold) & (~protected_mask)
    
    count = np.sum(to_delete)
    if count > 0:
        labels_memmap[to_delete] = 0
        labels_memmap.flush()
        print(f"    Deleted {count} artifact pixels.")
    else:
        print("    No artifact pixels found to delete.")


def _trim_zero_data_edges_2d(
    labels_memmap: np.memmap,
    image: np.ndarray,
    spacing: Tuple[float, float],
    distance_threshold: float
) -> None:
    """
    Specifically removes artifacts at the boundary of Missing Tiles (Pixel Value 0).
    """
    print("  [ZeroTrim] Removing Missing Tile Artifacts...")
    
    # 1. Identify 'True Zero' regions (Missing Tiles)
    is_zero = (image < 1e-4)
    
    if not np.any(is_zero):
        print("    No zero-value regions found.")
        return

    # 2. Calculate distance FROM the void
    dist_from_void = distance_transform_edt(~is_zero, sampling=spacing)
    
    # 3. Hard Delete
    mask_distance = (dist_from_void < distance_threshold)
    
    if distance_threshold > 0:
        avg_pixel_size = np.mean(spacing)
        if distance_threshold < avg_pixel_size:
            # Ensure immediate boundary is caught even if thresh is small
            mask_distance |= (dist_from_void <= (avg_pixel_size * 1.5))
    
    to_delete = (labels_memmap[:] > 0) & mask_distance
    
    count = np.sum(to_delete)
    if count > 0:
        labels_memmap[to_delete] = 0
        labels_memmap.flush()
        print(f"    Deleted {count} pixels at tile boundaries.")
    else:
        print("    No tile boundary artifacts found within threshold.")


def apply_hull_trimming_2d(
    raw_labels_path: str,
    original_image: np.ndarray,
    spacing: Union[Tuple[float, float], Tuple[float, float, float]],
    segmentation_threshold: float,
    hull_boundary_thickness_phys: float,
    edge_trim_distance_threshold: float,
    brightness_cutoff_factor: float,
    min_size_pixels: int,
    smoothing_iterations: int = 1,
    heal_iterations: int = 1,
    temp_root_path: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[np.ndarray]]:
    """
    Main Entry Point for Step 2 (2D).
    Applies Zero-Edge Trimming, Hull Generation, Edge Trimming, and Size Filtering.

    Args:
        smoothing_iterations: Mapped to hull_closing_radius (1->10, 2->15, etc).
    """
    print(f"\n--- Applying Hull Generation and Edge Trimming (2D) ---")
    
    try:
        if len(spacing) == 3:
            spacing_2d = tuple(float(s) for s in spacing[1:])
        else:
            spacing_2d = tuple(float(s) for s in spacing)
    except:
        spacing_2d = (1.0, 1.0)

    original_shape = original_image.shape
    
    # Safe temp dir
    final_output_temp_dir = _get_safe_temp_dir(temp_root_path, "trimmed_final_2d")
    
    trimmed_labels_memmap = None
    hull_boundary_for_return = None

    # Map iterations to radius
    hull_closing_radius = (max(1, smoothing_iterations) * 5) + 5
    print(f"  Hull Closing Radius: {hull_closing_radius}")

    try:
        # 1. Output Setup
        final_output_path = os.path.join(final_output_temp_dir, 'trimmed_labels.dat')
        shutil.copyfile(raw_labels_path, final_output_path)
        trimmed_labels_memmap = np.memmap(
            final_output_path, dtype=np.int32, mode='r+', shape=original_shape
        )

        # 2. Edge Trimming
        if edge_trim_distance_threshold > 0:
            
            # A. ZERO EDGE TRIM (Hard)
            _trim_zero_data_edges_2d(
                trimmed_labels_memmap, original_image, spacing_2d, 
                edge_trim_distance_threshold
            )

            # B. Generate Hull (Tissue Edge)
            hull_mask = generate_tight_hull_2d(
                original_image, trimmed_labels_memmap, 
                hull_closing_radius=hull_closing_radius,
                downsample_factor=4
            )

            # C. Recalculate Threshold
            print("  [Filter] Checking reference intensity...")
            valid = original_image[original_image > 0]
            if valid.size > 0:
                sub = np.random.choice(valid, min(valid.size, 100000))
                raw_otsu = threshold_otsu(sub)
                vol_max = np.max(sub)
            else:
                raw_otsu = 0
                vol_max = 0
            
            ref_thresh = raw_otsu if (vol_max > 5 and segmentation_threshold < 2.0) else segmentation_threshold
            global_brightness_cutoff = ref_thresh * brightness_cutoff_factor
            
            print(f"  Edge Trim Active: Dist<{edge_trim_distance_threshold}um, "
                  f"CoreBrightness>{global_brightness_cutoff:.2f}")

            # D. Trim Tissue Edges (with Protection)
            trim_edges_with_core_protection_2d(
                trimmed_labels_memmap,
                image=original_image,
                hull_mask=hull_mask,
                spacing=spacing_2d,
                distance_threshold=edge_trim_distance_threshold,
                global_brightness_cutoff=global_brightness_cutoff
            )

            # E. Generate Boundary for Viz
            struct = generate_binary_structure(2, 1)
            eroded_hull = ndimage.binary_erosion(hull_mask, structure=struct, iterations=1)
            hull_boundary_for_return = (hull_mask ^ eroded_hull)

        else:
            print("  Edge Trim Disabled (Dist=0).")
            hull_boundary_for_return = np.zeros(original_shape, dtype=bool)

        # 3. Final Cleanup (Relabel & Size Filter)
        if min_size_pixels > 0:
            relabel_and_filter_fragments_2d(trimmed_labels_memmap, min_size_pixels)

        _safe_close_memmap(trimmed_labels_memmap)

        return final_output_path, final_output_temp_dir, hull_boundary_for_return

    except Exception as e:
        print(f"\n!!! ERROR during 2D Hull Trimming: {e} !!!")
        traceback.print_exc()
        if final_output_temp_dir and os.path.exists(final_output_temp_dir):
            shutil.rmtree(final_output_temp_dir, ignore_errors=True)
        return None, None, None
    finally:
        if 'trimmed_labels_memmap' in locals():
            _safe_close_memmap(trimmed_labels_memmap)
        gc.collect()