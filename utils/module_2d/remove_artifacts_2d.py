import os
import gc
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
    binary_fill_holes  # Moved here from skimage
)
from skimage.filters import threshold_otsu  # type: ignore
from skimage.transform import resize  # type: ignore
from skimage.morphology import disk, binary_closing  # type: ignore

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
    Re-labels connected components and filters small fragments in 2D.
    Matches 3D logic: Label -> Size Filter -> Re-index.

    Args:
        labels_memmap: Memory-mapped 2D array of labels (modified in-place).
        min_size_pixels: Minimum size in pixels to retain.
    """
    if min_size_pixels <= 0:
        return

    print(f"  [Refine] Re-labeling and filtering fragments < {min_size_pixels} px...")

    # Load into memory for processing (2D is usually small enough)
    mask = labels_memmap[:] > 0
    
    # 1. Label connected components
    structure = generate_binary_structure(2, 1)  # 4-connectivity
    labeled_array, num_features = ndimage.label(mask, structure=structure)

    if num_features == 0:
        print("    No objects remaining.")
        labels_memmap[:] = 0
        labels_memmap.flush()
        return

    # 2. Calculate Sizes
    # bincount is fast; index 0 is background
    counts = np.bincount(labeled_array.ravel())
    
    # 3. Filter
    # Create a mapping: old_label -> new_label (or 0 if filtered)
    # We want to keep contiguous IDs
    new_labels = np.zeros(num_features + 1, dtype=np.int32)
    current_new_id = 1
    
    # Identify indices (labels) that meet the size criteria
    # We skip index 0 (background)
    keep_indices = np.where(counts[1:] >= min_size_pixels)[0] + 1
    kept_count = len(keep_indices)
    
    for old_id in keep_indices:
        new_labels[old_id] = current_new_id
        current_new_id += 1
        
    print(f"    Fragments: {num_features}. Kept: {kept_count}. Removed: {num_features - kept_count}.")

    # 4. Apply mapping and save
    # This maps the labeled_array integers through the new_labels array
    final_labels = new_labels[labeled_array]
    
    labels_memmap[:] = final_labels[:]
    labels_memmap.flush()


def generate_tight_hull_2d(
    image: np.ndarray,
    cell_mask: np.ndarray,
    downsample_factor: int = 4
) -> np.ndarray:
    """
    Generates a tight hull using morphological closing on downsampled image.
    Matches 3D logic: Downsample -> Otsu -> Close -> Fill -> Upsample.

    Args:
        image: Original 2D intensity image.
        cell_mask: Current segmentation mask.
        downsample_factor: Factor to scale down for morphology.

    Returns:
        np.ndarray: Boolean hull mask (in-memory).
    """
    print(f"  [HullGen] Generating Tight Hull (Downsample {downsample_factor}x)...")
    
    # Calculate global threshold for tissue
    # Sample pixels to avoid scanning whole image if large
    valid = image[image > 0]
    if valid.size > 0:
        sample = np.random.choice(valid, min(valid.size, 100000), replace=False)
        thresh = threshold_otsu(sample) * 0.5
    else:
        thresh = 0

    # Downsampled dimensions
    h, w = image.shape
    small_h = h // downsample_factor
    small_w = w // downsample_factor
    
    # Create raw mask: Tissue OR Segmentation
    mask_raw = (image > thresh) | (cell_mask > 0)
    
    if not np.any(mask_raw):
        return np.zeros_like(image, dtype=bool)

    # Downsample
    small_mask = resize(
        mask_raw, (small_h, small_w),
        order=0, preserve_range=True, anti_aliasing=False
    ).astype(bool)

    # Morphology
    # Using disk(10) to match 3D hardcoded value
    struct_elem = disk(10)
    closed = binary_closing(small_mask, footprint=struct_elem)
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
    Matches 3D logic: Distance Map -> Core Protection -> Filter.

    Args:
        labels_memmap: 2D Labels (modified in-place).
        image: Original 2D intensity image.
        hull_mask: Boolean hull mask.
        spacing: Pixel spacing (y, x).
        distance_threshold: Physical distance to trim (um).
        global_brightness_cutoff: Intensity threshold for core protection.
    """
    print("  [EdgeTrim] Trimming with Core Protection (2D)...")

    # 1. Distance Map
    # Distance from background (outside hull) into the tissue
    # distance_transform_edt calculates distance to nearest zero. 
    # Since hull_mask is 1 (Tissue) and 0 (Background), this gives dist from edge inwards.
    print("    Calculating distance map...")
    dist_map = distance_transform_edt(hull_mask, sampling=spacing)

    # 2. Protection
    # Protect if Bright Core
    protection_radius_um = 6.0
    # Determine iterations: radius / pixel_size
    avg_spacing = np.mean(spacing)
    protection_iter = max(2, min(10, int(protection_radius_um / avg_spacing)))
    
    print(f"    Applying Protection (Dilate Bright Cores {protection_iter}x)...")
    
    # Identify cores
    mask_labels = labels_memmap[:] > 0
    mask_bright = image > global_brightness_cutoff
    core_mask = mask_labels & mask_bright
    
    # Dilate cores
    if np.any(core_mask):
        struct = generate_binary_structure(2, 1)
        protected_mask = binary_dilation(
            core_mask, structure=struct, iterations=protection_iter
        )
    else:
        protected_mask = np.zeros_like(core_mask)

    # 3. Kill Logic
    # Delete if: Labeled AND Near Edge AND Not Protected
    # Near Edge means Distance < Threshold
    to_delete = (mask_labels) & (dist_map < distance_threshold) & (~protected_mask)
    
    count = np.sum(to_delete)
    if count > 0:
        labels_memmap[to_delete] = 0
        labels_memmap.flush()
        print(f"    Deleted {count} artifact pixels.")
    else:
        print("    No artifact pixels found to delete.")


def apply_hull_trimming_2d(
    raw_labels_path: str,
    original_image: np.ndarray,
    spacing: Union[Tuple[float, float], Tuple[float, float, float]],
    segmentation_threshold: float,
    hull_boundary_thickness_phys: float,  # Unused in main logic, kept for signature parity
    edge_trim_distance_threshold: float,
    brightness_cutoff_factor: float,
    min_size_pixels: int,
    smoothing_iterations: int = 1,  # Unused in main logic, kept for signature parity
    heal_iterations: int = 1        # Unused in main logic, kept for signature parity
) -> Tuple[Optional[str], Optional[str], Optional[np.ndarray]]:
    """
    Main Entry Point for Step 2 (2D).
    Applies Hull Generation, Edge Trimming, and Size Filtering.

    Args:
        raw_labels_path: Path to input .dat file.
        original_image: 2D Intensity image.
        spacing: Pixel spacing.
        segmentation_threshold: Threshold from Step 1.
        hull_boundary_thickness_phys: (Unused parity arg)
        edge_trim_distance_threshold: Distance to trim.
        brightness_cutoff_factor: Multiplier for core protection.
        min_size_pixels: Size filter.
        smoothing_iterations: (Unused parity arg)
        heal_iterations: (Unused parity arg)

    Returns:
        Tuple: (output_path, temp_dir, hull_boundary_mask_for_viz)
    """
    print(f"\n--- Applying Hull Generation and Edge Trimming (2D) ---")
    
    # Validate spacing
    try:
        if len(spacing) == 3:
            spacing_2d = tuple(float(s) for s in spacing[1:])
        else:
            spacing_2d = tuple(float(s) for s in spacing)
    except:
        spacing_2d = (1.0, 1.0)

    original_shape = original_image.shape
    final_output_temp_dir = tempfile.mkdtemp(prefix="trimmed_final_2d_")
    trimmed_labels_memmap = None
    hull_boundary_for_return = None

    try:
        # 1. Output Setup
        final_output_path = os.path.join(final_output_temp_dir, 'trimmed_labels.dat')
        shutil.copyfile(raw_labels_path, final_output_path)
        trimmed_labels_memmap = np.memmap(
            final_output_path, dtype=np.int32, mode='r+', shape=original_shape
        )

        # 2. Edge Trimming
        if edge_trim_distance_threshold > 0:
            # A. Generate Hull
            hull_mask = generate_tight_hull_2d(
                original_image, trimmed_labels_memmap, downsample_factor=4
            )

            # B. Recalculate Threshold (Robustness)
            print("  [Filter] Checking reference intensity...")
            valid = original_image[original_image > 0]
            if valid.size > 0:
                # Subsample if large
                sub = np.random.choice(valid, min(valid.size, 100000))
                raw_otsu = threshold_otsu(sub)
                vol_max = np.max(sub)
            else:
                raw_otsu = 0
                vol_max = 0
            
            # Heuristic: If image is dim or seg threshold suspicious, fallback to otsu
            ref_thresh = raw_otsu if (vol_max > 5 and segmentation_threshold < 2.0) else segmentation_threshold
            global_brightness_cutoff = ref_thresh * brightness_cutoff_factor
            
            print(f"  Edge Trim Active: Dist<{edge_trim_distance_threshold}um, "
                  f"CoreBrightness>{global_brightness_cutoff:.2f}")

            # C. Trim
            trim_edges_with_core_protection_2d(
                labels_memmap=trimmed_labels_memmap,
                image=original_image,
                hull_mask=hull_mask,
                spacing=spacing_2d,
                distance_threshold=edge_trim_distance_threshold,
                global_brightness_cutoff=global_brightness_cutoff
            )

            # D. Generate Boundary for Viz
            # Erode hull slightly to get a rim
            struct = generate_binary_structure(2, 1)
            eroded_hull = ndimage.binary_erosion(hull_mask, structure=struct, iterations=1)
            hull_boundary_for_return = (hull_mask ^ eroded_hull)

        else:
            print("  Edge Trim Disabled (Dist=0).")
            hull_boundary_for_return = np.zeros(original_shape, dtype=bool)

        # 3. Final Cleanup (Relabel & Size Filter)
        if min_size_pixels > 0:
            relabel_and_filter_fragments_2d(trimmed_labels_memmap, min_size_pixels)

        # Cleanup handle
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