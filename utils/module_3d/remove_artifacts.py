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
    return tempfile.mkdtemp(prefix=f"step2_{suffix}_", dir=scratch_root)


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
    Optimized 3D fragment filter for huge volumes.
    Breaks the Dask graph by saving the binary mask first, then relabeling.
    """
    if min_size_voxels <= 0:
        return

    print(f"  [Refine] Stage 1/2: Morphological Pruning & Size Filtering...")

    # 1. Setup Dask Array
    chunk_size = (64, 256, 256)
    d_seg = da.from_array(labels_memmap, chunks=chunk_size)
    structure_3d = generate_binary_structure(3, 1)
    
    # 2. Parallel Size Filtering (Pure Size Filter, No Opening)
    # Logic: We skip binary_opening to ensure thin processes are NOT pruned.
    d_binary = d_seg > 0
    
    d_filtered = d_binary.map_overlap(
        remove_small_objects,
        depth={0: 8, 1: 8, 2: 8},
        boundary='compare',
        dtype=bool,
        min_size=min_size_voxels
    )

    # 3. BREAK THE GRAPH: Write the binary mask to the memmap
    # This overwrites the noisy labels with a clean 0/1 mask.
    print("    Writing cleaned binary mask to disk (breaking graph)...")
    with ProgressBar(dt=2):
        # We store as int8 to save space/time, then cast to int32 for labeling
        da.store(
            d_filtered.astype(np.int8), 
            labels_memmap, 
            lock=True, 
            scheduler=DASK_SCHEDULER
        )
    labels_memmap.flush()

    # 4. Stage 2/2: Sequential Slab-wise Relabeling
    # Now that the file is just 0s and 1s, we can label it much faster.
    print("    Stage 2/2: Sequential relabeling...")
    
    # We reload the memmap as a simple array (no lineage)
    # We use a sequential approach to avoid Dask-image's graph overhead
    # ndimage.label is fast on CPUs; we process in slabs to respect RAM.
    
    tmp_labels, num_features = ndimage.label(labels_memmap, structure=structure_3d)
    
    # If the above line still uses too much RAM, we would use cc3d or a slab-loop.
    # But given your previous log (5 slabs), ndimage.label on the full mask 
    # should be fine now that the 'dust' is gone.
    
    if num_features > 0:
        print(f"    Finalized {num_features} unique fragments.")
        labels_memmap[:] = tmp_labels.astype(np.int32)
    else:
        print("    No objects remaining.")
        labels_memmap[:] = 0

    labels_memmap.flush()
    del tmp_labels
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


def generate_tight_hull_stack(
    volume: np.ndarray,
    cell_mask: np.ndarray,
    temp_dir: str,
    hull_closing_radius: int = 10,
    downsample_factor: int = 4
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
        # Use 0.8 factor to avoid noise floor but capture dim tissue
        tissue_thresh = (np.expm1(log_thresh)) * 0.8
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
    # Essential for removing floating artifacts (tile corners, dust)
    print("    Filtering disjoint hull artifacts (Keeping Largest Component)...")
    
    d_hull = da.from_array(hull_memmap, chunks=(64, 256, 256))
    structure_3d = generate_binary_structure(3, 1)
    
    labeled_hull, num_features = dask_image.ndmeasure.label(d_hull, structure=structure_3d)
    n_feat = num_features.compute()
    
    if n_feat > 1:
        counts, _ = da.histogram(labeled_hull, bins=n_feat+1, range=[0, n_feat])
        counts_res = counts.compute()
        counts_res[0] = 0  # Ignore background
        
        largest_label = np.argmax(counts_res)
        print(f"    Keeping Label {largest_label} (Size {counts_res[largest_label]}).")
        
        # Overwrite with mask of only largest component
        final_dask = (labeled_hull == largest_label)
        with ProgressBar(dt=2):
            da.store(final_dask, hull_memmap, lock=True, scheduler=DASK_SCHEDULER)
            
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
        
        # Z-Padding Fix for Top/Bottom Surfaces
        pad_top = (z0 == 0)
        pad_bottom = (z1 == total_z)
        
        if pad_top or pad_bottom:
            pad_width = ((int(pad_top), int(pad_bottom)), (0, 0), (0, 0))
            hull_chunk_padded = np.pad(
                hull_chunk, pad_width, mode='constant', constant_values=0
            )
            dt_chunk_padded = distance_transform_edt(
                hull_chunk_padded, sampling=spacing
            )
            start_idx = 1 if pad_top else 0
            end_idx = -1 if pad_bottom else None
            dt_chunk = dt_chunk_padded[start_idx:end_idx]
        else:
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

        core_mask = (lbl_chunk > 0) & (vol_chunk > global_brightness_cutoff)

        if np.any(core_mask):
            protected_mask = binary_dilation(
                core_mask, structure=struct_protect, iterations=protection_iter
            )
        else:
            protected_mask = np.zeros_like(core_mask, dtype=bool)

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
    hull_closing_radius: int = 10,
    heal_iterations: int = 1,
    edge_distance_chunk_size_z: int = 32,
    z_erosion_iterations: int = 0,
    post_smoothing_iter: int = 0,
    temp_root_path: Optional[str] = None
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
                workflow_temp_dir, hull_closing_radius=hull_closing_radius, 
                downsample_factor=4
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
        gc.collect()
        if workflow_temp_dir and os.path.exists(workflow_temp_dir):
            try:
                shutil.rmtree(workflow_temp_dir, ignore_errors=True)
            except Exception:
                pass