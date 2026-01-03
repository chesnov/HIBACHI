import os
import gc
import math
import time
import shutil
import tempfile
import traceback
import multiprocessing as mp
from functools import partial
from typing import Tuple, List, Dict, Any, Optional, Union, Generator

import numpy as np
import zarr
import dask.array as da
import dask_image.ndmorph
import dask_image.ndmeasure
from dask.diagnostics import ProgressBar
from scipy import ndimage
from scipy.ndimage import gaussian_filter, generate_binary_structure, white_tophat
from skimage.filters import frangi, sato  # type: ignore
from skimage.morphology import remove_small_objects, disk  # type: ignore
from tqdm import tqdm

# Set fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)


def _init_worker() -> None:
    """Initializes worker processes to use single-threaded BLAS/OMP."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _get_safe_temp_dir(base_path: Optional[str], suffix: str = "") -> str:
    """Creates a temporary directory in the project folder to prevent /tmp overflows."""
    if base_path and os.path.isdir(base_path):
        scratch_root = os.path.join(base_path, "hibachi_scratch")
    else:
        scratch_root = os.path.abspath("hibachi_scratch")
    
    os.makedirs(scratch_root, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"step1_2d_{suffix}_", dir=scratch_root)


def _cleanup_registry(registry: Dict[str, Tuple[Any, str, str]]) -> None:
    """
    Helper to clean up intermediate memmaps tracked in a registry.
    Args:
        registry: Dict mapping key -> (memmap_obj, path, temp_dir)
    """
    for key in list(registry.keys()):
        mm, path, d = registry.pop(key)
        # Close memmap handle
        if isinstance(mm, np.memmap) and hasattr(mm, '_mmap') and mm._mmap is not None:
            try:
                mm._mmap.close()
            except Exception:
                pass
        del mm
        
        # Remove directory
        if d and os.path.exists(d):
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception:
                pass


def _get_chunk_slices_2d(
    shape: Tuple[int, ...],
    chunk_shape: Tuple[int, ...],
    overlap: int = 0
) -> Generator[Tuple[Tuple[slice, ...], Tuple[slice, ...]], None, None]:
    """
    Generates read/write slices for 2D chunked processing.
    
    Args:
        shape: (Y, X)
        chunk_shape: (Y_chunk, X_chunk)
    """
    y_shape, x_shape = shape
    cy, cx = chunk_shape

    for y in range(0, y_shape, cy):
        for x in range(0, x_shape, cx):
            # Valid output region (no overlap)
            y_start, y_stop = y, min(y + cy, y_shape)
            x_start, x_stop = x, min(x + cx, x_shape)
            
            write_slice = (
                slice(y_start, y_stop),
                slice(x_start, x_stop)
            )

            # Input region (with overlap, clamped)
            y_start_pad = max(0, y_start - overlap)
            y_stop_pad = min(y_shape, y_stop + overlap)
            x_start_pad = max(0, x_start - overlap)
            x_stop_pad = min(x_shape, x_stop + overlap)

            read_slice = (
                slice(y_start_pad, y_stop_pad),
                slice(x_start_pad, x_stop_pad)
            )

            yield read_slice, write_slice


def _process_block_worker_2d(
    chunk_info: Tuple[Tuple[slice, ...], Tuple[slice, ...]],
    input_memmap_info: Tuple[str, Tuple[int, ...], Any],
    output_memmap_info: Tuple[str, Tuple[int, ...], Any],
    sigmas_voxel_2d: List[float],
    sigma_voxel_2d_smooth: Optional[Tuple[float, float]],
    black_ridges: bool,
    frangi_alpha: float,
    frangi_beta: float,
    frangi_gamma: float,
    subtract_background_radius: int
) -> Optional[str]:
    """
    Worker function for processing a 2D block.
    Matches 3D logic: Scale-Independent Filtering with Intensity Gating.
    """
    input_memmap = None
    output_memmap = None
    try:
        read_slices, write_slices = chunk_info
        input_path, input_shape, input_dtype = input_memmap_info
        output_path, output_shape, output_dtype = output_memmap_info

        # Open memmaps
        input_memmap = np.memmap(
            input_path, dtype=input_dtype, mode='r', shape=input_shape
        )
        output_memmap = np.memmap(
            output_path, dtype=output_dtype, mode='r+', shape=output_shape
        )

        # Load Chunk
        block_data = input_memmap[read_slices].astype(np.float32)

        # 0. Pre-Smoothing
        if sigma_voxel_2d_smooth is not None:
            block_data = gaussian_filter(
                block_data, sigma=sigma_voxel_2d_smooth, mode='reflect'
            )

        # Background Subtraction
        if subtract_background_radius > 0:
            struct_size = int(2 * subtract_background_radius + 1)
            block_data = white_tophat(block_data, size=struct_size)

        # --- Scale-Independent Filter Logic (Same as 3D) ---
        combined_scales = np.zeros_like(block_data, dtype=np.float32)

        for sigma in sigmas_voxel_2d:
            # DYNAMIC BETA:
            # Small scales (< 2.0): Enforce tubes (beta=frangi_beta)
            # Large scales (>= 2.0): Allow blobs/somas (beta=1.0)
            beta_val = 1.0 if sigma >= 2.0 else frangi_beta
            
            f_res = frangi(
                block_data, sigmas=[sigma], 
                alpha=frangi_alpha, beta=beta_val, gamma=frangi_gamma,
                black_ridges=black_ridges, mode='reflect'
            )
            s_res = sato(
                block_data, sigmas=[sigma], 
                black_ridges=black_ridges, mode='reflect'
            )
            
            scale_res = np.maximum(f_res, s_res)
            
            # INTENSITY GATING:
            # If scale is large (soma), weight response by raw intensity
            if sigma >= 2.0:
                scale_res *= block_data
            
            combined_scales = np.maximum(combined_scales, scale_res)

        # Crop padding
        y_start_rel = write_slices[0].start - read_slices[0].start
        y_stop_rel = y_start_rel + (write_slices[0].stop - write_slices[0].start)
        x_start_rel = write_slices[1].start - read_slices[1].start
        x_stop_rel = x_start_rel + (write_slices[1].stop - write_slices[1].start)

        result_crop = combined_scales[y_start_rel:y_stop_rel, x_start_rel:x_stop_rel]

        # Write
        output_memmap[write_slices] = result_crop
        return None

    except Exception as e:
        return f"Error_chunk_{chunk_info}: {str(e)}"
    finally:
        try:
            del input_memmap, output_memmap, block_data, combined_scales
        except:
            pass
        gc.collect()


def enhance_tubular_structures_blocked_2d(
    image: np.ndarray,
    scales: List[float],
    spacing: Tuple[float, float],
    temp_root_path: Optional[str],
    black_ridges: bool = False,
    frangi_alpha: float = 0.5,
    frangi_beta: float = 0.5,
    frangi_gamma: float = 2.0,
    apply_smoothing: bool = True,
    smoothing_sigma_phys: float = 0.5,
    skip_enhancement: bool = False,
    subtract_background_radius: int = 0
) -> Tuple[np.memmap, str, str]:
    """
    Enhances tubular structures using 2D chunked processing.
    """
    print(f"  [Enhance] Image: {image.shape}, Spacing: {spacing} (Block Mode 2D)")
    
    # Output Setup
    output_temp_dir = _get_safe_temp_dir(temp_root_path, 'tubular_output')
    output_path = os.path.join(output_temp_dir, 'processed_image.dat')
    output_memmap = np.memmap(
        output_path, dtype=np.float32, mode='w+', shape=image.shape
    )

    if skip_enhancement:
        print("  [Enhance] Skipping filters (Pass-through).")
        chunk_gen = _get_chunk_slices_2d(image.shape, (2048, 2048), overlap=0)
        for _, write_slice in tqdm(list(chunk_gen), desc="  [Enhance] Copying"):
            output_memmap[write_slice] = image[write_slice].astype(np.float32)
        output_memmap.flush()
        return output_memmap, output_path, output_temp_dir

    # Configuration
    sigma_voxel_2d_smooth = None
    overlap_px = 0
    if apply_smoothing and smoothing_sigma_phys > 0:
        sigma_voxel_2d_smooth = tuple(
            smoothing_sigma_phys / s if s > 0 else 0 for s in spacing
        )
        overlap_px = math.ceil(max(sigma_voxel_2d_smooth) * 3)

    # Filters
    avg_spacing = np.mean(spacing)
    sigmas_voxel_2d = sorted([s / avg_spacing for s in scales if s > 0])
    
    max_filter_sigma = max(sigmas_voxel_2d) if sigmas_voxel_2d else 0
    overlap_px = max(overlap_px, math.ceil(max_filter_sigma * 4))
    overlap_px = max(overlap_px, 32) # Minimum safe overlap

    # Chunks
    # Use larger chunks for 2D (e.g. 2048x2048) since no Z dimension consumes RAM
    chunk_shape = (2048, 2048) 
    chunks = list(_get_chunk_slices_2d(image.shape, chunk_shape, overlap=overlap_px))
    
    # Input Dump (if not memmap)
    temp_dirs_created: List[str] = []
    if not isinstance(image, np.memmap):
        print("  [Enhance] Dumping input to temp memmap...")
        dump_dir = _get_safe_temp_dir(temp_root_path, 'input_dump')
        temp_dirs_created.append(dump_dir)
        dump_path = os.path.join(dump_dir, 'input_dump.dat')
        input_memmap = np.memmap(
            dump_path, dtype=image.dtype, mode='w+', shape=image.shape
        )
        input_memmap[:] = image[:]
        input_memmap.flush()
        input_info = (dump_path, image.shape, image.dtype)
    else:
        input_info = (image.filename, image.shape, image.dtype)

    output_info = (output_path, image.shape, np.float32)
    num_workers = max(1, os.cpu_count() - 2 if os.cpu_count() > 2 else 1)

    worker_func = partial(
        _process_block_worker_2d,
        input_memmap_info=input_info,
        output_memmap_info=output_info,
        sigmas_voxel_2d=sigmas_voxel_2d,
        sigma_voxel_2d_smooth=sigma_voxel_2d_smooth,
        black_ridges=black_ridges,
        frangi_alpha=frangi_alpha,
        frangi_beta=frangi_beta,
        frangi_gamma=frangi_gamma,
        subtract_background_radius=subtract_background_radius
    )

    print(f"  [Enhance] Processing {len(chunks)} blocks (overlap {overlap_px})...")
    
    pool = mp.Pool(processes=num_workers, initializer=_init_worker, maxtasksperchild=5)
    try:
        results = list(tqdm(
            pool.imap_unordered(worker_func, chunks),
            total=len(chunks),
            desc="  [Enhance] 2D Filtering"
        ))
        errors = [r for r in results if r is not None]
        if errors:
            raise RuntimeError(f"Block processing failed: {errors[0]}")
        output_memmap.flush()
    finally:
        pool.terminate()
        pool.join()
        for d in temp_dirs_created:
            if os.path.exists(d):
                shutil.rmtree(d, ignore_errors=True)
        gc.collect()

    return output_memmap, output_path, output_temp_dir


def segment_cells_first_pass_raw_2d(
    image: np.ndarray,
    spacing: Union[Tuple[float, float], Tuple[float, float, float]],
    tubular_scales: List[float] = [0.5, 1.0, 2.0, 3.0],
    smooth_sigma: float = 0.5,
    connect_max_gap_physical: float = 1.0,
    min_size_pixels: int = 50,
    low_threshold_percentile: float = 95.0,
    high_threshold_percentile: float = 100.0,
    skip_tubular_enhancement: bool = False,
    subtract_background_radius: int = 0,
    temp_root_path: Optional[str] = None,
    **kwargs: Any
) -> Tuple[Optional[str], Optional[str], float, Dict[str, Any]]:
    """
    Executes Step 1: Raw Segmentation Pipeline (2D).
    Pipeline: Normalize -> Enhance -> Threshold -> Connect -> Clean -> Label.
    Uses Block-Based processing for memory safety on large XY planes.
    """
    print(f"\n--- Step 1: Raw 2D Segmentation (Chunk-Optimized) ---")

    # Extract 2D spacing
    try:
        if len(spacing) == 3:
            spacing_2d = tuple(float(s) for s in spacing[1:])
        else:
            spacing_2d = tuple(float(s) for s in spacing)
    except:
        spacing_2d = (1.0, 1.0)

    # Check Skip
    if len(tubular_scales) == 1 and abs(tubular_scales[0]) < 1e-9:
        skip_tubular_enhancement = True
    elif not tubular_scales:
        skip_tubular_enhancement = True

    temp_dirs_to_clean: List[str] = []
    final_labels_memmap = None
    labels_path = None
    labels_temp_dir = None
    segmentation_threshold = 0.0

    # Registry for cleanup
    memmap_registry: Dict[str, Tuple[Any, str, str]] = {}

    try:
        # --- Stage 1: Normalization (Chunked) ---
        print("\n  [Step 1.1] Normalizing image brightness...")
        normalized_temp_dir = _get_safe_temp_dir(temp_root_path, 'normalize')
        temp_dirs_to_clean.append(normalized_temp_dir)
        normalized_path = os.path.join(normalized_temp_dir, 'normalized.dat')
        normalized_memmap = np.memmap(
            normalized_path, dtype=np.float32, mode='w+', shape=image.shape
        )
        memmap_registry['normalized'] = (normalized_memmap, normalized_path, normalized_temp_dir)

        # 1a. Global Percentile (Reservoir Sampling)
        MAX_SAMPLES = 5_000_000
        # Convert generator to list for reuse
        processing_chunks = list(_get_chunk_slices_2d(image.shape, (2048, 2048), overlap=0))
        reservoir = []
        
        for read_sl, _ in tqdm(processing_chunks, desc="    Sampling Histogram"):
            chunk = image[read_sl]
            # Subsample
            vals = chunk[::8, ::8].ravel()
            vals = vals[vals > 0] # Ignore strict zero background
            if vals.size > 0:
                if len(reservoir) < MAX_SAMPLES:
                    take = min(len(vals), MAX_SAMPLES - len(reservoir))
                    reservoir.extend(vals[:take])
        
        if reservoir:
            high_val = np.percentile(reservoir, high_threshold_percentile)
            if high_val < 1e-9: high_val = 1.0
        else:
            high_val = 1.0
            
        print(f"    Normalization Max (p{high_threshold_percentile}): {high_val:.2f}")

        # 1b. Apply Normalization
        for read_sl, _ in tqdm(processing_chunks, desc="    Applying Normalization"):
            chunk = image[read_sl].astype(np.float32)
            normalized_memmap[read_sl] = chunk / high_val
        
        normalized_memmap.flush()

        # --- Stage 2: Enhancement (Blocked) ---
        print("\n  [Step 1.2] Feature Enhancement (on Normalized Data)...")
        enhanced_memmap, enh_path, enhanced_temp_dir = enhance_tubular_structures_blocked_2d(
            normalized_memmap,
            scales=tubular_scales,
            spacing=spacing_2d,
            temp_root_path=temp_root_path,
            apply_smoothing=(smooth_sigma > 0),
            smoothing_sigma_phys=smooth_sigma,
            skip_enhancement=skip_tubular_enhancement,
            subtract_background_radius=subtract_background_radius,
            frangi_gamma=2.0 # Fixed to 2.0 as per 3D tuning
        )
        memmap_registry['enhanced'] = (enhanced_memmap, enh_path, enhanced_temp_dir)
        temp_dirs_to_clean.append(enhanced_temp_dir)

        # Cleanup Normalization (save space)
        _cleanup_registry({'normalized': memmap_registry.pop('normalized')})

        # --- Stage 3: Thresholding ---
        print("\n  [Step 1.3] Calculating threshold...")
        thresh_samples = []
        # Reuse processing_chunks list
        for read_sl, _ in tqdm(processing_chunks[::2], desc="    Sampling Enhanced"):
            chunk = enhanced_memmap[read_sl]
            vals = chunk[::8, ::8].ravel()
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > 1e-6]
            if vals.size > 0:
                thresh_samples.append(vals)
        
        if thresh_samples:
            all_s = np.concatenate(thresh_samples)
            if all_s.size > MAX_SAMPLES:
                all_s = np.random.choice(all_s, MAX_SAMPLES, replace=False)
            segmentation_threshold = float(np.percentile(all_s, low_threshold_percentile))
        
        print(f"    Calculated Threshold: {segmentation_threshold:.6f}")

        # Create Binary
        binary_temp_dir = _get_safe_temp_dir(temp_root_path, 'binary')
        temp_dirs_to_clean.append(binary_temp_dir)
        binary_path = os.path.join(binary_temp_dir, 'b.dat')
        binary_memmap = np.memmap(
            binary_path, dtype=bool, mode='w+', shape=image.shape
        )
        memmap_registry['binary'] = (binary_memmap, binary_path, binary_temp_dir)

        # Reuse processing_chunks list
        for read_sl, _ in tqdm(processing_chunks, desc="    Creating Binary Mask"):
            chunk = enhanced_memmap[read_sl]
            binary_memmap[read_sl] = (chunk > segmentation_threshold)
        
        binary_memmap.flush()

        # Cleanup Enhanced
        _cleanup_registry({'enhanced': memmap_registry.pop('enhanced')})
        gc.collect()

        # --- Stage 4: Cleaning & Labeling (Dask) ---
        print("\n  [Step 1.4] Cleaning & Labeling (Dask)...")
        
        # Use smaller chunks for Dask graph
        dask_chunk_size = (4096, 4096)
        
        connected_temp_dir = _get_safe_temp_dir(temp_root_path, 'connected')
        temp_dirs_to_clean.append(connected_temp_dir)
        connected_path = os.path.join(connected_temp_dir, 'connected.dat')
        connected_memmap = np.memmap(
            connected_path, dtype=bool, mode='w+', shape=image.shape
        )
        memmap_registry['connected'] = (connected_memmap, connected_path, connected_temp_dir)

        binary_dask = da.from_array(binary_memmap, chunks=dask_chunk_size)
        
        # Morphological Closing
        avg_spacing = np.mean(spacing_2d)
        radius_px = math.ceil((connect_max_gap_physical / 2) / avg_spacing)
        if radius_px > 0:
            structure = disk(radius_px)
            # Dask binary_closing for 2D needs 2D structure
            connected_dask = dask_image.ndmorph.binary_closing(
                binary_dask, structure=structure
            )
        else:
            connected_dask = binary_dask
            
        print("    Applying binary closing...")
        with ProgressBar(dt=5):
            da.store(connected_dask, connected_memmap, scheduler='threads')
        
        # Cleanup Binary
        _cleanup_registry({'binary': memmap_registry.pop('binary')})

        # Labeling
        labeled_temp_dir = _get_safe_temp_dir(temp_root_path, 'labeled_zarr')
        temp_dirs_to_clean.append(labeled_temp_dir)
        labeled_zarr_path = os.path.join(labeled_temp_dir, 'labeled.zarr')
        
        d_conn = da.from_array(connected_memmap, chunks=dask_chunk_size)
        structure_lab = generate_binary_structure(2, 1) # 4-conn
        
        print("    Labeling connected components...")
        labeled_dask, num_features_dask = dask_image.ndmeasure.label(
            d_conn, structure=structure_lab
        )
        
        with ProgressBar(dt=5):
            labeled_dask.to_zarr(labeled_zarr_path, overwrite=True)
            
        num_features = num_features_dask.compute()
        print(f"      Found {num_features} objects.")
        
        # Cleanup Connected
        _cleanup_registry({'connected': memmap_registry.pop('connected')})

        # --- Stage 5: Size Filtering (Block-wise Remap) ---
        print(f"    Filtering objects < {min_size_pixels} pixels...")
        labels_temp_dir = _get_safe_temp_dir(temp_root_path, 'labels_final')
        labels_path = os.path.join(labels_temp_dir, 'l.dat')
        final_labels_memmap = np.memmap(
            labels_path, dtype=np.int32, mode='w+', shape=image.shape
        )
        
        initial_labeled_zarr = zarr.open(labeled_zarr_path, mode='r')
        d_lbl = da.from_zarr(labeled_zarr_path)
        
        print("      Calculating sizes...")
        counts, bins = da.histogram(
            d_lbl, bins=num_features + 1, range=[-0.5, num_features + 0.5]
        )
        counts_res = counts.compute()
        
        valid_indices = np.where(counts_res[1:] >= min_size_pixels)[0] + 1
        print(f"      Retaining {len(valid_indices)} / {num_features} objects.")
        
        lookup = np.zeros(num_features + 1, dtype=np.int32)
        current_new = 1
        for old_id in valid_indices:
            lookup[old_id] = current_new
            current_new += 1
            
        print("      Writing filtered labels...")
        io_chunks = (2048, 2048)
        io_gen = _get_chunk_slices_2d(image.shape, io_chunks, overlap=0)
        
        for read_sl, write_sl in tqdm(list(io_gen), desc="      Remapping"):
            block_data = initial_labeled_zarr[read_sl]
            filtered_block = lookup[block_data]
            final_labels_memmap[write_sl] = filtered_block
            
        final_labels_memmap.flush()

        params_record = {
            'spacing': spacing_2d,
            'tubular_scales': tubular_scales,
            'smooth_sigma': smooth_sigma,
            'connect_max_gap_physical': connect_max_gap_physical,
            'min_size_pixels': min_size_pixels,
            'low_threshold_percentile': low_threshold_percentile,
            'high_threshold_percentile': high_threshold_percentile
        }
        return labels_path, labels_temp_dir, segmentation_threshold, params_record

    except Exception as e:
        print(f"Error during raw segmentation: {e}")
        traceback.print_exc()
        _cleanup_registry(memmap_registry)
        return None, None, 0.0, {}

    finally:
        # Final cleanup if anything remains in registry
        _cleanup_registry(memmap_registry)
        
        if final_labels_memmap is not None:
            del final_labels_memmap
        print("\n  [Step 1] Cleaning up intermediate temp directories...")
        for d_dir in temp_dirs_to_clean:
            if d_dir and os.path.exists(d_dir):
                try:
                    shutil.rmtree(d_dir, ignore_errors=True)
                except Exception:
                    pass
        gc.collect()