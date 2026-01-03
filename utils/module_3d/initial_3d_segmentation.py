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
    return tempfile.mkdtemp(prefix=f"step1_{suffix}_", dir=scratch_root)


def _get_chunk_slices(
    shape: Tuple[int, ...],
    chunk_shape: Tuple[int, ...],
    overlap: int = 0
) -> Generator[Tuple[Tuple[slice, ...], Tuple[slice, ...]], None, None]:
    """
    Generates read/write slices for chunked processing with overlap.
    """
    z_shape, y_shape, x_shape = shape
    cz, cy, cx = chunk_shape

    for z in range(0, z_shape, cz):
        for y in range(0, y_shape, cy):
            for x in range(0, x_shape, cx):
                # Valid output region (no overlap)
                z_start, z_stop = z, min(z + cz, z_shape)
                y_start, y_stop = y, min(y + cy, y_shape)
                x_start, x_stop = x, min(x + cx, x_shape)
                
                write_slice = (
                    slice(z_start, z_stop),
                    slice(y_start, y_stop),
                    slice(x_start, x_stop)
                )

                # Input region (with overlap, clamped to bounds)
                z_start_pad = max(0, z_start - overlap)
                z_stop_pad = min(z_shape, z_stop + overlap)
                y_start_pad = max(0, y_start - overlap)
                y_stop_pad = min(y_shape, y_stop + overlap)
                x_start_pad = max(0, x_start - overlap)
                x_stop_pad = min(x_shape, x_stop + overlap)

                read_slice = (
                    slice(z_start_pad, z_stop_pad),
                    slice(y_start_pad, y_stop_pad),
                    slice(x_start_pad, x_stop_pad)
                )

                yield read_slice, write_slice


def _process_block_worker(
    chunk_info: Tuple[Tuple[slice, ...], Tuple[slice, ...]],
    input_memmap_info: Tuple[str, Tuple[int, ...], Any],
    output_memmap_info: Tuple[str, Tuple[int, ...], Any],
    sigmas_voxel_2d: List[float],
    sigma_voxel_3d: Optional[Tuple[float, float, float]],
    black_ridges: bool,
    frangi_alpha: float,
    frangi_beta: float,
    frangi_gamma: float,
    subtract_background_radius: int
) -> Optional[str]:
    """
    Worker function for processing a 3D block (Enhancement Step).
    Performs Scale-Independent Filtering with Intensity Gating for large scales.
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

        # Load Chunk into RAM
        block_data = input_memmap[read_slices].astype(np.float32)

        # 0. 3D Pre-Smoothing
        if sigma_voxel_3d is not None:
            block_data = gaussian_filter(
                block_data, sigma=sigma_voxel_3d, mode='reflect'
            )

        # Indices to crop padding
        z_start_rel = write_slices[0].start - read_slices[0].start
        y_start_rel = write_slices[1].start - read_slices[1].start
        x_start_rel = write_slices[2].start - read_slices[2].start
        
        z_stop_rel = z_start_rel + (write_slices[0].stop - write_slices[0].start)
        y_stop_rel = y_start_rel + (write_slices[1].stop - write_slices[1].start)
        x_stop_rel = x_start_rel + (write_slices[2].stop - write_slices[2].start)

        valid_shape = (
            z_stop_rel - z_start_rel,
            y_stop_rel - y_start_rel,
            x_stop_rel - x_start_rel
        )
        result_block = np.zeros(valid_shape, dtype=np.float32)

        # 1. Slice-by-Slice Processing (Z-axis of the block)
        for z in range(block_data.shape[0]):
            # Skip padding slices
            if z < z_start_rel or z >= z_stop_rel:
                continue

            slice_2d = block_data[z]

            if subtract_background_radius > 0:
                struct_size = int(2 * subtract_background_radius + 1)
                slice_2d = white_tophat(slice_2d, size=struct_size)

            # Accumulator for the "OR" operation
            combined_scales = np.zeros_like(slice_2d, dtype=np.float32)
            
            for sigma in sigmas_voxel_2d:
                # DYNAMIC BETA:
                # Small scales (< 2.0): Enforce tubes (beta=frangi_beta)
                # Large scales (>= 2.0): Allow blobs/somas (beta=1.0)
                beta_val = 1.0 if sigma >= 2.0 else frangi_beta
                
                f_res = frangi(
                    slice_2d, sigmas=[sigma], 
                    alpha=frangi_alpha, beta=beta_val, gamma=frangi_gamma,
                    black_ridges=black_ridges, mode='reflect'
                )
                s_res = sato(
                    slice_2d, sigmas=[sigma], 
                    black_ridges=black_ridges, mode='reflect'
                )
                
                # Combine filters for this scale
                scale_res = np.maximum(f_res, s_res)
                
                # INTENSITY GATING for Large Scales:
                # If scale is large (soma), weight the response by the raw intensity.
                # This suppresses "blob noise" in the background (where intensity is low)
                # while preserving real somas (where intensity is high).
                if sigma >= 2.0:
                    scale_res *= slice_2d
                
                # "OR" Operation (Max Projection)
                combined_scales = np.maximum(combined_scales, scale_res)

            # Crop YX padding and store
            result_block[z - z_start_rel] = \
                combined_scales[y_start_rel:y_stop_rel, x_start_rel:x_stop_rel]

        output_memmap[write_slices] = result_block
        return None

    except Exception as e:
        return f"Error_chunk_{chunk_info}: {str(e)}"
    finally:
        try:
            del input_memmap
            del output_memmap
            del block_data
            del result_block
        except:
            pass
        gc.collect()


def enhance_tubular_structures_blocked(
    volume: np.ndarray,
    scales: List[float],
    spacing: Tuple[float, float, float],
    temp_root_path: Optional[str],
    black_ridges: bool = False,
    frangi_alpha: float = 0.5,
    frangi_beta: float = 0.5,
    frangi_gamma: float = 2,
    apply_3d_smoothing: bool = True,
    smoothing_sigma_phys: float = 0.5,
    skip_tubular_enhancement: bool = False,
    subtract_background_radius: int = 0
) -> Tuple[np.memmap, str, str]:
    """
    Enhances tubular structures using chunked 3D processing.
    """
    print(f"  [Enhance] Volume: {volume.shape}, Spacing: {spacing} (Block Mode)")
    
    spacing_float = tuple(float(s) for s in spacing)
    
    # Output Setup
    output_temp_dir = _get_safe_temp_dir(temp_root_path, 'tubular_output')
    output_path = os.path.join(output_temp_dir, 'processed_volume.dat')
    output_memmap = np.memmap(
        output_path, dtype=np.float32, mode='w+', shape=volume.shape
    )

    if skip_tubular_enhancement:
        print("  [Enhance] Skipping filters (Pass-through).")
        chunk_gen = _get_chunk_slices(volume.shape, (64, 512, 512), overlap=0)
        for _, write_slice in tqdm(list(chunk_gen), desc="  [Enhance] Copying"):
            output_memmap[write_slice] = volume[write_slice].astype(np.float32)
        output_memmap.flush()
        return output_memmap, output_path, output_temp_dir

    # Configuration
    sigma_voxel_3d = None
    overlap_px = 0
    if apply_3d_smoothing and smoothing_sigma_phys > 0:
        sigma_voxel_3d = tuple(
            smoothing_sigma_phys / s if s > 0 else 0 for s in spacing_float
        )
        overlap_px = math.ceil(max(sigma_voxel_3d) * 3)

    xy_spacing = spacing_float[1:]
    avg_xy_spacing = np.mean(xy_spacing)
    sigmas_voxel_2d = sorted([s / avg_xy_spacing for s in scales])
    
    max_filter_sigma = max(sigmas_voxel_2d) if sigmas_voxel_2d else 0
    overlap_px = max(overlap_px, math.ceil(max_filter_sigma * 4))
    overlap_px = max(overlap_px, 16)
    
    # Prepare Tasks
    chunk_shape = (64, 512, 512)
    chunks = list(_get_chunk_slices(volume.shape, chunk_shape, overlap=overlap_px))
    
    # Ensure Input is Disk-Backed
    if not isinstance(volume, np.memmap):
        print("  [Enhance] Dumping input to temp memmap...")
        dump_dir = _get_safe_temp_dir(temp_root_path, 'input_dump')
        dump_path = os.path.join(dump_dir, 'input_dump.dat')
        input_memmap = np.memmap(
            dump_path, dtype=volume.dtype, mode='w+', shape=volume.shape
        )
        input_memmap[:] = volume[:]
        input_memmap.flush()
        input_info = (dump_path, volume.shape, volume.dtype)
    else:
        input_info = (volume.filename, volume.shape, volume.dtype)
        # Placeholder for cleanup logic if we didn't create it
        dump_dir = None 

    output_info = (output_path, volume.shape, np.float32)

    num_workers = max(1, os.cpu_count() - 2 if os.cpu_count() > 2 else 1)
    
    worker_func = partial(
        _process_block_worker,
        input_memmap_info=input_info,
        output_memmap_info=output_info,
        sigmas_voxel_2d=sigmas_voxel_2d,
        sigma_voxel_3d=sigma_voxel_3d,
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
            desc="  [Enhance] 3D/2D Filtering"
        ))
        errors = [r for r in results if r is not None]
        if errors:
            raise RuntimeError(f"Block processing failed: {errors[0]}")
        output_memmap.flush()
    finally:
        pool.terminate()
        pool.join()
        if dump_dir and os.path.exists(dump_dir):
            shutil.rmtree(dump_dir, ignore_errors=True)
        gc.collect()

    return output_memmap, output_path, output_temp_dir


def segment_cells_first_pass_raw(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    tubular_scales: List[float] = [0.5, 1.0, 2.0, 3.0],
    smooth_sigma: float = 0.5,
    connect_max_gap_physical: float = 1.0,
    min_size_voxels: int = 50,
    low_threshold_percentile: float = 25.0,
    high_threshold_percentile: float = 95.0,
    skip_tubular_enhancement: bool = False,
    subtract_background_radius: int = 0,
    temp_root_path: Optional[str] = None,
    **kwargs: Any
) -> Tuple[Optional[str], Optional[str], float, Dict[str, Any]]:
    """
    Executes Step 1: Raw Segmentation Pipeline.
    Pipeline: Normalize -> Enhance -> Threshold.
    """
    print(f"\n--- Step 1: Raw Segmentation (Chunk-Optimized + Safe Temp) ---")

    # Warn if deprecated args used
    if 'soma_preservation_factor' in kwargs:
        # print("  [Warn] 'soma_preservation_factor' is deprecated and ignored.")
        pass

    temp_dirs_to_clean: List[str] = []
    final_labels_memmap = None
    labels_path = None
    labels_temp_dir = None
    segmentation_threshold = 0.0

    try:
        # --- Stage 1: Normalization (Reservoir) ---
        print("\n  [Step 1.1] Normalizing volume brightness...")
        normalized_temp_dir = _get_safe_temp_dir(temp_root_path, 'normalize')
        temp_dirs_to_clean.append(normalized_temp_dir)
        normalized_path = os.path.join(normalized_temp_dir, 'normalized.dat')
        normalized_memmap = np.memmap(
            normalized_path, dtype=np.float32, mode='w+', shape=volume.shape
        )

        raw_high_percentiles = []
        slice_indices = np.arange(volume.shape[0])
        
        stats_chunk_gen = _get_chunk_slices(volume.shape, (64, 512, 512), overlap=0)
        z_stats_accum: Dict[int, List[float]] = {}
        MAX_SAMPLES_PER_Z = 10000

        for read_sl, _ in tqdm(list(stats_chunk_gen), desc="    Sampling Z-profile"):
            chunk = volume[read_sl]
            if np.any(np.isfinite(chunk)) and np.max(chunk) > 0:
                sub = chunk[:, ::16, ::16]
                if sub.size > 0:
                    z_start = read_sl[0].start
                    for i in range(sub.shape[0]):
                        idx = z_start + i
                        vals = sub[i].ravel()
                        vals = vals[vals > 0]
                        if vals.size > 0:
                            if idx not in z_stats_accum: z_stats_accum[idx] = []
                            if len(z_stats_accum[idx]) < MAX_SAMPLES_PER_Z:
                                take = min(len(vals), MAX_SAMPLES_PER_Z - len(z_stats_accum[idx]))
                                z_stats_accum[idx].extend(vals[:take])

        for z in slice_indices:
            if z in z_stats_accum and len(z_stats_accum[z]) > 10:
                val = np.percentile(z_stats_accum[z], high_threshold_percentile)
                raw_high_percentiles.append(val)
            else:
                raw_high_percentiles.append(np.nan)

        raw_high_percentiles_arr = np.array(raw_high_percentiles)
        valid_indices = ~np.isnan(raw_high_percentiles_arr)
        idealized_high_percentiles = np.ones_like(raw_high_percentiles_arr)

        if np.any(valid_indices):
            poly_coeffs = np.polyfit(
                slice_indices[valid_indices],
                raw_high_percentiles_arr[valid_indices], 2
            )
            poly_func = np.poly1d(poly_coeffs)
            idealized_high_percentiles = poly_func(slice_indices)
            safe_min = np.nanpercentile(raw_high_percentiles_arr, 10)
            idealized_high_percentiles[idealized_high_percentiles < safe_min] = safe_min

        # 1b. Apply Normalization
        apply_gen = _get_chunk_slices(volume.shape, (64, 512, 512), overlap=0)
        for read_sl, _ in tqdm(list(apply_gen), desc="    Applying Normalization"):
            chunk = volume[read_sl].astype(np.float32)
            z_start = read_sl[0].start
            z_stop = read_sl[0].stop
            
            factors = idealized_high_percentiles[z_start:z_stop]
            factors = factors[:, np.newaxis, np.newaxis]
            
            factors_safe = np.where(factors > 1e-9, factors, 1.0)
            chunk_norm = chunk / factors_safe
            normalized_memmap[read_sl] = chunk_norm

        normalized_memmap.flush()

        # --- Stage 2: Feature Enhancement (Blocked) ---
        print("\n  [Step 1.2] Feature Enhancement (on Normalized Data)...")
        enhanced_memmap, _, enhanced_temp_dir = enhance_tubular_structures_blocked(
            normalized_memmap,
            scales=tubular_scales,
            spacing=spacing,
            apply_3d_smoothing=(smooth_sigma > 0),
            smoothing_sigma_phys=smooth_sigma,
            skip_tubular_enhancement=skip_tubular_enhancement,
            subtract_background_radius=subtract_background_radius,
            temp_root_path=temp_root_path
        )
        temp_dirs_to_clean.append(enhanced_temp_dir)

        # --- Stage 3: Thresholding ---
        print("\n  [Step 1.3] Calculating global threshold...")
        all_samples_list = []
        block_gen = list(_get_chunk_slices(volume.shape, (64, 512, 512), overlap=0))
        
        import random
        selected_blocks = random.sample(block_gen, max(1, len(block_gen)//10))
        
        for read_sl, _ in tqdm(selected_blocks, desc="    Sampling Histogram"):
            chunk = enhanced_memmap[read_sl]
            vals = chunk[::2, ::4, ::4].ravel()
            vals = vals[np.isfinite(vals)]
            vals = vals[vals > 1e-6] 
            if vals.size > 0:
                all_samples_list.append(vals)
        
        if all_samples_list:
            all_samples = np.concatenate(all_samples_list)
            if all_samples.size > 0:
                if all_samples.size > 5_000_000:
                    all_samples = np.random.choice(all_samples, 5_000_000, replace=False)
                segmentation_threshold = float(
                    np.percentile(all_samples, low_threshold_percentile)
                )

        print(f"    Calculated Threshold: {segmentation_threshold:.6f}")

        # Create Binary Mask
        binary_temp_dir = _get_safe_temp_dir(temp_root_path, 'binary')
        temp_dirs_to_clean.append(binary_temp_dir)
        binary_path = os.path.join(binary_temp_dir, 'b.dat')
        binary_memmap = np.memmap(
            binary_path, dtype=bool, mode='w+', shape=volume.shape
        )

        for read_sl, _ in tqdm(block_gen, desc="    Creating Binary Mask"):
            chunk_enh = enhanced_memmap[read_sl]
            chunk_bin = (chunk_enh > segmentation_threshold)
            binary_memmap[read_sl] = chunk_bin
        
        binary_memmap.flush()

        del normalized_memmap, enhanced_memmap
        gc.collect()

        # --- Stage 4: Dask Cleaning ---
        print("\n  [Step 1.4] Global Cleaning (Dask)...")
        dask_chunk_size = (128, 512, 512) 

        connected_temp_dir = _get_safe_temp_dir(temp_root_path, 'connected')
        temp_dirs_to_clean.append(connected_temp_dir)
        connected_path = os.path.join(connected_temp_dir, 'connected.dat')
        connected_memmap = np.memmap(
            connected_path, dtype=bool, mode='w+', shape=volume.shape
        )

        binary_dask = da.from_array(binary_memmap, chunks=dask_chunk_size)

        radius_vox = [
            math.ceil((connect_max_gap_physical / 2) / s) if s > 1e-9 else 0
            for s in spacing
        ]
        structure = None
        if any(r > 0 for r in radius_vox):
            structure = np.ones(tuple(max(1, 2 * r + 1) for r in radius_vox), dtype=bool)

        if structure is not None:
            connected_dask = dask_image.ndmorph.binary_closing(
                binary_dask, structure=structure
            )
        else:
            connected_dask = binary_dask

        print("    Applying binary closing...")
        with ProgressBar(dt=5):
            da.store(connected_dask, connected_memmap, scheduler='threads')

        del binary_dask, connected_dask, binary_memmap
        gc.collect()

        # --- Stage 5: Labeling ---
        print("    Labeling connected components...")
        labeled_temp_dir = _get_safe_temp_dir(temp_root_path, 'labeled_zarr')
        temp_dirs_to_clean.append(labeled_temp_dir)
        labeled_zarr_path = os.path.join(labeled_temp_dir, 'labeled.zarr')

        connected_dask = da.from_array(connected_memmap, chunks=dask_chunk_size)
        s = generate_binary_structure(3, 1)
        labeled_dask, num_features_dask = dask_image.ndmeasure.label(
            connected_dask, structure=s
        )

        with ProgressBar(dt=5):
            labeled_dask.to_zarr(labeled_zarr_path, overwrite=True)

        num_features = num_features_dask.compute()
        print(f"      Found {num_features} objects.")

        del connected_dask, labeled_dask
        gc.collect()

        # --- Stage 6: Filtering ---
        print(f"    Filtering objects < {min_size_voxels} voxels...")
        labels_temp_dir = _get_safe_temp_dir(temp_root_path, 'labels_final')
        labels_path = os.path.join(labels_temp_dir, 'l.dat')
        final_labels_memmap = np.memmap(
            labels_path, dtype=np.int32, mode='w+', shape=volume.shape
        )

        initial_labeled_zarr = zarr.open(labeled_zarr_path, mode='r')

        print("      Calculating object sizes...")
        d_lbl = da.from_zarr(labeled_zarr_path)
        counts, bins = da.histogram(
            d_lbl, bins=num_features + 1, range=[-0.5, num_features + 0.5]
        )
        counts_res = counts.compute()
        
        valid_indices = np.where(counts_res[1:] >= min_size_voxels)[0] + 1
        print(f"      Retaining {len(valid_indices)} / {num_features} objects.")
        
        lookup = np.zeros(num_features + 1, dtype=np.int32)
        current_new = 1
        for old_id in valid_indices:
            lookup[old_id] = current_new
            current_new += 1
            
        print("      Writing filtered labels (Manual Block-wise)...")
        io_chunks = (64, 512, 512)
        io_gen = _get_chunk_slices(volume.shape, io_chunks, overlap=0)
        
        for read_sl, write_sl in tqdm(list(io_gen), desc="      Remapping"):
            block_data = initial_labeled_zarr[read_sl]
            filtered_block = lookup[block_data]
            final_labels_memmap[write_sl] = filtered_block
            
        final_labels_memmap.flush()

        first_pass_params = {
            'spacing': tuple(float(s) for s in spacing),
            'tubular_scales': tubular_scales,
            'smooth_sigma': smooth_sigma,
            'connect_max_gap_physical': connect_max_gap_physical,
            'min_size_voxels': min_size_voxels,
            'low_threshold_percentile': low_threshold_percentile,
            'high_threshold_percentile': high_threshold_percentile,
            'original_shape': volume.shape,
            'skip_tubular_enhancement': skip_tubular_enhancement,
            'subtract_background_radius': subtract_background_radius
        }
        return labels_path, labels_temp_dir, segmentation_threshold, first_pass_params

    finally:
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