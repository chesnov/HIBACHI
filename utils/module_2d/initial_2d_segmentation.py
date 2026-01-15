import os
import gc
import math
import time
import shutil
import tempfile
import multiprocessing as mp
from functools import partial
from typing import Tuple, List, Dict, Any, Optional, Union, Generator

import numpy as np
import zarr
import dask.array as da
import dask_image.ndmorph
import dask_image.ndfilters
import dask_image.ndmeasure
from dask.diagnostics import ProgressBar
from scipy import ndimage
from scipy.ndimage import generate_binary_structure, white_tophat
from skimage.filters import frangi, sato  # type: ignore
from skimage.morphology import disk  # type: ignore
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
    """Creates a temporary directory in the project folder."""
    scratch_root = base_path if (base_path and os.path.isdir(base_path)) else tempfile.gettempdir()
    
    return tempfile.mkdtemp(prefix=f"step1_2d_{suffix}_", dir=scratch_root)


def _get_chunk_slices_2d(
    shape: Tuple[int, ...],
    chunk_shape: Tuple[int, ...],
    overlap: int = 0
) -> Generator[Tuple[Tuple[slice, ...], Tuple[slice, ...]], None, None]:
    """Generates read/write slices for 2D chunked processing."""
    y_shape, x_shape = shape
    cy, cx = chunk_shape

    for y in range(0, y_shape, cy):
        for x in range(0, x_shape, cx):
            y_start, y_stop = y, min(y + cy, y_shape)
            x_start, x_stop = x, min(x + cx, x_shape)
            
            write_slice = (slice(y_start, y_stop), slice(x_start, x_stop))

            y_start_pad = max(0, y_start - overlap)
            y_stop_pad = min(y_shape, y_stop + overlap)
            x_start_pad = max(0, x_start - overlap)
            x_stop_pad = min(x_shape, x_stop + overlap)

            read_slice = (slice(y_start_pad, y_stop_pad), slice(x_start_pad, x_stop_pad))

            yield read_slice, write_slice


def _process_block_worker_2d(
    chunk_info: Tuple[Tuple[slice, ...], Tuple[slice, ...]],
    input_memmap_info: Tuple[str, Tuple[int, ...], Any],
    output_memmap_info: Tuple[str, Tuple[int, ...], Any],
    sigmas_voxel_2d: List[float],
    black_ridges: bool,
    frangi_alpha: float,
    frangi_beta: float,
    frangi_gamma: float,
    subtract_background_radius: int
) -> Optional[str]:
    """Worker function for processing a 2D block (Vesselness Mode)."""
    input_memmap = None
    output_memmap = None
    try:
        read_slices, write_slices = chunk_info
        input_path, input_shape, input_dtype = input_memmap_info
        output_path, output_shape, output_dtype = output_memmap_info

        input_memmap = np.memmap(input_path, dtype=input_dtype, mode='r', shape=input_shape)
        output_memmap = np.memmap(output_path, dtype=output_dtype, mode='r+', shape=output_shape)

        block_data = input_memmap[read_slices].astype(np.float32)

        if subtract_background_radius > 0:
            struct_size = int(2 * subtract_background_radius + 1)
            block_data = white_tophat(block_data, size=struct_size)

        combined_scales = np.zeros_like(block_data, dtype=np.float32)

        for sigma in sigmas_voxel_2d:
            beta_val = 1.0 if sigma >= 2.0 else frangi_beta
            f_res = frangi(block_data, sigmas=[sigma], alpha=frangi_alpha, 
                           beta=beta_val, gamma=frangi_gamma, black_ridges=black_ridges)
            s_res = sato(block_data, sigmas=[sigma], black_ridges=black_ridges)
            scale_res = np.maximum(f_res, s_res)
            
            if sigma >= 2.0:
                scale_res *= block_data
            
            combined_scales = np.maximum(combined_scales, scale_res)

        y_start_rel = write_slices[0].start - read_slices[0].start
        y_stop_rel = y_start_rel + (write_slices[0].stop - write_slices[0].start)
        x_start_rel = write_slices[1].start - read_slices[1].start
        x_stop_rel = x_start_rel + (write_slices[1].stop - write_slices[1].start)

        output_memmap[write_slices] = combined_scales[y_start_rel:y_stop_rel, x_start_rel:x_stop_rel]
        return None

    except Exception as e:
        return f"Error_chunk_{chunk_info}: {str(e)}"
    finally:
        try:
            del input_memmap, output_memmap, block_data, combined_scales
        except: pass
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
    skip_enhancement: bool = False,
    subtract_background_radius: int = 0
) -> Tuple[np.memmap, str, str]:
    """Enhances tubular structures using 2D chunked processing (Vesselness Mode)."""
    print(f"  [Enhance] Image: {image.shape}, Spacing: {spacing}")
    
    output_temp_dir = _get_safe_temp_dir(temp_root_path, 'tubular_output')
    output_path = os.path.join(output_temp_dir, 'processed_image.dat')
    output_memmap = np.memmap(output_path, dtype=np.float32, mode='w+', shape=image.shape)

    if skip_enhancement:
        chunk_gen = _get_chunk_slices_2d(image.shape, (2048, 2048), overlap=0)
        for _, write_slice in tqdm(list(chunk_gen), desc="  [Enhance] Copying"):
            output_memmap[write_slice] = image[write_slice].astype(np.float32)
        output_memmap.flush()
        return output_memmap, output_path, output_temp_dir

    avg_spacing = np.mean(spacing)
    sigmas_voxel_2d = sorted([s / avg_spacing for s in scales if s > 0])
    if not sigmas_voxel_2d:
        return enhance_tubular_structures_blocked_2d(image, [], spacing, temp_root_path, skip_enhancement=True)

    overlap_px = max(32, math.ceil(max(sigmas_voxel_2d) * 4))
    
    input_info = (image.filename, image.shape, image.dtype) if isinstance(image, np.memmap) else None
    dump_dir = None
    if input_info is None:
        dump_dir = _get_safe_temp_dir(temp_root_path, 'input_dump')
        dump_path = os.path.join(dump_dir, 'input_dump.dat')
        input_mm = np.memmap(dump_path, dtype=image.dtype, mode='w+', shape=image.shape)
        input_mm[:] = image[:]
        input_mm.flush()
        input_info = (dump_path, image.shape, image.dtype)

    worker_func = partial(_process_block_worker_2d, input_memmap_info=input_info, 
                          output_memmap_info=(output_path, image.shape, np.float32),
                          sigmas_voxel_2d=sigmas_voxel_2d, black_ridges=black_ridges, 
                          frangi_alpha=frangi_alpha, frangi_beta=frangi_beta, 
                          frangi_gamma=frangi_gamma, subtract_background_radius=subtract_background_radius)

    chunks = list(_get_chunk_slices_2d(image.shape, (2048, 2048), overlap=overlap_px))
    pool = mp.Pool(processes=max(1, os.cpu_count()-2), initializer=_init_worker)
    try:
        results = list(tqdm(pool.imap_unordered(worker_func, chunks), total=len(chunks), desc="  [Enhance] Vessel Filters"))
        if any(r is not None for r in results): raise RuntimeError(f"Error: {next(r for r in results if r)}")
        output_memmap.flush()
    finally:
        pool.terminate(); pool.join()
        if dump_dir: shutil.rmtree(dump_dir, ignore_errors=True)
        gc.collect()

    return output_memmap, output_path, output_temp_dir


class SimpleTimer:
    def __init__(self, name: str): self.name = name
    def __enter__(self): 
        self.start = time.perf_counter()
        print(f"    [Timer] Starting: {self.name}..."); return self
    def __exit__(self, *args):
        print(f"    [Timer] Finished: {self.name} in {time.perf_counter()-self.start:.2f}s")


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
    """Step 1: Raw 2D Segmentation (Independent Smoothing + Threshold-then-OR)."""
    print(f"\n--- Step 1: Raw 2D Segmentation (Strict Independence Mode) ---")
    temp_dirs_to_clean, threshold_history = [], {}
    final_labels_memmap = None

    try:
        spacing_2d = tuple(float(s) for s in spacing[-2:])

        # --- Normalization ---
        with SimpleTimer("Stage 1.1: Normalization"):
            norm_dir = _get_safe_temp_dir(temp_root_path, 'normalize'); temp_dirs_to_clean.append(norm_dir)
            norm_path = os.path.join(norm_dir, 'norm.dat')
            norm_mm = np.memmap(norm_path, dtype=np.float32, mode='w+', shape=image.shape)
            
            # Sampling global high percentile
            samples = image[::8, ::8].ravel(); samples = samples[samples > 0]
            high_val = np.percentile(samples, high_threshold_percentile) if samples.size > 0 else 1.0
            high_val = max(high_val, 1e-9)
            print(f"    Normalization Max (p{high_threshold_percentile}): {high_val:.2f}")

            chunk_gen = list(_get_chunk_slices_2d(image.shape, (2048, 2048), overlap=0))
            for read_sl, _ in tqdm(chunk_gen, desc="    Applying"):
                norm_mm[read_sl] = image[read_sl].astype(np.float32) / high_val
            norm_mm.flush()

        # --- Global 2D Smoothing (Preprocessing) ---
        smoothed_mm = norm_mm
        if smooth_sigma > 0:
            with SimpleTimer(f"Stage 1.2: Global 2D Smoothing (sigma={smooth_sigma})"):
                smooth_dir = _get_safe_temp_dir(temp_root_path, 'smoothing'); temp_dirs_to_clean.append(smooth_dir)
                smooth_path = os.path.join(smooth_dir, 'smoothed.dat')
                smoothed_mm = np.memmap(smooth_path, dtype=np.float32, mode='w+', shape=image.shape)
                
                sigma_vox = [smooth_sigma / s if s > 0 else 0 for s in spacing_2d]
                d_norm = da.from_array(norm_mm, chunks=(4096, 4096))
                d_smooth = dask_image.ndfilters.gaussian_filter(d_norm, sigma=sigma_vox)
                
                with ProgressBar(dt=5):
                    da.store(d_smooth, smoothed_mm, scheduler='threads')
                smoothed_mm.flush()

        # --- Multi-Scale Logic (Threshold-then-OR) ---
        master_dir = _get_safe_temp_dir(temp_root_path, 'master'); temp_dirs_to_clean.append(master_dir)
        master_mm = np.memmap(os.path.join(master_dir, 'm.dat'), dtype=np.uint8, mode='w+', shape=image.shape)
        master_mm[:] = 0

        # Closing structure
        radius_px = math.ceil((connect_max_gap_physical / 2) / np.mean(spacing_2d))
        struct = disk(radius_px) if radius_px > 0 else np.ones((1,1), dtype=bool)

        for scale in tubular_scales:
            with SimpleTimer(f"Scale sigma={scale}"):
                if scale == 0:
                    enh_mm = smoothed_mm
                    enh_dir = None
                else:
                    enh_mm, _, enh_dir = enhance_tubular_structures_blocked_2d(
                        smoothed_mm, scales=[scale], spacing=spacing_2d,
                        skip_enhancement=skip_tubular_enhancement,
                        subtract_background_radius=subtract_background_radius, temp_root_path=temp_root_path
                    )
                
                # Independent Thresholding
                samples = enh_mm[::16, ::16].ravel(); samples = samples[samples > 1e-7]
                thresh = float(np.percentile(samples, low_threshold_percentile)) if samples.size > 1000 else 1e9
                thresh = max(thresh, 1e-5); threshold_history[scale] = thresh
                print(f"      [Scale {scale}] Isolated Threshold: {thresh:.6f}")

                if thresh < 1e6:
                    enh_dask = da.from_array(enh_mm, chunks=(4096, 4096))
                    clean_dask = dask_image.ndmorph.binary_closing((enh_dask > thresh), structure=struct)
                    
                    chunk_gen = list(_get_chunk_slices_2d(image.shape, (2048, 2048), overlap=0))
                    for read_sl, _ in tqdm(chunk_gen, desc="      Merging"):
                        master_mm[read_sl] |= clean_dask[read_sl].compute().astype(np.uint8)
                
                master_mm.flush()
                if enh_dir:
                    del enh_mm; shutil.rmtree(enh_dir, ignore_errors=True); gc.collect()

        del smoothed_mm, norm_mm; gc.collect()

        # --- Labeling and Size Filtering ---
        print("\n  [Step 1.4] Labeling Objects...")
        final_dir = _get_safe_temp_dir(temp_root_path, 'final'); labels_temp_dir = final_dir
        labels_path = os.path.join(final_dir, 'l.dat')
        final_mm = np.memmap(labels_path, dtype=np.int32, mode='w+', shape=image.shape)
        
        lab_dir = _get_safe_temp_dir(temp_root_path, 'lab_zarr'); temp_dirs_to_clean.append(lab_dir)
        m_dask = da.from_array(master_mm, chunks=(4096, 4096))
        labeled_dask, num_feats_dask = dask_image.ndmeasure.label((m_dask > 0), structure=generate_binary_structure(2, 1))
        labeled_dask.to_zarr(os.path.join(lab_dir, 'l.zarr'), overwrite=True)
        num_feats = num_feats_dask.compute()

        d_lbl = da.from_zarr(os.path.join(lab_dir, 'l.zarr'))
        counts, _ = da.histogram(d_lbl, bins=num_feats+1, range=[-0.5, num_feats+0.5])
        valid = np.where(counts.compute()[1:] >= min_size_pixels)[0] + 1
        
        lookup = np.zeros(num_feats + 1, dtype=np.int32)
        for i, old_id in enumerate(valid): lookup[old_id] = i + 1
            
        lz = zarr.open(os.path.join(lab_dir, 'l.zarr'), mode='r')
        chunk_gen = list(_get_chunk_slices_2d(image.shape, (2048, 2048), overlap=0))
        for rs, ws in tqdm(chunk_gen, desc="    Filtering"):
            final_mm[ws] = lookup[lz[rs]]
        final_mm.flush()

        # Explicitly release master_mm before returning
        if 'master_mm' in locals():
            del master_mm

        return labels_path, labels_temp_dir, threshold_history.get(tubular_scales[0], 0.0), {'threshold_history': threshold_history}

    finally:
        # Close any local memmap handles to avoid PermissionError on cleanup
        for var in ['final_mm', 'norm_mm', 'smoothed_mm', 'master_mm', 'input_mm']:
            if var in locals():
                del locals()[var]
        
        for d in temp_dirs_to_clean:
            shutil.rmtree(d, ignore_errors=True)
        gc.collect()