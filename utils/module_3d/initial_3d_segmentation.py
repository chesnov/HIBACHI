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
    """Creates a temporary directory strictly inside the project temp folder."""
    if base_path and os.path.isdir(base_path):
        scratch_root = base_path # Use the project's temp_artifacts folder
    else:
        scratch_root = os.path.join(tempfile.gettempdir(), "hibachi_scratch")
    
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
    black_ridges: bool,
    frangi_alpha: float,
    frangi_beta: float,
    frangi_gamma: float,
    subtract_background_radius: int
) -> Optional[str]:
    """
    Worker function for processing a 3D block (Enhancement Step).
    Performs Scale-Independent Vesselness (Frangi/Sato) Filtering.
    """
    input_memmap = None
    output_memmap = None
    try:
        read_slices, write_slices = chunk_info
        input_path, input_shape, input_dtype = input_memmap_info
        output_path, output_shape, output_dtype = output_memmap_info

        input_memmap = np.memmap(input_path, dtype=input_dtype, mode='r', shape=input_shape)
        output_memmap = np.memmap(output_path, dtype=output_dtype, mode='r+', shape=output_shape)

        block_data = input_memmap[read_slices].astype(np.float32)

        # Indices to crop padding
        z_start_rel = write_slices[0].start - read_slices[0].start
        y_start_rel = write_slices[1].start - read_slices[1].start
        x_start_rel = write_slices[2].start - read_slices[2].start
        
        z_stop_rel = z_start_rel + (write_slices[0].stop - write_slices[0].start)
        y_stop_rel = y_start_rel + (write_slices[1].stop - write_slices[1].start)
        x_stop_rel = x_start_rel + (write_slices[2].stop - write_slices[2].start)

        valid_shape = (z_stop_rel - z_start_rel, y_stop_rel - y_start_rel, x_stop_rel - x_start_rel)
        result_block = np.zeros(valid_shape, dtype=np.float32)

        for z in range(block_data.shape[0]):
            if z < z_start_rel or z >= z_stop_rel:
                continue

            slice_2d = block_data[z]
            if subtract_background_radius > 0:
                struct_size = int(2 * subtract_background_radius + 1)
                slice_2d = white_tophat(slice_2d, size=struct_size)

            combined_scales = np.zeros_like(slice_2d, dtype=np.float32)
            
            for sigma in sigmas_voxel_2d:
                # Strictly Vesselness (Frangi/Sato)
                beta_val = 1.0 if sigma >= 2.0 else frangi_beta
                f_res = frangi(slice_2d, sigmas=[sigma], alpha=frangi_alpha, 
                               beta=beta_val, gamma=frangi_gamma, black_ridges=black_ridges)
                s_res = sato(slice_2d, sigmas=[sigma], black_ridges=black_ridges)
                scale_res = np.maximum(f_res, s_res)
                
                if sigma >= 2.0:
                    scale_res *= slice_2d
                
                combined_scales = np.maximum(combined_scales, scale_res)

            result_block[z - z_start_rel] = \
                combined_scales[y_start_rel:y_stop_rel, x_start_rel:x_stop_rel]

        output_memmap[write_slices] = result_block
        return None

    except Exception as e:
        return f"Error_chunk_{chunk_info}: {str(e)}"
    finally:
        try:
            del input_memmap, output_memmap, block_data, result_block
        except: pass
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
    skip_tubular_enhancement: bool = False,
    subtract_background_radius: int = 0
) -> Tuple[np.memmap, str, str]:
    """
    Enhances structures using chunked 3D processing (Vesselness Mode).
    Note: smoothing is now expected to be done externally for independence.
    """
    print(f"  [Enhance] Volume: {volume.shape}, Spacing: {spacing}")
    spacing_float = tuple(float(s) for s in spacing)
    
    output_temp_dir = _get_safe_temp_dir(temp_root_path, 'tubular_output')
    output_path = os.path.join(output_temp_dir, 'processed_volume.dat')
    output_memmap = np.memmap(output_path, dtype=np.float32, mode='w+', shape=volume.shape)

    if skip_tubular_enhancement:
        chunk_gen = _get_chunk_slices(volume.shape, (64, 512, 512), overlap=0)
        for _, write_slice in tqdm(list(chunk_gen), desc="  [Enhance] Copying"):
            output_memmap[write_slice] = volume[write_slice].astype(np.float32)
        output_memmap.flush()
        return output_memmap, output_path, output_temp_dir

    xy_spacing = spacing_float[1:]
    sigmas_voxel_2d = sorted([s / np.mean(xy_spacing) for s in scales if s > 0])
    if not sigmas_voxel_2d:
        return enhance_tubular_structures_blocked(volume, [], spacing, temp_root_path, skip_tubular_enhancement=True)

    overlap_px = max(16, math.ceil(max(sigmas_voxel_2d) * 4))
    
    input_info = (volume.filename, volume.shape, volume.dtype) if isinstance(volume, np.memmap) else None
    dump_dir = None
    if input_info is None:
        dump_dir = _get_safe_temp_dir(temp_root_path, 'input_dump')
        dump_path = os.path.join(dump_dir, 'input_dump.dat')
        input_mm = np.memmap(dump_path, dtype=volume.dtype, mode='w+', shape=volume.shape)
        input_mm[:] = volume[:]
        input_mm.flush()
        input_info = (dump_path, volume.shape, volume.dtype)

    worker_func = partial(_process_block_worker, input_memmap_info=input_info, 
                          output_memmap_info=(output_path, volume.shape, np.float32),
                          sigmas_voxel_2d=sigmas_voxel_2d, 
                          black_ridges=black_ridges, frangi_alpha=frangi_alpha, 
                          frangi_beta=frangi_beta, frangi_gamma=frangi_gamma,
                          subtract_background_radius=subtract_background_radius)

    chunks = list(_get_chunk_slices(volume.shape, (64, 512, 512), overlap=overlap_px))
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
    """Step 1: Raw Segmentation (Independent Smoothing + Threshold-then-OR)."""
    print(f"\n--- Step 1: Raw Segmentation (Strict Independence Mode) ---")
    temp_dirs_to_clean, threshold_history = [], {}
    final_labels_memmap = None

    try:
        # --- Stage 1.1: Normalization ---
        with SimpleTimer("Stage 1.1: Normalization"):
            norm_dir = _get_safe_temp_dir(temp_root_path, 'normalize'); temp_dirs_to_clean.append(norm_dir)
            norm_path = os.path.join(norm_dir, 'norm.dat')
            norm_mm = np.memmap(norm_path, dtype=np.float32, mode='w+', shape=volume.shape)
            
            z_stats = {}
            for read_sl, _ in tqdm(list(_get_chunk_slices(volume.shape, (64, 512, 512))), desc="    Sampling"):
                sub = volume[read_sl][:, ::16, ::16]
                for i in range(sub.shape[0]):
                    vals = sub[i].ravel(); vals = vals[vals > 0]
                    if vals.size > 0:
                        idx = read_sl[0].start + i
                        if idx not in z_stats: z_stats[idx] = []
                        z_stats[idx].extend(vals[:5000])
            
            ideal = np.ones(volume.shape[0])
            z_indices = np.arange(volume.shape[0])
            hp = np.array([np.percentile(z_stats[z], high_threshold_percentile) if z in z_stats else np.nan for z in z_indices])
            if np.any(~np.isnan(hp)):
                valid = ~np.isnan(hp); p = np.poly1d(np.polyfit(z_indices[valid], hp[valid], 2))
                ideal = np.maximum(p(z_indices), np.nanpercentile(hp, 10))

            for read_sl, _ in tqdm(list(_get_chunk_slices(volume.shape, (64, 512, 512))), desc="    Applying"):
                factors = ideal[read_sl[0].start:read_sl[0].stop][:, np.newaxis, np.newaxis]
                norm_mm[read_sl] = volume[read_sl].astype(np.float32) / np.where(factors > 1e-9, factors, 1.0)
            norm_mm.flush()

        # --- Stage 1.2: Independent 3D Smoothing ---
        # This makes smooth_sigma a fixed preprocessing step for all scales (including scale=0)
        smoothed_mm = norm_mm
        if smooth_sigma > 0:
            with SimpleTimer(f"Stage 1.2: Global 3D Smoothing (sigma={smooth_sigma})"):
                smooth_dir = _get_safe_temp_dir(temp_root_path, 'smoothing'); temp_dirs_to_clean.append(smooth_dir)
                smooth_path = os.path.join(smooth_dir, 'smoothed.dat')
                smoothed_mm = np.memmap(smooth_path, dtype=np.float32, mode='w+', shape=volume.shape)
                
                # Use Dask for global smoothing
                sigma_vox = [smooth_sigma / s if s > 0 else 0 for s in spacing]
                d_norm = da.from_array(norm_mm, chunks=(128, 512, 512))
                d_smooth = dask_image.ndfilters.gaussian_filter(d_norm, sigma=sigma_vox)
                
                with ProgressBar(dt=5):
                    da.store(d_smooth, smoothed_mm, scheduler='threads')
                smoothed_mm.flush()

        # --- Stage 2 & 3: Multi-Scale Logic ---
        master_dir = _get_safe_temp_dir(temp_root_path, 'master'); temp_dirs_to_clean.append(master_dir)
        master_mm = np.memmap(os.path.join(master_dir, 'm.dat'), dtype=np.uint8, mode='w+', shape=volume.shape)
        master_mm[:] = 0

        # Closing structure for masks
        rv = [math.ceil((connect_max_gap_physical/2)/s) if s>1e-9 else 0 for s in spacing]
        struct = np.ones(tuple(max(1, 2*r+1) for r in rv), dtype=bool)

        for scale in tubular_scales:
            with SimpleTimer(f"Scale sigma={scale}"):
                if scale == 0:
                    # Pass-through: uses the same smoothed intensity data
                    enh_mm = smoothed_mm
                    enh_dir = None
                else:
                    # Vesselness: enhancement run on the same smoothed intensity data
                    enh_mm, _, enh_dir = enhance_tubular_structures_blocked(
                        smoothed_mm, scales=[scale], spacing=spacing,
                        skip_tubular_enhancement=skip_tubular_enhancement,
                        subtract_background_radius=subtract_background_radius, temp_root_path=temp_root_path
                    )
                
                # Independent Thresholding Pass
                samples = enh_mm[::4, ::16, ::16].ravel(); samples = samples[samples > 1e-7]
                thresh = float(np.percentile(samples, low_threshold_percentile)) if samples.size > 1000 else 1e9
                thresh = max(thresh, 1e-5); threshold_history[scale] = thresh
                print(f"      [Scale {scale}] Isolated Threshold: {thresh:.6f}")

                # Binary Creation, Closing, and OR-ing
                if thresh < 1e6:
                    enh_dask = da.from_array(enh_mm, chunks=(128, 512, 512))
                    clean_dask = dask_image.ndmorph.binary_closing((enh_dask > thresh), structure=struct)
                    
                    for read_sl, _ in tqdm(list(_get_chunk_slices(volume.shape, (64, 512, 512))), desc="      Merging"):
                        master_mm[read_sl] |= clean_dask[read_sl].compute().astype(np.uint8)
                
                master_mm.flush()
                if enh_dir:
                    del enh_mm; shutil.rmtree(enh_dir, ignore_errors=True); gc.collect()

        # Cleanup intermediate smoothed volume
        del smoothed_mm, norm_mm; gc.collect()

        # --- Labeling and Size Filtering ---
        print("\n  [Step 1.4] Labeling Objects...")
        final_dir = _get_safe_temp_dir(temp_root_path, 'final'); labels_temp_dir = final_dir
        labels_path = os.path.join(final_dir, 'l.dat')
        final_mm = np.memmap(labels_path, dtype=np.int32, mode='w+', shape=volume.shape)
        
        lab_dir = _get_safe_temp_dir(temp_root_path, 'lab_zarr'); temp_dirs_to_clean.append(lab_dir)
        m_dask = da.from_array(master_mm, chunks=(128, 512, 512))
        labeled_dask, num_feats_dask = dask_image.ndmeasure.label((m_dask > 0), structure=generate_binary_structure(3, 1))
        labeled_dask.to_zarr(os.path.join(lab_dir, 'l.zarr'), overwrite=True)
        num_feats = num_feats_dask.compute()

        d_lbl = da.from_zarr(os.path.join(lab_dir, 'l.zarr'))
        counts, _ = da.histogram(d_lbl, bins=num_feats+1, range=[-0.5, num_feats+0.5])
        valid = np.where(counts.compute()[1:] >= min_size_voxels)[0] + 1
        
        lookup = np.zeros(num_feats + 1, dtype=np.int32)
        for i, old_id in enumerate(valid): lookup[old_id] = i + 1
            
        lz = zarr.open(os.path.join(lab_dir, 'l.zarr'), mode='r')
        for rs, ws in tqdm(list(_get_chunk_slices(volume.shape, (64, 512, 512))), desc="    Filtering"):
            final_mm[ws] = lookup[lz[rs]]
        final_mm.flush()

        # Explicitly release large internal memmaps
        if 'master_mm' in locals():
            del master_mm

        return labels_path, labels_temp_dir, threshold_history.get(tubular_scales[0], 0.0), {'threshold_history': threshold_history}

    finally:
        if final_labels_memmap is not None: del final_labels_memmap
        # Close any local memmap handles to release file locks
        for var in ['final_mm', 'norm_mm', 'smoothed_mm', 'master_mm', 'input_mm', 'enh_mm']:
            if var in locals():
                try: del locals()[var]
                except: pass
                
        for d in temp_dirs_to_clean:
            shutil.rmtree(d, ignore_errors=True)
        gc.collect()