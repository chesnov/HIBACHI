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


def _orientation_coherence(smoothed_mm, spacing, scale, chunks):
    """Structure-tensor orientation coherence in [0, 1].

    0 = locally isotropic (texture / noise / flat); →1 = a single dominant
    local gradient orientation (an edge or ridge flank). A general image
    operator with no model of what is imaged: it is computed from the
    gradient-covariance (structure) tensor and normalised fractional-anisotropy
    style, so it is invariant to intensity scale. Dimension-generic — derives
    ndim from the array — so 2D and 3D share identical logic. Fully elementwise
    on dask arrays, so it is chunk-safe on large memmaps.
    """
    ndim = smoothed_mm.ndim
    n = float(ndim)
    d = da.from_array(smoothed_mm, chunks=chunks).astype(np.float32)

    # Derivative (noise) and integration scales, in voxels, tied to `scale`.
    sigma_grad = [max(1.0, 0.5 * scale / s) if s > 1e-9 else 1.0 for s in spacing]
    rho = [max(1.5, 2.0 * scale / s) if s > 1e-9 else 1.5 for s in spacing]

    sm = dask_image.ndfilters.gaussian_filter(d, sigma=sigma_grad)
    grads = da.gradient(sm)
    if ndim == 1:
        grads = [grads]

    # Structure-tensor components, integrated (smoothed) at scale rho.
    diag = [dask_image.ndfilters.gaussian_filter(g * g, sigma=rho) for g in grads]
    trace = diag[0]
    jnorm2 = diag[0] * diag[0]
    for dd in diag[1:]:
        trace = trace + dd
        jnorm2 = jnorm2 + dd * dd
    for a in range(ndim):
        for b in range(a + 1, ndim):
            jab = dask_image.ndfilters.gaussian_filter(grads[a] * grads[b], sigma=rho)
            jnorm2 = jnorm2 + 2.0 * (jab * jab)

    # Fractional-anisotropy-style coherence from tensor invariants (no explicit
    # eigen-decomposition): ||J_dev||^2 = ||J||^2 - trace^2 / n.
    eps = 1e-12
    dev2 = da.maximum(jnorm2 - (trace * trace) / n, 0.0)
    coh = da.sqrt((n / (n - 1.0)) * dev2 / (jnorm2 + eps))
    return da.clip(coh, 0.0, 1.0)


def segment_cells_first_pass_raw_2d(
    image: np.ndarray,
    spacing: Union[Tuple[float, float], Tuple[float, float, float]],
    tubular_scales: List[float] = [0.5, 1.0, 2.0, 3.0],
    smooth_sigma: Union[float, List[float]] = 0.5,
    connect_max_gap_physical: Union[float, List[float]] = 1.0,
    min_size_pixels: Union[int, List[int]] = 50,
    low_threshold_percentile: Union[float, List[float]] = 95.0,
    high_threshold_percentile: Union[float, List[float]] = 100.0,
    seed_threshold: Union[float, List[float]] = 0.0,
    threshold_mode: str = "Percentile",
    skip_tubular_enhancement: bool = False,
    subtract_background_radius: int = 0,
    coherence_floor: float = 0.0,
    temp_root_path: Optional[str] = None,
    **kwargs: Any
) -> Tuple[Optional[str], Optional[str], float, Dict[str, Any]]:
    """
    Step 1: Raw 2D Segmentation (Independent per-scale Smoothing/Gap +
    Threshold-then-OR, GLOBAL size filter).

    Each tubularity scale in `tubular_scales` is smoothed, enhanced, thresholded
    and gap-closed independently, then OR-merged. `smooth_sigma` and
    `connect_max_gap_physical` are per-scale (scalar broadcasts to all scales;
    a list gives one entry per scale).

    Unlike per-scale filtering, the minimum-size filter is applied GLOBALLY,
    once, AFTER the merge (matching the 3D pipeline), so `min_size_pixels` is a
    single value. A list is accepted for backward compatibility and collapsed to
    its smallest (most permissive) entry.
    """
    n_scales = len(tubular_scales)

    if n_scales == 0:
        raise ValueError(
            "tubular_scales must contain at least one scale; got an empty "
            "list. Check the scale-profile table upstream."
        )

    if isinstance(low_threshold_percentile, (int, float)):
        low_thresh_list = [float(low_threshold_percentile)] * n_scales
    else:
        low_thresh_list = [float(x) for x in low_threshold_percentile]

    if isinstance(high_threshold_percentile, (int, float)):
        high_thresh_list = [float(high_threshold_percentile)] * n_scales
    else:
        high_thresh_list = [float(x) for x in high_threshold_percentile]

    if len(low_thresh_list) != n_scales or len(high_thresh_list) != n_scales:
        raise ValueError("low/high_threshold_percentile lists must match length of tubular_scales.")

    # Per-scale hysteresis seed (Absolute mode; 0 = off). Scalar broadcasts.
    if isinstance(seed_threshold, (int, float)):
        seed_thresh_list = [float(seed_threshold)] * n_scales
    else:
        seed_thresh_list = [float(x) for x in seed_threshold]
    if len(seed_thresh_list) != n_scales:
        raise ValueError("seed_threshold list must match length of tubular_scales.")

    # --- Per-scale filter parameters ---
    # Each entry below applies ONLY to its corresponding tubular scale.
    if isinstance(smooth_sigma, (int, float)):
        smooth_sigma_list = [float(smooth_sigma)] * n_scales
    else:
        smooth_sigma_list = [float(x) for x in smooth_sigma]

    if isinstance(connect_max_gap_physical, (int, float)):
        connect_gap_list = [float(connect_max_gap_physical)] * n_scales
    else:
        connect_gap_list = [float(x) for x in connect_max_gap_physical]

    if (len(smooth_sigma_list) != n_scales
            or len(connect_gap_list) != n_scales):
        raise ValueError(
            "smooth_sigma/connect_max_gap_physical lists "
            "must match length of tubular_scales."
        )

    # Minimum-size filtering is GLOBAL (applied once, after all scales merge),
    # matching the 3D pipeline. A single value is used for all scales; if a list
    # is passed for backward compatibility, the smallest (most permissive) value
    # is used.
    if isinstance(min_size_pixels, (int, float)):
        min_size_global = int(min_size_pixels)
    else:
        min_size_global = int(min(min_size_pixels)) if len(min_size_pixels) else 0

    temp_dirs_to_clean, threshold_history = [], {}
    final_labels_memmap = None

    try:
        spacing_2d = tuple(float(s) for s in spacing[-2:])

        # --- Normalization (shared preprocessing across all scales) ---
        with SimpleTimer("Stage 1.1: Normalization"):
            norm_dir = _get_safe_temp_dir(temp_root_path, 'normalize'); temp_dirs_to_clean.append(norm_dir)
            norm_path = os.path.join(norm_dir, 'norm.dat')
            norm_mm = np.memmap(norm_path, dtype=np.float32, mode='w+', shape=image.shape)
            
            chunk_gen = list(_get_chunk_slices_2d(image.shape, (2048, 2048), overlap=0))
            
            if threshold_mode == "Absolute":
                norm_factor = 1.0
                if np.issubdtype(image.dtype, np.integer):
                    norm_factor = float(np.iinfo(image.dtype).max)
                    
                print(f"    Normalization skipped for Absolute mode; scaling by DType Max ({norm_factor}) to [0, 1] range.")
                for read_sl, _ in tqdm(chunk_gen, desc="    Applying"):
                    norm_mm[read_sl] = image[read_sl].astype(np.float32) / norm_factor
                norm_mm.flush()
            else:
                # ORIGINAL PERCENTILE NORMALIZATION LOGIC REMAINS HERE
                global_high_p = max(high_thresh_list)
                norm_stride = max(1, min(8, min(image.shape) // 256))
                samples = image[::norm_stride, ::norm_stride].ravel(); samples = samples[samples > 0]
                high_val = np.percentile(samples, global_high_p) if samples.size > 0 else 1.0
                high_val = max(high_val, 1e-9)
                print(f"    Normalization Max (p{global_high_p}): {high_val:.2f}")

                for read_sl, _ in tqdm(chunk_gen, desc="    Applying"):
                    norm_mm[read_sl] = image[read_sl].astype(np.float32) / high_val
                norm_mm.flush()

        # --- Multi-Scale Logic (Independent Smoothing/Gap + Threshold-then-OR) ---
        # Smoothing and gap-closing are per-scale; the minimum-size filter is
        # GLOBAL and applied once after the merge (see the labeling stage),
        # matching the 3D pipeline.
        master_dir = _get_safe_temp_dir(temp_root_path, 'master'); temp_dirs_to_clean.append(master_dir)
        master_mm = np.memmap(os.path.join(master_dir, 'm.dat'), dtype=np.uint8, mode='w+', shape=image.shape)
        master_mm[:] = 0

        # Seed accumulator for hysteresis (Absolute mode). Marks confident
        # (`high`) detections from hysteresis scales, plus ALL pixels from
        # non-hysteresis scales. After the merge, a labelled object is kept only
        # if it contains a seed — so an object made only of dim (`low`) pixels
        # with no confident core is dropped, while dim processes that connect to
        # a bright seed (or to a plain-threshold detection such as a soma) are
        # retained. With no seed configured (all rows high>=1.0) every pixel is
        # its own seed, so this reduces exactly to the previous size-only filter.
        master_seed_dir = _get_safe_temp_dir(temp_root_path, 'master_seed'); temp_dirs_to_clean.append(master_seed_dir)
        master_seed_mm = np.memmap(os.path.join(master_seed_dir, 'seed.dat'), dtype=np.uint8, mode='w+', shape=image.shape)
        master_seed_mm[:] = 0

        # Use enumerate to index each scale's own parameters
        for i, scale in enumerate(tubular_scales):
            current_low_p = low_thresh_list[i]
            current_smooth_sigma = smooth_sigma_list[i]
            current_connect_gap = connect_gap_list[i]

            with SimpleTimer(f"Scale sigma={scale} (p{current_low_p})"):
                # --- Per-Scale Smoothing (Preprocessing) ---
                scale_smooth_dir = None
                if current_smooth_sigma > 0:
                    scale_smooth_dir = _get_safe_temp_dir(temp_root_path, f'smoothing_s{i}')
                    scale_smooth_path = os.path.join(scale_smooth_dir, 'smoothed.dat')
                    smoothed_mm = np.memmap(scale_smooth_path, dtype=np.float32, mode='w+', shape=image.shape)

                    sigma_vox = [current_smooth_sigma / s if s > 0 else 0 for s in spacing_2d]
                    d_norm = da.from_array(norm_mm, chunks=(4096, 4096))
                    d_smooth = dask_image.ndfilters.gaussian_filter(d_norm, sigma=sigma_vox)

                    with ProgressBar(dt=5):
                        da.store(d_smooth, smoothed_mm, scheduler='threads')
                    smoothed_mm.flush()
                else:
                    smoothed_mm = norm_mm

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
                if threshold_mode == "Absolute":
                    thresh = current_low_p
                    thresh = max(thresh, 1e-5); threshold_history[scale] = thresh
                    print(f"      [Scale {scale}] Absolute Threshold: {thresh:.6f}")
                else:
                    stride = max(1, min(16, min(image.shape) // 128))
                    samples = enh_mm[::stride, ::stride].ravel(); samples = samples[samples > 1e-7]
                    thresh = float(np.percentile(samples, current_low_p)) if samples.size > 1000 else 1e9
                    thresh = max(thresh, 1e-5); threshold_history[scale] = thresh
                    print(f"      [Scale {scale}] Isolated Threshold (p{current_low_p}): {thresh:.6f}")

                # `high` is an intuitive UPPER bound (band-pass): reject
                # response above it — too-bright structures (e.g. saturated
                # autofluorescence). Absolute mode only; high >= 1.0 disables it.
                # `seed` is a SEPARATE, opt-in hysteresis control: keep dim
                # (`low`) regions only where they connect to a `seed`-bright
                # pixel. Absolute mode only; 0 disables (scale seeds itself).
                current_high = high_thresh_list[i]
                current_seed = seed_thresh_list[i]
                use_upper = (
                    threshold_mode == "Absolute"
                    and current_high < 1.0
                    and current_high > thresh
                )
                seed_active = (
                    threshold_mode == "Absolute"
                    and current_seed > 0.0
                    and current_seed > thresh
                )
                # Coherence now SEEDS (it no longer gates the grow): a component
                # survives only if it contains an oriented (coherent) core, so
                # incoherent background is dropped whole while real structure —
                # grown by magnitude — is never fragmented. Tubular path only.
                coh_active = (coherence_floor > 0.0 and scale != 0)

                if thresh < 1e6:
                    enh_dask = da.from_array(enh_mm, chunks=(4096, 4096))

                    # Per-scale gap-closing structure
                    radius_px = math.ceil((current_connect_gap / 2) / np.mean(spacing_2d))
                    struct = disk(radius_px) if radius_px > 0 else np.ones((1, 1), dtype=bool)

                    band = enh_dask > thresh
                    if use_upper:
                        band = band & (enh_dask < current_high)
                        print(f"      [Scale {scale}] Upper bound (reject > {current_high:.6f})")
                    # Grow is magnitude-only: a structure fills contiguously
                    # through its own faint/incoherent stretches, so coherence
                    # can never fragment a real branch.
                    clean_dask = dask_image.ndmorph.binary_closing(band, structure=struct)

                    # Seed mask defines the confident cores a component must
                    # contain to survive the hysteresis test below.
                    #   coherence on  -> a coherent core (oriented structure);
                    #                    optionally also a bright core if `seed`
                    #                    is set. Incoherent background has no
                    #                    coherent core and is dropped whole,
                    #                    without cutting into real structure.
                    #   coherence off -> the bright `seed` core (or self-seed).
                    if coh_active:
                        coh = _orientation_coherence(
                            smoothed_mm, spacing_2d, scale, chunks=(4096, 4096)
                        )
                        seed_dask = (coh > coherence_floor) & band
                        if seed_active:
                            seed_dask = seed_dask | (enh_dask > current_seed)
                        print(f"      [Scale {scale}] Coherent-core seed (floor={coherence_floor})")
                    elif seed_active:
                        seed_dask = enh_dask > current_seed
                        if use_upper:
                            seed_dask = seed_dask & (enh_dask < current_high)
                        print(f"      [Scale {scale}] Hysteresis seed @ {current_seed:.6f}")
                    else:
                        seed_dask = clean_dask

                    # Merge this scale's detections into the master mask. Size
                    # filtering is deferred to a single GLOBAL pass after all
                    # scales are merged (see the labeling stage below).
                    chunk_gen = list(_get_chunk_slices_2d(image.shape, (2048, 2048), overlap=0))
                    for read_sl, _ in tqdm(chunk_gen, desc="      Merging"):
                        master_mm[read_sl] |= clean_dask[read_sl].compute().astype(np.uint8)
                        master_seed_mm[read_sl] |= seed_dask[read_sl].compute().astype(np.uint8)

                master_mm.flush()
                master_seed_mm.flush()

                # Clean up this scale's intermediate buffers. `enh_mm` is
                # dropped first since, for scale==0 or smooth_sigma==0, it may
                # just be an alias for `smoothed_mm` (or `norm_mm`) rather than
                # a distinct memmap.
                if enh_dir:
                    del enh_mm; shutil.rmtree(enh_dir, ignore_errors=True)
                elif 'enh_mm' in locals():
                    del enh_mm
                if scale_smooth_dir:
                    del smoothed_mm; shutil.rmtree(scale_smooth_dir, ignore_errors=True)
                gc.collect()

        del norm_mm; gc.collect()

        # --- Final Labeling + GLOBAL Size Filter ---
        # A single minimum-size filter is applied here to the merged mask,
        # matching the 3D pipeline: objects smaller than `min_size_global`
        # (after all scales are combined) are dropped, and survivors are given
        # contiguous IDs.
        print("\n  [Step 1.4] Labeling Objects...")
        final_dir = _get_safe_temp_dir(temp_root_path, 'final'); labels_temp_dir = final_dir
        labels_path = os.path.join(final_dir, 'l.dat')
        final_mm = np.memmap(labels_path, dtype=np.int32, mode='w+', shape=image.shape)

        lab_dir = _get_safe_temp_dir(temp_root_path, 'lab_zarr'); temp_dirs_to_clean.append(lab_dir)
        m_dask = da.from_array(master_mm, chunks=(4096, 4096))
        labeled_dask, num_feats_dask = dask_image.ndmeasure.label(
            (m_dask > 0), structure=generate_binary_structure(2, 1)
        )
        labeled_dask.to_zarr(os.path.join(lab_dir, 'l.zarr'), overwrite=True)
        num_feats = int(num_feats_dask.compute())

        chunk_gen = list(_get_chunk_slices_2d(image.shape, (2048, 2048), overlap=0))
        if num_feats == 0:
            final_mm[:] = 0
        else:
            d_lbl = da.from_zarr(os.path.join(lab_dir, 'l.zarr'))
            counts, _ = da.histogram(d_lbl, bins=num_feats + 1, range=[-0.5, num_feats + 0.5])
            size_ok = counts.compute()[1:] >= min_size_global  # position k -> label k+1

            lz = zarr.open(os.path.join(lab_dir, 'l.zarr'), mode='r')

            # Hysteresis resolution: an object survives only if it BOTH meets the
            # global size floor AND contains at least one seed pixel. When no
            # scale set a `high` seed, master_seed_mm == master_mm, so every
            # object is seeded and this reduces to the plain size filter.
            seeded = np.zeros(num_feats + 1, dtype=bool)
            for rs, _ in chunk_gen:
                seed_chunk = master_seed_mm[rs] > 0
                if seed_chunk.any():
                    seeded[np.unique(lz[rs][seed_chunk])] = True
            seeded[0] = False

            keep = np.zeros(num_feats + 1, dtype=bool)
            keep[1:] = size_ok
            keep &= seeded

            lookup = np.zeros(num_feats + 1, dtype=np.int32)
            new_id = 0
            for old_id in range(1, num_feats + 1):
                if keep[old_id]:
                    new_id += 1
                    lookup[old_id] = new_id

            for rs, ws in tqdm(chunk_gen, desc="    Filtering"):
                final_mm[ws] = lookup[lz[rs]]
        final_mm.flush()

        # Explicitly release master buffers before returning
        for _m in ('master_mm', 'master_seed_mm'):
            if _m in locals():
                del locals()[_m]

        return labels_path, labels_temp_dir, threshold_history.get(tubular_scales[0], 0.0), {'threshold_history': threshold_history}

    finally:
        # Close any local memmap handles to avoid PermissionError on cleanup
        for var in ['final_mm', 'norm_mm', 'smoothed_mm', 'master_mm', 'master_seed_mm', 'input_mm']:
            if var in locals():
                del locals()[var]
        
        for d in temp_dirs_to_clean:
            shutil.rmtree(d, ignore_errors=True)
        gc.collect()