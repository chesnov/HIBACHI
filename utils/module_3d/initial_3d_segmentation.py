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


def _trace_link_fragments(final_mm, image, spacing, max_gap, step=1.0,
                          angle_tol_deg=45.0, momentum=0.5, recenter_radius=3):
    """Orientation-following gap tracing to reconnect a structure broken by a
    dim stretch. Walks the local intensity ridge outward from each fragment
    endpoint, branch-by-branch -- direction comes from the recentred path, and
    each step re-snaps transversally onto the local ridge -- so it follows a
    process across a faint gap and handles arbors (every endpoint has its own
    local heading; no global cell axis). Fragments whose traces reach one
    another are merged and the traced path is painted so the object is
    spatially contiguous.

    Memory-light: the label and image arrays stay on disk (memmaps); only small
    local windows are read per endpoint/step, and relabelling streams one slab
    at a time -- the full mask is never held in RAM. Opt-in (max_gap <= 0 is a
    no-op upstream). Returns the new object count, or None if nothing linked.
    General: uses only intensity ridges and geometry, no model of the imaged
    object. Returns None quietly if scipy/skimage are unavailable.
    """
    try:
        from scipy import ndimage as _ndi
        from skimage.morphology import skeletonize as _skel
    except Exception:
        return None

    ndim = final_mm.ndim
    sp = np.asarray(spacing[-ndim:], dtype=float)
    mean_sp = float(sp.mean())
    max_steps = max(1, int(round(max_gap / (step * max(mean_sp, 1e-9)))))
    shape = np.asarray(final_mm.shape)
    cos_tol = float(np.cos(np.deg2rad(angle_tol_deg)))

    def _recenter(cur, d):
        ci = np.round(cur).astype(int)
        sl = tuple(slice(max(0, ci[k] - recenter_radius),
                         min(int(shape[k]), ci[k] + recenter_radius + 1)) for k in range(ndim))
        w = np.asarray(image[sl], dtype=float)
        s = w.sum()
        if s <= 1e-9:
            return cur, 0.0
        coords = np.indices(w.shape).reshape(ndim, -1)
        base = np.array([x.start for x in sl])
        cen = base + (coords * w.ravel()).sum(1) / s
        off = cen - cur
        off = off - (off @ d) * d          # keep only the transverse component
        return cur + off, float(w.max())

    def _trace(start, direction, own):
        cur = start.astype(float).copy()
        d = direction.astype(float).copy()
        try:
            min_int = 0.25 * float(image[tuple(np.round(start).astype(int))])
        except Exception:
            min_int = 0.0
        path = [cur.copy()]
        lost = 0
        for _ in range(max_steps):
            prev = cur.copy()
            cur = cur + step * d
            cur, peak = _recenter(cur, d)
            disp = cur - prev
            dn = np.linalg.norm(disp)
            if dn > 1e-6:
                nd = disp / dn
                if nd @ d < cos_tol:
                    return 0, None          # path bent too sharply -> stop
                d = momentum * d + (1 - momentum) * nd
                d /= (np.linalg.norm(d) + 1e-12)
            ci = np.round(cur).astype(int)
            if np.any(ci < 0) or np.any(ci >= shape):
                return 0, None
            path.append(cur.copy())
            lab = int(final_mm[tuple(ci)])
            if lab != 0 and lab != own:
                return lab, path            # reached another fragment
            if peak < min_int:
                lost += 1
                if lost > 5:
                    return 0, None          # trail faded out
            else:
                lost = 0
        return 0, None

    parent = {}

    def _find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    objs = _ndi.find_objects(final_mm)       # streams the memmap; O(labels) memory
    bridges = []
    for idx, sl in enumerate(objs):
        lab = idx + 1
        if sl is None:
            continue
        sub = np.asarray(final_mm[sl]) == lab
        if sub.sum() < 3:
            continue
        try:
            sk = _skel(sub)
        except Exception:
            continue
        pts = np.argwhere(sk)
        if len(pts) < 3:
            continue
        nb = _ndi.convolve(sk.astype(int), np.ones((3,) * ndim, dtype=int), mode='constant') - 1
        base = np.array([x.start for x in sl])
        for epl in np.argwhere(sk & (nb == 1)):
            dd = pts - epl
            near = pts[(dd ** 2).sum(1) <= 100]      # ~10 px for a stable tangent
            c = near.mean(0)
            t = epl - c
            tn = np.linalg.norm(t)
            if tn < 1e-9:
                continue
            hit, path = _trace((epl + base).astype(float), t / tn, lab)
            if hit and _find(hit) != _find(lab):
                parent[_find(lab)] = _find(hit)
                bridges.append((path, lab))

    if not parent:
        return None

    maxid = int(final_mm.max())
    root_of = np.arange(maxid + 1)
    for i in range(1, maxid + 1):
        root_of[i] = _find(i) if i in parent else i
    uniq = sorted(set(int(root_of[i]) for i in range(1, maxid + 1)))
    compact = {r: k + 1 for k, r in enumerate(uniq)}
    lut = np.zeros(maxid + 1, dtype=np.int32)
    for i in range(1, maxid + 1):
        lut[i] = compact[int(root_of[i])]

    # Paint traced bridges with a small radius so each path is solid & contiguous.
    br = 1
    ball = np.argwhere(np.ones((2 * br + 1,) * ndim)) - br
    ball = ball[(ball ** 2).sum(1) <= br * br + 1]
    for path, lab in bridges:
        for p in path:
            ci = np.round(p).astype(int)
            for o in ball:
                q = ci + o
                if np.all(q >= 0) and np.all(q < shape):
                    final_mm[tuple(q)] = lab

    # Relabel in place, streaming one leading-axis slab at a time.
    for i0 in range(int(shape[0])):
        final_mm[i0] = lut[final_mm[i0]]
    return len(uniq)



def segment_cells_first_pass_raw(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    tubular_scales: List[float] = [0.5, 1.0, 2.0, 3.0],
    smooth_sigma: Union[float, List[float]] = 0.5,
    connect_max_gap_physical: Union[float, List[float]] = 1.0,
    min_size_voxels: int = 50,
    low_threshold_percentile: Union[float, List[float]] = 25.0,
    high_threshold_percentile: Union[float, List[float]] = 95.0,
    threshold_mode: str = "Percentile",
    skip_tubular_enhancement: bool = False,
    subtract_background_radius: int = 0,
    trace_max_gap: float = 0.0,
    temp_root_path: Optional[str] = None,
    **kwargs: Any
) -> Tuple[Optional[str], Optional[str], float, Dict[str, Any]]:
    """Step 1: Raw Segmentation (Independent per-scale Smoothing + Threshold-then-OR).

    Smoothing and gap-closing are applied independently per tubular scale
    (mirroring the 2D pipeline); ``smooth_sigma`` and ``connect_max_gap_physical``
    may each be a scalar (broadcast to every scale) or a per-scale list. With a
    single scalar value the result is identical to the previous global behavior.

    Unlike the 2D pipeline, the minimum-size filter is applied GLOBALLY, once,
    AFTER all scales are merged (see the labeling stage), so ``min_size_voxels``
    remains a single value rather than a per-scale list.
    """
    print(f"\n--- Step 1: Raw Segmentation (Strict Independence Mode) ---")
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

    # --- Per-scale smoothing / gap-closing parameters ---
    # Each entry applies ONLY to its corresponding tubular scale. A scalar is
    # broadcast to every scale. (The min-size filter is deliberately NOT
    # per-scale here; it stays global, applied after the merge.)
    if isinstance(smooth_sigma, (int, float)):
        smooth_sigma_list = [float(smooth_sigma)] * n_scales
    else:
        smooth_sigma_list = [float(x) for x in smooth_sigma]

    if isinstance(connect_max_gap_physical, (int, float)):
        connect_gap_list = [float(connect_max_gap_physical)] * n_scales
    else:
        connect_gap_list = [float(x) for x in connect_max_gap_physical]

    if len(smooth_sigma_list) != n_scales or len(connect_gap_list) != n_scales:
        raise ValueError(
            "smooth_sigma/connect_max_gap_physical lists must match length "
            "of tubular_scales."
        )

    temp_dirs_to_clean, threshold_history = [], {}
    final_labels_memmap = None

    try:
        # --- Stage 1.1: Normalization ---
        with SimpleTimer("Stage 1.1: Normalization"):
            norm_dir = _get_safe_temp_dir(temp_root_path, 'normalize'); temp_dirs_to_clean.append(norm_dir)
            norm_path = os.path.join(norm_dir, 'norm.dat')
            norm_mm = np.memmap(norm_path, dtype=np.float32, mode='w+', shape=volume.shape)
            
            if threshold_mode == "Absolute":
                # Convert to[0, 1] based on bit depth to stabilize Frangi/Sato filters
                norm_factor = 1.0
                if np.issubdtype(volume.dtype, np.integer):
                    norm_factor = float(np.iinfo(volume.dtype).max)
                
                print(f"    Normalization skipped for Absolute mode; scaling by DType Max ({norm_factor}) to [0, 1] range.")
                for read_sl, _ in tqdm(list(_get_chunk_slices(volume.shape, (64, 512, 512))), desc="    Applying"):
                    norm_mm[read_sl] = volume[read_sl].astype(np.float32) / norm_factor
                norm_mm.flush()
            else:
                # ORIGINAL PERCENTILE NORMALIZATION LOGIC REMAINS HERE
                global_high_p = max(high_thresh_list)
                
                z_stats = {}
                stride_xy = max(1, min(16, min(volume.shape[1:]) // 128))
                
                for read_sl, _ in tqdm(list(_get_chunk_slices(volume.shape, (64, 512, 512))), desc="    Sampling"):
                    sub = volume[read_sl][:, ::stride_xy, ::stride_xy]
                    for i in range(sub.shape[0]):
                        vals = sub[i].ravel(); vals = vals[vals > 0]
                        if vals.size > 0:
                            idx = read_sl[0].start + i
                            if idx not in z_stats: z_stats[idx] = []
                            z_stats[idx].extend(vals[:5000])

                z_indices = np.arange(volume.shape[0])
                hp = np.array([np.percentile(z_stats[z], global_high_p) if z in z_stats else np.nan for z in z_indices])

                ideal = np.ones(volume.shape[0])
                if np.any(~np.isnan(hp)):
                    valid = ~np.isnan(hp)
                    p = np.poly1d(np.polyfit(z_indices[valid], hp[valid], 2))
                    raw_ideal = p(z_indices)
                    max_amplification = 3.0
                    hp_valid = hp[valid]
                    median_hp = float(np.median(hp_valid))
                    amp_floor = median_hp / max_amplification
                    abs_floor = np.nanpercentile(hp, 10)
                    ideal = np.maximum(raw_ideal, max(amp_floor, abs_floor))
                    print(f"    Normalization: median brightness={median_hp:.2f}, "
                          f"amp floor={amp_floor:.2f} (≤{max_amplification:.0f}× boost), "
                          f"abs floor={abs_floor:.2f}")

                for read_sl, _ in tqdm(list(_get_chunk_slices(volume.shape, (64, 512, 512))), desc="    Applying"):
                    factors = ideal[read_sl[0].start:read_sl[0].stop][:, np.newaxis, np.newaxis]
                    norm_mm[read_sl] = volume[read_sl].astype(np.float32) / np.where(factors > 1e-9, factors, 1.0)
                norm_mm.flush()

        # --- Stage 2 & 3: Multi-Scale Logic (per-scale smoothing + gap-closing,
        # threshold-then-OR). Smoothing and gap-closing now run independently per
        # scale inside the loop, mirroring the 2D pipeline. With a single global
        # smooth_sigma / connect_max_gap value this is identical to the previous
        # global behavior; per-scale lists let each scale differ.
        # NOTE: the minimum-size filter is intentionally kept GLOBAL and applied
        # AFTER the merge (labeling stage below), unlike the 2D pipeline which
        # filters per scale before merging. ---
        master_dir = _get_safe_temp_dir(temp_root_path, 'master'); temp_dirs_to_clean.append(master_dir)
        master_mm = np.memmap(os.path.join(master_dir, 'm.dat'), dtype=np.uint8, mode='w+', shape=volume.shape)
        master_mm[:] = 0

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
                    smoothed_mm = np.memmap(scale_smooth_path, dtype=np.float32, mode='w+', shape=volume.shape)

                    sigma_vox = [current_smooth_sigma / s if s > 0 else 0 for s in spacing]
                    d_norm = da.from_array(norm_mm, chunks=(128, 512, 512))
                    d_smooth = dask_image.ndfilters.gaussian_filter(d_norm, sigma=sigma_vox)

                    with ProgressBar(dt=5):
                        da.store(d_smooth, smoothed_mm, scheduler='threads')
                    smoothed_mm.flush()
                else:
                    smoothed_mm = norm_mm

                if scale == 0:
                    # Pass-through
                    enh_mm = smoothed_mm
                    enh_dir = None
                else:
                    # Vesselness
                    enh_mm, _, enh_dir = enhance_tubular_structures_blocked(
                        smoothed_mm, scales=[scale], spacing=spacing,
                        skip_tubular_enhancement=skip_tubular_enhancement,
                        subtract_background_radius=subtract_background_radius, temp_root_path=temp_root_path
                    )
                
                # Independent Thresholding Pass
                stride_z = max(1, min(4, volume.shape[0] // 32))
                stride_xy = max(1, min(16, min(volume.shape[1:]) // 128))
                
                if threshold_mode == "Absolute":
                    thresh = current_low_p
                    thresh = max(thresh, 1e-5); threshold_history[scale] = thresh
                    print(f"      [Scale {scale}] Absolute Threshold: {thresh:.6f}")
                else:
                    samples = enh_mm[::stride_z, ::stride_xy, ::stride_xy].ravel(); samples = samples[samples > 1e-7]
                    thresh = float(np.percentile(samples, current_low_p)) if samples.size > 1000 else 1e9
                    thresh = max(thresh, 1e-5); threshold_history[scale] = thresh
                    print(f"      [Scale {scale}] Isolated Threshold (p{current_low_p}): {thresh:.6f}")

                # Binary Creation, Closing, and OR-ing
                if thresh < 1e6:
                    enh_dask = da.from_array(enh_mm, chunks=(128, 512, 512))

                    # Per-scale gap-closing structure
                    rv = [math.ceil((current_connect_gap / 2) / s) if s > 1e-9 else 0 for s in spacing]
                    struct = np.ones(tuple(max(1, 2 * r + 1) for r in rv), dtype=bool)

                    clean_dask = dask_image.ndmorph.binary_closing((enh_dask > thresh), structure=struct)
                    
                    for read_sl, _ in tqdm(list(_get_chunk_slices(volume.shape, (64, 512, 512))), desc="      Merging"):
                        master_mm[read_sl] |= clean_dask[read_sl].compute().astype(np.uint8)
                
                master_mm.flush()

                # Clean up this scale's intermediate buffers. `enh_mm` is dropped
                # first since, for scale==0 or smooth_sigma==0, it may just be an
                # alias for `smoothed_mm` (or `norm_mm`) rather than a distinct
                # memmap.
                if enh_dir:
                    del enh_mm; shutil.rmtree(enh_dir, ignore_errors=True)
                elif 'enh_mm' in locals():
                    del enh_mm
                if scale_smooth_dir:
                    del smoothed_mm; shutil.rmtree(scale_smooth_dir, ignore_errors=True)
                gc.collect()

        # Cleanup normalized volume
        del norm_mm; gc.collect()

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

        if trace_max_gap > 0.0:
            # Trace-link BEFORE the size filter: write raw labels, reconnect
            # fragments broken by dim gaps, THEN apply the global size filter to
            # the merged labels. Memory-light throughout.
            lz = zarr.open(os.path.join(lab_dir, 'l.zarr'), mode='r')
            for rs, ws in tqdm(list(_get_chunk_slices(volume.shape, (64, 512, 512))), desc="    Writing labels"):
                final_mm[ws] = lz[rs]
            final_mm.flush()
            try:
                n_obj = _trace_link_fragments(final_mm, volume, spacing, trace_max_gap)
                if n_obj is not None:
                    print(f"    [DIAG] orientation trace-link -> {n_obj} objects")
                    final_mm.flush()
            except Exception as exc:
                print(f"    [trace-link] skipped: {exc}")
            maxid = int(final_mm.max())
            csz = np.zeros(maxid + 1, dtype=np.int64)
            for z in range(int(volume.shape[0])):
                csz += np.bincount(final_mm[z].ravel(), minlength=maxid + 1)
            lut = np.zeros(maxid + 1, dtype=np.int32)
            nid = 0
            for i in range(1, maxid + 1):
                if csz[i] >= min_size_voxels:
                    nid += 1
                    lut[i] = nid
            for z in range(int(volume.shape[0])):
                final_mm[z] = lut[final_mm[z]]
            final_mm.flush()
        else:
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