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


# Internal toggle for the bilateral crest test (NOT a GUI parameter). It refines
# the tubular response for every config, so there is no slider to A/B it against
# -- flip this to compare with/without, then leave it on once proven.
_CREST_TEST = True


def _crest_pairs(ndim: int):
    """Unit vectors, one per antipodal direction pair, for the crest sampling.
    2D: 4 pairs (axes + diagonals); 3D: the 13 unique 3x3x3 neighbourhood axes."""
    if ndim == 2:
        base = [(1, 0), (0, 1), (1, 1), (1, -1)]
    else:
        base = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    v = (dz, dy, dx)
                    if v == (0, 0, 0):
                        continue
                    if v > tuple(-c for c in v):   # keep one of each antipodal pair
                        base.append(v)
    out = []
    for v in base:
        a = np.asarray(v, dtype=np.float32)
        out.append(a / np.linalg.norm(a))
    return out


def _crest_weight(block, sigma, delta_factor: float = 2.0):
    """Bilateral crest weight in [0, 1] that penalises edge-like responses.

    A true process is a *two-sided* bright crest: along some direction (across
    the ridge) the intensity drops on BOTH flanks. A step/edge or the boundary
    of diffuse thick background rises on one side and stays high -- it is not a
    two-sided crest -- yet its Hessian signature fools Frangi/Sato into a tube-
    like response. This samples the smoothed intensity at +/- (delta_factor *
    sigma) along a fixed set of directions and, per antipodal pair, takes the
    two-sided margin min(centre - flank+, centre - flank-). The best two-sided
    margin normalised by the best one-sided drop is ~1 for a genuine crest and
    ~0 for an edge. Scale-aware (offset scales with sigma, so a larger-`scale`
    profile row widens the test for thick processes) and contrast-invariant (a
    ratio, so it behaves identically in dim and bright fields). No eigen-
    decomposition -- cheap and identical in 2D and 3D.
    """
    from scipy.ndimage import gaussian_filter, map_coordinates
    Is = gaussian_filter(block.astype(np.float32), float(sigma))
    d = max(1.0, delta_factor * float(sigma))
    idx = np.indices(Is.shape).astype(np.float32)
    best_two = np.full(Is.shape, -np.inf, dtype=np.float32)
    best_one = np.zeros(Is.shape, dtype=np.float32)
    for u in _crest_pairs(Is.ndim):
        plus = map_coordinates(
            Is, [idx[k] + d * u[k] for k in range(Is.ndim)], order=1, mode='nearest')
        minus = map_coordinates(
            Is, [idx[k] - d * u[k] for k in range(Is.ndim)], order=1, mode='nearest')
        a = Is - plus
        b = Is - minus
        best_two = np.maximum(best_two, np.minimum(a, b))
        best_one = np.maximum(best_one, np.maximum(a, b))
    return np.clip(best_two, 0.0, None) / (best_one + 1e-6)


# --- Crest gate tuning (NOT GUI parameters) --------------------------------
# Recovery radius factor: the crest validation is grown out to R = round(scale *
# sigma) voxels before it is applied, so a process' one-sided shoulders inherit
# the weight of their own centerline crest instead of being eroded away. R ~
# sigma matches the half-width of a structure detected at that scale. Larger
# values recover more width but let a little more edge response through.
_CREST_RECOVER_SCALE = 1.0
# Floor in [0, 1): response that is never near any validated crest survives at
# this fraction of its raw strength. 0.0 = strict edge rejection (recommended);
# raise slightly only if faint, genuinely un-crested processes are being lost.
_CREST_FLOOR = 0.0


def _crest_gated_response(scale_res, block, sigma, delta_factor: float = 2.0):
    """Edge-suppressed tubular response that PRESERVES true process width.

    The previous approach multiplied `scale_res` by the bilateral crest weight
    pointwise. Because only the ridge *centerline* is a genuine two-sided crest,
    that multiplication also drove the tube's one-sided shoulders to ~0 and
    eroded every object toward its skeleton -- the mask ended up far thinner
    than the real processes.

    Here the crest weight is used to VALIDATE, not to attenuate. The weight is
    grown by a grey dilation over a scale-matched disk (radius ~ sigma) so that
    every voxel within one process half-width of a validated crest inherits that
    crest's weight; the raw `scale_res` is then scaled by this widened weight.
    Genuine tubes are restored to their full detected width, while edge/boundary
    responses -- which have no validated crest anywhere in their neighbourhood --
    stay suppressed. Unlike a morphological reconstruction the operation is
    strictly local: a stray response can only lift weight within R voxels of
    itself and can never flood a whole connected region.

    Operates on a single 2D array, so it is byte-for-byte identical in the 2D
    pipeline and in the 2D-per-slice 3D pipeline.
    """
    from skimage.morphology import disk
    from scipy.ndimage import grey_dilation
    w = _crest_weight(block, sigma, delta_factor=delta_factor).astype(np.float32)
    radius = max(1, int(round(_CREST_RECOVER_SCALE * float(sigma))))
    w = grey_dilation(w, footprint=disk(radius))
    if _CREST_FLOOR > 0.0:
        w = _CREST_FLOOR + (1.0 - _CREST_FLOOR) * w
    return (scale_res * w).astype(np.float32)


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

            # Penalise edge/boundary responses the Hessian filter mistakes for
            # tubes, WITHOUT eroding true process width: validate on the crest,
            # then reconstruct the full response (scale-aware, no GUI param).
            if _CREST_TEST:
                scale_res = _crest_gated_response(scale_res, block_data, sigma)

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


def _trace_link_fragments(final_mm, image, spacing, max_gap, step=1.0,
                          angle_tol_deg=45.0, momentum=0.5, recenter_radius=3,
                          soma_lut=None, link_radius=3):
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

    Soma-aware linking (`soma_lut`): scale-0 detections are compact blobs
    (somata), not tubular processes, so they are handled differently. Their
    medial skeletons have no stable outward tangent, so they are NOT used as
    trace sources; and a process ridge walking toward a soma tends to stall on
    the soma's bright boundary rim (the big adjacent bright mass yanks the
    transverse re-centre past the bend tolerance) a voxel or two before it ever
    lands on a soma-labelled voxel. To fix both, `soma_lut` is a boolean array
    indexed by label id (True where the label came from scale 0). A process
    trace links to a soma as soon as it comes within `link_radius` voxels of
    one, instead of having to land exactly on it -- so proximity, not exact
    ridge contact, closes the process-to-soma gap. Process-to-process linking is
    unchanged (still requires landing on the target label). When `soma_lut` is
    None the routine behaves exactly as before.
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

    def _near_soma(cur, own):
        """Return a soma label within `link_radius` of `cur` (0 if none). Lets a
        process trace link to a soma on proximity, before the bend guard can
        abort it at the soma's bright boundary rim."""
        if soma_lut is None:
            return 0
        ci = np.round(cur).astype(int)
        sl = tuple(slice(max(0, ci[k] - link_radius),
                         min(int(shape[k]), ci[k] + link_radius + 1)) for k in range(ndim))
        win = np.asarray(final_mm[sl])
        m = soma_lut[win]
        if not m.any():
            return 0
        cand = win[m]
        cand = cand[cand != own]
        if cand.size == 0:
            return 0
        vals, cnts = np.unique(cand, return_counts=True)
        return int(vals[np.argmax(cnts)])   # nearest-dominant soma label

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
            shit = _near_soma(cur, own)     # soma proximity link (before bend guard)
            if shit:
                path.append(cur.copy())
                return shit, path
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
        if soma_lut is not None and lab < soma_lut.size and soma_lut[lab]:
            continue        # somata are link *targets*, not trace sources
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



def segment_cells_first_pass_raw_2d(
    image: np.ndarray,
    spacing: Union[Tuple[float, float], Tuple[float, float, float]],
    tubular_scales: List[float] = [0.5, 1.0, 2.0, 3.0],
    smooth_sigma: Union[float, List[float]] = 0.5,
    connect_max_gap_physical: Union[float, List[float]] = 1.0,
    min_size_pixels: Union[int, List[int]] = 50,
    low_threshold_percentile: Union[float, List[float]] = 95.0,
    high_threshold_percentile: Union[float, List[float]] = 100.0,
    threshold_mode: str = "Percentile",
    skip_tubular_enhancement: bool = False,
    subtract_background_radius: int = 0,
    trace_max_gap: float = 0.0,
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

        # Scale-0 provenance: somata are detected by the scale-0 (non-tubular)
        # pass. Accumulate their detections into a separate mask so the trace
        # linker can treat those objects differently (see soma-aware linking).
        # Allocated only when scale 0 is actually requested.
        scale0_mm = None
        if 0 in tubular_scales:
            scale0_dir = _get_safe_temp_dir(temp_root_path, 'scale0'); temp_dirs_to_clean.append(scale0_dir)
            scale0_mm = np.memmap(os.path.join(scale0_dir, 's0.dat'), dtype=np.uint8, mode='w+', shape=image.shape)
            scale0_mm[:] = 0

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

                if thresh < 1e6:
                    enh_dask = da.from_array(enh_mm, chunks=(4096, 4096))

                    # Per-scale gap-closing structure
                    radius_px = math.ceil((current_connect_gap / 2) / np.mean(spacing_2d))
                    struct = disk(radius_px) if radius_px > 0 else np.ones((1, 1), dtype=bool)

                    clean_dask = dask_image.ndmorph.binary_closing((enh_dask > thresh), structure=struct)

                    # Merge this scale's detections into the master mask. Size
                    # filtering is deferred to a single GLOBAL pass after all
                    # scales are merged (see the labeling stage below).
                    chunk_gen = list(_get_chunk_slices_2d(image.shape, (2048, 2048), overlap=0))
                    record_s0 = (scale == 0 and scale0_mm is not None)
                    for read_sl, _ in tqdm(chunk_gen, desc="      Merging"):
                        blk = clean_dask[read_sl].compute().astype(np.uint8)
                        master_mm[read_sl] |= blk
                        if record_s0:
                            scale0_mm[read_sl] |= blk
                
                master_mm.flush()
                if scale0_mm is not None:
                    scale0_mm.flush()

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
        elif trace_max_gap > 0.0:
            # Trace-link BEFORE the size filter: write the raw labels, reconnect
            # fragments broken by dim gaps (so two halves each below min_size can
            # fuse and survive as one object), THEN apply the global size filter
            # to the merged labels. Memory-light throughout.
            lz = zarr.open(os.path.join(lab_dir, 'l.zarr'), mode='r')
            for rs, ws in tqdm(chunk_gen, desc="    Writing labels"):
                final_mm[ws] = lz[rs]
            final_mm.flush()
            # Flag which labels are somata (scale-0 objects): a label is a soma
            # if the majority of its voxels came from the scale-0 mask. Streamed
            # one row-slab at a time so the full arrays never sit in RAM.
            soma_lut = None
            if scale0_mm is not None:
                maxid = int(final_mm.max())
                tot = np.zeros(maxid + 1, dtype=np.int64)
                s0 = np.zeros(maxid + 1, dtype=np.int64)
                for i0 in range(int(image.shape[0])):
                    row = final_mm[i0].ravel()
                    tot += np.bincount(row, minlength=maxid + 1)
                    sel = scale0_mm[i0].ravel().astype(bool)
                    if sel.any():
                        s0 += np.bincount(row[sel], minlength=maxid + 1)
                soma_lut = (s0 >= 0.5 * np.maximum(tot, 1)) & (tot > 0)
                soma_lut[0] = False
                print(f"    [DIAG] soma (scale-0) labels flagged: {int(soma_lut.sum())}")
            try:
                n_obj = _trace_link_fragments(final_mm, image, spacing_2d, trace_max_gap,
                                              soma_lut=soma_lut)
                if n_obj is not None:
                    print(f"    [DIAG] orientation trace-link -> {n_obj} objects")
                    final_mm.flush()
            except Exception as exc:
                print(f"    [trace-link] skipped: {exc}")
            # Global size filter on the merged labels, streamed one row-slab at a time.
            maxid = int(final_mm.max())
            csz = np.zeros(maxid + 1, dtype=np.int64)
            for i0 in range(int(image.shape[0])):
                csz += np.bincount(final_mm[i0].ravel(), minlength=maxid + 1)
            lut = np.zeros(maxid + 1, dtype=np.int32)
            nid = 0
            for i in range(1, maxid + 1):
                if csz[i] >= min_size_global:
                    nid += 1
                    lut[i] = nid
            for i0 in range(int(image.shape[0])):
                final_mm[i0] = lut[final_mm[i0]]
        else:
            d_lbl = da.from_zarr(os.path.join(lab_dir, 'l.zarr'))
            counts, _ = da.histogram(d_lbl, bins=num_feats + 1, range=[-0.5, num_feats + 0.5])
            valid = np.where(counts.compute()[1:] >= min_size_global)[0] + 1

            lookup = np.zeros(num_feats + 1, dtype=np.int32)
            for new_id, old_id in enumerate(valid):
                lookup[old_id] = new_id + 1

            lz = zarr.open(os.path.join(lab_dir, 'l.zarr'), mode='r')
            for rs, ws in tqdm(chunk_gen, desc="    Filtering"):
                final_mm[ws] = lookup[lz[rs]]
        final_mm.flush()

        # Explicitly release master_mm before returning
        if 'master_mm' in locals():
            del master_mm

        return labels_path, labels_temp_dir, threshold_history.get(tubular_scales[0], 0.0), {'threshold_history': threshold_history}

    finally:
        # Close any local memmap handles to avoid PermissionError on cleanup
        for var in ['final_mm', 'norm_mm', 'smoothed_mm', 'master_mm', 'input_mm', 'scale0_mm']:
            if var in locals():
                del locals()[var]
        
        for d in temp_dirs_to_clean:
            shutil.rmtree(d, ignore_errors=True)
        gc.collect()