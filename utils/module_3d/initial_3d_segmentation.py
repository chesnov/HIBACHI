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

                # Penalise edge/boundary responses without eroding true width
                # (matches the 2D path; the 3D enhancement is 2D-per-slice, so
                # the crest gate is per-slice too).
                if _CREST_TEST:
                    scale_res = _crest_gated_response(scale_res, slice_2d, sigma)

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
                          angle_tol_deg=45.0, momentum=0.5, recenter_radius=3,
                          soma_lut=None, link_radius=3, absorb_below=0):
    """Tensor-voting gap linker: reconnect the pieces of one process by
    perceptual good-continuation rather than an intensity walk.

    Each fragment becomes a set of tokens: an oriented "stick" at every skeleton
    endpoint (pointing along the local axis) or, when the fragment is too small
    to have a stable direction, a single orientation-less "ball" token at its
    centroid. Tokens then vote for one another with the standard tensor-voting
    field -- a token propagates the orientation of the smoothest (co-circular)
    curve that could pass through a neighbour, with a strength that decays with
    gap length and curvature. Votes accumulate into a structure tensor at each
    token; the eigen-gap (lambda1 - lambda2) is the curve saliency and the
    leading eigenvector the emergent orientation.

    This inverts the old ridge-walk's bias. Orientation is an *ensemble*
    property, so a chain of dots -- each individually too small to have a
    direction -- takes on a shared orientation from its neighbours and links into
    a line ("connect the dots"), while a fragment collinear with nothing accrues
    little saliency. A link is accepted only between tokens whose orientations
    *both* point along the connecting chord (bidirectional good continuation) and
    that are each other's strongest available partner, so an off-line noise
    branch is rejected even when it is nearer than the true continuation --
    because it is not on the same line. `max_gap` sets the voting scale / maximum
    link distance; `angle_tol_deg` is the association-field half-angle. (`step`,
    `momentum`, `recenter_radius` are retained for signature compatibility.)

    Memory-light (per-fragment skeletons; the full mask is never held in RAM),
    nD-generic (2D and 3D identical), opt-in (max_gap <= 0 is a no-op upstream).
    Returns the new object count, or None if nothing linked / SciPy unavailable.

    Soma-aware linking (`soma_lut`) is unchanged: somata (scale-0 blobs) are not
    used as voting tokens, and each process endpoint first tries a direct
    proximity link to a nearby soma before entering the voting pool.
    """
    try:
        from scipy import ndimage as _ndi
        from skimage.morphology import skeletonize as _skel
    except Exception:
        return None

    ndim = final_mm.ndim
    sp = np.asarray(spacing[-ndim:], dtype=float)
    mean_sp = float(sp.mean())
    max_dist = max(1.0, max_gap / max(mean_sp, 1e-9))   # gap budget, in voxels
    shape = np.asarray(final_mm.shape)
    cos_tol = float(np.cos(np.deg2rad(angle_tol_deg)))
    sigma = float(max_dist)                             # tensor-voting scale
    bend_w = 1.5                                        # curvature penalty in vote decay
    A_min = 0.20                                        # min affinity to accept a link
    near_scale = max(4.0, 2.0 * float(link_radius))     # below this gap, proximity
    #                                                     dominates and the angle
    #                                                     requirement is relaxed
    absorb_reach = max(3.0, float(link_radius) + 2.0)   # swallow specks within this
    #                                                     many px of a link bridge

    parent = {}

    def _find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    _size_cache = {}
    _cen_cache = {}
    _absorb_off = [None]   # lazy small-ball offsets for off-centerline absorption

    def _label_size(lab):
        lab = int(lab)
        if lab not in _size_cache:
            sl = objs[lab - 1] if 1 <= lab <= len(objs) else None
            _size_cache[lab] = 0 if sl is None else \
                int((np.asarray(final_mm[sl]) == lab).sum())
        return _size_cache[lab]

    def _label_centroid(lab):
        lab = int(lab)
        if lab not in _cen_cache:
            sl = objs[lab - 1] if 1 <= lab <= len(objs) else None
            if sl is None:
                _cen_cache[lab] = None
            else:
                base = np.array([s.start for s in sl], dtype=float)
                pts = np.argwhere(np.asarray(final_mm[sl]) == lab)
                _cen_cache[lab] = None if len(pts) == 0 else pts.mean(0) + base
        return _cen_cache[lab]

    def _bridge_check(pa, pb, fa, fb):
        """Inspect the straight segment pa->pb. Returns (blocked, absorb):
        a LARGE foreign mask on the centerline blocks the link (never cut across a
        real structure); SMALL foreign specks on or within `absorb_reach` of the
        bridge -- ones the size filter would delete anyway -- are collected in
        `absorb` so the link swallows them into the merged process rather than
        being vetoed by them."""
        if _absorb_off[0] is None:
            r = int(max(1, round(absorb_reach)))
            off = np.argwhere(np.ones((2 * r + 1,) * ndim)) - r
            _absorb_off[0] = off[(off ** 2).sum(1) <= r * r + 1]
        ra, rb = _find(int(fa)), _find(int(fb))
        pa = np.asarray(pa, dtype=float); pb = np.asarray(pb, dtype=float)
        n = int(max(1, round(float(np.linalg.norm(pb - pa)))))
        blocked = False
        absorb = set()
        for k in range(n + 1):
            ci = np.round(pa + (pb - pa) * (k / n)).astype(int)
            if np.all(ci >= 0) and np.all(ci < shape):
                v = int(final_mm[tuple(ci)])
                if v != 0 and _find(v) not in (ra, rb):
                    if absorb_below > 0 and _label_size(v) < absorb_below:
                        absorb.add(v)
                    else:
                        blocked = True
            if absorb_below > 0:
                for o in _absorb_off[0]:
                    q = ci + o
                    if np.all(q >= 0) and np.all(q < shape):
                        w = int(final_mm[tuple(q)])
                        if w != 0 and _find(w) not in (ra, rb) and _label_size(w) < absorb_below:
                            absorb.add(w)
        return blocked, absorb

    objs = _ndi.find_objects(final_mm)       # streams the memmap; O(labels) memory
    bridges = []
    n_soma_links = 0
    absorbed_labels = set()

    def _emit_bridge(waypoints, lab):
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
            nseg = int(max(1, round(float(np.linalg.norm(b - a)))))
            bridges.append(([a + (b - a) * (t / nseg) for t in range(nseg + 1)], lab))

    def _route(pa, pb, absorb):
        # Route the bridge THROUGH absorbed specks (ordered along the chord) so the
        # merged label stays a single connected component even when a speck sits
        # off the straight line.
        pa = np.asarray(pa, dtype=float); pb = np.asarray(pb, dtype=float)
        chord = pb - pa; L2 = float(chord @ chord) + 1e-9
        mids = []
        for a in absorb:
            c = _label_centroid(int(a))
            if c is None:
                continue
            t = float((np.asarray(c, dtype=float) - pa) @ chord) / L2
            mids.append((min(1.0, max(0.0, t)), np.asarray(c, dtype=float)))
        mids.sort(key=lambda z: z[0])
        return [pa] + [c for _, c in mids] + [pb]

    # Soma geometry (unchanged): process endpoints link straight to a nearby soma.
    soma_info = []
    if soma_lut is not None:
        for _idx, _sl in enumerate(objs):
            _lab = _idx + 1
            if _sl is None or _lab >= soma_lut.size or not soma_lut[_lab]:
                continue
            _b = np.array([s.start for s in _sl], dtype=float)
            _ext = np.array([s.stop - s.start for s in _sl], dtype=float)
            soma_info.append((_lab, _sl, _b, _b + 0.5 * _ext, 0.5 * float(np.linalg.norm(_ext))))

    def _link_to_soma(start, d):
        """Nearest soma label within `max_dist` of endpoint `start`, preferring
        somata ahead of the outward tangent `d`. Returns (label, target) or (0, None)."""
        best_lab, best_tgt, best_dist = 0, None, np.inf
        for lab, sl, base_s, cen, rad in soma_info:
            if np.linalg.norm(cen - start) > max_dist + rad + 2.0:
                continue
            sub = np.asarray(final_mm[sl]) == lab
            pts = np.argwhere(sub).astype(float) + base_s
            if pts.shape[0] == 0:
                continue
            diff = pts - start
            dist = np.sqrt((diff ** 2).sum(1))
            fwd = (diff @ d) / (dist + 1e-9)
            ok = ((dist <= max_dist) & (fwd >= 0.0)) | (dist <= max(2.0, float(link_radius)))
            if not ok.any():
                continue
            j = int(np.argmin(np.where(ok, dist, np.inf)))
            if dist[j] < best_dist:
                best_lab, best_tgt, best_dist = lab, pts[j], dist[j]
        return best_lab, best_tgt

    # ---- Build tokens (stick at each endpoint; ball at a dot's centroid) ---- #
    pos, ori, is_ball, frag = [], [], [], []
    for idx, sl in enumerate(objs):
        lab = idx + 1
        if sl is None:
            continue
        if soma_lut is not None and lab < soma_lut.size and soma_lut[lab]:
            continue                              # somata are targets, not tokens
        sub = np.asarray(final_mm[sl]) == lab
        if int(sub.sum()) < 1:
            continue
        base = np.array([x.start for x in sl], dtype=float)
        try:
            sk = _skel(sub)
        except Exception:
            sk = sub
        skpts = np.argwhere(sk).astype(float)
        endpoints = []
        if len(skpts) >= 3:
            nb = _ndi.convolve(sk.astype(int), np.ones((3,) * ndim, dtype=int),
                               mode='constant') - 1
            tan_r2 = max(3.0, min(0.5 * sigma, 8.0)) ** 2   # local heading, not over-smoothed
            for epl in np.argwhere(sk & (nb == 1)).astype(float):
                nearpts = skpts[((skpts - epl) ** 2).sum(1) <= tan_r2]
                if len(nearpts) < 2:
                    continue
                t = epl - nearpts.mean(0)
                tn = np.linalg.norm(t)
                if tn < 1e-9:
                    continue
                endpoints.append((epl + base, t / tn))
        if endpoints:
            for gpos, gdir in endpoints:
                if soma_lut is not None:
                    slab, stgt = _link_to_soma(gpos, gdir)
                    if slab and _find(slab) != _find(lab):
                        blocked, absorb = _bridge_check(gpos, stgt, lab, slab)
                        if not blocked:
                            parent[_find(lab)] = _find(slab)
                            for _a in absorb:
                                parent[_find(int(_a))] = _find(int(slab))
                                absorbed_labels.add(int(_a))
                            _emit_bridge(_route(gpos, stgt, absorb), lab)
                            n_soma_links += 1
                            continue
                pos.append(gpos); ori.append(gdir); is_ball.append(False); frag.append(lab)
        else:
            cen = (skpts.mean(0) + base) if len(skpts) else \
                  (np.argwhere(sub).astype(float).mean(0) + base)
            pos.append(cen); ori.append(np.zeros(ndim)); is_ball.append(True); frag.append(lab)

    if soma_lut is not None:
        print(f"    [DIAG] direct process->soma links: {n_soma_links}")

    M = len(pos)
    if M >= 2:
        pos = np.asarray(pos, dtype=float)
        ori = np.asarray(ori, dtype=float)
        is_ball = np.asarray(is_ball, dtype=bool)
        frag = np.asarray(frag, dtype=int)

        def _vote(orient, ball):
            """One tensor-voting pass -> (emergent_orientation, curve_saliency)."""
            S = np.zeros((M, ndim, ndim), dtype=float)
            for i in range(M):
                d = pos - pos[i]
                s = np.sqrt((d ** 2).sum(1))
                m = (s > 1e-6) & (s <= max_dist)
                if not m.any():
                    continue
                idxs = np.where(m)[0]
                u = d[idxs] / s[idxs][:, None]
                if ball[i]:
                    vv = u.copy()                          # ball voter: radial
                    cosang = np.ones(len(idxs))
                else:
                    vi = orient[i]
                    cosang = np.abs(u @ vi)
                    keep = cosang >= cos_tol
                    if not keep.any():
                        continue
                    idxs = idxs[keep]; u = u[keep]; cosang = cosang[keep]
                    vv = 2.0 * (u @ vi)[:, None] * u - vi   # co-circular tangent
                nrm = np.linalg.norm(vv, axis=1)
                good = nrm > 1e-9
                if not good.any():
                    continue
                idxs = idxs[good]; u = u[good]; cosang = cosang[good]
                vv = vv[good] / nrm[good][:, None]
                sij = s[idxs]
                sin2 = np.clip(1.0 - cosang ** 2, 0.0, 1.0)
                kappa2 = 4.0 * sin2 / (sij ** 2 + 1e-12)
                DF = np.exp(-(sij ** 2) / (sigma ** 2) - bend_w * kappa2)
                contrib = DF[:, None, None] * (vv[:, :, None] * vv[:, None, :])
                np.add.at(S, idxs, contrib)
            evals, evecs = np.linalg.eigh(S)               # ascending eigenvalues
            emergent = evecs[:, :, -1]
            sal = (evals[:, -1] - evals[:, -2]) if ndim >= 2 else evals[:, -1]
            return emergent, np.clip(sal, 0.0, None)

        # Pass 1: sticks vote along their tangent, balls radially -> dots acquire
        # an orientation from their neighbours. Pass 2: everyone votes as a stick
        # with the emergent orientation, sharpening dot chains.
        emer, sal = _vote(ori, is_ball)
        stick_ori = np.where(is_ball[:, None], emer, ori)
        emer, sal = _vote(stick_ori, np.zeros(M, dtype=bool))

        sal_ref = float(np.median(sal[sal > 0])) if np.any(sal > 0) else 1.0
        sal_ref = max(sal_ref, 1e-9)

        # ---- Candidate links (bidirectional good continuation) -------------- #
        A_list, I_list, J_list, Si_list, Sj_list = [], [], [], [], []
        for i in range(M):
            d = pos[i + 1:] - pos[i]
            if len(d) == 0:
                continue
            s = np.sqrt((d ** 2).sum(1))
            jj = np.arange(i + 1, M)
            m = (s > 1e-6) & (s <= max_dist) & (frag[jj] != frag[i])
            if not m.any():
                continue
            jj = jj[m]; u = d[m] / s[m][:, None]; ss = s[m]
            di = ori[i] if not is_ball[i] else emer[i]
            dj = np.where(is_ball[jj][:, None], emer[jj], ori[jj])
            ai = u @ di
            aj = -(u * dj).sum(1)                    # j should point back along -chord
            # Forward sense: a stick endpoint may only link to a partner ahead of
            # it (never backward through its own body); a ball has no sign.
            fwd = (ai >= 0.0) if not is_ball[i] else np.ones(len(jj), bool)
            fwd = fwd & np.where(is_ball[jj], True, aj >= 0.0)
            if not fwd.any():
                continue
            jj = jj[fwd]; ss = ss[fwd]; u = u[fwd]; ai = ai[fwd]; aj = aj[fwd]
            align_i = np.abs(ai) if is_ball[i] else ai
            align_j = np.where(is_ball[jj], np.abs(aj), aj)
            # Distance-graded good continuation: closeness raises confidence, so
            # collinearity is (almost) ignored for a few-pixel gap between large
            # fragments and fully enforced at long range (to still reject off-line
            # noise). w = 0 near -> angle irrelevant; w = 1 far -> full alignment.
            w = np.clip(ss / near_scale, 0.0, 1.0)
            geom = (1.0 - w) + w * (align_i * align_j)
            satw = 0.5 + 0.5 * np.minimum(1.0, np.minimum(sal[i], sal[jj]) / sal_ref)
            A = geom * np.exp(-(ss ** 2) / (sigma ** 2)) * satw
            good = A >= A_min
            if not good.any():
                continue
            jj = jj[good]; A = A[good]; ai = ai[good]; aj = aj[good]
            side_i = np.where(ai >= 0, 1, -1)
            side_j = np.where(aj >= 0, 1, -1)        # aj = (-chord).dj already
            for k in range(len(jj)):
                A_list.append(float(A[k])); I_list.append(i); J_list.append(int(jj[k]))
                Si_list.append(int(side_i[k])); Sj_list.append(int(side_j[k]))

        order = np.argsort(A_list)[::-1] if A_list else []
        used = {}   # token -> occupied sides (+1 forward, -1 back)

        def _free(tok, side, ball):
            occ = used.get(tok, set())
            return (side not in occ) if ball else (len(occ) == 0)

        n_link = 0
        for k in order:
            i = I_list[k]; j = J_list[k]; si = Si_list[k]; sj = Sj_list[k]
            if _find(frag[i]) == _find(frag[j]):
                continue
            if not _free(i, si, bool(is_ball[i])) or not _free(j, sj, bool(is_ball[j])):
                continue
            blocked, absorb = _bridge_check(pos[i], pos[j], int(frag[i]), int(frag[j]))
            if blocked:
                continue                     # would cut across a real structure
            parent[_find(frag[i])] = _find(frag[j])
            for _a in absorb:                # swallow tiny specks lying in the gap
                parent[_find(int(_a))] = _find(int(frag[j]))
                absorbed_labels.add(int(_a))
            used.setdefault(i, set()).add(si)
            used.setdefault(j, set()).add(sj)
            _emit_bridge(_route(pos[i], pos[j], absorb), int(frag[i]))
            n_link += 1
        print(f"    [DIAG] tensor-voting links: {n_link}")
        print(f"    [DIAG] absorbed specks: {len(absorbed_labels)}")

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

    # Paint bridges with a small radius so each link is solid & contiguous.
    br = 1
    ball_off = np.argwhere(np.ones((2 * br + 1,) * ndim)) - br
    ball_off = ball_off[(ball_off ** 2).sum(1) <= br * br + 1]
    for path, lab in bridges:
        for p in path:
            ci = np.round(p).astype(int)
            for o in ball_off:
                q = ci + o
                if np.all(q >= 0) and np.all(q < shape) and final_mm[tuple(q)] == 0:
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

        # Scale-0 provenance: somata are detected by the scale-0 (non-tubular)
        # pass. Accumulate their detections into a separate mask so the trace
        # linker can treat those objects differently (see soma-aware linking).
        # Allocated only when scale 0 is actually requested.
        scale0_mm = None
        if 0 in tubular_scales:
            scale0_dir = _get_safe_temp_dir(temp_root_path, 'scale0'); temp_dirs_to_clean.append(scale0_dir)
            scale0_mm = np.memmap(os.path.join(scale0_dir, 's0.dat'), dtype=np.uint8, mode='w+', shape=volume.shape)
            scale0_mm[:] = 0

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
                    
                    record_s0 = (scale == 0 and scale0_mm is not None)
                    for read_sl, _ in tqdm(list(_get_chunk_slices(volume.shape, (64, 512, 512))), desc="      Merging"):
                        blk = clean_dask[read_sl].compute().astype(np.uint8)
                        master_mm[read_sl] |= blk
                        if record_s0:
                            scale0_mm[read_sl] |= blk
                
                master_mm.flush()
                if scale0_mm is not None:
                    scale0_mm.flush()

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
        labeled_dask, num_feats_dask = dask_image.ndmeasure.label((m_dask > 0), structure=generate_binary_structure(3, 3))
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
            # Flag which labels are somata (scale-0 objects): a label is a soma
            # if the majority of its voxels came from the scale-0 mask. Streamed
            # one z-slab at a time so the full arrays never sit in RAM.
            soma_lut = None
            if scale0_mm is not None:
                maxid = int(final_mm.max())
                tot = np.zeros(maxid + 1, dtype=np.int64)
                s0 = np.zeros(maxid + 1, dtype=np.int64)
                for z in range(int(volume.shape[0])):
                    row = final_mm[z].ravel()
                    tot += np.bincount(row, minlength=maxid + 1)
                    sel = scale0_mm[z].ravel().astype(bool)
                    if sel.any():
                        s0 += np.bincount(row[sel], minlength=maxid + 1)
                soma_lut = (s0 >= 0.5 * np.maximum(tot, 1)) & (tot > 0)
                soma_lut[0] = False
                print(f"    [DIAG] soma (scale-0) labels flagged: {int(soma_lut.sum())}")
            try:
                n_obj = _trace_link_fragments(final_mm, volume, spacing, trace_max_gap, absorb_below=min_size_voxels,
                                              soma_lut=soma_lut)
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
        for var in ['final_mm', 'norm_mm', 'smoothed_mm', 'master_mm', 'input_mm', 'enh_mm', 'scale0_mm']:
            if var in locals():
                try: del locals()[var]
                except: pass
                
        for d in temp_dirs_to_clean:
            shutil.rmtree(d, ignore_errors=True)
        gc.collect()