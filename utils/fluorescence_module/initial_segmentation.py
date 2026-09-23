import os
import gc
import math
import time
import shutil
import tempfile
import multiprocessing as mp
from functools import partial
from typing import Tuple, List, Dict, Any, Optional, Sequence, Union, Generator

import numpy as np
import zarr
import dask
import dask.array as da
import dask_image.ndmorph
import dask_image.ndfilters
import dask_image.ndmeasure
from dask.diagnostics import ProgressBar
from scipy import ndimage
from scipy.special import ndtri
from scipy.ndimage import generate_binary_structure
from skimage.filters import frangi, sato  # type: ignore
from tqdm import tqdm

# Shared 2D/3D primitives. See dim_utils for why rank-varying operations are
# centralised rather than open-coded.
try:
    from . import resource_budget
    from .dim_utils import (
        open_worker_memmap,
        worker_memmap_handle,
        binary_structure,
        chunk_read_write_slices,
        min_inplane_spacing,
        normalise_spacing,
        planes_of,
        write_offset_in_read,
    )
except ImportError:  # pragma: no cover - direct script execution
    import resource_budget
    from dim_utils import (
        open_worker_memmap,
        worker_memmap_handle,
        binary_structure,
        chunk_read_write_slices,
        min_inplane_spacing,
        normalise_spacing,
        planes_of,
        write_offset_in_read,
    )

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
    # Temporary files MUST live in the project directory. The caller passes the
    # project's temp folder as base_path; there is deliberately NO hidden or
    # OS-temp fallback (that risks silent junk accumulation and RAM-backed
    # tmpfs). A missing base_path is a bug, so fail loudly instead.
    if not base_path:
        raise ValueError(
            "_get_safe_temp_dir requires a project temp directory (temp_root_path); "
            "temporary files must live in the project directory."
        )
    os.makedirs(base_path, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"step1_{suffix}_", dir=base_path)


def _get_chunk_slices(shape, chunk_shape, overlap=0):
    """
    Read/write slice pairs for chunked processing, at either rank.

    Delegates to `dim_utils.chunk_read_write_slices`. The 3D and 2D copies of
    this were the same algorithm with the axis count written out by hand.
    """
    yield from chunk_read_write_slices(shape, chunk_shape, overlap)

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
    input_memmap_info: Tuple[str, int, Tuple[int, ...], str],
    output_memmap_info: Tuple[str, int, Tuple[int, ...], str],
    sigmas_voxel_2d: List[float],
    black_ridges: bool,
    frangi_alpha: float,
    frangi_beta: float,
    frangi_gamma: float,
) -> Optional[str]:
    """
    Vesselness (Frangi/Sato) enhancement of one chunk, at either rank.

    The filter is inherently two-dimensional in both tracks: the 3D path ran it
    slice by slice over Z, the 2D path ran it once. That is the same code with a
    loop whose length happens to be 1, so `dim_utils.planes_of` supplies the
    planes and the body below is shared. Nothing about the per-plane arithmetic
    changed in the merge.

    The block is read with padding for filter context and only the owned write
    region is stored; `dim_utils.write_offset_in_read` computes that crop, which
    both tracks previously open-coded with per-axis index arithmetic.
    """
    input_memmap = None
    output_memmap = None
    block_data = None
    result_block = None
    try:
        read_slices, write_slices = chunk_info
        input_memmap = open_worker_memmap(input_memmap_info, mode='r')
        output_memmap = open_worker_memmap(output_memmap_info, mode='r+')

        block_data = input_memmap[read_slices].astype(np.float32)

        # Where the owned region sits inside the padded block.
        crop = write_offset_in_read(read_slices, write_slices)
        result_block = np.zeros(
            tuple(sl.stop - sl.start for sl in crop), dtype=np.float32
        )
        # In 3D the leading axis is Z and only the owned planes are computed;
        # in 2D there is a single plane and the crop is purely in-plane.
        lead = crop[0] if block_data.ndim == 3 else slice(0, 1)

        for idx, plane in planes_of(block_data):
            if block_data.ndim == 3 and (idx < lead.start or idx >= lead.stop):
                continue

            combined_scales = np.zeros_like(plane, dtype=np.float32)

            for sigma in sigmas_voxel_2d:
                # Strictly Vesselness (Frangi/Sato)
                beta_val = 1.0 if sigma >= 2.0 else frangi_beta
                f_res = frangi(plane, sigmas=[sigma], alpha=frangi_alpha,
                               beta=beta_val, gamma=frangi_gamma,
                               black_ridges=black_ridges)
                s_res = sato(plane, sigmas=[sigma], black_ridges=black_ridges)
                scale_res = np.maximum(f_res, s_res)

                # Penalise edge/boundary responses without eroding true width.
                if _CREST_TEST:
                    scale_res = _crest_gated_response(scale_res, plane, sigma)

                if sigma >= 2.0:
                    scale_res *= plane

                combined_scales = np.maximum(combined_scales, scale_res)

            inplane = combined_scales[crop[-2:]]
            if block_data.ndim == 3:
                result_block[idx - lead.start] = inplane
            else:
                result_block[...] = inplane

        output_memmap[write_slices] = result_block
        return None

    except Exception as e:
        return f"Error_chunk_{chunk_info}: {str(e)}"
    finally:
        try:
            del input_memmap, output_memmap, block_data, result_block
        except Exception:
            pass
        gc.collect()

#: Minimum chunk overlap in pixels for the vesselness enhancement. The overlap
#: must exceed the filter's reach or a structure crossing a chunk edge is
#: enhanced against truncated context and the seam shows in the output. The two
#: tracks had drifted to different floors -- 16 in 3D, 32 in 2D -- with no stated
#: reason for either; unified at the larger, since too much overlap costs only
#: redundant computation while too little corrupts the result.
_ENHANCE_MIN_OVERLAP_PX = 32


def _dask_chunks(ndim: int) -> Tuple[int, ...]:
    """
    Dask chunk shape for the whole-array filtering and labelling steps.

    PINNED, not budget-driven, and the values are exactly what they always
    were. This one function feeds `gaussian_filter`, `binary_closing` and
    `ndmeasure.label`. The two filters go through `map_overlap` with a
    sigma-derived depth and should therefore be chunk-invariant, but `label`
    is not: it numbers each block independently and resolves equivalences
    afterwards, so the PARTITION is chunk-independent while the IDs are not.

    That is not cosmetic. Step 1 consumes those IDs structurally (`soma_lut`,
    the `1..maxid` walk in the size/seed filter) and, more seriously, step 3
    iterates clumps in ascending label order against a peak grid that holds
    only prior-label peaks -- so whichever clump is reached first places its
    soma and a nearby one from another clump is shrunk or dropped. Renumbering
    therefore changes which somata survive.

    Splitting this into "safe for filters, pinned for labelling" would rest on
    dask_image's overlap depth being provably sufficient, which is not
    verifiable from here. So the geometry is fixed and the speedup comes from
    `_dask_workers` instead, which cannot affect any result.

    See `resource_budget.PINNED['dask_chunk_shape_3d']`.
    """
    return resource_budget.pinned(
        'dask_chunk_shape_3d' if ndim == 3 else 'dask_chunk_shape_2d')


def _dask_workers(budget: "resource_budget.Budget", ndim: int,
                  operation: str = "float32_pass") -> int:
    """
    How many dask tasks may be in flight at once, given the RAM ceiling.

    The chunk shape is fixed (see `_dask_chunks`), so the only thing left to
    scale is concurrency -- and concurrency is exactly what was missing: dask's
    threaded scheduler defaults to one worker per core with no notion of how
    large a chunk is, which is the same "chunk size and worker count chosen
    independently" mistake the hardcoded constants made everywhere else.

    Bit-identical by construction: the number of threads changes which order
    chunks complete in, not what any chunk contains. The one caveat is that
    each thread holds a chunk, so peak memory is workers x chunk bytes, which
    is precisely what is being budgeted here.
    """
    chunk = _dask_chunks(ndim)
    per_element = resource_budget.cost_bytes_per_voxel(operation)
    per_chunk = per_element
    for v in chunk:
        per_chunk *= v
    workers = int(budget.plannable_bytes // max(1, int(per_chunk)))
    return max(1, min(budget.cores, workers))


#: Base chunk shape for the vesselness enhancement pass, scaled UP by the
#: budget. Unlike `_dask_chunks` this one is safe to scale, and the reason is
#: worth stating because it is the test every budget-driven geometry has to
#: pass: every operator in `_process_block_worker` has a bounded reach --
#: Frangi and Sato are Gaussian derivatives with `truncate=4` (reach 4*sigma),
#: `_crest_weight` samples at 2*sigma, and the recovery `grey_dilation` adds
#: about sigma -- and `overlap_px` is `max(32, ceil(4*sigma))`, which covers
#: the largest of them. Only the owned region is written, and at the true
#: volume boundary the read block is clipped identically whatever the chunk
#: shape, so `mode='nearest'` sees the same context. The output is therefore
#: Quantile of the surviving (off-floor) values used to recover the noise
#: scale. Low enough to sit inside the noise rather than in the signal, high
#: enough not to be set by a handful of voxels.
_FLOOR_PROBE_QUANTILE = 0.25

#: Above this fraction on the floor there is too little of the distribution
#: left to say anything about its width.
_FLOOR_MAX_FRACTION = 0.98

#: At or above this fraction sitting on one value, the MAD is not describing a
#: noise distribution and must not be used -- even when it is not exactly zero.
#: With exactly half the voxels clipped the MAD is set by the single boundary
#: element: measured 0.10 where the true noise was 62.2, which passed a
#: "greater than zero" test and would have divided the whole volume by it.
_FLOOR_MIN_FRACTION = 0.25


def _robust_scale(values, floor_tol: float = 0.0) -> float:
    """Noise scale of a residual, including when most of it sits on a floor.

    The plain estimate is 1.4826 x MAD, which is right whenever the bulk of the
    residual is background.

    It breaks on a CLIPPED residual. With `illumination_block_um` set, the
    illumination stage subtracts a background surface and clips at zero, so the
    majority of voxels are exactly 0. The median is then 0, every deviation
    from it is the value itself, and the MAD is 0 too. Two fallbacks have
    already failed here, both measured against an illumination stage that
    reported the true noise of the same image as 62.2:

        mean of the non-zero absolute values          1205.5
        median of the off-floor values / 0.6745        471.5

    The first averages signal. The second assumed the off-floor voxels were the
    positive half of zero-mean noise, but the background has ALREADY been
    subtracted, so the pooled median is 0, subtracting it changes nothing, and
    the off-floor values are the whole image -- tissue included.

    What clipping does leave behind is the FRACTION on the floor. If a fraction
    p of a N(0, sigma) distribution was clipped away, the surviving values are
    its upper 1-p, so the q-th quantile of what survives is the
    (p + q(1-p))-th quantile of the original, and

        sigma = value / Phi^-1(p + q(1-p))

    recovers the width from any one surviving quantile. `q` is taken low
    (`_FLOOR_PROBE_QUANTILE`) so the probe sits in the noise rather than in the
    tissue at the top of the distribution.

    Falls back to 1.0 when there is nothing off the floor, or when so much was
    clipped that the remainder says nothing about the width.

    KNOWN LIMIT: the probe quantile has to sit in the noise, so this fails once
    the tissue occupies more of the surviving distribution than the probe. At a
    58% floor it held to within 1% for tissue up to 15% of the volume and broke
    at 40% (reading 3241 for a true 62.2). A volume that is 40% tissue has
    little background left to measure.
    """
    v = np.asarray(values, dtype=np.float32).ravel()
    if v.size == 0:
        return 1.0

    on_floor = int(np.count_nonzero(v <= floor_tol))
    p = on_floor / float(v.size)

    if p < _FLOOR_MIN_FRACTION:
        mad = float(np.median(np.abs(v - float(np.median(v)))))
        scale = 1.4826 * mad
        if np.isfinite(scale) and scale >= 1e-6:
            return max(scale, 1e-6)

    off_floor = v[v > floor_tol]
    if off_floor.size == 0 or p >= _FLOOR_MAX_FRACTION:
        return 1.0

    probe = float(np.quantile(off_floor, _FLOOR_PROBE_QUANTILE))
    # Position of that probe in the ORIGINAL, unclipped distribution.
    original_q = p + _FLOOR_PROBE_QUANTILE * (1.0 - p)
    z = float(ndtri(min(max(original_q, 1e-6), 1.0 - 1e-6)))
    if z > 1e-6 and np.isfinite(probe) and probe > 0:
        return max(probe / z, 1e-6)
    return 1.0


#: Physical size of the local-background window, in microns, in every
#: direction. The opening that estimates the pedestal removes whatever is
#: SMALLER than its window, so this has to be comfortably larger than the
#: largest thing that must survive -- a cell body is tens of microns, so a
#: window of a few tens preserves it while still following background that
#: varies over hundreds. It is the same order as the illumination stage's own
#: `block_um`, which is the same quantity measured for the same reason.
#:
#: A length rather than a voxel count, so the correction does not change when
#: the same specimen is sampled more finely, and does not depend on the rank of
#: the data or on whether the vesselness filters are configured.
_BACKGROUND_WINDOW_UM = 50.0

#: Ceiling in voxels per axis. A window of tens of microns on very finely
#: sampled data would otherwise become hundreds of pixels across, at which
#: point the opening is slow and is no longer local to anything.
_BACKGROUND_WINDOW_MAX_PX = 151

#: Pixels sampled across the whole volume when estimating the volume-wide
#: background and noise sigma for Stage 1.1. An exact median would need the
#: volume resident; a sample of this size puts the estimate well inside the
#: rounding of the float32 arithmetic it feeds.
_STATS_SAMPLE_PIXELS = 4_000_000

#: Seeded, so the same stack yields the same normalisation on every run.
_STATS_SAMPLE_SEED = 42


#: byte-for-byte independent of this shape.
_ENHANCE_BASE_CHUNK_3D = (64, 512, 512)
_ENHANCE_BASE_CHUNK_2D = 2048


def _enhance_base_chunk_shape(ndim: int) -> Tuple[int, ...]:
    """Unscaled enhancement chunk shape: each track's historical default."""
    return (_ENHANCE_BASE_CHUNK_3D if ndim == 3
            else (_ENHANCE_BASE_CHUNK_2D,) * ndim)


def _traversal_chunk_shape(
    budget: "resource_budget.Budget",
    shape: Tuple[int, ...],
    ndim: int,
    operation: str = "float32_pass",
    name: str = "traversal",
) -> Tuple[int, ...]:
    """
    Chunk shape for a POINTWISE traversal: copies, casts, OR-merges, writes.

    These loops used to borrow `_enhance_chunk_shape`, which was never what it
    described -- they do not run the vesselness filter, they walk the array
    copying or OR-ing blocks. Nothing about a pointwise operation depends on
    where the block boundaries fall, so the shape is purely a memory choice and
    the budget sets it. On a small machine it lands near the old constant; on
    the workstation it is much larger, which is most of the win in these loops
    because they are dominated by per-block overhead rather than arithmetic.
    """
    plan = budget.plan_scaled_block(
        shape, _enhance_base_chunk_shape(ndim), operation,
        overlap=0, max_workers=1, name=name,
    )
    return plan.block_shape


# --------------------------------------------------------------------------
# Exact percentiles without materialising the sample
# --------------------------------------------------------------------------
#: Below this many sampled voxels the sample is small enough that materialising
#: it is free, so the original one-liner is used verbatim. Above it the
#: streaming path runs. Both produce bit-identical numbers (verified against
#: `np.percentile` over ~11k cases spanning uniform, heavy-tailed, tied and
#: degenerate float32 distributions), so the switch is a memory decision only.
_SAMPLE_INLINE_LIMIT = 32 << 20   # 32 Mi samples = 128 MB as float32


def _sample_blocks(source: np.ndarray, rows_per_block: int):
    """Yield contiguous float32 blocks of `source`, split along axis 0.

    `source` is the STRIDED view of the enhanced volume, so each block is a
    strided read that `ascontiguousarray` materialises one slab at a time.
    Order is preserved but irrelevant: every consumer below is either an order
    statistic or a count, both order-independent.
    """
    n_rows = source.shape[0]
    step = max(1, int(rows_per_block))
    for start in range(0, n_rows, step):
        blk = np.ascontiguousarray(source[start:start + step], dtype=np.float32)
        yield blk.ravel()


def _order_statistics_streaming(block_iter_factory, ranks, min_value: float):
    """
    Exact k-th smallest values of the sample, in O(1) memory.

    The trick is that POSITIVE IEEE-754 float32 values are order-isomorphic to
    their uint32 bit patterns: if x < y then bits(x) < bits(y). So a histogram
    over those patterns is a histogram over sorted position, and two passes are
    enough to pin an order statistic to an exact bit pattern -- one over the
    high 16 bits to find which of 65536 buckets the rank falls in, one over the
    low 16 bits within that bucket. The result is the exact value that would
    have been at that index of the fully sorted sample; nothing is estimated,
    interpolated or binned away.

    Every sample here is `> min_value >= 0`, so the positivity precondition
    holds. NaN and +inf sort above every finite value in bit order too, which
    matches how `np.sort` places them, but they cannot occur downstream of the
    normalisation and are not relied on.

    Returns ``(values_by_rank, n)``.
    """
    ranks = sorted({int(r) for r in ranks})
    hi_hist = np.zeros(1 << 16, dtype=np.int64)
    n = 0
    for blk in block_iter_factory():
        sel = blk[blk > min_value]
        if not sel.size:
            continue
        bits = sel.view(np.uint32)
        n += sel.size
        hi_hist += np.bincount(bits >> np.uint32(16), minlength=1 << 16)
    if n == 0:
        return {}, 0

    cum = np.cumsum(hi_hist)
    wanted: Dict[int, List[Tuple[int, int]]] = {}
    for k in ranks:
        kk = min(max(k, 0), n - 1)
        bucket = int(np.searchsorted(cum, kk, side='right'))
        below = int(cum[bucket - 1]) if bucket else 0
        wanted.setdefault(bucket, []).append((k, kk - below))

    out: Dict[int, np.float32] = {}
    for bucket, targets in wanted.items():
        lo_hist = np.zeros(1 << 16, dtype=np.int64)
        for blk in block_iter_factory():
            sel = blk[blk > min_value]
            if not sel.size:
                continue
            bits = sel.view(np.uint32)
            inside = bits[(bits >> np.uint32(16)) == np.uint32(bucket)]
            if inside.size:
                lo_hist += np.bincount(inside & np.uint32(0xFFFF),
                                       minlength=1 << 16)
        lcum = np.cumsum(lo_hist)
        for k, local_rank in targets:
            low = int(np.searchsorted(lcum, local_rank, side='right'))
            pattern = np.uint32((bucket << 16) | low)
            out[k] = np.array([pattern], dtype=np.uint32).view(np.float32)[0]
    return out, n


def _percentile_from_order_statistics(stats: Dict[int, Any], n: int,
                                      p: float) -> float:
    """
    `np.percentile(sample, p)` reproduced bit-for-bit from order statistics.

    Three details in numpy's implementation have to be copied exactly, and each
    one moves the last bit if it is not:

    1. The virtual index for method='linear' is ``(n - 1) * q`` with
       ``q = p / 100`` in float64.
    2. The two bracketing order statistics are combined by numpy's `_lerp`,
       which for ``gamma >= 0.5`` interpolates DOWN from the upper value
       (``b - diff*(1-gamma)``) instead of up from the lower one. Those are not
       the same expression in floating point.
    3. The arithmetic happens in FLOAT32, not float64. `gamma` reaches `_lerp`
       as a Python float, and under NEP 50 weak promotion a Python float does
       not upcast a float32 operand -- so the difference, the product and the
       sum all stay float32. Computing in float64 and rounding at the end gives
       a different answer.
    """
    q = np.float64(p) / np.float64(100)
    virtual = (n - 1) * q
    if virtual >= n - 1:
        lo = hi = n - 1
    elif virtual <= 0:
        lo = hi = 0
    else:
        lo = int(np.floor(virtual))
        hi = lo + 1
    gamma = float(virtual - np.floor(virtual)) if lo != hi else 0.0
    a = np.float32(stats[lo])
    b = np.float32(stats[hi])
    diff = np.float32(b - a)
    if gamma >= 0.5:
        return float(np.float32(b - diff * (1.0 - gamma)))
    return float(np.float32(a + diff * gamma))


def _count_above_streaming(block_iter_factory, min_value: float,
                           threshold: float) -> Tuple[int, int]:
    """``(count of sample > threshold, sample size)``, streamed.

    Replaces ``np.mean(samples > grow_thresh)``. `np.mean` of a boolean array
    accumulates 1.0s into a float64, which is exact for any count below 2**53,
    so a count divided by the total is the identical float.
    """
    above = 0
    total = 0
    for blk in block_iter_factory():
        sel = blk[blk > min_value]
        total += int(sel.size)
        if sel.size:
            above += int(np.count_nonzero(sel > threshold))
    return above, total


def enhance_tubular_structures_blocked(
    volume: np.ndarray,
    scales: List[float],
    spacing: Sequence[float],
    temp_root_path: Optional[str],
    black_ridges: bool = False,
    frangi_alpha: float = 0.5,
    frangi_beta: float = 0.5,
    frangi_gamma: float = 2.0,
    skip_tubular_enhancement: bool = False,
) -> Tuple[np.memmap, str, str]:
    """
    Vesselness enhancement over chunks, at either rank.

    The filter itself is two-dimensional in both tracks (see
    `_process_block_worker`), so the only rank-dependent choices here are the
    chunk shape and which spacing axes count as in-plane.

    `spacing` is ordered like the array axes and is required; sigmas are derived
    from it, so a substituted value would silently change the physical scale
    every filter responds to.
    """
    ndim = int(volume.ndim)
    spacing_float = normalise_spacing(spacing, ndim)
    print(f"  [Enhance] Data: {volume.shape}, Spacing: {spacing_float}")

    budget = resource_budget.open_budget("step 1.3 vesselness")
    output_temp_dir = _get_safe_temp_dir(temp_root_path, 'tubular_output')
    output_path = os.path.join(output_temp_dir, 'processed_volume.dat')
    output_memmap = np.memmap(output_path, dtype=np.float32, mode='w+', shape=volume.shape)

    if skip_tubular_enhancement:
        # A pure copy: no filter reach, so no overlap and a plain traversal
        # shape. Sized by the budget like every other pointwise loop.
        chunk_gen = _get_chunk_slices(
            volume.shape,
            _traversal_chunk_shape(budget, volume.shape, ndim,
                                   "float32_pass", "enhance passthrough"),
            overlap=0)
        for _, write_slice in tqdm(list(chunk_gen), desc="  [Enhance] Copying"):
            output_memmap[write_slice] = volume[write_slice].astype(np.float32)
        output_memmap.flush()
        return output_memmap, output_path, output_temp_dir

    # In-plane spacing: the last two axes at either rank. The 3D copy wrote
    # `spacing[1:]` and the 2D copy the whole tuple -- the same rule twice.
    xy_spacing = spacing_float[-2:]
    sigmas_voxel_2d = sorted([s / np.mean(xy_spacing) for s in scales if s > 0])
    if not sigmas_voxel_2d:
        return enhance_tubular_structures_blocked(
            volume, [], spacing_float, temp_root_path,
            skip_tubular_enhancement=True
        )

    overlap_px = max(_ENHANCE_MIN_OVERLAP_PX,
                     math.ceil(max(sigmas_voxel_2d) * 4))
    
    # Offset-aware and view-refusing; see `worker_memmap_handle`. The former
    # (filename, shape, dtype) triple read a tifffile memmap from byte 0 of the
    # TIFF, i.e. every voxel shifted by the header, and a crop view as a block
    # from the start of the file. Anything that cannot be reopened exactly is
    # copied to disk below, which was already the path for in-RAM input.
    input_info = worker_memmap_handle(volume)
    dump_dir = None
    if input_info is None:
        dump_dir = _get_safe_temp_dir(temp_root_path, 'input_dump')
        dump_path = os.path.join(dump_dir, 'input_dump.dat')
        input_mm = np.memmap(dump_path, dtype=volume.dtype, mode='w+', shape=volume.shape)
        input_mm[:] = volume[:]
        input_mm.flush()
        input_info = worker_memmap_handle(input_mm)
        if input_info is None:  # a fresh whole-file memmap always qualifies
            raise RuntimeError(f"could not hand {dump_path} to the vesselness workers")

    worker_func = partial(_process_block_worker, input_memmap_info=input_info, 
                          output_memmap_info=(output_path, 0, tuple(volume.shape),
                                              np.dtype(np.float32).str),
                          sigmas_voxel_2d=sigmas_voxel_2d, 
                          black_ridges=black_ridges, frangi_alpha=frangi_alpha, 
                          frangi_beta=frangi_beta, frangi_gamma=frangi_gamma)

    # Chunk shape AND worker count from one plan, which is the whole point:
    # the old code took the shape from a constant and the process count from
    # `os.cpu_count() - 2`, so peak memory was the product of two numbers that
    # never met. On a 16-core 8 GB laptop that product was already ~3.5 GB
    # before anything else in the app was counted.
    plan = budget.report(budget.plan_scaled_block(
        volume.shape, _enhance_base_chunk_shape(ndim), "vesselness",
        overlap=overlap_px, name=f"vesselness (overlap {overlap_px}px)",
    ))
    chunk_shape = plan.block_shape

    chunks = list(_get_chunk_slices(volume.shape, chunk_shape, overlap=overlap_px))
    pool = mp.Pool(processes=max(1, plan.workers), initializer=_init_worker)
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
    near_scale = max(8.0, 3.0 * float(link_radius))     # below this gap, proximity
    #                                                     dominates and the angle
    #                                                     (and forward-sense) tests
    #                                                     are relaxed
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
        # Decision tally (always on): why candidate pairs are/aren't linked, so a
        # single summary line explains the linker's behaviour on any image.
        rej_facing = rej_weak = rej_joined = rej_taken = rej_cross = 0
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
            # Forward-sense gate, graded by distance: at long range a stick must
            # point ahead (ai>=0, never backward through its own body); at short
            # range proximity overrules a slightly backward-pointing (bent/hooked)
            # tip, so the allowed backward margin `back` relaxes the sign the same
            # way the magnitude is relaxed, decaying to strict at near_scale.
            back = np.clip(1.0 - ss / near_scale, 0.0, 1.0)
            fwd = (ai >= -back) if not is_ball[i] else np.ones(len(jj), bool)
            fwd = fwd & np.where(is_ball[jj], True, aj >= -back)
            rej_facing += int((~fwd).sum())      # partner not ahead of endpoint
            if not fwd.any():
                continue
            jj = jj[fwd]; ss = ss[fwd]; u = u[fwd]; ai = ai[fwd]; aj = aj[fwd]
            # Alignment MAGNITUDE only: direction is already handled by the graded
            # forward-sense gate above, so a slightly-backward (bent) tip the gate
            # admitted is not additionally penalised in the affinity.
            align_i = np.abs(ai)
            align_j = np.abs(aj)
            # Distance-graded good continuation, with proximity dominance falling
            # off QUADRATICALLY: for a few-pixel gap the angle barely matters
            # (geom ~ 1), ramping to full collinearity at near_scale (still rejects
            # off-line noise at range). w = 0 near -> angle irrelevant; w = 1 far.
            w = np.clip(ss / near_scale, 0.0, 1.0) ** 2
            geom = (1.0 - w) + w * (align_i * align_j)
            satw = 0.5 + 0.5 * np.minimum(1.0, np.minimum(sal[i], sal[jj]) / sal_ref)
            A = geom * np.exp(-(ss ** 2) / (sigma ** 2)) * satw
            good = A >= A_min
            rej_weak += int((~good).sum())       # affinity below A_min
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
                rej_joined += 1              # fragments already in the same object
                continue
            if not _free(i, si, bool(is_ball[i])) or not _free(j, sj, bool(is_ball[j])):
                rej_taken += 1               # an endpoint side is already used
                continue
            blocked, absorb = _bridge_check(pos[i], pos[j], int(frag[i]), int(frag[j]))
            if blocked:
                rej_cross += 1               # bridge would cross a real structure
                continue
            parent[_find(frag[i])] = _find(frag[j])
            for _a in absorb:                # swallow tiny specks lying in the gap
                parent[_find(int(_a))] = _find(int(frag[j]))
                absorbed_labels.add(int(_a))
            used.setdefault(i, set()).add(si)
            used.setdefault(j, set()).add(sj)
            _emit_bridge(_route(pos[i], pos[j], absorb), int(frag[i]))
            n_link += 1
        print(f"    [DIAG] tensor-voting links: {n_link} | absorbed specks: "
              f"{len(absorbed_labels)} | tokens: {M}")
        print(f"    [DIAG] link rejections -> facing-away:{rej_facing} "
              f"weak-affinity:{rej_weak} already-joined:{rej_joined} "
              f"endpoint-taken:{rej_taken} crossing:{rej_cross}")

        # ---- Fallback: attach an unlinked tip to a nearby mask BODY --------- #
        # Tensor voting pairs endpoint-to-endpoint; a tip that lands on the
        # *flank* of another mask has no endpoint to pair with, so it is never a
        # candidate above. Such an orphan tip is attached here by proximity to
        # that mask's body, using the SAME `max_dist` budget as all other linking
        # (no new distance constant). A tip meeting a flank is an attachment, not
        # a continuation, so there is deliberately no collinearity/forward test --
        # but it is still `_bridge_check`-gated, so it cannot cut across a third
        # structure, and it only fires for tips the collinear linker left unlinked.
        R = int(np.ceil(max_dist))
        n_tip_body = 0
        for i in range(M):
            if is_ball[i] or i in used:
                continue                        # only tips left unlinked by voting
            ci = np.round(pos[i]).astype(int)
            lo = np.maximum(ci - R, 0)
            hi = np.minimum(ci + R + 1, shape)
            sl = tuple(slice(int(lo[k]), int(hi[k])) for k in range(ndim))
            win = np.asarray(final_mm[sl])
            fg = win != 0
            if not fg.any():
                continue
            ri = _find(int(frag[i]))
            labs = win[fg]
            good = {int(l) for l in np.unique(labs)
                    if _find(int(l)) != ri
                    and (absorb_below <= 0 or _label_size(int(l)) >= absorb_below)}
            if not good:
                continue
            keep = np.isin(labs, list(good))
            gco = (np.argwhere(fg) + lo)[keep]
            d2 = ((gco - ci) ** 2).sum(1)
            within = d2 <= max_dist * max_dist
            if not within.any():
                continue
            k = int(np.argmin(np.where(within, d2, np.inf)))
            tgt = gco[k].astype(float); vlab = int(win[fg][keep][k])
            blocked, absorb = _bridge_check(pos[i], tgt, int(frag[i]), vlab)
            if blocked:
                continue
            parent[_find(int(frag[i]))] = _find(vlab)
            for _a in absorb:
                parent[_find(int(_a))] = _find(vlab); absorbed_labels.add(int(_a))
            used.setdefault(i, set()).add(1)
            _emit_bridge(_route(pos[i], tgt, absorb), int(frag[i]))
            n_tip_body += 1
        print(f"    [DIAG] tip->body attachments: {n_tip_body}")

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
    spacing: Sequence[float],
    *,
    tubular_scales: List[float],
    smooth_sigma: Union[float, List[float]],
    connect_max_gap_physical: Union[float, List[float]],
    low_threshold_percentile: Union[float, List[float]],
    high_threshold_percentile: Union[float, List[float]],
    threshold_mode: str,
    trace_max_gap: float,
    min_size: int,
    skip_tubular_enhancement: bool = False,
    temp_root_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], float, Dict[str, Any]]:
    """Step 1: Raw Segmentation (Independent per-scale Smoothing + Threshold-then-OR).

    Smoothing and gap-closing are applied independently per tubular scale
    (mirroring the 2D pipeline); ``smooth_sigma`` and ``connect_max_gap_physical``
    may each be a scalar (broadcast to every scale) or a per-scale list. With a
    single scalar value the result is identical to the previous global behavior.

    Unlike the 2D pipeline, the minimum-size filter is applied GLOBALLY, once,
    AFTER all scales are merged (see the labeling stage), so ``min_size``
    remains a single value rather than a per-scale list.
    """
    # Config-owned parameters are keyword-only with NO defaults. Every one of
    # them is supplied by the YAML in the app, so a default here would be a
    # second, invisible place to configure the pipeline -- and the two tracks had
    # already drifted to different values (low/high percentiles of 25/95 in 3D
    # against 95/100 in 2D) that nothing ever read. Omitting one is now an error
    # at the call site rather than a silent substitution.
    #
    # `min_size` is the config's own key. The former tracks spelled it
    # `min_size_voxels` and `min_size_pixels`; those aliases arrived through
    # `**kwargs`, which is gone, so a wrong name is now a TypeError.
    min_size_voxels = int(min_size)

    # Rank from the data. Drives chunk shapes, the plane loop and which spacing
    # axes count as in-plane.
    ndim = int(volume.ndim)
    if ndim not in (2, 3):
        raise ValueError(
            f"raw segmentation handles 2D and 3D data; got a {ndim}D array"
        )
    spacing = normalise_spacing(spacing, ndim)

    print(f"\n--- Step 1: Raw Segmentation (Strict Independence Mode) ---")
    # One budget for the whole step. Opened here rather than per stage so the
    # log carries the machine and the ceiling once, and so every stage below
    # plans against the same figure.
    _budget = resource_budget.open_budget("step 1 raw segmentation")
    print(resource_budget.describe_environment(_budget.settings))
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
                _trav = _traversal_chunk_shape(_budget, volume.shape, ndim,
                                               "float32_pass", "dtype scaling")
                for read_sl, _ in tqdm(list(_get_chunk_slices(volume.shape, _trav)), desc="    Applying"):
                    norm_mm[read_sl] = volume[read_sl].astype(np.float32) / norm_factor
                norm_mm.flush()
            else:
                # --- Relative mode: LOCAL background + LOCAL-noise standardization ---
                # (Item 1) Replaces the old per-z global brightness gain. Each voxel
                # becomes its height above the LOCAL background divided by the LOCAL
                # response scale, i.e. a per-region SNR. This flattens patchy /
                # structured background (e.g. nonspecific antibody staining) per
                # region instead of assuming a single uniform noise level, so the
                # threshold stage sees comparable statistics across clean and noisy
                # images.
                # Window selection (no GUI knob): auto-derive a window a few times
                # the LARGEST tubular scale (converted to xy pixels), so the opening
                # removes regional background while preserving real processes; if no
                # usable scale exists, fall back to per-slice global stats.
                # ---- ONE TRACK AT BOTH RANKS ------------------------------
                # The statistics below used to be per PLANE. At rank 2 that is
                # the whole image, because there is one plane; at rank 3 it was
                # one background and one sigma per z-slice, which is a different
                # correction from the one 2D receives.
                #
                # It also cancelled the z correction outright. Scale a plane by
                # any constant c and its background scales by c, its residual by
                # c and its MAD sigma by c, so (c*x - c*bg) / (c*sigma) is
                # exactly what it was -- verified to float32 rounding over
                # factors 0.5 to 6. `correct_illumination`'s z step multiplies
                # each plane by exactly such a constant, so a per-plane sigma
                # undid it completely and the threshold stage saw the same data
                # either way. Measured, plane 0 versus plane 23 of an
                # attenuating stack, as a ratio of normalised tissue value:
                #
                #     per-plane,   z correction OFF    2.24
                #     per-plane,   z correction ON     2.24   no effect at all
                #     volume-wide, z correction OFF    2.24
                #     volume-wide, z correction ON     0.99   flattened
                #
                # Note what this does NOT say: a per-plane sigma does not erase
                # raw depth attenuation, because attenuation dims the tissue
                # while leaving the camera pedestal where it is, which is not a
                # uniform scaling. Both rows read 2.24 with the correction off.
                # What the per-plane sigma erases is specifically the CORRECTION.
                #
                # Volume-wide statistics make rank 3 behave as rank 2 does and
                # let depth reach the threshold, so the z correction becomes
                # load-bearing rather than cancelled. At rank 2 the numbers are
                # unchanged by construction: one plane means volume-wide and
                # per-plane are the same estimate.
                #
                # `win > 0` keeps its LOCAL windowed background, which is what
                # 2D does too -- local in XY, not per plane. Only the noise
                # sigma becomes volume-wide there.
                #
                # Set False to get the old per-plane behaviour back for
                # comparison on the same stack. Nothing else changes either way.
                _VOLUME_WIDE_STATS = True

                # ---- THE BACKGROUND WINDOW ------------------------------
                # A fixed PHYSICAL size, converted to voxels per axis.
                #
                # It used to be derived from `tubular_scales`, as six times the
                # largest one, and to collapse to NO window at all when no scale
                # was set -- `win = 0`, which fell back to a single global
                # background for the whole plane. That made the entire local
                # background estimate conditional on the vesselness filters
                # being configured, which is wrong twice over: the filters are
                # optional, and a background pedestal exists whether or not
                # anything is being enhanced. On a run with `Scale sigma=0.0`
                # there was no local background subtraction anywhere.
                #
                # `_BACKGROUND_WINDOW_UM` replaces that. It is a length in
                # microns, so the same tissue gives the same correction at any
                # pixel size, any z step, any rank, and with the filters on or
                # off. When tubular scales ARE configured the window is widened
                # to cover them if they ask for more, so a run that previously
                # got a wider window still gets it.
                #
                # Converted PER AXIS, which on anisotropic data is very
                # different numbers for the same distance: at 0.276 um pixels
                # and a 2 um step, 50 um is 181 pixels in plane but only 25
                # planes deep. An isotropic voxel window would reach 181 planes
                # -- 362 um -- and open away the specimen itself.
                #
                # A z extent of 1 means one plane, i.e. the old 2D behaviour,
                # which is what a 2D image gets and what a stack whose z step
                # is coarser than the window gets.
                phys = max([s for s in tubular_scales if s and s > 0], default=0.0)
                window_um = max(_BACKGROUND_WINDOW_UM, 6.0 * phys)

                def _axis_window(step_um: float) -> int:
                    if not (step_um > 0):
                        return 1
                    n = int(round(window_um / step_um))
                    n = max(1, min(n, _BACKGROUND_WINDOW_MAX_PX))
                    return n if n % 2 == 1 else n + 1

                # Finest in-plane axis: the last two at either rank. No 1e-9
                # clamp -- `spacing` is validated positive and finite on entry,
                # so a guard here would only mask a bad value.
                xy_spacing = min_inplane_spacing(spacing)
                win = _axis_window(xy_spacing)
                win_z = (_axis_window(float(spacing[0]))
                         if volume.ndim == 3 else 1)

                bg_report: List[float] = []
                sc_report: List[float] = []
                # Normalize per FULL z-slice. We iterate z directly instead of tiling
                # the XY plane into (…, 512, 512) blocks: the background and the noise
                # sigma below are slice-GLOBAL statistics, so any XY tiling here would
                # estimate a *different* sigma per tile and bake a visible per-tile
                # strictness seam into the downstream (global) percentile threshold.
                # Z-slices are independent, so looping z is both correct and
                # memory-safe -- only one full XY plane is resident at a time.
                # Per-plane normalisation. In 3D these are the Z slices; in 2D
                # there is one plane and the loop runs once. Background and noise
                # sigma are plane-GLOBAL statistics, which is why the plane is
                # the unit and XY tiling would be wrong: each tile would estimate
                # a different sigma and bake a per-tile strictness seam into the
                # downstream global percentile threshold.
                _planes = list(planes_of(volume))

                # One pass to estimate the volume-wide statistics, before the
                # pass that applies them. Streamed a plane at a time like
                # everything else here; only a bounded sample of each plane is
                # kept, so the memory cost is the sample and not the volume.
                #
                # A sample, because a median over a 192 x 24615 x 18462 volume
                # cannot be taken exactly without holding it. Seeded, so two
                # runs of the same stack give the same correction.
                _vol_bg = None
                _vol_sigma = None
                if _VOLUME_WIDE_STATS and len(_planes) > 1:
                    _rng = np.random.default_rng(_STATS_SAMPLE_SEED)
                    _per_plane = max(1, _STATS_SAMPLE_PIXELS // len(_planes))
                    _samples = []
                    for _pidx, _plane_src in tqdm(_planes, desc="    Sampling",
                                                  total=len(_planes)):
                        _flat = np.asarray(_plane_src, dtype=np.float32).ravel()
                        if _flat.size == 0:
                            continue
                        if _flat.size > _per_plane:
                            _flat = _rng.choice(_flat, _per_plane, replace=False)
                        _samples.append(_flat)
                    if _samples:
                        _pool = np.concatenate(_samples)
                        del _samples
                        if win > 0:
                            # The background stays local and per plane (as in
                            # 2D); only the noise scale is shared. Estimating it
                            # from the raw pool would include the pedestal the
                            # opening removes, so the residual is formed against
                            # the pool's own median, which is what the per-plane
                            # code compares against when win == 0.
                            _vol_bg = None
                        else:
                            _vol_bg = float(np.median(_pool))
                        _centre = float(np.median(_pool))
                        _vol_sigma = _robust_scale(_pool - _centre)
                        del _pool
                        print(f"    [Relative/local-SNR] volume-wide statistics: "
                              f"background="
                              f"{'local per plane' if _vol_bg is None else f'{_vol_bg:.3f}'}"
                              f", noise sigma={_vol_sigma:.3f} "
                              f"(one value for all {len(_planes)} planes)")
                # Three plane buffers, allocated ONCE and reused, with the
                # arithmetic done in place. The previous version allocated a
                # fresh float32 plane for each of `s2d`, `bg`, `resid` and
                # `absr` on every iteration, and `np.clip(...)/sigma` added two
                # more, so peak was about six plane-sized buffers plus the copy
                # `np.median` makes internally to partition. At the 24615x18462
                # cross-section this module's own comments cite, that is ~1.8 GB
                # each and roughly 11 GB peak -- infeasible on an 8 GB machine
                # for reasons no chunk setting can fix.
                #
                # Every operation below is the same operation on the same
                # values: `np.subtract(x, y, out=x)` is `x - y`, and
                # `np.clip(x, 0, None, out=x); x /= sigma` is
                # `np.clip(x, 0, None) / sigma`. In-place only changes WHERE
                # the result lands. `sigma` and `bg` stay Python floats so NEP
                # 50 weak promotion keeps everything float32, exactly as before.
                #
                # `_s_buf` must be a real copy rather than the `np.asarray`
                # view the old code used: for a float32 input volume that
                # returned a view onto the source memmap, and writing through
                # it in place would corrupt the input.
                # ---- 3D SLABS, when the window has a z extent --------------
                # The halo is `win_z - 1` planes on each side, NOT `win_z // 2`.
                # An opening is an erosion FOLLOWED BY a dilation, and each of
                # those reaches `win_z // 2` planes, so the composition reaches
                # twice as far. Measured: with a 3-plane z window and a 7-pixel
                # in-plane one, a halo of 1 disagreed with the whole-volume
                # opening by up to 12.9 in 3 of 4 slab sizes tried, while a halo
                # of 2 was exact. It happens to pass for a wide in-plane window,
                # where the in-plane extreme dominates and the z context never
                # decides the result -- which is exactly the kind of accident
                # that makes a halo bug survive testing.
                #
                # With the correct halo this is EXACT, not approximate:
                # `grey_opening` of a slab equals the opening of the whole
                # volume everywhere the halo is complete, and where it is not --
                # the first and last slab -- the slab edge IS the volume edge,
                # so the same boundary mode applies as to a single whole-volume
                # call. Verified against the whole-volume result for z windows
                # of 1, 3, 5, 7, 9 and 13 at four slab sizes each.
                #
                # No XY tiling, for the reason the per-plane comments give: the
                # sigma is a global statistic and a tile would estimate its own,
                # baking a per-tile strictness seam into the global percentile.
                # Slabs split z only, and z carries no such statistic.
                #
                # Slab depth comes from the budget. Two slab-sized float32
                # buffers are live at once (the input and the opening's output),
                # and a plane of this module's cited 24615 x 18462 cross-section
                # is 1.8 GB on its own, so a 13-plane halo would be 23 GB before
                # any interior. When even the minimum slab does not fit, the z
                # extent is reduced and the reduction is reported rather than
                # silently swapping.
                _halo = max(0, int(win_z) - 1) if (win > 0 and volume.ndim == 3) else 0
                _fit = 1
                if _halo > 0 and len(_planes) > 1:
                    _pshape = _planes[0][1].shape
                    _pbytes = 4 * int(np.prod(_pshape))
                    _fit = max(1, int(_budget.plannable_bytes // (2 * _pbytes)))
                    # Reduce the window, not the halo, when the slab will not
                    # fit: a halo shorter than `win_z - 1` is not a cheaper
                    # approximation, it is a different answer from the one the
                    # window asks for.
                    _want = int(win_z)
                    while win_z > 1 and (2 * (win_z - 1) + 1) > _fit:
                        win_z -= 2
                    win_z = max(1, win_z)
                    _halo = max(0, win_z - 1)
                    if win_z != _want:
                        print(f"    [resources] 3D background window reduced "
                              f"from {_want} to {win_z} planes deep: the "
                              f"{2 * (_want - 1) + 1}-plane slab it needs at "
                              f"{_pshape} exceeds the "
                              f"{_budget.plannable_bytes / (1024 ** 3):.2f} GB "
                              f"available to this step")

                if _halo > 0:
                    _depth = len(_planes)
                    if 2 * _halo + 1 >= _depth:
                        # The window reaches further than the stack is deep, so
                        # one slab IS the whole volume and the halo is moot.
                        # Without this the step below would be 1 and the whole
                        # volume would be re-opened once per plane: at a 0.3 um
                        # z step a 50 um window is 151 planes, so a 300-plane
                        # halo on a 200-plane stack is not a corner case.
                        _step = _depth
                    else:
                        _step = max(1, _fit - 2 * _halo)
                    _size = (win_z, win, win)
                    print(f"    [Relative/local-SNR] 3D background window "
                          f"{win_z} x {win} x {win} voxels "
                          f"({window_um:.1f} um in every direction), "
                          f"slabs of {_step} planes + {_halo} halo")
                    for _start in tqdm(range(0, _depth, _step),
                                       desc="    Standardizing",
                                       total=(_depth + _step - 1) // _step):
                        _stop = min(_depth, _start + _step)
                        _lo = max(0, _start - _halo)
                        _hi = min(_depth, _stop + _halo)
                        _slab = np.asarray(volume[_lo:_hi], dtype=np.float32)
                        _bg = ndimage.grey_opening(_slab, size=_size)
                        np.subtract(_slab, _bg, out=_slab)
                        del _bg
                        if _vol_sigma is not None:
                            sigma = _vol_sigma
                        else:
                            sigma = _robust_scale(_slab)
                        np.clip(_slab, 0.0, None, out=_slab)
                        np.divide(_slab, sigma, out=_slab)
                        norm_mm[_start:_stop] = _slab[_start - _lo:_stop - _lo]
                        # One entry per plane written, so the medians below mean
                        # the same thing as they did per plane.
                        for _ in range(_stop - _start):
                            bg_report.append(0.0)
                            sc_report.append(sigma)
                        del _slab
                    norm_mm.flush()
                    win_desc = f"{win_z}x{win}x{win}"
                    print(f"    [Relative/local-SNR] window={win_desc}px | "
                          f"median noise sigma={np.median(sc_report):.3f} "
                          f"(map is now in noise-sigma units)")
                    _planes = []

                _plane_shape = None
                _s_buf = _bg_buf = _absr_buf = None
                _plane_bytes = None
                # `_planes` is emptied by the slab path above, so this loop is
                # skipped entirely rather than drawing a second empty progress
                # bar for zero iterations.
                for _pidx, _plane_src in (
                        tqdm(_planes, desc="    Standardizing",
                             total=len(_planes)) if _planes else ()):
                    if _plane_shape != _plane_src.shape:
                        _plane_shape = _plane_src.shape
                        _s_buf = np.empty(_plane_shape, dtype=np.float32)
                        _bg_buf = (np.empty(_plane_shape, dtype=np.float32)
                                   if win > 0 else None)
                        _absr_buf = np.empty(_plane_shape, dtype=np.float32)
                        # Four plane-sized buffers live at peak: these three
                        # plus the copy np.median makes. Reported rather than
                        # worked around, because the plane is genuinely the unit
                        # here -- the background and the noise sigma are
                        # plane-GLOBAL statistics and tiling them would bake a
                        # per-tile strictness seam into the global percentile.
                        _plane_bytes = 4 * int(np.prod(_plane_shape))
                        _needed = 4 * _plane_bytes
                        if _needed > _budget.plannable_bytes:
                            print(
                                f"    [resources] *** one {_plane_shape} plane "
                                f"needs ~{_needed / (1024 ** 3):.2f} GB of "
                                f"working buffers, above the "
                                f"{_budget.plannable_bytes / (1024 ** 3):.2f} GB "
                                f"available to this step. Normalisation will be "
                                f"attempted anyway and may swap or fail. Raise "
                                f"the RAM ceiling in Settings. ***"
                            )
                    np.copyto(_s_buf, _plane_src, casting='unsafe')
                    s2d = _s_buf
                    # Local background: removes the (possibly structured) pedestal
                    # while preserving structures smaller than the window.
                    if win > 0:
                        bg = ndimage.grey_opening(s2d, size=(win, win),
                                                  output=_bg_buf)
                        bg_center = float(np.median(bg))
                    elif _vol_bg is not None:
                        bg = _vol_bg
                        bg_center = bg
                    else:
                        bg = float(np.median(s2d))
                        bg_center = bg
                    np.subtract(s2d, bg, out=s2d)
                    resid = s2d

                    # Noise scale: a SINGLE robust GLOBAL value per FULL slice, NOT a
                    # spatially-varying local RMS. A local RMS is dominated by the
                    # signal itself -- it inflates around bright cells (suppressing
                    # them and haloing) and, on clean backgrounds, its floor
                    # collapses toward zero and amplifies noise into large "clouds".
                    # MAD is robust to the sparse bright tail, so it estimates the
                    # background noise even with cells present. Fallback covers the
                    # degenerate case of a large constant region (e.g. zero-padding
                    # outside the FOV) where the MAD would otherwise be 0.
                    if _vol_sigma is not None:
                        sigma = _vol_sigma
                    else:
                        sigma = _robust_scale(resid)

                    np.clip(resid, 0.0, None, out=resid)
                    np.divide(resid, sigma, out=resid)
                    if volume.ndim == 3:
                        norm_mm[_pidx] = resid
                    else:
                        norm_mm[...] = resid
                    # Accumulated at BOTH ranks. The 2D track omitted these, so
                    # its log gave no way to tell a failed normalisation from a
                    # correct one.
                    bg_report.append(bg_center)
                    sc_report.append(sigma)
                if _halo <= 0:
                    norm_mm.flush()
                    win_desc = str(win) if win > 0 else "global(per-slice)"
                    print(f"    [Relative/local-SNR] window={win_desc}px | "
                          f"median local background={np.median(bg_report):.3f}, "
                          f"median noise sigma={np.median(sc_report):.3f} "
                          f"(map is now in noise-sigma units)")

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

        # Optional hysteresis seed gate, OFF by default. It is enabled only when
        # `high` is a STRICTER percentile than `low` and below 100. With the shipped
        # default (high = 100) detection is a single percentile threshold on `low`
        # and nothing can be dropped by seeding. When enabled, only components
        # containing a `high`-percentile seed survive (connectivity-based rejection).
        use_seed_gate = any(
            low_thresh_list[i] < high_thresh_list[i] < 100.0 for i in range(n_scales)
        )
        seed_mm = None
        if threshold_mode != "Absolute" and use_seed_gate:
            seed_dir = _get_safe_temp_dir(temp_root_path, 'seed'); temp_dirs_to_clean.append(seed_dir)
            seed_mm = np.memmap(os.path.join(seed_dir, 'seed.dat'), dtype=np.uint8, mode='w+', shape=volume.shape)
            seed_mm[:] = 0

        for i, scale in enumerate(tubular_scales):
            current_low_p = low_thresh_list[i]
            current_high_p = high_thresh_list[i]
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
                    d_norm = da.from_array(norm_mm, chunks=_dask_chunks(ndim))
                    d_smooth = dask_image.ndfilters.gaussian_filter(d_norm, sigma=sigma_vox)

                    # `num_workers` is the budgeted quantity here, not the chunk
                    # shape (see `_dask_chunks`). Thread count changes the order
                    # chunks complete in, never their contents.
                    _nw = _dask_workers(_budget, ndim, "float32_pass")
                    with ProgressBar(dt=5):
                        da.store(d_smooth, smoothed_mm, scheduler='threads',
                                 num_workers=_nw)
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
                        temp_root_path=temp_root_path
                    )
                
                # Independent Thresholding Pass
                # Subsampling for the percentile estimate. The in-plane rule is
                # the same at both ranks -- the 2D track wrote `min(shape)` and
                # the 3D track `min(shape[1:])`, which agree once "in-plane"
                # means the last two axes. Only 3D has a leading axis to stride.
                stride_xy = max(1, min(16, min(volume.shape[-2:]) // 128))
                stride_lead = (max(1, min(4, volume.shape[0] // 32))
                               if ndim == 3 else None)
                sample_sel = (((slice(None, None, stride_lead),) if ndim == 3 else ())
                              + (slice(None, None, stride_xy),) * 2)
                
                seed_thresh = None
                if threshold_mode == "Absolute":
                    grow_thresh = current_low_p
                    grow_thresh = max(grow_thresh, 1e-5); threshold_history[scale] = grow_thresh
                    print(f"      [Scale {scale}] Absolute Threshold: {grow_thresh:.6f}")
                else:
                    # Relative: `low`/`high` are PERCENTILES (intuitive). They are
                    # applied to the item-1 SNR-normalized response, so a fixed
                    # percentile lands at a consistent level across images of
                    # different noise (item 1 flattened the noise scale). `low` is
                    # the detection threshold; `high` optionally adds a stricter
                    # connectivity seed (only when low < high < 100; off at high=100).
                    # The sample is NEVER materialised as a whole.
                    #
                    # `sample_sel` strides by at most 16 in-plane and 4 along
                    # the leading axis, so the number of samples grows LINEARLY
                    # with the volume: at brain scale `enh_mm[sample_sel]` is
                    # tens of GB, and `np.percentile` then copies it again to
                    # partition. Those strides cannot be raised to bound it,
                    # because the sampled SET is the threshold estimate -- see
                    # `resource_budget.PINNED['threshold_sample_stride_*']`.
                    #
                    # So the sampled set is unchanged and only the arithmetic
                    # moves out of core. Sampling remains a strided view; the
                    # percentiles come from exact order statistics found by a
                    # two-level histogram over float32 bit patterns. The
                    # numbers are bit-identical to the previous
                    # `np.percentile` calls, so thresholds -- and therefore
                    # every downstream mask -- are unchanged.
                    _sample_view = enh_mm[sample_sel]
                    _n_sample_est = int(np.prod(_sample_view.shape))
                    _rows = max(1, int(
                        _budget.plannable_bytes
                        // max(1, 4 * int(np.prod(_sample_view.shape[1:])))))
                    _blocks = (lambda: _sample_blocks(_sample_view, _rows))

                    if _n_sample_est <= _SAMPLE_INLINE_LIMIT:
                        # Small enough that a copy is free; keep the original
                        # expression verbatim rather than route the common case
                        # through new code.
                        samples = _sample_view.ravel()
                        samples = samples[samples > 1e-7]
                        _n = int(samples.size)

                        def _pctl(_p, _s=samples):
                            return float(np.percentile(_s, _p))

                        def _occupancy(_t, _s=samples):
                            return float(np.mean(_s > _t)) * 100.0
                    else:
                        _stats_cache: Dict[int, Any] = {}
                        _n_holder = _order_statistics_streaming(_blocks, [0], 1e-7)
                        _n = _n_holder[1]

                        def _pctl(_p, _n=_n):
                            q = np.float64(_p) / np.float64(100)
                            virtual = (_n - 1) * q
                            if virtual >= _n - 1:
                                need = [_n - 1]
                            elif virtual <= 0:
                                need = [0]
                            else:
                                need = [int(np.floor(virtual)),
                                        int(np.floor(virtual)) + 1]
                            missing = [k for k in need if k not in _stats_cache]
                            if missing:
                                got, _ = _order_statistics_streaming(
                                    _blocks, missing, 1e-7)
                                _stats_cache.update(got)
                            return _percentile_from_order_statistics(
                                _stats_cache, _n, _p)

                        def _occupancy(_t):
                            above, total = _count_above_streaming(
                                _blocks, 1e-7, _t)
                            return (float(above) / float(total) * 100.0
                                    if total else 0.0)

                        print(f"      [Scale {scale}] {_n} sampled voxels; "
                              f"percentiles computed out of core "
                              f"(sample would be "
                              f"{_n * 4 / (1024 ** 3):.2f} GB in RAM)")

                    if _n > 1000:
                        low_p = min(max(float(current_low_p), 0.0), 100.0)
                        grow_thresh = max(_pctl(low_p), 1e-5)
                        high_p = float(current_high_p)
                        if low_p < high_p < 100.0:
                            seed_thresh = max(_pctl(high_p), grow_thresh)
                        occ = _occupancy(grow_thresh)
                        seed_msg = (f", seed(p{high_p:g})={seed_thresh:.5f}"
                                    if seed_thresh is not None else ", seed OFF")
                        print(f"      [Scale {scale}] grow(p{low_p:g})={grow_thresh:.5f} "
                              f"[{occ:.2f}% of sampled]{seed_msg}")
                    else:
                        grow_thresh = 1e9
                        print(f"      [Scale {scale}] too few voxels to estimate threshold; scale skipped.")
                    threshold_history[scale] = grow_thresh

                # Binary Creation, Closing, and OR-ing
                if grow_thresh < 1e6:
                    enh_dask = da.from_array(enh_mm, chunks=_dask_chunks(ndim))

                    # Per-scale gap-closing structure
                    rv = [math.ceil((current_connect_gap / 2) / s) if s > 1e-9 else 0 for s in spacing]
                    struct = np.ones(tuple(max(1, 2 * r + 1) for r in rv), dtype=bool)

                    clean_dask = dask_image.ndmorph.binary_closing((enh_dask > grow_thresh), structure=struct)
                    _nw = _dask_workers(_budget, ndim, "binary_morphology")
                    
                    record_s0 = (scale == 0 and scale0_mm is not None)
                    _trav = _traversal_chunk_shape(
                        _budget, volume.shape, ndim, "binary_morphology",
                        "threshold/close merge")
                    for read_sl, _ in tqdm(list(_get_chunk_slices(volume.shape, _trav)), desc="      Merging"):
                        blk = clean_dask[read_sl].compute(
                            scheduler='threads', num_workers=_nw).astype(np.uint8)
                        master_mm[read_sl] |= blk
                        if record_s0:
                            scale0_mm[read_sl] |= blk
                        if seed_mm is not None and seed_thresh is not None:
                            seed_mm[read_sl] |= (enh_mm[read_sl] > seed_thresh).astype(np.uint8)
                
                master_mm.flush()
                if scale0_mm is not None:
                    scale0_mm.flush()
                if seed_mm is not None:
                    seed_mm.flush()

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
        m_dask = da.from_array(master_mm, chunks=_dask_chunks(ndim))
        labeled_dask, num_feats_dask = dask_image.ndmeasure.label((m_dask > 0), structure=binary_structure(ndim))
        # Labelling geometry is pinned; only the thread count is budgeted.
        _nw = _dask_workers(_budget, ndim, "copy_int32")
        with dask.config.set(scheduler='threads', num_workers=_nw):
            labeled_dask.to_zarr(os.path.join(lab_dir, 'l.zarr'), overwrite=True)
            num_feats = num_feats_dask.compute()

        if trace_max_gap > 0.0:
            # Trace-link BEFORE the size filter: write raw labels, reconnect
            # fragments broken by dim gaps, THEN apply the global size filter to
            # the merged labels. Memory-light throughout.
            lz = zarr.open(os.path.join(lab_dir, 'l.zarr'), mode='r')
            _trav = _traversal_chunk_shape(_budget, volume.shape, ndim,
                                           "copy_int32", "label writeout")
            for rs, ws in tqdm(list(_get_chunk_slices(volume.shape, _trav)), desc="    Writing labels"):
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
            shits = np.zeros(maxid + 1, dtype=np.int64) if seed_mm is not None else None
            for z in range(int(volume.shape[0])):
                row = final_mm[z].ravel()
                csz += np.bincount(row, minlength=maxid + 1)
                if shits is not None:
                    sel = seed_mm[z].ravel().astype(bool)
                    if sel.any():
                        shits += np.bincount(row[sel], minlength=maxid + 1)
            # Hysteresis gate: a label must contain a seed (high-sigma) voxel. If no
            # seeds exist at all (e.g. thresholds mis-tuned), skip the gate rather
            # than wipe the image, and warn.
            seed_gate = shits
            if shits is not None and int(shits[1:].sum()) == 0:
                print("    [WARN] no seed voxels; skipping significance (seed) gate this run.")
                seed_gate = None
            lut = np.zeros(maxid + 1, dtype=np.int32)
            nid = 0
            for i in range(1, maxid + 1):
                if csz[i] >= min_size_voxels and (seed_gate is None or seed_gate[i] > 0):
                    nid += 1
                    lut[i] = nid
            for z in range(int(volume.shape[0])):
                final_mm[z] = lut[final_mm[z]]
            if seed_gate is not None:
                print(f"    [DIAG] hysteresis: kept {nid} seeded objects "
                      f"(of {int((csz[1:] >= min_size_voxels).sum())} size-passing).")
            final_mm.flush()
        else:
            d_lbl = da.from_zarr(os.path.join(lab_dir, 'l.zarr'))
            counts, _ = da.histogram(d_lbl, bins=num_feats+1, range=[-0.5, num_feats+0.5])
            size_counts = counts.compute()[1:]
            size_ok = size_counts >= min_size_voxels

            # Hysteresis gate: keep only components containing a seed (high-sigma)
            # voxel. Streamed one z-slab at a time so full arrays never sit in RAM.
            lz = zarr.open(os.path.join(lab_dir, 'l.zarr'), mode='r')
            keep = size_ok
            if seed_mm is not None:
                shits = np.zeros(num_feats + 1, dtype=np.int64)
                for z in range(int(volume.shape[0])):
                    sel = seed_mm[z].ravel().astype(bool)
                    if sel.any():
                        shits += np.bincount(np.asarray(lz[z]).ravel()[sel], minlength=num_feats + 1)
                if int(shits[1:].sum()) == 0:
                    print("    [WARN] no seed voxels; skipping significance (seed) gate this run.")
                else:
                    keep = size_ok & (shits[1:] > 0)
                    print(f"    [DIAG] hysteresis: kept {int(keep.sum())} seeded objects "
                          f"(of {int(size_ok.sum())} size-passing).")
            valid = np.where(keep)[0] + 1

            lookup = np.zeros(num_feats + 1, dtype=np.int32)
            for i, old_id in enumerate(valid): lookup[old_id] = i + 1

            _trav = _traversal_chunk_shape(_budget, volume.shape, ndim,
                                           "copy_int32", "size filter")
            for rs, ws in tqdm(list(_get_chunk_slices(volume.shape, _trav)), desc="    Filtering"):
                final_mm[ws] = lookup[lz[rs]]
            final_mm.flush()

        # Explicitly release large internal memmaps
        if 'master_mm' in locals():
            del master_mm
        if seed_mm is not None:
            del seed_mm; seed_mm = None

        return labels_path, labels_temp_dir, threshold_history.get(tubular_scales[0], 0.0), {'threshold_history': threshold_history}

    finally:
        if final_labels_memmap is not None: del final_labels_memmap
        # Close any local memmap handles to release file locks
        for var in ['final_mm', 'norm_mm', 'smoothed_mm', 'master_mm', 'input_mm', 'enh_mm', 'scale0_mm', 'seed_mm']:
            if var in locals():
                try: del locals()[var]
                except: pass
                
        for d in temp_dirs_to_clean:
            shutil.rmtree(d, ignore_errors=True)
        gc.collect()