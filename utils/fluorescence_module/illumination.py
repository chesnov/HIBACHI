"""illumination: flatten uneven illumination before segmentation.

Why
---
Nothing in the pipeline compensated for illumination that varies across the
field: one side brighter than the other, darker corners, or -- in a stack --
planes dimming with depth as excitation and emission are absorbed. A single
threshold cannot serve a field like that. It is met in the bright region and
missed in the dim one, so cells vanish from one corner of every image and the
loss looks like biology.

Two corrections, because they are two different physical effects:

XY (both ranks)
    A background surface is estimated with a rolling ball and SUBTRACTED.
    Subtraction rather than division, deliberately: dividing by a smoothed
    local mean assumes the frame is filled, and wherever an image is mostly
    empty that mean approaches zero, so dividing turns background into
    amplified noise -- local contrast normalisation, not illumination
    correction. A rolling ball assumes nothing about how much of the frame is
    occupied. It is estimated on a downsampled copy, because the ball is
    expensive and a background broad enough to be illumination survives
    downsampling intact.

Z (stacks only)
    Deeper planes are dimmer. Each plane is scaled to a common level, measured
    from its own bright pixels so that a plane containing less tissue is not
    mistaken for a dimmer one.

Order matters: Z first, then XY. The XY background is estimated from a
projection through the stack, and estimating it before Z correction would fold
depth attenuation into a surface meant to describe the field of view.

Rank comes from the array, per convention 2 -- `ndim = int(volume.ndim)`,
never from a flag or a mode string. Z correction is simply absent at rank 2:
there is no depth to correct.

What this is NOT
----------------
Not a substitute for even illumination. Subtracting a background cannot
recover signal that was never collected, so a severely vignetted corner ends
up dark and clean rather than correct. The correction exists so a threshold can
be set once for the whole frame, and its output is written out precisely so it
can be looked at rather than trusted.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import numpy as np

#: Smallest block, in pixels, whatever physical size is asked for: percentiles
#: within a handful of pixels are noise.
_MIN_BLOCK_PX = 8

#: Percentile taken within a block as its signal level, and the one taken over
#: the WHOLE image as the floor everything is measured above.
_ENVELOPE_PERCENTILE = 95.0
_BACKGROUND_PERCENTILE = 10.0

#: Multiple of the measured noise a block's signal must clear to count as
#: holding anything.
_SIGNAL_OVER_NOISE = 3.0

#: Fraction of blocks that must hold signal before an envelope is trusted. Too
#: few and there is nothing to interpolate between, so the correction declines.
_ENVELOPE_MIN_COVERAGE = 0.15


def _noise_sigma(residual) -> float:
    """Noise level of a background-subtracted image, robustly.

    From the median absolute deviation of the quiet half, so cells cannot
    inflate it. This replaces deriving a floor from the block signals
    themselves, which failed badly: with blocks large enough to contain both
    cells and background, every block's signal looked large, the floor rose
    with it, and almost no block cleared it -- 0.7% of a frame that was full of
    cells.
    """
    values = np.asarray(residual, dtype=np.float32).ravel()
    if values.size == 0:
        return 1.0
    quiet = values[values <= np.median(values)]
    if quiet.size == 0:
        quiet = values
    mad = float(np.median(np.abs(quiet - np.median(quiet))))
    return max(1e-6, 1.4826 * mad)


def _block_grid(shape, block) -> Tuple[Tuple[int, ...], Tuple[slice, ...]]:
    """Number of blocks per axis, and a helper to slice block `idx` out."""
    return tuple(max(1, int(shape[k]) // int(block[k]))
                 for k in range(len(shape)))


def _block_slices(shape, block, grid, idx) -> Tuple[slice, ...]:
    """Slices of block `idx`. The last block on each axis takes the remainder."""
    out = []
    for k in range(len(shape)):
        start = idx[k] * int(block[k])
        stop = int(shape[k]) if idx[k] == grid[k] - 1 else (idx[k] + 1) * int(block[k])
        out.append(slice(start, stop))
    return tuple(out)


def _sample_grid(values, shape, region: Optional[Tuple[slice, ...]] = None):
    """Linear upsampling of a per-block array to full resolution.

    Replaces `skimage.transform.resize(..., order=1, mode="edge")` and matches
    its coordinate convention -- output centres map to
    `(i + 0.5) * in/out - 0.5`, clamped at the edges -- but samples only a
    REGION of the output, so the correction can be applied to the volume a slab
    at a time without materialising a surface the size of the whole image.

    `region` is a tuple of slices into the full output. Sampling a region gives
    exactly the values the full-resolution surface would have there, so slabs
    carry no seams: every output voxel's coordinate is computed from its GLOBAL
    index.
    """
    from scipy.ndimage import map_coordinates  # type: ignore

    values = np.asarray(values, dtype=np.float32)
    nd = len(shape)
    if region is None:
        region = tuple(slice(0, int(shape[k])) for k in range(nd))

    coords = []
    for k in range(nd):
        out_n = int(shape[k])
        in_n = int(values.shape[k])
        idx = np.arange(region[k].start, region[k].stop, dtype=np.float64)
        if out_n == in_n:
            c = idx
        else:
            c = (idx + 0.5) * (in_n / out_n) - 0.5
        np.clip(c, 0.0, in_n - 1.0, out=c)
        coords.append(c)

    mesh = np.meshgrid(*coords, indexing="ij")
    sampled = map_coordinates(values, np.asarray(mesh), order=1, mode="nearest")
    return np.asarray(sampled, dtype=np.float32)


def _block_surfaces(data, block, max_gain: float, report: dict):
    """(background_grid, gain_grid) from block percentiles, or (grid, None).

    RANK-AGNOSTIC. The image is tiled into blocks along EVERY axis and the same
    two percentiles are taken in each: a low one is that block's background and
    a high one its signal level. In 2D that is a grid of squares over the
    image; in 3D a grid of boxes over the volume, so depth attenuation is
    corrected by the same code that corrects in-plane shading, as one more
    direction in which the illumination varies. There is no branch on the
    number of axes anywhere in this function, and no separate stage for depth.
    
    It used to tile only the last two axes and average the volume down to one
    plane first, which is what made depth a special case needing its own
    per-plane scaling stage.

    `block` sets everything, one size per axis. A block must be large enough to
    contain background but small enough that illumination is roughly constant
    across it. Too large and its "background" percentile sits inside tissue: at
    1477 px the estimated background spanned 79x, which is a picture of the
    cells, not of the illumination.

    A block whose signal does not clear the measured noise is DISCARDED, not
    recorded as zero -- counting empty blocks is what makes a divided
    correction explode on a sparse image. Those blocks inherit a neighbour's
    factor, so a region with nothing in it is left alone.

    Returns grids, not full-resolution surfaces: the caller upsamples them a
    slab at a time through `_sample_grid`. A surface the size of a 400-megavoxel
    volume is 1.6 GB per surface, and there are two.

    Returns None for the gain when too little of the image holds signal to
    define one: decline rather than guess.
    """
    import itertools

    from scipy.ndimage import (  # type: ignore
        distance_transform_edt, gaussian_filter,
    )

    shape = tuple(int(v) for v in data.shape)
    nd = len(shape)
    block = tuple(max(_MIN_BLOCK_PX, int(b)) for b in block)
    grid = _block_grid(shape, block)

    low = np.zeros(grid, dtype=np.float32)
    high = np.zeros(grid, dtype=np.float32)
    # Voxel noise WITHIN each block, so the "does this block hold anything"
    # test below compares a block's signal against a real noise level.
    spread = np.zeros(grid, dtype=np.float32)

    # Walked one slab of blocks at a time along axis 0, so only that slab is
    # resident: the percentiles need the voxels, and the whole image as float32
    # is what this is avoiding.
    for i0 in range(grid[0]):
        sl0 = _block_slices(shape, block, grid, (i0,) + (0,) * (nd - 1))[0]
        slab = np.asarray(data[sl0], dtype=np.float32)
        for rest in itertools.product(*[range(grid[k]) for k in range(1, nd)]):
            idx = (i0,) + rest
            sub = _block_slices(shape, block, grid, idx)
            tile = slab[(slice(None),) + sub[1:]]
            if tile.size:
                low[idx] = float(np.percentile(tile, _BACKGROUND_PERCENTILE))
                high[idx] = float(np.percentile(tile, _ENVELOPE_PERCENTILE))
                centre = float(np.median(tile))
                spread[idx] = 1.4826 * float(np.median(np.abs(tile - centre)))
        del slab

    report["blocks"] = [int(g) for g in grid]
    report["block_px"] = [int(b) for b in block]

    # ---- THE FLOOR IS GLOBAL, NOT PER BLOCK -----------------------------
    # The background used to be each block's own low percentile, smoothed. That
    # is right when every block contains some empty field -- sparse cells on a
    # dark background, which is what a 2D frame of this data looks like -- and
    # wrong when a block sits entirely INSIDE the specimen, which is what
    # happens through the middle of a solid object in 3D. Measured on blocks
    # wholly within a squashed sphere of tissue about 2000 bright:
    #
    #     block p10 = 867    block p95 = 1316    p95 - p10 = 449
    #
    # The low percentile IS the tissue there, so subtracting it removed most of
    # the tissue (1598 -> 737 on a real test) and the "signal" driving the gain
    # was the tissue's internal texture rather than its brightness. Depth
    # attenuation was invisible to it: a 2.35x top-to-bottom ratio came out at
    # 2.36.
    #
    # Nor can a bigger block fix it. For a block's low percentile to be
    # background it must be larger than the specimen; for the gain to resolve
    # depth it must be smaller than the attenuation lengthscale. On a specimen
    # filling most of the field those cannot both hold -- at 40 um the grid was
    # [1, 1, 1].
    #
    # So the floor is one number for the whole image: the low percentile of the
    # block lows, which is the darkest part of the field and therefore the
    # camera offset plus stray light. Blocks measure their brightness ABOVE it,
    # which inside solid tissue is the tissue's brightness -- the thing
    # attenuation changes.
    #
    # What this gives up is real: local background variation, patchy
    # nonspecific staining for instance, is no longer removed. That was a
    # genuine benefit of the per-block floor on 2D data, and it is the price of
    # a correction that works through a solid object.
    floor = float(np.percentile(low, _BACKGROUND_PERCENTILE)) if low.size else 0.0
    background = np.full(grid, floor, dtype=np.float32)
    report["background_floor"] = round(floor, 2)
    report["background_min"] = round(floor, 2)
    report["background_max"] = round(floor, 2)

    if max_gain <= 1.0:
        report["gain_skipped"] = "maximum gain is 1, so only the background was removed"
        return background, None

    # Noise is the MEDIAN of the within-block MADs.
    #
    # NOT `_noise_sigma(high - low)`, which is what this was and which measures
    # the wrong quantity: (high - low) is each block's SPREAD, so the MAD of
    # those is a spread of spreads. On blocks of pure background with a voxel
    # noise of 50 it returned 0.70, making the test below `signal > 2.1` while
    # an empty block's own signal is about 2.93 sigma = 146. Every block passed:
    # a real run reported `signal_coverage: 1.0` on a volume that is mostly
    # empty, the gain was fitted to empty background, and `gain_max` pinned
    # itself to the cap -- so background noise was amplified tenfold. That is
    # the correction going wrong exactly where there is nothing to correct.
    #
    # A within-block MAD is the voxel noise of that block, and the median over
    # blocks is robust to the ones full of tissue, so this recovers 50.00 on
    # the same test. It costs nothing: the tile is already in hand.
    noise = float(np.median(spread[spread > 0])) if np.any(spread > 0) else 1.0
    noise = max(noise, 1e-6)
    # Brightness above the global floor, not contrast within the block.
    signal = np.clip(high - floor, 0.0, None)
    has_signal = signal > (_SIGNAL_OVER_NOISE * noise)
    coverage = float(has_signal.mean()) if has_signal.size else 0.0
    report["noise_sigma"] = round(noise, 2)
    report["signal_coverage"] = round(coverage, 3)

    if has_signal.any():
        present = signal[has_signal]
        # 90th over 10th percentile, not max over min: the sparsest block that
        # still clears the noise sets the minimum, so max/min reported a 155x
        # spread on a frame varying 2.5x and advised a cap of 155.
        spread = float(np.percentile(present, 90)
                       / max(1e-6, np.percentile(present, 10)))
        # Reported, NOT turned into advice. A block's signal level tracks how
        # much is in it as well as how brightly it is lit: a block holding two
        # cells has a low high-percentile because it is sparse, not because it
        # is dim. So this number is an upper bound on the illumination
        # variation, badly inflated wherever density varies -- on a frame with
        # a 4x brightness ramp it read 46x.
        report["signal_spread_upper_bound"] = round(spread, 2)

    if coverage < _ENVELOPE_MIN_COVERAGE:
        report["gain_declined"] = (
            f"only {coverage * 100:.0f}% of the image holds signal above the "
            f"noise ({noise:.1f}), too little to tell uneven illumination from "
            f"empty space -- the foreground was left unscaled"
        )
        print(f"  [Illumination] {report['gain_declined']}")
        return background, None

    filled = signal.copy()
    if not has_signal.all():
        indices = distance_transform_edt(
            ~has_signal, return_distances=False, return_indices=True)
        filled = signal[tuple(indices)]
    filled = gaussian_filter(filled, 1.0, mode="nearest")

    middle = float(np.median(filled[has_signal]))
    if middle <= 0:
        return background, None
    gain = filled / middle
    np.clip(gain, 1.0 / float(max_gain), float(max_gain), out=gain)
    report["gain_min"] = round(float(gain.min()), 3)
    report["gain_max"] = round(float(gain.max()), 3)
    return background, np.asarray(gain, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Reading back a corrected image written by an earlier step
# --------------------------------------------------------------------------- #
#: Rows examined at a time when checking a corrected image is not blank. A
#: plane here can be hundreds of megapixels, so the check streams instead of
#: materialising a whole-array comparison.
_CHECK_ROWS = 64


def _infer_corrected_dtype(path: str, shape: Sequence[int],
                           preferred=None) -> np.dtype:
    """The dtype a corrected-image file is actually stored in.

    Derived from the file SIZE rather than assumed, for two reasons. The
    artifact is written in the INPUT's dtype (see `correct_illumination`), so
    there is no single right answer to hardcode; and an older build, or the
    stale comment in `ARTIFACT_PATTERNS` that still calls this artifact
    float32, means a project on disk may disagree with what this build would
    write. Reading a uint16 file as float32 does not fail -- it silently
    produces an array of the wrong shape's worth of garbage -- so the size is
    checked rather than trusted.

    Raises if the size matches no plausible dtype, which also catches a
    truncated or partly-written file.
    """
    voxels = 1
    for dim in shape:
        voxels *= int(dim)
    size = os.path.getsize(path)
    if voxels <= 0:
        raise ValueError("image shape is empty")
    if size % voxels:
        raise ValueError(
            f"the corrected image is {size} bytes, which is not a whole "
            f"number of values for a {tuple(shape)} image. It may be "
            "truncated or left over from a different image; re-run step 1."
        )
    itemsize = size // voxels

    # The dtype this build would have written comes first, so the common case
    # needs no guessing at all.
    candidates = []
    if preferred is not None:
        candidates.append(np.dtype(preferred))
    candidates += [np.dtype(t) for t in
                   (np.uint16, np.float32, np.uint8, np.uint32, np.float64,
                    np.int16, np.int32)]
    for dtype in candidates:
        if dtype.itemsize == itemsize:
            return dtype
    raise ValueError(
        f"the corrected image stores {itemsize} bytes per value, which "
        "matches no expected pixel type. Re-run step 1."
    )


def _looks_written(array) -> bool:
    """True if an array holds any nonzero value.

    The guard this exists for: the step that writes the corrected image
    creates its memmap with ``mode="w+"`` BEFORE calling
    `correct_illumination`, and that function returns early without writing
    when it decides no correction applies. So the artifact can exist, be
    exactly the right size, and be entirely zeros -- an image that would pass
    every structural check and then silently flatten every intensity gate
    downstream to no contrast at all.

    Streams with an early exit, so the usual case stops at the first block and
    only a genuinely blank file is read in full -- which is the case that must
    not be got wrong.
    """
    rows = int(array.shape[0])
    for start in range(0, rows, _CHECK_ROWS):
        block = np.asarray(array[start:min(start + _CHECK_ROWS, rows)])
        if block.any():
            return True
    return False


def open_corrected(path: Optional[str], shape: Sequence[int],
                   preferred_dtype=None):
    """Memmap a corrected image written by an earlier step, or raise why not.

    Raises rather than returning None so the caller cannot quietly fall back to
    the raw image. A run configured to measure intensity on the corrected image
    and silently measuring it on the raw one would look like a successful run
    of the analysis that was asked for, and be a different one -- the same
    reasoning as `soma_source.resolve`.

    Read-only, and at the image's own shape, so this never rewrites or
    reinterprets the artifact it was handed.
    """
    if not path:
        raise FileNotFoundError(
            "this run measures intensity on the illumination-corrected image, "
            "but this build has no such artifact for this segment."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(
            "this run measures intensity on the illumination-corrected image, "
            "but step 1 did not write one. Turn on illumination correction "
            "there (a block size above zero, or Z correction) and re-run it, "
            "or untick this."
        )

    dtype = _infer_corrected_dtype(path, shape, preferred_dtype)
    array = np.memmap(path, dtype=dtype, mode="r", shape=tuple(shape))
    if not _looks_written(array):
        del array
        raise ValueError(
            "the illumination-corrected image for this segment is entirely "
            "zero, which means step 1 created the file but decided no "
            "correction applied. Check step 1's illumination settings and "
            "re-run it, or untick this to measure on the raw image."
        )
    return array


def correct_illumination(
    volume,
    spacing: Sequence[float],
    block_um: float = 0.0,
    max_gain: float = 1.0,
    out=None,
    progress=None,
) -> Tuple[Optional[np.ndarray], dict]:
    """Even out illumination across an image. ONE function, ANY number of axes.

    Tiles the image into blocks of a fixed PHYSICAL size along every axis,
    takes a low percentile in each block as its background and a high one as
    its signal level, then subtracts the background and divides by the bounded
    signal level. Nothing here asks how many axes the input has.

    That is the point of this version. It used to tile only the last two axes,
    average a stack down to a single plane in order to do so, and then carry a
    SEPARATE per-plane scaling stage to deal with depth -- a stage with no 2D
    counterpart, its own level estimator, its own fit, its own noise-floor
    test and its own failure modes. Depth is not a special direction: a plane
    dim because the excitation was absorbed on the way in is dim for the same
    reason a corner of a field of view is dim, and one grid of blocks over the
    whole image corrects both. At rank 2 the grid is squares over an image; at
    rank 3 it is boxes over a volume; the code is the same code.

    `block_um` is the block edge in microns, converted per axis with that
    axis's own spacing -- the same physical size in every direction, so on
    anisotropic data it is very different numbers of voxels: at 0.276 um pixels
    and a 1 um z step, 60 um is 217 pixels in plane and 60 planes deep. A block
    must hold background but be small enough that illumination is roughly even
    across it. 0 disables the correction.

    `max_gain` caps how much the foreground may be evened out; 1 means
    background subtraction only.

    `out` is an array to write into -- normally a memmap over the artifact
    file, so the corrected image is persistent and can be reopened. Written a
    slab at a time, and the surfaces are kept at block resolution and upsampled
    per slab rather than materialised at full size, so peak memory is a slab
    whatever the image size.

    The report records what was measured and applied, for the run's provenance:
    a corrected image nobody can trace back to a correction factor is not
    reproducible.
    """
    data = volume
    shape = tuple(int(v) for v in data.shape)
    nd = len(shape)

    spacing_arr = np.asarray(spacing, dtype=np.float64)
    if spacing_arr.size < nd:
        raise ValueError("spacing must have one entry per axis")
    spacing_arr = spacing_arr[-nd:]
    if not np.all(spacing_arr > 0):
        raise ValueError("spacing must be positive on every axis")

    report: dict = {
        "ndim": nd,
        "block_um": float(block_um),
        "max_gain": float(max_gain),
        "spacing": [round(float(v), 6) for v in spacing_arr],
        "dtype": str(np.dtype(getattr(volume, "dtype", np.float32))),
    }

    if not (block_um and block_um > 0):
        report["applied"] = False
        report["skipped"] = "block size is 0, so no correction was requested"
        return None, report

    # One physical size, converted per axis. This is the only place the axes
    # differ from one another, and they differ by how finely they are sampled,
    # not by being depth rather than width.
    block = tuple(max(_MIN_BLOCK_PX,
                      min(int(shape[k]),
                          int(round(float(block_um) / float(spacing_arr[k])))))
                  for k in range(nd))

    background_grid, gain_grid = _block_surfaces(
        data, block, float(max_gain), report)
    if background_grid is None:
        report["applied"] = False
        return None, report
    report["applied"] = True

    dtype = np.dtype(getattr(data, "dtype", np.float32))
    if out is None:
        out = np.empty(shape, dtype=dtype)

    # Written in the INPUT's dtype. The pipeline expresses an absolute
    # threshold as a fraction of the dtype range -- "scaling by DType Max" --
    # so handing it float32 silently redefines every absolute threshold: 0.055
    # of 65535 is not 0.055 of 1.0. Keeping the dtype also halves what the
    # image costs to store.
    is_integer = np.issubdtype(dtype, np.integer)
    info = np.iinfo(dtype) if is_integer else None

    # Applied in slabs along axis 0, which is rank-agnostic: at rank 3 a slab
    # is a group of planes, at rank 2 a group of rows, and the arithmetic is
    # identical. The surfaces are sampled at the slab's GLOBAL coordinates, so
    # slabs carry no seams.
    step = max(1, min(shape[0], int(block[0])))
    for start_i in range(0, shape[0], step):
        stop_i = min(shape[0], start_i + step)
        region = (slice(start_i, stop_i),) + tuple(
            slice(0, shape[k]) for k in range(1, nd))
        chunk = np.asarray(data[start_i:stop_i], dtype=np.float32)

        # Subtract, never divide, and clamp at zero: a background estimate
        # above the signal means an empty region, not a negative one.
        chunk -= _sample_grid(background_grid, shape, region)
        np.clip(chunk, 0.0, None, out=chunk)

        if gain_grid is not None:
            # Divide by the local SIGNAL level, bounded, so one threshold is
            # reachable across the image. This assumes the true signal is
            # even -- from a single image, dim-because-unlit and
            # dim-because-less-antigen are indistinguishable -- which is why
            # the result is a segmentation input only and every measurement is
            # taken from the original image.
            chunk /= _sample_grid(gain_grid, shape, region)

        if is_integer:
            np.clip(chunk, float(info.min), float(info.max), out=chunk)
            np.rint(chunk, out=chunk)
        out[start_i:stop_i] = chunk.astype(dtype, copy=False)
        if progress is not None:
            progress(stop_i, shape[0])
    if hasattr(out, "flush"):
        out.flush()
    return out, report