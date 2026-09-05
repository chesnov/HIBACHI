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

from typing import Optional, Sequence, Tuple

import numpy as np

#: Smallest block, in pixels, whatever physical size is asked for: percentiles
#: within a handful of pixels are noise.
_MIN_BLOCK_PX = 8

#: Percentile taken within a block as its local signal level, and the one
#: taken as its background.
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


def _block_surfaces(reference, block_px: int, max_gain: float, report: dict):
    """(background, gain) surfaces from block percentiles, or (background, None).

    `block_px` sets everything. A LOW percentile within a block is its
    background and a HIGH percentile its signal level, so the block must be
    large enough to contain background but small enough that illumination is
    roughly constant across it. Too large and its "background" percentile sits
    inside tissue: at 1477 px the estimated background spanned 79x, which is a
    picture of the cells, not of the illumination.

    A block whose signal does not clear the measured noise is DISCARDED, not
    recorded as zero -- counting empty blocks is what makes a divided
    correction explode on a sparse image. Those blocks inherit a neighbour's
    factor, so a region with nothing in it is left alone.

    Returns None for the gain when too little of the frame holds signal to
    define one: decline rather than guess.
    """
    from scipy.ndimage import (  # type: ignore
        distance_transform_edt, gaussian_filter,
    )
    from skimage.transform import resize  # type: ignore

    plane = np.asarray(reference, dtype=np.float32)
    height, width = plane.shape
    block = max(_MIN_BLOCK_PX, int(block_px))
    rows = max(1, height // block)
    cols = max(1, width // block)

    low = np.zeros((rows, cols), dtype=np.float32)
    high = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            r1 = height if r == rows - 1 else (r + 1) * block
            c1 = width if c == cols - 1 else (c + 1) * block
            tile = plane[r * block:r1, c * block:c1]
            if tile.size:
                low[r, c] = float(np.percentile(tile, _BACKGROUND_PERCENTILE))
                high[r, c] = float(np.percentile(tile, _ENVELOPE_PERCENTILE))

    report["blocks"] = [int(rows), int(cols)]
    report["block_px"] = int(block)

    background = resize(gaussian_filter(low, 1.0, mode="nearest"), plane.shape,
                        order=1, mode="edge", preserve_range=True,
                        anti_aliasing=False)
    background = np.asarray(background, dtype=np.float32)
    report["background_min"] = round(float(background.min()), 2)
    report["background_max"] = round(float(background.max()), 2)

    if max_gain <= 1.0:
        report["gain_skipped"] = "maximum gain is 1, so only the background was removed"
        return background, None

    noise = _noise_sigma(plane - background)
    signal = np.clip(high - low, 0.0, None)
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
        # a 4x brightness ramp it read 46x. It is worth seeing, because a small
        # value means density is even and the estimate is trustworthy, but it
        # must not be used to pick the cap.
        report["signal_spread_upper_bound"] = round(spread, 2)

    if coverage < _ENVELOPE_MIN_COVERAGE:
        report["gain_declined"] = (
            f"only {coverage * 100:.0f}% of the frame holds signal above the "
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

    surface = resize(gain, plane.shape, order=1, mode="edge",
                     preserve_range=True, anti_aliasing=False)
    return background, np.asarray(surface, dtype=np.float32)


def correct_illumination(
    volume,
    spacing: Sequence[float],
    block_um: float = 0.0,
    max_gain: float = 1.0,
    correct_z: bool = False,
    out=None,
    progress=None,
) -> Tuple[Optional[np.ndarray], dict]:
    """Write an illumination-corrected copy of `volume`. Returns (out, report).

    `block_um` is the size of the block both surfaces are measured in, in
    MICRONS, and 0 disables the whole XY correction. It has to be large enough
    that a block contains background as well as objects, and small enough that
    illumination is roughly constant across it.

    `max_gain` caps how much the foreground may be evened out. 1 means
    background subtraction only. Whatever the estimate says, no region is
    scaled by more than this, so a misjudged surface cannot amplify a corner
    into noise -- and the report says how far the signal actually varies, so
    the cap can be set from the image rather than guessed.

    `correct_z` scales each plane of a stack to a common level. Ignored for a
    2D image, which has no depth.

    `out` is an array to write into -- normally a memmap over the artifact
    file, so the corrected image is persistent and can be reopened. Written
    plane by plane, so peak memory is one plane whatever the image size.

    The report records what was measured and applied, for the run's provenance:
    a corrected image nobody can trace back to a correction factor is not
    reproducible.
    """
    data = volume
    ndim = int(np.asarray(data.shape).size)
    if ndim not in (2, 3):
        raise ValueError(f"illumination correction needs a 2D or 3D image, got {ndim}D")

    spacing_arr = np.asarray(spacing, dtype=np.float64)
    if spacing_arr.size < ndim:
        raise ValueError("spacing must have one entry per axis")
    # In-plane spacing is the LAST TWO entries at either rank -- spacing[-2:],
    # never spacing[1:], which silently drops Y in 2D (convention 4).
    in_plane = spacing_arr[-2:]
    pixel_um = float(np.mean(in_plane)) if np.all(in_plane > 0) else 1.0

    report: dict = {
        "ndim": ndim,
        "block_um": float(block_um),
        "max_gain": float(max_gain),
        "correct_z": bool(correct_z and ndim == 3),
        "pixel_um": pixel_um,
        "dtype": str(np.dtype(getattr(volume, "dtype", np.float32))),
    }

    is_3d = ndim == 3
    depth = int(data.shape[0]) if is_3d else 1

    # ---- Z first: the XY field is estimated from a projection, and doing it
    # the other way round folds depth attenuation into the field of view.
    scale_per_plane = np.ones(depth, dtype=np.float32)
    if is_3d and correct_z:
        levels = z_levels(data)
        usable = levels[levels > 0]
        target = float(np.median(usable)) if usable.size else 0.0
        if target > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                scale_per_plane = np.where(levels > 0, target / levels, 1.0)
            scale_per_plane = np.asarray(scale_per_plane, dtype=np.float32)
        report["z_levels"] = [round(float(v), 3) for v in levels]
        report["z_scales"] = [round(float(v), 4) for v in scale_per_plane]

    # ---- XY field, from a z-corrected mean projection so one field serves
    # every plane. The field of view does not change with depth; measuring it
    # per plane would only add noise.
    # ---- XY background, from a z-corrected mean projection so one surface
    # serves every plane. The field of view does not change with depth;
    # estimating it per plane would only add noise.
    # ---- Background and gain, from a z-corrected mean projection so one pair
    # of surfaces serves every plane. The field of view does not change with
    # depth; estimating it per plane would only add noise.
    background = None
    gain_surface = None
    if block_um and block_um > 0:
        if is_3d:
            accum = np.zeros(data.shape[-2:], dtype=np.float64)
            for z in range(depth):
                accum += np.asarray(data[z], dtype=np.float32) * scale_per_plane[z]
            reference = accum / max(1, depth)
        else:
            reference = np.asarray(data, dtype=np.float32)
        block_px = max(_MIN_BLOCK_PX,
                       int(round(float(block_um) / max(1e-6, pixel_um))))
        background, gain_surface = _block_surfaces(
            reference, block_px, float(max_gain), report)

    if background is None and not report["correct_z"]:
        report["applied"] = False
        return None, report
    report["applied"] = True

    dtype = np.dtype(getattr(data, "dtype", np.float32))
    if out is None:
        out = np.empty(data.shape, dtype=dtype)

    # Written in the INPUT's dtype. The pipeline expresses an absolute
    # threshold as a fraction of the dtype range -- "scaling by DType Max" --
    # so handing it float32 silently redefines every absolute threshold: 0.055
    # of 65535 is not 0.055 of 1.0. Keeping the dtype also halves what the
    # image costs to store.
    is_integer = np.issubdtype(dtype, np.integer)
    info = np.iinfo(dtype) if is_integer else None

    for z in range(depth):
        plane = np.asarray(data[z] if is_3d else data, dtype=np.float32)
        if scale_per_plane[z] != 1.0:
            plane = plane * scale_per_plane[z]
        if background is not None:
            # Subtract, never divide, and clamp at zero: a background estimate
            # above the signal means an empty region, not a negative one.
            plane = plane - background
            np.clip(plane, 0.0, None, out=plane)
        if gain_surface is not None:
            # Divide by the local SIGNAL level, bounded, so one threshold is
            # reachable across the frame. This assumes the true signal is
            # even -- from a single image, dim-because-unlit and
            # dim-because-less-antigen are indistinguishable -- which is why
            # the result is a segmentation input only and every measurement is
            # taken from the original image.
            plane = plane / gain_surface
        if is_integer:
            np.clip(plane, float(info.min), float(info.max), out=plane)
            plane = np.rint(plane)
        if is_3d:
            out[z] = plane.astype(dtype, copy=False)
        else:
            out[...] = plane.astype(dtype, copy=False)
        if progress is not None:
            progress(z + 1, depth)
    if hasattr(out, "flush"):
        out.flush()
    return out, report
