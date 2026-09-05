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

#: Longest edge the background is estimated on. A rolling ball is expensive and
#: the background is smooth, so it is estimated on a downsampled copy and
#: resized back -- the approach skimage itself recommends for large images.
_BACKGROUND_MAX_EDGE = 1024


def _plane_level(plane) -> float:
    """A robust brightness level for one plane, from its bright pixels.

    Mean of the pixels above the plane's own median. A plane-wide mean would
    track how much tissue happens to sit in that plane rather than how brightly
    it is lit, so a sparse plane would read as a dim one and be scaled up.
    """
    values = np.asarray(plane, dtype=np.float32)
    if values.size == 0:
        return 0.0
    middle = float(np.median(values))
    bright = values > middle
    return float(values[bright].mean()) if bright.any() else float(values.mean())


def z_levels(volume) -> np.ndarray:
    """Per-plane brightness level of a stack, for judging depth attenuation."""
    return np.asarray([_plane_level(volume[z]) for z in range(volume.shape[0])],
                      dtype=np.float32)


def _xy_background(reference, radius_px: float) -> np.ndarray:
    """The background surface of a 2D image, by rolling ball.

    SUBTRACTIVE, and that is the point. Dividing by a smoothed local mean
    assumes the field of view is filled: wherever an image is mostly empty the
    local mean approaches zero, and dividing by it turns background into
    amplified noise -- local contrast normalisation rather than illumination
    correction. A rolling ball makes no assumption about how much of the frame
    is occupied. It rolls a ball of the given radius beneath the intensity
    surface and takes the highest surface it can reach as the background, so an
    empty region yields its own low background and a crowded one yields a
    higher one, with nothing ever divided.

    Estimated on a downsampled copy and resized back, because the ball is
    expensive and a background broad enough to be illumination survives
    downsampling intact.
    """
    from skimage.restoration import rolling_ball  # type: ignore
    from skimage.transform import resize  # type: ignore

    plane = np.asarray(reference, dtype=np.float32)
    longest = max(plane.shape)
    factor = max(1, int(np.ceil(longest / _BACKGROUND_MAX_EDGE)))
    small = plane[::factor, ::factor] if factor > 1 else plane
    background_small = rolling_ball(small, radius=max(1.0, radius_px / factor))
    if factor > 1:
        background = resize(background_small, plane.shape, order=1, mode="edge",
                            preserve_range=True, anti_aliasing=False)
    else:
        background = background_small
    return np.asarray(background, dtype=np.float32)


def correct_illumination(
    volume,
    spacing: Sequence[float],
    xy_scale_um: float = 0.0,
    correct_z: bool = False,
    out=None,
    progress=None,
) -> Tuple[Optional[np.ndarray], dict]:
    """Write an illumination-corrected copy of `volume`. Returns (out, report).

    `xy_scale_um` is the rolling ball's radius: structures narrower than it are
    kept, and background broader than it is subtracted. It must be comfortably
    larger than anything being measured, or the ball rolls over the objects
    themselves and subtracts them. 0 disables the XY correction. A PHYSICAL
    length, converted through `spacing`, so the same value means the same thing
    at any pixel size.

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
        "xy_scale_um": float(xy_scale_um),
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
    background = None
    if xy_scale_um and xy_scale_um > 0:
        radius_px = float(xy_scale_um) / max(1e-6, pixel_um)
        if is_3d:
            accum = np.zeros(data.shape[-2:], dtype=np.float64)
            for z in range(depth):
                accum += np.asarray(data[z], dtype=np.float32) * scale_per_plane[z]
            reference = accum / max(1, depth)
        else:
            reference = np.asarray(data, dtype=np.float32)
        background = _xy_background(reference, radius_px)
        report["xy_radius_px"] = round(radius_px, 2)
        report["xy_background_min"] = round(float(background.min()), 2)
        report["xy_background_max"] = round(float(background.max()), 2)

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
