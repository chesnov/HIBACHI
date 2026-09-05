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
    Excitation and collection efficiency vary across the field of view. The
    result is a smooth multiplicative gradient, so it is divided out. The field
    is estimated by heavily smoothing a DOWNSAMPLED copy: it is low-frequency
    by definition, so estimating it at a fraction of the resolution loses
    nothing and costs a fraction of the work -- which matters when a plane is
    928 megapixels.

Z (stacks only)
    Deeper planes are dimmer. Each plane is scaled to a common level, measured
    from its own bright pixels so that a plane containing less tissue is not
    mistaken for a dimmer one.

Order matters: Z first, then XY. The XY field is estimated from a projection
through the stack, and estimating it before Z correction would fold depth
attenuation into a field that is meant to describe the field of view.

Rank comes from the array, per convention 2 -- `ndim = int(volume.ndim)`,
never from a flag or a mode string. Z correction is simply absent at rank 2:
there is no depth to correct.

What this is NOT
----------------
Not a substitute for even illumination. Dividing by an estimated field
amplifies noise wherever the field is small, so a badly vignetted corner
becomes a noisy corner rather than a correct one. The correction exists so a
threshold can be set once for the whole field, and its output is written out
precisely so it can be looked at rather than trusted.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

#: Target element count for the downsampled copy the XY field is estimated
#: from. The field is smooth, so this is about how finely it must be sampled,
#: not about how much of the image is inspected.
_FIELD_SAMPLES = 4_000_000

#: Floor applied to the normalised field before dividing. A field value near
#: zero would turn its region into amplified noise; clipping trades an
#: uncorrected dark corner for a garbage one.
_MIN_FIELD = 0.05

#: Dynamic range above which the estimated field is not describing
#: illumination. Real illumination across a frame varies by a factor of two or
#: three; a field spanning far more than that is following the cells, because
#: the scale is comparable to them rather than to the frame. Dividing by such a
#: field flattens the structure the segmentation needs.
_FIELD_RANGE_WARN = 5.0


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


def _xy_field(reference, sigma_px: float) -> np.ndarray:
    """The smooth illumination field of a 2D reference image.

    Estimated on a strided copy and resized back. Normalised to its own median
    so dividing by it preserves the overall intensity level -- the correction
    should flatten the field, not rescale the image, or every downstream
    absolute threshold would shift.
    """
    from scipy.ndimage import gaussian_filter  # type: ignore
    from skimage.transform import resize  # type: ignore

    plane = np.asarray(reference, dtype=np.float32)
    step = max(1, int(np.ceil(np.sqrt(plane.size / _FIELD_SAMPLES))))
    small = plane[::step, ::step]
    field_small = gaussian_filter(small, max(1.0, sigma_px / step),
                                  mode="nearest")
    field = resize(field_small, plane.shape, order=1, mode="edge",
                   preserve_range=True, anti_aliasing=False)
    field = np.asarray(field, dtype=np.float32)
    middle = float(np.median(field))
    if middle <= 0:
        return np.ones_like(field)
    field /= middle
    np.clip(field, _MIN_FIELD, None, out=field)
    return field


def correct_illumination(
    volume,
    spacing: Sequence[float],
    xy_scale_um: float = 0.0,
    correct_z: bool = False,
    out=None,
    progress=None,
) -> Tuple[Optional[np.ndarray], dict]:
    """Write an illumination-corrected copy of `volume`. Returns (out, report).

    `xy_scale_um` is the length above which intensity variation is treated as
    illumination rather than structure: the field is smoothed with that sigma,
    so features smaller than it survive and gradients broader than it are
    removed. 0 disables the XY correction. It is a PHYSICAL length, converted
    through `spacing`, so the same value means the same thing at any pixel
    size.

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
    field = None
    if xy_scale_um and xy_scale_um > 0:
        sigma_px = float(xy_scale_um) / max(1e-6, pixel_um)
        if is_3d:
            accum = np.zeros(data.shape[-2:], dtype=np.float64)
            for z in range(depth):
                accum += np.asarray(data[z], dtype=np.float32) * scale_per_plane[z]
            reference = accum / max(1, depth)
        else:
            reference = np.asarray(data, dtype=np.float32)
        field = _xy_field(reference, sigma_px)
        report["xy_sigma_px"] = round(sigma_px, 2)
        report["xy_field_min"] = round(float(field.min()), 4)
        report["xy_field_max"] = round(float(field.max()), 4)

        # A field with a huge dynamic range is not an illumination field. Say
        # so: the correction still runs, because refusing silently would be
        # worse, but a scale this small removes structure rather than gradient
        # and the resulting image will look flat and washed out.
        span = float(field.max()) / max(1e-6, float(field.min()))
        report["xy_field_span"] = round(span, 1)
        if span > _FIELD_RANGE_WARN:
            suggested = max(4.0 * float(xy_scale_um), 20.0 * pixel_um)
            report["warning"] = (
                f"the estimated field spans {span:.0f}x, which is far more "
                f"than illumination varies across a frame -- a scale of "
                f"{xy_scale_um:g} um is {sigma_px:.1f} pixels here, comparable "
                f"to the cells, so the field is following them and the "
                f"correction is removing structure. Try {suggested:.0f} um or "
                f"more."
            )
            print(f"  [Illumination] WARNING: {report['warning']}")

    if field is None and not report["correct_z"]:
        report["applied"] = False
        return None, report
    report["applied"] = True

    if out is None:
        out = np.empty(data.shape, dtype=np.float32)

    for z in range(depth):
        plane = np.asarray(data[z] if is_3d else data, dtype=np.float32)
        if scale_per_plane[z] != 1.0:
            plane = plane * scale_per_plane[z]
        if field is not None:
            plane = plane / field
        if is_3d:
            out[z] = plane
        else:
            out[...] = plane
        if progress is not None:
            progress(z + 1, depth)
    if hasattr(out, "flush"):
        out.flush()
    return out, report
