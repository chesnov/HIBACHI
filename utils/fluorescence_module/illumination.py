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


#: Pixels sampled per plane when measuring its level. A robust statistic does
#: not need the whole plane, and reading one costs a sort of it.
_Z_SAMPLE_PIXELS = 100_000

#: Fewest pixels -- finite, and then foreground -- a plane needs before it is
#: measured at all. A statistic of a handful of values is noise.
_Z_MIN_PIXELS = 100

#: Sampling is random, so it is seeded. The same stack must produce the same
#: correction on every run, or two runs of the pipeline are not comparable.
_Z_SAMPLE_SEED = 42

#: How far above the background the reported LEVEL must sit, in units of the
#: plane's own noise, before the plane counts as measured.
#:
#: A single pixel only has to clear `_SIGNAL_OVER_NOISE` sigma to be called
#: foreground. But if the only pixels clearing it are the background's own
#: upper tail, the median of that tail hugs the threshold -- so the LEVEL
#: sitting barely above the cut is the signature of a plane with no tissue
#: left. Requiring twice the per-pixel margin separates the two.
#:
#: This replaced a test on the SIZE of the foreground against a Gaussian tail
#: fraction, which does not work: a background clipped at zero, as a real
#: camera's is, has a much heavier upper tail than a Gaussian, so the test
#: never fired. Measured in noise units instead, which needs no assumption
#: about the shape of the background.
#:
#: This is a DETECTION THRESHOLD and it trades the two failures against each
#: other; there is no assumption-free value. Too low and a noise tail is taken
#: for tissue, which is the bug it exists to fix. Too high and a genuinely dim
#: plane is discarded and filled from the fit, so a real drop goes uncorrected.
#: Both observed, on the two cases that matter:
#:
#:     tissue at 1000 over background 218 +- 123   5.9 sigma   must be KEPT
#:     noise tail measuring 700, same background   3.9 sigma   must be DROPPED
#:
#: 5.0 sits between them. A plane closer to the floor than that cannot be
#: told apart from the tail by any statistic of the plane alone.
_Z_LEVEL_OVER_NOISE = 5.0

#: Anchor for the peak-matching factor. NOT the maximum: a saturated raw image
#: has a maximum of exactly the dtype ceiling, which says nothing about where
#: the signal actually tops out, and anchoring to it scales the whole stack by
#: an arbitrary amount. Measured on a real run, the max was 65535 and the
#: factor came out 0.88 -- a 12% darkening with no justification.
_Z_PEAK_PERCENTILE = 99.9

#: Fit the measured levels, or follow them plane by plane.
#:
#: Following them is exact wherever the measurement is trustworthy, and makes
#: no assumption about the shape of the profile -- a two-plane drop, a bright
#: edge plane, any number of peaks. But it cannot correct a plane whose tissue
#: has fallen below the noise floor, because such a plane contains no evidence
#: of how bright tissue would be at that depth: its measurement collapses onto
#: the background tail and reads several times too high.
#:
#: The fit exists for those planes. It is taken over the planes that ARE
#: measurable and extrapolated to the ones that are not, which is a narrower
#: job than the smoothing it used to do -- it is not asked to decide what is
#: content and what is attenuation, only to continue a trend past the point
#: where the signal ran out.
_Z_FIT = True


def z_levels(data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-plane statistics of a stack: (measured, used, peak), one per z.

    `correct_illumination` divides by `used` to bring every plane to a common
    brightness. `measured` is returned alongside it so the report records what
    was measured as well as what was applied, and so an unmeasurable plane is
    visible as such.

    `peak` is each plane's true maximum, over every pixel rather than the
    sample. The caller needs it to choose one global factor that puts the
    corrected maximum exactly at the raw maximum: that way the correction
    spends the range it is given, nothing saturates, and the corrected stack
    can be displayed on the same scale as the raw one and compared against it.
    It costs nothing to collect -- the plane is already read.

    WHAT IS MEASURED, and why it is not a percentile of the plane. A plane's
    brightness moves for two reasons: how brightly the tissue in it appears,
    which is what depth attenuation changes and what this should remove, and
    how MUCH tissue is in it, which is signal and must survive. A fixed
    percentile of the whole plane confuses the two -- it reports the
    background until tissue covers more than (100 - p)% of the plane and then
    jumps to the tissue level. Measured on synthetic planes holding tissue at
    ONE fixed brightness of 1000, with only the covered fraction changing:

        tissue fraction     p95 of plane      this function
             0.5%                 228                 981
             2.0%                 231                 997
             5.0%                 297                 998
            10.0%                1000                1000
            30.0%                1001                 999
            60.0%                1001            (declines)
            90.0%                1013                1101

    A statistic of the FOREGROUND pixels alone reports the same level whatever
    fraction of the plane the tissue covers, which is the property that makes
    the measurement usable directly. See KNOWN LIMITS for the dense end.

    WHY NO SMOOTHING, NO FIT, NO MODEL. Once the measurement is independent of
    content, every plane can be corrected from its own value. Nothing has to
    assume the profile decays, or decays smoothly, or has one peak: a level
    that falls over two planes, a bright first or last plane that is an
    acquisition artifact rather than tissue, a whole-mount with a long tail, a
    slice that is brightest in the middle -- all are followed as measured. An
    earlier version fitted a monotone exponential, which on a stack that
    brightens as it enters the tissue pinned the first thirty planes to one
    value and under-corrected the deepest by threefold. A smoother needs a
    lengthscale, and there is no lengthscale that is right for both a
    two-plane drop and a two-hundred-plane gradient.

    HOW FOREGROUND IS FOUND. The background's CENTRE and spread, both robust:
    the median of the plane and 1.4826 x its MAD, which are the background's
    own when background is the majority of the plane -- the ordinary case. A
    pixel is foreground when it clears that centre by `_SIGNAL_OVER_NOISE`
    sigma.

    NOT the module's block rule (`_BACKGROUND_PERCENTILE` + `_noise_sigma`).
    That rule compares a BLOCK's p95-minus-p10 against the noise, and reusing
    its pieces to threshold individual PIXELS puts the cut in the wrong place:
    the 10th percentile is the background's low tail, not its centre, so on a
    plane of N(200, 40) background the cut landed at 227 -- below a third of
    the background pixels. A third of the plane came back as "foreground" and
    its median was the background, 258 where the tissue was 1000. Measured,
    not reasoned: that was the first version of this function.

    If that strict cut finds too little, it is tried again with the noise
    measured from the quiet half only (`_noise_sigma`), which is lower and so
    more permissive. A plane that fails both is not measured.

    The statistic over the foreground is the MEDIAN, not a percentile -- half
    above and half below, so neither a few saturated pixels nor the exact
    placement of the threshold moves it much.

    A plane with no measurable tissue gets NaN in `measured` and inherits from
    its neighbours by linear interpolation in `used` (nearest value held at
    the ends). Nothing else is possible: a plane with nothing in it contains no
    evidence of how bright tissue would appear at that depth.

    Returns zeros in `used` when no plane can be measured at all, which the
    caller reads as "do not scale".

    KNOWN LIMITS, since none of this is free.

    DENSE PLANES. Once tissue is the majority of a plane, the median IS the
    tissue and the threshold rises above it, so there is no foreground left to
    take a statistic of. Around 60% coverage this function declines to measure
    and the plane inherits; by 90% it measures again but reads ~10% high,
    because what clears the threshold is the bright half of the tissue. A
    plane that is uniformly ONE thing is genuinely ambiguous from its own
    histogram -- all background and all tissue look alike -- and the only way
    to tell them apart is to compare against other planes, which this does not
    do. A volume where MOST planes are that dense will measure nothing and the
    correction will decline entirely; the report shows that as NaN throughout,
    rather than doing something quietly wrong.

    LOW CONTRAST. Tissue that does not clear the background by
    `_SIGNAL_OVER_NOISE` sigma is not foreground, so a plane whose signal is
    at the noise floor is not measured. That is the intended behaviour: there
    is nothing there to measure the brightness of.

    CIRCULARITY. The threshold that defines foreground is itself affected by
    illumination. It is computed per plane and relative to that plane, so it
    largely cancels, but not exactly.

    FEW PIXELS. A plane whose tissue is a few hundred pixels is measured from
    a few hundred pixels.
    """
    depth = int(data.shape[0])
    measured = np.full(depth, np.nan, dtype=np.float64)
    peak = np.zeros(depth, dtype=np.float64)
    rng = np.random.default_rng(_Z_SAMPLE_SEED)

    for z in range(depth):
        plane = np.asarray(data[z], dtype=np.float32).ravel()
        if plane.size == 0:
            continue
        sample = plane[np.isfinite(plane)]
        if sample.size:
            peak[z] = float(np.percentile(sample, _Z_PEAK_PERCENTILE))
        if sample.size < _Z_MIN_PIXELS:
            continue
        if sample.size > _Z_SAMPLE_PIXELS:
            sample = rng.choice(sample, _Z_SAMPLE_PIXELS, replace=False)

        centre = float(np.median(sample))
        spread = 1.4826 * float(np.median(np.abs(sample - centre)))
        foreground = sample[sample > centre + _SIGNAL_OVER_NOISE * spread]
        if foreground.size < _Z_MIN_PIXELS:
            # Relax to the quiet half's noise, which is the smaller estimate,
            # before giving up on the plane.
            quiet = _noise_sigma(sample - centre)
            foreground = sample[sample > centre + _SIGNAL_OVER_NOISE * quiet]
        if foreground.size < _Z_MIN_PIXELS:
            continue

        value = float(np.median(foreground))

        # Is that tissue, or the background's own upper tail? Past the depth
        # where tissue drops below the threshold, what clears it is noise, and
        # noise does not attenuate -- so the measurement stops falling and
        # levels off several times above the truth. Seen on a real stack:
        # background 217, noise sigma 123, and the deepest twenty planes all
        # measured 700-740 and then ticked back UP. A level that does not clear
        # the background by `_Z_LEVEL_OVER_NOISE` sigma is not tissue; the
        # plane is left unmeasured and the fit supplies its level.
        if spread > 0 and (value - centre) < _Z_LEVEL_OVER_NOISE * spread:
            continue
        if np.isfinite(value) and value > 0:
            measured[z] = value

    usable = np.isfinite(measured) & (measured > 0)
    if not np.any(usable):
        return measured, np.zeros(depth, dtype=np.float64), peak

    z_index = np.arange(depth, dtype=np.float64)
    used = np.interp(z_index, z_index[usable], measured[usable])

    if _Z_FIT and int(np.count_nonzero(usable)) >= 2:
        # Log-linear, because attenuation is Beer-Lambert -- exponential in
        # depth, a straight line in log, and monotone by construction.
        #
        # Theil-Sen, the median of the pairwise slopes, not least squares. One
        # plane that reads the noise tail is an enormous outlier in log space;
        # measured on a synthetic stack, a least-squares line was pulled 14-22%
        # off across the WHOLE depth by a single such plane, worst at the ends.
        # Theil-Sen ignores it. O(n^2) in the number of measurable planes,
        # which is a stack depth: tens of thousands of pairs at most.
        zs = z_index[usable]
        ys = np.log(measured[usable])
        i, j = np.triu_indices(zs.size, k=1)
        dz = zs[j] - zs[i]
        ok = dz != 0
        if np.any(ok):
            slope = float(np.median((ys[j][ok] - ys[i][ok]) / dz[ok]))
            intercept = float(np.median(ys - slope * zs))
            curve = np.exp(intercept + slope * z_index)
            # The fit REPLACES nothing that was measured. Where a plane was
            # measurable its own value is used -- that is the whole point of a
            # content-independent measurement, and a monotone curve cannot
            # represent the rise as a stack enters the tissue anyway. The curve
            # only fills the planes that had no measurable tissue, which the
            # interpolation above could otherwise only fill by holding the
            # nearest value flat.
            used = np.where(usable, measured, curve)

    return measured, np.asarray(used, dtype=np.float64), peak


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
        # `levels` is the per-plane tissue brightness, followed as measured.
        # It is safe to divide by directly because it is measured from each
        # plane's foreground only, so it does not move when a plane simply
        # holds less tissue. See z_levels.
        measured, levels, peak = z_levels(data)
        usable = levels[levels > 0]
        target = float(np.median(usable)) if usable.size else 0.0
        if target > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                scale_per_plane = np.where(levels > 0, target / levels, 1.0)
            scale_per_plane = np.asarray(scale_per_plane, dtype=np.float64)

            # Equalising to the median moves half the planes up and half down,
            # which is what makes a dim plane brighter and an over-bright one
            # dimmer. But it also moves the brightest pixel in the stack, and
            # the direction depends on which plane it happened to be in: the
            # corrected image would sit on a different scale from the raw one
            # and could not be compared with it by eye, and if it moved up it
            # would saturate against the dtype ceiling and lose the top of the
            # range outright.
            #
            # So one global factor afterwards, putting the corrected maximum
            # exactly where the raw maximum was. Global, so it changes no
            # RELATIVE brightness between planes -- the equalisation is
            # untouched -- and it is the largest factor that cannot clip,
            # because it is derived from the true per-pixel maxima rather than
            # from the levels.
            raw_peak = float(peak.max()) if peak.size else 0.0
            scaled_peak = float(np.max(peak * scale_per_plane)) if peak.size else 0.0
            if raw_peak > 0 and scaled_peak > 0:
                headroom = raw_peak / scaled_peak
                scale_per_plane = scale_per_plane * headroom
                report["z_peak_match"] = round(float(headroom), 4)
                report["z_raw_peak"] = round(raw_peak, 1)
            scale_per_plane = np.asarray(scale_per_plane, dtype=np.float32)
        # Both recorded: the measurement is the evidence, the curve is what was
        # applied, and a correction nobody can trace back to both is not
        # reproducible.
        report["z_levels_measured"] = [
            (None if not np.isfinite(v) else round(float(v), 3)) for v in measured
        ]
        # How many planes had no measurable tissue, so their level came from the
        # fit rather than from themselves. A high count on a stack you expected
        # to be bright throughout is the signal that the detection threshold,
        # not the correction, is what wants looking at.
        report["z_planes_unmeasured"] = int(np.sum(~np.isfinite(measured)))
        report["z_fit_used"] = bool(_Z_FIT)
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