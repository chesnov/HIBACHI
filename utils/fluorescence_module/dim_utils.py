"""
Dimension-agnostic primitives shared by the 2D and 3D pipelines.

Why this exists
---------------
The 2D and 3D tracks were maintained as parallel copies, and keeping them at
parity by hand did not work: a fix landed in one and was forgotten in the other,
and the drift was invisible until a result looked wrong. Measured across the
seven paired modules, only about a fifth of the divergence was ever about
dimensionality -- the rest was accumulated drift.

So the goal is one implementation per step, taking its rank from the array it is
handed. Everything in here is the small set of operations that genuinely differ
between 2D and 3D, expressed once:

  * structuring elements  -- ``disk`` vs ``ball``, ``(3,3)`` vs ``(3,3,3)``
  * spacing conventions   -- ``(Y,X)`` vs ``(Z,Y,X)``, and "in-plane" meaning
                             the last two axes either way
  * physical-to-pixel     -- microns to voxels, per axis or against the finest
                             in-plane axis
  * tiling                -- overlapping tiles of a bounding box, any rank

Nothing here knows anything about somata, watersheds or cells. If a helper needs
pipeline context, it belongs in the step that uses it, not here.

Conventions
-----------
``spacing`` is always ordered like the array axes and given in microns per voxel:
``(Y, X)`` for 2D, ``(Z, Y, X)`` for 3D. "In-plane" therefore means the last two
entries in both cases, which is why several helpers index ``[-2:]`` rather than
``[1:]`` -- the latter silently drops Y in 2D.
"""

import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class InvalidSpacingError(ValueError):
    """
    Raised when physical voxel spacing is missing, non-finite or non-positive.

    Deliberately an error and not a default. There is no defensible value to
    substitute: assuming 1.0 um/voxel silently reinterprets every micron-valued
    parameter in the pipeline as a pixel count, and every distance, size and
    density it reports is then wrong by the ratio of the real spacing to 1.0 --
    with nothing in the output to say so.

    This matches `metadata.require_dimensions`, which refuses to start a run
    without dimensions for the same reason: a run that cannot be trusted must not
    start.
    """


def _warn(msg: str) -> None:
    """Loud, greppable, and on stdout so it lands in the process log."""
    print(msg)
    sys.stdout.flush()

__all__ = [
    "InvalidSpacingError",
    "ndim_of",
    "structuring_element",
    "adjacency_footprint",
    "binary_structure",
    "inplane_spacing",
    "min_inplane_spacing",
    "normalise_spacing",
    "pixels_from_physical",
    "sigma_from_physical",
    "generate_tiles",
    "chunk_read_write_slices",
    "write_offset_in_read",
    "planes_of",
    "tile_slices",
    "tile_target_contains",
]


def ndim_of(arr) -> int:
    """Rank of an array, memmap or anything else exposing ``.ndim``."""
    return int(np.asarray(arr).ndim if not hasattr(arr, "ndim") else arr.ndim)


# --------------------------------------------------------------------------
# Structuring elements
# --------------------------------------------------------------------------

def structuring_element(ndim: int, radius: int):
    """
    Ball (3D) or disk (2D) of the given radius, for morphological operations.

    A radius of 1 or less degenerates to the immediate-neighbour footprint,
    matching what both tracks did with ``radius > 1`` guards at their call sites.
    """
    from skimage.morphology import ball, disk
    if radius <= 1:
        return adjacency_footprint(ndim)
    if ndim == 3:
        return ball(radius)
    if ndim == 2:
        return disk(radius)
    raise ValueError(f"structuring_element supports 2D and 3D, got {ndim}D")


def adjacency_footprint(ndim: int):
    """The 3-wide cube/square: every immediate neighbour including diagonals."""
    from skimage.morphology import footprint_rectangle
    return footprint_rectangle((3,) * ndim)


def binary_structure(ndim: int, connectivity: Optional[int] = None):
    """
    ``scipy.ndimage`` connectivity structure.

    ``connectivity=None`` means full connectivity (8 in 2D, 26 in 3D), which is
    what both tracks used for labelling fragments. Passing 1 gives face-only
    (4 in 2D, 6 in 3D).
    """
    from scipy import ndimage
    return ndimage.generate_binary_structure(
        ndim, ndim if connectivity is None else connectivity
    )


# --------------------------------------------------------------------------
# Spacing
# --------------------------------------------------------------------------

def normalise_spacing(spacing: Optional[Sequence[float]], ndim: int) -> Tuple[float, ...]:
    """
    Spacing as exactly ``ndim`` positive finite floats, in axis order.

    A longer sequence keeps its trailing axes, so a 3D ``(Z, Y, X)`` spacing given
    to a 2D call site yields ``(Y, X)`` -- the conversion both strategies were
    doing inline as ``spacing[1:] if len(spacing) == 3 else spacing``. That is a
    real reduction of supplied information, not an invention.

    Anything missing, too short, non-finite or non-positive raises
    `InvalidSpacingError`. It does NOT fall back to 1.0: physical scale is data,
    and inventing it corrupts every measurement downstream while looking like a
    successful run.
    """
    if spacing is None:
        raise InvalidSpacingError(
            f"No voxel spacing supplied for a {ndim}D array. Physical spacing "
            "cannot be defaulted -- assuming 1.0 um/voxel would silently turn "
            "every micron-valued parameter into a pixel count. Pass the spacing "
            "derived from the image's physical dimensions."
        )
    try:
        s = tuple(float(v) for v in spacing)
    except (TypeError, ValueError):
        raise InvalidSpacingError(f"Spacing is not a sequence of numbers: {spacing!r}")
    if len(s) < ndim:
        raise InvalidSpacingError(
            f"Spacing {s} has {len(s)} axes but the array is {ndim}D. The missing "
            "axis cannot be defaulted."
        )
    if len(s) > ndim:
        s = s[-ndim:]
    for i, v in enumerate(s):
        if not np.isfinite(v) or v <= 0:
            raise InvalidSpacingError(
                f"Spacing axis {i} is {v!r}, which cannot be a physical voxel "
                f"size. Full spacing: {s}."
            )
    return s


def inplane_spacing(spacing: Sequence[float]) -> Tuple[float, ...]:
    """The in-plane (Y, X) entries: the last two, in 2D and 3D alike."""
    return tuple(float(v) for v in spacing[-2:])


def min_inplane_spacing(spacing: Sequence[float]) -> float:
    """Finest in-plane resolution. ``min(spacing[-2:])``, not ``min(spacing)``.

    In 3D these differ whenever Z is the coarsest axis, which it usually is, and
    using the whole-spacing minimum would silently change lateral distances.
    """
    return float(min(inplane_spacing(spacing)))


def pixels_from_physical(spacing: Sequence[float], physical_distance: float,
                         min_pixels: int = 3, label: str = "distance") -> int:
    """
    A physical distance in microns as a pixel count, against the finest in-plane
    axis.

    In-plane rather than per-axis because this feeds arguments that take a single
    scalar, such as `peak_local_max`'s ``min_distance``. On anisotropic data the
    same micron value is therefore many more pixels laterally than the Z step
    would suggest, which is worth knowing when tuning.

    ``min_pixels`` is an ALGORITHMIC floor, not a physical one. Peak separation
    below a pixel or two is meaningless to the neighbourhood operations this
    feeds, so a smaller request cannot be honoured. When that happens the
    returned count corresponds to a LARGER physical distance than was asked for,
    and this function says so on stdout rather than substituting quietly:

        [SPACING|CLAMP] min peak separation: requested 0.5 um at 1.4485 um/px
        = 0 px, below the 3 px minimum. USING 3 px = 4.35 um (8.7x the
        requested value). Lower the pixel size or raise the requested distance.

    The previous implementations clamped to 3 silently, so a request of 0.1 um on
    a 1.45 um/px image became 4.35 um -- a 43x inflation with nothing in the log.
    The default is kept at 3 so tuned configs behave exactly as before; only the
    silence is removed.

    Invalid spacing raises `InvalidSpacingError`. It does not return the floor:
    the old code did, which produced a pixel count bearing no relation to either
    the requested distance or the real spacing.
    """
    sp = normalise_spacing(spacing, len(tuple(spacing)))
    m = min_inplane_spacing(sp)
    d = float(physical_distance)
    honest = int(round(d / m))
    if honest < min_pixels:
        used_um = min_pixels * m
        ratio = (used_um / d) if d > 0 else float("inf")
        _warn(
            f"  [SPACING|CLAMP] {label}: requested {d:g} um at {m:g} um/px "
            f"= {honest} px, below the {min_pixels} px minimum. "
            f"USING {min_pixels} px = {used_um:.4g} um"
            + (f" ({ratio:.1f}x the requested value)" if np.isfinite(ratio)
               else " (requested value was zero)")
            + ". Lower the pixel size or raise the requested distance."
        )
        return min_pixels
    return honest


def sigma_from_physical(spacing: Sequence[float], physical_sigma: float) -> Tuple[float, ...]:
    """
    Per-axis Gaussian sigma in voxels for a sigma given in microns.

    Per axis, not a scalar: an isotropic blur in voxel space is an anisotropic
    blur in physical space whenever the Z step differs from the lateral one.
    """
    if physical_sigma <= 0:
        return tuple(0.0 for _ in spacing)
    return tuple(float(physical_sigma) / max(float(s), 1e-9) for s in spacing)


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------

def generate_tiles(
    bbox: Sequence[slice],
    tile_size: Optional[Sequence[int]] = None,
    padding: int = 20,
) -> List[Dict[str, Tuple[int, ...]]]:
    """
    Split a bounding box into overlapping tiles of any rank.

    Returns one dict per tile with two flat coordinate tuples:

      ``target`` -- ``(*starts, *stops)``, the region this tile alone owns, so
                    targets tile the box exactly once and a detection is
                    attributed to exactly one tile.
      ``pad``    -- ``(*starts, *stops)``, the target grown by ``padding`` and
                    clipped to the box, giving distance transforms and watersheds
                    enough context not to see a tile edge as an object edge.

    The flat "starts then stops" packing is what both original implementations
    used, so consumers index ``t['target'][k]`` and ``t['target'][k + ndim]``.
    Use `tile_slices` instead of unpacking by hand.

    ``tile_size`` defaults to ``(128, 512, 512)`` in 3D and ``(2048, 2048)`` in
    2D, preserving each track's original default. A scalar is broadcast.
    """
    bbox = tuple(bbox)
    nd = len(bbox)
    if tile_size is None:
        tile_size = (128, 512, 512) if nd == 3 else (2048,) * nd
    elif np.isscalar(tile_size):
        tile_size = (int(tile_size),) * nd
    tile_size = tuple(int(t) for t in tile_size)
    if len(tile_size) != nd:
        tile_size = tile_size[-nd:] if len(tile_size) > nd else \
            (tile_size[0],) * (nd - len(tile_size)) + tile_size

    starts = [int(s.start) for s in bbox]
    stops = [int(s.stop) for s in bbox]

    def rec(axis: int, origin: List[int]):
        if axis == nd:
            tgt_lo = tuple(origin)
            tgt_hi = tuple(min(origin[k] + tile_size[k], stops[k]) for k in range(nd))
            pad_lo = tuple(max(starts[k], origin[k] - padding) for k in range(nd))
            pad_hi = tuple(min(stops[k], tgt_hi[k] + padding) for k in range(nd))
            yield {"target": tgt_lo + tgt_hi, "pad": pad_lo + pad_hi}
            return
        for v in range(starts[axis], stops[axis], tile_size[axis]):
            yield from rec(axis + 1, origin + [v])

    return list(rec(0, []))


def chunk_read_write_slices(
    shape: Sequence[int],
    chunk_shape: Optional[Sequence[int]] = None,
    overlap: int = 0,
):
    """
    Yield ``(read_slice, write_slice)`` pairs for chunked processing, any rank.

    ``write_slice`` regions tile the array exactly once; ``read_slice`` is the
    same region grown by ``overlap`` and clipped, giving a filter enough context
    that it does not see a chunk edge as a structure edge. A worker computes over
    the read region and stores only the write region, which is why the two are
    returned together -- the offset between them is what the worker needs to crop
    its result, and computing it by hand per rank is exactly the code that
    duplicated.

    ``chunk_shape`` defaults to ``(64, 512, 512)`` in 3D and ``(2048, 2048)`` in
    2D, preserving each track's original default. A scalar is broadcast.
    """
    shape = tuple(int(v) for v in shape)
    nd = len(shape)
    if chunk_shape is None:
        chunk_shape = (64, 512, 512) if nd == 3 else (2048,) * nd
    elif np.isscalar(chunk_shape):
        chunk_shape = (int(chunk_shape),) * nd
    chunk_shape = tuple(int(c) for c in chunk_shape)
    if len(chunk_shape) != nd:
        chunk_shape = (chunk_shape[-nd:] if len(chunk_shape) > nd
                       else (chunk_shape[0],) * (nd - len(chunk_shape)) + chunk_shape)

    def rec(axis: int, w: List[slice], r: List[slice]):
        if axis == nd:
            yield tuple(r), tuple(w)
            return
        for start in range(0, shape[axis], chunk_shape[axis]):
            stop = min(start + chunk_shape[axis], shape[axis])
            yield from rec(
                axis + 1,
                w + [slice(start, stop)],
                r + [slice(max(0, start - overlap),
                           min(shape[axis], stop + overlap))],
            )

    yield from rec(0, [], [])


def write_offset_in_read(read_slices: Sequence[slice],
                         write_slices: Sequence[slice]) -> Tuple[slice, ...]:
    """
    Where the write region sits inside the read region, as a slice tuple.

    A worker computes over the padded read block and must store only the part it
    owns; this is the crop. Both tracks open-coded it with per-axis
    ``*_start_rel`` / ``*_stop_rel`` arithmetic, which is rank-specific by
    construction and had to be rewritten for 2D.
    """
    out = []
    for r, w in zip(read_slices, write_slices):
        lo = w.start - r.start
        out.append(slice(lo, lo + (w.stop - w.start)))
    return tuple(out)


def planes_of(block: np.ndarray):
    """
    Iterate ``(index, 2D plane)`` over a block, whatever its rank.

    A 3D block yields its Z planes; a 2D block yields itself once, at index 0.
    This is what lets one implementation serve both ranks wherever a filter is
    inherently two-dimensional -- the vesselness enhancement, for instance, runs
    ``frangi`` slice by slice in 3D and once in 2D, which is the same code with a
    loop that has length 1.
    """
    if block.ndim == 2:
        yield 0, block
    elif block.ndim == 3:
        for i in range(block.shape[0]):
            yield i, block[i]
    else:
        raise ValueError(f"planes_of supports 2D and 3D blocks, got {block.ndim}D")


def tile_slices(coords: Sequence[int]) -> Tuple[slice, ...]:
    """A flat ``(*starts, *stops)`` tuple as a slice tuple for indexing."""
    n = len(coords) // 2
    return tuple(slice(int(coords[k]), int(coords[k + n])) for k in range(n))


def tile_target_contains(target: Sequence[int], point: Sequence[float]) -> bool:
    """
    Is ``point`` inside a tile's exclusively-owned target region?

    Used to attribute a detection to exactly one tile. Both tracks open-coded
    this with hardcoded index arithmetic, which is precisely the kind of thing
    that has to be rewritten per rank if it is not centralised.
    """
    n = len(target) // 2
    return all(target[k] <= point[k] < target[k + n] for k in range(n))
