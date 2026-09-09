"""
roi_sharing: apply ONE region of interest to every channel of a multi-channel sample.

Why this works at all
---------------------
Every channel of a sample is extracted from the same source acquisition, so all
of a sample's channels have identical pixel dimensions. An ROI polygon is stored
by ``gui_manager.confirm_roi`` in *full-image YX pixel coordinates* (plus the Z
indices it was drawn on), so the very same polygon is valid verbatim in every
channel -- no coordinate transform is needed.

Why it's cheap
--------------
``GUIManager._try_load_existing_roi_session`` already rebuilds everything it
needs from ``roi_polygon.json`` alone:

  * ``roi_image_crop.dat`` is rebuilt from *that channel's* image when absent
    (gui_manager.py, "if os.path.exists(crop_path) ... else _build_crop_memmap"),
  * ``processing_config_<mode>.yaml`` is rebuilt from *that channel's* config
    when absent (via ``_build_roi_config``).

So propagating an ROI means writing one small JSON file per channel. Each
channel then derives its own crop and its own rescaled config on next open,
which is exactly right: the polygon is shared, the pixel data is not.

The catch this module exists to handle
--------------------------------------
The loader prefers an existing ``roi_image_crop.dat``. Dropping a NEW polygon
next to a STALE crop would silently load the old sub-region while the JSON
claims the new one. Propagation therefore has to clear the derived artifacts of
any previous ROI session, which also discards that session's results -- so the
work is split into a plan (inspect, report, no writes) and an apply step, letting
the caller confirm before anything is destroyed.

Everything here is pure filesystem + numpy logic with no Qt or napari imports,
so it is unit-testable independently of the viewer.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# The one rule for "these two folders are the same sample in different
# channels". Imported rather than reimplemented so sibling_channel_dirs agrees
# with ProjectManager.build_consolidated_sample_registry instead of inventing a
# second, subtly different notion of sameness. gui_text_utils is Qt-free, so
# this module stays importable in a batch worker.
from .gui_text_utils import clean_filename_for_matching

ROI_DIR_SUFFIX = "_roi"
ROI_JSON_NAME = "roi_polygon.json"
ROI_CROP_NAME = "roi_image_crop.dat"

# Smallest usable crop, mirroring the guard in gui_manager.confirm_roi so the
# overlay path can't create an ROI the per-channel path would have rejected.
MIN_CROP_PX = 10

# Peak scratch memory a single polygon/crop band may use. Bands are sized from
# this, so RAM stays flat as the region grows instead of scaling with its area.
ROI_BAND_BYTES = 128 << 20  # 128 MB
# Rows per band are clamped so a very wide region still bands, and a very narrow
# one doesn't degenerate into a million one-row iterations.
ROI_BAND_MIN_ROWS = 8
ROI_BAND_MAX_ROWS = 4096


# --------------------------------------------------------------------------- #
# Polygon rasterisation
# --------------------------------------------------------------------------- #
# Why this is hand-rolled instead of skimage.draw.polygon
# -------------------------------------------------------
# Memory. ``skimage.draw.polygon`` returns the interior as two int64 COORDINATE
# arrays (rr, cc). That costs ~80 bytes of peak RAM per interior pixel -- the
# two 8-byte output arrays plus the growable buffers behind them -- against 1
# byte per pixel for the boolean mask that is all any caller here actually
# wants. A region covering a whole coverslip is billions of pixels, so the
# coordinate arrays alone run to hundreds of GB and the process is killed before
# the mask is ever built. Its point-in-polygon inner loop is also
# O(area x vertices), so tracing a circle with a few hundred vertices multiplies
# the cost again.
#
# The scanline fill below is O(rows x vertices + area) in time, allocates only
# one band at a time, and defines the mask rule explicitly (see
# ``rasterize_polygon_band``) rather than inheriting it from whatever skimage is
# installed.
#
# On the relationship to skimage
# ------------------------------
# The geometry is the same and the two agree pixel-for-pixel on every outline a
# user can draw. They can differ by single pixels in two situations, both
# deliberate:
#
#   * A pixel centre within ~1 ULP of an edge. skimage's answer there is a
#     rounding artifact of its compiled kernel, not a rule: perturbing a
#     circle's vertices by 1e-12 px changes skimage's pixel count. This module
#     uses an explicit ``_BOUNDARY_ULPS`` tolerance instead, so the mask is
#     stable under sub-nanopixel coordinate noise and does not change when
#     skimage is upgraded or rebuilt on a different compiler.
#   * A collinear outline -- every vertex on one straight line. skimage
#     returns just the vertex pixels, which is inconsistent with its own
#     boundary-inclusive behaviour on real polygons. Such an outline encloses no
#     region, so it is rejected in ``roi_record_from_polygons`` and yields an
#     empty mask here. Note this is a collinearity test, not an area test: a
#     self-crossing figure-of-eight has zero signed area but is a real region,
#     and is filled by the even-odd rule as before.
#
# Deliberately NOT matched: skimage's exact last-bit rounding. Pinning the ROI
# mask to that would make every region's pixel count -- and therefore every
# density derived from it -- a function of the installed skimage build.
# How close a pixel centre must be to an edge, in units of floating-point
# spacing (ULPs), to count as lying on it. This is a tolerance rather than an
# exact `==` because vertices arrive as the output of trigonometry and affine
# scaling: a coordinate that is mathematically 190 routinely arrives as
# 189.99999999999997, one ULP away. Testing exact equality would drop the
# tangent pixel at the top and bottom of a circle; a tolerance keeps it, and
# keeps it under coordinate nudges far smaller than a pixel.
#
# 32 ULPs (about 1e-12 px at whole-slide magnitudes) is wide enough to absorb
# accumulated rounding and far too narrow to reach a genuinely different pixel.
_BOUNDARY_ULPS = 32


class PolygonEdges:
    """A polygon's edges, pre-split for scanline filling.

    Built once and reused for every band and every Z slice, so per-band work is
    a small array operation over a handful of edges rather than a fresh sweep.

    ``slanted`` holds the non-horizontal edges (the ones that produce crossings)
    as ``(y_i, x_i, dy, dx, ymin, ymax)``, indexed the way a crossing kernel
    indexes them: base vertex ``i``, other endpoint ``j = i - 1``. ``level``
    holds the horizontal edges as ``(y, xmin, xmax)``; they yield no crossings
    but can still put pixel centres on the boundary.

    An outline with fewer than three vertices, or with zero signed area (every
    vertex collinear), is treated as empty: it encloses no region, so there is
    no mask to build. ``roi_record_from_polygons`` rejects those before they get
    here, and this is the backstop.
    """

    __slots__ = ("y_i", "x_i", "dy", "dx", "ymin", "ymax",
                 "level_y", "level_x0", "level_x1", "y_lo", "y_hi", "empty")

    def __init__(self, poly_yx):
        p = np.asarray(poly_yx, dtype=float)
        if (p.ndim != 2 or p.shape[0] < 3 or p.shape[1] != 2
                or polygon_is_degenerate(p)):
            p = np.empty((0, 2), dtype=float)

        y_i, x_i = p[:, 0], p[:, 1]
        y_j, x_j = np.roll(y_i, 1), np.roll(x_i, 1)

        slanted = y_i != y_j
        self.y_i, self.x_i = y_i[slanted], x_i[slanted]
        self.dy = y_j[slanted] - self.y_i
        self.dx = x_j[slanted] - self.x_i
        self.ymin = np.minimum(self.y_i, y_j[slanted])
        self.ymax = np.maximum(self.y_i, y_j[slanted])

        level = ~slanted
        self.level_y = y_i[level]
        self.level_x0 = np.minimum(x_i[level], x_j[level])
        self.level_x1 = np.maximum(x_i[level], x_j[level])

        if p.shape[0]:
            self.y_lo, self.y_hi = float(y_i.min()), float(y_i.max())
        else:
            self.y_lo = self.y_hi = 0.0
        self.empty = p.shape[0] == 0

    def row_span(self, height: int) -> Tuple[int, int]:
        """Rows of a mask this polygon can touch, clipped to ``[0, height)``.

        Rows outside the outline's own Y extent are all-False, so skipping them
        makes a region tucked into the corner of a huge frame proportionally
        cheap instead of costing a full-frame sweep.
        """
        if self.empty:
            return 0, 0
        row0 = max(0, int(np.floor(self.y_lo)))
        row1 = min(int(height), int(np.ceil(self.y_hi)) + 1)
        return (row0, row1) if row1 > row0 else (0, 0)


def polygon_is_degenerate(poly_yx) -> bool:
    """True when an outline encloses nothing because its vertices are collinear.

    NOT a signed-area test. The shoelace area of a figure-of-eight is exactly
    zero -- the two lobes wind in opposite directions and cancel -- so an area
    test would reject a self-crossing lasso, which is a perfectly ordinary thing
    to draw and which the even-odd rule handles correctly.

    What is actually degenerate is an outline with no width: every vertex on one
    straight line (or all vertices identical). Measured as the largest
    perpendicular deviation from the longest chord, against a tolerance scaled
    to the outline's own extent, so a legitimately thin sliver still counts as a
    region.
    """
    p = np.asarray(poly_yx, dtype=float)
    if p.ndim != 2 or p.shape[0] < 3 or p.shape[1] != 2:
        return True
    if not np.isfinite(p).all():
        return True

    d = p - p[0]
    lengths = (d * d).sum(axis=1)
    longest = int(np.argmax(lengths))
    span = float(np.sqrt(lengths[longest]))
    if span == 0.0:
        return True  # every vertex identical

    # Perpendicular distance of each vertex from the line through p[0] and the
    # furthest vertex.
    cross = d[:, 0] * d[longest, 1] - d[:, 1] * d[longest, 0]
    deviation = float(np.abs(cross).max()) / span
    return deviation <= _BOUNDARY_ULPS * np.spacing(span)


def polygon_edges(poly_yx) -> PolygonEdges:
    """Pre-compute a polygon's edges once for repeated band filling."""
    return PolygonEdges(poly_yx)


def rasterize_polygon_band(
    edges: PolygonEdges,
    row0: int,
    row1: int,
    width: int,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fill rows ``[row0, row1)`` of a polygon into a (row1-row0, width) mask.

    `out` is fully overwritten when given, which lets a caller allocate one band
    buffer and reuse it for every band and every Z slice. Scratch is
    O((row1 - row0) x n_edges) -- independent of the region's area, which is the
    whole point.

    The rule has two terms:

      1. Interior, by the even-odd rule. An edge is counted over the half-open
         span ``[min(y), max(y))`` so a shared vertex isn't counted twice, and a
         column ``c`` is inside a crossing pair ``(a, b)`` when ``a <= c < b``.
         This is the same geometry ``skimage.draw.polygon`` uses.
      2. Boundary: pixel centres lying on an edge, to within
         ``_BOUNDARY_ULPS``. The interior term alone is half-open, so without
         this an axis-aligned outline loses its bottom row and right column, and
         a circle loses its top and bottom tangent pixel.

    Self-intersecting outlines follow the even-odd rule, as before -- a lasso
    that crosses itself leaves the overlap unfilled.
    """
    height = int(row1) - int(row0)
    width = int(width)
    if out is None:
        out = np.zeros((height, width), dtype=bool)
    else:
        out[:] = False
    if edges.empty or height <= 0:
        return out

    rows = np.arange(row0, row1, dtype=float)[:, None]

    if edges.y_i.size:
        xint = edges.dx * (rows - edges.y_i) / edges.dy + edges.x_i

        # --- term 1: interior ---
        crosses = (edges.ymin <= rows) & (rows < edges.ymax)
        if crosses.any():
            # Unhit edges are parked at +inf so sorting leaves each row's real
            # crossings first, in order.
            ordered = np.sort(np.where(crosses, xint, np.inf), axis=1)
            counts = crosses.sum(axis=1)
            for i in np.nonzero(counts)[0]:
                pairs = ordered[i, :counts[i]]
                for k in range(0, pairs.size - 1, 2):
                    start = max(0, int(np.ceil(pairs[k])))
                    stop = min(width, int(np.ceil(pairs[k + 1])))
                    if stop > start:
                        out[i, start:stop] = True

        # --- term 2: pixel centres sitting on a slanted edge ---
        # Closed in y here: an endpoint row is on the boundary even though the
        # interior term deliberately excludes it.
        nearest = np.round(xint)
        tol = _BOUNDARY_ULPS * np.spacing(np.maximum(np.abs(xint), 1.0))
        on_edge = ((edges.ymin <= rows) & (rows <= edges.ymax)
                   & (np.abs(xint - nearest) <= tol)
                   & (nearest >= 0) & (nearest < width))
        if on_edge.any():
            ri, ei = np.nonzero(on_edge)
            out[ri, nearest[ri, ei].astype(np.intp)] = True

    # --- term 2b: horizontal edges, which produce no crossings at all ---
    for y_level, x_start, x_stop in zip(edges.level_y, edges.level_x0,
                                        edges.level_x1):
        row_f = np.round(y_level)
        if abs(y_level - row_f) > _BOUNDARY_ULPS * np.spacing(
                max(abs(y_level), 1.0)):
            continue
        row = int(row_f) - int(row0)
        if not (0 <= row < height):
            continue
        start = max(0, int(np.ceil(x_start)))
        stop = min(width, int(np.floor(x_stop)) + 1)
        if stop > start:
            out[row, start:stop] = True

    return out


# Bytes of (rows x edges) scratch the band filler holds per row per edge: two
# float64 intersection arrays plus two boolean predicates, rounded up.
_BYTES_PER_ROW_EDGE = 40


def band_rows(width: int, bytes_per_row_px: int = 1, n_edges: int = 0,
              budget: int = ROI_BAND_BYTES) -> int:
    """How many rows fit in the scratch budget at this width and vertex count.

    `bytes_per_row_px` is the total per-pixel cost of one band across every
    array held at once, so a caller can account for its own buffers (a uint16
    image band plus its mask, say) and not just the mask.

    `n_edges` matters because the filler's working set is (rows x edges): an
    outline traced with a few thousand vertices would otherwise blow the budget
    on the intersection table even though the mask itself is small.
    """
    per_row = (int(width) * max(1, int(bytes_per_row_px))
               + max(0, int(n_edges)) * _BYTES_PER_ROW_EDGE)
    rows = int(budget) // max(1, per_row)
    return int(np.clip(rows, ROI_BAND_MIN_ROWS, ROI_BAND_MAX_ROWS))


def polygon_mask(poly_yx, shape: Sequence[int]) -> np.ndarray:
    """Boolean interior mask of one polygon, built band by band.

    A drop-in replacement for::

        rr, cc = skimage.draw.polygon(poly[:, 0], poly[:, 1], shape=shape)
        mask = np.zeros(shape, bool); mask[rr, cc] = True

    with identical output and no per-pixel coordinate arrays. The mask itself is
    still shape[0]*shape[1] bytes, so prefer ``rasterize_polygon_band`` where the
    consumer can work a band at a time.
    """
    height, width = int(shape[0]), int(shape[1])
    mask = np.zeros((height, width), dtype=bool)
    edges = polygon_edges(poly_yx)
    row0, row1 = edges.row_span(height)
    if row1 <= row0:
        return mask

    step = band_rows(width, 1, edges.y_i.size)
    for r0 in range(row0, row1, step):
        r1 = min(row1, r0 + step)
        rasterize_polygon_band(edges, r0, r1, width, out=mask[r0:r1])
    return mask


def polygon_pixel_count(poly_yx, shape: Sequence[int]) -> int:
    """Interior pixel count of one polygon without ever holding a full mask.

    Peak memory is one band, so this is safe to call on a region of any size.
    """
    height, width = int(shape[0]), int(shape[1])
    edges = polygon_edges(poly_yx)
    row0, row1 = edges.row_span(height)
    if row1 <= row0:
        return 0

    step = band_rows(width, 1, edges.y_i.size)
    scratch = np.zeros((step, width), dtype=bool)
    total = 0
    for r0 in range(row0, row1, step):
        r1 = min(row1, r0 + step)
        band = rasterize_polygon_band(edges, r0, r1, width,
                                      out=scratch[:r1 - r0])
        total += int(np.count_nonzero(band))
    return total


# --------------------------------------------------------------------------- #
# Building the shared ROI record
# --------------------------------------------------------------------------- #
def roi_record_from_polygons(
    z_polygons: Dict[int, Any],
    full_shape: Sequence[int],
) -> Dict[str, Any]:
    """Build the on-disk v2 ROI record from drawn polygons.

    This is the same computation ``gui_manager.confirm_roi`` performs before
    writing ``roi_polygon.json`` -- union YX bounding box, then a Z range that
    extrudes a lone polygon through the whole stack but clips to the span of
    several. It lives here so the overlay and per-channel paths cannot drift
    apart in how they interpret the same drawing.

    Args:
        z_polygons: {z_index: (N,2) array of (row, col) vertices in FULL-image
            pixel coordinates}. For 2D images use a single entry at z=0.
        full_shape: shape of the full image, (Z,Y,X) or (Y,X).

    Returns:
        The dict to serialise as roi_polygon.json.

    Raises:
        ValueError: if there are no polygons, or the resulting crop is smaller
            than MIN_CROP_PX on a side.
    """
    if not z_polygons:
        raise ValueError("No polygons were provided.")

    full_shape = tuple(int(v) for v in full_shape)
    is_3d = len(full_shape) == 3
    img_h, img_w = full_shape[-2], full_shape[-1]

    arrays = {}
    for z, poly in z_polygons.items():
        arr = np.asarray(poly, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
            raise ValueError(
                f"Polygon at Z={z} must be an (N>=3, 2) array of YX vertices, "
                f"got shape {arr.shape}."
            )
        if not np.isfinite(arr).all():
            raise ValueError(
                f"Polygon at Z={z} contains non-finite vertices."
            )
        # Collinear vertices enclose nothing. Caught here rather than at raster
        # time so the failure is a message about the drawing instead of a
        # region whose crop is silently a blank rectangle.
        if polygon_is_degenerate(arr):
            raise ValueError(
                f"The outline at Z={z} encloses no area -- its vertices are "
                "all in a straight line. Draw a closed shape with some width "
                "to it."
            )
        arrays[int(z)] = arr

    all_yx = np.vstack(list(arrays.values()))
    y0 = max(0, int(np.floor(all_yx[:, 0].min())))
    x0 = max(0, int(np.floor(all_yx[:, 1].min())))
    y1 = min(img_h, int(np.ceil(all_yx[:, 0].max())) + 1)
    x1 = min(img_w, int(np.ceil(all_yx[:, 1].max())) + 1)

    if (y1 - y0) < MIN_CROP_PX or (x1 - x0) < MIN_CROP_PX:
        raise ValueError(
            f"The selected region is too small ({y1 - y0} x {x1 - x0} px); "
            f"it must be at least {MIN_CROP_PX} px on each side."
        )

    if is_3d:
        zs = sorted(arrays)
        if len(zs) == 1:
            # One polygon means "this shape, all the way through Z".
            z0, z1 = 0, full_shape[0]
        else:
            z0 = max(0, zs[0])
            z1 = min(full_shape[0], zs[-1] + 1)
    else:
        z0, z1 = 0, None

    return {
        "format": "v2",
        "z_polygons": [
            {"z": z, "polygon_yx": arrays[z].tolist()} for z in sorted(arrays)
        ],
        "bbox": {"y0": y0, "x0": x0, "y1": y1, "x1": x1, "z0": z0, "z1": z1},
        "full_image_shape": list(full_shape),
    }


# --------------------------------------------------------------------------- #
# Locating each channel's ROI directory
# --------------------------------------------------------------------------- #
def _read_yaml(path: str) -> Dict[str, Any]:
    import yaml  # local: keeps this module importable without a yaml dependency
    try:
        with open(path, "r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _image_shape(path: str) -> Optional[Tuple[int, ...]]:
    """Shape of a TIFF read from its header, without loading pixel data."""
    import tifffile as tiff  # local, as above
    try:
        with tiff.TiffFile(path) as tf:
            return tuple(int(v) for v in tf.series[0].shape)
    except Exception:
        return None


def describe_channel(sample_dir: str) -> Optional[Dict[str, Any]]:
    """Resolve where a channel's ROI lives, or None if the folder isn't usable.

    Mirrors the naming the pipeline uses everywhere else:
    ``<sample>/<tif basename>_processed_<mode>`` for results, with the ROI
    session in that path plus ``_roi``. ``mode`` comes from the channel's own
    config, since channels are free to carry different modes.
    """
    try:
        contents = os.listdir(sample_dir)
    except OSError:
        return None

    tif = next((f for f in contents if f.lower().endswith((".tif", ".tiff"))), None)
    yml = next((f for f in contents if f.lower().endswith((".yaml", ".yml"))), None)
    if not tif or not yml:
        return None

    cfg = _read_yaml(os.path.join(sample_dir, yml))
    mode = cfg.get("mode")
    if not mode or mode in ("unknown", "error"):
        return None

    basename = os.path.splitext(tif)[0]
    processed_dir = os.path.join(sample_dir, f"{basename}_processed_{mode}")
    return {
        "sample_dir": sample_dir,
        "channel": os.path.basename(os.path.dirname(sample_dir)),
        "tif": os.path.join(sample_dir, tif),
        "basename": basename,
        "mode": mode,
        "processed_dir": processed_dir,
        # Legacy single-ROI path. Kept because roi_session_dir(sample, None) and
        # every pre-multi-ROI caller resolve through it; use list_roi_sessions()
        # to enumerate all of a channel's regions.
        "roi_dir": processed_dir + ROI_DIR_SUFFIX,
    }


def _looks_like_sample_folder(path: str) -> bool:
    """True for a folder holding an image and a config, i.e. one channel's sample.

    The same test ``ProjectManager.build_consolidated_sample_registry`` applies,
    kept identical on purpose: the two decide which folders are channels of a
    sample, and if they disagree the ROI channel list and the overlay's channel
    list would disagree. Says nothing about whether the channel is *ready* to be
    processed -- that is describe_channel's stricter question.
    """
    try:
        contents = os.listdir(path)
    except OSError:
        return False
    return (any(f.lower().endswith((".tif", ".tiff")) for f in contents)
            and any(f.lower().endswith((".yaml", ".yml")) for f in contents))


def sibling_channel_dirs(sample_dir: str) -> List[str]:
    """Every channel folder holding the same sample as `sample_dir`, itself first.

    A project is laid out as ``<root>/<channel project>/<sample>``, so a sample's
    other channels are the identically-named folders under the sibling channel
    projects.

    Resolved from the filesystem rather than from a ProjectManager on purpose.
    The single-channel viewer is reachable without one (`project_manager` is an
    optional argument to ``interactive_segmentation_with_config``), and this
    module has to stay importable in a batch worker where no GUI object exists.
    Where a registry IS available the two agree, because both normalise names
    through ``clean_filename_for_matching``.

    `sample_dir` always comes back first and is always included even if it does
    not look like a sample folder -- the caller is standing in it, so excluding
    it would mean refusing to write the region the user just drew.

    A candidate qualifies on the image + config pair alone, which is
    deliberately the registry's rule and NOT the stricter ``describe_channel``
    one. describe_channel also demands a usable processing mode, and filtering on
    that here made a channel that has never been given a mode vanish from the
    caller's channel list instead of appearing in it as an unusable row with a
    reason. Deciding a channel cannot take the region is
    ``plan_roi_propagation``'s job -- it has the UNUSABLE status for exactly
    this -- so the only thing being answered here is "is this a channel of this
    sample at all".

    Returns a single-element list when there is nothing to match against, which
    is what makes every caller safe on a one-channel project.
    """
    sample_dir = os.path.abspath(sample_dir)
    root = os.path.dirname(os.path.dirname(sample_dir))
    target = clean_filename_for_matching(os.path.basename(sample_dir))

    out = [sample_dir]
    # realpath, so a symlinked channel project can't yield the same folder twice
    # and have propagation delete a directory it is about to write into.
    seen = {os.path.realpath(sample_dir)}

    try:
        channel_projects = sorted(os.listdir(root))
    except OSError:
        return out

    for channel in channel_projects:
        channel_path = os.path.join(root, channel)
        if not os.path.isdir(channel_path):
            continue
        try:
            folders = sorted(os.listdir(channel_path))
        except OSError:
            continue
        for folder in folders:
            if clean_filename_for_matching(folder) != target:
                continue
            path = os.path.join(channel_path, folder)
            real = os.path.realpath(path)
            if real in seen or not os.path.isdir(path):
                continue
            if not _looks_like_sample_folder(path):
                continue
            seen.add(real)
            out.append(path)
    return out


# --------------------------------------------------------------------------- #
# Plan / apply
# --------------------------------------------------------------------------- #
# Per-channel plan statuses
NEW = "new"                  # no ROI session yet; nothing will be lost
REPLACE = "replace"          # an ROI session exists; its outputs will be cleared
SHAPE_MISMATCH = "shape_mismatch"  # channel's image doesn't match the drawing
UNUSABLE = "unusable"        # not a valid sample folder (missing tif/yaml/mode)


def choose_shared_roi_name(sample_dirs: Sequence[str],
                           reserved: Sequence[str] = ()) -> str:
    """An ROI name free in EVERY channel of a sample.

    Regions are propagated under one shared name so that "ROI 2" means the same
    region in every channel. Cross-channel analysis within a region depends on
    that correspondence, so the name has to be free everywhere rather than
    allocated per channel.

    `reserved` are names already handed out in this same operation but not yet
    on disk. One Apply can create several regions, and they are all named before
    any is written, so disk state alone would name every one of them the same.

    With a single-element `sample_dirs` and no reservations this is exactly
    ``next_roi_name``, which is why a one-channel project sees no change in
    behaviour from routing through here.
    """
    used = set()

    def _claim(name: Any) -> None:
        for token in str(name).split():
            if token.isdigit():
                used.add(int(token))

    for sample_dir in sample_dirs:
        for session in list_roi_sessions(sample_dir):
            _claim(session["name"])
    for name in reserved:
        _claim(name)

    n = 1
    while n in used:
        n += 1
    return f"{ROI_AUTO_PREFIX} {n}"


def plan_roi_propagation(
    sample_dirs: Sequence[str],
    full_shape: Sequence[int],
    roi_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Inspect each channel and report what propagating the ROI would do.

    Performs NO writes, so the caller can show a confirmation listing exactly
    which channels gain an ROI and which lose existing ROI results.

    A channel whose image shape differs from the shape the ROI was drawn on is
    flagged rather than written to: the polygon's pixel coordinates would be
    meaningless there. In a well-formed project this cannot happen (all channels
    come from one acquisition), so it indicates a hand-edited or mixed project
    and is worth surfacing instead of silently corrupting.
    """
    target = tuple(int(v) for v in full_shape)
    # None means "add a new region"; a name means "replace that region".
    shared_name = roi_name or choose_shared_roi_name(sample_dirs)
    plan: List[Dict[str, Any]] = []

    for sample_dir in sample_dirs:
        info = describe_channel(sample_dir)
        if info is None:
            plan.append({
                "sample_dir": sample_dir,
                "channel": os.path.basename(os.path.dirname(sample_dir)),
                "status": UNUSABLE,
                "reason": "no image + config pair, or no processing mode set",
            })
            continue

        shape = _image_shape(info["tif"])
        if shape is not None and shape != target:
            info.update({
                "status": SHAPE_MISMATCH,
                "shape": shape,
                "reason": f"image is {shape}, ROI was drawn on {target}",
            })
            plan.append(info)
            continue

        roi_dir = roi_session_dir(sample_dir, shared_name) or info["roi_dir"]
        info["roi_name"] = shared_name
        existing = os.path.isfile(os.path.join(roi_dir, ROI_JSON_NAME))
        stale: List[str] = []
        if os.path.isdir(roi_dir):
            try:
                stale = sorted(
                    f for f in os.listdir(roi_dir) if f != ROI_JSON_NAME
                )
            except OSError:
                stale = []

        info.update({
            "roi_dir": roi_dir,
            "status": REPLACE if existing else NEW,
            "shape": shape,
            # Files that will be deleted so the new polygon actually takes
            # effect: the old crop, the old rescaled config, old checkpoints.
            "discards": stale,
        })
        plan.append(info)

    return plan


def apply_roi_propagation(
    plan: Sequence[Dict[str, Any]],
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Write the ROI record into every writable channel in `plan`.

    Only entries with status NEW or REPLACE are touched; SHAPE_MISMATCH and
    UNUSABLE entries are reported back untouched.

    For each target the existing ROI directory is removed outright before the
    record is written. That is deliberate rather than lazy: the session loader
    prefers an existing ``roi_image_crop.dat`` over the JSON, so leaving one
    behind would make a channel silently process the PREVIOUS sub-region. A
    clean directory forces the loader down its rebuild path, deriving the crop
    and the rescaled config from this channel's own image and config.
    """
    written: List[str] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for entry in plan:
        if entry.get("status") not in (NEW, REPLACE):
            skipped.append(entry)
            continue

        roi_dir = entry["roi_dir"]
        try:
            if os.path.isdir(roi_dir):
                shutil.rmtree(roi_dir)
            os.makedirs(roi_dir, exist_ok=True)
            # Write via a temp file + replace so an interrupted write can't leave
            # a truncated JSON that the loader would fail on.
            tmp = os.path.join(roi_dir, ROI_JSON_NAME + ".tmp")
            with open(tmp, "w") as fh:
                json.dump(record, fh, indent=2)
            os.replace(tmp, os.path.join(roi_dir, ROI_JSON_NAME))
            written.append(entry["sample_dir"])
        except Exception as exc:
            errors.append({"sample_dir": entry["sample_dir"], "error": str(exc)})

    return {"written": written, "skipped": skipped, "errors": errors}


def load_existing_rois(sample_dirs: Sequence[str]) -> List[Dict[str, Any]]:
    """Read back the ROI session of each channel that has one.

    The overlay is where an ROI is defined, but the record is stored per channel,
    so the overlay has no memory of its own: without this it opens blank even
    though every channel is cropped. Reading the channels back is what makes the
    ROI visible again after the overlay is closed and reopened.

    Returns one entry per channel that has a loadable ``roi_polygon.json``:
    ``{channel, sample_dir, roi_dir, record, z_polygons, bbox}`` where
    ``z_polygons`` maps Z index -> (N,2) YX array in full-image pixel
    coordinates. Both the v2 format and the legacy single-polygon v1 format are
    accepted, matching what ``_try_load_existing_roi_session`` tolerates.
    """
    out: List[Dict[str, Any]] = []
    for sample_dir in sample_dirs:
        info = describe_channel(sample_dir)
        if info is None:
            continue
        # Every session in the channel, so the overlay can show all regions
        # rather than only the legacy one.
        for session in list_roi_sessions(sample_dir):
            if not session["has_polygon"]:
                continue
            path = os.path.join(session["roi_dir"], ROI_JSON_NAME)
            try:
                with open(path, "r") as fh:
                    record = json.load(fh)
            except Exception:
                continue

            try:
                z_polys = record_polygons(record)
            except Exception:
                continue
            if not z_polys:
                continue

            entry = dict(info)
            entry.update({
                "roi_dir": session["roi_dir"],
                "roi_name": session["name"],
                "record": record,
                "z_polygons": z_polys,
                "bbox": record.get("bbox") or {},
            })
            out.append(entry)
    return out


def group_rois_by_name(
    loaded: Sequence[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Group loaded sessions by region name, preserving discovery order.

    A sample now has several regions, each present in several channels, so the
    overlay works per REGION rather than per channel.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in loaded:
        grouped.setdefault(entry.get("roi_name") or "ROI 1", []).append(entry)
    return grouped


def rois_are_identical(loaded: Sequence[Dict[str, Any]]) -> bool:
    """True if every channel agrees on the geometry of each named region.

    Propagation writes byte-identical records under one shared name, so this is
    normally true. False means some channel was cropped separately -- e.g.
    per-channel regions drawn before regions were shared -- and the overlay should
    say so rather than silently showing one of them. Compared WITHIN each name:
    two differently-named regions are supposed to differ.
    """
    for entries in group_rois_by_name(loaded).values():
        if len(entries) <= 1:
            continue
        first = entries[0]["record"].get("z_polygons")
        if any(e["record"].get("z_polygons") != first for e in entries[1:]):
            return False
    return True


def summarize_plan(plan: Sequence[Dict[str, Any]]) -> str:
    """One-paragraph, user-facing description of what `plan` will do."""
    by_status: Dict[str, List[Dict[str, Any]]] = {}
    for entry in plan:
        by_status.setdefault(entry.get("status", UNUSABLE), []).append(entry)

    lines: List[str] = []
    if by_status.get(NEW):
        lines.append(
            f"{len(by_status[NEW])} channel(s) will get the ROI: "
            + ", ".join(e["channel"] for e in by_status[NEW])
        )
    if by_status.get(REPLACE):
        n_out = sum(len(e.get("discards") or []) for e in by_status[REPLACE])
        lines.append(
            f"{len(by_status[REPLACE])} channel(s) already have an ROI session; "
            f"theirs will be replaced and {n_out} existing ROI output file(s) "
            "will be deleted: "
            + ", ".join(e["channel"] for e in by_status[REPLACE])
        )
    for status, label in ((SHAPE_MISMATCH, "skipped (image size doesn't match)"),
                          (UNUSABLE, "skipped (not a usable image folder)")):
        if by_status.get(status):
            lines.append(
                f"{len(by_status[status])} channel(s) {label}: "
                + ", ".join(
                    f"{e['channel']} — {e.get('reason', '')}".rstrip(" —")
                    for e in by_status[status]
                )
            )
    return "\n\n".join(lines) if lines else "Nothing to do."


def propagation_rows(plan: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Checkbox rows describing a propagation plan, one per channel.

    Returns ``{entry, enabled, default, label, tooltip}`` dicts, ready to hand to
    a checkbox dialog. Qt-free: the caller builds the widgets.

    Shared by the overlay panel and the single-channel viewer so the two paths
    describe the same plan in the same words. When this lived inside the overlay
    panel the per-channel viewer had no way to reuse it, which is how the two
    drawing paths came to behave differently in the first place.

    Ineligible channels come back as disabled rows carrying their reason rather
    than being dropped: a channel silently missing from the list reads as a bug,
    while a greyed row saying "image size doesn't match" explains itself.
    """
    rows: List[Dict[str, Any]] = []
    for entry in plan:
        status = entry.get("status")
        channel = entry.get("channel", "?")
        if status == NEW:
            rows.append({
                "entry": entry, "enabled": True, "default": True,
                "label": f"{channel}  \u2014  no ROI yet",
                "tooltip": entry.get("roi_dir", ""),
            })
        elif status == REPLACE:
            n = len(entry.get("discards") or [])
            rows.append({
                "entry": entry, "enabled": True, "default": True,
                "label": (f"{channel}  \u2014  replaces existing ROI"
                          + (f" ({n} result file(s) deleted)" if n else "")),
                "tooltip": entry.get("roi_dir", ""),
            })
        else:
            reason = entry.get("reason") or (
                "image size doesn't match" if status == SHAPE_MISMATCH
                else "not a usable image folder"
            )
            rows.append({
                "entry": entry, "enabled": False,
                "label": f"{channel}  \u2014  cannot apply: {reason}",
            })
    return rows


# --------------------------------------------------------------------------- #
# Returning channels to the full image
# --------------------------------------------------------------------------- #
# Per-channel statuses for the clear path
HAS_ROI = "has_roi"      # a loadable ROI session: this channel opens cropped
ORPHAN = "orphan"        # ROI folder with no roi_polygon.json -> leftover files only
NO_ROI = "no_roi"        # nothing to do; already opens on the full image


def plan_roi_clear(sample_dirs: Sequence[str],
                   roi_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Inspect each channel and report what returning it to the full image means.

    Performs no writes. ``HAS_ROI`` means ``roi_polygon.json`` is present, so
    ``_try_load_existing_roi_session`` currently offers that channel its cropped
    session on open. ``ORPHAN`` means an ROI folder survives without its polygon
    file -- the channel already opens on the full image, but the leftover crop and
    checkpoints are still occupying disk and can be cleaned up in the same pass.

    Note this is deliberately *stronger* than the per-channel ``clear_roi``
    button, which switches the live session back but leaves ROI outputs on disk
    (so the next open re-offers them). There is no live session to switch here --
    the overlay acts on disk -- so the only meaningful bulk action is to remove
    the session, which is also what makes channels stop prompting.
    """
    plan: List[Dict[str, Any]] = []

    for sample_dir in sample_dirs:
        info = describe_channel(sample_dir)
        if info is None:
            plan.append({
                "sample_dir": sample_dir,
                "channel": os.path.basename(os.path.dirname(sample_dir)),
                "status": UNUSABLE,
                "reason": "no image + config pair, or no processing mode set",
            })
            continue

        # One entry per SESSION, not per channel: a channel can now hold several
        # regions, and clearing is a per-region decision.
        sessions = list_roi_sessions(sample_dir)
        if roi_name is not None:
            sessions = [se for se in sessions if se["name"] == roi_name]

        if not sessions:
            entry = dict(info)
            entry.update({"status": NO_ROI, "discards": [], "outputs": [],
                          "roi_name": roi_name})
            plan.append(entry)
            continue

        for session in sessions:
            roi_dir = session["roi_dir"]
            try:
                files = sorted(os.listdir(roi_dir))
            except OSError:
                files = []
            entry = dict(info)
            entry.update({
                "roi_dir": roi_dir,
                "roi_name": session["name"],
                "status": HAS_ROI if session["has_polygon"] else (
                    ORPHAN if files else NO_ROI),
                # Everything that will be deleted, polygon file included.
                "discards": files,
                # Result files only, which is what the user actually cares about
                # losing (the polygon itself is cheap to redraw).
                "outputs": [f for f in files if f != ROI_JSON_NAME],
            })
            plan.append(entry)

    return plan


def apply_roi_clear(plan: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Delete the ROI session folder of every clearable channel in `plan`.

    Only ``HAS_ROI`` and ``ORPHAN`` entries are touched. Removing the folder is
    what returns a channel to the full image: with no ``roi_polygon.json``,
    ``_try_load_existing_roi_session`` returns False and the viewer opens the full
    image without prompting.
    """
    cleared: List[str] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for entry in plan:
        if entry.get("status") not in (HAS_ROI, ORPHAN):
            skipped.append(entry)
            continue
        roi_dir = entry["roi_dir"]
        try:
            if os.path.isdir(roi_dir):
                shutil.rmtree(roi_dir)
            cleared.append(entry["sample_dir"])
        except Exception as exc:
            errors.append({"sample_dir": entry["sample_dir"], "error": str(exc)})

    return {"cleared": cleared, "skipped": skipped, "errors": errors}


def summarize_clear_plan(plan: Sequence[Dict[str, Any]]) -> str:
    """One-paragraph, user-facing description of what a clear plan will do."""
    active = [e for e in plan if e.get("status") == HAS_ROI]
    orphan = [e for e in plan if e.get("status") == ORPHAN]
    none = [e for e in plan if e.get("status") == NO_ROI]

    lines: List[str] = []
    if active:
        n_out = sum(len(e.get("outputs") or []) for e in active)
        lines.append(
            f"{len(active)} channel(s) will return to the full image, deleting "
            f"{n_out} ROI result file(s): "
            + ", ".join(e["channel"] for e in active)
        )
    if orphan:
        lines.append(
            f"{len(orphan)} channel(s) already open on the full image but have "
            "leftover ROI files that can be removed: "
            + ", ".join(e["channel"] for e in orphan)
        )
    if none:
        lines.append(
            f"{len(none)} channel(s) have no ROI session: "
            + ", ".join(e["channel"] for e in none)
        )
    return "\n\n".join(lines) if lines else "Nothing to do."


# --------------------------------------------------------------------------- #
# How much area / volume was actually analysed
# --------------------------------------------------------------------------- #
def hull_pixel_count(
    edge_mask_path: Optional[str],
    image_shape: Sequence[int],
) -> Optional[int]:
    """Size of the tissue hull step 2 kept, recovered from its saved edge mask.

    Step 2 persists only the hull's one-voxel-thick boundary
    (``hull ^ eroded_hull``), not its interior. The interior is recoverable
    because step 2 builds each hull slice with ``binary_fill_holes``, so a hull
    has no interior voids -- and filling the boundary slice by slice therefore
    reproduces it exactly. Verified voxel-for-voxel against hulls built the way
    step 2 builds them, including hulls that reach the image border (the 3x3x3
    erosion strips the border-adjacent voxels, so the boundary is closed there
    too).

    Read one slice at a time: the boundary mask is the full image shape, which
    for a 189x1536x1536 stack is 446 MB as one array against 2.4 MB per slice.

    Returns None when there is no mask to read or nothing in it -- which is what
    edge trimming being switched off looks like -- leaving the caller to fall
    back to the full extent.
    """
    if not edge_mask_path or not os.path.isfile(edge_mask_path):
        return None

    shape = tuple(int(v) for v in image_shape)
    try:
        from scipy.ndimage import binary_fill_holes
    except Exception:
        return None

    try:
        shell = np.memmap(edge_mask_path, dtype=bool, mode="r", shape=shape)
    except Exception as exc:
        print(f"  [Extent] could not read the edge mask ({exc}).")
        return None

    total = 0
    try:
        if len(shape) == 2:
            plane = np.asarray(shell)
            if plane.any():
                total = int(np.count_nonzero(binary_fill_holes(plane)))
        else:
            for z in range(shape[0]):
                plane = np.asarray(shell[z])
                if plane.any():
                    total += int(np.count_nonzero(binary_fill_holes(plane)))
    except Exception as exc:
        print(f"  [Extent] could not measure the hull ({exc}).")
        return None
    finally:
        del shell

    return total if total > 0 else None


def masked_pixel_count(
    record: Dict[str, Any],
    image_shape: Optional[Sequence[int]] = None,
) -> Optional[int]:
    """Number of pixels (2D) or voxels (3D) INSIDE an ROI's polygons.

    This is deliberately not the crop's pixel count. ``_build_crop_memmap``
    writes a bounding-box-shaped array and then zeroes everything outside the
    polygon, so the array is larger than the region actually analysed. For a
    diagonal or otherwise irregular polygon the bounding box can be nearly twice
    the polygon's area, and using it would inflate every density derived from it.

    Mirrors ``_build_crop_memmap``'s nearest-polygon-per-slice rule exactly, so
    the count matches the pixels that were really kept: slices between two drawn
    Z levels take the nearer polygon, and a lone polygon applies to every slice.

    Returns None when the record can't be interpreted, or when `image_shape` is
    given and disagrees with the crop the record describes (which means the
    record belongs to a different image and must not be trusted).
    """
    bbox = record.get("bbox") or {}
    try:
        y0, x0 = int(bbox["y0"]), int(bbox["x0"])
        y1, x1 = int(bbox["y1"]), int(bbox["x1"])
    except (KeyError, TypeError, ValueError):
        return None
    crop_h, crop_w = y1 - y0, x1 - x0
    if crop_h <= 0 or crop_w <= 0:
        return None

    try:
        if "z_polygons" in record:
            z_polys = {int(e["z"]): np.asarray(e["polygon_yx"], dtype=float)
                       for e in record["z_polygons"]}
        else:
            z_polys = {0: np.asarray(record["polygon_yx"], dtype=float)}
    except Exception:
        return None
    if not z_polys:
        return None

    z0 = int(bbox.get("z0") or 0)
    z1 = bbox.get("z1")
    full_shape = record.get("full_image_shape") or []
    is_3d = len(full_shape) == 3 or (image_shape is not None and len(image_shape) == 3)

    if is_3d:
        if z1 is None:
            z1 = int(full_shape[0]) if len(full_shape) == 3 else None
            if z1 is None and image_shape is not None:
                z1 = z0 + int(image_shape[0])
        if z1 is None:
            return None
        crop_depth = int(z1) - z0
        if crop_depth <= 0:
            return None
        expected = (crop_depth, crop_h, crop_w)
    else:
        expected = (crop_h, crop_w)

    if image_shape is not None and tuple(int(v) for v in image_shape) != expected:
        return None

    sorted_zs = sorted(z_polys)

    def _count_for(nearest_z: int) -> int:
        # Banded: a whole-slide region's mask would be gigabytes as one array,
        # and its coordinate list many times that again.
        poly = z_polys[nearest_z] - np.array([y0, x0], dtype=float)
        return polygon_pixel_count(poly, (crop_h, crop_w))

    cache: Dict[int, int] = {}
    if not is_3d:
        nearest = min(sorted_zs, key=lambda z: abs(z - 0))
        return _count_for(nearest)

    total = 0
    for local_z in range(expected[0]):
        global_z = z0 + local_z
        nearest = min(sorted_zs, key=lambda z: abs(z - global_z))
        if nearest not in cache:
            cache[nearest] = _count_for(nearest)
        total += cache[nearest]
    return total


def analyzed_extent(
    processed_dir: str,
    image_shape: Sequence[int],
    spacing: Sequence[float],
    is_2d: bool,
    edge_mask_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Physical area (2D) or volume (3D) that a run actually analysed.

    Answers "what was the denominator?" so counts from a full image and from a
    sub-region can be turned into comparable densities.

    An ROI session is detected by ``roi_polygon.json`` sitting in
    ``processed_dir`` -- which is what the pipeline's processed_dir IS once
    ``_switch_to_roi_mode`` has run. Deriving the region from that file rather
    than from a value stamped at crop time means sessions created before this
    existed are measured correctly too, and there is nothing to keep in sync.

    `spacing` is per-pixel (z, y, x) as the strategies hold it; z is 1.0 in 2D.
    Returns keys: ``region`` ('full_image' | 'roi'), ``pixels`` (count),
    ``area_um2`` or ``volume_um3``, and for an ROI ``bbox_pixels`` plus
    ``polygon_fraction_of_bbox`` so the shape's efficiency is visible.
    """
    shape = tuple(int(v) for v in image_shape)
    try:
        zs, ys, xs = (float(spacing[0]), float(spacing[1]), float(spacing[2]))
    except (IndexError, TypeError, ValueError):
        zs = ys = xs = 1.0

    unit_key = "area_um2" if is_2d else "volume_um3"
    per_pixel = (ys * xs) if is_2d else (zs * ys * xs)

    bbox_pixels = 1
    for dim in shape:
        bbox_pixels *= int(dim)

    # Two extents are reported, always:
    #   TOTAL  -- the whole image, or the ROI polygon, i.e. the region offered
    #             to the pipeline.
    #   TISSUE -- the hull step 2 kept inside that region.
    # They are equal when no hull exists (edge trimming off) or when the hull
    # reached the edges of the region, both of which are ordinary outcomes.
    hull_pixels = hull_pixel_count(edge_mask_path, shape)
    tissue_key = "tissue_area_um2" if is_2d else "tissue_volume_um3"

    def _with_tissue(base: Dict[str, Any], total_pixels: int,
                     no_hull_basis: str) -> Dict[str, Any]:
        tissue = hull_pixels if hull_pixels else total_pixels
        base["tissue_pixels"] = tissue
        base[tissue_key] = tissue * per_pixel
        base["extent_basis"] = ("tissue_hull" if hull_pixels else no_hull_basis)
        return base

    out: Dict[str, Any] = _with_tissue({
        "region": "full_image",
        "pixels": bbox_pixels,
        unit_key: bbox_pixels * per_pixel,
    }, bbox_pixels, "full_image")

    roi_json = os.path.join(processed_dir or "", ROI_JSON_NAME)
    if not os.path.isfile(roi_json):
        return out

    try:
        with open(roi_json, "r") as fh:
            record = json.load(fh)
    except Exception:
        return out

    count = masked_pixel_count(record, image_shape=shape)
    if count is None or count <= 0:
        # An ROI session whose polygon can't be measured must not silently report
        # the bounding box as if it were the region: say the region is an ROI and
        # flag the number as unverified rather than quietly overstating it.
        out["region"] = "roi"
        out["polygon_measured"] = False
        return out

    out.update(_with_tissue({
        "region": "roi",
        "pixels": count,
        unit_key: count * per_pixel,
        "bbox_pixels": bbox_pixels,
        "polygon_fraction_of_bbox": (count / bbox_pixels) if bbox_pixels else None,
        "polygon_measured": True,
    }, count, "polygon"))
    return out


# --------------------------------------------------------------------------- #
# ROI session engine
# --------------------------------------------------------------------------- #
# Phase 0 of multi-ROI support: this section holds the machinery that used to
# live inside DynamicGUIManager. It is Qt-free and napari-free on purpose --
# batch processing runs in a child process with no GUI, so while the crop builder
# and the config deriver were methods on the GUI class, an ROI could not be
# processed except interactively. Nothing here imports from gui_manager.

ROI_CONFIG_PREFIX = "processing_config_"


def processed_dir_name(tif_basename: str, mode: str) -> str:
    """Directory name a strategy writes its results into.

    ONE definition of this, on purpose. The pattern
    ``<basename>_processed_<mode>`` was previously re-derived independently in
    project_selection.sample_status, batch_processor (twice) and
    project_scaffolding.apply_template_config_to_project. Parallel derivations of
    the same rule are how the 2D pipeline ended up passing a temp directory the
    3D pipeline passed and the 2D one didn't -- so multi-ROI support routes every
    caller through here instead of adding a fifth copy.
    """
    return f"{tif_basename}_processed_{mode}"


def roi_dir_name(tif_basename: str, mode: str, roi_name: Optional[str] = None) -> str:
    """Directory name for one ROI session.

    ``roi_name=None`` gives the legacy unnamed form ``..._roi``, which is adopted
    in place rather than migrated: renaming a directory that may hold completed
    results is a needless risk when reading the old name costs nothing.
    """
    base = processed_dir_name(tif_basename, mode) + ROI_DIR_SUFFIX
    return base if not roi_name else f"{base}_{slugify_roi_name(roi_name)}"


def slugify_roi_name(name: str) -> str:
    """Filesystem-safe form of an ROI name."""
    out = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))
    return out.strip("_") or "roi"


# Auto-assigned names look like "ROI 1". The number is what appears on disk, so
# the slug of "ROI 1" is "ROI_1".
ROI_AUTO_PREFIX = "ROI"


def roi_display_name(dir_name: str) -> str:
    """Human name for an ROI directory, inverse of roi_dir_name.

    The legacy unnamed directory reads as "ROI 1" so it takes its place in a
    numbered list without being renamed on disk.
    """
    base = os.path.basename(str(dir_name).rstrip("/\\"))
    marker = ROI_DIR_SUFFIX + "_"
    if marker in base:
        return base.rsplit(marker, 1)[1].replace("_", " ")
    return f"{ROI_AUTO_PREFIX} 1"


def list_roi_sessions(sample_dir: str) -> List[Dict[str, Any]]:
    """Every ROI session in a sample folder, ordered for display.

    Returns dicts of ``{name, dir_name, roi_dir, has_polygon, legacy}``. The
    legacy unnamed session sorts first so adopting it in place keeps it as
    "ROI 1".
    """
    info = describe_channel(sample_dir)
    if info is None:
        return []
    processed = os.path.basename(info["processed_dir"])
    prefix = processed + ROI_DIR_SUFFIX
    try:
        entries = sorted(os.listdir(sample_dir))
    except OSError:
        return []

    out: List[Dict[str, Any]] = []
    for entry in entries:
        full = os.path.join(sample_dir, entry)
        if not os.path.isdir(full) or not entry.startswith(prefix):
            continue
        legacy = (entry == prefix)
        out.append({
            "name": roi_display_name(entry),
            "dir_name": entry,
            "roi_dir": full,
            "has_polygon": os.path.isfile(os.path.join(full, ROI_JSON_NAME)),
            "legacy": legacy,
        })
    out.sort(key=lambda e: (not e["legacy"], _name_sort_key(e["name"])))
    return out


def _name_sort_key(name: str):
    """Sort 'ROI 2' before 'ROI 10' rather than lexically."""
    import re
    parts = re.split(r"(\d+)", str(name))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def next_roi_name(sample_dir: str) -> str:
    """Next auto-assigned ROI name for a sample folder, e.g. 'ROI 3'.

    Numbers are derived from the sessions currently on disk, so a number DOES
    become available again once its session is deleted -- deleting "ROI 2" and
    drawing a new region gives you "ROI 2" back. That is the behaviour most people
    expect from an auto-numbered list, and avoiding it would mean persisting a
    counter. The tradeoff worth knowing: results already exported from the old
    "ROI 2" refer to a region the new "ROI 2" is not, so an exported CSV is only
    unambiguous alongside the run it came from.
    """
    used = set()
    for session in list_roi_sessions(sample_dir):
        for token in str(session["name"]).split():
            if token.isdigit():
                used.add(int(token))
    n = 1
    while n in used:
        n += 1
    return f"{ROI_AUTO_PREFIX} {n}"


def roi_session_dir(sample_dir: str, roi_name: Optional[str] = None) -> Optional[str]:
    """Absolute path of one ROI session directory, or None if unresolvable.

    With ``roi_name=None`` this returns the legacy unnamed path, preserving the
    behaviour of every existing caller.
    """
    info = describe_channel(sample_dir)
    if info is None:
        return None
    if roi_name is None:
        return info["roi_dir"]
    # An existing session wins over a freshly derived name so a legacy folder
    # adopted as "ROI 1" resolves to its real directory.
    for session in list_roi_sessions(sample_dir):
        if session["name"] == roi_name:
            return session["roi_dir"]
    basename = os.path.splitext(os.path.basename(info["tif"]))[0]
    return os.path.join(sample_dir,
                        roi_dir_name(basename, info["mode"], roi_name))


# --------------------------------------------------------------------------- #
# Building a session's derived artifacts
# --------------------------------------------------------------------------- #
def build_crop_memmap(
    src,
    y0: int, x0: int, y1: int, x1: int,
    z_polygons: Dict[int, Any],
    out_path: str,
    z0_crop: int = 0,
    z1_crop: Optional[int] = None,
    quiet: bool = False,
):
    """Write a cropped, polygon-masked copy of `src` and return an 'r+' memmap.

    Moved verbatim from DynamicGUIManager._build_crop_memmap so that batch
    processing -- which has no GUI object -- can build an ROI crop.

    `z_polygons` maps global Z indices to YX polygon arrays in FULL-IMAGE
    coordinates. Each crop slice uses the nearest defined polygon, which covers
    three cases with one rule: a 2D image (one entry at z=0), a 3D region extruded
    through Z (one entry, applied to every slice), and a true 3D region (one entry
    per drawn level, nearest polygon in between). Slices outside the drawn range
    take the first or last polygon rather than extrapolating to empty.

    Memory is bounded by ``ROI_BAND_BYTES`` regardless of how big the region is.
    The crop is written in horizontal bands, and each band's mask is rasterised
    on the spot: nothing full-region-sized is ever held, which is what lets a
    region covering an entire slide scan be created at all. Rows the polygon
    cannot reach are left as the zeros the memmap was created with rather than
    being read, masked and written back.
    """
    is_3d = src.ndim == 3
    crop_h, crop_w = int(y1 - y0), int(x1 - x0)

    if is_3d:
        if z1_crop is None:
            z1_crop = src.shape[0]
        crop_depth = int(z1_crop - z0_crop)
        crop_shape = (crop_depth, crop_h, crop_w)
    else:
        crop_depth = 1
        crop_shape = (crop_h, crop_w)

    crop_mm = np.memmap(out_path, dtype=src.dtype, mode='w+', shape=crop_shape)
    sorted_zs = sorted(int(z) for z in z_polygons.keys())

    # One edge table per drawn Z level, in crop-relative coordinates. Built once
    # for the whole run: the per-band cost is then a small array operation over
    # these edges, not a fresh point-in-polygon sweep of the region.
    offset = np.array([y0, x0], dtype=float)
    edges_by_z = {
        z: polygon_edges(np.asarray(z_polygons[z], dtype=float) - offset)
        for z in sorted_zs
    }

    # Only rows some polygon can reach need touching at all.
    row0, row1 = crop_h, 0
    for z in sorted_zs:
        r0, r1 = edges_by_z[z].row_span(crop_h)
        if r1 > r0:
            row0, row1 = min(row0, r0), max(row1, r1)
    if row1 <= row0:
        crop_mm.flush()
        return crop_mm

    # Per band we hold: the image band (itemsize), one inverse-mask buffer, and
    # one rasterised mask per drawn Z level. Sizing from the real total keeps the
    # budget honest on 16-bit data and on 3D regions with many drawn levels.
    per_px = int(np.dtype(src.dtype).itemsize) + 1 + max(1, len(sorted_zs))
    max_edges = max((edges_by_z[z].y_i.size for z in sorted_zs), default=0)
    step = band_rows(crop_w, per_px, max_edges)

    if not quiet:
        what = (f"{crop_depth} slices x {crop_h} x {crop_w}" if is_3d
                else f"{crop_h} x {crop_w}")
        print(f"  [ROI] Building {'3D' if is_3d else '2D'} crop ({what}) in "
              f"bands of {step} rows...")

    # Reused across every band and slice; nothing else is allocated in the loop.
    mask_scratch = {z: np.zeros((step, crop_w), dtype=bool) for z in sorted_zs}
    inv_scratch = np.zeros((step, crop_w), dtype=bool)

    for band0 in range(row0, row1, step):
        band1 = min(row1, band0 + step)
        rows = band1 - band0

        band_masks = {
            z: rasterize_polygon_band(edges_by_z[z], band0, band1, crop_w,
                                      out=mask_scratch[z][:rows])
            for z in sorted_zs
        }

        for local_z in range(crop_depth):
            global_z = z0_crop + local_z
            nearest_z = min(sorted_zs, key=lambda z: abs(z - global_z))
            mask2d = band_masks[nearest_z]

            if is_3d:
                data = np.array(src[global_z, y0 + band0:y0 + band1, x0:x1])
            else:
                data = np.array(src[y0 + band0:y0 + band1, x0:x1])

            # `~mask2d` would allocate a fresh band-sized bool on every slice of
            # every band; writing into the scratch buffer keeps the loop
            # allocation-free. Explicit zeroing (rather than multiplying by the
            # mask) preserves the original behaviour for float data with NaNs.
            outside = np.logical_not(mask2d, out=inv_scratch[:rows])
            data[outside] = 0

            if is_3d:
                crop_mm[local_z, band0:band1, :] = data
            else:
                crop_mm[band0:band1, :] = data

        # Push each band to disk as it is finished so dirty pages don't
        # accumulate into a second copy of the region in RAM.
        crop_mm.flush()

    crop_mm.flush()
    return crop_mm


def build_roi_config(
    y0: int, x0: int, y1: int, x1: int,
    base_config: Dict[str, Any],
    full_shape: Sequence[int],
    mode: str,
    z0: int = 0,
    z1: Optional[int] = None,
) -> Dict[str, Any]:
    """Deep-copied config with physical dimensions rescaled to the crop extent.

    Moved out of DynamicGUIManager and given `full_shape` and `mode` explicitly
    instead of reading them off the GUI object, so batch processing can call it.

    The YAMLs store TOTAL physical extent rather than per-voxel size, so crop
    dimensions scale linearly with pixel count:
        new_x_um = original_x_um * (crop_w / full_w)
    which leaves per-voxel spacing unchanged while making the config
    self-consistent for the smaller array.

    IMPORTANT for multi-ROI: this is a one-time SEED for a new ROI's config, not
    something to recompute on every open. Each ROI owns its config once created,
    so re-deriving it would silently discard parameters the user tuned for that
    region.
    """
    import copy as _copy

    from .metadata import (DIMENSIONS_KEY, LEGACY_DIMENSION_KEYS, config_ndim,
                           require_dimensions)

    roi_config = _copy.deepcopy(base_config)

    # Rank comes from the config's own dimension block, not from the mode string
    # (which is now the same for every project) and not from `full_shape`, whose
    # length is not a reliable rank signal -- a 2D image with a channel axis is
    # also three-dimensional. The block either declares a z extent or it does
    # not, and that is the acquisition's own statement about its rank.
    ndim = config_ndim(base_config)

    # No fallback. A region derives its extent by scaling the full image's, so an
    # absent or mismatched dimension block cannot be substituted for -- doing so
    # produced sub-micron totals and silently rescaled every physical parameter.
    orig_dims = require_dimensions(base_config,
                                   source="the full image's config")
    orig_x = orig_dims['x']
    orig_y = orig_dims['y']

    full_h = int(full_shape[-2])
    full_w = int(full_shape[-1])

    # Built from the validated axes only, so a stray 'z' cannot leak into a 2D
    # config -- a dimension block carrying an unexpected z is a sign of the old
    # fallback.
    new_dims = {}
    new_dims['x'] = orig_x * ((x1 - x0) / full_w)
    new_dims['y'] = orig_y * ((y1 - y0) / full_h)

    if ndim == 3 and 'z' in orig_dims and len(full_shape) == 3:
        orig_z = orig_dims['z']
        full_z = int(full_shape[0])
        effective_z1 = z1 if z1 is not None else full_z
        new_dims['z'] = orig_z * ((effective_z1 - z0) / full_z)

    roi_config[DIMENSIONS_KEY] = new_dims
    # Drop any legacy block copied from the base config. Leaving one behind
    # would keep the FULL image's extent in the ROI's config: harmless while the
    # unified key is preferred, but a trap for anything that reads the legacy key
    # directly, and misleading to a human reading the file.
    for _legacy in LEGACY_DIMENSION_KEYS:
        roi_config.pop(_legacy, None)
    return roi_config


def load_roi_record(roi_dir: str) -> Optional[Dict[str, Any]]:
    """The polygon record in an ROI directory, or None."""
    path = os.path.join(roi_dir, ROI_JSON_NAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except Exception:
        return None


def record_polygons(record: Dict[str, Any]) -> Dict[int, Any]:
    """{z: (N,2) YX array} from a v2 or legacy v1 record."""
    if "z_polygons" in record:
        return {int(e["z"]): np.asarray(e["polygon_yx"], dtype=float)
                for e in record["z_polygons"]}
    return {0: np.asarray(record["polygon_yx"], dtype=float)}


def ensure_roi_artifacts(
    sample_dir: str,
    roi_name: Optional[str] = None,
    image_stack=None,
    base_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Make an ROI session ready to process, rebuilding what is missing.

    This is what lets an ROI be processed without a GUI. Given a sample folder and
    an ROI name it guarantees the crop memmap and the ROI config exist, deriving
    each from the channel's own image and config when absent -- the same lazy
    rebuild the interactive path already performed, but callable from a batch
    worker.

    Returns ``{roi_dir, config, config_path, crop_path, crop_shape, mode,
    record}`` or None if the session has no polygon to work from.

    `image_stack` may be omitted when the crop already exists on disk; it is only
    read when the crop has to be rebuilt.
    """
    import yaml as _yaml
    from .metadata import MissingDimensionsError

    info = describe_channel(sample_dir)
    if info is None:
        return None
    roi_dir = roi_session_dir(sample_dir, roi_name)
    if not roi_dir or not os.path.isdir(roi_dir):
        return None

    record = load_roi_record(roi_dir)
    if record is None:
        return None
    bbox = record.get("bbox") or {}
    try:
        y0, x0 = int(bbox["y0"]), int(bbox["x0"])
        y1, x1 = int(bbox["y1"]), int(bbox["x1"])
    except (KeyError, TypeError, ValueError):
        return None
    z0 = int(bbox.get("z0") or 0)
    z1 = bbox.get("z1")

    mode = info["mode"]
    crop_path = os.path.join(roi_dir, ROI_CROP_NAME)

    if not os.path.isfile(crop_path) or os.path.getsize(crop_path) == 0:
        if image_stack is None:
            import tifffile as _tiff
            image_stack = _tiff.memmap(info["tif"], mode="r")
        build_crop_memmap(image_stack, y0, x0, y1, x1,
                          record_polygons(record), crop_path,
                          z0_crop=z0, z1_crop=z1, quiet=True)

    config_path = os.path.join(roi_dir, f"{ROI_CONFIG_PREFIX}{mode}.yaml")
    if os.path.isfile(config_path):
        # Persisted config wins: it may carry parameters tuned for this region,
        # and re-deriving would throw them away.
        try:
            with open(config_path, "r") as fh:
                config = _yaml.safe_load(fh) or {}
        except Exception:
            config = None
    else:
        config = None

    if config is None:
        if base_config is None:
            # Not wrapped in a swallow: an unreadable channel config used to become
            # {}, which the dimension fallback then turned into invented extents.
            channel_cfg = _channel_config_path(info)
            if not channel_cfg or not os.path.isfile(channel_cfg):
                raise MissingDimensionsError(
                    f"No config found for {os.path.basename(sample_dir)}, so this "
                    "region's physical dimensions cannot be derived.")
            with open(channel_cfg, "r") as fh:
                base_config = _yaml.safe_load(fh) or {}
        full_shape = _full_shape_of(info)
        config = build_roi_config(y0, x0, y1, x1, base_config, full_shape,
                                  mode, z0=z0, z1=z1)
        try:
            with open(config_path, "w") as fh:
                _yaml.safe_dump(config, fh, sort_keys=False)
        except OSError as exc:
            print(f"  [ROI] could not persist ROI config: {exc}")

    is_3d = len(_full_shape_of(info) or ()) == 3
    crop_h, crop_w = y1 - y0, x1 - x0
    if is_3d:
        eff_z1 = z1 if z1 is not None else (_full_shape_of(info) or (0,))[0]
        crop_shape = (int(eff_z1) - z0, crop_h, crop_w)
    else:
        crop_shape = (crop_h, crop_w)

    # The crop is a raw memmap with no header, so its dtype has to come from the
    # source image -- a caller reopening it needs both shape and dtype or it will
    # read the bytes wrongly.
    crop_dtype = None
    try:
        import tifffile as _tiff
        with _tiff.TiffFile(info["tif"]) as tf:
            crop_dtype = np.dtype(tf.series[0].dtype)
    except Exception:
        crop_dtype = np.dtype(np.uint16)

    return {
        "roi_dir": roi_dir, "config": config, "config_path": config_path,
        "crop_path": crop_path, "crop_shape": crop_shape,
        "crop_dtype": crop_dtype, "mode": mode,
        "record": record, "sample_dir": sample_dir,
        "roi_name": roi_name or roi_display_name(os.path.basename(roi_dir)),
    }


# Files in a region session that are NOT results: the region's definition, its
# cached crop (derived from the polygon, independent of any config) and the config
# itself. Everything else is output and is removed when the config changes.
_REGION_KEEP_PREFIXES = (ROI_JSON_NAME, ROI_CROP_NAME, ROI_CONFIG_PREFIX)


def is_region_result_file(name: str) -> bool:
    """True if a file in a region session is a computed OUTPUT.

    The single definition of "result" for a region, used both to clear them and to
    report status. The polygon, the cached crop and the config are not results: the
    crop is derived from the polygon and is independent of any parameter, so a
    region that has merely been opened has produced nothing.
    """
    return not os.path.basename(str(name)).startswith(_REGION_KEEP_PREFIXES)


def region_result_files(roi_dir: str) -> List[str]:
    """Names of a region's computed outputs, or [] if it has none."""
    try:
        return [f for f in sorted(os.listdir(roi_dir))
                if is_region_result_file(f)]
    except OSError:
        return []


def clear_region_results(roi_dir: str) -> int:
    """Delete a region's computed outputs, keeping its definition and config.

    Applying a new config to a region MUST discard what the old parameters
    produced. Leaving them means the region opens showing segmentation from
    parameters that are no longer displayed anywhere -- there is then no way to
    tell which settings produced which result.

    Implemented as a keep-list rather than a delete-list on purpose: a
    delete-list has to be updated whenever the pipeline learns to write a new
    output, and anything it misses survives as stale data. Only three things are
    ever worth keeping, and they are all config-independent.

    Returns the number of files removed.
    """
    if not roi_dir or not os.path.isdir(roi_dir):
        return 0
    removed = 0
    for name in region_result_files(roi_dir):
        path = os.path.join(roi_dir, name)
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed += 1
        except OSError as exc:
            print(f"  [ROI] could not remove {name}: {exc}")
    return removed


def apply_template_to_regions(
    sample_dirs: Sequence[str],
    template: Dict[str, Any],
    only_regions: Optional[Sequence[str]] = None,
    targets: Optional[Sequence] = None,
    config_name: str = "",
) -> Dict[str, Any]:
    """Push a template's PARAMETERS into saved regions, keeping true dimensions.

    Dimensions are taken from each channel's own full-image config and rescaled to
    the region's crop. They are never taken from the template, because physical
    extent is a property of the IMAGE, not of the settings being applied. A config
    exported from a region already carries crop-scaled extents, so scaling those
    again by the crop fraction shrank them a second time -- measured at 4x wrong,
    which passes every sanity check while making every physical result wrong.

    `only_regions` restricts the update to named regions across `sample_dirs`.
    `targets` is more precise -- a sequence of (sample_dir, region_name) pairs --
    and is what the project view passes when specific region ROWS were checked, so
    a channel's other regions are left alone.

    Returns ``{updated: [...], skipped: [...], errors: [...]}``.
    """
    import copy as _copy
    import yaml as _yaml

    from .metadata import (DIMENSIONS_KEY, LEGACY_DIMENSION_KEYS, config_ndim,
                           require_dimensions)

    updated: List[str] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []
    cleared = 0
    wanted = set(only_regions) if only_regions is not None else None

    # Explicit pairs win, and define which folders are visited at all.
    per_folder: Optional[Dict[str, set]] = None
    if targets:
        per_folder = {}
        for folder, region in targets:
            per_folder.setdefault(folder, set()).add(region)
        sample_dirs = list(per_folder)

    for sample_dir in sample_dirs:
        info = describe_channel(sample_dir)
        if info is None:
            continue
        mode = info["mode"]

        # Extents come from the channel, so an ROI-derived template works too.
        channel_cfg_path = _channel_config_path(info)
        try:
            with open(channel_cfg_path, "r") as fh:
                channel_cfg = _yaml.safe_load(fh) or {}
            full_dims = require_dimensions(
                channel_cfg,
                source=f"{os.path.basename(sample_dir)} (full image)")
        except Exception as exc:
            errors.append({"roi_dir": sample_dir, "error": str(exc)})
            continue

        # The template's parameter blocks belong to a specific pipeline, so a
        # template from a differently-shaped project must not be applied. This
        # used to compare mode strings; with a single mode that comparison is
        # always satisfied and the protection would have disappeared silently.
        # Rank is the property that actually made the parameters
        # non-interchangeable, so it is what gets compared -- a 2D template's
        # sizes and distances mean something different in a z-stack.
        tmpl_ndim = config_ndim(template)
        chan_ndim = config_ndim(channel_cfg)
        if tmpl_ndim and chan_ndim and tmpl_ndim != chan_ndim:
            errors.append({
                "roi_dir": sample_dir,
                "error": (f"template is for a {tmpl_ndim}D acquisition but this "
                          f"channel is {chan_ndim}D; the parameters are not "
                          f"interchangeable"),
            })
            continue

        base = _copy.deepcopy(template)
        base[DIMENSIONS_KEY] = dict(full_dims)
        base["mode"] = mode          # follows the channel, never the template
        for _legacy in LEGACY_DIMENSION_KEYS:
            base.pop(_legacy, None)  # no stale block from a template
        if config_name:
            # Recorded so the project view's Config column names the config that
            # was applied. Without it the column showed the fixed on-disk filename
            # ("processing_config_<mode>"), which never changed however many
            # different configs were applied.
            base["config_name"] = config_name

        allowed = per_folder.get(sample_dir) if per_folder is not None else wanted
        full_shape = _full_shape_of(info)
        for session in list_roi_sessions(sample_dir):
            if allowed is not None and session["name"] not in allowed:
                continue
            if not session["has_polygon"]:
                skipped.append({"roi_dir": session["roi_dir"],
                                "reason": "no polygon"})
                continue
            record = load_roi_record(session["roi_dir"])
            bbox = (record or {}).get("bbox") or {}
            try:
                y0, x0 = int(bbox["y0"]), int(bbox["x0"])
                y1, x1 = int(bbox["y1"]), int(bbox["x1"])
            except (KeyError, TypeError, ValueError):
                skipped.append({"roi_dir": session["roi_dir"],
                                "reason": "unreadable bbox"})
                continue
            try:
                new_cfg = build_roi_config(
                    y0, x0, y1, x1, base, full_shape or (1, 1), mode,
                    z0=int(bbox.get("z0") or 0), z1=bbox.get("z1"))
                path = os.path.join(
                    session["roi_dir"], f"{ROI_CONFIG_PREFIX}{mode}.yaml")

                # Destructive by design: results from the previous parameters are
                # removed before the new config lands, so a region can never show
                # data that its displayed settings did not produce. Skipped only
                # when the config is byte-identical, where there is nothing to
                # be confused about.
                previous = None
                if os.path.isfile(path):
                    try:
                        with open(path, "r") as fh:
                            previous = _yaml.safe_load(fh) or {}
                    except Exception:
                        previous = None
                if previous != new_cfg:
                    cleared += clear_region_results(session["roi_dir"])

                with open(path, "w") as fh:
                    _yaml.safe_dump(new_cfg, fh, default_flow_style=False,
                                    sort_keys=False)
                updated.append(path)
            except Exception as exc:
                errors.append({"roi_dir": session["roi_dir"], "error": str(exc)})

    return {"updated": updated, "skipped": skipped, "errors": errors,
            "cleared": cleared}


def count_regions(sample_dirs: Sequence[str]) -> int:
    """How many saved regions the given channels hold in total."""
    return sum(1 for d in sample_dirs
               for se in list_roi_sessions(d) if se["has_polygon"])


def regions_common_to_channels(
    channel_dirs: Sequence[str],
    require_segmentation: bool = False,
) -> List[str]:
    """Region names present in EVERY given channel, in display order.

    Cross-channel analysis compares one region's mask across channels, so a region
    only qualifies if every channel has it. Because regions propagate under one
    shared name with one shared polygon, a name present everywhere is guaranteed to
    describe the same crop everywhere -- which is what makes the masks line up
    voxel-for-voxel.

    With `require_segmentation` a region also has to have been processed in every
    channel, so the picker offers only what can actually be analysed rather than
    failing partway through a recipe.
    """
    if not channel_dirs:
        return []

    per_channel: List[List[str]] = []
    for sample_dir in channel_dirs:
        names = []
        for session in list_roi_sessions(sample_dir):
            if not session["has_polygon"]:
                continue
            if require_segmentation and not _has_segmentation(session["roi_dir"]):
                continue
            names.append(session["name"])
        per_channel.append(names)

    common = set(per_channel[0])
    for names in per_channel[1:]:
        common &= set(names)
    # Keep the first channel's order so the picker matches the tree.
    return [n for n in per_channel[0] if n in common]


def _has_segmentation(directory: str) -> bool:
    """True if a results directory holds a final segmentation mask."""
    try:
        return any(f.startswith("final_segmentation") and f.endswith(".dat")
                   for f in os.listdir(directory))
    except OSError:
        return False


def region_geometry(sample_dir: str, roi_name: str) -> Optional[Dict[str, Any]]:
    """Crop shape and per-voxel spacing of one region, for relational analysis.

    Relational analysis memmaps each channel's mask against a shape and converts
    distances with a spacing, and for a region both must describe the CROP. Taking
    them from the channel's full-resolution TIFF -- as the full-image path does --
    would read past the end of a region's mask.

    Returns ``{shape, spacing, roi_dir, mode}`` or None.
    """
    art = ensure_roi_artifacts(sample_dir, roi_name)
    if art is None:
        return None
    shape = tuple(int(v) for v in art["crop_shape"])
    config = art["config"] or {}
    # Whichever key the ROI config carries: unified for anything written since
    # the modes merged, legacy for a project created before it. Derived from the
    # config rather than from `art["mode"]`, which is the same string for every
    # project now and so cannot select a key.
    from .metadata import MissingDimensionsError, find_dimensions
    try:
        _dim_key, dims = find_dimensions(config)
    except MissingDimensionsError:
        # Ambiguous (both legacy blocks present). Handled below exactly as a
        # missing block is, preserving this function's Optional contract.
        dims = None
    dims = dims or {}

    # The config stores TOTAL microns for the crop, so per-voxel spacing is that
    # divided by the crop's pixel count -- matching how the full-image path derives
    # spacing from its own dimensions.
    #
    # `_per_px` returns None for a missing or unusable total instead of the 1.0
    # it used to substitute. Inventing a scale here does not fail loudly: the
    # region's masks memmap fine and relational analysis produces distances and
    # densities that read as microns while actually being in voxels. Refusing
    # costs nothing, because this function's contract is already Optional and
    # both callers (cross_channel_window._resolve_geometry and
    # spatial_null.runner._geometry_fallback) already treat None as "geometry
    # unavailable" and say so.
    def _per_px(total, count):
        try:
            total = float(total)
        except (TypeError, ValueError):
            return None
        if not (count and total > 0):
            return None
        return total / count

    axes = ("z", "y", "x") if len(shape) == 3 else ("y", "x")
    spacing = []
    for axis, count in zip(axes, shape):
        per_px = _per_px(dims.get(axis), count)
        if per_px is None:
            print(f"[ROI] {roi_name!r} in {sample_dir}: no usable {axis!r} "
                  f"extent in the region config; cannot derive a spacing.")
            return None
        spacing.append(per_px)

    return {"shape": shape, "spacing": tuple(spacing), "roi_dir": art["roi_dir"],
            "mode": art["mode"]}


def _channel_config_path(info: Dict[str, Any]) -> str:
    """Path of a channel's own YAML config."""
    sample_dir = info["sample_dir"]
    for f in sorted(os.listdir(sample_dir)):
        if f.lower().endswith((".yaml", ".yml")):
            return os.path.join(sample_dir, f)
    return ""


def _full_shape_of(info: Dict[str, Any]) -> Optional[Tuple[int, ...]]:
    """Full image shape of a channel, from the TIFF header."""
    return _image_shape(info["tif"])