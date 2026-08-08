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

ROI_DIR_SUFFIX = "_roi"
ROI_JSON_NAME = "roi_polygon.json"
ROI_CROP_NAME = "roi_image_crop.dat"

# Smallest usable crop, mirroring the guard in gui_manager.confirm_roi so the
# overlay path can't create an ROI the per-channel path would have rejected.
MIN_CROP_PX = 10


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


# --------------------------------------------------------------------------- #
# Plan / apply
# --------------------------------------------------------------------------- #
# Per-channel plan statuses
NEW = "new"                  # no ROI session yet; nothing will be lost
REPLACE = "replace"          # an ROI session exists; its outputs will be cleared
SHAPE_MISMATCH = "shape_mismatch"  # channel's image doesn't match the drawing
UNUSABLE = "unusable"        # not a valid sample folder (missing tif/yaml/mode)


def choose_shared_roi_name(sample_dirs: Sequence[str]) -> str:
    """An ROI name free in EVERY channel of a sample.

    Regions are propagated under one shared name so that "ROI 2" means the same
    region in every channel. Cross-channel analysis within a region depends on
    that correspondence, so the name has to be free everywhere rather than
    allocated per channel.
    """
    used = set()
    for sample_dir in sample_dirs:
        for session in list_roi_sessions(sample_dir):
            for token in str(session["name"]).split():
                if token.isdigit():
                    used.add(int(token))
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
    from skimage.draw import polygon as _raster  # lazy: keeps this module light

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
        poly = z_polys[nearest_z] - np.array([y0, x0], dtype=float)
        rr, cc = _raster(poly[:, 0], poly[:, 1], shape=(crop_h, crop_w))
        mask = np.zeros((crop_h, crop_w), dtype=bool)
        mask[rr, cc] = True
        return int(mask.sum())

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

    out: Dict[str, Any] = {
        "region": "full_image",
        "pixels": bbox_pixels,
        unit_key: bbox_pixels * per_pixel,
    }

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

    out.update({
        "region": "roi",
        "pixels": count,
        unit_key: count * per_pixel,
        "bbox_pixels": bbox_pixels,
        "polygon_fraction_of_bbox": (count / bbox_pixels) if bbox_pixels else None,
        "polygon_measured": True,
    })
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
    """
    from skimage.draw import polygon as skimage_polygon

    is_3d = src.ndim == 3
    crop_h, crop_w = y1 - y0, x1 - x0

    if is_3d:
        if z1_crop is None:
            z1_crop = src.shape[0]
        crop_depth = z1_crop - z0_crop
        crop_shape = (crop_depth, crop_h, crop_w)
    else:
        crop_shape = (crop_h, crop_w)

    crop_mm = np.memmap(out_path, dtype=src.dtype, mode='w+', shape=crop_shape)
    sorted_zs = sorted(z_polygons.keys())

    def _mask_for_z(global_z: int):
        nearest_z = min(sorted_zs, key=lambda z: abs(z - global_z))
        poly = np.asarray(z_polygons[nearest_z], dtype=float) - np.array(
            [y0, x0], dtype=float)
        rr, cc = skimage_polygon(poly[:, 0], poly[:, 1], shape=(crop_h, crop_w))
        m = np.zeros((crop_h, crop_w), dtype=bool)
        m[rr, cc] = True
        return m

    if is_3d:
        if not quiet:
            print(f"  [ROI] Building 3D crop "
                  f"({crop_depth} slices x {crop_h} x {crop_w})...")
        mask_cache: Dict[int, Any] = {}
        for local_z in range(crop_depth):
            global_z = z0_crop + local_z
            nearest_z = min(sorted_zs, key=lambda z: abs(z - global_z))
            if nearest_z not in mask_cache:
                mask_cache[nearest_z] = _mask_for_z(global_z)
            mask2d = mask_cache[nearest_z]
            slice_data = np.array(src[global_z, y0:y1, x0:x1])
            slice_data[~mask2d] = 0
            crop_mm[local_z] = slice_data
    else:
        mask2d = _mask_for_z(0)
        crop_mm[:] = src[y0:y1, x0:x1]
        crop_mm[~mask2d] = 0

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

    roi_config = _copy.deepcopy(base_config)
    is_2d_mode = str(mode).endswith('_2d')
    dim_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'

    orig_dims = base_config.get(dim_key, {'x': 1.0, 'y': 1.0, 'z': 1.0})
    orig_x = float(orig_dims.get('x', 1.0))
    orig_y = float(orig_dims.get('y', 1.0))

    full_h = int(full_shape[-2])
    full_w = int(full_shape[-1])

    new_dims = dict(orig_dims)
    new_dims['x'] = orig_x * ((x1 - x0) / full_w)
    new_dims['y'] = orig_y * ((y1 - y0) / full_h)

    if not is_2d_mode and 'z' in orig_dims and len(full_shape) == 3:
        orig_z = float(orig_dims.get('z', 1.0))
        full_z = int(full_shape[0])
        effective_z1 = z1 if z1 is not None else full_z
        new_dims['z'] = orig_z * ((effective_z1 - z0) / full_z)

    roi_config[dim_key] = new_dims
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
            try:
                with open(os.path.join(sample_dir, os.path.basename(
                        _channel_config_path(info))), "r") as fh:
                    base_config = _yaml.safe_load(fh) or {}
            except Exception:
                base_config = {}
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


def apply_template_to_regions(
    sample_dirs: Sequence[str],
    template: Dict[str, Any],
) -> Dict[str, Any]:
    """Push a template config into every saved region of the given channels.

    A region's config is the template's parameter blocks with the DIMENSION block
    re-derived for that region's own crop -- reusing the channel's dimensions would
    describe an array of the wrong size, and copying the region's old parameters
    would defeat the point of applying a template. Each region's existing bbox is
    read from its polygon record so nothing has to be recomputed from pixels.

    Returns ``{updated: [...], skipped: [...], errors: [...]}``.
    """
    import yaml as _yaml

    updated: List[str] = []
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for sample_dir in sample_dirs:
        info = describe_channel(sample_dir)
        if info is None:
            continue
        full_shape = _full_shape_of(info)
        for session in list_roi_sessions(sample_dir):
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
                    y0, x0, y1, x1, template, full_shape or (1, 1),
                    info["mode"], z0=int(bbox.get("z0") or 0),
                    z1=bbox.get("z1"))
                # mode follows the channel, never the template, so a template
                # exported from a 3D project cannot silently convert a 2D region.
                new_cfg["mode"] = info["mode"]
                path = os.path.join(
                    session["roi_dir"],
                    f"{ROI_CONFIG_PREFIX}{info['mode']}.yaml")
                with open(path, "w") as fh:
                    _yaml.safe_dump(new_cfg, fh, default_flow_style=False,
                                    sort_keys=False)
                updated.append(path)
            except Exception as exc:
                errors.append({"roi_dir": session["roi_dir"], "error": str(exc)})

    return {"updated": updated, "skipped": skipped, "errors": errors}


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
    is_2d = str(art["mode"]).endswith("_2d")
    dim_key = "pixel_dimensions" if is_2d else "voxel_dimensions"
    dims = config.get(dim_key) or {}

    # The config stores TOTAL microns for the crop, so per-voxel spacing is that
    # divided by the crop's pixel count -- matching how the full-image path derives
    # spacing from its own dimensions.
    def _per_px(total, count):
        try:
            total = float(total)
        except (TypeError, ValueError):
            return 1.0
        return (total / count) if (count and total > 0) else 1.0

    if len(shape) == 3:
        spacing = (_per_px(dims.get("z", 1.0), shape[0]),
                   _per_px(dims.get("y", 1.0), shape[1]),
                   _per_px(dims.get("x", 1.0), shape[2]))
    else:
        spacing = (_per_px(dims.get("y", 1.0), shape[0]),
                   _per_px(dims.get("x", 1.0), shape[1]))

    return {"shape": shape, "spacing": spacing, "roi_dir": art["roi_dir"],
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