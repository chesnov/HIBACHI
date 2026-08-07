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
        "mode": mode,
        "processed_dir": processed_dir,
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


def plan_roi_propagation(
    sample_dirs: Sequence[str],
    full_shape: Sequence[int],
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

        roi_dir = info["roi_dir"]
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


def plan_roi_clear(sample_dirs: Sequence[str]) -> List[Dict[str, Any]]:
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

        roi_dir = info["roi_dir"]
        has_json = os.path.isfile(os.path.join(roi_dir, ROI_JSON_NAME))
        files: List[str] = []
        if os.path.isdir(roi_dir):
            try:
                files = sorted(os.listdir(roi_dir))
            except OSError:
                files = []

        if has_json:
            status = HAS_ROI
        elif files:
            status = ORPHAN
        else:
            status = NO_ROI

        info.update({
            "status": status,
            # Everything that will be deleted, polygon file included.
            "discards": files,
            # Result files only, which is what the user actually cares about
            # losing (the polygon itself is cheap to redraw).
            "outputs": [f for f in files if f != ROI_JSON_NAME],
        })
        plan.append(info)

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
