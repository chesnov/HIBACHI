"""collapse_2d: turn a stack already in a project into a single plane.

Why this exists
---------------
These coverslips are imaged at several focal planes but only one is in focus;
the rest contribute out-of-focus haze. Thirteen planes of a slide-scanner
channel are 24 GB, one plane is 1.86 GB, and twelve regions across four
channels come down from 1.15 TB to 89 GB -- which fits on a laptop instead of
requiring the acquisition drive.

Why it happens AFTER import and not during
------------------------------------------
Choosing between a maximum projection and a single plane, and choosing WHICH
plane, needs someone to look at the image. The user of this dataset could not
open these files in any other program, so until they are in a project there is
nothing to look at and no basis for the choice. Collapsing at setup would
therefore be asking a question that cannot yet be answered.

What it touches
---------------
The project's extracted TIFF is overwritten in place. The original source --
the .vsi and its sidecar, or whatever the channel was extracted from -- is
never opened or altered, so a collapse can always be undone by re-importing.

Everything the project computed from the stack is deleted: results, the run's
provenance config, saved regions, and the sample's relational analyses. They
describe an image that no longer exists, and a stale 3D result sitting beside a
2D image is worse than no result -- volumes would be compared against areas in
one table.

Rank needs no plumbing. `axes_for_mode(ndim=...)` and
`_probe_pixel_counts_quiet` both read rank from the array, and
`calculate_features` dispatches on `ndim`, so a one-plane image simply IS a 2D
project. The display pyramid invalidates itself, since its manifest
fingerprints the image's size and mtime.

Per channel, per sample
-----------------------
The METHOD is chosen per channel: different fluorophores focus at different
depths, so the best plane for DAPI need not be the best for Cy5. The
APPLICATION is per sample -- every channel or none -- because
`cross_channel_window._geometry_for` takes shape and spacing from the first
channel and applies them to all, so a sample with one channel collapsed and
another not would silently measure one of them against the other's geometry.
"""

from __future__ import annotations

import glob
import os
import shutil
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile as tiff  # type: ignore
import yaml  # type: ignore

#: Collapse methods.
MAX_PROJECTION = "max"
SINGLE_PLANE = "plane"

#: Key recording how a collapse was done. Added to `_PASSTHROUGH_KEYS` in
#: processing_strategies, without which `save_config` drops it on the next save
#: -- the whitelist that already lost `dimensions` once.
COLLAPSE_KEY = "collapse_2d"



#: Longest in-plane edge a preview level may have. Big enough to pan around and
#: recognise an artifact, small enough to convert to an image and page through
#: planes without waiting.
_PREVIEW_MAX_EDGE = 4096


def preview_planes(tif_path: str):
    """A reduced (Z, Y, X) view for previewing planes, or None.

    Comes from the display pyramid, which keeps every plane at every level, so
    paging through thirteen planes reads megabytes rather than the 24 GB the
    full-resolution stack would. Picks the finest level that is still small
    enough to render.

    Strictly for LOOKING. The focus metric never reads this -- gradient energy
    is only meaningful between adjacent pixels, and a reduced level has none
    of the original ones. Two data paths for two purposes.

    Returns None when there is no current pyramid; the caller should build one
    first rather than fall back to reading full-resolution planes.
    """
    try:
        from .display_pyramid import open_levels
        levels = open_levels(tif_path)
    except Exception:
        return None
    if not levels or len(levels) < 2:
        return None
    for level in levels[1:]:
        if max(int(level.shape[-2]), int(level.shape[-1])) <= _PREVIEW_MAX_EDGE:
            return level
    return levels[-1]


def _spec(method: str, z_index: Optional[int] = None) -> Dict[str, Any]:
    """A normalised collapse spec."""
    if method == SINGLE_PLANE:
        if z_index is None:
            raise ValueError("a single-plane collapse needs a z index")
        return {"method": SINGLE_PLANE, "z_index": int(z_index)}
    if method != MAX_PROJECTION:
        raise ValueError(f"unknown collapse method {method!r}")
    return {"method": MAX_PROJECTION}


def sample_tif(sample_dir: str) -> Optional[str]:
    """The image in a sample folder, or None.

    A sample folder holds exactly one TIFF -- `ProjectManager` only recognises
    it as a sample if that is true -- so this is unambiguous.
    """
    try:
        for name in sorted(os.listdir(sample_dir)):
            if name.lower().endswith((".tif", ".tiff")):
                return os.path.join(sample_dir, name)
    except OSError:
        pass
    return None


def sample_config(sample_dir: str) -> Optional[str]:
    try:
        for name in sorted(os.listdir(sample_dir)):
            if name.lower().endswith((".yaml", ".yml")):
                return os.path.join(sample_dir, name)
    except OSError:
        pass
    return None


def plane_count(tif_path: str) -> int:
    """Planes in the image, read from its header by axis letter."""
    from .metadata import probe_tiff_axes

    sizes = probe_tiff_axes(tif_path).get("sizes") or {}
    return int(sizes.get("Z", 1)) if sizes else 1


def is_collapsible(tif_path: str) -> bool:
    """Whether this image is a stack with something to collapse."""
    return plane_count(tif_path) > 1


#: Rows per sampled band, and bands per plane. A band spans the FULL width, so
#: it is one sequential read; scattered square tiles would be hundreds of
#: thousands of kilobyte-sized seeks on the drive this data lives on, which is
#: fast only when read sequentially.
_BAND_ROWS = 512
_BANDS = 3

#: Width of a scoring tile within a band.
_TILE = 512


def _tenengrad(tile) -> float:
    """Mean squared gradient magnitude above the tile's own median.

    Gradient energy rises with edge steepness, so an in-focus plane scores
    higher than the same signal blurred. Masking to pixels above the median
    keeps a mostly-empty field from being scored on the noise between its
    cells.
    """
    values = np.asarray(tile, dtype=np.float32)
    if values.size < 16:
        return 0.0
    gy, gx = np.gradient(values)
    energy = gy * gy + gx * gx
    bright = values > float(np.median(values))
    return float(energy[bright].mean()) if bright.any() else float(energy.mean())


def focus_scores(tif_path: str) -> Dict[str, Any]:
    """Per-plane focus scores, the sharpest plane, and how clear the win is.

    Each plane is scored as the MEDIAN of per-tile Tenengrad values, and the
    median is the whole point. Every other aggregate is hijacked by a single
    bright, sharp artifact -- a speck of dust on the coverslip has steeper
    edges than any cell, and on a test stack with one such speck the mean over
    tiles, the mean over the sharpest tiles, the whole-plane mean and a strided
    sample ALL picked the artifact's plane, by factors of 20 to 100. The median
    picked the truly focused plane and rated the artifact's plane unremarkable.
    An artifact is local; focus is not.

    Tiles are cut from full-width bands of rows rather than scattered over the
    plane: a band is one sequential read, and this data lives on a drive that
    manages ~84 MB/s sequentially and a fifth of that when seeking. Tiles are
    full resolution, because gradient energy is only meaningful between
    adjacent pixels.

    `margin` is the best score over the runner-up, so a caller can tell a clear
    win from a coin toss. It cannot tell whether the sharpest plane is the one
    the user wants: the bulk of a cell and its soma need not lie in the same
    plane, and no metric knows which was intended. Treat the result as a
    suggestion with the scores shown.

    Returns ``{}`` if the image is not a stack or cannot be read.
    """
    depth = plane_count(tif_path)
    if depth <= 1:
        return {}
    try:
        stack = tiff.memmap(tif_path, mode="r")
    except Exception:
        return {}

    try:
        height, width = int(stack.shape[-2]), int(stack.shape[-1])
        band_rows = min(_BAND_ROWS, height)
        starts = np.linspace(0, max(0, height - band_rows), _BANDS)
        starts = sorted({int(s) for s in starts})
        columns = list(range(0, max(1, width - _TILE + 1), _TILE)) or [0]

        scores: List[float] = []
        for z in range(depth):
            per_tile: List[float] = []
            for row in starts:
                band = np.asarray(stack[z][row:row + band_rows])
                for col in columns:
                    per_tile.append(_tenengrad(band[:, col:col + _TILE]))
            scores.append(float(np.median(per_tile)) if per_tile else 0.0)
    except Exception:
        return {}
    finally:
        del stack

    order = sorted(range(depth), key=lambda i: scores[i], reverse=True)
    best = order[0]
    runner_up = scores[order[1]] if depth > 1 else 0.0
    margin = ((scores[best] - runner_up) / scores[best]) if scores[best] else 0.0
    return {"scores": scores, "best": int(best), "margin": float(margin)}


def _collapse_array(tif_path: str, spec: Dict[str, Any],
                    progress=None, should_cancel=None) -> np.ndarray:
    """The single plane a spec asks for.

    A maximum projection reads every plane and keeps a running maximum, so peak
    memory is two planes whatever the depth. A single-plane collapse reads one
    plane and nothing else.
    """
    from .slide_reader import SetupCancelled

    stack = tiff.memmap(tif_path, mode="r")
    try:
        depth = int(stack.shape[0])
        if spec["method"] == SINGLE_PLANE:
            index = int(spec["z_index"])
            if not 0 <= index < depth:
                raise ValueError(
                    f"plane {index} was asked for but the image has {depth}")
            if progress is not None:
                progress(1, 1)
            return np.array(stack[index])

        out = np.array(stack[0])
        for z in range(1, depth):
            if should_cancel is not None and should_cancel():
                raise SetupCancelled("cancelled while collapsing")
            np.maximum(out, stack[z], out=out)
            if progress is not None:
                progress(z + 1, depth)
        return out
    finally:
        del stack


def _write_plane(dest_path: str, plane: np.ndarray, source_tif: str) -> None:
    """Write one plane over `dest_path`, preserving in-plane calibration.

    Uncompressed contiguous ImageJ, like everything else the project writes,
    because `app_launch` opens these with `tiff.memmap`. Written to a `.part`
    and renamed, so an interrupted collapse cannot leave a truncated image
    where the project expects a whole one.
    """
    resolution = None
    unit = "micron"
    try:
        with tiff.TiffFile(source_tif) as handle:
            page = handle.pages[0]
            x_res = page.tags.get("XResolution")
            y_res = page.tags.get("YResolution")

            def _density(tag):
                value = tag.value
                if isinstance(value, tuple) and len(value) == 2 and value[1]:
                    return value[0] / value[1]
                return float(value)

            if x_res is not None and y_res is not None:
                resolution = (_density(x_res), _density(y_res))
            info = handle.imagej_metadata or {}
            unit = str(info.get("unit") or unit)
    except Exception:
        resolution = None

    kwargs: Dict[str, Any] = {
        "imagej": True,
        "photometric": "minisblack",
        # 'axes' explicitly: tifffile's ImageJ writer labels an unannotated
        # array's leading axis as channels, which is how every extracted stack
        # came to claim 13 channels rather than 13 slices.
        "metadata": {"axes": "YX", "unit": unit},
    }
    if resolution is not None:
        kwargs["resolution"] = resolution

    partial = dest_path + ".part"
    try:
        tiff.imwrite(partial, plane, **kwargs)
        os.replace(partial, dest_path)
    except BaseException:
        try:
            if os.path.isfile(partial):
                os.remove(partial)
        except OSError:
            pass
        raise


def derived_paths(sample_dir: str, project_root: Optional[str] = None,
                  sample_name: Optional[str] = None) -> List[str]:
    """Everything the project computed from this image, for deletion.

    Globs rather than exact names, for the reason `ARTIFACT_PATTERNS` uses
    them: a legacy project's results are named after a mode string that no
    longer exists, and deleting only the current build's names leaves the real
    results on disk while reporting success.

    Covers the results directory, saved ROI sessions (named
    ``<basename>_processed_<mode>_roi[_<slug>]``, so the same glob finds them),
    the run provenance config that lives inside the results directory, the
    display pyramid, and the sample's subfolder in each relational analysis.
    Does NOT cover the image or its config: one is rewritten, the other edited.
    """
    targets: List[str] = []
    for pattern in ("*_processed_*", "_display", "relational_preview_temp"):
        targets.extend(sorted(glob.glob(os.path.join(sample_dir, pattern))))

    if project_root and sample_name:
        analyses = os.path.join(project_root, "RELATIONAL_ANALYSIS")
        if os.path.isdir(analyses):
            targets.extend(sorted(
                glob.glob(os.path.join(analyses, "*", sample_name))))
    # Deduplicate while keeping order: a pattern can match a path another
    # already did, and deleting twice would raise on the second attempt.
    return list(dict.fromkeys(targets))


def purge_derived(sample_dir: str, project_root: Optional[str] = None,
                  sample_name: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """Delete everything `derived_paths` names. Returns (deleted, errors)."""
    deleted: List[str] = []
    errors: List[str] = []
    for path in derived_paths(sample_dir, project_root, sample_name):
        try:
            if os.path.isdir(path):
                shutil.rmtree(path)
            elif os.path.exists(path):
                os.remove(path)
            deleted.append(path)
        except OSError as exc:
            errors.append(f"{os.path.basename(path)}: {exc}")
    return deleted, errors


def rewrite_config(config_path: str, spec: Dict[str, Any],
                   depth: int, scores: Optional[Sequence[float]] = None) -> None:
    """Drop the z extent and record how the collapse was done.

    The z extent has to go: a 2D image has no depth, and every consumer derives
    spacing as extent divided by pixel count, so leaving it would divide the
    old depth by one plane. `get_sample_metadata` treats the presence of a 'z'
    entry as what makes an acquisition a stack, so removing it is also what
    makes this a 2D project.

    The collapse record is provenance, not decoration: a maximum projection and
    plane 6 give different numbers from the same acquisition, and a result
    nobody can trace back to the choice that produced it is not reproducible.
    """
    from .metadata import find_dimensions

    with open(config_path) as handle:
        config = yaml.safe_load(handle) or {}

    key, block = find_dimensions(config)
    if key and isinstance(block, dict):
        block = dict(block)
        block.pop("z", None)
        config[key] = block

    record: Dict[str, Any] = {
        "method": spec["method"],
        "source_planes": int(depth),
    }
    if spec["method"] == SINGLE_PLANE:
        record["z_index"] = int(spec["z_index"])
    if scores is not None:
        record["focus_scores"] = [round(float(s), 6) for s in scores]
    config[COLLAPSE_KEY] = record

    partial = config_path + ".part"
    with open(partial, "w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    os.replace(partial, config_path)


def collapse_channel(sample_dir: str, spec: Dict[str, Any],
                     project_root: Optional[str] = None,
                     scores: Optional[Sequence[float]] = None,
                     progress=None, should_cancel=None) -> Dict[str, Any]:
    """Collapse one channel's copy of one sample. Returns a summary.

    Order matters. The image is written first and the derived files are purged
    only after it is in place: purging first would leave a project with no
    results AND a stack, which reads as an unprocessed 3D sample and invites a
    re-run of exactly the work that was about to be discarded.
    """
    tif_path = sample_tif(sample_dir)
    if tif_path is None:
        raise FileNotFoundError(f"no image found in {sample_dir}")
    depth = plane_count(tif_path)
    if depth <= 1:
        return {"sample_dir": sample_dir, "skipped": "already a single plane"}

    plane = _collapse_array(tif_path, spec, progress=progress,
                            should_cancel=should_cancel)
    _write_plane(tif_path, plane, tif_path)
    del plane

    config_path = sample_config(sample_dir)
    if config_path is not None:
        rewrite_config(config_path, spec, depth, scores)

    deleted, errors = purge_derived(
        sample_dir, project_root, os.path.basename(os.path.normpath(sample_dir)))
    return {
        "sample_dir": sample_dir,
        "method": spec["method"],
        "z_index": spec.get("z_index"),
        "source_planes": depth,
        "deleted": deleted,
        "errors": errors,
    }
