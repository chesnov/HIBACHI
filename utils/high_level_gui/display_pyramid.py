"""display_pyramid: reduced-resolution copies used ONLY to draw pixels.

Why this exists
---------------
A slide-scanner plane is 30402 x 30527 -- 928 megapixels, 1.86 GB at uint16.
Every time napari renders a z-slice it needs that whole plane, and a
four-channel composite needs four of them, 7.4 GB per z-step. On the external
drive this data lives on, measured at ~25 MB/s (a Bulk-Only USB enclosure over
a 4800 rpm portable disk, verified healthy), that is minutes per slice before
anything appears on screen. GPUs also cap textures near 16384 px, so a plane
this wide has to be downscaled in software before it can be uploaded at all.

Given a list of progressively reduced arrays and `multiscale=True`, napari
reads only the visible extent of whichever level matches the current zoom.
Fully zoomed out that is a ~950 px level -- a few megabytes across four
channels instead of 7.4 GB.

This is a DISPLAY pyramid and nothing else reads it. Every measurement the
pipeline makes -- skeletons, branch counts, spur pruning -- depends on detail
that a reduced level does not carry, so the full-resolution TIFF remains the
only input to processing. Keeping the two apart is the whole point: the
prohibition on downsampling applies to what is measured, not to what is drawn.

Layout
------
``<sample>/_display/level_04.tif`` and friends, plus a ``manifest.json``.

A SUBDIRECTORY, not a sibling file, and that is a correctness requirement
rather than tidiness. Eight sites locate a sample's image by taking the first
name in the folder ending in .tif, six of them via `next()` over `os.listdir`,
whose order is arbitrary -- so a second .tif could be segmented instead of the
image. Worse, `ProjectManager` accepts a sample folder only when it holds
EXACTLY one .tif and one .yaml, so an extra .tif would make the sample vanish
from the project view. No discovery site recurses, and none of them tests
directory names, so a subdirectory is inert.

Staleness
---------
`manifest.json` records the full-resolution file's shape, dtype, size and
modification time. A pyramid that does not match its image is ignored rather
than drawn: showing a stale reduction of a re-extracted channel would be a
picture of the wrong data, which is worse than being slow.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile as tiff  # type: ignore

#: Subdirectory holding the reduced levels.
DISPLAY_DIR = "_display"

MANIFEST = "manifest.json"

#: Reduction of the coarsest level to build down to: stop once the largest
#: in-plane axis is at or below this. ~950 px fills a screen and is a few
#: megabytes.
_TARGET_SIDE = 1024

#: Smallest reduction stored. Level 1 (half resolution) is a third of the
#: full-resolution file on its own -- 6 GB per channel here -- and napari does
#: not need it: it picks a level by zoom and reads the full-resolution file
#: directly once you are close enough for half resolution to matter. Starting
#: at a quarter keeps the pyramid to ~8% of the image.
_FIRST_FACTOR = 4


def display_dir(tif_path: str) -> str:
    """The pyramid directory belonging to a full-resolution image."""
    return os.path.join(os.path.dirname(os.path.abspath(tif_path)), DISPLAY_DIR)


def _level_name(factor: int) -> str:
    return f"level_{factor:02d}.tif"


def _fingerprint(tif_path: str) -> Dict[str, Any]:
    """What the pyramid was built from, for staleness checks."""
    stat = os.stat(tif_path)
    with tiff.TiffFile(tif_path) as handle:
        series = handle.series[0]
        shape = [int(n) for n in series.shape]
        dtype = str(series.dtype)
        axes = str(series.axes)
    return {
        "source": os.path.basename(tif_path),
        "shape": shape,
        "dtype": dtype,
        "axes": axes,
        "size": int(stat.st_size),
        "mtime": int(stat.st_mtime),
    }


def factors_for(shape: Tuple[int, ...]) -> List[int]:
    """Reductions worth storing for an image of this shape.

    Halving from `_FIRST_FACTOR` until the largest in-plane axis is small
    enough to draw whole. Returns [] for an image already small enough, which
    is the honest answer: it needs no pyramid.
    """
    plane = shape[-2:]
    longest = max(int(plane[0]), int(plane[1]))
    if longest <= _TARGET_SIDE * (_FIRST_FACTOR // 2):
        return []
    factors: List[int] = []
    factor = _FIRST_FACTOR
    while True:
        factors.append(factor)
        if longest // factor <= _TARGET_SIDE or factor >= 256:
            break
        factor *= 2
    return factors


def build(tif_path: str, progress=None, should_cancel=None) -> List[str]:
    """Build the display pyramid for a full-resolution image. Returns paths.

    Reads the image ONCE, plane by plane, and fills every level from that one
    plane -- so the cost is one pass over the file, not one per level. The
    levels themselves are held in memory while building (all of them together
    are a few percent of the image) and written sequentially afterwards, which
    avoids memory-mapping a write onto a FUSE-mounted drive.

    Reduction is by strided decimation, not averaging. Microglial processes are
    one or two pixels wide and bright against near-zero background; a box mean
    dims them toward the background and they disappear from the overview,
    while taking every Nth pixel keeps their intensity where it lands. This is
    a preview, so aliasing is acceptable and losing the processes is not.

    `progress` is called as progress(done_planes, total_planes).
    """
    from .slide_reader import SetupCancelled

    with tiff.TiffFile(tif_path) as handle:
        series = handle.series[0]
        shape = tuple(int(n) for n in series.shape)
        dtype = np.dtype(str(series.dtype))

    factors = factors_for(shape)
    if not factors:
        return []

    is_3d = len(shape) == 3
    depth = int(shape[0]) if is_3d else 1
    height, width = int(shape[-2]), int(shape[-1])

    levels: Dict[int, np.ndarray] = {}
    for factor in factors:
        lh = -(-height // factor)
        lw = -(-width // factor)
        levels[factor] = np.empty((depth, lh, lw) if is_3d else (lh, lw),
                                  dtype=dtype)

    full = tiff.memmap(tif_path, mode="r")
    try:
        for z in range(depth):
            if should_cancel is not None and should_cancel():
                raise SetupCancelled("cancelled while building the preview")
            plane = np.asarray(full[z] if is_3d else full)
            for factor, target in levels.items():
                reduced = plane[::factor, ::factor]
                if is_3d:
                    target[z] = reduced
                else:
                    target[:] = reduced
            if progress is not None:
                progress(z + 1, depth)
    finally:
        del full

    out_dir = display_dir(tif_path)
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    for factor in factors:
        path = os.path.join(out_dir, _level_name(factor))
        partial = path + ".part"
        try:
            tiff.imwrite(
                partial, levels[factor], imagej=True,
                photometric="minisblack",
                metadata={"axes": "ZYX" if is_3d else "YX"},
            )
            os.replace(partial, path)
        except BaseException:
            try:
                if os.path.isfile(partial):
                    os.remove(partial)
            except OSError:
                pass
            raise
        written.append(path)

    manifest = _fingerprint(tif_path)
    manifest["factors"] = factors
    with open(os.path.join(out_dir, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return written


def is_current(tif_path: str) -> bool:
    """Whether a pyramid exists and was built from THIS version of the image."""
    out_dir = display_dir(tif_path)
    manifest_path = os.path.join(out_dir, MANIFEST)
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path) as fh:
            recorded = json.load(fh)
        current = _fingerprint(tif_path)
    except Exception:
        return False
    for key in ("shape", "dtype", "size", "mtime"):
        if recorded.get(key) != current.get(key):
            return False
    factors = recorded.get("factors") or []
    if not factors:
        return False
    return all(os.path.isfile(os.path.join(out_dir, _level_name(f)))
               for f in factors)


def open_levels(tif_path: str) -> Optional[List[np.ndarray]]:
    """[full_resolution, reduced...] for napari `multiscale=True`, or None.

    None means "no usable pyramid" -- absent, stale, or unreadable -- and the
    caller should show the full-resolution image alone, exactly as before. A
    preview is an optimisation; failing to find one must never stop the image
    from opening.
    """
    if not is_current(tif_path):
        return None
    out_dir = display_dir(tif_path)
    try:
        with open(os.path.join(out_dir, MANIFEST)) as fh:
            factors = json.load(fh).get("factors") or []
        arrays = [tiff.memmap(tif_path, mode="r")]
        for factor in factors:
            arrays.append(
                tiff.memmap(os.path.join(out_dir, _level_name(factor)),
                            mode="r"))
    except Exception as exc:
        print(f"  [display] could not open the preview for "
              f"{os.path.basename(tif_path)} ({exc}); using full resolution")
        return None
    # napari requires strictly decreasing sizes; a malformed set would raise
    # inside the viewer, where the failure is far less obvious than here.
    sizes = [int(np.prod(a.shape)) for a in arrays]
    if any(b >= a for a, b in zip(sizes, sizes[1:])):
        print(f"  [display] preview levels for "
              f"{os.path.basename(tif_path)} are not decreasing; ignoring")
        return None
    return arrays
