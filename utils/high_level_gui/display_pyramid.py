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


class Accumulator:
    """Collects reduced levels one plane at a time, then writes them.

    Exists so the pyramid can be produced from planes that are ALREADY being
    read, rather than by reading the image again. Extraction streams every
    plane through a buffer on its way to disk; feeding those planes here costs
    a strided copy each and no extra I/O, which on a drive that reads at tens
    of MB/s is the difference between free and five minutes per channel.

    `build` below uses the same class, so there is one implementation of what a
    level is and how it is written, whichever end it is driven from.

    Holds every level in memory while filling -- together they are a few
    percent of the image, ~2 GB for a 24 GB channel -- and writes sequentially
    at the end rather than memory-mapping writes onto a FUSE-mounted drive.
    """

    def __init__(self, shape: Tuple[int, ...], dtype) -> None:
        self.shape = tuple(int(n) for n in shape)
        self.dtype = np.dtype(dtype)
        self.factors = factors_for(self.shape)
        self.is_3d = len(self.shape) == 3
        self.depth = int(self.shape[0]) if self.is_3d else 1
        height, width = int(self.shape[-2]), int(self.shape[-1])
        self.levels: Dict[int, np.ndarray] = {}
        for factor in self.factors:
            lh = -(-height // factor)
            lw = -(-width // factor)
            self.levels[factor] = np.empty(
                (self.depth, lh, lw) if self.is_3d else (lh, lw),
                dtype=self.dtype)

    def wanted(self) -> bool:
        """Whether this image is large enough for a pyramid to be worth it."""
        return bool(self.factors)

    def add_plane(self, z_index: int, plane) -> None:
        """Reduce one full-resolution plane into every level.

        Strided decimation, not averaging. Microglial processes are one or two
        pixels wide and bright against near-zero background; a box mean dims
        them toward the background and they vanish from the overview, while
        taking every Nth pixel keeps their intensity where it lands. This is a
        preview, so aliasing is acceptable and losing the processes is not.
        """
        for factor, target in self.levels.items():
            reduced = plane[::factor, ::factor]
            if self.is_3d:
                target[z_index] = reduced
            else:
                target[:] = reduced

    def write(self, tif_path: str) -> List[str]:
        """Write the levels beside `tif_path` and record the manifest."""
        if not self.factors:
            return []
        out_dir = display_dir(tif_path)
        os.makedirs(out_dir, exist_ok=True)
        written: List[str] = []
        for factor in self.factors:
            path = os.path.join(out_dir, _level_name(factor))
            partial = path + ".part"
            try:
                tiff.imwrite(
                    partial, self.levels[factor], imagej=True,
                    photometric="minisblack",
                    metadata={"axes": "ZYX" if self.is_3d else "YX"},
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

        # The manifest is written LAST and is what `is_current` keys on, so an
        # interrupted build leaves no manifest and the pyramid is ignored
        # rather than half-used.
        manifest = _fingerprint(tif_path)
        manifest["factors"] = self.factors
        with open(os.path.join(out_dir, MANIFEST), "w") as fh:
            json.dump(manifest, fh, indent=2)
        return written


def build(tif_path: str, progress=None, should_cancel=None) -> List[str]:
    """Build the display pyramid for an existing image. Returns paths.

    Reads the image ONCE, plane by plane, feeding each plane to every level --
    so the cost is one pass over the file, not one per level. For a channel
    already on disk this is the only option; new extractions fill an
    `Accumulator` as they write and pay nothing.

    `progress` is called as progress(done_planes, total_planes).
    """
    from .slide_reader import SetupCancelled

    with tiff.TiffFile(tif_path) as handle:
        series = handle.series[0]
        shape = tuple(int(n) for n in series.shape)
        dtype = np.dtype(str(series.dtype))

    accumulator = Accumulator(shape, dtype)
    if not accumulator.wanted():
        return []

    full = tiff.memmap(tif_path, mode="r")
    try:
        for z in range(accumulator.depth):
            if should_cancel is not None and should_cancel():
                raise SetupCancelled("cancelled while building the preview")
            plane = np.asarray(full[z] if accumulator.is_3d else full)
            accumulator.add_plane(z, plane)
            if progress is not None:
                progress(z + 1, accumulator.depth)
    finally:
        del full

    return accumulator.write(tif_path)


def needs_preview(tif_path: str) -> bool:
    """Whether this image wants a preview and does not have a current one."""
    try:
        if is_current(tif_path):
            return False
        with tiff.TiffFile(tif_path) as handle:
            shape = tuple(int(n) for n in handle.series[0].shape)
        return bool(factors_for(shape))
    except Exception:
        return False


def _plane_count(tif_path: str) -> int:
    try:
        with tiff.TiffFile(tif_path) as handle:
            shape = tuple(int(n) for n in handle.series[0].shape)
        return int(shape[0]) if len(shape) == 3 else 1
    except Exception:
        return 1


def build_with_progress(tif_paths, parent=None) -> None:
    """Build any missing previews, showing a modal progress window first.

    Called BEFORE a viewer is created, and blocking on purpose. The
    alternative -- building in the background while the image opens at full
    resolution -- read well but behaved badly: napari builds a thumbnail and
    downscales a 928-megapixel plane on the main thread to get under the GPU's
    texture limit, so the window froze for minutes per channel exactly as it
    did before, while the background build competed with it for a drive that
    collapses to seek-bound speed under concurrent readers. Waiting visibly is
    both faster and honest.

    Builds are sequential for the same reason: one reader at a time is far
    quicker on a spinning disk than several interleaved.

    Cancelling stops after the current image and opens the rest at full
    resolution, which is slow but correct -- the viewer never depends on a
    preview existing.

    Qt is imported here rather than at module scope so extraction, batch runs
    and tests can use the rest of this module headless.
    """
    pending = [p for p in dict.fromkeys(tif_paths) if p and needs_preview(p)]
    if not pending:
        return

    try:
        from PyQt5.QtCore import Qt  # type: ignore
        from PyQt5.QtWidgets import (  # type: ignore
            QApplication, QProgressDialog,
        )
    except Exception:
        for path in pending:
            build(path)
        return

    totals = {p: max(1, _plane_count(p)) for p in pending}
    grand_total = sum(totals.values())
    done_before = 0

    dialog = QProgressDialog(
        "Preparing a fast preview for this sample…", "Skip", 0, grand_total,
        parent)
    dialog.setWindowTitle("Preparing preview")
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setValue(0)
    QApplication.processEvents()

    try:
        for index, path in enumerate(pending, start=1):
            name = os.path.basename(os.path.dirname(os.path.abspath(path)))
            base = done_before

            def _tick(done: int, total: int, _n=name, _i=index, _b=base) -> None:
                dialog.setLabelText(
                    f"Preparing a fast preview ({_i} of {len(pending)})\n"
                    f"{_n} — plane {done} of {total}"
                )
                dialog.setValue(_b + done)
                # The build runs on this thread, so the dialog only repaints
                # and only notices the Skip button if we let Qt run.
                QApplication.processEvents()

            try:
                build(path, progress=_tick,
                      should_cancel=lambda: dialog.wasCanceled())
            except Exception as exc:
                # A preview is an optimisation; losing one must not stop the
                # sample from opening.
                print(f"  [display] could not build a preview for "
                      f"{os.path.basename(path)} ({exc})")
                if dialog.wasCanceled():
                    break
            done_before += totals[path]
            dialog.setValue(done_before)
            QApplication.processEvents()
            if dialog.wasCanceled():
                break
    finally:
        dialog.close()
        QApplication.processEvents()


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


def contrast_limits_for(data) -> Optional[Tuple[float, float]]:
    """Display range for an image, computed from the SMALLEST array available.

    napari picks contrast limits itself when none are given, and to do that it
    has to scan pixel data: for a plain array it samples whole planes, which on
    a 928-megapixel plane over a slow drive is most of the time it takes to
    open an image, and for a multiscale layer it reads a different array than
    the non-multiscale path did -- which is why the slider handles land
    somewhere new the moment a pyramid appears.

    Passing limits explicitly removes both effects. `data` may be a level list
    (the coarsest level is used, ~23 MB) or a single array (a strided sample of
    its middle plane). Percentiles rather than min/max: one hot pixel or a
    saturated speck would otherwise stretch the range and render everything
    else black.

    Returns None if nothing can be read, in which case the caller should let
    napari do what it did before rather than guess.
    """
    try:
        array = data[-1] if isinstance(data, (list, tuple)) else data
        sample = np.asarray(array)
        if sample.ndim == 3:
            sample = sample[sample.shape[0] // 2]
        # Cap the sample regardless of where it came from: without a pyramid
        # this is a full plane, and reading all of one is what we are avoiding.
        stride = max(1, int(np.ceil(np.sqrt(sample.size / 4_000_000))))
        sample = sample[::stride, ::stride]
        low = float(np.percentile(sample, 1.0))
        high = float(np.percentile(sample, 99.8))
    except Exception:
        return None
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    return (low, high)


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
