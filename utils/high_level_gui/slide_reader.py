"""
slide_reader: read whole-slide formats (VSI and friends) at full resolution.

Two problems this solves, both specific to slide scanners.

ONE FILE IS SEVERAL IMAGES
    A single .vsi from a VS-series scanner holds one scene per scanned tissue
    region -- six, on the file this was built against -- each with its own
    channels. HIBACHI's project model is one file per image, so a slide is
    addressed by a SOURCE KEY of the form ``Image.vsi::20x_01`` that names the
    file and the scene inside it. Existing single-scene formats keep working
    unchanged, because a key with no separator just means "the whole file".

FULL RESOLUTION WITHOUT LOADING IT
    The largest tested scene is 997 megapixels; one uint16 channel is 2 GB.
    ``Scene.read_block`` returns a single numpy array, so asking it for a whole
    channel means a 2 GB allocation. Extraction therefore walks the scene in
    tiles and writes each tile straight into a memmapped output TIFF, which keeps
    heap use flat in the size of the image (measured: 8 MB peak for a 96 MB
    image, versus 96 MB for the read-it-all approach). Nothing here downsamples
    by default -- ``level`` exists for callers who want a pyramid level, and
    defaults to full resolution.

NOT EVERYTHING IS SLIDEIO
    Leica .lif is served by lif_reader and Zarr / OME-Zarr by zarr_reader.
    Dispatch happens once, in ``_backend``, so every caller -- the setup wizard,
    unorganized_sources, the dimension probe, extraction -- gets each new format
    without changing, because those source keys are shaped exactly like a
    multi-scene slide key. Adding a backend means one entry in ``_BACKENDS``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .slide_formats import backend_for_path, spec_for_path, inspect_slide

# Separates a file path from a scene name inside it. Chosen because it cannot
# appear in a filename on Windows and is vanishingly rare on POSIX.
SOURCE_SEP = "::"


class SetupCancelled(Exception):
    """Raised by a long operation when the caller asks it to stop.

    Lives here rather than in project_scaffolding because slide_reader sits below
    it in the import order; both re-export it so callers can catch cancellation
    separately from a genuine failure and avoid reporting an error the user caused
    deliberately.
    """

# Tile edge in pixels. 2048 keeps a uint16 tile at 8 MB, comfortably above the
# scanner's native 512 px tiles (so we don't fight the pyramid) and far below any
# level that would matter for memory.
DEFAULT_TILE = 2048


# --------------------------------------------------------------------------- #
# Source keys
# --------------------------------------------------------------------------- #
def make_source_key(filename: str, scene_name: Optional[str] = None) -> str:
    """Combine a filename and optional scene into one source key."""
    if not scene_name:
        return filename
    return f"{filename}{SOURCE_SEP}{scene_name}"


def parse_source_key(key: str) -> Tuple[str, Optional[str]]:
    """Split a source key into (filename, scene_name or None)."""
    text = str(key)
    if SOURCE_SEP in text:
        filename, _, scene = text.partition(SOURCE_SEP)
        return filename, (scene or None)
    return text, None


def is_slide_source(key: str) -> bool:
    """True if this source should be read through slideio."""
    filename, _ = parse_source_key(key)
    return spec_for_path(filename) is not None


#: Compound suffixes that splitext leaves behind. ``Image.ome.zarr`` splits to
#: ``Image.ome``, and the extracted channel is then written as
#: ``Image.ome.tif`` -- a name that spec_for_path classifies as an OME-TIFF and
#: routes to slideio, even though the file is a plain ImageJ TIFF this code just
#: wrote with tifffile. slideio then fails to read a scale from it, the failure
#: is swallowed, and the sample records no pixel size at all despite the
#: resolution tags being correct. Stripping the inner suffix keeps the extracted
#: name unambiguous. Applies to .ome.tif sources too, which had the same latent
#: problem.
_INNER_SUFFIXES: Tuple[str, ...] = (".ome",)


def _source_stem(filename: str) -> str:
    """Basename with its extension, and any compound inner suffix, removed."""
    stem = os.path.splitext(os.path.basename(str(filename).rstrip("/\\")))[0]
    for suffix in _INNER_SUFFIXES:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)].rstrip(".")
    return stem


def folder_name_for_source(key: str) -> str:
    """Filesystem-safe folder name for a source key.

    ``Image.vsi::20x_01`` becomes ``Image_20x_01`` so the six scenes of one slide
    land in six sibling sample folders rather than colliding on one name. A Zarr
    array path is sanitised the same way, so ``store.zarr::volumes/raw`` becomes
    ``store_volumes_raw`` rather than a nested directory.
    """
    filename, scene = parse_source_key(key)
    stem = _source_stem(filename)
    if not scene:
        return stem
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in scene)
    return f"{stem}_{safe}".strip("_")


# --------------------------------------------------------------------------- #
# Backend dispatch
# --------------------------------------------------------------------------- #
# Backend dispatch
# --------------------------------------------------------------------------- #
# Everything below is slideio except the formats listed here, which slideio has
# no driver for. Dispatching once means every existing caller gets each format
# without changing, because these source keys are shaped exactly like a
# multi-scene slide key.
#
# Suffixes are matched longest-first so ".ome.zarr" cannot be shadowed by
# ".zarr" -- both route to the same module today, but a table that depends on
# dict ordering for correctness is a trap for whoever adds the next format.
_BACKENDS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    ((".lif",), "lif_reader"),
    ((".ome.zarr", ".zarr"), "zarr_reader"),
)


def _backend(source_key: str, root: str = ""):
    """The reader module serving this source key, or None for slideio.

    Returns None both for "this is a slideio format" and for "the backend module
    could not be imported", which is deliberate: the slideio path then reports a
    driver error naming the format, which is more useful than an ImportError
    traceback from a module the user never asked for by name.
    """
    filename, _ = parse_source_key(source_key)
    name = str(filename).rstrip("/\\").lower()
    for suffixes, module_name in _BACKENDS:
        if not name.endswith(suffixes):
            continue
        try:
            import importlib
            return importlib.import_module(f".{module_name}", __package__)
        except ImportError:
            return None

    # Sidecar-TIFF slides cannot be dispatched by extension: a .vsi is served
    # by slideio when its regions are ETS tile stacks and by vsi_sidecar when
    # they are TIFF pyramids, and only the file on disk says which. The test is
    # a filesystem one -- does this key's scene resolve to a readable TIFF under
    # the sidecar directory -- so an ETS scene name resolves to nothing and
    # falls through to the driver, and no name is pattern-matched.
    #
    # `root` matters here and nowhere above: extension dispatch reads the key
    # alone, but resolving a path needs the same root the accessors were given.
    # Without it a caller using `scene_shape(key, root=...)` rather than a
    # joined path silently got the slideio path and a wrong answer.
    try:
        from . import vsi_sidecar
    except ImportError:
        return None
    if vsi_sidecar.claims(source_key, root):
        return vsi_sidecar
    return None


def _lif_backend(source_key: str, root: str = ""):
    """Deprecated alias for :func:`_backend`, kept so nothing breaks silently.

    Retained because this module's dispatch was LIF-only before Zarr was added;
    any out-of-tree caller that reached for the private name still works, and
    still gets the right backend rather than None for a .zarr key.
    """
    return _backend(source_key, root)


def backend_name(source_key: str) -> Optional[str]:
    """Which library serves a key ('slideio' | 'readlif' | 'zarr' | None).

    None means HIBACHI has no format registered for that path at all, which is
    different from "slideio cannot open it".
    """
    filename, _ = parse_source_key(source_key)
    return backend_for_path(filename)


def list_sources(path: str) -> List[str]:
    """Source keys for every usable image in a slide file.

    Returns one key per tissue scene. A single-scene slide yields a bare filename
    so it behaves exactly like the existing formats. Returns [] if the file can't
    be read, and says why: the reason was previously computed by
    ``inspect_slide`` and thrown away, so a caller could only report "no
    readable scenes" for a file whose actual problem was already known.
    """
    backend = _backend(path)
    if backend is not None:
        return backend.list_sources(path)

    info = inspect_slide(path)
    scenes = info.tissue_scenes
    if info.error or not scenes:
        # A VS-series slide whose regions were written as TIFF pyramids rather
        # than ETS tile stacks: slideio's VSI driver opens it and reports no
        # scenes at all, so the images are readable but invisible to the driver.
        try:
            from . import vsi_sidecar
            fallback = vsi_sidecar.list_sources(path)
        except ImportError:
            fallback = []
        if fallback:
            print(f"  [slide] {os.path.basename(path)}: the "
                  f"{info.driver or '?'} driver reported no usable scenes; "
                  f"reading {len(fallback)} region(s) from its sidecar "
                  "TIFF data instead")
            return fallback
        if info.error:
            print(f"  [slide] {os.path.basename(path)}: {info.error}")
        return []

    name = os.path.basename(path)
    if len(scenes) <= 1:
        return [name]
    return [make_source_key(name, s.name) for s in scenes]


# --------------------------------------------------------------------------- #
# Opening a scene
# --------------------------------------------------------------------------- #
class _SceneHandle:
    """Context manager yielding a slideio Scene, closing the slide afterwards."""

    def __init__(self, path: str, scene_name: Optional[str]):
        self.path = path
        self.scene_name = scene_name
        self._slide = None

    def __enter__(self):
        import slideio
        spec = spec_for_path(self.path)
        if spec is None:
            raise ValueError(f"not a slide format HIBACHI reads: {self.path}")
        self._slide = slideio.open_slide(self.path, spec.driver)

        if self.scene_name:
            # Prefer name lookup, but fall back to scanning: not every driver
            # guarantees get_scene_by_name.
            try:
                return self._slide.get_scene_by_name(self.scene_name)
            except Exception:
                for i in range(self._slide.num_scenes):
                    sc = self._slide.get_scene(i)
                    if str(getattr(sc, "name", "")) == self.scene_name:
                        return sc
                raise ValueError(
                    f"scene {self.scene_name!r} not found in "
                    f"{os.path.basename(self.path)}")

        # No scene named: use the first scene the format layer considers tissue,
        # which for SVS-like formats is emphatically not just "scene 0 exists".
        info = inspect_slide(self.path)
        tissue = info.tissue_scenes
        if not tissue:
            raise ValueError(info.error or "no usable scene in this slide")
        return self._slide.get_scene(tissue[0].index)

    def __exit__(self, *exc):
        if self._slide is not None:
            try:
                self._slide.close()
            except Exception:
                pass
        return False


def open_scene(source_key: str, root: str = "") -> _SceneHandle:
    """Context manager for the scene a source key refers to."""
    filename, scene = parse_source_key(source_key)
    path = os.path.join(root, filename) if root else filename
    return _SceneHandle(path, scene)


# --------------------------------------------------------------------------- #
# Metadata, in the shape MetadataExtractor uses
# --------------------------------------------------------------------------- #
def scene_channel_count(source_key: str, root: str = "") -> int:
    """Channels in the scene a source key names, or 1 if it can't be read."""
    backend = _backend(source_key, root)
    if backend is not None:
        return backend.scene_channel_count(source_key, root)
    try:
        with open_scene(source_key, root) as scene:
            return int(scene.num_channels)
    except Exception as exc:
        print(f"    Could not read channel count from {source_key}: {exc}")
        return 1


def scene_metadata(source_key: str, root: str = "") -> Dict[str, Any]:
    """Physical scale of a scene, as {'x','y','z','found'} in MICRONS.

    Matches ``MetadataExtractor.read_tiff_metadata``'s contract so callers can
    treat slides and TIFFs alike. slideio reports metres per pixel, hence the
    1e6 conversion. A slide with no Z calibration reports z=0, which would become
    a zero voxel dimension downstream, so it falls back to 1.0 and says so.
    """
    backend = _backend(source_key, root)
    if backend is not None:
        return backend.scene_metadata(source_key, root)

    meta: Dict[str, Any] = {"x": 1.0, "y": 1.0, "z": 1.0, "found": False}
    try:
        with open_scene(source_key, root) as scene:
            try:
                rx, ry = scene.resolution
                um_x, um_y = float(rx) * 1e6, float(ry) * 1e6
            except Exception:
                um_x = um_y = 0.0
            try:
                um_z = float(scene.z_resolution) * 1e6
            except Exception:
                um_z = 0.0

            if um_x > 0 and um_y > 0:
                meta.update({"x": um_x, "y": um_y, "found": True})
            # A single-slice scene has no meaningful Z spacing; 1.0 keeps the
            # recorded depth equal to the slice count instead of zero.
            meta["z"] = um_z if um_z > 0 else 1.0
    except Exception as exc:
        print(f"    Could not read scale from {source_key}: {exc}")
    return meta


def scene_shape(source_key: str, root: str = "") -> Optional[Tuple[int, ...]]:
    """(Z, Y, X) or (Y, X) pixel shape of a scene, without reading pixels."""
    backend = _backend(source_key, root)
    if backend is not None:
        return backend.scene_shape(source_key, root)
    try:
        with open_scene(source_key, root) as scene:
            w, h = (int(v) for v in scene.size)
            z = int(getattr(scene, "num_z_slices", 1) or 1)
            return (z, h, w) if z > 1 else (h, w)
    except Exception:
        return None


def scene_channel_names(source_key: str, root: str = "") -> List[str]:
    """Channel names as the scanner recorded them, e.g. ['DAPI','FITC','Cy5']."""
    backend = _backend(source_key, root)
    if backend is not None:
        return backend.scene_channel_names(source_key, root)
    try:
        with open_scene(source_key, root) as scene:
            out = []
            for c in range(int(scene.num_channels)):
                try:
                    out.append(str(scene.get_channel_name(c)))
                except Exception:
                    out.append(f"ch{c}")
            return out
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _level_size(scene, level: int) -> Tuple[int, int]:
    """(width, height) of a pyramid level. Level 0 is full resolution."""
    w, h = (int(v) for v in scene.size)
    if level <= 0:
        return w, h
    factor = 2 ** level
    return max(1, w // factor), max(1, h // factor)


def extract_scene_channel(
    source_key: str,
    dest_path: str,
    channel_idx: int,
    root: str = "",
    tile: int = DEFAULT_TILE,
    level: int = 0,
    progress=None,
    should_cancel=None,
) -> bool:
    """Write one channel of one scene to a TIFF, tile by tile.

    Full resolution by default. Peak memory is one tile, because the output is a
    memmapped TIFF filled in place rather than an array assembled in RAM and
    handed to imwrite -- which for the tested slide would have meant a 2 GB
    allocation per channel.

    `progress` is called as progress(done_tiles, total_tiles) if given, and
    `should_cancel` is polled between tiles -- the tile loop is the only place a
    multi-minute extraction can be interrupted, since a single read_block cannot
    be broken into.
    Returns True only if a non-empty file was produced.
    """
    backend = _backend(source_key, root)
    if backend is not None:
        return backend.extract_scene_channel(
            source_key, dest_path, channel_idx, root=root, tile=tile,
            level=level, progress=progress, should_cancel=should_cancel)

    import tifffile as tiff

    with open_scene(source_key, root) as scene:
        n_ch = int(scene.num_channels)
        if not (0 <= channel_idx < n_ch):
            raise ValueError(
                f"channel {channel_idx} requested but this scene has {n_ch}")

        out_w, out_h = _level_size(scene, level)
        src_w, src_h = (int(v) for v in scene.size)
        z_slices = int(getattr(scene, "num_z_slices", 1) or 1)

        try:
            dtype = np.dtype(scene.get_channel_data_type(channel_idx))
        except Exception:
            dtype = np.dtype(np.uint16)

        meta = scene_metadata(source_key, root)
        shape = (z_slices, out_h, out_w) if z_slices > 1 else (out_h, out_w)

        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        # imagej=True is required: without it tifffile writes ResolutionUnit=INCH
        # and the micron scale reads back roughly 25400x too large.
        mm = tiff.memmap(
            dest_path, shape=shape, dtype=dtype, imagej=True,
            resolution=(1.0 / meta["x"] if meta["x"] > 0 else 1.0,
                        1.0 / meta["y"] if meta["y"] > 0 else 1.0),
            metadata={"unit": "micron", "spacing": meta["z"]},
        )
        try:
            scale = out_w / src_w if src_w else 1.0
            n_tiles = (((out_h + tile - 1) // tile)
                       * ((out_w + tile - 1) // tile) * max(1, z_slices))
            done = 0

            for z in range(max(1, z_slices)):
                for oy in range(0, out_h, tile):
                    oh = min(tile, out_h - oy)
                    for ox in range(0, out_w, tile):
                        ow = min(tile, out_w - ox)

                        # rect is in SOURCE pixels and ordered (x, y, w, h);
                        # size is the requested OUTPUT size of the block.
                        sx = int(round(ox / scale)) if scale else ox
                        sy = int(round(oy / scale)) if scale else oy
                        sw = min(int(round(ow / scale)) if scale else ow,
                                 src_w - sx)
                        sh = min(int(round(oh / scale)) if scale else oh,
                                 src_h - sy)
                        if sw <= 0 or sh <= 0:
                            continue

                        kwargs: Dict[str, Any] = {
                            "rect": (sx, sy, sw, sh),
                            "size": (ow, oh),
                            "channel_indices": [channel_idx],
                        }
                        if z_slices > 1:
                            kwargs["slices"] = (z, z + 1)

                        block = scene.read_block(**kwargs)
                        block = np.squeeze(np.asarray(block))
                        if block.ndim != 2:
                            block = block.reshape(oh, ow)

                        if z_slices > 1:
                            mm[z, oy:oy + oh, ox:ox + ow] = block[:oh, :ow]
                        else:
                            mm[oy:oy + oh, ox:ox + ow] = block[:oh, :ow]

                        done += 1
                        if progress is not None:
                            progress(done, n_tiles)
                        if should_cancel is not None and should_cancel():
                            raise SetupCancelled(
                                "extraction cancelled by the user")
            mm.flush()
        except SetupCancelled:
            # A half-written channel would pass an existence check and be
            # organized as if complete, so remove it.
            del mm
            try:
                if os.path.isfile(dest_path):
                    os.remove(dest_path)
            except OSError:
                pass
            raise
        finally:
            try:
                del mm
            except Exception:
                pass

    ok = os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0
    if not ok:
        print(f"    Extraction produced no data at {dest_path}")
    return ok