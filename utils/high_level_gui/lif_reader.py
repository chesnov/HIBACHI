"""
lif_reader: Leica LAS X `.lif` support, behind the same interface as slide_reader.

A `.lif` is a container: one file holds many acquisitions ("images" in Leica's
terms), each with its own dimensions, channel count and calibration. That is the
same shape as a whole-slide file holding several scanned scenes, so LIF reuses
slide_reader's ``file.lif::image`` source-key model and every caller that already
handles slides works unchanged.

Why a separate backend rather than another entry in FORMATS' slideio table:
slideio has no LIF driver (its drivers are AFI, CZI, DCM, GDAL, NDPI, OMETIFF,
PHTIFF, QPTIFF, SCN, SVS, VSI, ZVI). Declaring LIF there would match on
extension and then fail when the driver was requested -- after the setup wizard
had already started. Reading is done by ``readlif`` instead, which is pure Python
and needs no JVM, matching the constraint the slide layer was built under.

Three things about the format drove this code, all verified against readlif's
source rather than assumed:

1. **readlif reports scale as PIXELS PER MICRON**, computed as
   ``(n_pixels - 1) / length_in_um``. That is the reciprocal of what slideio
   gives (metres per pixel) and of what HIBACHI stores. Getting this backwards
   would not raise -- it would silently produce dimensions wrong by the square of
   the pixel size, which is precisely the class of error the dimension-provenance
   work exists to prevent. Hence ``_spacing_um_per_px`` and its test.

2. **Scale can legitimately be absent.** readlif sets it to ``None`` when the
   Leica XML has no Length attribute. Absent scale is reported as not-found so it
   flows into the manual dimension prompt, rather than defaulting to 1.0 and
   pretending the image is calibrated.

3. **A LIF acquisition can be a mosaic or a time series.** readlif exposes mosaic
   tiles individually -- it does not stitch them -- and HIBACHI has no time axis.
   Both are refused with a specific reason instead of silently importing tile 0
   or timepoint 0, which would look successful and be wrong.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

EXTENSION = ".lif"
SOURCE_SEP = "::"


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def is_available() -> bool:
    """True if the reader dependency is installed."""
    try:
        import readlif  # noqa: F401
        return True
    except ImportError:
        return False


def missing_dependency_message() -> str:
    return (
        "Leica .lif support needs the 'readlif' package, which is not installed.\n\n"
        "Install it with:  pip install 'readlif>=0.6.5'\n\n"
        "It is pure Python and needs no Java."
    )


# --------------------------------------------------------------------------- #
# Scale
# --------------------------------------------------------------------------- #
def _spacing_um_per_px(scale_value) -> Optional[float]:
    """Convert readlif's px/um scale to HIBACHI's um/px spacing.

    readlif computes ``scale = (n_pixels - 1) / length_um``, so the spacing
    between adjacent pixels is its reciprocal. Returns None when the value is
    missing or unusable, which is reported as "no calibration" rather than
    silently replaced with 1.0.
    """
    if scale_value is None:
        return None
    try:
        scale = float(scale_value)
    except (TypeError, ValueError):
        return None
    if scale != scale or scale <= 0:          # NaN or non-positive
        return None
    spacing = 1.0 / scale
    if spacing != spacing or spacing <= 0:
        return None
    return spacing


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
@dataclass
class LifImageInfo:
    """One acquisition inside a .lif file."""
    index: int
    name: str
    width: int
    height: int
    z_slices: int
    channels: int
    timepoints: int = 1
    mosaic_tiles: int = 1
    channel_names: List[str] = field(default_factory=list)
    bit_depth: int = 16
    # Per-pixel spacing in microns; None where the file carries no calibration.
    um_x: Optional[float] = None
    um_y: Optional[float] = None
    um_z: Optional[float] = None
    usable: bool = True
    excluded_reason: str = ""

    @property
    def is_3d(self) -> bool:
        return self.z_slices > 1


@dataclass
class LifFileInfo:
    path: str
    images: List[LifImageInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def usable_images(self) -> List[LifImageInfo]:
        return [im for im in self.images if im.usable]


def _describe_image(idx: int, image: Any) -> LifImageInfo:
    """Turn a readlif LifImage into a LifImageInfo, deciding if it is usable."""
    dims = image.dims                     # named tuple (x, y, z, t, m)
    width = int(getattr(dims, "x", 1) or 1)
    height = int(getattr(dims, "y", 1) or 1)
    z_slices = int(getattr(dims, "z", 1) or 1)
    timepoints = int(getattr(dims, "t", 1) or 1)
    tiles = int(getattr(dims, "m", 1) or 1)
    channels = int(getattr(image, "channels", 1) or 1)

    scale = tuple(getattr(image, "scale", ()) or ())
    um_x = _spacing_um_per_px(scale[0] if len(scale) > 0 else None)
    um_y = _spacing_um_per_px(scale[1] if len(scale) > 1 else None)
    um_z = _spacing_um_per_px(scale[2] if len(scale) > 2 else None)

    depth = 16
    try:
        bits = getattr(image, "bit_depth", None)
        if bits:
            depth = int(max(bits))
    except Exception:
        pass

    names: List[str] = []
    try:
        # LIF channel names live in the XML; readlif does not surface them as a
        # list, so fall back to positional names. Callers only use these as
        # labels, and a wrong guess would be more confusing than "ch0".
        for c in range(channels):
            names.append(f"ch{c}")
    except Exception:
        names = []

    info = LifImageInfo(
        index=idx, name=str(getattr(image, "name", f"image_{idx}")),
        width=width, height=height, z_slices=z_slices, channels=channels,
        timepoints=timepoints, mosaic_tiles=tiles, channel_names=names,
        bit_depth=depth, um_x=um_x, um_y=um_y, um_z=um_z,
    )

    # Refuse rather than import something misleading.
    if tiles > 1:
        info.usable = False
        info.excluded_reason = (
            f"mosaic / tile scan ({tiles} tiles). The tiles are stored "
            "separately and are not stitched by the reader, so importing this "
            "would give one tile, not the assembled field. Stitch it in LAS X "
            "and export the result."
        )
    elif timepoints > 1:
        info.usable = False
        info.excluded_reason = (
            f"time series ({timepoints} timepoints). HIBACHI analyses one image "
            "per sample and has no time axis, so importing this would silently "
            "keep only the first frame. Export the timepoints separately."
        )
    elif width <= 1 or height <= 1:
        info.usable = False
        info.excluded_reason = (
            f"not a 2D image ({width}x{height}); most likely a line scan or "
            "single-point measurement."
        )
    return info


def inspect_lif(path: str) -> LifFileInfo:
    """Describe every acquisition in a .lif file. Reads no pixel data."""
    out = LifFileInfo(path=path)

    if not is_available():
        out.error = missing_dependency_message()
        return out
    if not os.path.isfile(path):
        out.error = f"file not found: {path}"
        return out

    try:
        from readlif.reader import LifFile
        handle = LifFile(path)
    except Exception as exc:
        out.error = (
            f"{os.path.basename(path)} could not be opened as a Leica .lif "
            f"file ({exc}). If it was exported from a newer LAS X, try updating "
            "readlif, or export the images as OME-TIFF."
        )
        return out

    try:
        for idx, image in enumerate(handle.get_iter_image()):
            try:
                out.images.append(_describe_image(idx, image))
            except Exception as exc:
                out.images.append(LifImageInfo(
                    index=idx, name=f"image_{idx}", width=0, height=0,
                    z_slices=0, channels=0, usable=False,
                    excluded_reason=f"could not be described ({exc})",
                ))
    except Exception as exc:
        out.error = f"could not list the images in {os.path.basename(path)}: {exc}"
        return out

    if not out.images:
        out.error = f"{os.path.basename(path)} contains no images."
        return out

    skipped = [im for im in out.images if not im.usable]
    if skipped:
        out.warnings.append(
            f"{len(skipped)} of {len(out.images)} acquisition(s) skipped: "
            + "; ".join(f"{im.name} ({im.excluded_reason.split('.')[0]})"
                        for im in skipped[:4])
        )

    # A LIF mixing 2D and 3D acquisitions is normal (an overview snap beside a
    # stack). Worth saying, because the two need different processing modes and
    # the wizard asks for one preset per channel.
    kinds = {im.is_3d for im in out.usable_images}
    if len(kinds) > 1:
        out.warnings.append(
            "this file mixes 2D and 3D acquisitions; they need different "
            "processing modes, so set them up as separate projects"
        )
    return out


# --------------------------------------------------------------------------- #
# Source keys  (identical semantics to slide_reader)
# --------------------------------------------------------------------------- #
def _parse(key: str) -> Tuple[str, Optional[str]]:
    text = str(key)
    if SOURCE_SEP in text:
        filename, _, scene = text.partition(SOURCE_SEP)
        return filename, (scene or None)
    return text, None


def is_lif_source(key: str) -> bool:
    """True if this source key refers to a .lif file."""
    filename, _ = _parse(key)
    return str(filename).lower().endswith(EXTENSION)


def list_sources(path: str) -> List[str]:
    """One source key per usable acquisition in a .lif.

    A file with a single usable acquisition yields a bare filename, so it behaves
    exactly like a plain TIFF downstream. Returns [] when nothing is readable,
    leaving the caller to report the reason from ``inspect_lif``.
    """
    info = inspect_lif(path)
    if info.error:
        return []
    usable = info.usable_images
    name = os.path.basename(path)
    if not usable:
        return []
    if len(usable) == 1:
        return [name]
    return [f"{name}{SOURCE_SEP}{im.name}" for im in usable]


def _resolve(source_key: str, root: str = "") -> Tuple[str, LifImageInfo]:
    """(path, image info) for a source key, or raise with a usable message."""
    filename, wanted = _parse(source_key)
    path = os.path.join(root, filename) if root else filename
    info = inspect_lif(path)
    if info.error:
        raise ValueError(info.error)

    usable = info.usable_images
    if wanted:
        for im in usable:
            if im.name == wanted:
                return path, im
        # Named but unusable: say WHY rather than "not found".
        for im in info.images:
            if im.name == wanted:
                raise ValueError(
                    f"acquisition {wanted!r} cannot be imported: "
                    f"{im.excluded_reason}")
        raise ValueError(
            f"acquisition {wanted!r} not found in {os.path.basename(path)}")

    if not usable:
        raise ValueError(
            f"no importable acquisition in {os.path.basename(path)}")
    return path, usable[0]


def _open_image(path: str, index: int):
    from readlif.reader import LifFile
    return LifFile(path).get_image(index)


# --------------------------------------------------------------------------- #
# Metadata, in the shape MetadataExtractor uses
# --------------------------------------------------------------------------- #
def scene_channel_count(source_key: str, root: str = "") -> int:
    """Channels in the acquisition a source key names, or 1 if unreadable."""
    try:
        _path, im = _resolve(source_key, root)
        return int(im.channels)
    except Exception as exc:
        print(f"    Could not read channel count from {source_key}: {exc}")
        return 1


def scene_metadata(source_key: str, root: str = "") -> Dict[str, Any]:
    """Per-pixel spacing as {'x','y','z','found'} in MICRONS.

    Same contract as ``MetadataExtractor.read_tiff_metadata`` and
    ``slide_reader.scene_metadata``, so callers treat LIF, slides and TIFFs
    alike.

    ``found`` is True only when X and Y are genuinely calibrated. An uncalibrated
    file reports found=False and unit spacings, which routes it into the manual
    dimension prompt instead of being recorded as a calibrated 1 um/pixel image.
    """
    meta: Dict[str, Any] = {"x": 1.0, "y": 1.0, "z": 1.0, "found": False}
    try:
        _path, im = _resolve(source_key, root)
    except Exception as exc:
        print(f"    Could not read scale from {source_key}: {exc}")
        return meta

    if im.um_x and im.um_y:
        meta.update({"x": im.um_x, "y": im.um_y, "found": True})
    # Z spacing is only meaningful for a stack. A single plane keeps 1.0 so the
    # recorded depth equals the slice count rather than zero.
    if im.z_slices > 1 and im.um_z:
        meta["z"] = im.um_z
    return meta


def scene_shape(source_key: str, root: str = "") -> Optional[Tuple[int, ...]]:
    """(Z, Y, X) or (Y, X) pixel shape, without reading pixels."""
    try:
        _path, im = _resolve(source_key, root)
    except Exception:
        return None
    if im.z_slices > 1:
        return (im.z_slices, im.height, im.width)
    return (im.height, im.width)


def scene_channel_names(source_key: str, root: str = "") -> List[str]:
    """Channel labels. LIF does not reliably name them, so these are positional."""
    try:
        _path, im = _resolve(source_key, root)
        return list(im.channel_names)
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_scene_channel(
    source_key: str,
    dest_path: str,
    channel_idx: int,
    root: str = "",
    tile: int = 0,            # accepted for interface parity; LIF reads by plane
    level: int = 0,           # LIF has no pyramid; only level 0 is meaningful
    progress=None,
    should_cancel=None,
) -> bool:
    """Write one channel of one acquisition to a TIFF, one Z plane at a time.

    Peak memory is a single plane: the output is a memmapped TIFF filled in
    place, matching how the slide path avoids assembling a whole volume in RAM.

    `progress` is called as progress(done, total) and `should_cancel` is polled
    between planes -- a plane read cannot be interrupted, so that is the finest
    granularity a cancel can act on.

    Returns True only if a non-empty file was produced.
    """
    import tifffile as tiff

    path, im = _resolve(source_key, root)
    if not (0 <= channel_idx < im.channels):
        raise ValueError(
            f"channel {channel_idx} requested but this acquisition has "
            f"{im.channels}")
    if level != 0:
        # Said out loud rather than ignored: a caller asking for a lower
        # resolution would otherwise silently receive full resolution.
        print(f"    [lif] {os.path.basename(path)} has no pyramid; "
              f"ignoring level={level} and reading full resolution.")

    image = _open_image(path, im.index)
    dtype = np.uint8 if im.bit_depth <= 8 else np.uint16
    shape = ((im.z_slices, im.height, im.width) if im.z_slices > 1
             else (im.height, im.width))

    meta = scene_metadata(source_key, root)
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)

    # imagej=True is required: without it tifffile writes ResolutionUnit=INCH and
    # the micron scale reads back roughly 25400x too large.
    mm = tiff.memmap(
        dest_path, shape=shape, dtype=dtype, imagej=True,
        resolution=(1.0 / meta["x"] if meta["x"] > 0 else 1.0,
                    1.0 / meta["y"] if meta["y"] > 0 else 1.0),
        metadata={"unit": "micron", "spacing": meta["z"]},
    )
    from .slide_reader import SetupCancelled

    try:
        total = max(1, im.z_slices)
        for z in range(total):
            if should_cancel is not None and should_cancel():
                raise SetupCancelled("cancelled during .lif extraction")

            frame = np.asarray(image.get_frame(z=z, t=0, c=channel_idx, m=0))
            if frame.ndim != 2:
                frame = frame.reshape(im.height, im.width)

            if im.z_slices > 1:
                mm[z] = frame[:im.height, :im.width]
            else:
                mm[:] = frame[:im.height, :im.width]

            if progress is not None:
                progress(z + 1, total)
        mm.flush()
    except SetupCancelled:
        # A half-written channel would pass an existence check and be organized
        # as if complete, so remove it. The slideio path in
        # slide_reader.extract_scene_channel has always done this; this path did
        # not, so a cancelled .lif setup left a folder holding a TIFF and no
        # config -- which fails the one-tif-one-yaml check and makes the whole
        # project read as empty, with a truncated image sitting inside it.
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