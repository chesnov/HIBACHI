"""vsi_sidecar: a VS-series slide whose pixel data is sidecar TIFF pyramids.

Why this exists
---------------
A .vsi is a small header; its pixels live in a sibling ``_<stem>_/`` directory.
On the export slideio was verified against, each scanned region is an ETS tile
stack and slideio's VSI driver reports one scene per region.

On the export this module was written for, the same scanner wrote OME-TIFF
pyramids instead -- ``_<stem>_/stackNNNNN/frame_t_0.tif``, one directory per
region -- and slideio's VSI driver reports ZERO scenes. It opens the file
without error and finds nothing, so ``inspect_slide`` returns "the reader
opened the file but reported no scenes" and every image in the slide is
invisible to the wizard.

slideio can be made to read the TIFF by naming its GDAL driver explicitly, but
what comes back is 364 unnamed single-channel 2D scenes: every pyramid level
and every (channel, z) plane flattened into a separate "scene", with the CZYX
structure discarded. Reconstructing axes from that means inferring them from
scene dimensions and ordering, which is the positional guessing this codebase
has been removing. ``tifffile`` reports ``axes='CZYX'`` from the header
directly, so this backend reads the sidecar TIFFs itself.

Extraction delegates to ``MetadataExtractor._stream_tiff_channel``, which
already writes one channel of exactly this kind of file a plane at a time. The
only work here is resolving which TIFF a region means and answering the scene
protocol from its header.

The slide label and macro overview live in the same sidecar, as interleaved RGB
brightfield with ``axes='YXS'`` and OME image names 'Label' and 'Overview'.
They are excluded by ``tiff_reject_reason`` -- the same predicate detection
applies to loose TIFFs, so a file refused in one place cannot be accepted in
another.

Scene naming
------------
A region's scene name is its directory name, ``stack10002``, so a sample folder
comes out as ``20260901_stack10002`` and what appears in the project matches
what is on disk. Unlike the slideio path, a key is emitted even when the slide
holds only one region: the ``::scene`` suffix is what lets ``_backend`` route a
key here rather than to slideio, so a bare filename would dispatch back to the
driver that cannot read it.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

#: Suffix of a source key's scene part is a directory under the sidecar; no
#: pattern is matched against it. Resolution is a filesystem test -- the
#: directory either exists with a readable TIFF in it or this backend does not
#: claim the key.
_TIFF_EXTS = ('.tif', '.tiff')


def _stem(filename: str) -> str:
    return os.path.splitext(os.path.basename(str(filename).rstrip('/\\')))[0]


def sidecar_dir(path: str) -> Optional[str]:
    """The ``_<stem>_`` directory beside `path`, or None.

    Matched EXACTLY, not by substring. ``sidecar_status`` in slide_formats tests
    ``stem in item``, so ``20260901.vsi`` claims ``_20260901_01_`` and
    ``_20260901_02_`` as well -- its siblings' data -- and reports their bytes
    as its own. A slide would then look like it had its pixels because the slide
    next to it did.
    """
    folder = os.path.dirname(os.path.abspath(path))
    candidate = os.path.join(folder, f"_{_stem(path)}_")
    return candidate if os.path.isdir(candidate) else None


def _region_tiff(stack_dir: str) -> Optional[str]:
    """The TIFF holding a region's pixels, or None if there isn't exactly one.

    A region directory in the tested export holds one TIFF (plus, for the
    overview, an .ets blob and a .meta annotation set that carry no image data
    this pipeline can use). Several TIFFs would mean several frames or
    timepoints, which this backend does not model; the first is used and the
    rest are reported rather than silently dropped.
    """
    try:
        names = sorted(f for f in os.listdir(stack_dir)
                       if f.lower().endswith(_TIFF_EXTS))
    except OSError:
        return None
    if not names:
        return None
    if len(names) > 1:
        print(f"    [vsi] {os.path.basename(stack_dir)} holds "
              f"{len(names)} TIFFs; using {names[0]} and ignoring the rest")
    return os.path.join(stack_dir, names[0])


def _regions(path: str) -> List[Tuple[str, str]]:
    """[(scene_name, tiff_path)] for every region directory in the sidecar."""
    side = sidecar_dir(path)
    if side is None:
        return []
    try:
        entries = sorted(os.listdir(side))
    except OSError:
        return []
    out: List[Tuple[str, str]] = []
    for name in entries:
        full = os.path.join(side, name)
        if not os.path.isdir(full):
            continue
        tiff_path = _region_tiff(full)
        if tiff_path is not None:
            out.append((name, tiff_path))
    return out


def resolve(source_key: str, root: str = "") -> Optional[str]:
    """The TIFF a source key names, or None if this backend does not serve it.

    Purely a filesystem question, which is what makes it safe to use for
    dispatch: an ETS slide's scene ``20x_01`` resolves to no directory, so it
    falls through to slideio, and a key with no scene part never resolves at
    all.
    """
    from .slide_reader import parse_source_key

    filename, scene = parse_source_key(source_key)
    if not scene:
        return None
    path = os.path.join(root, filename) if root else filename
    side = sidecar_dir(path)
    if side is None:
        return None
    stack_dir = os.path.join(side, scene)
    if not os.path.isdir(stack_dir):
        return None
    return _region_tiff(stack_dir)


def claims(source_key: str, root: str = "") -> bool:
    """Whether this backend serves `source_key`."""
    return resolve(source_key, root) is not None


def list_sources(path: str) -> List[str]:
    """Source keys for every usable region in a sidecar-TIFF slide.

    Regions whose TIFF the pipeline cannot process -- the label and the macro
    overview -- are left out, with the reason printed, because an image that
    cannot be segmented should not appear in the wizard at all.
    """
    from .metadata import tiff_reject_reason
    from .slide_reader import make_source_key

    name = os.path.basename(path)
    keys: List[str] = []
    for scene, tiff_path in _regions(path):
        reason = tiff_reject_reason(tiff_path)
        if reason:
            print(f"    [vsi] {name}::{scene} excluded: {reason}")
            continue
        keys.append(make_source_key(name, scene))
    return keys


def _sizes(source_key: str, root: str = "") -> Dict[str, int]:
    from .metadata import probe_tiff_axes

    tiff_path = resolve(source_key, root)
    if tiff_path is None:
        return {}
    return probe_tiff_axes(tiff_path).get('sizes') or {}


def scene_channel_count(source_key: str, root: str = "") -> int:
    """Channels in a region, from its TIFF header."""
    sizes = _sizes(source_key, root)
    return int(sizes.get('C', 1)) if sizes else 1


def scene_shape(source_key: str, root: str = "") -> Optional[Tuple[int, ...]]:
    """(Z, Y, X) or (Y, X) pixel shape, read by axis letter."""
    sizes = _sizes(source_key, root)
    if 'Y' not in sizes or 'X' not in sizes:
        return None
    depth = int(sizes.get('Z', 1))
    height, width = int(sizes['Y']), int(sizes['X'])
    return (depth, height, width) if depth > 1 else (height, width)


def scene_metadata(source_key: str, root: str = "") -> Dict[str, Any]:
    """Physical scale in MICRONS, as {'x','y','z','found'}.

    Delegates to ``read_tiff_metadata``, which already falls back to the OME
    PhysicalSize attributes these files carry -- their TIFF resolution tags are
    absent, so the OME-XML is the only scale on offer.
    """
    from .metadata import MetadataExtractor

    tiff_path = resolve(source_key, root)
    if tiff_path is None:
        return {'x': 1.0, 'y': 1.0, 'z': 1.0, 'found': False}
    return MetadataExtractor.read_tiff_metadata(tiff_path)


def scene_channel_names(source_key: str, root: str = "") -> List[str]:
    """Channel names as the scanner recorded them, or [] if it recorded none.

    Read from the OME-XML's Channel elements. Not derived from the OME image
    name (``20x_DAPI_FITC_TRITC_Cy5_Z_01``), which does list the channels but
    only by convention: splitting it would be guessing at a string the scanner
    never promised to shape that way, and a wrong channel label is worse than
    no label. Used for display only.
    """
    import tifffile as tiff

    tiff_path = resolve(source_key, root)
    if tiff_path is None:
        return []
    try:
        with tiff.TiffFile(tiff_path) as handle:
            xml = str(handle.ome_metadata or '')
    except Exception:
        return []
    if not xml:
        return []
    # Only the first Pixels block: a pyramid repeats the whole Image element
    # once per level, so the channel list appears several times over.
    head = xml.split('</Pixels>', 1)[0]
    names: List[str] = []
    for match in re.finditer(r'<(?:\w+:)?Channel\b[^>]*>', head):
        tag = match.group(0)
        found = (re.search(r'\bName="([^"]*)"', tag)
                 or re.search(r'\bFluor="([^"]*)"', tag))
        if found and found.group(1).strip():
            names.append(found.group(1).strip())
    return names


def extract_scene_channel(
    source_key: str,
    dest_path: str,
    channel_idx: int,
    root: str = "",
    tile: int = 0,
    level: int = 0,
    progress=None,
    should_cancel=None,
) -> bool:
    """Write one channel of one region to a TIFF, a plane at a time.

    `tile` is accepted and ignored: this reads whole pages, because the source
    is a page-per-(channel, z) TIFF rather than a tiled scene that has to be
    assembled. `level` must be 0 -- the pyramid's reduced levels exist in these
    files, but the morphology this pipeline measures depends on the fine
    processes that downsampling destroys, so a request for one is refused
    instead of quietly answered at full resolution.

    `progress` is called as progress(done_planes, total_planes) and
    `should_cancel` is polled between planes.
    """
    from .metadata import ChannelExtractionError, MetadataExtractor

    if int(level) != 0:
        raise ChannelExtractionError(
            f"level {level} was requested, but this backend extracts full "
            "resolution only: the pipeline's morphology metrics depend on "
            "detail that a reduced pyramid level does not carry"
        )

    tiff_path = resolve(source_key, root)
    if tiff_path is None:
        raise ChannelExtractionError(
            f"no sidecar image data could be found for {source_key}")

    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
    source_meta = MetadataExtractor.read_tiff_metadata(tiff_path)
    MetadataExtractor._stream_tiff_channel(
        tiff_path, dest_path, channel_idx, source_meta,
        progress=progress, should_cancel=should_cancel,
    )
    return (os.path.isfile(dest_path) and os.path.getsize(dest_path) > 0)
