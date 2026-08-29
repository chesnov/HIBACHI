"""
zarr_reader: Zarr / OME-Zarr (NGFF) support, behind the same interface as
slide_reader and lif_reader.

A Zarr store is a CONTAINER holding many arrays, which is the same shape as a
whole-slide file holding several scenes, so this reuses slide_reader's
``store.zarr::array/path`` source-key model and every caller that already
handles slides works unchanged. Pixels are extracted to TIFF at project setup
exactly as they are for slides and .lif, so nothing downstream ever touches a
zarr store.

Why a separate backend rather than another entry in FORMATS' slideio table:
slideio has no Zarr driver, and Zarr is not a file at all -- it is a directory
tree. Declaring it there would match on extension and then fail when the driver
was requested.

FIVE THINGS ABOUT THE FORMAT DROVE THIS CODE, all verified against real stores
rather than assumed:

1. **A STORE IS A DIRECTORY, NOT A FILE.** Every other format HIBACHI reads is
   a single file, so the discovery layer gates on ``os.path.isfile``. A zarr
   store must be recognised by ``is_zarr_store`` instead. Nothing here assumes
   a file, and ``sidecar_dir`` does not apply -- the "sidecar" IS the store.

2. **ZARR v2 AND v3 USE DIFFERENT METADATA FILENAMES.** v2 writes ``.zgroup`` /
   ``.zarray`` / ``.zattrs``; v3 writes ``zarr.json``. A detector that knows
   only one silently reports a perfectly good store as unreadable, so both are
   accepted. HIBACHI pins zarr 3.x, which reads both layouts.

3. **AXIS ORDER IS ONLY KNOWN FOR OME-ZARR.** NGFF declares axes by name in
   ``multiscales``. A plain zarr array declares nothing, so a 4D array could be
   (C,Z,Y,X), (T,Z,Y,X) or (Z,C,Y,X) and guessing wrong produces a geometrically
   wrong result that still looks like an image. Plain arrays therefore get an
   explicitly-stated heuristic whose assumption is reported in the warnings and
   can be overridden by the caller via ``axes=``. Declared axes are never
   overridden by the heuristic.

4. **NGFF SCALE CARRIES A UNIT, AND IT IS NOT ALWAYS MICRONS.** The axis ``unit``
   field may be micrometer, nanometer, millimeter or absent. Ignoring it would
   scale every measurement by 1000x while passing every sanity check -- the same
   class of error as the LIF pixels-per-micron trap. Hence ``_UNIT_TO_UM`` and
   its test. An absent scale is reported as not-found so it flows into the manual
   dimension prompt rather than defaulting to 1.0 and pretending to be
   calibrated.

5. **A STORE USUALLY HOLDS LABELS TOO.** FISBe stores ``volumes/raw`` beside
   ``volumes/gt_instances``; NGFF puts segmentations under ``labels/``. Importing
   those as samples would manufacture junk samples out of annotation data, so
   they are excluded BY NAME with the reason stated, never silently dropped.
   Multiscale pyramid levels below the first are likewise not samples.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Recognised store suffixes. ``.ome.zarr`` must be tested before ``.zarr``.
EXTENSIONS: Tuple[str, ...] = (".ome.zarr", ".zarr")
SOURCE_SEP = "::"

#: Metadata filenames that mark a directory as a zarr store. v3 first.
_STORE_MARKERS: Tuple[str, ...] = ("zarr.json", ".zgroup", ".zarray")

#: Array-name fragments that mean "this is annotation data, not an image".
#: Matched case-insensitively against the array's path inside the store.
_LABEL_HINTS: Tuple[str, ...] = (
    "gt_instances", "gt_", "_gt", "groundtruth", "ground_truth",
    "label", "labels", "mask", "masks", "instance", "instances",
    "segmentation", "seg_", "annotation", "annotations",
)

#: NGFF axis unit -> microns. NGFF permits UDUNITS names and a few aliases.
#: Getting this wrong is silent, so unknown units are refused rather than
#: assumed to be microns.
_UNIT_TO_UM: Dict[str, float] = {
    "micrometer": 1.0, "micrometre": 1.0, "micron": 1.0, "um": 1.0,
    "\u00b5m": 1.0, "\u03bcm": 1.0,
    "nanometer": 1e-3, "nanometre": 1e-3, "nm": 1e-3,
    "millimeter": 1e3, "millimetre": 1e3, "mm": 1e3,
    "centimeter": 1e4, "centimetre": 1e4, "cm": 1e4,
    "meter": 1e6, "metre": 1e6, "m": 1e6,
}

#: Smallest-axis length still plausible as a channel axis when guessing the
#: layout of a plain (non-NGFF) 4D array. A 4D array whose smallest axis is
#: larger than this is treated as having no channel axis at all.
MAX_GUESSED_CHANNELS = 8

#: Slices written per progress callback during extraction.
_PROGRESS_EVERY = 1


# --------------------------------------------------------------------------- #
# Availability
# --------------------------------------------------------------------------- #
def is_available() -> bool:
    """True if the reader dependency is installed."""
    try:
        import zarr  # noqa: F401
        return True
    except ImportError:
        return False


def missing_dependency_message() -> str:
    return (
        "Zarr support needs the 'zarr' package, which is not installed.\n\n"
        "Install it with:  pip install 'zarr>=2.16'\n\n"
        "It is pure Python and needs no Java."
    )


# --------------------------------------------------------------------------- #
# Store detection
# --------------------------------------------------------------------------- #
def has_zarr_extension(path: str) -> bool:
    """True if a path is NAMED like a zarr store, whether or not it is one."""
    return str(path).rstrip("/\\").lower().endswith(EXTENSIONS)


def is_zarr_store(path: str) -> bool:
    """True if `path` is a directory holding zarr v2 or v3 metadata.

    Checked by looking for the metadata file rather than by trying to open the
    store, so a directory that merely ends in ``.zarr`` is rejected cheaply and
    without a traceback.
    """
    p = str(path).rstrip("/\\")
    if not os.path.isdir(p):
        return False
    return any(os.path.exists(os.path.join(p, m)) for m in _STORE_MARKERS)


def store_refusal_message(path: str) -> str:
    """Explain why a ``.zarr``-named directory could not be opened."""
    name = os.path.basename(str(path).rstrip("/\\"))
    return (
        f"'{name}' is named like a Zarr store but holds no Zarr metadata "
        f"(none of {', '.join(_STORE_MARKERS)}).\n\n"
        "If this came from an archive, check that it extracted completely and "
        "that you are pointing at the store directory itself rather than a "
        "folder containing it."
    )


# --------------------------------------------------------------------------- #
# Axis handling
# --------------------------------------------------------------------------- #
@dataclass
class AxisLayout:
    """Which array axis is which. ``None`` means the axis is absent."""
    order: str                      # e.g. "czyx", one char per array axis
    declared: bool = False          # True if read from NGFF, False if guessed
    note: str = ""

    def index_of(self, kind: str) -> Optional[int]:
        i = self.order.find(kind)
        return i if i >= 0 else None

    @property
    def spatial(self) -> str:
        return "".join(c for c in self.order if c in "zyx")


def _guess_layout(shape: Sequence[int]) -> AxisLayout:
    """Propose an axis order for an array with no declared axes.

    The heuristic, stated so it can be argued with rather than discovered:
      2D -> (Y, X)
      3D -> (Z, Y, X). A 3-length leading axis is NOT read as RGB, because a
            3-slice stack is far more common in microscopy than an RGB volume
            stored without any metadata saying so.
      4D -> the SHORTEST axis, if at most MAX_GUESSED_CHANNELS long, is the
            channel axis; the remaining three are (Z, Y, X) in order. Ties go to
            the earliest axis, which is the conventional position.
      5D -> (T, C, Z, Y, X), matching NGFF's canonical order.
    Anything else is refused rather than guessed at.
    """
    n = len(shape)
    if n == 2:
        return AxisLayout("yx", False, "2D array read as (Y, X)")
    if n == 3:
        return AxisLayout(
            "zyx", False,
            "3D array read as (Z, Y, X); a leading 3-length axis is NOT "
            "assumed to be RGB")
    if n == 4:
        smallest = min(range(4), key=lambda i: (shape[i], i))
        if shape[smallest] <= MAX_GUESSED_CHANNELS:
            order = ["", "", "", ""]
            order[smallest] = "c"
            for kind, i in zip("zyx", [i for i in range(4) if i != smallest]):
                order[i] = kind
            return AxisLayout(
                "".join(order), False,
                f"4D array: axis {smallest} (length {shape[smallest]}) read as "
                f"channels, giving ({', '.join(c.upper() for c in order)}). "
                "Override this if the array is really ordered differently.")
        return AxisLayout(
            "tzyx", False,
            f"4D array with no short axis (smallest is {shape[smallest]}); "
            "read as (T, Z, Y, X)")
    if n == 5:
        return AxisLayout("tczyx", False,
                          "5D array read as (T, C, Z, Y, X), NGFF's order")
    return AxisLayout("", False, f"{n}-dimensional arrays are not supported")


def _layout_from_axes(axes: Sequence[Any]) -> Optional[AxisLayout]:
    """Build a layout from NGFF ``axes``, or None if they can't be read."""
    order = ""
    for ax in axes:
        name = (ax.get("name") if isinstance(ax, dict) else str(ax)) or ""
        ch = str(name).strip().lower()[:1]
        if ch not in "tczyx":
            return None
        order += ch
    if "y" not in order or "x" not in order:
        return None
    return AxisLayout(order, True, "axis order declared by OME-Zarr metadata")


def parse_axes_override(text: str, ndim: int) -> Optional[AxisLayout]:
    """Turn a user-supplied order like 'czyx' into a layout, or None if invalid.

    Exposed so a dialog can offer a correction for a guessed layout without
    reimplementing the validation.
    """
    order = re.sub(r"[^tczyx]", "", str(text).strip().lower())
    if len(order) != ndim or len(set(order)) != len(order):
        return None
    if "y" not in order or "x" not in order:
        return None
    return AxisLayout(order, True, f"axis order set by the user as '{order}'")


# --------------------------------------------------------------------------- #
# Scale handling
# --------------------------------------------------------------------------- #
def _unit_factor(unit: Any) -> Optional[float]:
    """Microns per `unit`, or None if the unit is unrecognised."""
    if unit is None or unit == "":
        return None
    return _UNIT_TO_UM.get(str(unit).strip().lower())


#: Attribute keys that plain (non-NGFF) stores use for voxel size, in the order
#: they are tried. Values are assumed to be in microns and ordered like the
#: array's spatial axes, which is the only convention these informal keys have.
_PLAIN_SCALE_KEYS: Tuple[str, ...] = (
    "resolution", "voxel_size", "voxelSize", "voxel_size_um",
    "spacing", "pixelResolution", "scale", "element_size_um",
)


def _scale_from_plain_attrs(attrs: Dict[str, Any],
                            n_spatial: int) -> Optional[List[float]]:
    """Voxel size in microns from informal attrs, or None if absent."""
    for key in _PLAIN_SCALE_KEYS:
        if key not in attrs:
            continue
        val = attrs[key]
        if isinstance(val, dict):                       # {'unit':..,'values':..}
            values = val.get("values") or val.get("value") or val.get("size")
            factor = _unit_factor(val.get("unit")) or 1.0
        else:
            values, factor = val, 1.0
        try:
            nums = [float(v) for v in values]
        except (TypeError, ValueError):
            continue
        if len(nums) < n_spatial or any(v <= 0 for v in nums):
            continue
        return [v * factor for v in nums[-n_spatial:]]
    return None


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
@dataclass
class ZarrArrayInfo:
    """One candidate image inside a store."""
    path: str                        # array path within the store, e.g. volumes/raw
    shape: Tuple[int, ...]
    dtype: str
    layout: AxisLayout
    um_x: float = 1.0
    um_y: float = 1.0
    um_z: float = 0.0
    scale_found: bool = False
    channel_names: List[str] = field(default_factory=list)
    multiscale_levels: int = 1
    usable: bool = True
    excluded_reason: str = ""

    @property
    def name(self) -> str:
        """Scene name used in the source key. '/' is legal in a zarr path."""
        return self.path

    @property
    def channels(self) -> int:
        i = self.layout.index_of("c")
        return int(self.shape[i]) if i is not None else 1

    @property
    def z_slices(self) -> int:
        i = self.layout.index_of("z")
        return int(self.shape[i]) if i is not None else 1

    @property
    def height(self) -> int:
        i = self.layout.index_of("y")
        return int(self.shape[i]) if i is not None else 0

    @property
    def width(self) -> int:
        i = self.layout.index_of("x")
        return int(self.shape[i]) if i is not None else 0

    @property
    def timepoints(self) -> int:
        i = self.layout.index_of("t")
        return int(self.shape[i]) if i is not None else 1

    @property
    def is_3d(self) -> bool:
        return self.z_slices > 1

    def mode(self) -> str:
        return "fluorescence" if self.is_3d else "fluorescence_2d"


@dataclass
class ZarrStoreInfo:
    path: str
    arrays: List[ZarrArrayInfo] = field(default_factory=list)
    is_ngff: bool = False
    zarr_format: str = ""
    warnings: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def usable_arrays(self) -> List[ZarrArrayInfo]:
        return [a for a in self.arrays if a.usable]


def _iter_arrays(group, prefix: str = ""):
    """Yield (path, array) for every array in a group, depth first."""
    try:
        items = list(group.arrays())
    except Exception:
        items = []
    for name, arr in items:
        yield (f"{prefix}{name}", arr)
    try:
        subgroups = list(group.groups())
    except Exception:
        subgroups = []
    for name, sub in subgroups:
        yield from _iter_arrays(sub, f"{prefix}{name}/")


def _attrs_dict(obj) -> Dict[str, Any]:
    try:
        return dict(obj.attrs)
    except Exception:
        return {}


def _is_label_path(path: str) -> bool:
    low = str(path).lower()
    return any(h in low for h in _LABEL_HINTS)


def _multiscale_groups(root, root_attrs: Dict[str, Any]):
    """Yield (group_path, group, multiscales_entry) for every NGFF image."""
    if "multiscales" in root_attrs:
        ms = root_attrs["multiscales"]
        if isinstance(ms, list) and ms:
            yield ("", root, ms[0])
    try:
        subgroups = list(root.groups())
    except Exception:
        subgroups = []
    for name, sub in subgroups:
        sub_attrs = _attrs_dict(sub)
        if "multiscales" in sub_attrs:
            ms = sub_attrs["multiscales"]
            if isinstance(ms, list) and ms:
                yield (f"{name}/", sub, ms[0])


def _ngff_channel_names(attrs: Dict[str, Any], n: int) -> List[str]:
    omero = attrs.get("omero") or {}
    chans = omero.get("channels") if isinstance(omero, dict) else None
    names: List[str] = []
    if isinstance(chans, list):
        for c in chans:
            if isinstance(c, dict):
                names.append(str(c.get("label") or c.get("name") or ""))
            else:
                names.append(str(c))
    names = [x for x in names if x]
    return names if len(names) == n else []


def _describe_ngff(group_path: str, group, entry: Dict[str, Any],
                   root_attrs: Dict[str, Any]) -> Optional[ZarrArrayInfo]:
    """Describe the full-resolution level of one NGFF multiscale image."""
    datasets = entry.get("datasets") or []
    if not datasets:
        return None
    axes = entry.get("axes") or []
    first = datasets[0]
    arr_rel = str(first.get("path") or "")
    if not arr_rel:
        return None
    try:
        arr = group[arr_rel]
    except Exception:
        return None

    layout = _layout_from_axes(axes) if axes else None
    if layout is None:
        layout = _guess_layout(arr.shape)
        layout.note = ("OME-Zarr metadata present but axes unreadable; "
                       + layout.note)
    if not layout.order:
        return None

    info = ZarrArrayInfo(
        path=f"{group_path}{arr_rel}",
        shape=tuple(int(v) for v in arr.shape),
        dtype=str(arr.dtype),
        layout=layout,
        multiscale_levels=len(datasets),
    )

    # Scale: match coordinateTransformations to the declared axes, honouring the
    # per-axis unit. Both the dataset-level and multiscale-level transforms are
    # multiplied, which is what NGFF specifies.
    scale = [1.0] * len(layout.order)
    got = False
    for source in (entry.get("coordinateTransformations") or [],
                   first.get("coordinateTransformations") or []):
        for tf in source:
            if not isinstance(tf, dict) or tf.get("type") != "scale":
                continue
            vals = tf.get("scale") or []
            if len(vals) != len(layout.order):
                continue
            try:
                scale = [s * float(v) for s, v in zip(scale, vals)]
                got = True
            except (TypeError, ValueError):
                pass

    units: Dict[str, Optional[float]] = {}
    for ax, ch in zip(axes, layout.order):
        units[ch] = _unit_factor(ax.get("unit") if isinstance(ax, dict) else None)

    def _phys(ch: str, default: float) -> Tuple[float, bool]:
        i = layout.index_of(ch)
        if i is None or not got:
            return default, False
        factor = units.get(ch)
        if factor is None:
            return default, False        # unknown unit: refuse to assume microns
        v = float(scale[i]) * factor
        return (v, True) if v > 0 else (default, False)

    info.um_x, ok_x = _phys("x", 1.0)
    info.um_y, ok_y = _phys("y", 1.0)
    info.um_z, ok_z = _phys("z", 0.0)
    info.scale_found = bool(ok_x and ok_y)

    info.channel_names = _ngff_channel_names(
        {**root_attrs, **_attrs_dict(group)}, info.channels)
    return info


def _describe_plain(path: str, arr,
                    override: Optional[AxisLayout] = None
                    ) -> Optional[ZarrArrayInfo]:
    """Describe a plain (non-NGFF) array."""
    shape = tuple(int(v) for v in arr.shape)
    layout = override or _guess_layout(shape)
    if not layout.order:
        return ZarrArrayInfo(
            path=path, shape=shape, dtype=str(arr.dtype), layout=layout,
            usable=False,
            excluded_reason=f"{len(shape)}-dimensional array; HIBACHI reads "
                            "2D to 5D only")

    info = ZarrArrayInfo(path=path, shape=shape, dtype=str(arr.dtype),
                         layout=layout)
    attrs = _attrs_dict(arr)
    spatial = layout.spatial
    scale = _scale_from_plain_attrs(attrs, len(spatial))
    if scale:
        mapping = dict(zip(spatial, scale))
        info.um_x = float(mapping.get("x", 1.0))
        info.um_y = float(mapping.get("y", 1.0))
        info.um_z = float(mapping.get("z", 0.0))
        info.scale_found = info.um_x > 0 and info.um_y > 0
    return info


def _mark_unusable(info: ZarrArrayInfo) -> ZarrArrayInfo:
    """Apply the exclusion rules that are independent of how we found the array."""
    if _is_label_path(info.path):
        info.usable = False
        info.excluded_reason = (
            "the array name marks this as annotation or label data, not an "
            "image. Segmentation references are compared against results "
            "rather than imported as samples.")
        return info
    if info.timepoints > 1:
        info.usable = False
        info.excluded_reason = (
            f"time series ({info.timepoints} timepoints). HIBACHI analyses one "
            "image per sample and has no time axis, so importing this would "
            "silently keep only the first frame. Extract the timepoints "
            "separately.")
        return info
    if info.width < 2 or info.height < 2:
        info.usable = False
        info.excluded_reason = (
            f"degenerate in-plane size ({info.height} x {info.width})")
    return info


def inspect_store(path: str,
                  overrides: Optional[Dict[str, AxisLayout]] = None
                  ) -> ZarrStoreInfo:
    """Describe a zarr store: its candidate images, their shapes and scale.

    Loads no pixel data. Every array is reported, including ones excluded as
    labels, pyramid levels or time series, so a wrong guess is visible rather
    than silent.
    """
    out = ZarrStoreInfo(path=path)
    if not is_available():
        out.error = missing_dependency_message()
        return out
    if not is_zarr_store(path):
        out.error = store_refusal_message(path)
        return out

    import zarr

    try:
        node = zarr.open(path, mode="r")
    except Exception as exc:
        out.error = f"could not open the Zarr store: {exc}"
        return out

    out.zarr_format = str(getattr(node, "metadata", None) and
                          getattr(node.metadata, "zarr_format", "") or "")

    # A store may be a bare array rather than a group.
    if hasattr(node, "shape") and not hasattr(node, "arrays"):
        info = _describe_plain("", node, (overrides or {}).get(""))
        if info is not None:
            out.arrays.append(_mark_unusable(info))
        _finalise(out)
        return out

    root_attrs = _attrs_dict(node)
    ngff_paths: set = set()

    for group_path, group, entry in _multiscale_groups(node, root_attrs):
        info = _describe_ngff(group_path, group, entry, root_attrs)
        if info is None:
            continue
        out.is_ngff = True
        datasets = entry.get("datasets") or []
        for ds in datasets:
            rel = str(ds.get("path") or "")
            if rel:
                ngff_paths.add(f"{group_path}{rel}")
        if (overrides or {}).get(info.path):
            info.layout = overrides[info.path]
        out.arrays.append(_mark_unusable(info))

    for arr_path, arr in _iter_arrays(node):
        if arr_path in ngff_paths:
            continue            # already described, or a lower pyramid level
        info = _describe_plain(arr_path, arr, (overrides or {}).get(arr_path))
        if info is None:
            continue
        out.arrays.append(_mark_unusable(info))

    _finalise(out)
    return out


def _finalise(out: ZarrStoreInfo) -> None:
    """Attach the warnings a caller must surface."""
    if not out.arrays:
        out.error = ("the Zarr store holds no arrays HIBACHI can read as "
                     "images.")
        return

    usable = out.usable_arrays
    if not usable:
        reasons = "; ".join(
            f"{a.path or '(root)'}: {a.excluded_reason.split('.')[0]}"
            for a in out.arrays[:4])
        out.error = f"no usable images in this Zarr store ({reasons})."
        return

    skipped = [a for a in out.arrays if not a.usable]
    if skipped:
        out.warnings.append(
            f"{len(skipped)} of {len(out.arrays)} array(s) skipped: "
            + "; ".join(f"{a.path or '(root)'} "
                        f"({a.excluded_reason.split('.')[0]})"
                        for a in skipped[:4]))

    guessed = [a for a in usable if not a.layout.declared]
    if guessed:
        out.warnings.append(
            "axis order was GUESSED for "
            + ", ".join(f"'{a.path or '(root)'}'" for a in guessed[:4])
            + " because the store declares none. "
            + guessed[0].layout.note
            + " Check the shape against your acquisition before processing.")

    uncal = [a for a in usable if not a.scale_found]
    if uncal:
        out.warnings.append(
            f"{len(uncal)} array(s) carry no readable pixel size; HIBACHI will "
            "ask for the physical dimensions instead of assuming them.")

    kinds = {a.is_3d for a in usable}
    if len(kinds) > 1:
        out.warnings.append(
            "this store mixes 2D and 3D arrays; they need different processing "
            "modes, so set them up as separate projects")


# --------------------------------------------------------------------------- #
# Source keys  (identical semantics to slide_reader / lif_reader)
# --------------------------------------------------------------------------- #
def _parse(key: str) -> Tuple[str, Optional[str]]:
    text = str(key)
    if SOURCE_SEP in text:
        filename, _, scene = text.partition(SOURCE_SEP)
        return filename, (scene or None)
    return text, None


def is_zarr_source(key: str) -> bool:
    """True if this source key refers to a zarr store."""
    filename, _ = _parse(key)
    return has_zarr_extension(filename)


def list_sources(path: str) -> List[str]:
    """One source key per usable image in a zarr store.

    A store with a single usable array yields a bare name, so it behaves exactly
    like a plain TIFF downstream. Returns [] when nothing is readable, leaving
    the caller to report the reason from ``inspect_store``.
    """
    info = inspect_store(path)
    if info.error:
        return []
    usable = info.usable_arrays
    name = os.path.basename(str(path).rstrip("/\\"))
    if not usable:
        return []
    if len(usable) == 1:
        return [name]
    return [f"{name}{SOURCE_SEP}{a.path}" for a in usable]


def _resolve(source_key: str, root: str = "") -> Tuple[str, ZarrArrayInfo]:
    """(path, array info) for a source key, or raise with a usable message."""
    filename, wanted = _parse(source_key)
    path = os.path.join(root, filename) if root else filename
    info = inspect_store(path)
    if info.error:
        raise ValueError(info.error)

    usable = info.usable_arrays
    if wanted:
        for a in usable:
            if a.path == wanted:
                return path, a
        for a in info.arrays:
            if a.path == wanted:
                raise ValueError(
                    f"'{wanted}' cannot be imported: {a.excluded_reason}")
        raise ValueError(
            f"'{wanted}' is not an array in this store. Available: "
            + ", ".join(a.path for a in usable[:6]))
    if not usable:
        raise ValueError("no usable images in this Zarr store")
    return path, usable[0]


def _open_array(path: str, array_path: str):
    import zarr
    node = zarr.open(path, mode="r")
    if not array_path:
        return node
    return node[array_path]


# --------------------------------------------------------------------------- #
# Metadata, in the shape MetadataExtractor uses
# --------------------------------------------------------------------------- #
def scene_channel_count(source_key: str, root: str = "") -> int:
    """Channels in the array a source key names, or 1 if it can't be read."""
    try:
        _, info = _resolve(source_key, root)
        return int(info.channels)
    except Exception as exc:
        print(f"    Could not read channel count from {source_key}: {exc}")
        return 1


def scene_metadata(source_key: str, root: str = "") -> Dict[str, Any]:
    """Physical scale as {'x','y','z','found'} in MICRONS.

    Matches ``MetadataExtractor.read_tiff_metadata``'s contract so callers can
    treat zarr stores, slides and TIFFs alike. ``found`` is False whenever the
    store carries no scale or carries one in a unit this module does not
    recognise, so the caller prompts instead of assuming.
    """
    meta: Dict[str, Any] = {"x": 1.0, "y": 1.0, "z": 1.0, "found": False}
    try:
        _, info = _resolve(source_key, root)
    except Exception as exc:
        print(f"    Could not read scale from {source_key}: {exc}")
        return meta
    if info.scale_found:
        meta.update({"x": float(info.um_x), "y": float(info.um_y),
                     "found": True})
    # A single-slice array has no meaningful Z spacing; 1.0 keeps the recorded
    # depth equal to the slice count instead of zero.
    meta["z"] = float(info.um_z) if info.um_z and info.um_z > 0 else 1.0
    return meta


def scene_shape(source_key: str, root: str = "") -> Optional[Tuple[int, ...]]:
    """(Z, Y, X) or (Y, X) pixel shape, without reading pixels."""
    try:
        _, info = _resolve(source_key, root)
    except Exception:
        return None
    z, h, w = info.z_slices, info.height, info.width
    return (z, h, w) if z > 1 else (h, w)


def scene_channel_names(source_key: str, root: str = "") -> List[str]:
    """Channel names if the store records them, else []."""
    try:
        _, info = _resolve(source_key, root)
        return list(info.channel_names)
    except Exception:
        return []


def scene_axis_note(source_key: str, root: str = "") -> str:
    """How this array's axis order was decided. For display, not for logic."""
    try:
        _, info = _resolve(source_key, root)
        return info.layout.note
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _plane_selector(info: ZarrArrayInfo, channel_idx: int,
                    z: Optional[int]) -> Tuple[Any, ...]:
    """Index tuple selecting one (Y, X) plane, in the array's own axis order."""
    sel: List[Any] = []
    for ch in info.layout.order:
        if ch == "c":
            sel.append(channel_idx)
        elif ch == "t":
            sel.append(0)
        elif ch == "z":
            sel.append(0 if z is None else z)
        else:
            sel.append(slice(None))
    return tuple(sel)


def _oriented(plane: np.ndarray, info: ZarrArrayInfo) -> np.ndarray:
    """Ensure a selected plane is (Y, X) even if the array stores (X, Y)."""
    if plane.ndim != 2:
        plane = np.asarray(plane).reshape(info.height, info.width)
        return plane
    yi = info.layout.index_of("y")
    xi = info.layout.index_of("x")
    if yi is not None and xi is not None and xi < yi:
        plane = plane.T
    return plane


def extract_scene_channel(
    source_key: str,
    dest_path: str,
    channel_idx: int,
    root: str = "",
    tile: int = 2048,          # accepted for interface parity; unused
    level: int = 0,            # accepted for interface parity; unused
    progress=None,
    should_cancel=None,
) -> bool:
    """Write one channel of one array to a TIFF, plane by plane.

    Peak memory is one (Y, X) plane, because the output is a memmapped TIFF
    filled in place rather than an array assembled in RAM -- the same reason
    slide_reader tiles rather than calling read_block once.

    ``tile`` and ``level`` exist for signature parity with the slide backend.
    Zarr's own chunking already bounds each read, and a multiscale store's lower
    levels are separate arrays addressed by their own source keys, so neither is
    used here rather than being silently half-honoured.
    """
    import tifffile as tiff

    path, info = _resolve(source_key, root)
    n_ch = info.channels
    if not (0 <= channel_idx < n_ch):
        raise ValueError(
            f"channel {channel_idx} requested but this array has {n_ch}")

    arr = _open_array(path, info.path)
    z_slices = info.z_slices
    shape = (z_slices, info.height, info.width) if z_slices > 1 \
        else (info.height, info.width)

    # Boolean and signed label dtypes are widened rather than written as-is:
    # the pipeline's memmaps and the ImageJ TIFF writer both expect an unsigned
    # intensity image, and a silent dtype surprise here surfaces much later as
    # an unreadable checkpoint.
    dtype = np.dtype(arr.dtype)
    if dtype == np.bool_:
        dtype = np.dtype(np.uint8)
    elif dtype.kind == "i":
        dtype = np.dtype(np.uint16 if dtype.itemsize <= 2 else np.uint32)
    elif dtype.kind not in "ufi":
        raise ValueError(f"cannot write an image of dtype {arr.dtype}")

    meta = scene_metadata(source_key, root)
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)

    # imagej=True is required: without it tifffile writes ResolutionUnit=INCH
    # and the micron scale reads back roughly 25400x too large.
    mm = tiff.memmap(
        dest_path, shape=shape, dtype=dtype, imagej=True,
        resolution=(1.0 / meta["x"] if meta["x"] > 0 else 1.0,
                    1.0 / meta["y"] if meta["y"] > 0 else 1.0),
        metadata={"unit": "micron", "spacing": meta["z"]},
    )
    from .slide_reader import SetupCancelled

    try:
        total = max(1, z_slices)
        for z in range(total):
            if should_cancel is not None and should_cancel():
                raise SetupCancelled("cancelled during Zarr extraction")

            sel = _plane_selector(info, channel_idx,
                                  z if z_slices > 1 else None)
            plane = _oriented(np.asarray(arr[sel]), info)
            plane = plane[:info.height, :info.width]

            if z_slices > 1:
                mm[z] = plane.astype(dtype, copy=False)
            else:
                mm[:] = plane.astype(dtype, copy=False)

            if progress is not None and (z % _PROGRESS_EVERY == 0
                                         or z == total - 1):
                progress(z + 1, total)
        mm.flush()
    except SetupCancelled:
        # A half-written channel would pass an existence check and be organized
        # as if complete, so remove it. Same reasoning as the slideio path in
        # slide_reader.extract_scene_channel; a cancelled setup must leave
        # nothing behind that a later run would mistake for finished work.
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