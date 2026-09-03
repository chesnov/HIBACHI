"""
slide_formats: what HIBACHI can read through slideio, and what a slide contains.

This is the inspection layer only -- it answers "what is in this file, and can we
trust it?" It deliberately does not extract pixels, because how much of a
gigapixel slide to extract is a scientific decision, not a format one.

Two things drove the design, both learned from a real Olympus VS-series export
rather than from documentation:

1. ONE SLIDE FILE IS USUALLY SEVERAL SAMPLES. The test slide holds six separate
   20x tissue scans, each with three channels (DAPI/FITC/Cy5). HIBACHI's project
   model assumes one file == one image, so a slide has to expand into several
   samples or five of the six scans are silently discarded.

2. WHAT A "SCENE" MEANS DEPENDS ENTIRELY ON THE DRIVER. In VSI the extra images
   (label, overview) are slide-level AUXILIARY images, so every scene is real
   tissue. In SVS the opposite holds: scene 0 is the tissue and the remaining
   scenes are the thumbnail, label and macro. Treating all scenes as samples
   would be correct for VSI and would manufacture junk samples for SVS.

Because of (2), formats are declared individually rather than "whatever slideio
opens". Anything not verified against real data is marked EXPERIMENTAL, reports
why, and is expected to be surfaced to the user as such.

3. NOT EVERY FORMAT IS A FILE, AND NOT EVERY FORMAT IS SLIDEIO. Leica .lif is
   read by readlif and Zarr / OME-Zarr by the zarr package, so `backend` names
   the library per format rather than assuming slideio. Zarr goes further: a
   store is a DIRECTORY, not a file, so `directory_store` marks the formats that
   every path-walking caller must test with os.path.isdir instead of isfile.
   Without that flag a perfectly readable store is invisible to discovery, which
   is a silent failure rather than a reported one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Support tiers
# --------------------------------------------------------------------------- #
TESTED = "tested"              # verified against a real file from this format
EXPERIMENTAL = "experimental"  # driver exists, but nobody has run one through it

# How a driver lays out its scenes. This is the field that cannot be guessed.
ALL_TISSUE = "all_tissue"      # every scene is a real image (VSI)
FIRST_ONLY = "first_only"      # scene 0 is the image; the rest are label/macro (SVS)
BY_SIZE = "by_size"            # tissue and label scenes are mixed; fall back to area

_SUPPORT_UNKNOWN_NOTE = (
    "no file of this format has been tested with HIBACHI; scene layout, channel "
    "order and pixel size are taken on trust from the reader"
)


@dataclass(frozen=True)
class FormatSpec:
    driver: str                 # slideio driver id
    label: str                  # human name
    extensions: Tuple[str, ...]
    support: str
    scenes: str
    note: str = ""
    # A file that is only a header, with pixels in a sibling directory. Copying
    # the header alone leaves a readable file with no image data.
    sidecar_dir: bool = False
    # Which library reads this format. Everything here is slideio except Leica
    # .lif, which slideio has no driver for and which is read by readlif instead,
    # and Zarr, which is read by the zarr package. Declared per format rather
    # than inferred so a format cannot silently be routed to a library that
    # cannot open it.
    backend: str = "slideio"
    # True if this "file" is really a directory tree (Zarr). Callers that walk a
    # folder must test these with os.path.isdir; an isfile test skips them
    # entirely and reports the containing folder as holding no images.
    directory_store: bool = False


# Ordered most-specific-extension first; .ome.tif must beat .tif.
FORMATS: Tuple[FormatSpec, ...] = (
    FormatSpec(
        driver="VSI", label="Olympus / EVIDENT slide scanner",
        extensions=(".vsi",), support=TESTED, scenes=ALL_TISSUE,
        sidecar_dir=True,
        note="verified on a VS-series export: 6 tissue scenes x 3 channels, "
             "label and overview exposed as slide auxiliary images",
    ),
    FormatSpec(
        driver="SVS", label="Aperio", extensions=(".svs",),
        support=EXPERIMENTAL, scenes=FIRST_ONLY,
        note="scene 0 is the image; remaining scenes are thumbnail, label and "
             "macro and must not become samples. " + _SUPPORT_UNKNOWN_NOTE,
    ),
    FormatSpec(
        driver="NDPI", label="Hamamatsu", extensions=(".ndpi",),
        support=EXPERIMENTAL, scenes=BY_SIZE, note=_SUPPORT_UNKNOWN_NOTE,
    ),
    FormatSpec(
        driver="SCN", label="Leica SCN400", extensions=(".scn",),
        support=EXPERIMENTAL, scenes=BY_SIZE,
        note="mixes tissue scans, thumbnails, labels and barcodes as sibling "
             "scenes, so tissue is picked by area. " + _SUPPORT_UNKNOWN_NOTE,
    ),
    FormatSpec(
        driver="AFI", label="Aperio fluorescence", extensions=(".afi",),
        support=EXPERIMENTAL, scenes=ALL_TISSUE, sidecar_dir=True,
        note="an .afi is an XML index pointing at sibling .svs files, one per "
             "channel. " + _SUPPORT_UNKNOWN_NOTE,
    ),
    FormatSpec(
        driver="QPTIFF", label="Akoya / PerkinElmer Phenoptics",
        extensions=(".qptiff",), support=EXPERIMENTAL, scenes=BY_SIZE,
        note=_SUPPORT_UNKNOWN_NOTE,
    ),
    FormatSpec(
        driver="ZVI", label="Zeiss AxioVision", extensions=(".zvi",),
        support=EXPERIMENTAL, scenes=ALL_TISSUE,
        note="a ZVI slide always holds exactly one scene. " + _SUPPORT_UNKNOWN_NOTE,
    ),
    FormatSpec(
        driver="OMETIFF", label="OME-TIFF",
        extensions=(".ome.tif", ".ome.tiff"), support=EXPERIMENTAL, scenes=BY_SIZE,
        note="plain .tif/.tiff keeps using tifffile; only the .ome.tif variants "
             "route here. " + _SUPPORT_UNKNOWN_NOTE,
    ),
    FormatSpec(
        driver="DCM", label="DICOM / DICOM WSI", extensions=(".dcm",),
        support=EXPERIMENTAL, scenes=BY_SIZE,
        note="one scene per DICOM series. " + _SUPPORT_UNKNOWN_NOTE,
    ),
    FormatSpec(
        driver="LIF", label="Leica LAS X", extensions=(".lif",),
        support=EXPERIMENTAL, scenes=ALL_TISSUE, backend="readlif",
        note="one .lif holds many acquisitions, each becoming its own sample. "
             "Read by readlif, not slideio (which has no LIF driver). Mosaic "
             "and time-series acquisitions are skipped with a reason rather "
             "than partially imported. " + _SUPPORT_UNKNOWN_NOTE,
    ),
    # ---- Zarr ------------------------------------------------------------- #
    # .ome.zarr must be declared BEFORE .zarr so the more specific suffix wins,
    # matching the .ome.tif / .tif ordering above. Both route to the same
    # backend; the distinction is only which metadata it expects to find.
    FormatSpec(
        driver="OMEZARR", label="OME-Zarr / NGFF",
        extensions=(".ome.zarr",), support=EXPERIMENTAL, scenes=ALL_TISSUE,
        backend="zarr", directory_store=True,
        note="a directory, not a file. Axis order, pixel size and channel "
             "names are read from the store's own NGFF metadata rather than "
             "guessed. Only the full-resolution level of each multiscale image "
             "becomes a sample; lower pyramid levels and anything under "
             "labels/ are reported and skipped. " + _SUPPORT_UNKNOWN_NOTE,
    ),
    FormatSpec(
        driver="ZARR", label="Zarr", extensions=(".zarr",),
        support=EXPERIMENTAL, scenes=ALL_TISSUE,
        backend="zarr", directory_store=True,
        note="a directory, not a file. One store holds many arrays, each "
             "becoming its own sample. A plain store declares no axis order, so "
             "it is GUESSED by an explicit rule and the assumption is reported "
             "-- check it before processing. Arrays whose name marks them as "
             "labels or ground truth are skipped with a reason. "
             + _SUPPORT_UNKNOWN_NOTE,
    ),
)

#: Backends other than slideio, so callers can dispatch without hardcoding.
BACKEND_SLIDEIO = "slideio"
BACKEND_READLIF = "readlif"
BACKEND_ZARR = "zarr"


def backend_for_path(path: str) -> Optional[str]:
    """Which reader library serves this path, or None if HIBACHI can't read it."""
    spec = spec_for_path(path)
    return spec.backend if spec else None


# --------------------------------------------------------------------------- #
# Directory-tree formats
# --------------------------------------------------------------------------- #
# Every format above except Zarr is a single file, which is why the discovery
# layer historically tested os.path.isfile. These three helpers exist so that
# test can be widened once, here, rather than by scattering ".zarr" literals
# through the callers -- and so that adding another directory format later needs
# no further change outside this module.
def directory_store_extensions() -> Tuple[str, ...]:
    """Extensions whose "file" is really a directory."""
    out: List[str] = []
    for spec in FORMATS:
        if spec.directory_store:
            out.extend(spec.extensions)
    return tuple(out)


def is_directory_store(path: str) -> bool:
    """True if `path` is an existing directory in a directory-tree format.

    Name AND type are both checked: a directory merely named ``x.zarr`` that
    holds no store is still a directory, and the backend refuses it with a
    reason. This function only answers "should discovery treat this as an
    image candidate rather than as a subfolder".
    """
    p = str(path).rstrip("/\\")
    if not p.lower().endswith(directory_store_extensions()):
        return False
    return os.path.isdir(p)


def image_extensions() -> Tuple[str, ...]:
    """Every extension HIBACHI reads, files and directory stores together."""
    return BASE_IMAGE_EXTENSIONS + supported_extensions()


def is_image_path(path: str) -> bool:
    """True if `path` is a readable image, whether it is a file or a store.

    The single predicate a folder walk should use. Replaces
    ``os.path.isfile(p) and p.endswith(exts)``, which silently skipped every
    Zarr store because a store is a directory. Covers TIFF and CZI as well as
    the declared formats, so a caller never needs a second extension list.
    """
    p = str(path)
    if is_directory_store(p):
        return True
    if not os.path.isfile(p):
        return False
    name = str(os.path.basename(p)).lower()
    # A directory-store extension on an actual FILE is not readable -- a file
    # called "x.zarr" is not a store. Excluding those here keeps discovery from
    # offering something the backend would then refuse.
    file_exts = BASE_IMAGE_EXTENSIONS + tuple(
        e for e in supported_extensions()
        if e not in directory_store_extensions())
    return name.endswith(file_exts)

# Deliberately NOT routed through slideio:
#   .czi          -- already handled by aicspylibczi; rerouting risks a working path
#   .tif / .tiff  -- already handled by tifffile
# slideio's CZI and GDAL drivers could serve both, but replacing a working reader
# is a separate decision from adding new formats.
EXCLUDED_EXTENSIONS = (".czi", ".tif", ".tiff")

#: Formats read by their own libraries rather than declared in FORMATS above:
#: TIFF via tifffile, CZI via aicspylibczi. They are listed here so that
#: `is_image_path` can be the ONE predicate a folder walk needs. Without them it
#: would silently answer False for a plain TIFF -- which is the format almost
#: every HIBACHI project is actually made of.
BASE_IMAGE_EXTENSIONS: Tuple[str, ...] = (".tif", ".tiff", ".czi")

# Fraction of the largest scene's area below which a scene is treated as a
# thumbnail / label rather than tissue, under BY_SIZE. The tested VSI slide's six
# scans span 767-997 megapixels (all within 1.3x of each other) while a label or
# macro image is a few megapixels, so this separates them by two orders of
# magnitude rather than by a fine margin.
TISSUE_AREA_FRACTION = 0.10


# --------------------------------------------------------------------------- #
# Known-but-unsupported microscopy formats
# --------------------------------------------------------------------------- #
# Formats a user will plausibly hand HIBACHI that no reader here can open. They
# are listed so an unreadable file gets NAMED rather than silently ignored: an
# unrecognised extension previously fell through every filter, so a folder
# holding one classified as empty ("No images or projects found") while the user
# was looking straight at an image file. Being told "HIBACHI cannot read Leica
# .lif" is actionable; being told the folder is empty is not.
#
# None of these can be served by slideio: its driver set is AFI, CZI, DCM, GDAL,
# NDPI, OMETIFF, PHTIFF, QPTIFF, SCN, SVS, VSI, ZVI. Adding one here to FORMATS
# would match on extension and then fail when the driver was requested, which is
# worse than a clear refusal.
UNSUPPORTED_FORMATS: Tuple[Tuple[str, str], ...] = (
    # .lif is READ (see the LIF FormatSpec above) and must not be listed here.
    (".xlef", "Leica LAS X (project index)"),
    (".nd2", "Nikon NIS-Elements"),
    (".oib", "Olympus FluoView"),
    (".oif", "Olympus FluoView"),
    (".lsm", "Zeiss LSM"),
    (".ims", "Bitplane Imaris"),
    (".sld", "3i SlideBook"),
    (".ipl", "Image-Pro"),
    (".nrrd", "NRRD"),
    (".mrc", "MRC"),
)


def unsupported_format_label(path: str) -> Optional[str]:
    """Vendor name if `path` is a known format HIBACHI cannot read, else None."""
    name = os.path.basename(str(path)).lower()
    for ext, label in UNSUPPORTED_FORMATS:
        if name.endswith(ext):
            return label
    return None


def unsupported_format_message(path: str) -> str:
    """Explain that a file's format cannot be read, and what to do instead.

    Deliberately names the format and gives a concrete route out. The failure a
    user hits here is not "something went wrong" but "this file was never
    readable", and the fix is an export step in their acquisition software.
    """
    name = os.path.basename(str(path))
    label = unsupported_format_label(path) or "this"
    ext = os.path.splitext(name)[1] or ""
    return (
        f"HIBACHI cannot read {label} files ({ext}).\n\n"
        f"File:\n{name}\n\n"
        "Export the image from your acquisition software as OME-TIFF or TIFF "
        "(one file per image), then set the project up from those files. "
        "Multi-channel exports are supported: HIBACHI extracts each channel "
        "itself.\n\n"
        "Readable formats: TIFF (.tif/.tiff), Zeiss CZI (.czi), Leica LIF "
        "(.lif), Zarr and OME-Zarr stores (.zarr/.ome.zarr), and whole-slide "
        "formats (.vsi, .svs, .ndpi, .scn, .afi, .qptiff, .zvi, .ome.tif, .dcm)."
    )


def spec_for_path(path: str) -> Optional[FormatSpec]:
    """FormatSpec for a path, or None if HIBACHI has no format for it.

    A directory store may arrive with a trailing separator (``a/b.zarr/``),
    which would make os.path.basename return "" and match nothing, so the
    separator is stripped first.
    """
    name = os.path.basename(str(path).rstrip("/\\")).lower()
    if name.endswith(EXCLUDED_EXTENSIONS) and not name.endswith((".ome.tif", ".ome.tiff")):
        return None
    for spec in FORMATS:
        if name.endswith(spec.extensions):
            return spec
    return None


def supported_extensions(include_experimental: bool = True) -> Tuple[str, ...]:
    """Extensions HIBACHI will offer, optionally tested-only."""
    out: List[str] = []
    for spec in FORMATS:
        if spec.support == TESTED or include_experimental:
            out.extend(spec.extensions)
    return tuple(out)


# --------------------------------------------------------------------------- #
# Sidecar data
# --------------------------------------------------------------------------- #
def sidecar_status(path: str, spec: Optional[FormatSpec] = None) -> Dict[str, Any]:
    """Report whether a header-plus-sidecar format has its pixel data present.

    A .vsi is a ~3 MB header; the pixels live in sibling `_<stem>_/` directories
    holding .ets tiles (the tested slide had two such directories totalling
    4.7 GB). A user who copies or emails only the .vsi ends up with a file that
    opens and contains nothing, so this is checked before anything else.
    """
    spec = spec or spec_for_path(path)
    result: Dict[str, Any] = {"expects_sidecar": bool(spec and spec.sidecar_dir),
                              "directories": [], "bytes": 0, "ok": True}
    if not result["expects_sidecar"]:
        return result

    folder = os.path.dirname(os.path.abspath(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        entries = sorted(os.listdir(folder))
    except OSError:
        result["ok"] = False
        return result

    # EXACTLY `_<stem>_`, not any directory containing the stem. `stem in item`
    # made 20260901.vsi claim _20260901_01_ and _20260901_02_ as well -- the
    # data belonging to the two slides beside it -- and report their bytes as
    # its own. A slide missing its pixels would then pass this check because a
    # differently-named sibling had some.
    expected = f"_{stem}_"
    total = 0
    for item in entries:
        full = os.path.join(folder, item)
        if not os.path.isdir(full) or item != expected:
            continue
        result["directories"].append(item)
        for root, _dirs, files in os.walk(full):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    result["bytes"] = total
    result["ok"] = bool(result["directories"]) and total > 0
    return result


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
@dataclass
class SceneInfo:
    index: int
    name: str
    width: int
    height: int
    channels: int
    z_slices: int
    channel_names: List[str] = field(default_factory=list)
    um_x: float = 1.0
    um_y: float = 1.0
    um_z: float = 0.0
    magnification: Optional[float] = None
    zoom_levels: int = 0
    is_tissue: bool = True
    excluded_reason: str = ""

    @property
    def megapixels(self) -> float:
        return (self.width * self.height) / 1e6

    @property
    def is_3d(self) -> bool:
        return self.z_slices > 1

    def mode(self) -> str:
        """HIBACHI processing mode implied by this scene's dimensionality."""
        return "fluorescence" if self.is_3d else "fluorescence_2d"


@dataclass
class SlideInfo:
    path: str
    driver: str
    label: str
    support: str
    scenes: List[SceneInfo] = field(default_factory=list)
    aux_images: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    error: str = ""

    @property
    def tissue_scenes(self) -> List[SceneInfo]:
        return [s for s in self.scenes if s.is_tissue]

    @property
    def is_experimental(self) -> bool:
        return self.support == EXPERIMENTAL


def _classify_scenes(spec: FormatSpec, scenes: List[SceneInfo]) -> List[str]:
    """Mark which scenes are real images. Returns any warnings raised.

    This is where the per-driver difference lives, and it is the reason formats
    are declared one by one instead of trusting slideio's driver list wholesale.
    """
    warnings: List[str] = []
    if not scenes:
        return warnings

    if spec.scenes == ALL_TISSUE:
        for s in scenes:
            s.is_tissue = True
        return warnings

    if spec.scenes == FIRST_ONLY:
        for s in scenes:
            s.is_tissue = (s.index == 0)
            if not s.is_tissue:
                s.excluded_reason = "thumbnail / label / macro (not scene 0)"
        if len(scenes) > 1:
            warnings.append(
                f"{spec.label}: using scene 0 as the image and ignoring "
                f"{len(scenes) - 1} other scene(s), which this format uses for "
                "the thumbnail, label and macro."
            )
        return warnings

    # BY_SIZE: tissue and label scenes are siblings, so separate them by area.
    biggest = max(s.width * s.height for s in scenes)
    for s in scenes:
        frac = (s.width * s.height) / biggest if biggest else 0.0
        s.is_tissue = frac >= TISSUE_AREA_FRACTION
        if not s.is_tissue:
            s.excluded_reason = (
                f"only {frac * 100:.1f}% of the largest scene's area; "
                "treated as a thumbnail or label")
    dropped = [s for s in scenes if not s.is_tissue]
    if dropped:
        warnings.append(
            f"{spec.label}: {len(dropped)} small scene(s) were treated as "
            "thumbnails/labels by area, because this format's scene layout has "
            "not been verified. Check the sample list before processing."
        )
    return warnings


def inspect_slide(path: str) -> SlideInfo:
    """Describe a slide: its scenes, channels, pixel size and trust level.

    Loads no pixel data. Every scene is reported, including ones excluded as
    labels or thumbnails, so a wrong guess is visible rather than silent.
    """
    spec = spec_for_path(path)
    if spec is None:
        return SlideInfo(path=path, driver="", label="", support=EXPERIMENTAL,
                         error="not a format HIBACHI reads through slideio")

    # A format served by another library must not be probed with slideio: the
    # driver does not exist, so this would report a confusing driver error for a
    # file HIBACHI can in fact read. Callers route these via that backend
    # instead (see slide_reader._lif_backend).
    if spec.backend != BACKEND_SLIDEIO:
        return SlideInfo(
            path=path, driver=spec.driver, label=spec.label,
            support=spec.support,
            error=f"{spec.label} is read by '{spec.backend}', not slideio; "
                  "use that backend's inspector instead "
                  "(lif_reader.inspect_lif / zarr_reader.inspect_store)",
        )

    info = SlideInfo(path=path, driver=spec.driver, label=spec.label,
                     support=spec.support)
    if spec.note:
        info.warnings.append(spec.note)

    side = sidecar_status(path, spec)
    if side["expects_sidecar"] and not side["ok"]:
        info.error = (
            f"{spec.label} stores its image data in a companion folder next to "
            f"the file, and none was found beside {os.path.basename(path)}. "
            "Copy the whole folder, not just the file."
        )
        return info

    try:
        import slideio
    except ImportError:
        info.error = ("slideio is not installed, so this format cannot be read. "
                      "Install it with:  pip install 'slideio>=2.7.4'")
        return info

    slide = None
    try:
        slide = slideio.open_slide(path, spec.driver)
        try:
            info.aux_images = list(slide.get_aux_image_names())
        except Exception:
            info.aux_images = []

        for i in range(slide.num_scenes):
            try:
                sc = slide.get_scene(i)
            except Exception as exc:
                info.warnings.append(f"scene {i} could not be opened: {exc}")
                continue

            width, height = (int(v) for v in sc.size)
            # slideio reports metres per pixel; HIBACHI works in microns.
            try:
                rx, ry = sc.resolution
                um_x, um_y = float(rx) * 1e6, float(ry) * 1e6
            except Exception:
                um_x = um_y = 0.0
            try:
                um_z = float(sc.z_resolution) * 1e6
            except Exception:
                um_z = 0.0

            names: List[str] = []
            n_ch = int(sc.num_channels)
            for c in range(n_ch):
                try:
                    names.append(str(sc.get_channel_name(c)))
                except Exception:
                    names.append(f"ch{c}")

            scene = SceneInfo(
                index=i, name=str(getattr(sc, "name", "") or f"scene{i}"),
                width=width, height=height, channels=n_ch,
                z_slices=int(getattr(sc, "num_z_slices", 1) or 1),
                channel_names=names, um_x=um_x or 1.0, um_y=um_y or 1.0,
                um_z=um_z,
                magnification=_safe_float(getattr(sc, "magnification", None)),
                zoom_levels=int(getattr(sc, "num_zoom_levels", 0) or 0),
            )
            info.scenes.append(scene)

            # A pixel size of zero means the reader found no calibration. Left
            # unsaid it becomes 1 um/px downstream, which silently corrupts every
            # physical measurement -- the same failure mode as an uncalibrated TIFF.
            if not um_x or not um_y:
                info.warnings.append(
                    f"scene {i} ({scene.name}): no pixel size reported; "
                    "dimensions will need to be supplied manually.")

        info.warnings.extend(_classify_scenes(spec, info.scenes))

        if not info.scenes:
            info.error = "the reader opened the file but reported no scenes"
        elif not info.tissue_scenes:
            info.error = ("every scene was classified as a thumbnail or label, "
                          "so there is nothing to process")

    except Exception as exc:
        info.error = f"{type(exc).__name__}: {exc}"
    finally:
        if slide is not None:
            try:
                slide.close()
            except Exception:
                pass
    return info


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def describe(info: SlideInfo) -> str:
    """User-facing summary of what a slide contains and how far to trust it."""
    if info.error:
        return f"{os.path.basename(info.path)}: {info.error}"

    lines = [f"{os.path.basename(info.path)} — {info.label} ({info.driver})"]
    if info.is_experimental:
        lines.append(
            "EXPERIMENTAL FORMAT: this has not been tested with a real file of "
            "this type. Check the images and their dimensions before trusting "
            "any results.")

    tissue = info.tissue_scenes
    lines.append(f"{len(tissue)} image(s) found"
                 + (f", {len(info.scenes) - len(tissue)} scene(s) skipped"
                    if len(tissue) != len(info.scenes) else ""))
    for s in tissue:
        chans = ", ".join(s.channel_names) or f"{s.channels} channel(s)"
        lines.append(
            f"  {s.name}: {s.width} x {s.height} px ({s.megapixels:.0f} MP), "
            f"{s.channels} ch [{chans}], "
            f"{'3D, ' + str(s.z_slices) + ' slices, ' if s.is_3d else '2D, '}"
            f"{s.um_x:.4f} um/px, {s.zoom_levels} pyramid level(s)")
    for s in info.scenes:
        if not s.is_tissue:
            lines.append(f"  (skipped) {s.name}: {s.excluded_reason}")
    if info.aux_images:
        lines.append(f"auxiliary images: {', '.join(info.aux_images)}")
    for w in info.warnings:
        lines.append(f"note: {w}")
    return "\n".join(lines)