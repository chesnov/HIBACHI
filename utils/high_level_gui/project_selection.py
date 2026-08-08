"""
project_selection: friendlier project/image selection for HIBACHI.

This module has two independently useful halves:

1. Pure logic (no Qt, unit-testable):
     * classify_path(path)      -> Classification   "what did the user pick?"
     * RecentProjects           -> JSON-backed list of recently opened projects

2. A Qt welcome widget (defined only if PyQt5 imports) that presents recent
   projects as clickable rows, accepts drag-and-drop of a folder or image, and
   offers Browse buttons -- routing every selection through classify_path so the
   host window can react intelligently instead of failing on a wrong pick.

Design goals (see the UX discussion): the user should not have to know whether a
folder is "already a project" or "raw images", and picking an image file by
mistake should Just Work (we use its containing folder). All navigation still
falls back to the native OS dialog; we never reimplement a file browser.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .gui_text_utils import is_os_sidecar

# Image extensions HIBACHI understands as raw input.
_RAW_IMAGE_EXTS = (".tif", ".tiff", ".czi")
try:  # whole-slide formats count as raw images for folder classification
    from .slide_formats import supported_extensions as _slide_exts
    _RAW_IMAGE_EXTS = _RAW_IMAGE_EXTS + tuple(_slide_exts())
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Classification (pure logic)
# --------------------------------------------------------------------------- #
# Kinds returned by classify_path:
PROJECT = "project"                    # organized: subfolders each with 1 tif + 1 yaml
MULTICHANNEL_PROJECT = "multichannel_project"  # has Channel_* project subfolders (one dataset)
RAW_IMAGES = "raw_images"              # loose images present, not organized yet
PARENT_OF_PROJECTS = "parent_of_projects"  # contains folders that are themselves projects
EMPTY = "empty"                        # a directory with nothing we recognize
MISSING = "missing"                    # path does not exist


@dataclass
class Classification:
    """Result of inspecting a user-selected path."""
    kind: str
    path: str                          # directory to act on (parent, if a file was picked)
    project_folders: List[str] = field(default_factory=list)  # PROJECT: valid image subfolders
    project_roots: List[str] = field(default_factory=list)    # PARENT_OF_PROJECTS: child projects
    channel_dirs: List[str] = field(default_factory=list)     # MULTICHANNEL_PROJECT: Channel_* dirs
    raw_images: List[str] = field(default_factory=list)       # RAW_IMAGES: loose image files
    redirected_from_file: bool = False  # True if the user picked a file and we used its folder
    note: str = ""                      # short human-readable summary for the UI

    @property
    def is_openable(self) -> bool:
        return self.kind == PROJECT

    @property
    def needs_organizing(self) -> bool:
        return self.kind == RAW_IMAGES


def _valid_image_subfolders(directory: str) -> List[str]:
    """Subfolders that look like a processing unit: exactly one tif and one yaml.

    Mirrors ProjectManager._find_valid_image_folders so the welcome screen and
    the loader agree on what "a project" means.
    """
    out: List[str] = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return out
    for item in entries:
        sub = os.path.join(directory, item)
        if not os.path.isdir(sub):
            continue
        try:
            contents = os.listdir(sub)
        except OSError:
            continue
        contents = [f for f in contents if not is_os_sidecar(f)]
        tifs = [f for f in contents if f.lower().endswith((".tif", ".tiff"))]
        yamls = [f for f in contents if f.lower().endswith((".yaml", ".yml"))]
        if len(tifs) == 1 and len(yamls) == 1:
            out.append(sub)
    return out


def _loose_images(directory: str) -> List[str]:
    """Image files sitting directly in `directory` (not inside subfolders)."""
    out: List[str] = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return out
    for f in entries:
        full = os.path.join(directory, f)
        # A macOS sidecar carries the real file's extension, so without this an
        # otherwise-empty folder classifies as RAW_IMAGES and offers to be set up.
        if is_os_sidecar(f):
            continue
        if os.path.isfile(full) and f.lower().endswith(_RAW_IMAGE_EXTS):
            out.append(full)
    return out


def _child_project_roots(directory: str) -> List[str]:
    """Immediate subfolders that are themselves projects (have valid image subfolders)."""
    out: List[str] = []
    try:
        entries = os.listdir(directory)
    except OSError:
        return out
    for item in entries:
        sub = os.path.join(directory, item)
        if os.path.isdir(sub) and _valid_image_subfolders(sub):
            out.append(sub)
    return out


def _channel_project_dirs(directory: str) -> List[str]:
    """
    Immediate 'Channel_*' subfolders that are themselves projects.

    These are what the multi-channel scaffolder produces (Channel_0_Microglia,
    Channel_1_Plaques, ...), dropped alongside the raw source images. Their
    presence means the folder is already an organized multi-channel project even
    though loose raw images are still sitting next to them.
    """
    out: List[str] = []
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return out
    for item in entries:
        sub = os.path.join(directory, item)
        if (item.lower().startswith("channel_")
                and os.path.isdir(sub) and _valid_image_subfolders(sub)):
            out.append(sub)
    return out


def channel_display_name(channel_dir: str) -> str:
    """'Channel_1_Plaques' -> 'Channel 1 · Plaques' for display."""
    base = os.path.basename(channel_dir.rstrip("/\\"))
    parts = base.split("_")
    if len(parts) >= 3 and parts[0].lower() == "channel":
        return f"Channel {parts[1]} · {' '.join(parts[2:])}"
    if len(parts) >= 2 and parts[0].lower() == "channel":
        return f"Channel {parts[1]}"
    return base


def channel_number_label(channel_dir: str, position: int) -> str:
    """
    Short label for a channel shortcut button.

    Uses the channel's real number from its folder name ('Channel_0_Microglia'
    -> '0') so the button matches the 'Channel 0 ·' text under each image, even
    if channels are non-contiguous. Falls back to the 0-based position when the
    folder name has no parseable channel number.
    """
    base = os.path.basename(channel_dir.rstrip("/\\"))
    parts = base.split("_")
    if len(parts) >= 2 and parts[0].lower() == "channel" and parts[1] != "":
        return parts[1]
    return str(position)


def build_channel_registry(channel_dirs: List[str]) -> "OrderedDict":
    """
    Map sample -> {channel_dir: sample_folder_path}, sample-first for the tree.

    Samples are matched by their subfolder basename, which the scaffolder keeps
    identical across channels (they come from the same source files). Sample
    order follows natural order of the first channel; channels keep input order.
    """
    from collections import OrderedDict
    registry: "OrderedDict[str, OrderedDict]" = OrderedDict()
    for ch_dir in channel_dirs:
        for sample_folder in _valid_image_subfolders(ch_dir):
            sample = os.path.basename(sample_folder.rstrip("/\\"))
            registry.setdefault(sample, OrderedDict())[ch_dir] = sample_folder
    # natural-ish sort of samples (stable, digits-aware)
    def _key(name: str):
        import re
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", name)]
    return OrderedDict(sorted(registry.items(), key=lambda kv: _key(kv[0])))


# Sentinel channel key used for single-channel projects. Every image in a
# single-channel project belongs to this one implicit channel, so any subset of
# a single-channel project is, by definition, "within one channel".
SINGLE_CHANNEL_KEY = "__single_channel__"


def build_single_channel_registry(image_folders: List[str]) -> "OrderedDict":
    """
    Map image_name -> {SINGLE_CHANNEL_KEY: image_folder} for a normal project.

    This mirrors the shape of build_channel_registry so a single content widget
    can render both project kinds. Each image folder is its own sample with one
    implicit channel. Order follows the given list (already natural-sorted by
    ProjectManager).
    """
    from collections import OrderedDict
    registry: "OrderedDict[str, OrderedDict]" = OrderedDict()
    for folder in image_folders:
        name = os.path.basename(folder.rstrip("/\\"))
        registry[name] = OrderedDict([(SINGLE_CHANNEL_KEY, folder)])
    return registry


def format_last_edited(ts: float) -> str:
    """
    Human-readable 'last edited' string.

    Within the past week we show a relative label ('3 days ago'); beyond a week
    we show the actual date. Returns '' for a missing/zero timestamp.
    """
    if not ts:
        return ""
    delta = time.time() - ts
    if delta < 0:
        delta = 0
    DAY = 86400
    if delta < 60:
        return "just now"
    if delta < 3600:
        m = int(delta // 60)
        return f"{m} minute{'s' if m != 1 else ''} ago"
    if delta < DAY:
        h = int(delta // 3600)
        return f"{h} hour{'s' if h != 1 else ''} ago"
    if delta < 7 * DAY:
        d = int(delta // DAY)
        return "yesterday" if d == 1 else f"{d} days ago"
    return datetime.fromtimestamp(ts).strftime("%b %d, %Y")


def folder_last_edited(folder: str) -> float:
    """
    Newest mtime across a sample folder's config YAML and its processed output.

    Reflects 'when this image was last worked on' without loading pixel data:
    the config yaml plus the '*_processed_*' directory (scanned one level deep).
    """
    latest = 0.0
    try:
        for f in os.listdir(folder):
            full = os.path.join(folder, f)
            if f.lower().endswith((".yaml", ".yml")):
                latest = max(latest, os.path.getmtime(full))
            elif os.path.isdir(full) and "_processed_" in f:
                latest = max(latest, os.path.getmtime(full))
                try:
                    for g in os.listdir(full):
                        latest = max(latest, os.path.getmtime(os.path.join(full, g)))
                except OSError:
                    pass
    except OSError:
        pass
    return latest


# Processing-status values for a single (channel, sample) unit.
STATUS_PROCESSED = "processed"
STATUS_IN_PROGRESS = "in_progress"
STATUS_UNPROCESSED = "unprocessed"
STATUS_UNKNOWN = "unknown"


def sample_status(sample_folder: str) -> str:
    """
    Cheap, disk-only processing status for one sample folder.

    Reads the folder's yaml for `mode` and the tif basename, then looks for the
    strategy's output dir `<basename>_processed_<mode>/`:
        metrics csv present  -> processed
        dir present, no csv  -> in_progress
        no dir               -> unprocessed
    This mirrors where ProcessingStrategy writes its checkpoints, without loading
    any image data or instantiating a strategy (so the tree renders instantly).
    """
    try:
        contents = os.listdir(sample_folder)
    except OSError:
        return STATUS_UNKNOWN
    tif = next((f for f in contents if f.lower().endswith((".tif", ".tiff"))), None)
    yml = next((f for f in contents if f.lower().endswith((".yaml", ".yml"))), None)
    if not tif or not yml:
        return STATUS_UNKNOWN

    mode = "unknown"
    try:
        import yaml  # type: ignore
        with open(os.path.join(sample_folder, yml), "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        mode = cfg.get("mode", "unknown")
    except Exception:
        pass

    basename = os.path.splitext(tif)[0]
    processed_dir = os.path.join(sample_folder, f"{basename}_processed_{mode}")
    if not os.path.isdir(processed_dir):
        return STATUS_UNPROCESSED
    try:
        out_files = os.listdir(processed_dir)
    except OSError:
        return STATUS_IN_PROGRESS
    if any(f.startswith("metrics_df_") and f.endswith(".csv") for f in out_files):
        return STATUS_PROCESSED
    return STATUS_IN_PROGRESS


# --------------------------------------------------------------------------- #
# Leaf identity
# --------------------------------------------------------------------------- #
# A tree leaf used to be a sample-folder path. Regions add a third level, so a
# leaf is now either a plain folder (the full image) or "<folder>::<region>".
# Same separator as slide source keys, so there is one convention for "a thing
# inside a thing" rather than two.
LEAF_SEP = "::"


def make_leaf_key(sample_folder: str, roi_name: Optional[str] = None) -> str:
    """Identity string for a tree leaf."""
    return f"{sample_folder}{LEAF_SEP}{roi_name}" if roi_name else sample_folder


def split_leaf_key(key: str) -> tuple:
    """(sample_folder, roi_name or None) for a leaf key.

    Every consumer of checked_folders() must route through this: a bare folder
    means the full image, a key with a region name means that region. Passing a
    key straight to something expecting a folder would look up a directory that
    does not exist.
    """
    text = str(key)
    if LEAF_SEP in text:
        folder, _, roi = text.partition(LEAF_SEP)
        return folder, (roi or None)
    return text, None


def is_roi_leaf(key: str) -> bool:
    """True if this leaf identifies a region rather than a full image."""
    return split_leaf_key(key)[1] is not None


def roi_status(sample_folder: str, roi_name: str) -> str:
    """Processing status of one region, mirroring sample_status for full images.

    Reads the region's own session directory rather than the channel's
    ``<basename>_processed_<mode>``: a region has its own checkpoints and its own
    metrics, so a processed full image says nothing about whether its regions have
    been processed.
    """
    try:
        from .roi_sharing import roi_session_dir
        roi_dir = roi_session_dir(sample_folder, roi_name)
    except Exception:
        return STATUS_UNKNOWN
    if not roi_dir or not os.path.isdir(roi_dir):
        return STATUS_UNKNOWN
    try:
        out_files = os.listdir(roi_dir)
    except OSError:
        return STATUS_UNKNOWN
    if any(f.startswith("metrics_df_") and f.endswith(".csv") for f in out_files):
        return STATUS_PROCESSED
    # A polygon on its own is a defined-but-unprocessed region; anything more
    # means the pipeline has started on it.
    others = [f for f in out_files if f != "roi_polygon.json"]
    return STATUS_IN_PROGRESS if others else STATUS_UNPROCESSED


def prettify_step_name(method: str) -> str:
    """'execute_raw_segmentation' -> 'Raw Segmentation'; strips a mode suffix."""
    if not method:
        return ""
    name = method
    if name.startswith("execute_"):
        name = name[len("execute_"):]
    for sfx in ("_2d", "_3d"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]
    name = name.replace("_", " ").strip()
    return name.title() if name else method


def partial_step_label(sample_folder: str) -> str:
    """
    For a partially-processed folder, name the last fully completed pipeline step.

    Returns a string like 'Step 3/7: Soma Extraction', or '' if it can't be
    determined. This instantiates the strategy (header + config only, no pixel
    data), so callers should invoke it ONLY for in-progress folders to keep the
    tree fast.
    """
    try:
        import os as _os
        import yaml  # type: ignore
        import tifffile as _tiff  # type: ignore

        contents = _os.listdir(sample_folder)
        tif = next((f for f in contents if f.lower().endswith((".tif", ".tiff"))), None)
        yml = next((f for f in contents if f.lower().endswith((".yaml", ".yml"))), None)
        if not tif or not yml:
            return ""
        with open(_os.path.join(sample_folder, yml), "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        mode = cfg.get("mode", "unknown")

        from ..module_3d._3D_strategy import FluorescenceStrategy  # type: ignore
        from ..module_2d._2D_strategy import Fluorescence2DStrategy  # type: ignore
        strat_cls = {
            "fluorescence": FluorescenceStrategy,
            "fluorescence_2d": Fluorescence2DStrategy,
        }.get(mode)
        if strat_cls is None:
            return ""

        with _tiff.TiffFile(_os.path.join(sample_folder, tif)) as tf:
            shape = tf.series[0].shape if tf.series else (1,)
        basename = _os.path.splitext(tif)[0]
        processed_dir = _os.path.join(sample_folder, f"{basename}_processed_{mode}")

        # Spacing/scale don't affect on-disk checkpoint detection, so pass
        # neutral values — we only read get_last_completed_step() + step names.
        strat = strat_cls(
            config=cfg.copy(), processed_dir=processed_dir, image_shape=shape,
            spacing=(1.0, 1.0, 1.0), scale_factor=1.0,
        )
        last = strat.get_last_completed_step()
        total = strat.num_steps
        if last <= 0 or not getattr(strat, "steps", None):
            return ""
        method = strat.steps[last - 1].get("method", "")
        return f"Step {last}/{total}: {prettify_step_name(method)}"
    except Exception as exc:
        # Falling back to a generic "in progress" label is fine, but do it
        # loudly: a silent failure here is why a freshly-processed image can look
        # like a generic in-progress row instead of naming its completed step.
        print(f"[status] partial_step_label({os.path.basename(sample_folder)}) "
              f"failed: {type(exc).__name__}: {exc}")
        return ""


def classify_path(path: Optional[str]) -> Classification:
    """
    Inspect a user-selected path and describe what it is, so the caller can do
    the right thing. If a file is selected, we transparently use its containing
    folder (and flag redirected_from_file).

    Precedence when a folder is ambiguous: an organized PROJECT wins over loose
    RAW_IMAGES (a folder that has both is treated as a project you can open),
    which in turn wins over PARENT_OF_PROJECTS.
    """
    if not path:
        return Classification(MISSING, "", note="No path selected.")

    path = os.path.abspath(path)
    redirected = False
    if os.path.isfile(path):
        redirected = True
        path = os.path.dirname(path)

    if not os.path.isdir(path):
        return Classification(MISSING, path, redirected_from_file=redirected,
                              note="That location doesn't exist.")

    project_folders = _valid_image_subfolders(path)
    if project_folders:
        return Classification(
            PROJECT, path, project_folders=project_folders,
            redirected_from_file=redirected,
            note=f"Project with {len(project_folders)} image folder"
                 f"{'s' if len(project_folders) != 1 else ''} ready to process.",
        )

    # Before treating loose images as "unorganized", check whether this folder
    # already holds Channel_* projects. The multi-channel scaffolder leaves the
    # raw source images in place next to the channel folders, so a folder can be
    # BOTH "has loose images" and "already an organized multi-channel project" --
    # the latter wins, so we don't wrongly offer to build a new project.
    channel_dirs = _channel_project_dirs(path)
    if channel_dirs:
        return Classification(
            MULTICHANNEL_PROJECT, path, channel_dirs=channel_dirs,
            redirected_from_file=redirected,
            note=f"Multi-channel project with {len(channel_dirs)} channel"
                 f"{'s' if len(channel_dirs) != 1 else ''}.",
        )

    raw = _loose_images(path)
    if raw:
        return Classification(
            RAW_IMAGES, path, raw_images=raw, redirected_from_file=redirected,
            note=f"{len(raw)} unorganized image{'s' if len(raw) != 1 else ''} — "
                 f"can be set up as a new project.",
        )

    roots = _child_project_roots(path)
    if roots:
        return Classification(
            PARENT_OF_PROJECTS, path, project_roots=roots,
            redirected_from_file=redirected,
            note=f"Contains {len(roots)} project"
                 f"{'s' if len(roots) != 1 else ''} — choose one to open.",
        )

    return Classification(EMPTY, path, redirected_from_file=redirected,
                          note="No images or projects found in this folder.")


# --------------------------------------------------------------------------- #
# Recent projects (JSON-backed, pure logic)
# --------------------------------------------------------------------------- #
def _default_state_dir() -> str:
    return os.environ.get("HIBACHI_STATE_DIR") or os.path.join(
        os.path.expanduser("~"), ".hibachi"
    )


@dataclass
class RecentEntry:
    path: str
    name: str
    exists: bool


class RecentProjects:
    """A small, corruption-tolerant JSON list of recently opened project paths."""

    def __init__(self, state_dir: Optional[str] = None, limit: int = 12):
        self.limit = limit
        self._file = os.path.join(state_dir or _default_state_dir(), "recent_projects.json")

    def _read(self) -> List[str]:
        try:
            with open(self._file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                # keep only strings, preserve order, dedupe (first wins)
                seen, out = set(), []
                for p in data:
                    if isinstance(p, str) and p not in seen:
                        seen.add(p)
                        out.append(p)
                return out
        except (OSError, ValueError):
            pass
        return []

    def _write(self, paths: List[str]) -> None:
        try:
            os.makedirs(os.path.dirname(self._file), exist_ok=True)
            with open(self._file, "w", encoding="utf-8") as fh:
                json.dump(paths[: self.limit], fh, indent=2)
        except OSError as exc:  # pragma: no cover - best-effort
            print(f"[recent] could not save recent projects: {exc}")

    def add(self, path: str) -> None:
        """Record `path` as most-recently opened (moves it to the top)."""
        if not path:
            return
        path = os.path.abspath(path)
        paths = [p for p in self._read() if os.path.abspath(p) != path]
        paths.insert(0, path)
        self._write(paths)

    def remove(self, path: str) -> None:
        path = os.path.abspath(path)
        self._write([p for p in self._read() if os.path.abspath(p) != path])

    def clear(self) -> None:
        self._write([])

    def list(self) -> List[RecentEntry]:
        """Recent entries, newest first, annotated with whether they still exist."""
        out: List[RecentEntry] = []
        for p in self._read()[: self.limit]:
            out.append(RecentEntry(path=p, name=os.path.basename(p.rstrip("/\\")) or p,
                                   exists=os.path.isdir(p)))
        return out


# --------------------------------------------------------------------------- #
# Welcome widget (Qt) -- defined only if PyQt5 is importable, so the logic above
# can be imported and unit-tested in a headless environment.
# --------------------------------------------------------------------------- #
try:
    from PyQt5.QtCore import Qt, pyqtSignal  # type: ignore
    from PyQt5.QtWidgets import (  # type: ignore
        QAbstractItemView, QComboBox, QFileDialog, QFrame, QHBoxLayout, QLabel,
        QListWidget, QListWidgetItem, QMenu, QPushButton, QTreeWidget,
        QTreeWidgetItem, QTreeWidgetItemIterator, QVBoxLayout, QWidget,
    )
    _HAVE_QT = True
except Exception:  # pragma: no cover - headless / no Qt
    _HAVE_QT = False


if _HAVE_QT:

    class _DropFrame(QFrame):
        """A styled frame that accepts a dropped folder or image and emits its path."""

        dropped = pyqtSignal(str)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setAcceptDrops(True)
            self.setFrameShape(QFrame.StyledPanel)
            self.setMinimumHeight(90)
            self._base_style = (
                "QFrame { border: 2px dashed #9aa0a6; border-radius: 10px; "
                "background: rgba(127,127,127,0.06); }"
            )
            self._hover_style = (
                "QFrame { border: 2px dashed #2E8B57; border-radius: 10px; "
                "background: rgba(46,139,87,0.12); }"
            )
            self.setStyleSheet(self._base_style)
            lay = QVBoxLayout(self)
            msg = QLabel("Drop a folder of images, a project folder,\n"
                         "or even a single image here")
            msg.setAlignment(Qt.AlignCenter)
            msg.setStyleSheet("border: none; color: #6b7075;")
            lay.addWidget(msg)

        def _first_local_path(self, event) -> Optional[str]:
            if not event.mimeData().hasUrls():
                return None
            for url in event.mimeData().urls():
                p = url.toLocalFile()
                if p:
                    return p
            return None

        def dragEnterEvent(self, event):  # noqa: N802 (Qt naming)
            if self._first_local_path(event):
                self.setStyleSheet(self._hover_style)
                event.acceptProposedAction()
            else:
                event.ignore()

        def dragLeaveEvent(self, event):  # noqa: N802
            self.setStyleSheet(self._base_style)

        def dropEvent(self, event):  # noqa: N802
            self.setStyleSheet(self._base_style)
            p = self._first_local_path(event)
            if p:
                event.acceptProposedAction()
                self.dropped.emit(p)
            else:
                event.ignore()

    class WelcomeWidget(QWidget):
        """
        Landing panel shown before a project is loaded.

        Emits `path_chosen(str)` for any user selection (recent row, drop, or a
        Browse dialog). The host window is expected to call classify_path() on it
        and act accordingly, then call refresh_recents() so the list stays current.
        """

        path_chosen = pyqtSignal(str)

        def __init__(self, recent: Optional[RecentProjects] = None, parent=None):
            super().__init__(parent)
            self.recent = recent or RecentProjects()
            self._last_dir = ""  # remember where the user browsed last
            self._build()
            self.refresh_recents()

        # ---- UI construction ------------------------------------------------ #
        def _build(self) -> None:
            root = QVBoxLayout(self)
            root.setSpacing(10)

            title = QLabel("Open a project or start a new one")
            title.setStyleSheet("font-size: 15px; font-weight: bold;")
            root.addWidget(title)

            drop = _DropFrame()
            drop.dropped.connect(self.path_chosen.emit)
            root.addWidget(drop)

            # One button for everything: it accepts a project folder, a folder of
            # raw images, or an image file (its folder is used) -- classify_path
            # figures out which and open_path acts accordingly.
            open_btn = QPushButton("Open…")
            open_btn.setToolTip(
                "Choose a project folder, a folder of images, or an image file."
            )
            open_btn.clicked.connect(self._browse)
            root.addWidget(open_btn)

            # Recent projects live behind a single dropdown button rather than an
            # always-visible list, to give the project tree the vertical space.
            self.recent_btn = QPushButton("Recent projects")
            self.recent_btn.setToolTip("Open a recently used project.")
            self.recent_menu = QMenu(self.recent_btn)
            self.recent_btn.setMenu(self.recent_menu)
            root.addWidget(self.recent_btn)

        # ---- recent menu --------------------------------------------------- #
        def refresh_recents(self) -> None:
            self.recent_menu.clear()
            entries = self.recent.list()
            if not entries:
                act = self.recent_menu.addAction("No recent projects yet.")
                act.setEnabled(False)
                self.recent_btn.setEnabled(False)
                return
            self.recent_btn.setEnabled(True)
            for e in entries:
                text = e.name if e.exists else f"{e.name}   (missing)"
                act = self.recent_menu.addAction(text)
                act.setToolTip(e.path)
                # Bind the path per-action; open on trigger.
                act.triggered.connect(lambda _checked=False, p=e.path: self._open_recent(p))
            # Compact removal, so the capability survives without panel space.
            self.recent_menu.addSeparator()
            remove_menu = self.recent_menu.addMenu("Remove from list")
            for e in entries:
                r = remove_menu.addAction(e.name)
                r.triggered.connect(lambda _checked=False, p=e.path: self._forget(p))

        def _open_recent(self, path: str) -> None:
            if path:
                self.path_chosen.emit(path)

        def _forget(self, path: str) -> None:
            if path:
                self.recent.remove(path)
            self.refresh_recents()

        # ---- browse dialog -------------------------------------------------- #
        def _browse(self) -> None:
            folder = QFileDialog.getExistingDirectory(
                self, "Select a project folder or a folder of images", self._last_dir
            )
            if folder:
                self._last_dir = folder
                self.path_chosen.emit(folder)

    _STATUS_TEXT = {
        STATUS_PROCESSED: "✓ processed",
        STATUS_IN_PROGRESS: "… in progress",
        STATUS_UNPROCESSED: "— not processed",
        STATUS_UNKNOWN: "? unknown",
    }
    _STATUS_COLOR = {
        STATUS_PROCESSED: "#2E8B57",   # green
        STATUS_IN_PROGRESS: "#B8860B", # amber
        STATUS_UNPROCESSED: "#888888", # grey
        STATUS_UNKNOWN: "#888888",
    }

    class _CheckTree(QTreeWidget):
        """QTreeWidget whose Space bar toggles the checkboxes of every currently
        highlighted leaf at once (a convenience on top of clicking each box)."""

        def keyPressEvent(self, event):  # noqa: N802 (Qt naming)
            if event.key() in (Qt.Key_Space, Qt.Key_Select):
                leaves = [it for it in self.selectedItems()
                          if it.data(0, Qt.UserRole)]
                if leaves:
                    cur = self.currentItem()
                    anchor = cur if cur in leaves else leaves[0]
                    new_state = (Qt.Unchecked
                                 if anchor.checkState(0) == Qt.Checked
                                 else Qt.Checked)
                    for it in leaves:
                        it.setCheckState(0, new_state)
                    return
            super().keyPressEvent(event)

    class ProjectContentsView(QWidget):
        """
        Unified, checkbox-driven contents view for single- and multi-channel
        projects.

        Multi-channel layout (sample-first):

            [ ] image_01
                [ ] Channel 0 · Microglia   iMG      ✓ processed (fluorescence)  2 days ago
                [ ] Channel 1 · Plaques     ps129    — not processed             Mar 04, 2026
            [ ] image_02
                [ ] Channel 0 · Microglia   iMG      … in progress              yesterday

        Single-channel layout (each image is its own row / implicit channel):

            [ ] image_01   iMG   ✓ processed (fluorescence)   3 days ago
            [ ] image_02   iMG   — not processed              Jan 12, 2026

        Checkboxes are the single source of truth for what actions apply to.
        Row highlighting (shift/ctrl) only exists so Space can bulk-toggle checks.
        Checking a multi-channel image checks all of its channels (so it then
        spans several channels); the host uses checked_channel_keys() to decide
        whether channel-scoped actions (Set Config) are allowed.

        Signals:
            open_requested(str)         a sample folder to open (double-click)
            selection_changed()         the checked set changed
        Signals:
            open_requested(str)         a sample folder to open (double-click leaf)
            overlay_requested(str)      a sample name to overlay (double-click parent)
            selection_changed()         the checked set changed
            add_channel_requested(str)  project dir (multi-channel only)
            resetup_requested(str)      project dir (multi-channel only)
        """

        open_requested = pyqtSignal(str)
        overlay_requested = pyqtSignal(str)
        selection_changed = pyqtSignal()
        add_channel_requested = pyqtSignal(str)
        resetup_requested = pyqtSignal(str)

        # Roles used on tree items:
        #   column 0, Qt.UserRole      -> leaf: actionable sample-folder path
        #   column 1, Qt.UserRole      -> leaf: channel identity key
        #   column 0, _SAMPLE_ROLE     -> parent: sample name (overlay target)
        _SAMPLE_ROLE = Qt.UserRole + 1

        def __init__(self, registry, channel_dirs=None, project_dir: str = "",
                     multichannel: Optional[bool] = None, analyses=None, parent=None):
            super().__init__(parent)
            self._registry = registry                    # sample -> {channel_key: folder}
            self._channel_dirs = list(channel_dirs or [])  # canonical channel order
            self._project_dir = project_dir
            self._multichannel = (bool(self._channel_dirs)
                                  if multichannel is None else multichannel)
            self._analyses = list(analyses or [])        # cross-channel analysis names
            self._loading = False
            self._build()
            self.reload(registry)

        # ---- construction --------------------------------------------------- #
        def _build(self) -> None:
            root = QVBoxLayout(self)

            top = QHBoxLayout()
            top.addWidget(QLabel("Select:"))
            self.select_all_btn = QPushButton("All images")
            self.select_all_btn.setToolTip("Check every image (and every channel).")
            self.select_all_btn.clicked.connect(lambda: self._check_all(Qt.Checked))
            top.addWidget(self.select_all_btn)

            self.clear_btn = QPushButton("Clear")
            self.clear_btn.setToolTip("Uncheck everything.")
            self.clear_btn.clicked.connect(lambda: self._check_all(Qt.Unchecked))
            top.addWidget(self.clear_btn)

            # Channel shortcuts labelled by real channel number (0-based), so a
            # button matches the 'Channel 0 ·' text shown under each image.
            if self._multichannel and self._channel_dirs:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setFrameShadow(QFrame.Sunken)
                top.addWidget(sep)
                top.addWidget(QLabel("Channel:"))
                for i, ch_dir in enumerate(self._channel_dirs):
                    b = QPushButton(channel_number_label(ch_dir, i))
                    b.setMaximumWidth(34)
                    b.setToolTip(
                        f"Select {channel_display_name(ch_dir)} for all images."
                    )
                    b.clicked.connect(
                        lambda _=False, key=ch_dir: self._select_channel(key)
                    )
                    top.addWidget(b)

            # Cross-channel overlay picker (multi-channel only). Sits to the right
            # of the channel selectors. Hidden until at least one saved analysis
            # exists; double-clicking an image row then overlays the chosen one.
            self._overlay_widgets = []
            self.analysis_combo = None
            if self._multichannel:
                osep = QFrame()
                osep.setFrameShape(QFrame.VLine)
                osep.setFrameShadow(QFrame.Sunken)
                olabel = QLabel("Overlay:")
                self.analysis_combo = QComboBox()
                self.analysis_combo.setToolTip(
                    "Double-click an image (parent) row to overlay this saved "
                    "cross-channel analysis. Channels always open on their own."
                )
                top.addWidget(osep)
                top.addWidget(olabel)
                top.addWidget(self.analysis_combo)
                self._overlay_widgets = [osep, olabel, self.analysis_combo]
                self._populate_analysis_combo(keep_selection=False)

            top.addStretch(1)

            # Project-structure actions live up here, next to the selection tools,
            # so the bottom action bar stays fixed across project kinds.
            if self._multichannel:
                self.add_channel_btn = QPushButton("＋ Add channel…")
                self.add_channel_btn.setToolTip(
                    "Extract another channel from the raw source images that "
                    "remain in the project folder."
                )
                self.add_channel_btn.clicked.connect(
                    lambda: self.add_channel_requested.emit(self._project_dir)
                )
                top.addWidget(self.add_channel_btn)

            # Re-set up is available for both single- and multi-channel projects.
            self.resetup_btn = QPushButton("Re-set up project…")
            self.resetup_btn.setToolTip(
                "Delete the organized structure and processed results, then set "
                "the project up again from scratch. Raw images are kept."
            )
            self.resetup_btn.clicked.connect(
                lambda: self.resetup_requested.emit(self._project_dir)
            )
            top.addWidget(self.resetup_btn)
            root.addLayout(top)

            self.tree = _CheckTree()
            self.tree.setHeaderLabels(["Image / Channel", "Config", "Status", "Last edited"])
            self.tree.setColumnWidth(0, 300)
            self.tree.setColumnWidth(1, 110)
            self.tree.setColumnWidth(2, 200)
            self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.tree.setRootIsDecorated(self._multichannel)
            self.tree.itemDoubleClicked.connect(self._on_double_click)
            self.tree.itemChanged.connect(self._on_item_changed)
            root.addWidget(self.tree)

        # ---- population ----------------------------------------------------- #
        def reload(self, registry) -> None:
            """Rebuild the tree (recomputing status/last-edited) from a registry."""
            self._registry = registry
            self._loading = True
            self.tree.clear()
            if self._multichannel:
                for sample, channels in registry.items():
                    parent = QTreeWidgetItem([sample, "", "", ""])
                    parent.setData(0, Qt.UserRole, None)   # not a leaf: no folder path
                    parent.setData(0, self._SAMPLE_ROLE, sample)  # overlay target
                    parent.setFlags(
                        parent.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate
                    )
                    parent.setCheckState(0, Qt.Unchecked)
                    done = 0
                    for ch_dir, folder in channels.items():
                        if sample_status(folder) == STATUS_PROCESSED:
                            done += 1
                        parent.addChild(self._make_leaf(
                            channel_display_name(ch_dir), folder, ch_dir))
                    parent.setText(2, f"{done}/{len(channels)} processed")
                    self.tree.addTopLevelItem(parent)
                self.tree.expandAll()
            else:
                for name, channels in registry.items():
                    (ch_key, folder), = channels.items()
                    self.tree.addTopLevelItem(self._make_leaf(name, folder, ch_key))
            self._loading = False
            self.selection_changed.emit()

        def refresh(self) -> None:
            """Recompute the tree from the current registry, keeping checks."""
            checked = set(self.checked_folders())
            self.reload(self._registry)
            if checked:
                self._loading = True
                it = QTreeWidgetItemIterator(self.tree)
                while it.value():
                    item = it.value()
                    p = item.data(0, Qt.UserRole)
                    if p and p in checked:
                        item.setCheckState(0, Qt.Checked)
                    it += 1
                self._loading = False
            self.selection_changed.emit()

        def highlight_folder(self, leaf_key: str) -> None:
            """Make `leaf_key`'s row current and scroll it into view.

            Used when returning from an image so the row you were just looking at
            stays marked instead of the list losing its place. Takes a leaf key, so
            returning from a region highlights that region's row rather than
            jumping up to its channel.
            """
            if not leaf_key:
                return
            it = QTreeWidgetItemIterator(self.tree)
            while it.value():
                item = it.value()
                if item.data(0, Qt.UserRole) == leaf_key:
                    # Region rows live under a collapsed-by-default parent in
                    # single-channel projects, so make sure it is visible.
                    parent = item.parent()
                    if parent is not None:
                        parent.setExpanded(True)
                    self.tree.setCurrentItem(item)
                    self.tree.scrollToItem(item)
                    break
                it += 1

        def _make_leaf(self, name: str, folder: str, channel_key: str) -> "QTreeWidgetItem":
            """One channel row, with a child row per saved region.

            The channel row stays actionable in its own right -- it IS the full
            image, and processing the whole image is still a normal thing to do.
            Regions hang beneath it as siblings of nothing, each independently
            checkable, so batch selection can mix full images and regions freely.
            """
            leaf = self._make_image_row(name, folder, channel_key)

            try:
                from .roi_sharing import list_roi_sessions
                sessions = [se for se in list_roi_sessions(folder)
                            if se["has_polygon"]]
            except Exception:
                sessions = []

            if sessions:
                # Deliberately NOT ItemIsAutoTristate. Qt's auto-tristate couples a
                # parent's check state to its children in both directions, which is
                # wrong here: "process the full image" and "process its regions"
                # are different requests, not a whole and its parts. With the flag
                # set, checking one region marked the channel too, and a channel
                # showing Checked is collected by checked_folders() -- so asking for
                # a region would also have queued the full image.
                for session in sessions:
                    leaf.addChild(self._make_roi_row(
                        folder, session["name"], channel_key))
            return leaf

        def _make_roi_row(self, folder: str, roi_name: str,
                          channel_key: str) -> "QTreeWidgetItem":
            """One region row beneath its channel."""
            status = roi_status(folder, roi_name)
            status_txt = _STATUS_TEXT.get(status, status)
            roi_dir = ""
            try:
                from .roi_sharing import roi_session_dir
                roi_dir = roi_session_dir(folder, roi_name) or ""
            except Exception:
                pass
            row = QTreeWidgetItem([
                roi_name,
                self._roi_config_name(roi_dir),
                status_txt,
                format_last_edited(folder_last_edited(roi_dir)) if roi_dir else "",
            ])
            # Identity is the compound key; the channel key is inherited so
            # "checked images span at most one channel" logic keeps working.
            row.setData(0, Qt.UserRole, make_leaf_key(folder, roi_name))
            row.setData(1, Qt.UserRole, channel_key)
            row.setFlags(row.flags() | Qt.ItemIsUserCheckable)
            row.setCheckState(0, Qt.Unchecked)
            color = _STATUS_COLOR.get(status)
            if color:
                from PyQt5.QtGui import QColor, QBrush  # type: ignore
                row.setForeground(2, QBrush(QColor(color)))
            return row

        @staticmethod
        def _roi_config_name(roi_dir: str) -> str:
            """Name of a region's own config, which it owns independently."""
            if not roi_dir:
                return ""
            try:
                for f in sorted(os.listdir(roi_dir)):
                    if f.lower().endswith((".yaml", ".yml")):
                        try:
                            import yaml  # type: ignore
                            with open(os.path.join(roi_dir, f), "r",
                                      encoding="utf-8") as fh:
                                cfg = yaml.safe_load(fh) or {}
                            named = cfg.get("config_name") or cfg.get("name")
                            if named:
                                return str(named)
                        except Exception:
                            pass
                        return os.path.splitext(f)[0]
            except OSError:
                pass
            return ""

        def _make_image_row(self, name: str, folder: str,
                            channel_key: str) -> "QTreeWidgetItem":
            status = sample_status(folder)
            mode = self._read_mode(folder)
            status_txt = _STATUS_TEXT.get(status, status)
            # For a partially-processed folder, name the last completed step
            # instead of a generic "in progress". Computed here only (per the
            # in-progress branch) so complete/unprocessed rows stay instant.
            if status == STATUS_IN_PROGRESS:
                step_lbl = partial_step_label(folder)
                if step_lbl:
                    status_txt = step_lbl
            if mode:
                status_txt += f"  ({mode})"
            leaf = QTreeWidgetItem([
                name,
                self._config_name(folder),
                status_txt,
                format_last_edited(folder_last_edited(folder)),
            ])
            leaf.setData(0, Qt.UserRole, make_leaf_key(folder))  # actionable path
            leaf.setData(1, Qt.UserRole, channel_key)    # channel identity
            leaf.setFlags(leaf.flags() | Qt.ItemIsUserCheckable)
            leaf.setCheckState(0, Qt.Unchecked)
            color = _STATUS_COLOR.get(status)
            if color:
                from PyQt5.QtGui import QColor, QBrush  # type: ignore
                leaf.setForeground(2, QBrush(QColor(color)))
            return leaf

        @staticmethod
        def _config_name(folder: str) -> str:
            # Prefer the config's recorded name (stamped when a preset/template is
            # applied), so the column reflects the actual config rather than the
            # folder's YAML filename. Falls back to the filename stem.
            try:
                for f in os.listdir(folder):
                    if f.lower().endswith((".yaml", ".yml")):
                        try:
                            import yaml  # type: ignore
                            with open(os.path.join(folder, f), "r", encoding="utf-8") as fh:
                                data = yaml.safe_load(fh) or {}
                            name = data.get("config_name")
                            if name:
                                return str(name)
                        except Exception:
                            pass
                        return os.path.splitext(f)[0]
            except OSError:
                pass
            return ""

        @staticmethod
        def _read_mode(sample_folder: str) -> str:
            try:
                import yaml  # type: ignore
                contents = os.listdir(sample_folder)
                yml = next((f for f in contents if f.lower().endswith((".yaml", ".yml"))), None)
                if not yml:
                    return ""
                with open(os.path.join(sample_folder, yml), "r", encoding="utf-8") as fh:
                    return (yaml.safe_load(fh) or {}).get("mode", "") or ""
            except Exception:
                return ""

        # ---- checked-set queries -------------------------------------------- #
        def checked_folders(self) -> List[str]:
            """Sample folders whose leaf is checked (the actionable set)."""
            out: List[str] = []
            it = QTreeWidgetItemIterator(self.tree)
            while it.value():
                item = it.value()
                p = item.data(0, Qt.UserRole)
                if p and item.checkState(0) == Qt.Checked:
                    out.append(p)
                it += 1
            return out

        def checked_channel_keys(self) -> set:
            """Distinct channel identities among the checked leaves."""
            keys = set()
            it = QTreeWidgetItemIterator(self.tree)
            while it.value():
                item = it.value()
                p = item.data(0, Qt.UserRole)
                if p and item.checkState(0) == Qt.Checked:
                    keys.add(item.data(1, Qt.UserRole))
                it += 1
            return keys

        # ---- selection helpers ---------------------------------------------- #
        def _check_all(self, state) -> None:
            self._loading = True
            it = QTreeWidgetItemIterator(self.tree)
            while it.value():
                item = it.value()
                if item.data(0, Qt.UserRole):
                    item.setCheckState(0, state)
                it += 1
            self._loading = False
            self.selection_changed.emit()

        def _select_channel(self, channel_key: str) -> None:
            """Check exactly the leaves of one channel across all images."""
            self._loading = True
            it = QTreeWidgetItemIterator(self.tree)
            while it.value():
                item = it.value()
                p = item.data(0, Qt.UserRole)
                if p:
                    item.setCheckState(
                        0,
                        Qt.Checked if item.data(1, Qt.UserRole) == channel_key
                        else Qt.Unchecked,
                    )
                it += 1
            self._loading = False
            self.selection_changed.emit()

        # ---- events --------------------------------------------------------- #
        def _on_item_changed(self, _item, _col) -> None:
            # No manual propagation guard is needed. Qt only pushes a check state
            # down to children from an item that itself carries
            # ItemIsAutoTristate, and only recomputes a parent from its children
            # under the same condition. Channel rows do not carry the flag, so
            # checking a sample stops at its channels and checking a region never
            # marks its channel.
            if not self._loading:
                self.selection_changed.emit()

        def _on_double_click(self, item, _column) -> None:
            p = item.data(0, Qt.UserRole)
            if p:
                # Leaf: always open this one channel's normal processing view.
                self.open_requested.emit(p)
                return
            sample = item.data(0, self._SAMPLE_ROLE)
            if sample is not None:
                # Parent image row: open the multi-channel sample viewer (raw
                # channels visible, segmentation hidden). If the picker has an
                # analysis selected, the host also overlays its derived layers.
                self.overlay_requested.emit(sample)

        # ---- cross-channel overlay picker ----------------------------------- #
        def _populate_analysis_combo(self, keep_selection: bool = True) -> None:
            combo = self.analysis_combo
            if combo is None:
                return
            previous = combo.currentData() if keep_selection else None
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("— none —", None)          # neutral default: no overlay
            for name in self._analyses:
                combo.addItem(name, name)
            if previous is not None:
                idx = combo.findData(previous)
                combo.setCurrentIndex(idx if idx >= 0 else 0)
            else:
                combo.setCurrentIndex(0)
            combo.blockSignals(False)
            # Only expose the picker once at least one analysis exists.
            visible = bool(self._analyses)
            for w in self._overlay_widgets:
                w.setVisible(visible)

        def set_analyses(self, analyses) -> None:
            """Refresh the list of available cross-channel analyses in the picker."""
            self._analyses = list(analyses or [])
            self._populate_analysis_combo(keep_selection=True)

        def current_analysis(self) -> Optional[str]:
            """The selected analysis name, or None when on the neutral entry."""
            combo = self.analysis_combo
            return combo.currentData() if combo is not None else None

    # Backwards-compatible alias: older call sites referred to MultiChannelView.
    MultiChannelView = ProjectContentsView