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

# Image extensions HIBACHI understands as raw input.
_RAW_IMAGE_EXTS = (".tif", ".tiff", ".czi")

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
        QAbstractItemView, QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
        QListWidgetItem, QMenu, QPushButton, QTreeWidget, QTreeWidgetItem,
        QTreeWidgetItemIterator, QVBoxLayout, QWidget,
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

            recent_label = QLabel("Recent projects")
            recent_label.setStyleSheet("font-weight: bold; margin-top: 6px;")
            root.addWidget(recent_label)

            self.recent_list = QListWidget()
            # Use only itemActivated (Enter / double-click, per platform). Connecting
            # both itemActivated and itemDoubleClicked could fire the slot twice, and
            # relying on the passed item is fragile -- we read the selection instead.
            self.recent_list.itemActivated.connect(self._open_recent)
            root.addWidget(self.recent_list)

            tools = QHBoxLayout()
            tools.addStretch(1)
            self.forget_btn = QPushButton("Remove from list")
            self.forget_btn.clicked.connect(self._forget_selected)
            self.forget_btn.setEnabled(False)
            tools.addWidget(self.forget_btn)
            root.addLayout(tools)

            self.recent_list.itemSelectionChanged.connect(
                lambda: self.forget_btn.setEnabled(bool(self.recent_list.selectedItems()))
            )

        # ---- recent list ---------------------------------------------------- #
        def refresh_recents(self) -> None:
            self.recent_list.clear()
            entries = self.recent.list()
            if not entries:
                placeholder = QListWidgetItem("No recent projects yet.")
                placeholder.setFlags(Qt.NoItemFlags)
                self.recent_list.addItem(placeholder)
                return
            for e in entries:
                label = f"{e.name}\n    {e.path}"
                if not e.exists:
                    label += "   (missing)"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, e.path)
                if not e.exists:
                    item.setForeground(Qt.gray)
                item.setToolTip(e.path)
                self.recent_list.addItem(item)

        def _open_recent(self, item: "QListWidgetItem" = None) -> None:
            # Be defensive: some Qt builds/paths can invoke this without a valid
            # item (the reported 'NoneType' has no attribute 'data'). Fall back to
            # the current selection, and ignore the non-selectable placeholder row.
            if item is None:
                item = self.recent_list.currentItem()
            if item is None:
                return
            path = item.data(Qt.UserRole)
            if path:
                self.path_chosen.emit(path)

        def _forget_selected(self) -> None:
            for item in self.recent_list.selectedItems():
                path = item.data(Qt.UserRole)
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
            add_channel_requested(str)  project dir (multi-channel only)
            resetup_requested(str)      project dir (multi-channel only)
        """

        open_requested = pyqtSignal(str)
        selection_changed = pyqtSignal()
        add_channel_requested = pyqtSignal(str)
        resetup_requested = pyqtSignal(str)

        def __init__(self, registry, channel_dirs=None, project_dir: str = "",
                     multichannel: Optional[bool] = None, parent=None):
            super().__init__(parent)
            self._registry = registry                    # sample -> {channel_key: folder}
            self._channel_dirs = list(channel_dirs or [])  # canonical channel order
            self._project_dir = project_dir
            self._multichannel = (bool(self._channel_dirs)
                                  if multichannel is None else multichannel)
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

            # Positional channel shortcuts: 1 = the first channel shown under each
            # image, 2 = the second, and so on. Numbers stay compact because each
            # channel already appears with its full config name in the tree.
            if self._multichannel and self._channel_dirs:
                sep = QFrame()
                sep.setFrameShape(QFrame.VLine)
                sep.setFrameShadow(QFrame.Sunken)
                top.addWidget(sep)
                top.addWidget(QLabel("Channel:"))
                for i, ch_dir in enumerate(self._channel_dirs, start=1):
                    b = QPushButton(str(i))
                    b.setMaximumWidth(34)
                    b.setToolTip(
                        f"Select {channel_display_name(ch_dir)} for all images."
                    )
                    b.clicked.connect(
                        lambda _=False, key=ch_dir: self._select_channel(key)
                    )
                    top.addWidget(b)

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

                self.resetup_btn = QPushButton("Re-set up project…")
                self.resetup_btn.setToolTip(
                    "Delete the current channel structure and processed results, "
                    "then set the project up again from scratch. Raw images kept."
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
                    parent.setData(0, Qt.UserRole, None)   # parents aren't openable
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

        def _make_leaf(self, name: str, folder: str, channel_key: str) -> "QTreeWidgetItem":
            status = sample_status(folder)
            mode = self._read_mode(folder)
            status_txt = _STATUS_TEXT.get(status, status)
            if mode:
                status_txt += f"  ({mode})"
            leaf = QTreeWidgetItem([
                name,
                self._config_name(folder),
                status_txt,
                format_last_edited(folder_last_edited(folder)),
            ])
            leaf.setData(0, Qt.UserRole, folder)         # actionable path
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
            try:
                for f in os.listdir(folder):
                    if f.lower().endswith((".yaml", ".yml")):
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
            if not self._loading:
                self.selection_changed.emit()

        def _on_double_click(self, item, _column) -> None:
            p = item.data(0, Qt.UserRole)
            if p:
                self.open_requested.emit(p)

    # Backwards-compatible alias: older call sites referred to MultiChannelView.
    MultiChannelView = ProjectContentsView