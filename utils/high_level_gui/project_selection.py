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
        QVBoxLayout, QWidget,
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

    class MultiChannelView(QWidget):
        """
        Sample-first tree for a multi-channel project.

            image_01
                Channel 0 · Microglia   ✓ processed (fluorescence)
                Channel 1 · Plaques     — not processed
            image_02
                Channel 0 · Microglia   … in progress

        Only channels that actually exist for a sample are shown (no ghost rows).
        Double-click a channel leaf to open it. Select any set of channel leaves
        (across samples/channels) and 'Process selected' routes exactly those to
        batch processing; a right-click offers 'Select all in this channel'.

        The widget is presentation only. It emits:
            open_requested(str)        -> a single sample folder to open
            batch_requested(list)      -> sample folders to batch-process
            cross_channel_requested()  -> open the cross-channel analyzer
        Leaf rows carry their sample-folder path in Qt.UserRole; parent rows
        carry None so they're easy to skip.
        """

        open_requested = pyqtSignal(str)
        batch_requested = pyqtSignal(list)
        cross_channel_requested = pyqtSignal()

        def __init__(self, registry, parent=None):
            super().__init__(parent)
            self._registry = registry  # OrderedDict sample -> {channel_dir: sample_folder}
            self._build()
            self.reload(registry)

        def _build(self) -> None:
            root = QVBoxLayout(self)

            self.tree = QTreeWidget()
            self.tree.setHeaderLabels(["Sample / Channel", "Status"])
            self.tree.setColumnWidth(0, 340)
            self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
            self.tree.customContextMenuRequested.connect(self._context_menu)
            self.tree.itemDoubleClicked.connect(self._on_double_click)
            root.addWidget(self.tree)

            btns = QHBoxLayout()
            self.process_btn = QPushButton("Process selected")
            self.process_btn.setToolTip(
                "Batch-process every selected channel row. Tip: right-click a row "
                "to select an entire channel at once."
            )
            self.process_btn.clicked.connect(self._process_selected)
            self.process_btn.setEnabled(False)
            btns.addWidget(self.process_btn)

            self.cross_btn = QPushButton("Cross-channel analysis")
            self.cross_btn.clicked.connect(self.cross_channel_requested.emit)
            btns.addWidget(self.cross_btn)
            btns.addStretch(1)
            root.addLayout(btns)

            self.tree.itemSelectionChanged.connect(
                lambda: self.process_btn.setEnabled(bool(self._selected_leaf_paths()))
            )

        # ---- population ----------------------------------------------------- #
        def reload(self, registry) -> None:
            """Rebuild the tree (and recompute status) from a registry."""
            self._registry = registry
            self.tree.clear()
            for sample, channels in registry.items():
                parent = QTreeWidgetItem([sample, ""])
                parent.setData(0, Qt.UserRole, None)  # parent rows are not openable
                done = 0
                for ch_dir, sample_folder in channels.items():
                    status = sample_status(sample_folder)
                    if status == STATUS_PROCESSED:
                        done += 1
                    mode = self._read_mode(sample_folder)
                    label = channel_display_name(ch_dir)
                    status_txt = _STATUS_TEXT.get(status, status)
                    if mode:
                        status_txt += f"  ({mode})"
                    leaf = QTreeWidgetItem([label, status_txt])
                    leaf.setData(0, Qt.UserRole, sample_folder)
                    leaf.setData(1, Qt.UserRole, os.path.basename(ch_dir.rstrip("/\\")))
                    color = _STATUS_COLOR.get(status)
                    if color:
                        from PyQt5.QtGui import QColor, QBrush  # type: ignore
                        leaf.setForeground(1, QBrush(QColor(color)))
                    parent.addChild(leaf)
                total = len(channels)
                parent.setText(1, f"{done}/{total} processed")
                self.tree.addTopLevelItem(parent)
            self.tree.expandAll()

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

        # ---- selection / actions -------------------------------------------- #
        def _selected_leaf_paths(self) -> List[str]:
            paths = []
            for item in self.tree.selectedItems():
                p = item.data(0, Qt.UserRole)
                if p:  # leaf rows only (parents carry None)
                    paths.append(p)
            return paths

        def _on_double_click(self, item, _column) -> None:
            p = item.data(0, Qt.UserRole)
            if p:
                self.open_requested.emit(p)

        def _process_selected(self) -> None:
            paths = self._selected_leaf_paths()
            if paths:
                self.batch_requested.emit(paths)

        def _context_menu(self, pos) -> None:
            item = self.tree.itemAt(pos)
            if item is None:
                return
            menu = QMenu(self)
            leaf_path = item.data(0, Qt.UserRole)
            channel_key = item.data(1, Qt.UserRole)
            if leaf_path and channel_key:
                act_all = menu.addAction(f"Select all in {channel_display_name(channel_key)}")
                act_open = menu.addAction("Open this channel")
                chosen = menu.exec_(self.tree.viewport().mapToGlobal(pos))
                if chosen == act_all:
                    self._select_entire_channel(channel_key)
                elif chosen == act_open:
                    self.open_requested.emit(leaf_path)

        def _select_entire_channel(self, channel_key: str) -> None:
            """Select every leaf belonging to the given channel (folder basename)."""
            self.tree.clearSelection()
            for i in range(self.tree.topLevelItemCount()):
                parent = self.tree.topLevelItem(i)
                for j in range(parent.childCount()):
                    leaf = parent.child(j)
                    if leaf.data(1, Qt.UserRole) == channel_key:
                        leaf.setSelected(True)