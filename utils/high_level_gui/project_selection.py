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
        QFileDialog, QFrame, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
        QPushButton, QVBoxLayout, QWidget,
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

            hint = QLabel(
                "A project is a folder whose sub-folders each hold one image and "
                "its config:\n"
                "    my_project/  →  sample_01/  (image.tif + config.yaml)\n"
                "Pick a folder of raw images to create one, or a project folder to open it."
            )
            hint.setStyleSheet("color: #6b7075;")
            root.addWidget(hint)

            drop = _DropFrame()
            drop.dropped.connect(self.path_chosen.emit)
            root.addWidget(drop)

            btns = QHBoxLayout()
            open_btn = QPushButton("Open project or images…")
            open_btn.clicked.connect(self._browse_folder)
            btns.addWidget(open_btn)

            file_btn = QPushButton("Pick by image file…")
            file_btn.setToolTip("Select any image; HIBACHI uses its containing folder.")
            file_btn.clicked.connect(self._browse_file)
            btns.addWidget(file_btn)
            root.addLayout(btns)

            recent_label = QLabel("Recent projects")
            recent_label.setStyleSheet("font-weight: bold; margin-top: 6px;")
            root.addWidget(recent_label)

            self.recent_list = QListWidget()
            self.recent_list.itemActivated.connect(self._open_recent)
            self.recent_list.itemDoubleClicked.connect(self._open_recent)
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

        def _open_recent(self, item: "QListWidgetItem") -> None:
            path = item.data(Qt.UserRole)
            if path:
                self.path_chosen.emit(path)

        def _forget_selected(self) -> None:
            for item in self.recent_list.selectedItems():
                path = item.data(Qt.UserRole)
                if path:
                    self.recent.remove(path)
            self.refresh_recents()

        # ---- browse dialogs ------------------------------------------------- #
        def _browse_folder(self) -> None:
            folder = QFileDialog.getExistingDirectory(
                self, "Select a project folder or a folder of images", self._last_dir
            )
            if folder:
                self._last_dir = folder
                self.path_chosen.emit(folder)

        def _browse_file(self) -> None:
            fname, _ = QFileDialog.getOpenFileName(
                self, "Select an image (its folder will be used)", self._last_dir,
                "Images (*.tif *.tiff *.czi);;All files (*)"
            )
            if fname:
                self._last_dir = os.path.dirname(fname)
                self.path_chosen.emit(fname)  # classify_path redirects file -> folder
