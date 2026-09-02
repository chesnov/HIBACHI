"""
organize_wizard: a guided, multi-step flow for turning raw images into a project.

Replaces the old chain of separate QInputDialog prompts in
project_view_window._load_or_organize with a single QWizard that:

  * detects the raw images and how many channels they have,
  * lets the user confirm single- vs multi-channel,
  * picks a preset per channel (with an auto-derived channel folder name),
  * shows a summary, then runs the scaffolding with a progress dialog.

The same wizard backs Q5's two multi-channel actions:
  * "Add a channel"   -> mode="add",     extracts one more channel from the raw
                          images that are still sitting in the project folder.
  * "Re-set up"       -> mode="resetup", deletes the existing Channel_* structure
                          (and its processed outputs) and runs setup from scratch.

The pure helpers (detect_raw, channel_target_name, existing_channel_indices,
reset_multichannel_project) carry the risky logic and are unit-tested; the Qt
wizard is defined only when PyQt5 is importable.
"""

from __future__ import annotations

import datetime
import os
import re
import shutil
import yaml
from typing import Dict, List, Optional, Sequence, Tuple

_RAW_EXTS = (".tif", ".tiff", ".czi")

# Multi-image formats: whole-slide files read through slideio, Leica .lif, and
# Zarr / OME-Zarr stores. Declared separately because one such "file" expands
# into several samples -- one per scanned scene, or per array in a store.
#
# Zarr is a DIRECTORY, so every walk below tests `os.path.isfile(...) or
# _is_store(...)`. An isfile test alone silently yields zero sources for a
# folder made of stores, which looks identical to an empty folder.
try:
    from .slide_formats import (
        supported_extensions as _slide_exts,
        is_directory_store as _is_store,
    )
    _SLIDE_EXTS = tuple(_slide_exts())
except Exception:
    _SLIDE_EXTS = ()

    def _is_store(_path):  # type: ignore[misc]
        return False
_RAW_EXTS = _RAW_EXTS + _SLIDE_EXTS


def _is_source_entry(full_path: str, name: str, exts) -> bool:
    """True if a directory entry is a readable source of one of `exts`.

    One predicate for both files and directory stores, so the isfile/isdir
    distinction lives here rather than at each walk.
    """
    if not name.lower().endswith(tuple(exts)):
        return False
    return os.path.isfile(full_path) or bool(_is_store(full_path))

# Sidecar filtering lives in gui_text_utils so every path that enumerates raw
# images agrees on what counts as an image (see is_os_sidecar for why this
# matters on macOS external volumes).
from .gui_text_utils import is_os_sidecar  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure logic (unit-testable)
# --------------------------------------------------------------------------- #
def detect_raw(raw_dir: str) -> Dict[str, object]:
    """
    Inspect a folder's raw images.

    Returns {'files': [basename,...], 'max_channels': int, 'has_czi': bool}.
    max_channels is the largest channel count across the images (1 if single).
    """
    from .metadata import MetadataExtractor  # lazy: heavy import, keeps helpers testable

    files: List[str] = []
    skipped: List[str] = []
    try:
        for f in sorted(os.listdir(raw_dir)):
            if not _is_source_entry(os.path.join(raw_dir, f), f, _RAW_EXTS):
                continue
            # macOS AppleDouble sidecars share the real file's extension and sort
            # first, so they must be dropped before anything inspects "the first
            # image" or hands this list to the scaffolder.
            if is_os_sidecar(f):
                skipped.append(f)
                continue

            # A slide file is not one image: each scanned scene becomes its own
            # source, keyed "file::scene". A single-scene slide yields a bare
            # filename, so nothing else in the pipeline has to care.
            if f.lower().endswith(_SLIDE_EXTS):
                try:
                    from .slide_reader import list_sources
                    scene_keys = list_sources(os.path.join(raw_dir, f))
                except Exception as exc:
                    print(f"  [detect] could not read slide {f}: {exc}")
                    scene_keys = []
                if scene_keys:
                    files.extend(scene_keys)
                    if len(scene_keys) > 1:
                        print(f"  [detect] {f} contains "
                              f"{len(scene_keys)} scenes")
                else:
                    print(f"  [detect] {f} yielded no readable scenes; skipped")
                continue

            files.append(f)
    except OSError:
        pass
    if skipped:
        print(f"  [detect] ignored {len(skipped)} operating-system sidecar "
              f"file(s), e.g. {skipped[0]}")

    max_channels = 1
    for f in files:
        try:
            n = MetadataExtractor.get_channel_count(os.path.join(raw_dir, f))
            if isinstance(n, int) and n > max_channels:
                max_channels = n
        except Exception:
            pass
    return {
        "files": files,
        "max_channels": max_channels,
        "has_czi": any(f.lower().endswith(".czi") for f in files),
        "skipped_sidecars": skipped,
    }


def channel_target_name(channel_idx: int, preset_key: str) -> str:
    """Mirror the historical naming: 'Channel_0_Microglia' from preset 'Microglia (3D)'."""
    first = preset_key.split()[0] if preset_key else f"ch{channel_idx}"
    # keep it filesystem-safe
    first = re.sub(r"[^A-Za-z0-9_-]", "", first) or f"ch{channel_idx}"
    return f"Channel_{channel_idx}_{first}"


def existing_channel_indices(project_dir: str) -> List[int]:
    """Channel indices already extracted, parsed from Channel_<n>_* folder names."""
    out: List[int] = []
    try:
        for item in os.listdir(project_dir):
            if os.path.isdir(os.path.join(project_dir, item)):
                m = re.match(r"(?i)channel_(\d+)_", item)
                if m:
                    out.append(int(m.group(1)))
    except OSError:
        pass
    return sorted(set(out))


def reset_multichannel_project(project_dir: str) -> List[str]:
    """
    Delete the organized structure so the project can be set up from scratch,
    WITHOUT touching the raw source images.

    Removes every Channel_* subfolder (each contains that channel's extracted
    images, configs, and processed outputs). Returns the list of removed paths.
    Raw images (.tif/.tiff/.czi files sitting directly in project_dir) are left
    in place -- they are the source the wizard re-extracts from.
    """
    removed: List[str] = []
    try:
        entries = os.listdir(project_dir)
    except OSError:
        return removed
    for item in entries:
        full = os.path.join(project_dir, item)
        if os.path.isdir(full) and re.match(r"(?i)channel_\d+_", item):
            shutil.rmtree(full, ignore_errors=True)
            removed.append(full)
    return removed


def reset_single_channel_project(project_dir: str) -> List[str]:
    """
    Undo a single-channel organize so it can be set up again from scratch.

    Single-channel setup MOVES each image into its own per-image subfolder (with
    a config and, later, processed outputs). This reverses that: every organized
    subfolder (exactly one image + one config at its top level) has its image
    moved back to the project root, then the subfolder is deleted (dropping its
    config and processed outputs). The raw images themselves are preserved.

    Returns the list of removed subfolder paths.
    """
    removed: List[str] = []
    try:
        entries = os.listdir(project_dir)
    except OSError:
        return removed
    for item in entries:
        sub = os.path.join(project_dir, item)
        if not os.path.isdir(sub):
            continue
        try:
            contents = os.listdir(sub)
        except OSError:
            continue
        tifs = [f for f in contents if f.lower().endswith((".tif", ".tiff"))]
        yamls = [f for f in contents if f.lower().endswith((".yaml", ".yml"))]
        if len(tifs) == 1 and len(yamls) == 1:
            src = os.path.join(sub, tifs[0])
            dst = os.path.join(project_dir, tifs[0])
            try:
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            except OSError:
                pass
            shutil.rmtree(sub, ignore_errors=True)
            removed.append(sub)
    return removed


def is_synthetic_channel(channel_dir: str) -> bool:
    """
    True if a channel folder was procedurally generated.

    Synthetic channels carry `synthetic: true` in their per-sample config YAMLs
    (written by the synthetic engine); real extracted channels carry
    `synthetic: false`. A channel whose configs lack the key entirely is treated
    as real. Returns True if any sample config in the channel is marked synthetic.
    """
    try:
        entries = os.listdir(channel_dir)
    except OSError:
        return False
    for item in entries:
        sub = os.path.join(channel_dir, item)
        if not os.path.isdir(sub):
            continue
        try:
            files = os.listdir(sub)
        except OSError:
            continue
        yml = next((f for f in files if f.lower().endswith((".yaml", ".yml"))), None)
        if not yml:
            continue
        try:
            with open(os.path.join(sub, yml), "r") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            continue
        if bool(cfg.get("synthetic", False)):
            return True
    return False


def purge_derived_artifacts(project_dir: str) -> List[str]:
    """
    Remove derived artifacts that must not survive a re-setup.

    Deletes the entire RELATIONAL_ANALYSIS folder (all saved cross-channel runs)
    and any synthetic channels (identified via `synthetic: true` in their
    configs). Real channels are left untouched. Returns the list of removed paths.
    """
    removed: List[str] = []
    rel = os.path.join(project_dir, "RELATIONAL_ANALYSIS")
    if os.path.isdir(rel):
        shutil.rmtree(rel, ignore_errors=True)
        removed.append(rel)
    try:
        entries = os.listdir(project_dir)
    except OSError:
        entries = []
    for item in entries:
        full = os.path.join(project_dir, item)
        if (os.path.isdir(full) and re.match(r"(?i)channel_\d+_", item)
                and is_synthetic_channel(full)):
            shutil.rmtree(full, ignore_errors=True)
            removed.append(full)
    return removed


def detect_default_mode(raw_dir: str) -> Optional[str]:
    """Best-effort guess of a raw folder's processing mode ('fluorescence' /
    'fluorescence_2d'), or None when it can't be told confidently.

    This is used only to *filter* the preset list so a plainly-3D dataset doesn't
    offer 2D configs (and vice-versa). It is deliberately conservative: any doubt
    returns None and the wizard shows every preset, so a wrong guess can never
    leave the user unable to pick a config. The user's preset choice, not this
    guess, ultimately sets the mode.
    """
    import tifffile as tiff  # type: ignore

    try:
        files = [f for f in sorted(os.listdir(raw_dir))
                 if _is_source_entry(os.path.join(raw_dir, f), f,
                                     (".tif", ".tiff") + _SLIDE_EXTS)
                 and not is_os_sidecar(f)]
    except OSError:
        return None
    if not files:
        return None  # e.g. only .czi present -> don't guess

    # Try each candidate rather than only files[0]. One unreadable file used to
    # collapse the whole guess to None, which silently unfiltered the preset list
    # and offered 3D configs for a 2D dataset.
    shape = None
    # Whether `shape` came from a backend's scene_shape (already canonical) or
    # from a raw TIFF series (ambiguous). This distinction decides whether the
    # channel heuristic below may be applied at all -- see the ndim == 3 branch.
    shape_is_canonical = False
    for name in files:
        try:
            if name.lower().endswith(_SLIDE_EXTS):
                from .slide_reader import list_sources, scene_shape
                keys = list_sources(os.path.join(raw_dir, name))
                shape = scene_shape(keys[0], raw_dir) if keys else None
                if shape is None:
                    continue
                shape_is_canonical = True
                break
            with tiff.TiffFile(os.path.join(raw_dir, name)) as tf:
                shape = tf.series[0].shape
            break
        except Exception as exc:
            print(f"  [detect] could not read {name} for mode detection: {exc}")
    if shape is None:
        return None

    ndim = len(shape)
    if ndim == 2:
        return "fluorescence_2d"                       # single plane
    if ndim >= 4:
        return "fluorescence"                          # has Z and channel axes
    if ndim == 3:
        # A backend's scene_shape is ALREADY canonical: slide_reader and
        # zarr_reader both return (Z, Y, X) only when Z > 1 and (Y, X)
        # otherwise, having resolved the channel axis themselves. Applying the
        # TIFF channel heuristic to it re-introduces an ambiguity that no
        # longer exists, and reported any slide, .lif or Zarr stack with fewer
        # than 10 slices as 2D -- so a plainly-3D dataset was offered 2D
        # presets, the exact outcome this function exists to prevent.
        if shape_is_canonical:
            return "fluorescence"
        # A raw TIFF series shape is genuinely ambiguous: (Z, Y, X) stack vs
        # (C, Y, X) multi-channel plane. Mirror the channel heuristic used
        # elsewhere -- a small leading axis (< 10 and < the next) reads as
        # channels -> a 2D image; otherwise it's a Z stack -> 3D.
        if shape[0] < 10 and shape[0] < shape[1]:
            return "fluorescence_2d"
        return "fluorescence"
    return None


# --------------------------------------------------------------------------- #
# Qt wizard (only if PyQt5 is present)
# --------------------------------------------------------------------------- #
try:
    from PyQt5.QtCore import QEventLoop, Qt, QThread, pyqtSignal  # type: ignore
    from PyQt5.QtWidgets import (  # type: ignore
        QApplication, QComboBox, QFormLayout, QFrame, QHBoxLayout, QLabel,
        QListWidget, QListWidgetItem, QMessageBox, QProgressDialog, QPushButton,
        QRadioButton, QVBoxLayout, QButtonGroup, QWizard, QWizardPage, QWidget,
    )

    def _hline() -> "QFrame":
        """A thin separator, so the image list reads as its own section."""
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        return line
    from .project_scaffolding import (
        SetupCancelled,
        organize_channel_project, organize_processing_dir, scan_available_presets,
    )
    _HAVE_QT = True
except Exception:  # pragma: no cover - headless
    _HAVE_QT = False


if _HAVE_QT:

    # Wizard field keys
    _F_MULTI = "is_multichannel"

    class _DetectPage(QWizardPage):
        def __init__(self, wiz: "OrganizeWizard"):
            super().__init__()
            self._wiz = wiz
            self.setTitle("Set up a project")
            lay = QVBoxLayout(self)

            info = wiz.detect
            n = len(info["files"])  # type: ignore[index]
            maxc = info["max_channels"]  # type: ignore[index]
            self.summary = QLabel(
                f"Found {n} image{'s' if n != 1 else ''} in:\n{wiz.raw_dir}\n\n"
                f"Detected up to {maxc} channel{'s' if maxc != 1 else ''} per image."
            )
            self.summary.setWordWrap(True)
            lay.addWidget(self.summary)

            self.rb_single = QRadioButton("Single-channel project")
            self.rb_multi = QRadioButton("Multi-channel project (one project per channel)")
            grp = QButtonGroup(self)
            grp.addButton(self.rb_single)
            grp.addButton(self.rb_multi)
            if maxc > 1:
                self.rb_multi.setChecked(True)
            else:
                self.rb_single.setChecked(True)
                self.rb_multi.setEnabled(False)
            lay.addWidget(self.rb_single)
            lay.addWidget(self.rb_multi)

            # ---- which images to include -------------------------------------
            # Everything is checked by default, so the default behaviour is
            # unchanged. Unchecking lets a project be set up on one image to try
            # something quickly; the rest stay in the folder and can be added later
            # with "Add images...".
            lay.addWidget(_hline())
            self._file_hint = QLabel("")
            self._file_hint.setWordWrap(True)
            lay.addWidget(self._file_hint)

            self.file_list = QListWidget()
            self.file_list.setSelectionMode(QListWidget.NoSelection)
            self.file_list.setMaximumHeight(190)
            for name in info["files"]:  # type: ignore[index]
                item = QListWidgetItem(str(name))
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked)
                self.file_list.addItem(item)
            self.file_list.itemChanged.connect(self._refresh_file_hint)
            lay.addWidget(self.file_list)

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            btn_all = QPushButton("All")
            btn_none = QPushButton("None")
            btn_all.clicked.connect(lambda: self._set_all_files(Qt.Checked))
            btn_none.clicked.connect(lambda: self._set_all_files(Qt.Unchecked))
            row.addWidget(btn_all)
            row.addWidget(btn_none)
            row.addStretch(1)
            lay.addLayout(row)
            self._refresh_file_hint()

            # expose as a wizard field so pages can branch
            self.registerField(_F_MULTI, self.rb_multi)

        def _set_all_files(self, state) -> None:
            for i in range(self.file_list.count()):
                self.file_list.item(i).setCheckState(state)

        def _refresh_file_hint(self, *_a) -> None:
            chosen = len(self.selected_files())
            total = self.file_list.count()
            if chosen == total:
                self._file_hint.setText(
                    f"Include all {total} image{'s' if total != 1 else ''}.")
            else:
                self._file_hint.setText(
                    f"Include {chosen} of {total} images. The other "
                    f"{total - chosen} stay in the folder and can be added later "
                    "with 'Add images\u2026' in the project view.")
            try:
                self.completeChanged.emit()
            except Exception:
                pass

        def selected_files(self) -> List[str]:
            """Image filenames the user checked, in the order they were listed."""
            out: List[str] = []
            for i in range(self.file_list.count()):
                item = self.file_list.item(i)
                if item.checkState() == Qt.Checked:
                    out.append(item.text())
            return out

        def isComplete(self) -> bool:  # noqa: N802
            # Setting up a project with no images would produce nothing, so the
            # Next button stays disabled until at least one is chosen.
            return bool(self.selected_files())

        def nextId(self) -> int:  # noqa: N802
            return self._wiz.page_presets_id

        def isFinalPage(self) -> bool:  # noqa: N802
            # Never final: the user must pass through the presets page (which is
            # where a preset is actually chosen and recorded) before finishing.
            return False

    class _PresetsPage(QWizardPage):
        def __init__(self, wiz: "OrganizeWizard"):
            super().__init__()
            self._wiz = wiz
            self.setTitle("Choose processing presets")
            self._outer = QVBoxLayout(self)
            self._hint = QLabel("")
            self._hint.setWordWrap(True)
            self._outer.addWidget(self._hint)
            self._form_host = QWidget()
            self._form = QFormLayout(self._form_host)
            self._outer.addWidget(self._form_host)
            self._combos: Dict[int, QComboBox] = {}

        def initializePage(self) -> None:  # noqa: N802
            # (re)build the combos depending on single/multi and mode
            while self._form.rowCount():
                self._form.removeRow(0)
            self._combos.clear()

            presets = self._wiz.preset_keys()
            if not presets:
                self._hint.setText("No configuration presets were found.")
                return

            if self._wiz.mode == "add":
                avail = self._wiz.available_channels
                self._hint.setText("Pick the channel to add and its preset.")
                self._chan_combo = QComboBox()
                for idx in avail:
                    self._chan_combo.addItem(f"Channel {idx}", idx)
                self._form.addRow("Channel:", self._chan_combo)
                cb = QComboBox(); cb.addItems(presets)
                self._form.addRow("Preset:", cb)
                self._combos[-1] = cb  # -1 marks the single add-combo
                return

            is_multi = bool(self.field(_F_MULTI)) and self._wiz.detect["max_channels"] > 1  # type: ignore[index]
            if is_multi:
                self._hint.setText("Choose a preset for each channel.")
                for idx in range(int(self._wiz.detect["max_channels"])):  # type: ignore[index]
                    cb = QComboBox(); cb.addItems(presets)
                    self._form.addRow(f"Channel {idx}:", cb)
                    self._combos[idx] = cb
            else:
                self._hint.setText("Choose a preset for this project.")
                cb = QComboBox(); cb.addItems(presets)
                self._form.addRow("Preset:", cb)
                self._combos[0] = cb

        def validatePage(self) -> bool:  # noqa: N802
            # record selections onto the wizard
            if self._wiz.mode == "add":
                idx = self._chan_combo.currentData()
                self._wiz.selections = {int(idx): self._combos[-1].currentText()}
            else:
                self._wiz.selections = {
                    idx: cb.currentText() for idx, cb in self._combos.items()
                }
                self._wiz.is_multichannel = (
                    bool(self.field(_F_MULTI))
                    and int(self._wiz.detect["max_channels"]) > 1  # type: ignore[index]
                )
            return bool(self._wiz.selections)

        def isFinalPage(self) -> bool:  # noqa: N802
            return True

        def nextId(self) -> int:  # noqa: N802
            return -1

    class _SetupWorker(QThread):
        """Runs project setup off the GUI thread.

        Mirrors StepWorker/OptimizeWorker: no Qt widgets are touched here, only
        signals are emitted, so every dialog stays on the main thread. Results are
        left on the instance for the caller to read once `done` has fired.
        """

        progress = pyqtSignal(str, int, int)   # message, done, total
        done = pyqtSignal()

        def __init__(self, raw_files: List[str], raw_dir: str,
                     plan: List[Tuple], single: bool,
                     manual_overrides: Optional[dict] = None):
            super().__init__()
            self.raw_files = raw_files
            self.raw_dir = raw_dir
            self.plan = plan
            self.single = single
            self.manual_overrides = manual_overrides or {}
            self.targets: List[str] = []
            self.summaries: List[dict] = []
            self.error: str = ""
            self.cancelled: bool = False
            self.current_step: int = 0
            self._cancel_requested = False

        def request_cancel(self) -> None:
            """Ask the run to stop at the next tile or file boundary.

            Called from the GUI thread; only ever sets a flag, which the worker
            polls. A running read_block cannot be interrupted, so the stop happens
            at the next boundary rather than instantly.
            """
            self._cancel_requested = True

        def _should_cancel(self) -> bool:
            return self._cancel_requested

        def run(self) -> None:
            try:
                for index, (ch_idx, preset, target) in enumerate(self.plan):
                    self.current_step = index
                    if self._cancel_requested:
                        self.cancelled = True
                        break
                    if self.single:
                        self.progress.emit("Organizing project…", 0, 1)
                        summary = organize_processing_dir(
                            self.raw_dir, preset,
                            only_files=self.raw_files or None,
                            manual_overrides=self.manual_overrides)
                        self.targets.append(target)
                        # Collected just like the multi-channel branch. This
                        # branch used to discard its result, which is why an
                        # uncalibrated single-channel project warned about
                        # nothing at all.
                        if isinstance(summary, dict):
                            self.summaries.append(summary)
                    else:
                        self.progress.emit(
                            f"Extracting channel {ch_idx}…", 0, 1)
                        summary = organize_channel_project(
                            self.raw_files, self.raw_dir, target, ch_idx, preset,
                            progress=lambda msg, d, t: self.progress.emit(msg, d, t),
                            should_cancel=self._should_cancel,
                            manual_overrides=self.manual_overrides,
                        )
                        self.targets.append(target)
                        if isinstance(summary, dict):
                            self.summaries.append(summary)
            except SetupCancelled:
                self.cancelled = True
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.error = f"{type(exc).__name__}: {exc}"
            finally:
                self.done.emit()


    class OrganizeWizard(QWizard):
        """
        Guided setup. Construct with the raw project dir; call exec_() and check
        the return value (QDialog.Accepted). On accept it has already run the
        scaffolding. `mode`:
            "new"     - fresh setup (single or multi, per detection/choice)
            "add"     - add exactly one channel (multi-channel projects)
            "resetup" - caller has already reset; behaves like "new"
        """

        def __init__(self, raw_dir: str, mode: str = "new",
                     project_dir: Optional[str] = None, parent=None):
            super().__init__(parent)
            self.raw_dir = raw_dir
            self.mode = mode
            self.project_dir = project_dir or raw_dir
            self.detect = detect_raw(raw_dir)
            self.presets = scan_available_presets()
            # Best-effort mode guess used only to filter the preset list; None
            # means "show everything" (see detect_default_mode).
            self.mode_filter = detect_default_mode(raw_dir)
            self.selections: Dict[int, str] = {}   # channel_idx -> preset_key
            # Images to include. Empty means "all", which is what the detect page
            # starts with, so narrowing is opt-in.
            self.selected_files: List[str] = []
            self.is_multichannel = False

            if mode == "add":
                used = set(existing_channel_indices(self.project_dir))
                maxc = int(self.detect["max_channels"])  # type: ignore[index]
                self.available_channels = [i for i in range(max(maxc, 1)) if i not in used]
                if not self.available_channels:
                    # nothing left to add; still show, but empty
                    self.available_channels = []

            self.setWindowTitle("HIBACHI — Project setup")
            self.setWizardStyle(QWizard.ModernStyle)

            self.page_presets_id = 1
            self._presets_page = _PresetsPage(self)
            if mode == "add":
                self.setPage(0, self._presets_page)
                self.page_presets_id = 0
            else:
                self._detect_page = _DetectPage(self)
                self.setPage(0, self._detect_page)
                self.setPage(1, self._presets_page)

            self.button(QWizard.FinishButton).setText("Organize")

        def preset_keys(self) -> List[str]:
            """Preset labels to offer, filtered by the detected mode when possible.

            Filters to presets whose ``default_mode`` matches the detected mode so
            a 2D project doesn't list 3D configs (and vice-versa). If the mode is
            unknown, or filtering would leave nothing to choose, every preset is
            returned -- the wizard must never be left with an empty picker.
            """
            all_keys = list(self.presets.keys())
            if not self.mode_filter:
                return all_keys
            filtered = [
                k for k in all_keys
                if self.presets[k].get("default_mode") == self.mode_filter
            ]
            return filtered or all_keys

        def _collect_selections(self) -> None:
            """
            Read the chosen presets straight from the presets page's combos.

            Called at finish time so we never depend on QWizard having invoked
            the page's validatePage() (which only fires if that page is the one
            Finish is clicked on). This makes "No presets were chosen" impossible
            whenever the presets page was actually shown and populated.
            """
            # Which images to include, from the detect page. Absent in "add"
            # mode, which has no detect page, so it falls back to every image.
            detect_page = getattr(self, "_detect_page", None)
            if detect_page is not None:
                self.selected_files = list(detect_page.selected_files())

            page = self._presets_page
            combos = getattr(page, "_combos", {})
            if self.mode == "add":
                chan = getattr(page, "_chan_combo", None)
                if chan is not None and -1 in combos:
                    self.selections = {int(chan.currentData()): combos[-1].currentText()}
            else:
                self.selections = {idx: cb.currentText() for idx, cb in combos.items()}
                self.is_multichannel = (
                    bool(self.field(_F_MULTI))
                    and int(self.detect["max_channels"]) > 1  # type: ignore[index]
                )

        def accept(self) -> None:  # noqa: N802
            # runs when the user clicks Organize/Finish
            try:
                self._collect_selections()
                self._run()
            except SetupCancelled as exc:
                # The user asked for this, so it is not a failure. A red critical
                # box here would read as a crash. The wizard stays open so they
                # can adjust their choices and try again.
                QMessageBox.information(self, "Setup cancelled", str(exc))
                return
            except Exception as exc:  # pragma: no cover - surfaced to user
                QMessageBox.critical(self, "Setup failed", str(exc))
                return
            super().accept()

        def _run(self) -> None:
            """Run setup on a worker thread, keeping the GUI alive throughout.

            Setup used to run inline on the main thread with a single
            processEvents() per channel. For a plain TIFF folder that is
            imperceptible, but a whole-slide project extracts one ~2 GB channel
            per scene, so the event loop could be starved for many minutes and the
            window was reported as "not responding" by the OS -- which is a
            liveness problem, not an unavoidable cost of large images.

            The work now runs in a QThread (matching StepWorker and
            OptimizeWorker elsewhere in the app) while a nested QEventLoop keeps
            painting and lets Cancel through. The nested loop is what allows
            accept() to stay synchronous, which the caller relies on for its
            return value.
            """
            if not self.selections:
                raise ValueError("No presets were chosen.")

            # Only the images the user checked. Everything is checked by default,
            # so this is the full list unless they narrowed it.
            raw_files = list(self.selected_files or self.detect["files"])  # type: ignore[index]
            steps = sorted(self.selections.items())
            single = (self.mode != "add") and not self.is_multichannel

            plan: List[tuple] = []
            for ch_idx, preset_key in steps:
                target = (self.raw_dir if single else os.path.join(
                    self.project_dir, channel_target_name(ch_idx, preset_key)))
                plan.append((ch_idx, self.presets[preset_key], target))

            # Physical scale is resolved BEFORE any work starts, on the GUI
            # thread. Two reasons it belongs here rather than inside the worker:
            # a modal dialog cannot be raised from a QThread, and asking after
            # extraction would mean either re-writing configs afterwards or
            # aborting a multi-minute run. Nothing is asked when the metadata
            # (or a CSV) already supplies every axis, so a calibrated dataset
            # never sees this.
            manual_overrides = self._resolve_missing_dimensions(raw_files, plan)
            if manual_overrides is None:
                return False   # user cancelled at the dimension prompt

            dialog = QProgressDialog("Setting up project…", "Cancel", 0, 100, self)
            dialog.setWindowModality(Qt.WindowModal)
            dialog.setMinimumDuration(0)
            dialog.setAutoClose(False)
            dialog.setAutoReset(False)
            dialog.setValue(0)

            worker = _SetupWorker(raw_files, self.raw_dir, plan, single,
                                  manual_overrides=manual_overrides)
            loop = QEventLoop()

            def _on_progress(message: str, done: int, total: int) -> None:
                dialog.setLabelText(message)
                # Overall progress spans all channels; `done/total` is progress
                # within the current one.
                span = 100.0 / max(1, len(plan))
                base = worker.current_step * span
                frac = (done / total) if total else 0.0
                dialog.setValue(int(min(99, base + frac * span)))

            worker.progress.connect(_on_progress)
            worker.done.connect(lambda: loop.quit())
            dialog.canceled.connect(worker.request_cancel)

            worker.start()
            loop.exec_()          # GUI stays responsive here
            worker.wait()
            dialog.close()

            if worker.cancelled:
                # Remove the channel folders this run created. Left behind, a
                # half-finished set of Channel_* folders classifies as a valid
                # multi-channel project and would open with channels silently
                # missing. The single-channel target is the raw folder itself, so
                # it is never touched.
                removed = 0
                if not single:
                    for target in worker.targets:
                        try:
                            if os.path.isdir(target):
                                shutil.rmtree(target)
                                removed += 1
                        except OSError as exc:
                            print(f"  Could not remove {target}: {exc}")
                raise SetupCancelled(
                    "Project setup was cancelled."
                    + (f"\n\n{removed} partly written channel folder(s) were "
                       "removed." if removed else "")
                    + "\n\nYour raw images were not modified.")
            if worker.error:
                raise ValueError(worker.error)

            self._verify_created(worker.targets, worker.summaries)
            self._report_unscaled(worker.summaries)

        def _resolve_missing_dimensions(self, raw_files, plan):
            """Ask for physical extents that automatic detection could not supply.

            Runs on the GUI thread before setup starts. Returns an overrides dict
            (possibly empty) shaped like the CSV overrides, or ``None`` if the
            user cancelled.

            An axis is only ever asked about when BOTH automatic routes fail --
            the file's own metadata has no trustworthy spacing for it, and no CSV
            row pins it down. A correctly calibrated dataset therefore sees
            nothing, which is the whole requirement: manual entry is a fallback,
            not a step.

            Probing reads headers only (never pixel data), except for the pixel
            counts used to prefill the dialog, which come from the shape.
            """
            from .dimension_entry import collect_manual_dimensions, plan_manual_entry
            from .metadata import MetadataExtractor
            from .gui_text_utils import clean_filename_for_matching
            from .project_scaffolding import (
                _match_dimension_override, load_dimension_overrides,
            )

            # Which axes to ask about is decided PER IMAGE, from that image's own
            # probed rank, inside plan_manual_entry. The presets' modes are no
            # longer consulted: they all carry the single unified mode now, so
            # the old superset heuristic below resolved to the 3D axes for every
            # project and asked for a depth on 2D data.
            #
            #   modes = {preset['default_mode'] for ... in plan}
            #   mode = '' if any(not m.endswith('_2d') for m in modes) else 'fluorescence_2d'
            #
            # `mode` is still passed through as the fallback for an image whose
            # shape could not be probed at all.
            from ..fluorescence_module.config_migration import UNIFIED_MODE
            mode = UNIFIED_MODE

            csv_overrides = {}
            try:
                csv_overrides = load_dimension_overrides(self.raw_dir) or {}
            except Exception as exc:
                print(f"[dimensions] could not read metadata CSV ({exc}).")

            files_meta = []
            for name in raw_files:
                path = os.path.join(self.raw_dir, name)
                meta, pixels = None, {}
                try:
                    if MetadataExtractor._slide_source(name)[0] is not None:
                        meta = MetadataExtractor.read_slide_metadata(path)
                    elif name.lower().endswith('.czi'):
                        meta = MetadataExtractor.get_czi_metadata(path)
                    else:
                        meta = MetadataExtractor.read_tiff_metadata(path)
                except Exception as exc:
                    # An unreadable header is a missing scale, not a crash: the
                    # user is asked, which is exactly the desired outcome.
                    print(f"[dimensions] could not read metadata of {name} ({exc}).")
                try:
                    pixels = self._probe_pixel_counts(path, name, self.raw_dir)
                except Exception:
                    pixels = {}
                files_meta.append((name, meta, pixels))

            needs = plan_manual_entry(
                files_meta, mode,
                csv_overrides=csv_overrides,
                match_override=_match_dimension_override,
            )
            if not needs:
                return {}

            values = collect_manual_dimensions(
                self, needs, mode, clean_name=clean_filename_for_matching)
            if values is None:
                return None
            return values

        @staticmethod
        def _probe_pixel_counts(path, name, self_root=""):
            """Pixel counts per axis, to prefill and sanity-check the dialog.

            Best-effort and cheap: shape only, no pixel data where avoidable. The
            counts are shown next to each field so the user can see the spacing
            their number implies, which is the easiest way to catch a factor-of-
            ten typo.
            """
            from .metadata import MetadataExtractor
            from .dimension_entry import NDIM_KEY
            # The rank travels with the counts. It is recorded here because this
            # is the only place the shape's LENGTH is seen: both branches report
            # z=1 for a 2D image, so the counts alone cannot say whether a depth
            # is a question -- and the mode string no longer can either.
            if MetadataExtractor._slide_source(name)[0] is not None:
                # Slide scenes are sized via the slide reader rather than
                # tifffile. Prefill is a convenience, so failing to get counts
                # only costs the hint next to the field, never the prompt.
                try:
                    from .slide_reader import scene_shape
                    # Returns (Z, Y, X) or (Y, X), or None on any failure.
                    shape = scene_shape(name, self_root) or ()
                    if len(shape) == 3:
                        return {'z': shape[0], 'y': shape[1], 'x': shape[2],
                                NDIM_KEY: 3}
                    if len(shape) == 2:
                        return {'z': 1, 'y': shape[0], 'x': shape[1],
                                NDIM_KEY: 2}
                except Exception:
                    pass
                return {}
            import tifffile as tiff
            with tiff.TiffFile(path) as handle:
                series = handle.series[0]
                shape = series.shape
            if len(shape) >= 3:
                return {'z': shape[0], 'y': shape[-2], 'x': shape[-1],
                        NDIM_KEY: 3}
            if len(shape) == 2:
                return {'z': 1, 'y': shape[0], 'x': shape[1], NDIM_KEY: 2}
            return {}

        def _report_unscaled(self, summaries: List[dict]) -> None:
            """Tell the user when images ended up with pixel counts for dimensions.

            An image with no usable scale metadata and no matching CSV row still
            organizes fine, but its recorded size is really its pixel count. Left
            unsaid, that silently propagates into every downstream measurement, so
            it's worth one dialog naming the files and how to fix them.
            """
            unscaled, csvs = [], set()
            for summary in summaries:
                unscaled.extend(summary.get('unscaled') or [])
                if summary.get('csv'):
                    csvs.add(summary['csv'])
            if not unscaled:
                return

            names = sorted(set(unscaled))
            shown = "\n".join(f"\u2022 {n}" for n in names[:8])
            if len(names) > 8:
                shown += f"\n\u2026 and {len(names) - 8} more"
            if csvs:
                csv_note = (
                    f"{', '.join(sorted(csvs))} was read, but it has no row "
                    "matching these files."
                )
            else:
                csv_note = "No metadata CSV was found next to the images."

            QMessageBox.warning(
                self, "Image dimensions are in pixels",
                f"{len(names)} image(s) had no usable scale metadata:\n\n{shown}"
                f"\n\n{csv_note}\n\nTheir dimensions were recorded as pixel "
                "counts, which will make physical measurements wrong. To fix "
                "this, put a CSV next to your raw images with 'Filename', "
                "'Width (um)', 'Height (um)' and 'Depth (um)' columns (one row "
                "per source image covers all of its channels) and re-set up the "
                "project, or correct the dimensions per image in the project view."
            )

        # ---- diagnostics ---------------------------------------------------- #
        def _environment(self) -> List[str]:
            """Environment facts worth having when setup behaves differently on one
            machine than another -- which is exactly the report this exists for."""
            import platform
            lines = [
                f"platform      : {platform.platform()}",
                f"python        : {platform.python_version()}",
            ]
            for mod in ("tifffile", "numpy", "pandas"):
                try:
                    m = __import__(mod)
                    lines.append(f"{mod:<14}: {getattr(m, '__version__', '?')}")
                except Exception:
                    lines.append(f"{mod:<14}: NOT IMPORTABLE")
            try:
                from .metadata import HAS_CZI
                lines.append("czi support   : "
                             + ("yes" if HAS_CZI else "no (aicspylibczi missing)"))
            except Exception:
                lines.append("czi support   : unknown")
            return lines

        def _inspect_file(self, name: str) -> str:
            """Why one raw file can or cannot be used, in a single line.

            This is the part that identifies a cause. Shape, channel count and
            resolved scale together distinguish "not a TIFF at all" from "no
            channel to extract" from "junk resolution tag", which the old message
            lumped into one unexplained empty folder.
            """
            path = os.path.join(self.raw_dir, name)
            try:
                size = os.path.getsize(path)
            except OSError as exc:
                return f"unreadable ({exc})"
            bits = [f"{size / 1e6:.2f} MB"]

            if name.lower().endswith(".czi"):
                try:
                    from .metadata import HAS_CZI, MetadataExtractor
                    if not HAS_CZI:
                        return f"{bits[0]}, .czi but aicspylibczi is NOT installed"
                    bits.append(f"channels={MetadataExtractor.get_channel_count(path)}")
                except Exception as exc:
                    return f"{bits[0]}, czi inspection failed: {exc}"
                return ", ".join(bits)

            try:
                import tifffile as tiff
                with tiff.TiffFile(path) as tf:
                    bits.append(f"shape={tf.series[0].shape}")
                    bits.append(f"dtype={tf.series[0].dtype}")
                    bits.append("imagej=" + ("yes" if tf.imagej_metadata else "no"))
                    bits.append("ome=" + ("yes" if tf.ome_metadata else "no"))
                from .metadata import MetadataExtractor
                bits.append(f"channels={MetadataExtractor.get_channel_count(path)}")
                meta = MetadataExtractor.read_tiff_metadata(path)
                bits.append(
                    f"scale_found={meta.get('found')} x={float(meta.get('x', 1)):g} "
                    f"y={float(meta.get('y', 1)):g} z={float(meta.get('z', 1)):g}"
                )
            except Exception as exc:
                bits.append(f"NOT READABLE AS TIFF: {type(exc).__name__}: {exc}")
            return ", ".join(bits)

        def _diagnostics(self, targets: Sequence[str],
                         summaries: Sequence[dict]) -> List[str]:
            """Everything setup saw and did, for both the log and the dialog."""
            from .project_selection import classify_path

            lines: List[str] = ["=== HIBACHI project setup diagnostics ==="]
            lines.append("when          : "
                         + datetime.datetime.now().isoformat(timespec="seconds"))
            lines.extend(self._environment())
            lines.append("")
            lines.append(f"raw dir       : {self.raw_dir}")
            lines.append(f"project dir   : {self.project_dir}")
            lines.append(f"wizard mode   : {self.mode}")
            lines.append(f"multichannel  : {self.is_multichannel}")
            lines.append(f"detected mode : {self.mode_filter or 'undetermined'}")
            lines.append(f"max channels  : {self.detect.get('max_channels')}")
            lines.append(f"presets chosen: {self.selections}")

            files = list(self.detect.get("files") or [])
            sidecars = list(self.detect.get("skipped_sidecars") or [])
            lines.append("")
            lines.append(f"raw images    : {len(files)}")
            if sidecars:
                lines.append(f"os sidecars ignored: {len(sidecars)} "
                             f"(e.g. {sidecars[0]})")

            lines.append("")
            lines.append("--- per-file inspection ---")
            for f in files[:20]:
                lines.append(f"  {f}: {self._inspect_file(f)}")
            if len(files) > 20:
                lines.append(f"  ... and {len(files) - 20} more")

            lines.append("")
            lines.append("--- per-channel results ---")
            for summary in summaries:
                lines.append(
                    f"  channel {summary.get('channel_idx')} -> "
                    f"{os.path.basename(str(summary.get('target')))}"
                    f"  mode={summary.get('mode')}"
                )
                lines.append(f"      organized      : "
                             f"{len(summary.get('organized') or [])}")
                lines.append(f"      missing channel: "
                             f"{summary.get('missing_channel') or []}")
                lines.append(f"      sidecars       : "
                             f"{summary.get('skipped_sidecars') or []}")
                lines.append(f"      csv used       : {summary.get('csv')}")
                lines.append(f"      unscaled       : "
                             f"{summary.get('unscaled') or []}")
                for fail in (summary.get("failed") or []):
                    if isinstance(fail, dict):
                        lines.append(
                            f"      FAILED {fail.get('file')}: "
                            f"{fail.get('reason')} "
                            f"(channels detected: {fail.get('channels_detected')})"
                        )
                    else:
                        lines.append(f"      FAILED {fail}")

            lines.append("")
            lines.append("--- resulting folders ---")
            for target in dict.fromkeys(targets):
                lines.append(f"  {target}")
                try:
                    lines.append(f"      classified as: {classify_path(target).kind}")
                except Exception as exc:
                    lines.append(f"      classify error: {exc}")
                try:
                    for sub in sorted(os.listdir(target))[:10]:
                        full = os.path.join(target, sub)
                        if os.path.isdir(full):
                            lines.append(
                                f"      {sub}/ -> {sorted(os.listdir(full))[:6]}")
                        else:
                            lines.append(f"      {sub}")
                except OSError as exc:
                    lines.append(f"      (unreadable: {exc})")
            return lines

        def _write_log(self, lines: Sequence[str]) -> Optional[str]:
            """Write the diagnostics to disk; returns the path, or None if nowhere.

            The project folder is tried first so the log sits with the data it
            describes, then the raw folder, then temp -- a read-only or full volume
            is exactly the sort of condition that causes the failure being
            diagnosed, so the last fallback must not be on the user's drive.
            """
            import tempfile
            body = "\n".join(lines) + "\n"
            name = ("hibachi_setup_log_"
                    + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt")
            for folder in (self.project_dir, self.raw_dir, tempfile.gettempdir()):
                if not folder:
                    continue
                try:
                    path = os.path.join(folder, name)
                    with open(path, "w") as fh:
                        fh.write(body)
                    return path
                except Exception:
                    continue
            return None

        def _verify_created(self, targets: List[str],
                            summaries: Sequence[dict] = ()) -> None:
            """Confirm setup actually produced something openable.

            Every organize step can decline work for individually benign reasons:
            an image that doesn't have the requested channel, a metadata row
            matching no file on disk, an unreadable file. When *all* of them
            decline, the step still returns normally -- so accept() reported
            success, the caller re-classified the folder, found the same loose raw
            images, and re-opened this wizard. That was the endless
            project-creation loop, with no project and no error message.

            When it does fail, the message has to carry the reason. The first
            version named only the empty folders, which told a user nothing about
            *why* they were empty, so the per-file findings and a written log go in
            here too.
            """
            from .project_selection import (  # lazy: pure logic, no Qt at import
                classify_path, MULTICHANNEL_PROJECT, PROJECT,
            )

            openable = {PROJECT, MULTICHANNEL_PROJECT}
            made, failed = [], []
            for target in dict.fromkeys(targets):  # de-duplicated, order kept
                if classify_path(target).kind in openable:
                    made.append(target)
                else:
                    failed.append(target)

            if not made:
                lines = self._diagnostics(targets, summaries)
                log_path = self._write_log(lines)
                for line in lines:
                    print(line)

                # Lead with the actual per-file reasons: they are what identifies
                # the cause. Fall back to the folder list only if there are none.
                reasons: List[str] = []
                for summary in summaries:
                    for fail in (summary.get("failed") or []):
                        if isinstance(fail, dict):
                            reasons.append(
                                f"\u2022 {fail.get('file')}: {fail.get('reason')}")
                        else:
                            reasons.append(f"\u2022 {fail}")
                missing = sorted({
                    f for summary in summaries
                    for f in (summary.get("missing_channel") or [])
                })

                if reasons:
                    uniq = list(dict.fromkeys(reasons))
                    detail = ("None of the images could be extracted:\n"
                              + "\n".join(uniq[:6]))
                    if len(uniq) > 6:
                        detail += f"\n\u2026 and {len(uniq) - 6} more"
                elif missing:
                    detail = (
                        f"{len(missing)} image(s) did not contain the requested "
                        "channel, e.g. " + ", ".join(missing[:3]) + ".\n"
                        "If these images are single-channel, set the project up as "
                        "single-channel instead."
                    )
                else:
                    where = "\n".join(f"\u2022 {t}" for t in (failed or targets))
                    detail = f"Nothing usable was written to:\n{where}"

                log_note = (
                    f"\n\nA diagnostic log was saved to:\n{log_path}\n"
                    "Please send this file to the developer."
                    if log_path else
                    "\n\nDiagnostics were printed to the console."
                )

                raise ValueError(
                    "Setup finished without creating any image folders, so there "
                    f"is no project to open.\n\n{detail}\n\nYour raw images were "
                    f"left untouched.{log_note}"
                )

            # Partial success (e.g. one channel of several came up empty) is worth
            # reporting, but not worth discarding the channels that did work.
            if failed:
                names = "\n".join(f"\u2022 {os.path.basename(t)}" for t in failed)
                QMessageBox.warning(
                    self, "Some channels were empty",
                    f"{len(made)} of {len(made) + len(failed)} channels were set "
                    f"up. No images were written for:\n\n{names}\n\nThis usually "
                    "means the source images don't contain that channel."
                )

    def run_organize_wizard(parent, raw_dir: str, mode: str = "new",
                            project_dir: Optional[str] = None) -> bool:
        """Convenience: build, run, and report whether setup completed."""
        # Surface any broken config files up front so they're an explicit warning
        # rather than silently missing from the preset list.
        try:
            from .config_library import scan_problems
            problems = scan_problems()
        except Exception:
            problems = []
        if problems:
            preview = "\n".join(f"\u2022 {p}\n    {m}" for p, m in problems[:6])
            if len(problems) > 6:
                preview += f"\n\u2026 and {len(problems) - 6} more"
            QMessageBox.warning(
                parent, "Some configs could not be read",
                f"{len(problems)} config file(s) were skipped and won't appear as "
                f"presets:\n\n{preview}"
            )
        wiz = OrganizeWizard(raw_dir, mode=mode, project_dir=project_dir, parent=parent)
        if not wiz.presets:
            QMessageBox.critical(parent, "No presets",
                                 "No configuration presets were found.")
            return False
        return wiz.exec_() == QWizard.Accepted