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

import os
import re
import shutil
from typing import Dict, List, Optional

_RAW_EXTS = (".tif", ".tiff", ".czi")


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
    try:
        for f in sorted(os.listdir(raw_dir)):
            if (f.lower().endswith(_RAW_EXTS)
                    and os.path.isfile(os.path.join(raw_dir, f))):
                files.append(f)
    except OSError:
        pass

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


# --------------------------------------------------------------------------- #
# Qt wizard (only if PyQt5 is present)
# --------------------------------------------------------------------------- #
try:
    from PyQt5.QtCore import Qt  # type: ignore
    from PyQt5.QtWidgets import (  # type: ignore
        QApplication, QComboBox, QFormLayout, QLabel, QMessageBox, QProgressDialog,
        QRadioButton, QVBoxLayout, QButtonGroup, QWizard, QWizardPage, QWidget,
    )
    from .project_scaffolding import (  # type: ignore
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

            # expose as a wizard field so pages can branch
            self.registerField(_F_MULTI, self.rb_multi)

        def nextId(self) -> int:  # noqa: N802
            return self._wiz.page_presets_id

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

            presets = list(self._wiz.presets.keys())
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
            self.selections: Dict[int, str] = {}   # channel_idx -> preset_key
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
            if mode == "add":
                self.setPage(0, _PresetsPage(self))
                self.page_presets_id = 0
            else:
                self.setPage(0, _DetectPage(self))
                self.setPage(1, _PresetsPage(self))

            self.button(QWizard.FinishButton).setText("Organize")

        def accept(self) -> None:  # noqa: N802
            # runs when the user clicks Organize/Finish
            try:
                self._run()
            except Exception as exc:  # pragma: no cover - surfaced to user
                QMessageBox.critical(self, "Setup failed", str(exc))
                return
            super().accept()

        def _run(self) -> None:
            if not self.selections:
                raise ValueError("No presets were chosen.")

            raw_files = list(self.detect["files"])  # type: ignore[index]
            steps = sorted(self.selections.items())
            progress = QProgressDialog("Setting up project…", None, 0, len(steps), self)
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)

            single = (self.mode != "add") and not self.is_multichannel

            for i, (ch_idx, preset_key) in enumerate(steps):
                progress.setValue(i)
                QApplication.processEvents()
                preset = self.presets[preset_key]
                if single:
                    progress.setLabelText("Organizing project…")
                    organize_processing_dir(self.raw_dir, preset)
                else:
                    target = os.path.join(
                        self.project_dir, channel_target_name(ch_idx, preset_key)
                    )
                    progress.setLabelText(f"Extracting channel {ch_idx}…")
                    organize_channel_project(
                        raw_files, self.raw_dir, target, ch_idx, preset
                    )
            progress.setValue(len(steps))

    def run_organize_wizard(parent, raw_dir: str, mode: str = "new",
                            project_dir: Optional[str] = None) -> bool:
        """Convenience: build, run, and report whether setup completed."""
        wiz = OrganizeWizard(raw_dir, mode=mode, project_dir=project_dir, parent=parent)
        if not wiz.presets:
            QMessageBox.critical(parent, "No presets",
                                 "No configuration presets were found.")
            return False
        return wiz.exec_() == QWizard.Accepted
