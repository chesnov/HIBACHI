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
import yaml
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
                 if f.lower().endswith((".tif", ".tiff"))
                 and os.path.isfile(os.path.join(raw_dir, f))]
    except OSError:
        return None
    if not files:
        return None  # e.g. only .czi present -> don't guess

    try:
        with tiff.TiffFile(os.path.join(raw_dir, files[0])) as tf:
            shape = tf.series[0].shape
    except Exception:
        return None

    ndim = len(shape)
    if ndim == 2:
        return "fluorescence_2d"                       # single plane
    if ndim >= 4:
        return "fluorescence"                          # has Z and channel axes
    if ndim == 3:
        # (Z, Y, X) stack vs (C, Y, X) multi-channel plane. Mirror the channel
        # heuristic used elsewhere: a small leading axis (< 10 and < the next)
        # reads as channels -> a 2D image; otherwise it's a Z stack -> 3D.
        if shape[0] < 10 and shape[0] < shape[1]:
            return "fluorescence_2d"
        return "fluorescence"
    return None


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
                self.setPage(0, _DetectPage(self))
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

            # Where each step was supposed to write, so the result can be checked.
            targets: List[str] = []
            summaries: List[dict] = []
            try:
                for i, (ch_idx, preset_key) in enumerate(steps):
                    progress.setValue(i)
                    QApplication.processEvents()
                    preset = self.presets[preset_key]
                    if single:
                        progress.setLabelText("Organizing project…")
                        organize_processing_dir(self.raw_dir, preset)
                        targets.append(self.raw_dir)
                    else:
                        target = os.path.join(
                            self.project_dir, channel_target_name(ch_idx, preset_key)
                        )
                        progress.setLabelText(f"Extracting channel {ch_idx}…")
                        summary = organize_channel_project(
                            raw_files, self.raw_dir, target, ch_idx, preset
                        )
                        targets.append(target)
                        if isinstance(summary, dict):
                            summaries.append(summary)
                progress.setValue(len(steps))
            finally:
                # Without this, a failing step leaves the modal progress dialog
                # on screen underneath the error box (it's parented to the
                # wizard, so going out of scope doesn't destroy it).
                progress.close()

            self._verify_created(targets)
            self._report_unscaled(summaries)

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

        def _verify_created(self, targets: List[str]) -> None:
            """Confirm setup actually produced something openable.

            Every organize step can decline work for individually benign reasons:
            an image that doesn't have the requested channel, a metadata row
            matching no file on disk, an unreadable file. When *all* of them
            decline, the step still returns normally -- so accept() reported
            success, the caller re-classified the folder, found the same loose raw
            images, and re-opened this wizard. That was the endless
            project-creation loop, with no project and no error message.

            Checking the on-disk result closes that hole for good: whatever the
            underlying reason, "nothing was created" now surfaces as a message the
            user can act on instead of a wizard that reappears forever.
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
                where = "\n".join(f"\u2022 {t}" for t in (failed or targets))
                raise ValueError(
                    "Setup finished without creating any image folders, so there "
                    "is no project to open.\n\nNothing usable was written to:\n"
                    f"{where}\n\nYour raw images were left untouched. Check that "
                    "the images can be read, and that any metadata CSV in the "
                    "folder lists filenames matching the files on disk."
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