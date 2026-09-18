"""cross_channel_window: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
from typing import Optional
import traceback
import yaml  # type: ignore
import numpy as np
import pandas as pd
import tifffile as tiff  # type: ignore
import napari  # type: ignore
from PyQt5.QtCore import Qt, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QMessageBox, QMainWindow, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QWidget, QLabel, QInputDialog, QComboBox,
    QDialog, QDialogButtonBox, QCheckBox, QFormLayout, QGroupBox, QDoubleSpinBox,
    QSizePolicy, QDockWidget
)
from .relational_engine import RelationalEngine

from .metadata import get_sample_metadata


PREVIOUS_RESULT = "PREVIOUS_RESULT"


def _wide_combo(minimum_chars=26):
    """A QComboBox that shows its contents instead of collapsing.

    QComboBox defaults to AdjustToMinimumContentsLengthWithIcon, so inside a
    QFormLayout its sizeHint is roughly one icon wide and the text only becomes
    readable once the popup opens.
    """
    box = QComboBox()
    box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
    box.setMinimumContentsLength(minimum_chars)
    box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    return box


class _RelateDialog(QDialog):
    """One dialog for one relation step, replacing a chain of prompts.

    Building a single overlap step used to take four separate modal prompts:
    checking two channels queued a step per channel, and each asked for a
    partner and then which side was primary. Intersection took up to three of
    its own (labelling mode, then ID preservation, then a follow-up asking
    whether to add the overlap step that would recompute the same geometry).
    None of those questions needed to be sequential -- they are all facets of
    one decision -- so they are one form, with the consequences written out
    underneath as they are chosen.

    `kind` is "overlap" or "distance". Distance shows only the two combo boxes,
    because there is nothing else to decide about it.
    """

    def __init__(self, parent, kind, choices, has_previous, size_word):
        super().__init__(parent)
        self.kind = kind
        self._choices = list(choices)
        if has_previous:
            self._choices.append((PREVIOUS_RESULT, "Previous result"))

        self.setWindowTitle("Overlap" if kind == "overlap" else "Distance")
        self.setMinimumWidth(460)
        outer = QVBoxLayout(self)

        lead = ("Measure how much of one channel sits inside another."
                if kind == "overlap" else
                "Measure how far each object is from its nearest partner.")
        _lead = QLabel(lead)
        _lead.setWordWrap(True)
        outer.addWidget(_lead)

        form = QFormLayout()
        self.cb_primary = _wide_combo()
        self.cb_partner = _wide_combo()
        for key, disp in self._choices:
            self.cb_primary.addItem(disp, key)
        form.addRow("Primary (one row per object):", self.cb_primary)
        form.addRow("Partner:", self.cb_partner)
        outer.addLayout(form)

        if kind == "overlap":
            box = QGroupBox("What to produce")
            bl = QVBoxLayout(box)
            self.chk_coverage = QCheckBox(
                "Coverage percentages (per object, plus a per-sample summary)")
            self.chk_coverage.setChecked(True)
            self.chk_regions = QCheckBox(
                "Size and shape of each overlap region")
            self.chk_keep = QCheckBox(
                "Keep the overlap as a mask for later steps in this recipe")
            for w in (self.chk_coverage, self.chk_regions, self.chk_keep):
                bl.addWidget(w)

            # Labelling only matters if something downstream will use the mask,
            # so it stays hidden until then instead of being asked up front.
            self.cb_label = _wide_combo()
            self.lbl_label = QLabel("Label the overlap by:")
            lf = QFormLayout()
            lf.addRow(self.lbl_label, self.cb_label)
            bl.addLayout(lf)
            outer.addWidget(box)
        else:
            self.chk_coverage = self.chk_regions = self.chk_keep = None
            self.cb_label = None

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("color: #555; padding-top: 6px;")
        outer.addWidget(self.summary)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        outer.addWidget(self.buttons)

        self.cb_primary.currentIndexChanged.connect(self._refresh_partners)
        self.cb_partner.currentIndexChanged.connect(self._refresh)
        if kind == "overlap":
            for w in (self.chk_coverage, self.chk_regions, self.chk_keep):
                w.toggled.connect(self._refresh)
        self._refresh_partners()

    # -- wiring ------------------------------------------------------------
    def _refresh_partners(self):
        """Partner list excludes whatever is currently primary.

        Cheaper than letting both sides hold the same channel and then
        explaining the error afterwards.
        """
        primary = self.cb_primary.currentData()
        keep = self.cb_partner.currentData()
        self.cb_partner.blockSignals(True)
        self.cb_partner.clear()
        for key, disp in self._choices:
            if key != primary:
                self.cb_partner.addItem(disp, key)
        if keep is not None:
            idx = self.cb_partner.findData(keep)
            if idx >= 0:
                self.cb_partner.setCurrentIndex(idx)
        self.cb_partner.blockSignals(False)
        self._refresh()

    def _refresh(self):
        a = self.cb_primary.currentText() or "?"
        b = self.cb_partner.currentText() or "?"

        if self.kind == "distance":
            self.summary.setText(
                f"Distance from each {a} object to its nearest {b}, edge to edge.")
            self.buttons.button(QDialogButtonBox.Ok).setEnabled(
                self.cb_partner.count() > 0)
            return

        if self.cb_label is not None:
            cur = self.cb_label.currentIndex()
            self.cb_label.clear()
            self.cb_label.addItem("A number per overlap region", "connected")
            self.cb_label.addItem("One ID for all overlap", "binary")
            self.cb_label.addItem(f"{a}'s object IDs", "parent_a")
            self.cb_label.addItem(f"{b}'s object IDs", "parent_b")
            if cur >= 0:
                self.cb_label.setCurrentIndex(cur)
            visible = self.chk_keep.isChecked()
            self.cb_label.setVisible(visible)
            self.lbl_label.setVisible(visible)

        lines = []
        if self.chk_coverage.isChecked():
            lines.append(f"\u2022 what % of {a} lies inside {b}, and the reverse")
        if self.chk_regions.isChecked():
            lines.append(f"\u2022 the size and shape of each {a}\u2229{b} region")
        if self.chk_keep.isChecked():
            lines.append(f"\u2022 {a}\u2229{b} becomes the input to the next step")
        self.summary.setText("\n".join(lines) if lines
                             else "Nothing selected \u2014 this step would do nothing.")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(
            bool(lines) and self.cb_partner.count() > 0)

    # -- result ------------------------------------------------------------
    def step(self):
        """The recipe step, named so the list reads as what it does."""
        a_key = self.cb_primary.currentData()
        b_key = self.cb_partner.currentData()
        a, b = self.cb_primary.currentText(), self.cb_partner.currentText()

        if self.kind == "distance":
            return {"type": "relate", "primary": a_key, "target": b_key,
                    "measure_coverage": False, "measure_distance": True,
                    "measure_regions": False, "keep_mask": False,
                    "name": f"Distance: {a} \u2192 nearest {b}"}

        label_mode = self.cb_label.currentData() or "connected"
        parts = []
        if self.chk_coverage.isChecked():
            parts.append(f"% of {a} in {b}")
        if self.chk_regions.isChecked():
            parts.append("region sizes")
        if self.chk_keep.isChecked():
            parts.append("kept as mask")
        return {
            "type": "relate", "primary": a_key, "target": b_key,
            "measure_coverage": self.chk_coverage.isChecked(),
            "measure_distance": False,
            "measure_regions": self.chk_regions.isChecked(),
            "keep_mask": self.chk_keep.isChecked(),
            "label_mode": label_mode,
            # Asking to keep a parent's IDs IS asking to preserve them; it was a
            # second modal prompt for a question the first one had answered.
            "preserve_ids": label_mode in ("parent_a", "parent_b"),
            "name": f"Overlap {a} & {b} \u2014 " + ", ".join(parts),
        }


class _FilterDialog(QDialog):
    """Size filter: source and threshold in one form rather than two prompts."""

    def __init__(self, parent, choices, has_previous, size_word, unit):
        super().__init__(parent)
        self.setWindowTitle("Size filter")
        self.setMinimumWidth(420)
        outer = QVBoxLayout(self)
        form = QFormLayout()

        self.cb_source = _wide_combo()
        if has_previous:
            self.cb_source.addItem("Previous result", PREVIOUS_RESULT)
        for key, disp in choices:
            self.cb_source.addItem(disp, key)
        form.addRow("Filter:", self.cb_source)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(0.0, 1e9)
        self.spin.setDecimals(2)
        self.spin.setValue(10.0)
        self.spin.setSuffix(f" {unit}")
        form.addRow(f"Minimum {size_word.lower()}:", self.spin)
        outer.addLayout(form)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)
        self.cb_source.setEnabled(self.cb_source.count() > 1)

    def step(self, unit):
        key = self.cb_source.currentData()
        disp = self.cb_source.currentText()
        # 'min_vol' is kept as the key for recipes already saved to disk.
        return {"type": "filter", "min_vol": self.spin.value(),
                "size_unit": unit,
                "input": None if key == PREVIOUS_RESULT else key,
                "name": f"Filter {disp}: keep \u2265 {self.spin.value():g} {unit}"}



def project_is_2d_for(sample_registry) -> bool:
    """True when a project's images are planes rather than stacks.

    Free function so the recipe panel can answer it without owning a project
    manager. Rank comes from a sample config, not from array shapes: a 3-axis
    array can be (Z, Y, X) or (C, Y, X), so the array alone cannot settle it.
    """
    for sample_data in (sample_registry or {}).values():
        for ch_path in sample_data.values():
            try:
                dims, _mode = get_sample_metadata(ch_path)
            except Exception:
                continue
            if dims:
                return dims.get("z") is None
    return False


class _RecipeLibraryDialog(QDialog):
    """Browse, load, save, rename, delete and share saved recipes.

    Deliberately one dialog rather than a scatter of buttons: managing a
    library is its own task, and the dock should stay a recipe builder.
    """

    def __init__(self, parent, current_steps, available_channels=()):
        super().__init__(parent)
        self._current = list(current_steps or [])
        self._available = list(available_channels or [])
        self.chosen_steps = None          # set when the user loads one

        self.setWindowTitle("Recipe library")
        self.setMinimumWidth(560)
        outer = QVBoxLayout(self)

        lead = QLabel(
            "Recipes saved here are available in every project. A recipe names "
            "channels by number, so it applies to any project with those "
            "channels.")
        lead.setWordWrap(True)
        outer.addWidget(lead)

        self.listw = QListWidget()
        self.listw.itemDoubleClicked.connect(self._load)
        self.listw.currentRowChanged.connect(self._refresh)
        outer.addWidget(self.listw)

        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color: #555;")
        outer.addWidget(self.detail)

        row = QHBoxLayout()
        self.btn_load = QPushButton("Load")
        self.btn_save = QPushButton("Save current\u2026")
        self.btn_rename = QPushButton("Rename\u2026")
        self.btn_delete = QPushButton("Delete")
        self.btn_import = QPushButton("Import\u2026")
        self.btn_export = QPushButton("Export\u2026")
        self.btn_load.clicked.connect(self._load)
        self.btn_save.clicked.connect(self._save)
        self.btn_rename.clicked.connect(self._rename)
        self.btn_delete.clicked.connect(self._delete)
        self.btn_import.clicked.connect(self._import)
        self.btn_export.clicked.connect(self._export)
        for b in (self.btn_load, self.btn_save, self.btn_rename,
                  self.btn_delete, self.btn_import, self.btn_export):
            row.addWidget(b)
        outer.addLayout(row)

        close = QDialogButtonBox(QDialogButtonBox.Close)
        close.rejected.connect(self.reject)
        outer.addWidget(close)

        self._reload()

    # -- state -------------------------------------------------------------
    def _reload(self):
        from . import recipe_library as rl
        self.entries = rl.list_library()
        self.listw.clear()
        for e in self.entries:
            self.listw.addItem(e.label)
        if self.entries:
            self.listw.setCurrentRow(0)
        # A file that is in the folder but is not a recipe is reported rather
        # than silently skipped, so a misfiled config does not just vanish.
        problems = rl.scan_problems()
        if problems:
            self.listw.addItem(QListWidgetItem(
                f"({len(problems)} file(s) here are not recipes)"))
        self._refresh()

    def _selected(self):
        i = self.listw.currentRow()
        return self.entries[i] if 0 <= i < len(self.entries) else None

    def _refresh(self, *_):
        from . import recipe_library as rl
        e = self._selected()
        has = e is not None
        for b in (self.btn_load, self.btn_rename, self.btn_delete,
                  self.btn_export):
            b.setEnabled(has)
        self.btn_save.setEnabled(bool(self._current))
        if not has:
            self.detail.setText(
                "" if self.entries else "No saved recipes yet.")
            return
        try:
            steps = rl.load(e)
        except rl.RecipeLibraryError as exc:
            self.detail.setText(f"Unreadable: {exc}")
            return
        lines = [f"\u2022 {s.get('name', s.get('type'))}" for s in steps]
        missing = rl.missing_channels(steps, self._available)
        if missing:
            lines.append("")
            lines.append("This project has no " + ", ".join(missing)
                         + " \u2014 those steps would be skipped.")
        self.detail.setText("\n".join(lines))

    # -- actions -----------------------------------------------------------
    def _load(self, *_):
        from . import recipe_library as rl
        e = self._selected()
        if e is None:
            return
        try:
            steps = rl.load(e)
        except rl.RecipeLibraryError as exc:
            QMessageBox.warning(self, "Could not load", str(exc))
            return
        if self._current and QMessageBox.question(
                self, "Replace the current recipe?",
                f"The dock holds {len(self._current)} step(s). Replace them "
                f"with '{e.name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return
        self.chosen_steps = steps
        self.accept()

    def _save(self, *_):
        from . import recipe_library as rl
        name, ok = QInputDialog.getText(self, "Save recipe", "Name:")
        if not ok or not name.strip():
            return
        try:
            rl.save(self._current, name)
        except FileExistsError:
            if QMessageBox.question(
                    self, "Replace?", f"'{name}' already exists. Replace it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No) != QMessageBox.Yes:
                return
            rl.save(self._current, name, overwrite=True)
        except rl.RecipeLibraryError as exc:
            QMessageBox.warning(self, "Could not save", str(exc))
            return
        self._reload()

    def _rename(self, *_):
        from . import recipe_library as rl
        e = self._selected()
        if e is None:
            return
        name, ok = QInputDialog.getText(self, "Rename recipe", "New name:",
                                        text=e.name)
        if not ok or not name.strip():
            return
        try:
            rl.rename(e, name)
        except (FileExistsError, OSError) as exc:
            QMessageBox.warning(self, "Could not rename", str(exc))
            return
        self._reload()

    def _delete(self, *_):
        from . import recipe_library as rl
        e = self._selected()
        if e is None:
            return
        if QMessageBox.question(
                self, "Delete recipe", f"Delete '{e.name}'?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return
        try:
            rl.delete(e)
        except OSError as exc:
            QMessageBox.warning(self, "Could not delete", str(exc))
            return
        self._reload()

    def _import(self, *_):
        from . import recipe_library as rl
        from PyQt5.QtWidgets import QFileDialog
        # Defaults to the project's results folder: every run writes its
        # recipe.yaml there, so "reuse what I ran last month" is this button.
        path, _f = QFileDialog.getOpenFileName(
            self, "Import a recipe (or a run's recipe.yaml)", "",
            "Recipes (*.yaml *.yml)")
        if not path:
            return
        try:
            rl.import_file(path)
        except FileExistsError:
            base = os.path.splitext(os.path.basename(path))[0]
            if QMessageBox.question(
                    self, "Replace?", f"'{base}' already exists. Replace it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No) != QMessageBox.Yes:
                return
            rl.import_file(path, overwrite=True)
        except (rl.RecipeLibraryError, OSError) as exc:
            QMessageBox.warning(self, "Could not import", str(exc))
            return
        self._reload()

    def _export(self, *_):
        from . import recipe_library as rl
        from PyQt5.QtWidgets import QFileDialog
        e = self._selected()
        if e is None:
            return
        path, _f = QFileDialog.getSaveFileName(
            self, "Export recipe", f"{e.name}.yaml", "Recipes (*.yaml)")
        if not path:
            return
        try:
            rl.export(e, path)
        except OSError as exc:
            QMessageBox.warning(self, "Could not export", str(exc))


class RecipePanel(QWidget):
    """The ordered list of steps, and the buttons that add to it.

    Extracted so the recipe stack is a widget rather than window state. It is
    the only part of the cross-channel analyzer that the main window does not
    already have a better version of: selection lives in the project tree, and
    viewing results lives in the overlay the tree already opens. Everything
    here is host-agnostic -- what the channels are, and whether the project is
    2D, arrive as callables -- so the same widget serves the standalone
    analyzer and a dock on the main window.

    Owns `recipe_steps`. Hosts should read `steps()` rather than keeping their
    own copy, so there is one recipe rather than two that can disagree.
    """

    changed = pyqtSignal()
    runRequested = pyqtSignal()

    def __init__(self, channel_provider, is_2d_provider, parent=None,
                 show_run=False):
        super().__init__(parent)
        self._channel_provider = channel_provider
        self._is_2d_provider = is_2d_provider
        self.recipe_steps = []
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        self.recipe_list = QListWidget()
        self.recipe_list.setAlternatingRowColors(True)
        root.addWidget(self.recipe_list)

        add_row = QHBoxLayout()
        self.btn_overlap = QPushButton("+ Overlap")
        self.btn_dist = QPushButton("+ Distance")
        self.btn_filter = QPushButton("+ Size Filter")
        self.btn_overlap.setToolTip(
            "How much of one channel sits inside another. Choose any of: "
            "coverage percentages, the size of each overlap region, or keeping "
            "the overlap as a mask for later steps.")
        self.btn_dist.setToolTip(
            "How far each object is from its nearest partner, edge to edge.")
        self.btn_filter.setToolTip("Drop objects below a size threshold.")
        self.btn_overlap.clicked.connect(lambda: self.add_relate_step("overlap"))
        self.btn_dist.clicked.connect(lambda: self.add_relate_step("distance"))
        self.btn_filter.clicked.connect(self.add_filter_step)
        for b in (self.btn_overlap, self.btn_dist, self.btn_filter):
            add_row.addWidget(b)
        root.addLayout(add_row)

        edit_row = QHBoxLayout()
        self.btn_remove = QPushButton("Remove")
        self.btn_up = QPushButton("Up")
        self.btn_down = QPushButton("Down")
        self.btn_clear = QPushButton("Clear")
        self.btn_remove.clicked.connect(self.remove_step)
        self.btn_up.clicked.connect(lambda: self.move_step(-1))
        self.btn_down.clicked.connect(lambda: self.move_step(1))
        self.btn_clear.clicked.connect(self.clear_steps)
        for b in (self.btn_remove, self.btn_up, self.btn_down, self.btn_clear):
            edit_row.addWidget(b)
        root.addLayout(edit_row)

        # The run belongs with the buttons that build the recipe, not off in a
        # menu: everything else about a recipe is done here. Optional because
        # the standalone analyzer has its own batch button.
        self.btn_library = QPushButton("Recipes\u2026")
        self.btn_library.setToolTip(
            "Saved recipes: load one, save this one, or import the recipe.yaml "
            "from a previous run.")
        self.btn_library.clicked.connect(self.open_library)
        edit_row.addWidget(self.btn_library)

        self.btn_run = QPushButton("Run on checked images\u2026")
        self.btn_run.setToolTip(
            "Run this recipe on every image and region checked in the project "
            "tree.")
        self.btn_run.clicked.connect(self.runRequested.emit)
        self.btn_run.setVisible(bool(show_run))
        root.addWidget(self.btn_run)

        self._refresh()

    # ---- state ----------------------------------------------------------- #
    def steps(self):
        return list(self.recipe_steps)

    def set_steps(self, steps):
        self.recipe_steps = [dict(s) for s in (steps or [])]
        self._rebuild_list()

    def _rebuild_list(self):
        self.recipe_list.clear()
        for s in self.recipe_steps:
            self.recipe_list.addItem(s.get("name", s.get("type", "step")))
        self._refresh()

    def _refresh(self):
        """Empty-state hint, and buttons that are off when they cannot act."""
        empty = not self.recipe_steps
        if empty and self.recipe_list.count() == 0:
            hint = QListWidgetItem(
                "No steps yet \u2014 add one below. Order matters: a step can "
                "use the previous step's result.")
            hint.setFlags(Qt.NoItemFlags)
            self.recipe_list.addItem(hint)
        for b in (self.btn_remove, self.btn_up, self.btn_down, self.btn_clear):
            b.setEnabled(not empty)
        # Enabled on having a recipe; the host disables it further when nothing
        # is checked in the tree, which it is the only one that can see.
        self.btn_run.setEnabled(not empty)
        # Always available: the library is how an empty dock gets a recipe.
        self.btn_library.setEnabled(True)
        self.changed.emit()

    # ---- inputs ----------------------------------------------------------- #
    def channel_choices(self):
        return list(self._channel_provider() or [])

    def size_unit(self):
        return "um\u00b2" if self._is_2d_provider() else "um\u00b3"

    def size_word(self):
        return "Area" if self._is_2d_provider() else "Volume"

    def has_previous_result(self):
        """True when an earlier step leaves a mask for this one to act on."""
        return any(s.get('type') == 'filter'
                   or (s.get('type') in ('relate', 'intersect')
                       and s.get('keep_mask', s.get('type') == 'intersect'))
                   for s in self.recipe_steps)

    # ---- editing ---------------------------------------------------------- #
    def add_relate_step(self, kind):
        choices = self.channel_choices()
        has_prev = self.has_previous_result()
        if len(choices) + (1 if has_prev else 0) < 2:
            QMessageBox.warning(self, "Not enough inputs",
                                "This needs two things to compare: either two "
                                "channels, or one channel and a result from an "
                                "earlier step.")
            return
        dlg = _RelateDialog(self, kind, choices, has_prev, self.size_word())
        if dlg.exec_() != QDialog.Accepted:
            return
        self._append(dlg.step())

    def add_filter_step(self):
        choices = self.channel_choices()
        if not choices and not self.has_previous_result():
            QMessageBox.warning(self, "No channels",
                                "No channels are available to filter.")
            return
        dlg = _FilterDialog(self, choices, self.has_previous_result(),
                            self.size_word(), self.size_unit())
        if dlg.exec_() != QDialog.Accepted:
            return
        self._append(dlg.step(self.size_unit()))

    def _append(self, step):
        if not self.recipe_steps:
            self.recipe_list.clear()        # drop the empty-state hint
        self.recipe_steps.append(step)
        self.recipe_list.addItem(step["name"])
        self._refresh()

    def open_library(self):
        """Browse saved recipes; load one over the current steps if chosen."""
        dlg = _RecipeLibraryDialog(
            self, self.recipe_steps,
            [k for k, _d in self.channel_choices()])
        dlg.exec_()
        if dlg.chosen_steps is not None:
            self.set_steps(dlg.chosen_steps)

    def remove_step(self):
        row = self.recipe_list.currentRow()
        if 0 <= row < len(self.recipe_steps):
            self.recipe_steps.pop(row)
            self.recipe_list.takeItem(row)
            self._refresh()

    def move_step(self, delta):
        row = self.recipe_list.currentRow()
        new = row + delta
        if 0 <= row < len(self.recipe_steps) and 0 <= new < len(self.recipe_steps):
            self.recipe_steps[row], self.recipe_steps[new] = \
                self.recipe_steps[new], self.recipe_steps[row]
            self._rebuild_list()
            self.recipe_list.setCurrentRow(new)

    def clear_steps(self):
        if not self.recipe_steps:
            return
        if QMessageBox.question(self, "Clear recipe", "Clear entire recipe?",
                                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.recipe_steps = []
        self.recipe_list.clear()
        self._refresh()


class RecipeDock(QDockWidget):
    """RecipePanel as a dock, for hosting on the main project window."""

    def __init__(self, channel_provider, is_2d_provider, parent=None):
        super().__init__("Cross-channel recipe", parent)
        self.setObjectName("CrossChannelRecipeDock")
        self.panel = RecipePanel(channel_provider, is_2d_provider, self,
                                 show_run=True)
        self.setWidget(self.panel)

    def steps(self):
        return self.panel.steps()

    @property
    def runRequested(self):
        return self.panel.runRequested



def _spacing_from_extents(meta, shape, where=""):
    """Per-axis spacing from total extents, or None if any axis is unusable.

    Both viewer paths used to inline this as
    ``meta.get('z', 1.0) / shape[0]``, so a config with no dimension block --
    or, before `get_sample_metadata` stopped returning a placeholder, one that
    merely looked like it had one -- produced a spacing of 1.0/N microns per
    pixel. That is a fabricated physical scale: the layers render at the wrong
    aspect and any measurement taken off them is wrong by that factor, with
    nothing to indicate it. Returning None lets the caller keep its own
    isotropic default and, crucially, PRINTS the reason -- so an unscaled
    preview is visible in the log instead of looking like a calibrated one.

    Only the display paths use this. The measurement paths (`_resolve_geometry`
    and `roi_sharing.region_geometry`) refuse outright rather than fall back,
    because a wrong spacing there corrupts reported distances.
    """
    if not meta:
        if where:
            print(f"[cross-channel] {where}: no dimension block in the config; "
                  f"layers will not be physically scaled.")
        return None
    axes = ('z', 'y', 'x') if len(shape) == 3 else ('y', 'x')
    out = []
    for axis, count in zip(axes, shape):
        try:
            total = float(meta.get(axis))
        except (TypeError, ValueError):
            total = 0.0
        if not (total > 0 and count):
            if where:
                print(f"[cross-channel] {where}: no usable {axis!r} extent; "
                      f"layers will not be physically scaled.")
            return None
        out.append(total / count)
    return tuple(out)



def _raw_for_display(tif_file: str):
    """A lazily-read view of a raw channel TIFF, for DISPLAY only.

    Returns either a mapped array or, when a display pyramid exists, the list
    of levels napari wants for `multiscale=True`. Callers pass
    `multiscale=isinstance(result, list)`.

    `tiff.memmap`, not `tiff.imread`. One channel of a slide-scanner stack is
    24 GB, and the callers below load one per channel in a loop, so opening a
    four-channel composite meant roughly 96 GB of allocations just to look at
    it. A mapped array lets napari page in the plane it is rendering and
    nothing else, which is why the single-channel viewer in `app_launch` has
    always opened these with `tiff.memmap` and works on the same files. The
    segmentation labels added beside the raw intensity were already
    `np.memmap`; only the intensity was not.

    Mapping fixed the memory but not the time: a plane is 1.86 GB, four
    channels are 7.4 GB per z-step, and the drive this data lives on reads at
    ~25 MB/s. The pyramid is what makes the composite open at all -- zoomed
    out napari reads a ~950 px level per channel instead of a 928-megapixel
    one. Without a pyramid the behaviour is exactly as before, just slow.

    Read-only, and only safe because this is the display path: napari does not
    write to it. The optimizer and the synthetic-null engine also read whole
    images, and deliberately still do -- they operate on the array rather than
    just showing it.

    Falls back to a full read if the file cannot be mapped, which happens for a
    compressed or tiled TIFF, and says so: that is exactly the case where a
    large file will still hurt, so it should not be silent.
    """
    try:
        from .display_pyramid import open_levels
        levels = open_levels(tif_file)
        if levels:
            return levels
    except Exception as exc:
        # A missing or broken preview must never stop the image from opening.
        print(f"  [display] preview unavailable for "
              f"{os.path.basename(tif_file)} ({exc})")
    try:
        return tiff.memmap(tif_file, mode='r')
    except Exception as exc:
        print(f"  [display] {os.path.basename(tif_file)} cannot be memory "
              f"mapped ({exc}); reading it in full instead")
        return tiff.imread(tif_file)



def _display_range(data) -> dict:
    """`contrast_limits` kwarg for add_image, or {} to let napari decide.

    A dict so that failing to compute a range falls back to the previous
    behaviour rather than forcing a wrong one.
    """
    try:
        from .display_pyramid import contrast_limits_for
        limits = contrast_limits_for(data)
    except Exception:
        limits = None
    return {"contrast_limits": list(limits)} if limits else {}



def _prepare_previews(sample_data, parent=None) -> None:
    """Build any missing previews for a sample's channels, with a progress bar.

    Before the viewer exists, deliberately. Adding a full-resolution
    928-megapixel layer freezes napari's main thread while it downscales the
    plane for the GPU and builds a thumbnail, so a composite with even one
    unprepared channel locks the window for minutes. Waiting on a progress bar
    is both faster and visible.
    """
    try:
        from .display_pyramid import build_with_progress
        paths = []
        for ch_path in sample_data.values():
            paths.append(next(
                (os.path.join(ch_path, f) for f in os.listdir(ch_path)
                 if f.lower().endswith(('.tif', '.tiff'))), None))
        build_with_progress([p for p in paths if p], parent=parent)
    except Exception as exc:
        # The sample still opens without previews, just slowly.
        print(f"  [display] could not prepare previews ({exc})")


def _safe_name(name: str) -> str:
    """Folder-safe form of a region name, e.g. 'ROI 2' -> 'ROI_2'.

    Region results are written to a subfolder named after the region, so the same
    recipe run on the full image and on a region cannot overwrite each other.
    """
    return "".join(c if (c.isalnum() or c in "-_") else "_"
                   for c in str(name)).strip("_") or "region"


# The standalone CrossChannelAnalyzerWindow lived here. It was a second copy of
# the project window's job: its own sample dropdown, its own region dropdown and
# its own channel checkboxes, alongside a Preview button that RE-RAN a recipe in
# order to show you anything. The project window already has a checkable tree of
# samples, channels and regions, an analyses picker and an overlay viewer, so
# what remained unique here was the recipe itself -- which is now RecipePanel,
# hosted as a dock on that window.
#
# Everything the window did survives: building a recipe (RecipePanel), running
# one (run_relational_recipe, scoped by the tree's checked leaves rather than
# "every sample"), and viewing results (the tree's own overlay, now filtered to
# the channels the analysis used).





# ============================================================================
# Running a recipe. Scope is a list of targets rather than "every sample", so
# the caller decides -- the analyzer passes every sample, and the main window
# passes whatever is checked in the project tree, which can mix full images
# and regions freely.
# ============================================================================

def geometry_for(sample_data: dict, roi_name):
    """(shape, spacing) for a sample, honouring the selected region.

    A region's masks are the CROP, so its shape and spacing must come from the
    region rather than the channel's full-resolution TIFF -- memmapping a
    region's mask against the full shape would read past the end of the file.
    Returns (None, None) when it cannot be determined.
    """
    first_ch = list(sample_data.values())[0]
    if roi_name:
        from .roi_sharing import region_geometry
        geo = region_geometry(first_ch, roi_name)
        if geo is None:
            return None, None
        return geo["shape"], geo["spacing"]

    tif_path = next((os.path.join(first_ch, f) for f in os.listdir(first_ch)
                     if f.lower().endswith((".tif", ".tiff"))), None)
    if not tif_path:
        return None, None
    with tiff.TiffFile(tif_path) as tif:
        shape = tuple(int(s) for s in tif.series[0].shape)
    meta, _mode = get_sample_metadata(first_ch)
    if not meta:
        # No usable dimension block. Previously this fell through to a
        # spacing of 1.0/N per axis, which is a physical scale nobody
        # supplied; every distance the analysis then reported was wrong and
        # still labelled microns. The caller documents (None, None) as
        # "could not be determined" and handles it.
        print(f"[cross-channel] {first_ch}: config has no usable "
              f"dimension block; geometry cannot be determined.")
        return None, None

    # A (C, Z, Y, X) or (C, Y, X) file would otherwise hand back a shape the
    # .dat memmaps do not have. How many trailing axes are spatial comes from
    # the dimension block's rank -- a 'z' extent means a stack -- because a
    # 3-axis array may be (Z, Y, X) or (C, Y, X) and the array cannot say
    # which. This used to read the mode string, which is now the same for
    # both ranks, so `want` was always 3 and a 2D project's (C, Y, X) file
    # kept its channel axis as if it were Z. Keep exactly the `want` trailing
    # spatial axes and drop any leading (channel) axes. Do NOT squeeze
    # singleton axes generally: a genuine thin-Z volume is (1, Y, X) and must
    # stay 3D, or the 3D viewport and turntable would be lost for
    # single-channel / few-slice samples.
    want = 3 if meta.get('z') is not None else 2
    if len(shape) > want:
        shape = shape[-want:]
    if len(shape) < want:
        print(f"[cross-channel] {first_ch}: config describes a {want}D "
              f"acquisition but the image has shape {shape}; geometry "
              f"cannot be determined.")
        return None, None

    axes = ('z', 'y', 'x') if want == 3 else ('y', 'x')
    spacing = []
    for axis, count in zip(axes, shape):
        try:
            total = float(meta.get(axis))
        except (TypeError, ValueError):
            total = 0.0
        if not (total > 0 and count):
            print(f"[cross-channel] {first_ch}: no usable {axis!r} extent "
                  f"in the config; refusing to invent a spacing.")
            return None, None
        spacing.append(total / count)
    return shape, tuple(spacing)

def targets_from_leaf_keys(leaf_keys):
    """[(sample_name, roi_name)] for the project tree's checked leaves.

    A leaf key is "<sample_folder>" or "<sample_folder>::<region>", and the
    folder is channel-specific -- checking all four channels of one image
    yields four leaves for the same sample. The relational engine works on a
    sample across every channel at once, so they collapse to one target.
    Order is preserved so a run reads in the order the tree shows.
    """
    from .gui_text_utils import clean_filename_for_matching
    from .project_selection import split_leaf_key

    out, seen = [], set()
    for key in leaf_keys or []:
        folder, roi = split_leaf_key(key)
        base = os.path.basename(str(folder).rstrip("/\\"))
        if not base:
            continue
        # The tree keys samples by folder basename; the consolidated registry
        # keys them by the CLEANED name (lowercased, extensions and " #N"
        # scene suffixes stripped) -- see build_consolidated_sample_registry.
        # Passing the raw basename matched nothing, so every target was
        # skipped and the run reported success having done nothing. The
        # analysis output folders are named by the registry key too, so this
        # is also what makes the results findable afterwards.
        sample = clean_filename_for_matching(base)
        if (sample, roi) in seen:
            continue
        seen.add((sample, roi))
        out.append((sample, roi))
    return out


def _clear_result_leaf(leaf_dir):
    """Empty one sample's result folder before writing a fresh run into it.

    Re-running under an analysis name that already exists used to leave the
    previous run's files in place, because a run only overwrites the files it
    happens to produce. Change the labelling between runs and the two produce
    DIFFERENT filenames, so both survive -- and the viewer, which lists every
    `.dat` it finds, then showed two derived masks for one step with no way to
    tell which run each came from. Reading the stale one is reading the old
    settings.

    Scoped hard: only a directory inside RELATIONAL_ANALYSIS is touched, and
    only its files, so a mistaken path cannot delete anything that was not
    generated. Other samples' leaves are untouched, so adding a sample to an
    existing analysis still works.
    """
    if not os.path.isdir(leaf_dir):
        return
    if "RELATIONAL_ANALYSIS" not in os.path.abspath(leaf_dir).split(os.sep):
        print(f"  [Warn] refusing to clear {leaf_dir}: not a results folder")
        return
    for name in os.listdir(leaf_dir):
        path = os.path.join(leaf_dir, name)
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError as exc:
            print(f"  [Warn] could not remove stale {name}: {exc}")


def run_relational_recipe(pm, recipe_steps, analysis_name, targets, parent=None):
    """Run `recipe_steps` over `targets`, returning the output folder or None.

    `targets` is [(sample_name, roi_name_or_None)]. Mixing full images and
    regions in one run is allowed and normal: each target writes to its own
    leaf under <analysis>/<sample>[/<region>]/, which is the layout
    list_relational_analyses already reads back.
    """
    if not recipe_steps or not targets:
        return None

    project_root = os.path.dirname(pm.project_path)
    batch_out_dir = os.path.join(project_root, "RELATIONAL_ANALYSIS", analysis_name)
    os.makedirs(batch_out_dir, exist_ok=True)

    with open(os.path.join(batch_out_dir, "recipe.yaml"), "w") as fh:
        yaml.dump(list(recipe_steps), fh)

    # Replaces the old single-line region.txt, which recorded ONE region for
    # the whole analysis. That was true only while a run was "all samples, one
    # region"; a run over checked leaves can span several regions and full
    # images at once, and a file claiming otherwise would be worse than none.
    # Nothing ever read region.txt, so this loses no consumer.
    try:
        with open(os.path.join(batch_out_dir, "targets.txt"), "w") as fh:
            for sample, roi in targets:
                fh.write(f"{sample}\t{roi or 'Full image'}\n")
    except OSError:
        pass

    total = len(targets)
    QApplication.setOverrideCursor(Qt.WaitCursor)
    print(f"\n{'='*60}\nSTARTING RELATIONAL RUN: {analysis_name}\n{'='*60}")
    leaf_dirs = []
    try:
        for i, (s_name, roi_name) in enumerate(targets):
            label = s_name + (f" [{roi_name}]" if roi_name else "")
            print(f"Processing {i+1}/{total}: {label}...")
            sample_data = pm.sample_registry.get(s_name)
            if not sample_data:
                print(f"  [Skip] {s_name}: not in the channel registry")
                continue

            shape, spacing = geometry_for(sample_data, roi_name)
            if shape is None:
                print(f"  [Skip] {label}: "
                      + (f"region {roi_name!r} not available in every channel"
                         if roi_name else "no readable image"))
                continue

            sample_out = os.path.join(batch_out_dir, s_name) if not roi_name \
                else os.path.join(batch_out_dir, s_name, _safe_name(roi_name))
            _clear_result_leaf(sample_out)
            leaf_dirs.append((s_name, sample_out))

            RelationalEngine.run_recipe(
                s_name, pm.sample_registry, list(recipe_steps),
                sample_out, shape, spacing, roi_name=roi_name
            )

        QApplication.restoreOverrideCursor()
        print(f"\n{'='*60}\nRUN FINISHED\n{'='*60}")

        # Every target skipped means the run produced nothing. Returning the
        # folder anyway made the caller offer to open results that did not
        # exist, so the first sign of trouble was a confusing "no channels"
        # error from the viewer rather than the actual failure here.
        if not leaf_dirs:
            msg = ("None of the selected images could be analysed. "
                   "Check that they are processed in every channel"
                   + (", and that the selected regions exist in all of them."
                      if any(r for _s, r in targets) else "."))
            print(f"  [Run] {msg}")
            if parent is not None:
                QMessageBox.warning(parent, "Nothing was analysed", msg)
            return None
    except Exception as exc:                                    # noqa: BLE001
        QApplication.restoreOverrideCursor()
        print(f"FATAL ERROR IN RELATIONAL RUN: {exc}")
        traceback.print_exc()
        if parent is not None:
            QMessageBox.critical(parent, "Run error", str(exc))
        return None

    # Master tables, gathered from the leaves this run actually wrote.
    all_csvs, all_summaries = [], []
    for s_name, leaf_dir in leaf_dirs:
        # A run can produce one per-object table per primary, so glob rather
        # than expect a single known filename. The "_coverage" files are the
        # partner view -- a different row grain -- and must not be mixed in.
        for f in sorted(_per_object_files(leaf_dir)):
            df = pd.read_csv(os.path.join(leaf_dir, f))
            df["sample_name"] = s_name
            all_csvs.append(df)
        sum_p = os.path.join(leaf_dir, "overlap_summary.csv")
        if os.path.exists(sum_p):
            all_summaries.append(pd.read_csv(sum_p))

    if all_csvs:
        pd.concat(all_csvs, ignore_index=True).to_csv(
            os.path.join(batch_out_dir, "MASTER_PER_OBJECT.csv"), index=False)
        print("Successfully generated MASTER_PER_OBJECT.csv")
    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(
            os.path.join(batch_out_dir, "MASTER_OVERLAP_SUMMARY.csv"), index=False)
        print("Successfully generated MASTER_OVERLAP_SUMMARY.csv "
              "(one row per sample and channel pair)")

    return batch_out_dir


# ============================================================================
# Reusable cross-channel overlay helpers (module-level, callable from the main
# project window as well as this analyzer).
# ============================================================================

# Separates an analysis name from a region name inside it, matching the
# convention used for slide scenes and tree leaves.
ANALYSIS_SEP = "::"


def make_analysis_key(analysis: str, region_dir: Optional[str] = None) -> str:
    """Identity for one overlay choice: an analysis, optionally scoped to a region."""
    return f"{analysis}{ANALYSIS_SEP}{region_dir}" if region_dir else analysis


def split_analysis_key(key: str):
    """(analysis, region_dir or None) for an overlay key."""
    text = str(key)
    if ANALYSIS_SEP in text:
        analysis, _, region = text.partition(ANALYSIS_SEP)
        return analysis, (region or None)
    return text, None


def _per_object_files(directory: str):
    """Per-object tables in a result folder: the PRIMARY view, one per primary.

    Excludes "..._coverage.csv", which is the partner view at a different row
    grain -- one row per partner object -- and would corrupt a concatenation.
    """
    try:
        return [f for f in os.listdir(directory)
                if f.startswith("per_object_") and f.endswith(".csv")
                and not f.endswith("_coverage.csv")]
    except OSError:
        return []


def _has_result_files(directory: str) -> bool:
    """True if a directory holds relational outputs directly inside it."""
    try:
        return any(f.endswith(".dat") or f.endswith(".csv")
                   for f in os.listdir(directory))
    except OSError:
        return False


def list_relational_analyses(project_root: str):
    """Saved cross-channel analyses under <project_root>/RELATIONAL_ANALYSIS.

    Returns (display, key) pairs. An analysis run on a saved region writes to
    ``<analysis>/<sample>/<region>/`` rather than ``<analysis>/<sample>/``, so one
    analysis name can hold several variants -- the full image, and one per region.
    Listing only the top-level name meant selecting it found nothing at the level
    the overlay looks, or worse, silently showed the FULL-IMAGE result when a
    region was intended. Each variant is therefore its own entry.

    Detected from the directory layout rather than from a recorded label, so an
    analysis name used for both a full image and a region reports both.
    """
    rel_dir = os.path.join(project_root, "RELATIONAL_ANALYSIS")
    if not os.path.isdir(rel_dir):
        return []
    try:
        analyses = sorted(d for d in os.listdir(rel_dir)
                          if os.path.isdir(os.path.join(rel_dir, d)))
    except OSError:
        return []

    out = []
    for analysis in analyses:
        a_dir = os.path.join(rel_dir, analysis)
        has_full = False
        regions = set()
        try:
            samples = [s for s in os.listdir(a_dir)
                       if os.path.isdir(os.path.join(a_dir, s))]
        except OSError:
            samples = []
        for sample in samples:
            s_dir = os.path.join(a_dir, sample)
            if _has_result_files(s_dir):
                has_full = True
            try:
                for sub in os.listdir(s_dir):
                    sub_path = os.path.join(s_dir, sub)
                    if os.path.isdir(sub_path) and _has_result_files(sub_path):
                        regions.add(sub)
            except OSError:
                pass

        # An analysis with no detectable outputs still gets a plain entry, so a
        # run that failed part-way is visible rather than vanishing.
        if has_full or not regions:
            out.append((analysis, make_analysis_key(analysis)))
        for region in sorted(regions):
            # Folder names are slugified ("ROI_2"); show them as drawn ("ROI 2").
            out.append((f"{analysis}  \u203a  {region.replace('_', ' ')}",
                        make_analysis_key(analysis, region)))
    return out


def draw_proximity_bridges(viewer, df, shape, spacing):
    """
    Parse a metrics dataframe for Source/Target coordinates and draw red
    connection lines (bridges) between interacting biological objects.
    """
    partners = [c.replace('src_y_', '') for c in df.columns if c.startswith('src_y_')]
    is_3d = (len(shape) == 3)
    z_scale = spacing[0] / spacing[-1] if is_3d else 1.0
    display_scale = (z_scale, 1, 1) if is_3d else (1, 1)

    for p in partners:
        lines = []
        for _, row in df.iterrows():
            if pd.notna(row.get(f'dist_um_{p}')):
                try:
                    if is_3d:
                        src = [row[f'src_z_{p}'], row[f'src_y_{p}'], row[f'src_x_{p}']]
                        tgt = [row[f'tgt_z_{p}'], row[f'tgt_y_{p}'], row[f'tgt_x_{p}']]
                    else:
                        src = [row[f'src_y_{p}'], row[f'src_x_{p}']]
                        tgt = [row[f'tgt_y_{p}'], row[f'tgt_x_{p}']]
                    lines.append([src, tgt])
                except KeyError:
                    continue
        if lines:
            viewer.add_shapes(
                lines, shape_type='line', edge_color='red',
                edge_width=2 if not is_3d else 1,
                name=f"Bridges to {p}", scale=display_scale, blending='additive',
            )
    print(f"  [Visualizer] Plotted connection bridges for partners: {partners}")


def channels_used_by_recipe(recipe_steps):
    """Channel keys a recipe actually touches, in first-seen order.

    Reads the keys a step can name a channel with: `primary`, `target`,
    `input` (size filter), and the legacy `inputs` pair. PREVIOUS_RESULT is a
    reference to an earlier step, not a channel, so it is skipped.
    """
    out = []
    for step in recipe_steps or []:
        if not isinstance(step, dict):
            continue
        candidates = [step.get('primary'), step.get('target'), step.get('input')]
        candidates.extend(step.get('inputs') or [])
        for c in candidates:
            if (isinstance(c, str) and c and c != PREVIOUS_RESULT
                    and c not in out):
                out.append(c)
    return out


def recipe_for_analysis(analysis_dir):
    """The recipe a saved analysis was produced by, or None.

    Every run writes `recipe.yaml` at the analysis root. Runs made before that
    existed -- and anything hand-assembled -- have none, which callers must
    treat as "no information" rather than "no channels".
    """
    try:
        with open(os.path.join(analysis_dir, "recipe.yaml")) as fh:
            data = yaml.safe_load(fh)
    except (OSError, yaml.YAMLError):
        return None
    return data if isinstance(data, list) else None


def open_sample_overlay(project_manager, sample_name, analysis_name=None,
                        parent=None, channels=None):
    """
    Open a napari viewer for one multi-channel sample.

    Always loads every channel's raw intensity (visible) and its base
    segmentation (added but hidden, so the viewer isn't a mess of overlapping
    label layers — the user can toggle any on). When `analysis_name` is given,
    the cross-channel-specific layers are added on top and shown: the derived
    masks (.dat) from that analysis and the proximity bridges from its metrics.

    `sample_name` is the consolidated-registry key (the clean sample name), which
    is also the analysis output subfolder name. Returns True if a viewer opened.

    With an analysis selected, only the channels that analysis USED are loaded,
    read from its `recipe.yaml`. Opening a two-channel overlap in a four-channel
    project used to add all eight raw and segmentation layers, burying the two
    that the analysis was about. `channels` overrides that with an explicit
    allow-list; an analysis with no recipe on disk falls back to every channel,
    since absence of a recipe is not evidence of which channels were involved.
    """
    pm = project_manager
    if not pm.sample_registry:
        pm.build_consolidated_sample_registry()

    sample_data = pm.sample_registry.get(sample_name)
    if not sample_data:
        QMessageBox.warning(
            parent, "Not Found", f"No channels found for sample '{sample_name}'."
        )
        return False

    sample_out_dir = None
    analysis_label = analysis_name
    region_dir = None
    recipe = None
    if analysis_name:
        # The picker hands over a key, which may name a region inside the analysis.
        analysis_name, region_dir = split_analysis_key(analysis_name)
        project_root = os.path.dirname(pm.project_path)
        sample_out_dir = os.path.join(
            project_root, "RELATIONAL_ANALYSIS", analysis_name, sample_name
        )
        if region_dir:
            sample_out_dir = os.path.join(sample_out_dir, region_dir)
        analysis_label = (f"{analysis_name} \u203a {region_dir.replace('_', ' ')}"
                          if region_dir else analysis_name)
        if not os.path.isdir(sample_out_dir):
            QMessageBox.warning(
                parent, "Not Found",
                f"No data found for sample '{sample_name}' in "
                f"'{analysis_label}'."
                + ("\n\nThis sample may not have that region." if region_dir else "")
            )
            return False

        recipe = recipe_for_analysis(
            os.path.join(project_root, "RELATIONAL_ANALYSIS", analysis_name))

    # Narrow to the channels this view is about. Order follows the registry so
    # colour assignment stays stable for a channel across analyses.
    wanted = list(channels) if channels is not None else channels_used_by_recipe(recipe)
    if wanted:
        shown = {k: v for k, v in sample_data.items() if k in set(wanted)}
        if shown:
            skipped = [k for k in wanted if k not in sample_data]
            if skipped:
                # The recipe names a channel this sample does not have -- a
                # renamed folder, or a sample not processed in that channel.
                print(f"  [overlay] recipe channels missing from {sample_name}: "
                      + ", ".join(skipped))
            sample_data = shown
        else:
            # None of them resolved; showing nothing would be worse than
            # showing everything.
            print(f"  [overlay] none of the recipe's channels matched "
                  f"{sample_name}; showing all channels.")

    title = (f"Overlay: {analysis_label} | {sample_name}"
             if analysis_name else f"Sample: {sample_name}")
    _prepare_previews(sample_data, parent=parent)
    viewer = napari.Viewer(title=title)
    # Any failure below leaves a live napari window (Qt owns it, so it is not
    # collected when this frame unwinds), holding a GL context and the loaded
    # image for the rest of the session. Close it instead.
    #
    # Imported lazily: a module-level import would close the cycle
    # cross_channel_window -> app_launch -> project_view_window -> here.
    from .app_launch import close_viewer_on_error
    with close_viewer_on_error(viewer):
        try:
            _qw = viewer.window._qt_window
            _qw.showMaximized(); _qw.raise_(); _qw.activateWindow()
        except Exception:
            pass
        colormaps = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue']
        shape = None
        spacing = (1.0, 1.0, 1.0)

        # 1. Raw intensity (visible) + base segmentation (hidden, toggle-able).
        for i, (ch_name, ch_path) in enumerate(sample_data.items()):
            tif_file = next((os.path.join(ch_path, f) for f in os.listdir(ch_path)
                             if f.lower().endswith(('.tif', '.tiff'))), None)
            dat_file = RelationalEngine._find_dat(ch_path)

            if shape is None and tif_file:
                with tiff.TiffFile(tif_file) as tif:
                    shape = tif.series[0].shape
                meta, _ = get_sample_metadata(ch_path)
                # Unusable extents leave the isotropic default (display path).
                _sp = _spacing_from_extents(meta, shape, ch_path)
                if _sp is not None:
                    spacing = _sp

            if tif_file:
                raw_img = _raw_for_display(tif_file)
                cmap = colormaps[i % len(colormaps)]
                viewer.add_image(raw_img, name=f"Raw: {ch_name}", colormap=cmap,
                                 blending='additive', opacity=0.5,
                                 multiscale=isinstance(raw_img, list),
                                 **_display_range(raw_img))
            if dat_file:
                seg_data = np.memmap(dat_file, dtype=np.int32, mode='r', shape=shape)
                viewer.add_labels(seg_data, name=f"Seg: {ch_name}", opacity=0.3,
                                  visible=False)

        # 2. Cross-channel-specific layers (only for a selected analysis; shown).
        if sample_out_dir:
            for f in [x for x in os.listdir(sample_out_dir) if x.endswith('.dat')]:
                try:
                    data = np.memmap(os.path.join(sample_out_dir, f), dtype=np.int32, mode='r', shape=shape)
                    viewer.add_labels(data, name=f"DERIVED: {f.replace('.dat', '')}")
                except Exception as e:
                    print(f"Could not load {f}: {e}")

            # Bridges come from a per-object table. With several primaries
            # there are several; draw each, since each holds its own lines.
            for f in sorted(_per_object_files(sample_out_dir)):
                try:
                    draw_proximity_bridges(
                        viewer, pd.read_csv(os.path.join(sample_out_dir, f)),
                        shape, spacing)
                except Exception as e:
                    print(f"Could not draw bridges from {f}: {e}")

        # 3. Viewport for 3D.
        if shape and len(shape) == 3:
            viewer.dims.ndisplay = 3
            z_scale = spacing[0]/spacing[2] if len(spacing) == 3 else 1.0
            for layer in viewer.layers:
                layer.scale = (z_scale, 1, 1)

        # One-click hide/show-all toggle under the layer list (shared with the
        # per-channel segmenter view). Lazy import avoids a circular import.
        try:
            from .app_launch import add_channel_visibility_toggle
            add_channel_visibility_toggle(viewer)
        except Exception as exc:
            print(f"Could not add channel visibility toggle: {exc}")

        # 3D rotation recorder (3D samples only), docked beneath the layer list.
        if shape and len(shape) == 3:
            try:
                from ..fluorescence_module.turntable import add_turntable_button
                add_turntable_button(viewer)
            except Exception as exc:
                print(f"Could not add 3D rotation recorder: {exc}")

        # Shared sub-region controls. Every channel of a sample has the same pixel
        # dimensions, so a polygon drawn here is valid in all of them -- this is the
        # one place an ROI can be defined once and handed to several channels.
        # Requires `shape`, which is the coordinate frame the polygon is stored in.
        if shape:
            try:
                from .roi_overlay_panel import add_overlay_roi_panel
                add_overlay_roi_panel(
                    viewer, sample_name, list(sample_data.values()), shape
                )
            except Exception as exc:
                print(f"Could not add shared ROI panel: {exc}")

        return True