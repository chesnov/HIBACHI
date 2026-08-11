"""
null_dialog.py -- the analyzer's entry point for the mask-preserving spatial null.

Replaces the old standalone "Generate Synthetic Channel" dialog. It reads the
recipe the user has already built, because the meaning of the run depends on it:

  * ONE channel checked, no intersection in the recipe
        Randomise that channel's objects inside its tissue hull (or the whole
        field when edge trimming was off).

  * An INTERSECTION in the recipe
        Randomise the OVERLAP objects, and ask what the domain should be. That
        choice is the null hypothesis wearing a different hat, so the dialog
        says what each option holds fixed rather than presenting four
        equivalent-looking radio buttons:

          Whole field  -- tests the conjunction of everything at once, including
              facts already known (that overlap sits inside both channels). It
              will essentially always reject and says little. Offered, not
              default.
          Channel A    -- holds "overlap is inside A" fixed and asks whether,
              within A, the material sits where chance would put it. Usually the
              question worth asking.
          Channel B    -- the mirror.
          A and B      -- holds both memberships fixed and tests arrangement
              alone. The most conservative option.

Nothing here computes a p-value. A HIBACHI project is normally one biological
replicate whose images are technical replicates, so the dialog produces the raw
export and the diagnostics needed to judge it, and points at the notebook layer
for inference.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

DOMAIN_CHOICES = [
    ("hull", "Tissue hull (recommended for a single channel)",
     "Objects are randomised inside the segmented tissue, excluding background "
     "outside it. Falls back to the whole field if edge trimming was disabled."),
    ("field", "Whole image / field of view",
     "Tests everything at once, including what you already know. It will almost "
     "always reject and tells you little. Use only as a sanity check."),
    ("parent_a", "Inside channel A",
     "Holds 'the objects are inside A' fixed and asks whether, within A, they "
     "sit where chance would put them. Usually the informative choice."),
    ("parent_b", "Inside channel B",
     "The mirror of the above."),
    ("parent_both", "Inside A and B (their overlap)",
     "Holds both memberships fixed and tests arrangement alone. Most "
     "conservative."),
]


def _describe_program_step(step) -> str:
    """One recipe step in words, for the dialog and the manifest."""
    kind = step.get("type")
    if kind == "channel":
        return str(step.get("channel", "")).split("_", 2)[-1]
    if kind == "intersect":
        a = str(step.get("channel") or "previous").split("_", 2)[-1]
        b = str(step.get("channel_b", "")).split("_", 2)[-1]
        return f"{a} ∩ {b}"
    if kind == "filter":
        unit = step.get("size_unit") or "um²/um³"
        return f"keep > {float(step.get('min_size') or 0):g} {unit}"
    return str(kind)


class _NullWorker(QThread):
    """Runs the null off the GUI thread.

    A thread rather than a process: the engine is pure numpy/scipy with no Qt
    and no native segmentation code, and numpy releases the GIL for the heavy
    transforms, so the UI stays responsive. Cancellation is cooperative and
    checked between draws, because a running distance transform cannot be
    interrupted part-way.
    """

    progress = pyqtSignal(dict)
    logline = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, jobs, params, out_dir, project_name, channels):
        super().__init__()
        self._jobs = jobs
        self._params = params
        self._out_dir = out_dir
        self._project = project_name
        self._channels = channels
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from .runner import run_project
            result = run_project(
                self._jobs, self._params, out_dir=self._out_dir,
                project_name=self._project, channels=self._channels,
                log=lambda m: self.logline.emit(str(m)),
                progress_cb=lambda **kw: self.progress.emit(dict(kw)),
                cancel_check=lambda: self._cancel)
            self.finished_ok.emit(result)
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class _NullProgressDialog(QDialog):
    """Two bars plus a console, matching batch_progress_dialog's shape.

    The spinner runs on its own timer so the window is visibly alive even while
    a single long draw produces no events -- otherwise a slow sample looks
    indistinguishable from a hang, which is what prompted this.
    """

    _SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, n_samples: int, n_draws: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Spatial null — running")
        self.setMinimumWidth(560)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)
        self._frame = 0
        self._cancelled = False

        v = QVBoxLayout(self)
        self.lbl_head = QLabel("Preparing…")
        self.lbl_head.setStyleSheet("font-weight: bold;")
        v.addWidget(self.lbl_head)

        self.bar_sample = QProgressBar()
        self.bar_sample.setRange(0, max(1, n_samples))
        self.bar_sample.setFormat("image %v of %m")
        v.addWidget(self.bar_sample)

        self.bar_draw = QProgressBar()
        self.bar_draw.setRange(0, max(1, n_draws))
        self.bar_draw.setFormat("randomisation %v of %m")
        v.addWidget(self.bar_draw)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(150)
        v.addWidget(self.console)

        row = QHBoxLayout()
        row.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        row.addWidget(self.btn_cancel)
        v.addLayout(row)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._spin)
        self._timer.start(120)

    def _spin(self):
        self._frame = (self._frame + 1) % len(self._SPINNER)
        base = self.lbl_head.text().lstrip(self._SPINNER + " ")
        self.lbl_head.setText(f"{self._SPINNER[self._frame]} {base}")

    def _on_cancel(self):
        self._cancelled = True
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.setText("Cancelling…")
        self.append("Cancel requested; stopping after the current randomisation.")

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def append(self, text: str):
        self.console.appendPlainText(text.rstrip())

    def update_progress(self, info: dict):
        phase = info.get("phase")
        sample = info.get("sample", "")
        if phase == "prepare":
            self.lbl_head.setText(f"Reading masks and domains — {sample}")
            self.bar_sample.setValue(int(info.get("sample_index", 0)))
            self.bar_sample.setMaximum(max(1, int(info.get("n_samples", 1))))
        elif phase == "run":
            self.lbl_head.setText(f"Randomising — {sample}")
            self.bar_sample.setMaximum(max(1, int(info.get("n_samples", 1))))
            self.bar_sample.setValue(int(info.get("sample_index", 0)) + 1)
            self.bar_draw.setMaximum(max(1, int(info.get("n_draws", 1))))
            self.bar_draw.setValue(int(info.get("draw", 0)) + 1)

    def close_when_done(self):
        self._timer.stop()


class SpatialNullDialog(QDialog):
    """Configure and launch a spatial-null run over the project's samples."""

    def __init__(self, project_manager, checked_channels: Optional[List[str]] = None,
                 recipe: Optional[List[Dict[str, Any]]] = None,
                 roi_name: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.pm = project_manager
        self.recipe = recipe or []
        self.roi_name = roi_name
        self.project_root = os.path.dirname(self.pm.project_path)

        self.all_channels = sorted(
            {c for chans in self.pm.sample_registry.values() for c in chans})
        self.checked = [c for c in (checked_channels or []) if c in self.all_channels]
        self.intersect_step = self._recipe_intersect_step()
        self.intersect_inputs = self._recipe_intersection()

        self.setWindowTitle("Spatial Null (mask-preserving randomisation)")
        self.setMinimumWidth(620)
        self._build()

    # -- recipe interpretation ------------------------------------------------

    def _recipe_intersection(self) -> List[str]:
        """Channels of the last intersection step, if the recipe has one."""
        step = self._recipe_intersect_step()
        if not step:
            return []
        return [c for c in (step.get("inputs") or []) if c != "PREVIOUS_RESULT"]

    def _recipe_intersect_step(self) -> Optional[Dict[str, Any]]:
        """The last intersect step, kept whole for its label_mode/preserve_ids.

        Those matter: the overlap is recomputed here from the two segmentations
        using the same settings, so the null does not require a relational batch
        to have been run first.
        """
        for step in reversed(self.recipe):
            if step.get("type") == "intersect":
                inputs = [c for c in (step.get("inputs") or [])
                          if c != "PREVIOUS_RESULT"]
                if len(inputs) >= 2:
                    return step
        return None

    def _size_unit(self) -> str:
        """'um²' in 2D, 'um³' in 3D, from a sample's recorded pipeline mode.

        Falls back to 3D wording only if no mode can be read; a 3-axis array is
        ambiguous between (Z, Y, X) and (C, Y, X), so shape is not usable here.
        """
        try:
            from .runner import _mode_of
        except ImportError:
            return "um\u00b3"
        for channels in self.pm.sample_registry.values():
            for path in channels.values():
                mode = _mode_of(path)
                if mode:
                    return "um\u00b2" if str(mode).endswith("_2d") else "um\u00b3"
        return "um\u00b3"

    def _recipe_source_channel(self):
        """Channel a filter-first recipe operates on: the checked one."""
        for ch in self.checked:
            if ch in self.all_channels:
                return ch
        # Fall back to the first real channel offered, so a recipe is still
        # usable when nothing was checked.
        for i in range(self.cb_primary.count()):
            data = self.cb_primary.itemData(i)
            if isinstance(data, str) and not data.startswith("__"):
                return data
        return None

    def _recipe_program(self):
        """The recipe's mask-producing steps, as a program for the runner.

        Only 'intersect' and 'filter' change the mask; analysis steps do not.
        The chain is linear (each step consumes the previous result), which is how
        the cross-channel recipe already works, so a size filter after an
        intersection is expressible without a special case.
        """
        program, names = [], []
        for step in self.recipe:
            kind = step.get("type")
            if kind == "intersect":
                inputs = [c for c in (step.get("inputs") or [])
                          if c != "PREVIOUS_RESULT"]
                if not program:
                    if len(inputs) < 2:
                        return [], ""
                    program.append({
                        "type": "intersect", "channel": inputs[0],
                        "channel_b": inputs[1],
                        "label_mode": step.get("label_mode") or "connected",
                        "preserve_ids": bool(step.get("preserve_ids"))})
                    names = [c.split("_", 2)[-1] for c in inputs[:2]]
                else:
                    if not inputs:
                        return [], ""
                    program.append({
                        "type": "intersect", "channel_b": inputs[0],
                        "label_mode": step.get("label_mode") or "connected",
                        "preserve_ids": bool(step.get("preserve_ids"))})
                    names.append(inputs[0].split("_", 2)[-1])
            elif kind == "filter":
                if not program:
                    # A filter with nothing before it needs a source channel. It
                    # cannot come from the primary selector, which is sitting on
                    # "Recipe result" -- that would be circular. The channel the
                    # user checked in the analyzer is what the recipe operates on,
                    # so that is the source.
                    src = self._recipe_source_channel()
                    if not src:
                        return [], ""
                    program.append({"type": "channel", "channel": src})
                    names = [src.split("_", 2)[-1]]
                program.append({
                    "type": "filter",
                    "min_size": float(step.get("min_vol") or 0.0),
                    # The recipe step records the unit it prompted with; reuse it
                    # rather than re-deriving, so the dialog, the manifest and the
                    # recipe list all read identically.
                    "size_unit": step.get("size_unit") or self._size_unit()})
        if not program:
            return [], ""

        label = " & ".join(names) if names else "recipe"
        sizes = [st["min_size"] for st in program if st["type"] == "filter"]
        if sizes:
            label += f" >{max(sizes):g}"
        return program, label

    @property
    def overlap_label(self) -> str:
        names = [c.split("_", 2)[-1] for c in self.intersect_inputs]
        return (f"Overlap: {names[0]} ∩ {names[1]}  (from recipe)"
                if len(names) > 1 else "Overlap (from recipe)")

    # -- UI -------------------------------------------------------------------

    def _build(self):
        layout = QVBoxLayout(self)

        mode = ("Recipe contains an intersection: the OVERLAP objects will be "
                "randomised." if self.intersect_inputs else
                "No intersection in the recipe: a single channel's objects will "
                "be randomised.")
        head = QLabel(mode)
        head.setWordWrap(True)
        head.setStyleSheet("font-weight: bold;")
        layout.addWidget(head)

        src = QGroupBox("Objects and partner")
        form = QFormLayout(src)

        self.cb_primary = QComboBox()
        self._has_filter = any(st.get("type") == "filter" for st in self.recipe)
        if self._has_filter:
            # A size/volume filter in the recipe is the supported route to
            # randomising a size-restricted population: the threshold then lives
            # in the recipe and is recorded in the run's manifest, instead of
            # being a choice made after the null was already built (which cannot
            # work, because the engine reduces to a nearest distance per draw).
            self.cb_primary.addItem("Recipe result (see below)", "__RECIPE__")
        if self.intersect_inputs:
            # Selectable, not forced: with an intersection in the recipe you may
            # want to randomise the overlap OR either whole channel.
            self.cb_primary.addItem(self.overlap_label, "__OVERLAP__")
        for ch in self.all_channels:
            self.cb_primary.addItem(ch, ch)
        if self._has_filter or self.intersect_inputs:
            self.cb_primary.setCurrentIndex(0)
            self.cb_primary.setToolTip(
                "The overlap is recomputed from the two channels' "
                "segmentations using the recipe's label mode, so no relational "
                "batch has to be run first.")
        elif self.checked:
            self.cb_primary.setCurrentText(self.checked[0])
        form.addRow("Objects to randomise:", self.cb_primary)

        self.lbl_program = QLabel("")
        self.lbl_program.setWordWrap(True)
        self.lbl_program.setStyleSheet("color: grey;")
        form.addRow("", self.lbl_program)

        self.cb_partner = QComboBox()
        self.cb_partner.addItem("None (no cross-distances)")
        self.cb_partner.addItems(self.all_channels)
        self.cb_partner.setToolTip(
            "Held fixed and never moved. Its distance field is computed once, "
            "so cross-distances cost almost nothing per draw.")
        # With an overlap selected the interesting partner is a THIRD channel,
        # not one of the two that formed it -- distance to a constituent is
        # degenerate, since the overlap lies inside it by construction.
        third = [c for c in self.all_channels if c not in self.intersect_inputs]
        if self.intersect_inputs and third:
            self.cb_partner.setCurrentText(third[0])
        elif len(self.checked) > 1:
            self.cb_partner.setCurrentText(self.checked[1])
        elif len(self.intersect_inputs) > 1:
            self.cb_partner.setCurrentText(self.intersect_inputs[1])
        form.addRow("Fixed partner channel:", self.cb_partner)

        self.lbl_partner_warn = QLabel("")
        self.lbl_partner_warn.setWordWrap(True)
        self.lbl_partner_warn.setStyleSheet("color: #b45309;")
        form.addRow("", self.lbl_partner_warn)
        layout.addWidget(src)

        dom = QGroupBox("Domain — this IS the null hypothesis")
        dv = QVBoxLayout(dom)
        self.cb_domain = QComboBox()
        allowed = DOMAIN_CHOICES if self.intersect_inputs else DOMAIN_CHOICES[:2]
        names = [c.split("_", 2)[-1] for c in self.intersect_inputs]
        for key, label, _ in allowed:
            # Substitute the real channel names so the choice is unambiguous.
            if key == "parent_a" and len(names) > 0:
                label = f"Inside {names[0]}"
            elif key == "parent_b" and len(names) > 1:
                label = f"Inside {names[1]}"
            elif key == "parent_both" and len(names) > 1:
                label = f"Inside {names[0]} and {names[1]} (their overlap)"
            self.cb_domain.addItem(label, key)
        dv.addWidget(self.cb_domain)
        self.lbl_domain = QLabel()
        self.lbl_domain.setWordWrap(True)
        self.lbl_domain.setStyleSheet("color: grey;")
        dv.addWidget(self.lbl_domain)

        self.chk_per_parent = QCheckBox(
            "Require each object to fit inside a SINGLE parent object")
        self.chk_per_parent.setToolTip(
            "Off: an object need only lie within the union of the parents, so it "
            "may straddle two — which no real child could do.\n"
            "On: closer to the biology, but much harder to satisfy, and "
            "impossible when an object is larger than every parent. Watch the "
            "unplaced count.")
        dv.addWidget(self.chk_per_parent)

        self.sp_erode = QDoubleSpinBox()
        self.sp_erode.setRange(0.0, 100.0)
        self.sp_erode.setSuffix(" µm")
        self.sp_erode.setToolTip(
            "Optional inward margin, to keep objects clear of the domain edge.")
        row = QHBoxLayout()
        row.addWidget(QLabel("Inward margin:"))
        row.addWidget(self.sp_erode)
        row.addStretch()
        dv.addLayout(row)
        layout.addWidget(dom)

        # Connected and primed only now that every widget the handler touches
        # exists. Doing it earlier raised AttributeError on chk_per_parent,
        # because adding items to the combo fires currentIndexChanged.


        nul = QGroupBox("Null model")
        nf = QFormLayout(nul)

        self.chk_rotate = QCheckBox("Random rotation (uniform, no reflection)")
        self.chk_rotate.setChecked(True)
        self.chk_rotate.setToolTip(
            "Rotation happens in physical space so anisotropic spacing does not "
            "shear objects, and voxel count is restored exactly afterwards. "
            "Reflections are excluded: they change chirality and are not "
            "rotations.")
        nf.addRow(self.chk_rotate)

        self.chk_hardcore = QCheckBox("Objects may not overlap each other")
        self.chk_hardcore.setChecked(True)
        self.chk_hardcore.setToolTip(
            "Real segmented labels are disjoint, so the null should be too.")
        nf.addRow(self.chk_hardcore)

        self.sp_sep = QDoubleSpinBox()
        self.sp_sep.setRange(0.0, 100.0)
        self.sp_sep.setSuffix(" µm")
        nf.addRow("Minimum separation:", self.sp_sep)

        self.sp_ref = QSpinBox()
        self.sp_ref.setRange(19, 2000)
        self.sp_ref.setValue(199)
        self.sp_ref.setToolTip("Draws used to estimate the reference curve.")
        nf.addRow("Reference draws:", self.sp_ref)

        self.sp_test = QSpinBox()
        self.sp_test.setRange(19, 2000)
        self.sp_test.setValue(199)
        self.sp_test.setToolTip(
            "A SECOND, independent set used for the spread around the "
            "reference. Sharing one set would shrink the spread and overstate "
            "significance.")
        nf.addRow("Test draws:", self.sp_test)

        # Direction is a computation-time choice because the two are different
        # quantities, not two views of one.
        self.cb_direction = QComboBox()
        self.cb_direction.addItem(
            "From the randomised objects → nearest partner", "primary")
        self.cb_direction.addItem(
            "From the fixed partner → nearest randomised object", "partner")
        self.cb_direction.setToolTip(
            "Which population the distances are summarised over.\n\n"
            "'From the randomised objects' answers e.g. 'how far is each "
            "aggregate from the nearest microglia'.\n"
            "'From the fixed partner' answers 'how far is each microglia from "
            "the nearest aggregate'.\n\n"
            "These differ whenever the two counts differ. Both are computed and "
            "exported either way; this picks the one reported here and drawn in "
            "the QC images.")
        nf.addRow("Measure distances:", self.cb_direction)

        self.chk_both = QCheckBox("Also export the opposite direction")
        self.chk_both.setChecked(True)
        self.chk_both.setToolTip(
            "Costs one extra distance transform per draw, and means the "
            "direction can be changed later without recomputing. Untick only if "
            "runtime matters and you are certain of the direction.")
        nf.addRow(self.chk_both)

        self.cb_stat = QComboBox()
        self.cb_stat.addItems(["median", "mean", "min"])
        self.cb_stat.setToolTip(
            "Only affects the in-app summary. The export keeps every object, so "
            "this can be changed downstream without re-running anything.")
        nf.addRow("Cross-distance summary:", self.cb_stat)

        self.chk_f = QCheckBox("Compute F (empty space)")
        self.chk_f.setChecked(True)
        self.chk_g = QCheckBox("Compute G (nearest neighbour)")
        self.chk_g.setChecked(True)
        nf.addRow(self.chk_f)
        nf.addRow(self.chk_g)

        self.sp_seed = QSpinBox()
        self.sp_seed.setRange(0, 10 ** 6)
        nf.addRow("Random seed:", self.sp_seed)
        layout.addWidget(nul)

        outg = QGroupBox("Output")
        of = QFormLayout(outg)

        # A project holds one named run per pairing, so results never overwrite
        # one another: randomise A against C, then B against C, then A inside B.
        self.le_name = QLineEdit()
        self.le_name.setToolTip(
            "Names this run's folder under SPATIAL_NULL/. Each pairing gets its "
            "own, so several can coexist in one project. The default is the "
            "next free ordinal with the pairing appended; it stops updating once "
            "you type your own.")
        self._name_edited = False
        # textEdited fires only on user input, not on setText, so the default can
        # keep tracking the selection until the user actually types.
        self.le_name.textEdited.connect(self._on_name_edited)
        of.addRow("Run name:", self.le_name)

        self.lbl_path = QLabel("")
        self.lbl_path.setStyleSheet("color: grey;")
        self.lbl_path.setWordWrap(True)
        of.addRow("", self.lbl_path)

        self.lbl_existing = QLabel("")
        self.lbl_existing.setStyleSheet("color: grey;")
        self.lbl_existing.setWordWrap(True)
        of.addRow("", self.lbl_existing)

        self.chk_qc = QCheckBox("Write QC images (JPG) of the randomisations")
        self.chk_qc.setChecked(False)
        self.chk_qc.setToolTip(
            "One JPG per randomisation, plus one of the real data to compare "
            "against. Each shows the stationary partner, the randomised objects, "
            "and a red segment per object marking the shortest distance the "
            "algorithm measured — the endpoints come from the same distance "
            "transform as the statistic, so the picture verifies the maths.")
        of.addRow(self.chk_qc)

        self.sp_qc = QSpinBox()
        self.sp_qc.setRange(1, 2000)
        self.sp_qc.setValue(10)
        self.sp_qc.setEnabled(False)
        self.sp_qc.setToolTip(
            "Images PER SAMPLE. This multiplies: 398 draws across 20 samples is "
            "~8,000 files and well over a gigabyte. Set it to the full draw "
            "count only if you really want every one.")
        self.lbl_qc = QLabel("")
        self.lbl_qc.setStyleSheet("color: grey;")
        row_qc = QHBoxLayout()
        row_qc.addWidget(QLabel("Images per sample:"))
        row_qc.addWidget(self.sp_qc)
        row_qc.addWidget(self.lbl_qc)
        row_qc.addStretch()
        of.addRow(row_qc)

        self.chk_qc_annotate = QCheckBox("Print the distance beside each line")
        self.chk_qc_annotate.setChecked(True)
        self.chk_qc_annotate.setEnabled(False)
        of.addRow(self.chk_qc_annotate)

        self.chk_csv = QCheckBox("Also write the null table as gzipped CSV")
        of.addRow(self.chk_csv)
        layout.addWidget(outg)

        note = QPlainTextEdit()
        note.setReadOnly(True)
        note.setMaximumHeight(74)
        note.setPlainText(
            "This produces raw per-object distances, not statistics. A project "
            "is normally one biological replicate whose images are technical "
            "replicates, so no test run here would be a biological result. "
            "Pool several projects' exports with hibachi_null_io in a notebook "
            "and do the inference there.")
        layout.addWidget(note)

        # Every signal is connected here, after ALL widgets exist. Connecting
        # mid-build is a latent crash: Qt fires currentIndexChanged while a combo
        # is being populated, so a handler can run before the widgets it touches
        # have been created.
        self._wire_signals()

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.btn_run = QPushButton("Run and export")
        self.btn_run.setStyleSheet(
            "background-color: #2E8B57; color: white; font-weight: bold;")
        buttons.addButton(self.btn_run, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        self.btn_run.clicked.connect(self.run)
        layout.addWidget(buttons)

    @property
    def primary_is_overlap(self) -> bool:
        return self.cb_primary.currentData() == "__OVERLAP__"

    @property
    def primary_is_recipe(self) -> bool:
        return self.cb_primary.currentData() == "__RECIPE__"

    def _refresh_program(self, *_):
        """Show the resolved chain, so what gets randomised is never a guess."""
        if not self.primary_is_recipe:
            self.lbl_program.setText("")
            return
        program, label = self._recipe_program()
        if not program:
            self.lbl_program.setText(
                "The recipe's filter has no channel or intersection before it — "
                "add one, or pick a channel above.")
            return
        self.lbl_program.setText(
            " → ".join(_describe_program_step(st) for st in program)
            + f"      (recorded as '{label}')")

    def _primary_base_name(self) -> str:
        """Channel name(s) only, with any size restriction stripped.

        This is what goes in the manifest as `primary_name`, so a size-restricted
        run still MATCHES the pairing of its unrestricted sibling and the two are
        separated by the recorded program instead. Folding the threshold into the
        pairing name would make the restricted run invisible to a pairing lookup.
        """
        if self.primary_is_recipe:
            program, _ = self._recipe_program()
            names = []
            for st in program:
                for key in ("channel", "channel_b"):
                    ch = st.get(key)
                    if ch:
                        names.append(str(ch).split("_", 2)[-1])
            return "_and_".join(dict.fromkeys(names)) or "recipe"
        if self.primary_is_overlap:
            names = [c.split("_", 2)[-1] for c in self.intersect_inputs]
            return "_and_".join(names[:2]) if len(names) > 1 else "overlap"
        return str(self.cb_primary.currentData() or "").split("_", 2)[-1]

    def _primary_display(self) -> str:
        """Name for the run FOLDER, which does include the size restriction."""
        if self.primary_is_recipe:
            label = self._recipe_program()[1] or "recipe"
            label = label.replace(" & ", "_and_").replace(" >", "_gt").replace(">", "gt")
            return "".join(c if c.isalnum() or c in "-_." else "_" for c in label)
        if self.primary_is_overlap:
            names = [c.split("_", 2)[-1] for c in self.intersect_inputs]
            return "_and_".join(names[:2]) if len(names) > 1 else "overlap"
        return str(self.cb_primary.currentData() or "").split("_", 2)[-1]

    def _check_partner(self, *_):
        """Warn when the partner cannot yield a meaningful distance."""
        partner = self.cb_partner.currentText()
        if partner.startswith("None"):
            self.lbl_partner_warn.setText("")
            return
        msg = ""
        if self.primary_is_overlap and partner in self.intersect_inputs:
            msg = (f"The overlap lies inside {partner.split('_', 2)[-1]} by "
                   f"construction, so its distance to it is always 0 — pick a "
                   f"third channel.")
        elif partner == self.cb_primary.currentData():
            msg = "The partner is the same channel as the objects being randomised."
        self.lbl_partner_warn.setText(msg)

    def _wire_signals(self):
        """Connect signals and prime derived labels. Call once, last."""
        self.cb_domain.currentIndexChanged.connect(self._domain_help)
        self.cb_direction.currentIndexChanged.connect(self._refresh_name)
        self.cb_primary.currentIndexChanged.connect(self._check_partner)
        self.cb_primary.currentIndexChanged.connect(self._refresh_program)
        self.cb_primary.currentIndexChanged.connect(self._refresh_name)
        self.cb_partner.currentIndexChanged.connect(self._check_partner)
        for widget in (self.cb_primary, self.cb_partner, self.cb_domain):
            widget.currentIndexChanged.connect(self._refresh_name)
        self.le_name.textChanged.connect(self._refresh_path_label)
        self.chk_qc.toggled.connect(self.sp_qc.setEnabled)
        self.chk_qc.toggled.connect(self.chk_qc_annotate.setEnabled)
        self.chk_qc.toggled.connect(self._qc_estimate)
        self.sp_qc.valueChanged.connect(self._qc_estimate)

        self._domain_help()
        self._check_partner()
        self._refresh_program()
        self._refresh_name()
        self._show_existing_runs()
        self._qc_estimate()

    @property
    def null_root(self) -> str:
        return os.path.join(self.project_root, "SPATIAL_NULL")

    def _on_name_edited(self, _text):
        self._name_edited = True

    def _direction_tag(self) -> str:
        return ("to" if str(self.cb_direction.currentData() or "primary")
                == "primary" else "from")

    def _refresh_name(self, *_):
        """Re-derive the default name, unless the user has typed their own."""
        if self._name_edited:
            self._refresh_path_label()
            return
        try:
            from .runner import suggest_run_name
            partner = self.cb_partner.currentText()
            name = suggest_run_name(
                self.null_root,
                primary=("Channel_x_" + self._primary_display()
                         if (self.primary_is_overlap or self.primary_is_recipe)
                         else self.cb_primary.currentText()),
                partner=None if partner.startswith("None") else partner,
                domain_choice=str(self.cb_domain.currentData() or "hull"),
                roi_name=self.roi_name,
                direction=str(self.cb_direction.currentData() or "primary"))
        except Exception:
            name = "01"
        self.le_name.setText(name)
        self._refresh_path_label()

    def _refresh_path_label(self, *_):
        name = self.le_name.text().strip() or "<name required>"
        self.lbl_path.setText(os.path.join(self.null_root, name))

    def _show_existing_runs(self):
        """List what is already there, so a clash is visible before running."""
        try:
            from .runner import list_runs
            runs = list_runs(self.null_root)
        except Exception:
            runs = []
        if not runs:
            self.lbl_existing.setText("No previous runs in this project.")
            return
        shown = "; ".join(
            f"{r['run_name']} ({r.get('primary') or '?'}"
            + (f" vs {r['partner']}" if r.get("partner") else "")
            + f", n={r.get('n_images')})" for r in runs[:4])
        more = f" … and {len(runs) - 4} more" if len(runs) > 4 else ""
        self.lbl_existing.setText(f"Existing runs: {shown}{more}")

    def _qc_estimate(self, *_):
        """Show the file count and size before anything is written."""
        if not getattr(self, "chk_qc", None) or not self.chk_qc.isChecked():
            if getattr(self, "lbl_qc", None):
                self.lbl_qc.setText("")
            return
        from .qc_render import estimate_qc_output
        n_samples = max(1, len(self.pm.sample_registry))
        count, mb = estimate_qc_output(n_samples, self.sp_qc.value())
        self.lbl_qc.setText(f"≈ {count} files, {mb:.0f} MB across "
                            f"{n_samples} sample(s)")

    def _domain_help(self):
        """Show what the selected domain holds fixed, and gate the parent option.

        Written to tolerate being called mid-construction: Qt fires
        currentIndexChanged while a combo is being populated, so a handler that
        assumes every widget already exists is a latent crash.
        """
        key = self.cb_domain.currentData()
        for k, _, help_text in DOMAIN_CHOICES:
            if k == key:
                if getattr(self, "lbl_domain", None) is not None:
                    self.lbl_domain.setText(help_text)
                break
        chk = getattr(self, "chk_per_parent", None)
        if chk is not None:
            chk.setEnabled(str(key).startswith("parent"))

    # -- parameters -----------------------------------------------------------

    def parameters(self) -> Dict[str, Any]:
        partner = self.cb_partner.currentText()
        return {
            "primary_channel": (None if (self.primary_is_overlap
                                        or self.primary_is_recipe)
                                else self.cb_primary.currentData()),
            "primary_is_overlap": self.primary_is_overlap,
            "primary_is_recipe": self.primary_is_recipe,
            "primary_recipe": (self._recipe_program()[0]
                               if self.primary_is_recipe else None),
            "primary_program_label": (
                " -> ".join(_describe_program_step(st)
                            for st in self._recipe_program()[0])
                if self.primary_is_recipe else ""),
            "primary_display": self._primary_display(),
            "primary_base_name": self._primary_base_name(),
            "intersection_spec": ({
                "a_channel": self.intersect_inputs[0],
                "b_channel": self.intersect_inputs[1],
                "label_mode": (self.intersect_step or {}).get("label_mode")
                              or "connected",
                "preserve_ids": bool((self.intersect_step or {}).get("preserve_ids")),
            } if self.primary_is_overlap and len(self.intersect_inputs) > 1
              else None),
            "partner_channel": None if partner.startswith("None") else partner,
            "domain_choice": self.cb_domain.currentData(),
            "per_parent_containment": self.chk_per_parent.isChecked(),
            "erode_um": float(self.sp_erode.value()),
            "rotate": self.chk_rotate.isChecked(),
            "hardcore": self.chk_hardcore.isChecked(),
            "min_separation_um": float(self.sp_sep.value()),
            "n_reference": int(self.sp_ref.value()),
            "n_test": int(self.sp_test.value()),
            "cross_statistic": self.cb_stat.currentText(),
            "statistic_direction": str(self.cb_direction.currentData() or "primary"),
            "measure_from": ("both" if self.chk_both.isChecked()
                             else str(self.cb_direction.currentData() or "primary")),
            "compute_f": self.chk_f.isChecked(),
            "compute_g": self.chk_g.isChecked(),
            "seed": int(self.sp_seed.value()),
            "n_qc_images": (self.sp_qc.value() if self.chk_qc.isChecked() else 0),
            "qc_annotate_distances": self.chk_qc_annotate.isChecked(),
            "also_csv": self.chk_csv.isChecked(),
            "run_name": self.le_name.text().strip(),
            "roi_name": self.roi_name,
            "intersection_inputs": self.intersect_inputs,
            "domain_a_channel": (self.intersect_inputs[0]
                                 if len(self.intersect_inputs) > 0 else None),
            "domain_b_channel": (self.intersect_inputs[1]
                                 if len(self.intersect_inputs) > 1 else None),
        }

    # -- run ------------------------------------------------------------------

    def run(self):
        from .runner import RunParameters, jobs_from_registry, run_project

        p = self.parameters()
        if p["primary_is_recipe"] and not p["primary_recipe"]:
            QMessageBox.warning(
                self, "Recipe not usable",
                "The recipe's filter step has no channel or intersection before "
                "it, so there is nothing to filter. Add one, or randomise a "
                "channel directly.")
            return
        if (not p["primary_channel"] and not p["intersection_spec"]
                and not p["primary_recipe"]):
            QMessageBox.warning(self, "Nothing selected",
                                "Choose a channel, or the recipe's overlap, to "
                                "randomise.")
            return
        if p["intersection_spec"] and p["partner_channel"] in self.intersect_inputs:
            if QMessageBox.question(
                    self, "Degenerate partner",
                    "The overlap lies inside that channel by construction, so "
                    "every cross-distance will be 0.\n\nRun anyway?",
                    QMessageBox.Yes | QMessageBox.Cancel,
                    QMessageBox.Cancel) != QMessageBox.Yes:
                return
        if p["domain_choice"] in ("parent_a", "parent_b", "parent_both") \
                and not p["intersection_inputs"]:
            QMessageBox.warning(
                self, "No intersection",
                "A parent-object domain needs an intersection in the recipe.")
            return

        run_name = p["run_name"]
        if not run_name:
            QMessageBox.warning(self, "Name required",
                                "Give this run a name so it does not overwrite "
                                "another pairing.")
            return
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in run_name)
        if safe != run_name:
            self.le_name.setText(safe)
            run_name = safe

        # The region goes in the name rather than the path, so every run sits at
        # one predictable depth and the notebook loader needs no special case.
        out_dir = os.path.join(self.null_root, run_name)
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            choice = QMessageBox.question(
                self, "Run already exists",
                f"'{run_name}' already exists and holds results.\n\n"
                "Overwrite it, or pick a different name?",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
            if choice != QMessageBox.Yes:
                return

        # Both of these used to stop at the manifest and never reach the
        # engine, so a parent-object domain silently became the whole field and
        # per-parent containment did nothing.
        params = RunParameters(
            n_reference=p["n_reference"], n_test=p["n_test"],
            rotate=p["rotate"], hardcore=p["hardcore"],
            min_separation_um=p["min_separation_um"],
            use_hull=(p["domain_choice"] == "hull"),
            domain_choice=p["domain_choice"],
            per_parent_containment=p["per_parent_containment"],
            erode_um=p["erode_um"], compute_f=p["compute_f"],
            compute_g=p["compute_g"], cross_statistic=p["cross_statistic"],
            measure_from=p["measure_from"],
            statistic_direction=p["statistic_direction"],
            seed=p["seed"], roi_name=p["roi_name"], run_name=run_name,
            primary_program=p["primary_program_label"],
            keep_first_draw=False, also_csv=p["also_csv"],
            n_qc_images=p["n_qc_images"],
            qc_annotate_distances=p["qc_annotate_distances"])

        try:
            dom_a, dom_b = p["domain_a_channel"], p["domain_b_channel"]
            jobs = jobs_from_registry(
                self.pm.sample_registry, p["primary_channel"],
                p["partner_channel"],
                domain_a_channel=dom_a, domain_b_channel=dom_b,
                primary_intersection=p["intersection_spec"],
                primary_recipe=p["primary_recipe"],
                roi_name=p["roi_name"])
        except Exception as exc:
            QMessageBox.critical(self, "Run failed", str(exc))
            return

        if not jobs:
            QMessageBox.warning(
                self, "No samples",
                "No sample had a final segmentation for that channel"
                + (f" in region {p['roi_name']!r}." if p["roi_name"] else "."))
            return

        # Run off the GUI thread so the window stays responsive and cancellable
        # instead of the OS offering to kill it.
        prog = _NullProgressDialog(len(jobs), params.n_reference + params.n_test,
                                   parent=self)
        worker = _NullWorker(
            jobs, params, out_dir, os.path.basename(self.project_root),
            # The pairing uses the base channel name; the program string carries
            # the size restriction (see _primary_base_name).
            {"primary": (p["primary_channel"]
                         or f"Channel_x_{p['primary_base_name']}"),
             "partner": p["partner_channel"],
             "domain_choice": p["domain_choice"]})

        state: Dict[str, Any] = {"result": None, "error": None}
        worker.progress.connect(prog.update_progress)
        worker.logline.connect(prog.append)
        worker.finished_ok.connect(lambda r: (state.__setitem__("result", r),
                                             prog.accept()))
        worker.failed.connect(lambda m: (state.__setitem__("error", m),
                                         prog.reject()))
        poll = QTimer(prog)
        poll.timeout.connect(lambda: worker.cancel() if prog.cancelled else None)
        poll.start(200)

        worker.start()
        prog.exec_()
        prog.close_when_done()
        poll.stop()
        if worker.isRunning():
            worker.cancel()
            worker.wait(30000)

        if state["error"]:
            QMessageBox.critical(self, "Run failed", state["error"])
            return
        result = state["result"] or {}
        if prog.cancelled and not result.get("n_samples"):
            QMessageBox.information(self, "Cancelled",
                                    "The run was cancelled; nothing was written.")
            return

        if not result.get("n_samples"):
            QMessageBox.warning(self, "Nothing scored",
                                "No sample produced a usable result. See the "
                                "console for per-sample reasons.")
            return

        md = result["metadata"]
        concerns = []
        if "packing_warning" in md.columns and md["packing_warning"].any():
            concerns.append(
                f"{int(md['packing_warning'].sum())} image(s) at high occupancy: "
                f"a non-overlapping null is forced toward regularity there, so "
                f"regularity may be a packing artefact.")
        if "placement_warning" in md.columns and md["placement_warning"].any():
            concerns.append(
                f"{int(md['placement_warning'].sum())} image(s) had draws where "
                f"not every object could be placed.")
        if "orientation_acceptance_rate" in md.columns:
            low = int((md["orientation_acceptance_rate"] < 0.5).sum())
            if low:
                concerns.append(
                    f"{low} image(s) accepted under half of proposed "
                    f"orientations, so the domain boundary constrained which "
                    f"orientations were possible.")

        text = (f"Scored {result['n_samples']} image(s).\n\n"
                f"Exported to:\n{out_dir}\n\n"
                "The export holds raw per-object distances for every draw. "
                "Pool several projects with hibachi_null_io to do statistics.")
        if result.get("qc_disabled_reason"):
            concerns.append(
                "QC images were requested but the renderer failed its "
                f"self-test, so none were written: {result['qc_disabled_reason']}")
        elif result.get("qc_errors"):
            concerns.append(
                f"{result['qc_errors']} QC image(s) failed to render; see the "
                f"console for the reason.")
        if result.get("qc_dir"):
            text += (f"\n\nQC images:\n{result['qc_dir']}\n"
                     "Each sample folder holds 000_observed.jpg plus one image "
                     "per randomisation. Compare the draws against the observed "
                     "one.")
        if concerns:
            text += "\n\nWorth checking:\n- " + "\n- ".join(concerns)
        QMessageBox.information(self, "Spatial null complete", text)
        # Leave the dialog open so another pairing can be run straight away --
        # that is the point of named runs.
        self._name_edited = False
        self._refresh_name()
        self._show_existing_runs()