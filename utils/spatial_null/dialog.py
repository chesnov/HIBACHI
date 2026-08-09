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

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
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
        self.intersect_inputs = self._recipe_intersection()

        self.setWindowTitle("Spatial Null (mask-preserving randomisation)")
        self.setMinimumWidth(620)
        self._build()

    # -- recipe interpretation ------------------------------------------------

    def _recipe_intersection(self) -> List[str]:
        """Channels of the last intersection step, if the recipe has one."""
        for step in reversed(self.recipe):
            if step.get("type") == "intersect":
                return [c for c in (step.get("inputs") or [])
                        if c != "PREVIOUS_RESULT"]
        return []

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
        self.cb_primary.addItems(self.all_channels)
        if self.intersect_inputs:
            self.cb_primary.setEnabled(False)
            self.cb_primary.setToolTip(
                "Taken from the recipe's intersection result.")
        elif self.checked:
            self.cb_primary.setCurrentText(self.checked[0])
        form.addRow("Objects to randomise:", self.cb_primary)

        self.cb_partner = QComboBox()
        self.cb_partner.addItem("None (no cross-distances)")
        self.cb_partner.addItems(self.all_channels)
        self.cb_partner.setToolTip(
            "Held fixed and never moved. Its distance field is computed once, "
            "so cross-distances cost almost nothing per draw.")
        if len(self.checked) > 1:
            self.cb_partner.setCurrentText(self.checked[1])
        elif len(self.intersect_inputs) > 1:
            self.cb_partner.setCurrentText(self.intersect_inputs[1])
        form.addRow("Fixed partner channel:", self.cb_partner)
        layout.addWidget(src)

        dom = QGroupBox("Domain — this IS the null hypothesis")
        dv = QVBoxLayout(dom)
        self.cb_domain = QComboBox()
        allowed = DOMAIN_CHOICES if self.intersect_inputs else DOMAIN_CHOICES[:2]
        for key, label, _ in allowed:
            self.cb_domain.addItem(label, key)
        dv.addWidget(self.cb_domain)
        self.lbl_domain = QLabel()
        self.lbl_domain.setWordWrap(True)
        self.lbl_domain.setStyleSheet("color: grey;")
        dv.addWidget(self.lbl_domain)
        self.cb_domain.currentIndexChanged.connect(self._domain_help)
        self._domain_help()

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
        self.chk_overlay = QCheckBox(
            "Show the first draw as napari layers, with connection lines")
        self.chk_overlay.setChecked(False)
        self.chk_overlay.setToolTip(
            "Off by default because it is per-sample and heavy. Worth turning on "
            "once to confirm the randomisation looks right rather than trusting "
            "it blindly.")
        of.addRow(self.chk_overlay)
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

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.btn_run = QPushButton("Run and export")
        self.btn_run.setStyleSheet(
            "background-color: #2E8B57; color: white; font-weight: bold;")
        buttons.addButton(self.btn_run, QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(self.reject)
        self.btn_run.clicked.connect(self.run)
        layout.addWidget(buttons)

    def _domain_help(self):
        key = self.cb_domain.currentData()
        for k, _, help_text in DOMAIN_CHOICES:
            if k == key:
                self.lbl_domain.setText(help_text)
                break
        self.chk_per_parent.setEnabled(str(key).startswith("parent"))

    # -- parameters -----------------------------------------------------------

    def parameters(self) -> Dict[str, Any]:
        partner = self.cb_partner.currentText()
        return {
            "primary_channel": self.cb_primary.currentText(),
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
            "compute_f": self.chk_f.isChecked(),
            "compute_g": self.chk_g.isChecked(),
            "seed": int(self.sp_seed.value()),
            "show_overlay": self.chk_overlay.isChecked(),
            "also_csv": self.chk_csv.isChecked(),
            "roi_name": self.roi_name,
            "intersection_inputs": self.intersect_inputs,
        }

    # -- run ------------------------------------------------------------------

    def run(self):
        from .runner import RunParameters, jobs_from_registry, run_project

        p = self.parameters()
        if not p["primary_channel"]:
            QMessageBox.warning(self, "Nothing selected",
                                "Choose a channel whose objects to randomise.")
            return
        if p["domain_choice"] in ("parent_a", "parent_b", "parent_both") \
                and not p["intersection_inputs"]:
            QMessageBox.warning(
                self, "No intersection",
                "A parent-object domain needs an intersection in the recipe.")
            return

        out_dir = os.path.join(self.project_root, "SPATIAL_NULL")
        if p["roi_name"]:
            out_dir = os.path.join(out_dir, p["roi_name"])

        params = RunParameters(
            n_reference=p["n_reference"], n_test=p["n_test"],
            rotate=p["rotate"], hardcore=p["hardcore"],
            min_separation_um=p["min_separation_um"],
            use_hull=(p["domain_choice"] == "hull"),
            erode_um=p["erode_um"], compute_f=p["compute_f"],
            compute_g=p["compute_g"], cross_statistic=p["cross_statistic"],
            seed=p["seed"], roi_name=p["roi_name"],
            keep_first_draw=p["show_overlay"], also_csv=p["also_csv"])

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            jobs = jobs_from_registry(
                self.pm.sample_registry, p["primary_channel"],
                p["partner_channel"], roi_name=p["roi_name"])
            if not jobs:
                QApplication.restoreOverrideCursor()
                QMessageBox.warning(
                    self, "No samples",
                    "No sample had a final segmentation for that channel"
                    + (f" in region {p['roi_name']!r}." if p["roi_name"] else "."))
                return

            result = run_project(
                jobs, params, out_dir=out_dir,
                project_name=os.path.basename(self.project_root),
                channels={"primary": p["primary_channel"],
                          "partner": p["partner_channel"],
                          "domain_choice": p["domain_choice"]})
            QApplication.restoreOverrideCursor()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Run failed", str(exc))
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
        if concerns:
            text += "\n\nWorth checking:\n- " + "\n- ".join(concerns)
        QMessageBox.information(self, "Spatial null complete", text)

        if p["show_overlay"] and result.get("first_draw_labels"):
            self._offer_overlay(result)
        self.accept()

    def _offer_overlay(self, result: Dict[str, Any]):
        """Show one sample's observed and first-draw masks side by side."""
        first = result.get("first_draw_labels") or {}
        if not first:
            return
        try:
            import napari
        except ImportError:
            QMessageBox.information(
                self, "napari unavailable",
                "Install napari to view the overlay; the export is unaffected.")
            return
        sample, labels = next(iter(first.items()))
        viewer = napari.Viewer(title=f"Spatial null — {sample}")
        viewer.add_labels(labels, name=f"{sample} randomised (draw 1)")
        QMessageBox.information(
            self, "Overlay",
            f"Showing the first draw for {sample}. Compare it against the real "
            f"segmentation to confirm sizes and shapes are preserved and that "
            f"nothing sits outside the domain.")
