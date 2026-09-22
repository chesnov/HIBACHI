"""
The application Settings dialog: per-install preferences, in tabs.

*   **Resources** -- the one place a user sets how much of their machine the
    pipeline may use.
*   **Project setup** -- whether setup asks for physical dimensions it could
    not establish on its own (`dimension_entry`).

Resources: three numbers -- RAM, cores, VRAM -- stored per install, not per project. The
reasoning for "three, and only three" is in `resource_budget`: how a step spends
its allowance (more workers or bigger blocks) is decided in code, because that
trade differs per step and nobody should have to tune fifteen knobs to make one
machine faster than another.

This module is Qt; `resource_budget` and `dimension_entry` are not, and the
dependency runs one way only. Pipeline steps -- including ones inside worker processes -- import the
budget; nothing in the budget imports this.
"""

from typing import Optional

from PyQt5.QtCore import Qt  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

# This module lives in `high_level_gui/`; `resource_budget` lives in
# `fluorescence_module/` beside the pipeline steps that consume it. Same
# cross-package form `project_view_window` already uses for
# `config_migration.normalise_mode`. Getting this wrong is silent until the
# dialog is opened, at which point the import fails.
try:
    from ..fluorescence_module import resource_budget
except ImportError:  # pragma: no cover - direct script execution
    import resource_budget

try:
    from . import dimension_entry
except ImportError:  # pragma: no cover - direct script execution
    import dimension_entry  # type: ignore


class SettingsDialog(QDialog):
    """Per-install settings: resource ceiling and project-setup behaviour."""

    TAB_RESOURCES = 0
    TAB_SETUP = 1

    def __init__(self, parent=None, initial_tab: int = TAB_RESOURCES):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)

        # Re-measured on open rather than taken from the cache: `available` moves
        # constantly and is the number that tells the user how much of their
        # machine is free right now, which is the context for choosing a ceiling.
        self.device = resource_budget.probe_device(refresh=True)
        self.limits = resource_budget.feasible_range(self.device)
        current = resource_budget.load_settings(self.device)
        self._initial_prompt = dimension_entry.prompt_enabled()

        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)

        resources_tab = QWidget()
        layout = QVBoxLayout(resources_tab)

        # ---- what the machine has --------------------------------------
        detected = QGroupBox("This machine")
        det_form = QFormLayout(detected)
        det_form.addRow(
            "Memory:",
            QLabel(f"{self.device.total_ram_gb:.1f} GB total, "
                   f"{self.device.available_ram_gb:.1f} GB free right now"),
        )
        det_form.addRow(
            "Processors:",
            QLabel(f"{self.device.physical_cores} physical cores, "
                   f"{self.device.logical_cores} logical"),
        )
        gpu_text = (f"{self.device.gpu_name} "
                    f"({self.device.total_vram_gb:.1f} GB)"
                    if self.device.gpu_name else "none detected")
        det_form.addRow("Graphics:", QLabel(gpu_text))
        layout.addWidget(detected)

        # ---- the ceiling ------------------------------------------------
        limits_box = QGroupBox("Maximum this application may use")
        form = QFormLayout(limits_box)

        ram_lo, ram_hi = self.limits["ram_gb"]
        self.ram_spin = QDoubleSpinBox()
        self.ram_spin.setRange(ram_lo, ram_hi)
        self.ram_spin.setSingleStep(0.5)
        self.ram_spin.setDecimals(1)
        self.ram_spin.setSuffix(" GB")
        self.ram_spin.setValue(current.ram_gb)
        form.addRow("Memory:", self.ram_spin)

        core_lo, core_hi = self.limits["cores"]
        self.core_spin = QSpinBox()
        self.core_spin.setRange(core_lo, core_hi)
        self.core_spin.setValue(current.cores)
        form.addRow("Processor cores:", self.core_spin)

        vram_lo, vram_hi = self.limits["vram_gb"]
        self.vram_spin = QDoubleSpinBox()
        self.vram_spin.setRange(vram_lo, vram_hi)
        self.vram_spin.setSingleStep(0.5)
        self.vram_spin.setDecimals(1)
        self.vram_spin.setSuffix(" GB")
        self.vram_spin.setValue(current.vram_gb)
        # Disabled, and honestly labelled. No step computes on the GPU: matching
        # the CPU result bit-for-bit is not currently possible for the two most
        # expensive operations (Frangi and Sato both go through `exp`, which
        # IEEE-754 does not require to be correctly rounded, so CUDA's and
        # glibc's answers differ in the last bits). A pipeline that produced
        # different numbers on a machine with a GPU than on one without would
        # not be worth the speed. The setting exists and is persisted so that
        # nothing has to be retrofitted if that changes.
        self.vram_spin.setEnabled(self.device.total_vram_gb > 0)
        form.addRow("Graphics memory:", self.vram_spin)
        layout.addWidget(limits_box)

        # ---- what the numbers mean --------------------------------------
        note = QLabel(
            "The memory ceiling covers the WHOLE application, including this "
            "window and any open image viewers \u2014 not just the pipeline. "
            "Processing sizes its blocks and its worker count to stay inside "
            "it, and slows itself down rather than exceed it.\n\n"
            "Raising these makes large datasets process faster. Lowering them "
            "leaves more of the machine for everything else. Results do not "
            "change either way: the same image processed at any setting "
            "produces identical output."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        layout.addWidget(note)

        self.gpu_note = QLabel(
            "Graphics memory is unused: no processing step runs on the GPU, "
            "because GPU and CPU results would differ in the last decimal "
            "places and the pipeline must give the same answer on every "
            "machine."
        )
        self.gpu_note.setWordWrap(True)
        self.gpu_note.setStyleSheet("color: #777; font-style: italic;")
        layout.addWidget(self.gpu_note)

        # An environment override silently wins over anything saved here, so say
        # so rather than let the user set a value that does nothing.
        if current.origin == "environment":
            warn = QLabel(
                "\u26a0 HIBACHI_RAM_GB / HIBACHI_CORES / HIBACHI_VRAM_GB is set "
                "in the environment and overrides whatever is saved here. "
                "Unset it for these values to take effect."
            )
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #a05000; font-weight: bold;")
            layout.addWidget(warn)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        path_label = QLabel(f"Saved in: {resource_budget.settings_path()}")
        path_label.setStyleSheet("color: #777; font-size: 10px;")
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(path_label)

        row = QHBoxLayout()
        self.reset_btn = QPushButton("Recommended for this machine")
        self.reset_btn.setToolTip(
            "Reset to the values chosen automatically on a machine of this "
            "size: about 60% of memory, or all but 4 GB, whichever is smaller, "
            "and two cores fewer than the processor has."
        )
        self.reset_btn.clicked.connect(self._restore_defaults)
        row.addWidget(self.reset_btn)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
        self.tabs.addTab(resources_tab, "Resources")

        self.tabs.addTab(self._build_setup_tab(), "Project setup")
        self.tabs.setCurrentIndex(initial_tab)

        # What the spin boxes showed on open, after their own rounding. Save
        # compares against this rather than against `current`, whose extra
        # decimals would make an untouched box look edited.
        self._initial_resources = self._resource_values()

        # ---- buttons ------------------------------------------------------
        # One Save for every tab: the user should not have to wonder whether
        # switching tabs discarded what they set on the other one.
        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.setMinimumWidth(520)

    # ------------------------------------------------------------------
    def _build_setup_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        box = QGroupBox("Physical dimensions")
        box_layout = QVBoxLayout(box)
        self.dim_prompt_check = QCheckBox(
            "Ask for dimensions when an image is not calibrated"
        )
        self.dim_prompt_check.setChecked(self._initial_prompt)
        self.dim_prompt_check.setToolTip(
            "Shown during project setup when an image's size could not be read "
            "from its metadata or a metadata CSV, or when the recorded size is "
            "identical to its pixel count (exactly 1 \u00b5m per pixel)."
        )
        box_layout.addWidget(self.dim_prompt_check)

        layout.addWidget(box)

        # Outside the group box, like the notes on the Resources tab: a
        # word-wrapped QLabel inside a QGroupBox gets its height from the
        # unwrapped width and is clipped.
        note = QLabel(
            "When this is off, project setup no longer stops to ask. Images "
            "whose scale could not be read keep their pixel counts as their "
            "dimensions, so sizes, distances and densities measured on them "
            "are in pixels, not microns.\n\n"
            "A metadata CSV next to the raw images is still used whenever it "
            "has a matching row, and uncalibrated images are still recorded as "
            "such in their config (dimensions_source: pixels_assumed). "
            "Dimensions can be corrected per image in the project view."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        layout.addWidget(note)

        layout.addStretch(1)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        path_label = QLabel(f"Saved in: {dimension_entry.preferences_path()}")
        path_label.setStyleSheet("color: #777; font-size: 10px;")
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(path_label)
        return tab

    # ------------------------------------------------------------------
    def _restore_defaults(self) -> None:
        """Put the first-run values back, without saving them yet."""
        proposed = resource_budget._default_settings(self.device)
        self.ram_spin.setValue(proposed.ram_gb)
        self.core_spin.setValue(proposed.cores)
        self.vram_spin.setValue(proposed.vram_gb)

    def _resource_values(self):
        return (float(self.ram_spin.value()), float(self.vram_spin.value()),
                int(self.core_spin.value()))

    def _save(self) -> None:
        # Resources are written only when they were edited. Saving them on
        # every Save would turn a first-run default -- recomputed for whatever
        # machine this install is on -- into a fixed "user" value just because
        # someone changed an unrelated setting on the other tab.
        if self._resource_values() != self._initial_resources:
            if not self._save_resources():
                return
        prompt = self.dim_prompt_check.isChecked()
        if prompt != self._initial_prompt:
            try:
                dimension_entry.set_prompt_enabled(prompt)
            except OSError as exc:
                QMessageBox.critical(
                    self, "Could not save",
                    f"The project-setup settings could not be written to\n"
                    f"{dimension_entry.preferences_path()}\n\n{exc}",
                )
                return
        self.accept()

    def _save_resources(self) -> bool:
        """Write the resource ceiling. False if cancelled or it failed."""
        chosen = resource_budget.ResourceSettings(
            ram_gb=float(self.ram_spin.value()),
            vram_gb=float(self.vram_spin.value()),
            cores=int(self.core_spin.value()),
            origin="user",
        )
        # A ceiling above what the machine has is not a preference, it is a
        # crash waiting for a large dataset. `save_settings` clamps, but warn
        # first so the user is not surprised by a different number next time.
        if chosen.ram_gb > self.device.available_ram_gb:
            reply = QMessageBox.question(
                self, "Above what is currently free",
                f"You have set a {chosen.ram_gb:.1f} GB ceiling, but only "
                f"{self.device.available_ram_gb:.1f} GB is free right now. "
                "Processing will still keep itself inside the ceiling, but if "
                "other applications are using the difference the machine may "
                "slow down or swap.\n\nSave anyway?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.tabs.setCurrentIndex(self.TAB_RESOURCES)
                return False
        try:
            resource_budget.save_settings(chosen, self.device)
        except OSError as exc:
            QMessageBox.critical(
                self, "Could not save",
                f"The resource settings could not be written to\n"
                f"{resource_budget.settings_path()}\n\n{exc}",
            )
            return False
        return True


#: The dialog's previous name, kept so existing imports keep working.
ResourceSettingsDialog = SettingsDialog


def open_settings(parent=None, initial_tab: int = SettingsDialog.TAB_RESOURCES) -> bool:
    """Show the dialog. True when the user saved."""
    dlg = SettingsDialog(parent, initial_tab=initial_tab)
    return dlg.exec_() == QDialog.Accepted


#: Previous name of `open_settings`.
open_resource_settings = open_settings