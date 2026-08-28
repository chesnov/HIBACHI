"""
Physical dimensions: provenance, gap detection, and manual entry.
=================================================================

An image whose physical scale is unknown used to be recorded as
``1.0 um/pixel x pixel_count``, i.e. its dimensions became its pixel counts.
That is indistinguishable from a genuinely 1-micron-per-pixel image, so
``require_dimensions`` -- which checks only that the numbers are present,
finite and positive -- accepted it, and every downstream measurement was
silently wrong by the ratio of the true spacing to 1.0. For a 0.325 um/px
confocal image that is 3x on every length and ~29x on every volume.

This module closes that hole in three parts:

1.  **Detect the gap per axis.** ``scale_gaps`` says which axes have no
    trustworthy scale, treating a unit (1.0) spacing as "no calibration"
    because many writers emit XResolution=(1,1)/ResolutionUnit=NONE on
    uncalibrated images. Per-axis rather than per-image, because Z is the axis
    microscopes most often fail to record, and a partial CSV can pin X and Y
    while leaving Z unknown.

2.  **Ask the user.** ``DimensionEntryDialog`` collects the missing extents.
    It appears ONLY for axes that automatic detection could not supply, so a
    correctly calibrated dataset never sees it.

3.  **Record where the numbers came from.** ``dimensions_source`` /
    ``stamp_dimensions_source`` persist a ``dimensions_source`` key in the
    config. Legacy configs lack it and read back as ``unknown`` rather than
    failing, so nothing that already works breaks.

Everything above the dialog is pure and Qt-free, so it can be unit tested and
used headlessly; the Qt import is deferred into the dialog helpers.
"""

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
DIMENSION_SOURCE_KEY = "dimensions_source"

SOURCE_METADATA = "metadata"          # read from the image file's own header
SOURCE_CSV = "csv"                    # from a user-supplied metadata CSV
SOURCE_MANUAL = "manual"              # typed by the user in the dialog
SOURCE_MIXED = "mixed"                # axes came from different sources
SOURCE_PIXELS_ASSUMED = "pixels_assumed"  # NOT calibrated: extents are pixel counts
SOURCE_UNKNOWN = "unknown"            # legacy config written before this existed

#: Sources that mean "these numbers are a real physical measurement".
TRUSTED_SOURCES = frozenset({SOURCE_METADATA, SOURCE_CSV, SOURCE_MANUAL, SOURCE_MIXED})

_SOURCE_LABEL = {
    SOURCE_METADATA: "read from the image metadata",
    SOURCE_CSV: "imported from a metadata CSV",
    SOURCE_MANUAL: "entered manually",
    SOURCE_MIXED: "from more than one source",
    SOURCE_PIXELS_ASSUMED: "NOT CALIBRATED - these are pixel counts",
    SOURCE_UNKNOWN: "unrecorded (config predates dimension tracking)",
}


def dimensions_source(config: Dict[str, Any]) -> str:
    """Where a config's physical dimensions came from.

    Returns ``SOURCE_UNKNOWN`` for any config that does not carry the key --
    which is every config written before this existed. Legacy configs are
    therefore readable and processable exactly as before; they simply cannot
    claim to be calibrated.
    """
    if not isinstance(config, dict):
        return SOURCE_UNKNOWN
    value = config.get(DIMENSION_SOURCE_KEY)
    if not isinstance(value, str) or not value.strip():
        return SOURCE_UNKNOWN
    value = value.strip().lower()
    return value if value in _SOURCE_LABEL else SOURCE_UNKNOWN


def describe_dimensions_source(config_or_source) -> str:
    """Human-readable phrase for a config or a bare source string."""
    source = (config_or_source if isinstance(config_or_source, str)
              else dimensions_source(config_or_source))
    return _SOURCE_LABEL.get(source, _SOURCE_LABEL[SOURCE_UNKNOWN])


def stamp_dimensions_source(config: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Record the provenance of a config's dimensions, in place."""
    if isinstance(config, dict):
        config[DIMENSION_SOURCE_KEY] = (
            source if source in _SOURCE_LABEL else SOURCE_UNKNOWN
        )
    return config


def combine_sources(per_axis: Dict[str, str]) -> str:
    """Collapse per-axis provenance into one value for the config.

    Any uncalibrated axis dominates: an image with a real X/Y but an assumed Z
    is not a calibrated image, and saying ``mixed`` there would imply the
    numbers can be trusted.
    """
    values = {v for v in per_axis.values() if v}
    if not values:
        return SOURCE_UNKNOWN
    if SOURCE_PIXELS_ASSUMED in values:
        return SOURCE_PIXELS_ASSUMED
    if SOURCE_UNKNOWN in values:
        return SOURCE_UNKNOWN
    if len(values) == 1:
        return next(iter(values))
    return SOURCE_MIXED


def is_calibrated(config: Dict[str, Any]) -> bool:
    """True only when the dimensions are known to be a real measurement."""
    return dimensions_source(config) in TRUSTED_SOURCES


# --------------------------------------------------------------------------- #
# Gap detection
# --------------------------------------------------------------------------- #
def axes_for_mode(mode) -> Tuple[str, ...]:
    """Axes a mode needs: 2D has no Z."""
    return ("x", "y") if str(mode).endswith("_2d") else ("x", "y", "z")


def _is_real_spacing(value) -> bool:
    """A per-pixel spacing worth trusting.

    A spacing of exactly 1.0 is rejected. Many writers (tifffile included)
    store XResolution=(1,1) with ResolutionUnit=NONE on an uncalibrated image,
    which resolves to exactly 1.0 micron/pixel and is reported as found. That
    is indistinguishable from no calibration at all, so it is treated as
    missing and the user is asked, rather than silently producing dimensions
    that are really just pixel counts.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return False
    if num != num or num <= 0:            # NaN or non-positive
        return False
    return num != 1.0


def scale_gaps(meta: Optional[Dict[str, Any]], mode) -> Tuple[str, ...]:
    """Axes for which `meta` provides no trustworthy scale.

    Per-axis on purpose. The previous check looked only at X and Y, so a 3D
    image with a genuine X/Y but no Z spacing passed as fully calibrated -- and
    Z is exactly the axis that is most often missing.
    """
    axes = axes_for_mode(mode)
    if not isinstance(meta, dict) or not meta.get("found"):
        return axes
    return tuple(a for a in axes if not _is_real_spacing(meta.get(a)))


def plan_manual_entry(
    files_meta: Sequence[Tuple[str, Optional[Dict[str, Any]], Dict[str, int]]],
    mode,
    csv_overrides: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    match_override=None,
) -> List[Dict[str, Any]]:
    """Work out who needs to be asked, and for which axes.

    `files_meta` is a sequence of ``(filename, metadata_dict, pixel_counts)``
    where `pixel_counts` maps axis -> pixel count, used to prefill the dialog.

    An axis is asked about only when BOTH automatic routes fail: the file's own
    metadata has no real spacing for it, and no CSV row pins it down. This is
    what keeps the dialog out of the way of correctly calibrated data, and it
    also closes the partial-CSV hole -- a CSV giving only 'Width (um)' used to
    suppress the warning for Y and Z as well.

    Returns one entry per file needing input:
    ``{'filename', 'axes', 'pixels', 'from_csv'}``.
    """
    csv_overrides = csv_overrides or {}
    plan: List[Dict[str, Any]] = []

    for filename, meta, pixels in files_meta:
        missing = set(scale_gaps(meta, mode))
        if not missing:
            continue

        override = None
        if csv_overrides:
            if match_override is not None:
                override = match_override(csv_overrides, filename)
            else:
                override = csv_overrides.get(filename)

        from_csv = set()
        if override:
            for axis in list(missing):
                value = override.get(axis)
                if value is not None:
                    missing.discard(axis)
                    from_csv.add(axis)

        if missing:
            plan.append({
                "filename": filename,
                "axes": tuple(a for a in axes_for_mode(mode) if a in missing),
                "pixels": dict(pixels or {}),
                "from_csv": tuple(sorted(from_csv)),
            })
    return plan


def per_axis_sources(
    mode,
    meta: Optional[Dict[str, Any]],
    csv_axes: Iterable[str] = (),
    manual_axes: Iterable[str] = (),
) -> Dict[str, str]:
    """Provenance of each axis, for `combine_sources`."""
    csv_axes, manual_axes = set(csv_axes), set(manual_axes)
    gaps = set(scale_gaps(meta, mode))
    out: Dict[str, str] = {}
    for axis in axes_for_mode(mode):
        if axis in manual_axes:
            out[axis] = SOURCE_MANUAL
        elif axis in csv_axes:
            out[axis] = SOURCE_CSV
        elif axis not in gaps:
            out[axis] = SOURCE_METADATA
        else:
            out[axis] = SOURCE_PIXELS_ASSUMED
    return out


# --------------------------------------------------------------------------- #
# Manual entry dialog (Qt imported lazily so the logic above stays headless)
# --------------------------------------------------------------------------- #
_AXIS_LABEL = {"x": "Width (µm)", "y": "Height (µm)", "z": "Depth (µm)"}
_AXIS_PIXEL_LABEL = {"x": "width", "y": "height", "z": "slices"}


def collect_manual_dimensions(parent, plan, mode, clean_name=None):
    """Ask the user for the extents named in `plan`.

    Returns an overrides dict shaped exactly like the CSV overrides
    (``{cleaned_filename: {'x': float|None, 'y': ..., 'z': ...}}``) so it can be
    merged into the existing precedence chain without a second code path, or
    ``None`` if the user cancelled.

    Returns an empty dict without showing anything when `plan` is empty, which
    is the normal case for calibrated data.
    """
    if not plan:
        return {}

    from PyQt5.QtWidgets import QDialog  # noqa: F401  (import check / lazy load)

    dialog = DimensionEntryDialog(plan, mode, parent=parent)
    if dialog.exec_() != dialog.Accepted:
        return None

    values = dialog.values()
    if clean_name is None:
        return values
    return {clean_name(name): axes for name, axes in values.items()}


def _build_dialog_classes():
    """Defined inside a function so importing this module never requires Qt."""
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
        QFrame, QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget,
    )

    class _DimensionEntryDialog(QDialog):
        """Collect physical extents for images whose scale could not be read.

        Shows one group per image and one spin box per MISSING axis only --
        axes that metadata or a CSV already supplied are not shown, so the
        dialog cannot be used to accidentally overwrite a good value.

        Values are TOTAL microns per axis, matching the config's
        ``voxel_dimensions`` / ``pixel_dimensions`` blocks and the CSV's
        'Width (um)' columns, rather than per-pixel spacing. Totals are what
        the rest of the pipeline stores, and asking for the same quantity the
        file format uses avoids a units conversion the user has to get right.
        """

        def __init__(self, plan, mode, parent=None):
            super().__init__(parent)
            self._plan = list(plan)
            self._mode = mode
            self._spins = {}
            self.setWindowTitle("Enter image dimensions")
            self.setModal(True)
            self._build_ui()

        # -- construction ------------------------------------------------- #
        def _build_ui(self):
            outer = QVBoxLayout(self)

            n = len(self._plan)
            head = QLabel(
                f"<b>{n} image(s) have no usable scale information.</b><br><br>"
                "HIBACHI could not read a physical scale from the image "
                "metadata, and no metadata CSV supplied one. Enter the "
                "<b>total</b> physical size of each image below.<br><br>"
                "Without this, every size, distance and density would be "
                "measured in pixels while being reported in microns."
            )
            head.setWordWrap(True)
            outer.addWidget(head)

            self._apply_all = QCheckBox(
                "Apply the first image's values to all images below")
            self._apply_all.setToolTip(
                "Use when every image was acquired with the same objective and "
                "zoom, which is the usual case for one experiment.")
            self._apply_all.toggled.connect(self._on_apply_all)
            outer.addWidget(self._apply_all)

            body = QWidget()
            body_layout = QVBoxLayout(body)
            body_layout.setContentsMargins(0, 0, 0, 0)

            for idx, entry in enumerate(self._plan):
                body_layout.addWidget(self._build_group(idx, entry))

            body_layout.addStretch(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(body)
            scroll.setMinimumHeight(260)
            outer.addWidget(scroll, 1)

            self._warning = QLabel()
            self._warning.setWordWrap(True)
            self._warning.setStyleSheet("color: #b00020;")
            self._warning.hide()
            outer.addWidget(self._warning)

            buttons = QDialogButtonBox(
                QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
            buttons.button(QDialogButtonBox.Ok).setText("Use these dimensions")
            buttons.button(QDialogButtonBox.Cancel).setText("Cancel setup")
            buttons.accepted.connect(self._on_accept)
            buttons.rejected.connect(self.reject)
            outer.addWidget(buttons)

            self.resize(560, 520)

        def _build_group(self, idx, entry):
            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            layout = QVBoxLayout(frame)

            title = QLabel(f"<b>{entry['filename']}</b>")
            title.setWordWrap(True)
            layout.addWidget(title)

            if entry.get("from_csv"):
                supplied = ", ".join(
                    _AXIS_LABEL.get(a, a) for a in entry["from_csv"])
                note = QLabel(
                    f"<i>{supplied} already supplied by the metadata CSV.</i>")
                note.setWordWrap(True)
                layout.addWidget(note)

            form = QFormLayout()
            pixels = entry.get("pixels") or {}
            for axis in entry["axes"]:
                spin = QDoubleSpinBox()
                spin.setDecimals(4)
                spin.setRange(0.0, 1e9)
                spin.setSingleStep(1.0)
                spin.setSpecialValueText("(not set)")
                spin.setValue(0.0)          # 0 == not set; validated on accept
                px = pixels.get(axis)
                if px:
                    spin.setSuffix(
                        f"   ({int(px)} px {_AXIS_PIXEL_LABEL.get(axis, axis)})")
                    spin.setToolTip(
                        f"Total extent along {axis.upper()}. This axis is "
                        f"{int(px)} pixels, so entering T gives a spacing of "
                        f"T/{int(px)} µm per pixel.")
                form.addRow(_AXIS_LABEL.get(axis, axis), spin)
                self._spins[(idx, axis)] = spin
            layout.addLayout(form)
            return frame

        # -- behaviour ---------------------------------------------------- #
        def _on_apply_all(self, checked):
            """Mirror the first image's values into the rest.

            Only mirrors axes the target image is actually missing, so this can
            never inject a value for an axis a CSV already pinned.
            """
            if not checked or not self._plan:
                self._set_followers_enabled(True)
                return
            first_axes = {
                axis: spin.value()
                for (idx, axis), spin in self._spins.items() if idx == 0
            }
            for (idx, axis), spin in self._spins.items():
                if idx == 0:
                    continue
                if axis in first_axes:
                    spin.setValue(first_axes[axis])
            self._set_followers_enabled(False)

        def _set_followers_enabled(self, enabled):
            for (idx, _axis), spin in self._spins.items():
                if idx != 0:
                    spin.setEnabled(enabled)

        def _on_accept(self):
            """Every requested axis must be a positive number.

            Enforced here rather than accepted-and-defaulted: a zero would be
            written into the config as an extent, and `require_dimensions`
            would reject it later at a point far from where it was introduced.
            """
            if self._apply_all.isChecked():
                self._on_apply_all(True)

            missing = []
            for (idx, axis), spin in self._spins.items():
                if spin.value() <= 0:
                    missing.append(
                        f"{self._plan[idx]['filename']} — {_AXIS_LABEL.get(axis, axis)}")
            if missing:
                shown = "<br>".join(f"• {m}" for m in missing[:6])
                if len(missing) > 6:
                    shown += f"<br>… and {len(missing) - 6} more"
                self._warning.setText(
                    "These values are still unset. Every axis needs a positive "
                    f"size:<br>{shown}")
                self._warning.show()
                return
            self._warning.hide()
            self.accept()

        # -- result ------------------------------------------------------- #
        def values(self):
            """``{filename: {axis: float}}`` for the axes that were asked about."""
            out = {}
            for idx, entry in enumerate(self._plan):
                axes = {}
                for axis in entry["axes"]:
                    spin = self._spins.get((idx, axis))
                    if spin is not None and spin.value() > 0:
                        axes[axis] = float(spin.value())
                if axes:
                    out[entry["filename"]] = axes
            return out

    return _DimensionEntryDialog


class _LazyDialog:
    """Builds the real class on first use so `import` never needs Qt."""

    _cls = None

    def __call__(self, *args, **kwargs):
        if _LazyDialog._cls is None:
            _LazyDialog._cls = _build_dialog_classes()
        return _LazyDialog._cls(*args, **kwargs)


DimensionEntryDialog = _LazyDialog()
