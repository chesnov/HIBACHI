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
#: Rank of the image a pixel-count dict was probed from, carried alongside the
#: per-axis counts. The counts alone cannot answer it: a 2D probe reports
#: ``z: 1`` and so does a single-slice z-stack, and the distinction decides
#: whether the user is asked for a depth at all. Every consumer reads this dict
#: with ``.get(axis)``, so the extra key is inert for them.
NDIM_KEY = "ndim"


def pixels_ndim(pixels: Optional[Dict[str, Any]]):
    """Rank recorded by a pixel-count probe, or None if it did not record one."""
    try:
        value = (pixels or {}).get(NDIM_KEY)
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def axes_for_mode(mode=None, ndim=None) -> Tuple[str, ...]:
    """Axes an acquisition needs: a 2D image has no Z.

    Rank comes from `ndim` -- the image's own dimensionality -- whenever the
    caller can supply it. `mode` is honoured only as a fallback, for callers
    with no image to consult yet, and is otherwise ignored.

    The mode string can no longer answer this. With one mode,
    ``mode.endswith('_2d')`` is always False, so this returned ('x','y','z')
    for every image including 2D ones -- which asks the user for a depth that
    does not exist, writes a z extent into a 2D config, and reports z as an
    unscaled axis on every 2D organize. Same shape as the two guards in §5.5:
    the test was only ever meaningful because two mode strings differed.
    """
    if ndim is not None:
        try:
            return ("x", "y", "z") if int(ndim) >= 3 else ("x", "y")
        except (TypeError, ValueError):
            pass
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


#: How close an implied spacing must be to exactly 1.0 um/pixel before the total
#: is treated as "this is really just the pixel count". Deliberately tight: the
#: failure mode being caught produces a ratio of exactly 1.0, so a loose
#: tolerance would only add false prompts on genuinely near-unit images.
UNIT_SCALE_REL_TOL = 1e-3

#: Reason an axis is being asked about, so the dialog can word itself correctly.
REASON_MISSING = "missing"        # no scale at all -> field starts empty
REASON_UNIT_SCALE = "unit_scale"  # total == pixel count -> field pre-filled to confirm


def implied_spacing(total, pixels) -> Optional[float]:
    """Microns per pixel implied by a total extent and a pixel count."""
    try:
        total_f, pixels_f = float(total), float(pixels)
    except (TypeError, ValueError):
        return None
    if total_f != total_f or pixels_f != pixels_f:      # NaN
        return None
    if pixels_f <= 0 or total_f <= 0:
        return None
    return total_f / pixels_f


def looks_like_pixel_count(total, pixels, rel_tol: float = UNIT_SCALE_REL_TOL) -> bool:
    """True if `total` microns is indistinguishable from `pixels` pixels.

    A second line of defence, deliberately placed on the RESULT rather than on
    the metadata. ``scale_gaps`` can only judge a per-pixel spacing the image
    file reported; it cannot see a total that arrived some other way. The
    clearest example is a metadata CSV with pixel counts pasted into the
    'Width (um)' column: those numbers are present, positive and finite, so
    every existing check passes them and the config is stamped as calibrated
    from a CSV -- while implying exactly 1.0 um/pixel.

    A real 1.0 um/pixel image does exist, so this must never be treated as an
    error. It only decides whether to ASK, with the value pre-filled so
    confirming is one click.
    """
    spacing = implied_spacing(total, pixels)
    if spacing is None:
        return False
    return abs(spacing - 1.0) <= rel_tol


def unit_scale_axes(
    totals: Optional[Dict[str, Any]],
    pixels: Optional[Dict[str, Any]],
    mode=None,
    rel_tol: float = UNIT_SCALE_REL_TOL,
    ndim=None,
) -> Tuple[str, ...]:
    """Axes whose total extent is indistinguishable from their pixel count.

    Rank defaults to whatever the probe in `pixels` recorded, so a caller that
    already has the counts does not have to supply it twice.
    """
    if not totals or not pixels:
        return ()
    if ndim is None:
        ndim = pixels_ndim(pixels)
    return tuple(
        a for a in axes_for_mode(mode, ndim)
        if looks_like_pixel_count(totals.get(a), pixels.get(a), rel_tol)
    )


def scale_gaps(meta: Optional[Dict[str, Any]], mode=None,
               ndim=None) -> Tuple[str, ...]:
    """Axes for which `meta` provides no trustworthy scale.

    Per-axis on purpose. The previous check looked only at X and Y, so a 3D
    image with a genuine X/Y but no Z spacing passed as fully calibrated -- and
    Z is exactly the axis that is most often missing.
    """
    axes = axes_for_mode(mode, ndim)
    if not isinstance(meta, dict) or not meta.get("found"):
        return axes
    return tuple(a for a in axes if not _is_real_spacing(meta.get(a)))


def resulting_totals(
    meta: Optional[Dict[str, Any]],
    pixels: Optional[Dict[str, Any]],
    mode=None,
    csv_override: Optional[Dict[str, Optional[float]]] = None,
    ndim=None,
) -> Dict[str, Optional[float]]:
    """The total extent each axis would end up with, before any manual entry.

    Mirrors what the scaffolding computes (``spacing * pixel_count``, with a CSV
    row replacing the product per axis) so the suspicious-total check judges the
    same numbers that would actually be written to the config.
    """
    out: Dict[str, Optional[float]] = {}
    pixels = pixels or {}
    if ndim is None:
        ndim = pixels_ndim(pixels)
    for axis in axes_for_mode(mode, ndim):
        if csv_override and csv_override.get(axis) is not None:
            out[axis] = float(csv_override[axis])
            continue
        spacing = (meta or {}).get(axis)
        count = pixels.get(axis)
        try:
            out[axis] = float(spacing) * float(count)
        except (TypeError, ValueError):
            out[axis] = None
    return out


def plan_manual_entry(
    files_meta: Sequence[Tuple[str, Optional[Dict[str, Any]], Dict[str, int]]],
    mode=None,
    csv_overrides: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
    match_override=None,
    check_unit_scale: bool = True,
) -> List[Dict[str, Any]]:
    """Work out who needs to be asked, and for which axes, and why.

    `files_meta` is a sequence of ``(filename, metadata_dict, pixel_counts)``
    where `pixel_counts` maps axis -> pixel count.

    An axis is asked about for one of two reasons:

    ``REASON_MISSING``
        Both automatic routes failed -- the file's metadata has no trustworthy
        spacing for the axis and no CSV row pins it down. The field starts empty.
        This is also what closes the partial-CSV hole: a CSV giving only
        'Width (um)' used to suppress the warning for Y and Z as well.

    ``REASON_UNIT_SCALE``
        A value *was* found, but the resulting total is indistinguishable from
        the axis's pixel count, i.e. exactly 1.0 um/pixel. Checked on the RESULT
        rather than on the metadata, so it catches a total that arrived from a
        CSV as well -- pixel counts pasted into the 'Width (um)' column are
        present, positive and finite, so nothing else questions them. A genuine
        1.0 um/pixel image is possible, so the field is PRE-FILLED with the
        value and confirming it is one click.

    Returns one entry per file needing input:
    ``{'filename', 'axes', 'pixels', 'from_csv', 'reasons', 'current'}``.
    """
    csv_overrides = csv_overrides or {}
    plan: List[Dict[str, Any]] = []

    for filename, meta, pixels in files_meta:
        override = None
        if csv_overrides:
            if match_override is not None:
                override = match_override(csv_overrides, filename)
            else:
                override = csv_overrides.get(filename)

        # Rank per image, from that image's own probe -- not from the preset's
        # mode, which is now the same string for 2D and 3D acquisitions. A
        # mixed-rank batch therefore asks each file only about the axes it has.
        ndim = pixels_ndim(pixels)

        missing = set(scale_gaps(meta, mode, ndim))
        from_csv = set()
        if override:
            for axis in list(missing):
                if override.get(axis) is not None:
                    missing.discard(axis)
                    from_csv.add(axis)

        totals = resulting_totals(meta, pixels, mode, override, ndim)

        # Suspicious totals only matter for axes we believe we HAVE a value for;
        # a missing axis is already being asked about, and its placeholder total
        # (pixel count x 1.0) would otherwise flag every one of them redundantly.
        suspicious = set()
        if check_unit_scale:
            suspicious = {
                a for a in unit_scale_axes(totals, pixels, mode, ndim=ndim)
                if a not in missing
            }

        ask = missing | suspicious
        if not ask:
            continue

        reasons = {}
        current: Dict[str, Optional[float]] = {}
        for axis in axes_for_mode(mode, ndim):
            if axis in missing:
                reasons[axis] = REASON_MISSING
            elif axis in suspicious:
                reasons[axis] = REASON_UNIT_SCALE
                current[axis] = totals.get(axis)

        plan.append({
            "filename": filename,
            "axes": tuple(a for a in axes_for_mode(mode, ndim) if a in ask),
            "pixels": dict(pixels or {}),
            "from_csv": tuple(sorted(from_csv)),
            "reasons": reasons,
            "current": current,
        })
    return plan


def per_axis_sources(
    mode,
    meta: Optional[Dict[str, Any]],
    csv_axes: Iterable[str] = (),
    manual_axes: Iterable[str] = (),
    ndim=None,
) -> Dict[str, str]:
    """Provenance of each axis, for `combine_sources`."""
    csv_axes, manual_axes = set(csv_axes), set(manual_axes)
    gaps = set(scale_gaps(meta, mode, ndim))
    out: Dict[str, str] = {}
    for axis in axes_for_mode(mode, ndim):
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
        ``dimensions`` block and the CSV's
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
            # Counted over AXES, not files: one image can be missing a Z scale
            # while its X total merely looks suspicious, and describing that as
            # purely a confirmation would understate what the user has to do.
            all_reasons = [
                r for e in self._plan for r in (e.get("reasons") or {}).values()
            ]
            any_missing = any(r == REASON_MISSING for r in all_reasons)
            any_suspect = any(r == REASON_UNIT_SCALE for r in all_reasons)
            if any_suspect and not any_missing:
                lead = (
                    f"<b>Please confirm the dimensions of {n} image(s).</b><br><br>"
                    "Their recorded size is exactly the same as their size in "
                    "pixels, which means 1 micron per pixel. That is sometimes "
                    "correct, but far more often it means the scale was never "
                    "set, or pixel counts were entered in a microns column."
                )
            elif any_suspect:
                lead = (
                    f"<b>{n} image(s) need their dimensions checked.</b><br><br>"
                    "Some have no usable scale at all. Others have a recorded "
                    "size identical to their pixel count (1 micron per pixel), "
                    "which is occasionally right but usually means the scale was "
                    "never set. Pre-filled values are the current ones -- change "
                    "them or confirm them."
                )
            else:
                lead = (
                    f"<b>{n} image(s) have no usable scale information.</b><br><br>"
                    "HIBACHI could not read a physical scale from the image "
                    "metadata, and no metadata CSV supplied one."
                )
            head = QLabel(
                lead
                + "<br><br>Enter the <b>total</b> physical size of each image "
                "below. Without it, every size, distance and density would be "
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

            reasons = entry.get("reasons") or {}
            current = entry.get("current") or {}
            if any(r == REASON_UNIT_SCALE for r in reasons.values()):
                warn = QLabel(
                    "<i>Recorded size equals the pixel count (1 µm/pixel). "
                    "Correct it, or confirm if that is genuinely right.</i>")
                warn.setWordWrap(True)
                layout.addWidget(warn)

            form = QFormLayout()
            pixels = entry.get("pixels") or {}
            for axis in entry["axes"]:
                spin = QDoubleSpinBox()
                spin.setDecimals(4)
                spin.setRange(0.0, 1e9)
                spin.setSingleStep(1.0)
                spin.setSpecialValueText("(not set)")

                # A suspicious-but-present axis is PRE-FILLED with its current
                # value, so a user who knows the image really is 1 µm/pixel just
                # presses OK. A missing axis starts at 0 ("(not set)") and must
                # be filled in. Same widget, different starting point, because
                # the two cases need different amounts of work from the user.
                prefill = current.get(axis)
                spin.setValue(float(prefill) if prefill else 0.0)

                px = pixels.get(axis)
                if px:
                    spin.setSuffix(
                        f"   ({int(px)} px {_AXIS_PIXEL_LABEL.get(axis, axis)})")
                    spin.setToolTip(
                        f"Total extent along {axis.upper()}. This axis is "
                        f"{int(px)} pixels, so entering T gives a spacing of "
                        f"T/{int(px)} µm per pixel.")
                label = _AXIS_LABEL.get(axis, axis)
                if reasons.get(axis) == REASON_UNIT_SCALE:
                    label += " \u26a0"        # marks the axis to double-check
                form.addRow(label, spin)
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