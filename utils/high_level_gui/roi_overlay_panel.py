"""
roi_overlay_panel: draw one ROI in the multi-channel overlay and hand it to any
subset of that sample's channels.

The overlay view (``cross_channel_window.open_sample_overlay``) already shows
every channel of a sample at the same pixel dimensions, which makes it the
natural place to define a sub-region once instead of redrawing it per channel.

Three actions are docked next to the layer list:

  * Draw ROI                -- polygon layer with per-Z tagging, mirroring
                               ``GUIManager.draw_roi`` so the two paths behave
                               identically.
  * Apply to channels...    -- pick channels with checkboxes, then write the same
                               polygon into each one's ROI folder.
  * Return to full image... -- pick channels with checkboxes, then remove their
                               ROI session so they open uncropped again.

All filesystem reasoning lives in ``roi_sharing`` (pure, unit-tested); this module
is only the viewer wiring and the dialogs. Nothing here computes a bounding box
or decides what to delete.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from PyQt5.QtCore import Qt  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from .roi_sharing import (
    HAS_ROI, NEW, NO_ROI, ORPHAN, REPLACE, SHAPE_MISMATCH, UNUSABLE,
    apply_roi_clear, apply_roi_propagation, plan_roi_clear,
    plan_roi_propagation, roi_record_from_polygons,
)

ROI_LAYER_NAME = "ROI Selection"


# --------------------------------------------------------------------------- #
# Channel picker
# --------------------------------------------------------------------------- #
class ChannelSelectDialog(QDialog):
    """Checkbox list of channels, with a per-row note and a consequence summary.

    Entries whose status makes them ineligible are shown disabled with the reason
    attached, rather than hidden: a channel silently missing from the list looks
    like a bug, while a greyed row with "image size doesn't match" explains
    itself.
    """

    def __init__(self, title: str, intro: str, rows: Sequence[dict],
                 accept_label: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(460)

        outer = QVBoxLayout(self)

        header = QLabel(intro)
        header.setWordWrap(True)
        outer.addWidget(header)

        # Scroll, so a sample with many channels can't push the buttons off screen.
        host = QWidget()
        inner = QVBoxLayout(host)
        inner.setContentsMargins(2, 2, 2, 2)
        inner.setSpacing(2)

        self._boxes: List[tuple] = []  # (QCheckBox, entry)
        for row in rows:
            entry = row["entry"]
            box = QCheckBox(row["label"])
            box.setEnabled(bool(row["enabled"]))
            box.setChecked(bool(row["enabled"]) and bool(row.get("default", True)))
            if row.get("tooltip"):
                box.setToolTip(row["tooltip"])
            if not row["enabled"]:
                box.setStyleSheet("color: #8a8a8a;")
            inner.addWidget(box)
            self._boxes.append((box, entry))

        inner.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(host)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        # Bulk toggles only make sense when more than one row is selectable.
        selectable = [b for b, _ in self._boxes if b.isEnabled()]
        if len(selectable) > 1:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            all_btn = QPushButton("All")
            none_btn = QPushButton("None")
            all_btn.clicked.connect(lambda: self._set_all(True))
            none_btn.clicked.connect(lambda: self._set_all(False))
            row.addWidget(all_btn)
            row.addWidget(none_btn)
            row.addStretch(1)
            outer.addLayout(row)

        self._consequence = QLabel("")
        self._consequence.setWordWrap(True)
        self._consequence.setStyleSheet("color: #c98a00;")
        outer.addWidget(self._consequence)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(accept_label)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._ok = buttons.button(QDialogButtonBox.Ok)
        for box, _ in self._boxes:
            box.toggled.connect(self._refresh)
        self._refresh()

    def _set_all(self, state: bool) -> None:
        for box, _ in self._boxes:
            if box.isEnabled():
                box.setChecked(state)

    def _refresh(self) -> None:
        chosen = self.selected()
        # Nothing checked means nothing to do, so don't offer to proceed.
        self._ok.setEnabled(bool(chosen))
        lost = sum(len(e.get("outputs") or e.get("discards") or []) for e in chosen)
        self._consequence.setText(
            f"{len(chosen)} channel(s) selected \u2014 {lost} existing ROI file(s) "
            f"will be deleted." if lost else
            f"{len(chosen)} channel(s) selected."
        )

    def selected(self) -> List[dict]:
        return [e for box, e in self._boxes if box.isEnabled() and box.isChecked()]


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #
class OverlayROIPanel:
    """Owns the ROI drawing state and the three buttons for one overlay viewer.

    `sample_dirs` are the per-channel sample folders of the sample on screen (the
    values of the consolidated registry entry), and `full_shape` is the shape the
    channels share -- the coordinate frame every drawn polygon is expressed in.
    """

    def __init__(self, viewer, sample_name: str, sample_dirs: Sequence[str],
                 full_shape: Optional[Sequence[int]]):
        self.viewer = viewer
        self.sample_name = sample_name
        self.sample_dirs = list(sample_dirs)
        self.full_shape = tuple(int(v) for v in full_shape) if full_shape else None
        self._z_polygons: Dict[int, np.ndarray] = {}
        self._last_count = 0

    # ---- drawing -------------------------------------------------------- #
    def draw_roi(self) -> None:
        """Add a polygon layer in draw mode, tagging each polygon with its Z slice.

        Forced into 2D slice view for the same reason ``GUIManager.draw_roi``
        forces it: in 3D perspective mode napari stamps every vertex with the
        camera's focal Z, so polygons drawn on different slices become
        indistinguishable.
        """
        if self.full_shape is None:
            QMessageBox.warning(
                None, "No image",
                "The overlay has no readable image, so an ROI can't be drawn."
            )
            return

        if ROI_LAYER_NAME in self.viewer.layers:
            self.viewer.layers.remove(ROI_LAYER_NAME)

        is_3d = len(self.full_shape) == 3
        if is_3d:
            self.viewer.dims.ndisplay = 2

        self.viewer.add_shapes(
            name=ROI_LAYER_NAME,
            shape_type="polygon",
            edge_color="yellow",
            face_color=[1, 1, 0, 0.08],
            edge_width=3,
        )
        layer = self.viewer.layers[ROI_LAYER_NAME]
        layer.mode = "add_polygon"

        self._z_polygons = {}
        self._last_count = 0

        def _on_data_changed(event=None):
            if ROI_LAYER_NAME not in self.viewer.layers:
                return
            shapes = self.viewer.layers[ROI_LAYER_NAME]
            count = len(shapes.data)
            if count <= self._last_count:
                return  # an edit or deletion, not a new polygon
            self._last_count = count
            z = int(self.viewer.dims.current_step[0]) if is_3d else 0
            raw = np.array(shapes.data[-1], dtype=float)
            # In 3D, ndisplay=2 still yields (N,3) vertices; drop the Z column.
            self._z_polygons[z] = raw[:, 1:] if raw.shape[1] > 2 else raw

        layer.events.data.connect(_on_data_changed)

        if is_3d:
            msg = (
                "Draw polygons on any Z slices to define the 3D sub-region.\n\n"
                "  1. Scroll to a Z slice\n"
                "  2. Click to add vertices, press Escape to close the polygon\n"
                "  3. Scroll to the next relevant slice and repeat\n\n"
                "Drawing on only ONE slice extrudes that shape through the whole "
                "Z stack.\n\nWhen finished, click  \u2713 Apply to channels\u2026"
            )
        else:
            msg = (
                "Draw a polygon on the image to define the sub-region.\n\n"
                "  \u2022 Click to add vertices\n"
                "  \u2022 Press Escape to close the polygon\n\n"
                "When finished, click  \u2713 Apply to channels\u2026"
            )
        QMessageBox.information(None, "Draw ROI", msg)

    def _collect_polygons(self) -> Dict[int, np.ndarray]:
        """Drawn polygons, preferring the Z-tagged map built while drawing.

        Falls back to reading Z from the vertex arrays, which covers the case
        where the user drew before pressing Draw ROI (or reused a layer), exactly
        as ``confirm_roi`` does.
        """
        if self._z_polygons:
            return dict(self._z_polygons)
        if ROI_LAYER_NAME not in self.viewer.layers:
            return {}
        out: Dict[int, np.ndarray] = {}
        for raw in self.viewer.layers[ROI_LAYER_NAME].data:
            arr = np.array(raw, dtype=float)
            if arr.shape[1] == 3:
                out[int(round(float(arr[:, 0].mean())))] = arr[:, 1:]
            else:
                out[0] = arr
        return out

    # ---- apply ---------------------------------------------------------- #
    def apply_to_channels(self) -> None:
        if self.full_shape is None:
            QMessageBox.warning(None, "No image", "The overlay has no readable image.")
            return

        polygons = self._collect_polygons()
        if not polygons:
            QMessageBox.warning(
                None, "No ROI",
                "Draw a region first using  \u270f Draw ROI."
            )
            return

        try:
            record = roi_record_from_polygons(polygons, self.full_shape)
        except ValueError as exc:
            QMessageBox.warning(None, "ROI not usable", str(exc))
            return

        plan = plan_roi_propagation(self.sample_dirs, self.full_shape)
        rows = []
        for entry in plan:
            status = entry.get("status")
            channel = entry.get("channel", "?")
            if status == NEW:
                rows.append({
                    "entry": entry, "enabled": True, "default": True,
                    "label": f"{channel}  \u2014  no ROI yet",
                    "tooltip": entry.get("roi_dir", ""),
                })
            elif status == REPLACE:
                n = len(entry.get("discards") or [])
                rows.append({
                    "entry": entry, "enabled": True, "default": True,
                    "label": (f"{channel}  \u2014  replaces existing ROI"
                              + (f" ({n} result file(s) deleted)" if n else "")),
                    "tooltip": entry.get("roi_dir", ""),
                })
            else:
                reason = entry.get("reason") or (
                    "image size doesn't match" if status == SHAPE_MISMATCH
                    else "not a usable image folder"
                )
                rows.append({
                    "entry": entry, "enabled": False,
                    "label": f"{channel}  \u2014  cannot apply: {reason}",
                })

        bbox = record["bbox"]
        z_note = ""
        if bbox.get("z1") is not None and len(self.full_shape) == 3:
            z_note = (f", Z {bbox['z0']}\u2013{bbox['z1']}"
                      if bbox["z1"] - bbox["z0"] != self.full_shape[0]
                      else ", all Z")
        intro = (
            f"Apply this region to the channels of '{self.sample_name}'.\n\n"
            f"Region: {bbox['y1'] - bbox['y0']} \u00d7 {bbox['x1'] - bbox['x0']} px"
            f"{z_note}, from {len(record['z_polygons'])} polygon(s).\n\n"
            "Each channel crops its own image to this region and rebuilds its own "
            "config on next open. Processing starts again from step 1 for the "
            "channels you select."
        )

        dlg = ChannelSelectDialog(
            "Apply ROI to channels", intro, rows, "Apply ROI"
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        chosen = dlg.selected()
        if not chosen:
            return

        result = apply_roi_propagation(chosen, record)
        self._report(
            "ROI applied",
            f"{len(result['written'])} channel(s) now use this region.",
            result["errors"],
            tail=("Open each channel from the project view to segment the "
                  "sub-region. Choose 'Yes' when it asks to load the ROI session."),
        )

    # ---- clear ---------------------------------------------------------- #
    def return_to_full_image(self) -> None:
        plan = plan_roi_clear(self.sample_dirs)
        rows = []
        for entry in plan:
            status = entry.get("status")
            channel = entry.get("channel", "?")
            if status == HAS_ROI:
                n = len(entry.get("outputs") or [])
                rows.append({
                    "entry": entry, "enabled": True, "default": True,
                    "label": (f"{channel}  \u2014  ROI session active"
                              + (f" ({n} result file(s) deleted)" if n else "")),
                    "tooltip": entry.get("roi_dir", ""),
                })
            elif status == ORPHAN:
                n = len(entry.get("discards") or [])
                rows.append({
                    "entry": entry, "enabled": True, "default": False,
                    "label": (f"{channel}  \u2014  already full image, "
                              f"{n} leftover file(s) can be removed"),
                    "tooltip": entry.get("roi_dir", ""),
                })
            elif status == NO_ROI:
                rows.append({
                    "entry": entry, "enabled": False,
                    "label": f"{channel}  \u2014  no ROI session",
                })
            else:
                rows.append({
                    "entry": entry, "enabled": False,
                    "label": (f"{channel}  \u2014  "
                              f"{entry.get('reason', 'not a usable image folder')}"),
                })

        if not any(r["enabled"] for r in rows):
            QMessageBox.information(
                None, "Nothing to clear",
                f"No channel of '{self.sample_name}' has an ROI session.\n\n"
                "They all already open on the full image."
            )
            return

        intro = (
            f"Return channels of '{self.sample_name}' to the full image.\n\n"
            "This deletes each selected channel's ROI session, including any "
            "results computed on the sub-region. Full-image results are NOT "
            "affected \u2014 they live in a separate folder. The channel then opens "
            "uncropped, without asking about an ROI."
        )
        dlg = ChannelSelectDialog(
            "Return to full image", intro, rows, "Return to full image"
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        chosen = dlg.selected()
        if not chosen:
            return

        result = apply_roi_clear(chosen)
        self._report(
            "Returned to full image",
            f"{len(result['cleared'])} channel(s) will now open on the full image.",
            result["errors"],
        )

    # ---- shared reporting ----------------------------------------------- #
    @staticmethod
    def _report(title: str, headline: str, errors: Sequence[dict],
                tail: str = "") -> None:
        body = headline
        if tail:
            body += f"\n\n{tail}"
        if errors:
            detail = "\n".join(
                f"\u2022 {os.path.basename(e['sample_dir'])}: {e['error']}"
                for e in errors
            )
            QMessageBox.warning(
                None, title,
                f"{body}\n\n{len(errors)} channel(s) could not be updated:\n{detail}"
            )
        else:
            QMessageBox.information(None, title, body)


# --------------------------------------------------------------------------- #
# Docking
# --------------------------------------------------------------------------- #
def add_overlay_roi_panel(viewer, sample_name: str, sample_dirs: Sequence[str],
                          full_shape: Optional[Sequence[int]]):
    """Dock the ROI controls into an overlay viewer. Returns (dock, panel).

    Reuses the panel styling helpers from ``app_launch`` so this dock matches the
    segmentation viewer's controls; falls back to plain buttons if those helpers
    can't be imported, since a styling failure must not cost the feature.
    """
    panel = OverlayROIPanel(viewer, sample_name, sample_dirs, full_shape)

    try:
        from .app_launch import (
            _compact_button, _give_layer_list_room, _lock_panel_height,
            _section_header,
        )
    except Exception:
        _compact_button = None
        _section_header = None
        _lock_panel_height = None
        _give_layer_list_room = None

    def _button(text: str, tooltip: str) -> QPushButton:
        if _compact_button is not None:
            return _compact_button(text, tooltip)
        btn = QPushButton(text)
        btn.setToolTip(tooltip)
        return btn

    container = QWidget()
    outer = QVBoxLayout(container)
    outer.setContentsMargins(6, 4, 6, 6)
    outer.setSpacing(4)

    if _section_header is not None:
        outer.addWidget(_section_header("Shared sub-region (ROI)"))

    btn_draw = _button(
        "\u270f Draw ROI",
        "Draw a sub-region on the overlay.\n"
        "Click to add vertices, press Escape to close the polygon.\n"
        "In 3D, draw on one slice to extrude through all Z, or on several\n"
        "slices to define a true 3D region.",
    )
    btn_apply = _button(
        "\u2713 Apply to channels\u2026",
        "Give the drawn region to any subset of this sample's channels.\n"
        "Each channel crops its own image and rebuilds its own config,\n"
        "then segments the sub-region independently.",
    )
    btn_clear = _button(
        "\u21a9 Return to full image\u2026",
        "Remove the ROI session from any subset of this sample's channels\n"
        "so they open uncropped. Deletes results computed on the\n"
        "sub-region; full-image results are kept.",
    )

    btn_draw.clicked.connect(panel.draw_roi)
    btn_apply.clicked.connect(panel.apply_to_channels)
    btn_clear.clicked.connect(panel.return_to_full_image)

    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(btn_draw)
    row.addWidget(btn_apply)
    outer.addLayout(row)
    outer.addWidget(btn_clear)

    dock = viewer.window.add_dock_widget(container, area="left", name="ROI")
    if _lock_panel_height is not None:
        _lock_panel_height(container, dock)
    if _give_layer_list_room is not None:
        _give_layer_list_room(viewer, dock)
    return dock, panel
