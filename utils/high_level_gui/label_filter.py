"""Show only some labels of a Labels layer, or hide some, without touching its data.

Opened from the viewer's Layers menu ("Label Filter..."), as a small floating
window: pick a Labels layer, type label IDs
("93, 95, 66" or ranges like "10-20"), and choose "Show only" or "Hide". The
layer's colours are swapped for a colormap in which the other (or the listed)
labels are transparent; the segmentation itself is never modified. "Show all"
restores the layer's own colours.
"""

from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import Qt  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QAction, QComboBox, QDialog, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout, QWidget,
)
from napari.layers import Labels  # type: ignore
from napari.utils.colormaps import DirectLabelColormap  # type: ignore

_MAX_DIRECT = 200_000   # "Hide" lists a colour for every other label up to the largest one


def parse_ids(text: str) -> List[int]:
    """'93, 95 66;10-12' -> [10, 11, 12, 66, 93, 95]. Raises ValueError on bad input."""
    ids = set()
    for tok in text.replace(";", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                lo, hi = hi, lo
            ids.update(range(lo, hi + 1))
        else:
            ids.add(int(tok))
    ids.discard(0)
    return sorted(ids)


class LabelFilterWidget(QWidget):
    """Dock widget that shows or hides chosen labels of a Labels layer."""

    def __init__(self, viewer, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.viewer = viewer
        self._original: Dict[int, object] = {}     # id(layer) -> its own colormap
        self._max_label: Dict[int, int] = {}       # id(layer) -> largest label (cached)

        self.layer_box = QComboBox()
        self.ids_edit = QLineEdit()
        self.ids_edit.setPlaceholderText("e.g. 93, 95, 66 or 10-20")
        self.btn_apply = QPushButton("Apply")
        self.btn_reset = QPushButton("Show all")
        self.status = QLabel("")
        self.status.setWordWrap(True)

        self.mode_box = QComboBox()
        self.mode_box.addItems(["Show only", "Hide"])
        self.layer_box.setToolTip("Labels layer to filter")
        self.ids_edit.setToolTip("Label IDs: commas or spaces, ranges like 10-20; Enter applies")
        self.btn_reset.setToolTip("Restore all labels of this layer")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)
        row = QHBoxLayout()
        row.addWidget(self.layer_box, 1)
        row.addWidget(self.mode_box)
        lay.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.ids_edit, 1)
        row.addWidget(self.btn_apply)
        row.addWidget(self.btn_reset)
        lay.addLayout(row)
        lay.addWidget(self.status)

        self.btn_apply.clicked.connect(self.apply)
        self.btn_reset.clicked.connect(self.reset)
        self.ids_edit.returnPressed.connect(self.apply)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self._refresh_layers()

    # ------------------------------------------------------------------ helpers
    def _labels_layers(self) -> List[Labels]:
        return [l for l in self.viewer.layers if isinstance(l, Labels)]

    def _refresh_layers(self, *_args) -> None:
        current = self.layer_box.currentText()
        names = [l.name for l in self._labels_layers()]
        self.layer_box.blockSignals(True)
        self.layer_box.clear()
        self.layer_box.addItems(names)
        if current in names:
            self.layer_box.setCurrentText(current)
        self.layer_box.blockSignals(False)
        live = {id(l) for l in self._labels_layers()}
        for k in [k for k in self._original if k not in live]:
            self._original.pop(k, None)
            self._max_label.pop(k, None)

    def _layer(self) -> Optional[Labels]:
        name = self.layer_box.currentText()
        if name and name in self.viewer.layers and isinstance(self.viewer.layers[name], Labels):
            return self.viewer.layers[name]
        return None

    def _max_of(self, layer: Labels) -> int:
        key = id(layer)
        if key not in self._max_label:
            self._max_label[key] = int(np.asarray(layer.data).max())
        return self._max_label[key]

    # ------------------------------------------------------------------ actions
    def apply(self) -> None:
        layer = self._layer()
        if layer is None:
            self.status.setText("No Labels layer selected.")
            return
        try:
            ids = parse_ids(self.ids_edit.text())
        except ValueError:
            self.status.setText("Could not read the IDs: use numbers, commas and ranges like 10-20.")
            return
        if not ids:
            self.status.setText("Enter at least one label ID.")
            return
        key = id(layer)
        if key not in self._original:
            self._original[key] = layer.colormap
        own = self._original[key]
        clear = np.zeros(4, dtype=np.float32)
        if self.mode_box.currentIndex() == 0:          # Show only
            cols = own.map(np.asarray(ids))
            color_dict = {None: clear, 0: clear}
            color_dict.update({int(i): c for i, c in zip(ids, cols)})
            msg = f"Showing {len(ids)} label(s) on '{layer.name}'."
        else:
            top = self._max_of(layer)
            if top > _MAX_DIRECT:
                self.status.setText(f"Too many labels ({top}) to hide selectively; use 'Show only'.")
                return
            keep = np.setdiff1d(np.arange(1, top + 1), np.asarray(ids))
            cols = own.map(keep) if len(keep) else np.zeros((0, 4))
            color_dict = {None: clear, 0: clear}
            color_dict.update({int(i): c for i, c in zip(keep, cols)})
            msg = f"Hiding {len(ids)} label(s) on '{layer.name}'."
        layer.colormap = DirectLabelColormap(color_dict=color_dict)
        layer.refresh()
        self.status.setText(msg)

    def reset(self) -> None:
        layer = self._layer()
        if layer is None:
            return
        own = self._original.pop(id(layer), None)
        if own is not None:
            layer.colormap = own
            layer.refresh()
        self.status.setText(f"All labels shown on '{layer.name}'.")


def install_label_filter_menu(viewer) -> None:
    """Adds "Label Filter..." to the viewer's Layers menu; it opens a small floating window."""
    qt_window = viewer.window._qt_window
    menu = None
    for act in qt_window.menuBar().actions():
        if act.menu() is not None and act.text().replace("&", "").strip().lower() == "layers":
            menu = act.menu()
            break
    if menu is None:                                  # no Layers menu: add our own
        menu = qt_window.menuBar().addMenu("&Labels")
    state = {"dialog": None}

    def open_filter() -> None:
        dlg = state["dialog"]
        if dlg is None:
            dlg = QDialog(qt_window, Qt.Tool)
            dlg.setWindowTitle("Label Filter")
            box = QVBoxLayout(dlg)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(LabelFilterWidget(viewer, dlg))
            dlg.resize(360, dlg.sizeHint().height())
            state["dialog"] = dlg
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    menu.addSeparator()
    action = QAction("Label Filter...", qt_window)
    action.setStatusTip("Show only, or hide, chosen labels of a Labels layer")
    action.triggered.connect(open_filter)
    menu.addAction(action)
    viewer.window._label_filter_action = action       # keep a reference alive