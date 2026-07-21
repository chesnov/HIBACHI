"""parameter_widgets: extracted from helper_funcs.py (auto-split along functional seams)."""


from magicgui import magicgui  # type: ignore
from typing import Dict, Any, List, Optional, Callable
from PyQt5.QtCore import pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QAbstractSpinBox
)



class ScalesTableWidget(QWidget):
    """
    A unified table to manage Scales, Low/High Thresholds, and the per-scale
    filter parameters (Smooth Sigma, Max Connect Gap) together.

    Smoothing and gap-closing are per-scale (one value per row). The
    minimum-size filter is NOT here: it is a single GLOBAL parameter on both the
    2D and 3D tracks, exposed via the top-level `min_size` control, so it is
    never duplicated as a per-row column.

    Returns a list of dicts:
    [{'scale': 1.0, 'low': 95.0, 'high': 100.0,
      'smooth_sigma': 0.1, 'connect_max_gap_physical': 0.0}, ...]
    """
    valueChanged = pyqtSignal(object)

    # Column layout shared by both percentile and absolute variants.
    # NOTE: `min_size` is intentionally NOT a per-row column. Minimum-size
    # filtering is a single GLOBAL parameter (applied after all scales merge) on
    # both the 2D and 3D tracks, so it lives only in the top-level `min_size`
    # spinbox -- never duplicated here.
    _COLUMNS = [
        ("scale", "Scale"),
        ("low", None),   # header text depends on is_absolute
        ("high", None),  # header text depends on is_absolute
        ("smooth_sigma", "Smooth \u03c3"),
        ("connect_max_gap_physical", "Max Gap (\u00b5m)"),
    ]

    # Defaults used when a row is missing a key (e.g. legacy configs) or
    # when a brand-new row is added via "+". Low/High are threshold-mode
    # dependent: percentiles run 0-100, absolute intensities run 0.0-1.0, so a
    # new row must be seeded on the right scale (picked via _row_defaults()).
    _DEFAULTS = {
        "scale": 1.0,
        "low": 95.0,
        "high": 100.0,
        "seed": 0.0,
        "smooth_sigma": 0.1,
        "connect_max_gap_physical": 0.0,
    }

    # Absolute-intensity variant: values are normalised to [0.0, 1.0].
    _ABSOLUTE_DEFAULTS = {
        "scale": 1.0,
        "low": 0.2,
        "high": 1.0,
        "seed": 0.0,
        "smooth_sigma": 0.1,
        "connect_max_gap_physical": 0.0,
    }

    def __init__(self, initial_value: List[Dict[str, float]], label: str = "", is_absolute: bool = False):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.is_absolute = is_absolute
        
        # Controls
        btn_layout = QHBoxLayout()
        self.lbl = QLabel(label)
        self.btn_add = QPushButton("+")
        self.btn_add.setFixedWidth(30)
        self.btn_rem = QPushButton("-")
        self.btn_rem.setFixedWidth(30)
        
        self.btn_add.clicked.connect(self.add_row)
        self.btn_rem.clicked.connect(self.remove_row)
        
        btn_layout.addWidget(self.lbl)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_rem)
        self.layout.addLayout(btn_layout)

        # Column set is mode-dependent: the hysteresis `seed` column is only
        # meaningful in Absolute mode (see the segmentation stage), so the
        # Percentile table doesn't show it. Built per-instance from _COLUMNS.
        self._columns = list(self._COLUMNS)
        if is_absolute:
            insert_at = next(
                (i for i, (k, _) in enumerate(self._columns) if k == "high"), len(self._columns) - 1
            ) + 1
            self._columns.insert(insert_at, ("seed", None))

        # Table
        headers = []
        for key, text in self._columns:
            if key == "low":
                headers.append("Low (min)" if is_absolute else "Low %")
            elif key == "high":
                headers.append("High (max)" if is_absolute else "High %")
            elif key == "seed":
                headers.append("Seed (0=off)")
            else:
                headers.append(text)

        self.table = QTableWidget()
        self.table.setColumnCount(len(self._columns))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(150)
        self.table.itemChanged.connect(self._emit_change)
        self.layout.addWidget(self.table)
        
        # Populate
        self.set_value(initial_value)

    def _row_defaults(self) -> Dict[str, float]:
        """Default row values appropriate to the current threshold mode."""
        return self._ABSOLUTE_DEFAULTS if self.is_absolute else self._DEFAULTS

    def set_value(self, data: List[Dict[str, float]]):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        defaults = self._row_defaults()
        if isinstance(data, list):
            for row_idx, item in enumerate(data):
                if isinstance(item, dict):
                    self.table.insertRow(row_idx)
                    for col_idx, (key, _) in enumerate(self._columns):
                        self._set_item(row_idx, col_idx, item.get(key, defaults[key]))
        self.table.blockSignals(False)

    def _set_item(self, row, col, val):
        item = QTableWidgetItem(str(val))
        self.table.setItem(row, col, item)

    def add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        defaults = self._row_defaults()
        for col_idx, (key, _) in enumerate(self._columns):
            self._set_item(r, col_idx, defaults[key])
        self._emit_change()

    def remove_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._emit_change()

    def _emit_change(self):
        data = []
        for r in range(self.table.rowCount()):
            try:
                row_dict = {}
                for col_idx, (key, _) in enumerate(self._columns):
                    raw = float(self.table.item(r, col_idx).text())
                    # Min Size is a pixel count; keep it as an int for downstream use.
                    row_dict[key] = int(round(raw)) if key == "min_size" else raw
                data.append(row_dict)
            except (ValueError, AttributeError):
                pass 
        self.valueChanged.emit(data)

    @property
    def native(self):
        return self

def create_parameter_widget(
    param_name: str,
    param_config: Dict[str, Any],
    callback: Callable[[Any], None]
) -> Optional[Any]:
    """Creates a MagicGUI widget for a specific parameter definition."""
    param_type = param_config.get("type", "float")
    label = param_config.get("label", param_name)
    widget = None

    try:
        # --- Handle Percentile Table ---
        if param_type == "scale_table" or param_type == "scale_table_percentile":
            initial_val = param_config.get("value",[])
            widget = ScalesTableWidget(initial_val, label, is_absolute=False)
            widget.valueChanged.connect(callback)
            return widget
        
        # --- Handle Absolute Table ---
        elif param_type == "scale_table_absolute":
            initial_val = param_config.get("value",[])
            widget = ScalesTableWidget(initial_val, label, is_absolute=True)
            widget.valueChanged.connect(callback)
            return widget
        
        if param_type == "list":
            initial_list = param_config.get("value", [])
            if not isinstance(initial_list, list):
                initial_list = []
            initial_str = ", ".join(map(str, initial_list))

            def list_widget(value_str: str = initial_str):
                try:
                    new_list = [
                        float(x.strip()) for x in value_str.split(',') if x.strip()
                    ] if value_str.strip() else []
                    callback(new_list)
                    if hasattr(list_widget, 'native'):
                        list_widget.native.setStyleSheet("")
                    return value_str
                except ValueError:
                    if hasattr(list_widget, 'native'):
                        list_widget.native.setStyleSheet("background-color: #FFDDDD;")
                    return initial_str

            widget = magicgui(
                list_widget, auto_call=True,
                value_str={"widget_type": "LineEdit", "label": label}
            )

        elif param_type == "float":
            def float_widget(value: float = float(param_config.get("value", 0.0))):
                callback(value)
                return value
            widget = magicgui(
                float_widget, auto_call=True,
                value={
                    "widget_type": "FloatSpinBox", "label": label,
                    "min": float(param_config.get("min", 0)),
                    "max": float(param_config.get("max", 100)),
                    "step": float(param_config.get("step", 0.1))
                }
            )

        elif param_type == "int":
            def int_widget(value: int = int(param_config.get("value", 0))):
                callback(value)
                return value
            widget = magicgui(
                int_widget, auto_call=True,
                value={
                    "widget_type": "SpinBox", "label": label,
                    "min": int(param_config.get("min", 0)),
                    "max": int(param_config.get("max", 100)),
                    "step": int(param_config.get("step", 1))
                }
            )

        elif param_type == "bool":
            def bool_widget(value: bool = bool(param_config.get("value", False))):
                callback(value)
                return value
            widget = magicgui(
                bool_widget, auto_call=True,
                value={"widget_type": "CheckBox", "label": label}
            )

        else:
            def fallback(value: str = str(param_config.get("value", ""))):
                callback(value)
                return value
            widget = magicgui(
                fallback, auto_call=True,
                value={"widget_type": "LineEdit", "label": label}
            )

        if widget:
            widget.param_name = param_name

            # Spin boxes (int/float params) otherwise re-validate on every
            # keystroke with auto_call, which fights multi-digit entry: typing
            # "600" into a field with a minimum snaps the partial value and the
            # digits get eaten. Committing on Enter/focus-out instead makes them
            # behave like normal numeric fields. Applied app-wide here since
            # every parameter widget is built through this function.
            try:
                for sb in widget.native.findChildren(QAbstractSpinBox):
                    sb.setKeyboardTracking(False)
            except Exception:
                pass

    except Exception:
        return None
    return widget