"""
turntable.py -- Record a movie of the current napari view.

This adds a "🎥 Record Movie" button to the single-channel and channel-merged
napari viewers. Clicking it opens a small settings dialog offering two kinds of
movie, then renders it one frame at a time, grabbing a screenshot per frame:

* **Rotate the 3D view** -- a turntable: the camera spins at a chosen speed,
  direction and axis.
* **Fly through Z slices** -- the view steps through the stack's slices in 2D,
  over a chosen range, speed and direction (including back and forth). The
  current zoom and pan are kept, so a zoomed-in region can be flown through.

Both share the output options (fps, format, resolution, which layers), and both
put the viewer back exactly as it was afterwards.

Design notes
------------
* Rendering is done by direct per-frame screenshotting rather than by handing
  keyframes to napari-animation. That gives us exact constant-speed control,
  arbitrary direction/axis, a live progress bar, and a Cancel button -- and it
  keeps the core path free of a hard plugin dependency.
* MP4 export uses imageio with the ffmpeg binary bundled by ``imageio-ffmpeg``
  (a static build shipped on PyPI for Windows / Linux / macOS incl. Apple
  Silicon), so no system ffmpeg install is required on any OS.
* GIF export is written with Pillow, which needs no ffmpeg at all, so the
  feature still works even if the ffmpeg binary is somehow unavailable.

Public API
----------
    make_turntable_widget(viewer) -> QWidget
    add_turntable_button(viewer) -> QDockWidget | None
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import time
import logging
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np

from PyQt5.QtCore import Qt, QSettings
from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QDoubleSpinBox, QSpinBox, QRadioButton, QButtonGroup,
    QListWidget, QListWidgetItem, QPushButton, QWidget, QLineEdit, QFileDialog,
    QMessageBox, QProgressDialog, QCheckBox,
)

logger = logging.getLogger(__name__)

# Suppress the console window that would flash when a GUI (pythonw) process
# spawns ffmpeg on Windows. 0 (no flag) on other platforms.
_CREATE_NO_WINDOW = 0x08000000 if sys.platform.startswith("win") else 0

# Axis choices: label -> index of the napari camera.angles component we sweep.
# napari's camera.angles is a 3-tuple of Euler angles (degrees). Which one reads
# as the "vertical" spin depends on the current view, so all three are exposed
# and the user can switch if the default axis looks wrong.
_AXIS_CHOICES = [
    ("Vertical axis (turntable)", 0),
    ("Horizontal axis (tumble)", 1),
    ("Depth axis (roll)", 2),
]
_FORMATS = ["mp4", "gif"]

MODE_ROTATE = "rotate"
MODE_ZSWEEP = "zsweep"
_MODES = [(MODE_ROTATE, "Rotate the 3D view"), (MODE_ZSWEEP, "Fly through Z slices")]

Z_FORWARD = "forward"
Z_BACKWARD = "backward"
Z_PINGPONG = "pingpong"
_Z_DIRECTIONS = [(Z_FORWARD, "First slice \u2192 last"),
                 (Z_BACKWARD, "Last slice \u2192 first"),
                 (Z_PINGPONG, "Back and forth")]

_SETTINGS_ORG = "HIBACHI"
_SETTINGS_APP = "Turntable"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class TurntableSettings:
    """User-tunable turntable options. Persisted between sessions via QSettings."""
    speed_dps: float = 90.0        # rotation speed, degrees per second
    clockwise: bool = True         # direction of spin
    axis_index: int = 0            # which camera.angles component to sweep
    turns: float = 1.0             # number of full revolutions
    fps: int = 30                  # frames per second of the output
    canvas_only: bool = True       # capture just the canvas (no napari UI chrome)
    scale: float = 1.0             # resolution multiplier for the screenshot
    fmt: str = "mp4"               # "mp4" or "gif"
    use_visible_layers: bool = True  # True: whatever is currently visible; False: custom set
    custom_layer_names: List[str] = field(default_factory=list)
    last_dir: str = ""             # remembered output directory
    mode: str = MODE_ROTATE        # MODE_ROTATE or MODE_ZSWEEP
    z_speed: float = 10.0          # fly-through speed, slices per second
    z_direction: str = Z_FORWARD   # Z_FORWARD | Z_BACKWARD | Z_PINGPONG
    # The slice range is the data's, not a preference: never persisted, and
    # set from the viewer each time the dialog opens.
    z_start: int = 0
    z_end: int = -1                # -1: through the last slice

    # -- persistence --------------------------------------------------------- #
    @classmethod
    def load(cls) -> "TurntableSettings":
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        d = cls()
        try:
            d.speed_dps = float(s.value("speed_dps", d.speed_dps))
            d.clockwise = _as_bool(s.value("clockwise", d.clockwise))
            d.axis_index = int(s.value("axis_index", d.axis_index))
            d.turns = float(s.value("turns", d.turns))
            d.fps = int(s.value("fps", d.fps))
            d.canvas_only = _as_bool(s.value("canvas_only", d.canvas_only))
            d.scale = float(s.value("scale", d.scale))
            d.fmt = str(s.value("fmt", d.fmt))
            d.use_visible_layers = _as_bool(s.value("use_visible_layers", d.use_visible_layers))
            d.last_dir = str(s.value("last_dir", d.last_dir) or "")
            mode = str(s.value("mode", d.mode))
            d.mode = mode if mode in dict(_MODES) else MODE_ROTATE
            d.z_speed = float(s.value("z_speed", d.z_speed))
            z_dir = str(s.value("z_direction", d.z_direction))
            d.z_direction = z_dir if z_dir in dict(_Z_DIRECTIONS) else Z_FORWARD
        except Exception as exc:  # corrupt/legacy value -> fall back to defaults
            logger.warning("Could not load turntable settings (%s); using defaults.", exc)
            d = cls()
        return d

    def save(self) -> None:
        s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        for k, v in asdict(self).items():
            if k in ("custom_layer_names", "z_start", "z_end"):
                continue  # view-/data-specific; not worth persisting
            s.setValue(k, v)

    # -- derived quantities -------------------------------------------------- #
    @property
    def total_degrees(self) -> float:
        return 360.0 * max(0.0, self.turns)

    @property
    def total_frames(self) -> int:
        if self.speed_dps <= 0:
            return 1
        duration_s = self.total_degrees / self.speed_dps
        return max(1, int(round(duration_s * self.fps)))

    @property
    def duration_s(self) -> float:
        return self.total_degrees / self.speed_dps if self.speed_dps > 0 else 0.0


def zsweep_plan(start: int, end: int, speed: float, fps: int,
                direction: str = Z_FORWARD) -> List[int]:
    """The slice index shown in each frame of a Z fly-through.

    Pure, so it can be tested without a viewer. `start` and `end` are inclusive
    and may be given in either order.

    `speed` is slices per second, so each slice is on screen for 1/speed s:

    * slower than the frame rate -> every slice is held for the SAME number of
      frames (uneven holds read as judder);
    * as fast or faster -> slices are evenly skipped, landing exactly on both
      ends.

    Back and forth holds the turning slice once, not twice, and ends before the
    first slice so a looping GIF or player does not stutter on a repeat.
    """
    lo, hi = sorted((int(start), int(end)))
    count = hi - lo + 1
    if count <= 1:
        return [lo]
    speed = max(1e-6, float(speed))
    fps = max(1, int(fps))
    frames_per_slice = fps / speed
    if frames_per_slice >= 1.0:
        hold = max(1, int(round(frames_per_slice)))
        one_way = [lo + i for i in range(count) for _ in range(hold)]
    else:
        n = max(2, int(round((count - 1) / speed * fps)) + 1)
        one_way = [lo + int(round(i * (count - 1) / (n - 1))) for i in range(n)]
    if direction == Z_BACKWARD:
        return [lo + hi - z for z in one_way]
    if direction == Z_PINGPONG:
        back = [z for z in reversed(one_way) if z not in (lo, hi)]
        return one_way + back
    return one_way


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# ffmpeg discovery (cross-OS)
# --------------------------------------------------------------------------- #
# The app launches the environment's Python directly (no `conda activate`), so
# the env's bin/ directory is usually NOT on PATH. A perfectly good ffmpeg living
# inside the env can therefore be invisible to imageio unless we point at it
# explicitly. _ensure_ffmpeg checks, in order: an existing env var, the
# imageio-ffmpeg managed binary, a real ffmpeg inside the active env, and finally
# a system ffmpeg on PATH -- and records *why* it failed so the UI can say
# something actionable.
_FFMPEG_REASON = ""


def _env_ffmpeg_candidates() -> List[str]:
    """Possible ffmpeg locations inside the active environment (which may not be
    on PATH because the app runs the env interpreter without activation)."""
    prefix = sys.prefix
    if os.name == "nt":
        dirs = (os.path.join(prefix, "Library", "bin"),
                os.path.join(prefix, "Scripts"),
                prefix)
        names = ("ffmpeg.exe",)
    else:
        dirs = (os.path.join(prefix, "bin"),)
        names = ("ffmpeg",)
    return [os.path.join(d, n) for d in dirs for n in names]


def _which_via_login_shell(binary: str) -> Optional[str]:
    """Resolve `binary` through the user's *login* shell PATH.

    A GUI-launched process inherits a minimal PATH that often omits the
    directories a package manager installs into (e.g. Homebrew on macOS). The
    login shell loads the user's real PATH exactly as an interactive terminal
    would, so this finds the binary wherever it actually lives -- without this
    code hardcoding any directory. Windows GUI processes already inherit the
    full user PATH, so this is a POSIX-only fallback. Best-effort; returns None
    on any failure (e.g. an exotic shell that lacks `command -v`).
    """
    if os.name == "nt":
        return None
    shell = os.environ.get("SHELL") or "/bin/sh"
    try:
        proc = subprocess.run(
            [shell, "-l", "-c", f"command -v {binary}"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return None
    for line in reversed(proc.stdout.splitlines()):
        cand = line.strip()
        if cand and os.path.isabs(cand) and os.path.exists(cand):
            return cand
    return None


def _ensure_ffmpeg() -> bool:
    """Locate an mp4-capable ffmpeg and point imageio at it. Returns True on
    success; on failure sets _FFMPEG_REASON to a human-readable explanation."""
    global _FFMPEG_REASON

    # 1. Already configured and present.
    exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if exe and os.path.exists(exe):
        return True

    tried: List[str] = []

    # 2. imageio-ffmpeg's managed binary (bundled in recent wheels).
    try:
        import imageio_ffmpeg
        try:
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            exe = None
            tried.append(f"imageio-ffmpeg installed but no usable binary ({exc})")
        if exe and os.path.exists(exe):
            os.environ["IMAGEIO_FFMPEG_EXE"] = exe
            return True
    except Exception:
        tried.append("imageio-ffmpeg not importable in this environment")

    # 3. A real ffmpeg inside the active env (env bin is often not on PATH).
    for cand in _env_ffmpeg_candidates():
        if os.path.exists(cand):
            os.environ["IMAGEIO_FFMPEG_EXE"] = cand
            return True

    # 4. System ffmpeg on the current PATH.
    w = shutil.which("ffmpeg")
    if w and os.path.exists(w):
        os.environ["IMAGEIO_FFMPEG_EXE"] = w
        return True

    # 5. A system ffmpeg that is installed but off our (possibly stripped) PATH.
    #    A GUI-launched process (desktop icon / Finder / .desktop) inherits a
    #    minimal PATH. Rather than hardcode install directories -- which differ
    #    by OS, distro and package manager and would not replicate elsewhere --
    #    resolve portably: (a) the OS's own default binary path, and (b) whatever
    #    the user's login shell resolves, without us ever naming a directory.
    defpath = os.pathsep.join(p for p in os.defpath.split(os.pathsep) if p)
    for cand in (shutil.which("ffmpeg", path=defpath) if defpath else None,
                 _which_via_login_shell("ffmpeg")):
        if cand and os.path.exists(cand):
            os.environ["IMAGEIO_FFMPEG_EXE"] = cand
            return True

    _FFMPEG_REASON = ("; ".join(tried)
                      or "no ffmpeg found on PATH, in the environment, or via the login shell")
    logger.info("ffmpeg unavailable: %s", _FFMPEG_REASON)
    return False


# --------------------------------------------------------------------------- #
# Settings dialog
# --------------------------------------------------------------------------- #
class TurntableDialog(QDialog):
    """Collects turntable settings and an output path for a given viewer."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("Record Movie")
        self.setMinimumWidth(440)
        self.settings = TurntableSettings.load()
        self._mp4_ok = _ensure_ffmpeg()
        self._z_axis, self._z_count = _z_axis_and_count(viewer)
        self._build_ui()
        self._sync_from_settings()
        self._on_mode_changed()
        self._update_estimate()

    # -- UI construction ----------------------------------------------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)

        # --- Kind of movie ------------------------------------------------- #
        kind_row = QHBoxLayout()
        kind_row.addWidget(QLabel("Movie:"))
        self.combo_mode = QComboBox()
        for _key, label in _MODES:
            self.combo_mode.addItem(label)
        kind_row.addWidget(self.combo_mode, 1)
        root.addLayout(kind_row)

        # --- Motion: rotation ------------------------------------------------ #
        motion = QGroupBox("Rotation")
        self.group_rotate = motion
        form = QFormLayout(motion)

        self.spin_speed = QDoubleSpinBox()
        self.spin_speed.setRange(1.0, 3600.0)
        self.spin_speed.setSuffix(" °/s")
        self.spin_speed.setDecimals(0)
        form.addRow("Rotation speed:", self.spin_speed)

        self.combo_dir = QComboBox()
        self.combo_dir.addItems(["Clockwise", "Counter-clockwise"])
        form.addRow("Direction:", self.combo_dir)

        self.combo_axis = QComboBox()
        for label, _ in _AXIS_CHOICES:
            self.combo_axis.addItem(label)
        self.combo_axis.setToolTip("If the spin looks wrong, try a different axis.")
        form.addRow("Axis:", self.combo_axis)

        self.spin_turns = QDoubleSpinBox()
        self.spin_turns.setRange(0.1, 100.0)
        self.spin_turns.setSingleStep(0.5)
        self.spin_turns.setSuffix(" turn(s)")
        form.addRow("Revolutions:", self.spin_turns)

        root.addWidget(motion)

        # --- Motion: Z fly-through ---------------------------------------- #
        zbox = QGroupBox("Z fly-through")
        self.group_z = zbox
        zform = QFormLayout(zbox)
        last = max(0, self._z_count - 1)
        self.spin_z_start = QSpinBox()
        self.spin_z_end = QSpinBox()
        for w in (self.spin_z_start, self.spin_z_end):
            w.setRange(0, last)
        zrange = QHBoxLayout()
        zrange.addWidget(self.spin_z_start)
        zrange.addWidget(QLabel("to"))
        zrange.addWidget(self.spin_z_end)
        zrange.addWidget(QLabel(f"(of {self._z_count})"), 1)
        zform.addRow("Slices:", zrange)

        self.spin_z_speed = QDoubleSpinBox()
        self.spin_z_speed.setRange(0.5, 500.0)
        self.spin_z_speed.setDecimals(1)
        self.spin_z_speed.setSuffix(" slices/s")
        zform.addRow("Speed:", self.spin_z_speed)

        self.combo_z_dir = QComboBox()
        for _key, label in _Z_DIRECTIONS:
            self.combo_z_dir.addItem(label)
        zform.addRow("Direction:", self.combo_z_dir)

        znote = QLabel("Shown as 2D slices at the current zoom and position, "
                       "so a zoomed-in region can be flown through.")
        znote.setWordWrap(True)
        znote.setStyleSheet("color: gray;")
        zform.addRow(znote)
        root.addWidget(zbox)

        # --- Output -------------------------------------------------------- #
        out = QGroupBox("Output")
        oform = QFormLayout(out)

        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(1, 120)
        self.spin_fps.setSuffix(" fps")
        oform.addRow("Frame rate:", self.spin_fps)

        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems([f.upper() for f in _FORMATS])
        oform.addRow("Format:", self.combo_fmt)

        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(0.25, 4.0)
        self.spin_scale.setSingleStep(0.25)
        self.spin_scale.setSuffix(" ×")
        self.spin_scale.setToolTip("Resolution multiplier applied to the canvas screenshot.")
        oform.addRow("Resolution:", self.spin_scale)

        self.chk_canvas_only = QCheckBox("Canvas only (exclude napari toolbars/panels)")
        oform.addRow("", self.chk_canvas_only)

        path_row = QHBoxLayout()
        self.edit_path = QLineEdit()
        self.edit_path.setPlaceholderText("Choose where to save the movie…")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse)
        path_row.addWidget(self.edit_path)
        path_row.addWidget(btn_browse)
        oform.addRow("Save to:", path_row)

        root.addWidget(out)

        # --- Layers -------------------------------------------------------- #
        layers = QGroupBox("Layers")
        lform = QVBoxLayout(layers)
        self.radio_visible = QRadioButton("Use currently visible layers")
        self.radio_custom = QRadioButton("Choose layers…")
        grp = QButtonGroup(self)
        grp.addButton(self.radio_visible)
        grp.addButton(self.radio_custom)
        lform.addWidget(self.radio_visible)
        lform.addWidget(self.radio_custom)

        self.list_layers = QListWidget()
        self.list_layers.setMaximumHeight(120)
        for layer in self.viewer.layers:
            item = QListWidgetItem(layer.name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if getattr(layer, "visible", False) else Qt.Unchecked)
            self.list_layers.addItem(item)
        lform.addWidget(self.list_layers)
        self.radio_visible.toggled.connect(
            lambda on: self.list_layers.setEnabled(not on)
        )
        root.addWidget(layers)

        # --- Estimate + buttons ------------------------------------------- #
        self.lbl_estimate = QLabel("")
        self.lbl_estimate.setStyleSheet("color: gray;")
        root.addWidget(self.lbl_estimate)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_ok = QPushButton("🎥 Record")
        self.btn_ok.setDefault(True)
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_ok.clicked.connect(self._on_accept)
        btns.addWidget(self.btn_cancel)
        btns.addWidget(self.btn_ok)
        root.addLayout(btns)

        # Live estimate updates
        for w in (self.spin_speed, self.spin_turns, self.spin_fps,
                  self.spin_z_start, self.spin_z_end, self.spin_z_speed):
            w.valueChanged.connect(self._update_estimate)
        self.combo_z_dir.currentIndexChanged.connect(self._update_estimate)
        self.combo_fmt.currentIndexChanged.connect(self._on_fmt_changed)
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)

    # -- state <-> widgets --------------------------------------------------- #
    def _sync_from_settings(self):
        s = self.settings
        self.spin_speed.setValue(s.speed_dps)
        self.combo_dir.setCurrentIndex(0 if s.clockwise else 1)
        self.combo_axis.setCurrentIndex(max(0, min(2, s.axis_index)))
        self.spin_turns.setValue(s.turns)
        self.spin_fps.setValue(s.fps)
        self.combo_fmt.setCurrentIndex(_FORMATS.index(s.fmt) if s.fmt in _FORMATS else 0)
        self.spin_scale.setValue(s.scale)
        self.chk_canvas_only.setChecked(s.canvas_only)
        self.radio_visible.setChecked(s.use_visible_layers)
        self.radio_custom.setChecked(not s.use_visible_layers)
        self.list_layers.setEnabled(not s.use_visible_layers)
        self.combo_mode.setCurrentIndex(
            [k for k, _ in _MODES].index(s.mode) if s.mode in dict(_MODES) else 0)
        self.spin_z_start.setValue(0)
        self.spin_z_end.setValue(max(0, self._z_count - 1))
        self.spin_z_speed.setValue(s.z_speed)
        self.combo_z_dir.setCurrentIndex(
            [k for k, _ in _Z_DIRECTIONS].index(s.z_direction)
            if s.z_direction in dict(_Z_DIRECTIONS) else 0)
        self._suggest_path()

    def _collect(self) -> TurntableSettings:
        s = self.settings
        s.speed_dps = float(self.spin_speed.value())
        s.clockwise = self.combo_dir.currentIndex() == 0
        s.axis_index = _AXIS_CHOICES[self.combo_axis.currentIndex()][1]
        s.turns = float(self.spin_turns.value())
        s.fps = int(self.spin_fps.value())
        s.fmt = _FORMATS[self.combo_fmt.currentIndex()]
        s.scale = float(self.spin_scale.value())
        s.canvas_only = self.chk_canvas_only.isChecked()
        s.use_visible_layers = self.radio_visible.isChecked()
        s.custom_layer_names = [
            self.list_layers.item(i).text()
            for i in range(self.list_layers.count())
            if self.list_layers.item(i).checkState() == Qt.Checked
        ]
        s.mode = _MODES[self.combo_mode.currentIndex()][0]
        s.z_start = int(self.spin_z_start.value())
        s.z_end = int(self.spin_z_end.value())
        s.z_speed = float(self.spin_z_speed.value())
        s.z_direction = _Z_DIRECTIONS[self.combo_z_dir.currentIndex()][0]
        return s

    def _mode(self) -> str:
        return _MODES[self.combo_mode.currentIndex()][0]

    def _on_mode_changed(self, *_):
        z = self._mode() == MODE_ZSWEEP
        self.group_rotate.setVisible(not z)
        self.group_z.setVisible(z)
        # Keep the suggested file name in step with the kind of movie, unless
        # the user has typed their own.
        cur = self.edit_path.text().strip()
        if not cur or "_turntable_" in cur or "_zsweep_" in cur:
            self._suggest_path()
        self._update_estimate()
        self.adjustSize()

    # -- helpers ------------------------------------------------------------- #
    def _default_filename(self) -> str:
        title = getattr(self.viewer, "title", "") or "view"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title).strip("_")
        stamp = time.strftime("%Y%m%d_%H%M%S")
        kind = "zsweep" if self._mode() == MODE_ZSWEEP else "turntable"
        return f"{safe or 'view'}_{kind}_{stamp}.{_FORMATS[self.combo_fmt.currentIndex()]}"

    def _suggest_path(self):
        base_dir = self.settings.last_dir or os.path.expanduser("~")
        self.edit_path.setText(os.path.join(base_dir, self._default_filename()))

    def _on_fmt_changed(self):
        # keep the file extension in sync with the chosen format
        cur = self.edit_path.text().strip()
        ext = _FORMATS[self.combo_fmt.currentIndex()]
        if cur:
            root, _ = os.path.splitext(cur)
            self.edit_path.setText(root + "." + ext)
        self._update_estimate()

    def _browse(self):
        ext = _FORMATS[self.combo_fmt.currentIndex()]
        start = self.edit_path.text().strip() or os.path.join(
            self.settings.last_dir or os.path.expanduser("~"), self._default_filename()
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "Save movie", start, f"{ext.upper()} (*.{ext})"
        )
        if path:
            if not path.lower().endswith("." + ext):
                path += "." + ext
            self.edit_path.setText(path)

    def _update_estimate(self):
        try:
            fps = int(self.spin_fps.value())
            if self._mode() == MODE_ZSWEEP:
                frames = len(zsweep_plan(
                    self.spin_z_start.value(), self.spin_z_end.value(),
                    self.spin_z_speed.value(), fps,
                    _Z_DIRECTIONS[self.combo_z_dir.currentIndex()][0]))
                duration = frames / max(1, fps)
            else:
                tmp = TurntableSettings(
                    speed_dps=float(self.spin_speed.value()),
                    turns=float(self.spin_turns.value()),
                    fps=fps,
                )
                frames, duration = tmp.total_frames, tmp.duration_s
            mp4_note = "" if self._mp4_ok else "  (ffmpeg not found — MP4 disabled, use GIF)"
            self.lbl_estimate.setText(
                f"≈ {frames} frames · {duration:.1f}s at {fps} fps{mp4_note}"
            )
        except Exception:
            self.lbl_estimate.setText("")

    def _on_accept(self):
        path = self.edit_path.text().strip()
        if not path:
            QMessageBox.warning(self, "No output path", "Please choose where to save the movie.")
            return
        if self.combo_fmt.currentIndex() == _FORMATS.index("mp4") and not self._mp4_ok:
            QMessageBox.warning(
                self, "MP4 unavailable",
                "No ffmpeg binary could be found, so MP4 can't be written.\n\n"
                f"Reason: {_FFMPEG_REASON or 'unknown'}\n\n"
                "Install ffmpeg so it's on your PATH (via your system or conda "
                "package manager), or install the 'imageio-ffmpeg' Python package "
                "into this environment, then reopen this dialog.\n\n"
                "Or choose GIF, which needs no ffmpeg.",
            )
            return
        parent = os.path.dirname(path) or "."
        if not os.path.isdir(parent):
            QMessageBox.warning(self, "Bad folder", f"Folder does not exist:\n{parent}")
            return
        self.settings = self._collect()
        self.settings.last_dir = parent
        self.settings.save()
        self.output_path = path
        self.accept()


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _max_layer_ndim(viewer) -> int:
    return max((getattr(l, "ndim", 0) for l in viewer.layers), default=0)


def _z_axis_and_count(viewer):
    """(axis, number of slices) the fly-through steps along. See `_z_reference`."""
    axis, count, _origin, _step = _z_reference(viewer)
    return axis, count


def _z_reference(viewer):
    """(axis, slices, world origin, world step) of the stack to fly through.

    The axis is the one the Z slider moves in 2D view: the first non-displayed
    axis in the 2D arrangement (computed for 2D whatever the current display,
    since a 3D view displays that axis rather than slicing it).

    Slices are counted in the IMAGE, not in napari's slider steps. The slider
    step is the finest scale of any layer, so one layer with a different Z
    scale -- an unscaled annotation, say -- makes the slider walk many steps
    per real slice, and a fly-through driven by it shows the same slice over
    and over. Positions are therefore computed from the largest image layer's
    own scale and offset, which is exact whatever else is loaded. napari's dims
    are the fallback when no image layer spans the axis.
    """
    try:
        dims = viewer.dims
        order = list(dims.order)
        candidates = order[:len(order) - 2]
        axis = next((a for a in candidates if int(dims.nsteps[a]) > 1),
                    candidates[-1] if candidates else 0)
    except Exception:
        return 0, 1, 0.0, 1.0

    best = None
    ndim = len(viewer.dims.nsteps)
    for layer in viewer.layers:
        try:
            layer_axis = axis - (ndim - int(layer.ndim))
            if layer_axis < 0:
                continue
            data = layer.data
            shape = getattr(data, "shape", None)
            if shape is None and isinstance(data, (list, tuple)) and data:
                shape = data[0].shape        # multiscale: full resolution
            n = int(shape[layer_axis])
            if n <= 1:
                continue
            is_image = type(layer).__name__ == "Image"
            key = (is_image, n)
            if best is None or key > best[0]:
                best = (key, n, float(layer.translate[layer_axis]),
                        float(layer.scale[layer_axis]))
        except Exception:
            continue
    if best is not None:
        _key, n, origin, step = best
        return axis, n, origin, step
    try:
        lo, _hi, step = viewer.dims.range[axis]
        return axis, int(viewer.dims.nsteps[axis]), float(lo), float(step)
    except Exception:
        return axis, 1, 0.0, 1.0


def render_movie(viewer, settings: TurntableSettings, out_path: str,
                 parent: Optional[QWidget] = None) -> bool:
    """Render whichever kind of movie `settings.mode` selects."""
    if settings.mode == MODE_ZSWEEP:
        return render_zsweep(viewer, settings, out_path, parent=parent)
    return render_turntable(viewer, settings, out_path, parent=parent)


def render_zsweep(viewer, settings: TurntableSettings, out_path: str,
                  parent: Optional[QWidget] = None) -> bool:
    """Fly through the Z slices in 2D and save the movie. True on success.

    The current zoom and pan are kept. Display mode, the current slice and
    layer visibility are restored afterwards, whatever happens.
    """
    if _max_layer_ndim(viewer) < 3:
        QMessageBox.warning(parent, "Need 3D data",
                            "A Z fly-through needs a 3D (or higher) dataset in the viewer.")
        return False

    axis, count, origin, step = _z_reference(viewer)
    if count <= 1:
        QMessageBox.warning(parent, "Only one slice",
                            "This image has a single Z slice, so there is nothing to fly through.")
        return False
    last = count - 1
    start = max(0, min(last, int(settings.z_start)))
    end = last if int(settings.z_end) < 0 else max(0, min(last, int(settings.z_end)))
    plan = zsweep_plan(start, end, settings.z_speed, settings.fps, settings.z_direction)

    prior_ndisplay = viewer.dims.ndisplay
    prior_point = tuple(viewer.dims.point)
    prior_vis = _apply_layer_selection(viewer, settings)
    viewer.dims.ndisplay = 2

    progress = QProgressDialog("Rendering Z fly-through\u2026", "Cancel", 0,
                               len(plan), parent)
    progress.setWindowTitle("Recording Z Fly-through")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    frames = []
    cancelled = False
    try:
        shown = None
        shot = None
        for i, z in enumerate(plan):
            if progress.wasCanceled():
                cancelled = True
                break
            if z != shown:
                # The slice's own world position (see _z_reference), not a
                # slider step.
                viewer.dims.set_point(axis, origin + z * step)
                QApplication.processEvents()  # let the canvas redraw the slice
                # Same note as render_turntable: no `scale` argument here.
                shot = viewer.screenshot(canvas_only=settings.canvas_only,
                                         flash=False)
                shot = _resize_rgb(np.asarray(shot)[..., :3], settings.scale)
                shown = z
            # A held slice reuses its frame rather than re-rendering it.
            frames.append(shot)
            progress.setValue(i + 1)
    finally:
        try:
            viewer.dims.ndisplay = prior_ndisplay
            viewer.dims.set_point(range(len(prior_point)), prior_point)
            for layer, vis in prior_vis.items():
                try:
                    layer.visible = vis
                except Exception:
                    pass
        except Exception:
            pass
        progress.close()

    if cancelled or not frames:
        return False
    return _save_movie(frames, settings, out_path, parent, "Fly-through saved")


def _save_movie(frames, settings: TurntableSettings, out_path: str,
                parent, title: str) -> bool:
    """Write `frames` in the chosen format and report it. True on success."""
    try:
        if settings.fmt == "gif":
            _write_gif(out_path, frames, settings.fps)
        else:
            _write_mp4(out_path, frames, settings.fps)
    except Exception as exc:
        logger.exception("Movie export failed")
        QMessageBox.critical(parent, "Export failed", f"Could not write the movie:\n{exc}")
        return False
    QMessageBox.information(parent, title,
                            f"Saved {len(frames)} frames to:\n{out_path}")
    return True


def _apply_layer_selection(viewer, settings: TurntableSettings):
    """Return a dict of {layer: prior_visibility} so it can be restored, after
    setting visibility to the chosen selection. In 'visible' mode nothing is
    changed and an empty dict is returned."""
    if settings.use_visible_layers:
        return {}
    prior = {}
    wanted = set(settings.custom_layer_names)
    for layer in viewer.layers:
        prior[layer] = getattr(layer, "visible", True)
        try:
            layer.visible = layer.name in wanted
        except Exception:
            pass
    return prior


def render_turntable(viewer, settings: TurntableSettings, out_path: str,
                     parent: Optional[QWidget] = None) -> bool:
    """Render the turntable to ``out_path``. Returns True on success."""
    if _max_layer_ndim(viewer) < 3:
        QMessageBox.warning(parent, "Need 3D data",
                            "A rotation needs a 3D (or higher) dataset in the viewer.")
        return False

    # Force a 3D display; remember prior state to restore afterwards.
    prior_ndisplay = viewer.dims.ndisplay
    prior_angles = tuple(viewer.camera.angles)
    prior_vis = _apply_layer_selection(viewer, settings)
    viewer.dims.ndisplay = 3

    n_frames = settings.total_frames
    sign = 1.0 if settings.clockwise else -1.0
    delta = sign * settings.total_degrees / n_frames  # per-frame step (seamless loop)
    axis = settings.axis_index

    progress = QProgressDialog("Rendering rotation…", "Cancel", 0, n_frames, parent)
    progress.setWindowTitle("Recording 3D Rotation")
    progress.setWindowModality(Qt.WindowModal)
    progress.setMinimumDuration(0)
    progress.setValue(0)

    frames = []
    cancelled = False
    try:
        base = list(prior_angles)
        for i in range(n_frames):
            if progress.wasCanceled():
                cancelled = True
                break
            angles = list(base)
            angles[axis] = base[axis] + delta * i
            viewer.camera.angles = tuple(angles)
            QApplication.processEvents()  # let the canvas redraw at the new angle
            # NB: do NOT pass `scale` to napari.screenshot. Internally it does an
            # in-place `size *= scale` on an int array, which raises under
            # numpy>=2 for any float scale (even 1.0). We capture at native size
            # and apply the resolution multiplier ourselves below.
            shot = viewer.screenshot(canvas_only=settings.canvas_only, flash=False)
            frame = np.asarray(shot)[..., :3]  # drop alpha
            frames.append(_resize_rgb(frame, settings.scale))
            progress.setValue(i + 1)
    finally:
        # Always restore the viewer to how the user left it.
        try:
            viewer.camera.angles = prior_angles
            viewer.dims.ndisplay = prior_ndisplay
            for layer, vis in prior_vis.items():
                try:
                    layer.visible = vis
                except Exception:
                    pass
        except Exception:
            pass
        progress.close()

    if cancelled or not frames:
        return False
    return _save_movie(frames, settings, out_path, parent, "Rotation saved")


def _resize_rgb(arr: np.ndarray, scale: float) -> np.ndarray:
    """Scale an RGB frame by `scale`. Done here rather than via napari's
    screenshot `scale` argument, which multiplies an int size array by a float
    in place and raises under numpy>=2."""
    if not scale or abs(scale - 1.0) < 1e-6:
        return arr
    from PIL import Image
    h, w = arr.shape[:2]
    new_w = max(2, int(round(w * scale)))
    new_h = max(2, int(round(h * scale)))
    return np.asarray(Image.fromarray(arr).resize((new_w, new_h), Image.LANCZOS))


def _even(a: np.ndarray) -> np.ndarray:
    """Crop to even height/width (required by yuv420p / libx264)."""
    h, w = a.shape[:2]
    return a[: h - (h % 2), : w - (w % 2)]


def _write_mp4(out_path: str, frames: List[np.ndarray], fps: int) -> None:
    """Encode frames to H.264 MP4 by piping raw RGB to the ffmpeg binary.

    Uses the executable located by _ensure_ffmpeg (system, env, or the copy
    bundled with imageio-ffmpeg). This needs only an ffmpeg *binary* -- not the
    imageio or imageio-ffmpeg Python packages, which may be missing in some
    environments even when a system ffmpeg is present.
    """
    if not _ensure_ffmpeg():
        raise RuntimeError(_FFMPEG_REASON or "no ffmpeg binary available")
    exe = os.environ["IMAGEIO_FFMPEG_EXE"]

    # Uniform, contiguous, even-sized RGB uint8 frames (yuv420p needs even dims).
    prepared = [np.ascontiguousarray(_even(f)[..., :3].astype(np.uint8)) for f in frames]
    h, w = prepared[0].shape[:2]

    def _run(codec: str):
        cmd = [exe, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
               "-an", "-c:v", codec, "-pix_fmt", "yuv420p"]
        if codec == "libx264":
            cmd += ["-crf", "18", "-preset", "medium"]
        cmd.append(out_path)
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, creationflags=_CREATE_NO_WINDOW,
        )
        try:
            for fr in prepared:
                proc.stdin.write(fr.tobytes())
        except (BrokenPipeError, OSError):
            pass  # ffmpeg exited early; the error is captured from stderr below
        try:
            proc.stdin.close()
        except OSError:
            pass
        err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
        proc.wait()
        return proc.returncode, err

    # Prefer H.264; fall back to a near-universal codec if this ffmpeg build
    # lacks libx264 (some minimal system builds do).
    rc, err = _run("libx264")
    if rc != 0 and ("x264" in err.lower() or "encoder" in err.lower()):
        rc, err = _run("mpeg4")
    if rc != 0:
        raise RuntimeError(f"ffmpeg failed (code {rc}): {err.strip()[:500] or 'no error output'}")


def _write_gif(out_path: str, frames: List[np.ndarray], fps: int) -> None:
    # Pillow needs no ffmpeg, so GIF always works as a fallback.
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        out_path, save_all=True, append_images=imgs[1:],
        duration=int(round(1000.0 / max(1, fps))), loop=0, disposal=2,
    )


# --------------------------------------------------------------------------- #
# Viewer integration
# --------------------------------------------------------------------------- #
def _locate_layer_list_dock(viewer):
    """Find napari's layer-list dock across versions so the button can sit
    directly beneath it. Kept self-contained (no import from the GUI package)
    so this module has no upward dependency on it.

    A local choice, not a project rule. The convention this used to cite -- that
    the pipeline package never imports the GUI package -- was never true:
    `fluorescence_strategy` has always imported `high_level_gui.
    processing_strategies`, because that module is the seam between the pipeline
    and the GUI. The wording has been removed. Keeping THIS module self-contained
    is still worth it: it is a display helper with no other reason to know the
    GUI exists, and a version-tolerant dock lookup is easier to reason about
    where the versions are handled."""
    for accessor in (lambda: viewer.window._qt_viewer.dockLayerList,
                     lambda: viewer.window.qt_viewer.dockLayerList):
        try:
            d = accessor()
            if d is not None:
                return d
        except Exception:
            pass
    try:
        from PyQt5.QtWidgets import QDockWidget
        for d in viewer.window._qt_window.findChildren(QDockWidget):
            if "layer list" in d.windowTitle().lower():
                return d
    except Exception:
        pass
    return None


def make_turntable_widget(viewer) -> QWidget:
    """The '🎥 Record Movie' button in its container, not docked.

    For callers that place it themselves, like the segmentation viewer's
    scrollable side panel. `add_turntable_button` docks the same widget.
    """
    btn = QPushButton("🎥 Record Movie")
    btn.setToolTip("Save an MP4 or GIF of the view: rotate it in 3D, or fly "
                   "through the Z slices.")

    def _open_dialog():
        if _max_layer_ndim(viewer) < 3:
            QMessageBox.information(
                viewer.window._qt_window if hasattr(viewer.window, "_qt_window") else None,
                "Need 3D data",
                "Load a 3D dataset to record a movie.",
            )
            return
        parent = None
        try:
            parent = viewer.window._qt_window
        except Exception:
            pass
        dlg = TurntableDialog(viewer, parent=parent)
        if dlg.exec_() == QDialog.Accepted:
            render_movie(viewer, dlg.settings, dlg.output_path, parent=parent)

    btn.clicked.connect(_open_dialog)

    container = QWidget()
    lay = QVBoxLayout(container)
    lay.setContentsMargins(5, 3, 5, 3)
    lay.addWidget(btn)
    return container


def add_turntable_button(viewer):
    """Dock a '🎥 Record Movie' button beneath the layer list. Mirrors the
    placement of add_channel_visibility_toggle so the two controls sit together.
    Returns the QDockWidget (or None on failure)."""
    container = make_turntable_widget(viewer)
    # A section of the viewer's scrollable side panel when there is one, so
    # the multi-channel overlay gets the same single scrolling column as the
    # segmentation viewer; a separate dock otherwise, as before.
    try:
        from ..high_level_gui.app_launch import add_side_panel_section
        side = add_side_panel_section(viewer, "Movie", container, "rotation")
        if side is not None:
            return side
    except Exception:
        logger.debug("side panel unavailable; docking the movie button", exc_info=True)
    dock = viewer.window.add_dock_widget(container, area="left", name="Movie")

    # Sit it directly beneath the layer list.
    try:
        ll = _locate_layer_list_dock(viewer)
        if ll is not None:
            viewer.window._qt_window.splitDockWidget(ll, dock, Qt.Vertical)
    except Exception:
        pass
    return dock