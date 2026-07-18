"""
batch_progress_dialog: live progress UI for a batch run executed in a child process.

Shows an always-animating busy spinner (so the window is visibly alive even
during a long, event-less native stage), an outer "image X of N" bar, an inner
"stage Y of M" bar, and an expandable console pane fed by the child's stdout.
Cancel terminates the worker process immediately (see batch_runner.py for why a
separate process rather than a thread).
"""

from __future__ import annotations

import multiprocessing as mp
import queue as _pyqueue
from typing import Dict, List

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QPlainTextEdit,
    QPushButton, QSizePolicy,
)

from .batch_runner import run_batch_process
try:
    from .project_selection import prettify_step_name
except Exception:  # pragma: no cover - fallback if import shape changes
    def prettify_step_name(method: str) -> str:
        return method

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class _QueueReader(QThread):
    """Reads the child's IPC queue and re-emits messages as Qt signals on the
    GUI thread. Also detects an unexpected child exit."""

    progress = pyqtSignal(object)
    logline = pyqtSignal(str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, queue, proc):
        super().__init__()
        self._q = queue
        self._proc = proc
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _dispatch(self, kind, payload) -> None:
        if kind == "log":
            self.logline.emit(payload)
        elif kind == "progress":
            self.progress.emit(payload)
        elif kind == "done":
            self.completed.emit(payload)
        elif kind == "error":
            self.failed.emit(payload)

    def run(self) -> None:
        while not self._stop:
            try:
                kind, payload = self._q.get(timeout=0.2)
            except _pyqueue.Empty:
                # No message: if the child has died, drain and stop.
                if self._proc is not None and not self._proc.is_alive():
                    got_terminal = False
                    try:
                        while True:
                            kind, payload = self._q.get_nowait()
                            if kind in ("done", "error"):
                                got_terminal = True
                            self._dispatch(kind, payload)
                    except _pyqueue.Empty:
                        pass
                    if not got_terminal and not self._stop:
                        self.failed.emit("The batch process exited unexpectedly.")
                    return
                continue
            except (EOFError, OSError):
                return
            self._dispatch(kind, payload)
            if kind in ("done", "error"):
                return


class BatchProgressDialog(QDialog):
    """Modal progress window for a batch run. Emits finished_batch when done."""

    #                       success, failed, skipped, cancelled
    finished_batch = pyqtSignal(int, int, int, bool)

    def __init__(self, folders: List[str], force_map: Dict[str, bool], parent=None):
        super().__init__(parent)
        self._folders = list(folders)
        self._force_map = dict(force_map)
        self._ctx = None
        self._queue = None
        self._proc = None
        self._reader = None
        self._cancelled = False
        self._finished = False
        self._counts = (0, 0, 0)
        self._spin_idx = 0

        self.setWindowTitle("Batch Processing")
        self.setModal(True)
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumWidth(560)
        self._build()

        self._spin_timer = QTimer(self)
        self._spin_timer.timeout.connect(self._tick_spinner)

    # ---- construction --------------------------------------------------- #
    def _build(self) -> None:
        root = QVBoxLayout(self)

        header = QHBoxLayout()
        self.spinner = QLabel(_SPINNER_FRAMES[0])
        self.spinner.setFixedWidth(20)
        self.title = QLabel("Preparing…")
        self.title.setStyleSheet("font-weight: bold;")
        header.addWidget(self.spinner)
        header.addWidget(self.title)
        header.addStretch(1)
        root.addLayout(header)

        self.folder_label = QLabel("Image – of –")
        root.addWidget(self.folder_label)
        self.folder_bar = QProgressBar()
        self.folder_bar.setRange(0, max(1, len(self._folders)))
        self.folder_bar.setValue(0)
        root.addWidget(self.folder_bar)

        self.step_label = QLabel("Stage –")
        root.addWidget(self.step_label)
        self.step_bar = QProgressBar()
        self.step_bar.setRange(0, 1)
        self.step_bar.setValue(0)
        root.addWidget(self.step_bar)

        # Console toggle + pane
        self.console_toggle = QPushButton("Hide console ▾")
        self.console_toggle.setCheckable(True)
        self.console_toggle.setChecked(True)
        self.console_toggle.clicked.connect(self._toggle_console)
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(self.console_toggle)
        toggle_row.addStretch(1)
        root.addLayout(toggle_row)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(5000)  # cap memory on very chatty runs
        self.console.setStyleSheet(
            "font-family: Menlo, Consolas, monospace; font-size: 11px;"
        )
        self.console.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.console.setMinimumHeight(180)
        root.addWidget(self.console, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        buttons.addWidget(self.cancel_btn)
        self.close_btn = QPushButton("Close")
        self.close_btn.setEnabled(False)
        self.close_btn.clicked.connect(self.accept)
        buttons.addWidget(self.close_btn)
        root.addLayout(buttons)

    # ---- lifecycle ------------------------------------------------------ #
    def start(self) -> None:
        """Spawn the worker process and begin streaming progress."""
        try:
            self._ctx = mp.get_context("spawn")
            self._queue = self._ctx.Queue()
            self._proc = self._ctx.Process(
                target=run_batch_process,
                args=(self._folders, self._force_map, self._queue),
                daemon=True,
            )
            self._proc.start()
        except Exception as exc:
            self._append(f"[Fatal] Could not start batch process: {exc}\n")
            self._finish(cancelled=False)
            return

        self._reader = _QueueReader(self._queue, self._proc)
        self._reader.progress.connect(self._on_progress)
        self._reader.logline.connect(self._append)
        self._reader.completed.connect(self._on_completed)
        self._reader.failed.connect(self._on_failed)
        self._reader.start()

        self.title.setText("Processing…")
        self._spin_timer.start(100)

    def _tick_spinner(self) -> None:
        self._spin_idx = (self._spin_idx + 1) % len(_SPINNER_FRAMES)
        self.spinner.setText(_SPINNER_FRAMES[self._spin_idx])

    # ---- signal slots --------------------------------------------------- #
    def _on_progress(self, event: dict) -> None:
        kind = event.get("kind")
        if kind == "folder":
            total = int(event.get("total_folders", 1) or 1)
            idx = int(event.get("folder_idx", 0))
            name = event.get("folder_name", "")
            self.folder_bar.setRange(0, total)
            self.folder_bar.setValue(idx)
            self.folder_label.setText(f"Image {idx + 1} of {total}: {name}")
            self.step_bar.setRange(0, 1)
            self.step_bar.setValue(0)
            self.step_label.setText("Stage –")
        elif kind == "step":
            total = int(event.get("total_steps", 1) or 1)
            idx = int(event.get("step_idx", 1))
            name = prettify_step_name(event.get("step_name", ""))
            self.step_bar.setRange(0, total)
            self.step_bar.setValue(max(0, idx - 1))
            self.step_label.setText(f"Stage {idx} of {total}: {name}")

    def _append(self, text: str) -> None:
        # Preserve the child's own newlines (print already includes them).
        self.console.moveCursor(self.console.textCursor().End)
        self.console.insertPlainText(text)
        self.console.ensureCursorVisible()

    def _on_completed(self, counts: dict) -> None:
        self._counts = (
            int(counts.get("success", 0)),
            int(counts.get("failed", 0)),
            int(counts.get("skipped", 0)),
        )
        self._finish(cancelled=False)

    def _on_failed(self, message: str) -> None:
        self._append(f"\n[Error] {message}\n")
        self._finish(cancelled=False)

    # ---- cancel / finish ------------------------------------------------ #
    def _cancel(self) -> None:
        if self._finished:
            return
        self._cancelled = True
        self.title.setText("Cancelling…")
        self.cancel_btn.setEnabled(False)
        self._kill_process()
        self._append("\n[Cancelled] Batch stopped by user.\n")
        self._finish(cancelled=True)

    def _kill_process(self) -> None:
        if self._proc is not None:
            try:
                if self._proc.is_alive():
                    self._proc.terminate()
                self._proc.join(2)
            except Exception:
                pass

    def _finish(self, cancelled: bool) -> None:
        if self._finished:
            return
        self._finished = True
        self._spin_timer.stop()
        self.spinner.setText("•")

        if self._reader is not None:
            self._reader.stop()
            self._reader.wait(1500)
            self._reader = None
        self._kill_process()

        s, f, k = self._counts
        if cancelled:
            self.title.setText("Cancelled")
        elif f > 0:
            self.title.setText(f"Finished with errors — {s} ok, {f} failed, {k} skipped")
            self.folder_bar.setValue(self.folder_bar.maximum())
        else:
            self.title.setText(f"Complete — {s} processed, {k} skipped")
            self.folder_bar.setValue(self.folder_bar.maximum())
            self.step_bar.setValue(self.step_bar.maximum())

        self.cancel_btn.setEnabled(False)
        self.close_btn.setEnabled(True)
        self.close_btn.setDefault(True)
        self.finished_batch.emit(s, f, k, cancelled)

    # ---- misc ----------------------------------------------------------- #
    def _toggle_console(self) -> None:
        show = self.console_toggle.isChecked()
        self.console.setVisible(show)
        self.console_toggle.setText("Hide console ▾" if show else "Show console ▸")

    def closeEvent(self, event) -> None:  # noqa: N802
        # Closing the window mid-run cancels the batch (never leave an orphan).
        if not self._finished:
            self._cancelled = True
            self._spin_timer.stop()
            if self._reader is not None:
                self._reader.stop()
                self._reader.wait(1500)
            self._kill_process()
            self._finished = True
            self.finished_batch.emit(0, 0, 0, True)
        super().closeEvent(event)
