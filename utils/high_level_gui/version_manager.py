"""
version_manager.py -- unobtrusive in-app version control for HIBACHI.

Adds a small, muted "v<hash>" indicator to the main window's status bar. Clicking
it opens a dialog to check for updates and switch to a previous version.

Design notes:
* Reuses the launcher's `updater` module (stdlib-only: git via subprocess), so
  there is no duplicated logic and no heavy imports.
* A running Python process cannot hot-swap its own code, and dependency updates
  are the launcher's job. So this dialog never installs in place: it switches
  the checkout (fast, local) or reports an available update, and asks the user
  to restart -- the launcher applies everything cleanly on the next start.
* Everything is best-effort and guarded: if this isn't a git checkout, or the
  launcher/updater can't be found, the indicator simply doesn't appear.
"""

from __future__ import annotations

import os
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

# repo layout: <repo>/utils/high_level_gui/version_manager.py
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_LAUNCHER = os.path.join(_REPO_ROOT, "launcher")


def _load_updater():
    """Import the launcher's updater module, or return None if unavailable."""
    try:
        if _LAUNCHER not in sys.path:
            sys.path.insert(0, _LAUNCHER)
        import updater  # type: ignore
        return updater
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[version] updater unavailable: {exc}")
        return None


def _describe(updater, repo_root, rev=None):
    """Return {rev, short, date, subject} for `rev` (default HEAD), or None."""
    rev = rev or updater.current_rev(repo_root)
    if not rev:
        return None
    rc, out, _ = updater._git(["log", "-1", "--format=%h%x1f%cs%x1f%s", rev], repo_root)
    if rc == 0 and out:
        parts = (out.split("\x1f") + ["", "", ""])[:3]
        return {"rev": rev, "short": parts[0], "date": parts[1], "subject": parts[2]}
    return {"rev": rev, "short": rev[:7], "date": "", "subject": ""}


class _CheckWorker(QThread):
    """Runs updater.check_for_update() off the UI thread (git fetch can block)."""
    done = pyqtSignal(object)

    def __init__(self, updater, repo_root, parent=None):
        super().__init__(parent)
        self._u = updater
        self._root = repo_root

    def run(self):  # noqa: D401
        try:
            res = self._u.check_for_update(self._root, logger=lambda m: None)
        except Exception as exc:  # pragma: no cover - defensive
            res = exc
        self.done.emit(res)


class VersionDialog(QDialog):
    def __init__(self, updater, repo_root, parent=None):
        super().__init__(parent)
        self._u = updater
        self._root = repo_root
        self._worker = None

        self.setWindowTitle("HIBACHI versions")
        self.setMinimumWidth(460)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 16)
        root.setSpacing(10)

        cur = _describe(updater, repo_root) or {"short": "unknown", "date": "", "subject": ""}
        title = QLabel(f"Current version:  <b>{cur['short']}</b>")
        root.addWidget(title)
        sub = QLabel(f"{cur['date']}  ·  {cur['subject']}".strip(" ·"))
        sub.setStyleSheet("color:#6b7280;")
        sub.setWordWrap(True)
        root.addWidget(sub)

        # --- update check row ---
        check_row = QHBoxLayout()
        self.check_btn = QPushButton("Check for updates")
        self.check_btn.clicked.connect(self._check)
        check_row.addWidget(self.check_btn)
        self.check_status = QLabel("")
        self.check_status.setStyleSheet("color:#6b7280;")
        self.check_status.setWordWrap(True)
        check_row.addWidget(self.check_status, 1)
        root.addLayout(check_row)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#e2e4e8;")
        root.addWidget(line)

        root.addWidget(QLabel("Switch to a previous version:"))
        self.list = QListWidget()
        self.list.setStyleSheet("QListWidget{font-family:monospace;}")
        self._populate_versions(cur.get("rev", ""))
        root.addWidget(self.list, 1)

        note = QLabel("Switching takes effect after you restart HIBACHI.")
        note.setStyleSheet("color:#6b7280;")
        root.addWidget(note)

        # --- footer buttons ---
        footer = QHBoxLayout()
        self.switch_btn = QPushButton("Switch to selected")
        self.switch_btn.clicked.connect(self._switch)
        footer.addWidget(self.switch_btn)
        footer.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        root.addLayout(footer)

    def _populate_versions(self, current_rev: str):
        self.list.clear()
        try:
            versions = self._u.list_versions(self._root, limit=15)
        except Exception as exc:
            versions = []
            print(f"[version] could not list versions: {exc}")
        for v in versions:
            here = v["rev"].startswith(current_rev) or (current_rev and current_rev.startswith(v["rev"]))
            marker = "   ◀ current" if here else ""
            item = QListWidgetItem(f'{v["date"]}   {v["short"]}   {v["subject"]}{marker}')
            item.setData(Qt.UserRole, v["rev"])
            self.list.addItem(item)
            if here:
                self.list.setCurrentItem(item)

    def _check(self):
        self.check_btn.setEnabled(False)
        self.check_status.setText("Checking…")
        self._worker = _CheckWorker(self._u, self._root, self)
        self._worker.done.connect(self._on_check)
        self._worker.start()

    def _on_check(self, res):
        self.check_btn.setEnabled(True)
        u = self._u
        if isinstance(res, Exception):
            self.check_status.setText(f"Check failed: {res}")
            return
        status = getattr(res, "status", None)
        if status == u.UPDATE_AVAILABLE:
            short = (res.new_rev or "")[:7]
            self.check_status.setText(f"Update {short} available — restart to install.")
            self._offer_restart(
                "Update available",
                f"Version {short} is available.\n\n"
                "HIBACHI installs updates on startup. Quit now and reopen to install it?",
            )
        elif status == u.UP_TO_DATE:
            self.check_status.setText("You're on the latest version.")
        elif status == u.OFFLINE:
            self.check_status.setText("Offline — couldn't reach the update server.")
        elif status == u.LOCAL_AHEAD:
            self.check_status.setText("Development checkout (ahead of server); no update.")
        else:
            self.check_status.setText(getattr(res, "message", "No update.") or "No update.")

    def _switch(self):
        item = self.list.currentItem()
        if item is None:
            return
        rev = item.data(Qt.UserRole)
        short = rev[:7]
        if QMessageBox.question(
            self, "Switch version",
            f"Switch HIBACHI to version {short}?\n\n"
            "Your projects and data are untouched. The change takes effect after "
            "you restart the app.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok, msg = self._u.rollback_to(self._root, rev, logger=lambda m: print(f"[version] {m}"))
        finally:
            QApplication.restoreOverrideCursor()

        if not ok:
            QMessageBox.warning(self, "Switch failed", msg)
            return

        # Don't let the launcher immediately re-prompt to update back to the tip.
        try:
            tip = self._u.remote_tip(self._root)
            if tip:
                self._u.set_skipped_rev(tip)
        except Exception:
            pass

        self._populate_versions(rev)
        self._offer_restart(
            "Switched version",
            f"{msg}\n\nQuit now and reopen to run this version?",
        )

    def _offer_restart(self, title: str, question: str):
        if QMessageBox.question(
            self, title, question, QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        ) == QMessageBox.Yes:
            QApplication.quit()


def attach_version_status(main_window) -> None:
    """
    Add a small, muted version indicator to `main_window`'s status bar. Safe
    no-op if this isn't a git checkout or the launcher/updater is unavailable.
    """
    updater = _load_updater()
    if updater is None:
        return
    repo_root = updater.find_repo_root(_LAUNCHER) or _REPO_ROOT
    if not repo_root or not os.path.isdir(os.path.join(repo_root, ".git")):
        return  # not a tracked install; nothing to manage

    info = _describe(updater, repo_root)
    label = info["short"] if info else "unknown"

    btn = QPushButton(f"v {label}")
    btn.setFlat(True)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setToolTip("Manage versions / check for updates")
    btn.setStyleSheet(
        "QPushButton{color:#8a8f98; border:none; padding:0 6px;}"
        "QPushButton:hover{color:#c75b39;}"
    )
    f = btn.font()
    f.setPointSize(max(8, f.pointSize() - 1))
    btn.setFont(f)
    btn.clicked.connect(lambda: VersionDialog(updater, repo_root, main_window).exec_())

    main_window.statusBar().addPermanentWidget(btn)
