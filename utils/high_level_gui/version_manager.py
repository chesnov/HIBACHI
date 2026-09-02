"""
version_manager.py -- unobtrusive in-app version control for HIBACHI.

Adds a small, muted "v<hash>" indicator to the main window's status bar. Clicking
it opens a dialog to check for updates, switch release channel, and pin or
unpin a specific version.

Design notes:
* Reuses the launcher's `updater` module (stdlib-only: git via subprocess), so
  there is no duplicated logic and no heavy imports.
* A running Python process cannot hot-swap its own code, and dependency updates
  are the launcher's job. So this dialog never installs in place: it switches
  the checkout (fast, local) or reports an available update, and asks the user
  to restart -- the launcher applies everything cleanly on the next start.
* That last point is why a channel switch here does NOT rebuild the conda
  environment: we are running inside the environment that would be rebuilt, and
  cannot re-exec ourselves. `updater.switch_channel` instead records
  `pending_env_update` in the launcher's state file, and the launcher applies it
  on the next start before the app is launched. Without that handshake an
  in-app switch would leave one channel's code running against the other's
  pinned numerics, silently.
* The version list is loaded from local refs only (`fetch=False`), so opening
  the dialog never blocks on the network. "Check for updates" is what fetches,
  and it does so on a worker thread.
* Everything is best-effort and guarded: if this isn't a git checkout, or the
  launcher/updater can't be found, the indicator simply doesn't appear.
"""

from __future__ import annotations

import os
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QButtonGroup, QDialog, QFrame, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QRadioButton,
    QVBoxLayout, QWidget,
)

# repo layout: <repo>/utils/high_level_gui/version_manager.py
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
_LAUNCHER = os.path.join(_REPO_ROOT, "launcher")

_CHANNEL_LABELS = {"stable": "Stable", "dev": "Development"}

# Selecting this row means "follow this channel and keep auto-updating";
# selecting a real commit means "pin to exactly this version".
_LATEST_TEXT = "Latest  --  follow this channel (auto-update)"

_DEV_WARNING = ("Development builds are unreleased and may be broken. "
                "You can switch back to Stable here at any time.")

_MUTED = "color:#6b7280;"


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
    """
    Runs the two network calls off the UI thread (git fetch can block).

    Both are done in one worker so a single button press leaves the dialog in a
    consistent state: the update verdict and the version lists come from the
    same fetch, rather than one being refreshed and the other stale.
    """
    done = pyqtSignal(object)

    def __init__(self, updater, repo_root, parent=None):
        super().__init__(parent)
        self._u = updater
        self._root = repo_root

    def run(self):  # noqa: D401
        try:
            res = self._u.check_for_update(self._root, logger=lambda m: None)
            overview = self._u.channel_overview(self._root, limit=15, fetch=True)
            self.done.emit((res, overview))
        except Exception as exc:  # pragma: no cover - defensive
            self.done.emit(exc)


class VersionDialog(QDialog):
    def __init__(self, updater, repo_root, parent=None):
        super().__init__(parent)
        self._u = updater
        self._root = repo_root
        self._worker = None
        # Local refs only: opening the dialog must not wait on the network.
        self._overview = updater.channel_overview(repo_root, limit=15, fetch=False)

        self.setWindowTitle("HIBACHI versions")
        self.setMinimumWidth(560)
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 16)
        root.setSpacing(10)

        cur = _describe(updater, repo_root) or {"short": "unknown", "date": "", "subject": ""}
        self._title = QLabel()
        root.addWidget(self._title)
        self._sub = QLabel(f"{cur['date']}  ·  {cur['subject']}".strip(" ·"))
        self._sub.setStyleSheet(_MUTED)
        self._sub.setWordWrap(True)
        root.addWidget(self._sub)
        self._refresh_title(cur)

        # --- update check row ---
        check_row = QHBoxLayout()
        self.check_btn = QPushButton("Check for updates")
        self.check_btn.clicked.connect(self._check)
        check_row.addWidget(self.check_btn)
        self.check_status = QLabel("")
        self.check_status.setStyleSheet(_MUTED)
        self.check_status.setWordWrap(True)
        check_row.addWidget(self.check_status, 1)
        root.addLayout(check_row)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color:#e2e4e8;")
        root.addWidget(line)

        # --- channel toggle ---
        chan_row = QHBoxLayout()
        chan_row.addWidget(QLabel("Channel:"))
        self._radios = {}
        self._group = QButtonGroup(self)
        for name in self._channel_order():
            rb = QRadioButton(_CHANNEL_LABELS.get(name, name.title()))
            rb.setProperty("channel", name)
            self._group.addButton(rb)
            self._radios[name] = rb
            chan_row.addWidget(rb)
            rb.toggled.connect(self._on_channel_toggled)
        chan_row.addStretch(1)
        root.addLayout(chan_row)

        self._note = QLabel("")
        self._note.setStyleSheet(_MUTED)
        self._note.setWordWrap(True)
        root.addWidget(self._note)

        self.list = QListWidget()
        self.list.setStyleSheet("QListWidget{font-family:monospace;}")
        root.addWidget(self.list, 1)

        restart_note = QLabel("Changes take effect after you restart HIBACHI.")
        restart_note.setStyleSheet(_MUTED)
        root.addWidget(restart_note)

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

        self._apply_overview()

    # ---------------------------------------------------------------- state #
    def _channel_order(self):
        chans = self._overview.get("channels") or {}
        return [c for c in ("stable", "dev") if c in chans] or sorted(chans)

    def _current_channel(self):
        chans = self._overview.get("channels") or {}
        cur = self._overview.get("current")
        return cur if cur in chans else (self._channel_order() or [None])[0]

    def _selected_channel(self):
        btn = self._group.checkedButton()
        return btn.property("channel") if btn else self._current_channel()

    def _refresh_title(self, cur):
        chan = self._overview.get("current") or "stable"
        bits = [f"Current version:  <b>{cur['short']}</b>",
                f"{_CHANNEL_LABELS.get(chan, chan)} channel"]
        if self._overview.get("pinned"):
            bits.append("<b>pinned</b>")
        self._title.setText("&nbsp;·&nbsp;".join(bits))

    def _apply_overview(self):
        """Re-sync the radios and the list to `self._overview`."""
        chans = self._overview.get("channels") or {}
        current = self._current_channel()
        for name, rb in self._radios.items():
            entry = chans.get(name) or {}
            rb.setEnabled(bool(entry.get("available")))
            if not entry.get("available"):
                rb.setToolTip(f"Unavailable: {entry.get('reason', 'unknown')}")
            else:
                rb.setToolTip("")
        rb = self._radios.get(current)
        if rb is not None:
            rb.blockSignals(True)
            rb.setChecked(True)
            rb.blockSignals(False)
        self._populate()

    def _on_channel_toggled(self, checked):
        # QRadioButton emits for both the newly checked and newly unchecked
        # button; only act on the one that turned on, or the list is rebuilt
        # twice per click.
        if checked:
            self._populate()

    def _populate(self):
        """Fill the list for the selected channel. Never borrows another
        channel's commits: an unavailable channel gets an empty list and a
        stated reason, because showing the wrong history under a channel's
        name invites pinning to a version that is not what it says."""
        self.list.clear()
        name = self._selected_channel()
        chans = self._overview.get("channels") or {}
        entry = chans.get(name) or {}
        current = self._current_channel()
        pinned = bool(self._overview.get("pinned"))
        head = self._overview.get("head") or ""

        if not entry.get("available"):
            self._note.setText(
                f"The {_CHANNEL_LABELS.get(name, name)} channel is not "
                f"available: {entry.get('reason', 'unknown')}. "
                "Try Check for updates."
            )
            self.switch_btn.setEnabled(False)
            return
        self.switch_btn.setEnabled(True)

        msgs = []
        if name != current:
            msgs.append(f"You are currently on "
                        f"{_CHANNEL_LABELS.get(current, current)}. Switching "
                        f"replaces the application files and may update "
                        f"dependencies on the next start.")
        if name == "dev":
            msgs.append(_DEV_WARNING)
        if pinned and name == current:
            msgs.append("This version is pinned, so updates are paused. "
                        "Choose Latest to resume them.")
        self._note.setText(" ".join(msgs))

        is_here = (name == current)
        tracking_tip = is_here and not pinned
        item = QListWidgetItem(
            _LATEST_TEXT + ("   \u25c0 current" if tracking_tip else ""))
        item.setData(Qt.UserRole, None)
        self.list.addItem(item)
        select = item

        for v in entry.get("versions") or []:
            here = is_here and bool(head) and v["rev"] == head
            marker = "   \u25c0 pinned here" if (here and pinned) else ""
            it = QListWidgetItem(
                f'{v["date"]}   {v["short"]}   {v["subject"]}{marker}')
            it.setData(Qt.UserRole, v["rev"])
            self.list.addItem(it)
            if here and pinned:
                select = it
        self.list.setCurrentItem(select)

    # ------------------------------------------------------------- checking #
    def _check(self):
        self.check_btn.setEnabled(False)
        self.check_status.setText("Checking…")
        self._worker = _CheckWorker(self._u, self._root, self)
        self._worker.done.connect(self._on_check)
        self._worker.start()

    def _on_check(self, payload):
        self.check_btn.setEnabled(True)
        u = self._u
        if isinstance(payload, Exception):
            self.check_status.setText(f"Check failed: {payload}")
            return
        res, overview = payload
        self._overview = overview
        self._apply_overview()
        cur = _describe(u, self._root) or {"short": "unknown", "date": "", "subject": ""}
        self._refresh_title(cur)

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
        elif status == u.PINNED:
            self.check_status.setText(
                "Pinned to a chosen version — updates are paused. "
                "Select Latest below to resume them.")
        elif status == u.OFFLINE:
            self.check_status.setText("Offline — couldn't reach the update server.")
        elif status == u.LOCAL_AHEAD:
            self.check_status.setText("Development checkout (ahead of server); no update.")
        else:
            self.check_status.setText(getattr(res, "message", "No update.") or "No update.")

    # ------------------------------------------------------------ switching #
    def _switch(self):
        """
        Apply the selection: channel switch, then pin or unpin.

        The order is forced -- the pin target only exists locally once the
        switch has fetched it. Dependencies are deliberately NOT touched here;
        see the module docstring.
        """
        item = self.list.currentItem()
        if item is None:
            return
        rev = item.data(Qt.UserRole)
        target = self._selected_channel()
        current = self._current_channel()
        log = lambda m: print(f"[version] {m}")  # noqa: E731

        if target == current and rev is None and not self._overview.get("pinned"):
            QMessageBox.information(
                self, "Nothing to change",
                f"You are already following the latest version of the "
                f"{_CHANNEL_LABELS.get(target, target)} channel.")
            return

        what = (f"the {_CHANNEL_LABELS.get(target, target)} channel"
                if target != current else "this version")
        detail = (f"Switch HIBACHI to {what}"
                  + (f", version {rev[:7]}?" if rev else "?"))
        if QMessageBox.question(
            self, "Switch version",
            f"{detail}\n\n"
            "Your projects and data are untouched. The change takes effect "
            "after you restart the app.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return

        notes = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if target != current:
                res = self._u.switch_channel(self._root, target, logger=log)
                if getattr(res, "status", None) != self._u.UPDATED:
                    QApplication.restoreOverrideCursor()
                    QMessageBox.warning(self, "Switch failed",
                                        getattr(res, "message", "Unknown error."))
                    return
                notes.append(res.message)
                if res.env_changed:
                    # switch_channel has recorded pending_env_update; the
                    # launcher applies it on the next start. We cannot: this
                    # process is running inside the environment in question.
                    notes.append("Dependencies for this channel will be "
                                 "installed the next time HIBACHI starts.")

            if rev:
                ok, msg = self._u.pin_to(self._root, rev, logger=log)
            elif self._u.is_pinned(self._root):
                ok, msg = self._u.unpin(self._root, target, logger=log)
            else:
                ok, msg = True, ""
        finally:
            QApplication.restoreOverrideCursor()

        if not ok:
            QMessageBox.warning(self, "Switch failed", msg)
            return
        if msg:
            notes.append(msg)

        # No skip marker is written any more. `pin_to` detaches HEAD, so the
        # launcher reports PINNED and offers nothing until the pin is released;
        # the old set_skipped_rev(remote_tip) call existed only to suppress the
        # re-offer caused by rewinding the branch pointer.
        self._overview = self._u.channel_overview(self._root, limit=15, fetch=False)
        self._apply_overview()
        cur = _describe(self._u, self._root) or {"short": "unknown", "date": "", "subject": ""}
        self._refresh_title(cur)
        self._sub.setText(f"{cur['date']}  ·  {cur['subject']}".strip(" ·"))
        self._offer_restart(
            "Switched version",
            "\n".join(notes) + "\n\nQuit now and reopen to run this version?",
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

    # Surface a non-default state in the indicator itself: running a
    # development build, or a pinned version with updates paused, should be
    # visible without opening anything.
    try:
        bits = [f"v {label}"]
        if updater.get_channel() != updater.DEFAULT_CHANNEL:
            bits.append(updater.get_channel())
        if updater.is_pinned(repo_root):
            bits.append("pinned")
        text = " · ".join(bits)
    except Exception:  # pragma: no cover - defensive
        text = f"v {label}"

    btn = QPushButton(text)
    btn.setFlat(True)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setToolTip("Manage versions / channel / check for updates")
    btn.setStyleSheet(
        "QPushButton{color:#8a8f98; border:none; padding:0 6px;}"
        "QPushButton:hover{color:#c75b39;}"
    )
    f = btn.font()
    f.setPointSize(max(8, f.pointSize() - 1))
    btn.setFont(f)
    btn.clicked.connect(lambda: VersionDialog(updater, repo_root, main_window).exec_())

    main_window.statusBar().addPermanentWidget(btn)
