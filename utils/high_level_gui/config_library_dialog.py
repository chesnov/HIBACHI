"""config_library_dialog: the Qt manager for the user's config library.

A single ``QDialog`` for browsing and curating configs. Every data operation goes
through ``config_library`` (aliased ``cl``) -- this file adds no logic of its own,
it only presents the library and forwards user actions. Built-ins are read-only
(the editing actions are disabled for them, and the logic layer also raises
``PermissionError`` as a backstop).

Transparency (see handoff "NO SILENT FALLBACKS"):
  * On open we call ``cl.scan_problems()`` and show any broken config files in a
    banner, so a config that failed to resolve is an explicit warning rather than
    a mysteriously missing row.
  * Every ``cl.*`` call is wrapped so the raised error is shown to the user in a
    ``QMessageBox`` -- never swallowed.

Qt-guarded like ``project_selection`` so the module imports without PyQt5.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from . import config_library as cl
from .config_library import ConfigLibraryError, ConfigModeError

try:
    from PyQt5.QtCore import Qt  # type: ignore
    from PyQt5.QtWidgets import (  # type: ignore
        QDialog, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget,
        QMessageBox, QPushButton, QSplitter, QTextEdit, QTreeWidget,
        QTreeWidgetItem, QVBoxLayout, QWidget,
    )
    _HAVE_QT = True
except Exception:  # pragma: no cover - headless / no Qt
    _HAVE_QT = False


# Order the two mode groups deterministically (3D first, then 2D).
_MODE_ORDER = [(cl.MODE_3D, "3D"), (cl.MODE_2D, "2D")]
_SOURCE_ORDER = [("builtin", "Built-in"), ("library", "My Library")]


if _HAVE_QT:

    class ConfigLibraryDialog(QDialog):
        """Browse / import / duplicate / rename / delete / export library configs."""

        def __init__(self, parent: Optional[QWidget] = None,
                     live_config: Optional[Dict[str, Any]] = None) -> None:
            super().__init__(parent)
            self.setWindowTitle("Config Library")
            self.setModal(True)
            self.resize(820, 560)
            self._live_config = live_config  # optional: enables "Save current…"

            root = QVBoxLayout(self)

            # -- Problems banner (only shown when something is broken) ---------
            self._problem_box = QWidget()
            pb_layout = QVBoxLayout(self._problem_box)
            pb_layout.setContentsMargins(0, 0, 0, 0)
            self._problem_label = QLabel()
            self._problem_label.setWordWrap(True)
            self._problem_label.setStyleSheet(
                "color: #b00020; font-weight: bold;"
            )
            self._problem_list = QListWidget()
            self._problem_list.setMaximumHeight(90)
            pb_layout.addWidget(self._problem_label)
            pb_layout.addWidget(self._problem_list)
            root.addWidget(self._problem_box)

            # -- Split: tree on the left, details on the right -----------------
            split = QSplitter(Qt.Horizontal)

            self.tree = QTreeWidget()
            self.tree.setHeaderHidden(True)
            self.tree.setMinimumWidth(320)
            self.tree.currentItemChanged.connect(self._on_selection_changed)
            self.tree.itemDoubleClicked.connect(lambda *_: self._reveal())
            split.addWidget(self.tree)

            details = QWidget()
            dlay = QVBoxLayout(details)
            self._name_lbl = QLabel("Select a config")
            self._name_lbl.setStyleSheet("font-size: 15px; font-weight: bold;")
            self._name_lbl.setWordWrap(True)
            self._meta_lbl = QLabel("")
            self._meta_lbl.setWordWrap(True)
            self._meta_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._prov = QTextEdit()
            self._prov.setReadOnly(True)
            dlay.addWidget(self._name_lbl)
            dlay.addWidget(self._meta_lbl)
            dlay.addWidget(QLabel("Provenance:"))
            dlay.addWidget(self._prov, stretch=1)
            split.addWidget(details)
            split.setStretchFactor(0, 1)
            split.setStretchFactor(1, 1)
            root.addWidget(split, stretch=1)

            # -- Action buttons ------------------------------------------------
            actions = QHBoxLayout()
            self._btn_import = QPushButton("Import\u2026")
            self._btn_duplicate = QPushButton("Duplicate")
            self._btn_rename = QPushButton("Rename")
            self._btn_delete = QPushButton("Delete")
            self._btn_export = QPushButton("Export preset\u2026")
            self._btn_reveal = QPushButton("Reveal in file browser")

            self._btn_import.clicked.connect(self._import)
            self._btn_duplicate.clicked.connect(self._duplicate)
            self._btn_rename.clicked.connect(self._rename)
            self._btn_delete.clicked.connect(self._delete)
            self._btn_export.clicked.connect(self._export)
            self._btn_reveal.clicked.connect(self._reveal)

            for b in (self._btn_import, self._btn_duplicate, self._btn_rename,
                      self._btn_delete, self._btn_export, self._btn_reveal):
                actions.addWidget(b)

            if self._live_config is not None:
                self._btn_save_current = QPushButton("Save current config\u2026")
                self._btn_save_current.clicked.connect(self._save_current)
                actions.addWidget(self._btn_save_current)

            actions.addStretch(1)
            self._btn_close = QPushButton("Close")
            self._btn_close.clicked.connect(self.accept)
            actions.addWidget(self._btn_close)
            root.addLayout(actions)

            self._refresh()

        # ---- helpers ------------------------------------------------------- #
        def _current_entry(self):
            item = self.tree.currentItem()
            if item is None:
                return None
            return item.data(0, Qt.UserRole)

        def _show_problems(self) -> None:
            """List every unresolved config so it is visible, not silently dropped."""
            try:
                problems = cl.scan_problems()
            except Exception as exc:  # never let the banner itself break the dialog
                problems = [("<scan failed>", str(exc))]
            if problems:
                self._problem_label.setText(
                    f"\u26a0 {len(problems)} config file(s) could not be read and "
                    "were skipped. Fix or remove them:"
                )
                self._problem_list.clear()
                for path, msg in problems:
                    self._problem_list.addItem(f"{path}\n    {msg}")
                self._problem_box.setVisible(True)
            else:
                self._problem_box.setVisible(False)

        def _refresh(self, select_path: Optional[str] = None) -> None:
            """Rebuild the tree grouped Mode -> Source, and restore selection."""
            self._show_problems()
            self.tree.clear()

            try:
                entries = cl.list_builtins() + cl.list_library()
            except Exception as exc:
                QMessageBox.critical(self, "Config error", str(exc))
                entries = []

            by_mode: Dict[str, Dict[str, list]] = {}
            for e in entries:
                by_mode.setdefault(e.mode, {}).setdefault(e.source, []).append(e)

            target_item = None
            for mode, mode_label in _MODE_ORDER:
                sources = by_mode.get(mode)
                if not sources:
                    continue
                mode_node = QTreeWidgetItem([mode_label])
                f = mode_node.font(0)
                f.setBold(True)
                mode_node.setFont(0, f)
                mode_node.setFlags(Qt.ItemIsEnabled)
                self.tree.addTopLevelItem(mode_node)
                for source, source_label in _SOURCE_ORDER:
                    items = sources.get(source)
                    if not items:
                        continue
                    src_node = QTreeWidgetItem([source_label])
                    src_node.setFlags(Qt.ItemIsEnabled)
                    mode_node.addChild(src_node)
                    for entry in sorted(items, key=lambda x: x.name.lower()):
                        # Leaf shows the bare name; mode + source are already the
                        # parent groups, so the full entry.label would be redundant
                        # here. The LibraryEntry itself is carried in item data.
                        leaf = QTreeWidgetItem([entry.name])
                        leaf.setData(0, Qt.UserRole, entry)
                        src_node.addChild(leaf)
                        if select_path and os.path.abspath(entry.path) == os.path.abspath(select_path):
                            target_item = leaf
                mode_node.setExpanded(True)
                for i in range(mode_node.childCount()):
                    mode_node.child(i).setExpanded(True)

            if target_item is not None:
                self.tree.setCurrentItem(target_item)
            self._on_selection_changed()

        def _on_selection_changed(self, *args) -> None:
            entry = self._current_entry()
            editable = bool(entry and entry.editable)

            # Duplicate/Export work on any config; Rename/Delete are library-only.
            has_entry = entry is not None
            self._btn_duplicate.setEnabled(has_entry)
            self._btn_export.setEnabled(has_entry)
            self._btn_rename.setEnabled(editable)
            self._btn_delete.setEnabled(editable)

            if entry is None:
                self._name_lbl.setText("Select a config")
                self._meta_lbl.setText("")
                self._prov.setPlainText("")
                return

            src_label = "Built-in (read-only)" if not entry.editable else "My Library"
            mode_label = dict(_MODE_ORDER).get(entry.mode, entry.mode)
            self._name_lbl.setText(entry.name)
            self._meta_lbl.setText(
                f"Mode: {mode_label}    Source: {src_label}\n{entry.path}"
            )

            # Provenance preview (read-only; never raises on a bad mode).
            try:
                prov = cl.read_provenance(entry.path)
            except Exception as exc:
                self._prov.setPlainText(f"(could not read provenance: {exc})")
                return
            ver = prov.get("hibachi_version")
            if isinstance(ver, dict):
                ver_text = ver.get("short") or ver.get("commit") or ver.get("processed_at") or "present"
            else:
                ver_text = ver if ver else "(none recorded)"
            kind = ("Full run record (carries saved_state / dimensions)"
                    if prov.get("is_full_run")
                    else "Portable preset")
            self._prov.setPlainText(
                f"Kind: {kind}\n"
                f"hibachi_version: {ver_text}\n"
                f"has_saved_state: {prov.get('has_saved_state')}\n"
                f"has_dimensions:  {prov.get('has_dimensions')}"
            )

        # ---- actions ------------------------------------------------------- #
        def _import(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, "Import config into library", "",
                "YAML Files (*.yaml *.yml);;All Files (*)"
            )
            if not path:
                return
            try:
                entry = cl.import_config(path)
            except ConfigModeError as exc:
                QMessageBox.critical(
                    self, "Config error",
                    f"That file has no valid 'mode' and can't be imported:\n\n{exc}"
                )
                return
            except FileExistsError:
                reply = QMessageBox.question(
                    self, "Already exists",
                    "A library config with that name already exists.\n\nOverwrite it?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return
                try:
                    entry = cl.import_config(path, overwrite=True)
                except ConfigLibraryError as exc:
                    QMessageBox.critical(self, "Config error", str(exc))
                    return
            except (ConfigLibraryError, OSError) as exc:
                QMessageBox.critical(self, "Config error", str(exc))
                return
            self._refresh(select_path=entry.path)

        def _duplicate(self) -> None:
            entry = self._current_entry()
            if entry is None:
                return
            new_name, ok = QInputDialog.getText(
                self, "Duplicate config", "New name:", text=f"{entry.name} copy"
            )
            if not ok or not new_name.strip():
                return
            try:
                created = cl.duplicate_config(entry, new_name.strip())
            except FileExistsError:
                QMessageBox.warning(
                    self, "Name in use",
                    f"A library config named '{new_name.strip()}' already exists."
                )
                return
            except (ConfigLibraryError, OSError) as exc:
                QMessageBox.critical(self, "Config error", str(exc))
                return
            self._refresh(select_path=created.path)

        def _rename(self) -> None:
            entry = self._current_entry()
            if entry is None or not entry.editable:
                return
            new_name, ok = QInputDialog.getText(
                self, "Rename config", "New name:", text=entry.name
            )
            if not ok or not new_name.strip():
                return
            try:
                renamed = cl.rename_config(entry, new_name.strip())
            except FileExistsError:
                QMessageBox.warning(
                    self, "Name in use",
                    f"A library config named '{new_name.strip()}' already exists."
                )
                return
            except (PermissionError, ConfigLibraryError, OSError) as exc:
                QMessageBox.critical(self, "Config error", str(exc))
                return
            self._refresh(select_path=renamed.path)

        def _delete(self) -> None:
            entry = self._current_entry()
            if entry is None or not entry.editable:
                return
            reply = QMessageBox.question(
                self, "Delete config",
                f"Delete '{entry.name}' from your library?\n\nThis cannot be undone.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
            try:
                cl.delete_config(entry)
            except (PermissionError, ConfigLibraryError, OSError) as exc:
                QMessageBox.critical(self, "Config error", str(exc))
                return
            self._refresh()

        def _export(self) -> None:
            entry = self._current_entry()
            if entry is None:
                return
            dst, _ = QFileDialog.getSaveFileName(
                self, "Export preset", f"{entry.name}.yaml",
                "YAML Files (*.yaml *.yml);;All Files (*)"
            )
            if not dst:
                return
            try:
                cl.export_config(entry, dst)
            except (ConfigLibraryError, OSError) as exc:
                QMessageBox.critical(self, "Config error", str(exc))
                return
            QMessageBox.information(self, "Exported", f"Saved preset to:\n{dst}")

        def _reveal(self) -> None:
            entry = self._current_entry()
            target = entry.path if entry is not None else None
            try:
                ok = cl.reveal_in_file_browser(target or cl.library_root())
            except Exception as exc:
                ok = False
                QMessageBox.critical(self, "Config error", str(exc))
                return
            if not ok:
                QMessageBox.warning(
                    self, "Could not open",
                    "HIBACHI could not open a file browser here. The library lives "
                    f"at:\n\n{cl.library_root()}"
                )

        def _save_current(self) -> None:
            if self._live_config is None:
                return
            try:
                default_name = "tuned config"
                name, ok = QInputDialog.getText(
                    self, "Save current config", "Name:", text=default_name
                )
                if not ok or not name.strip():
                    return
                try:
                    entry = cl.save_to_library(self._live_config, name.strip())
                except FileExistsError:
                    reply = QMessageBox.question(
                        self, "Already exists",
                        f"A library config named '{name.strip()}' already exists.\n\n"
                        "Overwrite it?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return
                    entry = cl.save_to_library(self._live_config, name.strip(), overwrite=True)
            except ConfigModeError as exc:
                QMessageBox.critical(
                    self, "Config error",
                    f"This config has no valid 'mode', so it can't be saved:\n\n{exc}"
                )
                return
            except (ConfigLibraryError, OSError) as exc:
                QMessageBox.critical(self, "Config error", str(exc))
                return
            self._refresh(select_path=entry.path)


def open_config_library(parent=None, live_config: Optional[Dict[str, Any]] = None) -> None:
    """Convenience launcher for the manager dialog."""
    if not _HAVE_QT:  # pragma: no cover - defensive
        raise RuntimeError("open_config_library requires PyQt5 (no display available).")
    dlg = ConfigLibraryDialog(parent, live_config=live_config)
    dlg.exec_()
