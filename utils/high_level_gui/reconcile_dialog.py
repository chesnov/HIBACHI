"""reconcile_dialog: the transparency prompt shown before a config is rewritten.

When a template/preset is applied to a project, ``apply_template_config_to_project``
reconciles it against the current canonical built-in schema (``default.yaml``)
*before* anything is written. If the template differs from the current pipeline
(a step was added/removed, a parameter's range changed, a value fell out of
range, ...), the difference must be shown to the user and explicitly confirmed
rather than applied silently. This module renders that difference.

The core rule (see the handoff, "NO SILENT FALLBACKS") lives in the logic layer:
``config_library.reconcile`` raises ``ConfigModeError`` / ``ReferenceMissingError``
for problems it cannot resolve, and those propagate to the caller. This dialog is
purely presentational -- it takes a ready ``ReconcileResult`` and returns the
user's yes/no. It never swallows an error and never invents a fallback.

Qt-guarded like ``project_selection``: the module imports cleanly with no Qt so
the rest of the package (and the headless tests) are unaffected; the callables
raise a clear error only if actually invoked without Qt.
"""

from __future__ import annotations

from typing import Any, Callable

# --------------------------------------------------------------------------- #
# Qt guard -- mirror project_selection so importing this module never requires
# PyQt5 (keeps headless import / tests working).
# --------------------------------------------------------------------------- #
try:
    from PyQt5.QtCore import Qt  # type: ignore
    from PyQt5.QtWidgets import (  # type: ignore
        QDialog, QDialogButtonBox, QLabel, QTreeWidget, QTreeWidgetItem,
        QVBoxLayout, QWidget,
    )
    _HAVE_QT = True
except Exception:  # pragma: no cover - headless / no Qt
    _HAVE_QT = False


_HEADER_TEXT = (
    "This config was tuned against a different version of the pipeline.\n\n"
    "HIBACHI will bring it up to date with the current pipeline: the step and "
    "parameter structure below comes from the current pipeline, and your tuned "
    "values are preserved wherever they still apply. Review the changes and "
    "choose Apply to continue, or Cancel to leave everything unchanged."
)


def _populate_tree(tree: "QTreeWidget", result: Any) -> None:
    """Fill ``tree`` with three grouped, read-only sections built from the result.

    Sections: Steps added / Steps removed / Parameter changes. Empty sections are
    shown with an explicit "(none)" child so the absence is visible rather than
    ambiguous.
    """
    tree.clear()
    tree.setHeaderHidden(True)

    def _section(title: str, rows: list) -> None:
        parent = QTreeWidgetItem([f"{title}  ({len(rows)})"])
        f = parent.font(0)
        f.setBold(True)
        parent.setFont(0, f)
        parent.setFlags(Qt.ItemIsEnabled)  # group header: not selectable
        tree.addTopLevelItem(parent)
        if not rows:
            child = QTreeWidgetItem(["(none)"])
            child.setFlags(Qt.ItemIsEnabled)
            parent.addChild(child)
        else:
            for text in rows:
                child = QTreeWidgetItem([text])
                child.setFlags(Qt.ItemIsEnabled)
                parent.addChild(child)
        parent.setExpanded(True)

    # Steps added: present in the current pipeline but missing from the source.
    added = [f"{s}   (added from current defaults)" for s in result.added_steps]
    # Steps removed: in the source but no longer defined by the pipeline.
    removed = [f"{s}   (no longer in the pipeline)" for s in result.removed_steps]
    # Parameter changes: added / removed / type_changed / clamped / reset_invalid.
    param_rows = []
    for c in result.param_changes:
        tail = f"  \u2014 {c.detail}" if getattr(c, "detail", "") else ""
        param_rows.append(f"{c.step} / {c.param}: {c.kind}{tail}")

    _section("Steps added", added)
    _section("Steps removed", removed)
    _section("Parameter changes", param_rows)


if _HAVE_QT:

    class ReconcileDialog(QDialog):
        """Modal dialog that renders a ``ReconcileResult`` and asks Apply/Cancel."""

        def __init__(self, result: Any, parent: "QWidget" = None, *,
                     title: str = "Update config to current pipeline?",
                     context: str = "", impact_lines=None) -> None:
            super().__init__(parent)
            self.setWindowTitle(title)
            self.setModal(True)
            self.setMinimumSize(560, 440)

            layout = QVBoxLayout(self)

            if context:
                ctx = QLabel(context)
                ctx.setStyleSheet("font-weight: bold; color: #2E8B57;")
                ctx.setWordWrap(True)
                layout.addWidget(ctx)

            header = QLabel(_HEADER_TEXT)
            header.setWordWrap(True)
            layout.addWidget(header)

            # Prominent, unmissable warning about results that will be deleted.
            # Old outputs computed with the stale parameters must not survive, or
            # they'd masquerade as having been produced by the new ones.
            if impact_lines:
                warn = QLabel(
                    "\u26a0 The parameters below changed, so the results already "
                    "computed for these steps (and every step after them) will be "
                    "deleted and must be re-processed:"
                )
                warn.setWordWrap(True)
                warn.setStyleSheet(
                    "color: #b00020; font-weight: bold; "
                    "border: 1px solid #b00020; border-radius: 6px; padding: 6px;"
                )
                layout.addWidget(warn)
                impact = QLabel("\n".join(f"   \u2022 {ln}" for ln in impact_lines))
                impact.setWordWrap(True)
                impact.setStyleSheet("color: #b00020;")
                layout.addWidget(impact)

            # Read-only, scrollable tree grouped under the three headers. A
            # QTreeWidget is itself scrollable, so it satisfies the "inside a
            # scroll area" requirement without a redundant QScrollArea wrapper.
            self.tree = QTreeWidget()
            self.tree.setSelectionMode(QTreeWidget.NoSelection)
            self.tree.setFocusPolicy(Qt.NoFocus)
            _populate_tree(self.tree, result)
            layout.addWidget(self.tree, stretch=1)

            buttons = QDialogButtonBox()
            apply_label = "Apply & clear results" if impact_lines else "Apply"
            self._apply_btn = buttons.addButton(apply_label, QDialogButtonBox.AcceptRole)
            buttons.addButton("Cancel", QDialogButtonBox.RejectRole)
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            self._apply_btn.setDefault(True)
            layout.addWidget(buttons)


def confirm_reconcile(parent, result: Any, *,
                      title: str = "Update config to current pipeline?",
                      context: str = "", impact_lines=None) -> bool:
    """Show the reconcile diff and return True iff the user chose Apply.

    ``result`` is a ``config_library.ReconcileResult``. A clean result (no
    differences) returns True without prompting -- there is nothing to confirm.
    ``impact_lines`` is an optional list of human-readable step labels whose
    already-computed results will be deleted if the user proceeds; when given,
    the dialog shows a prominent warning and the Apply button says so.
    """
    if getattr(result, "is_clean", False):
        return True
    if not _HAVE_QT:  # pragma: no cover - defensive; UI path always has Qt
        raise RuntimeError("confirm_reconcile requires PyQt5 (no display available).")
    dlg = ReconcileDialog(result, parent, title=title, context=context,
                          impact_lines=impact_lines)
    return dlg.exec_() == QDialog.Accepted


def make_reconcile_confirm(parent, *, context: str = "") -> Callable[[Any], bool]:
    """Return a ``callable(result) -> bool`` to pass as ``reconcile_confirm=``.

    This is what you hand to ``apply_template_config_to_project`` so it can
    surface a stale template's diff and wait for the user before writing.
    """
    def _confirm(result: Any) -> bool:
        return confirm_reconcile(parent, result, context=context)
    return _confirm