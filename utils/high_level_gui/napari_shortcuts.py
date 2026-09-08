"""napari_shortcuts: describe napari's own layer-editing modes with live keys.

Why this exists
---------------
napari's Shapes layer controls already provide everything needed to fix a
polygon after it has been closed -- move a vertex, insert one, remove one,
select a whole shape. HIBACHI does not reimplement any of that; it just points
at it. What HIBACHI does need is to *name* those modes in its own dialogs, since
the controls are a row of small unlabelled icons and the keyboard shortcuts are
not guessable.

Why the keys are read rather than written down
----------------------------------------------
They are neither obvious nor stable. In napari 0.4.19 the shapes bindings are
vertex-remove '1', vertex-insert '2', direct '4', select '5' -- and '3' is
DELETE SELECTED SHAPES, so a hard-coded hint that guesses wrong can tell a user
to destroy the polygon they were trying to repair. Users can also rebind any of
them in napari's preferences.

So the keys come from napari's settings at call time. If that lookup fails --
a moved settings layout, a napari version that renames an action -- the mode is
still named, just without a key. A missing hint is a small annoyance; a wrong
one is a lost region.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

# (napari action name, what it does in plain words). Order is the order a user
# would want them: fix a vertex, add one, remove one, then the broader actions.
SHAPE_EDIT_ACTIONS: Tuple[Tuple[str, str], ...] = (
    ("napari:activate_direct_mode", "move a vertex"),
    ("napari:activate_vertex_insert_mode", "insert a vertex"),
    ("napari:activate_vertex_remove_mode", "remove a vertex"),
    ("napari:activate_select_mode", "select or delete a whole polygon"),
    ("napari:activate_add_polygon_mode", "draw another polygon"),
)


def bound_keys() -> dict:
    """napari's current action -> shortcut-string map, or {} if unavailable."""
    try:
        from napari.settings import get_settings
        raw = get_settings().shortcuts.shortcuts
    except Exception:
        return {}

    out = {}
    try:
        for action, keys in raw.items():
            try:
                if keys:
                    out[str(action)] = str(list(keys)[0])
            except Exception:
                continue
    except Exception:
        return {}
    return out


def shape_edit_hints(
    actions: Sequence[Tuple[str, str]] = SHAPE_EDIT_ACTIONS,
    indent: str = "  ",
) -> str:
    """Bulleted lines describing each shape-editing mode and its current key.

    Safe to call at any time: never raises, and degrades to naming the modes
    without keys rather than guessing at them.
    """
    keys = bound_keys()
    lines = []
    for action, description in actions:
        key = keys.get(action, "")
        if key:
            lines.append(f"{indent}\u2022 {key}  \u2014  {description}")
        else:
            lines.append(f"{indent}\u2022 {description}")
    return "\n".join(lines)


def shape_edit_block(lead: str = "") -> str:
    """`shape_edit_hints` wrapped in the sentence HIBACHI's dialogs share.

    One definition of this text, so the drawing dialog and the two failure
    messages that offer the same advice cannot drift apart.
    """
    body = (
        "Edit it with napari's shape controls, at the top of the layer list "
        "panel:\n\n" + shape_edit_hints()
    )
    return f"{lead}\n{body}" if lead else body
