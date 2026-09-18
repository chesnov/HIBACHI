"""recipe_library: saved cross-channel recipes, reusable across projects.

A recipe is an ordered list of relation steps -- the thing the recipe dock
builds. Until now the only copy of one was the `recipe.yaml` a run left in its
output folder, so reusing last week's five steps meant rebuilding them by hand
or digging a file out of `RELATIONAL_ANALYSIS/`.

This mirrors `config_library` deliberately: same state directory, same
"user files live under <state_dir>, built-ins are read-only" shape, same
sanitize/save/import/export/rename/delete surface. It is much smaller because a
recipe has no modes, no reconcile against a reference, and no computed state --
it is a plain list of steps.

Qt-free on purpose, so it can be tested and scripted without a display.

Portability note: steps name channels by their registry key (`Channel_0`).
A recipe therefore travels between projects that number their channels the
same way, and `missing_channels` lets a caller say so plainly when they do not.
"""
from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import yaml  # type: ignore

from .project_selection import _default_state_dir

#: Subdirectory under the state dir. Sibling of `configs/`, which holds the
#: per-channel processing presets; these are whole cross-channel recipes.
_SUBDIR = "recipes"

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ +-]+")

#: Step types the dock and the engine understand. `intersect` and `analyze`
#: are the pre-merge spellings; `RelationalEngine.normalise_recipe` maps them,
#: so a recipe saved before the merge is still loadable and still valid.
_KNOWN_TYPES = {"relate", "filter", "primary", "intersect", "analyze"}


class RecipeLibraryError(Exception):
    """A recipe file is unreadable or is not a recipe."""


@dataclass
class RecipeEntry:
    """One saved recipe."""
    name: str          # human-facing base name, e.g. "Olig2 in aggregates"
    path: str          # absolute path to the .yaml
    n_steps: int       # step count, for the picker
    channels: Tuple[str, ...] = ()   # channel keys the recipe names

    @property
    def label(self) -> str:
        step_word = "step" if self.n_steps == 1 else "steps"
        return f"{self.name} \u2014 {self.n_steps} {step_word}"


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
def library_root() -> str:
    """Absolute path to ``<state_dir>/recipes`` (not guaranteed to exist)."""
    return os.path.join(_default_state_dir(), _SUBDIR)


def ensure_library() -> str:
    """Create the library folder if missing; return it."""
    root = library_root()
    os.makedirs(root, exist_ok=True)
    return root


def sanitize_name(name: str) -> str:
    """Filesystem-safe base name, no extension."""
    stem = os.path.splitext(str(name).strip())[0]
    stem = _SAFE_NAME.sub("_", stem).strip(" ._")
    return stem or "recipe"


def _target_path(name: str) -> str:
    return os.path.join(library_root(), f"{sanitize_name(name)}.yaml")


def entry_exists(name: str) -> bool:
    return os.path.exists(_target_path(name))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
def validate(steps: Any) -> List[Dict[str, Any]]:
    """Return `steps` as a recipe, or raise RecipeLibraryError.

    Checked structurally rather than against the engine, so the library stays
    independent of it: a recipe is a non-empty list of dicts, each carrying a
    step type the engine knows. Anything else is a YAML file that happens to
    be in the folder -- most likely a processing config someone filed in the
    wrong place -- and saying so is more use than failing later with a
    KeyError mid-run.
    """
    if not isinstance(steps, list) or not steps:
        raise RecipeLibraryError("not a recipe: expected a non-empty list of steps")
    out: List[Dict[str, Any]] = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            raise RecipeLibraryError(f"step {i + 1} is not a mapping")
        stype = step.get("type")
        if stype not in _KNOWN_TYPES:
            raise RecipeLibraryError(
                f"step {i + 1} has unknown type {stype!r}")
        out.append(dict(step))
    return out


def channels_in(steps: Sequence[Dict[str, Any]]) -> Tuple[str, ...]:
    """Channel keys a recipe names, in first-seen order.

    Mirrors `cross_channel_window.channels_used_by_recipe`, duplicated rather
    than imported so this module stays free of the GUI package. The two are
    pinned together by a test.
    """
    out: List[str] = []
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        candidates = [step.get("primary"), step.get("target"), step.get("input")]
        candidates.extend(step.get("inputs") or [])
        for c in candidates:
            if (isinstance(c, str) and c and c != "PREVIOUS_RESULT"
                    and c not in out):
                out.append(c)
    return tuple(out)


def load(entry: Union[RecipeEntry, str]) -> List[Dict[str, Any]]:
    """The steps of a saved recipe. Raises RecipeLibraryError if unreadable."""
    path = entry.path if isinstance(entry, RecipeEntry) else str(entry)
    try:
        with open(path) as fh:
            data = yaml.safe_load(fh)
    except OSError as exc:
        raise RecipeLibraryError(f"could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RecipeLibraryError(f"not valid YAML: {exc}") from exc
    return validate(data)


def _entry_for(path: str) -> RecipeEntry:
    steps = load(path)
    return RecipeEntry(
        name=os.path.splitext(os.path.basename(path))[0],
        path=path, n_steps=len(steps), channels=channels_in(steps),
    )


def list_library() -> List[RecipeEntry]:
    """Every readable saved recipe, name-sorted. Unreadable files are skipped.

    Skipped rather than raised on, so one bad file cannot make the picker
    empty; `scan_problems` reports them so they are not invisible either.
    """
    root = library_root()
    out: List[RecipeEntry] = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for fname in names:
        if not fname.lower().endswith((".yaml", ".yml")):
            continue
        try:
            out.append(_entry_for(os.path.join(root, fname)))
        except RecipeLibraryError:
            continue
    return sorted(out, key=lambda e: e.name.lower())


def scan_problems() -> List[Tuple[str, str]]:
    """(path, reason) for files in the library that are not usable recipes."""
    root = library_root()
    problems: List[Tuple[str, str]] = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return problems
    for fname in names:
        if not fname.lower().endswith((".yaml", ".yml")):
            continue
        path = os.path.join(root, fname)
        try:
            load(path)
        except RecipeLibraryError as exc:
            problems.append((path, str(exc)))
    return problems


def missing_channels(steps: Sequence[Dict[str, Any]],
                     available: Sequence[str]) -> List[str]:
    """Channels a recipe needs that this project does not have.

    A recipe is portable between projects that number channels the same way.
    When they do not, the caller can say which channels are missing instead of
    loading a recipe that will skip every step at run time.
    """
    have = set(available or ())
    return [c for c in channels_in(steps) if c not in have]


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
def save(steps: Sequence[Dict[str, Any]], name: str,
         overwrite: bool = False) -> RecipeEntry:
    """Write `steps` into the library under `name`.

    Raises FileExistsError on collision unless `overwrite`, so the UI decides
    whether to replace.
    """
    clean = validate(list(steps))
    ensure_library()
    dest = _target_path(name)
    if os.path.exists(dest) and not overwrite:
        raise FileExistsError(f"A recipe named '{name}' already exists.")
    with open(dest, "w") as fh:
        yaml.dump(clean, fh, sort_keys=True)
    return _entry_for(dest)


def import_file(src_path: str, name: Optional[str] = None,
                overwrite: bool = False) -> RecipeEntry:
    """Import an external recipe .yaml into the library.

    This is also how a recipe is recovered from a finished run: every run
    writes `recipe.yaml` beside its results, and that file is exactly this
    format.
    """
    if not os.path.isfile(src_path):
        raise FileNotFoundError(src_path)
    steps = load(src_path)
    chosen = name or os.path.splitext(os.path.basename(src_path))[0]
    return save(steps, chosen, overwrite=overwrite)


def export(entry: Union[RecipeEntry, str], dst_path: str) -> str:
    """Copy a recipe out for sharing, byte-for-byte."""
    src = entry.path if isinstance(entry, RecipeEntry) else str(entry)
    parent = os.path.dirname(os.path.abspath(dst_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    shutil.copy2(src, dst_path)
    return dst_path


def delete(entry: Union[RecipeEntry, str]) -> None:
    path = entry.path if isinstance(entry, RecipeEntry) else str(entry)
    os.remove(path)


def rename(entry: RecipeEntry, new_name: str) -> RecipeEntry:
    """Rename in place. Raises FileExistsError rather than clobbering."""
    dest = _target_path(new_name)
    if os.path.abspath(dest) == os.path.abspath(entry.path):
        return entry
    if os.path.exists(dest):
        raise FileExistsError(f"A recipe named '{new_name}' already exists.")
    ensure_library()
    os.rename(entry.path, dest)
    return _entry_for(dest)
