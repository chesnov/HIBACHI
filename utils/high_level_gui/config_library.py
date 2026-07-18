"""config_library: a user-owned, cross-project library of processing configs.

Design notes
------------
A HIBACHI config YAML is *both schema and values*: every label, ``type``,
``min``/``max``/``step`` and default lives in the file, and the interactive GUI
is built entirely from it. There is therefore no in-code parameter schema to
validate against. Instead, the curated in-repo **built-in** config for each mode
(``default.yaml``) is the canonical reference schema, and ``reconcile()`` merges
a source config against it: current structure from the built-in, tuned values
carried over from the source, with every add/remove/clamp reported so the change
can be shown to the user rather than applied silently.

The library itself lives outside the repo, under the same state dir the launcher
already uses (``$HIBACHI_STATE_DIR`` or ``~/.hibachi``), so it is never committed,
never clobbered by a ``git pull`` / rollback, and shared across all projects:

    <state_dir>/configs/2d/<name>.yaml     # mode == fluorescence_2d
    <state_dir>/configs/3d/<name>.yaml     # mode == fluorescence

Files are stored under a mode subfolder for tidiness, but ``mode`` is always
read from the file (top-level ``mode:``, falling back to the ``_2d`` step-key
suffix), so a misplaced or externally-shared file still resolves correctly.

This module is Qt-free and headless-testable, mirroring ``RecentProjects``.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import sys
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml  # type: ignore

# Reuse the launcher's state-dir resolution so the library sits alongside
# recent_projects.json. This is a hard import on purpose: there is exactly one
# source of truth for where state lives, and no silent fallback that could put
# the library somewhere else if the import ever failed.
from .project_selection import _default_state_dir


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MODE_3D = "fluorescence"
MODE_2D = "fluorescence_2d"

# The canonical reference file per mode (see module docstring).
REFERENCE_FILENAME = "default.yaml"

# Keys that are image-specific or run-specific and must never travel with a
# reusable library *preset*. Note that ``hibachi_version`` is deliberately NOT
# stripped: it is provenance (which pipeline version tuned this preset) and is
# safe/useful to share. Full run reproducibility (which also needs saved_state
# and dimensions) is handled separately via the run-config export helpers below.
_IMAGE_SPECIFIC_KEYS = (
    "saved_state",
    "voxel_dimensions",
    "pixel_dimensions",
    "synthetic",
)

_SOURCE_BUILTIN = "builtin"
_SOURCE_LIBRARY = "library"

_SOURCE_LABEL = {_SOURCE_BUILTIN: "Built-in", _SOURCE_LIBRARY: "My Library"}
_MODE_LABEL = {MODE_3D: "3D", MODE_2D: "2D"}
_MODE_SUBDIR = {MODE_3D: "3d", MODE_2D: "2d"}


# --------------------------------------------------------------------------- #
# Errors — surfaced to the user; never swallowed with a silent fallback.
# --------------------------------------------------------------------------- #
class ConfigLibraryError(Exception):
    """Base class for config-library problems the user should be told about."""


class ConfigModeError(ConfigLibraryError):
    """A config's processing mode could not be determined unambiguously."""


class ReferenceMissingError(ConfigLibraryError):
    """No canonical reference (default.yaml) exists for a mode."""


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class LibraryEntry:
    """A single discoverable config (built-in or user library)."""
    name: str          # human-facing base name, e.g. "iMG"
    path: str          # absolute path to the .yaml
    mode: str          # MODE_3D or MODE_2D
    source: str        # _SOURCE_BUILTIN or _SOURCE_LIBRARY

    @property
    def editable(self) -> bool:
        return self.source == _SOURCE_LIBRARY

    @property
    def label(self) -> str:
        src = _SOURCE_LABEL.get(self.source, self.source)
        md = _MODE_LABEL.get(self.mode, "?")
        return f"{self.name} \u2014 {src} ({md})"


@dataclass
class ParamChange:
    """One reconcile difference within a step block."""
    step: str
    param: str
    kind: str          # 'added' | 'removed' | 'type_changed' | 'clamped' | 'structure'
    detail: str = ""


@dataclass
class ReconcileResult:
    """Outcome of merging a source config against the canonical reference."""
    merged: Dict[str, Any]
    added_steps: List[str] = field(default_factory=list)
    removed_steps: List[str] = field(default_factory=list)
    param_changes: List[ParamChange] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (self.added_steps or self.removed_steps or self.param_changes)

    def summary_lines(self) -> List[str]:
        """Flat, human-readable diff lines for a prompt (no Qt here)."""
        out: List[str] = []
        for s in self.added_steps:
            out.append(f"+ step added from current pipeline:  {s}")
        for s in self.removed_steps:
            out.append(f"- step no longer in pipeline (dropped):  {s}")
        for c in self.param_changes:
            sign = {"added": "+", "removed": "-"}.get(c.kind, "~")
            tail = f"  ({c.detail})" if c.detail else ""
            out.append(f"{sign} {c.step} / {c.param}: {c.kind}{tail}")
        return out


# --------------------------------------------------------------------------- #
# YAML helpers
# --------------------------------------------------------------------------- #
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _dump_yaml(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)


# --------------------------------------------------------------------------- #
# Mode detection
# --------------------------------------------------------------------------- #
def mode_of(config: Union[str, Dict[str, Any]]) -> str:
    """Resolve the mode of a config from its explicit top-level ``mode:`` key.

    There is intentionally no inference: a config that does not declare a valid
    ``mode`` is treated as an error the user must fix, not silently guessed. This
    keeps the pipeline honest about what a file actually is.

    Raises:
        ConfigModeError: if ``mode`` is absent or not one of the known modes.
    """
    if isinstance(config, str):
        src = config
        data = _load_yaml(config)
    else:
        src = "<in-memory config>"
        data = config or {}

    declared = data.get("mode")
    if declared in (MODE_2D, MODE_3D):
        return declared
    raise ConfigModeError(
        f"Config has no valid 'mode' (found {declared!r}) in {src}. "
        f"Expected '{MODE_3D}' (3D) or '{MODE_2D}' (2D)."
    )


# --------------------------------------------------------------------------- #
# Library location & naming
# --------------------------------------------------------------------------- #
def library_root() -> str:
    """Absolute path to ``<state_dir>/configs`` (not guaranteed to exist yet)."""
    return os.path.join(_default_state_dir(), "configs")


def _mode_dir(mode: str) -> str:
    return os.path.join(library_root(), _MODE_SUBDIR.get(mode, "3d"))


def ensure_library() -> str:
    """Create the library folder tree if missing; return the root."""
    for mode in (MODE_2D, MODE_3D):
        os.makedirs(_mode_dir(mode), exist_ok=True)
    return library_root()


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ +-]+")


def sanitize_name(name: str) -> str:
    """Filesystem-safe base name (no extension), collapsing junk to underscores."""
    stem = os.path.splitext(str(name).strip())[0]
    stem = _SAFE_NAME.sub("_", stem).strip(" ._") or "config"
    return stem


def _display_name_from_file(filename: str) -> str:
    """Display name for a config = its filename stem, verbatim.

    Names are shown exactly as saved (e.g. ``iMG``, ``ps129``, ``cortex v2``)
    rather than being title-cased, so casing and spacing are the user's choice.
    """
    return os.path.splitext(filename)[0]


# --------------------------------------------------------------------------- #
# Built-in discovery
# --------------------------------------------------------------------------- #
def _builtin_dirs() -> List[Tuple[str, str]]:
    """(config_dir, mode) for the two in-repo module config folders."""
    here = os.path.dirname(os.path.abspath(__file__))
    return [
        (os.path.join(here, "..", "module_3d", "configs"), MODE_3D),
        (os.path.join(here, "..", "module_2d", "configs"), MODE_2D),
    ]


def _scan_dir(
    config_dir: str, source: str
) -> Tuple[List[LibraryEntry], List[Tuple[str, str]]]:
    """Scan one config folder.

    Returns (entries, problems). ``problems`` is a list of (path, message) for
    files whose mode can't be resolved — they are omitted from ``entries`` but
    reported so the UI can warn the user rather than the file vanishing silently.
    """
    entries: List[LibraryEntry] = []
    problems: List[Tuple[str, str]] = []
    if not os.path.isdir(config_dir):
        return entries, problems
    for f in sorted(os.listdir(config_dir)):
        if not f.lower().endswith((".yaml", ".yml")):
            continue
        full = os.path.abspath(os.path.join(config_dir, f))
        try:
            file_mode = mode_of(full)
        except Exception as exc:  # ConfigModeError or malformed YAML
            problems.append((full, str(exc)))
            continue
        entries.append(LibraryEntry(
            name=_display_name_from_file(f),
            path=full, mode=file_mode, source=source,
        ))
    return entries, problems


def list_builtins() -> List[LibraryEntry]:
    entries: List[LibraryEntry] = []
    for config_dir, _mode in _builtin_dirs():
        entries.extend(_scan_dir(config_dir, _SOURCE_BUILTIN)[0])
    return entries


def list_library() -> List[LibraryEntry]:
    entries: List[LibraryEntry] = []
    for mode in (MODE_3D, MODE_2D):
        entries.extend(_scan_dir(_mode_dir(mode), _SOURCE_LIBRARY)[0])
    return entries


def scan_problems() -> List[Tuple[str, str]]:
    """Every config file (built-in or library) that failed to resolve, as
    (path, message). The manager/wizard should display these so a broken config
    is an explicit warning, not a mysteriously missing entry."""
    problems: List[Tuple[str, str]] = []
    for config_dir, _mode in _builtin_dirs():
        problems.extend(_scan_dir(config_dir, _SOURCE_BUILTIN)[1])
    for mode in (MODE_3D, MODE_2D):
        problems.extend(_scan_dir(_mode_dir(mode), _SOURCE_LIBRARY)[1])
    return problems


def list_all(mode: Optional[str] = None) -> List[LibraryEntry]:
    """All discoverable configs (built-in + library), optionally filtered by mode.

    Both sources are surfaced side by side with distinct labels rather than one
    overriding the other: showing a built-in and a same-named library config
    together is more transparent than silently hiding one.
    """
    entries = list_builtins() + list_library()
    if mode is not None:
        entries = [e for e in entries if e.mode == mode]
    return entries


def scan_available_presets() -> Dict[str, Dict[str, str]]:
    """Drop-in replacement for the original ``scan_available_presets``.

    Returns ``{label: {"path", "default_mode", "source"}}``, merging built-in and
    library configs. ``default_mode`` now comes from the file itself (via
    ``mode_of``) instead of being inferred from the folder, and ``source`` is
    added for callers that want to group/label. The label's first whitespace
    token remains the clean config name, preserving ``channel_target_name``.
    """
    presets: Dict[str, Dict[str, str]] = {}
    for entry in list_all():
        presets[entry.label] = {
            "path": entry.path,
            "default_mode": entry.mode,
            "source": entry.source,
        }
    return presets


# --------------------------------------------------------------------------- #
# Sanitization (make a config safe to store as a reusable library entry)
# --------------------------------------------------------------------------- #
def sanitize_for_library(config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Return a copy with image/run-specific keys removed and ``mode`` preserved."""
    data = _load_yaml(config) if isinstance(config, str) else copy.deepcopy(config)
    clean: Dict[str, Any] = {}
    resolved_mode = mode_of(data)
    for key, val in data.items():
        if key in _IMAGE_SPECIFIC_KEYS:
            continue
        clean[key] = copy.deepcopy(val)
    clean["mode"] = resolved_mode
    return clean


# --------------------------------------------------------------------------- #
# Library CRUD
# --------------------------------------------------------------------------- #
def _target_path(name: str, mode: str) -> str:
    return os.path.join(_mode_dir(mode), f"{sanitize_name(name)}.yaml")


def entry_exists(name: str, mode: str) -> bool:
    return os.path.exists(_target_path(name, mode))


def save_to_library(
    config: Union[str, Dict[str, Any]],
    name: str,
    mode: Optional[str] = None,
    overwrite: bool = False,
) -> LibraryEntry:
    """Sanitize and write ``config`` into the library under ``name``.

    ``mode`` defaults to the config's own resolved mode. Raises ``FileExistsError``
    on collision unless ``overwrite`` is True, so the UI can decide.
    """
    clean = sanitize_for_library(config)
    resolved_mode = mode or clean.get("mode") or MODE_3D
    clean["mode"] = resolved_mode

    ensure_library()
    dest = _target_path(name, resolved_mode)
    if os.path.exists(dest) and not overwrite:
        raise FileExistsError(f"A library config named '{name}' already exists.")
    _dump_yaml(clean, dest)
    return LibraryEntry(
        name=_display_name_from_file(os.path.basename(dest)),
        path=dest, mode=resolved_mode, source=_SOURCE_LIBRARY,
    )


def import_config(
    src_path: str, name: Optional[str] = None, overwrite: bool = False
) -> LibraryEntry:
    """Import an external ``.yaml`` into the library (sanitized)."""
    if not os.path.isfile(src_path):
        raise FileNotFoundError(src_path)
    data = _load_yaml(src_path)
    chosen = name or _display_name_from_file(os.path.basename(src_path))
    return save_to_library(data, chosen, mode=mode_of(data), overwrite=overwrite)


def export_config(entry: Union[LibraryEntry, str], dst_path: str) -> str:
    """Copy a config out to an arbitrary path for sharing, byte-for-byte.

    The file is copied verbatim, so exporting a library preset shares the
    preset (sanitized, but retaining its ``hibachi_version`` provenance), while
    exporting a processed-run config (see ``export_run_config``) shares the full,
    reproducible record.
    """
    src = entry.path if isinstance(entry, LibraryEntry) else entry
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    shutil.copy2(src, dst_path)
    return dst_path


def export_run_config(run_config_path: str, dst_path: str) -> str:
    """Export a processed run's config verbatim, for full reproducibility.

    Unlike saving to the library, nothing is stripped: ``saved_state`` (computed
    thresholds), ``voxel/pixel_dimensions`` (calibration) and ``hibachi_version``
    (the exact pipeline commit) are all preserved so a collaborator can reproduce
    the run — e.g. by checking out the recorded version. This is the intended way
    to share a non-stripped config.
    """
    if not os.path.isfile(run_config_path):
        raise FileNotFoundError(run_config_path)
    return export_config(run_config_path, dst_path)


def read_provenance(path: str) -> Dict[str, Any]:
    """Summarise what a shared config file contains, for display before use.

    Returns a dict with ``mode`` (or None if unresolved), ``hibachi_version``,
    ``has_saved_state``, ``has_dimensions``, and ``is_full_run`` (True when it
    carries run-specific state, i.e. it's a reproducibility record rather than a
    portable preset). Never raises on a bad mode — this is a read-only preview.
    """
    data = _load_yaml(path)
    try:
        resolved = mode_of(data)
    except ConfigModeError:
        resolved = None
    has_state = "saved_state" in data
    has_dims = "voxel_dimensions" in data or "pixel_dimensions" in data
    return {
        "mode": resolved,
        "hibachi_version": data.get("hibachi_version"),
        "has_saved_state": has_state,
        "has_dimensions": has_dims,
        "is_full_run": has_state or has_dims,
    }


def _require_editable(entry: LibraryEntry) -> None:
    if not entry.editable:
        raise PermissionError("Built-in configs are read-only.")


def delete_config(entry: LibraryEntry) -> None:
    _require_editable(entry)
    if os.path.exists(entry.path):
        os.remove(entry.path)


def rename_config(entry: LibraryEntry, new_name: str) -> LibraryEntry:
    _require_editable(entry)
    dest = _target_path(new_name, entry.mode)
    if os.path.abspath(dest) == os.path.abspath(entry.path):
        return entry
    if os.path.exists(dest):
        raise FileExistsError(f"A library config named '{new_name}' already exists.")
    os.rename(entry.path, dest)
    return LibraryEntry(
        name=_display_name_from_file(os.path.basename(dest)),
        path=dest, mode=entry.mode, source=_SOURCE_LIBRARY,
    )


def duplicate_config(entry: LibraryEntry, new_name: str) -> LibraryEntry:
    """Copy any config (built-in or library) into the library under a new name."""
    return save_to_library(entry.path, new_name, mode=entry.mode, overwrite=False)


# --------------------------------------------------------------------------- #
# Reconcile against the canonical reference
# --------------------------------------------------------------------------- #
def builtin_reference(mode: str) -> Dict[str, Any]:
    """Load the canonical reference config (``default.yaml``) for a mode.

    Raises:
        ReferenceMissingError: if the mode has no ``default.yaml``. There is no
        fallback to some other built-in: the canonical schema must be explicit,
        or the user is told to add one, rather than silently reconciling against
        an arbitrary file.
    """
    for config_dir, dir_mode in _builtin_dirs():
        if dir_mode != mode:
            continue
        ref = os.path.join(config_dir, REFERENCE_FILENAME)
        if os.path.isfile(ref):
            return _load_yaml(ref)
    raise ReferenceMissingError(
        f"No canonical reference '{REFERENCE_FILENAME}' for mode '{mode}'. "
        f"Add one to the module's configs/ folder."
    )


def _step_keys(config: Dict[str, Any]) -> List[str]:
    return [k for k in config if isinstance(k, str) and k.startswith("execute_")]


def _coerce_value(value: Any, pdef: Dict[str, Any]) -> Tuple[Any, Optional[ParamChange]]:
    """Coerce a carried-over value to the reference param's type/range.

    Returns (value, change_or_None). Complex types (list, scale_table*) are kept
    verbatim; a structural note is emitted if the reference type differs.
    """
    ptype = pdef.get("type")
    if ptype in ("float", "int"):
        try:
            num = float(value)
        except (TypeError, ValueError):
            # Not silently swapped in: reported as a change so the user sees it.
            return pdef.get("value"), ParamChange(
                step="", param="", kind="reset_invalid",
                detail=f"value {value!r} not a number; reset to default "
                       f"{pdef.get('value')!r}",
            )
        lo, hi = pdef.get("min"), pdef.get("max")
        clamped = num
        if isinstance(lo, (int, float)):
            clamped = max(clamped, float(lo))
        if isinstance(hi, (int, float)):
            clamped = min(clamped, float(hi))
        if ptype == "int":
            clamped = int(round(clamped))
        if clamped != num:
            return clamped, ParamChange(
                step="", param="", kind="clamped",
                detail=f"{num} -> {clamped} (new range {lo}..{hi})",
            )
        return (int(clamped) if ptype == "int" else clamped), None
    if ptype == "bool":
        return bool(value), None
    return value, None  # list / scale_table / str: keep as-is


def reconcile(
    source: Union[str, Dict[str, Any]],
    reference: Optional[Dict[str, Any]] = None,
    mode: Optional[str] = None,
) -> ReconcileResult:
    """Merge ``source`` onto the canonical reference, reporting every change.

    Structure (which steps/params exist, and their metadata) comes from the
    reference; tuned *values* are carried over from the source where keys match.
    Non-processing keys the source legitimately owns (dimensions, saved_state)
    are preserved, and ``mode`` is normalised to the reference mode. Nothing is
    dropped or added silently — the caller shows ``summary_lines()`` first.
    """
    src = _load_yaml(source) if isinstance(source, str) else copy.deepcopy(source)
    resolved_mode = mode or mode_of(src)
    # No silent passthrough: if the mode has no canonical reference this raises
    # ReferenceMissingError, which the caller surfaces to the user.
    ref = reference if reference is not None else builtin_reference(resolved_mode)

    merged: Dict[str, Any] = copy.deepcopy(ref)
    result = ReconcileResult(merged=merged)

    ref_steps = set(_step_keys(ref))
    src_steps = set(_step_keys(src))

    # Steps present in the reference but not the source -> added from defaults.
    for step in sorted(ref_steps - src_steps):
        result.added_steps.append(step)

    # Steps in the source that the pipeline no longer defines -> dropped.
    for step in sorted(src_steps - ref_steps):
        result.removed_steps.append(step)

    # Shared steps: carry values, adopt reference metadata, diff parameters.
    for step in sorted(ref_steps & src_steps):
        ref_params = (ref[step] or {}).get("parameters", {}) or {}
        src_params = (src[step] or {}).get("parameters", {}) or {}

        for pname, ref_pdef in ref_params.items():
            if pname in src_params:
                src_pdef = src_params[pname] or {}
                new_val, change = _coerce_value(src_pdef.get("value"), ref_pdef)
                merged[step]["parameters"][pname]["value"] = new_val
                if src_pdef.get("type") != ref_pdef.get("type"):
                    result.param_changes.append(ParamChange(
                        step, pname, "type_changed",
                        f"{src_pdef.get('type')} -> {ref_pdef.get('type')}",
                    ))
                if change is not None:
                    change.step, change.param = step, pname
                    result.param_changes.append(change)
            else:
                result.param_changes.append(ParamChange(
                    step, pname, "added",
                    f"default {ref_pdef.get('value')!r}",
                ))

        for pname in src_params:
            if pname not in ref_params:
                result.param_changes.append(ParamChange(step, pname, "removed"))

    # Preserve the source's non-processing keys (dimensions, saved_state, etc.),
    # but normalise mode to the reference. Reference-only extras are left intact.
    for key, val in src.items():
        if key.startswith("execute_") or key == "mode":
            continue
        merged[key] = copy.deepcopy(val)
    merged["mode"] = resolved_mode

    return result


# --------------------------------------------------------------------------- #
# Reveal in the OS file browser (cross-platform)
# --------------------------------------------------------------------------- #
def reveal_in_file_browser(path: str) -> bool:
    """Open the OS file browser at ``path`` (or its parent, if it's a file).

    Best-effort: returns True if a command was launched. On Windows a file is
    selected in Explorer; elsewhere the containing folder is opened.
    """
    if not path:
        return False
    target = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    try:
        if sys.platform.startswith("win"):
            if os.path.isfile(path):
                subprocess.run(["explorer", "/select,", os.path.abspath(path)])
            else:
                os.startfile(target)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            if os.path.isfile(path):
                subprocess.run(["open", "-R", os.path.abspath(path)])
            else:
                subprocess.run(["open", target])
        else:
            subprocess.run(["xdg-open", target])
        return True
    except Exception as exc:  # pragma: no cover - platform dependent
        print(f"[config_library] could not open file browser: {exc}")
        return False