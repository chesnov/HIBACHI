"""
Config migration: legacy per-rank configs to the unified schema.

Why this exists
---------------
The pipeline used to ship two configs, `fluorescence` and `fluorescence_2d`,
with keys that differed by rank. There is now one schema and one mode, because
every step infers rank from the data. Saved projects still carry the old
strings and keys, so they have to keep opening.

`normalise_config` does that translation. It is deliberately explicit about the
one case where migration cannot be purely mechanical -- see `smooth_sigma`
below -- because silently choosing a value there would change what a saved
project computes without saying so.

Translations
------------
    mode: fluorescence_2d           -> fluorescence
    voxel_dimensions / pixel_dimensions -> dimensions
    min_size_voxels / min_size_pixels   -> min_size
    execute_*_fluorescence_2d       -> execute_*_fluorescence
    scale profile rows              -> every row gains smooth_sigma and
                                       connect_max_gap_physical

The last one is the awkward one. The 3D percentile table historically omitted
those two keys, while its own absolute table and both 2D tables carried them.
The step code read them as `params.get("smooth_sigma", 1.3)` in 3D and
`..., 0.1)` in 2D -- top-level keys that do not exist in either config, so the
HARDCODED number was always what took effect. Migration therefore backfills the
value that was actually in force for the config's original mode, which keeps a
migrated project computing what it computed before, and logs each backfill.
A config already carrying the keys is left alone.
"""

import copy
import os
from typing import Any, Dict, Optional

__all__ = ["normalise_config", "normalise_mode", "LEGACY_MODES", "UNIFIED_MODE",
           "processed_dir_name", "find_processed_dir",
           "config_basename", "find_config_path"]

UNIFIED_MODE = "fluorescence"
LEGACY_MODES = ("fluorescence_2d",)

#: Value the step code applied when the per-scale key was absent, per legacy
#: mode. Not a preference -- a record of what each track actually did, used so
#: migration does not silently change a saved project's smoothing.
_LEGACY_SMOOTH_SIGMA = {"fluorescence": 1.3, "fluorescence_2d": 0.1}
_LEGACY_CONNECT_GAP = {"fluorescence": 1.0, "fluorescence_2d": 0.0}

#: Modes earlier versions wrote that are the same pipeline under another name.
_RETIRED_MODES = ("ramified", "ramified_2d")

_SIZE_ALIASES = ("min_size_voxels", "min_size_pixels")
_DIM_ALIASES = ("voxel_dimensions", "pixel_dimensions")


# --------------------------------------------------------------------------- #
# Mode strings
# --------------------------------------------------------------------------- #
def normalise_mode(mode: Any) -> str:
    """
    The unified mode for any historical mode string.

    Anything this build recognises as the fluorescence pipeline -- including the
    retired `ramified` names -- maps to `UNIFIED_MODE`. Anything else is
    returned unchanged, so an unknown mode still fails a registry lookup rather
    than being silently accepted as fluorescence.
    """
    text = str(mode or "")
    if text in (UNIFIED_MODE,) + LEGACY_MODES or text in _RETIRED_MODES:
        return UNIFIED_MODE
    return text


# --------------------------------------------------------------------------- #
# On-disk names
# --------------------------------------------------------------------------- #
# The mode string is embedded in the results directory (`<basename>_processed_
# <mode>`) and the run config filename (`processing_config_<mode>.yaml`). Those
# names exist on users' disks, so collapsing the mode cannot simply change them:
# a project processed as `fluorescence_2d` would have its results orphaned --
# still on disk, but invisible, and the project would report as unprocessed.
#
# So these resolvers prefer the unified name and fall back to a legacy one that
# actually exists. New work gets clean names; old projects keep opening; nothing
# is renamed or moved. The same shape as `metadata.find_dimensions`, on purpose.
def processed_dir_name(basename: str, mode: Any = UNIFIED_MODE) -> str:
    """The results directory name for `basename` under `mode`."""
    return f"{basename}_processed_{normalise_mode(mode)}"


def find_processed_dir(parent: str, basename: str, log=None) -> str:
    """
    Absolute results directory for `basename` in `parent`.

    Returns the unified path when it exists, otherwise an existing legacy path,
    otherwise the unified path (so callers creating a new one get the clean
    name). A fallback is logged when `log` is given: silently reading a legacy
    directory is fine, but it should be visible that it happened.
    """
    unified = os.path.join(parent, processed_dir_name(basename))
    if os.path.isdir(unified):
        return unified
    for legacy_mode in LEGACY_MODES + _RETIRED_MODES:
        candidate = os.path.join(parent,
                                 f"{basename}_processed_{legacy_mode}")
        if os.path.isdir(candidate):
            if log:
                log(f"  [ConfigMigration] using legacy results directory "
                    f"{os.path.basename(candidate)} (not renamed)")
            return candidate
    return unified


def config_basename(mode: Any = UNIFIED_MODE) -> str:
    """The run-config filename for `mode`."""
    return f"processing_config_{normalise_mode(mode)}.yaml"


def find_config_path(directory: str, log=None) -> Optional[str]:
    """
    Absolute run-config path inside `directory`, or None if there is none.

    Unified name first, then legacy. Returns None rather than a non-existent
    path so callers can tell "no config here" from "config to be created".
    """
    unified = os.path.join(directory, config_basename())
    if os.path.isfile(unified):
        return unified
    for legacy_mode in LEGACY_MODES + _RETIRED_MODES:
        candidate = os.path.join(directory,
                                 f"processing_config_{legacy_mode}.yaml")
        if os.path.isfile(candidate):
            if log:
                log(f"  [ConfigMigration] using legacy run config "
                    f"{os.path.basename(candidate)} (not renamed)")
            return candidate
    return None


def normalise_config(config: Dict[str, Any], log=print) -> Dict[str, Any]:
    """
    Return a copy of `config` in the unified schema.

    Idempotent: a config already unified passes through unchanged and silently.
    Never mutates the input.
    """
    cfg = copy.deepcopy(config)
    original_mode = str(cfg.get("mode") or UNIFIED_MODE)
    changes = []

    # ---- mode ----------------------------------------------------------
    if original_mode in LEGACY_MODES:
        cfg["mode"] = UNIFIED_MODE
        changes.append(f"mode {original_mode!r} -> {UNIFIED_MODE!r}")

    # ---- dimensions ----------------------------------------------------
    if "dimensions" not in cfg:
        for alias in _DIM_ALIASES:
            if alias in cfg:
                cfg["dimensions"] = cfg.pop(alias)
                changes.append(f"{alias} -> dimensions")
                break

    # ---- step blocks ---------------------------------------------------
    for key in list(cfg.keys()):
        block = cfg[key]
        if not (isinstance(block, dict) and key.startswith("execute_")):
            continue

        # `execute_x_fluorescence_2d` -> `execute_x_fluorescence`
        new_key = key
        if key.endswith("_2d"):
            new_key = key[:-3]
            if new_key != key:
                cfg[new_key] = cfg.pop(key)
                block = cfg[new_key]
                changes.append(f"{key} -> {new_key}")

        params = block.get("parameters") or {}

        # size key
        if "min_size" not in params:
            for alias in _SIZE_ALIASES:
                if alias in params:
                    params["min_size"] = params.pop(alias)
                    changes.append(f"{new_key}: {alias} -> min_size")
                    break
        else:
            for alias in _SIZE_ALIASES:
                if alias in params:
                    params.pop(alias)
                    changes.append(f"{new_key}: dropped redundant {alias}")

        # per-scale filter keys on the profile tables
        for tbl in ("scale_profiles_percentile", "scale_profiles_absolute"):
            entry = params.get(tbl)
            if not isinstance(entry, dict):
                continue
            rows = entry.get("value")
            if not isinstance(rows, list):
                continue
            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                for field, table in (("smooth_sigma", _LEGACY_SMOOTH_SIGMA),
                                     ("connect_max_gap_physical",
                                      _LEGACY_CONNECT_GAP)):
                    if field not in row or row[field] is None:
                        val = table.get(original_mode, table[UNIFIED_MODE])
                        row[field] = val
                        changes.append(
                            f"{new_key}: {tbl}[{idx}].{field} backfilled with "
                            f"{val} -- the value the {original_mode!r} step code "
                            f"applied when this key was absent"
                        )

        # step titles carried the rank
        if isinstance(block.get("name"), str):
            cleaned = block["name"].replace(" (2D)", "").replace(" (3D)", "")
            if cleaned != block["name"]:
                block["name"] = cleaned

    if changes:
        log(f"  [ConfigMigration] {len(changes)} change(s) applied:")
        for c in changes:
            log(f"      {c}")

    return cfg


def is_legacy(config: Dict[str, Any]) -> bool:
    """True when `config` needs migrating."""
    if str(config.get("mode") or "") in LEGACY_MODES:
        return True
    if "dimensions" not in config and any(a in config for a in _DIM_ALIASES):
        return True
    for key, block in config.items():
        if key.startswith("execute_") and isinstance(block, dict):
            if key.endswith("_2d"):
                return True
            if any(a in (block.get("parameters") or {}) for a in _SIZE_ALIASES):
                return True
    return False
