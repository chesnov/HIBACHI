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
from typing import Any, Dict

__all__ = ["normalise_config", "LEGACY_MODES", "UNIFIED_MODE"]

UNIFIED_MODE = "fluorescence"
LEGACY_MODES = ("fluorescence_2d",)

#: Value the step code applied when the per-scale key was absent, per legacy
#: mode. Not a preference -- a record of what each track actually did, used so
#: migration does not silently change a saved project's smoothing.
_LEGACY_SMOOTH_SIGMA = {"fluorescence": 1.3, "fluorescence_2d": 0.1}
_LEGACY_CONNECT_GAP = {"fluorescence": 1.0, "fluorescence_2d": 0.0}

_SIZE_ALIASES = ("min_size_voxels", "min_size_pixels")
_DIM_ALIASES = ("voxel_dimensions", "pixel_dimensions")


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
