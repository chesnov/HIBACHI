"""Joint parameter optimization across several images (initial segmentation).

Problem
-------
Each image can be tuned to its own optimal config, but those per-image optima do
not transfer: image A likes scale-1.0 ``low`` = 0.013, image B likes 0.02, and a
single shared config has to pick one value. The naive answer -- average the
per-image optima -- is wrong, because it ignores how *sensitive* each image is to
the parameter. If A degrades sharply away from 0.013 while B barely notices
around 0.02, the fair shared value sits closer to 0.013.

Method (object-integrity, safe side of every cliff)
---------------------------------------------------
What matters most is not pixel overlap but whether every real cell still
segments and no spurious object appears. Pixel-overlap metrics (Dice) miss this:
losing a whole small cell barely moves Dice, and a lost cell can even be masked
by a gained artifact of similar size. So the objective is object-level: compared
to each image's own reference segmentation, count missed cells (a reference
object no candidate covers) and spurious objects (a candidate object matching no
reference), which is weighted far above the boundary (Dice) term.

Cell appearance/disappearance is a discrete threshold-crossing -- a cliff, not a
smooth bowl -- so instead of estimating curvature we find, per image and per
parameter, the SAFE INTERVAL: the range around that image's optimum within which
no cell is lost and no artifact appears. We locate each cliff by scanning
outward from the optimum to bracket it, then bisecting to localize it. The
shared value is then placed in the INTERSECTION of all images' safe intervals --
in the middle, for maximum margin to every cliff, so the shared config sits on
the safe side for every image at once. If the safe intervals do not overlap (the
images genuinely want different settings), we fall back to the value with the
smallest worst-image object loss (minimax), and flag it.

Parameters are treated one at a time (each swept with the others held at each
image's optimum); genuine coupling is out of scope here. Evaluation runs the real
segmentation on the whole image (crops de-calibrate the intensity normalization
that absolute thresholds depend on), so it is compute-heavy but faithful.

The core (`optimize_initial_segmentation`) is GUI-agnostic: it takes progress and
cancel callbacks. `OptimizeWorker` / `run_optimization_dialog` wrap it for the
PyQt5 project view. The 2D and 3D pipelines are driven through the same code path
(only the segment entry point differs by mode), so behaviour is at parity.
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import yaml


class OptimizationCancelled(Exception):
    """Raised when the user cancels a running optimization."""


class OptimizationError(Exception):
    """Raised for unrecoverable setup problems (bad inputs, missing files)."""


# Fields inside a scale-profile row that are worth optimizing, plus the two
# top-level scalars. Everything else is left exactly as the template has it.
_ROW_FIELDS = ("low", "high", "smooth_sigma", "connect_max_gap_physical")
_SCALAR_FIELDS = ("min_size", "trace_max_gap")
_SCALAR_KEY = "__scalar__"

# Relative perturbation used to probe local curvature, with per-kind floors so a
# tiny value (e.g. an absolute threshold of 0.01) still gets a meaningful step.
_REL_STEP = 0.25
_STEP_FLOOR = {
    "low": 2e-3, "high": 2e-3, "smooth_sigma": 0.05,
    "connect_max_gap_physical": 0.25, "min_size": 5.0, "trace_max_gap": 0.5,
}
# A deviation below this is treated as "the mask did not really change", so the
# probe grows the perturbation (thresholding has flat plateaus; too small a step
# yields an identical mask and a misleading zero curvature).
_MEASURABLE = 5e-3
_PROBE_MULTS = (1.0, 3.0, 9.0)   # grow the step until the mask responds
_MIN_REF_PIXELS = 20             # a crop whose reference mask is ~empty is unusable
# Run the probe on the whole image when it fits in this budget; only crop
# (foreground-dense) above it. Kept large so a crop, if ever needed, stays close
# to the whole-image intensity normalization the absolute thresholds rely on.
_MAX_2D_PIXELS = 12_000_000      # ~3460 x 3460
_MAX_3D_VOXELS = 12_000_000
# Deviation blends object-level disagreement (lost cells / spurious artifacts)
# with pixel-level Dice. Object error is weighted higher: whether a cell
# segments at all matters more than its exact pixel extent.
_W_OBJ = 1.0
_W_PIX = 0.05
_MATCH_FRAC = 0.25               # overlap needed to call an object matched
_GRID_POINTS = 7                 # nominal scan budget per parameter (progress est.)
_SCAN_MAXOUT = 5                 # geometric outward steps when bracketing a cliff
_BISECT_ITERS = 3                # bisection steps to localize a cliff


# --------------------------------------------------------------------------- #
# Config / image plumbing
# --------------------------------------------------------------------------- #
def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _find_config_and_image(folder: str) -> Tuple[str, str]:
    """Return (config_yaml_path, image_tif_path) for a project image folder."""
    tif = cfg = None
    for f in sorted(os.listdir(folder)):
        low = f.lower()
        if tif is None and low.endswith((".tif", ".tiff")):
            tif = os.path.join(folder, f)
        elif cfg is None and low.endswith((".yaml", ".yml")):
            cfg = os.path.join(folder, f)
    if not tif or not cfg:
        raise OptimizationError(
            f"'{os.path.basename(folder)}' must contain exactly one image and "
            "one config; could not find both."
        )
    return cfg, tif


def _resolve_config(folder: str, mode: str) -> Tuple[Dict[str, Any], str, str]:
    """Return (config, source, tif_path) for an image folder.

    Prefers the PROCESSED run config -- ``<basename>_processed_<mode>/
    processing_config_<mode>.yaml`` -- because that is where the values the user
    actively tuned in the GUI are persisted (the widgets write straight into its
    ``parameters[*].value`` fields on save). The plain folder ``.yaml`` is only
    the initial template that was applied to the project, so reconciling it would
    combine untouched defaults, not the tuned settings. Falls back to the
    template only when no processed run exists. ``source`` is "tuned" or
    "template".
    """
    from ..fluorescence_module.config_migration import (
        find_config_path, find_processed_dir)

    template_path, tif = _find_config_and_image(folder)
    basename = os.path.splitext(os.path.basename(tif))[0]
    # Both the directory and the filename carried the mode, so both need the
    # legacy fallback: a project tuned before the modes merged keeps its
    # `..._processed_fluorescence_2d/processing_config_fluorescence_2d.yaml`,
    # and building only the unified names would silently read the untouched
    # template instead of the values the user actually tuned.
    processed = find_config_path(find_processed_dir(folder, basename)) or ""
    if processed and os.path.isfile(processed):
        return _load_yaml(processed), "tuned", tif
    return _load_yaml(template_path), "template", tif


def _load_image(path: str) -> np.ndarray:
    try:
        import tifffile
        return np.asarray(tifffile.imread(path))
    except Exception:
        from skimage.io import imread  # fallback
        return np.asarray(imread(path))


def _raw_seg_step_key(config: Dict[str, Any]) -> str:
    """The config key holding the raw-segmentation step (mode-independent)."""
    for k in config:
        if isinstance(k, str) and k.startswith("execute_raw_segmentation"):
            return k
    raise OptimizationError(
        "No 'execute_raw_segmentation...' step found in the config."
    )


def _spacing_from_config(config: Dict[str, Any], shape: Tuple[int, ...],
                         is_2d: bool) -> Tuple[float, ...]:
    """Per-pixel spacing = physical extent / shape (matches gui_manager)."""
    from .metadata import find_dimensions
    _dim_key, dim = find_dimensions(config)
    dim = dim or {}
    try:
        tx = float(dim.get("x", 1.0)); ty = float(dim.get("y", 1.0))
        tz = float(dim.get("z", 1.0))
    except (ValueError, TypeError):
        tx = ty = tz = 1.0
    if len(shape) == 2:
        ys = ty / shape[0] if shape[0] else 1.0
        xs = tx / shape[1] if shape[1] else 1.0
        return (1.0, ys, xs)
    if len(shape) == 3:
        zs = tz / shape[0] if shape[0] else 1.0
        ys = ty / shape[1] if shape[1] else 1.0
        xs = tx / shape[2] if shape[2] else 1.0
        return (zs, ys, xs)
    return (1.0, 1.0, 1.0)


def _active_profiles(params: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return (table_key, profiles_list) for the threshold table in force."""
    is_absolute = bool(params.get("use_absolute_thresholds", False))
    if is_absolute and "scale_profiles_absolute" in params:
        return "scale_profiles_absolute", params["scale_profiles_absolute"]
    if not is_absolute and "scale_profiles_percentile" in params:
        return "scale_profiles_percentile", params["scale_profiles_percentile"]
    if "scale_profiles" in params:
        return "scale_profiles", params["scale_profiles"]
    raise OptimizationError("Config has no scale-profile table to optimize.")


def _flatten_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve a step's parameter block to plain values.

    Saved configs use the rich schema (each param is ``{label, type, value}``,
    like default.yaml); the pipeline resolves that to bare values before it runs
    a step. Do the same here: unwrap any ``{... 'value': X}`` entry to X, and
    leave genuine mapping/list values (e.g. pixel_dimensions, the scale table's
    row list) untouched. Works whether the input is already flat or rich.
    """
    flat: Dict[str, Any] = {}
    for key, val in (raw or {}).items():
        if isinstance(val, dict) and "value" in val:
            flat[key] = val["value"]
        else:
            flat[key] = val
    return flat


def _write_leaf_into_config(rich_params: Dict[str, Any], leaf: Tuple[Any, str],
                            value: float, table_key: str) -> None:
    """Write an optimized leaf back into a rich-schema parameter block, in
    place, preserving the ``{label, type, value}`` wrappers so the output config
    stays a valid template. Handles both rich and already-flat blocks."""
    scale, field = leaf

    def _rows(entry):
        if isinstance(entry, dict) and "value" in entry:
            return entry["value"]
        return entry

    if scale == _SCALAR_KEY:
        out = int(round(value)) if field == "min_size" else float(value)
        entry = rich_params.get(field)
        if isinstance(entry, dict) and "value" in entry:
            entry["value"] = out
        else:
            rich_params[field] = out
        return

    rows = _rows(rich_params.get(table_key))
    if not isinstance(rows, list):
        return
    for row in rows:
        if isinstance(row, dict) and abs(float(row.get("scale", 1e18)) - float(scale)) < 1e-9:
            row[field] = float(value)
            return


def _params_to_kwargs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror _2D_strategy.execute_raw_segmentation's parameter parsing."""
    _tbl, profiles = _active_profiles(params)
    is_absolute = bool(params.get("use_absolute_thresholds", False))
    base_smooth = float(params.get("smooth_sigma", 0.1))
    base_gap = float(params.get("connect_max_gap_physical", 0.0))
    return dict(
        threshold_mode="Absolute" if is_absolute else "Percentile",
        tubular_scales=[p["scale"] for p in profiles],
        low_threshold_percentile=[p["low"] for p in profiles],
        high_threshold_percentile=[p["high"] for p in profiles],
        smooth_sigma=[float(p.get("smooth_sigma", base_smooth)) for p in profiles],
        connect_max_gap_physical=[
            float(p.get("connect_max_gap_physical", base_gap)) for p in profiles],
        min_size=int(params.get("min_size", 200)),
        trace_max_gap=float(params.get("trace_max_gap", 0.0)),
    )


# --------------------------------------------------------------------------- #
# Segmentation adapter (mode-dispatched, at parity)
# --------------------------------------------------------------------------- #
def _segment_labels(image: np.ndarray, spacing: Tuple[float, ...],
                    params: Dict[str, Any], is_2d: bool, temp_root: str) -> np.ndarray:
    """Run raw segmentation on `image` with `params`; return the int32 LABEL
    image (object identities preserved, not just a binary mask).

    Reads the label memmap into RAM (the run region is bounded) and removes all
    temp output so repeated probing does not accumulate on disk. `temp_root` is
    the project image folder: all scratch lives inside the project directory
    (never the OS temp dir) and is cleaned up here per call.
    """
    kw = _params_to_kwargs(params)
    run_tmp = os.path.join(temp_root, f"hibachi_opt_{uuid.uuid4().hex}")
    os.makedirs(run_tmp, exist_ok=True)
    labels_dir = None
    try:
        # One segmenter for both ranks: it infers rank from `volume`. The two
        # former branches differed only in argument spelling (image/volume,
        # min_size_pixels/min_size_voxels) and converged on the same 4-tuple,
        # so there is nothing left to dispatch on.
        from ..fluorescence_module.initial_segmentation import (
            segment_cells_first_pass_raw as seg)
        result = seg(
            volume=image, spacing=spacing,
            tubular_scales=kw["tubular_scales"],
            smooth_sigma=kw["smooth_sigma"],
            connect_max_gap_physical=kw["connect_max_gap_physical"],
            min_size=kw["min_size"],
            low_threshold_percentile=kw["low_threshold_percentile"],
            high_threshold_percentile=kw["high_threshold_percentile"],
            threshold_mode=kw["threshold_mode"],
            trace_max_gap=kw["trace_max_gap"],
            temp_root_path=run_tmp,
        )
        dat_path, labels_dir, _thr, _ = result
        if not dat_path or not os.path.exists(dat_path):
            raise OptimizationError("Segmentation produced no output.")
        mm = np.memmap(dat_path, dtype=np.int32, mode="r", shape=image.shape)
        labels = np.array(mm, dtype=np.int32)   # copy so the memmap can be freed
        del mm
        return labels
    finally:
        if labels_dir and os.path.isdir(labels_dir):
            shutil.rmtree(labels_dir, ignore_errors=True)
        shutil.rmtree(run_tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Metric + crop selection
# --------------------------------------------------------------------------- #
def _dice_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - Dice(a, b) on boolean masks; 0 = identical, 1 = disjoint."""
    a = a.astype(bool); b = b.astype(bool)
    tot = int(a.sum()) + int(b.sum())
    if tot == 0:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    return 1.0 - (2.0 * inter) / tot


def _object_error(ref: np.ndarray, cand: np.ndarray) -> Tuple[float, int, int, int]:
    """Object-level disagreement between two LABEL images.

    A reference object counts as *missed* (a lost cell) if less than
    `_MATCH_FRAC` of its pixels are covered by any candidate object; a candidate
    object counts as *spurious* (a noise artifact) if less than `_MATCH_FRAC` of
    its pixels overlap any reference object. Returns
    (error, missed, spurious, n_ref) where error = (missed + spurious) / n_ref.

    Unlike Dice, this does not scale with object area, so dropping one whole
    small cell or gaining one small artifact registers as a full unit of error.
    """
    rf = ref.ravel(); cf = cand.ravel()
    ref_fg = (rf > 0); cand_fg = (cf > 0)
    n_ref = int(rf.max()) if rf.size else 0
    n_cand = int(cf.max()) if cf.size else 0

    missed = 0
    if n_ref > 0:
        size = np.bincount(rf, minlength=n_ref + 1).astype(float)
        cov = np.bincount(rf, weights=cand_fg.astype(float), minlength=n_ref + 1)
        frac = np.divide(cov, size, out=np.zeros_like(cov), where=size > 0)
        # count only real objects (size > 0); label-id gaps are not objects
        missed = int(np.sum((frac[1:] < _MATCH_FRAC) & (size[1:] > 0)))

    spurious = 0
    if n_cand > 0:
        size_c = np.bincount(cf, minlength=n_cand + 1).astype(float)
        cov_c = np.bincount(cf, weights=ref_fg.astype(float), minlength=n_cand + 1)
        frac_c = np.divide(cov_c, size_c, out=np.zeros_like(cov_c), where=size_c > 0)
        spurious = int(np.sum((frac_c[1:] < _MATCH_FRAC) & (size_c[1:] > 0)))

    n_present = int(np.count_nonzero(np.bincount(rf)[1:])) if n_ref > 0 else 0
    denom = max(1, n_present)
    return (missed + spurious) / denom, missed, spurious, n_present


def _deviation(ref: np.ndarray, cand: np.ndarray) -> Tuple[float, int, int]:
    """Blended deviation of a candidate LABEL image from the reference:
    object-level disagreement (lost cells + artifacts) plus pixel-level Dice.
    Object error is weighted higher because *whether a cell segments at all*
    matters more than exactly which pixels it occupies. Returns
    (deviation, missed, spurious)."""
    oe, missed, spurious, _n = _object_error(ref, cand)
    pix = _dice_distance(ref > 0, cand > 0)
    return _W_OBJ * oe + _W_PIX * pix, missed, spurious


def _pick_crop(image: np.ndarray, is_2d: bool) -> Tuple[slice, ...]:
    """Choose the region to run the probe segmentations on.

    Default to the WHOLE image. The segmenter normalizes intensity internally,
    and absolute thresholds are calibrated against that whole-image
    normalization, so cropping would change the normalization statistics and
    de-calibrate the thresholds (a crop can segment to nothing even though the
    full image is fine). Only fall back to a crop for images too large to run
    repeatedly, and then pick a FOREGROUND-dense window (not merely bright) and
    keep it large so the normalization stays close to the full image's.
    """
    budget = _MAX_2D_PIXELS if is_2d else _MAX_3D_VOXELS
    if image.size <= budget:
        return tuple(slice(0, s) for s in image.shape)

    # Oversized: size a window to the budget, biased to keep full Z in 3D.
    if is_2d:
        side = int(np.sqrt(budget))
        win = (min(image.shape[0], side), min(image.shape[1], side))
    else:
        z = min(image.shape[0], 48)
        side = int(np.sqrt(max(1, budget // max(1, z))))
        win = (z, min(image.shape[1], side), min(image.shape[2], side))
    if all(w >= s for w, s in zip(win, image.shape)):
        return tuple(slice(0, s) for s in image.shape)

    # Score strided window origins by FOREGROUND count (image above a high
    # percentile), so the crop locks onto structures rather than a bright edge.
    img = image.astype(np.float32)
    thr = float(np.percentile(img, 92.0))
    fg = (img > thr)
    best_origin = tuple(0 for _ in image.shape)
    best_score = -1.0
    steps = [max(1, (s - w) // 6) for s, w in zip(image.shape, win)]
    ranges = [range(0, s - w + 1, st) for s, w, st in zip(image.shape, win, steps)]

    def _iter_origins(rs):
        if len(rs) == 2:
            for y in rs[0]:
                for x in rs[1]:
                    yield (y, x)
        else:
            for z0 in rs[0]:
                for y in rs[1]:
                    for x in rs[2]:
                        yield (z0, y, x)

    for origin in _iter_origins(ranges):
        sl = tuple(slice(o, o + w) for o, w in zip(origin, win))
        score = float(fg[sl].sum())
        if score > best_score:
            best_score, best_origin = score, origin
    return tuple(slice(o, o + w) for o, w in zip(best_origin, win))


# --------------------------------------------------------------------------- #
# Leaf handling
# --------------------------------------------------------------------------- #
def _leaf_value(params: Dict[str, Any], leaf: Tuple[Any, str]) -> Optional[float]:
    """Current value of a leaf, or None if this config lacks it."""
    scale, field = leaf
    if scale == _SCALAR_KEY:
        if field == "min_size":
            return float(params.get("min_size", 200))
        if field == "trace_max_gap":
            return float(params.get("trace_max_gap", 0.0))
        return None
    _tbl, profiles = _active_profiles(params)
    for row in profiles:
        if abs(float(row.get("scale", 1e18)) - float(scale)) < 1e-9:
            if field in row:
                return float(row[field])
            # low/high are always present; smooth/gap may inherit the base scalar
            base = {"smooth_sigma": params.get("smooth_sigma", 0.1),
                    "connect_max_gap_physical": params.get("connect_max_gap_physical", 0.0)}
            return float(base.get(field)) if field in base else None
    return None


def _set_leaf(params: Dict[str, Any], leaf: Tuple[Any, str], value: float) -> None:
    """Write a leaf value into a params dict in place."""
    scale, field = leaf
    if scale == _SCALAR_KEY:
        params[field] = int(round(value)) if field == "min_size" else float(value)
        return
    _tbl, profiles = _active_profiles(params)
    for row in profiles:
        if abs(float(row.get("scale", 1e18)) - float(scale)) < 1e-9:
            row[field] = float(value)
            return


def _collect_leaves(per_image_params: List[Dict[str, Any]]) -> List[Tuple[Any, str]]:
    """Every (scale, field) / scalar leaf that at least one image exposes."""
    leaves: List[Tuple[Any, str]] = []
    seen = set()
    for params in per_image_params:
        try:
            _tbl, profiles = _active_profiles(params)
        except OptimizationError:
            profiles = []
        for row in profiles:
            scale = float(row.get("scale", 1.0))
            for field in _ROW_FIELDS:
                if _leaf_value(params, (scale, field)) is None:
                    continue
                key = (round(scale, 6), field)
                if key not in seen:
                    seen.add(key); leaves.append((scale, field))
    for field in _SCALAR_FIELDS:
        key = (_SCALAR_KEY, field)
        if key not in seen:
            seen.add(key); leaves.append((_SCALAR_KEY, field))
    return leaves


def _perturbation(field: str, value: float) -> float:
    return max(_REL_STEP * abs(value), _STEP_FLOOR.get(field, abs(value) * _REL_STEP or 1e-3))


# --------------------------------------------------------------------------- #
# Core optimization
# --------------------------------------------------------------------------- #
def _bias_label(bias: float) -> str:
    if bias >= 0.9:
        return "never delete cells"
    if bias > 0.6:
        return "lean: keep cells"
    if bias >= 0.4:
        return "balanced"
    if bias > 0.1:
        return "lean: avoid noise"
    return "never add cells"


def optimize_initial_segmentation(
    folders: List[str],
    mode: str,
    bias: float = 0.5,
    progress: Optional[Callable[[float, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Find one shared raw-segmentation config that minimizes deviation from
    each image's own desired result.

    `bias` in [0, 1] sets the preference used only when a value must trade a lost
    cell against a spurious one (a genuine conflict): 0.5 weighs them equally,
    1.0 = "never delete cells" (avoid missed cells at the cost of tolerating
    noise), 0.0 = "never add cells" (avoid noise at the cost of dropping cells).
    It does not affect parameters that have a conflict-free (safe) value.

    Returns (merged_config_dict, report). Raises OptimizationCancelled if the
    user cancels, or OptimizationError on bad input.
    """
    bias = float(min(1.0, max(0.0, bias)))
    w_miss = bias            # penalty per lost real cell
    w_fp = 1.0 - bias        # penalty per spurious object (noise)
    bias_txt = _bias_label(bias)

    def _tick(frac: float, msg: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, frac)), msg)

    def _check() -> None:
        if is_cancelled is not None and is_cancelled():
            raise OptimizationCancelled()

    if len(folders) < 2:
        raise OptimizationError("Select at least two images to optimize across.")

    # Rank is decided per image, from the image, once it is loaded. Deriving it
    # from the mode string cannot work with a single mode -- the test would be
    # False for every project, so a 2D image would get the 3D crop budget and a
    # 3-axis spacing. `config_ndim` is used as a cross-check only: the array's
    # own ndim is the authority here, since that is what the segmenter sees.
    from .metadata import config_ndim

    # ---- Load every image, config, crop, and per-image optimum -------------- #
    _tick(0.0, "Loading images and configs…")
    images, spacings, params_list, step_keys, base_configs = [], [], [], [], []
    template_only: List[str] = []
    for k, folder in enumerate(folders):
        _check()
        config, source, tif_path = _resolve_config(folder, mode)
        if source == "template":
            template_only.append(os.path.basename(folder))
        step_key = _raw_seg_step_key(config)
        params = _flatten_params(config[step_key].get("parameters", config[step_key]))
        image = _load_image(tif_path)
        # A 2D acquisition can arrive with singleton axes; squeeze first so the
        # rank below is the real one.
        if config_ndim(config) == 2 and image.ndim != 2:
            image = np.squeeze(image)
        is_2d = (image.ndim == 2)
        crop = _pick_crop(image, is_2d)
        image = np.ascontiguousarray(image[crop])
        spacing = _spacing_from_config(config, image.shape, is_2d)
        images.append(image); spacings.append(spacing)
        params_list.append(copy.deepcopy(params)); step_keys.append(step_key)
        base_configs.append(config)
        _tick(0.05 * (k + 1) / len(folders),
              f"Loaded {os.path.basename(folder)} [{source}] (region {tuple(image.shape)})")

    leaves = _collect_leaves(params_list)

    # Only optimize leaves whose per-image optimal values actually disagree;
    # anything the images already agree on is kept verbatim.
    optimizable: List[Tuple[Any, str]] = []
    per_image_vals: Dict[Tuple[Any, str], List[Optional[float]]] = {}
    for leaf in leaves:
        vals = [_leaf_value(p, leaf) for p in params_list]
        per_image_vals[leaf] = vals
        present = [v for v in vals if v is not None]
        if len(present) >= 2 and (max(present) - min(present)) > 1e-9:
            optimizable.append(leaf)

    if not optimizable:
        raise OptimizationError(
            "The selected images already share identical initial-segmentation "
            "parameters — there is nothing to reconcile."
        )

    # ---- Reference segmentation per image (its own desired result) ---------- #
    refs: List[Optional[np.ndarray]] = [None] * len(images)
    empty_refs: List[str] = []
    for i, (image, spacing, params) in enumerate(zip(images, spacings, params_list)):
        _check()
        name = os.path.basename(folders[i])
        _tick(0.05 + 0.10 * (i + 1) / len(images),
              f"[{name}] reference segmentation…")
        ref = _segment_labels(image, spacing, params, is_2d, folders[i])
        if int((ref > 0).sum()) < _MIN_REF_PIXELS:
            empty_refs.append(name)
            print(f"[optimize] {name}: reference nearly empty "
                  f"({int((ref > 0).sum())} px) — skipping this image.")
        else:
            refs[i] = ref

    _check()
    if all(r is None for r in refs):
        raise OptimizationError(
            "Every image's reference segmentation came out essentially empty, so "
            "there is nothing to measure. The images ran on their own optimal "
            "configs, so this usually means the saved config's absolute "
            "thresholds don't reproduce a mask here (e.g. a different pipeline "
            "version) or min-size removed everything. Re-process one image and "
            "confirm its raw segmentation is non-empty before optimizing."
        )
    active = [i for i in range(len(images)) if refs[i] is not None]

    done = 0
    total_scan = max(1, len(optimizable) * len(active) * _GRID_POINTS)

    def _eval(i: int, leaf, x: float):
        """Segment image i with leaf set to x (others at its optimum) and score
        against its own reference. Returns (missed_cells, spurious, dice) or
        None on failure."""
        nonlocal done
        done += 1
        trial = copy.deepcopy(params_list[i])
        _set_leaf(trial, leaf, x)
        try:
            lab = _segment_labels(images[i], spacings[i], trial, is_2d, folders[i])
        except OptimizationCancelled:
            raise
        except Exception as exc:
            print(f"[optimize] eval {leaf}={x:g} on {os.path.basename(folders[i])} "
                  f"failed: {exc}")
            return None
        _oe, miss, spur, _n = _object_error(refs[i], lab)
        return miss, spur, _dice_distance(refs[i] > 0, lab > 0)

    # ---- Per-parameter: safe interval per image, then the safe-side choice --- #
    merged_config = copy.deepcopy(base_configs[0])
    _merged_step = merged_config[step_keys[0]]
    merged_params = _merged_step.get("parameters", _merged_step)
    table_key, _ = _active_profiles(params_list[0])
    report_values: Dict[str, Dict[str, Any]] = {}
    n_conflicts = 0

    _NONNEG = ("low", "high", "smooth_sigma", "connect_max_gap_physical",
               "min_size", "trace_max_gap")

    for leaf in optimizable:
        _check()
        scale, field = leaf
        label = field if scale == _SCALAR_KEY else f"scale{scale:g}.{field}"
        opt = {i: per_image_vals[leaf][i] for i in active
               if per_image_vals[leaf][i] is not None}
        per_image = [None if per_image_vals[leaf][i] is None
                     else round(per_image_vals[leaf][i], 6)
                     for i in range(len(images))]

        if len(opt) < 2:
            value = float(next(iter(opt.values()))) if opt else 0.0
            _write_leaf_into_config(merged_params, leaf, value, table_key)
            report_values[label] = {
                "per_image": per_image,
                "shared": int(round(value)) if field == "min_size" else round(value, 6),
                "method": "single image (kept as-is)"}
            continue

        step = _perturbation(field, float(np.median(list(opt.values()))))
        floor = (1.0 if field == "min_size" else 0.0) if field in _NONNEG else None

        def _clamp(x: float) -> float:
            return x if floor is None else max(floor, x)

        evals: Dict[int, Dict[float, Optional[Tuple[int, int, float]]]] = {i: {} for i in opt}

        def _ev(i: int, x: float):
            x = round(_clamp(x), 9)
            if x in evals[i]:
                return evals[i][x]
            _check()
            _tick(0.15 + 0.8 * done / total_scan,
                  f"[{os.path.basename(folders[i])}] scanning {label}…")
            r = _eval(i, leaf, x)
            evals[i][x] = r
            return r

        def _cliff(i: int, v0: float, direction: int) -> float:
            """Furthest value from v0 (in `direction`) that keeps image i's cells
            intact and adds no artifact. Geometric outward search to bracket the
            cliff, then bisection to localize it."""
            x_safe = v0
            x_unsafe = None
            for k in range(_SCAN_MAXOUT):
                x = _clamp(v0 + direction * step * (2 ** k))
                r = _ev(i, x)
                if r is not None and r[0] == 0 and r[1] == 0:
                    x_safe = x
                    if x == floor:            # hit the physical bound; can't go on
                        return x_safe
                else:
                    x_unsafe = x
                    break
            if x_unsafe is None:              # never cliffed within reach
                return x_safe
            for _ in range(_BISECT_ITERS):    # localize the cliff between safe/unsafe
                mid = 0.5 * (x_safe + x_unsafe)
                r = _ev(i, mid)
                if r is not None and r[0] == 0 and r[1] == 0:
                    x_safe = mid
                else:
                    x_unsafe = mid
            return x_safe

        safe: Dict[int, Tuple[float, float]] = {}
        for i, v0 in opt.items():
            evals[i][round(v0, 9)] = (0, 0, 0.0)   # optimum safe by construction
            safe[i] = (_cliff(i, v0, -1), _cliff(i, v0, +1))

        L = max(s[0] for s in safe.values())
        H = min(s[1] for s in safe.values())
        if L <= H + 1e-12:
            # A window where every image keeps all its cells and gains no
            # artifacts. Default to the consensus of the images' tuned values and
            # stay there -- moving off it is only justified to gain margin from a
            # real cliff. If the parameter is insensitive the window balloons, so
            # its midpoint is meaningless; the consensus keeps us near what the
            # images actually used. Only when the consensus falls OUTSIDE a
            # genuinely narrow window do we retreat to the midpoint (max margin,
            # so we don't sit on a cliff edge).
            consensus = float(np.median(list(opt.values())))
            if L - 1e-12 <= consensus <= H + 1e-12:
                value = consensus
            else:
                value = 0.5 * (L + H)
            how = "safe (all cells kept in every image)"
        else:
            # No shared-safe window: a lost cell must be traded against a
            # spurious one somewhere. Weigh them by `bias` (worst image first),
            # then minimize the weighted total, then boundaries. Sample the gap
            # between the images' safe intervals.
            n_conflicts += 1
            gap_lo = min(s[1] for s in safe.values())
            gap_hi = max(s[0] for s in safe.values())
            cands = list(np.linspace(gap_lo, gap_hi, 5)) + list(opt.values())
            best_key, best_x, best_ms = None, float(np.median(list(opt.values()))), (0, 0)
            for x in sorted(set(round(_clamp(float(c)), 9) for c in cands)):
                worst = 0.0
                tmiss = tspur = 0
                dsum = 0.0
                ok = True
                for i in opt:
                    r = _ev(i, x)
                    if r is None:
                        ok = False
                        break
                    miss, spur, dc = r
                    worst = max(worst, w_miss * miss + w_fp * spur)
                    tmiss += miss
                    tspur += spur
                    dsum += dc
                if not ok:
                    continue
                key = (round(worst, 6), round(w_miss * tmiss + w_fp * tspur, 6),
                       tmiss + tspur, round(dsum, 4))
                if best_key is None or key < best_key:
                    best_key, best_x, best_ms = key, x, (tmiss, tspur)
            value = best_x
            how = (f"conflict [{bias_txt}]: {best_ms[0]} cell(s) lost, "
                   f"{best_ms[1]} spurious")

        value = _clamp(float(value))
        _write_leaf_into_config(merged_params, leaf, value, table_key)
        report_values[label] = {
            "per_image": per_image,
            "shared": int(round(value)) if field == "min_size" else round(value, 6),
            "method": how,
            "safe_intervals": {
                os.path.basename(folders[i]): [round(safe[i][0], 4),
                                               round(safe[i][1], 4)] for i in opt},
        }

    report = {
        "images": [os.path.basename(f) for f in folders],
        "optimized": report_values,
        "n_optimized": len(optimizable),
        "n_conflicts": n_conflicts,
        "empty_refs": empty_refs,
        "template_only": template_only,
        "bias": round(bias, 3),
        "bias_label": bias_txt,
    }
    if n_conflicts:
        report["warning"] = (
            f"{n_conflicts} parameter(s) had no value that keeps every cell in "
            "every image at once — the images genuinely want different settings "
            "there. The least-bad value (fewest cells lost in the worst image) "
            "was chosen; see each parameter's method line."
        )
    _tick(1.0, "Done.")
    return merged_config, report


# --------------------------------------------------------------------------- #
# PyQt5 worker + dialog
# --------------------------------------------------------------------------- #
try:
    from PyQt5.QtCore import QThread, pyqtSignal, Qt
    from PyQt5.QtWidgets import (
        QProgressDialog, QInputDialog, QFileDialog, QMessageBox,
        QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QDialogButtonBox,
    )
    _HAVE_QT = True
except Exception:  # headless / import-time safety
    _HAVE_QT = False


if _HAVE_QT:

    class OptimizeWorker(QThread):
        """Runs the optimization off the GUI thread."""
        progress = pyqtSignal(float, str)
        finished_ok = pyqtSignal(dict, dict)   # merged_config, report
        failed = pyqtSignal(str)
        cancelled = pyqtSignal()

        def __init__(self, folders: List[str], mode: str, bias: float = 0.5, parent=None):
            super().__init__(parent)
            self._folders = list(folders)
            self._mode = mode
            self._bias = float(bias)
            self._cancel = False

        def cancel(self) -> None:
            self._cancel = True

        def run(self) -> None:
            try:
                merged, report = optimize_initial_segmentation(
                    self._folders, self._mode, bias=self._bias,
                    progress=lambda f, m: self.progress.emit(f, m),
                    is_cancelled=lambda: self._cancel,
                )
            except OptimizationCancelled:
                self.cancelled.emit()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.failed.emit(str(exc))
            else:
                self.finished_ok.emit(merged, report)

    def _ask_bias(parent) -> Optional[float]:
        """Slider preference for the missed-vs-spurious trade-off. Returns a
        bias in [0, 1] (0 = never add cells, 1 = never delete cells) or None if
        cancelled."""
        dlg = QDialog(parent)
        dlg.setWindowTitle("Optimization preference")
        dlg.setMinimumWidth(460)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel(
            "When no single setting can keep every real cell AND avoid every\n"
            "noise artifact, which mistake should the optimizer avoid more?\n"
            "(This only affects parameters where the images genuinely conflict.)"))
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(50)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(10)
        lay.addWidget(slider)
        row = QHBoxLayout()
        row.addWidget(QLabel("Never add cells\n(avoid noise)"))
        row.addStretch(1)
        row.addWidget(QLabel("Equal"))
        row.addStretch(1)
        row.addWidget(QLabel("Never delete cells\n(keep all)"))
        lay.addLayout(row)
        live = QLabel("")
        live.setAlignment(Qt.AlignCenter)
        lay.addWidget(live)

        def _upd(v):
            t = v / 100.0
            live.setText(f"keep-cell weight {t:.2f}   ·   avoid-noise weight {1 - t:.2f}"
                         f"   —   {_bias_label(t)}")
        slider.valueChanged.connect(_upd)
        _upd(50)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        lay.addWidget(bb)
        if dlg.exec_() != QDialog.Accepted:
            return None
        return slider.value() / 100.0

    def run_optimization_dialog(parent, folders: List[str], mode: str) -> None:
        """Ask the missed-vs-spurious preference, then show a cancellable
        progress dialog, run the optimization, and on success prompt for a name +
        save location and write the merged config."""
        bias = _ask_bias(parent)
        if bias is None:
            return

        dlg = QProgressDialog("Preparing…", "Cancel", 0, 100, parent)
        dlg.setWindowTitle("Optimize Initial-Segmentation Parameters")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumWidth(460)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)

        worker = OptimizeWorker(folders, mode, bias, parent)

        def _on_progress(frac: float, msg: str) -> None:
            dlg.setValue(int(frac * 100))
            dlg.setLabelText(msg)

        def _on_cancel_requested() -> None:
            dlg.setLabelText("Cancelling…")
            worker.cancel()

        def _finish() -> None:
            dlg.close()

        def _on_ok(merged: Dict[str, Any], report: Dict[str, Any]) -> None:
            _finish()
            _present_and_save(parent, merged, report, mode)

        def _on_failed(msg: str) -> None:
            _finish()
            QMessageBox.critical(parent, "Optimization failed", msg)

        def _on_cancelled() -> None:
            _finish()
            QMessageBox.information(parent, "Cancelled",
                                    "Parameter optimization was cancelled.")

        worker.progress.connect(_on_progress)
        worker.finished_ok.connect(_on_ok)
        worker.failed.connect(_on_failed)
        worker.cancelled.connect(_on_cancelled)
        dlg.canceled.connect(_on_cancel_requested)

        parent._optimize_worker = worker  # keep alive
        worker.start()
        dlg.show()

    def _summary_text(report: Dict[str, Any]) -> str:
        lines = [f"Reconciled {report['n_optimized']} parameter(s) across "
                 f"{len(report['images'])} images "
                 f"(preference: {report.get('bias_label', 'balanced')}):\n"]
        for label, info in report["optimized"].items():
            per = ", ".join("—" if v is None else f"{v:g}" for v in info["per_image"])
            lines.append(f"  • {label}: [{per}] → {info['shared']}  ({info['method']})")
        if report.get("empty_refs"):
            lines.append("\nSkipped (empty reference segmentation): "
                         + ", ".join(report["empty_refs"]))
        if report.get("template_only"):
            lines.append("\nUsed template (no processed run found, so no tuned "
                         "values): " + ", ".join(report["template_only"]))
        if report.get("warning"):
            lines.append("\n⚠ " + report["warning"])
        return "\n".join(lines)

    def _present_and_save(parent, merged: Dict[str, Any], report: Dict[str, Any],
                          mode: str) -> None:
        box = QMessageBox(parent)
        box.setWindowTitle("Optimization complete")
        box.setIcon(QMessageBox.Warning if report.get("warning")
                    else QMessageBox.Information)
        box.setText(_summary_text(report))
        box.exec_()

        name, ok = QInputDialog.getText(
            parent, "Name the shared config", "Config name:",
            text="joint_optimized")
        if not ok or not name.strip():
            return
        name = name.strip()
        merged["config_name"] = name
        merged.setdefault("mode", mode)

        # Same choice as Export run config: a reusable preset in the Config
        # Library, or a standalone file.
        dest = QMessageBox(parent)
        dest.setWindowTitle("Save shared config")
        dest.setIcon(QMessageBox.Question)
        dest.setText("Where should the shared config go?")
        dest.setInformativeText(
            "Save it as a preset in your Config Library to reuse it on other "
            "images, or write it to a file.")
        preset_btn = dest.addButton("Save to Config Library", QMessageBox.AcceptRole)
        file_btn = dest.addButton("Save to file\u2026", QMessageBox.ActionRole)
        dest.addButton("Cancel", QMessageBox.RejectRole)
        dest.setDefaultButton(preset_btn)
        dest.exec_()
        clicked = dest.clickedButton()

        if clicked == preset_btn:
            _save_to_library(parent, merged, name, mode)
        elif clicked == file_btn:
            _save_to_file(parent, merged, name)

    def _save_to_library(parent, merged: Dict[str, Any], name: str,
                         mode: str) -> None:
        try:
            from . import config_library as cl
            from .config_library import ConfigLibraryError
        except Exception as exc:
            QMessageBox.critical(parent, "Config Library unavailable", str(exc))
            return
        try:
            entry = cl.save_to_library(merged, name=name, mode=mode)
        except FileExistsError:
            reply = QMessageBox.question(
                parent, "Already exists",
                f"A library preset named '{name}' already exists.\n\nOverwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply != QMessageBox.Yes:
                return
            try:
                entry = cl.save_to_library(merged, name=name, mode=mode, overwrite=True)
            except (ConfigLibraryError, OSError) as exc:
                QMessageBox.critical(parent, "Config error", str(exc))
                return
        except (ConfigLibraryError, OSError) as exc:
            QMessageBox.critical(parent, "Config error", str(exc))
            return
        QMessageBox.information(
            parent, "Saved to library",
            f"Saved '{getattr(entry, 'name', name)}' to your Config Library. It "
            "will now appear in the config picker for matching images.")

    def _save_to_file(parent, merged: Dict[str, Any], name: str) -> None:
        try:
            from . import config_library as cl
            default_dir = cl.library_root()
        except Exception:
            default_dir = os.path.expanduser("~")
        default_path = os.path.join(default_dir, f"{name}.yaml")
        dst, _ = QFileDialog.getSaveFileName(
            parent, "Save shared config", default_path,
            "YAML Files (*.yaml *.yml);;All Files (*)")
        if not dst:
            return
        try:
            with open(dst, "w", encoding="utf-8") as fh:
                yaml.safe_dump(merged, fh, default_flow_style=False, sort_keys=False)
        except OSError as exc:
            QMessageBox.critical(parent, "Save failed", str(exc))
            return
        QMessageBox.information(
            parent, "Saved", f"Saved shared config '{name}' to:\n\n{dst}")