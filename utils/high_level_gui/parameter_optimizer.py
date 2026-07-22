"""Joint parameter optimization across several images (initial segmentation).

Problem
-------
Each image can be tuned to its own optimal config, but those per-image optima do
not transfer: image A likes scale-1.0 ``low`` = 0.013, image B likes 0.02, and a
single shared config has to pick one value. The naive answer -- average the
per-image optima -- is wrong, because it ignores how *sensitive* each image is to
the parameter. If A degrades sharply away from 0.013 while B barely notices
around 0.02, the fair shared value sits closer to 0.013.

Method (curvature-weighted compromise)
--------------------------------------
For each image *i* we treat the deviation of its segmentation from its own
desired result as a loss L_i(theta) with a minimum at that image's optimum
theta_i* (there L_i ~ 0, since the desired result IS what theta_i* produces). We
probe the step near theta_i* -- one small step up and down in each optimizable
parameter -- and estimate the local curvature h_ij (how fast L_i rises when
parameter j moves). Modelling each L_i as a local quadratic bowl gives a
closed-form shared optimum, per parameter:

    theta*_j = sum_i h_ij * theta_ij*  /  sum_i h_ij

i.e. a curvature-weighted mean of the per-image optima. It reduces to the plain
average when every image is equally sensitive, and pulls toward whichever image
cares most otherwise. It naturally handles many parameters at once. (Parameters
are treated independently here -- a diagonal-curvature approximation; genuine
coupling would need a full joint search, which is a later tier.)

Deviation metric: 1 - Dice on the foreground mask (labels > 0), evaluated on a
representative crop of each image so the many re-segmentations stay cheap.

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
    dim = config.get("pixel_dimensions" if is_2d else "voxel_dimensions", {}) or {}
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
def _segment_mask(image: np.ndarray, spacing: Tuple[float, ...],
                  params: Dict[str, Any], is_2d: bool) -> np.ndarray:
    """Run raw segmentation on `image` with `params`; return a boolean mask.

    Reads the label memmap into RAM (crops are small) and removes all temp
    output so repeated probing does not accumulate on disk.
    """
    kw = _params_to_kwargs(params)
    run_tmp = os.path.join(tempfile.gettempdir(), f"hibachi_opt_{uuid.uuid4().hex}")
    os.makedirs(run_tmp, exist_ok=True)
    labels_dir = None
    try:
        if is_2d:
            from ..module_2d.initial_2d_segmentation import (
                segment_cells_first_pass_raw_2d as seg)
            result = seg(
                image=image, spacing=spacing,
                tubular_scales=kw["tubular_scales"],
                smooth_sigma=kw["smooth_sigma"],
                connect_max_gap_physical=kw["connect_max_gap_physical"],
                min_size_pixels=kw["min_size"],
                low_threshold_percentile=kw["low_threshold_percentile"],
                high_threshold_percentile=kw["high_threshold_percentile"],
                threshold_mode=kw["threshold_mode"],
                trace_max_gap=kw["trace_max_gap"],
                temp_root_path=run_tmp,
            )
        else:
            from ..module_3d.initial_3d_segmentation import (
                segment_cells_first_pass_raw as seg)
            result = seg(
                volume=image, spacing=spacing,
                tubular_scales=kw["tubular_scales"],
                smooth_sigma=kw["smooth_sigma"],
                connect_max_gap_physical=kw["connect_max_gap_physical"],
                min_size_voxels=kw["min_size"],
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
        mask = np.asarray(mm) > 0
        del mm
        return mask
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


def _pick_crop(image: np.ndarray, is_2d: bool) -> Tuple[slice, ...]:
    """A representative window centred on the brightest region (cheap proxy for
    where structures are), so probing runs on real signal but stays small."""
    if is_2d:
        win = (min(image.shape[0], 512), min(image.shape[1], 512))
    else:
        win = (min(image.shape[0], 24), min(image.shape[1], 256),
               min(image.shape[2], 256))
    if all(w >= s for w, s in zip(win, image.shape)):
        return tuple(slice(0, s) for s in image.shape)

    # Coarse search: score strided window origins by summed intensity.
    img = image.astype(np.float32)
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
            for z in rs[0]:
                for y in rs[1]:
                    for x in rs[2]:
                        yield (z, y, x)

    for origin in _iter_origins(ranges):
        sl = tuple(slice(o, o + w) for o, w in zip(origin, win))
        score = float(img[sl].sum())
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
def optimize_initial_segmentation(
    folders: List[str],
    mode: str,
    progress: Optional[Callable[[float, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Find one shared raw-segmentation config that minimizes deviation from
    each image's own desired result.

    Returns (merged_config_dict, report). Raises OptimizationCancelled if the
    user cancels, or OptimizationError on bad input.
    """
    def _tick(frac: float, msg: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, frac)), msg)

    def _check() -> None:
        if is_cancelled is not None and is_cancelled():
            raise OptimizationCancelled()

    if len(folders) < 2:
        raise OptimizationError("Select at least two images to optimize across.")

    is_2d = mode.endswith("_2d")

    # ---- Load every image, config, crop, and per-image optimum -------------- #
    _tick(0.0, "Loading images and configs…")
    images, spacings, params_list, step_keys, base_configs = [], [], [], [], []
    for k, folder in enumerate(folders):
        _check()
        cfg_path, tif_path = _find_config_and_image(folder)
        config = _load_yaml(cfg_path)
        step_key = _raw_seg_step_key(config)
        params = _flatten_params(config[step_key].get("parameters", config[step_key]))
        image = _load_image(tif_path)
        if is_2d and image.ndim != 2:
            image = np.squeeze(image)
        crop = _pick_crop(image, is_2d)
        image = np.ascontiguousarray(image[crop])
        spacing = _spacing_from_config(config, image.shape, is_2d)
        images.append(image); spacings.append(spacing)
        params_list.append(copy.deepcopy(params)); step_keys.append(step_key)
        base_configs.append(config)
        _tick(0.05 * (k + 1) / len(folders),
              f"Loaded {os.path.basename(folder)} (crop {tuple(image.shape)})")

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

    # ---- Probe curvature around each image's optimum ------------------------ #
    # Total segmentation runs = sum over images of (1 reference + 2 per leaf the
    # image exposes). Track progress across all of them.
    runs_per_image = [1 + 2 * sum(1 for lf in optimizable
                                  if per_image_vals[lf][i] is not None)
                      for i in range(len(images))]
    total_runs = max(1, sum(runs_per_image))
    done = 0

    # weight[leaf] accumulates sum_i h_ij ; wnum[leaf] accumulates sum_i h_ij*theta_ij*
    wsum: Dict[Tuple[Any, str], float] = {lf: 0.0 for lf in optimizable}
    wnum: Dict[Tuple[Any, str], float] = {lf: 0.0 for lf in optimizable}
    curvature_report: Dict[str, List[Dict[str, float]]] = {}
    empty_refs: List[str] = []
    n_measurements = 0          # how many (image, leaf) probes yielded real signal

    def _probe(image, spacing, params, ref, leaf, v0) -> Tuple[float, float]:
        """Grow the perturbation until the segmentation actually changes; return
        (curvature, deviation_at_that_step). (0, 0) if it never responds."""
        nonlocal done
        field = leaf[1]
        base = _perturbation(field, v0)
        last = (0.0, 0.0)
        for mult in _PROBE_MULTS:
            _check()
            delta = base * mult
            devs = []
            for sign in (+1.0, -1.0):
                _check()
                trial = copy.deepcopy(params)
                _set_leaf(trial, leaf, v0 + sign * delta)
                try:
                    m = _segment_mask(image, spacing, trial, is_2d)
                    devs.append(_dice_distance(ref, m))
                except OptimizationCancelled:
                    raise
                except Exception as exc:
                    print(f"[optimize] probe {leaf} {sign:+g} failed: {exc}")
                finally:
                    done += 1
            if not devs:
                return 0.0, 0.0
            dev = sum(devs) / len(devs)
            last = (dev / (delta * delta), dev)
            if dev >= _MEASURABLE:      # mask responded -> curvature is meaningful
                return last
        return last                     # never responded -> ~flat (h ~ 0)

    for i, (image, spacing, params) in enumerate(zip(images, spacings, params_list)):
        _check()
        name = os.path.basename(folders[i])
        _tick(0.05 + 0.9 * done / total_runs, f"[{name}] reference segmentation…")
        ref = _segment_mask(image, spacing, params, is_2d)
        done += 1
        if int(ref.sum()) < _MIN_REF_PIXELS:
            # No segmentable signal on this crop, so nothing can be measured from
            # it; record and skip rather than contributing spurious zeros.
            empty_refs.append(name)
            print(f"[optimize] {name}: reference crop nearly empty "
                  f"({int(ref.sum())} px) — skipping this image.")
            continue

        for leaf in optimizable:
            v0 = per_image_vals[leaf][i]
            if v0 is None:
                continue
            _check()
            scale, field = leaf
            label = field if scale == _SCALAR_KEY else f"scale{scale:g}.{field}"
            _tick(0.05 + 0.9 * done / total_runs, f"[{name}] probing {label}…")
            h, dev = _probe(image, spacing, params, ref, leaf, v0)
            h = max(h, 0.0)
            if dev >= _MEASURABLE:
                n_measurements += 1
            wsum[leaf] += h
            wnum[leaf] += h * v0
            curvature_report.setdefault(label, []).append(
                {"image": name, "optimum": v0, "curvature": round(h, 6),
                 "deviation": round(dev, 4)})

    _check()
    _tick(0.97, "Combining into a shared config…")

    # ---- Curvature-weighted compromise -> merged config --------------------- #
    merged_config = copy.deepcopy(base_configs[0])
    _merged_step = merged_config[step_keys[0]]
    merged_params = _merged_step.get("parameters", _merged_step)
    table_key, _ = _active_profiles(params_list[0])
    report_values: Dict[str, Dict[str, Any]] = {}

    for leaf in optimizable:
        scale, field = leaf
        vals = [v for v in per_image_vals[leaf] if v is not None]
        if wsum[leaf] > 1e-12:
            value = wnum[leaf] / wsum[leaf]
            how = "curvature-weighted"
        else:
            value = float(np.mean(vals))       # all images flat -> plain mean
            how = "mean (no sensitivity)"
        # Never extrapolate beyond the range the images actually used.
        value = float(np.clip(value, min(vals), max(vals)))
        _write_leaf_into_config(merged_params, leaf, value, table_key)
        label = field if scale == _SCALAR_KEY else f"scale{scale:g}.{field}"
        report_values[label] = {
            "per_image": [None if v is None else round(v, 6)
                          for v in per_image_vals[leaf]],
            "shared": round(value, 6) if field != "min_size" else int(round(value)),
            "method": how,
        }

    report = {
        "images": [os.path.basename(f) for f in folders],
        "optimized": report_values,
        "curvature": curvature_report,
        "n_optimized": len(optimizable),
        "n_measurements": n_measurements,
        "empty_refs": empty_refs,
    }
    if n_measurements == 0:
        report["warning"] = (
            "No parameter changed the segmentation on any image's crop, so "
            "sensitivity could not be measured and every value fell back to a "
            "plain average. This usually means the probe crops had too little "
            "signal (min-size filtered them out, or the threshold was above the "
            "crop's intensities). The result is only a mean — not a "
            "curvature-weighted optimum."
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

        def __init__(self, folders: List[str], mode: str, parent=None):
            super().__init__(parent)
            self._folders = list(folders)
            self._mode = mode
            self._cancel = False

        def cancel(self) -> None:
            self._cancel = True

        def run(self) -> None:
            try:
                merged, report = optimize_initial_segmentation(
                    self._folders, self._mode,
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

    def run_optimization_dialog(parent, folders: List[str], mode: str) -> None:
        """Show a cancellable progress dialog, run the optimization, and on
        success prompt for a name + save location and write the merged config."""
        dlg = QProgressDialog("Preparing…", "Cancel", 0, 100, parent)
        dlg.setWindowTitle("Optimize Initial-Segmentation Parameters")
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumWidth(460)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)

        worker = OptimizeWorker(folders, mode, parent)

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
                 f"{len(report['images'])} images:\n"]
        for label, info in report["optimized"].items():
            per = ", ".join("—" if v is None else f"{v:g}" for v in info["per_image"])
            lines.append(f"  • {label}: [{per}] → {info['shared']}  ({info['method']})")
        if report.get("empty_refs"):
            lines.append("\nSkipped (no signal on probe crop): "
                         + ", ".join(report["empty_refs"]))
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