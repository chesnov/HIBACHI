"""
synthetic_engine.py -- procedural synthetic channel generation (bug-fix revision).

The public contract is unchanged: the same `SyntheticDataDialog` and the same
`generate_synthetic_channel(pm, template_ch, rel_filter, out_dir)` entry point.
The following defects are corrected:

  1. METRICS CSV NEVER FOUND IN 3D. The engine looked for
     `metrics_df_fluorescence_2d.csv` and then `metrics_df.csv`. The pipeline
     actually writes `metrics_df_{mode_name}.csv`, i.e. `metrics_df_fluorescence.csv`
     in 3D, and nothing ever writes `metrics_df.csv`. In 3D the metrics frame
     stayed None and the output was pure background noise with zero objects.
     Now the mode is read from the sample config and the filename derived from
     it, with a glob fallback.

  2. 2D-ONLY COLUMN NAMES. `pixel_count` / `skan_num_skeleton_pixels` are the
     2D names; 3D emits `voxel_count` / `skan_num_skeleton_voxels`. Both
     silently defaulted (10 and 0), and because the zero skeleton length killed
     the `branches > 0 and skel > 0` test, every 3D object collapsed to a
     radius-1 blob. Metric lookup now goes through an alias list with physical
     fallbacks (`volume_um3` / `area_um2`, `skan_total_length_um`).

  3. VOXEL SPACING IGNORED, WRONG RADIUS INVERSION. Objects were built in
     isotropic index space, so anisotropic z produced geometrically wrong
     shapes, and `(count / pi) ** (1 / ndim)` is not the sphere or disc
     inversion in either dimension. Geometry is now specified in microns and
     rasterised through an anisotropic distance transform, so a target size is
     matched exactly regardless of spacing.

  4. UNCONSTRAINED PLACEMENT AND ADDITIVE COMPOSITING. Centres were drawn
     uniformly over the whole array with no bounds fit (objects near an edge
     were silently truncated, changing their size) and no overlap test, and
     intensities were summed where objects collided. Placement now fits the
     whole footprint inside the array and uses rejection sampling against an
     occupancy mask with an optional minimum physical separation; object
     intensities composite by maximum, not by sum.

  5. NaN CRASH. `int(row.get('true_num_branches', 0))` raises ValueError on
     NaN, and NaN is the normal value for any object that has no skeleton
     statistics, because the feature pipeline merges skeleton results with
     `how='left'`. All metric reads are now NaN-safe.

  6. UNSAFE IMAGE, DTYPE AND FOLDER HANDLING. `shape[-3:]` mis-slices a
     multi-axis TIFF while background statistics were computed over the whole
     (possibly multi-channel) array; float dtypes were clipped to 1.0
     regardless of their real range; the next channel index was parsed with a
     case-sensitive `startswith` and an unguarded `split('_')[1]`.

  7. FRAGILE RELATIONAL FILTER. The parent-ID column was located by the
     substring test `template_ch.split('_')[-1] in c`, which collides on
     similarly named channels. It now resolves against the same biological
     name the relational engine derives, handles the `A_in_B` intersection
     form, and reports ambiguity instead of guessing.

Background compositing also changed on purpose: `mean_intensity` from the
metrics CSV is measured on the raw image and therefore already contains the
background pedestal, so the old `noise + mean_intensity` double-counted it.
Object voxels are now set to their measured mean intensity and the noise
residual added on top.
"""

import os
import re
import glob
import yaml
import numpy as np
import pandas as pd
import tifffile as tiff
import scipy.ndimage as ndi
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QLineEdit, QPushButton, QMessageBox, QApplication
)
from PyQt5.QtCore import Qt


# =============================================================================
# Metric column aliases
# =============================================================================
# 2D and 3D feature pipelines emit different names for the same quantity.
SIZE_COUNT_COLS = ('voxel_count', 'pixel_count')
SIZE_PHYS_COLS = ('volume_um3', 'area_um2')
SKEL_COUNT_COLS = ('skan_num_skeleton_voxels', 'skan_num_skeleton_pixels')
SKEL_PHYS_COLS = ('skan_total_length_um',)
BRANCH_COLS = ('true_num_branches',)
INTENSITY_COLS = ('mean_intensity', 'median_intensity')

DEFAULT_TARGET_VOXELS = 10
SOMA_VOLUME_FRACTION = 0.30
MAX_GRID_PER_AXIS = 512          # guard against absurd metrics blowing up RAM
PLACEMENT_ATTEMPTS = 200


# =============================================================================
# Small numeric helpers
# =============================================================================

def _safe_float(value, default=np.nan):
    """float() that survives None, strings and NaN/inf."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _metric(row, names, default=0.0):
    """First finite value among `names` in a metrics row, else `default`.

    Fixes bug 2 (2D/3D column-name divergence) and bug 5 (NaN crash).
    """
    for name in names:
        if name in row.index:
            val = _safe_float(row[name])
            if np.isfinite(val):
                return val
    return float(default)


def _radius_for_size(physical_size, ndim):
    """Radius of the ball/disc of the given physical volume/area, in microns.

    The original `(count / pi) ** (1 / ndim)` is neither the 3D sphere nor the
    2D disc inversion. Correct forms are r = (3V / 4pi)^(1/3) and r = (A / pi)^(1/2).
    """
    physical_size = max(float(physical_size), 0.0)
    if physical_size <= 0:
        return 0.0
    if ndim == 3:
        return (3.0 * physical_size / (4.0 * np.pi)) ** (1.0 / 3.0)
    return (physical_size / np.pi) ** 0.5


def _physical_distance_field(grid_shape, centre, spacing):
    """Euclidean distance in microns from `centre`, broadcast over the grid."""
    total = np.zeros(grid_shape, dtype=np.float32)
    for axis, (size, c, sp) in enumerate(zip(grid_shape, centre, spacing)):
        offs = ((np.arange(size, dtype=np.float32) - float(c)) * float(sp)) ** 2
        shape = [1] * len(grid_shape)
        shape[axis] = size
        total += offs.reshape(shape)
    return np.sqrt(total, out=total)


# =============================================================================
# Sample introspection: mode, spacing, image, metrics
# =============================================================================

def _read_sample_config(ch_path):
    """Load the per-sample YAML from a channel's sample folder ({} if absent)."""
    try:
        names = sorted(f for f in os.listdir(ch_path)
                       if f.lower().endswith(('.yaml', '.yml')))
    except OSError:
        return {}
    for name in names:
        try:
            with open(os.path.join(ch_path, name), 'r') as fh:
                cfg = yaml.safe_load(fh) or {}
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            continue
    return {}


def _mode_name(cfg):
    """Pipeline mode string ('fluorescence' / 'fluorescence_2d'), or ''."""
    mode = cfg.get('mode', '')
    return mode if isinstance(mode, str) else ''


def _spatial_ndim(cfg, img_ndim):
    """Number of spatial axes: authoritative from the config mode, else the image.

    Using the mode disambiguates a 3-axis array that could be (Z, Y, X) or
    (C, Y, X) -- part of bug 6.
    """
    mode = _mode_name(cfg)
    if mode.endswith('_2d'):
        return 2
    if mode:
        return 3
    return 2 if img_ndim <= 2 else 3


def _reduce_to_spatial(img, ndim_spatial, sample_name):
    """Drop leading non-spatial axes so the array has exactly `ndim_spatial` dims.

    Fixes bug 6: the old `shape[-3:]` sliced the shape but left `real_img`
    multi-channel, so background statistics were pooled across channels and
    the noise array shape disagreed with the image.
    """
    out = np.squeeze(img)
    while out.ndim > ndim_spatial:
        print(f"  [{sample_name}] Template has {out.ndim} axes for a "
              f"{ndim_spatial}D sample; taking index 0 of axis 0 "
              f"(shape {out.shape}).")
        out = out[0]
    if out.ndim < ndim_spatial:
        raise ValueError(
            f"Template image has {out.ndim} spatial axes but the sample "
            f"config declares {ndim_spatial}D."
        )
    return out


def _read_spacing(cfg, shape, sample_name):
    """Physical spacing per axis in microns, from total extent / voxel count.

    Configs store `voxel_dimensions` (3D) / `pixel_dimensions` (2D) as TOTAL
    physical extents, which is how `run_batch_analysis` derives spacing too.
    """
    ndim = len(shape)
    dims = cfg.get('voxel_dimensions') or cfg.get('pixel_dimensions') or {}
    if not isinstance(dims, dict):
        dims = {}
    keys = ('z', 'y', 'x') if ndim == 3 else ('y', 'x')

    spacing, missing = [], False
    for key, size in zip(keys, shape):
        extent = _safe_float(dims.get(key))
        if np.isfinite(extent) and extent > 0 and size > 0:
            spacing.append(extent / float(size))
        else:
            spacing.append(1.0)
            missing = True

    if missing:
        print(f"  [{sample_name}] Warning: incomplete "
              f"{'voxel' if ndim == 3 else 'pixel'}_dimensions in config; "
              f"using 1.0 um for the missing axes. Object geometry will be "
              f"in voxel units for those axes.")
    return tuple(spacing)


def _find_metrics_csv(ch_path, mode):
    """Locate the per-object metrics CSV inside a `*_processed_*` folder.

    Fixes bug 1. Preferred name is `metrics_df_{mode}.csv` as written by
    `ProcessingStrategy.get_checkpoint_files`; falls back to any
    `metrics_df*.csv` so a renamed or legacy run still resolves.
    """
    try:
        entries = sorted(os.listdir(ch_path))
    except OSError:
        return None

    proc_dirs = [os.path.join(ch_path, d) for d in entries
                 if "_processed_" in d and os.path.isdir(os.path.join(ch_path, d))]

    if mode:
        for proc in proc_dirs:
            exact = os.path.join(proc, f"metrics_df_{mode}.csv")
            if os.path.exists(exact):
                return exact
    for proc in proc_dirs:
        hits = sorted(glob.glob(os.path.join(proc, "metrics_df*.csv")))
        if hits:
            return hits[0]
    return None


# =============================================================================
# Relational filter
# =============================================================================

def _biological_name(channel_key):
    """Channel key -> biological name, matching RelationalEngine's derivation."""
    parts = channel_key.split('_', 2)
    return parts[-1] if len(parts) > 2 else channel_key


def _resolve_parent_id_column(rel_df, template_ch):
    """Pick the parent-ID column belonging to `template_ch`.

    Fixes bug 7. Accepts either `parent_id_{name}` (a plain channel used as
    primary) or `parent_id_{name}_in_{other}` (that channel used as the A side
    of an intersection with IDs preserved). Falls back to a lone parent_id
    column, and refuses to guess when several are plausible.
    """
    bio = _biological_name(template_ch)
    candidates = [c for c in rel_df.columns if c.startswith('parent_id_')]
    if not candidates:
        return None

    exact = f"parent_id_{bio}"
    if exact in candidates:
        return exact

    prefixed = [c for c in candidates
                if c[len('parent_id_'):].startswith(f"{bio}_in_")]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        print(f"  Warning: several parent-ID columns match '{bio}': "
              f"{prefixed}. Using {prefixed[0]}.")
        return prefixed[0]

    if len(candidates) == 1:
        print(f"  Warning: no parent-ID column named for '{bio}'; falling "
              f"back to the only one present ({candidates[0]}).")
        return candidates[0]

    print(f"  Warning: cannot decide which parent-ID column belongs to "
          f"'{bio}' among {candidates}; relational filter not applied.")
    return None


def _apply_relational_filter(df, rel_dir, rel_filter, sample_key, template_ch):
    """Restrict `df` to objects that appear in a saved cross-channel run."""
    rel_csv = os.path.join(rel_dir, rel_filter, sample_key,
                           f"{sample_key}_relational_metrics.csv")
    if not os.path.exists(rel_csv):
        print(f"  [{sample_key}] Relational filter '{rel_filter}' has no "
              f"metrics CSV for this sample; using all objects.")
        return df
    try:
        rel_df = pd.read_csv(rel_csv)
    except Exception as exc:
        print(f"  [{sample_key}] Could not read {rel_csv}: {exc}")
        return df

    col = _resolve_parent_id_column(rel_df, template_ch)
    if col is None or 'label' not in df.columns:
        return df

    valid = pd.to_numeric(rel_df[col], errors='coerce').dropna().astype(int).unique()
    filtered = df[df['label'].astype(int).isin(valid)]
    print(f"  [{sample_key}] Relational filter kept {len(filtered)}/{len(df)} "
          f"objects via '{col}'.")
    return filtered


# =============================================================================
# Object rasterisation
# =============================================================================

def _draw_line_nd(mask, start, end):
    """Draw an N-dimensional line on a boolean mask (max-norm connected)."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    steps = int(np.max(np.abs(end - start))) + 1
    for i in range(steps + 1):
        t = i / float(steps)
        pt = np.round(start + t * (end - start)).astype(int)
        pt = np.clip(pt, 0, np.array(mask.shape) - 1)
        mask[tuple(pt)] = True


def _grow_to_size(dist, lo, hi, target_voxels):
    """Mask of exactly `target_voxels` voxels, filling nearest-first.

    `lo` and `hi` bracket the target: `dist <= lo` is short of it and
    `dist <= hi` reaches or exceeds it. Voxels in the shell between them are
    added in ascending distance order until the count is exact, so a target
    size is honoured even though the distance field is discrete.
    """
    mask = dist <= lo
    deficit = target_voxels - int(np.count_nonzero(mask))
    if deficit <= 0:
        return mask
    shell = np.flatnonzero((dist > lo) & (dist <= hi))
    if shell.size == 0:
        return dist <= hi
    if deficit >= shell.size:
        mask |= (dist > lo) & (dist <= hi)
        return mask
    order = np.argsort(dist.reshape(-1)[shell], kind='stable')[:deficit]
    flat = mask.reshape(-1)
    flat[shell[order]] = True
    return mask


def _trim_to_size(mask, radial, target_voxels):
    """Shrink `mask` to exactly `target_voxels` by dropping the outermost voxels."""
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size <= target_voxels:
        return mask
    keep = idx[np.argsort(radial.reshape(-1)[idx], kind='stable')[:target_voxels]]
    out = np.zeros(mask.size, dtype=bool)
    out[keep] = True
    return out.reshape(mask.shape)


def _generate_object_mask(target_voxels, branches, skel_len_um, spacing,
                          rng, soma_fraction=SOMA_VOLUME_FRACTION):
    """Boolean mask of one procedural object, sized in physical units.

    Fixes bug 3. A seed (soma ball, plus radial branch polylines when the
    template has skeleton statistics) is rasterised in index space, then
    thickened by thresholding an anisotropic Euclidean distance transform.
    Bisecting the threshold matches `target_voxels` exactly and honours
    anisotropic spacing, replacing the old isotropic capped-dilation loop
    that could neither reach nor respect the target.
    """
    ndim = len(spacing)
    spacing = tuple(float(s) for s in spacing)
    target_voxels = max(1, int(target_voxels))
    unit_size = float(np.prod(spacing))
    target_physical = target_voxels * unit_size

    branches = int(max(0, round(branches)))
    ramified = branches > 0 and skel_len_um > 0

    if ramified:
        soma_radius = _radius_for_size(target_physical * soma_fraction, ndim)
        branch_len = float(skel_len_um) / branches
        reach = soma_radius + branch_len * 1.5
    else:
        soma_radius = _radius_for_size(target_physical, ndim)
        branch_len = 0.0
        reach = soma_radius * 1.25

    # Grid large enough for the seed plus the thickening headroom.
    headroom = max(reach, soma_radius) + max(spacing)
    half = [min(MAX_GRID_PER_AXIS // 2, int(np.ceil(headroom / sp)) + 2)
            for sp in spacing]
    grid_shape = tuple(2 * h + 1 for h in half)
    centre = tuple(half)

    radial = _physical_distance_field(grid_shape, centre, spacing)
    seed = radial <= soma_radius
    seed[centre] = True                      # never leave an empty seed

    if ramified:
        for _ in range(branches):
            direction = rng.normal(size=ndim)
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue
            direction /= norm
            length = branch_len * rng.uniform(0.5, 1.5)
            end_index = np.array(centre, dtype=float) + (direction * length) / np.array(spacing)
            _draw_line_nd(seed, centre, end_index)

    seed_count = int(np.count_nonzero(seed))
    if seed_count > target_voxels:
        # The template's skeleton implies more voxels than its size allows
        # (inconsistent metrics). Shorten the branches by dropping the voxels
        # farthest from the centre, which preserves the soma.
        mask = _trim_to_size(seed, radial, target_voxels)
    elif seed_count == target_voxels:
        mask = seed
    else:
        # One EDT, then bisect the physical thickening radius. Thresholding a
        # discrete distance field is a step function, so bisection alone
        # overshoots; the shell between the bracketing radii is then filled
        # nearest-first to land on the target exactly.
        dist = ndi.distance_transform_edt(~seed, sampling=spacing).astype(np.float32)
        hi = float(dist.max())
        if np.count_nonzero(dist <= hi) < target_voxels:
            mask = dist <= hi          # grid-bound; accept the largest we can build
        else:
            lo = 0.0
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if np.count_nonzero(dist <= mid) < target_voxels:
                    lo = mid
                else:
                    hi = mid
            mask = _grow_to_size(dist, lo, hi, target_voxels)

    coords = np.argwhere(mask)
    if coords.size == 0:
        out = np.zeros((1,) * ndim, dtype=bool)
        out[(0,) * ndim] = True
        return out
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    return mask[tuple(slice(mins[d], maxs[d]) for d in range(ndim))].copy()


def _expand_mask_physical(mask, spacing, margin_um):
    """Mask grown outward by `margin_um` microns (anisotropy-aware)."""
    if margin_um <= 0:
        return mask
    pad = [int(np.ceil(margin_um / float(sp))) + 1 for sp in spacing]
    padded = np.pad(mask, [(p, p) for p in pad], mode='constant', constant_values=False)
    dist = ndi.distance_transform_edt(~padded, sampling=spacing)
    return dist <= margin_um


# =============================================================================
# Placement
# =============================================================================

def _place_object(occupancy, mask, spacing, rng, min_separation_um,
                  attempts=PLACEMENT_ATTEMPTS):
    """Find an origin where `mask` fits fully in bounds and clears occupancy.

    Fixes bug 4: the old code drew a centre uniformly over the array and let
    `_add_local_mask` clip whatever hung over the edge, so edge objects lost
    volume, and it never checked whether the site was already taken.
    Returns the origin index tuple, or None if no free site was found.
    """
    ndim = mask.ndim
    shape = occupancy.shape
    test_mask = _expand_mask_physical(mask, spacing, min_separation_um)

    # Offset of the expanded mask relative to the original.
    offset = [(t - m) // 2 for t, m in zip(test_mask.shape, mask.shape)]

    limits = []
    for d in range(ndim):
        room = shape[d] - test_mask.shape[d]
        if room < 0:
            return None, 'oversized'          # object larger than the image
        limits.append(room + 1)

    for _ in range(attempts):
        origin_test = [int(rng.integers(0, limits[d])) for d in range(ndim)]
        sl = tuple(slice(origin_test[d], origin_test[d] + test_mask.shape[d])
                   for d in range(ndim))
        if not np.any(occupancy[sl] & test_mask):
            return tuple(origin_test[d] + offset[d] for d in range(ndim)), 'ok'
    return None, 'crowded'


def _stamp(target, mask, origin, value, mode='max'):
    """Write `value` where `mask` is True, at `origin`, in place."""
    sl = tuple(slice(origin[d], origin[d] + mask.shape[d])
               for d in range(mask.ndim))
    view = target[sl]
    if mode == 'max':
        np.maximum(view, mask * value, out=view)
    else:
        view[mask] = True


# =============================================================================
# Core engine
# =============================================================================

def generate_synthetic_channel(pm, template_ch, rel_filter, out_dir,
                               seed=None, min_separation_um=0.0,
                               allow_overlap=False):
    """Generate a synthetic channel from a template channel's object statistics.

    Args:
        pm: ProjectManager holding a built `sample_registry`.
        template_ch: Channel key to draw per-object statistics from.
        rel_filter: Name of a saved RELATIONAL_ANALYSIS run to restrict
            objects to, or None for all objects.
        out_dir: Destination channel folder.
        seed: Optional integer for reproducible placement.
        min_separation_um: Minimum physical gap enforced between objects.
        allow_overlap: When True, skip the occupancy test entirely (restores
            the old unconstrained behaviour, minus the additive intensities).
    """
    rng = np.random.default_rng(seed)
    rel_dir = os.path.join(os.path.dirname(pm.project_path), "RELATIONAL_ANALYSIS")

    for sample_key, ch_dict in pm.sample_registry.items():
        if template_ch not in ch_dict:
            continue

        ch_path = ch_dict[template_ch]
        # `sample_key` is the lowercased, cleaned MATCHING key; the on-disk
        # folder keeps its original case. Name the output after the ORIGINAL so
        # the project-view tree -- which groups channels by case-sensitive
        # basename -- nests this channel under the existing sample instead of
        # listing it separately. The clean key is still used for
        # RELATIONAL_ANALYSIS lookups, whose folders use that form.
        orig_sample_name = os.path.basename(os.path.normpath(ch_path))

        tif_file = next((f for f in sorted(os.listdir(ch_path))
                         if f.lower().endswith(('.tif', '.tiff'))), None)
        if not tif_file:
            print(f"  [{sample_key}] No template TIFF; skipped.")
            continue

        cfg = _read_sample_config(ch_path)
        mode = _mode_name(cfg)

        raw = tiff.imread(os.path.join(ch_path, tif_file))
        ndim_spatial = _spatial_ndim(cfg, raw.ndim)
        try:
            real_img = _reduce_to_spatial(raw, ndim_spatial, sample_key)
        except ValueError as exc:
            print(f"  [{sample_key}] {exc} Skipped.")
            continue
        del raw

        shape = real_img.shape
        spacing = _read_spacing(cfg, shape, sample_key)

        # Background model, computed on the spatial array only (bug 6).
        bg_mean = float(np.median(real_img))
        low = real_img < np.percentile(real_img, 90)
        bg_std = float(np.std(real_img[low])) if np.any(low) else 1.0
        if not np.isfinite(bg_std) or bg_std <= 0:
            bg_std = 1.0

        # Per-object statistics (bug 1).
        csv_path = _find_metrics_csv(ch_path, mode)
        df = None
        if csv_path:
            try:
                df = pd.read_csv(csv_path)
            except Exception as exc:
                print(f"  [{sample_key}] Could not read {csv_path}: {exc}")
        else:
            print(f"  [{sample_key}] No metrics CSV found under "
                  f"{ch_path}; writing background only.")

        if df is not None and not df.empty and rel_filter:
            df = _apply_relational_filter(df, rel_dir, rel_filter,
                                          sample_key, template_ch)

        object_layer = np.zeros(shape, dtype=np.float32)
        occupancy = np.zeros(shape, dtype=bool)
        unit_size = float(np.prod(spacing))
        drawn = 0
        skipped = {'oversized': 0, 'crowded': 0}

        if df is not None and not df.empty:
            for _, row in df.iterrows():
                # Target size: prefer the direct count, else derive from the
                # physical measurement (bug 2).
                count = _metric(row, SIZE_COUNT_COLS, 0.0)
                if count < 1:
                    physical = _metric(row, SIZE_PHYS_COLS, 0.0)
                    count = physical / unit_size if (physical > 0 and unit_size > 0) else 0.0
                target_voxels = int(round(count)) if count >= 1 else DEFAULT_TARGET_VOXELS

                # Skeleton extent in microns; the physical column is now the
                # preferred input because the generator works in physical units.
                skel_len_um = _metric(row, SKEL_PHYS_COLS, 0.0)
                if skel_len_um <= 0:
                    skel_voxels = _metric(row, SKEL_COUNT_COLS, 0.0)
                    skel_len_um = skel_voxels * float(np.mean(spacing)) if skel_voxels > 0 else 0.0

                branches = _metric(row, BRANCH_COLS, 0.0)
                intensity = _metric(row, INTENSITY_COLS, bg_mean + bg_std * 10.0)

                mask = _generate_object_mask(target_voxels, branches,
                                             skel_len_um, spacing, rng)

                if allow_overlap:
                    limits = [shape[d] - mask.shape[d] + 1 for d in range(ndim_spatial)]
                    if any(lim <= 0 for lim in limits):
                        origin, reason = None, 'oversized'
                    else:
                        origin = tuple(int(rng.integers(0, lim)) for lim in limits)
                        reason = 'ok'
                else:
                    origin, reason = _place_object(occupancy, mask, spacing, rng,
                                                  min_separation_um)
                if origin is None:
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue

                _stamp(object_layer, mask, origin, intensity, mode='max')
                _stamp(occupancy, mask, origin, True, mode='set')
                drawn += 1

        # Compose: object voxels sit at their measured mean intensity, which
        # already includes the background pedestal, so the pedestal is not
        # added twice. Noise is applied everywhere.
        composed = np.where(object_layer > 0, object_layer, bg_mean).astype(np.float32)
        composed += rng.normal(loc=0.0, scale=bg_std, size=shape).astype(np.float32)
        composed = ndi.gaussian_filter(composed, sigma=1.0)

        # Dtype-correct clipping (bug 6): floats were previously clipped to 1.0.
        if np.issubdtype(real_img.dtype, np.integer):
            info = np.iinfo(real_img.dtype)
            lo_clip, hi_clip = float(info.min), float(info.max)
        else:
            hi_clip = float(np.nanmax(real_img))
            if not np.isfinite(hi_clip) or hi_clip <= 0:
                hi_clip = 1.0
            lo_clip = min(0.0, float(np.nanmin(real_img)))
        synth_img = np.clip(composed, lo_clip, hi_clip).astype(real_img.dtype)

        sample_out = os.path.join(out_dir, orig_sample_name)
        os.makedirs(sample_out, exist_ok=True)
        tiff.imwrite(os.path.join(sample_out, f"{orig_sample_name}.tif"), synth_img)

        # Copy the config, tagged so the channel can be identified later
        # (e.g. purged on re-setup by organize_wizard.is_synthetic_channel).
        yml = next((f for f in sorted(os.listdir(ch_path))
                    if f.lower().endswith(('.yaml', '.yml'))), None)
        if yml:
            out_cfg = dict(cfg)
            out_cfg['synthetic'] = True
            out_cfg['synthetic_source_channel'] = template_ch
            if rel_filter:
                out_cfg['synthetic_relational_filter'] = rel_filter
            with open(os.path.join(sample_out, yml), 'w') as fh:
                yaml.dump(out_cfg, fh, default_flow_style=False, sort_keys=False)

        notes = []
        if skipped.get('oversized'):
            notes.append(f"{skipped['oversized']} too large for the field of view")
        if skipped.get('crowded'):
            notes.append(f"{skipped['crowded']} found no free site "
                         f"in {PLACEMENT_ATTEMPTS} attempts")
        note = f" ({'; '.join(notes)} -- skipped)" if notes else ""
        print(f"[{sample_key}] Generated {drawn} procedural objects{note} "
              f"(spacing={tuple(round(s, 4) for s in spacing)}).")


# =============================================================================
# Dialog
# =============================================================================

_CHANNEL_DIR_RE = re.compile(r"(?i)^channel_(\d+)_")


def _next_channel_index(project_root):
    """Highest existing channel number + 1, case-insensitively and safely.

    Fixes bug 6: the old parse was case-sensitive, did not check isdir, and
    would IndexError on a folder named exactly 'Channel_'.
    """
    best = -1
    try:
        entries = os.listdir(project_root)
    except OSError:
        entries = []
    for name in entries:
        if not os.path.isdir(os.path.join(project_root, name)):
            continue
        m = _CHANNEL_DIR_RE.match(name)
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


class SyntheticDataDialog(QDialog):
    def __init__(self, project_manager, parent=None):
        super().__init__(parent)
        self.pm = project_manager
        self.setWindowTitle("Generate Procedural Synthetic Data")
        self.setMinimumWidth(450)

        self.project_root = os.path.dirname(self.pm.project_path)
        self.rel_dir = os.path.join(self.project_root, "RELATIONAL_ANALYSIS")

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        layout.addWidget(QLabel("1. Select Template Channel (Source of stats):"))
        self.cb_channel = QComboBox()
        channels = set()
        for _sample, ch_dict in self.pm.sample_registry.items():
            channels.update(ch_dict.keys())
        self.cb_channel.addItems(sorted(channels))
        layout.addWidget(self.cb_channel)

        layout.addWidget(QLabel("2. Filter by Relational Analysis (Optional):"))
        layout.addWidget(QLabel("<i>(Only generates objects that matched this condition)</i>"))
        self.cb_filter = QComboBox()
        self.cb_filter.addItem("None (Use all objects in channel)")
        if os.path.isdir(self.rel_dir):
            runs = sorted(d for d in os.listdir(self.rel_dir)
                          if os.path.isdir(os.path.join(self.rel_dir, d)))
            self.cb_filter.addItems(runs)
        layout.addWidget(self.cb_filter)

        layout.addWidget(QLabel("3. New Channel Name:"))
        self.le_output = QLineEdit("Synthetic_Data")
        layout.addWidget(self.le_output)

        btn_layout = QHBoxLayout()
        btn_run = QPushButton("Generate")
        btn_run.setStyleSheet("background-color: #2E8B57; color: white; font-weight: bold;")
        btn_run.clicked.connect(self.run_generation)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_run)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

    def run_generation(self):
        template_ch = self.cb_channel.currentText()
        rel_filter = self.cb_filter.currentText()
        if rel_filter.startswith("None"):
            rel_filter = None

        out_name = self.le_output.text().strip()
        if not out_name:
            QMessageBox.warning(self, "Error", "Please provide an output name.")
            return

        next_idx = _next_channel_index(self.project_root)
        out_channel_dir = os.path.join(self.project_root, f"Channel_{next_idx}_{out_name}")
        os.makedirs(out_channel_dir, exist_ok=True)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            generate_synthetic_channel(self.pm, template_ch, rel_filter, out_channel_dir)
            QApplication.restoreOverrideCursor()
            QMessageBox.information(
                self, "Success",
                f"Synthetic data generated!\nSaved to: {out_channel_dir}\n\n"
                "It will appear in the project view automatically when you return to it."
            )
            self.accept()
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Generation failed:\n{exc}")