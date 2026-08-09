"""
null_export.py -- on-disk schema for HIBACHI's spatial-null export.

A project is typically ONE biological replicate whose images are technical
replicates, so no test run inside a project can be the biological claim. This
module therefore writes the RAW material for downstream analysis rather than
conclusions: enough that a notebook pooling several projects can compute any
statistic without re-running the Monte-Carlo, which is the expensive part.

WHAT MAKES THAT POSSIBLE
    The null table is per-draw AND per-object, and every row carries
    `template_label`. Exporting only per-draw summaries would permanently lock
    the analysis into the choices made here -- no switching the summary
    statistic, no size cutoffs, no recomputing the index with a different tail
    convention -- because all of those need the individual objects back.

    Centroids are exported for observed and null alike, so second-order
    statistics (Ripley's K, the pair correlation function) remain computable
    later for BOTH sides. Full pairwise distances are deliberately not
    exported: no statistic in this method uses them, on the null side they
    would be ~158M rows for a modest project, and exporting them for the
    observed data alone would give a statistic with no null to compare against.

ARTIFACTS (per project, ~30-40 MB for 20 images x 200 objects x 398 draws)
    manifest.json         schema version, parameters, comparability key
    image_metadata.csv    one row per image: geometry, diagnostics, indices
    observed_objects.csv  one row per real object
    null_objects.npz      one row per (image, draw, object) -- the bulk
    f_curves.npz          F CDFs on the run's shared grid

FORMAT
    `.npz` for the bulk arrays and `.csv` for the small tables. Deliberately no
    parquet: pyarrow is not in HIBACHI's validated environment, and that file is
    the single source of truth that triggers an environment rebuild for every
    user when it changes. numpy and pandas alone are enough here, and the
    loader turns the npz back into a tidy frame in one call.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "hibachi-spatial-null/1"

# Parameters that MUST match before two projects may be pooled. A domain or
# distance mismatch invalidates pooling outright; the loader refuses rather than
# quietly returning a confident, meaningless answer. Replicate counts and seeds
# may differ freely and are not in the key.
COMPARABILITY_KEY = (
    "schema_version",
    "ndim",
    # The REALISED domain, collected from the images actually run -- not the
    # requested parameter. `use_hull=True` silently falls back to the whole
    # field when a project has no persisted hull, so trusting the request would
    # let a hull run and a field run pool without complaint.
    "domain_source",
    "use_hull",
    "erode_um",
    "rotate",
    "hardcore",
    "min_separation_um",
    "distance_semantics",
    "cross_statistic",
    "f_grid_max_um",
    "f_grid_points",
)

# The F grid is derived from the data, so two projects can legitimately differ
# here. Unlike the rest of the key, a mismatch is recoverable: CDFs are
# monotone, bounded on [0,1] and sampled densely, so linear interpolation onto
# a common grid is lossless well below any meaningful difference. Treated as a
# warning by the loader, not a refusal.
SOFT_KEYS = ("f_grid_max_um", "f_grid_points")

_NULL_COLUMNS_MIN = ("image_id", "draw", "set", "template_label", "voxels")


def build_manifest(project_name: str,
                   ndim: int,
                   parameters: Dict[str, Any],
                   grid_info: Dict[str, Any],
                   channels: Dict[str, Any],
                   n_images: int,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Manifest describing one export, including its comparability key."""
    man: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "project": project_name,
        "ndim": int(ndim),
        "n_images": int(n_images),
        "channels": channels,
        # Boundary-to-boundary, matching interaction_analysis*. Recorded because
        # a future change here would silently make old exports incomparable.
        "distance_semantics": "surface_to_surface_um",
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
    }
    man.update({k: parameters.get(k) for k in (
        "rotate", "hardcore", "min_separation_um", "cross_statistic",
        "n_reference", "n_test", "seed", "use_hull",
        "roi_name", "erode_um", "compute_f", "compute_g", "max_attempts")})
    man.update({k: grid_info.get(k) for k in
                ("f_grid_max_um", "f_grid_raw_max_um", "f_grid_points")})
    if extra:
        man.update(extra)
    man["comparability_key"] = {k: man.get(k) for k in COMPARABILITY_KEY}
    return man


def write_project_export(out_dir: str,
                         manifest: Dict[str, Any],
                         image_metadata: pd.DataFrame,
                         observed_objects: pd.DataFrame,
                         null_objects: pd.DataFrame,
                         f_curves: Optional[Dict[str, Any]] = None,
                         also_csv: bool = False) -> Dict[str, str]:
    """Write the five artifacts. Returns {artifact: path}."""
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}

    p = os.path.join(out_dir, "manifest.json")
    with open(p, "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    written["manifest"] = p

    p = os.path.join(out_dir, "image_metadata.csv")
    image_metadata.to_csv(p, index=False)
    written["image_metadata"] = p

    p = os.path.join(out_dir, "observed_objects.csv")
    observed_objects.to_csv(p, index=False)
    written["observed_objects"] = p

    # The bulk table goes out columnar: ~15 MB against ~23 MB gzipped CSV, and
    # it round-trips without dtype guessing. Object columns (image ids) are
    # cast to fixed-width unicode, which numpy stores natively -- leaving them
    # as objects would force allow_pickle on load, which is both a security
    # footgun and unnecessary here.
    p = os.path.join(out_dir, "null_objects.npz")
    payload = {}
    for c in null_objects.columns:
        col = null_objects[c].to_numpy()
        if col.dtype == object:
            col = col.astype(str)
        payload[c] = col
    np.savez_compressed(p, **payload)
    written["null_objects"] = p
    if also_csv:
        q = os.path.join(out_dir, "null_objects.csv.gz")
        null_objects.to_csv(q, index=False, compression="gzip")
        written["null_objects_csv"] = q

    if f_curves:
        p = os.path.join(out_dir, "f_curves.npz")
        np.savez_compressed(p, **f_curves)
        written["f_curves"] = p

    return written


def stack_f_curves(per_image: Sequence[Dict[str, Any]],
                   image_ids: Sequence[Any],
                   grid: np.ndarray) -> Dict[str, Any]:
    """Pack per-image F curves into one array set on the shared grid."""
    observed, reference, nulls, img_of_null, set_of_null, ids = [], [], [], [], [], []
    for img_id, cur in zip(image_ids, per_image):
        if not cur:
            continue
        ids.append(img_id)
        observed.append(np.asarray(cur["observed"], dtype=np.float32))
        reference.append(np.asarray(cur["reference"], dtype=np.float32))
        n = np.asarray(cur["null"], dtype=np.float32)
        nulls.append(n)
        img_of_null.append(np.full(n.shape[0], len(ids) - 1, dtype=np.int16))
        set_of_null.append(np.asarray(cur["set"], dtype=np.int8))
    if not ids:
        return {}
    return {
        "grid": np.asarray(grid, dtype=np.float32),
        "image_ids": np.asarray([str(i) for i in ids]),
        "observed": np.stack(observed),
        "reference": np.stack(reference),
        "null": np.concatenate(nulls, axis=0),
        "null_image_index": np.concatenate(img_of_null),
        "null_set": np.concatenate(set_of_null),
    }


def concat_null_frames(frames: Iterable[pd.DataFrame],
                       image_ids: Iterable[Any]) -> pd.DataFrame:
    """Tag each image's null frame with its id and concatenate."""
    out: List[pd.DataFrame] = []
    for img_id, frame in zip(image_ids, frames):
        if frame is None or frame.empty:
            continue
        f = frame.copy()
        f.insert(0, "image_id", str(img_id))
        out.append(f)
    if not out:
        return pd.DataFrame(columns=list(_NULL_COLUMNS_MIN))
    return pd.concat(out, ignore_index=True)


def concat_observed_frames(frames: Iterable[pd.DataFrame],
                           image_ids: Iterable[Any]) -> pd.DataFrame:
    """Tag each image's observed frame with its id and concatenate."""
    out: List[pd.DataFrame] = []
    for img_id, frame in zip(image_ids, frames):
        if frame is None or frame.empty:
            continue
        f = frame.copy()
        f.insert(0, "image_id", str(img_id))
        out.append(f)
    if not out:
        return pd.DataFrame(columns=["image_id", "template_label"])
    return pd.concat(out, ignore_index=True)
