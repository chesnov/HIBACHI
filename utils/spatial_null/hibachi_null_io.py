"""
hibachi_null_io.py -- load and pool HIBACHI spatial-null exports in a notebook.

Standalone by design: needs only numpy, pandas and scipy, so it runs on a
laptop with no HIBACHI install. Copy it next to your notebook.

WHY THIS EXISTS
    A project is usually one biological replicate, so inference has to happen
    across projects. Building that as an application layer would mean sample
    registries, UI and state; as a loader over a fixed on-disk schema it is a
    few hundred lines. This is the loader.

WHAT IT GUARDS
    Pooling two projects whose nulls were defined differently -- one on a
    tissue hull, one on the whole field -- produces a confident and meaningless
    answer. Every export carries a comparability key, and `load_projects`
    refuses on a hard mismatch, naming the offending key. A grid mismatch is
    recoverable and only warns: F curves are monotone, bounded and densely
    sampled, so they interpolate onto a common grid losslessly.

TYPICAL USE
    import hibachi_null_io as hio

    runs = hio.discover_runs(["/data/rep1", "/data/rep2", "/data/rep3"])
    print(runs[["project", "run_name", "primary", "partner", "domain"]])

    paths = hio.matching_runs(runs, primary="Aggregates", partner="Microglia")
    ds = hio.load_projects(paths, group_from_name=r"(WT|KO)")

    # direction="primary": per randomised object. "partner": per fixed partner.
    eff = hio.per_image_effects(ds, statistic="median", direction="partner")
    res = hio.replicate_test(eff, effect_col="effect_z")

INFERENTIAL UNIT
    `replicate_test` aggregates images to one value per project (the replicate)
    before testing, because images within a project are technical replicates.
    Testing images directly would inflate n by roughly the images-per-project
    factor. The default effect is standardised (`effect_z`): the same shift in
    microns is much stronger evidence against a tight null than a wide one, and
    pooling raw microns weights images as if that were not so.
"""

from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "hibachi-spatial-null/1"
SOFT_KEYS = ("f_grid_max_um", "f_grid_points")

SDI_INTERPRETATION = {
    "F": "low = regular/evenly spread, high = clustered",
    "G": "low = clustered, high = regular/evenly spread",
    "cross": "low = closer to the partner than chance, high = farther",
}


# =============================================================================
# Dataset
# =============================================================================

@dataclass
class NullDataset:
    """Several projects' exports, pooled and keyed consistently."""
    images: pd.DataFrame                  # one row per image
    observed: pd.DataFrame                # one row per real object
    null: pd.DataFrame                    # one row per (image, draw, object)
    # Reverse direction: per FIXED partner object. Different quantity, not a
    # different view -- see per_image_effects(direction=...).
    observed_partners: pd.DataFrame = field(default_factory=pd.DataFrame)
    null_partners: pd.DataFrame = field(default_factory=pd.DataFrame)
    manifests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    f_grid: Optional[np.ndarray] = None
    f_curves: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (f"NullDataset({len(self.manifests)} projects, "
                f"{len(self.images)} images, {len(self.observed):,} observed "
                f"objects, {len(self.null):,} null rows)")

    @property
    def projects(self) -> List[str]:
        return sorted(self.manifests)


def discover_runs(project_roots: Sequence[str]) -> "pd.DataFrame":
    """Every named run under each project, with its pairing.

    A project holds one run per pairing -- randomise A against C, B against C,
    A inside B -- so the first job in a notebook is usually to see what exists
    and pick the same pairing from each replicate.

    Accepts either a project root (its SPATIAL_NULL is found) or a SPATIAL_NULL
    directory itself.
    """
    rows: List[Dict[str, Any]] = []
    for root in project_roots:
        base = root
        if not os.path.basename(os.path.normpath(base)) == "SPATIAL_NULL":
            candidate = os.path.join(root, "SPATIAL_NULL")
            if os.path.isdir(candidate):
                base = candidate
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            man_path = os.path.join(d, "manifest.json")
            if not os.path.isfile(man_path):
                continue
            try:
                with open(man_path) as fh:
                    m = json.load(fh)
            except Exception:
                continue
            rows.append({
                "project": m.get("project") or os.path.basename(
                    os.path.dirname(os.path.normpath(base))),
                "run_name": m.get("run_name") or name,
                "path": d,
                "primary": m.get("primary_name"),
                "partner": m.get("partner_name"),
                "domain": m.get("domain_choice"),
                "domain_source": (m.get("comparability_key") or {}).get(
                    "domain_source"),
                "roi": m.get("roi_name") or "",
                "n_images": m.get("n_images"),
                "n_reference": m.get("n_reference"),
                "n_test": m.get("n_test"),
                "created": m.get("created"),
            })
    return pd.DataFrame(rows)


def matching_runs(runs: "pd.DataFrame", primary: str,
                  partner: Optional[str] = None,
                  domain: Optional[str] = None) -> List[str]:
    """Paths of the runs describing one pairing, ready for `load_projects`."""
    sel = runs[runs["primary"] == primary]
    if partner is not None:
        sel = sel[sel["partner"] == partner]
    if domain is not None:
        sel = sel[sel["domain"] == domain]
    return list(sel["path"])


def _load_one(path: str) -> Dict[str, Any]:
    """Read one export directory into raw parts."""
    man_path = os.path.join(path, "manifest.json")
    if not os.path.isfile(man_path):
        raise FileNotFoundError(f"no manifest.json in {path}")
    with open(man_path) as fh:
        manifest = json.load(fh)

    images = pd.read_csv(os.path.join(path, "image_metadata.csv"))
    observed = pd.read_csv(os.path.join(path, "observed_objects.csv"))

    npz_path = os.path.join(path, "null_objects.npz")
    if os.path.isfile(npz_path):
        # allow_pickle stays False: the export casts string columns to
        # fixed-width unicode precisely so loading never needs it.
        with np.load(npz_path, allow_pickle=False) as z:
            null = pd.DataFrame({k: z[k] for k in z.files})
    else:
        gz = os.path.join(path, "null_objects.csv.gz")
        null = pd.read_csv(gz) if os.path.isfile(gz) else pd.DataFrame()

    obs_partners = pd.DataFrame()
    q = os.path.join(path, "observed_partners.csv")
    if os.path.isfile(q):
        obs_partners = pd.read_csv(q)

    null_partners = pd.DataFrame()
    q = os.path.join(path, "null_partners.npz")
    if os.path.isfile(q):
        with np.load(q, allow_pickle=False) as z:
            null_partners = pd.DataFrame({k: z[k] for k in z.files})
    else:
        q = os.path.join(path, "null_partners.csv.gz")
        if os.path.isfile(q):
            null_partners = pd.read_csv(q)

    curves: Dict[str, Any] = {}
    f_path = os.path.join(path, "f_curves.npz")
    if os.path.isfile(f_path):
        with np.load(f_path, allow_pickle=False) as z:
            curves = {k: z[k] for k in z.files}
    if "image_ids" in curves:
        curves["image_ids"] = curves["image_ids"].astype(str)

    return {"manifest": manifest, "images": images, "observed": observed,
            "null": null, "curves": curves,
            "observed_partners": obs_partners, "null_partners": null_partners}


def _check_comparability(manifests: Dict[str, Dict[str, Any]]) -> List[str]:
    """Refuse hard mismatches; warn on the recoverable ones."""
    if len(manifests) < 2:
        return []
    keys = {name: (m.get("comparability_key") or {}) for name, m in manifests.items()}
    ref_name = next(iter(keys))
    ref = keys[ref_name]
    problems, notes = [], []
    for name, key in keys.items():
        if name == ref_name:
            continue
        for k in set(ref) | set(key):
            a, b = ref.get(k), key.get(k)
            if a == b:
                continue
            msg = f"{k}: {ref_name}={a!r} vs {name}={b!r}"
            (notes if k in SOFT_KEYS else problems).append(msg)
    if problems:
        pairing = [m for m in problems if m.split(":")[0] in
                   ("primary_name", "partner_name", "domain_choice")]
        hint = ""
        if pairing:
            hint = ("\n\nThese are different PAIRINGS, not different replicates "
                    "of one pairing. Use discover_runs() to see what each "
                    "project contains, then matching_runs() to select the same "
                    "pairing from each.")
        raise ValueError(
            "Refusing to pool exports whose nulls were defined differently -- "
            "the result would not mean anything. Mismatched keys:\n  "
            + "\n  ".join(problems) + hint)
    return notes


def _resample_curves(curves: Dict[str, Any], target: np.ndarray) -> Dict[str, Any]:
    """Interpolate CDF blocks onto a common grid."""
    src = np.asarray(curves["grid"], dtype=float)
    out = dict(curves)
    for key in ("observed", "reference", "null"):
        if key not in curves:
            continue
        block = np.atleast_2d(np.asarray(curves[key], dtype=float))
        out[key] = np.stack([np.interp(target, src, row) for row in block]
                            ).astype(np.float32)
    out["grid"] = target.astype(np.float32)
    return out


def load_projects(paths: Sequence[str],
                  group_from_name: Optional[str] = None,
                  group_delimiter: str = "_",
                  group_field: int = 0,
                  project_names: Optional[Sequence[str]] = None) -> NullDataset:
    """Load and pool several export directories.

    Args:
        paths: export directories (each containing manifest.json).
        group_from_name: regex applied to the IMAGE name; its first capturing
            group becomes `group` (e.g. r"(WT|KO)" for a condition). When None,
            the token at `group_field` after splitting on `group_delimiter`.
        project_names: overrides the manifest's project name, one per path.
    """
    parts, manifests, notes = [], {}, []
    for i, path in enumerate(paths):
        raw = _load_one(path)
        name = (project_names[i] if project_names is not None
                else raw["manifest"].get("project") or os.path.basename(
                    os.path.normpath(path)))
        if name in manifests:
            name = f"{name}#{i}"
        raw["project"] = name
        manifests[name] = raw["manifest"]
        parts.append(raw)

    notes += _check_comparability(manifests)
    for note in notes:
        warnings.warn(f"Recoverable mismatch, curves will be interpolated -- {note}",
                      RuntimeWarning, stacklevel=2)

    images, observed, nulls = [], [], []
    obs_p, null_p = [], []
    for raw in parts:
        for frame, sink in ((raw["images"], images),
                            (raw["observed"], observed),
                            (raw["null"], nulls),
                            (raw["observed_partners"], obs_p),
                            (raw["null_partners"], null_p)):
            if frame is None or frame.empty:
                continue
            f = frame.copy()
            f.insert(0, "project", raw["project"])
            if "run_name" not in f.columns:
                f.insert(1, "run_name", raw["manifest"].get("run_name") or "")
            sink.append(f)

    images_df = pd.concat(images, ignore_index=True) if images else pd.DataFrame()
    observed_df = pd.concat(observed, ignore_index=True) if observed else pd.DataFrame()
    null_df = pd.concat(nulls, ignore_index=True) if nulls else pd.DataFrame()
    obs_p_df = pd.concat(obs_p, ignore_index=True) if obs_p else pd.DataFrame()
    null_p_df = pd.concat(null_p, ignore_index=True) if null_p else pd.DataFrame()

    # Group label. Derived from the image name so one project can hold several
    # conditions, and defaulting to the project when no image name is present.
    def _group(name: str) -> str:
        if group_from_name:
            m = re.search(group_from_name, str(name))
            if m:
                return m.group(1) if m.groups() else m.group(0)
            return "ungrouped"
        toks = str(name).split(group_delimiter)
        return toks[group_field] if 0 <= group_field < len(toks) else "ungrouped"

    for frame in (images_df, observed_df, null_df, obs_p_df, null_p_df):
        if frame.empty:
            continue
        basis = frame["sample"] if "sample" in frame.columns else frame["project"]
        frame["group"] = [_group(v) for v in basis]

    # F curves onto one grid.
    curve_sets = [(raw["project"], raw["curves"]) for raw in parts if raw["curves"]]
    f_grid, merged = None, {}
    if curve_sets:
        grids = [np.asarray(c["grid"], dtype=float) for _, c in curve_sets]
        widest = max(grids, key=lambda g: (g[-1], g.size))
        f_grid = widest
        for proj, c in curve_sets:
            merged[proj] = (c if np.array_equal(np.asarray(c["grid"], float), widest)
                            else _resample_curves(c, widest))

    return NullDataset(images=images_df, observed=observed_df, null=null_df,
                       observed_partners=obs_p_df, null_partners=null_p_df,
                       manifests=manifests, f_grid=f_grid, f_curves=merged,
                       warnings=notes)


# =============================================================================
# Recomputing effects from raw data
# =============================================================================

_AGG = {"median": np.median, "mean": np.mean, "min": np.min, "max": np.max}


def per_image_effects(ds: NullDataset,
                      index: str = "cross",
                      statistic: str = "median",
                      direction: str = "primary",
                      value_col: Optional[str] = None,
                      size_min: Optional[float] = None,
                      size_max: Optional[float] = None,
                      size_col: str = "physical_size",
                      reference_set: int = 0,
                      test_set: int = 1) -> pd.DataFrame:
    """Recompute per-image effects and indices from the raw tables.

    None of this needs the Monte-Carlo re-run, which is the point of exporting
    per-object rows. The summary statistic, the size cutoff, the direction and
    the tail convention are all decided here rather than upstream.

    `direction` selects which population the distances are summarised over:

      'primary'  one value per randomised object -- "how far is each aggregate
                 from the nearest microglia"
      'partner'  one value per fixed partner object -- "how far is each
                 microglia from the nearest aggregate"

    These are different quantities and diverge whenever the two counts differ,
    so the choice is explicit rather than defaulted silently.

    A size filter is applied to observed and null alike, matched on the object
    id, so both sides describe the same objects. Filtering only the observed
    side would compare different populations. Note that with
    `direction='partner'` the filter applies to the PARTNER objects, and it
    cannot subset the randomised population -- doing that needs a separate run
    restricted at computation time, so that the threshold is recorded in the
    manifest rather than chosen after the fact.
    """
    if direction not in ("primary", "partner"):
        raise ValueError("direction must be 'primary' or 'partner'")

    if direction == "partner":
        obs, null = ds.observed_partners, ds.null_partners
        id_col = "partner_label"
        default_value = "to_primary_um"
        if obs is None or obs.empty or null is None or null.empty:
            raise KeyError(
                "This export has no reverse-direction tables. Re-run with "
                "measure_from='both' (or 'partner'), or use "
                "direction='primary'.")
    else:
        obs, null = ds.observed, ds.null
        id_col = "template_label"
        default_value = {"cross": "cross_um", "G": "g_um"}.get(index, index)

    if value_col is None:
        value_col = default_value
    if value_col not in null.columns or value_col not in obs.columns:
        raise KeyError(f"{value_col!r} is not in both {direction} tables; "
                       f"observed has {list(obs.columns)}")
    agg = _AGG[statistic]
    if size_min is not None or size_max is not None:
        if size_col not in obs.columns:
            raise KeyError(f"{size_col!r} not in the observed {direction} table")
        keep = obs[["project", "image_id", id_col, size_col]].copy()
        if size_min is not None:
            keep = keep[keep[size_col] >= size_min]
        if size_max is not None:
            keep = keep[keep[size_col] <= size_max]
        pairs = set(map(tuple, keep[["project", "image_id", id_col]].values))
        obs = obs[[tuple(r) in pairs for r in
                   obs[["project", "image_id", id_col]].values]]
        null = null[[tuple(r) in pairs for r in
                     null[["project", "image_id", id_col]].values]]

    def _stat(v: pd.Series) -> float:
        a = pd.to_numeric(v, errors="coerce").to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        return float(agg(a)) if a.size else np.nan

    obs_stat = (obs.groupby(["project", "image_id"])[value_col]
                   .apply(_stat).rename("observed"))
    per_draw = (null.groupby(["project", "image_id", "set", "draw"])[value_col]
                    .apply(_stat).rename("value").reset_index())

    ref = per_draw[per_draw["set"] == reference_set]
    test = per_draw[per_draw["set"] == test_set]

    ref_stats = (ref.groupby(["project", "image_id"])["value"]
                    .agg(null_mean="mean", null_median="median",
                         null_sd=lambda s: float(np.std(s, ddof=1)) if len(s) > 1 else np.nan,
                         n_reference="count"))

    rows: List[Dict[str, Any]] = []
    test_groups = {k: g["value"].to_numpy(dtype=float)
                   for k, g in test.groupby(["project", "image_id"])}
    for key, o in obs_stat.items():
        if key not in ref_stats.index:
            continue
        r = ref_stats.loc[key]
        draws = test_groups.get(key, np.array([]))
        sd = float(r["null_sd"])
        rows.append({
            "project": key[0], "image_id": key[1],
            "observed": o,
            "null_mean": float(r["null_mean"]),
            "null_median": float(r["null_median"]),
            "null_sd": sd,
            "n_reference": int(r["n_reference"]),
            "n_test": int(draws.size),
            # Signed so positive always means "closer / more contact than
            # chance" for a distance statistic.
            "effect_um": float(r["null_median"]) - o,
            "effect_z": ((float(r["null_mean"]) - o) / sd
                         if np.isfinite(sd) and sd > 0 else np.nan),
            "sdi": 1.0 - _mid_p_rank(o, draws),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    meta_cols = [c for c in ("group", "sample", "occupancy_fraction",
                             "orientation_acceptance_rate", "packing_warning",
                             "placement_warning", "n_objects", "domain_source")
                 if c in ds.images.columns]
    if meta_cols:
        out = out.merge(
            ds.images[["project", "sample"] + [c for c in meta_cols if c != "sample"]]
            .rename(columns={"sample": "image_id"}),
            on=["project", "image_id"], how="left", suffixes=("", "_meta"))
    out.attrs["index"] = index
    out.attrs["statistic"] = statistic
    out.attrs["direction"] = direction
    return out


def _mid_p_rank(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return np.nan
    greater = float(np.sum(null > observed))
    equal = float(np.sum(null == observed))
    return (greater + 0.5 * equal + 0.5) / (null.size + 1.0)


# =============================================================================
# Inference
# =============================================================================

def replicate_test(effects: pd.DataFrame,
                   effect_col: str = "effect_z",
                   group_col: str = "group",
                   replicate_col: str = "project",
                   sdi_col: str = "sdi") -> pd.DataFrame:
    """Inference with the biological replicate as the unit.

    Images inside a project are technical replicates, so each project collapses
    to one number (the median of its images' effects) before testing. Testing
    images directly would inflate n by the images-per-project factor and turn a
    batch effect into a p-value.

    Reported alongside: a per-replicate KS of that replicate's image indices
    against Uniform(0,1). That is the paper's test applied at the level where it
    is defensible -- images within one session are far closer to exchangeable
    than images across animals. A POOLED KS across replicates is deliberately
    not offered here, because it is pseudoreplicated.

    Disagreements are diagnosed rather than merely flagged, since the patterns
    mean different things (see the `diagnosis` column).
    """
    from scipy.stats import kstest, ttest_1samp

    rows: List[Dict[str, Any]] = []
    grouper = ([group_col] if group_col in effects.columns else [])

    for gkey, block in (effects.groupby(grouper) if grouper
                        else [("all", effects)]):
        gname = gkey if isinstance(gkey, str) else (gkey[0] if gkey else "all")

        per_rep = (block.groupby(replicate_col)[effect_col]
                        .median().dropna())
        vals = per_rep.to_numpy(dtype=float)
        k = vals.size
        mean_e = float(np.mean(vals)) if k else np.nan
        sd_e = float(np.std(vals, ddof=1)) if k >= 2 else np.nan

        if k >= 2 and np.isfinite(sd_e) and sd_e > 0:
            t, p = ttest_1samp(vals, 0.0)
            dz = mean_e / sd_e
            half = 1.96 * sd_e / np.sqrt(k)
            ci = (mean_e - half, mean_e + half)
        else:
            t = p = dz = np.nan
            ci = (np.nan, np.nan)

        # Per-replicate KS on that replicate's image indices.
        ks_rows = []
        for rep, rblock in block.groupby(replicate_col):
            s = pd.to_numeric(rblock.get(sdi_col), errors="coerce").dropna().to_numpy()
            if s.size >= 3:
                D, pk = kstest(s, "uniform")
                ks_rows.append({"replicate": rep, "n_images": int(s.size),
                                "ks_D": float(D), "ks_p": float(pk)})
            else:
                ks_rows.append({"replicate": rep, "n_images": int(s.size),
                                "ks_D": np.nan, "ks_p": np.nan})
        ks_df = pd.DataFrame(ks_rows)
        n_ks_sig = int((ks_df["ks_p"] < 0.05).sum()) if not ks_df.empty else 0
        signs = np.sign(vals[np.isfinite(vals)])
        consistent = bool(signs.size and np.all(signs == signs[0]))

        t_sig = bool(np.isfinite(p) and p < 0.05)
        ks_any = n_ks_sig > 0
        if ks_any and not t_sig and consistent:
            diagnosis = ("underpowered at the replicate level: departure is "
                         "detectable per replicate and the direction agrees, "
                         "so more replicates would likely resolve it")
        elif ks_any and not t_sig and not consistent:
            diagnosis = ("between-replicate heterogeneity: replicates depart "
                         "from the null in OPPOSITE directions -- do not claim "
                         "the effect, investigate batch")
        elif t_sig and not ks_any:
            diagnosis = ("small but consistent directional shift; a directional "
                         "test detects it where KS does not")
        elif t_sig and ks_any:
            diagnosis = "consistent: both the replicate test and per-replicate KS agree"
        else:
            diagnosis = ("no evidence -- check power, occupancy and placement "
                         "diagnostics before reading this as absence")

        rows.append({
            "group": gname, "index": effects.attrs.get("index", ""),
            "effect_col": effect_col,
            "n_replicates": k, "n_images": int(len(block)),
            "mean_effect": mean_e, "sd_effect": sd_e,
            "ci95_low": ci[0], "ci95_high": ci[1],
            "t_stat": float(t) if np.isfinite(t) else np.nan,
            "p_value": float(p) if np.isfinite(p) else np.nan,
            "cohens_dz": float(dz) if np.isfinite(dz) else np.nan,
            "n_replicates_ks_sig": n_ks_sig,
            "direction_consistent": consistent,
            "diagnosis": diagnosis,
            "power_note": ("n<=3 replicates: low power, and 'not significant' "
                           "is not evidence of absence" if k <= 3 else ""),
        })
    return pd.DataFrame(rows)


def compare_groups(effects: pd.DataFrame,
                   effect_col: str = "effect_z",
                   group_col: str = "group",
                   replicate_col: str = "project") -> pd.DataFrame:
    """Between-group comparison on per-replicate values (Welch t)."""
    from scipy.stats import ttest_ind
    if group_col not in effects.columns:
        return pd.DataFrame()
    per_rep = (effects.groupby([group_col, replicate_col])[effect_col]
                      .median().reset_index())
    groups = sorted(per_rep[group_col].dropna().unique())
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            va = per_rep.loc[per_rep[group_col] == a, effect_col].dropna().to_numpy()
            vb = per_rep.loc[per_rep[group_col] == b, effect_col].dropna().to_numpy()
            if va.size >= 2 and vb.size >= 2:
                t, p = ttest_ind(va, vb, equal_var=False)
            else:
                t = p = np.nan
            pooled = np.sqrt((np.var(va, ddof=1) + np.var(vb, ddof=1)) / 2) \
                if va.size >= 2 and vb.size >= 2 else np.nan
            rows.append({
                "group_a": a, "group_b": b,
                "n_a": va.size, "n_b": vb.size,
                "mean_a": float(np.mean(va)) if va.size else np.nan,
                "mean_b": float(np.mean(vb)) if vb.size else np.nan,
                "t_stat": float(t) if np.isfinite(t) else np.nan,
                "p_value": float(p) if np.isfinite(p) else np.nan,
                "cohens_d": (float((np.mean(va) - np.mean(vb)) / pooled)
                             if np.isfinite(pooled) and pooled > 0 else np.nan),
            })
    return pd.DataFrame(rows)


def quality_report(ds: NullDataset) -> pd.DataFrame:
    """Per-image reasons an result might not be interpretable.

    Read this before the statistics. High occupancy means a hardcore null is
    forced toward regularity regardless of biology; a low orientation
    acceptance rate means the domain boundary constrained which orientations
    were possible; unplaced objects mean the field was too packed to realise
    the null at all.
    """
    cols = [c for c in ("project", "sample", "group", "n_objects", "n_partner",
                        "domain_source", "occupancy_fraction",
                        "orientation_acceptance_rate", "mean_unplaced_per_draw",
                        "draws_incomplete", "packing_warning",
                        "placement_warning") if c in ds.images.columns]
    out = ds.images[cols].copy() if cols else pd.DataFrame()
    if out.empty:
        return out
    flags = []
    for _, r in out.iterrows():
        f = []
        if r.get("packing_warning"):
            f.append("high occupancy: null forced toward regularity")
        if float(r.get("orientation_acceptance_rate") or 1.0) < 0.5:
            f.append("orientation strongly constrained by the domain")
        if r.get("placement_warning"):
            f.append("some objects could not be placed")
        if int(r.get("n_objects") or 0) < 5:
            f.append("very few objects: index is coarse")
        flags.append("; ".join(f))
    out["concerns"] = flags
    return out