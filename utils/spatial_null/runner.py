"""
null_runner.py -- HIBACHI-facing driver for the mask-preserving spatial null.

Sits between the analyzer UI and `synthetic_null`'s engine. Responsibilities:

  * resolve each sample's masks, geometry, domain and ROI crop
  * derive the run's SHARED F grid in a cheap first pass, before any
    Monte-Carlo, so every image and draw shares one coordinate frame
  * run the per-sample null and collect tidy per-object frames
  * write the export artifacts and manifest
  * optionally hand back the first draw's labels for visual verification

The application deliberately produces no inference. A HIBACHI project is
typically one biological replicate whose images are technical replicates, so the
only defensible tests live downstream across projects (see `hibachi_null_io`).
What is produced here is raw material plus the diagnostics needed to judge
whether an image is interpretable at all.
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

from .engine import (
    Domain, build_domain, derive_f_grid, f_grid_probe, monte_carlo_null,
    describe_within_project, extract_templates,
)
from .qc_render import (
    estimate_qc_output, qc_paths, render_draw, render_observed,
    self_test as qc_self_test,
)
from .export import (
    build_manifest, concat_frames, concat_null_frames, concat_observed_frames,
    stack_f_curves, write_project_export,
)

DEFAULT_N_REFERENCE = 199
DEFAULT_N_TEST = 199


# =============================================================================
# Sample resolution
# =============================================================================

def _is_roi_dir(name: str) -> bool:
    base = os.path.basename(str(name).rstrip("/\\"))
    return base.endswith("_roi") or "_roi_" in base


def find_final_segmentation(sample_dir: str,
                            roi_name: Optional[str] = None) -> Optional[str]:
    """`final_segmentation*.dat` for a sample, honouring a region.

    Full-image sessions are preferred explicitly. An ROI session directory also
    contains "_processed_", so a naive scan can return the CROP's segmentation
    for a channel that happens to have a region -- a different array with a
    different shape from the one the caller is memmapping against.
    """
    if not sample_dir or not os.path.isdir(sample_dir):
        return None

    if roi_name:
        try:
            from ..high_level_gui.roi_sharing import roi_session_dir
        except ImportError:
            return None
        rdir = roi_session_dir(sample_dir, roi_name)
        if not rdir or not os.path.isdir(rdir):
            return None
        for f in sorted(os.listdir(rdir)):
            if f.startswith("final_segmentation") and f.endswith(".dat"):
                return os.path.join(rdir, f)
        return None

    try:
        entries = sorted(os.listdir(sample_dir))
    except OSError:
        return None
    for d in entries:
        full = os.path.join(sample_dir, d)
        if "_processed_" not in d or not os.path.isdir(full) or _is_roi_dir(d):
            continue
        for f in sorted(os.listdir(full)):
            if f.startswith("final_segmentation") and f.endswith(".dat"):
                return os.path.join(full, f)
    return None


def roi_crop_spec(sample_dir: str, roi_name: str,
                  full_shape: Tuple[int, ...]) -> Optional[Dict[str, Any]]:
    """Everything needed to crop a FULL-IMAGE array the way the masks were cropped.

    A region is a bounding box AND a per-slice polygon. Cropping by the box
    alone would admit the corners between box and polygon, where the masks were
    forcibly zeroed -- no real object can exist there, so including them would
    inflate the domain and bias F toward apparent clustering.

    Uses `roi_sharing`'s own record and polygon accessors, and reproduces
    `build_crop_memmap`'s nearest-defined-polygon rule, so the domain and the
    masks provably share one definition instead of drifting apart.
    """
    try:
        from ..high_level_gui.roi_sharing import (roi_session_dir,
                                                  load_roi_record,
                                                  record_polygons)
    except ImportError:
        return None

    rdir = roi_session_dir(sample_dir, roi_name)
    if not rdir:
        return None
    record = load_roi_record(rdir)
    if not record:
        return None
    bbox = record.get("bbox") or {}
    polys = record_polygons(record)
    if not bbox or not polys:
        return None

    is_3d = len(full_shape) == 3
    y0 = int(bbox.get("y0") or 0)
    x0 = int(bbox.get("x0") or 0)
    y1 = int(bbox.get("y1") or (full_shape[1] if is_3d else full_shape[0]))
    x1 = int(bbox.get("x1") or (full_shape[2] if is_3d else full_shape[1]))
    crop_h, crop_w = y1 - y0, x1 - x0

    try:
        from skimage.draw import polygon as skpoly
    except ImportError:
        return None

    slice_masks: Dict[int, np.ndarray] = {}
    for z, poly in polys.items():
        p = np.asarray(poly, dtype=float) - np.array([y0, x0], dtype=float)
        rr, cc = skpoly(p[:, 0], p[:, 1], shape=(crop_h, crop_w))
        m = np.zeros((crop_h, crop_w), dtype=bool)
        m[rr, cc] = True
        slice_masks[int(z)] = m

    return {"bbox": {"z0": int(bbox.get("z0") or 0),
                     "z1": (int(bbox["z1"]) if bbox.get("z1") is not None
                            else (full_shape[0] if is_3d else None)),
                     "y0": y0, "y1": y1, "x0": x0, "x1": x1},
            "slice_masks": slice_masks,
            "full_shape": tuple(int(v) for v in full_shape)}


@dataclass
class SampleJob:
    """One image's resolved inputs."""
    sample: str
    shape: Tuple[int, ...]
    spacing: Tuple[float, ...]
    primary_path: str
    partner_path: Optional[str] = None
    primary_dir: Optional[str] = None
    mode: Optional[str] = None
    roi_crop: Optional[Dict[str, Any]] = None
    # Segmentations of the channels that can serve as a parent-object domain.
    # Needed because "inside channel A" is a different null from "inside the
    # tissue", and the domain has to be built from A's actual masks.
    domain_a_path: Optional[str] = None
    domain_b_path: Optional[str] = None
    # When the objects to randomise are an OVERLAP rather than a channel. The
    # intersection is recomputed here from the two segmentations, so the spatial
    # null does not depend on a relational batch having been run first.
    primary_kind: str = "channel"     # 'channel' | 'intersection' | 'recipe'
    intersect_a_path: Optional[str] = None
    intersect_b_path: Optional[str] = None
    intersect_label_mode: str = "connected"
    intersect_preserve_ids: bool = False
    # A resolved chain of mask-producing recipe steps. When present it takes
    # precedence, so "intersect then size-filter" is expressible without the
    # runner needing a special case per combination.
    primary_steps: List[Dict[str, Any]] = field(default_factory=list)

    def load_primary(self) -> np.ndarray:
        """The objects to randomise: a channel, an overlap, or a recipe result."""
        if self.primary_steps:
            return evaluate_primary_steps(self)
        if self.primary_kind != "intersection":
            return self.load(self.primary_path)
        a = self.load(self.intersect_a_path)
        b = self.load(self.intersect_b_path)
        return intersection_labels(a, b, self.intersect_label_mode,
                                   self.intersect_preserve_ids)

    def load(self, path: str) -> np.ndarray:
        return np.array(np.memmap(path, dtype=np.int32, mode="r",
                                  shape=self.shape))


# =============================================================================
# Run
# =============================================================================

def _count_disconnected(labels: np.ndarray) -> int:
    """How many labels consist of more than one connected piece."""
    values = np.unique(labels[labels > 0])
    if values.size == 0:
        return 0
    struct = ndimage.generate_binary_structure(labels.ndim, labels.ndim)
    slices = ndimage.find_objects(labels)
    n = 0
    for lbl in values:
        i = int(lbl) - 1
        if i < 0 or i >= len(slices) or slices[i] is None:
            continue
        _, count = ndimage.label(labels[slices[i]] == lbl, structure=struct)
        if count > 1:
            n += 1
    return n


def intersection_labels(a: np.ndarray, b: np.ndarray,
                        label_mode: str = "connected",
                        preserve_ids: bool = False) -> np.ndarray:
    """Overlap of two label images, labelled as the recipe's mode specifies.

    Mirrors `RelationalEngine.intersect_masks`. One deliberate difference:
    'binary' is promoted to connected components. That mode writes a single
    label over every overlap region, which as a randomisation template would be
    one enormous disconnected "object" moved rigidly as a unit -- almost
    certainly not what is wanted, and it would make the null meaningless.
    """
    overlap = (a > 0) & (b > 0)
    out = np.zeros(a.shape, dtype=np.int32)
    if not overlap.any():
        return out

    if label_mode == "parent_a":
        out[:] = np.where(overlap, a, 0)
    elif label_mode == "parent_b":
        out[:] = np.where(overlap, b, 0)
    else:                                   # 'connected', or 'binary' promoted
        labelled, _ = ndimage.label(
            overlap, structure=ndimage.generate_binary_structure(a.ndim, a.ndim))
        return labelled.astype(np.int32)

    if not preserve_ids:
        # Relabel 1..N: inherited ids can be non-contiguous, and a parent
        # contributing two separate overlap fragments would otherwise be one
        # object that cannot be moved rigidly.
        labelled, _ = ndimage.label(
            out > 0, structure=ndimage.generate_binary_structure(a.ndim, a.ndim))
        return labelled.astype(np.int32)
    return out


def _describe_step(step: Dict[str, Any]) -> str:
    """Short human description of one resolved primary step, for the log."""
    kind = step.get("type")
    if kind == "channel":
        return "channel"
    if kind == "intersect":
        return f"intersect({step.get('label_mode') or 'connected'})"
    if kind == "filter":
        lo = step.get("min_size") or 0
        hi = step.get("max_size")
        return f"filter(>={lo:g}" + (f", <={hi:g})" if hi else ")")
    return str(kind)


def filter_labels_by_size(labels: np.ndarray,
                          spacing: Sequence[float],
                          min_size: float = 0.0,
                          max_size: Optional[float] = None) -> np.ndarray:
    """Keep objects within a physical size range, relabelled 1..N.

    Mirrors `RelationalEngine.filter_by_volume`, including its one surprising
    behaviour: the incoming labels are DISCARDED and connected components
    re-derived from the binary mask, so objects that touch are merged before the
    size test. Reproducing that matters -- if this filtered differently from the
    recipe step, the randomised population would not be the population the
    recipe's own analysis reports on.

    `min_size`/`max_size` are um^2 in 2D and um^3 in 3D, matching the recipe's
    "Min Volume" prompt.
    """
    struct = ndimage.generate_binary_structure(labels.ndim, labels.ndim)
    comps, _ = ndimage.label(labels > 0, structure=struct)
    if comps.max() == 0:
        return np.zeros_like(labels, dtype=np.int32)

    unit = float(np.prod(spacing))
    counts = np.bincount(comps.reshape(-1))
    sizes = counts * unit

    keep = np.zeros(counts.size, dtype=bool)
    keep[1:] = sizes[1:] >= float(min_size)
    if max_size is not None:
        keep[1:] &= sizes[1:] <= float(max_size)

    # Sequential relabelling, as the recipe does after a volume filter.
    lut = np.zeros(counts.size, dtype=np.int32)
    lut[np.flatnonzero(keep)] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return lut[comps]


def evaluate_primary_steps(job: "SampleJob") -> np.ndarray:
    """Build the objects to randomise by walking a resolved recipe program.

    Each step consumes the previous result, exactly as the cross-channel recipe
    does, so a chain such as "intersect pS129 with MAP2, then keep objects above
    2 um^2" produces the same population the recipe would.

    Steps (paths already resolved when the job was built):
        {'type': 'channel',   'path': ...}
        {'type': 'intersect', 'path_a':, 'path_b':, 'label_mode':, 'preserve_ids':}
        {'type': 'filter',    'min_size':, 'max_size': (optional)}
    """
    current = None
    for step in job.primary_steps:
        kind = step.get("type")
        if kind == "channel":
            current = job.load(step["path"])
        elif kind == "intersect":
            a = job.load(step["path_a"]) if step.get("path_a") else current
            b = job.load(step["path_b"])
            if a is None:
                raise ValueError("intersect step has no first input")
            current = intersection_labels(a, b,
                                          step.get("label_mode") or "connected",
                                          bool(step.get("preserve_ids")))
        elif kind == "filter":
            if current is None:
                raise ValueError("filter step has nothing to filter")
            current = filter_labels_by_size(current, job.spacing,
                                            float(step.get("min_size") or 0.0),
                                            step.get("max_size"))
        else:
            raise ValueError(f"unsupported primary step {kind!r}")
    if current is None:
        raise ValueError("empty primary program")
    return current


def _parent_domain(job: "SampleJob", choice: str):
    """(mask, parent_labels, reason) for a parent-object domain.

    Returns (None, None, reason) when the required channel's segmentation is
    missing, so the caller can skip the sample rather than quietly substituting
    a different domain -- and therefore a different hypothesis.
    """
    def _load(path):
        if not path:
            return None
        try:
            return np.array(np.memmap(path, dtype=np.int32, mode="r",
                                      shape=job.shape))
        except (ValueError, OSError):
            return None

    a = _load(job.domain_a_path)
    b = _load(job.domain_b_path)

    if choice == "parent_a":
        if a is None:
            return None, None, "domain channel A has no segmentation"
        return a > 0, a, ""
    if choice == "parent_b":
        if b is None:
            return None, None, "domain channel B has no segmentation"
        return b > 0, b, ""
    if choice == "parent_both":
        if a is None or b is None:
            return None, None, "both domain channels are needed for A and B"
        mask = (a > 0) & (b > 0)
        if not mask.any():
            return None, None, "channels A and B do not overlap"
        # Label the overlap itself, so per-parent containment means "inside one
        # connected overlap region" rather than "inside one A object".
        labelled, _ = ndimage.label(mask,
                                    structure=ndimage.generate_binary_structure(
                                        mask.ndim, mask.ndim))
        return mask, labelled.astype(np.int32), ""
    return None, None, f"unknown domain choice {choice!r}"


@dataclass
class RunParameters:
    """Everything that defines the null. Recorded verbatim in the manifest."""
    n_reference: int = DEFAULT_N_REFERENCE
    n_test: int = DEFAULT_N_TEST
    rotate: bool = True
    hardcore: bool = True
    min_separation_um: float = 0.0
    use_hull: bool = True
    # 'hull' | 'field' | 'parent_a' | 'parent_b' | 'parent_both'. Kept explicit
    # rather than inferred from use_hull, because the parent options need the
    # domain built from another channel's masks.
    domain_choice: str = "hull"
    per_parent_containment: bool = False
    erode_um: float = 0.0
    compute_f: bool = True
    compute_g: bool = True
    cross_statistic: str = "median"
    # 'primary'  distances FROM each randomised object to the nearest partner
    # 'partner'  distances FROM each fixed partner to the nearest randomised
    # 'both'     compute both (default; one extra transform per draw)
    measure_from: str = "both"
    # Which direction drives the reported index and the QC segments.
    statistic_direction: str = "primary"
    max_attempts: int = 2000
    seed: Optional[int] = 0
    grid_points: int = 512
    roi_name: Optional[str] = None
    keep_first_draw: bool = True
    also_csv: bool = False
    # QC images: one JPG per draw, plus one of the observed data per sample.
    # Defaults to 0 because the count multiplies by samples -- 398 draws across
    # 20 images is ~8,000 files. The caller is warned with an estimate.
    n_qc_images: int = 0
    qc_annotate_distances: bool = True
    # Names this pairing on disk, so a project can hold many: randomise A
    # against C, randomise B against C, randomise A inside B, and so on.
    run_name: str = ""
    # Human-readable summary of the recipe program that produced the randomised
    # objects, e.g. "pS129 -> filter(>=2 um^2)". Stored in the manifest.
    primary_program: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def run_project(jobs: Sequence[SampleJob],
                params: Optional[RunParameters] = None,
                out_dir: Optional[str] = None,
                project_name: str = "project",
                channels: Optional[Dict[str, Any]] = None,
                explicit_domains: Optional[Dict[str, np.ndarray]] = None,
                log: Callable[[str], None] = print,
                progress: Optional[Any] = None,
                progress_cb: Optional[Callable[..., None]] = None,
                cancel_check: Optional[Callable[[], bool]] = None
                ) -> Dict[str, Any]:
    """Run the null across a project's samples and write the export.

    Two passes on purpose. The first only probes each image for one scalar so
    the run's shared F grid can be derived before any Monte-Carlo starts; a grid
    derived per image would give every image a different x-axis and the curves
    could never be pooled, which is the whole point of exporting them.
    """
    params = params or RunParameters()
    channels = channels or {}
    rng = np.random.default_rng(params.seed)

    # ---- pass 1: geometry, domains, F-grid probes ---------------------------
    prepared: List[Dict[str, Any]] = []
    probes: List[float] = []

    def _report(**kw):
        if progress_cb is not None:
            try:
                progress_cb(**kw)
            except Exception:
                pass                       # never let the UI break the run

    for idx, job in enumerate(jobs):
        if cancel_check is not None and cancel_check():
            log("Cancelled during setup.")
            return {"n_samples": 0, "cancelled": True}
        _report(phase="prepare", sample=job.sample,
                sample_index=idx, n_samples=len(jobs))
        try:
            primary = job.load_primary()
        except (ValueError, OSError, TypeError) as exc:
            log(f"  [{job.sample}] cannot read segmentation ({exc}); skipped.")
            continue
        if not (primary > 0).any():
            log(f"  [{job.sample}] "
                + ("the two channels do not overlap" if job.primary_kind ==
                   "intersection" else "no objects") + "; skipped.")
            continue
        if job.primary_kind == "recipe":
            log(f"  [{job.sample}] recipe program gives "
                f"{int(np.unique(primary[primary > 0]).size)} object(s): "
                + " -> ".join(_describe_step(st) for st in job.primary_steps))
        if job.primary_kind == "intersection":
            log(f"  [{job.sample}] overlap gives "
                f"{int(np.unique(primary[primary > 0]).size)} object(s) "
                f"({job.intersect_label_mode}).")

        # A label spanning disconnected pieces is moved as ONE rigid template,
        # keeping the fragments' relative arrangement fixed. That is a coherent
        # null but a surprising one, and it arises silently from
        # parent_a/parent_b with IDs preserved, so it is reported.
        n_split = _count_disconnected(primary)
        if n_split:
            log(f"  [{job.sample}] {n_split} label(s) span disconnected "
                f"fragments and will be moved as single rigid units. If that is "
                f"not intended, use the 'connected' label mode or do not "
                f"preserve IDs.")

        explicit = (explicit_domains or {}).get(job.sample)
        parent_labels = None
        choice = str(params.domain_choice or "hull")

        if explicit is None and choice.startswith("parent"):
            explicit, parent_labels, why = _parent_domain(job, choice)
            if explicit is None:
                # Falling back silently would answer a different question from
                # the one the user asked, so the sample is skipped instead.
                log(f"  [{job.sample}] {why}; skipped.")
                continue

        domain = build_domain(
            shape=job.shape, spacing=job.spacing,
            channel_sample_dir=job.primary_dir, mode=job.mode,
            use_hull=(choice == "hull"), roi_crop=job.roi_crop,
            explicit_mask=explicit, erode_um=params.erode_um)
        if parent_labels is not None:
            domain.parent_labels = parent_labels
            domain.source = choice
            domain.diagnostics["parent_objects"] = int(
                np.unique(parent_labels[parent_labels > 0]).size)

        if domain.voxels == 0:
            log(f"  [{job.sample}] empty domain; skipped.")
            continue
        # Objects outside the domain cannot be reproduced by any draw, so the
        # observed and null populations would differ. Report rather than hide.
        outside = int(np.count_nonzero((primary > 0) & ~domain.mask))
        if outside:
            log(f"  [{job.sample}] {outside} object voxels lie OUTSIDE the "
                f"domain ({domain.source}); the null cannot reproduce them.")

        partner = None
        if job.partner_path:
            try:
                partner = job.load(job.partner_path)
            except (ValueError, OSError) as exc:
                log(f"  [{job.sample}] cannot read partner ({exc}).")

        probe = f_grid_probe(primary, domain) if params.compute_f else 0.0
        probes.append(probe)
        prepared.append({"job": job, "primary": primary, "partner": partner,
                         "domain": domain, "outside_voxels": outside,
                         "disconnected_labels": n_split})

    if not prepared:
        log("No scorable samples.")
        return {"n_samples": 0}

    total_draws = int(params.n_reference) + int(params.n_test)
    f_grid, grid_info = derive_f_grid(probes, params.grid_points)
    if params.compute_f:
        log(f"Shared F grid: 0-{grid_info['f_grid_max_um']:g} um "
            f"({grid_info['f_grid_points']} points), from a raw maximum of "
            f"{grid_info['f_grid_raw_max_um']:.3g} um across {len(probes)} images.")

    # ---- pass 2: Monte-Carlo ----------------------------------------------
    qc_failure_reason = None
    qc_error_total = 0
    results, meta_rows = [], []
    obs_frames, null_frames, curve_sets, image_ids = [], [], [], []
    obs_partner_frames, null_partner_frames = [], []
    first_draws: Dict[str, np.ndarray] = {}

    n_qc = max(0, int(params.n_qc_images))
    if n_qc and out_dir:
        # Prove the renderer works BEFORE spending the run on it. Discovering a
        # rendering problem afterwards, as an empty directory, wastes the whole
        # computation and gives no clue why.
        ok, why = qc_self_test(out_dir)
        if not ok:
            log("QC images DISABLED: the renderer failed its self-test.")
            log(f"  {why.strip()}")
            qc_failure_reason = why.strip().splitlines()[0] if why.strip() else "unknown"
            n_qc = 0
        else:
            count, mb = estimate_qc_output(len(prepared), min(n_qc, total_draws))
            log(f"QC images: up to {count} JPGs (~{mb:.0f} MB) under "
                f"{os.path.join(out_dir, 'qc_images')}.")

    for idx, item in enumerate(prepared):
        job: SampleJob = item["job"]
        if cancel_check is not None and cancel_check():
            log("Cancelled.")
            break
        log(f"  [{job.sample}] {int(np.unique(item['primary'][item['primary']>0]).size)} "
            f"objects, domain={item['domain'].source}, "
            f"{params.n_reference}+{params.n_test} draws...")
        _report(phase="run", sample=job.sample, sample_index=idx,
                n_samples=len(prepared), draw=0, n_draws=total_draws)

        qc_hook = None
        if n_qc and out_dir:
            qc_hook = _make_qc_hook(job, item, out_dir, params, n_qc, log)

        try:
            res = monte_carlo_null(
                item["primary"], item["domain"], f_grid=f_grid,
                sample=job.sample, partner_labels=item["partner"],
                n_reference=params.n_reference, n_test=params.n_test, rng=rng,
                rotate=params.rotate, hardcore=params.hardcore,
                min_separation_um=params.min_separation_um,
                compute_f=params.compute_f, compute_g=params.compute_g,
                cross_statistic=params.cross_statistic,
                measure_from=params.measure_from,
                statistic_direction=params.statistic_direction,
                keep_first_draw=params.keep_first_draw,
                max_attempts=params.max_attempts,
                per_parent_containment=params.per_parent_containment,
                qc_hook=qc_hook, n_qc_draws=(n_qc if qc_hook else 0),
                cancel_check=cancel_check,
                draw_callback=lambda i, n, _s=job.sample, _x=idx: _report(
                    phase="run", sample=_s, sample_index=_x,
                    n_samples=len(prepared), draw=i, n_draws=n),
                progress=progress)
        except Exception as exc:                       # one bad image, not the run
            log(f"  [{job.sample}] FAILED: {exc}")
            traceback.print_exc()
            continue

        row = res.metadata_row()
        row["shape"] = "x".join(str(s) for s in job.shape)
        row["spacing_um"] = ",".join(f"{s:g}" for s in job.spacing)
        row["roi_name"] = params.roi_name or ""
        row["run_name"] = params.run_name or ""
        row["primary_channel"] = channels.get("primary") or ""
        row["partner_channel"] = channels.get("partner") or ""
        row["object_voxels_outside_domain"] = item["outside_voxels"]
        row["disconnected_labels"] = item.get("disconnected_labels", 0)
        row["primary_kind"] = job.primary_kind
        meta_rows.append(row)

        results.append(res)
        image_ids.append(job.sample)
        obs_frames.append(res.observed_objects)
        null_frames.append(res.null_objects)
        obs_partner_frames.append(res.observed_partners)
        null_partner_frames.append(res.null_partners)
        curve_sets.append(res.f_curves)
        if res.first_draw_labels is not None:
            first_draws[job.sample] = res.first_draw_labels

        warn = []
        if res.diagnostics.get("packing_warning"):
            warn.append(f"occupancy {res.diagnostics['occupancy_fraction']:.1%}")
        if res.diagnostics.get("placement_warning"):
            warn.append(f"{res.diagnostics['draws_incomplete']} draws incomplete")
        if res.diagnostics.get("qc_errors"):
            qc_error_total += int(res.diagnostics["qc_errors"])
            # Reported per sample, not buried in a CSV column.
            log(f"      QC: {res.diagnostics['qc_errors']} image(s) failed to "
                f"render -- {res.diagnostics.get('qc_last_error', 'unknown')}")
        acc = res.diagnostics.get("orientation_acceptance_rate")
        if acc is not None and np.isfinite(acc) and acc < 0.5:
            warn.append(f"orientation acceptance {acc:.0%}")
        if warn:
            log(f"      concerns: {'; '.join(warn)}")

    metadata = pd.DataFrame(meta_rows)
    observed = concat_observed_frames(obs_frames, image_ids)
    nulls = concat_null_frames(null_frames, image_ids)
    obs_partners = concat_frames(obs_partner_frames, image_ids,
                                 ("image_id", "partner_label"))
    null_partners = concat_frames(null_partner_frames, image_ids,
                                  ("image_id", "draw", "set", "partner_label"))
    curves = (stack_f_curves(curve_sets, image_ids, f_grid)
              if params.compute_f else {})

    out: Dict[str, Any] = {
        "n_samples": len(results), "metadata": metadata,
        "observed_objects": observed, "null_objects": nulls,
        "observed_partners": obs_partners, "null_partners": null_partners,
        "f_curves": curves, "f_grid": f_grid, "grid_info": grid_info,
        "first_draw_labels": first_draws, "results": results,
        "qc_dir": (os.path.join(out_dir, "qc_images")
                   if (out_dir and n_qc) else None),
        "qc_requested": int(params.n_qc_images),
        "qc_disabled_reason": qc_failure_reason,
        "qc_errors": int(qc_error_total),
        "description": describe_within_project(metadata)
        if not metadata.empty else pd.DataFrame(),
    }

    if out_dir:
        ndim = len(jobs[0].shape) if jobs else 2
        # The realised domain, not the requested one: `use_hull=True` falls
        # back to the field when no hull is persisted, and pooling a hull run
        # with a field run is exactly the mistake the key exists to catch.
        realised = sorted(set(metadata["domain_source"].dropna().astype(str))) \
            if "domain_source" in metadata.columns else []
        manifest = build_manifest(
            project_name=project_name, ndim=ndim,
            parameters=params.as_dict(), grid_info=grid_info,
            channels=channels, n_images=len(results),
            run_name=params.run_name,
            extra={"domain_source": "+".join(realised) or None,
                   # Recording the program makes a size threshold part of the
                   # null's definition rather than an undocumented choice, and
                   # lets the loader tell two otherwise-identical runs apart.
                   "primary_program": params.primary_program or None,
                   "primary_kind": (jobs[0].primary_kind if jobs else None)})
        written = write_project_export(
            out_dir, manifest, metadata, observed, nulls, curves,
            also_csv=params.also_csv,
            observed_partners=obs_partners, null_partners=null_partners)
        if not out["description"].empty:
            out["description"].to_csv(
                os.path.join(out_dir, "within_project_description.csv"),
                index=False)
        out["written"] = written
        log(f"\nExported {len(results)} images -> {out_dir}")
        log("  " + "\n  ".join(f"{k}: {os.path.basename(v)}"
                               for k, v in written.items()))
        log("\nThis export contains no inference. A project is one biological "
            "replicate, so pool several with hibachi_null_io and test there.")

    return out


def _make_qc_hook(job: "SampleJob", item: Dict[str, Any], out_dir: str,
                  params: "RunParameters", n_qc: int,
                  log: Callable[[str], None]):
    """Build the per-draw QC renderer, and write the observed reference first.

    The observed image is written once per sample because without it the draws
    have nothing to be compared against.
    """
    from .engine import cross_distance_field, nearest_cross_pairs

    directory = qc_paths(out_dir, job.sample)
    domain = item["domain"]
    partner = item["partner"]
    spacing = job.spacing

    try:
        if partner is not None:
            field, indices = cross_distance_field(partner, spacing,
                                                  return_indices=True)
            obs_pairs = nearest_cross_pairs(item["primary"], field, indices,
                                            spacing)
        else:
            obs_pairs = []
        render_observed(os.path.join(directory, "000_observed.jpg"),
                        item["primary"], partner, obs_pairs, spacing,
                        domain_mask=domain.mask, sample=job.sample)
    except Exception as exc:
        import traceback
        log(f"      QC: could not render the observed image -- "
            f"{type(exc).__name__}: {exc}")
        log("      " + traceback.format_exc().replace("\n", "\n      ").strip())

    def hook(draw_index: int, set_index: int, labels, pairs):
        name = f"draw_{draw_index + 1:03d}_{'ref' if set_index == 0 else 'test'}.jpg"
        dists = [p["distance_um"] for p in pairs
                 if np.isfinite(p.get("distance_um", np.inf))]
        summary = (f"n={len(dists)} objects · median nearest "
                   f"{np.median(dists):.2f} µm" if dists else "no distances")
        render_draw(
            os.path.join(directory, name), labels, partner, pairs, spacing,
            domain_mask=domain.mask,
            title=f"{job.sample} — randomisation {draw_index + 1}",
            subtitle=summary,
            annotate_distances=params.qc_annotate_distances)

    return hook


def suggest_run_name(root: str,
                     primary: Optional[str] = None,
                     partner: Optional[str] = None,
                     domain_choice: str = "hull",
                     roi_name: Optional[str] = None,
                     direction: str = "primary") -> str:
    """Next free ordinal, with the pairing appended for legibility.

    A bare ordinal is unambiguous but useless when browsing a folder months
    later, so the pairing is folded in; the ordinal still guarantees uniqueness.
    """
    from .export import biological_name

    existing = set()
    if os.path.isdir(root):
        for name in os.listdir(root):
            if os.path.isdir(os.path.join(root, name)):
                existing.add(name)
    n = 1
    while any(x.startswith(f"{n:02d}_") or x == f"{n:02d}" for x in existing):
        n += 1

    bits = [f"{n:02d}"]
    p = biological_name(primary)
    if p:
        bits.append(p)
    if domain_choice.startswith("parent"):
        bits.append("in")
        which = {"parent_a": primary, "parent_b": partner}.get(domain_choice)
        bits.append(biological_name(which) or "parent")
    elif domain_choice == "hull":
        bits.append("in_tissue")
    q = biological_name(partner)
    if q:
        # The preposition encodes the direction, so two runs differing only in
        # direction do not collide on one folder name.
        bits += ["to" if direction == "primary" else "from", q]
    if roi_name:
        bits.append(str(roi_name))
    safe = "_".join(str(b) for b in bits if b)
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in safe)


def list_runs(root: str) -> List[Dict[str, Any]]:
    """Existing named runs under a SPATIAL_NULL directory, newest first."""
    import json
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        man = os.path.join(d, "manifest.json")
        if not os.path.isfile(man):
            continue
        try:
            with open(man) as fh:
                m = json.load(fh)
        except Exception:
            m = {}
        out.append({"run_name": name, "path": d,
                    "primary": m.get("primary_name"),
                    "partner": m.get("partner_name"),
                    "domain_choice": m.get("domain_choice"),
                    "n_images": m.get("n_images"),
                    "created": m.get("created")})
    return sorted(out, key=lambda r: str(r.get("created") or ""), reverse=True)


# =============================================================================
# Registry driver
# =============================================================================

def jobs_from_registry(sample_registry: Dict[str, Dict[str, str]],
                       primary_channel: str,
                       partner_channel: Optional[str] = None,
                       domain_a_channel: Optional[str] = None,
                       domain_b_channel: Optional[str] = None,
                       primary_intersection: Optional[Dict[str, Any]] = None,
                       primary_recipe: Optional[Sequence[Dict[str, Any]]] = None,
                       roi_name: Optional[str] = None,
                       geometry_for: Optional[Callable[[Dict[str, str], Optional[str]],
                                                       Tuple[Any, Any]]] = None,
                       log: Callable[[str], None] = print) -> List[SampleJob]:
    """Build jobs from HIBACHI's `sample_registry`.

    `geometry_for(sample_channels, roi_name) -> (shape, spacing)` should be the
    analyzer's own resolver, so this agrees with the rest of the app about what
    an ROI's shape and spacing are.
    """
    jobs: List[SampleJob] = []
    spec = primary_intersection or {}
    a_ch, b_ch = spec.get("a_channel"), spec.get("b_channel")
    recipe = list(primary_recipe or [])
    # Channels the program needs, so a sample missing any of them is skipped
    # rather than failing mid-run.
    recipe_channels = [c for st in recipe
                       for c in (st.get("channel"), st.get("channel_b"))
                       if c]

    for sample, channels in sample_registry.items():
        if recipe:
            missing = [c for c in recipe_channels if c not in channels]
            if missing:
                log(f"  [{sample}] recipe needs channel(s) "
                    f"{', '.join(missing)}; skipped.")
                continue
            primary_dir = channels[recipe_channels[0]]
        elif spec:
            if a_ch not in channels or b_ch not in channels:
                log(f"  [{sample}] missing one of the intersection channels "
                    f"({a_ch}, {b_ch}); skipped.")
                continue
            primary_dir = channels[a_ch]
        else:
            if primary_channel not in channels:
                continue
            primary_dir = channels[primary_channel]

        if geometry_for is not None:
            shape, spacing = geometry_for(channels, roi_name)
        else:
            shape, spacing = _geometry_fallback(primary_dir, roi_name)
        if shape is None or spacing is None:
            log(f"  [{sample}] no geometry"
                + (f" for region {roi_name!r}" if roi_name else "") + "; skipped.")
            continue

        def _seg_for(ch, folder):
            return find_final_segmentation(folder, roi_name)

        if recipe:
            a_path = b_path = None
            resolved, ok = [], True
            for st in recipe:
                out = dict(st)
                for key, ch_key in (("path", "channel"), ("path_a", "channel"),
                                    ("path_b", "channel_b")):
                    ch = st.get(ch_key)
                    if not ch:
                        continue
                    seg = _seg_for(ch, channels[ch])
                    if not seg:
                        log(f"  [{sample}] no final segmentation for {ch}; skipped.")
                        ok = False
                        break
                    if st.get("type") == "channel" and key == "path":
                        out["path"] = seg
                    elif st.get("type") == "intersect" and key in ("path_a", "path_b"):
                        out[key] = seg
                if not ok:
                    break
                out.pop("channel", None)
                out.pop("channel_b", None)
                resolved.append(out)
            if not ok:
                continue
            primary_path = next((r.get("path") or r.get("path_a")
                                 for r in resolved if r.get("path") or r.get("path_a")),
                                None)
        elif spec:
            resolved = []
            a_path = _seg_for(a_ch, channels[a_ch])
            b_path = _seg_for(b_ch, channels[b_ch])
            if not a_path or not b_path:
                log(f"  [{sample}] the intersection needs segmentations for both "
                    f"{a_ch} and {b_ch}; skipped.")
                continue
            primary_path = a_path
        else:
            resolved = []
            a_path = b_path = None
            primary_path = find_final_segmentation(primary_dir, roi_name)
            if not primary_path:
                log(f"  [{sample}] no final segmentation for "
                    f"{primary_channel}; skipped.")
                continue

        partner_path = None
        if partner_channel and partner_channel in channels:
            partner_path = find_final_segmentation(channels[partner_channel], roi_name)
            if not partner_path:
                log(f"  [{sample}] no final segmentation for the partner "
                    f"{partner_channel}; cross-distances will be omitted.")

        crop = None
        if roi_name:
            full_shape = _full_shape(primary_dir)
            if full_shape:
                crop = roi_crop_spec(primary_dir, roi_name, full_shape)
                if crop is None:
                    log(f"  [{sample}] region {roi_name!r} geometry unavailable; "
                        f"the hull cannot be cropped, so the field will be used.")

        def _seg(ch):
            return (find_final_segmentation(channels[ch], roi_name)
                    if ch and ch in channels else None)

        jobs.append(SampleJob(
            sample=sample, shape=tuple(int(s) for s in shape),
            spacing=tuple(float(s) for s in spacing),
            primary_path=primary_path, partner_path=partner_path,
            primary_dir=primary_dir, mode=_mode_of(primary_dir), roi_crop=crop,
            domain_a_path=_seg(domain_a_channel),
            domain_b_path=_seg(domain_b_channel),
            primary_kind=("recipe" if recipe else
                          "intersection" if spec else "channel"),
            primary_steps=resolved,
            intersect_a_path=a_path, intersect_b_path=b_path,
            intersect_label_mode=str(spec.get("label_mode") or "connected"),
            intersect_preserve_ids=bool(spec.get("preserve_ids"))))
    return jobs


def _read_config(sample_dir: str) -> Dict[str, Any]:
    import yaml
    try:
        names = sorted(f for f in os.listdir(sample_dir)
                       if f.lower().endswith((".yaml", ".yml")))
    except OSError:
        return {}
    for n in names:
        try:
            with open(os.path.join(sample_dir, n)) as fh:
                cfg = yaml.safe_load(fh) or {}
            if isinstance(cfg, dict):
                return cfg
        except Exception:
            continue
    return {}


def _mode_of(sample_dir: str) -> Optional[str]:
    m = _read_config(sample_dir).get("mode")
    return m if isinstance(m, str) else None


def _full_shape(sample_dir: str) -> Optional[Tuple[int, ...]]:
    """Spatial shape of the full image, disambiguated by the config mode."""
    import tifffile
    cfg = _read_config(sample_dir)
    is_2d = str(cfg.get("mode", "")).endswith("_2d")
    want = 2 if is_2d else 3
    try:
        tif = next(os.path.join(sample_dir, f) for f in sorted(os.listdir(sample_dir))
                   if f.lower().endswith((".tif", ".tiff")))
    except (StopIteration, OSError):
        return None
    with tifffile.TiffFile(tif) as t:
        shape = tuple(int(s) for s in t.series[0].shape)
    shape = tuple(s for s in shape if s > 1) or shape
    # Trailing axes are the spatial ones; leading axes are channels or time.
    return shape[-want:] if len(shape) >= want else None


def _geometry_fallback(sample_dir: str,
                       roi_name: Optional[str]) -> Tuple[Any, Any]:
    """(shape, spacing) when the analyzer's resolver is unavailable.

    Config dimensions are TOTAL microns, so per-voxel spacing is that divided by
    the voxel count -- the same derivation the analyzer uses.
    """
    if roi_name:
        try:
            from ..high_level_gui.roi_sharing import region_geometry
            geo = region_geometry(sample_dir, roi_name)
            return (geo["shape"], geo["spacing"]) if geo else (None, None)
        except Exception:
            return None, None

    shape = _full_shape(sample_dir)
    if shape is None:
        return None, None
    cfg = _read_config(sample_dir)
    dims = cfg.get("voxel_dimensions") or cfg.get("pixel_dimensions") or {}
    keys = ("z", "y", "x") if len(shape) == 3 else ("y", "x")
    spacing = []
    for k, n in zip(keys, shape):
        try:
            total = float(dims.get(k, 0) or 0)
        except (TypeError, ValueError):
            total = 0.0
        spacing.append(total / n if total > 0 and n else 1.0)
    return shape, tuple(spacing)