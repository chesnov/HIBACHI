"""
Soma Extraction Module (3D)
==========================

This module extracts cell bodies (somas) from 3D segmentation masks. It uses
a label-first processing strategy with early-stopping conditions to optimize
speed and memory usage, particularly for large undersegmented clumps.

Strategy:
1. Population Analysis: Determine reference volumes and thicknesses.
2. Strategy Definition: Create priority-ordered intensity and distance maps.
3. Label-First Iteration: Process each segmented object independently.
4. Spatial Tiling: Large objects are processed in overlapping 3D chunks.
5. Early Stopping: Percentile strategies are skipped if cores become too small.
6. Greedy Placement: Somas are placed based on score priority and spatial separation.
"""

import os
import gc
import math
import time
import traceback
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max  # type: ignore
from skimage.segmentation import watershed  # type: ignore
from skimage.measure import regionprops  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from tqdm import tqdm

# Shared 2D/3D primitives. Everything that genuinely differs between ranks --
# structuring elements, spacing conventions, tiling -- lives there, so this
# module carries one implementation instead of two that drift apart.
try:
    from . import resource_budget
    from .dim_utils import (
        open_worker_memmap,
        worker_memmap_handle,
        generate_tiles,
        normalise_spacing,
        pixels_from_physical,
        tile_slices,
        tile_target_contains,
    )
except ImportError:  # pragma: no cover - direct script execution
    import resource_budget
    from dim_utils import (
        open_worker_memmap,
        worker_memmap_handle,
        generate_tiles,
        normalise_spacing,
        pixels_from_physical,
        tile_slices,
        tile_target_contains,
    )


# ==== injected shared soma-fix helpers (bug #1 fragmentation, bug #2 push-apart) ====
from skimage.measure import label as _cc_label
from sklearn.decomposition import PCA


def _core_is_elongated(coords_local, spacing, max_aspect, ndim):
    """PCA elongation test with correct handling of the degenerate case.
    A near-zero smallest principal axis means an (effectively) 1-voxel-thick
    line, i.e. maximally elongated -> True. Fewer than 11 voxels -> not judged
    (False). Same rule used by the first-pass check and the recovery re-check."""
    if coords_local.shape[0] <= 10:
        return False
    try:
        cp = coords_local * np.array(spacing)
        pca = PCA(n_components=ndim).fit(cp)
        ev = np.sort(np.abs(pca.explained_variance_))[::-1]
    except Exception:
        return False
    smallest = ev[-1]
    if smallest <= 1e-12:
        return True
    return (math.sqrt(ev[0]) / math.sqrt(smallest)) > max_aspect


def _finalize_core(coords, dt_vals, spacing, min_seed_vol, max_aspect, ndim):
    """Bug #1 primitive. Given candidate voxel `coords` (N x ndim int, any
    consistent frame) and their DT values, keep only the LARGEST connected
    fragment, recompute the peak as that fragment's max-DT voxel, and re-apply
    the size + aspect gates (aspect at the SAME max_aspect). Returns
    (ok, keep_mask, peak_coord) where keep_mask is a boolean over the INPUT rows
    (so the caller can index its own aligned arrays with no coordinate matching).
    Used by the aspect-recovery path and by every tighter-percentile probe."""
    if coords.shape[0] < min_seed_vol:
        return False, None, None
    mn = coords.min(0)
    loc = coords - mn
    shp = tuple(loc.max(0) + 1)
    m = np.zeros(shp, bool)
    m[tuple(loc.T)] = True
    lab = _cc_label(m, connectivity=ndim)  # full connectivity
    if lab.max() <= 0:
        return False, None, None
    ids = lab[tuple(loc.T)]                 # fragment id per input voxel
    counts = np.bincount(ids)
    counts[0] = 0
    keep_id = int(counts.argmax())
    keep = ids == keep_id                    # boolean over input rows
    if keep.sum() < min_seed_vol:
        return False, None, None
    kc = coords[keep]
    peak = kc[int(np.argmax(dt_vals[keep]))]
    if _core_is_elongated(kc, spacing, max_aspect, ndim):
        return False, None, None
    return True, keep, peak


class _PeakGrid:
    """O(1) spatial hash of placed peaks (physical units), cell size =
    min_physical_peak_separation. Only prior-label peaks are inserted, so any
    hit is a cross-label conflict. Works for 2D or 3D by point length."""
    def __init__(self, cell):
        self.cell = float(cell) if cell and cell > 0 else 1.0
        self.d = {}

    def _key(self, p):
        return tuple(int(math.floor(c / self.cell)) for c in p)

    def add(self, p):
        self.d.setdefault(self._key(p), []).append(np.asarray(p, float))

    def min_dist(self, p):
        """Smallest distance from p to any stored peak (searches the 3^ndim
        neighbourhood of cells). Returns np.inf if none nearby."""
        p = np.asarray(p, float)
        base = self._key(p)
        best = np.inf
        rng = [-1, 0, 1]
        import itertools
        for off in itertools.product(rng, repeat=len(base)):
            k = tuple(b + o for b, o in zip(base, off))
            for q in self.d.get(k, ()):
                dd = float(np.linalg.norm(p - q))
                if dd < best:
                    best = dd
        return best


def _shrink_to_clear(coords, dt_vals, rank_vals, spacing, grid,
                     min_sep, min_seed_vol, max_aspect, ndim):
    """Bug #2 primitive (Option C, asymmetric, same-family). Shrink the newcomer
    by keeping the brightest/thickest fraction of its OWN `rank_vals`
    (intensity for an Int candidate, DT for a DT candidate -> family preserved),
    searching for the LOOSEST shrink that both stays valid and clears every
    prior-label peak in `grid` by `min_sep`. Returns (kept_coords, peak_phys)
    or None (-> pushed_and_dropped)."""

    def core_at(q):
        # q in [0,100): keep rank_vals >= percentile(rank_vals, q); q=0 keeps all
        if q <= 0:
            sel = np.ones(rank_vals.shape[0], bool)
        else:
            sel = rank_vals >= np.percentile(rank_vals, q)
        if sel.sum() < min_seed_vol:
            return None
        ok, keep, peak = _finalize_core(coords[sel], dt_vals[sel], spacing,
                                        min_seed_vol, max_aspect, ndim)
        if not ok:
            return None
        # return the surviving global coords + peak for this probe
        return coords[sel][keep], peak

    def clears(fin):
        if fin is None:
            return False
        _, peak = fin
        return grid.min_dist(np.asarray(peak, float) * np.array(spacing)) >= min_sep

    # Step 1: monotone binary search for q_max = largest q with a still-valid core.
    lo, hi = 0.0, 99.0
    if core_at(lo) is None:
        return None  # even the full candidate is not a valid core (shouldn't happen)
    q_max = lo
    for _ in range(24):  # fine resolution on [0,99]
        mid = (lo + hi) / 2.0
        if core_at(mid) is not None:
            q_max = mid; lo = mid
        else:
            hi = mid

    # Step 2: smallest q in (0, q_max] whose core clears. Binary search assuming
    # monotone clearance, then a linear scan fallback to catch non-monotonicity.
    lo, hi = 0.0, q_max
    found = None
    for _ in range(24):
        mid = (lo + hi) / 2.0
        fin = core_at(mid)
        if fin is not None and clears(fin):
            found = mid; hi = mid
        else:
            lo = mid
    if found is not None:
        fin = core_at(found)
        if clears(fin):
            kc, peak = fin
            return kc, np.asarray(peak, float) * np.array(spacing)
    # Fallback: linear scan of integer percentiles up to q_max (catches any
    # non-monotone clearance the binary search stepped over).
    for q in range(1, int(math.floor(q_max)) + 1):
        fin = core_at(float(q))
        if clears(fin):
            kc, peak = fin
            return kc, np.asarray(peak, float) * np.array(spacing)
    return None

# ==== end injected helpers ====


# Attempt to get psutil for RAM profiling
try:
    import psutil

    def get_ram_usage() -> float:
        """Returns the current Resident Set Size (RSS) in gigabytes."""
        return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

except ImportError:

    def get_ram_usage() -> float:
        """Fallback if psutil is not installed."""
        return 0.0


def get_min_distance_pixels(
    spacing: Sequence[float], physical_distance: float,
    label: str = "min peak separation",
) -> int:
    """
    Minimum peak separation in pixels for a distance given in microns.

    Delegates to `dim_utils.pixels_from_physical`, which measures against the
    finest IN-PLANE axis (the last two, in 2D and 3D alike) and floors at 3. The
    3D original used ``min(spacing[1:])`` and the 2D original ``min(spacing)``;
    those are the same rule written two ways, and `[-2:]` is that rule once.
    """
    return pixels_from_physical(spacing, physical_distance,
                                min_pixels=3, label=label)




def peak_separation_radii(spacing, physical_distance, ndim):
    """Per-axis voxel radii for a physical separation.

    `peak_local_max(min_distance=N)` treats N as a voxel count on EVERY axis,
    and `get_min_distance_pixels` derives N from the finest IN-PLANE axis. On
    anisotropic data that silently demands far more separation in z than was
    asked for: 3 um at 0.156 um/px in-plane is 19 px, and 19 PLANES at a 2 um
    step is 38 um, in a stack 22 um deep. Two cells at different depths could
    never both be kept.
    """
    radii = []
    for sp in list(spacing)[:ndim]:
        r = int(round(float(physical_distance) / float(sp)))
        radii.append(max(1, r))
    return tuple(radii)


def peak_search_box(radii, ndim):
    """Separable box size for the local-maximum pre-pass.

    Deliberately the box INSCRIBED in the ellipsoid of `radii` (each half-
    extent divided by sqrt(ndim)), not the ellipsoid itself, for two reasons.

    Speed: scipy's maximum_filter is separable for `size=` and for a footprint
    that is a full box, but takes a generic per-voxel path for any other
    footprint. An ellipsoidal footprint of radii (2, 19, 19) measured 981 ms
    per call against 5.5 ms for the equivalent `size=` -- which is where a 20x
    slowdown came from when this used `footprint=`. (A benchmark using a full
    BOX footprint hides this completely, because scipy routes that back to the
    separable path.)

    Correctness: an inscribed box can only ever be LESS suppressive than the
    ellipsoid, so no peak the caller wanted is lost here. The exact Euclidean
    separation is then enforced by `dedupe_peaks_physical`, which is where it
    belonged anyway -- a voxel-grid footprint can only ever approximate it.
    """
    import math as _math
    shrink = _math.sqrt(ndim)
    return tuple(max(1, int(r / shrink)) * 2 + 1 for r in radii)


def local_maxima(values, mask, box):
    """Local maxima of `values` within `mask`, using a separable box filter."""
    mx = ndimage.maximum_filter(values, size=box, mode="nearest")
    hits = (values == mx) & mask & (values > 0)
    return np.argwhere(hits)


def dedupe_peaks_physical(peaks, values, spacing, physical_distance):
    """Keep the strongest peak in each `physical_distance` neighbourhood.

    `peak_local_max` has no tie-breaking: every voxel of a plateau that is
    maximal within the footprint comes back as its own peak. On this data
    plateaus are the norm rather than the exception -- a core spanning every
    plane of its own crop has no background above or below it, so z contributes
    nothing to the distance transform and an entire z column shares one DT
    value. A round blob three planes deep therefore returned three "peaks" at
    the same (y, x).

    That matters because `len(peaks) > 1` is what triggers the clump-splitting
    watershed. Three markers down one column split a single soma into three
    slices, each of which then has to clear `min_fragment_size` on its own --
    so a soma that would have been placed whole can be dropped entirely. The
    within-label separation gate would have discarded the duplicates later
    anyway, but only after the split had already fragmented the core.

    Greedy in descending peak value, which is what the footprint was meant to
    achieve and is unaffected by ties.
    """
    peaks = np.asarray(peaks)
    if peaks.shape[0] <= 1:
        return peaks
    phys = peaks * np.asarray(spacing, dtype=float)
    order = np.argsort(np.asarray(values), kind="stable")[::-1]
    kept: List[int] = []
    for idx in order:
        p = phys[idx]
        if all(float(np.linalg.norm(p - phys[k])) >= physical_distance
               for k in kept):
            kept.append(int(idx))
    # Back to the order peak_local_max produced, so marker numbering (and hence
    # the watershed's label ids) does not depend on DT ties.
    return peaks[sorted(kept)]


# --------------------------------------------------------------------------
# Candidate generation, split out so it can run in parallel
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class _CandidateParams:
    """Everything `_generate_label_candidates` needs besides the label itself.

    Built once per run and pickled to each pool worker. A frozen dataclass
    rather than a dict unpacked with ``**``: a misspelled field is an error
    where the object is built, not a keyword silently absent at the far end.
    """
    spacing: Sequence[float]
    ndim: int
    strategies: List[Dict[str, Any]]
    tile_size: Sequence[int]
    min_seed_vol: int
    absolute_min_thickness_um: float
    absolute_max_thickness_um: float
    max_allowed_core_aspect_ratio: float
    intensity_smooth_um: float
    intensity_weight: float
    int_peak_sep: int
    peak_box: tuple
    peak_separation_um: float
    memmap_voxel_threshold: int
    soma_shape: str
    temp_dir: Optional[str]


def _generate_label_candidates(
    lbl: int,
    sl: Tuple[slice, ...],
    segmentation_mask: np.ndarray,
    intensity_image: np.ndarray,
    params: _CandidateParams,
    show_tile_bar: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Every candidate one label offers, plus that label's diagnostic tallies.

    Lifted verbatim out of the main loop. Nothing in here reads or writes the
    placed-peak grid, the output mask or the running label id, which is the
    property that makes it safe to run several labels at once: generation is a
    pure function of the label's bounding box and the two read-only images.
    Placement is the part that carries state, and it stays sequential and in
    ascending label order in the caller.

    Two ordering guarantees this function must keep, because the caller's
    `sort(key=(score, vol), reverse=True)` is STABLE and therefore breaks ties
    by insertion order:

      * tiles are visited in `generate_tiles` order, and
      * strategies in `strategies` order,

    so `label_candidates` comes back in exactly the sequence the single-threaded
    version built it. Reassembling label results out of order, or appending
    across labels, would silently re-break those ties.

    `diag_stats` is returned as a delta rather than mutated in place, since a
    worker process cannot share the caller's dict. The counters are pure sums,
    so merging deltas in any order gives the same totals.
    """
    if params.soma_shape == "elongated":
        return _elongated_label_candidates(
            lbl, sl, segmentation_mask, intensity_image, params)

    # Local names, so the body reads exactly as it did when these were
    # separate parameters.
    spacing = params.spacing
    ndim = params.ndim
    strategies = params.strategies
    tile_size = params.tile_size
    min_seed_vol = params.min_seed_vol
    absolute_min_thickness_um = params.absolute_min_thickness_um
    absolute_max_thickness_um = params.absolute_max_thickness_um
    max_allowed_core_aspect_ratio = params.max_allowed_core_aspect_ratio
    intensity_smooth_um = params.intensity_smooth_um
    intensity_weight = params.intensity_weight
    int_peak_sep = params.int_peak_sep
    peak_box = params.peak_box
    peak_separation_um = params.peak_separation_um
    memmap_voxel_threshold = params.memmap_voxel_threshold

    diag_stats = {
        "cores_evaluated": 0,
        "cores_too_small": 0,
        "thickness_rejected": 0,
        "aspect_ratio_rejected": 0,
        "spatial_overlap_rejected": 0,
        "pushed_and_dropped": 0,
    }
    # `sl` arrives as a parameter now; the loop used to read it from the
    # module-level `slices` list, which a worker process does not have.
    num_voxels = np.prod([s.stop - s.start for s in sl])
    is_huge = num_voxels > memmap_voxel_threshold

    # Tile clump if it exceeds threshold
    tiles = generate_tiles(
        sl, tile_size,
        padding=int(absolute_max_thickness_um / min(spacing) + 2)
    )
    label_candidates = []

    # Per-label deduplication list (Fix 1 from previous iteration):
    # prevents two seeds from the same merged object being placed too close.
    # Cross-label deduplication is handled by the pixel-overlap check.
    label_placed_peaks: List = []

    tile_pbar = tqdm(
        tiles, desc=f"  ↳ Clump {lbl}", leave=False, unit="tile",
        disable=not (is_huge and show_tile_bar)
    )

    for t_idx, t in enumerate(tile_pbar):
        pad_sl = tile_slices(t["pad"])
        t_mask = segmentation_mask[pad_sl] == lbl
        if not np.any(t_mask):
            continue

        t_int = intensity_image[pad_sl]
        offset = np.array([sl_.start for sl_ in pad_sl])
        dt_obj = ndimage.distance_transform_edt(t_mask, sampling=spacing)
        max_dt_val = np.max(dt_obj)

        def process_frag_logic(mask_arr, sub_off):
            """Checks morphological validity and converts to global coords."""
            local_coords = np.argwhere(mask_arr)
            tile_coords = local_coords + sub_off
            g_coords = tile_coords + offset

            # Thickness: max inscribed radius in the full object (Fix 1)
            dt_vals = dt_obj[tuple(tile_coords.T)]
            max_thick = np.max(dt_vals)

            # Lower bound is a hard rejection: the fragment is too thin regardless
            # of how it is sub-sampled, so discard immediately.
            if max_thick < absolute_min_thickness_um:
                diag_stats["thickness_rejected"] += 1
                return

            # Upper bound: rather than discarding the whole fragment, attempt to
            # recover a sub-kernel — the voxels whose inscribed-sphere radius is
            # within the accepted thickness window.  This preserves somas that have
            # already been selected from a neighbouring strategy while still
            # honouring the morphological constraint at the kernel level.
            if max_thick > absolute_max_thickness_um:
                # Keep the inner core, discard the periphery
                min_allowed_dt = max_thick - absolute_max_thickness_um
                within_upper = dt_vals >= min_allowed_dt
                
                if not np.any(within_upper):
                    diag_stats["thickness_rejected"] += 1
                    return
                
                sub_dt_vals = dt_vals[within_upper]
                
                # Effective thickness is the internal radius from the new boundary to the peak
                effective_thickness = np.max(sub_dt_vals) - np.min(sub_dt_vals)
                if effective_thickness < absolute_min_thickness_um:
                    diag_stats["thickness_rejected"] += 1
                    return
                    
                # Narrow all coordinate arrays to the valid sub-kernel voxels.
                local_coords = local_coords[within_upper]
                tile_coords  = tile_coords[within_upper]
                g_coords     = g_coords[within_upper]
                dt_vals      = sub_dt_vals
                max_thick    = np.max(sub_dt_vals)  # keeps the true peak value intact
                sub_min      = local_coords.min(axis=0)
                sub_shape    = local_coords.max(axis=0) - sub_min + 1
                mask_arr     = np.zeros(sub_shape, dtype=bool)
                mask_arr[tuple((local_coords - sub_min).T)] = True
                local_coords = local_coords - sub_min

            # DT peak voxel — nucleus geometric centre regardless of strategy
            peak_idx = int(np.argmax(dt_vals))
            peak_coord_g = g_coords[peak_idx]          # global voxel coords

            # Per-coord intensity, retained so an Int candidate can later be
            # shrunk by its OWN brightness (family-preserving push-apart).
            int_vals = t_int[tuple(tile_coords.T)]

            # Elongation check (3D). A near-zero minor axis (a 1-voxel-thick
            # line) is the degenerate, maximally-elongated case -> treated as
            # elongated by _core_is_elongated.
            if mask_arr.sum() > 10 and _core_is_elongated(
                local_coords, spacing, max_allowed_core_aspect_ratio, ndim
            ):
                # Recovery: shave the low-DT tails, then keep the LARGEST
                # connected fragment, recompute the peak, and RE-CHECK size +
                # aspect (same threshold). A survivor that is still elongated
                # (a real process with no compact body) is rejected.
                core_threshold = max_thick - (absolute_min_thickness_um * 0.5)
                valid_core = dt_vals >= core_threshold
                ok, keep, pk = _finalize_core(
                    g_coords[valid_core], dt_vals[valid_core], spacing,
                    min_seed_vol, max_allowed_core_aspect_ratio, ndim
                )
                if not ok:
                    diag_stats["aspect_ratio_rejected"] += 1
                    return
                # Re-align every per-coord array to the recovered core.
                vc = valid_core
                g_coords = g_coords[vc][keep]
                dt_vals  = dt_vals[vc][keep]
                int_vals = int_vals[vc][keep]
                peak_coord_g = pk
                mask_arr = np.ones(g_coords.shape[0], dtype=bool)  # vol == len(coords)

            # Tiling check: use mean centroid (unchanged)
            cent = np.mean(g_coords, axis=0)
            if tile_target_contains(t["target"], cent):
                rank_vals = int_vals if strat["type"] == "Int" else dt_vals
                label_candidates.append(
                    {
                        "coords": g_coords.astype(np.int32),
                        "peak_coord": peak_coord_g,
                        "vol": int(mask_arr.sum()),
                        "score": strat["score"],
                        "strat_name": f"{strat['type']}_{strat['val']}",
                        "frag_max_thick": max_thick,
                        "family": strat["type"],
                        "dt_vals": np.asarray(dt_vals, np.float32),
                        "rank_vals": np.asarray(rank_vals, np.float32),
                    }
                )

        # Strategy Loop with Early Stopping
        _t_int_smooth = None  # per-tile cache of the smoothed intensity
        for strat in strategies:
            if is_huge and show_tile_bar:
                tile_pbar.set_postfix(
                    {
                        "Strat": f"{strat['type']}{strat['val']}",
                        "Cands": len(label_candidates),
                        "RAM": f"{get_ram_usage():.1f}G",
                    }
                )

            if strat["type"] == "DT":
                thresh = max_dt_val * strat["val"]
                if thresh <= 0:
                    continue
                core = (dt_obj >= thresh) & t_mask
                dt_ref = dt_obj
            else:
                # Intensity percentile strategy
                #
                # Threshold a SMOOTHED copy when intensity_smooth_um > 0.
                # Percentile thresholding inside a nucleus selects speckle
                # texture, so the surviving core is a dendritic web rather than a
                # blob: measured on a Hoechst stack, 72% of placed seeds had a
                # bounding-box fill below 0.30, and the p85 core of one object was
                # 1357 voxels spread over 35 disconnected fragments, nearly all of
                # which then died on min_fragment_size. Smoothing is applied in
                # PHYSICAL units so anisotropic z is honoured, and is cached once
                # per tile. 0.0 um reproduces the previous behaviour exactly.
                t_int_thresh = t_int
                _sm_um = float(intensity_smooth_um)
                if _sm_um > 0:
                    if _t_int_smooth is None:
                        _t_int_smooth = ndimage.gaussian_filter(
                            t_int.astype(np.float32),
                            tuple(_sm_um / sp for sp in spacing),
                        )
                    t_int_thresh = _t_int_smooth
                vals = t_int_thresh[t_mask]
                if vals.size == 0:
                    continue
                core = (t_int_thresh >= np.percentile(vals, strat["val"])) & t_mask
                # Calculate local DT for peak splitting
                dt_ref = ndimage.distance_transform_edt(core, sampling=spacing)

            # Early Stopping: If core is already too small for priority, skip lower strats
            if np.sum(core) < min_seed_vol:
                continue

            # Island detection via connected components
            labeled_core, n = ndimage.label(core)
            for region in regionprops(labeled_core):
                diag_stats["cores_evaluated"] += 1
                if region.area < min_seed_vol:
                    diag_stats["cores_too_small"] += 1
                    continue

                # Local Watershed Splitting for clumped peaks
                frag_crop = region.image
                frag_dt = ndimage.distance_transform_edt(frag_crop, sampling=spacing)
                # Blend intensity into the split field when intensity_weight > 0.
                #
                #     field = dt * (1 + intensity_weight * normalised_intensity)
                #
                # This is the same functional form, meaning and range as
                # `intensity_weight` in the separation step
                # (cell_splitting._expansion_speed), so the parameter reads the
                # same way in both places; 0.0 is pure DT, i.e. previous
                # behaviour. It matters because an elongated core covering two
                # touching nuclei has a single DT maximum, so the peak search
                # finds one marker and the pair is never split. Response is a
                # broad plateau: 0.25 to 5.0 gave identical results on the test
                # stack, so the exact value is not critical.
                _iw = float(intensity_weight)
                if _iw > 0:
                    _bb = region.bbox
                    _isl = tile_slices(_bb)
                    _fi = ndimage.gaussian_filter(
                        np.asarray(t_int[_isl], dtype=np.float32) * frag_crop,
                        tuple(0.5 / sp for sp in spacing))
                    _v = _fi[frag_crop]
                    if _v.size:
                        _lo, _hi = float(_v.min()), float(_v.max())
                        if _hi > _lo:
                            frag_dt = frag_dt * (
                                1.0 + _iw * ((_fi - _lo) / (_hi - _lo))
                            ) * frag_crop

                # exclude_border=False is REQUIRED here, not cosmetic.
                # peak_local_max defaults exclude_border to min_distance and
                # applies it to EVERY axis. A z stack a few slices deep is then
                # entirely inside the excluded border and peak_local_max returns
                # nothing: `len(peaks) > 1` was never true and the
                # clump-splitting watershed below never executed on a single
                # fragment. Measured on a 2 um z-step Hoechst stack: 0 peaks for
                # every one of the 8 largest clumps, 2-6 peaks each once fixed.
                #
                # `footprint` rather than `min_distance` for the same reason, on
                # the other axis of the same units confusion: min_distance is a
                # voxel count applied isotropically, so an in-plane-derived
                # value silently demanded that separation in PLANES too. See
                # `peak_separation_footprint`.
                # Cheap separable pre-pass, then the exact physical rule.
                # See peak_search_box for why the box rather than an
                # ellipsoidal footprint.
                peaks = local_maxima(frag_dt, frag_crop, peak_box)
                if len(peaks) > 1:
                    peaks = dedupe_peaks_physical(
                        peaks, frag_dt[tuple(np.asarray(peaks).T)],
                        spacing, peak_separation_um,
                    )

                if len(peaks) > 1:
                    markers = np.zeros(frag_crop.shape, dtype=np.int32)
                    for idx, pk in enumerate(peaks):
                        markers[tuple(int(v) for v in pk)] = idx + 1
                    ws = watershed(-frag_dt, markers, mask=frag_crop)
                    for wid in range(1, len(peaks) + 1):
                        m_ws = ws == wid
                        if m_ws.sum() >= min_seed_vol:
                            process_frag_logic(m_ws, region.bbox[:ndim])
                else:
                    process_frag_logic(region.image, region.bbox[:ndim])

            del core
            if 'dt_ref' in locals() and dt_ref is not dt_obj:
                del dt_ref

        del t_mask, t_int, dt_obj
        if t_idx % 5 == 0:
            gc.collect()


    return label_candidates, diag_stats


# =============================================================================
# Elongated somas (soma_shape="elongated")
# =============================================================================
#: The two soma shapes step 3 knows. "compact" is the only behaviour this step
#: had before the setting existed, and stays the default.
SOMA_SHAPES = ("compact", "elongated")

# Every scale below is a multiple of the object's measured fibre radius r, so
# none of them is a parameter. The multiples sit in the middle of ranges over
# which the result did not change on the spindle test stack:
#   along-fibre smoothing  10 r   (flat from 5 r to 32 r)
#   single-fibre width      4 r   (flat from 3 r to 6 r; 8 r starts accepting
#                                  two fibres side by side)
#   direction averaging     2 r   (stabilises the local direction estimate)
_ALONG_SMOOTH_R = 10.0
_WIDTH_LIMIT_R = 4.0
_DIRECTION_AVG_R = 2.0


# ---- exact percentiles with bounded memory -------------------------------- #
def _sortable_keys(v: np.ndarray) -> np.ndarray:
    """float32 values as uint32 keys that sort in the same order (IEEE-754:
    flip the sign bit of non-negatives, every bit of negatives)."""
    u = np.ascontiguousarray(v, dtype=np.float32).view(np.uint32)
    neg = (u >> 31).astype(bool)
    return np.where(neg, ~u, u | np.uint32(0x80000000))


def _key_to_float(k: int) -> np.float32:
    k = np.uint32(k)
    u = (k ^ np.uint32(0x80000000)) if (k >> np.uint32(31)) else ~k
    return np.array([u], dtype=np.uint32).view(np.float32)[0]


def _exact_percentiles(chunks, percentiles):
    """`np.percentile(all_values, p)` (method 'linear') for each p, where
    `chunks()` yields the values piecewise, holding at most one chunk and two
    65,536-bin histograms. Order statistics are found exactly by selecting on
    the sortable integer form of the floats, 16 bits per pass; the
    interpolation is numpy's own, so the result is bit-identical."""
    import numpy.lib._function_base_impl as _fb
    n = 0
    hi = np.zeros(65536, np.int64)
    for v in chunks():
        if v.size:
            n += v.size
            hi += np.bincount(_sortable_keys(v) >> np.uint32(16), minlength=65536)
    if n == 0:
        return [np.float32(0)] * len(percentiles)
    method = _fb._QuantileMethods["linear"]
    plan = []
    for p in percentiles:
        vi = float(np.asanyarray(method["get_virtual_index"](n, np.float64(p) / 100.0)))
        lo_i = min(max(int(np.floor(vi)), 0), n - 1)
        hi_i = min(lo_i + 1, n - 1) if vi < n - 1 else n - 1
        if vi >= n - 1:
            lo_i = n - 1
        gamma = float(method["fix_gamma"](np.asanyarray(vi - np.floor(vi)), np.asanyarray(vi)))
        plan.append((lo_i, hi_i, gamma))
    ranks = sorted({r for lo_i, hi_i, _ in plan for r in (lo_i, hi_i)})
    cum = np.cumsum(hi)
    prefix = {r: int(np.searchsorted(cum, r, side="right")) for r in ranks}
    below = {r: int(cum[prefix[r]] - hi[prefix[r]]) for r in ranks}
    lows = {pf: np.zeros(65536, np.int64) for pf in set(prefix.values())}
    for v in chunks():
        if not v.size:
            continue
        k = _sortable_keys(v); top = k >> np.uint32(16)
        for pf, h in lows.items():
            sel = k[top == pf]
            if sel.size:
                h += np.bincount(sel & np.uint32(0xFFFF), minlength=65536)
    value = {}
    for r in ranks:
        pf = prefix[r]; c = np.cumsum(lows[pf])
        low = int(np.searchsorted(c, r - below[r], side="right"))
        value[r] = _key_to_float((pf << 16) | low)
    out = []
    for lo_i, hi_i, gamma in plan:
        out.append(np.float32(_fb._lerp(value[lo_i], value[hi_i], gamma)))
    return out


# ---- the three per-tile operators ------------------------------------------ #
def _ridge_and_direction(img, planes, spacing, r, is_stack):
    """Bright-ridge strength and the along-fibre direction, plane by plane.

    2D Hessian at sigma = r, per plane, as step 1's tubularity filter works.
    Strength is minus the most negative eigenvalue (the curvature ACROSS a
    bright line), clipped at 0 and scale-normalised. The direction along the
    fibre is the eigenvector of the other eigenvalue:
    arctan2(2*Hrc, Hrr - Hcc) / 2 is the angle of the eigenvector of the
    LARGER eigenvalue, which on a bright ridge points along it. It is averaged
    over 2 r in the doubled-angle representation, weighted by strength, so it
    is stable and has no 180-degree ambiguity. Returns R and the unit vector
    (C, S) in (row, col) order.
    """
    from skimage.feature import hessian_matrix
    s_px = max(0.5, r / float(spacing[-1]))
    R = np.zeros(img.shape, np.float32)
    C = np.zeros(img.shape, np.float32)
    S = np.zeros(img.shape, np.float32)
    for z in planes:
        Hrr, Hrc, Hcc = hessian_matrix(np.asarray(img[z], np.float32), sigma=s_px,
                                       order="rc", use_gaussian_derivatives=True)
        tr = Hrr + Hcc
        disc = np.sqrt(((Hrr - Hcc) / 2.0) ** 2 + Hrc ** 2)
        R[z] = np.maximum(0.0, -(tr / 2.0 - disc)) * s_px ** 2
        th = 0.5 * np.arctan2(2.0 * Hrc, Hrr - Hcc)
        c2 = ndimage.gaussian_filter(np.cos(2 * th) * R[z], _DIRECTION_AVG_R * s_px)
        s2 = ndimage.gaussian_filter(np.sin(2 * th) * R[z], _DIRECTION_AVG_R * s_px)
        a = 0.5 * np.arctan2(s2, c2)
        C[z], S[z] = np.cos(a), np.sin(a)
    if is_stack:  # smooth across planes at the same physical scale
        R = ndimage.gaussian_filter1d(R, r / float(spacing[0]), axis=0)
    return R, C, S


def _along_fibre_smooth(R, C, S, where, spacing, sigma_um):
    """Gaussian average of R along each voxel's own fibre direction, for the
    voxels in `where`: a straight line of +-3 sigma, linear interpolation."""
    zz, yy, xx = np.nonzero(where)
    s_px = sigma_um / float(spacing[-1])
    ts = np.arange(-3 * s_px, 3 * s_px + 1, max(1.0, s_px / 4))
    w = np.exp(-0.5 * (ts / s_px) ** 2)
    w /= w.sum()
    dy = C[zz, yy, xx].astype(np.float64)
    dx = S[zz, yy, xx].astype(np.float64)
    ny, nx = R.shape[1], R.shape[2]
    acc = np.zeros(zz.size, np.float32)
    for t, wt in zip(ts, w):
        # Bilinear sample within the voxel's own plane. The whole and
        # fractional parts of the offset come from t*d alone, never from the
        # absolute position, so a voxel gets bit-identical values whichever
        # tile crop it is computed in. Edges clamp, as mode="nearest".
        oy, ox = t * dy, t * dx
        fy, fx = np.floor(oy), np.floor(ox)
        wy, wx = oy - fy, ox - fx
        y0 = np.clip(yy + fy.astype(np.int64), 0, ny - 1)
        y1 = np.clip(yy + fy.astype(np.int64) + 1, 0, ny - 1)
        x0 = np.clip(xx + fx.astype(np.int64), 0, nx - 1)
        x1 = np.clip(xx + fx.astype(np.int64) + 1, 0, nx - 1)
        v = ((1 - wy) * ((1 - wx) * R[zz, y0, x0] + wx * R[zz, y0, x1])
             + wy * ((1 - wx) * R[zz, y1, x0] + wx * R[zz, y1, x1]))
        acc += np.float32(wt) * v.astype(np.float32)
    out = np.zeros(R.shape, np.float32)
    out[zz, yy, xx] = acc
    return out


def _piece_width(vox, spacing, C, S, seg_um):
    """Widest in-plane spread across the local fibre direction (5-95 pct).

    Measured in stretches of length seg_um along the piece, each against the
    direction of its own voxels, so a curved fibre is not mistaken for a wide
    one."""
    yx = vox[:, 1:] * spacing[1:]
    d = np.c_[C[tuple(vox.T)], S[tuple(vox.T)]].mean(0)
    d /= np.linalg.norm(d) + 1e-9
    along = yx @ d
    nb = max(1, int(np.ceil(np.ptp(along) / seg_um)))
    edges = np.linspace(along.min(), along.max() + 1e-6, nb + 1)
    worst = 0.0
    for b in range(nb):
        m = (along >= edges[b]) & (along < edges[b + 1])
        if m.sum() < 5:
            continue
        dl = np.c_[C[tuple(vox[m].T)], S[tuple(vox[m].T)]].mean(0)
        dl /= np.linalg.norm(dl) + 1e-9
        p = yx[m] @ np.array([-dl[1], dl[0]])
        worst = max(worst, float(np.percentile(p, 95) - np.percentile(p, 5)))
    return worst


# ---- tiling ------------------------------------------------------------------ #
def _tile_targets(box, tile):
    """The tiles' own regions: `box` cut into blocks of `tile`, each voxel in
    exactly one."""
    import itertools
    axes = [range(b.start, b.stop, t) for b, t in zip(box, tile)]
    for starts in itertools.product(*axes):
        yield tuple(slice(s0, min(s0 + t, b.stop))
                    for s0, t, b in zip(starts, tile, box))


def _grow(target, halo, bounds):
    """`target` grown by `halo` voxels per axis, clamped to `bounds`."""
    return tuple(slice(max(bd.start, t.start - h), min(bd.stop, t.stop + h))
                 for t, h, bd in zip(target, halo, bounds))


def _local(inner, outer):
    """`inner` expressed in the coordinates of the crop `outer`."""
    return tuple(slice(i.start - o.start, i.stop - o.start)
                 for i, o in zip(inner, outer))


class _Union:
    def __init__(self):
        self.p = [0]

    def new(self):
        self.p.append(len(self.p))
        return len(self.p) - 1

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _elongated_label_candidates(lbl, sl, segmentation_mask, intensity_image,
                                params: "_CandidateParams"):
    """Seeds for spindle- or fibre-shaped cells, one per fibre, full length.

    Thresholding intensity cannot give these: brightness varies as much along
    a fibre as it dips between touching fibres, so pieces break into stubs
    before they separate. What does separate them is direction. This works on
    a ridge response smoothed ALONG each fibre, and peels with the opposite
    preference to compact mode: ascending through the configured percentiles,
    a connected piece only one fibre wide is accepted whole, and a wider piece
    (several fibres merged) is re-examined one level up, inside itself only.

    Tiled, with the same pinned tile shape as compact mode, so working memory
    is one tile plus its halo whatever the object's size. Four passes over the
    tiles; everything object-sized lives in disk memmaps in the step's temp
    folder, not in RAM:

    1. Fibre radius r: median inscribed radius on the object's centre lines.
       Each tile's distance transform is recomputed with a larger halo until
       it is exact inside the tile.
    2. Ridge response and direction, with a halo covering every filter's full
       reach, so both are identical to a whole-object computation; written to
       disk.
    3. The percentile thresholds over the whole object, exactly
       (`_exact_percentiles`).
    4. Peeling, per tile plus an overlap margin of one along-fibre smoothing
       reach. A fibre crossing a tile boundary is found by both tiles, and
       pieces that share voxels are joined into one seed, so fibres are not
       cut at tile boundaries.

    Config inputs: the intensity percentiles and min_fragment_size. Every
    scale derives from r (see the constants above).
    """
    import shutil
    import tempfile

    diag = {"cores_evaluated": 0, "cores_too_small": 0, "thickness_rejected": 0,
            "aspect_ratio_rejected": 0, "spatial_overlap_rejected": 0,
            "pushed_and_dropped": 0}
    percentiles = sorted(s["val"] for s in params.strategies if s["type"] == "Int")
    if not percentiles:
        return [], diag

    # One code path for both ranks: a plane is a stack of one.
    is_stack = params.ndim == 3
    seg = segmentation_mask if is_stack else segmentation_mask[None]
    inten = intensity_image if is_stack else intensity_image[None]
    box = tuple(sl) if is_stack else (slice(0, 1),) + tuple(sl)
    tile = tuple(int(t) for t in params.tile_size)
    tile = tile if is_stack else (1,) + tile
    sp = np.asarray(params.spacing, float)
    sp = sp if is_stack else np.r_[1.0, sp]
    image_bounds = tuple(slice(0, n) for n in seg.shape)
    struct = ndimage.generate_binary_structure(3, 1)
    targets = list(_tile_targets(box, tile))

    # ---- pass 1: fibre radius, object planes, voxel count -------------------
    ridge_vals, obj_planes, n_obj = [], np.zeros(seg.shape[0], bool), 0
    for tg in targets:
        h_um = 5.0
        while True:
            halo = [int(np.ceil(h_um / s)) + 1 for s in sp]
            if not is_stack:
                halo[0] = 0
            crop = _grow(tg, halo, image_bounds)
            obj = np.asarray(seg[crop]) == lbl
            loc = _local(tg, crop)
            if not obj[loc].any():
                dt = None
                break
            dt = ndimage.distance_transform_edt(obj, sampling=sp)
            # Exact inside the tile when no inner distance could reach past
            # the halo; otherwise widen it and recompute.
            if float(dt[loc].max()) < h_um - float(sp.max()):
                break
            h_um *= 2.0
        if dt is None:
            continue
        ridge = obj & (dt >= ndimage.maximum_filter(dt, size=3)) & (dt > 0)
        ridge_vals.append(dt[loc][ridge[loc]].astype(np.float32))
        o = obj[loc]
        obj_planes[tg[0].start:tg[0].stop] |= o.any(axis=(1, 2))
        n_obj += int(o.sum())
        del obj, dt, ridge
    rv = np.concatenate(ridge_vals) if ridge_vals else np.zeros(0, np.float32)
    if n_obj == 0 or rv.size == 0:
        return [], diag
    r = float(np.median(rv))
    del ridge_vals, rv
    if not (r > 0):
        return [], diag

    s_px = max(0.5, r / float(sp[-1]))
    sL_px = _ALONG_SMOOTH_R * r / float(sp[-1])
    hess_px = int(4 * s_px + 0.5)                     # gaussian truncate = 4
    dir_px = int(4 * _DIRECTION_AVG_R * s_px + 0.5)
    along_px = int(np.ceil(3 * sL_px)) + 1            # line reach + interpolation
    halo_xy = max(hess_px + dir_px, hess_px + along_px) + 2
    halo_z = (int(4 * r / float(sp[0]) + 0.5) + 1) if is_stack else 0
    # Region the response may read from: the object's box, in-plane widened
    # by the along-fibre reach so fibres ending at the box still see real
    # image; across planes only the box.
    reach_px = int(np.ceil(3 * sL_px)) + 2
    read_bounds = (
        slice(box[0].start, box[0].stop),
        slice(max(0, box[1].start - reach_px), min(seg.shape[1], box[1].stop + reach_px)),
        slice(max(0, box[2].start - reach_px), min(seg.shape[2], box[2].stop + reach_px)),
    )
    bshape = tuple(b.stop - b.start for b in box)
    boff = np.array([b.start for b in box])
    in_box = lambda t: tuple(slice(a.start - b.start, a.stop - b.start) for a, b in zip(t, box))

    workdir = tempfile.mkdtemp(prefix=f"elongated_{lbl}_", dir=params.temp_dir)
    try:
        resp_mm = np.memmap(os.path.join(workdir, "resp.dat"), np.float32, "w+", shape=bshape)
        c_mm = np.memmap(os.path.join(workdir, "c.dat"), np.float32, "w+", shape=bshape)
        s_mm = np.memmap(os.path.join(workdir, "s.dat"), np.float32, "w+", shape=bshape)
        plane_list = np.nonzero(obj_planes)[0]

        # ---- pass 2: ridge response and direction, exact per tile ----------
        for tg in targets:
            crop = _grow(tg, (halo_z, halo_xy, halo_xy), read_bounds)
            loc = _local(tg, crop)
            obj_t = np.asarray(seg[tg]) == lbl
            if not obj_t.any():
                continue
            img = np.asarray(inten[crop], np.float32)
            planes = [z - crop[0].start for z in plane_list
                      if crop[0].start <= z < crop[0].stop]
            R, C, S = _ridge_and_direction(img, planes, sp, r, is_stack)
            del img
            where = np.zeros(R.shape, bool)
            where[loc] = obj_t
            resp = _along_fibre_smooth(R, C, S, where, sp, _ALONG_SMOOTH_R * r)
            resp_mm[in_box(tg)] = resp[loc]
            c_mm[in_box(tg)] = C[loc]
            s_mm[in_box(tg)] = S[loc]
            del R, C, S, resp, where
        resp_mm.flush(); c_mm.flush(); s_mm.flush()

        # ---- pass 3: percentile thresholds over the whole object -----------
        def _values():
            for tg in targets:
                o = np.asarray(seg[tg]) == lbl
                if o.any():
                    yield np.asarray(resp_mm[in_box(tg)])[o]
        thresholds = _exact_percentiles(_values, percentiles)

        # ---- pass 4: peeling per tile + overlap, joined across tiles -------
        piece_mm = np.memmap(os.path.join(workdir, "pieces.dat"), np.int32, "w+", shape=bshape)
        uf = _Union()
        pad = (int(np.ceil(3 * sL_px * sp[-1] / sp[0])) + 1 if is_stack else 0,
               int(np.ceil(3 * sL_px)) + 2, int(np.ceil(3 * sL_px)) + 2)
        for tg in targets:
            crop = _grow(tg, pad, box)
            loc = _local(tg, crop)
            obj = np.asarray(seg[crop]) == lbl
            if not obj[loc].any():
                continue
            bc = in_box(crop)
            resp = np.asarray(resp_mm[bc]); C = np.asarray(c_mm[bc]); S = np.asarray(s_mm[bc])
            in_target = np.zeros(obj.shape, bool)
            in_target[loc] = True
            open_ = obj.copy()
            for p, thr in zip(percentiles, thresholds):
                core = (resp >= thr) & open_
                lab, _n = ndimage.label(core, structure=struct)
                for i, psl in enumerate(ndimage.find_objects(lab), 1):
                    if psl is None:
                        continue
                    diag["cores_evaluated"] += 1
                    vox = np.argwhere(lab[psl] == i) + np.array([q.start for q in psl])
                    if len(vox) < params.min_seed_vol:
                        diag["cores_too_small"] += 1
                        open_[tuple(vox.T)] = False
                        continue
                    if _piece_width(vox, sp, C, S, _ALONG_SMOOTH_R * r) > _WIDTH_LIMIT_R * r:
                        continue  # several fibres: look again one level up
                    open_[tuple(vox.T)] = False
                    if not in_target[tuple(vox.T)].any():
                        continue  # a neighbouring tile owns this piece
                    # Join with whatever a neighbouring tile already wrote for
                    # the same fibre; claim only voxels still free.
                    gb = vox + np.array([c.start for c in bc])
                    gidx = tuple(gb.T)
                    have = piece_mm[gidx]
                    nid = uf.new()
                    for other in np.unique(have[have > 0]):
                        uf.union(nid, int(other))
                    free = have == 0
                    piece_mm[tuple(gb[free].T)] = nid
            del obj, resp, C, S, open_, in_target
        piece_mm.flush()

        # ---- assemble one candidate per joined fibre ------------------------
        coords, best = {}, {}
        for tg in targets:
            ids = np.asarray(piece_mm[in_box(tg)])
            if not ids.any():
                continue
            rsp = np.asarray(resp_mm[in_box(tg)])
            vox = np.argwhere(ids > 0)
            roots = np.array([uf.find(int(i)) for i in ids[tuple(vox.T)]])
            vals = rsp[tuple(vox.T)]
            g = vox + boff + np.array([t.start - b.start for t, b in zip(tg, box)])
            for rt in np.unique(roots):
                m = roots == rt
                coords.setdefault(rt, []).append(g[m].astype(np.int32))
                k = int(np.argmax(vals[m]))
                if rt not in best or vals[m][k] > best[rt][0]:
                    best[rt] = (float(vals[m][k]), g[m][k])
        candidates = []
        for rt in sorted(coords):
            cc = np.concatenate(coords[rt])
            pk = best[rt][1]
            if not is_stack:
                cc, pk = cc[:, 1:], pk[1:]
            candidates.append({
                "coords": cc.astype(np.int32),
                "peak_coord": np.asarray(pk),
                "vol": int(len(cc)),
                "score": 1.0,
                "strat_name": "Fibre",
                "frag_max_thick": r,
                "family": "Fibre",
                "dt_vals": np.zeros(len(cc), np.float32),
                "rank_vals": np.zeros(len(cc), np.float32),
            })
        del resp_mm, c_mm, s_mm, piece_mm
        return candidates, diag
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


#: Per-worker handles, set once by the pool initializer. The images are reopened
#: inside each process rather than pickled per task: they are memmaps, so this
#: costs one mapping per worker and no data movement at all, where sending them
#: with every label would copy each bounding box through a pipe.
_WORKER_IMAGES: Dict[str, Any] = {}


def _soma_worker_init(seg_info, int_info) -> None:
    """Open the two images in this worker and pin BLAS to one thread.

    Single-threaded BLAS both to stop `workers x cores` oversubscription and so
    every worker computes identically. Verified that PCA on the Nx2 and Nx3
    matrices this step builds returns bit-identical eigenvalues at one thread
    and at four, so pinning does not move the aspect-ratio verdict.
    """
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = "1"
    _WORKER_IMAGES["seg"] = open_worker_memmap(seg_info)
    _WORKER_IMAGES["int"] = open_worker_memmap(int_info)


def _soma_worker(task):
    """Generate one label's candidates inside a pool worker."""
    lbl, sl, params = task
    try:
        return lbl, _generate_label_candidates(
            lbl, sl, _WORKER_IMAGES["seg"], _WORKER_IMAGES["int"],
            params, show_tile_bar=False
        ), None
    except Exception as exc:  # pragma: no cover - surfaced by the caller
        return lbl, None, f"{type(exc).__name__}: {exc}"


def _empty_seed_mask(segmentation_mask, memmap_dir, memmap_final_mask,
                     memmap_output_path):
    """An all-zero seed mask, on disk when the caller asked for disk.

    Same decision the populated path makes, so the empty case does not quietly
    become the one place that allocates the whole volume in RAM.
    """
    if memmap_dir is not None and memmap_final_mask:
        os.makedirs(memmap_dir, exist_ok=True)
        path = memmap_output_path or os.path.join(memmap_dir,
                                                  "final_seed_mask.mmp")
        out = np.memmap(path, dtype="int32", mode="w+",
                        shape=segmentation_mask.shape)
        out[:] = 0
        out.flush()
        return out
    return np.zeros_like(segmentation_mask, dtype=np.int32)


def extract_soma_masks(
    segmentation_mask: np.ndarray,
    intensity_image: np.ndarray,
    spacing: Sequence[float],
    *,
    min_fragment_size: int,
    intensity_smooth_um: float,
    intensity_weight: float,
    ratios_to_process: List[float],
    intensity_percentiles_to_process: List[float],
    min_physical_peak_separation: float,
    max_allowed_core_aspect_ratio: float,
    absolute_min_thickness_um: float,
    absolute_max_thickness_um: float,
    memmap_dir: Optional[str],
    memmap_final_mask: bool = True,
    memmap_output_path: Optional[str] = None,
    memmap_voxel_threshold: int = 25_000_000,
    tile_size: Optional[Sequence[int]] = None,
    soma_shape: str = "compact",
) -> np.ndarray:
    """
    Memory-efficient 3D Soma Extraction logic.

    Processes labels individually to minimize peak RAM. For huge clumps, it uses
    spatial tiling. Early stopping prevents unnecessary calculations on labels
    that are already "filled" or too small.

    Config-owned parameters are keyword-only with NO defaults, the same rule
    as `initial_segmentation.segment_cells_first_pass_raw`: every one comes
    from the YAML, so a default here would be a second, invisible place to
    configure the step. There is no `**kwargs`, so a misspelled or retired
    parameter name is a TypeError at the call site instead of being silently
    ignored.

    Args:
        segmentation_mask: Labeled segmentation, 2D or 3D.
        intensity_image: Intensity image, same shape.
        spacing: Voxel spacing in microns, ordered like the array axes.
        min_fragment_size: Hard minimum voxel (pixel) count for a seed.
        intensity_smooth_um: Gaussian sigma (microns) applied to the intensity
            before percentile thresholding. 0 disables it.
        intensity_weight: Weight of intensity relative to the distance transform
            when splitting a fused core, as dt * (1 + w * norm_intensity). Same
            meaning and range as the separation step's parameter of that name.
            0 disables it.
        ratios_to_process: DT thresholds relative to max DT.
        intensity_percentiles_to_process: Intensity thresholds.
        min_physical_peak_separation: Minimum global distance between seeds (um).
        max_allowed_core_aspect_ratio: Max elongation (PCA ratio).
        absolute_min_thickness_um: Hard lower bound for soma thickness.
        absolute_max_thickness_um: Hard upper bound for soma thickness.
        memmap_dir: Directory for the output memmap. Required, with no default:
            the former default was the relative path "ramiseg_temp_memmap",
            i.e. a folder in whatever the working directory happened to be.
            None keeps the result in RAM.
        memmap_final_mask: If True (and memmap_dir is set), the result is a
            memmap file in memmap_dir.
        memmap_output_path: Exact output file, overriding the name in memmap_dir.
        memmap_voxel_threshold: Voxel count above which a clump is reported as
            "huge". DISPLAY ONLY -- it controls whether the per-tile progress
            bar and its RAM readout are shown, nothing else. Tiling is
            unconditional: `generate_tiles` is called for every label.
        tile_size: Tile shape for huge clumps; None uses the pinned default.
        soma_shape: "compact" (default, the step's original behaviour) or
            "elongated" for spindle/fibre-shaped cells: one seed per fibre,
            full length. Elongated mode reads only the intensity percentiles
            and min_fragment_size, and is tiled like compact mode, keeping
            its object-sized working arrays as memmaps in `memmap_dir` (the
            system temp folder when that is None); see
            `_elongated_label_candidates`.

    Returns:
        np.ndarray: 3D labeled mask containing extracted soma seeds.
    """
    t_start_global = time.time()

    # Rank is taken from the data, not from which module was imported. Every
    # dimension-specific decision below reads this: structuring elements, tile
    # generation, PCA rank, marker placement, sub-bbox slicing.
    ndim = int(segmentation_mask.ndim)
    if ndim not in (2, 3):
        raise ValueError(
            f"soma extraction handles 2D and 3D data; got a {ndim}D array"
        )
    unit = "voxel" if ndim == 3 else "pixel"
    if soma_shape not in SOMA_SHAPES:
        raise ValueError(f"soma_shape must be one of {SOMA_SHAPES}; got {soma_shape!r}")
    elongated = soma_shape == "elongated"

    print("\n" + "=" * 60)
    print(f"{ndim}D SOMA EXTRACTION: STARTING")
    print("=" * 60)

    # 1. Setup & Spacing
    # No fallback to isotropic 1.0. Every parameter this step takes is in microns
    # -- thicknesses, peak separation, smoothing sigma -- so a substituted spacing
    # reinterprets all of them as pixel counts and every seed it places is wrong
    # by that factor, with nothing in the output to show it. Raises instead, the
    # same stance `metadata.require_dimensions` takes upstream.
    spacing = normalise_spacing(spacing, ndim)
    min_seed_vol = max(1, min_fragment_size)

    # Consolidated peak separation used for both global deduplication and internal splitting
    int_peak_sep = get_min_distance_pixels(
        spacing, min_physical_peak_separation, label="min peak separation"
    )
    # Built once here rather than per fragment: it depends only on the spacing
    # and the requested separation, and it is pickled to each worker.
    _peak_radii = peak_separation_radii(
        spacing, min_physical_peak_separation, ndim)
    _peak_box = peak_search_box(_peak_radii, ndim)
    if ndim == 3 and len(set(float(s) for s in spacing)) > 1:
        _fp_r = _peak_radii
        print(f"  Peak separation footprint: radii {_fp_r} voxels "
              f"(= {min_physical_peak_separation:.2f} µm on every axis; "
              f"an isotropic min_distance would have used "
              f"{int_peak_sep} on all three)")

    # Find labels via slices (efficient bounding boxes)
    slices = ndimage.find_objects(segmentation_mask)
    valid_labels = [i + 1 for i, s in enumerate(slices) if s is not None]
    if not valid_labels:
        # A memmap when one is available, not `np.zeros_like`. The empty case
        # allocated the whole volume as int32 in RAM -- the largest single
        # allocation in the step, to return nothing.
        return _empty_seed_mask(segmentation_mask, memmap_dir, memmap_final_mask,
                                memmap_output_path)

    # 2. Absolute Mode Initialization & Profiling
    print(f"  Absolute Mode Enforced: Processing {len(valid_labels)} labels...")
    print(f"  Soma shape: {soma_shape}" + (
        " (one seed per fibre; reads only the intensity percentiles and the "
        "min seed size)" if elongated else ""))
    print(f"  Thresh: Min Volume = {min_seed_vol} {unit}s")
    print(f"  Thresh: Thickness = [{absolute_min_thickness_um:.2f} - {absolute_max_thickness_um:.2f}] µm")
    print(f"  Thresh: Peak Separation = {min_physical_peak_separation:.2f} µm")

    diag_stats = {
        "cores_evaluated": 0,
        "cores_too_small": 0,
        "thickness_rejected": 0,
        "aspect_ratio_rejected": 0,
        "spatial_overlap_rejected": 0,
        "pushed_and_dropped": 0
    }

    # 3. Output Initialization
    # restored Orchestrator compatibility: explicitly checking memmap_dir and filename
    final_seed_mask = None
    if memmap_dir is not None and memmap_final_mask:
        os.makedirs(memmap_dir, exist_ok=True)
        mmp_path = memmap_output_path or os.path.join(
            memmap_dir, "final_seed_mask.mmp"
        )
        final_seed_mask = np.memmap(
            mmp_path, dtype="int32", mode="w+", shape=segmentation_mask.shape
        )
        final_seed_mask[:] = 0
        print(f"  Initialized output memmap at: {mmp_path}")
    else:
        # Explicitly opted out of a memmap, so this is the caller's choice; it
        # is still a full-volume int32 allocation and says so.
        _ram_gb = 4 * int(np.prod(segmentation_mask.shape)) / (1024 ** 3)
        if _ram_gb > 1.0:
            print(f"  [resources] memmap_final_mask=False: holding the "
                  f"{_ram_gb:.2f} GB output mask in RAM. Pass a memmap_dir to "
                  f"stream it to disk instead.")
        final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32)

    next_label_id = 1
    # Global spatial hash of placed peaks (physical units), cell size =
    # min_physical_peak_separation. Committed per-label AFTER placement, so during
    # a label it holds only prior-label peaks -> every hit is a cross-label conflict.
    _placed_grid = _PeakGrid(min_physical_peak_separation)

    # 4. Strategy Definitions
    # Strategies are ordered by Strict Priority Score (higher score = wins overlap)
    strategies = []
    for p in sorted(intensity_percentiles_to_process, reverse=True):
        strategies.append({"type": "Int", "val": p, "score": 2.0 + (p / 1000.0)})
    for r in sorted(ratios_to_process, reverse=True):
        strategies.append({"type": "DT", "val": r, "score": r + (r / 1000.0)})

    strategies.sort(key=lambda x: x["score"], reverse=True)

    # 5. Processing Loop (Label-First)
    main_pbar = tqdm(total=len(valid_labels), desc="Total Labels", unit="label",
                     dynamic_ncols=True)

    # --- Resource plan -------------------------------------------------
    # The tile shape is PINNED. `max_dt_val` is computed PER TILE and every DT
    # strategy thresholds at `max_dt_val * ratio`, so the tile extent sets every
    # threshold; a detection is also attributed to the tile whose target region
    # contains its centroid. Scaling it with the budget would make the somata
    # found depend on the machine. So the geometry is fixed and only the number
    # of tiles in flight is budgeted.
    #
    # Passed explicitly rather than left as `tile_size=None`. The default inside
    # `generate_tiles` happens to be the same tuple, but a default that must
    # never change is not a default -- it is a pinned constant, and it should
    # say so at the call site.
    if tile_size is None:
        tile_size = resource_budget.pinned(
            'soma_tile_shape_3d' if ndim == 3 else 'soma_tile_shape_2d')

    _budget = resource_budget.open_budget("step 3 soma extraction")
    _plan = _budget.report(_budget.plan_pinned(
        segmentation_mask.shape,
        'soma_tile_shape_3d' if ndim == 3 else 'soma_tile_shape_2d',
        "edt_3d" if ndim == 3 else "edt_2d",
        name="soma tile (pinned geometry, budgeted concurrency)",
    ))

    # Parallel generation needs both images on disk so each worker can map them
    # instead of receiving bounding boxes through a pipe. When either is a plain
    # in-RAM array -- a direct caller, or a test -- generation stays in-process.
    # `worker_memmap_handle` carries the byte offset and refuses views. The
    # former (filename, shape, dtype) triple made every worker read the image
    # from byte 0 of its file: for a TIFF opened with tifffile.memmap -- which
    # is how the app opens every image -- that shifted all intensities by the
    # header size, so the pool computed cores from the wrong pixels (223 somas
    # instead of 190 on the test stack) while in-process runs were correct.
    _seg_info = worker_memmap_handle(segmentation_mask)
    _int_info = worker_memmap_handle(intensity_image)
    _gen_workers = _plan.workers if (_seg_info and _int_info) else 1
    _gen_workers = max(1, min(_gen_workers, len(valid_labels)))
    if _gen_workers > 1 and not _plan.fits:
        # One pinned tile already exceeds the ceiling; adding concurrency would
        # multiply an allocation that is too large to begin with.
        _gen_workers = 1

    # Window size bounds how many labels' candidate lists are held at once.
    # Four per worker keeps the pool fed without letting generation run far
    # ahead of the much cheaper placement loop and accumulate coordinate arrays.
    _gen_window = max(1, _gen_workers * 4)
    print(f"  [resources] candidate generation on {_gen_workers} worker(s), "
          f"window {_gen_window} labels; placement sequential")

    # Generation runs in parallel; PLACEMENT does not, and must not.
    #
    # Everything expensive in this step -- the per-tile distance transforms, the
    # percentile thresholds, the connected components, the splitting watershed,
    # the PCA -- happens while building a label's candidate list, and that work
    # touches nothing but the label's own bounding box in two read-only images.
    # Placement is the opposite: `_placed_grid` holds only PRIOR-label peaks, so
    # whichever clump is reached first places its soma and a nearby candidate
    # from another clump is shrunk by `_shrink_to_clear` or dropped. That makes
    # the placement loop order-defining, not merely order-sensitive, and it
    # stays exactly as it was -- sequential, ascending label id.
    #
    # Labels are therefore generated in ordered windows and placed in the same
    # order they always were. `pool.map` preserves input order, so the candidate
    # lists arrive in ascending label order and each list is internally in tile
    # then strategy order; the stable sort below breaks ties identically to the
    # single-threaded run.
    _gen_params = _CandidateParams(
        spacing=spacing, ndim=ndim, strategies=strategies, tile_size=tile_size,
        min_seed_vol=min_seed_vol,
        absolute_min_thickness_um=absolute_min_thickness_um,
        absolute_max_thickness_um=absolute_max_thickness_um,
        max_allowed_core_aspect_ratio=max_allowed_core_aspect_ratio,
        intensity_smooth_um=intensity_smooth_um,
        intensity_weight=intensity_weight, int_peak_sep=int_peak_sep,
        peak_box=_peak_box,
        peak_separation_um=float(min_physical_peak_separation),
        memmap_voxel_threshold=memmap_voxel_threshold,
        soma_shape=soma_shape,
        # Elongated mode keeps its object-sized working arrays on disk here.
        temp_dir=memmap_dir,
    )

    _pool = None
    if _gen_workers > 1:
        try:
            _pool = mp.Pool(
                processes=_gen_workers, initializer=_soma_worker_init,
                initargs=(_seg_info, _int_info),
            )
        except Exception as exc:
            print(f"  [resources] worker pool unavailable ({exc}); "
                  "generating sequentially.")
            _pool = None

    try:
        for _w0 in range(0, len(valid_labels), _gen_window):
            _batch = valid_labels[_w0:_w0 + _gen_window]

            if _pool is None:
                _results = [
                    (lb, _generate_label_candidates(
                        lb, slices[lb - 1], segmentation_mask, intensity_image,
                        _gen_params), None)
                    for lb in _batch
                ]
            else:
                _results = _pool.map(
                    _soma_worker,
                    [(lb, slices[lb - 1], _gen_params) for lb in _batch],
                )

            for lbl, _payload, _err in _results:
                if _err is not None:
                    raise RuntimeError(
                        f"soma candidate generation failed on label {lbl}: {_err}"
                    )
                label_candidates, _diag_delta = _payload
                for _k, _v in _diag_delta.items():
                    diag_stats[_k] += _v
                main_pbar.update(1)

                # Per-label deduplication list: prevents two seeds from the same
                # merged object being placed too close. Cross-label
                # deduplication is handled by the pixel-overlap check.
                label_placed_peaks: List = []

                # 6. Placement (Greedy based on Priority and Spatial Separation)
                if label_candidates:
                    # Sort by Priority Score descending, then Volume descending
                    label_candidates.sort(key=lambda x: (x["score"], x["vol"]), reverse=True)
                    this_label_peaks = []  # committed to the global grid after this label
                    for cand in label_candidates:
                        coords = cand["coords"]
                        if elongated:
                            # Elongated candidates are disjoint by construction
                            # (each accepted piece leaves the pool), so the
                            # peak-distance gates below do not apply; only a
                            # clash with an already written seed is refused.
                            idx_tuple = tuple(coords.T)
                            if np.any(final_seed_mask[idx_tuple] > 0):
                                diag_stats["spatial_overlap_rejected"] += 1
                                continue
                            final_seed_mask[idx_tuple] = next_label_id
                            next_label_id += 1
                            continue
                        peak_phys = cand["peak_coord"] * np.array(spacing)

                        # Within-label proximity gate
                        if label_placed_peaks:
                            dists = np.linalg.norm(
                                np.array(label_placed_peaks) - peak_phys, axis=1
                            )
                            min_dist = np.min(dists)
                            if min_dist < min_physical_peak_separation:
                                diag_stats["spatial_overlap_rejected"] += 1
                                continue
                            else:
                                # --- THE TRAP: We are placing multiple somas in one label! ---
                                print(
                                    f"\n  [TRAP] Label {lbl} got MULTIPLE somas!"
                                    f"\n    -> New Edge/Extra Soma: Strategy {cand.get('strat_name', 'Unknown')}, Vol {cand['vol']}, Thick {cand.get('frag_max_thick', 0):.1f}"
                                    f"\n    -> Distance to nearest existing soma: {min_dist:.1f} µm (Limit is {min_physical_peak_separation:.1f} µm)"
                                )

                        # Cross-label separation (bug #2): grid holds only prior-label
                        # peaks, so any hit is a different cell. Shrink the newcomer
                        # asymmetrically within its own strategy family to a tighter core
                        # whose peak clears by min_physical_peak_separation; drop if none.
                        if _placed_grid.min_dist(peak_phys) < min_physical_peak_separation:
                            res = _shrink_to_clear(
                                coords, cand['dt_vals'], cand['rank_vals'], np.array(spacing),
                                _placed_grid, min_physical_peak_separation,
                                min_seed_vol, max_allowed_core_aspect_ratio, ndim
                            )
                            if res is None:
                                diag_stats["pushed_and_dropped"] += 1
                                continue
                            coords, peak_phys = res
                            coords = coords.astype(np.int32)

                        # Pixel Overlap Check (cross-label deduplication)
                        idx_tuple = tuple(coords.T)
                        if np.any(final_seed_mask[idx_tuple] > 0):
                            diag_stats["spatial_overlap_rejected"] += 1
                            continue

                        # Place Seed
                        final_seed_mask[idx_tuple] = next_label_id
                        next_label_id += 1
                        label_placed_peaks.append(peak_phys)
                        this_label_peaks.append(peak_phys)

                    for _p in this_label_peaks:
                        _placed_grid.add(_p)

                # Main Progress Update
                main_pbar.set_postfix(
                    {"Seeds": next_label_id - 1, "RAM": f"{get_ram_usage():.1f}G"}
                )
                label_candidates.clear()


    finally:
        if _pool is not None:
            _pool.terminate()
            _pool.join()
        gc.collect()

    t_total = time.time() - t_start_global
    print("\n" + "=" * 60)
    print(f"{ndim}D EXTRACTION COMPLETE")
    print(f"  Total Somas Placed: {next_label_id - 1}")
    print(f"  Execution Time: {t_total/60:.2f} mins")
    print("-" * 60)
    print("  DIAGNOSTICS (Absolute Mode Tracking):")
    print(f"    Total Core Fragments Evaluated: {diag_stats['cores_evaluated']}")
    print(f"    Rejected -> Too Small:          {diag_stats['cores_too_small']}")
    print(f"    Rejected -> Thickness Bound:    {diag_stats['thickness_rejected']}")
    print(f"    Rejected -> Aspect Ratio:       {diag_stats['aspect_ratio_rejected']}")
    print(f"    Rejected -> Spatial Overlap:    {diag_stats['spatial_overlap_rejected']}")
    print(f"    Pushed Apart -> Dropped:        {diag_stats['pushed_and_dropped']}")
    print("=" * 60 + "\n")

    # Final cleanup and persistence
    if isinstance(final_seed_mask, np.memmap):
        final_seed_mask.flush()

    return final_seed_mask