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
        generate_tiles,
        normalise_spacing,
        pixels_from_physical,
        tile_slices,
        tile_target_contains,
    )
except ImportError:  # pragma: no cover - direct script execution
    import resource_budget
    from dim_utils import (
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

# --------------------------------------------------------------------------
# Candidate generation, split out so it can run in parallel
# --------------------------------------------------------------------------
def _generate_label_candidates(
    lbl: int,
    sl: Tuple[slice, ...],
    segmentation_mask: np.ndarray,
    intensity_image: np.ndarray,
    spacing: Sequence[float],
    ndim: int,
    strategies: List[Dict[str, Any]],
    tile_size: Sequence[int],
    min_seed_vol: int,
    absolute_min_thickness_um: float,
    absolute_max_thickness_um: float,
    max_allowed_core_aspect_ratio: float,
    intensity_smooth_um: float,
    intensity_weight: float,
    int_peak_sep: int,
    memmap_voxel_threshold: int,
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
                # applies it to EVERY axis. int_peak_sep comes from the lateral
                # spacing, so on anisotropic data it is large in voxel terms
                # (2.5 um / 0.156 um = 16 px). A z stack a few slices deep is
                # then entirely inside the excluded border and peak_local_max
                # returns nothing: `len(peaks) > 1` was never true and the
                # clump-splitting watershed below never executed on a single
                # fragment. Measured on a 2 um z-step Hoechst stack: 0 peaks for
                # every one of the 8 largest clumps, 2-6 peaks each once fixed.
                peaks = peak_local_max(
                    frag_dt, min_distance=int_peak_sep, labels=frag_crop,
                    exclude_border=False
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
    _WORKER_IMAGES["seg"] = np.memmap(seg_info[0], dtype=seg_info[2],
                                      mode="r", shape=seg_info[1])
    _WORKER_IMAGES["int"] = np.memmap(int_info[0], dtype=int_info[2],
                                      mode="r", shape=int_info[1])


def _soma_worker(task):
    """Generate one label's candidates inside a pool worker."""
    lbl, sl, params = task
    try:
        return lbl, _generate_label_candidates(
            lbl, sl, _WORKER_IMAGES["seg"], _WORKER_IMAGES["int"],
            show_tile_bar=False, **params
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
    spacing: Optional[Sequence[float]],
    min_fragment_size: int = 30,
    intensity_smooth_um: float = 0.0,
    intensity_weight: float = 0.0,
    ratios_to_process: List[float] = [0.3, 0.4, 0.5, 0.6],
    intensity_percentiles_to_process: List[int] = [100, 90, 80, 70, 60, 50, 40, 30],
    min_physical_peak_separation: float = 7.0,
    max_allowed_core_aspect_ratio: float = 10.0,
    absolute_min_thickness_um: float = 1.5,
    absolute_max_thickness_um: float = 10.0,
    memmap_dir: Optional[str] = "ramiseg_temp_memmap",
    memmap_voxel_threshold: int = 25_000_000,
    memmap_final_mask: bool = True,
    memmap_output_path: Optional[str] = None,
    tile_size: Optional[Sequence[int]] = None,
    **kwargs,
) -> np.ndarray:
    """
    Memory-efficient 3D Soma Extraction logic.

    Processes labels individually to minimize peak RAM. For huge clumps, it uses
    spatial tiling. Early stopping prevents unnecessary calculations on labels
    that are already "filled" or too small.

    Args:
        segmentation_mask: 3D labeled segmentation image.
        intensity_image: 3D intensity image.
        spacing: Voxel spacing (Z, Y, X).
        smallest_quantile: Quantile (0-100) to find reference single somas.
        min_fragment_size: Hard minimum voxel limit for a seed.
        core_volume_target_factor_lower: Min volume relative to median.
        core_volume_target_factor_upper: Max volume relative to median.
        intensity_smooth_um: Gaussian sigma (microns) applied to the intensity
            before percentile thresholding. 0 disables it.
        intensity_weight: Weight of intensity relative to the distance transform
            when splitting a fused core, as dt * (1 + w * norm_intensity). Same
            meaning and range as the separation step's parameter of that name.
            0 disables it.
        ratios_to_process: DT thresholds relative to max DT.
        intensity_percentiles_to_process: Intensity thresholds.
        min_physical_peak_separation: Minimum global distance between seeds (um).
        seeding_min_distance_um: Override for internal peak splitting.
        max_allowed_core_aspect_ratio: Max elongation (PCA ratio).
        ref_vol_percentile_lower/upper: Population bounds for thickness calculation.
        ref_thickness_percentile_lower: Percentile to set min accepted thickness.
        absolute_min_thickness_um: Hard lower bound for soma thickness.
        absolute_max_thickness_um: Hard upper bound for soma thickness.
        memmap_dir: Directory to save the final memmap result.
        memmap_voxel_threshold: Voxel count above which a clump is reported as
            "huge". DISPLAY ONLY -- it controls whether the per-tile progress
            bar and its RAM readout are shown, nothing else. Tiling is
            unconditional: `generate_tiles` is called for every label. The name
            and the previous description ("Voxel count to trigger tiling
            logic") both predate that and were misleading; note also that
            `fluorescence_strategy` passes this parameter to step 4, which does
            not accept it, and never passes it here.
        memmap_final_mask: If True, saves result as a file in memmap_dir.

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
    def _memmap_info(arr):
        fn = getattr(arr, "filename", None)
        if fn and os.path.exists(fn):
            return (fn, tuple(arr.shape), np.dtype(arr.dtype))
        return None

    _seg_info = _memmap_info(segmentation_mask)
    _int_info = _memmap_info(intensity_image)
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
    _gen_params = dict(
        spacing=spacing, ndim=ndim, strategies=strategies, tile_size=tile_size,
        min_seed_vol=min_seed_vol,
        absolute_min_thickness_um=absolute_min_thickness_um,
        absolute_max_thickness_um=absolute_max_thickness_um,
        max_allowed_core_aspect_ratio=max_allowed_core_aspect_ratio,
        intensity_smooth_um=intensity_smooth_um,
        intensity_weight=intensity_weight, int_peak_sep=int_peak_sep,
        memmap_voxel_threshold=memmap_voxel_threshold,
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
                        **_gen_params), None)
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


def extract_soma_masks_2d(
    segmentation_mask: np.ndarray,
    intensity_image: np.ndarray,
    spacing: Optional[Sequence[float]] = None,
    *,
    tile_size_threshold: int = 2048,
    pixel_area_threshold: int = 4_000_000,
    memmap_output_path: Optional[str] = None,
    **kwargs,
):
    """
    2D entry point, kept so existing callers and saved workflows keep working.

    `extract_soma_masks` handles both ranks now; this only translates the three
    arguments the 2D module spelled differently:

        tile_size_threshold  -> tile_size   (a scalar, broadcast to (N, N))
        pixel_area_threshold -> memmap_voxel_threshold
        memmap_output_path   -> honoured directly

    New code should call `extract_soma_masks`.
    """
    return extract_soma_masks(
        segmentation_mask,
        intensity_image,
        spacing,
        tile_size=(int(tile_size_threshold),) * int(segmentation_mask.ndim),
        memmap_voxel_threshold=int(pixel_area_threshold),
        memmap_output_path=memmap_output_path,
        memmap_final_mask=memmap_output_path is not None,
        memmap_dir=os.path.dirname(memmap_output_path) if memmap_output_path else None,
        **kwargs,
    )