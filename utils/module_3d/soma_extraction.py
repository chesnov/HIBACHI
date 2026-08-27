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
from typing import List, Dict, Optional, Tuple, Any, Union

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max  # type: ignore
from skimage.segmentation import watershed  # type: ignore
from skimage.measure import regionprops  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from tqdm import tqdm


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


def get_min_distance_pixels_3d(
    spacing: Tuple[float, float, float], physical_distance: float
) -> int:
    """
    Calculates minimum distance in pixels for peak detection.

    Uses the minimum in-plane resolution (YX) to determine the voxel separation
    required to satisfy a physical distance requirement.

    Args:
        spacing: Voxel spacing (Z, Y, X).
        physical_distance: Desired minimum separation in physical units (um).

    Returns:
        int: Minimum distance in pixels (minimum of 3).
    """
    min_spacing_yx = min(spacing[1:])
    if min_spacing_yx <= 1e-6:
        return 3
    pixels = int(round(physical_distance / min_spacing_yx))
    return max(3, pixels)


def _generate_3d_tiles(
    bbox: Tuple[slice, slice, slice],
    tile_size: Tuple[int, int, int] = (128, 512, 512),
    padding: int = 20,
) -> List[Dict[str, Tuple[int, ...]]]:
    """
    Splits a large bounding box into overlapping 3D tiles.

    Args:
        bbox: The bounding box slices (Z, Y, X) of the label.
        tile_size: Dimensions of each tile chunk (Z, Y, X).
        padding: Voxel padding for context around the tile.

    Returns:
        List of dicts containing 'target' (no overlap) and 'pad' (context) coords.
    """
    z_range, y_range, x_range = bbox
    z0, z1 = z_range.start, z_range.stop
    y0, y1 = y_range.start, y_range.stop
    x0, x1 = x_range.start, x_range.stop

    tiles = []
    for z in range(z0, z1, tile_size[0]):
        for y in range(y0, y1, tile_size[1]):
            for x in range(x0, x1, tile_size[2]):
                # Target area (the unique region this tile is responsible for)
                tz1 = min(z + tile_size[0], z1)
                ty1 = min(y + tile_size[1], y1)
                tx1 = min(x + tile_size[2], x1)

                # Padded context area (to ensure DT and Watershed have context)
                pz0, pz1 = max(z0, z - padding), min(z1, tz1 + padding)
                py0, py1 = max(y0, y - padding), min(y1, ty1 + padding)
                px0, px1 = max(x0, x - padding), min(x1, tx1 + padding)

                tiles.append(
                    {
                        "target": (z, y, x, tz1, ty1, tx1),
                        "pad": (pz0, py0, px0, pz1, py1, px1),
                    }
                )
    return tiles


def extract_soma_masks(
    segmentation_mask: np.ndarray,
    intensity_image: np.ndarray,
    spacing: Optional[Tuple[float, float, float]],
    min_fragment_size: int = 30,
    erosion_iterations: int = 0,
    ratios_to_process: List[float] = [0.3, 0.4, 0.5, 0.6],
    intensity_percentiles_to_process: List[int] = [100, 90, 80, 70, 60, 50, 40, 30],
    min_physical_peak_separation: float = 7.0,
    max_allowed_core_aspect_ratio: float = 10.0,
    absolute_min_thickness_um: float = 1.5,
    absolute_max_thickness_um: float = 10.0,
    memmap_dir: Optional[str] = "ramiseg_temp_memmap",
    memmap_voxel_threshold: int = 25_000_000,
    memmap_final_mask: bool = True,
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
        erosion_iterations: Iterations to erode core masks.
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
        memmap_voxel_threshold: Voxel count to trigger tiling logic.
        memmap_final_mask: If True, saves result as a file in memmap_dir.

    Returns:
        np.ndarray: 3D labeled mask containing extracted soma seeds.
    """
    t_start_global = time.time()
    print("\n" + "=" * 60)
    print("3D SOMA EXTRACTION: STARTING")
    print("=" * 60)

    # 1. Setup & Spacing
    if spacing is None:
        spacing = (1.0, 1.0, 1.0)
    spacing = tuple(float(s) for s in spacing)
    min_seed_vol = max(1, min_fragment_size)

    # Consolidated peak separation used for both global deduplication and internal splitting
    int_peak_sep = get_min_distance_pixels_3d(spacing, min_physical_peak_separation)

    # Find labels via slices (efficient bounding boxes)
    slices = ndimage.find_objects(segmentation_mask)
    valid_labels = [i + 1 for i, s in enumerate(slices) if s is not None]
    if not valid_labels:
        return np.zeros_like(segmentation_mask, dtype=np.int32)

    # 2. Absolute Mode Initialization & Profiling
    print(f"  Absolute Mode Enforced: Processing {len(valid_labels)} labels...")
    print(f"  Thresh: Min Volume = {min_seed_vol} voxels")
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
        mmp_path = os.path.join(memmap_dir, "final_seed_mask.mmp")
        final_seed_mask = np.memmap(
            mmp_path, dtype="int32", mode="w+", shape=segmentation_mask.shape
        )
        final_seed_mask[:] = 0
        print(f"  Initialized output memmap at: {mmp_path}")
    else:
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
    main_pbar = tqdm(valid_labels, desc="Total Labels", unit="label", dynamic_ncols=True)

    for lbl_idx, lbl in enumerate(main_pbar):
        sl = slices[lbl - 1]
        num_voxels = np.prod([s.stop - s.start for s in sl])
        is_huge = num_voxels > memmap_voxel_threshold

        # Tile clump if it exceeds threshold
        tiles = _generate_3d_tiles(
            sl, padding=int(absolute_max_thickness_um / min(spacing) + 2)
        )
        label_candidates = []

        # Per-label deduplication list (Fix 1 from previous iteration):
        # prevents two seeds from the same merged object being placed too close.
        # Cross-label deduplication is handled by the pixel-overlap check.
        label_placed_peaks: List = []

        tile_pbar = tqdm(
            tiles, desc=f"  ↳ Clump {lbl}", leave=False, unit="tile", disable=not is_huge
        )

        for t_idx, t in enumerate(tile_pbar):
            z0, y0, x0, z1, y1, x1 = t["pad"]
            t_mask = segmentation_mask[z0:z1, y0:y1, x0:x1] == lbl
            if not np.any(t_mask):
                continue

            t_int = intensity_image[z0:z1, y0:y1, x0:x1]
            offset = np.array([z0, y0, x0])
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
                    local_coords, spacing, max_allowed_core_aspect_ratio, 3
                ):
                    # Recovery: shave the low-DT tails, then keep the LARGEST
                    # connected fragment, recompute the peak, and RE-CHECK size +
                    # aspect (same threshold). A survivor that is still elongated
                    # (a real process with no compact body) is rejected.
                    core_threshold = max_thick - (absolute_min_thickness_um * 0.5)
                    valid_core = dt_vals >= core_threshold
                    ok, keep, pk = _finalize_core(
                        g_coords[valid_core], dt_vals[valid_core], spacing,
                        min_seed_vol, max_allowed_core_aspect_ratio, 3
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
                if (
                    t["target"][0] <= cent[0] < t["target"][3]
                    and t["target"][1] <= cent[1] < t["target"][4]
                    and t["target"][2] <= cent[2] < t["target"][5]
                ):
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
            for strat in strategies:
                if is_huge:
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
                    vals = t_int[t_mask]
                    if vals.size == 0:
                        continue
                    core = (t_int >= np.percentile(vals, strat["val"])) & t_mask
                    # Calculate local DT for peak splitting
                    dt_ref = ndimage.distance_transform_edt(core, sampling=spacing)

                # Early Stopping: If core is already too small for priority, skip lower strats
                if np.sum(core) < min_seed_vol:
                    continue

                if erosion_iterations > 0:
                    core = ndimage.binary_erosion(core, iterations=erosion_iterations)
                    if not np.any(core):
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
                    peaks = peak_local_max(
                        frag_dt, min_distance=int_peak_sep, labels=frag_crop
                    )

                    if len(peaks) > 1:
                        markers = np.zeros(frag_crop.shape, dtype=np.int32)
                        for idx, pk in enumerate(peaks):
                            markers[pk[0], pk[1], pk[2]] = idx + 1
                        ws = watershed(-frag_dt, markers, mask=frag_crop)
                        for wid in range(1, len(peaks) + 1):
                            m_ws = ws == wid
                            if m_ws.sum() >= min_seed_vol:
                                process_frag_logic(m_ws, region.bbox[:3])
                    else:
                        process_frag_logic(region.image, region.bbox[:3])

                del core
                if 'dt_ref' in locals() and dt_ref is not dt_obj:
                    del dt_ref

            del t_mask, t_int, dt_obj
            if t_idx % 5 == 0:
                gc.collect()

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
                        min_seed_vol, max_allowed_core_aspect_ratio, 3
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
        if 'label_candidates' in locals():
            label_candidates.clear()
            del label_candidates
        if lbl_idx % 20 == 0:
            gc.collect()

    t_total = time.time() - t_start_global
    print("\n" + "=" * 60)
    print("3D EXTRACTION COMPLETE")
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