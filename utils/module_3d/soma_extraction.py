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
from scipy.spatial import KDTree
from skimage.feature import peak_local_max  # type: ignore
from skimage.segmentation import watershed  # type: ignore
from skimage.measure import regionprops  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from tqdm import tqdm

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
        "spatial_overlap_rejected": 0
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
    all_placed_centroids = []
    spatial_index: Optional[KDTree] = None

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

                    def process_frag_logic(mask_arr, sub_off):
                        """Checks morphological validity and converts to global coords."""
                        local_coords = np.argwhere(mask_arr)
                        tile_coords = local_coords + sub_off
                        g_coords = tile_coords + offset

                        # Absolute Thickness check (Max inscribed radius)
                        max_thick = np.max(dt_ref[tuple(tile_coords.T)])
                        if not (absolute_min_thickness_um <= max_thick <= absolute_max_thickness_um):
                            diag_stats["thickness_rejected"] += 1
                            return

                        # 3D PCA elongation check
                        if mask_arr.sum() > 10:
                            try:
                                coords_phys = local_coords * np.array(spacing)
                                pca = PCA(n_components=3).fit(coords_phys)
                                ev = np.sort(np.abs(pca.explained_variance_))[::-1]
                                if (
                                    ev[2] > 1e-12
                                    and (math.sqrt(ev[0]) / math.sqrt(ev[2]))
                                    > max_allowed_core_aspect_ratio
                                ):
                                    diag_stats["aspect_ratio_rejected"] += 1
                                    return
                            except Exception:
                                pass

                        # Tiling check: ensure centroid falls in responsible target tile
                        cent = np.mean(g_coords, axis=0)
                        if (
                            t["target"][0] <= cent[0] < t["target"][3]
                            and t["target"][1] <= cent[1] < t["target"][4]
                            and t["target"][2] <= cent[2] < t["target"][5]
                        ):
                            label_candidates.append(
                                {
                                    "coords": g_coords.astype(np.int32),
                                    "vol": mask_arr.sum(),
                                    "score": strat["score"],
                                }
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
            for cand in label_candidates:
                coords = cand["coords"]
                cent_phys = np.mean(coords, axis=0) * np.array(spacing)

                # Robust Physical Proximity Check
                if all_placed_centroids:
                    # Calculate distance to all previously placed centroids
                    dists = np.linalg.norm(np.array(all_placed_centroids) - cent_phys, axis=1)
                    if np.min(dists) < min_physical_peak_separation:
                        diag_stats["spatial_overlap_rejected"] += 1
                        continue

                # Pixel Overlap Check
                idx_tuple = tuple(coords.T)
                if np.any(final_seed_mask[idx_tuple] > 0):
                    diag_stats["spatial_overlap_rejected"] += 1
                    continue

                # Place Seed
                final_seed_mask[idx_tuple] = next_label_id
                next_label_id += 1
                all_placed_centroids.append(cent_phys)

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
    print("=" * 60 + "\n")

    # Final cleanup and persistence
    if isinstance(final_seed_mask, np.memmap):
        final_seed_mask.flush()

    return final_seed_mask