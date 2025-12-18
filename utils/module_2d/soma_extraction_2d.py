import time
import math
import gc
import traceback
from typing import List, Dict, Optional, Tuple, Any, Set

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max  # type: ignore
from skimage.segmentation import watershed  # type: ignore
from skimage.measure import regionprops  # type: ignore
from sklearn.decomposition import PCA  # type: ignore
from tqdm import tqdm

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)


def get_min_distance_pixels_2d(
    spacing: Tuple[float, float],
    physical_distance: float
) -> int:
    """
    Calculates minimum distance in pixels for peak_local_max (2D).

    Args:
        spacing: Pixel spacing (y, x).
        physical_distance: Desired minimum separation in physical units (e.g., um).

    Returns:
        int: Minimum distance in pixels (at least 3).
    """
    min_spacing = min(spacing)
    if min_spacing <= 1e-6:
        return 3
    pixels = int(round(physical_distance / min_spacing))
    return max(3, pixels)


def _filter_candidate_fragment_2d(
    fragment_mask: np.ndarray,
    parent_dt: np.ndarray,
    spacing: Tuple[float, float],
    min_seed_fragment_area: int,
    min_accepted_core_area: float,
    max_accepted_core_area: float,
    min_accepted_thickness_um: float,
    max_accepted_thickness_um: float,
    max_allowed_core_aspect_ratio: float
) -> Tuple[str, Optional[str], Optional[np.ndarray], Optional[float], float]:
    """
    Filters a single 2D candidate seed fragment based on morphology.

    Args:
        fragment_mask: Boolean mask of the candidate fragment.
        parent_dt: Distance transform of the parent core.
        spacing: Pixel spacing (y, x).
        min_seed_fragment_area: Minimum pixel area hard limit.
        min_accepted_core_area: Minimum acceptable area.
        max_accepted_core_area: Maximum acceptable area.
        min_accepted_thickness_um: Min thickness (inscribed radius).
        max_accepted_thickness_um: Max thickness (inscribed radius).
        max_allowed_core_aspect_ratio: Max elongation (PCA ratio).

    Returns:
        Tuple containing:
        - status (str): 'valid', 'fallback', or 'discard'.
        - reason (str or None): Reason for rejection/fallback.
        - mask_copy (ndarray): Copy of the fragment mask.
        - area (float): Calculated area.
        - duration (float): Time taken.
    """
    t_start = time.time()
    try:
        fragment_area = np.sum(fragment_mask)
        if fragment_area < min_seed_fragment_area:
            return 'discard', 'too_small', None, None, time.time() - t_start

        # Thickness Check (Max inscribed disk radius)
        max_dist_in_fragment_um = np.max(parent_dt[fragment_mask])
        passes_thickness = (
            min_accepted_thickness_um <= max_dist_in_fragment_um <=
            max_accepted_thickness_um
        )

        # Area Check
        passes_area_range = (
            min_accepted_core_area <= fragment_area <= max_accepted_core_area
        )

        # Aspect Ratio Check (PCA)
        passes_aspect = True
        if fragment_area > 2:
            try:
                coords_vox = np.argwhere(fragment_mask)
                coords_phys = coords_vox * np.array(spacing)
                
                pca = PCA(n_components=2)
                pca.fit(coords_phys)
                ev = pca.explained_variance_
                ev_sorted = np.sort(np.abs(ev))[::-1]

                if ev_sorted[1] > 1e-12:
                    ratio = math.sqrt(ev_sorted[0]) / math.sqrt(ev_sorted[1])
                    if ratio > max_allowed_core_aspect_ratio:
                        passes_aspect = False
            except Exception:
                # Fallback if PCA fails (e.g., collinear points)
                passes_aspect = True

        mask_copy = fragment_mask.copy()
        duration = time.time() - t_start

        if passes_thickness and passes_area_range and passes_aspect:
            return 'valid', None, mask_copy, fragment_area, duration
        else:
            reason = 'unknown'
            if not passes_thickness:
                reason = 'thickness'
            elif not passes_area_range:
                reason = 'area'
            elif not passes_aspect:
                reason = 'aspect'
            return 'fallback', reason, mask_copy, fragment_area, duration

    except Exception as e:
        print(f"Warn: Unexpected error filtering 2D fragment: {e}")
        return 'discard', 'error', None, None, time.time() - t_start


def extract_soma_masks_2d(
    segmentation_mask: np.ndarray,
    intensity_image: np.ndarray,
    spacing: Optional[Tuple[float, float]],
    smallest_quantile: float = 0.25,
    min_fragment_size: int = 15,
    core_volume_target_factor_lower: float = 0.1,
    core_volume_target_factor_upper: float = 10.0,
    erosion_iterations: int = 0,
    ratios_to_process: List[float] = [0.3, 0.4, 0.5, 0.6],
    intensity_percentiles_to_process: List[int] = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
    min_physical_peak_separation: float = 5.0,
    max_allowed_core_aspect_ratio: float = 10.0,
    ref_vol_percentile_lower: int = 30,
    ref_vol_percentile_upper: int = 70,
    ref_thickness_percentile_lower: int = 1,
    absolute_min_thickness_um: float = 1.0,
    absolute_max_thickness_um: float = 7.0
) -> np.ndarray:
    """
    Extracts cell bodies (somas) from a 2D segmentation mask.

    Strategy:
    1. Filter small, single-soma objects based on population statistics.
    2. For large objects, analyze internal Distance Transform and Intensity peaks.
    3. Generate candidate seeds (cores) and filter them by morphology.
    4. Place seeds into a final mask, ensuring global physical separation.

    Args:
        segmentation_mask: 2D labeled image.
        intensity_image: 2D intensity image.
        spacing: Pixel spacing (y, x).
        smallest_quantile: Quantile (0-1) to identify small cells.
        min_fragment_size: Minimum pixel area for a seed.
        core_volume_target_factor_lower: Min area factor relative to median.
        core_volume_target_factor_upper: Max area factor relative to median.
        erosion_iterations: Iterations to erode cores before analysis.
        ratios_to_process: DT thresholds (relative to max DT).
        intensity_percentiles_to_process: Intensity thresholds.
        min_physical_peak_separation: Min distance between seeds (um).
        max_allowed_core_aspect_ratio: Max elongation of a seed.
        ref_vol_percentile_lower: For reference population stats.
        ref_vol_percentile_upper: For reference population stats.
        ref_thickness_percentile_lower: For reference thickness stats.
        absolute_min_thickness_um: Hard min thickness limit.
        absolute_max_thickness_um: Hard max thickness limit.

    Returns:
        np.ndarray: 2D labeled mask of extracted somas.
    """
    print("--- Starting 2D Soma Extraction ---")

    # 1. Parameter Validation & Setup
    if spacing is None:
        spacing = (1.0, 1.0)
    else:
        try:
            spacing = tuple(float(s) for s in spacing)
            assert len(spacing) == 2
        except Exception:
            print(f"Warning: Invalid spacing {spacing}. Using (1.0, 1.0).")
            spacing = (1.0, 1.0)
            
    print(f"  Spacing (y,x): {spacing}")
    
    # Clean inputs
    intensity_percentiles_to_process = sorted(
        [p for p in intensity_percentiles_to_process if 0 < p < 100], reverse=True
    )
    
    min_seed_fragment_area = max(1, min_fragment_size)
    MAX_CORE_PIXELS_FOR_WS = 250_000
    min_peak_sep_pixels = get_min_distance_pixels_2d(spacing, min_physical_peak_separation)

    # 2. Population Analysis
    unique_labels, counts = np.unique(
        segmentation_mask[segmentation_mask > 0], return_counts=True
    )
    if len(unique_labels) == 0:
        return np.zeros_like(segmentation_mask, dtype=np.int32)

    initial_areas = dict(zip(unique_labels, counts))
    all_areas_list = list(initial_areas.values())

    # Calculate Object Slices (Optimization)
    label_to_slice = {}
    slices = ndimage.find_objects(segmentation_mask)
    for lbl in unique_labels:
        idx = lbl - 1
        if 0 <= idx < len(slices) and slices[idx]:
            label_to_slice[lbl] = slices[idx]

    # Determine thresholds
    # Smallest quantile logic: identify "small cells" that likely contain 1 soma
    smallest_thresh_area = np.percentile(all_areas_list, smallest_quantile * 100)
    small_cell_labels = {
        l for l in unique_labels if initial_areas[l] <= smallest_thresh_area
    }

    # Reference statistics for "good" somas
    target_median_area = np.median(
        [v for l, v in initial_areas.items() if l in small_cell_labels] 
        or all_areas_list
    )
    min_accepted_core_area = max(
        min_seed_fragment_area, target_median_area * core_volume_target_factor_lower
    )
    max_accepted_core_area = target_median_area * core_volume_target_factor_upper
    
    print(f"  Target Area Range: [{min_accepted_core_area:.1f} - {max_accepted_core_area:.1f}] px")

    # Thickness Analysis (on reference population)
    vol_p_low = np.percentile(all_areas_list, ref_vol_percentile_lower)
    vol_p_high = np.percentile(all_areas_list, ref_vol_percentile_upper)
    ref_labels = [
        l for l in unique_labels 
        if vol_p_low < initial_areas[l] <= vol_p_high
    ]
    if len(ref_labels) < 5:
        ref_labels = list(unique_labels)

    max_thicknesses_um = []
    # Sample subset for speed
    for l in ref_labels[:min(50, len(ref_labels))]:
        sl = label_to_slice[l]
        m = (segmentation_mask[sl] == l)
        dt = ndimage.distance_transform_edt(m, sampling=spacing)
        max_thicknesses_um.append(np.max(dt))

    if max_thicknesses_um:
        calc_min = np.percentile(max_thicknesses_um, ref_thickness_percentile_lower)
        min_accepted_thick = max(absolute_min_thickness_um, calc_min)
    else:
        min_accepted_thick = absolute_min_thickness_um
    
    min_accepted_thick = min(min_accepted_thick, absolute_max_thickness_um - 0.1)
    print(f"  Target Thickness Range: [{min_accepted_thick:.2f} - {absolute_max_thickness_um:.2f}] um")

    # 3. Initialize Output
    final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
    next_final_label = 1
    processed_labels = set()

    # 4. Process Small Cells (Pass-through)
    print(f"  Processing {len(small_cell_labels)} small objects...")
    for label in tqdm(list(small_cell_labels), desc="Small Cells"):
        # Just take the whole object as a seed if it meets size criteria
        if initial_areas[label] >= min_seed_fragment_area:
            mask_loc = (segmentation_mask == label)
            final_seed_mask[mask_loc] = next_final_label
            next_final_label += 1
            processed_labels.add(label)

    # 5. Process Large Cells (Candidate Extraction)
    large_labels = [l for l in unique_labels if l not in processed_labels]
    print(f"  Processing {len(large_labels)} large objects...")
    
    struct_erosion = ndimage.generate_binary_structure(2, 2) if erosion_iterations > 0 else None

    for label in tqdm(large_labels, desc="Large Cells"):
        bbox = label_to_slice.get(label)
        if not bbox: continue

        # Crop
        pad = 1
        sl = tuple(
            slice(max(0, s.start - pad), min(dim, s.stop + pad))
            for s, dim in zip(bbox, segmentation_mask.shape)
        )
        offset = np.array([s.start for s in sl])
        
        seg_crop = segmentation_mask[sl]
        mask_crop = (seg_crop == label)
        int_crop = intensity_image[sl]
        
        if not np.any(mask_crop): continue

        # --- Candidate Generation Strategies ---
        candidates = [] # List of dicts: {'coords', 'area', 'score'}

        # Fix: Renamed function to process_core to match calls
        def process_core(core_mask, parent_dt, score):
            """Internal: Labels core components and filters them."""
            labeled_cores, num_cores = ndimage.label(core_mask)
            if num_cores == 0: return

            for i in range(1, num_cores + 1):
                c_mask = (labeled_cores == i)
                c_area = np.sum(c_mask)
                if c_area < min_seed_fragment_area: continue

                # Watershed splitting if huge
                frags = []
                if c_area > MAX_CORE_PIXELS_FOR_WS:
                    frags.append(c_mask)
                else:
                    # Split clumps using peaks
                    dt_core = ndimage.distance_transform_edt(c_mask, sampling=spacing)
                    peaks = peak_local_max(
                        dt_core, min_distance=min_peak_sep_pixels,
                        labels=c_mask, exclude_border=False
                    )
                    if len(peaks) > 1:
                        markers = np.zeros_like(dt_core, dtype=np.int32)
                        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
                        ws = watershed(-dt_core, markers, mask=c_mask)
                        for w_id in np.unique(ws):
                            if w_id > 0: frags.append(ws == w_id)
                    else:
                        frags.append(c_mask)

                # Filter Fragments
                for f_mask in frags:
                    res, _, m_copy, area, _ = _filter_candidate_fragment_2d(
                        f_mask, parent_dt, spacing,
                        min_seed_fragment_area, min_accepted_core_area, max_accepted_core_area,
                        min_accepted_thick, absolute_max_thickness_um,
                        max_allowed_core_aspect_ratio
                    )
                    if res == 'valid':
                        candidates.append({
                            'mask': m_copy,
                            'area': area,
                            'score': score
                        })
                    elif res == 'fallback':
                        # Store with lower score
                        candidates.append({
                            'mask': m_copy,
                            'area': area,
                            'score': score * 0.5
                        })

        # A. Distance Transform Strategy
        dt_obj = ndimage.distance_transform_edt(mask_crop, sampling=spacing)
        max_dist = np.max(dt_obj)
        
        if max_dist > 1e-9:
            for ratio in ratios_to_process:
                thresh = max_dist * ratio
                initial_core = (dt_obj >= thresh) & mask_crop
                if erosion_iterations > 0:
                    initial_core = ndimage.binary_erosion(
                        initial_core, structure=struct_erosion, iterations=erosion_iterations
                    )
                if np.any(initial_core):
                    process_core(initial_core, dt_obj, score=ratio)

        # B. Intensity Strategy
        vals = int_crop[mask_crop]
        if vals.size > 0:
            for perc in intensity_percentiles_to_process:
                thresh = np.percentile(vals, perc)
                initial_core = (int_crop >= thresh) & mask_crop
                if erosion_iterations > 0:
                    initial_core = ndimage.binary_erosion(
                        initial_core, structure=struct_erosion, iterations=erosion_iterations
                    )
                if np.any(initial_core):
                    # For intensity cores, recalculate DT relative to that core shape
                    core_dt = ndimage.distance_transform_edt(initial_core, sampling=spacing)
                    process_core(initial_core, core_dt, score=2.0) # High priority

        # 6. Placement (Greedy with overlap check)
        if candidates:
            # Sort by Score desc, then Area desc
            candidates.sort(key=lambda x: (x['score'], x['area']), reverse=True)
            
            for cand in candidates:
                mask_local = cand['mask']
                # Check global overlap
                coords_local = np.argwhere(mask_local)
                coords_global = coords_local + offset
                
                # Check if these pixels are already occupied in final mask
                # using tuple indexing for advanced assignment
                idx_tuple = tuple(coords_global.T)
                existing = final_seed_mask[idx_tuple]
                
                if np.any(existing > 0):
                    continue
                
                # Place
                final_seed_mask[idx_tuple] = next_final_label
                next_final_label += 1

        del seg_crop, mask_crop, int_crop, dt_obj
    
    gc.collect()
    print(f"  Extracted {next_final_label - 1} seeds total.")
    return final_seed_mask