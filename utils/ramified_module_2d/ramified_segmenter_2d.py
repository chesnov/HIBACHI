# --- START OF REVISED FILE ramified_segmenter_2d.py ---

import numpy as np
from scipy import ndimage
from tqdm import tqdm
from shutil import rmtree
import gc
from functools import partial
from skimage.measure import regionprops, label as skimage_label # type: ignore
from skimage.segmentation import watershed # type: ignore
from skimage.graph import route_through_array # type: ignore
from skimage.morphology import binary_dilation, disk # type: ignore # Use disk for 2D
import math
from sklearn.decomposition import PCA # type: ignore
from skimage.feature import peak_local_max # type: ignore
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from skimage.segmentation import relabel_sequential # type: ignore
import traceback

seed = 42
np.random.seed(seed)         # For NumPy


# Helper function for physical size based min_distance (2D version)
def get_min_distance_pixels_2d(spacing, physical_distance):
    """Calculates minimum distance in pixels for peak_local_max based on physical distance (2D)."""
    min_spacing_yx = min(spacing) if spacing else 1.0
    if min_spacing_yx <= 1e-6: return 3
    pixels = int(round(physical_distance / min_spacing_yx))
    return max(3, pixels)


# --- Helper function for filtering candidate fragments (2D) ---
def _filter_candidate_fragment_2d(
    fragment_mask: np.ndarray,
    spacing: Tuple[float, float],
    avg_spacing: float,
    min_seed_fragment_area: int,
    min_accepted_core_area: float,
    max_accepted_core_area: float,
    min_accepted_minor_axis_um: float,
    # Using a relatively permissive aspect ratio for filtering now,
    # as erosion handles the main separation task.
    max_allowed_core_aspect_ratio: float = 10.0
) -> Tuple[Optional[bool], Optional[float], Optional[np.ndarray]]:
    """
    Filters a single 2D candidate fragment based on area, shape, and minor axis.
    """
    try:
        fragment_area = np.sum(fragment_mask)
        if fragment_area < min_seed_fragment_area:
            return None, None, None

        fragment_mask_copy = fragment_mask.copy()

        passes_minor_axis = False
        passes_area_range = False
        passes_aspect_ratio = True # Assume true unless check fails

        props_fragment = None
        minor_axis_um = -1.0
        aspect_ratio = -1.0

        try:
            labeled_fragment = skimage_label(fragment_mask.astype(np.uint8))
            props_list = regionprops(labeled_fragment)
            if props_list:
                props_fragment = props_list[0]
                minor_axis_px = props_fragment.minor_axis_length
                minor_axis_um = minor_axis_px * avg_spacing
                if minor_axis_um >= min_accepted_minor_axis_um:
                    passes_minor_axis = True

                maj_ax = props_fragment.major_axis_length
                min_ax = props_fragment.minor_axis_length
                if min_ax > 1e-7:
                    aspect_ratio = maj_ax / min_ax
                    if aspect_ratio > max_allowed_core_aspect_ratio: # Use the default/passed AR
                        passes_aspect_ratio = False
                # else aspect ratio undefined, pass

        except Exception as e_prop:
            print(f"Warn: Regionprops failed during filtering: {e_prop}. Assuming pass for shape checks.")
            passes_minor_axis = True
            passes_aspect_ratio = True

        if min_accepted_core_area <= fragment_area <= max_accepted_core_area:
            passes_area_range = True

        if passes_minor_axis and passes_area_range and passes_aspect_ratio:
            return True, fragment_area, fragment_mask_copy
        else:
            return False, fragment_area, fragment_mask_copy

    except Exception as e_filt:
        print(f"Warn: Unexpected error during fragment filtering: {e_filt}")
        return None, None, None
    finally:
        pass


# --- extract_soma_masks_2d ---
def extract_soma_masks_2d(
    segmentation_mask: np.ndarray, # 2D mask
    intensity_image: Optional[np.ndarray], # 2D intensity image
    spacing: Optional[Tuple[float, float]], # 2D spacing (y, x)
    smallest_quantile: int = 25,
    min_fragment_size: int = 30, # Pixels
    core_area_target_factor_lower: float = 0.4,
    core_area_target_factor_upper: float = 10,
    intensity_percentiles_to_process: List[int] = [100 - i*5 for i in range(20)],
    erosion_iterations: int = 1 # <<< NEW PARAMETER for uniform erosion
) -> np.ndarray:
    """
    Generates candidate seed mask using hybrid approach + uniform erosion + mini-watershed (2D).
    Applies uniform erosion after initial core identification to separate thin bridges.
    """
    print("--- Starting 2D Soma Extraction (DT/Intensity + Erosion + Mini-WS) ---")

    # --- Input Validation & Setup ---
    run_intensity_path = False
    if intensity_image is None: print("Info: `intensity_image` None. Intensity path skipped.")
    elif intensity_image.shape != segmentation_mask.shape: print(f"Error: `intensity_image` shape mismatch. Intensity path disabled.")
    elif not np.issubdtype(intensity_image.dtype, np.number): print(f"Error: `intensity_image` not numeric. Intensity path disabled.")
    else: run_intensity_path = True; print("Info: Valid `intensity_image`. Intensity path enabled."); intensity_image = intensity_image.astype(np.float32, copy=False)
    ratios_to_process = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; print(f"DT ratios: {ratios_to_process}")
    if run_intensity_path: intensity_percentiles_to_process = sorted([p for p in intensity_percentiles_to_process if 0<p<100], reverse=True); print(f"Intensity Percentiles: {intensity_percentiles_to_process}")
    # Print erosion setting
    print(f"Uniform Erosion Iterations: {erosion_iterations}")
    min_physical_peak_separation = 7.0; min_seed_fragment_area = max(10, min_fragment_size); absolute_min_minor_axis_um = 1.0
    if spacing is None: spacing = (1.0, 1.0); print("Warn: No 2D spacing, assume isotropic.")
    else:
        try: spacing = tuple(float(s) for s in spacing); assert len(spacing) == 2
        except: print(f"Warn: Invalid 2D spacing {spacing}. Using default."); spacing = (1.0, 1.0)
    print(f"Using 2D spacing (y,x): {spacing}"); avg_spacing = np.mean(spacing)

    # --- 1. Perform Setup Calculations ---
    # (Same as before - including minor axis calculation with cap)
    print("Calculating initial object properties (2D)...")
    unique_labels_all, counts = np.unique(segmentation_mask[segmentation_mask > 0], return_counts=True);
    if len(unique_labels_all) == 0: return np.zeros_like(segmentation_mask, dtype=np.int32)
    initial_areas = dict(zip(unique_labels_all, counts))
    print("Finding object bounding boxes and validating labels (2D)...")
    label_to_slice: Dict[int, Tuple[slice,...]] = {}; valid_unique_labels_list: List[int] = []
    try:
        object_slices = ndimage.find_objects(segmentation_mask)
        if object_slices:
            num_slices = len(object_slices)
            for label in unique_labels_all:
                idx = label - 1
                if 0 <= idx < num_slices and object_slices[idx] and len(object_slices[idx]) == 2 and all(si.start < si.stop for si in object_slices[idx]):
                    label_to_slice[label] = object_slices[idx]; valid_unique_labels_list.append(label)
        if not valid_unique_labels_list: print("Error: No valid bounding boxes."); return np.zeros_like(segmentation_mask, dtype=np.int32)
        unique_labels = np.array(valid_unique_labels_list)
    except Exception as e: print(f"Error finding boxes: {e}"); return np.zeros_like(segmentation_mask, dtype=np.int32)
    areas = {label: initial_areas[label] for label in unique_labels}; all_areas_list = list(areas.values())
    if not all_areas_list: print("Error: No valid areas."); return np.zeros_like(segmentation_mask, dtype=np.int32)
    min_samples_for_median = 5
    smallest_thresh_area = np.percentile(all_areas_list, smallest_quantile) if len(all_areas_list) > 1 else (all_areas_list[0] + 1 if all_areas_list else 1)
    smallest_object_labels_set = {label for label, area in areas.items() if area <= smallest_thresh_area}
    target_soma_area = np.median(all_areas_list) if len(all_areas_list)<min_samples_for_median*2 else np.median([areas[l] for l in smallest_object_labels_set if l in areas] or all_areas_list)
    target_soma_area = max(target_soma_area, 1.0); min_accepted_core_area = max(min_seed_fragment_area, target_soma_area * core_area_target_factor_lower); max_accepted_core_area = target_soma_area * core_area_target_factor_upper
    print(f"Core Area filter range: [{min_accepted_core_area:.2f} - {max_accepted_core_area:.2f}] pixels (Abs min: {min_seed_fragment_area})")
    reference_object_labels_for_ref = [l for l in unique_labels if l in smallest_object_labels_set]
    if len(reference_object_labels_for_ref) < min_samples_for_median: print(f"Warn: Only {len(reference_object_labels_for_ref)} small objects. Using all {len(unique_labels)} for ref minor axis."); reference_object_labels_for_ref = list(unique_labels)
    print(f"Calculating reference minor axis from {len(reference_object_labels_for_ref)} ref objects..."); minor_axes_ref_px = []
    disable_tqdm = len(reference_object_labels_for_ref) < 10
    for label_ref in tqdm(reference_object_labels_for_ref, desc="Calc Ref Minor Axis", disable=disable_tqdm):
        bbox = label_to_slice.get(label_ref);
        if bbox is None: continue
        try:
            mask_sub = (segmentation_mask[bbox] == label_ref);
            if not np.any(mask_sub): continue
            props = regionprops(skimage_label(mask_sub.astype(np.uint8)))
            if props and props[0].minor_axis_length > 0: minor_axes_ref_px.append(props[0].minor_axis_length)
        except Exception as e: print(f"Warn: Ref Minor Axis Err L{label_ref}: {e}"); continue
    min_accepted_minor_axis_um = absolute_min_minor_axis_um
    if minor_axes_ref_px:
        minor_axes_ref_um = [ax * avg_spacing for ax in minor_axes_ref_px]
        if minor_axes_ref_um: min_accepted_minor_axis_um = max(absolute_min_minor_axis_um, np.percentile(minor_axes_ref_um, 15))
    reasonable_upper_cap_um = 15.0
    min_accepted_minor_axis_um = min(min_accepted_minor_axis_um, reasonable_upper_cap_um)
    print(f"Final Core Minor Axis filter minimum (capped at {reasonable_upper_cap_um:.2f} um): {min_accepted_minor_axis_um:.2f} um")
    min_peak_sep_pixels = get_min_distance_pixels_2d(spacing, min_physical_peak_separation)

    # --- 2. Process Small Cells ---
    # (Same as before)
    print("\nProcessing objects (2D)..."); final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32); next_final_label = 1; added_small_labels: Set[int] = set()
    small_cell_labels = [l for l in unique_labels if l in smallest_object_labels_set]; print(f"Processing {len(small_cell_labels)} small objects (2D)...")
    disable_tqdm = len(small_cell_labels) < 10
    for label in tqdm(small_cell_labels, desc="Small Cells", disable=disable_tqdm):
        bbox = label_to_slice.get(label);
        if bbox is None: continue
        try:
            mask_sub = (segmentation_mask[bbox] == label); area = np.sum(mask_sub)
            if area >= min_seed_fragment_area:
                offset = (bbox[0].start, bbox[1].start); coords = np.argwhere(mask_sub)
                gy = coords[:, 0] + offset[0]; gx = coords[:, 1] + offset[1]; valid = (gy>=0)&(gy<final_seed_mask.shape[0])&(gx>=0)&(gx<final_seed_mask.shape[1])
                vy = gy[valid]; vx = gx[valid]
                if vy.size > 0:
                     coords_write = (vy, vx)
                     if np.all(final_seed_mask[coords_write] == 0): final_seed_mask[coords_write] = next_final_label; print(f"  Placed small cell seed L{label} as new label {next_final_label}"); next_final_label += 1; added_small_labels.add(label)
        except Exception as e: print(f"Warn: Error small L{label}: {e}")
    print(f"Added {len(added_small_labels)} initial seeds from small cells."); gc.collect()


    # ==============================================================
    # --- 3. Generate Candidates from Large Cells ---
    # ==============================================================
    large_cell_labels = [l for l in unique_labels if l not in added_small_labels]; print(f"Generating candidates from {len(large_cell_labels)} large objects (2D - DT & Intensity)...")
    valid_candidates = []; fallback_candidates = []

    # Define structuring element for erosion once
    struct_el_erosion = ndimage.generate_binary_structure(2, 2) # 2D, connectivity=2 (3x3 square)

    for label in tqdm(large_cell_labels, desc="Large Cell Candidates"):
        bbox = label_to_slice.get(label);
        if bbox is None: continue
        pad = 1; ymin=max(0,bbox[0].start-pad); xmin=max(0,bbox[1].start-pad); ymax=min(segmentation_mask.shape[0],bbox[0].stop+pad); xmax=min(segmentation_mask.shape[1],bbox[1].stop+pad)
        if ymin >= ymax or xmin >= xmax: continue
        slice_obj = np.s_[ymin:ymax, xmin:xmax]; offset = (ymin, xmin)
        valid_cands_lbl = []; fallback_cands_lbl = []
        try:
            seg_sub = segmentation_mask[slice_obj]; mask_sub = (seg_sub == label);
            if not np.any(mask_sub): continue
            int_sub = intensity_image[slice_obj] if run_intensity_path and intensity_image is not None else None

            # --- A) DT Path ---
            try:
                dt_obj = ndimage.distance_transform_edt(mask_sub, sampling=spacing); max_dt = np.max(dt_obj)
                if max_dt > 1e-9:
                    for ratio in ratios_to_process:
                        initial_core_region_mask_sub = (dt_obj >= max_dt * ratio) & mask_sub; # Renamed for clarity
                        if not np.any(initial_core_region_mask_sub): continue

                        # *** APPLY EROSION STEP ***
                        core_mask_for_labeling = initial_core_region_mask_sub # Start with original
                        if erosion_iterations > 0:
                            eroded_core_mask = ndimage.binary_erosion(initial_core_region_mask_sub,
                                                                      structure=struct_el_erosion,
                                                                      iterations=erosion_iterations)
                            if not np.any(eroded_core_mask): continue # Erosion removed everything
                            core_mask_for_labeling = eroded_core_mask # Use eroded mask

                        # *** Label the (potentially eroded) core mask ***
                        labeled_cores, num_cores = ndimage.label(core_mask_for_labeling)
                        if num_cores == 0: continue

                        for core_lbl in range(1, num_cores + 1):
                            # Process component from the LABELED eroded mask
                            core_component_mask_sub = (labeled_cores == core_lbl);
                            if not np.any(core_component_mask_sub): continue

                            frags = []; dt_core, peaks, markers_core, ws_core = None, None, None, None
                            try: # --- Mini-WS using DT peaks (REVERTED) ---
                                dt_core = ndimage.distance_transform_edt(core_component_mask_sub, sampling=spacing)
                                if np.max(dt_core) > 1e-9:
                                    peaks = peak_local_max(dt_core, # Use DT peaks again
                                                           min_distance=min_peak_sep_pixels,
                                                           labels=core_component_mask_sub,
                                                           num_peaks_per_label=0,
                                                           exclude_border=False)
                                else:
                                     peaks = np.empty((0, dt_core.ndim), dtype=int) # Ensure defined

                                if peaks.shape[0] > 1:
                                    markers_core = np.zeros(dt_core.shape, dtype=np.int32); markers_core[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                                    ws_core = watershed(-dt_core, markers_core, mask=core_component_mask_sub, watershed_line=False)
                                    for ws_l in np.unique(ws_core):
                                        if ws_l == 0: continue; f_mask = (ws_core == ws_l);
                                        if np.any(f_mask): frags.append(f_mask)
                                else: frags.append(core_component_mask_sub)
                            except Exception as e_ws: print(f"Warn: Mini-WS DT L{label} R{ratio}: {e_ws}. Using core."); frags.append(core_component_mask_sub)
                            finally: dt_core, peaks, markers_core, ws_core = None, None, None, None # Assign None

                            for i, f_mask in enumerate(frags): # Filter fragments
                                try:
                                    # Use the default permissive aspect ratio here again
                                    is_valid, area, m_copy = _filter_candidate_fragment_2d(
                                        f_mask, spacing, avg_spacing, min_seed_fragment_area,
                                        min_accepted_core_area, max_accepted_core_area,
                                        min_accepted_minor_axis_um
                                        # max_allowed_core_aspect_ratio=10.0 # Implicit default
                                    )
                                    if is_valid is True: valid_cands_lbl.append({'mask': m_copy, 'area': area, 'offset': offset})
                                    elif is_valid is False: fallback_cands_lbl.append({'mask': m_copy, 'area': area, 'offset': offset})
                                except Exception as e_filt: print(f"Warn: Filter DT Error L{label} R{ratio} F{i}: {e_filt}")
            except Exception as e_dt: print(f"Warn: Error in DT path L{label}: {e_dt}")
            finally: # Safely delete dt_obj if it exists
                if 'dt_obj' in locals(): del dt_obj

            # --- B) Intensity Path ---
            if run_intensity_path and int_sub is not None:
                try:
                    ints_obj = int_sub[mask_sub];
                    if ints_obj.size == 0: raise ValueError("No intensity")
                    for perc in intensity_percentiles_to_process:
                        try: thresh = np.percentile(ints_obj, perc)
                        except IndexError: continue
                        initial_core_region_mask_sub = (int_sub >= thresh) & mask_sub;
                        if not np.any(initial_core_region_mask_sub): continue

                        # *** APPLY EROSION STEP ***
                        core_mask_for_labeling = initial_core_region_mask_sub # Start with original
                        if erosion_iterations > 0:
                            eroded_core_mask = ndimage.binary_erosion(initial_core_region_mask_sub,
                                                                      structure=struct_el_erosion,
                                                                      iterations=erosion_iterations)
                            if not np.any(eroded_core_mask): continue # Erosion removed everything
                            core_mask_for_labeling = eroded_core_mask # Use eroded mask

                        # *** Label the (potentially eroded) core mask ***
                        cores_lbl, n_cores = ndimage.label(core_mask_for_labeling)
                        if n_cores == 0: continue

                        for core_lbl in range(1, n_cores + 1):
                             # Process component from the LABELED eroded mask
                            comp_mask = (cores_lbl == core_lbl);
                            if not np.any(comp_mask): continue

                            frags = []; dt_core, peaks, markers_core, ws_core = None, None, None, None
                            try: # --- Mini-WS using DT peaks (REVERTED) ---
                                dt_core = ndimage.distance_transform_edt(comp_mask, sampling=spacing)
                                if np.max(dt_core) > 1e-9:
                                    peaks = peak_local_max(dt_core, # Use DT peaks again
                                                           min_distance=min_peak_sep_pixels,
                                                           labels=comp_mask,
                                                           num_peaks_per_label=0,
                                                           exclude_border=False)
                                else:
                                    peaks = np.empty((0, dt_core.ndim), dtype=int) # Ensure defined

                                if peaks.shape[0] > 1:
                                    markers_core = np.zeros(dt_core.shape, dtype=np.int32); markers_core[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                                    ws_core = watershed(-dt_core, markers_core, mask=comp_mask, watershed_line=False)
                                    for ws_l in np.unique(ws_core):
                                         if ws_l == 0: continue; f_mask = (ws_core == ws_l);
                                         if np.any(f_mask): frags.append(f_mask)
                                else: frags.append(comp_mask)
                            except Exception as e_ws: print(f"Warn: Mini-WS Int L{label} P{perc}: {e_ws}. Using core."); frags.append(comp_mask)
                            finally: dt_core, peaks, markers_core, ws_core = None, None, None, None # Assign None

                            for i, f_mask in enumerate(frags): # Filter fragments
                                try:
                                     # Use the default permissive aspect ratio here again
                                    is_valid, area, m_copy = _filter_candidate_fragment_2d(
                                        f_mask, spacing, avg_spacing, min_seed_fragment_area,
                                        min_accepted_core_area, max_accepted_core_area,
                                        min_accepted_minor_axis_um
                                        # max_allowed_core_aspect_ratio=10.0 # Implicit default
                                    )
                                    if is_valid is True: valid_cands_lbl.append({'mask': m_copy, 'area': area, 'offset': offset})
                                    elif is_valid is False: fallback_cands_lbl.append({'mask': m_copy, 'area': area, 'offset': offset})
                                except Exception as e_filt: print(f"Warn: Filter Int Error L{label} P{perc} F{i}: {e_filt}")
                except ValueError: pass
                except Exception as e_int: print(f"Warn: Error in Intensity path L{label}: {e_int}")

            # --- Combine Candidates ---
            if valid_cands_lbl:
                valid_candidates.extend(valid_cands_lbl)
            # Always add all fallback candidates from this label to the global list.
            # The global sorting and non-overlapping placement logic for fallbacks
            # will then decide if they get placed.
            # This allows a parent object to contribute both valid and fallback candidates.
            if fallback_cands_lbl:
                fallback_candidates.extend(fallback_cands_lbl)

        except MemoryError: print(f"Warn: MemError L{label}."); gc.collect()
        except Exception as e: print(f"Warn: Uncaught Error L{label}: {e}"); traceback.print_exc()
        finally: # Cleanup outer loop label vars
             if 'seg_sub' in locals(): del seg_sub
             if 'mask_sub' in locals(): del mask_sub
             if 'int_sub' in locals(): del int_sub
             del valid_cands_lbl, fallback_cands_lbl # Delete the lists themselves
             gc.collect()
        # --- End Label Loop ---

    print(f"Generated {len(valid_candidates)} valid candidates and {len(fallback_candidates)} fallback candidates.")

    # --- 4. Place Valid Candidates ---
    # (Same as before, with debug prints)
    print("Placing smallest valid seeds..."); valid_candidates.sort(key=lambda x: x['area']); processed_count = 0
    print(f"Attempting to place {len(valid_candidates)} valid. Next label = {next_final_label}")
    for idx, cand in enumerate(tqdm(valid_candidates, desc="Placing Valid Seeds")):
        print_debug = idx < 10; mask_sub = cand.get('mask'); offset = cand.get('offset')
        if mask_sub is None or offset is None or mask_sub.size == 0: 
            if print_debug: print(f"Debug V{idx}: Skip Null/0size"); continue
        try:
            coords = np.argwhere(mask_sub); n_coords = coords.shape[0];
            if n_coords == 0: 
                if print_debug: print(f"Debug V{idx}: Skip No Coords"); continue
            if print_debug: print(f"\nDebug V{idx}: Area={cand.get('area',-1):.0f}, Off={offset}, CoordsSub={n_coords}")
            gy=coords[:,0]+offset[0]; gx=coords[:,1]+offset[1]; valid=(gy>=0)&(gy<final_seed_mask.shape[0])&(gx>=0)&(gx<final_seed_mask.shape[1])
            if not np.any(valid): 
                if print_debug: print(f"Debug V{idx}: Skip No Valid Bounds"); continue
            vy=gy[valid]; vx=gx[valid]; n_global = vy.size
            if n_global == 0: 
                if print_debug: print(f"Debug V{idx}: Skip Empty Coords Post-Bound"); continue
            if print_debug: print(f"Debug V{idx}: CoordsGlobal={n_global}")
            coords_tuple = (vy, vx); existing = final_seed_mask[coords_tuple]
            is_empty = np.all(existing == 0)
            if is_empty:
                 if np.all(final_seed_mask[coords_tuple] == 0): # Check again before write
                     final_seed_mask[coords_tuple] = next_final_label
                     if print_debug: print(f"Debug V{idx}: --> PLACED seed {next_final_label}")
                     next_final_label += 1; processed_count += 1
            else:
                if print_debug: print(f"Debug V{idx}: --> SKIPPED - Overlap with {np.unique(existing[existing>0])}")
        except Exception as e: print(f"Warn: Error Place Valid V{idx}: {e}.")
        finally:
             if 'mask' in cand: del cand['mask']
    print(f"Placed {processed_count} seeds from valid candidates."); del valid_candidates; gc.collect()


    # --- 5. Place Fallback Candidates ---
    # (Same as before, with debug prints)
    print("Placing fallback seeds..."); fallback_candidates.sort(key=lambda x: x['area']); fallback_count = 0
    print(f"Attempting to place {len(fallback_candidates)} fallback. Next label = {next_final_label}")
    for idx, fallbk in enumerate(tqdm(fallback_candidates, desc="Placing Fallbacks")):
        print_debug = idx < 5; mask_sub = fallbk.get('mask'); offset = fallbk.get('offset')
        if mask_sub is None or offset is None or mask_sub.size == 0: 
            if print_debug: print(f"Debug F{idx}: Skip Null/0size"); continue
        try:
            coords = np.argwhere(mask_sub); n_coords = coords.shape[0];
            if n_coords == 0: 
                if print_debug: print(f"Debug F{idx}: Skip No Coords"); continue
            if print_debug: print(f"\nDebug F{idx}: Area={fallbk.get('area',-1):.0f}, Off={offset}, CoordsSub={n_coords}")
            gy=coords[:,0]+offset[0]; gx=coords[:,1]+offset[1]; valid=(gy>=0)&(gy<final_seed_mask.shape[0])&(gx>=0)&(gx<final_seed_mask.shape[1])
            if not np.any(valid): 
                if print_debug: print(f"Debug F{idx}: Skip No Valid Bounds"); continue
            vy=gy[valid]; vx=gx[valid]; n_global = vy.size
            if n_global == 0: 
                if print_debug: print(f"Debug F{idx}: Skip Empty Coords Post-Bound"); continue
            if print_debug: print(f"Debug F{idx}: CoordsGlobal={n_global}")
            coords_tuple = (vy, vx); existing = final_seed_mask[coords_tuple]
            is_empty = np.all(existing == 0)
            if is_empty:
                 if np.all(final_seed_mask[coords_tuple] == 0):
                     final_seed_mask[coords_tuple] = next_final_label
                     if print_debug: print(f"Debug F{idx}: --> PLACED fallback seed {next_final_label}")
                     next_final_label += 1; fallback_count += 1
            else:
                if print_debug: print(f"Debug F{idx}: --> SKIPPED fallback - Overlap with {np.unique(existing[existing>0])}")
        except Exception as e: print(f"Warn: Error Place Fallback F{idx}: {e}.")
        finally:
            if 'mask' in fallbk: del fallbk['mask']
    print(f"Placed {fallback_count} seeds from fallback candidates."); del fallback_candidates; gc.collect()


    total_final_seeds = next_final_label - 1
    print(f"\nGenerated {total_final_seeds} final aggregated seeds."); print("--- Finished 2D Intermediate Seed Extraction ---")
    return final_seed_mask


# --- refine_seeds_pca_2d ---
def refine_seeds_pca_2d(
    intermediate_seed_mask: np.ndarray, # 2D mask
    spacing: Optional[Tuple[float, float]], # 2D spacing
    target_aspect_ratio: float = 1.1,
    projection_percentile_crop: int = 10,
    min_fragment_size: int = 30 # Pixels
) -> np.ndarray:
    """
    Refines preliminary 2D seeds using PCA to make them more compact. Adapted for 2D.
    """
    print("--- Starting 2D PCA Seed Refinement ---")
    if spacing is None: spacing = (1.0, 1.0); print("Warn: No 2D spacing, assume isotropic.")
    else: print(f"Using 2D spacing (y,x): {spacing}")
    print(f"Parameters: target_aspect_ratio={target_aspect_ratio}, projection_crop%={projection_percentile_crop}, min_area={min_fragment_size}")

    final_refined_mask = np.zeros_like(intermediate_seed_mask, dtype=np.int32) # 2D output
    next_final_label = 1; kept_refined_count = 0
    seed_labels = np.unique(intermediate_seed_mask); seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels) == 0: return final_refined_mask

    seed_slices = ndimage.find_objects(intermediate_seed_mask) # Works for 2D
    label_to_slice = {}
    if seed_slices is not None:
        label_to_slice = {label: seed_slices[label-1] for label in seed_labels if label-1 < len(seed_slices) and seed_slices[label-1] is not None and len(seed_slices[label-1])==2}

    for label in tqdm(seed_labels, desc="Refining Seeds PCA (2D)"):
        bbox_slice = label_to_slice.get(label);
        if bbox_slice is None: continue
        offset = (bbox_slice[0].start, bbox_slice[1].start); slice_obj = bbox_slice # 2D offset/slice
        # Define variables before try
        seed_mask_sub, coords_vox, coords_phys, pca, refined_mask_sub = None, None, None, None, None
        coords_centered_phys, proj1, kept_coords_vox, refined_coords = None, None, None, None
        try:
            seed_mask_sub = (intermediate_seed_mask[slice_obj] == label); current_area = np.sum(seed_mask_sub)
            if seed_mask_sub.size == 0 or not np.any(seed_mask_sub) or current_area < min_fragment_size: continue

            coords_vox = np.argwhere(seed_mask_sub)
            if coords_vox.shape[0] <= 2: refined_mask_sub = seed_mask_sub # Need > 2 points for PCA
            else:
                coords_phys = coords_vox * np.array(spacing) # Apply 2D spacing
                pca = PCA(n_components=2); pca.fit(coords_phys); eigenvalues = pca.explained_variance_; eigenvectors = pca.components_ # 2 components
                sorted_indices = np.argsort(eigenvalues)[::-1]; eigenvalues = eigenvalues[sorted_indices]; eigenvectors = eigenvectors[sorted_indices]
                if eigenvalues[1] < 1e-9: eigenvalues[1] = 1e-9 # Avoid division by zero for aspect ratio
                aspect_ratio = eigenvalues[0] / eigenvalues[1] # 2D aspect ratio

                if aspect_ratio >= target_aspect_ratio:
                    coords_centered_phys = coords_phys - pca.mean_; proj1 = coords_centered_phys @ eigenvectors[0] # Project onto longest axis
                    min_p = np.percentile(proj1, projection_percentile_crop); max_p = np.percentile(proj1, 100 - projection_percentile_crop)
                    voxel_indices_to_keep = (proj1 >= min_p) & (proj1 <= max_p)
                    refined_mask_sub = np.zeros_like(seed_mask_sub, dtype=bool)
                    kept_coords_vox = coords_vox[voxel_indices_to_keep]
                    if kept_coords_vox.shape[0] > 0: refined_mask_sub[tuple(kept_coords_vox.T)] = True
                    refined_area = np.sum(refined_mask_sub)
                    if refined_area < min_fragment_size: refined_mask_sub = None # Discard if refinement makes it too small
                else: refined_mask_sub = seed_mask_sub # Keep original

            if refined_mask_sub is not None and np.any(refined_mask_sub):
                 final_area = np.sum(refined_mask_sub)
                 if final_area >= min_fragment_size: # Check final area again
                     refined_coords = np.argwhere(refined_mask_sub)
                     if refined_coords.size > 0:
                         global_coords_y=refined_coords[:,0]+offset[0]; global_coords_x=refined_coords[:,1]+offset[1]
                         valid_indices=((global_coords_y>=0)&(global_coords_y<final_refined_mask.shape[0])& (global_coords_x>=0)&(global_coords_x<final_refined_mask.shape[1]))
                         valid_global_y = global_coords_y[valid_indices]
                         valid_global_x = global_coords_x[valid_indices]
                         # Ensure not empty before writing
                         if valid_global_y.size > 0:
                             final_refined_mask[valid_global_y, valid_global_x] = next_final_label
                             next_final_label += 1; kept_refined_count += 1
        except Exception as e: print(f"Warning: Error during 2D PCA refinement for seed label {label}: {e}")
        finally: # Cleanup vars
            del seed_mask_sub, coords_vox, coords_phys, pca, refined_mask_sub
            del coords_centered_phys, proj1, kept_coords_vox, refined_coords # Delete vars defined in try

    print(f"Kept {kept_refined_count} refined 2D seeds after PCA.")
    print("--- Finished 2D PCA Seed Refinement ---"); gc.collect(); return final_refined_mask


def separate_multi_soma_cells_2d(
    segmentation_mask: np.ndarray, # 2D
    intensity_image: np.ndarray, # 2D
    soma_mask: np.ndarray, # 2D
    spacing: Optional[Tuple[float, float]], # 2D
    min_size_threshold: int = 100, # Pixels
    intensity_weight: float = 0.0,
    max_seed_centroid_dist: float = 15.0,
    min_path_intensity_ratio: float = 0.6,
    post_merge_min_interface_pixels: int = 10 # Replaces thickness check
    ) -> np.ndarray:
    """
    Separates 2D cell segmentations containing multiple seeds. Adapted for 2D.
    Uses path intensity heuristic, post-watershed merging based on interface size.
    RELABELED OUTPUT V2: Ensures final mask labels are sequential 1..N.
    (Corrected variable names version)
    """
    print(f"--- Starting 2D Multi-Soma Separation (Path Heuristics + Interface Merge) ---")

    # --- Parameter & Spacing Checks (2D) ---
    print(f"Post-merge check: min_interface_pixels={post_merge_min_interface_pixels}")
    if spacing is None: spacing = (1.0, 1.0); print("Warn: No 2D spacing, assume isotropic.")
    else:
        try: spacing = tuple(float(s) for s in spacing); assert len(spacing) == 2
        except: print(f"Error: Invalid 2D spacing ({spacing}). Using default."); spacing = (1.0, 1.0)
    print(f"Using 2D spacing (y,x): {spacing}")
    print(f"Seed merging heuristics: max_dist={max_seed_centroid_dist}, min_path_intensity_ratio={min_path_intensity_ratio}")

    # --- Input Validation & Type Checks (2D) ---
    if segmentation_mask.ndim != 2 or intensity_image.ndim != 2 or soma_mask.ndim != 2:
        raise ValueError("All input masks and image must be 2D.")
    if segmentation_mask.shape != intensity_image.shape or segmentation_mask.shape != soma_mask.shape:
        raise ValueError("Input mask/image shapes do not match.")
    # Ensure float32 for intensity, integers for masks
    if not np.issubdtype(intensity_image.dtype, np.floating): intensity_image = intensity_image.astype(np.float32, copy=False)
    else: intensity_image = intensity_image.astype(np.float32, copy=False) # Ensure float32
    if not np.issubdtype(segmentation_mask.dtype, np.integer): segmentation_mask = segmentation_mask.astype(np.int32)
    if not np.issubdtype(soma_mask.dtype, np.integer): soma_mask = soma_mask.astype(np.int32)

    separated_mask = np.copy(segmentation_mask).astype(np.int32) # 2D copy

    # --- Mapping (2D) ---
    print("Mapping 2D seeds to original cell segments...")
    unique_initial_labels = np.unique(segmentation_mask); unique_initial_labels = unique_initial_labels[unique_initial_labels > 0]
    if unique_initial_labels.size == 0: print("No initial segments found."); return separated_mask
    cell_to_somas: Dict[int, Set[int]] = {cell_label: set() for cell_label in unique_initial_labels}
    present_soma_labels = np.unique(soma_mask); present_soma_labels = present_soma_labels[present_soma_labels > 0]
    if present_soma_labels.size == 0: print("No soma seeds found."); return separated_mask

    disable_tqdm_map = len(present_soma_labels) < 10
    for soma_label in tqdm(present_soma_labels, desc="Mapping Seeds (2D)", disable=disable_tqdm_map):
        soma_loc_mask = (soma_mask == soma_label)
        if not np.any(soma_loc_mask): continue
        try:
            cell_labels_under_soma = np.unique(segmentation_mask[soma_loc_mask])
            cell_labels_under_soma = cell_labels_under_soma[cell_labels_under_soma > 0]
            for cell_label in cell_labels_under_soma:
                if cell_label in cell_to_somas: cell_to_somas[cell_label].add(soma_label)
        except IndexError: print(f"Warn: IndexError mapping soma {soma_label}. Skipping.")

    # --- Identify candidates & Label Management (2D) ---
    multi_soma_cell_labels = [lbl for lbl, somas in cell_to_somas.items() if len(somas) > 1]
    print(f"Found {len(multi_soma_cell_labels)} initial 2D segments with multiple seeds...")
    if not multi_soma_cell_labels:
        print("No multi-soma cells identified. Relabeling original mask sequentially.")
        try:
            # Ensure input is integer type
            if not np.issubdtype(separated_mask.dtype, np.integer): separated_mask = separated_mask.astype(np.int32)
            final_sequential_mask, _, inv_map = relabel_sequential(separated_mask)
            final_max_label = len(inv_map) - 1 if len(inv_map) > 0 else 0 # Max label is N
            print(f"Relabeled 2D mask contains {final_max_label} objects.")
            return final_sequential_mask.astype(np.int32) # Ensure correct dtype
        except Exception as e_relabel:
            print(f"Error during sequential relabeling (no splits): {e_relabel}. Returning original mask.")
            return separated_mask.astype(np.int32) # Ensure int type

    max_orig_label = np.max(unique_initial_labels) if unique_initial_labels.size > 0 else 0
    max_soma_label = np.max(present_soma_labels) if present_soma_labels.size > 0 else 0
    next_label = max(max_orig_label, max_soma_label) + 1
    print(f"Tentative starting label for new 2D segments: {next_label}")

    # --- Separation Loop (2D) ---
    skipped_count = 0; processed_count = 0
    current_max_label_in_use = next_label - 1

    for cell_label in tqdm(multi_soma_cell_labels, desc="Separating Segments (2D)"):
        processed_count += 1
        # --- Initialize loop variables ---
        cell_mask = None; bbox_slice = None; slice_obj = None; offset = None
        cell_mask_sub = None; soma_sub = None; intensity_sub = None; cell_soma_sub_mask = None
        seed_props = None; seed_prop_dict = None; adj = None; cost_array = None
        watershed_markers = None; marker_id_reverse_map = None; used_new_labels = set()
        distance_transform = None; landscape = None; watershed_result = None; temp_mask = None
        full_subvol = None

        try:
            # --- Get Sub-Image (2D) ---
            cell_mask = segmentation_mask == cell_label
            obj_slice_list = ndimage.find_objects(cell_mask)
            if not obj_slice_list or obj_slice_list[0] is None or len(obj_slice_list[0]) != 2: continue
            bbox_slice = obj_slice_list[0]; pad = 5
            y_min=max(0,bbox_slice[0].start-pad); x_min=max(0,bbox_slice[1].start-pad)
            y_max=min(segmentation_mask.shape[0],bbox_slice[0].stop+pad); x_max=min(segmentation_mask.shape[1],bbox_slice[1].stop+pad)
            if y_min >= y_max or x_min >= x_max: continue
            slice_obj=np.s_[y_min:y_max, x_min:x_max]; offset=(y_min,x_min)

            cell_mask_sub = cell_mask[slice_obj] # Use cell_mask_sub consistently
            if cell_mask_sub.size == 0 or not np.any(cell_mask_sub): continue
            soma_sub = soma_mask[slice_obj]; intensity_sub = intensity_image[slice_obj] # Use intensity_sub consistently

            # Correct indexing here: use cell_mask_sub
            cell_soma_sub_mask = np.zeros_like(soma_sub, dtype=np.int32); cell_soma_sub_mask[cell_mask_sub] = soma_sub[cell_mask_sub]

            soma_labels_in_cell = np.unique(cell_soma_sub_mask); soma_labels_in_cell = soma_labels_in_cell[soma_labels_in_cell > 0] # Use soma_labels_in_cell consistently
            if len(soma_labels_in_cell) <= 1: continue

            # --- Seed Merging Heuristic ---
            try:
                seed_props = regionprops(cell_soma_sub_mask, intensity_image=intensity_sub) # Pass correct intensity_sub
                seed_prop_dict = {prop.label: prop for prop in seed_props}
                # Validate against the *original* list before filtering
                valid_soma_labels_in_cell = [lbl for lbl in soma_labels_in_cell if lbl in seed_prop_dict]
                if len(valid_soma_labels_in_cell) <= 1: continue
                soma_labels_in_cell = np.array(valid_soma_labels_in_cell) # Update list to only valid ones
            except Exception as e_rp: print(f"Warn: RP Error L{cell_label}: {e_rp}"); continue

            num_seeds = len(soma_labels_in_cell); seed_indices = {lbl: i for i, lbl in enumerate(soma_labels_in_cell)}
            adj = np.zeros((num_seeds, num_seeds), dtype=bool); merge_candidates = False
            max_I_sub = np.max(intensity_sub[cell_mask_sub]) if np.any(cell_mask_sub) else 1.0
            cost_array = np.full(intensity_sub.shape, np.inf, dtype=np.float32); cost_array[cell_mask_sub] = np.maximum(1e-6, max_I_sub - intensity_sub[cell_mask_sub])

            for i in range(num_seeds): # Path check loop
                for j in range(i + 1, num_seeds):
                    # Use the consistent variable name 'soma_labels_in_cell'
                    label1, label2 = soma_labels_in_cell[i], soma_labels_in_cell[j];
                    prop1, prop2 = seed_prop_dict.get(label1), seed_prop_dict.get(label2);
                    if prop1 is None or prop2 is None: continue
                    c1p = np.array(prop1.centroid)*np.array(spacing); c2p = np.array(prop2.centroid)*np.array(spacing); dist=np.linalg.norm(c1p-c2p)
                    if dist > max_seed_centroid_dist: continue
                    c1b = tuple(np.clip(int(round(c)),0,s-1) for c,s in zip(prop1.centroid, cost_array.shape)); c2b = tuple(np.clip(int(round(c)),0,s-1) for c,s in zip(prop2.centroid, cost_array.shape))
                    if not cell_mask_sub[c1b] or not cell_mask_sub[c2b] or cost_array[c1b]==np.inf or cost_array[c2b]==np.inf: continue
                    median_intensity_on_path = 0;
                    try:
                        indices, _ = route_through_array(cost_array,c1b,c2b,True,False);
                        if isinstance(indices,np.ndarray) and indices.ndim==2 and indices.shape[0]==2 and indices.shape[1]>0:
                            path_intensities=intensity_sub[tuple(indices)]; # Use correct intensity_sub
                            if path_intensities.size>0: median_intensity_on_path=np.median(path_intensities)
                    except ValueError: median_intensity_on_path=0
                    except Exception as er: print(f"Warn L{cell_label}: Route Err {label1}/{label2}: {er}")
                    # Use correct prop1, prop2 variables
                    m1=prop1.mean_intensity if hasattr(prop1,'mean_intensity') and prop1.mean_intensity>0 else 1.0;
                    m2=prop2.mean_intensity if hasattr(prop2,'mean_intensity') and prop2.mean_intensity>0 else 1.0
                    refI=max(m1,m2); ratio = median_intensity_on_path/refI if refI>1e-6 else 0.0
                    if ratio >= min_path_intensity_ratio: adj[seed_indices[label1],seed_indices[label2]]=adj[seed_indices[label2],seed_indices[label1]]=True; merge_candidates=True

            # --- Prepare Watershed Markers ---
            watershed_markers = np.zeros_like(cell_mask_sub, dtype=np.int32); marker_id_reverse_map = {}; largest_soma_label = -1; max_soma_size = -1; num_markers_final = 0; current_marker_id = 1
            if merge_candidates:
                n_comp, comp_labels = ndimage.label(adj);
                if n_comp == 1: print(f"    L{cell_label}: All seeds merged. Skip split."); skipped_count+=1; continue
                print(f"    L{cell_label}: {n_comp} groups after merge.")
                for group_idx in range(n_comp): # comp_labels are 0-based for ndimage.label output
                    group_marker_id=current_marker_id; seed_indices_in_group=np.where(comp_labels==group_idx)[0]; group_mask=np.zeros_like(cell_mask_sub,dtype=bool); original_labels_in_group=set(); group_max_soma_size=-1; group_largest_soma_label=-1
                    for seed_idx in seed_indices_in_group:
                         # Use the consistent variable name 'soma_labels_in_cell'
                         original_label=soma_labels_in_cell[seed_idx]; original_labels_in_group.add(original_label); group_mask |= (cell_soma_sub_mask==original_label); prop=seed_prop_dict.get(original_label);
                         if prop and prop.area > group_max_soma_size: group_max_soma_size=prop.area; group_largest_soma_label=original_label
                    watershed_markers[group_mask]=group_marker_id; marker_id_reverse_map[group_marker_id]={'orig_labels': tuple(sorted(list(original_labels_in_group))), 'largest_label': group_largest_soma_label}; current_marker_id+=1
                    if group_max_soma_size > max_soma_size: max_soma_size = group_max_soma_size; largest_soma_label = group_largest_soma_label
                num_markers_final = n_comp
            else: # Individual seeds
                 print(f"    L{cell_label}: No merges. Using {num_seeds} individual markers.")
                 # Use the consistent variable name 'soma_labels_in_cell'
                 for soma_label in soma_labels_in_cell:
                     prop=seed_prop_dict.get(soma_label);
                     if not prop: continue
                     watershed_markers[cell_soma_sub_mask == soma_label]=current_marker_id; marker_id_reverse_map[current_marker_id]={'orig_labels': (soma_label,), 'largest_label': soma_label}
                     if prop.area > max_soma_size: max_soma_size=prop.area; largest_soma_label=soma_label
                     current_marker_id+=1
                 num_markers_final = num_seeds
            if num_markers_final <= 1: print(f"    L{cell_label}: Only {num_markers_final} marker generated. Skip split."); skipped_count+=1; continue

            # --- Landscape and Watershed ---
            distance_transform = ndimage.distance_transform_edt(cell_mask_sub, sampling=spacing); landscape = -distance_transform
            if intensity_weight>1e-6:
                 I_cell=intensity_sub[cell_mask_sub]; # Use correct intensity_sub
                 if I_cell.size>0:
                     minI,maxI=np.min(I_cell),np.max(I_cell); rangeI=maxI-minI;
                     if rangeI>1e-9:
                          # Simplified calculation for inverted intensity term
                          inverted_intensity_term=np.zeros_like(distance_transform,dtype=np.float32);
                          inverted_intensity_term[cell_mask_sub]=(maxI-intensity_sub[cell_mask_sub])/rangeI; # Use intensity_sub
                          max_dist=np.max(distance_transform); # Use max_dist consistently
                          landscape+=intensity_weight*inverted_intensity_term*(max_dist if max_dist>1e-6 else 1.0)
            watershed_markers[~cell_mask_sub]=0; watershed_result = watershed(landscape, watershed_markers, mask=cell_mask_sub, watershed_line=True)

            # --- Post-processing ---
            temp_mask=np.zeros_like(cell_mask_sub, dtype=np.int32); used_new_labels.clear(); current_temp_next_label=current_max_label_in_use+1
            # Use consistent variable name marker_id_for_main_label
            marker_id_for_main_label = next((mid for mid,data in marker_id_reverse_map.items() if data['largest_label']==largest_soma_label), -1)
            ws_labels=np.unique(watershed_result); ws_labels=ws_labels[ws_labels>0]
            for marker_id in ws_labels:
                if marker_id not in marker_id_reverse_map: continue
                final_label=cell_label if marker_id==marker_id_for_main_label else current_temp_next_label # Use consistent marker_id_for_main_label
                if final_label!=cell_label: used_new_labels.add(final_label); current_temp_next_label+=1
                temp_mask[watershed_result==marker_id]=final_label; current_max_label_in_use=max(current_max_label_in_use, final_label)
            # Line fill
            ws_lines=(watershed_result==0)&cell_mask_sub;
            if np.any(ws_lines):
                 labeled_pixels_mask=temp_mask!=0;
                 if np.any(labeled_pixels_mask):
                      _, nearest_label_indices=ndimage.distance_transform_edt(~labeled_pixels_mask,return_indices=True,sampling=spacing); idx_y=np.clip(nearest_label_indices[0],0,temp_mask.shape[0]-1); idx_x=np.clip(nearest_label_indices[1],0,temp_mask.shape[1]-1); nearest_labels=temp_mask[idx_y,idx_x]; temp_mask[ws_lines]=nearest_labels[ws_lines]
                 else: temp_mask[cell_mask_sub]=cell_label
            # Frag merge
            fragments_merged=True; iter_count=0; max_iters=10
            while fragments_merged and iter_count<max_iters:
                 fragments_merged=False; iter_count+=1; labels_to_check=np.unique(temp_mask); labels_to_check=labels_to_check[labels_to_check>0];
                 if len(labels_to_check)<=1: break
                 for lbl in labels_to_check:
                      lbl_mask=temp_mask==lbl; size=np.sum(lbl_mask);
                      if 0<size<min_size_threshold:
                           struct=ndimage.generate_binary_structure(temp_mask.ndim,1); dilated_mask=ndimage.binary_dilation(lbl_mask,structure=struct); neighbor_region=dilated_mask&(~lbl_mask)&cell_mask_sub&(temp_mask!=0);
                           if not np.any(neighbor_region): continue
                           neighbor_labels, neighbor_counts = np.unique(temp_mask[neighbor_region], return_counts=True);
                           if neighbor_labels.size==0: continue
                           largest_neighbor_label=neighbor_labels[np.argmax(neighbor_counts)]; temp_mask[lbl_mask]=largest_neighbor_label; fragments_merged=True;
                           if lbl in used_new_labels: used_new_labels.discard(lbl)
                           break # Restart check
            # Interface merge
            struct_dilate=ndimage.generate_binary_structure(temp_mask.ndim,1); labels_after_frag_merge=np.unique(temp_mask); labels_after_frag_merge=labels_after_frag_merge[labels_after_frag_merge>0]; main_label=cell_label if cell_label in labels_after_frag_merge else -1; new_labels_remaining=used_new_labels.intersection(labels_after_frag_merge)
            if main_label!=-1 and new_labels_remaining:
                 main_mask=(temp_mask==main_label); merged_in_pass=True; merge_iter=0; max_merge_iter=5
                 while merged_in_pass and merge_iter<max_merge_iter:
                    merged_in_pass=False; merge_iter+=1; current_new_labels=list(new_labels_remaining)
                    for new_lbl in current_new_labels:
                        if new_lbl not in np.unique(temp_mask):
                            if new_lbl in new_labels_remaining: new_labels_remaining.remove(new_lbl); continue
                        new_lbl_mask=(temp_mask==new_lbl); dilated_new_mask=ndimage.binary_dilation(new_lbl_mask,structure=struct_dilate); interface_mask=main_mask&dilated_new_mask&cell_mask_sub;
                        if np.any(interface_mask):
                             interface_pixels_count=np.sum(interface_mask);
                             if interface_pixels_count>=post_merge_min_interface_pixels: temp_mask[new_lbl_mask]=main_label;
                             if new_lbl in used_new_labels: used_new_labels.discard(new_lbl)
                             if new_lbl in new_labels_remaining: new_labels_remaining.remove(new_lbl)
                             main_mask=(temp_mask==main_label); merged_in_pass=True # Re-evaluate after merge

            # Update max label
            final_labels_in_temp=np.unique(temp_mask); final_labels_in_temp=final_labels_in_temp[final_labels_in_temp>0];
            if final_labels_in_temp.size>0: current_max_label_in_use=max(current_max_label_in_use, np.max(final_labels_in_temp))

            # --- Map Back Sub-Image to Full Mask ---
            full_subvol = separated_mask[slice_obj]; original_cell_pixels_in_sub=(segmentation_mask[slice_obj]==cell_label)
            full_subvol[original_cell_pixels_in_sub]=temp_mask[original_cell_pixels_in_sub]; separated_mask[slice_obj]=full_subvol

        except MemoryError as e_mem: print(f"MEM ERR L{cell_label}: {e_mem}"); gc.collect(); continue
        except Exception as e_outer: print(f"ERR L{cell_label}: {e_outer}"); traceback.print_exc(); continue
        finally: # Cleanup loop vars
            # Safely delete variables, checking if they exist in locals() first
            vars_to_del = ['cell_mask', 'bbox_slice', 'slice_obj', 'offset',
                           'cell_mask_sub', 'soma_sub', 'intensity_sub', 'cell_soma_sub_mask',
                           'seed_props', 'seed_prop_dict', 'adj', 'cost_array',
                           'watershed_markers', 'marker_id_reverse_map', 'used_new_labels',
                           'distance_transform', 'landscape', 'watershed_result', 'temp_mask',
                           'full_subvol']
            for var_name in vars_to_del:
                if var_name in locals():
                    del locals()[var_name]
            # Periodic GC
            if processed_count > 0 and processed_count % 50 == 0: gc.collect()

    # --- End Separation Loop ---

    print(f"Finished processing {processed_count} 2D multi-soma cells. Skipped splitting {skipped_count}.")
    gc.collect()

    # --- Sequential Relabeling ---
    print("\nRelabeling final 2D mask sequentially...")
    try:
        if not np.issubdtype(separated_mask.dtype, np.integer): separated_mask = separated_mask.astype(np.int32)
        final_unique = np.unique(separated_mask);
        if len(final_unique) <= 1 and 0 in final_unique: print("Final mask empty or contains only background."); return separated_mask
        final_sequential_mask, _, inv_map = relabel_sequential(separated_mask);
        final_max_label = len(inv_map)-1 if len(inv_map)>0 else 0
        print(f"Relabeled 2D mask contains {final_max_label} objects."); return final_sequential_mask.astype(np.int32)
    except Exception as e_relabel:
        print(f"Error during sequential relabeling: {e_relabel}."); traceback.print_exc(); return separated_mask.astype(np.int32)