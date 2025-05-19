# --- START OF REVISED FILE ramified_segmenter.py ---

import numpy as np
from scipy import ndimage
from tqdm import tqdm
from shutil import rmtree
import gc
from functools import partial
from skimage.measure import regionprops, label as skimage_label # type: ignore
from skimage.segmentation import watershed # type: ignore
from skimage.graph import route_through_array # type: ignore
from skimage.morphology import binary_dilation # type: ignore
import math
from sklearn.decomposition import PCA # type: ignore
from skimage.feature import peak_local_max # type: ignore
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from skimage.segmentation import relabel_sequential # type: ignore
import traceback # For detailed error logging

seed = 42
np.random.seed(seed)         # For NumPy


# Helper function for physical size based min_distance (keep as is)
def get_min_distance_pixels(spacing, physical_distance):
    """Calculates minimum distance in pixels for peak_local_max based on physical distance."""
    min_spacing_yx = min(spacing[1:]) # Use YX for in-plane separation
    if min_spacing_yx <= 1e-6: return 3
    pixels = int(round(physical_distance / min_spacing_yx))
    return max(3, pixels)


# --- Helper function for filtering candidate fragments (3D) ---
def _filter_candidate_fragment_3d(
    fragment_mask: np.ndarray, # sub-mask (boolean)
    spacing: Tuple[float, float, float],
    min_seed_fragment_volume: int,
    min_accepted_core_volume: float,
    max_accepted_core_volume: float,
    min_accepted_thickness_um: float, # From fragment's own DT, in um
    max_accepted_thickness_um: float, # From fragment's own DT, in um
    max_allowed_core_aspect_ratio: float
) -> Tuple[Optional[bool], Optional[float], Optional[np.ndarray]]:
    """
    Filters a single 3D candidate fragment based on volume, shape, and thickness.
    Returns: (is_valid, volume, mask_copy)
    is_valid=True: passes all checks.
    is_valid=False: passes min_seed_fragment_volume but fails others (fallback).
    is_valid=None: fails min_seed_fragment_volume (discard).
    """
    # Define variables for finally block
    dt_fragment, labeled_fragment_for_props = None, None
    fragment_mask_copy_local = None # Use a distinct name

    try:
        fragment_volume = np.sum(fragment_mask)
        if fragment_volume < min_seed_fragment_volume:
            return None, None, None # Fails basic size check

        fragment_mask_copy_local = fragment_mask.copy() # Make copy for potential return

        # --- Calculate fragment's own thickness (max radius of inscribed sphere) ---
        dt_fragment = ndimage.distance_transform_edt(fragment_mask_copy_local, sampling=spacing)
        # max_dist_in_fragment_um is already in physical units due to sampling=spacing
        max_dist_in_fragment_um = np.max(dt_fragment)

        passes_thickness = (min_accepted_thickness_um <= max_dist_in_fragment_um <= max_accepted_thickness_um)
        passes_volume_range = (min_accepted_core_volume <= fragment_volume <= max_accepted_core_volume)

        passes_aspect = True # Assume true unless check fails
        aspect_ratio = -1.0
        # Aspect ratio calculation for 3D objects from inertia tensor eigenvalues
        if fragment_volume > 3: # Need a few points for regionprops/inertia tensor
            try:
                # regionprops needs a labeled array. Convert bool to int.
                labeled_fragment_for_props, num_feat_prop = ndimage.label(fragment_mask_copy_local.astype(np.uint8))
                if num_feat_prop > 0:
                    props_list = regionprops(labeled_fragment_for_props) # Can be slow for large fragments
                    if props_list:
                        frag_prop = props_list[0]
                        eigvals_sq = frag_prop.inertia_tensor_eigvals # principal moments of inertia
                        if eigvals_sq is not None and len(eigvals_sq) == 3 and np.all(np.isfinite(eigvals_sq)):
                            # Ensure eigenvalues are positive before sqrt
                            eigvals_sq_abs = np.abs(eigvals_sq)
                            eigvals_sq_sorted = np.sort(eigvals_sq_abs)[::-1] # Largest to smallest
                            if eigvals_sq_sorted[2] > 1e-12: # Smallest eigenvalue (related to shortest axis)
                                # lengths are proportional to sqrt of eigenvalues
                                aspect_ratio = math.sqrt(eigvals_sq_sorted[0]) / math.sqrt(eigvals_sq_sorted[2])
                                if aspect_ratio > max_allowed_core_aspect_ratio:
                                    passes_aspect = False
                            # else: aspect ratio ill-defined (e.g., too flat/thin), assume pass or handle as error
            except Exception as e_prop:
                # print(f"Warn: Regionprops for aspect ratio failed: {e_prop}. Assuming pass for aspect.") # Can be too verbose
                passes_aspect = True
        # else: too small for robust aspect ratio, assume pass

        if passes_thickness and passes_volume_range and passes_aspect:
            return True, fragment_volume, fragment_mask_copy_local # Passed all
        else:
            # Passed min_seed_fragment_volume but failed one or more specific checks
            return False, fragment_volume, fragment_mask_copy_local # Fallback candidate

    except Exception as e_filt:
        print(f"Warn: Unexpected error during 3D fragment filtering: {e_filt}")
        traceback.print_exc()
        if fragment_mask_copy_local is not None: del fragment_mask_copy_local # Clean up if error occurred after copy
        return None, None, None # Discard on unexpected error
    finally:
        if dt_fragment is not None: del dt_fragment
        if labeled_fragment_for_props is not None: del labeled_fragment_for_props


def extract_soma_masks(
    segmentation_mask: np.ndarray, # 3D mask
    intensity_image: Optional[np.ndarray], # 3D intensity image, NEW
    spacing: Optional[Tuple[float, float, float]],
    smallest_quantile: int = 25,
    min_fragment_size: int = 30, # Voxels
    core_volume_target_factor_lower: float = 0.4,
    core_volume_target_factor_upper: float = 10,
    erosion_iterations: int = 1, # NEW PARAMETER for uniform erosion
    intensity_percentiles_to_process: List[int] = [100 - i*5 for i in range(20)] # NEW PARAMETER
) -> np.ndarray:
    """
    Generates candidate seed mask using hybrid approach (DT/Intensity) + uniform erosion + mini-watershed (3D).
    Aggregates results: all valid candidates are prioritized, then fallbacks fill remaining gaps.
    """
    print("--- Starting 3D Soma Extraction (DT/Intensity + Erosion + Mini-WS + Aggregation) ---")

    # --- Input Validation & Setup ---
    run_intensity_path = False
    if intensity_image is None: print("Info: `intensity_image` None. Intensity path skipped.")
    elif intensity_image.shape != segmentation_mask.shape: print(f"Error: `intensity_image` shape mismatch ({intensity_image.shape} vs {segmentation_mask.shape}). Intensity path disabled.")
    elif not np.issubdtype(intensity_image.dtype, np.number): print(f"Error: `intensity_image` not numeric. Intensity path disabled.")
    else: run_intensity_path = True; print("Info: Valid `intensity_image`. Intensity path enabled."); intensity_image = intensity_image.astype(np.float32, copy=False) # Ensure float32 for percentile calcs

    ratios_to_process = [0.3, 0.4, 0.5, 0.6] # For DT path
    print(f"DT ratios: {ratios_to_process}")
    if run_intensity_path:
        intensity_percentiles_to_process = sorted([p for p in intensity_percentiles_to_process if 0 < p < 100], reverse=True)
        print(f"Intensity Percentiles: {intensity_percentiles_to_process}")
    print(f"Uniform Erosion Iterations: {erosion_iterations}")


    # --- Internal Parameters & Setup ---
    min_physical_peak_separation = 7.0 # um, for mini-watershed peak detection
    max_allowed_core_aspect_ratio = 10.0 # For _filter_candidate_fragment_3d
    min_seed_fragment_volume = max(30, min_fragment_size) # Absolute min voxel count for any fragment
    ref_vol_percentile_lower, ref_vol_percentile_upper = 30, 70 # For selecting typical objects for ref thickness
    ref_thickness_percentile_lower = 15 # For deriving min_accepted_thickness_um from reference objects
    absolute_min_thickness_um, absolute_max_thickness_um = 1.5, 10.0 # Hard bounds for fragment thickness

    if spacing is None: spacing = (1.0, 1.0, 1.0); print("Warn: No 3D spacing, assume isotropic.")
    else:
        try: spacing = tuple(float(s) for s in spacing); assert len(spacing) == 3
        except: print(f"Warn: Invalid 3D spacing {spacing}. Using default."); spacing = (1.0, 1.0, 1.0)
    print(f"Using 3D spacing (z,y,x): {spacing}")

    # --- 1. Perform Setup Calculations (Only Once) ---
    print("Calculating initial object properties (3D)...")
    unique_labels_all, counts = np.unique(segmentation_mask[segmentation_mask > 0], return_counts=True)
    if len(unique_labels_all) == 0: return np.zeros_like(segmentation_mask, dtype=np.int32)
    initial_volumes = dict(zip(unique_labels_all, counts))

    print("Finding object bounding boxes and validating labels (3D)...")
    label_to_slice: Dict[int, Tuple[slice, ...]] = {}
    valid_unique_labels_list: List[int] = []
    try:
        object_slices = ndimage.find_objects(segmentation_mask)
        if object_slices is not None:
            num_slices_found = len(object_slices)
            for label in unique_labels_all:
                idx = label - 1
                if 0 <= idx < num_slices_found and object_slices[idx] is not None:
                    s = object_slices[idx]
                    if len(s) == 3 and all(si.start < si.stop for si in s): # Ensure 3D and valid slice
                        label_to_slice[label] = s; valid_unique_labels_list.append(label)
        if not valid_unique_labels_list: print("Error: No valid bounding boxes."); return np.zeros_like(segmentation_mask, dtype=np.int32)
        unique_labels = np.array(valid_unique_labels_list)
    except Exception as e: print(f"Error finding boxes: {e}"); return np.zeros_like(segmentation_mask, dtype=np.int32)

    volumes = {label: initial_volumes[label] for label in unique_labels}; all_volumes_list = list(volumes.values())
    if not all_volumes_list: print("Error: No valid volumes."); return np.zeros_like(segmentation_mask, dtype=np.int32)

    min_samples_for_median = 5
    smallest_thresh_volume = np.percentile(all_volumes_list, smallest_quantile) if len(all_volumes_list) > 1 else (all_volumes_list[0] + 1 if all_volumes_list else 1)
    smallest_object_labels_set = {label for label, vol in volumes.items() if vol <= smallest_thresh_volume}
    target_soma_volume = np.median(all_volumes_list) if len(all_volumes_list) < min_samples_for_median*2 else np.median([volumes[l] for l in smallest_object_labels_set if l in volumes] or all_volumes_list)
    target_soma_volume = max(target_soma_volume, 1.0)
    min_accepted_core_volume = max(min_seed_fragment_volume, target_soma_volume * core_volume_target_factor_lower)
    max_accepted_core_volume = target_soma_volume * core_volume_target_factor_upper
    print(f"Core Volume filter range: [{min_accepted_core_volume:.2f} - {max_accepted_core_volume:.2f}] voxels (Abs min: {min_seed_fragment_volume})")

    # Reference Thickness Calculation (based on max inscribed sphere radius of typical objects)
    if len(all_volumes_list) <= 1: vol_thresh_lower, vol_thresh_upper = min(all_volumes_list)-1 if all_volumes_list else 0, max(all_volumes_list)+1 if all_volumes_list else 1
    else: vol_thresh_lower, vol_thresh_upper = np.percentile(all_volumes_list, ref_vol_percentile_lower), np.percentile(all_volumes_list, ref_vol_percentile_upper)
    reference_object_labels_for_ref = [l for l in unique_labels if vol_thresh_lower < volumes[l] <= vol_thresh_upper]
    if len(reference_object_labels_for_ref) < min_samples_for_median: print(f"Warn: Only {len(reference_object_labels_for_ref)} ref objects. Using all {len(unique_labels)} for ref thickness."); reference_object_labels_for_ref = list(unique_labels)
    print(f"Calculating reference thickness from {len(reference_object_labels_for_ref)} ref objects...");
    max_thicknesses_reference_objs_um = [] # Store thickness in um
    for label_ref in tqdm(reference_object_labels_for_ref, desc="Calc Ref Thickness", disable=len(reference_object_labels_for_ref) < 10):
        bbox_slice = label_to_slice.get(label_ref);
        if bbox_slice is None: continue
        sub_segmentation, object_mask_sub, dist_transform_obj_ref = None, None, None
        try:
            sub_segmentation = segmentation_mask[bbox_slice]
            object_mask_sub = (sub_segmentation == label_ref)
            if not np.any(object_mask_sub): continue
            # DT gives distance in physical units if spacing is provided
            dist_transform_obj_ref = ndimage.distance_transform_edt(object_mask_sub, sampling=spacing)
            max_dist_um = np.max(dist_transform_obj_ref) # This is already in um
            if max_dist_um > 0: max_thicknesses_reference_objs_um.append(max_dist_um)
        except Exception as e: print(f"Warn: Ref Thick Err L{label_ref}: {e}"); continue
        finally:
            if dist_transform_obj_ref is not None: del dist_transform_obj_ref
            if object_mask_sub is not None: del object_mask_sub
            if sub_segmentation is not None: del sub_segmentation

    min_accepted_thickness_um = absolute_min_thickness_um
    if max_thicknesses_reference_objs_um:
        min_accepted_thickness_um = max(absolute_min_thickness_um, np.percentile(max_thicknesses_reference_objs_um, ref_thickness_percentile_lower))
    # Cap the min_accepted_thickness to be less than absolute_max_thickness
    min_accepted_thickness_um = min(min_accepted_thickness_um, absolute_max_thickness_um - 1e-6) # Ensure it's less
    print(f"Final Core Thickness filter range (for fragments): [{min_accepted_thickness_um:.2f} - {absolute_max_thickness_um:.2f}] um")
    min_peak_sep_pixels = get_min_distance_pixels(spacing, min_physical_peak_separation)


    # --- 2. Process Small Cells (Directly to Final Mask) ---
    print("\nProcessing objects (3D)..."); final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32); next_final_label = 1; added_small_labels: Set[int] = set()
    small_cell_labels = [l for l in unique_labels if l in smallest_object_labels_set]; print(f"Processing {len(small_cell_labels)} small objects (3D)...")
    disable_tqdm = len(small_cell_labels) < 10
    for label in tqdm(small_cell_labels, desc="Small Cells", disable=disable_tqdm):
        bbox_slice = label_to_slice.get(label);
        if bbox_slice is None: continue
        sub_segmentation, object_mask_sub, obj_coords = None, None, None # Init for finally
        try:
            sub_segmentation = segmentation_mask[bbox_slice]
            object_mask_sub = (sub_segmentation == label); obj_volume = np.sum(object_mask_sub)
            if obj_volume >= min_seed_fragment_volume:
                offset = (bbox_slice[0].start, bbox_slice[1].start, bbox_slice[2].start)
                obj_coords = np.argwhere(object_mask_sub)
                gz=obj_coords[:,0]+offset[0]; gy=obj_coords[:,1]+offset[1]; gx=obj_coords[:,2]+offset[2]
                valid=(gz>=0)&(gz<final_seed_mask.shape[0])&(gy>=0)&(gy<final_seed_mask.shape[1])&(gx>=0)&(gx<final_seed_mask.shape[2])
                vg = (gz[valid], gy[valid], gx[valid])
                if vg[0].size > 0 and np.all(final_seed_mask[vg] == 0):
                     final_seed_mask[vg] = next_final_label; next_final_label += 1; added_small_labels.add(label)
        except Exception as e: print(f"Warn: Error small L{label}: {e}")
        finally:
            if obj_coords is not None: del obj_coords
            if object_mask_sub is not None: del object_mask_sub
            if sub_segmentation is not None: del sub_segmentation
    print(f"Added {len(added_small_labels)} initial seeds from small cells."); gc.collect()


    # --- 3. Generate Candidates from Large Cells ---
    large_cell_labels = [l for l in unique_labels if l not in added_small_labels]; print(f"Generating candidates from {len(large_cell_labels)} large objects (3D - DT & Intensity)...")
    valid_candidates = []; fallback_candidates = [] # Global lists

    # Define structuring element for erosion once (3D, full connectivity for 3x3x3 neighborhood)
    struct_el_erosion = ndimage.generate_binary_structure(3, 3) if erosion_iterations > 0 else None

    for label in tqdm(large_cell_labels, desc="Large Cell Candidates"):
        bbox_slice = label_to_slice.get(label);
        if bbox_slice is None: continue
        pad = 1
        z_min=max(0,bbox_slice[0].start-pad); y_min=max(0,bbox_slice[1].start-pad); x_min=max(0,bbox_slice[2].start-pad)
        z_max=min(segmentation_mask.shape[0],bbox_slice[0].stop+pad); y_max=min(segmentation_mask.shape[1],bbox_slice[1].stop+pad); x_max=min(segmentation_mask.shape[2],bbox_slice[2].stop+pad)
        if z_min >= z_max or y_min >= y_max or x_min >= x_max: continue
        slice_obj = np.s_[z_min:z_max, y_min:y_max, x_min:x_max]; offset = (z_min, y_min, x_min)

        # Temporary lists for candidates from THIS specific parent label (across all its iterations)
        valid_candidates_for_this_label: List[Dict[str, Any]] = []
        fallback_candidates_for_this_label: List[Dict[str, Any]] = []

        # Define variables used in both paths for potential cleanup in finally
        seg_sub, mask_sub, int_sub_local = None, None, None # Renamed int_sub to avoid conflict
        dist_transform_obj = None # For DT path

        try:
            seg_sub = segmentation_mask[slice_obj]
            mask_sub = (seg_sub == label) # Boolean mask of the parent object in subvolume
            if not np.any(mask_sub): continue

            # --- A) DT Path ---
            try:
                dist_transform_obj = ndimage.distance_transform_edt(mask_sub, sampling=spacing)
                max_dist_in_obj = np.max(dist_transform_obj)
                if max_dist_in_obj > 1e-9:
                    for ratio in ratios_to_process:
                        initial_core_region_mask_sub = (dist_transform_obj >= max_dist_in_obj * ratio) & mask_sub
                        if not np.any(initial_core_region_mask_sub): continue

                        core_mask_for_labeling = initial_core_region_mask_sub
                        if erosion_iterations > 0 and struct_el_erosion is not None:
                            eroded_core_mask = ndimage.binary_erosion(initial_core_region_mask_sub, structure=struct_el_erosion, iterations=erosion_iterations)
                            if not np.any(eroded_core_mask): continue
                            core_mask_for_labeling = eroded_core_mask
                        # else: use initial_core_region_mask_sub

                        labeled_cores, num_cores = ndimage.label(core_mask_for_labeling)
                        if num_cores == 0: continue

                        for core_lbl in range(1, num_cores + 1):
                            core_component_mask_sub = (labeled_cores == core_lbl) # Boolean
                            if not np.any(core_component_mask_sub): continue
                            
                            frags_masks = [] # List of boolean fragment masks
                            dt_core, peaks, markers_core, ws_core = None, None, None, None # Init for finally
                            try: # Mini-WS
                                dt_core = ndimage.distance_transform_edt(core_component_mask_sub, sampling=spacing)
                                if np.max(dt_core) > 1e-9:
                                    peaks = peak_local_max(dt_core, min_distance=min_peak_sep_pixels, labels=core_component_mask_sub, num_peaks_per_label=0, exclude_border=False)
                                else: peaks = np.empty((0, dt_core.ndim), dtype=int)

                                if peaks.shape[0] > 1:
                                    markers_core = np.zeros(dt_core.shape, dtype=np.int32); markers_core[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                                    ws_core = watershed(-dt_core, markers_core, mask=core_component_mask_sub, watershed_line=False)
                                    for ws_l in np.unique(ws_core):
                                        if ws_l == 0: continue
                                        f_mask = (ws_core == ws_l) # Boolean
                                        if np.any(f_mask): frags_masks.append(f_mask)
                                else: # 0 or 1 peak
                                    frags_masks.append(core_component_mask_sub)
                            except Exception as e_ws: print(f"Warn: Mini-WS DT L{label} R{ratio} C{core_lbl}: {e_ws}. Using core."); frags_masks.append(core_component_mask_sub)
                            finally:
                                if dt_core is not None: del dt_core
                                if peaks is not None: del peaks
                                if markers_core is not None: del markers_core
                                if ws_core is not None: del ws_core
                            
                            for f_mask_sub in frags_masks: # Filter fragments
                                is_valid, vol, m_copy = _filter_candidate_fragment_3d(
                                    f_mask_sub, spacing, min_seed_fragment_volume,
                                    min_accepted_core_volume, max_accepted_core_volume,
                                    min_accepted_thickness_um, absolute_max_thickness_um, # Use absolute_max_thickness_um here
                                    max_allowed_core_aspect_ratio
                                )
                                if is_valid is True: valid_candidates_for_this_label.append({'mask': m_copy, 'volume': vol, 'offset': offset})
                                elif is_valid is False: fallback_candidates_for_this_label.append({'mask': m_copy, 'volume': vol, 'offset': offset})
                                # if None, m_copy is not returned/created by _filter
                        # End core_lbl loop
                        if 'labeled_cores' in locals(): del labeled_cores
                    # End ratio loop
            except Exception as e_dt: print(f"Warn: Error in DT path L{label}: {e_dt}")
            finally:
                if dist_transform_obj is not None: del dist_transform_obj

            # --- B) Intensity Path ---
            if run_intensity_path and intensity_image is not None: # intensity_image is full-size
                int_sub_local = intensity_image[slice_obj] # Get subvolume for intensity
                try:
                    ints_obj_values = int_sub_local[mask_sub] # Intensities only within parent object mask
                    if ints_obj_values.size == 0: raise ValueError("No intensity values in object mask")

                    for perc in intensity_percentiles_to_process:
                        try: thresh = np.percentile(ints_obj_values, perc)
                        except IndexError: continue # Should not happen if perc is 0-100

                        initial_core_region_mask_sub = (int_sub_local >= thresh) & mask_sub
                        if not np.any(initial_core_region_mask_sub): continue

                        core_mask_for_labeling = initial_core_region_mask_sub
                        if erosion_iterations > 0 and struct_el_erosion is not None:
                            eroded_core_mask = ndimage.binary_erosion(initial_core_region_mask_sub, structure=struct_el_erosion, iterations=erosion_iterations)
                            if not np.any(eroded_core_mask): continue
                            core_mask_for_labeling = eroded_core_mask
                        
                        labeled_cores, num_cores = ndimage.label(core_mask_for_labeling)
                        if num_cores == 0: continue

                        for core_lbl in range(1, num_cores + 1):
                            core_component_mask_sub = (labeled_cores == core_lbl) # Boolean
                            if not np.any(core_component_mask_sub): continue
                            
                            frags_masks = []
                            dt_core, peaks, markers_core, ws_core = None, None, None, None # Init for finally
                            try: # Mini-WS (same logic as DT path)
                                dt_core = ndimage.distance_transform_edt(core_component_mask_sub, sampling=spacing)
                                if np.max(dt_core) > 1e-9:
                                    peaks = peak_local_max(dt_core, min_distance=min_peak_sep_pixels, labels=core_component_mask_sub, num_peaks_per_label=0, exclude_border=False)
                                else: peaks = np.empty((0, dt_core.ndim), dtype=int)
                                
                                if peaks.shape[0] > 1:
                                    markers_core = np.zeros(dt_core.shape, dtype=np.int32); markers_core[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                                    ws_core = watershed(-dt_core, markers_core, mask=core_component_mask_sub, watershed_line=False)
                                    for ws_l in np.unique(ws_core):
                                        if ws_l == 0: continue
                                        f_mask = (ws_core == ws_l)
                                        if np.any(f_mask): frags_masks.append(f_mask)
                                else: frags_masks.append(core_component_mask_sub)
                            except Exception as e_ws: print(f"Warn: Mini-WS Int L{label} P{perc} C{core_lbl}: {e_ws}. Using core."); frags_masks.append(core_component_mask_sub)
                            finally:
                                if dt_core is not None: del dt_core
                                if peaks is not None: del peaks
                                if markers_core is not None: del markers_core
                                if ws_core is not None: del ws_core

                            for f_mask_sub in frags_masks: # Filter fragments
                                is_valid, vol, m_copy = _filter_candidate_fragment_3d(
                                    f_mask_sub, spacing, min_seed_fragment_volume,
                                    min_accepted_core_volume, max_accepted_core_volume,
                                    min_accepted_thickness_um, absolute_max_thickness_um,
                                    max_allowed_core_aspect_ratio
                                )
                                if is_valid is True: valid_candidates_for_this_label.append({'mask': m_copy, 'volume': vol, 'offset': offset})
                                elif is_valid is False: fallback_candidates_for_this_label.append({'mask': m_copy, 'volume': vol, 'offset': offset})
                        # End core_lbl loop
                        if 'labeled_cores' in locals(): del labeled_cores
                    # End percentile loop
                except ValueError: pass # No intensity values, or other expected issue
                except Exception as e_int: print(f"Warn: Error in Intensity path L{label}: {e_int}")
                finally:
                    if int_sub_local is not None: del int_sub_local # Clean up intensity subvolume
            # End Intensity Path

            # --- Combine Candidates from this parent label to global lists ---
            if valid_candidates_for_this_label:
                valid_candidates.extend(valid_candidates_for_this_label)
            if fallback_candidates_for_this_label: # Add all fallbacks collected for this parent
                fallback_candidates.extend(fallback_candidates_for_this_label)

        except MemoryError: print(f"Warn: MemError processing L{label}."); gc.collect()
        except Exception as e_label_proc: print(f"Warn: Uncaught Error L{label}: {e_label_proc}"); traceback.print_exc()
        finally: # Cleanup for current parent label
            if seg_sub is not None: del seg_sub
            if mask_sub is not None: del mask_sub
            # Clean up masks within the per-label lists as they are now in global or discarded
            for item in valid_candidates_for_this_label:
                if 'mask' in item: del item['mask']
            for item in fallback_candidates_for_this_label:
                if 'mask' in item: del item['mask']
            del valid_candidates_for_this_label, fallback_candidates_for_this_label
            gc.collect()
        # --- End Label Loop (Large Cells) ---

    print(f"Generated {len(valid_candidates)} valid candidates and {len(fallback_candidates)} fallback candidates.")

    # --- 4. Place Valid Candidates (Smallest First) ---
    print("Placing smallest valid seeds..."); valid_candidates.sort(key=lambda x: x['volume'])
    processed_count = 0
    for cand in tqdm(valid_candidates, desc="Placing Valid Seeds"):
        mask_sub = cand.get('mask'); offset_cand = cand.get('offset')
        if mask_sub is None or offset_cand is None or mask_sub.size == 0: continue
        try:
            coords = np.argwhere(mask_sub)
            if coords.size == 0: continue
            gz=coords[:,0]+offset_cand[0]; gy=coords[:,1]+offset_cand[1]; gx=coords[:,2]+offset_cand[2]
            valid=(gz>=0)&(gz<final_seed_mask.shape[0])&(gy>=0)&(gy<final_seed_mask.shape[1])&(gx>=0)&(gx<final_seed_mask.shape[2])
            vg = (gz[valid], gy[valid], gx[valid])
            if vg[0].size > 0 and np.all(final_seed_mask[vg] == 0):
                final_seed_mask[vg] = next_final_label; next_final_label += 1; processed_count += 1
        except Exception as e: print(f"Warn: Error Place Valid L: {e}")
        finally:
             if 'mask' in cand: del cand['mask'] # Critical: delete mask copy
    print(f"Placed {processed_count} seeds from valid candidates."); del valid_candidates; gc.collect()


    # --- 5. Place Fallback Candidates (Fill Gaps, Smallest First) ---
    print("Placing fallback seeds..."); fallback_candidates.sort(key=lambda x: x['volume'])
    fallback_count = 0
    for fallbk in tqdm(fallback_candidates, desc="Placing Fallbacks"):
        mask_sub = fallbk.get('mask'); offset_fallbk = fallbk.get('offset')
        if mask_sub is None or offset_fallbk is None or mask_sub.size == 0: continue
        try:
            coords = np.argwhere(mask_sub)
            if coords.size == 0: continue
            gz=coords[:,0]+offset_fallbk[0]; gy=coords[:,1]+offset_fallbk[1]; gx=coords[:,2]+offset_fallbk[2]
            valid=(gz>=0)&(gz<final_seed_mask.shape[0])&(gy>=0)&(gy<final_seed_mask.shape[1])&(gx>=0)&(gx<final_seed_mask.shape[2])
            vg = (gz[valid], gy[valid], gx[valid])
            if vg[0].size > 0 and np.all(final_seed_mask[vg] == 0):
                final_seed_mask[vg] = next_final_label; next_final_label += 1; fallback_count += 1
        except Exception as e: print(f"Warn: Error Place Fallback: {e}")
        finally:
            if 'mask' in fallbk: del fallbk['mask'] # Critical: delete mask copy
    print(f"Placed {fallback_count} seeds from fallback candidates."); del fallback_candidates; gc.collect()

    total_final_seeds = next_final_label - 1
    print(f"\nGenerated {total_final_seeds} final aggregated seeds (3D)."); print("--- Finished 3D Intermediate Seed Extraction ---")
    return final_seed_mask


# --- refine_seeds_pca function remains the same ---
def refine_seeds_pca(
    intermediate_seed_mask: np.ndarray,
    spacing: Optional[Tuple[float, float, float]],
    target_aspect_ratio: float = 1.1,
    projection_percentile_crop: int = 10,
    min_fragment_size: int = 30
) -> np.ndarray:
    """
    Refines the shape of preliminary seeds using PCA to make them more compact.
    (Docstring from original file - details parameters and their influence)
    """
    print("--- Starting 3D PCA Seed Refinement ---")
    if spacing is None: spacing = (1.0, 1.0, 1.0); print("Warn: No 3D spacing, assume isotropic.")
    else: print(f"Using 3D spacing (z,y,x): {spacing}")
    print(f"Parameters: target_aspect_ratio={target_aspect_ratio}, projection_crop%={projection_percentile_crop}, min_vol={min_fragment_size}")
    final_refined_mask = np.zeros_like(intermediate_seed_mask, dtype=np.int32)
    next_final_label = 1; kept_refined_count = 0
    seed_labels = np.unique(intermediate_seed_mask); seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels) == 0: print("Input intermediate seed mask is empty."); return final_refined_mask

    # Efficiently get bounding boxes
    seed_slices = ndimage.find_objects(intermediate_seed_mask)
    label_to_slice = {}
    if seed_slices is not None:
        label_to_slice = {label: seed_slices[label-1] for label in seed_labels if label-1 < len(seed_slices) and seed_slices[label-1] is not None and len(seed_slices[label-1])==3}

    for label in tqdm(seed_labels, desc="Refining Seeds PCA (3D)"):
        bbox_slice = label_to_slice.get(label);
        if bbox_slice is None: continue
        offset = (bbox_slice[0].start, bbox_slice[1].start, bbox_slice[2].start); slice_obj = bbox_slice
        # Define variables before try for robust finally block
        seed_mask_sub, coords_vox, coords_phys, pca, refined_mask_sub = None, None, None, None, None
        coords_centered_phys, proj1, kept_coords_vox, refined_coords = None, None, None, None
        try:
            seed_mask_sub = (intermediate_seed_mask[slice_obj] == label); current_volume = np.sum(seed_mask_sub)
            if not np.any(seed_mask_sub) or current_volume < min_fragment_size: continue # Check if empty or too small

            coords_vox = np.argwhere(seed_mask_sub)
            if coords_vox.shape[0] <= 3: refined_mask_sub = seed_mask_sub # Need > 3 points for 3D PCA
            else:
                coords_phys = coords_vox * np.array(spacing) # Apply 3D spacing
                pca = PCA(n_components=3); pca.fit(coords_phys); eigenvalues = pca.explained_variance_; eigenvectors = pca.components_ # 3 components
                sorted_indices = np.argsort(eigenvalues)[::-1]; eigenvalues = eigenvalues[sorted_indices]; eigenvectors = eigenvectors[sorted_indices]
                if eigenvalues[2] < 1e-9: eigenvalues[2] = 1e-9 # Avoid division by zero for aspect ratio
                aspect_ratio = math.sqrt(eigenvalues[0]) / math.sqrt(eigenvalues[2]) # Longest/Shortest axis length ratio

                if aspect_ratio >= target_aspect_ratio:
                    coords_centered_phys = coords_phys - pca.mean_; proj1 = coords_centered_phys @ eigenvectors[0] # Project onto longest axis
                    min_p = np.percentile(proj1, projection_percentile_crop); max_p = np.percentile(proj1, 100 - projection_percentile_crop)
                    voxel_indices_to_keep = (proj1 >= min_p) & (proj1 <= max_p)
                    refined_mask_sub = np.zeros_like(seed_mask_sub, dtype=bool)
                    kept_coords_vox = coords_vox[voxel_indices_to_keep]
                    if kept_coords_vox.shape[0] > 0: refined_mask_sub[tuple(kept_coords_vox.T)] = True
                    refined_volume = np.sum(refined_mask_sub)
                    if refined_volume < min_fragment_size: refined_mask_sub = None # Discard if refinement makes it too small
                else: refined_mask_sub = seed_mask_sub # Keep original

            if refined_mask_sub is not None and np.any(refined_mask_sub):
                 final_volume = np.sum(refined_mask_sub) # Re-check volume
                 if final_volume >= min_fragment_size:
                     refined_coords = np.argwhere(refined_mask_sub)
                     if refined_coords.size > 0:
                         gz=refined_coords[:,0]+offset[0]; gy=refined_coords[:,1]+offset[1]; gx=refined_coords[:,2]+offset[2]
                         valid=((gz>=0)&(gz<final_refined_mask.shape[0])&(gy>=0)&(gy<final_refined_mask.shape[1])&(gx>=0)&(gx<final_refined_mask.shape[2]))
                         vgz,vgy,vgx = gz[valid],gy[valid],gx[valid]
                         if vgz.size > 0: # Ensure not empty before writing
                             final_refined_mask[vgz, vgy, vgx] = next_final_label
                             next_final_label += 1; kept_refined_count += 1
        except Exception as e: print(f"Warning: Error during 3D PCA refinement for seed label {label}: {e}")
        finally: # Cleanup vars
            if seed_mask_sub is not None: del seed_mask_sub
            if coords_vox is not None: del coords_vox
            if coords_phys is not None: del coords_phys
            if pca is not None: del pca
            if refined_mask_sub is not None: del refined_mask_sub
            if coords_centered_phys is not None: del coords_centered_phys
            if proj1 is not None: del proj1
            if kept_coords_vox is not None: del kept_coords_vox
            if refined_coords is not None: del refined_coords

    print(f"Kept {kept_refined_count} refined 3D seeds after PCA.")
    print("--- Finished 3D PCA Seed Refinement ---"); gc.collect(); return final_refined_mask


def separate_multi_soma_cells(
    segmentation_mask: np.ndarray, # 3D
    intensity_volume: np.ndarray, # 3D
    soma_mask: np.ndarray, # 3D, output from refine_seeds_pca
    spacing: Optional[Tuple[float, float, float]], # 3D
    min_size_threshold: int = 100, # Voxels, for merging small fragments post-watershed
    intensity_weight: float = 0.0, # For watershed landscape
    max_seed_centroid_dist: float = 15.0, # um, for seed merging heuristic
    min_path_intensity_ratio: float = 0.6, # For seed merging heuristic
    post_merge_min_interface_voxels: int = 20 # MODIFIED: Replaces neck thickness for simpler merge
    ) -> np.ndarray:
    """
    Separates 3D cell segmentations containing multiple seeds.
    Uses path intensity heuristic for seed merging, watershed, and post-watershed merging
    based on interface voxel count. Output is sequentially relabeled.
    """
    print(f"--- Starting 3D Multi-Soma Separation (Path Heuristics + Interface Voxel Merge) ---")

    # --- Parameter & Spacing Checks (3D) ---
    print(f"Post-merge check: min_interface_voxels={post_merge_min_interface_voxels}") # MODIFIED parameter
    if spacing is None: spacing = (1.0, 1.0, 1.0); print("Warn: No 3D spacing, assume isotropic.")
    else:
        try: spacing = tuple(float(s) for s in spacing); assert len(spacing) == 3
        except: print(f"Error: Invalid 3D spacing ({spacing}). Using default."); spacing = (1.0, 1.0, 1.0)
    print(f"Using 3D spacing (z,y,x): {spacing}")
    print(f"Seed merging heuristics: max_dist={max_seed_centroid_dist}um, min_path_intensity_ratio={min_path_intensity_ratio}")

    # --- Input Validation & Type Checks (3D) ---
    if segmentation_mask.ndim != 3 or intensity_volume.ndim != 3 or soma_mask.ndim != 3:
        raise ValueError("All input masks and volume must be 3D.")
    if segmentation_mask.shape != intensity_volume.shape or segmentation_mask.shape != soma_mask.shape:
        raise ValueError("Input mask/volume shapes do not match.")
    if not np.issubdtype(intensity_volume.dtype, np.floating): intensity_volume = intensity_volume.astype(np.float32, copy=False)
    else: intensity_volume = intensity_volume.astype(np.float32, copy=False) # Ensure float32
    if not np.issubdtype(segmentation_mask.dtype, np.integer): segmentation_mask = segmentation_mask.astype(np.int32)
    if not np.issubdtype(soma_mask.dtype, np.integer): soma_mask = soma_mask.astype(np.int32)

    separated_mask = np.copy(segmentation_mask).astype(np.int32) # Work on a 3D copy

    # --- Mapping (3D) ---
    print("Mapping 3D seeds to original cell segments...")
    unique_initial_labels = np.unique(segmentation_mask); unique_initial_labels = unique_initial_labels[unique_initial_labels > 0]
    if unique_initial_labels.size == 0: print("No initial segments found."); return separated_mask # Already int32
    cell_to_somas: Dict[int, Set[int]] = {cell_label: set() for cell_label in unique_initial_labels}
    present_soma_labels = np.unique(soma_mask); present_soma_labels = present_soma_labels[present_soma_labels > 0]
    if present_soma_labels.size == 0: print("No soma seeds found."); return separated_mask # Already int32

    for soma_label in tqdm(present_soma_labels, desc="Mapping Seeds (3D)", disable=len(present_soma_labels) < 10):
        soma_loc_mask = (soma_mask == soma_label)
        if not np.any(soma_loc_mask): continue
        try:
            cell_labels_under_soma = np.unique(segmentation_mask[soma_loc_mask])
            cell_labels_under_soma = cell_labels_under_soma[cell_labels_under_soma > 0]
            for cell_label in cell_labels_under_soma:
                if cell_label in cell_to_somas: cell_to_somas[cell_label].add(soma_label)
        except IndexError: print(f"Warn: IndexError mapping soma {soma_label}. Skipping.")


    multi_soma_cell_labels = [lbl for lbl, somas in cell_to_somas.items() if len(somas) > 1]
    print(f"Found {len(multi_soma_cell_labels)} initial 3D segments with multiple seeds...")
    if not multi_soma_cell_labels:
        print("No multi-soma cells identified. Relabeling original mask sequentially.")
        try: # Relabel and return if no splits needed
            final_sequential_mask, _, inv_map = relabel_sequential(separated_mask.astype(np.int32)) # Ensure int before relabel
            final_max_label = len(inv_map) -1 if len(inv_map) > 0 else 0
            print(f"Relabeled 3D mask contains {final_max_label} objects.")
            return final_sequential_mask.astype(np.int32)
        except Exception as e_relabel:
            print(f"Error during sequential relabeling (no splits): {e_relabel}. Returning original mask.");
            return separated_mask.astype(np.int32)

    max_orig_label = np.max(unique_initial_labels) if unique_initial_labels.size > 0 else 0
    max_soma_label = np.max(present_soma_labels) if present_soma_labels.size > 0 else 0
    next_label = max(max_orig_label, max_soma_label) + 1
    print(f"Tentative starting label for new 3D segments: {next_label}")

    skipped_count = 0; processed_count = 0
    current_max_label_in_use = next_label - 1

    for cell_label in tqdm(multi_soma_cell_labels, desc="Separating Segments (3D)"):
        processed_count += 1
        # Initialize loop variables for finally block
        cell_mask, bbox_slice, slice_obj, offset = None, None, None, None
        cell_mask_sub, soma_sub, intensity_sub_local, cell_soma_sub_mask = None, None, None, None
        seed_props, seed_prop_dict, adj, cost_array = None, None, None, None
        watershed_markers, marker_id_reverse_map, used_new_labels = None, None, set()
        distance_transform, landscape, watershed_result, temp_mask = None, None, None, None
        full_subvol = None

        try:
            cell_mask = (segmentation_mask == cell_label)
            obj_slice_list = ndimage.find_objects(cell_mask)
            if not obj_slice_list or obj_slice_list[0] is None or len(obj_slice_list[0]) != 3: continue
            bbox_slice = obj_slice_list[0]; pad = 5
            z_min=max(0,bbox_slice[0].start-pad); y_min=max(0,bbox_slice[1].start-pad); x_min=max(0,bbox_slice[2].start-pad)
            z_max=min(segmentation_mask.shape[0],bbox_slice[0].stop+pad); y_max=min(segmentation_mask.shape[1],bbox_slice[1].stop+pad); x_max=min(segmentation_mask.shape[2],bbox_slice[2].stop+pad)
            if z_min >= z_max or y_min >= y_max or x_min >= x_max: continue
            slice_obj=np.s_[z_min:z_max, y_min:y_max, x_min:x_max]; offset=(z_min,y_min,x_min)

            cell_mask_sub = cell_mask[slice_obj]
            if not np.any(cell_mask_sub): continue
            soma_sub = soma_mask[slice_obj]; intensity_sub_local = intensity_volume[slice_obj]

            cell_soma_sub_mask = np.zeros_like(soma_sub, dtype=np.int32); cell_soma_sub_mask[cell_mask_sub] = soma_sub[cell_mask_sub]
            soma_labels_in_cell = np.unique(cell_soma_sub_mask); soma_labels_in_cell = soma_labels_in_cell[soma_labels_in_cell > 0]
            if len(soma_labels_in_cell) <= 1: continue

            try:
                seed_props = regionprops(cell_soma_sub_mask, intensity_image=intensity_sub_local)
                seed_prop_dict = {prop.label: prop for prop in seed_props}
                valid_soma_labels_in_cell = [lbl for lbl in soma_labels_in_cell if lbl in seed_prop_dict]
                if len(valid_soma_labels_in_cell) <= 1: continue
                soma_labels_in_cell = np.array(valid_soma_labels_in_cell)
            except Exception as e_rp: print(f"Warn: RP Error L{cell_label}: {e_rp}"); continue

            num_seeds = len(soma_labels_in_cell); seed_indices = {lbl: i for i, lbl in enumerate(soma_labels_in_cell)}
            adj = np.zeros((num_seeds, num_seeds), dtype=bool); merge_candidates = False
            max_I_sub = np.max(intensity_sub_local[cell_mask_sub]) if np.any(cell_mask_sub) else 1.0
            cost_array = np.full(intensity_sub_local.shape, np.inf, dtype=np.float32)
            cost_array[cell_mask_sub] = np.maximum(1e-6, max_I_sub - intensity_sub_local[cell_mask_sub])

            for i in range(num_seeds):
                for j in range(i + 1, num_seeds):
                    label1, label2 = soma_labels_in_cell[i], soma_labels_in_cell[j]
                    prop1, prop2 = seed_prop_dict.get(label1), seed_prop_dict.get(label2)
                    if prop1 is None or prop2 is None: continue
                    c1p=np.array(prop1.centroid)*np.array(spacing); c2p=np.array(prop2.centroid)*np.array(spacing); dist=np.linalg.norm(c1p-c2p)
                    if dist > max_seed_centroid_dist: continue
                    c1b=tuple(np.clip(int(round(c)),0,s-1) for c,s in zip(prop1.centroid, cost_array.shape))
                    c2b=tuple(np.clip(int(round(c)),0,s-1) for c,s in zip(prop2.centroid, cost_array.shape))
                    if not cell_mask_sub[c1b] or not cell_mask_sub[c2b] or cost_array[c1b]==np.inf or cost_array[c2b]==np.inf: continue
                    
                    median_intensity_on_path = 0.0
                    try: # Pathfinding
                        indices, _ = route_through_array(cost_array, c1b, c2b, fully_connected=True, geometric=False)
                        if isinstance(indices, np.ndarray) and indices.ndim == 2 and indices.shape[0] == 3 and indices.shape[1] > 0:
                            path_intensities = intensity_sub_local[tuple(indices)]
                            if path_intensities.size > 0: median_intensity_on_path = np.median(path_intensities)
                    except ValueError: median_intensity_on_path = 0.0 # Path not found
                    except Exception as er: print(f"Warn L{cell_label}: Route Err {label1}/{label2}: {er}")

                    # Using mean_intensity as in 2D for reference
                    m1 = prop1.mean_intensity if hasattr(prop1, 'mean_intensity') and prop1.mean_intensity is not None and prop1.mean_intensity > 1e-6 else 1.0
                    m2 = prop2.mean_intensity if hasattr(prop2, 'mean_intensity') and prop2.mean_intensity is not None and prop2.mean_intensity > 1e-6 else 1.0
                    ref_I = max(m1, m2)
                    ratio = median_intensity_on_path / ref_I if ref_I > 1e-6 else 0.0
                    if ratio >= min_path_intensity_ratio:
                        adj[seed_indices[label1], seed_indices[label2]] = adj[seed_indices[label2], seed_indices[label1]] = True
                        merge_candidates = True
            
            watershed_markers = np.zeros_like(cell_mask_sub, dtype=np.int32); marker_id_reverse_map = {}
            largest_soma_label = -1; max_soma_size = -1; num_markers_final = 0; current_marker_id = 1
            if merge_candidates:
                n_comp, comp_labels = ndimage.label(adj) # comp_labels are 0-based if input is bool, 1-based if int
                # Assuming ndimage.label on boolean adj gives 0-based components if any, or 0 if no components.
                # If n_comp is based on unique labels in comp_labels (excluding 0 if it's background), range should be fine.
                # Let's adjust comp_labels to be 1-based for groups if they are 0-based.
                # If adj is all False, n_comp could be 0. If all True, n_comp=1.
                if n_comp == num_seeds and not np.any(adj): # No merges actually happened, all are separate components
                     pass # Fall through to individual seed marker logic
                elif n_comp == 1 and np.all(adj): # All seeds merged into one group
                    print(f"    L{cell_label}: All seeds merged. Skip split."); skipped_count+=1; continue
                
                # Process merged groups
                print(f"    L{cell_label}: {n_comp} seed groups after merge attempt.")
                unique_comp_labels = np.unique(comp_labels) # These are the actual component labels assigned by ndimage.label
                
                processed_as_merged_group = False
                for group_val in unique_comp_labels: # Iterate through actual component labels
                    # This loop structure assumes comp_labels from ndimage.label are 0..N-1 or 1..N for N components
                    # It's safer to iterate unique_comp_labels and map them to current_marker_id
                    if np.sum(comp_labels == group_val) == 0 : continue # Skip if somehow an empty component label shows up

                    group_marker_id = current_marker_id
                    seed_indices_in_group = np.where(comp_labels == group_val)[0]
                    if not seed_indices_in_group.size: continue

                    processed_as_merged_group = True
                    group_mask = np.zeros_like(cell_mask_sub, dtype=bool); original_labels_in_group = set()
                    group_max_soma_size = -1; group_largest_soma_label = -1
                    for seed_idx in seed_indices_in_group:
                        original_label = soma_labels_in_cell[seed_idx]; original_labels_in_group.add(original_label)
                        group_mask |= (cell_soma_sub_mask == original_label)
                        prop = seed_prop_dict.get(original_label)
                        if prop and prop.area > group_max_soma_size: group_max_soma_size = prop.area; group_largest_soma_label = original_label
                    
                    watershed_markers[group_mask] = group_marker_id
                    marker_id_reverse_map[group_marker_id] = {'orig_labels': tuple(sorted(list(original_labels_in_group))), 'largest_label': group_largest_soma_label}
                    current_marker_id += 1
                    if group_max_soma_size > max_soma_size: max_soma_size = group_max_soma_size; largest_soma_label = group_largest_soma_label
                
                num_markers_final = current_marker_id -1 # Number of groups formed
                if not processed_as_merged_group : # Fallback to individual if merge logic was skipped
                    merge_candidates = False # Force individual processing

            if not merge_candidates: # Individual seeds (no merges or merge attempt resulted in no actual groups)
                 print(f"    L{cell_label}: No merges. Using {num_seeds} individual markers.")
                 for soma_label_val in soma_labels_in_cell: # Use a different var name
                     prop = seed_prop_dict.get(soma_label_val)
                     if not prop: continue
                     watershed_markers[cell_soma_sub_mask == soma_label_val] = current_marker_id
                     marker_id_reverse_map[current_marker_id] = {'orig_labels': (soma_label_val,), 'largest_label': soma_label_val}
                     if prop.area > max_soma_size: max_soma_size = prop.area; largest_soma_label = soma_label_val
                     current_marker_id += 1
                 num_markers_final = num_seeds

            if num_markers_final <= 1: print(f"    L{cell_label}: Only {num_markers_final} marker. Skip split."); skipped_count+=1; continue

            distance_transform = ndimage.distance_transform_edt(cell_mask_sub, sampling=spacing); landscape = -distance_transform
            if intensity_weight > 1e-6:
                 I_cell = intensity_sub_local[cell_mask_sub]
                 if I_cell.size > 0:
                     minI, maxI = np.min(I_cell), np.max(I_cell); rangeI = maxI - minI
                     if rangeI > 1e-9:
                          inverted_intensity_term = np.zeros_like(distance_transform, dtype=np.float32)
                          inverted_intensity_term[cell_mask_sub] = (maxI - intensity_sub_local[cell_mask_sub]) / rangeI
                          max_dist_val = np.max(distance_transform) # Consistent var name
                          landscape += intensity_weight * inverted_intensity_term * (max_dist_val if max_dist_val > 1e-6 else 1.0)
            
            watershed_markers[~cell_mask_sub] = 0 # Ensure markers are only within the cell mask
            watershed_result = watershed(landscape, watershed_markers, mask=cell_mask_sub, watershed_line=True)

            temp_mask = np.zeros_like(cell_mask_sub, dtype=np.int32); used_new_labels.clear()
            current_temp_next_label = current_max_label_in_use + 1
            marker_id_for_main_label = next((mid for mid, data in marker_id_reverse_map.items() if data['largest_label'] == largest_soma_label), -1)

            ws_labels_unique = np.unique(watershed_result); ws_labels_unique = ws_labels_unique[ws_labels_unique > 0]
            for marker_id_val in ws_labels_unique: # Iterate over actual watershed labels present
                if marker_id_val not in marker_id_reverse_map: continue # Should not happen if markers are correct
                final_label_val = cell_label if marker_id_val == marker_id_for_main_label else current_temp_next_label
                if final_label_val != cell_label: used_new_labels.add(final_label_val); current_temp_next_label += 1
                temp_mask[watershed_result == marker_id_val] = final_label_val
                current_max_label_in_use = max(current_max_label_in_use, final_label_val)

            ws_lines = (watershed_result == 0) & cell_mask_sub
            if np.any(ws_lines):
                 labeled_pixels_mask = temp_mask != 0
                 if np.any(labeled_pixels_mask):
                      _, nearest_label_indices = ndimage.distance_transform_edt(~labeled_pixels_mask, return_indices=True, sampling=spacing)
                      # Clip indices to be within bounds of temp_mask
                      idx_z = np.clip(nearest_label_indices[0], 0, temp_mask.shape[0]-1)
                      idx_y = np.clip(nearest_label_indices[1], 0, temp_mask.shape[1]-1)
                      idx_x = np.clip(nearest_label_indices[2], 0, temp_mask.shape[2]-1)
                      nearest_labels = temp_mask[idx_z, idx_y, idx_x]
                      temp_mask[ws_lines] = nearest_labels[ws_lines]
                 else: temp_mask[cell_mask_sub] = cell_label # Whole cell became watershed line

            fragments_merged = True; iter_count = 0; max_iters = 10 # Small frag merge
            while fragments_merged and iter_count < max_iters:
                 fragments_merged = False; iter_count += 1
                 labels_to_check = np.unique(temp_mask); labels_to_check = labels_to_check[labels_to_check > 0]
                 if len(labels_to_check) <= 1: break
                 for lbl_val in labels_to_check:
                      lbl_mask = temp_mask == lbl_val; size = np.sum(lbl_mask)
                      if 0 < size < min_size_threshold:
                           struct = ndimage.generate_binary_structure(temp_mask.ndim, 1) # 6-connectivity for neighbors
                           dilated_mask = ndimage.binary_dilation(lbl_mask, structure=struct)
                           neighbor_region = dilated_mask & (~lbl_mask) & cell_mask_sub & (temp_mask != 0)
                           if not np.any(neighbor_region): continue
                           neighbor_labels, neighbor_counts = np.unique(temp_mask[neighbor_region], return_counts=True)
                           if neighbor_labels.size == 0: continue
                           largest_neighbor_label = neighbor_labels[np.argmax(neighbor_counts)]
                           temp_mask[lbl_mask] = largest_neighbor_label; fragments_merged = True
                           if lbl_val in used_new_labels: used_new_labels.discard(lbl_val)
                           break # Restart check
            
            # MODIFIED Post-Watershed Merging (Interface Voxel Count)
            struct_dilate = ndimage.generate_binary_structure(temp_mask.ndim, 1) # For finding interface
            labels_after_frag_merge = np.unique(temp_mask); labels_after_frag_merge = labels_after_frag_merge[labels_after_frag_merge > 0]
            main_label_in_temp = cell_label if cell_label in labels_after_frag_merge else -1 # Check if original label still exists
            new_labels_remaining = used_new_labels.intersection(labels_after_frag_merge)

            if main_label_in_temp != -1 and new_labels_remaining:
                 main_mask = (temp_mask == main_label_in_temp)
                 merged_in_pass = True; merge_iter = 0; max_merge_iter=5 # Allow multiple passes for chained merges
                 while merged_in_pass and merge_iter < max_merge_iter:
                    merged_in_pass = False; merge_iter +=1
                    current_new_labels_list = list(new_labels_remaining) # Iterate over a copy
                    for new_lbl in current_new_labels_list:
                        if new_lbl not in np.unique(temp_mask): # Already merged in this pass by another
                            if new_lbl in new_labels_remaining: new_labels_remaining.remove(new_lbl)
                            continue
                        new_lbl_mask = (temp_mask == new_lbl)
                        # Find interface: dilate new_lbl_mask and AND with main_mask AND cell_mask_sub
                        dilated_new_mask = ndimage.binary_dilation(new_lbl_mask, structure=struct_dilate)
                        interface_mask = main_mask & dilated_new_mask & cell_mask_sub # Ensure interface is within cell
                        
                        if np.any(interface_mask):
                             interface_voxels_count = np.sum(interface_mask)
                             if interface_voxels_count >= post_merge_min_interface_voxels:
                                 temp_mask[new_lbl_mask] = main_label_in_temp # Merge
                                 if new_lbl in used_new_labels: used_new_labels.discard(new_lbl)
                                 if new_lbl in new_labels_remaining: new_labels_remaining.remove(new_lbl)
                                 main_mask = (temp_mask == main_label_in_temp) # Update main_mask
                                 merged_in_pass = True # Signal a merge happened, might need another pass
            
            final_labels_in_temp = np.unique(temp_mask); final_labels_in_temp = final_labels_in_temp[final_labels_in_temp > 0]
            if final_labels_in_temp.size > 0: current_max_label_in_use = max(current_max_label_in_use, np.max(final_labels_in_temp))

            full_subvol = separated_mask[slice_obj]
            original_cell_pixels_in_sub = (segmentation_mask[slice_obj] == cell_label)
            full_subvol[original_cell_pixels_in_sub] = temp_mask[original_cell_pixels_in_sub]
            separated_mask[slice_obj] = full_subvol

        except MemoryError as e_mem: print(f"MEM ERR L{cell_label}: {e_mem}"); gc.collect(); continue
        except Exception as e_outer: print(f"ERR L{cell_label}: {e_outer}"); traceback.print_exc(); continue
        finally:
            vars_to_del = [
                'cell_mask', 'bbox_slice', 'slice_obj', 'offset', 'cell_mask_sub', 'soma_sub',
                'intensity_sub_local', 'cell_soma_sub_mask', 'seed_props', 'seed_prop_dict',
                'adj', 'cost_array', 'watershed_markers', 'marker_id_reverse_map',
                'distance_transform', 'landscape', 'watershed_result', 'temp_mask', 'full_subvol'
            ]
            for var_name in vars_to_del:
                if var_name in locals() and locals()[var_name] is not None:
                    del locals()[var_name]
            if 'used_new_labels' in locals(): del used_new_labels # it's a set
            if processed_count > 0 and processed_count % 20 == 0: gc.collect() # More frequent GC

    print(f"Finished processing {processed_count} 3D multi-soma cells. Skipped splitting {skipped_count}.")
    gc.collect()

    print("\nRelabeling final 3D mask sequentially...")
    try:
        if not np.issubdtype(separated_mask.dtype, np.integer): separated_mask = separated_mask.astype(np.int32)
        final_unique_vals = np.unique(separated_mask)
        if len(final_unique_vals) <=1 and 0 in final_unique_vals: print("Final mask empty or only background."); return separated_mask
        
        final_sequential_mask, _, inv_map = relabel_sequential(separated_mask)
        final_max_label = len(inv_map)-1 if len(inv_map)>0 else 0 # Max label is N if N objects
        print(f"Relabeled 3D mask contains {final_max_label} objects.")
        return final_sequential_mask.astype(np.int32)
    except Exception as e_relabel_final:
        print(f"Error during final sequential relabeling: {e_relabel_final}."); traceback.print_exc()
        return separated_mask.astype(np.int32) # Return as is, but ensure int32