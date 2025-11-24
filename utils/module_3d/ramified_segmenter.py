import numpy as np # type: ignore
from scipy import ndimage, sparse # type: ignore
from tqdm import tqdm # type: ignore
from shutil import rmtree
import gc
import time # For profiling
from functools import partial
from skimage.measure import regionprops, label as skimage_label # type: ignore
from skimage.segmentation import watershed # type: ignore
from skimage.graph import route_through_array # type: ignore
from skimage.morphology import binary_erosion, binary_dilation, ball # type: ignore
import math
from sklearn.decomposition import PCA # type: ignore
from skimage.feature import peak_local_max # type: ignore
from typing import List, Dict, Optional, Tuple, Union, Any, Set
from skimage.segmentation import relabel_sequential # type: ignore
import traceback # For detailed error logging
from skimage.measure import label, regionprops # type: ignore
from collections import deque
from scipy.ndimage import binary_fill_holes, find_objects # type: ignore
from skimage.morphology import binary_dilation, ball, footprint_rectangle # type: ignore
from scipy.ndimage import binary_fill_holes, find_objects, gaussian_filter, _ni_support, _nd_image # type: ignore
import os
from shutil import rmtree
import sys # Make sure sys is imported
import SimpleITK as sitk
import graph_tool.all as gt
from scipy.sparse import csr_matrix

seed = 42
np.random.seed(seed)         # For NumPy

# Add this import at the top of your file
import SimpleITK as sitk

def _watershed_with_simpleitk(
    landscape: np.ndarray,
    markers: np.ndarray,
    log_prefix: str = ""
) -> np.ndarray:
    """
    Performs the watershed algorithm using SimpleITK, which is generally more
    memory-efficient for large 3D volumes than scikit-image. This version includes
    detailed profiling and data sanitization to ensure compatibility with the
    strict ITK backend.

    Args:
        landscape (np.ndarray): The watershed landscape.
        markers (np.ndarray): The labeled markers for the watershed.
        log_prefix (str): A prefix for logging messages.

    Returns:
        np.ndarray: The resulting labeled segmentation mask.
    """
    flush_print(f"{log_prefix} Executing watershed using SimpleITK...")
    ws_start_time = time.time()
    
    try:
        # --- [STEP 1: PROFILING & SANITIZING] ---
        flush_print(f"{log_prefix}   - Input landscape (numpy): shape={landscape.shape}, dtype={landscape.dtype}")
        flush_print(f"{log_prefix}   - Input markers (numpy):   shape={markers.shape}, dtype={markers.dtype}")

        if not np.all(np.isfinite(landscape)):
            flush_print(f"{log_prefix}   - WARNING: Non-finite values (inf/nan) found in landscape. Sanitizing...")
            max_finite_val = np.max(landscape[np.isfinite(landscape)]) if np.any(np.isfinite(landscape)) else 0
            landscape = np.nan_to_num(landscape, nan=max_finite_val + 1, posinf=max_finite_val + 1, neginf=max_finite_val - 1)
            flush_print(f"{log_prefix}   - Sanitization complete.")
        
        # --- [STEP 2: CONVERT TO SITK IMAGES] ---
        # We use standard dtypes that are widely supported.
        landscape_sitk = sitk.GetImageFromArray(landscape.astype(np.float64))
        markers_sitk = sitk.GetImageFromArray(markers.astype(np.uint32))

        flush_print(f"{log_prefix}   - Landscape (sitk):    size={landscape_sitk.GetSize()}, pixel_type={landscape_sitk.GetPixelIDTypeAsString()}")
        flush_print(f"{log_prefix}   - Markers (sitk):      size={markers_sitk.GetSize()}, pixel_type={markers_sitk.GetPixelIDTypeAsString()}")
        
        flush_print(f"{log_prefix}   - Calling sitk.MorphologicalWatershedFromMarkers...")
        
        # --- [THE DEFINITIVE FIX] ---
        # The correct argument order is (landscapeImage, markerImage).
        result_sitk = sitk.MorphologicalWatershedFromMarkers(
            landscape_sitk,
            markers_sitk,
            markWatershedLine=True,
            fullyConnected=False
        )

        # --- [STEP 3: CONVERT RESULT BACK TO NUMPY] ---
        result_np = sitk.GetArrayFromImage(result_sitk)
        result_np = result_np.astype(markers.dtype, copy=False)

        flush_print(f"{log_prefix} SimpleITK watershed finished. Time: {time.time() - ws_start_time:.2f}s")
        return result_np

    except Exception as e:
        flush_print(f"{log_prefix} CRITICAL: SimpleITK watershed failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback
        if isinstance(markers, np.memmap):
            return np.array(markers)
        return markers.copy()

def flush_print(*args, **kwargs):
    """A wrapper for flush_print() that forces an immediate flush of the output buffer."""
    print(*args, **kwargs)
    sys.stdout.flush()

def _distance_tranform_arg_check(distances_out, indices_out,
                                 return_distances, return_indices):
    """Raise a RuntimeError if the arguments are invalid"""
    error_msgs = []
    if (not return_distances) and (not return_indices):
        error_msgs.append(
            'at least one of return_distances/return_indices must be True')
    if distances_out and not return_distances:
        error_msgs.append(
            'return_distances must be True if distances is supplied'
        )
    if indices_out and not return_indices:
        error_msgs.append('return_indices must be True if indices is supplied')
    if error_msgs:
        raise RuntimeError(', '.join(error_msgs))

def distance_transform_edt(input, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None,
                           output=None):
    """
    Exact Euclidean distance transform.

    This function is a modified version of scipy.ndimage.distance_transform_edt
    with an added 'output' parameter for memory-mapped computation.

    Parameters
    ----------
    input : array_like
        Input data to transform. Can be any type but will be converted
        into binary: 1 wherever input equates to True, 0 elsewhere.
    sampling : float, or sequence of float, optional
        Spacing of elements along each dimension. If a sequence, must be of
        length equal to the input rank; if a single number, this is used for
        all axes. If not specified, a grid spacing of unity is implied.
    return_distances : bool, optional
        Whether to calculate the distance transform.
        Default is True.
    return_indices : bool, optional
        Whether to calculate the feature transform.
        Default is False.
    distances : float64 ndarray, optional
        An output array to store the calculated distance transform, instead of
        returning it.
        `return_distances` must be True.
        It must be the same shape as `input`.
    indices : int32 ndarray, optional
        An output array to store the calculated feature transform, instead of
        returning it.
        `return_indices` must be True.
        Its shape must be ``(input.ndim,) + input.shape``.
    output : numpy.memmap, optional
        A memory-mapped array to use for offloading intermediate,
        RAM-intensive computations. If provided, it must have a shape of
        ``(input.ndim,) + input.shape`` and a dtype of `numpy.float64`.

    Returns
    -------
    distances : float64 ndarray, optional
        The calculated distance transform. Returned only when
        `return_distances` is True and `distances` is not supplied.
        It will have the same shape as the input array.
    indices : int32 ndarray, optional
        The calculated feature transform. It has an input-shaped array for each
        dimension of the input. See example below.
        Returned only when `return_indices` is True and `indices` is not
        supplied.
    """
    ft_inplace = isinstance(indices, np.ndarray)
    dt_inplace = isinstance(distances, np.ndarray)

    # This is a private function in SciPy for argument validation.
    _distance_tranform_arg_check(
        dt_inplace, ft_inplace, return_distances, return_indices
    )

    input = np.atleast_1d(np.where(input, 1, 0).astype(np.int8))
    if sampling is not None:
        sampling = _ni_support._normalize_sequence(sampling, input.ndim)
        sampling = np.asarray(sampling, dtype=np.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()

    if ft_inplace:
        ft = indices
        if ft.shape != (input.ndim,) + input.shape:
            raise RuntimeError('indices array has wrong shape')
        if ft.dtype.type != np.int32:
            raise RuntimeError('indices array must be int32')
    else:
        ft = np.zeros((input.ndim,) + input.shape, dtype=np.int32)

    _nd_image.euclidean_feature_transform(input, sampling, ft)

    if return_distances:
        if output is not None:
            # Validate the provided output memmap array
            expected_shape = (input.ndim,) + input.shape
            if output.shape != expected_shape:
                raise ValueError(f"The 'output' memmap must have shape {expected_shape}, but has {output.shape}")
            if output.dtype != np.float64:
                raise TypeError(f"The 'output' memmap must have dtype {np.float64}, but has {output.dtype}")

            dt = output
            # Use the memmap as the output buffer to perform the calculation, saving RAM
            np.subtract(ft, np.indices(input.shape, dtype=ft.dtype), out=dt)
        else:
            # Original behavior: create a new array in memory
            dt = ft - np.indices(input.shape, dtype=ft.dtype)
            dt = dt.astype(np.float64)

        if sampling is not None:
            for ii in range(len(sampling)):
                dt[ii, ...] *= sampling[ii]
        np.multiply(dt, dt, dt)

        if dt_inplace:
            # This reduction creates a smaller temporary array in memory
            reduced_dt = np.add.reduce(dt, axis=0)
            if distances.shape != reduced_dt.shape:
                raise RuntimeError('distances array has wrong shape')
            if distances.dtype.type != np.float64:
                raise RuntimeError('distances array must be float64')
            np.sqrt(reduced_dt, distances)
        else:
            dt = np.add.reduce(dt, axis=0)
            dt = np.sqrt(dt)

    # Construct and return the result
    result = []
    if return_distances and not dt_inplace:
        result.append(dt)
    if return_indices and not ft_inplace:
        result.append(ft)

    if len(result) == 2:
        return tuple(result)
    elif len(result) == 1:
        return result[0]
    else:
        return None

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
    parent_dt: np.ndarray, # OPTIMIZED: DT of the core this fragment came from
    spacing: Tuple[float, float, float],
    min_seed_fragment_volume: int,
    min_accepted_core_volume: float,
    max_accepted_core_volume: float,
    min_accepted_thickness_um: float, # From fragment's own DT, in um
    max_accepted_thickness_um: float, # From fragment's own DT, in um
    max_allowed_core_aspect_ratio: float
) -> Tuple[str, Optional[str], Optional[np.ndarray], Optional[float], float]:
    """
    Filters a single 3D candidate fragment and returns a detailed status.
    Returns: (status, reason, mask_copy, volume, duration)
    - status: 'valid', 'fallback', or 'discard'.
    - reason: The specific reason for fallback/discard (e.g., 'thickness', 'aspect').
    """
    t_start = time.time()
    
    try:
        fragment_volume = np.sum(fragment_mask)
        if fragment_volume < min_seed_fragment_volume:
            return 'discard', 'too_small', None, None, time.time() - t_start

        # --- Thickness Check (Already Spacing-Aware) ---
        max_dist_in_fragment_um = np.max(parent_dt[fragment_mask])
        passes_thickness = (min_accepted_thickness_um <= max_dist_in_fragment_um <= max_accepted_thickness_um)
        
        # --- Volume Check ---
        passes_volume_range = (min_accepted_core_volume <= fragment_volume <= max_accepted_core_volume)

        # --- Aspect Ratio Check (NOW SPACING-AWARE) ---
        passes_aspect = True
        if fragment_volume > 3: # Need more than 3 points for PCA
            try:
                # Get voxel coordinates and convert to physical coordinates
                coords_vox = np.argwhere(fragment_mask)
                coords_phys = coords_vox * np.array(spacing)

                # Use PCA on physical coordinates to get axis lengths
                pca = PCA(n_components=3)
                pca.fit(coords_phys)
                eigenvalues = pca.explained_variance_ # These are variances along principal axes
                eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
                
                if eigenvalues_sorted[2] > 1e-12: # Avoid division by zero
                    aspect_ratio = math.sqrt(eigenvalues_sorted[0]) / math.sqrt(eigenvalues_sorted[2])
                    if aspect_ratio > max_allowed_core_aspect_ratio:
                        passes_aspect = False
            except Exception:
                passes_aspect = True
        
        fragment_mask_copy = fragment_mask.copy()
        duration = time.time() - t_start

        if passes_thickness and passes_volume_range and passes_aspect:
            return 'valid', None, fragment_mask_copy, fragment_volume, duration
        else:
            # Determine the first reason for failure
            reason = 'unknown'
            if not passes_thickness: reason = 'thickness'
            elif not passes_volume_range: reason = 'volume'
            elif not passes_aspect: reason = 'aspect'
            return 'fallback', reason, fragment_mask_copy, fragment_volume, duration
            
    except Exception as e_filt:
        flush_print(f"Warn: Unexpected error during 3D fragment filtering: {e_filt}")
        return 'discard', 'error', None, None, time.time() - t_start

import psutil # Add this import

def log_memory_usage(label=""):
    """Logs the current RAM usage of the process."""
    process = psutil.Process(os.getpid())
    # RSS: Resident Set Size - the non-swapped physical memory a process has used.
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)
    flush_print(f"    [MEM_PROFILE] {label}: {mem_gb:.3f} GB")

# (Ensure flush_print, time, os, gc, rmtree, psutil, and other necessary imports are at the top of your file)

def extract_soma_masks(
    segmentation_mask: np.ndarray, # 3D mask
    intensity_image: np.ndarray, # 3D intensity image
    spacing: Optional[Tuple[float, float, float]],
    smallest_quantile: int = 25,
    min_fragment_size: int = 30, # Voxels
    core_volume_target_factor_lower: float = 0.1,
    core_volume_target_factor_upper: float = 10,
    erosion_iterations: int = 0,
    ratios_to_process = [0.3, 0.4, 0.5, 0.6],
    intensity_percentiles_to_process: List[int] = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
    min_physical_peak_separation: float = 7.0, # um
    max_allowed_core_aspect_ratio: float = 10.0,
    ref_vol_percentile_lower: int = 30,
    ref_vol_percentile_upper: int = 70,
    ref_thickness_percentile_lower: int = 1,
    absolute_min_thickness_um: float = 1.5,
    absolute_max_thickness_um: float = 10.0,
    # --- Parameters for memory optimization ---
    memmap_dir: Optional[str] = "ramiseg_temp_memmap",
    memmap_voxel_threshold: int = 25_000_000,
    memmap_final_mask: bool = True
) -> np.ndarray:
    """
    Generates candidate seed mask with optimizations for baseline and peak RAM usage.
    """
    flush_print("--- Starting 3D Soma Extraction (Optimized for Baseline RAM) ---")
    
    use_memmap_feature = memmap_dir is not None
    if use_memmap_feature:
        os.makedirs(memmap_dir, exist_ok=True)
        flush_print(f"INFO: Using temporary directory for large arrays: {memmap_dir}")

    if min_fragment_size is None or min_fragment_size < 1: min_seed_fragment_volume = 30
    else: min_seed_fragment_volume = min_fragment_size
    MAX_CORE_VOXELS_FOR_WS = 500_000
    if spacing is None: spacing = (1.0, 1.0, 1.0)
    else:
        try: spacing = tuple(float(s) for s in spacing); assert len(spacing) == 3
        except: spacing = (1.0, 1.0, 1.0)
    unique_labels_all, counts = np.unique(segmentation_mask[segmentation_mask > 0], return_counts=True)
    if len(unique_labels_all) == 0: return np.zeros_like(segmentation_mask, dtype=np.int32)
    initial_volumes = dict(zip(unique_labels_all, counts))
    label_to_slice: Dict[int, Tuple[slice, ...]] = {}
    valid_unique_labels_list: List[int] = []
    object_slices = ndimage.find_objects(segmentation_mask)
    if object_slices:
        for label in unique_labels_all:
            idx = label - 1
            if 0 <= idx < len(object_slices) and object_slices[idx]:
                s = object_slices[idx]
                if len(s) == 3 and all(si.start < si.stop for si in s):
                    label_to_slice[label] = s; valid_unique_labels_list.append(label)
    if not valid_unique_labels_list: return np.zeros_like(segmentation_mask, dtype=np.int32)
    unique_labels = np.array(valid_unique_labels_list)
    volumes = {label: initial_volumes[label] for label in unique_labels}; all_volumes_list = list(volumes.values())
    if not all_volumes_list: return np.zeros_like(segmentation_mask, dtype=np.int32)
    smallest_thresh_volume = np.percentile(all_volumes_list, smallest_quantile) if len(all_volumes_list) > 1 else (all_volumes_list[0] + 1 if all_volumes_list else 1)
    smallest_object_labels_set = {label for label, vol in volumes.items() if vol <= smallest_thresh_volume}
    target_soma_volume = np.median([v for l,v in volumes.items() if l in smallest_object_labels_set] or all_volumes_list)
    min_accepted_core_volume = max(min_seed_fragment_volume, target_soma_volume * core_volume_target_factor_lower)
    max_accepted_core_volume = target_soma_volume * core_volume_target_factor_upper
    vol_thresh_lower, vol_thresh_upper = np.percentile(all_volumes_list, ref_vol_percentile_lower), np.percentile(all_volumes_list, ref_vol_percentile_upper)
    reference_object_labels = [l for l in unique_labels if vol_thresh_lower < volumes[l] <= vol_thresh_upper]
    if len(reference_object_labels) < 5: reference_object_labels = list(unique_labels)
    max_thicknesses_um = []
    for label_ref in tqdm(reference_object_labels, desc="Calc Ref Thickness", disable=len(reference_object_labels) < 10):
        bbox_slice = label_to_slice.get(label_ref);
        if bbox_slice:
            mask_sub = (segmentation_mask[bbox_slice] == label_ref)
            if np.any(mask_sub):
                dt = ndimage.distance_transform_edt(mask_sub, sampling=spacing)
                max_thicknesses_um.append(np.max(dt))
    min_accepted_thickness_um = max(absolute_min_thickness_um, np.percentile(max_thicknesses_um, ref_thickness_percentile_lower)) if max_thicknesses_um else absolute_min_thickness_um
    min_accepted_thickness_um = min(min_accepted_thickness_um, absolute_max_thickness_um - 1e-6)
    min_peak_sep_pixels = get_min_distance_pixels(spacing, min_physical_peak_separation)
    gc.collect()

    final_seed_mask_path = None
    final_seed_mask = None
    
    try:
        if use_memmap_feature and memmap_final_mask:
            flush_print(f"Creating final_seed_mask as a memory-mapped file to reduce baseline RAM.")
            final_seed_mask_path = os.path.join(memmap_dir, 'final_seed_mask.mmp')
            final_seed_mask = np.memmap(final_seed_mask_path, dtype=np.int32, mode='w+', shape=segmentation_mask.shape)
        else:
            flush_print(f"Creating final_seed_mask as an in-memory NumPy array.")
            final_seed_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
            
        next_final_label = 1
        added_small_labels: Set[int] = set()
        small_cell_labels = [l for l in unique_labels if l in smallest_object_labels_set]
        flush_print(f"\nProcessing {len(small_cell_labels)} small objects (3D)...")
        for label in tqdm(small_cell_labels, desc="Small Cells", disable=len(small_cell_labels) < 10):
            if volumes.get(label, 0) >= min_seed_fragment_volume:
                coords = np.argwhere(segmentation_mask == label)
                if coords.size > 0:
                    final_seed_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = next_final_label
                    next_final_label += 1; added_small_labels.add(label)
        flush_print(f"Added {len(added_small_labels)} initial seeds from small cells."); gc.collect()

        large_cell_labels = [l for l in unique_labels if l not in added_small_labels]
        flush_print(f"Generating and placing candidates from {len(large_cell_labels)} large objects (3D)...")
        struct_el_erosion = ndimage.generate_binary_structure(3, 3) if erosion_iterations > 0 else None

        for label in tqdm(large_cell_labels, desc="Large Cell Candidates"):
            flush_print(f"\n--- [START] Processing Large Cell L{label} ---")
            bbox_slice = label_to_slice.get(label)
            if not bbox_slice:
                flush_print(f"L{label}: No bbox slice found. Skipping.")
                continue
            
            valid_candidate_paths: List[Dict[str, Any]] = []
            fallback_candidate_paths: List[Dict[str, Any]] = []
            
            seg_sub, mask_sub, int_sub, dist_transform_obj = None, None, None, None
            dt_intermediate_path = None
            int_sub_path = None

            try:
                pad = 1
                slice_obj_tuple = (slice(max(0, s.start-pad), min(sh, s.stop+pad)) for s, sh in zip(bbox_slice, segmentation_mask.shape))
                slice_obj = tuple(slice_obj_tuple)
                offset = (slice_obj[0].start, slice_obj[1].start, slice_obj[2].start)
                local_shape = tuple(s.stop - s.start for s in slice_obj)
                num_voxels = np.prod(local_shape)
                flush_print(f"L{label}: Local bbox shape is {local_shape} ({num_voxels:,} voxels)")
                
                seg_sub = segmentation_mask[slice_obj]
                mask_sub = (seg_sub == label)
                if not np.any(mask_sub): continue
                
                def process_core_mask(core_mask_for_labeling: np.ndarray, parent_dt: np.ndarray, time_log: Dict[str, float]) -> Tuple[int, int, Dict[str, int]]:
                    flush_print(f"  (process_core_mask): Starting.")
                    rejection_stats = {'too_small': 0, 'thickness': 0, 'volume': 0, 'aspect': 0, 'error': 0}
                    labeled_cores, num_cores = ndimage.label(core_mask_for_labeling)
                    if num_cores == 0: return 0, 0, rejection_stats
                    core_volumes = ndimage.sum_labels(core_mask_for_labeling, labeled_cores, range(1, num_cores + 1))
                    
                    kept_fragments = 0
                    for core_idx in np.where(core_volumes >= min_seed_fragment_volume)[0]:
                        core_lbl = core_idx + 1
                        core_component_mask = (labeled_cores == core_lbl)
                        core_volume = core_volumes[core_idx]
                        
                        t_ws_start = time.time()
                        ws_core_full_size = np.zeros_like(parent_dt, dtype=np.int32)
                        if core_volume > MAX_CORE_VOXELS_FOR_WS:
                            ws_core_full_size[core_component_mask] = 1
                        else:
                            core_slices = find_objects(core_component_mask)
                            if not core_slices: continue
                            core_bbox = core_slices[0]
                            mask_cropped = core_component_mask[core_bbox]
                            dt_cropped = parent_dt[core_bbox]
                            peaks = peak_local_max(dt_cropped, min_distance=min_peak_sep_pixels, labels=mask_cropped, exclude_border=False)
                            if peaks.shape[0] > 1:
                                markers_cropped = np.zeros(dt_cropped.shape, dtype=np.int32)
                                markers_cropped[tuple(peaks.T)] = np.arange(1, peaks.shape[0] + 1)
                                ws_core_cropped = watershed(-dt_cropped, markers_cropped, mask=mask_cropped, watershed_line=False)
                                ws_core_full_size[core_bbox][mask_cropped] = ws_core_cropped[mask_cropped]
                            else:
                                ws_core_full_size[core_component_mask] = 1
                        time_log['mini_ws'] += time.time() - t_ws_start
                        
                        unique_labels_ws = np.unique(ws_core_full_size)
                        unique_labels_ws = unique_labels_ws[unique_labels_ws > 0]

                        flush_print(f"    (process_core_mask): Filtering {len(unique_labels_ws)} fragments from core {core_lbl} one-by-one...")
                        for fragment_label in unique_labels_ws:
                            f_mask = (ws_core_full_size == fragment_label)
                            status, reason, m_copy, vol, dur = _filter_candidate_fragment_3d(f_mask, parent_dt, spacing, min_seed_fragment_volume, min_accepted_core_volume, max_accepted_core_volume, min_accepted_thickness_um, absolute_max_thickness_um, max_allowed_core_aspect_ratio)
                            time_log['filtering'] += dur
                            
                            if status in ['valid', 'fallback']:
                                coords = np.argwhere(m_copy)
                                if coords.size == 0:
                                    if m_copy is not None: del m_copy
                                    continue
                                timestamp = int(time.time() * 1_000_000)
                                cand_path = os.path.join(memmap_dir, f"cell_{label}_cand_{timestamp}.mmp")
                                cand_memmap = np.memmap(cand_path, dtype=coords.dtype, mode='w+', shape=coords.shape)
                                cand_memmap[:] = coords[:]
                                cand_memmap.flush()
                                del cand_memmap
                                candidate_info = {'path': cand_path, 'volume': vol, 'offset': offset, 'dtype': coords.dtype, 'shape': coords.shape}
                                if status == 'valid':
                                    valid_candidate_paths.append(candidate_info)
                                    kept_fragments += 1
                                else:
                                    fallback_candidate_paths.append(candidate_info)
                                    if reason: rejection_stats[reason] += 1
                            elif status == 'discard' and reason:
                                rejection_stats[reason] += 1
                            if m_copy is not None: del m_copy
                    return num_cores, kept_fragments, rejection_stats
                
                use_memmap_for_this_cell = use_memmap_feature and (num_voxels > memmap_voxel_threshold)
                if use_memmap_for_this_cell:
                    dt_intermediate_path = os.path.join(memmap_dir, f"cell_{label}_dt_intermediate.mmp")
                    intermediate_shape = (mask_sub.ndim,) + mask_sub.shape
                    dt_intermediate_memmap = np.memmap(dt_intermediate_path, dtype=np.float64, mode='w+', shape=intermediate_shape)
                    distance_transform_edt(mask_sub, sampling=spacing, output=dt_intermediate_memmap)
                    final_dt_squared = np.add.reduce(dt_intermediate_memmap, axis=0)
                    dist_transform_obj = np.sqrt(final_dt_squared)
                    del dt_intermediate_memmap
                else:
                    dist_transform_obj = ndimage.distance_transform_edt(mask_sub, sampling=spacing)
                
                max_dist = np.max(dist_transform_obj)
                if max_dist > 1e-9:
                    for ratio in ratios_to_process:
                        flush_print(f"  L{label}: Processing DT ratio {ratio:.1f}...")
                        t_iter_start = time.time(); time_log = {'labeling': 0, 'summing': 0, 'mini_ws': 0, 'filtering': 0}
                        initial_core = (dist_transform_obj >= max_dist * ratio) & mask_sub
                        if not np.any(initial_core): continue
                        eroded_core = ndimage.binary_erosion(initial_core, footprint=struct_el_erosion, iterations=erosion_iterations) if erosion_iterations > 0 else initial_core
                        if np.any(eroded_core):
                            num_found, num_kept, stats = process_core_mask(eroded_core, dist_transform_obj, time_log)
                            total_time = time.time() - t_iter_start
                            flush_print(f"    L{label} DT R{ratio:.1f}: Found {num_found} cores, kept {num_kept} frags. Total time: {total_time:.2f}s")

                flush_print(f"L{label}: Starting Intensity Path...")
                if use_memmap_for_this_cell:
                    flush_print(f"L{label}: Cell is large. Creating intensity crop (int_sub) as a memmap file.")
                    int_sub_path = os.path.join(memmap_dir, f"cell_{label}_int_sub.mmp")
                    int_sub = np.memmap(int_sub_path, dtype=intensity_image.dtype, mode='w+', shape=local_shape)
                    int_sub[:] = intensity_image[slice_obj]
                    int_sub.flush()
                else:
                    flush_print(f"L{label}: Cell is small. Creating intensity crop (int_sub) in RAM.")
                    int_sub = intensity_image[slice_obj]
                flush_print(f"L{label}: int_sub created. shape={int_sub.shape}, dtype={int_sub.dtype}, is_memmap={isinstance(int_sub, np.memmap)}")
                
                ints_obj_vals = int_sub[mask_sub]
                if ints_obj_vals.size > 0:
                    candidates_have_been_reset = False
                    dt_int_path_reusable = os.path.join(memmap_dir, f"cell_{label}_dt_intensity_REUSABLE.mmp")
                    
                    for perc in intensity_percentiles_to_process:
                        log_memory_usage(f"L{label} P{perc}: Start of loop")
                        flush_print(f"  L{label}: Processing Intensity percentile {perc}...")
                        t_iter_start = time.time(); time_log = {'labeling': 0, 'summing': 0, 'mini_ws': 0, 'filtering': 0}
                        
                        thresh = np.percentile(ints_obj_vals, perc)
                        initial_core = (int_sub >= thresh) & mask_sub
                        log_memory_usage(f"L{label} P{perc}: After 'initial_core' creation")
                        
                        if not np.any(initial_core): continue
                        
                        eroded_core = ndimage.binary_erosion(initial_core, footprint=struct_el_erosion, iterations=erosion_iterations) if erosion_iterations > 0 else initial_core
                        log_memory_usage(f"L{label} P{perc}: After 'eroded_core' creation")
                        del initial_core
                        
                        if np.any(eroded_core):
                            if not candidates_have_been_reset:
                                flush_print(f"    L{label} P{perc}: Found valid intensity core. Clearing previous DT-based candidate files.")
                                for cand_info in valid_candidate_paths:
                                    if os.path.exists(cand_info['path']): os.remove(cand_info['path'])
                                for cand_info in fallback_candidate_paths:
                                    if os.path.exists(cand_info['path']): os.remove(cand_info['path'])
                                valid_candidate_paths.clear()
                                fallback_candidate_paths.clear()
                                candidates_have_been_reset = True
                            
                            temp_dt_for_intensity_path = np.zeros_like(eroded_core, dtype=np.float64)
                            log_memory_usage(f"L{label} P{perc}: After creating 'temp_dt_for_intensity_path'")
                            
                            core_slices = find_objects(eroded_core)
                            if not core_slices:
                                flush_print(f"    L{label} P{perc}: Core became empty after erosion, skipping.")
                                del eroded_core
                                continue
                            
                            core_bbox = core_slices[0]
                            cropped_core = eroded_core[core_bbox]
                            log_memory_usage(f"L{label} P{perc}: After 'cropped_core' creation")
                            
                            cropped_core_voxels = cropped_core.size
                            use_memmap_for_cropped_core = use_memmap_feature and (cropped_core_voxels > memmap_voxel_threshold)

                            cropped_dt = None
                            if use_memmap_for_cropped_core:
                                flush_print(f"    L{label} P{perc}: Cropped intensity core is large ({cropped_core_voxels:,} voxels). Using REUSABLE MEMMAP for DT.")
                                int_intermediate_shape = (cropped_core.ndim,) + cropped_core.shape
                                dt_int_memmap = np.memmap(dt_int_path_reusable, dtype=np.float64, mode='w+', shape=int_intermediate_shape)
                                log_memory_usage(f"L{label} P{perc}: After creating reusable memmap")
                                distance_transform_edt(cropped_core, sampling=spacing, output=dt_int_memmap)
                                log_memory_usage(f"L{label} P{perc}: After running custom EDT")
                                final_dt_sq = np.add.reduce(dt_int_memmap, axis=0)
                                cropped_dt = np.sqrt(final_dt_sq)
                                log_memory_usage(f"L{label} P{perc}: After calculating final cropped_dt from memmap")
                                del dt_int_memmap, final_dt_sq
                            else:
                                flush_print(f"    L{label} P{perc}: Cropped intensity core is small ({cropped_core_voxels:,} voxels). Using RAM for DT.")
                                cropped_dt = ndimage.distance_transform_edt(cropped_core, sampling=spacing)
                                log_memory_usage(f"L{label} P{perc}: After calculating cropped_dt in RAM")

                            temp_dt_for_intensity_path[core_bbox] = cropped_dt
                            log_memory_usage(f"L{label} P{perc}: After placing cropped_dt into full-size array")
                            del cropped_dt, cropped_core # UNLIKE BEFORE, WE KEEP eroded_core for process_core_mask
                            
                            num_found, num_kept, stats = process_core_mask(eroded_core, temp_dt_for_intensity_path, time_log)
                            del temp_dt_for_intensity_path, eroded_core # Cleanup after use
                            log_memory_usage(f"L{label} P{perc}: After process_core_mask and cleanup")
                            
                            total_time = time.time() - t_iter_start
                            flush_print(f"    L{label} INT P{int(perc):02d}: Found {num_found} cores, kept {num_kept} frags. Total time: {total_time:.2f}s")
                    
                    if os.path.exists(dt_int_path_reusable):
                        try: os.remove(dt_int_path_reusable)
                        except OSError as e: flush_print(f"L{label}: WARNING - Could not remove reusable intensity memmap: {e}")
                
                flush_print(f"L{label}: Starting placement logic...")
                candidates_to_process = []
                if valid_candidate_paths: candidates_to_process.extend(valid_candidate_paths)
                elif fallback_candidate_paths: candidates_to_process.append(max(fallback_candidate_paths, key=lambda x: x['volume']))
                if candidates_to_process:
                    candidates_to_process.sort(key=lambda x: x['volume'])
                    flush_print(f"L{label}: Placing {len(candidates_to_process)} candidates from disk...")
                    for i, cand_info in enumerate(candidates_to_process):
                        coords_sub = np.memmap(cand_info['path'], dtype=cand_info['dtype'], mode='r', shape=cand_info['shape'])
                        offset_cand = cand_info['offset']
                        if coords_sub.size > 0:
                            gz, gy, gx = coords_sub[:, 0] + offset_cand[0], coords_sub[:, 1] + offset_cand[1], coords_sub[:, 2] + offset_cand[2]
                            vg = (gz, gy, gx)
                            if np.all(final_seed_mask[vg] == 0):
                                final_seed_mask[vg] = next_final_label
                                next_final_label += 1
                        del coords_sub
                        try:
                            os.remove(cand_info['path'])
                        except OSError as e:
                            flush_print(f"  L{label}: WARNING - could not remove candidate file {cand_info['path']}: {e}")
            except MemoryError:
                flush_print(f"CRITICAL: MemoryError processing L{label}. Skipping this cell."); gc.collect()
            except Exception as e:
                flush_print(f"Warn: Uncaught Error L{label}: {e}"); traceback.print_exc()
            finally:
                flush_print(f"L{label}: Entering 'finally' block for cleanup...")
                if dt_intermediate_path and os.path.exists(dt_intermediate_path):
                    try: os.remove(dt_intermediate_path)
                    except OSError as e: flush_print(f"L{label}: WARNING - Could not remove intermediate memmap file: {e}")
                if int_sub_path and os.path.exists(int_sub_path):
                    try:
                        os.remove(int_sub_path)
                        flush_print(f"L{label}: Successfully removed int_sub memmap file.")
                    except OSError as e:
                        flush_print(f"L{label}: WARNING - Could not remove int_sub memmap file: {e}")
                del valid_candidate_paths, fallback_candidate_paths
                if seg_sub is not None: del seg_sub
                if mask_sub is not None: del mask_sub
                if int_sub is not None: del int_sub
                if dist_transform_obj is not None: del dist_transform_obj
                gc.collect()
                flush_print(f"--- [END] Processing Large Cell L{label} ---")

        if isinstance(final_seed_mask, np.memmap):
            flush_print("Loading final memory-mapped seed mask into RAM before returning...")
            in_memory_final_mask = np.array(final_seed_mask)
            return in_memory_final_mask
        else:
            return final_seed_mask
    finally:
        flush_print("--- Entering final cleanup block ---")
        if 'final_seed_mask' in locals() and isinstance(final_seed_mask, np.memmap):
            del final_seed_mask
            gc.collect()
        if final_seed_mask_path and os.path.exists(final_seed_mask_path):
            try:
                os.remove(final_seed_mask_path)
                flush_print(f"Successfully removed final_seed_mask memmap file: {final_seed_mask_path}")
            except OSError as e:
                flush_print(f"WARNING: Could not remove final_seed_mask memmap file: {e}")
        if use_memmap_feature and memmap_dir and os.path.exists(memmap_dir):
            flush_print(f"Final cleanup of memmap directory: {memmap_dir}")
            try:
                rmtree(memmap_dir)
            except OSError as e:
                flush_print(f"WARNING: Could not remove memmap directory during final cleanup: {e}")

# --- Helper: Analyze local intensity difference ---
def _analyze_local_intensity_difference_optimized(
    interface_mask: np.ndarray, # The pre-computed interface between segments
    region1_mask: np.ndarray, region2_mask: np.ndarray,
    intensity_vol_local: np.ndarray, local_analysis_radius: int,
    min_local_intensity_difference_threshold: float, log_prefix: str = "    [LID]"
) -> bool:
    """
    Optimized version of LID analysis that works by dilating only the small
    interface mask, avoiding large intermediate array allocations.
    """
    t_start_lid = time.time()
    
    # --- The key optimization is here ---
    # Instead of dilating the entire region masks (which are large), we only
    # dilate the much smaller interface_mask. This creates a narrow "analysis zone"
    # around the boundary, which is all we need.
    footprint_elem = ball(local_analysis_radius) if local_analysis_radius > 1 else footprint_rectangle((3,3,3))
    analysis_zone = binary_dilation(interface_mask, footprint=footprint_elem)

    # The local adjacent regions are simply the parts of this analysis zone
    # that fall within each original segment. This is extremely fast and memory-efficient.
    la_r1 = analysis_zone & region1_mask
    la_r2 = analysis_zone & region2_mask

    sa1, sa2 = np.sum(la_r1), np.sum(la_r2)
    flush_print(f"{log_prefix} Adj. region sizes: LA_R1={sa1}, LA_R2={sa2}")
    
    MIN_ADJ_SIZE = 20
    if sa1 < MIN_ADJ_SIZE or sa2 < MIN_ADJ_SIZE:
        flush_print(f"{log_prefix} Small adj. regions. LID returns False. Time:{time.time()-t_start_lid:.3f}s")
        # In this case, the regions are distinct enough because the interface is thin/sparse
        return True # Returning True means "don't merge"

    i1, i2 = intensity_vol_local[la_r1], intensity_vol_local[la_r2]
    m1, m2 = (np.mean(i1) if i1.size > 0 else 0), (np.mean(i2) if i2.size > 0 else 0)
    flush_print(f"{log_prefix} Adj. means: M1={m1:.2f}, M2={m2:.2f}")
    
    ref_i = max(m1, m2)
    if ref_i < 1e-6: 
        flush_print(f"{log_prefix} Ref intensity near zero. LID True. Time:{time.time()-t_start_lid:.3f}s"); 
        return True # Distinct enough
        
    rel_d = abs(m1 - m2) / ref_i
    is_distinct_enough = rel_d >= min_local_intensity_difference_threshold
    flush_print(f"{log_prefix} Rel.Diff={rel_d:.3f} (Thresh {min_local_intensity_difference_threshold:.3f}). Distinct={is_distinct_enough}. Time:{time.time()-t_start_lid:.3f}s")
    return is_distinct_enough

def _calculate_interface_metrics(
    mask_A_local: np.ndarray, # Boolean mask of segment A in its local crop
    mask_B_local: np.ndarray, # Boolean mask of segment B in its local crop
    parent_mask_local: np.ndarray, # Boolean mask of the original cell these segments belong to
    intensity_local: np.ndarray, # Intensity image for the crop
    avg_soma_intensity_for_interface: float, # Avg intensity of somas seeding A & B
    spacing_tuple: Tuple[float, float, float],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float, # Pass this for comparison printout
    log_prefix: str = "      [IfaceMetrics]"
) -> Dict[str, Any]:
    """
    Calculates metrics for the interface between two adjacent segments A and B.
    Now includes mean_interface_intensity and ratio checks.
    """
    metrics = {
        'interface_voxel_count': 0,
        'mean_interface_intensity': 0.0,
        'interface_intensity_ratio': float('inf'), # High if no interface or low soma intensity
        'ratio_threshold_passed': False, # Did it pass the min_path_intensity_ratio check? (i.e., dark enough)
        'lid_passed_separation': True, # Assume distinct unless LID fails (i.e., True if different enough)
        'should_merge_decision': False # Final decision based on heuristics
    }
    if not np.any(mask_A_local) or not np.any(mask_B_local):
        flush_print(f"{log_prefix} One or both masks empty, cannot calculate metrics.")
        return metrics

    footprint_dilation = footprint_rectangle((3,3,3))

    dilated_A = binary_dilation(mask_A_local, footprint=footprint_dilation)
    interface_A_B = dilated_A & mask_B_local 

    dilated_B = binary_dilation(mask_B_local, footprint=footprint_dilation)
    interface_B_A = dilated_B & mask_A_local 

    combined_interface_mask = (interface_A_B | interface_B_A) & parent_mask_local
    metrics['interface_voxel_count'] = np.sum(combined_interface_mask)
    flush_print(f"{log_prefix} Interface Voxel Count: {metrics['interface_voxel_count']}")

    if metrics['interface_voxel_count'] > 0:
        interface_intensity_values = intensity_local[combined_interface_mask]
        if interface_intensity_values.size > 0:
            metrics['mean_interface_intensity'] = np.mean(interface_intensity_values)
        
        if avg_soma_intensity_for_interface > 1e-6:
            metrics['interface_intensity_ratio'] = metrics['mean_interface_intensity'] / avg_soma_intensity_for_interface
        
        # Check against min_path_intensity_ratio heuristic
        # A "good" separation has a LOW ratio (dark interface compared to somas)
        metrics['ratio_threshold_passed'] = metrics['interface_intensity_ratio'] < min_path_intensity_ratio_heuristic
        
        flush_print(f"{log_prefix} Mean Interface Intensity: {metrics['mean_interface_intensity']:.2f}")
        flush_print(f"{log_prefix} Avg Soma Intensity Ref: {avg_soma_intensity_for_interface:.2f}")
        flush_print(f"{log_prefix} Interface/Soma Intensity Ratio: {metrics['interface_intensity_ratio']:.3f} (Threshold for merge if >= {min_path_intensity_ratio_heuristic:.3f})")
        if metrics['ratio_threshold_passed']:
            flush_print(f"{log_prefix} --> Path Ratio Check: PASSED (interface is dark enough relative to somas)")
        else:
            flush_print(f"{log_prefix} --> Path Ratio Check: FAILED (interface is too bright, suggests merge)")


        metrics['lid_passed_separation'] = _analyze_local_intensity_difference_optimized(
            combined_interface_mask, # Pass the pre-computed interface mask
            mask_A_local, mask_B_local, intensity_local,
            local_analysis_radius, min_local_intensity_difference,
            log_prefix=log_prefix + " LID"
        )
        if metrics['lid_passed_separation']:
            flush_print(f"{log_prefix} --> LID Check: PASSED (segments are locally distinct)")
        else:
            flush_print(f"{log_prefix} --> LID Check: FAILED (segments are too similar locally, suggests merge)")
        
        # Decision to merge:
        # Merge if EITHER the interface is too bright (FAILS ratio_threshold_passed)
        # OR if the local regions are too similar (FAILS lid_passed_separation)
        if not metrics['ratio_threshold_passed'] or not metrics['lid_passed_separation']:
            metrics['should_merge_decision'] = True
            flush_print(f"{log_prefix} ===> Merge Decision: YES (RatioFail: {not metrics['ratio_threshold_passed']}, LIDFail: {not metrics['lid_passed_separation']})")
        else:
            metrics['should_merge_decision'] = False
            flush_print(f"{log_prefix} ===> Merge Decision: NO (Separation seems valid by both heuristics)")

    else: # No direct interface voxels found
        flush_print(f"{log_prefix} No interface voxels found between segments. Defaulting to NO MERGE.")
        metrics['ratio_threshold_passed'] = True # No interface to be "too bright"
        metrics['lid_passed_separation'] = True  # No interface for LID to fail on
        metrics['should_merge_decision'] = False

    return metrics

def _build_adjacency_graph_for_cell(
    current_cell_segments_mask_local: np.ndarray, 
    original_cell_mask_local: np.ndarray,
    soma_mask_local: np.ndarray, # Original full soma mask, in local crop
    soma_props_for_cell: Dict[int, Dict[str, Any]], # Pre-calculated props for ORIGINAL somas in this cell's crop
    intensity_local: np.ndarray, 
    spacing_tuple: Tuple[float, float, float],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float, # Pass this for comparison printout in metrics
    log_prefix: str = "    [GraphBuild]"
) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]]]:
    flush_print(f"{log_prefix} Building graph. SegMaskShape:{current_cell_segments_mask_local.shape}")
    nodes: Dict[int, Dict[str, Any]] = {}
    edges: Dict[Tuple[int, int], Dict[str, Any]] = {}
    seg_lbls = np.unique(current_cell_segments_mask_local[current_cell_segments_mask_local > 0])
    flush_print(f"{log_prefix} Found {len(seg_lbls)} P1 segments: {seg_lbls}")

    if len(seg_lbls) <= 1:
        # ... (single node creation logic - same as before) ...
        if len(seg_lbls) == 1:
            lbl = seg_lbls[0]; m_node = (current_cell_segments_mask_local == lbl)
            s_in_node_vals = soma_mask_local[m_node] # Values from soma_mask_local
            s_in_node_unique = np.unique(s_in_node_vals[s_in_node_vals > 0]) # Unique positive labels
            s_in_node = sorted([s_lbl for s_lbl in s_in_node_unique if s_lbl in soma_props_for_cell]) # Filter by those in props

            obj_sl = find_objects(m_node.astype(np.int32))
            nodes[lbl] = {'mask_bbox_local': obj_sl[0] if obj_sl and obj_sl[0] else None, 
                          'orig_somas': s_in_node, 'volume': np.sum(m_node)}
        flush_print(f"{log_prefix} <=1 segment. Graph: {len(nodes)} nodes, {len(edges)} edges.")
        return nodes, edges
    
    footprint_d_graph = footprint_rectangle((3,3,3)) # Renamed

    for lbl_node in seg_lbls: # Renamed lbl to lbl_node
        m_node_build = (current_cell_segments_mask_local == lbl_node) # Renamed
        if not np.any(m_node_build): continue
        
        s_in_node_vals_build = soma_mask_local[m_node_build] # Renamed
        s_in_node_unique_build = np.unique(s_in_node_vals_build[s_in_node_vals_build > 0]) # Renamed
        # Filter somas to only those for which we have properties (valid original somas)
        orig_somas_list_build = sorted([s_lbl for s_lbl in s_in_node_unique_build if s_lbl in soma_props_for_cell]) # Renamed
        
        obj_sl_build = find_objects(m_node_build.astype(np.int32)) # Renamed
        nodes[lbl_node] = {
            'mask_bbox_local': obj_sl_build[0] if obj_sl_build and obj_sl_build[0] else None,
            'orig_somas': orig_somas_list_build, 
            'volume': np.sum(m_node_build)
        }
        flush_print(f"{log_prefix}   Node {lbl_node}: Vol={nodes[lbl_node]['volume']}, OrigSomas={nodes[lbl_node]['orig_somas']}")

    for i_idx_graph in range(len(seg_lbls)): # Renamed i_idx
        lbl_A_graph = seg_lbls[i_idx_graph] # Renamed
        mask_A_graph = (current_cell_segments_mask_local == lbl_A_graph) # Renamed
        if not np.any(mask_A_graph) or lbl_A_graph not in nodes: continue # Ensure node_A exists

        dil_A_graph = binary_dilation(mask_A_graph, footprint=footprint_d_graph) # Renamed
        pot_neigh_mask_graph = dil_A_graph & original_cell_mask_local & \
                               (current_cell_segments_mask_local != lbl_A_graph) & \
                               (current_cell_segments_mask_local > 0) # Renamed
        if not np.any(pot_neigh_mask_graph): continue
        
        neigh_lbls_graph = np.unique(current_cell_segments_mask_local[pot_neigh_mask_graph]) # Renamed
        for lbl_B_graph in neigh_lbls_graph: # Renamed
            if lbl_B_graph <= lbl_A_graph or lbl_B_graph not in nodes: continue # Avoid duplicates and ensure node_B exists
            
            mask_B_graph = (current_cell_segments_mask_local == lbl_B_graph) # Renamed
            if not np.any(mask_B_graph): continue

            edge_key_graph = tuple(sorted((lbl_A_graph, lbl_B_graph))) # Renamed
            if edge_key_graph in edges: continue

            flush_print(f"{log_prefix}   Checking interface: {edge_key_graph}")
            
            # Calculate avg_soma_intensity for this specific interface
            somas_A = nodes[lbl_A_graph].get('orig_somas', [])
            somas_B = nodes[lbl_B_graph].get('orig_somas', [])
            combined_interface_somas = list(set(somas_A + somas_B))
            
            interface_soma_intensities = [soma_props_for_cell[s_lbl]['mean_intensity'] 
                                          for s_lbl in combined_interface_somas 
                                          if s_lbl in soma_props_for_cell and 'mean_intensity' in soma_props_for_cell[s_lbl]]
            
            avg_soma_intensity_for_this_interface = np.mean(interface_soma_intensities) if interface_soma_intensities else 1.0 # Default to 1.0 to avoid div by zero
            if avg_soma_intensity_for_this_interface < 1e-6 : avg_soma_intensity_for_this_interface = 1.0
            
            flush_print(f"{log_prefix}     Edge{edge_key_graph}: OrigSomasA={somas_A}, OrigSomasB={somas_B}, AvgSomaIntensityForIface={avg_soma_intensity_for_this_interface:.2f}")

            if_metrics_graph = _calculate_interface_metrics( # Renamed
                mask_A_graph, mask_B_graph, original_cell_mask_local, intensity_local, 
                avg_soma_intensity_for_this_interface, # Pass the calculated average
                spacing_tuple, local_analysis_radius, min_local_intensity_difference,
                min_path_intensity_ratio_heuristic, # For comparison printout
                log_prefix=f"{log_prefix}     Edge{edge_key_graph}")
            edges[edge_key_graph] = if_metrics_graph
            
    flush_print(f"{log_prefix} Graph built: {len(nodes)} nodes, {len(edges)} edges.")
    return nodes, edges

def separate_multi_soma_cells(
    segmentation_mask: np.ndarray, intensity_volume: np.ndarray, soma_mask: np.ndarray,
    spacing: Optional[Tuple[float, float, float]], 
    chunk_shape: Tuple[int, int, int] = (128, 512, 512),
    overlap: int = 32,
    **kwargs 
) -> np.ndarray:
    """
    A wrapper function that processes a large volume in manageable, overlapping chunks,
    performs a pixel-perfect stitch, and then runs a final global re-merging phase.
    """
    log_wrapper_prefix = "[SepMultiSoma_WRAPPER]"
    flush_print(f"{log_wrapper_prefix} Starting chunked processing...")
    
    memmap_dir = kwargs.get("memmap_dir", "ramiseg_temp_memmap")
    if memmap_dir and not os.path.exists(memmap_dir):
        os.makedirs(memmap_dir)

    # --- Initial analysis on the full volume to get multi-soma list ---
    unique_initial_labels = np.unique(segmentation_mask[segmentation_mask > 0])
    if unique_initial_labels.size == 0: return np.zeros_like(segmentation_mask, dtype=np.int32)
    
    cell_to_somas_orig: Dict[int, Set[int]] = {lbl: set() for lbl in unique_initial_labels}
    present_soma_labels_orig = np.unique(soma_mask[soma_mask > 0])
    if present_soma_labels_orig.size == 0: 
        if np.any(segmentation_mask): return relabel_sequential(segmentation_mask.astype(np.int32), offset=1)[0]
        return segmentation_mask.astype(np.int32)

    for soma_lbl_init in present_soma_labels_orig:
        soma_loc_mask_init = (soma_mask == soma_lbl_init)
        cell_lbls_under_soma_init = np.unique(segmentation_mask[soma_loc_mask_init])
        for cell_lbl_init in cell_lbls_under_soma_init:
            if cell_lbl_init > 0 and cell_lbl_init in cell_to_somas_orig:
                cell_to_somas_orig[cell_lbl_init].add(soma_lbl_init)
    multi_soma_cell_labels_list = [lbl for lbl, somas in cell_to_somas_orig.items() if len(somas) > 1]
    
    if not multi_soma_cell_labels_list:
        if np.any(segmentation_mask): return relabel_sequential(segmentation_mask.astype(np.int32), offset=1)[0]
        return segmentation_mask.astype(np.int32)

    # --- Phase 1: Process Chunks and save individual results ---
    next_label_offset = (np.max(unique_initial_labels) if unique_initial_labels.size > 0 else 0) + 1
    global_provenance_map = {}
    chunk_slices = list(_get_chunk_slices(segmentation_mask.shape, chunk_shape, overlap))
    chunk_result_paths = {}
    print(f'Splitting in {len(chunk_slices)} chunks')

    for i, chunk_slice in enumerate(tqdm(chunk_slices, desc=f"{log_wrapper_prefix} Processing Chunks")):
        seg_chunk = np.array(segmentation_mask[chunk_slice])
        intensity_chunk = np.array(intensity_volume[chunk_slice])
        soma_chunk = np.array(soma_mask[chunk_slice])
        
        chunk_result, chunk_provenance = _separate_multi_soma_cells_chunk(
            segmentation_mask=seg_chunk, intensity_volume=intensity_chunk, soma_mask=soma_chunk,
            spacing=spacing, label_offset=next_label_offset,
            multi_soma_cell_labels_list=multi_soma_cell_labels_list,
            **kwargs
        )
        
        global_provenance_map.update(chunk_provenance)
        max_label_in_chunk = np.max(chunk_result)
        if max_label_in_chunk >= next_label_offset:
            next_label_offset = max_label_in_chunk + 1
        
        chunk_path = os.path.join(memmap_dir, f"chunk_{i}_result.mmp")
        chunk_memmap = np.memmap(chunk_path, dtype=np.int32, mode='w+', shape=seg_chunk.shape)
        chunk_memmap[:] = chunk_result[:]
        chunk_memmap.flush()
        chunk_result_paths[i] = (chunk_path, seg_chunk.shape)
        del chunk_memmap, chunk_result, seg_chunk, intensity_chunk, soma_chunk
        gc.collect()

    # --- Phase 2: Pixel-Perfect Stitching ---
    flush_print(f"\n{log_wrapper_prefix} All chunks processed. Starting Stitching Phase 1 (Pixel Correspondence)...")
    label_map = {}
    shape_in_chunks = [
        len(range(0, segmentation_mask.shape[0], chunk_shape[0] - overlap)),
        len(range(0, segmentation_mask.shape[1], chunk_shape[1] - overlap)),
        len(range(0, segmentation_mask.shape[2], chunk_shape[2] - overlap))
    ]
    for i, chunk_slice1 in enumerate(tqdm(chunk_slices, desc=f"{log_wrapper_prefix} Analyzing Overlaps")):
        # (Neighbor finding logic is correct and remains)
        cz, cy, cx = np.unravel_index(i, shape_in_chunks)
        neighbors_indices = []
        if cz + 1 < shape_in_chunks[0]: neighbors_indices.append(np.ravel_multi_index((cz + 1, cy, cx), shape_in_chunks))
        if cy + 1 < shape_in_chunks[1]: neighbors_indices.append(np.ravel_multi_index((cz, cy + 1, cx), shape_in_chunks))
        if cx + 1 < shape_in_chunks[2]: neighbors_indices.append(np.ravel_multi_index((cz, cy, cx + 1), shape_in_chunks))

        for j in neighbors_indices:
            if j >= len(chunk_slices): continue
            chunk_slice2 = chunk_slices[j]
            overlap_slice = tuple(slice(max(s1.start, s2.start), min(s1.stop, s2.stop)) for s1, s2 in zip(chunk_slice1, chunk_slice2))
            if any(s.start >= s.stop for s in overlap_slice): continue
            
            relative_overlap1 = tuple(slice(s.start - cs.start, s.stop - cs.start) for s, cs in zip(overlap_slice, chunk_slice1))
            relative_overlap2 = tuple(slice(s.start - cs.start, s.stop - cs.start) for s, cs in zip(overlap_slice, chunk_slice2))

            path1, shape1 = chunk_result_paths[i]
            res1 = np.memmap(path1, dtype=np.int32, mode='r', shape=shape1)
            overlap1 = res1[relative_overlap1]
            path2, shape2 = chunk_result_paths[j]
            res2 = np.memmap(path2, dtype=np.int32, mode='r', shape=shape2)
            overlap2 = res2[relative_overlap2]
            
            correspondence_mask = (overlap1 > 0) & (overlap2 > 0)
            if not np.any(correspondence_mask): continue
            
            labels1 = overlap1[correspondence_mask]
            labels2 = overlap2[correspondence_mask]
            unique_pairs = np.unique(np.vstack([labels1, labels2]), axis=1).T
            
            for l1, l2 in unique_pairs:
                root1 = label_map.get(l1, l1)
                root2 = label_map.get(l2, l2)
                if root1 != root2:
                    unified_label = min(root1, root2)
                    for k, v in label_map.items():
                        if v == max(root1, root2): label_map[k] = unified_label
                    label_map[max(root1, root2)] = unified_label
                    label_map[min(root1, root2)] = unified_label

    # --- Phase 3: Assembling the Provisionally Stitched Mask ---
    flush_print(f"\n{log_wrapper_prefix} Stitching Phase 1 complete. Assembling provisional mask...")
    prov_stitched_path = os.path.join(memmap_dir, "prov_stitched.mmp")
    prov_stitched_mask = np.memmap(prov_stitched_path, dtype=np.int32, mode='w+', shape=segmentation_mask.shape)
    
    untouched_mask = ~np.isin(segmentation_mask, multi_soma_cell_labels_list)
    prov_stitched_mask[untouched_mask] = segmentation_mask[untouched_mask]
    
    for i, chunk_slice in enumerate(tqdm(chunk_slices, desc=f"{log_wrapper_prefix} Assembling Mask")):
        path, shape = chunk_result_paths[i]
        chunk_result = np.memmap(path, dtype=np.int32, mode='r+', shape=shape)
        unique_labels_in_chunk = np.unique(chunk_result)
        for old_label in unique_labels_in_chunk:
            if old_label == 0: continue
            root_label = label_map.get(old_label, old_label)
            if root_label != old_label:
                chunk_result[chunk_result == old_label] = root_label
        
        paste_mask = np.isin(segmentation_mask[chunk_slice], multi_soma_cell_labels_list)
        prov_stitched_mask[chunk_slice][paste_mask] = chunk_result[paste_mask]

    prov_stitched_mask.flush()

    # --- Phase 4: Global Re-merging of Stitched Siblings (YOUR CORRECT LOGIC) ---
    flush_print(f"\n{log_wrapper_prefix} Starting Stitching Phase 3 (Global Re-merging)...")
    
    parent_to_children = {}
    for child, parent in global_provenance_map.items():
        stitched_child = label_map.get(child, child)
        if parent not in parent_to_children:
            parent_to_children[parent] = set()
        parent_to_children[parent].add(stitched_child)

    final_remerge_map = {} 
    footprint = footprint_rectangle((3,3,3))

    for parent_label, child_labels_set in tqdm(parent_to_children.items(), desc=f"{log_wrapper_prefix} Re-merging Siblings"):
        child_labels = list(child_labels_set)
        if len(child_labels) <= 1: continue

        for i in range(len(child_labels)):
            for j in range(i + 1, len(child_labels)):
                label_a = child_labels[i]
                label_b = child_labels[j]
                
                root_a = final_remerge_map.get(label_a, label_a)
                root_b = final_remerge_map.get(label_b, label_b)
                if root_a == root_b: continue

                mask_a_full = (prov_stitched_mask == label_a)
                dilated_a = binary_dilation(mask_a_full, footprint=footprint)
                interface = dilated_a & (prov_stitched_mask == label_b)
                if not np.any(interface): continue
                
                interface_slices = find_objects(interface)
                if not interface_slices: continue
                bbox = interface_slices[0]
                
                mask_a_local = mask_a_full[bbox]
                mask_b_local = (prov_stitched_mask[bbox] == label_b)
                intensity_local = np.array(intensity_volume[bbox])
                parent_mask_local = mask_a_local | mask_b_local
                
                # We do not have easy access to the specific soma intensities here.
                # The _calculate_interface_metrics function is robust enough to work
                # with a placeholder and rely more on the LID heuristic.
                avg_soma_intensity = 15000 

                metrics = _calculate_interface_metrics(
                    mask_A_local=mask_a_local, mask_B_local=mask_b_local,
                    parent_mask_local=parent_mask_local, intensity_local=intensity_local,
                    avg_soma_intensity_for_interface=avg_soma_intensity,
                    spacing_tuple=spacing,
                    local_analysis_radius=kwargs.get("local_analysis_radius", 10),
                    min_local_intensity_difference=kwargs.get("min_local_intensity_difference", 0.05),
                    min_path_intensity_ratio_heuristic=kwargs.get("min_path_intensity_ratio", 0.8),
                    log_prefix=f"{log_wrapper_prefix}   - Sibling Edge ({label_a},{label_b})"
                )

                if metrics.get('should_merge_decision', False):
                    flush_print(f"{log_wrapper_prefix}     Decision: MERGE siblings {label_a} and {label_b}.")
                    root1 = final_remerge_map.get(label_a, label_a)
                    root2 = final_remerge_map.get(label_b, label_b)
                    if root1 != root2:
                        unified = min(root1, root2)
                        # Ensure transitive merges point to the single smallest root
                        for k, v in final_remerge_map.items():
                           if v == max(root1, root2): final_remerge_map[k] = unified
                        final_remerge_map[max(root1, root2)] = unified
                        final_remerge_map[label_a] = unified
                        final_remerge_map[label_b] = unified

    flush_print(f"{log_wrapper_prefix} Global re-merging analysis complete. Resolving {len(final_remerge_map)} merge rules...")

    # Create a final, clean map that resolves all chains (e.g., C->B, B->A becomes C->A)
    # This is a robust way to handle the merges and prevents any possibility of an infinite loop.
    final_clean_map = {}
    for old_label in list(final_remerge_map.keys()) + list(final_remerge_map.values()):
        root = final_remerge_map.get(old_label, old_label)
        # Follow the chain to the ultimate root
        visited_path = [root]
        while root in final_remerge_map:
            root = final_remerge_map[root]
            # Cycle detection
            if root in visited_path:
                # This case should ideally not happen with the min() logic, but this is a safeguard.
                # We break the cycle by choosing the smallest label in the cycle as the root.
                cycle_root = min(visited_path[visited_path.index(root):])
                root = cycle_root
                break
            visited_path.append(root)

        final_clean_map[old_label] = root
    
    flush_print(f"{log_wrapper_prefix} Applying {len(final_clean_map)} final merges to mask...")
    # Apply the clean, resolved map to the provisionally stitched mask
    for old_label, new_label in tqdm(final_clean_map.items(), desc=f"{log_wrapper_prefix} Finalizing Merges"):
        if old_label != new_label:
            prov_stitched_mask[prov_stitched_mask == old_label] = new_label

    # --- Final Cleanup and Relabeling ---
    final_result, _, _ = relabel_sequential(np.array(prov_stitched_mask), offset=1)
    
    del prov_stitched_mask
    gc.collect()
    if os.path.exists(prov_stitched_path): os.remove(prov_stitched_path)
    for path, _ in chunk_result_paths.values():
        if os.path.exists(path): os.remove(path)
    
    return final_result

def _get_chunk_slices(volume_shape, chunk_shape, overlap):
    """Generator that yields slices for overlapping chunks."""
    for z in range(0, volume_shape[0], chunk_shape[0] - overlap):
        for y in range(0, volume_shape[1], chunk_shape[1] - overlap):
            for x in range(0, volume_shape[2], chunk_shape[2] - overlap):
                yield (
                    slice(z, min(z + chunk_shape[0], volume_shape[0])),
                    slice(y, min(y + chunk_shape[1], volume_shape[1])),
                    slice(x, min(x + chunk_shape[2], volume_shape[2])),
                )

def _separate_multi_soma_cells_chunk(
    segmentation_mask: np.ndarray, intensity_volume: np.ndarray, soma_mask: np.ndarray,
    spacing: Optional[Tuple[float, float, float]], min_size_threshold: int,
    intensity_weight: float, max_seed_centroid_dist: float,
    min_path_intensity_ratio: float, min_local_intensity_difference: float, 
    local_analysis_radius: int, label_offset: int,
    memmap_dir: Optional[str], memmap_voxel_threshold: int,
    multi_soma_cell_labels_list: List[int]
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Worker function. Processes only the multi-soma cells present in its chunk and
    returns a mask of ONLY the new fragments, plus a provenance map.
    """
    
    log_main_prefix = "[SepMultiSoma_CHUNK]"
    provenance_map = {}

    try:
        if spacing is None: spacing_arr = np.array([1.0, 1.0, 1.0]); spacing_tuple = (1.0,1.0,1.0)
        else:
            try: spacing_tuple = tuple(float(s) for s in spacing); assert len(spacing_tuple) == 3; spacing_arr = np.array(spacing_tuple)
            except: spacing_tuple = (1.0,1.0,1.0); spacing_arr = np.array([1.0,1.0,1.0])
        flush_print(f"{log_main_prefix} Using 3D spacing (z,y,x): {spacing_tuple}")
        
        # This mask will ONLY contain the new fragments we create and untouched cells.
        final_output_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
        unique_initial_labels = np.unique(segmentation_mask[segmentation_mask > 0])
        if unique_initial_labels.size == 0: 
            flush_print(f"{log_main_prefix} Seg mask empty in chunk."); 
            return final_output_mask, {}
        
        cell_to_somas_orig: Dict[int, Set[int]] = {lbl: set() for lbl in unique_initial_labels}
        present_soma_labels_orig = np.unique(soma_mask[soma_mask > 0])
        if present_soma_labels_orig.size == 0: 
            flush_print(f"{log_main_prefix} Soma mask empty in chunk. Returning original seg.")
            return segmentation_mask.copy(), {}

        for soma_lbl_init in present_soma_labels_orig:
            soma_loc_mask_init = (soma_mask == soma_lbl_init)
            cell_lbls_under_soma_init = np.unique(segmentation_mask[soma_loc_mask_init])
            for cell_lbl_init in cell_lbls_under_soma_init:
                if cell_lbl_init > 0 and cell_lbl_init in cell_to_somas_orig:
                    cell_to_somas_orig[cell_lbl_init].add(soma_lbl_init)

        # Determine which of the GLOBAL multi-soma cells are present in THIS chunk's segmentation mask
        chunk_multi_soma_labels = [lbl for lbl in multi_soma_cell_labels_list if lbl in unique_initial_labels and len(cell_to_somas_orig.get(lbl, [])) > 1]
        
        # Copy the single-soma cells (or non-multi-soma cells) to the output directly
        for lbl_init in unique_initial_labels:
            if lbl_init not in chunk_multi_soma_labels:
                final_output_mask[segmentation_mask == lbl_init] = lbl_init

        if not chunk_multi_soma_labels:
            flush_print(f"{log_main_prefix} No multi-soma cells from the global list were found in this chunk.")
            return final_output_mask, {}
        
        flush_print(f"{log_main_prefix} Found {len(chunk_multi_soma_labels)} multi-soma cells to process in this chunk.")
        next_global_label_offset_val = label_offset
        flush_print(f"{log_main_prefix} Initial next_global_label_offset for chunk: {next_global_label_offset_val}")

        footprint_p1_dilation_val = footprint_rectangle((3,3,3)) 

        for cell_idx_p1, cell_label_p1 in enumerate(tqdm(chunk_multi_soma_labels, desc=f"{log_main_prefix} P1:CellProc")):
            gc.collect()
            flush_print(f"\n{log_main_prefix}   P1 Processing cell L{cell_label_p1} ({cell_idx_p1+1}/{len(chunk_multi_soma_labels)})")
            
            original_cell_mask_full_p1gws = (segmentation_mask == cell_label_p1)
            obj_slices_p1gws = find_objects(original_cell_mask_full_p1gws)
            if not obj_slices_p1gws or obj_slices_p1gws[0] is None: continue
            bbox_p1gws = obj_slices_p1gws[0]
            pad_p1gws = 3 
            local_bbox_slices_p1gws = tuple(slice(max(0, s.start - pad_p1gws), min(dim_size, s.stop + pad_p1gws)) for s, dim_size in zip(bbox_p1gws, segmentation_mask.shape))
            local_shape = tuple(s.stop - s.start for s in local_bbox_slices_p1gws)
            num_voxels = np.prod(local_shape)
            use_memmap_for_this_cell = memmap_dir is not None and (num_voxels > memmap_voxel_threshold)

            original_cell_mask_local_for_gws_p1gws = original_cell_mask_full_p1gws[local_bbox_slices_p1gws]
            if not np.any(original_cell_mask_local_for_gws_p1gws): continue
            soma_mask_local_orig_labels_gws_p1gws = soma_mask[local_bbox_slices_p1gws]
            intensity_local_gws_p1gws = intensity_volume[local_bbox_slices_p1gws]
            active_soma_labels_in_cell_gws_p1gws = sorted(list(cell_to_somas_orig[cell_label_p1]))
            
            soma_props_local_gws_p1gws: Dict[int, Dict[str, Any]] = {} 
            temp_soma_labeled_for_props_gws_p1gws = np.zeros_like(soma_mask_local_orig_labels_gws_p1gws, dtype=np.int32)
            for sl_orig_gws_p1gws in active_soma_labels_in_cell_gws_p1gws:
                temp_soma_labeled_for_props_gws_p1gws[(soma_mask_local_orig_labels_gws_p1gws == sl_orig_gws_p1gws) & original_cell_mask_local_for_gws_p1gws] = sl_orig_gws_p1gws
            if np.any(temp_soma_labeled_for_props_gws_p1gws):
                try:
                    props_gws_p1gws = regionprops(temp_soma_labeled_for_props_gws_p1gws, intensity_image=intensity_local_gws_p1gws)
                    for p_item_gws_p1gws in props_gws_p1gws:
                        if p_item_gws_p1gws.area > 0 : 
                             soma_props_local_gws_p1gws[p_item_gws_p1gws.label] = {'centroid': p_item_gws_p1gws.centroid, 'area': p_item_gws_p1gws.area, 'mean_intensity': p_item_gws_p1gws.mean_intensity if hasattr(p_item_gws_p1gws, 'mean_intensity') else 0.0}
                except Exception:
                    for sl_orig_gws_p1gws in active_soma_labels_in_cell_gws_p1gws:
                        coords_gws_p1gws = np.argwhere((soma_mask_local_orig_labels_gws_p1gws == sl_orig_gws_p1gws) & original_cell_mask_local_for_gws_p1gws)
                        if coords_gws_p1gws.shape[0] > 0:
                            soma_mean_int = np.mean(intensity_local_gws_p1gws[tuple(coords_gws_p1gws.T)])
                            soma_props_local_gws_p1gws[sl_orig_gws_p1gws] = {'centroid': np.mean(coords_gws_p1gws, axis=0), 'area': coords_gws_p1gws.shape[0], 'mean_intensity': soma_mean_int}

            valid_soma_labels_for_ws_gws_p1gws = [lbl for lbl in active_soma_labels_in_cell_gws_p1gws if lbl in soma_props_local_gws_p1gws and soma_props_local_gws_p1gws[lbl]['area'] > 0]
            if len(valid_soma_labels_for_ws_gws_p1gws) <= 1: continue
            
            num_seeds_gws_p1gws = len(valid_soma_labels_for_ws_gws_p1gws); soma_idx_map_gws_p1gws = {lbl: i for i, lbl in enumerate(valid_soma_labels_for_ws_gws_p1gws)}; adj_matrix_gws_p1gws = np.zeros((num_seeds_gws_p1gws, num_seeds_gws_p1gws), dtype=bool)
            
            if use_memmap_for_this_cell:
                cost_array_path = os.path.join(memmap_dir, f"chunk_cell_{cell_label_p1}_cost.mmp")
                cost_array_local_gws_p1gws = np.memmap(cost_array_path, dtype=np.float32, mode='w+', shape=local_shape)
            else:
                cost_array_local_gws_p1gws = np.empty(local_shape, dtype=np.float32)
            cost_array_local_gws_p1gws[:] = np.inf

            if np.any(original_cell_mask_local_for_gws_p1gws):
                max_intensity_val_local_gws_p1gws = np.max(intensity_local_gws_p1gws[original_cell_mask_local_for_gws_p1gws])
                coords = np.argwhere(original_cell_mask_local_for_gws_p1gws)
                chunk_size = 5_000_000
                num_chunks = (len(coords) + chunk_size - 1) // chunk_size
                for i in range(num_chunks):
                    start_idx = i * chunk_size
                    end_idx = start_idx + chunk_size
                    chunk_coords = coords[start_idx:end_idx]
                    z, y, x = chunk_coords[:, 0], chunk_coords[:, 1], chunk_coords[:, 2]
                    intensities_chunk = intensity_local_gws_p1gws[z, y, x]
                    cost_values = np.maximum(1e-6, max_intensity_val_local_gws_p1gws - intensities_chunk)
                    cost_array_local_gws_p1gws[z, y, x] = cost_values
                if isinstance(cost_array_local_gws_p1gws, np.memmap): cost_array_local_gws_p1gws.flush()
                del coords

            for i_gws_p1gws in range(num_seeds_gws_p1gws):
                for j_gws_p1gws in range(i_gws_p1gws + 1, num_seeds_gws_p1gws):
                    lbl1_gws_p1gws, lbl2_gws_p1gws = valid_soma_labels_for_ws_gws_p1gws[i_gws_p1gws], valid_soma_labels_for_ws_gws_p1gws[j_gws_p1gws]
                    prop1_gws_p1gws, prop2_gws_p1gws = soma_props_local_gws_p1gws[lbl1_gws_p1gws], soma_props_local_gws_p1gws[lbl2_gws_p1gws]
                    c1_phys_gws_p1gws = np.array(prop1_gws_p1gws['centroid']) * spacing_arr
                    c2_phys_gws_p1gws = np.array(prop2_gws_p1gws['centroid']) * spacing_arr
                    dist_um_gws_p1gws = np.linalg.norm(c1_phys_gws_p1gws - c2_phys_gws_p1gws)
                    if dist_um_gws_p1gws > max_seed_centroid_dist: continue
                    
                    c1_vox_gws_p1gws = tuple(np.round(prop1_gws_p1gws['centroid']).astype(int))
                    c2_vox_gws_p1gws = tuple(np.round(prop2_gws_p1gws['centroid']).astype(int))
                    c1_vox_c_gws_p1gws = tuple(np.clip(c1_vox_gws_p1gws[d_idx],0,s-1) for d_idx,s in enumerate(cost_array_local_gws_p1gws.shape))
                    c2_vox_c_gws_p1gws = tuple(np.clip(c2_vox_gws_p1gws[d_idx],0,s-1) for d_idx,s in enumerate(cost_array_local_gws_p1gws.shape))
                    
                    if not original_cell_mask_local_for_gws_p1gws[c1_vox_c_gws_p1gws] or not original_cell_mask_local_for_gws_p1gws[c2_vox_c_gws_p1gws]: continue
                    
                    path_median_intensity_gws_p1gws = 0.0
                    try:
                        path_indices_tup_gws_p1gws, _ = route_through_array(cost_array_local_gws_p1gws, c1_vox_c_gws_p1gws, c2_vox_c_gws_p1gws, fully_connected=True, geometric=False)
                        if isinstance(path_indices_tup_gws_p1gws, np.ndarray) and path_indices_tup_gws_p1gws.ndim==2 and path_indices_tup_gws_p1gws.shape[1]>0:
                            path_intensities_vals_gws_p1gws = intensity_local_gws_p1gws[path_indices_tup_gws_p1gws[0], path_indices_tup_gws_p1gws[1], path_indices_tup_gws_p1gws[2]]
                            if path_intensities_vals_gws_p1gws.size > 0: path_median_intensity_gws_p1gws = np.median(path_intensities_vals_gws_p1gws)
                    except (ValueError, IndexError):
                        pass
                    
                    ref_intensity_val_gws_p1gws = max(prop1_gws_p1gws.get('mean_intensity',1.0), prop2_gws_p1gws.get('mean_intensity',1.0), 1e-6)
                    ratio_val_gws_p1gws = path_median_intensity_gws_p1gws / ref_intensity_val_gws_p1gws if ref_intensity_val_gws_p1gws > 1e-6 else float('inf')
                    if ratio_val_gws_p1gws >= min_path_intensity_ratio:
                        adj_matrix_gws_p1gws[soma_idx_map_gws_p1gws[lbl1_gws_p1gws], soma_idx_map_gws_p1gws[lbl2_gws_p1gws]] = True
                        adj_matrix_gws_p1gws[soma_idx_map_gws_p1gws[lbl2_gws_p1gws], soma_idx_map_gws_p1gws[lbl1_gws_p1gws]] = True

            if use_memmap_for_this_cell:
                markers_path = os.path.join(memmap_dir, f"chunk_cell_{cell_label_p1}_markers.mmp")
                ws_markers_local_gws_p1gws = np.memmap(markers_path, dtype=np.int32, mode='w+', shape=local_shape)
            else:
                ws_markers_local_gws_p1gws = np.zeros(local_shape, dtype=np.int32)
            ws_markers_local_gws_p1gws[:] = 0

            current_ws_marker_id_gws_p1gws = 1
            ws_marker_to_orig_somas_gws_p1gws: Dict[int, List[int]] = {}; orig_soma_to_ws_marker_gws_p1gws: Dict[int, int] = {}
            if np.any(adj_matrix_gws_p1gws):
                n_comps_adj_gws, comp_lbls_adj_gws = ndimage.label(adj_matrix_gws_p1gws)
                for k_comp_adj in range(1, n_comps_adj_gws + 1):
                    soma_indices_in_group_gws = np.where(comp_lbls_adj_gws == k_comp_adj)[0]
                    if not soma_indices_in_group_gws.size: continue
                    group_marker_id_gws = current_ws_marker_id_gws_p1gws
                    current_group_somas_gws = []
                    for seed_idx_gws in soma_indices_in_group_gws:
                        orig_s_lbl_gws = valid_soma_labels_for_ws_gws_p1gws[seed_idx_gws]
                        current_group_somas_gws.append(orig_s_lbl_gws)
                        ws_markers_local_gws_p1gws[(soma_mask_local_orig_labels_gws_p1gws == orig_s_lbl_gws) & original_cell_mask_local_for_gws_p1gws] = group_marker_id_gws
                        orig_soma_to_ws_marker_gws_p1gws[orig_s_lbl_gws] = group_marker_id_gws
                    if current_group_somas_gws: ws_marker_to_orig_somas_gws_p1gws[group_marker_id_gws] = sorted(current_group_somas_gws); current_ws_marker_id_gws_p1gws +=1
            for sl_orig_gws in valid_soma_labels_for_ws_gws_p1gws:
                if sl_orig_gws not in orig_soma_to_ws_marker_gws_p1gws:
                    group_marker_id_gws = current_ws_marker_id_gws_p1gws
                    ws_markers_local_gws_p1gws[(soma_mask_local_orig_labels_gws_p1gws == sl_orig_gws) & original_cell_mask_local_for_gws_p1gws] = group_marker_id_gws
                    orig_soma_to_ws_marker_gws_p1gws[sl_orig_gws] = group_marker_id_gws
                    ws_marker_to_orig_somas_gws_p1gws[group_marker_id_gws] = [sl_orig_gws]; current_ws_marker_id_gws_p1gws +=1
            
            num_final_ws_markers_gws_p1gws = current_ws_marker_id_gws_p1gws - 1
            if num_final_ws_markers_gws_p1gws <= 1:
                new_label = next_global_label_offset_val
                final_output_mask[original_cell_mask_full_p1gws] = new_label
                provenance_map[new_label] = cell_label_p1
                next_global_label_offset_val += 1
                continue 
            
            else:
                if use_memmap_for_this_cell:
                    dt_intermediate_path = os.path.join(memmap_dir, f"chunk_cell_{cell_label_p1}_dt_intermediate.mmp")
                    intermediate_shape = (len(local_shape),) + tuple(local_shape)
                    dt_intermediate_memmap = np.memmap(dt_intermediate_path, dtype=np.float64, mode='w+', shape=intermediate_shape)
                    distance_transform_edt(original_cell_mask_local_for_gws_p1gws, sampling=spacing_tuple, output=dt_intermediate_memmap)
                    ws_landscape_path = os.path.join(memmap_dir, f"chunk_cell_{cell_label_p1}_landscape.mmp")
                    ws_landscape_gws = np.memmap(ws_landscape_path, dtype=np.float32, mode='w+', shape=local_shape)
                    final_dt_squared = np.add.reduce(dt_intermediate_memmap, axis=0)
                    np.sqrt(final_dt_squared, out=final_dt_squared)
                    np.negative(final_dt_squared, out=ws_landscape_gws)
                    dt_local_gws = final_dt_squared
                    del dt_intermediate_memmap
                else:
                    dt_local_gws = distance_transform_edt(original_cell_mask_local_for_gws_p1gws, sampling=spacing_tuple)
                    ws_landscape_gws = -dt_local_gws.astype(np.float32)

                if intensity_weight > 1e-6:
                    icell_gws = intensity_local_gws_p1gws[original_cell_mask_local_for_gws_p1gws]
                    if icell_gws.size > 0:
                        min_ic_gws, max_ic_gws = np.min(icell_gws), np.max(icell_gws)
                        if (max_ic_gws - min_ic_gws) > 1e-6:
                            norm_int_term_gws = np.zeros_like(ws_landscape_gws); norm_int_term_gws[original_cell_mask_local_for_gws_p1gws] = (max_ic_gws - icell_gws) / (max_ic_gws - min_ic_gws)
                            max_dt_gws = np.max(dt_local_gws); ws_landscape_gws += intensity_weight * norm_int_term_gws * (max_dt_gws if max_dt_gws > 1e-6 else 1.0)
                
                ws_markers_local_gws_p1gws[~original_cell_mask_local_for_gws_p1gws] = 0
                
                global_ws_result_local_p1gws = _watershed_with_simpleitk(landscape=ws_landscape_gws, markers=ws_markers_local_gws_p1gws, log_prefix=f"{log_main_prefix}     L{cell_label_p1}")
                
                unique_ws_res_lbls_gws = np.unique(global_ws_result_local_p1gws[global_ws_result_local_p1gws > 0])
                current_next_lbl_cell_gws = next_global_label_offset_val
                label_map = {}
                for res_lbl_gws in unique_ws_res_lbls_gws:
                    new_label = current_next_lbl_cell_gws
                    label_map[res_lbl_gws] = new_label
                    provenance_map[new_label] = cell_label_p1
                    current_next_lbl_cell_gws += 1
                
                temp_gws_mask_local_p1gws = np.zeros(local_shape, dtype=np.int32)
                for old_label, new_label in label_map.items():
                    temp_gws_mask_local_p1gws[global_ws_result_local_p1gws == old_label] = new_label
                next_global_label_offset_val = current_next_lbl_cell_gws
                
                del global_ws_result_local_p1gws
                del cost_array_local_gws_p1gws, dt_local_gws, ws_landscape_gws, ws_markers_local_gws_p1gws
                gc.collect()

                ws_lines_gws = (temp_gws_mask_local_p1gws == 0) & original_cell_mask_local_for_gws_p1gws
                if np.any(ws_lines_gws):
                    line_coords_gws = np.argwhere(ws_lines_gws)
                    for zlg,ylg,xlg in line_coords_gws:
                        nh_slice_gws = tuple(slice(max(0,c-1),min(s,c+2))for c,s in zip((zlg,ylg,xlg),temp_gws_mask_local_p1gws.shape))
                        nh_vals_gws = temp_gws_mask_local_p1gws[nh_slice_gws]; un_nh_gws,cts_nh_gws=np.unique(nh_vals_gws[nh_vals_gws>0],return_counts=True)
                        if un_nh_gws.size > 0: temp_gws_mask_local_p1gws[zlg,ylg,xlg] = un_nh_gws[np.argmax(cts_nh_gws)]
                
                frag_merged_gws,sfm_iters_gws=True,0
                while frag_merged_gws and sfm_iters_gws < 10:
                    sfm_iters_gws+=1; frag_merged_gws=False
                    curr_lbls_tgws = np.unique(temp_gws_mask_local_p1gws[temp_gws_mask_local_p1gws>0])
                    if len(curr_lbls_tgws)<=1:break
                    for lbl_sfm_gws in curr_lbls_tgws:
                        curr_frag_gws=(temp_gws_mask_local_p1gws==lbl_sfm_gws);size_sfm_gws=np.sum(curr_frag_gws)
                        if size_sfm_gws > 0 and size_sfm_gws < min_size_threshold:
                            dil_frag_gws=binary_dilation(curr_frag_gws,footprint=footprint_p1_dilation_val)
                            neigh_reg_gws = dil_frag_gws & (~curr_frag_gws) & original_cell_mask_local_for_gws_p1gws & (temp_gws_mask_local_p1gws!=0) & (temp_gws_mask_local_p1gws!=lbl_sfm_gws)
                            if not np.any(neigh_reg_gws): continue
                            neigh_lbls_gws,neigh_cts_gws=np.unique(temp_gws_mask_local_p1gws[neigh_reg_gws],return_counts=True)
                            if neigh_lbls_gws.size==0:continue
                            largest_neigh_lbl_gws=neigh_lbls_gws[np.argmax(neigh_cts_gws)];temp_gws_mask_local_p1gws[curr_frag_gws]=largest_neigh_lbl_gws
                            frag_merged_gws=True;break
                
                # --- Phase 1.5: Build Local Graph & Merge Weak Interfaces for THIS CELL ---
            flush_print(f"{log_main_prefix}     P1 L{cell_label_p1}: Building local graph for P1.5 merges on {np.sum(np.unique(temp_gws_mask_local_p1gws)>0)} GWS segments.")
            local_nodes_p15, local_edges_p15 = _build_adjacency_graph_for_cell(
                temp_gws_mask_local_p1gws,
                original_cell_mask_local_for_gws_p1gws,
                soma_mask_local_orig_labels_gws_p1gws,
                soma_props_local_gws_p1gws,
                intensity_local_gws_p1gws,
                spacing_tuple,
                local_analysis_radius, min_local_intensity_difference,
                min_path_intensity_ratio,
                log_prefix=f"{log_main_prefix}       GraphBuild_P1.5 L{cell_label_p1}"
            )

            if local_edges_p15: 
                flush_print(f"{log_main_prefix}     P1 L{cell_label_p1}: P1.5 - Checking {len(local_edges_p15)} interfaces for merging.")
                p1_5_local_merge_passes = 0; MAX_P1_5_LOCAL_PASSES = 5
                p1_5_merged_in_pass_loc = True
                while p1_5_merged_in_pass_loc and p1_5_local_merge_passes < MAX_P1_5_LOCAL_PASSES:
                    p1_5_local_merge_passes += 1; p1_5_merged_in_pass_loc = False
                    flush_print(f"{log_main_prefix}       P1.5 L{cell_label_p1} Merge Pass {p1_5_local_merge_passes}")
                    
                    current_local_labels_in_temp_gws = np.unique(temp_gws_mask_local_p1gws[temp_gws_mask_local_p1gws > 0])
                    if len(current_local_labels_in_temp_gws) <=1: flush_print(f"{log_main_prefix}         P1.5 L{cell_label_p1}: Only {len(current_local_labels_in_temp_gws)} segment(s) left. No more P1.5 merges."); break

                    sorted_local_edges_to_check_p15 = sorted(list(local_edges_p15.keys()))

                    for edge_key_local_p1_5_val in sorted_local_edges_to_check_p15:
                        lbl_A_loc_p15, lbl_B_loc_p15 = edge_key_local_p1_5_val
                        edge_metrics_local_p1_5_val = local_edges_p15.get(edge_key_local_p1_5_val)

                        if not edge_metrics_local_p1_5_val: continue 

                        mask_A_exists_loc_p15 = np.any(temp_gws_mask_local_p1gws == lbl_A_loc_p15)
                        mask_B_exists_loc_p15 = np.any(temp_gws_mask_local_p1gws == lbl_B_loc_p15)
                        if not mask_A_exists_loc_p15 or not mask_B_exists_loc_p15 or lbl_A_loc_p15 == lbl_B_loc_p15: 
                            if edge_key_local_p1_5_val in local_edges_p15: del local_edges_p15[edge_key_local_p1_5_val] 
                            continue
                        
                        should_merge_local_p1_5_val = edge_metrics_local_p1_5_val.get('should_merge_decision', False)
                        
                        if should_merge_local_p1_5_val:
                            flush_print(f"{log_main_prefix}       P1.5 L{cell_label_p1} Edge ({lbl_A_loc_p15},{lbl_B_loc_p15}): Metrics indicate MERGE.")
                            label_to_keep_loc_p15 = lbl_A_loc_p15; label_to_remove_loc_p15 = lbl_B_loc_p15
                            vol_A_loc_p15 = local_nodes_p15.get(lbl_A_loc_p15,{}).get('volume',0)
                            vol_B_loc_p15 = local_nodes_p15.get(lbl_B_loc_p15,{}).get('volume',0)

                            # This logic now correctly references only new labels, not cell_label_p1
                            if vol_A_loc_p15 < vol_B_loc_p15:
                                label_to_keep_loc_p15, label_to_remove_loc_p15 = lbl_B_loc_p15, lbl_A_loc_p15
                            
                            flush_print(f"{log_main_prefix}         Merging L{label_to_remove_loc_p15} into L{label_to_keep_loc_p15} in temp_gws_mask_local_p1gws.")
                            temp_gws_mask_local_p1gws[temp_gws_mask_local_p1gws == label_to_remove_loc_p15] = label_to_keep_loc_p15
                            p1_5_merged_in_pass_loc = True
                            
                            if label_to_remove_loc_p15 in local_nodes_p15: 
                                local_nodes_p15[label_to_keep_loc_p15]['volume'] += local_nodes_p15[label_to_remove_loc_p15]['volume']
                                local_nodes_p15[label_to_keep_loc_p15]['orig_somas'] = sorted(list(set(local_nodes_p15[label_to_keep_loc_p15]['orig_somas'] + local_nodes_p15[label_to_remove_loc_p15]['orig_somas'])))
                                del local_nodes_p15[label_to_remove_loc_p15]
                            
                            stale_edges_p15 = [ek for ek in local_edges_p15 if label_to_remove_loc_p15 in ek]
                            for sek_p15 in stale_edges_p15:
                                if sek_p15 in local_edges_p15: del local_edges_p15[sek_p15]
                            break 
                    if p1_5_merged_in_pass_loc: continue 
                    if not p1_5_merged_in_pass_loc: break 
            else:
                flush_print(f"{log_main_prefix}     P1 L{cell_label_p1}: P1.5 - No interfaces or no merges needed based on graph.")
            # --- [END] RE-INSERTED PHASE 1.5 BLOCK ---

            flush_print(f"{log_main_prefix}     P1 L{cell_label_p1} Writing final segments to global mask...")
            output_mask_local_view_final_p1_val = final_output_mask[local_bbox_slices_p1gws]
            output_mask_local_view_final_p1_val[original_cell_mask_local_for_gws_p1gws] = \
                temp_gws_mask_local_p1gws[original_cell_mask_local_for_gws_p1gws]
            final_output_mask[local_bbox_slices_p1gws] = output_mask_local_view_final_p1_val
            
            max_lbl_in_cell_after_p15_val = np.max(temp_gws_mask_local_p1gws) if np.any(temp_gws_mask_local_p1gws) else 0
            if max_lbl_in_cell_after_p15_val >= next_global_label_offset_val : 
                 next_global_label_offset_val = max(next_global_label_offset_val, max_lbl_in_cell_after_p15_val + 1)
            
            flush_print(f"{log_main_prefix}     P1 L{cell_label_p1}: Finished P1 & P1.5. Max label in cell: {max_lbl_in_cell_after_p15_val}. Next global offset: {next_global_label_offset_val}.")
            del temp_gws_mask_local_p1gws, local_nodes_p15, local_edges_p15

        return final_output_mask, provenance_map

    except Exception as e:
        flush_print(f"CRITICAL ERROR in _separate_multi_soma_cells_chunk: {e}")
        traceback.print_exc()
        return np.zeros_like(segmentation_mask, dtype=np.int32), {}

# --- fill_internal_voids (ensure it's defined as before) ---
def fill_internal_voids(segmentation_mask_input: np.ndarray) -> np.ndarray:
    log_fill_prefix = "[FillVoids]"
    flush_print(f"{log_fill_prefix} Starting Internal Void Filling...")
    t_start_fill = time.time()
    filled_mask_output = segmentation_mask_input.copy()
    
    # find_objects requires positive integer labels. Ensure input is appropriate.
    # If mask contains 0s, find_objects will skip them.
    # It returns a list where index i corresponds to label i+1.
    input_for_find_objects = segmentation_mask_input.astype(np.int32)
    if np.max(input_for_find_objects) == 0 : # All zeros or empty
        flush_print(f"{log_fill_prefix} Input mask is empty or all zeros. No voids to fill.")
        flush_print(f"{log_fill_prefix} Void Filling Complete. Time: {time.time()-t_start_fill:.2f}s")
        return filled_mask_output

    bboxes = find_objects(input_for_find_objects)
    
    voids_filled_count = 0
    voxels_added_total = 0

    if bboxes is None: # Should not happen if max > 0, but defensive.
        flush_print(f"{log_fill_prefix} No objects found by find_objects. No voids to fill.")
        flush_print(f"{log_fill_prefix} Void Filling Complete. Time: {time.time()-t_start_fill:.2f}s")
        return filled_mask_output

    for i, bbox_slices in enumerate(bboxes):
        label_val_fill = i + 1 # find_objects list is indexed by label-1
        
        if bbox_slices is None: continue # This label doesn't exist (e.g., if labels are not sequential from 1)

        if not (isinstance(bbox_slices, tuple) and len(bbox_slices) == segmentation_mask_input.ndim and \
                all(isinstance(s, slice) for s in bbox_slices)):
            flush_print(f"{log_fill_prefix}   Skipping label {label_val_fill} due to invalid bbox: {bbox_slices}")
            continue

        original_roi_fill = segmentation_mask_input[bbox_slices] # Use original mask for ROI
        cell_mask_roi_fill = (original_roi_fill == label_val_fill) # Create binary mask for current label in ROI
        
        if not np.any(cell_mask_roi_fill):
            flush_print(f"{log_fill_prefix}   ROI for label {label_val_fill} is empty for this label. Skipping fill.")
            continue

        original_voxels_fill = np.sum(cell_mask_roi_fill)
        try:
            # binary_fill_holes expects a 2D or 3D binary image.
            filled_cell_roi_fill = binary_fill_holes(cell_mask_roi_fill)
        except Exception as e_fill:
            flush_print(f"{log_fill_prefix}   Error filling holes for label {label_val_fill} in ROI: {e_fill}. Skipping.")
            continue

        filled_voxels_fill = np.sum(filled_cell_roi_fill)
        if filled_voxels_fill > original_voxels_fill:
            voxels_added_fill = filled_voxels_fill - original_voxels_fill
            voxels_added_total += voxels_added_fill
            voids_filled_count += 1
            flush_print(f"{log_fill_prefix}   Filled void in label {label_val_fill}: added {voxels_added_fill} voxels.")
            
            # Update the corresponding ROI in our result mask (filled_mask_output)
            # We use the filled_cell_roi_fill (a boolean mask) to specify which
            # voxels *within the bounding box* should be set to the current label.
            result_roi_view_fill = filled_mask_output[bbox_slices] # Get a view of the output mask's ROI
            result_roi_view_fill[filled_cell_roi_fill] = label_val_fill # Modify the view (updates filled_mask_output)
    
    flush_print(f"{log_fill_prefix} --- Void Filling Summary ---")
    if voids_filled_count > 0:
        flush_print(f"{log_fill_prefix}   Filled voids in {voids_filled_count} objects.")
        flush_print(f"{log_fill_prefix}   Total voxels added: {voxels_added_total}")
    else:
        flush_print(f"{log_fill_prefix}   No internal voids were found to fill.")
    flush_print(f"{log_fill_prefix} Void Filling Complete. Time: {time.time()-t_start_fill:.2f}s")
    return filled_mask_output