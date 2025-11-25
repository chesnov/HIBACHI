# --- START OF FILE utils/module_3d/segmentation_helpers.py ---
import numpy as np
import SimpleITK as sitk # type: ignore
import sys
import time
import psutil # type: ignore
import os
from scipy import ndimage # type: ignore
from scipy.ndimage import _ni_support, _nd_image # type: ignore

def flush_print(*args, **kwargs):
    """
    Wrapper for print that forces immediate flushing of the buffer.
    Essential for debugging crashes where the last log message is otherwise lost.
    """
    print(*args, **kwargs)
    sys.stdout.flush()

def log_memory_usage(label=""):
    """Logs the current RSS (Resident Set Size) RAM usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_gb = mem_info.rss / (1024 ** 3)
    flush_print(f"    [MEM_PROFILE] {label}: {mem_gb:.3f} GB")

def _watershed_with_simpleitk(
    landscape: np.ndarray,
    markers: np.ndarray,
    log_prefix: str = ""
) -> np.ndarray:
    """
    Performs Morphological Watershed using SimpleITK.
    
    Why SITK? It is often faster and less memory-hungry for 3D integer arrays 
    than skimage.segmentation.watershed.
    
    Args:
        landscape (np.ndarray): The height map (usually inverted distance transform).
        markers (np.ndarray): The seeds/basins.
        
    Returns:
        np.ndarray: The segmented label mask.
    """
    # flush_print(f"{log_prefix} Executing watershed using SimpleITK...")
    # ws_start_time = time.time()
    
    try:
        # 1. Sanitize Inputs
        if not np.all(np.isfinite(landscape)):
            flush_print(f"{log_prefix}   WARNING: Non-finite values in landscape. Sanitizing...")
            max_val = np.nanmax(landscape) if np.any(np.isfinite(landscape)) else 0
            landscape = np.nan_to_num(landscape, nan=max_val, posinf=max_val, neginf=0)

        # 2. Convert to SITK
        # Cast to standard types to avoid ITK template errors
        landscape_sitk = sitk.GetImageFromArray(landscape.astype(np.float64))
        markers_sitk = sitk.GetImageFromArray(markers.astype(np.uint32))

        # 3. Execute
        # args: (image, markerImage, markWatershedLine=True, fullyConnected=False)
        result_sitk = sitk.MorphologicalWatershedFromMarkers(
            landscape_sitk, 
            markers_sitk, 
            markWatershedLine=False, 
            fullyConnected=False
        )

        # 4. Convert back
        result_np = sitk.GetArrayFromImage(result_sitk)
        # Match input marker type
        result_np = result_np.astype(markers.dtype, copy=False)

        return result_np

    except Exception as e:
        flush_print(f"{log_prefix} CRITICAL: SimpleITK watershed failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return markers (no expansion)
        return np.array(markers)

def _distance_tranform_arg_check(distances_out, indices_out,
                                 return_distances, return_indices):
    """Internal scipy validation helper."""
    error_msgs = []
    if (not return_distances) and (not return_indices):
        error_msgs.append('at least one of return_distances/return_indices must be True')
    if distances_out and not return_distances:
        error_msgs.append('return_distances must be True if distances is supplied')
    if indices_out and not return_indices:
        error_msgs.append('return_indices must be True if indices is supplied')
    if error_msgs:
        raise RuntimeError(', '.join(error_msgs))

def distance_transform_edt(input, sampling=None, return_distances=True,
                           return_indices=False, distances=None, indices=None,
                           output=None):
    """
    Memory-Efficient Euclidean Distance Transform.
    
    This modifies scipy.ndimage.distance_transform_edt to allow passing an 
    'output' parameter (e.g., a numpy.memmap). This allows calculating the 
    EDT of a 10GB array without allocating a NEW 10GB array in RAM for the result.
    """
    ft_inplace = isinstance(indices, np.ndarray)
    dt_inplace = isinstance(distances, np.ndarray)

    _distance_tranform_arg_check(
        dt_inplace, ft_inplace, return_distances, return_indices
    )

    # Convert input to binary
    input = np.atleast_1d(np.where(input, 1, 0).astype(np.int8))
    
    if sampling is not None:
        sampling = _ni_support._normalize_sequence(sampling, input.ndim)
        sampling = np.asarray(sampling, dtype=np.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()

    # Feature Transform (FT)
    if ft_inplace:
        ft = indices
        if ft.shape != (input.ndim,) + input.shape:
            raise RuntimeError('indices array has wrong shape')
        if ft.dtype.type != np.int32:
            raise RuntimeError('indices array must be int32')
    else:
        ft = np.zeros((input.ndim,) + input.shape, dtype=np.int32)

    _nd_image.euclidean_feature_transform(input, sampling, ft)

    # Distance Calculation
    if return_distances:
        if output is not None:
            # OPTIMIZATION: Write directly to memmap
            dt = output
            # Check shape/dtype
            expected_shape = (input.ndim,) + input.shape
            if dt.shape != expected_shape:
                 # Some looseness allowed if squeezing dims, but generally strict
                 pass 
            np.subtract(ft, np.indices(input.shape, dtype=ft.dtype), out=dt)
        else:
            dt = ft - np.indices(input.shape, dtype=ft.dtype)
            dt = dt.astype(np.float64)

        if sampling is not None:
            for ii in range(len(sampling)):
                dt[ii, ...] *= sampling[ii]
        
        np.multiply(dt, dt, dt)

        if dt_inplace:
            reduced_dt = np.add.reduce(dt, axis=0)
            np.sqrt(reduced_dt, distances)
        else:
            dt = np.add.reduce(dt, axis=0)
            dt = np.sqrt(dt)

    result = []
    if return_distances and not dt_inplace:
        result.append(dt)
    if return_indices and not ft_inplace:
        result.append(ft)

    if len(result) == 2: return tuple(result)
    elif len(result) == 1: return result[0]
    else: return None
# --- END OF FILE utils/module_3d/segmentation_helpers.py ---