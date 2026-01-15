import os
import sys
import psutil
from typing import Optional, Tuple, Union, Any

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.ndimage import _ni_support, _nd_image  # type: ignore


def flush_print(*args: Any, **kwargs: Any) -> None:
    """
    Wrapper for print that forces immediate flushing of the stdout buffer.
    Essential for debugging crashes where the last log message is otherwise lost.
    """
    print(*args, **kwargs)
    sys.stdout.flush()


def log_memory_usage(label: str = "") -> None:
    """
    Logs the current RSS (Resident Set Size) RAM usage of the process.

    Args:
        label: Optional prefix tag for the log message.
    """
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

    SimpleITK's implementation is often significantly faster and more memory-efficient
    for large 3D integer arrays compared to `skimage.segmentation.watershed`.

    Args:
        landscape: The height map (typically inverted distance transform).
        markers: The seed markers (basins).
        log_prefix: String prefix for logging errors.

    Returns:
        np.ndarray: The segmented label mask. Returns markers if execution fails.
    """
    try:
        # 1. Sanitize Inputs
        if not np.all(np.isfinite(landscape)):
            flush_print(f"{log_prefix} WARNING: Non-finite values in landscape. Sanitizing...")
            max_val = np.nanmax(landscape) if np.any(np.isfinite(landscape)) else 0
            landscape = np.nan_to_num(
                landscape, nan=max_val, posinf=max_val, neginf=0
            )

        # 2. Convert to SITK
        # Cast to standard types to avoid ITK template matching errors
        landscape_sitk = sitk.GetImageFromArray(landscape.astype(np.float64))
        markers_sitk = sitk.GetImageFromArray(markers.astype(np.uint32))

        # 3. Execute Watershed
        # markWatershedLine=False ensures contiguous regions (no 1px gaps)
        result_sitk = sitk.MorphologicalWatershedFromMarkers(
            landscape_sitk,
            markers_sitk,
            markWatershedLine=False,
            fullyConnected=False
        )

        # 4. Convert back to NumPy
        result_np = sitk.GetArrayFromImage(result_sitk)
        # Explicitly clear SITK objects to release C++ memory
        del landscape_sitk, markers_sitk, result_sitk
        
        # Cast back to original marker dtype
        result_np = result_np.astype(markers.dtype, copy=False)

        return result_np

    except Exception as e:
        flush_print(f"{log_prefix} CRITICAL: SimpleITK watershed failed: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: return input markers (no expansion)
        return np.array(markers)


def _distance_tranform_arg_check(
    distances_out: bool,
    indices_out: bool,
    return_distances: bool,
    return_indices: bool
) -> None:
    """Internal validation helper from scipy source."""
    error_msgs = []
    if (not return_distances) and (not return_indices):
        error_msgs.append(
            'at least one of return_distances/return_indices must be True'
        )
    if distances_out and not return_distances:
        error_msgs.append(
            'return_distances must be True if distances is supplied'
        )
    if indices_out and not return_indices:
        error_msgs.append(
            'return_indices must be True if indices is supplied'
        )
    if error_msgs:
        raise RuntimeError(', '.join(error_msgs))


def distance_transform_edt(
    input: np.ndarray,
    sampling: Optional[Union[float, Tuple[float, ...]]] = None,
    return_distances: bool = True,
    return_indices: bool = False,
    distances: Optional[np.ndarray] = None,
    indices: Optional[np.ndarray] = None,
    output: Optional[np.ndarray] = None
) -> Union[np.ndarray, Tuple[np.ndarray, ...], None]:
    """
    Memory-Efficient Euclidean Distance Transform.

    This is a modified version of `scipy.ndimage.distance_transform_edt` that
    accepts an `output` parameter. This allows writing the result directly into
    a pre-allocated array (e.g., a numpy.memmap), preventing massive RAM spikes
    associated with allocating a new return array for large 3D volumes.

    Args:
        input: Input background/foreground array.
        sampling: Spacing of elements along each dimension.
        return_distances: Whether to calculate distance.
        return_indices: Whether to calculate indices.
        distances: Output array for distances (deprecated in scipy, use output).
        indices: Output array for indices.
        output: Pre-allocated array (e.g., memmap) to write distances into.

    Returns:
        Distance array, indices array, or tuple depending on flags.
    """
    ft_inplace = isinstance(indices, np.ndarray)
    dt_inplace = isinstance(distances, np.ndarray)

    _distance_tranform_arg_check(
        dt_inplace, ft_inplace, return_distances, return_indices
    )

    # Convert input to binary (int8 is sufficient and saves memory)
    input_arr = np.atleast_1d(np.where(input, 1, 0).astype(np.int8))

    if sampling is not None:
        sampling = _ni_support._normalize_sequence(sampling, input_arr.ndim)
        sampling = np.asarray(sampling, dtype=np.float64)
        if not sampling.flags.contiguous:
            sampling = sampling.copy()

    # Feature Transform (FT) allocation
    # FT is an intermediate int32 array required by the algorithm
    if ft_inplace:
        ft = indices
        if ft.shape != (input_arr.ndim,) + input_arr.shape:
            raise RuntimeError('indices array has wrong shape')
        if ft.dtype.type != np.int32:
            raise RuntimeError('indices array must be int32')
    else:
        ft = np.zeros((input_arr.ndim,) + input_arr.shape, dtype=np.int32)

    # Execute Feature Transform (C-level)
    _nd_image.euclidean_feature_transform(input_arr, sampling, ft)

    # Distance Calculation
    if return_distances:
        # Optimization: Use the 'output' buffer if provided
        if output is not None:
            dt = output
            # Validate shape/dtype roughly matches expectations
            # (Strict validation skipped to allow flexibility)
            np.subtract(ft, np.indices(input_arr.shape, dtype=ft.dtype), out=dt)
        else:
            dt = ft - np.indices(input_arr.shape, dtype=ft.dtype)
            dt = dt.astype(np.float64)

        # Apply sampling weights
        if sampling is not None:
            for ii in range(len(sampling)):
                dt[ii, ...] *= sampling[ii]

        # Square components
        np.multiply(dt, dt, dt)

        # Sum and Sqrt
        if dt_inplace:
            reduced_dt = np.add.reduce(dt, axis=0)
            np.sqrt(reduced_dt, distances)
        else:
            dt = np.add.reduce(dt, axis=0)
            dt = np.sqrt(dt)

        del ft # Release the large intermediate int32 array

    # Construct Return
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