import os
import gc
import math
import time
import shutil
import traceback
import tempfile
import warnings
from typing import Tuple, List, Dict, Any, Optional, Union

import numpy as np
import psutil
from scipy import ndimage
from scipy.ndimage import gaussian_filter, generate_binary_structure, label as ndimage_label
from skimage.filters import frangi, sato  # type: ignore
from skimage.morphology import remove_small_objects, disk  # type: ignore

# Set seed for reproducibility
SEED = 42
np.random.seed(SEED)


def create_memmap(
    data: Optional[np.ndarray] = None,
    dtype: Optional[Any] = None,
    shape: Optional[Tuple[int, ...]] = None,
    prefix: str = 'temp_2d',
    directory: Optional[str] = None
) -> Tuple[np.memmap, str, str]:
    """
    Creates a memory-mapped array for efficient 2D data handling.

    Args:
        data: Optional data to write immediately.
        dtype: Data type of the array.
        shape: Shape of the array.
        prefix: Filename prefix.
        directory: Directory for the temp file. Created if None.

    Returns:
        Tuple[np.memmap, str, str]: (memmap_obj, file_path, directory_path)
    """
    if directory is None:
        directory = tempfile.mkdtemp(prefix=f"{prefix}_dir_")
    
    path = os.path.join(directory, f'{prefix}.dat')
    
    if data is not None:
        shape = data.shape
        dtype = data.dtype
        result = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
        result[:] = data[:]
        result.flush()
    else:
        if shape is None or dtype is None:
            raise ValueError("Shape and dtype must be provided if data is None")
        result = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    
    return result, path, directory


def enhance_tubular_structures_2d(
    image: np.ndarray,
    scales: List[float],
    spacing: Union[Tuple[float, float], Tuple[float, float, float]],
    black_ridges: bool = False,
    frangi_alpha: float = 0.5,
    frangi_beta: float = 0.5,
    frangi_gamma: float = 15.0,
    apply_smoothing: bool = True,
    smoothing_sigma_phys: float = 0.5,
    skip_enhancement: bool = False
) -> Tuple[Optional[np.memmap], Optional[str], Optional[str]]:
    """
    Enhances tubular structures in a 2D image using Frangi/Sato filters.
    Can skip enhancement (pass-through) if skip_enhancement is True.

    Args:
        image: 2D Input image.
        scales: List of physical scales (um) for vessel detection.
        spacing: Pixel spacing (y, x) or (z, y, x).
        black_ridges: If True, detect black ridges on white.
        frangi_alpha, frangi_beta, frangi_gamma: Filter parameters.
        apply_smoothing: Whether to apply pre-filter Gaussian smoothing.
        smoothing_sigma_phys: Smoothing sigma in microns.
        skip_enhancement: If True, returns smoothed input without filtering.

    Returns:
        Tuple: (output_memmap, output_path, output_temp_dir) or (None, None, None) on error.
    """
    print(f"Starting 2D tubular enhancement...")
    if image.ndim != 2:
        print(f"Error: Input image must be 2D. Got shape {image.shape}")
        return None, None, None

    # --- Extract Spacing ---
    try:
        if len(spacing) == 3:
            spacing_yx = tuple(float(s) for s in spacing[1:])
        elif len(spacing) == 2:
            spacing_yx = tuple(float(s) for s in spacing)
        else:
            raise ValueError(f"Invalid spacing length: {len(spacing)}")
        
        if not all(s > 1e-9 for s in spacing_yx):
            raise ValueError("Spacing must be positive")
    except Exception as e:
        print(f"Warning: Invalid spacing {spacing}. Using (1.0, 1.0). Error: {e}")
        spacing_yx = (1.0, 1.0)

    # --- Check Scales vs Smoothing (Only if not skipping) ---
    if not skip_enhancement:
        min_scale = min(scales) if scales else 0
        if apply_smoothing and smoothing_sigma_phys >= min_scale:
            print(f"  WARNING: Smoothing sigma ({smoothing_sigma_phys} um) >= smallest scale ({min_scale} um).")
            print(f"           Fine structures may be washed out. Consider reducing smooth_sigma.")

    # --- Pre-Smoothing ---
    input_image_processed = image
    smoothing_applied = False
    
    if apply_smoothing and smoothing_sigma_phys > 0:
        print(f"  Applying initial 2D smoothing (sigma={smoothing_sigma_phys} um)...")
        try:
            sigma_pixels = tuple(smoothing_sigma_phys / s for s in spacing_yx)
            img_float = input_image_processed.astype(np.float32, copy=False)
            smoothed_image = gaussian_filter(img_float, sigma=sigma_pixels, mode='reflect')
            input_image_processed = smoothed_image
            smoothing_applied = True
        except Exception as e:
            print(f"  Error during smoothing: {e}. Skipping.")
    
    # Ensure float32
    if input_image_processed.dtype != np.float32:
        input_image_processed = input_image_processed.astype(np.float32)

    # --- Skip Logic (Pass-Through) ---
    if skip_enhancement:
        print("  Skipping filter calculation (Pass-through).")
        try:
            out_memmap, out_path, out_dir = create_memmap(
                data=input_image_processed, dtype=np.float32, prefix='enhanced_2d'
            )
            if smoothing_applied:
                del input_image_processed
            gc.collect()
            return out_memmap, out_path, out_dir
        except Exception as e:
            print(f"Error saving pass-through memmap: {e}")
            return None, None, None

    # --- Filter Calculation ---
    avg_spacing = np.mean(spacing_yx)
    sigmas_pixels = sorted([s / avg_spacing for s in scales if s > 0])
    
    if not sigmas_pixels:
        print("Error: No valid positive scales provided.")
        return None, None, None

    print(f"  Applying Frangi/Sato filters (sigmas={sigmas_pixels})...")
    enhanced_result = None
    
    try:
        start_time = time.time()
        
        frangi_res = frangi(
            input_image_processed, sigmas=sigmas_pixels, alpha=frangi_alpha,
            beta=frangi_beta, gamma=frangi_gamma, black_ridges=black_ridges, mode='reflect'
        )
        sato_res = sato(
            input_image_processed, sigmas=sigmas_pixels, black_ridges=black_ridges, mode='reflect'
        )
        
        enhanced_result = np.maximum(frangi_res, sato_res)
        
        del frangi_res, sato_res
        if smoothing_applied:
            del input_image_processed
        gc.collect()
        
        print(f"  Filtering finished in {time.time() - start_time:.2f}s.")
        
    except Exception as e:
        print(f"Error during 2D filtering: {e}")
        traceback.print_exc()
        return None, None, None

    # --- Save to Memmap ---
    try:
        out_memmap, out_path, out_dir = create_memmap(
            data=enhanced_result, dtype=np.float32, prefix='enhanced_2d'
        )
        del enhanced_result
        gc.collect()
        return out_memmap, out_path, out_dir
    except Exception as e:
        print(f"Error creating output memmap: {e}")
        return None, None, None


def connect_fragmented_processes_2d(
    binary_mask: np.ndarray,
    spacing: Tuple[float, float],
    max_gap_physical: float = 1.0
) -> Tuple[np.memmap, str]:
    """
    Connects fragmented 2D binary structures using morphological closing.

    Args:
        binary_mask: Input boolean mask.
        spacing: Pixel spacing (y, x).
        max_gap_physical: Maximum gap to bridge (um).

    Returns:
        Tuple: (connected_memmap, temp_dir)
    """
    print(f"Connecting fragments (2D) max gap: {max_gap_physical} um...")
    
    avg_spacing = np.mean(spacing)
    radius_pix = math.ceil((max_gap_physical / 2) / avg_spacing)
    structure = disk(radius_pix)
    
    print(f"  Closing radius: {radius_pix} pixels")
    
    temp_dir = tempfile.mkdtemp(prefix="connect_2d_")
    out_path = os.path.join(temp_dir, 'connected.dat')
    connected_mask = np.memmap(out_path, dtype=bool, mode='w+', shape=binary_mask.shape)
    
    try:
        ndimage.binary_closing(binary_mask, structure=structure, output=connected_mask)
        connected_mask.flush()
    except Exception as e:
        print(f"Error during binary closing: {e}")
        del connected_mask
        shutil.rmtree(temp_dir)
        raise e

    return connected_mask, temp_dir


def segment_cells_first_pass_raw_2d(
    image: np.ndarray,
    spacing: Union[Tuple[float, float], Tuple[float, float, float]],
    tubular_scales: List[float] = [0.5, 1.0, 2.0, 3.0],
    smooth_sigma: float = 0.5,
    connect_max_gap_physical: float = 1.0,
    min_size_pixels: int = 50,
    low_threshold_percentile: float = 95.0,
    high_threshold_percentile: float = 100.0
) -> Tuple[Optional[str], Optional[str], float, Dict[str, Any]]:
    """
    Executes the Raw 2D Segmentation Pipeline:
    Enhance -> Normalize -> Threshold -> Connect -> Clean -> Label.

    Args:
        image: 2D Input array.
        spacing: Pixel dimensions.
        tubular_scales: Scales for Frangi filter. If [0.0], enhancement is skipped.
        smooth_sigma: Pre-smoothing sigma (um).
        connect_max_gap_physical: Gap closing distance (um).
        min_size_pixels: Minimum object size to retain.
        low_threshold_percentile: Percentile for binary thresholding.
        high_threshold_percentile: Percentile for normalization max.

    Returns:
        Tuple: (labels_path, labels_temp_dir, calculated_threshold, params_dict)
    """
    if image.ndim != 2:
        print(f"Error: Image must be 2D. Got {image.shape}")
        return None, None, 0.0, {}

    # Extract spacing
    try:
        if len(spacing) == 3:
            spacing_2d = tuple(float(s) for s in spacing[1:])
        else:
            spacing_2d = tuple(float(s) for s in spacing)
    except:
        spacing_2d = (1.0, 1.0)

    params_record = {
        'spacing': spacing_2d,
        'tubular_scales': tubular_scales,
        'smooth_sigma': smooth_sigma,
        'connect_max_gap_physical': connect_max_gap_physical,
        'min_size_pixels': min_size_pixels,
        'low_threshold_percentile': low_threshold_percentile,
        'high_threshold_percentile': high_threshold_percentile
    }

    # Detect Skip Condition (0.0 implies disable)
    skip_enhancement = False
    if len(tubular_scales) == 1 and abs(tubular_scales[0]) < 1e-9:
        skip_enhancement = True
    elif not tubular_scales:
        skip_enhancement = True

    print("\n--- Starting Raw 2D Segmentation ---")
    memmap_registry = {}  # Keep track of intermediates to clean up
    
    def cleanup_registry():
        for key in list(memmap_registry.keys()):
            cleanup_registry_item(memmap_registry, key)

    segmentation_threshold = 0.0
    labels_path = None
    labels_temp_dir = None

    try:
        # --- 1. Enhancement ---
        enhanced_mm, enh_path, enh_dir = enhance_tubular_structures_2d(
            image, scales=tubular_scales, spacing=spacing_2d,
            apply_smoothing=(smooth_sigma > 0), smoothing_sigma_phys=smooth_sigma,
            skip_enhancement=skip_enhancement
        )
        if enhanced_mm is None:
            raise RuntimeError("Enhancement failed.")
        memmap_registry['enhanced'] = (enhanced_mm, enh_path, enh_dir)

        # --- 2. Normalization ---
        print("\nStep 2: Normalizing...")
        norm_dir = tempfile.mkdtemp(prefix="norm_2d_")
        norm_path = os.path.join(norm_dir, 'norm.dat')
        norm_mm = np.memmap(norm_path, dtype=np.float32, mode='w+', shape=image.shape)
        memmap_registry['normalized'] = (norm_mm, norm_path, norm_dir)

        # Sample for high percentile
        valid_pixels = enhanced_mm[np.isfinite(enhanced_mm)]
        if valid_pixels.size > 0:
            sample = np.random.choice(valid_pixels, min(valid_pixels.size, 500000))
            high_val = np.percentile(sample, high_threshold_percentile)
            if high_val < 1e-9: high_val = 1.0 # Prevent div by zero
        else:
            high_val = 1.0

        norm_mm[:] = enhanced_mm[:] / high_val
        norm_mm.flush()
        
        cleanup_registry_item(memmap_registry, 'enhanced')

        # --- 3. Thresholding ---
        print("\nStep 3: Thresholding...")
        bin_dir = tempfile.mkdtemp(prefix="bin_2d_")
        bin_path = os.path.join(bin_dir, 'bin.dat')
        bin_mm = np.memmap(bin_path, dtype=bool, mode='w+', shape=image.shape)
        memmap_registry['binary'] = (bin_mm, bin_path, bin_dir)

        # Calc threshold
        valid_norm = norm_mm[np.isfinite(norm_mm)]
        if valid_norm.size > 0:
            sample = np.random.choice(valid_norm, min(valid_norm.size, 500000))
            segmentation_threshold = float(np.percentile(sample, low_threshold_percentile))
        else:
            segmentation_threshold = 0.0
        
        print(f"  Threshold ({low_threshold_percentile}%): {segmentation_threshold:.4f}")
        
        bin_mm[:] = norm_mm[:] > segmentation_threshold
        bin_mm.flush()
        
        cleanup_registry_item(memmap_registry, 'normalized')

        # --- 4. Connection ---
        print("\nStep 4: Connecting fragments...")
        conn_mm, conn_dir = connect_fragmented_processes_2d(
            bin_mm, spacing=spacing_2d, max_gap_physical=connect_max_gap_physical
        )
        memmap_registry['connected'] = (conn_mm, conn_mm.filename, conn_dir)
        
        cleanup_registry_item(memmap_registry, 'binary')

        # --- 5. Cleaning ---
        print(f"\nStep 5: Cleaning objects < {min_size_pixels} pixels...")
        clean_dir = tempfile.mkdtemp(prefix="clean_2d_")
        clean_path = os.path.join(clean_dir, 'clean.dat')
        clean_mm = np.memmap(clean_path, dtype=bool, mode='w+', shape=image.shape)
        memmap_registry['cleaned'] = (clean_mm, clean_path, clean_dir)

        # In-memory processing for small object removal (fast for 2D)
        temp_arr = np.array(conn_mm)
        remove_small_objects(temp_arr, min_size=min_size_pixels, connectivity=1, out=clean_mm)
        clean_mm.flush()
        del temp_arr
        
        cleanup_registry_item(memmap_registry, 'connected')

        # --- 6. Labeling ---
        print("\nStep 6: Labeling...")
        labels_temp_dir = tempfile.mkdtemp(prefix="labels_2d_")
        labels_path = os.path.join(labels_temp_dir, 'labels.dat')
        labels_mm = np.memmap(labels_path, dtype=np.int32, mode='w+', shape=image.shape)
        
        temp_arr = np.array(clean_mm)
        structure = generate_binary_structure(2, 1) # 4-connectivity
        
        # Output logic for ndimage.label (returns count only when output is specified)
        num = ndimage_label(temp_arr, structure=structure, output=labels_mm)
        
        labels_mm.flush()
        print(f"  Found {num} objects.")
        
        # We don't delete labels_mm here; we return the path
        if hasattr(labels_mm, '_mmap'): labels_mm._mmap.close()
        del labels_mm
        
        cleanup_registry_item(memmap_registry, 'cleaned')
        gc.collect()

        return labels_path, labels_temp_dir, segmentation_threshold, params_record

    except Exception as e:
        print(f"Error during raw segmentation: {e}")
        traceback.print_exc()
        cleanup_registry()
        return None, None, 0.0, {}


def cleanup_registry_item(registry, key):
    """Helper to clean up a specific item from the registry."""
    if key in registry:
        mm, path, d = registry.pop(key)
        if isinstance(mm, np.memmap) and hasattr(mm, '_mmap'):
            try: mm._mmap.close()
            except: pass
        del mm
        if d and os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)