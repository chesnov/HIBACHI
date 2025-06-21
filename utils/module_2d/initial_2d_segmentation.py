# --- START OF FILE utils/ramified_module_2d/initial_2d_segmentation.py ---

import numpy as np
from scipy import ndimage
# Make sure all necessary ndimage functions are imported
from scipy.ndimage import (gaussian_filter, generate_binary_structure, label as ndimage_label)
from skimage.filters import frangi, sato # type: ignore
from skimage.morphology import remove_small_objects, disk # type: ignore
import tempfile
import os
from tqdm import tqdm # Keep for potential batching or long single steps
import time
import psutil
from shutil import rmtree
import gc
import os
import tempfile
import gc
import numpy as np
from tqdm import tqdm
import math
seed = 42
np.random.seed(seed)         # For NumPy
import traceback

# Import 2D artifact removal functions (will be created later)
# Assuming the file will be named remove_artifacts_2d.py
from .remove_artifacts_2d import generate_hull_boundary_and_stack_2d, trim_object_edges_by_distance_2d

# Helper function for memory mapping (can remain the same, useful for consistency)
def create_memmap(data=None, dtype=None, shape=None, prefix='temp_2d', directory=None):
    """Helper function to create memory-mapped arrays for 2D data"""
    if directory is None:
        directory = tempfile.mkdtemp()
    path = os.path.join(directory, f'{prefix}.dat')
    if data is not None:
        shape = data.shape; dtype = data.dtype
        result = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
        result[:] = data[:] # Direct assignment for 2D is usually fine
        result.flush()
    else:
        result = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    return result, path, directory

# Removed: _process_slice_worker (Not needed for direct 2D processing)

def enhance_tubular_structures_2d(image, scales, spacing, black_ridges=False,
                                   frangi_alpha=0.5, frangi_beta=0.5, frangi_gamma=15,
                                   apply_smoothing=True, smoothing_sigma_phys=0.5):
    """
    Enhance tubular structures directly on a 2D image. Returns a MEMMAP object.

    Args:
        image (np.ndarray): Input 2D image.
        scales (list): List of physical scales (e.g., in microns).
        spacing (tuple): Pixel spacing. Can be 2D (y, x) or 3D [Z, Y, X] format;
                         only Y and X components will be used.
        black_ridges (bool): Passed to Frangi/Sato.
        frangi_alpha, frangi_beta, frangi_gamma: Frangi parameters.
        apply_smoothing (bool): Whether to apply initial Gaussian smoothing.
        smoothing_sigma_phys (float): Sigma for Gaussian smoothing in physical units.

    Returns:
        tuple: (output_memmap, output_path, output_temp_dir)
               Returns (None, None, None) on failure.
    """
    print(f"Starting 2D tubular enhancement...")
    if image.ndim != 2:
        print(f"Error: Input image must be 2D. Got shape {image.shape}")
        return None, None, None
    print(f"  Image shape: {image.shape}, Input spacing format: {spacing}")
    print(f"  Initial memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # --- Extract YX spacing correctly ---
    try:
        if len(spacing) == 3: # Assume [Z, Y, X] format from GUI Manager/Strategy
            spacing_yx = tuple(float(s) for s in spacing[1:]) # Extract Y, X
            print(f"  Extracted YX spacing for 2D calculations: {spacing_yx}")
        elif len(spacing) == 2: # Assume [Y, X] format
            spacing_yx = tuple(float(s) for s in spacing)
            print(f"  Using provided 2D spacing (y,x): {spacing_yx}")
        else:
            raise ValueError(f"Unexpected spacing format length: {len(spacing)}")
        # Validate spacing values
        if not all(s > 1e-9 for s in spacing_yx):
             raise ValueError(f"Spacing values must be positive: {spacing_yx}")
    except (ValueError, TypeError, IndexError) as e:
        print(f"Warning: Invalid spacing provided ({spacing}). Error: {e}. Assuming isotropic (1,1).")
        spacing_yx = (1.0, 1.0)
    # --- End spacing extraction ---

    image_shape = image.shape
    temp_dirs_to_clean = []
    input_image_processed = image # Start with the original image reference
    smoothing_applied = False

    # Initial Smoothing
    if apply_smoothing and smoothing_sigma_phys > 0:
        print(f"  Applying initial 2D smoothing (sigma_phys={smoothing_sigma_phys})...")
        try:
            # Calculate 2D sigma using the extracted spacing_yx
            sigma_voxel_2d_smooth = tuple(smoothing_sigma_phys / s for s in spacing_yx)
            print(f"    Calculated smoothing sigma (pixels): {sigma_voxel_2d_smooth}")

            # Ensure float32 for filtering
            img_for_smooth = input_image_processed.astype(np.float32, copy=False)

            # Apply filter with the correct 2D sigma
            smoothed_image = gaussian_filter(img_for_smooth, sigma=sigma_voxel_2d_smooth, mode='reflect')

            # Update the reference to the processed image
            input_image_processed = smoothed_image
            smoothing_applied = True # Mark that smoothing created a new array potentially
            print(f"  2D smoothing done.")
            # Don't delete img_for_smooth here if it might be the same object as input_image_processed
            # Let garbage collection handle it after input_image_processed is updated
            del smoothed_image # Delete intermediate array handle
            gc.collect()
        except Exception as e_smooth:
            print(f"  Error during smoothing: {e_smooth}. Proceeding without smoothing.")
            # Ensure input_image_processed is float32 for filters even if smoothing failed
            if input_image_processed is image: # Check if it's still the original
                 input_image_processed = input_image_processed.astype(np.float32, copy=False)
            elif not np.issubdtype(input_image_processed.dtype, np.floating):
                 input_image_processed = input_image_processed.astype(np.float32, copy=False)


    else:
        print("  Skipping initial 2D smoothing.")
        # Ensure float32 for filters later if no smoothing applied
        if not np.issubdtype(input_image_processed.dtype, np.floating):
             input_image_processed = input_image_processed.astype(np.float32, copy=False)


    # Calculate filter scales using YX spacing
    avg_xy_spacing = np.mean(spacing_yx)
    sigmas_voxel_2d_filter = sorted([s / avg_xy_spacing for s in scales if s > 0]) # Ensure positive scales
    if not sigmas_voxel_2d_filter:
        print("Error: No valid positive scales provided for filtering.")
        return None, None, None
    print(f"  Using 2D voxel sigmas for Frangi/Sato: {sigmas_voxel_2d_filter}")

    # Apply Frangi and Sato filters
    print("  Applying Frangi and Sato filters...")
    start_time = time.time()
    enhanced_result = None
    try:
        # Ensure input is float32, might be redundant but safe
        if input_image_processed.dtype != np.float32:
             img_for_filter = input_image_processed.astype(np.float32)
             # If smoothing was applied, input_image_processed might be the smoothed result.
             # If smoothing was skipped or failed, this might be the original converted.
             # We don't need the original reference 'image' anymore if smoothing happened.
             if smoothing_applied and input_image_processed is not image:
                 del input_image_processed # Safe to delete the intermediate smoothed ref
             input_image_processed = img_for_filter # Update main ref to the float32 version
        else:
            img_for_filter = input_image_processed # Already float32, use directly

        frangi_result_2d = frangi(img_for_filter, sigmas=sigmas_voxel_2d_filter, alpha=frangi_alpha,
                                  beta=frangi_beta, gamma=frangi_gamma, black_ridges=black_ridges,
                                  mode='reflect')
        sato_result_2d = sato(img_for_filter, sigmas=sigmas_voxel_2d_filter, black_ridges=black_ridges,
                              mode='reflect')
        enhanced_result = np.maximum(frangi_result_2d, sato_result_2d)

        # Cleanup intermediate results
        del frangi_result_2d, sato_result_2d
        # If img_for_filter was a temporary copy (due to dtype conversion), delete it.
        # If it was just a reference to input_image_processed, don't delete yet.
        if img_for_filter is not input_image_processed:
             del img_for_filter
        gc.collect()
        print(f"  Filtering finished in {time.time() - start_time:.2f} seconds.")
    except MemoryError as mem_err:
        print(f"An error occurred during 2D filtering (MemoryError): {mem_err}")
        traceback.print_exc()
        # Cleanup potentially large arrays on error
        del enhanced_result
        if 'img_for_filter' in locals(): del img_for_filter
        if 'input_image_processed' in locals(): del input_image_processed
        gc.collect()
        return None, None, None # Indicate failure
    except Exception as e:
        print(f"An error occurred during 2D filtering: {e}")
        traceback.print_exc()
        # Cleanup
        del enhanced_result
        if 'img_for_filter' in locals(): del img_for_filter
        if 'input_image_processed' in locals(): del input_image_processed
        gc.collect()
        return None, None, None # Indicate failure


    # Create output memmap
    output_temp_dir = None
    output_path = None
    output_memmap = None
    try:
        output_temp_dir = tempfile.mkdtemp(prefix='tubular_enhance_2d_')
        temp_dirs_to_clean.append(output_temp_dir) # Track for potential cleanup
        output_path = os.path.join(output_temp_dir, 'enhanced_image_2d.dat')
        output_dtype = np.float32
        # Create memmap from the enhanced_result numpy array
        output_memmap, _, _ = create_memmap(data=enhanced_result, dtype=output_dtype, shape=image_shape, directory=output_temp_dir)
        print(f"  Output memmap created: {output_path}")
    except Exception as e_memmap:
        print(f"Error creating output memmap: {e_memmap}")
        if output_temp_dir and os.path.exists(output_temp_dir):
            rmtree(output_temp_dir, ignore_errors=True)
        del enhanced_result # Cleanup result array
        if 'input_image_processed' in locals(): del input_image_processed
        gc.collect()
        return None, None, None

    # Delete in-memory version of enhanced result after saving to memmap
    del enhanced_result
    # Delete the potentially modified input_image_processed (e.g., smoothed or float32 version)
    # unless it was just a reference to the original input 'image'
    if input_image_processed is not image:
        del input_image_processed
    gc.collect()


    print(f"Enhanced 2D image generated as memmap: {output_path}")
    print(f"Memory usage after 2D enhancement: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # Return the memmap object itself, its path, and the directory containing it
    return output_memmap, output_path, output_temp_dir


# Replaced connect_fragmented_processes with a 2D version
def connect_fragmented_processes_2d(binary_mask, spacing, max_gap_physical=1.0):
    """Connect fragmented processes in 2D using isotropic morphological closing."""
    print(f"Connecting fragmented processes (2D) with max physical gap: {max_gap_physical}")
    print(f"  Input mask shape: {binary_mask.shape}, Spacing: {spacing}")

    # Calculate radius for isotropic closing
    avg_spacing = np.mean(spacing)
    radius_pix = math.ceil((max_gap_physical / 2) / avg_spacing)
    # Use skimage.morphology.disk for a circular structure
    # Structure size needs to be diameter + 1 if using ndimage,
    # but disk radius is directly usable here. Let's use disk.
    structure = disk(radius_pix)
    print(f"  Calculated isotropic structure radius (pixels): {radius_pix}")
    print(f"  Using structure shape: {structure.shape}")


    temp_dir = tempfile.mkdtemp(prefix="connect_frag_2d_")
    result_path = os.path.join(temp_dir, 'connected_mask_2d.dat')
    connected_mask = np.memmap(result_path, dtype=np.bool_, mode='w+', shape=binary_mask.shape)
    print("  Applying isotropic binary closing...")
    start_time = time.time()
    try:
        # binary_closing works directly with the mask and structure
        ndimage.binary_closing(binary_mask, structure=structure, output=connected_mask, border_value=0)
        connected_mask.flush()
        print(f"  Binary closing completed in {time.time() - start_time:.2f} seconds.")
    except MemoryError:
        print("  MemoryError during binary closing.");
        del connected_mask; gc.collect(); rmtree(temp_dir);
        raise MemoryError("Failed binary closing.") from None
    except Exception as e:
        print(f"  Error during binary closing: {e}");
        del connected_mask; gc.collect(); rmtree(temp_dir);
        raise e
    # Return the memmap handle and its directory
    return connected_mask, temp_dir


def segment_cells_first_pass_raw_2d(
    image, # Original 2D intensity image
    spacing, # 2D pixel spacing (y, x) or 3D [Z,Y,X]
    tubular_scales=[0.5, 1.0, 2.0, 3.0],
    smooth_sigma=0.5,
    connect_max_gap_physical=1.0,
    min_size_pixels=50, # Changed from voxels
    low_threshold_percentile=25.0,
    high_threshold_percentile=95.0
    ):
    """
    Performs initial 2D segmentation steps 1-6 (Enhance, Normalize, Threshold,
    Connect, Clean, Label). Uses memmaps. Incorporates robust cleanup.

    Returns:
    --------
    labels_path : str or None
        Path to the final 2D labeled segmentation memmap file, or None on failure.
    labels_temp_dir : str or None
        Path to the temporary directory containing the labels memmap, or None on failure.
    segmentation_threshold : float
        The calculated absolute threshold used for segmentation.
    first_pass_params : dict
        Dictionary containing parameters used.
    """
    # --- Initial Setup ---
    if image.ndim != 2:
        print(f"Error: Input image must be 2D. Got shape {image.shape}")
        return None, None, 0.0, {}

    # Extract 2D spacing if 3D is provided
    try:
        if len(spacing) == 3: spacing_2d = tuple(float(s) for s in spacing[1:])
        elif len(spacing) == 2: spacing_2d = tuple(float(s) for s in spacing)
        else: raise ValueError("Invalid spacing dimensions")
        if not all(s > 1e-9 for s in spacing_2d): raise ValueError("Spacing must be positive")
    except (ValueError, TypeError, IndexError) as e:
        print(f"Warning: Invalid spacing {spacing}. Error: {e}. Assuming isotropic (1,1).")
        spacing_2d = (1.0, 1.0)

    low_threshold_percentile = max(0.0, min(100.0, low_threshold_percentile))
    high_threshold_percentile = max(0.0, min(100.0, high_threshold_percentile))

    original_shape = image.shape # Shape of the 2D image
    # Store only params used in this function
    first_pass_params = {
        'spacing': spacing_2d, # Store the 2D spacing used
        'tubular_scales': tubular_scales, 'smooth_sigma': smooth_sigma,
        'connect_max_gap_physical': connect_max_gap_physical, 'min_size_pixels': min_size_pixels,
        'low_threshold_percentile': low_threshold_percentile,
        'high_threshold_percentile': high_threshold_percentile,
        'original_shape': original_shape
        }
    print(f"\n--- Starting Raw First Pass 2D Segmentation ---")
    initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_mem:.2f} MB")

    memmap_registry = {} # To track intermediate memmaps: {name: (handle, path, dir)}
    segmentation_threshold = 0.0     # Default if calculation fails
    labels_path = None
    labels_temp_dir = None

    # Function for robust cleanup
    def _cleanup_memmap(registry, name):
        if name in registry:
            mm, p, d = registry.pop(name)
            print(f"    Cleaning up '{name}': Path='{p}', Dir='{d}'")
            if mm is not None:
                # Check if it's a memmap instance and has the _mmap attribute
                if isinstance(mm, np.memmap) and hasattr(mm, '_mmap') and mm._mmap is not None:
                    try: mm._mmap.close()
                    except Exception as e_close: print(f"      Warn: Error closing memmap handle: {e_close}")
                print(f"      Deleting handle for '{name}' memmap...")
                del mm; gc.collect(); time.sleep(0.1)
            if p and os.path.exists(p):
                print(f"      Unlinking file: {p}")
                try: os.unlink(p)
                except Exception as e_unlink: print(f"      Warn: Failed to unlink file {p}: {e_unlink}")
            elif p: print(f"      File already gone before unlink: {p}")
            if d and os.path.isdir(d):
                print(f"      Removing directory: {d}")
                try: rmtree(d, ignore_errors=True)
                except Exception as e_rmtree: print(f"      Warn: Failed to remove directory {d}: {e_rmtree}")
            elif d: print(f"      Directory already gone/invalid: {d}")
        else:
             print(f"    Warn: '{name}' not found in memmap registry for cleanup.")

    try:
        # --- Step 1: Enhance ---
        print("\nStep 1: Enhancing structures (2D)...")
        enhanced_memmap, enhance_path, enhance_temp_dir = enhance_tubular_structures_2d(
            image, scales=tubular_scales, spacing=spacing_2d, # Pass 2D spacing
            apply_smoothing=(smooth_sigma > 0), smoothing_sigma_phys=smooth_sigma
        )
        if enhanced_memmap is None: # Check if enhancement failed
             raise RuntimeError("Tubular enhancement step failed.")
        memmap_registry['enhanced'] = (enhanced_memmap, enhance_path, enhance_temp_dir)
        print("2D Enhancement memmap created.")
        gc.collect()

        # --- Step 2: Normalize ---
        print("\nStep 2: Normalizing signal (2D Global)...")
        normalized_temp_dir = tempfile.mkdtemp(prefix="normalize_2d_")
        normalized_path = os.path.join(normalized_temp_dir, 'normalized_2d.dat')
        normalized_memmap = np.memmap(normalized_path, dtype=np.float32, mode='w+', shape=enhanced_memmap.shape)
        memmap_registry['normalized'] = (normalized_memmap, normalized_path, normalized_temp_dir)

        # Calculate global high percentile from the enhanced image
        target_intensity = 1.0
        print("  Calculating global normalization factor...")
        enhanced_flat = enhanced_memmap[:].ravel() # Read whole 2D memmap
        finite_enhanced = enhanced_flat[np.isfinite(enhanced_flat)]
        high_perc_value = 0.0
        if finite_enhanced.size > 0:
             sample_size_norm = min(2_000_000, finite_enhanced.size)
             samples = np.random.choice(finite_enhanced, sample_size_norm, replace=False)
             try:
                 high_perc_value = np.percentile(samples, high_threshold_percentile)
                 if not np.isfinite(high_perc_value): high_perc_value = 0.0
             except Exception as e:
                  print(f"  Warn: Percentile calculation failed: {e}. Using max.")
                  high_perc_value = np.max(samples) if samples.size > 0 else 0.0
             del samples
        del enhanced_flat, finite_enhanced; gc.collect()

        print(f"  Global High Percentile ({high_threshold_percentile:.1f}%) value: {high_perc_value:.4f}")

        # Apply normalization globally
        if high_perc_value > 1e-6:
            scale_factor = target_intensity / high_perc_value
            enhanced_data = enhanced_memmap[:]
            normalized_memmap[:] = (enhanced_data * scale_factor).astype(np.float32)
            del enhanced_data
        else:
            print("  Warn: High percentile value too low. Copying enhanced data without scaling.")
            normalized_memmap[:] = enhanced_memmap[:]

        normalized_memmap.flush()
        print("Normalization finished.")
        # Cleanup enhanced
        _cleanup_memmap(memmap_registry, 'enhanced')


        # --- Step 3: Thresholding & Calc Thresholds ---
        print("\nStep 3: Thresholding and Calculating Thresholds (2D Global)...")
        binary_temp_dir = tempfile.mkdtemp(prefix="binary_2d_")
        binary_path = os.path.join(binary_temp_dir, 'b_2d.dat')
        binary_memmap = np.memmap(binary_path, dtype=bool, mode='w+', shape=original_shape)
        memmap_registry['binary'] = (binary_memmap, binary_path, binary_temp_dir)

        # Sample globally from normalized image
        print("  Sampling normalized image for threshold...")
        normalized_flat = normalized_memmap[:].ravel()
        finite_normalized = normalized_flat[np.isfinite(normalized_flat)]
        segmentation_threshold = 0.0
        if finite_normalized.size > 0:
             sample_size_thresh = min(2_000_000, finite_normalized.size)
             samples = np.random.choice(finite_normalized, sample_size_thresh, replace=False)
             try:
                 segmentation_threshold = float(np.percentile(samples, low_threshold_percentile)) # Ensure float
                 if not np.isfinite(segmentation_threshold): segmentation_threshold = 0.0
             except Exception as e:
                 print(f" Warn: Percentile failed: {e}. Using median.")
                 segmentation_threshold = float(np.median(samples)) if samples.size > 0 else 0.0
             del samples
        else:
            print(" Warn: No finite samples found in normalized image for threshold calculation.")
        del normalized_flat, finite_normalized; gc.collect()

        print(f"  Segmentation threshold ({low_threshold_percentile:.1f} perc): {segmentation_threshold:.4f}")

        # Apply threshold globally
        norm_data = normalized_memmap[:]
        binary_memmap[:] = norm_data > segmentation_threshold
        binary_memmap.flush()
        print("Thresholding done.")
        del norm_data; gc.collect()
        # Cleanup normalized
        _cleanup_memmap(memmap_registry, 'normalized')


        # --- Step 4: Connect ---
        print("\nStep 4: Connecting fragments (2D)...")
        connected_binary_memmap, connected_temp_dir = connect_fragmented_processes_2d(
            binary_memmap, spacing=spacing_2d, max_gap_physical=connect_max_gap_physical # Use 2D spacing
        )
        if connected_binary_memmap is None: # Check if connection failed
            raise RuntimeError("Fragment connection step failed.")
        memmap_registry['connected'] = (connected_binary_memmap, connected_binary_memmap.filename, connected_temp_dir)
        print("2D Connection memmap created.")
        # Cleanup binary
        _cleanup_memmap(memmap_registry, 'binary')


        # --- Step 5: Clean Small Objects ---
        print(f"\nStep 5: Cleaning objects < {min_size_pixels} pixels...")
        cleaned_binary_dir=tempfile.mkdtemp(prefix="cleaned_2d_")
        cleaned_binary_path=os.path.join(cleaned_binary_dir, 'c_2d.dat')
        cleaned_binary_memmap=np.memmap(cleaned_binary_path, dtype=bool, mode='w+', shape=original_shape)
        memmap_registry['cleaned'] = (cleaned_binary_memmap, cleaned_binary_path, cleaned_binary_dir)

        connected_memmap_obj = memmap_registry['connected'][0]
        connected_in_memory = np.array(connected_memmap_obj) # Load 2D mask into memory
        remove_small_objects(connected_in_memory, min_size=min_size_pixels, connectivity=1, out=cleaned_binary_memmap)
        del connected_in_memory; gc.collect()

        cleaned_binary_memmap.flush()
        print("Cleaning step done.")
        # Cleanup connected
        _cleanup_memmap(memmap_registry, 'connected')


        # --- Step 6: Label ---
        print("\nStep 6: Labeling components (2D)...")
        labels_temp_dir = tempfile.mkdtemp(prefix="labels_2d_")
        labels_path = os.path.join(labels_temp_dir, 'l_2d.dat') # This path will be returned
        # Use int32 for labels, consistent with 3D
        first_pass_segmentation_memmap=np.memmap(labels_path, dtype=np.int32, mode='w+', shape=original_shape)
        # Don't add labels to registry here, we return its path/dir

        cleaned_memmap_obj = memmap_registry['cleaned'][0]
        cleaned_mask_in_memory = np.array(cleaned_memmap_obj) # Load 2D mask
        # Use 2D structure (connectivity=1 -> 4-neighbors, connectivity=2 -> 8-neighbors)
        structure_2d = generate_binary_structure(2, 1) # 4-connectivity
        num_features = ndimage_label(cleaned_mask_in_memory, structure=structure_2d, output=first_pass_segmentation_memmap)
        del cleaned_mask_in_memory; gc.collect()

        first_pass_segmentation_memmap.flush()
        print(f"Labeling done ({num_features} features found). Output memmap: {labels_path}")
        # IMPORTANT: Close the labels memmap handle here so the file can be used potentially
        if hasattr(first_pass_segmentation_memmap, '_mmap') and first_pass_segmentation_memmap._mmap is not None:
             first_pass_segmentation_memmap._mmap.close()
        del first_pass_segmentation_memmap; gc.collect()
        # Cleanup cleaned
        _cleanup_memmap(memmap_registry, 'cleaned')

        # --- Finish ---
        print(f"\n--- Raw First Pass 2D Segmentation Finished ---")
        final_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"Final memory usage: {final_mem:.2f} MB")

        # Return path to labeled memmap and calculated threshold
        return labels_path, labels_temp_dir, segmentation_threshold, first_pass_params

    except Exception as e:
        print(f"\n!!! ERROR during Raw First Pass 2D Segmentation: {e} !!!")
        traceback.print_exc()
        # Ensure cleanup is attempted on error
        # We raise e below, but the finally block will execute first
        raise e # Re-raise the error
    finally:
        # Final cleanup of any remaining tracked memmaps
        print("\nFinal cleanup check for 2D raw pass...")
        registry_keys = list(memmap_registry.keys())
        for name in registry_keys:
             _cleanup_memmap(memmap_registry, name)
        gc.collect()

# --- END OF FILE utils/ramified_module_2d/initial_2d_segmentation.py ---