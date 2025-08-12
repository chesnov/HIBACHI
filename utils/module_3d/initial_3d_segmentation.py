# --- START OF FILE initial_3d_segmentation.py ---

import numpy as np
from scipy import ndimage
# Make sure all necessary ndimage functions are imported
from scipy.ndimage import (generate_binary_structure)
from skimage.filters import frangi, sato # type: ignore
from skimage.morphology import remove_small_objects # type: ignore
import tempfile
import os
from tqdm import tqdm
from multiprocessing import Pool
import time
import psutil
from shutil import rmtree
import gc
from functools import partial
import math
import traceback
import zarr # Ensure zarr is imported

# Dask and dask-image are now essential
import dask.array as da
from dask.diagnostics import ProgressBar
import dask_image.ndmorph
import dask_image.ndmeasure
import dask_image.ndfilters

seed = 42
np.random.seed(seed)         # For NumPy

from .remove_artifacts import * # Assuming this exists and is needed by other parts not shown

# Helper function for memory mapping (remains the same)
def create_memmap(data=None, dtype=None, shape=None, prefix='temp', directory=None):
    """Helper function to create memory-mapped arrays"""
    if directory is None:
        directory = tempfile.mkdtemp()
    path = os.path.join(directory, f'{prefix}.dat')
    if data is not None:
        shape = data.shape; dtype = data.dtype
        result = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
        chunk_size = min(100, shape[0]) if shape[0] > 0 else 1
        for i in range(0, shape[0], chunk_size):
            end = min(i + chunk_size, shape[0])
            result[i:end] = data[i:end]
        result.flush()
    else:
        result = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    return result, path, directory


# +++ START: NEW Dask-based 3D Smoothing Function (Internal Helper) +++
def _smooth_volume_dask_internal(volume, spacing, smoothing_sigma_phys):
    """
    Internal helper to apply 3D Gaussian smoothing using Dask.
    Returns a memmap of the smoothed volume and its temp directory.
    """
    print("Applying initial 3D smoothing using Dask for memory safety...")
    dask_chunk_size = (64, 256, 256)
    dask_volume = da.from_array(volume, chunks=dask_chunk_size)
    
    sigma_voxel_3d = tuple(smoothing_sigma_phys / s if s > 0 else 0 for s in spacing)
    print(f"  Using 3D voxel sigma for smoothing: {sigma_voxel_3d}")

    smoothed_dask_array = dask_image.ndfilters.gaussian_filter(
        dask_volume, sigma=sigma_voxel_3d, mode='reflect'
    ).astype(np.float32)

    temp_dir = tempfile.mkdtemp(prefix="dask_smooth_output_")
    output_path = os.path.join(temp_dir, 'smoothed_volume.dat')
    output_memmap = np.memmap(output_path, dtype=np.float32, mode='w+', shape=volume.shape)

    print("  Executing Dask smoothing and storing to memmap...")
    # Corrected ProgressBar call without the 'label' argument
    with ProgressBar():
        da.store(smoothed_dask_array, output_memmap)
    
    print("  Dask 3D smoothing complete.")
    return output_memmap, temp_dir
# +++ END: NEW Dask-based 3D Smoothing Function +++


def _process_slice_worker(z, input_memmap_info, output_memmap_info,
                          sigmas_voxel_2d, black_ridges,
                          frangi_alpha, frangi_beta, frangi_gamma):
    """Worker function to process a single slice. (Unchanged)"""
    try:
        input_path, input_shape, input_dtype = input_memmap_info
        output_path, output_shape, output_dtype = output_memmap_info
        input_memmap = np.memmap(input_path, dtype=input_dtype, mode='r', shape=input_shape)
        output_memmap = np.memmap(output_path, dtype=output_dtype, mode='r+', shape=output_shape)
        
        slice_data = input_memmap[z, :, :].copy().astype(np.float32)

        frangi_result_2d = frangi(slice_data, sigmas=sigmas_voxel_2d, alpha=frangi_alpha, beta=frangi_beta, gamma=frangi_gamma, black_ridges=black_ridges, mode='reflect')
        sato_result_2d = sato(slice_data, sigmas=sigmas_voxel_2d, black_ridges=black_ridges, mode='reflect')
        slice_enhanced = np.maximum(frangi_result_2d, sato_result_2d)
        
        output_memmap[z, :, :] = slice_enhanced.astype(output_dtype)
        output_memmap.flush()
        del slice_data, frangi_result_2d, sato_result_2d, slice_enhanced, input_memmap, output_memmap; gc.collect()
        return None
    except Exception as e:
        print(f"ERROR in worker processing slice {z}: {e}")
        traceback.print_exc(); return f"Error_slice_{z}"

# +++ START: REFACTORED ENHANCEMENT FUNCTION WITH ORIGINAL SIGNATURE +++
def enhance_tubular_structures_slice_by_slice(
    volume, scales, spacing, black_ridges=False,
    frangi_alpha=0.5, frangi_beta=0.5, frangi_gamma=15,
    apply_3d_smoothing=True, smoothing_sigma_phys=0.5,
    ram_safety_factor=0.8, mem_factor_per_slice=8.0,
    skip_tubular_enhancement=False
):
    """
    Enhance tubular structures slice-by-slice, now with memory-safe Dask smoothing.
    The function signature is identical to the original script.
    """
    print(f"  Volume shape: {volume.shape}, Spacing: {spacing}")
    volume_shape = volume.shape
    num_slices = volume_shape[0]
    
    # This list tracks temporary directories created inside this function
    internal_temp_dirs = []
    
    # --- Step 1: Prepare input volume (Smoothing or Copying) ---
    # This will be the input for the enhancement step.
    enhancement_input_memmap = None
    
    if apply_3d_smoothing and smoothing_sigma_phys > 0:
        # Use the new Dask-based smoothing function
        smoothed_memmap, temp_dir = _smooth_volume_dask_internal(volume, spacing, smoothing_sigma_phys)
        enhancement_input_memmap = smoothed_memmap
        internal_temp_dirs.append(temp_dir)
    else:
        print("  Skipping initial 3D smoothing. Creating a float32 memmap copy.")
        # Create a float32 copy if no smoothing is done, as subsequent steps expect it.
        temp_dir = tempfile.mkdtemp(prefix="input_copy_")
        path = os.path.join(temp_dir, 'input_copy.dat')
        memmap_copy = np.memmap(path, dtype=np.float32, mode='w+', shape=volume.shape)
        # Copy chunk by chunk to be safe
        for i in tqdm(range(0, volume.shape[0], 100), desc="Copying to float32 memmap"):
            end = min(i + 100, volume.shape[0])
            memmap_copy[i:end] = volume[i:end].astype(np.float32)
        memmap_copy.flush()
        enhancement_input_memmap = memmap_copy
        internal_temp_dirs.append(temp_dir)

    # --- Step 2: Perform Enhancement or just copy the smoothed/prepared data ---
    output_temp_dir = tempfile.mkdtemp(prefix='tubular_output_')
    output_path = os.path.join(output_temp_dir, 'processed_volume.dat')
    output_memmap = np.memmap(output_path, dtype=np.float32, mode='w+', shape=volume_shape)

    if skip_tubular_enhancement:
        print("  Skipping tubular structure enhancement as requested.")
        for i in tqdm(range(0, volume_shape[0], 100), desc="Copying (enhancement skipped)"):
            end = min(i + 100, volume_shape[0])
            output_memmap[i:end] = enhancement_input_memmap[i:end]
        output_memmap.flush()
    else:
        print(f"Starting slice-by-slice (2.5D) tubular enhancement...")
        xy_spacing = tuple(float(s) for s in spacing)[1:]
        avg_xy_spacing = np.mean(xy_spacing)
        sigmas_voxel_2d = sorted([s / avg_xy_spacing for s in scales])
        print(f"  Using 2D voxel sigmas: {sigmas_voxel_2d}")
        
        try:
            total_cores = os.cpu_count()
            num_workers = max(1, total_cores - 2 if total_cores > 2 else 1)
        except Exception:
            num_workers = 1

        start_time = time.time()
        try:
            input_info = (enhancement_input_memmap.filename, enhancement_input_memmap.shape, enhancement_input_memmap.dtype)
            output_info = (output_path, output_memmap.shape, output_memmap.dtype)
            worker_func = partial(_process_slice_worker, input_memmap_info=input_info, output_memmap_info=output_info, sigmas_voxel_2d=sigmas_voxel_2d, black_ridges=black_ridges, frangi_alpha=frangi_alpha, frangi_beta=frangi_beta, frangi_gamma=frangi_gamma)
            
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap_unordered(worker_func, range(num_slices)), total=num_slices, desc="Applying 2D Filters (Parallel)"))
            
            errors = [r for r in results if r is not None]
            if errors:
                raise RuntimeError(f"Parallel slice processing failed: {errors}")
            print(f"Parallel processing finished in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"An error occurred during parallel processing: {e}"); traceback.print_exc(); raise

    # --- Step 3: Cleanup and Return ---
    # Clean up the intermediate smoothed/copied data
    del enhancement_input_memmap; gc.collect()
    for d_path in internal_temp_dirs:
        if d_path and os.path.exists(d_path):
            rmtree(d_path, ignore_errors=True)

    print(f"Enhanced volume generated as memmap: {output_path}")
    return output_memmap, output_path, output_temp_dir
# +++ END: REFACTORED ENHANCEMENT FUNCTION +++


def connect_fragmented_processes_dask(binary_dask_array, spacing, max_gap_physical=1.0):
    """Connect fragmented processes using Dask for memory-efficient morphological closing."""
    print(f"Connecting fragmented processes with max physical gap: {max_gap_physical} using Dask.")
    radius_vox = [math.ceil((max_gap_physical / 2) / s) if s > 1e-9 else 0 for s in spacing]
    structure_shape = tuple(max(1, 2 * r + 1) for r in radius_vox)
    if all(s == 1 for s in structure_shape):
        print("  Warning: Structure shape is all ones. Binary closing will have no effect.")
        return binary_dask_array
    structure = np.ones(structure_shape, dtype=bool)
    print("  Applying Dask anisotropic binary closing...")
    return dask_image.ndmorph.binary_closing(binary_dask_array, structure=structure)

def remove_small_objects_dask(binary_dask_array, min_size_voxels):
    """Remove small objects from a binary Dask array in a memory-efficient way."""
    if min_size_voxels <= 1:
        return binary_dask_array
    print(f"  Removing objects smaller than {min_size_voxels} voxels using Dask.")
    s = generate_binary_structure(binary_dask_array.ndim, 1)
    labeled_array, num_features = dask_image.ndmeasure.label(binary_dask_array, structure=s)
    
    print("  Calculating number of features...")
    # Corrected ProgressBar call
    with ProgressBar():
        num_features = num_features.compute()
    print(f"  Found {num_features} initial objects.")
    if num_features == 0:
        return binary_dask_array

    print("  Calculating sizes of all objects...")
    # Corrected ProgressBar call
    with ProgressBar():
        bins = np.arange(num_features + 2)
        object_sizes, _ = da.histogram(labeled_array, bins=bins)
        object_sizes = object_sizes.compute()
    
    small_objects_labels = np.where(object_sizes[1:] < min_size_voxels)[0] + 1
    if small_objects_labels.size == 0:
        print("  No small objects to remove.")
        return binary_dask_array
    print(f"  Found {len(small_objects_labels)} small objects to remove.")
    small_objects_mask = da.isin(labeled_array, small_objects_labels)
    return da.where(small_objects_mask, False, binary_dask_array)


# === Main Function for Raw Segmentation (Unchanged Call Signature) ===
def segment_cells_first_pass_raw(
    volume, # Original intensity volume
    spacing,
    tubular_scales=[0.5, 1.0, 2.0, 3.0],
    smooth_sigma=0.5,
    connect_max_gap_physical=1.0,
    min_size_voxels=50,
    low_threshold_percentile=25.0,
    high_threshold_percentile=95.0,
    skip_tubular_enhancement=False
    ):

    original_shape = volume.shape
    spacing_float = tuple(float(s) for s in spacing)
    first_pass_params = {
        'spacing': spacing_float, 'tubular_scales': tubular_scales, 'smooth_sigma': smooth_sigma,
        'connect_max_gap_physical': connect_max_gap_physical, 'min_size_voxels': min_size_voxels,
        'low_threshold_percentile': low_threshold_percentile,
        'high_threshold_percentile': high_threshold_percentile,
        'original_shape': original_shape,
        'skip_tubular_enhancement': skip_tubular_enhancement
    }
    
    print(f"\n--- Starting Raw First Pass Segmentation ---")
    initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_mem:.2f} MB")

    memmap_registry = {}
    segmentation_threshold = 0.0

    try:
        # --- Step 1: Enhance (or skip) ---
        print(f"\nStep 1: Preparing processed input volume...")
        processed_input_memmap, processed_input_path, processed_input_temp_dir = enhance_tubular_structures_slice_by_slice(
            volume, scales=tubular_scales, spacing=spacing_float,
            apply_3d_smoothing=(smooth_sigma > 0), smoothing_sigma_phys=smooth_sigma,
            skip_tubular_enhancement=skip_tubular_enhancement
        )
        memmap_registry['processed_input'] = (processed_input_memmap, processed_input_path, processed_input_temp_dir)
        print(f"Step 1 output memmap created: {processed_input_path}")
        gc.collect()

        # --- Step 2: Normalize ---
        print("\nStep 2: Normalizing signal...")
        normalized_temp_dir = tempfile.mkdtemp(prefix="normalize_")
        normalized_path = os.path.join(normalized_temp_dir, 'normalized.dat')
        normalized_memmap = np.memmap(normalized_path, dtype=np.float32, mode='w+', shape=original_shape)
        memmap_registry['normalized'] = (normalized_memmap, normalized_path, normalized_temp_dir)
        
        # This normalization logic is already chunked and safe.
        z_high_percentile_values = []
        target_intensity = 1.0
        for z_start in tqdm(range(0, original_shape[0], 50), desc="  Calc z-stats"):
            chunk = processed_input_memmap[z_start:z_start+50]
            for i_slice, z_val in enumerate(range(z_start, z_start + chunk.shape[0])):
                plane = chunk[i_slice]
                finite_plane_values = plane[np.isfinite(plane)].ravel()
                if finite_plane_values.size == 0:
                    z_high_percentile_values.append((z_val, 0.0))
                    continue
                samples_count = min(100000, finite_plane_values.size)
                samples = np.random.choice(finite_plane_values, samples_count, replace=False) if samples_count > 0 else np.array([])
                high_perc_value = np.percentile(samples, high_threshold_percentile) if samples.size > 0 else 0.0
                if not np.isfinite(high_perc_value): high_perc_value = 0.0
                z_high_percentile_values.append((z_val, float(high_perc_value)))
        z_stats_dict = dict(z_high_percentile_values)
        for z_start in tqdm(range(0, original_shape[0], 50), desc="  Normalize z-planes"):
             chunk_to_normalize = processed_input_memmap[z_start:z_start+50]
             normalized_chunk = np.zeros_like(chunk_to_normalize, dtype=np.float32)
             for i, z_val in enumerate(range(z_start, z_start + chunk_to_normalize.shape[0])):
                 high_perc = z_stats_dict.get(z_val, 0.0)
                 if high_perc > 1e-9:
                     normalized_chunk[i] = (chunk_to_normalize[i] * (target_intensity / high_perc)).astype(np.float32)
                 else:
                     normalized_chunk[i] = chunk_to_normalize[i]
             normalized_memmap[z_start:z_start+50] = normalized_chunk
        normalized_memmap.flush()
        del memmap_registry['processed_input']; gc.collect(); rmtree(processed_input_temp_dir, ignore_errors=True)

        # --- Step 3: Thresholding ---
        print("\nStep 3: Thresholding...")
        binary_temp_dir = tempfile.mkdtemp(prefix="binary_")
        binary_path = os.path.join(binary_temp_dir, 'b.dat')
        binary_memmap = np.memmap(binary_path, dtype=bool, mode='w+', shape=original_shape)
        memmap_registry['binary'] = (binary_memmap, binary_path, binary_temp_dir)
        
        # +++ START: REVERTED TO ROBUST, MEMORY-SAFE MANUAL SAMPLING +++
        print("  Sampling for threshold using memory-safe manual chunks...")
        sample_size_target = 5_000_000
        all_samples_list = []
        collected_count = 0
        
        # Get a random order of slices to read
        z_indices = np.arange(original_shape[0])
        np.random.shuffle(z_indices)
        
        # Read a small number of random slices at a time
        slices_per_chunk = 20 
        
        for i in tqdm(range(0, len(z_indices), slices_per_chunk), desc="  Sampling Norm. Vol."):
            if collected_count >= sample_size_target:
                break
            
            # Get the indices for this chunk of random slices
            z_chunk_indices = z_indices[i:i + slices_per_chunk]
            
            # Load only these slices into memory
            slices_data = normalized_memmap[z_chunk_indices, :, :]
            
            # Take only finite values
            finite_values = slices_data[np.isfinite(slices_data)].ravel()
            del slices_data; gc.collect()

            if finite_values.size > 0:
                samples_to_take = min(len(finite_values), sample_size_target - collected_count)
                chosen_samples = np.random.choice(finite_values, samples_to_take, replace=False)
                all_samples_list.append(chosen_samples)
                collected_count += len(chosen_samples)
            
            del finite_values; gc.collect()

        if all_samples_list:
            all_samples_np = np.concatenate(all_samples_list)
            del all_samples_list; gc.collect()
            if all_samples_np.size > 0:
                segmentation_threshold = float(np.percentile(all_samples_np, low_threshold_percentile))
            else:
                segmentation_threshold = 0.0
            del all_samples_np; gc.collect()
        else:
            segmentation_threshold = 0.0
        
        print(f"  Segmentation threshold ({low_threshold_percentile:.1f} perc): {segmentation_threshold:.4f}")
        # +++ END: REVERTED SAMPLING LOGIC +++
        
        # Memory-safe thresholding loop
        for i_thresh in tqdm(range(0, original_shape[0], 100), desc="  Applying Threshold"):
             end_idx_thresh = min(i_thresh + 100, original_shape[0])
             norm_chunk = normalized_memmap[i_thresh:end_idx_thresh]
             for i in range(norm_chunk.shape[0]):
                 binary_memmap[i_thresh + i, :, :] = norm_chunk[i, :, :] > segmentation_threshold
        binary_memmap.flush()
        del memmap_registry['normalized']; gc.collect(); rmtree(normalized_temp_dir, ignore_errors=True)

        # --- Steps 4, 5, 6 with Dask/Zarr (Unchanged) ---
        dask_chunk_size = (64, 256, 256)
        binary_dask_array = da.from_array(binary_memmap, chunks=dask_chunk_size)
        
        print("\nStep 4: Connecting fragments (Dask)...")
        connected_dask_array = connect_fragmented_processes_dask(binary_dask_array, spacing_float, connect_max_gap_physical)
        
        print(f"\nStep 5: Cleaning objects < {min_size_voxels} voxels (Dask)...")
        cleaned_dask_array = remove_small_objects_dask(connected_dask_array, min_size_voxels)
        
        print("\nStep 6: Labeling components (Dask)...")
        s = generate_binary_structure(cleaned_dask_array.ndim, 1)
        labeled_dask_array, num_features_dask = dask_image.ndmeasure.label(cleaned_dask_array, structure=s)
        
        labels_temp_dir = tempfile.mkdtemp(prefix="labels_")
        labels_path = os.path.join(labels_temp_dir, 'l.dat')
        temp_zarr_path = os.path.join(labels_temp_dir, 'temp_labels.zarr')
        
        print(f"  Computing and storing result to temporary Zarr store: {temp_zarr_path}")
        with ProgressBar():
            labeled_dask_array.to_zarr(temp_zarr_path, overwrite=True)
        
        print("  Copying Zarr to final memmap file.")
        final_labels_memmap = np.memmap(labels_path, dtype=np.int32, mode='w+', shape=original_shape)
        zarr_array = zarr.open(temp_zarr_path, mode='r')
        for i in tqdm(range(0, original_shape[0], 50), desc="  Copying Zarr to Memmap"):
            start, end = i, min(i + 50, original_shape[0])
            final_labels_memmap[start:end, :, :] = zarr_array[start:end, :, :]
        final_labels_memmap.flush()
        
        rmtree(temp_zarr_path)
        with ProgressBar():
            num_features = num_features_dask.compute()
        print(f"Labeling done ({num_features} features found).")

        print(f"\n--- Raw First Pass Segmentation Finished ---")
        return labels_path, labels_temp_dir, segmentation_threshold, first_pass_params

    except Exception as e:
        print(f"\n!!! ERROR during Raw First Pass Segmentation: {e} !!!")
        traceback.print_exc()
        raise
    finally:
        print("\nFinal cleanup check...")
        registry_keys = list(memmap_registry.keys())
        for name_key in registry_keys:
             items = memmap_registry.pop(name_key)
             mm_obj, d_dir = items[0], items[-1]
             print(f"  Cleaning up leftover {name_key}...")
             if hasattr(mm_obj, '_mmap') and mm_obj._mmap is not None: del mm_obj
             if d_dir and os.path.exists(d_dir):
                 try: rmtree(d_dir, ignore_errors=True)
                 except Exception as e_rmtree: print(f"    Error removing dir {d_dir}: {e_rmtree}")
        gc.collect()