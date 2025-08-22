# --- START OF FILE initial_3d_segmentation.py ---

import numpy as np
from scipy import ndimage
from scipy.ndimage import (gaussian_filter, generate_binary_structure)
from skimage.filters import frangi, sato # type: ignore
from skimage.morphology import remove_small_objects as skimage_remove_small_objects
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
import dask.array as da
from dask.diagnostics import ProgressBar
import dask_image.ndmorph
import dask_image.ndmeasure
import zarr

from .remove_artifacts import * # Assuming this exists

seed = 42
np.random.seed(seed)

def _process_slice_worker(z, input_memmap_info, output_memmap_info, sigmas_voxel_2d, black_ridges, frangi_alpha, frangi_beta, frangi_gamma):
    """Worker function for parallel tubular enhancement with robust cleanup."""
    input_memmap, output_memmap = None, None
    try:
        input_path, input_shape, input_dtype = input_memmap_info
        output_path, output_shape, output_dtype = output_memmap_info
        input_memmap = np.memmap(input_path, dtype=input_dtype, mode='r', shape=input_shape)
        output_memmap = np.memmap(output_path, dtype=output_dtype, mode='r+', shape=output_shape)
        slice_data = input_memmap[z].astype(np.float32, copy=True)
        frangi_result_2d = frangi(slice_data, sigmas=sigmas_voxel_2d, alpha=frangi_alpha, beta=frangi_beta, gamma=frangi_gamma, black_ridges=black_ridges, mode='reflect')
        sato_result_2d = sato(slice_data, sigmas=sigmas_voxel_2d, black_ridges=black_ridges, mode='reflect')
        output_memmap[z] = np.maximum(frangi_result_2d, sato_result_2d)
    except Exception as e:
        traceback.print_exc(); return f"Error_slice_{z}"
    finally:
        if output_memmap is not None: output_memmap.flush()
        del input_memmap, output_memmap
        gc.collect()


def enhance_tubular_structures_slice_by_slice(
    volume, scales, spacing, black_ridges=False,
    frangi_alpha=0.5, frangi_beta=0.5, frangi_gamma=15,
    apply_3d_smoothing=True, smoothing_sigma_phys=0.5,
    skip_tubular_enhancement=False
):
    """Enhancement function with robust process management to prevent memory leaks."""
    print(f"  Volume shape: {volume.shape}, Spacing: {spacing}")
    spacing = tuple(float(s) for s in spacing); volume_shape = volume.shape
    temp_dirs_created_internally = []
    
    input_volume_memmap = None
    if apply_3d_smoothing and smoothing_sigma_phys > 0:
        print(f"Applying initial 3D smoothing (output to float32 memmap)...")
        sigma_voxel_3d = tuple(smoothing_sigma_phys / s if s > 0 else 0 for s in spacing)
        temp_dir = tempfile.mkdtemp(prefix="pre_smooth_"); temp_dirs_created_internally.append(temp_dir)
        smooth_path = os.path.join(temp_dir, 'smoothed_3d.dat')
        input_volume_memmap = np.memmap(smooth_path, dtype=np.float32, mode='w+', shape=volume_shape)
        chunk_size = min(50, volume_shape[0]) or 1; overlap = math.ceil(3 * sigma_voxel_3d[0]) if sigma_voxel_3d[0] > 0 else 0
        for i in tqdm(range(0, volume_shape[0], chunk_size), desc="3D Pre-Smoothing"):
            start_r, end_r = max(0, i - overlap), min(volume_shape[0], i + chunk_size + overlap)
            start_w, end_w = i, min(volume_shape[0], i + chunk_size)
            lws, lwe = start_w - start_r, end_w - start_r
            if start_r >= end_r or lws >= lwe: continue
            chunk = volume[start_r:end_r].astype(np.float32, copy=False)
            smoothed_chunk = gaussian_filter(chunk, sigma=sigma_voxel_3d, mode='reflect')
            input_volume_memmap[start_w:end_w] = smoothed_chunk[lws:lwe]
        input_volume_memmap.flush()
    else:
        print("  Skipping initial 3D smoothing. Creating float32 memmap copy if needed.")
        if isinstance(volume, np.memmap) and volume.dtype == np.float32:
            input_volume_memmap = volume
        else:
            temp_dir = tempfile.mkdtemp(prefix="input_float32_copy_"); temp_dirs_created_internally.append(temp_dir)
            path = os.path.join(temp_dir, 'input_float32.dat')
            input_volume_memmap = np.memmap(path, dtype=np.float32, mode='w+', shape=volume.shape)
            for i in tqdm(range(0, volume.shape[0]), desc="Copying to float32 memmap"):
                input_volume_memmap[i] = volume[i].astype(np.float32)
            input_volume_memmap.flush()

    output_temp_dir = tempfile.mkdtemp(prefix='tubular_output_')
    output_path = os.path.join(output_temp_dir, 'processed_volume.dat')
    output_memmap = np.memmap(output_path, dtype=np.float32, mode='w+', shape=volume_shape)

    if skip_tubular_enhancement:
        print("  Copying input to output as enhancement is skipped.")
        for i in tqdm(range(volume_shape[0]), desc="Copying to output (skip enhancement)"):
            output_memmap[i] = input_volume_memmap[i]
        output_memmap.flush()
    else:
        pool = None
        try:
            print(f"Starting slice-by-slice (2.5D) tubular enhancement...")
            num_workers = max(1, os.cpu_count() - 2 if os.cpu_count() > 2 else 1)
            xy_spacing = spacing[1:]; avg_xy_spacing = np.mean(xy_spacing)
            sigmas_voxel_2d = sorted([s / avg_xy_spacing for s in scales])
            input_info = (input_volume_memmap.filename, input_volume_memmap.shape, input_volume_memmap.dtype)
            output_info = (output_path, output_memmap.shape, output_memmap.dtype)
            worker_func = partial(_process_slice_worker, input_memmap_info=input_info, output_memmap_info=output_info, sigmas_voxel_2d=sigmas_voxel_2d, black_ridges=black_ridges, frangi_alpha=frangi_alpha, frangi_beta=frangi_beta, frangi_gamma=frangi_gamma)
            
            pool = Pool(processes=num_workers)
            results = list(tqdm(pool.imap_unordered(worker_func, range(volume_shape[0])), total=volume_shape[0], desc="Applying 2D Filters (Parallel)"))
            if any(r is not None for r in results): raise RuntimeError("Parallel slice processing failed.")
        finally:
            if pool: print("  Forcefully terminating multiprocessing pool..."); pool.terminate(); pool.join(); print("  Pool terminated.")

    del input_volume_memmap; gc.collect()
    for d_path in temp_dirs_created_internally: rmtree(d_path, ignore_errors=True)
    return output_memmap, output_path, output_temp_dir


def segment_cells_first_pass_raw(
    volume, spacing, tubular_scales=[0.5, 1.0, 2.0, 3.0],
    smooth_sigma=0.5, connect_max_gap_physical=1.0, min_size_voxels=50,
    low_threshold_percentile=25.0, high_threshold_percentile=95.0,
    skip_tubular_enhancement=False
    ):

    print(f"\n--- Starting Memory-Safe Hybrid Segmentation (Polynomial Trend Normalization) ---")
    
    temp_dirs_to_clean = []
    final_labels_memmap = None

    try:
        # --- Stage 1 & 2: Pre-processing with CLAMPED POLYNOMIAL-FITTED NORMALIZATION ---
        print("\n--- STAGES 1 & 2: Pre-processing and Thresholding ---")
        
        processed_input_memmap, _, processed_input_temp_dir = enhance_tubular_structures_slice_by_slice(
            volume, scales=tubular_scales, spacing=spacing,
            apply_3d_smoothing=(smooth_sigma > 0), smoothing_sigma_phys=smooth_sigma,
            skip_tubular_enhancement=skip_tubular_enhancement
        )
        temp_dirs_to_clean.append(processed_input_temp_dir)
        
        # +++ START: CLAMPED POLYNOMIAL-FITTED NORMALIZATION LOGIC +++
        
        print("  Step 2.1: Normalizing volume brightness via clamped polynomial trend fitting...")
        normalized_temp_dir = tempfile.mkdtemp(prefix="normalize_"); temp_dirs_to_clean.append(normalized_temp_dir)
        normalized_path = os.path.join(normalized_temp_dir, 'normalized.dat'); normalized_memmap = np.memmap(normalized_path, dtype=np.float32, mode='w+', shape=volume.shape)
        
        # First, calculate the raw peak signal (high percentile) for every slice.
        raw_high_percentiles = []
        slice_indices = np.arange(volume.shape[0])
        for z in tqdm(slice_indices, desc="  Calc z-stats (1/2)"):
            plane = processed_input_memmap[z]; finite_plane_values = plane[np.isfinite(plane)].ravel()
            if finite_plane_values.size < 100:
                raw_high_percentiles.append(np.nan)
                continue
            samples = np.random.choice(finite_plane_values, min(100000, finite_plane_values.size), replace=False)
            high_perc = np.percentile(samples, high_threshold_percentile) if samples.size > 0 else np.nan
            raw_high_percentiles.append(float(high_perc if np.isfinite(high_perc) else np.nan))
        
        raw_high_percentiles = np.array(raw_high_percentiles)
        valid_indices = ~np.isnan(raw_high_percentiles)
        
        # Fit a simple, robust 2nd-degree polynomial to the valid raw percentile values.
        idealized_high_percentiles = np.zeros_like(raw_high_percentiles)
        if np.any(valid_indices):
            poly_coeffs = np.polyfit(slice_indices[valid_indices], raw_high_percentiles[valid_indices], 2)
            poly_func = np.poly1d(poly_coeffs)
            idealized_high_percentiles = poly_func(slice_indices)

            # +++ CRITICAL FIX: Clamp the polynomial to prevent extreme edge effects +++
            # Find a "safe minimum" normalization factor from the raw data (e.g., 10th percentile).
            safe_minimum_percentile = np.nanpercentile(raw_high_percentiles, 10)
            # Prevent the idealized curve from dipping below this safe floor.
            idealized_high_percentiles[idealized_high_percentiles < safe_minimum_percentile] = safe_minimum_percentile
            print(f"  Clamping normalization factor to a minimum of {safe_minimum_percentile:.2f}")

        else:
            idealized_high_percentiles.fill(1.0) # Fallback if no signal

        # Apply the smooth, clamped normalization factor to each slice.
        for z in tqdm(range(volume.shape[0]), desc="  Normalize z-planes (2/2)"):
            ideal_high_perc = idealized_high_percentiles[z]
            if ideal_high_perc > 1e-9:
                normalized_memmap[z] = (processed_input_memmap[z] / ideal_high_perc)
            else:
                normalized_memmap[z] = processed_input_memmap[z]
        normalized_memmap.flush()
        print("  Normalization complete.")
        
        # Step 2.2: Calculate a single, global threshold from the now-consistent normalized volume.
        print("  Step 2.2: Calculating global segmentation threshold...")
        all_samples_list = []
        z_indices_shuffled = np.arange(volume.shape[0]); np.random.shuffle(z_indices_shuffled)
        for i in tqdm(range(0, len(z_indices_shuffled), 20), desc="  Sampling Norm. Vol."):
            if len(all_samples_list) > 50: break 
            slices_data = normalized_memmap[z_indices_shuffled[i:i + 20]]; finite_values = slices_data[np.isfinite(slices_data)].ravel()
            if finite_values.size > 0: all_samples_list.append(np.random.choice(finite_values, min(100000, finite_values.size), replace=False))
        
        segmentation_threshold = 0.0
        if all_samples_list:
            all_samples = np.concatenate(all_samples_list)
            if all_samples.size > 0:
                segmentation_threshold = float(np.percentile(all_samples, low_threshold_percentile))
        
        print(f"  Global segmentation threshold: {segmentation_threshold:.4f}")

        # Step 2.3: Apply the single global threshold to the normalized volume.
        binary_temp_dir = tempfile.mkdtemp(prefix="binary_"); temp_dirs_to_clean.append(binary_temp_dir)
        binary_path = os.path.join(binary_temp_dir, 'b.dat'); binary_memmap = np.memmap(binary_path, dtype=bool, mode='w+', shape=volume.shape)
        for z in tqdm(range(volume.shape[0]), desc="  Applying Threshold"):
            binary_memmap[z] = normalized_memmap[z] > segmentation_threshold
            
        binary_memmap.flush()
        
        # --- VERIFICATION PROFILING ---
        print("\n  --- VERIFICATION PROFILE ---")
        center_slice_idx = volume.shape[0] // 2
        sample_indices = [5, center_slice_idx, volume.shape[0] - 5]
        for idx in sample_indices:
            p_slice = processed_input_memmap[idx]; n_slice = normalized_memmap[idx]; b_slice = binary_memmap[idx]
            segmented_area_percent = b_slice.mean() * 100
            print(f"  Slice {idx}:")
            print(f"    - Processed (Raw Signal): 95th_perc={np.percentile(p_slice[p_slice>0], 95):.2f}")
            print(f"    - Idealized Norm Factor: {idealized_high_percentiles[idx]:.2f}")
            print(f"    - Normalized Signal:     95th_perc={np.percentile(n_slice[n_slice>0], 95):.2f} (Should be ~1.0)")
            print(f"    - Resulting Binary Mask:   Segmented Area = {segmented_area_percent:.2f}%")

        print("\nStages 1 & 2 complete."); del processed_input_memmap, normalized_memmap; gc.collect()
        
        # --- Stage 3: Dask-based Global Cleaning and Labeling ---
        print("\n--- STAGE 3: Memory-Safe Global Cleaning and Labeling ---")
        dask_chunk_size = (32, 256, 256)
        
        print("  Step 3.1: Connecting fragments with Dask...")
        connected_temp_dir = tempfile.mkdtemp(prefix="connected_"); temp_dirs_to_clean.append(connected_temp_dir)
        connected_path = os.path.join(connected_temp_dir, 'connected.dat'); connected_memmap = np.memmap(connected_path, dtype=bool, mode='w+', shape=volume.shape)
        binary_dask_array = da.from_array(binary_memmap, chunks=dask_chunk_size)
        radius_vox = [math.ceil((connect_max_gap_physical / 2) / s) if s > 1e-9 else 0 for s in spacing]
        structure = np.ones(tuple(max(1, 2 * r + 1) for r in radius_vox), dtype=bool) if any(r > 0 for r in radius_vox) else None
        connected_dask_array = dask_image.ndmorph.binary_closing(binary_dask_array, structure=structure) if structure is not None else binary_dask_array
        with ProgressBar(dt=2):
            da.store(connected_dask_array, connected_memmap, scheduler='threads')
        print("  Dask binary closing complete."); del binary_dask_array, connected_dask_array, binary_memmap; gc.collect()

        print("  Step 3.2: Finding all connected components globally with Dask...")
        labeled_temp_dir = tempfile.mkdtemp(prefix="labeled_zarr_"); temp_dirs_to_clean.append(labeled_temp_dir)
        labeled_zarr_path = os.path.join(labeled_temp_dir, 'labeled.zarr')
        connected_dask_array = da.from_array(connected_memmap, chunks=dask_chunk_size)
        s = generate_binary_structure(3, 1)
        labeled_dask_array, num_features_dask = dask_image.ndmeasure.label(connected_dask_array, structure=s)
        print("    Executing label graph and saving to temporary Zarr store...")
        with ProgressBar(dt=2):
            labeled_dask_array.to_zarr(labeled_zarr_path, overwrite=True)
        with ProgressBar(dt=1):
            num_features = num_features_dask.compute()
        print(f"  Found {num_features} initial objects."); del connected_dask_array, labeled_dask_array; gc.collect()

        print(f"  Step 3.3: Filtering objects smaller than {min_size_voxels} voxels...")
        labels_temp_dir = tempfile.mkdtemp(prefix="labels_final_")
        labels_path = os.path.join(labels_temp_dir, 'l.dat')
        final_labels_memmap = np.memmap(labels_path, dtype=np.int32, mode='w+', shape=volume.shape)
        
        initial_labeled_zarr = zarr.open(labeled_zarr_path, mode='r')
        final_label_counter = 0
        object_slices = ndimage.find_objects(initial_labeled_zarr, max_label=num_features)
        
        bbox_chunk_size = 50
        for i in tqdm(range(num_features), desc="  Filtering objects"):
            label = i + 1; slices = object_slices[i]
            if slices is None: continue
            
            z_slice, y_slice, x_slice = slices; total_voxels = 0
            for z_start in range(z_slice.start, z_slice.stop, bbox_chunk_size):
                z_end = min(z_start + bbox_chunk_size, z_slice.stop)
                box_chunk = initial_labeled_zarr[slice(z_start, z_end), y_slice, x_slice]
                total_voxels += np.sum(box_chunk == label)

            if total_voxels >= min_size_voxels:
                final_label_counter += 1
                for z_start in range(z_slice.start, z_slice.stop, bbox_chunk_size):
                    z_end = min(z_start + bbox_chunk_size, z_slice.stop)
                    chunk_slices = (slice(z_start, z_end), y_slice, x_slice)
                    box_chunk = initial_labeled_zarr[chunk_slices]
                    obj_mask_in_chunk = box_chunk == label
                    output_chunk_slice = final_labels_memmap[chunk_slices]
                    output_chunk_slice[obj_mask_in_chunk] = final_label_counter
                    final_labels_memmap[chunk_slices] = output_chunk_slice
        
        final_labels_memmap.flush()
        print("Stage 3 complete.")
        
        num_final_features = final_label_counter
        print(f"\nLabeling done ({num_final_features} features found).")

        first_pass_params = {
            'spacing': tuple(float(s) for s in spacing), 'tubular_scales': tubular_scales, 'smooth_sigma': smooth_sigma,
            'connect_max_gap_physical': connect_max_gap_physical, 'min_size_voxels': min_size_voxels,
            'low_threshold_percentile': low_threshold_percentile, 'high_threshold_percentile': high_threshold_percentile,
            'original_shape': volume.shape, 'skip_tubular_enhancement': skip_tubular_enhancement
        }
        return labels_path, labels_temp_dir, segmentation_threshold, first_pass_params

    finally:
        print("\nFinal cleanup check...")
        if 'final_labels_memmap' in locals() and final_labels_memmap is not None:
             del final_labels_memmap
        for d_dir in temp_dirs_to_clean:
            if d_dir and os.path.exists(d_dir):
                print(f"  Removing temp dir: {d_dir}")
                rmtree(d_dir, ignore_errors=True)
        gc.collect()
        print("Cleanup finished.")