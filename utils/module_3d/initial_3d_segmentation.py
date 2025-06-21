# --- START OF FILE initial_3d_segmentation.py ---

import numpy as np
from scipy import ndimage
# Make sure all necessary ndimage functions are imported
from scipy.ndimage import (gaussian_filter, generate_binary_structure)
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
# import os # duplicate
# import tempfile # duplicate
# import gc # duplicate
# import numpy as np # duplicate
# from tqdm import tqdm # duplicate
from functools import partial
import math
seed = 42
np.random.seed(seed)         # For NumPy
import traceback

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

def _process_slice_worker(z, input_memmap_info, output_memmap_info,
                          sigmas_voxel_2d, black_ridges,
                          frangi_alpha, frangi_beta, frangi_gamma):
    """Worker function to process a single slice."""
    try:
        input_path, input_shape, input_dtype = input_memmap_info
        output_path, output_shape, output_dtype = output_memmap_info
        # Ensure input_dtype for memmap is float32 as expected by frangi/sato after smoothing
        if input_dtype != np.float32:
             # This should not happen if previous steps ensure float32 input to this worker
            print(f"Warning: _process_slice_worker received input_dtype {input_dtype}, expected np.float32. Path: {input_path}")

        input_memmap = np.memmap(input_path, dtype=input_dtype, mode='r', shape=input_shape)
        output_memmap = np.memmap(output_path, dtype=output_dtype, mode='r+', shape=output_shape)
        
        slice_data = input_memmap[z, :, :].copy() # Copy to ensure it's in memory
        # Convert to float32 if not already, frangi/sato expect float
        if not np.issubdtype(slice_data.dtype, np.floating): slice_data = slice_data.astype(np.float32)
        elif slice_data.dtype != np.float32: slice_data = slice_data.astype(np.float32)

        frangi_result_2d = frangi(slice_data, sigmas=sigmas_voxel_2d, alpha=frangi_alpha, beta=frangi_beta, gamma=frangi_gamma, black_ridges=black_ridges, mode='reflect')
        sato_result_2d = sato(slice_data, sigmas=sigmas_voxel_2d, black_ridges=black_ridges, mode='reflect')
        slice_enhanced = np.maximum(frangi_result_2d, sato_result_2d)
        
        output_memmap[z, :, :] = slice_enhanced.astype(output_dtype)
        output_memmap.flush()
        del slice_data, frangi_result_2d, sato_result_2d, slice_enhanced, input_memmap, output_memmap; gc.collect()
        return None
    except Exception as e:
        print(f"ERROR in worker processing slice {z}: {e}")
        import traceback; traceback.print_exc(); return f"Error_slice_{z}"

def enhance_tubular_structures_slice_by_slice(
    volume, scales, spacing, black_ridges=False,
    frangi_alpha=0.5, frangi_beta=0.5, frangi_gamma=15,
    apply_3d_smoothing=True, smoothing_sigma_phys=0.5,
    ram_safety_factor=0.8, mem_factor_per_slice=8.0,
    skip_tubular_enhancement=False # New parameter
):
    """
    Enhance tubular structures slice-by-slice in parallel, or skip if requested.
    Returns a MEMMAP object, its path, and the directory containing it.
    The output memmap will always be float32.
    """
    if skip_tubular_enhancement:
        print(f"Skipping tubular structure enhancement as per request.")
    # Initial message for enhancement moved to the 'else' block for skip_tubular_enhancement

    print(f"  Volume shape: {volume.shape}, Spacing: {spacing}")
    print(f"  Initial memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    spacing = tuple(float(s) for s in spacing); volume_shape = volume.shape
    slice_shape = volume_shape[1:]; num_slices = volume_shape[0]; xy_spacing = spacing[1:]
    
    input_volume_memmap = None # This will point to the data to be processed (or copied)
    input_memmap_path = None   # Path to input_volume_memmap's data file
    input_memmap_dir_local = None # Directory if input_volume_memmap is a local temp copy/smooth
    temp_dirs_created_internally = [] # Tracks dirs made by this func for internal data

    # --- Step 1: Prepare input_volume_memmap (ensuring it's float32 and a memmap) ---
    if apply_3d_smoothing and smoothing_sigma_phys > 0:
        print(f"Applying initial 3D smoothing (output to float32 memmap)...")
        sigma_voxel_3d = tuple(smoothing_sigma_phys / s if s > 0 else 0 for s in spacing) # Avoid div by zero
        input_memmap_dir_local = tempfile.mkdtemp(prefix="pre_smooth_")
        temp_dirs_created_internally.append(input_memmap_dir_local)
        smooth_path = os.path.join(input_memmap_dir_local, 'smoothed_3d.dat')
        smoothed_dtype = np.float32
        input_volume_memmap = np.memmap(smooth_path, dtype=smoothed_dtype, mode='w+', shape=volume_shape)
        input_memmap_path = smooth_path

        chunk_size_z_smooth = min(50, volume_shape[0]) if volume_shape[0] > 0 else 1
        overlap_z_smooth = math.ceil(3 * sigma_voxel_3d[0]) if sigma_voxel_3d[0] > 0 else 0

        for i in tqdm(range(0, volume_shape[0], chunk_size_z_smooth), desc="3D Pre-Smoothing"):
            start_read = max(0, i - overlap_z_smooth)
            end_read = min(volume_shape[0], i + chunk_size_z_smooth + overlap_z_smooth)
            start_write = i; end_write = min(volume_shape[0], i + chunk_size_z_smooth)
            local_write_start = start_write - start_read; local_write_end = end_write - start_read
            
            if start_read >= end_read or local_write_start >= local_write_end : continue
            
            chunk = volume[start_read:end_read].astype(smoothed_dtype).copy()
            smoothed_chunk = gaussian_filter(chunk, sigma=sigma_voxel_3d, mode='reflect')
            input_volume_memmap[start_write:end_write, :, :] = smoothed_chunk[local_write_start:local_write_end, :, :]
            del chunk, smoothed_chunk; gc.collect()
        input_volume_memmap.flush()
        print(f"  3D pre-smoothing to float32 memmap done.")
    else:
        print("  Skipping initial 3D smoothing.")
        if isinstance(volume, np.memmap) and volume.dtype == np.float32:
            print("  Input is already a float32 memmap. Using it directly.")
            input_volume_memmap = volume
            input_memmap_path = volume.filename
            # input_memmap_dir_local remains None
        else:
            dtype_msg = f"memmap of dtype {volume.dtype}" if isinstance(volume, np.memmap) else f"numpy array of dtype {volume.dtype}"
            print(f"  Input is {dtype_msg}. Converting to a new temporary float32 memmap...")
            input_memmap_dir_local = tempfile.mkdtemp(prefix="input_float32_copy_")
            temp_dirs_created_internally.append(input_memmap_dir_local)
            _temp_input_path = os.path.join(input_memmap_dir_local, 'input_float32.dat')
            input_volume_memmap = np.memmap(_temp_input_path, dtype=np.float32, mode='w+', shape=volume.shape)
            input_memmap_path = _temp_input_path

            chunk_size_copy = min(100, volume.shape[0]) if volume.shape[0] > 0 else 1
            for i_chunk in tqdm(range(0, volume.shape[0], chunk_size_copy), desc="Copying input to float32 memmap"):
                end_chunk = min(i_chunk + chunk_size_copy, volume.shape[0])
                input_volume_memmap[i_chunk:end_chunk] = volume[i_chunk:end_chunk].astype(np.float32)
            input_volume_memmap.flush()
            print("  Input copied to float32 memmap.")

    # --- Step 2: Prepare Output Memmap ---
    output_temp_dir = tempfile.mkdtemp(prefix='tubular_output_')
    output_path = os.path.join(output_temp_dir, 'processed_volume.dat')
    output_dtype = np.float32
    output_memmap = np.memmap(output_path, dtype=output_dtype, mode='w+', shape=volume_shape)
    print(f"  Output memmap created: {output_path}")

    # --- Step 3: Perform Enhancement or Copy ---
    if skip_tubular_enhancement:
        print("  Copying (potentially smoothed) input to output memmap as enhancement is skipped.")
        chunk_size_copy = min(100, volume_shape[0]) if volume_shape[0] > 0 else 1
        for i in tqdm(range(0, volume_shape[0], chunk_size_copy), desc="Copying to output (skip enhancement)"):
            start_idx = i; end_idx = min(i + chunk_size_copy, volume_shape[0])
            chunk_data = input_volume_memmap[start_idx:end_idx] # Already float32
            output_memmap[start_idx:end_idx] = chunk_data
            del chunk_data; gc.collect()
        output_memmap.flush()
        print("  Copying finished.")
    else:
        print(f"Starting slice-by-slice (2.5D) tubular enhancement with PARALLEL processing...")
        print(f"  xy spacing is: {xy_spacing}, scales: {scales}")
        avg_xy_spacing = np.mean(xy_spacing); sigmas_voxel_2d = sorted([s / avg_xy_spacing for s in scales])
        print(f"  Using 2D voxel sigmas: {sigmas_voxel_2d}")
        
        try:
            total_cores = os.cpu_count()
            # Try to leave at least 1 or 2 cores free for system stability
            max_cpu_workers = max(1, total_cores - 2 if total_cores > 2 else (total_cores - 1 if total_cores > 1 else 1))
            available_ram = psutil.virtual_memory().available; usable_ram = available_ram * ram_safety_factor
            input_dtype_bytes = np.dtype(input_volume_memmap.dtype).itemsize # Should be float32
            slice_mem_bytes = slice_shape[0] * slice_shape[1] * input_dtype_bytes
            estimated_worker_ram = slice_mem_bytes * mem_factor_per_slice
            if estimated_worker_ram <= 0: estimated_worker_ram = 1024 * 1024 # Min 1MB safeguard
            max_mem_workers = max(1, int(usable_ram // estimated_worker_ram)) if estimated_worker_ram > 0 else 1
            num_workers = min(max_cpu_workers, max_mem_workers, num_slices)
            print(f"  Resource Check: Total Cores={total_cores} (using {num_workers}), Avail RAM={available_ram / (1024**3):.2f}GB, Usable RAM={usable_ram / (1024**3):.2f}GB")
            print(f"  Est. RAM/worker={estimated_worker_ram / (1024**2):.2f}MB -> Max RAM Workers={max_mem_workers}")
        except Exception as e:
            print(f"  Warning: Could not determine resources automatically ({e}). Defaulting to 1 worker."); num_workers = 1

        start_time = time.time()
        try:
            input_info = (input_memmap_path, input_volume_memmap.shape, input_volume_memmap.dtype)
            output_info = (output_path, output_memmap.shape, output_dtype)
            worker_func_partial = partial(_process_slice_worker, input_memmap_info=input_info, output_memmap_info=output_info, sigmas_voxel_2d=sigmas_voxel_2d, black_ridges=black_ridges, frangi_alpha=frangi_alpha, frangi_beta=frangi_beta, frangi_gamma=frangi_gamma)
            print(f"Processing {num_slices} slices using {num_workers} workers...")
            with Pool(processes=num_workers) as pool:
                results = list(tqdm(pool.imap_unordered(worker_func_partial, range(num_slices)), total=num_slices, desc="Applying 2D Filters (Parallel)"))
            errors = [r for r in results if isinstance(r, str) and r.startswith("Error_slice_")]
            if errors: print(f"\n!!! Encountered {len(errors)} errors during parallel processing: {errors[:5]}"); raise RuntimeError(f"Parallel slice processing failed: {errors}")
            print(f"Parallel processing finished in {time.time() - start_time:.2f} seconds.")
        except Exception as e:
            print(f"An error occurred during parallel processing: {e}"); import traceback; traceback.print_exc(); raise

    # --- Step 4: Cleanup and Return ---
    if input_volume_memmap is not None:
        if hasattr(input_volume_memmap, '_mmap') and input_volume_memmap._mmap is not None:
            input_volume_memmap.flush()
            if input_memmap_dir_local is not None: # It's a temp memmap created here, del Python object
                 del input_volume_memmap
            # else: input_volume_memmap was the original 'volume' passed in, don't del its Python object.
        gc.collect()

    for d_path in temp_dirs_created_internally:
        if d_path and os.path.exists(d_path):
            print(f"Cleaning up temporary internal directory: {d_path}")
            try: rmtree(d_path, ignore_errors=True)
            except Exception as e: print(f"Warning: Could not delete temp internal directory {d_path}: {e}")
    temp_dirs_created_internally.clear()

    if skip_tubular_enhancement:
        print(f"Skipped enhancement. Output memmap (copy of input/smoothed): {output_path}")
    else:
        print(f"Enhanced volume generated as memmap: {output_path}")
    print(f"Memory usage after enhance function (before returning memmap): {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return output_memmap, output_path, output_temp_dir

# Connection function (remains the same with minor robustness improvement)
def connect_fragmented_processes(binary_mask, spacing, max_gap_physical=1.0):
    """Connect fragmented processes using anisotropic morphological closing."""
    print(f"Connecting fragmented processes with max physical gap: {max_gap_physical}")
    print(f"  Input mask shape: {binary_mask.shape}, Spacing: {spacing}")
    radius_vox = [math.ceil((max_gap_physical / 2) / s) if s > 1e-9 else 0 for s in spacing] # Avoid div by zero/small
    structure_shape = tuple(max(1, 2 * r + 1) for r in radius_vox)
    print(f"  Calculated anisotropic structure shape (voxels): {structure_shape}")
    
    if all(s == 1 for s in structure_shape):
        print("  Warning: Structure shape is all ones. Binary closing will likely have no effect. Check max_gap_physical and spacing.")

    structure = np.ones(structure_shape, dtype=bool)
    temp_dir = tempfile.mkdtemp(prefix="connect_frag_")
    result_path = os.path.join(temp_dir, 'connected_mask.dat')
    connected_mask = np.memmap(result_path, dtype=np.bool_, mode='w+', shape=binary_mask.shape)
    print("  Applying anisotropic binary closing...")
    start_time = time.time()
    try:
        ndimage.binary_closing(binary_mask, structure=structure, output=connected_mask, border_value=0)
        connected_mask.flush(); print(f"  Binary closing completed in {time.time() - start_time:.2f} seconds.")
    except MemoryError: print("  MemoryError during binary closing."); del connected_mask; gc.collect(); rmtree(temp_dir); raise MemoryError("Failed binary closing.") from None
    except Exception as e: print(f"  Error during binary closing: {e}"); del connected_mask; gc.collect(); rmtree(temp_dir); raise e
    return connected_mask, temp_dir

# === Function for Raw Segmentation (Steps 1-6) Modified ===
def segment_cells_first_pass_raw(
    volume, # Original intensity volume
    spacing,
    tubular_scales=[0.5, 1.0, 2.0, 3.0],
    smooth_sigma=0.5,
    connect_max_gap_physical=1.0,
    min_size_voxels=50,
    low_threshold_percentile=25.0,
    high_threshold_percentile=95.0,
    skip_tubular_enhancement=False # New parameter
    ):
    """
    Performs initial segmentation steps 1-6 (Enhance or Skip, Normalize, Threshold,
    Connect, Clean, Label) without edge trimming. Uses memmaps.

    Returns:
    --------
    labels_path : str
        Path to the final labeled segmentation memmap file.
    labels_temp_dir : str
        Path to the temporary directory containing the labels memmap.
    segmentation_threshold : float
        The calculated absolute threshold used for segmentation.
    first_pass_params : dict
        Dictionary containing parameters used.
    """
    low_threshold_percentile = max(0.0, min(100.0, low_threshold_percentile))
    high_threshold_percentile = max(0.0, min(100.0, high_threshold_percentile))

    original_shape = volume.shape
    spacing_float = tuple(float(s) for s in spacing) # Ensure spacing is float tuple
    first_pass_params = {
        'spacing': spacing_float, 'tubular_scales': tubular_scales, 'smooth_sigma': smooth_sigma,
        'connect_max_gap_physical': connect_max_gap_physical, 'min_size_voxels': min_size_voxels,
        'low_threshold_percentile': low_threshold_percentile,
        'high_threshold_percentile': high_threshold_percentile,
        'original_shape': original_shape,
        'skip_tubular_enhancement': skip_tubular_enhancement
        }
    print(f"\n--- Starting Raw First Pass Segmentation ---")
    if skip_tubular_enhancement:
        print("NOTE: Tubular enhancement step will be SKIPPED.")
    initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_mem:.2f} MB")

    memmap_registry = {}
    segmentation_threshold = 0.0

    try:
        # --- Step 1: Enhance (or skip) ---
        step1_input_description = "(potentially smoothed) input" if skip_tubular_enhancement else "enhanced structures"
        print(f"\nStep 1: Preparing {step1_input_description}...")

        processed_input_memmap, processed_input_path, processed_input_temp_dir = enhance_tubular_structures_slice_by_slice(
            volume, scales=tubular_scales, spacing=spacing_float, # Use spacing_float
            apply_3d_smoothing=(smooth_sigma > 0), smoothing_sigma_phys=smooth_sigma,
            skip_tubular_enhancement=skip_tubular_enhancement
        )
        memmap_registry['processed_input'] = (processed_input_memmap, processed_input_path, processed_input_temp_dir)
        print(f"Step 1 output ('{step1_input_description}') memmap created: {processed_input_path}")
        gc.collect()

        # --- Step 2: Normalize ---
        current_processed_input_memmap = memmap_registry['processed_input'][0]
        print("\nStep 2: Normalizing signal...")
        normalized_temp_dir = tempfile.mkdtemp(prefix="normalize_")
        normalized_path = os.path.join(normalized_temp_dir, 'normalized.dat')
        normalized_memmap = np.memmap(normalized_path, dtype=np.float32, mode='w+', shape=current_processed_input_memmap.shape)
        memmap_registry['normalized'] = (normalized_memmap, normalized_path, normalized_temp_dir)
        
        z_high_percentile_values = []
        target_intensity = 1.0
        chunk_size_norm_read = min(50, current_processed_input_memmap.shape[0]) if current_processed_input_memmap.shape[0] > 0 else 1
        for z_start in tqdm(range(0, current_processed_input_memmap.shape[0], chunk_size_norm_read), desc="  Calc z-stats"):
            z_end = min(z_start + chunk_size_norm_read, current_processed_input_memmap.shape[0])
            if z_start >= z_end: continue
            
            current_chunk = current_processed_input_memmap[z_start:z_end]
            finite_chunk_values = current_chunk[np.isfinite(current_chunk)] # For overall check
            
            if finite_chunk_values.size == 0: # All values in chunk are NaN/inf
                 for z_idx in range(z_start, z_end): z_high_percentile_values.append((z_idx, 0.0))
                 del current_chunk; gc.collect()
                 continue
            
            for i_slice, z_val in enumerate(range(z_start, z_end)):
                plane = current_chunk[i_slice]
                finite_plane_values = plane[np.isfinite(plane)].ravel()
                if finite_plane_values.size == 0:
                    z_high_percentile_values.append((z_val, 0.0))
                    continue
                
                samples_count = min(100000, finite_plane_values.size)
                # Ensure random.choice gets a non-empty array if samples_count is 0
                samples = np.random.choice(finite_plane_values, samples_count, replace=False) if samples_count > 0 else np.array([])
                
                try:
                    high_perc_value = np.percentile(samples, high_threshold_percentile) if samples.size > 0 else 0.0
                except IndexError: # Should be rare if samples.size check is done
                    high_perc_value = 0.0
                
                if not np.isfinite(high_perc_value): high_perc_value = 0.0
                z_high_percentile_values.append((z_val, float(high_perc_value)))
            del current_chunk, finite_chunk_values; gc.collect()

        z_stats_dict = dict(z_high_percentile_values)
        chunk_size_norm_write = min(50, current_processed_input_memmap.shape[0]) if current_processed_input_memmap.shape[0] > 0 else 1
        for z_start in tqdm(range(0, current_processed_input_memmap.shape[0], chunk_size_norm_write), desc="  Normalize z-planes"):
             z_end = min(z_start + chunk_size_norm_write, current_processed_input_memmap.shape[0])
             if z_start >= z_end: continue

             current_chunk_to_normalize = current_processed_input_memmap[z_start:z_end]
             normalized_chunk_data = np.zeros_like(current_chunk_to_normalize, dtype=np.float32)
             for i_slice, z_val in enumerate(range(z_start, z_end)):
                 current_slice_data = current_chunk_to_normalize[i_slice]
                 high_perc_value = z_stats_dict.get(z_val, 0.0)
                 
                 if not np.any(np.isfinite(current_slice_data)): # If slice is all NaN/inf
                     normalized_chunk_data[i_slice] = 0.0
                     continue
                 
                 if high_perc_value > 1e-9: # Use a small epsilon for division
                     scale_factor = target_intensity / high_perc_value
                     normalized_chunk_data[i_slice] = (current_slice_data * scale_factor).astype(np.float32)
                 else:
                     normalized_chunk_data[i_slice] = current_slice_data.astype(np.float32) # Already float32
             normalized_memmap[z_start:z_end] = normalized_chunk_data
             del current_chunk_to_normalize, normalized_chunk_data; gc.collect()
        normalized_memmap.flush(); print("Normalization finished.")
        name = 'processed_input'; mm, p, d = memmap_registry.pop(name); del mm; gc.collect(); os.unlink(p); rmtree(d, ignore_errors=True)

        # --- Step 3: Thresholding & Calc Thresholds ---
        print("\nStep 3: Thresholding and Calculating Thresholds...")
        binary_temp_dir = tempfile.mkdtemp(prefix="binary_")
        binary_path = os.path.join(binary_temp_dir, 'b.dat')
        binary_memmap = np.memmap(binary_path, dtype=bool, mode='w+', shape=original_shape)
        memmap_registry['binary'] = (binary_memmap, binary_path, binary_temp_dir)
        
        sample_size = min(5_000_000, normalized_memmap.size)
        collected_count = 0;
        chunk_size_thresh_sample = min(200, normalized_memmap.shape[0]) if normalized_memmap.shape[0] > 0 else 1
        indices_z = np.arange(normalized_memmap.shape[0]); np.random.shuffle(indices_z)
        all_samples_list = []

        for z_start_idx_loop in tqdm(range(0, len(indices_z), chunk_size_thresh_sample), desc="  Sampling Norm. Vol."):
            if collected_count >= sample_size: break
            z_indices_chunk = indices_z[z_start_idx_loop : z_start_idx_loop + chunk_size_thresh_sample]
            if not z_indices_chunk.size: continue

            slices_data = normalized_memmap[z_indices_chunk, :, :]
            finite_samples_in_chunk = slices_data[np.isfinite(slices_data)].ravel()
            del slices_data; gc.collect()

            samples_to_take_this_round = min(len(finite_samples_in_chunk), sample_size - collected_count)
            if samples_to_take_this_round > 0:
                chosen_samples = np.random.choice(finite_samples_in_chunk, samples_to_take_this_round, replace=False)
                all_samples_list.append(chosen_samples)
                collected_count += samples_to_take_this_round
            del finite_samples_in_chunk; gc.collect()
        
        if all_samples_list:
            all_samples_np = np.concatenate(all_samples_list); del all_samples_list; gc.collect()
            if all_samples_np.size > 0:
                 try: segmentation_threshold = float(np.percentile(all_samples_np, low_threshold_percentile))
                 except IndexError:
                    print(" Warn: No valid samples for threshold calculation (IndexError). Defaulting to 0.0."); segmentation_threshold = 0.0
            else: print(" Warn: No samples for threshold after concatenation. Defaulting to 0.0."); segmentation_threshold = 0.0
            if 'all_samples_np' in locals(): del all_samples_np; gc.collect()
        else: print(" Warn: No samples collected for thresholding. Defaulting to 0.0."); segmentation_threshold = 0.0
        print(f"  Segmentation threshold ({low_threshold_percentile:.1f} perc): {segmentation_threshold:.4f}")

        chunk_size_thresh_apply = min(100, original_shape[0]) if original_shape[0] > 0 else 1
        for i_thresh in tqdm(range(0, original_shape[0], chunk_size_thresh_apply), desc="  Applying Threshold"):
             end_idx_thresh = min(i_thresh + chunk_size_thresh_apply, original_shape[0])
             if i_thresh >= end_idx_thresh: continue
             if hasattr(normalized_memmap, '_mmap') and normalized_memmap._mmap is not None:
                  norm_chunk = normalized_memmap[i_thresh:end_idx_thresh]
                  binary_memmap[i_thresh:end_idx_thresh] = norm_chunk > segmentation_threshold
                  del norm_chunk; gc.collect()
             else:
                  print(f"Error: 'normalized' memmap closed or invalid during thresholding chunk {i_thresh}. Filling with False.");
                  binary_memmap[i_thresh:end_idx_thresh] = False
        binary_memmap.flush(); print("Thresholding done.")
        name = 'normalized'; mm, p, d = memmap_registry.pop(name); del mm; gc.collect(); os.unlink(p); rmtree(d, ignore_errors=True)

        # --- Step 4: Connect ---
        print("\nStep 4: Connecting fragments...")
        connected_binary_memmap, connected_temp_dir = connect_fragmented_processes(
            binary_memmap, spacing=spacing_float, max_gap_physical=connect_max_gap_physical # Use spacing_float
        )
        memmap_registry['connected'] = (connected_binary_memmap, connected_binary_memmap.filename, connected_temp_dir)
        print("Connection memmap created.")
        name = 'binary'; mm, p, d = memmap_registry.pop(name); del mm; gc.collect(); os.unlink(p); rmtree(d, ignore_errors=True)

        # --- Step 5: Clean Small Objects ---
        print(f"\nStep 5: Cleaning objects < {min_size_voxels} voxels...")
        cleaned_binary_dir=tempfile.mkdtemp(prefix="cleaned_")
        cleaned_binary_path=os.path.join(cleaned_binary_dir, 'c.dat')
        cleaned_binary_memmap=np.memmap(cleaned_binary_path, dtype=bool, mode='w+', shape=original_shape)
        memmap_registry['cleaned'] = (cleaned_binary_memmap, cleaned_binary_path, cleaned_binary_dir)
        
        connected_memmap_obj = memmap_registry['connected'][0]
        try:
            print("  Attempting to load connected mask into memory for small object removal.")
            connected_in_memory = np.array(connected_memmap_obj)
            print(f"  Connected mask loaded ({connected_in_memory.nbytes / 1024**2:.2f} MB). Removing small objects.")
            remove_small_objects(connected_in_memory, min_size=min_size_voxels, connectivity=1, out=cleaned_binary_memmap)
            del connected_in_memory; gc.collect()
            print("  Small object removal from in-memory array complete, written to memmap.")
        except MemoryError:
            print("  MemoryError loading full connected mask. Fallback: Copying connected mask (small object removal skipped).")
            chunk_size_copy_clean = min(100, original_shape[0]) if original_shape[0] > 0 else 1
            for i_clean in tqdm(range(0, original_shape[0], chunk_size_copy_clean), desc="  Copying Connected (MemFallback)"):
                start_idx_clean = i_clean; end_idx_clean = min(i_clean + chunk_size_copy_clean, original_shape[0])
                if start_idx_clean >= end_idx_clean: continue
                cleaned_binary_memmap[start_idx_clean:end_idx_clean] = connected_memmap_obj[start_idx_clean:end_idx_clean]
        cleaned_binary_memmap.flush(); print("Cleaning step done.")
        name = 'connected'; mm, p, d = memmap_registry.pop(name); del mm; gc.collect(); os.unlink(p); rmtree(d, ignore_errors=True)

        # --- Step 6: Label ---
        print("\nStep 6: Labeling components...")
        labels_temp_dir = tempfile.mkdtemp(prefix="labels_")
        labels_path = os.path.join(labels_temp_dir, 'l.dat')
        first_pass_segmentation_memmap=np.memmap(labels_path, dtype=np.int32, mode='w+', shape=original_shape)
        
        cleaned_memmap_obj = memmap_registry['cleaned'][0]
        print("  Applying ndimage.label directly to memmap...")
        num_features = ndimage.label(cleaned_memmap_obj, structure=generate_binary_structure(3, 1), output=first_pass_segmentation_memmap)
        first_pass_segmentation_memmap.flush()
        print(f"Labeling done ({num_features} features found). Output memmap: {labels_path}")
        del first_pass_segmentation_memmap; gc.collect()
        name = 'cleaned'; mm, p, d = memmap_registry.pop(name); del mm; gc.collect(); os.unlink(p); rmtree(d, ignore_errors=True)

        print(f"\n--- Raw First Pass Segmentation Finished ---")
        final_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"Final memory usage: {final_mem:.2f} MB")
        return labels_path, labels_temp_dir, segmentation_threshold, first_pass_params

    except Exception as e:
        print(f"\n!!! ERROR during Raw First Pass Segmentation: {e} !!!")
        traceback.print_exc()
        raise
    finally:
        print("\nFinal cleanup check for raw pass...")
        registry_keys = list(memmap_registry.keys())
        for name_key in registry_keys:
             mm_obj, p_path, d_dir = memmap_registry.pop(name_key)
             print(f"  Cleaning up leftover {name_key} (Path: {p_path})...")
             if hasattr(mm_obj, '_mmap') and mm_obj._mmap is not None: del mm_obj; gc.collect()
             if p_path and os.path.exists(p_path):
                 try: os.unlink(p_path)
                 except Exception as e_unlink: print(f"    Error unlinking {p_path}: {e_unlink}")
             if d_dir and os.path.exists(d_dir):
                 try: rmtree(d_dir, ignore_errors=True)
                 except Exception as e_rmtree: print(f"    Error removing dir {d_dir}: {e_rmtree}")
        gc.collect()

# --- END OF FILE initial_3d_segmentation.py ---