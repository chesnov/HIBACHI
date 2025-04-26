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
import os
import tempfile
import gc
import numpy as np
from tqdm import tqdm
from functools import partial
import math
seed = 42
np.random.seed(seed)         # For NumPy
import traceback

from .remove_artifacts import *

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
        input_memmap = np.memmap(input_path, dtype=input_dtype, mode='r', shape=input_shape)
        output_memmap = np.memmap(output_path, dtype=output_dtype, mode='r+', shape=output_shape)
        slice_data = input_memmap[z, :, :].copy()
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

def enhance_tubular_structures_slice_by_slice(volume, scales, spacing, black_ridges=False, frangi_alpha=0.5, frangi_beta=0.5, frangi_gamma=15, apply_3d_smoothing=True, smoothing_sigma_phys=0.5, ram_safety_factor=0.8, mem_factor_per_slice=8.0 ):
    """ Enhance tubular structures slice-by-slice in parallel. Returns a MEMMAP object."""
    # ... (Initial setup, smoothing, parallel processing logic - remains the same) ...
    # --- Start of changes near the end of the function ---
    print(f"Starting slice-by-slice (2.5D) tubular enhancement with PARALLEL processing...")
    print(f"  Volume shape: {volume.shape}, Spacing: {spacing}")
    print(f"  Initial memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    spacing = tuple(float(s) for s in spacing); volume_shape = volume.shape
    slice_shape = volume_shape[1:]; num_slices = volume_shape[0]; xy_spacing = spacing[1:]
    input_volume_memmap = None; input_memmap_path = None; input_memmap_dir = None; source_volume_cleaned = False
    temp_dirs_to_clean = [] # Keep track of temp dirs

    if apply_3d_smoothing and smoothing_sigma_phys > 0:
        print(f"Applying initial 3D smoothing...")
        sigma_voxel_3d = tuple(smoothing_sigma_phys / s for s in spacing)
        smooth_temp_dir = tempfile.mkdtemp(prefix="pre_smooth_"); temp_dirs_to_clean.append(smooth_temp_dir)
        smooth_path = os.path.join(smooth_temp_dir, 'smoothed_3d.dat')
        smoothed_dtype = np.float32 if not np.issubdtype(volume.dtype, np.floating) else volume.dtype
        input_volume_memmap = np.memmap(smooth_path, dtype=smoothed_dtype, mode='w+', shape=volume_shape)
        input_memmap_path = smooth_path; input_memmap_dir = smooth_temp_dir
        chunk_size_z_smooth = min(50, volume_shape[0]); overlap_z_smooth = math.ceil(3 * sigma_voxel_3d[0])
        for i in tqdm(range(0, volume_shape[0], chunk_size_z_smooth), desc="3D Pre-Smoothing"):
            start_read = max(0, i - overlap_z_smooth); end_read = min(volume_shape[0], i + chunk_size_z_smooth + overlap_z_smooth)
            start_write = i; end_write = min(volume_shape[0], i + chunk_size_z_smooth)
            local_write_start = start_write - start_read; local_write_end = end_write - start_read
            if start_read >= end_read: continue
            chunk = volume[start_read:end_read].astype(smoothed_dtype).copy()
            smoothed_chunk = gaussian_filter(chunk, sigma=sigma_voxel_3d, mode='reflect')
            if local_write_end > local_write_start: input_volume_memmap[start_write:end_write, :, :] = smoothed_chunk[local_write_start:local_write_end, :, :]
            del chunk, smoothed_chunk; gc.collect()
        input_volume_memmap.flush(); 
        print(f"  3D pre-smoothing done.")
    else:
        print("  Skipping initial 3D smoothing.");
        if isinstance(volume, np.memmap): input_volume_memmap = volume; input_memmap_path = volume.filename
        else:
            print(f"  Converting input volume to temporary memmap..."); input_memmap_dir = tempfile.mkdtemp(prefix="input_volume_"); temp_dirs_to_clean.append(input_memmap_dir)
            input_volume_memmap, input_memmap_path, _ = create_memmap(data=volume, directory=input_memmap_dir)
            source_volume_cleaned = True # Mark this temp dir for cleaning later

    print(f"xy spacing is: {xy_spacing}, scales: {scales}")
    avg_xy_spacing = np.mean(xy_spacing); sigmas_voxel_2d = sorted([s / avg_xy_spacing for s in scales])
    print(f"  Using 2D voxel sigmas: {sigmas_voxel_2d}")
    output_temp_dir = tempfile.mkdtemp(prefix='tubular_enhance_parallel_'); temp_dirs_to_clean.append(output_temp_dir) # Track this dir
    output_path = os.path.join(output_temp_dir, 'enhanced_volume_parallel.dat')
    output_dtype = np.float32; output_memmap = np.memmap(output_path, dtype=output_dtype, mode='w+', shape=volume_shape)
    print(f"  Output memmap created: {output_path}")
    try:
        # ... (Resource check logic remains same) ...
        total_cores = os.cpu_count(); max_cpu_workers = max(1, total_cores - 1 if total_cores else 1)
        available_ram = psutil.virtual_memory().available; usable_ram = available_ram * ram_safety_factor
        input_dtype_bytes = np.dtype(input_volume_memmap.dtype).itemsize if input_volume_memmap is not None else np.dtype(np.float32).itemsize
        slice_mem_bytes = slice_shape[0] * slice_shape[1] * input_dtype_bytes; estimated_worker_ram = slice_mem_bytes * mem_factor_per_slice
        if estimated_worker_ram <= 0: estimated_worker_ram = 1 # Avoid division by zero or negative
        max_mem_workers = max(1, int(usable_ram // estimated_worker_ram)) if estimated_worker_ram > 0 else 1
        num_workers = min(max_cpu_workers, max_mem_workers, num_slices)
        print(f"  Resource Check: Cores={total_cores}, Avail RAM={available_ram / (1024**3):.2f}GB, Use RAM={usable_ram / (1024**3):.2f}GB")
        print(f"  Est. RAM/worker={estimated_worker_ram / (1024**2):.2f}MB -> Max RAM Workers={max_mem_workers}")
        print(f"  Using {num_workers} parallel workers.")
    except Exception as e: print(f"  Warning: Could not determine resources automatically ({e}). Defaulting to 1 worker."); num_workers = 1

    start_time = time.time(); pool = None
    try:
        input_info = (input_memmap_path, input_volume_memmap.shape, input_volume_memmap.dtype); output_info = (output_path, output_memmap.shape, output_dtype)
        worker_func_partial = partial(_process_slice_worker, input_memmap_info=input_info, output_memmap_info=output_info, sigmas_voxel_2d=sigmas_voxel_2d, black_ridges=black_ridges, frangi_alpha=frangi_alpha, frangi_beta=frangi_beta, frangi_gamma=frangi_gamma)
        print(f"Processing {num_slices} slices using {num_workers} workers...")
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(worker_func_partial, range(num_slices)), total=num_slices, desc="Applying 2D Filters (Parallel)"))
        errors = [r for r in results if isinstance(r, str) and r.startswith("Error_slice_")]
        if errors: print(f"\n!!! Encountered {len(errors)} errors: {errors[:5]}"); raise RuntimeError(f"Parallel slice processing failed.")
        print(f"Parallel processing finished in {time.time() - start_time:.2f} seconds.")
    except Exception as e: print(f"An error occurred during parallel processing: {e}"); import traceback; traceback.print_exc(); raise
    finally:
        # Don't delete output_memmap here, it's the result we want to return
        # Close the input memmap if it was opened
        if input_volume_memmap is not None and hasattr(input_volume_memmap, '_mmap'):
             input_volume_memmap.flush() # Ensure writes are flushed if 'w+'
             del input_volume_memmap # Close the memmap object
        gc.collect()

    # Clean up temporary input/smoothing directories if they were created
    if source_volume_cleaned and input_memmap_dir and input_memmap_dir in temp_dirs_to_clean:
        print(f"Cleaning up temporary input volume memmap: {input_memmap_dir}");
        try: rmtree(input_memmap_dir, ignore_errors=True); temp_dirs_to_clean.remove(input_memmap_dir)
        except Exception as e: print(f"Warning: Could not delete temp input directory {input_memmap_dir}: {e}")
    elif apply_3d_smoothing and input_memmap_dir and input_memmap_dir in temp_dirs_to_clean:
         print(f"Cleaning up temporary pre-smoothed volume: {input_memmap_dir}");
         try: rmtree(input_memmap_dir, ignore_errors=True); temp_dirs_to_clean.remove(input_memmap_dir)
         except Exception as e: print(f"Warning: Could not delete temp smooth directory {input_memmap_dir}: {e}")

    # --- DO NOT load into memory here ---
    # final_enhanced_memmap = np.memmap(output_path, dtype=output_dtype, mode='r', shape=volume_shape)
    # result_in_memory = np.array(final_enhanced_memmap)
    # del final_enhanced_memmap; gc.collect();

    print(f"Enhanced volume generated as memmap: {output_path}")
    print(f"Memory usage after slice-wise enhancement (before returning memmap): {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    # Return the memmap object itself, its path, and the directory containing it
    return output_memmap, output_path, output_temp_dir # Modified return

# Connection function (remains the same)
def connect_fragmented_processes(binary_mask, spacing, max_gap_physical=1.0):
    """Connect fragmented processes using anisotropic morphological closing."""
    print(f"Connecting fragmented processes with max physical gap: {max_gap_physical}")
    print(f"  Input mask shape: {binary_mask.shape}, Spacing: {spacing}")
    radius_vox = [math.ceil((max_gap_physical / 2) / s) for s in spacing]
    structure_shape = tuple(2 * r + 1 for r in radius_vox)
    print(f"  Calculated anisotropic structure shape (voxels): {structure_shape}")
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

# === NEW Function for Raw Segmentation (Steps 1-6) ===
def segment_microglia_first_pass_raw(
    volume, # Original intensity volume
    spacing,
    tubular_scales=[0.5, 1.0, 2.0, 3.0],
    smooth_sigma=0.5,
    connect_max_gap_physical=1.0,
    min_size_voxels=50,
    low_threshold_percentile=25.0,
    high_threshold_percentile=95.0
    ):
    """
    Performs initial segmentation steps 1-6 (Enhance, Normalize, Threshold,
    Connect, Clean, Label) without edge trimming. Uses memmaps.

    Returns:
    --------
    labels_path : str
        Path to the final labeled segmentation memmap file.
    labels_temp_dir : str
        Path to the temporary directory containing the labels memmap.
    segmentation_threshold : float
        The calculated absolute threshold used for segmentation.
    global_brightness_cutoff : float
        The calculated brightness cutoff derived from the threshold.
    first_pass_params : dict
        Dictionary containing parameters used.
    """
    # --- Initial Setup ---
    low_threshold_percentile = max(0.0, min(100.0, low_threshold_percentile))
    high_threshold_percentile = max(0.0, min(100.0, high_threshold_percentile))

    original_shape = volume.shape
    spacing = tuple(float(s) for s in spacing)
    # Store only params used in this function
    first_pass_params = {
        'spacing': spacing, 'tubular_scales': tubular_scales, 'smooth_sigma': smooth_sigma,
        'connect_max_gap_physical': connect_max_gap_physical, 'min_size_voxels': min_size_voxels,
        'low_threshold_percentile': low_threshold_percentile,
        'high_threshold_percentile': high_threshold_percentile,
        'original_shape': original_shape
        }
    print(f"\n--- Starting Raw First Pass Segmentation ---")
    initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_mem:.2f} MB")

    memmap_registry = {}
    global_brightness_cutoff = np.inf # Default if calculation fails
    segmentation_threshold = 0.0     # Default if calculation fails

    try:
        # --- Step 1: Enhance ---
        print("\nStep 1: Enhancing structures...")
        enhanced_memmap, enhance_path, enhance_temp_dir = enhance_tubular_structures_slice_by_slice(
            volume, scales=tubular_scales, spacing=spacing,
            apply_3d_smoothing=(smooth_sigma > 0), smoothing_sigma_phys=smooth_sigma
        )
        memmap_registry['enhanced'] = (enhanced_memmap, enhance_path, enhance_temp_dir)
        print("Enhancement memmap created.")
        gc.collect()

        # --- Step 2: Normalize ---
        print("\nStep 2: Normalizing signal...")
        normalized_temp_dir = tempfile.mkdtemp(prefix="normalize_")
        normalized_path = os.path.join(normalized_temp_dir, 'normalized.dat')
        normalized_memmap = np.memmap(normalized_path, dtype=np.float32, mode='w+', shape=enhanced_memmap.shape)
        memmap_registry['normalized'] = (normalized_memmap, normalized_path, normalized_temp_dir)
        # ... (Normalization logic using z_high_percentile_values - same as before) ...
        z_high_percentile_values = []
        target_intensity = 1.0
        chunk_size_norm_read = min(50, enhanced_memmap.shape[0]) if enhanced_memmap.shape[0] > 0 else 1
        for z_start in tqdm(range(0, enhanced_memmap.shape[0], chunk_size_norm_read), desc="  Calc z-stats"):
            z_end = min(z_start + chunk_size_norm_read, enhanced_memmap.shape[0])
            enhanced_chunk = enhanced_memmap[z_start:z_end]; finite_chunk = enhanced_chunk[np.isfinite(enhanced_chunk)]
            if finite_chunk.size == 0: # Handle all-NaN/inf chunks
                 for z in range(z_start, z_end): z_high_percentile_values.append((z, 0.0))
                 continue
            for i, z in enumerate(range(z_start, z_end)):
                plane = enhanced_chunk[i]; finite_plane = plane[np.isfinite(plane)].ravel()
                if finite_plane.size == 0: z_high_percentile_values.append((z, 0.0)); continue
                samples = np.random.choice(finite_plane, min(100000, finite_plane.size), replace=False)
                try: high_perc_value = np.percentile(samples, high_threshold_percentile) if samples.size > 0 else 0.0
                except: high_perc_value = 0.0
                if not np.isfinite(high_perc_value): high_perc_value = 0.0
                z_high_percentile_values.append((z, float(high_perc_value)))
            del enhanced_chunk, finite_chunk; gc.collect()
        # Apply normalization...
        z_stats_dict = dict(z_high_percentile_values)
        chunk_size_norm_write = min(50, enhanced_memmap.shape[0]) if enhanced_memmap.shape[0] > 0 else 1
        for z_start in tqdm(range(0, enhanced_memmap.shape[0], chunk_size_norm_write), desc="  Normalize z-planes"):
             z_end = min(z_start + chunk_size_norm_write, enhanced_memmap.shape[0])
             enhanced_chunk = enhanced_memmap[z_start:z_end]
             normalized_chunk = np.zeros_like(enhanced_chunk, dtype=np.float32)
             for i, z in enumerate(range(z_start, z_end)):
                 current_slice = enhanced_chunk[i]; high_perc_value = z_stats_dict.get(z, 0.0)
                 if current_slice is None or not np.any(np.isfinite(current_slice)): normalized_chunk[i] = 0.0; continue
                 if high_perc_value > 1e-6: scale_factor = target_intensity / high_perc_value; normalized_chunk[i] = (current_slice * scale_factor).astype(np.float32)
                 else: normalized_chunk[i] = current_slice.astype(np.float32)
             normalized_memmap[z_start:z_end] = normalized_chunk; del enhanced_chunk, normalized_chunk; gc.collect()
        normalized_memmap.flush(); print("Normalization finished.")
        # Cleanup enhanced
        name = 'enhanced'; mm, p, d = memmap_registry.pop(name); del mm; gc.collect(); os.unlink(p); rmtree(d, ignore_errors=True)

        # --- Step 3: Thresholding & Calc Thresholds ---
        print("\nStep 3: Thresholding and Calculating Thresholds...")
        binary_temp_dir = tempfile.mkdtemp(prefix="binary_")
        binary_path = os.path.join(binary_temp_dir, 'b.dat')
        binary_memmap = np.memmap(binary_path, dtype=bool, mode='w+', shape=original_shape)
        memmap_registry['binary'] = (binary_memmap, binary_path, binary_temp_dir)
        # ... (Sampling logic to calculate segmentation_threshold - same as before) ...
        sample_size = min(5_000_000, normalized_memmap.size)
        samples_collected = []; collected_count = 0; num_samples_needed = sample_size
        chunk_size_thresh_sample = min(200, normalized_memmap.shape[0]) if normalized_memmap.shape[0] > 0 else 1
        indices_z = np.arange(normalized_memmap.shape[0]); np.random.shuffle(indices_z)
        all_samples = None
        for z_start_idx in tqdm(range(0, len(indices_z), chunk_size_thresh_sample), desc="  Sampling Norm. Vol."):
            if collected_count >= num_samples_needed: break
            z_indices_chunk = indices_z[z_start_idx : z_start_idx + chunk_size_thresh_sample]
            slices_data = normalized_memmap[z_indices_chunk, :, :]; finite_samples_in_chunk = slices_data[np.isfinite(slices_data)].ravel(); del slices_data; gc.collect()
            samples_to_take = min(len(finite_samples_in_chunk), num_samples_needed - collected_count)
            if samples_to_take > 0: samples_collected.append(np.random.choice(finite_samples_in_chunk, samples_to_take, replace=False)); collected_count += samples_to_take
            del finite_samples_in_chunk; gc.collect()
        if samples_collected:
            all_samples = np.concatenate(samples_collected); del samples_collected; gc.collect()
            if all_samples.size > 0:
                 try: segmentation_threshold = float(np.percentile(all_samples, low_threshold_percentile)) # Ensure float
                 except Exception as e: print(f" Warn: Percentile failed: {e}. Using median."); segmentation_threshold = float(np.median(all_samples))
            else: print(" Warn: No samples for threshold."); segmentation_threshold = 0.0
        else: print(" Warn: No samples collected."); segmentation_threshold = 0.0
        print(f"  Segmentation threshold ({low_threshold_percentile:.1f} perc): {segmentation_threshold:.4f}")
        # Calculate brightness cutoff based on segmentation threshold (will be returned)
        # The factor itself is passed later during trimming step
        # global_brightness_cutoff = segmentation_threshold * brightness_cutoff_factor
        # print(f"  Potential Brightness Cutoff (factor {brightness_cutoff_factor:.2f}): {global_brightness_cutoff:.4f}") # Just informational
        if all_samples is not None: del all_samples; gc.collect()
        # Apply threshold
        chunk_size_thresh_apply = min(100, original_shape[0]) if original_shape[0] > 0 else 1
        for i in tqdm(range(0, original_shape[0], chunk_size_thresh_apply), desc="  Applying Threshold"):
             end_idx = min(i + chunk_size_thresh_apply, original_shape[0])
             if hasattr(normalized_memmap, '_mmap') and normalized_memmap._mmap is not None:
                  norm_chunk = normalized_memmap[i:end_idx]; binary_memmap[i:end_idx] = norm_chunk > segmentation_threshold; del norm_chunk; gc.collect()
             else: print(f"Error: 'normalized' memmap closed during thresholding chunk {i}"); binary_memmap[i:end_idx] = False # Handle error case
        binary_memmap.flush(); print("Thresholding done.")
        # Cleanup normalized
        name = 'normalized'; mm, p, d = memmap_registry.pop(name); del mm; gc.collect(); os.unlink(p); rmtree(d, ignore_errors=True)

        # --- Step 4: Connect ---
        print("\nStep 4: Connecting fragments...")
        connected_binary_memmap, connected_temp_dir = connect_fragmented_processes(
            binary_memmap, spacing=spacing, max_gap_physical=connect_max_gap_physical
        )
        memmap_registry['connected'] = (connected_binary_memmap, connected_binary_memmap.filename, connected_temp_dir)
        print("Connection memmap created.")
        # Cleanup binary
        name = 'binary'; mm, p, d = memmap_registry.pop(name); del mm; gc.collect(); os.unlink(p); rmtree(d, ignore_errors=True)

        # --- Step 5: Clean Small Objects ---
        print(f"\nStep 5: Cleaning objects < {min_size_voxels} voxels...")
        cleaned_binary_dir=tempfile.mkdtemp(prefix="cleaned_")
        cleaned_binary_path=os.path.join(cleaned_binary_dir, 'c.dat')
        cleaned_binary_memmap=np.memmap(cleaned_binary_path, dtype=bool, mode='w+', shape=original_shape)
        memmap_registry['cleaned'] = (cleaned_binary_memmap, cleaned_binary_path, cleaned_binary_dir)
        connected_memmap_obj = memmap_registry['connected'][0]
        try: # Try loading into memory for speed if possible
            connected_in_memory = np.array(connected_memmap_obj)
            remove_small_objects(connected_in_memory, min_size=min_size_voxels, connectivity=1, out=cleaned_binary_memmap)
            del connected_in_memory; gc.collect()
        except MemoryError: # Fallback to chunking if needed (more complex, omitted for brevity, copy used before)
            print("  MemoryError loading connected mask. Skipping small object removal for now (can be reapplied later).")
            print("  Copying connected mask to cleaned mask..."); chunk_size_copy = min(100, original_shape[0]) if original_shape[0] > 0 else 1
            for i in tqdm(range(0, original_shape[0], chunk_size_copy), desc="  Copying Connected"): cleaned_binary_memmap[i:min(i+chunk_size_copy, original_shape[0])] = connected_memmap_obj[i:min(i+chunk_size_copy, original_shape[0])]
        cleaned_binary_memmap.flush(); print("Cleaning step done.")
        # Cleanup connected
        name = 'connected'; mm, p, d = memmap_registry.pop(name); del mm; gc.collect(); os.unlink(p); rmtree(d, ignore_errors=True)

        # --- Step 6: Label ---
        print("\nStep 6: Labeling components...")
        labels_temp_dir = tempfile.mkdtemp(prefix="labels_")
        labels_path = os.path.join(labels_temp_dir, 'l.dat') # This path will be returned
        first_pass_segmentation_memmap=np.memmap(labels_path, dtype=np.int32, mode='w+', shape=original_shape)
        # Don't add labels to registry here, we return its path/dir
        cleaned_memmap_obj = memmap_registry['cleaned'][0]
        num_features = ndimage.label(cleaned_memmap_obj, structure=generate_binary_structure(3, 1), output=first_pass_segmentation_memmap)
        first_pass_segmentation_memmap.flush()
        print(f"Labeling done ({num_features} features found). Output memmap: {labels_path}")
        # IMPORTANT: Close the labels memmap handle here so the file can be used by the next step
        del first_pass_segmentation_memmap; gc.collect()
        # Cleanup cleaned
        name = 'cleaned'; mm, p, d = memmap_registry.pop(name); del mm; gc.collect(); os.unlink(p); rmtree(d, ignore_errors=True)

        # --- Finish ---
        print(f"\n--- Raw First Pass Segmentation Finished ---")
        final_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"Final memory usage: {final_mem:.2f} MB")

        # Return path to labeled memmap and calculated thresholds
        return labels_path, labels_temp_dir, segmentation_threshold, first_pass_params

    except Exception as e:
        print(f"\n!!! ERROR during Raw First Pass Segmentation: {e} !!!")
        traceback.print_exc()
        # Ensure cleanup is attempted on error
        raise e # Re-raise the error
    finally:
        # Final cleanup of any remaining tracked memmaps (should be none ideally)
        print("\nFinal cleanup check for raw pass...")
        registry_keys = list(memmap_registry.keys())
        for name in registry_keys:
             mm, p, d = memmap_registry.pop(name)
             print(f"  Cleaning up leftover {name} (Path: {p})...")
             if hasattr(mm, '_mmap') and mm._mmap is not None: del mm; gc.collect()
             if p and os.path.exists(p): os.unlink(p)
             if d and os.path.exists(d): rmtree(d, ignore_errors=True)
        gc.collect()


def apply_hull_trimming(
    raw_labels_path,          # Path to the memmap from previous step
    original_volume,          # Original intensity volume (can be memmap or array)
    spacing,
    # hull_opening_radius_phys, # Parameter NOT USED by generate_hull_boundary_and_stack
    # hull_closing_radius_phys, # Parameter NOT USED by generate_hull_boundary_and_stack
    hull_boundary_thickness_phys, # USED to calculate erosion iterations
    edge_trim_distance_threshold,
    brightness_cutoff_factor, # Factor to apply to seg_threshold
    segmentation_threshold,   # Absolute threshold calculated previously
    min_size_voxels,          # For re-cleaning after trimming
    edge_distance_chunk_size_z = 32, # For trim_object_edges fallback
    smoothing_iterations = 1, # For generate_hull_boundary_and_stack
    heal_iterations = 1       # For trim_object_edges_by_distance
    ):
    """
    Applies hull generation and edge trimming to a raw labeled segmentation.
    MODIFIES the data pointed to by raw_labels_path if it's writable, OR
    creates a new temporary memmap for the output.

    Args:
        raw_labels_path (str): Path to the labeled segmentation memmap (.dat file).
                               MUST BE WRITABLE (mode='r+' or 'w+') for in-place modification.
        original_volume (np.ndarray or np.memmap): Original intensity data.
        spacing (tuple): Voxel spacing (z, y, x).
        hull_boundary_thickness_phys (float): Physical thickness for hull boundary.
        edge_trim_distance_threshold (float): Physical distance for trimming.
        brightness_cutoff_factor (float): Factor to multiply segmentation_threshold by.
        segmentation_threshold (float): The absolute threshold used previously.
        min_size_voxels (int): Minimum size for final object cleaning.
        edge_distance_chunk_size_z (int): Chunk size for distance calc.
        smoothing_iterations (int): Iterations for hull smoothing.
        heal_iterations (int): Iterations for grey closing healing.


    Returns:
    --------
    trimmed_labels_path : str
        Path to the memmap file containing the trimmed labels (this will be the
        same as raw_labels_path if modified in-place, or a new temp path).
    trimmed_labels_temp_dir : str or None
        Directory containing the trimmed labels memmap ONLY IF a new temp file
        was created. None if modified in-place.
    hull_boundary_mask : np.ndarray (bool)
        The calculated hull boundary mask (in memory).
    """
    print(f"\n--- Applying Hull Generation and Edge Trimming (Outputting New File) ---")
    print(f"Params: hull_thick={hull_boundary_thickness_phys}, smoothing_iters={smoothing_iterations}, heal_iters={heal_iterations}, trim_dist={edge_trim_distance_threshold:.2f}, "
          f"bright_factor={brightness_cutoff_factor:.2f}, seg_thresh={segmentation_threshold:.4f}")
    initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage for trimming: {initial_mem:.2f} MB")

    # --- Get shape and validate input ---
    if not os.path.exists(raw_labels_path):
        print(f"Error: Input raw labels file not found: {raw_labels_path}")
        return None, None, None
    # Infer shape from input file (assuming headerless .dat needs original_shape)
    # It's safer if original_shape is passed or read from metadata if possible.
    # For now, assume original_volume has the correct shape.
    if original_volume is None:
        print("Error: original_volume is required to get shape.")
        return None, None, None
    original_shape = original_volume.shape

    spacing = tuple(float(s) for s in spacing)
    hull_boundary_mask = np.zeros(original_shape, dtype=bool) # Default

    global_brightness_cutoff = segmentation_threshold * brightness_cutoff_factor if brightness_cutoff_factor > 0 else np.inf
    print(f"  Using Global Brightness Cutoff: {global_brightness_cutoff:.4f}")

    # --- Create a *NEW* temporary directory and memmap for the output ---
    trimmed_labels_temp_dir = None
    trimmed_labels_path = None
    trimmed_labels_memmap = None # Handle for the new output memmap

    try:
        trimmed_labels_temp_dir = tempfile.mkdtemp(prefix="trimmed_labels_")
        trimmed_labels_path = os.path.join(trimmed_labels_temp_dir, 'trimmed_l.dat')
        print(f"  Creating NEW output memmap for trimmed labels: {trimmed_labels_path}")
        trimmed_labels_memmap = np.memmap(trimmed_labels_path, dtype=np.int32, mode='w+', shape=original_shape)

        # --- Open the input raw labels memmap for reading ---
        print(f"  Opening raw labels for reading: {raw_labels_path}")
        raw_labels_memmap = np.memmap(raw_labels_path, dtype=np.int32, mode='r', shape=original_shape)

        # --- Copy data from input to new output memmap (chunked) ---
        chunk_size_copy = min(100, original_shape[0]) if original_shape[0] > 0 else 1
        print("  Copying raw labels to new output memmap...")
        for i in tqdm(range(0, original_shape[0], chunk_size_copy), desc="  Copying labels"):
            end_idx = min(i + chunk_size_copy, original_shape[0])
            trimmed_labels_memmap[i:end_idx] = raw_labels_memmap[i:end_idx]
        trimmed_labels_memmap.flush()
        # --- Close the input raw labels handle ---
        del raw_labels_memmap; gc.collect()
        print("  Copying complete.")

        # --- Generate Smooth Hull Boundary (using copied data if needed by func) ---
        print("  Generating smoothed hull boundary...")
        min_spacing_val = min(spacing) if min(spacing) > 1e-9 else 1.0
        erosion_iterations = math.ceil(hull_boundary_thickness_phys / min_spacing_val)
        print(f"    Calculated hull erosion iterations: {erosion_iterations}")

        # generate_hull needs the cell mask - use the newly created writable memmap
        hull_boundary_mask, smoothed_hull_stack = generate_hull_boundary_and_stack(
            volume=original_volume,
            cell_mask=trimmed_labels_memmap, # Pass handle to the NEW memmap
            hull_erosion_iterations=erosion_iterations,
            smoothing_iterations=smoothing_iterations
        )
        if smoothed_hull_stack is not None: del smoothed_hull_stack; gc.collect()

        # --- Trim based on the new boundary (modifies the NEW trimmed_labels_memmap in-place) ---
        if hull_boundary_mask is None or not np.any(hull_boundary_mask):
            print("  Hull boundary mask empty/not generated. Skipping edge trimming steps.")
            if hull_boundary_mask is None: hull_boundary_mask = np.zeros(original_shape, dtype=bool)
        else:
            print("  Applying edge trimming to the NEW output memmap...")
            # Pass the handle to the NEW writable memmap
            trimmed_voxels_mask = trim_object_edges_by_distance(
                segmentation_memmap=trimmed_labels_memmap, # Modify this NEW memmap
                original_volume=original_volume,
                hull_boundary_mask=hull_boundary_mask,
                spacing=spacing,
                distance_threshold=edge_trim_distance_threshold,
                global_brightness_cutoff=global_brightness_cutoff,
                min_remaining_size=min_size_voxels,
                chunk_size_z=edge_distance_chunk_size_z,
                heal_iterations=heal_iterations
            )
            trimmed_labels_memmap.flush() # Ensure modifications are written
            print(f"  Trimming applied. Mask of initially trimmed voxels captured (Sum: {np.sum(trimmed_voxels_mask)}).")

        # --- Finalization ---
        print("\n--- Hull Trimming Step Finished ---")
        final_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"Final memory usage for trimming step: {final_mem:.2f} MB")

        # --- IMPORTANT: Close the output memmap handle before returning path ---
        if trimmed_labels_memmap is not None and hasattr(trimmed_labels_memmap, '_mmap'):
            print(f"  Closing NEW trimmed labels memmap handle: {trimmed_labels_path}")
            del trimmed_labels_memmap
            gc.collect()

        # Return path to *new* temporary trimmed labels memmap, its dir, and the hull mask
        return trimmed_labels_path, trimmed_labels_temp_dir, hull_boundary_mask

    except Exception as e:
        print(f"\n!!! ERROR during Hull Trimming: {e} !!!")
        traceback.print_exc()
        # Ensure handles are closed on error
        if 'raw_labels_memmap' in locals() and raw_labels_memmap is not None and hasattr(raw_labels_memmap, '_mmap'): del raw_labels_memmap; gc.collect()
        if 'trimmed_labels_memmap' in locals() and trimmed_labels_memmap is not None and hasattr(trimmed_labels_memmap, '_mmap'): del trimmed_labels_memmap; gc.collect()
        # Attempt cleanup of the output temp dir if created
        if 'trimmed_labels_temp_dir' in locals() and trimmed_labels_temp_dir and os.path.exists(trimmed_labels_temp_dir):
            rmtree(trimmed_labels_temp_dir, ignore_errors=True)
        # Return None to indicate failure
        return None, None, np.zeros(original_shape, dtype=bool)
    finally:
         gc.collect() # General cleanup

# --- END OF FILE utils/ramified_module_3d/initial_3d_segmentation.py ---