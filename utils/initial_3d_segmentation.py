import numpy as np
from scipy import ndimage
# Make sure all necessary ndimage functions are imported
from scipy.ndimage import (gaussian_filter, generate_binary_structure)
from skimage.filters import frangi, sato
from skimage.morphology import remove_small_objects
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

from remove_artifacts import *

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

def segment_microglia_first_pass(
    volume, # Original intensity volume
    spacing,
    tubular_scales=[0.5, 1.0, 2.0, 3.0],
    smooth_sigma=0.5,
    connect_max_gap_physical=1.0,
    min_size_voxels=50,
    # --- Intensity Percentile Parameters ---
    low_threshold_percentile=25.0,   # Used for segmentation AND basis for brightness cutoff
    high_threshold_percentile=95.0,  # For normalization
    # --- NEW: Factor to derive brightness cutoff from segmentation threshold ---
    brightness_cutoff_factor=10000,    # e.g., 1.5 means cutoff is at segmentation_threshold * 1.5
                                     # Set to <=1.0 to effectively disable brightness check beyond segmentation threshold
    # --- Hull/Edge Parameters ---
    hull_opening_radius_phys=0.5,      # Pre-smoothing opening radius (physical units)
    hull_closing_radius_phys=1.0,      # Pre-smoothing closing radius (physical units)
    hull_boundary_thickness_phys=2.0,  # Desired physical thickness of boundary
    edge_trim_distance_threshold=2.0,
    # --- Chunking ---
    edge_distance_chunk_size_z=32      # Chunk size for distance calc fallback in trimming
    ):
    """
    First pass segmentation using slice-wise enhancement, percentile-based
    normalization/thresholding, and combined distance/brightness edge trimming.
    Uses memmaps extensively to reduce peak RAM usage. The brightness cutoff for
    trimming is derived globally from the low_threshold_percentile used for
    segmentation, made stricter by brightness_cutoff_factor.

    Parameters:
    -----------
    volume : ndarray or memmap
        Input 3D image volume.
    spacing : tuple
        Physical voxel spacing (z, y, x).
    tubular_scales : list[float]
        Scales (in physical units) for the Frangi/Sato filters.
    smooth_sigma : float
        Sigma (in physical units) for optional initial Gaussian smoothing. 0 to disable.
    connect_max_gap_physical : float
        Maximum physical distance to bridge gaps during morphological closing.
    min_size_voxels : int
        Minimum voxel count for an object to be kept after initial segmentation and trimming.
    low_threshold_percentile : float (0-100)
        Percentile of the *normalized* enhanced signal distribution used for final segmentation thresholding.
        Also serves as the base for the edge trimming brightness cutoff. Default: 25.0.
    high_threshold_percentile : float (0-100)
        Percentile of the *enhanced* signal within each slice used as the target
        for intensity normalization. Default: 95.0.
    brightness_cutoff_factor : float (>= 1.0)
        Multiplier applied to the segmentation threshold to get the brightness cutoff
        for edge trimming. E.g., 1.5 makes the brightness cutoff 1.5x higher than
        the segmentation threshold. Values <= 1.0 effectively disable the brightness
        check beyond the main segmentation threshold. Default: 1.5.
    hull_erosion_iterations : int
        Number of iterations to erode the slice-wise hull stack to define the boundary.
    edge_trim_distance_threshold : float
        Physical distance threshold for trimming objects near the hull boundary.
    edge_distance_chunk_size_z : int
        Chunk size along Z for distance calculation within edge trimming.

    Returns:
    --------
    final_segmentation : ndarray (int32)
        Final labeled segmentation mask (loaded in memory).
    first_pass_params : dict
        Dictionary containing the parameters used for the segmentation.
    hull_boundary_mask : ndarray (bool)
        Boolean mask indicating the hull boundary used for trimming.
    """
    # --- Initial Setup ---
    low_threshold_percentile = max(0.0, min(100.0, low_threshold_percentile))
    high_threshold_percentile = max(0.0, min(100.0, high_threshold_percentile))

    original_shape = volume.shape
    spacing = tuple(float(s) for s in spacing)
    first_pass_params = { # Store all params
        'spacing': spacing, 'tubular_scales': tubular_scales, 'smooth_sigma': smooth_sigma,
        'connect_max_gap_physical': connect_max_gap_physical, 'min_size_voxels': min_size_voxels,
        'low_threshold_percentile': low_threshold_percentile,
        'high_threshold_percentile': high_threshold_percentile,
        'brightness_cutoff_factor': brightness_cutoff_factor,
        # Store NEW hull parameters
        'hull_opening_radius_phys': hull_opening_radius_phys,
        'hull_closing_radius_phys': hull_closing_radius_phys,
        'hull_boundary_thickness_phys': hull_boundary_thickness_phys,
        # Store remaining parameters
        'edge_trim_distance_threshold': edge_trim_distance_threshold,
        'edge_distance_chunk_size_z': edge_distance_chunk_size_z,
        'original_shape': original_shape
        }
    print(f"\n--- Starting First Pass Segmentation (Smooth Hull) ---")
    print(f"Params: ... hull_open={hull_opening_radius_phys}, hull_close={hull_closing_radius_phys}, "
          f"hull_thick={hull_boundary_thickness_phys}, trim_dist={edge_trim_distance_threshold:.2f}")
    initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_mem:.2f} MB")

    # Keep track of memmaps and their temporary directories for cleanup
    memmap_registry = {} # {name: (memmap_object, path, temp_dir)}
    global_brightness_cutoff = None # Initialize brightness cutoff value

    try:
        # --- Step 1: Enhance Tubular Structures (Returns Memmap) ---
        print("\nStep 1: Enhancing structures...")
        enhanced_memmap, enhance_path, enhance_temp_dir = enhance_tubular_structures_slice_by_slice(
            volume,
            scales=tubular_scales,
            spacing=spacing,
            apply_3d_smoothing=(smooth_sigma > 0),
            smoothing_sigma_phys=smooth_sigma
            # Pass other enhance parameters if needed (frangi_alpha, etc.)
        )
        memmap_registry['enhanced'] = (enhanced_memmap, enhance_path, enhance_temp_dir)
        print("Enhancement memmap created.")
        gc.collect()

        # --- Step 2: Normalize Intensity per Z-slice using Percentiles (Reads enhanced_memmap) ---
        print("\nStep 2: Normalizing signal across depth using percentiles...")
        normalized_temp_dir = tempfile.mkdtemp(prefix="normalize_")
        normalized_path = os.path.join(normalized_temp_dir, 'normalized.dat')
        normalized_memmap = np.memmap(normalized_path, dtype=np.float32, mode='w+',
                               shape=enhanced_memmap.shape)
        memmap_registry['normalized'] = (normalized_memmap, normalized_path, normalized_temp_dir)

        # Calculate z-slice high percentile values...
        z_high_percentile_values = []
        print("  Calculating z-plane high percentile values...")
        target_intensity = 1.0
        print(f"  Target intensity for high percentile ({high_threshold_percentile:.1f}): {target_intensity}")
        chunk_size_norm_read = min(50, enhanced_memmap.shape[0]) if enhanced_memmap.shape[0] > 0 else 1
        for z_start in tqdm(range(0, enhanced_memmap.shape[0], chunk_size_norm_read), desc="  Calculating z-stats"):
            z_end = min(z_start + chunk_size_norm_read, enhanced_memmap.shape[0])
            enhanced_chunk = enhanced_memmap[z_start:z_end]
            for i, z in enumerate(range(z_start, z_end)):
                plane = enhanced_chunk[i]; finite_plane = plane[np.isfinite(plane)].ravel()
                if finite_plane.size == 0: z_high_percentile_values.append((z, 0.0)); continue
                samples = np.random.choice(finite_plane, min(100000, finite_plane.size), replace=False)
                if samples.size > 0:
                    try:
                        high_perc_value = np.percentile(samples, high_threshold_percentile)
                        if not np.isfinite(high_perc_value): high_perc_value = 0.0
                        z_high_percentile_values.append((z, float(high_perc_value)))
                    except Exception as e: print(f"  Warn: Error calculating percentile for slice {z}: {e}. Using 0."); z_high_percentile_values.append((z, 0.0))
                else: z_high_percentile_values.append((z, 0.0))
            del enhanced_chunk; gc.collect()

        # Apply normalization scaling...
        print("  Applying normalization scaling...")
        z_stats_dict = dict(z_high_percentile_values)
        chunk_size_norm_write = min(50, enhanced_memmap.shape[0]) if enhanced_memmap.shape[0] > 0 else 1
        for z_start in tqdm(range(0, enhanced_memmap.shape[0], chunk_size_norm_write), desc="  Normalizing z-planes"):
             z_end = min(z_start + chunk_size_norm_write, enhanced_memmap.shape[0])
             enhanced_chunk = enhanced_memmap[z_start:z_end]
             normalized_chunk = np.zeros_like(enhanced_chunk, dtype=np.float32)
             for i, z in enumerate(range(z_start, z_end)):
                 current_slice = enhanced_chunk[i]; high_perc_value = z_stats_dict.get(z, 0.0)
                 if current_slice is None or not np.any(np.isfinite(current_slice)): normalized_chunk[i] = 0.0; continue
                 if high_perc_value > 1e-6: scale_factor = target_intensity / high_perc_value; normalized_chunk[i] = (current_slice * scale_factor).astype(np.float32)
                 else: normalized_chunk[i] = current_slice.astype(np.float32)
             normalized_memmap[z_start:z_end] = normalized_chunk
             del enhanced_chunk, normalized_chunk; gc.collect()
        normalized_memmap.flush()
        print("Normalization step finished.")

        # Cleanup enhanced memmap now
        name = 'enhanced'
        if name in memmap_registry:
            mm, p, d = memmap_registry[name]; del mm; gc.collect()
            try: os.unlink(p); rmtree(d, ignore_errors=True)
            except Exception as e: print(f"Warn: Could not clean {name} temp files: {e}")
            del memmap_registry[name]
        gc.collect()

        # --- Step 3: Thresholding AND Calculate Global Brightness Cutoff ---
        print("\nStep 3: Thresholding and Calculating Global Brightness Cutoff...")
        binary_temp_dir = tempfile.mkdtemp(prefix="binary_")
        binary_path = os.path.join(binary_temp_dir, 'b.dat')
        binary_memmap = np.memmap(binary_path, dtype=bool, mode='w+', shape=original_shape)
        memmap_registry['binary'] = (binary_memmap, binary_path, binary_temp_dir)

        # Calculate segmentation threshold using chunked sampling
        sample_size = min(5_000_000, normalized_memmap.size)
        print(f"  Sampling {sample_size} points from normalized memmap for thresholds...")
        segmentation_threshold = 0.0
        samples_collected = []; collected_count = 0
        num_samples_needed = sample_size
        chunk_size_thresh_sample = min(200, normalized_memmap.shape[0]) if normalized_memmap.shape[0] > 0 else 1
        indices_z = np.arange(normalized_memmap.shape[0]); np.random.shuffle(indices_z)
        all_samples = None # Define all_samples outside loop

        if not hasattr(normalized_memmap, '_mmap') or normalized_memmap._mmap is None:
            raise RuntimeError("Normalized memmap closed unexpectedly before threshold sampling.")

        for z_start_idx in tqdm(range(0, len(indices_z), chunk_size_thresh_sample), desc="  Sampling Norm. Vol."):
            if collected_count >= num_samples_needed: break
            z_indices_chunk = indices_z[z_start_idx : z_start_idx + chunk_size_thresh_sample]
            slices_data = normalized_memmap[z_indices_chunk, :, :]; finite_samples_in_chunk = slices_data[np.isfinite(slices_data)].ravel(); del slices_data; gc.collect()
            samples_to_take = min(len(finite_samples_in_chunk), num_samples_needed - collected_count)
            if samples_to_take > 0: samples_collected.append(np.random.choice(finite_samples_in_chunk, samples_to_take, replace=False)); collected_count += samples_to_take
            del finite_samples_in_chunk
            if collected_count >= num_samples_needed: break

        if not samples_collected:
            print("  Warn: No finite samples collected. Using 0.0 threshold.")
            segmentation_threshold = 0.0
        else:
            all_samples = np.concatenate(samples_collected); del samples_collected; gc.collect()
            if all_samples.size > 0:
                 try: segmentation_threshold = np.percentile(all_samples, low_threshold_percentile)
                 except Exception as e: print(f"  Warn: Percentile calculation failed: {e}. Using median."); segmentation_threshold = np.median(all_samples)
            else: print("  Warn: Concatenated samples array empty. Using 0.0 threshold."); segmentation_threshold = 0.0
        print(f"  Segmentation threshold ({low_threshold_percentile:.1f} perc): {segmentation_threshold:.4f}")

        # Calculate GLOBAL Brightness Cutoff
        if all_samples is not None and all_samples.size > 0 :
             global_brightness_cutoff = segmentation_threshold * brightness_cutoff_factor
             print(f"  Global Brightness Cutoff ({brightness_cutoff_factor:.2f}x): {global_brightness_cutoff:.4f}")
        else:
             print("  Warn: No samples available. Cannot determine brightness cutoff. Setting to Inf.")
             global_brightness_cutoff = np.inf
        if all_samples is not None: del all_samples; gc.collect()

        # Apply segmentation threshold in chunks
        chunk_size_thresh_apply = min(100, original_shape[0]) if original_shape[0] > 0 else 1
        print(f"  Applying segmentation threshold {segmentation_threshold:.4f}...")
        for i in tqdm(range(0, original_shape[0], chunk_size_thresh_apply), desc="  Applying Threshold"):
             end_idx = min(i + chunk_size_thresh_apply, original_shape[0])
             if hasattr(normalized_memmap, '_mmap') and normalized_memmap._mmap is not None:
                  norm_chunk = normalized_memmap[i:end_idx]; binary_memmap[i:end_idx] = norm_chunk > segmentation_threshold; del norm_chunk; gc.collect()
             else: print(f"Error: 'normalized' memmap closed during thresholding chunk {i}"); binary_memmap[i:end_idx] = False
        binary_memmap.flush()
        print("Thresholding done.")

        # Cleanup normalized memmap
        name = 'normalized'
        if name in memmap_registry:
            mm, p, d = memmap_registry[name]; del mm; gc.collect()
            try: os.unlink(p); rmtree(d, ignore_errors=True)
            except Exception as e: print(f"Warn: Could not clean {name} temp files: {e}")
            del memmap_registry[name]
        gc.collect()

        # --- Step 4: Connect Fragmented Processes (Reads binary_memmap) ---
        print("\nStep 4: Connecting fragments...")
        connected_binary_memmap, connected_temp_dir = connect_fragmented_processes(
            binary_memmap,
            spacing=spacing,
            max_gap_physical=connect_max_gap_physical
        )
        connected_path = connected_binary_memmap.filename
        memmap_registry['connected'] = (connected_binary_memmap, connected_path, connected_temp_dir)
        print("Connection memmap created.")

        # Cleanup binary memmap
        name = 'binary'
        if name in memmap_registry:
            mm, p, d = memmap_registry[name]; del mm; gc.collect()
            try: os.unlink(p); rmtree(d, ignore_errors=True)
            except Exception as e: print(f"Warn: Could not clean {name} temp files: {e}")
            del memmap_registry[name]
        gc.collect()

        # --- Step 5: Remove Small Objects (Reads connected_memmap) ---
        print(f"\nStep 5: Cleaning objects smaller than {min_size_voxels} voxels...")
        cleaned_binary_dir=tempfile.mkdtemp(prefix="cleaned_")
        cleaned_binary_path=os.path.join(cleaned_binary_dir, 'c.dat')
        cleaned_binary_memmap=np.memmap(cleaned_binary_path, dtype=bool, mode='w+', shape=original_shape)
        memmap_registry['cleaned'] = (cleaned_binary_memmap, cleaned_binary_path, cleaned_binary_dir)
        print("  Loading connected mask for small object removal...")
        connected_memmap_obj = memmap_registry['connected'][0]
        try:
            connected_in_memory = np.array(connected_memmap_obj)
            print("  Applying remove_small_objects...")
            remove_small_objects(connected_in_memory, min_size=min_size_voxels, connectivity=1, out=cleaned_binary_memmap)
            del connected_in_memory; gc.collect()
        except MemoryError:
             print("  MemoryError loading connected mask. Skipping small object removal.")
             print("  Copying connected mask to cleaned mask..."); chunk_size_copy = min(100, original_shape[0]) if original_shape[0] > 0 else 1
             for i in tqdm(range(0, original_shape[0], chunk_size_copy), desc="  Copying Connected"): cleaned_binary_memmap[i:min(i+chunk_size_copy, original_shape[0])] = connected_memmap_obj[i:min(i+chunk_size_copy, original_shape[0])]
        except Exception as e:
            print(f"  Error during small object removal: {e}. Skipping.")
            print("  Copying connected mask to cleaned mask..."); chunk_size_copy = min(100, original_shape[0]) if original_shape[0] > 0 else 1
            for i in tqdm(range(0, original_shape[0], chunk_size_copy), desc="  Copying Connected"): cleaned_binary_memmap[i:min(i+chunk_size_copy, original_shape[0])] = connected_memmap_obj[i:min(i+chunk_size_copy, original_shape[0])]
        cleaned_binary_memmap.flush()
        print("Cleaning step done.")

        # Cleanup connected memmap
        name = 'connected'
        if name in memmap_registry:
            mm, p, d = memmap_registry[name]; del mm; gc.collect()
            try: os.unlink(p); rmtree(d, ignore_errors=True)
            except Exception as e: print(f"Warn: Could not clean {name} temp files: {e}")
            del memmap_registry[name]
        gc.collect()

        # --- Step 6: Label Connected Components (Reads cleaned_memmap) ---
        print("\nStep 6: Labeling components...")
        labels_temp_dir = None; labels_path = None; first_pass_segmentation_memmap = None
        try:
            labels_temp_dir = tempfile.mkdtemp(prefix="labels_")
            labels_path = os.path.join(labels_temp_dir, 'l.dat')
            first_pass_segmentation_memmap=np.memmap(labels_path, dtype=np.int32, mode='w+', shape=original_shape)
            memmap_registry['labels'] = (first_pass_segmentation_memmap, labels_path, labels_temp_dir)
            print("  Applying ndimage.label...")
            cleaned_memmap_obj = memmap_registry.get('cleaned', (None,))[0]
            if cleaned_memmap_obj is None or not hasattr(cleaned_memmap_obj, '_mmap') or cleaned_memmap_obj._mmap is None: raise RuntimeError("Input 'cleaned' memmap for labeling is invalid or missing.")
            try:
                num_features = ndimage.label(cleaned_memmap_obj, structure=generate_binary_structure(3, 1), output=first_pass_segmentation_memmap)
                first_pass_segmentation_memmap.flush()
                print(f"Labeling done ({num_features} features found).")
            except Exception as label_error:
                 print(f"\n!!! ERROR during ndimage.label or flush: {label_error}"); import traceback; traceback.print_exc()
                 name = 'labels'; # Attempt immediate cleanup
                 if name in memmap_registry: mm, p, d = memmap_registry[name]; del mm; gc.collect(); 
                 try: os.unlink(p); rmtree(d, ignore_errors=True); 
                 except Exception as e_clean: print(f"Warn cleanup {name}: {e_clean}"); del memmap_registry[name]
                 raise label_error from label_error
            # Cleanup cleaned binary memmap (ONLY if labeling succeeded)
            name = 'cleaned'
            if name in memmap_registry:
                print(f"  Cleaning up {name} after successful labeling...")
                mm, p, d = memmap_registry[name]; del mm; gc.collect()
                try: os.unlink(p); rmtree(d, ignore_errors=True)
                except Exception as e: print(f"Warn: Could not clean {name} temp files: {e}")
                del memmap_registry[name]
            gc.collect()
        except Exception as step6_setup_error:
             print(f"\n!!! ERROR during Step 6 setup (before labeling): {step6_setup_error}"); import traceback; traceback.print_exc()
             name = 'labels'; # Attempt immediate cleanup
             if name in memmap_registry: mm, p, d = memmap_registry[name]; del mm; gc.collect(); del memmap_registry[name]
             else: p, d = labels_path, labels_temp_dir
             try: 
                 if p and os.path.exists(p): os.unlink(p); 
                 if d and os.path.exists(d): rmtree(d, ignore_errors=True)
             except Exception as e_clean: print(f"Warn cleanup {name} setup fail: {e_clean}")
             raise step6_setup_error from step6_setup_error

        # --- Step 7: Generate Smooth Hull Boundary AND Trim Edge Artifacts ---
        print("\nStep 7: Generating Smooth Hull and Trimming Edges...")
        hull_boundary_mask = None # Initialize
        smoothed_hull_stack = None # Initialize
        labels_memmap_obj = None # Initialize labels object variable

        try:
            # --- Retrieve the Labeled Segmentation Memmap ---
            # This memmap was created in Step 6 and should still exist.
            print("  Locating labeled segmentation mask for hull generation and trimming...")
            labels_memmap_obj, labels_path, labels_temp_dir = memmap_registry.get('labels', (None, None, None))

            # **** CRITICAL CHECK ****
            if labels_memmap_obj is None or not hasattr(labels_memmap_obj, '_mmap') or labels_memmap_obj._mmap is None:
                # This check ensures the memmap object itself is valid and hasn't been closed/deleted unexpectedly.
                # It checks the internal _mmap attribute which is essential for operation.
                raise RuntimeError("Labeled segmentation memmap ('labels') is invalid or missing before Step 7.")
            # Check if file path exists too, as an extra safety measure
            if not os.path.exists(labels_path):
                 raise RuntimeError(f"Labeled segmentation memmap file ('{labels_path}') not found before Step 7.")
            print(f"  Found labels memmap: {labels_path}, Mode: {labels_memmap_obj.mode}")
            # Note: labels_memmap_obj was created with 'w+', so it's readable.

            # --- 7a. Generate Smooth Hull Boundary using the Labeled Mask ---
            print("  Generating smoothed hull boundary using the labeled segmentation...")
            # **** MODIFIED CALL using LABELS memmap ****
            hull_boundary_mask, smoothed_hull_stack = generate_hull_boundary_and_stack(
                volume=volume,                          # Original intensity volume
                cell_mask=labels_memmap_obj,            # Use the LABELED mask from Step 6
                hull_erosion_iterations=math.ceil(hull_boundary_thickness_phys / min(spacing)) if min(spacing)>0 else 1, # Calc based on physical thickness
                smoothing_iterations=1                  # Default or use a new parameter if added
            )
            # **** END MODIFIED CALL ****

            # --- 7b. Trim based on the new boundary ---
            if hull_boundary_mask is None or not np.any(hull_boundary_mask):
                print("  Hull boundary mask empty/not generated. Skipping edge trimming.")
                if hull_boundary_mask is None: # Ensure it's a valid empty mask if needed later
                     hull_boundary_mask = np.zeros(original_shape, dtype=bool)
            else:
                 print("  Proceeding with edge trimming using the generated smooth boundary...")

                 # --- Reopen labels memmap in r+ mode IF NEEDED ---
                 # We need to modify the labels memmap in-place during trimming.
                 # It was created with 'w+', which allows writing, but 'r+' is explicitly
                 # for reading and writing to an existing file. Depending on the numpy
                 # version and OS, directly writing after reading might work with 'w+',
                 # but reopening with 'r+' is safer and more explicit for modification.

                 labels_memmap_rplus = None # Initialize variable for the r+ handle
                 current_mode = labels_memmap_obj.mode
                 print(f"    Labels memmap current mode for trimming: {current_mode}")

                 if current_mode != 'r+':
                     print(f"    Reopening labels memmap ({labels_path}) in 'r+' mode for trimming.")
                     # Ensure the previous handle is closed before reopening
                     if hasattr(labels_memmap_obj, '_mmap') and labels_memmap_obj._mmap is not None:
                         labels_memmap_obj.flush() # Flush writes if any were made (shouldn't be yet)
                         del labels_memmap_obj
                         gc.collect()
                     # Reopen
                     labels_memmap_rplus = np.memmap(labels_path, dtype=np.int32, mode='r+', shape=original_shape)
                     # Update the registry TO POINT TO THE NEW HANDLE
                     memmap_registry['labels'] = (labels_memmap_rplus, labels_path, labels_temp_dir)
                     print("    Labels memmap reopened in 'r+' mode.")
                 else:
                     # It's already in a writable mode ('r+' or potentially 'w+' still works)
                     print("    Labels memmap already in a writable mode.")
                     labels_memmap_rplus = labels_memmap_obj # Use the existing handle

                 # --- Perform Trimming ---
                 # Ensure the handle we pass is valid
                 if labels_memmap_rplus is None or not hasattr(labels_memmap_rplus, '_mmap') or labels_memmap_rplus._mmap is None:
                      raise RuntimeError("Labels memmap became invalid before trimming could start.")

                 _ = trim_object_edges_by_distance(
                     segmentation_memmap=labels_memmap_rplus,
                     original_volume=volume,
                     hull_boundary_mask=hull_boundary_mask,
                     spacing=spacing,
                     distance_threshold=edge_trim_distance_threshold,
                     global_brightness_cutoff=global_brightness_cutoff,
                     min_remaining_size=min_size_voxels,
                     chunk_size_z=edge_distance_chunk_size_z,
                     heal_iterations=20
                 )
                 labels_memmap_rplus.flush() # Ensure trimming/healing changes are saved
                 print("    Trimming applied to labels memmap.")

        except Exception as e:
             print(f"\nERROR during hull generation or edge trimming: {e}")
             import traceback
             traceback.print_exc()
             # Ensure hull_boundary_mask exists even on error for final return
             if 'hull_boundary_mask' not in locals() or hull_boundary_mask is None:
                  hull_boundary_mask = np.zeros(original_shape, dtype=bool)
             # Attempt cleanup of labels memmap if it was opened and caused the error
             if 'labels_memmap_obj' in locals() and labels_memmap_obj is not None and hasattr(labels_memmap_obj, '_mmap'):
                 del labels_memmap_obj
             if 'labels_memmap_rplus' in locals() and labels_memmap_rplus is not None and hasattr(labels_memmap_rplus, '_mmap'):
                 del labels_memmap_rplus
             gc.collect()
             # We might want to re-raise the error here depending on desired behavior
             # raise e

        # --- Step 8: Load FINAL Labeled Segmentation into Memory ---
        # This part should now correctly use the potentially modified 'labels' handle from the registry
        print("\nStep 8: Loading final trimmed labels into memory...")
        # Retrieve the potentially updated handle from the registry again
        final_labels_memmap_obj, _, _ = memmap_registry.get('labels', (None, None, None))

        if final_labels_memmap_obj is None or not hasattr(final_labels_memmap_obj, '_mmap') or final_labels_memmap_obj._mmap is None:
            raise RuntimeError("Final labels memmap object not found or invalid before loading into memory.")

        # Ensure data is flushed before reading into memory
        final_labels_memmap_obj.flush()
        final_segmentation = np.array(final_labels_memmap_obj)
        print("Final labels loaded into memory.")


        # --- Finalization ---
        # ... (Log final stats, same as before) ...
        print(f"\n--- First Pass Segmentation Finished ---")
        final_unique_labels = np.unique(final_segmentation)
        final_object_count = len(final_unique_labels[final_unique_labels != 0])
        print(f"Final labeled mask shape: {final_segmentation.shape}")
        print(f"Number of labeled objects remaining: {final_object_count}")
        final_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"Final memory usage: {final_mem:.2f} MB")

        # Return final result and the generated hull boundary mask
        return final_segmentation, first_pass_params, hull_boundary_mask

    finally:
        # --- FINAL Cleanup ALL remaining memmaps ---
        print("\nCleaning up any remaining temporary memmap files...")
        registry_keys = list(memmap_registry.keys())
        for name in registry_keys:
            if name in memmap_registry:
                mm, p, d = memmap_registry[name]
                print(f"  Cleaning up {name} (Path: {p})...")
                if hasattr(mm, '_mmap') and mm._mmap is not None:
                    try: del mm; gc.collect()
                    except Exception as e_del: print(f"    Warn: Error deleting memmap object for {name}: {e_del}")
                elif hasattr(mm, 'close'): 
                    try: mm.close(); 
                    except: pass
                try:
                    if p and os.path.exists(p): os.unlink(p); # print(f"    Deleted file: {p}")
                except Exception as e_unlink: print(f"    Warn: Error unlinking file {p}: {e_unlink}")
                try:
                    if d and os.path.exists(d): rmtree(d, ignore_errors=True); # print(f"    Removed directory: {d}")
                except Exception as e_rmtree: print(f"    Warn: Error removing directory {d}: {e_rmtree}")
                if name in memmap_registry: del memmap_registry[name]
        print("Final cleanup attempts finished.")
        gc.collect()