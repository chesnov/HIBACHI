import numpy as np
from scipy import ndimage
# Make sure all necessary ndimage functions are imported
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           binary_erosion, generate_binary_structure)
from skimage.filters import frangi, threshold_otsu, sato
from skimage.morphology import convex_hull_image, remove_small_objects
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
from skimage.filters import frangi, sato
from functools import partial
from skimage.measure import regionprops
import math
seed = 42
np.random.seed(seed)         # For NumPy

# Assuming nuclear_segmenter.py might be needed elsewhere, keep imports
# from nuclear_segmenter import downsample_for_isotropic, upsample_segmentation

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

# Parallel slice-wise enhancement function (remains the same)
def enhance_tubular_structures_slice_by_slice(volume, scales, spacing, black_ridges=False, frangi_alpha=0.5, frangi_beta=0.5, frangi_gamma=15, apply_3d_smoothing=True, smoothing_sigma_phys=0.5, ram_safety_factor=0.8, mem_factor_per_slice=8.0 ):
    """ Enhance tubular structures slice-by-slice in parallel. """
    print(f"Starting slice-by-slice (2.5D) tubular enhancement with PARALLEL processing...")
    print(f"  Volume shape: {volume.shape}, Spacing: {spacing}")
    print(f"  Initial memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    spacing = tuple(float(s) for s in spacing); volume_shape = volume.shape
    slice_shape = volume_shape[1:]; num_slices = volume_shape[0]; xy_spacing = spacing[1:]
    input_volume_memmap = None; input_memmap_path = None; input_memmap_dir = None; source_volume_cleaned = False
    if apply_3d_smoothing and smoothing_sigma_phys > 0:
        print(f"Applying initial 3D smoothing...")
        sigma_voxel_3d = tuple(smoothing_sigma_phys / s for s in spacing)
        smooth_temp_dir = tempfile.mkdtemp(prefix="pre_smooth_"); smooth_path = os.path.join(smooth_temp_dir, 'smoothed_3d.dat')
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
        input_volume_memmap.flush(); print(f"  3D pre-smoothing done.")
    else:
        print("  Skipping initial 3D smoothing.");
        if isinstance(volume, np.memmap): input_volume_memmap = volume; input_memmap_path = volume.filename
        else:
            print(f"  Converting input volume to temporary memmap..."); input_memmap_dir = tempfile.mkdtemp(prefix="input_volume_")
            input_volume_memmap, input_memmap_path, _ = create_memmap(data=volume, directory=input_memmap_dir)
            source_volume_cleaned = True
    avg_xy_spacing = np.mean(xy_spacing); sigmas_voxel_2d = sorted([s / avg_xy_spacing for s in scales])
    print(f"  Using 2D voxel sigmas: {sigmas_voxel_2d}")
    output_temp_dir = tempfile.mkdtemp(prefix='tubular_enhance_parallel_'); output_path = os.path.join(output_temp_dir, 'enhanced_volume_parallel.dat')
    output_dtype = np.float32; output_memmap = np.memmap(output_path, dtype=output_dtype, mode='w+', shape=volume_shape)
    print(f"  Output memmap created: {output_path}")
    try:
        total_cores = os.cpu_count(); max_cpu_workers = max(1, total_cores - 1 if total_cores else 1)
        available_ram = psutil.virtual_memory().available; usable_ram = available_ram * ram_safety_factor
        input_dtype_bytes = np.dtype(input_volume_memmap.dtype).itemsize if input_volume_memmap is not None else np.dtype(np.float32).itemsize
        slice_mem_bytes = slice_shape[0] * slice_shape[1] * input_dtype_bytes; estimated_worker_ram = slice_mem_bytes * mem_factor_per_slice
        if estimated_worker_ram <= 0: estimated_worker_ram = 1
        max_mem_workers = max(1, int(usable_ram // estimated_worker_ram)); num_workers = min(max_cpu_workers, max_mem_workers, num_slices)
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
        del output_memmap
        if input_volume_memmap is not None and hasattr(input_volume_memmap, '_mmap'): del input_volume_memmap
        gc.collect()
    if source_volume_cleaned and input_memmap_dir:
        print(f"Cleaning up temporary input volume memmap: {input_memmap_dir}"); 
        try: rmtree(input_memmap_dir, ignore_errors=True)
        except Exception as e: print(f"Warning: Could not delete temp input directory {input_memmap_dir}: {e}")
    elif apply_3d_smoothing and input_memmap_dir:
         print(f"Cleaning up temporary pre-smoothed volume: {input_memmap_dir}"); 
         try: rmtree(input_memmap_dir, ignore_errors=True)
         except Exception as e: print(f"Warning: Could not delete temp smooth directory {input_memmap_dir}: {e}")
    print("Loading final enhanced result from memmap...")
    final_enhanced_memmap = np.memmap(output_path, dtype=output_dtype, mode='r', shape=volume_shape)
    result_in_memory = np.array(final_enhanced_memmap)
    del final_enhanced_memmap; gc.collect(); print("Final result loaded into memory.")
    print(f"Final memory usage after slice-wise enhancement: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return result_in_memory, output_temp_dir

def generate_anisotropic_structure(rank, connectivity, spacing):
    """Generates a connectivity structure based on physical spacing."""
    if rank != 3: raise ValueError("Only rank 3 supported")
    base_structure = generate_binary_structure(rank, connectivity)
    return base_structure

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


def generate_hull_boundary_and_stack(volume, hull_erosion_iterations=1):
    """ Generates boundary and hull stack from slice-wise 2D hulls. """
    # ... (Implementation as provided before) ...
    print("Generating hull boundary mask AND stack using slice-wise hulls...")
    original_shape = volume.shape; hull_boundary = None; hull_stack = None
    print("  Creating tissue mask..."); # ... Otsu logic ...
    otsu_sample_size = min(2_000_000, volume.size)
    if otsu_sample_size == volume.size: otsu_samples = volume.ravel()
    else: otsu_indices = np.random.choice(volume.size, otsu_sample_size, replace=False); otsu_coords = np.unravel_index(otsu_indices, volume.shape); otsu_samples = volume[otsu_coords]
    if np.all(otsu_samples == otsu_samples[0]): tissue_thresh = otsu_samples[0]; print(f"  Warn: All samples identical ({tissue_thresh}).")
    else: tissue_thresh = threshold_otsu(otsu_samples)
    print(f"  Tissue threshold: {tissue_thresh:.2f}"); tissue_mask = volume > tissue_thresh; del otsu_samples; gc.collect()
    if not np.any(tissue_mask): print("  Warn: Tissue mask empty."); return None, None
    print("  Calculating 2D convex hull slice-by-slice..."); hull_stack = np.zeros_like(tissue_mask, dtype=bool)
    for z in tqdm(range(original_shape[0]), desc="  Processing Slices for Hull"):
        tissue_slice = tissue_mask[z, :, :];
        if np.any(tissue_slice):
             if not tissue_slice.flags['C_CONTIGUOUS']: tissue_slice = np.ascontiguousarray(tissue_slice)
             hull_stack[z, :, :] = convex_hull_image(tissue_slice)
    del tissue_mask; gc.collect(); print("  Slice-wise hull complete.")
    if hull_erosion_iterations <= 0: print("  Warn: No hull erosion."); hull_boundary = np.zeros(original_shape, dtype=bool); return hull_boundary, hull_stack
    print(f"  Eroding 3D hull stack (iter={hull_erosion_iterations})..."); struct_3d = generate_binary_structure(3, 1)
    eroded_hull_stack = binary_erosion(hull_stack, structure=struct_3d, iterations=hull_erosion_iterations); print("  Erosion complete.")
    print("  Calculating hull boundary..."); hull_boundary = hull_stack & (~eroded_hull_stack); # Keep hull_stack for return
    print(f"  Hull boundary mask created ({np.sum(hull_boundary)} voxels).")
    # Return boundary AND original hull stack (in case caller needs it, though trimming doesn't)
    return hull_boundary, hull_stack


# --- NEW Helper Function: Trim Object Edges ---

def trim_object_edges_by_distance(
    segmentation,            # Labeled segmentation (modified in-place)
    hull_boundary_mask,      # Boolean mask of the hull boundary
    spacing,                 # Voxel spacing for physical distance
    distance_threshold,      # Physical distance threshold
    min_remaining_size=10,   # Minimum size for object remnants after trimming
    chunk_size_z=32          # Chunk size for potential chunked EDT/application
    ):
    """
    Removes voxels from labeled objects that are closer than a threshold
    distance to the hull boundary. Optionally removes objects that become
    too small after trimming.

    Parameters:
    -----------
    segmentation : ndarray (int)
        The labeled segmentation image. Will be MODIFIED IN-PLACE.
    hull_boundary_mask : ndarray (bool)
        Mask where boundary voxels are True.
    spacing : tuple
        Physical voxel spacing (z, y, x).
    distance_threshold : float
        Physical distance. Voxels closer than this to the boundary will be removed.
    min_remaining_size : int
        After trimming, objects smaller than this voxel count will be removed entirely.
        Set to 0 or less to disable remnant removal.
    chunk_size_z : int
        Chunk size along Z for processing distance transform and trimming if
        full volume calculation fails.

    Returns:
    --------
    trimmed_voxels_mask : ndarray (bool)
        A boolean mask indicating which voxels were removed (set to 0).
    """
    print(f"\n--- Trimming object edges closer than {distance_threshold:.2f} units to hull boundary ---")
    original_shape = segmentation.shape
    trimmed_voxels_mask = np.zeros(original_shape, dtype=bool) # Track what was removed

    if not np.any(hull_boundary_mask):
        print("  Hull boundary mask is empty. No trimming performed.")
        return trimmed_voxels_mask

    distance_from_boundary = None # Initialize

    # --- Calculate Distance Transform ---
    try:
        print("  Calculating full distance transform from hull boundary...")
        start_edt = time.time()
        # Compute distance from non-boundary points to the nearest boundary point
        distance_from_boundary = distance_transform_edt(
            ~hull_boundary_mask,
            sampling=spacing
        )
        print(f"  Full EDT calculated in {time.time()-start_edt:.2f}s.")
        gc.collect()

        # --- Apply Trimming (Full Volume) ---
        print(f"  Applying trimming threshold ({distance_threshold:.2f})...")
        # Find all voxels (across all objects) below the threshold
        voxels_to_trim = (distance_from_boundary < distance_threshold)

        # Apply this mask to the segmentation WHERE objects exist (label > 0)
        trim_target_mask = voxels_to_trim & (segmentation > 0)
        num_trimmed = np.sum(trim_target_mask)
        print(f"  Identified {num_trimmed} object voxels for trimming.")

        if num_trimmed > 0:
            # Store which voxels were trimmed before modifying segmentation
            trimmed_voxels_mask[trim_target_mask] = True
            # Set trimmed voxels to background (0) IN-PLACE
            segmentation[trim_target_mask] = 0
            print("  Trimming applied.")
        else:
            print("  No voxels met the trimming criteria.")

        del distance_from_boundary, voxels_to_trim, trim_target_mask
        gc.collect()

    except MemoryError:
        # --- Fallback to Chunked Processing ---
        print("\n  MemoryError calculating full distance transform. Falling back to chunked processing...")
        gc.collect() # Try to free memory before chunking

        num_chunks = math.ceil(original_shape[0] / chunk_size_z)
        total_trimmed_in_chunks = 0

        for i in tqdm(range(num_chunks), desc="  Processing Chunks"):
            z_start = i * chunk_size_z
            z_end = min((i + 1) * chunk_size_z, original_shape[0])
            if z_start >= z_end: continue

            # Load INVERSE boundary chunk
            inv_boundary_chunk = ~hull_boundary_mask[z_start:z_end]
            if not np.any(~inv_boundary_chunk): # If boundary covers whole chunk
                 edt_chunk = np.zeros(inv_boundary_chunk.shape, dtype=np.float32)
            else:
                 edt_chunk = distance_transform_edt(inv_boundary_chunk, sampling=spacing)

            # Identify voxels below threshold in this chunk
            voxels_to_trim_chunk = (edt_chunk < distance_threshold)

            # Apply to the corresponding segmentation slice VIEW
            seg_chunk_view = segmentation[z_start:z_end]
            trim_target_mask_chunk = voxels_to_trim_chunk & (seg_chunk_view > 0)
            num_trimmed_chunk = np.sum(trim_target_mask_chunk)

            if num_trimmed_chunk > 0:
                 # Update the global tracking mask
                 trimmed_voxels_mask[z_start:z_end][trim_target_mask_chunk] = True
                 # Modify the segmentation VIEW in-place
                 seg_chunk_view[trim_target_mask_chunk] = 0
                 total_trimmed_in_chunks += num_trimmed_chunk

            del inv_boundary_chunk, edt_chunk, voxels_to_trim_chunk
            del seg_chunk_view, trim_target_mask_chunk
            gc.collect()

        print(f"  Chunked trimming applied. Total voxels trimmed: {total_trimmed_in_chunks}")

    # --- Optional: Remove Small Remnants ---
    if min_remaining_size > 0:
        print(f"  Removing remnants smaller than {min_remaining_size} voxels...")
        start_rem = time.time()
        # remove_small_objects works on boolean or labeled arrays.
        # Need to apply it efficiently. We can relabel briefly or work on boolean.
        # Option 1: Relabel (cleaner but uses more memory temporarily)
        # temp_labels, num_final = label(segmentation > 0)
        # remove_small_objects(temp_labels, min_size=min_remaining_size, connectivity=1, in_place=True)
        # segmentation[temp_labels == 0] = 0 # Ensure background is zero
        # del temp_labels

        # Option 2: Boolean mask and selective removal (more complex code, less memory)
        # Create boolean mask of current objects
        bool_seg = segmentation > 0
        # Remove small objects from boolean mask
        cleaned_bool_seg = remove_small_objects(bool_seg, min_size=min_remaining_size, connectivity=1)
        # Identify voxels that were part of small objects
        small_remnant_mask = bool_seg & (~cleaned_bool_seg)
        num_remnants_removed = np.sum(segmentation[small_remnant_mask] > 0) # Count actual labeled voxels removed
        # Set identified remnants to 0 in the original labeled array
        segmentation[small_remnant_mask] = 0
        print(f"  Removed {num_remnants_removed} voxels belonging to small remnants "
              f"in {time.time()-start_rem:.2f}s.")
        del bool_seg, cleaned_bool_seg, small_remnant_mask
        gc.collect()

    print("--- Edge trimming finished ---")
    return trimmed_voxels_mask

def segment_microglia_first_pass(
    volume, # Use original volume for tissue mask
    spacing,
    tubular_scales=[0.5, 1.0, 2.0, 3.0],
    smooth_sigma=0.5,
    connect_max_gap_physical=1.0,
    min_size_voxels=50,
    sensitivity=0.8,
    background_level=50,
    target_level=75,
    hull_erosion_iterations=1,          # For defining hull boundary
    edge_trim_distance_threshold=2.0,   # Physical distance for trimming
    # --- Chunking ---
    edge_distance_chunk_size_z=32      # Chunk size for distance calc fallback
    ):
    """
    First pass segmentation using slice-wise enhancement and trimming edge artifacts
    based on distance from the hull boundary.
    """
    # --- Initial Setup ---
    sensitivity = max(0.01, min(1.0, sensitivity))
    original_shape = volume.shape
    spacing = tuple(float(s) for s in spacing)
    first_pass_params = { # ... Store all params ...
        'spacing': spacing, 'tubular_scales': tubular_scales, 'smooth_sigma': smooth_sigma,
        'connect_max_gap_physical': connect_max_gap_physical, 'min_size_voxels': min_size_voxels,
        'sensitivity': sensitivity, 'background_level': background_level, 'target_level': target_level,
        'hull_erosion_iterations': hull_erosion_iterations,
        'edge_trim_distance_threshold': edge_trim_distance_threshold, # New
        'edge_distance_chunk_size_z': edge_distance_chunk_size_z,
        'original_shape': original_shape }
    print(f"\n--- Starting First Pass Segmentation (Edge Trimming Method) ---")
    print(f"Params: ... hull_erosion={hull_erosion_iterations}, "
          f"trim_dist={edge_trim_distance_threshold:.2f}, ")
    initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_mem:.2f} MB")

    # --- Step 1: Enhance Tubular Structures ---
    print("\nStep 1: Enhancing structures...")
    # Assuming enhance_tubular_structures_slice_by_slice is defined elsewhere
    enhanced, enhance_temp_dir = enhance_tubular_structures_slice_by_slice(
        volume,
        scales=tubular_scales,
        spacing=spacing,
        apply_3d_smoothing=(smooth_sigma > 0),
        smoothing_sigma_phys=smooth_sigma
        # Pass other enhance parameters if needed (frangi_alpha, etc.)
    )
    print("Enhancement done.")
    gc.collect()

    # --- Step 2: Normalize Intensity per Z-slice ---
    print("\nStep 2: Normalizing signal across depth...")
    normalized_temp_dir = tempfile.mkdtemp(prefix="normalize_")
    normalized_path = os.path.join(normalized_temp_dir, 'normalized.dat')
    normalized = np.memmap(normalized_path, dtype=np.float32, mode='w+',
                           shape=enhanced.shape)

    # Calculate z-stats
    z_stats = []
    print("  Calculating z-plane statistics...")
    for z in tqdm(range(enhanced.shape[0]), desc="  Calculating z-stats"):
        plane = enhanced[z]
        if plane is None or plane.size == 0 or not np.any(np.isfinite(plane)):
             z_stats.append((z, 0, 1)); continue # Fallback for bad slice

        max_samples = 50000
        finite_plane = plane[np.isfinite(plane)]
        if finite_plane.size == 0:
             z_stats.append((z, 0, 1)); continue # Fallback if no finite values

        if finite_plane.size < max_samples:
             samples = finite_plane
        else:
             samples = np.random.choice(finite_plane, max_samples, replace=False)

        if samples.size > 0:
            try:
                # USE background_level PARAMETER HERE
                noise_level = np.percentile(samples, background_level)
                foreground = samples[samples > noise_level]

                median_val = np.median(foreground) if len(foreground) > 0 else np.median(samples)
                std_val = np.std(foreground) if len(foreground) > 0 else np.std(samples)

                if not np.isfinite(median_val): median_val = 0.0
                if not np.isfinite(std_val) or std_val <= 0: std_val = 1.0

                z_stats.append((z, float(median_val), float(std_val)))
            except Exception as e:
                 print(f"  Warn: Error calculating stats for slice {z}: {e}. Using defaults.")
                 z_stats.append((z, 0, 1))
        else:
             z_stats.append((z, 0, 1)) # Should not happen if finite_plane check passed

    # Apply normalization
    z_medians = [item[1] for item in z_stats
                 if isinstance(item, (tuple, list)) and len(item)==3 and item[1] > 0]
    if not z_medians:
        print("  Warning: No valid foreground medians found. Skipping normalization scaling.")
        target_intensity = 1.0
        print("  Copying enhanced data directly to normalized memmap...")
        chunk_size = min(100, enhanced.shape[0]) if enhanced.shape[0] > 0 else 1
        for i in range(0, enhanced.shape[0], chunk_size):
             end = min(i + chunk_size, enhanced.shape[0])
             normalized[i:end] = enhanced[i:end]
    else:
        # USE target_level PARAMETER HERE
        target_intensity = np.percentile(z_medians, target_level)
        print(f"  Target intensity for normalization: {target_intensity:.4f}")
        print("  Applying normalization scaling...")
        for item in tqdm(z_stats, desc="  Normalizing z-planes"):
            if isinstance(item, (tuple, list)) and len(item) == 3:
                z, median, std = item
                current_slice = enhanced[z]
                if current_slice is None or not np.any(np.isfinite(current_slice)):
                     normalized[z] = 0.0; continue
                if median > 0:
                    scale_factor = target_intensity / median
                    normalized[z] = (current_slice * scale_factor).astype(np.float32)
                else:
                    normalized[z] = current_slice.astype(np.float32)
            else: # Fallback for malformed item
                 print(f"  Warn: Skipping bad item in normalization: {item}")
                 try:
                     z_index = int(item[0]) if isinstance(item, (tuple, list)) else item
                     if 0 <= z_index < enhanced.shape[0]:
                          normalized[z_index] = enhanced[z_index].astype(np.float32)
                 except: pass # Ignore if index determination fails

    normalized.flush()
    print("Normalization step finished.")
    del enhanced; gc.collect()
    try: rmtree(enhance_temp_dir)
    except Exception as e: print(f"Warn: Could not clean enhance temp dir: {e}")
    gc.collect()

    # --- Step 3: Thresholding ---
    print("\nStep 3: Thresholding...")
    binary_temp_dir = tempfile.mkdtemp(prefix="binary_")
    binary_path = os.path.join(binary_temp_dir, 'b.dat')
    binary = np.memmap(binary_path, dtype=bool, mode='w+', shape=original_shape)

    # Calculate threshold (Otsu on sample)
    sample_size = min(2_000_000, normalized.size)
    print(f"  Sampling {sample_size} points for Otsu...")
    if sample_size == normalized.size:
         samples = normalized[:].ravel()
    else:
        # Ensure memmap is still valid before sampling
        if normalized._mmap is None: raise RuntimeError("Normalized memmap closed unexpectedly before Otsu sampling.")
        indices = np.random.choice(normalized.size, sample_size, replace=False)
        coords = np.unravel_index(indices, normalized.shape); samples = normalized[coords]

    adjusted_threshold = 0 # Default
    if samples.size > 0 and np.any(samples > np.min(samples)): # Check variance
        try:
            # Basic foreground guess (e.g., above 10th percentile) for robustness
            foreground_samples = samples[samples > np.percentile(samples, 10)]
            if foreground_samples.size == 0: foreground_samples = samples # Fallback
            threshold = threshold_otsu(foreground_samples)
            # USE sensitivity PARAMETER HERE
            adjusted_threshold = threshold * (1 - (sensitivity * 0.5))
            print(f"  Global Otsu threshold: {threshold:.4f}, Adjusted threshold: {adjusted_threshold:.4f}")
        except ValueError:
            print("  Warn: Otsu failed. Using median threshold.")
            median_val = np.median(samples)
            adjusted_threshold = median_val * (1 - (sensitivity * 0.5)) # Apply sensitivity to median
            print(f"  Using median-based adjusted threshold: {adjusted_threshold:.4f}")
    else:
         print("  Warn: No valid samples/variance for Otsu. Using median.")
         median_val = np.median(samples) if samples.size > 0 else 0
         adjusted_threshold = median_val * (1 - (sensitivity * 0.5)) # Apply sensitivity to median
         print(f"  Using median-based adjusted threshold: {adjusted_threshold:.4f}")

    # Apply threshold in chunks
    chunk_size_z = min(100, original_shape[0])
    print(f"  Applying threshold {adjusted_threshold:.4f}...")
    for i in tqdm(range(0, original_shape[0], chunk_size_z), desc="  Applying Threshold"):
         end_idx = min(i + chunk_size_z, original_shape[0])
         if normalized._mmap is not None:
              norm_chunk = normalized[i:end_idx]
              binary[i:end_idx] = norm_chunk > adjusted_threshold
              del norm_chunk
         else:
              print(f"Error: 'normalized' memmap closed before thresholding chunk {i}")
              binary[i:end_idx] = False # Fill failed chunk with False

    binary.flush()
    print("Thresholding done.")
    # Cleanup normalized memmap
    if 'normalized' in locals() and normalized._mmap is not None: del normalized
    gc.collect()
    try: os.unlink(normalized_path); rmtree(normalized_temp_dir)
    except Exception as e: print(f"Warn: Could not clean norm temp files: {e}")
    gc.collect()

    # --- Step 4: Connect Fragmented Processes ---
    print("\nStep 4: Connecting fragments...")
    # Assuming connect_fragmented_processes is defined elsewhere
    connected_binary, connected_temp_dir = connect_fragmented_processes(
        binary,
        spacing=spacing,
        max_gap_physical=connect_max_gap_physical # Use parameter
    )
    print("Connection done.")
    del binary; gc.collect()
    try: os.unlink(binary_path); rmtree(binary_temp_dir)
    except Exception as e: print(f"Warn: Could not clean binary temp files: {e}")
    gc.collect()

    # --- Step 5: Remove Small Objects ---
    print(f"\nStep 5: Cleaning objects smaller than {min_size_voxels} voxels...")
    cleaned_binary_dir=tempfile.mkdtemp(prefix="cleaned_")
    cleaned_binary_path=os.path.join(cleaned_binary_dir, 'c.dat')
    cleaned_binary=np.memmap(cleaned_binary_path, dtype=bool, mode='w+',
                              shape=original_shape)
    # Assuming remove_small_objects is available
    remove_small_objects(connected_binary,
                         min_size=min_size_voxels, # Use parameter
                         connectivity=1, # Use face connectivity
                         out=cleaned_binary)
    cleaned_binary.flush()
    print("Cleaning done.")
    conn_bin_fn = connected_binary.filename
    del connected_binary; gc.collect()
    try: os.unlink(conn_bin_fn); rmtree(connected_temp_dir)
    except Exception as e: print(f"Warn: Could not clean connected temp files: {e}")
    gc.collect()

    # --- Step 6: Label Connected Components ---
    print("\nStep 6: Labeling components...")
    labels_temp_dir=tempfile.mkdtemp(prefix="labels_")
    labels_path=os.path.join(labels_temp_dir, 'l.dat')
    first_pass_segmentation_memmap=np.memmap(labels_path, dtype=np.int32,
                                             mode='w+', shape=original_shape)
    num_features = ndimage.label(
        cleaned_binary,
        structure=generate_binary_structure(3, 1), # Face connectivity
        output=first_pass_segmentation_memmap
    )
    first_pass_segmentation_memmap.flush()
    print(f"Labeling done ({num_features} features found).")
    cleaned_bin_fn = cleaned_binary.filename
    del cleaned_binary; gc.collect()
    try: os.unlink(cleaned_bin_fn); rmtree(cleaned_binary_dir)
    except Exception as e: print(f"Warn: Could not clean cleaned binary temp files: {e}")
    gc.collect()

    # --- Step 7: Load Labeled Segmentation into Memory ---
    print("\nStep 7: Loading labels into memory...")
    first_pass_segmentation = np.array(first_pass_segmentation_memmap)
    labels_memmap_filename = first_pass_segmentation_memmap.filename
    del first_pass_segmentation_memmap; gc.collect()
    try: os.unlink(labels_memmap_filename); rmtree(labels_temp_dir)
    except Exception as e: print(f"Warn: Could not clean label temp files: {e}")
    gc.collect()
    print("Labels loaded into memory.")

    # --- Step 8: Trim Edge Artifacts ---
    print("\n--- Trimming Edge Artifacts by Distance ---")
    hull_boundary_mask = np.zeros(original_shape, dtype=bool) # Initialize
    trimmed_voxels_mask = np.zeros(original_shape, dtype=bool) # Initialize
    hull_stack = None # Ensure cleanup

    try:
        # --- 8a. Generate Hull Boundary ---
        hull_boundary_mask, hull_stack = generate_hull_boundary_and_stack(
            volume, hull_erosion_iterations=hull_erosion_iterations
        )
        if hull_stack is not None: del hull_stack; gc.collect() # Don't need stack

        if hull_boundary_mask is None or not np.any(hull_boundary_mask):
            print("  Hull boundary mask empty/not generated. Skipping edge trimming.")
            hull_boundary_mask = np.zeros(original_shape, dtype=bool) # Ensure return is valid shape
        else:
            # --- 8b. Call Trimming Helper Function ---
            # Pass segmentation array (will be modified in-place)
            trimmed_voxels_mask = trim_object_edges_by_distance(
                segmentation=first_pass_segmentation, # Pass the array to modify
                hull_boundary_mask=hull_boundary_mask,
                spacing=spacing,
                distance_threshold=edge_trim_distance_threshold,
                min_remaining_size=min_size_voxels,
                chunk_size_z=edge_distance_chunk_size_z
            )

    except Exception as e:
         print(f"\nERROR during edge artifact trimming: {e}")
         import traceback; traceback.print_exc()
         # Ensure masks are valid empty defaults on error
         hull_boundary_mask = np.zeros(original_shape, dtype=bool)
         trimmed_voxels_mask = np.zeros(original_shape, dtype=bool)
         gc.collect()
    finally:
         # Cleanup hull_stack if it somehow still exists
         if 'hull_stack' in locals() and hull_stack is not None:
             del hull_stack; gc.collect()


    # --- Finalization ---
    print(f"\n--- First Pass Segmentation Finished ---")
    # Recalculate final count after trimming and remnant removal
    final_unique_labels = np.unique(first_pass_segmentation)
    final_object_count = len(final_unique_labels[final_unique_labels != 0])
    print(f"Final labeled mask shape: {first_pass_segmentation.shape}")
    print(f"Number of labeled objects remaining: {final_object_count}")
    final_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Final memory usage: {final_mem:.2f} MB")

    return first_pass_segmentation, first_pass_params, hull_boundary_mask

def extract_soma_masks(segmentation_mask, 
                      small_object_percentile=50,  # Changed to percentile
                      thickness_percentile=80):
    """
    Memory-efficient soma extraction with percentile-based small object removal and label reassignment.
    
    Parameters:
    - segmentation_mask: 3D numpy array with labeled segments
    - small_object_percentile: Percentile of object volumes to keep (e.g., 50 keeps top 50%)
    - thickness_percentile: Percentile for thickness-based soma detection
    """
    
    # Create output soma mask
    soma_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
    
    # Get unique labels, excluding background
    unique_labels = np.unique(segmentation_mask)[1:]
    
    # Keep track of the next available label for reassignment
    next_label = np.max(unique_labels) + 1 if len(unique_labels) > 0 else 1
    
    # Process each label
    for label in tqdm(unique_labels):
        # Extract current cell mask
        cell_mask = segmentation_mask == label
        
        # Get bounding box for the cell
        props = regionprops(cell_mask.astype(int))[0]
        bbox = props.bbox
        
        # Extract subvolumes using bounding box with padding
        z_min, y_min, x_min, z_max, y_max, x_max = bbox
        z_min = max(0, z_min - 2)
        y_min = max(0, y_min - 2)
        x_min = max(0, x_min - 2)
        z_max = min(segmentation_mask.shape[0], z_max + 2)
        y_max = min(segmentation_mask.shape[1], y_max + 2)
        x_max = min(segmentation_mask.shape[2], x_max + 2)
        
        # Extract subarrays
        cell_mask_subvolume = cell_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Compute distance transform on cell mask subvolume
        distance_map = ndimage.distance_transform_edt(cell_mask_subvolume)
        
        # Compute thickness threshold
        thickness_threshold = np.percentile(distance_map[cell_mask_subvolume], thickness_percentile)
        
        # Create max thickness mask
        max_thickness_mask = np.zeros_like(distance_map, dtype=bool)
        max_thickness_mask[np.logical_and(distance_map >= thickness_threshold, cell_mask_subvolume)] = True
        
        # Label connected components in the subvolume
        labeled_somas, num_features = ndimage.label(max_thickness_mask)
        
        # If no somas detected, skip to next label
        if num_features == 0:
            continue
        
        # Map back to full volume
        full_max_thickness_mask = np.zeros_like(cell_mask, dtype=np.int32)
        full_max_thickness_mask[z_min:z_max, y_min:y_max, x_min:x_max] = labeled_somas
        
        # Get properties of connected components
        soma_props = regionprops(full_max_thickness_mask)
        
        # If only one object, keep it regardless of size
        if num_features == 1:
            soma_mask[full_max_thickness_mask > 0] = label
        else:
            # Compute volumes of all objects
            volumes = [prop.area for prop in soma_props]
            
            # Calculate the volume threshold based on percentile
            volume_threshold = np.percentile(volumes, small_object_percentile)
            
            # Filter objects above the percentile and reassign labels
            for prop in soma_props:
                if prop.area >= volume_threshold:  # Keep if above threshold
                    soma_mask[full_max_thickness_mask == prop.label] = next_label
                    next_label += 1
    
    return soma_mask

def separate_multi_soma_cells(segmentation_mask, intensity_volume, soma_mask, min_size_threshold=100):
    """
    Separates cell segmentations with multiple somas into distinct masks by using watershed
    transform with distance transforms to ensure separation along thinnest regions between somas.
    
    Parameters:
    - segmentation_mask: 3D numpy array with labeled cell segments
    - intensity_volume: 3D numpy array with original intensity values
    - soma_mask: 3D numpy array with labeled soma segments (output from extract_soma_masks)
    - min_size_threshold: Minimum voxel size for a separated component (smaller ones are merged unless original)
    
    Returns:
    - separated_mask: 3D numpy array with updated cell segmentations
    """
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops
    from skimage.segmentation import watershed
    from tqdm import tqdm
    
    # Create output mask, initially copying the original segmentation
    separated_mask = np.copy(segmentation_mask).astype(np.int32)
    
    # Get unique cell labels and their original sizes from segmentation_mask
    unique_cell_labels = np.unique(segmentation_mask)[1:]
    original_sizes = {lbl: np.sum(segmentation_mask == lbl) for lbl in unique_cell_labels}
    
    # Keep track of the next available label
    next_label = np.max(segmentation_mask) + 1 if len(unique_cell_labels) > 0 else 1
    
    # Process each cell
    for cell_label in tqdm(unique_cell_labels):
        # Extract current cell mask
        cell_mask = segmentation_mask == cell_label
        
        # Get bounding box for the cell
        props = regionprops(cell_mask.astype(int))[0]
        bbox = props.bbox
        z_min, y_min, x_min, z_max, y_max, x_max = bbox
        
        # Add slight padding
        z_min = max(0, z_min - 2)
        y_min = max(0, y_min - 2)
        x_min = max(0, x_min - 2)
        z_max = min(segmentation_mask.shape[0], z_max + 2)
        y_max = min(segmentation_mask.shape[1], y_max + 2)
        x_max = min(segmentation_mask.shape[2], x_max + 2)
        
        # Extract subvolumes
        cell_mask_sub = cell_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        intensity_sub = intensity_volume[z_min:z_max, y_min:y_max, x_min:x_max]
        cell_soma_sub = soma_mask[z_min:z_max, y_min:y_max, x_min:x_max] * cell_mask_sub
        
        # Get unique soma labels within this cell, excluding background
        soma_labels = np.unique(cell_soma_sub)[1:]
        
        # Skip if no somas or only one soma
        if len(soma_labels) <= 1:
            continue
        
        # Number of somas
        num_somas = len(soma_labels)
        print(f"Cell {cell_label} has {num_somas} somas, separating...")
        
        # Create markers for watershed segmentation
        # First, ensure soma regions are well defined
        soma_markers = np.zeros_like(cell_mask_sub, dtype=np.int32)
        for i, soma_label in enumerate(soma_labels):
            # Mark each soma with a unique index starting from 1
            soma_region = cell_soma_sub == soma_label
            # Dilate soma slightly to ensure good markers
            soma_region = ndimage.binary_dilation(soma_region, iterations=1)
            soma_markers[soma_region] = i + 1
        
        # Compute distance transform of the cell mask
        # This helps identify thin regions (smaller distance values)
        # The negative distance transform has peaks at the center of large regions
        distance_transform = ndimage.distance_transform_edt(cell_mask_sub)
        
        # Create a special weighting for somas to avoid cutting through them
        # Make distances through somas artificially high
        soma_weighting = np.zeros_like(distance_transform)
        for soma_label in soma_labels:
            soma_region = cell_soma_sub == soma_label
            soma_weighting[soma_region] = 1000  # Very high value to avoid cutting through somas
        
        # Modified distance transform that penalizes paths through somas
        modified_distance = distance_transform + soma_weighting
        
        # Apply watershed with markers (somas) and weights (distance transform)
        # This will separate along the thinnest regions (valleys in the distance transform)
        watershed_result = watershed(modified_distance, soma_markers, mask=cell_mask_sub)
        
        # Create temp_mask with watershed result
        temp_mask = np.zeros_like(watershed_result, dtype=np.int32)
        label_map = [cell_label] + [next_label + i for i in range(num_somas - 1)]
        
        # Map watershed labels to cell labels
        for i in range(num_somas):
            region_mask = watershed_result == (i + 1)
            temp_mask[region_mask] = label_map[i]
        
        # Ensure continuity: Check for discontinuous components
        for i, lbl in enumerate(label_map):
            lbl_mask = temp_mask == lbl
            labeled_components, num_components = ndimage.label(lbl_mask)
            if num_components > 1:
                print(f"Warning: soma {i} of cell {cell_label} is discontinuous, merging...")
                props = regionprops(labeled_components)
                main_component = max(props, key=lambda p: p.area).label
                main_mask = labeled_components == main_component
                for prop in props:
                    if prop.label != main_component:
                        dilated = ndimage.binary_dilation(labeled_components == prop.label, iterations=1)
                        touching_labels = np.unique(temp_mask[dilated & (labeled_components != prop.label)])
                        valid_touching = [l for l in touching_labels if l != 0 and l != lbl]
                        if valid_touching:
                            temp_mask[labeled_components == prop.label] = valid_touching[0]
                        else:
                            temp_mask[labeled_components == prop.label] = lbl
                            temp_mask[main_mask] = lbl
        
        # Enforce size threshold, preserving original small regions
        final_labels = np.unique(temp_mask)[1:]  # Exclude background
        for lbl in final_labels:
            lbl_mask = temp_mask == lbl
            size = np.sum(lbl_mask)
            if size < min_size_threshold and size != original_sizes.get(lbl, float('inf')):
                print(f"Merging soma {lbl} of cell {cell_label} due to size {size}")
                # Merge if below threshold and not the original size
                dilated = ndimage.binary_dilation(lbl_mask, iterations=1)
                touching_labels = np.unique(temp_mask[dilated & ~lbl_mask])
                valid_touching = [l for l in touching_labels if l != 0 and np.sum(temp_mask == l) >= min_size_threshold]
                if valid_touching:
                    temp_mask[lbl_mask] = valid_touching[0]  # Merge with largest valid neighbor
                else:
                    temp_mask[lbl_mask] = label_map[0]  # Merge with original label
        
        # Update next_label based on used labels
        used_labels = np.unique(temp_mask)[1:]
        if len(used_labels) > 0:
            next_label = max(next_label, np.max(used_labels) + 1)
        
        # Map back to full volume
        # Get the current subvolume from the separated_mask
        full_subvol = separated_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        # Only replace voxels that belong to the current cell
        current_cell_voxels = cell_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        # Update only the voxels belonging to the current cell
        full_subvol[current_cell_voxels] = temp_mask[current_cell_voxels]
        # Write back to the full separated_mask
        separated_mask[z_min:z_max, y_min:y_max, x_min:x_max] = full_subvol
    
    return separated_mask