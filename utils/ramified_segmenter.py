import numpy as np
from scipy import ndimage
# Make sure all necessary ndimage functions are imported
from scipy.ndimage import (gaussian_filter, distance_transform_edt,
                           binary_erosion, generate_binary_structure)
from scipy.ndimage import (distance_transform_edt, label as ndimage_label, find_objects,
                           binary_opening, binary_closing)
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
        input_volume_memmap.flush(); print(f"  3D pre-smoothing done.")
    else:
        print("  Skipping initial 3D smoothing.");
        if isinstance(volume, np.memmap): input_volume_memmap = volume; input_memmap_path = volume.filename
        else:
            print(f"  Converting input volume to temporary memmap..."); input_memmap_dir = tempfile.mkdtemp(prefix="input_volume_"); temp_dirs_to_clean.append(input_memmap_dir)
            input_volume_memmap, input_memmap_path, _ = create_memmap(data=volume, directory=input_memmap_dir)
            source_volume_cleaned = True # Mark this temp dir for cleaning later

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


def generate_hull_boundary_and_stack(
    volume,                     # Original intensity volume (for Otsu threshold)
    cell_mask,                  # Labeled or boolean mask of segmented cells
    hull_erosion_iterations=1,  # Iterations for final boundary erosion
    smoothing_iterations=1      # Iterations for 3D closing/opening to smooth Z-axis
    ):
    """
    Generates a 3D hull stack and its boundary based on tissue intensity
    and an existing cell segmentation, ensuring cells are included and
    smoothing the hull across the Z-axis.

    Steps:
    1. Creates a tissue mask using Otsu thresholding on the input volume.
    2. For each Z-slice:
        a. Combines the tissue mask slice with the cell mask slice (logical OR).
        b. Computes the 2D convex hull of this combined mask.
    3. Stacks the 2D hulls into an initial 3D `hull_stack`.
    4. (Optional) Applies 3D morphological closing then opening to smooth
       the `hull_stack` along the Z-axis, reducing slice-to-slice jumps.
    5. Erodes the smoothed `hull_stack` using a 3D structure.
    6. Calculates the `hull_boundary` as the difference between the smoothed
       hull and the eroded hull.

    Parameters:
    -----------
    volume : ndarray
        Input 3D intensity image volume. Used for Otsu thresholding.
    cell_mask : ndarray (bool or int)
        A mask (boolean or labeled integers) where non-zero values indicate
        segmented cells that *must* be contained within the hull.
        Must have the same shape as `volume`.
    hull_erosion_iterations : int, optional
        Number of iterations for the final 3D binary erosion to define the
        thickness of the boundary. Defaults to 1.
    smoothing_iterations : int, optional
        Number of iterations for 3D binary closing and opening applied to the
        initial hull stack to smooth it along the Z-axis. Set to 0 to disable.
        Defaults to 1.

    Returns:
    --------
    hull_boundary : ndarray (bool)
        Boolean mask indicating the boundary region of the final smoothed hull.
        Shape matches the input volume. Returns None if hull generation fails.
    smoothed_hull_stack : ndarray (bool)
        The final, smoothed 3D hull stack (after potential closing/opening).
        Shape matches the input volume. Returns None if hull generation fails.
    """
    print("\n--- Generating Smoothed Hull Boundary and Stack ---")
    original_shape = volume.shape
    if cell_mask.shape != original_shape:
        raise ValueError(f"Shape mismatch: volume {original_shape} vs cell_mask {cell_mask.shape}")

    print(f"Input shape: {original_shape}")
    print(f"Smoothing iterations: {smoothing_iterations}")
    print(f"Boundary erosion iterations: {hull_erosion_iterations}")
    initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Memory usage at start: {initial_mem:.2f} MB")

    # --- Step 1: Create Tissue Mask using Otsu ---
    print("Step 1: Creating base tissue mask using Otsu threshold...")
    tissue_mask = np.zeros(original_shape, dtype=bool)
    try:
        # Sample for Otsu to avoid memory issues on huge arrays
        otsu_sample_size = min(2_000_000, volume.size)
        if otsu_sample_size == volume.size:
            otsu_samples = volume.ravel()
        else:
            # Ensure we sample non-zero voxels if possible, otherwise random
            non_zero_indices = np.flatnonzero(volume)
            if len(non_zero_indices) > otsu_sample_size:
                 sample_indices = np.random.choice(non_zero_indices, otsu_sample_size, replace=False)
            elif len(non_zero_indices) > 0:
                 sample_indices = non_zero_indices # Use all non-zeros
            else: # If all zero, just sample randomly
                 sample_indices = np.random.choice(volume.size, otsu_sample_size, replace=False)

            otsu_coords = np.unravel_index(sample_indices, volume.shape)
            otsu_samples = volume[otsu_coords]

        if np.all(otsu_samples == otsu_samples[0]):
            # Handle constant intensity case
            tissue_thresh = otsu_samples[0]
            print(f"  Warning: Sampled intensity values are constant ({tissue_thresh}). Threshold set.")
        elif otsu_samples.size > 0:
            tissue_thresh = threshold_otsu(otsu_samples)
            print(f"  Otsu tissue threshold determined: {tissue_thresh:.2f}")
        else:
             print("  Warning: No valid samples for Otsu. Using threshold 0.")
             tissue_thresh = 0

        # Apply threshold chunk-wise if needed, but direct comparison is often fine
        tissue_mask = volume > tissue_thresh
        del otsu_samples # Free memory
        gc.collect()

        if not np.any(tissue_mask):
            print("  Warning: Otsu tissue mask is empty based on threshold.")
            # Continue, as the cell_mask might still define the hull

    except Exception as e:
        print(f"  Error during Otsu thresholding: {e}. Proceeding without tissue mask.")
        tissue_mask = np.zeros(original_shape, dtype=bool) # Ensure it exists

    # --- Step 2: Combine Masks and Generate Slice-wise 2D Hulls ---
    print("\nStep 2: Generating initial hull stack from slice-wise combined masks...")
    # Ensure cell_mask is boolean
    cell_mask_bool = cell_mask > 0
    initial_hull_stack = np.zeros(original_shape, dtype=bool)

    for z in tqdm(range(original_shape[0]), desc="  Processing Slices for Hull"):
        tissue_slice = tissue_mask[z, :, :]
        cell_slice = cell_mask_bool[z, :, :]

        # Combine tissue and cell masks for this slice
        combined_mask_slice = tissue_slice | cell_slice

        if np.any(combined_mask_slice):
            try:
                # convex_hull_image needs contiguous array
                if not combined_mask_slice.flags['C_CONTIGUOUS']:
                    combined_mask_slice = np.ascontiguousarray(combined_mask_slice)
                initial_hull_stack[z, :, :] = convex_hull_image(combined_mask_slice)
            except Exception as e:
                print(f"\nWarning: Failed to compute convex hull for slice {z}: {e}")
                # Leave slice as zeros in the stack
        # else: Slice remains False (empty)

    del tissue_mask, cell_mask_bool, combined_mask_slice, tissue_slice, cell_slice # Free memory
    gc.collect()
    print("  Initial slice-wise hull stack generated.")

    if not np.any(initial_hull_stack):
         print("  Warning: Initial hull stack is empty after processing all slices. Cannot proceed.")
         return None, None # Return None if hull is empty

    # --- Step 3: Smooth Hull Stack in 3D (Optional) ---
    smoothed_hull_stack = initial_hull_stack # Start with the initial stack
    if smoothing_iterations > 0:
        print(f"\nStep 3: Smoothing hull stack with {smoothing_iterations} iteration(s) of 3D closing/opening...")
        struct_3d = generate_binary_structure(3, 1) # Connectivity=1 for 3x3x3 cube
        try:
            # Closing fills holes and gaps
            print("  Applying binary closing...")
            smoothed_hull_stack = binary_closing(smoothed_hull_stack, structure=struct_3d, iterations=smoothing_iterations, border_value=0)
            # Opening removes small protrusions/thin connections
            print("  Applying binary opening...")
            smoothed_hull_stack = binary_opening(smoothed_hull_stack, structure=struct_3d, iterations=smoothing_iterations, border_value=0)
            print("  3D smoothing complete.")
        except MemoryError:
            print("  Warning: MemoryError during 3D smoothing. Using unsmoothed hull stack.")
            smoothed_hull_stack = initial_hull_stack # Fallback to original
        except Exception as e:
            print(f"  Warning: Error during 3D smoothing ({e}). Using unsmoothed hull stack.")
            smoothed_hull_stack = initial_hull_stack # Fallback to original
        gc.collect()
    else:
        print("\nStep 3: Skipping 3D hull stack smoothing (smoothing_iterations=0).")

    # --- Step 4: Erode Smoothed Hull Stack ---
    hull_boundary = np.zeros(original_shape, dtype=bool) # Initialize boundary mask
    if hull_erosion_iterations > 0:
        print(f"\nStep 4: Eroding smoothed hull stack ({hull_erosion_iterations} iterations)...")
        struct_3d = generate_binary_structure(3, 1) # Use same structure for consistency
        try:
            eroded_hull_stack = binary_erosion(smoothed_hull_stack, structure=struct_3d, iterations=hull_erosion_iterations, border_value=0)
            print("  Erosion complete.")

            # --- Step 5: Calculate Hull Boundary ---
            print("\nStep 5: Calculating final hull boundary...")
            hull_boundary = smoothed_hull_stack & (~eroded_hull_stack)
            boundary_voxel_count = np.sum(hull_boundary)
            print(f"  Hull boundary mask created ({boundary_voxel_count} voxels).")
            del eroded_hull_stack # Free memory

        except MemoryError:
            print("  Warning: MemoryError during hull erosion or boundary calculation. Boundary will be empty.")
            # hull_boundary remains all False
        except Exception as e:
            print(f"  Warning: Error during hull erosion or boundary calculation ({e}). Boundary will be empty.")
            # hull_boundary remains all False
        gc.collect()
    else:
        print("\nSteps 4 & 5: Skipping hull erosion and boundary calculation (hull_erosion_iterations=0).")
        # hull_boundary remains all False

    final_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Memory usage at end: {final_mem:.2f} MB")
    print("--- Hull generation finished ---")

    # Return the boundary AND the (potentially smoothed) hull stack
    return hull_boundary, smoothed_hull_stack


# Helper function for memory estimation
def estimate_memory(shape, *dtypes, overhead_factor=2.5):
    """Estimates memory needed for arrays of given shape and dtypes."""
    bytes_needed = 0
    for dtype in dtypes:
        bytes_needed += np.prod(shape) * np.dtype(dtype).itemsize
    return bytes_needed * overhead_factor

def trim_object_edges_by_distance(
    segmentation_memmap,     # Labeled segmentation MEMMAP (opened r+ OR w+)
    original_volume,         # Original intensity volume (ndarray or memmap)
    hull_boundary_mask,      # Boolean mask of the hull boundary (in-memory)
    spacing,                 # Voxel spacing for distance calculation
    distance_threshold,      # Physical distance threshold
    global_brightness_cutoff,# GLOBAL intensity value threshold for trimming
    min_remaining_size=10,   # Minimum size for object remnants after trimming/relabeling
    chunk_size_z=32          # Chunk size for chunked processing
    ):
    """
    Removes voxels from labeled objects that are BOTH close to the hull boundary
    AND below a GLOBAL brightness cutoff.
    Optionally removes objects that become too small after trimming.
    If trimming disconnects an object, the resulting components are relabeled.
    Operates IN-PLACE on the input segmentation_memmap.
    Uses chunked distance transform calculation exclusively to save RAM.
    """
    print(f"\n--- Trimming object edges (Dist < {distance_threshold:.2f} AND Brightness < {global_brightness_cutoff:.4f}) ---")
    original_shape = segmentation_memmap.shape
    trimmed_voxels_mask = np.zeros(original_shape, dtype=bool) # Track in memory

    if not np.any(hull_boundary_mask):
        print("  Hull boundary mask is empty. No trimming performed.")
        return trimmed_voxels_mask

    # Check if memmap is writable... (as before)
    writable_modes = ['r+', 'w+']
    if segmentation_memmap.mode not in writable_modes:
         raise ValueError(f"segmentation_memmap must be opened in a writable mode ({writable_modes}) for in-place modification. Found mode: '{segmentation_memmap.mode}'")

    # Get original labels present before trimming... (as before)
    print("  Finding unique labels before trimming...")
    try:
        original_labels_present = np.unique(segmentation_memmap)
        original_labels_present = original_labels_present[original_labels_present != 0]
        print(f"  Found {len(original_labels_present)} unique labels initially.")
        if len(original_labels_present) == 0:
             print("  No objects found in segmentation. Skipping trimming.")
             return trimmed_voxels_mask
    except MemoryError:
         raise MemoryError("Failed to get initial unique labels from segmentation memmap.")
    except Exception as e:
         raise RuntimeError(f"Failed to get initial unique labels: {e}") from e


    # --- Chunked Processing for Distance Calc and Combined Trimming ---
    print("\n  Executing chunked distance transform and combined (distance+global brightness) trimming...")
    gc.collect()

    num_chunks = math.ceil(original_shape[0] / chunk_size_z)
    total_trimmed_in_chunks = 0
    start_chunked_time = time.time()

    if 'hull_boundary_mask' not in locals() or hull_boundary_mask is None:
         raise RuntimeError("Hull boundary mask is not available for chunked processing.")
    hull_mem_gb = hull_boundary_mask.nbytes / (1024**3)
    print(f"  Processing {num_chunks} chunks (chunk_size_z={chunk_size_z}). Hull mask size: {hull_mem_gb:.2f} GB")

    for i in tqdm(range(num_chunks), desc="  Processing Chunks (Trim)"):
        z_start = i * chunk_size_z
        z_end = min((i + 1) * chunk_size_z, original_shape[0])
        if z_start >= z_end: continue

        try:
            # 1. Calculate Distance Transform for the chunk
            inv_boundary_chunk = ~hull_boundary_mask[z_start:z_end]
            edt_chunk = None
            # ... (EDT calculation as before) ...
            if not np.any(inv_boundary_chunk): edt_chunk = np.zeros(inv_boundary_chunk.shape, dtype=np.float32)
            elif not np.all(inv_boundary_chunk): edt_chunk = distance_transform_edt(inv_boundary_chunk, sampling=spacing).astype(np.float32)
            else: edt_chunk = np.full(inv_boundary_chunk.shape, np.finfo(np.float32).max, dtype=np.float32)
            close_to_boundary_chunk = (edt_chunk < distance_threshold)
            del inv_boundary_chunk, edt_chunk
            gc.collect()

            # 2. Load segmentation and intensity chunks
            if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None:
                 raise IOError(f"Segmentation memmap became invalid before reading chunk {i}")
            # Read seg chunk data directly into memory for boolean ops
            seg_chunk_data = np.array(segmentation_memmap[z_start:z_end])
            intensity_chunk = original_volume[z_start:z_end]

            # 3. Identify voxels below GLOBAL brightness cutoff
            dim_voxels_chunk = (intensity_chunk < global_brightness_cutoff)
            del intensity_chunk # Free intensity chunk memory
            gc.collect()
            print(f"Number of dim voxels in chunk {i}: {np.sum(dim_voxels_chunk)}")
            print(f"Number of close-to-boundary voxels in chunk {i}: {np.sum(close_to_boundary_chunk)}")
            print(f"Number of dim and close-to-boundary voxels in chunk {i}: {np.sum(dim_voxels_chunk & close_to_boundary_chunk)}")

            # 4. Combine criteria: Must be object, close, AND dim
            trim_target_mask_chunk = (seg_chunk_data > 0) & close_to_boundary_chunk & dim_voxels_chunk
            del seg_chunk_data, close_to_boundary_chunk, dim_voxels_chunk # Free memory
            gc.collect()

            # 5. Apply Trimming (Set identified voxels to 0)
            num_trimmed_chunk = np.sum(trim_target_mask_chunk)
            if num_trimmed_chunk > 0:
                 seg_chunk_view_write = segmentation_memmap[z_start:z_end]
                 trimmed_voxels_mask[z_start:z_end][trim_target_mask_chunk] = True
                 seg_chunk_view_write[trim_target_mask_chunk] = 0
                 segmentation_memmap.flush()
                 total_trimmed_in_chunks += num_trimmed_chunk
                 del seg_chunk_view_write

            # --- Clean up chunk variables ---
            del trim_target_mask_chunk
            gc.collect()

        # ... (Error handling: MemoryError, other Exception - as before) ...
        except MemoryError as mem_err:
            # ... (Log memory details) ...
            raise MemoryError(f"Chunked processing failed at chunk {i} due to MemoryError.") from mem_err
        except Exception as chunk_err:
            # ... (Log traceback) ...
             raise RuntimeError(f"Chunked processing failed at chunk {i}.") from chunk_err


    print(f"  Chunked trimming applied. Total voxels trimmed: {total_trimmed_in_chunks} in {time.time() - start_chunked_time:.2f}s.")
    gc.collect()


    # --- Step 2: Relabel Disconnected Components ---
    # ... (This section remains exactly the same as before) ...
    print("\n  Checking for and relabeling disconnected components...")
    start_relabel_time = time.time()
    try:
        print("    Finding max label after trimming...")
        max_label = 0
        chunk_size_max = min(500, original_shape[0]) if original_shape[0] > 0 else 1
        for i in tqdm(range(0, original_shape[0], chunk_size_max), desc="    Finding Max Label", disable=original_shape[0]<=chunk_size_max):
             seg_chunk = segmentation_memmap[i:min(i+chunk_size_max, original_shape[0])]
             chunk_max = np.max(seg_chunk); del seg_chunk
             if chunk_max > max_label: max_label = chunk_max
        next_available_label = max_label + 1
        print(f"    Max label found: {max_label}. Next new label: {next_available_label}")
    except MemoryError: raise MemoryError("Failed to find max label for relabeling.")
    except Exception as e: raise RuntimeError(f"Failed to find max label for relabeling: {e}") from e
    relabel_count = 0
    for label_val in tqdm(original_labels_present, desc="  Relabeling Components"):
        if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None: raise IOError(f"Segmentation memmap became invalid before relabeling label {label_val}")
        try: label_mask_full = (segmentation_memmap == label_val); locations = find_objects(label_mask_full.astype(np.uint8))
        except MemoryError: print(f"\n!!! MemoryError creating full mask for find_objects (label {label_val}). Skipping relabeling."); del label_mask_full; gc.collect(); continue
        except Exception as find_obj_err: print(f"\n!!! Error using find_objects for label {label_val}: {find_obj_err}. Skipping relabeling."); 
        if 'label_mask_full' in locals(): del label_mask_full; gc.collect(); continue
        if not locations: 
            if 'label_mask_full' in locals(): del label_mask_full; gc.collect(); continue
        loc = locations[0]; del locations
        label_mask_sub = label_mask_full[loc]; del label_mask_full; gc.collect()
        if not np.any(label_mask_sub): del label_mask_sub; gc.collect(); continue
        labeled_components_sub, num_components = ndimage_label(label_mask_sub, structure=generate_binary_structure(3,1))
        if num_components > 1:
            relabel_count += (num_components - 1)
            seg_sub_view_write = segmentation_memmap[loc]
            for c_idx in range(2, num_components + 1):
                component_mask_sub = (labeled_components_sub == c_idx)
                seg_sub_view_write[component_mask_sub] = next_available_label
                next_available_label += 1
            segmentation_memmap.flush()
            del seg_sub_view_write
        del loc, label_mask_sub, labeled_components_sub; gc.collect()
    print(f"  Relabeling finished. Created {relabel_count} new labels in {time.time() - start_relabel_time:.2f}s."); gc.collect()



    # --- Step 3: Optional: Remove Small Remnants (based on final labels) ---
    if min_remaining_size > 0:
        print(f"  Removing remnants smaller than {min_remaining_size} voxels (on Memmap, post-relabel)...")
        # ... (The existing remnant removal logic should work here,
        #      operating on the now potentially relabeled segmentation_memmap) ...
        start_rem = time.time()
        print("    Creating boolean mask (chunked)...")
        if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None:
            raise IOError("Segmentation memmap became invalid before remnant removal step")
        bool_seg_temp_dir = tempfile.mkdtemp(prefix="bool_remnant_")
        bool_seg_path = os.path.join(bool_seg_temp_dir, 'temp_bool_seg.dat')
        bool_seg_memmap = None
        try:
            bool_seg_memmap = np.memmap(bool_seg_path, dtype=bool, mode='w+', shape=original_shape)
            chunk_size_rem = min(100, original_shape[0]) if original_shape[0] > 0 else 1
            any_objects_found = False
            for i in tqdm(range(0, original_shape[0], chunk_size_rem), desc="    Bool Mask Gen"):
                z_start = i; z_end = min(i + chunk_size_rem, original_shape[0])
                if z_start >= z_end: continue
                seg_chunk = segmentation_memmap[z_start:z_end]
                bool_chunk = seg_chunk > 0
                bool_seg_memmap[z_start:z_end] = bool_chunk
                if not any_objects_found and np.any(bool_chunk): any_objects_found = True
                del seg_chunk, bool_chunk
            bool_seg_memmap.flush(); gc.collect()
            num_remnants_removed = 0
            if any_objects_found:
                print("    Loading boolean mask for small object removal...")
                try:
                    bool_seg_in_memory = np.array(bool_seg_memmap)
                    del bool_seg_memmap; bool_seg_memmap = None; gc.collect()
                    os.unlink(bool_seg_path); rmtree(bool_seg_temp_dir); bool_seg_temp_dir = None
                    print("    Applying remove_small_objects...")
                    cleaned_bool_seg = remove_small_objects(bool_seg_in_memory, min_size=min_remaining_size, connectivity=1)
                    del bool_seg_in_memory; gc.collect()
                    print("    Applying removal mask to segmentation memmap (chunked)...")
                    chunk_size_apply_rem = min(100, original_shape[0]) if original_shape[0] > 0 else 1
                    for i in tqdm(range(0, original_shape[0], chunk_size_apply_rem), desc="    Applying Removal"):
                        z_start = i; z_end = min(i + chunk_size_apply_rem, original_shape[0])
                        if z_start >= z_end: continue
                        if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None: raise IOError("Segmentation memmap became invalid during remnant removal application")
                        seg_chunk_view = segmentation_memmap[z_start:z_end]
                        cleaned_bool_chunk = cleaned_bool_seg[z_start:z_end]
                        small_remnant_mask_chunk = (seg_chunk_view > 0) & (~cleaned_bool_chunk)
                        chunk_remnants_count = np.sum(small_remnant_mask_chunk)
                        if chunk_remnants_count > 0:
                            seg_chunk_view[small_remnant_mask_chunk] = 0
                            segmentation_memmap.flush()
                            num_remnants_removed += chunk_remnants_count
                        del cleaned_bool_chunk, small_remnant_mask_chunk
                    del cleaned_bool_seg; gc.collect()
                    print(f"  Removed {num_remnants_removed} voxels belonging to small remnants in {time.time()-start_rem:.2f}s.")
                except MemoryError: print("    MemoryError loading boolean mask for small object removal. Skipping remnant removal."); gc.collect()
                except Exception as e_rem: print(f"    Error during small remnant removal processing: {e_rem}. Skipping."); gc.collect()
            else: print("  Segmentation memmap appears empty after trimming/relabeling, skipping remnant removal.")
        finally:
             if 'bool_seg_memmap' in locals() and bool_seg_memmap is not None and hasattr(bool_seg_memmap, '_mmap'): del bool_seg_memmap; gc.collect()
             if 'bool_seg_temp_dir' in locals() and bool_seg_temp_dir is not None and os.path.exists(bool_seg_temp_dir):
                 try: rmtree(bool_seg_temp_dir, ignore_errors=True)
                 except Exception as e_final_clean: print(f"    Warn: Error cleaning up remnant temp dir {bool_seg_temp_dir}: {e_final_clean}")
             gc.collect()


    print("--- Edge trimming (combined criteria + relabeling) finished ---")
    # The segmentation_memmap has been modified in-place.
    return trimmed_voxels_mask # Return mask of initially trimmed voxels



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
                     segmentation_memmap=labels_memmap_rplus, # Use the handle assured to be writable
                     original_volume=volume,
                     hull_boundary_mask=hull_boundary_mask, # Use the mask from 7a
                     spacing=spacing,
                     distance_threshold=edge_trim_distance_threshold,
                     global_brightness_cutoff=global_brightness_cutoff,
                     min_remaining_size=min_size_voxels,
                     chunk_size_z=edge_distance_chunk_size_z
                 )
                 labels_memmap_rplus.flush() # Ensure trimming changes are saved
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