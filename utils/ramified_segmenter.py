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


# Helper function for memory estimation
def estimate_memory(shape, *dtypes, overhead_factor=2.5):
    """Estimates memory needed for arrays of given shape and dtypes."""
    bytes_needed = 0
    for dtype in dtypes:
        bytes_needed += np.prod(shape) * np.dtype(dtype).itemsize
    return bytes_needed * overhead_factor

def trim_object_edges_by_distance(
    segmentation_memmap,     # Labeled segmentation MEMMAP (opened r+ OR w+)
    hull_boundary_mask,      # Boolean mask of the hull boundary (in-memory)
    spacing,                 # Voxel spacing for physical distance
    distance_threshold,      # Physical distance threshold
    min_remaining_size=10,   # Minimum size for object remnants after trimming
    chunk_size_z=32,         # Chunk size for chunked EDT/application
    # ram_safety_factor is no longer needed for EDT check
    ):
    """
    Removes voxels from labeled objects that are closer than a threshold
    distance to the hull boundary. Optionally removes objects that become
    too small after trimming. Operates IN-PLACE on the input segmentation_memmap.
    *** Uses chunked distance transform calculation exclusively to save RAM. ***
    """
    print(f"\n--- Trimming object edges closer than {distance_threshold:.2f} units to hull boundary (on Memmap, Chunked EDT) ---")
    original_shape = segmentation_memmap.shape
    trimmed_voxels_mask = np.zeros(original_shape, dtype=bool) # Track in memory

    if not np.any(hull_boundary_mask):
        print("  Hull boundary mask is empty. No trimming performed.")
        return trimmed_voxels_mask # Return empty mask

    # --- Check if memmap is writable ---
    writable_modes = ['r+', 'w+']
    if segmentation_memmap.mode not in writable_modes:
         raise ValueError(f"segmentation_memmap must be opened in a writable mode ({writable_modes}) for in-place modification. Found mode: '{segmentation_memmap.mode}'")

    # --- Skip Full EDT Attempt - Proceed Directly to Chunked Processing ---
    print("\n  Executing chunked distance transform and trimming...")
    gc.collect() # Free memory before starting chunks

    num_chunks = math.ceil(original_shape[0] / chunk_size_z)
    total_trimmed_in_chunks = 0
    start_chunked_time = time.time()

    # Need hull_boundary_mask accessible here - it should still be in memory
    if 'hull_boundary_mask' not in locals() or hull_boundary_mask is None:
         raise RuntimeError("Hull boundary mask is not available for chunked processing.")
    # Log hull mask memory usage (approximate)
    hull_mem_gb = hull_boundary_mask.nbytes / (1024**3)
    print(f"  Processing {num_chunks} chunks (chunk_size_z={chunk_size_z}). Hull mask size: {hull_mem_gb:.2f} GB")


    for i in tqdm(range(num_chunks), desc="  Processing Chunks (EDT+Trim)"):
        z_start = i * chunk_size_z
        z_end = min((i + 1) * chunk_size_z, original_shape[0])
        if z_start >= z_end: continue

        # --- Process one chunk ---
        try:
            # Load INVERSE boundary chunk (boolean)
            inv_boundary_chunk = ~hull_boundary_mask[z_start:z_end]
            edt_chunk = None # Initialize

            # Calculate EDT for the chunk (relatively small)
            if not np.any(inv_boundary_chunk):
                 edt_chunk = np.zeros(inv_boundary_chunk.shape, dtype=np.float32)
            elif not np.all(inv_boundary_chunk):
                 # This is the memory-intensive part *for the chunk*
                 edt_chunk = distance_transform_edt(inv_boundary_chunk, sampling=spacing).astype(np.float32) # Ensure float32
            else: # All points are inside boundary in this chunk
                 edt_chunk = np.full(inv_boundary_chunk.shape, np.finfo(np.float32).max, dtype=np.float32)

            # Identify voxels below threshold in this chunk
            voxels_to_trim_chunk = (edt_chunk < distance_threshold)
            del edt_chunk # Free EDT chunk memory ASAP
            gc.collect()

            # Apply to the corresponding segmentation memmap chunk VIEW
            # Ensure memmap is still valid
            if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None:
                 raise IOError(f"Segmentation memmap became invalid before reading chunk {i} in chunked mode")

            seg_chunk_view = segmentation_memmap[z_start:z_end] # Get view for reading and writing

            # Ensure shapes match before boolean indexing
            if voxels_to_trim_chunk.shape != seg_chunk_view.shape:
                 raise ValueError(f"Chunk shape mismatch at z={z_start}:{z_end}. Trim chunk: {voxels_to_trim_chunk.shape}, Seg chunk: {seg_chunk_view.shape}")

            # Read seg chunk data into memory *after* EDT chunk is deleted
            seg_chunk_data = np.array(seg_chunk_view) # Read data for boolean op

            trim_target_mask_chunk = voxels_to_trim_chunk & (seg_chunk_data > 0)
            del seg_chunk_data # Free memory copy
            num_trimmed_chunk = np.sum(trim_target_mask_chunk)

            if num_trimmed_chunk > 0:
                 # Update the global tracking mask (in memory)
                 trimmed_voxels_mask[z_start:z_end][trim_target_mask_chunk] = True
                 # Modify the segmentation VIEW in-place on the memmap
                 # Using the view directly for writing should be fine
                 seg_chunk_view[trim_target_mask_chunk] = 0
                 segmentation_memmap.flush() # Flush after modification
                 total_trimmed_in_chunks += num_trimmed_chunk

            # --- Clean up chunk variables ---
            del inv_boundary_chunk, voxels_to_trim_chunk
            del seg_chunk_view, trim_target_mask_chunk
            gc.collect() # Collect garbage after each chunk

        except MemoryError:
            print(f"\n!!! MemoryError occurred during chunked processing (Chunk {i}, z={z_start}-{z_end}).")
            print(f"    Chunk shape: {tuple(s for s in hull_boundary_mask.shape[1:])}")
            # Estimate RAM needed for just the EDT chunk
            chunk_shape = (z_end - z_start,) + hull_boundary_mask.shape[1:]
            edt_chunk_mem_mb = (np.prod(chunk_shape) * np.dtype(np.float32).itemsize) / (1024**2)
            print(f"    Approx. RAM for EDT chunk calculation: {edt_chunk_mem_mb:.1f} MB")
            print(f"    Consider using a smaller chunk_size_z (current: {chunk_size_z}).")
            raise MemoryError(f"Chunked processing failed at chunk {i} due to MemoryError.") from None
        except Exception as chunk_err:
             print(f"\n!!! Error processing chunk {i} (z={z_start}-{z_end}): {chunk_err}")
             import traceback; traceback.print_exc()
             raise RuntimeError(f"Chunked processing failed at chunk {i}.") from chunk_err


    print(f"  Chunked EDT+trimming applied. Total voxels trimmed: {total_trimmed_in_chunks} in {time.time() - start_chunked_time:.2f}s.")

    # --- Optional: Remove Small Remnants (Operating on Memmap) ---
    # This logic remains the same as it happens after the EDT step
    if min_remaining_size > 0:
        print(f"  Removing remnants smaller than {min_remaining_size} voxels (on Memmap)...")
        # ... (Rest of the remnant removal logic - unchanged) ...
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
            else: print("  Segmentation memmap appears empty after trimming, skipping remnant removal.")
        finally:
             if 'bool_seg_memmap' in locals() and bool_seg_memmap is not None and hasattr(bool_seg_memmap, '_mmap'): del bool_seg_memmap; gc.collect()
             if 'bool_seg_temp_dir' in locals() and bool_seg_temp_dir is not None and os.path.exists(bool_seg_temp_dir):
                 try: rmtree(bool_seg_temp_dir, ignore_errors=True)
                 except Exception as e_final_clean: print(f"    Warn: Error cleaning up remnant temp dir {bool_seg_temp_dir}: {e_final_clean}")
             gc.collect()


    print("--- Edge trimming on Memmap finished ---")
    return trimmed_voxels_mask

def segment_microglia_first_pass(
    volume, # Use original volume for tissue mask
    spacing,
    tubular_scales=[0.5, 1.0, 2.0, 3.0],
    smooth_sigma=0.5,
    connect_max_gap_physical=1.0,
    min_size_voxels=50,
    low_threshold_percentile=25.0,
    high_threshold_percentile=95.0,
    hull_erosion_iterations=1,
    edge_trim_distance_threshold=2.0,
    edge_distance_chunk_size_z=32
    ):
    """
    First pass segmentation using slice-wise enhancement, percentile-based
    normalization/thresholding, and distance-based edge trimming. Uses memmaps
    extensively to reduce peak RAM usage.
    """
    # --- Initial Setup ---
    low_threshold_percentile = max(0.0, min(100.0, low_threshold_percentile))
    high_threshold_percentile = max(0.0, min(100.0, high_threshold_percentile))
    if low_threshold_percentile >= high_threshold_percentile:
         print(f"Warning: low_threshold_percentile ({low_threshold_percentile}) >= high_threshold_percentile ({high_threshold_percentile}). Adjusting low threshold.")
         low_threshold_percentile = max(0.0, high_threshold_percentile - 1.0)

    original_shape = volume.shape
    spacing = tuple(float(s) for s in spacing)
    first_pass_params = { # Store all params
        'spacing': spacing, 'tubular_scales': tubular_scales, 'smooth_sigma': smooth_sigma,
        'connect_max_gap_physical': connect_max_gap_physical, 'min_size_voxels': min_size_voxels,
        'low_threshold_percentile': low_threshold_percentile,
        'high_threshold_percentile': high_threshold_percentile,
        'hull_erosion_iterations': hull_erosion_iterations,
        'edge_trim_distance_threshold': edge_trim_distance_threshold,
        'edge_distance_chunk_size_z': edge_distance_chunk_size_z,
        'original_shape': original_shape }
    print(f"\n--- Starting First Pass Segmentation (Percentile Method, Memmap Optimized) ---")
    print(f"Params: low_perc={low_threshold_percentile:.1f}, high_perc={high_threshold_percentile:.1f}, "
          f"hull_erosion={hull_erosion_iterations}, trim_dist={edge_trim_distance_threshold:.2f}")
    initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Initial memory usage: {initial_mem:.2f} MB")

    # Keep track of memmaps and their temporary directories for cleanup
    memmap_registry = {} # {name: (memmap_object, path, temp_dir)}

    enhanced_memmap = None # Initialize

    try:
        # --- Step 1: Enhance Tubular Structures (Returns Memmap) ---
        print("\nStep 1: Enhancing structures...")
        enhanced_memmap, enhance_path, enhance_temp_dir = enhance_tubular_structures_slice_by_slice(
            volume,
            scales=tubular_scales,
            spacing=spacing,
            apply_3d_smoothing=(smooth_sigma > 0),
            smoothing_sigma_phys=smooth_sigma
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

        z_high_percentile_values = []
        print("  Calculating z-plane high percentile values...")
        target_intensity = 1.0
        print(f"  Target intensity for high percentile ({high_threshold_percentile:.1f}): {target_intensity}")

        chunk_size_norm_read = min(50, enhanced_memmap.shape[0]) if enhanced_memmap.shape[0] > 0 else 1
        for z_start in tqdm(range(0, enhanced_memmap.shape[0], chunk_size_norm_read), desc="  Calculating z-stats"):
            z_end = min(z_start + chunk_size_norm_read, enhanced_memmap.shape[0])
            # Read chunk from enhanced memmap
            enhanced_chunk = enhanced_memmap[z_start:z_end]
            for i, z in enumerate(range(z_start, z_end)):
                plane = enhanced_chunk[i] # Access plane within the chunk
                if plane is None or plane.size == 0 or not np.any(np.isfinite(plane)):
                    z_high_percentile_values.append((z, 0.0)); continue

                max_samples = 100000
                finite_plane = plane[np.isfinite(plane)].ravel()
                if finite_plane.size == 0:
                    z_high_percentile_values.append((z, 0.0)); continue

                samples = np.random.choice(finite_plane, min(max_samples, finite_plane.size), replace=False)

                if samples.size > 0:
                    try:
                        high_perc_value = np.percentile(samples, high_threshold_percentile)
                        if not np.isfinite(high_perc_value): high_perc_value = 0.0
                        z_high_percentile_values.append((z, float(high_perc_value)))
                    except Exception as e:
                        print(f"  Warn: Error calculating percentile for slice {z}: {e}. Using 0.")
                        z_high_percentile_values.append((z, 0.0))
                else:
                    z_high_percentile_values.append((z, 0.0))
            del enhanced_chunk # Release chunk memory
            gc.collect()

        print("  Applying normalization scaling...")
        z_stats_dict = dict(z_high_percentile_values)
        chunk_size_norm_write = min(50, enhanced_memmap.shape[0]) if enhanced_memmap.shape[0] > 0 else 1
        for z_start in tqdm(range(0, enhanced_memmap.shape[0], chunk_size_norm_write), desc="  Normalizing z-planes"):
             z_end = min(z_start + chunk_size_norm_write, enhanced_memmap.shape[0])
             enhanced_chunk = enhanced_memmap[z_start:z_end] # Read chunk again
             normalized_chunk = np.zeros_like(enhanced_chunk, dtype=np.float32) # Create chunk in memory

             for i, z in enumerate(range(z_start, z_end)):
                 current_slice = enhanced_chunk[i]
                 high_perc_value = z_stats_dict.get(z, 0.0)

                 if current_slice is None or not np.any(np.isfinite(current_slice)):
                     normalized_chunk[i] = 0.0; continue

                 if high_perc_value > 1e-6:
                     scale_factor = target_intensity / high_perc_value
                     normalized_chunk[i] = (current_slice * scale_factor).astype(np.float32)
                 else:
                     normalized_chunk[i] = current_slice.astype(np.float32) # Don't scale

             # Write normalized chunk to memmap
             normalized_memmap[z_start:z_end] = normalized_chunk
             del enhanced_chunk, normalized_chunk # Free chunk memory
             gc.collect()

        normalized_memmap.flush()
        print("Normalization step finished.")
        # Clean up enhanced memmap now that normalization is done
        name = 'enhanced'
        if name in memmap_registry:
            mm, p, d = memmap_registry[name]
            if hasattr(mm, '_mmap'): del mm; gc.collect()
            try:
                if os.path.exists(p): os.unlink(p)
                if os.path.exists(d): rmtree(d, ignore_errors=True)
            except Exception as e: print(f"Warn: Could not clean {name} temp files: {e}")
            del memmap_registry[name]
        gc.collect()

        # --- Step 3: Thresholding using Low Percentile (Reads normalized_memmap) ---
        print("\nStep 3: Thresholding using low percentile...")
        binary_temp_dir = tempfile.mkdtemp(prefix="binary_")
        binary_path = os.path.join(binary_temp_dir, 'b.dat')
        binary_memmap = np.memmap(binary_path, dtype=bool, mode='w+', shape=original_shape)
        memmap_registry['binary'] = (binary_memmap, binary_path, binary_temp_dir)

        # --- Calculate threshold (chunked sampling from normalized memmap) ---
        sample_size = min(5_000_000, normalized_memmap.size)
        print(f"  Sampling {sample_size} points from normalized memmap for threshold...")
        final_threshold = 0.0
        samples_collected = []
        num_samples_needed = sample_size
        chunk_size_thresh_sample = min(200, normalized_memmap.shape[0]) if normalized_memmap.shape[0] > 0 else 1
        indices_z = np.arange(normalized_memmap.shape[0])
        np.random.shuffle(indices_z) # Process z-slices in random order

        # Ensure normalized memmap is accessible
        if not hasattr(normalized_memmap, '_mmap') or normalized_memmap._mmap is None:
            raise RuntimeError("Normalized memmap closed unexpectedly before threshold sampling.")

        collected_count = 0
        for z_start_idx in tqdm(range(0, len(indices_z), chunk_size_thresh_sample), desc="  Sampling Norm. Vol."):
            if collected_count >= num_samples_needed: break
            z_indices_chunk = indices_z[z_start_idx : z_start_idx + chunk_size_thresh_sample]
            # Read slices specified by indices - might be slow if indices are sparse
            # A potentially faster way for memmap is contiguous reads, but let's try this first
            # Note: Reading multiple non-contiguous slices can be inefficient
            # Alternative: Read contiguous chunks and sample proportionally from them
            slices_data = normalized_memmap[z_indices_chunk, :, :] # Reads multiple slices
            finite_samples_in_chunk = slices_data[np.isfinite(slices_data)].ravel()
            del slices_data; gc.collect()

            samples_to_take = min(len(finite_samples_in_chunk), num_samples_needed - collected_count)
            if samples_to_take > 0:
                chosen_samples = np.random.choice(finite_samples_in_chunk, samples_to_take, replace=False)
                samples_collected.append(chosen_samples)
                collected_count += samples_to_take
            del finite_samples_in_chunk
            if collected_count >= num_samples_needed: break # Exit loop early if enough samples

        if not samples_collected:
            print("  Warn: No finite samples collected. Using 0.0 threshold.")
            final_threshold = 0.0
        else:
            all_samples = np.concatenate(samples_collected)
            del samples_collected; gc.collect()
            if all_samples.size > 0:
                 try:
                     final_threshold = np.percentile(all_samples, low_threshold_percentile)
                     print(f"  Calculated threshold at {low_threshold_percentile:.1f} percentile: {final_threshold:.4f}")
                 except Exception as e:
                     print(f"  Warn: Percentile calculation failed: {e}. Using median fallback.")
                     final_threshold = np.median(all_samples)
                     print(f"  Using median fallback threshold: {final_threshold:.4f}")
            else:
                 print("  Warn: Concatenated samples array is empty. Using 0.0 threshold.")
                 final_threshold = 0.0
            del all_samples; gc.collect()

        # Apply threshold in chunks
        chunk_size_thresh_apply = min(100, original_shape[0]) if original_shape[0] > 0 else 1
        print(f"  Applying threshold {final_threshold:.4f}...")
        for i in tqdm(range(0, original_shape[0], chunk_size_thresh_apply), desc="  Applying Threshold"):
             end_idx = min(i + chunk_size_thresh_apply, original_shape[0])
             if hasattr(normalized_memmap, '_mmap') and normalized_memmap._mmap is not None:
                  norm_chunk = normalized_memmap[i:end_idx]
                  binary_memmap[i:end_idx] = norm_chunk > final_threshold
                  del norm_chunk; gc.collect()
             else:
                  print(f"Error: 'normalized' memmap closed or invalid during thresholding chunk {i}")
                  binary_memmap[i:end_idx] = False # Fill failed chunk with False
        binary_memmap.flush()
        print("Thresholding done.")
        # Cleanup normalized memmap
        name = 'normalized'
        if name in memmap_registry:
            mm, p, d = memmap_registry[name]
            if hasattr(mm, '_mmap'): del mm; gc.collect()
            try:
                if os.path.exists(p): os.unlink(p)
                if os.path.exists(d): rmtree(d, ignore_errors=True)
            except Exception as e: print(f"Warn: Could not clean {name} temp files: {e}")
            del memmap_registry[name]
        gc.collect()

        # --- Step 4: Connect Fragmented Processes (Reads binary_memmap) ---
        print("\nStep 4: Connecting fragments...")
        connected_binary_memmap, connected_temp_dir = connect_fragmented_processes(
            binary_memmap, # Use the thresholded binary memmap
            spacing=spacing,
            max_gap_physical=connect_max_gap_physical
        )
        connected_path = connected_binary_memmap.filename
        memmap_registry['connected'] = (connected_binary_memmap, connected_path, connected_temp_dir)
        print("Connection memmap created.")
        # Cleanup binary memmap
        name = 'binary'
        if name in memmap_registry:
            mm, p, d = memmap_registry[name]
            if hasattr(mm, '_mmap'): del mm; gc.collect()
            try:
                if os.path.exists(p): os.unlink(p)
                if os.path.exists(d): rmtree(d, ignore_errors=True)
            except Exception as e: print(f"Warn: Could not clean {name} temp files: {e}")
            del memmap_registry[name]
        gc.collect()

        # --- Step 5: Remove Small Objects (Reads connected_memmap) ---
        print(f"\nStep 5: Cleaning objects smaller than {min_size_voxels} voxels...")
        cleaned_binary_dir=tempfile.mkdtemp(prefix="cleaned_")
        cleaned_binary_path=os.path.join(cleaned_binary_dir, 'c.dat')
        cleaned_binary_memmap=np.memmap(cleaned_binary_path, dtype=bool, mode='w+',
                                      shape=original_shape)
        memmap_registry['cleaned'] = (cleaned_binary_memmap, cleaned_binary_path, cleaned_binary_dir)
        # remove_small_objects needs array input, try loading connected if possible, else chunk
        print("  Loading connected mask for small object removal...")
        connected_memmap_obj = memmap_registry['connected'][0]
        try:
            connected_in_memory = np.array(connected_memmap_obj)
            print("  Applying remove_small_objects...")
            remove_small_objects(connected_in_memory,
                                 min_size=min_size_voxels,
                                 connectivity=1,
                                 out=cleaned_binary_memmap) # Output to memmap
            del connected_in_memory; gc.collect()
        except MemoryError:
             print("  MemoryError loading connected mask. Cannot remove small objects effectively. Skipping.")
             # Copy connected to cleaned as a fallback
             print("  Copying connected mask to cleaned mask...")
             chunk_size_copy = min(100, original_shape[0]) if original_shape[0] > 0 else 1
             for i in tqdm(range(0, original_shape[0], chunk_size_copy), desc="  Copying Connected"):
                 cleaned_binary_memmap[i:min(i+chunk_size_copy, original_shape[0])] = connected_memmap_obj[i:min(i+chunk_size_copy, original_shape[0])]
        except Exception as e:
            print(f"  Error during small object removal: {e}. Skipping.")
            # Copy connected to cleaned as a fallback
            print("  Copying connected mask to cleaned mask...")
            chunk_size_copy = min(100, original_shape[0]) if original_shape[0] > 0 else 1
            for i in tqdm(range(0, original_shape[0], chunk_size_copy), desc="  Copying Connected"):
                 cleaned_binary_memmap[i:min(i+chunk_size_copy, original_shape[0])] = connected_memmap_obj[i:min(i+chunk_size_copy, original_shape[0])]

        cleaned_binary_memmap.flush()
        print("Cleaning step done.")
        # Cleanup connected memmap
        name = 'connected'
        if name in memmap_registry:
            mm, p, d = memmap_registry[name]
            if hasattr(mm, '_mmap'): del mm; gc.collect()
            try:
                if os.path.exists(p): os.unlink(p)
                if os.path.exists(d): rmtree(d, ignore_errors=True)
            except Exception as e: print(f"Warn: Could not clean {name} temp files: {e}")
            del memmap_registry[name]
        gc.collect()

        # --- Step 6: Label Connected Components (Reads cleaned_memmap) ---
        # --- Step 6: Label Connected Components (Reads cleaned_memmap) ---
        print("\nStep 6: Labeling components...")
        labels_temp_dir = None # Initialize to None
        labels_path = None
        first_pass_segmentation_memmap = None # Initialize

        try:
            labels_temp_dir = tempfile.mkdtemp(prefix="labels_")
            labels_path = os.path.join(labels_temp_dir, 'l.dat')

            # *** FIX: Change mode from 'r+' to 'w+' to CREATE the file ***
            print(f"  Creating labels memmap file: {labels_path}") # Add log
            first_pass_segmentation_memmap = np.memmap(labels_path, dtype=np.int32,
                                                     mode='w+', # <-- CORRECTED MODE
                                                     shape=original_shape)
            # Add to registry ONLY after successful memmap creation
            memmap_registry['labels'] = (first_pass_segmentation_memmap, labels_path, labels_temp_dir)
            print("  Labels memmap object created.")

            # --- Rest of Step 6 ---
            print("  Applying ndimage.label...")
            # Check input memmap validity before labeling
            cleaned_memmap_obj = memmap_registry.get('cleaned', (None,))[0]
            if cleaned_memmap_obj is None or not hasattr(cleaned_memmap_obj, '_mmap') or cleaned_memmap_obj._mmap is None:
                 raise RuntimeError("Input 'cleaned' memmap for labeling is invalid or missing from registry.")

            try:
                num_features = ndimage.label(
                    cleaned_memmap_obj,
                    structure=generate_binary_structure(3, 1),
                    output=first_pass_segmentation_memmap
                )
                first_pass_segmentation_memmap.flush()
                print(f"Labeling done ({num_features} features found).")

            except Exception as label_error:
                 print(f"\n!!! ERROR during ndimage.label or flush: {label_error}")
                 import traceback
                 traceback.print_exc()
                 # Immediate cleanup attempt for 'labels'
                 name = 'labels'
                 if name in memmap_registry:
                     mm, p, d = memmap_registry[name]
                     if hasattr(mm, '_mmap'): del mm; gc.collect()
                     try:
                         if p and os.path.exists(p): os.unlink(p)
                         if d and os.path.exists(d): rmtree(d, ignore_errors=True)
                     except Exception as e_clean: print(f"Warn: Error cleaning up failed {name} resources: {e_clean}")
                     del memmap_registry[name]
                 raise label_error from label_error

            # Cleanup cleaned binary memmap (ONLY if labeling succeeded)
            name = 'cleaned'
            # ... (cleanup logic for 'cleaned' remains the same) ...
            if name in memmap_registry:
                print(f"  Cleaning up {name} after successful labeling...") # Add log
                mm, p, d = memmap_registry[name]
                if hasattr(mm, '_mmap'): del mm; gc.collect()
                try:
                    if p and os.path.exists(p): os.unlink(p)
                    if d and os.path.exists(d): rmtree(d, ignore_errors=True)
                except Exception as e: print(f"Warn: Could not clean {name} temp files during step 6 cleanup: {e}")
                del memmap_registry[name]
            gc.collect()


        except Exception as step6_setup_error:
             # Catch errors during setup of step 6 (e.g., mkdtemp, memmap creation itself)
             print(f"\n!!! ERROR during Step 6 setup (before labeling): {step6_setup_error}")
             import traceback
             traceback.print_exc()
             # Attempt cleanup of potentially created labels dir/file
             # ... (cleanup logic in this except block remains the same) ...
             name = 'labels'
             if name in memmap_registry: # If it got added to registry
                  mm, p, d = memmap_registry[name]
                  if hasattr(mm, '_mmap'): del mm; gc.collect()
                  del memmap_registry[name] # Remove from registry first
             else: # If memmap object creation failed, use paths directly
                  p, d = labels_path, labels_temp_dir
             try:
                 if p and os.path.exists(p): os.unlink(p)
                 if d and os.path.exists(d): rmtree(d, ignore_errors=True)
             except Exception as e_clean: print(f"Warn: Error cleaning up failed {name} resources during setup error: {e_clean}")
             raise step6_setup_error from step6_setup_error

        try:
            # --- 7a. Generate Hull Boundary ---
            hull_boundary_mask, hull_stack = generate_hull_boundary_and_stack(
                volume, hull_erosion_iterations=hull_erosion_iterations
            )

            if hull_boundary_mask is None or not np.any(hull_boundary_mask):
                print("  Hull boundary mask empty/not generated. Skipping edge trimming.")
                hull_boundary_mask = np.zeros(original_shape, dtype=bool)
            else:
                # --- 7b. Call Trimming Helper Function (operates on memmap) ---
                # Pass the memmap opened in 'r+' mode
                _ = trim_object_edges_by_distance( # Discard returned trim mask for now
                    segmentation_memmap=first_pass_segmentation_memmap, # Pass the memmap
                    hull_boundary_mask=hull_boundary_mask,
                    spacing=spacing,
                    distance_threshold=edge_trim_distance_threshold,
                    min_remaining_size=min_size_voxels,
                    chunk_size_z=edge_distance_chunk_size_z
                )
                # Ensure changes are flushed after trimming function returns
                first_pass_segmentation_memmap.flush()

        except Exception as e:
             print(f"\nERROR during edge artifact trimming: {e}")
             import traceback; traceback.print_exc()
             hull_boundary_mask = np.zeros(original_shape, dtype=bool) # Ensure valid default
             gc.collect()
        finally:
             if hull_stack is not None: del hull_stack; gc.collect()
             # Do NOT clean up labels memmap yet

        # --- Step 8: Load FINAL Labeled Segmentation into Memory ---
        print("\nStep 8: Loading final trimmed labels into memory...")
        final_segmentation = np.array(first_pass_segmentation_memmap)
        print("Final labels loaded into memory.")

        # --- Finalization ---
        print(f"\n--- First Pass Segmentation Finished ---")
        final_unique_labels = np.unique(final_segmentation)
        final_object_count = len(final_unique_labels[final_unique_labels != 0])
        print(f"Final labeled mask shape: {final_segmentation.shape}")
        print(f"Number of labeled objects remaining: {final_object_count}")
        final_mem = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"Final memory usage: {final_mem:.2f} MB")

        # Return the final in-memory segmentation, parameters, and boundary mask
        return final_segmentation, first_pass_params, hull_boundary_mask

    finally:
        # --- FINAL Cleanup ALL remaining memmaps ---
        # The final cleanup logic with os.path.exists checks remains the same
        # It will handle any memmaps still left in the registry
        print("\nCleaning up any remaining temporary memmap files...")
        registry_keys = list(memmap_registry.keys())
        for name in registry_keys:
            if name in memmap_registry: # Check if still present
                mm, p, d = memmap_registry[name]
                print(f"  Cleaning up {name} (Path: {p})...")
                # 1. Delete memmap object
                if hasattr(mm, '_mmap') and mm._mmap is not None:
                    try: del mm; gc.collect()
                    except Exception as e_del: print(f"    Warn: Error deleting memmap object for {name}: {e_del}")
                elif hasattr(mm, 'close'):
                     try: mm.close()
                     except: pass
                # 2. Delete file
                try:
                    if p and os.path.exists(p): os.unlink(p); print(f"    Deleted file: {p}")
                except Exception as e_unlink: print(f"    Warn: Error unlinking file {p}: {e_unlink}")
                # 3. Delete directory
                try:
                    if d and os.path.exists(d): rmtree(d, ignore_errors=True); print(f"    Removed directory: {d}")
                except Exception as e_rmtree: print(f"    Warn: Error removing directory {d}: {e_rmtree}")
                # 4. Remove from registry
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