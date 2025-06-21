import numpy as np
from scipy import ndimage
from scipy.ndimage import (distance_transform_edt,
                           binary_erosion, generate_binary_structure)
from scipy.ndimage import (distance_transform_edt, label as ndimage_label, find_objects,
                           binary_opening, binary_closing, binary_dilation)
from skimage.filters import threshold_otsu # type: ignore
from skimage.morphology import convex_hull_image, remove_small_objects # type: ignore
import traceback
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
from collections import defaultdict
import math
seed = 42
np.random.seed(seed)         # For NumPy

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
                chunk_size_z=edge_distance_chunk_size_z)
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


def find_set(i):
    """Finds the representative (root) of the set containing i, with path compression."""
    global union_find_parent
    root = i
    # Find the root
    while union_find_parent.get(root, root) != root:
        root = union_find_parent.get(root, root)
    # Path compression
    curr = i
    while union_find_parent.get(curr, curr) != root:
        next_node = union_find_parent.get(curr, curr)
        union_find_parent[curr] = root # Point directly to root
        curr = next_node
    # Ensure the node itself is in the parent dict if it wasn't processed before
    if i not in union_find_parent:
        union_find_parent[i] = i
    return union_find_parent.get(i, i) # Return the direct parent after compression (which is the root)

def unite_sets(i, j):
    """Merges the sets containing i and j, merging into the smaller root ID."""
    global union_find_parent
    root_i = find_set(i)
    root_j = find_set(j)
    if root_i != root_j:
        # Merge the set with the larger root ID into the one with the smaller root ID
        if root_i < root_j:
            union_find_parent[root_j] = root_i
        else:
            union_find_parent[root_i] = root_j
# --- End Union-Find ---


# --- trim_object_edges_by_distance is MODIFIED ---
def trim_object_edges_by_distance(
    segmentation_memmap, original_volume, hull_boundary_mask, spacing,
    distance_threshold, global_brightness_cutoff, min_remaining_size=10, chunk_size_z=32
    ):
    """
    Applies operations in the order: Trim -> Remove Small -> Relabel -> Heal -> Merge -> Final Relabel -> Final Cleanup.
    """
    ## MODIFICATION: Updated docstring and title to reflect new processing order.
    print("\n--- Processing Order: Trim -> Remove Small -> Relabel -> Heal -> Merge -> Final Relabel -> Final Cleanup ---")
    print("--- (Assuming scikit-image is installed for required steps) ---")

    original_shape = segmentation_memmap.shape
    trimmed_voxels_mask = np.zeros(original_shape, dtype=bool)

    if not np.any(hull_boundary_mask): print("  Hull boundary mask is empty. No processing performed."); return trimmed_voxels_mask

    writable_modes = ['r+', 'w+']
    if segmentation_memmap.mode not in writable_modes:
         raise ValueError(f"segmentation_memmap must be opened in a writable mode ({writable_modes}) for in-place modification. Found mode: '{segmentation_memmap.mode}'")

    print("  Finding unique labels before processing...")
    original_labels_present = np.unique(segmentation_memmap); original_labels_present = original_labels_present[original_labels_present != 0]
    print(f"  Found {len(original_labels_present)} unique labels initially.")
    if len(original_labels_present) == 0: print("  No objects found in segmentation. Skipping all steps."); return trimmed_voxels_mask

    # --- Step 0: Create Copy of Original Segmentation ---
    original_segmentation_memmap = None; original_seg_temp_dir = None; copy_success = False
    print("  Creating copy of original segmentation for healing reference...")
    start_copy_time = time.time()
    try:
        original_seg_temp_dir = tempfile.mkdtemp(prefix="orig_seg_copy_")
        original_seg_path = os.path.join(original_seg_temp_dir, 'original_seg.dat')
        original_segmentation_memmap = np.memmap(original_seg_path, dtype=segmentation_memmap.dtype, mode='w+', shape=original_shape)
        chunk_size_copy = min(100, original_shape[0]) if original_shape[0] > 0 else 1
        for i in tqdm(range(0, original_shape[0], chunk_size_copy), desc="    Copying Seg"):
            z_start = i; z_end = min(i + chunk_size_copy, original_shape[0])
            if z_start >= z_end: continue
            if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None: raise IOError("Source segmentation memmap became invalid during copy")
            original_segmentation_memmap[z_start:z_end] = segmentation_memmap[z_start:z_end]
        original_segmentation_memmap.flush(); copy_success = True
        print(f"  Original segmentation copied in {time.time() - start_copy_time:.2f}s."); gc.collect()
    except Exception as e_copy:
        print(f"\n!!! ERROR creating copy of original segmentation: {e_copy}. Cannot perform convex hull healing.")
        if original_segmentation_memmap is not None and hasattr(original_segmentation_memmap, '_mmap'): del original_segmentation_memmap; gc.collect(); original_segmentation_memmap = None
        if original_seg_temp_dir and os.path.exists(original_seg_temp_dir): rmtree(original_seg_temp_dir, ignore_errors=True); original_seg_temp_dir = None
        print("    Healing step will be skipped due to copy failure.")

    # --- Step 1: Trimming ---
    print(f"\n--- 1. Trimming object edges (Dist < {distance_threshold:.2f} AND Brightness < {global_brightness_cutoff:.4f}) ---")
    gc.collect(); num_chunks = math.ceil(original_shape[0] / chunk_size_z); total_trimmed_in_chunks = 0
    start_trim_time = time.time(); modified_labels_during_trim = set()
    print(f"  Processing {num_chunks} chunks (chunk_size_z={chunk_size_z}).")
    for i in tqdm(range(num_chunks), desc="  Processing Chunks (Trim)"):
        z_start = i * chunk_size_z; z_end = min((i + 1) * chunk_size_z, original_shape[0])
        if z_start >= z_end: continue
        try:
            inv_boundary_chunk = ~hull_boundary_mask[z_start:z_end]
            if not np.any(inv_boundary_chunk): edt_chunk = np.zeros(inv_boundary_chunk.shape, dtype=np.float32)
            elif not np.all(inv_boundary_chunk): edt_chunk = distance_transform_edt(inv_boundary_chunk, sampling=spacing).astype(np.float32)
            else: edt_chunk = np.full(inv_boundary_chunk.shape, np.finfo(np.float32).max, dtype=np.float32)
            close_to_boundary_chunk = (edt_chunk < distance_threshold); del inv_boundary_chunk, edt_chunk; gc.collect()
            if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None: raise IOError(f"Segmentation memmap invalid reading chunk {i}")
            seg_chunk_data = np.array(segmentation_memmap[z_start:z_end]); intensity_chunk = original_volume[z_start:z_end]
            dim_voxels_chunk = (intensity_chunk < global_brightness_cutoff); del intensity_chunk; gc.collect()
            trim_target_mask_chunk = (seg_chunk_data > 0) & close_to_boundary_chunk & dim_voxels_chunk
            num_trimmed_chunk = np.sum(trim_target_mask_chunk)
            if num_trimmed_chunk > 0:
                 unique_trimmed_in_chunk = np.unique(seg_chunk_data[trim_target_mask_chunk])
                 modified_labels_during_trim.update(unique_trimmed_in_chunk[unique_trimmed_in_chunk != 0])
                 seg_chunk_view_write = segmentation_memmap[z_start:z_end]
                 trimmed_voxels_mask[z_start:z_end][trim_target_mask_chunk] = True
                 seg_chunk_view_write[trim_target_mask_chunk] = 0
                 segmentation_memmap.flush(); total_trimmed_in_chunks += num_trimmed_chunk; del seg_chunk_view_write
            del seg_chunk_data, close_to_boundary_chunk, dim_voxels_chunk, trim_target_mask_chunk; gc.collect()
        except MemoryError as mem_err: raise MemoryError(f"Trimming failed chunk {i}: MemoryError.") from mem_err
        except Exception as chunk_err: raise RuntimeError(f"Trimming failed chunk {i}.") from chunk_err
    print(f"  Trimming finished. Voxels trimmed: {total_trimmed_in_chunks}. Labels affected: {len(modified_labels_during_trim)}.")
    print(f"  Time taken: {time.time() - start_trim_time:.2f}s."); gc.collect()

    # --- Step 2: Remove Small Remnants ---
    if min_remaining_size > 0:
        print(f"\n--- 2. Removing remnants smaller than {min_remaining_size} voxels ---")
        num_remnants_removed = 0; start_rem_time = time.time(); bool_seg_temp_dir = None; bool_seg_memmap = None
        try:
            bool_seg_temp_dir = tempfile.mkdtemp(prefix="bool_remnant_")
            bool_seg_path = os.path.join(bool_seg_temp_dir, 'temp_bool_seg.dat')
            bool_seg_memmap = np.memmap(bool_seg_path, dtype=bool, mode='w+', shape=original_shape)
            chunk_size_rem = min(100, original_shape[0]) if original_shape[0] > 0 else 1
            any_objects_found = False
            for i in tqdm(range(0, original_shape[0], chunk_size_rem), desc="    Bool Mask Gen"):
                z_start = i; z_end = min(i + chunk_size_rem, original_shape[0])
                if z_start >= z_end: continue
                bool_chunk = segmentation_memmap[z_start:z_end] > 0
                bool_seg_memmap[z_start:z_end] = bool_chunk
                if not any_objects_found and np.any(bool_chunk): any_objects_found = True
            bool_seg_memmap.flush(); gc.collect()
            if any_objects_found:
                try:
                    bool_seg_in_memory = np.array(bool_seg_memmap); del bool_seg_memmap; bool_seg_memmap = None; gc.collect(); os.unlink(bool_seg_path); rmtree(bool_seg_temp_dir); bool_seg_temp_dir = None
                    cleaned_bool_seg = remove_small_objects(bool_seg_in_memory, min_size=min_remaining_size, connectivity=1)
                    small_remnant_mask_overall = bool_seg_in_memory & (~cleaned_bool_seg); del bool_seg_in_memory, cleaned_bool_seg; gc.collect()
                    chunk_size_apply_rem = min(100, original_shape[0]) if original_shape[0] > 0 else 1
                    for i in tqdm(range(0, original_shape[0], chunk_size_apply_rem), desc="    Applying Removal"):
                        z_start = i; z_end = min(i + chunk_size_apply_rem, original_shape[0]);
                        if z_start >= z_end: continue
                        seg_chunk_view = segmentation_memmap[z_start:z_end]; small_remnant_mask_chunk = small_remnant_mask_overall[z_start:z_end]
                        chunk_remnants_count = np.sum(seg_chunk_view[small_remnant_mask_chunk] > 0)
                        if chunk_remnants_count > 0: seg_chunk_view[small_remnant_mask_chunk] = 0; segmentation_memmap.flush(); num_remnants_removed += chunk_remnants_count
                    del small_remnant_mask_overall; gc.collect()
                    print(f"  Small object removal finished. Removed {num_remnants_removed} voxels.")
                except MemoryError: print("    MemoryError during small object removal. Skipping remaining removal steps."); gc.collect()
                except Exception as e_rem: print(f"    Error during small remnant removal processing: {e_rem}. Skipping."); gc.collect()
            else: print("  Segmentation memmap appears empty after trimming. Skipping remnant removal.")
        finally:
             if bool_seg_memmap is not None and hasattr(bool_seg_memmap, '_mmap'): del bool_seg_memmap; gc.collect()
             if bool_seg_temp_dir is not None and os.path.exists(bool_seg_temp_dir): rmtree(bool_seg_temp_dir, ignore_errors=True)
             gc.collect()
        print(f"  Time taken: {time.time() - start_rem_time:.2f}s.")
    else: print(f"\n--- 2. Skipping small object removal (min_remaining_size={min_remaining_size}) ---")

    # --- Step 3: Initial Relabeling of Disconnected Components ---
    print("\n--- 3. Initial relabeling of disconnected components ---")
    start_relabel_time = time.time(); relabel_map = {}; next_available_label = 0; relabel_count = 0
    try:
        max_label = 0; chunk_size_max = min(500, original_shape[0]) if original_shape[0] > 0 else 1
        for i in tqdm(range(0, original_shape[0], chunk_size_max), desc="    Finding Max Label", disable=original_shape[0]<=chunk_size_max):
             seg_chunk = segmentation_memmap[i:min(i+chunk_size_max, original_shape[0])]
             try: chunk_max = np.max(seg_chunk) if seg_chunk.size > 0 else 0
             except ValueError: chunk_max = 0
             if chunk_max > max_label: max_label = int(chunk_max)
        next_available_label = max_label + 1
        print(f"    Max label found: {max_label}. Next new label: {next_available_label}")
        current_labels_present = np.unique(segmentation_memmap); current_labels_present = current_labels_present[current_labels_present != 0]
        print(f"    Found {len(current_labels_present)} unique labels to check.")
        for lbl in current_labels_present: relabel_map[lbl] = lbl
        structure_relabel = generate_binary_structure(3, 1)
        all_locations = find_objects(segmentation_memmap); print(f"    Found {len(all_locations)} bounding boxes (some may be None).")
        for label_val in tqdm(current_labels_present, desc="  Initial Relabeling"):
            if label_val <= 0 or label_val > len(all_locations) or all_locations[label_val - 1] is None: continue
            loc = all_locations[label_val - 1]
            try:
                seg_sub_data = np.array(segmentation_memmap[loc]); label_mask_sub = (seg_sub_data == label_val)
                if not np.any(label_mask_sub): del seg_sub_data, label_mask_sub; continue
                labeled_components_sub, num_components = ndimage_label(label_mask_sub, structure=structure_relabel)
                if num_components > 1:
                    component_sizes = np.bincount(labeled_components_sub.ravel()); component_sizes[0] = 0
                    largest_component_idx = np.argmax(component_sizes); original_root_label = relabel_map[label_val]
                    modified_in_sub = False
                    for c_idx in range(1, num_components + 1):
                        if c_idx == largest_component_idx: continue
                        component_mask_sub = (labeled_components_sub == c_idx); new_label = next_available_label
                        seg_sub_data[component_mask_sub] = new_label; modified_in_sub = True
                        relabel_map[new_label] = original_root_label; next_available_label += 1; relabel_count += 1
                    if modified_in_sub: segmentation_memmap[loc] = seg_sub_data; segmentation_memmap.flush()
                del seg_sub_data, label_mask_sub, labeled_components_sub
            except MemoryError: print(f"\n!!! MemoryError during initial relabeling of label {label_val}. Skipping this label."); gc.collect(); continue
            except Exception as rel_err: print(f"\n!!! Error during initial relabeling of label {label_val}: {rel_err}. Skipping this label."); gc.collect(); continue
        print(f"  Initial relabeling finished. Created {relabel_count} new labels for fragmented objects.")
    except MemoryError: print("MemoryError during initial relabeling setup. Skipping relabeling."); gc.collect()
    except Exception as e: print(f"Error during initial relabeling setup: {e}. Skipping relabeling."); gc.collect()
    print(f"  Time taken: {time.time() - start_relabel_time:.2f}s."); gc.collect()

    # --- Step 4: Heal Porous Edges using Convex Hull ---
    if copy_success and original_segmentation_memmap is not None:
        print(f"\n--- 4. Applying Convex Hull Healing ---")
        print(f"  (Healing based on {len(modified_labels_during_trim)} labels initially modified by trimming)")
        voxels_restored_count = 0; start_heal_time = time.time(); original_to_current_map = defaultdict(list)
        all_current_mapped_labels = list(relabel_map.keys())
        for current_label in tqdm(all_current_mapped_labels, desc="    Building Inverse Map"):
             original_root_label = relabel_map.get(current_label, 0)
             if original_root_label != 0: original_to_current_map[original_root_label].append(current_label)
        print(f"    Inverse map built for {len(original_to_current_map)} original roots.")
        for original_label in tqdm(list(modified_labels_during_trim), desc="  Healing Labels"):
            try:
                current_labels_for_this_object = original_to_current_map.get(original_label)
                if not current_labels_for_this_object: continue
                combined_mask_for_findobj = None
                try: combined_mask_for_findobj = np.isin(segmentation_memmap, current_labels_for_this_object)
                except MemoryError: print(f"\n!!! MemoryError creating boolean mask for find_objects (label {original_label}). Skipping."); gc.collect(); continue
                locations = find_objects(combined_mask_for_findobj); del combined_mask_for_findobj; gc.collect()
                if not locations: continue
                min_coords = [min(s[d].start for s in locations) for d in range(3)]; max_coords = [max(s[d].stop for s in locations) for d in range(3)]
                global_loc = tuple(slice(min_coords[d], max_coords[d]) for d in range(3)); del locations; gc.collect()
                seg_sub_current_view = segmentation_memmap[global_loc]; seg_sub_current_data = np.array(seg_sub_current_view); orig_sub_data = original_segmentation_memmap[global_loc]
                object_mask_current_sub = np.isin(seg_sub_current_data, current_labels_for_this_object)
                if np.sum(object_mask_current_sub) < 4: del seg_sub_current_view, seg_sub_current_data, orig_sub_data, object_mask_current_sub; gc.collect(); continue
                hull_mask_sub = convex_hull_image(object_mask_current_sub)
                restore_mask_sub = hull_mask_sub & (orig_sub_data == original_label)
                currently_zero_mask_sub = (seg_sub_current_data == 0); actual_restore_mask_sub = restore_mask_sub & currently_zero_mask_sub
                num_restored_for_label = np.sum(actual_restore_mask_sub); voxels_restored_count += num_restored_for_label
                if num_restored_for_label > 0: seg_sub_current_view[actual_restore_mask_sub] = original_label; segmentation_memmap.flush()
                del global_loc, seg_sub_current_view, seg_sub_current_data, orig_sub_data, object_mask_current_sub, hull_mask_sub, restore_mask_sub, currently_zero_mask_sub, actual_restore_mask_sub; gc.collect()
            except Exception as e_heal_label: print(f"\n!!! ERROR healing {original_label}: {e_heal_label}. Skip."); gc.collect(); continue
        print(f"  Convex hull healing finished. Restored approx. {voxels_restored_count} voxels."); print(f"  Time taken: {time.time() - start_heal_time:.2f}s.")
    else: print("\n--- 4. Skipping convex hull healing step (original segmentation copy failed or unavailable) ---")

    # --- Step 5: Merge Touching Labels ---
    print("\n--- 5. Merging touching labels ---")
    start_merge_time = time.time(); global union_find_parent; union_find_parent = {}; merged_labels_count = 0
    try:
        labels_after_healing = np.unique(segmentation_memmap); labels_after_healing = labels_after_healing[labels_after_healing != 0]
        print(f"    Found {len(labels_after_healing)} unique labels to check for merging.")
        if len(labels_after_healing) > 1:
            for lbl in labels_after_healing: union_find_parent[lbl] = lbl
            neighbor_structure = generate_binary_structure(3, 1)
            print("    Identifying touching label pairs...")
            for label_val in tqdm(labels_after_healing, desc="  Checking Neighbors"):
                try:
                    locations = find_objects(segmentation_memmap == label_val)
                    if not locations: continue
                    for loc in locations:
                        seg_sub_view = segmentation_memmap[loc]; is_current_label_sub = (seg_sub_view == label_val)
                        dilated_mask_sub = binary_dilation(is_current_label_sub, structure=neighbor_structure); neighbor_mask_sub = dilated_mask_sub & ~is_current_label_sub
                        if np.any(neighbor_mask_sub):
                            neighbor_labels = seg_sub_view[neighbor_mask_sub]; unique_neighbor_labels = np.unique(neighbor_labels[neighbor_labels != 0])
                            for neighbor_label in unique_neighbor_labels:
                                if neighbor_label in union_find_parent:
                                    ## MODIFICATION: Reverted to the original "dumb" merge logic as requested.
                                    unite_sets(label_val, neighbor_label)
                        del seg_sub_view, is_current_label_sub, dilated_mask_sub, neighbor_mask_sub
                        if 'neighbor_labels' in locals(): del neighbor_labels
                        if 'unique_neighbor_labels' in locals(): del unique_neighbor_labels
                        gc.collect()
                except Exception as err_touch: print(f"\n!!! Error checking neighbors for label {label_val}: {err_touch}. Skipping."); gc.collect(); continue
            
            merge_map_final = {lbl: find_set(lbl) for lbl in labels_after_healing}
            num_labels_to_change = sum(1 for old, new in merge_map_final.items() if old != new); merged_labels_count = num_labels_to_change
            print(f"    Merge analysis complete. {merged_labels_count} labels will be merged into others.")
            if num_labels_to_change > 0:
                print("    Applying merge map to segmentation (chunked)...")
                chunk_size_merge = min(100, original_shape[0]) if original_shape[0] > 0 else 1
                for i in tqdm(range(0, original_shape[0], chunk_size_merge), desc="    Applying Merge"):
                    z_start = i; z_end = min(i + chunk_size_merge, original_shape[0])
                    if z_start >= z_end: continue
                    seg_chunk = np.array(segmentation_memmap[z_start:z_end]); seg_chunk_modified = np.copy(seg_chunk)
                    unique_labels_in_chunk = np.unique(seg_chunk)
                    labels_changed_in_chunk = False
                    for old_label in unique_labels_in_chunk:
                        if old_label == 0: continue
                        final_label = merge_map_final.get(old_label, old_label)
                        if final_label != old_label: seg_chunk_modified[seg_chunk == old_label] = final_label; labels_changed_in_chunk = True
                    if labels_changed_in_chunk: segmentation_memmap[z_start:z_end] = seg_chunk_modified; segmentation_memmap.flush()
                    del seg_chunk, seg_chunk_modified, unique_labels_in_chunk; gc.collect()
                print(f"    Merge applied.")
            else: print("    No labels needed merging.")
        else: print("    Skipping merge check: Only one label present.")
    except Exception as e_merge: print(f"!!! Error during merge step: {e_merge}. Merge step incomplete."); gc.collect()
    print(f"  Time taken for merging: {time.time() - start_merge_time:.2f}s."); gc.collect()

    ## MODIFICATION: ADDED a final, definitive relabeling step after the merge.
    # --- Step 6: Final Relabeling of Disconnected Components ---
    print("\n--- 6. Final relabeling of disconnected components (post-merge) ---")
    start_final_relabel_time = time.time(); final_relabel_count = 0
    try:
        # This is a full rerun of the relabeling logic on the now-merged data.
        max_label = 0; chunk_size_max = min(500, original_shape[0]) if original_shape[0] > 0 else 1
        for i in tqdm(range(0, original_shape[0], chunk_size_max), desc="    Finding Max Label (Final)", disable=original_shape[0]<=chunk_size_max):
             seg_chunk = segmentation_memmap[i:min(i+chunk_size_max, original_shape[0])]
             try: chunk_max = np.max(seg_chunk) if seg_chunk.size > 0 else 0
             except ValueError: chunk_max = 0
             if chunk_max > max_label: max_label = int(chunk_max)
        next_available_label = max_label + 1
        print(f"    Max label post-merge: {max_label}. Next new label: {next_available_label}")
        
        current_labels_present = np.unique(segmentation_memmap); current_labels_present = current_labels_present[current_labels_present != 0]
        print(f"    Found {len(current_labels_present)} unique labels to check for final relabeling.")

        structure_relabel = generate_binary_structure(3, 1)
        all_locations = find_objects(segmentation_memmap); print(f"    Found {len(all_locations)} bounding boxes (some may be None).")
        
        for label_val in tqdm(current_labels_present, desc="  Final Relabeling"):
            if label_val <= 0 or label_val > len(all_locations) or all_locations[label_val - 1] is None: continue
            loc = all_locations[label_val - 1]
            try:
                seg_sub_data = np.array(segmentation_memmap[loc]); label_mask_sub = (seg_sub_data == label_val)
                if not np.any(label_mask_sub): del seg_sub_data, label_mask_sub; continue
                
                labeled_components_sub, num_components = ndimage_label(label_mask_sub, structure=structure_relabel)
                if num_components > 1:
                    component_sizes = np.bincount(labeled_components_sub.ravel()); component_sizes[0] = 0
                    largest_component_idx = np.argmax(component_sizes)
                    modified_in_sub = False
                    for c_idx in range(1, num_components + 1):
                        if c_idx == largest_component_idx: continue
                        component_mask_sub = (labeled_components_sub == c_idx); new_label = next_available_label
                        seg_sub_data[component_mask_sub] = new_label; modified_in_sub = True
                        next_available_label += 1; final_relabel_count += 1
                    if modified_in_sub: segmentation_memmap[loc] = seg_sub_data; segmentation_memmap.flush()
                del seg_sub_data, label_mask_sub, labeled_components_sub
            except MemoryError: print(f"\n!!! MemoryError during final relabeling of label {label_val}. Skipping this label."); gc.collect(); continue
            except Exception as rel_err: print(f"\n!!! Error during final relabeling of label {label_val}: {rel_err}. Skipping this label."); gc.collect(); continue
        print(f"  Final relabeling finished. Created {final_relabel_count} new labels to enforce unique IDs.")
    except Exception as e: print(f"Error during final relabeling: {e}."); gc.collect()
    print(f"  Time taken: {time.time() - start_final_relabel_time:.2f}s."); gc.collect()

    # --- Step 7: Final Cleanup of Small Objects ---
    ## MODIFICATION: Renumbered this to Step 7.
    if min_remaining_size > 0:
        print(f"\n--- 7. Final cleanup of objects smaller than {min_remaining_size} voxels ---")
        final_remnants_removed = 0; start_final_rem_time = time.time(); bool_seg_temp_dir = None; bool_seg_memmap = None
        try:
            # Re-run the small object removal logic from step 2
            bool_seg_temp_dir = tempfile.mkdtemp(prefix="bool_final_remnant_")
            bool_seg_path = os.path.join(bool_seg_temp_dir, 'temp_bool_seg.dat')
            bool_seg_memmap = np.memmap(bool_seg_path, dtype=bool, mode='w+', shape=original_shape)
            any_objects_found = False
            for i in tqdm(range(0, original_shape[0], 100), desc="    Final Bool Mask Gen"):
                z_start, z_end = i, min(i + 100, original_shape[0])
                if z_start >= z_end: continue
                bool_chunk = segmentation_memmap[z_start:z_end] > 0
                bool_seg_memmap[z_start:z_end] = bool_chunk
                if not any_objects_found and np.any(bool_chunk): any_objects_found = True
            bool_seg_memmap.flush(); gc.collect()
            if any_objects_found:
                try:
                    bool_seg_in_memory = np.array(bool_seg_memmap); del bool_seg_memmap; os.unlink(bool_seg_path); rmtree(bool_seg_temp_dir)
                    cleaned_bool_seg = remove_small_objects(bool_seg_in_memory, min_size=min_remaining_size, connectivity=1)
                    small_remnant_mask_overall = bool_seg_in_memory & (~cleaned_bool_seg); del bool_seg_in_memory, cleaned_bool_seg; gc.collect()
                    for i in tqdm(range(0, original_shape[0], 100), desc="    Applying Final Removal"):
                        z_start, z_end = i, min(i + 100, original_shape[0])
                        if z_start >= z_end: continue
                        seg_chunk_view = segmentation_memmap[z_start:z_end]; small_remnant_mask_chunk = small_remnant_mask_overall[z_start:z_end]
                        chunk_remnants_count = np.sum(seg_chunk_view[small_remnant_mask_chunk] > 0)
                        if chunk_remnants_count > 0: seg_chunk_view[small_remnant_mask_chunk] = 0; segmentation_memmap.flush(); final_remnants_removed += chunk_remnants_count
                    del small_remnant_mask_overall; print(f"  Final small object removal finished. Removed {final_remnants_removed} voxels.")
                except Exception as e_rem: print(f"    Error during final small remnant removal: {e_rem}. Skipping."); gc.collect()
            else: print("  Segmentation memmap appears empty. Skipping final remnant removal.")
        finally:
             if 'bool_seg_memmap' in locals() and bool_seg_memmap is not None and hasattr(bool_seg_memmap, '_mmap'): del bool_seg_memmap
             if 'bool_seg_temp_dir' in locals() and bool_seg_temp_dir is not None and os.path.exists(bool_seg_temp_dir): rmtree(bool_seg_temp_dir, ignore_errors=True)
             gc.collect()
        print(f"  Time taken: {time.time() - start_final_rem_time:.2f}s.")
    else: print(f"\n--- 7. Skipping final small object removal (min_remaining_size={min_remaining_size}) ---")

    # --- Final Cleanup ---
    print("\n--- Cleaning up temporary files ---")
    if original_segmentation_memmap is not None and hasattr(original_segmentation_memmap, '_mmap'): del original_segmentation_memmap; gc.collect()
    if original_seg_temp_dir and os.path.exists(original_seg_temp_dir):
        try: rmtree(original_seg_temp_dir, ignore_errors=True)
        except Exception as e_clean: print(f"  Warn: Error cleaning up original seg temp dir: {e_clean}")
    gc.collect()

    print("\n--- Processing sequence finished ---")
    return trimmed_voxels_mask