import numpy as np
from scipy import ndimage
from scipy.ndimage import (distance_transform_edt,
                           binary_erosion, generate_binary_structure)
from scipy.ndimage import (distance_transform_edt, label as ndimage_label, find_objects,
                           binary_opening, binary_closing)
from skimage.filters import threshold_otsu
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


def trim_object_edges_by_distance(
    segmentation_memmap,     # Labeled segmentation MEMMAP (opened r+ OR w+)
    original_volume,         # Original intensity volume (ndarray or memmap)
    hull_boundary_mask,      # Boolean mask of the hull boundary (in-memory)
    spacing,                 # Voxel spacing for distance calculation
    distance_threshold,      # Physical distance threshold
    global_brightness_cutoff,# GLOBAL intensity value threshold for trimming
    min_remaining_size=10,   # Minimum size for object remnants after trimming/relabeling
    chunk_size_z=32,         # Chunk size for chunked processing
    heal_iterations=1        # <-- NEW: Iterations for grey_closing to heal edges
    ):
    """
    Removes voxels from labeled objects near the hull boundary AND below a
    GLOBAL brightness cutoff. Then attempts to heal small gaps created by
    trimming using grey closing, constrained by the object shapes before trimming.
    Optionally removes objects that become too small after trimming/healing.
    If trimming/healing disconnects an object, components are relabeled.
    Operates IN-PLACE on the input segmentation_memmap.
    Uses chunked processing for distance calculation and trimming.
    """
    print(f"\n--- Trimming object edges (Dist < {distance_threshold:.2f} AND Brightness < {global_brightness_cutoff:.4f}) ---")
    print(f"--- Healing porous edges ({heal_iterations} iter) ---") # Announce healing
    original_shape = segmentation_memmap.shape
    trimmed_voxels_mask = np.zeros(original_shape, dtype=bool) # Track initial trims

    if not np.any(hull_boundary_mask):
        print("  Hull boundary mask is empty. No trimming or healing performed.")
        return trimmed_voxels_mask

    # Check if memmap is writable
    writable_modes = ['r+', 'w+']
    if segmentation_memmap.mode not in writable_modes:
         raise ValueError(f"segmentation_memmap must be opened in a writable mode ({writable_modes}) for in-place modification. Found mode: '{segmentation_memmap.mode}'")

    # Get original labels present before trimming
    print("  Finding unique labels before trimming...")
    original_labels_present = np.unique(segmentation_memmap)
    original_labels_present = original_labels_present[original_labels_present != 0]
    print(f"  Found {len(original_labels_present)} unique labels initially.")
    if len(original_labels_present) == 0:
         print("  No objects found in segmentation. Skipping trimming and healing.")
         return trimmed_voxels_mask

    # --- Step 0: Create Pre-Trimming Mask ---
    # This mask defines the maximum extent allowed after healing.
    print("  Creating pre-trimming boolean mask (chunked)...")
    pre_trim_mask_memmap = None
    pre_trim_temp_dir = None
    try:
        pre_trim_temp_dir = tempfile.mkdtemp(prefix="pre_trim_mask_")
        pre_trim_mask_path = os.path.join(pre_trim_temp_dir, 'pre_trim_mask.dat')
        pre_trim_mask_memmap = np.memmap(pre_trim_mask_path, dtype=bool, mode='w+', shape=original_shape)

        chunk_size_mask_gen = min(100, original_shape[0]) if original_shape[0] > 0 else 1
        for i in tqdm(range(0, original_shape[0], chunk_size_mask_gen), desc="    Pre-Trim Mask Gen"):
            z_start = i; z_end = min(i + chunk_size_mask_gen, original_shape[0])
            if z_start >= z_end: continue
            # Ensure memmap is valid before reading
            if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None:
                raise IOError("Segmentation memmap became invalid during pre-trim mask generation")
            seg_chunk = segmentation_memmap[z_start:z_end]
            pre_trim_mask_memmap[z_start:z_end] = seg_chunk > 0
            del seg_chunk
        pre_trim_mask_memmap.flush()
        print("  Pre-trimming mask created.")
        gc.collect()
    except Exception as e_mask:
        print(f"\n!!! ERROR creating pre-trim mask: {e_mask}. Cannot perform healing.")
        # Cleanup if creation failed partially
        if pre_trim_mask_memmap is not None and hasattr(pre_trim_mask_memmap, '_mmap'):
            del pre_trim_mask_memmap
            gc.collect()
        if pre_trim_temp_dir and os.path.exists(pre_trim_temp_dir):
            rmtree(pre_trim_temp_dir, ignore_errors=True)
        # Set heal_iterations to 0 so the healing step is skipped
        heal_iterations = 0
        pre_trim_mask_memmap = None # Ensure it's None if failed


    # --- Step 1: Chunked Processing for Distance Calc and Combined Trimming ---
    print("\n  Executing chunked distance transform and combined (distance+global brightness) trimming...")
    gc.collect()
    num_chunks = math.ceil(original_shape[0] / chunk_size_z)
    total_trimmed_in_chunks = 0
    start_chunked_time = time.time()

    # ... (hull boundary check remains same) ...
    print(f"  Processing {num_chunks} chunks (chunk_size_z={chunk_size_z}).")

    for i in tqdm(range(num_chunks), desc="  Processing Chunks (Trim)"):
        z_start = i * chunk_size_z
        z_end = min((i + 1) * chunk_size_z, original_shape[0])
        if z_start >= z_end: continue

        try:
            # 1. Calculate Distance Transform for the chunk
            # ... (Distance calculation logic remains the same) ...
            inv_boundary_chunk = ~hull_boundary_mask[z_start:z_end]
            edt_chunk = None
            if not np.any(inv_boundary_chunk): edt_chunk = np.zeros(inv_boundary_chunk.shape, dtype=np.float32)
            elif not np.all(inv_boundary_chunk): edt_chunk = distance_transform_edt(inv_boundary_chunk, sampling=spacing).astype(np.float32)
            else: edt_chunk = np.full(inv_boundary_chunk.shape, np.finfo(np.float32).max, dtype=np.float32)
            close_to_boundary_chunk = (edt_chunk < distance_threshold)
            del inv_boundary_chunk, edt_chunk; gc.collect()


            # 2. Load segmentation and intensity chunks
            if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None:
                 raise IOError(f"Segmentation memmap became invalid before reading chunk {i}")
            seg_chunk_data = np.array(segmentation_memmap[z_start:z_end]) # Read into memory for check
            intensity_chunk = original_volume[z_start:z_end]

            # 3. Identify voxels below GLOBAL brightness cutoff
            dim_voxels_chunk = (intensity_chunk < global_brightness_cutoff)
            del intensity_chunk; gc.collect()

            # 4. Combine criteria: Must be object, close, AND dim
            trim_target_mask_chunk = (seg_chunk_data > 0) & close_to_boundary_chunk & dim_voxels_chunk
            del seg_chunk_data, close_to_boundary_chunk, dim_voxels_chunk; gc.collect()

            # 5. Apply Trimming (Set identified voxels to 0)
            num_trimmed_chunk = np.sum(trim_target_mask_chunk)
            if num_trimmed_chunk > 0:
                 seg_chunk_view_write = segmentation_memmap[z_start:z_end] # Get view for writing
                 # Record which voxels were initially trimmed FOR DEBUGGING/INFO
                 trimmed_voxels_mask[z_start:z_end][trim_target_mask_chunk] = True
                 # Apply trimming
                 seg_chunk_view_write[trim_target_mask_chunk] = 0
                 segmentation_memmap.flush() # Flush changes for this chunk
                 total_trimmed_in_chunks += num_trimmed_chunk
                 del seg_chunk_view_write # Release view

            del trim_target_mask_chunk
            gc.collect()

        except MemoryError as mem_err:
            # ... (Error handling) ...
            raise MemoryError(f"Chunked trimming failed at chunk {i} due to MemoryError.") from mem_err
        except Exception as chunk_err:
            # ... (Error handling) ...
             raise RuntimeError(f"Chunked trimming failed at chunk {i}.") from chunk_err

    print(f"  Chunked trimming applied. Total voxels initially trimmed: {total_trimmed_in_chunks} in {time.time() - start_chunked_time:.2f}s.")
    gc.collect()

    # --- Step 1.5: Heal Porous Edges (Optional) ---
    if heal_iterations > 0 and pre_trim_mask_memmap is not None:
        print(f"\n  Applying grey closing ({heal_iterations} iter) to heal edges...")
        start_heal_time = time.time()
        healed_labels_memmap = None
        healed_temp_dir = None
        try:
            # Create temporary memmap for the healed result
            healed_temp_dir = tempfile.mkdtemp(prefix="healed_labels_")
            healed_labels_path = os.path.join(healed_temp_dir, 'healed_labels.dat')
            healed_labels_memmap = np.memmap(healed_labels_path, dtype=segmentation_memmap.dtype, mode='w+', shape=original_shape)

            # Define structure for closing
            closing_structure = generate_binary_structure(3, 1) # Connectivity 1 (3x3x3 neighborhood)

            # Apply grey closing (chunked if necessary, but often feasible directly on memmap)
            # Note: grey_closing reads the input memmap and writes to the output memmap.
            # It's generally memory-efficient for moderate structures/iterations.
            print("    Executing grey closing...")
            ndimage.grey_closing(
                input=segmentation_memmap,
                size=None, # Size is ignored if structure is provided
                footprint=closing_structure,
                output=healed_labels_memmap,
                mode='reflect', # Or choose appropriate mode
                # iterations parameter doesn't exist for grey_closing, apply multiple times if needed
                # For multiple iterations: loop apply grey_closing swapping input/output or use intermediate storage
            )
            if heal_iterations > 1: # Handle multiple iterations if specified
                 print(f"    Applying grey closing for {heal_iterations} total iterations...")
                 temp_swap_memmap = np.memmap(os.path.join(healed_temp_dir, 'swap.dat'), dtype=segmentation_memmap.dtype, mode='w+', shape=original_shape)
                 current_input = healed_labels_memmap # Start with result of first iteration
                 current_output = temp_swap_memmap
                 for it in range(heal_iterations - 1):
                     print(f"      Iteration {it+2}/{heal_iterations}")
                     ndimage.grey_closing(current_input, footprint=closing_structure, output=current_output, mode='reflect')
                     current_input, current_output = current_output, current_input # Swap for next iteration
                 # Ensure the final result is in healed_labels_memmap
                 if (heal_iterations-1) % 2 == 0: # Odd number of extra iterations, result is in temp_swap
                     healed_labels_memmap[:,:,:] = temp_swap_memmap[:,:,:]
                 del temp_swap_memmap
                 gc.collect()


            healed_labels_memmap.flush()
            print("    Grey closing finished.")

            # Apply the pre-trim mask to constrain the result
            print("    Applying pre-trim mask to constrain healed result (chunked)...")
            chunk_size_mask_apply = min(100, original_shape[0]) if original_shape[0] > 0 else 1
            voxels_restored_count = 0
            original_zeros_mask_chunk = None # To calculate restored count accurately
            healed_chunk = None

            for i in tqdm(range(0, original_shape[0], chunk_size_mask_apply), desc="    Applying Heal Mask"):
                z_start = i; z_end = min(i + chunk_size_mask_apply, original_shape[0])
                if z_start >= z_end: continue

                # Read chunks needed
                if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None: raise IOError("Seg memmap invalid during heal masking")
                if not hasattr(healed_labels_memmap, '_mmap') or healed_labels_memmap._mmap is None: raise IOError("Healed memmap invalid during heal masking")
                if not hasattr(pre_trim_mask_memmap, '_mmap') or pre_trim_mask_memmap._mmap is None: raise IOError("Pre-trim mask invalid during heal masking")

                original_zeros_mask_chunk = (segmentation_memmap[z_start:z_end] == 0)
                healed_chunk = healed_labels_memmap[z_start:z_end]
                pre_trim_mask_chunk = pre_trim_mask_memmap[z_start:z_end]

                # Apply the mask: Set voxels outside pre-trim mask back to 0
                healed_chunk[~pre_trim_mask_chunk] = 0

                # Calculate how many zero voxels were filled *within* the pre-trim mask
                restored_in_chunk = np.sum(original_zeros_mask_chunk & (healed_chunk != 0))
                voxels_restored_count += restored_in_chunk

                # Write the masked healed chunk back to the ORIGINAL segmentation memmap
                segmentation_memmap[z_start:z_end] = healed_chunk
                segmentation_memmap.flush() # Flush changes for the chunk

                # Clean up chunk memory
                del original_zeros_mask_chunk, healed_chunk, pre_trim_mask_chunk
                gc.collect()

            print(f"  Edge healing finished. Filled approx. {voxels_restored_count} voxels in {time.time() - start_heal_time:.2f}s.")

        except MemoryError as mem_err:
            print(f"  !!! MemoryError during healing step: {mem_err}. Skipping healing.")
        except Exception as e_heal:
            print(f"  !!! ERROR during healing step: {e_heal}. Skipping healing.")
        finally:
            # Clean up temporary healed memmap and pre-trim mask
            if healed_labels_memmap is not None and hasattr(healed_labels_memmap, '_mmap'):
                del healed_labels_memmap; gc.collect()
            if healed_temp_dir and os.path.exists(healed_temp_dir):
                rmtree(healed_temp_dir, ignore_errors=True)
            if pre_trim_mask_memmap is not None and hasattr(pre_trim_mask_memmap, '_mmap'):
                del pre_trim_mask_memmap; gc.collect()
            if pre_trim_temp_dir and os.path.exists(pre_trim_temp_dir):
                rmtree(pre_trim_temp_dir, ignore_errors=True)
            gc.collect()
    elif heal_iterations > 0:
        print("\n  Skipping healing step because pre-trim mask generation failed.")
    else:
        # Cleanup pre-trim mask if it was created but healing was skipped
        if pre_trim_mask_memmap is not None and hasattr(pre_trim_mask_memmap, '_mmap'):
            del pre_trim_mask_memmap; gc.collect()
        if pre_trim_temp_dir and os.path.exists(pre_trim_temp_dir):
            rmtree(pre_trim_temp_dir, ignore_errors=True)


    # --- Step 2: Relabel Disconnected Components ---
    # This step now runs on the potentially healed segmentation
    print("\n  Checking for and relabeling disconnected components (post-trim/heal)...")
    # ... (Relabeling logic remains exactly the same) ...
    start_relabel_time = time.time()
    try:
        print("    Finding max label after trimming/healing...")
        max_label = 0
        chunk_size_max = min(500, original_shape[0]) if original_shape[0] > 0 else 1
        for i in tqdm(range(0, original_shape[0], chunk_size_max), desc="    Finding Max Label", disable=original_shape[0]<=chunk_size_max):
             # Check memmap validity before reading chunk
             if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None:
                 raise IOError(f"Segmentation memmap became invalid while finding max label (chunk {i})")
             seg_chunk = segmentation_memmap[i:min(i+chunk_size_max, original_shape[0])]
             chunk_max = np.max(seg_chunk); del seg_chunk
             if chunk_max > max_label: max_label = chunk_max
        next_available_label = max_label + 1
        print(f"    Max label found: {max_label}. Next new label: {next_available_label}")
    except MemoryError: raise MemoryError("Failed to find max label for relabeling.")
    except Exception as e: raise RuntimeError(f"Failed to find max label for relabeling: {e}") from e
    relabel_count = 0
    structure_relabel = generate_binary_structure(3,1) # Define structure once
    for label_val in tqdm(original_labels_present, desc="  Relabeling Components"):
        # Check if label still exists after trimming/healing before processing
        # (Could add a check here, but find_objects handles empty results)

        # Ensure memmap is valid for each label
        if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None:
            raise IOError(f"Segmentation memmap became invalid before relabeling label {label_val}")

        try:
            # Find bounding box slice for the label
            locations = find_objects(segmentation_memmap == label_val) # Still operates on boolean check per label
            if not locations: continue # Label might have been fully removed
            loc = locations[0]
            del locations # Free memory

            # Extract sub-volume view for labeling
            seg_sub_view = segmentation_memmap[loc]
            label_mask_sub = (seg_sub_view == label_val) # Create mask within sub-volume

            if not np.any(label_mask_sub):
                del loc, seg_sub_view, label_mask_sub; gc.collect(); continue # Should not happen if find_objects worked, but safety check

            # Label components within the sub-volume mask
            labeled_components_sub, num_components = ndimage_label(label_mask_sub, structure=structure_relabel)

            # If disconnected, relabel the smaller components
            if num_components > 1:
                relabel_count += (num_components - 1)
                # Modify the sub-volume view directly
                for c_idx in range(2, num_components + 1): # Start from component 2
                    component_mask_sub = (labeled_components_sub == c_idx)
                    # Assign the next available global label
                    seg_sub_view[component_mask_sub] = next_available_label
                    next_available_label += 1
                segmentation_memmap.flush() # Flush changes made through the view

            # Clean up sub-volume variables
            del loc, seg_sub_view, label_mask_sub, labeled_components_sub; gc.collect()

        except MemoryError: print(f"\n!!! MemoryError during find_objects or labeling for label {label_val}. Skipping relabeling."); gc.collect(); continue
        except Exception as find_obj_err: print(f"\n!!! Error during relabeling process for label {label_val}: {find_obj_err}. Skipping."); gc.collect(); continue

    print(f"  Relabeling finished. Created {relabel_count} new labels in {time.time() - start_relabel_time:.2f}s."); gc.collect()


    # --- Step 3: Optional: Remove Small Remnants (based on final labels) ---
    # This now operates on the trimmed, healed, and relabeled segmentation
    if min_remaining_size > 0:
        print(f"\n  Removing remnants smaller than {min_remaining_size} voxels (on Memmap, post-relabel)...")
        # ... (Small remnant removal logic remains the same) ...
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
                if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None: raise IOError("Seg memmap invalid during bool mask gen")
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
                    # Load boolean mask into memory for remove_small_objects
                    bool_seg_in_memory = np.array(bool_seg_memmap)
                    # Close and delete the temporary boolean memmap file NOW to save disk space/handles
                    del bool_seg_memmap; bool_seg_memmap = None; gc.collect()
                    os.unlink(bool_seg_path); rmtree(bool_seg_temp_dir); bool_seg_temp_dir = None

                    print("    Applying remove_small_objects...")
                    # remove_small_objects requires boolean input
                    cleaned_bool_seg = remove_small_objects(bool_seg_in_memory, min_size=min_remaining_size, connectivity=1)
                    # Calculate the difference mask (objects TO BE removed)
                    small_remnant_mask_overall = bool_seg_in_memory & (~cleaned_bool_seg)
                    del bool_seg_in_memory, cleaned_bool_seg; gc.collect() # Free memory

                    print("    Applying removal mask to segmentation memmap (chunked)...")
                    chunk_size_apply_rem = min(100, original_shape[0]) if original_shape[0] > 0 else 1
                    for i in tqdm(range(0, original_shape[0], chunk_size_apply_rem), desc="    Applying Removal"):
                        z_start = i; z_end = min(i + chunk_size_apply_rem, original_shape[0])
                        if z_start >= z_end: continue
                        if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None: raise IOError("Segmentation memmap became invalid during remnant removal application")

                        # Get view of segmentation chunk and mask chunk
                        seg_chunk_view = segmentation_memmap[z_start:z_end]
                        small_remnant_mask_chunk = small_remnant_mask_overall[z_start:z_end]

                        # Find where small remnants exist in the chunk
                        chunk_remnants_count = np.sum(seg_chunk_view[small_remnant_mask_chunk] > 0) # Count only labeled pixels being removed
                        if chunk_remnants_count > 0:
                            # Set labels in the segmentation memmap to 0 where the mask is True
                            seg_chunk_view[small_remnant_mask_chunk] = 0
                            segmentation_memmap.flush() # Flush changes
                            num_remnants_removed += chunk_remnants_count

                    del small_remnant_mask_overall # Free memory of the full mask
                    gc.collect()
                    print(f"  Removed {num_remnants_removed} voxels belonging to small remnants in {time.time()-start_rem:.2f}s.")
                except MemoryError: print("    MemoryError during small object removal processing (loading bool mask or applying). Skipping remnant removal."); gc.collect()
                except Exception as e_rem: print(f"    Error during small remnant removal processing: {e_rem}. Skipping."); gc.collect()
            else: print("  Segmentation memmap appears empty after trimming/healing/relabeling, skipping remnant removal.")
        finally:
             # Ensure temp dir/files are cleaned up even if loading bool mask failed
             if 'bool_seg_memmap' in locals() and bool_seg_memmap is not None and hasattr(bool_seg_memmap, '_mmap'): del bool_seg_memmap; gc.collect()
             if 'bool_seg_temp_dir' in locals() and bool_seg_temp_dir is not None and os.path.exists(bool_seg_temp_dir):
                 try: rmtree(bool_seg_temp_dir, ignore_errors=True)
                 except Exception as e_final_clean: print(f"    Warn: Error cleaning up remnant temp dir {bool_seg_temp_dir}: {e_final_clean}")
             gc.collect()


    print("--- Edge trimming, healing, and cleanup finished ---")
    # The segmentation_memmap has been modified in-place.
    return trimmed_voxels_mask # Return mask of voxels *initially* trimmed (before healing)
