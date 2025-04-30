# --- START OF FILE utils/ramified_module_2d/remove_artifacts_2d.py ---

import numpy as np
from scipy import ndimage
from scipy.ndimage import (distance_transform_edt,
                           binary_erosion, generate_binary_structure)
from scipy.ndimage import (distance_transform_edt, label as ndimage_label, find_objects,
                           binary_opening, binary_closing, binary_dilation)
from skimage.filters import threshold_otsu # type: ignore
from skimage.morphology import convex_hull_image, remove_small_objects, disk # type: ignore
import tempfile
import os
from tqdm import tqdm
# Multiprocessing likely not needed for single 2D image steps
# from multiprocessing import Pool
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

# --- generate_hull_boundary_and_stack_2d ---
def generate_hull_boundary_and_stack_2d(
    image,                      # Original 2D intensity image
    cell_mask,                  # Labeled or boolean mask of segmented cells (2D)
    hull_erosion_iterations=1,  # Iterations for final boundary erosion (2D)
    smoothing_iterations=1      # Iterations for 2D closing/opening to smooth hull
    ):
    """
    Generates a 2D hull and its boundary based on tissue intensity
    and an existing cell segmentation.

    Steps:
    1. Creates a tissue mask using Otsu thresholding on the input image.
    2. Combines the tissue mask with the cell mask (logical OR).
    3. Computes the 2D convex hull of this combined mask to get `initial_hull`.
    4. (Optional) Applies 2D morphological closing then opening to smooth
       the `initial_hull`.
    5. Erodes the smoothed hull using a 2D structure.
    6. Calculates the `hull_boundary` as the difference between the smoothed
       hull and the eroded hull.

    Parameters:
    -----------
    image : ndarray (2D)
        Input 2D intensity image. Used for Otsu thresholding.
    cell_mask : ndarray (bool or int, 2D)
        A mask (boolean or labeled integers) where non-zero values indicate
        segmented cells that *must* be contained within the hull.
        Must have the same shape as `image`.
    hull_erosion_iterations : int, optional
        Number of iterations for the final 2D binary erosion to define the
        thickness of the boundary. Defaults to 1.
    smoothing_iterations : int, optional
        Number of iterations for 2D binary closing and opening applied to the
        initial hull to smooth it. Set to 0 to disable. Defaults to 1.

    Returns:
    --------
    hull_boundary : ndarray (bool, 2D)
        Boolean mask indicating the boundary region of the final smoothed hull.
        Returns None if hull generation fails.
    smoothed_hull : ndarray (bool, 2D)
        The final, smoothed 2D hull (after potential closing/opening).
        Returns None if hull generation fails.
    """
    print("\n--- Generating Smoothed 2D Hull Boundary and Image ---")
    original_shape = image.shape
    if cell_mask.shape != original_shape:
        raise ValueError(f"Shape mismatch: image {original_shape} vs cell_mask {cell_mask.shape}")
    if image.ndim != 2:
        raise ValueError(f"Input image must be 2D, got {image.ndim} dimensions.")

    print(f"Input shape: {original_shape}")
    print(f"Smoothing iterations: {smoothing_iterations}")
    print(f"Boundary erosion iterations: {hull_erosion_iterations}")
    initial_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Memory usage at start: {initial_mem:.2f} MB")

    # --- Step 1: Create Tissue Mask using Otsu (2D) ---
    print("Step 1: Creating base tissue mask using Otsu threshold (2D)...")
    tissue_mask = np.zeros(original_shape, dtype=bool)
    try:
        # Sample for Otsu (can sample directly from 2D)
        otsu_sample_size = min(2_000_000, image.size)
        otsu_samples = np.random.choice(image.ravel(), otsu_sample_size, replace=False)

        if np.all(otsu_samples == otsu_samples[0]):
            tissue_thresh = otsu_samples[0]
            print(f"  Warning: Sampled intensity values are constant ({tissue_thresh}).")
        elif otsu_samples.size > 0:
            tissue_thresh = threshold_otsu(otsu_samples)
            print(f"  Otsu tissue threshold determined: {tissue_thresh:.2f}")
        else:
             print("  Warning: No valid samples for Otsu. Using threshold 0.")
             tissue_thresh = 0

        # Apply threshold directly to 2D image
        tissue_mask = image > tissue_thresh
        del otsu_samples
        gc.collect()

        if not np.any(tissue_mask):
            print("  Warning: Otsu tissue mask is empty.")

    except Exception as e:
        print(f"  Error during Otsu thresholding: {e}. Proceeding without tissue mask.")
        tissue_mask = np.zeros(original_shape, dtype=bool)

    # --- Step 2: Combine Masks and Generate 2D Hull ---
    print("\nStep 2: Generating initial 2D hull from combined masks...")
    cell_mask_bool = cell_mask > 0
    initial_hull = np.zeros(original_shape, dtype=bool)

    # Combine tissue and cell masks (2D)
    combined_mask = tissue_mask | cell_mask_bool

    if np.any(combined_mask):
        try:
            if not combined_mask.flags['C_CONTIGUOUS']:
                combined_mask = np.ascontiguousarray(combined_mask)
            initial_hull = convex_hull_image(combined_mask)
            print("  Initial 2D hull generated.")
        except Exception as e:
            print(f"\nWarning: Failed to compute 2D convex hull: {e}")
            # Leave hull as zeros
    else:
        print("  Warning: Combined mask for hull generation is empty.")

    del tissue_mask, cell_mask_bool, combined_mask
    gc.collect()

    if not np.any(initial_hull):
         print("  Warning: Initial 2D hull is empty. Cannot proceed.")
         return None, None

    # --- Step 3: Smooth Hull in 2D (Optional) ---
    smoothed_hull = initial_hull # Start with the initial hull
    if smoothing_iterations > 0:
        print(f"\nStep 3: Smoothing 2D hull with {smoothing_iterations} iteration(s) of closing/opening...")
        # Use 2D structure (connectivity=1 -> diamond, connectivity=2 -> square)
        struct_2d = generate_binary_structure(2, 1) # Diamond shape (4-connectivity)
        # Alternative: struct_2d = disk(1) # Square connectivity equivalent for radius 1
        try:
            print("  Applying 2D binary closing...")
            smoothed_hull = binary_closing(smoothed_hull, structure=struct_2d, iterations=smoothing_iterations, border_value=0)
            print("  Applying 2D binary opening...")
            smoothed_hull = binary_opening(smoothed_hull, structure=struct_2d, iterations=smoothing_iterations, border_value=0)
            print("  2D smoothing complete.")
        except MemoryError:
            print("  Warning: MemoryError during 2D smoothing. Using unsmoothed hull.")
            smoothed_hull = initial_hull
        except Exception as e:
            print(f"  Warning: Error during 2D smoothing ({e}). Using unsmoothed hull.")
            smoothed_hull = initial_hull
        gc.collect()
    else:
        print("\nStep 3: Skipping 2D hull smoothing.")

    # --- Step 4: Erode Smoothed Hull (2D) ---
    hull_boundary = np.zeros(original_shape, dtype=bool)
    if hull_erosion_iterations > 0:
        print(f"\nStep 4: Eroding smoothed 2D hull ({hull_erosion_iterations} iterations)...")
        struct_2d = generate_binary_structure(2, 1) # Use same structure
        try:
            eroded_hull = binary_erosion(smoothed_hull, structure=struct_2d, iterations=hull_erosion_iterations, border_value=0)
            print("  2D Erosion complete.")

            # --- Step 5: Calculate Hull Boundary (2D) ---
            print("\nStep 5: Calculating final 2D hull boundary...")
            hull_boundary = smoothed_hull & (~eroded_hull)
            boundary_pixel_count = np.sum(hull_boundary)
            print(f"  2D Hull boundary mask created ({boundary_pixel_count} pixels).")
            del eroded_hull

        except MemoryError: print("  Warning: MemoryError during 2D hull erosion/boundary calculation. Boundary empty.")
        except Exception as e: print(f"  Warning: Error during 2D hull erosion/boundary calculation ({e}). Boundary empty.")
        gc.collect()
    else:
        print("\nSteps 4 & 5: Skipping 2D hull erosion and boundary calculation.")

    final_mem = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"Memory usage at end: {final_mem:.2f} MB")
    print("--- 2D Hull generation finished ---")

    return hull_boundary, smoothed_hull


# --- Union-Find Helper Functions (Keep as they are logic-independent of dimension) ---
union_find_parent = {} # Global dictionary for the current operation

def find_set(i):
    """Finds the representative (root) of the set containing i, with path compression."""
    global union_find_parent
    root = i
    while union_find_parent.get(root, root) != root:
        root = union_find_parent.get(root, root)
    curr = i
    while union_find_parent.get(curr, curr) != root:
        next_node = union_find_parent.get(curr, curr)
        union_find_parent[curr] = root
        curr = next_node
    if i not in union_find_parent:
        union_find_parent[i] = i
    return union_find_parent.get(i, i)

def unite_sets(i, j):
    """Merges the sets containing i and j, merging into the smaller root ID."""
    global union_find_parent
    root_i = find_set(i)
    root_j = find_set(j)
    if root_i != root_j:
        if root_i < root_j:
            union_find_parent[root_j] = root_i
        else:
            union_find_parent[root_i] = root_j
# --- End Union-Find ---


# --- trim_object_edges_by_distance_2d ---
def trim_object_edges_by_distance_2d(
    segmentation_memmap,     # Labeled 2D segmentation MEMMAP (opened r+ OR w+)
    original_image,          # Original 2D intensity image (ndarray or memmap)
    hull_boundary_mask,      # Boolean mask of the 2D hull boundary (in-memory)
    spacing,                 # Pixel spacing (y, x) for distance calculation
    distance_threshold,      # Physical distance threshold
    global_brightness_cutoff,# GLOBAL intensity value threshold for trimming
    min_remaining_size=10,   # Minimum size (pixels) for object remnants
    heal_iterations=1        # Parameter for potential future healing methods (unused by convex hull)
    ):
    """
    Applies operations in the order: Trim -> Remove Small -> Relabel -> Heal -> Merge Touching (2D).
    1. Trims pixels from labeled objects near hull boundary AND below global brightness cutoff.
    2. Removes objects/fragments smaller than min_remaining_size.
    3. Relabels disconnected components.
    4. Heals porosity using the 2D convex hull method.
    5. Merges any distinct labels that are directly touching after healing.
    Assumes scikit-image is installed. Operates IN-PLACE on the input segmentation_memmap.
    """
    print("\n--- 2D Processing Order: Trim -> Remove Small -> Relabel -> Heal -> Merge Touching ---")

    original_shape = segmentation_memmap.shape
    if original_shape != original_image.shape or original_shape != hull_boundary_mask.shape:
        raise ValueError("Shape mismatch between segmentation, original image, and hull boundary.")
    if len(original_shape) != 2:
        raise ValueError("Input arrays must be 2D.")

    trimmed_pixels_mask = np.zeros(original_shape, dtype=bool) # Track initial trims

    if not np.any(hull_boundary_mask):
        print("  Hull boundary mask is empty. No processing performed.")
        return trimmed_pixels_mask

    writable_modes = ['r+', 'w+']
    if segmentation_memmap.mode not in writable_modes:
         raise ValueError(f"segmentation_memmap must be writable. Found mode: '{segmentation_memmap.mode}'")

    print("  Finding unique labels before processing...")
    original_labels_present = np.unique(segmentation_memmap)
    original_labels_present = original_labels_present[original_labels_present != 0]
    print(f"  Found {len(original_labels_present)} unique labels initially.")
    if len(original_labels_present) == 0: return trimmed_pixels_mask # Nothing to process

    # --- Step 0: Create Copy of Original Segmentation (Needed for Healing Reference) ---
    original_segmentation_memmap = None
    original_seg_temp_dir = None
    copy_success = False
    print("  Creating copy of original 2D segmentation for healing...")
    start_copy_time = time.time()
    try:
        original_seg_temp_dir = tempfile.mkdtemp(prefix="orig_seg_copy_2d_")
        original_seg_path = os.path.join(original_seg_temp_dir, 'original_seg_2d.dat')
        original_segmentation_memmap = np.memmap(original_seg_path, dtype=segmentation_memmap.dtype, mode='w+', shape=original_shape)
        # Direct copy for 2D
        original_segmentation_memmap[:] = segmentation_memmap[:]
        original_segmentation_memmap.flush()
        copy_success = True
        print(f"  Original 2D segmentation copied in {time.time() - start_copy_time:.2f}s.")
        gc.collect()
    except Exception as e_copy:
        print(f"\n!!! ERROR creating copy of original 2D segmentation: {e_copy}. Cannot perform healing.")
        copy_success = False
        if original_segmentation_memmap is not None and hasattr(original_segmentation_memmap, '_mmap'): del original_segmentation_memmap; gc.collect(); original_segmentation_memmap = None
        if original_seg_temp_dir and os.path.exists(original_seg_temp_dir): rmtree(original_seg_temp_dir, ignore_errors=True); original_seg_temp_dir = None
        print("    Healing step will be skipped.")

    # --- Step 1: Trimming (2D - No Chunking) ---
    print(f"\n--- 1. Trimming object edges (Dist < {distance_threshold:.2f} AND Brightness < {global_brightness_cutoff:.4f}) ---")
    gc.collect()
    total_trimmed_pixels = 0
    start_trim_time = time.time()
    modified_labels_during_trim = set()

    try:
        # Distance calculation (2D)
        inv_boundary = ~hull_boundary_mask
        print(f"DEBUG trim_object_edges_by_distance_2d: Input 'inv_boundary' shape: {inv_boundary.shape}, ndim: {inv_boundary.ndim}")
        print(f"DEBUG trim_object_edges_by_distance_2d: 'spacing' received by function: {spacing}, type: {type(spacing)}, len: {len(spacing)}")
        edt = distance_transform_edt(inv_boundary, sampling=spacing).astype(np.float32)
        close_to_boundary = (edt < distance_threshold)
        del inv_boundary, edt; gc.collect()

        # Load segmentation data (can work directly on memmap view)
        seg_data_view = segmentation_memmap[:] # Get view/copy of current state

        # Brightness check (2D)
        dim_pixels = (original_image < global_brightness_cutoff)

        # Combine criteria & Apply Trim
        trim_target_mask = (seg_data_view > 0) & close_to_boundary & dim_pixels
        num_trimmed = np.sum(trim_target_mask)

        if num_trimmed > 0:
             unique_trimmed_in_mask = np.unique(seg_data_view[trim_target_mask])
             modified_labels_during_trim.update(unique_trimmed_in_mask[unique_trimmed_in_mask != 0])

             # Apply modification directly to the memmap
             segmentation_memmap[trim_target_mask] = 0
             segmentation_memmap.flush()
             total_trimmed_pixels = num_trimmed
             # Update the tracking mask
             trimmed_pixels_mask[trim_target_mask] = True

        del seg_data_view, close_to_boundary, dim_pixels, trim_target_mask
        gc.collect()
        print(f"  Trimming finished. Pixels trimmed: {total_trimmed_pixels}. Labels affected: {len(modified_labels_during_trim)}.")
        print(f"  Time taken: {time.time() - start_trim_time:.2f}s.")

    except MemoryError as mem_err: print(f"Trimming failed: MemoryError."); raise mem_err
    except Exception as trim_err: print(f"Trimming failed: {trim_err}."); raise trim_err
    gc.collect()

    # --- Step 2: Remove Small Remnants (2D) ---
    num_remnants_removed = 0
    if min_remaining_size > 0:
        print(f"\n--- 2. Removing remnants smaller than {min_remaining_size} pixels ---")
        start_rem_time = time.time()
        print("    Creating boolean mask...")
        if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None:
            raise IOError("Segmentation memmap invalid before remnant removal.")

        try:
            # Load boolean mask into memory for remove_small_objects
            bool_seg_in_memory = segmentation_memmap[:] > 0

            if np.any(bool_seg_in_memory):
                print("    Applying remove_small_objects (skimage)...")
                # Use 2D connectivity
                cleaned_bool_seg = remove_small_objects(bool_seg_in_memory, min_size=min_remaining_size, connectivity=1) # connectivity=1 for 4-neighb
                small_remnant_mask_overall = bool_seg_in_memory & (~cleaned_bool_seg)
                del cleaned_bool_seg; gc.collect()

                num_remnants_to_remove_calc = np.sum(small_remnant_mask_overall)
                if num_remnants_to_remove_calc > 0:
                    print("    Applying removal mask to segmentation memmap...")
                    # Apply mask directly
                    segmentation_memmap[small_remnant_mask_overall] = 0
                    segmentation_memmap.flush()
                    num_remnants_removed = num_remnants_to_remove_calc
                else:
                    print("    No small remnants found to remove.")

                del small_remnant_mask_overall
                gc.collect()
            else:
                print("  Segmentation is empty after trimming. Skipping remnant removal.")

            del bool_seg_in_memory
            gc.collect()
            print(f"  Small object removal finished. Removed {num_remnants_removed} pixels.")

        except MemoryError: print("    MemoryError during small object removal. Skipping."); gc.collect()
        except Exception as e_rem: print(f"    Error during small remnant removal: {e_rem}. Skipping."); gc.collect()
        print(f"  Time taken: {time.time() - start_rem_time:.2f}s.")
    else:
        print(f"\n--- 2. Skipping small object removal (min_remaining_size={min_remaining_size}) ---")

    # --- Step 3: Relabel Disconnected Components (2D) ---
    print("\n--- 3. Checking for and relabeling disconnected components (2D) ---")
    start_relabel_time = time.time()
    relabel_map = {}
    next_available_label = 0
    relabel_count = 0
    try:
        print("    Finding current max label...")
        max_label = np.max(segmentation_memmap[:]) # Read max directly for 2D
        next_available_label = max_label + 1
        print(f"    Max label found: {max_label}. Next new label: {next_available_label}")

        current_labels_present = np.unique(segmentation_memmap)
        current_labels_present = current_labels_present[current_labels_present != 0]
        print(f"    Found {len(current_labels_present)} unique labels to check.")

        for lbl in current_labels_present: relabel_map[lbl] = lbl

        structure_relabel = generate_binary_structure(2, 1) # 2D structure

        for label_val in tqdm(current_labels_present, desc="  Relabeling Components"):
            if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None: raise IOError(f"Seg memmap invalid relabeling {label_val}")
            try:
                # Use find_objects directly on the 2D memmap
                locations = find_objects(segmentation_memmap == label_val)
                if not locations: continue

                # If find_objects returns multiple slices for a single label, it means the object got split
                # The logic handles this correctly by reassigning labels to components found after the first one
                if len(locations) == 1: # Check within the single bounding box if it split
                    loc = locations[0]
                    # Load sub-array into memory
                    seg_sub_array = np.array(segmentation_memmap[loc])
                    label_mask_sub = (seg_sub_array == label_val)
                    if not np.any(label_mask_sub): continue

                    labeled_components_sub, num_components = ndimage_label(label_mask_sub, structure=structure_relabel)

                    if num_components > 1:
                        component_sizes = np.bincount(labeled_components_sub.ravel())[1:] # Ignore background 0
                        if len(component_sizes) == 0: continue # Should not happen if num_components > 1
                        largest_component_idx = np.argmax(component_sizes) + 1 # +1 because bincount indexing starts at 0
                        original_root_label = relabel_map[label_val]

                        # Create a copy to modify and write back
                        seg_sub_array_modified = np.copy(seg_sub_array)
                        changed_in_sub = False
                        for c_idx in range(1, num_components + 1):
                            if c_idx == largest_component_idx: continue
                            component_mask_sub = (labeled_components_sub == c_idx)
                            new_label = next_available_label
                            seg_sub_array_modified[component_mask_sub] = new_label
                            relabel_map[new_label] = original_root_label
                            next_available_label += 1
                            relabel_count += 1
                            changed_in_sub = True
                        # Write back modified sub-array if changes occurred
                        if changed_in_sub:
                            segmentation_memmap[loc] = seg_sub_array_modified
                            segmentation_memmap.flush()

                        del seg_sub_array_modified

                    del seg_sub_array, label_mask_sub, labeled_components_sub
                elif len(locations) > 1: # Object already split across multiple find_objects slices
                    original_root_label = relabel_map[label_val]
                    # Keep the first component with the original label, relabel others
                    for i in range(1, len(locations)):
                        loc = locations[i]
                        seg_sub_array = np.array(segmentation_memmap[loc])
                        component_mask_sub = (seg_sub_array == label_val)
                        if np.any(component_mask_sub):
                            new_label = next_available_label
                            seg_sub_array[component_mask_sub] = new_label # Modify copy
                            # Write back modified sub-array
                            segmentation_memmap[loc] = seg_sub_array
                            segmentation_memmap.flush()
                            relabel_map[new_label] = original_root_label
                            next_available_label += 1
                            relabel_count += 1
                        del seg_sub_array, component_mask_sub
                del locations
            except MemoryError: print(f"MemoryError relabeling {label_val}. Skip."); gc.collect(); continue
            except Exception as rel_err: print(f"Error relabeling {label_val}: {rel_err}. Skip."); gc.collect(); continue

        print(f"  Relabeling finished. Created {relabel_count} new labels.")

    except MemoryError: print("MemoryError during relabeling setup. Skipping."); gc.collect()
    except Exception as e: print(f"Error during relabeling setup: {e}. Skipping."); gc.collect()
    print(f"  Time taken: {time.time() - start_relabel_time:.2f}s.")
    gc.collect()

    # --- Step 4: Heal Porous Edges using Convex Hull (2D) ---
    voxels_restored_count = 0
    if copy_success and original_segmentation_memmap is not None:
        print(f"\n--- 4. Applying 2D Convex Hull Healing ---")
        print(f"  (Healing based on {len(modified_labels_during_trim)} labels initially modified)")
        start_heal_time = time.time()
        failed_labels_heal = 0
        processed_labels_heal = 0

        original_to_current_map = defaultdict(list)
        print("    Building inverse map (original -> current labels)...")
        for current_label, original_root_label in relabel_map.items():
            if original_root_label != 0: original_to_current_map[original_root_label].append(current_label)

        labels_to_heal = list(modified_labels_during_trim)

        for original_label in tqdm(labels_to_heal, desc="  Healing Labels"):
            processed_labels_heal += 1
            try:
                current_labels_for_this_object = original_to_current_map.get(original_label)
                if not current_labels_for_this_object: continue

                if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None: raise IOError("Current seg invalid")
                if not hasattr(original_segmentation_memmap, '_mmap') or original_segmentation_memmap._mmap is None: raise IOError("Original seg invalid")

                # Create mask for current labels directly on 2D memmap view
                combined_mask_for_findobj = np.isin(segmentation_memmap[:], current_labels_for_this_object)

                locations = find_objects(combined_mask_for_findobj)
                del combined_mask_for_findobj; gc.collect()
                if not locations: continue

                # Find global bounding box for all components
                min_coords = [min(s[d].start for s in locations) for d in range(2)]
                max_coords = [max(s[d].stop for s in locations) for d in range(2)]
                global_loc = tuple(slice(min_coords[d], max_coords[d]) for d in range(2))
                del locations; gc.collect()

                # Load sub-arrays into memory
                seg_sub_current_data = np.array(segmentation_memmap[global_loc])
                orig_sub_data = np.array(original_segmentation_memmap[global_loc])

                object_mask_current_sub = np.isin(seg_sub_current_data, current_labels_for_this_object)

                # Skip if object is too small for hull
                if np.sum(object_mask_current_sub) < 3: # Need 3 points for 2D hull
                    del seg_sub_current_data, orig_sub_data, object_mask_current_sub; gc.collect(); continue

                hull_mask_sub = convex_hull_image(object_mask_current_sub)
                restore_mask_sub = hull_mask_sub & (orig_sub_data == original_label)
                currently_zero_mask_sub = (seg_sub_current_data == 0)
                actual_restore_mask_sub = restore_mask_sub & currently_zero_mask_sub
                num_restored_for_label = np.sum(actual_restore_mask_sub)
                voxels_restored_count += num_restored_for_label

                if num_restored_for_label > 0:
                    # Modify the loaded sub-array and write it back
                    seg_sub_current_data[actual_restore_mask_sub] = original_label
                    segmentation_memmap[global_loc] = seg_sub_current_data # Write back
                    segmentation_memmap.flush()

                del global_loc, seg_sub_current_data, orig_sub_data, object_mask_current_sub, hull_mask_sub, restore_mask_sub, currently_zero_mask_sub, actual_restore_mask_sub; gc.collect()

            except MemoryError as mem_err: failed_labels_heal += 1; print(f"MemoryError healing {original_label}. Skip."); gc.collect(); continue
            except ValueError as ve: failed_labels_heal += 1; print(f"ValueError healing {original_label}: {ve}. Skip."); gc.collect(); continue # convex_hull needs >= 3 points
            except Exception as e_heal_label: failed_labels_heal += 1; print(f"ERROR healing {original_label}: {e_heal_label}. Skip."); gc.collect(); continue

        print(f"  2D Convex hull healing finished. Restored approx. {voxels_restored_count} pixels.")
        print(f"  Processed {processed_labels_heal}/{len(labels_to_heal)} labels ({failed_labels_heal} failed).")
        print(f"  Time taken: {time.time() - start_heal_time:.2f}s.")
    else:
        print("\n--- 4. Skipping 2D convex hull healing step (copy failed or unavailable) ---")

    # --- Step 5: Merge Touching Labels (2D) ---
    print("\n--- 5. Merging touching labels (2D) ---")
    start_merge_time = time.time()
    global union_find_parent
    union_find_parent = {}
    touching_pairs = set()
    merged_labels_count = 0
    try:
        print("    Finding unique labels present before merging...")
        labels_after_healing = np.unique(segmentation_memmap)
        labels_after_healing = labels_after_healing[labels_after_healing != 0]
        print(f"    Found {len(labels_after_healing)} unique labels to check.")

        if len(labels_after_healing) > 1:
            for lbl in labels_after_healing: union_find_parent[lbl] = lbl

            # 2D neighbor structure
            neighbor_structure = generate_binary_structure(2, 1) # 4-connectivity

            print("    Identifying touching label pairs (2D)...")
            for label_val in tqdm(labels_after_healing, desc="  Checking Neighbors"):
                try:
                    if not hasattr(segmentation_memmap, '_mmap') or segmentation_memmap._mmap is None: raise IOError(f"Seg memmap invalid check {label_val}")
                    locations = find_objects(segmentation_memmap == label_val)
                    if not locations: continue

                    for loc in locations:
                        seg_sub_view = segmentation_memmap[loc]
                        is_current_label_sub = (seg_sub_view == label_val)
                        # 2D dilation
                        dilated_mask_sub = binary_dilation(is_current_label_sub, structure=neighbor_structure)
                        neighbor_mask_sub = dilated_mask_sub & ~is_current_label_sub

                        if np.any(neighbor_mask_sub):
                            neighbor_labels = seg_sub_view[neighbor_mask_sub]
                            unique_neighbor_labels = np.unique(neighbor_labels[neighbor_labels != 0])
                            for neighbor_label in unique_neighbor_labels:
                                if neighbor_label in union_find_parent:
                                    pair = tuple(sorted((label_val, neighbor_label)))
                                    if pair not in touching_pairs:
                                        touching_pairs.add(pair)
                                        unite_sets(label_val, neighbor_label)

                        del seg_sub_view, is_current_label_sub, dilated_mask_sub, neighbor_mask_sub
                        if 'neighbor_labels' in locals(): del neighbor_labels
                        if 'unique_neighbor_labels' in locals(): del unique_neighbor_labels
                        gc.collect()
                except MemoryError: print(f"MemoryError check neighbor {label_val}. Skip."); gc.collect(); continue
                except Exception as err_touch: print(f"Error check neighbor {label_val}: {err_touch}. Skip."); gc.collect(); continue

            print(f"    Found {len(touching_pairs)} touching pairs.")

            # --- Apply the merge mapping (2D - No Chunking) ---
            print("    Applying merge map to 2D segmentation...")
            merge_map_final = {lbl: find_set(lbl) for lbl in labels_after_healing}
            num_labels_to_change = sum(1 for old, new in merge_map_final.items() if old != new)
            merged_labels_count = num_labels_to_change

            if num_labels_to_change > 0:
                # Read whole 2D array, apply map, write back
                seg_array = np.array(segmentation_memmap[:])
                seg_array_modified = np.copy(seg_array)
                unique_labels_in_array = np.unique(seg_array)

                labels_changed = False
                for old_label in unique_labels_in_array:
                    if old_label == 0: continue
                    final_label = merge_map_final.get(old_label, old_label)
                    if final_label != old_label:
                        seg_array_modified[seg_array == old_label] = final_label
                        labels_changed = True

                if labels_changed:
                    segmentation_memmap[:] = seg_array_modified # Write back modified array
                    segmentation_memmap.flush()
                    print(f"    Merge applied. {merged_labels_count} labels merged.")
                else:
                    print("    No labels needed merging (internal check).") # Should align with num_labels_to_change check

                del seg_array, seg_array_modified, unique_labels_in_array
                gc.collect()
            else:
                print("    No labels needed merging.")
        else:
            print("    Skipping merge check: Only one label present.")

    except MemoryError: print("MemoryError during 2D merge. Incomplete."); gc.collect()
    except Exception as e_merge: print(f"Error during 2D merge: {e_merge}. Incomplete."); gc.collect()

    print(f"  Time taken for merging: {time.time() - start_merge_time:.2f}s.")
    gc.collect()

    # --- Final Cleanup ---
    print("\n--- Cleaning up 2D temporary files ---")
    if original_segmentation_memmap is not None and hasattr(original_segmentation_memmap, '_mmap'): del original_segmentation_memmap; gc.collect()
    if original_seg_temp_dir and os.path.exists(original_seg_temp_dir): rmtree(original_seg_temp_dir, ignore_errors=True)
    gc.collect()

    print("\n--- 2D Processing sequence finished ---")
    return trimmed_pixels_mask # Return mask of pixels *initially* trimmed

# --- END OF FILE utils/ramified_module_2d/remove_artifacts_2d.py ---