# --- START OF FILE remove_artifacts.py ---

import numpy as np
from scipy import ndimage
from scipy.ndimage import (distance_transform_edt,
                           generate_binary_structure)
from scipy.ndimage import (label as ndimage_label, find_objects)
from skimage.filters import threshold_otsu # type: ignore
from skimage.morphology import convex_hull_image, remove_small_objects as skimage_remove_small_objects # type: ignore
import traceback
import tempfile
import os
from tqdm import tqdm
import psutil
from shutil import rmtree, copyfile
import gc
import math
import dask.array as da
from dask.diagnostics import ProgressBar
import dask_image.ndmorph
import dask_image.ndmeasure
import zarr

DASK_SCHEDULER = 'threads'

def generate_hull_boundary_and_stack(
    volume,
    cell_mask,
    parent_temp_dir,
    hull_erosion_iterations=1,
    smoothing_iterations=1
    ):
    """
    MEMORY-SAFE version using the robust Dask -> Zarr -> Memmap pattern.
    """
    print("\n--- Generating Smoothed Hull Boundary and Stack (Memory-Safe) ---")
    original_shape = volume.shape
    internal_temp_dir = tempfile.mkdtemp(prefix="hullgen_internal_", dir=parent_temp_dir)
    
    # Define file paths in the outer scope for clarity
    smoothed_hull_path = os.path.join(parent_temp_dir, 'smoothed_hull.dat')
    boundary_path = os.path.join(parent_temp_dir, 'boundary.dat') if hull_erosion_iterations > 0 else None
    
    try:
        # Step 1 & 2: Create initial hull stack on disk
        print("Step 1 & 2: Creating initial hull stack...")
        initial_hull_path = os.path.join(internal_temp_dir, 'initial_hull.dat')
        initial_hull_stack = np.memmap(initial_hull_path, dtype=bool, mode='w+', shape=original_shape)
        otsu_sample_size = min(2_000_000, volume.size)
        otsu_samples = volume.ravel() if otsu_sample_size == volume.size else np.random.choice(volume.ravel(), otsu_sample_size, replace=False)
        tissue_thresh = threshold_otsu(otsu_samples) if np.any(otsu_samples) else 0
        del otsu_samples; gc.collect()
        for z in tqdm(range(original_shape[0]), desc="  Processing Slices for Hull"):
            combined_mask_slice = (volume[z] > tissue_thresh) | (cell_mask[z] > 0)
            if np.any(combined_mask_slice):
                initial_hull_stack[z] = convex_hull_image(np.ascontiguousarray(combined_mask_slice))
        initial_hull_stack.flush()
        if not np.any(initial_hull_stack):
             print("  Warning: Initial hull stack is empty. Cannot proceed."); return None, None

        # Step 3: Smooth the hull stack using Dask
        dask_chunk_size = (32, 256, 256)
        hull_dask_array = da.from_array(initial_hull_stack, chunks=dask_chunk_size)
        struct_3d = generate_binary_structure(3, 1)
        if smoothing_iterations > 0:
            print(f"\nStep 3: Smoothing hull stack with Dask ({smoothing_iterations} iterations)...")
            hull_dask_array = dask_image.ndmorph.binary_closing(hull_dask_array, structure=struct_3d, iterations=smoothing_iterations)
            hull_dask_array = dask_image.ndmorph.binary_opening(hull_dask_array, structure=struct_3d, iterations=smoothing_iterations)

        temp_zarr_path = os.path.join(internal_temp_dir, 'temp.zarr')
        
        print("  Executing Dask smoothing and saving to temporary Zarr store...")
        with ProgressBar(dt=2):
            hull_dask_array.to_zarr(temp_zarr_path, overwrite=True)
        
        smoothed_hull_stack = np.memmap(smoothed_hull_path, dtype=bool, mode='w+', shape=original_shape)
        zarr_array = zarr.open(temp_zarr_path, mode='r')
        for i in tqdm(range(original_shape[0]), desc="  Copying smoothed hull"):
            smoothed_hull_stack[i] = zarr_array[i]
        smoothed_hull_stack.flush()
        
        # Step 4 & 5: Calculate boundary and save to disk
        if hull_erosion_iterations > 0:
            print(f"\nStep 4 & 5: Calculating boundary with Dask ({hull_erosion_iterations} iterations)...")
            boundary_memmap = np.memmap(boundary_path, dtype=bool, mode='w+', shape=original_shape)
            eroded_hull_dask = dask_image.ndmorph.binary_erosion(hull_dask_array, structure=struct_3d, iterations=hull_erosion_iterations)
            boundary_dask = hull_dask_array & (~eroded_hull_dask)
            
            boundary_zarr_path = os.path.join(internal_temp_dir, 'boundary.zarr')
            print("  Executing Dask boundary calculation and saving to temporary Zarr store...")
            with ProgressBar(dt=2):
                boundary_dask.to_zarr(boundary_zarr_path, overwrite=True)
            
            print("  Copying boundary from Zarr to memmap...")
            boundary_zarr = zarr.open(boundary_zarr_path, mode='r')
            for i in tqdm(range(original_shape[0]), desc="  Copying boundary"):
                boundary_memmap[i] = boundary_zarr[i]
            boundary_memmap.flush()
        
        print("--- Hull generation finished ---")
        return boundary_path, smoothed_hull_path

    finally:
        print("\n  Cleaning up internal hull generation files...")
        if 'initial_hull_stack' in locals(): del initial_hull_stack
        if 'smoothed_hull_stack' in locals(): del smoothed_hull_stack
        if 'boundary_memmap' in locals(): del boundary_memmap
        gc.collect()
        if 'internal_temp_dir' in locals() and locals()['internal_temp_dir'] and os.path.exists(locals()['internal_temp_dir']):
            rmtree(locals()['internal_temp_dir'], ignore_errors=True)


def trim_object_edges_by_distance(
    segmentation_memmap, original_volume, hull_boundary_path, spacing,
    parent_temp_dir, distance_threshold, global_brightness_cutoff, min_remaining_size=10
    ):
    """
    MEMORY-SAFE version using staged Dask graph with checkpoints in a specified directory.
    """
    print("\n--- Trimming Object Edges by Distance (Staged Dask Strategy) ---")
    original_shape = segmentation_memmap.shape
    
    trim_temp_dir = tempfile.mkdtemp(prefix="trim_stages_", dir=parent_temp_dir)
    hull_boundary_memmap = None
    
    try:
        # --- Stage A: Distance Transform ---
        print(f"  Stage A: Calculating distance transform...")
        dask_chunk_size = (64, 256, 256)
        
        hull_boundary_memmap = np.memmap(hull_boundary_path, dtype=bool, mode='r', shape=original_shape)
        boundary_dask = da.from_array(hull_boundary_memmap, chunks=dask_chunk_size)
        
        edt_dask = boundary_dask.map_blocks(
            lambda block: distance_transform_edt(~block, sampling=spacing),
            dtype=np.float32
        )
        
        edt_path = os.path.join(trim_temp_dir, 'edt.zarr')
        print("    Executing and saving distance transform to checkpoint...")
        with ProgressBar(dt=2):
            edt_dask.to_zarr(edt_path, overwrite=True)
        del boundary_dask, edt_dask; gc.collect()
        
        # --- Stage B: Apply Trimming ---
        print(f"\n  Stage B: Applying trimming criteria...")
        edt_zarr = zarr.open(edt_path, mode='r')
        edt_dask_from_zarr = da.from_zarr(edt_zarr)
        seg_dask = da.from_array(segmentation_memmap, chunks=dask_chunk_size)
        vol_dask = da.from_array(original_volume, chunks=dask_chunk_size)
        
        trim_mask_dask = ( (seg_dask > 0) & (edt_dask_from_zarr < distance_threshold) & (vol_dask < global_brightness_cutoff) )
        trimmed_seg_dask = da.where(trim_mask_dask, 0, seg_dask)
        
        trimmed_path = os.path.join(trim_temp_dir, 'trimmed_seg.zarr')
        print("    Executing trimming and saving to checkpoint...")
        with ProgressBar(dt=2):
            trimmed_seg_dask.to_zarr(trimmed_path, overwrite=True)
        del edt_zarr, edt_dask_from_zarr, seg_dask, vol_dask, trim_mask_dask, trimmed_seg_dask
        gc.collect()
        
        # --- Stage C: Final Cleaning and Relabeling ---
        print(f"\n  Stage C: Re-labeling and cleaning fragmented objects...")
        trimmed_zarr = zarr.open(trimmed_path, mode='r')
        trimmed_dask_from_zarr = da.from_zarr(trimmed_zarr)
        
        s = generate_binary_structure(3, 1)
        labeled_array, num_features = dask_image.ndmeasure.label(trimmed_dask_from_zarr > 0, structure=s)
        
        with ProgressBar(dt=1):
            num_features_val = num_features.compute(scheduler=DASK_SCHEDULER)
        
        if num_features_val > 0:
            with ProgressBar(dt=1):
                object_sizes, _ = da.histogram(labeled_array, bins=np.arange(num_features_val + 2))
                object_sizes_val = object_sizes.compute(scheduler=DASK_SCHEDULER)
            
            large_labels = np.where(object_sizes_val[1:] >= min_remaining_size)[0] + 1
            print(f"    Found {len(large_labels)} objects large enough to keep.")
            
            if len(large_labels) < num_features_val:
                keep_mask_dask = da.isin(labeled_array, large_labels)
                final_labeled_dask, _ = dask_image.ndmeasure.label(keep_mask_dask, structure=s)
            else:
                print("    No small objects were removed."); final_labeled_dask = labeled_array
        else:
            print("    No objects remaining after trimming."); final_labeled_dask = da.zeros_like(trimmed_dask_from_zarr, dtype=np.int32)
        
        print("    Executing final cleanup and writing to output file...")
        with ProgressBar(dt=2):
            da.store(final_labeled_dask, segmentation_memmap, scheduler=DASK_SCHEDULER)
        segmentation_memmap.flush()

    finally:
        print("\n  Cleaning up trimming temporary files...")
        if hull_boundary_memmap is not None: del hull_boundary_memmap
        gc.collect()
        if 'trim_temp_dir' in locals() and locals()['trim_temp_dir'] and os.path.exists(locals()['trim_temp_dir']):
            rmtree(locals()['trim_temp_dir'], ignore_errors=True)
        print("--- Edge trimming process finished ---")


# +++ THIS IS THE MAIN ENTRY POINT WITH AUTOMATIC TEMP DIR MANAGEMENT +++
def apply_hull_trimming(
    raw_labels_path,
    original_volume,
    spacing,
    hull_boundary_thickness_phys,
    edge_trim_distance_threshold,
    brightness_cutoff_factor,
    segmentation_threshold,
    min_size_voxels,
    smoothing_iterations=1,
    heal_iterations=1,
    edge_distance_chunk_size_z=32
    ):
    """
    Main workflow for applying memory-safe hull generation and edge trimming.
    """
    print(f"\n--- Applying Hull Generation and Edge Trimming (Memory-Safe Workflow) ---")
    
    if heal_iterations != 1:
        print(f"  WARNING: 'heal_iterations' parameter is deprecated and will be ignored.")

    original_shape = original_volume.shape
    
    # +++ CRITICAL FIX: Create one main temporary directory for all intermediates
    # in a safe location (next to the code) that will be robustly cleaned up. +++
    script_dir = os.path.dirname(os.path.realpath(__file__))
    workflow_temp_dir = tempfile.mkdtemp(prefix="hull_trim_workflow_", dir=script_dir)
    print(f"  Using main temporary directory for intermediates: {workflow_temp_dir}")
    
    final_output_temp_dir = None
    
    try:
        # The final output is created in the system's default temp location (e.g., /tmp)
        # to ensure it is not deleted by our main cleanup block.
        final_output_temp_dir = tempfile.mkdtemp(prefix="trimmed_labels_")
        final_output_path = os.path.join(final_output_temp_dir, 'trimmed_labels.dat')
        
        print(f"  Robustly copying raw labels to new output location...")
        copyfile(raw_labels_path, final_output_path)
        print("  Copying complete.")
        
        trimmed_labels_memmap = np.memmap(final_output_path, dtype=np.int32, mode='r+', shape=original_shape)

        min_spacing_val = min(s for s in spacing if s > 1e-9)
        erosion_iterations = math.ceil(hull_boundary_thickness_phys / min_spacing_val) if hull_boundary_thickness_phys > 0 else 0

        # Pass the main workflow temp dir to the generator function
        hull_boundary_path, smoothed_hull_path = generate_hull_boundary_and_stack(
            volume=original_volume,
            cell_mask=trimmed_labels_memmap,
            parent_temp_dir=workflow_temp_dir,
            hull_erosion_iterations=erosion_iterations,
            smoothing_iterations=smoothing_iterations
        )
        
        hull_boundary_for_return = np.zeros(original_shape, dtype=bool)
        if hull_boundary_path and os.path.exists(hull_boundary_path) and edge_trim_distance_threshold > 0:
            global_brightness_cutoff = segmentation_threshold * brightness_cutoff_factor
            
            trim_object_edges_by_distance(
                segmentation_memmap=trimmed_labels_memmap,
                original_volume=original_volume,
                hull_boundary_path=hull_boundary_path,
                spacing=spacing,
                parent_temp_dir=workflow_temp_dir,
                distance_threshold=edge_trim_distance_threshold,
                global_brightness_cutoff=global_brightness_cutoff,
                min_remaining_size=min_size_voxels
            )
            # Load the final boundary into memory for returning, as required by the calling function.
            hull_boundary_for_return = np.array(np.memmap(hull_boundary_path, dtype=bool, mode='r', shape=original_shape))
        else:
            print("  Skipping edge trimming as no hull boundary was generated or distance_threshold is zero.")
        
        # On success, we return the path to the final output, and the CALLER is responsible for it.
        # The intermediate workflow_temp_dir will be cleaned up by the finally block.
        return final_output_path, final_output_temp_dir, hull_boundary_for_return

    except Exception as e:
        print(f"\n!!! ERROR during Hull Trimming Workflow: {e} !!!"); traceback.print_exc()
        # If we failed, we DO want to clean up the final output dir we created.
        if final_output_temp_dir and os.path.exists(final_output_temp_dir):
            rmtree(final_output_temp_dir, ignore_errors=True)
        return None, None, None
        
    finally:
        print("\n  Final cleanup for hull trimming workflow...")
        if 'trimmed_labels_memmap' in locals() and locals()['trimmed_labels_memmap'] is not None: del trimmed_labels_memmap
        gc.collect()
        # This single block now robustly cleans up ALL intermediate files.
        if 'workflow_temp_dir' in locals() and locals()['workflow_temp_dir'] and os.path.exists(locals()['workflow_temp_dir']):
            print(f"  Removing main temporary directory: {workflow_temp_dir}")
            rmtree(locals()['workflow_temp_dir'], ignore_errors=True)