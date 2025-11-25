# --- START OF FILE utils/module_3d/remove_artifacts.py ---
import numpy as np
from scipy.ndimage import generate_binary_structure
from scipy.ndimage import distance_transform_edt as scipy_edt 
from skimage.filters import threshold_otsu # type: ignore
from skimage.morphology import convex_hull_image # type: ignore
import traceback
import tempfile
import os
from tqdm import tqdm
from shutil import rmtree, copyfile
import gc
import math

# Dask imports for out-of-core 3D processing
import dask.array as da
from dask.diagnostics import ProgressBar
import dask_image.ndmorph
import dask_image.ndmeasure
import zarr

# Dask configuration
# 'threads' is safest for local file IO.
DASK_SCHEDULER = 'threads'

def _safe_close_memmap(memmap_obj):
    """
    Helper to flush and close memmap to ensure data is written to disk.
    Returns None so the caller can assign the result to the variable,
    releasing the reference.
    """
    if memmap_obj is None:
        return None
    try:
        if hasattr(memmap_obj, 'flush'):
            memmap_obj.flush()
        if hasattr(memmap_obj, '_mmap') and memmap_obj._mmap is not None:
            memmap_obj._mmap.close()
    except Exception:
        pass
    return None

def generate_hull_boundary_and_stack(
    volume,
    cell_mask,
    parent_temp_dir,
    hull_erosion_iterations=1,
    smoothing_iterations=1
    ):
    """
    Generates a smooth 3D Convex Hull around the tissue and extracts the boundary layer.
    """
    print("\n  [HullGen] Generating Smoothed Hull Boundary...")
    original_shape = volume.shape
    
    internal_temp_dir = tempfile.mkdtemp(prefix="hullgen_internal_", dir=parent_temp_dir)
    smoothed_hull_path = os.path.join(parent_temp_dir, 'smoothed_hull.dat')
    boundary_path = os.path.join(parent_temp_dir, 'boundary.dat') if hull_erosion_iterations > 0 else None
    
    # Initialize variables to None so 'finally' block can check them safely
    initial_hull_stack = None
    smoothed_hull_stack = None
    boundary_memmap = None
    
    try:
        # --- Step 1: Create Initial Hull (Slice-by-Slice) ---
        print("    [1/4] Creating initial 2D hull stack...")
        initial_hull_path = os.path.join(internal_temp_dir, 'initial_hull.dat')
        initial_hull_stack = np.memmap(initial_hull_path, dtype=bool, mode='w+', shape=original_shape)
        
        # Calculate Otsu threshold
        otsu_samples = volume[::100].ravel() # Subsample every 100th pixel for speed
        valid_samples = otsu_samples[otsu_samples > 0]
        
        if valid_samples.size > 0:
            tissue_thresh = threshold_otsu(valid_samples)
            print(f"      Tissue Threshold: {tissue_thresh:.2f}")
        else:
            tissue_thresh = 0
            print("      Warning: Volume appears empty (all zeros). Threshold set to 0.")
        
        del otsu_samples, valid_samples

        # Process slices
        nonzero_slices = 0
        for z in tqdm(range(original_shape[0]), desc="    Calculating 2D Hulls"):
            combined_mask_slice = (volume[z] > tissue_thresh) | (cell_mask[z] > 0)
            if np.any(combined_mask_slice):
                try:
                    initial_hull_stack[z] = convex_hull_image(np.ascontiguousarray(combined_mask_slice))
                    nonzero_slices += 1
                except Exception as e:
                    # convex_hull_image can fail on weird inputs (e.g. single point)
                    pass
        
        # Close memmap to force write to disk before Dask reads it
        # CRITICAL FIX: Do NOT use 'del' here. The assignment to None releases the object.
        initial_hull_stack = _safe_close_memmap(initial_hull_stack)
        
        # Verify data exists
        check_hull = np.memmap(initial_hull_path, dtype=bool, mode='r', shape=original_shape)
        total_hull_pixels = np.count_nonzero(check_hull)
        print(f"      Initial Hull Pixels: {total_hull_pixels} (in {nonzero_slices} slices)")
        del check_hull

        if total_hull_pixels == 0:
             print("    Warning: Initial hull stack is empty. Cannot proceed.")
             return None, None

        # --- Step 2: Smooth Hull (Dask 3D) ---
        print(f"    [2/4] Smoothing 3D hull ({smoothing_iterations} iter) with Dask...")
        
        # Re-open as Read-Only for Dask
        initial_hull_input = np.memmap(initial_hull_path, dtype=bool, mode='r', shape=original_shape)
        
        dask_chunk_size = (32, 256, 256)
        hull_dask_array = da.from_array(initial_hull_input, chunks=dask_chunk_size)
        
        struct_3d = generate_binary_structure(3, 1)
        
        if smoothing_iterations > 0:
            hull_dask_array = dask_image.ndmorph.binary_closing(hull_dask_array, structure=struct_3d, iterations=smoothing_iterations)
            hull_dask_array = dask_image.ndmorph.binary_opening(hull_dask_array, structure=struct_3d, iterations=smoothing_iterations)

        temp_zarr_path = os.path.join(internal_temp_dir, 'temp_smoothed.zarr')
        print("      Writing smoothed hull to Zarr...")
        with ProgressBar(dt=2):
            hull_dask_array.to_zarr(temp_zarr_path, overwrite=True)
            
        # Close input source
        del hull_dask_array, initial_hull_input
        gc.collect()
        
        # Write Zarr -> Memmap
        smoothed_hull_stack = np.memmap(smoothed_hull_path, dtype=bool, mode='w+', shape=original_shape)
        zarr_array = zarr.open(temp_zarr_path, mode='r')
        
        # Efficient copy
        batch_size = 10
        for i in tqdm(range(0, original_shape[0], batch_size), desc="    Exporting smoothed hull"):
            end = min(i + batch_size, original_shape[0])
            smoothed_hull_stack[i:end] = zarr_array[i:end]
            
        smoothed_hull_stack = _safe_close_memmap(smoothed_hull_stack)
        
        # Verify
        check_smooth = np.memmap(smoothed_hull_path, dtype=bool, mode='r', shape=original_shape)
        print(f"      Smoothed Hull Pixels: {np.count_nonzero(check_smooth)}")
        del check_smooth
        
        # --- Step 3: Calculate Boundary (Erosion) ---
        if hull_erosion_iterations > 0:
            print(f"    [3/4] Calculating boundary zone ({hull_erosion_iterations} iter erosion)...")
            
            smoothed_dask = da.from_zarr(temp_zarr_path)
            
            eroded_hull_dask = dask_image.ndmorph.binary_erosion(smoothed_dask, structure=struct_3d, iterations=hull_erosion_iterations)
            
            # Boundary = Hull AND (NOT ErodedHull)
            boundary_dask = smoothed_dask & (~eroded_hull_dask)
            
            boundary_zarr_path = os.path.join(internal_temp_dir, 'boundary.zarr')
            print("      Writing boundary to Zarr...")
            with ProgressBar(dt=2):
                boundary_dask.to_zarr(boundary_zarr_path, overwrite=True)
            
            del smoothed_dask, eroded_hull_dask, boundary_dask
            gc.collect()
            
            boundary_memmap = np.memmap(boundary_path, dtype=bool, mode='w+', shape=original_shape)
            boundary_zarr = zarr.open(boundary_zarr_path, mode='r')
            
            for i in tqdm(range(0, original_shape[0], batch_size), desc="    Exporting boundary"):
                end = min(i + batch_size, original_shape[0])
                boundary_memmap[i:end] = boundary_zarr[i:end]
                
            boundary_memmap = _safe_close_memmap(boundary_memmap)
            
            # Verify
            check_bound = np.memmap(boundary_path, dtype=bool, mode='r', shape=original_shape)
            print(f"      Boundary Pixels: {np.count_nonzero(check_bound)}")
            del check_bound
        
        return boundary_path, smoothed_hull_path

    finally:
        # Extra cleanup safety
        # Check if variable exists AND is not None before closing/deleting
        if 'initial_hull_stack' in locals() and initial_hull_stack is not None:
             del initial_hull_stack
        if 'smoothed_hull_stack' in locals() and smoothed_hull_stack is not None:
             del smoothed_hull_stack
        if 'boundary_memmap' in locals() and boundary_memmap is not None:
             del boundary_memmap
        
        gc.collect()
        
        if internal_temp_dir and os.path.exists(internal_temp_dir):
            try: rmtree(internal_temp_dir, ignore_errors=True)
            except: pass


def trim_object_edges_by_distance(
    segmentation_memmap, original_volume, hull_boundary_path, spacing,
    parent_temp_dir, distance_threshold, global_brightness_cutoff, min_remaining_size=10
    ):
    """
    Removes segmented objects that are close to the hull boundary.
    """
    print("\n  [EdgeTrim] Trimming Object Edges by Distance...")
    original_shape = segmentation_memmap.shape
    trim_temp_dir = tempfile.mkdtemp(prefix="trim_stages_", dir=parent_temp_dir)
    hull_boundary_memmap = None
    
    try:
        # --- Stage A: Distance Transform ---
        print(f"    [A] Calculating distance transform from boundary...")
        dask_chunk_size = (64, 256, 256)
        
        # Read-Only access to boundary
        hull_boundary_memmap = np.memmap(hull_boundary_path, dtype=bool, mode='r', shape=original_shape)
        boundary_dask = da.from_array(hull_boundary_memmap, chunks=dask_chunk_size)
        
        def block_edt(block, spacing):
            return scipy_edt(~block, sampling=spacing).astype(np.float32)

        edt_dask = boundary_dask.map_blocks(
            block_edt,
            spacing=spacing,
            dtype=np.float32
        )
        
        edt_path = os.path.join(trim_temp_dir, 'edt.zarr')
        print("      Saving EDT to Zarr...")
        with ProgressBar(dt=2):
            edt_dask.to_zarr(edt_path, overwrite=True)
        
        # Cleanup Dask graphs
        del boundary_dask, edt_dask
        # Close the memmap file explicitly
        hull_boundary_memmap = _safe_close_memmap(hull_boundary_memmap)
        gc.collect()
        
        # --- Stage B: Apply Trimming Mask ---
        print(f"    [B] Applying trim mask...")
        edt_zarr = zarr.open(edt_path, mode='r')
        edt_dask_from_zarr = da.from_zarr(edt_zarr)
        
        # Ensure segmentation map is flushed before reading into Dask
        segmentation_memmap.flush()
        
        seg_dask = da.from_array(segmentation_memmap, chunks=dask_chunk_size)
        vol_dask = da.from_array(original_volume, chunks=dask_chunk_size)
        
        trim_mask_dask = ( 
            (seg_dask > 0) & 
            (edt_dask_from_zarr < distance_threshold) & 
            (vol_dask < global_brightness_cutoff) 
        )
        
        trimmed_seg_dask = da.where(trim_mask_dask, 0, seg_dask)
        
        trimmed_path = os.path.join(trim_temp_dir, 'trimmed_seg.zarr')
        print("      Saving trimmed segmentation to Zarr...")
        with ProgressBar(dt=2):
            trimmed_seg_dask.to_zarr(trimmed_path, overwrite=True)
            
        del edt_zarr, edt_dask_from_zarr, seg_dask, vol_dask, trim_mask_dask, trimmed_seg_dask
        gc.collect()
        
        # --- Stage C: Clean Fragments ---
        print(f"    [C] Cleaning fragments and re-labeling...")
        trimmed_zarr = zarr.open(trimmed_path, mode='r')
        trimmed_dask_from_zarr = da.from_zarr(trimmed_zarr)
        
        s = generate_binary_structure(3, 1)
        labeled_array, num_features = dask_image.ndmeasure.label(trimmed_dask_from_zarr > 0, structure=s)
        
        with ProgressBar(dt=1):
            num_features_val = num_features.compute(scheduler=DASK_SCHEDULER)
        
        final_labeled_dask = None
        if num_features_val > 0:
            print(f"      Found {num_features_val} objects. Filtering small ones...")
            with ProgressBar(dt=1):
                object_sizes, _ = da.histogram(labeled_array, bins=np.arange(num_features_val + 2))
                object_sizes_val = object_sizes.compute(scheduler=DASK_SCHEDULER)
            
            large_labels = np.where(object_sizes_val[1:] >= min_remaining_size)[0] + 1
            
            if len(large_labels) < num_features_val:
                print(f"      Removing {num_features_val - len(large_labels)} small objects...")
                keep_mask_dask = da.isin(labeled_array, large_labels)
                final_labeled_dask, _ = dask_image.ndmeasure.label(keep_mask_dask, structure=s)
            else:
                final_labeled_dask = labeled_array
        else:
            final_labeled_dask = da.zeros_like(trimmed_dask_from_zarr, dtype=np.int32)
        
        # Write back to segmentation_memmap (inplace)
        print("      Writing final trimmed result to output file...")
        with ProgressBar(dt=2):
            da.store(final_labeled_dask, segmentation_memmap, scheduler=DASK_SCHEDULER)
            
        segmentation_memmap.flush()
        print(f"    [EdgeTrim] Done.")

    finally:
        if 'hull_boundary_memmap' in locals() and hull_boundary_memmap is not None:
            del hull_boundary_memmap
        gc.collect()
        
        if trim_temp_dir and os.path.exists(trim_temp_dir):
            try: rmtree(trim_temp_dir, ignore_errors=True)
            except: pass


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
    Main Entry Point for Step 2.
    """
    print(f"\n--- Applying Hull Generation and Edge Trimming ---")
    
    original_shape = original_volume.shape
    
    # Create a robust temp directory
    workflow_temp_dir = tempfile.mkdtemp(prefix="hull_trim_workflow_")
    final_output_temp_dir = None
    
    trimmed_labels_memmap = None
    hull_boundary_for_return = None
    
    try:
        # 1. Setup Output File
        final_output_temp_dir = tempfile.mkdtemp(prefix="trimmed_labels_")
        final_output_path = os.path.join(final_output_temp_dir, 'trimmed_labels.dat')
        
        copyfile(raw_labels_path, final_output_path)
        
        # Open as Read/Write
        trimmed_labels_memmap = np.memmap(final_output_path, dtype=np.int32, mode='r+', shape=original_shape)

        # 2. Generate Hull
        min_spacing_val = min(s for s in spacing if s > 1e-9)
        erosion_iterations = math.ceil(hull_boundary_thickness_phys / min_spacing_val) if hull_boundary_thickness_phys > 0 else 0

        hull_boundary_path, smoothed_hull_path = generate_hull_boundary_and_stack(
            volume=original_volume,
            cell_mask=trimmed_labels_memmap,
            parent_temp_dir=workflow_temp_dir,
            hull_erosion_iterations=erosion_iterations,
            smoothing_iterations=smoothing_iterations
        )
        
        # 3. Trim Edges
        if hull_boundary_path and os.path.exists(hull_boundary_path) and edge_trim_distance_threshold > 0:
            
            global_brightness_cutoff = segmentation_threshold * brightness_cutoff_factor
            print(f"  Edge Trim: DistThresh={edge_trim_distance_threshold}, BrightCutoff={global_brightness_cutoff:.4f}")
            
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
            
            # Load boundary for return (Read-Only)
            temp_bound = np.memmap(hull_boundary_path, dtype=bool, mode='r', shape=original_shape)
            hull_boundary_for_return = np.array(temp_bound)
            del temp_bound
            
        else:
            print("  Skipping edge trimming (Threshold=0 or Hull failed).")
            hull_boundary_for_return = np.zeros(original_shape, dtype=bool)
        
        # Clean close output file
        trimmed_labels_memmap = _safe_close_memmap(trimmed_labels_memmap)
        
        return final_output_path, final_output_temp_dir, hull_boundary_for_return

    except Exception as e:
        print(f"\n!!! ERROR during Hull Trimming Workflow: {e} !!!")
        traceback.print_exc()
        if final_output_temp_dir and os.path.exists(final_output_temp_dir):
            rmtree(final_output_temp_dir, ignore_errors=True)
        return None, None, None
        
    finally:
        # Ensure everything is closed
        if 'trimmed_labels_memmap' in locals():
            trimmed_labels_memmap = _safe_close_memmap(trimmed_labels_memmap)
        gc.collect()
        
        # Cleanup temp dir
        if workflow_temp_dir and os.path.exists(workflow_temp_dir):
            try: rmtree(workflow_temp_dir, ignore_errors=True)
            except: pass
# --- END OF FILE utils/module_3d/remove_artifacts.py ---