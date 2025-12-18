import os
import gc
import math
import time
import shutil
import tempfile
import traceback
import multiprocessing as mp
from functools import partial
from typing import Tuple, List, Dict, Any, Optional, Union

import numpy as np
import zarr
import dask.array as da
import dask_image.ndmorph
import dask_image.ndmeasure
from dask.diagnostics import ProgressBar
from scipy import ndimage
from scipy.ndimage import gaussian_filter, generate_binary_structure, white_tophat
from skimage.filters import frangi, sato  # type: ignore
from tqdm import tqdm

# Set fixed seed for reproducibility
SEED = 42
np.random.seed(SEED)


def _init_worker() -> None:
    """Initializes worker processes to use single-threaded BLAS/OMP."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"


def _process_slice_worker(
    z: int,
    input_memmap_info: Tuple[str, Tuple[int, ...], Any],
    output_memmap_info: Tuple[str, Tuple[int, ...], Any],
    sigmas_voxel_2d: List[float],
    black_ridges: bool,
    frangi_alpha: float,
    frangi_beta: float,
    frangi_gamma: float,
    soma_preservation_factor: float,
    subtract_background_radius: int
) -> Optional[str]:
    """
    Worker function for parallel slice-by-slice 2D filtering.

    Args:
        z: Z-index to process.
        input_memmap_info: (filename, shape, dtype) for input volume.
        output_memmap_info: (filename, shape, dtype) for output volume.
        sigmas_voxel_2d: List of scales in voxel units.
        black_ridges: Filter parameter.
        frangi_alpha, frangi_beta, frangi_gamma: Frangi parameters.
        soma_preservation_factor: Factor to preserve raw intensity (somas).
        subtract_background_radius: Radius for top-hat background subtraction.

    Returns:
        None if successful, error string otherwise.
    """
    input_memmap = None
    output_memmap = None
    try:
        # Unpack connection info
        input_path, input_shape, input_dtype = input_memmap_info
        output_path, output_shape, output_dtype = output_memmap_info

        # Open memmaps
        input_memmap = np.memmap(
            input_path, dtype=input_dtype, mode='r', shape=input_shape
        )
        output_memmap = np.memmap(
            output_path, dtype=output_dtype, mode='r+', shape=output_shape
        )

        # Load data (float32)
        slice_data = input_memmap[z].astype(np.float32, copy=True)

        # 0. Background Subtraction
        if subtract_background_radius > 0:
            struct_size = int(2 * subtract_background_radius + 1)
            slice_data = white_tophat(slice_data, size=struct_size)

        # 1. Apply Vesselness Filters
        frangi_result_2d = frangi(
            slice_data, sigmas=sigmas_voxel_2d, alpha=frangi_alpha,
            beta=frangi_beta, gamma=frangi_gamma, black_ridges=black_ridges,
            mode='reflect'
        )
        sato_result_2d = sato(
            slice_data, sigmas=sigmas_voxel_2d, black_ridges=black_ridges,
            mode='reflect'
        )

        # Combine filters (Max Projection)
        filters_combined = np.maximum(frangi_result_2d, sato_result_2d)

        # 2. Soma Rescue (Hybrid Fusion)
        if soma_preservation_factor > 0:
            f_max = np.percentile(filters_combined, 99)
            if f_max > 1e-9:
                s_max = np.percentile(slice_data, 99)
                if s_max > 1e-9:
                    scale_factor = f_max / s_max
                    normalized_raw = slice_data * scale_factor
                    filters_combined = np.maximum(
                        filters_combined, normalized_raw * soma_preservation_factor
                    )

        # Write result
        output_memmap[z] = filters_combined
        return None

    except Exception as e:
        return f"Error_slice_{z}: {str(e)}"
    finally:
        # Explicit delete to help GC with memmap handles
        del input_memmap
        del output_memmap


def enhance_tubular_structures_slice_by_slice(
    volume: np.ndarray,
    scales: List[float],
    spacing: Tuple[float, float, float],
    black_ridges: bool = False,
    frangi_alpha: float = 0.5,
    frangi_beta: float = 0.5,
    frangi_gamma: float = 15.0,
    apply_3d_smoothing: bool = True,
    smoothing_sigma_phys: float = 0.5,
    skip_tubular_enhancement: bool = False,
    soma_preservation_factor: float = 0.0,
    subtract_background_radius: int = 0
) -> Tuple[np.memmap, str, str]:
    """
    Enhances tubular structures in a 3D volume using slice-by-slice 2D filtering.
    Includes pre-smoothing, background subtraction, and soma preservation logic.

    Args:
        volume: Input 3D array (Z, Y, X).
        scales: List of scales in physical units (um).
        spacing: Voxel spacing (Z_um, Y_um, X_um).
        black_ridges: If True, detect black ridges on white.
        frangi_alpha, frangi_beta, frangi_gamma: Frangi parameters.
        apply_3d_smoothing: Apply 3D Gaussian smoothing before 2D filters.
        smoothing_sigma_phys: Smoothing sigma in microns.
        skip_tubular_enhancement: If True, skip filters (pass-through).
        soma_preservation_factor: Weight for raw intensity retention.
        subtract_background_radius: Radius for top-hat filter.

    Returns:
        Tuple containing:
        - output_memmap (np.memmap): The enhanced volume.
        - output_path (str): Path to the .dat file.
        - output_temp_dir (str): Path to the temp directory.
    """
    print(f"  [Enhance] Volume shape: {volume.shape}, Spacing: {spacing}")

    spacing_float = tuple(float(s) for s in spacing)
    volume_shape = volume.shape
    temp_dirs_created_internally: List[str] = []

    input_volume_memmap = None

    # --- Step A: Pre-Smoothing ---
    if apply_3d_smoothing and smoothing_sigma_phys > 0:
        print(f"  [Enhance] Applying initial 3D smoothing (sigma={smoothing_sigma_phys})...")
        sigma_voxel_3d = tuple(
            smoothing_sigma_phys / s if s > 0 else 0 for s in spacing_float
        )

        temp_dir = tempfile.mkdtemp(prefix="pre_smooth_")
        temp_dirs_created_internally.append(temp_dir)
        smooth_path = os.path.join(temp_dir, 'smoothed_3d.dat')
        input_volume_memmap = np.memmap(
            smooth_path, dtype=np.float32, mode='w+', shape=volume_shape
        )

        chunk_size = min(50, volume_shape[0])
        overlap = math.ceil(3 * sigma_voxel_3d[0]) if sigma_voxel_3d[0] > 0 else 0

        for i in tqdm(range(0, volume_shape[0], chunk_size),
                      desc="  [Enhance] 3D Pre-Smoothing"):
            start_r = max(0, i - overlap)
            end_r = min(volume_shape[0], i + chunk_size + overlap)
            start_w = i
            end_w = min(volume_shape[0], i + chunk_size)
            lws = start_w - start_r
            lwe = end_w - start_r

            if start_r >= end_r or lws >= lwe:
                continue

            chunk = volume[start_r:end_r].astype(np.float32, copy=False)
            smoothed_chunk = gaussian_filter(
                chunk, sigma=sigma_voxel_3d, mode='reflect'
            )
            input_volume_memmap[start_w:end_w] = smoothed_chunk[lws:lwe]

        input_volume_memmap.flush()
    else:
        print("  [Enhance] Skipping initial 3D smoothing.")
        if isinstance(volume, np.memmap) and volume.dtype == np.float32:
            input_volume_memmap = volume
        else:
            temp_dir = tempfile.mkdtemp(prefix="input_float32_copy_")
            temp_dirs_created_internally.append(temp_dir)
            path = os.path.join(temp_dir, 'input_float32.dat')
            input_volume_memmap = np.memmap(
                path, dtype=np.float32, mode='w+', shape=volume.shape
            )

            batch_sz = 10
            for i in tqdm(range(0, volume.shape[0], batch_sz),
                          desc="  [Enhance] Copying to float32"):
                end_idx = min(i + batch_sz, volume.shape[0])
                input_volume_memmap[i:end_idx] = volume[i:end_idx].astype(np.float32)
            input_volume_memmap.flush()

    # --- Step B: Prepare Output ---
    output_temp_dir = tempfile.mkdtemp(prefix='tubular_output_')
    output_path = os.path.join(output_temp_dir, 'processed_volume.dat')
    output_memmap = np.memmap(
        output_path, dtype=np.float32, mode='w+', shape=volume_shape
    )

    if skip_tubular_enhancement:
        print("  [Enhance] Skipping filter calculation (Pass-through).")
        chunk_sz = 10
        for i in tqdm(range(0, volume_shape[0], chunk_sz),
                      desc="  [Enhance] Copying"):
            end_i = min(i + chunk_sz, volume_shape[0])
            output_memmap[i:end_i] = input_volume_memmap[i:end_i]
        output_memmap.flush()
    else:
        pool = None
        try:
            print(f"  [Enhance] Starting slice-by-slice (2.5D) enhancement...")
            num_workers = max(1, os.cpu_count() - 2 if os.cpu_count() > 2 else 1)

            xy_spacing = spacing_float[1:]
            avg_xy_spacing = np.mean(xy_spacing)
            sigmas_voxel_2d = sorted([s / avg_xy_spacing for s in scales])

            input_info = (
                input_volume_memmap.filename,
                input_volume_memmap.shape,
                input_volume_memmap.dtype
            )
            output_info = (
                output_path,
                output_memmap.shape,
                output_memmap.dtype
            )

            worker_func = partial(
                _process_slice_worker,
                input_memmap_info=input_info,
                output_memmap_info=output_info,
                sigmas_voxel_2d=sigmas_voxel_2d,
                black_ridges=black_ridges,
                frangi_alpha=frangi_alpha,
                frangi_beta=frangi_beta,
                frangi_gamma=frangi_gamma,
                soma_preservation_factor=soma_preservation_factor,
                subtract_background_radius=subtract_background_radius
            )

            pool = mp.Pool(processes=num_workers, initializer=_init_worker)

            results = list(tqdm(
                pool.imap_unordered(worker_func, range(volume_shape[0])),
                total=volume_shape[0],
                desc="  [Enhance] Applying 2D Filters (Parallel)"
            ))

            errors = [r for r in results if r is not None]
            if errors:
                raise RuntimeError(f"Parallel processing failed. First error: {errors[0]}")

            output_memmap.flush()

        finally:
            if pool:
                pool.terminate()
                pool.join()

    # Cleanup input copy if we created one
    del input_volume_memmap
    gc.collect()
    for d_path in temp_dirs_created_internally:
        if os.path.exists(d_path):
            shutil.rmtree(d_path, ignore_errors=True)

    return output_memmap, output_path, output_temp_dir


def segment_cells_first_pass_raw(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    tubular_scales: List[float] = [0.5, 1.0, 2.0, 3.0],
    smooth_sigma: float = 0.5,
    connect_max_gap_physical: float = 1.0,
    min_size_voxels: int = 50,
    low_threshold_percentile: float = 25.0,
    high_threshold_percentile: float = 95.0,
    skip_tubular_enhancement: bool = False,
    soma_preservation_factor: float = 0.0,
    subtract_background_radius: int = 0
) -> Tuple[Optional[str], Optional[str], float, Dict[str, Any]]:
    """
    Executes Step 1: Raw Segmentation Pipeline.
    Pipeline: Enhance -> Normalize -> Threshold -> Connect -> Clean -> Label.

    Args:
        volume: Input 3D volume.
        spacing: Voxel dimensions (Z, Y, X).
        tubular_scales: Frangi filter scales (um).
        smooth_sigma: Initial smoothing sigma (um).
        connect_max_gap_physical: Gap size to close (um).
        min_size_voxels: Minimum object size to retain.
        low_threshold_percentile: Percentile for thresholding.
        high_threshold_percentile: Percentile for normalization.
        skip_tubular_enhancement: If True, skips filters.
        soma_preservation_factor: Weight for raw intensity.
        subtract_background_radius: Radius for background subtraction.

    Returns:
        Tuple: (labels_path, labels_temp_dir, calculated_threshold, params_dict)
    """
    print(f"\n--- Step 1: Raw Segmentation (Memory-Safe) ---")

    temp_dirs_to_clean: List[str] = []
    final_labels_memmap = None
    labels_path = None
    labels_temp_dir = None
    segmentation_threshold = 0.0

    try:
        # --- Stage 1: Feature Enhancement ---
        print("\n  [Step 1.1] Feature Enhancement...")
        processed_input_memmap, _, processed_input_temp_dir = enhance_tubular_structures_slice_by_slice(
            volume,
            scales=tubular_scales,
            spacing=spacing,
            apply_3d_smoothing=(smooth_sigma > 0),
            smoothing_sigma_phys=smooth_sigma,
            skip_tubular_enhancement=skip_tubular_enhancement,
            soma_preservation_factor=soma_preservation_factor,
            subtract_background_radius=subtract_background_radius
        )
        temp_dirs_to_clean.append(processed_input_temp_dir)

        # --- Stage 2: Normalization ---
        print("\n  [Step 1.2] Normalizing volume brightness...")
        normalized_temp_dir = tempfile.mkdtemp(prefix="normalize_")
        temp_dirs_to_clean.append(normalized_temp_dir)
        normalized_path = os.path.join(normalized_temp_dir, 'normalized.dat')
        normalized_memmap = np.memmap(
            normalized_path, dtype=np.float32, mode='w+', shape=volume.shape
        )

        # Z-profile Calculation
        raw_high_percentiles = []
        slice_indices = np.arange(volume.shape[0])

        for z in tqdm(slice_indices, desc="    Calculating Z-profile"):
            plane = processed_input_memmap[z]
            finite_mask = np.isfinite(plane)
            if np.sum(finite_mask) < 100:
                raw_high_percentiles.append(np.nan)
                continue
            finite_vals = plane[finite_mask].ravel()
            samples = np.random.choice(
                finite_vals, min(100000, finite_vals.size), replace=False
            )
            val = np.percentile(samples, high_threshold_percentile)
            raw_high_percentiles.append(float(val) if np.isfinite(val) else np.nan)

        # Curve Fitting
        raw_high_percentiles_arr = np.array(raw_high_percentiles)
        valid_indices = ~np.isnan(raw_high_percentiles_arr)
        idealized_high_percentiles = np.ones_like(raw_high_percentiles_arr)

        if np.any(valid_indices):
            poly_coeffs = np.polyfit(
                slice_indices[valid_indices],
                raw_high_percentiles_arr[valid_indices], 2
            )
            poly_func = np.poly1d(poly_coeffs)
            idealized_high_percentiles = poly_func(slice_indices)
            safe_min = np.nanpercentile(raw_high_percentiles_arr, 10)
            idealized_high_percentiles[idealized_high_percentiles < safe_min] = safe_min

        # Apply Normalization
        for z in tqdm(range(volume.shape[0]), desc="    Applying Normalization"):
            factor = idealized_high_percentiles[z]
            if factor > 1e-9:
                normalized_memmap[z] = processed_input_memmap[z] / factor
            else:
                normalized_memmap[z] = processed_input_memmap[z]
        normalized_memmap.flush()

        # --- Stage 3: Thresholding ---
        print("\n  [Step 1.3] Calculating global threshold...")
        all_samples_list = []
        z_indices_shuffled = np.arange(volume.shape[0])
        np.random.shuffle(z_indices_shuffled)

        for i in tqdm(range(0, len(z_indices_shuffled), 20), desc="    Sampling"):
            if len(all_samples_list) > 50:
                break
            slices_data = normalized_memmap[z_indices_shuffled[i:i + 20]]
            finite_values = slices_data[np.isfinite(slices_data)].ravel()
            if finite_values.size > 0:
                sample = np.random.choice(
                    finite_values, min(100000, finite_values.size), replace=False
                )
                all_samples_list.append(sample)

        if all_samples_list:
            all_samples = np.concatenate(all_samples_list)
            if all_samples.size > 0:
                segmentation_threshold = float(
                    np.percentile(all_samples, low_threshold_percentile)
                )

        print(f"    Calculated Threshold: {segmentation_threshold:.4f}")

        # Create Binary Mask
        binary_temp_dir = tempfile.mkdtemp(prefix="binary_")
        temp_dirs_to_clean.append(binary_temp_dir)
        binary_path = os.path.join(binary_temp_dir, 'b.dat')
        binary_memmap = np.memmap(
            binary_path, dtype=bool, mode='w+', shape=volume.shape
        )

        for z in tqdm(range(volume.shape[0]), desc="    Creating Binary Mask"):
            binary_memmap[z] = normalized_memmap[z] > segmentation_threshold
        binary_memmap.flush()

        del processed_input_memmap, normalized_memmap
        gc.collect()

        # --- Stage 4: Dask Cleaning (Connection) ---
        print("\n  [Step 1.4] Global Cleaning (Dask)...")
        dask_chunk_size = (32, 256, 256)

        connected_temp_dir = tempfile.mkdtemp(prefix="connected_")
        temp_dirs_to_clean.append(connected_temp_dir)
        connected_path = os.path.join(connected_temp_dir, 'connected.dat')
        connected_memmap = np.memmap(
            connected_path, dtype=bool, mode='w+', shape=volume.shape
        )

        binary_dask = da.from_array(binary_memmap, chunks=dask_chunk_size)

        radius_vox = [
            math.ceil((connect_max_gap_physical / 2) / s) if s > 1e-9 else 0
            for s in spacing
        ]
        structure = None
        if any(r > 0 for r in radius_vox):
            structure = np.ones(tuple(max(1, 2 * r + 1) for r in radius_vox), dtype=bool)

        if structure is not None:
            connected_dask = dask_image.ndmorph.binary_closing(
                binary_dask, structure=structure
            )
        else:
            connected_dask = binary_dask

        print("    Applying binary closing...")
        with ProgressBar(dt=2):
            da.store(connected_dask, connected_memmap, scheduler='threads')

        del binary_dask, connected_dask, binary_memmap
        gc.collect()

        # --- Stage 5: Labeling ---
        print("    Labeling connected components...")
        labeled_temp_dir = tempfile.mkdtemp(prefix="labeled_zarr_")
        temp_dirs_to_clean.append(labeled_temp_dir)
        labeled_zarr_path = os.path.join(labeled_temp_dir, 'labeled.zarr')

        connected_dask = da.from_array(connected_memmap, chunks=dask_chunk_size)
        s = generate_binary_structure(3, 1)
        labeled_dask, num_features_dask = dask_image.ndmeasure.label(
            connected_dask, structure=s
        )

        with ProgressBar(dt=2):
            labeled_dask.to_zarr(labeled_zarr_path, overwrite=True)

        with ProgressBar(dt=1):
            num_features = num_features_dask.compute()
        print(f"      Found {num_features} objects.")

        del connected_dask, labeled_dask
        gc.collect()

        # --- Stage 6: Filtering ---
        print(f"    Filtering objects < {min_size_voxels} voxels...")
        labels_temp_dir = tempfile.mkdtemp(prefix="labels_final_")
        labels_path = os.path.join(labels_temp_dir, 'l.dat')
        final_labels_memmap = np.memmap(
            labels_path, dtype=np.int32, mode='w+', shape=volume.shape
        )

        initial_labeled_zarr = zarr.open(labeled_zarr_path, mode='r')

        try:
            # find_objects might fail if objects are extremely fragmented/complex
            # but usually fine on labeled Zarr array if it fits index in RAM
            object_slices = ndimage.find_objects(
                initial_labeled_zarr, max_label=num_features
            )
        except Exception:
            object_slices = None

        final_label_counter = 0
        if object_slices:
            for i in tqdm(range(num_features), desc="      Filtering"):
                label = i + 1
                slices = object_slices[i]
                if slices is None:
                    continue
                try:
                    box_data = initial_labeled_zarr[slices]
                    obj_mask = (box_data == label)
                    if np.count_nonzero(obj_mask) >= min_size_voxels:
                        final_label_counter += 1
                        current_out = final_labels_memmap[slices]
                        current_out[obj_mask] = final_label_counter
                        final_labels_memmap[slices] = current_out
                except MemoryError:
                    pass

        final_labels_memmap.flush()
        print(f"    Labeling complete. {final_label_counter} objects retained.")

        first_pass_params = {
            'spacing': tuple(float(s) for s in spacing),
            'tubular_scales': tubular_scales,
            'smooth_sigma': smooth_sigma,
            'connect_max_gap_physical': connect_max_gap_physical,
            'min_size_voxels': min_size_voxels,
            'low_threshold_percentile': low_threshold_percentile,
            'high_threshold_percentile': high_threshold_percentile,
            'original_shape': volume.shape,
            'skip_tubular_enhancement': skip_tubular_enhancement,
            'soma_preservation_factor': soma_preservation_factor,
            'subtract_background_radius': subtract_background_radius
        }
        return labels_path, labels_temp_dir, segmentation_threshold, first_pass_params

    finally:
        if final_labels_memmap is not None:
            del final_labels_memmap
        print("\n  [Step 1] Cleaning up intermediate temp directories...")
        for d_dir in temp_dirs_to_clean:
            if d_dir and os.path.exists(d_dir):
                try:
                    shutil.rmtree(d_dir, ignore_errors=True)
                except Exception:
                    pass
        gc.collect()