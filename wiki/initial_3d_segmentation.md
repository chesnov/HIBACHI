# initial_3d_segmentation.py

**Location:** `utils/module_3d/initial_3d_segmentation.py`

## Overview
This module performs the **Step 1: Raw Segmentation** of the workflow. It takes the raw intensity image and converts it into a rough binary mask of candidate cells.

## The Algorithm
1.  **Tubular Enhancement (Frangi & Sato Filters):**
    *   Microglia processes are thin and tube-like.
    *   We calculate the Hessian matrix (second-order derivatives) for every pixel.
    *   We compute "Vesselness" metrics (Frangi and Sato) to highlight tube-like structures while suppressing noise.
    *   **Optimization:** This is done slice-by-slice (2.5D) using multiprocessing to save RAM, as true 3D Hessian calculation on a laptop is often prohibitively expensive.
2.  **Normalization (Polynomial Trend):**
    *   Microscopy images often have uneven lighting (vignetting) or Z-decay (deeper slices are darker).
    *   We calculate the background trend using a polynomial fit and divide the signal by this trend to flatten the brightness profile.
3.  **Thresholding:**
    *   A global threshold is calculated based on the normalized histogram.
4.  **Dask-based Cleaning:**
    *   We use `dask-image` to perform morphological operations (closing gaps) and finding connected components (labeling) on the entire 3D volume without loading it all into RAM at once.
    *   Small artifacts (< `min_size_voxels`) are filtered out.

## Key Functions
*   `segment_cells_first_pass_raw`: The main pipeline function.
*   `enhance_tubular_structures_slice_by_slice`: Handles the parallel execution of filters.
*   `_process_slice_worker`: The static worker function that runs in a separate process for each Z-slice.

## Memory Safety Features
*   **Memmap Input/Output:** Input volumes and intermediate steps (smoothing, normalization) are written to disk immediately.
*   **Zarr Storage:** Connected components are stored in a Chunked Zarr array (a format optimized for parallel read/writes) instead of a giant Numpy array.
*   **Float32:** Explicitly casts data to 32-bit floats to halve memory bandwidth usage.