# initial_3d_segmentation.py

**Location:** `utils/module_3d/initial_3d_segmentation.py`

## Overview
This module implements **Step 1: Raw Segmentation**. It transforms the raw, noisy grayscale microscopy volume into a clean binary mask (Foreground vs. Background). It handles the massive data volume by processing slice-by-slice or using Dask for whole-volume operations.

## The Algorithm

### 1. Tubular Enhancement (Frangi & Sato Filters)
Microglia are characterized by thin, ramified processes. Standard thresholding often loses these fine structures.
*   **Hessian Matrix:** We calculate the second-order derivatives of the image intensity.
*   **Vesselness Filters:** We compute the Frangi and Sato metrics. These filters respond strongly to tube-like structures (eigenvalues $\lambda_1 \approx 0, \lambda_2 \approx \lambda_3 \ll 0$) and suppress planar or spherical noise.
*   **2.5D Approximation:** Because 3D Hessian calculation is extremely RAM-heavy, we process the volume slice-by-slice (XY plane). We combine the results of Frangi and Sato using a maximum projection to capture the best of both.

### 2. Normalization (Clamped Polynomial Trend)
Microscopy images often suffer from uneven illumination (vignetting) or signal decay along the Z-axis (deeper tissue is darker).
*   **Z-Profile Extraction:** We sample the 95th percentile intensity from every Z-slice.
*   **Polynomial Fit:** We fit a robust 2nd-degree polynomial to this profile. This models the background lighting trend without overfitting to specific cells.
*   **Clamping:** To prevent mathematical explosions at the edges of the stack (where the polynomial might dip to 0), we clamp the normalization factor to a safe minimum.
*   **Correction:** The raw volume is divided by this trend, resulting in a "flat" image where a cell at the bottom is as bright as a cell at the top.

### 3. Global Thresholding
*   We construct a histogram from the *normalized* volume.
*   A single global percentile (e.g., 98th percentile) is chosen as the cutoff.
*   Result: A binary boolean mask.

### 4. Morphological Cleaning (Dask)
The binary mask is often fragmented (dotted lines).
*   **Binary Closing:** We use `dask-image` to apply a morphological closing operation. This bridges small gaps in the processes.
*   **Connected Components:** We identify all distinct objects using `dask_image.ndmeasure.label`.
*   **Size Filtering:** Any object smaller than `min_size_voxels` is deleted as noise.

## Memory Management Strategy
*   **Multiprocessing:** The Hessian filters run in parallel processes. We use a static worker function (`_process_slice_worker`) and explicitly set environment variables (`OMP_NUM_THREADS=1`) to prevent thread explosion.
*   **No-Flush Writes:** Workers write directly to `numpy.memmap` on disk without explicit flushing per-slice, reducing I/O lock contention.
*   **Dask & Zarr:** The connected components step uses Dask backed by temporary Zarr storage, allowing it to label graph components on datasets larger than RAM.

## Key Functions
*   `segment_cells_first_pass_raw`: The main coordinator.
*   `enhance_tubular_structures_slice_by_slice`: Manages the multiprocessing pool for 2D filtering.
*   `_process_slice_worker`: The static worker that loads one slice, filters it, and saves it.

---

## Parameter Tuning Guide

### `tubular_scales` (List of floats)
*   **Physical Meaning:** The approximate radius (in microns) of the structures you want to detect.
*   **Tuning:**
    *   **Thick branches/somas:** Add larger numbers (e.g., `2.0, 3.0`).
    *   **Fine ramifications:** Add smaller numbers (e.g., `0.5`).
    *   *Note:* Adding more scales increases Step 1 processing time linearly.

### `low_threshold_percentile` (Float 0-100)
*   **Physical Meaning:** The cutoff point in the intensity histogram.
*   **Tuning:**
    *   **Result is too noisy / "Snowy":** **Increase** this (e.g., 99.0, 99.5). You are letting in too much background.
    *   **Faint processes are missing / "Broken" cells:** **Decrease** this (e.g., 90.0, 95.0). You are cutting off real signal.

### `connect_max_gap_physical` (Microns)
*   **Physical Meaning:** The maximum distance between two pixels that should be considered "connected."
*   **Tuning:**
    *   **Processes look dotted/dashed lines:** **Increase** this (e.g., 2.0 um). This creates a larger structuring element to bridge gaps.
    *   **Distinct cells are bridging together:** **Decrease** this.

### `min_size_voxels` (Integer)
*   **Physical Meaning:** The smallest blob size allowed.
*   **Tuning:**
    *   **Speckle noise remains:** Increase (e.g., 2000).
    *   **Small cells disappear:** Decrease.