# remove_artifacts.py

**Location:** `utils/module_3d/remove_artifacts.py`

## Overview
Implements **Step 2: Edge Trimming**.
This module addresses the "Edge Effect" common in tissue blocks: the physical cut surface of the tissue often fluoresces brightly or contains damaged cell debris, which segmentation algorithms mistake for massive cells.

## The Algorithm: "3D Shrink-Wrapping"

### 1. Slice-by-Slice Hull Generation
*   We iterate through every Z-slice.
*   We create a temporary mask combining the raw segmentation (Step 1) with a low-level Otsu threshold of the original image (to catch faint background tissue).
*   We calculate the **2D Convex Hull** for that slice. This is like wrapping a rubber band around the tissue section.

### 2. 3D Smoothing (Dask)
*   Stacking 2D hulls creates a "jagged" 3D shape.
*   We use **Dask** to perform 3D binary closing and opening on the hull stack. This smooths out the transitions between slices, creating a coherent "Shrink Wrap" around the entire tissue volume.

### 3. Boundary Extraction
*   We erode the smooth hull by a user-defined distance (`hull_boundary_thickness_phys`).
*   We subtract the eroded hull from the original hull. The result is a shell (the **Boundary Zone**) representing the outer layer of the tissue.

### 4. Distance-Based Trimming
*   We calculate the **Euclidean Distance Transform (EDT)** from the boundary inward.
*   We iterate through the segmentation mask. Any pixel that meets **all three** criteria is set to 0 (deleted):
    1.  It is currently segmented.
    2.  Its distance from the surface is less than `edge_trim_distance_threshold`.
    3.  (*Optional Safety*) Its intensity is below a brightness cutoff (to prevent deleting real, bright cells that happen to be near the edge).

## Memory Management
*   **Staged Dask Pipeline:** The 3D morphology and distance transforms are computed in stages. Each stage (Smoothing, EDT) writes its result to a temporary Zarr store on disk before the next stage begins. This "checkpointing" prevents the Dask graph from becoming too large and consuming all RAM.
*   **Safe Cleanup:** A dedicated `_safe_close_memmap` helper ensures file handles are released before temporary directories are deleted.

---

## Parameter Tuning Guide

### `edge_trim_distance_threshold` (Microns)
*   **Physical Meaning:** How deep is the "damaged layer" of your tissue?
*   **How to Tune:**
    1.  Open the raw image in Napari.
    2.  Look at the edges (top/bottom/sides). Measure how far the "smeary" or "garbage" signal extends into the tissue.
    3.  Set this value slightly higher than your measurement (e.g., if garbage is 10um deep, set to 12.0).
    4.  **Set to 0.0 to disable this step entirely.**

### `hull_boundary_thickness_phys` (Microns)
*   **Physical Meaning:** The thickness of the reference shell used for calculations.
*   **Tuning:** Usually `2.0` or `3.0` is fine. Only increase if your tissue surface is extremely irregular and the algorithm is missing crevices.

### `brightness_cutoff_factor` (Float)
*   **Physical Meaning:** A safety guard. "Only delete edge pixels if they are darker than `Threshold * Factor`."
*   **Tuning:**
    *   **Standard:** Set to `1000000.0` (effectively disabled) if you trust the distance trimming.
    *   **Conservative:** Set to `1.5` or `2.0`. This means "Delete edge artifacts, BUT if you see something *really* bright (like a real cell soma) at the edge, keep it."