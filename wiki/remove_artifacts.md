# remove_artifacts.py

**Location:** `utils/module_3d/remove_artifacts.py`

## Overview
This module implements **Step 2: Edge Trimming**. It is responsible for removing artifacts caused by tissue edges, which often appear as large, undifferentiated blocks of fluorescence in raw segmentation.

## The Algorithm: "Shrink-Wrapping"
1.  **Hull Generation (2D Stack):**
    *   We iterate through every Z-slice.
    *   We combine the raw segmentation mask with a rough Otsu threshold of the original image to find *anything* that looks like tissue.
    *   We calculate the **Convex Hull** (the shape formed if you wrapped a rubber band around the points) for that slice.
2.  **3D Smoothing (Dask):**
    *   The stack of 2D hulls can be jagged. We use **Dask** to apply 3D binary closing and opening. This bridges gaps between slices, creating a smooth 3D "shrink-wrap" of the tissue.
3.  **Boundary Extraction:**
    *   We erode the smooth hull by a specific physical distance (e.g., 2.0 um).
    *   The difference between the original hull and the eroded hull is the "Boundary Zone."
4.  **Distance-Based Trimming:**
    *   We calculate the **Euclidean Distance Transform (EDT)** from the boundary into the tissue.
    *   Any segmented object that falls within the "Boundary Zone" (distance < threshold) AND is very bright (potential edge flare) is deleted.

## Key Functions
*   `apply_hull_trimming`: The main coordinator function.
*   `generate_hull_boundary_and_stack`: Creates the 3D hull using slice-by-slice processing + Dask smoothing.
*   `trim_object_edges_by_distance`: Uses Dask to compute the distance transform and filter objects.

## Memory Management
*   **Zarr Intermediate Storage:** Dask operations write intermediate steps to temporary Zarr stores on disk to prevent RAM saturation during 3D morphology.
*   **Chunked Processing:** Operations are performed on small cubes (e.g., 64x256x256 voxels) at a time.