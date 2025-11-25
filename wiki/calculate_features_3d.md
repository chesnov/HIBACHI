# calculate_features_3d.py

**Location:** `utils/module_3d/calculate_features_3d.py`

## Overview
Implements **Step 4** of the workflow. It takes the final labeled segmentation mask and calculates biological metrics for every cell.

## Metrics Calculated
1.  **Morphology:**
    *   **Volume:** Voxel count $\times$ Voxel size.
    *   **Surface Area:** Calculated using a discrete surface difference approximation.
    *   **Sphericity:** A measure of how round a cell is ($1.0$ is a perfect sphere, $<0.2$ is highly ramified).
    *   **Depth:** Median Z-position (in microns) relative to the top of the stack.
2.  **Spatial Distribution (Heavy):**
    *   **Shortest Distance:** The exact Euclidean distance from the surface of Cell A to the surface of its nearest neighbor.
    *   **Nearest Neighbor Identity:** Which label is the closest?
    *   **Contact Points:** The specific $(Z, Y, X)$ coordinates where the distance is minimized.
3.  **Ramification (Skeletonization):**
    *   Uses `skimage.morphology.skeletonize` to thin the cell down to a 1-pixel line.
    *   **Custom Pruning:** Removes tiny artifacts (spurs < X microns) that don't represent real branches.
    *   **Topology Analysis:** Counts Junctions, Endpoints, and total Branch Length using `skan` and custom neighbor-counting logic.

## Memory Strategy
*   **Surface Caching:** Instead of keeping all cell surfaces in RAM for the distance matrix (which explodes for 10k+ cells), surfaces are extracted once and saved to temporary `.npy` files. Distance workers load only the two specific surfaces they need.
*   **Batching:** Skeletons are processed sequentially or in small batches to avoid overhead.

## Key Functions
*   `analyze_segmentation`: The main coordinator.
*   `shortest_distance`: Manages the $O(N^2)$ distance matrix calculation using a process pool.
*   `calculate_ramification_with_skan`: Handles skeletonization and topological cleanup.