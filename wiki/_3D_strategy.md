# _3D_strategy.py

**Location:** `utils/module_3d/_3D_strategy.py`

## Overview
The `RamifiedStrategy` class implements the specific 4-step workflow for segmenting complex, ramified microglia in 3D. It inherits from the abstract `ProcessingStrategy`.

## Workflow Steps
1.  **Raw Segmentation:** Uses multi-scale filters (Frangi/Sato) to detect tubular structures.
2.  **Edge Trimming:** Removes artifacts near the edges of the tissue using a convex hull approach.
3.  **Refine ROIs (Splitting):** The heaviest step. separating touching cells (multi-soma splitting) using watershed and graph-cut algorithms.
4.  **Feature Calculation:** Generates statistics (volume, sphericity, branch length) and skeletons.

## Memory Management Strategy
*   **Memory Mapping (Memmap):** Almost all intermediate data (binary masks, labels) are stored as `.dat` files on the hard drive using `numpy.memmap`. This allows processing files larger than physical RAM.
*   **Reference Clearing:** The strategy explicitly deletes local variables holding large arrays after passing them to the processing functions.
*   **Intermediate State:** Small metadata (thresholds) is stored in `self.intermediate_state`, while large data (the original image volume) is passed by reference only when needed.

## Key Methods
*   `execute_raw_segmentation`: Coordinates the initial tubule detection.
*   `execute_refine_rois`: Orchestrates the complex cell splitting logic. Handles the logic of saving the result, whether the splitting function returns a RAM array or a file path.
*   `_close_memmap(self, memmap_obj)`: **(New)** A helper to safely flush changes to disk and delete the python object, releasing the file lock.