# Step 4: Cell Separation (Instance Segmentation)

**Corresponding Modules:**
*   **3D:** `utils/module_3d/cell_splitting.py`
*   **2D:** `utils/fluorescence_module_2d/cell_splitting_2d.py`

## Overview
This step performs the actual **Instance Segmentation** of touching cells. It takes the binary mask (from Step 2) and the seeds (from Step 3) to partition complex clumps into individual, distinct cells.

It solves the classic "Watershed Problem" (over-segmentation) using a custom **Graph-Based Merging** algorithm that checks if a split line is biologically plausible.

### The Algorithm: "Divide, Watershed, Stitch"

1.  **Chunking (Divide):**
    *   To handle terabyte-scale images, the volume is divided into overlapping blocks (chunks).
    *   The algorithm identifies "Multi-Soma Objects" (blobs containing >1 seed). Single-soma objects are copied directly to the output to save computation time.

2.  **Seeded Watershed (The Split):**
    *   For each multi-soma object, a **Marker-Controlled Watershed** is executed.
    *   **Landscape:** The inverted Distance Transform (valleys = thickest parts of the cell). Optionally weighted by image intensity.
    *   **Markers:** The seeds detected in Step 3.
    *   **Result:** The object is strictly divided into territories, one per seed.

3.  **Graph-Based Merging (The Correction):**
    *   Watershed is "greedy" and often creates straight-line cuts across thick branches.
    *   We build a **Region Adjacency Graph (RAG)** where nodes are cell segments and edges are the boundaries between them.
    *   We analyze the **Interface** between every pair of touching segments using heuristics:
        *   **Valley Depth:** Is the boundary dark relative to the cell body? (Dark = Split, Bright = Merge).
        *   **Local Contrast:** Do the two regions have different mean intensities?
    *   Based on these metrics, false splits are merged back together.

4.  **Seed-Aware Stitching (Re-assembly):**
    *   Chunks are stitched back together.
    *   **Orphan Detection:** The algorithm ensures that every resulting fragment contains a valid seed. If a fragment is orphaned (cut off from its seed by a chunk boundary or bad split), it is merged into its best neighbor.

---

## 🔧 Parameter Tuning Guide

### 1. `min_path_intensity_ratio`
**Type:** Float (0.0 - 1.0) | **Default:** ~0.6 - 0.8

*   **High Level:** The "Valley Check". Decides if a line between two cells is a real gap or just a fake watershed cut.
*   **Detailed Explanation:** It calculates `Ratio = Mean_Intensity_at_Boundary / Mean_Intensity_of_Soma`.
    *   **Low Ratio (< Threshold):** The boundary is dark (a valley). **Keep Split.**
    *   **High Ratio (> Threshold):** The boundary is bright (same brightness as soma). **Merge.**
*   **How to Tune:**
    *   **Symptom: Real cells are merged into one blob.**
        *   *Fix:* **Increase** the threshold (e.g., `0.6` -> `0.85`). This tells the algorithm: "Even if the boundary is somewhat bright, keep them split."
    *   **Symptom: A single cell is cut in half (often by a straight line).**
        *   *Fix:* **Decrease** the threshold (e.g., `0.8` -> `0.5`). This tells the algorithm: "Only split if there is a *very* dark, deep valley."

### 2. `min_local_intensity_difference`
**Type:** Float (0.0 - 1.0) | **Default:** 0.01 - 0.05

*   **High Level:** The "Contrast Check". Are these two segments different *colors* (intensities)?
*   **Detailed Explanation:** If Segment A is very bright and Segment B is dim, they are likely distinct cells, even if they touch.
*   **How to Tune:**
    *   **Scenario:** You have touching cells with very different brightness levels that are being merged.
        *   *Fix:* **Increase** this value (e.g., to `0.1` or `0.2`).

### 3. `intensity_weight`
**Type:** Float | **Default:** 0.0 - 0.5

*   **High Level:** Influences *where* the cut happens.
*   **Detailed Explanation:** Modifies the Watershed landscape.
    *   **0.0 (Geometry Only):** Cuts are placed purely based on shape (halfway between seeds). Best for dense clumps where boundaries are invisible.
    *   **> 0.0 (Geometry + Intensity):** The cut line will try to snap to dark pixels.
*   **How to Tune:**
    *   **Scenario:** The cut line looks jagged or doesn't follow the visible dark gap between cells.
        *   *Fix:* Increase to `0.5` or `1.0`.

### 4. `local_analysis_radius`
**Type:** Integer (Microns) | **Default:** ~5 - 10

*   **High Level:** Context size.
*   **Detailed Explanation:** How far around the boundary should we look to calculate local contrast?
*   **How to Tune:**
    *   Standard values of `5` or `10` um are usually fine. Increase if your image is very noisy/speckled to average out the noise.

### 5. `min_size_threshold` (Final Size)
**Type:** Float/Int (Microns/Pixels)

*   **High Level:** Final cleanup.
*   **Detailed Explanation:** After splitting, any resulting cell fragment smaller than this is deleted (or merged into a neighbor).
*   **How to Tune:**
    *   Set this to the smallest size of a valid cell fragment you care about.

### 6. `max_hole_size`
**Type:** Integer (Pixels) | **Default:** 0 (Disabled)

*   **High Level:** Fills holes inside cells.
*   **Detailed Explanation:** Runs a binary fill operation. If a hole inside a cell is smaller than this many pixels, it gets filled in.
*   **How to Tune:**
    *   **Scenario:** Cells look like "webs" or donuts.
    *   *Fix:* Set to a moderate value (e.g., `100` pixels).