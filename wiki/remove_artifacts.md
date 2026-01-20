# Step 2: Edge Trimming & Artifact Removal

**Corresponding Modules:**
*   **3D:** `utils/module_3d/remove_artifacts.py`
*   **2D:** `utils/fluorescence_module_2d/remove_artifacts_2d.py`

## Overview
Microscopy tissue blocks often suffer from the **"Edge Effect"**: the physical cut surface of the tissue can fluoresce brightly due to damage, or contain sheared cell fragments. Standard segmentation algorithms often mistake this edge signal for a massive, continuous cell.

**Step 2** addresses this by mathematically "shrink-wrapping" the tissue volume and deleting any segmented objects that are too close to this outer surface, unless they are bright enough to be considered real somas.

### The Algorithm: "3D Shrink-Wrapping"
1.  **Hull Generation:**
    *   The algorithm downsamples the image and uses an automated threshold (Otsu * 0.5) to distinguish "Tissue" from "Empty Background".
    *   It computes the **Convex Hull** (2D) for each slice.
    *   **3D Smoothing:** In 3D mode, these 2D hulls are stacked and smoothed using binary closing/opening to create a continuous 3D volume that wraps the tissue.
2.  **Boundary Definition:**
    *   This "Shrink Wrapped" Hull is eroded by a specific thickness.
    *   The difference between the Hull and the Eroded Hull defines the **Boundary Zone** (the danger zone where artifacts live).
3.  **Distance-Based Trimming:**
    *   A Distance Transform is calculated from the edge of the Hull inwards.
    *   Any pixel in the segmentation mask is **deleted** if:
        *   It is within `edge_trim_distance_threshold` of the surface.
        *   **AND** it is darker than `brightness_cutoff_factor` (Safety Guard).
4.  **Z-Correction (3D Only):**
    *   Optionally performs "Clamped Erosion" in the Z-axis to fix anisotropy artifacts (where cells look like tall pillars due to poor Z-resolution).

---

## 🔧 Parameter Tuning Guide

### 1. `edge_trim_distance_threshold`
**Type:** Float (Microns) | **Default:** ~4.0 - 12.0

*   **High Level:** How "deep" into the tissue do we peel away the artifacts?
*   **Detailed Explanation:** This defines the width of the danger zone starting from the detected tissue surface. Any segmentation within this distance is considered a candidate for deletion.
*   **How to Tune:**
    *   **Scenario A: Edge artifacts remain.**
        *   *Fix:* **Increase** the value (e.g., `5.0` -> `15.0`). Measure the depth of the "garbage" layer in Napari using the line tool.
    *   **Scenario B: Real peripheral cells are being deleted.**
        *   *Fix:* **Decrease** the value (e.g., `5.0` -> `2.0`).
    *   **Disable:** Set to `0.0` to skip this step entirely.

### 2. `hull_boundary_thickness_phys`
**Type:** Float (Microns) | **Default:** ~2.0

*   **High Level:** The thickness of the reference shell used for calculations.
*   **Detailed Explanation:** This parameter controls the erosion of the initial hull to create the "Boundary Mask". It essentially sets the resolution of the boundary detection.
*   **How to Tune:**
    *   **Standard:** `2.0` or `3.0` is almost always correct.
    *   **Scenario:** If your tissue surface is extremely irregular (like a jagged coastline) and the hull is bridging over crevices too aggressively, **decrease** this to make the boundary hug the tissue tighter.

### 3. `brightness_cutoff_factor`
**Type:** Float | **Default:** 1.5 - 2.0 (or 1,000,000 to disable)

*   **High Level:** The "Safety Guard." Don't delete things at the edge if they are this bright.
*   **Detailed Explanation:** The algorithm calculates a reference brightness (usually the segmentation threshold or an Otsu threshold of the tissue).
    *   If a pixel at the edge has `Intensity > (Reference * Factor)`, it is **preserved**, even if it is inside the trim distance.
    *   This saves real cell bodies (which are bright) that happen to sit near the edge, while deleting the dim, smeary background noise.
*   **How to Tune:**
    *   **Scenario A: Real somas at the edge are being deleted.**
        *   *Fix:* Set to `1.5` or `2.0`. This tells the tool: "If it's twice as bright as the background, keep it."
    *   **Scenario B: Bright edge artifacts are NOT being deleted.**
        *   *Fix:* **Increase** the value (e.g., `5.0` or `100.0`) or set to a huge number (`1000000.0`) to effectively disable the safety guard and delete *everything* near the edge.

### 4. `min_size` (or `min_size_voxels`/`min_size_pixels`)
**Type:** Integer

*   **High Level:** Post-trim cleaning.
*   **Detailed Explanation:** Trimming the edges often leaves behind tiny floating fragments of segmentation. This step runs a connected component analysis *after* trimming and removes these shards.
*   **How to Tune:**
    *   Set this similar to or slightly smaller than the `min_size` used in Step 1.

### 5. `z_erosion_iterations` (3D Only)
**Type:** Integer | **Default:** 0 - 2

*   **High Level:** Fixes "stretched" cells in the Z-axis.
*   **Detailed Explanation:** In many confocal/light-sheet images, the Z-resolution is lower than XY, causing spherical cells to look like footballs or pillars ("Z-smearing"). This parameter applies binary erosion **only along the Z-axis**.
    *   **Clamped Logic:** It includes a safety check to ensure it doesn't erode an object completely out of existence; it will always leave at least 1 pixel in Z.
*   **How to Tune:**
    *   **Scenario: Cells look like tall cylinders instead of spheres.**
        *   *Fix:* Increase to `1` or `2`.
    *   **Scenario: Cells are flat or Z-resolution is isotropic.**
        *   *Fix:* Set to `0`.

### 6. `heal_iterations` / `smoothing_iterations`
**Type:** Integer | **Default:** 1

*   **High Level:** Smooths the hull shape.
*   **Detailed Explanation:** Number of morphological closing/opening iterations applied to the convex hull mask.
*   **How to Tune:**
    *   **Standard:** `1` is usually sufficient.
    *   **Scenario:** If the hull looks too jagged or noisy, increase to `2` or `3`.