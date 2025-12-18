# Step 6: Interaction Analysis (Multi-Channel)

**Corresponding Module:**
*   **Shared (2D & 3D):** `utils/module_3d/interaction_analysis.py`

## Overview
This step performs **Spatial Relation Analysis** between the cells segmented in the current run ("Primary") and objects segmented in a different channel ("Reference").

**Common Use Cases:**
*   Microglia (Primary) engulfing Amyloid Plaques (Reference).
*   Microglia (Primary) wrapping around Blood Vessels (Reference).
*   Synaptic Puncta (Primary) inside Dendrites (Reference).

This module is **N-Dimensional** and **Directional**. It calculates how much of the Reference object is covered by the Primary cell, and vice-versa.

### The Algorithm

1.  **Data Loading:**
    *   The user selects a *different* project folder containing the segmented results for the Reference channel.
    *   The algorithm automatically finds the corresponding `final_segmentation.dat` file for the current image.

2.  **Overlap Analysis (Intersection):**
    *   It computes the boolean intersection: `Primary_Mask AND Reference_Mask`.
    *   **Volume/Area:** Calculates the total physical size of the overlap.
    *   **Coverage Fraction:**
        *   How much of the Primary cell is touching the Reference? ($Overlap / Vol_{Primary}$)
        *   How much of the Reference object is covered by this specific Primary cell? ($Overlap / Vol_{Reference}$)
    *   **Dominance:** If a Primary cell touches multiple Reference objects (e.g., a microglia touching two plaques), it identifies the "Dominant" partner (the one with the largest overlap volume).

3.  **Distance Analysis (Proximity):**
    *   If the cells are not touching, how close are they?
    *   Calculates the **Euclidean Distance Transform (EDT)** from the surface of the Reference objects.
    *   For every Primary cell, it finds the **minimum** value on this distance map.
    *   **Result:** Exact distance from the edge of the Microglia to the nearest Plaque.

4.  **Intersection Mask Generation:**
    *   A new image file (`intersection_REFNAME.dat`) is created containing only the overlapping voxels/pixels.
    *   These are labeled with unique IDs, allowing you to visualize exactly *where* the interaction happens in 3D/2D space.

---

## 🔧 Parameter Tuning Guide

Since this step is analytic rather than segmentation-based, "tuning" mostly involves toggling features for performance.

### 1. `target_channel_folder`
**Type:** String (Path)

*   **High Level:** Where is the data you want to compare against?
*   **Detailed Explanation:** This must be the **Root Project Folder** of the other channel.
    *   *Example:* If you are currently processing `Project_Microglia`, select `Project_Amyloid`.
    *   The algorithm assumes the folder structure matches (i.e., if you are processing `Image_01` in Microglia, it looks for `Image_01` in Amyloid).
*   **How to set:** Use the "Analyze with Other Channels..." button in the GUI.

### 2. `calculate_distance`
**Type:** Boolean | **Default:** `True`

*   **High Level:** Performance Toggle.
*   **Detailed Explanation:** Calculating the Distance Transform for a large 3D volume (e.g., blood vessels) is RAM-intensive.
    *   If you only care about physical **Overlap** (engulfment), set this to `False`.
    *   If you need to know "Distance to nearest plaque" for cells *not* touching plaques, keep `True`.

### 3. `calculate_overlap`
**Type:** Boolean | **Default:** `True`

*   **High Level:** The core metric.
*   **Detailed Explanation:** Calculates Volume/Area intersection. Almost always keep this `True`.

---

## 💾 Outputs

### **1. `metrics_df_ramified.csv` (Updated)**
New columns are appended to the main cell metrics file:
*   `dist_to_nearest_REF_um`: Distance to the reference object.
*   `overlap_vol_um3_REF`: Volume of intersection.
*   `overlap_fraction_REF`: % of the cell volume that is overlapping.
*   `dominant_overlap_id_REF`: The Label ID of the specific reference object being touched.

### **2. `interaction_REF_coverage.csv`**
A separate report focused on the **Reference** objects.
*   *Rows:* Reference IDs (e.g., Plaque 1, Plaque 2).
*   *Columns:*
    *   `total_overlap_vol_um3`: Total volume of microglia covering this plaque.
    *   `percent_covered`: % of the plaque surface covered by microglia.
    *   `interacting_cell_count`: How many different microglia are touching this single plaque?

### **3. `intersection_REF.dat`**
A labeled mask file showing only the touching regions. Load this in Napari to create 3D renderings of the "synapse" or "contact point" between channels.