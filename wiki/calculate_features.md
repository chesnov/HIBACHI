# Step 5: Feature Calculation

**Corresponding Modules:**
*   **3D:** `utils/module_3d/calculate_features_3d.py`
*   **2D:** `utils/fluorescence_module_2d/calculate_features_2d.py`

## Overview
This final step generates biological insights from the segmented images. It processes the labeled mask (from Step 4) and the original intensity image to compute a wide range of morphometric, spatial, and intensity statistics for every single cell.

The results are saved as a CSV (`metrics_df_fluorescence.csv`) and optionally as an FCS file for flow cytometry software.

## Metrics Calculated

### 1. Morphology (Shape)
*   **Size:**
    *   **3D:** Volume ($\mu m^3$), Surface Area ($\mu m^2$), Sphericity ($0.0 - 1.0$).
    *   **2D:** Area ($\mu m^2$), Perimeter ($\mu m$), Circularity ($0.0 - 1.0$).
*   **Dimensions:** Bounding box size, Major/Minor Axis Length (via Ellipsoid fit).
*   **Position:** Centroid coordinates and Depth (Z-position relative to stack surface).

### 2. Intensity (Signal)
Statistics derived from the raw image within the cell mask:
*   Mean, Median, Standard Deviation, Max.
*   **Integrated Density:** Sum of all pixel intensities (Total Fluorescence).

### 3. Spatial Distribution (Distance)
*   **Shortest Distance:** The exact Euclidean distance from the edge of Cell A to the edge of its nearest neighbor.
*   **Nearest Neighbor Identity:** The Label ID of the closest cell.
*   **Contact Points:** The specific coordinates $(X, Y, Z)$ where the two cells are closest.

### 4. Ramification (Skeletonization)
Describes the branching complexity of the cell (critical for Microglia activation states).
*   **Algorithm:** Uses `skimage` to thin the cell to 1 pixel width and `skan` to analyze the graph topology.
*   **Metrics:**
    *   **True Branches:** Number of biological branches.
    *   **Junctions:** Number of split points.
    *   **Endpoints:** Number of tips.
    *   **Total Length:** Sum of all branch lengths ($\mu m$).
    *   **Avg Branch Length:** Average length per segment.

---

## 🔧 Parameter Tuning Guide

### 1. `prune_spurs_le_um`
**Type:** Float (Microns) | **Default:** ~5.0 - 10.0

*   **High Level:** Cleans up the skeleton. Removes tiny "hairs" caused by rough cell surfaces.
*   **Detailed Explanation:** Skeletonization algorithms often produce tiny branches at the surface of a rough object (artifacts). This parameter recursively removes terminal branches (spurs) that are shorter than this threshold.
*   **How to Tune:**
    *   **Scenario: Skeletons look "hairy" or branch counts are impossibly high.**
        *   *Fix:* **Increase** this value. A standard microglial branch is usually > 5um. Setting this to `0.0` keeps every single pixel protrusion.
    *   **Scenario: Real, fine branch tips are being cut off.**
        *   *Fix:* **Decrease** this value.

### 2. `calculate_distances`
**Type:** Boolean | **Default:** `True`

*   **High Level:** Performance Toggle.
*   **Detailed Explanation:** Calculating the exact surface-to-surface distance for every pair of cells is an $O(N^2)$ operation.
    *   **Performance:** For < 5,000 cells, it is fast. For > 50,000 cells, this step can take hours.
*   **Tuning:**
    *   **Set to `False`** if you only care about cell counts/volumes and want the pipeline to finish instantly.

### 3. `calculate_skeletons`
**Type:** Boolean | **Default:** `True`

*   **High Level:** Performance Toggle.
*   **Detailed Explanation:** Skeletonization is computationally expensive.
*   **Tuning:**
    *   **Set to `False`** if analyzing roughly spherical cells (e.g., nuclei, resting macrophages) where branching data is irrelevant.

---

## 💾 Outputs

### **1. `metrics_df_fluorescence.csv`**
The master spreadsheet containing one row per cell with all the metrics above.

### **2. `skeleton_array_fluorescence.dat`**
A visualizable label mask of the skeletons. Load this in Napari to visually verify if `prune_spurs_le_um` is set correctly.

### **3. `distance_matrix_fluorescence.csv`** (Optional)
A full matrix containing distances between *every* pair of cells (not just nearest neighbors). Useful for clustering analysis (e.g., Ripley's K).

### **4. `points_matrix_fluorescence.csv`** (Optional)
Contains coordinate pairs used to draw the red "connection lines" in the GUI between nearest neighbors.