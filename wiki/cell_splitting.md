# cell_splitting.py

**Location:** `utils/module_3d/cell_splitting.py`

## Overview
Implements **Step 4: Cell Separation**.
This module performs the final instance segmentation. It uses the seeds found in Step 3 to partition the binary mask from Step 2 into distinct cells.

## The Algorithm: "Divide, Watershed, Stitch"

### 1. Divide (Chunking)
*   The volume is divided into overlapping 3D chunks (e.g., `128x512x512`).
*   We identify "Multi-Soma Objects" (blobs containing >1 seed). Single-soma objects are copied directly to the output to save time.

### 2. Watershed (The Split)
*   Inside a chunk, for each multi-soma object, we run a **Marker-Controlled Watershed**.
*   **Landscape:** The inverted Distance Transform (valleys = thickest parts).
*   **Markers:** The seeds from Step 3.
*   **Result:** The object is divided into territories, one per seed.

### 3. Graph-Based Merging (The Correction)
*   Watershed is "greedy" and often over-segments (creates straight lines or shatters cells).
*   We build a **Region Adjacency Graph (RAG)**. Nodes are segments; Edges are the boundaries between them.
*   We analyze the **Interface** between every pair of touching segments:
    *   **Intensity Ratio:** Is the boundary dark relative to the cell body? (Dark = Split, Bright = Merge).
    *   **Local Contrast:** Do the two regions have different mean intensities?
*   Based on these metrics, we merge false splits back together.

### 4. Stitching (Re-assembly)
*   We process the overlap regions between chunks.
*   If Label A in Chunk 1 overlaps significantly with Label B in Chunk 2, we record a correspondence.
*   We apply a Union-Find algorithm to resolve all ID conflicts globally, ensuring seamless boundaries across the grid.

### 5. Void Filling
*   Finally, we run `binary_fill_holes` on every cell to ensure skeletons don't look like webs.

---

## Parameter Tuning Guide

### 1. The "Split" Logic
*   **`intensity_weight`** (Float, 0.0 - 10.0)
    *   **What it does:** Adds image intensity to the Watershed landscape.
    *   **0.0:** Pure geometric split (halfway between seeds). Best for dense clumps where boundaries are invisible.
    *   **> 0.0:** Biases the cut towards dark pixels. Use if you have visible dark gaps between cells.

### 2. The "Merge" Logic (Fixing Over-segmentation)
This is the most critical tuning section for fixing "Straight Line" cuts.

*   **`min_path_intensity_ratio`** (Float, 0.0 - 1.0)
    *   **Concept:** `Ratio = Interface_Brightness / Soma_Brightness`.
    *   **Logic:**
        *   Low Ratio (< Threshold) -> Dark Interface -> **Valid Split**.
        *   High Ratio (> Threshold) -> Bright Interface -> **Merge**.
    *   **Symptom: Cells are cut in half (Straight lines).**
        *   This means the algorithm kept a split it should have merged.
        *   The interface was bright (Ratio ~1.0).
        *   We need `Ratio > Threshold` to be TRUE to trigger a merge.
        *   **Action:** **Decrease** the threshold (e.g., from 0.8 to 0.6). This makes it stricter: "You must be *very* dark to stay split."
    *   **Symptom: Two distinct cells are merged into one.**
        *   **Action:** **Increase** the threshold (e.g., 0.9). Allow brighter interfaces to remain as splits.

*   **`min_local_intensity_difference`** (Float)
    *   **Concept:** If Region A is bright and Region B is dim, they are likely different cells.
    *   **Action:** **Increase** this to force splitting based on contrast differences, even if they touch.

### 3. Performance
*   **`local_analysis_radius`** (Microns)
    *   The size of the area around the boundary used to calculate contrast. ~10um is standard.