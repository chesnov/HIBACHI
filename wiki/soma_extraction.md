# soma_extraction.py

**Location:** `utils/module_3d/soma_extraction.py`

## Overview
Implements **Step 3a** of the workflow. Its goal is to identify the "core" of each cell (the soma) within the binary segmentation mask. These cores act as "seeds" for the subsequent splitting step.

## Algorithm
1.  **Classification:** Objects are sorted into "Small" (likely single cells) and "Large" (potential multi-soma clumps) based on volume.
2.  **Iterative Erosion (The "Peeling" method):**
    *   For large objects, we compute the Distance Transform (DT).
    *   We threshold the DT at various ratios (e.g., 30%, 50%, 70% of max thickness).
    *   This "peels away" the thin branches, leaving only the thickest parts (the somas).
3.  **Intensity Verification:**
    *   We also check the *intensity* image. Somas are usually brighter.
    *   We verify candidates by eroding bright regions.
4.  **Candidate Filtering:**
    *   Potential soma seeds are filtered based on:
        *   **Volume:** Must be above `min_fragment_size`.
        *   **Thickness:** Must be thick enough (`absolute_min_thickness_um`).
        *   **Shape:** Must not be too elongated (checking aspect ratio via PCA).

## Key Functions
*   `extract_soma_masks`: The main entry point.
*   `_filter_candidate_fragment_3d`: Validates if a blob is a plausible cell body.