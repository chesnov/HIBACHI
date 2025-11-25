# cell_splitting.py

**Location:** `utils/module_3d/cell_splitting.py`

## Overview
Implements **Step 3b** of the workflow. It takes the trimmed binary mask (Step 2) and the extracted soma seeds (Step 3a) to produce the final instance segmentation. Its primary goal is to correctly separate touching cells without over-segmenting single cells that have complex shapes.

## Key Algorithms

### 1. Multi-Soma Identification
*   The script scans the volume to identify "Multi-Soma Objects"—single connected components in the binary mask that contain more than one seed (soma).
*   Single-soma objects are skipped (copied directly to output) to save time.

### 2. Chunked Watershed (Divide)
*   The volume is divided into overlapping 3D chunks.
*   Inside each chunk:
    *   A **Distance Transform** is calculated for the local shape.
    *   A **Marker-Controlled Watershed** is run using the seeds found in Step 3a.
    *   This roughly cuts the object halfway between the seeds.

### 3. Graph-Based Merging (Conquer)
*   The watershed often over-segments.
*   We build a **Region Adjacency Graph (RAG)** where nodes are segments and edges are the boundaries between them.
*   **Heuristic:** We analyze the intensity of the boundary pixels.
    *   **Merge:** If the boundary intensity is high (similar to the cell body), it's likely just one cell.
    *   **Split:** If the boundary intensity is low (a "valley"), it's a true separation between cells.

### 4. Pixel-Perfect Stitching
*   After processing chunks, we must reassemble them.
*   We look at the **Overlap Regions** (e.g., 32 pixels).
*   If Label A in Chunk 1 overlaps significantly with Label B in Chunk 2, they are mapped to the same global ID.
*   This resolves the "Edge Artifacts" often seen in tiled processing.

## Key Functions
*   `separate_multi_soma_cells`: The main coordinator (Chunking -> Stitching).
*   `_separate_multi_soma_cells_chunk`: The worker function for a single block.
*   `_calculate_interface_metrics`: The "Judge" that decides if two segments should merge.