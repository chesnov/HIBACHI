# Step 5: Feature Calculation

**Corresponding modules:**
*   **3D:** `utils/module_3d/calculate_features_3d.py`
*   **2D:** `utils/module_2d/calculate_features_2d.py`

## What this step decides

Step 5 measures the finished cells. It changes nothing about the segmentation —
it reads the per-cell mask from [Step 4](cell_splitting.md) and the original
intensity image, and writes out one row of numbers per cell.

Because it only measures, there is nothing here to get "wrong" in the way the
earlier steps can be wrong. Its three parameters are a **skeleton cleanup
threshold** and **two on/off switches that trade completeness for runtime**. If a
number looks wrong at this stage, the cause is almost always upstream: the mask
being measured is wrong, or the image has no real physical scale.

> **Under the hood.** Measurement runs in four independent blocks — morphology,
> intensity, pairwise distances, skeletons — each producing a table keyed on
> `label`, which are then merged into one. The distance and skeleton blocks are
> the only expensive ones, and each has its own switch. The step finishes by
> stamping the analysed region onto the table and writing a one-row summary, so a
> count can be turned into a density without any other file.

---

## What gets measured

### Morphology

The 3D and 2D metric sets are **not** the same, because the meaningful shape
descriptors differ by dimensionality:

| | 3D | 2D |
| :--- | :--- | :--- |
| Size | `volume_um3` | `area_um2` |
| Boundary | `surface_area_um2` | `perimeter_um` |
| Compactness | `sphericity` | `circularity` |
| Raw count | `voxel_count` | `pixel_count` |
| Also | `solidity` *(see note)*, `depth_um` | `solidity`, `eccentricity`, `depth_um` *(see note)* |

Surface area in 3D is a face-counting approximation with the boundary faces
included, and sphericity is derived from it, so both inherit the staircase
roughness of a voxel surface. Perimeter in 2D is measured by tracing marching-
squares contours in physical units, which is why it is not simply a pixel count.

> **Two placeholder columns.** 3D `solidity` is written as a hardcoded `0.0` — it
> is **not computed**, and is present only to keep the 3D and 2D tables the same
> width. Likewise 2D `depth_um` is always `0.0`. Do not interpret either. (2D
> `solidity` *is* real, from `regionprops`.) In 3D, `depth_um` is real and is the
> **median** Z of the cell's voxels times the Z spacing — a depth-in-stack
> figure, not a distance from the tissue surface.

### Intensity

Computed from the raw image inside each cell mask, identically in 2D and 3D:
`mean_intensity`, `median_intensity`, `std_intensity`, `max_intensity`, and
`integrated_density` (the sum of all voxel/pixel values — total fluorescence).

### Nearest-neighbour distances

Two columns land on the metrics table: `shortest_distance_um` and
`closest_neighbor_label`. These are true **surface-to-surface** distances in
microns, not centroid distances — each cell is reduced to its boundary voxels and
the minimum separation between boundary point sets is found with a KD-tree.

The full N×N matrix and the coordinates of each closest-contact pair are written
to their own files (see *Outputs*), not into the metrics table.

### Ramification (skeletons)

Each cell is thinned to a one-voxel-wide skeleton and analysed as a graph.
Reported per cell:

| Column | Meaning |
| :--- | :--- |
| `true_num_branches` | Branch count, derived as `endpoints - 1 + junctions` |
| `true_num_junctions` | Voxels with 3+ skeleton neighbours |
| `true_num_endpoints` | Voxels with exactly 1 skeleton neighbour |
| `skan_total_length_um` | Sum of all branch lengths |
| `skan_avg_branch_length_um` | Mean branch length |
| `skan_num_skeleton_voxels` (3D) / `skan_num_skeleton_pixels` (2D) | Skeleton size |

Note the last column's name differs between modes — worth knowing when writing
analysis scripts that must handle both.

> **Under the hood.** The skeleton is forced to be a true tree, in three stages:
> a maximum-spanning-tree pass breaks graph cycles, keeping the thicker path at
> each loop (edges are weighted by mean distance-transform thickness, so a real
> thick process wins over a thin spurious link); spurs are then pruned by the
> micron threshold below; and finally a voxel-level pass detects residual
> diagonal loops and snaps them at their thinnest point. Branches that touch the
> image border are protected from pruning, since a tip cut off by the field of
> view is not a spur. `true_num_*` counts come from direct neighbour counting on
> the final skeleton, while `skan_*` values come from `skan`'s path summary —
> which is why they can disagree slightly on pathological shapes.

### Analysed region

Every run also stamps two constant columns on the metrics table:
`analyzed_region` (`full_image` or `roi`) and `analyzed_volume_um3` (3D) /
`analyzed_area_um2` (2D). These are the **denominator** for a density, and they
matter most when comparing a full image against a sub-region: for an ROI the
value is the **polygon's** area or volume, not the cropped array's, because the
crop is bounding-box shaped with everything outside the polygon zeroed. A
triangular ROI's array overstates the analysed region by roughly 2×.

---

## 🔧 Tuning this step

### Prune Skeleton Spurs ≤ (µm)

**`prune_spurs_le_um`** — float, µm. **Default `6.0`** in both 2D and 3D.

Skeletonising a rough object produces short terminal twigs that are surface
noise, not branches. This removes terminal branches shorter than the threshold,
repeatedly, so a twig hanging off a twig also goes.

*   **Skeletons look hairy; branch or endpoint counts are implausibly high** →
    **increase**. A real microglial process is typically well over 5 µm.
*   **Genuine fine process tips are disappearing** → **decrease**.
*   `0.0` disables pruning and keeps every single-voxel protrusion.

Judge this visually on the **Skeletons** layer rather than from the numbers
alone. Note this is the one parameter here that changes results rather than just
runtime, so it is worth setting deliberately.

### Calculate Distances

**`calculate_distances`** — bool, default `true`.

Runtime switch. Surface extraction plus the pairwise search scales roughly with
the square of the cell count, and it is parallelised across cores with the matrix
held on disk to keep RAM flat. Turn it **off** if you only need counts,
morphology and intensity. Turning it off also removes `shortest_distance_um` and
`closest_neighbor_label` from the metrics table, and suppresses the distance and
points files and the nearest-neighbour lines in the viewer.

### Calculate Skeletons

**`calculate_skeletons`** — bool, default `true`.

Runtime switch. Turn it **off** for roughly convex objects (nuclei, beads,
non-ramified cells) where branching is meaningless. This removes all
`true_num_*` and `skan_*` columns, the skeleton array and the branch-data file.

> **Tip.** While tuning Steps 1–4, switching both of these off makes each Step 5
> run fast, so you can iterate on the segmentation and only turn them back on for
> the final run and for batch processing.

---

## Parameter reference

| Parameter | Type | 3D default | 2D default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Prune Skeleton Spurs <= (um)** (`prune_spurs_le_um`) | float µm | `6.0` | `6.0` | Removes terminal branches below this length, recursively. `0` keeps everything. Border-touching tips are never pruned. |
| **Calculate Distances** (`calculate_distances`) | bool | `true` | `true` | Off = no distance columns, no matrix/points files, no viewer lines. |
| **Calculate Skeletons** (`calculate_skeletons`) | bool | `true` | `true` | Off = no ramification columns, no skeleton or branch outputs. |

---

## What you see in the viewer

*   **Skeletons** — the pruned one-voxel-wide centrelines, as a labels layer
    sharing IDs with the cells. This is how you check `prune_spurs_le_um`.
*   **Nearest-neighbour lines** — a vectors/shapes layer joining each cell to its
    closest neighbour at the exact contact points. Only present when
    **Calculate Distances** is on.

## Outputs on disk

Written to `<image>_processed_<mode>/`, where `<mode>` is `fluorescence` (3D) or
`fluorescence_2d` (2D):

| File | Contents |
| :--- | :--- |
| `metrics_df_<mode>.csv` | **The main result.** One row per cell, all columns above. |
| `analysis_summary_<mode>.csv` | One row for the whole image: mode, region, image shape, analysed pixels, analysed area/volume, object count, and `density_per_mm2` / `density_per_mm3`. Written **even when zero cells were found**, because "no cells in this much tissue" is a result. For an ROI it also records the bounding-box pixel count and the polygon's fraction of it. |
| `metrics_<mode>.fcs` | The numeric columns as an FCS file for flow-cytometry tools. Requires `fcswrite`; skipped silently if unavailable. |
| `skeleton_array_<mode>.dat` | Skeleton label mask (int32, image shape). Skeletons only. |
| `branch_data_<mode>.csv` | Per-**branch** `skan` statistics with coordinates in global image space — one row per branch, not per cell. Skeletons only. |
| `distances_matrix_<mode>.csv` | Full N×N surface-to-surface distance matrix, labelled on both axes. Useful for clustering or Ripley's-K style analysis. Distances only. |
| `points_matrix_<mode>.csv` | Contact-point coordinates for each nearest-neighbour pair (`mask1`, `mask2`, and `mask1_z/y/x`, `mask2_z/y/x`; `y/x` only in 2D). Drives the viewer lines. Distances only. |

> **Note on empty results.** `metrics_df_<mode>.csv` is only written when at least
> one cell was measured, but `analysis_summary_<mode>.csv` is always written. If
> you are aggregating a batch, read the summaries to distinguish "zero cells" from
> "never processed".

---

## Where to go next

*   Counts are only comparable between images once normalised — use the
    `analyzed_*` columns or `analysis_summary_*.csv` as the denominator,
    especially when mixing full-image and ROI runs.
*   To relate these cells to another channel, use the
    [Cross-Channel Analyzer](cross_channel_analysis.md) rather than anything in
    this step.
*   If a metric looks wrong, the mask is the usual cause: go back to
    [Step 4](cell_splitting.md), then [Step 3](soma_extraction.md).
