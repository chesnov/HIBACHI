# Step 5: Feature Calculation

**Corresponding modules:**
*   **3D:** `utils/module_3d/calculate_features_3d.py`
*   **2D:** `utils/module_2d/calculate_features_2d.py`

## What this step decides

Step 5 measures the finished cells: it reads the per-cell mask from
[Step 4](cell_splitting.md) and the original intensity image, and writes out one
row of numbers per cell. It does not alter the segmentation: the object count
and the object shapes are whatever [Step 4](cell_splitting.md) produced.

**Calculate Distances**, **Calculate Skeletons** and **Calculate Solidity** are
on/off switches. They decide whether a block of measurements is computed at all,
trading completeness for runtime, and cannot change a number that *is* computed.

**Prune Skeleton Spurs** is the one parameter here that changes a computed
value. It alters the skeleton, and so the ramification metrics derived from it.
Set it too high and real processes are removed: branch, junction and endpoint
counts fall and total length shrinks, without anything looking obviously wrong.
Check it against the **Skeletons** layer rather than accepting the default.

> **Absolute sizes depend on the image's calibration.** Every measurement in
> microns is derived from the physical dimensions recorded in the image's config,
> which are set during project setup — see
> [Project setup](project_setup.md#physical-dimensions). The config's
> `dimensions_source` field records where they came from:
>
> *   `metadata` — read from the image file's own header.
> *   `csv` — taken from a metadata CSV supplied at setup.
> *   `manual` — entered or confirmed in the dimension-entry dialog.
> *   `mixed` — different axes came from different sources above.
> *   `pixels_assumed` — at least one axis had no scale from any of those
>     sources, so its extent was recorded as that axis's pixel count, which
>     amounts to 1 µm per pixel on that axis. The setup log names which axes.
> *   `unknown` — the config does not carry the field, so no source was recorded.
>
> The field records provenance only. To check the dimensions themselves, compare
> the config's `voxel_dimensions` / `pixel_dimensions` against your acquisition
> settings.

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
| Also | `solidity` *(switchable)*, `depth_um` | `solidity` *(switchable)*, `eccentricity`, `depth_um` *(see note)* |

Surface area in 3D is a face-counting approximation with the boundary faces
included, and sphericity is derived from it, so both inherit the staircase
roughness of a voxel surface. Perimeter in 2D is measured by tracing marching-
squares contours in physical units, not by counting boundary pixels.

> **`solidity` is switchable in both modes.** It is the object's element count
> divided by its convex hull's, computed the same way in 2D and 3D, so the two
> are directly comparable — extruding a 2D shape into a 3D prism gives the same
> value. Being a ratio of areas or volumes it is dimensionless, so pixel
> anisotropy does not affect it. **Calculate Solidity** defaults to on in 2D and
> off in 3D, because the cost differs by orders of magnitude (see
> [below](#calculate-solidity)). When off, the column is `NaN`. An object with no
> hull — a single element, a line, or in 3D a plane — is also `NaN`.

> **2D `depth_um` is a placeholder.** It is always `0.0`, present to keep the 3D
> and 2D tables the same width. In 3D, `depth_um` is the **median** Z of the
> cell's voxels times the Z spacing — a depth-in-stack figure, not a distance
> from the tissue surface.

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

Every run stamps the region it measured onto the metrics table, as **two**
extents:

*   `analyzed_volume_um3` (3D) / `analyzed_area_um2` (2D) — the whole region the
    pipeline was given: the image, or the polygon of a drawn region.
*   `tissue_volume_um3` / `tissue_area_um2` — the tissue hull that
    [Step 2](remove_artifacts.md) kept inside that region.

Both are given because a count can reasonably be normalised by either, and which
one is right depends on the question. `analysis_summary_<mode>.csv` carries a
density against each — `density_per_mm3` and `tissue_density_per_mm3` in 3D,
`density_per_mm2` and `tissue_density_per_mm2` in 2D.

The two are **equal** when there is no hull to measure — edge trimming switched
off — or when the hull reached the edges of the region. `analyzed_region` says
whether the region was the full image or an ROI, and `extent_basis` says where
the tissue figure came from: `tissue_hull`, or `full_image` / `polygon` when no
hull was found and the tissue extent fell back to the total.

The hull is measured from the edge mask Step 2 saves, one slice at a time.

---

## 🔧 Tuning this step

### Prune Skeleton Spurs ≤ (µm)

**`prune_spurs_le_um`** — float, µm. **Default `6.0`** in both 2D and 3D.

Skeletonising a rough object produces short terminal twigs that are surface
noise, not branches. This removes terminal branches shorter than the threshold,
repeatedly, so a twig hanging off a twig also goes.

*   **Skeletons look hairy; branch or endpoint counts are higher than the
    structures you can see** → **increase**, up to the shortest branch length
    you consider real for your cells.
*   **Genuine fine process tips are disappearing** → **decrease**.
*   `0.0` disables spur pruning. The skeleton is still forced to a tree by the
    cycle-breaking passes described below, so it is not simply the raw thinning
    result.

**Judge this on the Skeletons layer, not on the numbers.** Every ramification
metric is derived from the pruned skeleton, so this parameter sets
`true_num_branches`, `true_num_junctions`, `true_num_endpoints`,
`skan_total_length_um` and `skan_avg_branch_length_um` together. Over-pruning
does not produce an obviously broken result — it produces cells that look less
ramified than they are. If two conditions are processed with different values,
a difference in ramification between them may come from the parameter rather
than the biology, so set it once and keep it fixed across a study.

Border-touching tips are exempt, so a process cut off by the edge of the field is
not mistaken for a spur and removed.

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

### Calculate Solidity

**`calculate_solidity`** — bool. **Default `true` in 2D, `false` in 3D.**

Fills the `solidity` column; left off it is `NaN`.

The defaults differ because the cost does. A convex hull is built per object over
that object's bounding box, so the work grows with the box rather than with the
element count — and a bounding box is an area in 2D and a volume in 3D. Measured
here: about **1 ms per cell in 2D**, against **0.5 s for a compact 3D cell** and
**46 s** for one 3D cell whose processes spanned a 150×500×500 box. The worst
case in 3D is exactly a widely-arborised cell: large box, few voxels.

So in 2D it is on and unlikely to be noticed. In 3D, turn it on when you want the
metric and expect Step 5 to take substantially longer. It does not change any
other column either way.

> **Tip.** While tuning Steps 1–4, switching these three off makes each Step 5
> run fast, so you can iterate on the segmentation and only turn them back on for
> the final run and for batch processing. Unlike the pruning threshold, toggling
> them changes nothing about the numbers you eventually get — only whether they
> are computed.

---

## Parameter reference

| Parameter | Type | 3D default | 2D default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Prune Skeleton Spurs <= (um)** (`prune_spurs_le_um`) | float µm | `6.0` | `6.0` | **Changes results.** Removes terminal branches below this length, recursively; sets every ramification metric. `0` keeps everything. Border-touching tips are never pruned. Keep fixed across a study. |
| **Calculate Distances** (`calculate_distances`) | bool | `true` | `true` | Off = no distance columns, no matrix/points files, no viewer lines. |
| **Calculate Skeletons** (`calculate_skeletons`) | bool | `true` | `true` | Off = no ramification columns, no skeleton or branch outputs. |
| **Calculate Solidity** (`calculate_solidity`) | bool | `false` | `true` | Off = `solidity` is `NaN`. Same definition in both modes, so values are comparable. Cost scales with each object's bounding box, which is why the defaults differ. |

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
| `analysis_summary_<mode>.csv` | One row for the whole image: mode, region, `extent_basis`, image shape, both extents (`analyzed_*` and `tissue_*`, in pixels and physical units), object count, and a density against each (`density_per_mm2`/`3` and `tissue_density_per_mm2`/`3`). Written **even when zero cells were found**, because "no cells in this much tissue" is a result. For an ROI it also records the bounding-box pixel count and the polygon's fraction of it. |
| `metrics_<mode>.fcs` | The numeric columns as an FCS file for flow-cytometry tools. Requires `fcswrite`; skipped silently if unavailable. |
| `skeleton_array_<mode>.dat` | Skeleton label mask (int32, image shape). Skeletons only. |
| `branch_data_<mode>.csv` | Per-**branch** `skan` statistics with coordinates in global image space — one row per branch, not per cell. Skeletons only. |
| `distances_matrix_<mode>.csv` | Full N×N surface-to-surface distance matrix, labelled on both axes. Useful for clustering or Ripley's-K style analysis. Distances only. |
| `points_matrix_<mode>.csv` | Contact-point coordinates for each nearest-neighbour pair (`mask1`, `mask2`, and `mask1_z/y/x`, `mask2_z/y/x`; `y/x` only in 2D). Drives the viewer lines. Distances only. |

> **Note on empty results.** `metrics_df_<mode>.csv` is only written when at
> least one cell was measured, whereas `analysis_summary_<mode>.csv` is written
> whether or not any cells were found. When aggregating a batch, read the
> summaries to tell "zero cells" apart from "never processed".

---

## Where to go next

*   To turn counts into densities, use the `analyzed_*` columns or
    `analysis_summary_*.csv` as the denominator. Full-image and ROI runs have
    different denominators, so they are recorded per run.
*   To relate these cells to another channel, use the
    [Cross-Channel Analyzer](cross_channel_analysis.md); this step measures one
    channel only.
*   This step reads the mask from [Step 4](cell_splitting.md) and does not
    modify it, so a metric that does not match the image is traced back through
    [Step 4](cell_splitting.md) and [Step 3](soma_extraction.md).
