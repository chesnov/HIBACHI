# HIBACHI
### Heuristic-Informed Batch Analysis for Cell Histological Identification

**HIBACHI** segments, separates and measures cells in large 2D and 3D microscopy
datasets. It uses memory mapping and chunked processing to work on volumes larger
than RAM, and graph-based heuristics to separate touching cells that a watershed
alone would either merge or over-split.

<p align="center">
  <img src="assets/example_segmentation.png" alt="HIBACHI segmentation example" width="900">
  <br>
  <em>Left: raw fluorescence intensity. Centre: separated cells, one colour per
  cell. Right: skeletonised cells with the shortest distance between neighbours
  marked in red.</em>
</p>

---

## What it does

*   **Works past RAM.** Every intermediate result is a memory-mapped file on
    disk, so image size is bounded by disk rather than memory.
*   **Separates clumped cells.** A marker-controlled watershed splits at cell
    cores, then a graph-based pass re-merges cuts that are not real boundaries,
    judged on intensity valleys and local contrast.
*   **Measures.** Morphology, intensity, ramification from skeletons, and
    surface-to-surface nearest-neighbour distances, per cell.
*   **Relates channels.** Overlap, containment and proximity between channels,
    with a Monte-Carlo spatial null to test whether an arrangement differs from
    random.
*   **Batches.** Whole projects unattended, with resume after an interruption.
*   **Interactive.** Built on **napari** and **PyQt5**: tune each step and see the
    result before committing.

---

## Installing

You do not need Python, conda or git.

*   **Windows** — download `HIBACHI-Setup.exe` from the
    [latest release](https://github.com/chesnov/HIBACHI/releases/latest) and
    double-click it.
*   **macOS** — download `HIBACHI.dmg` from the
    [latest release](https://github.com/chesnov/HIBACHI/releases/latest) and drag
    **HIBACHI** to Applications. macOS blocks unsigned apps on first open;
    [INSTALL.md](INSTALL.md) has the steps to allow it.
*   **Linux** —

    ```bash
    curl -fsSL https://raw.githubusercontent.com/chesnov/HIBACHI/main/install/install.sh | bash
    ```

The first run downloads the scientific packages, which takes a few minutes.
HIBACHI then checks for updates on launch and asks before installing one.

**[INSTALL.md](INSTALL.md)** covers troubleshooting, environment variables,
rolling back to an earlier version, uninstalling, and releasing.

---

## Getting started

1.  Put your images in a folder.
2.  Open that folder in HIBACHI — drag it onto the welcome screen, or use
    **Browse**.
3.  HIBACHI offers to set it up as a project: pick a starting config per channel,
    and confirm the physical dimensions if it cannot read them from the files.
4.  Open one image, tune the five steps, then process the rest as a batch.

Readable formats: **TIFF**, **Zeiss CZI**, **Leica LIF**, and whole-slide formats
(**VSI**, **SVS**, **NDPI**, **SCN**, **AFI**, **QPTIFF**, **ZVI**, **OME-TIFF**,
**DICOM**). A slide or `.lif` holding several acquisitions becomes several
samples.

Physical dimensions come from a metadata CSV if you supply one, otherwise from
the file's own metadata. When neither has them, HIBACHI asks rather than
assuming — every size, distance and density depends on them. Each config records
where its dimensions came from.

See **[Project setup](wiki/project_setup.md)** for the details.

---

## The pipeline

Five steps, each tunable in the sidebar with the result visible before you move
on.

| Step | Does |
| :--- | :--- |
| **1. Raw segmentation** | Multi-scale Hessian (Frangi/Sato) tubularity plus intensity thresholding, one threshold per scale, OR-merged |
| **2. Edge trimming** | Builds a per-slice tissue hull and removes objects damaged at the tissue edge |
| **3. Soma extraction** | Finds one core seed per cell by distance-transform peeling and intensity percentiles |
| **4. Cell separation** | Watershed from those seeds, then a graph pass that re-merges cuts which are not real boundaries |
| **5. Feature calculation** | Morphology, intensity, skeleton ramification, nearest-neighbour distances |

Beyond the per-channel pipeline:

*   **[Cross-Channel Analyzer](wiki/cross_channel_analysis.md)** — build a recipe
    of intersections, size filters and distance analyses, and run it across the
    project.
*   **[Spatial Null](wiki/spatial_null.md)** — re-place the real masks at random
    in the same tissue to test whether an observed arrangement is non-random.
*   **[Sub-regions](wiki/roi_regions.md)** — draw a polygon and treat it as an
    image of its own, for fast tuning or for analysing part of a sample.

---

## Outputs

Per image, in `<image>_processed_<mode>/`:

| File | Contents |
| :--- | :--- |
| `metrics_df_<mode>.csv` | One row per cell: morphology, intensity, skeleton and neighbour metrics |
| `analysis_summary_<mode>.csv` | One row per image: analysed extent, object count, densities |
| `final_segmentation_<mode>.dat` | The per-cell label mask |
| `skeleton_array_<mode>.dat` | One-voxel-wide skeletons |
| `branch_data_<mode>.csv` | Per-branch skeleton statistics |
| `distances_matrix_<mode>.csv` | Full pairwise surface-to-surface distances |
| `points_matrix_<mode>.csv` | Closest-approach coordinates, used for the red lines above |
| `metrics_<mode>.fcs` | The numeric columns, for flow-cytometry tools |
| `processing_config_<mode>.yaml` | The exact parameters used, with the pipeline version |

Cross-channel results go to `<project>/RELATIONAL_ANALYSIS/<analysis>/`, with a
`MASTER_RELATIONAL_RESULTS.csv` across samples.

---

## Documentation

In `wiki/`.

**Getting started**
*   [Project setup](wiki/project_setup.md) — opening projects, formats, physical
    dimensions
*   [Tuning workflow](wiki/tuning_workflow.md) — how to approach the parameters

**The five steps**
*   [Step 1: Raw segmentation](wiki/initial_segmentation.md)
*   [Step 2: Edge trimming](wiki/remove_artifacts.md)
*   [Step 3: Soma extraction](wiki/soma_extraction.md) — *read before tuning*
*   [Step 4: Cell separation](wiki/cell_splitting.md) — *read before tuning*
*   [Step 5: Feature calculation](wiki/calculate_features.md)

**Beyond one channel**
*   [Cross-Channel Analyzer](wiki/cross_channel_analysis.md)
*   [Spatial Null](wiki/spatial_null.md)
*   [Sub-regions](wiki/roi_regions.md)

**Working at scale**
*   [Batch processing](wiki/batch_processor.md)
*   [Config Library](wiki/config_library.md) — reusing tuned configs across
    projects

**Reference**
*   [Strategies](wiki/processing_strategies.md) — how the steps are wired together
*   [GUI Manager](wiki/gui_manager.md) — navigation, dirty state, ROI sessions
*   [`segment.py`](wiki/segment.md) — startup and diagnostics ordering
*   [Helpers](wiki/segmentation_helpers.md) — shared numerical utilities
*   [Diagnostics](wiki/diagnostics.md) — logs, crash reports, troubleshooting
*   [INSTALL.md](INSTALL.md) — installing, environment variables, releasing
