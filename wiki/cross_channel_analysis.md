# Cross-Channel Analyzer

**Corresponding modules:**
*   `utils/high_level_gui/cross_channel_window.py` — the window and recipe builder
*   `utils/high_level_gui/relational_engine.py` — execution
*   `utils/module_3d/interaction_analysis.py`,
    `utils/module_2d/interaction_analysis_2d.py` — the distance and overlap
    measurements

## What this does

The five pipeline steps measure one channel at a time. The Cross-Channel
Analyzer relates channels to each other: how much of one sits inside another, how
far each object is from its nearest partner, which objects touch.

Open it from **Cross-Channel Analyzer** in the project window. Every channel you
want to use must already have been processed through all five pipeline steps. The
analyzer reads each channel's final segmentation, and the per-channel
measurements from [Step 5](calculate_features.md) are what you join its output
against.

You build a **recipe** — an ordered list of operations — then either preview it on
one sample or run it across every sample in the project. The recipe is saved with
the results, so an analysis can be repeated.

---

## Building a recipe

Tick channels in the list, then add steps. Each step either produces a new mask
or produces measurements.

Steps that produce a mask leave it as the **previous result**, which the next step
can consume instead of a channel. That is what lets you chain: intersect B with
C, filter the result by size, then measure A against what survived.

### Intersection

Takes two inputs — two channels, or one channel and the previous result — and
keeps the voxels where both are present. You are asked how the result should be
labelled:

| Mode | Result |
| :--- | :--- |
| **Binary** | Every overlap voxel gets ID 1. One object. |
| **Connected Components** | Each separate overlapping fragment gets its own ID. |
| **Inherit Parent A** | Overlap regions keep the IDs of the first input. |
| **Inherit Parent B** | Overlap regions keep the IDs of the second input. |

For the two parent modes you are then asked whether to **keep the original IDs**
or **reset to sequential**. Keeping them means a result ID is the same number as
the object it came from in the source channel, so a row in the output can be
matched back to a row in that channel's `metrics_df`. Resetting renumbers 1…N,
which is the default and what the other modes do.

Either way, the output table carries a `parent_id_<name>` column recording the
mapping.

### Size filter

Removes objects below a threshold — µm² in 2D, µm³ in 3D. Applies to the previous
result if there is one; otherwise it asks which channel to filter, and records
that channel in the step.

Objects are renumbered after filtering, and the mapping is again kept as
`parent_id_<name>`.

### Distance analysis

Measures one side against another and writes the measurement tables. You choose
which side is **primary** — the objects the numbers are reported *for*. One row
per primary object.

Three shapes, depending on what is in the recipe already:

*   Two channels, nothing before them: you pick the partner and which of the two
    is primary.
*   A previous result exists: you choose whether the checked channel is primary
    and the previous result is the partner, or the reverse.
*   Several channels checked: each gets its own analysis step, and they merge into
    one table keyed on the primary object's ID.

### Spatial null

**Spatial Null (randomise masks)** opens a separate dialog, seeded from the
current recipe. It answers whether an observed amount of contact or proximity is
more than would arise from the same objects placed at random within the same
tissue. See [Spatial Null](spatial_null.md).

---

## Region scoping

The **region** selector applies the whole recipe to one saved sub-region instead
of the full image, for every sample that has a region of that name. Sub-regions
come from the ROI workflow — see [Sub-Regions](roi_regions.md).

The choice is recorded in `region.txt` next to the results.

---

## Running

### Preview

Runs the recipe on **one** sample and opens it in napari: the raw channels, the
segmentations, the intermediate masks each step produced, and lines drawn between
each primary object and its nearest partner. Results are written to disk, so a
preview is a single-sample run rather than a throwaway.

### Run on all samples

Asks for an analysis name, then works through every sample in the project. A
sample is skipped, with a reason printed, when it has no readable image — or, for
a region run, when that region is not present in every channel.

---

## Outputs

Everything lands under:

```
<project_root>/RELATIONAL_ANALYSIS/<analysis_name>/
├── recipe.yaml                     the recipe that produced this
├── region.txt                      which region it ran on
├── MASTER_RELATIONAL_RESULTS.csv   every sample's rows, concatenated
└── <sample>/
    ├── <sample>_relational_metrics.csv
    ├── coverage_stats_<partner>.csv
    ├── intersection_<partner>.dat
    └── (intermediate masks from each mask-producing step)
```

A region run nests one level deeper, `<sample>/<region>/`, so an analysis of the
full image and the same analysis of a region do not overwrite each other.

### `<sample>_relational_metrics.csv` — one row per primary object

Relationship measurements only — the primary object's own size and shape stay in
its channel's `metrics_df_<mode>.csv`, joined on the object ID. Each partner
contributes a set of columns suffixed with that partner's name:

| Column | Meaning |
| :--- | :--- |
| `dist_um_<partner>` | Surface-to-surface distance to the nearest partner object |
| `nearest_id_<partner>` | ID of that nearest partner |
| `is_touching_<partner>` | Whether the two overlap at all |
| `overlap_vol_with_<partner>_um3` | Overlapping volume |
| `pct_of_this_<primary>_inside_<partner>` | Fraction of this object inside the partner |
| `dominant_partner_id_<partner>` | Partner accounting for most of the overlap |
| `src_z/y/x_<partner>`, `tgt_z/y/x_<partner>` | Coordinates of the closest-approach pair |

Analysing several partners adds another set of these per partner, merged onto the
same rows — so one table answers "how far is each microglion from a neuron, and
from a vessel".

### `coverage_stats_<partner>.csv` — one row per **partner** object

The same relationship from the other side:

| Column | Meaning |
| :--- | :--- |
| `id_<partner>` | The partner object |
| `total_vol_of_<primary>_inside_this_<partner>` | How much primary material it contains |
| `count_of_unique_<primary>_touching_this_<partner>` | How many primary objects touch it |
| `list_of_<primary>_ids_touching_this_<partner>` | Which ones |

### `MASTER_RELATIONAL_RESULTS.csv`

Every sample's per-object rows concatenated, with a `sample_name` column
identifying each. This is the file to load for a cross-sample comparison.

---

## Viewing a saved analysis

Saved analyses appear in the project window's contents tree. Opening one loads
the sample with its raw channels, its segmentations, the analysis's intermediate
masks and the nearest-neighbour lines — the same view a preview produces, without
recomputing.

---

## Where to go next

*   [Sub-Regions](roi_regions.md) — restricting an analysis to a drawn region.
*   [Spatial Null](spatial_null.md) — testing an observed relationship against
    randomised placement.
*   [Step 5](calculate_features.md) — the per-channel measurements these build on.
