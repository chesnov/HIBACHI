# Cross-Channel Analyzer

**Corresponding modules:**
*   `utils/high_level_gui/cross_channel_window.py` — the window and recipe builder
*   `utils/high_level_gui/relational_engine.py` — execution
*   `utils/fluorescence_module/interaction_analysis.py` — the distance and
    overlap measurements, at both ranks (this was once two modules, one per
    rank; they are merged, and the 2D entry points remain as thin aliases)

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
keeps the voxels where both are present.

**This builds a mask; it does not quantify the overlap.** You get the overlap
region and, through the Step 5 feature pipeline, the size and shape of each
overlap fragment — but nothing relating that back to either parent channel, so
no coverage percentage. If what you want is "how much of A is inside B", add an
**Overlap / Colocalisation** step; the analyzer offers to do that for you when
you add an intersection to a recipe that measures nothing yet.

You are asked how the result should be labelled:

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

### Overlap / Colocalisation

Measures how much of one side sits inside the other. You choose which side is
**primary** — the objects the per-object numbers are reported *for*. One row per
primary object.

This produces two different things, and the distinction matters:

*   **Per primary object**, in `<sample>_relational_metrics.csv`: what fraction
    of *that individual object* lies inside a partner
    (`pct_of_this_<primary>_inside_<partner>`), how much overlap by extent,
    whether it overlaps at all, and which partner accounts for most of it.
*   **Per sample and channel pair**, in `overlap_summary.csv`: what fraction of
    the *whole channel* lies inside the other, in both directions. This is a
    ratio of totals, so it cannot be recovered by averaging the per-object
    column — that would weight a tiny object the same as a huge one.

Both directions are always reported in the summary, so the choice of primary
only decides whose objects the rows are, not which number you can get.

This step does not compute distances, and skips the distance transform
entirely.

### Distance / Proximity

Measures how far each object is from its nearest partner, edge to edge. Again
you choose which side is primary, and get one row per primary object:
`dist_um_<partner>`, the nearest partner's ID, and the closest-approach
coordinates that the preview draws its connection lines from.

This step does not compute overlap.

Adding both steps for the same primary and partner merges them onto the same
rows, which is what the old combined "Distance Analysis" button always did. A
recipe saved before the split has no measurement recorded on its steps, and is
run as *both* — so saved recipes reproduce exactly as before.

### Shape of a measurement step

Three shapes, depending on what is in the recipe already. This applies to
overlap and distance alike:

*   Two channels, nothing before them: you pick the partner and which of the two
    is primary.
*   A previous result exists: you choose whether the checked channel is primary
    and the previous result is the partner, or the reverse.
*   Several channels checked: each gets its own step. Steps sharing a primary
    merge into one table keyed on that primary's ID; steps with *different*
    primaries describe different objects, so they are written to one file each
    rather than forced together.

### Full pairwise distances

Every primary against every partner, rather than just the nearest, written to
`pairwise_distances_<partner>.csv`. **Off by default** — it is by far the most
expensive measurement here, and nothing else in the pipeline reads the file.
Set `pairwise: true` on an analyze step in `recipe.yaml` when you want the raw
pair list for your own statistics.

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
├── MASTER_RELATIONAL_RESULTS.csv   every sample's per-object rows, concatenated
├── MASTER_OVERLAP_SUMMARY.csv      every sample's sample-level overlap figures
└── <sample>/
    ├── <sample>_relational_metrics.csv
    ├── overlap_summary.csv
    ├── coverage_stats_<partner>.csv
    ├── pairwise_distances_<partner>.csv   (only if asked for)
    ├── intersection_<partner>.dat
    └── (intermediate masks from each mask-producing step)
```

When one recipe measures several different primaries, the per-object table is
written once per primary as `<sample>_relational_metrics_<primary>.csv` instead
of the single file above, since those rows describe different objects.

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

### `overlap_summary.csv` — one row per sample and channel pair

The sample-level answer to "what percentage of channel A is inside channel B".
Written by every **Overlap** step. Column names are generic (`primary` /
`partner` rather than the channel names) so rows for different pairs and
different samples concatenate into one tidy table.

| Column | Meaning |
| :--- | :--- |
| `sample_name`, `primary`, `partner` | Which pair, which way round |
| `n_primary`, `n_partner` | Object counts on each side |
| `total_primary_um3`, `total_partner_um3` | Total extent of each channel (`_um2` in 2D) |
| `overlap_um3` | Extent of the intersection |
| `pct_of_primary_inside_partner` | `overlap / total_primary` — coverage of A by B |
| `pct_of_partner_inside_primary` | `overlap / total_partner` — coverage of B by A |
| `n_primary_touching_partner` | How many primary objects overlap at all |
| `pct_of_primary_objects_touching_partner` | That count as a fraction of `n_primary` |

The last two are an *incidence* measure — what share of objects are involved —
which is a different question from what share of the material is. Both are
often wanted.

### `MASTER_RELATIONAL_RESULTS.csv` and `MASTER_OVERLAP_SUMMARY.csv`

The per-object rows and the sample-level overlap rows, each concatenated across
every sample with a `sample_name` column. `MASTER_OVERLAP_SUMMARY.csv` is
normally the one to load for a cross-sample comparison of overlap;
`MASTER_RELATIONAL_RESULTS.csv` is for per-object distributions.

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
