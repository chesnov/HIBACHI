# Cross-Channel Analyzer

**Corresponding modules:**
*   `utils/high_level_gui/recipe_library.py` — saved recipes, shared across
    projects
*   `utils/high_level_gui/cross_channel_window.py` — the recipe panel, the
    runner and the overlay viewer
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

Cross-channel analysis lives in the project window. **Analysis →
Cross-Channel Recipe…** opens a dock holding the recipe and three buttons:
**Overlap**, **Distance** and **Size Filter**. Each opens one form; nothing is
asked in a chain of follow-up prompts.

**Recipes…** in the dock opens the recipe library: saved recipes live under
`~/.hibachi/recipes`, alongside the processing configs in `~/.hibachi/configs`,
and are available in every project. You can save the current recipe, load,
rename, delete, export one to share, or import one — including the
`recipe.yaml` that every run leaves beside its results, which needs no
conversion. A recipe names channels by number, so the library tells you when a
recipe needs a channel this project does not have.

A run's scope is whatever is **checked in the project tree** — any mix of full
images and regions. Results are opened from the tree like any other view.
Spatial Null is a project-level tool under the same menu.

There is no separate analyzer window; it was a second copy of the project
window's sample list, region picker and viewer, and only the recipe was ever
unique to it.

Each step either produces a new mask
or produces measurements.

Steps that produce a mask leave it as the **previous result**, which the next step
can consume instead of a channel. That is what lets you chain: intersect B with
C, filter the result by size, then measure A against what survived.

### Overlap

One step, one form. You pick which side is **primary** (its objects are the
rows) and which is the **partner**, then tick what you want out of it:

*   **Coverage percentages** — what fraction of one channel sits inside the
    other. Per object in `<sample>_relational_metrics.csv`, and per sample and
    pair in `overlap_summary.csv`. Both directions are always reported, so the
    primary choice only decides whose objects the rows are, not which number
    you can get. This is a ratio of totals and cannot be recovered by averaging
    the per-object column, which would weight a tiny object the same as a huge
    one.
*   **Size and shape of each overlap region** — treats each overlap patch as an
    object in its own right and runs it through the Step 5 feature pipeline.
*   **Keep the overlap as a mask for later steps** — only this makes the
    overlap the input to whatever follows in the recipe. Choosing it reveals how
    to label the result: a number per region, one ID for all of it, or inherit
    one side's object IDs.

Overlap and Intersection used to be separate buttons. Both computed the same
intersection and both showed a mask; they differed only in which outputs they
kept, and asking for coverage *and* a reusable mask meant two steps that
computed the geometry twice. They are one step now, and when coverage and a
default-labelled mask are both requested the intersection is computed once and
reused.

### Distance

How far each primary object is from its nearest partner, edge to edge:
`dist_um_<partner>`, the nearest partner's ID, and the closest-approach
coordinates the preview draws its connection lines from. Nothing else to
decide, so the form is just the two channels.

Overlap and distance are separate steps because they are separate
measurements, not separate outputs of one. Add both for the same pair and they
merge onto the same rows.

### Either side can be an earlier result

When a previous step left a mask — a kept overlap, or a size filter — it
appears in both dropdowns as "Previous result", so a chain like
*A∩B kept as mask → distance from C to that* is two steps with one form each.

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

The scope is recorded in `targets.txt` next to the results, one line per
image or region covered. A run started from the main window takes its scope
from the checked rows in the project tree, so it can mix full images and
regions freely; the older analyzer window runs every sample at one region.

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
├── targets.txt                     which images and regions it covered
├── MASTER_PER_OBJECT.csv           every sample's per-object rows, concatenated
├── MASTER_OVERLAP_SUMMARY.csv      every sample's sample-level overlap figures
└── <sample>/
    ├── per_object_<primary>.csv            one row per primary object
    ├── per_object_<partner>_coverage.csv   one row per partner object
    ├── per_overlap_region_<a>_in_<b>.csv   one row per overlap region
    ├── overlap_summary.csv                 one row per channel pair
    ├── pairwise_distances_<partner>.csv   (only if asked for)
    ├── intersection_<partner>.dat
    └── (intermediate masks from each mask-producing step)
```

Every file is named for **what one of its rows is**. That distinction matters:
"how much of A is inside B" and "how much of B is covered by A" are different
questions at different row grains, so they cannot share a table. A recipe with
several primaries simply writes one `per_object_<primary>.csv` each.

Channels are named by their number -- `Channel_0`, `Channel_1` -- in every
column header, mask filename and viewer layer. The channel index is assigned at
setup and is unique by construction, so these names are stable and cannot
collide. A channel folder's suffix, where it has one, is the label of the config
preset chosen at setup; it is not a channel identity, never updates if the
channel is reprocessed, and is the same for every channel whenever the reference
config was used, so it is not used for naming.

A region run nests one level deeper, `<sample>/<region>/`, so an analysis of the
full image and the same analysis of a region do not overwrite each other.

### `per_object_<primary>.csv` — one row per primary object

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

### `per_object_<partner>_coverage.csv` — one row per **partner** object

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

### `MASTER_PER_OBJECT.csv` and `MASTER_OVERLAP_SUMMARY.csv`

The per-object rows and the sample-level overlap rows, each concatenated across
every sample with a `sample_name` column. `MASTER_OVERLAP_SUMMARY.csv` is
normally the one to load for a cross-sample comparison of overlap;
`MASTER_PER_OBJECT.csv` is for per-object distributions.

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
