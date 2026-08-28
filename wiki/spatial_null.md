# Spatial Null

**Corresponding modules:** `utils/spatial_null/`
*   `engine.py` — the domain, the placement, the statistics
*   `runner.py` — per-project execution and run parameters
*   `dialog.py` — the setup window
*   `export.py` — artifacts and manifest
*   `qc_render.py` — per-draw QC images
*   `hibachi_null_io.py` — loading runs for downstream inference

Reached from **🎲 Spatial Null (randomise masks)** in the
[Cross-Channel Analyzer](cross_channel_analysis.md), seeded from the current
recipe.

## The question it answers

Cross-channel analysis tells you that objects in one channel sit, on average, a
certain distance from objects in another. It does not tell you whether that
distance means anything: objects confined to a tissue hull are near each other
simply because they are all in the same place.

The spatial null answers that by re-placing **the actual segmented masks** at
random inside the same domain, many times, and asking where the real measurement
falls in that distribution.

Method after Andrey et al. 2010 (*PLoS Comput Biol* 6:e1000853), with the
substitutions needed for extended objects rather than points.

## Why the masks are reused rather than modelled

Each object keeps its own mask exactly. Per draw it gets a Haar-uniform random
**rotation** — not a reflection, which would change chirality — applied in
physical space so anisotropic spacing is honoured, resampled nearest-neighbour,
then repaired to its exact original voxel count. It is then dropped at a
uniform-random position lying wholly inside the domain, rejecting sites that
collide with an already-placed object.

Because size and shape are preserved exactly, the object volume fraction is
identical in every draw. F(0) therefore matches by construction, and the
comparison isolates **arrangement** instead of confounding it with abundance —
which a point- or sphere-based null cannot guarantee.

## The domain

The region objects may be placed in. Choices:

| `domain_choice` | Domain |
| :--- | :--- |
| `hull` | The filled tissue hull (default) |
| `field` | The whole image |
| `parent_a`, `parent_b`, `parent_both` | Built from another channel's masks |

The hull is reconstructed from the one-voxel boundary shell that Step 2 persists
(`<mode>_edge_mask.dat`). That reconstruction is exact for hulls this pipeline
produces, because both hull generators end their per-slice work with
`binary_fill_holes`, so no enclosed void can exist and shell-filling has nothing
to over-fill. With edge trimming disabled the shell is all zeros, and the domain
falls back to the whole field.

An ROI applies the same crop the masks get — bounding box **and** per-slice
polygon — so a region behaves as a smaller copy of the full field.

The `parent_*` options are a different hypothesis: they ask whether objects are
arranged non-randomly *within another structure*, rather than within the tissue.

## The statistics

All on the same edge-to-edge geometry the rest of HIBACHI uses.

*   **Cross** — per-object distance to the nearest object of a fixed partner
    channel. The partner never moves.
*   **F (empty space)** — distance from every domain voxel to the nearest object
    surface. Every domain voxel is used, so there is no sampling noise and no
    number-of-points parameter to choose.
*   **G** — nearest-neighbour distance between objects, surface to surface.

Cross-distances are computed and exported **both ways**, because they are
different quantities rather than two views of one. The median over aggregates of
the distance to the nearest microglion is not the median over microglia of the
distance to the nearest aggregate: they weight by different populations and
diverge whenever the two counts differ. Exporting both means the framing can be
revisited without recomputing.

## Two independent Monte-Carlo sets

The run draws two sets: the first estimates the reference curve, the second the
spread around it. Defaults are 199 each.

Reusing one set for both purposes shrinks the spread — every draw helped define
the mean it is then compared against — and inflates significance.

The per-sample index is a **mid-p Monte-Carlo rank** (SDI), which is uniform on
(0,1) under the null. That is what makes the population-level KS test valid.

## Diagnostics

Reported per sample: **occupancy fraction**, **placement rejection rate** and
**accepted-orientation rate**.

At high occupancy a hardcore null is *forced* toward regularity, because there is
nowhere else to put things. An apparently significant result can then be a
packing artefact rather than biology, and a high rejection rate is the signature
of that regime. These are diagnostics, not decoration.

## Parameters

Set in the dialog, recorded verbatim in the manifest. The ones that change the
question being asked:

| Parameter | Default | Effect |
| :--- | :--- | :--- |
| `n_reference`, `n_test` | 199, 199 | Draws in each independent set |
| `domain_choice` | `hull` | Where objects may go (above) |
| `rotate` | on | Random orientation per draw |
| `hardcore` | on | Reject overlapping placements |
| `min_separation_um` | 0 | Additional enforced gap |
| `erode_um` | 0 | Shrink the domain from its edge |
| `compute_f`, `compute_g` | on | Which statistic families to compute |
| `cross_statistic` | `median` | Summary of the cross distances |
| `measure_from` | `both` | Which direction(s) to measure |
| `statistic_direction` | `primary` | Which direction drives the index and QC |
| `max_attempts` | 2000 | Placement attempts before giving up |
| `seed` | 0 | Reproducibility |
| `roi_name` | — | Restrict to a saved region |
| `also_csv` | off | Also write the bulk arrays as gzipped CSV |
| `n_qc_images` | 0 | QC JPGs per draw |
| `run_name` | — | Names this pairing on disk |

`n_qc_images` defaults to 0 because the count multiplies by samples: 398 draws
across 20 images is roughly 8,000 files. The dialog warns with an estimate.

`run_name` exists so one project can hold several pairings — randomise A against
C, B against C, A inside B — side by side.

## Outputs

Roughly 30–40 MB for 20 images × 200 objects × 398 draws:

| File | Contents |
| :--- | :--- |
| `manifest.json` | Schema version, every parameter, comparability key |
| `image_metadata.csv` | One row per image: geometry, diagnostics, indices |
| `observed_objects.csv` | One row per real randomised-channel object |
| `null_objects.npz` | One row per (image, draw, object) — the bulk |
| `observed_partners.csv` | One row per fixed partner object |
| `null_partners.npz` | The reverse direction's bulk |
| `f_curves.npz` | F CDFs on the run's shared grid |

Setting `also_csv` additionally writes `null_objects.csv.gz` and
`null_partners.csv.gz` — the same bulk data as gzipped CSV, for reading outside
Python.

`.npz` for bulk arrays and `.csv` for small tables — deliberately not parquet,
because `pyarrow` is not in the validated environment and adding it would trigger
an environment rebuild for every user.

Full pairwise distances are deliberately **not** exported: no statistic here uses
them, they would run to ~158M rows on the null side, and exporting them for the
observed data alone would give a statistic with no null to compare against.

With QC enabled, one JPG per draw plus one of the observed data per sample, with
the measured segments annotated.

## Downstream inference

`hibachi_null_io.py` is the notebook-facing layer, for work across runs and
projects:

| Function | Purpose |
| :--- | :--- |
| `discover_runs()` | Find runs under one or more project roots |
| `matching_runs()` | Select runs for a given pairing |
| `load_projects()` | Load several into one dataset, checking comparability |
| `per_image_effects()` | Per-image effect sizes and indices |
| `replicate_test()` | Test consistency across replicates |
| `compare_groups()` | Compare between experimental groups |
| `quality_report()` | The diagnostics table, per image |

`load_projects()` checks the manifests against each other and reports where runs
were made with parameters that make them non-comparable, rather than pooling them
silently.

---

## Where to go next

*   [Cross-Channel Analyzer](cross_channel_analysis.md) — producing the objects
    this randomises.
*   [Sub-Regions](roi_regions.md) — restricting a run to a drawn region.
*   [Step 2](remove_artifacts.md) — the edge mask the hull domain is rebuilt from.
