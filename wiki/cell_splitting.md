# Step 4: Cell Separation

**Corresponding modules:**
*   **3D:** `utils/module_3d/cell_splitting.py`
*   **2D:** `utils/module_2d/cell_splitting_2d.py`

## What this step decides

Step 4 takes the seeds from Step 3 and turns the mask into individual cells. It
acts **only on cells that received more than one seed** — a cell with a single
seed passes through untouched. For each multi-seed cell it does two things, in
order:

1.  **Splits** the cell into one region per seed (a marker-controlled watershed).
2.  **Re-merges** any split that does not look like a real boundary between two
    cells, using two intensity-based tests.

That split-then-merge order is the whole basis for tuning this step: every cell
is first cut apart at its seeds, and then the merge tests decide which cuts to
keep. So you tune by first guaranteeing full separation, then relaxing the merge
tests to rejoin only the cells that were split by mistake (the over-seeded cells
from Step 3).

> **Under the hood.** The watershed floods from the seed markers over a landscape
> built from the distance transform (optionally modulated by intensity). Adjacent
> basins are then assembled into a graph, and each shared interface is scored by
> two independent metrics; a basin pair is merged back together if either metric
> says the boundary is not a genuine valley between two cells. Two clean-up passes
> then run on the whole image: seedless fragments are handed back to a touching
> neighbour, and any resulting cell below **Min Final Cell Size** is merged into
> its most-contacted neighbour. Neither pass ever deletes anything.

---

## The two merge levers

Both control whether a watershed cut is **kept** or **undone**, from different
angles. Understanding the direction of each is the key to Step 4.

**Min Path Intensity Ratio** — *valley depth.* Compares the intensity **at the
dividing interface** to the intensity of the cell's **somata**. A cut is kept
separate only when the interface is dark enough relative to the soma peaks (a real
valley between two cells).

*   A **high** value (e.g. `6`) makes almost every cut count as a valley, so
    nothing is merged back — separation is effectively guaranteed.
*   **Lowering** it toward `<1` tightens the "deep enough" bar, so more and more
    interfaces are judged not-a-real-boundary and their basins are **re-merged**.

**Min Local Intensity Diff** — *local contrast.* Compares how different the two
basins look **from each other** in a neighbourhood around the interface. A cut is
kept only when the two sides differ by at least this fraction.

*   A value of **`0`** means any difference passes, so this test never merges
    anything.
*   **Raising** it demands more contrast to stay separate, so basins that look
    locally similar get **re-merged**.

A cut is undone if **either** lever judges the boundary unreal, so they work
together. They are largely orthogonal — valley-depth looks at interface-vs-soma,
local-contrast looks at basin-vs-basin — which is what lets you fix most cases
with two dials.

A third mechanism can also undo a cut, after both levers have had their say:
**Min Final Cell Size** merges any resulting cell that is too small to be a cell
at all (see *Other controls* below). It has the last word, so if a split you
expected to survive keeps disappearing, check that floor before re-tuning the
levers.

---

## 🔧 Tuning this step

### Step A — guarantee full separation first

Start by making every cell fully separate, accepting over-splitting for now:

*   **Min Path Intensity Ratio = `6`** (or similarly high) — no cut is merged back.
*   **Min Local Intensity Diff = `0`** — the contrast test never merges.

At these permissive extremes every watershed cut is kept. Process and look at the
**Final segmentation** layer. If every cell is cleanly separated and nothing was
wrongly split, you are done with Step 4.

### Step B — rescue the cells that were split by mistake

If some cells were split when they should not have been (the over-seeded cells
from Step 3), bring the two levers off their extremes to re-merge just those:

*   **Lower Min Path Intensity Ratio** from ~`6` toward `<1`. This is the primary
    rescue dial — lower it until the mistakenly split cells rejoin, and stop before
    genuinely distinct cells begin merging.
*   **Raise Min Local Intensity Diff** from `0`. Use it as the second lever for
    over-splits that the valley-depth dial misses, since it judges a different
    signal (basin-to-basin contrast).

Tune Min Path Intensity Ratio first to do the bulk of the rescuing, then Min Local
Intensity Diff for the remainder. Watch the **Final segmentation** layer: the goal
is that truly distinct cells stay separated while the mistakenly split ones merge
back into single cells.

### Other controls

*   **Watershed Intensity Weight** — `0` cuts on geometry alone (distance
    transform); raising it above `0` makes bright regions "flood" faster, biasing
    the dividing line toward dark pixels so the cut snaps to a visible intensity
    valley rather than a straight geometric midline. Raise it if cuts land across
    bright tissue instead of at the dark gap. (Shipped default: `0` in 3D, `0.5`
    in 2D.)
*   **Max Seed Merge Distance** — an upper bound (µm) on how far apart two seeds
    can be and still be considered for merging. Lower it to stop distant seeds from
    being combined.
*   **Local Analysis Radius** — the neighbourhood size (µm) used by the local-
    contrast test around an interface.
*   **Min Final Cell Size** — the size floor for a finished cell. After splitting
    and merging, any cell smaller than this is **merged into its most-contacted
    neighbouring cell**. This applies whether or not the fragment contains a seed:
    a basin that came out too small to be a cell is real signal belonging to the
    cell it was cut from, so it is given back rather than kept as a separate object.
    Unit is **voxels** (3D) / **pixels** (2D).

    **Nothing is ever deleted by this step.** A cell below the floor that has *no
    neighbouring cell to merge into* is kept at full size. That is what protects a
    genuinely small, well-segmented cell — including one that Step 3 gave no seed,
    which passes through this step untouched. The same rule covers the case where
    every basin of one cell is undersized: the merges cascade until a single label
    remains, and that label is then kept whole.

    Because the floor is applied after the merge tests, **it overrides them**: a
    boundary the two levers above deliberately kept is still merged if the
    resulting cell is undersized. Raise this value to suppress small spurious
    cells; lower it (or set it to `0`, which disables the pass entirely) if real
    small cells are being absorbed into their neighbours. Every merge is named in
    the process log as a `[PROFILE|UNDERSIZE]` line, so you can see exactly which
    cells were combined and why.

If no setting of the two levers gets the count right, the seeds are the problem —
return to [Step 3](soma_extraction.md) and re-check the **Cell bodies** layer,
remembering that a *missing* seed can only be fixed there, never here.

---

## Parameter reference

| Parameter | Type | 3D default | 2D default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Min Path Intensity Ratio** (`min_path_intensity_ratio`) | float | `0.6` | `0.5` | Valley-depth merge test. High = keep separate; low = merge. |
| **Min Local Intensity Diff** (`min_local_intensity_difference`) | float | `0.01` | `0.01` | Local-contrast merge test. `0` = never merge; raise = merge more. |
| **Watershed Intensity Weight** (`intensity_weight`) | float | `0.0` | `0.5` | `0` = geometry-only cuts; higher biases cuts toward dark pixels. |
| **Max Seed Merge Distance** (`max_seed_centroid_dist`) | float µm | `20.0` | `20.0` | Upper bound on seed separation considered for merging. |
| **Local Analysis Radius** (`local_analysis_radius`) | int µm | `10` | `10` | Neighbourhood for the local-contrast test. |
| **Min Final Cell Size** (`min_size_threshold`) | number | `20000` **voxels** | `400` **pixels** | Size floor for a finished cell. A smaller cell is **merged** into its most-contacted neighbour (seeded or not); one with no neighbour is kept whole. Never deletes. `0` disables. Overrides the two merge levers. |

> **Under the hood.** A small third signal (a cell-mean intensity ratio) acts only
> as a tiebreaker when a cut is borderline on the valley-depth test; it is not a
> user parameter. Chunked processing for large volumes is controlled by an
> internal `memmap_voxel_threshold` that only affects tiling, not the result.

---

## What you see in the viewer

*   **Final segmentation** — the finished per-cell labelling. This is what you
    inspect while tuning, and the input to Step 5.

## Outputs on disk

Written to `<image>_processed_<mode>/`:

*   `final_segmentation_<mode>.dat` — the separated per-cell mask.

---

## Where to go next

*   Cells are separated correctly → continue to
    [Step 5: Feature Calculation](calculate_features.md).
*   For how splitting interacts with seeding and the rest of the pipeline → see the
    [Tuning Workflow](tuning_workflow.md).
