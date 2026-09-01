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
    cells, using intensity-based tests.

That split-then-merge order is the whole basis for tuning this step: every cell
is first cut apart at its seeds, and then the merge tests decide which cuts to
keep. So you tune by first guaranteeing full separation, then relaxing the merge
tests to rejoin only the cells that were split by mistake (the over-seeded cells
from Step 3).

Not every cut reaches the merge tests. Each boundary is either **scored** by the
tests, or **kept without being scored** because one of two preconditions says the
tests cannot give a meaningful answer there (see *When a cut is not scored at
all*, below). A kept-unscored boundary always survives, whatever the levers say.

> **Under the hood.** The watershed floods from the seed markers over a landscape
> built from the distance transform (modulated by intensity). Adjacent basins are
> then assembled into a graph, and each shared interface is scored by intensity
> metrics; a basin pair is merged back together if the boundary is judged not to be
> a genuine valley between two cells. Two clean-up passes then run on the whole
> image: seedless fragments are handed back to a touching neighbour, and any
> resulting cell below **Min Final Cell Size** is merged into its most-contacted
> neighbour. Neither pass ever deletes anything.

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

### The bright-cut check

A third, internal test sits alongside them: it compares the interface intensity to
the **whole cell's mean** and flags the boundary when the interface is not
appreciably darker than the cell body as a whole (the threshold is `0.85`, not
user-exposed). That pattern means the watershed made a geometric cut straight
through bright tissue rather than finding a valley — typical when two dim or tiny
seeds sit close together in the same lobe.

On its own it never merges anything. It can only turn a *keep* into a *merge* when
the valley-depth test passed but only barely — specifically when the valley ratio
is above **70% of your Min Path Intensity Ratio**. That band matters in practice:
at a Min Path Intensity Ratio of `6` the band starts at `4.2`, which real
interfaces never reach, so the bright-cut check cannot fire and the "set it to 6 to
guarantee separation" recipe below is safe. At the shipped `0.6` / `0.5` it can and
does fire. When it does, the interface log line carries `*** BRIGHT CUT ***`.

---

## When a cut is not scored at all

Two preconditions are checked before any intensity is measured. If either applies,
the boundary is recorded as **KEEP** and the levers never see it. Both appear in
the log with a reason, so you are never left guessing why a lever had no effect.

**Max Seed Merge Distance** (`max_seed_centroid_dist`) — an upper bound in µm on
how far apart two somata can be and still be considered for merging. Two cell
bodies further apart than this cannot be one cell however unconvincing the
boundary between them looks, so the tests are skipped and the cut stands. This is
a safety rail on the levers rather than a lever itself: it bounds how much damage
a permissive Min Path Intensity Ratio can do. Distances are measured between soma
centroids across the whole image, so a soma in a different part of the image is
still measured correctly, and the smallest soma-to-soma distance across the
interface is the one used. Set it to `0` to remove the bound entirely.
→ log: `KEEP: (a,b) (seeds_beyond_max_merge_distance)`.

**Propagated interfaces** — the image is processed in overlapping chunks, and a
chunk can inherit a partly-resolved cell from a neighbour it has already finished.
A boundary where *neither* side has a soma inside the current chunk is not
judgeable there: it sits between two inherited regions and lands at their midpoint,
usually in bright tissue rather than at the true contact. Scoring it would reliably
flag a bright cut and merge, and because merging pools the two cells' seeds, that
one verdict would spread through the stitcher and fuse two genuinely separate cells
across the whole image. Such boundaries are therefore kept unscored, so merge
decisions only happen next to the somata that can justify them.
→ log: `KEEP: (a,b) (propagated_interface_not_scored)`.

The practical consequence is worth knowing: **if one real cell is over-seeded and
its two somata land in different chunks, the levers cannot rejoin it.** That case
has to be fixed in Step 3 by not producing the second seed. It is rare — it needs
the two spurious seeds to straddle a chunk boundary — but it is a real limit, and
it is one more reason to get seeding right before tuning here.

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

If a cell refuses to rejoin no matter how far you move the levers, check the log
for that pair before moving them further — a `KEEP` with a reason attached means
one of the two preconditions above is holding the cut, and no lever setting will
override it. If the reason is `seeds_beyond_max_merge_distance`, raise **Max Seed
Merge Distance**.

### Other controls

*   **Watershed Intensity Weight** — `0` cuts on geometry alone (distance
    transform); raising it above `0` makes bright regions "flood" faster, biasing
    the dividing line toward dark pixels so the cut snaps to a visible intensity
    valley rather than a straight geometric midline. Raise it if cuts land across
    bright tissue instead of at the dark gap. (Shipped default: `0` in 3D, `0.5`
    in 2D.) Brightness is normalised against the cell's own soma intensities across
    the whole image rather than against each chunk's local maximum, so the same
    value behaves consistently from chunk to chunk.
*   **Max Seed Merge Distance** — described above; the µm bound beyond which two
    somata are never merged.
*   **Local Analysis Radius** — the neighbourhood size used by the local-contrast
    test around an interface. Note that despite the "(um)" in its label this is
    applied as a radius in **voxels (3D) / pixels (2D)**, with no conversion for
    voxel size — so the physical size of the neighbourhood changes with your image
    resolution.
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
    boundary the two levers above deliberately kept — including one kept unscored —
    is still merged if the resulting cell is undersized. Raise this value to
    suppress small spurious cells; lower it (or set it to `0`, which disables the
    pass entirely) if real small cells are being absorbed into their neighbours.
    Every merge is named in the process log as a `[PROFILE|UNDERSIZE]` line. The
    pass runs smallest-first and repeats until nothing more changes, up to 20
    rounds; hitting that cap prints a warning and usually means the floor is very
    large relative to your cells.

If no setting of the two levers gets the count right, the seeds are the problem —
return to [Step 3](soma_extraction.md) and re-check the **Cell bodies** layer,
remembering that a *missing* seed can only be fixed there, never here.

---

## Reading the log

Step 4 is as diagnostics-driven as Step 3, but its evidence is in the process log
rather than in a summary table. When a result is not what you expected, read these
before changing a parameter. They appear in the order the step runs:

| Line | Tells you |
| :--- | :--- |
| `[PROFILE|SEED]` | Per-seed intensity and size relative to its peers, flagging `DIM` or `TINY` seeds. Seeds are never dropped for this, but a bad cut next to a flagged seed usually points back to Step 3. |
| `[PROFILE|WS]` | The basins the watershed produced: label, size, mean intensity. |
| `[PROFILE|WS|BOUNDARY]` | Per basin, the mean intensity along its boundary with a verdict of *dark valley* (the cut is where it should be) or *bright cut* (the cut is through tissue). |
| `[PROFILE|INTERFACE]` | Every scored interface with all three test values and the resulting merge decision. This is the line that tells you which lever to move. |
| `[PROFILE|GRAPH]` | A `MERGE` or `KEEP` verdict per boundary. A `KEEP` with a reason in brackets was never scored. |
| `[PROFILE|STITCH]`, `[PROFILE|STITCH|GEO]` | How overlapping chunks were reconciled, including whether a contested region had a real intensity valley. |
| `[PROFILE|ISLAND]`, `[PROFILE|UNDERSIZE]` | The two clean-up passes: which fragments were handed to a neighbour, and which cells were merged for being under the size floor. |
| `[PROFILE|CONSERVE]` | Foreground totals at five points from input to output. The end-to-end delta should be zero; a loss is a bug worth reporting. |

---

## Parameter reference

| Parameter | Type | 3D default | 2D default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Min Path Intensity Ratio** (`min_path_intensity_ratio`) | float | `0.6` | `0.5` | Valley-depth merge test. High = keep separate; low = merge. |
| **Min Local Intensity Diff** (`min_local_intensity_difference`) | float | `0.01` | `0.01` | Local-contrast merge test. `0` = never merge; raise = merge more. |
| **Watershed Intensity Weight** (`intensity_weight`) | float | `0.0` | `0.5` | `0` = geometry-only cuts; higher biases cuts toward dark pixels. Distinct from Step 3's parameter of the same name. |
| **Max Seed Merge Distance** (`max_seed_centroid_dist`) | float µm | `20.0` | `20.0` | Somata further apart than this are never merged; the interface is kept unscored. `0` removes the bound. |
| **Local Analysis Radius** (`local_analysis_radius`) | int | `10` | `10` | Neighbourhood for the local-contrast test. Applied in voxels/pixels despite the "(um)" label. |
| **Min Final Cell Size** (`min_size_threshold`) | number | `20000` **voxels** | `400` **pixels** | Size floor for a finished cell. A smaller cell is **merged** into its most-contacted neighbour (seeded or not); one with no neighbour is kept whole. Never deletes. `0` disables. Overrides every merge decision above. |

### Internal parameters

Not exposed in the interface, listed so the behaviour above is reproducible. Change
these only in code.

| Name | Value | Effect |
| :--- | :--- | :--- |
| `max_interface_to_cell_mean_ratio` | `0.85` | Bright-cut threshold: interface intensity relative to the cell mean. |
| (borderline band) | `0.7 ×` Min Path Intensity Ratio | How close the valley-depth test must be to its threshold before the bright-cut check may override a keep. |
| `speed_power` | `1.5` | Exponent on the flooding speed; raises the cost of thin necks so cuts prefer them. |
| `chunk_shape` / `overlap` | `(128, 512, 512)` / `32` | Processing tile size. Unlike most tiling, these affect the result — see below. |

> **Under the hood: chunking is part of the algorithm, not just memory management.**
> The image is always processed in overlapping chunks. Chunks holding two or more
> of a cell's somata are visited first and the sweep radiates outward from them, so
> boundaries are decided where the evidence is. Each chunk is given the labels its
> already-finished neighbours wrote, together with the seeds those labels belong to,
> and uses them as additional watershed markers — this is how a cut computed near
> the somata continues into chunks that hold none. First writer wins. The stitcher
> then joins labels across chunk seams, refusing to join two groups whose seed sets
> are disjoint, and resolves any contested overlap with a small local watershed
> using the same intensity-guided landscape (or, when the contested region is not
> actually darker than either side, by giving it to the larger claimant rather than
> inventing a cut through bright tissue). Because chunk placement determines which
> boundaries are judgeable, changing the chunk size can change the result.

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
