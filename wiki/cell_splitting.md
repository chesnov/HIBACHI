# Step 4: Cell Separation

**Corresponding modules:**
*   **3D:** `utils/module_3d/cell_splitting.py`
*   **2D:** `utils/module_2d/cell_splitting_2d.py`
*   **Shared post-stitch passes:** `utils/module_3d/streaming_stats.py`,
    `utils/module_3d/streaming_passes.py` (used by both, dimension-agnostic)

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

The merge tests run in **two places**, and knowing which is which explains most
of what you will see in the log:

*   **Inside each chunk**, for interfaces where both basins own a soma in that
    same chunk.
*   **Once globally, after stitching**, for every interface between two finished
    cells — including the ones no chunk was able to judge.

The global pass is what rescues a cell that picked up several erroneous somata
when those somata happened to land in different chunks. No per-chunk test can see
that case; a chunk holding one soma has nothing to compare it against.

> **Under the hood.** The watershed floods from the seed markers over a landscape
> built from the distance transform, modulated by intensity. Adjacent basins are
> assembled into a graph and each shared interface is scored; a basin pair is
> merged if the boundary is judged not to be a genuine valley. After the chunks
> are stitched, one bounded-memory sweep collects per-label and per-interface
> aggregates, and three passes run on the assembled result: the global merge
> tests, then seedless-fragment reassignment, then the size floor. None of the
> three deletes anything.

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
appreciably darker than the cell body as a whole (threshold `0.85`, not exposed).
That pattern means the watershed made a geometric cut through bright tissue rather
than finding a valley — typical when two dim or tiny seeds sit close together in
the same lobe.

It is not an independent signal. It is the same interface measurement as the
valley-depth test under a different denominator, so it amounts to applying Min
Path Intensity Ratio at an adjusted threshold. The adjustment corrects for an
unrepresentative soma reference: when cell bodies are far brighter than the rest
of the cell, dividing by soma intensity makes every interface look like a deep
valley.

On its own it never merges anything. It can only turn a *keep* into a *merge* when
the valley-depth test passed but only barely — specifically when the valley ratio
is above **70% of your Min Path Intensity Ratio**. That band matters in practice:
at a ratio of `6` the band starts at `4.2`, which real interfaces never reach, so
the "set it to 6 to guarantee separation" recipe below is safe. At the shipped
`0.6` / `0.5` it can and does fire. When it does, the interface log line carries
`*** BRIGHT CUT ***`.

---

## When a cut is not scored inside its chunk

Two preconditions are checked before any intensity is measured in the per-chunk
tests. If either applies, the boundary is recorded as **KEEP** and the chunk's
levers never see it. Both appear in the log with a reason.

**Propagated interfaces.** A boundary where *neither* side has a soma inside the
current chunk is not judgeable there — it sits between two regions inherited from
neighbouring chunks. Scoring it would reliably flag a bright cut and merge, and
because merging pools the two cells' seeds, that one verdict would spread through
the stitcher and fuse two genuinely separate cells.
→ log: `KEEP: (a,b) (propagated_interface_not_scored)`.

**Max Seed Merge Distance** (`max_seed_centroid_dist`) — an upper bound in µm on
how far apart two somata can be and still be considered for merging. Two cell
bodies further apart than this cannot be one cell however unconvincing the
boundary looks, so the tests are skipped and the cut stands. Distances are the
smallest soma-to-soma distance across the interface, measured between whole-image
soma centroids, so a soma in a different part of the image is measured correctly.
→ log: `KEEP: (a,b) (seeds_beyond_max_merge_distance)`.

**Neither is a dead end.** Both kinds of interface are picked up by the global
pass below, which sees them with both cells whole. The distance bound is applied
there too — and that is where it does its real work, since with propagated
interfaces deferred, the global pass is where every long-range merge decision is
actually taken.

> **Setting Max Seed Merge Distance to a small non-zero value disables merging
> entirely.** `0` switches the bound off. A small positive value gates every
> candidate pair instead, and the global pass then reports
> `interfaces_tested=0` — no merge test runs anywhere in the run, and the
> over-seeding rescue silently does nothing. If a cell will not rejoin, read that
> summary line first.

---

## The global merge pass

After stitching, `global_merge_pass` walks every pair of adjacent finished cells,
in order of decreasing contact area, and applies the same three tests with the
same thresholds. It reads only each interface's own crop, so cost does not scale
with cell size.

Two quantities are better here than they can ever be inside a chunk, which is why
this is the authoritative pass:

*   **The soma reference** is the mean intensity of all somata belonging to the
    two cells, wherever in the image they sit.
*   **The cell mean** is over both cells in full, from the aggregates, not over
    whatever fell inside one chunk.

Merges are applied as a single relabel. It is a **one-shot pass**: pairs are
scored against the labels as they stand when the pass begins, and a merge does not
cause the surviving cell's other interfaces to be re-scored.

Log lines: `[PROFILE|GLOBALMERGE] MERGE a+b`, `[PROFILE|GLOBALMERGE] KEEP a+b`
with a reason, and a `[PROFILE|GLOBALMERGE|SUMMARY]` giving
`interfaces_tested`, `merged` and `beyond_max_merge_distance`.

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

**Min Final Cell Size** is the third lever here, and it works on a different
signal from the other two. Where they ask whether a boundary looks real, it asks
whether the resulting piece is large enough to be a cell at all — so it catches
over-splits the intensity tests miss, such as a thin sliver cut off a cell by a
weak spurious seed. Reach for it when a cell is being split into one plausible
piece and one obviously-too-small one. See its entry below.

### When the levers seem to do nothing

Check `[PROFILE|GLOBALMERGE|SUMMARY]` before moving a lever further.

*   `interfaces_tested=0` with a large `beyond_max_merge_distance` — **Max Seed
    Merge Distance** is gating everything. Set it to `0` or to a value comfortably
    above your real soma separations.
*   `interfaces_tested` high but `merged=0` — the tests are running and rejecting.
    That is a genuine lever question; lower Min Path Intensity Ratio.

### Other controls

*   **Watershed Intensity Weight** — `0` cuts on geometry alone (distance
    transform); raising it above `0` makes bright regions "flood" faster, biasing
    the dividing line toward dark pixels so the cut snaps to a visible intensity
    valley rather than a straight geometric midline. Raise it if cuts land across
    bright tissue instead of at the dark gap. (Shipped default: `0` in 3D, `0.5`
    in 2D.) Brightness is normalised against the cell's own soma intensities across
    the whole image rather than against each chunk's local maximum, so the same
    value behaves consistently from chunk to chunk.
*   **Max Seed Merge Distance** — described above. `0` disables.
*   **Local Analysis Radius** — the neighbourhood size used by the local-contrast
    test around an interface. Note that despite the "(um)" in its label this is
    applied as a radius in **voxels (3D) / pixels (2D)**, with no conversion for
    voxel size — so the physical size of the neighbourhood changes with your image
    resolution.
*   **Min Final Cell Size** — the size floor, and the **third lever against
    over-splitting**. Where the two intensity levers ask whether a boundary looks
    real, this one asks whether the resulting piece is big enough to be a cell.
    Any watershed basin below the floor is **merged into the neighbour it touches
    most**. Unit is **voxels** (3D) / **pixels** (2D).

    It applies to basins regardless of whether they own a soma. Every basin owns a
    soma — that is why it exists — so exempting soma-owning labels would disable
    the lever completely.

    **An original object is never affected.** A mask that arrived with one soma or
    none becomes one cell, full stop. That falls out of the rule rather than being
    special-cased: distinct objects in the Step 2 mask are separated by background,
    so only siblings from one split ever touch. A whole object has no neighbour to
    merge into and is kept at full size, whatever its size. **Nothing is ever
    deleted.**

    When several basins of one cell are all undersized the merges cascade until a
    single label remains, which then has no sibling left and is kept whole.

    Because the floor is applied after the merge tests, **it overrides them**: a
    boundary the intensity levers deliberately kept is still merged if the
    resulting basin is undersized. Raise it to collapse spurious over-splits;
    lower it, or set `0` to disable the pass, if genuinely small cells are being
    absorbed into their neighbours. The pass runs smallest-first and repeats until
    nothing changes, up to 10 rounds. Every merge is named in the log as
    `[PROFILE|UNDERSIZE]`.

    > `protect_seeded_cells` (internal, **default off**) exempts every soma-owning
    > label and therefore switches this lever off. The symptom is
    > `[PROFILE|UNDERSIZE|SUMMARY] merged=0` on an image that visibly needs
    > merging. Only turn it on to shield small-but-real cells from a threshold set
    > too high for the data, and prefer fixing the threshold instead.

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
| `[PROFILE|ORDER]` | The chunk grid, and that soma-bearing chunks are visited first. A `1, 1, 1` grid means a single chunk and no propagation at all. |
| `[PROFILE|SEED]` | Per-seed intensity and size relative to its peers, flagging `DIM` or `TINY` seeds. Seeds are never dropped for this, but a bad cut next to a flagged seed usually points back to Step 3. |
| `[PROFILE|LANDSCAPE]` | Which of the two landscapes was used for this cell in this chunk, and why. See *Two landscapes* below. |
| `[PROFILE|WS]` | The basins the watershed produced: label, size, mean intensity. Wildly lopsided basins are worth investigating. |
| `[PROFILE|WS|BOUNDARY]` | Per basin, the mean intensity along its boundary with a verdict of *dark valley* (the cut is where it should be) or *bright cut* (the cut is through tissue). |
| `[PROFILE|INTERFACE]` | Every scored interface with all three test values and the resulting decision, or the reason it was not scored. |
| `[PROFILE|GRAPH]` | A `MERGE` or `KEEP` verdict per boundary. A `KEEP` with a reason in brackets was never scored in this chunk. |
| `[PROFILE|ORPHAN]` | A seedless fragment with no differently-labelled neighbour, kept as an isolated satellite of its own cell. Never deleted. |
| `[PROFILE|STITCH]`, `[PROFILE|STITCH\|GEO]` | How overlapping chunks were reconciled, including whether a contested region had a real intensity valley. |
| `[PROFILE|GLOBALMERGE]` | The authoritative merge decisions, plus the summary described above. |
| `[PROFILE|ISLAND]`, `[PROFILE|UNDERSIZE]` | The two clean-up passes: which fragments were handed to a neighbour, and which cells were merged for being under the size floor. |
| `[PROFILE|CONSERVE]` | Foreground totals at each stage: `INPUT`, `POST-STITCH`, `POST-GLOBALMERGE`, `POST-ISLAND`, `POST-UNDERSIZE`, `OUTPUT`, and an `END-TO-END` delta. That delta should be zero; a loss is a bug worth reporting. |

---

## Parameter reference

| Parameter | Type | 3D default | 2D default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Min Path Intensity Ratio** (`min_path_intensity_ratio`) | float | `0.6` | `0.5` | Valley-depth merge test. High = keep separate; low = merge. |
| **Min Local Intensity Diff** (`min_local_intensity_difference`) | float | `0.01` | `0.01` | Local-contrast merge test. `0` = never merge; raise = merge more. |
| **Watershed Intensity Weight** (`intensity_weight`) | float | `0.0` | `0.5` | `0` = geometry-only cuts; higher biases cuts toward dark pixels. Distinct from Step 3's parameter of the same name. |
| **Max Seed Merge Distance** (`max_seed_centroid_dist`) | float µm | `20.0` | `20.0` | Somata further apart than this are never merged. **`0` disables; a small non-zero value gates everything.** |
| **Local Analysis Radius** (`local_analysis_radius`) | int | `10` | `10` | Neighbourhood for the local-contrast test. Applied in voxels/pixels despite the "(um)" label. |
| **Min Final Cell Size** (`min_size_threshold`) | number | `20000` **voxels** | `400` **pixels** | Size floor. A smaller unseeded cell is **merged** into its most-contacted neighbour; a seeded one, or one with no neighbour, is kept whole. Never deletes. `0` disables. |

### Internal parameters

Not exposed in the interface. Change these only in code.

| Name | Value | Effect |
| :--- | :--- | :--- |
| `max_interface_to_cell_mean_ratio` | `0.85` | Bright-cut threshold: interface intensity relative to the cell mean. |
| (borderline band) | `0.7 ×` Min Path Intensity Ratio | How close the valley-depth test must be to its threshold before the bright-cut check may override a keep. |
| `speed_power` | `1.5` | Exponent on the flooding speed; raises the cost of thin necks so cuts prefer them. |
| `protect_seeded_cells` | `True` | Exempt any label owning a soma from the size floor. |
| `chunk_shape` / `overlap` | `(128, 512, 512)` / `32` (3D), `(1024, 1024)` / `64` (2D) | Processing tile size. Unlike most tiling, these affect the result — see below. |
| `stats_block_shape` | `(128, 128, 128)` / `(128, 128)` | Block size for the post-stitch aggregate sweep. Memory only; does not affect the result. |
| `min_contact` | `1` | Minimum shared voxels for the global pass to consider a pair. |

`overlap` must be smaller than every entry of `chunk_shape`. A larger value is
clamped with a warning in the log, because an equal or larger overlap otherwise
leaves a zero or negative stride — which yields no chunks at all and an empty
result.

---

## Two landscapes, and why the cut sometimes looks angular

The watershed floods over one of two landscapes, chosen per cell per chunk. Which
one ran is printed on the `[PROFILE|LANDSCAPE]` line.

**Every identity has a soma in this chunk → `d_seeds / speed^p`.** `d_seeds` is
straight-line distance from the nearest marker. With point-like somata that is a
fair proxy, and it holds the cut near the geometric middle so one bright lobe
cannot flood the whole object and strand the other seed in a sliver. Inherited
markers from neighbouring chunks are **not used** in this case; all the evidence
needed is present.

**Some identity has no soma here → `1 / speed^p` (pure cost), with the
neighbours' labels as extra markers.** An inherited marker is a whole region, not
a point, so `d_seeds` measured from it puts the boundary at the Euclidean midpoint
between two regions — wherever the chunk seams happen to fall, typically in bright
tissue rather than at the dark contact. The pure cost field has no distance term,
so the boundary is set by the speed field alone and lands at the dark neck.

Both halves of that rule matter, and each fixes a distinct observed failure:

*   Using `d_seeds` with inherited region markers produced branches assigned to
    the wrong cell, with `WRONG (bright cut — Voronoi bias!)` on every affected
    boundary.
*   Using inherited markers *at all* when both somata were already present
    pre-claimed chunk-shaped territory, giving a straight, axis-aligned cut along
    the chunk face and a 5535/717 basin split on a cell that should have divided
    evenly.

**So if a cut looks unnaturally straight or angular**, check the
`[PROFILE|LANDSCAPE]` line for that cell and the basin sizes on the `[PROFILE|WS]`
line. A very lopsided split alongside `PURE COST` means the cut is being shaped by
chunk geometry rather than by the image.

> **Under the hood: chunking is part of the algorithm.** The image is always
> processed in overlapping chunks. Chunks holding two or more of a cell's somata
> are visited first and the sweep radiates outward, so boundaries are decided
> where the evidence is. A chunk that cannot decide inherits its neighbours'
> labels and seed tags as markers; first writer wins. The stitcher joins labels
> across seams, refusing to join two groups whose seed sets are disjoint, and
> resolves contested overlap with a small local watershed — or, when the contested
> region is not actually darker than either side, by giving it to the larger
> claimant rather than inventing a cut through bright tissue. Because chunk
> placement determines which boundaries are judgeable in-chunk, changing the chunk
> size can change the result; the global merge pass is what keeps the final merge
> decisions independent of it.

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
