# Tuning Workflow: Calibrating HIBACHI for a New Image Type

This page is the one to read first if you are starting on a **new kind of image**
— a different stain, magnification, cell type, tissue, or microscope — and the
default parameters give you a poor result. It explains the *order* in which to
approach the pipeline so you are never guessing at forty parameters at once.

The per-step pages ([Step 1](initial_segmentation.md), [Step 2](remove_artifacts.md),
[Step 3](soma_extraction.md), [Step 4](cell_splitting.md), [Step 5](calculate_features.md))
tell you how to tune each step in isolation. This page tells you how the steps
fit together and which one to fix when something looks wrong.

You never edit a config file by hand. Everything here is done by clicking:
opening an image, entering parameter values in the step panel, pressing
**Process Current Step**, and — once a recipe works — saving it to your **Config
Library** and applying it to the rest of your images.

---

## How the steps relate

HIBACHI is a chain: each step consumes the previous step's output. A few
relationships between the steps are worth knowing before you start, because they
tell you *where* a given problem is most likely to originate.

**Steps 3 and 4 together determine the cell count.** Step 3 (Soma Extraction)
places one *seed* per detected cell body; Step 4 (Cell Separation) uses those
seeds to split the mask into individual cells, then runs a merge pass that can
recombine over-split pieces. Both steps affect how many objects you end up with:

*   Step 3 sets the **number of seeds** — too few and cells that should be
    distinct never get separate seeds; too many and one cell gets several.
*   Step 4 can **merge seeds back together** (its merge heuristics and the seed-
    merge distance) and can **drop fragments** below its minimum size.

So when a cell count looks wrong, the cause can be in either step, and the two
interact. The practical consequence is the tuning order below: get the seeds
right in Step 3 *first*, so that when you look at Step 4 you are judging its
splitting and merging on good input rather than compensating for bad seeds.

> **Under the hood.** Step 3 emits a label image of seeds. Step 4 runs a
> marker-controlled watershed using those markers, then a graph-based merge pass
> that collapses adjacent basins when the boundary between them is not a real
> valley. A final size filter and orphan-reassignment pass can also remove or
> absorb small fragments. Watershed itself never creates an object without a
> seed, but the merge and size passes downstream can still change the final
> count relative to the number of seeds.

---

## Work top-down, and freeze each step before moving on

The pipeline is a chain: each step consumes the previous step's output. Because a
step can only work with the mask it is given, tune **in order**, and do not move
to the next step until the current one looks right on screen.

The viewer supports this directly. **Process Current Step** runs just the step
you are on. **Back** and **Forward** move between steps without recomputing.
Re-processing a step you have already run **clears every result after it** (they
were computed from the old output and are now stale), so you can always change
your mind on an earlier step and the later steps will rebuild cleanly from it.

The recommended loop for a brand-new image type:

1.  **Get Step 1 to a clean binary mask.** Do not care about separating touching
    cells yet — only that real signal is captured and background is not. See the
    [Step 1 baseline](#step-1-first-get-a-clean-mask) below.
2.  **Turn Step 2's corrections off to start**, then add them only if you need
    them (edge artefacts to remove, or Z-stretching to correct). See the
    [Step 2 note](#step-2-edge-trimming-and-z-correction) below.
3.  **Get Step 3 to place one seed per cell.** The seed count strongly shapes the
    final cell count, so get this right before judging Step 4.
4.  **Then tune Step 4.** With good seeds, check that cells are split and merged
    correctly. Step 4 also influences the count (through merging and its size
    filter), so look at the whole result, not just the boundaries.
5.  **Step 5 is measurement, not segmentation.** Its only real knob is skeleton
    cleanup. Leave it until the mask is final.

---

## Start from the simplest possible pipeline

When an image type is unfamiliar, turn *off* as much as possible, confirm the
basics are in range, then add sophistication one layer at a time. A parameter
you never understood the effect of is a parameter you cannot trust.

Every step has a documented **"off" or neutral value** — see each step page.
The two that matter most at the start:

*   **Step 1's scale table is additive.** Start with a single **Scale = `0.0`**
    row — the plain-intensity pass that captures cell bodies — and get its
    threshold right first. Then *add* tubular scales on top for processes; each
    row only contributes more foreground, so you are layering, not switching
    between modes. Keep the Scale = `0.0` row in place as the base layer.
*   **Step 2 edge trimming and Z-correction ship on** in the default config.
    When first tuning a new image type, turn its corrections off: set **Edge Trim
    Distance = 0** (and, in 3D, **Z-Anisotropy Correction = 0** — the 2D pipeline
    has no such control). With both corrections off, this step doesn't modify the
    masks — nothing is trimmed, split, or eroded — so with the post-trim size
    filter kept at or below Step 1's value, Step 1's result passes straight
    through to Step 3. You can confirm that before deciding whether either
    correction is needed (see the note below on setting the post-trim filter once
    trimming is on).

Then add layers back in the order each step page prescribes. The general shape
is always: *get the crude version working → add the feature that fixes the
biggest remaining problem → re-inspect → repeat.*

---

## Step 1: first, get a clean mask

Full detail on [the Step 1 page](initial_segmentation.md); the short version:

1.  Start with a single **Scale = `0.0`** row — the plain-intensity pass that
    captures cell bodies. Keep it as the base layer.
2.  Adjust the **detection threshold** (the `low` value in that row) until the
    compact signal on screen matches the real cell bodies — background dark. Lower
    captures more; higher captures less. (Thin processes come in at step 4.)
3.  **Set Minimum Size as low as it will go.** While tuning Step 1, keep this at
    its minimum. Smoothing, gap-closing, and merging are not enabled yet, so real
    signal is still fragmented into small pieces — a high size filter here would
    delete those pieces before you have had a chance to reconnect them. Raise it
    later, only once the mask is whole. (Unit differs by pipeline: **3D is
    voxels**, **2D is pixels** — a value right for one is wildly wrong for the
    other.)
4.  **Add** tubular scales on top of the Scale = `0.0` row — rows near your real
    process radii (e.g. `1.0`, `1.5`, `3.0` µm). Because the rows combine
    additively, these bring in thin processes without removing the cell bodies the
    intensity pass already captured.
5.  Add the finishing features last: per-scale smoothing and gap-closing, the
    optional seed gate, and orientation trace-linking. Each is described on the
    Step 1 page with the value that disables it.

Do not proceed until the binary mask on screen contains your cells and little
else. Everything downstream inherits this.

---

## Step 2: edge trimming and Z-correction

Step 2 does two independent things, and **both ship on** in the default config:
it removes accumulated stain along a tissue or coverslip edge (**Edge Trim
Distance**, default `4.0`), and it corrects cells stretched along Z by
anisotropic resolution (**Z-Anisotropy Correction**, default `2`, **3D only**).

Step 2 also has its own minimum-size filter (Min Size, Post-Trim), separate from
Step 1's and applied after trimming and Z-erosion have run. Because trimming
removes voxels from existing masks, a mask that passed Step 1's size filter can
be cut down below that size here, so a second filter is needed to clear the
under-size remnants left behind. Set it high enough to drop true debris but not
so high that legitimately small trimmed cells disappear. Setting it below Step
1's value can still remove masks if any trimming was done. Keeping it at the
level of step 1 or below is a safe option if no trimming is enabled.

When tuning a new image type, start by turning the two corrections off — **Edge
Trim Distance = 0** and, in 3D, **Z-Anisotropy Correction = 0** — so Step 2
passes Step 1's mask through unchanged (apart from that minimum-size filter).
Then work in the order below: **get the hull enclosing the tissue first**, because
the trim distance is measured from the hull boundary — until the hull correctly
outlines the tissue, neither edge trimming nor Z-correction can be judged.

### First: shape the tissue hull

Edge trimming works by first building a **tissue hull** — a solid outline of where
the real tissue is — and then trimming masks that sit more than **Edge Trim
Distance** outside that boundary. So before you touch the trim distance, the hull
itself has to be right: it should hug the true tissue outline, neither cutting
into the tissue nor ballooning out to the edges of the image volume. Two
parameters control its shape, and you tune them by watching the **Edge Mask**
layer in the viewer, which draws the hull boundary.

**Otsu Scale Factor** (default `0.8`) sets *what unsegmented intensity counts as
tissue*. The step computes an automatic tissue threshold (a log-space Otsu
threshold) and multiplies it by this factor; the tissue region is then everything
above that threshold **plus all already-segmented cells** (the two are OR'd
together). So the factor only governs how much *dim, unsegmented* material — halo,
faint tissue, background glow — is pulled into the hull. Segmented cells are
inside the hull no matter what the factor is.

*   **Too low** (well below `1.0`) — the threshold drops, so dim background and
    halo around the tissue get counted as tissue; the hull inflates outward and
    can **snap toward the volume edges**, defeating the point of trimming.
*   **Too high** — the threshold rises, so only bright unsegmented material is
    added; the hull tightens toward the segmented cells and any real-but-dim
    tissue between or around them stops being counted, so the outline can pull
    **inward** and leave dim edges of the tissue exposed to trimming.
*   **Tuning:** if the hull hugs the image borders instead of the tissue, raise
    this toward and past `1.0`; if it pulls in and leaves dim tissue outside,
    lower it. Move in small steps (±0.1) and re-check the **Edge Mask** layer.

**Hull Closing Radius** sets *how far across open gaps the hull reaches to stay
one solid piece along its border*. After the tissue mask is downsampled (4×), the
step dilates it slightly, then morphologically closes it with a disk of this
radius, and finally fills any fully-enclosed holes. Two consequences matter for
tuning:

*   Because holes are filled afterwards, this radius does **not** control gaps
    *inside* the tissue — those get filled regardless. It controls the
    **concavities open to the boundary**: notches and bays around the tissue's
    outer edge.
*   Because it acts on the 4×-downsampled mask, each unit of radius is roughly
    four original pixels, so small changes have a visible effect.

*   **Too small** — the hull traces every notch and inlet along the tissue border
    and can leave the outline broken into separate islands where cells are sparse,
    so parts of the real tissue fall outside it.
*   **Too large** — the closing bridges across wide open gaps at the border and
    pushes the outline **outward past the true tissue edge**, wrapping adjacent
    background as if it were tissue.
*   **Tuning:** increase it until the outline is a single continuous shape that
    follows the tissue border without breaking into islands; stop before it starts
    cutting straight across genuine bays and pulling the boundary outward. The
    shipped defaults differ by pipeline — **3D uses `1`, 2D uses `10`** — because
    the 2D hull is built from a single plane, where sparser in-plane coverage
    needs more bridging to close up; treat those as starting points, not targets.

The goal is an **Edge Mask** outline that traces the real tissue edge — snug,
continuous, and neither eating into the tissue nor snapping out to the borders of
the image. Only once it looks right should you move on to removing noise.

### Then: remove noise near the edges

With the hull enclosing the tissue, add back the corrections the image needs:

*   **Edge artefacts** (a tissue/coverslip edge with accumulated stain): turn edge
    trimming on (distance > 0). If real cells near the edge vanish, **lower** the
    **Brightness Cutoff Factor**: this parameter sets a brightness bar a voxel
    must exceed to be treated as a protected "core" (which, with a small margin
    around it, is spared from trimming). A lower factor lets more of the real
    signal clear that bar, so bright cells near the edge are protected; a higher
    factor protects less and trims more. (Note the shipped defaults differ sharply
    — 2D uses `1.5`, while 3D uses a very large value that puts the bar out of
    reach, effectively disabling core protection so trimming is purely
    distance-based. Lower the 3D value into a sensible range if you need edge
    cells protected.) Alternatively, lower the trim distance. If your image has no
    such edge — cultured cells, smears, and similar — there is nothing to trim, so
    leave the distance at 0.
*   **Z-stretching (3D only):** if cells look stretched into pillars along Z, raise
    **Z-Anisotropy Correction** to erode that stretch back. The **2D pipeline has
    no Z axis and no such control** — there is nothing to set.

If, after turning a correction on, Step 2 removes cells you wanted, adjust it here
rather than compensating in Step 1.

---

## Step 3: get one seed per cell

Step 3 places the **seeds** that Step 4 builds cells from, so it is worth getting
right before you look at Step 4. It scans each Step 2 mask for compact,
sufficiently thick, roughly round cores and drops one seed per core. Full detail
on [the Step 3 page](soma_extraction.md).

This is the most involved step to tune, so it has its own approach: **start
permissive so you find as many candidate somas as possible, then narrow down to
one per cell.** The step prints a diagnostic breakdown every run, so you tighten
by reading *why* cores were rejected — not by guessing why a cell came out bare,
which is the most frustrating way to debug this step.

### Pick a seeding mode first: ratios *or* percentiles

Step 3 has two ways to find cores, and in practice **you use one or the other,
not both.** They are exposed as two lists:

*   **Ratios to Process** (distance-transform peeling) — thresholds the cores by
    *thickness*: it keeps the voxels whose distance from the object surface is at
    least `ratio × (thickest point)`. It excels when cells are joined by **thin
    connections**, because the distance transform naturally pinches at a thin neck
    and separates the two bulbs.
*   **Intensity Percentiles to Process** — thresholds the cores by *brightness*:
    it keeps the pixels above a given intensity percentile. It is harder to tune
    but works far better for **tightly packed cells with thick interfaces**, where
    there is no thin neck for the distance transform to catch but the cell centres
    are still distinctly brighter than the shared border.

They are not additive. Internally, intensity strategies always outrank ratio
strategies when their cores overlap, so **if you specify any percentiles, they
override the ratios wherever the two compete** — the ratios only survive in space
no percentile claimed. Because of that, treat it as a mode switch: use ratios and
leave percentiles empty when your cells separate at thin necks; use percentiles
(and effectively ignore the ratios) when they don't. The shipped config populates
both lists, which means percentiles are doing the work by default.

**How the thresholds behave** — this is the crux of tuning either list:

*   **Ratio:** a *lower* ratio uses a lower thickness bar, so each core grows
    **larger** and reaches further from the centre; a *higher* ratio keeps only
    the thick core right at the centre. Listing several ratios lets the step try
    each and keep the best-separated result.
*   **Percentile:** a *lower* percentile includes **dimmer pixels, so cores are
    larger**; a *higher* percentile keeps only the **brightest pixels, so cores
    are much smaller**. To separate two touching cells, you go higher until each
    centre resolves into its own small bright core; to stop a single cell
    fragmenting, you go lower so its core stays whole.

### Then narrow with the gates — permissive first

Every candidate core, from whichever mode, must pass a series of gates before it
becomes a seed. Tune these from **loose to tight**: start so that essentially
every real cell produces a seed (accepting some extras), then tighten one gate at
a time, using the printed rejection counts to see which gate to touch.

*   **Min Seed Size** (`min_fragment_size`) — the smallest a core may be. Start
    low so small or dim somata are not dropped; raise it only if noise specks are
    seeding. Watched by the *Too Small* rejection count.
*   **Min / Max Absolute Thickness** — the accepted thickness window for a core.
    Start with a wide window (low min, high max) so real somata of varying size
    all qualify; narrow it to exclude thin processes (raise min) or huge blobs
    (lower max). Watched by the *Thickness Bound* count. (The max bound does not
    discard a fragment outright — it keeps the inner sub-core within the window.)
*   **Max Aspect Ratio** — how elongated a core may be before it is treated as a
    process rather than a soma. Start high (permissive) so compact somata are
    never rejected; lower it to reject stretched, worm-like detections. Watched by
    the *Aspect Ratio* count.
*   **Min Peak Separation** — the minimum physical distance between two seeds. If
    one cell keeps getting several seeds, raise this to merge the duplicates; if
    two genuinely close cells collapse into one seed, lower it. Watched by the
    *Spatial Overlap* count, and by the `[TRAP]` log line that fires whenever one
    mask receives multiple somas.

*   **Soma Erosion Iterations** — leave this at **0** almost always. It erodes
    every core before detection, which is a blunt, whole-image sledgehammer; it
    helps only in rare cases and otherwise just shrinks or destroys good seeds.

### Reading the result

Inspect the **Cell bodies** layer: the target is exactly one seed per cell,
sitting in the cell body. When a cell is missing its seed, do not guess — look at
the diagnostic counts printed for the run (*Too Small*, *Thickness Bound*,
*Aspect Ratio*, *Spatial Overlap*) to see which gate removed it, then loosen that
one gate. When a cell has several seeds, raise **Min Peak Separation** or tighten
the mode threshold (a higher percentile / higher ratio) so its centre resolves as
one core.

**Which errors Step 4 can rescue, and which it cannot.** Aim to get as close to
one-seed-per-cell as the image allows, but realistically a few cells per image
will not seed perfectly. The two error types are not equal:

*   A cell that ends up with an **extra seed or two** can be rescued downstream —
    Step 4 re-merges over-split cells, so a cell that gets two seeds can be put
    back together there.
*   A cell with **no seed cannot be recovered at all.** Step 4 only splits and
    merges around existing seeds; it never invents one. A cell with no soma is
    simply absent from the result.

So when you have to choose, err toward **too many seeds rather than too few** —
lean permissive. It is still important to get seeding as accurate as the image
permits (every extra seed is work for Step 4 and a chance for it to merge
wrongly), but given an unavoidable error, an extra seed is fixable and a missing
one is fatal.

> **Under the hood.** Thickness is the radius of the largest inscribed sphere
> (3D) or disc (2D) from the distance transform; aspect ratio is the elongation
> from a PCA of the core. Candidates are scored (intensity strategies above
> distance-transform ones) and placed greedily, so a higher-priority core claims
> its territory and lower-priority cores overlapping it are dropped — this is the
> mechanism behind percentiles overriding ratios. Both carry over to 2D unchanged
> in meaning, one fewer dimension.

---

## Step 4: splitting and merging

Step 4 takes each cell that received more than one seed in Step 3, **splits it
into one basin per seed** (a watershed cut), and then **re-merges** any basins
whose dividing line does not look like a real boundary between two cells. So it
always separates first and then selectively undoes separations — that ordering is
the key to tuning it. Full detail on [the Step 4 page](cell_splitting.md).

### First: get complete separation

Start by making sure **every cell is fully separated** — accept over-splitting for
now. The two levers that decide whether a split is kept or undone are **Min Path
Intensity Ratio** and **Min Local Intensity Diff**; both are merge thresholds, and
at their permissive settings every watershed cut is kept.

Set **Min Path Intensity Ratio high — around `6`** — and **Min Local Intensity
Diff to `0`**. Together these are the permissive extreme: at ratio `6` almost no
interface counts as bright-enough-to-merge, and at diff `0` the local-contrast
check always passes (any difference clears a zero bar), so neither lever merges
anything and every watershed cut is kept. Process and look at the result. If every
cell is cleanly separated and no cell was wrongly split, you are done with Step 4;
move on.

### Then: rescue over-split cells

If some cells were split when they should not have been — the over-seeded cells
from Step 3 — move those two levers off their permissive extremes to re-merge the
bad splits without disturbing the good ones (lower Min Path Intensity Ratio, raise
Min Local Intensity Diff). They work on **different, largely orthogonal signals**,
which is what lets you fix most cases:

*   **Min Path Intensity Ratio** (valley depth) — compares the brightness *at the
    dividing interface* to the brightness of the cell's *somata*. A split is kept
    only when the interface is dark enough relative to the soma peaks (a genuine
    valley between two cells). **Lower it from ~`6` toward `<1`** and the bar for
    "deep enough valley" tightens, so more and more interfaces are judged not-a-
    real-boundary and their basins are **re-merged**. This is your primary rescue
    dial: lower it until the erroneously split cells rejoin, but stop before
    genuinely distinct cells start merging.
*   **Min Local Intensity Diff** (local contrast) — compares how different the two
    basins look *from each other* in a neighbourhood around the interface. A split
    is kept only when the two sides differ by at least this fraction; **raising
    it** demands more contrast to stay separate, so basins that look locally
    similar get **re-merged**. Because it looks at basin-to-basin contrast rather
    than interface-to-soma valley depth, it catches over-splits that the valley-
    depth lever misses — use it as the second lever when lowering Min Path
    Intensity Ratio alone over- or under-merges.

The two are additive in effect: a split is undone if **either** lever judges the
boundary unreal, so tune them together — Min Path Intensity Ratio first to do the
bulk of the rescuing, then Min Local Intensity Diff to catch the cases it leaves
behind. Watch the **Final segmentation** layer: the goal is that the truly
distinct cells stay separated while the mistakenly split ones (a handful of
over-seeded cells) merge back into single cells.

### Other Step 4 controls

*   **Watershed Intensity Weight** — `0` cuts on geometry alone; raising it above
    `0` biases cuts toward dark pixels so the dividing line snaps to a visible
    intensity valley rather than a straight geometric midline. Raise it if splits
    land in the wrong place (across bright tissue) rather than at the dark gap.
*   **Max Seed Merge Distance** — an upper bound on how far apart two seeds can be
    and still be considered for merging; lower it to stop distant seeds being
    combined.
*   **Min Final Cell Size** — drops finished cells below this size. Lower it if
    small real cells vanish; raise it to clear debris. Unit is voxels (3D) /
    pixels (2D).

If no setting of these levers gets the count right, the seeds themselves are the
problem — return to Step 3 and re-check the **Cell bodies** layer, remembering
that a *missing* seed can only be fixed there, never here.

---

## Step 5: measurement

Step 5 computes the numbers; it does not change the segmentation. Its one tuning
knob is **Prune Skeleton Spurs**, which cleans hair-like artefacts off the
skeleton (raise it if branch counts look impossibly high). The two **Calculate
Distances** / **Calculate Skeletons** toggles are performance switches — turn
them off if you do not need those metrics and want the run to finish faster. See
[the Step 5 page](calculate_features.md).

---

## Once it works: save it and apply it to everything

Tuning is done on **one representative image**, opened from the project view.
When that image segments well, you capture the recipe so you never re-tune it:

1.  **Save the run as a reusable config.** Use **⬇ Export run config** in the
    project view to turn the parameters that just worked into a named entry in
    your **Config Library** (the 📚 button). The library lives outside any single
    project, so it is shared across all your work and survives updates.
2.  **Apply it to the rest of your images.** Check the other images of the same
    channel, click **⚙ Set New Channel Config…**, and pick your saved config. The
    app applies those parameters to every checked image while **always preserving
    each image's own physical dimensions**. If any of those images were already
    processed with different parameters, their now-stale results are cleared so
    they reopen unprocessed.
3.  **Batch-process.** Check the images and use **Process Selected** to run the
    whole pipeline on all of them unattended.

This is the intended arc: **tune once, interactively, on one image → save to the
Config Library → apply to the batch.** You should never have to open or edit a
`.yaml` file to do any of it.

> **Note on pipeline versions.** If you apply a config that was tuned on an older
> version of HIBACHI, the app reconciles it against the current pipeline and
> **shows you exactly what changed** (a parameter added, removed, or clamped into
> range) before writing anything. Nothing is altered silently.

