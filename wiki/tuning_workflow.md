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
    has no such control). This makes Step 2 a pass-through apart from its own
    minimum-size filter, which should be set equal to Step 1's Minimum Size or
    larger. You can then confirm Step 1's mask reaches Step 3 unaltered before
    deciding whether either correction is needed.

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
it removes bright junk at the cut face of a tissue block (**Edge Trim Distance**,
default `4.0`), and it corrects cells stretched along Z by anisotropic resolution
(**Z-Anisotropy Correction**, default `2`, **3D only**).

Step 2 also has its **own minimum-size filter** (**Min Size, Post-Trim**),
separate from Step 1's. While tuning, set it **equal to Step 1's Minimum Size or
larger** — never smaller. Step 1's filter has already removed everything below
its own threshold, so a smaller value here does nothing; a larger value is how
you raise the size floor once the mask is whole and you want to clear small
debris that survived Step 1.

When tuning a new image type, start by turning the two corrections off — **Edge
Trim Distance = 0** and, in 3D, **Z-Anisotropy Correction = 0** — so Step 2
passes Step 1's mask through unchanged (apart from that minimum-size filter). Then
add back only what the image needs:

*   **Edge artefacts** (a tissue block with a bright cut face): turn edge trimming
    on (distance > 0) and check the hull outline on screen wraps the tissue. If
    real cells near the edge vanish, raise the **Brightness Cutoff Factor** so
    bright (real) objects near the edge are protected, or lower the trim distance.
    If your image has no tissue edge — cultured cells, smears, and similar — there
    is nothing to trim, so leave this at 0.
*   **Z-stretching (3D only):** if cells look stretched into pillars along Z, raise
    **Z-Anisotropy Correction** to erode that stretch back. The **2D pipeline has
    no Z axis and no such control** — there is nothing to set.

If, after turning a correction on, Step 2 removes cells you wanted, adjust it here
rather than compensating in Step 1.

---

## Step 3: get one seed per cell

Step 3 places the seeds that Step 4 will build cells from, so it is worth getting
right before you look at Step 4. Full detail on
[the Step 3 page](soma_extraction.md).

The mental model: Step 3 looks for compact, roughly round, sufficiently thick
cores inside the mask and drops one seed per core. Two independent detectors feed
it — a **geometric** one (peeling by thickness) and an **intensity** one (bright
nuclei). You are tuning what counts as "a real cell body."

*   **Too few seeds → cells that should be distinct share one seed.** Loosen the
    shape/size gates: lower **Min Absolute Thickness**, raise **Max Aspect
    Ratio**, lower **Min Seed Size**. Add deeper peeling ratios to break thick
    necks between touching cells.
*   **Too many seeds → one cell gets several.** Tighten those same gates, and
    raise **Min Peak Separation** so two detections on one cell collapse to one.

Inspect the **Cell bodies** layer directly: you want one seed per cell, sitting
in the cell body. Good seeds make Step 4 much easier to judge.

> **Under the hood.** Thickness is the radius of the largest inscribed sphere
> (3D) or disc (2D) via the distance transform; aspect ratio is the elongation
> from a PCA of the core. Both carry over to 2D unchanged in meaning — one fewer
> dimension, same idea.

---

## Step 4: splitting and merging

With the seeds in good shape, Step 4's job is to split the mask into one cell per
seed and then merge back any pieces that were cut apart artificially. It affects
both the **boundaries** between cells and the final **count** (through its merge
pass and minimum-size filter), so judge it on the whole result. Full detail on
[the Step 4 page](cell_splitting.md).

*   Cells that should be separate are merged → the merge pass is too eager: raise
    **Min Path Intensity Ratio**, or lower **Max Seed Merge Distance** so distant
    seeds are not combined.
*   One cell cut by an unnatural straight line → the merge pass is too shy, or the
    cut ignored the visible dark gap: lower **Min Path Intensity Ratio**, or raise
    **Watershed Intensity Weight** above `0` so cuts snap to dark pixels.
*   Small real fragments disappearing → lower **Min Final Cell Size**.

If changing Step 4 cannot get the count right no matter how you set it, the seeds
themselves are likely wrong — return to Step 3 and re-check the **Cell bodies**
layer.

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

---

## Quick reference: what to change when

| Symptom in the final result | Most likely step | What to do |
| :--- | :--- | :--- |
| Background speckle counted as cells | Step 1 | Raise Minimum Size; raise detection threshold |
| Thin processes missing | Step 1 | Add smaller tubular scales; lower detection threshold |
| Real cells deleted near tissue edge | Step 2 | Raise Brightness Cutoff, lower Edge Trim Distance, or turn Step 2 off |
| Cells stretched along Z (3D) | Step 2 | Increase Z-Anisotropy Correction |
| **Two touching cells counted as one** | **Step 3 or 4** | Step 3: loosen shape/size gates, add deeper peeling ratios. Step 4: raise Min Path Intensity Ratio, lower Max Seed Merge Distance |
| **One cell counted as several** | **Step 3 or 4** | Step 3: tighten shape/size gates, raise Min Peak Separation. Step 4: check the split lines and merge behaviour |
| Cells merged despite good-looking seeds | Step 4 | Raise Min Path Intensity Ratio; lower Max Seed Merge Distance |
| One cell cut by a straight line | Step 4 | Lower Min Path Intensity Ratio; raise Watershed Intensity Weight |
| Impossibly high branch counts | Step 5 | Raise Prune Skeleton Spurs |
| Run takes too long, don't need all metrics | Step 5 | Turn off Calculate Distances / Skeletons |

When in doubt, work top-down: fix each step on screen before moving to the next,
and get the seeds right in Step 3 before judging Step 4.
