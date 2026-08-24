# Step 1: Raw Segmentation

**Corresponding modules:**
*   **3D:** `utils/module_3d/initial_3d_segmentation.py`
*   **2D:** `utils/module_2d/initial_2d_segmentation.py`

## What this step decides

Step 1 turns the raw grayscale image into a **binary mask** — every voxel/pixel
is either "cell" or "background." It does not separate touching cells yet (that
is Steps 3–4). Its only job is to capture the real signal, including thin
processes that a plain threshold would miss, while leaving background out.

Everything downstream inherits this mask. If Step 1 loses a cell, no later step
can recover it; if Step 1 floods the image with background noise, every later
step has to fight it. So the goal here is simple to state: **your cells present,
background absent.** Get that and move on — do not try to make touching cells
separate at this stage.

> **Under the hood.** Detection combines two ideas. A **tubular enhancement**
> filter (Frangi + Sato "vesselness", taking the stronger of the two per pixel)
> responds to tube-like structures — neurites, processes, vessels — and
> suppresses flat or blobby regions. The enhanced image is then **thresholded**
> to produce the mask. Multiple filter *scales* run independently and their
> masks are combined (logical OR), so thin and thick structures can each be
> caught at the scale that suits them. Because vesselness suppresses blobby
> regions, cell bodies are captured by a **Scale = `0.0`** row instead — a
> pass-through that thresholds raw intensity with no vesselness — merged in with
> the tubular rows. The minimum-size filter is applied **once, globally, after
> all scales are merged** — it is not per-scale.

---

## The parameter table (scale profiles)

Most of Step 1's controls live in one table, the **scale profile table**. Each
row is one filter scale and carries its own settings:

| Column | Meaning |
| :--- | :--- |
| **Scale** | Approximate radius (µm) of the tubular structure this row detects. `0.0` = the plain-intensity (non-tubular) pass, which captures compact signal such as cell bodies. Rows combine additively. |
| **low** | The **detection threshold** — what counts as foreground. In *Percentile* mode it is a percentile; in *Absolute* mode a fixed `0.0–1.0` level. |
| **high** | The optional **seed-gate** threshold (see below). `100` (percentile) or `1.0` (absolute) disables it. |
| **Smooth σ** | Per-scale pre-blur to suppress noise before this scale is analysed. `0` = off. |
| **Max Gap (µm)** | Per-scale gap-closing — bridges small breaks in this scale's mask. `0` = off. |

There are **two** tables — one for **Percentile** thresholds and one for
**Absolute** thresholds — and the **Threshold Mode** toggle chooses which is
live. You only ever edit the one that is showing. **Minimum Size** and **Trace
Gap** sit outside the table as single global controls.

> **Percentile vs Absolute — which to use.**
>
> *Percentile* sets the threshold from each image's **own** intensity
> distribution: a `low` of `95` keeps roughly the brightest 5% of that image.
> Because it re-derives the level per image, it absorbs brightness differences
> between images without re-tuning — but it also means every image keeps some
> foreground, even one that contains no real signal. It cannot represent "this
> image is empty."
>
> *Absolute* applies a **fixed** level (on the image scaled to `0.0–1.0` by its
> bit depth), the same for every image. An image with no signal above that level
> correctly comes out empty.
>
> Use **Absolute** when:
> *   staining intensity is consistent across your replicates (so one fixed level
>     is valid for all of them), and/or
> *   a condition in the project may **lack this channel entirely**, and you need
>     to detect its presence or absence reliably — a percentile threshold would
>     manufacture foreground in a genuinely empty image, so absolute is required
>     here.
>
> Use **Percentile** when staining brightness varies between images and every
> image is expected to contain the channel, so adapting the threshold per image
> is helpful rather than harmful.
>
> The shipped 3D default is Percentile; the shipped 2D default is Absolute.
> Whichever you choose, tune within that one table.

---

## 🔧 Tuning this step

### Minimal baseline (start here on a new image type)

The scale table is **additive**: each row is thresholded independently and the
results are combined, so every row you add can only *contribute* more foreground.
A **Scale = `0.0`** row is the plain-intensity (non-tubular) pass — it captures
blobby, thick signal such as cell bodies. Tubular rows (Scale > 0) add thin,
tube-like signal such as processes, at the radius each row is tuned to. You build
up the mask by **layering** these together, not by choosing between them.

The fastest way to get oriented is to start with the one foundational row,
confirm the basics, then add tubular scales on top one at a time.

1.  **Start with a Scale = `0.0` row.** This captures cell bodies and other
    compact signal by intensity alone. Keep it in the table — it is the base
    layer that later rows build on, not a temporary stand-in.
2.  **Set the detection threshold.** Adjust that row's **low** value and
    re-process until the foreground on screen roughly matches the compact signal
    — cell bodies present, background dark. Lower `low` captures more; higher
    captures less. (Thin processes will still be missing; the tubular rows below
    add those.)
3.  **Set Minimum Size as low as it will go — for now.** Smoothing, gap-closing,
    and the seed gate are still off, so real signal is fragmented into small
    pieces; a high size filter at this stage deletes those pieces before you can
    reconnect them. Keep it at its minimum while tuning the threshold and scales,
    and raise it later once the mask is whole. **Watch the unit:** 3D counts
    **voxels** (default `1000`), 2D counts **pixels** (default `200`) — they
    differ by orders of magnitude.

At this point you should have the compact signal captured. Now add the tubular
layers:

### Layer 1 — add tubular scales for processes

**Add** rows near your actual process radii on top of the Scale = `0.0` row — a
typical set is `1.0`, `1.5`, `3.0` µm — each with its own **low** value. Because
the rows combine additively, these bring in the thin structures the intensity
pass missed without removing anything already captured.

*   **Fine tips still missing?** Add a smaller scale (e.g. `0.5`).
*   **Grainy / noisy foreground?** Raise the **low** value on the smallest scale,
    or remove that row — a very small scale responds to pixel-level noise as well
    as to fine structures.
*   **Thick processes missing?** Add a larger scale (e.g. `3.0` or more) sized to
    their radius.

Re-process and inspect the **Raw Intermediate Segmentation** layer after each
change.

### Layer 2 — clean up per scale

Now the per-row finishing controls:

*   **Smooth σ** — raise a row's smoothing if that scale is picking up noise. Keep
    it well below the row's Scale, or you will blur away the very structures the
    scale is meant to find.
*   **Max Gap (µm)** — if a scale's branches look like dashed lines, raise its gap
    to bridge them. If distinct cells are fusing, lower it.

### Layer 3 — the seed gate (optional, precision)

The **high** column adds a stricter secondary threshold. When `high` is a
stricter percentile than `low` **and below 100**, only mask components that
contain at least one very-bright ("seed") voxel are kept — a connectivity-based
way to drop dim false-positive blobs while still growing real objects down to the
looser `low` level.

*   Leave `high` at `100` (percentile) / `1.0` (absolute) to **disable** it —
    detection is then a single threshold on `low`.
*   Enable it (lower `high` below 100) only if faint background blobs survive your
    `low` threshold but real cells always contain a bright core. If no seed voxels
    are found at all, the gate is skipped for that run rather than wiping the
    image (you will see a warning in the log).

> **Under the hood.** This is hysteresis thresholding: `low` defines candidate
> foreground, `high` defines confident seeds, and only candidate components
> connected to a seed survive.

### Layer 4 — trace-linking (optional, for broken arbors)

**Trace Gap (µm)** reconnects a structure broken by a dim stretch by walking the
local intensity ridge outward from each fragment tip and bridging to whatever it
reaches within the given distance. It is orientation-following and works
branch-by-branch, so heavily arborised cells are handled. `0` = off. Turn it on
only if real processes are fragmented *after* you have done what you can with
thresholds and gap-closing.

> **Under the hood.** Trace-linking runs *before* the global size filter, so
> fragments it rejoins are size-tested as one object. In 3D it is also
> soma-aware: objects detected by the Scale-0 pass are flagged so the linker can
> treat cell bodies differently from processes.

---

## Parameter reference

| Parameter | Type | 3D default | 2D default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Threshold Mode** (`use_absolute_thresholds`) | toggle | Percentile | Absolute | Chooses which scale table is live. |
| **Scale** (per row) | float µm | `1.0, 1.5, 3.0` | `1.0, 1.5, 3.0` | Tubular radius. `0.0` = plain-intensity pass (cell bodies). Rows combine additively. |
| **low** (per row) | float | percentiles | absolute `0.0–1.0` | Detection threshold. |
| **high** (per row) | float | seed gate | seed gate | `100`/`1.0` disables. |
| **Smooth σ** (per row) | float µm | `0.2` | `0.1` | Pre-blur. `0` = off. |
| **Max Gap** (per row) | float µm | `1.0` | `0.0` | Gap-closing. `0` = off. |
| **Minimum Size** (`min_size`) | int | `1000` **voxels** | `200` **pixels** | Global; applied after merge. |
| **Trace Gap** (`trace_max_gap`) | float µm | `0.0` | `0.0` | Orientation trace-linking. `0` = off. |

**Advanced / not in the default config** (present in code, run on built-in
defaults unless a config exposes them): `subtract_background_radius` (white
top-hat background removal, `0` = off), and the Frangi shape constants
`frangi_alpha` / `frangi_beta` / `frangi_gamma`.

---

## What you see in the viewer

*   **Raw Intermediate Segmentation** — the binary mask this step produces. This
    is what you inspect while tuning.

## Outputs on disk

Written to `<image>_processed_<mode>/`:

*   `raw_segmentation_<mode>.dat` — the mask (memory-mapped label file).

The computed detection threshold is recorded in the run's config under
`saved_state`, so Step 2 and a later resume can reuse it.

---

## Where to go next

*   Mask looks right → continue to [Step 2: Edge Trimming](remove_artifacts.md).
*   Not sure Step 1 is even your problem → see the
    [Tuning Workflow](tuning_workflow.md) for how the steps fit together and which
    one to fix for a given symptom.
