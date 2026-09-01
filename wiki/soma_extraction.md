# Step 3: Soma Extraction

**Corresponding modules:**
*   **3D:** `utils/module_3d/soma_extraction.py`
*   **2D:** `utils/module_2d/soma_extraction_2d.py`

## What this step decides

Step 3 finds the **cell bodies** (somata) inside the Step 2 mask and places one
**seed** per detected soma. Those seeds are what Step 4 uses to split and merge
the mask into individual cells.

This step effectively decides how many cells the pipeline will find. Two error
types are not equal, and this shapes how you tune:

*   A soma that gets an **extra seed or two** can be rescued later — Step 4 can
    re-merge an over-split cell.
*   A cell with **no seed cannot be recovered at all.** Step 4 only ever splits
    and merges around existing seeds; it never invents one. A cell with no soma is
    simply absent from the final result.

So aim for one seed per cell as closely as the image allows, but when forced to
choose, lean toward too many seeds rather than too few.

> **Under the hood.** For each labelled object the step tries a set of *strategies*
> to isolate a compact core, collects every candidate core, then places seeds
> greedily. A candidate must survive a chain of gates (size → thickness → aspect
> ratio → peak separation → cross-object overlap). The step prints a diagnostic
> breakdown of how many cores were rejected at each gate, which is how you tune it
> — by reading *why* a soma was rejected, not by guessing.

---

## Two seeding modes: ratios or percentiles

Step 3 has two independent ways to isolate a core, exposed as two lists. In
practice you use one or the other, not both.

*   **Ratios to Process** — distance-transform peeling. For each object it keeps
    the voxels whose distance from the surface is at least `ratio × (thickest
    point)`, so it thresholds cores by **thickness**. A *lower* ratio keeps a
    larger core reaching further from the centre; a *higher* ratio keeps only the
    thick centre. It excels when cells are joined by **thin connections**, because
    the distance transform pinches at a thin neck.
*   **Intensity Percentiles to Process** — keeps the pixels above a given
    intensity percentile, so it thresholds cores by **brightness**. A *lower*
    percentile includes dimmer pixels and makes **larger** cores; a *higher*
    percentile keeps only the brightest pixels and makes **much smaller** cores.
    It is harder to tune but works far better for **tightly packed cells with
    thick interfaces**, where there is no thin neck to catch but the centres are
    still brighter than the shared border.

**They are not additive.** Each candidate carries a priority score: intensity
strategies score above `2.0`, distance-transform strategies score below about
`1.1`. Candidates are placed greedily from highest score down, and a lower-scored
candidate overlapping an already-placed seed is dropped. So **if you specify any
percentiles, they override the ratios wherever the two compete** — the ratios only
place a seed in space no percentile claimed. Treat it as a mode switch: use ratios
(and leave percentiles empty) for thin-neck separation; use percentiles for
thick-interface separation. The shipped config populates both lists, which means
percentiles are doing the work by default.

---

## 🔧 Tuning this step

The approach is **find as many candidate somata as possible first, then narrow to
one per cell** — reading the diagnostic rejection counts to decide what to tighten,
rather than starting strict and guessing why a cell came out bare.

### Step A — pick a mode and get the threshold roughly right

Decide ratios vs percentiles from the paragraph above. Then set the threshold so
that essentially every real cell centre produces a core:

*   *Percentile mode:* start at a moderate percentile and, to separate two touching
    cells, go **higher** until each centre resolves into its own small bright core;
    to stop one cell fragmenting, go **lower** so its core stays whole. Listing
    several percentiles lets the step try each.
*   *Ratio mode:* a lower ratio grows larger cores, a higher ratio keeps tight
    central cores. Listing several ratios lets the step try each and keep the
    best-separated result.

### Step B — open the gates wide, then tighten one at a time

Every candidate core must pass a chain of gates. Start permissive so real somata
are not dropped, then tighten using the printed diagnostics. Each gate has a
matching rejection counter in the run output:

*   **Min Seed Size** (`min_fragment_size`) — smallest a core may be. Start low so
    small or dim somata survive; raise it only if noise specks seed. → *Rejected:
    Too Small.*
*   **Min / Max Absolute Thickness** — the accepted thickness window. Start wide
    (low min, high max). Raise the min to exclude thin processes; lower the max to
    exclude oversized blobs. → *Rejected: Thickness Bound.* (The max bound does not
    discard a fragment outright — it recovers the inner sub-core that falls within
    the window, and only rejects if even that is too thin.)
*   **Max Aspect Ratio** — how elongated a core may be before it is treated as a
    process rather than a soma. Start high (permissive); lower it to reject
    stretched, worm-like detections. → *Rejected: Aspect Ratio.* (Before rejecting,
    the step tries to shave the low-distance tails off an elongated core to isolate
    a spherical peak.)
*   **Min Peak Separation** — minimum physical distance between two seeds within one
    object. Raise it to merge duplicate seeds on one cell; lower it if two genuinely
    close cells collapse into one seed. → *Rejected: Spatial Overlap*, plus a
    `[TRAP]` line printed whenever one object receives multiple somas.

The same distance also applies **between seeds in different objects**. When a new
seed lands closer than this to an already-placed one, the step tries to shrink it
(toward its own brightest/thickest core, within the same seeding mode) until it
clears; if it cannot clear without falling apart, the seed is dropped.
→ *Pushed Apart -> Dropped.* A high count here means genuinely distinct cells are
sitting within **Min Peak Separation** of each other — lower that value if they
are being lost.

### Step C — leave erosion off

**Soma Erosion Iterations** should be **0** almost always. It erodes every core
before island detection — a blunt, whole-image operation that helps only in rare
cases and otherwise shrinks or destroys good seeds.

### Reading the result

Inspect the **Cell bodies** layer: one seed per cell, sitting in the cell body.
When a cell is missing its seed, read the run's diagnostic counts (*Too Small,
Thickness Bound, Aspect Ratio, Spatial Overlap, Pushed Apart -> Dropped*) to see why
it was removed, and loosen the matching control. When a cell has several seeds, raise **Min Peak Separation**
or tighten the mode threshold (higher percentile / higher ratio) so its centre
resolves as one core — or leave it, since Step 4 can re-merge a modest over-split.

> **Under the hood.** Thickness is the radius of the largest inscribed sphere (3D)
> or disc (2D) from the distance transform; aspect ratio is the elongation from a
> PCA of the core. A core containing several distance-transform peaks is
> watershed-split into separate candidates before the gates are applied. Candidates
> are then sorted by score (intensity above distance-transform) and volume, and
> placed greedily with the within-object separation gate and a cross-object pixel-
> overlap check — this is the mechanism behind percentiles overriding ratios. The
> logic is identical in 2D and 3D, one fewer dimension.

---

## Parameter reference

| Parameter | Type | 3D default | 2D default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Ratios to Process** (`ratios_to_process`) | list of float | `0.3, 0.4, 0.5, 0.6` | `0.3, 0.4, 0.5, 0.6` | DT-peeling thresholds. Lower = larger core. Leave empty to use percentiles only. |
| **Intensity Percentiles** (`intensity_percentiles_to_process`) | list of int | `99 … 1` (21 values) | `99 … 1` (21 values) | Brightness thresholds. Higher = smaller core. Override ratios where they compete. |
| **Min Seed Size** (`min_fragment_size`) | int | `2500` **voxels** | `500` **pixels** | Smallest core kept. Labelled "Min Seed Size" (3D) / "Min Fragment Size" (2D). |
| **Min Absolute Thickness** (`absolute_min_thickness_um`) | float µm | `1.5` | `1.5` | Lower bound of the thickness window (hard reject). |
| **Max Absolute Thickness** (`absolute_max_thickness_um`) | float µm | `10.0` | `10.0` | Upper bound; recovers the inner sub-core rather than discarding. |
| **Max Aspect Ratio** (`max_allowed_core_aspect_ratio`) | float | `10.0` | `5.0` | Elongation limit; tries tail-shaving recovery before rejecting. |
| **Min Peak Separation** (`min_physical_peak_separation`) | float µm | `7.0` | `20.0` | Minimum distance between two seeds. |

---

## What you see in the viewer

*   **Cell bodies** — the placed seeds, one label per detected soma. This is what
    you inspect while tuning.

The seed count and per-gate rejection counts are printed to the process log every
run; use them together with this layer.

## Outputs on disk

Written to `<image>_processed_<mode>/`:

*   **3D:** `cell_bodies.dat` (unsuffixed).
*   **2D:** `cell_bodies_<mode>.dat`.

---

## Where to go next

*   Seeds look right → continue to [Step 4: Cell Separation](cell_splitting.md).
*   For how seeding interacts with the rest of the pipeline → see the
    [Tuning Workflow](tuning_workflow.md).
