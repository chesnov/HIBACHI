# Step 2: Edge Trimming

**Corresponding modules:**
*   **3D:** `utils/module_3d/remove_artifacts.py`
*   **2D:** `utils/module_2d/remove_artifacts_2d.py`

## What this step decides

Step 2 cleans up the raw mask from Step 1 before cells are seeded. It does three
things, each of which can be turned off independently:

1.  **Z-anisotropy correction** (3D only) — thins objects that are smeared along
    the Z axis by anisotropic resolution.
2.  **Edge trimming** — removes segmentation that sits outside the real tissue, or
    up against missing/black regions of the image. This is the main job.
3.  **A post-trim minimum-size filter** — clears fragments left behind after the
    two operations above.

Nothing here creates or separates cells; it only removes or thins mask that
Step 1 produced. On many images Step 2 can be left almost entirely off — its
value depends on whether your image actually has a tissue boundary or edge
artefacts to remove.

> **Under the hood.** Edge trimming is built around a **tissue hull**: a solid,
> per-slice outline of where the real tissue is. The step generates the hull,
> then deletes any labelled object that lies more than a set distance outside it
> (with a brightness-based protection for genuine cells near the boundary). A
> separate pass deletes objects that sit against **true-zero** regions of the
> image — missing tiles or black borders. The size filter runs **last**, after
> trimming and Z-erosion have modified the masks.

---

## Order of operations

The step runs its stages in a fixed order. Knowing the order matters because each
stage feeds the next:

1.  **Z-erosion** (3D only), if `Z-Anisotropy Correction > 0`.
2.  **Edge trimming**, only if `Edge Trim Distance > 0`. This itself is two passes:
    first a hard delete of objects near true-zero regions, then hull generation
    followed by hull-based trimming with core protection.
3.  **Minimum-size filter**, if the post-trim minimum size `> 0`.

If `Edge Trim Distance = 0`, the entire edge-trimming block is skipped — no hull
is built, nothing is trimmed — and only Z-erosion (3D) and the size filter run.
The **2D pipeline has no Z-erosion stage** (there is no Z axis); otherwise the two
tracks are identical.

---

## 🔧 Tuning this step

### Minimal baseline (start here)

Both corrections ship **on** in the 3D default config and the edge trim ships
**off** in the 2D default, so do not assume defaults are neutral — set them
explicitly:

1.  **Turn edge trimming off:** `Edge Trim Distance = 0`.
2.  **Turn Z-correction off (3D):** `Z-Anisotropy Correction = 0`. (The 2D
    pipeline has no such control.)
3.  Leave the post-trim minimum-size filter at Step 1's Minimum Size or below (see
    the note under its parameter below).

With those off, Step 2 passes Step 1's mask through essentially unchanged, so you
can confirm the mask reaching Step 3 is the one you tuned in Step 1. Then add back
only what the image needs.

### Layer 1 — shape the tissue hull (only if you will trim)

If your image has a tissue boundary or edge artefacts to remove, edge trimming
needs a correct **hull** first — the trim distance is measured *from the hull
boundary*, so a wrong hull makes the trim meaningless. Turn on edge trimming
(`Edge Trim Distance > 0`) and watch the **Edge Mask** layer, which draws the hull
outline. Two parameters shape it:

*   **Otsu Scale Factor** (default `0.8`) sets what unsegmented intensity counts
    as tissue. The step computes an automatic tissue threshold (log-space Otsu)
    and multiplies it by this factor; the tissue region is everything above that
    threshold **plus all already-segmented cells** (OR'd together). So the factor
    only governs how much dim, unsegmented material — halo, faint tissue — is
    pulled into the hull; segmented cells are always inside it.
    *   *Too low* → dim background is counted as tissue, the hull inflates outward
        and can snap to the image borders.
    *   *Too high* → only bright material is added, the hull tightens toward the
        segmented cells and leaves dim real tissue exposed to trimming.
    *   Move in small steps (±0.1) and re-check the outline.
*   **Hull Closing Radius** sets how far across open gaps the hull reaches to stay
    one continuous shape along its border. The tissue mask is downsampled (4×),
    dilated slightly, closed with a disk of this radius, then interior holes are
    filled. Because holes are filled afterwards, this radius controls the
    **concavities open to the boundary** (bays and notches on the outer edge), not
    gaps inside the tissue. Because it acts on the 4×-downsampled mask, each unit
    is roughly four original pixels.
    *   *Too small* → the outline traces every notch and can break into islands,
        leaving parts of the tissue outside.
    *   *Too large* → the outline bridges wide gaps and pushes outward past the
        true edge, wrapping background as tissue.
    *   The default differs by pipeline (**3D `1`, 2D `10`**) because the 2D hull
        is built from a single plane, where sparser coverage needs more bridging;
        treat those as starting points.

The goal is an **Edge Mask** outline that traces the real tissue edge — snug,
continuous, and neither eating into the tissue nor snapping to the image borders.

### Layer 2 — trim, and protect real cells at the edge

With the hull correct, **Edge Trim Distance** sets how far outside the hull an
object may sit before it is deleted (in microns). Larger removes more; smaller is
more conservative. This same distance also governs the separate pass that deletes
objects against true-zero regions (missing tiles / black borders).

If real cells near the boundary are being deleted, **lower the Brightness Cutoff
Factor**. This parameter sets a brightness bar a voxel must exceed to be treated
as a protected "core" — cores, plus a small margin around them, are spared from
trimming. The bar is the reference tissue threshold multiplied by this factor, so:

*   A **lower** factor lets more real signal clear the bar, protecting more
    bright cells near the edge.
*   A **higher** factor protects less and trims more.

The shipped defaults differ sharply: **2D uses `1.5`** (protection active), while
**3D uses a very large value** that puts the bar out of reach, effectively
disabling protection so 3D trimming is purely distance-based. If you need edge
cells protected in 3D, bring this down into a sensible range (a small multiple of
the tissue threshold).

### Layer 3 — Z-anisotropy correction (3D only)

If cells look stretched into pillars along the Z axis, raise **Z-Anisotropy
Correction** to erode that stretch back. The erosion acts only along Z, and it is
"clamped": if a column would be erased entirely it restores a central voxel, so
objects are thinned rather than deleted. Leave it at `0` if cells are not
stretched. The **2D pipeline has no Z axis and no such control.**

### The post-trim size filter

Step 2 has its **own** minimum-size filter, separate from Step 1's, applied
**after** trimming and Z-erosion. It matters because trimming removes voxels — and
because it can *sever* one object into several (for example, when a bright bridge
between cells is trimmed away), the relabelled fragments can be small. Set it high
enough to drop real debris but not so high that legitimately small trimmed cells
disappear. Setting it below Step 1's value can still remove masks if any trimming
occurred; keeping it at Step 1's level or below is safe when no trimming is
enabled.

---

## Parameter reference

| Parameter | Type | 3D default | 2D default | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Edge Trim Distance** (`edge_trim_distance_threshold`) | float µm | `4.0` | `0.0` | Master switch for edge trimming. `0` = off. Distance outside the hull (and from true-zero regions) beyond which objects are deleted. |
| **Otsu Scale Factor** (`otsu_scale_factor`) | float | `0.8` | `0.8` | Multiplies the automatic tissue threshold that defines the hull. |
| **Hull Closing Radius** (`hull_closing_radius`) | int | `1` | `10` | Disk radius (on the 4×-downsampled mask) for closing the hull. Labelled "Hull closing radius" (3D) / "Hull Smoothing Radius" (2D). |
| **Brightness Cutoff Factor** (`brightness_cutoff_factor`) | float | `1000000.0` | `1.5` | Brightness bar (× tissue threshold) a voxel must exceed to be a protected core. Lower = protect more. The 3D default disables protection. |
| **Z-Anisotropy Correction** (`z_erosion_iterations`) | int | `2` | — | Z-only clamped erosion iterations. **3D only.** `0` = off. |
| **Min Size, Post-Trim** (`min_size_voxels` / `min_size_pixels`) | int | `2000` **voxels** | `120` **pixels** | Global connected-component size filter applied after trimming. |

---

## What you see in the viewer

*   **Trimmed Intermediate Segmentation** — the mask after this step. This is what
    you inspect to see what was removed.
*   **Edge Mask** — the hull boundary outline (a one-voxel shell). Use it to judge
    whether the hull correctly wraps the tissue. It is empty when edge trimming is
    off.

## Outputs on disk

Written to `<image>_processed_<mode>/`:

*   `trimmed_segmentation_<mode>.dat` — the trimmed mask.
*   `<mode>_edge_mask.dat` — the hull boundary.

---

## Where to go next

*   Mask looks clean → continue to [Step 3: Soma Extraction](soma_extraction.md).
*   Not sure Step 2 is your problem → see the
    [Tuning Workflow](tuning_workflow.md) for how the steps fit together.
