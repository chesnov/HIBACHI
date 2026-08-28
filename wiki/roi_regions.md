# Sub-Regions (ROIs)

**Corresponding modules:**
*   `utils/high_level_gui/roi_sharing.py` — polygons, crops, per-region configs,
    propagation (Qt-free)
*   `utils/high_level_gui/gui_manager.py` — the drawing and session controls
*   `utils/high_level_gui/roi_overlay_panel.py` — the same controls in an overlay
    viewer

## What a region is

A region is a polygon you draw on an image, which then behaves as an image in its
own right: it has its own config, its own parameters, its own five-step run and
its own results. The full image keeps its own separate results, and a channel can
hold several named regions at once.

Two things this is for:

*   **Tuning quickly.** Parameters that take minutes on a whole stack take
    seconds on a crop, so you can iterate, then apply the values you settled on to
    the full image.
*   **Analysing part of a sample.** A region can be processed, batch-processed and
    used in cross-channel analysis exactly like a whole image, with densities
    reported against the region rather than the frame.

---

## Drawing one

The controls sit under **Sub-region (ROI)** in the viewer's left-hand panel:

| Button | Does |
| :--- | :--- |
| **✏ Draw** | Start a polygon on a shapes layer. Click to place vertices. |
| **✓ Apply** | Turn the polygon into a region and switch the pipeline into it. |
| **✗ Clear** | Discard a polygon that has not been applied. |
| **⤷ Open region** | Reopen a saved region. |
| **🗑 Delete region** | Remove a saved region and its results. |

After **Apply**, the viewer shows the crop, the step controls act on it, and
results are written to the region's own directory.

Opening a channel on its full image outlines every saved region as a polygon
layer, each in a different edge colour, so you can see what already exists before
drawing another.

You can draw a polygon per Z level in a 3D stack. Slices between two drawn levels
use the nearer polygon; slices outside the drawn range take the first or last one
rather than tapering to nothing.

A polygon smaller than `MIN_CROP_PX` (10 px on a side) is rejected.

---

## What gets created

A region lives beside the full image's results:

```
<sample>/
├── <sample>.tif
├── <config>.yaml
├── <sample>_processed_<mode>/            full-image results
└── <sample>_processed_<mode>_<ROI_1>/    one region
    ├── roi_polygon.json                  the polygon, in full-image coordinates
    ├── roi_image_crop.dat                the cropped, masked image
    ├── processing_config_<mode>.yaml     this region's own config
    └── (that region's step artifacts and metrics)
```

The first, unnamed region uses the plain `..._roi` suffix and displays as
"ROI 1"; it is read under that name rather than renamed on disk.

### The crop is rectangular, the region is not

`roi_image_crop.dat` is the polygon's **bounding box**, with everything outside
the polygon set to zero. Measurements are unaffected — there is no signal outside
the polygon to find — but it means the array is larger than the region, which is
why the analysed extent is measured from the polygon rather than from the array.
See [Step 5](calculate_features.md#analysed-region).

### The config is seeded, then owned

A new region's config is a copy of the channel's, with the physical dimensions
scaled to the crop:

```
new_x_um = original_x_um * (crop_width / full_width)
```

Because the configs store **total** extent rather than per-pixel size, this
leaves the per-pixel spacing identical to the full image's, so a parameter in
microns means the same thing in both.

That derivation happens **once**, when the region is created. After that the
region owns its config: re-deriving it on every open would discard whatever you
tuned for that region.

---

## Sharing one region across channels

A sample's channels are extracted from the same acquisition, so they share pixel
dimensions. A polygon stored in full-image coordinates is therefore valid
verbatim in every channel, and propagating a region means writing one small JSON
per channel. Each channel builds its own crop and its own config the first time
it opens the region — the polygon is shared, the pixel data is not.

Propagation is a two-stage operation, because it can destroy results:

**`plan_roi_propagation()`** inspects and reports, writing nothing. Per channel
it returns one of:

| Status | Meaning |
| :--- | :--- |
| `new` | No region here yet; nothing will be lost. |
| `replace` | A region exists; its crop, config and **results** will be cleared. |
| `shape_mismatch` | This channel's image does not match the drawing; skipped. |
| `unusable` | Not a valid sample folder. |

**`apply_roi_propagation()`** then performs it. The split exists so the plan can
be shown for confirmation first.

The clearing is not optional. The loader prefers an existing crop, so a new
polygon dropped next to an old crop would load the old sub-region while the JSON
described the new one.

**Clearing** works the same way: `plan_roi_clear()` classifies each channel as
`has_roi`, `orphan` (a region directory with no polygon — leftover files) or
`no_roi`, and `apply_roi_clear()` acts on it.

---

## Using regions elsewhere

*   **Project window.** Regions appear as a third level in the contents tree,
    under their channel, with their own status and last-edited time. Tick one and
    **Process Selected** runs it like any other target — see
    [Batch Processing](batch_processor.md). **Delete Regions** removes checked
    regions and their results; checked channel rows are ignored rather than
    treated as delete targets.
*   **Set New Channel Config.** Applying a config to a channel can extend to its
    regions, with each region's dimensions rescaled to its own crop.
*   **Cross-channel analysis.** The region selector applies a whole recipe to one
    named region across every sample that has it. See
    [Cross-Channel Analyzer](cross_channel_analysis.md).
*   **Overlay viewers.** The same draw/confirm/open controls are docked into the
    cross-channel overlay viewer, so a region can be drawn while looking at
    several channels at once.

---

## Where to go next

*   [Tuning workflow](tuning_workflow.md) — using a region to iterate before
    committing to the full image.
*   [Step 5](calculate_features.md#analysed-region) — how a region's extent
    becomes the denominator for a density.
