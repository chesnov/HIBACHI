# Project Setup

**Corresponding modules:**
*   `utils/high_level_gui/project_selection.py` — classifying a path, the welcome
    screen, the contents tree
*   `utils/high_level_gui/organize_wizard.py` — the setup wizard
*   `utils/high_level_gui/project_scaffolding.py` — building the project on disk
*   `utils/high_level_gui/dimension_entry.py` — physical dimensions and their
    provenance

## What a project is

A project is a folder of per-image subfolders, each holding **one image and one
config**. That pairing is what makes an image processable: the config carries the
parameters and the image's physical size.

Setup takes a folder of raw images and produces that structure.

---

## Opening something

The welcome screen takes a folder or a single image, by drag-and-drop, by
**Browse**, or from the recent-projects list. Whatever you give it,
`classify_path()` decides what it is:

| Kind | What it means | What happens |
| :--- | :--- | :--- |
| `project` | Subfolders each with one image and one config | Opens it |
| `multichannel_project` | Contains `Channel_*` folders that are themselves projects | Opens the sample→channel tree |
| `raw_images` | Loose images, not organized | Offers to set up a project |
| `parent_of_projects` | Contains folders that are projects | Asks which one |
| `empty` | Nothing recognised | Says so |
| `missing` | Path does not exist | Says so |

Drop a **file** and the folder containing it is used. If that file is a format
HIBACHI cannot read, you are told which format it is rather than being shown an
empty folder. If it is readable but not part of the project it sits in, you are
offered the chance to add it.

An organized project wins over loose images: a folder holding both opens as a
project rather than offering to rebuild one. A multi-channel project wins too,
because the multi-channel scaffolder leaves the raw sources in place next to the
channel folders, so such a folder is legitimately both.

---

## Readable formats

| Format | Notes |
| :--- | :--- |
| `.tif`, `.tiff` | Read directly |
| `.czi` | Zeiss, via `aicspylibczi` |
| `.lif` | Leica LAS X, via `readlif`. One file holds several acquisitions |
| `.vsi` | Olympus / EVIDENT slide scanner |
| `.svs`, `.ndpi`, `.scn`, `.afi`, `.qptiff`, `.zvi`, `.ome.tif`, `.dcm` | Whole-slide formats via `slideio` |

### One file can be several samples

A slide file or a `.lif` holds several images. Each becomes its own sample, keyed
`file.ext::name`, and lands in its own folder — so six scans in one `.vsi` give
six samples rather than five being discarded.

Acquisitions that cannot be imported are skipped with a reason. For `.lif` those
are mosaic/tile scans (the tiles are stored separately and not stitched, so
importing would give one tile), time series (there is no time axis in the
pipeline, so only the first frame would survive) and line scans.

A format HIBACHI cannot read — `.nd2`, `.oib`, `.lsm`, `.ims` and similar — is
named as such, with the suggestion to export as OME-TIFF or TIFF.

`.vsi` and `.afi` keep their pixel data in a companion folder beside the file.
Copying the file alone leaves a readable header with no image, which is detected
and reported.

---

## The setup wizard

Three modes:

*   **new** — build a project from the raw images in a folder.
*   **add** — extract another channel from raw sources still sitting in the
    folder.
*   **resetup** — delete the organized structure and processed outputs, keep the
    raw images, and start again.

The wizard detects how many channels the images have and asks whether to treat
the set as single- or multi-channel. You then choose a **preset per channel** —
the preset list is filtered by the detected dimensionality, so a plainly-3D
dataset is not offered 2D configs. You can also process a subset of the images
rather than all of them.

Setup runs in a worker thread with a progress dialog and a working Cancel, and
writes a diagnostics log of what it did.

### What it builds

**Single-channel** — each image moves into its own subfolder alongside a copy of
the chosen preset:

```
project/
├── sample_A/  sample_A.tif  +  <preset>.yaml
└── sample_B/  sample_B.tif  +  <preset>.yaml
```

**Multi-channel** — each channel is extracted to its own TIFF, under one folder
per channel. Raw sources stay where they are.

```
project/
├── raw_001.czi                     (left in place)
├── Channel_0_Microglia/
│   ├── metadata.csv
│   └── raw_001/  raw_001.tif  +  <preset>.yaml
└── Channel_1_Neurons/
    └── raw_001/  raw_001.tif  +  <preset>.yaml
```

Channel folders are named `Channel_<n>_<Name>`, where the name comes from the
first word of the chosen preset.

The per-image config keeps the **preset's own filename**. The name
`processing_config_<mode>.yaml` belongs to the *processed* config written inside
the results directory, which is a different file.

---

## Physical dimensions

Every measurement in microns comes from the physical dimensions recorded in each
image's config, so setup is where they are established.

### Where they come from

In precedence order:

1.  **A metadata CSV** placed directly in the raw-image folder, with columns
    `Filename`, `Width (um)`, `Height (um)`, `Depth (um)`. Values are **total**
    extent per axis, not per-pixel size. One row per source image covers every
    channel extracted from it. Blank or non-positive cells are ignored per axis,
    so a CSV can pin X and Y and leave Z to the file.

    Only files sitting directly in the folder are considered, so the
    `metadata.csv` that multi-channel setup writes into each channel folder is
    never mistaken for an input on a re-run.

2.  **The image file's own metadata**, where it carries a real scale.

A spacing of exactly 1.0 µm/pixel is treated as *no* calibration, because many
writers store `XResolution=(1,1)` with `ResolutionUnit=NONE` on an uncalibrated
image, and that is indistinguishable from a genuine unit scale.

### When neither supplies a value

Setup prompts. The **Enter image dimensions** dialog asks only about the axes
that neither route resolved, showing each axis's pixel count so you can see the
spacing your number implies, with an apply-to-all option for a dataset shot at
one magnification.

It also prompts when a recorded total is **identical to that axis's pixel
count** — meaning exactly 1 µm per pixel. That is occasionally correct and often
means the scale was never set, or that pixel counts were pasted into a microns
column, so the field is pre-filled and confirming it is one click.

Cancelling the prompt cancels setup, rather than proceeding with unknown scale.

### Provenance

Each config records where its dimensions came from, in `dimensions_source`:

| Value | Meaning |
| :--- | :--- |
| `metadata` | Read from the image file's header |
| `csv` | From a metadata CSV |
| `manual` | Entered or confirmed in the dialog |
| `mixed` | Different axes from different sources above |
| `pixels_assumed` | At least one axis had no scale, so its extent is that axis's pixel count |
| `unknown` | The config does not carry the field |

The field records provenance only. To check the dimensions themselves, compare
the config's `voxel_dimensions` / `pixel_dimensions` against your acquisition
settings.

---

## Extending a project later

*   **＋ Add images…** organizes raw images still loose in the project folder,
    reusing each channel's existing config so a late arrival stays comparable
    with its siblings. Files it cannot read are named rather than silently
    ignored.
*   **＋ Add channel…** (wizard `add` mode) extracts another channel from the
    raw sources.
*   **⚙ Set New Channel Config…** applies one config to the checked images, keeping
    each image's dimensions. It can extend to their regions, rescaling
    dimensions to each region's crop. Results invalidated by the change are
    named and cleared.
*   **Re-set up project…** (wizard `resetup` mode) discards the organized
    structure and processed outputs, keeping the raw images.

---

## Where to go next

*   [Tuning workflow](tuning_workflow.md) — choosing parameters once a project
    opens.
*   [Config Library](config_library.md) — reusing a tuned config across projects.
*   [Batch Processing](batch_processor.md) — processing the whole project.
*   [Sub-Regions](roi_regions.md) — working on part of an image.
