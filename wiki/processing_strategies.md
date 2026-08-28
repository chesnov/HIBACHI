# Strategy Controller (`_3D_strategy.py` & `_2D_strategy.py`)

**Locations:**
*   `utils/module_3d/_3D_strategy.py`
*   `utils/module_2d/_2D_strategy.py`
*   **Base class:** `utils/high_level_gui/processing_strategies.py`

## Overview

The **Strategy Controller** orchestrates the segmentation pipeline. It implements
the Strategy pattern, decoupling the *definition* of the workflow from whatever
*drives* it — the GUI, the batch processor, or the parameter optimizer.

Whether running in 2D or 3D, the application relies on a concrete subclass of
`ProcessingStrategy` to:

1.  Define the sequence of steps.
2.  Manage file paths and checkpoints.
3.  Pass data (images, thresholds) between steps.
4.  Handle visualization in napari.
5.  Record what was analysed, and with which version of the code.

## The workflow definition

Both strategies define a **5-step** pipeline, and both implement **the same
method names** — the class differs, not the interface, which is what lets the GUI
and batch processor stay dimension-agnostic. The GUI uses
`get_step_definitions()` to generate its step list.

| Step | Method (both 2D and 3D) | Artifact key | File written |
| :--- | :--- | :--- | :--- |
| **1** | `execute_raw_segmentation` | `raw_segmentation` | `raw_segmentation_<mode>.dat` |
| **2** | `execute_trim_edges` | `trimmed_segmentation` | `trimmed_segmentation_<mode>.dat` |
| **3** | `execute_soma_extraction` | `cell_bodies` | `cell_bodies.dat` (3D) / `cell_bodies_<mode>.dat` (2D) |
| **4** | `execute_cell_separation` | `final_segmentation` | `final_segmentation_<mode>.dat` |
| **5** | `execute_calculate_features` | `metrics_df` | `metrics_df_<mode>.csv` |

`<mode>` is `fluorescence` for 3D and `fluorescence_2d` for 2D, from
`_get_mode_name()`. Note the one asymmetry: **3D writes `cell_bodies.dat` without
the mode suffix** while 2D includes it.

> **Cross-channel analysis is not a pipeline step.** It is a project-level tool,
> the [Cross-Channel Analyzer](cross_channel_analysis.md), which operates on the
> finished segmentations of several channels at once. Nothing in a strategy
> performs it.

### Where the mode suffix appears

*   **Method names have no suffix** — `execute_raw_segmentation` in both classes.
*   **Config keys do** — `get_config_key('execute_raw_segmentation')` yields
    `execute_raw_segmentation_fluorescence` or
    `..._fluorescence_2d`. This is how one YAML schema serves both modes.
*   **Most output filenames do** (see the table above).

## Core architectures

### 1. Memory mapping (the `.dat` files)

To handle datasets far larger than RAM, the strategies never hold the full
segmentation history in memory.

*   **Input:** the raw image is passed by reference.
*   **Intermediate:** every step writes its result to a memory-mapped file
    (`.dat`, at the full image shape). Label images are `int32`; the edge mask is
    `bool`.
*   **Hand-off:** step *N* closes its write handle; step *N+1* opens a read
    handle on that file.
*   **Cleanup:** `_close_memmap()` releases each handle as soon as a step is done
    with the file, so buffers are flushed and the file can be deleted. Windows
    refuses to delete a file that still has an open handle, raising
    `PermissionError`; Linux and macOS allow the delete and free the data when
    the last handle closes. A leaked handle therefore breaks cleanup only on
    Windows, which is why the closes are explicit and sit in `finally` blocks.

### 2. Intermediate state passing

Images live on disk, but small runtime values must pass between steps in memory.

*   **`self.intermediate_state`** — a dict shared across steps.
*   *Example:* Step 1 computes an automatic `segmentation_threshold`; Step 2 reads
    it back to decide which edge voxels are bright enough to keep.
*   `original_volume_ref` also lives here, holding the intensity image that Steps
    3–5 measure against.

### 3. Checkpointing & resume

*   **`get_checkpoint_files()`** maps logical keys to absolute paths. Subclasses
    extend the base dict, so `config` and `metrics_df` are defined once centrally.
*   **`get_last_completed_step()`** scans disk and returns the 1-based index of
    the last finished step. It walks the steps in order and **stops at the first
    missing artifact** — so completion is a prefix, never a set of holes.
*   A `StepDefinition` whose `artifact` is `None` deliberately **breaks the resume
    chain**: the step cannot be verified from disk, so nothing past it is assumed
    done. No current step uses this, but the mechanism is why an unverifiable step
    can never be silently skipped.
*   **Batch usage:** the batch processor asks the strategy where a folder left
    off and skips straight to the first unfinished step.

### 4. Provenance and analysed extent

Two records the base class produces:

*   **`_hibachi_version_stamp()`** writes a `hibachi_version` block into the saved
    config: commit, short hash, tag, date, branch, whether the working tree was
    dirty, and a UTC `processed_at`. It reuses the launcher's stdlib-only
    `updater` module, and is best-effort — a failure yields a minimal stamp rather
    than breaking the save. A result can therefore be traced to the code that
    produced it.
*   **`analyzed_extent()` / `stamp_analyzed_extent()` / `write_analysis_summary()`**
    record the region a run measured, as two extents: the whole image (or an
    ROI's polygon), and the tissue hull step 2 kept inside it. The hull is
    recovered from step 2's saved edge mask, one slice at a time.
    `extent_basis` records which produced the tissue figure. See
    [Step 5](calculate_features.md#analysed-region) for the columns and files.

### 5. Visualization abstraction

The GUI does not know how to display the data. It calls
`load_checkpoint_data(viewer, step)` and the strategy builds the layers.

*   **3D:** sets layer `scale` from Z-anisotropy so the volume is not squashed.
*   **2D:** sets `scale` from YX spacing.
*   **`_add_layer_safely()`** replaces a layer of the same name rather than
    stacking duplicates, which is what keeps repeated Process clicks from filling
    the layer list.
*   **`_build_label_pyramid()`** builds a max-pooling multiscale pyramid when an
    array exceeds the GPU texture limit (16384 px on a side), which is what lets
    a whole-slide image be displayed. Max-pooling rather than subsampling keeps a
    thin process visible at low zoom.

Layer names the segmentation steps produce: **Raw Intermediate Segmentation**,
**Trimmed Intermediate Segmentation**, **Edge Mask**, **Cell bodies**,
**Final segmentation**, **Skeletons**, plus the nearest-neighbour lines
from Step 5.

## Key methods

### `execute_step(step_index, viewer, image_stack, params)`

The dispatcher called by the GUI's worker thread or by the batch processor.

1.  Validates the step index against `num_steps`.
2.  Looks up the method name from the step definition.
3.  Calls it, returning its boolean success.

Note that state updates and persistence happen **inside each step method**, not
here — `execute_step` itself is a thin, uniform entry point, which is what lets
the batch processor drive the pipeline without knowing any step's internals.

### `save_config(current_config)`

Writes `processing_config_<mode>.yaml` into the processed directory: the
parameters used, the `saved_state` block (e.g. the auto-detected
`segmentation_threshold`), and the version stamp. This file is the record of how
a result was produced, and is what the reconcile machinery compares against.

### `cleanup_step_artifacts(viewer, step_number)`

Deletes the files and viewer layers belonging to **one** step, named by
`step_number` (1-based). Clearing a step and everything after it is the caller's
job: the GUI loops over the range it wants gone. Passing `viewer=None` deletes
the files and leaves the layers alone, which is how the batch processor calls it
and how the GUI re-runs a step without destroying its layer.

Called when a step is re-run, not when the user navigates backwards — navigation
is non-destructive (see [GUI Manager](gui_manager.md)).

### `get_config_key(step_name)`

Appends the mode suffix, mapping a method name to its config section.
