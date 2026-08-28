# Batch Processor (`batch_processor.py`)

**Location:** `utils/high_level_gui/batch_processor.py`
**Related:** `batch_runner.py` (child-process entry point),
`batch_progress_dialog.py` (the progress window)

## Overview

The `BatchProcessor` is the engine behind the **Process Selected** button in the
project window. You tick images, channels or saved regions in the contents tree
and it works through them unattended.

It acts as a **headless controller**: it instantiates the appropriate strategy
(2D or 3D) per target but passes `None` instead of a napari viewer, so the
pipeline runs purely on disk/CPU/RAM with no rendering overhead.

## Key responsibilities

### 1. Working through the selection

The processor works through `project_manager.image_folders`. For a batch, the
project window points that at exactly the checked set, then restores it
afterwards — so the processor never decides for itself what to process.

Each entry is a **leaf key**: either a folder path, or `<folder>::<region>` for a
saved sub-region. A region is therefore a batch target in its own right,
alongside a whole image.

Three helpers keep that uniform:

*   **`_leaf_mode()`** — a region has no directory of its own, so the key is split
    and the mode of its **parent channel** is used. Without this every region
    would be rejected as an invalid folder.
*   **`_leaf_label()`** — the readable name for logs and dialogs, e.g.
    `sample1 [ROI 2]`.
*   **`_resolve_target()`** — the single resolver for everything a strategy needs.
    A region differs from a whole image in all three respects: its image is the
    cropped memmap, its config is its own (with dimensions rescaled to the crop),
    and its results live in the region's session directory. One resolver means
    those three derivations cannot drift apart between entry points.

Targets whose mode is unsupported are skipped rather than attempted.

### 2. Spacing normalization — and a hard refusal

Physical dimensions come from the config, and are converted to per-pixel spacing:

*   **3D:** `(Z, Y, X)` spacing plus the Z-anisotropy scale factor.
*   **2D:** `(1.0, Y, X)`. Z is forced to `1.0` so functions written for 3D (such
    as distance transforms) operate correctly on a single plane.

**There is no fallback.** If an image has no physical dimensions,
`require_dimensions()` raises `MissingDimensionsError` and the target is recorded
with status `no_dimensions` instead of being processed with assumed values.
Assumed dimensions cannot be detected downstream: they are positive, finite
numbers like any others, so a run would complete and report sizes in microns
that are scaled by whatever the assumption happened to be.

Supply the dimensions (a CSV at setup, or re-running setup) and run the batch
again.

### 3. Memory safety

Nothing is left to garbage-collection timing:

*   The heavy image array and the strategy instance are deleted after every
    target.
*   `gc.collect()` is called between steps and between targets.

### 4. Smart resume

Resume comes from the strategy's `get_last_completed_step()`, which checks which
artifacts exist on disk. A target that stopped after Step 2 resumes at Step 3
instead of repeating work. Because completion is judged as a prefix, there is no
way to resume into a state with holes in it.

---

## How a run is driven

The GUI path is a three-stage split, which exists so that prompts happen on the
GUI thread while the work happens elsewhere:

### `prescan_folders()`

Categorises every target *before* anything is processed, returning
`(complete, partial, scan_results)`. Cheap, main-thread friendly, and shows no
dialogs.

### The reprocess prompt

If anything already has output, `_prompt_reprocess_choice()` shows what was found
and offers three outcomes, with button labels that adapt to whether the existing
output is complete, partial, or both:

*   **`restart_all`** — reprocess complete *and* partial targets from Step 1.
*   **`resume`** — resume partial targets from their last step; skip complete ones.
*   **`cancel`** — abort the run.

There is a console fallback when Qt is unavailable, so the batch remains usable
headless.

### `run_folders(force_map, progress_callback)`

Does the work from a **pre-resolved plan**: `force_map` maps each target to
whether it restarts from Step 1. This function scans nothing and prompts for
nothing, so it can run in a worker with no GUI available.
`progress_callback` receives `folder` and `step` events. Returns
`(success, failed, skipped)`.

> **Under the hood — why a child process, not a thread.** `batch_runner.py` runs
> the batch in a **separate OS process**, communicating over a tagged queue
> (`log` / `progress` / `done` / `error`).
>
> A long native call inside NumPy or SimpleITK does not return to the Python
> interpreter, so a thread running one cannot check a cancel flag until it
> finishes. A process can be killed outright, so **Cancel** takes effect
> immediately. A native crash also takes down only the child, leaving the GUI
> running to report it. On POSIX the child gets its own session (`setsid`), so
> signalling it reaches the worker pools the pipeline itself spawns.
>
> `batch_progress_dialog.py` renders the result: a spinner, an outer bar over
> targets, an inner bar over steps, and a live console pane.

### `process_all_folders(force_restart_all=False)`

A single-call entry point: scan, prompt and run in one. Useful for scripted or
headless runs. **Not** the path the GUI takes.

### `process_single_folder(...)`

One target, end to end: validate, load the image, build the strategy, run the
step loop, then free the image immediately.

---

## Where to go next

*   Regions come from the ROI workflow — see [Sub-Regions](roi_regions.md).
*   To apply one config across many images before batching, see the
    [Config Library](config_library.md) and **Set New Channel Config**.
*   Cross-channel work is not part of this batch. It has its own
    run-on-all-samples path in the
    [Cross-Channel Analyzer](cross_channel_analysis.md).
