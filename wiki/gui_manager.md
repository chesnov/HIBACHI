# GUI Manager (`gui_manager.py`)

**Location:** `utils/high_level_gui/gui_manager.py`

## Overview

`DynamicGUIManager` is the bridge between the logic (the
[Strategy](processing_strategies.md)) and the interface (napari). It builds the
step-by-step controls, decides what the user is allowed to do next, runs steps off
the GUI thread, and manages sub-region sessions.

It knows *that* there is an ordered list of steps; it does not know *what* they
are. Everything step-specific comes from the strategy and the config.

## Responsibilities

### 1. Widget generation

*   Reads the parameters for the current step from the YAML config.
*   Builds a widget per parameter via `parameter_widgets.create_parameter_widget`.
    `float`, `int`, `bool` and `list` use `magicgui`; the three `scale_table`
    types get a custom PyQt `ScalesTableWidget`, since a table of per-scale rows
    is not a shape `magicgui` builds from a function signature.
*   Writes changes straight back into the in-memory config and emits
    `params_edited` so the navigation buttons re-evaluate.

The controls live in a **single merged left-hand panel**, assembled in
`app_launch.build_segmentation_control_panel` rather than here.

### 2. Navigation: the valid frontier

**`valid_frontier()`** returns the 0-based index of the first step that is *not*
processed-and-current: every step below it has its artifact on disk and unchanged
parameters. The three button states all derive from that one number:

*   **`can_go_back()`** — any step above the first.
*   **`can_go_forward()`** — only into already-valid positions, capped by the
    frontier. Once every step is processed, Forward can land on a terminal
    "complete" state with no parameter widgets, mirroring where processing the
    last step leaves you, so Back and Forward stay symmetric at the end.
*   **`can_process()`** — only on the frontier itself. A clean, already-processed
    step has nothing to compute, so Process is unavailable there.

**Editing a processed step collapses the frontier to it**, so the steps after it
stop counting as valid until it is re-processed. `is_current_step_dirty()` detects
the edit by comparing the live widgets against the last committed values.

> **Navigation is non-destructive.** Going back never deletes results: you can
> walk back through a finished pipeline to inspect earlier layers and return
> without recomputing anything. Only **Process** clears downstream results.

If you go back with unsaved edits, you must resolve them: **Discard** (revert to
the last committed values via `_revert_current_edits()`), **Process now** (compute
this step, which does clear later results), or **Cancel**.

### 3. Config canonicalisation before compute

`_ensure_config_canonical()` gates every Process call. It reconciles the config
against the built-in reference schema for the mode, reports any additions,
removals or clamps for confirmation, and clears results invalidated by the
change.

Staleness is enforced **only at compute time**. A config tuned on a different
pipeline version stays fully viewable and analysable — you are stopped from
computing *new* results with a schema that does not match it, not from looking at
results you already have. See the
[Config Library](config_library.md) for the mechanism.

### 4. Execution

*   `StepWorker` (a `QThread`) runs the step so the UI stays responsive.
*   `OutputStream` redirects `stdout` into a persistent **Process Log** dock
    (`_init_persistent_log`), which is where step progress and the `[PROFILE|…]`
    diagnostic lines appear. Errors land here rather than in a popup per error.
*   Signals `process_started` / `process_finished` let the rest of the UI disable
    and re-enable itself around a run.
*   Before each step starts, `_snapshot_child_pids()` records the child PIDs that
    already exist. Teardown (`_stop_worker_safely`, reached from
    `shutdown_and_cleanup()`) then terminates only PIDs that appeared after that
    snapshot, so the pool workers a step spawned are killed without touching
    anything else the app is running.

### 5. Sub-region (ROI) sessions

A channel can carry several named regions, each with its own config, checkpoints
and results:

*   `draw_roi()` — start a polygon on a shapes layer.
*   `confirm_roi()` — crop, rescale the config's dimensions to the crop, and
    switch the pipeline into the region's session directory.
*   `clear_roi()` — discard an in-progress polygon.
*   `open_roi_session()` / `delete_roi_session()` — reopen or remove a saved
    region.
*   `_switch_to_roi_mode()` / `_try_load_existing_roi_session()` — the internals
    of pointing the strategy at a session, including on reopen.

`confirm_roi` stores the polygon in full-image YX pixel coordinates (plus the Z
index it was drawn on), so the same record can be applied to any channel of the
sample. See [Sub-Regions](roi_regions.md).

## Key methods

| Method | Purpose |
| :--- | :--- |
| `create_step_widgets(step)` | Clear the panel and populate it for a step. |
| `clear_current_widgets()` | Empty the panel (used by the terminal state). |
| `execute_processing_step()` | Reconcile, then run the current step in a worker. |
| `valid_frontier()` / `can_*()` | Navigation and button state. |
| `go_back()` / `go_forward()` | Non-destructive navigation. |
| `is_current_step_dirty()` | Live widgets vs. last committed values. |
| `_ensure_config_canonical(step)` | Reconcile gate before computing. |
| `shutdown_and_cleanup()` | Ordered teardown, including child processes. |

## Design pattern

The manager observes the `ProcessingStrategy` interface. Because it never
references a specific step, the same manager drives both the 2D and 3D pipelines,
and a step could be added or reordered in a strategy without touching this file.
