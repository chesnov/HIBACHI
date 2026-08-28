# Config Library

**Corresponding modules:**
*   `utils/high_level_gui/config_library.py` — the library and reconciliation
    (Qt-free)
*   `utils/high_level_gui/config_library_dialog.py` — the manager window
*   `utils/high_level_gui/reconcile_dialog.py` — the change-confirmation prompt

## What this is for

A HIBACHI config is a YAML file holding both the parameters and their values.
Configs otherwise live inside a project, so a set of parameters tuned in one
project is not reachable from another. The library stores them outside any
project, where every project can see them.

Open it from **Config Library** in the project window.

## Where it lives

```
$HIBACHI_STATE_DIR/configs/2d/<name>.yaml     mode fluorescence_2d
$HIBACHI_STATE_DIR/configs/3d/<name>.yaml     mode fluorescence
```

defaulting to `~/.hibachi/configs/`. That is outside the repository, so the
library survives a `git pull`, an update and a rollback, and is shared by every
project on the machine.

The mode subfolder is for tidiness only — the mode is always read from the file
itself (its top-level `mode:`, falling back to the `_2d` suffix on its step
keys), so a file that ends up in the wrong folder still resolves correctly.

## Two sources, shown together

The manager lists **Built-in** configs shipped with HIBACHI, which are read-only,
alongside **My Library** — yours, which you can modify. A built-in and a
same-named library config both appear rather than one hiding the other.

A config file that cannot be read is reported as an explicit problem
(`scan_problems()`), not omitted from the list. A missing entry with no
explanation is harder to diagnose than a named error.

## What you can do

| Action | Notes |
| :--- | :--- |
| **Save current…** | Store the config open in the viewer under a name. |
| **Import…** | Copy a config file in from anywhere. |
| **Duplicate** | Copy an entry, including a built-in, under a new name. |
| **Rename** | Library entries only. |
| **Delete** | Library entries only. |
| **Export preset…** | Copy an entry out to share. |
| **Reveal in file browser** | Open the containing folder. |

Saving and importing refuse to overwrite an existing name unless you confirm.
Built-ins raise rather than being modified in place; **Duplicate** is the way to
base something on one.

### What a saved preset does and does not carry

Saving to the library **strips** the keys that belong to one image rather than to
a parameter set: `saved_state` (computed thresholds), `voxel_dimensions` /
`pixel_dimensions` (calibration), `dimensions_source` and `synthetic`. A preset
carrying another image's dimensions would apply them wherever it was used.

`hibachi_version` is deliberately **kept**: it records which pipeline version the
preset was tuned on, which is useful and safe to share.

### Exporting a run instead of a preset

**Export run config** in the project window is a different operation. It copies a
*processed* run's config verbatim, stripping nothing — `saved_state`, the
dimensions and `hibachi_version` all travel — so the exact run can be
reproduced. Use that for reproducibility, and the library for reuse.

---

## Reconciliation

The config is both schema and values: every label, `type`, `min`, `max`, `step`
and default lives in the YAML, and the parameter controls are built from it.
There is no separate schema in the code to validate against.

So the **built-in `default.yaml` for each mode is the canonical reference**, and
`reconcile()` merges a config against it:

*   **Structure** comes from the reference — which steps and parameters exist,
    and their labels and bounds.
*   **Values** are carried over from your config wherever the keys match.
*   **Non-parameter keys you own** — dimensions, `saved_state` — are preserved.
*   `mode` is normalised to the reference's.

Every difference is recorded rather than applied silently, as a list of:

| Kind | Meaning |
| :--- | :--- |
| step added | The pipeline defines a step your config lacked; defaults filled in. |
| step removed | Your config had a step the pipeline no longer defines; dropped. |
| `added` / `removed` | Same, for one parameter. |
| `type_changed` | The parameter's type differs from the reference's. |
| `clamped` | Your value fell outside the reference's `min`/`max` and was pulled in. |

`ReconcileResult.is_clean` is true when there is nothing to report. Otherwise the
caller shows `summary_lines()` for confirmation before anything is written — see
`reconcile_dialog.py`.

If a mode has no built-in reference, `reconcile()` raises
`ReferenceMissingError` rather than passing the config through unchecked.

### When it runs

`gui_manager._ensure_config_canonical()` calls it before **every** Process. So
reconciliation gates *computing*, not *viewing*: a config from an earlier
pipeline version opens, displays and can be analysed as it is. You are stopped
only from computing new results against a structure that no longer matches.

When reconciling invalidates results already on disk, the affected steps are
named in the prompt and cleared if you accept, so nothing on disk claims to come
from parameters that were not used to produce it.

---

## Where to go next

*   [Tuning workflow](tuning_workflow.md) — arriving at a config worth saving.
*   [GUI Manager](gui_manager.md#3-config-canonicalisation-before-compute) — the
    reconcile gate in the processing flow.
*   [Batch Processing](batch_processor.md) — applying one config across a project
    with **Set New Channel Config**, then processing.
