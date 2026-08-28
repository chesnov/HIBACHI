# `segmentation_helpers.py`

**Location:** `utils/module_3d/segmentation_helpers.py`

## Overview

Low-level numerical and system utilities used by the cell-splitting step. Both
the 3D and 2D splitters import from here, so the watershed call, the distance
transform and the log format are defined once.

> **Despite living in `module_3d`, this is not 3D-only.**
> `cell_splitting_2d.py` imports `flush_print`, `_watershed_with_simpleitk` and
> `distance_transform_edt` from here, with local fallback definitions if the
> import fails. It is the only cross-module import between `module_2d` and
> `module_3d`.

## Functions

*   **`flush_print(*args, **kwargs)`** — `print` followed by
    `sys.stdout.flush()`. Without the flush, output still sitting in the buffer
    is lost when a native crash kills the process, so the log ends before the
    point of failure rather than at it.

*   **`log_memory_usage(label="")`** — prints the process's resident set size in
    GB via `psutil`, tagged `[MEM_PROFILE]`. Drop it into a loop to track memory
    growth across a long run.

*   **`_watershed_with_simpleitk(landscape, markers, log_prefix="")`** — runs
    SimpleITK's `MorphologicalWatershedFromMarkers`, which is faster and lighter
    than the skimage equivalent on large 3D integer arrays. Around that call it:
    *   replaces non-finite values in the landscape (`NaN` and `+inf` become the
        finite maximum, `-inf` becomes 0), since ITK will not accept them;
    *   casts to `float64` and `uint32` for ITK's template matching, then back to
        the markers' dtype;
    *   passes `markWatershedLine=False`, so basins meet directly instead of being
        separated by a one-pixel line;
    *   deletes the SimpleITK objects explicitly to release their C++ memory;
    *   on any failure, logs it and **returns the markers unchanged** — the caller
        gets unexpanded seeds rather than an exception.

*   **`distance_transform_edt(input, sampling=None, ..., output=None)`** — a copy
    of `scipy.ndimage.distance_transform_edt` extended with an `output`
    parameter, so the result can be written into an array that already exists —
    typically a `numpy.memmap` — instead of scipy allocating a new one. For a
    large volume that is the difference between writing to disk and needing the
    whole float array in RAM.

*   **`_distance_tranform_arg_check(...)`** — argument validation for the above,
    lifted from scipy alongside it.

## Used by

*   [Step 4: Cell Separation](cell_splitting.md) — the watershed wrapper for both
    the per-cell split and the stitch-conflict resolution, and the distance
    transform for the cost landscape.
