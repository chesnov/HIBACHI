# segmentation_helpers.py

**Location:** `utils/module_3d/segmentation_helpers.py`

## Overview
A utility module containing low-level mathematical and system functions used by both the Soma Extraction and Cell Splitting modules. Centralizing these functions ensures consistent behavior (e.g., logging format, distance transform precision) across the pipeline.

## Key Functions
*   `_watershed_with_simpleitk`: A robust wrapper around SimpleITK's morphological watershed. It handles type casting, `NaN` protection, and memory efficiency better than `skimage.segmentation.watershed` for large 3D volumes.
*   `distance_transform_edt`: A modified version of Scipy's Euclidean Distance Transform that allows writing the result directly into a `numpy.memmap` (disk) to avoid RAM spikes.
*   `flush_print`: Forces immediate console output, crucial for tracking progress in long-running batch jobs where stdout buffering might hide crashes.
*   `log_memory_usage`: A diagnostic tool to print current RAM consumption.