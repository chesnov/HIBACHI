# BatchProcessor.py

**Location:** `utils/high_level_gui/batch_processor.py`

## Overview
The `BatchProcessor` class handles the automated, sequential processing of multiple image folders. It acts as a bridge between the file system (iterating over folders) and the specific segmentation logic (Strategies).

## Key Responsibilities
1.  **Iteration:** Loops through a list of folders identified by the `ProjectManager`.
2.  **Strategy Selection:** dynamic instantiation of the correct processing strategy (e.g., `RamifiedStrategy`, `Ramified2DStrategy`) based on the `mode` defined in the folder's YAML config.
3.  **Memory Management:** 
    *   Loads the heavy image stack (`.tif`) into RAM.
    *   Passes it to the strategy.
    *   **Crucial:** Ensures the image stack is deleted and garbage collected (`gc.collect()`) immediately after processing a folder to prevent RAM saturation on laptops.
4.  **State Management:** Checks `checkpoint` files to determine if a folder is already finished or if it should resume from a specific step.

## Class: `BatchProcessor`

### `__init__(self, project_manager)`
*   **Input:** `project_manager` (Object holding the list of valid image folders).
*   **Action:** Registers supported strategies (Ramified 3D, Ramified 2D).

### `_calculate_spacing_for_batch(self, config, image_shape)`
*   **Purpose:** Derives physical voxel size (microns) from the YAML config.
*   **Logic:** Handles both 2D and 3D logic to calculate the `z_scale_factor` used for aspect-ratio correct visualization and isotropic processing.

### `process_single_folder(self, folder_path, target_strategy_key, force_restart)`
*   **Purpose:** Orchestrates the processing of ONE specific folder.
*   **Memory Safety:** Uses a `try...finally` block to strictly delete the `image_stack` numpy array and the `strategy_instance` after execution, regardless of success or failure.
*   **Workflow:**
    1.  Reads YAML config.
    2.  Loads TIFF image (High Memory Usage Event).
    3.  Instantiates the specific Strategy class.
    4.  Checks existing checkpoint files to see which steps are done.
    5.  Iterates through the remaining steps, calling `strategy_instance.execute_step`.

### `process_all_folders(self, force_restart_all)`
*   **Purpose:** The public entry point called by the GUI.
*   **Logic:** 
    1.  Scans all folders in the project.
    2.  Filters them by supported modes.
    3.  Calls `process_single_folder` one by one.
    4.  Aggregates success/failure statistics.

## Dependencies
*   `tifffile`: For loading microscopy data.
*   `gc`: Garbage collection interface (used aggressively).
*   `processing_strategies`: Base class and helper functions.
