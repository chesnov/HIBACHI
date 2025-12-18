# Strategy Controller (`_3D_strategy.py` & `_2D_strategy.py`)

**Locations:**
*   `utils/module_3d/_3D_strategy.py`
*   `utils/module_2d/_2D_strategy.py`
*   **Base Class:** `utils/high_level_gui/processing_strategies.py`

## Overview
The **Strategy Controller** is the orchestrator of the entire segmentation pipeline. It implements the **Strategy Design Pattern**, decoupling the *definition* of the workflow from the *execution* (GUI or Batch Processor).

Whether running in 2D or 3D, the application relies on a concrete subclass of `ProcessingStrategy` to:
1.  Define the sequence of steps.
2.  Manage file paths and checkpoints.
3.  Pass data (images, thresholds) between steps.
4.  Handle visualization in Napari.

## The Workflow Definitions

Both strategies define a strict 6-step pipeline. The GUI uses `get_step_definitions()` to dynamically generate the sidebar.

| Step | 3D Method | 2D Method | Artifact Created |
| :--- | :--- | :--- | :--- |
| **1** | `execute_raw_segmentation` | `execute_raw_segmentation_2d` | `raw_segmentation.dat` |
| **2** | `execute_trim_edges` | `execute_trim_edges_2d` | `trimmed_segmentation.dat` |
| **3** | `execute_soma_extraction` | `execute_soma_extraction_2d` | `cell_bodies.dat` |
| **4** | `execute_cell_separation` | `execute_cell_separation_2d` | `final_segmentation.dat` |
| **5** | `execute_calculate_features` | `execute_calculate_features_2d` | `metrics_df.csv` |
| **6** | `execute_interaction_analysis` | `execute_interaction_analysis` | `intersection_REF.dat` |

## Core Architectures

### 1. Memory Mapping (The `.dat` Files)
To handle terabyte-scale datasets on consumer hardware, the Strategy classes **never** hold the full segmentation history in RAM.
*   **Input:** The raw image is passed by reference.
*   **Intermediate:** Every step writes its result to a memory-mapped file on the hard drive (`.dat`).
*   **Hand-off:** Step $N$ closes its write handle, and Step $N+1$ opens a read handle to that file.
*   **Cleanup:** The `_close_memmap` helper ensures file locks are released immediately so the OS can flush buffers or delete temp files.

### 2. Intermediate State Passing
While images are stored on disk, small runtime variables (like calculated thresholds) must be passed between steps in memory.
*   **`self.intermediate_state`:** A dictionary shared across all steps.
*   *Example:* Step 1 calculates an automated `segmentation_threshold`. This float is stored in `intermediate_state` and retrieved by Step 2 to decide which edge pixels are "bright enough" to save.

### 3. Checkpointing & Resume
The `ProcessingStrategy` base class implements a robust crash-recovery system.
*   **`get_checkpoint_files()`:** Returns a dictionary mapping logical keys (e.g., "final_segmentation") to physical file paths.
*   **`get_last_completed_step()`:** Scans the disk. If `trimmed_segmentation.dat` exists but `cell_bodies.dat` is missing, it knows Step 2 is done but Step 3 failed.
*   **Batch Processor Usage:** When the Batch Processor starts a folder, it asks the Strategy: "Where did we leave off?" and skips directly to the unfinished step.

### 4. Visualization Abstraction
The GUI does not know how to display the data. It calls `load_checkpoint_data(viewer, step)`, and the Strategy handles the Napari layers.
*   **3D Strategy:** Sets `scale` based on Z-anisotropy.
*   **2D Strategy:** Sets `scale` based on YX spacing.
*   It ensures layer naming consistency (`Ref: Name`, `Overlap Regions`) so the user always sees a clean interface.

## Key Methods

### `execute_step(...)`
The template method called by the worker thread.
1.  Validates the step index.
2.  Lookups the method name dynamically.
3.  Executes the specific logic (e.g., `segment_cells_first_pass_raw`).
4.  Updates the `intermediate_state`.
5.  Persists the result to the predefined checkpoint path.

### `cleanup_step_artifacts(...)`
Used when the user clicks "Previous Step" or "Restart". It selectively deletes the `.dat` files and CSVs associated with specific steps to ensure a clean state for re-processing.