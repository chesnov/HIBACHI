# Batch Processor (`batch_processor.py`)

**Location:** `utils/high_level_gui/batch_processor.py`

## Overview
The `BatchProcessor` is the engine behind the "Process All Compatible Folders" button. It allows for "Set and Forget" automation, enabling you to process hundreds of datasets sequentially without user intervention.

It acts as a **Headless Controller**: it instantiates the appropriate Strategy (2D or 3D) for each folder but provides `None` instead of a Napari viewer, ensuring that the pipeline runs purely on disk/CPU/RAM without rendering overhead.

## Key Responsibilities

### 1. Project Iteration
*   It asks the `ProjectManager` for a list of valid image folders.
*   It filters them: checks if the `mode` defined in their `processing_config.yaml` is supported (e.g., `ramified` or `ramified_2d`).

### 2. Spacing Normalization
*   Before processing, it reads the physical dimensions from the config.
*   **3D Mode:** Calculates `(Z, Y, X)` spacing and the Z-anisotropy scale factor.
*   **2D Mode:** Calculates `(1.0, Y, X)` spacing. It forces Z=1.0 to ensure 3D functions (like distance transforms) work mathematically on 2D planes without errors.

### 3. Memory Safety
Batch processing terabytes of data is prone to "Memory Leaks" (RAM filling up over time). The Batch Processor handles this strictly:
*   **Explicit Deletion:** It manually deletes the heavy image array and the Strategy instance after every folder.
*   **Garbage Collection:** It forces `gc.collect()` between steps and between folders to reclaim memory immediately.

### 4. Smart Resume (Checkpointing)
*   It checks which `.dat` files already exist in the folder.
*   If a folder crashed at Step 3, the Batch Processor detects `trimmed_segmentation.dat` exists and skips Steps 1 & 2, resuming instantly at Step 3.

---

## Usage logic

### `process_all_folders(force_restart_all=False)`
The main entry point.
*   **`force_restart_all=False` (Default):** The "Resume" mode. It will skip any folder that is already 100% complete. It will finish any folder that is partially complete.
*   **`force_restart_all=True`:** The "Nuclear" option. It deletes **ALL** processing outputs for **ALL** folders and starts everything from scratch. Use with caution.

### `process_single_folder(...)`
*   **Input:** Path to a specific folder.
*   **Logic:**
    1.  Validates files (`.tif`, `.yaml`).
    2.  Loads the Image (RAM usage spikes here).
    3.  Initializes the correct Strategy (2D or 3D).
    4.  Runs the loop: `strategy.execute_step(...)`.
    5.  **Clean up:** Deletes the image from RAM immediately.

---

## Interaction Analysis in Batch Mode
To perform **Step 6 (Interaction Analysis)** in batch mode:
1.  Open **one** image in the GUI.
2.  Go to Step 6.
3.  Click "Analyze with Other Channels..." and select the reference project.
4.  **Save** (Run the step). This writes the `target_channel_folder` path into the `processing_config.yaml`.
5.  **Propagate:** Copy this line in the yaml config to your other folders (or use a script to update them).
6.  Run Batch Processing. The processor will see the config setting and execute the interaction analysis for every folder automatically.