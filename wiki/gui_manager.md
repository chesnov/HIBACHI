# gui_manager.py

**Location:** `utils/high_level_gui/gui_manager.py`

## Overview
The `DynamicGUIManager` class is the bridge between the logic (Strategy) and the user interface (Napari). It creates the interactive "Step-by-Step" sidebar.

## Responsibilities
1.  **Dynamic Widget Generation:**
    *   Reads the parameters required for the current step from the YAML config.
    *   Uses `magicgui` to create sliders, dropdowns, and input boxes for those parameters automatically.
    *   Updates the in-memory configuration when the user changes a value.
2.  **State Management:**
    *   Tracks the `current_step` index.
    *   Determines if the user can proceed to the next step.
    *   Handles "Resume" logic (loading data from disk for completed steps).
3.  **Execution:**
    *   Calls `strategy.execute_step()` when the "Run" button is clicked.
    *   Catches errors and displays them in a popup message box.
    *   Triggers cleanup of old layers when moving between steps.

## Key Methods
*   `create_step_widgets(step_name)`: Clears the sidebar and populates it with controls for the new step.
*   `execute_processing_step()`: Runs the current logic.
*   `restore_from_checkpoint()`: Checks the Strategy's state to see if we can fast-forward.

## Design Pattern
This class observes the `ProcessingStrategy` abstract base class. It does not know *what* the steps are (e.g., "Raw Segmentation" vs "Edge Trim"); it just knows there is a list of steps and executes them in order.