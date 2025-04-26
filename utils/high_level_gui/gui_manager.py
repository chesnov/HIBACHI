# --- START OF FILE utils/high_level_gui/gui_manager.py ---

import numpy as np
import os
import time
from PyQt5.QtWidgets import QMessageBox # type: ignore
import yaml # type: ignore
from typing import Dict, Any, List # Keep List
import sys
import traceback
import gc

# --- Assumed relative imports based on structure ---
from ..nuclear_module_3d._3D_nuclear_strategy import NuclearStrategy
from ..ramified_module_3d._3D_ramified_strategy import RamifiedStrategy
from ..high_level_gui.processing_strategies import ProcessingStrategy # Import base class for type hint

try:
    # Import helpers from the SAME directory (.)
    from .helper_funcs import create_parameter_widget, check_processing_state
except ImportError as e:
    expected_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Error importing helper_funcs in gui_manager.py: {e}")
    print(f"Ensure helper_funcs.py is in the directory: {expected_dir}")
    raise
# --- End Imports ---

class DynamicGUIManager:
    """
    Manages the GUI interactions, processing workflow execution,
    parameter handling, and checkpoint restoration.
    """

    def __init__(self, viewer, config, image_stack, file_loc, processing_mode):
        """
        Initializes the GUI Manager.

        Args:
            viewer: The Napari viewer instance.
            config (dict): The initial configuration loaded from YAML.
            image_stack (np.ndarray): The 3D image data.
            file_loc (str): The absolute path to the input image file.
            processing_mode (str): The selected mode ('nuclei' or 'ramified').
        """
        self.viewer = viewer
        self.initial_config = config.copy() # Keep original defaults
        self.config = config.copy() # Working config, may be updated by loaded state/GUI
        self.image_stack = image_stack
        self.file_loc = file_loc
        self.processing_mode = processing_mode
        self.current_widgets: Dict[Any, Any] = {} # Stores {dock_widget: magicgui_widget}
        self.current_step = {"value": 0} # 0-based index of the *next* step to run
        self.parameter_values: Dict[str, Any] = {} # Stores current params for active step UI

        # --- Setup Directories ---
        self.inputdir = os.path.dirname(self.file_loc)
        self.basename = os.path.basename(self.file_loc).split('.')[0]
        self.processed_dir = os.path.join(self.inputdir, f"{self.basename}_processed_{self.processing_mode}")
        # Note: Directory creation is handled within Strategy __init__

        # --- Calculate Spacing ---
        self._calculate_spacing()

        # --- Instantiate Strategy ---
        self.strategy: ProcessingStrategy # Type hint for clarity
        try:
            strategy_class = {'nuclei': NuclearStrategy, 'ramified': RamifiedStrategy}.get(self.processing_mode)
            if not strategy_class:
                raise ValueError(f"Unsupported processing mode: {self.processing_mode}")

            # Strategy __init__ gets the working config, calculates steps/num_steps
            self.strategy = strategy_class(
                self.config, # Pass the potentially updatable config
                self.processed_dir,
                self.image_stack.shape,
                self.spacing,
                self.z_scale_factor
            )
            # --- Get steps FROM strategy ---
            self.processing_steps: List[str] = self.strategy.get_step_names()
            self.num_steps: int = self.strategy.num_steps
            self.step_display_names: Dict[str, str] = {
                name: name.replace('execute_', '').replace('_', ' ').title()
                for name in self.processing_steps
            }
            print(f"Initialized strategy for '{self.processing_mode}' with {self.num_steps} steps: {self.processing_steps}")

        except Exception as e:
             print(f"FATAL ERROR initializing processing strategy: {e}")
             traceback.print_exc()
             # Optionally, show a critical error message to the user here
             raise # Re-raise to stop execution if strategy fails

        # --- Initialize Viewer ---
        self._initialize_layers()

        # --- Restore State (Loads config, transfers state, decides flow) ---
        self.restore_from_checkpoint()


    def _calculate_spacing(self):
        """Calculates voxel spacing and Z-scale factor from self.config."""
        voxel_dims = self.config.get('voxel_dimensions', {})
        try:
            vx = float(voxel_dims.get('x', 1.0))
            vy = float(voxel_dims.get('y', 1.0))
            vz = float(voxel_dims.get('z', 1.0))
        except (ValueError, TypeError):
            print("Warning: Invalid voxel dimensions in config. Using defaults (1.0, 1.0, 1.0).")
            vx, vy, vz = 1.0, 1.0, 1.0

        shape = self.image_stack.shape
        # Prevent division by zero if shape dimension is 0 or 1
        self.x_spacing = vx / shape[2] if shape and len(shape)>2 and shape[2] > 0 else vx
        self.y_spacing = vy / shape[1] if shape and len(shape)>1 and shape[1] > 0 else vy
        self.z_spacing = vz / shape[0] if shape and len(shape)>0 and shape[0] > 0 else vz
        self.spacing = [self.z_spacing, self.x_spacing, self.y_spacing] # ZXY order
        # Prevent division by zero for scale factor
        self.z_scale_factor = self.z_spacing / self.x_spacing if self.x_spacing > 1e-9 else 1.0
        print(f"Calculated Spacing (Z, X, Y): {self.spacing}")
        print(f"Calculated Z Scale Factor for display: {self.z_scale_factor:.4f}")


    def _initialize_layers(self):
        """Adds the initial base image layer to the viewer."""
        layer_name = f"Original stack ({self.processing_mode} mode)"
        # Remove layer if it exists from a previous run/restart
        if layer_name in self.viewer.layers:
            try: self.viewer.layers.remove(layer_name)
            except Exception as e: print(f"Note: Error removing existing original layer: {e}")
        # Add the image
        try:
            self.viewer.add_image(
                self.image_stack,
                name=layer_name,
                scale=(self.z_scale_factor, 1, 1) # Apply Z scaling for display
            )
        except Exception as e:
             print(f"ERROR adding original image layer: {e}")
             traceback.print_exc()


    def restore_from_checkpoint(self):
        """
        Checks for existing files, loads the saved configuration,
        transfers relevant saved state back to the strategy object,
        and determines the next step in the workflow.
        """
        checkpoint_files = self.strategy.get_checkpoint_files()
        checkpoint_step = check_processing_state(self.processed_dir, self.processing_mode, checkpoint_files, self.num_steps)

        loaded_config_path = None # Track if config was loaded

        # --- Load saved config first (if checkpoint > 0) ---
        if checkpoint_step > 0:
            print(f"Found completed steps up to step {checkpoint_step}/{self.num_steps}.")
            config_file = checkpoint_files.get("config")
            if config_file and os.path.exists(config_file):
                try:
                    print(f"Loading saved configuration from: {config_file}")
                    with open(config_file, 'r') as file: saved_config = yaml.safe_load(file)
                    if saved_config and isinstance(saved_config, dict):
                        # Update self.config (manager's copy)
                        for step_key, step_data in saved_config.items():
                            # (logic to update self.config parameters and saved_state dict - unchanged)
                            if step_key == 'voxel_dimensions' and isinstance(step_data, dict):
                                if step_key in self.config: self.config[step_key] = step_data
                            elif step_key.startswith("execute_") and step_key in self.config and isinstance(step_data, dict) and isinstance(self.config.get(step_key), dict):
                                if "parameters" in step_data and isinstance(step_data.get("parameters"), dict) and \
                                   "parameters" in self.config[step_key] and isinstance(self.config[step_key].get("parameters"), dict):
                                     for param_name, param_data in step_data["parameters"].items():
                                         if param_name in self.config[step_key]["parameters"]:
                                             value_to_set = param_data.get('value', param_data) if isinstance(param_data, dict) else param_data
                                             # Attempt type conversion if needed... (simplified here)
                                             self.config[step_key]["parameters"][param_name]['value'] = value_to_set
                            elif step_key == 'saved_state' and isinstance(step_data, dict):
                                 self.config['saved_state'] = step_data # Store loaded state in manager's config

                        print(f"Loaded and applied saved parameters to manager's config from {config_file}")
                        loaded_config_path = config_file

                        # --- Update strategy's config reference ---
                        self.strategy.config = self.config
                        # --- RECALCULATE MANAGER'S SPACING based on loaded config ---
                        self._calculate_spacing()
                        # --- !!! ADD THIS BLOCK: UPDATE STRATEGY'S SPACING/SCALE !!! ---
                        print("Updating strategy object's spacing and scale factor from loaded config...")
                        self.strategy.spacing = self.spacing # Update strategy's spacing
                        self.strategy.z_scale_factor = self.z_scale_factor # Update strategy's scale factor
                        print(f"  Strategy spacing updated to: {self.strategy.spacing}")
                        print(f"  Strategy z_scale_factor updated to: {self.strategy.z_scale_factor:.4f}")
                        # --- !!! END ADDED BLOCK !!! ---

                    else: print(f"Warning: Saved config file {config_file} was empty or invalid.")
                except Exception as e: print(f"Error loading/applying saved config: {e}"); traceback.print_exc()
            else: print(f"No saved config file found ({config_file}). Using parameters possibly modified since last run."); self.strategy.config = self.config # Ensure strategy has current config

            # --- Transfer specific saved state ---
            loaded_state = self.config.get('saved_state', {})
            if loaded_state:
                print(f"Transferring loaded state to strategy: {list(loaded_state.keys())}")
                if 'segmentation_threshold' in loaded_state:
                    try:
                        thresh_val = float(loaded_state['segmentation_threshold'])
                        self.strategy.intermediate_state['segmentation_threshold'] = thresh_val
                        print(f"  Restored 'segmentation_threshold' ({thresh_val:.4f}) to strategy.intermediate_state.")
                    except Exception as e: print(f"  Warn: Could not restore 'segmentation_threshold': {e}")
            else: print("No 'saved_state' found in loaded config to transfer.")

            # --- Ask user how to proceed ---
            if checkpoint_step == self.num_steps: # All completed
                # ... (unchanged message box logic) ...
                msg=QMessageBox();msg.setIcon(QMessageBox.Information);msg.setWindowTitle("Status");msg.setText(f"All {self.num_steps} steps complete.");msg.setInformativeText("View results or restart?");msg.setStandardButtons(QMessageBox.NoButton);view=msg.addButton("View",QMessageBox.YesRole);restart=msg.addButton("Restart",QMessageBox.NoRole);msg.exec_();
                if msg.clickedButton()==view: self.load_checkpoint_data(checkpoint_step);self.current_step["value"]=checkpoint_step;print("Displaying results.")
                else: self._confirm_restart()
            else: # Partial completion
                # ... (unchanged message box logic) ...
                last_idx=checkpoint_step-1; next_idx=checkpoint_step; last_name=self.step_display_names.get(self.processing_steps[last_idx],f"Step {checkpoint_step}"); next_name=self.step_display_names.get(self.processing_steps[next_idx],f"Step {next_idx+1}")
                msg=QMessageBox();msg.setIcon(QMessageBox.Question);msg.setWindowTitle("Resume?");msg.setText(f"Finished {last_name}.");msg.setInformativeText(f"Resume from {next_name} or restart?");msg.setStandardButtons(QMessageBox.NoButton);resume=msg.addButton("Resume",QMessageBox.YesRole);restart=msg.addButton("Restart",QMessageBox.NoRole);msg.exec_();
                if msg.clickedButton() == resume:
                    # Restore non-serializable state needed for resume
                    if 'original_volume_ref' not in self.strategy.intermediate_state: self.strategy.intermediate_state['original_volume_ref'] = self.image_stack; print("Restored volume ref for resume.")
                    # Load data using the NOW UPDATED strategy scale factor
                    self.load_checkpoint_data(checkpoint_step);
                    self.current_step["value"] = next_idx;
                    self.create_step_widgets(self.processing_steps[next_idx])
                else: self._confirm_restart()
        else: # Start from beginning
            print("No existing files found. Starting fresh."); self.config = self.initial_config.copy(); self.strategy.config = self.config; self.strategy.intermediate_state = {}; self.create_step_widgets(self.processing_steps[0])

    def _confirm_restart(self):
        """Asks user to confirm deleting files and restarts processing."""
        confirm_box = QMessageBox(); confirm_box.setIcon(QMessageBox.Warning); confirm_box.setWindowTitle("Confirm Restart"); confirm_box.setText(f"Restart '{self.processing_mode}' processing?"); confirm_box.setInformativeText(f"This will delete existing processed files in:\n{self.processed_dir}\n\nAre you sure?"); confirm_box.setStandardButtons(QMessageBox.NoButton)
        yes_button = confirm_box.addButton("Yes, delete & restart", QMessageBox.YesRole); no_button = confirm_box.addButton("No, cancel", QMessageBox.NoRole); confirm_box.exec_()
        if confirm_box.clickedButton() == yes_button:
            print("User chose to delete files and restart."); self.delete_all_checkpoint_files(); self.current_step["value"] = 0; self.strategy.intermediate_state = {}; print("Cleaning all artifacts...");
            for i in range(1, self.num_steps + 1): self.strategy.cleanup_step_artifacts(self.viewer, i)
            self.config = self.initial_config.copy(); self.strategy.config = self.config; # Reset config
            self._initialize_layers(); self.create_step_widgets(self.processing_steps[0]) # Reset UI
        else:
            print("User cancelled restart. Reloading state to be safe."); self.restore_from_checkpoint()


    def delete_all_checkpoint_files(self):
        """Deletes all checkpoint files defined by the strategy."""
        print(f"Attempting to delete all checkpoint files for mode: {self.processing_mode} in {self.processed_dir}");
        files_to_delete = self.strategy.get_checkpoint_files()
        # Close known memmap layers before deleting underlying files
        memmap_layer_keys = ["Raw Intermediate Segmentation", "Trimmed Intermediate Segmentation", "Final segmentation"] # Base names
        for key_base in memmap_layer_keys:
             layer_name = f"{key_base}_{self.strategy.mode_name}"
             if layer_name in self.viewer.layers and isinstance(self.viewer.layers[layer_name].data, np.memmap):
                  print(f"Closing memmap layer {layer_name} before delete.")
                  try:
                      memmap_obj = self.viewer.layers[layer_name].data
                      if hasattr(memmap_obj, '_mmap') and memmap_obj._mmap is not None: memmap_obj._mmap.close()
                      self.viewer.layers.remove(layer_name) # Remove layer after closing handle
                  except Exception as e: print(f"Warn: Error closing/removing memmap layer {layer_name}: {e}")
        gc.collect(); time.sleep(0.1) # Allow time for OS release

        # Delete files
        deleted_count = 0
        for key, file_path in files_to_delete.items():
            if self.strategy._remove_file_safely(file_path): # Use strategy's helper
                 deleted_count += 1
        print(f"Attempted deletion for {len(files_to_delete)} checkpoint keys (deleted {deleted_count} existing files).")


    def load_checkpoint_data(self, checkpoint_step: int):
        """Loads data artifacts using the strategy."""
        try:
            print(f"Loading data artifacts up to completed step {checkpoint_step}...")
            # Delegate entirely to the strategy's implementation
            self.strategy.load_checkpoint_data(self.viewer, checkpoint_step)
        except Exception as e:
            print(f"FATAL ERROR during load_checkpoint_data: {str(e)}")
            traceback.print_exc()


    def cleanup_step(self, step_number: int):
        """Cleans up artifacts for a specific step (1-based) using the strategy."""
        print(f"Requesting cleanup for artifacts of step {step_number}")
        # Delegate entirely to the strategy's implementation
        self.strategy.cleanup_step_artifacts(self.viewer, step_number)


    def execute_processing_step(self):
        """Executes the next processing step defined by the strategy."""
        start_time = time.time()
        step_index = self.current_step["value"] # 0-based index of step to run

        if step_index >= self.num_steps:
            print("All processing steps already completed."); msg = QMessageBox(); msg.setIcon(QMessageBox.Information); msg.setText("Processing complete."); msg.exec_(); return

        step_method_name = self.processing_steps[step_index]
        step_display_name = self.step_display_names.get(step_method_name, f"Step {step_index + 1}")
        success = False

        try:
            # 1. Get current parameters from GUI/state
            current_values = self.get_current_values()

            # --- MOVED: Save config *before* cleanup & execution (still might be needed) ---
            # Although the key state is set *during* the step, saving params *before*
            # running is still useful if the run fails midway. Let's keep this save.
            print(f"Saving parameters before executing {step_display_name}...")
            self.strategy.save_config(self.config)
            # --- END MOVE ---

            # 2. Clean up artifacts from this step and any subsequent steps
            print(f"Cleaning up potential old artifacts for {step_display_name} (Step {step_index+1}) onwards...")
            for i in range(step_index + 1, self.num_steps + 1):
                 self.cleanup_step(i) # Pass 1-based step number

            # 3. Execute the step using the strategy's generic method
            success = self.strategy.execute_step(
                step_index=step_index,
                viewer=self.viewer,
                image_stack_or_none=self.image_stack,
                params=current_values
            )

            # 4. Handle results
            if success:
                # --- ADDED: Save config AGAIN *after* successful step execution ---
                # This ensures any state calculated *during* the step (like segmentation_threshold)
                # is included in the saved YAML file.
                print(f"Step successful. Saving updated configuration and state after {step_display_name}...")
                self.strategy.save_config(self.config)
                # --- END ADDED SAVE ---

                self.current_step["value"] += 1 # Advance index for the *next* step
                if self.current_step["value"] < self.num_steps:
                    next_step_internal_name = self.processing_steps[self.current_step["value"]]
                    self.create_step_widgets(next_step_internal_name)
                else: # Processing finished
                    print("\n*** Processing complete! ***")
                    # Config already saved after last successful step
                    self.clear_current_widgets()
                    msg = QMessageBox(); msg.setIcon(QMessageBox.Information); msg.setWindowTitle("Complete"); msg.setText(f"All '{self.processing_mode}' steps finished."); msg.exec_()
            else: # Step explicitly returned False
                 print(f"{step_display_name} failed or returned False.")
                 msg = QMessageBox(); msg.setIcon(QMessageBox.Warning); msg.setWindowTitle("Step Not Completed"); msg.setText(f"Step '{step_display_name}' did not complete successfully."); msg.setInformativeText("Check console output. Adjust parameters and click 'Run Current Step' again."); msg.exec_()

        except Exception as e: # Catch unexpected errors during the flow
            print(f"FATAL ERROR during processing flow for {step_display_name}: {str(e)}")
            traceback.print_exc()
            msg = QMessageBox(); msg.setIcon(QMessageBox.Critical); msg.setWindowTitle("Processing Error"); msg.setText(f"An unexpected error occurred during '{step_display_name}'."); msg.setInformativeText(f"Error: {str(e)}\n\nCheck console output."); msg.exec_()
            # Do not advance step on general error

        finally:
            end_time = time.time(); print(f"'{step_display_name}' execution attempt took {end_time - start_time:.2f} seconds.")


    def clear_current_widgets(self):
        """Removes all currently displayed parameter widgets."""
        dock_widgets_to_remove = list(self.current_widgets.keys())
        for dock_widget in dock_widgets_to_remove:
            try:
                # Use Napari's removal method, which should handle vanished widgets
                self.viewer.window.remove_dock_widget(dock_widget)
            except Exception as e:
                 # Log unexpected errors during removal but don't stop
                 print(f"Warning: Error removing dock widget {dock_widget}: {e}")
            # Always remove from our tracking dictionary
            self.current_widgets.pop(dock_widget, None) # Use pop for safety
        self.current_widgets.clear()


    def create_step_widgets(self, step_method_name: str):
        """Creates and displays widgets for the parameters of a given step."""
        self.clear_current_widgets()
        self.parameter_values = {} # Reset dict for current step's params

        config_key = self.strategy.get_config_key(step_method_name)
        step_display_name = self.step_display_names.get(step_method_name, step_method_name)

        if config_key not in self.config:
            print(f"Error: Config key '{config_key}' not found for step '{step_display_name}'. Cannot create widgets.")
            return

        step_config = self.config.get(config_key, {})
        parameters = step_config.get("parameters") # Use .get for safety

        # Check if parameters exist and are a dictionary
        if not isinstance(parameters, dict) or not parameters:
            print(f"Warning: No valid 'parameters' dict found for '{config_key}'. No parameter widgets created for '{step_display_name}'.")
            # Optionally, add a label indicating no parameters for this step
            return

        print(f"Creating widgets for: '{step_display_name}' (using config key: '{config_key}')")
        for param_name, param_config in parameters.items():
            if not isinstance(param_config, dict):
                print(f"Warning: Parameter config for '{param_name}' ('{config_key}') is not a dict. Skipping.")
                continue
            try:
                # Value from self.config should be up-to-date (loaded or default)
                current_value = param_config.get('value')
                # Callback updates self.config and self.parameter_values
                callback = lambda value, key=config_key, pn=param_name: self.parameter_changed(key, pn, value)
                # Create the widget
                widget = create_parameter_widget(param_name, param_config, callback)
                # Add widget to viewer
                dock_widget = self.viewer.window.add_dock_widget(widget, area="right", name=f"{step_display_name}: {param_name}")
                self.current_widgets[dock_widget] = widget # Track widget
                self.parameter_values[param_name] = current_value # Store value for current step run
            except Exception as e:
                print(f"ERROR creating widget for parameter '{param_name}' in step '{step_display_name}': {str(e)}")
                traceback.print_exc()


    def parameter_changed(self, config_key: str, param_name: str, value: Any):
        """Callback when a parameter widget changes value."""
        try:
            # Update the main configuration dictionary (self.config)
            # Check structure defensively
            if config_key in self.config and \
               isinstance(self.config.get(config_key), dict) and \
               "parameters" in self.config[config_key] and \
               isinstance(self.config[config_key].get("parameters"), dict) and \
               param_name in self.config[config_key]["parameters"] and \
               isinstance(self.config[config_key]["parameters"].get(param_name), dict):
                # Update the 'value' field
                self.config[config_key]["parameters"][param_name]["value"] = value
            else:
                 print(f"Warning: Config path '{config_key} -> parameters -> {param_name}' structure invalid during update.")

            # Update the dictionary holding current values for the active step run
            self.parameter_values[param_name] = value
        except Exception as e:
             print(f"Error handling parameter change for {config_key}/{param_name}: {e}")


    def get_current_values(self) -> Dict[str, Any]:
        """Returns a copy of the current parameter values for the active step."""
        return self.parameter_values.copy()
        

# --- END OF FILE utils/high_level_gui/gui_manager.py ---