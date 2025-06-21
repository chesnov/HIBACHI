# --- START OF FILE utils/high_level_gui/gui_manager.py ---

import numpy as np
import os
import time
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QScrollArea, QLabel # type: ignore
import yaml # type: ignore
from typing import Dict, Any, List # Keep List
import sys
import traceback
import gc

# --- Assumed relative imports based on structure ---
from ..module_3d._3D_strategy import RamifiedStrategy
from ..module_2d._2D_strategy import Ramified2DStrategy
from ..high_level_gui.processing_strategies import ProcessingStrategy # Import base class for type hint

try:
    # Import helpers from the SAME directory (.)
    from .helper_funcs import create_parameter_widget
    from .processing_strategies import check_processing_state
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
            processing_mode (str): The selected mode (''ramified').
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
            strategy_class = {
                'ramified': RamifiedStrategy,
                'ramified_2d': Ramified2DStrategy # <-- ADD THIS MAPPING
            }.get(self.processing_mode)
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
        """
        Calculates voxel/pixel spacing and Z-scale factor from self.config.
        Assumes config dimensions are TOTAL physical size of each axis.
        Stores spacing as [Z, Y, X] for Napari consistency.
        """
        # Determine config keys based on mode
        is_2d_mode = self.processing_mode.endswith("_2d")
        dim_section_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'
        dimensions = self.config.get(dim_section_key, {})

        try:
            # Get total dimensions from config
            total_x_um = float(dimensions.get('x', 1.0))
            total_y_um = float(dimensions.get('y', 1.0))
            total_z_um = 1.0 # Default for 2D
            if not is_2d_mode:
                total_z_um = float(dimensions.get('z', 1.0))

        except (ValueError, TypeError):
            print(f"Warning: Invalid total dimensions in config section '{dim_section_key}'. Using defaults (1.0).")
            total_x_um, total_y_um, total_z_um = 1.0, 1.0, 1.0

        shape = self.image_stack.shape
        num_dims = len(shape)

        # Initialize pixel sizes
        x_pixel_size = total_x_um
        y_pixel_size = total_y_um
        z_pixel_size = total_z_um

        # Calculate pixel size based on image shape (Y, X for 2D; Z, Y, X for 3D)
        if num_dims == 2: # 2D: shape is (Y, X)
            if shape[0] > 0: y_pixel_size = total_y_um / shape[0]
            if shape[1] > 0: x_pixel_size = total_x_um / shape[1]
            z_pixel_size = 1.0 # Define Z pixel size as 1.0 for 2D case consistency
            self.spacing = [z_pixel_size, y_pixel_size, x_pixel_size] # Store as Z, Y, X
            self.z_scale_factor = 1.0 # No Z scaling needed for 2D display relative to XY

        elif num_dims == 3: # 3D: shape is (Z, Y, X)
            if shape[0] > 0: z_pixel_size = total_z_um / shape[0]
            if shape[1] > 0: y_pixel_size = total_y_um / shape[1]
            if shape[2] > 0: x_pixel_size = total_x_um / shape[2]
            self.spacing = [z_pixel_size, y_pixel_size, x_pixel_size] # Store as Z, Y, X
            # Calculate Z scale factor relative to X for visualization
            self.z_scale_factor = z_pixel_size / x_pixel_size if x_pixel_size > 1e-9 else 1.0

        else: # Handle unexpected dimensions
            print(f"Warning: Image stack has unexpected dimensions: {num_dims}. Using default spacing [1,1,1].")
            self.spacing = [1.0, 1.0, 1.0]
            self.z_scale_factor = 1.0

        # Use the CORRECT labels in the print statement
        print(f"Calculated {num_dims}D Pixel/Voxel Size (Z, Y, X): {self.spacing}")
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
            # --- MODIFICATION START ---
            image_ndim = self.image_stack.ndim
            if image_ndim == 2:
                # For 2D, Napari expects scale (y, x).
                # Use the y and x components from self.spacing if it's [Z, Y, X]
                # Or use (1,1) if spacing isn't critical for initial display
                # Assuming self.spacing was set to [1.0, y_spacing, x_spacing] for 2D
                # If self.spacing is [y_spacing, x_spacing] for 2D, adjust indices.
                # Let's assume [Z,Y,X] format for consistency from calculation step:
                if hasattr(self, 'spacing') and len(self.spacing) >= 3:
                     # Use Y and X spacing components
                     scale = (self.spacing[1], self.spacing[2])
                     print(f"Applying 2D scale: {scale}")
                else:
                     scale = (1, 1) # Default scale if spacing is unavailable/wrong format
                     print("Applying default 2D scale (1, 1)")
            elif image_ndim == 3:
                # Existing 3D logic
                scale = (self.z_scale_factor, 1, 1) # Apply Z scaling for display
                print(f"Applying 3D scale: {scale}")
            else:
                print(f"Warning: Unexpected image dimensionality ({image_ndim}). Using default scale.")
                scale = tuple([1.0] * image_ndim)
            # --- MODIFICATION END ---

            self.viewer.add_image(
                self.image_stack,
                name=layer_name,
                scale=scale # Use the calculated scale
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
        step_index = self.current_step["value"]

        if step_index >= self.num_steps:
            print("All processing steps already completed."); msg = QMessageBox(); msg.setIcon(QMessageBox.Information); msg.setText("Processing complete."); msg.exec_(); return

        # Get the LOGICAL step name (e.g., "execute_raw_segmentation")
        logical_step_method_name = self.processing_steps[step_index]
        step_display_name = self.step_display_names.get(logical_step_method_name, f"Step {step_index + 1}")
        success = False

        # --- MODIFICATION START: Determine the ACTUAL method name to call ---
        actual_method_name_to_call = logical_step_method_name
        # If the strategy is 2D (check mode name), append _2d to the method name
        if self.strategy.mode_name.endswith("_2d"): # Check if mode indicates 2D
            actual_method_name_to_call = f"{logical_step_method_name}_2d"
            print(f"  Mapping logical step '{logical_step_method_name}' to actual 2D method '{actual_method_name_to_call}'")
        # --- MODIFICATION END ---

        try:
            # --- MOVED: Get current parameters (no change here) ---
            current_values = self.get_current_values()

            # --- MOVED: Save config before cleanup (no change here) ---
            print(f"Saving parameters before executing {step_display_name}...")
            self.strategy.save_config(self.config)

            # --- Clean up artifacts (no change here) ---
            print(f"Cleaning up potential old artifacts for {step_display_name} (Step {step_index+1}) onwards...")
            for i in range(step_index + 1, self.num_steps + 1):
                 self.cleanup_step(i)

            # --- Execute the step using the strategy's execute_step method,
            #     BUT execute_step itself needs to know the ACTUAL method name.
            #     Let's modify execute_step in the base class slightly OR
            #     call the actual method directly here. Calling directly is simpler:

            print(f"\n--- Attempting Step {step_index + 1}/{self.num_steps}: '{step_display_name}' (Method: '{actual_method_name_to_call}') ---")

            try:
                # Get the actual method object from the strategy instance
                step_method = getattr(self.strategy, actual_method_name_to_call)
            except AttributeError:
                print(f"FATAL ERROR: Method '{actual_method_name_to_call}' not found in strategy class '{self.strategy.__class__.__name__}'.")
                return # Cannot proceed

            # Call the specific step method directly
            # Assumes step methods accept these arguments.
            success = step_method(viewer=self.viewer, image_stack=self.image_stack, params=current_values)

            # Validate return type (already present in base strategy execute_step, keep here too)
            if not isinstance(success, bool):
                 print(f"Warning: Step method '{actual_method_name_to_call}' did not return bool. Assuming failure.")
                 success = False

            print(f"--- Step {step_index + 1} ('{step_display_name}') finished (Success: {success}) ---")

            # --- Handle results (no change here) ---
            if success:
                print(f"Step successful. Saving updated configuration and state after {step_display_name}...")
                self.strategy.save_config(self.config)
                self.current_step["value"] += 1
                if self.current_step["value"] < self.num_steps:
                    next_step_logical_name = self.processing_steps[self.current_step["value"]]
                    self.create_step_widgets(next_step_logical_name) # Use logical name for widgets
                else:
                    print("\n*** Processing complete! ***")
                    self.clear_current_widgets()
                    msg = QMessageBox(); msg.setIcon(QMessageBox.Information); msg.setWindowTitle("Complete"); msg.setText(f"All '{self.processing_mode}' steps finished."); msg.exec_()
            else:
                 print(f"{step_display_name} failed or returned False.")
                 msg = QMessageBox(); msg.setIcon(QMessageBox.Warning); msg.setWindowTitle("Step Not Completed"); msg.setText(f"Step '{step_display_name}' did not complete successfully."); msg.setInformativeText("Check console output. Adjust parameters and click 'Run Current Step' again."); msg.exec_()

        except Exception as e:
            print(f"FATAL ERROR during processing flow for {step_display_name}: {str(e)}")
            traceback.print_exc()
            msg = QMessageBox(); msg.setIcon(QMessageBox.Critical); msg.setWindowTitle("Processing Error"); msg.setText(f"An unexpected error occurred during '{step_display_name}'."); msg.setInformativeText(f"Error: {str(e)}\n\nCheck console output."); msg.exec_()

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
        """
        Creates and displays widgets for the parameters of a given step inside a single, scrollable dock widget.
        """
        self.clear_current_widgets()
        self.parameter_values = {} # Reset dict for current step's params

        config_key = self.strategy.get_config_key(step_method_name)
        step_display_name = self.step_display_names.get(step_method_name, step_method_name)

        if config_key not in self.config:
            print(f"Error: Config key '{config_key}' not found for step '{step_display_name}'. Cannot create widgets.")
            return

        step_config = self.config.get(config_key, {})
        parameters = step_config.get("parameters")

        # --- Create the main scrollable container ---
        # This will be the content of our single dock widget
        
        # 1. The main content widget that will hold the layout
        scroll_content_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_content_widget)
        scroll_layout.setContentsMargins(10, 10, 10, 10) # Padding
        scroll_layout.setSpacing(8)                      # Spacing between widgets

        # 2. A title for the parameter group
        title_label = QLabel(f"Parameters for: {step_display_name}")
        font = title_label.font()
        font.setBold(True)
        font.setPointSize(12)
        title_label.setFont(font)
        scroll_layout.addWidget(title_label)
        
        # Check if parameters exist and are a dictionary
        if not isinstance(parameters, dict) or not parameters:
            print(f"Warning: No valid 'parameters' dict found for '{config_key}'.")
            # Add a label indicating no parameters
            no_params_label = QLabel("No parameters for this step.")
            scroll_layout.addWidget(no_params_label)
        else:
            print(f"Creating widgets for: '{step_display_name}' (using config key: '{config_key}')")
            for param_name, param_config in parameters.items():
                if not isinstance(param_config, dict):
                    print(f"Warning: Parameter config for '{param_name}' is not a dict. Skipping.")
                    continue
                try:
                    current_value = param_config.get('value')
                    callback = lambda value, key=config_key, pn=param_name: self.parameter_changed(key, pn, value)
                    
                    # Create the individual magicgui widget
                    widget = create_parameter_widget(param_name, param_config, callback)
                    
                    # Add its native Qt widget to our layout instead of a new dock
                    scroll_layout.addWidget(widget.native)
                    
                    self.parameter_values[param_name] = current_value
                except Exception as e:
                    print(f"ERROR creating widget for parameter '{param_name}': {str(e)}")
                    traceback.print_exc()
        
        # Add a spacer at the bottom to push everything up
        scroll_layout.addStretch()

        # 3. The scroll area itself
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) # This is crucial!
        scroll_area.setWidget(scroll_content_widget)

        # 4. Add the scroll_area as the *single* dock widget for this step
        dock_widget = self.viewer.window.add_dock_widget(
            scroll_area,
            area="right",
            name=f"Step: {step_display_name}"
        )

        # 5. Track the main dock widget for later cleanup
        self.current_widgets[dock_widget] = scroll_area


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