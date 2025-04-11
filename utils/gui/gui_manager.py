import numpy as np
import os
import time
from PyQt5.QtWidgets import QMessageBox
import yaml
from typing import Dict, Any
import sys # Import sys

os.environ["QT_SCALE_FACTOR"] = "1.5" # Keep if needed

seed = 42
np.random.seed(seed)

# --- UPDATED IMPORTS ---
# Import strategies using relative paths within the 'gui' package
from ._3D_nuclear_strategy import NuclearStrategy
from ._3D_ramified_strategy import RamifiedStrategy

# Add parent directory (project root) to path to find helper_funcs
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    # Import helpers from the parent directory
    from helper_funcs import create_parameter_widget, check_processing_state
except ImportError as e:
    print(f"Error importing helper_funcs in gui_manager.py: {e}")
    print(f"Ensure helper_funcs.py is in the directory: {parent_dir}")
    raise
# --- END OF UPDATED IMPORTS ---

class DynamicGUIManager:
    def __init__(self, viewer, config, image_stack, file_loc, processing_mode):
        self.viewer = viewer
        self.config = config # Store the original full config
        self.image_stack = image_stack
        self.file_loc = file_loc
        self.processing_mode = processing_mode
        self.current_widgets = {}
        self.current_step = {"value": 0}
        # Step names remain generic
        self.processing_steps = ["initial_segmentation", "refine_rois", "calculate_features"]
        self.parameter_values = {} # Stores current values for the active step's widgets

        # Set up processing directory
        self.inputdir = os.path.dirname(self.file_loc)
        self.basename = os.path.basename(self.file_loc).split('.')[0]
        self.processed_dir = os.path.join(self.inputdir, f"{self.basename}_processed_{self.processing_mode}")
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        # Calculate spacing once
        self._calculate_spacing()

        # Instantiate the correct strategy
        try:
            if self.processing_mode == 'nuclei':
                self.strategy = NuclearStrategy(self.config, self.processed_dir, self.image_stack.shape, self.spacing, self.z_scale_factor)
            elif self.processing_mode == 'ramified':
                self.strategy = RamifiedStrategy(self.config, self.processed_dir, self.image_stack.shape, self.spacing, self.z_scale_factor)
            else:
                raise ValueError(f"Unsupported processing mode: {self.processing_mode}")
        except Exception as e:
             print(f"Error initializing processing strategy: {e}")
             # Handle error appropriately, maybe raise or disable processing
             raise

        # Initialize viewer layers (common part)
        self._initialize_layers()

        # Check for existing processed files and restore state (uses strategy for file names)
        self.restore_from_checkpoint()

    def _calculate_spacing(self):
        """Calculate voxel spacing and scale factor."""
        voxel_x = self.config.get('voxel_dimensions', {}).get('x', 1)
        voxel_y = self.config.get('voxel_dimensions', {}).get('y', 1)
        voxel_z = self.config.get('voxel_dimensions', {}).get('z', 1)
        # Ensure dimensions are valid before calculating spacing
        shape = self.image_stack.shape
        self.x_spacing = voxel_x / shape[2] if shape[2] > 0 else 1.0 # Use dim 2 for X
        self.y_spacing = voxel_y / shape[1] if shape[1] > 0 else 1.0 # Use dim 1 for Y
        self.z_spacing = voxel_z / shape[0] if shape[0] > 0 else 1.0 # Use dim 0 for Z
        self.spacing = [self.z_spacing, self.x_spacing, self.y_spacing] # Consistent order
        self.z_scale_factor = self.z_spacing / self.x_spacing if self.x_spacing > 0 else 1.0
        print(f"Calculated Spacing (Z, X, Y): {self.spacing}")
        print(f"Calculated Z Scale Factor: {self.z_scale_factor}")


    def _initialize_layers(self):
        """Initialize the basic layers in the viewer."""
        # Add original image layer - this is common
        self.viewer.add_image(
            self.image_stack,
            name=f"Original stack ({self.processing_mode} mode)",
            scale=(self.z_scale_factor, 1, 1)
        )

    def restore_from_checkpoint(self):
        """Check for existing processed files and restore the processing state."""
        # check_processing_state likely needs strategy to know which files define completion
        # Assuming check_processing_state is adapted or uses strategy file list
        checkpoint_step = check_processing_state(self.processed_dir, self.processing_mode, self.strategy.get_checkpoint_files())

        if checkpoint_step > 0:
            step_names = {1: "initial segmentation", 2: "ROI refinement", 3: "feature calculation"}
            if checkpoint_step == 3:
                # All completed
                msg = QMessageBox()
                # ... (QMessageBox setup as before) ...
                view_button = msg.addButton("View Results", QMessageBox.YesRole)
                restart_button = msg.addButton("Restart Processing", QMessageBox.NoRole)
                msg.exec_()

                if msg.clickedButton() == view_button:
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                    print("All processing steps completed. Displaying results.")
                    # Maybe disable further processing buttons here
                else:
                    self._confirm_restart()
            else:
                # Partial completion
                msg = QMessageBox()
                # ... (QMessageBox setup as before, mentioning step_names[checkpoint_step]) ...
                resume_button = msg.addButton("Resume", QMessageBox.YesRole)
                restart_button = msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()

                if msg.clickedButton() == resume_button:
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                    if checkpoint_step < 3:
                        self.create_step_widgets(self.processing_steps[checkpoint_step])
                else:
                    self._confirm_restart()
        else:
            # Start from beginning
            self.create_step_widgets(self.processing_steps[0]) # Start with initial_segmentation

    def _confirm_restart(self):
        """Confirm and handle restarting from scratch."""
        confirm_box = QMessageBox()
        # ... (QMessageBox setup as before) ...
        yes_button = confirm_box.addButton("Yes, delete files", QMessageBox.YesRole)
        no_button = confirm_box.addButton("No, keep files", QMessageBox.NoRole)
        confirm_box.exec_()

        if confirm_box.clickedButton() == yes_button:
            self.delete_all_checkpoint_files() # Use the new method
            self.current_step["value"] = 0
            # Reset any loaded intermediate state in the strategy
            self.strategy.intermediate_state = {}
            # Remove potentially loaded layers from previous state
            self.cleanup_step(1)
            self.cleanup_step(2)
            self.cleanup_step(3)
            self.create_step_widgets(self.processing_steps[0])
        else:
            # User canceled deletion, load checkpoint data anyway
            checkpoint_step = check_processing_state(self.processed_dir, self.processing_mode, self.strategy.get_checkpoint_files())
            self.load_checkpoint_data(checkpoint_step)
            self.current_step["value"] = checkpoint_step
            if checkpoint_step < 3:
                self.create_step_widgets(self.processing_steps[checkpoint_step])
            else:
                 print("All processing steps completed. Displaying results.")

    def delete_all_checkpoint_files(self):
        """Delete all checkpoint files for the current processing mode using the strategy."""
        print(f"Attempting to delete all checkpoint files for mode: {self.processing_mode}")
        files_to_delete = self.strategy.get_checkpoint_files()
        for key, file_path in files_to_delete.items():
             self.strategy._remove_file_safely(file_path) # Use strategy's helper


    def load_checkpoint_data(self, checkpoint_step):
        """Load data from checkpoint files using the strategy."""
        try:
            # Load mode-specific config first if exists
            config_file = self.strategy.get_checkpoint_files()["config"]
            if os.path.exists(config_file):
                with open(config_file, 'r') as file:
                    saved_config = yaml.safe_load(file)
                    # Update only the relevant sections of the config
                    for key in saved_config:
                        if key in self.config: # Check if key exists in base config
                            self.config[key] = saved_config[key]
                        # Load potential intermediate state back into strategy
                        if key in self.strategy.intermediate_state:
                             self.strategy.intermediate_state[key] = saved_config[key]
                    print(f"Loaded saved configuration parameters from {config_file}")

            # Delegate the rest of the loading to the strategy
            self.strategy.load_checkpoint_data(self.viewer, checkpoint_step)

        except Exception as e:
            print(f"Error loading checkpoint data: {str(e)}")
            # raise # Re-raise for debugging if necessary

    def cleanup_step(self, step_number):
        """Clean up the results and layers from a specific step using the strategy."""
        print(f"Cleaning up artifacts for step {step_number}")
        self.strategy.cleanup_step_artifacts(self.viewer, step_number)


    def execute_processing_step(self):
        """Execute the next step in the processing pipeline using the strategy."""
        start_time = time.time()
        step_index = self.current_step["value"]
        step_name = self.processing_steps[step_index]
        success = False

        try:
            # Get current parameter values for this step
            current_values = self.get_current_values()

            # Clear previous step's artifacts before running the new step
            self.cleanup_step(step_index + 1) # cleanup step N+1 artifacts

            print(f"Executing step {step_index + 1}: {step_name}")
            if step_index == 0:
                success = self.strategy.execute_initial_segmentation(self.viewer, self.image_stack, current_values)
            elif step_index == 1:
                success = self.strategy.execute_refine_rois(self.viewer, self.image_stack, current_values)
            elif step_index == 2:
                success = self.strategy.execute_calculate_features(self.viewer, self.image_stack, current_values)
            else:
                print("All processing steps completed.")
                return # Don't increment step

            if success:
                self.current_step["value"] += 1
                if self.current_step["value"] < len(self.processing_steps):
                    next_step_name = self.processing_steps[self.current_step["value"]]
                    self.create_step_widgets(next_step_name)
                else:
                    print("Processing complete!")
                    self.strategy.save_config(self.config) # Save final config via strategy
                    self.clear_current_widgets() # Remove last step's widgets
                    # Maybe add a final "Done" widget or message
            else:
                 print(f"Step {step_index + 1} ({step_name}) failed or was aborted.")
                 # Optionally revert current_step or allow retry?

        except Exception as e:
            print(f"Error during processing step {step_index + 1} ({step_name}): {str(e)}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging
            # Consider showing an error message to the user
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Processing Error")
            msg.setText(f"An error occurred during {step_name}.")
            msg.setInformativeText(f"Error details: {str(e)}\n\nCheck the console output for more information.")
            msg.exec_()
            # Optionally revert step? self.current_step["value"] = step_index
            # raise # Re-raise for critical debugging

        finally:
            end_time = time.time()
            if step_index < len(self.processing_steps): # Only print timing if a step was attempted
                print(f"Step {step_index + 1} ({step_name}) processing took {end_time - start_time:.2f} seconds (Success: {success})")


    # --- Widget Management (Largely unchanged, but uses strategy for config key) ---

    def clear_current_widgets(self):
        """Remove all current widgets"""
        dock_widgets_to_remove = list(self.current_widgets.keys())
        for dock_widget in dock_widgets_to_remove:
            try:
                self.viewer.window.remove_dock_widget(dock_widget)
            except Exception as e:
                 # This can happen if the widget was already closed manually
                 print(f"Warning: Failed to remove dock widget (may have been closed already): {str(e)}")
            # Always remove from our tracking dictionary
            if dock_widget in self.current_widgets:
                 del self.current_widgets[dock_widget]
        # Ensure the dictionary is empty
        self.current_widgets.clear()


    def create_step_widgets(self, step_name: str):
        """Create all widgets for a processing step based on the processing mode via strategy."""
        self.clear_current_widgets()
        self.parameter_values = {} # Reset collected values for the new step

        # Use strategy to get the correct key (e.g., 'initial_segmentation_nuclei')
        config_key = self.strategy.get_config_key(step_name)

        if config_key not in self.config:
            print(f"Warning: Configuration key '{config_key}' not found in config for step '{step_name}'. Cannot create widgets.")
            # Maybe add a default widget or message?
            return

        step_config = self.config.get(config_key, {})
        if "parameters" not in step_config:
            print(f"Warning: No 'parameters' found for config key '{config_key}' for step '{step_name}'.")
            # Maybe add a default widget or message?
            return

        print(f"Creating widgets for step: '{step_name}' using config key: '{config_key}'")
        # Create parameter widgets
        for param_name, param_config in step_config["parameters"].items():
            try:
                # Ensure 'value' exists in param_config, provide default if necessary
                if 'value' not in param_config:
                    print(f"Warning: No 'value' key found for parameter '{param_name}' in '{config_key}'. Using default based on type or None.")
                    # Try to infer a default based on type, or set to None/0/""
                    param_type = param_config.get('type', 'float') # Assume float if type missing
                    if param_type in ['int', 'slider']:
                        param_config['value'] = 0
                    elif param_type == 'float':
                        param_config['value'] = 0.0
                    elif param_type == 'bool':
                        param_config['value'] = False
                    elif param_type == 'str':
                         param_config['value'] = ""
                    elif param_type == 'list': # Handle list type more carefully
                         param_config['value'] = [] # Or a sensible default list
                    else:
                         param_config['value'] = None


                # Create callback for this specific parameter
                # Pass config_key to callback so it knows which part of self.config to update
                callback = lambda value, key=config_key, pn=param_name: self.parameter_changed(key, pn, value)

                # Create widget using the helper function
                widget = create_parameter_widget(param_name, param_config, callback)
                dock_widget = self.viewer.window.add_dock_widget(widget, area="right", name=f"{step_name}: {param_name}") # Give dock unique name
                self.current_widgets[dock_widget] = widget # Store dock_widget reference

                # Store initial value for get_current_values()
                self.parameter_values[param_name] = param_config["value"]

            except Exception as e:
                print(f"Error creating widget for parameter '{param_name}' in step '{step_name}': {str(e)}")
                import traceback
                traceback.print_exc()


    def parameter_changed(self, config_key: str, param_name: str, value: Any):
        """Callback for when a parameter value changes. Updates both config and current values."""
        # print(f"Parameter changed: Step Key='{config_key}', Param='{param_name}', New Value='{value}'")
        if config_key in self.config and "parameters" in self.config[config_key]:
            if param_name in self.config[config_key]["parameters"]:
                self.config[config_key]["parameters"][param_name]["value"] = value
            else:
                 print(f"Warning: Parameter '{param_name}' not found in config under key '{config_key}' during update.")
        else:
            print(f"Warning: Config key '{config_key}' or its 'parameters' not found during update.")

        # Always update the current step's values dictionary
        self.parameter_values[param_name] = value


    def get_current_values(self) -> Dict[str, Any]:
        """Get current values for all parameters displayed in the current step's widgets."""
        # This now just returns the dictionary populated by create_step_widgets and parameter_changed
        # print(f"Getting current values for step {self.current_step['value']}: {self.parameter_values}")
        return self.parameter_values.copy()