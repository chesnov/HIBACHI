# --- START OF FILE utils/high_level_gui/gui_manager.py ---
import numpy as np
import os
import time
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QScrollArea, QLabel # type: ignore
import yaml # type: ignore
from typing import Dict, Any, List
import sys
import traceback
import gc

# --- Relative Imports ---
# We need the strategies to instantiate them based on mode
try:
    from ..module_3d._3D_strategy import RamifiedStrategy
    from ..module_2d._2D_strategy import Ramified2DStrategy
    from ..high_level_gui.processing_strategies import ProcessingStrategy
    
    # Import helpers from the SAME directory
    from .helper_funcs import create_parameter_widget
except ImportError as e:
    print(f"Error importing dependencies in gui_manager.py: {e}")
    raise

class DynamicGUIManager:
    """
    Manages the interactive GUI sidebar in Napari.
    Connects the abstract ProcessingStrategy to the user's clicks and inputs.
    """

    def __init__(self, viewer, config, image_stack, file_loc, processing_mode):
        """ Initializes the GUI Manager. """
        self.viewer = viewer
        self.initial_config = config.copy()
        self.config = config.copy()
        self.image_stack = image_stack
        self.file_loc = file_loc
        self.processing_mode = processing_mode
        
        # UI State
        self.current_widgets: Dict[Any, Any] = {}
        self.current_step = {"value": 0}
        self.parameter_values: Dict[str, Any] = {}

        # Paths
        self.inputdir = os.path.dirname(self.file_loc)
        self.basename = os.path.basename(self.file_loc).split('.')[0]
        self.processed_dir = os.path.join(self.inputdir, f"{self.basename}_processed_{self.processing_mode}")

        # Calculate geometry
        self._calculate_spacing()

        # Initialize Strategy
        try:
            strategy_class = {
                'ramified': RamifiedStrategy, 
                'ramified_2d': Ramified2DStrategy
            }.get(self.processing_mode)
            
            if not strategy_class: 
                raise ValueError(f"Unsupported processing mode: {self.processing_mode}")
            
            self.strategy = strategy_class(
                self.config, 
                self.processed_dir, 
                self.image_stack.shape, 
                self.spacing, 
                self.z_scale_factor
            )
            
            self.processing_steps: List[str] = self.strategy.get_step_names()
            self.num_steps: int = self.strategy.num_steps
            
            # Generate display names (e.g., "execute_raw_segmentation" -> "Raw Segmentation")
            self.step_display_names = { 
                name: name.replace('execute_', '').replace('_', ' ').title() 
                for name in self.processing_steps 
            }
            
            print(f"Initialized strategy '{self.processing_mode}' with {self.num_steps} steps.")
            
        except Exception as e:
             print(f"FATAL ERROR initializing processing strategy: {e}")
             traceback.print_exc()
             raise

        self._initialize_layers()
        self.restore_from_checkpoint()


    def _calculate_spacing(self):
        """ Calculates voxel/pixel spacing and Z-scale factor from self.config. """
        is_2d_mode = self.processing_mode.endswith("_2d")
        dim_section_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'
        dimensions = self.config.get(dim_section_key, {})
        
        try:
            total_x_um = float(dimensions.get('x', 1.0))
            total_y_um = float(dimensions.get('y', 1.0))
            total_z_um = 1.0 if is_2d_mode else float(dimensions.get('z', 1.0))
        except (ValueError, TypeError):
            total_x_um, total_y_um, total_z_um = 1.0, 1.0, 1.0

        shape = self.image_stack.shape
        num_dims = len(shape)

        if num_dims == 2:
            y_pixel_size = total_y_um / shape[0] if shape[0] > 0 else 1.0
            x_pixel_size = total_x_um / shape[1] if shape[1] > 0 else 1.0
            self.spacing = [1.0, y_pixel_size, x_pixel_size]
            self.z_scale_factor = 1.0
        elif num_dims == 3:
            z_pixel_size = total_z_um / shape[0] if shape[0] > 0 else 1.0
            y_pixel_size = total_y_um / shape[1] if shape[1] > 0 else 1.0
            x_pixel_size = total_x_um / shape[2] if shape[2] > 0 else 1.0
            self.spacing = [z_pixel_size, y_pixel_size, x_pixel_size]
            self.z_scale_factor = z_pixel_size / x_pixel_size if x_pixel_size > 1e-9 else 1.0
        else:
            self.spacing = [1.0, 1.0, 1.0]; self.z_scale_factor = 1.0

    def _initialize_layers(self):
        """Adds the initial base image layer to the viewer."""
        layer_name = f"Original stack ({self.processing_mode} mode)"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)
        
        image_ndim = self.image_stack.ndim
        if image_ndim == 2: scale = (self.spacing[1], self.spacing[2])
        elif image_ndim == 3: scale = (self.z_scale_factor, 1, 1)
        else: scale = tuple([1.0] * image_ndim)
            
        self.viewer.add_image(self.image_stack, name=layer_name, scale=scale)

    def restore_from_checkpoint(self):
        """ Checks for existing files and determines the next step. """
        # Use the new polymorphic method
        checkpoint_step = self.strategy.get_last_completed_step()

        if checkpoint_step > 0:
            # Load state if possible
            checkpoint_files = self.strategy.get_checkpoint_files()
            config_file = checkpoint_files.get("config")
            
            if config_file and os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as file: saved_config = yaml.safe_load(file)
                    if saved_config:
                        self.config.update(saved_config)
                        self.strategy.config = self.config
                        
                        # Restore intermediate state (e.g. threshold)
                        loaded_state = self.config.get('saved_state', {})
                        if 'segmentation_threshold' in loaded_state:
                            self.strategy.intermediate_state['segmentation_threshold'] = float(loaded_state['segmentation_threshold'])
                except Exception as e: print(f"Error loading saved config: {e}")

            # Prompt User
            msg = QMessageBox()
            if checkpoint_step == self.num_steps:
                msg.setText("All steps complete.")
                msg.setInformativeText("View results or restart?")
                view_btn = msg.addButton("View", QMessageBox.YesRole)
                restart_btn = msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()
                
                if msg.clickedButton() == view_btn:
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                else:
                    self._confirm_restart()
            else:
                msg.setText("Resume?")
                msg.setInformativeText(f"Resume from Step {checkpoint_step + 1} or restart?")
                resume_btn = msg.addButton("Resume", QMessageBox.YesRole)
                restart_btn = msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()
                
                if msg.clickedButton() == resume_btn:
                    # Load reference to image stack into strategy for resumption
                    self.strategy.intermediate_state['original_volume_ref'] = self.image_stack
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                    
                    # If we are at the end of list? No, logic allows resumption from next step.
                    if checkpoint_step < self.num_steps:
                        self.create_step_widgets(self.processing_steps[checkpoint_step])
                else:
                    self._confirm_restart()
        else:
            # Start fresh
            self.create_step_widgets(self.processing_steps[0])

    def _confirm_restart(self):
        """Asks user to confirm deleting files and restarts processing."""
        parent_widget = self.viewer.window._qt_window if self.viewer and self.viewer.window else None
        reply = QMessageBox.question(parent_widget, "Confirm Restart", 
                                     "This will delete existing processed files. Are you sure?", 
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.delete_all_checkpoint_files()
            self.current_step["value"] = 0
            self.strategy.intermediate_state = {}
            self.config = self.initial_config.copy()
            self.strategy.config = self.config
            self._initialize_layers()
            self.create_step_widgets(self.processing_steps[0])
        else:
            self.restore_from_checkpoint()

    def delete_all_checkpoint_files(self):
        """Deletes all checkpoint files defined by the strategy."""
        files_to_delete = self.strategy.get_checkpoint_files()
        for key, file_path in files_to_delete.items():
            self.strategy._remove_file_safely(file_path)

    def load_checkpoint_data(self, checkpoint_step: int):
        self.strategy.load_checkpoint_data(self.viewer, checkpoint_step)

    def cleanup_step(self, step_number: int):
        self.strategy.cleanup_step_artifacts(self.viewer, step_number)

    def execute_processing_step(self):
        """Executes the next processing step."""
        step_index = self.current_step["value"]
        if step_index >= self.num_steps: return

        # Method name in the Strategy class
        logical_step_method_name = self.processing_steps[step_index]
        step_display_name = self.step_display_names.get(logical_step_method_name, f"Step {step_index + 1}")
        
        parent_widget = self.viewer.window._qt_window if self.viewer and self.viewer.window else None
        
        try:
            # Prepare parameters
            current_values = self.get_current_values()
            
            # Save config before run
            self.strategy.save_config(self.config)
            
            # Cleanup future steps if we are re-running this one
            for i in range(step_index + 1, self.num_steps + 1):
                 self.cleanup_step(i)
            
            print(f"\n--- Attempting Step {step_index + 1}/{self.num_steps}: '{step_display_name}' ---")
            
            # Ensure strategy has reference to image stack
            if 'original_volume_ref' not in self.strategy.intermediate_state:
                self.strategy.intermediate_state['original_volume_ref'] = self.image_stack

            # Execute via Strategy Base Class
            # This handles method lookup and internal error logging
            success = self.strategy.execute_step(
                step_index=step_index, 
                viewer=self.viewer, 
                image_stack_or_none=self.image_stack, 
                params=current_values
            )

            if success:
                self.strategy.save_config(self.config)
                self.current_step["value"] += 1
                
                if self.current_step["value"] < self.num_steps:
                    # Move to next step widgets
                    next_step_name = self.processing_steps[self.current_step["value"]]
                    self.create_step_widgets(next_step_name)
                else:
                    # Finished
                    self.clear_current_widgets()
                    QMessageBox.information(parent_widget, "Complete", "All steps finished.")
            else:
                 QMessageBox.warning(parent_widget, "Step Not Completed", 
                                     f"Step '{step_display_name}' did not complete successfully.\nCheck console for details.")
        except Exception as e:
            QMessageBox.critical(parent_widget, "Processing Error", 
                                 f"An error occurred during '{step_display_name}':\n{e}")
            traceback.print_exc()

    def clear_current_widgets(self):
        """Removes all currently displayed parameter widgets."""
        for dock_widget in list(self.current_widgets.keys()):
            self.viewer.window.remove_dock_widget(dock_widget)
        self.current_widgets.clear()

    def create_step_widgets(self, step_method_name: str):
        """Creates and displays widgets for the parameters of a given step."""
        self.clear_current_widgets()
        self.parameter_values = {}

        config_key = self.strategy.get_config_key(step_method_name)
        step_display_name = self.step_display_names.get(step_method_name, step_method_name)
        
        # Safe get parameters
        step_config = self.config.get(config_key, {})
        if not isinstance(step_config, dict):
            print(f"Warning: Config for {config_key} is not a dict.")
            parameters = {}
        else:
            parameters = step_config.get("parameters", {})

        scroll_content_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_content_widget)
        title_label = QLabel(f"Parameters for: {step_display_name}")
        
        if self.viewer and self.viewer.window and self.viewer.window._qt_window:
            font = self.viewer.window._qt_window.font()
            font.setBold(True)
            title_label.setFont(font)
        scroll_layout.addWidget(title_label)
        
        if isinstance(parameters, dict):
            for param_name, param_config in parameters.items():
                try:
                    callback = lambda value, key=config_key, pn=param_name: self.parameter_changed(key, pn, value)
                    widget = create_parameter_widget(param_name, param_config, callback)
                    if widget: 
                        scroll_layout.addWidget(widget.native)
                        # Extract initial value
                        val = param_config.get('value')
                        self.parameter_values[param_name] = val
                except Exception as e: 
                    print(f"ERROR creating widget for '{param_name}': {e}")
        
        scroll_layout.addStretch()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content_widget)
        
        dock_widget = self.viewer.window.add_dock_widget(scroll_area, area="right", name=f"Step: {step_display_name}")
        self.current_widgets[dock_widget] = scroll_area

    def parameter_changed(self, config_key: str, param_name: str, value: Any):
        """Callback when a parameter widget changes value."""
        try:
            # Update config in memory
            self.config[config_key]["parameters"][param_name]["value"] = value
            # Update local current values
            self.parameter_values[param_name] = value
        except Exception as e:
             print(f"Error handling parameter change for {config_key}/{param_name}: {e}")

    def get_current_values(self) -> Dict[str, Any]:
        """Returns a copy of the current parameter values for the active step."""
        return self.parameter_values.copy()
# --- END OF FILE utils/high_level_gui/gui_manager.py ---