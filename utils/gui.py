import tifffile as tiff
import numpy as np
import os
import plotly.graph_objs as go
import napari
from skimage import morphology
from magicgui import magicgui
from skimage.measure import label
from PyQt5.QtWidgets import QApplication, QFileDialog, QInputDialog, QMessageBox
import sys
os.environ["QT_SCALE_FACTOR"] = "1.5"

from utils.ramified_segmenter import *
from utils.nuclear_segmenter import *
from utils.calculate_features import *

import yaml
from typing import Dict, Any


def create_parameter_widget(param_name: str, param_config: Dict[str, Any], callback):
    """Create a widget for a single parameter"""
    
    # Define the function with the appropriate type annotation
    if param_config["type"] == "float":
        def parameter_widget(value: float = param_config["value"]):
            callback(value)  # Modified to only pass the value
            return value
    elif param_config["type"] == "int":
        def parameter_widget(value: int = param_config["value"]):
            callback(value)  # Modified to only pass the value
            return value
    else:
        def parameter_widget(value: float = param_config["value"]):
            callback(value)  # Modified to only pass the value
            return value
    
    # Create the widget with magicgui
    widget = magicgui(
        parameter_widget,
        auto_call=True,
        value={
            "label": param_config["label"],
            "min": param_config["min"],
            "max": param_config["max"],
            "step": param_config["step"]
        }
    )
    
    # Store the original parameter name as an attribute
    widget.param_name = param_name
    
    return widget

class DynamicGUIManager:
    def __init__(self, viewer, config, image_stack, file_loc, processing_mode):
        self.viewer = viewer
        self.config = config
        self.image_stack = image_stack
        self.file_loc = file_loc
        self.processing_mode = processing_mode  # 'nuclei' or 'ramified'
        self.current_widgets = {}
        self.current_step = {"value": 0}
        self.processing_steps = ["initial_segmentation", "refine_rois", "calculate_features"]
        self.parameter_values = {}
        self.active_dock_widgets = set()
        
        # Set up processing directory
        self.inputdir = os.path.dirname(self.file_loc)
        self.basename = os.path.basename(self.file_loc).split('.')[0]
        self.processed_dir = os.path.join(self.inputdir, f"{self.basename}_processed_{self.processing_mode}")
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
        # Initialize processing state
        self.initial_segmentation = np.zeros_like(self.image_stack, dtype=np.int32)
        
        # Initialize viewer layers
        self._initialize_layers()


    def cleanup_step(self, step_number):
        """Clean up the results and layers from a specific step"""
        if step_number == 1:
            layer_name = "Intermediate segmentation"
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)
            segmented_cells_path = os.path.join(self.processed_dir, f"segmented_{self.processing_mode}.npy")
            if os.path.exists(segmented_cells_path):
                os.remove(segmented_cells_path)

        elif step_number == 2:
            layer_name = "Final segmentation"
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)
            merged_roi_array_loc = os.path.join(self.processed_dir, f"merged_roi_array_optimized_{self.processing_mode}.dat")
            if os.path.exists(merged_roi_array_loc):
                os.remove(merged_roi_array_loc)
        
        elif step_number == 3:
            connections_layer = "Closest Points Connections"
            if connections_layer in self.viewer.layers:
                self.viewer.layers.remove(connections_layer)


    def execute_processing_step(self):
        """Execute the next step in the processing pipeline based on the selected processing mode"""
        try:
            if self.current_step["value"] == 0:
                # Step 1: Initial cell segmentation
                print(f"Running initial {self.processing_mode} segmentation...")
                segmented_cells_path = os.path.join(self.processed_dir, f"segmented_{self.processing_mode}.npy")
                
                # Remove existing results if present
                self.cleanup_step(1)
                
                # Get current parameter values
                current_values = self.get_current_values()

                if self.processing_mode == 'nuclei':
                    # Process nuclei
                    labeled_cells, first_pass_params = segment_nuclei(self.image_stack, first_pass=None,
                      smooth_sigma=[0, 0.5, 1.0, 2.0],
                      min_distance=current_values.get("min_distance"),
                      min_size=current_values.get("min_size"),
                      contrast_threshold_factor=current_values.get("contrast_threshold_factor"),
                      spacing=[self.config.get('voxel_dimensions', {}).get('z', 1), self.x_spacing, self.x_spacing],
                      anisotropy_normalization_degree=current_values.get("anisotropy_normalization_degree"))
                    
                    self.first_pass_params = first_pass_params
                    
                else:  # ramified mode
                    # Process ramified cells
                    labeled_cells, first_pass_params = segment_microglia(self.image_stack, 
                     first_pass=None,
                     first_pass_params=None,
                     tubular_scales=[current_values.get("tubular_scales", 2)],
                     smooth_sigma=current_values.get("smooth_sigma", 1.0),
                     min_size=current_values.get("min_size", 100),
                     min_cell_body_size=current_values.get("min_cell_body_size", 200),
                     spacing=[self.config.get('voxel_dimensions', {}).get('z', 1), self.x_spacing, self.x_spacing],
                     anisotropy_normalization_degree=current_values.get("anisotropy_normalization_degree", 0.2),
                     threshold_method='otsu',
                     sensitivity=current_values.get("sensitivity", 0.2),
                     extract_features=False)
                
                np.save(segmented_cells_path, labeled_cells)
                self.viewer.add_labels(
                    labeled_cells,
                    name="Intermediate segmentation",
                    scale=(self.z_scale_factor, 1, 1)
                )
                
                self.current_step["value"] += 1
                self.create_step_widgets("refine_rois")

            elif self.current_step["value"] == 1:
                # Step 2: Refine ROIs
                print(f"Refining {self.processing_mode} ROIs...")
                
                # Remove existing results if present
                self.cleanup_step(2)
                
                # Get current parameter values
                current_values = self.get_current_values()
                segmented_cells_path = os.path.join(self.processed_dir, f"segmented_{self.processing_mode}.npy")
                labeled_cells = np.load(segmented_cells_path)
                
                if self.processing_mode == 'nuclei':
                    # Refine nuclear ROIs
                    merged_roi_array = segment_nuclei(self.image_stack, first_pass=labeled_cells, first_pass_params=self.first_pass_params,
                      smooth_sigma=[0, 0.5, 1.0, 2.0],
                      min_distance=current_values.get("min_distance", 10),
                      min_size=current_values.get("min_size", 100),
                      contrast_threshold_factor=current_values.get("contrast_threshold_factor", 1.5),
                      spacing=[self.config.get('voxel_dimensions', {}).get('z', 1), self.x_spacing, self.x_spacing],
                      anisotropy_normalization_degree=current_values.get("anisotropy_normalization_degree", 1.0))
                else:
                    # Refine ramified ROIs
                    merged_roi_array = segment_ramified_cells(self.image_stack, first_pass=labeled_cells, first_pass_params=self.first_pass_params,
                      smooth_sigma=[0, 0.5, 1.0, 2.0],
                      min_distance=current_values.get("min_distance", 10),
                      min_size=current_values.get("min_size", 100),
                      contrast_threshold_factor=current_values.get("contrast_threshold_factor", 1.5),
                      spacing=[self.config.get('voxel_dimensions', {}).get('z', 1), self.x_spacing, self.x_spacing],
                      connectivity=current_values.get("connectivity", 3))
                
                # Save the merged array
                merged_roi_array_loc = os.path.join(self.processed_dir, f"merged_roi_array_optimized_{self.processing_mode}.dat")
                merged_roi_array.tofile(merged_roi_array_loc)
                
                self.viewer.add_labels(
                    merged_roi_array,
                    name="Final segmentation",
                    scale=(self.z_scale_factor, 1, 1)
                )
                
                self.current_step["value"] += 1
                self.create_step_widgets("calculate_features")

            elif self.current_step["value"] == 2:
                # Step 3: Calculate features
                print(f"Calculating {self.processing_mode} features...")
                
                # Remove existing results if present
                self.cleanup_step(3)
                
                # Get current parameter values
                current_values = self.get_current_values()
                
                merged_roi_array_loc = os.path.join(self.processed_dir, f"merged_roi_array_optimized_{self.processing_mode}.dat")
                merged_roi_array = np.memmap(
                    merged_roi_array_loc,
                    dtype=np.int32,
                    mode='r',
                    shape=self.image_stack.shape
                )

                if self.processing_mode == 'nuclei':
                    # Calculate features for nuclei
                    distances_matrix, points_matrix, lines = shortest_distance(merged_roi_array, 
                                                                spacing=[self.config.get('voxel_dimensions', {}).get('z', 1), self.x_spacing, self.x_spacing])
                else:
                    # Calculate features for ramified cells - using different parameters or methods
                    distances_matrix, points_matrix, lines = calculate_ramified_features(merged_roi_array, 
                                                                spacing=[self.config.get('voxel_dimensions', {}).get('z', 1), self.x_spacing, self.x_spacing],
                                                                branch_threshold=current_values.get("branch_threshold", 10))
                
                # Save results
                distances_matrix_loc = os.path.join(self.processed_dir, f"distances_matrix_{self.processing_mode}.csv")
                points_matrix_loc = os.path.join(self.processed_dir, f"points_matrix_{self.processing_mode}.csv")
                np.savetxt(distances_matrix_loc, distances_matrix, delimiter=",")
                np.savetxt(points_matrix_loc, points_matrix, delimiter=",")

                # Add the lines connecting closest points as shapes in Napari
                self.viewer.add_shapes(
                    lines,
                    shape_type='line',
                    edge_color='red',
                    edge_width=1.5,
                    name="Closest Points Connections",
                    scale=(self.z_scale_factor, 1, 1)
                )
                
                self.current_step["value"] += 1
                print("Processing complete!")
                
                # Save final configuration
                self.save_updated_config()
            
            else:
                print("All processing steps completed.")
                
        except Exception as e:
            print(f"Error during {self.processing_mode} processing step {self.current_step['value']}: {str(e)}")
            raise

    def save_updated_config(self):
        """Save the current configuration to a YAML file"""
        config_save_path = os.path.join(self.processed_dir, f"processing_config_{self.processing_mode}.yaml")
        with open(config_save_path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)

    def clear_current_widgets(self):
        """Remove all current widgets"""
        # Get list of dock widgets from the viewer
        dock_widgets = list(self.viewer.window._dock_widgets.values())
        
        # Remove each widget
        for dock_widget in dock_widgets:
            if dock_widget in self.current_widgets:
                try:
                    self.viewer.window.remove_dock_widget(dock_widget)
                except Exception as e:
                    print(f"Warning: Failed to remove dock widget: {str(e)}")
                
        # Clear the tracking dictionary
        self.current_widgets.clear()

    def create_step_widgets(self, step_name: str):
        """Create all widgets for a processing step based on the processing mode"""
        try:
            # Remove existing widgets
            self.clear_current_widgets()
            
            # Determine the actual config key based on processing mode
            config_key = f"{step_name}_{self.processing_mode}" if f"{step_name}_{self.processing_mode}" in self.config else step_name
            
            # Create new widgets for each parameter in the step
            if config_key not in self.config:
                print(f"Warning: {config_key} not found in config")
                return
                
            step_config = self.config[config_key]
            if "parameters" not in step_config:
                print(f"Warning: no parameters found for {config_key}")
                return
            
            # Reset parameter values for this step
            self.parameter_values = {}
            
            # Create parameter widgets
            for param_name, param_config in step_config["parameters"].items():
                try:
                    # Create callback for this specific parameter
                    callback = lambda value, pn=param_name: self.parameter_changed(config_key, pn, value)
                    
                    # Create widget
                    widget = create_parameter_widget(param_name, param_config, callback)
                    dock_widget = self.viewer.window.add_dock_widget(widget, area="right")
                    self.current_widgets[dock_widget] = widget
                    
                    # Store initial value
                    self.parameter_values[param_name] = param_config["value"]
                except Exception as e:
                    print(f"Error creating widget for {param_name}: {str(e)}")
                    
        except Exception as e:
            print(f"Error in create_step_widgets: {str(e)}")

    def remove_widget(self, dock_widget):
        """Safely remove a single widget"""
        try:
            if dock_widget in self.current_widgets:
                self.viewer.window.remove_dock_widget(dock_widget)
                del self.current_widgets[dock_widget]
        except Exception as e:
            print(f"Warning: Failed to remove dock widget: {str(e)}")
        
    def _initialize_layers(self):
        """Initialize the basic layers in the viewer"""
        # Get voxel dimensions from config
        voxel_x = self.config.get('voxel_dimensions', {}).get('x', 1)
        voxel_y = self.config.get('voxel_dimensions', {}).get('y', 1)
        voxel_z = self.config.get('voxel_dimensions', {}).get('z', 1)
        self.x_spacing = voxel_x / self.image_stack.shape[1]
        self.z_scale_factor = voxel_z/self.x_spacing
        
        # Add layers
        self.viewer.add_image(
            self.image_stack, 
            name=f"Original stack ({self.processing_mode} mode)", 
            scale=(self.z_scale_factor, 1, 1)
        )
        
    def parameter_changed(self, step_name: str, param_name: str, value: Any):
        """Callback for when a parameter value changes"""
        if step_name in self.config and "parameters" in self.config[step_name]:
            self.config[step_name]["parameters"][param_name]["value"] = value
            self.parameter_values[param_name] = value

    def get_current_values(self) -> Dict[str, Any]:
        """Get current values for all parameters in the current step"""
        return self.parameter_values.copy()
    

def interactive_segmentation_with_config():
    """
    Launch interactive segmentation with dynamic GUI based on YAML configuration with PyQt5 dialogs
    """
    # Create PyQt5 app instance if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Prompt for config file
    config_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select config YAML file",
        "",
        "YAML files (*.yaml *.yml);;All files (*.*)"
    )
    
    # Load or create config
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        QMessageBox.critical(None, "Error", "No config file selected. Please select a config file.")
        return
    
    # Prompt for processing mode selection
    processing_modes = ["nuclei", "ramified"]
    processing_mode_idx, ok = QInputDialog.getItem(
        None, 
        "Processing Mode",
        "Select processing mode:",
        ["Nuclear morphology", "Ramified morphology"],
        0,  # Default to first item
        False  # Not editable
    )
    
    if not ok:
        QMessageBox.warning(None, "Warning", "No processing mode selected. Exiting.")
        return
    
    # Convert selection to mode string
    processing_mode = processing_modes[0] if "Nuclear" in processing_mode_idx else processing_modes[1]
    print(f"Selected processing mode: {processing_mode}")
    
    # Prompt for input file
    file_loc, _ = QFileDialog.getOpenFileName(
        None,
        "Select a .tif file",
        "",
        "TIFF files (*.tif);;All files (*.*)"
    )
    
    if not file_loc or not os.path.exists(file_loc):
        QMessageBox.warning(None, "Warning", "No file selected. Exiting.")
        return

    # Create processed directory with mode suffix
    inputdir = os.path.dirname(file_loc)
    basename = os.path.basename(file_loc).split('.')[0]
    processed_dir = os.path.join(inputdir, f"{basename}_processed_{processing_mode}")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Load the .tif file
    try:
        image_stack = tiff.imread(file_loc)
        print(f"Loaded stack with shape {image_stack.shape}")
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load TIFF file: {str(e)}")
        return

    # Initialize viewer and GUI manager with processing mode
    viewer = napari.Viewer(title=f"Microscopy Analysis - {processing_mode.capitalize()} Mode")
    gui_manager = DynamicGUIManager(viewer, config, image_stack, file_loc, processing_mode)
    
    @magicgui(call_button="Continue Processing")
    def continue_processing():
        """Execute the next step in the processing pipeline"""
        try:
            gui_manager.execute_processing_step()
            update_navigation_buttons()
        except Exception as e:
            QMessageBox.critical(None, "Processing Error", f"Error during processing: {str(e)}")

    @magicgui(call_button="Previous Step")
    def go_to_previous_step():
        """Go back one step in the processing pipeline"""
        if gui_manager.current_step["value"] > 0:
            gui_manager.current_step["value"] -= 1
            step_name = gui_manager.processing_steps[gui_manager.current_step["value"]]
            gui_manager.create_step_widgets(step_name)
            gui_manager.cleanup_step(gui_manager.current_step["value"] + 1)
            update_navigation_buttons()

    def update_navigation_buttons():
        """Update the state of navigation buttons"""
        previous_step_button.enabled = gui_manager.current_step["value"] > 0
        continue_processing_button.enabled = gui_manager.current_step["value"] < len(gui_manager.processing_steps)

    # Add navigation buttons
    continue_processing_button = continue_processing
    previous_step_button = go_to_previous_step
    viewer.window.add_dock_widget(continue_processing_button, area="right")
    viewer.window.add_dock_widget(previous_step_button, area="right")
    update_navigation_buttons()

    # Create initial GUI widgets
    gui_manager.create_step_widgets("initial_segmentation")

    napari.run()