import tifffile as tiff
import numpy as np
import os
import plotly.graph_objs as go
import napari
from skimage import morphology
from magicgui import magicgui
from skimage.measure import label
from tkinter import filedialog, Tk

from utils.initial_segmentation import *
from utils.split_rois import *
from utils.segmenter import *

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
    def __init__(self, viewer, config, image_stack, file_loc):
        self.viewer = viewer
        self.config = config
        self.image_stack = image_stack
        self.file_loc = file_loc
        self.current_widgets = {}
        self.current_step = {"value": 0}
        self.processing_steps = ["initial_segmentation", "merge_rois", "split_rois"]
        self.parameter_values = {}
        self.active_dock_widgets = set()
        
        # Set up processing directory
        self.inputdir = os.path.dirname(self.file_loc)
        self.basename = os.path.basename(self.file_loc).split('.')[0]
        self.processed_dir = os.path.join(self.inputdir, f"{self.basename}_processed")
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        
        # Initialize processing state
        self.initial_segmentation = np.zeros_like(self.image_stack, dtype=np.int32)
        self.slice_settings = {}
        
        # Set up image enhancement
        self.enhanced_stack = adaptive_contrast_enhancement(self.image_stack)
        
        # Initialize viewer layers
        self._initialize_layers()


    def cleanup_step(self, step_number):
        """Clean up the results and layers from a specific step"""
        if step_number == 1:
            if "Intermediate segmentation 1" in self.viewer.layers:
                self.viewer.layers.remove("Intermediate segmentation 1")
            segmented_cells_path = os.path.join(self.processed_dir, "segmented_cells.npy")
            if os.path.exists(segmented_cells_path):
                os.remove(segmented_cells_path)

        elif step_number == 2:
            if "Intermediate segmentation 2" in self.viewer.layers:
                self.viewer.layers.remove("Intermediate segmentation 2")
            merged_roi_array_loc = os.path.join(self.processed_dir, "merged_roi_array_optimized.dat")
            if os.path.exists(merged_roi_array_loc):
                os.remove(merged_roi_array_loc)

        elif step_number == 3:
            if "Segmentation without large volumes" in self.viewer.layers:
                self.viewer.layers.remove("Segmentation without large volumes")
            updated_stack_loc = os.path.join(self.processed_dir, "updated_stack.npy")
            if os.path.exists(updated_stack_loc):
                os.remove(updated_stack_loc)


    def execute_processing_step(self):
        """Execute the next step in the processing pipeline"""
        try:
            if self.current_step["value"] == 0:
                # Step 1: Initial cell segmentation
                print("Running initial cell segmentation...")
                segmented_cells_path = os.path.join(self.processed_dir, "segmented_cells.npy")
                
                # Remove existing results if present
                self.cleanup_step(1)
                
                # Get current parameter values
                current_values = self.get_current_values()

                #Create the first pass params dictionary:
                self.first_pass_params = {
                    "min_distance": current_values.get("min_distance"),
                    "min_size": current_values.get("min_size"),
                    "contrast_threshold_factor": current_values.get("contrast_threshold_factor"),
                    "spacing": [self.config.get('voxel_dimensions', {}).get('z', 1), self.x_spacing, self.x_spacing],
                    "anisotropy_normalization_degree": current_values.get("anisotropy_normalization_degree")
                }

                labeled_cells, first_pass_params = segment_nuclei(self.image_stack, first_pass=None,
                  smooth_sigma=[0, 0.5, 1.0, 2.0],
                  min_distance=self.first_pass_params['min_distance'],
                  min_size=self.first_pass_params['min_size'],
                  contrast_threshold_factor=self.first_pass_params['contrast_threshold_factor'],
                  spacing=[self.config.get('voxel_dimensions', {}).get('z', 1), self.x_spacing, self.x_spacing],
                  anisotropy_normalization_degree=self.first_pass_params['anisotropy_normalization_degree'])
                
                self.first_pass_params = first_pass_params
                
                np.save(segmented_cells_path, labeled_cells)
                self.viewer.add_labels(
                    labeled_cells,
                    name="Intermediate segmentation 1",
                    scale=(self.z_scale_factor, 1, 1)
                )
                
                self.current_step["value"] += 1
                self.create_step_widgets("merge_rois")

            elif self.current_step["value"] == 1:
                # Step 2: Split large ROIs
                print("Splitting large ROIs...")
                segmented_cells_path = os.path.join(self.processed_dir, "segmented_cells.npy")
                merged_roi_array_loc = os.path.join(self.processed_dir, "merged_roi_array_optimized.dat")
                
                # Remove existing results if present
                self.cleanup_step(2)
                
                # Get current parameter values
                current_values = self.get_current_values()
                
                labeled_cells = np.load(segmented_cells_path)
                

                merged_roi_array = segment_nuclei(self.image_stack, first_pass=labeled_cells, first_pass_params=self.first_pass_params,
                  smooth_sigma=[0, 0.5, 1.0, 2.0],
                  min_distance=current_values.get("min_distance", 10),
                  min_size=current_values.get("min_size", 100),
                  contrast_threshold_factor=current_values.get("contrast_threshold_factor", 1.5),
                  spacing=[self.config.get('voxel_dimensions', {}).get('z', 1), self.x_spacing, self.x_spacing],
                  anisotropy_normalization_degree=current_values.get("anisotropy_normalization_degree", 1.0))
                
                # Save the merged array
                merged_roi_array.tofile(merged_roi_array_loc)
                
                self.viewer.add_labels(
                    merged_roi_array,
                    name="Intermediate segmentation 2",
                    scale=(self.z_scale_factor, 1, 1)
                )
                
                self.current_step["value"] += 1
                self.create_step_widgets("split_rois")

            elif self.current_step["value"] == 2:
                # Step 3: Split large ROIs
                print("Splitting large ROIs...")
                merged_roi_array_loc = os.path.join(self.processed_dir, "merged_roi_array_optimized.dat")
                updated_stack_loc = os.path.join(self.processed_dir, "updated_stack.npy")
                
                # Remove existing results if present
                self.cleanup_step(3)
                
                # Get current parameter values
                current_values = self.get_current_values()
                
                merged_roi_array = np.memmap(
                    merged_roi_array_loc,
                    dtype=np.int32,
                    mode='r',
                    shape=self.image_stack.shape
                )
                
                updated_stack = split_large_rois_with_intensity(
                    merged_roi_array,
                    self.config['voxel_dimensions']['z'],
                    self.x_spacing,
                    mean_guess=current_values.get("mean_guess", 5000),
                    std_guess=current_values.get("std_guess", 2500),
                    tempdir=self.processed_dir,
                    max_iters=current_values.get("max_iters", 10)
                )
                
                np.save(updated_stack_loc, updated_stack)
                self.viewer.add_labels(
                    updated_stack,
                    name="Segmentation without large volumes",
                    scale=(self.z_scale_factor, 1, 1)
                )
                
                self.current_step["value"] += 1
                print("Processing complete!")
                
                # Save final configuration
                self.save_updated_config()
            
            else:
                print("All processing steps completed.")
                
        except Exception as e:
            print(f"Error during processing step {self.current_step['value']}: {str(e)}")
            raise

    def save_updated_config(self):
        """Save the current configuration to a YAML file"""
        config_save_path = os.path.join(self.processed_dir, "processing_config.yaml")
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
        """Create all widgets for a processing step"""
        try:
            # Remove existing widgets
            self.clear_current_widgets()
            
            # Create new widgets for each parameter in the step
            if step_name not in self.config:
                print(f"Warning: {step_name} not found in config")
                return
                
            step_config = self.config[step_name]
            if "parameters" not in step_config:
                print(f"Warning: no parameters found for {step_name}")
                return
            
            # Reset parameter values for this step
            self.parameter_values = {}
            
            # Create parameter widgets
            for param_name, param_config in step_config["parameters"].items():
                try:
                    # Create callback for this specific parameter
                    callback = lambda value, pn=param_name: self.parameter_changed(step_name, pn, value)
                    
                    # Create widget
                    widget = create_parameter_widget(param_name, param_config, callback)
                    dock_widget = self.viewer.window.add_dock_widget(widget, area="right")
                    self.current_widgets[dock_widget] = widget
                    
                    # Store initial value
                    self.parameter_values[param_name] = param_config["value"]
                except Exception as e:
                    print(f"Error creating widget for {param_name}: {str(e)}")
            
            # Add Update Mask button for initial segmentation
            if step_name == "initial_segmentation":
                try:
                    @magicgui(call_button="Update Mask")
                    def update_mask():
                        slice_idx = self.viewer.dims.current_step[0]
                        params = self.get_current_values()
                        self.apply_mask(
                            slice_idx,
                            params["intensity_threshold"],
                            params["min_volume"],
                            params["downsample_factor"]
                        )
                    
                    dock_widget = self.viewer.window.add_dock_widget(update_mask, area="right")
                    self.current_widgets[dock_widget] = update_mask
                except Exception as e:
                    print(f"Error creating update mask widget: {str(e)}")
                    
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
            name="Original stack", 
            scale=(self.z_scale_factor, 1, 1)
        )
        self.viewer.add_image(
            self.enhanced_stack, 
            name="Enhanced stack", 
            scale=(self.z_scale_factor, 1, 1)
        )
        self.viewer.add_labels(
            self.initial_segmentation, 
            name="Initial segmentation", 
            scale=(self.z_scale_factor, 1, 1)
        )
        
    def parameter_changed(self, step_name: str, param_name: str, value: Any):
        """Callback for when a parameter value changes"""
        if step_name in self.config and "parameters" in self.config[step_name]:
            self.config[step_name]["parameters"][param_name]["value"] = value
            self.parameter_values[param_name] = value
        
    def apply_mask(self, slice_index, intensity_threshold, min_volume, downsample_factor):
        """Generate and apply the segmentation mask for a given slice."""
        try:
            image_slice = self.enhanced_stack[slice_index]
            downsampled = downsample_stack(image_slice[None, ...], factor=downsample_factor)[0]
            seed_threshold = intensity_threshold * np.max(downsampled)
            seeds = np.argwhere(downsampled >= seed_threshold)

            mask = np.zeros_like(downsampled, dtype=np.int32)
            for seed in seeds:
                if downsampled[tuple(seed)] >= seed_threshold:
                    mask[tuple(seed)] = 1
            mask = morphology.binary_dilation(mask)
            labeled_mask = label(mask)
            filtered_mask = morphology.remove_small_objects(labeled_mask, min_size=min_volume)
            if downsample_factor > 1:
                filtered_mask = upsample_stack(filtered_mask, image_slice.shape)

            self.initial_segmentation[slice_index] = filtered_mask
            self.viewer.layers["Initial segmentation"].data = self.initial_segmentation
            self.slice_settings[slice_index] = {
                "intensity_threshold": intensity_threshold,
                "min_volume": min_volume,
                "downsample_factor": downsample_factor
            }
            
        except Exception as e:
            print(f"Error in apply_mask: {str(e)}")


    def get_current_values(self) -> Dict[str, Any]:
        """Get current values for all parameters in the current step"""
        return self.parameter_values.copy()
        
    def apply_initial_segmentation(self, values):
        """Apply initial segmentation with current parameters"""
        slice_idx = self.viewer.dims.current_step[0]
        self.apply_mask(
            slice_idx, 
            values["intensity_threshold"],
            values["min_volume"],
            values["downsample_factor"]
        )

    def apply_merge_rois(self, values):
        """Apply ROI merging with current parameters from the GUI"""
        if "Intermediate segmentation 1" not in self.viewer.layers:
            print("Error: Previous segmentation layer not found")
            return
            
        labeled_cells = self.viewer.layers["Intermediate segmentation 1"].data
        
        merged_roi_array = split_merged_masks(
            labeled_cells,
            self.enhanced_stack,
            min_distance=values["min_distance"],
            min_intensity_ratio=values["min_intensity_ratio"]
        )
        
        self.viewer.add_labels(
            merged_roi_array, 
            name="Intermediate segmentation 2", 
            scale=(self.z_scale_factor, 1, 1)
        )

    def apply_split_rois(self, values):
        """Apply ROI splitting with current parameters from the GUI"""
        if "Intermediate segmentation 2" not in self.viewer.layers:
            print("Error: Previous segmentation layer not found")
            return
            
        merged_roi_array = self.viewer.layers["Intermediate segmentation 2"].data
        
        updated_stack = split_large_rois_with_intensity(
            merged_roi_array,
            self.config['voxel_dimensions']['z'],
            self.x_spacing,
            mean_guess=values["mean_guess"],
            std_guess=values["std_guess"],
            tempdir=self.processed_dir,
            max_iters=values["max_iters"]
        )
        
        self.viewer.add_labels(
            updated_stack, 
            name="Segmentation without large volumes", 
            scale=(self.z_scale_factor, 1, 1)
        )

def interactive_segmentation_with_config():
    """
    Launch interactive segmentation with dynamic GUI based on YAML configuration
    """
    # First prompt for config file
    Tk().withdraw()
    config_path = filedialog.askopenfilename(
        title="Select config YAML file (optional)", 
        filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
    )
    
    # Load or create config
    if config_path:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    else:
        print("No config file selected. Please select a config file.")
        return
    
    # Prompt for input file
    file_loc = filedialog.askopenfilename(
        title="Select a .tif file", 
        filetypes=[("TIFF files", "*.tif")]
    )
    if not file_loc:
        print("No file selected. Exiting.")
        return

    # Create processed directory
    inputdir = os.path.dirname(file_loc)
    basename = os.path.basename(file_loc).split('.')[0]
    processed_dir = os.path.join(inputdir, f"{basename}_processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Load the .tif file
    image_stack = tiff.imread(file_loc)
    print(f"Loaded stack with shape {image_stack.shape}")

    # Initialize viewer and GUI manager
    viewer = napari.Viewer()
    gui_manager = DynamicGUIManager(viewer, config, image_stack, file_loc)
    
    @magicgui(call_button="Continue Processing")
    def continue_processing():
        """Execute the next step in the processing pipeline"""
        gui_manager.execute_processing_step()
        update_navigation_buttons()

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