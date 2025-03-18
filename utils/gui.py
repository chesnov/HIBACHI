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

seed = 42
np.random.seed(seed)         # For NumPy

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

def check_processing_state(processed_dir, processing_mode):
    """
    Check the processing state by looking for checkpoint files
    Returns the current step number (0, 1, 2, or 3) based on found files
    """
    # Check for step 3 completion (feature calculation)
    metrics_df_loc = os.path.join(processed_dir, f"metrics_df_{processing_mode}.csv")
    if os.path.exists(metrics_df_loc):
        return 3  # All steps completed
    
    # Check for step 2 completion (refined ROIs)
    merged_roi_array_loc = os.path.join(processed_dir, f"final_segmentation_{processing_mode}.dat")
    if os.path.exists(merged_roi_array_loc):
        return 2  # Ready for feature calculation
    
    # Check for step 1 completion (initial segmentation)
    segmented_cells_path = os.path.join(processed_dir, f"initial_segmentation_{processing_mode}.npy")
    if os.path.exists(segmented_cells_path):
        return 1  # Ready for ROI refinement
    
    # No checkpoints found
    return 0  # Ready for initial segmentation

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
        
        # Check for existing processed files and restore state
        self.restore_from_checkpoint()
    
    def restore_from_checkpoint(self):
        """Check for existing processed files and restore the processing state"""
        checkpoint_step = check_processing_state(self.processed_dir, self.processing_mode)
        
        # If checkpoints exist, ask user if they want to resume
        if checkpoint_step > 0:
            step_names = {
                1: "initial segmentation",
                2: "ROI refinement",
                3: "feature calculation"
            }
            
            if checkpoint_step == 3:
                # All processing completed
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Question)
                msg.setWindowTitle("Resume Processing")
                msg.setText(f"All processing steps already completed for this {self.processing_mode} dataset.")
                msg.setInformativeText("Do you want to view the results or restart processing?")
                view_button = msg.addButton("View Results", QMessageBox.YesRole)
                restart_button = msg.addButton("Restart Processing", QMessageBox.NoRole)
                msg.setDefaultButton(view_button)
                msg.exec_()
                
                if msg.clickedButton() == view_button:
                    # Load existing data and set current step
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                else:
                    # User chose to restart - confirm deletion
                    self._confirm_restart()
            else:
                # Partial processing completed
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Question)
                msg.setWindowTitle("Resume Processing")
                msg.setText(f"Found checkpoint after {step_names[checkpoint_step]}.")
                msg.setInformativeText("Do you want to resume from this point?")
                resume_button = msg.addButton("Resume", QMessageBox.YesRole)
                restart_button = msg.addButton("Restart", QMessageBox.NoRole)
                msg.setDefaultButton(resume_button)
                msg.exec_()
                
                if msg.clickedButton() == resume_button:
                    # Load existing data and set current step
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                    
                    # Setup widgets for the next step
                    if checkpoint_step < 3:
                        self.create_step_widgets(self.processing_steps[checkpoint_step])
                else:
                    # User chose to restart - confirm deletion
                    self._confirm_restart()
        else:
            # No checkpoints found, start from the beginning
            self.create_step_widgets("initial_segmentation")
        
    def _confirm_restart(self):
        """Confirm and handle restarting from scratch"""
        confirm_box = QMessageBox()
        confirm_box.setIcon(QMessageBox.Warning)
        confirm_box.setWindowTitle("Confirm Restart")
        confirm_box.setText("This will delete all existing processed files.")
        confirm_box.setInformativeText("Are you sure you want to restart?")
        
        yes_button = confirm_box.addButton("Yes, delete files", QMessageBox.YesRole)
        no_button = confirm_box.addButton("No, keep files", QMessageBox.NoRole)
        confirm_box.setDefaultButton(no_button)
        
        confirm_box.exec_()
        
        if confirm_box.clickedButton() == yes_button:
            # Delete all existing processed files
            self.delete_checkpoint_files()
            self.current_step["value"] = 0
            self.create_step_widgets("initial_segmentation")
        else:
            # User canceled deletion, load checkpoint data anyway
            checkpoint_step = check_processing_state(self.processed_dir, self.processing_mode)
            self.load_checkpoint_data(checkpoint_step)
            self.current_step["value"] = checkpoint_step
            if checkpoint_step < 3:
                self.create_step_widgets(self.processing_steps[checkpoint_step])
    
    def delete_checkpoint_files(self):
        """Delete all checkpoint files for the current processing mode"""
        files_to_delete = [
            os.path.join(self.processed_dir, f"initial_segmentation_{self.processing_mode}.npy"),
            os.path.join(self.processed_dir, f"final_segmentation_{self.processing_mode}.dat"),
            os.path.join(self.processed_dir, f"distances_matrix_{self.processing_mode}.csv"),
            os.path.join(self.processed_dir, f"points_matrix_{self.processing_mode}.csv"),
            os.path.join(self.processed_dir, f"metrics_df_{self.processing_mode}.csv"),
            os.path.join(self.processed_dir, f"ramification_metrics_{self.processing_mode}.csv"),
            os.path.join(self.processed_dir, f"cell_bodies_{self.processing_mode}.npy"),
            os.path.join(self.processed_dir, f"processing_config_{self.processing_mode}.yaml")
        ]
        
        for file_path in files_to_delete:
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.dat'):
                        # For memory-mapped files we need to ensure they're closed
                        # This is a simple approach - in production you might need more robust handling
                        import gc
                        gc.collect()  # Force garbage collection to close any open handles
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {str(e)}")
    
    def load_checkpoint_data(self, checkpoint_step):
        """Load data from checkpoint files and restore the appropriate visualization layers"""
        try:
            if checkpoint_step >= 1:
                # Load and display initial segmentation
                segmented_cells_path = os.path.join(self.processed_dir, f"initial_segmentation_{self.processing_mode}.npy")
                if os.path.exists(segmented_cells_path):
                    labeled_cells = np.load(segmented_cells_path)
                    # Check if layer already exists
                    if "Intermediate segmentation" not in self.viewer.layers:
                        self.viewer.add_labels(
                            labeled_cells,
                            name="Intermediate segmentation",
                            scale=(self.z_scale_factor, 1, 1)
                        )
                    print(f"Loaded initial segmentation from checkpoint with {np.max(labeled_cells)} objects")
                    
                    # If this is a nuclei segmentation, we need to also load the first_pass_params
                    if self.processing_mode == 'nuclei':
                        # Try to load first_pass_params from config if available
                        config_file = os.path.join(self.processed_dir, f"processing_config_{self.processing_mode}.yaml")
                        if os.path.exists(config_file):
                            with open(config_file, 'r') as file:
                                saved_config = yaml.safe_load(file)
                                if 'first_pass_params' in saved_config:
                                    self.first_pass_params = saved_config['first_pass_params']
                
            if checkpoint_step >= 2:
                # Load and display refined ROIs
                merged_roi_array_loc = os.path.join(self.processed_dir, f"final_segmentation_{self.processing_mode}.dat")
                if os.path.exists(merged_roi_array_loc):
                    segmented_somas_loc = os.path.join(self.processed_dir, f"cell_bodies_{self.processing_mode}.npy")
                    if os.path.exists(segmented_somas_loc):
                        cell_bodies = np.fromfile(segmented_somas_loc, dtype=np.int32).reshape(self.image_stack.shape)
                        if "Cell bodies" not in self.viewer.layers:
                            self.viewer.add_labels(
                                cell_bodies,
                                name="Cell bodies",
                                scale=(self.z_scale_factor, 1, 1)
                            )
                    # Map the file for memory efficiency
                    merged_roi_array = np.memmap(
                        merged_roi_array_loc,
                        dtype=np.int32,
                        mode='r',
                        shape=self.image_stack.shape
                    )
                    # Check if layer already exists
                    if "Final segmentation" not in self.viewer.layers:
                        self.viewer.add_labels(
                            merged_roi_array,
                            name="Final segmentation",
                            scale=(self.z_scale_factor, 1, 1)
                        )
                    print(f"Loaded refined ROIs from checkpoint")
            
            if checkpoint_step >= 3:
                # If features are calculated, load and display connections
                distances_matrix_loc = os.path.join(self.processed_dir, f"distances_matrix_{self.processing_mode}.csv")
                points_matrix_loc = os.path.join(self.processed_dir, f"points_matrix_{self.processing_mode}.csv")
                depth_df_loc = os.path.join(self.processed_dir, f"depth_df_{self.processing_mode}.csv")
                
                if os.path.exists(distances_matrix_loc) and os.path.exists(points_matrix_loc):
                    # We need to reconstruct the lines for visualization
                    import pandas as pd
                    
                    # Load lines from CSV into pandas df and convert into np array with shape (n_labels, 2, 3)
                    lines_loc = os.path.join(self.processed_dir, f"lines_{self.processing_mode}.csv")
                    lines = pd.read_csv(lines_loc, header=None).values.reshape(-1, 2, 3)
                    
                    # Check if layer already exists
                    if "Closest Points Connections" not in self.viewer.layers:
                        self.viewer.add_shapes(
                            lines,
                            shape_type='line',
                            edge_color='red',
                            edge_width=1,
                            name="Closest Points Connections",
                            scale=(self.z_scale_factor, 1, 1)
                        )

                    #Load the skeleton array
                    skeleton_array_loc = os.path.join(self.processed_dir, f"skeleton_array_{self.processing_mode}.npy")
                    if os.path.exists(skeleton_array_loc):
                        skeleton_array = np.load(skeleton_array_loc)
                        if "Skeletons" not in self.viewer.layers:
                            self.viewer.add_labels(
                                skeleton_array,
                                name="Skeletons",
                                scale=(self.z_scale_factor, 1, 1)
                            )
                    print(f"Loaded feature data and connections from checkpoint")
                    
                    # Show a summary of the metrics
                    metrics_df_loc = os.path.join(self.processed_dir, f"metrics_df_{self.processing_mode}.csv")
                    if os.path.exists(metrics_df_loc):
                        metrics_df = pd.read_csv(metrics_df_loc)
                        num_cells = len(metrics_df)
                        print(f"Analysis complete: {num_cells} {self.processing_mode} cells analyzed")
                        
                        # For ramified cells, also show ramification metrics
                        ramification_metrics_loc = os.path.join(self.processed_dir, f"ramification_metrics_{self.processing_mode}.csv")
                        if os.path.exists(ramification_metrics_loc) and self.processing_mode == 'ramified':
                            ramification_df = pd.read_csv(ramification_metrics_loc)
                            print(f"Ramification metrics available for {len(ramification_df)} cells")
                    #check if the depth_df exists
                    if not os.path.exists(depth_df_loc):
                        #call the function to calculate the depth_df
                        depth_df = calculate_depth_df(merged_roi_array, [self.z_spacing, self.x_spacing, self.y_spacing])
                        depth_df.to_csv(depth_df_loc)
            
            config_file = os.path.join(self.processed_dir, f"processing_config_{self.processing_mode}.yaml")
            if os.path.exists(config_file):
                with open(config_file, 'r') as file:
                    saved_config = yaml.safe_load(file)
                    # Update only the relevant sections of the config
                    for key in saved_config:
                        if key in self.config:
                            self.config[key] = saved_config[key]
                    print(f"Loaded saved configuration parameters")
            
        except Exception as e:
            print(f"Error loading checkpoint data: {str(e)}")
            raise
            
        except Exception as e:
            print(f"Error loading checkpoint data: {str(e)}")
            raise
    def cleanup_step(self, step_number):
        """Clean up the results and layers from a specific step"""
        if step_number == 1:
            layer_name = "Intermediate segmentation"
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)
            segmented_cells_path = os.path.join(self.processed_dir, f"initial_segmentation_{self.processing_mode}.npy")
            if os.path.exists(segmented_cells_path):
                os.remove(segmented_cells_path)

        elif step_number == 2:
            layer_name = "Final segmentation"
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)
            merged_roi_array_loc = os.path.join(self.processed_dir, f"final_segmentation_{self.processing_mode}.dat")
            if os.path.exists(merged_roi_array_loc):
                os.remove(merged_roi_array_loc)
            cell_bodies_loc = os.path.join(self.processed_dir, f"cell_bodies_{self.processing_mode}.npy")
            if os.path.exists(cell_bodies_loc):
                os.remove(cell_bodies_loc)
            if "Cell bodies" in self.viewer.layers:
                self.viewer.layers.remove("Cell bodies")
        
        elif step_number == 3:
            connections_layer = "Closest Points Connections"
            if connections_layer in self.viewer.layers:
                self.viewer.layers.remove(connections_layer)
            skeletons_layer = "Skeletons"
            if skeletons_layer in self.viewer.layers:
                self.viewer.layers.remove(skeletons_layer)
            metrics_df_loc = os.path.join(self.processed_dir, f"metrics_df_{self.processing_mode}.csv")
            if os.path.exists(metrics_df_loc):
                os.remove(metrics_df_loc)
            ramification_metrics_loc = os.path.join(self.processed_dir, f"ramification_summary_{self.processing_mode}.csv")
            if os.path.exists(ramification_metrics_loc):
                os.remove(ramification_metrics_loc)
            lines_loc = os.path.join(self.processed_dir, f"lines_{self.processing_mode}.csv")
            if os.path.exists(lines_loc):
                os.remove(lines_loc)
            skeleton_array_loc = os.path.join(self.processed_dir, f"skeleton_array_{self.processing_mode}.npy")
            if os.path.exists(skeleton_array_loc):
                os.remove(skeleton_array_loc)
            branch_data_loc = os.path.join(self.processed_dir, f"branch_data_{self.processing_mode}.csv")
            if os.path.exists(branch_data_loc):
                os.remove(branch_data_loc)
            endpoint_data_loc = os.path.join(self.processed_dir, f"endpoint_data_{self.processing_mode}.csv")
            if os.path.exists(endpoint_data_loc):
                os.remove(endpoint_data_loc)
            points_data_loc = os.path.join(self.processed_dir, f"points_matrix_{self.processing_mode}.csv")
            if os.path.exists(points_data_loc):
                os.remove(points_data_loc)
            distances_matrix_loc = os.path.join(self.processed_dir, f"distances_matrix_{self.processing_mode}.csv")
            if os.path.exists(distances_matrix_loc):
                os.remove(distances_matrix_loc)


    def execute_processing_step(self):
        """Execute the next step in the processing pipeline based on the selected processing mode"""
        #Start timer to measure how long this step takes
        start_time = time.time()
        try:
            if self.current_step["value"] == 0:
                # Step 1: Initial cell segmentation
                print(f"Running initial {self.processing_mode} segmentation...")
                segmented_cells_path = os.path.join(self.processed_dir, f"initial_segmentation_{self.processing_mode}.npy")
                
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
                      spacing=[self.z_spacing, self.x_spacing, self.y_spacing],
                      anisotropy_normalization_degree=current_values.get("anisotropy_normalization_degree"))
                    
                    self.first_pass_params = first_pass_params
                    
                else:  # ramified mode
                    # Process ramified cells
                    labeled_cells, first_pass_params = segment_microglia_first_pass(
                                                            self.image_stack,
                                                            tubular_scales=[current_values.get("tubular_scales", 2)],
                                                            smooth_sigma=current_values.get("smooth_sigma", 1.0),
                                                            min_size=current_values.get("min_size", 100),
                                                            spacing=[self.z_spacing, self.x_spacing, self.y_spacing],
                                                            anisotropy_normalization_degree=current_values.get("anisotropy_normalization_degree", 0.2),
                                                            sensitivity=current_values.get("sensitivity", 0.2)
                                                        )
                
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
                segmented_cells_path = os.path.join(self.processed_dir, f"initial_segmentation_{self.processing_mode}.npy")
                labeled_cells = np.load(segmented_cells_path)
                
                if self.processing_mode == 'nuclei':
                    # Refine nuclear ROIs
                    merged_roi_array = segment_nuclei(self.image_stack, first_pass=labeled_cells, first_pass_params=self.first_pass_params,
                      smooth_sigma=[0, 0.5, 1.0, 2.0],
                      min_distance=current_values.get("min_distance", 10),
                      min_size=current_values.get("min_size", 100),
                      contrast_threshold_factor=current_values.get("contrast_threshold_factor", 1.5),
                      spacing=[self.z_spacing, self.x_spacing, self.y_spacing],
                      anisotropy_normalization_degree=current_values.get("anisotropy_normalization_degree", 1.0))
                else:
                    # Refine ramified ROIs
                    cell_bodies = extract_soma_masks(labeled_cells, current_values.get("small_object_percentile"), current_values.get("thickness_percentile"))
                    merged_roi_array = separate_multi_soma_cells(labeled_cells, self.image_stack, cell_bodies, current_values.get('min_size_threshold'))

                    self.viewer.add_labels(
                        cell_bodies,
                        name="Cell bodies",
                        scale=(self.z_scale_factor, 1, 1)
                    )

                    #Save the cell bodies
                    cell_bodies_loc = os.path.join(self.processed_dir, f"cell_bodies_{self.processing_mode}.npy")
                    cell_bodies.tofile(cell_bodies_loc)
                
                # Save the merged array
                merged_roi_array_loc = os.path.join(self.processed_dir, f"final_segmentation_{self.processing_mode}.dat")
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
                
                merged_roi_array_loc = os.path.join(self.processed_dir, f"final_segmentation_{self.processing_mode}.dat")
                merged_roi_array = np.memmap(
                    merged_roi_array_loc,
                    dtype=np.int32,
                    mode='r',
                    shape=self.image_stack.shape
                )

                if self.processing_mode == 'nuclei':
                    # Calculate features for nuclei
                    distances_matrix, points_matrix, lines = shortest_distance(merged_roi_array, 
                                                                spacing=[self.z_spacing, self.x_spacing, self.y_spacing])
                    metrics_df, ramification_metrics = analyze_segmentation(merged_roi_array, 
                                                                            spacing=[self.z_spacing, self.x_spacing, self.y_spacing], 
                                                                            calculate_skeletons=False)
                else:
                    # Calculate features for ramified cells - using different parameters or methods
                    distances_matrix, points_matrix, lines = shortest_distance(merged_roi_array, 
                                                                spacing=[self.z_spacing, self.x_spacing, self.y_spacing])
                    metrics_df, ramification_metrics = analyze_segmentation(merged_roi_array, 
                                                                            spacing=[self.z_spacing, self.x_spacing, self.y_spacing], 
                                                                            calculate_skeletons=True)
                
                # Save results
                distances_matrix_loc = os.path.join(self.processed_dir, f"distances_matrix_{self.processing_mode}.csv")
                points_matrix_loc = os.path.join(self.processed_dir, f"points_matrix_{self.processing_mode}.csv")
                metrics_df_loc = os.path.join(self.processed_dir, f"metrics_df_{self.processing_mode}.csv")
                ramification_metrics_loc = os.path.join(self.processed_dir, f"ramification_metrics_{self.processing_mode}.csv")
                #save distances_matrix and points_matrix as pandas dataframes
                distances_matrix.to_csv(distances_matrix_loc)
                points_matrix.to_csv(points_matrix_loc)
                metrics_df.to_csv(metrics_df_loc)
                if ramification_metrics is not None:
                    ramification_summary_df, branch_data_df, endpoint_data_df, skeleton_array = ramification_metrics
                    ramification_summary_loc = os.path.join(self.processed_dir, f"ramification_summary_{self.processing_mode}.csv")
                    branch_data_loc = os.path.join(self.processed_dir, f"branch_data_{self.processing_mode}.csv")
                    endpoint_data_loc = os.path.join(self.processed_dir, f"endpoint_data_{self.processing_mode}.csv")
                    ramification_summary_df.to_csv(ramification_summary_loc)
                    branch_data_df.to_csv(branch_data_loc)  
                    endpoint_data_df.to_csv(endpoint_data_loc)

                    #Save the skeleton array
                    skeleton_array_loc = os.path.join(self.processed_dir, f"skeleton_array_{self.processing_mode}.npy")
                    np.save(skeleton_array_loc, skeleton_array)

                    #Save the lines into a csv file
                    lines_loc = os.path.join(self.processed_dir, f"lines_{self.processing_mode}.csv")
                    with open(lines_loc, 'w') as f:
                        for line in lines:
                            f.write(f"{line[0][0]},{line[0][1]},{line[0][2]},{line[1][0]},{line[1][1]},{line[1][2]}\n")

                    self.viewer.add_labels(
                        skeleton_array,
                        name="Skeletons",
                        scale=(self.z_scale_factor, 1, 1)
                    )

                # Add the lines connecting closest points as shapes in Napari
                self.viewer.add_shapes(
                    lines,
                    shape_type='line',
                    edge_color='red',
                    edge_width=1,
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
        finally:
            #End timer
            end_time = time.time()
            print(f"Step {self.current_step['value']} took {end_time - start_time} seconds")

    def save_updated_config(self):
        """Save the current configuration to a YAML file in the output directory"""
        config_save_path = os.path.join(self.processed_dir, f"processing_config_{self.processing_mode}.yaml")
        
        # Prepare updated config
        updated_config = self.config.copy()
        
        # Make sure 'mode' is preserved
        if 'mode' not in updated_config:
            updated_config['mode'] = self.processing_mode
        
        with open(config_save_path, 'w') as file:
            yaml.dump(updated_config, file, default_flow_style=False)
        print(f"Configuration saved to {config_save_path}")

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
        self.y_spacing = voxel_y / self.image_stack.shape[2]
        self.z_spacing = voxel_z / self.image_stack.shape[0]
        self.z_scale_factor = self.z_spacing/self.x_spacing
        
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

    # Find YAML file in the same directory as the TIF file
    input_dir = os.path.dirname(file_loc)
    yaml_files = [f for f in os.listdir(input_dir) if f.endswith(('.yaml', '.yml'))]
    
    if not yaml_files:
        QMessageBox.critical(None, "Error", "No YAML configuration file found in the same directory as the TIF file.")
        return
    
    if len(yaml_files) > 1:
        QMessageBox.critical(None, "Error", f"Multiple YAML files found in the directory. Please keep only one YAML file.")
        return
    
    config_path = os.path.join(input_dir, yaml_files[0])
    
    # Load config
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load YAML file: {str(e)}")
        return
    
    # Extract processing mode from config
    if 'mode' not in config:
        QMessageBox.critical(None, "Error", "The YAML file does not contain a 'mode' field. Please add 'mode: nuclei' or 'mode: ramified' to the YAML file.")
        return
    
    processing_mode = config['mode']
    if processing_mode not in ["nuclei", "ramified"]:
        QMessageBox.critical(None, "Error", f"Invalid processing mode '{processing_mode}' in YAML file. Mode must be 'nuclei' or 'ramified'.")
        return
    
    print(f"Using processing mode from config: {processing_mode}")
    
    # Create processed directory with mode suffix
    basename = os.path.basename(file_loc).split('.')[0]
    processed_dir = os.path.join(input_dir, f"{basename}_processed_{processing_mode}")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    # Check for existing processed files and config in the processed directory
    output_config_path = os.path.join(processed_dir, f"processing_config_{processing_mode}.yaml")
    
    # Check if we should use the config from the checkpoint
    checkpoint_step = check_processing_state(processed_dir, processing_mode)
    if checkpoint_step > 0 and os.path.exists(output_config_path):
        try:
            with open(output_config_path, 'r') as file:
                config = yaml.safe_load(file)
            print(f"Loaded configuration from checkpoint at step {checkpoint_step}")
        except Exception as e:
            print(f"Warning: Failed to load config from checkpoint: {str(e)}")
            # Continue with the original config if checkpoint config fails to load
    
    # Load the .tif file
    try:
        image_stack = tiff.imread(file_loc)
        print(f"Loaded stack with shape {image_stack.shape}")
    except Exception as e:
        QMessageBox.critical(None, "Error", f"Failed to load TIFF file: {str(e)}")
        return

    # Initialize viewer and GUI manager with processing mode
    viewer = napari.Viewer(title=f"Microscopy Analysis - {processing_mode.capitalize()} Mode")
    
    # Create a modified DynamicGUIManager that updates the config file after each step
    class ModifiedGUIManager(DynamicGUIManager):
        def __init__(self, viewer, config, image_stack, file_loc, processing_mode, original_config_path):
            super().__init__(viewer, config, image_stack, file_loc, processing_mode)
            self.original_config_path = original_config_path
            
        def execute_processing_step(self):
            """Override to save config after each step"""
            result = super().execute_processing_step()
            # Save updated config after each step
            self.save_updated_config()
            return result
            
        def save_updated_config(self):
            """Save the current configuration to a YAML file in the output directory"""
            config_save_path = os.path.join(self.processed_dir, f"processing_config_{self.processing_mode}.yaml")
            # Create a new config with updated values but preserve the original structure
            updated_config = self.config.copy()
            # Keep the mode in the saved config
            if 'mode' not in updated_config and hasattr(self, 'original_config_path'):
                try:
                    with open(self.original_config_path, 'r') as file:
                        orig_config = yaml.safe_load(file)
                        if 'mode' in orig_config:
                            updated_config['mode'] = orig_config['mode']
                except Exception:
                    pass
            
            with open(config_save_path, 'w') as file:
                yaml.dump(updated_config, file, default_flow_style=False)
            print(f"Saved updated configuration to {config_save_path}")
    
    # Use the modified GUI manager
    gui_manager = ModifiedGUIManager(viewer, config, image_stack, file_loc, processing_mode, config_path)
    
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
    
    # Initialize button states
    update_navigation_buttons()

    napari.run()