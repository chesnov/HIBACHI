from magicgui import magicgui
from typing import Dict, Any
import os
import pandas as pd
import napari
import sys
import yaml
import traceback
import tifffile as tiff
import shutil
from PyQt5.QtGui import QCloseEvent
from PyQt5.QtCore import Qt, QObject, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QMessageBox,
    QMainWindow, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QWidget, QLabel, QInputDialog
)


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


def organize_processing_dir(drctry, mode):
    """
    A function to organize the directory with multiple samples for processing with 3D segmentation
    Inputs:
    - drctry: the directory containing tif files and a single csv file with the mapping of x,y,z dimensions
              to the corresponding tif files; csv file should have columns:
              Filename, Width (um), Height (um), Slices, Depth (um)
    - mode: 'nuclear' or 'ramified', determines which config template to use
    """
    print(f"Organizing directory: {drctry} for mode: {mode}")
    # Make a list of all the tif files in the directory
    try:
        all_files = os.listdir(drctry)
        tif_files = [f for f in all_files if f.lower().endswith('.tif') and os.path.isfile(os.path.join(drctry, f))]
        csv_files = [f for f in all_files if f.lower().endswith('.csv') and os.path.isfile(os.path.join(drctry, f))]
    except OSError as e:
        raise OSError(f"Error listing files in directory {drctry}: {e}") from e

    if not tif_files:
        raise ValueError('No .tif files found directly in the selected directory.')

    # Check that there is only one csv file in the directory
    if len(csv_files) != 1:
        raise ValueError(f'Expected exactly one .csv file in the directory, found {len(csv_files)}.')

    # Load the csv file
    csv_file = csv_files[0]
    csv_path = os.path.join(drctry, csv_file)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {csv_path}: {e}") from e

    # Check that the csv file has the correct columns
    required_cols = ['Filename', 'Width (um)', 'Height (um)', 'Slices', 'Depth (um)']
    if not all([col in df.columns for col in required_cols]):
        raise ValueError(f'The CSV file must have columns: {", ".join(required_cols)}')

    # Check that the CSV file lists all the found TIF files (matching base names)
    csv_filenames = set(df['Filename'].astype(str))
    tif_basenames = set(os.path.splitext(f)[0] for f in tif_files)

    if csv_filenames != tif_basenames:
        missing_in_csv = tif_basenames - csv_filenames
        missing_in_folder = csv_filenames - tif_basenames
        error_msg = "Mismatch between CSV filenames and TIF files found:"
        if missing_in_csv:
            error_msg += f"\n - TIF files not listed in CSV: {', '.join(missing_in_csv)}"
        if missing_in_folder:
            error_msg += f"\n - CSV filenames without matching TIF: {', '.join(missing_in_folder)}"
        raise ValueError(error_msg)

    # Determine paths for template config files (assuming they are alongside this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if mode == 'nuclear':
        config_template_name = 'nuclear_config.yaml'
    elif mode == 'ramified':
        config_template_name = 'ramified_config.yaml'
    else:
        raise ValueError(f"Invalid mode '{mode}' specified.")

    config_template_path = os.path.join(script_dir, config_template_name)
    if not os.path.exists(config_template_path):
        # Fallback: Check current working directory if not found alongside script
        config_template_path = os.path.join(os.getcwd(), config_template_name)
        if not os.path.exists(config_template_path):
           raise FileNotFoundError(f"Config template '{config_template_name}' not found in {script_dir} or {os.getcwd()}. Please ensure it exists.")


    # Create a new directory for each tif file and process
    root_names = [os.path.splitext(f)[0] for f in tif_files]
    print(f"Found TIF roots: {root_names}")

    for root_name in root_names:
        print(f"Processing: {root_name}")
        new_dir = os.path.join(drctry, root_name)
        try:
            os.makedirs(new_dir, exist_ok=True)
        except OSError as e:
             raise OSError(f"Error creating directory {new_dir}: {e}") from e

        # Move the tif file to the new directory
        tif_file = root_name + '.tif' # Reconstruct filename assuming .tif extension
        # Find the actual tif filename case-insensitively if needed, but stick to root_name.tif for now
        original_tif_path = os.path.join(drctry, tif_file)
        # Find the actual TIF file matching the root name (case-insensitive check might be better)
        actual_tif_file = next((f for f in tif_files if os.path.splitext(f)[0] == root_name), None)
        if not actual_tif_file:
             print(f"Warning: Could not find exact TIF file for root '{root_name}'. Skipping move/config.")
             continue # Should not happen due to earlier check, but safety first
        original_tif_path = os.path.join(drctry, actual_tif_file)
        new_tif_path = os.path.join(new_dir, actual_tif_file) # Use original filename in new dir

        try:
             print(f"Moving {original_tif_path} to {new_tif_path}")
             shutil.move(original_tif_path, new_tif_path)
        except Exception as e:
             raise OSError(f"Error moving file {original_tif_path} to {new_tif_path}: {e}") from e


        # Copy the correct yaml config file to the new directory and populate it
        new_config_path = os.path.join(new_dir, config_template_name) # Use template name for consistency
        try:
            print(f"Copying template {config_template_path} to {new_config_path}")
            shutil.copy2(config_template_path, new_config_path) # copy2 preserves metadata

            # Populate the yaml file with the correct dimensions and mode
            row = df[df['Filename'] == root_name]
            if row.empty:
                print(f"Warning: No data found in CSV for Filename '{root_name}'. Skipping config update.")
                continue

            x = row['Width (um)'].values[0]
            y = row['Height (um)'].values[0]
            z = row['Depth (um)'].values[0]

            # Read, update, and write the YAML safely
            with open(new_config_path, 'r') as f:
                config_data = yaml.safe_load(f) # Load existing structure

            # Ensure keys exist before assigning
            if 'voxel_dimensions' not in config_data or not isinstance(config_data['voxel_dimensions'], dict):
                config_data['voxel_dimensions'] = {}
            config_data['voxel_dimensions']['x'] = float(x) # Ensure float
            config_data['voxel_dimensions']['y'] = float(y) # Ensure float
            config_data['voxel_dimensions']['z'] = float(z) # Ensure float

            # Add the mode
            config_data['mode'] = mode

            with open(new_config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"Updated config {new_config_path} with dimensions and mode.")

        except Exception as e:
            raise RuntimeError(f"Error processing config file for {root_name}: {e}") from e

    print(f"Directory organization complete for {drctry}")




class ApplicationState(QObject):
    """Singleton to manage application state, including the main project window."""
    show_project_view_signal = pyqtSignal()

    _instance = None
    project_view_window = None  # Add attribute to hold the window instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ApplicationState, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            super().__init__()
            self.__initialized = True

# Create global instance
app_state = ApplicationState()

class ProjectManager:
    """
    Manages project-level operations for image segmentation
    """
    def __init__(self):
        self.project_path = None
        self.image_folders = []
    
    def select_project_folder(self):
        """
        Open file dialog to select project folder
        """
        self.project_path = QFileDialog.getExistingDirectory(
            None, 
            "Select Project Folder", 
            ""
        )
        
        if not self.project_path:
            return None
        
        # Validate and collect image folders
        self._find_valid_image_folders()
        return self.project_path
    
    def _find_valid_image_folders(self):
        """
        Find all subfolders that contain a valid TIFF image and YAML configuration
        """
        self.image_folders = []
        for root, dirs, files in os.walk(self.project_path):
            # Check for TIFF and YAML files
            tif_files = [f for f in files if f.endswith('.tif')]
            yaml_files = [f for f in files if f.endswith(('.yaml', '.yml'))]
            
            if tif_files and yaml_files:
                # Ensure only one TIFF and YAML file
                if len(tif_files) == 1 and len(yaml_files) == 1:
                    self.image_folders.append(root)
    
    def get_image_details(self, folder_path):
        """
        Get details of images in a specific folder
        """
        tif_file = [f for f in os.listdir(folder_path) if f.endswith('.tif')][0]
        yaml_file = [f for f in os.listdir(folder_path) if f.endswith(('.yaml', '.yml'))][0]
        
        # Load YAML to get processing mode
        with open(os.path.join(folder_path, yaml_file), 'r') as file:
            config = yaml.safe_load(file)
        
        return {
            'path': folder_path,
            'tif_file': tif_file,
            'yaml_file': yaml_file,
            'mode': config.get('mode', 'unknown')
        }

class ProjectViewWindow(QMainWindow):
    """
    Main window for project view with list of images.
    Handles application quit on close.
    """
    def __init__(self, project_manager):
        super().__init__()
        self.project_manager = project_manager
        self.initUI()
        # No need to explicitly set WA_DeleteOnClose unless you have specific reasons.
        # Overriding closeEvent is the cleaner way to handle shutdown logic.

    def initUI(self):
        self.setWindowTitle("Image Segmentation Project")
        self.setGeometry(100, 100, 600, 400)

        # Central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Project path display
        self.project_path_label = QLabel("Project Path: Not Selected")
        layout.addWidget(self.project_path_label)

        # Image list
        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self.open_image_view)
        layout.addWidget(self.image_list)

        # Buttons
        button_layout = QHBoxLayout()

        select_project_btn = QPushButton("Select Project Folder")
        select_project_btn.clicked.connect(self.load_project)
        button_layout.addWidget(select_project_btn)

        layout.addLayout(button_layout)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_project(self):
        """
        Load project: select folder, check if needs organizing, organize if needed,
        then populate image list.
        """
        selected_path = self.project_manager.select_project_folder() # Use PM just to get path

        if not selected_path:
            # User cancelled selection
            # Optionally clear list or keep previous state
            # self.image_list.clear()
            # self.project_path_label.setText("Project Path: Not Selected")
            return

        self.project_path_label.setText(f"Project Path: {selected_path}")
        self.image_list.clear() # Clear previous list

        try:
            # --- Check if the selected folder needs organizing ---
            needs_organizing = False
            try:
                # List contents safely
                root_contents = os.listdir(selected_path)
                root_files = [f for f in root_contents if os.path.isfile(os.path.join(selected_path, f))]
                root_dirs = [d for d in root_contents if os.path.isdir(os.path.join(selected_path, d))]

                root_tifs = [f for f in root_files if f.lower().endswith('.tif')]
                root_csvs = [f for f in root_files if f.lower().endswith('.csv')]

                if root_tifs and len(root_csvs) == 1:
                    # Potential candidate. Check if subdirs matching TIF names already exist.
                    tif_basenames = set(os.path.splitext(f)[0] for f in root_tifs)
                    existing_matching_dirs = tif_basenames.intersection(set(root_dirs))
                    if not existing_matching_dirs: # If no subdirs matching TIFs exist, it needs organizing
                         needs_organizing = True
                         print(f"Detected unorganized project structure in: {selected_path}")

            except OSError as e:
                 QMessageBox.critical(self, "Error Listing Directory", f"Could not read directory contents:\n{selected_path}\n{e}")
                 return # Stop loading

            # --- Organize if needed ---
            if needs_organizing:
                modes = ["nuclei", "ramified"]
                selected_mode, ok = QInputDialog.getItem(
                    self,
                    "Select Processing Mode",
                    "Choose the processing mode for this dataset:",
                    modes,
                    0,  # Default index
                    False # Not editable
                )

                if ok and selected_mode:
                    try:
                        organize_processing_dir(selected_path, selected_mode)
                        QMessageBox.information(self, "Organization Complete",
                                                f"Project folder organized successfully for '{selected_mode}' mode.")
                    except (ValueError, OSError, FileNotFoundError, RuntimeError) as e:
                        QMessageBox.critical(self, "Organization Failed",
                                             f"Failed to organize the project folder:\n{e}\n\nPlease check the folder contents, CSV file, and config templates.")
                        return # Stop loading if organization fails
                    except Exception as e:
                         QMessageBox.critical(self, "Unexpected Error During Organization",
                                             f"An unexpected error occurred:\n{e}\n\n{traceback.format_exc()}")
                         return
                else:
                    # User cancelled mode selection
                    QMessageBox.warning(self, "Organization Cancelled",
                                        "Project organization cancelled. Cannot load unorganized project.")
                    self.project_path_label.setText("Project Path: Not Selected") # Reset label
                    return # Stop loading

            # --- Load project (either pre-existing or just organized) ---
            # Now, update the project manager's path and find valid folders
            self.project_manager.project_path = selected_path # Set the path in the manager
            self.project_manager._find_valid_image_folders() # This looks for the organized structure

            if not self.project_manager.image_folders:
                 QMessageBox.warning(self, "No Valid Images", f"No valid image subfolders (containing one .tif and one .yaml) found in:\n{selected_path}")

            # Populate image list
            for folder in self.project_manager.image_folders:
                try:
                    image_details = self.project_manager.get_image_details(folder)
                    display_text = (
                        f"{os.path.basename(folder)} - "
                        f"Mode: {image_details.get('mode', 'unknown')}" # Use .get for safety
                    )
                    # Use QListWidgetItem for potential future features
                    item_widget = QListWidgetItem(display_text)
                    self.image_list.addItem(item_widget)
                except Exception as e:
                    print(f"Error processing folder {folder} for display: {e}")
                    item_widget = QListWidgetItem(f"{os.path.basename(folder)} - Error loading details")
                    self.image_list.addItem(item_widget)

        except Exception as e:
            # Catch-all for unexpected errors during loading/checking
            QMessageBox.critical(self, "Error Loading Project",
                                 f"An unexpected error occurred while loading the project:\n{e}\n\n{traceback.format_exc()}")
            self.image_list.clear()
            self.project_path_label.setText("Project Path: Error")

    def open_image_view(self, item):
        """
        Open the image view for the selected item, hiding the project view.
        """
        try:
            index = self.image_list.row(item)
            if index < 0 or index >= len(self.project_manager.image_folders):
                QMessageBox.warning(self, "Warning", "Invalid item selected.")
                return

            selected_folder = self.project_manager.image_folders[index]

            self.hide() # Hide the project view

            # Launch image-specific segmentation view
            interactive_segmentation_with_config(selected_folder)

        except Exception as e:
            # Detailed error logging
            error_msg = f"Error opening image view:\n{str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
            print(error_msg)
            self.show() # Reshow project view on error

    # --- Override closeEvent ---
    def closeEvent(self, event: QCloseEvent):
        """
        Overrides the default close event handler.
        Called when the user clicks the 'X' button or Alt+F4 etc.
        Ensures the entire Qt application terminates.
        """
        print("ProjectViewWindow closeEvent received. Quitting application.")
        # Get the QApplication instance and call quit() to stop the event loop
        app = QApplication.instance()
        if app:
            app.quit()
        # Accept the event to allow the window to be closed and destroyed.
        event.accept()
        # If you wanted to prevent closing (e.g., ask "Are you sure?"),
        # you would call event.ignore() instead of event.accept() based on user response.

def interactive_segmentation_with_config(selected_folder=None):
    from gui import DynamicGUIManager
    """
    Launch interactive segmentation with dynamic GUI
    
    Args:
        selected_folder (str, optional): Path to the specific image folder. 
                                         If None, prompt for file selection.
    """
    # Create PyQt5 app instance if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    try:
        # If no folder selected, prompt for input file
        if selected_folder is None:
            file_loc, _ = QFileDialog.getOpenFileName(
                None,
                "Select a .tif file",
                "",
                "TIFF files (*.tif);;All files (*.*)"
            )
            
            if not file_loc or not os.path.exists(file_loc):
                QMessageBox.warning(None, "Warning", "No file selected. Exiting.")
                return
            input_dir = os.path.dirname(file_loc)
        else:
            # Use the selected folder from project view
            input_dir = selected_folder
            file_loc = [
                os.path.join(input_dir, f) 
                for f in os.listdir(input_dir) 
                if f.endswith('.tif')
            ][0]
        
        # Find YAML file in the same directory as the TIF file
        yaml_files = [f for f in os.listdir(input_dir) if f.endswith(('.yaml', '.yml'))]
        
        if not yaml_files:
            QMessageBox.critical(None, "Error", "No YAML configuration file found in the directory.")
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
        
        # Load the .tif file
        try:
            image_stack = tiff.imread(file_loc)
            print(f"Loaded stack with shape {image_stack.shape}")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Failed to load TIFF file: {str(e)}")
            return

        # Initialize viewer and GUI manager with processing mode
        viewer = napari.Viewer(title=f"Microscopy Analysis - {processing_mode.capitalize()} Mode")
        
        # Add back to project button
        back_to_project_button = create_back_to_project_button(viewer)
        viewer.window.add_dock_widget(back_to_project_button, area="right")
        
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
        print(f"Processing image: {file_loc}")
        print(f"Processing mode: {processing_mode}")
        # Add your existing implementation here
            
    except Exception as e:
        # Detailed error logging
        error_msg = f"Error in interactive segmentation:\n{str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"
        QMessageBox.critical(None, "Error", error_msg)
        print(error_msg)

def launch_image_segmentation_tool():
    """
    Main entry point for the application. Manages the ProjectViewWindow instance.
    """
    try:
        # Get existing app or create new one
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)
            # Ensure the application doesn't quit when the last window (potentially Napari) is closed.
            # We will handle quitting explicitly when the ProjectViewWindow is closed.
            app.setQuitOnLastWindowClosed(False) 

        def show_or_create_project_view():
            if app_state.project_view_window is None:
                print("Creating new ProjectViewWindow")
                project_manager = ProjectManager()
                app_state.project_view_window = ProjectViewWindow(project_manager)
                # Optional: Connect the window's closed signal to app.quit
                app_state.project_view_window.setAttribute(Qt.WA_DeleteOnClose, False) # Prevent accidental deletion
                # If you want closing the project window to exit the app:
                # app_state.project_view_window.destroyed.connect(app.quit) 
                # Or handle via closeEvent if more complex logic needed.
                
                # Load project automatically if path exists or prompt
                # For simplicity, we'll rely on the button inside the window
                
            print("Showing ProjectViewWindow")
            app_state.project_view_window.show()
            app_state.project_view_window.activateWindow() # Bring to front
            app_state.project_view_window.raise_()         # Bring to front


        # Connect signal to handler *before* first show
        # Use a direct connection type if issues arise, but default should work
        app_state.show_project_view_signal.connect(show_or_create_project_view)

        # Show initial project view
        show_or_create_project_view()

        # Start the application event loop ONLY if this is the main script execution
        if __name__ == "__main__" and not hasattr(sys, 'ps1'): # Check if not in interactive shell
             print("Starting Qt Application event loop...")
             exit_code = app.exec_()
             print(f"Qt Application event loop finished with exit code: {exit_code}")
             sys.exit(exit_code)
        # else: # If imported, let the caller manage the event loop
        #    pass

    except Exception as e:
        # Detailed error logging
        error_msg = f"Unhandled error in launch:\n{str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"
        print(error_msg)
        # Ensure sys.exit is called even on error if we started the app
        if app and __name__ == "__main__" and not hasattr(sys, 'ps1'):
            sys.exit(1)
        elif not app: # If app creation failed
             sys.exit(1)

def create_back_to_project_button(viewer):
    """
    Create a button to return to the project view.

    Args:
        viewer (napari.Viewer): Current napari viewer
    """
    @magicgui(call_button="Back to Project")
    def back_to_project():
        """
        Close the current image view and emit signal to reopen the project view.
        """
        print("Back to Project button clicked.")
        try:
            # Close the viewer window. This should trigger Napari's cleanup.
            # Using viewer.window.qt_window.close() might be slightly more direct
            # for the Qt part, but viewer.close() is the documented way.
            viewer.close()
            print("Napari viewer closed.")

            # Short delay might help ensure viewer is fully closed before signal processing
            # but ideally not needed if event loops are handled correctly.
            # QTimer.singleShot(50, app_state.show_project_view_signal.emit) 

            # Emit signal to show project view - the connected slot will handle showing the window
            print("Emitting show_project_view_signal.")
            app_state.show_project_view_signal.emit()

        except Exception as e:
             error_msg = f"Error during 'Back to Project':\n{str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"
             QMessageBox.critical(None, "Error", error_msg)
             print(error_msg)
             # Fallback: Try to emit signal anyway if viewer closing failed but app still runs
             if QApplication.instance(): 
                 app_state.show_project_view_signal.emit()


    return back_to_project

if __name__ == "__main__":
    # Ensure required config templates exist before launching GUI
    script_dir = os.path.dirname(os.path.abspath(__file__))
    missing_templates = []
    for template in ['nuclear_config.yaml', 'ramified_config.yaml']:
        path1 = os.path.join(script_dir, template)
        path2 = os.path.join(os.getcwd(), template)
        if not os.path.exists(path1) and not os.path.exists(path2):
            missing_templates.append(template)

    if missing_templates:
        print(f"ERROR: Missing required configuration template(s): {', '.join(missing_templates)}")
        print(f"Please ensure these files are present in the script directory ({script_dir}) or the current working directory ({os.getcwd()}).")
        # Optionally show a GUI message box here if QApplication is already running,
        # but printing to console is safer before the app starts.
        sys.exit(1) # Exit if templates are missing

    launch_image_segmentation_tool()



    
    