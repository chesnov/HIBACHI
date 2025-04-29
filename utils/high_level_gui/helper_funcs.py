from magicgui import magicgui # type: ignore
from typing import Dict, Any, List
import os
import re
import pandas as pd
import napari # type: ignore
import sys
import yaml # type: ignore
import traceback
import tifffile as tiff # type: ignore
import shutil
from PyQt5.QtGui import QCloseEvent # type: ignore
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer # type: ignore
from PyQt5.QtWidgets import ( # type: ignore
    QApplication, QFileDialog, QMessageBox,
    QMainWindow, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QWidget, QLabel, QInputDialog
)

def natural_sort_key(s):
    """
    Create a sort key for natural alphanumeric sorting (e.g., image1, image2, image10).
    Operates on the basename of the input path string.
    """
    # Extract the directory/file name from the full path
    basename = os.path.basename(s)
    # Split the basename into alternating non-digit and digit parts
    # Convert digit parts to integers for correct numerical comparison
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', basename)]


def create_parameter_widget(param_name: str, param_config: Dict[str, Any], callback):
    """
    Create a widget for a single parameter, supporting int, float, bool, and list (via LineEdit).
    """
    param_type = param_config.get("type", "float") # Default type if missing
    label = param_config.get("label", param_name) # Default label

    widget = None # Initialize widget variable

    try:
        if param_type == "list":
            # Handle list type using a LineEdit for comma-separated floats
            initial_list = param_config.get("value", [])
            # Ensure initial value is a list
            if not isinstance(initial_list, list):
                print(f"Warning: Initial value for list parameter '{param_name}' is not a list ({type(initial_list)}). Using empty list.")
                initial_list = []

            # Convert list to initial string representation
            initial_str = ", ".join(map(str, initial_list))
            last_valid_list = list(initial_list) # Store a mutable copy

            # Define the inner widget function accepting a string
            def list_widget(value_str: str = initial_str):
                nonlocal last_valid_list # Allow modification of the outer scope variable
                new_list = None
                try:
                    # Attempt to parse the string into a list of floats
                    cleaned_str = value_str.strip()
                    if not cleaned_str: # Handle empty string -> empty list
                        new_list = []
                    else:
                        # Split by comma, strip whitespace from each item, filter out empty strings, convert to float
                        new_list = [float(x.strip()) for x in cleaned_str.split(',') if x.strip()]

                    # --- Optional: Add validation for list elements ---
                    # e.g., check element count, type, range based on other config keys
                    # element_min = param_config.get("element_min")
                    # if element_min is not None and any(x < element_min for x in new_list):
                    #    raise ValueError(f"All scale values must be >= {element_min}")
                    # --- End Optional Validation ---

                    # If parsing and validation succeeded:
                    print(f"Parsed list for {param_name}: {new_list}") # Debugging
                    if new_list != last_valid_list: # Only call back if value changed
                         callback(new_list) # Call the original callback with the list
                         last_valid_list[:] = new_list # Update last valid list state correctly
                    # Reset background color on success if error was shown previously
                    if hasattr(list_widget, 'native') and list_widget.native:
                         list_widget.native.setStyleSheet("")
                    return value_str # Return the input string to keep it in the widget

                except ValueError as e:
                    # Handle parsing errors (e.g., non-float input)
                    print(f"Warning: Invalid list format for {param_name}: '{value_str}'. Error: {e}.")
                    print(f"         Keeping last valid value: {last_valid_list}")
                    # Indicate error visually (optional)
                    if hasattr(list_widget, 'native') and list_widget.native:
                        list_widget.native.setStyleSheet("background-color: #FFDDDD;") # Light red background
                    # Do NOT call the main callback
                    # Return the *string representation of the last valid list* to revert the widget
                    return ", ".join(map(str, last_valid_list))

            # Create the magicgui widget as a LineEdit
            widget = magicgui(
                list_widget,
                auto_call=True, # Call on change
                value_str={ # Parameter name in inner function must match key here
                    "widget_type": "LineEdit",
                    "label": label, # Use label from config
                }
            )

        elif param_type == "float":
            # Handle float type (Slider or SpinBox)
            default_value = float(param_config.get("value", 0.0)) # Ensure float
            min_val = float(param_config.get("min", 0.0))
            max_val = float(param_config.get("max", 100.0))
            step_val = float(param_config.get("step", 0.1))
            # Ensure min <= default <= max
            default_value = max(min_val, min(default_value, max_val))

            def float_widget(value: float = default_value):
                 callback(value)
                 return value
            widget = magicgui(
                 float_widget, auto_call=True,
                 value={"widget_type": "FloatSpinBox", # SpinBox might be better for floats
                        "label": label,
                        "min": min_val,
                        "max": max_val,
                        "step": step_val}
            )

        elif param_type == "int":
            # Handle int type (Slider or SpinBox)
            default_value = int(param_config.get("value", 0)) # Ensure int
            min_val = int(param_config.get("min", 0))
            max_val = int(param_config.get("max", 100))
            step_val = int(param_config.get("step", 1))
            # Ensure min <= default <= max
            default_value = max(min_val, min(default_value, max_val))

            def int_widget(value: int = default_value):
                 callback(value)
                 return value
            widget = magicgui(
                 int_widget, auto_call=True,
                 value={"widget_type": "SpinBox", # Use SpinBox for better precision control
                        "label": label,
                        "min": min_val,
                        "max": max_val,
                        "step": step_val}
            )

        elif param_type == "bool":
             # Handle boolean type
             default_value = bool(param_config.get("value", False)) # Ensure bool
             def bool_widget(value: bool = default_value):
                  callback(value)
                  return value
             widget = magicgui(
                 bool_widget, auto_call=True,
                 value={"widget_type": "CheckBox",
                        "label": label}
             )
        else:
            # Fallback for unsupported types - create a simple LineEdit displaying str(value)
            print(f"Warning: Unsupported parameter type '{param_type}' for '{param_name}'. Creating basic LineEdit.")
            default_value_str = str(param_config.get("value", "")) # Convert value to string
            last_valid_str = default_value_str

            def fallback_widget(value: str = default_value_str):
                nonlocal last_valid_str
                # Here, we don't know the target type, so we pass the string directly.
                # The receiving function might need to handle conversion.
                if value != last_valid_str:
                     callback(value) # Pass the raw string value
                     last_valid_str = value
                return value # Keep the string in the widget

            widget = magicgui(
                fallback_widget, auto_call=True,
                value={"widget_type": "LineEdit",
                       "label": label}
            )

        # Store the original parameter name as an attribute on the magicgui instance
        if widget:
            widget.param_name = param_name

    except Exception as e:
         print(f"ERROR creating widget for parameter '{param_name}': {e}")
         print(f"Config: {param_config}")
         print(traceback.format_exc())
         # Optionally create a disabled label indicating the error
         # error_label = magicgui(lambda: None, labels=False, auto_call=False)
         # error_label.native.setText(f"Error loading '{label}'")
         # error_label.native.setEnabled(False)
         # return error_label # Or return None, or raise error
         return None # Return None if widget creation failed

    return widget



def check_processing_state(processed_dir: str, mode: str, checkpoint_files: dict, num_steps: int) -> int:
    """
    Checks the processing state by looking for expected output files.

    Args:
        processed_dir: The directory where processed files are stored.
        mode: The current processing mode ('nuclei' or 'ramified').
        checkpoint_files: Dictionary mapping logical file keys to output file paths
                         (obtained from strategy.get_checkpoint_files()).
        num_steps: The total number of steps expected for this mode.

    Returns:
        int: The number of the last successfully completed step (0 if none).
             Returns num_steps if the final step's output exists.
    """
    if not os.path.isdir(processed_dir):
        print(f"Processed directory not found: {processed_dir}. Starting from step 0.")
        return 0

    print(f"Checking processing state in: {processed_dir} (Mode: {mode}, Steps: {num_steps})")
    print(f"  Available checkpoint keys: {list(checkpoint_files.keys())}") # Debug

    # --- Define key output files that signify step completion ---
    # These should map step number (1-based) to the KEY in checkpoint_files dict
    completion_file_keys = {}
    if mode == 'ramified':
        # --- THIS IS THE CRITICAL CHANGE ---
        completion_file_keys = {
            1: "raw_segmentation",       # Step 1 output KEY
            2: "trimmed_segmentation",   # Step 2 output KEY
            3: "final_segmentation",     # Step 3 output KEY (Main labeled result)
            4: "metrics_df"              # Step 4 output KEY (or skeleton_array etc.)
        }
        # --- END CRITICAL CHANGE ---
    elif mode == 'nuclei':
         completion_file_keys = {
            1: "initial_segmentation",   # Step 1 output KEY
            2: "final_segmentation",     # Step 2 output KEY
            3: "metrics_df"              # Step 3 output KEY
        }
    else:
        print(f"Warning: Unknown mode '{mode}' in check_processing_state.")
        return 0

    last_completed_step = 0
    for step in range(1, num_steps + 1):
        file_key = completion_file_keys.get(step) # Get the KEY for this step
        if not file_key:
            print(f"Warning: No completion file key defined for step {step} in mode '{mode}'. Cannot check.")
            break # Cannot check further

        file_to_check = checkpoint_files.get(file_key) # Get the PATH using the key

        if file_to_check and os.path.exists(file_to_check):
            last_completed_step = step
            print(f"  Found output for step {step} (Key: '{file_key}'): {os.path.basename(file_to_check)}")
        else:
            # If a step's output is missing, stop checking further steps
            print(f"  Output for step {step} (Key: '{file_key}', Path: {file_to_check}) not found.")
            break # Exit the loop early

    print(f"Determined last completed step: {last_completed_step}")
    return last_completed_step


# --- organize_processing_dir (Largely Unchanged) ---
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
        # Robustly filter TIF/TIFF extensions, case-insensitive
        tif_files = [f for f in all_files if f.lower().endswith(('.tif', '.tiff')) and os.path.isfile(os.path.join(drctry, f))]
        csv_files = [f for f in all_files if f.lower().endswith('.csv') and os.path.isfile(os.path.join(drctry, f))]
    except OSError as e:
        raise OSError(f"Error listing files in directory {drctry}: {e}") from e

    if not tif_files:
        # Check subdirectories too? No, instruction says 'directly in the selected directory'.
        raise ValueError('No .tif or .tiff files found directly in the selected directory.')

    # Check that there is only one csv file in the directory
    if len(csv_files) != 1:
        raise ValueError(f'Expected exactly one .csv file in the directory, found {len(csv_files)}: {", ".join(csv_files)}')

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
        raise ValueError(f'The CSV file must have columns: {", ".join(required_cols)}. Found: {", ".join(df.columns)}')

    # Check that the CSV file lists all the found TIF files (matching base names)
    # Ensure comparison is robust (e.g., string type)
    csv_filenames = set(df['Filename'].astype(str))
    tif_basenames = set(os.path.splitext(f)[0] for f in tif_files)

    if csv_filenames != tif_basenames:
        missing_in_csv = tif_basenames - csv_filenames
        missing_in_folder = csv_filenames - tif_basenames
        error_msg = "Mismatch between CSV 'Filename' column (without extension) and TIF/TIFF files found:"
        if missing_in_csv:
            error_msg += f"\n - TIF files not listed in CSV: {', '.join(missing_in_csv)}"
        if missing_in_folder:
            error_msg += f"\n - CSV filenames without matching TIF: {', '.join(missing_in_folder)}"
        raise ValueError(error_msg)

    # Determine paths for template config files (assuming they are alongside this script or CWD)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if mode == 'nuclear':
        config_template_name = os.path.join('nuclear_module_3d','nuclear_config.yaml')
    elif mode == 'ramified':
        config_template_name = os.path.join('ramified_module_3d', 'ramified_config.yaml')
    else:
        raise ValueError(f"Invalid mode '{mode}' specified.")

    config_template_path = os.path.join(script_dir, os.path.join('..', config_template_name))
    if not os.path.exists(config_template_path):
        # Fallback: Check current working directory if not found alongside script
        print(f"Template not found in {config_template_path}, checking CWD: {os.getcwd()}")
        config_template_path = os.path.join(os.getcwd(), config_template_name)
        if not os.path.exists(config_template_path):
           raise FileNotFoundError(f"Config template '{config_template_name}' not found in script directory ({script_dir}) or current working directory ({os.getcwd()}). Please ensure it exists.")
    print(f"Using config template: {config_template_path}")

    # Create a new directory for each tif file and process
    root_names = list(tif_basenames) # Use the set derived from actual files
    print(f"Found TIF roots: {root_names}")

    for root_name in root_names:
        print(f"Processing: {root_name}")
        new_dir = os.path.join(drctry, root_name)
        try:
            os.makedirs(new_dir, exist_ok=True)
        except OSError as e:
             raise OSError(f"Error creating directory {new_dir}: {e}") from e

        # Find the actual TIF/TIFF file matching the root name (case-insensitive check might be better if needed)
        actual_tif_file = next((f for f in tif_files if os.path.splitext(f)[0] == root_name), None)
        if not actual_tif_file:
             print(f"Critical Warning: Could not find exact TIF file for root '{root_name}' during processing loop. Skipping.")
             continue # Should not happen due to earlier check, but safety first

        original_tif_path = os.path.join(drctry, actual_tif_file)
        new_tif_path = os.path.join(new_dir, actual_tif_file) # Use original filename in new dir

        # Avoid moving if source and destination are the same (e.g., if rerun)
        if os.path.abspath(original_tif_path) == os.path.abspath(new_tif_path):
            print(f"File {actual_tif_file} is already in the target directory {new_dir}. Skipping move.")
        elif not os.path.exists(original_tif_path):
             print(f"Warning: Source file {original_tif_path} does not exist (might have been moved already?). Skipping move.")
             # Check if it exists in the new location already
             if not os.path.exists(new_tif_path):
                  raise FileNotFoundError(f"Critical: TIF file {actual_tif_file} missing from both source and destination.")
        else:
             try:
                 print(f"Moving {original_tif_path} to {new_tif_path}")
                 shutil.move(original_tif_path, new_tif_path)
             except Exception as e:
                 # Check if destination already exists (e.g., if interrupted and rerun)
                 if os.path.exists(new_tif_path):
                      print(f"Warning: Destination {new_tif_path} already exists. Assuming move was completed earlier. Skipping move.")
                 else:
                      raise OSError(f"Error moving file {original_tif_path} to {new_tif_path}: {e}") from e


        # Copy the correct yaml config file to the new directory and populate it
        # Always use the template name for the copied file for consistency
        config_filename = os.path.basename(config_template_name) # Get just the filename e.g., 'ramified_config.yaml'
        new_config_path = os.path.join(new_dir, config_filename) # Construct path like '.../N/ramified_config.yaml'
        try:
            # Copy template only if config doesn't exist or explicitly overwrite
            if not os.path.exists(new_config_path):
                 print(f"Copying template {config_template_path} to {new_config_path}")
                 shutil.copy2(config_template_path, new_config_path) # copy2 preserves metadata
            else:
                 print(f"Config file {new_config_path} already exists. Skipping copy, will update existing.")

            # Populate the yaml file with the correct dimensions and mode
            # Match filename case-insensitively in DataFrame if necessary, but CSV should match root_name
            row = df[df['Filename'].astype(str) == root_name]
            if row.empty:
                print(f"Warning: No data found in CSV for Filename '{root_name}'. Skipping config update.")
                continue

            # Safely get values, handle potential multiple matches (take first)
            x = row['Width (um)'].iloc[0]
            y = row['Height (um)'].iloc[0]
            z = row['Depth (um)'].iloc[0]

            # Read, update, and write the YAML safely
            config_data = {}
            if os.path.exists(new_config_path):
                try:
                    with open(new_config_path, 'r') as f:
                        config_data = yaml.safe_load(f) # Load existing structure
                        if config_data is None: # Handle empty YAML file
                             config_data = {}
                except yaml.YAMLError as ye:
                    print(f"Warning: Could not parse existing YAML {new_config_path}: {ye}. Will create from scratch.")
                    config_data = {} # Reset if unparseable
                except Exception as e:
                    print(f"Warning: Error reading existing YAML {new_config_path}: {e}. Will create from scratch.")
                    config_data = {}


            # Ensure keys exist before assigning, create if necessary
            if 'voxel_dimensions' not in config_data or not isinstance(config_data.get('voxel_dimensions'), dict):
                config_data['voxel_dimensions'] = {}
            config_data['voxel_dimensions']['x'] = float(x) # Ensure float
            config_data['voxel_dimensions']['y'] = float(y) # Ensure float
            config_data['voxel_dimensions']['z'] = float(z) # Ensure float

            # Add/Update the mode
            config_data['mode'] = mode

            with open(new_config_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"Updated config {new_config_path} with dimensions (X:{x}, Y:{y}, Z:{z}) and mode '{mode}'.")

        except Exception as e:
            raise RuntimeError(f"Error processing config file for {root_name}: {e}\n{traceback.format_exc()}") from e

    print(f"Directory organization complete for {drctry}")


# --- ApplicationState (Unchanged) ---
class ApplicationState(QObject):
    """Singleton to manage application state, including the main project window."""
    show_project_view_signal = pyqtSignal()

    _instance = None
    project_view_window = None  # Holds the single ProjectViewWindow instance

    def __new__(cls):
        if cls._instance is None:
            print("Creating ApplicationState instance")
            cls._instance = super(ApplicationState, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not getattr(self, '_ApplicationState__initialized', False): # Name mangling check
            super().__init__()
            self.__initialized = True
            print("Initializing ApplicationState")

# Create global instance
app_state = ApplicationState()

# --- ProjectManager (Unchanged) ---
class ProjectManager:
    """
    Manages project-level operations for image segmentation
    """
    def __init__(self):
        self.project_path = None
        self.image_folders = [] # List of full paths to valid image subdirectories

    def select_project_folder(self):
        """
        Open file dialog to select project folder
        """
        # Use parent=None if no specific parent window is needed
        selected_path = QFileDialog.getExistingDirectory(
            None,
            "Select Project Root Folder (containing organized image subfolders or files to be organized)",
            "" # Start directory (optional)
        )

        if not selected_path:
            print("Project folder selection cancelled.")
            return None

        self.project_path = selected_path
        print(f"Project folder selected: {self.project_path}")

        # Re-scan for valid folders after selection (important if organizing happened)
        self._find_valid_image_folders()
        return self.project_path

    def _find_valid_image_folders(self):
        """
        Find all direct subfolders of self.project_path that contain exactly
        one valid TIFF image and exactly one YAML configuration file.
        """
        self.image_folders = []
        if not self.project_path or not os.path.isdir(self.project_path):
            print("Project path not set or invalid, cannot find image folders.")
            return

        print(f"Scanning for valid image folders in: {self.project_path}")
        try:
            for item in os.listdir(self.project_path):
                potential_folder_path = os.path.join(self.project_path, item)
                if os.path.isdir(potential_folder_path):
                    # Check contents of this subdirectory
                    try:
                        folder_contents = os.listdir(potential_folder_path)
                        tif_files = [f for f in folder_contents if f.lower().endswith(('.tif', '.tiff')) and os.path.isfile(os.path.join(potential_folder_path, f))]
                        yaml_files = [f for f in folder_contents if f.lower().endswith(('.yaml', '.yml')) and os.path.isfile(os.path.join(potential_folder_path, f))]

                        # Check for exactly one of each required file type
                        if len(tif_files) == 1 and len(yaml_files) == 1:
                            print(f"  Found valid image folder: {item}")
                            self.image_folders.append(potential_folder_path)
                        # else:
                        #     if len(tif_files) > 1 or len(yaml_files) > 1:
                        #          print(f"  Skipping folder {item}: Found {len(tif_files)} TIFs, {len(yaml_files)} YAMLs (expected 1 each).")
                        #     elif not tif_files or not yaml_files:
                        #          print(f"  Skipping folder {item}: Missing TIF or YAML file.")

                    except OSError as e:
                        print(f"Warning: Could not read subdirectory {potential_folder_path}: {e}")
                        continue # Skip this subdirectory

        except OSError as e:
            print(f"Error listing contents of project path {self.project_path}: {e}")
            self.image_folders = [] # Reset on error

        self.image_folders.sort(key=natural_sort_key)

        print(f"Found {len(self.image_folders)} valid image folders.")


    def get_image_details(self, folder_path):
        """
        Get details (tif, yaml, mode) of image in a specific folder.
        Assumes folder_path is a valid image folder found by _find_valid_image_folders.
        """
        try:
            contents = os.listdir(folder_path)
            tif_file = next((f for f in contents if f.lower().endswith(('.tif', '.tiff'))), None)
            yaml_file = next((f for f in contents if f.lower().endswith(('.yaml', '.yml'))), None)

            if not tif_file or not yaml_file:
                raise FileNotFoundError(f"Required TIF/YAML file not found in {folder_path}")

            # Load YAML to get processing mode
            config = {}
            yaml_path = os.path.join(folder_path, yaml_file)
            try:
                with open(yaml_path, 'r') as file:
                    config = yaml.safe_load(file)
                    if config is None: config = {} # Handle empty file
            except yaml.YAMLError as ye:
                 print(f"Warning: Could not parse YAML {yaml_path}: {ye}")
                 config = {} # Treat as empty
            except Exception as e:
                print(f"Warning: Error reading YAML {yaml_path}: {e}")
                config = {} # Treat as empty


            return {
                'path': folder_path,
                'tif_file': tif_file,   # Just the filename
                'yaml_file': yaml_file, # Just the filename
                'mode': config.get('mode', 'unknown') # Default if mode key is missing
            }
        except Exception as e:
             print(f"Error getting image details for {folder_path}: {e}")
             # Return partial or error state
             return {
                 'path': folder_path,
                 'tif_file': 'Error',
                 'yaml_file': 'Error',
                 'mode': 'error'
             }

# --- ProjectViewWindow (Unchanged UI, logic for organization integrated) ---
class ProjectViewWindow(QMainWindow):
    """
    Main window for project view with list of images.
    Handles application quit on close.
    """
    def __init__(self, project_manager):
        super().__init__()
        self.project_manager = project_manager
        # Store the napari viewer instance if needed, maybe passed or created here
        # self.viewer = viewer # If managing a single viewer instance
        self.initUI()
        # Ensure this window closing quits the app if it's the main control window
        self.setAttribute(Qt.WA_QuitOnClose) # More direct than overriding closeEvent for simple quit

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

        select_project_btn = QPushButton("Select/Load Project Folder") # Changed label slightly
        select_project_btn.clicked.connect(self.load_project)
        button_layout.addWidget(select_project_btn)

        # Add other buttons if needed (e.g., "Organize Current Folder")

        layout.addLayout(button_layout)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def load_project(self):
        """
        Selects a project folder, checks if it needs organizing, organizes if necessary,
        and populates the image list with valid subfolders.
        """
        selected_path = self.project_manager.select_project_folder() # Uses PM to get/set path

        if not selected_path:
            return # User cancelled

        self.project_path_label.setText(f"Project Path: {selected_path}")
        self.image_list.clear() # Clear previous list

        try:
            # --- Check if the selected folder needs organizing ---
            # This logic checks for loose TIFs and a single CSV at the root
            needs_organizing = False
            root_tifs = []
            root_csvs = []
            root_dirs = []
            try:
                root_contents = os.listdir(selected_path)
                for item in root_contents:
                    item_path = os.path.join(selected_path, item)
                    if os.path.isfile(item_path):
                         if item.lower().endswith(('.tif', '.tiff')):
                             root_tifs.append(item)
                         elif item.lower().endswith('.csv'):
                             root_csvs.append(item)
                    elif os.path.isdir(item_path):
                         root_dirs.append(item)

                if root_tifs and len(root_csvs) == 1:
                    # Potential candidate. Check if subdirs matching TIF names already exist.
                    tif_basenames = set(os.path.splitext(f)[0] for f in root_tifs)
                    existing_matching_dirs = tif_basenames.intersection(set(root_dirs))
                    # If TIFs exist, CSV exists, AND no subdirs matching the TIFs exist, then organize.
                    if not existing_matching_dirs:
                         needs_organizing = True
                         print(f"Detected unorganized project structure in: {selected_path}")
                    # else:
                    #      print("Structure looks organized or partially organized. Will load existing subfolders.")

            except OSError as e:
                 QMessageBox.critical(self, "Error Listing Directory", f"Could not read directory contents to check organization:\n{selected_path}\n{e}")
                 return # Stop loading

            # --- Organize if needed ---
            if needs_organizing:
                reply = QMessageBox.question(self, "Organize Project?",
                                             f"Unorganized structure detected (TIFs and one CSV found at root level: {selected_path}).\n\n"
                                             "Do you want to organize this folder now?\n"
                                             "(This will move TIFs into subfolders and create config files).",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)

                if reply == QMessageBox.Yes:
                    modes = ["nuclei", "ramified"]
                    selected_mode, ok = QInputDialog.getItem(
                        self, "Select Processing Mode", "Choose the mode for organizing this dataset:",
                        modes, 0, False)

                    if ok and selected_mode:
                        try:
                            organize_processing_dir(selected_path, selected_mode)
                            QMessageBox.information(self, "Organization Complete",
                                                    f"Project folder organized successfully for '{selected_mode}' mode.")
                            # Re-scan project after organizing
                            self.project_manager._find_valid_image_folders()

                        except Exception as e:
                            QMessageBox.critical(self, "Organization Failed",
                                                 f"Failed to organize the project folder:\n{e}\n\n{traceback.format_exc()}")
                            # Decide whether to proceed or stop
                            # Allowing to proceed might show an empty list if organization failed partially
                            # Let's attempt to load whatever might be valid anyway.
                            self.project_manager._find_valid_image_folders() # Scan again
                    else:
                        # User cancelled mode selection
                        QMessageBox.warning(self, "Organization Cancelled",
                                            "Project organization cancelled by user.")
                        # Proceed to load whatever might be valid, even if unorganized
                        self.project_manager._find_valid_image_folders() # Scan again
                else:
                     # User chose not to organize
                     QMessageBox.information(self, "Organization Skipped",
                                         "Organization skipped. Loading existing valid subfolders only.")
                     # Scan for existing valid folders
                     self.project_manager._find_valid_image_folders() # Scan again

            # --- Populate list (using folders found by ProjectManager) ---
            if not self.project_manager.image_folders:
                 if not needs_organizing: # Only show warning if it wasn't expected to be empty
                     QMessageBox.warning(self, "No Valid Images Found",
                                         f"No valid image subfolders (containing one .tif/.tiff and one .yaml/.yml) found in:\n{selected_path}")
                 # List remains empty

            # Populate image list
            for folder_path in self.project_manager.image_folders:
                try:
                    details = self.project_manager.get_image_details(folder_path)
                    if details.get('mode') == 'error':
                         display_text = f"{os.path.basename(folder_path)} - Error loading details"
                    else:
                         display_text = (
                             f"{os.path.basename(folder_path)} - "
                             f"Mode: {details.get('mode', 'unknown')}"
                         )
                    item_widget = QListWidgetItem(display_text)
                    # Store the full path in the item's data for retrieval
                    item_widget.setData(Qt.UserRole, folder_path)
                    self.image_list.addItem(item_widget)
                except Exception as e:
                    print(f"Error processing folder {folder_path} for display list: {e}")
                    # Add an error item to the list
                    error_item = QListWidgetItem(f"{os.path.basename(folder_path)} - Error processing")
                    error_item.setData(Qt.UserRole, folder_path) # Still store path if possible
                    self.image_list.addItem(error_item)

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Project",
                                 f"An unexpected error occurred:\n{e}\n\n{traceback.format_exc()}")
            self.image_list.clear()
            self.project_path_label.setText("Project Path: Error")


    def open_image_view(self, item):
        """
        Open the image view for the selected item, hiding the project view.
        Retrieves folder path from item data.
        """
        try:
            selected_folder = item.data(Qt.UserRole) # Retrieve the stored path
            if not selected_folder or not os.path.isdir(selected_folder):
                QMessageBox.warning(self, "Error", f"Invalid folder path associated with selected item:\n{selected_folder}")
                return

            print(f"Opening image view for: {selected_folder}")
            self.hide() # Hide the project view

            # Launch image-specific segmentation view
            # This function will create its own Napari instance and run it.
            interactive_segmentation_with_config(selected_folder)

            # Note: After interactive_segmentation_with_config finishes (Napari closes),
            # control returns here. The ProjectView might need to be reshown if the
            # back button wasn't used, or handled by the ApplicationState signal.
            # The 'Back to Project' button is the cleaner way to return.

        except Exception as e:
            error_msg = f"Error opening image view from Project View:\n{str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Error", error_msg)
            print(error_msg)
            # Attempt to reshow project view on error, though state might be uncertain
            if not self.isVisible():
                self.show()


    # --- Override closeEvent (Optional but recommended for clean exit) ---
    def closeEvent(self, event: QCloseEvent):
        """
        Overrides the default close event handler.
        Ensures the entire Qt application terminates cleanly.
        """
        print("ProjectViewWindow closeEvent received. Quitting application.")
        reply = QMessageBox.question(self, 'Confirm Exit',
                                     "Are you sure you want to exit the application?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Clean up resources if necessary
            # e.g., close open files, stop threads

            # Get the QApplication instance and call quit()
            app = QApplication.instance()
            if app:
                app.quit() # Stops the event loop
            event.accept() # Allow the window to close
        else:
            event.ignore() # Prevent the window from closing


def _check_if_last_window():
    """Checks if the application should quit after a window closed."""
    app = QApplication.instance()
    if not app:
        print("CheckIfLastWindow: No App instance.")
        return # No application anymore

    project_window = app_state.project_view_window
    project_view_exists = project_window is not None
    # We need to check if the C++ object still exists too
    project_view_valid = False
    if project_view_exists:
        try:
            # Check if underlying Qt widget is still valid
            project_view_valid = project_window.isVisible() or not project_window.isHidden()
        except RuntimeError: # C++ object deleted
            print("CheckIfLastWindow: Project window C++ object deleted.")
            project_view_exists = False # Treat as non-existent
            project_view_valid = False


    # Get all potentially relevant top-level windows
    active_windows = app.topLevelWidgets()

    # Filter out the project window itself if it exists and any non-visible widgets
    other_visible_task_windows = [
        w for w in active_windows
        if w is not project_window and w.isVisible()
    ]

    # If the project view doesn't exist OR it's not valid/visible,
    # AND there are no other VISIBLE top-level windows, then quit.
    if (not project_view_exists or not project_view_valid) and not other_visible_task_windows:
         print("CheckIfLastWindow: Project View gone/hidden and no other visible windows found. Quitting.")
         app.quit()
    # If the project view IS visible, we never quit here (closing Napari goes back to project view)
    elif project_view_exists and project_view_valid and project_window.isVisible():
         print("CheckIfLastWindow: Project View is visible. Not quitting.")
    # If project view is hidden, but other windows ARE visible, don't quit
    elif project_view_exists and project_view_valid and not project_window.isVisible() and other_visible_task_windows:
         print(f"CheckIfLastWindow: Project View hidden, but {len(other_visible_task_windows)} other window(s) visible. Not quitting.")
    else:
         # Catch-all, includes case where project view is hidden and no other windows visible
         if not project_view_exists or not project_view_valid: # Double check the condition for quit
              if not other_visible_task_windows:
                   print("CheckIfLastWindow: Fallback check - Project View gone/hidden, no other visible windows. Quitting.")
                   app.quit()
              else:
                   print("CheckIfLastWindow: Fallback check - Project View gone/hidden, but other windows visible. Not quitting.")
         else: # Project view exists and is hidden
              if not other_visible_task_windows:
                   print("CheckIfLastWindow: Project view hidden, no other visible windows. Quitting.")
                   app.quit()
              # This else case should be covered above
              # else:
              #      print("CheckIfLastWindow: Project view hidden, but other windows visible. Not quitting.")


def _handle_napari_close():
    """Called when a Napari window's destroyed signal is emitted."""
    print("Napari window destroyed signal received.")
    # Use QTimer.singleShot to delay the check slightly.
    # This allows Qt's event loop to process the window closure fully
    # before we check the list of topLevelWidgets. 100ms is usually safe.
    QTimer.singleShot(100, _check_if_last_window)



# --- interactive_segmentation_with_config (MODIFIED - Removed ModifiedGUIManager) ---
def interactive_segmentation_with_config(selected_folder=None):
    """
    Launch interactive segmentation with dynamic GUI for a specific image folder.

    Args:
        selected_folder (str): Path to the specific image folder containing TIF and YAML.
                               Must be provided.
    """
    # --- Import DynamicGUIManager inside the function to avoid circular imports ---
    try:
        from .gui_manager import DynamicGUIManager
    except ImportError as e:
        print(f"Error importing DynamicGUIManager inside function: {e}")
        QMessageBox.critical(None, "Import Error", "Could not load core GUI component. Check installation/paths.")
        if QApplication.instance() and app_state:
             app_state.show_project_view_signal.emit() # Try to show project view on error
        return # Cannot proceed

    # Get existing app or create if necessary
    app = QApplication.instance()
    if app is None:
        print("Warning: No QApplication instance found, creating one.")
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False) # Important for multi-window apps

    viewer = None # Initialize viewer to None for cleanup/error handling
    try:
        # --- Input Validation ---
        if selected_folder is None or not os.path.isdir(selected_folder):
             raise ValueError("No valid image folder provided.")
        input_dir = selected_folder
        print(f"Starting interactive segmentation for folder: {input_dir}")

        # --- Find TIF and YAML files ---
        try:
             contents = os.listdir(input_dir)
             tif_files = [f for f in contents if f.lower().endswith(('.tif', '.tiff'))]
             yaml_files = [f for f in contents if f.lower().endswith(('.yaml', '.yml'))]

             if len(tif_files) != 1:
                 raise FileNotFoundError(f"Expected 1 TIF/TIFF file, found {len(tif_files)} in {input_dir}")
             if len(yaml_files) != 1:
                 raise FileNotFoundError(f"Expected 1 YAML/YML file, found {len(yaml_files)} in {input_dir}")

             file_loc = os.path.join(input_dir, tif_files[0])
             config_path = os.path.join(input_dir, yaml_files[0])
        except (FileNotFoundError, OSError) as e:
            # Show error and attempt to return to project view
            QMessageBox.critical(None, "File Error", f"Error finding required files in {input_dir}:\n{e}")
            if QApplication.instance() and app_state: app_state.show_project_view_signal.emit()
            return

        # --- Load Config ---
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            if config is None:
                 raise ValueError("YAML file is empty or invalid.")
        except Exception as e:
            QMessageBox.critical(None, "Config Error", f"Failed to load or parse YAML file: {config_path}\n{e}")
            if QApplication.instance() and app_state: app_state.show_project_view_signal.emit()
            return

        # --- Extract and Validate Processing Mode ---
        processing_mode = config.get('mode')
        if not processing_mode:
            QMessageBox.critical(None, "Config Error", f"The YAML file ({config_path}) must contain a 'mode' field ('nuclei' or 'ramified').")
            if QApplication.instance() and app_state: app_state.show_project_view_signal.emit()
            return
        if processing_mode not in ["nuclei", "ramified"]:
            QMessageBox.critical(None, "Config Error", f"Invalid processing mode '{processing_mode}' in YAML file. Must be 'nuclei' or 'ramified'.")
            if QApplication.instance() and app_state: app_state.show_project_view_signal.emit()
            return

        # --- Load Image Stack ---
        try:
            print(f"Loading image stack: {file_loc}")
            image_stack = tiff.imread(file_loc)
            print(f"Loaded stack with shape {image_stack.shape}, dtype {image_stack.dtype}")
            if image_stack.ndim != 3:
                 print(f"Warning: Expected 3D image stack, but got {image_stack.ndim} dimensions.")
            if image_stack.size == 0:
                 raise ValueError("Image stack is empty.")
        except Exception as e:
            QMessageBox.critical(None, "Image Load Error", f"Failed to load TIFF file: {file_loc}\n{e}\n\n{traceback.format_exc()}")
            if QApplication.instance() and app_state: app_state.show_project_view_signal.emit()
            return

        # --- Initialize Napari Viewer ---
        print("Initializing Napari viewer...")
        viewer = napari.Viewer(title=f"Segmentation: {os.path.basename(input_dir)} ({processing_mode.capitalize()} Mode)")

        # --- Connect Destroyed Signal for App Exit Logic ---
        qt_window_to_connect = None
        if viewer and viewer.window:
             if hasattr(viewer.window, '_qt_window') and viewer.window._qt_window:
                 qt_window_to_connect = viewer.window._qt_window
                 print("Connecting destroyed signal using viewer.window._qt_window")
        if qt_window_to_connect:
            try:
                 # Ensure _handle_napari_close is defined elsewhere in helper_funcs.py
                 qt_window_to_connect.destroyed.connect(_handle_napari_close)
                 print(f"Connected destroyed signal for viewer window: {qt_window_to_connect}")
            except NameError:
                 print("Warning: _handle_napari_close function not defined. Cannot connect destroyed signal.")
            except Exception as connect_error:
                 print(f"Warning: Failed to connect destroyed signal: {connect_error}")
        else:
            print("Warning: Could not find underlying Qt window to connect destroyed signal.")

        # --- Initialize GUI Manager ---
        print("Initializing DynamicGUIManager...")
        # This uses the DynamicGUIManager imported at the start of the function
        gui_manager = DynamicGUIManager(viewer, config, image_stack, file_loc, processing_mode)

        # --- Add Navigation and Control Widgets ---

        # 1. Back to Project Button (using standard QPushButton)
        # Ensure create_back_to_project_button is defined elsewhere in helper_funcs.py
        try:
            back_button_widget = create_back_to_project_button(viewer)
            viewer.window.add_dock_widget(back_button_widget, area="left", name="Navigation")
        except NameError:
             print("Warning: create_back_to_project_button function not defined. Cannot add back button.")
        except Exception as e:
             print(f"Error adding back button: {e}")


        # 2. Processing Step Buttons (using magicgui)
        @magicgui(call_button="▶ Next Step / Run Current")
        def continue_processing():
            """Execute the next step or rerun current step if parameters changed."""
            print("Continue Processing button clicked.")
            try:
                # Parent message box to the viewer window if possible
                parent_widget = viewer.window._qt_window if viewer and viewer.window and hasattr(viewer.window, '_qt_window') else None
                gui_manager.execute_processing_step()
                update_navigation_buttons()
            except Exception as e:
                error_msg = f"Error during processing step:\n{str(e)}\n\n{traceback.format_exc()}"
                QMessageBox.critical(parent_widget, "Processing Error", error_msg)
                print(error_msg)

        @magicgui(call_button="◀ Previous Step")
        def go_to_previous_step():
            """Go back one step, clearing results of the undone step."""
            print("Previous Step button clicked.")
            # Parent message box to the viewer window if possible
            parent_widget = viewer.window._qt_window if viewer and viewer.window and hasattr(viewer.window, '_qt_window') else None
            current_step_index = gui_manager.current_step["value"]
            if current_step_index > 0:
                try:
                    print(f"Cleaning up artifacts of step {current_step_index}")
                    gui_manager.cleanup_step(current_step_index)
                    gui_manager.current_step["value"] -= 1
                    prev_step_index = gui_manager.current_step["value"]
                    prev_step_name = gui_manager.processing_steps[prev_step_index]
                    print(f"Recreating widgets for step {prev_step_index+1}: {prev_step_name}")
                    gui_manager.create_step_widgets(prev_step_name)
                    update_navigation_buttons()
                except Exception as e:
                    error_msg = f"Error going to previous step:\n{str(e)}\n\n{traceback.format_exc()}"
                    QMessageBox.critical(parent_widget, "Navigation Error", error_msg)
                    print(error_msg)
            else:
                print("Already at the first step.")

        def update_navigation_buttons():
            """Update the enabled state and label of navigation buttons."""
            try:
                current_step_index = gui_manager.current_step["value"]
                total_steps = len(gui_manager.processing_steps)

                # Check if magicgui widgets exist before accessing them
                if hasattr(go_to_previous_step, 'enabled'):
                    go_to_previous_step.enabled = current_step_index > 0
                if hasattr(continue_processing, 'enabled'):
                    continue_processing.enabled = current_step_index < total_steps
                if hasattr(continue_processing, 'label'):
                    if current_step_index < total_steps:
                         next_step_name = gui_manager.processing_steps[current_step_index]
                         continue_processing.label = f"Run Step {current_step_index + 1}: {next_step_name}"
                    else:
                         continue_processing.label = "Processing Complete"
            except Exception as e:
                print(f"Error updating navigation buttons: {e}") # Prevent crash if widgets deleted

        # --- Add step buttons to a container dock widget ---
        step_widget_container = QWidget()
        step_layout = QVBoxLayout()
        step_widget_container.setLayout(step_layout)
        step_layout.addWidget(continue_processing.native) # Add magicgui widget's native Qt widget
        step_layout.addWidget(go_to_previous_step.native) # Add magicgui widget's native Qt widget
        step_layout.setContentsMargins(5,5,5,5) # Optional margins
        viewer.window.add_dock_widget(step_widget_container, area="left", name="Processing Control")

        # --- Initialize Button States ---
        update_navigation_buttons()

        # --- Start the Napari Event Loop ---
        print("Starting Napari event loop...")
        # This blocks until the Napari window associated with 'viewer' is closed.
        napari.run()

    except Exception as e:
        # Catch-all for errors during setup BEFORE napari.run()
        error_msg = f"Critical error setting up interactive segmentation:\n{str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"
        QMessageBox.critical(None, "Setup Error", error_msg)
        print(error_msg)
        # Clean up viewer if it was created but napari.run() wasn't reached
        if viewer is not None:
             try:
                 viewer.close()
             except Exception as close_err:
                 print(f"Error trying to close viewer after setup error: {close_err}")
        # Try to return to project view if possible
        if QApplication.instance() and app_state:
             app_state.show_project_view_signal.emit()


# --- launch_image_segmentation_tool (Unchanged, but check imports/paths) ---
def launch_image_segmentation_tool():
    """
    Main entry point for the application setup.
    Manages the ProjectViewWindow instance and returns the QApplication instance.
    The actual event loop (app.exec_()) should be called by the script
    that calls this function (e.g., segment.py).
    """
    app = None # Initialize app to None for error handling
    try:
        # Get existing app or create new one
        app = QApplication.instance()
        if app is None:
            print("Creating QApplication instance...")
            # Use sys.argv if available, otherwise provide an empty list.
            # This is standard practice for QApplication.
            app = QApplication(sys.argv if hasattr(sys, 'argv') else [])
            # Quit only when the designated main window (ProjectViewWindow) is closed.
            # Setting this False prevents unexpected quits if other top-level
            # windows (like transient dialogs or potentially detached viewers
            # if not handled carefully) are closed first. The ProjectViewWindow's
            # WA_QuitOnClose attribute and its closeEvent handle the actual quit.
            app.setQuitOnLastWindowClosed(False)
            print("QApplication created. QuitOnLastWindowClosed set to False.")
        else:
             print("Using existing QApplication instance.")


        def show_or_create_project_view():
            """
            Ensures the ProjectViewWindow exists and is visible.
            Creates it if it doesn't exist or its underlying Qt object was deleted.
            Connects to the ApplicationState.
            """
            # Check if a Python reference to the window exists in our state manager
            window_exists_py = app_state.project_view_window is not None
            window_valid_qt = False # Assume Qt object might be invalid initially

            if window_exists_py:
                 try:
                     # Attempt to interact with the Qt object to check its validity.
                     # Accessing a property like isVisible() is a common way.
                     # If the underlying C++ object has been deleted, this will
                     # raise a RuntimeError.
                     _ = app_state.project_view_window.isVisible() # Access property
                     window_valid_qt = True # If no error, Qt object is still alive
                     print("Existing ProjectViewWindow Qt object appears valid.")
                 except RuntimeError:
                     # This specific exception means the C++ part of the widget is gone
                     print("ProjectViewWindow Python reference exists, but underlying Qt widget was deleted. Recreating.")
                     app_state.project_view_window = None # Clear the stale Python reference
                     window_exists_py = False # Mark that Python reference is now None
                     window_valid_qt = False # It's definitely not valid
                 except Exception as e:
                     # Catch other potential errors during the check
                     print(f"Error checking existing project window state: {e}. Assuming invalid and recreating.")
                     app_state.project_view_window = None # Clear the reference
                     window_exists_py = False
                     window_valid_qt = False

            # Determine if a new window instance needs to be created
            create_new_window = not window_exists_py or not window_valid_qt

            if create_new_window:
                print("Creating new ProjectViewWindow instance.")
                # Ensure ProjectManager and ProjectViewWindow classes are defined above
                project_manager = ProjectManager() # Create a new manager for the new window
                app_state.project_view_window = ProjectViewWindow(project_manager)
                # Crucial: Set the attribute to make this window control app exit.
                # When this window with WA_QuitOnClose=True is closed, Qt *may*
                # quit the application IF QuitOnLastWindowClosed is also True (which
                # we set to False). However, setting this attribute is often used
                # in conjunction with custom closeEvent handlers or simply as a flag.
                # The ProjectViewWindow's closeEvent is the primary mechanism here.
                app_state.project_view_window.setAttribute(Qt.WA_QuitOnClose, True)
                print("ProjectViewWindow created and WA_QuitOnClose attribute set to True.")
            # else: # Window exists and is valid
            #      print("Using existing valid ProjectViewWindow instance.")


            # Now, show and activate the window (either existing or newly created)
            if app_state.project_view_window:
                 print("Showing and activating ProjectViewWindow...")
                 app_state.project_view_window.show()
                 app_state.project_view_window.activateWindow() # Bring to front if possible
                 app_state.project_view_window.raise_()         # Ensure it's raised above other windows
            else:
                 # This path indicates a failure in the creation logic above.
                 critical_error_msg = "Error: ProjectViewWindow is unexpectedly None after creation/check."
                 print(critical_error_msg)
                 # Show a message box about this internal error
                 try:
                     QMessageBox.critical(None, "Internal Error", critical_error_msg)
                 except Exception as msg_err:
                     print(f"Could not display critical error message: {msg_err}")
                 # We should probably not continue if the main window failed creation.
                 # Returning None from the outer function will handle this.
                 raise RuntimeError(critical_error_msg) # Raise error to be caught by outer try/except


        # --- Signal Connection ---
        # Connect the global state signal to the function that shows/creates the window.
        # This allows other parts of the app (like the 'Back' button) to bring back the project view.
        print("Setting up show_project_view_signal connection...")
        try:
             # Disconnect first to prevent multiple connections if this function
             # is somehow called again within the same application instance.
             app_state.show_project_view_signal.disconnect(show_or_create_project_view)
             print("Disconnected existing show_project_view_signal connection (if any).")
        except TypeError:
             # This is expected if the signal had no connections yet.
             print("No existing show_project_view_signal connection found to disconnect (normal).")
             pass
        except Exception as e:
             # Log unexpected errors during disconnect attempt
             print(f"Warning: Non-TypeError during signal disconnect attempt: {e}")

        # Connect the signal
        app_state.show_project_view_signal.connect(show_or_create_project_view)
        print("Connected show_project_view_signal to show_or_create_project_view.")


        # --- Initial Show ---
        # Explicitly call the function here to ensure the project view is created
        # and shown when the application starts for the first time.
        print("Performing initial call to show_or_create_project_view...")
        show_or_create_project_view()

        # --- Return Application Instance ---
        # The setup is complete. Return the QApplication instance to the caller.
        # The caller (segment.py) is responsible for starting the event loop (app.exec_()).
        print("launch_image_segmentation_tool finished setup successfully, returning app instance.")
        return app

    except Exception as e:
        # Catch any unexpected errors during the entire setup process
        error_msg = (f"Unhandled critical error in launch_image_segmentation_tool setup:\n{str(e)}\n\n"
                     f"Full Traceback:\n{traceback.format_exc()}")
        print(error_msg)
        # Attempt to display the error in a message box for the user
        try:
            # Need a QApplication instance to show a message box.
            # If 'app' is None here, it means QApplication creation failed very early.
            if app is None:
                 # Try creating a temporary, minimal app just for the error dialog.
                 print("Creating temporary QApplication instance for error message.")
                 error_app = QApplication(sys.argv if hasattr(sys, 'argv') else [])
                 QMessageBox.critical(None, "Fatal Launch Error", error_msg)
                 # We don't run error_app.exec_()
            else:
                 # Use the 'app' instance if it was created successfully before the error.
                 QMessageBox.critical(None, "Fatal Launch Error", error_msg)
        except Exception as msg_err:
             # Fallback if even the error message fails
             print(f"Failed to display the graphical error message: {msg_err}")

        # Indicate failure to the caller by returning None
        print("launch_image_segmentation_tool failed due to an error, returning None.")
        return None

# --- create_back_to_project_button (Unchanged logic, added checks) ---
def create_back_to_project_button(viewer):
    """
    Create a standard QPushButton in a QWidget container to act as the
    'Back to Project List' button.

    Args:
        viewer (napari.Viewer): The Napari viewer instance this button belongs to.

    Returns:
        QWidget: A container widget holding the button, suitable for add_dock_widget.
    """
    # Define the action function separately
    def _do_back_to_project():
        """
        Closes the current Napari viewer and emits signal to show project view.
        """
        print("Back to Project button clicked (standard button).")
        try:
            # Initiate viewer closing
            if viewer is not None:
                print(f"Attempting to close viewer: {viewer}")
                # viewer.close() triggers the destruction and 'destroyed' signal
                viewer.close()
                print("viewer.close() called.")
            else:
                 print("Viewer instance is None. Cannot call close().")

            # Emit signal immediately to bring back project view
            print("Emitting show_project_view_signal.")
            if app_state:
                 app_state.show_project_view_signal.emit()
            else:
                 print("Error: ApplicationState instance is not available.")

        except Exception as e:
             error_msg = f"Error during 'Back to Project':\n{str(e)}\n\nFull Traceback:\n{traceback.format_exc()}"
             QMessageBox.critical(None, "Navigation Error", error_msg)
             print(error_msg)
             # Fallback signal emission
             if QApplication.instance() and app_state:
                 print("Attempting to emit signal despite error during close.")
                 app_state.show_project_view_signal.emit()

    # Create the standard Qt button
    button = QPushButton("Back to Project List")
    button.clicked.connect(_do_back_to_project)

    # Create a container widget to hold the button
    # This container is what gets added as the dock widget
    container_widget = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(button)
    layout.setContentsMargins(5, 5, 5, 5) # Optional margins
    container_widget.setLayout(layout)

    return container_widget



# --- END OF FILE helper_funcs.py ---