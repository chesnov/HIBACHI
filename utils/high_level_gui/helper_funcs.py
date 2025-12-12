# --- START OF FILE utils/high_level_gui/helper_funcs.py ---
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

# Corrected path for BatchProcessor import
try:
    from .batch_processor import BatchProcessor
except ImportError as e:
    print(f"WARNING: Failed to import BatchProcessor in helper_funcs.py: {e}. Batch processing button will be disabled.")
    BatchProcessor = None


def natural_sort_key(s):
    """
    Create a sort key for natural alphanumeric sorting (e.g., image1, image2, image10).
    Operates on the basename of the input path string.
    """
    basename = os.path.basename(s)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', basename)]


def create_parameter_widget(param_name: str, param_config: Dict[str, Any], callback):
    """
    Create a widget for a single parameter, supporting int, float, bool, and list (via LineEdit).
    """
    param_type = param_config.get("type", "float")
    label = param_config.get("label", param_name)
    widget = None

    try:
        if param_type == "list":
            initial_list = param_config.get("value", [])
            if not isinstance(initial_list, list):
                print(f"Warning: Initial value for list parameter '{param_name}' is not a list ({type(initial_list)}). Using empty list.")
                initial_list = []
            initial_str = ", ".join(map(str, initial_list))
            last_valid_list = list(initial_list)

            def list_widget(value_str: str = initial_str):
                nonlocal last_valid_list
                new_list = None
                try:
                    cleaned_str = value_str.strip()
                    if not cleaned_str: new_list = []
                    else: new_list = [float(x.strip()) for x in cleaned_str.split(',') if x.strip()]
                    # print(f"Parsed list for {param_name}: {new_list}") # Optional Debug
                    if new_list != last_valid_list:
                         callback(new_list)
                         last_valid_list[:] = new_list
                    if hasattr(list_widget, 'native') and list_widget.native:
                         list_widget.native.setStyleSheet("")
                    return value_str
                except ValueError as e:
                    print(f"Warning: Invalid list format for {param_name}: '{value_str}'. Error: {e}.")
                    print(f"         Keeping last valid value: {last_valid_list}")
                    if hasattr(list_widget, 'native') and list_widget.native:
                        list_widget.native.setStyleSheet("background-color: #FFDDDD;")
                    return ", ".join(map(str, last_valid_list))
            widget = magicgui(list_widget, auto_call=True, value_str={"widget_type": "LineEdit", "label": label})

        elif param_type == "float":
            default_value = float(param_config.get("value", 0.0))
            min_val = float(param_config.get("min", 0.0))
            max_val = float(param_config.get("max", 100.0))
            step_val = float(param_config.get("step", 0.1))
            default_value = max(min_val, min(default_value, max_val))
            def float_widget(value: float = default_value): callback(value); return value
            widget = magicgui(float_widget, auto_call=True, value={"widget_type": "FloatSpinBox", "label": label, "min": min_val, "max": max_val, "step": step_val})

        elif param_type == "int":
            default_value = int(param_config.get("value", 0))
            min_val = int(param_config.get("min", 0))
            max_val = int(param_config.get("max", 100))
            step_val = int(param_config.get("step", 1))
            default_value = max(min_val, min(default_value, max_val))
            def int_widget(value: int = default_value): callback(value); return value
            widget = magicgui(int_widget, auto_call=True, value={"widget_type": "SpinBox", "label": label, "min": min_val, "max": max_val, "step": step_val})

        elif param_type == "bool":
             default_value = bool(param_config.get("value", False))
             def bool_widget(value: bool = default_value): callback(value); return value
             widget = magicgui(bool_widget, auto_call=True, value={"widget_type": "CheckBox", "label": label})
        else:
            print(f"Warning: Unsupported parameter type '{param_type}' for '{param_name}'. Creating basic LineEdit.")
            default_value_str = str(param_config.get("value", ""))
            last_valid_str = default_value_str
            def fallback_widget(value: str = default_value_str):
                nonlocal last_valid_str
                if value != last_valid_str: callback(value); last_valid_str = value
                return value
            widget = magicgui(fallback_widget, auto_call=True, value={"widget_type": "LineEdit", "label": label})

        if widget: widget.param_name = param_name
    except Exception as e:
         print(f"ERROR creating widget for parameter '{param_name}': {e}\nConfig: {param_config}\n{traceback.format_exc()}")
         return None
    return widget


def scan_available_presets():
    """
    Scans module directories for configuration presets in 'configs' subfolders.
    Returns a dict: { "Display Name": {"path": full_path, "default_mode": mode_string} }
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define where to look and what the fallback mode should be if the YAML doesn't specify it
    search_locations = [
        # (Relative Path to configs, Default Mode)
        (os.path.join(script_dir, '..', 'module_3d', 'configs'), 'ramified'),
        (os.path.join(script_dir, '..', 'module_2d', 'configs'), 'ramified_2d')
    ]

    presets = {}

    for config_dir, default_mode in search_locations:
        if not os.path.exists(config_dir):
            continue
            
        try:
            files = [f for f in os.listdir(config_dir) if f.lower().endswith(('.yaml', '.yml'))]
            for f in files:
                full_path = os.path.join(config_dir, f)
                # Create a readable name: "ramified_config.yaml" -> "Ramified Config (3D)"
                clean_name = os.path.splitext(f)[0].replace('_', ' ').title()
                
                # Tag it with 2D/3D based on the folder we found it in
                suffix = " (2D)" if "module_2d" in config_dir else " (3D)"
                display_name = f"{clean_name}{suffix}"
                
                presets[display_name] = {
                    "path": full_path,
                    "default_mode": default_mode
                }
        except Exception as e:
            print(f"Error scanning presets in {config_dir}: {e}")

    return presets


def organize_processing_dir(drctry, preset_details):
    """
    Organizes a directory using a specific configuration template.
    Args:
        drctry: Target folder path.
        preset_details: Dict containing {'path': str, 'default_mode': str}
    """
    config_template_path = preset_details['path']
    fallback_mode = preset_details['default_mode']
    
    print(f"Organizing directory: {drctry} using template: {os.path.basename(config_template_path)}")
    
    # --- Standard File Validation (No changes here) ---
    try:
        all_files = os.listdir(drctry)
        tif_files = [f for f in all_files if f.lower().endswith(('.tif', '.tiff')) and os.path.isfile(os.path.join(drctry, f))]
        csv_files = [f for f in all_files if f.lower().endswith('.csv') and os.path.isfile(os.path.join(drctry, f))]
    except OSError as e: raise OSError(f"Error listing files in directory {drctry}: {e}") from e

    if not tif_files: raise ValueError('No .tif or .tiff files found directly in the selected directory.')
    if len(csv_files) != 1: raise ValueError(f'Expected exactly one .csv file in the directory, found {len(csv_files)}: {", ".join(csv_files)}')

    csv_path = os.path.join(drctry, csv_files[0])
    try: df = pd.read_csv(csv_path)
    except Exception as e: raise ValueError(f"Error reading CSV file {csv_path}: {e}") from e

    # --- Determine Mode from Template ---
    # We read the template first to see if it dictates the mode (e.g. 2D vs 3D requirements)
    template_data = {}
    try:
        with open(config_template_path, 'r') as f:
            template_data = yaml.safe_load(f) or {}
    except Exception as e:
        raise ValueError(f"Selected template is invalid: {e}")

    # Use mode from YAML, or fallback to folder-based mode
    mode = template_data.get('mode', fallback_mode)
    
    is_2d_mode = mode.endswith('_2d')
    required_cols = ['Filename', 'Width (um)', 'Height (um)']
    dimension_section_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'
    
    if not is_2d_mode:
        required_cols.extend(['Slices', 'Depth (um)'])

    if not all([col in df.columns for col in required_cols]):
        raise ValueError(f'For preset mode "{mode}", CSV must have columns: {", ".join(required_cols)}. Found: {", ".join(df.columns)}')

    # --- Match Files (No changes here) ---
    csv_filenames = set(df['Filename'].astype(str))
    tif_basenames = set(os.path.splitext(f)[0] for f in tif_files)
    if csv_filenames != tif_basenames:
        # (Error handling logic same as before...)
        missing_in_csv = tif_basenames - csv_filenames
        missing_in_folder = csv_filenames - tif_basenames
        error_msg = "Mismatch between CSV 'Filename' column and TIF/TIFF files:"
        if missing_in_csv: error_msg += f"\n - TIFs not in CSV: {', '.join(missing_in_csv)}"
        if missing_in_folder: error_msg += f"\n - CSV names without TIF: {', '.join(missing_in_folder)}"
        raise ValueError(error_msg)

    root_names = list(tif_basenames)

    # --- Organization Loop ---
    for root_name in root_names:
        new_dir = os.path.join(drctry, root_name)
        try: os.makedirs(new_dir, exist_ok=True)
        except OSError as e: raise OSError(f"Error creating directory {new_dir}: {e}") from e

        actual_tif_file = next((f for f in tif_files if os.path.splitext(f)[0] == root_name), None)
        if not actual_tif_file: continue

        # Move TIF (Same as before)
        original_tif_path = os.path.join(drctry, actual_tif_file)
        new_tif_path = os.path.join(new_dir, actual_tif_file)

        if os.path.abspath(original_tif_path) != os.path.abspath(new_tif_path):
             if os.path.exists(original_tif_path):
                try: shutil.move(original_tif_path, new_tif_path)
                except Exception as e:
                     if not os.path.exists(new_tif_path): raise OSError(f"Error moving file {original_tif_path}: {e}") from e

        # --- Copy Config Template ---
        config_filename = os.path.basename(config_template_path)
        new_config_path = os.path.join(new_dir, config_filename)
        
        try:
            # Always copy the template to ensure we have the correct structure
            if not os.path.exists(new_config_path):
                 shutil.copy2(config_template_path, new_config_path)

            # Get Dimensions from CSV
            row = df[df['Filename'].astype(str) == root_name]
            if row.empty: continue

            x_um = row['Width (um)'].iloc[0]
            y_um = row['Height (um)'].iloc[0]
            z_um = 0.0 if is_2d_mode else row['Depth (um)'].iloc[0]

            # Update the YAML
            config_data = {}
            if os.path.exists(new_config_path):
                with open(new_config_path, 'r') as f: config_data = yaml.safe_load(f) or {}

            if dimension_section_key not in config_data: config_data[dimension_section_key] = {}
            
            config_data[dimension_section_key]['x'] = float(x_um)
            config_data[dimension_section_key]['y'] = float(y_um)
            
            if not is_2d_mode: 
                config_data[dimension_section_key]['z'] = float(z_um)
            
            # ENSURE MODE IS SET
            config_data['mode'] = mode

            with open(new_config_path, 'w') as f: 
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                
        except Exception as e: 
            raise RuntimeError(f"Error processing config file for {root_name}: {e}\n{traceback.format_exc()}") from e
            
    print(f"Directory organization complete for {drctry}")


class ApplicationState(QObject):
    show_project_view_signal = pyqtSignal()
    _instance = None
    project_view_window = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ApplicationState, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance
    def __init__(self):
        if not getattr(self, '_ApplicationState__initialized', False):
            super().__init__()
            self.__initialized = True
app_state = ApplicationState()


class ProjectManager:
    def __init__(self):
        self.project_path = None
        self.image_folders = [] # List of full paths to valid image subdirectories
    def select_project_folder(self):
        selected_path = QFileDialog.getExistingDirectory(None, "Select Project Root Folder", "")
        if not selected_path: print("Project folder selection cancelled."); return None
        self.project_path = selected_path
        print(f"Project folder selected: {self.project_path}")
        self._find_valid_image_folders() # Scan after selection
        return self.project_path
    def _find_valid_image_folders(self):
        self.image_folders = []
        if not self.project_path or not os.path.isdir(self.project_path):
            # print("Project path not set or invalid, cannot find image folders.") # Can be less verbose
            return
        # print(f"Scanning for valid image folders in: {self.project_path}") # Can be less verbose
        try:
            for item in os.listdir(self.project_path):
                potential_folder_path = os.path.join(self.project_path, item)
                if os.path.isdir(potential_folder_path):
                    try:
                        folder_contents = os.listdir(potential_folder_path)
                        tif_files = [f for f in folder_contents if f.lower().endswith(('.tif', '.tiff')) and os.path.isfile(os.path.join(potential_folder_path, f))]
                        yaml_files = [f for f in folder_contents if f.lower().endswith(('.yaml', '.yml')) and os.path.isfile(os.path.join(potential_folder_path, f))]
                        if len(tif_files) == 1 and len(yaml_files) == 1:
                            # print(f"  Found valid image folder: {item}") # Less verbose
                            self.image_folders.append(potential_folder_path)
                    except OSError as e: print(f"Warning: Could not read subdirectory {potential_folder_path}: {e}")
        except OSError as e: print(f"Error listing contents of project path {self.project_path}: {e}"); self.image_folders = []
        self.image_folders.sort(key=natural_sort_key)
        # print(f"Found {len(self.image_folders)} valid image folders.") # Can be less verbose
    def get_image_details(self, folder_path):
        try:
            contents = os.listdir(folder_path)
            tif_file = next((f for f in contents if f.lower().endswith(('.tif', '.tiff'))), None)
            yaml_file = next((f for f in contents if f.lower().endswith(('.yaml', '.yml'))), None)
            if not tif_file or not yaml_file: raise FileNotFoundError(f"Required TIF/YAML file not found in {folder_path}")
            config = {}
            yaml_path = os.path.join(folder_path, yaml_file)
            try:
                with open(yaml_path, 'r') as file: config = yaml.safe_load(file) or {}
            except Exception as e: print(f"Warning: Error reading YAML {yaml_path}: {e}"); config = {} # Keep it brief
            return {'path': folder_path, 'tif_file': tif_file, 'yaml_file': yaml_file, 'mode': config.get('mode', 'unknown')}
        except Exception as e:
             print(f"Error getting image details for {folder_path}: {e}")
             return {'path': folder_path, 'tif_file': 'Error', 'yaml_file': 'Error', 'mode': 'error'}


class ProjectViewWindow(QMainWindow):
    def __init__(self, project_manager):
        super().__init__()
        self.project_manager = project_manager
        self.initUI()
        self.setAttribute(Qt.WA_QuitOnClose)

    def initUI(self):
        self.setWindowTitle("Image Segmentation Project")
        self.setGeometry(100, 100, 700, 450) # Adjusted width for one batch button
        central_widget = QWidget(); layout = QVBoxLayout()
        self.project_path_label = QLabel("Project Path: Not Selected"); layout.addWidget(self.project_path_label)
        self.image_list = QListWidget(); self.image_list.itemDoubleClicked.connect(self.open_image_view); layout.addWidget(self.image_list)

        button_layout = QHBoxLayout()
        select_project_btn = QPushButton("Select/Load Project Folder"); select_project_btn.clicked.connect(self.load_project); button_layout.addWidget(select_project_btn)

        # --- SINGLE BATCH PROCESS BUTTON ---
        self.batch_process_all_btn = QPushButton("Process All Compatible Folders")
        self.batch_process_all_btn.clicked.connect(self.run_batch_processing_all_compatible)
        self.batch_process_all_btn.setEnabled(False) # Initially disabled
        if BatchProcessor is None:
            self.batch_process_all_btn.setToolTip("BatchProcessor module not available. Check console for import errors.")
        else:
            self.batch_process_all_btn.setToolTip("Load a project with compatible folders (e.g., 'ramified', 'ramified_2d') to enable.")
        button_layout.addWidget(self.batch_process_all_btn)
        # --- END SINGLE BATCH PROCESS BUTTON ---

        layout.addLayout(button_layout); central_widget.setLayout(layout); self.setCentralWidget(central_widget)

    def _update_batch_button_state(self):
        """Updates the enabled state of the single batch processing button."""
        if BatchProcessor is None:
            self.batch_process_all_btn.setEnabled(False)
            self.batch_process_all_btn.setToolTip("BatchProcessor module not available.")
            return

        if not self.project_manager or not self.project_manager.project_path or not self.project_manager.image_folders:
            self.batch_process_all_btn.setEnabled(False)
            self.batch_process_all_btn.setToolTip("Load a project to enable.")
            return

        # Check if any loaded folder has a mode supported by BatchProcessor
        # Assuming BatchProcessor instance is not needed just to check its supported_strategies keys
        # If BatchProcessor class is available, we can access its class variable or a temp instance.
        # For simplicity, let's assume we know the keys or can get them from a dummy instance.
        # More robustly, BatchProcessor could have a static method for supported keys.
        # For now, let's create a temporary instance just for this check if BatchProcessor is not None
        temp_processor = BatchProcessor(self.project_manager) # Minimal impact
        supported_modes = temp_processor.supported_strategies.keys()
        del temp_processor # Clean up

        has_compatible_folders = any(
            self.project_manager.get_image_details(fp).get('mode') in supported_modes
            for fp in self.project_manager.image_folders
        )

        if has_compatible_folders:
            self.batch_process_all_btn.setEnabled(True)
            self.batch_process_all_btn.setToolTip(f"Process all folders with supported modes: {', '.join(supported_modes)}.")
        else:
            self.batch_process_all_btn.setEnabled(False)
            self.batch_process_all_btn.setToolTip(f"No folders found with supported modes ({', '.join(supported_modes)}).")


    def load_project(self):
        selected_path = self.project_manager.select_project_folder()
        if not selected_path:
            self._update_batch_button_state() # Update even on cancel
            return

        self.project_path_label.setText(f"Project Path: {selected_path}"); self.image_list.clear()
        try:
            needs_organizing = False; root_tifs = []; root_csvs = []; root_dirs = []
            try:
                root_contents = os.listdir(selected_path)
                for item in root_contents:
                    item_path = os.path.join(selected_path, item)
                    if os.path.isfile(item_path):
                         if item.lower().endswith(('.tif', '.tiff')): root_tifs.append(item)
                         elif item.lower().endswith('.csv'): root_csvs.append(item)
                    elif os.path.isdir(item_path): root_dirs.append(item)
                if root_tifs and len(root_csvs) == 1:
                    tif_basenames = set(os.path.splitext(f)[0] for f in root_tifs)
                    existing_matching_dirs = tif_basenames.intersection(set(root_dirs))
                    if not existing_matching_dirs: needs_organizing = True; # print(f"Detected unorganized project structure in: {selected_path}") # Less verbose
            except OSError as e: QMessageBox.critical(self, "Error Listing Directory", f"Could not read directory contents:\n{selected_path}\n{e}"); self._update_batch_button_state(); return

            if needs_organizing:
                reply = QMessageBox.question(self, "Organize Project?", f"Unorganized structure detected in: {selected_path}.\nOrganize now?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    # --- NEW PRESET SELECTION LOGIC ---
                    presets = scan_available_presets()
                    
                    if not presets:
                        QMessageBox.critical(self, "No Presets Found", "Could not find any configuration presets in 'module_*/configs/'.")
                        return

                    preset_names = list(presets.keys())
                    preset_names.sort() # Alphabetical order
                    
                    selected_preset_name, ok = QInputDialog.getItem(self, "Select Configuration Preset", "Choose a preset to apply:", preset_names, 0, False)
                    
                    if ok and selected_preset_name:
                        selected_preset_details = presets[selected_preset_name]
                        try: 
                            # Pass the details dict, not just the name
                            organize_processing_dir(selected_path, selected_preset_details) 
                            QMessageBox.information(self, "Organization Complete", f"Project organized using '{selected_preset_name}'.")
                        except Exception as e: 
                            QMessageBox.critical(self, "Organization Failed", f"Failed to organize project:\n{e}\n{traceback.format_exc()}")
                    else: 
                        QMessageBox.warning(self, "Organization Cancelled", "Project organization cancelled.")
                    
                    self.project_manager._find_valid_image_folders() # Rescan
                else: QMessageBox.information(self, "Organization Skipped", "Loading existing valid subfolders only."); self.project_manager._find_valid_image_folders()

            if not self.project_manager.image_folders and not needs_organizing:
                 QMessageBox.warning(self, "No Valid Images", f"No valid image subfolders found in:\n{selected_path}")

            for folder_path in self.project_manager.image_folders:
                try:
                    details = self.project_manager.get_image_details(folder_path)
                    display_text = f"{os.path.basename(folder_path)} - {'Error loading' if details.get('mode') == 'error' else 'Mode: ' + details.get('mode', 'unknown')}"
                    item_widget = QListWidgetItem(display_text); item_widget.setData(Qt.UserRole, folder_path); self.image_list.addItem(item_widget)
                except Exception as e:
                    print(f"Error processing folder {folder_path} for display list: {e}")
                    error_item = QListWidgetItem(f"{os.path.basename(folder_path)} - Error processing"); error_item.setData(Qt.UserRole, folder_path); self.image_list.addItem(error_item)

        except Exception as e:
            QMessageBox.critical(self, "Error Loading Project", f"An unexpected error occurred:\n{e}\n{traceback.format_exc()}"); self.image_list.clear(); self.project_path_label.setText("Project Path: Error")
        finally:
            self._update_batch_button_state() # Update button state after loading attempt


    def open_image_view(self, item):
        try:
            selected_folder = item.data(Qt.UserRole)
            if not selected_folder or not os.path.isdir(selected_folder): QMessageBox.warning(self, "Error", f"Invalid folder path: {selected_folder}"); return
            print(f"Opening image view for: {selected_folder}"); self.hide()
            interactive_segmentation_with_config(selected_folder)
        except Exception as e: error_msg = f"Error opening image view:\n{str(e)}\n{traceback.format_exc()}"; QMessageBox.critical(self, "Error", error_msg); print(error_msg); self.show()


    def run_batch_processing_all_compatible(self):
        """Handles the click of the 'Process All Compatible Folders' button."""
        if not self.batch_process_all_btn.isEnabled():
            QMessageBox.warning(self, "Batch Processing Unavailable",
                                "Batch processing is currently not available. "
                                "Ensure BatchProcessor module is loaded and a project with compatible folders is open.");
            return

        # Determine compatible folders again, just to be sure
        temp_processor = BatchProcessor(self.project_manager)
        supported_modes = temp_processor.supported_strategies.keys()
        del temp_processor

        compatible_folders_info = [] # Store (folder_path, mode)
        for fp in self.project_manager.image_folders:
            details = self.project_manager.get_image_details(fp)
            mode = details.get('mode')
            if mode in supported_modes:
                compatible_folders_info.append((fp, mode))

        if not compatible_folders_info:
            QMessageBox.information(self, "No Compatible Folders",
                                    f"No folders configured for supported modes ({', '.join(supported_modes)}) found. "
                                    "The button should have been disabled.");
            self._update_batch_button_state() # Correct button state
            return

        num_folders = len(compatible_folders_info)
        # Create a more informative list for the confirmation dialog
        folder_summary_list = [f"  - {os.path.basename(fp_info[0])} (Mode: {fp_info[1]})" for fp_info in compatible_folders_info[:5]] # Show first 5
        if num_folders > 5: folder_summary_list.append("  - ... and more.")
        folder_summary_str = "\n".join(folder_summary_list)


        force_restart_processing = False
        reply_force = QMessageBox.question(self, f"Force Restart Option (All Compatible)",
                                     f"For the {num_folders} compatible folder(s):\n{folder_summary_str}\n\n"
                                     "Do you want to force reprocessing of ALL steps, even if previously completed?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply_force == QMessageBox.Yes:
            force_restart_processing = True
            print(f"User chose to FORCE RESTART all processing for compatible folders.")

        reply_confirm = QMessageBox.question(self, f"Confirm Batch Processing (All Compatible)",
                                     f"Process {num_folders} compatible folder(s)?\n{folder_summary_str}\n\n"
                                     f"{'ALL STEPS WILL BE REPROCESSED.' if force_restart_processing else 'Processing will attempt to resume incomplete folders.'}\n"
                                     f"Existing processed data will be affected.\nThis can take time.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply_confirm == QMessageBox.No: return

        print(f"Starting batch processing for {num_folders} compatible folders...");
        original_tooltip = self.batch_process_all_btn.toolTip()
        self.batch_process_all_btn.setEnabled(False);
        self.batch_process_all_btn.setToolTip("Processing... Please wait.")
        QApplication.processEvents()

        processor = BatchProcessor(self.project_manager)
        try:
            # Call the modified process_all_folders which infers mode per folder
            successful_count, failed_count, skipped_count = processor.process_all_folders(
                force_restart_all=force_restart_processing
            )
            summary_msg = f"Batch processing for all compatible folders finished.\n\n" \
                          f"Successfully processed/resumed/completed: {successful_count} folder(s)\n" \
                          f"Failed during processing: {failed_count} folder(s)\n" \
                          f"Skipped (unsupported mode/error before start): {skipped_count} folder(s)\n\n" \
                          f"Check console output for detailed logs and errors."
            if failed_count > 0: QMessageBox.warning(self, "Batch Processing Complete (with issues)", summary_msg)
            else: QMessageBox.information(self, "Batch Processing Complete", summary_msg)
        except Exception as e:
            print(f"Critical error during 'Process All Compatible Folders': {e}"); traceback.print_exc();
            QMessageBox.critical(self, "Batch Processing Error", f"Error: {e}\nCheck console.")
        finally:
            self.batch_process_all_btn.setToolTip(original_tooltip)
            self._update_batch_button_state()
            print(f"Batch processing GUI action for 'All Compatible Folders' finished.")

    # Generic batch processing runner
    def run_batch_processing_for_mode(self, mode_key: str):
        """Handles the click of a 'Process All' button for a specific mode."""
        
        target_button = None
        mode_display_name = ""
        if mode_key == "ramified":
            target_button = self.batch_process_ramified_btn
            mode_display_name = "Ramified 3D"
        elif mode_key == "ramified_2d":
            target_button = self.batch_process_ramified_2d_btn
            mode_display_name = "Ramified 2D"
        else:
            QMessageBox.critical(self, "Internal Error", f"Unknown mode key '{mode_key}' for batch processing.")
            return

        if target_button is None or not target_button.isEnabled():
            QMessageBox.warning(self, f"Batch Processing ({mode_display_name}) Unavailable",
                                f"Batch processing for {mode_display_name} mode is currently not available. "
                                f"Ensure a project with '{mode_key}' folders is loaded.");
            return

        # Confirm folders for the specific mode
        folders_to_process = [
            fp for fp in self.project_manager.image_folders
            if self.project_manager.get_image_details(fp).get('mode') == mode_key
        ]
        if not folders_to_process:
            QMessageBox.information(self, f"No {mode_display_name} Images",
                                    f"No folders configured for '{mode_key}' mode found. "
                                    f"The 'Process All ({mode_display_name})' button should have been disabled.");
            self._update_batch_buttons_state()
            return

        num_folders = len(folders_to_process)

        force_restart_processing = False
        reply_force = QMessageBox.question(self, f"Force Restart Option ({mode_display_name})",
                                     f"For the {num_folders} '{mode_key}' folder(s):\n"
                                     "Do you want to force reprocessing of ALL steps, even if previously completed?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply_force == QMessageBox.Yes:
            force_restart_processing = True
            print(f"User chose to FORCE RESTART all processing for '{mode_key}' mode.")


        reply_confirm = QMessageBox.question(self, f"Confirm Batch Processing ({mode_display_name})",
                                     f"Process {num_folders} '{mode_key}' mode folder(s)?\n"
                                     f"{'ALL STEPS WILL BE REPROCESSED.' if force_restart_processing else 'Processing will attempt to resume incomplete folders.'}\n"
                                     f"Existing processed data for {'all steps (if forcing restart)' if force_restart_processing else 'steps to be run'} will be overwritten.\n"
                                     "This can take time.",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply_confirm == QMessageBox.No: return

        print(f"Starting batch processing for {num_folders} '{mode_key}' folders...");
        original_tooltip = target_button.toolTip()
        target_button.setEnabled(False);
        target_button.setToolTip("Processing... Please wait.")
        QApplication.processEvents()

        processor = BatchProcessor(self.project_manager) # ProjectManager already has all folders
        try:
            successful_count, failed_count = processor.process_all_folders(
                target_strategy_key=mode_key, # Pass the specific mode key
                force_restart_all=force_restart_processing
            )
            summary_msg = f"Batch processing for '{mode_display_name}' finished.\n\n" \
                          f"Successfully processed/resumed/completed: {successful_count} folder(s)\n" \
                          f"Failed during processing: {failed_count} folder(s)\n\n" \
                          f"Check console output for detailed logs and errors."
            if failed_count > 0: QMessageBox.warning(self, f"Batch Processing ({mode_display_name}) Complete (with issues)", summary_msg)
            else: QMessageBox.information(self, f"Batch Processing ({mode_display_name}) Complete", summary_msg)
        except Exception as e:
            print(f"Critical error during batch processing for mode '{mode_key}': {e}"); traceback.print_exc();
            QMessageBox.critical(self, f"Batch Processing ({mode_display_name}) Error", f"Error: {e}\nCheck console.")
        finally:
            target_button.setToolTip(original_tooltip)
            self._update_batch_buttons_state() # Update all buttons based on current project state
            print(f"Batch processing GUI action for mode '{mode_key}' finished.")


    def closeEvent(self, event: QCloseEvent):
        # print("ProjectViewWindow closeEvent. Quitting application.") # Less verbose
        reply = QMessageBox.question(self, 'Confirm Exit', "Exit application?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes: app = QApplication.instance(); app.quit(); event.accept()
        else: event.ignore()


def _check_if_last_window():
    app = QApplication.instance()
    if not app: return
    project_window = app_state.project_view_window
    project_view_exists_valid = False
    if project_window:
        try: _ = project_window.isVisible(); project_view_exists_valid = True
        except RuntimeError: project_view_exists_valid = False # C++ object deleted
    active_windows = app.topLevelWidgets()
    other_visible_task_windows = [w for w in active_windows if w is not project_window and w.isVisible()]

    pv_is_visible = project_window.isVisible() if project_view_exists_valid else False

    if (not project_view_exists_valid or not pv_is_visible) and not other_visible_task_windows:
        # print("CheckIfLastWindow: Project View gone/hidden, no other visible windows. Quitting.") # Less verbose
        app.quit()
    # else: # Less verbose
        # print(f"CheckIfLastWindow: Project View visible or other windows active. Not quitting. (PV exists/valid: {project_view_exists_valid}, PV visible: {pv_is_visible}, Other visible: {len(other_visible_task_windows)})")


def _handle_napari_close():
    # print("Napari window destroyed signal.") # Less verbose
    QTimer.singleShot(100, _check_if_last_window)


def interactive_segmentation_with_config(selected_folder=None):
    try: from .gui_manager import DynamicGUIManager
    except ImportError as e: print(f"Error importing DynamicGUIManager: {e}"); QMessageBox.critical(None, "Import Error", "Could not load GUI. Check installation."); app_state.show_project_view_signal.emit(); return
    app = QApplication.instance()
    if app is None: app = QApplication(sys.argv if hasattr(sys, 'argv') else []); app.setQuitOnLastWindowClosed(False)
    viewer = None
    try:
        if selected_folder is None or not os.path.isdir(selected_folder): raise ValueError("No valid image folder provided.")
        input_dir = selected_folder; # print(f"Starting interactive segmentation for: {input_dir}") # Less verbose
        try:
             contents = os.listdir(input_dir)
             tif_files = [f for f in contents if f.lower().endswith(('.tif', '.tiff'))]; yaml_files = [f for f in contents if f.lower().endswith(('.yaml', '.yml'))]
             if len(tif_files) != 1: raise FileNotFoundError(f"Expected 1 TIF, found {len(tif_files)} in {input_dir}")
             if len(yaml_files) != 1: raise FileNotFoundError(f"Expected 1 YAML, found {len(yaml_files)} in {input_dir}")
             file_loc = os.path.join(input_dir, tif_files[0]); config_path = os.path.join(input_dir, yaml_files[0])
        except (FileNotFoundError, OSError) as e: QMessageBox.critical(None, "File Error", f"Error finding files in {input_dir}:\n{e}"); app_state.show_project_view_signal.emit(); return
        try:
            with open(config_path, 'r') as file: config = yaml.safe_load(file)
            if config is None: raise ValueError("YAML file empty/invalid.")
        except Exception as e: QMessageBox.critical(None, "Config Error", f"Failed to load/parse YAML: {config_path}\n{e}"); app_state.show_project_view_signal.emit(); return
        processing_mode = config.get('mode')
        if not processing_mode: QMessageBox.critical(None, "Config Error", f"YAML ({config_path}) must have 'mode'."); app_state.show_project_view_signal.emit(); return
        if processing_mode not in ["ramified", "ramified_2d"]: QMessageBox.critical(None, "Config Error", f"Invalid mode '{processing_mode}' in YAML."); app_state.show_project_view_signal.emit(); return
        try:
            image_stack = tiff.imread(file_loc); # print(f"Loaded: {file_loc}, shape {image_stack.shape}, dtype {image_stack.dtype}") # Less verbose
            expected_ndim = 2 if processing_mode == "ramified_2d" else 3
            if image_stack.ndim != expected_ndim: raise ValueError(f"Expected {expected_ndim}D image for mode '{processing_mode}', got {image_stack.ndim}D.")
            if image_stack.size == 0: raise ValueError("Image stack empty.")
        except Exception as e: QMessageBox.critical(None, "Image Load Error", f"Failed to load TIFF: {file_loc}\n{e}\n{traceback.format_exc()}"); app_state.show_project_view_signal.emit(); return

        viewer = napari.Viewer(title=f"Segmentation: {os.path.basename(input_dir)} ({processing_mode.capitalize()} Mode)")
        qt_window_to_connect = viewer.window._qt_window if viewer and viewer.window and hasattr(viewer.window, '_qt_window') else None
        if qt_window_to_connect:
            try: qt_window_to_connect.destroyed.connect(_handle_napari_close); # print(f"Connected destroyed signal for: {qt_window_to_connect}") # Less verbose
            except Exception as connect_error: print(f"Warning: Failed to connect destroyed signal: {connect_error}")
        # else: print("Warning: Could not find Qt window for destroyed signal.") # Less verbose

        gui_manager = DynamicGUIManager(viewer, config, image_stack, file_loc, processing_mode)
        try: back_button_widget = create_back_to_project_button(viewer); viewer.window.add_dock_widget(back_button_widget, area="left", name="Navigation")
        except Exception as e: print(f"Error adding back button: {e}")

        @magicgui(call_button="▶ Next Step / Run Current")
        def continue_processing():
            parent_widget = viewer.window._qt_window if viewer and viewer.window and hasattr(viewer.window, '_qt_window') else None
            try: gui_manager.execute_processing_step(); update_navigation_buttons()
            except Exception as e: error_msg = f"Error during processing:\n{str(e)}\n{traceback.format_exc()}"; QMessageBox.critical(parent_widget, "Processing Error", error_msg); print(error_msg)

        @magicgui(call_button="◀ Previous Step")
        def go_to_previous_step():
            parent_widget = viewer.window._qt_window if viewer and viewer.window and hasattr(viewer.window, '_qt_window') else None
            current_step_index = gui_manager.current_step["value"]
            if current_step_index > 0:
                try:
                    gui_manager.cleanup_step(current_step_index); gui_manager.current_step["value"] -= 1
                    prev_step_name = gui_manager.processing_steps[gui_manager.current_step["value"]]
                    gui_manager.create_step_widgets(prev_step_name); update_navigation_buttons()
                except Exception as e: error_msg = f"Error going to previous step:\n{str(e)}\n{traceback.format_exc()}"; QMessageBox.critical(parent_widget, "Navigation Error", error_msg); print(error_msg)
            # else: print("Already at first step.") # Less verbose

        def update_navigation_buttons():
            try:
                current_idx = gui_manager.current_step["value"]; total_steps = len(gui_manager.processing_steps)
                if hasattr(go_to_previous_step, 'enabled'): go_to_previous_step.enabled = current_idx > 0
                if hasattr(continue_processing, 'enabled'): continue_processing.enabled = current_idx < total_steps
                if hasattr(continue_processing, 'label'):
                    if current_idx < total_steps:
                        step_name_key = gui_manager.processing_steps[current_idx]
                        display_name = gui_manager.step_display_names.get(step_name_key, step_name_key.replace("execute_","").replace("_"," ").title())
                        continue_processing.label = f"Run Step {current_idx+1}: {display_name}"
                    else: continue_processing.label = "Processing Complete"
            except Exception as e: print(f"Error updating nav buttons: {e}")

        step_widget_container = QWidget(); step_layout = QVBoxLayout(); step_widget_container.setLayout(step_layout)
        step_layout.addWidget(continue_processing.native); step_layout.addWidget(go_to_previous_step.native); step_layout.setContentsMargins(5,5,5,5)
        viewer.window.add_dock_widget(step_widget_container, area="left", name="Processing Control")
        update_navigation_buttons()

    except Exception as e:
        error_msg = f"Critical error setting up interactive segmentation:\n{str(e)}\n{traceback.format_exc()}"
        QMessageBox.critical(None, "Setup Error", error_msg); print(error_msg)
        if viewer is not None:
             try: viewer.close()
             except Exception as close_err: print(f"Error closing viewer after setup error: {close_err}")
        if QApplication.instance() and app_state: app_state.show_project_view_signal.emit()


def launch_image_segmentation_tool():
    app = None
    try:
        app = QApplication.instance()
        if app is None: app = QApplication(sys.argv if hasattr(sys, 'argv') else []); app.setQuitOnLastWindowClosed(False)
        def show_or_create_project_view():
            window_exists_py = app_state.project_view_window is not None; window_valid_qt = False
            if window_exists_py:
                 try: _ = app_state.project_view_window.isVisible(); window_valid_qt = True
                 except RuntimeError: app_state.project_view_window = None; window_exists_py = False; window_valid_qt = False
                 except Exception as e: print(f"Error checking project window: {e}"); app_state.project_view_window = None; window_exists_py = False; window_valid_qt = False
            create_new_window = not window_exists_py or not window_valid_qt
            if create_new_window:
                project_manager = ProjectManager()
                app_state.project_view_window = ProjectViewWindow(project_manager)
                app_state.project_view_window.setAttribute(Qt.WA_QuitOnClose, True)
            if app_state.project_view_window: app_state.project_view_window.show(); app_state.project_view_window.activateWindow(); app_state.project_view_window.raise_()
            else: raise RuntimeError("ProjectViewWindow is None after creation/check.")
        try: app_state.show_project_view_signal.disconnect(show_or_create_project_view)
        except TypeError: pass
        except Exception as e: print(f"Warning: Error disconnecting signal: {e}")
        app_state.show_project_view_signal.connect(show_or_create_project_view)
        show_or_create_project_view()
        return app
    except Exception as e:
        error_msg = (f"Unhandled critical error in launch_image_segmentation_tool:\n{str(e)}\n{traceback.format_exc()}")
        print(error_msg)
        try:
            if app is None: error_app = QApplication(sys.argv if hasattr(sys, 'argv') else []); QMessageBox.critical(None, "Fatal Launch Error", error_msg)
            else: QMessageBox.critical(None, "Fatal Launch Error", error_msg)
        except Exception as msg_err: print(f"Failed to display graphical error: {msg_err}")
        return None


def create_back_to_project_button(viewer):
    def _do_back_to_project():
        # print("Back to Project button clicked.") # Less verbose
        try:
            if viewer is not None: viewer.close(); # print("viewer.close() called.") # Less verbose
            # else: print("Viewer instance is None.") # Less verbose
            if app_state: app_state.show_project_view_signal.emit()
            # else: print("Error: ApplicationState instance not available.") # Less verbose
        except Exception as e:
             error_msg = f"Error during 'Back to Project':\n{str(e)}\n{traceback.format_exc()}"; QMessageBox.critical(None, "Navigation Error", error_msg); print(error_msg)
             if QApplication.instance() and app_state: app_state.show_project_view_signal.emit()
    button = QPushButton("Back to Project List"); button.clicked.connect(_do_back_to_project)
    container_widget = QWidget(); layout = QVBoxLayout(); layout.addWidget(button); layout.setContentsMargins(5, 5, 5, 5); container_widget.setLayout(layout)
    return container_widget
# --- END OF FILE utils/high_level_gui/helper_funcs.py ---