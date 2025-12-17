# --- START OF FILE utils/high_level_gui/helper_funcs.py ---
"""
Helper functions for the GUI, Project Management, and Batch Processing.
Handles:
1. Metadata Extraction (CZI, TIFF, OME-XML).
2. Project Organization (Splitting channels, creating folder structures).
3. Widget Creation for Napari.
4. Application State Management.
"""

from magicgui import magicgui # type: ignore
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import re
import pandas as pd
import napari # type: ignore
import sys
import yaml # type: ignore
import traceback
import tifffile as tiff # type: ignore
import shutil
import numpy as np
from xml.etree import ElementTree as ET
from PyQt5.QtGui import QCloseEvent # type: ignore
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer # type: ignore
from PyQt5.QtWidgets import ( # type: ignore
    QApplication, QFileDialog, QMessageBox,
    QMainWindow, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QWidget, QLabel, QInputDialog, QProgressDialog
)

# --- Optional Import for CZI Support ---
try:
    from aicspylibczi import CziFile
    HAS_CZI = True
except ImportError:
    HAS_CZI = False
    print("Warning: 'aicspylibczi' not installed. CZI support disabled.")

# --- Import BatchProcessor ---
try:
    from .batch_processor import BatchProcessor
except ImportError as e:
    print(f"WARNING: Failed to import BatchProcessor in helper_funcs.py: {e}. Batch processing button will be disabled.")
    BatchProcessor = None


def natural_sort_key(s):
    """Sorts strings containing numbers naturally (Image_2 before Image_10)."""
    basename = os.path.basename(s)
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', basename)]

# =============================================================================
# METADATA EXTRACTION UTILITIES
# =============================================================================

class MetadataExtractor:
    """Helper class to parse dimensions and physical scales from microscopy files."""

    @staticmethod
    def get_channel_count(path: str) -> int:
        """Determines number of channels in a file (CZI or TIFF)."""
        ext = os.path.splitext(path)[1].lower()
        
        if ext == '.czi' and HAS_CZI:
            try:
                czi = CziFile(path)
                # Check for get_dims_shape (v3.0+) or dims_shape (older)
                dims_list = czi.get_dims_shape() if hasattr(czi, 'get_dims_shape') else czi.dims_shape()
                if dims_list:
                    dims = dims_list[0]
                    if 'C' in dims:
                        return dims['C'][1] - dims['C'][0]
                return 1
            except: return 1
            
        elif ext in ['.tif', '.tiff']:
            try:
                with tiff.TiffFile(path) as tif:
                    # 1. Check OME Metadata string for SizeC attribute
                    if tif.ome_metadata:
                        match = re.search(r'SizeC="(\d+)"', str(tif.ome_metadata))
                        if match: return int(match.group(1))
                    
                    # 2. Check Array Shape heuristic
                    # tif.series[0].shape -> (T, Z, C, Y, X) or similar variations
                    shape = tif.series[0].shape
                    ndim = len(shape)
                    
                    # Heuristic: If 3D, usually ZYX (1 channel).
                    # If 4D, could be CZYX or ZCYX.
                    if ndim == 4:
                        # Assume the smaller dimension < 10 is channels
                        if shape[0] < shape[1] and shape[0] < 10: return shape[0]
                        if shape[1] < shape[0] and shape[1] < 10: return shape[1]
            except: return 1
            
        return 1

    @staticmethod
    def read_tiff_metadata(path: str) -> Dict[str, float]:
        """Attempts to read physical scale (microns) from TIFF tags or OME-XML."""
        meta = {'x': 1.0, 'y': 1.0, 'z': 1.0, 'found': False}
        try:
            with tiff.TiffFile(path) as tif:
                # A. Try Standard TIFF Resolution Tags
                page = tif.pages[0]
                x_res = page.tags.get('XResolution')
                y_res = page.tags.get('YResolution')
                unit = page.tags.get('ResolutionUnit') 
                
                if x_res and y_res and unit:
                    x_val = x_res.value
                    y_val = y_res.value
                    u_val = unit.value
                    
                    # Normalize fraction tuples
                    x_dens = x_val[0] / x_val[1] if isinstance(x_val, tuple) else x_val
                    y_dens = y_val[0] / y_val[1] if isinstance(y_val, tuple) else y_val
                    
                    if x_dens > 0 and y_dens > 0:
                        if u_val == 2: # Inch -> Micron
                            meta['x'] = 25400.0 / x_dens
                            meta['y'] = 25400.0 / y_dens
                            meta['found'] = True
                        elif u_val == 3: # Centimeter -> Micron
                            meta['x'] = 10000.0 / x_dens
                            meta['y'] = 10000.0 / y_dens
                            meta['found'] = True
                        
                # B. Try OME-XML (Overrides standard tags if present)
                if tif.ome_metadata:
                    txt = str(tif.ome_metadata)
                    def extract_attr(name):
                        # Regex to find PhysicalSizeX="0.5"
                        match = re.search(rf'{name}="([\d\.]+)"', txt)
                        return float(match.group(1)) if match else None
                    
                    px = extract_attr("PhysicalSizeX")
                    py = extract_attr("PhysicalSizeY")
                    pz = extract_attr("PhysicalSizeZ")
                    
                    if px: meta['x'] = px
                    if py: meta['y'] = py
                    if pz: meta['z'] = pz
                    if px or py: meta['found'] = True

        except Exception as e:
            print(f"Metadata read error for {os.path.basename(path)}: {e}")
        
        return meta

    @staticmethod
    def _parse_czi_xml_scaling(xml_input) -> Dict[str, float]:
        """Parses CZI XML object/string to find scaling in MICRONS."""
        scales = {}
        try:
            root = None
            # Handle different XML representations (String vs ElementTree)
            if hasattr(xml_input, 'getroot'): root = xml_input.getroot()
            elif ET.iselement(xml_input): root = xml_input
            elif isinstance(xml_input, (str, bytes)):
                try:
                    if len(str(xml_input)) < 255 and os.path.exists(xml_input): 
                        root = ET.parse(xml_input).getroot()
                    else: 
                        root = ET.fromstring(xml_input)
                except: pass
            
            if root is not None:
                # Standard CZI Path: Scaling -> Items -> Distance -> Value
                for dist in root.iter('Distance'):
                    axis_id = dist.get('Id')
                    val_node = dist.find('Value')
                    if axis_id and val_node is not None and val_node.text:
                        try:
                            # CZI stores Meters. Convert to Microns.
                            scales[axis_id] = float(val_node.text) * 1e6
                        except: pass
        except Exception as e:
            print(f"    Error parsing CZI XML: {e}")
        return scales

    @staticmethod
    def extract_channel_to_tiff(src_path: str, dest_path: str, channel_idx: int):
        """Extracts a specific channel from CZI or Multi-Channel TIFF."""
        ext = os.path.splitext(src_path)[1].lower()
        
        if ext == '.czi' and HAS_CZI:
            czi = CziFile(src_path)
            # read_image(C=...) returns (data, shape_list)
            img, _ = czi.read_image(C=channel_idx)
            img = np.squeeze(img) # Remove singleton dimensions
            tiff.imwrite(dest_path, img, photometric='minisblack')
            
        elif ext in ['.tif', '.tiff']:
            vol = tiff.imread(src_path)
            
            # Handle Dimensions for TIFF
            # Heuristic: If 4D, determine which axis is Channel
            ch_data = vol
            if vol.ndim == 4:
                # Assume smaller dimension is Channel
                if vol.shape[0] < vol.shape[1]: 
                    ch_data = vol[channel_idx] # Format: C, Z, Y, X
                else: 
                    ch_data = vol[:, channel_idx, :, :] # Format: Z, C, Y, X
            elif vol.ndim == 3:
                # If 3D, assume single channel ZYX.
                # If user requested ch > 0, this might fail or imply RGB.
                if channel_idx == 0: ch_data = vol
                else: 
                    # RGB case? (Y, X, C) or (Z, Y, X)
                    # This is ambiguous in TIFFs. Fallback to whole volume.
                    pass 
            
            tiff.imwrite(dest_path, ch_data, photometric='minisblack')

    @staticmethod
    def get_czi_metadata(path: str) -> Dict[str, float]:
        """Wrapper to get metadata specifically for CZI."""
        if not HAS_CZI: return {'x':1.0, 'y':1.0, 'z':1.0, 'found':False}
        czi = CziFile(path)
        scale_map = {}
        
        # Method 1: Property
        if hasattr(czi, 'pixel_scaling'):
            try: scale_map = {k: v * 1e6 for k, v in czi.pixel_scaling.items()} 
            except: pass
        
        # Method 2: XML Parsing
        if not scale_map and hasattr(czi, 'meta'):
            xml = czi.meta() if callable(czi.meta) else czi.meta
            scale_map = MetadataExtractor._parse_czi_xml_scaling(xml)
            
        return {
            'x': scale_map.get('X', 1.0),
            'y': scale_map.get('Y', 1.0),
            'z': scale_map.get('Z', 1.0),
            'found': bool(scale_map)
        }

# =============================================================================
# GUI PARAMETER WIDGETS
# =============================================================================

def create_parameter_widget(param_name: str, param_config: Dict[str, Any], callback):
    """Creates a MagicGUI widget for a specific parameter definition."""
    param_type = param_config.get("type", "float")
    label = param_config.get("label", param_name)
    widget = None
    try:
        if param_type == "list":
            initial_list = param_config.get("value", [])
            if not isinstance(initial_list, list): initial_list = []
            initial_str = ", ".join(map(str, initial_list))
            def list_widget(value_str: str = initial_str):
                try:
                    new_list = [float(x.strip()) for x in value_str.split(',') if x.strip()] if value_str.strip() else []
                    callback(new_list)
                    if hasattr(list_widget, 'native'): list_widget.native.setStyleSheet("")
                    return value_str
                except ValueError:
                    if hasattr(list_widget, 'native'): list_widget.native.setStyleSheet("background-color: #FFDDDD;")
                    return initial_str
            widget = magicgui(list_widget, auto_call=True, value_str={"widget_type": "LineEdit", "label": label})
        elif param_type == "float":
            def float_widget(value: float = float(param_config.get("value", 0.0))): callback(value); return value
            widget = magicgui(float_widget, auto_call=True, value={"widget_type": "FloatSpinBox", "label": label, "min": float(param_config.get("min", 0)), "max": float(param_config.get("max", 100)), "step": float(param_config.get("step", 0.1))})
        elif param_type == "int":
            def int_widget(value: int = int(param_config.get("value", 0))): callback(value); return value
            widget = magicgui(int_widget, auto_call=True, value={"widget_type": "SpinBox", "label": label, "min": int(param_config.get("min", 0)), "max": int(param_config.get("max", 100)), "step": int(param_config.get("step", 1))})
        elif param_type == "bool":
             def bool_widget(value: bool = bool(param_config.get("value", False))): callback(value); return value
             widget = magicgui(bool_widget, auto_call=True, value={"widget_type": "CheckBox", "label": label})
        else:
            def fallback(value: str = str(param_config.get("value", ""))): callback(value); return value
            widget = magicgui(fallback, auto_call=True, value={"widget_type": "LineEdit", "label": label})
        if widget: widget.param_name = param_name
    except Exception: return None
    return widget

def scan_available_presets():
    """Scans the module directories for available YAML configuration presets."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_locations = [
        (os.path.join(script_dir, '..', 'module_3d', 'configs'), 'ramified'),
        (os.path.join(script_dir, '..', 'module_2d', 'configs'), 'ramified_2d')
    ]
    presets = {}
    for config_dir, default_mode in search_locations:
        if not os.path.exists(config_dir): continue
        try:
            files = [f for f in os.listdir(config_dir) if f.lower().endswith(('.yaml', '.yml'))]
            for f in files:
                full_path = os.path.join(config_dir, f)
                clean_name = os.path.splitext(f)[0].replace('_', ' ').title()
                suffix = " (2D)" if "module_2d" in config_dir else " (3D)"
                presets[f"{clean_name}{suffix}"] = {"path": full_path, "default_mode": default_mode}
        except Exception: pass
    return presets

# =============================================================================
# PROJECT ORGANIZATION LOGIC (SINGLE & MULTI-CHANNEL)
# =============================================================================

def organize_channel_project(source_files, source_root, target_root_dir, channel_idx, preset_details):
    """
    Setup Logic for MULTI-CHANNEL mode.
    Creates a new project folder (target_root_dir) and populates it with 
    extracted single-channel TIFFs from the source files.
    """
    config_template_path = preset_details['path']
    fallback_mode = preset_details['default_mode']
    
    print(f"  Organizing Channel {channel_idx} into: {target_root_dir}")
    os.makedirs(target_root_dir, exist_ok=True)
    
    # Load Template
    with open(config_template_path, 'r') as f: template_data = yaml.safe_load(f) or {}
    mode = template_data.get('mode', fallback_mode)
    is_2d_mode = mode.endswith('_2d')
    dimension_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'

    metadata_rows = []

    for src_file in source_files:
        src_path = os.path.join(source_root, src_file)
        basename = os.path.splitext(src_file)[0]
        
        # Subdirectory per image (Standard Format)
        img_subdir = os.path.join(target_root_dir, basename)
        os.makedirs(img_subdir, exist_ok=True)
        
        target_tif_name = f"{basename}.tif"
        target_tif_path = os.path.join(img_subdir, target_tif_name)
        
        # 1. Extract Data
        print(f"    Processing {src_file}...")
        try:
            MetadataExtractor.extract_channel_to_tiff(src_path, target_tif_path, channel_idx)
        except Exception as e:
            print(f"    Error extracting channel {channel_idx} from {src_file}: {e}")
            continue

        # 2. Extract Metadata (Scale)
        # We read metadata from the SOURCE file to get original scaling
        if src_file.lower().endswith('.czi'):
            meta = MetadataExtractor.get_czi_metadata(src_path)
        else:
            meta = MetadataExtractor.read_tiff_metadata(src_path)
        
        if meta['found']:
            print(f"      Scale detected: {meta['x']:.3f}, {meta['y']:.3f}, {meta['z']:.3f}")

        # 3. Get Image Dimensions (Shape)
        try:
            mem = tiff.imread(target_tif_path)
            shape = mem.shape
            z_slices = shape[0] if mem.ndim == 3 else 1
            width = shape[-1]
            height = shape[-2]
            del mem
        except: 
            z_slices, width, height = 1, 1, 1

        metadata_rows.append({
            'Filename': target_tif_name,
            'Width (um)': meta['x'] * width,
            'Height (um)': meta['y'] * height,
            'Depth (um)': meta['z'] * z_slices,
            'Slices': z_slices
        })

        # 4. Write Config
        new_config_path = os.path.join(img_subdir, os.path.basename(config_template_path))
        if not os.path.exists(new_config_path):
            shutil.copy2(config_template_path, new_config_path)
        
        try:
            with open(new_config_path, 'r') as f: cfg = yaml.safe_load(f) or {}
            if dimension_key not in cfg: cfg[dimension_key] = {}
            cfg[dimension_key]['x'] = float(meta['x'] * width)
            cfg[dimension_key]['y'] = float(meta['y'] * height)
            if not is_2d_mode: 
                cfg[dimension_key]['z'] = float(meta['z'] * z_slices)
            cfg['mode'] = mode
            with open(new_config_path, 'w') as f: yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        except: pass

    # 5. Write CSV
    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        df.sort_values('Filename', key=lambda col: col.map(natural_sort_key), inplace=True)
        csv_path = os.path.join(target_root_dir, "metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"    Saved metadata summary to {csv_path}")

def organize_processing_dir(drctry, preset_details):
    """
    Setup Logic for SINGLE-CHANNEL / LEGACY mode.
    Organizes a flat folder of TIFFs + CSV into project subfolders.
    Auto-generates CSV/Configs if missing.
    """
    config_template_path = preset_details['path']
    fallback_mode = preset_details['default_mode']
    
    print(f"Organizing Standard Project in: {drctry}")
    
    all_files = sorted(os.listdir(drctry))
    raw_images = [f for f in all_files if f.lower().endswith(('.tif', '.tiff'))]
    csv_files = [f for f in all_files if f.lower().endswith('.csv')]
    
    if not raw_images:
        raise ValueError('No .tif or .tiff files found.')

    # 1. Handle DataFrame
    df = None
    if len(csv_files) == 1:
        print("  Found existing metadata CSV.")
        df = pd.read_csv(os.path.join(drctry, csv_files[0]))
        df['Basename'] = df['Filename'].apply(lambda x: os.path.splitext(str(x))[0])
    elif len(csv_files) > 1:
        raise ValueError("Multiple CSV files found. Please keep only one.")
    else:
        print("  No CSV found. Auto-generating metadata from files...")
        df = pd.DataFrame(columns=['Filename', 'Width (um)', 'Height (um)', 'Depth (um)', 'Slices'])

    # 2. Config Template
    with open(config_template_path, 'r') as f: template_data = yaml.safe_load(f) or {}
    mode = template_data.get('mode', fallback_mode)
    is_2d_mode = mode.endswith('_2d')
    dimension_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'

    generated_rows = []

    # 3. Generate Metadata (if needed)
    if 'Basename' not in df.columns:
        for img_file in raw_images:
            full_path = os.path.join(drctry, img_file)
            basename = os.path.splitext(img_file)[0]
            
            print(f"  Analyzing: {img_file}")
            meta = MetadataExtractor.read_tiff_metadata(full_path)
            
            try:
                mem = tiff.imread(full_path)
                z_slices = mem.shape[0] if mem.ndim == 3 else 1
                spacing_x = meta['x'] if meta['found'] else 1.0
                spacing_y = meta['y'] if meta['found'] else 1.0
                spacing_z = meta['z'] if meta['found'] else 1.0
                
                generated_rows.append({
                    'Filename': img_file,
                    'Width (um)': spacing_x * (mem.shape[-1]),
                    'Height (um)': spacing_y * (mem.shape[-2]),
                    'Depth (um)': spacing_z * z_slices,
                    'Slices': z_slices,
                    'Basename': basename
                })
                del mem
            except Exception: pass

        if generated_rows:
            df = pd.DataFrame(generated_rows)
            df.drop(columns=['Basename'], errors='ignore').to_csv(os.path.join(drctry, "auto_generated_metadata.csv"), index=False)
            print("  Saved 'auto_generated_metadata.csv'.")
    
    if 'Basename' not in df.columns:
         df['Basename'] = df['Filename'].apply(lambda x: os.path.splitext(str(x))[0])

    # 4. Create Folder Structure
    for _, row in df.iterrows():
        root_name = row['Basename']
        
        found_file = None
        for f in os.listdir(drctry):
            if os.path.isfile(os.path.join(drctry, f)):
                if os.path.splitext(f)[0] == root_name and f.lower().endswith(('.tif', '.tiff')):
                    found_file = f
                    break
        
        if not found_file: continue

        new_dir = os.path.join(drctry, root_name)
        os.makedirs(new_dir, exist_ok=True)
        
        src = os.path.join(drctry, found_file)
        dst = os.path.join(new_dir, found_file)
        
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.move(src, dst)

        new_config_path = os.path.join(new_dir, os.path.basename(config_template_path))
        if not os.path.exists(new_config_path):
            shutil.copy2(config_template_path, new_config_path)
            
        try:
            with open(new_config_path, 'r') as f: cfg = yaml.safe_load(f) or {}
            if dimension_key not in cfg: cfg[dimension_key] = {}
            cfg[dimension_key]['x'] = float(row['Width (um)'])
            cfg[dimension_key]['y'] = float(row['Height (um)'])
            if not is_2d_mode: 
                cfg[dimension_key]['z'] = float(row['Depth (um)']) if 'Depth (um)' in row else 0.0
            cfg['mode'] = mode
            with open(new_config_path, 'w') as f: yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        except: pass

    print("Standard Organization complete.")

# =============================================================================
# APP STATE & MANAGER
# =============================================================================

class ApplicationState(QObject):
    show_project_view_signal = pyqtSignal()
    _instance = None
    project_view_window = None
    def __new__(cls):
        if cls._instance is None: cls._instance = super(ApplicationState, cls).__new__(cls); cls._instance.__initialized = False
        return cls._instance
    def __init__(self):
        if not getattr(self, '_ApplicationState__initialized', False): super().__init__(); self.__initialized = True
app_state = ApplicationState()

class ProjectManager:
    def __init__(self): self.project_path = None; self.image_folders = []
    def select_project_folder(self):
        self.project_path = QFileDialog.getExistingDirectory(None, "Select Project Root Folder", "")
        if self.project_path: self._find_valid_image_folders()
        return self.project_path
    def _find_valid_image_folders(self):
        self.image_folders = []
        if not self.project_path or not os.path.isdir(self.project_path): return
        try:
            for item in os.listdir(self.project_path):
                potential_folder_path = os.path.join(self.project_path, item)
                if os.path.isdir(potential_folder_path):
                    try:
                        folder_contents = os.listdir(potential_folder_path)
                        tif_files = [f for f in folder_contents if f.lower().endswith(('.tif', '.tiff'))]
                        yaml_files = [f for f in folder_contents if f.lower().endswith(('.yaml', '.yml'))]
                        if len(tif_files) == 1 and len(yaml_files) == 1: self.image_folders.append(potential_folder_path)
                    except OSError: pass
        except OSError: self.image_folders = []
        self.image_folders.sort(key=natural_sort_key)
    def get_image_details(self, folder_path):
        try:
            contents = os.listdir(folder_path)
            tif_file = next((f for f in contents if f.lower().endswith(('.tif', '.tiff'))), None)
            yaml_file = next((f for f in contents if f.lower().endswith(('.yaml', '.yml'))), None)
            if not tif_file or not yaml_file: return {'path': folder_path, 'mode': 'error'}
            config = {}
            try:
                with open(os.path.join(folder_path, yaml_file), 'r') as file: config = yaml.safe_load(file) or {}
            except Exception: pass
            return {'path': folder_path, 'tif_file': tif_file, 'yaml_file': yaml_file, 'mode': config.get('mode', 'unknown')}
        except Exception: return {'path': folder_path, 'mode': 'error'}

class ProjectViewWindow(QMainWindow):
    def __init__(self, project_manager):
        super().__init__(); self.project_manager = project_manager; self.initUI(); self.setAttribute(Qt.WA_QuitOnClose)
    def initUI(self):
        self.setWindowTitle("Image Segmentation Project"); self.setGeometry(100, 100, 700, 450)
        central_widget = QWidget(); layout = QVBoxLayout()
        self.project_path_label = QLabel("Project Path: Not Selected"); layout.addWidget(self.project_path_label)
        self.image_list = QListWidget(); self.image_list.itemDoubleClicked.connect(self.open_image_view); layout.addWidget(self.image_list)
        button_layout = QHBoxLayout()
        select_project_btn = QPushButton("Select/Load Project Folder"); select_project_btn.clicked.connect(self.load_project); button_layout.addWidget(select_project_btn)
        self.batch_process_all_btn = QPushButton("Process All Compatible Folders"); self.batch_process_all_btn.clicked.connect(self.run_batch_processing_all_compatible); self.batch_process_all_btn.setEnabled(False)
        button_layout.addWidget(self.batch_process_all_btn)
        layout.addLayout(button_layout); central_widget.setLayout(layout); self.setCentralWidget(central_widget)
    def _update_batch_button_state(self):
        if not BatchProcessor or not self.project_manager.image_folders: self.batch_process_all_btn.setEnabled(False); return
        self.batch_process_all_btn.setEnabled(True)
    def load_project(self):
        selected_path = self.project_manager.select_project_folder()
        if not selected_path: return
        self.project_path_label.setText(f"Project Path: {selected_path}"); self.image_list.clear()
        try:
            # Check for raw files that might need organization
            raw_files = [f for f in os.listdir(selected_path) if f.lower().endswith(('.tif', '.tiff', '.czi')) and os.path.isfile(os.path.join(selected_path, f))]
            csv_files = [f for f in os.listdir(selected_path) if f.lower().endswith('.csv')]
            
            # --- Logic to distinguish New Project vs Existing Project ---
            needs_organization = False
            is_multi_channel = False
            
            # If we have CZI, definitely multi-channel setup
            if any(f.endswith('.czi') for f in raw_files):
                needs_organization = True
                is_multi_channel = True
            
            # If we have TIFFs:
            elif raw_files:
                # If valid CSV exists, check if TIFFs are already in subfolders (Done) or flat (Needs Org)
                # But here we are scanning the ROOT. If TIFFs are in Root, they need org.
                
                # Check if they are Multi-Channel TIFFs (Hyperstacks)
                first_tif = os.path.join(selected_path, raw_files[0])
                if MetadataExtractor.get_channel_count(first_tif) > 1:
                    needs_organization = True
                    is_multi_channel = True
                else:
                    # Single channel TIFFs in root -> Needs Standard Org
                    needs_organization = True
                    is_multi_channel = False
            
            if needs_organization:
                msg = "Setup multi-channel project structure?" if is_multi_channel else "Organize single-channel project?"
                reply = QMessageBox.question(self, "Setup Project?", f"Found {len(raw_files)} raw images.\n{msg}", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                
                if reply == QMessageBox.Yes:
                    presets = scan_available_presets()
                    if not presets: QMessageBox.critical(self, "Error", "No config presets found."); return
                    
                    if is_multi_channel:
                        # Multi-Channel Logic
                        max_channels = 1
                        for f in raw_files:
                            max_channels = max(max_channels, MetadataExtractor.get_channel_count(os.path.join(selected_path, f)))
                        print(f"Detected {max_channels} channels max.")
                        
                        for ch in range(max_channels):
                            preset_key, ok = QInputDialog.getItem(self, f"Channel {ch} Configuration", f"Select Preset for Channel {ch}:", sorted(list(presets.keys())), 0, False)
                            if ok and preset_key:
                                target_dir = os.path.join(selected_path, f"Channel_{ch}_{preset_key.split()[0]}")
                                QApplication.setOverrideCursor(Qt.WaitCursor)
                                try: organize_channel_project(raw_files, selected_path, target_dir, ch, presets[preset_key])
                                except Exception as e: QMessageBox.critical(self, "Error", f"Failed Ch{ch}: {e}")
                                finally: QApplication.restoreOverrideCursor()
                        QMessageBox.information(self, "Done", "Project setup complete. Load specific Channel folder.")
                        return
                    else:
                        # Standard/Legacy Logic
                        preset_key, ok = QInputDialog.getItem(self, "Select Preset", "Choose configuration:", sorted(list(presets.keys())), 0, False)
                        if ok and preset_key:
                            QApplication.setOverrideCursor(Qt.WaitCursor)
                            try: organize_processing_dir(selected_path, presets[preset_key])
                            except Exception as e: QMessageBox.critical(self, "Error", f"Failed: {e}")
                            finally: QApplication.restoreOverrideCursor()
                        # Continue to load...

            # Standard Load (Subfolders)
            self.project_manager._find_valid_image_folders()
            
            for folder_path in self.project_manager.image_folders:
                details = self.project_manager.get_image_details(folder_path)
                item = QListWidgetItem(f"{os.path.basename(folder_path)} - Mode: {details.get('mode')}"); item.setData(Qt.UserRole, folder_path); self.image_list.addItem(item)
        except Exception as e: QMessageBox.critical(self, "Error", str(e))
        self._update_batch_button_state()
    def open_image_view(self, item):
        folder = item.data(Qt.UserRole)
        if folder: self.hide(); interactive_segmentation_with_config(folder)
    def run_batch_processing_all_compatible(self):
        if not self.batch_process_all_btn.isEnabled(): return
        reply = QMessageBox.question(self, "Confirm", "Process all folders?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            processor = BatchProcessor(self.project_manager)
            processor.process_all_folders(force_restart_all=False)
            QMessageBox.information(self, "Done", "Batch processing complete.")
    def closeEvent(self, event: QCloseEvent):
        if QMessageBox.question(self, 'Exit', "Exit application?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            QApplication.instance().quit(); event.accept()
        else: event.ignore()

def _check_if_last_window():
    app = QApplication.instance()
    if not app: return
    pv = app_state.project_view_window
    valid = False
    try: valid = pv.isVisible()
    except: pass
    if not valid: app.quit()

def _handle_napari_close():
    QTimer.singleShot(100, _check_if_last_window)

def interactive_segmentation_with_config(selected_folder=None):
    try: from .gui_manager import DynamicGUIManager
    except ImportError: return
    app = QApplication.instance() or QApplication(sys.argv)
    viewer = None
    try:
        if not selected_folder: raise ValueError("No folder.")
        contents = os.listdir(selected_folder)
        tif = next((f for f in contents if f.lower().endswith(('.tif', '.tiff'))), None)
        yml = next((f for f in contents if f.lower().endswith(('.yaml', '.yml'))), None)
        if not tif or not yml: raise FileNotFoundError("Missing TIF/YAML.")
        
        file_loc = os.path.join(selected_folder, tif)
        with open(os.path.join(selected_folder, yml), 'r') as f: config = yaml.safe_load(f)
        mode = config.get('mode')
        
        image_stack = tiff.imread(file_loc)
        viewer = napari.Viewer(title=f"Segmentation: {os.path.basename(selected_folder)}")
        
        qt_window = viewer.window._qt_window
        qt_window.destroyed.connect(_handle_napari_close)

        gui_manager = DynamicGUIManager(viewer, config, image_stack, file_loc, mode)
        viewer.window.add_dock_widget(create_back_to_project_button(viewer), area="left", name="Navigation")

        @magicgui(call_button="▶ Next Step / Run Current")
        def continue_processing():
            gui_manager.execute_processing_step()

        @magicgui(call_button="◀ Previous Step")
        def go_to_previous_step():
            idx = gui_manager.current_step["value"]
            if idx > 0:
                gui_manager.cleanup_step(idx)
                gui_manager.current_step["value"] -= 1
                gui_manager.create_step_widgets(gui_manager.processing_steps[gui_manager.current_step["value"]])
                update_navigation_buttons()

        def update_navigation_buttons():
            idx = gui_manager.current_step["value"]
            total = len(gui_manager.processing_steps)
            go_to_previous_step.enabled = (idx > 0)
            continue_processing.enabled = (idx < total)
            if idx < total:
                step_name = gui_manager.processing_steps[idx]
                continue_processing.label = f"Run Step {idx+1}: {gui_manager.step_display_names.get(step_name, step_name)}"
            else:
                continue_processing.label = "Processing Complete"

        def disable_buttons_during_process():
            continue_processing.enabled = False
            go_to_previous_step.enabled = False
            continue_processing.label = "Processing... (Please Wait)"

        gui_manager.process_started.connect(disable_buttons_during_process)
        gui_manager.process_finished.connect(update_navigation_buttons)

        container = QWidget(); l = QVBoxLayout(); container.setLayout(l)
        l.addWidget(continue_processing.native); l.addWidget(go_to_previous_step.native); l.setContentsMargins(5,5,5,5)
        viewer.window.add_dock_widget(container, area="left", name="Processing Control")
        update_navigation_buttons()

    except Exception as e:
        QMessageBox.critical(None, "Error", str(e)); app_state.show_project_view_signal.emit()

def launch_image_segmentation_tool():
    app = QApplication.instance() or QApplication(sys.argv)
    def show_pv():
        if not app_state.project_view_window:
            app_state.project_view_window = ProjectViewWindow(ProjectManager())
        app_state.project_view_window.show(); app_state.project_view_window.raise_()
    app_state.show_project_view_signal.connect(show_pv); show_pv()
    return app

def create_back_to_project_button(viewer):
    def _do():
        if viewer: viewer.close()
        app_state.show_project_view_signal.emit()
    btn = QPushButton("Back to Project List"); btn.clicked.connect(_do)
    w = QWidget(); l = QVBoxLayout(); w.setLayout(l); l.addWidget(btn); l.setContentsMargins(5,5,5,5)
    return w
# --- END OF FILE utils/high_level_gui/helper_funcs.py ---