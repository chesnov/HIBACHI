import os
import re
import sys
import shutil
import time
import traceback
import yaml  # type: ignore
from xml.etree import ElementTree as ET
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

import numpy as np
import pandas as pd
import tifffile as tiff  # type: ignore
import napari  # type: ignore
from magicgui import magicgui  # type: ignore
from PyQt5.QtGui import QCloseEvent  # type: ignore
from PyQt5.QtCore import Qt, QObject, pyqtSignal, QTimer  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QFileDialog, QMessageBox,
    QMainWindow, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton,
    QWidget, QLabel, QInputDialog
)

# --- Optional Import for CZI Support ---
try:
    from aicspylibczi import CziFile  # type: ignore
    HAS_CZI = True
except ImportError:
    HAS_CZI = False
    print("Warning: 'aicspylibczi' not installed. CZI support disabled.")

# --- Import BatchProcessor ---
try:
    from .batch_processor import BatchProcessor
except ImportError as e:
    print(f"WARNING: Failed to import BatchProcessor: {e}. "
          "Batch processing button will be disabled.")
    BatchProcessor = None  # type: ignore


def natural_sort_key(s: str) -> List[Union[int, str]]:
    """
    Sorts strings containing numbers naturally (e.g., Image_2 before Image_10).
    """
    basename = os.path.basename(s)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split('([0-9]+)', basename)
    ]


# =============================================================================
# METADATA EXTRACTION UTILITIES
# =============================================================================

class MetadataExtractor:
    """Helper class to parse dimensions and physical scales from microscopy files."""

    @staticmethod
    def get_channel_count(path: str) -> int:
        """Determines the number of channels in a file (CZI or TIFF)."""
        ext = os.path.splitext(path)[1].lower()

        if ext == '.czi' and HAS_CZI:
            try:
                czi = CziFile(path)
                dims_list = (
                    czi.get_dims_shape()
                    if hasattr(czi, 'get_dims_shape') else czi.dims_shape()
                )
                if dims_list:
                    dims = dims_list[0]
                    if 'C' in dims:
                        return dims['C'][1] - dims['C'][0]
                return 1
            except Exception:
                return 1

        elif ext in ['.tif', '.tiff']:
            try:
                with tiff.TiffFile(path) as tif:
                    # 1. Check OME Metadata
                    if tif.ome_metadata:
                        match = re.search(r'SizeC="(\d+)"', str(tif.ome_metadata))
                        if match:
                            return int(match.group(1))
                    
                    # 2. Check ImageJ Metadata
                    if tif.imagej_metadata:
                        ij_channels = tif.imagej_metadata.get('channels')
                        if ij_channels:
                            return int(ij_channels)

                    # 3. Heuristic based on shape
                    if len(tif.series) > 0:
                        shape = tif.series[0].shape
                        ndim = len(shape)
                        if ndim == 4: # (C, Z, Y, X) or (Z, C, Y, X)
                            return min(shape[0], shape[1]) 
                        if ndim == 3: # (C, Y, X) for 2D or (Z, Y, X) for 1ch-3D
                            # If the first dimension is very small, it's likely Channels
                            if shape[0] < 10 and shape[0] < shape[1] and shape[0] < shape[2]:
                                return shape[0]
            except Exception:
                return 1
        return 1

    @staticmethod
    def read_tiff_metadata(path: str) -> Dict[str, Union[float, bool]]:
        """Attempts to read physical scale (microns) from TIFF tags, ImageJ, or OME-XML."""
        meta: Dict[str, Union[float, bool]] = {
            'x': 1.0, 'y': 1.0, 'z': 1.0, 'found': False
        }
        try:
            with tiff.TiffFile(path) as tif:
                # 1. Try ImageJ Resolution (Common for 2D TIFs)
                if tif.imagej_metadata:
                    ij = tif.imagej_metadata
                    # ImageJ stores spacing in 'spacing' or 'unit'
                    unit = ij.get('unit')
                    if unit in ['micron', 'µm', 'um']:
                        if 'spacing' in ij:
                            meta['z'] = float(ij['spacing'])
                        meta['found'] = True

                # 2. Try Standard TIFF Tags
                if tif.pages:
                    page = tif.pages[0]
                    x_res = page.tags.get('XResolution')
                    y_res = page.tags.get('YResolution')
                    unit = page.tags.get('ResolutionUnit')

                    if x_res and y_res and unit:
                        x_val = x_res.value
                        y_val = y_res.value
                        u_val = unit.value
                        x_dens = x_val[0] / x_val[1] if isinstance(x_val, tuple) else x_val
                        y_dens = y_val[0] / y_val[1] if isinstance(y_val, tuple) else y_val

                        if x_dens > 0 and y_dens > 0:
                            if u_val == 2:  # Inch -> Micron
                                meta['x'] = 25400.0 / x_dens
                                meta['y'] = 25400.0 / y_dens
                                meta['found'] = True
                            elif u_val == 3:  # Centimeter -> Micron
                                meta['x'] = 10000.0 / x_dens
                                meta['y'] = 10000.0 / y_dens
                                meta['found'] = True

                # 3. Try OME Metadata (Most robust for modern Bio-Formats/Zen)
                if tif.ome_metadata:
                    txt = str(tif.ome_metadata)
                    def extract_attr(name: str) -> Optional[float]:
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
    def _parse_czi_xml_scaling(xml_input: Any) -> Dict[str, float]:
        """Parses CZI XML object/string to find scaling in MICRONS."""
        scales = {}
        try:
            root = None
            if hasattr(xml_input, 'getroot'):
                root = xml_input.getroot()
            elif ET.iselement(xml_input):
                root = xml_input
            elif isinstance(xml_input, (str, bytes)):
                try:
                    if len(str(xml_input)) < 255 and os.path.exists(xml_input):
                        root = ET.parse(xml_input).getroot()
                    else:
                        root = ET.fromstring(xml_input)
                except Exception:
                    pass

            if root is not None:
                for dist in root.iter('Distance'):
                    axis_id = dist.get('Id')
                    val_node = dist.find('Value')
                    if axis_id and val_node is not None and val_node.text:
                        try:
                            scales[axis_id] = float(val_node.text) * 1e6
                        except ValueError:
                            pass
        except Exception as e:
            print(f"    Error parsing CZI XML: {e}")
        return scales

    @staticmethod
    def extract_channel_to_tiff(
        src_path: str, dest_path: str, channel_idx: int
    ) -> None:
        """Extracts a specific channel from CZI or Multi-Channel TIFF."""
        ext = os.path.splitext(src_path)[1].lower()

        if ext == '.czi' and HAS_CZI:
            czi = CziFile(src_path)
            img, _ = czi.read_image(C=channel_idx)
            img = np.squeeze(img)
            tiff.imwrite(dest_path, img, photometric='minisblack')

        elif ext in ['.tif', '.tiff']:
            vol = tiff.imread(src_path)
            ch_data = vol
            if vol.ndim == 4:
                # Guess (C, Z, Y, X) vs (Z, C, Y, X)
                if vol.shape[0] < vol.shape[1]:
                    ch_data = vol[channel_idx]
                else:
                    ch_data = vol[:, channel_idx, :, :]
            elif vol.ndim == 3:
                # For 2D multi-channel (C, Y, X)
                # If we are extracting channel index > 0, we must slice
                if vol.shape[0] < vol.shape[1] and vol.shape[0] < vol.shape[2]:
                    ch_data = vol[channel_idx]
                else:
                    # It's likely a single channel 3D stack (Z, Y, X)
                    ch_data = vol
            
            tiff.imwrite(dest_path, ch_data, photometric='minisblack')

    @staticmethod
    def get_czi_metadata(path: str) -> Dict[str, Union[float, bool]]:
        """Wrapper to get metadata specifically for CZI files."""
        if not HAS_CZI:
            return {'x': 1.0, 'y': 1.0, 'z': 1.0, 'found': False}
        czi = CziFile(path)
        scale_map = {}
        if hasattr(czi, 'pixel_scaling'):
            try:
                scale_map = {k: v * 1e6 for k, v in czi.pixel_scaling.items()}
            except Exception:
                pass
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

def create_parameter_widget(
    param_name: str,
    param_config: Dict[str, Any],
    callback: Callable[[Any], None]
) -> Optional[Any]:
    """Creates a MagicGUI widget for a specific parameter definition."""
    param_type = param_config.get("type", "float")
    label = param_config.get("label", param_name)
    widget = None

    try:
        if param_type == "list":
            initial_list = param_config.get("value", [])
            if not isinstance(initial_list, list):
                initial_list = []
            initial_str = ", ".join(map(str, initial_list))

            def list_widget(value_str: str = initial_str):
                try:
                    new_list = [
                        float(x.strip()) for x in value_str.split(',') if x.strip()
                    ] if value_str.strip() else []
                    callback(new_list)
                    if hasattr(list_widget, 'native'):
                        list_widget.native.setStyleSheet("")
                    return value_str
                except ValueError:
                    if hasattr(list_widget, 'native'):
                        list_widget.native.setStyleSheet("background-color: #FFDDDD;")
                    return initial_str

            widget = magicgui(
                list_widget, auto_call=True,
                value_str={"widget_type": "LineEdit", "label": label}
            )

        elif param_type == "float":
            def float_widget(value: float = float(param_config.get("value", 0.0))):
                callback(value)
                return value
            widget = magicgui(
                float_widget, auto_call=True,
                value={
                    "widget_type": "FloatSpinBox", "label": label,
                    "min": float(param_config.get("min", 0)),
                    "max": float(param_config.get("max", 100)),
                    "step": float(param_config.get("step", 0.1))
                }
            )

        elif param_type == "int":
            def int_widget(value: int = int(param_config.get("value", 0))):
                callback(value)
                return value
            widget = magicgui(
                int_widget, auto_call=True,
                value={
                    "widget_type": "SpinBox", "label": label,
                    "min": int(param_config.get("min", 0)),
                    "max": int(param_config.get("max", 100)),
                    "step": int(param_config.get("step", 1))
                }
            )

        elif param_type == "bool":
            def bool_widget(value: bool = bool(param_config.get("value", False))):
                callback(value)
                return value
            widget = magicgui(
                bool_widget, auto_call=True,
                value={"widget_type": "CheckBox", "label": label}
            )

        else:
            def fallback(value: str = str(param_config.get("value", ""))):
                callback(value)
                return value
            widget = magicgui(
                fallback, auto_call=True,
                value={"widget_type": "LineEdit", "label": label}
            )

        if widget:
            widget.param_name = param_name

    except Exception:
        return None
    return widget


def scan_available_presets() -> Dict[str, Dict[str, str]]:
    """Scans the module directories for available YAML configuration presets."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_locations = [
        (os.path.join(script_dir, '..', 'module_3d', 'configs'), 'ramified'),
        (os.path.join(script_dir, '..', 'module_2d', 'configs'), 'ramified_2d')
    ]
    presets = {}
    
    for config_dir, default_mode in search_locations:
        if not os.path.exists(config_dir):
            continue
        try:
            files = [
                f for f in os.listdir(config_dir)
                if f.lower().endswith(('.yaml', '.yml'))
            ]
            for f in files:
                full_path = os.path.join(config_dir, f)
                clean_name = os.path.splitext(f)[0].replace('_', ' ').title()
                suffix = " (2D)" if "module_2d" in config_dir else " (3D)"
                presets[f"{clean_name}{suffix}"] = {
                    "path": full_path,
                    "default_mode": default_mode
                }
        except Exception:
            pass
    return presets


# =============================================================================
# PROJECT ORGANIZATION LOGIC
# =============================================================================

def clean_filename_for_matching(name: str) -> str:
    """
    Normalizes filenames for matching.
    1. Lowercase.
    2. Remove common extensions (czi, tif, etc.).
    3. Remove trailing ' #N' suffixes often added by Zen imports.
    """
    n = name.lower()
    # Remove extensions (iteratively to handle .czi.tif)
    for ext in ['.tif', '.tiff', '.czi', '.lsm', '.nd2', '.oib', '.lif']:
        n = n.replace(ext, '')
    # Remove scene suffixes like " #1", " #2"
    n = re.sub(r'\s+#\d+$', '', n)
    return n.strip()


def organize_channel_project(
    source_files: List[str],
    source_root: str,
    target_root_dir: str,
    channel_idx: int,
    preset_details: Dict[str, str]
) -> None:
    """Setup Logic for MULTI-CHANNEL mode."""
    config_template_path = preset_details['path']
    fallback_mode = preset_details['default_mode']

    print(f"  Organizing Channel {channel_idx} into: {target_root_dir}")
    os.makedirs(target_root_dir, exist_ok=True)

    with open(config_template_path, 'r') as f:
        template_data = yaml.safe_load(f) or {}
    
    mode = template_data.get('mode', fallback_mode)
    is_2d_mode = mode.endswith('_2d')
    dimension_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'

    metadata_rows = []

    for src_file in source_files:
        src_path = os.path.join(source_root, src_file)
        # Check if file actually has this channel
        if MetadataExtractor.get_channel_count(src_path) <= channel_idx:
            continue

        basename = os.path.splitext(src_file)[0]
        img_subdir = os.path.join(target_root_dir, basename)
        os.makedirs(img_subdir, exist_ok=True)

        target_tif_name = f"{basename}.tif"
        target_tif_path = os.path.join(img_subdir, target_tif_name)

        print(f"    Processing {src_file}...")
        try:
            MetadataExtractor.extract_channel_to_tiff(
                src_path, target_tif_path, channel_idx
            )
        except Exception as e:
            print(f"    Error extracting channel {channel_idx} from {src_file}: {e}")
            continue

        if src_file.lower().endswith('.czi'):
            meta = MetadataExtractor.get_czi_metadata(src_path)
        else:
            meta = MetadataExtractor.read_tiff_metadata(src_path)

        # Re-read the extracted file to get actual pixel counts
        try:
            mem = tiff.imread(target_tif_path)
            shape = mem.shape
            if mem.ndim == 3: # (Z, Y, X)
                z_slices = shape[0]
                height = shape[-2]
                width = shape[-1]
            else: # (Y, X)
                z_slices = 1
                height = shape[-2]
                width = shape[-1]
            del mem
        except Exception:
            z_slices, width, height = 1, 1, 1

        # Use 1.0 as fallback if metadata wasn't found in file
        scale_x = float(meta['x'])
        scale_y = float(meta['y'])
        scale_z = float(meta['z'])

        metadata_rows.append({
            'Filename': target_tif_name,
            'Width (um)': scale_x * width,
            'Height (um)': scale_y * height,
            'Depth (um)': scale_z * z_slices,
            'Slices': z_slices
        })

        new_config_path = os.path.join(
            img_subdir, os.path.basename(config_template_path)
        )
        if not os.path.exists(new_config_path):
            shutil.copy2(config_template_path, new_config_path)

        try:
            with open(new_config_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
            if dimension_key not in cfg:
                cfg[dimension_key] = {}
            
            # Save the TOTAL dimensions in microns as expected by your logic
            cfg[dimension_key]['x'] = scale_x * width
            cfg[dimension_key]['y'] = scale_y * height
            if not is_2d_mode:
                cfg[dimension_key]['z'] = scale_z * z_slices
            cfg['mode'] = mode
            with open(new_config_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        except Exception:
            pass

    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        df.sort_values(
            'Filename', key=lambda col: col.map(natural_sort_key), inplace=True
        )
        csv_path = os.path.join(target_root_dir, "metadata.csv")
        df.to_csv(csv_path, index=False)
        print(f"    Saved metadata summary to {csv_path}")


def organize_processing_dir(drctry: str, preset_details: Dict[str, str]) -> None:
    """
    Setup Logic for SINGLE-CHANNEL / LEGACY mode.
    Includes Robust Matching for CSV filenames vs Disk filenames.
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
        # FORCE comment=None to prevent '#' from stripping characters
        df = pd.read_csv(os.path.join(drctry, csv_files[0]), comment=None)
        
        # Use filename as basename initially, strip extensions later
        if 'Filename' in df.columns:
            df['Basename'] = df['Filename'].apply(lambda x: os.path.splitext(str(x))[0])
        else:
            print("  Warning: CSV missing 'Filename' column.")
            
    elif len(csv_files) > 1:
        raise ValueError("Multiple CSV files found. Please keep only one.")
    else:
        print("  No CSV found. Auto-generating metadata from files...")
        df = pd.DataFrame(columns=[
            'Filename', 'Width (um)', 'Height (um)', 'Depth (um)', 'Slices'
        ])

    # 2. Config Template
    with open(config_template_path, 'r') as f:
        template_data = yaml.safe_load(f) or {}
    
    mode = template_data.get('mode', fallback_mode)
    is_2d_mode = mode.endswith('_2d')
    dimension_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'

    generated_rows = []

    # 3. Generate Metadata (if needed)
    if df is not None and 'Filename' not in df.columns: # Fallback if empty CSV
        for img_file in raw_images:
            full_path = os.path.join(drctry, img_file)
            basename = os.path.splitext(img_file)[0]
            print(f"  Analyzing: {img_file}")
            meta = MetadataExtractor.read_tiff_metadata(full_path)
            try:
                mem = tiff.imread(full_path)
                z_slices = mem.shape[0] if mem.ndim == 3 else 1
                spacing_x = float(meta['x']) if meta['found'] else 1.0
                spacing_y = float(meta['y']) if meta['found'] else 1.0
                spacing_z = float(meta['z']) if meta['found'] else 1.0
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
            df.to_csv(os.path.join(drctry, "auto_generated_metadata.csv"), index=False)
            print("  Saved 'auto_generated_metadata.csv'.")

    if 'Basename' not in df.columns and 'Filename' in df.columns:
        df['Basename'] = df['Filename'].apply(lambda x: os.path.splitext(str(x))[0])

    # 4. Create Folder Structure (Robust Matching)
    files_moved = 0
    missing_files = []

    # Prepare file map for fast lookup
    # Map cleaned_name -> real_filename
    disk_files_map = {}
    for f in raw_images:
        disk_files_map[clean_filename_for_matching(f)] = f

    for _, row in df.iterrows():
        raw_csv_name = str(row['Filename']).strip()
        if not raw_csv_name: continue
        
        # Clean the CSV name
        clean_csv_name = clean_filename_for_matching(raw_csv_name)
        
        # Match Logic:
        # 1. Exact clean match
        matched_file = disk_files_map.get(clean_csv_name)
        
        # 2. Substring match (if exact fails)
        if not matched_file:
            for clean_disk, real_disk in disk_files_map.items():
                if clean_disk in clean_csv_name or clean_csv_name in clean_disk:
                    matched_file = real_disk
                    break
        
        if not matched_file:
            missing_files.append(raw_csv_name)
            continue

        # Use the matched file to create folder
        root_name = os.path.splitext(matched_file)[0] # Folder name derived from actual file
        new_dir = os.path.join(drctry, root_name)
        os.makedirs(new_dir, exist_ok=True)

        src = os.path.join(drctry, matched_file)
        dst = os.path.join(new_dir, matched_file)

        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.move(src, dst)
            files_moved += 1

        new_config_path = os.path.join(new_dir, os.path.basename(config_template_path))
        if not os.path.exists(new_config_path):
            shutil.copy2(config_template_path, new_config_path)

        try:
            with open(new_config_path, 'r') as f: cfg = yaml.safe_load(f) or {}
            if dimension_key not in cfg: cfg[dimension_key] = {}
            cfg[dimension_key]['x'] = float(row.get('Width (um)', 1.0))
            cfg[dimension_key]['y'] = float(row.get('Height (um)', 1.0))
            if not is_2d_mode:
                cfg[dimension_key]['z'] = float(row.get('Depth (um)', 0.0))
            cfg['mode'] = mode
            with open(new_config_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        except Exception: pass

    print(f"Standard Organization complete. Moved {files_moved} files.")
    if missing_files:
        print(f"Warning: Could not find images for {len(missing_files)} CSV entries.")
        if len(missing_files) < 10:
            print(f"Missing: {missing_files}")
        else:
            print(f"Examples: {missing_files[:3]} ...")
        
        # Debug Help
        print("Available files on disk (cleaned):")
        print(list(disk_files_map.keys())[:10])


# =============================================================================
# APP STATE & MANAGER
# =============================================================================

class ApplicationState(QObject):
    """Singleton to manage global application signals and windows."""
    show_project_view_signal = pyqtSignal()
    _instance = None
    project_view_window: Optional[QMainWindow] = None

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
    """Handles folder selection and validation of image projects."""
    
    def __init__(self):
        self.project_path: Optional[str] = None
        self.image_folders: List[str] = []

    def select_project_folder(self) -> Optional[str]:
        self.project_path = QFileDialog.getExistingDirectory(
            None, "Select Project Root Folder", ""
        )
        if self.project_path:
            self._find_valid_image_folders()
        return self.project_path

    def _find_valid_image_folders(self) -> None:
        self.image_folders = []
        if not self.project_path or not os.path.isdir(self.project_path):
            return
        try:
            for item in os.listdir(self.project_path):
                potential_folder_path = os.path.join(self.project_path, item)
                if os.path.isdir(potential_folder_path):
                    try:
                        folder_contents = os.listdir(potential_folder_path)
                        tif_files = [
                            f for f in folder_contents
                            if f.lower().endswith(('.tif', '.tiff'))
                        ]
                        yaml_files = [
                            f for f in folder_contents
                            if f.lower().endswith(('.yaml', '.yml'))
                        ]
                        # Must contain exactly one image and one config
                        if len(tif_files) == 1 and len(yaml_files) == 1:
                            self.image_folders.append(potential_folder_path)
                    except OSError:
                        pass
        except OSError:
            self.image_folders = []
        
        self.image_folders.sort(key=natural_sort_key)

    def get_image_details(self, folder_path: str) -> Dict[str, Any]:
        """Reads metadata from the folder."""
        try:
            contents = os.listdir(folder_path)
            tif_file = next(
                (f for f in contents if f.lower().endswith(('.tif', '.tiff'))), None
            )
            yaml_file = next(
                (f for f in contents if f.lower().endswith(('.yaml', '.yml'))), None
            )
            
            if not tif_file or not yaml_file:
                return {'path': folder_path, 'mode': 'error'}
            
            config = {}
            try:
                with open(os.path.join(folder_path, yaml_file), 'r') as file:
                    config = yaml.safe_load(file) or {}
            except Exception:
                pass
                
            return {
                'path': folder_path,
                'tif_file': tif_file,
                'yaml_file': yaml_file,
                'mode': config.get('mode', 'unknown')
            }
        except Exception:
            return {'path': folder_path, 'mode': 'error'}


class ProjectViewWindow(QMainWindow):
    """The main entry window for selecting a project."""
    
    def __init__(self, project_manager: ProjectManager):
        super().__init__()
        self.project_manager = project_manager
        self.initUI()
        self.setAttribute(Qt.WA_QuitOnClose)

    def initUI(self) -> None:
        self.setWindowTitle("Image Segmentation Project")
        self.setGeometry(100, 100, 700, 450)
        
        central_widget = QWidget()
        layout = QVBoxLayout()
        
        self.project_path_label = QLabel("Project Path: Not Selected")
        layout.addWidget(self.project_path_label)
        
        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self.open_image_view)
        layout.addWidget(self.image_list)
        
        button_layout = QHBoxLayout()
        select_project_btn = QPushButton("Select/Load Project Folder")
        select_project_btn.clicked.connect(self.load_project)
        button_layout.addWidget(select_project_btn)
        
        self.batch_process_all_btn = QPushButton("Process All Compatible Folders")
        self.batch_process_all_btn.clicked.connect(self.run_batch_processing_all_compatible)
        self.batch_process_all_btn.setEnabled(False)
        button_layout.addWidget(self.batch_process_all_btn)
        
        layout.addLayout(button_layout)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def _update_batch_button_state(self) -> None:
        if not BatchProcessor or not self.project_manager.image_folders:
            self.batch_process_all_btn.setEnabled(False)
            return
        self.batch_process_all_btn.setEnabled(True)

    def load_project(self) -> None:
        selected_path = self.project_manager.select_project_folder()
        if not selected_path:
            return
            
        self.project_path_label.setText(f"Project Path: {selected_path}")
        self.image_list.clear()
        
        try:
            # Check for raw files that might need organization
            raw_files = [
                f for f in os.listdir(selected_path)
                if f.lower().endswith(('.tif', '.tiff', '.czi')) and
                os.path.isfile(os.path.join(selected_path, f))
            ]
            
            # Logic to distinguish New Project vs Existing Project
            needs_organization = False
            is_multi_channel = False

            if any(f.endswith('.czi') for f in raw_files):
                needs_organization = True
                is_multi_channel = True
            elif raw_files:
                first_tif = os.path.join(selected_path, raw_files[0])
                if MetadataExtractor.get_channel_count(first_tif) > 1:
                    needs_organization = True
                    is_multi_channel = True
                else:
                    needs_organization = True
                    is_multi_channel = False

            if needs_organization:
                msg = (
                    "Setup multi-channel project structure?" if is_multi_channel
                    else "Organize single-channel project?"
                )
                reply = QMessageBox.question(
                    self, "Setup Project?",
                    f"Found {len(raw_files)} raw images.\n{msg}",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    presets = scan_available_presets()
                    if not presets:
                        QMessageBox.critical(self, "Error", "No config presets found.")
                        return

                    if is_multi_channel:
                        # Multi-Channel Logic
                        max_channels = 1
                        for f in raw_files:
                            path = os.path.join(selected_path, f)
                            max_channels = max(max_channels, MetadataExtractor.get_channel_count(path))
                        print(f"Detected {max_channels} channels max.")

                        for ch in range(max_channels):
                            preset_key, ok = QInputDialog.getItem(
                                self, f"Channel {ch} Configuration",
                                f"Select Preset for Channel {ch}:",
                                sorted(list(presets.keys())), 0, False
                            )
                            if ok and preset_key:
                                target_dir = os.path.join(
                                    selected_path,
                                    f"Channel_{ch}_{preset_key.split()[0]}"
                                )
                                QApplication.setOverrideCursor(Qt.WaitCursor)
                                try:
                                    organize_channel_project(
                                        raw_files, selected_path, target_dir,
                                        ch, presets[preset_key]
                                    )
                                except Exception as e:
                                    QMessageBox.critical(self, "Error", f"Failed Ch{ch}: {e}")
                                finally:
                                    QApplication.restoreOverrideCursor()
                        
                        QMessageBox.information(
                            self, "Done",
                            "Project setup complete. Load specific Channel folder."
                        )
                        return
                    else:
                        # Standard/Legacy Logic
                        preset_key, ok = QInputDialog.getItem(
                            self, "Select Preset", "Choose configuration:",
                            sorted(list(presets.keys())), 0, False
                        )
                        if ok and preset_key:
                            QApplication.setOverrideCursor(Qt.WaitCursor)
                            try:
                                organize_processing_dir(selected_path, presets[preset_key])
                            except Exception as e:
                                QMessageBox.critical(self, "Error", f"Failed: {e}")
                            finally:
                                QApplication.restoreOverrideCursor()

            # Standard Load (Subfolders)
            self.project_manager._find_valid_image_folders()

            for folder_path in self.project_manager.image_folders:
                details = self.project_manager.get_image_details(folder_path)
                item = QListWidgetItem(
                    f"{os.path.basename(folder_path)} - Mode: {details.get('mode')}"
                )
                item.setData(Qt.UserRole, folder_path)
                self.image_list.addItem(item)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
        self._update_batch_button_state()

    def open_image_view(self, item: QListWidgetItem) -> None:
        folder = item.data(Qt.UserRole)
        if folder:
            self.hide()
            interactive_segmentation_with_config(folder)

    def run_batch_processing_all_compatible(self) -> None:
        if not self.batch_process_all_btn.isEnabled():
            return
        reply = QMessageBox.question(
            self, "Confirm", "Process all folders?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            processor = BatchProcessor(self.project_manager)
            processor.process_all_folders(force_restart_all=False)
            QMessageBox.information(self, "Done", "Batch processing complete.")

    def closeEvent(self, event: QCloseEvent) -> None:
        reply = QMessageBox.question(
            self, 'Exit', "Exit application?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            QApplication.instance().quit()
            event.accept()
        else:
            event.ignore()


def _check_if_last_window() -> None:
    """Checks if the project window is closed; if so, quits the app."""
    app = QApplication.instance()
    if not app:
        return
    pv = app_state.project_view_window
    valid = False
    try:
        if pv:
            valid = pv.isVisible()
    except Exception:
        pass
    
    if not valid:
        app.quit()


def _handle_napari_close() -> None:
    """Callback when Napari closes."""
    QTimer.singleShot(100, _check_if_last_window)


def interactive_segmentation_with_config(selected_folder: str = None) -> None:
    """Launches Napari with the DynamicGUIManager for a single sample."""
    try:
        from .gui_manager import DynamicGUIManager
    except ImportError:
        return

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = None
    
    try:
        if not selected_folder:
            raise ValueError("No folder selected.")
            
        contents = os.listdir(selected_folder)
        tif = next(
            (f for f in contents if f.lower().endswith(('.tif', '.tiff'))), None
        )
        yml = next(
            (f for f in contents if f.lower().endswith(('.yaml', '.yml'))), None
        )
        
        if not tif or not yml:
            raise FileNotFoundError("Missing TIF/YAML files in folder.")

        file_loc = os.path.join(selected_folder, tif)
        with open(os.path.join(selected_folder, yml), 'r') as f:
            config = yaml.safe_load(f)
        mode = config.get('mode')

        image_stack = tiff.imread(file_loc)
        viewer = napari.Viewer(title=f"Segmentation: {os.path.basename(selected_folder)}")

        qt_window = viewer.window._qt_window
        qt_window.destroyed.connect(_handle_napari_close)

        gui_manager = DynamicGUIManager(viewer, config, image_stack, file_loc, mode)
        viewer.window.add_dock_widget(
            create_back_to_project_button(viewer), area="left", name="Navigation"
        )

        @magicgui(call_button="▶ Next Step / Run Current")
        def continue_processing():
            gui_manager.execute_processing_step()

        @magicgui(call_button="◀ Previous Step")
        def go_to_previous_step():
            idx = gui_manager.current_step["value"]
            if idx > 0:
                gui_manager.cleanup_step(idx)
                gui_manager.current_step["value"] -= 1
                gui_manager.create_step_widgets(
                    gui_manager.processing_steps[gui_manager.current_step["value"]]
                )
                update_navigation_buttons()

        def update_navigation_buttons():
            idx = gui_manager.current_step["value"]
            total = len(gui_manager.processing_steps)
            go_to_previous_step.enabled = (idx > 0)
            continue_processing.enabled = (idx < total)
            
            if idx < total:
                step_name = gui_manager.processing_steps[idx]
                display = gui_manager.step_display_names.get(step_name, step_name)
                continue_processing.label = f"Run Step {idx + 1}: {display}"
            else:
                continue_processing.label = "Processing Complete"

        def disable_buttons_during_process():
            continue_processing.enabled = False
            go_to_previous_step.enabled = False
            continue_processing.label = "Processing... (Please Wait)"

        gui_manager.process_started.connect(disable_buttons_during_process)
        gui_manager.process_finished.connect(update_navigation_buttons)

        container = QWidget()
        l = QVBoxLayout()
        container.setLayout(l)
        l.addWidget(continue_processing.native)
        l.addWidget(go_to_previous_step.native)
        l.setContentsMargins(5, 5, 5, 5)
        
        viewer.window.add_dock_widget(
            container, area="left", name="Processing Control"
        )
        update_navigation_buttons()

    except Exception as e:
        QMessageBox.critical(None, "Error", str(e))
        app_state.show_project_view_signal.emit()


def launch_image_segmentation_tool() -> QApplication:
    """Main entry point for the GUI application."""
    app = QApplication.instance() or QApplication(sys.argv)

    def show_pv():
        if not app_state.project_view_window:
            app_state.project_view_window = ProjectViewWindow(ProjectManager())
        app_state.project_view_window.show()
        app_state.project_view_window.raise_()

    app_state.show_project_view_signal.connect(show_pv)
    show_pv()
    return app


def create_back_to_project_button(viewer: napari.Viewer) -> QWidget:
    """Creates the 'Back to Project List' button widget."""
    def _do():
        if viewer:
            viewer.close()
        app_state.show_project_view_signal.emit()

    btn = QPushButton("Back to Project List")
    btn.clicked.connect(_do)
    w = QWidget()
    l = QVBoxLayout()
    w.setLayout(l)
    l.addWidget(btn)
    l.setContentsMargins(5, 5, 5, 5)
    return w