import os
import re
import sys
import shutil
import time
import traceback
import gc
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
    QWidget, QLabel, QInputDialog, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView
)

from .relational_engine import RelationalEngine

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
                dims_list = czi.get_dims_shape() if hasattr(czi, 'get_dims_shape') else czi.dims_shape()
                if dims_list:
                    dims = dims_list[0]
                    if 'C' in dims: return dims['C'][1] - dims['C'][0]
                return 1
            except Exception: return 1

        elif ext in ['.tif', '.tiff']:
            try:
                with tiff.TiffFile(path) as tif:
                    if tif.imagej_metadata:
                        return int(tif.imagej_metadata.get('channels', 1))
                    if tif.ome_metadata:
                        match = re.search(r'SizeC="(\d+)"', str(tif.ome_metadata))
                        if match: return int(match.group(1))
                    if len(tif.series) > 0:
                        shape = tif.series[0].shape
                        if len(shape) == 3 and shape[0] < 10 and shape[0] < shape[1]: return shape[0]
                        if len(shape) == 4: return min(shape[0], shape[1])
            except Exception: return 1
        return 1

    @staticmethod
    def read_tiff_metadata(path: str) -> Dict[str, Union[float, bool]]:
        """Attempts to read physical scale (microns) with robust ImageJ support."""
        meta: Dict[str, Union[float, bool]] = {'x': 1.0, 'y': 1.0, 'z': 1.0, 'found': False}
        try:
            with tiff.TiffFile(path) as tif:
                ij = tif.imagej_metadata or {}
                # 1. Capture Z-Spacing from ImageJ immediately
                if 'spacing' in ij:
                    meta['z'] = float(ij['spacing'])
                    meta['found'] = True

                # 2. Capture X/Y from Tags or ImageJ
                if tif.pages:
                    page = tif.pages[0]
                    x_res = page.tags.get('XResolution')
                    y_res = page.tags.get('YResolution')
                    u_tag = page.tags.get('ResolutionUnit')
                    
                    if x_res and y_res:
                        x_val, y_val = x_res.value, y_res.value
                        x_dens = x_val[0]/x_val[1] if isinstance(x_val, tuple) else x_val
                        y_dens = y_val[0]/y_val[1] if isinstance(y_val, tuple) else y_val
                        
                        # Unit detection: Tag says 'None' (1), but ImageJ string might say 'micron'
                        unit_str = str(ij.get('unit', '')).lower()
                        u_val = u_tag.value if u_tag else 1
                        
                        if x_dens > 0:
                            # Case: Unit is Microns (Standard for Fiji calibration)
                            if u_val == 3 or unit_str in ['micron', 'µm', 'um']:
                                # If unit is cm (3), density is px/cm. 10000/dens = um/px
                                # If unit is micron, density is px/um. 1/dens = um/px
                                factor = 10000.0 if u_val == 3 else 1.0
                                meta['x'], meta['y'] = factor/x_dens, factor/y_dens
                                meta['found'] = True
                            # Case: Unit is Inches (DPI)
                            elif u_val == 2:
                                meta['x'], meta['y'] = 25400.0/x_dens, 25400.0/y_dens
                                meta['found'] = True
                            # Case: Unit is "None" but we have numbers (often happens in bio-formats)
                            elif u_val == 1:
                                if x_dens < 1.0: # Likely already microns per pixel
                                    meta['x'], meta['y'] = x_dens, y_dens
                                else: # Likely pixels per micron
                                    meta['x'], meta['y'] = 1.0/x_dens, 1.0/y_dens
                                meta['found'] = True

                # 3. OME-XML Fallback
                if not meta['found'] and tif.ome_metadata:
                    txt = str(tif.ome_metadata)
                    for ax in ['X', 'Y', 'Z']:
                        m = re.search(rf'PhysicalSize{ax}="([\d\.]+)"', txt)
                        if m: 
                            meta[ax.lower()] = float(m.group(1))
                            meta['found'] = True

        except Exception as e:
            print(f"Metadata read error: {e}")
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
    def extract_channel_to_tiff(src_path: str, dest_path: str, channel_idx: int) -> None:
        """Extracts a channel and preserves the spatial resolution tags."""
        try:
            ext = os.path.splitext(src_path)[1].lower()
            ch_data = None
            source_meta = {'x': 1.0, 'y': 1.0, 'z': 1.0}

            # --- BRANCH 1: CZI FILES ---
            if ext == '.czi' and HAS_CZI:
                # 1. Get Metadata specifically for CZI
                source_meta = MetadataExtractor.get_czi_metadata(src_path)
                
                # 2. Extract Data using aicspylibczi
                try:
                    czi = CziFile(src_path)
                    # Read specific channel. explicit T=0 ensures we get a volume, not a 4D hyperstack if time exists
                    # This returns (data, list_of_dims). Data usually has shape (1, 1, Z, Y, X) or similar.
                    data, dims = czi.read_image(C=channel_idx)
                    ch_data = np.squeeze(data)
                except Exception as czi_e:
                    print(f"    CZI Read Error: {czi_e}")
                    return

            # --- BRANCH 2: TIFF FILES ---
            elif ext in ['.tif', '.tiff']:
                # 1. Get Metadata specifically for TIFF
                source_meta = MetadataExtractor.read_tiff_metadata(src_path)
                
                # 2. Extract Data using tifffile
                vol = tiff.imread(src_path)
                
                # Handle ImageJ Hyperstacks (Z vs C vs T)
                if vol.ndim == 3:
                    # Differentiate (C,Y,X) from (Z,Y,X)
                    # Heuristic: Channels usually < 10, Z usually < Y/X
                    if vol.shape[0] < 10 and vol.shape[0] < vol.shape[1]: 
                        ch_data = vol[channel_idx]
                    else: 
                        # Assumes single channel Z-stack
                        ch_data = vol
                elif vol.ndim == 4:
                    # Usually (C, Z, Y, X) or (Z, C, Y, X). 
                    # Simplistic assumption: Smallest dim is C.
                    if vol.shape[0] < vol.shape[1]: # (C, Z, Y, X)
                        ch_data = vol[channel_idx]
                    else: # (Z, C, Y, X)
                        ch_data = vol[:, channel_idx, :, :]
                else:
                    ch_data = vol
            
            else:
                print(f"    Unsupported file type for extraction: {ext}")
                return

            # --- COMMON: SAVE TO DISK ---
            if ch_data is not None:
                # Calculate resolution for ImageJ/Fiji (pixels per unit)
                # If meta is 1.0 (default), res is 1.0
                res_x = 1.0 / source_meta['x'] if source_meta['x'] > 0 else 1.0
                res_y = 1.0 / source_meta['y'] if source_meta['y'] > 0 else 1.0

                tiff.imwrite(
                    dest_path, ch_data, 
                    photometric='minisblack',
                    resolution=(res_x, res_y),
                    metadata={'unit': 'micron', 'spacing': source_meta['z']}
                )

        except Exception as e:
            print(f"Extraction failed for {os.path.basename(src_path)}: {e}")
            traceback.print_exc()

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

class ScalesTableWidget(QWidget):
    """
    A unified table to manage Scales, Low Thresholds, and High Thresholds together.
    Returns a list of dicts: [{'scale': 1.0, 'low': 95.0, 'high': 100.0}, ...]
    """
    valueChanged = pyqtSignal(object)

    def __init__(self, initial_value: List[Dict[str, float]], label: str = "", is_absolute: bool = False):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # Controls
        btn_layout = QHBoxLayout()
        self.lbl = QLabel(label)
        self.btn_add = QPushButton("+")
        self.btn_add.setFixedWidth(30)
        self.btn_rem = QPushButton("-")
        self.btn_rem.setFixedWidth(30)
        
        self.btn_add.clicked.connect(self.add_row)
        self.btn_rem.clicked.connect(self.remove_row)
        
        btn_layout.addWidget(self.lbl)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_rem)
        self.layout.addLayout(btn_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Scale", "Low Val", "High Val"] if is_absolute else ["Scale", "Low %", "High %"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(150)
        self.table.itemChanged.connect(self._emit_change)
        self.layout.addWidget(self.table)
        
        # Populate
        self.set_value(initial_value)

    def set_value(self, data: List[Dict[str, float]]):
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        if isinstance(data, list):
            for row_idx, item in enumerate(data):
                if isinstance(item, dict):
                    self.table.insertRow(row_idx)
                    self._set_item(row_idx, 0, item.get('scale', 1.0))
                    self._set_item(row_idx, 1, item.get('low', 95.0))
                    self._set_item(row_idx, 2, item.get('high', 100.0))
        self.table.blockSignals(False)

    def _set_item(self, row, col, val):
        item = QTableWidgetItem(str(val))
        self.table.setItem(row, col, item)

    def add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        self._set_item(r, 0, 1.0)
        self._set_item(r, 1, 95.0)
        self._set_item(r, 2, 100.0)
        self._emit_change()

    def remove_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self._emit_change()

    def _emit_change(self):
        data = []
        for r in range(self.table.rowCount()):
            try:
                s = float(self.table.item(r, 0).text())
                l = float(self.table.item(r, 1).text())
                h = float(self.table.item(r, 2).text())
                data.append({'scale': s, 'low': l, 'high': h})
            except (ValueError, AttributeError):
                pass 
        self.valueChanged.emit(data)

    @property
    def native(self):
        return self
    

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
        # --- Handle Percentile Table ---
        if param_type == "scale_table" or param_type == "scale_table_percentile":
            initial_val = param_config.get("value",[])
            widget = ScalesTableWidget(initial_val, label, is_absolute=False)
            widget.valueChanged.connect(callback)
            return widget
        
        # --- Handle Absolute Table ---
        elif param_type == "scale_table_absolute":
            initial_val = param_config.get("value",[])
            widget = ScalesTableWidget(initial_val, label, is_absolute=True)
            widget.valueChanged.connect(callback)
            return widget
        
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
        (os.path.join(script_dir, '..', 'module_3d', 'configs'), 'fluorescence'),
        (os.path.join(script_dir, '..', 'module_2d', 'configs'), 'fluorescence_2d')
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
        
        # Ensure file has the requested channel
        if MetadataExtractor.get_channel_count(src_path) <= channel_idx:
            continue

        basename = os.path.splitext(src_file)[0]
        img_subdir = os.path.join(target_root_dir, basename)
        os.makedirs(img_subdir, exist_ok=True)

        target_tif_name = f"{basename}.tif"
        target_tif_path = os.path.join(img_subdir, target_tif_name)

        print(f"    Processing {src_file}...")
        try:
            MetadataExtractor.extract_channel_to_tiff(src_path, target_tif_path, channel_idx)
        except Exception as e:
            print(f"    Error extracting channel {channel_idx} from {src_file}: {e}")
            continue

        # Extract metadata from original source (richer metadata than extracted single channel)
        if src_file.lower().endswith('.czi'):
            meta = MetadataExtractor.get_czi_metadata(src_path)
        else:
            meta = MetadataExtractor.read_tiff_metadata(src_path)

        # Get pixel counts from the extracted file
        try:
            mem = tiff.imread(target_tif_path)
            shape = mem.shape
            z_slices = shape[0] if mem.ndim == 3 else 1
            height, width = shape[-2], shape[-1]
            del mem
            gc.collect() # Force immediate release during heavy batch organization
        except Exception:
            z_slices, width, height = 1, 1, 1

        # Use the extracted scale to calculate TOTAL microns
        total_w = float(meta['x']) * width
        total_h = float(meta['y']) * height
        total_d = float(meta['z']) * z_slices

        metadata_rows.append({
            'Filename': target_tif_name,
            'Width (um)': total_w,
            'Height (um)': total_h,
            'Depth (um)': total_d,
            'Slices': z_slices
        })

        new_config_path = os.path.join(img_subdir, os.path.basename(config_template_path))
        if not os.path.exists(new_config_path):
            shutil.copy2(config_template_path, new_config_path)

        try:
            with open(new_config_path, 'r') as f: cfg = yaml.safe_load(f) or {}
            if dimension_key not in cfg: cfg[dimension_key] = {}
            cfg[dimension_key]['x'] = total_w
            cfg[dimension_key]['y'] = total_h
            if not is_2d_mode:
                cfg[dimension_key]['z'] = total_d
            cfg['mode'] = mode
            with open(new_config_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        except Exception:
            pass

    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        df.sort_values('Filename', key=lambda col: col.map(natural_sort_key), inplace=True)
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


# =============================================================================
# Config Template Application
# =============================================================================

def apply_template_config_to_project(
    template_yaml_path: str,
    project_manager: Any,
    target_mode: Optional[str] = None
) -> Dict[str, Any]:
    """
    Applies a template YAML config to every matching image folder in a project.

    Merges strategies:
    - All ``execute_*`` parameter blocks are taken wholesale from the template.
    - ``voxel_dimensions`` / ``pixel_dimensions`` are preserved from each image's
      own YAML so physical calibration is never overwritten.
    - ``mode`` is preserved from each folder's existing value.
    - For folders that already have a processed config (Config 2), ``saved_state``
      (e.g. auto-detected thresholds) is also preserved so a partial run can still
      be resumed after the template swap.

    Args:
        template_yaml_path: Absolute path to the source ``.yaml`` template file.
        project_manager:    A ``ProjectManager`` instance with populated
                            ``image_folders`` and ``get_image_details()``.
        target_mode:        If given, only folders whose ``mode`` field matches
                            this string are updated.  If ``None``, the mode is
                            taken from the template itself and used as the filter.

    Returns:
        dict with keys ``success``, ``failed``, ``skipped``, ``updated_folders``.
    """
    try:
        with open(template_yaml_path, 'r') as fh:
            template: Dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise ValueError(f"Cannot read template file: {exc}") from exc

    effective_filter_mode = target_mode or template.get('mode')

    results: Dict[str, Any] = {
        'success': 0, 'failed': 0, 'skipped': 0, 'updated_folders': []
    }

    for folder_path in project_manager.image_folders:
        details = project_manager.get_image_details(folder_path)
        folder_name = os.path.basename(folder_path)

        # Skip broken or missing-file folders
        if details.get('mode') == 'error' or not details.get('yaml_file'):
            results['skipped'] += 1
            continue

        folder_mode = details.get('mode', 'unknown')

        # Mode filtering — only apply to folders that match the template's mode
        if effective_filter_mode and folder_mode not in ('unknown', effective_filter_mode):
            results['skipped'] += 1
            continue

        yaml_path = os.path.join(folder_path, details['yaml_file'])

        # ── Config 1: main YAML in the image folder ────────────────────────
        try:
            with open(yaml_path, 'r') as fh:
                current_main: Dict[str, Any] = yaml.safe_load(fh) or {}
        except Exception:
            results['failed'] += 1
            continue

        merged_main: Dict[str, Any] = {}

        # Take every execute_* block from the template
        for key, val in template.items():
            if key.startswith('execute_'):
                merged_main[key] = val

        # Preserve per-image physical calibration (never overwrite with template values)
        for dim_key in ('voxel_dimensions', 'pixel_dimensions'):
            if dim_key in current_main:
                merged_main[dim_key] = current_main[dim_key]
            elif dim_key in template:
                # Fallback: use template dimensions only if the image has none
                merged_main[dim_key] = template[dim_key]

        # Keep the folder's existing mode; use template mode only as a last resort
        merged_main['mode'] = (
            folder_mode if folder_mode != 'unknown'
            else template.get('mode', 'unknown')
        )

        try:
            with open(yaml_path, 'w') as fh:
                yaml.safe_dump(merged_main, fh, default_flow_style=False, sort_keys=False)
        except Exception:
            results['failed'] += 1
            continue

        # ── Config 2: processed config inside *_processed_* sub-folder ────
        tif_file = details.get('tif_file')
        if tif_file:
            basename = os.path.splitext(tif_file)[0]
            effective_mode = merged_main['mode']
            proc_dir = os.path.join(
                folder_path, f"{basename}_processed_{effective_mode}"
            )
            proc_config_path = os.path.join(
                proc_dir, f"processing_config_{effective_mode}.yaml"
            )

            if os.path.exists(proc_config_path):
                try:
                    with open(proc_config_path, 'r') as fh:
                        existing_proc: Dict[str, Any] = yaml.safe_load(fh) or {}

                    merged_proc = dict(merged_main)

                    # Preserve image-specific computed values (e.g. auto-threshold)
                    if 'saved_state' in existing_proc:
                        merged_proc['saved_state'] = existing_proc['saved_state']

                    with open(proc_config_path, 'w') as fh:
                        yaml.safe_dump(
                            merged_proc, fh, default_flow_style=False, sort_keys=False
                        )
                except Exception:
                    # Non-critical: failure here doesn't fail the whole folder
                    pass

        results['success'] += 1
        results['updated_folders'].append(folder_name)

    return results


class ProjectManager:
    """Handles folder selection and validation of image projects."""
    
    def __init__(self):
        self.project_path: Optional[str] = None
        self.image_folders: List[str] = []
        self.sample_registry: Dict[str, Dict[str, str]] = {}

    def select_project_folder(self) -> Optional[str]:
        self.project_path = QFileDialog.getExistingDirectory(
            None, "Select Project Root Folder", ""
        )
        if self.project_path:
            self.sample_registry.clear()
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
        
    def build_consolidated_sample_registry(self) -> Dict[str, Dict[str, str]]:
        """
        Scans the directory containing the current project to find related channels.
        
        Returns:
            Dict mapping Sample_Name -> {Channel_Name: Folder_Path}
            Example:
            {
                "Animal_01_ROI_1": {
                    "Microglia": "/path/to/Channel_Microglia/Animal_01_ROI_1",
                    "Plaques": "/path/to/Channel_Plaques/Animal_01_ROI_1"
                }
            }
        """
        if not self.project_path:
            return {}

        # 1. Identify the 'Project Root' (The folder containing all Channel projects)
        parent_dir = os.path.dirname(self.project_path)
        all_channel_projects = [
            os.path.join(parent_dir, d) for d in os.listdir(parent_dir)
            if os.path.isdir(os.path.join(parent_dir, d))
        ]

        registry = {}

        # 2. Iterate through every Channel Project (e.g., Channel_0_Microglia)
        for channel_path in all_channel_projects:
            channel_name = os.path.basename(channel_path)
            
            # 3. Look for valid sample folders inside this channel
            for sample_folder in os.listdir(channel_path):
                sample_path = os.path.join(channel_path, sample_folder)
                if not os.path.isdir(sample_path):
                    continue
                
                # Check if it's a valid processing folder (has TIF and YAML)
                files = os.listdir(sample_path)
                has_tif = any(f.lower().endswith(('.tif', '.tiff')) for f in files)
                has_yml = any(f.lower().endswith(('.yaml', '.yml')) for f in files)
                
                if has_tif and has_yml:
                    # Use a 'Clean' name for the sample to handle slight naming mismatches
                    clean_name = clean_filename_for_matching(sample_folder)
                    
                    if clean_name not in registry:
                        registry[clean_name] = {}
                    
                    # Store the actual path mapped to this channel
                    registry[clean_name][channel_name] = sample_path
        
        self.sample_registry = registry
        return registry


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

        self.cross_channel_btn = QPushButton("Open Cross-Channel Analyzer")
        self.cross_channel_btn.clicked.connect(self.open_cross_channel_analyzer)
        self.cross_channel_btn.setEnabled(False) # Enable only after project load
        button_layout.addWidget(self.cross_channel_btn)

        self.set_config_btn = QPushButton("⚙ Set New Channel Config…")
        self.set_config_btn.setToolTip(
            "Choose a YAML config template and apply its processing parameters\n"
            "to every image in the project (image dimensions are preserved)."
        )
        self.set_config_btn.clicked.connect(self.set_channel_config)
        self.set_config_btn.setEnabled(False)  # Enable only after project load
        button_layout.addWidget(self.set_config_btn)

    def _update_batch_button_state(self) -> None:
        if not BatchProcessor or not self.project_manager.image_folders:
            self.batch_process_all_btn.setEnabled(False)
            self.set_config_btn.setEnabled(False)
            return
        self.batch_process_all_btn.setEnabled(True)
        self.set_config_btn.setEnabled(True)

    def open_cross_channel_analyzer(self):
        self.project_manager.build_consolidated_sample_registry()
            
        self.analyzer_window = CrossChannelAnalyzerWindow(self.project_manager)
        self.analyzer_window.show()

    def load_project(self) -> None:
        selected_path = self.project_manager.select_project_folder()
        self.cross_channel_btn.setEnabled(True)
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
            interactive_segmentation_with_config(folder, project_manager=self.project_manager)

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

    def set_channel_config(self) -> None:
        """Opens a file dialog to pick a template YAML and applies it to all folders."""
        template_path, _ = QFileDialog.getOpenFileName(
            self, "Select Config Template", "",
            "YAML Files (*.yaml *.yml);;All Files (*)"
        )
        if not template_path:
            return

        # Preview what mode the template targets
        try:
            with open(template_path, 'r') as fh:
                template_preview = yaml.safe_load(fh) or {}
            template_mode = template_preview.get('mode', 'unknown')
            execute_keys = [k for k in template_preview if k.startswith('execute_')]
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not read template:\n{exc}")
            return

        total = len(self.project_manager.image_folders)
        reply = QMessageBox.question(
            self,
            "Apply Config Template",
            f"Template:  {os.path.basename(template_path)}\n"
            f"Mode:      {template_mode}\n"
            f"Steps:     {len(execute_keys)}\n\n"
            f"Apply to all {total} image folder(s) in the project?\n\n"
            f"• Processing parameters will be replaced.\n"
            f"• Image dimensions are always preserved.\n"
            f"• Folders with a different mode will be skipped.\n"
            f"• Existing computed state (e.g. thresholds) is preserved.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            results = apply_template_config_to_project(
                template_path, self.project_manager
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Config application failed:\n{exc}")
            return
        QApplication.restoreOverrideCursor()

        summary = (
            f"Config template applied.\n\n"
            f"Updated : {results['success']}\n"
            f"Skipped : {results['skipped']}  (different mode or invalid)\n"
            f"Failed  : {results['failed']}\n"
        )
        if results['updated_folders']:
            preview = results['updated_folders'][:8]
            summary += "\nUpdated folders:\n" + "\n".join(f"  \u2022 {n}" for n in preview)
            if len(results['updated_folders']) > 8:
                summary += f"\n  \u2026 and {len(results['updated_folders']) - 8} more"

        if results['failed'] > 0:
            QMessageBox.warning(self, "Partial Success", summary)
        else:
            QMessageBox.information(self, "Done", summary)

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

class CrossChannelAnalyzerWindow(QMainWindow):
    def __init__(self, project_manager):
        super().__init__()
        self.pm = project_manager
        self.setWindowTitle("Cross-Channel Relational Analyzer")
        self.setGeometry(150, 150, 1000, 650)
        
        main_layout = QHBoxLayout()
        
        # --- 1. LEFT PANEL: Channels ---
        left_panel = QVBoxLayout()
        self.channel_list = QListWidget()
        
        # Safely get channels
        if not self.pm.sample_registry:
            self.pm.build_consolidated_sample_registry()
            
        first_sample = list(self.pm.sample_registry.keys())[0]
        channels = sorted(list(self.pm.sample_registry[first_sample].keys()))
        for ch in channels:
            item = QListWidgetItem(ch)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.channel_list.addItem(item)
        
        left_panel.addWidget(QLabel("<b>1. Select Input Channels:</b>"))
        left_panel.addWidget(self.channel_list)
        
        # --- 2. MIDDLE PANEL: Recipe List & Controls ---
        mid_panel = QVBoxLayout()
        mid_panel.addWidget(QLabel("<b>2. Analysis Recipe (Order Matters):</b>"))
        
        recipe_hbox = QHBoxLayout()
        self.recipe_list = QListWidget()
        recipe_hbox.addWidget(self.recipe_list)
        
        step_controls = QVBoxLayout()
        self.btn_remove = QPushButton("❌ Remove")
        self.btn_up = QPushButton("🔼 Up")
        self.btn_down = QPushButton("🔽 Down")
        self.btn_clear = QPushButton("🗑️ Clear All")
        
        self.btn_remove.clicked.connect(self.remove_step)
        self.btn_up.clicked.connect(lambda: self.move_step(-1))
        self.btn_down.clicked.connect(lambda: self.move_step(1))
        self.btn_clear.clicked.connect(self.clear_recipe)
        
        step_controls.addWidget(self.btn_remove)
        step_controls.addWidget(self.btn_up)
        step_controls.addWidget(self.btn_down)
        step_controls.addStretch()
        step_controls.addWidget(self.btn_clear)
        recipe_hbox.addLayout(step_controls)
        
        mid_panel.addLayout(recipe_hbox)
        
        # --- 3. ADD STEP BUTTONS ---
        add_step_layout = QHBoxLayout()
        
        self.btn_synth = QPushButton("🧬 Generate Synthetic Channel")
        self.btn_synth.setStyleSheet("background-color: #8A2BE2; color: white;") # Purple
        
        self.btn_intersect = QPushButton("+ Intersection")
        self.btn_filter = QPushButton("+ Volume Filter")
        self.btn_dist = QPushButton("+ Distance Analysis")
        
        self.btn_synth.clicked.connect(self.open_synthetic_dialog)
        self.btn_intersect.clicked.connect(self.add_intersect_step)
        self.btn_filter.clicked.connect(self.add_filter_step)
        self.btn_dist.clicked.connect(self.add_analysis_step)
        
        add_step_layout.addWidget(self.btn_synth)
        add_step_layout.addWidget(self.btn_intersect)
        add_step_layout.addWidget(self.btn_filter)
        add_step_layout.addWidget(self.btn_dist)
        mid_panel.addLayout(add_step_layout)

        # --- 4. EXECUTION PANEL ---
        exec_layout = QVBoxLayout()
        
        # Sample Selector
        selector_layout = QHBoxLayout()
        selector_layout.addWidget(QLabel("Preview Target Sample:"))
        self.sample_selector = QComboBox()
        self.sample_selector.addItems(sorted(list(self.pm.sample_registry.keys())))
        selector_layout.addWidget(self.sample_selector)
        exec_layout.addLayout(selector_layout)

        self.btn_preview = QPushButton("👁️ Preview Recipe (Napari)")
        self.btn_preview.setFixedHeight(40)
        self.btn_preview.clicked.connect(self.preview_recipe)
        
        self.btn_load_prev = QPushButton("📂 Load Previous Analysis for Sample")
        self.btn_load_prev.setFixedHeight(40)
        self.btn_load_prev.clicked.connect(self.load_previous_analysis)
        
        self.btn_batch = QPushButton("🚀 RUN RECIPE ON ALL SAMPLES")
        self.btn_batch.setFixedHeight(50)
        self.btn_batch.setStyleSheet("background-color: #2E8B57; color: white; font-weight: bold;")
        self.btn_batch.clicked.connect(self.run_batch_analysis)
        
        exec_layout.addWidget(self.btn_preview)
        exec_layout.addWidget(self.btn_load_prev)
        exec_layout.addWidget(self.btn_batch)
        mid_panel.addLayout(exec_layout)

        container = QWidget()
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(mid_panel, 3)
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        self.recipe_steps = []

    # =========================================================================
    # RECIPE EDITING METHODS
    # =========================================================================

    def remove_step(self):
        row = self.recipe_list.currentRow()
        if row >= 0:
            self.recipe_steps.pop(row)
            self.recipe_list.takeItem(row)

    def move_step(self, direction):
        """direction: -1 for up, 1 for down"""
        row = self.recipe_list.currentRow()
        new_row = row + direction
        if 0 <= new_row < self.recipe_list.count():
            # Swap in logic list
            self.recipe_steps[row], self.recipe_steps[new_row] = \
                self.recipe_steps[new_row], self.recipe_steps[row]
            
            # Swap in UI list
            item = self.recipe_list.takeItem(row)
            self.recipe_list.insertItem(new_row, item)
            self.recipe_list.setCurrentRow(new_row)

    def clear_recipe(self):
        reply = QMessageBox.question(self, "Confirm", "Clear entire recipe?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.recipe_steps = []
            self.recipe_list.clear()

    # =========================================================================
    # ADD STEP METHODS
    # =========================================================================

    def get_checked_channels(self):
        return [self.channel_list.item(i).text() for i in range(self.channel_list.count()) 
                if self.channel_list.item(i).checkState() == Qt.Checked]
    def open_synthetic_dialog(self):
        try:
            from .synthetic_engine import SyntheticDataDialog
            dialog = SyntheticDataDialog(self.pm, self)
            dialog.exec_()
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Failed to load synthetic engine:\n{e}")

    def add_intersect_step(self):
        checked = self.get_checked_channels()
        if not (len(checked) == 2 or (len(checked) == 1 and self.recipe_steps)):
            QMessageBox.warning(self, "Error", "Select channels for intersection.")
            return

        # NEW: Ask for Labeling Mode
        modes = {
            "Binary (All overlaps = ID 1)": "binary",
            "Connected Components (Every fragment unique)": "connected",
            "Inherit Parent A (Keep IDs of first channel)": "parent_a",
            "Inherit Parent B (Keep IDs of second channel)": "parent_b"
        }
        
        mode_display, ok = QInputDialog.getItem(
            self, "Intersection Mode", 
            "How should the resulting overlap mask be labeled?", 
            list(modes.keys()), 0, False
        )
        
        if not ok: return
        label_mode = modes[mode_display]

        if len(checked) == 2:
            step = {
                "type": "intersect", "inputs": checked, "label_mode": label_mode,
                "name": f"Overlap ({label_mode}): {checked[0]} & {checked[1]}"
            }
        else:
            step = {
                "type": "intersect", "inputs": [checked[0], "PREVIOUS_RESULT"], "label_mode": label_mode,
                "name": f"Overlap ({label_mode}): {checked[0]} with previous"
            }
        
        self.recipe_steps.append(step)
        self.recipe_list.addItem(step["name"])

    def add_filter_step(self):
        val, ok = QInputDialog.getDouble(self, "Filter", "Min Volume (um³):", 10.0, 0, 1000000, 2)
        if ok:
            step = {"type": "filter", "min_vol": val, "name": f"Filter: Keep objects > {val} um³"}
            self.recipe_steps.append(step)
            self.recipe_list.addItem(step["name"])

    def add_analysis_step(self):
        checked = self.get_checked_channels()
        if not checked:
            QMessageBox.warning(self, "Error", "Check at least one channel for distance analysis.")
            return

        # Determine whether there is an accumulated pipeline result (intersection / filter)
        # that could act as one side of the analysis.
        has_previous_result = any(
            s['type'] in ('intersect', 'filter') for s in self.recipe_steps
        )

        for ch in checked:
            if has_previous_result:
                # --- Case A: previous accumulated result exists ---
                # Ask which role the checked channel plays.
                role, ok = QInputDialog.getItem(
                    self,
                    "Select Primary Channel",
                    "Which side should be the PRIMARY (objects distances are reported FOR)?",
                    [
                        f"{ch}  →  primary  (measure FROM {ch} TO the previous result)",
                        f"Previous result  →  primary  (measure FROM previous result TO {ch})",
                    ],
                    0, False
                )
                if not ok:
                    return

                if role.startswith(ch):
                    # ch is primary, previous result is the partner
                    step = {
                        "type":    "analyze",
                        "primary": ch,
                        "target":  "PREVIOUS_RESULT",
                        "name":    f"Analyze {ch} → distance to previous result",
                    }
                else:
                    # previous result is primary, ch is the partner
                    step = {
                        "type":   "analyze",
                        "target": ch,
                        "name":   f"Analyze previous result → distance to {ch}",
                    }

            else:
                # --- Case B: simple two-channel analysis, no prior pipeline result ---
                # The second channel must be chosen from the channel list.
                other_channels = [
                    self.channel_list.item(i).text()
                    for i in range(self.channel_list.count())
                    if self.channel_list.item(i).text() != ch
                ]
                if not other_channels:
                    QMessageBox.warning(self, "Error", "Need at least two channels for distance analysis.")
                    return

                partner, ok = QInputDialog.getItem(
                    self,
                    "Select Partner Channel",
                    f"Measure distance FROM  '{ch}'  TO which channel?",
                    other_channels, 0, False
                )
                if not ok:
                    return

                # Ask which of the two is primary
                role, ok = QInputDialog.getItem(
                    self,
                    "Select Primary Channel",
                    "Which side should be the PRIMARY (objects distances are reported FOR)?",
                    [
                        f"{ch}  →  primary  (measure FROM {ch} TO {partner})",
                        f"{partner}  →  primary  (measure FROM {partner} TO {ch})",
                    ],
                    0, False
                )
                if not ok:
                    return

                if role.startswith(ch):
                    primary_ch, partner_ch = ch, partner
                else:
                    primary_ch, partner_ch = partner, ch

                step = {
                    "type":    "analyze",
                    "primary": primary_ch,
                    "target":  partner_ch,
                    "name":    f"Analyze {primary_ch} → distance to {partner_ch}",
                }

            self.recipe_steps.append(step)
            self.recipe_list.addItem(step["name"])

    def preview_recipe(self):
        if not self.recipe_steps:
            QMessageBox.information(self, "Info", "Recipe is empty.")
            return

        # 1. Ask for a name to make this a "Single Run"
        analysis_name, ok = QInputDialog.getText(
            self, "Single Sample Run", 
            "Enter a name for this analysis (files will be saved):",
            text="Preview_Run"
        )
        if not ok or not analysis_name: return

        sample_name = self.sample_selector.currentText()
        sample_data = self.pm.sample_registry[sample_name]
        
        # 2. Setup Viewer
        # Setup permanent path for this sample's relational results
        project_root = os.path.dirname(self.pm.project_path)
        sample_out_dir = os.path.join(
            project_root, "RELATIONAL_ANALYSIS", analysis_name, sample_name
        )
        os.makedirs(sample_out_dir, exist_ok=True)

        viewer = napari.Viewer(title=f"Cross-Channel Preview: {sample_name}")
        
        # Predefined colormaps for raw channels
        colormaps = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue']
        
        # 3. Load Raw Data and Segmentation for EVERY channel in this sample
        shape = None
        spacing = (1.0, 1.0, 1.0)
        
        for i, (ch_name, ch_path) in enumerate(sample_data.items()):
            # Find files
            tif_file = next((os.path.join(ch_path, f) for f in os.listdir(ch_path) if f.lower().endswith(('.tif', '.tiff'))), None)
            dat_file = RelationalEngine._find_dat(ch_path)
            
            # Fetch metadata from the first valid channel we find
            if shape is None and tif_file:
                with tiff.TiffFile(tif_file) as tif:
                    shape = tif.series[0].shape
                # Try to get spacing from strategy config
                meta, _ = get_sample_metadata(ch_path)
                if meta:
                    # Very simple spacing calc: total_um / pixels
                    # (Note: In a production version, we use the exact strategy spacing)
                    spacing = (meta.get('z', 1.0)/shape[0], meta.get('y', 1.0)/shape[1], meta.get('x', 1.0)/shape[2]) if len(shape)==3 else (meta.get('y', 1.0)/shape[0], meta.get('x', 1.0)/shape[1])

            # Add Raw Intensity
            if tif_file:
                raw_img = tiff.imread(tif_file)
                cmap = colormaps[i % len(colormaps)]
                viewer.add_image(raw_img, name=f"Raw: {ch_name}", colormap=cmap, blending='additive', opacity=0.5)

            # Add Segmentation Labels (Semi-transparent)
            if dat_file:
                seg_data = np.memmap(dat_file, dtype=np.int32, mode='r', shape=shape)
                viewer.add_labels(seg_data, name=f"Seg: {ch_name}", opacity=0.3)

        # 4. Execute Relational Recipe
        temp_dir = os.path.join(list(sample_data.values())[0], "relational_preview_temp")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Run calculation
        derived_masks, metrics_df = RelationalEngine.run_recipe(
            sample_name, self.pm.sample_registry, self.recipe_steps, 
            sample_out_dir, shape, spacing
        )

        # Add the Red Proximity Lines
        if metrics_df is not None:
            self._draw_proximity_bridges(viewer, metrics_df, shape, spacing)

        # 5. Add Derived Results (High Opacity Labels)
        for res in derived_masks:
            data = np.memmap(res['path'], dtype=np.int32, mode='r', shape=shape)
            viewer.add_labels(data, name=f"DERIVED: {res['name']}")

        # Final Adjustments
        if len(shape) == 3:
            viewer.dims.ndisplay = 3
            # Set scale to handle anisotropy if 3D
            # (Napari scale is z_scale, y_scale, x_scale)
            # Use spacing[0]/spacing[2] for z-scale factor
            z_scale = spacing[0]/spacing[2] if len(spacing)==3 else 1.0
            for layer in viewer.layers:
                layer.scale = (z_scale, 1, 1)

    def load_previous_analysis(self):
        project_root = os.path.dirname(self.pm.project_path)
        rel_dir = os.path.join(project_root, "RELATIONAL_ANALYSIS")
        
        if not os.path.exists(rel_dir):
            QMessageBox.warning(self, "Not Found", "No RELATIONAL_ANALYSIS folder found.")
            return

        # Find existing analysis folders
        analyses =[d for d in os.listdir(rel_dir) if os.path.isdir(os.path.join(rel_dir, d))]
        if not analyses:
            QMessageBox.information(self, "Info", "No previous analyses found.")
            return

        # Prompt user to select an analysis run
        analysis_name, ok = QInputDialog.getItem(
            self, "Load Analysis", "Select previous analysis to load:", analyses, 0, False
        )
        if not ok or not analysis_name:
            return

        sample_name = self.sample_selector.currentText()
        sample_out_dir = os.path.join(rel_dir, analysis_name, sample_name)

        if not os.path.exists(sample_out_dir):
            QMessageBox.warning(self, "Not Found", f"No data found for sample '{sample_name}' in analysis '{analysis_name}'.")
            return

        sample_data = self.pm.sample_registry[sample_name]
        
        # Setup Viewer
        viewer = napari.Viewer(title=f"Loaded Analysis: {analysis_name} | {sample_name}")
        colormaps = ['cyan', 'magenta', 'yellow', 'green', 'red', 'blue']
        
        shape = None
        spacing = (1.0, 1.0, 1.0)

        # 1. Load Raw Data and Base Segmentation (identical to preview function)
        for i, (ch_name, ch_path) in enumerate(sample_data.items()):
            tif_file = next((os.path.join(ch_path, f) for f in os.listdir(ch_path) if f.lower().endswith(('.tif', '.tiff'))), None)
            dat_file = RelationalEngine._find_dat(ch_path)
            
            if shape is None and tif_file:
                with tiff.TiffFile(tif_file) as tif:
                    shape = tif.series[0].shape
                meta, _ = get_sample_metadata(ch_path)
                if meta:
                    spacing = (meta.get('z', 1.0)/shape[0], meta.get('y', 1.0)/shape[1], meta.get('x', 1.0)/shape[2]) if len(shape)==3 else (meta.get('y', 1.0)/shape[0], meta.get('x', 1.0)/shape[1])

            if tif_file:
                raw_img = tiff.imread(tif_file)
                cmap = colormaps[i % len(colormaps)]
                viewer.add_image(raw_img, name=f"Raw: {ch_name}", colormap=cmap, blending='additive', opacity=0.5)

            if dat_file:
                seg_data = np.memmap(dat_file, dtype=np.int32, mode='r', shape=shape)
                viewer.add_labels(seg_data, name=f"Seg: {ch_name}", opacity=0.3)

        # 2. Load Derived Masks (.dat files from the analysis output folder)
        derived_files =[f for f in os.listdir(sample_out_dir) if f.endswith('.dat')]
        for f in derived_files:
            dat_path = os.path.join(sample_out_dir, f)
            try:
                data = np.memmap(dat_path, dtype=np.int32, mode='r', shape=shape)
                viewer.add_labels(data, name=f"DERIVED: {f.replace('.dat', '')}")
            except Exception as e:
                print(f"Could not load {f}: {e}")

        # 3. Load Metrics & Draw Bridges
        csv_path = os.path.join(sample_out_dir, f"{sample_name}_relational_metrics.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                self._draw_proximity_bridges(viewer, df, shape, spacing)
            except Exception as e:
                print(f"Could not draw bridges: {e}")

        # 4. Final Viewport Adjustments
        if shape and len(shape) == 3:
            viewer.dims.ndisplay = 3
            z_scale = spacing[0]/spacing[2] if len(spacing)==3 else 1.0
            for layer in viewer.layers:
                layer.scale = (z_scale, 1, 1)

    def _draw_proximity_bridges(self, viewer, df, shape, spacing):
        """
        Parses the metrics dataframe for Source/Target coordinates and draws 
        red connection lines (bridges) between interacting biological objects.
        """
        # Identify which partners were analyzed (e.g., 'Microglia', 'Neurons')
        # We find them by looking for the biological suffixes in the coordinate columns
        partners = [c.replace('src_y_', '') for c in df.columns if c.startswith('src_y_')]
        
        is_3d = (len(shape) == 3)
        
        # Calculate visualization scale (handles anisotropy)
        z_scale = spacing[0] / spacing[-1] if is_3d else 1.0
        display_scale = (z_scale, 1, 1) if is_3d else (1, 1)

        for p in partners:
            lines = []
            for _, row in df.iterrows():
                # Only draw a bridge if a nearest neighbor was successfully found (not NaN)
                if pd.notna(row.get(f'dist_um_{p}')):
                    try:
                        if is_3d:
                            # Extract 3D Bridge: [Z, Y, X]
                            src = [row[f'src_z_{p}'], row[f'src_y_{p}'], row[f'src_x_{p}']]
                            tgt = [row[f'tgt_z_{p}'], row[f'tgt_y_{p}'], row[f'tgt_x_{p}']]
                        else:
                            # Extract 2D Bridge: [Y, X]
                            src = [row[f'src_y_{p}'], row[f'src_x_{p}']]
                            tgt = [row[f'tgt_y_{p}'], row[f'tgt_x_{p}']]
                        
                        lines.append([src, tgt])
                    except KeyError:
                        # Skip if this specific partner doesn't have coordinates in the table
                        continue
            
            if lines:
                # Add the 'Red Lines' as a Shapes layer
                viewer.add_shapes(
                    lines, 
                    shape_type='line', 
                    edge_color='red', 
                    edge_width=2 if not is_3d else 1, # Thicker for 2D visibility
                    name=f"Bridges to {p}", 
                    scale=display_scale,
                    blending='additive'
                )
        
        print(f"  [Visualizer] Plotted connection bridges for partners: {partners}")

    def run_batch_analysis(self):
        if not self.recipe_steps:
            return

        # 1. Ask for an Analysis Name (to create a subfolder)
        analysis_name, ok = QInputDialog.getText(self, "Batch Run", "Enter name for this analysis:")
        if not ok or not analysis_name:
            return

        # 2. Setup root results folder
        project_root = os.path.dirname(self.pm.project_path)
        batch_out_dir = os.path.join(project_root, "RELATIONAL_ANALYSIS", analysis_name)
        os.makedirs(batch_out_dir, exist_ok=True)

        # 3. Save the recipe itself for reproducibility
        with open(os.path.join(batch_out_dir, "recipe.yaml"), 'w') as f:
            yaml.dump(self.recipe_steps, f)

        # 4. Process all samples
        samples = sorted(list(self.pm.sample_registry.keys()))
        total = len(samples)
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        print(f"\n{'='*60}\nSTARTING RELATIONAL BATCH: {analysis_name}\n{'='*60}")
        
        try:
            for i, s_name in enumerate(samples):
                print(f"Processing {i+1}/{total}: {s_name}...")
                sample_data = self.pm.sample_registry[s_name]
                
                # Fetch shape/spacing from the first channel of the sample
                first_ch = list(sample_data.values())[0]
                tif_path = next(os.path.join(first_ch, f) for f in os.listdir(first_ch) if f.lower().endswith(('.tif', '.tiff')))
                
                with tiff.TiffFile(tif_path) as tif:
                    shape = tif.series[0].shape
                
                # Retrieve spacing from strategy (fallback to 1.0)
                meta, _ = get_sample_metadata(first_ch)
                # Simple spacing calc
                if len(shape) == 3:
                    spacing = (meta.get('z', 1.0)/shape[0], meta.get('y', 1.0)/shape[1], meta.get('x', 1.0)/shape[2])
                else:
                    spacing = (meta.get('y', 1.0)/shape[0], meta.get('x', 1.0)/shape[1])

                # Sample-specific output folder
                sample_out = os.path.join(batch_out_dir, s_name)
                
                # EXECUTE ENGINE
                RelationalEngine.run_recipe(
                    s_name, self.pm.sample_registry, self.recipe_steps,
                    sample_out, shape, spacing
                )

            QApplication.restoreOverrideCursor()
            QMessageBox.information(self, "Success", f"Batch Complete!\nResults saved to: {batch_out_dir}")
            print(f"\n{'='*60}\nBATCH FINISHED\n{'='*60}")

        except Exception as e:
            QApplication.restoreOverrideCursor()
            print(f"FATAL ERROR IN BATCH: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Batch Error", str(e))

        # 5. Create Master Summary Table
        all_csvs = []
        for s_name in samples:
            csv_p = os.path.join(batch_out_dir, s_name, f"{s_name}_relational_results.csv")
            if os.path.exists(csv_p):
                df = pd.read_csv(csv_p)
                df['sample_name'] = s_name
                all_csvs.append(df)
        
        if all_csvs:
            master_df = pd.concat(all_csvs, ignore_index=True)
            master_df.to_csv(os.path.join(batch_out_dir, "MASTER_RELATIONAL_RESULTS.csv"), index=False)
            print("Successfully generated MASTER_RELATIONAL_RESULTS.csv")

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


def interactive_segmentation_with_config(selected_folder: str = None, project_manager=None) -> None:
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

        image_stack = tiff.memmap(file_loc, mode='r') 
        viewer = napari.Viewer(title=f"Segmentation: {os.path.basename(selected_folder)}")

        qt_window = viewer.window._qt_window
        qt_window.destroyed.connect(_handle_napari_close)

        gui_manager = DynamicGUIManager(viewer, config, image_stack, file_loc, mode,
                                        project_manager=project_manager)
        viewer.window.add_dock_widget(
            create_back_to_project_button(viewer, gui_manager), area="left", name="Navigation"
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

        # --- ROI / Sub-region panel ---
        roi_container = QWidget()
        roi_layout = QVBoxLayout()
        roi_container.setLayout(roi_layout)
        roi_layout.setContentsMargins(5, 5, 5, 5)
        roi_layout.setSpacing(4)

        roi_header = QLabel("Sub-region (ROI)")
        roi_header.setStyleSheet(
            "font-weight: bold; font-size: 11px; color: #aaa; padding-top: 4px;"
        )
        roi_layout.addWidget(roi_header)

        btn_draw = QPushButton("✏  Draw ROI")
        btn_draw.setToolTip(
            "Add a polygon layer to draw a sub-region.\n"
            "Click to add vertices, double-click to close."
        )
        btn_draw.clicked.connect(gui_manager.draw_roi)
        roi_layout.addWidget(btn_draw)

        btn_confirm = QPushButton("✓  Confirm ROI")
        btn_confirm.setToolTip(
            "Crop to the drawn polygon and process only that region.\n"
            "Parameters tuned here can be transferred to the full image\n"
            "via 'Set New Channel Config' in the Project View."
        )
        btn_confirm.clicked.connect(gui_manager.confirm_roi)
        roi_layout.addWidget(btn_confirm)

        btn_clear = QPushButton("✗  Clear ROI")
        btn_clear.setToolTip(
            "Remove the ROI layer.\n"
            "If an ROI session is active, offers to return to full-image mode.\n"
            "Existing ROI outputs are kept on disk."
        )
        btn_clear.clicked.connect(gui_manager.clear_roi)
        roi_layout.addWidget(btn_clear)

        viewer.window.add_dock_widget(
            roi_container, area="left", name="ROI / Sub-region"
        )

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


def create_back_to_project_button(viewer: napari.Viewer, gui_manager: Any) -> QWidget:
    """Creates the 'Back to Project List' button widget."""
    def _do():
        nonlocal viewer, gui_manager
        
        # 1. Clean up background tasks and data
        if gui_manager:
            gui_manager.shutdown_and_cleanup()
            gui_manager = None
            
        # 2. Show project view first
        app_state.show_project_view_signal.emit()
        
        # 3. Safely close Napari
        if viewer:
            v = viewer  # Transfer to local variable
            viewer = None  # BREAK THE REFERENCE CYCLE! This lets Python safely destroy the window.
            
            try:
                # Hide immediately for a snappy user experience
                if hasattr(v.window, '_qt_window'):
                    v.window._qt_window.hide()
                
                # Ask Napari to cleanly close itself without manually deleting C++ objects
                QTimer.singleShot(0, v.close)
            except Exception:
                pass
        
        # 4. Force garbage collection so the orphaned viewer object is actually purged
        QTimer.singleShot(100, gc.collect)

    btn = QPushButton("Back to Project List")
    btn.clicked.connect(_do)
    w = QWidget()
    l = QVBoxLayout()
    w.setLayout(l)
    l.addWidget(btn)
    l.setContentsMargins(5, 5, 5, 5)
    return w

def get_sample_metadata(folder_path):
    """Retrieves shape and spacing from the YAML in a project folder."""
    for f in os.listdir(folder_path):
        if f.endswith(('.yaml', '.yml')):
            with open(os.path.join(folder_path, f), 'r') as file:
                cfg = yaml.safe_load(file)
                mode = cfg.get('mode', '')
                is_2d = mode.endswith('_2d')
                dim_key = 'pixel_dimensions' if is_2d else 'voxel_dimensions'
                dims = cfg.get(dim_key, {'x':1, 'y':1, 'z':1})
                # Note: We'd need actual pixel counts to calculate spacing, 
                # but for preview, we can often rely on the Strategy to provide this.
                return dims, mode
    return None, None