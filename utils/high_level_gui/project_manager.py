"""project_manager: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
import yaml  # type: ignore
from typing import Dict, Any, List, Optional
from PyQt5.QtCore import QObject, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QFileDialog, QMainWindow
)

from .gui_text_utils import clean_filename_for_matching, natural_sort_key



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
