"""project_view_window: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
import yaml  # type: ignore
from PyQt5.QtGui import QCloseEvent, QIcon  # type: ignore
from PyQt5.QtCore import Qt  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QApplication, QFileDialog, QMessageBox, QMainWindow, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QPushButton, QWidget, QLabel, QInputDialog
)

from .gui_text_utils import app_icon_path
from .cross_channel_window import CrossChannelAnalyzerWindow
from .metadata import MetadataExtractor
from .project_manager import ProjectManager
from .project_scaffolding import apply_template_config_to_project, organize_channel_project, organize_processing_dir, scan_available_presets

# --- Optional BatchProcessor import ---
try:
    from .batch_processor import BatchProcessor
except ImportError as e:
    print(f"WARNING: Failed to import BatchProcessor: {e}. "
          "Batch processing button will be disabled.")
    BatchProcessor = None  # type: ignore



class ProjectViewWindow(QMainWindow):
    """The main entry window for selecting a project."""
    
    def __init__(self, project_manager: ProjectManager):
        super().__init__()
        self.project_manager = project_manager
        self.initUI()
        self.setAttribute(Qt.WA_QuitOnClose)

    def initUI(self) -> None:
        self.setWindowTitle("Image Segmentation Project")
        _icon = app_icon_path()
        if _icon:
            self.setWindowIcon(QIcon(_icon))
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

        # Unobtrusive version indicator in the status bar (check for updates /
        # switch versions). Guarded so it can never block the home window from
        # opening -- if this isn't a git checkout it simply won't appear.
        try:
            from .version_manager import attach_version_status
            attach_version_status(self)
        except Exception as _exc:
            print(f"[version] status widget unavailable: {_exc}")

    def _update_batch_button_state(self) -> None:
        if not BatchProcessor or not self.project_manager.image_folders:
            self.batch_process_all_btn.setEnabled(False)
            self.set_config_btn.setEnabled(False)
            return
        self.batch_process_all_btn.setEnabled(True)
        self.set_config_btn.setEnabled(True)

    def open_cross_channel_analyzer(self):
        registry = self.project_manager.build_consolidated_sample_registry()
        if not registry:
            QMessageBox.warning(
                self, 
                "No Compatible Data", 
                "Could not find any multi-channel samples in the parent directory.\n\n"
                "Ensure your project is organized into 'Channel_X' folders, and that they share matching sample names."
            )
            return
            
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
            from .app_launch import interactive_segmentation_with_config  # lazy: avoid import cycle
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