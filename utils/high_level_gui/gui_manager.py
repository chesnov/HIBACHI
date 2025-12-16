# --- START OF FILE utils/high_level_gui/gui_manager.py ---
import numpy as np
import os
import time
import sys
import traceback
import gc
import yaml # type: ignore
from typing import Dict, Any, List

# Qt Imports
from PyQt5.QtWidgets import ( # type: ignore
    QMessageBox, QWidget, QVBoxLayout, QScrollArea, QLabel, 
    QTextEdit, QProgressBar, QApplication
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt # type: ignore
from PyQt5.QtGui import QTextCursor # type: ignore

# --- Relative Imports ---
try:
    from ..module_3d._3D_strategy import RamifiedStrategy
    from ..module_2d._2D_strategy import Ramified2DStrategy
    from ..high_level_gui.processing_strategies import ProcessingStrategy
    from .helper_funcs import create_parameter_widget
except ImportError as e:
    print(f"Error importing dependencies in gui_manager.py: {e}")
    raise

# --- 1. Output Redirector (To show tqdm/print in GUI) ---
class OutputStream(QObject):
    text_written = pyqtSignal(str)
    def write(self, text):
        self.text_written.emit(str(text))
    def flush(self):
        pass

# --- 2. Background Worker Thread ---
class StepWorker(QThread):
    finished_signal = pyqtSignal(bool)
    error_signal = pyqtSignal(str)

    def __init__(self, strategy, step_index, image_stack, params):
        super().__init__()
        self.strategy = strategy
        self.step_index = step_index
        self.image_stack = image_stack
        self.params = params

    def run(self):
        try:
            # CRITICAL: viewer=None prevents segmentation faults.
            # GUI updates happen in the main thread upon completion.
            success = self.strategy.execute_step(
                step_index=self.step_index, 
                viewer=None, 
                image_stack_or_none=self.image_stack, 
                params=self.params
            )
            self.finished_signal.emit(success)
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))
            self.finished_signal.emit(False)

# --- 3. Main GUI Manager ---
class DynamicGUIManager(QObject):  # <--- Changed to inherit QObject
    """
    Manages the interactive GUI sidebar in Napari.
    """
    # Signals to communicate with helper_funcs.py
    process_started = pyqtSignal()
    process_finished = pyqtSignal()

    def __init__(self, viewer, config, image_stack, file_loc, processing_mode):
        super().__init__() # <--- Initialize QObject
        self.viewer = viewer
        self.initial_config = config.copy()
        self.config = config.copy()
        self.image_stack = image_stack
        self.file_loc = file_loc
        self.processing_mode = processing_mode
        
        self.current_widgets: Dict[Any, Any] = {}
        self.current_step = {"value": 0}
        self.parameter_values: Dict[str, Any] = {}
        
        self.worker = None
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

        self.inputdir = os.path.dirname(self.file_loc)
        self.basename = os.path.basename(self.file_loc).split('.')[0]
        self.processed_dir = os.path.join(self.inputdir, f"{self.basename}_processed_{self.processing_mode}")

        self._calculate_spacing()

        try:
            strategy_class = {
                'ramified': RamifiedStrategy, 
                'ramified_2d': Ramified2DStrategy
            }.get(self.processing_mode)
            
            if not strategy_class: 
                raise ValueError(f"Unsupported processing mode: {self.processing_mode}")
            
            self.strategy = strategy_class(
                self.config, 
                self.processed_dir, 
                self.image_stack.shape, 
                self.spacing, 
                self.z_scale_factor
            )
            
            self.processing_steps: List[str] = self.strategy.get_step_names()
            self.num_steps: int = self.strategy.num_steps
            
            self.step_display_names = { 
                name: name.replace('execute_', '').replace('_', ' ').title() 
                for name in self.processing_steps 
            }
            
            print(f"Initialized strategy '{self.processing_mode}' with {self.num_steps} steps.")
            
        except Exception as e:
             print(f"FATAL ERROR initializing processing strategy: {e}")
             traceback.print_exc()
             raise

        self._initialize_layers()
        self.restore_from_checkpoint()

    def _calculate_spacing(self):
        is_2d_mode = self.processing_mode.endswith("_2d")
        dim_section_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'
        dimensions = self.config.get(dim_section_key, {})
        try:
            total_x_um = float(dimensions.get('x', 1.0))
            total_y_um = float(dimensions.get('y', 1.0))
            total_z_um = 1.0 if is_2d_mode else float(dimensions.get('z', 1.0))
        except (ValueError, TypeError):
            total_x_um, total_y_um, total_z_um = 1.0, 1.0, 1.0

        shape = self.image_stack.shape
        num_dims = len(shape)

        if num_dims == 2:
            y_pixel_size = total_y_um / shape[0] if shape[0] > 0 else 1.0
            x_pixel_size = total_x_um / shape[1] if shape[1] > 0 else 1.0
            self.spacing = [1.0, y_pixel_size, x_pixel_size]
            self.z_scale_factor = 1.0
        elif num_dims == 3:
            z_pixel_size = total_z_um / shape[0] if shape[0] > 0 else 1.0
            y_pixel_size = total_y_um / shape[1] if shape[1] > 0 else 1.0
            x_pixel_size = total_x_um / shape[2] if shape[2] > 0 else 1.0
            self.spacing = [z_pixel_size, y_pixel_size, x_pixel_size]
            self.z_scale_factor = z_pixel_size / x_pixel_size if x_pixel_size > 1e-9 else 1.0
        else:
            self.spacing = [1.0, 1.0, 1.0]; self.z_scale_factor = 1.0

    def _initialize_layers(self):
        layer_name = f"Original stack ({self.processing_mode} mode)"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)
        image_ndim = self.image_stack.ndim
        scale = (self.z_scale_factor, 1, 1) if image_ndim == 3 else (self.spacing[1], self.spacing[2])
        self.viewer.add_image(self.image_stack, name=layer_name, scale=scale)

    def restore_from_checkpoint(self):
        checkpoint_step = self.strategy.get_last_completed_step()
        if checkpoint_step > 0:
            checkpoint_files = self.strategy.get_checkpoint_files()
            config_file = checkpoint_files.get("config")
            if config_file and os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as file: saved_config = yaml.safe_load(file)
                    if saved_config:
                        self.config.update(saved_config)
                        self.strategy.config = self.config
                        loaded_state = self.config.get('saved_state', {})
                        if 'segmentation_threshold' in loaded_state:
                            self.strategy.intermediate_state['segmentation_threshold'] = float(loaded_state['segmentation_threshold'])
                except Exception as e: print(f"Error loading saved config: {e}")

            msg = QMessageBox()
            if checkpoint_step == self.num_steps:
                msg.setText("All steps complete.")
                msg.setInformativeText("View results or restart?")
                view_btn = msg.addButton("View", QMessageBox.YesRole)
                restart_btn = msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()
                if msg.clickedButton() == view_btn:
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                else: self._confirm_restart()
            else:
                msg.setText("Resume?")
                msg.setInformativeText(f"Resume from Step {checkpoint_step + 1} or restart?")
                resume_btn = msg.addButton("Resume", QMessageBox.YesRole)
                restart_btn = msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()
                if msg.clickedButton() == resume_btn:
                    self.strategy.intermediate_state['original_volume_ref'] = self.image_stack
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                    if checkpoint_step < self.num_steps:
                        self.create_step_widgets(self.processing_steps[checkpoint_step])
                else: self._confirm_restart()
        else:
            self.create_step_widgets(self.processing_steps[0])

    def _confirm_restart(self):
        parent_widget = self.viewer.window._qt_window if self.viewer and self.viewer.window else None
        reply = QMessageBox.question(parent_widget, "Confirm Restart", "Delete existing files?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.delete_all_checkpoint_files()
            self.current_step["value"] = 0
            self.strategy.intermediate_state = {}
            self.config = self.initial_config.copy()
            self.strategy.config = self.config
            self._initialize_layers()
            self.create_step_widgets(self.processing_steps[0])
        else: self.restore_from_checkpoint()

    def delete_all_checkpoint_files(self):
        files_to_delete = self.strategy.get_checkpoint_files()
        for key, file_path in files_to_delete.items():
            self.strategy._remove_file_safely(file_path)

    def load_checkpoint_data(self, checkpoint_step: int):
        self.strategy.load_checkpoint_data(self.viewer, checkpoint_step)

    def cleanup_step(self, step_number: int):
        self.strategy.cleanup_step_artifacts(self.viewer, step_number)

    # --- Threaded Execution ---

    def execute_processing_step(self):
        step_index = self.current_step["value"]
        if step_index >= self.num_steps: return

        logical_step_method_name = self.processing_steps[step_index]
        step_display_name = self.step_display_names.get(logical_step_method_name, f"Step {step_index + 1}")
        
        current_values = self.get_current_values()
        self.strategy.save_config(self.config)
        
        for i in range(step_index + 1, self.num_steps + 1):
             self.cleanup_step(i)
        
        if 'original_volume_ref' not in self.strategy.intermediate_state:
            self.strategy.intermediate_state['original_volume_ref'] = self.image_stack

        # Setup Logs
        self.log_widget.clear()
        self.log_widget.append(f"--- Starting {step_display_name} ---\n")
        self._set_ui_busy(True)
        
        # Signal start (Disables buttons in helper_funcs)
        self.process_started.emit()

        self.output_stream = OutputStream()
        self.output_stream.text_written.connect(self._append_log)
        sys.stdout = self.output_stream
        sys.stderr = self.output_stream 

        # Start Worker
        self.worker = StepWorker(self.strategy, step_index, self.image_stack, current_values)
        self.worker.finished_signal.connect(self._on_step_finished)
        self.worker.start()

    def _append_log(self, text):
        cursor = self.log_widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.log_widget.setTextCursor(cursor)
        self.log_widget.ensureCursorVisible()

    def _on_step_finished(self, success):
        # Restore Output
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        self._set_ui_busy(False)
        self.worker = None 

        step_index = self.current_step["value"]
        logical_step_method_name = self.processing_steps[step_index]
        step_display_name = self.step_display_names.get(logical_step_method_name, f"Step {step_index + 1}")
        parent_widget = self.viewer.window._qt_window if self.viewer else None

        if success:
            self.log_widget.append(f"\n--- {step_display_name} COMPLETED ---")
            
            # Update GUI on Main Thread
            try:
                self.strategy.load_checkpoint_data(self.viewer, step_index + 1)
            except Exception as e:
                print(f"Error updating visualization: {e}")
            
            self.strategy.save_config(self.config)
            self.current_step["value"] += 1
            
            if self.current_step["value"] < self.num_steps:
                next_step_name = self.processing_steps[self.current_step["value"]]
                self.create_step_widgets(next_step_name)
            else:
                self.clear_current_widgets()
                QMessageBox.information(parent_widget, "Complete", "All steps finished.")
        else:
            self.log_widget.append(f"\n!!! {step_display_name} FAILED !!!")
            QMessageBox.warning(parent_widget, "Step Failed", 
                                f"Step '{step_display_name}' failed.\nCheck the log below for details.")
                                
        # Signal finish (Updates buttons in helper_funcs)
        self.process_finished.emit()

    def _set_ui_busy(self, is_busy):
        if self.current_widgets:
             for dock in self.current_widgets.keys():
                 dock.widget().setEnabled(not is_busy)

    def clear_current_widgets(self):
        for dock_widget in list(self.current_widgets.keys()):
            self.viewer.window.remove_dock_widget(dock_widget)
        self.current_widgets.clear()

    def create_step_widgets(self, step_method_name: str):
        self.clear_current_widgets()
        self.parameter_values = {}

        config_key = self.strategy.get_config_key(step_method_name)
        step_display_name = self.step_display_names.get(step_method_name, step_method_name)
        step_config = self.config.get(config_key, {})
        parameters = step_config.get("parameters", {}) if isinstance(step_config, dict) else {}

        scroll_content_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_content_widget)
        
        title_label = QLabel(f"Parameters: {step_display_name}")
        if self.viewer and self.viewer.window and self.viewer.window._qt_window:
            font = self.viewer.window._qt_window.font()
            font.setBold(True)
            title_label.setFont(font)
        scroll_layout.addWidget(title_label)
        
        if isinstance(parameters, dict):
            for param_name, param_config in parameters.items():
                try:
                    callback = lambda value, key=config_key, pn=param_name: self.parameter_changed(key, pn, value)
                    widget = create_parameter_widget(param_name, param_config, callback)
                    if widget: 
                        scroll_layout.addWidget(widget.native)
                        val = param_config.get('value')
                        self.parameter_values[param_name] = val
                except Exception as e: print(f"ERROR creating widget for '{param_name}': {e}")
        
        # Log Widget
        log_label = QLabel("Processing Log (ETA):")
        scroll_layout.addWidget(log_label)
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(150)
        self.log_widget.setStyleSheet("font-family: monospace;")
        scroll_layout.addWidget(self.log_widget)

        scroll_layout.addStretch()
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(scroll_content_widget)
        
        dock_widget = self.viewer.window.add_dock_widget(scroll_area, area="right", name=f"Step: {step_display_name}")
        self.current_widgets[dock_widget] = scroll_area

    def parameter_changed(self, config_key: str, param_name: str, value: Any):
        try:
            self.config[config_key]["parameters"][param_name]["value"] = value
            self.parameter_values[param_name] = value
        except Exception as e: pass

    def get_current_values(self) -> Dict[str, Any]:
        return self.parameter_values.copy()
# --- END OF FILE utils/high_level_gui/gui_manager.py ---