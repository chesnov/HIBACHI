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
    QTextEdit, QProgressBar, QApplication, QPushButton, QFileDialog
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

# --- 1. Output Redirector ---
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
class DynamicGUIManager(QObject):
    process_started = pyqtSignal()
    process_finished = pyqtSignal()

    def __init__(self, viewer, config, image_stack, file_loc, processing_mode):
        super().__init__()
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
            
            if not strategy_class: raise ValueError(f"Unsupported mode: {self.processing_mode}")
            
            self.strategy = strategy_class(
                self.config, self.processed_dir, 
                self.image_stack.shape, self.spacing, self.z_scale_factor
            )
            
            self.processing_steps = self.strategy.get_step_names()
            self.num_steps = self.strategy.num_steps
            self.step_display_names = { 
                name: name.replace('execute_', '').replace('_', ' ').title() 
                for name in self.processing_steps 
            }
            print(f"Initialized strategy '{self.processing_mode}' with {self.num_steps} steps.")
        except Exception as e:
             print(f"FATAL ERROR: {e}"); traceback.print_exc(); raise

        self._initialize_layers()
        self.restore_from_checkpoint()

    def _calculate_spacing(self):
        is_2d_mode = self.processing_mode.endswith("_2d")
        dim_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'
        dim = self.config.get(dim_key, {})
        try:
            tx, ty, tz = float(dim.get('x', 1.0)), float(dim.get('y', 1.0)), float(dim.get('z', 1.0))
        except: tx, ty, tz = 1.0, 1.0, 1.0

        shape = self.image_stack.shape
        if len(shape) == 2:
            ys, xs = 1.0, 1.0
            if shape[0]>0: ys = ty/shape[0]
            if shape[1]>0: xs = tx/shape[1]
            self.spacing = [1.0, ys, xs]; self.z_scale_factor = 1.0
        elif len(shape) == 3:
            zs, ys, xs = 1.0, 1.0, 1.0
            if shape[0]>0: zs = tz/shape[0]
            if shape[1]>0: ys = ty/shape[1]
            if shape[2]>0: xs = tx/shape[2]
            self.spacing = [zs, ys, xs]
            self.z_scale_factor = zs/xs if xs > 1e-9 else 1.0
        else:
            self.spacing = [1.0, 1.0, 1.0]; self.z_scale_factor = 1.0

    def _initialize_layers(self):
        layer_name = f"Original stack ({self.processing_mode} mode)"
        if layer_name in self.viewer.layers: self.viewer.layers.remove(layer_name)
        scale = (self.z_scale_factor, 1, 1) if self.image_stack.ndim == 3 else (self.spacing[1], self.spacing[2])
        self.viewer.add_image(self.image_stack, name=layer_name, scale=scale)

    def restore_from_checkpoint(self):
        checkpoint_step = self.strategy.get_last_completed_step()
        if checkpoint_step > 0:
            files = self.strategy.get_checkpoint_files()
            if files.get("config") and os.path.exists(files["config"]):
                try:
                    with open(files["config"], 'r') as f: 
                        saved = yaml.safe_load(f)
                        if saved: 
                            self.config.update(saved)
                            self.strategy.config = self.config
                            if 'saved_state' in self.config:
                                s = self.config['saved_state']
                                if 'segmentation_threshold' in s:
                                    self.strategy.intermediate_state['segmentation_threshold'] = float(s['segmentation_threshold'])
                except: pass

            msg = QMessageBox()
            if checkpoint_step == self.num_steps:
                msg.setText("All steps complete."); msg.setInformativeText("View results or restart?")
                view = msg.addButton("View", QMessageBox.YesRole)
                rst = msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()
                if msg.clickedButton() == view:
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                else: self._confirm_restart()
            else:
                msg.setText("Resume?"); msg.setInformativeText(f"Resume from Step {checkpoint_step + 1}?")
                res = msg.addButton("Resume", QMessageBox.YesRole)
                rst = msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()
                if msg.clickedButton() == res:
                    self.strategy.intermediate_state['original_volume_ref'] = self.image_stack
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                    if checkpoint_step < self.num_steps:
                        self.create_step_widgets(self.processing_steps[checkpoint_step])
                else: self._confirm_restart()
        else:
            self.create_step_widgets(self.processing_steps[0])

    def _confirm_restart(self):
        reply = QMessageBox.question(self.viewer.window._qt_window, "Confirm Restart", "Delete existing files?", QMessageBox.Yes | QMessageBox.No)
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
        for k, v in self.strategy.get_checkpoint_files().items():
            self.strategy._remove_file_safely(v)

    def load_checkpoint_data(self, checkpoint_step: int):
        self.strategy.load_checkpoint_data(self.viewer, checkpoint_step)

    def cleanup_step(self, step_number: int):
        self.strategy.cleanup_step_artifacts(self.viewer, step_number)

    # --- Step Execution ---
    def execute_processing_step(self):
        step_index = self.current_step["value"]
        if step_index >= self.num_steps: return

        logical_step = self.processing_steps[step_index]
        step_display = self.step_display_names.get(logical_step, f"Step {step_index + 1}")
        
        current_values = self.get_current_values()
        
        if logical_step == "execute_interaction_analysis":
            if not current_values.get("target_channel_folder"):
                QMessageBox.warning(None, "Missing Input", "Please select a Reference Channel folder first.")
                return

        self.strategy.save_config(self.config)
        
        # Don't cleanup if repeating step 6
        if logical_step != "execute_interaction_analysis":
             for i in range(step_index + 1, self.num_steps + 1):
                 self.cleanup_step(i)
        
        if 'original_volume_ref' not in self.strategy.intermediate_state:
            self.strategy.intermediate_state['original_volume_ref'] = self.image_stack

        self.log_widget.clear(); self.log_widget.append(f"--- Starting {step_display} ---\n")
        self._set_ui_busy(True)
        self.process_started.emit()

        self.output_stream = OutputStream()
        self.output_stream.text_written.connect(self._append_log)
        sys.stdout = self.output_stream; sys.stderr = self.output_stream

        self.worker = StepWorker(self.strategy, step_index, self.image_stack, current_values)
        self.worker.finished_signal.connect(self._on_step_finished)
        self.worker.start()

    def _append_log(self, text):
        cursor = self.log_widget.textCursor()
        cursor.movePosition(QTextCursor.End); cursor.insertText(text)
        self.log_widget.setTextCursor(cursor); self.log_widget.ensureCursorVisible()

    def _on_step_finished(self, success):
        sys.stdout = self.original_stdout; sys.stderr = self.original_stderr
        self._set_ui_busy(False); self.worker = None
        
        step_index = self.current_step["value"]
        logical_step = self.processing_steps[step_index]
        step_display = self.step_display_names.get(logical_step, f"Step {step_index + 1}")

        if success:
            self.log_widget.append(f"\n--- {step_display} COMPLETED ---")
            
            # --- CRITICAL FIX: Log Errors in GUI if Viz fails ---
            try: 
                self.strategy.load_checkpoint_data(self.viewer, step_index + 1)
            except Exception as e:
                err_msg = f"\n!!! Visualization Failed !!!\n{str(e)}\nSee console for traceback."
                self.log_widget.append(err_msg)
                print(f"Viz Error: {e}")
                traceback.print_exc()

            self.strategy.save_config(self.config)
            
            if logical_step == "execute_interaction_analysis":
                QMessageBox.information(None, "Analysis Complete", "Interaction analysis finished.\nSee the log below for details.\nYou can select another channel to compare against.")
            else:
                self.current_step["value"] += 1
                if self.current_step["value"] < self.num_steps:
                    self.create_step_widgets(self.processing_steps[self.current_step["value"]])
                else:
                    self.clear_current_widgets()
                    QMessageBox.information(None, "Complete", "All steps finished.")
        else:
            self.log_widget.append(f"\n!!! {step_display} FAILED !!!")
            QMessageBox.warning(None, "Step Failed", "Check log for details.")
        
        self.process_finished.emit()

    def _set_ui_busy(self, is_busy):
        if self.current_widgets:
             for dock in self.current_widgets.keys(): dock.widget().setEnabled(not is_busy)

    def clear_current_widgets(self):
        for dock in list(self.current_widgets.keys()):
            self.viewer.window.remove_dock_widget(dock)
        self.current_widgets.clear()

    # --- Widget Creation ---

    def create_step_widgets(self, step_method_name: str):
        self.clear_current_widgets()
        self.parameter_values = {}
        config_key = self.strategy.get_config_key(step_method_name)
        step_display = self.step_display_names.get(step_method_name, step_method_name)
        
        if step_method_name == "execute_interaction_analysis":
            self.create_interaction_widgets(step_display, config_key)
            return

        step_config = self.config.get(config_key, {})
        parameters = step_config.get("parameters", {}) if isinstance(step_config, dict) else {}

        scroll_w = QWidget(); scroll_l = QVBoxLayout(scroll_w)
        lbl = QLabel(f"Parameters: {step_display}"); lbl.setStyleSheet("font-weight: bold;")
        scroll_l.addWidget(lbl)

        if isinstance(parameters, dict):
            for pname, pconf in parameters.items():
                try:
                    cb = lambda val, k=config_key, p=pname: self.parameter_changed(k, p, val)
                    w = create_parameter_widget(pname, pconf, cb)
                    if w: 
                        scroll_l.addWidget(w.native)
                        self.parameter_values[pname] = pconf.get('value')
                except: pass
        
        self._add_log_widget(scroll_l)
        self._dock_widget(scroll_w, step_display)

    def create_interaction_widgets(self, step_display, config_key):
        scroll_w = QWidget(); layout = QVBoxLayout(scroll_w)
        
        lbl = QLabel(f"{step_display}"); lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(lbl)
        
        desc = QLabel("Select a processed project folder for another channel (e.g. Plaques/Vessels).")
        desc.setWordWrap(True); layout.addWidget(desc)

        self.btn_select_ref = QPushButton("Analyze with Other Channels...")
        self.btn_select_ref.setStyleSheet("padding: 8px; font-weight: bold;")
        self.btn_select_ref.clicked.connect(self.select_reference_channel)
        layout.addWidget(self.btn_select_ref)
        
        self.lbl_ref_path = QLabel("No reference selected")
        self.lbl_ref_path.setStyleSheet("color: #666; font-style: italic; margin-bottom: 10px;")
        self.lbl_ref_path.setWordWrap(True)
        layout.addWidget(self.lbl_ref_path)

        step_config = self.config.get(config_key, {})
        parameters = step_config.get("parameters", {})
        
        if isinstance(parameters, dict):
            for pname, pconf in parameters.items():
                if pname == "target_channel_folder": continue
                try:
                    cb = lambda val, k=config_key, p=pname: self.parameter_changed(k, p, val)
                    w = create_parameter_widget(pname, pconf, cb)
                    if w: 
                        layout.addWidget(w.native)
                        self.parameter_values[pname] = pconf.get('value')
                except: pass

        self._add_log_widget(layout)
        self._dock_widget(scroll_w, step_display)

    def select_reference_channel(self):
        start_dir = os.path.dirname(self.inputdir)
        folder = QFileDialog.getExistingDirectory(None, "Select Reference Channel Project", start_dir)
        
        if folder:
            self.parameter_values['target_channel_folder'] = folder
            display_name = os.path.basename(folder)
            self.lbl_ref_path.setText(f"Selected: {display_name}")
            self.lbl_ref_path.setStyleSheet("color: #2E8B57; font-weight: bold;")

    def _add_log_widget(self, layout):
        layout.addWidget(QLabel("Log:"))
        self.log_widget = QTextEdit(); self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(150)
        self.log_widget.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.log_widget)
        layout.addStretch()

    def _dock_widget(self, widget, name):
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(widget)
        dock = self.viewer.window.add_dock_widget(scroll, area="right", name=f"Step: {name}")
        self.current_widgets[dock] = scroll

    def parameter_changed(self, config_key: str, param_name: str, value: Any):
        try:
            self.config[config_key]["parameters"][param_name]["value"] = value
            self.parameter_values[param_name] = value
        except Exception: pass

    def get_current_values(self) -> Dict[str, Any]:
        return self.parameter_values.copy()
# --- END OF FILE utils/high_level_gui/gui_manager.py ---