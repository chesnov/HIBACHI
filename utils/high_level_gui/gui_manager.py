import os
import sys
import gc
import time
import traceback
import yaml  # type: ignore
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
from PyQt5.QtWidgets import (  # type: ignore
    QMessageBox, QWidget, QVBoxLayout, QScrollArea, QLabel,
    QTextEdit, QProgressBar, QApplication, QPushButton, QFileDialog, QDockWidget
)
from PyQt5.QtCore import QThread, pyqtSignal, QObject, Qt  # type: ignore
from PyQt5.QtGui import QTextCursor  # type: ignore
import napari  # type: ignore

# --- Relative Imports ---
try:
    from ..module_3d._3D_strategy import RamifiedStrategy
    from ..module_2d._2D_strategy import Ramified2DStrategy
    from .processing_strategies import ProcessingStrategy
    from .helper_funcs import create_parameter_widget
except ImportError as e:
    print(f"Error importing dependencies in gui_manager.py: {e}")
    raise


# =============================================================================
# 1. Output Redirector
# =============================================================================

class OutputStream(QObject):
    """
    Redirects stdout/stderr to a Qt Signal for display in the GUI log widget.
    """
    text_written = pyqtSignal(str)

    def write(self, text: str) -> None:
        self.text_written.emit(str(text))

    def flush(self) -> None:
        pass


# =============================================================================
# 2. Background Worker Thread
# =============================================================================

class StepWorker(QThread):
    """
    Executes a processing step in a separate thread to keep the GUI responsive.
    """
    finished_signal = pyqtSignal(bool)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        strategy: ProcessingStrategy,
        step_index: int,
        image_stack: Optional[np.ndarray],
        params: Dict[str, Any]
    ):
        super().__init__()
        self.strategy = strategy
        self.step_index = step_index
        self.image_stack = image_stack
        self.params = params

    def run(self) -> None:
        try:
            success = self.strategy.execute_step(
                step_index=self.step_index,
                viewer=None,  # Viewer is handled by main thread, not worker
                image_stack_or_none=self.image_stack,
                params=self.params
            )
            self.finished_signal.emit(success)
        except Exception as e:
            traceback.print_exc()
            self.error_signal.emit(str(e))
            self.finished_signal.emit(False)


# =============================================================================
# 3. Main GUI Manager
# =============================================================================

class DynamicGUIManager(QObject):
    """
    Manages the Napari GUI state, step navigation, and widget generation.

    It acts as the Controller between the View (Napari/Qt) and the Model
    (ProcessingStrategy). It dynamically builds parameter widgets based on
    the YAML configuration of the current strategy.
    """
    process_started = pyqtSignal()
    process_finished = pyqtSignal()

    def __init__(
        self,
        viewer: napari.Viewer,
        config: Dict[str, Any],
        image_stack: np.ndarray,
        file_loc: str,
        processing_mode: str
    ):
        super().__init__()
        self.viewer = viewer
        self.initial_config = config.copy()
        self.config = config.copy()
        self.image_stack = image_stack
        self.file_loc = file_loc
        self.processing_mode = processing_mode

        # UI State
        self.current_widgets: Dict[QDockWidget, QScrollArea] = {}
        self.current_step = {"value": 0}
        self.parameter_values: Dict[str, Any] = {}
        self.worker: Optional[StepWorker] = None
        
        # Console Redirection
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.output_stream: Optional[OutputStream] = None
        self.log_widget: Optional[QTextEdit] = None

        # Project Paths
        self.inputdir = os.path.dirname(self.file_loc)
        basename = os.path.basename(self.file_loc)
        self.basename = os.path.splitext(basename)[0]
        self.processed_dir = os.path.join(
            self.inputdir, f"{self.basename}_processed_{self.processing_mode}"
        )

        # Spacing
        self.spacing: Union[Tuple[float, float, float], Tuple[float, float]] = (1.0, 1.0, 1.0)
        self.z_scale_factor: float = 1.0
        self._calculate_spacing()

        # Initialize Strategy
        try:
            strategy_class = {
                'ramified': RamifiedStrategy,
                'ramified_2d': Ramified2DStrategy
            }.get(self.processing_mode)

            if not strategy_class:
                raise ValueError(f"Unsupported mode: {self.processing_mode}")

            self.strategy = strategy_class(
                self.config,
                self.processed_dir,
                self.image_stack.shape,
                self.spacing,
                self.z_scale_factor
            )

            self.processing_steps = self.strategy.get_step_names()
            self.num_steps = self.strategy.num_steps
            
            # Prettify step names for display
            self.step_display_names = {
                name: name.replace('execute_', '').replace('_', ' ').title()
                for name in self.processing_steps
            }
            print(f"Initialized strategy '{self.processing_mode}' with "
                  f"{self.num_steps} steps.")

        except Exception as e:
            print(f"FATAL ERROR: {e}")
            traceback.print_exc()
            raise

        self._initialize_layers()
        self.restore_from_checkpoint()

    def _calculate_spacing(self) -> None:
        """Parses spacing from config or defaults to 1.0."""
        is_2d_mode = self.processing_mode.endswith("_2d")
        dim_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'
        dim = self.config.get(dim_key, {})
        
        try:
            tx = float(dim.get('x', 1.0))
            ty = float(dim.get('y', 1.0))
            tz = float(dim.get('z', 1.0))
        except (ValueError, TypeError):
            tx, ty, tz = 1.0, 1.0, 1.0

        shape = self.image_stack.shape
        
        if len(shape) == 2:
            # 2D Case
            ys = ty / shape[0] if shape[0] > 0 else 1.0
            xs = tx / shape[1] if shape[1] > 0 else 1.0
            self.spacing = (1.0, ys, xs)  # Z=1.0 for compatibility
            self.z_scale_factor = 1.0
            
        elif len(shape) == 3:
            # 3D Case
            zs = tz / shape[0] if shape[0] > 0 else 1.0
            ys = ty / shape[1] if shape[1] > 0 else 1.0
            xs = tx / shape[2] if shape[2] > 0 else 1.0
            self.spacing = (zs, ys, xs)
            self.z_scale_factor = zs / xs if xs > 1e-9 else 1.0
            
        else:
            self.spacing = (1.0, 1.0, 1.0)
            self.z_scale_factor = 1.0

    def _initialize_layers(self) -> None:
        """Adds the original image to Napari."""
        layer_name = f"Original stack ({self.processing_mode} mode)"
        if layer_name in self.viewer.layers:
            self.viewer.layers.remove(layer_name)
            
        scale = (
            (self.z_scale_factor, 1, 1) if self.image_stack.ndim == 3
            else (self.spacing[1], self.spacing[2])
        )
        self.viewer.add_image(
            self.image_stack, name=layer_name, scale=scale
        )

    def restore_from_checkpoint(self) -> None:
        """
        Checks for existing outputs and prompts user to Resume or Restart.
        """
        checkpoint_step = self.strategy.get_last_completed_step()
        
        if checkpoint_step > 0:
            # Load saved config
            files = self.strategy.get_checkpoint_files()
            if files.get("config") and os.path.exists(files["config"]):
                try:
                    with open(files["config"], 'r') as f:
                        saved = yaml.safe_load(f)
                        if saved:
                            self.config.update(saved)
                            self.strategy.config = self.config
                            # Restore intermediate state (e.g. threshold)
                            if 'saved_state' in self.config:
                                s = self.config['saved_state']
                                if 'segmentation_threshold' in s:
                                    self.strategy.intermediate_state['segmentation_threshold'] = \
                                        float(s['segmentation_threshold'])
                except Exception:
                    pass

            msg = QMessageBox()
            if checkpoint_step == self.num_steps:
                msg.setText("All steps complete.")
                msg.setInformativeText("View results or restart from beginning?")
                view = msg.addButton("View Results", QMessageBox.YesRole)
                msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()
                
                if msg.clickedButton() == view:
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                else:
                    self._confirm_restart()
            else:
                msg.setText("Resume previous session?")
                msg.setInformativeText(f"Found data up to Step {checkpoint_step}.\n"
                                       f"Resume from Step {checkpoint_step + 1}?")
                res = msg.addButton("Resume", QMessageBox.YesRole)
                msg.addButton("Restart", QMessageBox.NoRole)
                msg.exec_()
                
                if msg.clickedButton() == res:
                    # Restore state
                    self.strategy.intermediate_state['original_volume_ref'] = self.image_stack
                    self.load_checkpoint_data(checkpoint_step)
                    self.current_step["value"] = checkpoint_step
                    
                    if checkpoint_step < self.num_steps:
                        self.create_step_widgets(
                            self.processing_steps[checkpoint_step]
                        )
                else:
                    self._confirm_restart()
        else:
            self.create_step_widgets(self.processing_steps[0])

    def _confirm_restart(self) -> None:
        """Deletes old files and restarts from Step 1."""
        reply = QMessageBox.question(
            self.viewer.window._qt_window,
            "Confirm Restart",
            "This will delete all existing processing files for this mode.\nAre you sure?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.delete_all_checkpoint_files()
            self.current_step["value"] = 0
            self.strategy.intermediate_state = {}
            self.config = self.initial_config.copy()
            self.strategy.config = self.config
            self._initialize_layers()
            self.create_step_widgets(self.processing_steps[0])
        else:
            self.restore_from_checkpoint()

    def delete_all_checkpoint_files(self) -> None:
        """Helper to clear disk artifacts."""
        for _, path in self.strategy.get_checkpoint_files().items():
            self.strategy._remove_file_safely(path)

    def load_checkpoint_data(self, checkpoint_step: int) -> None:
        """Loads visualization data."""
        self.strategy.load_checkpoint_data(self.viewer, checkpoint_step)

    def cleanup_step(self, step_number: int) -> None:
        """Cleans artifacts for a specific step."""
        self.strategy.cleanup_step_artifacts(self.viewer, step_number)

    # --- Step Execution ---

    def execute_processing_step(self) -> None:
        """
        Triggers execution of the current step in a background thread.
        Handles UI locking, log redirection, and parameter validation.
        """
        step_index = self.current_step["value"]
        if step_index >= self.num_steps:
            return

        logical_step = self.processing_steps[step_index]
        step_display = self.step_display_names.get(
            logical_step, f"Step {step_index + 1}"
        )
        
        current_values = self.get_current_values()
        
        # Validation for Interaction Step
        if logical_step == "execute_interaction_analysis":
            if not current_values.get("target_channel_folder"):
                QMessageBox.warning(
                    None, "Missing Input",
                    "Please select a Reference Channel folder first."
                )
                return

        # Prepare Execution
        self.strategy.save_config(self.config)
        
        # If not repeating interaction analysis, clean subsequent steps
        if logical_step != "execute_interaction_analysis":
            for i in range(step_index + 1, self.num_steps + 1):
                self.cleanup_step(i)
        
        # Ensure state has image reference
        if 'original_volume_ref' not in self.strategy.intermediate_state:
            self.strategy.intermediate_state['original_volume_ref'] = self.image_stack

        # Setup UI
        if self.log_widget:
            self.log_widget.clear()
            self.log_widget.append(f"--- Starting {step_display} ---\n")
        
        self._set_ui_busy(True)
        self.process_started.emit()

        # Redirect Stdout
        self.output_stream = OutputStream()
        self.output_stream.text_written.connect(self._append_log)
        sys.stdout = self.output_stream
        sys.stderr = self.output_stream

        # Start Worker
        self.worker = StepWorker(
            self.strategy, step_index, self.image_stack, current_values
        )
        self.worker.finished_signal.connect(self._on_step_finished)
        self.worker.start()

    def _append_log(self, text: str) -> None:
        """Appends text to the GUI log widget (Thread-safe)."""
        # Python check: object might be None if cleaned up
        if not self.log_widget:
            return
            
        try:
            # C++ check: might raise RuntimeError if wrapped object deleted
            cursor = self.log_widget.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertText(text)
            self.log_widget.setTextCursor(cursor)
            self.log_widget.ensureCursorVisible()
        except RuntimeError:
            # The widget has been destroyed by Qt
            self.log_widget = None

    def _on_step_finished(self, success: bool) -> None:
        """Callback when the worker thread finishes."""
        # 1. Immediate Stdout Restoration
        # This prevents print statements from crashing if the log widget is gone
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        self.worker = None
        
        # 2. Re-enable UI (if it exists)
        try:
            self._set_ui_busy(False)
        except RuntimeError:
            # Main window closed during processing
            return
        
        step_index = self.current_step["value"]
        logical_step = self.processing_steps[step_index]
        step_display = self.step_display_names.get(
            logical_step, f"Step {step_index + 1}"
        )

        if success:
            try:
                if self.log_widget:
                    self.log_widget.append(f"\n--- {step_display} COMPLETED ---")
            except RuntimeError:
                pass
            
            # Visualization Phase (Main Thread)
            try: 
                self.strategy.load_checkpoint_data(self.viewer, step_index + 1)
            except Exception as e:
                err_msg = f"\n!!! Visualization Failed !!!\n{str(e)}"
                try:
                    if self.log_widget:
                        self.log_widget.append(err_msg)
                except RuntimeError:
                    pass
                print(f"Viz Error: {e}")
                traceback.print_exc()

            self.strategy.save_config(self.config)
            
            if logical_step == "execute_interaction_analysis":
                try:
                    QMessageBox.information(
                        None, "Analysis Complete",
                        "Interaction analysis finished.\n"
                        "You can select another channel to compare against."
                    )
                except Exception:
                    pass
            else:
                self.current_step["value"] += 1
                if self.current_step["value"] < self.num_steps:
                    next_step = self.processing_steps[self.current_step["value"]]
                    self.create_step_widgets(next_step)
                else:
                    self.clear_current_widgets()
                    try:
                        QMessageBox.information(None, "Complete", "All steps finished.")
                    except Exception:
                        pass
        else:
            try:
                if self.log_widget:
                    self.log_widget.append(f"\n!!! {step_display} FAILED !!!")
                QMessageBox.warning(None, "Step Failed", "Check log for details.")
            except Exception:
                pass
        
        self.process_finished.emit()

    def _set_ui_busy(self, is_busy: bool) -> None:
        """Disables/Enables parameter widgets during processing."""
        if not self.current_widgets:
            return
            
        # Iterate safely
        for dock in list(self.current_widgets.keys()):
            try:
                # Check C++ validity
                if dock.widget():
                    dock.widget().setEnabled(not is_busy)
            except RuntimeError:
                continue

    def clear_current_widgets(self) -> None:
        """Removes current parameter widgets from the viewer."""
        for dock in list(self.current_widgets.keys()):
            try:
                self.viewer.window.remove_dock_widget(dock)
            except Exception:
                pass
        self.current_widgets.clear()

    # --- Widget Creation ---

    def create_step_widgets(self, step_method_name: str) -> None:
        """Generates parameter widgets for the given step."""
        self.clear_current_widgets()
        self.parameter_values = {}
        
        config_key = self.strategy.get_config_key(step_method_name)
        step_display = self.step_display_names.get(step_method_name, step_method_name)
        
        # Special Case: Interaction Analysis
        if step_method_name == "execute_interaction_analysis":
            self.create_interaction_widgets(step_display, config_key)
            return

        # Generic Case: Params from Config
        step_config = self.config.get(config_key, {})
        parameters = step_config.get("parameters", {}) if isinstance(step_config, dict) else {}

        scroll_w = QWidget()
        scroll_l = QVBoxLayout(scroll_w)
        lbl = QLabel(f"Parameters: {step_display}")
        lbl.setStyleSheet("font-weight: bold;")
        scroll_l.addWidget(lbl)

        if isinstance(parameters, dict):
            for pname, pconf in parameters.items():
                try:
                    # Create callback closure
                    cb = lambda val, k=config_key, p=pname: self.parameter_changed(k, p, val)
                    
                    # Create widget using helper
                    w = create_parameter_widget(pname, pconf, cb)
                    if w: 
                        scroll_l.addWidget(w.native)
                        self.parameter_values[pname] = pconf.get('value')
                except Exception:
                    pass
        
        self._add_log_widget(scroll_l)
        self._dock_widget(scroll_w, step_display)

    def create_interaction_widgets(self, step_display: str, config_key: str) -> None:
        """Creates specialized widgets for the Interaction Analysis step."""
        scroll_w = QWidget()
        layout = QVBoxLayout(scroll_w)
        
        lbl = QLabel(f"{step_display}")
        lbl.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(lbl)
        
        desc = QLabel(
            "Select a processed project folder for another channel (e.g. Plaques/Vessels)."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # File Selection Button
        self.btn_select_ref = QPushButton("Analyze with Other Channels...")
        self.btn_select_ref.setStyleSheet("padding: 8px; font-weight: bold;")
        self.btn_select_ref.clicked.connect(self.select_reference_channel)
        layout.addWidget(self.btn_select_ref)
        
        self.lbl_ref_path = QLabel("No reference selected")
        self.lbl_ref_path.setStyleSheet("color: #666; font-style: italic; margin-bottom: 10px;")
        self.lbl_ref_path.setWordWrap(True)
        layout.addWidget(self.lbl_ref_path)

        # Other parameters (bools)
        step_config = self.config.get(config_key, {})
        parameters = step_config.get("parameters", {})
        
        if isinstance(parameters, dict):
            for pname, pconf in parameters.items():
                if pname == "target_channel_folder":
                    continue
                try:
                    cb = lambda val, k=config_key, p=pname: self.parameter_changed(k, p, val)
                    w = create_parameter_widget(pname, pconf, cb)
                    if w: 
                        layout.addWidget(w.native)
                        self.parameter_values[pname] = pconf.get('value')
                except Exception:
                    pass

        self._add_log_widget(layout)
        self._dock_widget(scroll_w, step_display)

    def select_reference_channel(self) -> None:
        """Opens dialog to select reference project folder."""
        start_dir = os.path.dirname(self.inputdir)
        folder = QFileDialog.getExistingDirectory(
            None, "Select Reference Channel Project", start_dir
        )
        
        if folder:
            self.parameter_values['target_channel_folder'] = folder
            display_name = os.path.basename(folder)
            self.lbl_ref_path.setText(f"Selected: {display_name}")
            self.lbl_ref_path.setStyleSheet("color: #2E8B57; font-weight: bold;")

    def _add_log_widget(self, layout: QVBoxLayout) -> None:
        """Adds a log text box to the layout."""
        layout.addWidget(QLabel("Log:"))
        self.log_widget = QTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setMinimumHeight(150)
        self.log_widget.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.log_widget)
        layout.addStretch()

    def _dock_widget(self, widget: QWidget, name: str) -> None:
        """Docks the given widget into the Napari window."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        dock = self.viewer.window.add_dock_widget(
            scroll, area="right", name=f"Step: {name}"
        )
        self.current_widgets[dock] = scroll

    def parameter_changed(
        self, config_key: str, param_name: str, value: Any
    ) -> None:
        """Updates internal config when UI widgets change."""
        try:
            self.config[config_key]["parameters"][param_name]["value"] = value
            self.parameter_values[param_name] = value
        except Exception:
            pass

    def get_current_values(self) -> Dict[str, Any]:
        """Returns the current state of parameters."""
        return self.parameter_values.copy()