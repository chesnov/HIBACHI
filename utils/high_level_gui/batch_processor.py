import os
import gc
import time
import traceback
import yaml  # type: ignore
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import tifffile as tiff  # type: ignore

# Corrected relative imports
try:
    from ..module_3d._3D_strategy import FluorescenceStrategy
    from ..module_2d._2D_strategy import Fluorescence2DStrategy
    from .processing_strategies import ProcessingStrategy
except ImportError as e:
    print(f"Error importing modules in batch_processor.py: {e}")
    raise

# Optional Qt import for interactive prompts
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox  # type: ignore
    HAS_QT = True
except ImportError:
    HAS_QT = False


class BatchProcessor:
    """
    Manages the sequential processing of multiple image datasets without GUI intervention.
    """

    def __init__(self, project_manager: Any):
        """
        Initialize the BatchProcessor.

        Args:
            project_manager: Instance of ProjectManager containing the list of image folders.
        """
        self.project_manager = project_manager
        self.supported_strategies = {
            "fluorescence": FluorescenceStrategy,
            "fluorescence_2d": Fluorescence2DStrategy,
        }
        print(f"BatchProcessor initialized. Supported modes: {list(self.supported_strategies.keys())}")

    def _calculate_spacing_for_batch(
        self,
        config: Dict[str, Any],
        image_shape: Tuple[int, ...]
    ) -> Tuple[Union[Tuple[float, float, float], Tuple[float, float, float]], float]:
        """
        Calculates voxel spacing based on config dimensions and image shape.

        Args:
            config: Configuration dictionary.
            image_shape: Shape of the loaded image array.

        Returns:
            Tuple containing:
            - spacing (tuple): (Z, Y, X) or (1.0, Y, X) spacing values.
            - z_scale_factor (float): Anisotropy factor for visualization/calculations.
        """
        is_2d_mode = config.get('mode', '').endswith("_2d")
        dim_section_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'
        dimensions = config.get(dim_section_key, {})

        try:
            total_x_um = float(dimensions.get('x', 1.0))
            total_y_um = float(dimensions.get('y', 1.0))
            total_z_um = 1.0 if is_2d_mode else float(dimensions.get('z', 1.0))
        except (ValueError, TypeError):
            total_x_um, total_y_um, total_z_um = 1.0, 1.0, 1.0

        num_dims = len(image_shape)
        spacing_val: Tuple[float, ...] = (1.0, 1.0, 1.0)
        z_scale_factor_val = 1.0

        if num_dims == 2:
            # 2D Case
            y_pixel_size = total_y_um / image_shape[0] if image_shape[0] > 0 else 1.0
            x_pixel_size = total_x_um / image_shape[1] if image_shape[1] > 0 else 1.0
            spacing_val = (1.0, y_pixel_size, x_pixel_size)
        elif num_dims == 3:
            # 3D Case
            z_pixel_size = total_z_um / image_shape[0] if image_shape[0] > 0 else 1.0
            y_pixel_size = total_y_um / image_shape[1] if image_shape[1] > 0 else 1.0
            x_pixel_size = total_x_um / image_shape[2] if image_shape[2] > 0 else 1.0
            spacing_val = (z_pixel_size, y_pixel_size, x_pixel_size)

            # Z-scale factor for visualization aspect ratio
            z_scale_factor_val = z_pixel_size / x_pixel_size if x_pixel_size > 1e-9 else 1.0

        return spacing_val, z_scale_factor_val

    def _scan_folder_status(self, folder_path: str) -> Dict[str, Any]:
        """
        Performs a lightweight status check on a single folder without loading pixel data.

        Reads only the TIFF header (for image shape) and the YAML config, then
        instantiates the strategy solely to call get_last_completed_step().

        Args:
            folder_path: Path to the image folder.

        Returns:
            Dict with keys:
              - 'status': 'complete' | 'partial' | 'unprocessed' | 'unsupported' | 'invalid'
              - 'last_step': int — last completed step index (0 = none)
              - 'num_steps': int — total steps for this strategy
              - 'mode': str | None
        """
        result: Dict[str, Any] = {
            'status': 'invalid',
            'last_step': 0,
            'num_steps': 0,
            'mode': None,
        }
        strategy_instance = None
        try:
            details = self.project_manager.get_image_details(folder_path)
            if details['mode'] == 'error' or not details['tif_file'] or not details['yaml_file']:
                return result

            tif_filename: str = details['tif_file']
            yaml_filename: str = details['yaml_file']
            mode: str = details.get('mode', 'unknown')
            result['mode'] = mode

            StrategyClass = self.supported_strategies.get(mode)
            if not StrategyClass:
                result['status'] = 'unsupported'
                return result

            # Read TIFF header only — no pixel data loaded into RAM.
            tif_path = os.path.join(folder_path, tif_filename)
            with tiff.TiffFile(tif_path) as tf:
                image_shape = tf.series[0].shape if tf.series else (1,)

            config_path = os.path.join(folder_path, yaml_filename)
            with open(config_path, 'r') as f:
                config_params = yaml.safe_load(f)

            basename = os.path.splitext(tif_filename)[0]
            processed_dir = os.path.join(
                folder_path, f"{basename}_processed_{mode}"
            )
            spacing, z_scale = self._calculate_spacing_for_batch(config_params, image_shape)

            strategy_instance = StrategyClass(
                config=config_params.copy(),
                processed_dir=processed_dir,
                image_shape=image_shape,
                spacing=spacing,
                scale_factor=z_scale,
            )

            num_steps: int = strategy_instance.num_steps
            last_step: int = strategy_instance.get_last_completed_step()

            result['num_steps'] = num_steps
            result['last_step'] = last_step

            if last_step == 0:
                result['status'] = 'unprocessed'
            elif last_step >= num_steps:
                result['status'] = 'complete'
            else:
                result['status'] = 'partial'

        except Exception as e:
            print(f"  [Scan Error] {os.path.basename(folder_path)}: {e}")
        finally:
            if strategy_instance is not None:
                del strategy_instance
            gc.collect()

        return result

    def _prompt_reprocess_choice(
        self,
        complete_folders: List[str],
        partial_folders: List[str],
    ) -> str:
        """
        Displays a summary of pre-existing processed folders and asks the user
        how to proceed.

        Args:
            complete_folders: Folders where all steps are done.
            partial_folders: Folders where processing started but did not finish.

        Returns:
            One of: 'restart_all' | 'resume' | 'cancel'
              - 'restart_all' — reprocess both complete and partial from Step 1.
              - 'resume'      — resume partial from their last step; skip complete ones.
              - 'cancel'      — abort the entire batch run.
        """
        # ---- Build summary text ----
        lines = ["The following folders have existing processing output:\n"]

        if complete_folders:
            lines.append(f"  Fully complete  ({len(complete_folders)} folder(s)):")
            for fp in complete_folders:
                lines.append(f"    • {os.path.basename(fp)}")
            lines.append("")

        if partial_folders:
            lines.append(f"  Partially complete  ({len(partial_folders)} folder(s)):")
            for fp in partial_folders:
                lines.append(f"    • {os.path.basename(fp)}")
            lines.append("")

        lines.append("How would you like to proceed?")
        summary_text = "\n".join(lines)

        # ---- Qt dialog (preferred) ----
        if HAS_QT and QApplication.instance():
            msg = QMessageBox()
            msg.setWindowTitle("Existing Processing Output Detected")
            msg.setText(summary_text)
            msg.setIcon(QMessageBox.Question)

            # Dynamically label the buttons based on what was found.
            if complete_folders and partial_folders:
                btn_restart = msg.addButton(
                    "Restart All (from Step 1)", QMessageBox.ResetRole
                )
                btn_resume = msg.addButton(
                    "Resume Partial  /  Skip Complete", QMessageBox.AcceptRole
                )
            elif complete_folders:
                btn_restart = msg.addButton(
                    "Reprocess Complete Folders", QMessageBox.ResetRole
                )
                btn_resume = msg.addButton(
                    "Skip Complete Folders", QMessageBox.AcceptRole
                )
            else:
                # partial only
                btn_restart = msg.addButton(
                    "Restart Partial Folders (from Step 1)", QMessageBox.ResetRole
                )
                btn_resume = msg.addButton(
                    "Resume Partial Folders", QMessageBox.AcceptRole
                )

            btn_cancel = msg.addButton("Cancel Batch Run", QMessageBox.RejectRole)
            msg.setDefaultButton(btn_resume)
            msg.exec_()

            clicked = msg.clickedButton()
            if clicked == btn_restart:
                return 'restart_all'
            elif clicked == btn_resume:
                return 'resume'
            else:
                return 'cancel'

        # ---- Console fallback (no Qt available) ----
        print("\n" + "=" * 60)
        print(summary_text)
        if complete_folders and partial_folders:
            print("  [1] Restart All — reprocess complete + partial from Step 1")
            print("  [2] Resume — resume partial, skip complete  (default)")
        elif complete_folders:
            print("  [1] Reprocess complete folders")
            print("  [2] Skip complete folders  (default)")
        else:
            print("  [1] Restart partial folders from Step 1")
            print("  [2] Resume partial folders  (default)")
        print("  [3] Cancel batch run")
        print("=" * 60)

        try:
            raw = input("Choice [1/2/3, default=2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return 'cancel'

        if raw == '1':
            return 'restart_all'
        elif raw == '3':
            return 'cancel'
        else:
            return 'resume'

    def process_single_folder(
        self,
        folder_path: str,
        target_strategy_key: str,
        force_restart: bool = False
    ) -> bool:
        """
        Runs the processing pipeline for a single image folder.

        Args:
            folder_path: Path to the specific image folder.
            target_strategy_key: Processing mode (e.g., 'fluorescence').
            force_restart: If True, deletes previous outputs and runs from Step 1.

        Returns:
            bool: True if processing completed successfully (or was already done), False otherwise.
        """
        folder_name = os.path.basename(folder_path)
        print(f"\n--- Batch Processing: {folder_name} ---")
        print(f"    Mode: {target_strategy_key} | Force Restart: {force_restart}")

        start_time_folder = time.time()
        image_stack = None
        strategy_instance = None

        try:
            # 1. Validation
            details = self.project_manager.get_image_details(folder_path)
            if details['mode'] == 'error' or not details['tif_file'] or not details['yaml_file']:
                print(f"  [Error] Invalid folder structure or missing files. Skipping.")
                return False

            tif_filename, yaml_filename = details['tif_file'], details['yaml_file']
            StrategyClass = self.supported_strategies.get(target_strategy_key)

            if not StrategyClass:
                print(f"  [Error] Strategy '{target_strategy_key}' not supported.")
                return False

            # 2. Load Image
            try:
                image_file_path = os.path.join(folder_path, tif_filename)
                # Use mmap_mode to prevent full RAM load
                image_stack = tiff.memmap(image_file_path, mode='r')
            except MemoryError:
                print(f"  [CRITICAL] Out of Memory loading {tif_filename}. Skipping.")
                return False
            except Exception as e:
                print(f"  [Error] Loading TIFF failed: {e}")
                return False

            if image_stack.size == 0:
                raise ValueError("Image stack is empty.")

            # 3. Load Config
            config_path = os.path.join(folder_path, yaml_filename)
            with open(config_path, 'r') as f:
                config_params = yaml.safe_load(f)

            basename = os.path.splitext(tif_filename)[0]
            processed_dir = os.path.join(
                folder_path, f"{basename}_processed_{target_strategy_key}"
            )

            spacing, z_scale = self._calculate_spacing_for_batch(
                config_params, image_stack.shape
            )

            # 4. Instantiate Strategy
            strategy_instance = StrategyClass(
                config=config_params.copy(),
                processed_dir=processed_dir,
                image_shape=image_stack.shape,
                spacing=spacing,
                scale_factor=z_scale
            )

            # 5. Check State
            num_total_steps = strategy_instance.num_steps
            last_completed_step = 0

            if not force_restart:
                last_completed_step = strategy_instance.get_last_completed_step()
                if last_completed_step == num_total_steps:
                    print(f"  [Skip] All {num_total_steps} steps already completed.")
                    return True

            current_step_idx = 0
            if not force_restart and last_completed_step > 0:
                current_step_idx = last_completed_step
                print(f"  [Resume] Resuming from Step {current_step_idx + 1}.")

                # Restore intermediate state from config if available
                loaded_state = config_params.get('saved_state', {})
                if loaded_state and hasattr(strategy_instance, 'intermediate_state'):
                    if 'segmentation_threshold' in loaded_state:
                        strategy_instance.intermediate_state['segmentation_threshold'] = \
                            float(loaded_state['segmentation_threshold'])

            # 6. Cleanup future artifacts (if restarting or re-running from middle)
            start_cleanup = 1 if force_restart else (current_step_idx + 1)
            if start_cleanup <= num_total_steps:
                print(f"  [Cleanup] Clearing artifacts for steps {start_cleanup} to {num_total_steps}...")
                for i in range(start_cleanup, num_total_steps + 1):
                    strategy_instance.cleanup_step_artifacts(viewer=None, step_number=i)

            gc.collect()

            # 7. Execution Loop
            all_success = True

            for step_idx in range(current_step_idx, num_total_steps):
                step_def = strategy_instance.steps[step_idx]
                method_name = step_def['method']
                step_num = step_idx + 1

                # Handle 2D method naming convention if needed
                actual_method = method_name
                if strategy_instance.mode_name.endswith("_2d") and not method_name.endswith("_2d"):
                    # Check if the 2D strategy actually uses suffix or not.
                    # Based on refactor, names in get_step_definitions match method names directly.
                    # So strict mapping is preferred.
                    if hasattr(strategy_instance, f"{method_name}_2d"):
                        actual_method = f"{method_name}_2d"

                print(f"  [Exec] Step {step_num}/{num_total_steps}: {method_name}...")
                start_step = time.time()

                # Extract parameters
                cfg_key = strategy_instance.get_config_key(method_name)
                step_params = {}
                if cfg_key in strategy_instance.config:
                    entry = strategy_instance.config[cfg_key]
                    if isinstance(entry, dict) and "parameters" in entry:
                        for k, v in entry["parameters"].items():
                            step_params[k] = v.get('value', v) if isinstance(v, dict) else v

                # Inject Image Reference into State
                if hasattr(strategy_instance, 'intermediate_state') and \
                   'original_volume_ref' not in strategy_instance.intermediate_state:
                    strategy_instance.intermediate_state['original_volume_ref'] = image_stack

                # Run Step
                try:
                    method = getattr(strategy_instance, actual_method)
                    # Viewer is None in batch mode
                    success = method(viewer=None, image_stack=image_stack, params=step_params)
                except Exception as e:
                    print(f"    [Error] Exception in step {step_num}: {e}")
                    traceback.print_exc()
                    success = False

                if success:
                    print(f"    -> Done ({time.time() - start_step:.2f}s).")
                    if hasattr(strategy_instance, 'save_config'):
                        strategy_instance.save_config(strategy_instance.config)
                else:
                    print(f"    -> FAILED.")
                    all_success = False
                    break

                # Intermediate GC
                gc.collect()

            if all_success:
                print(f"  [Success] Folder complete ({time.time() - start_time_folder:.2f}s).")
                return True
            return False

        except Exception as e:
            print(f"  [Fatal] Uncaught exception processing folder: {e}")
            traceback.print_exc()
            return False

        finally:
            # Aggressive Cleanup
            if strategy_instance is not None:
                # Explicitly wipe the temp directory defined in the strategy
                if hasattr(strategy_instance, 'cleanup_temporary_files'):
                    strategy_instance.cleanup_temporary_files()
                if hasattr(strategy_instance, 'intermediate_state'):
                    strategy_instance.intermediate_state.clear()
                del strategy_instance
            if image_stack is not None:
                del image_stack
            gc.collect()

    def process_all_folders(self, force_restart_all: bool = False) -> Tuple[int, int, int]:
        """
        Iterates through all valid folders in the project manager and processes them.

        Before processing begins, scans all folders for existing output. If any
        fully or partially processed folders are found (and force_restart_all is
        False), the user is prompted to choose how to handle them.

        Args:
            force_restart_all: If True, skips the pre-scan prompt and forces
                               reprocessing of all folders from Step 1.

        Returns:
            Tuple (success_count, failed_count, skipped_count).
        """
        if not self.project_manager or not self.project_manager.image_folders:
            print("No images found in project.")
            return 0, 0, 0

        total = len(self.project_manager.image_folders)
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING: {total} Folder(s) Found")
        print(f"{'='*60}")

        # ------------------------------------------------------------------
        # PRE-SCAN: categorise every folder before touching any data.
        # ------------------------------------------------------------------
        restart_complete = force_restart_all
        restart_partial = force_restart_all

        if not force_restart_all:
            print("\nScanning folders for existing processing output...")
            complete_folders: List[str] = []
            partial_folders: List[str] = []
            scan_results: Dict[str, Dict[str, Any]] = {}

            for fp in self.project_manager.image_folders:
                status_info = self._scan_folder_status(fp)
                scan_results[fp] = status_info
                name = os.path.basename(fp)
                status = status_info['status']

                if status == 'complete':
                    complete_folders.append(fp)
                    print(
                        f"  [Complete]     {name}  "
                        f"({status_info['num_steps']}/{status_info['num_steps']} steps)"
                    )
                elif status == 'partial':
                    partial_folders.append(fp)
                    print(
                        f"  [Partial]      {name}  "
                        f"({status_info['last_step']}/{status_info['num_steps']} steps done)"
                    )
                elif status == 'unprocessed':
                    print(f"  [Unprocessed]  {name}")
                elif status == 'unsupported':
                    print(f"  [Unsupported]  {name}  (mode: {status_info['mode']})")
                else:
                    print(f"  [Invalid]      {name}")

            # Prompt only when there is something already processed.
            if complete_folders or partial_folders:
                choice = self._prompt_reprocess_choice(complete_folders, partial_folders)

                if choice == 'cancel':
                    print("\nBatch run cancelled by user.")
                    return 0, 0, 0

                restart_complete = (choice == 'restart_all')
                restart_partial = (choice == 'restart_all')
                # 'resume': restart_complete=False, restart_partial=False
                #   → complete folders will be skipped inside process_single_folder
                #   → partial folders will be resumed from their last step
            else:
                print("  No existing output found. Starting fresh.")
        else:
            # Populate scan_results with a sentinel so the loop below can still
            # resolve per-folder force_restart without a second scan.
            scan_results = {
                fp: {'status': 'unknown'} for fp in self.project_manager.image_folders
            }

        # ------------------------------------------------------------------
        # MAIN PROCESSING LOOP
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING STARTED")
        print(f"{'='*60}")

        start_batch = time.time()
        success, failed, skipped = 0, 0, 0

        for i, fp in enumerate(self.project_manager.image_folders):
            print(f"\nProcessing {i+1}/{total}...")

            details = self.project_manager.get_image_details(fp)
            mode = details.get('mode', 'unknown')

            if mode not in self.supported_strategies:
                if details['mode'] != 'error':
                    print(f"  [Skip] {os.path.basename(fp)} — unsupported mode: {mode}")
                else:
                    print(f"  [Skip] {os.path.basename(fp)} — invalid folder.")
                skipped += 1
                continue

            # Resolve the force_restart flag for this specific folder.
            folder_status = scan_results.get(fp, {}).get('status', 'unknown')
            if folder_status == 'complete':
                force_this = restart_complete
            elif folder_status == 'partial':
                force_this = restart_partial
            else:
                # Unprocessed, unknown (force_restart_all path), or scan error:
                # always run from the beginning.
                force_this = False

            result = self.process_single_folder(fp, mode, force_restart=force_this)
            if result:
                success += 1
            else:
                failed += 1

            # Small delay to allow OS to flush file IO buffers
            time.sleep(0.2)

        print(f"\n{'='*60}")
        print(f"BATCH SUMMARY")
        print(f"Total Time: {time.time() - start_batch:.2f}s")
        print(f"Successful: {success}")
        print(f"Failed:     {failed}")
        print(f"Skipped:    {skipped}")
        print(f"{'='*60}")

        return success, failed, skipped