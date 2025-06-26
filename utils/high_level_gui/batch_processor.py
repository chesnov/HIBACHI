# --- START OF FILE utils/high_level_gui/batch_processor.py ---
import os
import tifffile as tiff # type: ignore
import yaml # type: ignore
import numpy as np
import traceback
import gc
import time

# Corrected relative imports from its new location in high_level_gui
try:
    from ..module_3d._3D_strategy import RamifiedStrategy
    from ..module_2d._2D_strategy import Ramified2DStrategy
    # from ..nuclear_module_3d._3D_nuclear_strategy import NuclearStrategy # Example for future
    from .processing_strategies import ProcessingStrategy # Sibling import
    from .processing_strategies import check_processing_state as global_check_processing_state # Sibling import
except ImportError as e:
    print(f"Error importing modules in high_level_gui/batch_processor.py: {e}")
    print("Ensure strategy files and helper_funcs.py are correctly located relative to utils/ and utils/high_level_gui/.")
    raise


class BatchProcessor:
    def __init__(self, project_manager):
        self.project_manager = project_manager
        self.supported_strategies = {
            "ramified": RamifiedStrategy,
            "ramified_2d": Ramified2DStrategy,
            # "nuclei": NuclearStrategy, # Add when nuclear strategy is batch-ready
        }
        print("BatchProcessor initialized with supported strategies:", list(self.supported_strategies.keys()))

    def _calculate_spacing_for_batch(self, config, image_shape):
        # --- This method remains unchanged ---
        is_2d_mode = config.get('mode', '').endswith("_2d")
        dim_section_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'
        dimensions = config.get(dim_section_key, {})
        try:
            total_x_um = float(dimensions.get('x', 1.0))
            total_y_um = float(dimensions.get('y', 1.0))
            total_z_um = 1.0
            if not is_2d_mode:
                total_z_um = float(dimensions.get('z', 1.0))
        except (ValueError, TypeError) as e:
            print(f"  Warning: Invalid total dimensions in config '{dim_section_key}': {e}. Using defaults (1.0).")
            total_x_um, total_y_um, total_z_um = 1.0, 1.0, 1.0

        num_dims = len(image_shape)
        x_pixel_size, y_pixel_size, z_pixel_size = total_x_um, total_y_um, total_z_um
        spacing_val = [1.0, 1.0, 1.0] # Default ZYX
        z_scale_factor_val = 1.0

        if num_dims == 2: # 2D: shape is (Y, X)
            if image_shape[0] > 0: y_pixel_size = total_y_um / image_shape[0]
            if image_shape[1] > 0: x_pixel_size = total_x_um / image_shape[1]
            z_pixel_size = 1.0
            spacing_val = [z_pixel_size, y_pixel_size, x_pixel_size]
            z_scale_factor_val = 1.0
        elif num_dims == 3: # 3D: shape is (Z, Y, X)
            if image_shape[0] > 0: z_pixel_size = total_z_um / image_shape[0]
            if image_shape[1] > 0: y_pixel_size = total_y_um / image_shape[1]
            if image_shape[2] > 0: x_pixel_size = total_x_um / image_shape[2]
            spacing_val = [z_pixel_size, y_pixel_size, x_pixel_size]
            z_scale_factor_val = z_pixel_size / x_pixel_size if x_pixel_size > 1e-9 else 1.0
        else:
            print(f"  Warning: Image stack has unexpected dimensions: {num_dims}. Using default spacing [1,1,1].")
        return spacing_val, z_scale_factor_val

    def process_single_folder(self, folder_path: str, target_strategy_key: str, force_restart: bool = False):
        # --- This method remains largely unchanged from the previous version ---
        # --- It expects target_strategy_key to be passed to it by process_all_folders ---
        print(f"\n--- Processing Folder: {os.path.basename(folder_path)} (Strategy: {target_strategy_key}, Force Restart: {force_restart}) ---")
        start_time_folder = time.time()
        image_stack = None
        strategy_instance = None

        try:
            details = self.project_manager.get_image_details(folder_path)
            # This initial check ensures the folder itself is valid before we even consider its mode
            if details['mode'] == 'error' or not details['tif_file'] or not details['yaml_file']:
                print(f"  Error: Could not get valid TIF/YAML details for {folder_path}. Skipping.")
                return False # Indicate failure for this folder

            tif_filename, yaml_filename = details['tif_file'], details['yaml_file']
            # config_mode_from_yaml = details['mode'] # This is the mode from YAML

            # The target_strategy_key is now determined by the caller (process_all_folders)
            # and passed to this function. So, we directly use target_strategy_key.

            StrategyClass = self.supported_strategies.get(target_strategy_key)
            if not StrategyClass:
                # This should ideally not happen if process_all_folders filters correctly
                print(f"  Error: Strategy '{target_strategy_key}' (determined for this folder) is not supported by BatchProcessor. Skipping.")
                return False

            image_file_path = os.path.join(folder_path, tif_filename)
            image_stack = tiff.imread(image_file_path)
            print(f"  Loaded image: {tif_filename} (Shape: {image_stack.shape}, Dtype: {image_stack.dtype})")
            if image_stack.size == 0: raise ValueError("Image stack is empty.")

            expected_ndim = 2 if target_strategy_key.endswith("_2d") else 3
            if image_stack.ndim != expected_ndim:
                print(f"  Error: Expected {expected_ndim}D image for mode '{target_strategy_key}', but got {image_stack.ndim}D. Skipping.")
                # Clean up loaded image if an error occurs before strategy instantiation
                if image_stack is not None: del image_stack; gc.collect()
                return False

            config_file_path = os.path.join(folder_path, yaml_filename)
            with open(config_file_path, 'r') as f: config_params_from_yaml = yaml.safe_load(f)
            if not isinstance(config_params_from_yaml, dict): raise ValueError("YAML content is not a dictionary.")
            print(f"  Loaded config: {yaml_filename}")

            basename = os.path.splitext(tif_filename)[0]
            # The processed_dir name must use the target_strategy_key for this specific run
            processed_dir = os.path.join(folder_path, f"{basename}_processed_{target_strategy_key}")

            spacing, z_scale_factor = self._calculate_spacing_for_batch(config_params_from_yaml, image_stack.shape)

            strategy_instance = StrategyClass(
                config=config_params_from_yaml.copy(),
                processed_dir=processed_dir,
                image_shape=image_stack.shape,
                spacing=spacing,
                scale_factor=z_scale_factor
            )
            print(f"  Instantiated {strategy_instance.mode_name} strategy. Processed dir: {processed_dir}.")

            checkpoint_files = strategy_instance.get_checkpoint_files()
            num_total_steps_for_strategy = strategy_instance.num_steps
            
            last_completed_step = 0
            if not force_restart:
                _1_based_completed_step = global_check_processing_state(
                    processed_dir, 
                    strategy_instance.mode_name, 
                    checkpoint_files, 
                    num_total_steps_for_strategy
                )
                last_completed_step = _1_based_completed_step
                
                if last_completed_step == num_total_steps_for_strategy:
                    print(f"  All {num_total_steps_for_strategy} steps already completed for this folder ({strategy_instance.mode_name} mode). Skipping further processing.")
                    return True

            current_step_0_indexed_to_run = 0
            if not force_restart and last_completed_step > 0:
                current_step_0_indexed_to_run = last_completed_step
                print(f"  Resuming {strategy_instance.mode_name} mode from Step {current_step_0_indexed_to_run + 1} (completed up to step {last_completed_step}).")
                
                loaded_state_from_config = config_params_from_yaml.get('saved_state', {})
                if loaded_state_from_config and hasattr(strategy_instance, 'intermediate_state'):
                    print(f"    Restoring intermediate state from config: {list(loaded_state_from_config.keys())}")
                    if 'segmentation_threshold' in loaded_state_from_config :
                        try:
                            thresh_val = float(loaded_state_from_config['segmentation_threshold'])
                            strategy_instance.intermediate_state['segmentation_threshold'] = thresh_val
                            print(f"      Restored 'segmentation_threshold': {thresh_val:.4f}")
                        except Exception as e_thresh:
                            print(f"      Warning: Could not restore 'segmentation_threshold' from config: {e_thresh}")
                elif hasattr(strategy_instance, 'intermediate_state'):
                    print(f"    No 'saved_state' found in config {yaml_filename} for resuming, or strategy does not use intermediate_state dict for this.")

            steps_to_clean_from = 1
            if force_restart or current_step_0_indexed_to_run == 0:
                steps_to_clean_from = 1
            else:
                steps_to_clean_from = current_step_0_indexed_to_run + 1

            print(f"  Cleaning artifacts for {strategy_instance.mode_name} mode from step {steps_to_clean_from} onwards in {processed_dir}...")
            for i in range(steps_to_clean_from, num_total_steps_for_strategy + 1):
                 strategy_instance.cleanup_step_artifacts(viewer=None, step_number=i)
            gc.collect()

            all_steps_successful = True
            print(f"  Starting execution of {strategy_instance.mode_name} mode from step {current_step_0_indexed_to_run + 1} (0-indexed: {current_step_0_indexed_to_run}).")

            base_step_method_names = strategy_instance.get_step_names()

            for step_index in range(current_step_0_indexed_to_run, num_total_steps_for_strategy):
                base_step_method_name = base_step_method_names[step_index]
                step_display_number = step_index + 1

                actual_method_name_to_call = base_step_method_name
                if strategy_instance.mode_name.endswith("_2d"): # Check if the strategy IS a 2D strategy
                    actual_method_name_to_call = f"{base_step_method_name}_2d"
                
                print(f"\n  Executing Step {step_display_number}/{num_total_steps_for_strategy}: {base_step_method_name} (Actual method: {actual_method_name_to_call})...")
                start_time_step = time.time()

                step_config_key = strategy_instance.get_config_key(base_step_method_name)
                step_params_for_method = {}
                current_folder_config_params = strategy_instance.config
                if step_config_key in current_folder_config_params and \
                   isinstance(current_folder_config_params.get(step_config_key), dict) and \
                   "parameters" in current_folder_config_params[step_config_key] and \
                   isinstance(current_folder_config_params[step_config_key].get("parameters"), dict):
                    for param_name, param_data in current_folder_config_params[step_config_key]["parameters"].items():
                        step_params_for_method[param_name] = param_data.get('value', param_data) if isinstance(param_data, dict) else param_data
                else:
                    print(f"    Warning: No parameters found in config for step key '{step_config_key}'. Method may use defaults or parameters from strategy's internal config if any.")

                if hasattr(strategy_instance, 'intermediate_state') and 'original_volume_ref' not in strategy_instance.intermediate_state and image_stack is not None:
                    strategy_instance.intermediate_state['original_volume_ref'] = image_stack

                try:
                    step_method = getattr(strategy_instance, actual_method_name_to_call)
                except AttributeError:
                    print(f"    FATAL ERROR: Method '{actual_method_name_to_call}' not found in strategy class '{strategy_instance.__class__.__name__}'.")
                    all_steps_successful = False
                    break 

                step_success = False
                try:
                    step_success = step_method(
                        viewer=None,
                        image_stack=image_stack,
                        params=step_params_for_method
                    )
                except Exception as e_step:
                    print(f"    ERROR during execution of step method '{actual_method_name_to_call}': {e_step}")
                    traceback.print_exc()
                    step_success = False

                if not isinstance(step_success, bool):
                    print(f"    Warning: Step method '{actual_method_name_to_call}' did not return bool. Assuming failure.")
                    step_success = False

                if step_success:
                    print(f"    Step {step_display_number} '{base_step_method_name}' completed successfully. (Time: {time.time() - start_time_step:.2f}s)")
                    if hasattr(strategy_instance, 'save_config'):
                        strategy_instance.save_config(strategy_instance.config)
                else:
                    print(f"    Step {step_display_number} '{base_step_method_name}' FAILED. Aborting processing for this folder.")
                    all_steps_successful = False
                    break

            if all_steps_successful:
                print(f"  All required steps for {strategy_instance.mode_name} mode completed successfully for {os.path.basename(folder_path)}.")
                folder_time = time.time() - start_time_folder
                print(f"  Total time for folder: {folder_time:.2f}s")
                return True
            return False

        except Exception as e:
            print(f"  FATAL ERROR processing folder {os.path.basename(folder_path)} with strategy {target_strategy_key}: {e}")
            traceback.print_exc()
            return False
        finally:
            if image_stack is not None: del image_stack
            if strategy_instance is not None: del strategy_instance
            gc.collect()

    def process_all_folders(self, force_restart_all: bool = False): # Removed target_strategy_key
        """
        Iterates through all valid image folders, determines their mode from YAML,
        and processes them using the appropriate supported strategy.
        Args:
            force_restart_all (bool): If True, forces restart for ALL folders.
        """
        if not self.project_manager or not self.project_manager.project_path:
            print("Error: ProjectManager not initialized or no project loaded.")
            return 0, 0, 0 # successful, failed, skipped
        if not self.project_manager.image_folders:
            print("No image folders found by ProjectManager to process.")
            return 0, 0, 0 # successful, failed, skipped

        num_total_folders_in_project = len(self.project_manager.image_folders)
        print(f"\n===== Starting Batch Processing for All Compatible Folders (Force Restart All: {force_restart_all}) =====")
        overall_start_time = time.time()
        successful_folders, failed_folders, skipped_unsupported_mode = 0, 0, 0
        
        folders_to_attempt_processing = []
        
        # First, identify folders that have a mode supported by this BatchProcessor
        for folder_path in self.project_manager.image_folders:
            details = self.project_manager.get_image_details(folder_path)
            folder_mode = details.get('mode', 'unknown')
            if folder_mode in self.supported_strategies:
                folders_to_attempt_processing.append((folder_path, folder_mode))
            else:
                if details['mode'] != 'error': # Don't count folders with read errors as "unsupported mode"
                    print(f"  Folder {os.path.basename(folder_path)}: Mode '{folder_mode}' is not currently supported by BatchProcessor. Skipping.")
                skipped_unsupported_mode += 1
        
        num_compatible_folders = len(folders_to_attempt_processing)
        print(f"Project contains {num_total_folders_in_project} folders. Found {num_compatible_folders} compatible with supported strategies: {list(self.supported_strategies.keys())}.")

        if num_compatible_folders == 0:
            print(f"No folders found with modes supported by this BatchProcessor. Nothing to process.")
        else:
            for i, (folder_path, folder_strategy_key) in enumerate(folders_to_attempt_processing):
                print(f"\n--- Processing Folder {i+1}/{num_compatible_folders} (Identified Mode: {folder_strategy_key}): {os.path.basename(folder_path)} ---")
                # Pass the folder_strategy_key as the target_strategy_key to process_single_folder
                if self.process_single_folder(folder_path, folder_strategy_key, force_restart=force_restart_all):
                    successful_folders += 1
                else:
                    failed_folders += 1
                print("-" * 60)

        overall_time = time.time() - overall_start_time
        print("\n===== Batch Processing Summary =====")
        print(f"Total folders in project: {num_total_folders_in_project}")
        print(f"Folders with supported modes attempted: {num_compatible_folders}")
        print(f"  Successfully processed/resumed/already complete: {successful_folders}")
        print(f"  Failed during processing:                      {failed_folders}")
        print(f"Folders skipped (unsupported mode or read error):  {skipped_unsupported_mode}")
        print(f"Total batch processing time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
        print("=" * 36)
        return successful_folders, failed_folders, skipped_unsupported_mode

# --- END OF FILE utils/high_level_gui/batch_processor.py ---