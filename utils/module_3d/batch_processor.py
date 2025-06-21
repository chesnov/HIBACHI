# --- START OF FILE utils/ramified_module_3d/batch_processor.py ---
import os
import tifffile as tiff # type: ignore
import yaml # type: ignore
import numpy as np
import traceback
import gc
import time

# Relative imports for strategies
try:
    from ._3D_strategy import RamifiedStrategy
    from ..high_level_gui.processing_strategies import ProcessingStrategy
    # For check_processing_state, we can adapt it or use strategy's file knowledge
    from ..high_level_gui.processing_strategies import check_processing_state as global_check_processing_state
except ImportError as e:
    print(f"Error importing strategies/helpers in batch_processor.py: {e}")
    raise


class BatchProcessor:
    def __init__(self, project_manager):
        self.project_manager = project_manager
        self.supported_strategies = {
            "ramified": RamifiedStrategy,
        }
        print("BatchProcessor initialized.")

    def _calculate_spacing_for_batch(self, config, image_shape):
        """Calculates voxel/pixel spacing and Z-scale factor from config."""
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
        spacing_val = [1.0, 1.0, 1.0]
        z_scale_factor_val = 1.0

        if num_dims == 2:
            if image_shape[0] > 0: y_pixel_size = total_y_um / image_shape[0]
            if image_shape[1] > 0: x_pixel_size = total_x_um / image_shape[1]
            z_pixel_size = 1.0
            spacing_val = [z_pixel_size, y_pixel_size, x_pixel_size]
            z_scale_factor_val = 1.0
        elif num_dims == 3:
            if image_shape[0] > 0: z_pixel_size = total_z_um / image_shape[0]
            if image_shape[1] > 0: y_pixel_size = total_y_um / image_shape[1]
            if image_shape[2] > 0: x_pixel_size = total_x_um / image_shape[2]
            spacing_val = [z_pixel_size, y_pixel_size, x_pixel_size]
            z_scale_factor_val = z_pixel_size / x_pixel_size if x_pixel_size > 1e-9 else 1.0
        else:
            print(f"  Warning: Image stack has unexpected dimensions: {num_dims}. Using default spacing [1,1,1].")
        # print(f"  Batch calculated {num_dims}D Pixel/Voxel Size (Z,Y,X): {spacing_val}") # Less verbose
        # print(f"  Batch calculated Z Scale Factor for display (if used): {z_scale_factor_val:.4f}") # Less verbose
        return spacing_val, z_scale_factor_val

    def process_single_folder(self, folder_path: str, target_strategy_key: str = "ramified", force_restart: bool = False):
        """
        Processes a single image folder without GUI.
        Args:
            folder_path (str): Path to the image folder.
            target_strategy_key (str): The key for the strategy to use (e.g., "ramified").
            force_restart (bool): If True, will clean and reprocess all steps regardless of prior completion.
        """
        print(f"\n--- Processing Folder: {os.path.basename(folder_path)} (Strategy: {target_strategy_key}, Force Restart: {force_restart}) ---")
        start_time_folder = time.time()
        image_stack = None
        strategy_instance = None

        try:
            details = self.project_manager.get_image_details(folder_path)
            if details['mode'] == 'error' or not details['tif_file'] or not details['yaml_file']:
                print(f"  Error: Could not get valid details for {folder_path}. Skipping.")
                return False

            tif_filename, yaml_filename = details['tif_file'], details['yaml_file']
            config_mode_from_yaml = details['mode']

            if config_mode_from_yaml != target_strategy_key:
                print(f"  Info: Mode in YAML ('{config_mode_from_yaml}') does not match target batch strategy ('{target_strategy_key}'). Skipping.")
                return False

            StrategyClass = self.supported_strategies.get(target_strategy_key)
            if not StrategyClass:
                print(f"  Error: Strategy '{target_strategy_key}' is not supported. Skipping.")
                return False

            image_file_path = os.path.join(folder_path, tif_filename)
            image_stack = tiff.imread(image_file_path)
            print(f"  Loaded image: {tif_filename} (Shape: {image_stack.shape}, Dtype: {image_stack.dtype})")
            if image_stack.size == 0: raise ValueError("Image stack is empty.")

            expected_ndim = 2 if target_strategy_key.endswith("_2d") else 3
            if image_stack.ndim != expected_ndim:
                print(f"  Error: Expected {expected_ndim}D image for mode '{target_strategy_key}', but got {image_stack.ndim}D. Skipping.")
                return False

            config_file_path = os.path.join(folder_path, yaml_filename)
            with open(config_file_path, 'r') as f: config_params_from_yaml = yaml.safe_load(f)
            if not isinstance(config_params_from_yaml, dict): raise ValueError("YAML content is not a dictionary.")
            print(f"  Loaded config: {yaml_filename}")

            basename = os.path.splitext(tif_filename)[0]
            processed_dir = os.path.join(folder_path, f"{basename}_processed_{target_strategy_key}")

            spacing, z_scale_factor = self._calculate_spacing_for_batch(config_params_from_yaml, image_stack.shape)

            strategy_instance = StrategyClass(
                config=config_params_from_yaml.copy(),
                processed_dir=processed_dir,
                image_shape=image_stack.shape,
                spacing=spacing,
                scale_factor=z_scale_factor
            )
            print(f"  Instantiated {strategy_instance.mode_name} strategy for {processed_dir}.")

            # --- Check existing processing state ---
            checkpoint_files = strategy_instance.get_checkpoint_files()
            num_total_steps_for_strategy = strategy_instance.num_steps
            
            last_completed_step = 0 # 0-based index of last completed step, or -1 if none
            if not force_restart:
                # Use the global_check_processing_state which returns 1-based step number
                # It needs: processed_dir, mode, checkpoint_files_dict, num_steps
                # Mode here is strategy_instance.mode_name (e.g., "ramified")
                _1_based_completed_step = global_check_processing_state(
                    processed_dir, 
                    strategy_instance.mode_name, 
                    checkpoint_files, 
                    num_total_steps_for_strategy
                )
                last_completed_step = _1_based_completed_step # Store 1-based for clarity
                
                if last_completed_step == num_total_steps_for_strategy:
                    print(f"  All {num_total_steps_for_strategy} steps already completed for this folder. Skipping.")
                    return True # Consider this a success for batch summary
            
            start_step_index = 0 # Default to start from the first step (0-indexed)
            if not force_restart and last_completed_step > 0:
                start_step_index = last_completed_step # Next step to run is 1-based last_completed_step
                                                         # which is last_completed_step_0_indexed + 1
                print(f"  Resuming from Step {start_step_index + 1} (completed up to step {last_completed_step}).")
                # Load necessary intermediate state from config if resuming
                # This is crucial for strategies that pass state between steps (e.g., segmentation_threshold)
                loaded_state_from_config = config_params_from_yaml.get('saved_state', {})
                if loaded_state_from_config:
                    print(f"    Restoring intermediate state from config: {list(loaded_state_from_config.keys())}")
                    if 'segmentation_threshold' in loaded_state_from_config:
                        try:
                            thresh_val = float(loaded_state_from_config['segmentation_threshold'])
                            strategy_instance.intermediate_state['segmentation_threshold'] = thresh_val
                            print(f"      Restored 'segmentation_threshold': {thresh_val:.4f}")
                        except Exception as e_thresh:
                            print(f"      Warning: Could not restore 'segmentation_threshold' from config: {e_thresh}")
                    # Add other state restorations as needed by the strategy
                else:
                    print(f"    No 'saved_state' found in config {yaml_filename} for resuming.")

            # --- Clean up artifacts for steps that will be run ---
            # If force_restart or starting from scratch, clean all.
            # If resuming, clean artifacts from the start_step_index onwards.
            # Steps are 1-based for cleanup_step_artifacts
            steps_to_clean_from = 1 if force_restart or start_step_index == 0 else start_step_index + 1
            
            print(f"  Cleaning artifacts from step {steps_to_clean_from} onwards in {processed_dir}...")
            for i in range(steps_to_clean_from, num_total_steps_for_strategy + 1):
                 strategy_instance.cleanup_step_artifacts(viewer=None, step_number=i)
            gc.collect()


            all_steps_successful = True
            # Iterate from the determined start_step_index (0-based for list access)
            # Note: get_step_names() returns a list of method names.
            # start_step_index from global_check_processing_state is 1-based step *number* of NEXT step,
            # so convert to 0-based index for slicing/iteration.
            # If global_check_processing_state said step 2 is next, that's index 1.
            
            # If last_completed_step was 0 (meaning nothing done), start_step_index should be 0.
            # If last_completed_step was 1 (step 1 done), next to run is step 2 (index 1).
            
            # Let's adjust: last_completed_step is 1-based number of LAST COMPLETED step.
            # So if last_completed_step = 1, we need to run from index 1 (which is step 2).
            # If last_completed_step = 0, we run from index 0 (step 1).
            
            current_step_0_indexed_to_run = 0
            if not force_restart:
                current_step_0_indexed_to_run = last_completed_step # if last_completed_step is 1-based 'k', next is 'k+1', which is index 'k'
            
            print(f"  Starting execution from step {current_step_0_indexed_to_run + 1} (0-indexed: {current_step_0_indexed_to_run}).")


            for step_index in range(current_step_0_indexed_to_run, num_total_steps_for_strategy):
                step_method_name = strategy_instance.get_step_names()[step_index]
                step_display_number = step_index + 1 # 1-based for display

                print(f"\n  Executing Step {step_display_number}/{num_total_steps_for_strategy}: {step_method_name}...")
                start_time_step = time.time()

                step_config_key = strategy_instance.get_config_key(step_method_name)
                step_params_for_method = {}
                if step_config_key in config_params_from_yaml and \
                   isinstance(config_params_from_yaml.get(step_config_key), dict) and \
                   "parameters" in config_params_from_yaml[step_config_key] and \
                   isinstance(config_params_from_yaml[step_config_key].get("parameters"), dict):
                    for param_name, param_data in config_params_from_yaml[step_config_key]["parameters"].items():
                        step_params_for_method[param_name] = param_data.get('value', param_data) if isinstance(param_data, dict) else param_data
                else:
                    print(f"    Warning: No parameters found in config for step key '{step_config_key}'. Method may use defaults.")

                if 'original_volume_ref' not in strategy_instance.intermediate_state and image_stack is not None:
                    strategy_instance.intermediate_state['original_volume_ref'] = image_stack

                step_method = getattr(strategy_instance, step_method_name)
                step_success = False
                try:
                    step_success = step_method(
                        viewer=None,
                        image_stack=image_stack,
                        params=step_params_for_method
                    )
                except Exception as e_step:
                    print(f"    ERROR during execution of step method '{step_method_name}': {e_step}")
                    traceback.print_exc()
                    step_success = False

                if not isinstance(step_success, bool):
                    print(f"    Warning: Step method '{step_method_name}' did not return bool. Assuming failure.")
                    step_success = False

                if step_success:
                    print(f"    Step {step_display_number} '{step_method_name}' completed successfully. (Time: {time.time() - start_time_step:.2f}s)")
                    strategy_instance.save_config(strategy_instance.config)
                else:
                    print(f"    Step {step_display_number} '{step_method_name}' FAILED. Aborting processing for this folder.")
                    all_steps_successful = False
                    break

            if all_steps_successful:
                print(f"  All required steps completed successfully for {os.path.basename(folder_path)}.")
                folder_time = time.time() - start_time_folder
                print(f"  Total time for folder: {folder_time:.2f}s")
                return True
            return False

        except Exception as e:
            print(f"  FATAL ERROR processing folder {os.path.basename(folder_path)}: {e}")
            traceback.print_exc()
            return False
        finally:
            if image_stack is not None: del image_stack
            if strategy_instance is not None: del strategy_instance
            gc.collect()

    def process_all_folders(self, target_strategy_key: str = "ramified", force_restart_all: bool = False):
        """
        Iterates through all valid image folders and processes them.
        Args:
            target_strategy_key (str): The key for the strategy to use.
            force_restart_all (bool): If True, forces restart for ALL folders.
        """
        if not self.project_manager or not self.project_manager.project_path:
            print("Error: ProjectManager not initialized or no project loaded.")
            return 0, 0
        if not self.project_manager.image_folders:
            print("No image folders found by ProjectManager to process.")
            return 0, 0

        num_total_folders = len(self.project_manager.image_folders)
        print(f"\n===== Starting Batch Processing for {num_total_folders} Folders (Target Strategy: {target_strategy_key}, Force Restart All: {force_restart_all}) =====")
        overall_start_time = time.time()
        successful_folders, failed_folders = 0, 0

        for i, folder_path in enumerate(self.project_manager.image_folders):
            print(f"\n--- Folder {i+1}/{num_total_folders} ---")
            if self.process_single_folder(folder_path, target_strategy_key, force_restart=force_restart_all):
                successful_folders += 1
            else:
                failed_folders += 1
            print("-" * 60)

        overall_time = time.time() - overall_start_time
        print("\n===== Batch Processing Summary =====")
        print(f"Total folders considered: {successful_folders + failed_folders}") # Changed from "processed"
        print(f"Successfully processed/resumed/skipped (completed): {successful_folders}")
        print(f"Failed during processing: {failed_folders}")
        print(f"Total batch processing time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
        print("=" * 36)
        return successful_folders, failed_folders

# --- END OF FILE utils/ramified_module_3d/batch_processor.py ---