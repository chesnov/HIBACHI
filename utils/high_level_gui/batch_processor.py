# --- START OF FILE utils/high_level_gui/batch_processor.py ---
import os
import tifffile as tiff # type: ignore
import yaml # type: ignore
import numpy as np
import traceback
import gc
import time

# Corrected relative imports
try:
    from ..module_3d._3D_strategy import RamifiedStrategy
    from ..module_2d._2D_strategy import Ramified2DStrategy
    from .processing_strategies import ProcessingStrategy
except ImportError as e:
    print(f"Error importing modules in batch_processor.py: {e}")
    raise


class BatchProcessor:
    """
    Manages the sequential processing of multiple image datasets.
    """

    def __init__(self, project_manager):
        self.project_manager = project_manager
        self.supported_strategies = {
            "ramified": RamifiedStrategy,
            "ramified_2d": Ramified2DStrategy,
        }
        print("BatchProcessor initialized with supported strategies:", list(self.supported_strategies.keys()))

    def _calculate_spacing_for_batch(self, config, image_shape):
        """Calculates voxel spacing."""
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
        spacing_val = [1.0, 1.0, 1.0] 
        z_scale_factor_val = 1.0

        if num_dims == 2:
            y_pixel_size = total_y_um / image_shape[0] if image_shape[0] > 0 else 1.0
            x_pixel_size = total_x_um / image_shape[1] if image_shape[1] > 0 else 1.0
            spacing_val = [1.0, y_pixel_size, x_pixel_size]
        elif num_dims == 3:
            z_pixel_size = total_z_um / image_shape[0] if image_shape[0] > 0 else 1.0
            y_pixel_size = total_y_um / image_shape[1] if image_shape[1] > 0 else 1.0
            x_pixel_size = total_x_um / image_shape[2] if image_shape[2] > 0 else 1.0
            spacing_val = [z_pixel_size, y_pixel_size, x_pixel_size]
            z_scale_factor_val = z_pixel_size / x_pixel_size if x_pixel_size > 1e-9 else 1.0
            
        return spacing_val, z_scale_factor_val

    def process_single_folder(self, folder_path: str, target_strategy_key: str, force_restart: bool = False):
        print(f"\n--- Processing Folder: {os.path.basename(folder_path)} (Strategy: {target_strategy_key}, Force Restart: {force_restart}) ---")
        start_time_folder = time.time()
        
        image_stack = None
        strategy_instance = None
        
        try:
            details = self.project_manager.get_image_details(folder_path)
            if details['mode'] == 'error' or not details['tif_file'] or not details['yaml_file']:
                print(f"  Error: Invalid folder details. Skipping.")
                return False

            tif_filename, yaml_filename = details['tif_file'], details['yaml_file']
            StrategyClass = self.supported_strategies.get(target_strategy_key)
            
            if not StrategyClass:
                print(f"  Error: Strategy '{target_strategy_key}' unsupported.")
                return False

            # Load Image
            try:
                image_file_path = os.path.join(folder_path, tif_filename)
                image_stack = tiff.imread(image_file_path)
            except MemoryError:
                print(f"  CRITICAL ERROR: OOM loading {tif_filename}.")
                return False
            except Exception as e:
                 print(f"  Error loading TIFF: {e}")
                 return False

            if image_stack.size == 0: raise ValueError("Image stack empty.")

            # Load Config
            config_path = os.path.join(folder_path, yaml_filename)
            with open(config_path, 'r') as f: config_params = yaml.safe_load(f)
            
            basename = os.path.splitext(tif_filename)[0]
            processed_dir = os.path.join(folder_path, f"{basename}_processed_{target_strategy_key}")
            spacing, z_scale = self._calculate_spacing_for_batch(config_params, image_stack.shape)

            # Instantiate Strategy
            strategy_instance = StrategyClass(
                config=config_params.copy(),
                processed_dir=processed_dir,
                image_shape=image_stack.shape,
                spacing=spacing,
                scale_factor=z_scale
            )
            print(f"  Instantiated {strategy_instance.mode_name} strategy.")

            # Check State using internal method (Polymorphism)
            num_total_steps = strategy_instance.num_steps
            last_completed_step = 0
            
            if not force_restart:
                last_completed_step = strategy_instance.get_last_completed_step()
                if last_completed_step == num_total_steps:
                    print(f"  All steps completed. Skipping.")
                    return True

            current_step_idx = 0
            if not force_restart and last_completed_step > 0:
                current_step_idx = last_completed_step
                print(f"  Resuming from Step {current_step_idx + 1}.")
                
                # Restore state
                loaded_state = config_params.get('saved_state', {})
                if loaded_state and hasattr(strategy_instance, 'intermediate_state'):
                    if 'segmentation_threshold' in loaded_state:
                        strategy_instance.intermediate_state['segmentation_threshold'] = float(loaded_state['segmentation_threshold'])

            # Cleanup artifacts for steps we are about to run
            start_cleanup = 1 if force_restart else (current_step_idx + 1)
            if start_cleanup <= num_total_steps:
                print(f"  Cleaning steps {start_cleanup} to {num_total_steps}...")
                for i in range(start_cleanup, num_total_steps + 1):
                     strategy_instance.cleanup_step_artifacts(viewer=None, step_number=i)
            
            gc.collect()

            # Execution Loop
            all_success = True
            
            for step_idx in range(current_step_idx, num_total_steps):
                step_def = strategy_instance.steps[step_idx]
                method_name = step_def['method']
                step_num = step_idx + 1
                
                # 2D Handling (if applicable)
                actual_method = method_name
                if strategy_instance.mode_name.endswith("_2d"):
                    actual_method = f"{method_name}_2d"
                
                print(f"\n  Executing Step {step_num}: {method_name}...")
                start_step = time.time()

                # Get params
                cfg_key = strategy_instance.get_config_key(method_name)
                step_params = {}
                if cfg_key in strategy_instance.config:
                    entry = strategy_instance.config[cfg_key]
                    if isinstance(entry, dict) and "parameters" in entry:
                        for k, v in entry["parameters"].items():
                            step_params[k] = v.get('value', v) if isinstance(v, dict) else v

                # Prepare state
                if hasattr(strategy_instance, 'intermediate_state') and \
                   'original_volume_ref' not in strategy_instance.intermediate_state:
                    strategy_instance.intermediate_state['original_volume_ref'] = image_stack

                # Run
                try:
                    method = getattr(strategy_instance, actual_method)
                    success = method(viewer=None, image_stack=image_stack, params=step_params)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    traceback.print_exc()
                    success = False

                if success:
                    print(f"    Step {step_num} done ({time.time() - start_step:.2f}s).")
                    if hasattr(strategy_instance, 'save_config'):
                        strategy_instance.save_config(strategy_instance.config)
                else:
                    print(f"    Step {step_num} FAILED.")
                    all_success = False
                    break

            if all_success:
                print(f"  Folder complete ({time.time() - start_time_folder:.2f}s).")
                return True
            return False

        except Exception as e:
            print(f"  FATAL ERROR: {e}")
            traceback.print_exc()
            return False
        
        finally:
            if image_stack is not None: del image_stack
            if strategy_instance is not None:
                if hasattr(strategy_instance, 'intermediate_state'):
                    strategy_instance.intermediate_state.clear()
                del strategy_instance
            gc.collect()

    def process_all_folders(self, force_restart_all: bool = False):
        if not self.project_manager or not self.project_manager.image_folders:
            print("No images found.")
            return 0, 0, 0

        print(f"\n===== Batch Processing: {len(self.project_manager.image_folders)} Folders =====")
        start = time.time()
        success, failed, skipped = 0, 0, 0
        
        for fp in self.project_manager.image_folders:
            details = self.project_manager.get_image_details(fp)
            mode = details.get('mode', 'unknown')
            
            if mode in self.supported_strategies:
                if self.process_single_folder(fp, mode, force_restart=force_restart_all):
                    success += 1
                else:
                    failed += 1
            else:
                if details['mode'] != 'error':
                    print(f"  Skipping {os.path.basename(fp)} (Mode: {mode})")
                skipped += 1
            time.sleep(0.2)

        print(f"\n===== Summary: {success} OK, {failed} Fail, {skipped} Skip ({time.time() - start:.2f}s) =====")
        return success, failed, skipped
# --- END OF FILE utils/high_level_gui/batch_processor.py ---