"""project_scaffolding: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
import shutil
import gc
import yaml  # type: ignore
import pandas as pd
import tifffile as tiff  # type: ignore
from typing import Dict, Any, List, Optional

from .gui_text_utils import clean_filename_for_matching, natural_sort_key
from .metadata import MetadataExtractor



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
            cfg['synthetic'] = False  # real extracted channel
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
            cfg['synthetic'] = False  # real image (not procedurally generated)
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