"""project_scaffolding: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
import shutil
import gc
import yaml  # type: ignore
import pandas as pd
import tifffile as tiff  # type: ignore
from typing import Dict, Any, List, Optional, Callable

from .gui_text_utils import clean_filename_for_matching, natural_sort_key
from .metadata import MetadataExtractor



def scan_available_presets() -> Dict[str, Dict[str, str]]:
    """Discover configuration presets, merging in-repo built-ins with the user's
    config library.

    Kept here as the stable public entry point (``helper_funcs`` and the organize
    wizard import it from this module), but the logic now lives in
    ``config_library`` so built-in and user-library configs are surfaced together,
    each labelled with its source, with every preset's ``default_mode`` read from
    the file's explicit ``mode``. Only valid configs are returned; malformed ones
    are surfaced (with their error) in the Config Library manager instead.

    Returns:
        ``{label: {"path", "default_mode", "source"}}``. The label's first
        whitespace token remains the clean config name, preserving
        ``channel_target_name``.
    """
    # Imported lazily to avoid any import-order coupling at module load.
    from .config_library import scan_available_presets as _impl
    return _impl()

def _execute_params_differ(new_cfg: Dict[str, Any], old_cfg: Dict[str, Any]) -> bool:
    """True if the two configs' ``execute_*`` parameter *values* differ.

    Compares only the tuned values (not labels/ranges/metadata), so a config
    that merely carries the same values in a newer schema doesn't read as a
    change. Used to decide whether applying a new config to a processed folder
    actually invalidates that folder's results."""
    def _values(cfg: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, block in (cfg or {}).items():
            if key.startswith("execute_") and isinstance(block, dict):
                params = block.get("parameters") or {}
                out[key] = {
                    pname: (pconf or {}).get("value")
                    for pname, pconf in params.items()
                    if isinstance(pconf, dict)
                }
        return out
    return _values(new_cfg) != _values(old_cfg)


def _clear_processed_result_files(folder_path: str, details: Dict[str, Any]) -> int:
    """Delete a folder's on-disk result artifacts so it reads as unprocessed.

    The processed config (and the folder's main YAML) are left in place; only the
    computed checkpoint files are removed. This is used when a new config is
    applied to an already-processed folder: the old results came from different
    parameters, so leaving them would make the folder open showing stale data
    (the reported bug). Returns the number of files removed.

    Checkpoint paths are taken from the strategy itself (the single source of
    truth) rather than re-derived here, so this can't drift from what the
    pipeline actually writes. Best-effort throughout: any failure returns 0
    rather than raising, since a failed cleanup must not abort a config apply.
    """
    tif_file = details.get("tif_file")
    mode = details.get("mode")
    yaml_file = details.get("yaml_file")
    if not tif_file or not yaml_file or not mode or mode in ("unknown", "error"):
        return 0

    basename = os.path.splitext(tif_file)[0]
    proc_dir = os.path.join(folder_path, f"{basename}_processed_{mode}")
    if not os.path.isdir(proc_dir):
        return 0

    try:
        with open(os.path.join(folder_path, yaml_file), "r") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        cfg = {}

    try:
        from ..module_3d._3D_strategy import FluorescenceStrategy  # type: ignore
        from ..module_2d._2D_strategy import Fluorescence2DStrategy  # type: ignore
    except Exception:
        return 0
    strat_cls = {
        "fluorescence": FluorescenceStrategy,
        "fluorescence_2d": Fluorescence2DStrategy,
    }.get(mode)
    if strat_cls is None:
        return 0

    # Shape/spacing don't affect checkpoint *paths* (only processed_dir + mode),
    # so neutral values are fine — mirrors the status probe in project_selection.
    try:
        strat = strat_cls(
            config=cfg, processed_dir=proc_dir, image_shape=(1, 1, 1),
            spacing=(1.0, 1.0, 1.0), scale_factor=1.0,
        )
        files = strat.get_checkpoint_files()
    except Exception:
        return 0

    removed = 0
    for path in files.values():
        try:
            if path and os.path.isfile(path):
                os.remove(path)
                removed += 1
        except OSError:
            pass
    return removed


def apply_template_config_to_project(
    template_yaml_path: str,
    project_manager: Any,
    target_mode: Optional[str] = None,
    reconcile_confirm: Optional[Callable[[Any], bool]] = None,
    clear_stale_results: bool = False
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
        reconcile_confirm:  Optional callback used to reconcile the template
                            against the current canonical built-in *before*
                            anything is written. When supplied and the template
                            differs from the pipeline's current step/parameter
                            schema, it is called with a ``ReconcileResult``; it
                            must return True to proceed (using the reconciled
                            config) or False to abort. Any problem (missing mode,
                            no canonical reference) is raised to the caller to
                            show the user -- never worked around silently. When
                            ``None`` the template is applied verbatim (legacy
                            behaviour), so existing callers are unaffected.

    Returns:
        dict with keys ``success``, ``failed``, ``skipped``, ``updated_folders``,
        and ``aborted`` (True only if ``reconcile_confirm`` declined the diff).
    """
    try:
        with open(template_yaml_path, 'r') as fh:
            template: Dict[str, Any] = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise ValueError(f"Cannot read template file: {exc}") from exc

    results: Dict[str, Any] = {
        'success': 0, 'failed': 0, 'skipped': 0,
        'updated_folders': [], 'aborted': False,
        'cleared': 0, 'cleared_folders': []
    }

    # -- Reconcile the template against the canonical built-in schema --------
    # Structure comes from the current pipeline; the template's tuned values are
    # carried over. Nothing is written until the user confirms the diff, so a
    # stale template surfaces its differences instead of silently breaking. Any
    # problem (missing mode, no canonical reference) propagates to the caller.
    if reconcile_confirm is not None:
        from .config_library import reconcile as _reconcile
        recon = _reconcile(template, mode=target_mode or template.get('mode'))
        if not recon.is_clean:
            if not reconcile_confirm(recon):
                results['aborted'] = True
                return results
        template = recon.merged

    effective_filter_mode = target_mode or template.get('mode')

    # The human-facing name of the config being applied = the template's file
    # stem (e.g. a library preset "iMG_22Jul26.yaml" -> "iMG_22Jul26"). Stamped
    # into each folder's config below so the project view's Config column shows
    # the applied config, not just the folder's YAML filename.
    applied_config_name = os.path.splitext(os.path.basename(template_yaml_path))[0]

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

        # Record which config this folder was configured with, so the project
        # view can show the applied config name rather than the YAML filename.
        merged_main['config_name'] = applied_config_name

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

                    # The processed config records the parameters the on-disk
                    # results were computed from. If the incoming config changes
                    # those values, the results are now stale — clear them so the
                    # folder opens unprocessed instead of showing data from the
                    # old parameters. Gated on clear_stale_results (the caller
                    # confirms with the user first).
                    if clear_stale_results and _execute_params_differ(
                        merged_main, existing_proc
                    ):
                        n = _clear_processed_result_files(folder_path, details)
                        if n:
                            results['cleared'] += 1
                            results['cleared_folders'].append(folder_name)
                            # Stale computed state (e.g. an auto-threshold from
                            # the old params) must not carry over either.
                            existing_proc.pop('saved_state', None)

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
        # Single-channel setup reads TIFFs directly; .czi is only handled by the
        # multi-channel path, which extracts each channel to a TIFF first. Say so,
        # rather than reporting a bare "no files found" over a folder of .czi.
        czi = [f for f in all_files if f.lower().endswith('.czi')]
        if czi:
            raise ValueError(
                f"This folder contains {len(czi)} .czi file(s) but no .tif/.tiff "
                "files. Set it up as a multi-channel project so each channel is "
                "extracted to a TIFF first."
            )
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
    #
    # "Needed" means there are no usable per-image rows to drive step 4. That is
    # true in two cases: no CSV was found (the placeholder frame built above has
    # the right columns but ZERO rows), or a CSV was found that carries no
    # 'Filename' column to match on.
    #
    # This used to test only `'Filename' not in df.columns`, which never fired
    # for the no-CSV case because the placeholder frame is constructed *with* a
    # 'Filename' column. A 0-row frame then reached step 4, whose loop iterated
    # nothing: no folders created, no configs written, no images moved -- and no
    # error raised. Setup "succeeded" having done nothing, so the caller
    # re-classified the folder, still found only loose raw images, and re-opened
    # the setup wizard: the endless project-creation loop.
    if df is None or df.empty or 'Filename' not in df.columns:
        for img_file in raw_images:
            full_path = os.path.join(drctry, img_file)
            basename = os.path.splitext(img_file)[0]
            print(f"  Analyzing: {img_file}")

            try:
                meta = MetadataExtractor.read_tiff_metadata(full_path)
            except Exception:
                meta = {'found': False, 'x': 1.0, 'y': 1.0, 'z': 1.0}

            # Pixel counts come from the array, physical scale from the metadata.
            # If the array can't be read we still emit a row with neutral
            # dimensions so the image is organized (and its dimensions can be
            # corrected later in the UI) rather than silently dropped from the
            # project -- dropping every image is what produced an empty frame and
            # the loop above.
            width = height = z_slices = 1
            try:
                mem = tiff.imread(full_path)
                z_slices = mem.shape[0] if mem.ndim == 3 else 1
                height, width = mem.shape[-2], mem.shape[-1]
                del mem
                gc.collect()  # release promptly during heavy batch organization
            except Exception as exc:
                print(f"    Warning: could not read pixel data of {img_file} "
                      f"({exc}); using neutral dimensions.")

            found = bool(meta.get('found'))
            spacing_x = float(meta.get('x', 1.0)) if found else 1.0
            spacing_y = float(meta.get('y', 1.0)) if found else 1.0
            spacing_z = float(meta.get('z', 1.0)) if found else 1.0

            generated_rows.append({
                'Filename': img_file,
                'Width (um)': spacing_x * width,
                'Height (um)': spacing_y * height,
                'Depth (um)': spacing_z * z_slices,
                'Slices': z_slices,
                'Basename': basename,
            })

        if generated_rows:
            df = pd.DataFrame(generated_rows)
            try:
                df.to_csv(os.path.join(drctry, "auto_generated_metadata.csv"),
                          index=False)
                print("  Saved 'auto_generated_metadata.csv'.")
            except OSError as exc:
                # A read-only project folder must not abort setup: the rows are
                # already in memory, which is all step 4 needs.
                print(f"  Warning: could not write auto_generated_metadata.csv "
                      f"({exc}).")

    if 'Basename' not in df.columns and 'Filename' in df.columns:
        df['Basename'] = df['Filename'].apply(lambda x: os.path.splitext(str(x))[0])

    # Nothing below can organize anything from an empty frame, and returning
    # quietly here is exactly what let the caller loop back into the wizard.
    if df is None or df.empty:
        raise ValueError(
            f"Found {len(raw_images)} image file(s) in this folder but could not "
            "build metadata for any of them, so there was nothing to organize."
        )

    # 4. Create Folder Structure (Robust Matching)
    files_moved = 0
    missing_files = []
    organized_dirs = []

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

        # Tracked separately from files_moved: re-running setup over a folder
        # whose images are already in place moves nothing, but those folders are
        # still legitimately organized.
        organized_dirs.append(new_dir)

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

    # Setup must leave behind at least one organized image folder. If every row
    # matched nothing on disk (e.g. a metadata CSV listing filenames that don't
    # correspond to the actual images) the loop above completes having done
    # nothing at all. Returning normally there reports success to the caller,
    # which re-classifies the still-unorganized folder and re-opens the wizard --
    # the reported loop. Raise so the user sees what went wrong instead.
    if not organized_dirs:
        detail = ""
        if missing_files:
            examples = ", ".join(missing_files[:3])
            detail = (
                f"\n\nNone of the {len(missing_files)} filename(s) listed in the "
                f"metadata CSV matched an image on disk (e.g. {examples})."
            )
        raise ValueError(
            "Nothing was organized: no image folders could be created from the "
            f"{len(raw_images)} image file(s) in this folder.{detail}\n\n"
            "Your images were left untouched."
        )