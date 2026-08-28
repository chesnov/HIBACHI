"""project_scaffolding: extracted from helper_funcs.py (auto-split along functional seams)."""


import os
import shutil
import gc
import yaml  # type: ignore
import pandas as pd
import tifffile as tiff  # type: ignore
from typing import Dict, Any, List, Optional, Callable, Sequence

from .gui_text_utils import (
    clean_filename_for_matching, is_os_sidecar, natural_sort_key,
)
from .slide_reader import SetupCancelled  # re-exported for callers
from .metadata import MetadataExtractor



# HIBACHI's own record of what it detected, written by organize_processing_dir.
# It must never be read back as user input: after a partial setup it lists only
# the images from that run, and those have since been moved into their folders --
# so a later run would match nothing and organize nothing. Same class of mistake
# as reading a channel's own metadata.csv as an input.
AUTO_METADATA_CSV = "auto_generated_metadata.csv"


def _is_generated_csv(name: str) -> bool:
    """True for a CSV this module wrote into the folder being scanned.

    Only auto_generated_metadata.csv qualifies. 'metadata.csv' is deliberately NOT
    excluded: it is a perfectly ordinary name for a USER's dimension file, and the
    per-channel metadata.csv this module writes goes into the Channel_* directory,
    never into the folder that gets scanned for input.
    """
    return os.path.basename(str(name)).lower() == AUTO_METADATA_CSV.lower()


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

        # Carry the calibration's provenance with the calibration itself. merged_main
        # is rebuilt from scratch, so anything not copied here is dropped -- and
        # dropping this would silently downgrade a calibrated image to 'unknown'
        # just because the user applied a different parameter set.
        from .dimension_entry import DIMENSION_SOURCE_KEY
        if DIMENSION_SOURCE_KEY in current_main:
            merged_main[DIMENSION_SOURCE_KEY] = current_main[DIMENSION_SOURCE_KEY]

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


def _find_dimension_csv(source_root: str) -> Optional[str]:
    """The one metadata CSV to import dimensions from, or None.

    Only files sitting *directly* in ``source_root`` are considered, so the
    per-channel ``metadata.csv`` that ``organize_channel_project`` writes into
    each ``Channel_*`` folder is never mistaken for an input on a re-run.
    """
    try:
        entries = sorted(os.listdir(source_root))
    except OSError:
        return None
    csvs = [f for f in entries
            if f.lower().endswith('.csv')
            and os.path.isfile(os.path.join(source_root, f))
            and not is_os_sidecar(f)
            and not _is_generated_csv(f)]
    if not csvs:
        return None
    if len(csvs) == 1:
        return os.path.join(source_root, csvs[0])
    # Several CSVs: prefer an obviously-named one rather than guessing.
    for f in csvs:
        if 'metadata' in f.lower():
            return os.path.join(source_root, f)
    print(f"  Warning: {len(csvs)} CSV files found; skipping CSV dimension "
          "import. Keep only one, or name it 'metadata.csv'.")
    return None


def load_dimension_overrides(source_root: str) -> Dict[str, Dict[str, Optional[float]]]:
    """Map cleaned source filename -> physical dimensions from a metadata CSV.

    Multi-channel setup derives each channel's dimensions from the source image's
    own scale metadata, but plenty of TIFFs carry no usable calibration -- or
    carry a meaningless 1-per-unit resolution tag that reads as exactly
    1 micron/pixel. A CSV sitting next to the images is how the user supplies the
    real numbers, and it is authoritative when present: the same precedence
    ``organize_processing_dir`` already gives it for single-channel projects.

    ONE ROW PER SOURCE IMAGE is all that's needed, even for a multi-channel file.
    Every channel extracted from one acquisition shares that acquisition's
    physical extent, so the row is reused for each channel rather than needing to
    be repeated per channel.

    Values are read as TOTAL microns per axis (matching the 'Width (um)' /
    'Height (um)' / 'Depth (um)' columns the single-channel path writes and
    reads). Blank, non-numeric and non-positive cells are ignored per-axis, so a
    CSV that only pins down X and Y still lets Z come from the file. Returns {}
    when there is no usable CSV, leaving metadata-derived values untouched.
    """
    path = _find_dimension_csv(source_root)
    if not path:
        return {}
    try:
        # comment=None so a '#' inside a filename doesn't truncate the field.
        df = pd.read_csv(path, comment=None)
    except Exception as exc:
        print(f"  Warning: could not read {os.path.basename(path)} ({exc}); "
              "falling back to each file's own metadata.")
        return {}
    if 'Filename' not in df.columns:
        print(f"  Warning: {os.path.basename(path)} has no 'Filename' column; "
              "falling back to each file's own metadata.")
        return {}

    def _positive(row, key) -> Optional[float]:
        try:
            val = float(row.get(key))
        except (TypeError, ValueError):
            return None
        # val == val rejects NaN (an empty cell), which must never become a size.
        return val if (val == val and val > 0) else None

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for _, row in df.iterrows():
        name = str(row.get('Filename', '')).strip()
        if not name:
            continue
        entry = {
            'x': _positive(row, 'Width (um)'),
            'y': _positive(row, 'Height (um)'),
            'z': _positive(row, 'Depth (um)'),
        }
        if any(v is not None for v in entry.values()):
            out[clean_filename_for_matching(name)] = entry

    if out:
        print(f"  Importing dimensions for {len(out)} image(s) from "
              f"{os.path.basename(path)}.")
    else:
        print(f"  Warning: {os.path.basename(path)} contained no usable "
              "Width/Height/Depth values.")
    return out


def _match_dimension_override(
    overrides: Dict[str, Dict[str, Optional[float]]], src_file: str
) -> Optional[Dict[str, Optional[float]]]:
    """Find a CSV row for a source filename.

    Mirrors the matcher in ``organize_processing_dir`` -- exact cleaned match
    first, then a substring match either way -- so a CSV listing 'sample1.czi'
    still matches 'sample1.tif', and the two paths behave the same way.
    """
    if not overrides:
        return None
    key = clean_filename_for_matching(src_file)
    if key in overrides:
        return overrides[key]
    for cand, entry in overrides.items():
        if cand and (cand in key or key in cand):
            return entry
    return None


def _has_real_scale(meta: Dict[str, Any], mode: Any = "") -> bool:
    """True if `meta` carries a physical scale worth trusting on every axis.

    The ``found`` flag alone is not enough. Many writers (tifffile included)
    store XResolution=(1,1) with ResolutionUnit=NONE on an uncalibrated image,
    which ``read_tiff_metadata`` resolves to exactly 1.0 micron/pixel and reports
    as found=True. That is indistinguishable from "no calibration at all", so a
    unit spacing is treated as missing and the user is asked instead of silently
    producing dimensions that are really just pixel counts.

    Now delegates to ``dimension_entry.scale_gaps`` so the check is per-axis and
    mode-aware. The old version tested only X and Y, so a 3D image with a genuine
    X/Y but no Z spacing passed as fully calibrated -- and Z is the axis
    microscopes most often fail to record.
    """
    from .dimension_entry import scale_gaps
    return not scale_gaps(meta, mode)

def organize_channel_project(
    source_files: List[str],
    source_root: str,
    target_root_dir: str,
    channel_idx: int,
    preset_details: Dict[str, str],
    progress=None,
    should_cancel=None,
    manual_overrides: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
) -> Dict[str, Any]:
    """Setup Logic for MULTI-CHANNEL mode.

    `progress` is called as progress(message, done, total) so a caller can show
    what is happening during what may be a multi-minute extraction; `should_cancel`
    is polled and raises SetupCancelled when it returns True.

    `manual_overrides` carries dimensions the user typed into the dimension-entry
    dialog, keyed and shaped exactly like the CSV overrides so both flow through
    one precedence chain rather than two. It is consulted only for axes the CSV
    did not supply, because the dialog only ever asks about axes automatic
    detection could not resolve.

    Returns a summary: ``{'organized': [...], 'missing_channel': [...],
    'failed': [...], 'unscaled': [...], 'csv': name|None}``. ``unscaled`` lists
    source images that ended up with pixel counts for dimensions on at least one
    axis, so the caller can tell the user. With the dialog in place this should
    normally be empty -- a non-empty list means the user declined to supply them.
    """
    config_template_path = preset_details['path']
    fallback_mode = preset_details['default_mode']

    print(f"  Organizing Channel {channel_idx} into: {target_root_dir}")
    os.makedirs(target_root_dir, exist_ok=True)

    with open(config_template_path, 'r') as f:
        template_data = yaml.safe_load(f) or {}
    
    mode = template_data.get('mode', fallback_mode)
    is_2d_mode = mode.endswith('_2d')
    dimension_key = 'pixel_dimensions' if is_2d_mode else 'voxel_dimensions'

    # A metadata CSV next to the raw images overrides what the files claim.
    csv_path = _find_dimension_csv(source_root)
    overrides = load_dimension_overrides(source_root)

    metadata_rows = []
    summary: Dict[str, Any] = {
        'organized': [], 'missing_channel': [], 'failed': [],
        'skipped_sidecars': [], 'unscaled': [],
        'csv': os.path.basename(csv_path) if csv_path else None,
        'channel_idx': channel_idx, 'mode': mode,
        'target': target_root_dir, 'source_root': source_root,
    }

    total_files = len(source_files)
    for file_index, src_file in enumerate(source_files):
        if should_cancel is not None and should_cancel():
            from .slide_reader import SetupCancelled
            raise SetupCancelled("setup cancelled by the user")
        if progress is not None:
            progress(f"Channel {channel_idx}: {src_file}",
                     file_index, total_files)
        src_path = os.path.join(source_root, src_file)

        # Defensive: detect_raw already filters these, but organize_channel_project
        # is public API and a stale caller could still pass sidecars through.
        if is_os_sidecar(src_file):
            summary['skipped_sidecars'].append(src_file)
            continue

        # Ensure file has the requested channel
        try:
            n_ch = MetadataExtractor.get_channel_count(src_path)
        except Exception as exc:
            summary['failed'].append({'file': src_file,
                                      'reason': f'channel count unreadable: {exc}'})
            continue
        if n_ch <= channel_idx:
            summary['missing_channel'].append(src_file)
            continue

        # A slide source is "file::scene"; splitext would mangle it, and all six
        # scenes of one slide would collide on the same folder name.
        try:
            from .slide_reader import folder_name_for_source
            basename = folder_name_for_source(src_file)
        except Exception:
            basename = os.path.splitext(src_file)[0]
        img_subdir = os.path.join(target_root_dir, basename)
        os.makedirs(img_subdir, exist_ok=True)

        target_tif_name = f"{basename}.tif"
        target_tif_path = os.path.join(img_subdir, target_tif_name)

        print(f"    Processing {src_file}...")
        extracted = False
        reason = ''
        try:
            # Tile-level progress: a single 997 megapixel channel can take
            # minutes, so the file-level message above is not granular enough to
            # show that anything is happening.
            def _tile_progress(done, total, _name=src_file, _i=file_index):
                if progress is not None:
                    progress(f"Channel {channel_idx}: {_name}  "
                             f"(tile {done}/{total})", _i, total_files)

            extracted = bool(MetadataExtractor.extract_channel_to_tiff(
                src_path, target_tif_path, channel_idx,
                progress=_tile_progress, should_cancel=should_cancel,
            ))
        except Exception as e:
            from .slide_reader import SetupCancelled
            if isinstance(e, SetupCancelled):
                raise
            reason = str(e)
            print(f"    Error extracting channel {channel_idx} from {src_file}: {e}")

        # Trust the filesystem over the return value: this catches every failure
        # mode, including any the extractor doesn't recognise in itself.
        if not os.path.isfile(target_tif_path) or os.path.getsize(target_tif_path) == 0:
            extracted = False
            reason = reason or 'extraction wrote no image data'

        if not extracted:
            # Leave nothing behind. Writing the config anyway produced folders
            # holding a YAML and no image, which fail the "one tif + one yaml"
            # check and made a whole project look empty for no visible reason.
            try:
                if os.path.isfile(target_tif_path):
                    os.remove(target_tif_path)
                if not os.listdir(img_subdir):
                    os.rmdir(img_subdir)
            except OSError:
                pass
            summary['failed'].append({
                'file': src_file,
                'reason': reason or 'unknown extraction failure',
                'channels_detected': n_ch,
            })
            continue

        # Extract metadata from original source (richer metadata than extracted single channel)
        if MetadataExtractor._slide_source(src_file)[0] is not None:
            meta = MetadataExtractor.read_slide_metadata(src_path)
        elif src_file.lower().endswith('.czi'):
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

        # A metadata CSV is the user's explicit statement of physical size, so it
        # wins over whatever the file claims -- the same precedence the
        # single-channel path gives it. Its columns are already TOTAL microns per
        # axis, so they replace the products above rather than scaling them. One
        # row per source image covers every channel extracted from that image.
        # Per-axis precedence: CSV, then manually-entered values, then the file's
        # own metadata. Tracked per axis rather than per image because a CSV (or
        # dialog) can pin one axis and leave another unknown. The previous
        # `if override: ... elif not _has_real_scale(...)` meant ANY csv row
        # suppressed the unscaled report for the axes it did NOT supply, so a CSV
        # with only 'Width (um)' silently left Y and Z as pixel counts.
        from .dimension_entry import (
            SOURCE_PIXELS_ASSUMED, combine_sources, per_axis_sources,
            scale_gaps, stamp_dimensions_source, unit_scale_axes,
        )

        totals = {'x': total_w, 'y': total_h, 'z': total_d}
        gaps = set(scale_gaps(meta, mode))
        csv_axes, manual_axes = set(), set()

        override = _match_dimension_override(overrides, src_file)
        if override:
            for axis in ('x', 'y', 'z'):
                if override.get(axis) is not None:
                    totals[axis] = override[axis]
                    csv_axes.add(axis)
            if csv_axes:
                print(f"      Dimensions ({', '.join(sorted(csv_axes))}) "
                      "taken from CSV.")

        manual = _match_dimension_override(manual_overrides or {}, src_file)
        if manual:
            for axis in ('x', 'y', 'z'):
                if manual.get(axis) is not None:
                    # Manual wins over BOTH the CSV and the metadata. The dialog
                    # only asks about axes that were missing or whose total was
                    # indistinguishable from the pixel count, so a value coming
                    # back means a human has just entered or confirmed it -- and
                    # the thing it may be correcting is precisely a bad CSV cell.
                    totals[axis] = manual[axis]
                    manual_axes.add(axis)
                    csv_axes.discard(axis)
            if manual_axes:
                print(f"      Dimensions ({', '.join(sorted(manual_axes))}) "
                      "entered or confirmed manually.")

        total_w, total_h, total_d = totals['x'], totals['y'], totals['z']

        axis_sources = per_axis_sources(mode, meta, csv_axes, manual_axes)
        dim_source = combine_sources(axis_sources)

        still_missing = sorted(
            a for a in gaps if a not in csv_axes and a not in manual_axes)

        # Belt and braces: flag any FINAL total that is indistinguishable from
        # its pixel count and was not manually confirmed. The wizard normally
        # prompts for these, but this function is also reachable without it, and
        # a CSV carrying pixel counts in the micron columns passes every other
        # check. Excluded when manually confirmed -- the user has attested that
        # 1 um/pixel is correct for that axis.
        suspect = [
            a for a in unit_scale_axes(
                totals, {'x': width, 'y': height, 'z': z_slices}, mode)
            if a not in manual_axes and a not in still_missing
        ]

        if still_missing or suspect:
            summary['unscaled'].append(src_file)
        if still_missing:
            print(f"      Warning: no usable scale for axis/axes "
                  f"{', '.join(still_missing)} of {src_file}; those dimensions "
                  "are recorded in PIXELS.")
        if suspect:
            print(f"      Warning: {src_file} axis/axes {', '.join(suspect)} "
                  "have a total equal to the pixel count (1 um/pixel). This is "
                  "usually an uncalibrated image or pixel counts entered in a "
                  "microns column.")
            for _axis in suspect:
                axis_sources[_axis] = SOURCE_PIXELS_ASSUMED
            dim_source = combine_sources(axis_sources)

        summary['organized'].append(img_subdir)

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
            # Provenance of the numbers above. Legacy configs lack this key and
            # read back as 'unknown', so nothing existing breaks.
            stamp_dimensions_source(cfg, dim_source)
            with open(new_config_path, 'w') as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        except Exception:
            pass

    if metadata_rows:
        df = pd.DataFrame(metadata_rows)
        df.sort_values('Filename', key=lambda col: col.map(natural_sort_key), inplace=True)
        # Named distinctly from the *input* csv_path above: this is the summary
        # this channel writes, not the CSV dimensions were imported from.
        out_csv_path = os.path.join(target_root_dir, "metadata.csv")
        df.to_csv(out_csv_path, index=False)
        print(f"    Saved metadata summary to {out_csv_path}")

    return summary

def existing_image_folder_names(project_dir: str) -> set:
    """Folder names already organized in a project, single- or multi-channel.

    A multi-channel project keeps one subfolder per image inside each Channel_*
    directory; a single-channel project keeps them directly in the project folder.
    Both are checked so one caller works for either kind.
    """
    from .project_selection import _valid_image_subfolders, _channel_project_dirs

    names = set()
    try:
        channels = _channel_project_dirs(project_dir)
    except Exception:
        channels = []
    roots = channels or [project_dir]
    for root in roots:
        try:
            for sub in _valid_image_subfolders(root):
                names.add(os.path.basename(sub.rstrip("/\\")))
        except Exception:
            pass
    return names


def unorganized_sources(project_dir: str) -> List[str]:
    """Raw source keys sitting in a project that have not been organized yet.

    Setting a project up on a subset leaves the remaining images loose in the
    folder -- organizing does not consume them (multi-channel reads them in place,
    and single-channel only moves the ones it was given). This finds what is left,
    so a project can be extended later rather than re-set up from scratch.

    Returns source keys, so a slide file contributes one entry per scene.
    """
    from .organize_wizard import detect_raw
    from .slide_reader import folder_name_for_source

    try:
        detected = list(detect_raw(project_dir).get("files") or [])
    except Exception:
        return []
    done = existing_image_folder_names(project_dir)
    return [key for key in detected
            if folder_name_for_source(key) not in done]


def preset_from_existing_channel(channel_dir: str) -> Optional[Dict[str, str]]:
    """Reuse an already-organized image's config as a template for new images.

    Adding images to a project must not ask for a preset again: the channel already
    has one, and picking a different config for a late arrival would make it
    incomparable with its siblings. The config of an existing image in the same
    channel IS that channel's config, so it is copied -- dimensions are recomputed
    per image by the organizers, so only the parameter blocks carry over.
    """
    from .project_selection import _valid_image_subfolders

    try:
        folders = _valid_image_subfolders(channel_dir)
    except Exception:
        folders = []
    for folder in folders:
        try:
            contents = sorted(os.listdir(folder))
        except OSError:
            continue
        yml = next((f for f in contents
                    if f.lower().endswith((".yaml", ".yml"))), None)
        if not yml:
            continue
        path = os.path.join(folder, yml)
        try:
            with open(path, "r") as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception:
            continue
        mode = cfg.get("mode")
        if mode and mode not in ("unknown", "error"):
            return {"path": path, "default_mode": mode}
    return None


def add_sources_to_project(
    project_dir: str,
    source_keys: Sequence[str],
    progress=None,
    should_cancel=None,
) -> Dict[str, Any]:
    """Organize additional images into an existing project.

    Reuses each channel's existing config rather than asking again, so images added
    later are processed on the same terms as the ones already there.

    Returns ``{'added': [...], 'channels': n, 'errors': [...], 'summaries': [...]}``.
    """
    from .project_selection import _channel_project_dirs

    keys = [k for k in source_keys if k]
    result: Dict[str, Any] = {"added": [], "channels": 0, "errors": [],
                              "summaries": []}
    if not keys:
        return result

    channels = _channel_project_dirs(project_dir)
    if channels:
        for channel_dir in sorted(channels):
            preset = preset_from_existing_channel(channel_dir)
            if preset is None:
                result["errors"].append(
                    f"{os.path.basename(channel_dir)}: no existing config to "
                    "copy; add an image to this channel from scratch instead.")
                continue
            idx = _channel_index_of(channel_dir)
            if idx is None:
                result["errors"].append(
                    f"{os.path.basename(channel_dir)}: cannot tell which channel "
                    "this folder holds.")
                continue
            try:
                summary = organize_channel_project(
                    list(keys), project_dir, channel_dir, idx, preset,
                    progress=progress, should_cancel=should_cancel)
                result["summaries"].append(summary)
                result["channels"] += 1
            except SetupCancelled:
                raise
            except Exception as exc:
                result["errors"].append(f"{os.path.basename(channel_dir)}: {exc}")
    else:
        preset = preset_from_existing_channel(project_dir)
        if preset is None:
            result["errors"].append(
                "No existing config to copy; set the project up from scratch.")
            return result
        try:
            organize_processing_dir(project_dir, preset, only_files=[
                k for k in keys])
            result["channels"] = 1
        except Exception as exc:
            result["errors"].append(str(exc))

    result["added"] = list(keys)
    return result


def _channel_index_of(channel_dir: str) -> Optional[int]:
    """Channel index parsed from a 'Channel_<n>_<name>' folder."""
    base = os.path.basename(str(channel_dir).rstrip("/\\"))
    parts = base.split("_")
    if len(parts) >= 2 and parts[0].lower() == "channel" and parts[1].isdigit():
        return int(parts[1])
    return None


def _probe_pixel_counts_quiet(path: str) -> Dict[str, int]:
    """Pixel counts per axis from a TIFF, shape only. {} on any failure.

    Used to test whether a recorded total is really a pixel count, so it must
    never raise: a probe failure just means that check is skipped for the image.
    """
    try:
        with tiff.TiffFile(path) as handle:
            shape = handle.series[0].shape
    except Exception:
        return {}
    if len(shape) >= 3:
        return {'z': int(shape[0]), 'y': int(shape[-2]), 'x': int(shape[-1])}
    if len(shape) == 2:
        return {'z': 1, 'y': int(shape[0]), 'x': int(shape[1])}
    return {}


def organize_processing_dir(
    drctry: str,
    preset_details: Dict[str, str],
    only_files: Optional[Sequence[str]] = None,
    manual_overrides: Optional[Dict[str, Dict[str, Optional[float]]]] = None,
) -> Dict[str, Any]:
    """
    Setup Logic for SINGLE-CHANNEL / LEGACY mode.
    Includes Robust Matching for CSV filenames vs Disk filenames.

    `only_files` restricts the run to a subset of the folder's images, which is how
    a project can be set up on one image now and extended later. Files left out
    stay where they are, loose in the folder, and can be organized in a second pass
    -- the per-image subfolders this creates do not disturb them.

    `manual_overrides` carries dimensions the user typed into the dimension-entry
    dialog, shaped like the CSV overrides.

    Returns a summary dict in the same shape the multi-channel path returns
    (``{'organized', 'unscaled', 'csv', ...}``). It previously returned None,
    which is why the wizard's unscaled warning never fired for single-channel
    projects: the wizard only collects summaries it is given. An uncalibrated
    single-channel project was therefore completely silent.
    """
    config_template_path = preset_details['path']
    fallback_mode = preset_details['default_mode']

    print(f"Organizing Standard Project in: {drctry}")

    # Sidecars must go before anything counts images or CSVs: a macOS volume
    # yields a '._' twin for every file, which would otherwise be organized into
    # its own image-less folder and make a single CSV look like two.
    all_files = [f for f in sorted(os.listdir(drctry)) if not is_os_sidecar(f)]
    raw_images = [f for f in all_files if f.lower().endswith(('.tif', '.tiff'))]
    if only_files is not None:
        wanted = {os.path.basename(f) for f in only_files}
        skipped = [f for f in raw_images if f not in wanted]
        raw_images = [f for f in raw_images if f in wanted]
        if skipped:
            print(f"  Organizing {len(raw_images)} of "
                  f"{len(raw_images) + len(skipped)} image(s); the rest are left "
                  "in place and can be added later.")
    csv_files = [f for f in all_files
                 if f.lower().endswith('.csv') and not _is_generated_csv(f)]

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

    from .dimension_entry import (
        SOURCE_CSV, SOURCE_MANUAL, SOURCE_PIXELS_ASSUMED, SOURCE_UNKNOWN,
        combine_sources, per_axis_sources, scale_gaps, stamp_dimensions_source,
        unit_scale_axes,
    )

    generated_rows = []

    # Reported to the caller so the wizard can warn about pixel-count dimensions
    # for single-channel projects too. This path used to return None, so the
    # wizard had nothing to warn from.
    summary: Dict[str, Any] = {
        'organized': [],
        'unscaled': [],
        'csv': csv_files[0] if csv_files else None,
    }
    # filename -> {axis: source}. Only populated on the auto-generate path; a
    # user-supplied CSV is itself the statement of physical size, handled below.
    axis_sources: Dict[str, Dict[str, str]] = {}

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

            # Per-axis scale check, NOT the bare `found` flag. `found` is True for
            # a TIFF carrying XResolution=(1,1)/ResolutionUnit=NONE, whose spacing
            # resolves to exactly 1.0 um/px -- i.e. an uncalibrated image that
            # previously sailed through as calibrated, making its dimensions its
            # pixel counts with no warning anywhere.
            gaps = set(scale_gaps(meta, mode))
            spacing_x = float(meta.get('x', 1.0)) if 'x' not in gaps else 1.0
            spacing_y = float(meta.get('y', 1.0)) if 'y' not in gaps else 1.0
            spacing_z = float(meta.get('z', 1.0)) if 'z' not in gaps else 1.0

            totals = {
                'x': spacing_x * width,
                'y': spacing_y * height,
                'z': spacing_z * z_slices,
            }

            # Dimensions the user typed for the axes automatic detection missed.
            manual = _match_dimension_override(manual_overrides or {}, img_file)
            manual_axes = set()
            if manual:
                for axis in ('x', 'y', 'z'):
                    # Not restricted to `gaps`: the dialog also asks about axes
                    # whose total equals the pixel count, which ARE present in
                    # the metadata. Restricting to gaps would silently discard
                    # exactly those corrections.
                    if manual.get(axis) is not None:
                        totals[axis] = manual[axis]
                        manual_axes.add(axis)
                if manual_axes:
                    print(f"    Dimensions ({', '.join(sorted(manual_axes))}) "
                          "entered or confirmed manually.")

            per_axis = per_axis_sources(mode, meta, (), manual_axes)

            still_missing = sorted(a for a in gaps if a not in manual_axes)

            # Same belt-and-braces check as the multi-channel path: a final
            # total equal to its pixel count means 1 um/pixel, which is far more
            # often an uncalibrated image than a real one.
            suspect = [
                a for a in unit_scale_axes(
                    totals, {'x': width, 'y': height, 'z': z_slices}, mode)
                if a not in manual_axes and a not in still_missing
            ]
            for _axis in suspect:
                per_axis[_axis] = SOURCE_PIXELS_ASSUMED
            axis_sources[img_file] = per_axis

            if still_missing or suspect:
                summary['unscaled'].append(img_file)
            if still_missing:
                print(f"    Warning: no usable scale for axis/axes "
                      f"{', '.join(still_missing)} of {img_file}; those "
                      "dimensions are recorded in PIXELS.")
            if suspect:
                print(f"    Warning: {img_file} axis/axes {', '.join(suspect)} "
                      "have a total equal to the pixel count (1 um/pixel).")

            generated_rows.append({
                'Filename': img_file,
                'Width (um)': totals['x'],
                'Height (um)': totals['y'],
                'Depth (um)': totals['z'],
                'Slices': z_slices,
                'Basename': basename,
            })

        if generated_rows:
            df = pd.DataFrame(generated_rows)
            try:
                df.to_csv(os.path.join(drctry, AUTO_METADATA_CSV), index=False)
                print(f"  Saved '{AUTO_METADATA_CSV}'.")
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

        # Totals for this image, from whichever source supplied them. Rows on the
        # auto-generate path were built above (already manual-aware); rows from a
        # user CSV arrive here untouched, so the manual override and the
        # unit-scale check have to be applied at this point too -- doing it only
        # in the auto-generate branch left a CSV of pixel counts unchallenged.
        row_totals = {
            'x': float(row.get('Width (um)', 1.0) or 1.0),
            'y': float(row.get('Height (um)', 1.0) or 1.0),
            'z': float(row.get('Depth (um)', 0.0) or 0.0),
        }
        tracked = axis_sources.get(matched_file)
        row_manual_axes = set()

        if tracked is None:
            # Came from a user-supplied CSV.
            manual_row = _match_dimension_override(
                manual_overrides or {}, matched_file)
            if manual_row:
                for axis in ('x', 'y', 'z'):
                    if manual_row.get(axis) is not None:
                        row_totals[axis] = manual_row[axis]
                        row_manual_axes.add(axis)
                if row_manual_axes:
                    print(f"  Dimensions ({', '.join(sorted(row_manual_axes))}) "
                          f"of {matched_file} entered or confirmed manually.")

            row_pixels = _probe_pixel_counts_quiet(dst)
            row_suspect = [
                a for a in unit_scale_axes(row_totals, row_pixels, mode)
                if a not in row_manual_axes
            ]
            per_axis_row = {
                a: (SOURCE_MANUAL if a in row_manual_axes else SOURCE_CSV)
                for a in ('x', 'y') + (() if is_2d_mode else ('z',))
            }
            if row_suspect:
                for a in row_suspect:
                    per_axis_row[a] = SOURCE_PIXELS_ASSUMED
                summary['unscaled'].append(matched_file)
                print(f"  Warning: {matched_file} axis/axes "
                      f"{', '.join(row_suspect)} have a total equal to the pixel "
                      "count (1 um/pixel). This usually means pixel counts were "
                      "entered in a microns column.")
            tracked = per_axis_row

        try:
            with open(new_config_path, 'r') as f: cfg = yaml.safe_load(f) or {}
            if dimension_key not in cfg: cfg[dimension_key] = {}
            cfg[dimension_key]['x'] = row_totals['x']
            cfg[dimension_key]['y'] = row_totals['y']
            if not is_2d_mode:
                cfg[dimension_key]['z'] = row_totals['z']
            cfg['mode'] = mode
            cfg['synthetic'] = False  # real image (not procedurally generated)
            # Provenance, per axis, collapsed to one value for the config.
            stamp_dimensions_source(cfg, combine_sources(tracked))
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

    summary['organized'] = organized_dirs
    return summary