import os
import numpy as np
import pandas as pd
import dask.array as da
import dask_image.ndmeasure
from scipy import ndimage
from typing import List, Dict, Tuple, Any, Optional
import shutil

# Both entry points now live in one merged module. They are still two
# functions, and the `is_2d` dispatch below is kept: it is driven by the
# image's own rank (`len(shape) == 2`), never by the mode string, so it stays
# correct with a single mode.
from ..fluorescence_module.interaction_analysis import (
    calculate_interaction_metrics, calculate_interaction_metrics_2d)

class RelationalEngine:
    """
    Core logic for performing multi-channel relational algebra.
    Handles mask intersections, volume filtering, and proximity analysis.
    """

    @staticmethod
    def _find_dat(folder_path, include_roi: bool = False, roi_name=None):
        """Helper to find the final_segmentation.dat in a project folder.

        Skips ROI sub-region sessions by default. An ROI session lives in
        ``<basename>_processed_<mode>_roi``, which also contains the substring
        "_processed_", so the original scan matched it as readily as the
        full-image directory -- and returned whichever ``os.listdir`` happened to
        yield first. That made cross-channel analysis silently read the CROP's
        segmentation for a channel that had an ROI: a different array with a
        different shape from the full image the caller is memmapping it against.
        Which one won depended on filesystem ordering, so it reproduced on some
        machines and not others.

        Full-image directories are preferred explicitly rather than by luck, and
        results are sorted so the choice is deterministic when several exist.
        """
        if not folder_path or not os.path.isdir(folder_path):
            return None

        # A named region's results live directly in its session directory, which
        # IS the processed dir -- so it is searched directly rather than scanned
        # for a "_processed_" child. Every channel's copy of a region shares one
        # polygon and bounding box, which is what makes the masks comparable
        # across channels at all.
        if roi_name:
            try:
                from .roi_sharing import roi_session_dir
                roi_dir = roi_session_dir(folder_path, roi_name)
            except Exception:
                return None
            if not roi_dir or not os.path.isdir(roi_dir):
                return None
            try:
                for f in sorted(os.listdir(roi_dir)):
                    if f.startswith("final_segmentation") and f.endswith(".dat"):
                        return os.path.join(roi_dir, f)
            except OSError:
                pass
            return None

        try:
            entries = sorted(os.listdir(folder_path))
        except OSError:
            return None

        full_image, roi = [], []
        for d in entries:
            if "_processed_" not in d:
                continue
            if not os.path.isdir(os.path.join(folder_path, d)):
                continue
            (roi if RelationalEngine._is_roi_dir(d) else full_image).append(d)

        # Full-image dirs first; ROI sessions only when explicitly requested.
        for d in full_image + (roi if include_roi else []):
            proc_dir = os.path.join(folder_path, d)
            try:
                names = sorted(os.listdir(proc_dir))
            except OSError:
                continue
            for f in names:
                if f.startswith("final_segmentation") and f.endswith(".dat"):
                    return os.path.join(proc_dir, f)
        return None

    @staticmethod
    def _is_roi_dir(name: str) -> bool:
        """True for an ROI sub-region session directory.

        Matches the bare ``_roi`` suffix and any ``_roi_<label>`` variant, so
        named or numbered ROI sessions are excluded on the same rule.
        """
        base = os.path.basename(str(name).rstrip("/\\"))
        return base.endswith("_roi") or "_roi_" in base

    @staticmethod
    def relabel_sequentially(mask):
        """
        Remaps arbitrary or gapped IDs to sequential 1...N.
        Returns the relabeled mask and a mapping dictionary {new_id: old_id}.
        """
        unique_ids = np.unique(mask)
        unique_ids = unique_ids[unique_ids > 0]
        if len(unique_ids) == 0:
            return mask, {}
        
        # Create a mapping for the CSV traceability (New -> Old)
        mapping = {new_id: int(old_id) for new_id, old_id in enumerate(unique_ids, 1)}
        
        # Fast remapping using a lookup table
        lookup = np.zeros(int(unique_ids.max() + 1), dtype=np.int32)
        lookup[unique_ids] = np.arange(1, len(unique_ids) + 1)
        
        return lookup[mask], mapping
    
    @staticmethod
    def intersect_masks(path_a, path_b, out_path, shape, label_mode='binary', ndim=3, preserve_ids=False):
        """Boolean AND between two label files with configurable identity inheritance.
        
        Args:
            preserve_ids: When True and label_mode is 'parent_a' or 'parent_b', the
                          inherited label IDs are written as-is without any sequential
                          relabeling. This lets downstream steps trace result objects
                          back to their source masks by the original ID.
        """
        ma = np.memmap(path_a, dtype=np.int32, mode='r', shape=shape)
        mb = np.memmap(path_b, dtype=np.int32, mode='r', shape=shape)
        
        overlap_mask = (ma > 0) & (mb > 0)
        out = np.memmap(out_path, dtype=np.int32, mode='w+', shape=shape)
        
        if label_mode == 'binary':
            out[overlap_mask] = 1
        elif label_mode == 'parent_a':
            out[:] = np.where(overlap_mask, ma, 0)
        elif label_mode == 'parent_b':
            out[:] = np.where(overlap_mask, mb, 0)
        elif label_mode == 'connected':
            d_mask = da.from_array(overlap_mask, chunks=(64, 256, 256) if ndim==3 else (4096, 4096))
            labeled, _ = dask_image.ndmeasure.label(d_mask)
            da.store(labeled.astype(np.int32), out, lock=True)

        out.flush()
        del ma, mb, overlap_mask
        return out_path, preserve_ids and label_mode in ('parent_a', 'parent_b')

    @staticmethod
    def filter_by_volume(path_in, out_path, shape, spacing, min_vol_um3):
        """Keep objects at or above a physical size threshold.

        The parameter is named for volume because the signature predates 2D
        support, but the quantity is voxel count x voxel size: an AREA in 2D and a
        VOLUME in 3D. Callers should present it with the unit that matches the
        project's mode.
        """
        """Removes objects smaller than a physical volume threshold and relabels 1..N."""
        data = np.memmap(path_in, dtype=np.int32, mode='r', shape=shape)
        unit_vol = np.prod(spacing)
        
        # Ensure we start from binary to group fragments correctly
        d_data = da.from_array(data > 0, chunks=(64, 256, 256) if len(shape)==3 else (4096, 4096))
        labeled, _ = dask_image.ndmeasure.label(d_data)
        labeled_comp = labeled.compute().astype(np.int32)
        
        objs = ndimage.find_objects(labeled_comp)
        out = np.memmap(out_path, dtype=np.int32, mode='w+', shape=shape)
        out[:] = 0

        valid_count = 0
        for i, sl in enumerate(objs):
            if sl is None: continue
            lbl = i + 1
            mask = (labeled_comp[sl] == lbl)
            vol = np.count_nonzero(mask) * unit_vol
            if vol >= min_vol_um3:
                valid_count += 1
                out[sl][mask] = valid_count
        
        out.flush()
        del data, labeled_comp
        return out_path
    
    @staticmethod
    def _save_intersection_metrics_via_pipeline(
        mask_path, shape, spacing, mask_name, id_mapping, out_dir, sample_name, is_2d
    ):
        if is_2d:
            from ..fluorescence_module.calculate_features import (
                analyze_segmentation_2d)
        else:
            from ..fluorescence_module.calculate_features import (
                analyze_segmentation)

        mask = np.memmap(mask_path, dtype=np.int32, mode='r', shape=shape)

        if is_2d:
            sp = spacing if len(spacing) == 2 else (spacing[1], spacing[2])
            metrics_df, _ = analyze_segmentation_2d(
                mask,
                intensity_image=None,
                spacing_yx=sp,
                calculate_distances=False,   # Not needed for synthetic filtering
                calculate_skeletons=False,   # Expensive and unused downstream
            )
        else:
            metrics_df, _ = analyze_segmentation(
                mask,
                intensity_image=None,
                spacing=spacing,             # 3D takes (Z, Y, X) directly
                calculate_distances=False,
                calculate_skeletons=False,
            )

        del mask

        if metrics_df.empty:
            print(f"  [Intersect Metrics] No objects in {mask_name}, skipping CSV.")
            return

        # Attach parent ID mapping for traceability (same convention as analyze step)
        map_df = pd.DataFrame(
            list(id_mapping.items()),
            columns=['label', f'parent_id_{mask_name}']
        )
        metrics_df['label'] = metrics_df['label'].astype(int)
        metrics_df = pd.merge(map_df, metrics_df, on='label', how='right')

        # Step-scoped filename. This used to write the SAME path as run_recipe's
        # final table below, so in an intersect -> analyze recipe the analyze
        # result silently overwrote the intersection metrics, and with two
        # intersect steps the second overwrote the first. The overlap objects are
        # what the spatial null randomises, so losing them broke that path.
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                       for ch in str(mask_name))
        csv_path = os.path.join(out_dir, f"{sample_name}_{safe}_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        print(f"  [Intersect Metrics] Saved {len(metrics_df)} objects → {csv_path}")
    
    @staticmethod
    def run_recipe(sample_name, registry, recipe, out_dir, shape, spacing,
                   roi_name=None):
        """
        Executes a sequence of relational steps.
        Saves metrics, coverage stats, and connection coordinates.

        `roi_name` restricts the analysis to one saved region. Every channel holds
        that region under the same name with the same polygon, so the per-channel
        masks are the same crop and line up voxel-for-voxel -- the analysis itself
        needs no changes, only a different set of .dat files. `shape` and `spacing`
        must then describe the CROP, not the full image.
        """
        _dat = lambda folder: RelationalEngine._find_dat(folder, roi_name=roi_name)
        sample_channels = registry.get(sample_name, {})
        
        # 1. Biological Name Mapping
        name_registry = {}
        for ch_key in sample_channels.keys():
            # Update this line to use split('_', 2) to prevent naming collisions
            parts = ch_key.split('_', 2) 
            variety = parts[-1] if len(parts) > 2 else ch_key
            name_registry[ch_key] = variety

        last_mask_path = None
        last_mask_name = "Original" 
        results_to_viz = []
        final_metrics_df = None
        parent_id_map = {} 
        is_2d = (len(shape) == 2)

        os.makedirs(out_dir, exist_ok=True)

        for i, step in enumerate(recipe):
            step_type = step['type']
            step_out_path = os.path.join(out_dir, f"step_{i}_{step_type}.dat")

            if step_type == "primary":
                target_ch = step['target']
                ch_path = _dat(sample_channels.get(target_ch))
                ch_name = name_registry.get(target_ch, "Primary")

                # Guard: when a 'primary' step immediately precedes an 'analyze' step that
                # names the same channel as step['primary'], the UI is declaring WHICH channel
                # is the primary object for that analysis — it is NOT introducing a new pipeline
                # result.  We must NOT overwrite last_mask_path here, because last_mask_path
                # still holds the accumulated intermediate (e.g. the B∩C intersection mask)
                # that the analyze step needs as the *partner*.  Clobbering it causes the
                # analyze step to compare A against A, giving trivially-zero distances.
                next_step = recipe[i + 1] if i + 1 < len(recipe) else {}
                is_analyze_role_selector = (
                    next_step.get('type') == 'analyze' and
                    next_step.get('primary') == target_ch
                )
                if not is_analyze_role_selector:
                    last_mask_path = ch_path
                    last_mask_name = ch_name

                if ch_path:
                    results_to_viz.append({"name": ch_name, "path": ch_path})

            elif step_type == "intersect":
                inputs = step['inputs']
                path_a = _dat(sample_channels.get(inputs[0]))
                name_a = name_registry.get(inputs[0], "A")

                if inputs[1] == "PREVIOUS_RESULT":
                    path_b = last_mask_path
                    name_b = last_mask_name
                else:
                    path_b = _dat(sample_channels.get(inputs[1]))
                    name_b = name_registry.get(inputs[1], "B")

                if path_a and path_b:
                    label_mode = step.get('label_mode', 'binary')
                    preserve_ids = step.get('preserve_ids', False)
                    last_mask_path, ids_preserved = RelationalEngine.intersect_masks(
                        path_a, path_b, step_out_path, shape, label_mode, len(shape), preserve_ids
                    )
                    last_mask_name = f"{name_a}_in_{name_b}"
                    
                    # Relabel to sequential IDs unless the caller explicitly asked to keep
                    # the original parent IDs for downstream traceability.
                    if ids_preserved:
                        # Build an identity mapping so metrics CSV still gets a parent_id column
                        temp_mask = np.memmap(last_mask_path, dtype=np.int32, mode='r', shape=shape)
                        unique_ids = np.unique(temp_mask)
                        unique_ids = unique_ids[unique_ids > 0]
                        mapping = {int(uid): int(uid) for uid in unique_ids}
                        del temp_mask
                    else:
                        temp_mask = np.memmap(last_mask_path, dtype=np.int32, mode='r+', shape=shape)
                        new_mask, mapping = RelationalEngine.relabel_sequentially(temp_mask)
                        temp_mask[:] = new_mask[:]
                        temp_mask.flush()
                        del temp_mask
                    
                    parent_id_map = mapping
                    results_to_viz.append({"name": last_mask_name, "path": last_mask_path})
                    
                    # ── Reuse existing feature pipeline to generate metrics CSV ──
                    RelationalEngine._save_intersection_metrics_via_pipeline(
                        last_mask_path, shape, spacing, last_mask_name, mapping, out_dir, sample_name, is_2d
                    )

            elif step_type == "filter":
                # A filter with nothing before it applies to the channel recorded
                # on the step. Without this the step silently did nothing, which
                # looked like the filter had been applied when it had not.
                if not last_mask_path and step.get("input"):
                    src = _dat(sample_channels.get(step["input"]))
                    if src:
                        last_mask_path = src
                        last_mask_name = name_registry.get(step["input"],
                                                           step["input"])
                    else:
                        print(f"  [Size Filter] SKIPPED: no segmentation for "
                              f"{step['input']}")
                if not last_mask_path:
                    print("  [Size Filter] SKIPPED: nothing to filter. Add an "
                          "intersection first, or re-add the filter so it records "
                          "which channel it applies to.")
                if last_mask_path:
                    min_v = step['min_vol']
                    # Objects are areas in 2D and volumes in 3D; the threshold is
                    # the same number either way, only the unit differs.
                    _unit = "um\u00b2" if len(shape) == 2 else "um\u00b3"
                    last_mask_path = RelationalEngine.filter_by_volume(
                        last_mask_path, step_out_path, shape, spacing, min_v
                    )
                    last_mask_name = f"{last_mask_name}_Filtered"
                    print(f"  [Size Filter] Kept objects > {min_v:g} {_unit}")
                    
                    # Relabel after volume removal
                    temp_mask = np.memmap(last_mask_path, dtype=np.int32, mode='r+', shape=shape)
                    new_mask, mapping = RelationalEngine.relabel_sequentially(temp_mask)
                    temp_mask[:] = new_mask[:]
                    temp_mask.flush()
                    del temp_mask
                    
                    parent_id_map = mapping
                    results_to_viz.append({"name": last_mask_name, "path": last_mask_path})

            elif step_type == "analyze":
                # Resolve primary and partner paths/names.
                #
                # Three cases, all handled by whether 'primary' is set and what 'target' holds:
                #
                # Case 1 – no 'primary' key:
                #   last_mask_path is primary; step['target'] (a channel key) is the partner.
                #
                # Case 2 – 'primary' set, target == "PREVIOUS_RESULT":
                #   The named primary channel is looked up from the registry.
                #   last_mask_path (e.g. a B∩C intersection) is the partner.
                #
                # Case 3 – 'primary' set, target is a real channel key:
                #   Simple two-channel analysis. Both sides looked up from the registry.
                #   last_mask_path is NOT used, so it is left untouched.
                parent_id_map = {}
                if step.get('primary'):
                    primary_ch_key   = step['primary']
                    active_mask_path = _dat(sample_channels.get(primary_ch_key))
                    active_mask_name = name_registry.get(primary_ch_key, primary_ch_key)

                    if step.get('target') == "PREVIOUS_RESULT":
                        # Case 2: partner is the accumulated intermediate
                        partner_dat_path = last_mask_path
                        partner_bio_name = last_mask_name
                    else:
                        # Case 3: partner is a named channel — direct two-channel analysis
                        partner_ch_key   = step['target']
                        partner_dat_path = _dat(sample_channels.get(partner_ch_key))
                        partner_bio_name = name_registry.get(partner_ch_key, partner_ch_key)
                else:
                    # Case 1: default — previous result is primary, target channel is partner
                    active_mask_path = last_mask_path
                    active_mask_name = last_mask_name
                    partner_bio_name = name_registry.get(step['target'], "Partner")
                    partner_dat_path = _dat(sample_channels.get(step['target']))
                if active_mask_path and partner_dat_path:
                    # Execute proximity and overlap logic
                    if is_2d:
                        sp_2d = spacing if len(spacing)==2 else (spacing[1], spacing[2])
                        primary_df, partner_df, inter_path = calculate_interaction_metrics_2d(
                            active_mask_path, partner_dat_path, out_dir, shape, sp_2d,
                            active_mask_name, partner_bio_name
                        )
                        print(f"  [DEBUG] primary_df shape: {primary_df.shape}")
                        print(f"  [DEBUG] primary_df columns: {primary_df.columns.tolist()}")
                        print(f"  [DEBUG] primary_df head:\n{primary_df.head()}")
                        print(f"  [DEBUG] partner_df shape: {partner_df.shape}")
                        print(f"  [DEBUG] active_mask_path: {active_mask_path}")
                        print(f"  [DEBUG] partner_dat_path: {partner_dat_path}")
                    else:
                        primary_df, partner_df, inter_path = calculate_interaction_metrics(
                            active_mask_path, partner_dat_path, out_dir, shape, spacing,
                            active_mask_name, partner_bio_name
                        )

                    # CRITICAL: Append the intersection mask to the viewer list
                    if inter_path:
                        results_to_viz.append({"name": f"Overlap ({partner_bio_name})", "path": inter_path})

                    # Rename the ID column for the final merge
                    id_col = f"id_{active_mask_name}"

                    # Bug fix: calculate_interaction_metrics can return primary_df with the
                    # label column already renamed in some code paths — normalise before merging.
                    if 'label' not in primary_df.columns:
                        label_candidates = [c for c in primary_df.columns
                                            if c.lower() == 'label' or c.startswith('id_')]
                        if label_candidates:
                            primary_df = primary_df.rename(columns={label_candidates[0]: 'label'})
                        else:
                            print(f"  [Warning] primary_df missing 'label' column for partner "
                                  f"{partner_bio_name}; skipping merge. Columns: {list(primary_df.columns)}")
                            if not partner_df.empty:
                                partner_df.to_csv(
                                    os.path.join(out_dir, f"coverage_stats_{partner_bio_name}.csv"),
                                    index=False)
                            continue

                    if final_metrics_df is None:
                        final_metrics_df = primary_df.copy().rename(columns={'label': id_col})
                        # Insert parent ID mapping for biological traceability
                        if parent_id_map:
                            map_df = pd.DataFrame(list(parent_id_map.items()),
                                                 columns=[id_col, f"parent_id_{active_mask_name}"])
                            final_metrics_df = pd.merge(map_df, final_metrics_df, on=id_col)
                    else:
                        # Join additional partners (e.g. Neurons AND Microglia) to the same table
                        final_metrics_df = pd.merge(final_metrics_df, primary_df,
                                                   left_on=id_col, right_on='label',
                                                   how='outer').drop(columns=['label'])

                    # Save the Coverage Summary (Partner-view)
                    if not partner_df.empty:
                        partner_df.to_csv(os.path.join(out_dir, f"coverage_stats_{partner_bio_name}.csv"), index=False)

        # 5. Final Result Persistence
        if final_metrics_df is not None:
            csv_path = os.path.join(out_dir, f"{sample_name}_relational_metrics.csv")
            final_metrics_df.to_csv(csv_path, index=False)

        return results_to_viz, final_metrics_df