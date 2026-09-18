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
    def normalise_recipe(recipe: List[Dict]) -> List[Dict]:
        """Rewrite every relation step into one canonical `relate` form.

        There were three step types doing overlapping work -- `intersect` built
        the mask, `analyze` measured coverage, and `analyze` also measured
        distance -- so asking "how much of A is in B, and keep the mask" needed
        two steps that each recomputed A AND B independently. They are one
        operation with different outputs requested, so they are one step type
        with flags:

            measure_coverage   coverage percentages + the sample summary
            measure_distance   nearest-partner distances
            measure_regions    size and shape of each overlap region
            keep_mask          the overlap becomes the input to later steps

        Legacy recipes are mapped in here rather than in the executor, so
        `recipe.yaml` files saved before the merge keep reproducing exactly and
        only one code path ever runs:

            intersect                 -> keep_mask + measure_regions
            analyze (measure=overlap) -> measure_coverage
            analyze (measure=distance)-> measure_distance
            analyze (no measure key)  -> both, which is what it used to do
        """
        out: List[Dict] = []
        for step in recipe:
            stype = step.get('type')

            if stype == 'intersect':
                inputs = step.get('inputs') or []
                new = dict(step)
                new.update(
                    type='relate',
                    primary=inputs[0] if inputs else step.get('primary'),
                    target=inputs[1] if len(inputs) > 1 else step.get('target'),
                    measure_coverage=False, measure_distance=False,
                    measure_regions=True, keep_mask=True,
                )
                out.append(new)

            elif stype == 'analyze':
                measure = str(step.get('measure', 'both')).lower()
                new = dict(step)
                new.update(
                    type='relate',
                    measure_coverage=measure in ('overlap', 'both'),
                    measure_distance=measure in ('distance', 'both'),
                    measure_regions=False, keep_mask=False,
                )
                out.append(new)

            else:
                out.append(dict(step))
        return out

    @staticmethod
    def channel_name_registry(channel_keys):
        """Short, unique, column-safe name per channel key.

        These names end up in CSV headers (`pct_of_<partner>_occupied_by_
        <primary>`), in derived mask filenames and in viewer layer names, so
        they have to identify the channel and they have to differ from one
        another.

        The previous rule was `ch_key.split('_', 2)[-1]` -- the last chunk of
        the folder name. That gives "Microglia" for `Channel_0_Microglia`, but
        for any other layout it takes whatever the folder happens to end with.
        On a project whose channel folders end in the config name, EVERY
        channel resolved to the same word, producing headers like
        `total_vol_of_default_inside_this_default` where primary and partner
        were different channels with identical names.

        So: take the marker suffix only when the folder really follows the
        `Channel_<n>_<marker>` convention, fall back to the whole folder name
        otherwise, and if two channels still collide, use the full key for the
        colliding ones. A long header beats an ambiguous one.
        """
        keys = list(channel_keys)

        def short(key):
            parts = str(key).split('_')
            if len(parts) >= 3 and parts[0].lower() == 'channel':
                return '_'.join(parts[2:])
            return str(key)

        names = {k: short(k) for k in keys}
        by_name = {}
        for k, v in names.items():
            by_name.setdefault(v, []).append(k)
        for v, ks in by_name.items():
            if len(ks) > 1:
                for k in ks:
                    names[k] = str(k)

        def safe(text):
            out = "".join(c if (c.isalnum() or c in "-_") else "_"
                          for c in str(text)).strip("_")
            return out or "channel"

        return {k: safe(v) for k, v in names.items()}

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
        name_registry = RelationalEngine.channel_name_registry(
            sample_channels.keys())

        last_mask_path = None
        last_mask_name = "Original" 
        results_to_viz = []
        final_metrics_df = None
        # Per-object relational tables, keyed by their primary's ID column.
        #
        # There used to be a single accumulating frame, which assumed every
        # analyze step in a recipe shared the same primary. Nothing enforced
        # that -- checking two channels produces one step each and the primary
        # is chosen per step -- and a second step with a different primary
        # merged on an ID column that did not exist in the first step's frame,
        # raising KeyError mid-batch. Keying by primary means the two tables
        # stay separate (and both get written) instead of colliding.
        final_tables: Dict[str, pd.DataFrame] = {}
        summary_rows = []
        parent_id_map = {} 
        is_2d = (len(shape) == 2)

        os.makedirs(out_dir, exist_ok=True)

        recipe = RelationalEngine.normalise_recipe(recipe)

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
                    next_step.get('type') == 'relate' and
                    next_step.get('primary') == target_ch
                )
                if not is_analyze_role_selector:
                    last_mask_path = ch_path
                    last_mask_name = ch_name

                if ch_path:
                    results_to_viz.append({"name": ch_name, "path": ch_path})

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

            elif step_type == "relate":
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
                    # Flags set by normalise_recipe, so legacy and current
                    # recipes arrive here in the same shape.
                    want_overlap = bool(step.get('measure_coverage', False))
                    want_distance = bool(step.get('measure_distance', False))
                    want_regions = bool(step.get('measure_regions', False))
                    want_keep = bool(step.get('keep_mask', False))
                    # Opt-in per step; the full cross-product is the most
                    # expensive thing in the module and nothing downstream
                    # reads its CSV.
                    want_pairwise = bool(step.get('pairwise', False))

                    _asked = [n for n, f in (
                        ("coverage", want_overlap), ("distance", want_distance),
                        ("regions", want_regions), ("mask", want_keep),
                    ) if f]
                    print(f"  [Relate] {active_mask_name} vs {partner_bio_name}"
                          f" ({', '.join(_asked) or 'nothing requested'}"
                          f"{', pairwise' if want_pairwise else ''})")

                    inter_path = None
                    primary_df = pd.DataFrame()
                    partner_df = pd.DataFrame()
                    summary = {}

                    if want_overlap or want_distance:
                        if is_2d:
                            sp_2d = spacing if len(spacing) == 2 else (spacing[1], spacing[2])
                            primary_df, partner_df, inter_path, summary = calculate_interaction_metrics_2d(
                                active_mask_path, partner_dat_path, out_dir, shape, sp_2d,
                                active_mask_name, partner_bio_name,
                                calculate_distance=want_distance,
                                calculate_overlap=want_overlap,
                                calculate_pairwise=want_pairwise,
                            )
                        else:
                            primary_df, partner_df, inter_path, summary = calculate_interaction_metrics(
                                active_mask_path, partner_dat_path, out_dir, shape, spacing,
                                active_mask_name, partner_bio_name,
                                calculate_distance=want_distance,
                                calculate_overlap=want_overlap,
                                calculate_pairwise=want_pairwise,
                            )

                    if summary:
                        summary_rows.append({'sample_name': sample_name, **summary})

                    # ---- The overlap mask, if this step was asked for one ----
                    if want_keep or want_regions:
                        label_mode = step.get('label_mode', 'connected')
                        preserve_ids = bool(step.get('preserve_ids', False))
                        mask_path, mapping = None, {}

                        if (inter_path and label_mode == 'connected'
                                and not preserve_ids):
                            # The coverage pass already wrote a uniquely
                            # labelled intersection, so reuse it rather than
                            # computing A AND B a second time. Only valid for
                            # the default labelling; the parent-ID modes need
                            # intersect_masks, which is what knows about them.
                            mask_path = inter_path
                            _m = np.memmap(mask_path, dtype=np.int32, mode='r', shape=shape)
                            _u = np.unique(_m)
                            mapping = {int(u): int(u) for u in _u[_u > 0]}
                            del _m
                        else:
                            mask_path, ids_preserved = RelationalEngine.intersect_masks(
                                active_mask_path, partner_dat_path, step_out_path,
                                shape, label_mode, len(shape), preserve_ids
                            )
                            if ids_preserved:
                                _m = np.memmap(mask_path, dtype=np.int32, mode='r', shape=shape)
                                _u = np.unique(_m)
                                mapping = {int(u): int(u) for u in _u[_u > 0]}
                                del _m
                            else:
                                _m = np.memmap(mask_path, dtype=np.int32, mode='r+', shape=shape)
                                _new, mapping = RelationalEngine.relabel_sequentially(_m)
                                _m[:] = _new[:]
                                _m.flush()
                                del _m

                        overlap_name = f"{active_mask_name}_in_{partner_bio_name}"

                        # When the coverage pass wrote its own intersection and
                        # this step then built a differently-labelled mask, the
                        # first one is a superseded intermediate. Leaving it on
                        # disk put TWO derived layers in the viewer for one
                        # step -- `intersection_<partner>` and
                        # `step_<i>_relate` -- differing only in labelling, with
                        # nothing to say which was the step's actual result.
                        if inter_path and os.path.abspath(inter_path) != os.path.abspath(mask_path):
                            try:
                                os.remove(inter_path)
                            except OSError:
                                pass
                            inter_path = None

                        results_to_viz.append({"name": overlap_name, "path": mask_path})

                        if want_regions:
                            RelationalEngine._save_intersection_metrics_via_pipeline(
                                mask_path, shape, spacing, overlap_name, mapping,
                                out_dir, sample_name, is_2d
                            )

                        if want_keep:
                            # Only now does the overlap become what later steps
                            # act on. A step that merely measured coverage must
                            # leave the chain alone.
                            last_mask_path = mask_path
                            last_mask_name = overlap_name
                            parent_id_map = mapping

                    elif inter_path:
                        results_to_viz.append(
                            {"name": f"Overlap ({partner_bio_name})", "path": inter_path})

                    if primary_df.empty:
                        if not partner_df.empty:
                            partner_df.to_csv(os.path.join(
                                out_dir, f"coverage_stats_{partner_bio_name}.csv"), index=False)
                        continue
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

                    if id_col not in final_tables:
                        table = primary_df.copy().rename(columns={'label': id_col})
                        # Insert parent ID mapping for biological traceability
                        if parent_id_map:
                            map_df = pd.DataFrame(list(parent_id_map.items()),
                                                 columns=[id_col, f"parent_id_{active_mask_name}"])
                            table = pd.merge(map_df, table, on=id_col)
                        final_tables[id_col] = table
                    else:
                        # Join additional partners (e.g. Neurons AND Microglia) to the same
                        # table. Safe now that the lookup is keyed by primary, so this only
                        # ever merges frames that genuinely share this primary's IDs.
                        final_tables[id_col] = pd.merge(
                            final_tables[id_col], primary_df,
                            left_on=id_col, right_on='label',
                            how='outer').drop(columns=['label'])

                    # Save the Coverage Summary (Partner-view)
                    if not partner_df.empty:
                        partner_df.to_csv(os.path.join(out_dir, f"coverage_stats_{partner_bio_name}.csv"), index=False)

        # 5. Final Result Persistence
        #
        # One primary is the overwhelmingly common case and keeps the historic
        # filename. Several primaries in one recipe each get their own file,
        # named for the primary, rather than being forced into one table on
        # mismatched IDs.
        if final_tables:
            if len(final_tables) == 1:
                final_metrics_df = next(iter(final_tables.values()))
                final_metrics_df.to_csv(
                    os.path.join(out_dir, f"{sample_name}_relational_metrics.csv"),
                    index=False)
            else:
                print(f"  [Note] This recipe measured {len(final_tables)} different "
                      f"primaries. Their rows describe different objects, so they "
                      f"cannot share a table -- writing one file each.")
                for id_col, table in final_tables.items():
                    primary_label = id_col[3:] if id_col.startswith("id_") else id_col
                    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                                   for ch in str(primary_label))
                    table.to_csv(
                        os.path.join(
                            out_dir,
                            f"{sample_name}_relational_metrics_{safe}.csv"),
                        index=False)
                # Returned for the viewer's proximity bridges, which can only
                # draw one primary's lines.
                final_metrics_df = next(iter(final_tables.values()))

        # 6. Sample-level Overlap Summary
        #
        # The per-object table answers "how much of THIS object is inside a
        # partner". This answers "how much of the channel is", which is a ratio
        # of totals and cannot be recovered by averaging those rows.
        if summary_rows:
            summary_path = os.path.join(out_dir, "overlap_summary.csv")
            pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
            for row in summary_rows:
                print(f"  [Overlap] {row['pct_of_primary_inside_partner']:.2f}% of "
                      f"{row['primary']} lies inside {row['partner']}; "
                      f"{row['pct_of_partner_inside_primary']:.2f}% the other way.")

        return results_to_viz, final_metrics_df