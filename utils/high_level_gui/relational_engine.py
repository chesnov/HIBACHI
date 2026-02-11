import os
import numpy as np
import pandas as pd
import dask.array as da
import dask_image.ndmeasure
from scipy import ndimage
from typing import List, Dict, Tuple, Any, Optional
import shutil

# Correct relative imports
from ..module_3d.interaction_analysis import calculate_interaction_metrics
from ..module_2d.interaction_analysis_2d import calculate_interaction_metrics_2d

class RelationalEngine:
    """
    Core logic for performing multi-channel relational algebra.
    Handles mask intersections, volume filtering, and proximity analysis.
    """

    @staticmethod
    def _find_dat(folder_path):
        """Helper to find the final_segmentation.dat in a project folder."""
        if not folder_path or not os.path.isdir(folder_path): 
            return None
        
        # Look for the standard processed folder suffix
        for d in os.listdir(folder_path):
            if "_processed_" in d:
                proc_dir = os.path.join(folder_path, d)
                if not os.path.isdir(proc_dir): 
                    continue
                for f in os.listdir(proc_dir):
                    if f.startswith("final_segmentation") and f.endswith(".dat"):
                        return os.path.join(proc_dir, f)
        return None

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
    def intersect_masks(path_a, path_b, out_path, shape, label_mode='binary', ndim=3):
        """Boolean AND between two label files with configurable identity inheritance."""
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
        return out_path

    @staticmethod
    def filter_by_volume(path_in, out_path, shape, spacing, min_vol_um3):
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
    def run_recipe(sample_name, registry, recipe, out_dir, shape, spacing):
        """
        Executes a sequence of relational steps.
        Saves metrics, coverage stats, and connection coordinates.
        """
        sample_channels = registry.get(sample_name, {})
        
        # 1. Biological Name Mapping
        name_registry = {}
        for ch_key in sample_channels.keys():
            variety = ch_key.split('_')[-1] # "Channel_0_Microglia" -> "Microglia"
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

            if step_type == "intersect":
                inputs = step['inputs']
                path_a = RelationalEngine._find_dat(sample_channels.get(inputs[0]))
                name_a = name_registry.get(inputs[0], "A")

                if inputs[1] == "PREVIOUS_RESULT":
                    path_b = last_mask_path
                    name_b = last_mask_name
                else:
                    path_b = RelationalEngine._find_dat(sample_channels.get(inputs[1]))
                    name_b = name_registry.get(inputs[1], "B")

                if path_a and path_b:
                    label_mode = step.get('label_mode', 'binary')
                    last_mask_path = RelationalEngine.intersect_masks(
                        path_a, path_b, step_out_path, shape, label_mode, len(shape)
                    )
                    last_mask_name = f"{name_a}_in_{name_b}"
                    
                    # Relabel to ensure sequential IDs (Fixes numbering gaps)
                    temp_mask = np.memmap(last_mask_path, dtype=np.int32, mode='r+', shape=shape)
                    new_mask, mapping = RelationalEngine.relabel_sequentially(temp_mask)
                    temp_mask[:] = new_mask[:]
                    temp_mask.flush()
                    del temp_mask 
                    
                    parent_id_map = mapping 
                    results_to_viz.append({"name": last_mask_name, "path": last_mask_path})

            elif step_type == "filter":
                if last_mask_path:
                    min_v = step['min_vol']
                    last_mask_path = RelationalEngine.filter_by_volume(
                        last_mask_path, step_out_path, shape, spacing, min_v
                    )
                    last_mask_name = f"{last_mask_name}_Filtered"
                    
                    # Relabel after volume removal
                    temp_mask = np.memmap(last_mask_path, dtype=np.int32, mode='r+', shape=shape)
                    new_mask, mapping = RelationalEngine.relabel_sequentially(temp_mask)
                    temp_mask[:] = new_mask[:]
                    temp_mask.flush()
                    del temp_mask
                    
                    parent_id_map = mapping
                    results_to_viz.append({"name": last_mask_name, "path": last_mask_path})

            elif step_type == "analyze":
                partner_bio_name = name_registry.get(step['target'], "Partner")
                partner_dat_path = RelationalEngine._find_dat(sample_channels.get(step['target']))
                
                if last_mask_path and partner_dat_path:
                    # Execute proximity and overlap logic
                    if is_2d:
                        sp_2d = spacing if len(spacing)==2 else (spacing[1], spacing[2])
                        primary_df, partner_df, inter_path = calculate_interaction_metrics_2d(
                            last_mask_path, partner_dat_path, out_dir, shape, sp_2d, 
                            last_mask_name, partner_bio_name
                        )
                    else:
                        primary_df, partner_df, inter_path = calculate_interaction_metrics(
                            last_mask_path, partner_dat_path, out_dir, shape, spacing, 
                            last_mask_name, partner_bio_name
                        )
                    
                    # CRITICAL: Append the intersection mask to the viewer list
                    if inter_path:
                        results_to_viz.append({"name": f"Overlap ({partner_bio_name})", "path": inter_path})

                    # Rename the ID column for the final merge
                    id_col = f"id_{last_mask_name}"
                    
                    if final_metrics_df is None:
                        final_metrics_df = primary_df.copy().rename(columns={'label': id_col})
                        # Insert parent ID mapping for biological traceability
                        if parent_id_map:
                            map_df = pd.DataFrame(list(parent_id_map.items()), 
                                                 columns=[id_col, f"parent_id_{last_mask_name}"])
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