import os
import numpy as np
import pandas as pd
import dask.array as da
import dask_image.ndmeasure
from scipy import ndimage
import os
import numpy as np
from typing import List, Dict
from ..module_3d.interaction_analysis import calculate_interaction_metrics
from ..module_2d.interaction_analysis_2d import calculate_interaction_metrics_2d

class RelationalEngine:
    """Core logic for performing math between different channel masks."""
    
    @staticmethod
    def intersect_masks(path_a, path_b, out_path, shape):
        """Boolean AND between two label files."""
        ma = np.memmap(path_a, dtype=np.int32, mode='r', shape=shape)
        mb = np.memmap(path_b, dtype=np.int32, mode='r', shape=shape)
        
        out = np.memmap(out_path, dtype=np.int32, mode='w+', shape=shape)
        # We treat any label > 0 as 'foreground'
        mask = (ma > 0) & (mb > 0)
        out[mask] = 1 
        
        out.flush()
        return out_path

    @staticmethod
    def filter_by_volume(path_in, out_path, shape, spacing, min_vol_um3):
        """Removes objects smaller than a physical volume threshold."""
        data = np.memmap(path_in, dtype=np.int32, mode='r', shape=shape)
        unit_vol = np.prod(spacing)
        
        # Use dask for parallel labeling if it's just a binary mask
        d_data = da.from_array(data, chunks=(64, 256, 256) if len(shape)==3 else (4096, 4096))
        labeled, num = dask_image.ndmeasure.label(d_data)
        
        # Calculate volumes
        objs = ndimage.find_objects(labeled.compute()) # Compute labels for small-ish result
        out = np.memmap(out_path, dtype=np.int32, mode='w+', shape=shape)
        
        actual_data = labeled.compute()
        valid_count = 0
        for i, sl in enumerate(objs):
            if sl is None: continue
            lbl = i + 1
            mask = (actual_data[sl] == lbl)
            vol = np.count_nonzero(mask) * unit_vol
            if vol >= min_vol_um3:
                valid_count += 1
                out[sl][mask] = valid_count
        
        out.flush()
        return out_path
    
    @staticmethod
    def run_recipe(sample_name, registry, recipe, out_dir, shape, spacing):
        """
        Runs the full recipe on a sample.
        Saves CSVs and masks to out_dir.
        """
        sample_channels = registry.get(sample_name, {})
        last_mask_path = None
        results_to_viz = []
        final_metrics_df = None

        os.makedirs(out_dir, exist_ok=True)
        is_2d = (len(shape) == 2)

        for i, step in enumerate(recipe):
            step_type = step['type']
            step_out_path = os.path.join(out_dir, f"step_{i}_{step_type}.dat")

            if step_type == "intersect":
                path_a = RelationalEngine._find_dat(sample_channels.get(step['inputs'][0]))
                if step['inputs'][1] == "PREVIOUS_RESULT":
                    path_b = last_mask_path
                else:
                    path_b = RelationalEngine._find_dat(sample_channels.get(step['inputs'][1]))
                
                if path_a and path_b:
                    last_mask_path = RelationalEngine.intersect_masks(path_a, path_b, step_out_path, shape)
                    results_to_viz.append({"name": step['name'], "path": last_mask_path})

            elif step_type == "filter":
                if last_mask_path:
                    last_mask_path = RelationalEngine.filter_by_volume(last_mask_path, step_out_path, shape, spacing, step['min_vol'])
                    results_to_viz.append({"name": step['name'], "path": last_mask_path})

            elif step_type == "analyze":
                # THE CORE RELATIONAL CALCULATION
                ref_ch_path = sample_channels.get(step['target'])
                ref_dat_path = RelationalEngine._find_dat(ref_ch_path)
                
                if last_mask_path and ref_dat_path:
                    # Choose 2D or 3D logic
                    if is_2d:
                        # spacing for 2D is (y, x)
                        sp_2d = (spacing[0], spacing[1]) if len(spacing)==2 else (spacing[1], spacing[2])
                        primary_df, _, _ = calculate_interaction_metrics_2d(
                            last_mask_path, ref_dat_path, out_dir, shape, sp_2d, step['target']
                        )
                    else:
                        primary_df, _, _ = calculate_interaction_metrics(
                            last_mask_path, ref_dat_path, out_dir, shape, spacing, step['target']
                        )
                    
                    # Accumulate metrics
                    if final_metrics_df is None:
                        final_metrics_df = primary_df
                    else:
                        # Merge if we have multiple analysis steps in one recipe
                        final_metrics_df = pd.merge(final_metrics_df, primary_df, on='label', how='outer')

        # Save final metrics for this sample
        if final_metrics_df is not None:
            csv_name = f"{sample_name}_relational_results.csv"
            final_metrics_df.to_csv(os.path.join(out_dir, csv_name), index=False)

        return results_to_viz

    @staticmethod
    def _find_dat(folder_path):
        """Helper to find the final_segmentation.dat in a project folder."""
        if not folder_path or not os.path.isdir(folder_path): return None
        
        # We look for folders ending in _processed_ramified or _processed_ramified_2d
        for d in os.listdir(folder_path):
            if "_processed_" in d:
                proc_dir = os.path.join(folder_path, d)
                if not os.path.isdir(proc_dir): continue
                for f in os.listdir(proc_dir):
                    if f.startswith("final_segmentation") and f.endswith(".dat"):
                        return os.path.join(proc_dir, f)
        return None