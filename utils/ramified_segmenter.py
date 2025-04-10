import numpy as np
from scipy import ndimage
from tqdm import tqdm
from multiprocessing import Pool
from shutil import rmtree
import gc
import numpy as np
from tqdm import tqdm
from functools import partial
from skimage.measure import regionprops
import math
seed = 42
np.random.seed(seed)         # For NumPy

# Assuming nuclear_segmenter.py might be needed elsewhere, keep imports
# from nuclear_segmenter import downsample_for_isotropic, upsample_segmentation

def extract_soma_masks(segmentation_mask, 
                      small_object_percentile=50,  # Changed to percentile
                      thickness_percentile=80):
    """
    Memory-efficient soma extraction with percentile-based small object removal and label reassignment.
    
    Parameters:
    - segmentation_mask: 3D numpy array with labeled segments
    - small_object_percentile: Percentile of object volumes to keep (e.g., 50 keeps top 50%)
    - thickness_percentile: Percentile for thickness-based soma detection
    """
    
    # Create output soma mask
    soma_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
    
    # Get unique labels, excluding background
    unique_labels = np.unique(segmentation_mask)[1:]
    
    # Keep track of the next available label for reassignment
    next_label = np.max(unique_labels) + 1 if len(unique_labels) > 0 else 1
    
    # Process each label
    for label in tqdm(unique_labels):
        # Extract current cell mask
        cell_mask = segmentation_mask == label
        
        # Get bounding box for the cell
        props = regionprops(cell_mask.astype(int))[0]
        bbox = props.bbox
        
        # Extract subvolumes using bounding box with padding
        z_min, y_min, x_min, z_max, y_max, x_max = bbox
        z_min = max(0, z_min - 2)
        y_min = max(0, y_min - 2)
        x_min = max(0, x_min - 2)
        z_max = min(segmentation_mask.shape[0], z_max + 2)
        y_max = min(segmentation_mask.shape[1], y_max + 2)
        x_max = min(segmentation_mask.shape[2], x_max + 2)
        
        # Extract subarrays
        cell_mask_subvolume = cell_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        
        # Compute distance transform on cell mask subvolume
        distance_map = ndimage.distance_transform_edt(cell_mask_subvolume)
        
        # Compute thickness threshold
        thickness_threshold = np.percentile(distance_map[cell_mask_subvolume], thickness_percentile)
        
        # Create max thickness mask
        max_thickness_mask = np.zeros_like(distance_map, dtype=bool)
        max_thickness_mask[np.logical_and(distance_map >= thickness_threshold, cell_mask_subvolume)] = True
        
        # Label connected components in the subvolume
        labeled_somas, num_features = ndimage.label(max_thickness_mask)
        
        # If no somas detected, skip to next label
        if num_features == 0:
            continue
        
        # Map back to full volume
        full_max_thickness_mask = np.zeros_like(cell_mask, dtype=np.int32)
        full_max_thickness_mask[z_min:z_max, y_min:y_max, x_min:x_max] = labeled_somas
        
        # Get properties of connected components
        soma_props = regionprops(full_max_thickness_mask)
        
        # If only one object, keep it regardless of size
        if num_features == 1:
            soma_mask[full_max_thickness_mask > 0] = label
        else:
            # Compute volumes of all objects
            volumes = [prop.area for prop in soma_props]
            
            # Calculate the volume threshold based on percentile
            volume_threshold = np.percentile(volumes, small_object_percentile)
            
            # Filter objects above the percentile and reassign labels
            for prop in soma_props:
                if prop.area >= volume_threshold:  # Keep if above threshold
                    soma_mask[full_max_thickness_mask == prop.label] = next_label
                    next_label += 1
    
    return soma_mask

def separate_multi_soma_cells(segmentation_mask, intensity_volume, soma_mask, min_size_threshold=100):
    """
    Separates cell segmentations with multiple somas into distinct masks by using watershed
    transform with distance transforms to ensure separation along thinnest regions between somas.
    
    Parameters:
    - segmentation_mask: 3D numpy array with labeled cell segments
    - intensity_volume: 3D numpy array with original intensity values
    - soma_mask: 3D numpy array with labeled soma segments (output from extract_soma_masks)
    - min_size_threshold: Minimum voxel size for a separated component (smaller ones are merged unless original)
    
    Returns:
    - separated_mask: 3D numpy array with updated cell segmentations
    """
    import numpy as np
    from scipy import ndimage
    from skimage.measure import regionprops
    from skimage.segmentation import watershed
    from tqdm import tqdm
    
    # Create output mask, initially copying the original segmentation
    separated_mask = np.copy(segmentation_mask).astype(np.int32)
    
    # Get unique cell labels and their original sizes from segmentation_mask
    unique_cell_labels = np.unique(segmentation_mask)[1:]
    original_sizes = {lbl: np.sum(segmentation_mask == lbl) for lbl in unique_cell_labels}
    
    # Keep track of the next available label
    next_label = np.max(segmentation_mask) + 1 if len(unique_cell_labels) > 0 else 1
    
    # Process each cell
    for cell_label in tqdm(unique_cell_labels):
        # Extract current cell mask
        cell_mask = segmentation_mask == cell_label
        
        # Get bounding box for the cell
        props = regionprops(cell_mask.astype(int))[0]
        bbox = props.bbox
        z_min, y_min, x_min, z_max, y_max, x_max = bbox
        
        # Add slight padding
        z_min = max(0, z_min - 2)
        y_min = max(0, y_min - 2)
        x_min = max(0, x_min - 2)
        z_max = min(segmentation_mask.shape[0], z_max + 2)
        y_max = min(segmentation_mask.shape[1], y_max + 2)
        x_max = min(segmentation_mask.shape[2], x_max + 2)
        
        # Extract subvolumes
        cell_mask_sub = cell_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        intensity_sub = intensity_volume[z_min:z_max, y_min:y_max, x_min:x_max]
        cell_soma_sub = soma_mask[z_min:z_max, y_min:y_max, x_min:x_max] * cell_mask_sub
        
        # Get unique soma labels within this cell, excluding background
        soma_labels = np.unique(cell_soma_sub)[1:]
        
        # Skip if no somas or only one soma
        if len(soma_labels) <= 1:
            continue
        
        # Number of somas
        num_somas = len(soma_labels)
        print(f"Cell {cell_label} has {num_somas} somas, separating...")
        
        # Create markers for watershed segmentation
        # First, ensure soma regions are well defined
        soma_markers = np.zeros_like(cell_mask_sub, dtype=np.int32)
        for i, soma_label in enumerate(soma_labels):
            # Mark each soma with a unique index starting from 1
            soma_region = cell_soma_sub == soma_label
            # Dilate soma slightly to ensure good markers
            soma_region = ndimage.binary_dilation(soma_region, iterations=1)
            soma_markers[soma_region] = i + 1
        
        # Compute distance transform of the cell mask
        # This helps identify thin regions (smaller distance values)
        # The negative distance transform has peaks at the center of large regions
        distance_transform = ndimage.distance_transform_edt(cell_mask_sub)
        
        # Create a special weighting for somas to avoid cutting through them
        # Make distances through somas artificially high
        soma_weighting = np.zeros_like(distance_transform)
        for soma_label in soma_labels:
            soma_region = cell_soma_sub == soma_label
            soma_weighting[soma_region] = 1000  # Very high value to avoid cutting through somas
        
        # Modified distance transform that penalizes paths through somas
        modified_distance = distance_transform + soma_weighting
        
        # Apply watershed with markers (somas) and weights (distance transform)
        # This will separate along the thinnest regions (valleys in the distance transform)
        watershed_result = watershed(modified_distance, soma_markers, mask=cell_mask_sub)
        
        # Create temp_mask with watershed result
        temp_mask = np.zeros_like(watershed_result, dtype=np.int32)
        label_map = [cell_label] + [next_label + i for i in range(num_somas - 1)]
        
        # Map watershed labels to cell labels
        for i in range(num_somas):
            region_mask = watershed_result == (i + 1)
            temp_mask[region_mask] = label_map[i]
        
        # Ensure continuity: Check for discontinuous components
        for i, lbl in enumerate(label_map):
            lbl_mask = temp_mask == lbl
            labeled_components, num_components = ndimage.label(lbl_mask)
            if num_components > 1:
                print(f"Warning: soma {i} of cell {cell_label} is discontinuous, merging...")
                props = regionprops(labeled_components)
                main_component = max(props, key=lambda p: p.area).label
                main_mask = labeled_components == main_component
                for prop in props:
                    if prop.label != main_component:
                        dilated = ndimage.binary_dilation(labeled_components == prop.label, iterations=1)
                        touching_labels = np.unique(temp_mask[dilated & (labeled_components != prop.label)])
                        valid_touching = [l for l in touching_labels if l != 0 and l != lbl]
                        if valid_touching:
                            temp_mask[labeled_components == prop.label] = valid_touching[0]
                        else:
                            temp_mask[labeled_components == prop.label] = lbl
                            temp_mask[main_mask] = lbl
        
        # Enforce size threshold, preserving original small regions
        final_labels = np.unique(temp_mask)[1:]  # Exclude background
        for lbl in final_labels:
            lbl_mask = temp_mask == lbl
            size = np.sum(lbl_mask)
            if size < min_size_threshold and size != original_sizes.get(lbl, float('inf')):
                print(f"Merging soma {lbl} of cell {cell_label} due to size {size}")
                # Merge if below threshold and not the original size
                dilated = ndimage.binary_dilation(lbl_mask, iterations=1)
                touching_labels = np.unique(temp_mask[dilated & ~lbl_mask])
                valid_touching = [l for l in touching_labels if l != 0 and np.sum(temp_mask == l) >= min_size_threshold]
                if valid_touching:
                    temp_mask[lbl_mask] = valid_touching[0]  # Merge with largest valid neighbor
                else:
                    temp_mask[lbl_mask] = label_map[0]  # Merge with original label
        
        # Update next_label based on used labels
        used_labels = np.unique(temp_mask)[1:]
        if len(used_labels) > 0:
            next_label = max(next_label, np.max(used_labels) + 1)
        
        # Map back to full volume
        # Get the current subvolume from the separated_mask
        full_subvol = separated_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        # Only replace voxels that belong to the current cell
        current_cell_voxels = cell_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        # Update only the voxels belonging to the current cell
        full_subvol[current_cell_voxels] = temp_mask[current_cell_voxels]
        # Write back to the full separated_mask
        separated_mask[z_min:z_max, y_min:y_max, x_min:x_max] = full_subvol
    
    return separated_mask