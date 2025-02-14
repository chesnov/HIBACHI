import numpy as np
from scipy import ndimage
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import time

def validate_segment(segment, min_volume=100, min_solidity=0.5):
    """
    Validate a segment based on size and shape criteria.
    
    Parameters:
    -----------
    segment : np.ndarray
        Binary mask of the segment
    min_volume : int
        Minimum volume in voxels
    min_solidity : float
        Minimum solidity (ratio of volume to convex hull volume)
    
    Returns:
    --------
    bool
        True if segment passes validation
    """
    # Check volume
    volume = np.sum(segment)
    if volume < min_volume:
        return False
        
    # Check solidity
    # Get convex hull volume using bounding box fill ratio as approximation
    coords = np.where(segment)
    bbox_volume = (coords[0].max() - coords[0].min() + 1) * \
                 (coords[1].max() - coords[1].min() + 1) * \
                 (coords[2].max() - coords[2].min() + 1)
    solidity = volume / bbox_volume
    
    return solidity >= min_solidity

def process_single_mask(args):
    """
    Process a single mask for parallel execution with improved validation.
    """
    label, coords, raw_chunk, min_distance, min_intensity_ratio, min_volume, min_solidity = args
    
    # Extract mask bounds
    z_slice = slice(coords['z_min'], coords['z_max'])
    y_slice = slice(coords['y_min'], coords['y_max'])
    x_slice = slice(coords['x_min'], coords['x_max'])
    
    # Create mask in minimal volume
    mask = np.zeros_like(raw_chunk, dtype=bool)
    mask[coords['mask_indices']] = True
    
    # Skip small objects
    if not validate_segment(mask, min_volume, min_solidity):
        return label, mask, None
    
    # Get intensity image within current mask
    masked_intensity = np.where(mask, raw_chunk, 0)
    
    # Find local maxima
    max_intensity = np.max(masked_intensity)
    peaks = peak_local_max(
        masked_intensity,
        min_distance=min_distance,
        threshold_abs=max_intensity * min_intensity_ratio,
        exclude_border=False,
        labels=mask
    )
    
    # If only one peak, return original mask
    if len(peaks) <= 1:
        return label, mask, None
        
    # Prepare markers for watershed
    markers = np.zeros_like(mask, dtype=int)
    for i, peak in enumerate(peaks):
        markers[peak[0], peak[1], peak[2]] = i + 1
        
    # Apply watershed
    distance = ndimage.distance_transform_edt(mask)
    watershed_labels = watershed(-distance, markers, mask=mask)
    
    # Validate each watershed segment
    valid_labels = set()
    for i in range(1, watershed_labels.max() + 1):
        segment = watershed_labels == i
        if validate_segment(segment, min_volume, min_solidity):
            valid_labels.add(i)
    
    # If no valid segments after splitting, return original mask
    if len(valid_labels) <= 1:
        return label, mask, None
    
    # Create new watershed_labels with only valid segments
    new_watershed_labels = np.zeros_like(watershed_labels)
    new_label = 1
    for old_label in valid_labels:
        new_watershed_labels[watershed_labels == old_label] = new_label
        new_label += 1
    
    return label, mask, new_watershed_labels

def split_merged_masks(labeled_cells, raw_intensity, min_distance=50, 
                              min_intensity_ratio=0.3, min_volume=100, 
                              min_solidity=0.1, n_processes=None):
    """
    Parallel implementation of merged cell mask splitting with improved validation.
    
    Parameters:
    -----------
    labeled_cells : np.ndarray
        3D array where each unique value represents a cell mask
    raw_intensity : np.ndarray
        Original 3D intensity image
    min_distance : int
        Minimum distance between intensity peaks (in pixels)
    min_intensity_ratio : float
        Minimum ratio of peak intensity to max intensity
    min_volume : int
        Minimum volume for valid segments
    min_solidity : float
        Minimum solidity for valid segments
    n_processes : int, optional
        Number of processes to use
    """
    if n_processes is None:
        n_processes = max(1, cpu_count() - 1)
    
    print(f"Starting parallel processing with {n_processes} processes")
    print(f"Minimum volume threshold: {min_volume} voxels")
    print(f"Minimum solidity threshold: {min_solidity}")
    start_time = time.time()
    
    # Initialize output array
    output = np.zeros_like(labeled_cells)
    next_label = 1
    
    # Get all unique labels (excluding background)
    unique_labels = np.unique(labeled_cells)
    unique_labels = unique_labels[unique_labels > 0]
    
    print(f"Found {len(unique_labels)} cells to process")
    
    # Prepare arguments for parallel processing
    parallel_args = []
    for label in unique_labels:
        mask = labeled_cells == label
        z, y, x = np.where(mask)
        
        padding = max(5, min_distance)
        z_min, z_max = max(0, z.min() - padding), min(labeled_cells.shape[0], z.max() + padding + 1)
        y_min, y_max = max(0, y.min() - padding), min(labeled_cells.shape[1], y.max() + padding + 1)
        x_min, x_max = max(0, x.min() - padding), min(labeled_cells.shape[2], x.max() + padding + 1)
        
        local_z = z - z_min
        local_y = y - y_min
        local_x = x - x_min
        
        raw_chunk = raw_intensity[z_min:z_max, y_min:y_max, x_min:x_max]
        
        coords = {
            'z_min': z_min, 'z_max': z_max,
            'y_min': y_min, 'y_max': y_max,
            'x_min': x_min, 'x_max': x_max,
            'mask_indices': (local_z, local_y, local_x)
        }
        
        parallel_args.append((label, coords, raw_chunk, min_distance, 
                            min_intensity_ratio, min_volume, min_solidity))
    
    # Process in parallel with progress bar
    with Pool(n_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_mask, parallel_args),
            total=len(parallel_args),
            desc="Processing cells"
        ))
    
    # Combine results
    print("Combining results...")
    invalid_segments = 0
    for label, mask, watershed_result in results:
        if watershed_result is None:
            coords = parallel_args[label-1][1]
            z_slice = slice(coords['z_min'], coords['z_max'])
            y_slice = slice(coords['y_min'], coords['y_max'])
            x_slice = slice(coords['x_min'], coords['x_max'])
            
            output_mask = np.zeros_like(output[z_slice, y_slice, x_slice])
            output_mask[mask] = next_label
            output[z_slice, y_slice, x_slice] = np.where(
                output[z_slice, y_slice, x_slice] == 0,
                output_mask,
                output[z_slice, y_slice, x_slice]
            )
            next_label += 1
        else:
            coords = parallel_args[label-1][1]
            z_slice = slice(coords['z_min'], coords['z_max'])
            y_slice = slice(coords['y_min'], coords['y_max'])
            x_slice = slice(coords['x_min'], coords['x_max'])
            
            for i in range(1, watershed_result.max() + 1):
                output_mask = np.zeros_like(output[z_slice, y_slice, x_slice])
                output_mask[watershed_result == i] = next_label
                output[z_slice, y_slice, x_slice] = np.where(
                    output[z_slice, y_slice, x_slice] == 0,
                    output_mask,
                    output[z_slice, y_slice, x_slice]
                )
                next_label += 1
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    if invalid_segments > 0:
        print(f"Filtered out {invalid_segments} invalid segments")
    
    return output