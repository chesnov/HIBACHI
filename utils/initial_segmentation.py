import numpy as np
from scipy import ndimage
from tqdm import tqdm
import os
from skimage import morphology, exposure, transform, feature
from multiprocessing import Pool, cpu_count
from skimage.segmentation import watershed
from skimage.measure import label
from itertools import product

from scipy import optimize
from skimage import exposure
from scipy.ndimage import gaussian_filter1d

def downsample_stack(image_stack, factor=2):
    """Downsample a 3D image stack by a given factor."""
    return image_stack[:, ::factor, ::factor]

def upsample_stack(labeled_stack, original_shape):
    """
    Upsamples the labeled stack to the original shape.
    """
    if labeled_stack.ndim == 3:
        return np.array([transform.resize(labeled_stack[z], original_shape[1:], 
                                      order=0, preserve_range=True, anti_aliasing=False) 
                     for z in range(labeled_stack.shape[0])], dtype=np.int32)
    elif labeled_stack.ndim == 2:
        return transform.resize(labeled_stack, original_shape, order=0, preserve_range=True, anti_aliasing=False)
    else:
        raise ValueError("Input stack must be either 2D or 3D.")


def interpolate_settings(z_position, z_size, slice_settings):
    """
    Interpolate settings for a given z position based on slice settings.
    """
    if len(slice_settings) >= 2:
        z_indices = np.array(sorted(slice_settings.keys()))
        # Scale z_indices to match the full volume size
        z_indices = (z_indices * z_size / max(z_indices)).astype(int)
        
        # Get settings for interpolation
        intensity_thresholds = np.array([
            slice_settings[z].get("intensity_threshold", 0.5) 
            for z in sorted(slice_settings.keys())
        ])
        min_volumes = np.array([
            slice_settings[z].get("min_volume", 500) 
            for z in sorted(slice_settings.keys())
        ])
        
        # Interpolate values for the specific z position
        intensity_threshold = np.interp(z_position, z_indices, intensity_thresholds)
        min_volume = np.interp(z_position, z_indices, min_volumes)
    else:
        # Default values if insufficient slice settings
        intensity_threshold = 0.5
        min_volume = 500
        
    return {
        "intensity_threshold": intensity_threshold,
        "min_volume": min_volume
    }

def get_volume_slices(shape, block_size, overlap):
    """
    Generate overlapping volume coordinates for parallel processing.
    Returns list of tuples (z_start, z_end, y_start, y_end, x_start, x_end)
    """
    print("Generating volume slices...")
    slices = []
    for dim_size, block_dim, overlap_dim in zip(shape, block_size, overlap):
        starts = list(range(0, dim_size - overlap_dim, block_dim - overlap_dim))
        if starts[-1] + block_dim < dim_size:
            starts.append(dim_size - block_dim)
        slices.append([(start, start + block_dim) for start in starts])
    
    coords = list(product(slices[0], slices[1], slices[2]))
    print(f"Created {len(coords)} subvolumes for processing")
    return coords

def process_subvolume(args):
    """
    Process a single subvolume using 3D watershed segmentation with depth-adaptive parameters.
    """
    volume_data, coords = args
    z_start, z_end = coords[0]
    y_start, y_end = coords[1]
    x_start, x_end = coords[2]
    
    # Get subvolume
    subvolume = volume_data['volume'][z_start:z_end, y_start:y_end, x_start:x_end]
    
    # Get depth-adaptive settings for the middle of this subvolume
    z_mid = (z_start + z_end) // 2
    settings = interpolate_settings(
        z_mid, 
        volume_data['volume'].shape[0],
        volume_data['slice_settings']
    )
    
    # Find local maxima with adaptive threshold
    local_max_coords = feature.peak_local_max(
        subvolume,
        min_distance=int(settings['min_volume']/3),
        threshold_rel=settings['intensity_threshold'],
        exclude_border=False
    )
    
    local_max = np.zeros_like(subvolume, dtype=bool)
    if len(local_max_coords) > 0:
        local_max[tuple(local_max_coords.T)] = True
    
    struct = ndimage.generate_binary_structure(3, 2)
    mask = subvolume > (settings['intensity_threshold'] * np.max(subvolume))
    distance = ndimage.distance_transform_edt(mask)
    markers = label(local_max)
    labels = watershed(-distance, markers, mask=mask)
    filtered_labels = morphology.remove_small_objects(labels, min_size=settings['min_volume'])
    
    return coords, filtered_labels

def make_labels_unique(results, start_label=1):
    """
    Ensure all labels across all blocks are unique by adding offsets.
    Returns modified results and the next available label.
    """
    print("Making labels unique across blocks...")
    new_results = []
    current_label = start_label
    
    for coords, labels in tqdm(results, desc="Relabeling blocks"):
        max_label = np.max(labels) if labels.any() else 0
        if max_label > 0:
            # Create mapping for this block's labels
            unique_labels = np.unique(labels)
            unique_labels = unique_labels[unique_labels > 0]
            label_map = {old: new for old, new in zip(unique_labels, 
                                                     range(current_label, current_label + len(unique_labels)))}
            
            # Apply mapping
            new_labels = np.zeros_like(labels)
            for old, new in label_map.items():
                new_labels[labels == old] = new
            
            new_results.append((coords, new_labels))
            current_label += len(unique_labels)
        else:
            new_results.append((coords, labels))
    
    return new_results, current_label

def find_overlapping_labels(label_array1, label_array2):
    """
    Find corresponding labels in overlapping regions between two arrays.
    Returns a dictionary mapping labels from array2 to array1.
    """
    mapping = {}
    
    # Find overlapping non-zero labels
    mask1 = label_array1 > 0
    mask2 = label_array2 > 0
    overlap_mask = mask1 & mask2
    
    if overlap_mask.any():
        labels1 = label_array1[overlap_mask]
        labels2 = label_array2[overlap_mask]
        
        # Count occurrences of each label pair
        unique_pairs, counts = np.unique(np.vstack((labels1, labels2)), axis=1, return_counts=True)
        
        # Sort by count to handle multiple overlaps
        sort_idx = np.argsort(-counts)
        unique_pairs = unique_pairs[:, sort_idx]
        
        # Create mapping from label2 to most overlapping label1
        for label1, label2 in unique_pairs.T:
            if label2 not in mapping:
                mapping[label2] = label1
    
    return mapping

def stitch_volumes(results, full_shape, overlap):
    """
    Stitch overlapping subvolumes ensuring consistent labeling across blocks.
    """
    print("Starting volume stitching...")
    
    # First make all labels unique across blocks
    results, _ = make_labels_unique(results)
    
    # Initialize final volume
    final_labels = np.zeros(full_shape, dtype=np.int32)
    label_mappings = {}
    
    # First pass: Place blocks and find overlapping regions
    print("First pass: Finding overlapping regions...")
    for coords, labels in tqdm(results, desc="Processing blocks"):
        z_start, z_end = coords[0]
        y_start, y_end = coords[1]
        x_start, x_end = coords[2]
        
        # Check overlaps with existing labels
        if z_start > 0:  # Check overlap with previous z block
            overlap_slice = slice(z_start, z_start + overlap[0])
            existing = final_labels[overlap_slice, y_start:y_end, x_start:x_end]
            new = labels[:overlap[0], :, :]
            mapping = find_overlapping_labels(existing, new)
            label_mappings.update(mapping)
        
        if y_start > 0:  # Check overlap with previous y block
            overlap_slice = slice(y_start, y_start + overlap[1])
            existing = final_labels[z_start:z_end, overlap_slice, x_start:x_end]
            new = labels[:, :overlap[1], :]
            mapping = find_overlapping_labels(existing, new)
            label_mappings.update(mapping)
            
        if x_start > 0:  # Check overlap with previous x block
            overlap_slice = slice(x_start, x_start + overlap[2])
            existing = final_labels[z_start:z_end, y_start:y_end, overlap_slice]
            new = labels[:, :, :overlap[2]]
            mapping = find_overlapping_labels(existing, new)
            label_mappings.update(mapping)
        
        # Place the block in the final volume
        final_labels[z_start:z_end, y_start:y_end, x_start:x_end] = labels
    
    # Optional: Relabel to ensure consecutive labels
    print("Relabeling to ensure consecutive labels...")
    unified_labels = label(final_labels > 0)
    
    return unified_labels


def intensity_based_segmentation(image_stack, slice_settings=None, temp_dir=None, downsample_factor=2):
    """
    3D cell segmentation using parallel processing of overlapping subvolumes with depth-adaptive parameters.
    """
    print("Initializing 3D segmentation...")
    if slice_settings is None:
        slice_settings = {}
    os.makedirs(temp_dir, exist_ok=True)

    # Default block processing settings
    default_block_settings = {
        "block_size": (64, 64, 64),
        "overlap": (16, 16, 16)
    }
    
    print("Preprocessing image stack...")
    print("Enhancing contrast...")
    enhanced_stack = adaptive_contrast_enhancement(image_stack)
    print("Downsampling...")
    downsampled_stack = downsample_stack(enhanced_stack, factor=downsample_factor)
    
    
    volume_coords = get_volume_slices(
        downsampled_stack.shape,
        default_block_settings["block_size"],
        default_block_settings["overlap"]
    )
    
    # Package data for parallel processing
    volume_data = {
        'volume': downsampled_stack,
        'slice_settings': slice_settings  # Pass full slice settings for interpolation
    }
    
    num_workers = min(cpu_count(), len(volume_coords))
    print(f"Starting parallel processing with {num_workers} workers...")
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_subvolume, [(volume_data, coord) for coord in volume_coords]),
            total=len(volume_coords),
            desc="Processing subvolumes"
        ))
    
    print("Stitching subvolumes...")
    labeled_volume = stitch_volumes(results, downsampled_stack.shape, default_block_settings["overlap"])
    
    if downsample_factor > 1:
        print("Upsampling results...")
        labeled_volume = upsample_stack(labeled_volume, image_stack.shape)
    
    print("Saving results...")
    save_path = os.path.join(temp_dir, "segmented_cells.npy")
    np.save(save_path, labeled_volume)
    print(f"Results saved to: {save_path}")
    
    return labeled_volume


def get_depth_weight(z, z_depth, edge_protection=0.2):
    """
    Calculate weight for enhancement strength based on depth position.
    Reduces enhancement strength at the edges of the stack.
    
    Parameters:
    -----------
    z : int
        Current z position
    z_depth : int
        Total depth of stack
    edge_protection : float
        Controls how quickly enhancement strength drops at edges (0-1)
        Higher values mean more conservative enhancement at edges
    """
    # Convert to 0-1 range
    rel_pos = z / (z_depth - 1)
    
    # Create a curve that drops off at the edges
    # Using smooth cosine falloff
    edge_weight = np.cos((rel_pos - 0.5) * np.pi) * 0.5 + 0.5
    
    # Apply edge protection factor
    edge_weight = edge_weight ** edge_protection
    
    return edge_weight

def fit_percentile_curve(image_stack, percentile, window_size=5, smoothing_sigma=1.0):
    """
    Fits a smooth curve to intensity percentiles across z-depth.
    """
    z_depth = image_stack.shape[0]
    percentile_values = np.zeros(z_depth)
    
    # Calculate percentiles with moving window
    pad_size = window_size // 2
    padded_stack = np.pad(image_stack, ((pad_size, pad_size), (0, 0), (0, 0)), mode='reflect')
    
    for z in range(z_depth):
        window = padded_stack[z:z + window_size]
        percentile_values[z] = np.percentile(window, percentile)
    
    # Smooth the percentile curve
    smoothed_values = gaussian_filter1d(percentile_values, smoothing_sigma)
    
    return smoothed_values

def fit_depth_decay(z_values, intensity_values):
    """
    Fits an exponential decay curve to intensity values across depth.
    """
    def exp_decay(z, amplitude, decay_rate, offset):
        return amplitude * np.exp(-decay_rate * z) + offset
    
    # Initial parameter guesses
    p0 = [
        np.max(intensity_values) - np.min(intensity_values),
        1/len(z_values),
        np.min(intensity_values)
    ]
    
    try:
        params, _ = optimize.curve_fit(exp_decay, z_values, intensity_values, p0=p0)
        return params
    except RuntimeError:
        return None

def adaptive_contrast_enhancement(image_stack, low_percentile=25, high_percentile=99.9, 
                                   window_size=5, smoothing_sigma=1.0, edge_protection=0.4):
    """
    Enhances contrast for each slice using depth-aware percentile curves with edge protection.
    
    Parameters:
    -----------
    image_stack : ndarray
        3D image stack
    low_percentile : float
        Lower percentile for contrast stretching
    high_percentile : float
        Upper percentile for contrast stretching
    window_size : int
        Size of moving window for percentile calculation
    smoothing_sigma : float
        Sigma for Gaussian smoothing of percentile values
    edge_protection : float
        Strength of edge protection (0-1). Higher values mean more conservative 
        enhancement at stack edges.
    """
    z_depth = image_stack.shape[0]
    z_values = np.arange(z_depth)
    
    # Get smoothed percentile curves
    low_curve = fit_percentile_curve(image_stack, low_percentile, window_size, smoothing_sigma)
    high_curve = fit_percentile_curve(image_stack, high_percentile, window_size, smoothing_sigma)
    
    # Fit decay curves
    low_params = fit_depth_decay(z_values, low_curve)
    high_params = fit_depth_decay(z_values, high_curve)
    
    if low_params is not None and high_params is not None:
        # Use fitted curves
        def exp_decay(z, amplitude, decay_rate, offset):
            return amplitude * np.exp(-decay_rate * z) + offset
        
        low_fitted = exp_decay(z_values, *low_params)
        high_fitted = exp_decay(z_values, *high_params)
    else:
        # Fallback to smoothed curves if fitting fails
        low_fitted = low_curve
        high_fitted = high_curve
    
    # Apply enhancement using fitted curves with edge protection
    enhanced_stack = np.zeros_like(image_stack, dtype=np.float32)
    global_min = np.min(image_stack)
    global_max = np.max(image_stack)
    
    for z in range(z_depth):
        # Get depth-dependent weight
        weight = get_depth_weight(z, z_depth, edge_protection)
        
        # Calculate enhanced and conservative intensity ranges
        p_low_enhanced = low_fitted[z]
        p_high_enhanced = high_fitted[z]
        
        # Blend between enhanced and conservative ranges based on weight
        p_low = global_min * (1 - weight) + p_low_enhanced * weight
        p_high = global_max * (1 - weight) + p_high_enhanced * weight
        
        # Apply enhancement
        enhanced_stack[z] = exposure.rescale_intensity(
            image_stack[z],
            in_range=(p_low, p_high),
            out_range=(0, 1)
        )
    
    return enhanced_stack