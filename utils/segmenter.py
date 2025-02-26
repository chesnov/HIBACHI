import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label, distance_transform_edt, zoom
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import frangi, threshold_otsu
from skimage.morphology import binary_dilation, ball
import tempfile
from skimage.morphology import skeletonize
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import psutil
from shutil import rmtree

def get_object_roi(segmentation, volume, label, padding=5):
    """Extract ROI around a labeled object with padding."""
    # Get object mask and bounding box
    mask = segmentation == label
    z, y, x = np.where(mask)
    if len(z) == 0:
        return None
        
    # Calculate padded bounding box
    bb_min = np.maximum(0, [z.min() - padding, y.min() - padding, x.min() - padding])
    bb_max = np.minimum(
        [segmentation.shape[0], segmentation.shape[1], segmentation.shape[2]],
        [z.max() + padding + 1, y.max() + padding + 1, x.max() + padding + 1]
    )
    
    # Extract ROIs
    roi_seg = segmentation[bb_min[0]:bb_max[0], 
                          bb_min[1]:bb_max[1], 
                          bb_min[2]:bb_max[2]].copy()
    roi_vol = volume[bb_min[0]:bb_max[0], 
                    bb_min[1]:bb_max[1], 
                    bb_min[2]:bb_max[2]].copy()
    
    return {
        'roi_seg': roi_seg,
        'roi_vol': roi_vol,
        'bb_min': bb_min,
        'bb_max': bb_max,
        'label': label
    }

def process_single_object(roi_data):
    """Process a single object ROI."""
    try:
        if roi_data is None:
            return None
            
        roi_seg = roi_data['roi_seg']
        roi_vol = roi_data['roi_vol']
        label = roi_data['label']
        bb_min = roi_data['bb_min']
        bb_max = roi_data['bb_max']
        
        # Get object mask
        mask = roi_seg == label
        if not np.any(mask):
            return None
        
        # Calculate object properties
        coords = np.array(np.where(mask)).T
        if len(coords) < 10:  # Skip very small objects
            return None
            
        # Calculate elongation using PCA
        centered_coords = coords - np.mean(coords, axis=0)
        cov = np.cov(centered_coords.T)
        eigenvals = np.linalg.eigvals(cov)
        eigenvals = np.sqrt(np.abs(eigenvals))
        elongation = eigenvals.max() / np.clip(eigenvals.min(), 1e-10, None)
        
        # If object isn't elongated enough, skip it
        if elongation < 2.0:
            return None
            
        # Calculate distance transform
        distance = ndimage.distance_transform_edt(mask)
        
        # Find local maxima for seeds
        local_max = peak_local_max(
            distance,
            min_distance=5,
            threshold_rel=0.5,
            labels=mask
        )
        
        if len(local_max) <= 1:
            return None
            
        # Prepare markers for watershed
        markers = np.zeros_like(mask, dtype=int)
        for i, center in enumerate(local_max, 1):
            markers[tuple(center)] = i
            
        # Apply watershed to split the object
        split_mask = watershed(-distance, markers, mask=mask)
        
        # Prepare results with full volume coordinates
        results = []
        full_shape = (bb_max[0] - bb_min[0], 
                     bb_max[1] - bb_min[1], 
                     bb_max[2] - bb_min[2])
                     
        for i in range(1, split_mask.max() + 1):
            submask = split_mask == i
            if np.sum(submask) > 0:
                results.append({
                    'mask': submask,
                    'bb_min': bb_min,
                    'bb_max': bb_max,
                    'full_shape': full_shape
                })
        
        return results if results else None
        
    except Exception as e:
        print(f"Error processing label {label}: {str(e)}")
        return None

def split_merged_objects_parallel(segmentation, volume, n_processes=None):
    """Split merged objects using parallel processing with ROI extraction."""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    print("\nSystem information:")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Using {n_processes} processes")
    
    start_time = time.time()
    
    try:
        # Get unique labels
        labels = np.unique(segmentation)[1:]  # exclude background
        print(f"\nProcessing {len(labels)} objects...")
        
        # Prepare ROIs for parallel processing
        rois = []
        for label in tqdm(labels, desc="Extracting ROIs"):
            roi = get_object_roi(segmentation, volume, label)
            if roi is not None:
                rois.append(roi)
        
        # Process ROIs in parallel
        new_segmentation = np.zeros_like(segmentation)
        next_label = 1
        
        with mp.Pool(n_processes) as pool:
            results = list(tqdm(
                pool.imap(process_single_object, rois),
                total=len(rois),
                desc="Processing objects"
            ))
            
            # Process results
            n_split = 0
            for roi_data, result in zip(rois, results):
                if result is None:
                    # Copy original object
                    label = roi_data['label']
                    new_segmentation[segmentation == label] = next_label
                    next_label += 1
                else:
                    # Add split objects
                    for split_data in result:
                        submask = split_data['mask']
                        bb_min = split_data['bb_min']
                        bb_max = split_data['bb_max']
                        
                        # Create temporary full-size mask
                        temp_mask = np.zeros_like(segmentation)
                        temp_mask[bb_min[0]:bb_max[0],
                                bb_min[1]:bb_max[1],
                                bb_min[2]:bb_max[2]][submask] = 1
                                
                        new_segmentation[temp_mask > 0] = next_label
                        next_label += 1
                    n_split += 1
        
        elapsed_time = time.time() - start_time
        print(f"\nSplitting complete in {elapsed_time:.1f} seconds:")
        print(f"Original objects: {len(labels)}")
        print(f"Objects split: {n_split}")
        print(f"Final objects: {next_label - 1}")
        
        return new_segmentation
        
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        raise


def analyze_intensity_distribution(volume):
    """Analyze intensity distribution and suggest parameters."""
    print("\nAnalyzing intensity distribution...")
    
    # Get basic statistics
    mean_intensity = np.mean(volume)
    std_intensity = np.std(volume)
    max_intensity = np.max(volume)
    min_intensity = np.min(volume)
    
    # Calculate percentiles
    percentiles = np.percentile(volume, [1, 25, 50, 75, 99])
    
    print(f"Image statistics:")
    print(f"Min intensity: {min_intensity:.2f}")
    print(f"Max intensity: {max_intensity:.2f}")
    print(f"Mean intensity: {mean_intensity:.2f}")
    print(f"Std deviation: {std_intensity:.2f}")
    print(f"1st percentile: {percentiles[0]:.2f}")
    print(f"25th percentile: {percentiles[1]:.2f}")
    print(f"Median: {percentiles[2]:.2f}")
    print(f"75th percentile: {percentiles[3]:.2f}")
    print(f"99th percentile: {percentiles[4]:.2f}")
    
    # Plot histogram
    plt.figure(figsize=(10, 4))
    plt.hist(volume.ravel(), bins=100)
    plt.title('Intensity Distribution')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')
    plt.show()
    
    # Suggest parameters based on distribution
    suggested_params = {
        'soma_threshold': percentiles[3],  # 75th percentile for soma detection
        'process_threshold_factor': (percentiles[3] - percentiles[1]) / std_intensity,
        'background_threshold': percentiles[1]  # 25th percentile for background
    }
    
    return suggested_params

def segment_microglia(volume, 
                     soma_sigma=1.0,
                     soma_min_distance=10,
                     intensity_threshold_factor=None,
                     max_process_length=100,
                     bridge_max_gap=5,
                     direction_tolerance=30,
                     split_merged=True,
                     n_processes=None):
    """
    Segment ramified microglia from 3D fluorescence volume.
    """

    suggested_params = analyze_intensity_distribution(volume)
    if intensity_threshold_factor is None:
        intensity_threshold_factor = suggested_params['process_threshold_factor']
        print(f"\nUsing suggested intensity threshold factor: {intensity_threshold_factor:.2f}")
    
    print("\nStarting microglia segmentation...")
    
    # Detect somas
    print("Detecting somas...")
    smoothed = gaussian_filter(volume, sigma=soma_sigma)
    background = smoothed[smoothed < np.percentile(smoothed, 70)]
    bg_mean, bg_std = np.mean(background), np.std(background)
    threshold = bg_mean + 3 * bg_std
    
    soma_centers = peak_local_max(
        smoothed,
        min_distance=soma_min_distance,
        threshold_abs=threshold,
        exclude_border=False
    )
    
    print(f"Found {len(soma_centers)} potential soma centers")
    
    if len(soma_centers) == 0:
        print("Warning: No soma centers detected. Adjusting threshold...")
        threshold = bg_mean + 2 * bg_std
        soma_centers = peak_local_max(
            smoothed,
            min_distance=soma_min_distance,
            threshold_abs=threshold,
            exclude_border=False
        )
        print(f"After threshold adjustment: Found {len(soma_centers)} soma centers")
    
    # Create markers
    markers = np.zeros_like(volume, dtype=int)
    for i, center in enumerate(soma_centers, 1):
        markers[tuple(center)] = i
    
    # Enhance processes
    print("Enhancing processes...")
    enhanced = np.zeros_like(volume)
    for i in tqdm(range(volume.shape[0]), desc="Applying Frangi filter"):
        enhanced[i] = frangi(volume[i], scale_range=(1, 3), scale_step=1)
    
    # Calculate distance map
    print("Calculating distance map...")
    distance_map = ndimage.distance_transform_edt(markers == 0)
    
    # Adaptive threshold
    print("Applying adaptive threshold...")
    local_mean = ndimage.uniform_filter(volume, size=5)
    local_std = np.sqrt(
        ndimage.uniform_filter(volume**2, size=5) - local_mean**2
    )
    distance_factor = 1.0 / (1.0 + distance_map / max_process_length)
    threshold = local_mean + intensity_threshold_factor * local_std * distance_factor
    mask = (volume > threshold) | (enhanced > np.percentile(enhanced, 95))
    
    print(f"Mask coverage: {100 * np.mean(mask):.2f}% of volume")
    
    # Watershed segmentation
    print("Running watershed segmentation...")
    segmentation = watershed(-volume, markers, mask=mask)
    
    # Connect discontinuous processes
    print("Connecting discontinuous processes...")
    final_segmentation = connect_processes(segmentation, volume, bridge_max_gap, direction_tolerance)
    
    # Final statistics
    n_cells = len(np.unique(final_segmentation)) - 1  # subtract background
    print(f"\nSegmentation complete. Found {n_cells} cells")
    
    if split_merged:
        segmentation = split_merged_objects_parallel(
            segmentation,
            volume,
            n_processes=n_processes
        )
    
    return segmentation

def connect_processes(segmentation, volume, max_gap, direction_tolerance):
    """Connect discontinuous processes based on direction and proximity."""
    labeled_regions = np.unique(segmentation)[1:]
    result = segmentation.copy()
    
    for label in tqdm(labeled_regions, desc="Processing cells"):
        mask = segmentation == label
        skeleton = skeletonize(mask)
        endpoints = find_endpoints(skeleton)
        
        n_bridges = 0
        for i, ep1 in enumerate(endpoints):
            for ep2 in endpoints[i+1:]:
                distance = np.sqrt(np.sum((ep1 - ep2)**2))
                if distance > max_gap:
                    continue
                
                dir1 = get_endpoint_direction(skeleton, ep1)
                dir2 = get_endpoint_direction(skeleton, ep2)
                if dir1 is None or dir2 is None:
                    continue
                    
                angle = np.arccos(np.clip(np.dot(dir1, dir2), -1.0, 1.0))
                if np.degrees(angle) > direction_tolerance:
                    continue
                
                bridge = create_bridge(ep1, ep2)
                result[tuple(bridge.T)] = label
                n_bridges += 1
        
        if n_bridges > 0:
            print(f"Cell {label}: Connected {n_bridges} process bridges")
    
    return result

# Other helper functions remain the same
def find_endpoints(skeleton):
    """Find endpoints in skeleton."""
    coords = np.array(np.where(skeleton)).T
    endpoints = []
    
    for coord in coords:
        x, y, z = coord
        neighborhood = skeleton[
            max(0, x-1):x+2,
            max(0, y-1):y+2,
            max(0, z-1):z+2
        ]
        if np.sum(neighborhood) == 2:  # center point + one neighbor
            endpoints.append(coord)
    
    return np.array(endpoints)

def get_endpoint_direction(skeleton, endpoint):
    """Calculate direction vector at endpoint."""
    x, y, z = endpoint
    neighborhood = skeleton[
        max(0, x-2):x+3,
        max(0, y-2):y+3,
        max(0, z-2):z+3
    ]
    
    points = np.array(np.where(neighborhood)).T
    if len(points) < 2:
        return None
        
    center = np.array([min(2, x), min(2, y), min(2, z)])
    directions = points - center
    direction = np.mean(directions, axis=0)
    norm = np.linalg.norm(direction)
    
    return direction / norm if norm > 0 else None

def create_bridge(point1, point2):
    """Create points between two endpoints."""
    distance = np.sqrt(np.sum((point2 - point1)**2))
    num_points = int(np.ceil(distance))
    t = np.linspace(0, 1, num_points)
    
    bridge_points = np.array([
        point1[0] + (point2[0] - point1[0]) * t,
        point1[1] + (point2[1] - point1[1]) * t,
        point1[2] + (point2[2] - point1[2]) * t
    ]).T
    
    return bridge_points.astype(int)


def downsample_for_isotropic(volume, spacing, anisotropy_normalization_degree=1.0):
    """
    Downsample the volume in x-y/z to match the maximum axis resolution for isotropy, using memmap for output,
    with adjustable normalization degree. Degree = 0 means no downsampling (original volume),
    degree = 1 means full isotropic downsampling.
    
    Parameters:
    -----------
    volume : ndarray
        3D input volume
    spacing : list or tuple
        Original spacing (z, y, x) in physical units (e.g., µm)
    anisotropy_normalization_degree : float
        Degree of anisotropy normalization (0 to 1)
    
    Returns:
    --------
    downsampled_volume : ndarray or memmap
        Isotropically resampled volume
    new_spacing : tuple
        New isotropic spacing (or original if degree = 0)
    temp_dir : str
        Temporary directory for memmap files
    """
    if not 0 <= anisotropy_normalization_degree <= 1:
        raise ValueError("anisotropy_normalization_degree must be between 0 and 1.")
    
    z_spacing, y_spacing, x_spacing = spacing
    
    if anisotropy_normalization_degree == 0:
        # No downsampling, return original volume as memmap
        temp_dir = tempfile.mkdtemp()
        original_path = os.path.join(temp_dir, 'original_volume.dat')
        downsampled_volume = np.memmap(original_path, dtype=volume.dtype, mode='w+', shape=volume.shape)
        downsampled_volume[:] = volume[:]
        downsampled_volume.flush()
        new_spacing = spacing
        print(f"No anisotropy normalization (degree = 0).")
        print(f"Volume shape: {volume.shape}")
        print(f"Spacing: {new_spacing}")
        return downsampled_volume, new_spacing, temp_dir
    
    # Calculate full isotropic downsampling factors (target = max spacing for downsampling)
    target_spacing = max(z_spacing, y_spacing, x_spacing)
    full_zoom_factors = [
        target_spacing / z_spacing, 
        target_spacing / y_spacing, 
        target_spacing / x_spacing  
    ]

    interpolated_zoom_factors = [
        1/(f + (1.0 - f) * (1 - anisotropy_normalization_degree)) for f in full_zoom_factors
    ]
    
    print(f"Spacing (z, y, x): {spacing}")
    print(f"Target spacing: {target_spacing}")
    print(f"Full zoom factors (z, y, x): {full_zoom_factors}")
    print(f"Interpolated zoom factors (degree={anisotropy_normalization_degree}, z, y, x): {interpolated_zoom_factors}")
    
    # Create a temporary file for the downsampled volume
    temp_dir = tempfile.mkdtemp()
    downsampled_path = os.path.join(temp_dir, 'downsampled_volume.dat')
    
    # Downsample the volume
    temp_volume = zoom(volume, interpolated_zoom_factors, order=1)  # Linear interpolation
    expected_shape = temp_volume.shape
    print(f"Expected downsampled shape: {expected_shape}")
    downsampled_volume = np.memmap(downsampled_path, dtype=volume.dtype, mode='w+', shape=expected_shape)
    if temp_volume.shape != expected_shape:
        print(f"Warning: Zoom output shape {temp_volume.shape} differs from expected {expected_shape}")
    downsampled_volume[:] = temp_volume[:]
    downsampled_volume.flush()  # Ensure data is written to disk
    
    # Interpolate new spacing between original and isotropic
    new_spacing = (
        z_spacing * (1 - anisotropy_normalization_degree) + target_spacing * anisotropy_normalization_degree,
        y_spacing * (1 - anisotropy_normalization_degree) + target_spacing * anisotropy_normalization_degree,
        x_spacing * (1 - anisotropy_normalization_degree) + target_spacing * anisotropy_normalization_degree
    )
    print(f"New spacing (adjusted for degree): {new_spacing}")
    print(f"Downsampled volume shape: {downsampled_volume.shape}")
    
    return downsampled_volume, new_spacing, temp_dir

def upsample_segmentation(segmentation, original_shape, downsampled_shape):
    """
    Upsample segmentation labels to match the original volume shape, using memmap if needed.
    """
    zoom_factors = [
        original_shape[0] / downsampled_shape[0],
        original_shape[1] / downsampled_shape[1],
        original_shape[2] / downsampled_shape[2]
    ]
    upsampled_segmentation = zoom(segmentation, zoom_factors, order=0)  # Nearest neighbor for labels
    return upsampled_segmentation

def detect_nuclei_centers_LoG(volume, scales, spacing=(1.0, 1.0, 1.0)):
    """
    Detect nuclei centers using Laplacian of Gaussian (LoG) filtering at multiple scales.
    
    Parameters:
    -----------
    volume : ndarray
        3D input volume
    scales : list
        List of scales for LoG filtering (in physical units)
    spacing : tuple
        Spacing (z, y, x) in physical units
        
    Returns:
    --------
    centers : ndarray
        Array of detected center coordinates (z, y, x)
    """
    from skimage.feature import blob_log
    
    # Convert scales from physical units to voxels
    voxel_scales = [scale / min(spacing) for scale in scales]
    
    # Detect blobs using LoG
    blobs = blob_log(volume, 
                     min_sigma=voxel_scales[0],
                     max_sigma=voxel_scales[-1],
                     num_sigma=len(scales),
                     threshold=.1)
    
    # Extract centers (z, y, x)
    centers = blobs[:, :3].astype(int)
    
    return centers

def analyze_shape_features(segmentation):
    """
    Analyze shape features of segmented objects to identify potential clusters.
    
    Parameters:
    -----------
    segmentation : ndarray
        Labeled segmentation volume
        
    Returns:
    --------
    cluster_candidates : list
        List of label IDs that are likely to be clusters
    """
    from skimage.measure import regionprops
    from scipy.ndimage import binary_closing, binary_dilation
    
    props = regionprops(segmentation)
    cluster_candidates = []
    
    for prop in props:
        # Extract region
        label = prop.label
        bbox = prop.bbox
        region = segmentation[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]] == label
        
        # Calculate shape metrics
        volume = prop.area
        convex_volume = np.sum(prop.convex_image)
        solidity = volume / convex_volume if convex_volume > 0 else 1.0
        
        # Calculate elongation using principal axes
        if hasattr(prop, 'inertia_tensor_eigvals'):
            eigvals = prop.inertia_tensor_eigvals
            if min(eigvals) > 0:
                elongation = np.sqrt(max(eigvals) / min(eigvals))
            else:
                elongation = 100  # Arbitrary high value for extremely elongated objects
        else:
            # Fallback if inertia tensor is not available
            axes = prop.axis_major_length, prop.axis_minor_length, prop.axis_minor_length
            if min(axes) > 0:
                elongation = max(axes) / min(axes)
            else:
                elongation = 100
        
        # Look for concavities using morphological operations
        closed = binary_closing(region)
        concavity_score = np.sum(closed) / volume if volume > 0 else 1.0
        
        # Calculate number of potential nuclei based on volume
        # Assuming average nucleus volume is around 500-1000 voxels (adjust based on your data)
        avg_nucleus_volume = 800
        estimated_count = max(1, round(volume / avg_nucleus_volume))
        
        # Identify potential clusters
        if ((solidity < 0.85) or  # Objects with significant concavities
            (elongation > 2.5) or  # Elongated objects (likely multiple nuclei)
            (concavity_score > 1.2 and estimated_count > 1) or  # Objects with concavities
            (estimated_count >= 2)):  # Objects large enough for multiple nuclei
            cluster_candidates.append(label)
    
    return cluster_candidates

def graph_cut_splitting(volume, segmentation, label, spacing=(1.0, 1.0, 1.0)):
    """
    Split a potential nuclear cluster using graph cuts.
    
    Parameters:
    -----------
    volume : ndarray
        Original intensity volume
    segmentation : ndarray
        Current segmentation
    label : int
        Label of the cluster to split
    spacing : tuple
        Spacing (z, y, x) in physical units
        
    Returns:
    --------
    split_mask : ndarray
        Binary mask with the split result
    """
    try:
        # Only import these if actually using graph cuts
        from skimage.segmentation import random_walker
        from skimage.filters import gaussian
    except ImportError:
        print("Warning: scikit-image's random_walker not available. Using fallback method.")
        return segmentation == label
    
    # Extract the region
    mask = segmentation == label
    if not np.any(mask):
        return mask
    
    # Get bounding box with small margin
    z, y, x = np.where(mask)
    margin = 2
    z_min, z_max = max(0, min(z) - margin), min(volume.shape[0], max(z) + margin + 1)
    y_min, y_max = max(0, min(y) - margin), min(volume.shape[1], max(y) + margin + 1)
    x_min, x_max = max(0, min(x) - margin), min(volume.shape[2], max(x) + margin + 1)
    
    # Extract subvolumes
    sub_vol = volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()
    sub_mask = mask[z_min:z_max, y_min:y_max, x_min:x_max].copy()
    
    # Gaussian smoothing to reduce noise
    sub_vol_smooth = gaussian(sub_vol, sigma=0.5)
    
    # Create gradient magnitude volume (higher at edges)
    from scipy.ndimage import gaussian_gradient_magnitude
    gradient_mag = gaussian_gradient_magnitude(sub_vol_smooth, sigma=1.0)
    
    # Use distance transform to find potential nuclear centers
    from scipy.ndimage import distance_transform_edt
    distance = distance_transform_edt(sub_mask, sampling=spacing)
    
    # Find local maxima in the distance transform (potential centers)
    from skimage.feature import peak_local_max
    # Get peak coordinates directly from peak_local_max
    coordinates = peak_local_max(
        distance, 
        min_distance=5,
        exclude_border=False,
    )
    
    # Create a mask from the coordinates
    local_maxi_mask = np.zeros_like(distance, dtype=bool)
    for coord in coordinates:
        local_maxi_mask[tuple(coord)] = True
    
    # If we couldn't find multiple maxima, use watershed
    if len(coordinates) < 2:
        markers = np.zeros_like(sub_mask, dtype=int)
        for i, coord in enumerate(coordinates, 1):
            markers[tuple(coord)] = i
        markers[~sub_mask] = 0
        result = watershed(-distance, markers, mask=sub_mask)
        
        # Return full-sized result
        full_result = np.zeros_like(mask, dtype=bool)
        full_result[z_min:z_max, y_min:y_max, x_min:x_max] = result > 0
        return full_result
    
    # Create markers for random walker (foreground = center, background = outside)
    markers = np.zeros_like(sub_mask, dtype=int)
    for i, coord in enumerate(coordinates, 1):
        markers[tuple(coord)] = i
    markers[~sub_mask] = -1  # Outside is definitely background
    
    # Use random walker for splitting
    # We'll use the gradient magnitude as the edge weights
    try:
        labels = random_walker(gradient_mag, markers, beta=10000, mode='bf', spacing=spacing)
        split_result = labels * sub_mask
    except Exception as e:
        print(f"Random walker failed: {e}. Using fallback method.")
        split_result = sub_mask
    
    # Return full-sized result
    full_result = np.zeros_like(mask, dtype=bool)
    full_result[z_min:z_max, y_min:y_max, x_min:x_max] = split_result > 0
    return full_result

def segment_nuclei_first_pass(volume,
                             smooth_sigma=[0.5, 1.0, 2.0],
                             min_distance=10,
                             min_size=100,
                             spacing=(1.0, 1.0, 1.0),
                             anisotropy_normalization_degree=1.0):
    """
    First pass nuclei segmentation using a watershed on an isotropically resampled volume.
    
    Parameters:
    -----------
    volume : ndarray
        3D input volume
    smooth_sigma : list
        Gaussian smoothing sigmas for multiscale analysis
    min_distance : float
        Minimum distance between peaks in physical units
    min_size : int
        Minimum object size in voxels
    spacing : tuple
        Original spacing (z, y, x) in physical units
    anisotropy_normalization_degree : float
        Degree of anisotropy normalization (0 = no normalization, 1 = full normalization)
        
    Returns:
    --------
    upsampled_first_pass : ndarray
        First pass segmentation resampled to original volume shape
    first_pass_params : dict
        Dictionary containing parameters from first pass for use in second pass
    """
    # Create a params dictionary to store parameters
    first_pass_params = {
        'anisotropy_normalization_degree': anisotropy_normalization_degree,
        'smooth_sigma': smooth_sigma,
        'min_distance': min_distance,
        'min_size': min_size,
        'spacing': spacing
    }
    print(f"\nStarting first pass nuclei segmentation with anisotropy normalization degree = {anisotropy_normalization_degree}...")
    
    # Downsample to isotropic spacing
    downsampled_volume, isotropic_spacing, temp_dir = downsample_for_isotropic(volume, spacing, anisotropy_normalization_degree)
    original_shape = volume.shape
    downsampled_shape = downsampled_volume.shape
    
    # Store these values for future use
    first_pass_params['isotropic_spacing'] = isotropic_spacing
    first_pass_params['downsampled_shape'] = downsampled_shape
    
    # Create memmaps for smoothed scales
    smoothed_paths = [os.path.join(temp_dir, f'smoothed_scale_{i}.dat') for i in range(len(smooth_sigma))]
    smoothed_scales = []
    for i, sigma in enumerate(smooth_sigma):
        smoothed = np.memmap(smoothed_paths[i], dtype=downsampled_volume.dtype, mode='w+', shape=downsampled_shape)
        temp_smoothed = gaussian_filter(downsampled_volume, sigma=sigma)
        smoothed[:] = temp_smoothed[:]
        smoothed.flush()
        smoothed_scales.append(smoothed)
    
    # First pass - catch all nuclei (using largest scale for robustness)
    print("First pass - detecting all nuclei...")
    threshold = threshold_otsu(smoothed_scales[-1])
    first_pass_params['threshold'] = threshold  # Store threshold for second pass
    binary = smoothed_scales[-1] > threshold
    
    first_pass, num_labels = label(binary)
    print(f"Found {num_labels} connected components")
    
    # Create memmap for first_pass and upsample
    first_pass_path = os.path.join(temp_dir, 'first_pass.dat')
    first_pass_memmap = np.memmap(first_pass_path, dtype=np.int32, mode='w+', shape=first_pass.shape)
    first_pass_memmap[:] = first_pass[:]
    first_pass_memmap.flush()
    
    upsampled_first_pass = upsample_segmentation(first_pass_memmap, original_shape, downsampled_shape)
    
    # Clean up memmaps
    for path in [first_pass_path] + smoothed_paths:
        if os.path.exists(path):
            os.remove(path)
    rmtree(temp_dir)
    
    # Return both the segmentation result and the parameters used
    return upsampled_first_pass, first_pass_params


def segment_nuclei_second_pass(volume, 
                              first_pass,
                              first_pass_params,
                              smooth_sigma=None,
                              min_distance=None,
                              min_size=None,
                              contrast_threshold_factor=1.5,
                              spacing=None,
                              anisotropy_normalization_degree=None,
                              use_advanced_splitting=True):
    """
    Second pass nuclei segmentation to refine the results from first pass.
    
    Parameters:
    -----------
    volume : ndarray
        3D input volume
    first_pass : ndarray
        Initial segmentation from first pass
    first_pass_params : dict
        Dictionary containing parameters from first pass
    smooth_sigma : list or None
        Gaussian smoothing sigmas for multiscale analysis. If None, use first_pass_params value.
    min_distance : float or None
        Minimum distance between peaks in physical units. If None, use first_pass_params value.
    min_size : int or None
        Minimum object size in voxels. If None, use first_pass_params value.
    contrast_threshold_factor : float
        Factor to increase threshold for second pass
    spacing : tuple or None
        Original spacing (z, y, x) in physical units. If None, use first_pass_params value.
    anisotropy_normalization_degree : float or None
        Degree of anisotropy normalization. If None, use first_pass_params value.
    use_advanced_splitting : bool
        Whether to use advanced splitting methods for persistent clusters
        
    Returns:
    --------
    upsampled_segmentation : ndarray
        Final segmentation resampled to original volume shape
    """
    # Verify we have first_pass_params for required calculations
    if first_pass_params is None:
        raise ValueError("Second pass requires first_pass_params dictionary from first pass.")
    
    # Use first_pass parameters if not specified
    if smooth_sigma is None:
        smooth_sigma = first_pass_params.get('smooth_sigma', [0.5, 1.0, 2.0])
    if min_distance is None:
        min_distance = first_pass_params.get('min_distance', 10)
    if min_size is None:
        min_size = first_pass_params.get('min_size', 100)
    if spacing is None:
        spacing = first_pass_params.get('spacing', (1.0, 1.0, 1.0))
    if anisotropy_normalization_degree is None:
        anisotropy_normalization_degree = first_pass_params.get('anisotropy_normalization_degree', 1.0)
    
    print(f"\nStarting second pass nuclei segmentation with anisotropy normalization degree = {anisotropy_normalization_degree}...")
    print(f"Using original first pass parameters when recalculating first pass elements.")
    
    # Downsample to isotropic spacing using the CURRENT pass's parameters
    print(f"Using anisotropy normalization degree = {anisotropy_normalization_degree} for current pass calculations")
    downsampled_volume, isotropic_spacing, temp_dir = downsample_for_isotropic(volume, spacing, anisotropy_normalization_degree)
    original_shape = volume.shape
    downsampled_shape = downsampled_volume.shape
    
    # Create memmaps for smoothed scales using CURRENT pass parameters
    smoothed_paths = [os.path.join(temp_dir, f'smoothed_scale_{i}.dat') for i in range(len(smooth_sigma))]
    smoothed_scales = []
    for i, sigma in enumerate(smooth_sigma):
        smoothed = np.memmap(smoothed_paths[i], dtype=downsampled_volume.dtype, mode='w+', shape=downsampled_shape)
        temp_smoothed = gaussian_filter(downsampled_volume, sigma=sigma)
        smoothed[:] = temp_smoothed[:]
        smoothed.flush()
        smoothed_scales.append(smoothed)
    
    # Second pass - we need to recreate pass 1 downsampled volume using pass 1 parameters for consistency
    # But ONLY for the operations that depend on first pass
    first_pass_anisotropy = first_pass_params.get('anisotropy_normalization_degree')
    if first_pass_anisotropy != anisotropy_normalization_degree:
        print(f"Recalculating first pass downsampled volume with original anisotropy = {first_pass_anisotropy}")
        first_pass_downsampled, first_pass_spacing, first_pass_temp_dir = downsample_for_isotropic(
            volume, spacing, first_pass_anisotropy
        )
        first_pass_downsampled_shape = first_pass_downsampled.shape
    else:
        # If anisotropy is the same, we can reuse current calculations
        first_pass_downsampled = downsampled_volume
        first_pass_spacing = isotropic_spacing
        first_pass_downsampled_shape = downsampled_shape
        first_pass_temp_dir = None
    
    # ENHANCEMENT: Use LoG for more robust peak detection with CURRENT pass parameters
    if use_advanced_splitting:
        print("Using Laplacian of Gaussian for enhanced seed detection...")
        log_scales = [1.0, 2.0, 3.0]  # Scales in physical units
        log_peaks = detect_nuclei_centers_LoG(downsampled_volume, log_scales, isotropic_spacing)
        print(f"LoG detector found {len(log_peaks)} potential nuclei centers")
    
    # Second pass - multiscale peak detection for splitting
    print("Second pass - identifying potential splits...")
    all_peaks = []
    
    # Get original threshold from first pass
    first_pass_threshold = first_pass_params.get('threshold')
    if first_pass_threshold is None:
        # If not available, recalculate using the first pass parameters
        print("Warning: First pass threshold not available, recalculating...")
        first_pass_smoothed = gaussian_filter(first_pass_downsampled, sigma=first_pass_params.get('smooth_sigma', [0.5, 1.0, 2.0])[-1])
        first_pass_threshold = threshold_otsu(first_pass_smoothed)
    
    # Use CURRENT pass parameters for the contrast threshold factor
    threshold = first_pass_threshold  # Use the first pass threshold as base
    
    for smoothed in smoothed_scales:  # But use current pass smoothed scales
        high_contrast_threshold = threshold * contrast_threshold_factor
        high_contrast_binary = smoothed > high_contrast_threshold
        
        # Use memmap for distance transform
        distance_path = os.path.join(temp_dir, 'distance_high_contrast.dat')
        distance_high_contrast = np.memmap(distance_path, dtype=np.float32, mode='w+', shape=smoothed.shape)
        temp_distance = distance_transform_edt(high_contrast_binary, sampling=isotropic_spacing)
        distance_high_contrast[:] = temp_distance[:]
        distance_high_contrast.flush()
        
        # Use CURRENT pass min_distance parameter
        min_distance_voxels = [min_distance / s for s in isotropic_spacing]
        
        peaks = peak_local_max(
            distance_high_contrast,
            min_distance=int(min(min_distance_voxels)),
            labels=high_contrast_binary
        )
        all_peaks.append(peaks)
    
    # Combine standard peaks
    combined_peaks = np.unique(np.vstack(all_peaks), axis=0).astype(int)
    
    # Add LoG peaks if applicable
    if use_advanced_splitting and len(log_peaks) > 0:
        combined_peaks = np.unique(np.vstack([combined_peaks, log_peaks]), axis=0).astype(int)
    
    print(f"Found {len(combined_peaks)} potential split points across all methods")
    
    # Create memmap for split_markers
    split_markers_path = os.path.join(temp_dir, 'split_markers.dat')
    split_markers = np.memmap(split_markers_path, dtype=np.int32, mode='w+', shape=downsampled_shape)
    split_markers[:] = 0  # Initialize to zeros
    split_markers.flush()
    
    # Now we need to downsample the first_pass segmentation to match the CURRENT downsampled shape
    # This part depends on the CURRENT pass anisotropy
    downsampled_first_pass_path = os.path.join(temp_dir, 'downsampled_first_pass.dat')
    downsampled_first_pass = np.memmap(downsampled_first_pass_path, dtype=np.int32, mode='w+', shape=downsampled_shape)
    zoom_factors = [s / o for s, o in zip(downsampled_shape, original_shape)]
    temp_downsampled_first_pass = zoom(first_pass, zoom_factors, order=0)
    downsampled_first_pass[:] = temp_downsampled_first_pass[:]
    downsampled_first_pass.flush()
    
    # Get original first pass labels
    original_labels = np.unique(downsampled_first_pass)
    original_labels = original_labels[original_labels != 0]
    
    # Create a map to track first pass label to its new split labels
    label_mappings = {}
    
    next_marker = 1
    for center in combined_peaks:
        if 0 <= center[0] < downsampled_shape[0] and 0 <= center[1] < downsampled_shape[1] and 0 <= center[2] < downsampled_shape[2]:
            if downsampled_first_pass[tuple(center)] > 0:
                center_yx = center[1:3]  # y and x (indices 1 and 2 for [z, y, x])
                too_close = False
                existing_indices = np.where(split_markers > 0)
                
                if len(existing_indices[0]) > 0:
                    existing_yx = np.array([existing_indices[1], existing_indices[2]]).T
                    center_yx_expanded = np.array(center_yx)
                    spacing_yx = np.array(isotropic_spacing[1:3])
                    diff_yx = (existing_yx - center_yx_expanded)
                    
                    if diff_yx.shape[0] > 0:
                        scaled_diff = diff_yx * spacing_yx
                        dist_xy = np.sqrt(np.sum(scaled_diff**2, axis=1))
                        if np.any(dist_xy < min_distance):  # Use CURRENT min_distance
                            too_close = True
                
                if not too_close:
                    # Store the original first pass label of this marker
                    original_label = downsampled_first_pass[tuple(center)]
                    if original_label not in label_mappings:
                        label_mappings[original_label] = []
                    label_mappings[original_label].append(next_marker)
                    
                    split_markers[tuple(center)] = next_marker
                    next_marker += 1
    split_markers.flush()
    
    # Watershed with isotropic distance transform
    print("Performing watershed to split merged objects...")
    first_pass_distance_path = os.path.join(temp_dir, 'first_pass_distance.dat')
    first_pass_distance = np.memmap(first_pass_distance_path, dtype=np.float32, mode='w+', shape=downsampled_shape)
    temp_distance = distance_transform_edt(downsampled_first_pass > 0, sampling=isotropic_spacing)
    first_pass_distance[:] = temp_distance[:]
    first_pass_distance.flush()
    
    mask = downsampled_first_pass > 0
    if mask.shape != split_markers.shape:
        raise ValueError(f"Shape mismatch: markers {split_markers.shape}, mask {mask.shape}")
    
    # Ensure split_markers is loaded as a regular array for watershed
    split_markers_array = np.array(split_markers)  # Load into memory temporarily for watershed
    
    # Process the watershed on each first_pass ROI separately to preserve boundaries
    final_segmentation = np.zeros_like(downsampled_first_pass)
    
    # Perform the watershed for each original ROI separately
    for original_label in original_labels:
        # Create mask for this specific ROI
        roi_mask = downsampled_first_pass == original_label
        
        # Get markers that are within this ROI
        roi_markers = split_markers_array.copy()
        roi_markers[~roi_mask] = 0
        
        # If no markers in this ROI, keep the original label
        if np.max(roi_markers) == 0:
            final_segmentation[roi_mask] = original_label
            continue
        
        # Perform watershed on this ROI
        roi_distance = first_pass_distance.copy()
        roi_distance[~roi_mask] = 0
        
        roi_segmentation = watershed(-roi_distance, roi_markers, mask=roi_mask)
        
        # Add the ROI segmentation to the final result
        final_segmentation[roi_mask] = roi_segmentation[roi_mask]
    
    # ENHANCEMENT: Third pass for handling persistent clusters
    if use_advanced_splitting:
        print("Third pass - analyzing and splitting persistent clusters...")
        
        # Analyze shape features to identify potential clusters
        cluster_candidates = analyze_shape_features(final_segmentation)
        print(f"Identified {len(cluster_candidates)} potential clusters for advanced splitting")
        
        # Process each cluster candidate
        for cluster_label in cluster_candidates:
            # Apply graph-cut based splitting to persistent clusters
            split_result = graph_cut_splitting(
                downsampled_volume, 
                final_segmentation, 
                cluster_label, 
                spacing=isotropic_spacing
            )
            
            # If splitting was successful (produced multiple objects)
            if np.sum(split_result) > 0:
                # Label the split result
                split_labeled, num_split = label(split_result)
                
                if num_split > 1:
                    print(f"Successfully split cluster (label {cluster_label}) into {num_split} objects")
                    
                    # Remove the original cluster label
                    cluster_mask = final_segmentation == cluster_label
                    final_segmentation[cluster_mask] = 0
                    
                    # Add the new split labels, maintaining original ROI boundary
                    max_label = np.max(final_segmentation)
                    for i in range(1, num_split + 1):
                        split_mask = split_labeled == i
                        final_segmentation[split_mask] = max_label + i
    
    # Remove small objects but preserve the original ROI boundaries
    print("Removing small objects while preserving original ROI boundaries...")
    
    # Create a map from original label to all its split labels
    split_to_original = {}
    for original_label in original_labels:
        # Get all voxels belonging to this original label
        original_mask = downsampled_first_pass == original_label
        
        # Get all unique split labels in this original ROI
        split_labels = np.unique(final_segmentation[original_mask])
        split_labels = split_labels[split_labels != 0]
        
        for split_label in split_labels:
            split_to_original[split_label] = original_label
    
    # Find small objects
    for split_label in np.unique(final_segmentation)[1:]:
        split_region = final_segmentation == split_label
        if np.sum(split_region) < min_size:  # Use CURRENT min_size
            # Get the original label this split belongs to
            original_label = split_to_original.get(split_label)
            if original_label is None:
                # If we can't find the mapping, just remove the object
                final_segmentation[split_region] = 0
                continue
            
            # Get other splits in the same original ROI
            original_region = downsampled_first_pass == original_label
            other_splits = np.unique(final_segmentation[original_region])
            other_splits = other_splits[(other_splits != 0) & (other_splits != split_label)]
            
            if len(other_splits) > 0:
                # Reassign to the nearest other split in the same original ROI
                # For simplicity, just use the first other split
                reassign_to = other_splits[0]
                final_segmentation[split_region] = reassign_to
            else:
                # If no other splits, keep the original label
                final_segmentation[split_region] = original_label
    
    # Relabel consecutively while preserving the mapping to original ROIs
    if final_segmentation.max() > 0:
        labels = np.unique(final_segmentation)
        labels = labels[labels != 0]
        new_segmentation = np.zeros_like(final_segmentation)
        
        # Create a mapping from new labels to original ROIs
        new_to_original = {}
        
        for i, lbl in enumerate(labels, 1):
            new_segmentation[final_segmentation == lbl] = i
            
            # Preserve the mapping to original ROIs
            if lbl in split_to_original:
                new_to_original[i] = split_to_original[lbl]
        
        final_segmentation = new_segmentation
    
    # Upsample segmentation to original volume shape
    upsampled_segmentation = upsample_segmentation(final_segmentation, original_shape, downsampled_shape)
    
    final_count = len(np.unique(upsampled_segmentation)) - 1
    print(f"\nSegmentation complete.")
    print(f"Final nuclei count: {final_count}")
    
    # Clean up memmaps and temp directories
    downsampled_path = os.path.join(temp_dir, 'downsampled_volume.dat')
    for path in [downsampled_path] + smoothed_paths + [distance_path, split_markers_path, 
                                                      downsampled_first_pass_path, first_pass_distance_path]:
        if os.path.exists(path):
            os.remove(path)
    rmtree(temp_dir)
    
    # Clean up first pass temp directory if we created one
    if first_pass_temp_dir and first_pass_anisotropy != anisotropy_normalization_degree:
        rmtree(first_pass_temp_dir)
    
    return upsampled_segmentation


def segment_nuclei(volume, 
                  first_pass=None,
                  first_pass_params=None,
                  smooth_sigma=[0.5, 1.0, 2.0],
                  min_distance=10,
                  min_size=100,
                  contrast_threshold_factor=1.5,
                  spacing=(1.0, 1.0, 1.0),
                  anisotropy_normalization_degree=1.0,
                  use_advanced_splitting=True):
    """
    Segment nuclei using a multiscale watershed on an isotropically resampled volume, using memmaps,
    with adjustable anisotropy normalization degree (0 to 1).
    
    This is a wrapper function that calls either segment_nuclei_first_pass or 
    segment_nuclei_second_pass based on the provided parameters.
    
    Parameters:
    -----------
    volume : ndarray
        3D input volume
    first_pass : ndarray or None
        Initial segmentation from first pass
    first_pass_params : dict or None
        Dictionary containing parameters from first pass to reuse when recalculating first pass elements
    smooth_sigma : list
        Gaussian smoothing sigmas for multiscale analysis
    min_distance : float
        Minimum distance between peaks in physical units
    min_size : int
        Minimum object size in voxels
    contrast_threshold_factor : float
        Factor to increase threshold for second pass
    spacing : tuple
        Original spacing (z, y, x) in physical units
    anisotropy_normalization_degree : float
        Degree of anisotropy normalization (0 = no normalization, 1 = full normalization)
    use_advanced_splitting : bool
        Whether to use advanced splitting methods for persistent clusters
        
    Returns:
    --------
    If first_pass is None:
        upsampled_first_pass : ndarray
            First pass segmentation
        first_pass_params : dict
            Parameters from first pass
    Else:
        upsampled_segmentation : ndarray
            Final segmentation after second pass
    """
    if first_pass is None:
        # First pass mode
        return segment_nuclei_first_pass(
            volume,
            smooth_sigma=smooth_sigma,
            min_distance=min_distance,
            min_size=min_size,
            spacing=spacing,
            anisotropy_normalization_degree=anisotropy_normalization_degree
        )
    else:
        # Second pass mode
        if first_pass_params is None:
            raise ValueError("Second pass requires first_pass_params dictionary from first pass.")
        
        return segment_nuclei_second_pass(
            volume, 
            first_pass,
            first_pass_params,
            smooth_sigma=smooth_sigma,
            min_distance=min_distance,
            min_size=min_size,
            contrast_threshold_factor=contrast_threshold_factor,
            spacing=spacing,
            anisotropy_normalization_degree=anisotropy_normalization_degree,
            use_advanced_splitting=use_advanced_splitting
        )