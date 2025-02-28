import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter, label, distance_transform_edt, zoom
from skimage.feature import peak_local_max
from skimage.filters import frangi, threshold_otsu, threshold_local, sato
from skimage.morphology import binary_dilation, ball, skeletonize, binary_closing, remove_small_objects
import tempfile
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import psutil
from shutil import rmtree
import gc
import os
import tempfile
import gc
import numpy as np
from tqdm import tqdm
from skimage.filters import frangi, sato
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

from utils.nuclear_segmenter import downsample_for_isotropic, upsample_segmentation

def create_memmap(data=None, dtype=None, shape=None, prefix='temp', directory=None):
    """Helper function to create memory-mapped arrays"""
    if directory is None:
        directory = tempfile.mkdtemp()
    path = os.path.join(directory, f'{prefix}.dat')
    
    if data is not None:
        # Create from existing data
        shape = data.shape
        dtype = data.dtype
        result = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
        # Copy data in chunks to reduce peak memory usage
        chunk_size = min(100, shape[0])
        for i in range(0, shape[0], chunk_size):
            end = min(i + chunk_size, shape[0])
            result[i:end] = data[i:end]
        result.flush()
    else:
        # Create empty memmap
        result = np.memmap(path, dtype=dtype, mode='w+', shape=shape)
    
    return result, path, directory


def enhance_tubular_structures(volume, scales, spacing, black_ridges=False):
    """
    Enhance tubular/filamentous structures using absolute minimal memory footprint.
    This function processes one slice at a time, one scale at a time, with no parallelization.
    
    Parameters:
    -----------
    volume : ndarray
        3D input volume
    scales : list
        List of scales (sigmas) to use for enhancement
    spacing : tuple
        Spacing (z, y, x) in physical units
    black_ridges : bool
        If True, detect black ridges instead of white ones
    
    Returns:
    --------
    enhanced : ndarray
        Enhanced volume with tubular structures highlighted
    """
    # Create a temporary directory for memmaps
    temp_dir = tempfile.mkdtemp(prefix='tubular_enhance_minimal_')
    
    # Create output memory-mapped file initialized with zeros
    output_path = os.path.join(temp_dir, 'enhanced_volume.dat')
    enhanced = np.memmap(output_path, dtype=np.float32, mode='w+', shape=volume.shape)
    enhanced.fill(0)
    enhanced.flush()
    
    # Create 2D working space for a single slice
    z_size, y_size, x_size = volume.shape
    slice_shape = (y_size, x_size)
    
    # Create temporary files for single-slice processing
    slice_path = os.path.join(temp_dir, 'temp_slice.dat')
    frangi_path = os.path.join(temp_dir, 'temp_frangi.dat')
    sato_path = os.path.join(temp_dir, 'temp_sato.dat')
    result_path = os.path.join(temp_dir, 'temp_result.dat')
    
    print("Enhancing tubular structures with ultra-low memory usage...")
    print(f"Processing {z_size} slices with {len(scales)} scales")
    
    # Process each scale
    for scale_idx, scale in enumerate(scales):
        print(f"Processing scale {scale} ({scale_idx+1}/{len(scales)})...")
        
        # Adjust scale based on spacing
        scaled_sigma = scale
        if spacing != (1.0, 1.0, 1.0):
            min_spacing = min(spacing)
            scaled_sigma = np.asarray(scale) * min_spacing
        
        # Process each slice individually to minimize memory usage
        for z in tqdm(range(z_size), desc=f"Scale {scale}"):
            # Extract a single slice from the volume
            single_slice = volume[z].copy()
            
            # Apply filters to the single slice
            try:
                # Process with frangi filter
                frangi_result = frangi(single_slice, 
                                      sigmas=scaled_sigma,
                                      black_ridges=black_ridges, 
                                      mode='reflect')
                
                # Process with sato filter
                sato_result = sato(single_slice, 
                                  sigmas=scaled_sigma, 
                                  black_ridges=black_ridges, 
                                  mode='reflect')
                
                # Combine results (take maximum response)
                result = np.maximum(frangi_result, sato_result)
                
                # Update the result in the memory-mapped file
                current = enhanced[z].copy()
                enhanced[z] = np.maximum(current, result)
                enhanced.flush()
                
            except Exception as e:
                print(f"Error processing slice {z}: {e}")
                # Continue with next slice on error
            
            # Explicitly clean up to minimize memory
            del single_slice, frangi_result, sato_result, result, current
            gc.collect()
            
            # Short delay to allow OS to clean up memory
            time.sleep(0.05)
    
    # Load results from memmap
    # Create a copy that doesn't depend on the memmap - but do it slice by slice
    result = np.zeros_like(enhanced)
    for z in range(z_size):
        result[z] = enhanced[z]
        
    # Delete and close memmap before returning
    del enhanced
    gc.collect()
    
    return result, temp_dir

def connect_fragmented_processes(binary_mask, max_gap=3):
    """
    Connect fragmented microglia processes by applying morphological operations.
    Process in chunks to reduce memory usage.
    
    Parameters:
    -----------
    binary_mask : ndarray
        Binary segmentation mask
    max_gap : int
        Maximum gap to bridge in voxels
    
    Returns:
    --------
    connected_mask : ndarray
        Binary mask with connected processes
    """
    # Create a temporary directory and a memmap for the result
    temp_dir = tempfile.mkdtemp()
    result_path = os.path.join(temp_dir, 'connected_mask.dat')
    connected_mask = np.memmap(result_path, dtype=np.bool_, mode='w+', shape=binary_mask.shape)
    
    # Create the structural element
    struct_element = ball(max_gap)
    
    # Process in overlapping chunks to handle boundary effects
    overlap = max_gap * 2
    chunk_size = min(50, binary_mask.shape[0])
    
    for i in range(0, binary_mask.shape[0], chunk_size - overlap):
        # Handle boundaries
        start_idx = max(0, i)
        end_idx = min(i + chunk_size, binary_mask.shape[0])
        
        # Extract chunk with borders if possible
        chunk_start = max(0, start_idx - overlap)
        chunk_end = min(binary_mask.shape[0], end_idx + overlap)
        chunk = binary_mask[chunk_start:chunk_end].copy()
        
        # Apply closing
        closed_chunk = binary_closing(chunk, struct_element)
        
        # Determine where to save in the output (remove the borders)
        save_start = start_idx - chunk_start
        save_end = save_start + (end_idx - start_idx)
        
        # Save to output, avoiding overlap regions for chunks after the first
        if i == 0:
            connected_mask[start_idx:end_idx] = closed_chunk[save_start:save_end]
        else:
            connected_mask[start_idx:end_idx] = closed_chunk[save_start:save_end]
        
        # Clean up
        del chunk, closed_chunk
        gc.collect()
    
    connected_mask.flush()
    return connected_mask, temp_dir

def segment_microglia_first_pass(volume,
                               tubular_scales=[0.5, 1.0, 2.0, 3.0],
                               smooth_sigma=1.0,
                               min_size=50,  # Smaller than nuclei to capture processes
                               spacing=(1.0, 1.0, 1.0),
                               anisotropy_normalization_degree=1.0,
                               threshold_method='adaptive',
                               sensitivity=0.8):  # Higher sensitivity to capture dim processes
    """
    First pass microglia segmentation with focus on capturing processes.
    Memory-optimized version that processes data in chunks.
    
    Parameters:
    -----------
    volume : ndarray
        3D input volume
    tubular_scales : list
        Scales for tubular structure enhancement
    smooth_sigma : float or list
        Gaussian smoothing sigma(s)
    min_size : int
        Minimum object size in voxels (smaller than for nuclei)
    spacing : tuple
        Original spacing (z, y, x) in physical units
    anisotropy_normalization_degree : float
        Degree of anisotropy normalization (0 to 1)
    threshold_method : str
        'otsu', 'adaptive', or 'percentage'
    sensitivity : float
        Sensitivity factor for thresholding (0-1, higher = more sensitive)
        
    Returns:
    --------
    upsampled_first_pass : ndarray
        First pass segmentation resampled to original volume shape
    first_pass_params : dict
        Dictionary containing parameters from first pass for use in second pass
    """
    from scipy.ndimage import label
    # Track memory usage
    print(f"Initial memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    
    # Create a params dictionary to store parameters
    first_pass_params = {
        'anisotropy_normalization_degree': anisotropy_normalization_degree,
        'tubular_scales': tubular_scales,
        'smooth_sigma': smooth_sigma,
        'min_size': min_size,
        'spacing': spacing,
        'threshold_method': threshold_method,
        'sensitivity': sensitivity
    }
    print(f"\nStarting first pass microglia segmentation with anisotropy normalization degree = {anisotropy_normalization_degree}...")
    
    # Downsample to isotropic spacing
    downsampled_volume, isotropic_spacing, downsample_temp_dir = downsample_for_isotropic(volume, spacing, anisotropy_normalization_degree)
    original_shape = volume.shape
    downsampled_shape = downsampled_volume.shape
    
    # Store these values for future use
    first_pass_params['isotropic_spacing'] = isotropic_spacing
    first_pass_params['downsampled_shape'] = downsampled_shape
    
    # Smooth the volume to reduce noise
    print("Smoothing volume...")
    if isinstance(smooth_sigma, (list, tuple)):
        # Use the smallest sigma to preserve fine details
        smooth_sigma_value = smooth_sigma[0]
    else:
        smooth_sigma_value = smooth_sigma
    
    # Smooth in chunks to save memory
    smoothed_temp_dir = tempfile.mkdtemp()
    smoothed_path = os.path.join(smoothed_temp_dir, 'smoothed_volume.dat')
    smoothed = np.memmap(smoothed_path, dtype=downsampled_volume.dtype, mode='w+', shape=downsampled_shape)
    
    chunk_size = min(50, downsampled_volume.shape[0])
    for i in range(0, downsampled_volume.shape[0], chunk_size):
        end_idx = min(i + chunk_size, downsampled_volume.shape[0])
        chunk = downsampled_volume[i:end_idx].copy()
        
        # Apply smoothing
        smoothed_chunk = gaussian_filter(chunk, sigma=smooth_sigma_value)
        smoothed[i:end_idx] = smoothed_chunk
        
        # Clean up
        del chunk, smoothed_chunk
        gc.collect()
    
    smoothed.flush()
    
    # Enhance tubular structures (microglia processes)
    # Adjust scales for the isotropic spacing
    scaled_tubular_scales = [[scale/s for s in isotropic_spacing] for scale in tubular_scales]
    enhanced, enhanced_temp_dir = enhance_tubular_structures(smoothed, scaled_tubular_scales, isotropic_spacing)
    
    # Free memory by releasing smoothed volume
    del smoothed
    gc.collect()
    os.unlink(smoothed_path)
    rmtree(smoothed_temp_dir)
    
    # Create a temporary directory and a memmap for binary result
    binary_temp_dir = tempfile.mkdtemp()
    binary_path = os.path.join(binary_temp_dir, 'binary.dat')
    binary = np.memmap(binary_path, dtype=np.bool_, mode='w+', shape=enhanced.shape)
    
    # Thresholding based on chosen method - process in chunks
    print(f"Applying {threshold_method} thresholding with sensitivity {sensitivity}...")
    
    if threshold_method == 'otsu':
        # For Otsu, we need to compute the global threshold first
        # Sample a subset of the data to calculate threshold (to save memory)
        sample_size = min(1000000, enhanced.size)
        indices = np.random.choice(enhanced.size, sample_size, replace=False)
        flat_indices = np.unravel_index(indices, enhanced.shape)
        samples = enhanced[flat_indices]
        threshold = threshold_otsu(samples)
        adjusted_threshold = threshold * (2 - sensitivity)
        
        # Apply threshold in chunks
        chunk_size = min(50, enhanced.shape[0])
        for i in tqdm(range(0, enhanced.shape[0], chunk_size)):
            end_idx = min(i + chunk_size, enhanced.shape[0])
            binary[i:end_idx] = enhanced[i:end_idx] > adjusted_threshold
            
    elif threshold_method == 'adaptive':
        # Local adaptive thresholding
        block_size = max(3, int(10 * min(isotropic_spacing)))
        if block_size % 2 == 0:
            block_size += 1  # Ensure odd block size
        
        sensitivity_offset = (1 - sensitivity) * 0.1 * np.mean(enhanced)
        
        # Process in chunks with overlap to handle block effects
        overlap = block_size
        chunk_size = min(50, enhanced.shape[0] - overlap)
        
        for i in tqdm(range(0, enhanced.shape[0], chunk_size)):
            # Determine chunk boundaries with overlap
            start_idx = i
            end_idx = min(i + chunk_size + overlap, enhanced.shape[0])
            
            # Get chunk
            chunk = enhanced[start_idx:end_idx].copy()
            
            # Apply local thresholding
            local_thresh = threshold_local(chunk, block_size=block_size)
            chunk_binary = chunk > (local_thresh - sensitivity_offset)
            
            # Save only the non-overlapping portion (except for the last chunk)
            save_end = min(chunk_size, chunk.shape[0]) if end_idx < enhanced.shape[0] else chunk.shape[0]
            binary[start_idx:start_idx + save_end] = chunk_binary[:save_end]
            
            # Clean up
            del chunk, local_thresh, chunk_binary
            gc.collect()
            
    else:  # percentage
        # Calculate threshold based on percentile
        sample_size = min(1000000, enhanced.size)
        indices = np.random.choice(enhanced.size, sample_size, replace=False)
        flat_indices = np.unravel_index(indices, enhanced.shape)
        samples = enhanced[flat_indices]
        sorted_samples = np.sort(samples)
        idx = int((1 - sensitivity) * len(sorted_samples))
        threshold = sorted_samples[idx]
        
        # Apply threshold in chunks
        chunk_size = min(50, enhanced.shape[0])
        for i in range(0, enhanced.shape[0], chunk_size):
            end_idx = min(i + chunk_size, enhanced.shape[0])
            binary[i:end_idx] = enhanced[i:end_idx] > threshold
    
    binary.flush()
    
    # Free memory by releasing enhanced volume
    del enhanced
    gc.collect()
    rmtree(enhanced_temp_dir)
    
    # Connect fragmented processes
    print("Connecting fragmented processes...")
    max_gap = max(1, int(3 / min(isotropic_spacing)))  # Scale gap by spacing
    connected_binary, connected_temp_dir = connect_fragmented_processes(binary, max_gap=max_gap)
    
    # Free memory
    del binary
    gc.collect()
    os.unlink(binary_path)
    rmtree(binary_temp_dir)
    
    # Remove small objects - process in chunks
    print(f"Removing objects smaller than {min_size} voxels...")
    cleaned_binary_dir = tempfile.mkdtemp()
    cleaned_binary_path = os.path.join(cleaned_binary_dir, 'cleaned_binary.dat')
    cleaned_binary = np.memmap(cleaned_binary_path, dtype=np.bool_, mode='w+', shape=connected_binary.shape)
    
    # This is tricky to do in chunks because object size can cross chunk boundaries
    # We'll process the entire volume but use a memory-mapped output
    cleaned_binary_temp = remove_small_objects(connected_binary, min_size=min_size)
    cleaned_binary[:] = cleaned_binary_temp[:]
    cleaned_binary.flush()
    
    # Free memory
    del connected_binary, cleaned_binary_temp
    gc.collect()
    rmtree(connected_temp_dir)
    
    # Define a helper function to find the root label in the equivalence tree
    def find_root(equivalences, label):
        root = label
        # Follow chain of equivalences to find the root
        while root in equivalences:
            root = equivalences[root]
        
        # Path compression - update all nodes in the path to point to the root
        current = label
        while current in equivalences and equivalences[current] != root:
            next_node = equivalences[current]
            equivalences[current] = root
            current = next_node
            
        return root
    
    # Label connected components - this is memory intensive so we'll chunk it
    first_pass_dir = tempfile.mkdtemp()
    first_pass_path = os.path.join(first_pass_dir, 'first_pass.dat')
    first_pass = np.memmap(first_pass_path, dtype=np.int32, mode='w+', shape=cleaned_binary.shape)
    
    # Process in chunks with overlap
    overlap = 15  # Larger overlap to better handle crossing objects
    chunk_size = min(50, cleaned_binary.shape[0])
    max_label = 0
    
    # Dictionary to track label equivalences
    label_equivalences = {}
    
    print("Labeling connected components with improved stitching algorithm...")
    
    # First pass - label each chunk and build equivalence table
    for i in tqdm(range(0, cleaned_binary.shape[0], chunk_size - overlap), desc="First-pass labeling"):
        start_idx = max(0, i)
        end_idx = min(i + chunk_size, cleaned_binary.shape[0])
        
        # Skip if chunk is too small
        if end_idx - start_idx < overlap + 1:
            continue
            
        # Get chunk
        chunk = cleaned_binary[start_idx:end_idx].copy()
        
        # Label the chunk
        chunk_labels, num_labels = label(chunk)
        
        # Shift labels to avoid conflicts
        if num_labels > 0 and max_label > 0:
            chunk_labels[chunk_labels > 0] += max_label
        
        # Update max label
        if num_labels > 0:
            max_label += num_labels
        
        # Store in output
        first_pass[start_idx:end_idx] = chunk_labels
        
        # If not the first chunk, find equivalences in the overlap region
        if i > 0:
            overlap_start = start_idx
            overlap_end = min(start_idx + overlap, end_idx)
            
            # We need to check connectivity between current chunk and previous chunk
            # For each slice in the overlap region
            for z_offset in range(min(overlap, overlap_end - overlap_start)):
                curr_z = overlap_start + z_offset
                
                # Skip if we're at the boundary
                if curr_z <= 0 or curr_z >= cleaned_binary.shape[0]:
                    continue
                
                # Get current and previous slices
                curr_slice = cleaned_binary[curr_z]
                prev_slice = cleaned_binary[curr_z - 1]
                
                # Find where objects touch between slices
                touching_mask = np.logical_and(curr_slice, prev_slice)
                
                if np.any(touching_mask):
                    # Get labels for touching objects
                    prev_labels = first_pass[curr_z - 1][touching_mask]
                    curr_labels = first_pass[curr_z][touching_mask]
                    
                    # Create equivalence pairs
                    for j in range(len(prev_labels)):
                        prev_label = prev_labels[j]
                        curr_label = curr_labels[j]
                        
                        if prev_label == 0 or curr_label == 0:
                            continue  # Skip background
                            
                        # Find roots for both labels
                        root_prev = find_root(label_equivalences, prev_label)
                        root_curr = find_root(label_equivalences, curr_label)
                        
                        # Create equivalence if different
                        if root_prev != root_curr:
                            # Always use the smaller label as the root
                            if root_prev < root_curr:
                                label_equivalences[root_curr] = root_prev
                            else:
                                label_equivalences[root_prev] = root_curr
        
        # Clean up
        del chunk, chunk_labels
        gc.collect()
    
    # Flatten the equivalence tree
    print("Resolving label equivalences...")
    flattened_equivalences = {}
    for label in tqdm(np.unique(first_pass[first_pass > 0]), desc="Flattening equivalence tree"):
        flattened_equivalences[label] = find_root(label_equivalences, label)
    
    # Apply equivalences - process in chunks to save memory
    print("Applying equivalences to create consistent labeling...")
    for i in tqdm(range(0, first_pass.shape[0], chunk_size), desc="Second-pass relabeling"):
        end_idx = min(i + chunk_size, first_pass.shape[0])
        chunk = first_pass[i:end_idx].copy()
        
        # Apply mapping to all non-zero labels
        mask = chunk > 0
        if np.any(mask):
            # Vectorized mapping using masked array
            unique_labels = np.unique(chunk[mask])
            for label in unique_labels:
                if label in flattened_equivalences:
                    chunk[chunk == label] = flattened_equivalences[label]
        
        # Update first_pass
        first_pass[i:end_idx] = chunk
        first_pass.flush()
        
        # Clean up
        del chunk
        gc.collect()
    
    # Relabel to get consecutive labels
    print("Relabeling for consecutive indices...")
    # Process in chunks to minimize memory use
    max_label = 0
    label_map = {}
    
    # First find all unique labels
    for i in tqdm(range(0, first_pass.shape[0], chunk_size), desc="Finding unique labels"):
        end_idx = min(i + chunk_size, first_pass.shape[0])
        chunk = first_pass[i:end_idx]
        for label in np.unique(chunk):
            if label > 0 and label not in label_map:
                max_label += 1
                label_map[label] = max_label
    
    # Then apply the mapping
    for i in tqdm(range(0, first_pass.shape[0], chunk_size), desc="Applying new labels"):
        end_idx = min(i + chunk_size, first_pass.shape[0])
        chunk = first_pass[i:end_idx].copy()
        
        # Apply new consecutive labels
        for old_label, new_label in label_map.items():
            chunk[chunk == old_label] = new_label
        
        # Update memory-mapped array
        first_pass[i:end_idx] = chunk
        first_pass.flush()
        
        # Clean up
        del chunk
        gc.collect()
    
    print(f"Found {max_label} connected components after stitching")
    
    # Free memory
    del cleaned_binary
    gc.collect()
    os.unlink(cleaned_binary_path)
    rmtree(cleaned_binary_dir)
    
    # Upsample to original size
    print("Upsampling segmentation to original size...")
    upsampled_first_pass = upsample_segmentation(first_pass, original_shape, downsampled_shape)
    
    # Clean up memmaps
    del first_pass, downsampled_volume
    gc.collect()
    os.unlink(first_pass_path)
    rmtree(first_pass_dir)
    rmtree(downsample_temp_dir)
    
    # Report final memory usage
    print(f"Final memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    
    return upsampled_first_pass, first_pass_params

def extract_cell_bodies_and_processes(segmentation, volume, min_body_size=200):
    """
    Separate microglia cell bodies from processes based on intensity and size.
    
    Parameters:
    -----------
    segmentation : ndarray
        Labeled segmentation volume
    volume : ndarray
        Original intensity volume
    min_body_size : int
        Minimum size for a cell body in voxels
    
    Returns:
    --------
    cell_bodies : ndarray
        Binary mask of cell bodies
    processes : ndarray
        Binary mask of processes
    """
    from skimage.measure import regionprops
    from scipy.ndimage import label
    
    # Create output masks
    cell_bodies = np.zeros_like(segmentation, dtype=bool)
    processes = np.zeros_like(segmentation, dtype=bool)
    
    # For each segmented object
    for region in regionprops(segmentation, intensity_image=volume):
        label_id = region.label
        region_mask = segmentation == label_id
        
        # Get intensity and shape features
        mean_intensity = region.mean_intensity
        area = region.area
        
        # Create distance map from border (higher in center)
        from scipy.ndimage import distance_transform_edt
        distance_map = distance_transform_edt(region_mask)
        
        # Create a mask for the core of the region (potential cell body)
        # Higher threshold for larger regions
        body_threshold = max(0.5, min(0.7, 0.3 + 0.4 * (area / 5000)))
        max_distance = np.max(distance_map)
        if max_distance > 0:
            body_candidate = distance_map > (body_threshold * max_distance)
            
            # Only keep if large enough
            from skimage.measure import label
            labeled_bodies, num_bodies = label(body_candidate)
            
            for body_id in range(1, num_bodies + 1):
                body_mask = labeled_bodies == body_id
                if np.sum(body_mask) >= min_body_size:
                    # This is a cell body
                    cell_bodies[body_mask] = True
        
        # The rest are processes
        process_mask = region_mask & ~cell_bodies
        processes[process_mask] = True
    
    return cell_bodies, processes

def segment_microglia(volume, 
                     first_pass=None,
                     first_pass_params=None,
                     tubular_scales=[0.5, 1.0, 2.0, 3.0],
                     smooth_sigma=1.0,
                     min_size=50,
                     min_cell_body_size=200,
                     spacing=(1.0, 1.0, 1.0),
                     anisotropy_normalization_degree=1.0,
                     threshold_method='adaptive',
                     sensitivity=0.8,
                     extract_features=False):
    """
    Segment microglia using tubular enhancement and specialized processing for ramified cells.
    
    This is a wrapper function that calls either segment_microglia_first_pass or 
    segment_microglia_second_pass based on the provided parameters.
    
    Parameters:
    -----------
    volume : ndarray
        3D input volume
    first_pass : ndarray or None
        Initial segmentation from first pass
    first_pass_params : dict or None
        Dictionary containing parameters from first pass
    tubular_scales : list
        Scales for tubular structure enhancement
    smooth_sigma : float or list
        Gaussian smoothing sigma(s)
    min_size : int
        Minimum object size in voxels
    min_cell_body_size : int
        Minimum size for a cell body in voxels
    spacing : tuple
        Original spacing (z, y, x) in physical units
    anisotropy_normalization_degree : float
        Degree of anisotropy normalization (0 to 1)
    threshold_method : str
        'otsu', 'adaptive', or 'percentage'
    sensitivity : float
        Sensitivity factor for thresholding (0-1)
    extract_features : bool
        Whether to extract cell bodies and process features
        
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
        features : dict (optional)
            Dictionary with cell bodies and processes masks if extract_features=True
    """
    if first_pass is None:
        # First pass mode
        return segment_microglia_first_pass(
            volume,
            tubular_scales=tubular_scales,
            smooth_sigma=smooth_sigma,
            min_size=min_size,
            spacing=spacing,
            anisotropy_normalization_degree=anisotropy_normalization_degree,
            threshold_method=threshold_method,
            sensitivity=sensitivity
        )
    else:
        # Second pass mode - we'll implement this in the next step
        # For now, just return a placeholder and note
        print("Second pass segmentation function will be implemented next.")
        
        if extract_features:
            # Extract cell bodies and processes if requested
            cell_bodies, processes = extract_cell_bodies_and_processes(
                first_pass, 
                volume, 
                min_body_size=min_cell_body_size
            )
            
            features = {
                'cell_bodies': cell_bodies,
                'processes': processes
            }
            
            return first_pass, features
        else:
            return first_pass