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
    sensitivity : float
        Sensitivity factor for thresholding (0-1)
        For otsu: higher = more sensitive (selects more pixels)
        For adaptive: higher = less sensitive (selects fewer pixels)
        For percentage: higher = less sensitive (selects fewer pixels)
        
    Returns:
    --------
    upsampled_first_pass : ndarray
        First pass segmentation resampled to original volume shape
    first_pass_params : dict
        Dictionary containing parameters from first pass for use in second pass
    """
    from scipy.ndimage import label
    
    # Ensure sensitivity is within expected range
    sensitivity = max(0.01, min(0.99, sensitivity))

    print(f"Initial memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    
    # Create a params dictionary to store parameters
    first_pass_params = {
        'anisotropy_normalization_degree': anisotropy_normalization_degree,
        'tubular_scales': tubular_scales,
        'smooth_sigma': smooth_sigma,
        'min_size': min_size,
        'spacing': spacing,
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
    print(f"Applying Otsu thresholding with sensitivity {sensitivity}...")
    

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


def split_merged_microglia(original_volume, segmented_mask, 
                                          min_intensity_variation=0.1,
                                          min_concavity_ratio=0.7,
                                          watershed_compactness=0.6,
                                          min_object_size=50,
                                          max_split_iterations=2,
                                          spacing=(1.0, 1.0, 1.0),
                                          chunk_size=None,
                                          max_ram_gb=4):
    """
    Memory-efficient function to split merged microglia cells based on intensity and shape information.
    Processes individual objects one at a time and uses temporary files for large arrays.
    
    Parameters:
    -----------
    original_volume : ndarray
        Original 3D intensity volume
    segmented_mask : ndarray
        Integer-labeled segmentation mask from first pass segmentation
    min_intensity_variation : float
        Minimum variation in intensity required to consider splitting a cell (0-1)
    min_concavity_ratio : float
        Minimum concavity ratio to consider splitting a cell (0-1)
    watershed_compactness : float
        Compactness parameter for watershed algorithm (0-1)
    min_object_size : int
        Minimum size (in voxels) for an object to be considered valid after splitting
    max_split_iterations : int
        Maximum number of times to attempt splitting a single object
    spacing : tuple
        Voxel spacing (z, y, x) in physical units
    chunk_size : tuple or None
        Size of chunks to process (z, y, x). If None, automatically determined.
    max_ram_gb : float
        Maximum RAM to use in GB
        
    Returns:
    --------
    refined_mask : ndarray
        Refined segmentation mask with split cells
    split_stats : dict
        Statistics about the splitting process
    """
    import numpy as np
    import os
    import tempfile
    from shutil import rmtree
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    from skimage.measure import label
    from tqdm import tqdm
    import gc
    import psutil
    import warnings
    
    # Suppress specific warnings that might occur during processing
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    # Function to report memory usage
    def get_memory_usage_mb():
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    print(f"Starting memory-efficient cell splitting process...")
    print(f"Initial memory usage: {get_memory_usage_mb():.2f} MB")
    
    # Determine the maximum cell label
    max_label = int(np.max(segmented_mask))
    print(f"Processing {max_label} potential objects...")
    
    # Statistics dictionary
    split_stats = {
        'original_cell_count': max_label,
        'cells_evaluated': 0,
        'cells_split': 0,
        'total_new_cells': 0,
        'splits_by_intensity': 0,
        'splits_by_shape': 0
    }
    
    # Create a memory-mapped array for the refined mask
    temp_dir = tempfile.mkdtemp()
    refined_mask_path = os.path.join(temp_dir, 'refined_mask.dat')
    refined_mask = np.memmap(refined_mask_path, dtype=np.int32, mode='w+', shape=segmented_mask.shape)
    
    # Copy original segmentation to memory-mapped array in chunks
    chunk_z = min(50, segmented_mask.shape[0])
    for z in range(0, segmented_mask.shape[0], chunk_z):
        end_z = min(z + chunk_z, segmented_mask.shape[0])
        refined_mask[z:end_z] = segmented_mask[z:end_z]
    refined_mask.flush()
    
    # Extract bounding boxes for all objects to process them individually
    print("Extracting object bounding boxes...")
    object_bbox_dict = {}
    
    # Process in z-chunks to save memory
    for z in tqdm(range(0, segmented_mask.shape[0], chunk_z)):
        end_z = min(z + chunk_z, segmented_mask.shape[0])
        chunk = segmented_mask[z:end_z]
        
        # Get unique labels in this chunk
        unique_labels = np.unique(chunk)
        unique_labels = unique_labels[unique_labels > 0]  # Skip background
        
        for label in unique_labels:
            # Extract coordinates for this label in the chunk
            mask = chunk == label
            if np.any(mask):
                z_indices, y_indices, x_indices = np.where(mask)
                
                # Adjust z coordinates
                z_indices += z
                
                # Update or initialize bounding box
                if label in object_bbox_dict:
                    # Update existing bounding box
                    bbox = object_bbox_dict[label]
                    bbox['z_min'] = min(bbox['z_min'], np.min(z_indices))
                    bbox['z_max'] = max(bbox['z_max'], np.max(z_indices))
                    bbox['y_min'] = min(bbox['y_min'], np.min(y_indices))
                    bbox['y_max'] = max(bbox['y_max'], np.max(y_indices))
                    bbox['x_min'] = min(bbox['x_min'], np.min(x_indices))
                    bbox['x_max'] = max(bbox['x_max'], np.max(x_indices))
                else:
                    # Initialize new bounding box
                    object_bbox_dict[label] = {
                        'z_min': np.min(z_indices),
                        'z_max': np.max(z_indices),
                        'y_min': np.min(y_indices),
                        'y_max': np.max(y_indices),
                        'x_min': np.min(x_indices),
                        'x_max': np.max(x_indices)
                    }
        
        # Clean up
        del chunk
        gc.collect()
    
    # Calculate size-limited concavity using distance transform instead of full convex hull
    def calculate_efficient_concavity(binary_obj, spacing):
        """Calculate concavity using distance transform method instead of convex hull."""
        # Calculate distance transform
        distance = ndi.distance_transform_edt(binary_obj, sampling=spacing)
        
        # Find the maximum distance (approximates the radius of the largest inscribed sphere)
        max_distance = np.max(distance)
        
        # Count voxels near the boundary (distance < threshold)
        boundary_threshold = max_distance * 0.2  # Adjust this parameter as needed
        boundary_voxels = np.sum((distance < boundary_threshold) & binary_obj)
        
        # Calculate ratio of boundary voxels to total voxels
        total_voxels = np.sum(binary_obj)
        boundary_ratio = boundary_voxels / max(total_voxels, 1)
        
        # High boundary ratio indicates more complex/concave shape
        # Scale to similar range as convex hull based concavity
        concavity_estimate = min(0.9, boundary_ratio * 1.5)
        
        return concavity_estimate
    
    # Calculate intensity variation in a memory-efficient way
    def calculate_intensity_variation(binary_obj, intensity_data):
        """Calculate intensity variation for an object."""
        # Extract only values where the binary object is True
        obj_values = intensity_data[binary_obj]
        
        # Calculate statistics
        if len(obj_values) > 0:
            intensity_std = np.std(obj_values)
            intensity_mean = np.mean(obj_values)
            return intensity_std / max(intensity_mean, 1e-6)
        else:
            return 0.0
    
    # Determine if an object should be split based on shape and intensity
    def should_split_object(obj_binary, obj_intensity, obj_size):
        """Determine if an object should be split based on shape and intensity."""
        # Check object size
        if obj_size < min_object_size * 2:  # Object must be at least 2x min size
            return False, None
        
        # Calculate intensity variation
        intensity_variation = calculate_intensity_variation(obj_binary, obj_intensity)
        
        # Calculate shape concavity (efficient method)
        concavity = calculate_efficient_concavity(obj_binary, spacing)
        
        # Create split criteria
        split_by_intensity = intensity_variation > min_intensity_variation
        split_by_shape = concavity > min_concavity_ratio
        
        # Determine split method
        split_method = None
        if split_by_intensity and split_by_shape:
            split_method = 'both'
        elif split_by_intensity:
            split_method = 'intensity'
        elif split_by_shape:
            split_method = 'shape'
            
        return (split_by_intensity or split_by_shape), split_method
    
    # Split function using efficient methods
    def split_object_efficient(obj_binary, obj_intensity, obj_label, method):
        """Split an object using watershed with memory efficiency in mind."""
        # Create distance map for watershed
        if method == 'shape' or method == 'both':
            # Distance transform based
            distance = ndi.distance_transform_edt(obj_binary, sampling=spacing)
        else:
            # Intensity based
            distance = obj_intensity.copy()
            if np.max(distance) > np.min(distance):
                distance = (distance - np.min(distance)) / (np.max(distance) - np.min(distance))
            distance = distance * obj_binary  # Mask to object
            distance = 1 - distance  # Invert for watershed
        
        # Find markers for watershed (seed points)
        # For shape-based splitting
        if method == 'shape':
            # More conservative for memory efficiency
            min_distance = max(2, int(np.cbrt(np.sum(obj_binary) / 75)))
            coordinates = peak_local_max(
                distance, 
                footprint=np.ones((3, 3, 3)),
                labels=obj_binary.astype(np.int32),  # Cast to int32 to avoid warning
                min_distance=min_distance
            )

        elif method == 'intensity':
            # For intensity-based
            min_distance = max(2, int(np.cbrt(np.sum(obj_binary) / 50)))
            coordinates = peak_local_max(
                -obj_intensity, 
                footprint=np.ones((3, 3, 3)),
                labels=obj_binary.astype(np.int32),  # Cast to int32 to avoid warning
                min_distance=min_distance
            )
            
        else:  # 'both'
            # Combine intensity and shape
            combined_surface = distance.copy()
            # Normalize intensity
            norm_intensity = obj_intensity.copy()
            if np.max(norm_intensity) > np.min(norm_intensity):
                norm_intensity = (norm_intensity - np.min(norm_intensity)) / (np.max(norm_intensity) - np.min(norm_intensity))
            combined_surface = combined_surface * (1 - norm_intensity)
            min_distance = max(2, int(np.cbrt(np.sum(obj_binary) / 60)))
            coordinates = peak_local_max(
                combined_surface, 
                footprint=np.ones((3, 3, 3)),
                labels=obj_binary.astype(np.int32),  # Cast to int32 to avoid warning
                min_distance=min_distance
            )
        # Create a mask from the coordinates
        local_maxi = np.zeros_like(distance, dtype=bool)
        for coord in coordinates:
            local_maxi[tuple(coord)] = True
        
        # If no or just one marker found, try a simpler approach
        if np.sum(local_maxi) <= 1:
            # Get object size to determine number of markers
            obj_size = np.sum(obj_binary)
            n_markers = min(4, max(2, int(obj_size / (min_object_size * 3))))
            
            # Use simple distance-based seeds
            if np.max(distance) > 0:
                # Create markers based on distance thresholds
                thresholds = np.linspace(0.5 * np.max(distance), 0.95 * np.max(distance), n_markers)
                local_maxi = np.zeros_like(obj_binary, dtype=bool)
                
                for threshold in thresholds:
                    # Create a marker at high distance values
                    marker = distance > threshold
                    # Label connected components
                    marker_labels, num_markers = ndi.label(marker)
                    
                    # If multiple markers, keep the largest one
                    if num_markers > 0:
                        sizes = np.bincount(marker_labels.ravel())
                        largest_marker = np.argmax(sizes[1:]) + 1 if len(sizes) > 1 else 1
                        centroid = ndi.center_of_mass(marker_labels == largest_marker)
                        
                        # Add only the center point to avoid too many markers
                        try:
                            z, y, x = int(centroid[0]), int(centroid[1]), int(centroid[2])
                            if 0 <= z < local_maxi.shape[0] and 0 <= y < local_maxi.shape[1] and 0 <= x < local_maxi.shape[2]:
                                local_maxi[z, y, x] = True
                        except:
                            # If centroid calculation fails, skip this marker
                            continue
            
            # If still no markers, force at least two simple markers using binary erosion
            if np.sum(local_maxi) <= 1:
                # Use erosion to create markers (simpler than k-means and uses less memory)
                from scipy.ndimage import binary_erosion
                
                # Create at least two markers using erosion at different levels
                eroded1 = binary_erosion(obj_binary, iterations=3)
                eroded2 = binary_erosion(obj_binary, iterations=6)
                
                # Find centroids of these eroded regions
                if np.any(eroded1):
                    centroid1 = ndi.center_of_mass(eroded1)
                    try:
                        z1, y1, x1 = int(centroid1[0]), int(centroid1[1]), int(centroid1[2])
                        if 0 <= z1 < local_maxi.shape[0] and 0 <= y1 < local_maxi.shape[1] and 0 <= x1 < local_maxi.shape[2]:
                            local_maxi[z1, y1, x1] = True
                    except:
                        pass
                
                if np.any(eroded2):
                    centroid2 = ndi.center_of_mass(eroded2)
                    try:
                        z2, y2, x2 = int(centroid2[0]), int(centroid2[1]), int(centroid2[2])
                        if 0 <= z2 < local_maxi.shape[0] and 0 <= y2 < local_maxi.shape[1] and 0 <= x2 < local_maxi.shape[2]:
                            local_maxi[z2, y2, x2] = True
                    except:
                        pass
        
        # Create markers for watershed
        markers, num_markers = ndi.label(local_maxi)
        
        # If still no adequate markers, return False
        if num_markers <= 1:
            return False, obj_label
        
        # Apply watershed
        if method == 'shape':
            # For shape-based, use negative distance
            ws_labels = watershed(-distance, markers, mask=obj_binary, compactness=watershed_compactness)
        else:
            # For intensity or combined, use the original intensity
            ws_labels = watershed(distance, markers, mask=obj_binary, compactness=watershed_compactness)
        
        # Check if watershed actually split the object
        unique_labels = np.unique(ws_labels)
        unique_labels = unique_labels[unique_labels > 0]  # Remove background
        
        if len(unique_labels) <= 1:  # No split occurred
            return False, obj_label
        
        # Check if all resulting objects meet minimum size
        valid_split = True
        for split_label in unique_labels:
            if np.sum(ws_labels == split_label) < min_object_size:
                valid_split = False
                break
        
        if not valid_split:
            return False, obj_label
        
        # Get the next available label for new objects
        next_label = np.max(refined_mask) + 1
        
        # Store the split results
        for i, split_label in enumerate(unique_labels):
            # Create mask for this split
            split_mask = ws_labels == split_label
            
            # Use original label for first split, new labels for others
            use_label = obj_label if i == 0 else next_label + i - 1
            
            # Update the result
            obj_binary[split_mask] = use_label
        
        # Return success and the highest label used
        return True, next_label + len(unique_labels) - 2
    
    # Process each object individually
    print("Processing objects...")
    next_label = max_label + 1
    
    # Sort objects by size (largest first) for better potential impact
    object_sizes = {}
    for label, bbox in object_bbox_dict.items():
        z_min, z_max = bbox['z_min'], bbox['z_max'] + 1
        y_min, y_max = bbox['y_min'], bbox['y_max'] + 1
        x_min, x_max = bbox['x_min'], bbox['x_max'] + 1
        
        # Get a small chunk containing this object
        chunk_mask = refined_mask[z_min:z_max, y_min:y_max, x_min:x_max].copy()
        object_sizes[label] = np.sum(chunk_mask == label)
    
    # Sort objects by size (process largest first)
    sorted_labels = sorted(object_sizes.keys(), key=lambda l: object_sizes[l], reverse=True)
    
    # Process objects one by one
    for obj_label in tqdm(sorted_labels):
        # Skip if this label no longer exists (could be merged/removed already)
        if obj_label not in object_bbox_dict:
            continue
        
        # Check if memory usage is close to limit
        current_mem_mb = get_memory_usage_mb()
        if current_mem_mb > max_ram_gb * 1000:
            print(f"WARNING: Memory usage ({current_mem_mb:.2f} MB) exceeds limit. Flushing caches...")
            # Force garbage collection and flush memory-mapped array
            refined_mask.flush()
            gc.collect()
        
        # Get object bounding box with padding
        bbox = object_bbox_dict[obj_label]
        pad = 5  # Add padding for watershed
        z_min = max(0, bbox['z_min'] - pad)
        z_max = min(refined_mask.shape[0], bbox['z_max'] + pad + 1)
        y_min = max(0, bbox['y_min'] - pad)
        y_max = min(refined_mask.shape[1], bbox['y_max'] + pad + 1)
        x_min = max(0, bbox['x_min'] - pad)
        x_max = min(refined_mask.shape[2], bbox['x_max'] + pad + 1)
        
        # Get the subvolume containing this object
        subvolume_mask = refined_mask[z_min:z_max, y_min:y_max, x_min:x_max].copy()
        object_mask = subvolume_mask == obj_label
        
        # Skip if empty (could happen if object was removed in previous operations)
        if not np.any(object_mask):
            continue
        
        # Get original intensity data for this region
        intensity_data = original_volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()
        
        split_stats['cells_evaluated'] += 1
        
        # Determine if this object should be split
        should_split, split_method = should_split_object(
            object_mask, 
            intensity_data,
            np.sum(object_mask)
        )
        
        if should_split:
            # Create a working copy of the object mask with label values
            # Initialize with original label
            labeled_mask = object_mask.astype(np.int32) * obj_label
            
            iteration = 0
            current_label = obj_label
            split_success = True
            
            # Allow multiple iterations of splitting for complex objects
            while split_success and iteration < max_split_iterations:
                # Only process objects with the current label
                current_obj_mask = labeled_mask == current_label
                
                if not np.any(current_obj_mask):
                    break
                
                # Try to split this object
                split_success, last_label = split_object_efficient(
                    current_obj_mask,
                    intensity_data,
                    current_label,
                    split_method
                )
                
                if split_success:
                    # Update the working copy with split results
                    for i in range(current_label, last_label + 1):
                        mask_i = current_obj_mask == i
                        if np.any(mask_i):
                            labeled_mask[mask_i] = i
                    
                    if iteration == 0:
                        # Count the initial split
                        split_stats['cells_split'] += 1
                        if split_method == 'intensity':
                            split_stats['splits_by_intensity'] += 1
                        elif split_method == 'shape':
                            split_stats['splits_by_shape'] += 1
                        else:  # 'both'
                            split_stats['splits_by_intensity'] += 1
                            split_stats['splits_by_shape'] += 1
                    
                    # Update stats with new cells created
                    new_cells = last_label - current_label
                    split_stats['total_new_cells'] += new_cells
                    
                    # Update next label for future processing
                    next_label = max(next_label, last_label + 1)
                    
                    # Update for next iteration
                    current_label = last_label
                    iteration += 1
                
            # Update the global refined mask with the splits
            if iteration > 0:  # Only if we actually split something
                # Get all labels in the split result
                result_labels = np.unique(labeled_mask)
                result_labels = result_labels[result_labels > 0]  # Skip background
                
                # Update the refined mask with each split
                for label in result_labels:
                    label_mask = labeled_mask == label
                    if np.any(label_mask):
                        # Create global coordinates mask
                        global_mask = np.zeros_like(refined_mask, dtype=bool)
                        global_mask[z_min:z_max, y_min:y_max, x_min:x_max] = label_mask
                        
                        # Update refined mask
                        refined_mask[global_mask] = label
                
                # Force flush
                refined_mask.flush()
        
        # Clean up
        del subvolume_mask, object_mask, intensity_data
        if 'labeled_mask' in locals():
            del labeled_mask
        gc.collect()
    
    print(f"Finalizing results...")
    print(f"Current memory usage: {get_memory_usage_mb():.2f} MB")
    
    # Convert memory mapped array to regular numpy array in chunks
    print("Converting result to numpy array...")
    result_mask = np.zeros_like(refined_mask, dtype=np.int32)
    
    chunk_z = min(50, refined_mask.shape[0])
    for z in tqdm(range(0, refined_mask.shape[0], chunk_z)):
        end_z = min(z + chunk_z, refined_mask.shape[0])
        result_mask[z:end_z] = refined_mask[z:end_z]
    
    # Clean up memory mapped array
    del refined_mask
    gc.collect()
    try:
        os.unlink(refined_mask_path)
        rmtree(temp_dir)
    except:
        pass
    
    # Ensure consecutive labeling if needed
    if np.max(result_mask) > max_label + split_stats['total_new_cells']:
        print("Ensuring consecutive label indices...")
        # Get all unique labels
        all_labels = []
        for z in range(0, result_mask.shape[0], chunk_z):
            end_z = min(z + chunk_z, result_mask.shape[0])
            chunk_labels = np.unique(result_mask[z:end_z])
            all_labels.extend(chunk_labels)
        
        unique_labels = np.unique(all_labels)
        unique_labels = unique_labels[unique_labels > 0]  # Skip background
        
        # Create mapping
        label_map = {old_label: i+1 for i, old_label in enumerate(unique_labels)}
        
        # Apply mapping in chunks
        temp_result_dir = tempfile.mkdtemp()
        temp_result_path = os.path.join(temp_result_dir, 'remapped_result.dat')
        remapped_result = np.memmap(temp_result_path, dtype=np.int32, mode='w+', shape=result_mask.shape)
        
        for z in tqdm(range(0, result_mask.shape[0], chunk_z)):
            end_z = min(z + chunk_z, result_mask.shape[0])
            chunk = result_mask[z:end_z].copy()
            
            # Apply mapping to each value
            for old_label, new_label in label_map.items():
                chunk[chunk == old_label] = new_label
            
            remapped_result[z:end_z] = chunk
            remapped_result.flush()
            
            # Clean up
            del chunk
            gc.collect()
        
        # Convert back to numpy array
        final_result = np.zeros_like(result_mask, dtype=np.int32)
        for z in range(0, remapped_result.shape[0], chunk_z):
            end_z = min(z + chunk_z, remapped_result.shape[0])
            final_result[z:end_z] = remapped_result[z:end_z]
        
        # Clean up
        del remapped_result, result_mask
        gc.collect()
        try:
            os.unlink(temp_result_path)
            rmtree(temp_result_dir)
        except:
            pass
    else:
        final_result = result_mask
    
    # Update statistics
    split_stats['final_cell_count'] = len(np.unique(final_result)) - 1  # Subtract 1 for background
    
    print(f"Cell splitting complete:")
    print(f"  Original cells: {split_stats['original_cell_count']}")
    print(f"  Cells evaluated: {split_stats['cells_evaluated']}")
    print(f"  Cells split: {split_stats['cells_split']}")
    print(f"  New cells created: {split_stats['total_new_cells']}")
    print(f"  Final cell count: {split_stats['final_cell_count']}")
    print(f"Final memory usage: {get_memory_usage_mb():.2f} MB")
    
    return final_result


# def merge_segmentations(first_pass, second_pass, min_overlap_ratio=0.5, min_overlap_voxels=20):
#     """
#     Merge first and second pass segmentations, using second pass to separate merged cells
#     while preserving all first pass voxels.
    
#     Parameters:
#     -----------
#     first_pass : ndarray
#         First pass segmentation (more sensitive, may have merged cells)
#     second_pass : ndarray
#         Second pass segmentation (less sensitive, better separated cells)
#     min_overlap_ratio : float
#         Minimum ratio of overlap between second pass object and first pass object
#         to consider them as the same cell
#     min_overlap_voxels : int
#         Minimum number of overlapping voxels required
        
#     Returns:
#     --------
#     merged_segmentation : ndarray
#         Final segmentation with separated cells and preserved processes
#     """
#     import numpy as np
#     from scipy.ndimage import label
#     from skimage.measure import regionprops
#     from collections import defaultdict
#     import tempfile
#     import os
#     import gc
#     import psutil
#     from tqdm import tqdm
    
#     print(f"Initial memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
#     print("Merging first and second pass segmentations...")
    
#     # Create a temporary directory for memory-mapped arrays
#     temp_dir = tempfile.mkdtemp(prefix='merge_segmentations_')
    
#     # Create output memory-mapped array
#     merged_path = os.path.join(temp_dir, 'merged_segmentation.dat')
#     merged_segmentation = np.memmap(merged_path, dtype=np.int32, mode='w+', shape=first_pass.shape)
#     merged_segmentation.fill(0)
    
#     # Find overlaps between first and second pass objects
#     print("Analyzing overlaps between segmentations...")
    
#     # Map from first pass label to list of second pass labels that overlap with it
#     first_to_second_map = defaultdict(list)
    
#     # Map from second pass label to list of first pass labels it overlaps with
#     second_to_first_map = defaultdict(list)
    
#     # Map storing overlap sizes
#     overlap_sizes = {}
    
#     # Get unique labels 
#     first_labels = np.unique(first_pass)
#     first_labels = first_labels[first_labels > 0]  # Remove background
    
#     second_labels = np.unique(second_pass)
#     second_labels = second_labels[second_labels > 0]  # Remove background
    
#     # Calculate sizes of all objects
#     first_sizes = {}
#     for label in first_labels:
#         first_sizes[label] = np.sum(first_pass == label)
    
#     second_sizes = {}
#     for label in second_labels:
#         second_sizes[label] = np.sum(second_pass == label)
    
#     # Process in chunks to save memory
#     chunk_size = min(50, first_pass.shape[0])
#     for i in tqdm(range(0, first_pass.shape[0], chunk_size), desc="Computing overlaps"):
#         end_idx = min(i + chunk_size, first_pass.shape[0])
        
#         # Get chunk from both segmentations
#         first_chunk = first_pass[i:end_idx].copy()
#         second_chunk = second_pass[i:end_idx].copy()
        
#         # For each first pass object in this chunk
#         for first_label in np.unique(first_chunk):
#             if first_label == 0:
#                 continue
                
#             first_mask = (first_chunk == first_label)
            
#             # Find second pass objects that overlap with this first pass object
#             overlapping_second_labels = np.unique(second_chunk[first_mask])
#             overlapping_second_labels = overlapping_second_labels[overlapping_second_labels > 0]
            
#             for second_label in overlapping_second_labels:
#                 # Calculate overlap
#                 second_mask = (second_chunk == second_label)
#                 overlap_mask = first_mask & second_mask
#                 overlap_size = np.sum(overlap_mask)
                
#                 # Only consider significant overlaps
#                 if overlap_size >= min_overlap_voxels:
#                     # Calculate overlap ratio relative to second pass object
#                     overlap_ratio = overlap_size / second_sizes[second_label]
                    
#                     if overlap_ratio >= min_overlap_ratio:
#                         # Add to maps
#                         if second_label not in first_to_second_map[first_label]:
#                             first_to_second_map[first_label].append(second_label)
                        
#                         if first_label not in second_to_first_map[second_label]:
#                             second_to_first_map[second_label].append(first_label)
                        
#                         # Store overlap size
#                         overlap_sizes[(first_label, second_label)] = overlap_size
        
#         # Clean up
#         del first_chunk, second_chunk
#         gc.collect()
    
#     # Create a mapping from first pass labels to new labels
#     first_to_new_map = {}
#     next_label = 1
    
#     # Process simple cases first: first pass objects that map to only one second pass object
#     print("Resolving one-to-one mappings...")
#     for first_label in first_labels:
#         overlapping_second = first_to_second_map[first_label]
        
#         if len(overlapping_second) == 1:
#             # One-to-one mapping
#             second_label = overlapping_second[0]
            
#             # Check if this second label maps to only this first label
#             if len(second_to_first_map[second_label]) == 1:
#                 # Simple case: direct mapping
#                 first_to_new_map[first_label] = next_label
#                 next_label += 1
    
#     print("Resolving split objects...")
#     # Now handle split objects (one first pass object maps to multiple second pass objects)
#     for first_label in first_labels:
#         if first_label in first_to_new_map:
#             continue  # Already handled
        
#         overlapping_second = first_to_second_map[first_label]
        
#         if len(overlapping_second) > 1:
#             # First pass object overlaps with multiple second pass objects
#             # Assign different labels to each part based on second pass
            
#             for second_label in overlapping_second:
#                 # Create a new label for this part
#                 first_to_new_map[(first_label, second_label)] = next_label
#                 next_label += 1
                
#         elif len(overlapping_second) == 0:
#             # First pass object doesn't overlap with any second pass object
#             # Keep it as a separate object
#             first_to_new_map[first_label] = next_label
#             next_label += 1
            
#         else:
#             # One-to-one mapping but the second pass object maps to multiple first pass objects
#             # We'll handle these in the next step
#             pass
    
#     print("Resolving merged objects...")
#     # Handle merged objects (one second pass object maps to multiple first pass objects)
#     for second_label in second_labels:
#         overlapping_first = second_to_first_map[second_label]
        
#         # Skip if there's only one or no first pass objects
#         if len(overlapping_first) <= 1:
#             continue
            
#         # Check if any of these first pass objects are already assigned
#         unassigned_first = [f for f in overlapping_first if f not in first_to_new_map]
        
#         # For each unassigned first pass object
#         for first_label in unassigned_first:
#             # Check if this first pass object only overlaps with this second pass object
#             if len(first_to_second_map[first_label]) == 1:
#                 # Assign a new label
#                 first_to_new_map[first_label] = next_label
#                 next_label += 1
#             else:
#                 # This first pass object overlaps with multiple second pass objects
#                 # It should have been handled in the previous step as a split object
#                 pass
    
#     # Final check for any objects that weren't assigned
#     print("Assigning any remaining objects...")
#     for first_label in first_labels:
#         if first_label not in first_to_new_map and all((first_label, s) not in first_to_new_map for s in first_to_second_map[first_label]):
#             # Assign a new label
#             first_to_new_map[first_label] = next_label
#             next_label += 1
    
#     # Now apply the mapping to create the final segmentation
#     print("Creating final segmentation...")
#     # Process in chunks to save memory
#     for i in tqdm(range(0, first_pass.shape[0], chunk_size), desc="Creating merged segmentation"):
#         end_idx = min(i + chunk_size, first_pass.shape[0])
        
#         # Get chunk from both segmentations
#         first_chunk = first_pass[i:end_idx].copy()
#         second_chunk = second_pass[i:end_idx].copy()
        
#         # Create mask for this chunk
#         chunk_mask = np.zeros_like(first_chunk, dtype=np.int32)
        
#         # Apply mappings for simple cases first
#         for first_label, new_label in first_to_new_map.items():
#             if isinstance(first_label, int):  # Simple mapping
#                 chunk_mask[first_chunk == first_label] = new_label
        
#         # Then handle split objects
#         for key, new_label in first_to_new_map.items():
#             if isinstance(key, tuple):  # Split object mapping
#                 first_label, second_label = key
#                 # Find voxels that belong to both objects
#                 mask = (first_chunk == first_label) & (second_chunk == second_label)
#                 chunk_mask[mask] = new_label
        
#         # Update output
#         merged_segmentation[i:end_idx] = chunk_mask
#         merged_segmentation.flush()
        
#         # Clean up
#         del first_chunk, second_chunk, chunk_mask
#         gc.collect()
    
#     # Fix any remaining unassigned voxels (voxels in first pass but not assigned in merged)
#     print("Fixing unassigned voxels...")
#     for i in tqdm(range(0, first_pass.shape[0], chunk_size), desc="Fixing unassigned voxels"):
#         end_idx = min(i + chunk_size, first_pass.shape[0])
        
#         # Get chunks
#         first_chunk = first_pass[i:end_idx].copy()
#         merged_chunk = merged_segmentation[i:end_idx].copy()
        
#         # Find voxels that have a first pass label but no merged label
#         unassigned_mask = (first_chunk > 0) & (merged_chunk == 0)
        
#         if np.any(unassigned_mask):
#             # For each unassigned voxel, assign it to the closest assigned voxel
#             # This is computationally expensive, so we'll use a simple approach
#             # For each first pass label that has unassigned voxels
#             for first_label in np.unique(first_chunk[unassigned_mask]):
#                 # Find the new label for this first pass label if it exists
#                 new_label = first_to_new_map.get(first_label, None)
                
#                 if new_label is not None:
#                     # Assign all unassigned voxels with this first pass label to the new label
#                     mask = (first_chunk == first_label) & (merged_chunk == 0)
#                     merged_chunk[mask] = new_label
#                 else:
#                     # This first pass label was split, find the most overlapping second pass label
#                     best_second_label = None
#                     best_overlap = 0
                    
#                     for second_label in first_to_second_map[first_label]:
#                         overlap = overlap_sizes.get((first_label, second_label), 0)
#                         if overlap > best_overlap:
#                             best_overlap = overlap
#                             best_second_label = second_label
                    
#                     if best_second_label is not None:
#                         # Use the new label for this combination
#                         new_label = first_to_new_map.get((first_label, best_second_label), next_label)
#                         if (first_label, best_second_label) not in first_to_new_map:
#                             first_to_new_map[(first_label, best_second_label)] = new_label
#                             next_label += 1
                            
#                         # Assign unassigned voxels to this new label
#                         mask = (first_chunk == first_label) & (merged_chunk == 0)
#                         merged_chunk[mask] = new_label
#                     else:
#                         # Create a new label
#                         new_label = next_label
#                         next_label += 1
#                         first_to_new_map[first_label] = new_label
                        
#                         # Assign unassigned voxels to this new label
#                         mask = (first_chunk == first_label) & (merged_chunk == 0)
#                         merged_chunk[mask] = new_label
        
#         # Update output
#         merged_segmentation[i:end_idx] = merged_chunk
#         merged_segmentation.flush()
        
#         # Clean up
#         del first_chunk, merged_chunk
#         gc.collect()
    
#     # Create a copy to return (not memory-mapped)
#     result = np.zeros_like(merged_segmentation)
#     for i in range(0, merged_segmentation.shape[0], chunk_size):
#         end_idx = min(i + chunk_size, merged_segmentation.shape[0])
#         result[i:end_idx] = merged_segmentation[i:end_idx]
    
#     # Clean up
#     del merged_segmentation
#     gc.collect()
#     os.unlink(merged_path)
#     os.rmdir(temp_dir)
    
#     print(f"Merged segmentation complete. Final object count: {np.max(result)}")
#     print(f"Final memory usage: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    
#     return result


def segment_microglia(volume, 
                     first_pass=None,
                     tubular_scales=[0.5, 1.0, 2.0, 3.0],
                     smooth_sigma=1.0,
                     min_size=50,
                     min_cell_body_size=200,
                     spacing=(1.0, 1.0, 1.0),
                     anisotropy_normalization_degree=1.0,
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
            sensitivity=sensitivity
        )
    else:
        second_pass = split_merged_microglia(volume, first_pass, 
                           min_intensity_variation=0.1,
                           min_concavity_ratio=0.7,
                           watershed_compactness=0.9,
                           min_object_size=min_size,
                           max_split_iterations=3,
                           spacing=spacing)
        return second_pass
        # if extract_features:
        #     # Extract cell bodies and processes if requested
        #     cell_bodies, processes = extract_cell_bodies_and_processes(
        #         first_pass, 
        #         volume, 
        #         min_body_size=min_cell_body_size
        #     )
            
        #     features = {
        #         'cell_bodies': cell_bodies,
        #         'processes': processes
        #     }
            
        #     return first_pass, features
        # else:
        #     return first_pass