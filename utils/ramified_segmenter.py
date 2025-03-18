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
import skimage.measure as measure
from skimage.measure import regionprops
from skimage.measure import regionprops_table
seed = 42
np.random.seed(seed)         # For NumPy

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