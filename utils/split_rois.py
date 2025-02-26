import numpy as np
from sklearn.mixture import GaussianMixture
from scipy import ndimage, stats
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, shared_memory
import time
import psutil
import os

def get_cluster_batches(labeled_cells, batch_size=50, min_volume=100):
    """Same as before"""
    unique_labels = np.unique(labeled_cells)
    unique_labels = unique_labels[unique_labels > 0]
    
    valid_labels = []
    for label in unique_labels:
        if np.sum(labeled_cells == label) >= min_volume:
            valid_labels.append(label)
    
    batches = [valid_labels[i:i+batch_size] for i in range(0, len(valid_labels), batch_size)]
    return batches

def estimate_components_statistical(features, max_components=5, split_threshold=0.7):
    """
    Estimate optimal number of components using statistical criteria.
    
    Parameters:
    -----------
    features : np.ndarray
        Feature matrix (coordinates and intensity)
    max_components : int
        Maximum number of components to consider
    split_threshold : float
        Threshold for BIC improvement ratio to accept split (0-1)
        Higher values make splitting less likely
    
    Returns:
    --------
    int
        Optimal number of components
    """
    n_samples = features.shape[0]
    
    # If very few samples, don't split
    if n_samples < 200:  # Adjustable threshold
        return 1
    
    # Fit single Gaussian
    gmm_single = GaussianMixture(
        n_components=1,
        covariance_type='full',
        random_state=42,
        max_iter=200,
        reg_covar=1e-3
    )
    gmm_single.fit(features)
    bic_single = gmm_single.bic(features)
    
    # Try increasing number of components
    best_bic = bic_single
    best_n = 1
    
    for n in range(2, min(max_components + 1, n_samples // 50 + 1)):
        gmm = GaussianMixture(
            n_components=n,
            covariance_type='full',
            random_state=42,
            max_iter=200,
            reg_covar=1e-3
        )
        gmm.fit(features)
        bic = gmm.bic(features)
        
        # Calculate improvement ratio
        bic_improvement = (bic_single - bic) / abs(bic_single)
        
        # Only accept split if improvement is significant
        if bic_improvement > (1 - split_threshold):
            best_bic = bic
            best_n = n
        else:
            break
    
    return best_n

def process_cluster_batch(args):
    """
    Process a batch of cell clusters using two-step segmentation:
    1. Initial split using GMM
    2. Boundary refinement using watershed where needed
    """
    batch_idx, cluster_labels, shm_name_labeled, shape_labeled, \
    shm_name_intensity, shape_intensity, params = args
    
    try:
        shm_labeled = shared_memory.SharedMemory(name=shm_name_labeled)
        labeled_cells = np.ndarray(shape_labeled, dtype=np.int32, buffer=shm_labeled.buf)
        
        shm_intensity = shared_memory.SharedMemory(name=shm_name_intensity)
        raw_intensity = np.ndarray(shape_intensity, dtype=np.float32, buffer=shm_intensity.buf)
    except Exception as e:
        print(f"Error accessing shared memory: {e}")
        return []
    
    min_volume = params.get('min_volume', 100)
    intensity_weight = params.get('intensity_weight', 0.3)
    split_threshold = params.get('split_threshold', 0.7)
    
    batch_results = []
    
    for label_idx in cluster_labels:
        mask = labeled_cells == label_idx
        z_indices, y_indices, x_indices = np.where(mask)
        
        if len(z_indices) < min_volume:
            batch_results.append((label_idx, [{
                'coords': (z_indices.copy(), y_indices.copy(), x_indices.copy()),
                'volume': len(z_indices)
            }]))
            continue
        
        # Extract intensity values
        intensities = raw_intensity[z_indices, y_indices, x_indices]
        
        # Calculate features for GMM
        z_norm = (z_indices - z_indices.min()) / (z_indices.max() - z_indices.min() + 1e-6)
        y_norm = (y_indices - y_indices.min()) / (y_indices.max() - y_indices.min() + 1e-6)
        x_norm = (x_indices - x_indices.min()) / (x_indices.max() - x_indices.min() + 1e-6)
        
        int_norm = np.clip(
            (intensities - np.percentile(intensities, 5)) /
            (np.percentile(intensities, 95) - np.percentile(intensities, 5) + 1e-6),
            0, 1
        )
        
        features = np.column_stack([
            z_norm, y_norm, x_norm,
            int_norm * intensity_weight
        ])
        
        try:
            # First step: GMM-based splitting
            n_components = estimate_components_statistical(
                features,
                max_components=5,
                split_threshold=split_threshold
            )
            
            if n_components == 1:
                batch_results.append((label_idx, [{
                    'coords': (z_indices.copy(), y_indices.copy(), x_indices.copy()),
                    'volume': len(z_indices)
                }]))
                continue
            
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42,
                max_iter=200,
                reg_covar=1e-3,
                init_params='kmeans'
            )
            
            gmm_labels = gmm.fit_predict(features)
            
            # Process initial GMM splits
            initial_components = []
            for i in range(n_components):
                component_mask = gmm_labels == i
                if np.sum(component_mask) >= min_volume:
                    comp_z = z_indices[component_mask]
                    comp_y = y_indices[component_mask]
                    comp_x = x_indices[component_mask]
                    
                    # Calculate component properties
                    z_size = comp_z.max() - comp_z.min()
                    y_size = comp_y.max() - comp_y.min()
                    x_size = comp_x.max() - comp_x.min()
                    max_size = max(z_size, y_size, x_size)
                    min_size = min(z_size, y_size, x_size)
                    
                    if max_size / (min_size + 1e-6) < 5:  # Basic shape check
                        initial_components.append({
                            'coords': (comp_z.copy(), comp_y.copy(), comp_x.copy()),
                            'volume': len(comp_z),
                            'needs_refinement': max_size / (min_size + 1e-6) > 2  # Flag for watershed
                        })
            
            # If GMM splitting failed, keep original ROI
            if not initial_components:
                batch_results.append((label_idx, [{
                    'coords': (z_indices.copy(), y_indices.copy(), x_indices.copy()),
                    'volume': len(z_indices)
                }]))
                continue
            
            # Second step: Refine components that need it using watershed
            final_components = []
            for comp in initial_components:
                if not comp['needs_refinement']:
                    final_components.append({
                        'coords': comp['coords'],
                        'volume': comp['volume']
                    })
                    continue
                
                # Get component bounding box
                comp_z, comp_y, comp_x = comp['coords']
                z_min, z_max = comp_z.min(), comp_z.max()
                y_min, y_max = comp_y.min(), comp_y.max()
                x_min, x_max = comp_x.min(), comp_x.max()
                
                # Add padding
                padding = 2
                z_min = max(0, z_min - padding)
                y_min = max(0, y_min - padding)
                x_min = max(0, x_min - padding)
                z_max = min(shape_labeled[0] - 1, z_max + padding)
                y_max = min(shape_labeled[1] - 1, y_max + padding)
                x_max = min(shape_labeled[2] - 1, x_max + padding)
                
                # Create local mask and intensity array
                local_shape = (z_max - z_min + 1, y_max - y_min + 1, x_max - x_min + 1)
                local_mask = np.zeros(local_shape, dtype=bool)
                local_z = comp_z - z_min
                local_y = comp_y - y_min
                local_x = comp_x - x_min
                local_mask[local_z, local_y, local_x] = True
                
                local_intensity = raw_intensity[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
                
                # Generate watershed markers using intensity peaks
                smooth_intensity = ndimage.gaussian_filter(local_intensity * local_mask, sigma=1)
                local_max = ndimage.maximum_filter(smooth_intensity, size=3) == smooth_intensity
                local_max &= smooth_intensity > np.percentile(smooth_intensity[local_mask], 75)
                markers, n_markers = ndimage.label(local_max)
                
                if n_markers > 1:  # Only apply watershed if multiple peaks found
                    # Create watershed input using intensity gradient
                    gradient = ndimage.gaussian_gradient_magnitude(smooth_intensity, sigma=1)
                    watershed_labels = ndimage.watershed_ift(
                        gradient.astype(np.uint16),
                        markers
                    )
                    watershed_labels *= local_mask  # Mask to original region
                    
                    # Process watershed components
                    for label in range(1, n_markers + 1):
                        label_mask = watershed_labels == label
                        if np.sum(label_mask) >= min_volume:
                            # Get global coordinates
                            w_z, w_y, w_x = np.where(label_mask)
                            global_z = w_z + z_min
                            global_y = w_y + y_min
                            global_x = w_x + x_min
                            
                            final_components.append({
                                'coords': (global_z.copy(), global_y.copy(), global_x.copy()),
                                'volume': len(w_z)
                            })
                else:
                    # Keep original component if watershed couldn't split
                    final_components.append({
                        'coords': comp['coords'],
                        'volume': comp['volume']
                    })
            
            batch_results.append((label_idx, final_components))
            
        except Exception as e:
            print(f"Error processing label {label_idx}: {e}")
            batch_results.append((label_idx, [{
                'coords': (z_indices.copy(), y_indices.copy(), x_indices.copy()),
                'volume': len(z_indices)
            }]))
    
    shm_labeled.close()
    shm_intensity.close()
    
    return batch_results

def segment_with_gmm_batched(labeled_cells, raw_intensity, min_volume=100, 
                            intensity_weight=0.3, split_threshold=0.7,
                            n_processes=None, batch_size=50):
    """
    Memory-efficient batched GMM segmentation with statistical model selection.
    
    Parameters:
    -----------
    labeled_cells : np.ndarray
        3D array with labeled cell clusters
    raw_intensity : np.ndarray
        Original intensity image
    min_volume : int
        Minimum volume for valid segments
    intensity_weight : float
        Weight of intensity values relative to spatial coordinates
    split_threshold : float
        Threshold for accepting cluster splits (0-1)
        Higher values make splitting less likely
    n_processes : int, optional
        Number of processes to use
    batch_size : int
        Number of clusters to process in each batch
    """
    mem = psutil.virtual_memory()
    print(f"Memory available: {mem.available / 1e9:.1f} GB")
    
    if n_processes is None:
        n_processes = max(1, min(cpu_count() // 2, 4))
    
    print(f"Using {n_processes} processes for batched processing")
    
    labeled_cells = labeled_cells.astype(np.int32)
    raw_intensity = raw_intensity.astype(np.float32)
    
    shm_labeled = shared_memory.SharedMemory(create=True, size=labeled_cells.nbytes)
    shm_labeled_array = np.ndarray(labeled_cells.shape, 
                                  dtype=labeled_cells.dtype, 
                                  buffer=shm_labeled.buf)
    shm_labeled_array[:] = labeled_cells[:]
    
    shm_intensity = shared_memory.SharedMemory(create=True, size=raw_intensity.nbytes)
    shm_intensity_array = np.ndarray(raw_intensity.shape, 
                                    dtype=raw_intensity.dtype, 
                                    buffer=shm_intensity.buf)
    shm_intensity_array[:] = raw_intensity[:]
    
    batches = get_cluster_batches(labeled_cells, batch_size, min_volume)
    print(f"Created {len(batches)} batches with up to {batch_size} clusters each")
    
    params = {
        'min_volume': min_volume,
        'intensity_weight': intensity_weight,
        'split_threshold': split_threshold
    }
    
    parallel_args = [
        (
            batch_idx,
            batch,
            shm_labeled.name,
            labeled_cells.shape,
            shm_intensity.name,
            raw_intensity.shape,
            params
        )
        for batch_idx, batch in enumerate(batches)
    ]
    
    output = np.zeros_like(labeled_cells)
    next_label = 1
    
    start_time = time.time()
    
    with Pool(n_processes) as pool:
        batch_results = list(tqdm(
            pool.imap(process_cluster_batch, parallel_args),
            total=len(batches),
            desc="Processing batches"
        ))
    
    print("Combining results...")
    for batch_result in batch_results:
        for label_idx, component_results in batch_result:
            for component in component_results:
                z, y, x = component['coords']
                output[z, y, x] = next_label
                next_label += 1
    
    shm_labeled.close()
    shm_labeled.unlink()
    shm_intensity.close()
    shm_intensity.unlink()
    
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    print(f"Identified {next_label - 1} individual cells")
    
    return output