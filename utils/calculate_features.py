import numpy as np
from scipy.spatial import cKDTree
import multiprocessing as mp
import os
import tempfile
import time
import psutil

def extract_surface_points_worker(args):
    """Worker function to extract surface points from a mask with careful handling of boundaries"""
    label_idx, label, segmented_array, temp_dir, spacing = args
    
    # Create binary mask
    mask = (segmented_array == label)
    
    # Initialize surface array
    surface = np.zeros_like(mask, dtype=bool)
    
    # More careful boundary detection that handles array dimensions correctly
    z_max, x_max, y_max = mask.shape
    
    # Check Z direction
    if z_max > 1:  # Only process if there's more than one Z slice
        surface[:-1, :, :] |= (mask[:-1, :, :] & ~mask[1:, :, :])
        surface[1:, :, :] |= (mask[1:, :, :] & ~mask[:-1, :, :])
    
    # Check X direction
    surface[:, :-1, :] |= (mask[:, :-1, :] & ~mask[:, 1:, :])
    surface[:, 1:, :] |= (mask[:, 1:, :] & ~mask[:, :-1, :])
    
    # Check Y direction
    surface[:, :, :-1] |= (mask[:, :, :-1] & ~mask[:, :, 1:])
    surface[:, :, 1:] |= (mask[:, :, 1:] & ~mask[:, :, :-1])
    
    # Extract surface points
    z, x, y = np.where(surface)
    surface_points = np.column_stack((z, x, y))
    
    # Create file for the surface points
    surface_file = os.path.join(temp_dir, f"surface_{label}.npy")
    np.save(surface_file, surface_points)
    
    return label_idx, len(surface_points)

def calculate_pair_distances_worker(args):
    """Worker function to calculate distances between a pair of masks"""
    i, j, labels, temp_dir, spacing, distances_matrix, points_matrix = args
    
    if i == j:
        return i, j, 0, np.zeros(6)
    
    # Check if we've already computed the reverse direction
    if distances_matrix[j, i] > 0:
        distance = distances_matrix[j, i]
        points = np.array([points_matrix[j, i][3], points_matrix[j, i][4], points_matrix[j, i][5], 
                          points_matrix[j, i][0], points_matrix[j, i][1], points_matrix[j, i][2]])
        return i, j, distance, points
    
    label1 = labels[i]
    label2 = labels[j]
    
    # Load surface points for both labels
    try:
        surface_points1 = np.load(os.path.join(temp_dir, f"surface_{label1}.npy"))
        surface_points2 = np.load(os.path.join(temp_dir, f"surface_{label2}.npy"))
    except Exception as e:
        print(f"Error loading surface points for masks {label1} and {label2}: {e}")
        return i, j, np.inf, np.zeros(6)
    
    if len(surface_points1) == 0 or len(surface_points2) == 0:
        return i, j, np.inf, np.zeros(6)
    
    # Scale points according to spacing
    scaled_points1 = surface_points1 * np.array(spacing)
    scaled_points2 = surface_points2 * np.array(spacing)
    
    # Build KDTree for the second mask
    tree2 = cKDTree(scaled_points2)
    
    # Find closest points and distances
    distances, indices = tree2.query(scaled_points1, k=1)
    
    # Get minimum distance and corresponding points
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]
    
    # Get corresponding points
    point1 = surface_points1[min_idx]
    point2 = surface_points2[indices[min_idx]]
    
    result_points = np.array([point1[0], point1[1], point1[2], point2[0], point2[1], point2[2]])
    
    return i, j, min_distance, result_points



def shortest_distance_points(final_distances, final_points):
    """
    Get the shortest distance and corresponding points between two masks.
    Parameters:
    -----------
    final_distances : numpy.ndarray
        n x n matrix where n is the number of unique masks (excluding background).
        Entry (i,j) contains the shortest distance in microns between masks i+1 and j+1.
    final_points : numpy.ndarray
        n x n x 6 matrix where entry (i,j) contains the coordinates of the closest
        point pair between masks [i+1, j+1] in the format:
        [z1, x1, y1, z2, x2, y2] where (z1,x1,y1) is on mask i+1 and (z2,x2,y2) is on mask j+1.
    Returns:
    --------
    lines: a numpy array of length n x 2 x 3 where for each mask i, the corresponding entry is a tuple of points.
    First point is on the surface of mask i, the second point is on the surface of mask j that is the closest ROI to i.
    """
    n_labels = final_distances.shape[0]
    lines = np.zeros((n_labels, 2, 3))
    for i in range(n_labels):
        min_idx = min_idx = np.argsort(final_distances[i])[1] # Exclude self and break ties by taking the first one
        lines[i, 0] = final_points[i, min_idx, :3]
        lines[i, 1] = final_points[i, min_idx, 3:]
    return lines

def shortest_distance(segmented_array, spacing=(1.0, 1.0, 1.0), 
                                       temp_dir=None, n_jobs=None, max_memory_pct=75,
                                       batch_size=None):
    """
    Calculate the shortest distance between surfaces of all masks in a 3D segmented array
    with memory efficiency and parallelism.
    
    Parameters:
    -----------
    segmented_array : numpy.ndarray
        3D numpy array (z, x, y) containing integer labels for each mask.
        0 is assumed to be the background.
    spacing : tuple
        Tuple of (z_spacing, x_spacing, y_spacing) in microns.
    temp_dir : str, optional
        Directory to store temporary memory-mapped files. If None, uses the system temp directory.
    n_jobs : int, optional
        Number of parallel jobs. If None, uses number of CPU cores - 1.
    max_memory_pct : int, optional
        Maximum percentage of system memory to use (default: 75%).
    batch_size : int, optional
        Size of mask batches to process at once. If None, automatically determined.
        
    Returns:
    --------
    distances_matrix : numpy.ndarray
        n x n matrix where n is the number of unique masks (excluding background).
        Entry (i,j) contains the shortest distance in microns between masks i+1 and j+1.
    
    points_matrix : numpy.ndarray
        n x n x 6 matrix where entry (i,j) contains the coordinates of the closest
        point pair between masks [i+1, j+1] in the format:
        [z1, x1, y1, z2, x2, y2] where (z1,x1,y1) is on mask i+1 and (z2,x2,y2) is on mask j+1.
    """
    start_time = time.time()
    
    # Setup parallel processing
    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)
    
    # Create temp directory if not provided
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    
    # Get unique labels excluding background (0)
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    n_labels = len(labels)
    
    if n_labels == 0:
        print("No masks found in the segmented array.")
        return np.array([]), np.array([])
    
    print(f"Found {n_labels} unique masks. Starting processing with {n_jobs} parallel jobs.")
    
    # Calculate available memory and decide batch size
    available_memory = psutil.virtual_memory().available * (max_memory_pct / 100)
    voxel_size = segmented_array.itemsize
    mask_memory = np.prod(segmented_array.shape) * voxel_size
    
    if batch_size is None:
        # Estimate memory needed for a single surface extraction and distance calculation
        est_process_memory = 3 * mask_memory + n_labels * 100 * 3 * 8  # Assuming ~100 surface points per mask on average
        batch_size = max(1, int(available_memory / (est_process_memory * n_jobs)))
        batch_size = min(batch_size, n_labels)
    
    print(f"Processing in batches of {batch_size} masks.")
    
    # Initialize result matrices as memory-mapped files
    distances_filename = os.path.join(temp_dir, "distances_matrix.npy")
    points_filename = os.path.join(temp_dir, "points_matrix.npy")
    
    distances_matrix = np.memmap(distances_filename, dtype=np.float32, 
                               mode='w+', shape=(n_labels, n_labels))
    points_matrix = np.memmap(points_filename, dtype=np.float32,
                           mode='w+', shape=(n_labels, n_labels, 6))
    
    # Fill diagonal with zeros (distance to self)
    for i in range(n_labels):
        distances_matrix[i, i] = 0
    
    # Extract and store surface points for each mask
    print("Extracting surface points for all masks...")
    surface_info = {}
    
    # Prepare arguments for parallel processing
    extract_args = [(i, labels[i], segmented_array, temp_dir, spacing) 
                   for i in range(n_labels)]
    
    # Process surface extraction in parallel batches
    with mp.Pool(processes=n_jobs) as pool:
        for batch_start in range(0, n_labels, batch_size):
            batch_end = min(batch_start + batch_size, n_labels)
            batch_args = extract_args[batch_start:batch_end]
            
            try:
                results = pool.map(extract_surface_points_worker, batch_args)
                for label_idx, num_points in results:
                    surface_info[label_idx] = num_points
                print(f"Extracted surfaces for masks {batch_start+1}-{batch_end} of {n_labels}")
            except Exception as e:
                print(f"Error processing batch {batch_start+1}-{batch_end}: {e}")
                raise
    
    # Prepare all pairs to process
    all_pairs = [(i, j) for i in range(n_labels) for j in range(i+1, n_labels)]
    
    # Process in batches to manage memory
    pairs_batch_size = max(1, min(1000, int(len(all_pairs) / 10)))
    print(f"Processing distance calculations in batches of {pairs_batch_size} pairs.")
    
    # Write memmap data to disk to ensure it's accessible from other processes
    distances_matrix.flush()
    points_matrix.flush()
    
    # Create a shared memory view for distance calculations
    distance_calc_view = np.memmap(distances_filename, dtype=np.float32, 
                                 mode='r+', shape=(n_labels, n_labels))
    points_calc_view = np.memmap(points_filename, dtype=np.float32,
                               mode='r+', shape=(n_labels, n_labels, 6))
    
    # Process pair distance calculations in parallel batches
    with mp.Pool(processes=n_jobs) as pool:
        for batch_idx in range(0, len(all_pairs), pairs_batch_size):
            batch_pairs = all_pairs[batch_idx:batch_idx + pairs_batch_size]
            
            # Prepare args for this batch
            calc_args = [(i, j, labels, temp_dir, spacing, distance_calc_view, points_calc_view) 
                        for i, j in batch_pairs]
            
            try:
                results = pool.map(calculate_pair_distances_worker, calc_args)
                
                # Update distances manually to avoid race conditions
                for i, j, distance, points in results:
                    distances_matrix[i, j] = distance
                    distances_matrix[j, i] = distance
                    for k in range(6):
                        points_matrix[i, j, k] = points[k]
                    points_matrix[j, i, 0] = points[3]
                    points_matrix[j, i, 1] = points[4]
                    points_matrix[j, i, 2] = points[5]
                    points_matrix[j, i, 3] = points[0]
                    points_matrix[j, i, 4] = points[1]
                    points_matrix[j, i, 5] = points[2]
                
                # Ensure data is written to disk
                distances_matrix.flush()
                points_matrix.flush()
                
                completion = (batch_idx + len(batch_pairs)) / len(all_pairs) * 100
                elapsed = time.time() - start_time
                print(f"Completed {completion:.1f}% ({batch_idx + len(batch_pairs)}/{len(all_pairs)} pairs) in {elapsed:.1f} seconds")
            except Exception as e:
                print(f"Error processing batch at index {batch_idx}: {e}")
                raise
    
    # Load results into memory for return
    final_distances = np.array(distances_matrix)
    final_points = np.array(points_matrix)
    
    # Clean up temporary files
    try:
        os.unlink(distances_filename)
        os.unlink(points_filename)
        for label in labels:
            surface_file = os.path.join(temp_dir, f"surface_{label}.npy")
            if os.path.exists(surface_file):
                os.unlink(surface_file)
    except Exception as e:
        print(f"Warning: Could not clean up all temporary files: {e}")
    
    elapsed = time.time() - start_time
    print(f"Finished all calculations in {elapsed:.1f} seconds")
    
    lines = shortest_distance_points(final_distances, final_points)
    return final_distances, final_points, lines