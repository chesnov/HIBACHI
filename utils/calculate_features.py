import numpy as np
from scipy.spatial import cKDTree
import multiprocessing as mp
import os
import tempfile
import time
import psutil
import pandas as pd
from skimage.morphology import skeletonize
import networkx as nx
from tqdm import tqdm

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

    #Final points is a 3D array where the first two dimensions are the mask labels and the third dimension is the coordinates of the two points;
    #Let's convert it to a pandas dataframe with 4 columns: mask1, mask2, point1, point2
    final_points_df = pd.DataFrame(final_points.reshape(-1, 6), columns=['mask1_z', 'mask1_x', 'mask1_y', 'mask2_z', 'mask2_x', 'mask2_y'])
    final_points_df['mask1'] = np.repeat(labels, len(labels))
    final_points_df['mask2'] = np.tile(labels, len(labels))
    final_points_df = final_points_df[['mask1', 'mask2', 'mask1_z', 'mask1_x', 'mask1_y', 'mask2_z', 'mask2_x', 'mask2_y']]


    #Convert both final_distances and final_points into dataframes with the appropriate column names
    final_distances = pd.DataFrame(final_distances, index=labels, columns=labels)

    return final_distances, final_points_df, lines

def calculate_volume(segmented_array, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate the volume of each mask in a 3D segmented array in cubic microns.
    
    Parameters:
    -----------
    segmented_array : numpy.ndarray
        3D numpy array (z, x, y) containing integer labels for each mask.
        0 is assumed to be the background.
    spacing : tuple
        Tuple of (z_spacing, x_spacing, y_spacing) in microns.
    
    Returns:
    --------
    volume_df : pandas.DataFrame
        DataFrame with columns:
        - label: Mask label
        - volume_um3: Volume in cubic microns
        - surface_area_um2: Surface area in square microns (estimated)
        - voxel_count: Number of voxels in the mask
        - bounding_box_volume_um3: Volume of the bounding box in cubic microns
        - sphericity: Ratio comparing the shape to a sphere (1.0 is perfect sphere)
    """
    # Get unique labels excluding background (0)
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    
    # Calculate voxel volume in cubic microns
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    
    # Initialize results lists
    label_ids = []
    volumes = []
    surface_areas = []
    voxel_counts = []
    bbox_volumes = []
    sphericity = []
    
    # Calculate metrics for each mask
    for label in tqdm(labels):
        # Create binary mask
        mask = (segmented_array == label)
        
        # Count voxels in this mask
        voxel_count = np.sum(mask)
        
        # Calculate volume in cubic microns
        volume = voxel_count * voxel_volume
        
        # Get bounding box dimensions
        z_indices, x_indices, y_indices = np.where(mask)
        if len(z_indices) > 0:
            z_min, z_max = np.min(z_indices), np.max(z_indices)
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            
            # Calculate bounding box volume in cubic microns
            bbox_volume = (z_max - z_min + 1) * spacing[0] * \
                          (x_max - x_min + 1) * spacing[1] * \
                          (y_max - y_min + 1) * spacing[2]
        else:
            bbox_volume = 0
        
        # Estimate surface area using the boundary voxels
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
        
        # Count surface voxels
        surface_voxels = np.sum(surface)
        
        # Estimate surface area
        # We use a simple approximation based on the voxel faces
        avg_face_area = ((spacing[0] * spacing[1]) + 
                          (spacing[0] * spacing[2]) + 
                          (spacing[1] * spacing[2])) / 3
        surface_area = surface_voxels * avg_face_area
        
        # Calculate sphericity
        # Formula: π^(1/3) * (6 * Volume)^(2/3) / Surface Area
        # A perfect sphere has sphericity = 1.0
        if surface_area > 0:
            sph = np.pi**(1/3) * (6 * volume)**(2/3) / surface_area
        else:
            sph = 0
        
        # Store results
        label_ids.append(label)
        volumes.append(volume)
        surface_areas.append(surface_area)
        voxel_counts.append(voxel_count)
        bbox_volumes.append(bbox_volume)
        sphericity.append(sph)
    
    # Create DataFrame
    volume_df = pd.DataFrame({
        'label': label_ids,
        'volume_um3': volumes,
        'surface_area_um2': surface_areas,
        'voxel_count': voxel_counts,
        'bounding_box_volume_um3': bbox_volumes,
        'sphericity': sphericity
    })
    
    return volume_df

def extract_skeleton(segmented_array, label=None, spacing=(1.0, 1.0, 1.0)):
    """
    Extract the skeleton of masks in a 3D segmented array.
    
    Parameters:
    -----------
    segmented_array : numpy.ndarray
        3D numpy array (z, x, y) containing integer labels for each mask.
        0 is assumed to be the background.
    label : int, optional
        Specific label to skeletonize. If None, all masks are processed.
    spacing : tuple
        Tuple of (z_spacing, x_spacing, y_spacing) in microns.
    
    Returns:
    --------
    skeleton_df : pandas.DataFrame
        DataFrame with basic skeleton information
    skeleton_objects : dict
        Dictionary where keys are mask labels and values are tuples of:
        (skeleton_mask, skeleton_coords, skeleton_graph)
    """
    # Get labels to process
    if label is not None:
        labels = [label]
    else:
        labels = np.unique(segmented_array)
        labels = labels[labels > 0]
    
    skeletons = {}
    
    # Initialize lists for DataFrame
    label_ids = []
    skeleton_points = []
    skeleton_lengths = []
    
    for lbl in labels:
        # Create binary mask for this label
        mask = (segmented_array == lbl)
        
        # Skeletonize the mask
        skeleton = skeletonize(mask)
        
        # Get coordinates of skeleton points
        z_coords, x_coords, y_coords = np.where(skeleton)
        skeleton_coords = np.column_stack((z_coords, x_coords, y_coords))
        
        # Convert to micron coordinates
        skeleton_coords_microns = skeleton_coords * np.array(spacing)
        
        # Create a graph representation of the skeleton
        G = nx.Graph()
        
        # Add nodes for each skeleton point
        for i, coord in enumerate(skeleton_coords_microns):
            G.add_node(i, pos=coord)
        
        # Add edges between adjacent skeleton points
        # We consider two points adjacent if they are within √3 voxels of each other
        # (this covers all 26-connected neighbors in 3D)
        for i, coord1 in enumerate(skeleton_coords):
            for j, coord2 in enumerate(skeleton_coords):
                if i < j:  # To avoid duplicate edges
                    # Calculate Euclidean distance in voxel space
                    dist = np.sqrt(np.sum((coord1 - coord2) ** 2))
                    if dist <= np.sqrt(3):
                        # Calculate actual distance in microns
                        dist_microns = np.sqrt(np.sum(((coord1 - coord2) * np.array(spacing)) ** 2))
                        G.add_edge(i, j, weight=dist_microns)
        
        # Calculate total skeleton length
        total_length = sum(data['weight'] for _, _, data in G.edges(data=True))
        
        # Store results
        skeletons[lbl] = (skeleton, skeleton_coords_microns, G)
        label_ids.append(lbl)
        skeleton_points.append(len(skeleton_coords))
        skeleton_lengths.append(total_length)
    
    # Create DataFrame
    skeleton_df = pd.DataFrame({
        'label': label_ids,
        'skeleton_points': skeleton_points,
        'skeleton_length_um': skeleton_lengths
    })
    
    return skeleton_df, skeletons

def analyze_ramification(skeleton_graph, node_positions=None):
    """
    Analyze the ramification of a skeleton graph.
    
    Parameters:
    -----------
    skeleton_graph : networkx.Graph
        Graph representation of the skeleton.
    node_positions : dict, optional
        Dictionary mapping node IDs to 3D coordinates in microns.
    
    Returns:
    --------
    stats : dict
        Dictionary of ramification statistics.
    branch_df : pandas.DataFrame
        DataFrame containing information about individual branches.
    endpoint_df : pandas.DataFrame
        DataFrame containing information about individual endpoints.
    """
    if not nx.is_connected(skeleton_graph):
        # If graph is not connected, find the largest connected component
        largest_cc = max(nx.connected_components(skeleton_graph), key=len)
        skeleton_graph = skeleton_graph.subgraph(largest_cc).copy()
    
    # Calculate node degrees
    degrees = dict(skeleton_graph.degree())
    
    # Identify branch points (degree > 2) and endpoints (degree == 1)
    branch_points = [node for node, degree in degrees.items() if degree > 2]
    endpoints = [node for node, degree in degrees.items() if degree == 1]
    
    # Get node positions if available
    if node_positions is None:
        node_positions = nx.get_node_attributes(skeleton_graph, 'pos')
    
    # Calculate total length of all branches
    total_length = sum(data['weight'] for _, _, data in skeleton_graph.edges(data=True))
    
    # Find the longest path between any two endpoints
    longest_path = []
    max_path_length = 0
    longest_path_endpoints = (None, None)
    
    if len(endpoints) >= 2:
        for i, start in enumerate(endpoints):
            for end in endpoints[i+1:]:
                try:
                    path = nx.shortest_path(skeleton_graph, start, end, weight='weight')
                    path_length = nx.path_weight(skeleton_graph, path, weight='weight')
                    
                    if path_length > max_path_length:
                        max_path_length = path_length
                        longest_path = path
                        longest_path_endpoints = (start, end)
                except nx.NetworkXNoPath:
                    continue
    
    # Initialize branch data lists
    branch_ids = []
    branch_src_types = []
    branch_dst_types = []
    branch_lengths = []
    branch_tortuosity = []  # Ratio of path length to Euclidean distance
    
    # Initialize endpoint data lists
    endpoint_ids = []
    endpoint_x = []
    endpoint_y = []
    endpoint_z = []
    endpoint_distance_to_soma = []  # Using center of mass as proxy for soma
    
    # Calculate center of mass as proxy for soma
    if node_positions:
        center_of_mass = np.mean([pos for pos in node_positions.values()], axis=0)
    else:
        center_of_mass = np.zeros(3)
    
    # Analyze each endpoint
    for i, ep in enumerate(endpoints):
        if ep in node_positions:
            pos = node_positions[ep]
            endpoint_ids.append(i)
            endpoint_x.append(pos[0])
            endpoint_y.append(pos[1])
            endpoint_z.append(pos[2])
            
            # Calculate distance to center of mass
            dist_to_soma = np.sqrt(np.sum((pos - center_of_mass)**2))
            endpoint_distance_to_soma.append(dist_to_soma)
    
    # Analyze branches
    branch_id = 0
    analyzed_paths = set()
    
    # First analyze branches from branch points to endpoints
    for bp in branch_points:
        for ep in endpoints:
            if bp == ep:
                continue
                
            # Create a unique identifier for this path
            path_key = tuple(sorted([bp, ep]))
            if path_key in analyzed_paths:
                continue
            
            try:
                path = nx.shortest_path(skeleton_graph, bp, ep, weight='weight')
                
                # Skip if path includes another branch point
                has_other_bp = False
                for node in path[1:-1]:  # Exclude source and destination
                    if node in branch_points:
                        has_other_bp = True
                        break
                
                if not has_other_bp:
                    path_length = nx.path_weight(skeleton_graph, path, weight='weight')
                    
                    # Calculate Euclidean distance if positions are available
                    if node_positions and bp in node_positions and ep in node_positions:
                        euclidean_dist = np.sqrt(np.sum((node_positions[bp] - node_positions[ep])**2))
                        tort = path_length / euclidean_dist if euclidean_dist > 0 else 1.0
                    else:
                        tort = 1.0
                    
                    branch_ids.append(branch_id)
                    branch_src_types.append('branch_point')
                    branch_dst_types.append('endpoint')
                    branch_lengths.append(path_length)
                    branch_tortuosity.append(tort)
                    
                    branch_id += 1
                    analyzed_paths.add(path_key)
            except nx.NetworkXNoPath:
                continue
    
    # Analyze branches between branch points
    for i, bp1 in enumerate(branch_points):
        for bp2 in branch_points[i+1:]:
            # Create a unique identifier for this path
            path_key = tuple(sorted([bp1, bp2]))
            if path_key in analyzed_paths:
                continue
            
            try:
                path = nx.shortest_path(skeleton_graph, bp1, bp2, weight='weight')
                
                # Skip if path includes another branch point
                has_other_bp = False
                for node in path[1:-1]:  # Exclude source and destination
                    if node in branch_points and node != bp1 and node != bp2:
                        has_other_bp = True
                        break
                
                if not has_other_bp:
                    path_length = nx.path_weight(skeleton_graph, path, weight='weight')
                    
                    # Calculate Euclidean distance if positions are available
                    if node_positions and bp1 in node_positions and bp2 in node_positions:
                        euclidean_dist = np.sqrt(np.sum((node_positions[bp1] - node_positions[bp2])**2))
                        tort = path_length / euclidean_dist if euclidean_dist > 0 else 1.0
                    else:
                        tort = 1.0
                    
                    branch_ids.append(branch_id)
                    branch_src_types.append('branch_point')
                    branch_dst_types.append('branch_point')
                    branch_lengths.append(path_length)
                    branch_tortuosity.append(tort)
                    
                    branch_id += 1
                    analyzed_paths.add(path_key)
            except nx.NetworkXNoPath:
                continue
    
    # Create stats dictionary
    stats = {
        'num_nodes': len(skeleton_graph.nodes),
        'num_branch_points': len(branch_points),
        'num_endpoints': len(endpoints),
        'total_skeleton_length': total_length,
        'longest_path_length': max_path_length,
        'avg_branch_length': np.mean(branch_lengths) if branch_lengths else 0,
        'max_branch_length': np.max(branch_lengths) if branch_lengths else 0,
        'min_branch_length': np.min(branch_lengths) if branch_lengths else 0,
        'avg_tortuosity': np.mean(branch_tortuosity) if branch_tortuosity else 1.0,
        'branch_count': len(branch_lengths)
    }
    
    # Create branch DataFrame
    branch_df = pd.DataFrame({
        'branch_id': branch_ids,
        'source_type': branch_src_types,
        'destination_type': branch_dst_types,
        'length_um': branch_lengths,
        'tortuosity': branch_tortuosity
    })
    
    # Create endpoint DataFrame
    endpoint_df = pd.DataFrame({
        'endpoint_id': endpoint_ids,
        'x_um': endpoint_x,
        'y_um': endpoint_y,
        'z_um': endpoint_z,
        'distance_to_soma_um': endpoint_distance_to_soma
    })
    
    return stats, branch_df, endpoint_df

def calculate_ramification_stats(segmented_array, spacing=(1.0, 1.0, 1.0), labels=None):
    """
    Calculate ramification statistics for masks in a 3D segmented array with improved memory efficiency.
    
    Parameters:
    -----------
    segmented_array : numpy.ndarray
        3D numpy array (z, x, y) containing integer labels for each mask.
        0 is assumed to be the background.
    spacing : tuple
        Tuple of (z_spacing, x_spacing, y_spacing) in microns.
    labels : list, optional
        Specific labels to analyze. If None, all masks are processed.
    
    Returns:
    --------
    ramification_summary_df : pandas.DataFrame
        DataFrame with summary statistics for each mask.
    branch_data_df : pandas.DataFrame
        DataFrame with detailed information about each branch.
    endpoint_data_df : pandas.DataFrame
        DataFrame with detailed information about each endpoint.
    """
    import numpy as np
    import pandas as pd
    import gc  # For garbage collection
    from tqdm import tqdm
    
    # Determine which labels to process
    if labels is None:
        unique_labels = np.unique(segmented_array)
        labels = unique_labels[unique_labels > 0]
    
    # Initialize DataFrames for results
    ramification_summary_rows = []
    branch_data_dfs = []
    endpoint_data_dfs = []
    
    # Process one label at a time
    for label in tqdm(labels):
        # Extract mask for current label only
        mask = segmented_array == label
        
        # Skip if mask is empty
        if not np.any(mask):
            print(f"Warning: Label {label} not found in the segmented array.")
            continue
        
        # Extract skeleton for the current label only
        _, skeleton_result = extract_skeleton(mask, label=1, spacing=spacing)
        
        if 1 in skeleton_result:  # The label will be 1 since we made a binary mask
            _, skeleton_coords, skeleton_graph = skeleton_result[1]
            
            # Create node positions dictionary
            node_positions = {i: coord for i, coord in enumerate(skeleton_coords)}
            
            # Analyze ramification
            stats, branch_df, endpoint_df = analyze_ramification(skeleton_graph, node_positions)
            
            # Add label column to branch and endpoint DataFrames
            branch_df['label'] = label
            endpoint_df['label'] = label
            
            # Store results in row format
            ramification_summary_rows.append({
                'label': label,
                'num_skeleton_nodes': stats['num_nodes'],
                'num_branch_points': stats['num_branch_points'],
                'num_endpoints': stats['num_endpoints'],
                'total_skeleton_length_um': stats['total_skeleton_length'],
                'longest_path_length_um': stats['longest_path_length'],
                'avg_branch_length_um': stats['avg_branch_length'],
                'max_branch_length_um': stats['max_branch_length'],
                'min_branch_length_um': stats['min_branch_length'],
                'avg_tortuosity': stats['avg_tortuosity'],
                'branch_count': stats['branch_count']
            })
            
            # Save dataframes for this label
            branch_data_dfs.append(branch_df)
            endpoint_data_dfs.append(endpoint_df)
            
            # Clean up to free memory
            del skeleton_coords, skeleton_graph, node_positions
            gc.collect()
        else:
            print(f"Warning: Failed to extract skeleton for label {label}.")
        
        # Clean up the mask to free memory
        del mask
        gc.collect()
    
    # Create summary DataFrame from collected rows
    ramification_summary_df = pd.DataFrame(ramification_summary_rows)
    
    # Combine all branch and endpoint data
    branch_data_df = pd.concat(branch_data_dfs) if branch_data_dfs else pd.DataFrame()
    endpoint_data_df = pd.concat(endpoint_data_dfs) if endpoint_data_dfs else pd.DataFrame()
    
    return ramification_summary_df, branch_data_df, endpoint_data_df

def analyze_segmentation(segmented_array, spacing=(1.0, 1.0, 1.0), calculate_skeletons=True):
    """
    Comprehensive analysis of a 3D segmented array, combining volume and ramification metrics.
    
    Parameters:
    -----------
    segmented_array : numpy.ndarray
        3D numpy array (z, x, y) containing integer labels for each mask.
        0 is assumed to be the background.
    spacing : tuple
        Tuple of (z_spacing, x_spacing, y_spacing) in microns.
    calculate_skeletons : bool
        Whether to calculate skeleton-based metrics (which can be time-consuming).
    
    Returns:
    --------
    metrics_df : pandas.DataFrame
        DataFrame with combined metrics for each mask.
    ramification_metrics : tuple
        Tuple of (ramification_summary_df, branch_data_df, endpoint_data_df) if calculate_skeletons=True,
        otherwise None.
    """
    # Calculate volume metrics
    print("Calculating volume metrics...")
    volume_df = calculate_volume(segmented_array, spacing=spacing)
    
    if calculate_skeletons:
        # Calculate ramification stats
        print("Calculating ramification metrics...")
        ramification_summary_df, branch_data_df, endpoint_data_df = calculate_ramification_stats(
            segmented_array, spacing=spacing)
        
        # Merge volume and ramification metrics
        metrics_df = pd.merge(volume_df, ramification_summary_df, on='label', how='outer')
        ramification_metrics = (ramification_summary_df, branch_data_df, endpoint_data_df)
    else:
        metrics_df = volume_df
        ramification_metrics = None
    
    return metrics_df, ramification_metrics