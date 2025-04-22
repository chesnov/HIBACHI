import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import ndimage as ndi # For potential preprocessing
import multiprocessing as mp
import os
import tempfile
import time
import psutil
import gc
from skimage.morphology import skeletonize
from skan import Skeleton, summarize
import networkx as nx # Still potentially useful for graph ops if needed
from tqdm.auto import tqdm # Use auto for notebook/console compatibility
from scipy.sparse import csr_matrix
import traceback

seed = 42
np.random.seed(seed)         # For NumPy

# =============================================================================
# Distance Calculation Functions
# =============================================================================

def extract_surface_points_worker(args):
    """Worker function to extract surface points from a mask."""
    label_idx, label, segmented_array, temp_dir, spacing = args
    mask = (segmented_array == label)
    if not np.any(mask):
        surface_file = os.path.join(temp_dir, f"surface_{label}.npy")
        np.save(surface_file, np.empty((0, 3), dtype=int)) # Save empty array
        return label_idx, 0

    surface = np.zeros_like(mask, dtype=bool)
    z_max, x_max, y_max = mask.shape

    # Check Z direction (only if Z > 1)
    if z_max > 1:
        surface[:-1, :, :] |= (mask[:-1, :, :] & ~mask[1:, :, :])
        surface[1:, :, :] |= (mask[1:, :, :] & ~mask[:-1, :, :])
    elif z_max == 1: # Handle single slice case - surface is the mask itself
         surface[0,:,:] = mask[0,:,:]

    # Check X direction
    surface[:, :-1, :] |= (mask[:, :-1, :] & ~mask[:, 1:, :])
    surface[:, 1:, :] |= (mask[:, 1:, :] & ~mask[:, :-1, :])
    if x_max == 1: # Handle single column case
        surface[:, 0, :] = mask[:, 0, :]


    # Check Y direction
    surface[:, :, :-1] |= (mask[:, :, :-1] & ~mask[:, :, 1:])
    surface[:, :, 1:] |= (mask[:, :, 1:] & ~mask[:, :, :-1])
    if y_max == 1: # Handle single row case
        surface[:, :, 0] = mask[:, :, 0]


    z, x, y = np.where(surface)
    surface_points = np.column_stack((z, x, y))

    surface_file = os.path.join(temp_dir, f"surface_{label}.npy")
    np.save(surface_file, surface_points)

    return label_idx, len(surface_points)

def calculate_pair_distances_worker(args):
    """Worker function to calculate distances between a pair of masks."""
    i, j, labels, temp_dir, spacing, distances_matrix, points_matrix = args

    label1 = labels[i]
    label2 = labels[j]

    # Load surface points, handle potential errors or empty files
    try:
        surface_points1 = np.load(os.path.join(temp_dir, f"surface_{label1}.npy"))
        surface_points2 = np.load(os.path.join(temp_dir, f"surface_{label2}.npy"))
    except Exception as e:
        print(f"Warning: Error loading surface points for masks {label1} or {label2}: {e}")
        return i, j, np.inf, np.full(6, np.nan) # Use NaN for points if error

    if surface_points1.shape[0] == 0 or surface_points2.shape[0] == 0:
        return i, j, np.inf, np.full(6, np.nan) # Use NaN for points if no surface

    # Check if we've already computed (more robustly using NaN check)
    # This check might be less effective with parallel execution modifying the matrix directly
    # Relying on the main loop structure is safer.
    # if not np.isnan(distances_matrix[j, i]):
    #     distance = distances_matrix[j, i]
    #     points = np.array([points_matrix[j, i, 3], points_matrix[j, i, 4], points_matrix[j, i, 5],
    #                       points_matrix[j, i, 0], points_matrix[j, i, 1], points_matrix[j, i, 2]])
    #     return i, j, distance, points

    # Scale points according to spacing
    scaled_points1 = surface_points1 * np.array(spacing)
    scaled_points2 = surface_points2 * np.array(spacing)

    # Build KDTree for the second mask
    try:
        tree2 = cKDTree(scaled_points2)
    except ValueError: # Handle case where scaled_points2 is empty/invalid
         return i, j, np.inf, np.full(6, np.nan)

    # Find closest points and distances
    distances, indices = tree2.query(scaled_points1, k=1)

    # Get minimum distance and corresponding points
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    point1 = surface_points1[min_idx]
    point2 = surface_points2[indices[min_idx]]

    result_points = np.array([point1[0], point1[1], point1[2], point2[0], point2[1], point2[2]])

    return i, j, min_distance, result_points


def shortest_distance(segmented_array, spacing=(1.0, 1.0, 1.0),
                      temp_dir=None, n_jobs=None, max_memory_pct=75,
                      batch_size=None):
    """
    Calculate the shortest distance between surfaces of all masks pairs.

    Returns:
    --------
    distance_matrix_df : pandas.DataFrame
        n x n DataFrame where n is the number of unique masks.
        Entry (label1, label2) contains the shortest distance in microns.
    points_df : pandas.DataFrame
        DataFrame containing the coordinates of the closest point pair for
        every pair of masks [label1, label2]. Columns:
        'mask1', 'mask2', 'mask1_z', 'mask1_x', 'mask1_y',
        'mask2_z', 'mask2_x', 'mask2_y'.
    """
    start_time = time.time()

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    temp_dir_managed = False
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="shortest_dist_")
        temp_dir_managed = True # Flag to cleanup later
    os.makedirs(temp_dir, exist_ok=True)

    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    n_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}

    if n_labels <= 1:
        print("Need at least two masks to calculate distances.")
        if temp_dir_managed:
             try:
                 os.rmdir(temp_dir)
             except OSError:
                 pass # Ignore if not empty due to error files
        empty_df = pd.DataFrame(index=labels, columns=labels)
        empty_points = pd.DataFrame(columns=['mask1', 'mask2', 'mask1_z', 'mask1_x', 'mask1_y', 'mask2_z', 'mask2_x', 'mask2_y'])
        if n_labels == 1:
            empty_df.loc[labels[0], labels[0]] = 0.0
        return empty_df, empty_points

    print(f"Found {n_labels} masks. Calculating distances with {n_jobs} workers.")

    # --- Surface Point Extraction ---
    print("Extracting surface points...")
    extract_args = [(i, labels[i], segmented_array, temp_dir, spacing)
                   for i in range(n_labels)]

    with mp.Pool(processes=n_jobs) as pool:
         list(tqdm(pool.imap_unordered(extract_surface_points_worker, extract_args), total=n_labels, desc="Extracting Surfaces"))


    # --- Distance Calculation ---
    print("Calculating pairwise distances...")
    # Use standard numpy arrays for results, manage memory via batching if needed
    # For very large N, memmap might still be better, but adds complexity
    distances_matrix = np.full((n_labels, n_labels), np.inf, dtype=np.float32)
    points_matrix = np.full((n_labels, n_labels, 6), np.nan, dtype=np.float32)
    np.fill_diagonal(distances_matrix, 0)

    all_pairs = [(i, j) for i in range(n_labels) for j in range(i + 1, n_labels)]

    # Simple batching for progress reporting, not strict memory batching here
    # The memory bottleneck is more likely during KDTree query if surfaces are huge
    pairs_batch_size = max(1, min(1000, int(len(all_pairs) / 10))) if len(all_pairs) > 0 else 1

    calc_args = [(i, j, labels, temp_dir, spacing, distances_matrix, points_matrix)
                 for i, j in all_pairs] # Pass numpy arrays directly

    with mp.Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap_unordered(calculate_pair_distances_worker, calc_args), total=len(all_pairs), desc="Calculating Distances"))

    # Process results (update symmetric parts)
    for i, j, distance, points in results:
        distances_matrix[i, j] = distance
        distances_matrix[j, i] = distance
        points_matrix[i, j, :] = points
        if not np.any(np.isnan(points)): # Ensure points are valid before swapping
            points_matrix[j, i, :] = np.array([points[3], points[4], points[5],
                                               points[0], points[1], points[2]])

    # --- Convert to DataFrames ---
    distance_matrix_df = pd.DataFrame(distances_matrix, index=labels, columns=labels)

    points_flat = points_matrix.reshape(-1, 6)
    mask1_labels = np.repeat(labels, n_labels)
    mask2_labels = np.tile(labels, n_labels)

    points_df = pd.DataFrame({
        'mask1': mask1_labels,
        'mask2': mask2_labels,
        'mask1_z': points_flat[:, 0], 'mask1_x': points_flat[:, 1], 'mask1_y': points_flat[:, 2],
        'mask2_z': points_flat[:, 3], 'mask2_x': points_flat[:, 4], 'mask2_y': points_flat[:, 5]
    })
    # Remove self-comparisons where points are NaN
    points_df = points_df[points_df['mask1'] != points_df['mask2']].reset_index(drop=True)


    # --- Cleanup ---
    if temp_dir_managed:
        try:
            for label in labels:
                surface_file = os.path.join(temp_dir, f"surface_{label}.npy")
                if os.path.exists(surface_file):
                    os.remove(surface_file)
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Warning: Could not fully clean up temporary directory {temp_dir}: {e}")

    elapsed = time.time() - start_time
    print(f"Distance calculation finished in {elapsed:.1f} seconds")

    return distance_matrix_df, points_df

# =============================================================================
# Volume and Basic Shape Metrics
# =============================================================================

def calculate_volume(segmented_array, spacing=(1.0, 1.0, 1.0)):
    """Calculate volume and basic shape metrics for each mask."""
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    voxel_volume = np.prod(spacing)
    results = []

    for label in tqdm(labels, desc="Calculating Volumes"):
        mask = (segmented_array == label)
        voxel_count = np.sum(mask)
        if voxel_count == 0:
            continue

        volume = voxel_count * voxel_volume

        # Bounding box
        z_indices, x_indices, y_indices = np.where(mask)
        z_min, z_max = np.min(z_indices), np.max(z_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        bbox_volume = (z_max - z_min + 1) * spacing[0] * \
                      (x_max - x_min + 1) * spacing[1] * \
                      (y_max - y_min + 1) * spacing[2]

        # Surface area estimation (using boundary voxels)
        surface = np.zeros_like(mask, dtype=bool)
        z_dim, x_dim, y_dim = mask.shape
        if z_dim > 1:
            surface[:-1, :, :] |= (mask[:-1, :, :] != mask[1:, :, :])
            surface[1:, :, :] |= (mask[1:, :, :] != mask[:-1, :, :])
        if x_dim > 1:
            surface[:, :-1, :] |= (mask[:, :-1, :] != mask[:, 1:, :])
            surface[:, 1:, :] |= (mask[:, 1:, :] != mask[:, :-1, :])
        if y_dim > 1:
            surface[:, :, :-1] |= (mask[:, :, :-1] != mask[:, :, 1:])
            surface[:, :, 1:] |= (mask[:, :, 1:] != mask[:, :, :-1])
        # Include boundaries at the edge of the array
        surface[0, :, :] |= mask[0, :, :]
        if z_dim > 1: surface[-1, :, :] |= mask[-1, :, :]
        surface[:, 0, :] |= mask[:, 0, :]
        if x_dim > 1: surface[:, -1, :] |= mask[:, -1, :]
        surface[:, :, 0] |= mask[:, :, 0]
        if y_dim > 1: surface[:, :, -1] |= mask[:, :, -1]

        surface_voxels = np.sum(surface & mask) # Count only surface voxels belonging to the mask


        # A common surface area approximation: count exposed faces
        sx, sy, sz = spacing[1], spacing[2], spacing[0]
        area_xy = sx * sy
        area_xz = sx * sz
        area_yz = sy * sz
        surface_area = 0
        # Check differences along axes
        diff_z = np.diff(mask.astype(np.int8), axis=0)
        diff_x = np.diff(mask.astype(np.int8), axis=1)
        diff_y = np.diff(mask.astype(np.int8), axis=2)
        surface_area += np.sum(np.abs(diff_z)) * area_xy # Faces between z-slices
        surface_area += np.sum(np.abs(diff_x)) * area_yz # Faces between x-slices
        surface_area += np.sum(np.abs(diff_y)) * area_xz # Faces between y-slices
        # Add faces at the array boundaries if the mask touches them
        surface_area += np.sum(mask[0, :, :]) * area_xy
        surface_area += np.sum(mask[-1, :, :]) * area_xy
        surface_area += np.sum(mask[:, 0, :]) * area_yz
        surface_area += np.sum(mask[:, -1, :]) * area_yz
        surface_area += np.sum(mask[:, :, 0]) * area_xz
        surface_area += np.sum(mask[:, :, -1]) * area_xz


        # Sphericity
        if surface_area > 1e-6: # Avoid division by zero
            sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area
        else:
            sphericity = np.nan # Undefined for zero surface area

        results.append({
            'label': label,
            'volume_um3': volume,
            'surface_area_um2': surface_area,
            'voxel_count': voxel_count,
            'bounding_box_volume_um3': bbox_volume,
            'sphericity': sphericity
        })

    return pd.DataFrame(results)

# =============================================================================
# Depth Calculation
# =============================================================================

def calculate_depth_df(segmented_array, spacing=(1.0, 1.0, 1.0)):
    """Calculate the median depth of each mask in microns."""
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    depth_data = []

    for label in tqdm(labels, desc="Calculating Depths"):
        mask = (segmented_array == label)
        if np.any(mask):
            z_indices, _, _ = np.where(mask)
            median_z = np.median(z_indices)
            depth = median_z * spacing[0] # Depth based on Z-axis spacing
            depth_data.append({'label': label, 'depth_um': depth})
        else:
            depth_data.append({'label': label, 'depth_um': np.nan})

    return pd.DataFrame(depth_data)

# =============================================================================
# Skeletonization and Ramification (using skan)
# =============================================================================

def analyze_skeleton_with_skan(mask, spacing, label):
    """Analyzes a single binary mask using skan. Returns DataFrame and skeleton image."""
    if not np.any(mask):
        return pd.DataFrame(), np.zeros_like(mask, dtype=np.uint8), 0, 0, 0, None

    # Optional Preprocessing...

    try:
        skeleton_img = skeletonize(mask > 0).astype(np.uint8)
    except Exception as e:
         print(f"Label {label}: Skeletonize failed: {e}")
         return pd.DataFrame(), np.zeros_like(mask, dtype=np.uint8), 0, 0, 0, None

    if not np.any(skeleton_img):
        return pd.DataFrame(), skeleton_img, 0, 0, 0, None

    # Initialize return values
    branch_data = pd.DataFrame()
    graph_obj = None # Store the Skeleton object
    n_junctions = 0
    n_endpoints = 0
    total_nodes = 0

    try:
        # Create the skan Skeleton object (WITHOUT separator argument)
        graph_obj = Skeleton(skeleton_img, spacing=spacing, source_image=mask)

        # Summarize branches - this populates the internal graph structure
        # The warning about separator='-' vs '_' might reappear, ignore it for now or suppress externally
        branch_data = summarize(graph_obj) # Pass graph object

        # --- CONVERT ADJACENCY MATRIX TO NETWORKX GRAPH ---
        # Check if graph_obj.graph exists and is a sparse matrix
        nx_graph = None
        if hasattr(graph_obj, 'graph') and isinstance(graph_obj.graph, csr_matrix):
            # Convert the sparse matrix to a networkx graph
            nx_graph = nx.from_scipy_sparse_array(graph_obj.graph)
            total_nodes = nx_graph.number_of_nodes()
        elif hasattr(graph_obj, 'graph') and isinstance(graph_obj.graph, nx.Graph):
             # If it's already a networkx graph (newer skan versions)
             nx_graph = graph_obj.graph
             total_nodes = nx_graph.number_of_nodes()
        else:
             # Could not determine graph structure
             print(f"Label {label}: Could not get graph structure from skan object.")
             total_nodes = 0
        # --- END CONVERSION ---


        if total_nodes > 0 and nx_graph is not None:
            # Calculate degrees using the networkx graph
            degrees = pd.Series(dict(nx_graph.degree()))

            # Endpoints are nodes with degree 1
            n_endpoints = (degrees == 1).sum()

            # Junctions: Try direct attribute first (older skan), fallback to degrees
            if hasattr(graph_obj, 'n_junction'):
                 n_junctions = graph_obj.n_junction
            elif hasattr(graph_obj, 'n_junctions'):
                 n_junctions = graph_obj.n_junctions
            else:
                # Fallback: Junctions are nodes with degree > 2
                n_junctions = (degrees > 2).sum()

    except ValueError as ve:
        print(f"Label {label}: Skan ValueError (possibly trivial skeleton): {ve}")
        # If graph_obj was created, try to get basic node info using conversion
        if graph_obj is not None and hasattr(graph_obj, 'graph'):
             try:
                 nx_graph = None
                 if isinstance(graph_obj.graph, csr_matrix):
                     nx_graph = nx.from_scipy_sparse_array(graph_obj.graph)
                 elif isinstance(graph_obj.graph, nx.Graph):
                     nx_graph = graph_obj.graph

                 if nx_graph is not None:
                     total_nodes = nx_graph.number_of_nodes()
                     if total_nodes > 0:
                         degrees = pd.Series(dict(nx_graph.degree()))
                         n_endpoints = (degrees == 1).sum()
                         # Try direct attribute first for junctions in error case too
                         if hasattr(graph_obj, 'n_junction'): n_junctions = graph_obj.n_junction
                         elif hasattr(graph_obj, 'n_junctions'): n_junctions = graph_obj.n_junctions
                         else: n_junctions = (degrees > 2).sum() # Fallback
             except Exception as graph_err:
                  print(f"Label {label}: Failed to analyze graph even in error handler: {graph_err}")
                  pass # Ignore if getting degrees also fails

    except AttributeError as ae:
         # Catch potential AttributeErrors if graph conversion/analysis fails
         print(f"Label {label}: Skan AttributeError during analysis (API mismatch?): {ae}")
         # Return the skeleton image but empty data
         return pd.DataFrame(), skeleton_img, 0, 0, 0, None

    except Exception as e:
        print(f"Label {label}: Skan analysis failed with unexpected error: {e}")
        print(traceback.format_exc()) # Print full traceback for unexpected errors
        # Return the skeleton image but empty data
        return pd.DataFrame(), skeleton_img, 0, 0, 0, None

    return branch_data, skeleton_img, n_junctions, n_endpoints, total_nodes, graph_obj

def calculate_ramification_with_skan(segmented_array, spacing=(1.0, 1.0, 1.0), labels=None, skeleton_export_path=None):
    """
    Calculate ramification statistics using skan for specified labels.

    Returns:
    --------
    ramification_summary_df : pandas.DataFrame
        Summary stats per label (length, branches, junctions, etc.).
    detailed_branch_df : pandas.DataFrame
        Detailed branch info from skan for all processed labels.
    skeleton_array : numpy.ndarray or numpy.memmap
        Array containing the skeletons, labeled with original mask labels.
    """
    if labels is None:
        unique_labels = np.unique(segmented_array)
        labels_to_process = unique_labels[unique_labels > 0]
    else:
        labels_to_process = labels

    if not labels_to_process:
         print("No labels provided or found for skeletonization.")
         return pd.DataFrame(), pd.DataFrame(), np.zeros_like(segmented_array)


    # Prepare skeleton array (in memory or memmap)
    use_memmap = bool(skeleton_export_path)
    if use_memmap:
        print(f"Exporting skeletons to memory-mapped file: {skeleton_export_path}")
        if os.path.exists(skeleton_export_path):
             os.remove(skeleton_export_path) # Ensure fresh file
        skeleton_array = np.memmap(
            skeleton_export_path,
            dtype=segmented_array.dtype,
            mode='w+',
            shape=segmented_array.shape
        )
        skeleton_array[:] = 0 # Initialize
    else:
        skeleton_array = np.zeros_like(segmented_array)

    all_branch_data_dfs = []
    summary_stats_list = []

    # Process labels one by one (more memory efficient than loading all at once)
    for label in tqdm(labels_to_process, desc="Analyzing Skeletons (skan)"):
        mask = (segmented_array == label)

        branch_data, label_skeleton_img, n_junctions, n_endpoints, total_nodes, graph_obj = analyze_skeleton_with_skan(mask, spacing, label)

        # Store the skeleton image regardless of analysis success (if it exists)
        if label_skeleton_img is not None and np.any(label_skeleton_img):
            skeleton_array[label_skeleton_img > 0] = label
            if use_memmap:
                skeleton_array.flush() # Ensure data is written for memmap

        # Compile statistics if skan analysis produced results
        total_length = 0
        num_branches = 0
        avg_branch_length = np.nan
        # avg_branch_thickness = np.nan # Example if needed, from skan

        if not branch_data.empty:
            branch_data['label'] = label
            all_branch_data_dfs.append(branch_data)

            total_length = branch_data['branch-distance'].sum()
            num_branches = len(branch_data)
            avg_branch_length = branch_data['branch-distance'].mean()
            # avg_branch_thickness = branch_data['mean_thickness'].mean() # Requires source_image in Skeleton

        summary_stats_list.append({
            'label': label,
            'skan_num_branches': num_branches,
            'skan_total_length_um': total_length,
            'skan_avg_branch_length_um': avg_branch_length,
            'skan_num_junctions': n_junctions,
            'skan_num_endpoints': n_endpoints, # Now directly calculated/approximated
            'skan_num_skeleton_voxels': np.sum(label_skeleton_img > 0) if label_skeleton_img is not None else 0,
            # 'skan_avg_branch_thickness_um': avg_branch_thickness,
        })

        # Explicitly delete large objects to potentially free memory sooner
        del mask, branch_data, label_skeleton_img, graph_obj
        gc.collect()

    # Combine detailed branch data
    detailed_branch_df = pd.concat(all_branch_data_dfs, ignore_index=True) if all_branch_data_dfs else pd.DataFrame()

    # Create summary DataFrame
    ramification_summary_df = pd.DataFrame(summary_stats_list)

    return ramification_summary_df, detailed_branch_df, skeleton_array


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_segmentation(segmented_array, spacing=(1.0, 1.0, 1.0),
                         calculate_distances=True,
                         calculate_skeletons=True,
                         skeleton_export_path=None,
                         temp_dir=None, n_jobs=None,
                         return_detailed=False):
    """
    Comprehensive analysis of a 3D segmented array with consolidated outputs.

    Uses 'skan' for robust skeleton analysis.

    Parameters:
    -----------
    segmented_array : numpy.ndarray
        3D numpy array (z, x, y) containing integer labels. 0 is background.
    spacing : tuple
        Voxel spacing (z, x, y) in microns.
    calculate_distances : bool
        Whether to calculate shortest distances between masks.
    calculate_skeletons : bool
        Whether to calculate skeleton-based metrics using skan.
    skeleton_export_path : str, optional
        Path to export memory-mapped skeleton array if calculate_skeletons=True.
    temp_dir : str, optional
        Directory for temporary files (used for distances). If None, a temporary
        one is created and deleted.
    n_jobs : int, optional
        Number of parallel jobs for distance calculation. Defaults to CPU count - 1.
    return_detailed : bool
        If True, return detailed outputs (distance matrix, branch data, etc.).

    Returns:
    --------
    metrics_df : pandas.DataFrame
        Primary DataFrame with summary metrics per mask label, including:
        - Volume, surface area, sphericity, voxel count, bounding box volume
        - Median depth
        - (If distances calculated) Closest neighbor label, shortest distance,
          coordinates of the closest point pair.
        - (If skeletons calculated) skan summary: total length, # branches,
          # junctions, # endpoints, average branch length.
    detailed_outputs : dict
        Dictionary containing detailed DataFrames/arrays if return_detailed is True:
        - 'distance_matrix': Full pairwise distance DataFrame (if calculated).
        - 'all_pairs_points': DataFrame of closest points for all pairs (if calculated).
        - 'detailed_branches': DataFrame with detailed info on each skeleton branch from skan (if calculated).
        - 'skeleton_array': Numpy array (or memmap) of skeletons (if calculated).
    """
    overall_start_time = time.time()
    print("-" * 30)
    print("Starting Comprehensive Segmentation Analysis")
    print(f"Input array shape: {segmented_array.shape}")
    print(f"Voxel spacing (z,x,y): {spacing} um")
    print("-" * 30)


    labels = np.unique(segmented_array)
    if len(labels) == 0:
        print("No non-background labels found in the segmented array.")
        return pd.DataFrame(), {}

    print(f"Found {len(labels)} non-background labels.")

    # --- 1. Basic Volume & Depth Metrics ---
    print("\n[1/3] Calculating Volume & Depth Metrics...")
    volume_df = calculate_volume(segmented_array, spacing=spacing)
    depth_df = calculate_depth_df(segmented_array, spacing=spacing)

    # Start main metrics dataframe
    if not volume_df.empty:
         metrics_df = pd.merge(volume_df, depth_df, on='label', how='left')
    elif not depth_df.empty: # Handle cases where volume might fail?
         metrics_df = depth_df
         # Add empty volume cols? Or handle upstream? For now, assume volume_df is primary.
    else:
        print("Warning: Could not calculate volume or depth metrics.")
        metrics_df = pd.DataFrame({'label': labels}) # Basic df with just labels

    # Initialize dictionary for detailed outputs
    detailed_outputs = {}

    # --- 2. Distance Metrics ---
    if calculate_distances:
        print("\n[2/3] Calculating Pairwise Distances...")
        if len(labels) > 1:
            distance_matrix_df, all_pairs_points_df = shortest_distance(
                segmented_array, spacing=spacing, temp_dir=temp_dir, n_jobs=n_jobs
            )

            # Find closest neighbor and distance for each mask
            closest_neighbor_labels = []
            shortest_distances = []
            point_self_z, point_self_x, point_self_y = [], [], []
            point_neigh_z, point_neigh_x, point_neigh_y = [], [], []

            # Ensure matrix index/columns match the labels list for correct indexing
            distance_matrix_df = distance_matrix_df.reindex(index=labels, columns=labels)
            dist_values = distance_matrix_df.values

            for i, label in enumerate(labels):
                row_dists = dist_values[i, :].copy()
                row_dists[i] = np.inf # Ignore self-distance

                if np.all(np.isinf(row_dists)): # Handle masks with no neighbors
                    min_idx = -1 # Indicate no neighbor found
                    shortest_dist = np.inf
                    closest_neighbor = None
                else:
                    min_idx = np.nanargmin(row_dists) # Use nanargmin to handle potential infs
                    shortest_dist = row_dists[min_idx]
                    closest_neighbor = labels[min_idx]

                closest_neighbor_labels.append(closest_neighbor)
                shortest_distances.append(shortest_dist)

                # Find corresponding points from all_pairs_points_df
                if closest_neighbor is not None:
                    # Find the row for this specific pair (label, closest_neighbor)
                    pair_row = all_pairs_points_df[
                        (all_pairs_points_df['mask1'] == label) &
                        (all_pairs_points_df['mask2'] == closest_neighbor)
                    ]
                    if not pair_row.empty:
                        point_self_z.append(pair_row.iloc[0]['mask1_z'])
                        point_self_x.append(pair_row.iloc[0]['mask1_x'])
                        point_self_y.append(pair_row.iloc[0]['mask1_y'])
                        point_neigh_z.append(pair_row.iloc[0]['mask2_z'])
                        point_neigh_x.append(pair_row.iloc[0]['mask2_x'])
                        point_neigh_y.append(pair_row.iloc[0]['mask2_y'])
                    else: # Should not happen if distance was found, but fallback
                        point_self_z.append(np.nan); point_self_x.append(np.nan); point_self_y.append(np.nan)
                        point_neigh_z.append(np.nan); point_neigh_x.append(np.nan); point_neigh_y.append(np.nan)
                else:
                    point_self_z.append(np.nan); point_self_x.append(np.nan); point_self_y.append(np.nan)
                    point_neigh_z.append(np.nan); point_neigh_x.append(np.nan); point_neigh_y.append(np.nan)

            # Add distance info to metrics_df
            dist_info_df = pd.DataFrame({
                'label': labels,
                'closest_neighbor_label': closest_neighbor_labels,
                'shortest_distance_um': shortest_distances,
                'point_on_self_z': point_self_z, 'point_on_self_x': point_self_x, 'point_on_self_y': point_self_y,
                'point_on_neighbor_z': point_neigh_z, 'point_on_neighbor_x': point_neigh_x, 'point_on_neighbor_y': point_neigh_y
            })
            metrics_df = pd.merge(metrics_df, dist_info_df, on='label', how='left')


            if return_detailed:
                detailed_outputs['distance_matrix'] = distance_matrix_df
                detailed_outputs['all_pairs_points'] = all_pairs_points_df
        else:
            print("Skipping distance calculation (only 0 or 1 mask found).")
            # Add empty columns if they don't exist
            for col in ['closest_neighbor_label', 'shortest_distance_um',
                         'point_on_self_z', 'point_on_self_x', 'point_on_self_y',
                         'point_on_neighbor_z', 'point_on_neighbor_x', 'point_on_neighbor_y']:
                 if col not in metrics_df.columns:
                     metrics_df[col] = np.nan if 'point' in col else (np.inf if 'dist' in col else None)

            if return_detailed:
                 detailed_outputs['distance_matrix'] = pd.DataFrame(index=labels, columns=labels) if len(labels)==1 else pd.DataFrame()
                 detailed_outputs['all_pairs_points'] = pd.DataFrame()

    else:
         print("Skipping distance calculation as requested.")


    # --- 3. Skeleton Metrics (using skan) ---
    detailed_outputs['skeleton_array'] = None # Initialize even if not calculated
    detailed_outputs['detailed_branches'] = None

    if calculate_skeletons:
        print("\n[3/3] Calculating Skeleton Metrics (skan)...")
        # Pass only labels that exist in our current metrics_df
        labels_for_skeleton = list(metrics_df['label'].unique())

        ramification_summary_df, detailed_branch_df, skeleton_array = calculate_ramification_with_skan(
            segmented_array,
            spacing=spacing,
            labels=labels_for_skeleton,
            skeleton_export_path=skeleton_export_path
        )

        # Merge summary skeleton stats
        if not ramification_summary_df.empty:
             metrics_df = pd.merge(metrics_df, ramification_summary_df, on='label', how='left')
        else:
             # Add empty columns if skeletonization failed for all
             skel_cols = ['skan_num_branches', 'skan_total_length_um', 'skan_avg_branch_length_um',
                          'skan_num_junctions', 'skan_num_endpoints', 'skan_num_skeleton_voxels']
             for col in skel_cols:
                 if col not in metrics_df.columns:
                     metrics_df[col] = np.nan

        if return_detailed:
            detailed_outputs['detailed_branches'] = detailed_branch_df
            detailed_outputs['skeleton_array'] = skeleton_array # This is the actual array/memmap
    else:
        print("Skipping skeleton calculation as requested.")


    # Reorder columns for clarity, putting label first
    cols = ['label'] + [col for col in metrics_df if col != 'label']
    metrics_df = metrics_df[cols]

    # Final check for NaNs or Infs introduced
    metrics_df = metrics_df.fillna(value=np.nan) # Ensure NaNs are consistent

    print("-" * 30)
    print(f"Analysis completed in {time.time() - overall_start_time:.2f} seconds.")
    print("-" * 30)

    # Return based on return_detailed flag
    if return_detailed:
        return metrics_df, detailed_outputs
    else:
        # Clean up detailed outputs if not returned, especially memmap
        if isinstance(detailed_outputs.get('skeleton_array'), np.memmap):
            skel_path = detailed_outputs['skeleton_array'].filename
            del detailed_outputs['skeleton_array'] # Delete memmap object
            gc.collect() # Try to force collection
            time.sleep(0.1) # Short pause
            try:
                if os.path.exists(skel_path):
                    os.remove(skel_path)
                    print(f"Cleaned up skeleton memmap file: {skel_path}")
            except Exception as e:
                print(f"Warning: Could not delete skeleton memmap file {skel_path}: {e}")

        return metrics_df, {} # Return empty dict for detailed outputs