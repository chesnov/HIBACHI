# --- START OF FILE calculate_features_2d.py ---

# --- START OF FILE utils/ramified_module_2d/calculate_features_2d.py ---

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy import ndimage as ndi # For find_objects
import multiprocessing as mp
import os
import tempfile
import time
import psutil
import gc
from skimage.morphology import skeletonize, binary_erosion, binary_dilation, disk # type: ignore
from skimage.measure import regionprops, find_contours # type: ignore
from skan import Skeleton, summarize # type: ignore
import networkx as nx # type: ignore
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
import traceback
import math # For pi
from skan.csr import skeleton_to_csgraph # type: ignore
import networkx as nx # type: ignore
from skimage.graph import route_through_array # type: ignore
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy import ndimage

seed = 42
np.random.seed(seed)         # For NumPy

# --- DEBUG CONFIG ---
DEBUG_TARGET_LABEL = None # Set to a specific integer label ID to focus detailed prints, or None
# --- END DEBUG CONFIG ---

# Helper function for conditional printing
def debug_print(msg, label=None):
    if DEBUG_TARGET_LABEL is None or label == DEBUG_TARGET_LABEL:
        print(f"  DEBUG: {msg}")

# =============================================================================
# Distance Calculation Functions (2D Adapted)
# =============================================================================

# --- Reworked surface extraction worker for 2D regions ---
def extract_boundary_points_worker_2d(args):
    """Worker function to extract boundary points from a 2D SUB-REGION."""
    label_idx, label, region, temp_dir, spacing_yx, offset_yx = args
    # debug_print(f"[Worker {os.getpid()}] Extracting boundary for label {label} in region {region.shape} w/ offset {offset_yx}", label=label)

    # Create mask WITHIN the region
    mask = (region == label)
    if not np.any(mask):
        boundary_file = os.path.join(temp_dir, f"boundary_{label}.npy")
        np.save(boundary_file, np.empty((0, 2), dtype=int)) # Save empty (N, 2) array
        return label_idx, 0

    if label == DEBUG_TARGET_LABEL:
        debug_print(f"[Worker {os.getpid()}] Label {label}: Region shape {region.shape}, Offset {offset_yx}", label=label)

    # Find boundary pixels WITHIN the region using erosion
    # Use a simple 3x3 square connectivity for erosion
    struct = ndi.generate_binary_structure(2, 1) # 4-connectivity is usually enough for boundary
    eroded_mask = binary_erosion(mask, struct)
    boundary = mask & ~eroded_mask

    y_local, x_local = np.where(boundary)

    # Convert LOCAL coordinates back to GLOBAL coordinates using the offset
    boundary_points_global = np.column_stack((y_local + offset_yx[0], x_local + offset_yx[1]))

    if label == DEBUG_TARGET_LABEL:
        debug_print(f"[Worker {os.getpid()}] Label {label}: Found {len(boundary_points_global)} boundary points. First 5 GLOBAL points (Y,X):\n{boundary_points_global[:5]}", label=label)

    boundary_file = os.path.join(temp_dir, f"boundary_{label}.npy")
    np.save(boundary_file, boundary_points_global) # Save GLOBAL coordinates (Y, X)

    return label_idx, len(boundary_points_global)


def calculate_pair_distances_worker_2d(args):
    """Worker function to calculate distances between a pair of 2D masks."""
    i, j, labels, temp_dir, spacing_yx, distances_matrix, points_matrix = args

    label1 = labels[i]
    label2 = labels[j]
    is_target_pair = (DEBUG_TARGET_LABEL is not None and (label1 == DEBUG_TARGET_LABEL or label2 == DEBUG_TARGET_LABEL))

    try:
        boundary_points1 = np.load(os.path.join(temp_dir, f"boundary_{label1}.npy"))
        boundary_points2 = np.load(os.path.join(temp_dir, f"boundary_{label2}.npy"))
    except Exception as e:
        print(f"Warning: Error loading boundary points for masks {label1} or {label2}: {e}")
        return i, j, np.inf, np.full(4, np.nan) # Return 4 NaNs for Y1,X1,Y2,X2

    if boundary_points1.shape[0] == 0 or boundary_points2.shape[0] == 0:
        return i, j, np.inf, np.full(4, np.nan)

    # Scale points according to 2D spacing
    scaled_points1 = boundary_points1 * np.array(spacing_yx)
    scaled_points2 = boundary_points2 * np.array(spacing_yx)

    try:
        tree2 = cKDTree(scaled_points2)
    except ValueError:
         return i, j, np.inf, np.full(4, np.nan)

    distances, indices = tree2.query(scaled_points1, k=1)
    
    if distances.size == 0: # Handle empty distances array if scaled_points1 was empty after all
        return i, j, np.inf, np.full(4, np.nan)

    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    # ORIGINAL PIXEL INDICES (Y, X)
    point1 = boundary_points1[min_idx]
    point2 = boundary_points2[indices[min_idx]]

    result_points = np.array([point1[0], point1[1], point2[0], point2[1]]) # Y1, X1, Y2, X2

    if is_target_pair:
        debug_print(f"[Worker {os.getpid()}] Pair ({label1}, {label2}): Min dist {min_distance:.2f}. Closest point indices (Y,X): Point1={point1}, Point2={point2}", label=DEBUG_TARGET_LABEL)

    return i, j, min_distance, result_points


def shortest_distance_2d(segmented_array, spacing_yx=(1.0, 1.0),
                         temp_dir=None, n_jobs=None,
                         batch_size=None):
    """
    Calculate the shortest distance between boundaries of all 2D masks pairs.
    """
    start_time = time.time()
    if segmented_array.ndim != 2: raise ValueError("Input must be 2D")

    if n_jobs is None: n_jobs = max(1, mp.cpu_count() - 1)

    temp_dir_managed = False
    if temp_dir is None:
        base_temp_dir = os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp"
        temp_dir = tempfile.mkdtemp(prefix="shortest_dist_2d_", dir=base_temp_dir)
        temp_dir_managed = True; print(f"DEBUG [shortest_distance_2d] Using managed temp dir: {temp_dir}")
    else: os.makedirs(temp_dir, exist_ok=True); print(f"DEBUG [shortest_distance_2d] Using provided temp dir: {temp_dir}")

    unique_labels = np.unique(segmented_array); labels = unique_labels[unique_labels > 0]; n_labels = len(labels)

    if n_labels <= 1:
        print("Need at least two masks to calculate 2D distances.")
        if temp_dir_managed and os.path.exists(temp_dir): 
            try: import shutil; shutil.rmtree(temp_dir) 
            except OSError: pass
        empty_df = pd.DataFrame(index=labels, columns=labels); empty_points = pd.DataFrame(columns=['mask1', 'mask2', 'mask1_y', 'mask1_x', 'mask2_y', 'mask2_x'])
        if n_labels == 1: empty_df.loc[labels[0], labels[0]] = 0.0
        return empty_df, empty_points

    print(f"DEBUG [shortest_distance_2d] Found {n_labels} masks. Calculating distances with {n_jobs} workers.")

    # --- Boundary Point Extraction (2D Optimized) ---
    print("DEBUG [shortest_distance_2d] Finding 2D object bounding boxes...")
    locations = ndi.find_objects(segmented_array, max_label=labels.max())
    print(f"DEBUG [shortest_distance_2d] Found {len(locations if locations is not None else [])} potential 2D bounding boxes.")

    extract_args = []; labels_found_in_locations = []
    if locations is None: locations = [] # Ensure locations is iterable

    for i, label in enumerate(labels):
        if label -1 < len(locations) and locations[label-1] is not None:
            slices = locations[label - 1]
            buffered_slices = []
            for s, max_dim in zip(slices, segmented_array.shape): # Use 2D shape
                start = max(0, s.start - 1); stop = min(max_dim, s.stop + 1)
                buffered_slices.append(slice(start, stop))
            buffered_slices = tuple(buffered_slices)
            try:
                 region = segmented_array[buffered_slices]
                 offset_yx = np.array([s.start for s in buffered_slices]) # 2D offset (Y, X)
                 extract_args.append((i, label, region, temp_dir, spacing_yx, offset_yx))
                 labels_found_in_locations.append(label)
            except Exception as e: print(f"Warning: Error processing slices for label {label}: {e}")
        else: # Create empty file
             boundary_file = os.path.join(temp_dir, f"boundary_{label}.npy"); np.save(boundary_file, np.empty((0, 2), dtype=int))

    print(f"DEBUG [shortest_distance_2d] Prepared {len(extract_args)} tasks for boundary extraction.")
    if extract_args:
        with mp.Pool(processes=n_jobs) as pool:
             list(tqdm(pool.imap_unordered(extract_boundary_points_worker_2d, extract_args), total=len(extract_args), desc="Extracting Boundaries"))

    # --- Pairwise Distance Calculation (2D) ---
    labels_to_process_distance = labels_found_in_locations; n_labels_processed = len(labels_to_process_distance)
    if n_labels_processed <= 1:
        print("Need >= 2 valid masks to calculate distances.")
        if temp_dir_managed and os.path.exists(temp_dir): 
            try: import shutil; shutil.rmtree(temp_dir) 
            except OSError: pass
        final_distance_df = pd.DataFrame(np.inf, index=labels, columns=labels); np.fill_diagonal(final_distance_df.values, 0)
        final_points_df = pd.DataFrame(columns=['mask1', 'mask2', 'mask1_y', 'mask1_x', 'mask2_y', 'mask2_x'])
        return final_distance_df, final_points_df

    print(f"DEBUG [shortest_distance_2d] Calculating pairwise distances for {n_labels_processed} valid labels...")
    distances_matrix = np.full((n_labels_processed, n_labels_processed), np.inf, dtype=np.float32)
    points_matrix = np.full((n_labels_processed, n_labels_processed, 4), np.nan, dtype=np.float32) # Y1,X1,Y2,X2
    np.fill_diagonal(distances_matrix, 0)

    all_pairs = [(i, j) for i in range(n_labels_processed) for j in range(i + 1, n_labels_processed)]
    calc_args = [(i, j, labels_to_process_distance, temp_dir, spacing_yx, distances_matrix, points_matrix) for i, j in all_pairs]

    if calc_args:
        with mp.Pool(processes=n_jobs) as pool:
            results = list(tqdm(pool.imap_unordered(calculate_pair_distances_worker_2d, calc_args), total=len(all_pairs), desc="Calculating Distances (2D)"))

        for i_res, j_res, distance, points_res in results: # Renamed loop variables
            distances_matrix[i_res, j_res] = distances_matrix[j_res, i_res] = distance
            points_matrix[i_res, j_res, :] = points_res
            if not np.any(np.isnan(points_res)): points_matrix[j_res, i_res, :] = np.array([points_res[2], points_res[3], points_res[0], points_res[1]]) # Swap Y2,X2,Y1,X1

    # --- Convert to DataFrames (2D) ---
    distance_matrix_processed_df = pd.DataFrame(distances_matrix, index=labels_to_process_distance, columns=labels_to_process_distance)
    points_flat = points_matrix.reshape(-1, 4)
    mask1_labels_proc = np.repeat(labels_to_process_distance, n_labels_processed)
    mask2_labels_proc = np.tile(labels_to_process_distance, n_labels_processed)

    points_processed_df = pd.DataFrame({
        'mask1': mask1_labels_proc, 'mask2': mask2_labels_proc,
        'mask1_y': points_flat[:, 0], 'mask1_x': points_flat[:, 1],
        'mask2_y': points_flat[:, 2], 'mask2_x': points_flat[:, 3] # Correct column names
    })
    points_processed_df = points_processed_df[points_processed_df['mask1'] != points_processed_df['mask2']].dropna().reset_index(drop=True)

    # Expand distance matrix to original full label set
    final_distance_df = pd.DataFrame(np.inf, index=labels, columns=labels, dtype=np.float32)
    np.fill_diagonal(final_distance_df.values, 0)
    if not distance_matrix_processed_df.empty:
        final_distance_df.loc[labels_to_process_distance, labels_to_process_distance] = distance_matrix_processed_df

    final_points_df = points_processed_df # Return only calculated pairs

    # --- DEBUG ---
    print(f"DEBUG [shortest_distance_2d] Finished. Distance matrix shape {final_distance_df.shape}. Points DF shape {final_points_df.shape}")
    if DEBUG_TARGET_LABEL is not None and DEBUG_TARGET_LABEL in final_points_df['mask1'].values: debug_print(f"Points DF sample for label {DEBUG_TARGET_LABEL}:\n{final_points_df[final_points_df['mask1'] == DEBUG_TARGET_LABEL].head()}", label=DEBUG_TARGET_LABEL)
    elif DEBUG_TARGET_LABEL is None and not final_points_df.empty : print(f"DEBUG [shortest_distance_2d] Points DF head:\n{final_points_df.head()}")

    # --- Cleanup ---
    if temp_dir_managed and os.path.exists(temp_dir): 
        try: import shutil; shutil.rmtree(temp_dir) 
        except Exception as e: print(f"Warning: cleanup failed {temp_dir}: {e}")

    elapsed = time.time() - start_time; print(f"DEBUG [shortest_distance_2d] Distance calculation finished in {elapsed:.1f} seconds")
    return final_distance_df, final_points_df


# =============================================================================
# Area and Basic Shape Metrics (2D Adapted)
# =============================================================================

def calculate_area_and_shape_2d(segmented_array, spacing_yx=(1.0, 1.0)):
    """
    Calculate area and basic 2D shape metrics for each mask using regionprops.
    CORRECTED to handle anisotropic spacing for perimeter calculation.
    """
    print("DEBUG [calculate_area_shape_2d] Starting area/shape calculations...")
    if segmented_array.ndim != 2: raise ValueError("Input must be 2D")

    results = []
    print("DEBUG [calculate_area_shape_2d] Calculating regionprops...")
    try:
        # We still use regionprops for all other metrics, as they work correctly.
        # Spacing is passed so that props like major_axis_length are in physical units.
        props = regionprops(segmented_array.astype(np.int32), spacing=spacing_yx, intensity_image=None)
    except Exception as e:
        print(f"Error calculating regionprops: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    print(f"DEBUG [calculate_area_shape_2d] Found {len(props)} regions.")

    pixel_area_um2 = spacing_yx[0] * spacing_yx[1]
    # No longer need avg_spacing_um

    for prop in tqdm(props, desc="Processing Regions"):
        label = prop.label
        
        # These properties are calculated correctly by regionprops even with anisotropy
        pixel_count = prop.area
        area_um2 = pixel_count * pixel_area_um2
        min_r, min_c, max_r, max_c = prop.bbox
        bbox_height_px = max_r - min_r
        bbox_width_px = max_c - min_c
        bbox_area_um2 = (bbox_height_px * spacing_yx[0]) * (bbox_width_px * spacing_yx[1])
        eccentricity = prop.eccentricity
        solidity = prop.solidity
        major_axis_um = prop.major_axis_length
        minor_axis_um = prop.minor_axis_length

        # --- CORRECTED PERIMETER CALCULATION ---
        perimeter_um = 0.0
        try:
            # prop.image is the mask in its bounding box. Pad it to ensure contours are closed.
            padded_mask = np.pad(prop.image, pad_width=1, mode='constant', constant_values=0)
            
            # Find contours of the object. The level must be between 0 and 1.
            contours = find_contours(padded_mask, level=0.5)
            
            # A single object can have multiple contours (e.g., if it has holes)
            for contour in contours:
                # Calculate the difference between consecutive contour points
                delta = np.diff(contour, axis=0)
                # Scale these differences by the physical spacing
                delta_scaled = delta * np.array(spacing_yx)
                # Calculate the hypotenuse of each segment and sum them
                perimeter_um += np.sum(np.sqrt(np.sum(delta_scaled**2, axis=1)))
        except Exception as e_contour:
            print(f"Warning: Could not calculate perimeter for label {label}: {e_contour}")
            perimeter_um = np.nan
        # --- END OF CORRECTION ---

        # Circularity calculation now uses the new, more accurate perimeter
        circularity = np.nan
        if perimeter_um > 1e-6:
            circularity = (4 * math.pi * area_um2) / (perimeter_um**2)
            # Clamp to a realistic range [0, 1]
            circularity = min(1.0, max(0.0, circularity))

        results.append({
            'label': label,
            'area_um2': area_um2,
            'perimeter_um': perimeter_um, # Now correctly calculated
            'pixel_count': pixel_count,
            'bounding_box_area_um2': bbox_area_um2,
            'circularity': circularity, # Now correctly calculated
            'eccentricity': eccentricity,
            'solidity': solidity,
            'major_axis_length_um': major_axis_um,
            'minor_axis_length_um': minor_axis_um,
            'bbox_y_min': min_r, 'bbox_x_min': min_c,
            'bbox_y_max': max_r, 'bbox_x_max': max_c
        })

    print(f"DEBUG [calculate_area_shape_2d] Finished area/shape calculations for {len(results)} labels.")
    return pd.DataFrame(results)


# =============================================================================
# Skeletonization and Ramification (2D Adapted using skan)
# =============================================================================
def analyze_skeleton_with_skan_2d(label_region, offset_yx, spacing_yx, label, prune_spurs_le_um=0.0):
    """
    FIXED version that correctly counts biological branches.
    """
    branch_data = pd.DataFrame()
    skeleton_img_to_return_local = np.zeros_like(label_region, dtype=np.uint8)

    try:
        mask_local = (label_region == label)
        if not np.any(mask_local):
            return branch_data, skeleton_img_to_return_local, 0, 0, 0, 0.0

        skeleton_binary_local = skeletonize(mask_local)
        
        # Apply pruning
        if np.any(skeleton_binary_local):
            if prune_spurs_le_um > 0.0:
                skeleton_binary_local = _prune_skeleton_spurs(skeleton_binary_local, spacing_yx, prune_spurs_le_um)
            
            if np.any(skeleton_binary_local):
                skeleton_binary_local = _prune_internal_artifacts_custom(skeleton_binary_local)

        skeleton_img_to_return_local = skeleton_binary_local.astype(np.uint8)

        if not np.any(skeleton_img_to_return_local):
            return branch_data, skeleton_img_to_return_local, 0, 0, 0, 0.0

        # GET CORRECT TOPOLOGY AND MEASUREMENTS
        true_branches, n_junctions, n_endpoints, total_length = analyze_skeleton_topology_correctly(
            skeleton_img_to_return_local, spacing_yx)

        # Still get skan data for detailed analysis (but ignore its branch count)
        try:
            graph_obj = Skeleton(skeleton_img_to_return_local, spacing=spacing_yx)
            branch_data = summarize(graph_obj, separator='-')

            # Shift coordinates to global space
            if not branch_data.empty:
                y_offset, x_offset = offset_yx[0], offset_yx[1]
                for col in branch_data.columns:
                    if 'coord' in col and col.endswith('-0'):
                        branch_data[col] += y_offset
                    elif 'coord' in col and col.endswith('-1'):
                        branch_data[col] += x_offset
        except:
            pass

        return branch_data, skeleton_img_to_return_local, true_branches, n_junctions, n_endpoints, total_length
                    
    except Exception as e:
        print(f"Label {label}: Analysis failed: {e}")
        return pd.DataFrame(), np.zeros_like(label_region, dtype=np.uint8), 0, 0, 0, 0.0
    

def _prune_internal_artifacts_custom(skeleton_binary):
    """
    A simpler fix that focuses specifically on the issue in your example.
    This version is more aggressive about removing pixels that don't contribute
    to the skeleton's essential structure.
    """
    pruned_skeleton = skeleton_binary.copy().astype(np.uint8)
    
    neighbor_kernel = np.array([[1, 1, 1],
                                [1, 0, 1],
                                [1, 1, 1]], dtype=np.uint8)
    
    max_iterations = 50
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        neighbor_count = ndimage.convolve(pruned_skeleton, neighbor_kernel, mode='constant', cval=0)
        candidates = np.argwhere((pruned_skeleton == 1) & (neighbor_count >= 2))
        
        if candidates.shape[0] == 0:
            break
        
        pixels_to_remove = []
        padded_skeleton = np.pad(pruned_skeleton, 1, mode='constant')
        
        for r, c in candidates:
            neighborhood = padded_skeleton[r:r+3, c:c+3].copy()
            
            # Remove center pixel temporarily
            neighborhood[1, 1] = 0
            
            # Check connectivity
            labeled, num_components = ndimage.label(neighborhood)
            
            # More aggressive removal conditions:
            # 1. If neighbors form 0 or 1 connected component, remove
            # 2. If the original pixel had exactly 2 neighbors, and they're still connected, remove
            original_neighbor_count = np.sum(neighborhood)  # After center removal
            
            if num_components <= 1:
                pixels_to_remove.append((r, c))
            elif num_components == 2 and original_neighbor_count == 2:
                # Special case: if we had exactly 2 neighbors and they're now 2 separate components,
                # this pixel was connecting them. But if they're very close (adjacent),
                # the connection might be redundant
                neighbor_positions = np.argwhere(neighborhood == 1)
                if len(neighbor_positions) == 2:
                    pos1, pos2 = neighbor_positions
                    # Check if they're adjacent (distance = 1 in any direction)
                    distance = np.abs(pos1 - pos2)
                    if np.max(distance) == 1:  # Adjacent neighbors
                        # This is likely a redundant connection
                        pixels_to_remove.append((r, c))
        
        if not pixels_to_remove:
            break
        
        for r, c in pixels_to_remove:
            pruned_skeleton[r, c] = 0
    
    return pruned_skeleton

def _prune_skeleton_spurs(skeleton_binary, spacing_yx, max_spur_length_um):
    """
    Remove short terminal branches (spurs) from a binary skeleton based on length filtering.
    
    This performs length-based filtering to remove branches shorter than the specified 
    threshold, focusing on the biologically significant longer structures.
    
    Parameters:
    -----------
    skeleton_binary : numpy.ndarray
        2D binary skeleton image
    spacing_yx : tuple
        (y, x) pixel spacing in physical units (e.g., microns)
    max_spur_length_um : float
        Maximum length of terminal branches to remove, in microns
    
    Returns:
    --------
    numpy.ndarray
        Length-filtered skeleton with short spurs removed
    """
    from scipy import ndimage
    import numpy as np
    
    if max_spur_length_um <= 0:
        return skeleton_binary
    
    # Work on a copy
    pruned_skeleton = skeleton_binary.copy()
    
    # Iteratively remove short terminal branches
    changed = True
    iteration = 0
    max_iterations = 50  # Safety limit to prevent infinite loops
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        # Find endpoints (pixels with exactly one neighbor)
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]], dtype=np.uint8)
        
        neighbor_count = ndimage.convolve(pruned_skeleton.astype(np.uint8), 
                                        kernel, mode='constant', cval=0)
        
        # Endpoints have exactly 1 neighbor
        endpoints = (pruned_skeleton == 1) & (neighbor_count == 1)
        
        if not np.any(endpoints):
            break
        
        # Process each endpoint
        endpoint_coords = np.argwhere(endpoints)
        
        for ep_y, ep_x in endpoint_coords:
            if pruned_skeleton[ep_y, ep_x] == 0:  # Already removed
                continue
                
            # Trace the spur length and path from this endpoint
            spur_length, spur_path = _trace_spur_length_and_path(pruned_skeleton, ep_y, ep_x, spacing_yx)
            
            # Remove spurs that are shorter than or equal to the threshold
            if spur_length <= max_spur_length_um:
                _remove_spur_path(pruned_skeleton, spur_path)
                changed = True
    
    return pruned_skeleton


def _trace_spur_length_and_path(skeleton, start_y, start_x, spacing_yx):
    """
    Trace a spur from an endpoint and calculate its length and path.
    
    Returns the length in physical units and the path coordinates until 
    we hit a junction or another endpoint.
    """
    if skeleton[start_y, start_x] == 0:
        return 0.0, []
    
    visited = set()
    current_y, current_x = start_y, start_x
    total_length = 0.0
    path = [(current_y, current_x)]
    
    # 8-connectivity offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
               (0, 1), (1, -1), (1, 0), (1, 1)]
    
    while True:
        visited.add((current_y, current_x))
        
        # Find unvisited neighbors
        neighbors = []
        for dy, dx in offsets:
            ny, nx = current_y + dy, current_x + dx
            if (0 <= ny < skeleton.shape[0] and 
                0 <= nx < skeleton.shape[1] and
                skeleton[ny, nx] == 1 and 
                (ny, nx) not in visited):
                neighbors.append((ny, nx))
        
        # If no unvisited neighbors, we've reached the end
        if len(neighbors) == 0:
            break
        
        # If more than one neighbor, we've hit a junction - stop here
        if len(neighbors) > 1:
            break
        
        # Move to the single neighbor
        next_y, next_x = neighbors[0]
        
        # Calculate distance to next point
        dy = (next_y - current_y) * spacing_yx[0]
        dx = (next_x - current_x) * spacing_yx[1]
        distance = np.sqrt(dy**2 + dx**2)
        total_length += distance
        
        current_y, current_x = next_y, next_x
        path.append((current_y, current_x))
    
    return total_length, path


def _remove_spur_path(skeleton, path):
    """
    Remove pixels along the given path, stopping before junctions.
    """
    # Remove all pixels in the path except the last one if it's a junction
    for i, (y, x) in enumerate(path):
        # Check if this is the last pixel and if it's a junction
        if i == len(path) - 1:
            # Count neighbors to see if it's a junction
            kernel = np.array([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]], dtype=np.uint8)
            neighbor_count = ndimage.convolve(skeleton.astype(np.uint8), 
                                            kernel, mode='constant', cval=0)[y, x]
            # If it has more than 1 neighbor (after removing the spur), it's a junction - keep it
            if neighbor_count > 1:
                break
        
        skeleton[y, x] = 0

def count_skeleton_topology(skeleton_img):
    """
    Count actual junctions and endpoints in a binary skeleton image.
    This gives you the TRUE topological features, not skan's computational segments.
    """
    if skeleton_img is None or not np.any(skeleton_img):
        return 0, 0
    
    # Convert to binary if needed
    skeleton_binary = (skeleton_img > 0).astype(np.uint8)
    
    # Count neighbors for each skeleton pixel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1], 
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = ndimage.convolve(skeleton_binary, kernel, mode='constant', cval=0)
    
    # Only consider pixels that are part of the skeleton
    skeleton_pixels = skeleton_binary == 1
    
    # Endpoints have exactly 1 neighbor
    endpoints = skeleton_pixels & (neighbor_count == 1)
    n_endpoints = np.sum(endpoints)
    
    # Junctions have 3 or more neighbors
    junctions = skeleton_pixels & (neighbor_count >= 3)
    n_junctions = np.sum(junctions)
    
    return n_junctions, n_endpoints

def analyze_skeleton_topology_correctly(skeleton_binary, spacing_yx=(1.0, 1.0)):
    """
    Correctly analyze skeleton topology by counting actual biological features,
    not skan's computational segments.
    
    Returns:
    - true_branches: actual number of biological branches
    - n_junctions: pixels with 3+ neighbors (true branch points)
    - n_endpoints: pixels with 1 neighbor (true endpoints)
    - total_length: total skeleton length from skan
    """
    if not np.any(skeleton_binary):
        return 0, 0, 0, 0.0
    
    # Step 1: Count true topological features
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    
    neighbor_count = ndimage.convolve(skeleton_binary.astype(np.uint8), kernel, mode='constant', cval=0)
    skeleton_mask = skeleton_binary > 0
    
    # True endpoints and junctions
    endpoints = skeleton_mask & (neighbor_count == 1)
    junctions = skeleton_mask & (neighbor_count >= 3)
    
    n_endpoints = np.sum(endpoints)
    n_junctions = np.sum(junctions)
    
    # Step 2: Calculate TRUE number of biological branches
    # This uses graph theory - for a connected planar graph:
    # branches = endpoints + (sum of excess connections at junctions)
    
    if n_junctions == 0:
        # No junctions
        if n_endpoints == 0:
            # Closed loop
            true_branches = 0
        elif n_endpoints == 2:
            # Simple unbranched structure (line/curve)
            true_branches = 0  # This is NOT a branch, it's a single segment
        else:
            # This shouldn't happen in a clean skeleton
            true_branches = 0
    else:
        # With junctions: each junction contributes (degree - 2) extra branches
        # Plus one branch for each endpoint
        junction_coords = np.argwhere(junctions)
        excess_branches = 0
        
        for jy, jx in junction_coords:
            junction_degree = neighbor_count[jy, jx]
            excess_branches += max(0, junction_degree - 2)
        
        true_branches = excess_branches
    
    # Step 3: Get total length from skan (this part skan does correctly)
    total_length = 0.0
    try:
        graph_obj = Skeleton(skeleton_binary.astype(np.uint8), spacing=spacing_yx)
        branch_data = summarize(graph_obj, separator='-')
        if not branch_data.empty:
            total_length = branch_data['branch-distance'].sum()
    except:
        # Fallback: estimate length by counting pixels
        skeleton_pixels = np.sum(skeleton_binary)
        avg_spacing = np.mean(spacing_yx)
        total_length = skeleton_pixels * avg_spacing
    
    return true_branches, n_junctions, n_endpoints, total_length

# Alternative: Use skan's graph structure directly
def count_topology_from_skan_graph(graph_obj):
    """
    Extract topology directly from skan's graph object.
    This is the most accurate method.
    """
    try:
        # Get the networkx graph from skan
        G = graph_obj.graph
        
        # Count nodes by their degree
        degrees = dict(G.degree())
        
        n_endpoints = sum(1 for degree in degrees.values() if degree == 1)
        n_junctions = sum(1 for degree in degrees.values() if degree >= 3)
        
        return n_junctions, n_endpoints
    except:
        return 0, 0


def calculate_ramification_with_skan_2d(segmented_array, spacing_yx=(1.0, 1.0), labels=None, 
                                                  skeleton_export_path=None, prune_spurs_le_um=None):
    """
    CORRECTED version that reports true biological branch statistics.
    """
    print("DEBUG [calculate_ramification_2d] Starting CORRECTED 2D skeleton analysis...")
    
    if segmented_array.ndim != 2: 
        raise ValueError("Input must be 2D")

    # Handle labels
    if labels is None:
        unique_labels_in_array = np.unique(segmented_array)
        labels_to_process = unique_labels_in_array[unique_labels_in_array > 0]
    else:
        unique_labels_in_array = np.unique(segmented_array)
        present_labels_set = set(unique_labels_in_array[unique_labels_in_array > 0])
        labels = [int(lbl) for lbl in labels]
        labels_to_process = sorted([lbl for lbl in labels if lbl in present_labels_set])
    
    if not labels_to_process:
        return pd.DataFrame(), pd.DataFrame(), np.zeros_like(segmented_array, dtype=np.uint8)

    print(f"Processing {len(labels_to_process)} labels...")
    
    # Get bounding boxes
    locations = ndimage.find_objects(segmented_array.astype(np.int32), 
                                   max_label=max(labels_to_process) if labels_to_process else 0)
    if locations is None: 
        locations = []

    # Prepare skeleton array
    use_memmap = bool(skeleton_export_path)
    max_label_skel = max(labels_to_process) if labels_to_process else 0
    
    if max_label_skel < 2**8: 
        skeleton_dtype = np.uint8
    elif max_label_skel < 2**16: 
        skeleton_dtype = np.uint16
    elif max_label_skel < 2**32: 
        skeleton_dtype = np.uint32
    else: 
        skeleton_dtype = np.uint64

    skeleton_array = None
    try:
        if use_memmap:
            if os.path.exists(skeleton_export_path): 
                os.remove(skeleton_export_path)
            skeleton_array = np.memmap(skeleton_export_path, dtype=skeleton_dtype, 
                                     mode='w+', shape=segmented_array.shape)
        else: 
            skeleton_array = np.zeros(segmented_array.shape, dtype=skeleton_dtype)
        skeleton_array[:] = 0
        if use_memmap: 
            skeleton_array.flush()
    except Exception as e:
        print(f"Error preparing skeleton array: {e}")
        return pd.DataFrame(), pd.DataFrame(), None

    all_branch_data_dfs = []
    summary_stats_list = []

    for label_val in labels_to_process:
        loc_index = label_val - 1
        if loc_index < 0 or loc_index >= len(locations) or locations[loc_index] is None:
            summary_stats_list.append({
                'label': label_val,
                'true_num_branches': 0,
                'skan_total_length_um': 0.0,
                'skan_avg_branch_length_um': np.nan,
                'true_num_junctions': 0,
                'true_num_endpoints': 0,
                'skan_num_skeleton_pixels': 0
            })
            continue
        
        slices = locations[loc_index]
        try:
            label_region = segmented_array[slices]
            offset_yx = np.array([s.start for s in slices])
        except Exception as e:
            print(f"Error slicing label {label_val}: {e}")
            continue

        # Use the CORRECTED analysis function
        branch_data, label_skeleton_img_local, true_branches, n_junctions, n_endpoints, total_length = \
            analyze_skeleton_with_skan_2d(label_region, offset_yx, spacing_yx, label_val, prune_spurs_le_um)

        # Calculate average branch length
        avg_branch_length = (total_length / true_branches) if true_branches > 0 else 0.0
        num_skel_pixels = np.count_nonzero(label_skeleton_img_local)

        # Store CORRECTED statistics
        summary_stats_list.append({
            'label': label_val,
            'true_num_branches': true_branches,  # This is the corrected count
            'skan_total_length_um': total_length,
            'skan_avg_branch_length_um': avg_branch_length,
            'true_num_junctions': n_junctions,   # True junction count
            'true_num_endpoints': n_endpoints,   # True endpoint count
            'skan_num_skeleton_pixels': num_skel_pixels
        })

        # Store detailed branch data if available
        if not branch_data.empty:
            branch_data['label'] = label_val
            all_branch_data_dfs.append(branch_data.copy())
        
        # Map skeleton to global array
        if label_skeleton_img_local is not None and np.any(label_skeleton_img_local):
            try:
                local_coords = np.argwhere(label_skeleton_img_local > 0)
                if local_coords.size > 0:
                    global_coords = local_coords + offset_yx
                    idx_y, idx_x = global_coords[:, 0], global_coords[:, 1]
                    valid_idx = ((idx_y >= 0) & (idx_y < skeleton_array.shape[0]) & 
                               (idx_x >= 0) & (idx_x < skeleton_array.shape[1]))
                    idx_y, idx_x = idx_y[valid_idx], idx_x[valid_idx]
                    skeleton_array[idx_y, idx_x] = label_val
                    if use_memmap: 
                        skeleton_array.flush()
            except Exception as e:
                print(f"Error mapping skeleton for label {label_val}: {e}")

    # Combine results
    detailed_branch_df = pd.DataFrame()
    if all_branch_data_dfs:
        try:
            detailed_branch_df = pd.concat(all_branch_data_dfs, ignore_index=True)
        except Exception as e:
            print(f"Error concatenating branch data: {e}")

    ramification_summary_df = pd.DataFrame(summary_stats_list)

    print(f"DEBUG [calculate_ramification_2d] Finished CORRECTED analysis.")
    return ramification_summary_df, detailed_branch_df, skeleton_array

# =============================================================================
# Main Analysis Function (2D Adapted)
# =============================================================================

def analyze_segmentation_2d(segmented_array, spacing_yx=(1.0, 1.0),
                            calculate_distances=True,
                            calculate_skeletons=True,
                            skeleton_export_path=None,
                            temp_dir=None, n_jobs=None,
                            return_detailed=False,
                            prune_spurs_le_um=0): # <-- ADDED PARAM
    """
    Comprehensive analysis of a 2D segmented array with optional smoothing/pruning.
    """
    overall_start_time = time.time()
    print("\n" + "=" * 30)
    print("DEBUG [analyze_segmentation_2d] Starting Comprehensive 2D Analysis")
    if segmented_array.ndim != 2: raise ValueError("Input must be 2D")
    print(f"DEBUG [analyze_segmentation_2d] Input array shape: {segmented_array.shape}, dtype: {segmented_array.dtype}")
    print(f"DEBUG [analyze_segmentation_2d] Pixel spacing (y,x): {spacing_yx} um")
    print("=" * 30)

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    labels = np.unique(segmented_array); labels = labels[labels > 0]
    if len(labels) == 0: print("DEBUG [analyze_segmentation_2d] No labels found."); return pd.DataFrame(), {}
    print(f"DEBUG [analyze_segmentation_2d] Found {len(labels)} labels.")

    # --- 1. Basic Area & Shape Metrics (2D) ---
    print("\nDEBUG [analyze_segmentation_2d] [1/3] Calculating Area & Shape Metrics...")
    area_shape_df = calculate_area_and_shape_2d(segmented_array, spacing_yx=spacing_yx)
    metrics_df = area_shape_df # Start with these metrics
    print(f"DEBUG [analyze_segmentation_2d] Initial metrics_df shape after area/shape: {metrics_df.shape if not metrics_df.empty else 'Empty'}")

    detailed_outputs = {}

    if prune_spurs_le_um == 0.0:
        prune_spurs_le_um = None # Treat 0 as no smoothing

    # --- 2. Distance Metrics (2D) ---
    if calculate_distances:
        print("\nDEBUG [analyze_segmentation_2d] [2/3] Calculating Pairwise Distances (2D)...")
        if len(labels) > 1:
            distance_matrix_df, all_pairs_points_df = shortest_distance_2d(
                segmented_array, spacing_yx=spacing_yx, temp_dir=temp_dir, n_jobs=n_jobs
            )
            closest_neighbor_labels = []; shortest_distances = []
            point_self_y, point_self_x = [], []
            point_neigh_y, point_neigh_x = [], []

            if not distance_matrix_df.empty:
                distance_matrix_df = distance_matrix_df.reindex(index=labels, columns=labels).fillna(np.inf) # Ensure all labels present
                dist_values = distance_matrix_df.values

                for i, label_val in enumerate(labels): # Renamed label
                    # Find index of label_val in distance_matrix_df.index, as labels might not be contiguous or sorted same way
                    try:
                        label_idx_in_matrix = distance_matrix_df.index.get_loc(label_val)
                    except KeyError: # Label not in matrix (e.g., if it had no boundary points)
                        closest_neighbor_labels.append(None)
                        shortest_distances.append(np.inf)
                        point_self_y.append(np.nan); point_self_x.append(np.nan)
                        point_neigh_y.append(np.nan); point_neigh_x.append(np.nan)
                        continue

                    row_dists = dist_values[label_idx_in_matrix, :].copy()
                    
                    # Ensure self-distance is Inf for min finding
                    try:
                        self_col_idx = distance_matrix_df.columns.get_loc(label_val)
                        row_dists[self_col_idx] = np.inf
                    except KeyError: pass # Should not happen if reindexed correctly

                    if np.all(np.isinf(row_dists)): 
                        min_col_idx_in_matrix=-1; shortest_dist=np.inf; closest_neighbor=None
                    else: 
                        min_col_idx_in_matrix=np.nanargmin(row_dists)
                        shortest_dist=row_dists[min_col_idx_in_matrix]
                        closest_neighbor=distance_matrix_df.columns[min_col_idx_in_matrix] # Get label from matrix columns
                    
                    closest_neighbor_labels.append(closest_neighbor); shortest_distances.append(shortest_dist)

                    psy, psx, pny, pnx = np.nan, np.nan, np.nan, np.nan 
                    if closest_neighbor is not None and not all_pairs_points_df.empty:
                        pair_row = all_pairs_points_df[ (all_pairs_points_df['mask1'] == label_val) & (all_pairs_points_df['mask2'] == closest_neighbor) ]
                        if not pair_row.empty:
                            psy=pair_row.iloc[0]['mask1_y']; psx=pair_row.iloc[0]['mask1_x']
                            pny=pair_row.iloc[0]['mask2_y']; pnx=pair_row.iloc[0]['mask2_x']
                    point_self_y.append(psy); point_self_x.append(psx); point_neigh_y.append(pny); point_neigh_x.append(pnx)
            else: # distance_matrix_df is empty
                 for _ in labels:
                    closest_neighbor_labels.append(None); shortest_distances.append(np.inf)
                    point_self_y.append(np.nan); point_self_x.append(np.nan); point_neigh_y.append(np.nan); point_neigh_x.append(np.nan)


            dist_info_df = pd.DataFrame({
                'label': labels, 'closest_neighbor_label': closest_neighbor_labels, 'shortest_distance_um': shortest_distances,
                'point_on_self_y': point_self_y, 'point_on_self_x': point_self_x, 
                'point_on_neighbor_y': point_neigh_y, 'point_on_neighbor_x': point_neigh_x 
            })
            if not metrics_df.empty:
                metrics_df = pd.merge(metrics_df, dist_info_df, on='label', how='left')
            else: # If metrics_df was empty (e.g. area_shape failed)
                metrics_df = dist_info_df

            print(f"DEBUG [analyze_segmentation_2d] metrics_df shape after distance merge: {metrics_df.shape if not metrics_df.empty else 'Empty'}")

            if return_detailed: detailed_outputs['distance_matrix'] = distance_matrix_df; detailed_outputs['all_pairs_points'] = all_pairs_points_df
        else: print("DEBUG [analyze_segmentation_2d] Skipping 2D distance calculation (<= 1 label).")
    else: print("DEBUG [analyze_segmentation_2d] Skipping distance calculation as requested.")

    # --- 3. Skeleton Metrics (2D skan) ---
    detailed_outputs['skeleton_array'] = None; detailed_outputs['detailed_branches'] = None

    if calculate_skeletons:
        print("\nDEBUG [analyze_segmentation_2d] [3/3] Calculating Skeleton Metrics (2D skan)...")
        labels_for_skeleton = list(metrics_df['label'].unique()) if not metrics_df.empty else list(labels)
        
        ramification_summary_df, detailed_branch_df, skeleton_array = calculate_ramification_with_skan_2d(
            segmented_array, spacing_yx=spacing_yx, labels=labels_for_skeleton, 
            skeleton_export_path=skeleton_export_path,
            prune_spurs_le_um=prune_spurs_le_um
        )
        if not ramification_summary_df.empty: 
            if not metrics_df.empty:
                metrics_df = pd.merge(metrics_df, ramification_summary_df, on='label', how='left')
            else: # If metrics_df was empty
                metrics_df = ramification_summary_df
        else: # Add empty columns if ram_summary is empty
            skel_cols_2d = ['skan_num_branches', 'skan_total_length_um', 'skan_avg_branch_length_um', 'skan_num_junctions', 'skan_num_endpoints', 'skan_num_skeleton_pixels']
            if not metrics_df.empty:
                for col in skel_cols_2d:
                    if col not in metrics_df.columns: metrics_df[col] = np.nan
            else: # If metrics_df is still empty, create it with labels and these NaN cols
                metrics_df = pd.DataFrame({'label': labels_for_skeleton})
                for col in skel_cols_2d: metrics_df[col] = np.nan


        if return_detailed:
            detailed_outputs['detailed_branches'] = detailed_branch_df; detailed_outputs['skeleton_array'] = skeleton_array
            if skeleton_array is not None: print(f"DEBUG [analyze_segmentation_2d] Final 2D skeleton_array returned: shape={skeleton_array.shape}, dtype={skeleton_array.dtype}")
            else: print("DEBUG [analyze_segmentation_2d] Final 2D skeleton_array is None.")
    else: print("DEBUG [analyze_segmentation_2d] Skipping skeleton calculation as requested.")

    # Reorder columns and fill NaNs
    if not metrics_df.empty:
        cols = ['label'] + [col for col in metrics_df if col != 'label']
        metrics_df = metrics_df[cols]
        metrics_df = metrics_df.fillna(value=np.nan)
    else: # If all calculations failed or were skipped, return empty DF with a label column
        metrics_df = pd.DataFrame({'label': labels})


    print("-" * 30); print(f"DEBUG [analyze_segmentation_2d] Final metrics_df head:"); print(metrics_df.head()); print("-" * 30)
    print(f"DEBUG [analyze_segmentation_2d] Analysis completed in {time.time() - overall_start_time:.2f} seconds.")
    print("=" * 30 + "\n")

    if return_detailed: return metrics_df, detailed_outputs
    else: # Cleanup memmap if not returned
        if isinstance(detailed_outputs.get('skeleton_array'), np.memmap):
            skel_path = detailed_outputs['skeleton_array'].filename; del detailed_outputs['skeleton_array']; gc.collect(); time.sleep(0.1)
            try:
                if os.path.exists(skel_path): os.remove(skel_path); print(f"DEBUG [analyze_segmentation_2d] Cleaned up skeleton memmap file: {skel_path}")
            except Exception as e: print(f"Warning: Could not delete skeleton memmap file {skel_path}: {e}")
        return metrics_df, {}

# --- END OF FILE utils/ramified_module_2d/calculate_features_2d.py ---