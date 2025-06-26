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
from skimage.measure import regionprops # type: ignore # Use regionprops for 2D metrics
from skan import Skeleton, summarize # type: ignore
import networkx as nx # type: ignore
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
import traceback
import math # For pi

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
    """Calculate area and basic 2D shape metrics for each mask using regionprops."""
    print("DEBUG [calculate_area_shape_2d] Starting area/shape calculations...")
    if segmented_array.ndim != 2: raise ValueError("Input must be 2D")

    results = []
    print("DEBUG [calculate_area_shape_2d] Calculating regionprops...")
    try:
        # Calculate props for all labeled regions at once
        # Provide spacing for accurate physical measurements
        props = regionprops(segmented_array.astype(np.int32), spacing=spacing_yx, intensity_image=None) # Ensure int for labels
    except Exception as e:
        print(f"Error calculating regionprops: {e}")
        traceback.print_exc()
        return pd.DataFrame()

    print(f"DEBUG [calculate_area_shape_2d] Found {len(props)} regions.")

    pixel_area_um2 = spacing_yx[0] * spacing_yx[1]
    avg_spacing_um = np.mean(spacing_yx)

    for prop in tqdm(props, desc="Processing Regions"):
        label = prop.label
        # Area
        pixel_count = prop.area # Number of pixels
        area_um2 = pixel_count * pixel_area_um2

        # Bounding Box
        min_r, min_c, max_r, max_c = prop.bbox
        bbox_height_px = max_r - min_r
        bbox_width_px = max_c - min_c
        bbox_area_um2 = (bbox_height_px * spacing_yx[0]) * (bbox_width_px * spacing_yx[1])

        # Perimeter (Note: regionprops perimeter is pixel count on border)
        # Convert to physical length approximately
        perimeter_px = prop.perimeter
        perimeter_um = perimeter_px * avg_spacing_um # Approximate physical perimeter

        # Circularity (Compactness) = 4 * pi * Area / Perimeter^2
        # Use physical units for calculation
        circularity = np.nan
        if perimeter_um > 1e-6:
            circularity = (4 * math.pi * area_um2) / (perimeter_um**2)
            circularity = min(1.0, max(0.0, circularity)) # Clamp to [0, 1]

        # Other potentially useful props (already in physical units if spacing provided)
        eccentricity = prop.eccentricity
        solidity = prop.solidity # Ratio of pixels in region to convex hull
        major_axis_um = prop.major_axis_length # Physical length
        minor_axis_um = prop.minor_axis_length # Physical length

        results.append({
            'label': label,
            'area_um2': area_um2,
            'perimeter_um': perimeter_um,
            'pixel_count': pixel_count,
            'bounding_box_area_um2': bbox_area_um2,
            'circularity': circularity,
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

def analyze_skeleton_with_skan_2d(label_region, offset_yx, spacing_yx, label):
    """
    Analyzes a single 2D binary mask using skan. (CORRECTED based on 3D logic)
    Returns DataFrame, skeleton image (local coords), and stats.
    Offset_yx is the (y,x) start of label_region in the original array.
    """
    is_target = (label == DEBUG_TARGET_LABEL)
    
    mask_local = None; skeleton_img_to_return_local = None; skel_pixel_count = 0
    branch_data = pd.DataFrame(); graph_obj = None
    n_junctions = 0; n_endpoints = 0; total_nodes = 0

    try:
        mask_local = (label_region == label)

        if not np.any(mask_local):
            return pd.DataFrame(), np.zeros_like(label_region, dtype=np.uint8), 0, 0, 0, None

        skeleton_binary_local = skeletonize(mask_local)
        skel_pixel_count = np.count_nonzero(skeleton_binary_local)
        
        # This is the binary (0/1) skeleton image that will be RETURNED. It has local coordinates.
        skeleton_img_to_return_local = skeleton_binary_local.astype(np.uint8)

        if skel_pixel_count == 0:
            return pd.DataFrame(), skeleton_img_to_return_local, 0, 0, 0, None

        # --- START OF SKAN COMPATIBILITY FIX (mirroring 3D logic) ---
        skeleton_global_coords = np.argwhere(skeleton_binary_local) + offset_yx # offset_yx is (y_offset, x_offset)
        source_mask_global_coords = np.argwhere(mask_local) + offset_yx

        if skeleton_global_coords.size == 0 or source_mask_global_coords.size == 0: # Should not happen if skel_pixel_count > 0
            return pd.DataFrame(), skeleton_img_to_return_local, 0, 0, 0, None

        max_skel_coords = np.max(skeleton_global_coords, axis=0) if skeleton_global_coords.size > 0 else np.array([0,0])
        max_mask_coords = np.max(source_mask_global_coords, axis=0) if source_mask_global_coords.size > 0 else np.array([0,0])
        
        required_shape_2d = np.maximum(max_skel_coords, max_mask_coords) + 1 # (max_y+1, max_x+1)
        
        skeleton_for_skan_analysis_2d = np.zeros(required_shape_2d, dtype=np.min_scalar_type(label))
        source_for_skan_2d = np.zeros(required_shape_2d, dtype=bool)

        skeleton_for_skan_analysis_2d[tuple(skeleton_global_coords.T)] = label
        source_for_skan_2d[tuple(source_mask_global_coords.T)] = True
        
        # Call Skan without the 'offset' parameter, using the reconstructed arrays
        graph_obj = Skeleton(skeleton_for_skan_analysis_2d, spacing=spacing_yx, source_image=source_for_skan_2d)
        # --- END OF SKAN COMPATIBILITY FIX ---

        branch_data = summarize(graph_obj)

        nx_graph = None
        if hasattr(graph_obj, 'graph') and isinstance(graph_obj.graph, (csr_matrix, nx.Graph)):
            if isinstance(graph_obj.graph, csr_matrix):
                try: nx_graph = nx.from_scipy_sparse_array(graph_obj.graph)
                except Exception as nx_err: print(f"Warning: Label {label}: Failed to convert skan sparse graph to NetworkX: {nx_err}"); nx_graph = None
            else: nx_graph = graph_obj.graph

            if nx_graph is not None:
                total_nodes = nx_graph.number_of_nodes()
                if total_nodes > 0:
                    degrees = pd.Series(dict(nx_graph.degree()))
                    n_endpoints = (degrees == 1).sum()
                    if hasattr(graph_obj, 'n_junctions'): n_junctions = graph_obj.n_junctions
                    elif hasattr(graph_obj, 'n_junction'): n_junctions = graph_obj.n_junction # Older skan
                    else: n_junctions = (degrees > 2).sum()
    
    except ValueError as ve:
        print(f"Label {label}: Skan ValueError (possibly trivial skeleton): {ve}")
        if skeleton_img_to_return_local is None: skeleton_img_to_return_local = np.zeros_like(label_region, dtype=np.uint8)
        # Try to estimate some basic properties if skan fails but skeleton exists
        if skel_pixel_count > 0 and branch_data.empty:
            total_nodes = skel_pixel_count
            if skel_pixel_count == 1: n_endpoints = 1
            elif skel_pixel_count == 2: n_endpoints = 2 # Simple line
    except MemoryError as me:
        print(f"CRITICAL: MemoryError during 2D skeleton analysis for label {label}: {me}")
        if skeleton_img_to_return_local is None: skeleton_img_to_return_local = np.zeros_like(label_region, dtype=np.uint8)
        branch_data = pd.DataFrame(); n_junctions=0; n_endpoints=0; total_nodes=0; graph_obj=None
    except AttributeError as ae: # Catch issues like 'Skeleton' object has no attribute 'graph'
        print(f"Label {label}: Skan AttributeError during analysis: {ae}")
        if skeleton_img_to_return_local is None: skeleton_img_to_return_local = np.zeros_like(label_region, dtype=np.uint8)
        return pd.DataFrame(), skeleton_img_to_return_local, 0, 0, 0, None # Return empty/zero
    except Exception as e:
        print(f"Label {label}: Skan analysis failed unexpectedly: {e}"); traceback.print_exc()
        if skeleton_img_to_return_local is None: skeleton_img_to_return_local = np.zeros_like(label_region, dtype=np.uint8)
        return pd.DataFrame(), skeleton_img_to_return_local, 0, 0, 0, None # Return empty/zero
    finally:
        # Clean up temporary large arrays
        if 'skeleton_for_skan_analysis_2d' in locals(): del skeleton_for_skan_analysis_2d
        if 'source_for_skan_2d' in locals(): del source_for_skan_2d
        if 'graph_obj' in locals() and graph_obj is not None: del graph_obj
        if 'mask_local' in locals() and mask_local is not None: del mask_local

    if skeleton_img_to_return_local is None: skeleton_img_to_return_local = np.zeros_like(label_region, dtype=np.uint8)
    return branch_data, skeleton_img_to_return_local, n_junctions, n_endpoints, total_nodes, None


def calculate_ramification_with_skan_2d(segmented_array, spacing_yx=(1.0, 1.0), labels=None, skeleton_export_path=None):
    """
    Calculate 2D ramification statistics using skan. (CORRECTED)
    """
    print("DEBUG [calculate_ramification_2d] Starting 2D skeleton analysis...")
    if segmented_array.ndim != 2: raise ValueError("Input must be 2D")

    if labels is None:
        unique_labels_in_array = np.unique(segmented_array); labels_to_process = unique_labels_in_array[unique_labels_in_array > 0]
    else:
        unique_labels_in_array = np.unique(segmented_array); present_labels_set = set(unique_labels_in_array[unique_labels_in_array > 0])
        labels = [int(lbl) for lbl in labels]; labels_to_process = sorted([lbl for lbl in labels if lbl in present_labels_set])
    if not labels_to_process: return pd.DataFrame(), pd.DataFrame(), np.zeros_like(segmented_array, dtype=np.uint8)

    print("DEBUG [calculate_ramification_2d] Finding 2D object bounding boxes...")
    locations = ndi.find_objects(segmented_array.astype(np.int32), max_label=max(labels_to_process) if labels_to_process else 0)
    if locations is None: locations = [] # Ensure iterable
    print(f"DEBUG [calculate_ramification_2d] Found {len(locations)} potential 2D boxes.")

    # --- Prepare skeleton array (2D) ---
    use_memmap = bool(skeleton_export_path)
    
    # Robust dtype selection for skeleton_array (mirroring 3D)
    max_label_skel = max(labels_to_process) if labels_to_process else 0
    if max_label_skel == 0: skeleton_dtype = np.uint8 # Should not happen if labels_to_process is not empty
    elif max_label_skel < 2**8: skeleton_dtype = np.uint8
    elif max_label_skel < 2**16: skeleton_dtype = np.uint16
    elif max_label_skel < 2**32: skeleton_dtype = np.uint32 # Default for 2D was uint32
    else: skeleton_dtype = np.uint64 # Should be very rare for 2D label counts
    print(f"DEBUG [calculate_ramification_2d] Using dtype {skeleton_dtype} for 2D skeleton array based on max label {max_label_skel}.")


    skeleton_array = None
    try:
        if use_memmap:
            if os.path.exists(skeleton_export_path): os.remove(skeleton_export_path) # Ensure clean start
            skeleton_array = np.memmap(skeleton_export_path, dtype=skeleton_dtype, mode='w+', shape=segmented_array.shape)
        else: skeleton_array = np.zeros(segmented_array.shape, dtype=skeleton_dtype)
        skeleton_array[:] = 0;
        if use_memmap: skeleton_array.flush()
    except Exception as e_alloc: print(f"Error preparing 2D skeleton array: {e_alloc}"); return pd.DataFrame(), pd.DataFrame(), None

    all_branch_data_dfs = []; summary_stats_list = []

    for label_val in tqdm(labels_to_process, desc="Analyzing Skeletons (2D skan)"): # Renamed label to label_val
        loc_index = label_val - 1
        if loc_index < 0 or loc_index >= len(locations) or locations[loc_index] is None: 
            summary_stats_list.append({'label': label_val, 'skan_num_branches': 0, 'skan_total_length_um': 0.0, 'skan_avg_branch_length_um': np.nan, 'skan_num_junctions': 0, 'skan_num_endpoints': 0, 'skan_num_skeleton_pixels': 0})
            continue
        
        slices = locations[loc_index]; label_region = None; offset_yx = None
        try:
            label_region = segmented_array[slices]
            offset_yx = np.array([s.start for s in slices]) # 2D offset (Y, X)
        except Exception as e: print(f"Warning: Error slicing for skel label {label_val}: {e}"); continue

        # Analyze 2D sub-region with the corrected helper
        branch_data, label_skeleton_img_local, n_junctions, n_endpoints, total_nodes, _ = \
            analyze_skeleton_with_skan_2d(label_region, offset_yx, spacing_yx, label_val)

        num_skel_pixels = 0
        if label_skeleton_img_local is not None and label_skeleton_img_local.ndim == 2 :
            try:
                local_coords = np.argwhere(label_skeleton_img_local > 0) # This is binary
                if local_coords.size > 0:
                     global_coords = local_coords + offset_yx
                     idx_y, idx_x = global_coords[:, 0], global_coords[:, 1]
                     valid_idx = ((idx_y >= 0) & (idx_y < skeleton_array.shape[0]) & 
                                  (idx_x >= 0) & (idx_x < skeleton_array.shape[1]))
                     idx_y, idx_x = idx_y[valid_idx], idx_x[valid_idx]
                     skeleton_array[idx_y, idx_x] = label_val # Assign label ID
                     num_skel_pixels = len(idx_y)
                     if use_memmap: skeleton_array.flush()
                     del local_coords, global_coords, idx_y, idx_x, valid_idx
            except Exception as map_err: print(f"Warning: Error mapping 2D skeleton for label {label_val}: {map_err}"); num_skel_pixels = -1

        total_length = 0.0; num_branches = 0; avg_branch_length = np.nan
        if not branch_data.empty:
            try:
                branch_data['label'] = label_val; all_branch_data_dfs.append(branch_data.copy())
                total_length = branch_data['branch-distance'].sum(); num_branches = len(branch_data)
                if 'branch-distance' in branch_data.columns and branch_data['branch-distance'].notna().any(): 
                    avg_branch_length = branch_data['branch-distance'].mean()
            except Exception as agg_err: print(f"Warning: Error aggregating 2D branch stats label {label_val}: {agg_err}")

        summary_stats_list.append({
            'label': label_val, 'skan_num_branches': num_branches, 'skan_total_length_um': total_length,
            'skan_avg_branch_length_um': avg_branch_length, 'skan_num_junctions': n_junctions,
            'skan_num_endpoints': n_endpoints, 'skan_num_skeleton_pixels': num_skel_pixels,
        })
        del label_region, offset_yx, branch_data, label_skeleton_img_local
        # if i % 50 == 0: gc.collect()

    detailed_branch_df = pd.DataFrame()
    if all_branch_data_dfs: 
        try: detailed_branch_df = pd.concat(all_branch_data_dfs, ignore_index=True)
        except Exception as concat_err: print(f"Warning: Error concat 2D branches: {concat_err}")
    del all_branch_data_dfs; gc.collect()
    ramification_summary_df = pd.DataFrame(summary_stats_list)

    print(f"DEBUG [calculate_ramification_2d] Finished. Skeleton shape {skeleton_array.shape if skeleton_array is not None else 'None'}, dtype {skeleton_array.dtype if skeleton_array is not None else 'N/A'}.")
    return ramification_summary_df, detailed_branch_df, skeleton_array

# =============================================================================
# Main Analysis Function (2D Adapted)
# =============================================================================

def analyze_segmentation_2d(segmented_array, spacing_yx=(1.0, 1.0),
                            calculate_distances=True,
                            calculate_skeletons=True,
                            skeleton_export_path=None,
                            temp_dir=None, n_jobs=None,
                            return_detailed=False):
    """
    Comprehensive analysis of a 2D segmented array.
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
            segmented_array, spacing_yx=spacing_yx, labels=labels_for_skeleton, skeleton_export_path=skeleton_export_path
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