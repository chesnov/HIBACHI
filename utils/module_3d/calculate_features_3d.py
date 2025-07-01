import numpy as np # type: ignore
import pandas as pd # type: ignore
from scipy.spatial import cKDTree # type: ignore
from scipy import ndimage as ndi # type: ignore # For potential preprocessing
import multiprocessing as mp
import os
import tempfile
import time
import psutil # type: ignore
import gc
from skimage.morphology import skeletonize, ball # type: ignore
from skan import Skeleton, summarize # type: ignore
import networkx as nx # type: ignore # Still potentially useful for graph ops if needed
from tqdm.auto import tqdm # type: ignore # Use auto for notebook/console compatibility
from scipy.sparse import csr_matrix # type: ignore
import traceback

seed = 42
np.random.seed(seed)         # For NumPy

# --- DEBUG CONFIG ---
DEBUG_TARGET_LABEL = 2 # Set to a specific integer label ID to focus detailed prints, or None
# --- END DEBUG CONFIG ---

# Helper function for conditional printing
def debug_print(msg, label=None):
    if DEBUG_TARGET_LABEL is None or label == DEBUG_TARGET_LABEL:
        print(f"  DEBUG: {msg}")

# =============================================================================
# Distance Calculation Functions
# =============================================================================

def extract_surface_points_worker(args):
    """Worker function to extract surface points from a mask."""
    label_idx, label, segmented_array, temp_dir, spacing = args
    # debug_print(f"[Worker {os.getpid()}] Extracting surface for label {label}", label=label) # Less useful now
    mask = (segmented_array == label)
    if not np.any(mask):
        surface_file = os.path.join(temp_dir, f"surface_{label}.npy")
        np.save(surface_file, np.empty((0, 3), dtype=int)) # Save empty array
        return label_idx, 0

    # --- DEBUG ---
    if label == DEBUG_TARGET_LABEL or (DEBUG_TARGET_LABEL is None and label_idx < 3): # Limit printing
        debug_print(f"[Worker {os.getpid()}] Label {label}: Mask shape {mask.shape}", label=label)
    # --- END DEBUG ---

    surface = np.zeros_like(mask, dtype=bool)
    z_max, x_max, y_max = mask.shape

    # ... (surface finding logic - unchanged) ...
    if z_max > 1: surface[:-1, :, :] |= (mask[:-1, :, :] & ~mask[1:, :, :]); surface[1:, :, :] |= (mask[1:, :, :] & ~mask[:-1, :, :])
    elif z_max == 1: surface[0,:,:] = mask[0,:,:]
    surface[:, :-1, :] |= (mask[:, :-1, :] & ~mask[:, 1:, :]); surface[:, 1:, :] |= (mask[:, 1:, :] & ~mask[:, :-1, :])
    if x_max == 1: surface[:, 0, :] = mask[:, 0, :]
    surface[:, :, :-1] |= (mask[:, :, :-1] & ~mask[:, :, 1:]); surface[:, :, 1:] |= (mask[:, :, 1:] & ~mask[:, :, :-1])
    if y_max == 1: surface[:, :, 0] = mask[:, :, 0]


    z, x, y = np.where(surface)
    surface_points = np.column_stack((z, x, y))

    # --- DEBUG ---
    if label == DEBUG_TARGET_LABEL or (DEBUG_TARGET_LABEL is None and label_idx < 3): # Limit printing
        debug_print(f"[Worker {os.getpid()}] Label {label}: Found {len(surface_points)} surface points. Surface shape {surface.shape}. First 5 points (Z,X,Y):\n{surface_points[:5]}", label=label)
    # --- END DEBUG ---

    surface_file = os.path.join(temp_dir, f"surface_{label}.npy")
    np.save(surface_file, surface_points)

    return label_idx, len(surface_points)

def calculate_pair_distances_worker(args):
    """Worker function to calculate distances between a pair of masks."""
    i, j, labels, temp_dir, spacing, distances_matrix, points_matrix = args

    label1 = labels[i]
    label2 = labels[j]

    # --- DEBUG ---
    is_target_pair = (DEBUG_TARGET_LABEL is not None and (label1 == DEBUG_TARGET_LABEL or label2 == DEBUG_TARGET_LABEL))
    # --- END DEBUG ---

    # Load surface points, handle potential errors or empty files
    try:
        surface_points1 = np.load(os.path.join(temp_dir, f"surface_{label1}.npy"))
        surface_points2 = np.load(os.path.join(temp_dir, f"surface_{label2}.npy"))
        # --- DEBUG ---
        # if is_target_pair: debug_print(f"[Worker {os.getpid()}] Pair ({label1}, {label2}): Loaded surface shapes {surface_points1.shape}, {surface_points2.shape}", label=DEBUG_TARGET_LABEL)
        # --- END DEBUG ---
    except Exception as e:
        print(f"Warning: Error loading surface points for masks {label1} or {label2}: {e}")
        return i, j, np.inf, np.full(6, np.nan)

    if surface_points1.shape[0] == 0 or surface_points2.shape[0] == 0:
        return i, j, np.inf, np.full(6, np.nan)

    # Scale points according to spacing
    scaled_points1 = surface_points1 * np.array(spacing)
    scaled_points2 = surface_points2 * np.array(spacing)

    # Build KDTree for the second mask
    try:
        tree2 = cKDTree(scaled_points2)
    except ValueError:
         return i, j, np.inf, np.full(6, np.nan)

    # Find closest points and distances
    distances, indices = tree2.query(scaled_points1, k=1)

    # Get minimum distance and corresponding points
    min_idx = np.argmin(distances)
    min_distance = distances[min_idx]

    # --- These are the ORIGINAL PIXEL INDICES ---
    point1 = surface_points1[min_idx]
    point2 = surface_points2[indices[min_idx]]

    result_points = np.array([point1[0], point1[1], point1[2], point2[0], point2[1], point2[2]])

    # --- DEBUG ---
    if is_target_pair:
        debug_print(f"[Worker {os.getpid()}] Pair ({label1}, {label2}): Min dist {min_distance:.2f}. Closest point indices (Z,X,Y): Point1={point1}, Point2={point2}", label=DEBUG_TARGET_LABEL)
    # --- END DEBUG ---

    return i, j, min_distance, result_points

# --- Reworked surface extraction worker to handle region + offset ---
def extract_surface_points_worker_optimized(args):
    """Worker function to extract surface points from a SUB-REGION."""
    label_idx, label, region, temp_dir, spacing, offset = args
    # debug_print(f"[Worker {os.getpid()}] Extracting surface for label {label} in region {region.shape} w/ offset {offset}", label=label)

    # Create mask WITHIN the region
    mask = (region == label)
    if not np.any(mask):
        surface_file = os.path.join(temp_dir, f"surface_{label}.npy")
        np.save(surface_file, np.empty((0, 3), dtype=int)) # Save empty array
        return label_idx, 0

    if label == DEBUG_TARGET_LABEL:
        debug_print(f"[Worker {os.getpid()}] Label {label}: Region shape {region.shape}, Offset {offset}", label=label)

    # Find surface WITHIN the region
    surface = np.zeros_like(mask, dtype=bool)
    z_max, x_max, y_max = mask.shape
    if z_max > 1: surface[:-1, :, :] |= (mask[:-1, :, :] & ~mask[1:, :, :]); surface[1:, :, :] |= (mask[1:, :, :] & ~mask[:-1, :, :])
    elif z_max == 1: surface[0,:,:] = mask[0,:,:]
    if x_max > 1: surface[:, :-1, :] |= (mask[:, :-1, :] & ~mask[:, 1:, :]); surface[:, 1:, :] |= (mask[:, 1:, :] & ~mask[:, :-1, :])
    elif x_max == 1: surface[:, 0, :] = mask[:, 0, :]
    if y_max > 1: surface[:, :, :-1] |= (mask[:, :, :-1] & ~mask[:, :, 1:]); surface[:, :, 1:] |= (mask[:, :, 1:] & ~mask[:, :, :-1])
    elif y_max == 1: surface[:, :, 0] = mask[:, :, 0]

    z_local, x_local, y_local = np.where(surface)

    # Convert LOCAL coordinates back to GLOBAL coordinates using the offset
    surface_points_global = np.column_stack((z_local + offset[0], x_local + offset[1], y_local + offset[2]))

    if label == DEBUG_TARGET_LABEL:
        debug_print(f"[Worker {os.getpid()}] Label {label}: Found {len(surface_points_global)} surface points. First 5 GLOBAL points (Z,X,Y):\n{surface_points_global[:5]}", label=label)

    surface_file = os.path.join(temp_dir, f"surface_{label}.npy")
    np.save(surface_file, surface_points_global) # Save GLOBAL coordinates

    return label_idx, len(surface_points_global)


def shortest_distance(segmented_array, spacing=(1.0, 1.0, 1.0),
                      temp_dir=None, n_jobs=None, max_memory_pct=75,
                      batch_size=None):
    """
    Calculate the shortest distance between surfaces of all masks pairs.
    (Implementation unchanged, but relies on modified extract_surface_points_worker)
    """
    start_time = time.time()

    if n_jobs is None:
        n_jobs = max(1, mp.cpu_count() - 1)

    temp_dir_managed = False
    if temp_dir is None:
        # Create temp dir in a place less likely to cause issues
        base_temp_dir = os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp"
        temp_dir = tempfile.mkdtemp(prefix="shortest_dist_", dir=base_temp_dir)
        temp_dir_managed = True
        print(f"DEBUG [shortest_distance] Using managed temporary directory: {temp_dir}")
    else:
        os.makedirs(temp_dir, exist_ok=True)
        print(f"DEBUG [shortest_distance] Using provided temporary directory: {temp_dir}")


    unique_labels = np.unique(segmented_array)
    labels = unique_labels[unique_labels > 0]
    n_labels = len(labels)
    label_to_index = {label: i for i, label in enumerate(labels)}

    if n_labels <= 1:
        print("Need at least two masks to calculate distances.")
        if temp_dir_managed and os.path.exists(temp_dir):
            try: import shutil; shutil.rmtree(temp_dir);
            except OSError as e: print(f"Warning: Failed to remove temp dir {temp_dir}: {e}")
        empty_df = pd.DataFrame(index=labels, columns=labels)
        empty_points = pd.DataFrame(columns=['mask1', 'mask2', 'mask1_z', 'mask1_x', 'mask1_y', 'mask2_z', 'mask2_x', 'mask2_y'])
        if n_labels == 1: empty_df.loc[labels[0], labels[0]] = 0.0
        return empty_df, empty_points

    print(f"DEBUG [shortest_distance] Found {n_labels} masks. Labels: {labels[:10]}... Calculating distances with {n_jobs} workers.") # Print first few labels

    # --- Surface Point Extraction ---
    print("DEBUG [shortest_distance] Extracting surface points...")
    # --- Optimization: Pass slices to avoid full array copy in worker ---
    print("DEBUG [shortest_distance] Finding object bounding boxes...")
    locations = ndi.find_objects(segmented_array, max_label=labels.max()) # Find bounding boxes once
    print(f"DEBUG [shortest_distance] Found {len(locations)} potential bounding boxes.")

    extract_args = []
    labels_found_in_locations = [] # Keep track of labels that actually have a bbox
    for i, label in enumerate(labels):
        if label -1 < len(locations) and locations[label-1] is not None:
            slices = locations[label - 1]
            # Extract only the relevant sub-array for the worker
            # Add a small buffer (e.g., 1 voxel) if possible, to ensure surface isn't cut off
            buffered_slices = []
            for s, max_dim in zip(slices, segmented_array.shape):
                start = max(0, s.start - 1)
                stop = min(max_dim, s.stop + 1)
                buffered_slices.append(slice(start, stop))
            buffered_slices = tuple(buffered_slices)

            try:
                 # Pass the cropped region and the *global* offset of this region
                 region = segmented_array[buffered_slices]
                 offset = np.array([s.start for s in buffered_slices])
                 extract_args.append((i, label, region, temp_dir, spacing, offset)) # Pass offset
                 labels_found_in_locations.append(label)
            except IndexError:
                 print(f"Warning: Skipping label {label} due to invalid slice from find_objects: {buffered_slices}")
            except Exception as e:
                 print(f"Warning: Error processing slices for label {label}: {e}")
        else:
             debug_print(f"Label {label} not found or has no slice in 'locations' (index {label-1}). Skipping surface extraction.", label=label)
             # Still need to create an empty file so loading doesn't fail later
             surface_file = os.path.join(temp_dir, f"surface_{label}.npy")
             np.save(surface_file, np.empty((0, 3), dtype=int))

    print(f"DEBUG [shortest_distance] Prepared {len(extract_args)} tasks for surface extraction (labels with valid bounding boxes).")

    # Execute surface extraction
    with mp.Pool(processes=n_jobs) as pool:
         list(tqdm(pool.imap_unordered(extract_surface_points_worker_optimized, extract_args), total=len(extract_args), desc="Extracting Surfaces"))

    # Filter labels to only those processed (had bounding boxes) for distance calc
    labels_to_process_distance = labels_found_in_locations
    n_labels_processed = len(labels_to_process_distance)
    label_to_processed_index = {label: i for i, label in enumerate(labels_to_process_distance)}

    if n_labels_processed <= 1:
        print("Need at least two valid masks (with bounding boxes) to calculate distances.")
        if temp_dir_managed and os.path.exists(temp_dir):
            try: import shutil; shutil.rmtree(temp_dir);
            except OSError as e: print(f"Warning: Failed to remove temp dir {temp_dir}: {e}")
        # Return empty dataframes matching the original full label set structure if needed
        final_distance_df = pd.DataFrame(np.inf, index=labels, columns=labels)
        np.fill_diagonal(final_distance_df.values, 0)
        final_points_df = pd.DataFrame(columns=['mask1', 'mask2', 'mask1_z', 'mask1_x', 'mask1_y', 'mask2_z', 'mask2_x', 'mask2_y'])
        return final_distance_df, final_points_df


    print(f"DEBUG [shortest_distance] Calculating pairwise distances for {n_labels_processed} valid labels...")
    # Initialize matrices for the *processed* labels only
    distances_matrix = np.full((n_labels_processed, n_labels_processed), np.inf, dtype=np.float32)
    points_matrix = np.full((n_labels_processed, n_labels_processed, 6), np.nan, dtype=np.float32)
    np.fill_diagonal(distances_matrix, 0)

    # Calculate pairs based on the *processed* labels and their new indices
    all_pairs = [(i, j) for i in range(n_labels_processed) for j in range(i + 1, n_labels_processed)]
    print(f"DEBUG [shortest_distance] Total pairs to calculate: {len(all_pairs)}")

    # Pass the list of *processed* labels to the worker
    calc_args = [(i, j, labels_to_process_distance, temp_dir, spacing, distances_matrix, points_matrix)
                 for i, j in all_pairs]

    with mp.Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap_unordered(calculate_pair_distances_worker, calc_args), total=len(all_pairs), desc="Calculating Distances"))

    # Process results into the smaller matrices
    for i, j, distance, points in results:
        # i, j are indices relative to labels_to_process_distance
        distances_matrix[i, j] = distance
        distances_matrix[j, i] = distance
        points_matrix[i, j, :] = points
        if not np.any(np.isnan(points)):
            points_matrix[j, i, :] = np.array([points[3], points[4], points[5],
                                               points[0], points[1], points[2]])

    # --- Convert to DataFrames ---
    # Create DataFrames using the processed labels first
    distance_matrix_processed_df = pd.DataFrame(distances_matrix, index=labels_to_process_distance, columns=labels_to_process_distance)

    points_flat = points_matrix.reshape(-1, 6)
    mask1_labels_proc = np.repeat(labels_to_process_distance, n_labels_processed)
    mask2_labels_proc = np.tile(labels_to_process_distance, n_labels_processed)

    points_processed_df = pd.DataFrame({
        'mask1': mask1_labels_proc,
        'mask2': mask2_labels_proc,
        'mask1_z': points_flat[:, 0], 'mask1_x': points_flat[:, 1], 'mask1_y': points_flat[:, 2],
        'mask2_z': points_flat[:, 3], 'mask2_x': points_flat[:, 4], 'mask2_y': points_flat[:, 5]
    })
    points_processed_df = points_processed_df[points_processed_df['mask1'] != points_processed_df['mask2']].reset_index(drop=True)
    # Drop rows where points calculation failed (NaNs remain)
    points_processed_df = points_processed_df.dropna(subset=['mask1_z', 'mask2_z']) # Check one coord from each point


    # --- Expand to original full label set ---
    # Create final DFs with all original labels, fill with Inf/NaN initially
    final_distance_df = pd.DataFrame(np.inf, index=labels, columns=labels, dtype=np.float32)
    np.fill_diagonal(final_distance_df.values, 0)

    # Fill in the calculated values using .loc
    final_distance_df.loc[labels_to_process_distance, labels_to_process_distance] = distance_matrix_processed_df

    # For points, it's easier to just use the processed df, users can merge later if needed
    final_points_df = points_processed_df # Return only the calculated pairs

    # --- DEBUG ---
    print(f"DEBUG [shortest_distance] Finished distance calc. Final Distance matrix shape {final_distance_df.shape}. Final Points DF shape {final_points_df.shape}")
    if DEBUG_TARGET_LABEL is not None and DEBUG_TARGET_LABEL in final_points_df['mask1'].values:
         debug_print(f"Points DF sample for label {DEBUG_TARGET_LABEL}:\n{final_points_df[final_points_df['mask1'] == DEBUG_TARGET_LABEL].head()}", label=DEBUG_TARGET_LABEL)
    elif DEBUG_TARGET_LABEL is None:
         print(f"DEBUG [shortest_distance] Points DF head:\n{final_points_df.head()}")
    # --- END DEBUG ---

    # --- Cleanup ---
    if temp_dir_managed and os.path.exists(temp_dir):
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"DEBUG [shortest_distance] Successfully removed temp dir: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not fully clean up temp dir {temp_dir}: {e}")


    elapsed = time.time() - start_time
    print(f"DEBUG [shortest_distance] Distance calculation finished in {elapsed:.1f} seconds")

    return final_distance_df, final_points_df

# =============================================================================
# Volume and Basic Shape Metrics
# =============================================================================

def calculate_volume(segmented_array, spacing=(1.0, 1.0, 1.0)):
    """Calculate volume and basic shape metrics for each mask (Memory Optimized)."""
    print("DEBUG [calculate_volume] Starting volume calculations (Optimized)...")
    voxel_volume = np.prod(spacing)
    sx, sy, sz = spacing[1], spacing[2], spacing[0] # Surface area factors
    area_xy = sx * sy; area_xz = sx * sz; area_yz = sy * sz

    unique_labels = np.unique(segmented_array)
    labels = unique_labels[unique_labels > 0]
    if len(labels) == 0:
        print("DEBUG [calculate_volume] No positive labels found.")
        return pd.DataFrame()

    # Find bounding boxes for all labels at once
    print("DEBUG [calculate_volume] Finding object bounding boxes...")
    try:
        # Ensure max_label is sufficient if labels are not contiguous
        max_lbl = labels.max() if len(labels) > 0 else 0
        locations = ndi.find_objects(segmented_array, max_label=max_lbl)
        print(f"DEBUG [calculate_volume] Found {len(locations)} potential bounding boxes.")
        if max_lbl >= len(locations):
            print(f"Warning: Max label {max_lbl} exceeds length of locations {len(locations)}. Some labels might be missed if non-contiguous.")
    except Exception as e:
        print(f"Error calling ndi.find_objects: {e}")
        return pd.DataFrame()


    results = []

    # Use tqdm for progress bar
    for i, label in enumerate(tqdm(labels, desc="Calculating Volumes")):
        # Get the slice for the current label (adjusting for 0-based index)
        loc_index = label - 1
        if loc_index < 0 or loc_index >= len(locations) or locations[loc_index] is None:
            if label == DEBUG_TARGET_LABEL:
                debug_print(f"Label {label}: No bounding box found (index {loc_index}). Skipping.", label=label)
            continue # Skip labels not found by find_objects

        slices = locations[loc_index]

        # Extract the relevant sub-region (view if possible)
        try:
            label_region = segmented_array[slices]
        except Exception as e:
             print(f"Warning: Error slicing array for label {label} with slices {slices}: {e}")
             continue

        # Create mask ONLY within the sub-region (MUCH smaller)
        mask = (label_region == label)
        voxel_count = np.count_nonzero(mask) # Faster than np.sum for boolean

        if voxel_count == 0:
             if label == DEBUG_TARGET_LABEL: debug_print(f"Label {label}: Voxel count is 0 in region. Skipping.", label=label)
             continue # Should not happen if find_objects worked, but safety check

        volume = voxel_count * voxel_volume

        # Bounding box dimensions from slices (already in voxel units)
        # Add 1 because slice stop is exclusive
        z_len = slices[0].stop - slices[0].start
        x_len = slices[1].stop - slices[1].start
        y_len = slices[2].stop - slices[2].start
        bbox_volume = (z_len * spacing[0]) * (x_len * spacing[1]) * (y_len * spacing[2])

        if label == DEBUG_TARGET_LABEL:
            debug_print(f"Label {label}: Processing region of shape {label_region.shape} from slices {slices}", label=label)
            debug_print(f"Label {label}: Voxel count {voxel_count}. BBox dims (voxels) Z:{z_len}, X:{x_len}, Y:{y_len}", label=label)


        # --- Surface Area Calculation (on the smaller 'mask') ---
        surface_area = 0.0
        try:
             # Use the same logic, but applied to the small 'mask'
             z_dim, x_dim, y_dim = mask.shape

             # Differences along axes
             diff_z = np.diff(mask.astype(np.int8), axis=0)
             diff_x = np.diff(mask.astype(np.int8), axis=1)
             diff_y = np.diff(mask.astype(np.int8), axis=2)

             # Surface area from internal faces
             surface_area += np.count_nonzero(diff_z) * area_xy # count_nonzero is faster
             surface_area += np.count_nonzero(diff_x) * area_yz
             surface_area += np.count_nonzero(diff_y) * area_xz

             # Surface area from faces touching the bounding box of the 'mask'
             if z_dim > 0:
                 surface_area += np.count_nonzero(mask[0, :, :]) * area_xy
                 surface_area += np.count_nonzero(mask[-1, :, :]) * area_xy
             if x_dim > 0:
                 surface_area += np.count_nonzero(mask[:, 0, :]) * area_yz
                 surface_area += np.count_nonzero(mask[:, -1, :]) * area_yz
             if y_dim > 0:
                 surface_area += np.count_nonzero(mask[:, :, 0]) * area_xz
                 surface_area += np.count_nonzero(mask[:, :, -1]) * area_xz

        except Exception as e:
            print(f"Warning: Surface area calculation failed for label {label}: {e}")
            surface_area = np.nan # Assign NaN if calculation fails


        if surface_area > 1e-9: # Use a smaller threshold, np.count_nonzero doesn't have float issues
            sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / surface_area
        else:
            sphericity = np.nan
            if voxel_count > 0 and surface_area <= 1e-9 : # If it has volume but no surface, it's likely a single voxel
                 if label == DEBUG_TARGET_LABEL: debug_print(f"Label {label}: Voxel count > 0 but surface area ~0. Setting sphericity NaN.", label=label)


        results.append({
            'label': label, 'volume_um3': volume, 'surface_area_um2': surface_area,
            'voxel_count': voxel_count, 'bounding_box_volume_um3': bbox_volume, 'sphericity': sphericity
        })

        # --- Explicitly delete large arrays within the loop ---
        del label_region, mask, diff_z, diff_x, diff_y
        # Optional: Force garbage collection periodically if memory is still an issue
        # if i % 100 == 0: gc.collect()

    print("DEBUG [calculate_volume] Finished volume calculations.")
    return pd.DataFrame(results)

# =============================================================================
# Depth Calculation
# =============================================================================

def calculate_depth_df(segmented_array, spacing=(1.0, 1.0, 1.0)):
    """Calculate the median depth of each mask in microns."""
    print("DEBUG [calculate_depth_df] Starting depth calculations...")
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    depth_data = []

    for i, label in enumerate(tqdm(labels, desc="Calculating Depths")):
        mask = (segmented_array == label)
        if np.any(mask):
            z_indices, _, _ = np.where(mask)
            median_z = np.median(z_indices)
            depth = median_z * spacing[0]
            depth_data.append({'label': label, 'depth_um': depth})
            # --- DEBUG ---
            if label == DEBUG_TARGET_LABEL or (DEBUG_TARGET_LABEL is None and i < 3):
                 debug_print(f"Label {label}: Median Z index {median_z:.1f}, Depth {depth:.2f} um", label=label)
            # --- END DEBUG ---
        else:
            depth_data.append({'label': label, 'depth_um': np.nan})

    print("DEBUG [calculate_depth_df] Finished depth calculations.")
    return pd.DataFrame(depth_data)

# =============================================================================
# Skeletonization and Ramification (using skan)
# =============================================================================
def _trace_spur_length_and_path_3d(skeleton, start_z, start_x, start_y, spacing):
    """
    3D equivalent of _trace_spur_length_and_path. Traces a spur from an endpoint.
    """
    if skeleton[start_z, start_x, start_y] == 0:
        return 0.0, []
    
    visited = set()
    current_z, current_x, current_y = start_z, start_x, start_y
    total_length = 0.0
    path = [(current_z, current_x, current_y)]
    
    # 26-connectivity offsets for 3D
    offsets = [
        (dz, dx, dy) for dz in [-1, 0, 1] for dx in [-1, 0, 1] for dy in [-1, 0, 1]
        if not (dz == 0 and dx == 0 and dy == 0)
    ]
    
    while True:
        visited.add((current_z, current_x, current_y))
        
        neighbors = []
        for dz, dx, dy in offsets:
            nz, nx, ny = current_z + dz, current_x + dx, current_y + dy
            if (0 <= nz < skeleton.shape[0] and 
                0 <= nx < skeleton.shape[1] and
                0 <= ny < skeleton.shape[2] and
                skeleton[nz, nx, ny] == 1 and 
                (nz, nx, ny) not in visited):
                neighbors.append((nz, nx, ny))
        
        if len(neighbors) != 1: # Stop if we hit a junction or the end of a line
            break
        
        next_z, next_x, next_y = neighbors[0]
        
        distance = np.sqrt(
            ((next_z - current_z) * spacing[0])**2 +
            ((next_x - current_x) * spacing[1])**2 +
            ((next_y - current_y) * spacing[2])**2
        )
        total_length += distance
        
        current_z, current_x, current_y = next_z, next_x, next_y
        path.append((current_z, current_x, current_y))
    
    return total_length, path


def _prune_skeleton_spurs_3d(skeleton_binary, spacing, max_spur_length_um):
    """
    3D equivalent of _prune_skeleton_spurs.
    CORRECTED to use a kernel with a zeroed center.
    """
    if max_spur_length_um <= 0:
        return skeleton_binary
    
    pruned_skeleton = skeleton_binary.copy()
    
    # --- KERNEL FIX ---
    # Create the 26-connectivity kernel and explicitly set the center to 0.
    # This now correctly counts neighbors only, matching the 2D code's logic.
    kernel = ndi.generate_binary_structure(3, 3).astype(np.uint8)
    kernel[1, 1, 1] = 0

    changed = True
    max_iterations = 50
    iteration = 0
    
    while changed and iteration < max_iterations:
        changed = False
        iteration += 1
        
        neighbor_count = ndi.convolve(pruned_skeleton.astype(np.uint8), 
                                      kernel, mode='constant', cval=0)
        
        # This condition will now work correctly.
        endpoints = (pruned_skeleton == 1) & (neighbor_count == 1)
        
        if not np.any(endpoints):
            break
            
        endpoint_coords = np.argwhere(endpoints)
        
        for ep_z, ep_x, ep_y in endpoint_coords:
            if pruned_skeleton[ep_z, ep_x, ep_y] == 0:
                continue
                
            spur_length, spur_path = _trace_spur_length_and_path_3d(pruned_skeleton, ep_z, ep_x, ep_y, spacing)
            
            if spur_length <= max_spur_length_um:
                _remove_spur_path_3d(pruned_skeleton, spur_path)
                changed = True
                
    return pruned_skeleton

def _remove_spur_path_3d(skeleton, path):
    """
    3D equivalent of _remove_spur_path.
    CORRECTED to use a kernel with a zeroed center.
    """
    # --- KERNEL FIX ---
    kernel = ndi.generate_binary_structure(3, 3).astype(np.uint8)
    kernel[1, 1, 1] = 0
    
    for i, (z, x, y) in enumerate(path):
        if i == len(path) - 1:
            # Check the neighbor count of the last point to see if it's a junction.
            # This logic now works because the kernel is correct.
            neighbor_count = ndi.convolve(skeleton.astype(np.uint8), 
                                          kernel, mode='constant', cval=0)[z, x, y]
            # If it has more than 1 neighbor *other than the spur being removed*, it's a junction.
            if neighbor_count > 1:
                break
        skeleton[z, x, y] = 0

def analyze_skeleton_topology_correctly_3d(skeleton_binary, spacing=(1.0, 1.0, 1.0)):
    """
    3D equivalent of analyze_skeleton_topology_correctly.
    CORRECTED to use a kernel with a zeroed center.
    """
    if not np.any(skeleton_binary):
        return 0, 0, 0, 0.0
    
    # --- KERNEL FIX ---
    kernel = ndi.generate_binary_structure(3, 3).astype(np.uint8)
    kernel[1, 1, 1] = 0
    
    neighbor_count = ndi.convolve(skeleton_binary.astype(np.uint8), kernel, mode='constant', cval=0)
    skeleton_mask = skeleton_binary > 0
    
    # The endpoint and junction logic will now work correctly.
    endpoints = skeleton_mask & (neighbor_count == 1)
    junctions = skeleton_mask & (neighbor_count >= 3)
    
    n_endpoints = np.sum(endpoints)
    n_junctions = np.sum(junctions)
    
    if n_junctions == 0:
        true_branches = 0
    else:
        # This logic remains correct, but now operates on accurate neighbor counts.
        junction_degrees = neighbor_count[junctions]
        excess_branches = np.sum(junction_degrees - 2)
        true_branches = excess_branches
    
    total_length = 0.0
    try:
        graph_obj = Skeleton(skeleton_binary.astype(np.uint8), spacing=spacing)
        branch_data = summarize(graph_obj)
        if not branch_data.empty:
            total_length = branch_data['branch-distance'].sum()
    except:
        total_length = np.sum(skeleton_binary) * np.mean(spacing)
    
    return true_branches, n_junctions, n_endpoints, total_length

def analyze_skeleton_with_skan_3d(label_region, offset, spacing, label, prune_spurs_le_um=0.0):
    """
    3D equivalent of analyze_skeleton_with_skan_2d.
    Structure and logic are identical, adapted for 3D.
    """
    branch_data = pd.DataFrame()
    skeleton_img_to_return_local = np.zeros_like(label_region, dtype=np.uint8)

    try:
        mask_local = (label_region == label)
        if not np.any(mask_local):
            return branch_data, skeleton_img_to_return_local, 0, 0, 0, 0.0

        skeleton_binary_local = skeletonize(mask_local)
        
        # Apply pruning, matching the 2D logic
        if np.any(skeleton_binary_local):
            if prune_spurs_le_um > 0.0:
                skeleton_binary_local = _prune_skeleton_spurs_3d(skeleton_binary_local, spacing, prune_spurs_le_um)
            
        skeleton_img_to_return_local = skeleton_binary_local.astype(np.uint8)

        if not np.any(skeleton_img_to_return_local):
            return branch_data, skeleton_img_to_return_local, 0, 0, 0, 0.0

        # GET CORRECT TOPOLOGY AND MEASUREMENTS from the (potentially pruned) skeleton
        true_branches, n_junctions, n_endpoints, total_length = analyze_skeleton_topology_correctly_3d(
            skeleton_img_to_return_local, spacing)

        # Still get skan data for detailed analysis (but ignore its branch count)
        try:
            # Skan needs a minimal array containing the skeleton with global-like coordinates
            skeleton_global_coords = np.argwhere(skeleton_binary_local) + offset
            if skeleton_global_coords.size > 0:
                min_coords = np.min(skeleton_global_coords, axis=0)
                skan_shape = np.max(skeleton_global_coords, axis=0) - min_coords + 1
                relative_coords = skeleton_global_coords - min_coords
                
                skeleton_for_skan = np.zeros(skan_shape, dtype=np.uint8)
                skeleton_for_skan[tuple(relative_coords.T)] = 1

                graph_obj = Skeleton(skeleton_for_skan, spacing=spacing)
                branch_data = summarize(graph_obj)

                if not branch_data.empty:
                    # Shift coordinates back to global space
                    for i, suffix in enumerate(['-0', '-1', '-2']): # z, x, y
                         for col in branch_data.columns:
                            if col.endswith(suffix) and ('coord' in col):
                                branch_data[col] += min_coords[i]
        except:
            pass

        return branch_data, skeleton_img_to_return_local, true_branches, n_junctions, n_endpoints, total_length
                    
    except Exception as e:
        print(f"Label {label}: Analysis failed: {e}")
        return pd.DataFrame(), np.zeros_like(label_region, dtype=np.uint8), 0, 0, 0, 0.0

def calculate_ramification_with_skan(segmented_array, spacing=(1.0, 1.0, 1.0), labels=None, 
                                     skeleton_export_path=None, 
                                     prune_spurs_le_um=0.0): # <-- MODIFIED PARAMETER
    """
    CORRECTED version that reports true biological branch statistics for 3D data.
    """
    print("DEBUG [calculate_ramification_3d] Starting CORRECTED 3D skeleton analysis...")
    
    if segmented_array.ndim != 3: 
        raise ValueError("Input must be 3D")

    # Handle labels (logic is unchanged)
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

    print(f"Processing {len(labels_to_process)} labels with spur pruning <= {prune_spurs_le_um} um...")
    
    # Get bounding boxes (logic is unchanged)
    locations = ndi.find_objects(segmented_array.astype(np.int32), 
                                 max_label=max(labels_to_process) if labels_to_process else 0)
    if locations is None: locations = []

    # Prepare skeleton array (logic is unchanged)
    use_memmap = bool(skeleton_export_path)
    max_label_skel = max(labels_to_process) if labels_to_process else 0
    
    if max_label_skel < 2**8: skeleton_dtype = np.uint8
    elif max_label_skel < 2**16: skeleton_dtype = np.uint16
    else: skeleton_dtype = np.uint32

    skeleton_array = None
    if use_memmap:
        if os.path.exists(skeleton_export_path): os.remove(skeleton_export_path)
        skeleton_array = np.memmap(skeleton_export_path, dtype=skeleton_dtype, mode='w+', shape=segmented_array.shape)
    else: 
        skeleton_array = np.zeros(segmented_array.shape, dtype=skeleton_dtype)
    skeleton_array[:] = 0
    if use_memmap: skeleton_array.flush()

    all_branch_data_dfs = []
    summary_stats_list = []

    for label_val in tqdm(labels_to_process, desc="Analyzing 3D Skeletons"):
        loc_index = label_val - 1
        if loc_index < 0 or loc_index >= len(locations) or locations[loc_index] is None:
            summary_stats_list.append({
                'label': label_val, 'true_num_branches': 0, 'skan_total_length_um': 0.0,
                'skan_avg_branch_length_um': np.nan, 'true_num_junctions': 0,
                'true_num_endpoints': 0, 'skan_num_skeleton_voxels': 0
            })
            continue
        
        slices = locations[loc_index]
        label_region = segmented_array[slices]
        offset = np.array([s.start for s in slices])

        # --- Call the new 3D worker, matching the 2D file's structure ---
        branch_data, label_skeleton_img_local, true_branches, n_junctions, n_endpoints, total_length = \
            analyze_skeleton_with_skan_3d(label_region, offset, spacing, label_val, prune_spurs_le_um)

        avg_branch_length = (total_length / true_branches) if true_branches > 0 else 0.0
        num_skel_voxels = np.count_nonzero(label_skeleton_img_local)

        # Store CORRECTED statistics with names matching the 2D version
        summary_stats_list.append({
            'label': label_val, 'true_num_branches': true_branches,
            'skan_total_length_um': total_length, 'skan_avg_branch_length_um': avg_branch_length,
            'true_num_junctions': n_junctions, 'true_num_endpoints': n_endpoints,
            'skan_num_skeleton_voxels': num_skel_voxels
        })

        if not branch_data.empty:
            branch_data['label'] = label_val
            all_branch_data_dfs.append(branch_data.copy())
        
        # Map skeleton to global array (3D indexing)
        if label_skeleton_img_local is not None and np.any(label_skeleton_img_local):
            local_coords = np.argwhere(label_skeleton_img_local > 0)
            if local_coords.size > 0:
                global_coords = local_coords + offset
                idx_z, idx_x, idx_y = global_coords[:, 0], global_coords[:, 1], global_coords[:, 2]
                valid_idx = ((idx_z >= 0) & (idx_z < skeleton_array.shape[0]) & 
                             (idx_x >= 0) & (idx_x < skeleton_array.shape[1]) &
                             (idx_y >= 0) & (idx_y < skeleton_array.shape[2]))
                idx_z, idx_x, idx_y = idx_z[valid_idx], idx_x[valid_idx], idx_y[valid_idx]
                skeleton_array[idx_z, idx_x, idx_y] = label_val
                if use_memmap: skeleton_array.flush()

    detailed_branch_df = pd.concat(all_branch_data_dfs, ignore_index=True) if all_branch_data_dfs else pd.DataFrame()
    ramification_summary_df = pd.DataFrame(summary_stats_list)

    print("DEBUG [calculate_ramification_3d] Finished CORRECTED 3D analysis.")
    return ramification_summary_df, detailed_branch_df, skeleton_array


# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_segmentation(segmented_array, 
                         spacing=(1.0, 1.0, 1.0),
                         calculate_distances=True,
                         calculate_skeletons=True,
                         skeleton_export_path=None,
                         temp_dir=None, n_jobs=None,
                         return_detailed=False,
                         prune_spurs_le_um=None): # <-- ADDED PARAM
    """
    Comprehensive analysis of a 3D segmented array with optional smoothing/pruning.
    """
    overall_start_time = time.time()
    print("\n" + "=" * 30)
    print("DEBUG [analyze_segmentation] Starting Comprehensive Analysis")
    print(f"DEBUG [analyze_segmentation] Input array shape: {segmented_array.shape}, dtype: {segmented_array.dtype}")
    print(f"DEBUG [analyze_segmentation] Voxel spacing (z,x,y): {spacing} um")
    print("=" * 30)

    if prune_spurs_le_um == 0:
        prune_spurs_le_um = None # Treat 0 as no smoothing

    labels = np.unique(segmented_array)
    labels = labels[labels > 0] # Exclude background
    if len(labels) == 0:
        print("DEBUG [analyze_segmentation] No non-background labels found. Exiting.")
        return pd.DataFrame(), {}

    print(f"DEBUG [analyze_segmentation] Found {len(labels)} non-background labels. First few: {labels[:10]}")

    # --- 1. Basic Volume & Depth Metrics ---
    print("\nDEBUG [analyze_segmentation] [1/3] Calculating Volume & Depth Metrics...")
    volume_df = calculate_volume(segmented_array, spacing=spacing)
    depth_df = calculate_depth_df(segmented_array, spacing=spacing)

    if not volume_df.empty: metrics_df = pd.merge(volume_df, depth_df, on='label', how='left')
    elif not depth_df.empty: metrics_df = depth_df
    else: print("Warning: Could not calculate volume or depth metrics."); metrics_df = pd.DataFrame({'label': labels})
    print(f"DEBUG [analyze_segmentation] Initial metrics_df shape after vol/depth: {metrics_df.shape}")

    # Initialize dictionary for detailed outputs
    detailed_outputs = {}

    # --- 2. Distance Metrics ---
    if calculate_distances:
        print("\nDEBUG [analyze_segmentation] [2/3] Calculating Pairwise Distances...")
        if len(labels) > 1:
            distance_matrix_df, all_pairs_points_df = shortest_distance(
                segmented_array, spacing=spacing, temp_dir=temp_dir, n_jobs=n_jobs
            )

            closest_neighbor_labels = []; shortest_distances = []
            point_self_z, point_self_x, point_self_y = [], [], []
            point_neigh_z, point_neigh_x, point_neigh_y = [], [], []

            # Ensure matrix index/columns match the labels list for correct indexing
            distance_matrix_df = distance_matrix_df.reindex(index=labels, columns=labels)
            dist_values = distance_matrix_df.values

            for i, label in enumerate(labels):
                row_dists = dist_values[i, :].copy(); row_dists[i] = np.inf
                if np.all(np.isinf(row_dists)): min_idx = -1; shortest_dist = np.inf; closest_neighbor = None
                else: min_idx = np.nanargmin(row_dists); shortest_dist = row_dists[min_idx]; closest_neighbor = labels[min_idx]

                closest_neighbor_labels.append(closest_neighbor); shortest_distances.append(shortest_dist)

                psx, psy, psz, pnx, pny, pnz = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan # Init with NaN
                if closest_neighbor is not None:
                    pair_row = all_pairs_points_df[ (all_pairs_points_df['mask1'] == label) & (all_pairs_points_df['mask2'] == closest_neighbor) ]
                    if not pair_row.empty:
                        psz = pair_row.iloc[0]['mask1_z']; psx = pair_row.iloc[0]['mask1_x']; psy = pair_row.iloc[0]['mask1_y']
                        pnz = pair_row.iloc[0]['mask2_z']; pnx = pair_row.iloc[0]['mask2_x']; pny = pair_row.iloc[0]['mask2_y']

                point_self_z.append(psz); point_self_x.append(psx); point_self_y.append(psy)
                point_neigh_z.append(pnz); point_neigh_x.append(pnx); point_neigh_y.append(pny)

                # --- DEBUG ---
                if label == DEBUG_TARGET_LABEL or (DEBUG_TARGET_LABEL is None and i < 3):
                    debug_print(f"Label {label}: Closest neigh={closest_neighbor}, dist={shortest_dist:.2f}. PointSelf(ZXY)=({psz},{psx},{psy}), PointNeigh(ZXY)=({pnz},{pnx},{pny})", label=label)
                # --- END DEBUG ---

            # Add distance info to metrics_df
            dist_info_df = pd.DataFrame({
                'label': labels, 'closest_neighbor_label': closest_neighbor_labels, 'shortest_distance_um': shortest_distances,
                'point_on_self_z': point_self_z, 'point_on_self_x': point_self_x, 'point_on_self_y': point_self_y,
                'point_on_neighbor_z': point_neigh_z, 'point_on_neighbor_x': point_neigh_x, 'point_on_neighbor_y': point_neigh_y
            })
            metrics_df = pd.merge(metrics_df, dist_info_df, on='label', how='left')
            print(f"DEBUG [analyze_segmentation] metrics_df shape after distance merge: {metrics_df.shape}")

            if return_detailed:
                detailed_outputs['distance_matrix'] = distance_matrix_df
                detailed_outputs['all_pairs_points'] = all_pairs_points_df
        else: # Only 0/1 label
            print("DEBUG [analyze_segmentation] Skipping distance calculation (<= 1 label found).")
            for col in ['closest_neighbor_label', 'shortest_distance_um', 'point_on_self_z', 'point_on_self_x', 'point_on_self_y','point_on_neighbor_z', 'point_on_neighbor_x', 'point_on_neighbor_y']:
                 if col not in metrics_df.columns: metrics_df[col] = np.nan if 'point' in col else (np.inf if 'dist' in col else None)
            if return_detailed: detailed_outputs['distance_matrix'] = pd.DataFrame(index=labels, columns=labels) if len(labels)==1 else pd.DataFrame(); detailed_outputs['all_pairs_points'] = pd.DataFrame()
    else: print("DEBUG [analyze_segmentation] Skipping distance calculation as requested.")

    # --- 3. Skeleton Metrics (using skan) ---
    if calculate_skeletons:
        print("\nDEBUG [analyze_segmentation] [3/3] Calculating Skeleton Metrics (3D skan)...")
        labels_for_skeleton = list(metrics_df['label'].unique())
        
        # --- Call the modified function with the new parameter, matching 2D version ---
        ramification_summary_df, detailed_branch_df, skeleton_array = calculate_ramification_with_skan(
            segmented_array, spacing=spacing, labels=labels_for_skeleton, 
            skeleton_export_path=skeleton_export_path,
            prune_spurs_le_um=prune_spurs_le_um
        )
        if not ramification_summary_df.empty: 
            metrics_df = pd.merge(metrics_df, ramification_summary_df, on='label', how='left')
        else: # Add empty columns with new names to match 2D output
            skel_cols_3d = ['true_num_branches', 'skan_total_length_um', 'skan_avg_branch_length_um', 'true_num_junctions', 'true_num_endpoints', 'skan_num_skeleton_voxels']
            for col in skel_cols_3d:
                if col not in metrics_df.columns: metrics_df[col] = np.nan

        if return_detailed:
            detailed_outputs['detailed_branches'] = detailed_branch_df
            detailed_outputs['skeleton_array'] = skeleton_array

    else: print("DEBUG [analyze_segmentation] Skipping skeleton calculation as requested.")

    # Reorder columns
    cols = ['label'] + [col for col in metrics_df if col != 'label']
    metrics_df = metrics_df[cols]
    metrics_df = metrics_df.fillna(value=np.nan)

    print(f"DEBUG [analyze_segmentation] Analysis completed in {time.time() - overall_start_time:.2f} seconds.")
    print("=" * 30 + "\n")

    if return_detailed:
        return metrics_df, detailed_outputs
    else:
        # Cleanup memmap if not returned
        if isinstance(detailed_outputs.get('skeleton_array'), np.memmap):
            skel_path = detailed_outputs['skeleton_array'].filename
            del detailed_outputs['skeleton_array']; gc.collect(); time.sleep(0.1)
            try:
                if os.path.exists(skel_path): os.remove(skel_path); print(f"DEBUG [analyze_segmentation] Cleaned up skeleton memmap file: {skel_path}")
            except Exception as e: print(f"Warning: Could not delete skeleton memmap file {skel_path}: {e}")
        return metrics_df, {}