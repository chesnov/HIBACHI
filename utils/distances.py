import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import time

def shortest_surface_distances(segmented_array, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate the shortest distance between surfaces of all masks in a 3D segmented array,
    accounting for anisotropic voxel spacing.
    
    Parameters:
    -----------
    segmented_array : numpy.ndarray
        3D numpy array (z, x, y) containing integer labels for each mask.
        0 is assumed to be the background.
    spacing : tuple
        Tuple of (z_spacing, x_spacing, y_spacing) in microns.
    
    Returns:
    --------
    distances_matrix : numpy.ndarray
        n x n matrix where n is the number of unique masks (excluding background).
        Entry (i,j) contains the shortest distance in microns between the surface
        of mask i+1 and mask j+1.
    
    points_matrix : numpy.ndarray
        n x n x 6 matrix where entry (i,j) contains the coordinates of the closest
        point pair between masks [i+1, j+1] in the format:
        [z1, x1, y1, z2, x2, y2] where (z1,x1,y1) is on mask i+1 and (z2,x2,y2) is on mask j+1.
    """
    start_time = time.time()
    
    # Get unique labels excluding background (0)
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    n_labels = len(labels)
    
    # Initialize result matrices
    distances_matrix = np.zeros((n_labels, n_labels))
    points_matrix = np.zeros((n_labels, n_labels, 6))
    
    # Cache to avoid recalculating surface points
    surface_points_cache = {}
    kdtrees_cache = {}
    
    print(f"Found {n_labels} unique masks. Starting distance calculations...")
    
    def get_surface_points(label):
        """Helper function to extract surface points of a given mask"""
        if label in surface_points_cache:
            return surface_points_cache[label]
        
        # Create binary mask
        mask = (segmented_array == label)
        
        # Get surface by eroding
        eroded = mask.copy()
        eroded[1:-1, 1:-1, 1:-1] = np.logical_and(mask[1:-1, 1:-1, 1:-1], 
                                                  np.logical_and(mask[:-2, 1:-1, 1:-1], 
                                                  np.logical_and(mask[2:, 1:-1, 1:-1],
                                                  np.logical_and(mask[1:-1, :-2, 1:-1], 
                                                  np.logical_and(mask[1:-1, 2:, 1:-1], 
                                                  np.logical_and(mask[1:-1, 1:-1, :-2],
                                                                 mask[1:-1, 1:-1, 2:]))))))
        surface = np.logical_and(mask, np.logical_not(eroded))
        
        # Get surface points coordinates
        surface_points = np.array(np.where(surface)).T
        
        # Scale coordinates by spacing
        scaled_surface_points = surface_points * np.array(spacing)
        
        # Cache results
        surface_points_cache[label] = (surface_points, scaled_surface_points)
        
        return surface_points, scaled_surface_points
    
    def get_tree(label):
        """Helper function to get or create KDTree for a mask's surface points"""
        if label in kdtrees_cache:
            return kdtrees_cache[label]
        
        _, scaled_points = get_surface_points(label)
        tree = cKDTree(scaled_points)
        kdtrees_cache[label] = tree
        
        return tree
    
    # Calculate distances between all mask pairs
    for i, label1 in enumerate(labels):
        for j, label2 in enumerate(labels):
            if i == j:
                # Same mask, distance is 0
                distances_matrix[i, j] = 0
                continue
                
            if distances_matrix[j, i] > 0:
                # If we've already calculated distance(j,i), use symmetry
                distances_matrix[i, j] = distances_matrix[j, i]
                points_matrix[i, j] = np.array([*points_matrix[j, i][3:], *points_matrix[j, i][:3]])
                continue
            
            # Get surface points and their KDTree
            surface_points1, scaled_points1 = get_surface_points(label1)
            tree2 = get_tree(label2)
            
            # Query the nearest neighbor for each point in mask1
            distances, indices = tree2.query(scaled_points1)
            
            # Find the minimum distance and corresponding points
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            # Get corresponding points
            point1 = surface_points1[min_idx]
            _, scaled_points2 = get_surface_points(label2)
            point2 = np.array(np.where(segmented_array == label2)).T[indices[min_idx]]
            
            # Store results
            distances_matrix[i, j] = min_distance
            points_matrix[i, j] = np.array([*point1, *point2])
            
            if (i + j) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i*n_labels + j + 1}/{n_labels*n_labels} pairs in {elapsed:.2f} seconds")
    
    elapsed = time.time() - start_time
    print(f"Finished all calculations in {elapsed:.2f} seconds")
    return distances_matrix, points_matrix


def shortest_distance_optimized(segmented_array, spacing=(1.0, 1.0, 1.0)):
    """
    More efficient approach using distance transforms to calculate the shortest distances
    between masks in a 3D segmented array.
    
    Parameters:
    -----------
    segmented_array : numpy.ndarray
        3D numpy array (z, x, y) containing integer labels for each mask.
        0 is assumed to be the background.
    spacing : tuple
        Tuple of (z_spacing, x_spacing, y_spacing) in microns.
    
    Returns:
    --------
    distances_matrix : numpy.ndarray
        n x n matrix where n is the number of unique masks (excluding background).
        Entry (i,j) contains the shortest distance in microns between the surface
        of mask i+1 and mask j+1.
    
    points_matrix : numpy.ndarray
        n x n x 6 matrix where entry (i,j) contains the coordinates of the closest
        point pair between masks [i+1, j+1] in the format:
        [z1, x1, y1, z2, x2, y2] where (z1,x1,y1) is on mask i+1 and (z2,x2,y2) is on mask j+1.
    """
    start_time = time.time()
    
    # Get unique labels excluding background (0)
    labels = np.unique(segmented_array)
    labels = labels[labels > 0]
    n_labels = len(labels)
    
    # Initialize result matrices
    distances_matrix = np.zeros((n_labels, n_labels))
    points_matrix = np.zeros((n_labels, n_labels, 6))
    
    # Pre-compute distance maps for each label
    dist_maps = {}
    nearest_points = {}
    
    print(f"Found {n_labels} unique masks. Preprocessing...")
    
    for i, label in enumerate(labels):
        # Create binary mask
        mask = (segmented_array == label)
        
        # Calculate distance transform
        # The distance_transform_edt function takes account of the spacing parameter
        dist_map = distance_transform_edt(~mask, sampling=spacing)
        
        # Store distance map
        dist_maps[label] = dist_map
        
        # Create coordinate grids
        z, x, y = np.indices(segmented_array.shape)
        coords = np.stack([z, x, y], axis=-1)
        
        # Store coordinates for later use
        nearest_points[label] = coords
        
        print(f"Processed mask {i+1}/{n_labels}")
    
    print("Calculating distances between masks...")
    
    # Calculate distances between all mask pairs
    for i, label1 in enumerate(labels):
        mask1 = (segmented_array == label1)
        
        for j, label2 in enumerate(labels):
            if i == j:
                # Same mask, distance is 0
                distances_matrix[i, j] = 0
                continue
                
            if distances_matrix[j, i] > 0:
                # If we've already calculated distance(j,i), use symmetry
                distances_matrix[i, j] = distances_matrix[j, i]
                points_matrix[i, j] = np.array([*points_matrix[j, i][3:], *points_matrix[j, i][:3]])
                continue
            
            # Get distance map for label2
            dist_map2 = dist_maps[label2]
            
            # Extract distances from the surface of label2 to each voxel in label1
            mask1_distances = dist_map2[mask1]
            
            # Find minimum distance and its location
            if len(mask1_distances) > 0:
                min_distance = np.min(mask1_distances)
                
                # Find the point in mask1 closest to mask2
                point1_idx = np.where(mask1)[0][np.argmin(mask1_distances)]
                point1 = np.unravel_index(point1_idx, segmented_array.shape)
                
                # Find the closest point on surface of mask2
                mask2 = (segmented_array == label2)
                
                # Create binary image of the mask2 surface
                eroded2 = mask2.copy()
                eroded2[1:-1, 1:-1, 1:-1] = np.logical_and(mask2[1:-1, 1:-1, 1:-1], 
                                                         np.logical_and(mask2[:-2, 1:-1, 1:-1], 
                                                         np.logical_and(mask2[2:, 1:-1, 1:-1],
                                                         np.logical_and(mask2[1:-1, :-2, 1:-1], 
                                                         np.logical_and(mask2[1:-1, 2:, 1:-1], 
                                                         np.logical_and(mask2[1:-1, 1:-1, :-2],
                                                                        mask2[1:-1, 1:-1, 2:]))))))
                surface2 = np.logical_and(mask2, np.logical_not(eroded2))
                
                # Create a distance map from point1 to all points
                point_dist = np.sqrt(
                    ((np.indices(segmented_array.shape)[0] - point1[0]) * spacing[0])**2 +
                    ((np.indices(segmented_array.shape)[1] - point1[1]) * spacing[1])**2 +
                    ((np.indices(segmented_array.shape)[2] - point1[2]) * spacing[2])**2
                )
                
                # Find closest point on surface2 to point1
                surface2_distances = point_dist[surface2]
                point2_idx = np.where(surface2)[0][np.argmin(surface2_distances)]
                point2 = np.unravel_index(point2_idx, segmented_array.shape)
                
                # Store results
                distances_matrix[i, j] = min_distance
                points_matrix[i, j] = np.array([*point1, *point2])
            else:
                # This should not happen with valid masks
                distances_matrix[i, j] = np.inf
            
            if (i * n_labels + j) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {i*n_labels + j + 1}/{n_labels*n_labels} pairs in {elapsed:.2f} seconds")
    
    elapsed = time.time() - start_time
    print(f"Finished all calculations in {elapsed:.2f} seconds")
    return distances_matrix, points_matrix