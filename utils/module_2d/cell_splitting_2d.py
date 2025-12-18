import time
import gc
from typing import Dict, Any, Tuple, Optional, Set, List

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_dilation, footprint_rectangle, disk  # type: ignore
from skimage.segmentation import relabel_sequential, watershed  # type: ignore
from skimage.measure import regionprops  # type: ignore
from scipy.ndimage import distance_transform_edt, binary_fill_holes, find_objects  # type: ignore
from tqdm import tqdm


def _analyze_local_intensity_difference_2d(
    region1_mask: np.ndarray,
    region2_mask: np.ndarray,
    parent_cell_mask: np.ndarray,
    intensity_vol_local: np.ndarray,
    local_analysis_radius: int,
    threshold: float
) -> bool:
    """Analyzes intensity difference between two adjacent regions in 2D."""
    # Use disk for 2D neighborhood
    footprint = disk(local_analysis_radius) if local_analysis_radius > 1 else footprint_rectangle((3, 3))

    r1m = region1_mask.astype(bool)
    r2m = region2_mask.astype(bool)
    pcm = parent_cell_mask.astype(bool)

    dr1 = binary_dilation(r1m, footprint=footprint)
    dr2 = binary_dilation(r2m, footprint=footprint)

    # Zone where they touch/overlap within the parent
    interface_zone = dr1 & dr2 & pcm
    
    # Exclude the regions themselves to get the "boundary" pixels
    la_r1 = interface_zone & ~r1m & dr2
    la_r2 = interface_zone & ~r2m & dr1

    if np.sum(la_r1) < 5 or np.sum(la_r2) < 5:
        return False  # Too small to judge

    m1 = np.mean(intensity_vol_local[la_r1])
    m2 = np.mean(intensity_vol_local[la_r2])
    
    ref_i = max(m1, m2)
    if ref_i < 1e-6:
        return True
    
    rel_diff = abs(m1 - m2) / ref_i
    return rel_diff >= threshold


def _calculate_interface_metrics_2d(
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    parent_mask: np.ndarray,
    intensity: np.ndarray,
    ref_soma_intensity: float,
    radius: int,
    diff_thresh: float,
    ratio_thresh: float
) -> Dict[str, Any]:
    """Calculates metrics for merging decision between two segments."""
    metrics = {'should_merge': False}
    
    footprint = footprint_rectangle((3, 3))
    dilated_A = binary_dilation(mask_A, footprint=footprint)
    
    # Interface pixels
    interface_mask = dilated_A & mask_B & parent_mask
    
    if not np.any(interface_mask):
        return metrics

    mean_int_interface = np.mean(intensity[interface_mask])
    
    # 1. Path Intensity Ratio Check
    # Is the valley between them bright enough compared to the soma?
    ratio = mean_int_interface / max(ref_soma_intensity, 1e-6)
    ratio_pass = ratio < ratio_thresh # If ratio is LOW, it's a deep valley -> Separation is good.

    # 2. Local Intensity Difference Check
    # Do they look like distinct objects locally?
    lid_pass = _analyze_local_intensity_difference_2d(
        mask_A, mask_B, parent_mask, intensity, radius, diff_thresh
    )

    # If EITHER check fails (high valley OR similar intensity), we suggest merging
    if not ratio_pass or not lid_pass:
        metrics['should_merge'] = True
        
    return metrics


def _build_adjacency_graph_2d(
    segments_mask: np.ndarray,
    parent_mask: np.ndarray,
    soma_mask: np.ndarray,
    soma_props: Dict[int, Dict[str, Any]],
    intensity: np.ndarray,
    radius: int,
    diff_thresh: float,
    ratio_thresh: float
) -> Tuple[Dict[int, Any], Dict[Tuple[int, int], Any]]:
    """Builds graph of adjacent segments and calculates merge metrics."""
    nodes = {}
    edges = {}
    
    labels = np.unique(segments_mask[segments_mask > 0])
    if len(labels) <= 1:
        return nodes, edges

    # Initialize Nodes
    for lbl in labels:
        mask = (segments_mask == lbl)
        seeds = np.unique(soma_mask[mask])
        nodes[lbl] = {
            'area': np.sum(mask),
            'somas': [s for s in seeds if s > 0]
        }

    footprint = footprint_rectangle((3, 3))

    # Check Adjacency
    for i in range(len(labels)):
        lbl_A = labels[i]
        mask_A = (segments_mask == lbl_A)
        dil_A = binary_dilation(mask_A, footprint=footprint)
        
        # Find neighbors
        neighbors = np.unique(segments_mask[dil_A & (segments_mask != lbl_A) & (segments_mask > 0)])
        
        for lbl_B in neighbors:
            if lbl_B <= lbl_A: continue # Only check pair once
            
            edge_key = (lbl_A, lbl_B)
            if edge_key in edges: continue
            
            mask_B = (segments_mask == lbl_B)
            
            # Get reference intensity from somas in both segments
            somas_involved = nodes[lbl_A]['somas'] + nodes[lbl_B]['somas']
            soma_ints = [soma_props[s]['mean_intensity'] for s in somas_involved if s in soma_props]
            ref_int = np.mean(soma_ints) if soma_ints else 1.0

            edges[edge_key] = _calculate_interface_metrics_2d(
                mask_A, mask_B, parent_mask, intensity,
                ref_int, radius, diff_thresh, ratio_thresh
            )
            
    return nodes, edges


def separate_multi_soma_cells_2d(
    segmentation_mask: np.ndarray,
    intensity_volume: np.ndarray,
    soma_mask: np.ndarray,
    spacing: Optional[Tuple[float, float]],
    min_size_threshold: int = 50,
    max_seed_centroid_dist: float = 30.0,
    min_path_intensity_ratio: float = 0.8,
    min_local_intensity_difference: float = 0.05,
    local_analysis_radius: int = 5,
    max_hole_size: int = 0
) -> np.ndarray:
    """
    Separates merged cells in 2D using seeded watershed and graph-based merging.

    Args:
        segmentation_mask: 2D mask of cells.
        intensity_volume: 2D intensity image.
        soma_mask: 2D mask of seeds.
        spacing: Pixel spacing.
        min_size_threshold: Min area for final cells.
        max_seed_centroid_dist: (Unused in this logic, kept for sig match).
        min_path_intensity_ratio: Threshold for valley depth.
        min_local_intensity_difference: Threshold for local contrast.
        local_analysis_radius: Radius for local contrast check.
        max_hole_size: Max size of holes to fill.

    Returns:
        np.ndarray: Refined 2D segmentation mask.
    """
    print("[SepMultiSoma_2D] Starting Cell Separation...")
    
    if spacing is None: spacing = (1.0, 1.0)
    
    final_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
    
    # 1. Identify Multi-Soma Cells
    unique_cells = np.unique(segmentation_mask[segmentation_mask > 0])
    if len(unique_cells) == 0:
        return final_mask

    cell_to_somas = {}
    # Fast lookup
    soma_locs = find_objects(soma_mask)
    for i, sl in enumerate(soma_locs):
        if sl is None: continue
        soma_id = i + 1
        # Find which cell this soma belongs to
        cell_ids = np.unique(segmentation_mask[sl][soma_mask[sl] == soma_id])
        for cid in cell_ids:
            if cid > 0:
                cell_to_somas.setdefault(cid, set()).add(soma_id)

    multi_soma_cells = [c for c, s in cell_to_somas.items() if len(s) > 1]
    
    # Initialize output with single-soma cells (pass-through)
    for lbl in unique_cells:
        if lbl not in multi_soma_cells:
            final_mask[segmentation_mask == lbl] = lbl

    next_label = np.max(unique_cells) + 1 if len(unique_cells) > 0 else 1
    
    print(f"  Processing {len(multi_soma_cells)} multi-soma cells...")

    # 2. Process Multi-Soma Cells
    slices = find_objects(segmentation_mask)
    
    for cell_id in tqdm(multi_soma_cells, desc="Splitting Cells"):
        bbox = slices[cell_id - 1]
        if not bbox: continue
        
        # Pad bbox
        pad = 2
        sl = tuple(slice(max(0, s.start - pad), min(d, s.stop + pad)) 
                   for s, d in zip(bbox, segmentation_mask.shape))
        
        local_mask = (segmentation_mask[sl] == cell_id)
        local_soma = soma_mask[sl]
        local_int = intensity_volume[sl]
        
        seeds = np.unique(local_soma[local_mask])
        seeds = seeds[seeds > 0]
        
        if len(seeds) < 2:
            # Should not happen based on logic above, but safety check
            final_mask[sl][local_mask] = next_label
            next_label += 1
            continue

        # Calculate Soma Props for Intensity Ref
        soma_props = {}
        for s in seeds:
            m = (local_soma == s)
            mean_i = np.mean(local_int[m]) if np.any(m) else 0
            soma_props[s] = {'mean_intensity': mean_i}

        # A. Watershed
        dt = distance_transform_edt(local_mask, sampling=spacing)
        landscape = -dt
        # Simple markers
        markers = np.zeros_like(local_mask, dtype=np.int32)
        for i, s in enumerate(seeds):
            markers[local_soma == s] = i + 1
            
        ws = watershed(landscape, markers, mask=local_mask)
        
        # B. Graph Merge
        nodes, edges = _build_adjacency_graph_2d(
            ws, local_mask, local_soma, soma_props, local_int,
            local_analysis_radius, min_local_intensity_difference,
            min_path_intensity_ratio
        )
        
        # Resolve Merges (Union-Find)
        merge_map = {i: i for i in range(len(seeds) + 1)}
        for (id_a, id_b), metrics in edges.items():
            if metrics['should_merge']:
                root_a = merge_map[id_a]
                root_b = merge_map[id_b]
                target = min(root_a, root_b)
                source = max(root_a, root_b)
                # Update all pointing to source
                for k, v in merge_map.items():
                    if v == source:
                        merge_map[k] = target

        # Apply Merges
        merged_local = np.zeros_like(ws)
        for old_id in np.unique(ws[ws > 0]):
            new_id = merge_map[old_id]
            merged_local[ws == old_id] = new_id
            
        # Clean up (Relabel local sequentially)
        merged_local, _, _ = relabel_sequential(merged_local)
        
        # Write to global
        result_view = final_mask[sl]
        for l_id in np.unique(merged_local[merged_local > 0]):
            result_view[merged_local == l_id] = next_label
            next_label += 1
        final_mask[sl] = result_view

    # 3. Finalize (Fill Voids)
    if max_hole_size > 0:
        final_mask = fill_internal_voids_2d(final_mask, max_hole_size)
        
    return final_mask


def fill_internal_voids_2d(mask: np.ndarray, max_size: int) -> np.ndarray:
    """Fills small internal holes in the segmentation."""
    print(f"  Filling voids <= {max_size} px...")
    out = mask.copy()
    labels = np.unique(mask[mask > 0])
    slices = find_objects(mask)
    
    for lbl in tqdm(labels, desc="Filling"):
        sl = slices[lbl-1]
        if not sl: continue
        
        roi = out[sl]
        obj = (roi == lbl)
        
        filled = binary_fill_holes(obj)
        holes = filled & ~obj
        
        if not np.any(holes): continue
        
        # Check size of holes
        labeled_holes, n = ndimage.label(holes)
        if n == 0: continue
        
        sizes = np.bincount(labeled_holes.ravel())
        valid_holes = np.where((sizes > 0) & (sizes <= max_size))[0]
        
        if valid_holes.size > 0:
            fill_mask = np.isin(labeled_holes, valid_holes)
            roi[fill_mask] = lbl
            
    return out