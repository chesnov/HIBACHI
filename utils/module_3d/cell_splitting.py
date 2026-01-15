import os
import gc
import shutil
from typing import List, Dict, Optional, Tuple, Set, Iterator, Any

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_dilation, footprint_rectangle, ball  # type: ignore
from skimage.segmentation import relabel_sequential  # type: ignore
from tqdm import tqdm

# Import shared helpers
try:
    from .segmentation_helpers import (
        flush_print,
        _watershed_with_simpleitk,
        distance_transform_edt
    )
except ImportError:
    # Fallback for running script directly
    from segmentation_helpers import (
        flush_print,
        _watershed_with_simpleitk,
        distance_transform_edt
    )


def _get_chunk_slices(
    volume_shape: Tuple[int, ...],
    chunk_shape: Tuple[int, ...],
    overlap: int
) -> Iterator[Tuple[slice, ...]]:
    """
    Generator that yields slices for overlapping chunks in 3D.

    Args:
        volume_shape: Shape of the full volume (Z, Y, X).
        chunk_shape: Desired shape of each chunk.
        overlap: Overlap size in pixels/voxels.

    Yields:
        Tuple of slices defining the chunk coordinates.
    """
    for z in range(0, volume_shape[0], chunk_shape[0] - overlap):
        for y in range(0, volume_shape[1], chunk_shape[1] - overlap):
            for x in range(0, volume_shape[2], chunk_shape[2] - overlap):
                yield (
                    slice(z, min(z + chunk_shape[0], volume_shape[0])),
                    slice(y, min(y + chunk_shape[1], volume_shape[1])),
                    slice(x, min(x + chunk_shape[2], volume_shape[2])),
                )


# =============================================================================
# Graph & Metric Functions
# =============================================================================

def _analyze_local_intensity_difference_optimized(
    interface_mask: np.ndarray,
    region1_mask: np.ndarray,
    region2_mask: np.ndarray,
    intensity_vol_local: np.ndarray,
    local_analysis_radius: int,
    min_local_intensity_difference_threshold: float
) -> bool:
    """
    Analyzes relative intensity difference between two regions at their interface.

    Args:
        interface_mask: Mask of the interface boundary.
        region1_mask: Mask of object A.
        region2_mask: Mask of object B.
        intensity_vol_local: Local intensity volume.
        local_analysis_radius: Radius for dilation to find local neighborhood.
        min_local_intensity_difference_threshold: Threshold for relative difference.

    Returns:
        bool: True if the regions are distinct enough, False if they should merge.
    """
    footprint_elem = (
        ball(local_analysis_radius) if local_analysis_radius > 1
        else footprint_rectangle((3, 3, 3))
    )
    
    # Define local analysis zone around the interface
    analysis_zone = binary_dilation(interface_mask, footprint=footprint_elem)
    
    # Extract pixels belonging to R1 and R2 within that zone
    la_r1 = analysis_zone & region1_mask
    la_r2 = analysis_zone & region2_mask

    # If regions are too small locally, assume they are distinct (safe default)
    if np.sum(la_r1) < 20 or np.sum(la_r2) < 20:
        return True

    m1 = np.mean(intensity_vol_local[la_r1])
    m2 = np.mean(intensity_vol_local[la_r2])

    ref_i = max(m1, m2)
    if ref_i < 1e-6:
        return True

    # Calculate relative difference
    rel_diff = abs(m1 - m2) / ref_i
    return rel_diff >= min_local_intensity_difference_threshold


def _calculate_interface_metrics(
    mask_A_local: np.ndarray,
    mask_B_local: np.ndarray,
    parent_mask_local: np.ndarray,
    intensity_local: np.ndarray,
    avg_soma_intensity_for_interface: float,
    spacing_tuple: Optional[Tuple[float, float, float]],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float,
) -> Dict[str, Any]:
    """
    Calculates metrics to decide if two watershed basins should be merged.
    Considers 'Valley Depth' (intensity ratio) and 'Local Contrast'.
    """
    metrics = {'should_merge_decision': False}
    footprint_dilation = footprint_rectangle((3, 3, 3))
    
    # Identify interface pixels
    dilated_A = binary_dilation(mask_A_local, footprint=footprint_dilation)
    interface_mask = dilated_A & mask_B_local & parent_mask_local

    if not np.any(interface_mask):
        return metrics

    # 1. Valley Depth Check (Path Intensity Ratio)
    mean_interface_intensity = np.mean(intensity_local[interface_mask])
    ratio = mean_interface_intensity / max(avg_soma_intensity_for_interface, 1e-6)
    
    # If ratio is LOW, it means there is a deep, dark valley -> Keep separate
    ratio_threshold_passed = ratio < min_path_intensity_ratio_heuristic

    # 2. Local Contrast Check
    lid_passed = _analyze_local_intensity_difference_optimized(
        interface_mask, mask_A_local, mask_B_local, intensity_local,
        local_analysis_radius, min_local_intensity_difference
    )

    # If EITHER check fails (valley not deep enough OR objects look too similar),
    # we recommend merging.
    if not ratio_threshold_passed or not lid_passed:
        metrics['should_merge_decision'] = True
        
    return metrics


def _build_adjacency_graph_for_cell(
    current_cell_segments_mask_local: np.ndarray,
    original_cell_mask_local: np.ndarray,
    soma_mask_local: np.ndarray,
    soma_props_for_cell: Dict[int, Dict[str, Any]],
    intensity_local: np.ndarray,
    spacing_tuple: Optional[Tuple[float, float, float]],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float
) -> Tuple[Dict[int, Any], Dict[Tuple[int, int], Any]]:
    """
    Builds a Region Adjacency Graph (RAG) for segments within a single cell.
    Returns nodes and edges with merge metrics.
    """
    nodes = {}
    edges = {}
    
    seg_lbls = np.unique(
        current_cell_segments_mask_local[current_cell_segments_mask_local > 0]
    )
    if len(seg_lbls) <= 1:
        return nodes, edges

    footprint_d = footprint_rectangle((3, 3, 3))

    # Initialize Nodes
    for lbl in seg_lbls:
        mask = (current_cell_segments_mask_local == lbl)
        seeds_inside = np.unique(soma_mask_local[mask])
        nodes[lbl] = {
            'volume': np.sum(mask),
            'orig_somas': [s for s in seeds_inside if s > 0]
        }

    # Find Edges and Calculate Metrics
    for i in range(len(seg_lbls)):
        lbl_A = seg_lbls[i]
        mask_A = (current_cell_segments_mask_local == lbl_A)
        dil_A = binary_dilation(mask_A, footprint=footprint_d)
        
        # Find neighbors intersecting with dilation
        candidate_mask = (
            dil_A &
            (current_cell_segments_mask_local != lbl_A) &
            (current_cell_segments_mask_local > 0)
        )
        candidates = current_cell_segments_mask_local[candidate_mask]

        for lbl_B in np.unique(candidates):
            if lbl_B <= lbl_A:
                continue  # Avoid duplicate checks
            
            edge_key = (lbl_A, lbl_B)
            if edge_key in edges:
                continue

            mask_B = (current_cell_segments_mask_local == lbl_B)
            
            # Reference intensity: mean of somas involved
            somas_A = nodes[lbl_A]['orig_somas']
            somas_B = nodes[lbl_B]['orig_somas']
            all_somas = somas_A + somas_B
            soma_ints = [
                soma_props_for_cell[s]['mean_intensity']
                for s in all_somas if s in soma_props_for_cell
            ]
            ref_intensity = np.mean(soma_ints) if soma_ints else 1.0

            edges[edge_key] = _calculate_interface_metrics(
                mask_A, mask_B, original_cell_mask_local, intensity_local,
                ref_intensity, spacing_tuple, local_analysis_radius,
                min_local_intensity_difference, min_path_intensity_ratio_heuristic
            )
            
    return nodes, edges


# =============================================================================
# Worker Function
# =============================================================================

def _separate_multi_soma_cells_chunk(
    segmentation_mask: np.ndarray,
    intensity_volume: np.ndarray,
    soma_mask: np.ndarray,
    spacing: Optional[Tuple[float, float, float]],
    label_offset: int,
    multi_soma_cell_labels_list: List[int],
    **kwargs
) -> Tuple[np.ndarray, Dict, Dict[int, Set[int]]]:
    """
    Worker: Separates multi-soma cells within a specific 3D chunk.
    
    Returns:
        chunk_result: The processed sub-volume labels.
        (unused dict),
        label_to_seeds_map: Map of {new_label: set(original_seed_ids)} for stitching.
    """
    chunk_result = np.zeros_like(segmentation_mask, dtype=np.int32)
    label_to_seeds_map = {}

    unique_labels = np.unique(segmentation_mask[segmentation_mask > 0])

    # 1. Copy Single Cells (Pass-through)
    # If a cell is not flagged as multi-soma, we preserve it exactly.
    for lbl in unique_labels:
        if lbl not in multi_soma_cell_labels_list:
            chunk_result[segmentation_mask == lbl] = lbl
            seeds = np.unique(soma_mask[segmentation_mask == lbl])
            if seeds.size > 0:
                label_to_seeds_map[lbl] = set(seeds[seeds > 0])

    present_multi_soma = [
        l for l in multi_soma_cell_labels_list if l in unique_labels
    ]

    if not present_multi_soma:
        return chunk_result, {}, label_to_seeds_map

    next_local_label = label_offset
    min_size_thresh = kwargs.get('min_size_threshold', 0)

    # 2. Process Multi-Soma Objects
    for cell_label in present_multi_soma:
        cell_mask_full = (segmentation_mask == cell_label)
        slices = ndimage.find_objects(cell_mask_full)
        if not slices:
            continue
        
        bbox = slices[0]
        # Pad bounding box to avoid boundary artifacts during watershed
        bbox_padded = tuple(
            slice(max(0, s.start - 2), min(dim, s.stop + 2))
            for s, dim in zip(bbox, segmentation_mask.shape)
        )

        local_mask = cell_mask_full[bbox_padded]
        local_soma = soma_mask[bbox_padded]
        local_intensity = intensity_volume[bbox_padded]

        seeds_in_crop = np.unique(local_soma[local_mask])
        seeds_in_crop = seeds_in_crop[seeds_in_crop > 0]

        # Case: < 2 seeds visible in this chunk? Treat as single object.
        if len(seeds_in_crop) < 2:
            new_label = next_local_label
            chunk_view = chunk_result[bbox_padded]
            chunk_view[local_mask] = new_label
            chunk_result[bbox_padded] = chunk_view
            
            label_to_seeds_map[new_label] = set(seeds_in_crop)
            next_local_label += 1
            continue

        # Case: Multiple seeds -> Perform Separation
        soma_props = {}
        for s_id in seeds_in_crop:
            s_mask = (local_soma == s_id)
            mean_i = np.mean(local_intensity[s_mask]) if np.any(s_mask) else 1.0
            soma_props[s_id] = {'mean_intensity': mean_i}

        # A. Seeded Watershed
        # Use SimpleITK helper for speed/memory efficiency
        dt = distance_transform_edt(local_mask, sampling=spacing)
        landscape = -dt
        intensity_weight = kwargs.get('intensity_weight', 0.5)

        if intensity_weight > 0:
            norm_int = (local_intensity - local_intensity.min()) / \
                       (local_intensity.max() - local_intensity.min() + 1e-6)
            landscape += (norm_int * intensity_weight * dt.max())

        markers = np.zeros_like(local_mask, dtype=np.int32)
        for idx, s_id in enumerate(seeds_in_crop):
            markers[local_soma == s_id] = idx + 1

        ws_local = _watershed_with_simpleitk(landscape, markers)
        ws_local[~local_mask] = 0

        # B. Graph-Based Merging
        nodes, edges = _build_adjacency_graph_for_cell(
            ws_local, local_mask, local_soma, soma_props, local_intensity, spacing,
            kwargs.get('local_analysis_radius', 10),
            kwargs.get('min_local_intensity_difference', 0.0),
            kwargs.get('min_path_intensity_ratio', 1.0)
        )

        merge_map = {i: i for i in range(len(seeds_in_crop) + 2)}
        for (id_a, id_b), metrics in edges.items():
            if metrics['should_merge_decision']:
                root_a, root_b = merge_map[id_a], merge_map[id_b]
                target = min(root_a, root_b)
                # Update all pointers
                for k, v in merge_map.items():
                    if v == max(root_a, root_b):
                        merge_map[k] = target

        final_local_mask = np.zeros_like(ws_local)
        unique_ws_ids = np.unique(ws_local[ws_local > 0])
        for old_id in unique_ws_ids:
            merged_id = merge_map[old_id]
            final_local_mask[ws_local == old_id] = merged_id

        # C. Seed-Aware Orphan Reassignment
        # Ensure every fragment actually contains a seed. If not, merge it.
        unique_result_ids = np.unique(final_local_mask[final_local_mask > 0])
        dilation_struct = footprint_rectangle((3, 3, 3))

        for uid in unique_result_ids:
            cell_mask = (final_local_mask == uid)
            cc_labels, num_cc = ndimage.label(cell_mask)

            if num_cc <= 1:
                continue

            # Check disjoint fragments
            for i in range(1, num_cc + 1):
                frag_mask = (cc_labels == i)
                has_seed = np.any(markers[frag_mask] > 0)

                if not has_seed:
                    # Orphan detected: merge into best neighbor
                    dilated = binary_dilation(frag_mask, footprint=dilation_struct)
                    neighbor_labels = final_local_mask[dilated]
                    valid_neighbors = neighbor_labels[
                        (neighbor_labels != 0) & (neighbor_labels != uid)
                    ]

                    if valid_neighbors.size > 0:
                        n_ids, n_counts = np.unique(valid_neighbors, return_counts=True)
                        best_neighbor = n_ids[np.argmax(n_counts)]
                        final_local_mask[frag_mask] = best_neighbor
                    else:
                        if min_size_thresh > 0 and np.sum(frag_mask) < min_size_thresh:
                            final_local_mask[frag_mask] = 0

        # D. Map to Global IDs
        # Relabel locally to be sequential (1..N) before assigning global IDs
        final_local_mask_clean, _, _ = relabel_sequential(final_local_mask)
        
        chunk_result_view = chunk_result[bbox_padded]

        for local_id in np.unique(final_local_mask_clean[final_local_mask_clean > 0]):
            mask_l = (final_local_mask_clean == local_id)
            
            seeds_in_segment = np.unique(local_soma[mask_l])
            seeds_in_segment_set = set(seeds_in_segment[seeds_in_segment > 0])
            
            global_lbl = next_local_label
            chunk_result_view[mask_l] = global_lbl
            
            label_to_seeds_map[global_lbl] = seeds_in_segment_set
            next_local_label += 1

        chunk_result[bbox_padded] = chunk_result_view

    return chunk_result, {}, label_to_seeds_map


# =============================================================================
# Void Filling
# =============================================================================

def fill_internal_voids(segmentation_mask: np.ndarray) -> np.ndarray:
    """Fills internal holes in 3D objects."""
    labels = np.unique(segmentation_mask)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return segmentation_mask

    slices = ndimage.find_objects(segmentation_mask)
    for i, label in enumerate(tqdm(labels, desc="Filling Voids")):
        idx = label - 1
        if idx >= len(slices) or slices[idx] is None:
            continue
        
        sl = slices[idx]
        crop = segmentation_mask[sl]
        obj_mask = (crop == label)
        
        # 3D binary fill holes
        filled_mask = ndimage.binary_fill_holes(obj_mask)
        
        if np.any(filled_mask != obj_mask):
            # Only fill where original was 0 or Self
            safe_fill = filled_mask & ((crop == label) | (crop == 0))
            crop[safe_fill] = label
            segmentation_mask[sl] = crop
            
    return segmentation_mask


# =============================================================================
# Main Coordinator
# =============================================================================

def separate_multi_soma_cells(
    segmentation_mask: np.ndarray,
    intensity_volume: np.ndarray,
    soma_mask: np.ndarray,
    spacing: Optional[Tuple[float, float, float]],
    chunk_shape: Tuple[int, int, int] = (128, 512, 512),
    overlap: int = 32,
    **kwargs
) -> np.ndarray:
    """
    Main Coordinator for separating multi-soma cells in large 3D volumes.
    Uses chunking with seed-aware stitching to handle boundary artifacts.

    Args:
        segmentation_mask: 3D labeled segmentation.
        intensity_volume: 3D intensity volume.
        soma_mask: 3D mask of seeds.
        spacing: Voxel spacing.
        chunk_shape: Size of processing chunks.
        overlap: Overlap between chunks.
        **kwargs: Parameters for separation (weights, thresholds).

    Returns:
        np.ndarray: Refined 3D segmentation mask.
    """
    flush_print("[SepMultiSoma] Starting (Chunked + Seed-Aware)...")

    # 1. Identify Multi-Soma Cells (Global Check)
    unique_labels = np.unique(segmentation_mask[segmentation_mask > 0])
    cell_to_somas: Dict[int, Set[int]] = {}
    
    soma_locs = ndimage.find_objects(soma_mask)
    for s_idx, s_slice in enumerate(soma_locs):
        if s_slice is None:
            continue
        soma_id = s_idx + 1
        # Which cells overlap this soma?
        cells_under = np.unique(
            segmentation_mask[s_slice][soma_mask[s_slice] == soma_id]
        )
        for cell_id in cells_under:
            if cell_id == 0:
                continue
            if cell_id not in cell_to_somas:
                cell_to_somas[cell_id] = set()
            cell_to_somas[cell_id].add(soma_id)

    multi_soma_labels = [c for c, s in cell_to_somas.items() if len(s) > 1]
    
    if not multi_soma_labels:
        flush_print("  No multi-soma cells found. Returning original.")
        return segmentation_mask.copy()

    # 2. Process Chunks
    memmap_dir = kwargs.get("memmap_dir", "ramiseg_temp_memmap")
    if not os.path.exists(memmap_dir):
        os.makedirs(memmap_dir, exist_ok=True)

    chunk_slices = list(
        _get_chunk_slices(segmentation_mask.shape, chunk_shape, overlap)
    )
    chunk_data = {}  # Stores (path, shape, seed_map)

    flush_print(f"  Processing {len(chunk_slices)} chunks...")

    try:
        for i, sl in enumerate(tqdm(chunk_slices, desc="Processing Chunks")):
            seg_chunk = segmentation_mask[sl]
            int_chunk = intensity_volume[sl]
            soma_chunk = soma_mask[sl]

            # Use large offsets to avoid ID collisions between chunks initially
            chunk_offset = (i + 1) * 1_000_000

            res, _, seed_map = _separate_multi_soma_cells_chunk(
                seg_chunk, int_chunk, soma_chunk,
                spacing, chunk_offset, multi_soma_labels, **kwargs
            )

            path = os.path.join(memmap_dir, f"chunk_{i}_{os.getpid()}.npy")
            np.save(path, res)

            chunk_data[i] = {'path': path, 'shape': res.shape, 'seed_map': seed_map}
            del res # Free RAM immediately
            gc.collect()

        # 3. Seed-Aware Stitching Logic
        flush_print("  Stitching with Seed Verification...")
        label_map: Dict[int, int] = {}

        shape_in_chunks = [
            len(range(0, segmentation_mask.shape[0], chunk_shape[0] - overlap)),
            len(range(0, segmentation_mask.shape[1], chunk_shape[1] - overlap)),
            len(range(0, segmentation_mask.shape[2], chunk_shape[2] - overlap))
        ]

        for i, chunk_slice1 in enumerate(tqdm(chunk_slices, desc="Stitching Analysis")):
            if i not in chunk_data:
                continue

            # Determine neighbors
            cz, cy, cx = np.unravel_index(i, shape_in_chunks)
            neighbors = []
            if cz + 1 < shape_in_chunks[0]:
                neighbors.append(np.ravel_multi_index((cz + 1, cy, cx), shape_in_chunks))
            if cy + 1 < shape_in_chunks[1]:
                neighbors.append(np.ravel_multi_index((cz, cy + 1, cx), shape_in_chunks))
            if cx + 1 < shape_in_chunks[2]:
                neighbors.append(np.ravel_multi_index((cz, cy, cx + 1), shape_in_chunks))

            data1 = chunk_data[i]
            res1 = np.load(data1['path'])
            seeds1_map = data1['seed_map']

            for j in neighbors:
                if j not in chunk_data:
                    continue
                data2 = chunk_data[j]
                chunk_slice2 = chunk_slices[j]
                res2 = np.load(data2['path'])
                seeds2_map = data2['seed_map']

                # Calculate overlap slices
                overlap_slice_global = tuple(
                    slice(max(s1.start, s2.start), min(s1.stop, s2.stop))
                    for s1, s2 in zip(chunk_slice1, chunk_slice2)
                )

                local_slice1 = tuple(
                    slice(s.start - cs.start, s.stop - cs.start)
                    for s, cs in zip(overlap_slice_global, chunk_slice1)
                )
                local_slice2 = tuple(
                    slice(s.start - cs.start, s.stop - cs.start)
                    for s, cs in zip(overlap_slice_global, chunk_slice2)
                )

                crop1 = res1[local_slice1]
                crop2 = res2[local_slice2]

                mask_overlap = (crop1 > 0) & (crop2 > 0)
                if not np.any(mask_overlap):
                    continue

                l1_flat = crop1[mask_overlap]
                l2_flat = crop2[mask_overlap]
                stacked = np.vstack((l1_flat, l2_flat))
                unique_pairs = np.unique(stacked, axis=1).T

                for id1, id2 in unique_pairs:
                    # --- CRITICAL STITCHING LOGIC ---
                    # Retrieve the original soma seeds that generated these segments.
                    s1_set = seeds1_map.get(id1, set())
                    s2_set = seeds2_map.get(id2, set())

                    # If the sets of original seeds are disjoint, it means these two segments
                    # came from DIFFERENT nuclei. We should NOT merge them, even if they touch.
                    if s1_set and s2_set and s1_set.isdisjoint(s2_set):
                        continue

                    # Otherwise, they share a seed (are the same cell split by chunking), merge them.
                    root1 = label_map.get(id1, id1)
                    root2 = label_map.get(id2, id2)

                    if root1 != root2:
                        target = min(root1, root2)
                        source = max(root1, root2)
                        
                        # Redirect any existing mapping pointing to source
                        for k, v in list(label_map.items()):
                            if v == source:
                                label_map[k] = target

                        label_map[source] = target
                        label_map[root1] = target
                        label_map[root2] = target
                        label_map[id1] = target
                        label_map[id2] = target

        # 4. Construct Final Mask
        final_path = os.path.join(memmap_dir, "stitched.mmp")
        final_mask = np.memmap(
            final_path, dtype=np.int32, mode='w+', shape=segmentation_mask.shape
        )

        flush_print("  Writing stitched result...")
        for i, sl in enumerate(tqdm(chunk_slices, desc="Writing Result")):
            if i not in chunk_data:
                continue
            path = chunk_data[i]['path']
            res = np.load(path)

            uniques = np.unique(res)
            for u in uniques:
                if u == 0:
                    continue
                if u in label_map:
                    target = label_map[u]
                    if target != u:
                        res[res == u] = target

            mask_nz = res > 0
            canvas_view = final_mask[sl]
            canvas_view[mask_nz] = res[mask_nz]
            final_mask[sl] = canvas_view

            os.remove(path)

        # Convert to array for final steps (usually fits in RAM if chunks worked)
        ret = np.array(final_mask)
        del final_mask
        if os.path.exists(final_path):
            os.remove(final_path)

        flush_print("  Refining (Filling voids + Relabeling)...")
        ret = fill_internal_voids(ret)
        ret, _, _ = relabel_sequential(ret)

        return ret
    
    finally:
        # Emergency cleanup: remove any remaining .npy files in case of crash
        for i in chunk_data:
            if os.path.exists(chunk_data[i]['path']):
                try: os.remove(chunk_data[i]['path'])
                except: pass
        gc.collect()