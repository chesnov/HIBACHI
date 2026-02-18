"""
Cell Splitting Module (2D)
==========================

This module provides 2D cell separation logic that is exactly logically 
equivalent to the 3D implementation. It uses a chunked processing 
architecture with seed-aware stitching to handle large-scale data 
(e.g., whole-slide scans) without RAM overload.

Features:
- Intensity-weighted watershed landscape.
- Region Adjacency Graph (RAG) for interface analysis.
- Seed-aware orphan reassignment.
- Overlapping chunked processing with global seed-based stitching.
"""

import os
import gc
import sys
from typing import List, Dict, Optional, Tuple, Set, Iterator, Any

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_dilation, footprint_rectangle, disk  # type: ignore
from skimage.segmentation import relabel_sequential, watershed  # type: ignore
from tqdm import tqdm

# Import shared helpers using the portable sys.path method for robustness
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

try:
    # Attempt to import from the module structure
    from ..module_3d.segmentation_helpers import flush_print, distance_transform_edt
except ImportError:
    # Fallback if imports are not structured as a package
    def flush_print(*args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)
        sys.stdout.flush()
    from scipy.ndimage import distance_transform_edt


def _get_chunk_slices_2d(
    image_shape: Tuple[int, int],
    chunk_shape: Tuple[int, int],
    overlap: int
) -> Iterator[Tuple[slice, ...]]:
    """
    Generator that yields slices for overlapping chunks in 2D.

    Args:
        image_shape: Shape of the full image (Y, X).
        chunk_shape: Desired shape of each chunk.
        overlap: Overlap size in pixels.

    Yields:
        Tuple of slices defining the chunk coordinates.
    """
    for y in range(0, image_shape[0], chunk_shape[0] - overlap):
        for x in range(0, image_shape[1], chunk_shape[1] - overlap):
            yield (
                slice(y, min(y + chunk_shape[0], image_shape[0])),
                slice(x, min(x + chunk_shape[1], image_shape[1])),
            )


# =============================================================================
# Graph & Metric Functions (3D Parity)
# =============================================================================

def _analyze_local_intensity_difference_2d_aligned(
    interface_mask: np.ndarray,
    region1_mask: np.ndarray,
    region2_mask: np.ndarray,
    intensity_local: np.ndarray,
    local_analysis_radius: int,
    min_local_intensity_difference_threshold: float
) -> bool:
    """
    Analyzes relative intensity difference between two regions at their interface.
    Matches 3D logic: sampling local neighborhood around the shared boundary.
    """
    footprint_elem = (
        disk(local_analysis_radius) if local_analysis_radius > 1
        else footprint_rectangle((3, 3))
    )

    # Define local analysis zone around the interface
    analysis_zone = binary_dilation(interface_mask, footprint=footprint_elem)

    # Extract pixels belonging to R1 and R2 within that zone
    la_r1 = analysis_zone & region1_mask
    la_r2 = analysis_zone & region2_mask

    # If regions are too small locally, assume they are distinct (safe default)
    if np.sum(la_r1) < 20 or np.sum(la_r2) < 20:
        return True

    m1 = np.mean(intensity_local[la_r1])
    m2 = np.mean(intensity_local[la_r2])

    ref_i = max(m1, m2)
    if ref_i < 1e-6:
        return True

    rel_diff = abs(m1 - m2) / ref_i
    return rel_diff >= min_local_intensity_difference_threshold


def _calculate_interface_metrics_2d_aligned(
    mask_A_local: np.ndarray,
    mask_B_local: np.ndarray,
    parent_mask_local: np.ndarray,
    intensity_local: np.ndarray,
    avg_soma_intensity_for_interface: float,
    spacing_tuple: Optional[Tuple[float, float]],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float,
) -> Dict[str, Any]:
    """Calculates metrics to decide if two watershed basins should be merged."""
    metrics = {'should_merge_decision': False}
    footprint_dilation = footprint_rectangle((3, 3))

    # Identify interface pixels
    dilated_A = binary_dilation(mask_A_local, footprint=footprint_dilation)
    interface_mask = dilated_A & mask_B_local & parent_mask_local

    if not np.any(interface_mask):
        return metrics

    # 1. Valley Depth Check (Path Intensity Ratio)
    mean_interface_intensity = np.mean(intensity_local[interface_mask])
    ratio = mean_interface_intensity / max(avg_soma_intensity_for_interface, 1e-6)
    ratio_threshold_passed = ratio < min_path_intensity_ratio_heuristic

    # 2. Local Contrast Check
    lid_passed = _analyze_local_intensity_difference_2d_aligned(
        interface_mask, mask_A_local, mask_B_local, intensity_local,
        local_analysis_radius, min_local_intensity_difference
    )

    if not ratio_threshold_passed or not lid_passed:
        metrics['should_merge_decision'] = True

    return metrics


def _build_adjacency_graph_for_cell_2d(
    current_cell_segments_mask_local: np.ndarray,
    original_cell_mask_local: np.ndarray,
    soma_mask_local: np.ndarray,
    soma_props_for_cell: Dict[int, Dict[str, Any]],
    intensity_local: np.ndarray,
    spacing_tuple: Optional[Tuple[float, float]],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float
) -> Tuple[Dict[int, Any], Dict[Tuple[int, int], Any]]:
    """Builds a Region Adjacency Graph (RAG) for 2D segments."""
    nodes = {}
    edges = {}

    seg_lbls = np.unique(
        current_cell_segments_mask_local[current_cell_segments_mask_local > 0]
    )
    if len(seg_lbls) <= 1:
        return nodes, edges

    footprint_d = footprint_rectangle((3, 3))

    for lbl in seg_lbls:
        mask = (current_cell_segments_mask_local == lbl)
        seeds_inside = np.unique(soma_mask_local[mask])
        nodes[lbl] = {
            'volume': np.sum(mask),
            'orig_somas': [s for s in seeds_inside if s > 0]
        }

    for i in range(len(seg_lbls)):
        lbl_A = seg_lbls[i]
        mask_A = (current_cell_segments_mask_local == lbl_A)
        dil_A = binary_dilation(mask_A, footprint=footprint_d)
        candidate_mask = (
            dil_A &
            (current_cell_segments_mask_local != lbl_A) &
            (current_cell_segments_mask_local > 0)
        )
        candidates = current_cell_segments_mask_local[candidate_mask]

        for lbl_B in np.unique(candidates):
            if lbl_B <= lbl_A: continue
            edge_key = (lbl_A, lbl_B)
            if edge_key in edges: continue

            mask_B = (current_cell_segments_mask_local == lbl_B)
            somas_A = nodes[lbl_A]['orig_somas']
            somas_B = nodes[lbl_B]['orig_somas']
            all_somas = somas_A + somas_B
            soma_ints = [
                soma_props_for_cell[s]['mean_intensity']
                for s in all_somas if s in soma_props_for_cell
            ]
            ref_intensity = np.mean(soma_ints) if soma_ints else 1.0

            edges[edge_key] = _calculate_interface_metrics_2d_aligned(
                mask_A, mask_B, original_cell_mask_local, intensity_local,
                ref_intensity, spacing_tuple, local_analysis_radius,
                min_local_intensity_difference, min_path_intensity_ratio_heuristic
            )

    return nodes, edges

def _reassign_disconnected_islands_2d(
    segmentation: np.ndarray, 
    soma_mask: np.ndarray
) -> np.ndarray:
    """
    Post-processing (2D): Detects disconnected satellite fragments.
    Keeps the fragment with the seed; reassigns orphans to neighbors.
    """
    flush_print("  [Refine] Checking for disconnected satellite fragments (2D)...")
    objs = ndimage.find_objects(segmentation)
    struct = ndimage.generate_binary_structure(2, 1)
    dilate_struct = ndimage.generate_binary_structure(2, 1)

    for idx, sl in enumerate(tqdm(objs, desc="Reassigning Islands")):
        if sl is None: continue
        label_id = idx + 1
        
        sl_pad = tuple(slice(max(0, s.start-1), min(d, s.stop+1)) 
                       for s, d in zip(sl, segmentation.shape))
        
        target_view = segmentation[sl_pad]
        local_soma = soma_mask[sl_pad]
        
        obj_mask = (target_view == label_id)
        labeled_frags, num_frags = ndimage.label(obj_mask, structure=struct)
        
        if num_frags <= 1: continue
            
        for i in range(1, num_frags + 1):
            frag_mask = (labeled_frags == i)
            # Check for seed strictly inside
            has_seed = np.any(local_soma[frag_mask] > 0)
            
            if has_seed: continue 
            
            # Reassign to largest touching neighbor
            dilated_frag = ndimage.binary_dilation(frag_mask, structure=dilate_struct)
            neighbor_ids = target_view[dilated_frag]
            neighbor_ids = neighbor_ids[(neighbor_ids != 0) & (neighbor_ids != label_id)]
            
            if neighbor_ids.size > 0:
                counts = np.bincount(neighbor_ids)
                target_view[frag_mask] = np.argmax(counts)
            else:
                target_view[frag_mask] = 0

    return segmentation


# =============================================================================
# Worker Function
# =============================================================================

def _separate_multi_soma_cells_chunk_2d(
    segmentation_mask: np.ndarray,
    intensity_volume: np.ndarray,
    soma_mask: np.ndarray,
    spacing: Optional[Tuple[float, float]],
    label_offset: int,
    multi_soma_cell_labels_list: List[int],
    **kwargs
) -> Tuple[np.ndarray, Dict, Dict[int, Set[int]]]:
    """Worker: Separates multi-soma cells within a specific 2D chunk."""
    chunk_result = np.zeros_like(segmentation_mask, dtype=np.int32)
    label_to_seeds_map = {}

    unique_labels = np.unique(segmentation_mask[segmentation_mask > 0])

    for lbl in unique_labels:
        if lbl not in multi_soma_cell_labels_list:
            chunk_result[segmentation_mask == lbl] = lbl
            seeds = np.unique(soma_mask[segmentation_mask == lbl])
            if seeds.size > 0:
                label_to_seeds_map[lbl] = set(seeds[seeds > 0])

    present_multi_soma = [l for l in multi_soma_cell_labels_list if l in unique_labels]
    next_local_label = label_offset
    min_size_thresh = kwargs.get('min_size_threshold', 0)

    for cell_label in present_multi_soma:
        cell_mask_full = (segmentation_mask == cell_label)
        slices = ndimage.find_objects(cell_mask_full)
        if not slices: continue

        bbox = slices[0]
        bbox_padded = tuple(slice(max(0, s.start - 2), min(dim, s.stop + 2))
                           for s, dim in zip(bbox, segmentation_mask.shape))

        local_mask = cell_mask_full[bbox_padded]
        local_soma = soma_mask[bbox_padded]
        local_intensity = intensity_volume[bbox_padded]

        seeds_in_crop = np.unique(local_soma[local_mask])
        seeds_in_crop = seeds_in_crop[seeds_in_crop > 0]

        if len(seeds_in_crop) < 2:
            new_label = next_local_label
            chunk_result[bbox_padded][local_mask] = new_label
            label_to_seeds_map[new_label] = set(seeds_in_crop)
            next_local_label += 1
            continue

        soma_props = {}
        for s_id in seeds_in_crop:
            s_mask = (local_soma == s_id)
            mean_i = np.mean(local_intensity[s_mask]) if np.any(s_mask) else 1.0
            soma_props[s_id] = {'mean_intensity': mean_i}

        # Seeded Watershed with Intensity Weight (3D Logic)
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

        ws_local = watershed(landscape, markers, mask=local_mask)

        # Graph-Based Merging
        nodes, edges = _build_adjacency_graph_for_cell_2d(
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
                for k, v in merge_map.items():
                    if v == max(root_a, root_b): merge_map[k] = target

        final_local_mask = np.zeros_like(ws_local)
        for old_id in np.unique(ws_local[ws_local > 0]):
            final_local_mask[ws_local == old_id] = merge_map[old_id]

        # Seed-Aware Orphan Reassignment
        unique_result_ids = np.unique(final_local_mask[final_local_mask > 0])
        dilation_struct = footprint_rectangle((3, 3))
        for uid in unique_result_ids:
            uid_mask = (final_local_mask == uid)
            cc_labels, num_cc = ndimage.label(uid_mask)
            if num_cc > 1:
                for i in range(1, num_cc + 1):
                    frag_mask = (cc_labels == i)
                    
                    has_seed = np.any(markers[frag_mask] > 0)
                    if not has_seed:
                        frag_dilated = binary_dilation(frag_mask, footprint=disk(2))
                        touching_seeds = np.unique(local_soma[frag_dilated])
                        has_seed = np.any(np.isin(touching_seeds, seeds_in_crop))

                    if not has_seed:
                        dilated = binary_dilation(frag_mask, footprint=dilation_struct)
                        neighbors = final_local_mask[dilated & (final_local_mask != 0) & (final_local_mask != uid)]
                        if neighbors.size > 0:
                            n_ids, n_counts = np.unique(neighbors, return_counts=True)
                            final_local_mask[frag_mask] = n_ids[np.argmax(n_counts)]
                        elif min_size_thresh > 0 and np.sum(frag_mask) < min_size_thresh:
                            final_local_mask[frag_mask] = 0

        final_local_mask_clean, _, _ = relabel_sequential(final_local_mask)
        chunk_view = chunk_result[bbox_padded]
        for local_id in np.unique(final_local_mask_clean[final_local_mask_clean > 0]):
            mask_l = (final_local_mask_clean == local_id)
            seeds = np.unique(local_soma[mask_l])
            global_lbl = next_local_label
            chunk_view[mask_l] = global_lbl
            label_to_seeds_map[global_lbl] = set(seeds[seeds > 0])
            next_local_label += 1
        chunk_result[bbox_padded] = chunk_view

    return chunk_result, {}, label_to_seeds_map


# =============================================================================
# Main Coordinator
# =============================================================================

def separate_multi_soma_cells_2d(
    segmentation_mask: np.ndarray,
    intensity_volume: np.ndarray,
    soma_mask: np.ndarray,
    spacing: Optional[Tuple[float, float]],
    chunk_shape: Tuple[int, int] = (1024, 1024),
    overlap: int = 64,
    **kwargs
) -> np.ndarray:
    """Main Coordinator for 2D cell separation (3D Aligned)."""
    flush_print("[SepMultiSoma_2D] Starting (Chunked + Seed-Aware)...")

    # 1. Identify Multi-Soma Cells
    unique_labels = np.unique(segmentation_mask[segmentation_mask > 0])
    cell_to_somas: Dict[int, Set[int]] = {}
    soma_locs = ndimage.find_objects(soma_mask)
    for s_idx, s_slice in enumerate(soma_locs):
        if s_slice is None: continue
        s_id = s_idx + 1
        cells_under = np.unique(segmentation_mask[s_slice][soma_mask[s_slice] == s_id])
        for cell_id in cells_under:
            if cell_id > 0: cell_to_somas.setdefault(cell_id, set()).add(s_id)

    multi_soma_labels = [c for c, s in cell_to_somas.items() if len(s) > 1]
    if not multi_soma_labels:
        return segmentation_mask.copy()

    # 2. Process Chunks
    memmap_dir = kwargs.get("memmap_dir", "ramiseg_temp_memmap")
    os.makedirs(memmap_dir, exist_ok=True)

    chunk_slices = list(_get_chunk_slices_2d(segmentation_mask.shape, chunk_shape, overlap))
    chunk_data = {}

    try:
        for i, sl in enumerate(tqdm(chunk_slices, desc="Processing 2D Chunks")):
            res, _, seed_map = _separate_multi_soma_cells_chunk_2d(
                segmentation_mask[sl], intensity_volume[sl], soma_mask[sl],
                spacing, (i + 1) * 1_000_000, multi_soma_labels, **kwargs
            )
            path = os.path.join(memmap_dir, f"chunk2d_{i}_{os.getpid()}.npy")
            np.save(path, res)
            chunk_data[i] = {'path': path, 'seed_map': seed_map}
            del res
            gc.collect()

        # 3. Seed-Aware Stitching
        flush_print("  Stitching with Transitive Seed Verification...")
        label_map: Dict[int, int] = {}
        
        # Build global lookup
        global_seed_lookup = {}
        for idx, data in chunk_data.items():
            global_seed_lookup.update(data['seed_map'])
            
        group_seeds_cache = {}
        def get_group_seeds(lbl_id):
            root = label_map.get(lbl_id, lbl_id)
            if root not in group_seeds_cache:
                return global_seed_lookup.get(root, set())
            return group_seeds_cache[root]

        grid_w = len(range(0, segmentation_mask.shape[1], chunk_shape[1] - overlap))

        for i, sl1 in enumerate(tqdm(chunk_slices, desc="Stitching Analysis")):
            cy, cx = divmod(i, grid_w)
            for dy, dx in [(0, 1), (1, 0)]:
                ny, nx = cy + dy, cx + dx
                j = ny * grid_w + nx
                if j >= len(chunk_slices): continue

                sl2 = chunk_slices[j]
                overlap_g = tuple(slice(max(s1.start, s2.start), min(s1.stop, s2.stop)) for s1, s2 in zip(sl1, sl2))
                if (overlap_g[0].stop <= overlap_g[0].start) or (overlap_g[1].stop <= overlap_g[1].start): continue

                res1, res2 = np.load(chunk_data[i]['path']), np.load(chunk_data[j]['path'])
                c1 = res1[tuple(slice(s.start - cs.start, s.stop - cs.start) for s, cs in zip(overlap_g, sl1))]
                c2 = res2[tuple(slice(s.start - cs.start, s.stop - cs.start) for s, cs in zip(overlap_g, sl2))]

                valid = (c1 > 0) & (c2 > 0)
                if not np.any(valid): continue
                pairs = np.unique(np.vstack((c1[valid], c2[valid])), axis=1).T
                
                for id1, id2 in pairs:
                    root1, root2 = label_map.get(id1, id1), label_map.get(id2, id2)
                    if root1 == root2: continue

                    # Transitive Check
                    s1_set = get_group_seeds(root1)
                    s2_set = get_group_seeds(root2)
                    if s1_set and s2_set and s1_set.isdisjoint(s2_set): continue

                    # Merge
                    target, source = min(root1, root2), max(root1, root2)
                    for k, v in list(label_map.items()):
                        if v == source: label_map[k] = target
                    label_map[source] = target
                    label_map[id1] = label_map[id2] = target
                    
                    # Update seeds
                    new_seeds = s1_set.union(s2_set)
                    group_seeds_cache[target] = new_seeds
                    if source in group_seeds_cache: del group_seeds_cache[source]

        # 4. Final Construction
        final_mask = np.zeros_like(segmentation_mask, dtype=np.int32)
        for i, sl in enumerate(tqdm(chunk_slices, desc="Writing Result")):
            res = np.load(chunk_data[i]['path'])
            uniques = np.unique(res)
            for u in uniques:
                if u > 0 and u in label_map:
                    target = label_map[u]
                    if target != u: res[res == u] = target
            mask_nz = res > 0
            final_mask[sl][mask_nz] = res[mask_nz]
            os.remove(chunk_data[i]['path'])

        # 5. Finalize
        # Fix satellite islands (chunk boundary artifacts)
        final_mask = _reassign_disconnected_islands_2d(final_mask, soma_mask)
        
        # Standard cleanup
        final_mask = ndimage.binary_fill_holes(final_mask > 0).astype(np.int32) * final_mask
        final_mask, _, _ = relabel_sequential(final_mask)
        
        return final_mask
    finally:
        # Emergency cleanup: remove any remaining .npy files in case of crash
        for i in chunk_data:
            if os.path.exists(chunk_data[i]['path']):
                try: os.remove(chunk_data[i]['path'])
                except: pass
        gc.collect()