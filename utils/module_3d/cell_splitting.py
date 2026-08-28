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
    cell_mean_intensity: float,
    spacing_tuple: Optional[Tuple[float, float, float]],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float,
    max_interface_to_cell_mean_ratio: float = 0.85,
) -> Dict[str, Any]:
    """
    Calculates metrics to decide if two watershed basins should be merged.
    Three independent criteria are evaluated — ALL must pass to keep basins separate:

    1. Valley-depth check (soma-relative ratio): interface / avg_soma < threshold.
       Catches interfaces that are bright relative to the soma peaks.

    2. Bright-cut check (cell-mean-relative ratio): interface / cell_mean < max_ratio.
       Catches Voronoi-style cuts between dim/adjacent seeds where the interface sits
       at or above the overall cell body mean — indicating no real intensity valley.
       This is the critical check for dim/tiny seeds close together in the same lobe.

    3. Local contrast check: the two basins must look sufficiently different near the
       interface. Catches spurious splits where both sides have similar intensities.
    """
    metrics = {'should_merge_decision': False}
    footprint_dilation = footprint_rectangle((3, 3, 3))

    # Identify interface pixels
    dilated_A = binary_dilation(mask_A_local, footprint=footprint_dilation)
    interface_mask = dilated_A & mask_B_local & parent_mask_local

    if not np.any(interface_mask):
        return metrics

    mean_interface_intensity = float(np.mean(intensity_local[interface_mask]))

    # 1. Valley Depth Check — soma-relative
    ratio_soma = mean_interface_intensity / max(avg_soma_intensity_for_interface, 1e-6)
    # LOW ratio = deep dark valley relative to soma peaks -> keep separate
    soma_ratio_passed = ratio_soma < min_path_intensity_ratio_heuristic

    # 2. Bright-Cut Check — cell-mean-relative
    # If the interface sits at or above cell_mean * threshold, there is no valley:
    # the watershed made a Voronoi-style geometric cut through bright tissue.
    # This fires when dim/tiny seeds are adjacent within the same cell lobe.
    ratio_cell_mean = mean_interface_intensity / max(cell_mean_intensity, 1e-6)
    cell_mean_ratio_passed = ratio_cell_mean < max_interface_to_cell_mean_ratio

    # 3. Local Contrast Check
    lid_passed = _analyze_local_intensity_difference_optimized(
        interface_mask, mask_A_local, mask_B_local, intensity_local,
        local_analysis_radius, min_local_intensity_difference
    )

    # Merge decision logic:
    #
    # - lid_passed=False alone is sufficient to merge (regions are locally indistinguishable).
    # - soma_ratio_passed=False alone is sufficient to merge (no deep valley vs soma peaks).
    # - cell_mean_ratio_passed=False alone is NOT sufficient to merge. It only acts as a
    #   tiebreaker when soma_ratio_passed=True but is borderline: if the interface is near
    #   the cell body mean AND the soma-ratio valley is not clearly deep, merge.
    #   Specifically: it can only upgrade a "keep" to a "merge" when soma_ratio is within
    #   a tolerance band of the threshold (ratio > threshold * 0.7). This prevents the
    #   bright-cut check from overriding a confirmed strong dark valley.
    soma_ratio_is_borderline = (
        soma_ratio_passed and
        ratio_soma > min_path_intensity_ratio_heuristic * 0.7
    )
    if not soma_ratio_passed or not lid_passed:
        metrics['should_merge_decision'] = True
    elif not cell_mean_ratio_passed and soma_ratio_is_borderline:
        metrics['should_merge_decision'] = True

    # [PROFILING] Log all three checks and their raw values.
    bright_cut_warn = " *** BRIGHT CUT ***" if not cell_mean_ratio_passed else ""
    flush_print(f"  [PROFILE|INTERFACE] "
                f"mean_interface={mean_interface_intensity:.1f} | "
                f"soma_ref={avg_soma_intensity_for_interface:.1f} | "
                f"soma_ratio={ratio_soma:.4f} (thr={min_path_intensity_ratio_heuristic}, "
                f"passed={soma_ratio_passed}) | "
                f"cell_mean_ratio={ratio_cell_mean:.4f} (thr={max_interface_to_cell_mean_ratio}, "
                f"passed={cell_mean_ratio_passed}){bright_cut_warn} | "
                f"lid_passed={lid_passed} | "
                f"=> should_merge={metrics['should_merge_decision']}")

    return metrics


def _build_adjacency_graph_for_cell(
    current_cell_segments_mask_local: np.ndarray,
    original_cell_mask_local: np.ndarray,
    soma_mask_local: np.ndarray,
    soma_props_for_cell: Dict[int, Dict[str, Any]],
    intensity_local: np.ndarray,
    cell_mean_intensity: float,
    spacing_tuple: Optional[Tuple[float, float, float]],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float,
    max_interface_to_cell_mean_ratio: float = 0.85,
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
                ref_intensity, cell_mean_intensity, spacing_tuple,
                local_analysis_radius, min_local_intensity_difference,
                min_path_intensity_ratio_heuristic, max_interface_to_cell_mean_ratio
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
    # NOTE: `min_size_threshold` is deliberately NOT read here. The worker sees one
    # chunk, so a basin straddling a chunk boundary has an arbitrarily truncated
    # size and would be judged undersized purely because it was clipped. The size
    # rule is applied once, globally, after stitching -- see
    # `_merge_undersized_cells` in `separate_multi_soma_cells`.

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

        # [PROFILING] Seed quality report — flags seeds that are dim or tiny relative to peers.
        # These seeds are NOT removed (seed placement is trusted), but flagged so we can
        # correlate suspicious seeds with bad cuts in the output.
        if len(seeds_in_crop) > 1:
            max_soma_int = max(p['mean_intensity'] for p in soma_props.values())
            marker_counts = {
                s_id: int(np.sum(local_soma == s_id)) for s_id in seeds_in_crop
            }
            max_marker_count = max(marker_counts.values())
            for s_id in seeds_in_crop:
                n_markers   = marker_counts[s_id]
                soma_int    = soma_props[s_id]['mean_intensity']
                int_ratio   = soma_int / (max_soma_int + 1e-6)
                size_ratio  = n_markers / (max_marker_count + 1e-6)
                flags = []
                if int_ratio < 0.6:
                    flags.append(f"DIM (int_ratio={int_ratio:.2f})")
                if size_ratio < 0.3:
                    flags.append(f"TINY (size_ratio={size_ratio:.2f})")
                flag_str = " *** " + ", ".join(flags) + " ***" if flags else ""
                flush_print(f"  [PROFILE|SEED] seed={s_id} | markers={n_markers} | "
                            f"soma_intensity={soma_int:.1f} | "
                            f"int_frac_of_brightest={int_ratio:.2f} | "
                            f"size_frac_of_largest={size_ratio:.2f}{flag_str}")

        # A. Seeded Watershed
        # Use SimpleITK helper for speed/memory efficiency
        
        # First, generate the markers (seeds)
        markers = np.zeros_like(local_mask, dtype=np.int32)
        for idx, s_id in enumerate(seeds_in_crop):
            markers[local_soma == s_id] = idx + 1

        # ---- LANDSCAPE GENERATION ----
        # NOTE: We do NOT use d_seeds = distance_transform_edt(markers == 0) in the
        # landscape formula. The old formula  landscape = d_seeds / speed^p  contained
        # a Euclidean geometric bias: d_seeds grows proportionally to straight-line
        # distance from the nearest seed, ignoring the speed field along the path.
        # For seeds of unequal size or position, this pre-biases basin boundaries toward
        # the geometric midplane regardless of intensity valleys.
        #
        # The correct formulation is a pure cost field:
        #   landscape = 1 / speed^p
        # The ITK watershed floods simultaneously from all seeds in ascending cost order.
        # Bright, thick regions (high speed) have low cost → seeds expand there first.
        # Dark necks (low speed) have high cost → they act as barriers and the boundary
        # naturally falls at the dark valley. No Euclidean pre-bias is introduced.
        #
        # d_seeds is still computed below for profiling only (it helps interpret the
        # geometry of the crop), but it is no longer part of the landscape formula.

        # 1. Geometric thickness (used both in speed and for profiling)
        # dt is large in cell centers and drops near boundaries and thin necks.
        dt = distance_transform_edt(local_mask, sampling=spacing)

        # 2. Euclidean distance from seeds — PROFILING ONLY, not used in landscape.
        d_seeds = distance_transform_edt(markers == 0, sampling=spacing)

        # 3. Calculate local expansion "speed"
        dt = distance_transform_edt(local_mask, sampling=spacing)
        speed = dt + 1e-5
        
        # 4. Modulate speed by intensity
        intensity_weight = kwargs.get('intensity_weight', 0.5)
        if intensity_weight > 0:
            # Fetch global data passed from the coordinator
            cell_global_seeds = list(kwargs.get('cell_to_somas', {}).get(cell_label, seeds_in_crop))
            global_intensities = kwargs.get('global_soma_intensities', {})
            
            # FIX: Global Continuous Normalization
            # Instead of using local.max() (which varies wildly between chunks and causes jagged artifacts),
            # we normalize against the true global soma intensities.
            soma_ints = [global_intensities.get(s, 1.0) for s in cell_global_seeds]
            ref_int = np.mean(soma_ints) if soma_ints else (local_intensity.max() + 1e-6)
            
            # This ignores bright noise outliers and restores the true 0.0 to 1.0 gradient!
            norm_int = np.clip(local_intensity / (ref_int + 1e-6), 0.0, 1.0)
            
            # Restored your original, highly stable math
            speed = speed * (1.0 + intensity_weight * norm_int)

        # 5. Final Landscape
        # speed_power = 1.5 slightly increases the penalty of thin necks to prevent Voronoi cuts.
        speed_power = kwargs.get('speed_power', 1.5)
        landscape = d_seeds / (speed ** speed_power)

        ws_local = _watershed_with_simpleitk(landscape, markers)
        ws_local[~local_mask] = 0

        # [PROFILING] Log what the watershed actually produced.
        ws_ids, ws_counts = np.unique(ws_local[ws_local > 0], return_counts=True)
        flush_print(f"  [PROFILE|WS] watershed output: labels={ws_ids} | voxel_counts={ws_counts}")
        cell_mean_int = float(np.mean(local_intensity[local_mask]))
        for ws_id, ws_cnt in zip(ws_ids, ws_counts):
            seed_id = seeds_in_crop[ws_id - 1] if (ws_id - 1) < len(seeds_in_crop) else '?'
            soma_int = soma_props.get(seed_id, {}).get('mean_intensity', float('nan'))
            basin_mask = ws_local == ws_id
            basin_mean = float(np.mean(local_intensity[basin_mask]))
            # Fraction of voxels in this basin that are brighter than the cell mean.
            # A correct intensity-guided cut should send bright voxels to the bright-soma basin.
            frac_bright = float(np.mean(local_intensity[basin_mask] > cell_mean_int))
            flush_print(f"    ws_label={ws_id} (->seed {seed_id}, soma_intensity={soma_int:.1f}): "
                        f"{ws_cnt} voxels | mean_intensity={basin_mean:.1f} | "
                        f"frac_above_cell_mean={frac_bright:.2f}")

        # [PROFILING] BOUNDARY INTENSITY CHECK — the most direct test of whether the cut
        # is at a dark valley. Dilate each basin, intersect with all OTHER basin voxels.
        # If boundary_mean ≈ cell_mean_int → CUT IS AT WRONG PLACE (bright midpoint, Voronoi).
        # If boundary_mean << cell_mean_int → CUT IS CORRECT (dark intensity valley).
        flush_print(f"  [PROFILE|WS|BOUNDARY] cell_mean_intensity={cell_mean_int:.1f}")
        footprint_b = footprint_rectangle((3, 3, 3))
        for ws_id in ws_ids:
            basin_mask = ws_local == ws_id
            dilated = binary_dilation(basin_mask, footprint=footprint_b)
            boundary_voxels = dilated & (ws_local > 0) & (~basin_mask)
            if np.any(boundary_voxels):
                bnd_mean = float(np.mean(local_intensity[boundary_voxels]))
                bnd_frac_below_mean = float(np.mean(local_intensity[boundary_voxels] < cell_mean_int))
                verdict = "CORRECT (dark valley)" if bnd_mean < 0.7 * cell_mean_int else "WRONG (bright cut — Voronoi bias!)"
                flush_print(f"    boundary of ws_label={ws_id}: mean_intensity={bnd_mean:.1f} | "
                            f"frac_below_cell_mean={bnd_frac_below_mean:.2f} | => {verdict}")


        # B. Graph-Based Merging
        nodes, edges = _build_adjacency_graph_for_cell(
            ws_local, local_mask, local_soma, soma_props, local_intensity,
            cell_mean_int, spacing,
            kwargs.get('local_analysis_radius', 10),
            kwargs.get('min_local_intensity_difference', 0.0),
            kwargs.get('min_path_intensity_ratio', 1.0),
            kwargs.get('max_interface_to_cell_mean_ratio', 0.85),
        )

        # [PROFILING] Summarize what the graph decided before any merging happens.
        merge_edges = [(k, v) for k, v in edges.items() if v['should_merge_decision']]
        keep_edges  = [(k, v) for k, v in edges.items() if not v['should_merge_decision']]
        flush_print(f"  [PROFILE|GRAPH] cell={cell_label}: "
                    f"total_edges={len(edges)} | merge_edges={len(merge_edges)} | keep_edges={len(keep_edges)}")
        for edge_key, edge_val in merge_edges:
            flush_print(f"    [PROFILE|GRAPH] MERGE: {edge_key}")
        for edge_key, edge_val in keep_edges:
            flush_print(f"    [PROFILE|GRAPH] KEEP:  {edge_key}")

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

        # [PROFILING] Show the effective merge_map so we can detect runaway merging.
        flush_print(f"  [PROFILE|GRAPH] merge_map (ws_id -> final_id): {dict(list(merge_map.items())[:20])}")
        final_ids, final_counts = np.unique(final_local_mask[final_local_mask > 0], return_counts=True)
        flush_print(f"  [PROFILE|GRAPH] post-merge labels={final_ids} | voxel_counts={final_counts}")

        # C. Seed-Aware Orphan Reassignment
        # Ensure every fragment actually contains a seed. If not, merge it.
        unique_result_ids = np.unique(final_local_mask[final_local_mask > 0])
        dilation_struct = footprint_rectangle((3, 3, 3))
        cc_struct = ndimage.generate_binary_structure(3, 3)  # 26-conn incl. diagonals

        for uid in unique_result_ids:
            cell_mask = (final_local_mask == uid)
            cc_labels, num_cc = ndimage.label(cell_mask, structure=cc_struct)

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
                        # No differently-labelled neighbour: this is an isolated
                        # satellite of its OWN cell (already labelled `uid`, whose
                        # seed sits on another component). It is genuine foreground
                        # and is KEPT unconditionally.
                        #
                        # This branch used to delete the fragment when it fell below
                        # `min_size_threshold`, which was the only place step 4 ever
                        # destroyed signal -- and it destroyed exactly the case that
                        # must be preserved. Size-based handling now happens globally,
                        # post-stitch, in `_merge_undersized_cells`, where it MERGES
                        # rather than deletes and where fragment sizes are true (a
                        # chunk-clipped fragment looks arbitrarily small here).
                        flush_print(
                            f"  [PROFILE|ORPHAN] worker KEEP orphan: cell={cell_label} "
                            f"uid={uid} orphan_size={int(np.sum(frag_mask))} "
                            f"(isolated satellite; never deleted)"
                        )

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



def _reassign_disconnected_islands(
    segmentation: np.ndarray,
    soma_mask: np.ndarray,
) -> np.ndarray:
    """
    Post-processing: detect labels split into disconnected fragments by chunk
    boundaries. Keep the seed-bearing fragment; reassign a seedless fragment to
    its largest DIFFERENTLY-labelled touching neighbour when one exists.

    A seedless fragment with no differently-labelled neighbour is an isolated
    satellite of its OWN cell (it is already labelled ``label_id`` and that cell
    carries its seed on another fragment). It is left untouched -- it is genuine
    foreground and must not be removed here. This refinement pass is strictly
    foreground-conserving.

    Size-based handling is NOT done here and is no longer done in the worker
    either: it runs immediately after this pass, in ``_merge_undersized_cells``,
    which MERGES an undersized cell into a neighbour instead of deleting it.
    Outright removal of small objects belongs to the step 1 / step 2 size filters.

    Instrumented so foreground conservation is visible in the run log.
    """
    flush_print("  [Refine] Checking for disconnected satellite fragments...")

    objs = ndimage.find_objects(segmentation)
    struct = ndimage.generate_binary_structure(3, 3)
    dilate_struct = ndimage.generate_binary_structure(3, 3)

    # [PROFILE|ISLAND] Foreground accounting (must net to zero after the fix).
    fg_before = int(np.count_nonzero(segmentation))
    n_multi = n_orphans = n_reassigned = n_kept = 0
    px_reassigned = px_kept = 0
    detail_cap = 80
    detail_shown = 0

    for idx, sl in enumerate(tqdm(objs, desc="Reassigning Islands")):
        if sl is None:
            continue
        label_id = idx + 1

        sl_pad = tuple(
            slice(max(0, s.start - 1), min(d, s.stop + 1))
            for s, d in zip(sl, segmentation.shape)
        )

        target_view = segmentation[sl_pad]
        local_soma = soma_mask[sl_pad]

        obj_mask = (target_view == label_id)
        labeled_frags, num_frags = ndimage.label(obj_mask, structure=struct)

        if num_frags <= 1:
            continue

        n_multi += 1

        for i in range(1, num_frags + 1):
            frag_mask = (labeled_frags == i)
            has_seed = np.any(local_soma[frag_mask] > 0)

            if has_seed:
                continue  # main body, keep it.

            n_orphans += 1
            frag_size = int(np.sum(frag_mask))

            dilated_frag = ndimage.binary_dilation(frag_mask, structure=dilate_struct)

            # Neighbours of a DIFFERENT label. NOTE: the dilation includes the
            # fragment itself, so raw values always contain label_id; excluding
            # label_id here is what leaves only genuine other-label neighbours.
            raw_neighbors = target_view[dilated_frag]
            neighbor_ids = raw_neighbors[(raw_neighbors != 0) & (raw_neighbors != label_id)]

            if neighbor_ids.size > 0:
                # Seedless fragment abutting another cell -> hand it over.
                counts = np.bincount(neighbor_ids)
                best_neighbor = int(np.argmax(counts))
                target_view[frag_mask] = best_neighbor
                n_reassigned += 1
                px_reassigned += frag_size
                if detail_shown < detail_cap:
                    flush_print(
                        f"    [PROFILE|ISLAND] label={label_id} frag={i} "
                        f"size={frag_size} seedless -> REASSIGN to {best_neighbor}"
                    )
                    detail_shown += 1
            else:
                # Isolated satellite of its own cell -> KEEP (do not delete).
                # This is the fix: previously this branch zeroed the fragment,
                # silently dropping real foreground from a seeded cell.
                n_kept += 1
                px_kept += frag_size
                if detail_shown < detail_cap:
                    flush_print(
                        f"    [PROFILE|ISLAND] label={label_id} frag={i} "
                        f"size={frag_size} isolated seedless satellite -> KEEP "
                        f"(retained as label {label_id}; not deleted)"
                    )
                    detail_shown += 1

    fg_after = int(np.count_nonzero(segmentation))

    flush_print(
        f"  [PROFILE|ISLAND|SUMMARY] multi_frag_labels={n_multi} | orphans={n_orphans} "
        f"(reassigned={n_reassigned}, kept_isolated={n_kept}) | "
        f"voxels_reassigned={px_reassigned} | voxels_kept={px_kept}"
    )
    flush_print(
        f"  [PROFILE|ISLAND|SUMMARY] foreground voxels before={fg_before} "
        f"after={fg_after} delta={fg_after - fg_before}"
        + ("  *** FOREGROUND LOST -- UNEXPECTED ***" if fg_after < fg_before
           else "  (conserved)")
    )

    return segmentation


def _merge_undersized_cells(
    segmentation: np.ndarray,
    min_size_threshold: int,
    max_rounds: int = 20,
) -> np.ndarray:
    """
    Merge any final cell smaller than ``min_size_threshold`` into its most-contacted
    neighbouring cell. Never deletes anything.

    Why this exists
    ---------------
    The watershed in the worker splits a multi-soma cell into one basin per seed.
    A basin can come out genuinely too small to be a cell -- but it is real signal
    belonging to the cell it was cut from, so the right response is to give it back
    to a sibling basin, not to delete it. Before this pass, ``min_size_threshold``
    only ever applied to *seedless* fragments, and only to delete them; a small
    basin that happened to contain a seed was never size-tested at all.

    Why here and not in the worker
    ------------------------------
    The worker sees a single chunk, so a basin straddling a chunk boundary has an
    arbitrarily truncated size there and would be merged purely because it was
    clipped. Run after stitching, every label's size is its true, whole size.

    Guarantees
    ----------
    * **Nothing is ever deleted.** A label below the threshold with no
      differently-labelled neighbour is KEPT at full size. That is what preserves
      a genuinely small cell -- whether or not soma extraction gave it a seed, and
      including the degenerate case where every basin of one cell is undersized
      (merging then cascades until a single label remains, which has no sibling
      left and is kept whole). Foreground is exactly conserved.
    * **Smallest-first, and iterated.** Absorbing a fragment grows the recipient,
      which can lift it over the threshold, so the set is re-evaluated between
      rounds rather than judged once against stale sizes.
    * This rule OVERRIDES the graph merge decisions: a boundary the intensity
      heuristics deliberately kept is still merged if the basin is undersized.
      That is intentional -- the size floor is the final word on what counts as a
      cell -- and every such merge is logged.

    Only labels that touch another label can be merged, and basins only ever touch
    siblings cut from the same original object, so a cell that was never split
    cannot be affected by this pass.
    """
    if min_size_threshold is None or min_size_threshold <= 0:
        flush_print("  [Refine] Undersized-cell merge skipped (threshold <= 0).")
        return segmentation

    flush_print(
        f"  [Refine] Merging cells below {min_size_threshold} voxels into their "
        "largest-contact neighbour (nothing is deleted)..."
    )

    fg_before = int(np.count_nonzero(segmentation))
    dilate_struct = ndimage.generate_binary_structure(3, 3)

    n_merged = n_kept = 0
    px_merged = px_kept = 0
    detail_cap = 80
    detail_shown = 0
    kept_isolated: Set[int] = set()   # below threshold, no neighbour -> stop retrying

    round_idx = 0                     # defined even if max_rounds <= 0
    for round_idx in range(max_rounds):
        counts = np.bincount(segmentation.ravel())
        if counts.size <= 1:
            break
        # Candidate labels: present, below threshold, not already settled.
        candidates = [
            int(lbl) for lbl in np.nonzero(counts)[0]
            if lbl != 0
            and counts[lbl] < min_size_threshold
            and int(lbl) not in kept_isolated
        ]
        if not candidates:
            break

        # Smallest first: the least cell-like fragment is given away before it can
        # act as a recipient for another.
        candidates.sort(key=lambda l: counts[l])

        objs = ndimage.find_objects(segmentation)
        merged_this_round = 0

        for label_id in candidates:
            if label_id - 1 >= len(objs):
                continue
            sl = objs[label_id - 1]
            if sl is None:
                continue  # already absorbed earlier in this round

            sl_pad = tuple(
                slice(max(0, s.start - 1), min(d, s.stop + 1))
                for s, d in zip(sl, segmentation.shape)
            )
            target_view = segmentation[sl_pad]
            frag_mask = (target_view == label_id)
            frag_size = int(np.sum(frag_mask))
            if frag_size == 0:
                continue
            # Re-check against the live size: an earlier merge in this same round
            # may have grown this label past the threshold.
            if frag_size >= min_size_threshold:
                continue

            dilated = ndimage.binary_dilation(frag_mask, structure=dilate_struct)
            raw_neighbors = target_view[dilated]
            neighbor_ids = raw_neighbors[
                (raw_neighbors != 0) & (raw_neighbors != label_id)
            ]

            if neighbor_ids.size > 0:
                counts_n = np.bincount(neighbor_ids)
                best_neighbor = int(np.argmax(counts_n))
                target_view[frag_mask] = best_neighbor
                segmentation[sl_pad] = target_view
                n_merged += 1
                px_merged += frag_size
                merged_this_round += 1
                if detail_shown < detail_cap:
                    flush_print(
                        f"    [PROFILE|UNDERSIZE] label={label_id} size={frag_size} "
                        f"< {min_size_threshold} -> MERGE into {best_neighbor} "
                        f"(contact={int(counts_n[best_neighbor])} voxels)"
                    )
                    detail_shown += 1
            else:
                # Isolated and undersized: a small cell in its own right. Keep it.
                kept_isolated.add(label_id)
                n_kept += 1
                px_kept += frag_size
                if detail_shown < detail_cap:
                    flush_print(
                        f"    [PROFILE|UNDERSIZE] label={label_id} size={frag_size} "
                        f"< {min_size_threshold} -> KEEP (no neighbouring cell to "
                        f"merge into; small cells are never deleted)"
                    )
                    detail_shown += 1

        if merged_this_round == 0:
            break
    else:
        flush_print(
            f"  [PROFILE|UNDERSIZE] *** reached the {max_rounds}-round cap; some "
            "undersized cells may remain. This is safe (nothing is deleted) but "
            "suggests min_size_threshold is very large relative to your cells. ***"
        )

    fg_after = int(np.count_nonzero(segmentation))
    flush_print(
        f"  [PROFILE|UNDERSIZE|SUMMARY] merged={n_merged} (voxels={px_merged}) | "
        f"kept_small_isolated={n_kept} (voxels={px_kept}) | rounds_used"
        f"={round_idx + 1}"
    )
    flush_print(
        f"  [PROFILE|UNDERSIZE|SUMMARY] foreground voxels before={fg_before} "
        f"after={fg_after} delta={fg_after - fg_before}"
        + ("  *** FOREGROUND LOST -- UNEXPECTED ***" if fg_after != fg_before
           else "  (conserved)")
    )

    return segmentation


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

    # [PROFILE|CONSERVE] Input inventory for end-to-end foreground accounting.
    _fg_in = int(np.count_nonzero(segmentation_mask))
    _nobj_in = int(np.unique(segmentation_mask[segmentation_mask > 0]).size)
    flush_print(
        f"  [PROFILE|CONSERVE] INPUT: foreground_voxels={_fg_in} | objects={_nobj_in}"
    )

    # 1. Identify Multi-Soma Cells (Global Check)
    unique_labels = np.unique(segmentation_mask[segmentation_mask > 0])
    cell_to_somas: Dict[int, Set[int]] = {}
    
    global_soma_centroids = {}
    global_soma_intensities = {}

    soma_locs = ndimage.find_objects(soma_mask)
    for s_idx, s_slice in enumerate(soma_locs):
        if s_slice is None:
            continue
        soma_id = s_idx + 1
        
        # Calculate Global Centroid
        cz = (s_slice[0].start + s_slice[0].stop) / 2.0
        cy = (s_slice[1].start + s_slice[1].stop) / 2.0
        cx = (s_slice[2].start + s_slice[2].stop) / 2.0
        if spacing:
            cz *= spacing[0]; cy *= spacing[1]; cx *= spacing[2]
        global_soma_centroids[soma_id] = np.array([cz, cy, cx])
        
        # Calculate Global Intensity Reference
        s_mask = soma_mask[s_slice] == soma_id
        mean_i = np.mean(intensity_volume[s_slice][s_mask]) if np.any(s_mask) else 1.0
        global_soma_intensities[soma_id] = mean_i

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

    kwargs['cell_to_somas'] = cell_to_somas
    kwargs['global_soma_centroids'] = global_soma_centroids
    kwargs['global_soma_intensities'] = global_soma_intensities

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
                spacing, chunk_offset, multi_soma_labels, 
                global_offset=(sl[0].start, sl[1].start, sl[2].start), # <--- NEW
                **kwargs
            )

            path = os.path.join(memmap_dir, f"chunk_{i}_{os.getpid()}.npy")
            np.save(path, res)

            chunk_data[i] = {'path': path, 'shape': res.shape, 'seed_map': seed_map}
            del res # Free RAM immediately
            gc.collect()

        # 3. Seed-Aware Stitching Logic
        flush_print("  Stitching with Transitive Seed Verification...")
        label_map: Dict[int, int] = {}
        
        # Global registry of seeds for every label ID
        # We must aggregate this first to track seeds as they merge
        global_seed_lookup = {}
        for idx, data in chunk_data.items():
            global_seed_lookup.update(data['seed_map'])

        # Dynamic tracker: Maps 'Root Label' -> 'Set of Seeds in this merged group'
        # We use a helper to resolve current seeds for any label
        def get_group_seeds(lbl_id):
            root = label_map.get(lbl_id, lbl_id)
            # If we haven't tracked this root explicitly yet, fallback to initial lookup
            if root not in group_seeds_cache:
                return global_seed_lookup.get(root, set())
            return group_seeds_cache[root]

        group_seeds_cache = {}

        shape_in_chunks = [
            len(range(0, segmentation_mask.shape[0], chunk_shape[0] - overlap)),
            len(range(0, segmentation_mask.shape[1], chunk_shape[1] - overlap)),
            len(range(0, segmentation_mask.shape[2], chunk_shape[2] - overlap))
        ]

        for i, chunk_slice1 in enumerate(tqdm(chunk_slices, desc="Stitching Analysis")):
            if i not in chunk_data:
                continue

            # Determine neighbors (same as before)
            cz, cy, cx = np.unravel_index(i, shape_in_chunks)
            neighbors = []
            if cz + 1 < shape_in_chunks[0]:
                neighbors.append(np.ravel_multi_index((cz + 1, cy, cx), shape_in_chunks))
            if cy + 1 < shape_in_chunks[1]:
                neighbors.append(np.ravel_multi_index((cz, cy + 1, cx), shape_in_chunks))
            if cx + 1 < shape_in_chunks[2]:
                neighbors.append(np.ravel_multi_index((cz, cy, cx + 1), shape_in_chunks))

            res1 = np.load(chunk_data[i]['path'])

            for j in neighbors:
                if j not in chunk_data:
                    continue
                
                chunk_slice2 = chunk_slices[j]
                res2 = np.load(chunk_data[j]['path'])

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

                stacked = np.vstack((crop1[mask_overlap], crop2[mask_overlap]))
                unique_pairs = np.unique(stacked, axis=1).T

                for id1, id2 in unique_pairs:
                    root1 = label_map.get(id1, id1)
                    root2 = label_map.get(id2, id2)

                    if root1 == root2:
                        continue

                    # We check the seeds of the *Roots*, not the fragments.
                    # This catches the case where an "Empty" fragment has already 
                    # been merged into "Cell A", acquiring "Seed A".
                    s1_set = get_group_seeds(root1)
                    s2_set = get_group_seeds(root2)

                    # If both groups have known seeds, and they don't overlap -> CONFLICT.
                    if s1_set and s2_set and s1_set.isdisjoint(s2_set):
                        continue # Do not merge Cell A and Cell B

                    # Otherwise, merge is safe (or involves an untagged bridge)
                    target = min(root1, root2)
                    source = max(root1, root2)
                    
                    # Update Map
                    label_map[source] = target
                    label_map[root1] = target
                    label_map[root2] = target
                    
                    # Update Cache: Union of seeds
                    new_seeds = s1_set.union(s2_set)
                    # If we just merged a phantom fragment into a cell, 
                    # the phantom fragment effectively 'gains' that cell's seed.
                    group_seeds_cache[target] = new_seeds
                    
                    # Redirect source's cache to target (cleanup)
                    if source in group_seeds_cache:
                        del group_seeds_cache[source]
                    
                    # Path compression for existing mappings
                    # (Optional optimization, but good for deep chains)
                    keys_to_update = [k for k, v in label_map.items() if v == source]
                    for k in keys_to_update:
                        label_map[k] = target

        # 4. Construct Final Mask
        final_path = os.path.join(memmap_dir, "stitched.mmp")
        final_mask = np.memmap(
            final_path, dtype=np.int32, mode='w+', shape=segmentation_mask.shape
        )

        flush_print("  Writing stitched result with Geodesic Overlap Resolution...")
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
            
            # Identify conflict pixels where canvas already has a DIFFERENT label
            conflict_mask = mask_nz & (canvas_view > 0) & (canvas_view != res)
            
            # Safely write non-conflicting pixels
            non_conflict = mask_nz & ~conflict_mask
            canvas_view[non_conflict] = res[non_conflict]
            
            # [PROFILING] Log conflict stats for every chunk written.
            n_conflict = int(np.sum(conflict_mask))
            n_written  = int(np.sum(non_conflict))
            flush_print(f"  [PROFILE|STITCH] chunk={i} | written={n_written} | "
                        f"conflicts={n_conflict} | "
                        f"conflict_labels_existing={np.unique(canvas_view[conflict_mask]).tolist()} | "
                        f"conflict_labels_incoming={np.unique(res[conflict_mask]).tolist()}")
            
            if np.any(conflict_mask):
                unique_pairs = np.unique(
                    np.vstack((canvas_view[conflict_mask], res[conflict_mask])), axis=1
                ).T
                
                for e_lab, i_lab in unique_pairs:
                    pair_conflict = (canvas_view == e_lab) & (res == i_lab) & conflict_mask
                    if not np.any(pair_conflict):
                        continue
                        
                    # 1. Get GLOBAL coordinates of the conflict pixels
                    cz, cy, cx = np.where(pair_conflict)
                    gz, gy, gx = cz + sl[0].start, cy + sl[1].start, cx + sl[2].start
                    
                    # 2. Extract a padded block from the GLOBAL canvas so we can see both chunks
                    pad = overlap + 4
                    z_min, z_max = max(0, gz.min() - pad), min(final_mask.shape[0], gz.max() + pad + 1)
                    y_min, y_max = max(0, gy.min() - pad), min(final_mask.shape[1], gy.max() + pad + 1)
                    x_min, x_max = max(0, gx.min() - pad), min(final_mask.shape[2], gx.max() + pad + 1)
                    
                    sub_slice = (slice(z_min, z_max), slice(y_min, y_max), slice(x_min, x_max))
                    cv_sub = final_mask[sub_slice].copy()
                    
                    # 3. Project conflict pixels into local cv_sub coordinates
                    local_cz, local_cy, local_cx = gz - z_min, gy - y_min, gx - x_min
                    local_conflict = np.zeros(cv_sub.shape, dtype=bool)
                    local_conflict[local_cz, local_cy, local_cx] = True
                    
                    # 4. Find safe zones globally
                    local_e_safe = (cv_sub == e_lab) & ~local_conflict
                    local_i_safe = (cv_sub == i_lab) & ~local_conflict
                    
                    # Failsafe
                    if not np.any(local_e_safe) or not np.any(local_i_safe):
                        flush_print(f"  [STITCHER] Failsafe Triggered. e:{np.sum(local_e_safe)}, i:{np.sum(local_i_safe)}")
                        cv_sub[local_conflict] = i_lab
                        final_mask[sub_slice] = cv_sub
                        continue
                        
                    # 5. Run Geodesic Micro-Watershed
                    local_domain = local_conflict | local_e_safe | local_i_safe
                    markers = np.zeros(local_domain.shape, dtype=np.int32)
                    markers[local_e_safe] = 1
                    markers[local_i_safe] = 2
                    
                    dt = distance_transform_edt(local_domain, sampling=spacing)

                    # NOTE: d_seeds (Euclidean distance from markers) is intentionally
                    # NOT computed here. See the per-cell watershed comment for rationale:
                    # using d_seeds in the landscape introduces a Euclidean geometric bias
                    # that overrides intensity guidance when safe zones are unequal in size.
                    # The pure cost field 1/speed^p is the correct formulation.

                    # FIX: Use intensity-modulated speed, identical to the per-cell watershed.
                    # Previously this was geometry-only (speed = dt + 1e-5), which caused
                    # large conflict zones to be cut along the geometric midplane instead of
                    # the true dark intensity valley — producing the diagonal mask artifact.
                    stitch_intensity_weight = kwargs.get('intensity_weight', 0.5)
                    if stitch_intensity_weight > 0 and np.any(local_domain):
                        local_int_sub = intensity_volume[sub_slice].astype(float)
                        smoothed_sub = ndimage.gaussian_filter(local_int_sub, sigma=1.0)
                        p1_s, p99_s = np.percentile(smoothed_sub[local_domain], [1, 99])
                        norm_int_sub = np.clip(
                            (smoothed_sub - p1_s) / (p99_s - p1_s + 1e-6), 0.0, 1.0
                        )
                        max_dt_sub = float(np.max(dt)) if np.any(local_domain) else 1.0
                        speed = dt + (norm_int_sub * max_dt_sub * stitch_intensity_weight) + 1e-5

                        # [PROFILING] Confirm intensity contribution relative to geometry.
                        # Only alarm when intensity is genuinely absent (ratio < 0.1).
                        geom_c = dt[local_domain].mean()
                        int_c  = (norm_int_sub[local_domain] * max_dt_sub * stitch_intensity_weight).mean()
                        ratio_c = int_c / (geom_c + 1e-9)
                        contrib_warn = " *** WARN: intensity nearly absent from speed! ***" if ratio_c < 0.1 else ""
                        flush_print(f"    [PROFILE|STITCH|GEO] speed: geometry={geom_c:.3f} "
                                    f"intensity_term={int_c:.3f} ratio={ratio_c:.2f} | "
                                    f"p1={p1_s:.1f} p99={p99_s:.1f}{contrib_warn}")

                        # [PROFILING] SAFE ZONE INTENSITY CHECK — the root cause test.
                        # If e_safe and i_safe have SIMILAR mean intensity, the geodesic has
                        # no signal to guide the cut and will default to a geometric midplane.
                        # If the conflict zone is NOT the darkest region, the safe zones are
                        # on the wrong side of the true cell boundary (upstream watershed error).
                        e_safe_mean = float(np.mean(local_int_sub[local_e_safe])) if np.any(local_e_safe) else float('nan')
                        i_safe_mean = float(np.mean(local_int_sub[local_i_safe])) if np.any(local_i_safe) else float('nan')
                        conflict_mean = float(np.percentile(local_int_sub[local_conflict], 5))
                        domain_mean = float(np.mean(local_int_sub[local_domain]))
                        is_valley = conflict_mean < min(e_safe_mean, i_safe_mean) * 0.85
                        safe_contrast = abs(e_safe_mean - i_safe_mean) / (max(e_safe_mean, i_safe_mean) + 1e-6)
                        # Warn on safe-zone contrast only when it is genuinely negligible (< 5%).
                        # 10-30% contrast IS meaningful signal; do not alarm on it.
                        contrast_warn = " *** LOW CONTRAST: geodesic cut may be geometric ***" if safe_contrast < 0.05 else ""
                        flush_print(f"    [PROFILE|STITCH|GEO] INTENSITY MAP: "
                                    f"e_safe_mean={e_safe_mean:.1f} | conflict_mean={conflict_mean:.1f} | "
                                    f"i_safe_mean={i_safe_mean:.1f} | domain_mean={domain_mean:.1f}")
                        flush_print(f"    [PROFILE|STITCH|GEO] conflict_is_valley={is_valley} "
                                    f"(conflict={conflict_mean:.0f} vs safe_min={min(e_safe_mean,i_safe_mean):.0f}) | "
                                    f"safe_zone_contrast={safe_contrast:.3f}{contrast_warn}")
                        if not is_valley:
                            flush_print(f"    [PROFILE|STITCH|GEO] *** SAFE ZONE PROBLEM: conflict zone is NOT "
                                        f"the darkest region. Safe zones are likely on the wrong side of the "
                                        f"true cell boundary — upstream watershed error propagated here. ***")
                    else:
                        local_int_sub = None
                        speed = dt + 1e-5
                        flush_print(f"    [PROFILE|STITCH|GEO] pair=({e_lab},{i_lab}) | "
                                    f"intensity_weight=0 — falling back to geometry-only speed.")

                    speed_power = kwargs.get('speed_power', 1.5)

                    # When there is no dark valley between the two safe zones, running
                    # the geodesic watershed is pointless: any cut it produces will be at
                    # a bright point (WRONG), and the result depends on arbitrary safe-zone
                    # size asymmetry. Instead, directly assign all conflict voxels to
                    # whichever label has the larger safe zone (size-wins rule).
                    # This avoids the expensive watershed and avoids introducing a spurious
                    # bright-intensity cut boundary into the segmentation.
                    if local_int_sub is not None and not is_valley:
                        winner = e_lab if np.sum(local_e_safe) >= np.sum(local_i_safe) else i_lab
                        flush_print(f"    [PROFILE|STITCH|GEO] NO VALLEY: skipping geodesic — "
                                    f"assigning all {int(np.sum(local_conflict))} conflict voxels "
                                    f"to larger safe zone (winner={winner}, "
                                    f"e_safe={int(np.sum(local_e_safe))} vs "
                                    f"i_safe={int(np.sum(local_i_safe))})")
                        cv_sub[local_conflict] = winner
                        final_mask[sub_slice] = cv_sub
                        continue  # skip geodesic for this pair

                    # d_seeds provides equal-start guarantee: all markers enter the priority
                    # queue at elevation 0 regardless of local intensity, so dim/tiny seeds
                    # compete fairly. See per-cell watershed comment for full rationale.
                    d_seeds_stitch = distance_transform_edt(markers == 0, sampling=spacing)
                    landscape = d_seeds_stitch / (speed ** speed_power)
                    landscape[~local_domain] = 1e6
                    
                    ws_local = _watershed_with_simpleitk(landscape, markers)

                    # [PROFILING] Log how many conflict voxels were assigned to each label.
                    n_to_e = int(np.sum((ws_local == 1) & local_conflict))
                    n_to_i = int(np.sum((ws_local == 2) & local_conflict))
                    flush_print(f"    [PROFILE|STITCH|GEO] pair=({e_lab},{i_lab}) | "
                                f"conflict_voxels={int(np.sum(local_conflict))} | "
                                f"geodesic => e_lab={n_to_e} voxels, i_lab={n_to_i} voxels | "
                                f"e_safe_size={int(np.sum(local_e_safe))} i_safe_size={int(np.sum(local_i_safe))}")

                    # [PROFILING] RESOLVED BOUNDARY INTENSITY — checks if the geodesic placed
                    # the final cut at a dark valley or at a bright midpoint.
                    if local_int_sub is not None:
                        ws_e_mask = (ws_local == 1) & local_domain
                        ws_i_mask = (ws_local == 2) & local_domain
                        if np.any(ws_e_mask) and np.any(ws_i_mask):
                            dil_e = binary_dilation(ws_e_mask, footprint=footprint_rectangle((3, 3, 3)))
                            resolved_boundary = dil_e & ws_i_mask
                            if np.any(resolved_boundary):
                                bnd_mean = float(np.mean(local_int_sub[resolved_boundary]))
                                bnd_verdict = ("CORRECT (dark valley)" if bnd_mean < domain_mean * 0.85
                                               else "WRONG (bright cut)")
                                flush_print(f"    [PROFILE|STITCH|GEO] resolved boundary mean_intensity={bnd_mean:.1f} "
                                            f"vs domain_mean={domain_mean:.1f} | => {bnd_verdict}")

                    cv_sub[(ws_local == 2) & local_conflict] = i_lab
                    
                    # Write safely back to the global memmap
                    final_mask[sub_slice] = cv_sub

            final_mask[sl] = canvas_view
            os.remove(path)

        # Convert to array for final steps (usually fits in RAM if chunks worked)
        ret = np.array(final_mask)
        del final_mask
        if os.path.exists(final_path):
            os.remove(final_path)

        # [PROFILE|CONSERVE] Post-stitch, pre-refinement inventory.
        _fg_stitch = int(np.count_nonzero(ret))
        _nobj_stitch = int(np.unique(ret[ret > 0]).size)
        flush_print(
            f"  [PROFILE|CONSERVE] POST-STITCH (pre-island): foreground_voxels={_fg_stitch} "
            f"| objects={_nobj_stitch} | delta_vs_input={_fg_stitch - _fg_in}"
        )

        ret = _reassign_disconnected_islands(ret, soma_mask)

        # [PROFILE|CONSERVE] Post-island inventory.
        _fg_island = int(np.count_nonzero(ret))
        _nobj_island = int(np.unique(ret[ret > 0]).size)
        flush_print(
            f"  [PROFILE|CONSERVE] POST-ISLAND: foreground_voxels={_fg_island} "
            f"| objects={_nobj_island} | delta_vs_stitch={_fg_island - _fg_stitch}"
        )

        # Size floor, applied globally so every label's size is its true whole size
        # (a chunk-clipped basin looks arbitrarily small inside the worker). Runs
        # AFTER the island pass so satellites have been reattached and the sizes
        # being judged are final. Merges only -- never deletes.
        ret = _merge_undersized_cells(
            ret, int(kwargs.get('min_size_threshold', 0) or 0)
        )

        # [PROFILE|CONSERVE] Post-undersize inventory.
        _fg_undersize = int(np.count_nonzero(ret))
        _nobj_undersize = int(np.unique(ret[ret > 0]).size)
        flush_print(
            f"  [PROFILE|CONSERVE] POST-UNDERSIZE: foreground_voxels={_fg_undersize} "
            f"| objects={_nobj_undersize} | delta_vs_island="
            f"{_fg_undersize - _fg_island}"
        )

        flush_print("  Refining (Filling voids + Relabeling)...")
        ret, _, _ = relabel_sequential(ret)

        # [PROFILE|CONSERVE] Final inventory + end-to-end verdict.
        _fg_out = int(np.count_nonzero(ret))
        _nobj_out = int(np.unique(ret[ret > 0]).size)
        flush_print(
            f"  [PROFILE|CONSERVE] OUTPUT: foreground_voxels={_fg_out} | objects={_nobj_out}"
        )
        flush_print(
            f"  [PROFILE|CONSERVE] END-TO-END: foreground delta={_fg_out - _fg_in} "
            f"(input={_fg_in} -> output={_fg_out})"
            + ("  *** NET FOREGROUND LOSS ***" if _fg_out < _fg_in else "")
        )

        return ret
    
    finally:
        # Emergency cleanup: remove any remaining .npy files in case of crash
        for i in chunk_data:
            if os.path.exists(chunk_data[i]['path']):
                try: os.remove(chunk_data[i]['path'])
                except: pass
        gc.collect()