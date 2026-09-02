"""
Cell Splitting Module (2D)
==========================

This module provides 2D cell separation logic that is exactly logically
equivalent to the 3D implementation (cell_splitting.py). It uses a chunked
processing architecture with seed-aware stitching to handle large-scale data
(e.g., whole-slide scans) without RAM overload.

Features:
- Intensity-weighted watershed landscape with global soma intensity normalisation.
- Three-check Region Adjacency Graph (RAG) interface analysis (valley depth,
  bright-cut, and local contrast) matching the 3D criteria exactly.
- Seed-aware orphan reassignment using marker voxels (not soma-ID matching).
- Intensity-modulated geodesic stitch conflict resolution with no-valley shortcut.
- Per-label void filling that is safe across touching cell boundaries.
- Overlapping chunked processing with global seed-based stitching.
"""

import os
import gc
import sys
from typing import List, Dict, Optional, Tuple, Set, Iterator, Any

import numpy as np
from scipy import ndimage
from skimage.morphology import binary_dilation, footprint_rectangle, disk  # type: ignore
from skimage.segmentation import relabel_sequential  # type: ignore
from tqdm import tqdm

# Import shared helpers from the 3D module where available.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

try:
    from ..module_3d.segmentation_helpers import (
        flush_print,
        _watershed_with_simpleitk,
        distance_transform_edt,
    )
except ImportError:
    def flush_print(*args: Any, **kwargs: Any) -> None:
        print(*args, **kwargs)
        sys.stdout.flush()

    def _watershed_with_simpleitk(landscape: np.ndarray, markers: np.ndarray) -> np.ndarray:
        """Fallback: skimage watershed used when SimpleITK is unavailable."""
        from skimage.segmentation import watershed
        return watershed(landscape, markers)

    from scipy.ndimage import distance_transform_edt

# Post-stitch passes and aggregates, shared with the 3D module (dimension-agnostic).
try:
    from ..module_3d.streaming_stats import accumulate_label_statistics
    from ..module_3d.streaming_passes import (
        accumulate_label_soma_map,
        apply_label_mapping,
        global_merge_pass,
        merge_undersized_streaming,
    )
except ImportError:  # pragma: no cover - direct script execution
    from streaming_stats import accumulate_label_statistics
    from streaming_passes import (
        accumulate_label_soma_map,
        apply_label_mapping,
        global_merge_pass,
        merge_undersized_streaming,
    )


def _soma_first_chunk_order_2d(
    n_chunks: int,
    shape_in_chunks: Tuple[int, int],
    chunk_slices: List[Tuple[slice, slice]],
    soma_locs: List[Optional[Tuple[slice, slice]]],
    relevant_soma_ids: Set[int],
) -> List[int]:
    """
    Order in which to visit chunks: breadth-first outward from the chunks that hold
    somata, instead of raster order.

    Order did not matter before, because every chunk was solved in isolation. It
    matters once a chunk can continue a cut started by a neighbour: the sweep has to
    begin where the information is -- the chunks holding two or more somata of one
    cell, the only places the merge tests are meaningful -- and radiate outward.
    Breadth-first also keeps the two sides of a contested region roughly balanced.

    Chunks with one soma seed the sweep next; anything still unvisited is appended,
    so no chunk is ever skipped.
    """
    from collections import deque

    per_chunk: List[Set[int]] = [set() for _ in chunk_slices]
    for s_idx, s_slice in enumerate(soma_locs):
        if s_slice is None:
            continue
        soma_id = s_idx + 1
        if soma_id not in relevant_soma_ids:
            continue
        for i, cs in enumerate(chunk_slices):
            if all(a.start < b.stop and b.start < a.stop
                   for a, b in zip(s_slice, cs)):
                per_chunk[i].add(soma_id)
    counts = [len(x) for x in per_chunk]

    def neighbours(index: int) -> List[int]:
        cy, cx = np.unravel_index(index, shape_in_chunks)
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = int(cy) + dy, int(cx) + dx
                if not (0 <= ny < shape_in_chunks[0]):
                    continue
                if not (0 <= nx < shape_in_chunks[1]):
                    continue
                out.append(int(np.ravel_multi_index((ny, nx), shape_in_chunks)))
        return out

    order: List[int] = []
    visited: Set[int] = set()

    def sweep(sources: List[int]) -> None:
        q = deque()
        for x in sources:
            if x not in visited:
                visited.add(x)
                q.append(x)
        while q:
            i = q.popleft()
            order.append(i)
            for nb in neighbours(i):
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

    sweep(sorted([i for i in range(n_chunks) if counts[i] >= 2],
                 key=lambda i: (-counts[i], i)))
    sweep(sorted([i for i in range(n_chunks) if counts[i] == 1]))
    sweep(list(range(n_chunks)))
    return order


def _get_chunk_slices_2d(
    image_shape: Tuple[int, int],
    chunk_shape: Tuple[int, int],
    overlap: int,
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
    # max(1, ...) guard: with overlap >= a chunk dimension the step goes
    # negative, range() yields nothing, and the coordinator silently produces ZERO
    # chunks -- every foreground pixel is discarded and the function still reports
    # success. Measured on a 3D stack: foreground delta -435996, exit status fine.
    step = tuple(max(1, chunk_shape[k] - overlap) for k in range(2))
    for y in range(0, image_shape[0], step[0]):
        for x in range(0, image_shape[1], step[1]):
            yield (
                slice(y, min(y + chunk_shape[0], image_shape[0])),
                slice(x, min(x + chunk_shape[1], image_shape[1])),
            )


# =============================================================================
# Graph & Metric Functions
# =============================================================================

def _analyze_local_intensity_difference_2d_aligned(
    interface_mask: np.ndarray,
    region1_mask: np.ndarray,
    region2_mask: np.ndarray,
    intensity_local: np.ndarray,
    local_analysis_radius: int,
    min_local_intensity_difference_threshold: float,
) -> bool:
    """
    Analyzes relative intensity difference between two regions at their interface.

    Args:
        interface_mask: Mask of the interface boundary.
        region1_mask: Mask of object A.
        region2_mask: Mask of object B.
        intensity_local: Local intensity image.
        local_analysis_radius: Radius for dilation to find local neighborhood.
        min_local_intensity_difference_threshold: Threshold for relative difference.

    Returns:
        bool: True if the regions are distinct enough, False if they should merge.
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
    cell_mean_intensity: float,
    spacing_tuple: Optional[Tuple[float, float]],
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
    footprint_dilation = footprint_rectangle((3, 3))

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
    lid_passed = _analyze_local_intensity_difference_2d_aligned(
        interface_mask, mask_A_local, mask_B_local, intensity_local,
        local_analysis_radius, min_local_intensity_difference,
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
        soma_ratio_passed
        and ratio_soma > min_path_intensity_ratio_heuristic * 0.7
    )
    if not soma_ratio_passed or not lid_passed:
        metrics['should_merge_decision'] = True
    elif not cell_mean_ratio_passed and soma_ratio_is_borderline:
        metrics['should_merge_decision'] = True

    # [PROFILING] Log all three checks and their raw values.
    bright_cut_warn = " *** BRIGHT CUT ***" if not cell_mean_ratio_passed else ""
    flush_print(
        f"  [PROFILE|INTERFACE] "
        f"mean_interface={mean_interface_intensity:.1f} | "
        f"soma_ref={avg_soma_intensity_for_interface:.1f} | "
        f"soma_ratio={ratio_soma:.4f} (thr={min_path_intensity_ratio_heuristic}, "
        f"passed={soma_ratio_passed}) | "
        f"cell_mean_ratio={ratio_cell_mean:.4f} (thr={max_interface_to_cell_mean_ratio}, "
        f"passed={cell_mean_ratio_passed}){bright_cut_warn} | "
        f"lid_passed={lid_passed} | "
        f"=> should_merge={metrics['should_merge_decision']}"
    )

    return metrics


def _min_soma_separation_2d(
    somas_A: List[int],
    somas_B: List[int],
    soma_centroids: Optional[Dict[int, np.ndarray]],
) -> float:
    """
    Smallest physical distance between a soma on side A and a soma on side B.

    Centroids are whole-image and in physical units, so a soma outside the current
    crop is still measured correctly. Returns 0.0 when the distance cannot be
    established -- no table, or neither side has a soma in it -- so an interface is
    never gated on missing data.

    The MINIMUM across the two sides, not the maximum: after an earlier merge a node
    can own several somata, and the question is whether ANY pair across this
    boundary is close enough to plausibly be one cell.
    """
    if not soma_centroids or not somas_A or not somas_B:
        return 0.0
    ca = [soma_centroids[s] for s in somas_A if s in soma_centroids]
    cb = [soma_centroids[s] for s in somas_B if s in soma_centroids]
    if not ca or not cb:
        return 0.0
    return float(
        min(np.linalg.norm(np.asarray(a) - np.asarray(b)) for a in ca for b in cb)
    )


def _build_adjacency_graph_for_cell_2d(
    current_cell_segments_mask_local: np.ndarray,
    original_cell_mask_local: np.ndarray,
    soma_mask_local: np.ndarray,
    soma_props_for_cell: Dict[int, Dict[str, Any]],
    intensity_local: np.ndarray,
    cell_mean_intensity: float,
    spacing_tuple: Optional[Tuple[float, float]],
    local_analysis_radius: int,
    min_local_intensity_difference: float,
    min_path_intensity_ratio_heuristic: float,
    max_interface_to_cell_mean_ratio: float = 0.85,
    node_seed_tags: Optional[Dict[int, Set[int]]] = None,
    require_local_somas: bool = False,
    soma_centroids: Optional[Dict[int, np.ndarray]] = None,
    max_seed_centroid_dist: float = 0.0,
) -> Tuple[Dict[int, Any], Dict[Tuple[int, int], Any]]:
    """
    Builds a Region Adjacency Graph (RAG) for segments within a single cell.
    Returns nodes and edges with merge metrics.

    ``node_seed_tags`` is a fallback for ``orig_somas``: {segment label -> set of
    seed ids}. A segment grown from a marker inherited from a neighbouring chunk
    belongs to a real cell whose soma may sit in a different chunk entirely, so
    ``soma_mask_local`` shows nothing inside it.

    ``require_local_somas`` records an interface as KEEP without scoring it when
    either side has no soma in this crop. A chunk that only inherited markers can
    produce a boundary in the middle of bright tissue; the tests correctly call that
    a bright cut and merge, and because merging unions the seed sets that one
    verdict fuses two genuinely separate cells across the whole image. Merge
    decisions are therefore confined to the chunks that can judge them, and the
    cross-chunk case is handled once, globally, by ``global_merge_pass``.

    ``max_seed_centroid_dist`` (um, 0 disables) records an interface as KEEP without
    scoring when the two sides' somata are further apart than that. The same bound is
    applied in ``global_merge_pass``, which with ``require_local_somas`` active is
    where long-range decisions actually happen.

    Identical semantics to the 3D module; keep the two in step.
    """
    nodes = {}
    edges = {}

    seg_lbls = np.unique(
        current_cell_segments_mask_local[current_cell_segments_mask_local > 0]
    )
    if len(seg_lbls) <= 1:
        return nodes, edges

    footprint_d = footprint_rectangle((3, 3))

    # Initialize Nodes
    for lbl in seg_lbls:
        mask = (current_cell_segments_mask_local == lbl)
        seeds_inside = np.unique(soma_mask_local[mask])
        somas_present = [s for s in seeds_inside if s > 0]
        if not somas_present and node_seed_tags:
            # Segment grown from a marker inherited from a neighbouring chunk: its
            # soma sits elsewhere in the image, so soma_mask_local shows nothing.
            # Without this fallback ref_intensity collapses to the 1.0 default and
            # both merge tests are scored against a meaningless reference.
            somas_present = sorted(node_seed_tags.get(int(lbl), ()))
        nodes[lbl] = {
            'volume': np.sum(mask),
            'orig_somas': somas_present,
        }

    # Find Edges and Calculate Metrics
    for i in range(len(seg_lbls)):
        lbl_A = seg_lbls[i]
        mask_A = (current_cell_segments_mask_local == lbl_A)
        dil_A = binary_dilation(mask_A, footprint=footprint_d)

        candidate_mask = (
            dil_A
            & (current_cell_segments_mask_local != lbl_A)
            & (current_cell_segments_mask_local > 0)
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

            if require_local_somas and not (
                np.any(soma_mask_local[mask_A] > 0)
                and np.any(soma_mask_local[mask_B] > 0)
            ):
                # Propagated interface: not judgeable here. Keep the cut.
                edges[edge_key] = {
                    'should_merge_decision': False,
                    'reason': 'propagated_interface_not_scored',
                }
                continue

            sep = _min_soma_separation_2d(somas_A, somas_B, soma_centroids)
            if max_seed_centroid_dist > 0 and sep > max_seed_centroid_dist:
                # Somata too far apart to be one cell. Not scored; cut stands.
                edges[edge_key] = {
                    'should_merge_decision': False,
                    'reason': 'seeds_beyond_max_merge_distance',
                    'soma_separation_um': sep,
                }
                flush_print(
                    f"  [PROFILE|INTERFACE] pair=({lbl_A},{lbl_B}) | "
                    f"soma_separation={sep:.1f}um > max_seed_centroid_dist="
                    f"{max_seed_centroid_dist:.1f}um => KEEP (not scored)"
                )
                continue

            all_somas = somas_A + somas_B
            soma_ints = [
                soma_props_for_cell[s]['mean_intensity']
                for s in all_somas if s in soma_props_for_cell
            ]
            ref_intensity = np.mean(soma_ints) if soma_ints else 1.0

            edges[edge_key] = _calculate_interface_metrics_2d_aligned(
                mask_A, mask_B, original_cell_mask_local, intensity_local,
                ref_intensity, cell_mean_intensity, spacing_tuple,
                local_analysis_radius, min_local_intensity_difference,
                min_path_intensity_ratio_heuristic, max_interface_to_cell_mean_ratio,
            )

    return nodes, edges


def _reassign_disconnected_islands_2d(
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
    either: it runs immediately after this pass, in ``_merge_undersized_cells_2d``,
    which MERGES an undersized cell into a neighbour instead of deleting it.
    Outright removal of small objects belongs to the step 1 / step 2 size filters.

    Instrumented so foreground conservation is visible in the run log.
    """
    flush_print("  [Refine] Checking for disconnected satellite fragments...")

    objs = ndimage.find_objects(segmentation)
    struct = ndimage.generate_binary_structure(2, 2)
    dilate_struct = ndimage.generate_binary_structure(2, 2)

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
        f"pixels_reassigned={px_reassigned} | pixels_kept={px_kept}"
    )
    flush_print(
        f"  [PROFILE|ISLAND|SUMMARY] foreground pixels before={fg_before} "
        f"after={fg_after} delta={fg_after - fg_before}"
        + ("  *** FOREGROUND LOST -- UNEXPECTED ***" if fg_after < fg_before
           else "  (conserved)")
    )

    return segmentation


def _merge_undersized_cells_2d(
    segmentation: np.ndarray,
    min_size_threshold: int,
    max_rounds: int = 20,
) -> np.ndarray:
    """
    Merge any final cell smaller than ``min_size_threshold`` into its most-contacted
    neighbouring cell. Never deletes anything. 2D counterpart of
    ``_merge_undersized_cells``; see that function for the full rationale.

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
      differently-labelled neighbour is KEPT at full size -- which is what
      preserves a genuinely small cell, seeded or not, including the case where
      every basin of one cell is undersized (merging cascades until one label
      remains, which is then kept whole). Foreground is exactly conserved.
    * **Smallest-first, and iterated**, so a recipient that grows past the
      threshold is not judged against a stale size.
    * This rule OVERRIDES the graph merge decisions, by design; each such merge
      is logged.
    """
    if min_size_threshold is None or min_size_threshold <= 0:
        flush_print("  [Refine] Undersized-cell merge skipped (threshold <= 0).")
        return segmentation

    flush_print(
        f"  [Refine] Merging cells below {min_size_threshold} pixels into their "
        "largest-contact neighbour (nothing is deleted)..."
    )

    fg_before = int(np.count_nonzero(segmentation))
    dilate_struct = ndimage.generate_binary_structure(2, 2)

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
        candidates = [
            int(lbl) for lbl in np.nonzero(counts)[0]
            if lbl != 0
            and counts[lbl] < min_size_threshold
            and int(lbl) not in kept_isolated
        ]
        if not candidates:
            break

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
                        f"(contact={int(counts_n[best_neighbor])} pixels)"
                    )
                    detail_shown += 1
            else:
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
        f"  [PROFILE|UNDERSIZE|SUMMARY] merged={n_merged} (pixels={px_merged}) | "
        f"kept_small_isolated={n_kept} (pixels={px_kept}) | rounds_used"
        f"={round_idx + 1}"
    )
    flush_print(
        f"  [PROFILE|UNDERSIZE|SUMMARY] foreground pixels before={fg_before} "
        f"after={fg_after} delta={fg_after - fg_before}"
        + ("  *** FOREGROUND LOST -- UNEXPECTED ***" if fg_after != fg_before
           else "  (conserved)")
    )

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
    prior_labels: Optional[np.ndarray] = None,
    prior_seed_map: Optional[Dict[int, Set[int]]] = None,
    **kwargs,
) -> Tuple[np.ndarray, Dict, Dict[int, Set[int]]]:
    """
    Worker: Separates multi-soma cells within a specific 2D chunk.

    ``prior_labels`` / ``prior_seed_map`` carry what neighbouring, already-processed
    chunks decided: the labels they wrote inside this chunk's extent, and the seed set
    each was tagged with. They are used as additional watershed markers.

    This is what lets a cut travel. A chunk holding fewer than two of a cell's somata
    used to take the ``len(seeds_in_crop) < 2`` shortcut and dump its whole piece of
    the cell into one untagged label, so a boundary computed where the somata are
    stopped at the first chunk wall and the stitcher had no seed information to keep
    the two cells apart. With the neighbours' labels acting as markers, and their seed
    tags carried onto the labels produced here, the existing seed-aware stitcher does
    the rest.

    With no priors supplied every path below reduces to what it was, so a
    single-chunk run is unchanged.

    Returns:
        chunk_result: The processed sub-image labels.
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
    # `_merge_undersized_cells_2d` in `separate_multi_soma_cells_2d`.

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
        # ---- INHERITED MARKERS -------------------------------------------------
        # Labels a neighbouring chunk already resolved inside this crop, each with
        # the seed set it was tagged with. Two markers belong to the same cell when
        # their seed sets overlap, so seeds are grouped by union-find and one marker
        # is emitted per group. An untagged prior carries no claim and is ignored.
        seed_parent: Dict[int, int] = {}

        def _find_seed(x: int) -> int:
            seed_parent.setdefault(x, x)
            while seed_parent[x] != x:
                seed_parent[x] = seed_parent[seed_parent[x]]
                x = seed_parent[x]
            return x

        def _union_seeds(a: int, b: int) -> None:
            ra, rb = _find_seed(a), _find_seed(b)
            if ra != rb:
                seed_parent[max(ra, rb)] = min(ra, rb)

        for s_id in seeds_in_crop:
            _find_seed(int(s_id))

        prior_tags: Dict[int, Set[int]] = {}
        local_prior = None
        if prior_labels is not None and prior_seed_map:
            local_prior = prior_labels[bbox_padded]
            for p in np.unique(local_prior[local_mask]):
                p = int(p)
                if p <= 0:
                    continue
                tag = {int(t) for t in (prior_seed_map.get(p) or ())}
                if not tag:
                    continue
                prior_tags[p] = tag
                tl = sorted(tag)
                for t in tl[1:]:
                    _union_seeds(tl[0], t)

        # One identity per seed group. Seeds physically present are indexed first so
        # that, with no priors, marker numbering is exactly what it always was.
        identity_index: Dict[int, int] = {}
        identity_seeds: List[Set[int]] = []

        def _identity_of(seed: int) -> int:
            root = _find_seed(int(seed))
            idx = identity_index.get(root)
            if idx is None:
                identity_seeds.append(set())
                idx = len(identity_seeds)
                identity_index[root] = idx
            return idx

        for s_id in seeds_in_crop:
            identity_seeds[_identity_of(int(s_id)) - 1].add(int(s_id))
        for p, tag in prior_tags.items():
            identity_seeds[_identity_of(sorted(tag)[0]) - 1] |= tag

        n_identities = len(identity_seeds)

        # Does every identity have a soma physically inside this crop?
        #
        # This decides whether the inherited markers are used at all. Priors exist
        # to carry a decision INTO a chunk that cannot make it itself. Where all the
        # somata are present, the priors add nothing and actively harm: an inherited
        # marker is a region whose extent is bounded by the chunk seam, and using it
        # as a watershed marker pre-claims that territory, so the basin boundary
        # hugs a straight, axis-aligned plane instead of the intensity trough.
        # Measured: a cell with both somata in-crop and one inherited marker split
        # 5535/717 with both boundaries flagged `WRONG (bright cut)`, the cut lying
        # along the chunk face.
        _local_identity_ids = {_identity_of(int(_s)) for _s in seeds_in_crop}
        _all_identities_local = (len(_local_identity_ids) == n_identities)

        # Case: fewer than 2 cells to tell apart here -> treat as a single object.
        # Unchanged, except that the label now inherits the seed tags of whatever
        # reached it, so the stitcher can still tell whose piece this is.
        if n_identities < 2:
            new_label = next_local_label
            chunk_result_view = chunk_result[bbox_padded]
            chunk_result_view[local_mask] = new_label
            chunk_result[bbox_padded] = chunk_result_view

            inherited: Set[int] = set(int(x) for x in seeds_in_crop)
            for tag in prior_tags.values():
                inherited |= tag
            label_to_seeds_map[new_label] = inherited
            next_local_label += 1
            continue

        # Case: Multiple seeds -> Perform Separation

        # Soma props: compute from local crop (not global lookup), matching 3D.
        soma_props = {}
        for s_id in seeds_in_crop:
            s_mask = (local_soma == s_id)
            mean_i = np.mean(local_intensity[s_mask]) if np.any(s_mask) else 1.0
            soma_props[s_id] = {'mean_intensity': mean_i}

        # [PROFILING] Seed quality report — flags seeds that are dim or tiny relative to peers.
        if len(seeds_in_crop) > 1:
            max_soma_int = max(p['mean_intensity'] for p in soma_props.values())
            marker_counts = {
                s_id: int(np.sum(local_soma == s_id)) for s_id in seeds_in_crop
            }
            max_marker_count = max(marker_counts.values())
            for s_id in seeds_in_crop:
                n_markers  = marker_counts[s_id]
                soma_int   = soma_props[s_id]['mean_intensity']
                int_ratio  = soma_int / (max_soma_int + 1e-6)
                size_ratio = n_markers / (max_marker_count + 1e-6)
                flags = []
                if int_ratio < 0.6:
                    flags.append(f"DIM (int_ratio={int_ratio:.2f})")
                if size_ratio < 0.3:
                    flags.append(f"TINY (size_ratio={size_ratio:.2f})")
                flag_str = " *** " + ", ".join(flags) + " ***" if flags else ""
                flush_print(
                    f"  [PROFILE|SEED] seed={s_id} | markers={n_markers} | "
                    f"soma_intensity={soma_int:.1f} | "
                    f"int_frac_of_brightest={int_ratio:.2f} | "
                    f"size_frac_of_largest={size_ratio:.2f}{flag_str}"
                )

        # Somata belonging to an inherited marker sit outside this chunk, so their
        # intensity comes from the global table the coordinator already computes.
        # The merge tests need it for ref_intensity.
        _global_int = kwargs.get('global_soma_intensities', {})
        for _tag in identity_seeds:
            for _s in _tag:
                if _s not in soma_props and _s in _global_int:
                    soma_props[_s] = {'mean_intensity': _global_int[_s]}

        # A. Seeded Watershed
        #
        # Markers: indexed 1..N (one per identity in crop).  This makes ws_local
        # labels comparable to the merge_map indices, matching the 3D approach.
        markers = np.zeros_like(local_mask, dtype=np.int32)
        if not _all_identities_local:
            # Only then are the neighbours' labels needed as markers.
            for p, tag in prior_tags.items():
                markers[(local_prior == p) & local_mask] = _identity_of(sorted(tag)[0])
        for s_id in seeds_in_crop:
            markers[local_soma == s_id] = _identity_of(int(s_id))

        # ---- LANDSCAPE GENERATION ----
        # TWO landscapes, chosen by what the markers are. Read this before changing
        # the branch below, because the two cases are not interchangeable.
        #
        #   markers are all somata      ->  d_seeds / speed**p
        #       d_seeds is straight-line distance from the nearest marker. With
        #       point-like somata that is a fair proxy, and it holds the cut near
        #       the geometric middle so one bright lobe cannot flood the whole
        #       object and strand the other seed in a sliver.
        #
        #   any marker is inherited     ->  1 / speed**p   (pure cost)
        #       An inherited marker is a whole REGION handed over by a neighbouring
        #       chunk, not a point. d_seeds measured from a region puts the boundary
        #       at the Euclidean midpoint between the two regions, which is wherever
        #       the chunk seams happen to fall -- typically mid-tissue rather than at
        #       the dark contact. Symptom in the log: `WRONG (bright cut -- Voronoi
        #       bias!)`, and in the viewer, a branch assigned to the wrong cell.
        #       The pure cost field has no distance term, so the boundary is set by
        #       the speed field alone and lands at the dark neck.
        #
        # Identical to the 3D module. Keep the two in step.

        # 1. Geometric thickness — large in cell centres, drops at boundaries/necks.
        dt = distance_transform_edt(local_mask, sampling=spacing)

        # 2. Euclidean distance from the markers.
        d_seeds = distance_transform_edt(markers == 0, sampling=spacing)

        # 3. Base speed
        speed = dt + 1e-5

        # 4. Modulate speed by intensity using global soma intensity normalisation.
        #    Using the mean of the GLOBAL soma intensities (not local min-max) gives
        #    a consistent scale across all chunks — matches 3D exactly.
        intensity_weight = kwargs.get('intensity_weight', 0.5)
        if intensity_weight > 0:
            cell_global_seeds = list(
                kwargs.get('cell_to_somas', {}).get(cell_label, seeds_in_crop)
            )
            global_intensities = kwargs.get('global_soma_intensities', {})

            soma_ints = [global_intensities.get(s, 1.0) for s in cell_global_seeds]
            ref_int = np.mean(soma_ints) if soma_ints else (local_intensity.max() + 1e-6)

            norm_int = np.clip(local_intensity / (ref_int + 1e-6), 0.0, 1.0)
            speed = speed * (1.0 + intensity_weight * norm_int)

        # 5. Final landscape and watershed
        speed_power = kwargs.get('speed_power', 1.5)
        # No flag on this branch. A default of False silently reintroduces the
        # Voronoi bias on every propagated chunk, and the failure is invisible
        # except as misassigned branches in the viewer. Which landscape ran is
        # printed every time so it can never be in doubt.
        if not _all_identities_local:
            flush_print(
                f"  [PROFILE|LANDSCAPE] cell={cell_label}: PURE COST "
                f"(1/speed^{speed_power}) -- {len(prior_tags)} inherited marker(s) "
                f"in play; an inherited marker is a region, so d_seeds measured "
                f"from it would land the cut at the chunk seam"
            )
            landscape = 1.0 / (speed ** speed_power)
        else:
            flush_print(
                f"  [PROFILE|LANDSCAPE] cell={cell_label}: "
                f"d_seeds/speed^{speed_power} (every identity has a local soma; "
                f"priors not used as markers)"
            )
            landscape = d_seeds / (speed ** speed_power)

        ws_local = _watershed_with_simpleitk(landscape, markers)
        ws_local[~local_mask] = 0

        # [PROFILING] Log what the watershed actually produced.
        ws_ids, ws_counts = np.unique(ws_local[ws_local > 0], return_counts=True)
        flush_print(
            f"  [PROFILE|WS] watershed output: labels={ws_ids} | pixel_counts={ws_counts}"
        )
        cell_mean_int = float(np.mean(local_intensity[local_mask]))
        for ws_id, ws_cnt in zip(ws_ids, ws_counts):
            _ident = sorted(identity_seeds[ws_id - 1]) if (ws_id - 1) < len(identity_seeds) else []
            seed_id = _ident[0] if len(_ident) == 1 else (_ident or '?')
            soma_int = soma_props.get(seed_id, {}).get('mean_intensity', float('nan'))
            basin_mask = ws_local == ws_id
            basin_mean = float(np.mean(local_intensity[basin_mask]))
            frac_bright = float(np.mean(local_intensity[basin_mask] > cell_mean_int))
            flush_print(
                f"    ws_label={ws_id} (->seed {seed_id}, soma_intensity={soma_int:.1f}): "
                f"{ws_cnt} pixels | mean_intensity={basin_mean:.1f} | "
                f"frac_above_cell_mean={frac_bright:.2f}"
            )

        # [PROFILING] BOUNDARY INTENSITY CHECK
        flush_print(f"  [PROFILE|WS|BOUNDARY] cell_mean_intensity={cell_mean_int:.1f}")
        footprint_b = footprint_rectangle((3, 3))
        for ws_id in ws_ids:
            basin_mask = ws_local == ws_id
            dilated = binary_dilation(basin_mask, footprint=footprint_b)
            boundary_pixels = dilated & (ws_local > 0) & (~basin_mask)
            if np.any(boundary_pixels):
                bnd_mean = float(np.mean(local_intensity[boundary_pixels]))
                bnd_frac_below = float(np.mean(local_intensity[boundary_pixels] < cell_mean_int))
                verdict = (
                    "CORRECT (dark valley)"
                    if bnd_mean < 0.7 * cell_mean_int
                    else "WRONG (bright cut — Voronoi bias!)"
                )
                flush_print(
                    f"    boundary of ws_label={ws_id}: mean_intensity={bnd_mean:.1f} | "
                    f"frac_below_cell_mean={bnd_frac_below:.2f} | => {verdict}"
                )

        # B. Graph-Based Merging
        nodes, edges = _build_adjacency_graph_for_cell_2d(
            ws_local, local_mask, local_soma, soma_props, local_intensity,
            cell_mean_int, spacing,
            kwargs.get('local_analysis_radius', 10),
            kwargs.get('min_local_intensity_difference', 0.0),
            kwargs.get('min_path_intensity_ratio', 1.0),
            kwargs.get('max_interface_to_cell_mean_ratio', 0.85),
            node_seed_tags={i + 1: t for i, t in enumerate(identity_seeds)},
            require_local_somas=True,
            soma_centroids=kwargs.get('global_soma_centroids', {}),
            max_seed_centroid_dist=float(kwargs.get('max_seed_centroid_dist', 0.0)),
        )

        # [PROFILING] Summarize graph decisions.
        merge_edges = [(k, v) for k, v in edges.items() if v['should_merge_decision']]
        keep_edges  = [(k, v) for k, v in edges.items() if not v['should_merge_decision']]
        flush_print(
            f"  [PROFILE|GRAPH] cell={cell_label}: "
            f"total_edges={len(edges)} | merge_edges={len(merge_edges)} | "
            f"keep_edges={len(keep_edges)}"
        )
        for edge_key, _ in merge_edges:
            flush_print(f"    [PROFILE|GRAPH] MERGE: {edge_key}")
        for edge_key, edge_val in keep_edges:
            _why = edge_val.get('reason')
            flush_print(
                f"    [PROFILE|GRAPH] KEEP:  {edge_key}"
                + (f" ({_why})" if _why else "")
            )

        merge_map = {i: i for i in range(n_identities + 2)}
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
            final_local_mask[ws_local == old_id] = merge_map[old_id]

        # [PROFILING] Show merge_map and post-merge label counts.
        flush_print(
            f"  [PROFILE|GRAPH] merge_map (ws_id -> final_id): "
            f"{dict(list(merge_map.items())[:20])}"
        )
        final_ids, final_counts = np.unique(
            final_local_mask[final_local_mask > 0], return_counts=True
        )
        flush_print(
            f"  [PROFILE|GRAPH] post-merge labels={final_ids} | pixel_counts={final_counts}"
        )

        # C. Seed-Aware Orphan Reassignment
        # Ensures every fragment contains a seed; merges orphan fragments into
        # the best-touching neighbour.  Uses the markers array (indexed 1..N)
        # to detect seed presence — matches 3D logic exactly.
        unique_result_ids = np.unique(final_local_mask[final_local_mask > 0])
        dilation_struct = footprint_rectangle((3, 3))
        cc_struct = ndimage.generate_binary_structure(2, 2)  # 8-conn incl. diagonals

        for uid in unique_result_ids:
            cell_mask = (final_local_mask == uid)
            cc_labels, num_cc = ndimage.label(cell_mask, structure=cc_struct)

            if num_cc <= 1:
                continue

            for i in range(1, num_cc + 1):
                frag_mask = (cc_labels == i)
                has_seed = np.any(markers[frag_mask] > 0)

                if not has_seed:
                    # Orphan detected: merge into best neighbour
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
                        # post-stitch, in `_merge_undersized_cells_2d`, where it MERGES
                        # rather than deletes and where fragment sizes are true (a
                        # chunk-clipped fragment looks arbitrarily small here).
                        flush_print(
                            f"  [PROFILE|ORPHAN] worker KEEP orphan: cell={cell_label} "
                            f"uid={uid} orphan_size={int(np.sum(frag_mask))} "
                            f"(isolated satellite; never deleted)"
                        )

        # D. Map to Global IDs
        final_local_mask_clean, _, _ = relabel_sequential(final_local_mask)
        chunk_result_view = chunk_result[bbox_padded]

        for local_id in np.unique(final_local_mask_clean[final_local_mask_clean > 0]):
            mask_l = (final_local_mask_clean == local_id)

            seeds_in_segment = np.unique(local_soma[mask_l])
            seeds_in_segment_set = set(
                int(x) for x in seeds_in_segment[seeds_in_segment > 0]
            )
            # Plus the seed sets of any markers inside this segment. With no priors
            # every marker is a soma pixel, so this adds nothing and the tag is the
            # same set as before.
            for _mid in np.unique(markers[mask_l]):
                if _mid > 0 and (_mid - 1) < len(identity_seeds):
                    seeds_in_segment_set |= identity_seeds[_mid - 1]

            global_lbl = next_local_label
            chunk_result_view[mask_l] = global_lbl

            label_to_seeds_map[global_lbl] = seeds_in_segment_set
            next_local_label += 1

        chunk_result[bbox_padded] = chunk_result_view

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
    **kwargs,
) -> np.ndarray:
    """
    Main Coordinator for separating multi-soma cells in large 2D images.
    Uses chunking with seed-aware stitching to handle boundary artifacts.

    Args:
        segmentation_mask: 2D labeled segmentation.
        intensity_volume: 2D intensity image.
        soma_mask: 2D mask of seeds.
        spacing: Pixel spacing (Y, X).
        chunk_shape: Size of processing chunks.
        overlap: Overlap between chunks.
        **kwargs: Parameters for separation (weights, thresholds).

    Returns:
        np.ndarray: Refined 2D segmentation mask.
    """
    flush_print("[SepMultiSoma_2D] Starting (Chunked + Seed-Aware)...")

    # [PROFILE|CONSERVE] Input inventory for end-to-end foreground accounting.
    _fg_in = int(np.count_nonzero(segmentation_mask))
    _nobj_in = int(np.unique(segmentation_mask[segmentation_mask > 0]).size)
    flush_print(
        f"  [PROFILE|CONSERVE] INPUT: foreground_pixels={_fg_in} | objects={_nobj_in}"
    )

    # 1. Identify Multi-Soma Cells (Global Check)
    cell_to_somas: Dict[int, Set[int]] = {}

    global_soma_centroids = {}
    global_soma_intensities = {}

    soma_locs = ndimage.find_objects(soma_mask)
    for s_idx, s_slice in enumerate(soma_locs):
        if s_slice is None:
            continue
        soma_id = s_idx + 1

        # Calculate Global Centroid (2D)
        cy = (s_slice[0].start + s_slice[0].stop) / 2.0
        cx = (s_slice[1].start + s_slice[1].stop) / 2.0
        if spacing:
            cy *= spacing[0]; cx *= spacing[1]
        global_soma_centroids[soma_id] = np.array([cy, cx])

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

    # Overlap must leave a positive stride on every axis, or chunking degenerates:
    # an overlap equal to a chunk dimension gives a zero stride, and a larger one
    # yields no chunks at all, returning an empty image.
    _max_ov = max(0, min(chunk_shape) - 1)
    if overlap > _max_ov:
        flush_print(
            f"  [SepMultiSoma2D] overlap={overlap} is not smaller than the smallest "
            f"chunk dimension {min(chunk_shape)}; clamping to {_max_ov}. "
            f"Reduce `overlap` or increase `chunk_shape` to silence this."
        )
        overlap = _max_ov

    chunk_slices = list(
        _get_chunk_slices_2d(segmentation_mask.shape, chunk_shape, overlap)
    )
    if not chunk_slices:
        flush_print("  [SepMultiSoma2D] *** no chunks generated; returning input "
                    "unchanged ***")
        return segmentation_mask.copy()
    chunk_data = {}  # Stores (path, shape, seed_map)

    chunk_grid = (
        len(range(0, segmentation_mask.shape[0], max(1, chunk_shape[0] - overlap))),
        len(range(0, segmentation_mask.shape[1], max(1, chunk_shape[1] - overlap))),
    )
    relevant_somas: Set[int] = set()
    for _c in multi_soma_labels:
        relevant_somas |= set(cell_to_somas.get(_c, set()))
    chunk_order = _soma_first_chunk_order_2d(
        len(chunk_slices), chunk_grid, chunk_slices, soma_locs, relevant_somas
    )

    # Running record of what has been decided so far, so a chunk can pick up its
    # neighbours' cuts and continue them. Labels only -- the .npy files, the stitcher
    # and everything downstream are untouched.
    prior_path = os.path.join(memmap_dir, f"prior2d_{os.getpid()}.mmp")
    prior_mask = np.memmap(
        prior_path, dtype=np.int32, mode='w+', shape=segmentation_mask.shape
    )
    prior_mask[:] = 0
    prior_seed_map: Dict[int, Set[int]] = {}

    flush_print(f"  Processing {len(chunk_slices)} chunks...")
    flush_print(
        f"  [PROFILE|ORDER] visiting soma-bearing chunks first "
        f"(grid={chunk_grid}), then radiating outward"
    )

    try:
        for i in tqdm(chunk_order, desc="Processing Chunks"):
            sl = chunk_slices[i]
            seg_chunk   = segmentation_mask[sl]
            int_chunk   = intensity_volume[sl]
            soma_chunk  = soma_mask[sl]

            # Use large offsets to avoid ID collisions between chunks initially
            chunk_offset = (i + 1) * 1_000_000

            prior_chunk = np.array(prior_mask[sl])

            res, _, seed_map = _separate_multi_soma_cells_chunk_2d(
                seg_chunk, int_chunk, soma_chunk,
                spacing, chunk_offset, multi_soma_labels,
                prior_labels=prior_chunk,
                prior_seed_map=prior_seed_map,
                global_offset=(sl[0].start, sl[1].start),
                **kwargs,
            )
            del prior_chunk

            path = os.path.join(memmap_dir, f"chunk2d_{i}_{os.getpid()}.npy")
            np.save(path, res)

            chunk_data[i] = {'path': path, 'shape': res.shape, 'seed_map': seed_map}
            prior_seed_map.update(seed_map)

            # First writer wins, so a decision already taken stays put and the markers
            # handed to later chunks do not shift under them.
            prior_view = prior_mask[sl]
            _fill = (prior_view == 0) & (res > 0)
            prior_view[_fill] = res[_fill]
            prior_mask[sl] = prior_view

            del res, prior_view, _fill  # Free RAM immediately
            gc.collect()

        del prior_mask
        if os.path.exists(prior_path):
            os.remove(prior_path)

        # 3. Seed-Aware Stitching Logic
        flush_print("  Stitching with Transitive Seed Verification...")
        label_map: Dict[int, int] = {}

        # Global registry of seeds for every label ID
        global_seed_lookup = {}
        for idx, data in chunk_data.items():
            global_seed_lookup.update(data['seed_map'])

        # Dynamic tracker: Maps 'Root Label' -> 'Set of Seeds in this merged group'
        def get_group_seeds(lbl_id):
            root = label_map.get(lbl_id, lbl_id)
            if root not in group_seeds_cache:
                return global_seed_lookup.get(root, set())
            return group_seeds_cache[root]

        group_seeds_cache = {}

        # 2D grid dimensions for neighbour index arithmetic
        # Same stride floor as `_get_chunk_slices_2d`, so the stitcher's grid
        # matches the one the chunks were actually generated on.
        grid_w = len(range(0, segmentation_mask.shape[1], max(1, chunk_shape[1] - overlap)))
        grid_h = len(range(0, segmentation_mask.shape[0], max(1, chunk_shape[0] - overlap)))
        shape_in_chunks = [grid_h, grid_w]

        for i, chunk_slice1 in enumerate(tqdm(chunk_slices, desc="Stitching Analysis")):
            if i not in chunk_data:
                continue

            cy_idx, cx_idx = divmod(i, grid_w)
            neighbors = []
            if cy_idx + 1 < shape_in_chunks[0]:
                neighbors.append((cy_idx + 1) * grid_w + cx_idx)
            if cx_idx + 1 < shape_in_chunks[1]:
                neighbors.append(cy_idx * grid_w + (cx_idx + 1))

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

                    # Check seeds of the Roots (not the fragments), so that
                    # already-merged groups propagate their seed membership.
                    s1_set = get_group_seeds(root1)
                    s2_set = get_group_seeds(root2)

                    # Both groups have known seeds and they don't overlap -> CONFLICT
                    if s1_set and s2_set and s1_set.isdisjoint(s2_set):
                        continue  # Do not merge Cell A and Cell B

                    # Otherwise, merge is safe (or involves an untagged bridge)
                    target = min(root1, root2)
                    source = max(root1, root2)

                    # Update map
                    label_map[source] = target
                    label_map[root1] = target
                    label_map[root2] = target

                    # Update cache: union of seeds
                    new_seeds = s1_set.union(s2_set)
                    group_seeds_cache[target] = new_seeds

                    # Redirect source's cache to target (cleanup)
                    if source in group_seeds_cache:
                        del group_seeds_cache[source]

                    # Path compression for existing mappings
                    keys_to_update = [k for k, v in label_map.items() if v == source]
                    for k in keys_to_update:
                        label_map[k] = target

        # 4. Construct Final Mask
        final_path = os.path.join(memmap_dir, "stitched_2d.mmp")
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
            flush_print(
                f"  [PROFILE|STITCH] chunk={i} | written={n_written} | "
                f"conflicts={n_conflict} | "
                f"conflict_labels_existing={np.unique(canvas_view[conflict_mask]).tolist()} | "
                f"conflict_labels_incoming={np.unique(res[conflict_mask]).tolist()}"
            )

            if np.any(conflict_mask):
                unique_pairs = np.unique(
                    np.vstack((canvas_view[conflict_mask], res[conflict_mask])),
                    axis=1,
                ).T

                for e_lab, i_lab in unique_pairs:
                    pair_conflict = (
                        (canvas_view == e_lab) & (res == i_lab) & conflict_mask
                    )
                    if not np.any(pair_conflict):
                        continue

                    # 1. Get GLOBAL coordinates of the conflict pixels
                    cy_px, cx_px = np.where(pair_conflict)
                    gy, gx = cy_px + sl[0].start, cx_px + sl[1].start

                    # 2. Extract a padded block from the GLOBAL canvas
                    pad = overlap + 4
                    y_min = max(0, gy.min() - pad)
                    y_max = min(final_mask.shape[0], gy.max() + pad + 1)
                    x_min = max(0, gx.min() - pad)
                    x_max = min(final_mask.shape[1], gx.max() + pad + 1)

                    sub_slice = (slice(y_min, y_max), slice(x_min, x_max))
                    cv_sub = final_mask[sub_slice].copy()

                    # 3. Project conflict pixels into local cv_sub coordinates
                    local_cy2, local_cx2 = gy - y_min, gx - x_min
                    local_conflict = np.zeros(cv_sub.shape, dtype=bool)
                    local_conflict[local_cy2, local_cx2] = True

                    # 4. Find safe zones globally
                    local_e_safe = (cv_sub == e_lab) & ~local_conflict
                    local_i_safe = (cv_sub == i_lab) & ~local_conflict

                    # Failsafe
                    if not np.any(local_e_safe) or not np.any(local_i_safe):
                        flush_print(
                            f"  [STITCHER] Failsafe Triggered. "
                            f"e:{np.sum(local_e_safe)}, i:{np.sum(local_i_safe)}"
                        )
                        cv_sub[local_conflict] = i_lab
                        final_mask[sub_slice] = cv_sub
                        continue

                    # 5. Run Geodesic Micro-Watershed
                    local_domain = local_conflict | local_e_safe | local_i_safe
                    stitch_markers = np.zeros(local_domain.shape, dtype=np.int32)
                    stitch_markers[local_e_safe] = 1
                    stitch_markers[local_i_safe] = 2

                    dt = distance_transform_edt(local_domain, sampling=spacing)

                    # NOTE: d_seeds is NOT used here to avoid Euclidean bias; see
                    # the per-cell watershed comment for the full rationale. The pure
                    # cost field 1/speed^p is the correct formulation for the stitch.

                    # FIX: Use intensity-modulated speed, identical to the per-cell
                    # watershed. Previously this was geometry-only (speed = dt + 1e-5),
                    # which caused large conflict zones to be cut along the geometric
                    # midplane — producing diagonal mask artifacts.
                    stitch_intensity_weight = kwargs.get('intensity_weight', 0.5)
                    if stitch_intensity_weight > 0 and np.any(local_domain):
                        local_int_sub = intensity_volume[sub_slice].astype(float)
                        smoothed_sub = ndimage.gaussian_filter(local_int_sub, sigma=1.0)
                        p1_s, p99_s = np.percentile(
                            smoothed_sub[local_domain], [1, 99]
                        )
                        norm_int_sub = np.clip(
                            (smoothed_sub - p1_s) / (p99_s - p1_s + 1e-6), 0.0, 1.0
                        )
                        max_dt_sub = float(np.max(dt)) if np.any(local_domain) else 1.0
                        speed = (
                            dt
                            + (norm_int_sub * max_dt_sub * stitch_intensity_weight)
                            + 1e-5
                        )

                        # [PROFILING] Confirm intensity contribution vs geometry.
                        geom_c = dt[local_domain].mean()
                        int_c  = (
                            norm_int_sub[local_domain]
                            * max_dt_sub
                            * stitch_intensity_weight
                        ).mean()
                        ratio_c = int_c / (geom_c + 1e-9)
                        contrib_warn = (
                            " *** WARN: intensity nearly absent from speed! ***"
                            if ratio_c < 0.1 else ""
                        )
                        flush_print(
                            f"    [PROFILE|STITCH|GEO] speed: geometry={geom_c:.3f} "
                            f"intensity_term={int_c:.3f} ratio={ratio_c:.2f} | "
                            f"p1={p1_s:.1f} p99={p99_s:.1f}{contrib_warn}"
                        )

                        # [PROFILING] SAFE ZONE INTENSITY CHECK
                        e_safe_mean = float(np.mean(local_int_sub[local_e_safe])) if np.any(local_e_safe) else float('nan')
                        i_safe_mean = float(np.mean(local_int_sub[local_i_safe])) if np.any(local_i_safe) else float('nan')
                        conflict_mean = float(np.percentile(local_int_sub[local_conflict], 5))
                        domain_mean   = float(np.mean(local_int_sub[local_domain]))
                        is_valley = conflict_mean < min(e_safe_mean, i_safe_mean) * 0.85
                        safe_contrast = abs(e_safe_mean - i_safe_mean) / (
                            max(e_safe_mean, i_safe_mean) + 1e-6
                        )
                        contrast_warn = (
                            " *** LOW CONTRAST: geodesic cut may be geometric ***"
                            if safe_contrast < 0.05 else ""
                        )
                        flush_print(
                            f"    [PROFILE|STITCH|GEO] INTENSITY MAP: "
                            f"e_safe_mean={e_safe_mean:.1f} | conflict_mean={conflict_mean:.1f} | "
                            f"i_safe_mean={i_safe_mean:.1f} | domain_mean={domain_mean:.1f}"
                        )
                        flush_print(
                            f"    [PROFILE|STITCH|GEO] conflict_is_valley={is_valley} "
                            f"(conflict={conflict_mean:.0f} vs "
                            f"safe_min={min(e_safe_mean, i_safe_mean):.0f}) | "
                            f"safe_zone_contrast={safe_contrast:.3f}{contrast_warn}"
                        )
                        if not is_valley:
                            flush_print(
                                "    [PROFILE|STITCH|GEO] *** SAFE ZONE PROBLEM: conflict "
                                "zone is NOT the darkest region. Safe zones are likely on "
                                "the wrong side of the true cell boundary — upstream "
                                "watershed error propagated here. ***"
                            )
                    else:
                        local_int_sub = None
                        is_valley = True  # no intensity info; proceed with geodesic
                        speed = dt + 1e-5
                        flush_print(
                            f"    [PROFILE|STITCH|GEO] pair=({e_lab},{i_lab}) | "
                            f"intensity_weight=0 — falling back to geometry-only speed."
                        )

                    speed_power = kwargs.get('speed_power', 1.5)

                    # When there is no dark valley, running the geodesic watershed
                    # is pointless: any cut will be at a bright point. Instead,
                    # assign all conflict pixels to the larger safe zone.
                    if local_int_sub is not None and not is_valley:
                        winner = (
                            e_lab if np.sum(local_e_safe) >= np.sum(local_i_safe)
                            else i_lab
                        )
                        flush_print(
                            f"    [PROFILE|STITCH|GEO] NO VALLEY: skipping geodesic — "
                            f"assigning all {int(np.sum(local_conflict))} conflict pixels "
                            f"to larger safe zone (winner={winner}, "
                            f"e_safe={int(np.sum(local_e_safe))} vs "
                            f"i_safe={int(np.sum(local_i_safe))})"
                        )
                        cv_sub[local_conflict] = winner
                        final_mask[sub_slice] = cv_sub
                        continue  # skip geodesic for this pair

                    d_seeds_stitch = distance_transform_edt(
                        stitch_markers == 0, sampling=spacing
                    )
                    landscape = d_seeds_stitch / (speed ** speed_power)
                    landscape[~local_domain] = 1e6

                    ws_local = _watershed_with_simpleitk(landscape, stitch_markers)

                    # [PROFILING] Log how many conflict pixels went to each label.
                    n_to_e = int(np.sum((ws_local == 1) & local_conflict))
                    n_to_i = int(np.sum((ws_local == 2) & local_conflict))
                    flush_print(
                        f"    [PROFILE|STITCH|GEO] pair=({e_lab},{i_lab}) | "
                        f"conflict_pixels={int(np.sum(local_conflict))} | "
                        f"geodesic => e_lab={n_to_e} pixels, i_lab={n_to_i} pixels | "
                        f"e_safe_size={int(np.sum(local_e_safe))} "
                        f"i_safe_size={int(np.sum(local_i_safe))}"
                    )

                    # [PROFILING] RESOLVED BOUNDARY INTENSITY
                    if local_int_sub is not None:
                        ws_e_mask = (ws_local == 1) & local_domain
                        ws_i_mask = (ws_local == 2) & local_domain
                        if np.any(ws_e_mask) and np.any(ws_i_mask):
                            dil_e = binary_dilation(
                                ws_e_mask, footprint=footprint_rectangle((3, 3))
                            )
                            resolved_boundary = dil_e & ws_i_mask
                            if np.any(resolved_boundary):
                                bnd_mean = float(
                                    np.mean(local_int_sub[resolved_boundary])
                                )
                                bnd_verdict = (
                                    "CORRECT (dark valley)"
                                    if bnd_mean < domain_mean * 0.85
                                    else "WRONG (bright cut)"
                                )
                                flush_print(
                                    f"    [PROFILE|STITCH|GEO] resolved boundary "
                                    f"mean_intensity={bnd_mean:.1f} vs "
                                    f"domain_mean={domain_mean:.1f} | => {bnd_verdict}"
                                )

                    cv_sub[(ws_local == 2) & local_conflict] = i_lab

                    # Write safely back to the global memmap
                    final_mask[sub_slice] = cv_sub

            final_mask[sl] = canvas_view
            os.remove(path)

        # Convert to array for final steps
        ret = np.array(final_mask)
        del final_mask
        if os.path.exists(final_path):
            os.remove(final_path)

        # Aggregates for every pass below: label sizes, intensity sums, and the
        # adjacency graph with each interface's bounding box. One bounded-memory
        # sweep replaces the whole-image `np.unique` / `bincount` / `find_objects`
        # calls the refinement passes used to make. Shared with the 3D module.
        _stats_block = tuple(kwargs.get('stats_block_shape', (128, 128)))
        _stats = accumulate_label_statistics(
            ret, intensity_volume, block_shape=_stats_block
        )

        # [PROFILE|CONSERVE] Post-stitch, pre-refinement inventory.
        _fg_stitch = int(_stats.label_count.sum())
        _nobj_stitch = int(_stats.labels.size)
        flush_print(
            f"  [PROFILE|CONSERVE] POST-STITCH (pre-island): foreground_pixels={_fg_stitch} "
            f"| objects={_nobj_stitch} | delta_vs_input={_fg_stitch - _fg_in}"
        )

        # ---- Merge tests, once, on the assembled image -----------------------
        # The worker no longer scores an interface unless both basins own a soma in
        # its own chunk. It cannot: a chunk that only inherited its markers puts the
        # boundary between two marker regions rather than at a valley, the tests
        # correctly call that a bright cut, and because merging unions the seed sets
        # one wrong verdict fuses two cells across the whole image.
        #
        # Here every interface between two final labels is visible exactly once,
        # with both cells whole. THIS is what rejoins a cell that received several
        # erroneous somata when those somata landed in different chunks -- the
        # over-seeding rescue the per-chunk tests structurally cannot perform.
        #
        # Runs BEFORE the island pass on purpose: an over-split leaves seedless
        # fragments that the island pass would otherwise hand out on contact area
        # alone, cementing a split that should not have existed.
        _merge_params = {
            k: kwargs[k] for k in (
                'local_analysis_radius',
                'min_local_intensity_difference',
                'min_path_intensity_ratio',
                'max_interface_to_cell_mean_ratio',
            ) if k in kwargs
        }
        _merged = global_merge_pass(
            ret, intensity_volume, soma_mask, spacing,
            _calculate_interface_metrics_2d_aligned,
            stats=_stats,
            global_soma_intensities=kwargs.get('global_soma_intensities', {}),
            global_soma_centroids=kwargs.get('global_soma_centroids', {}),
            max_seed_centroid_dist=float(kwargs.get('max_seed_centroid_dist', 0.0)),
            block_shape=_stats_block,
            log=flush_print,
            **_merge_params
        )
        if _merged:
            _fg_merge = int(np.count_nonzero(ret))
            flush_print(
                f"  [PROFILE|CONSERVE] POST-GLOBALMERGE: foreground_pixels={_fg_merge} "
                f"| objects={int(np.unique(ret[ret > 0]).size)} "
                f"| delta_vs_stitch={_fg_merge - _fg_stitch}"
            )

        ret = _reassign_disconnected_islands_2d(ret, soma_mask)

        # Fresh aggregates: the island pass moved pixels between labels.
        _stats_island = accumulate_label_statistics(ret, None, block_shape=_stats_block)

        # [PROFILE|CONSERVE] Post-island inventory.
        _fg_island = int(_stats_island.label_count.sum())
        _nobj_island = int(_stats_island.labels.size)
        flush_print(
            f"  [PROFILE|CONSERVE] POST-ISLAND: foreground_pixels={_fg_island} "
            f"| objects={_nobj_island} | delta_vs_stitch={_fg_island - _fg_stitch}"
        )

        # Size floor, applied globally so every label's size is its true whole size
        # (a chunk-clipped basin looks arbitrarily small inside the worker). Runs
        # AFTER the island pass so satellites have been reattached and the sizes
        # being judged are final. Merges only -- never deletes.
        #
        # The size floor is a deliberate lever against over-splitting, so it must
        # apply to watershed basins even though every basin owns a soma -- owning a
        # soma is why the basin exists. Whether a label is a split basin or a whole
        # original object is already decided by whether it has an adjacent label:
        # distinct objects in the step 2 mask are separated by background, so only
        # siblings from one split touch each other. A whole object therefore has no
        # neighbour to merge into and is kept at full size whatever its size, which
        # is exactly the rule wanted: an original mask with one soma or none becomes
        # one cell, no questions.
        #
        # `protect_seeded_cells` exempts any soma-owning label from the floor. That
        # exempts every basin, which disables the lever entirely -- symptom in the
        # log is `[PROFILE|UNDERSIZE|SUMMARY] merged=0` on an image that visibly
        # needs merging. Default OFF for that reason. Turn it on only to shield
        # small-but-real cells from a threshold set too high for the data, and
        # prefer fixing the threshold or the intensity levers instead.
        _protect = set()
        if kwargs.get('protect_seeded_cells', False):
            _protect = {
                lbl for lbl, somas in accumulate_label_soma_map(
                    ret, soma_mask, block_shape=_stats_block
                ).items() if somas
            }
        merge_undersized_streaming(
            ret, int(kwargs.get('min_size_threshold', 0) or 0),
            stats=_stats_island,
            protected=_protect,
            block_shape=_stats_block,
            log=flush_print,
        )

        _stats_final = accumulate_label_statistics(ret, None, block_shape=_stats_block)

        # [PROFILE|CONSERVE] Post-undersize inventory.
        _fg_undersize = int(_stats_final.label_count.sum())
        _nobj_undersize = int(_stats_final.labels.size)
        flush_print(
            f"  [PROFILE|CONSERVE] POST-UNDERSIZE: foreground_pixels={_fg_undersize} "
            f"| objects={_nobj_undersize} | delta_vs_island="
            f"{_fg_undersize - _fg_island}"
        )

        flush_print("  Refining (Filling voids + Relabeling)...")
        # Same result as `relabel_sequential` -- ids 1..N in ascending order of the
        # old id -- but built from the labels already inventoried and applied as one
        # lookup table per block, instead of a whole-image pass.
        _seq = {
            int(old): new
            for new, old in enumerate(_stats_final.labels.tolist(), start=1)
            if int(old) != new
        }
        if _seq:
            apply_label_mapping(ret, _seq, block_shape=_stats_block)

        # [PROFILE|CONSERVE] Final inventory + end-to-end verdict.
        _fg_out = int(_stats_final.label_count.sum())
        _nobj_out = int(_stats_final.labels.size)
        flush_print(
            f"  [PROFILE|CONSERVE] OUTPUT: foreground_pixels={_fg_out} | objects={_nobj_out}"
        )
        flush_print(
            f"  [PROFILE|CONSERVE] END-TO-END: foreground delta={_fg_out - _fg_in} "
            f"(input={_fg_in} -> output={_fg_out})"
            + ("  *** NET FOREGROUND LOSS ***" if _fg_out < _fg_in else "")
        )

        return ret

    finally:
        if os.path.exists(prior_path):
            try:
                os.remove(prior_path)
            except OSError:
                pass
        # Emergency cleanup: remove any remaining .npy files in case of crash
        for i in chunk_data:
            if os.path.exists(chunk_data[i]['path']):
                try:
                    os.remove(chunk_data[i]['path'])
                except Exception:
                    pass
        gc.collect()