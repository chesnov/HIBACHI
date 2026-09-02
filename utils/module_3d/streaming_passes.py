"""
Post-stitch passes rebuilt on the streaming aggregates.

Two passes live here.

``merge_undersized_streaming`` replaces the size floor. The existing
``_merge_undersized_cells`` calls ``np.bincount(segmentation.ravel())`` and
``find_objects(segmentation)`` on the whole label volume, then dilates a crop per
undersized label. All of that information is already in the aggregates: a label's
size, and how much of its surface touches each neighbour. So the pass becomes a
graph operation with no volume access at all, and the volume is touched exactly
once at the end, by a lookup table applied block by block.

``global_merge_pass`` is the piece the propagation layer needs. Merge decisions
cannot safely be taken inside a chunk that only inherited its markers -- measured
consequence: the boundary lands in bright tissue, the tests correctly call it a
bright cut, and because merging unions the seed sets that single verdict fuses two
separate cells across the whole volume. Here, after stitching, every interface
between two final labels is visible exactly once, with both cells whole. That is
the only place ``ref_intensity`` and ``interface vs cell mean`` mean what they were
designed to mean, and it is where two spurious seeds on one real cell get rejoined
regardless of which chunks their somata landed in.

Because that is now the only place long-range merges happen, it is also where
``max_seed_centroid_dist`` has to be enforced -- see ``global_merge_pass``.

Both passes read the volume only in crops sized by the interface, never in full.

Both work in 2D and 3D: dimensionality comes from the array passed in, and
``block_shape`` is normalised to match, so the 2D and 3D splitting modules share
one copy of this module rather than keeping two in step by hand.
"""

from typing import Dict, Iterable, Optional, Sequence, Set

import numpy as np

try:
    from .streaming_stats import (
        _KEY_STRIDE,
        _Aggregator,
        LabelStatistics,
        accumulate_label_statistics,
        iter_blocks,
        normalise_block_shape,
    )
except ImportError:  # pragma: no cover - direct script execution
    from streaming_stats import (
        _KEY_STRIDE,
        _Aggregator,
        LabelStatistics,
        accumulate_label_statistics,
        iter_blocks,
        normalise_block_shape,
    )


# =============================================================================
# Shared helpers
# =============================================================================

class _UnionFind:
    """Merges between final labels. The smallest id in a group always survives."""

    def __init__(self) -> None:
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent.get(x, x) != x:
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        keep, drop = min(ra, rb), max(ra, rb)
        self.parent[drop] = keep
        return keep

    def mapping(self) -> Dict[int, int]:
        return {k: self.find(k) for k in self.parent}


def apply_label_mapping(
    labels,
    mapping: Dict[int, int],
    block_shape: Sequence[int] = (128, 128, 128),
) -> int:
    """
    Apply {old label -> new label} to a label volume in place, block by block via a
    lookup table. Returns the number of voxels changed.

    A per-label ``arr[arr == old] = new`` loop would sweep the volume once per
    label, which is the thing that makes a densely over-seeded mask crawl. One LUT
    pass costs one read and one write of each block.
    """
    if not mapping:
        return 0
    hi = max(max(mapping), max(mapping.values()))
    lut = np.arange(hi + 1, dtype=np.int64)
    for old, new in mapping.items():
        lut[int(old)] = int(new)

    ndim = len(labels.shape)
    block_shape = normalise_block_shape(block_shape, ndim)
    changed = 0
    for read, inner, _origin in iter_blocks(tuple(labels.shape), block_shape):
        own_sl = tuple(
            slice(read[k].start + inner[k].start, read[k].start + inner[k].stop)
            for k in range(ndim)
        )
        block = np.asarray(labels[own_sl])
        if block.size == 0:
            continue
        safe = block <= hi
        if not safe.any():
            continue
        out = block.copy()
        out[safe] = lut[block[safe]].astype(block.dtype)
        diff = int(np.count_nonzero(out != block))
        if diff:
            labels[own_sl] = out
            changed += diff
    return changed


def accumulate_label_soma_map(
    labels,
    soma_mask,
    block_shape: Sequence[int] = (128, 128, 128),
) -> Dict[int, Set[int]]:
    """
    {final label -> set of soma ids inside it}, in one bounded-memory sweep.

    Needed because ``ref_intensity`` in the merge tests is the mean intensity of the
    somata of the two cells involved, and after stitching a label's somata may sit
    anywhere in the volume. ``global_merge_pass`` also uses it to look up where
    those somata are, for the merge-distance bound.
    """
    ndim = len(labels.shape)
    block_shape = normalise_block_shape(block_shape, ndim)
    agg = _Aggregator(n_sum=1, n_box=0)
    for read, inner, _origin in iter_blocks(tuple(labels.shape), block_shape):
        own_sl = tuple(
            slice(read[k].start + inner[k].start, read[k].start + inner[k].stop)
            for k in range(ndim)
        )
        lab = np.asarray(labels[own_sl])
        som = np.asarray(soma_mask[own_sl])
        hit = (lab > 0) & (som > 0)
        if not hit.any():
            continue
        keys = lab[hit].astype(np.int64) * _KEY_STRIDE + som[hit].astype(np.int64)
        u, inv = np.unique(keys, return_inverse=True)
        cnt = np.bincount(inv.reshape(-1), minlength=u.size).astype(np.float64)
        agg.add(u, cnt.reshape(-1, 1))
    agg.compact()
    out: Dict[int, Set[int]] = {}
    for k in agg.keys.tolist():
        lbl, soma = int(k) // _KEY_STRIDE, int(k) % _KEY_STRIDE
        out.setdefault(lbl, set()).add(soma)
    return out


def _min_soma_separation(
    somas_A: Iterable[int],
    somas_B: Iterable[int],
    soma_centroids: Optional[Dict[int, np.ndarray]],
) -> float:
    """
    Smallest physical distance between a soma on side A and a soma on side B.

    Centroids are whole-image and in physical units, so a soma anywhere in the
    volume is measured correctly -- which is the point, because after stitching a
    label's somata can be nowhere near the interface being judged.

    Returns 0.0 whenever the distance cannot be established (no table, or neither
    side has a soma present in it), so a pair is never gated on missing data. The
    MINIMUM across the two sides, not the maximum: a label can own several somata,
    and the question is whether ANY pair across this boundary is close enough to
    plausibly belong to one cell.
    """
    somas_A = list(somas_A)
    somas_B = list(somas_B)
    if not soma_centroids or not somas_A or not somas_B:
        return 0.0
    ca = [soma_centroids[s] for s in somas_A if s in soma_centroids]
    cb = [soma_centroids[s] for s in somas_B if s in soma_centroids]
    if not ca or not cb:
        return 0.0
    return float(
        min(np.linalg.norm(np.asarray(p) - np.asarray(q)) for p in ca for q in cb)
    )


# =============================================================================
# Pass 1: size floor, as a graph operation
# =============================================================================

def merge_undersized_streaming(
    labels,
    min_size_threshold: int,
    stats: Optional[LabelStatistics] = None,
    max_rounds: int = 10,
    block_shape: Sequence[int] = (128, 128, 128),
    protected: Optional[Iterable[int]] = None,
    log=print,
) -> Dict[int, int]:
    """
    Merge every label below ``min_size_threshold`` into the neighbour it touches
    most, repeatedly, until nothing is left to merge or ``max_rounds`` is reached.

    Semantics kept from ``_merge_undersized_cells``: nothing is ever deleted, a
    label with no neighbour at all is kept whatever its size, and merging is
    iterated because a merged label can still be undersized.

    ``protected`` labels are never merged away. Passing the labels that own a soma
    prevents the failure in the reported trace, where a 7,893-voxel cell that had its
    own soma was merged into its neighbour purely for being small.

    Returns the mapping actually applied.
    """
    if min_size_threshold is None or min_size_threshold <= 0:
        return {}
    if stats is None:
        stats = accumulate_label_statistics(labels, None, block_shape=block_shape)

    size = {int(l): int(c) for l, c in zip(stats.labels, stats.label_count)}
    contact: Dict[int, Dict[int, int]] = {}
    for (a, b), n in zip(stats.pairs, stats.pair_contact):
        a, b, n = int(a), int(b), int(n)
        contact.setdefault(a, {})[b] = contact.setdefault(a, {}).get(b, 0) + n
        contact.setdefault(b, {})[a] = contact.setdefault(b, {}).get(a, 0) + n

    guard = {int(p) for p in (protected or ())}
    uf = _UnionFind()
    n_merged = px_merged = n_kept = px_kept = 0
    rounds_used = 0

    for rnd in range(max_rounds):
        rounds_used = rnd + 1
        small = sorted(
            (l for l, s in size.items() if s < min_size_threshold and l not in guard),
            key=lambda l: (size[l], l),
        )
        if not small:
            break
        progressed = False
        for lbl in small:
            if size.get(lbl, 0) >= min_size_threshold or lbl not in size:
                continue
            nb = {k: v for k, v in contact.get(lbl, {}).items() if k in size and k != lbl}
            if not nb:
                n_kept += 1
                px_kept += size[lbl]
                log(f"    [PROFILE|UNDERSIZE] label={lbl} size={size[lbl]} "
                    f"< {min_size_threshold} -> KEEP (no neighbouring cell to merge into)")
                guard.add(lbl)
                continue
            best = max(sorted(nb), key=lambda k: nb[k])
            keep = uf.union(lbl, best)
            gone = best if keep == lbl else lbl
            log(f"    [PROFILE|UNDERSIZE] label={gone} size={size[gone]} "
                f"< {min_size_threshold} -> MERGE into {keep} "
                f"(contact={nb[best]} voxels)")
            n_merged += 1
            px_merged += size[gone]
            # Fold the absorbed label's size and contacts into the survivor.
            size[keep] = size.pop(lbl) + size.pop(best) if keep in (lbl, best) else size[keep]
            merged_contacts = contact.pop(gone, {})
            tgt = contact.setdefault(keep, {})
            for k, v in merged_contacts.items():
                if k == keep:
                    continue
                tgt[k] = tgt.get(k, 0) + v
                contact.setdefault(k, {}).pop(gone, None)
                contact[k][keep] = contact[k].get(keep, 0) + v
            tgt.pop(gone, None)
            progressed = True
        if not progressed:
            break

    log(f"  [PROFILE|UNDERSIZE|SUMMARY] merged={n_merged} (voxels={px_merged}) | "
        f"kept_small_isolated={n_kept} (voxels={px_kept}) | rounds_used={rounds_used}")

    mapping = {k: v for k, v in uf.mapping().items() if k != v}
    if mapping:
        apply_label_mapping(labels, mapping, block_shape=block_shape)
    return mapping


# =============================================================================
# Pass 2: the merge tests, once, on the assembled volume
# =============================================================================

def global_merge_pass(
    labels,
    intensity,
    soma_mask,
    spacing,
    interface_metric_fn,
    stats: Optional[LabelStatistics] = None,
    global_soma_intensities: Optional[Dict[int, float]] = None,
    global_soma_centroids: Optional[Dict[int, np.ndarray]] = None,
    max_seed_centroid_dist: float = 0.0,
    min_contact: int = 1,
    block_shape: Sequence[int] = (128, 128, 128),
    log=print,
    **params,
) -> Dict[int, int]:
    """
    Run the interface merge tests once per adjacent pair of final labels, reading
    only each interface's own crop.

    ``interface_metric_fn`` is ``_calculate_interface_metrics`` from the splitting
    module, called with exactly the arguments it takes today. ``cell_mean_intensity``
    and ``ref_intensity`` come from the streaming aggregates, so they describe the
    whole cells rather than whatever happened to fall inside one chunk.

    ``max_seed_centroid_dist`` (um, 0 disables) is the upper bound on how far apart
    two somata can be and still be considered for merging. It is enforced here and
    not only in the worker: with ``require_local_somas`` active the worker refuses
    to score a propagated interface at all, so this pass is where every long-range
    merge decision is actually taken, and therefore where an unbounded
    ``min_path_intensity_ratio`` could otherwise fuse two distant cells. The check
    runs before the interface crop is read, so it costs nothing when it does not
    fire.

    Returns the mapping applied.
    """
    if stats is None:
        stats = accumulate_label_statistics(labels, intensity, block_shape=block_shape)
    label_somas = accumulate_label_soma_map(labels, soma_mask, block_shape=block_shape)
    soma_int = global_soma_intensities or {}
    soma_cen = global_soma_centroids or {}
    max_sep = float(max_seed_centroid_dist or 0.0)

    radius = int(params.get('local_analysis_radius', 10))
    pad = radius + 2
    shape = tuple(int(s) for s in labels.shape)
    uf = _UnionFind()

    order = np.argsort(-stats.pair_contact)
    n_tested = n_merged = n_far = 0

    for row in order.tolist():
        a, b = int(stats.pairs[row, 0]), int(stats.pairs[row, 1])
        if int(stats.pair_contact[row]) < min_contact:
            continue
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue

        # Max Seed Merge Distance. Two cell bodies further apart than this cannot
        # belong to one cell however unconvincing the boundary between them looks,
        # so the intensity tests are not consulted and the cut stands.
        if max_sep > 0:
            sep = _min_soma_separation(
                sorted(label_somas.get(a, set())),
                sorted(label_somas.get(b, set())),
                soma_cen,
            )
            if sep > max_sep:
                n_far += 1
                log(f"    [PROFILE|GLOBALMERGE] KEEP {a}+{b} | "
                    f"soma_separation={sep:.1f}um > {max_sep:.1f}um (not scored)")
                continue

        crop = stats.interface_bbox(a, b, pad=pad, shape=shape)
        if crop is None:
            continue
        lab = np.asarray(labels[crop])
        inten = np.asarray(intensity[crop]).astype(np.float32)
        mask_A = lab == a
        mask_B = lab == b
        if not mask_A.any() or not mask_B.any():
            continue
        cell_mask = mask_A | mask_B

        # Whole-cell quantities from the aggregates, not from the crop.
        na, nb_ = stats.count_of(a), stats.count_of(b)
        sa = float(stats.label_intensity_sum[stats._lab.lookup(a)]) if na else 0.0
        sb = float(stats.label_intensity_sum[stats._lab.lookup(b)]) if nb_ else 0.0
        cell_mean = (sa + sb) / max(1, na + nb_)

        refs = [
            soma_int[s]
            for s in sorted(label_somas.get(a, set()) | label_somas.get(b, set()))
            if s in soma_int
        ]
        ref_intensity = float(np.mean(refs)) if refs else 1.0

        metrics = interface_metric_fn(
            mask_A, mask_B, cell_mask, inten,
            ref_intensity, cell_mean, spacing,
            radius,
            params.get('min_local_intensity_difference', 0.0),
            params.get('min_path_intensity_ratio', 1.0),
            params.get('max_interface_to_cell_mean_ratio', 0.85),
        )
        n_tested += 1
        if metrics.get('should_merge_decision'):
            uf.union(a, b)
            n_merged += 1
            log(f"    [PROFILE|GLOBALMERGE] MERGE {a}+{b} "
                f"(contact={int(stats.pair_contact[row])})")

    log(f"  [PROFILE|GLOBALMERGE|SUMMARY] interfaces_tested={n_tested} | "
        f"merged={n_merged} | beyond_max_merge_distance={n_far}")

    mapping = {k: v for k, v in uf.mapping().items() if k != v}
    if mapping:
        apply_label_mapping(labels, mapping, block_shape=block_shape)
    return mapping
