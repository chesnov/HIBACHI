"""
Post-stitch resolution of deferred interfaces.

The problem
-----------
The chunk worker refuses to score an interface when neither side has a soma in
its crop, because the boundary there is placed between inherited marker regions
rather than against the somata, and scoring it produces verdicts that then
propagate through the stitcher's seed unioning. That refusal is correct in
isolation, but it leaves one real gap: a single cell that received two seeds
which landed either side of a chunk seam is split, and no lever can rejoin it,
because the interface between its two pieces is never scored anywhere.

The fix
-------
Resolve those interfaces once, after stitching, when the labels are whole. Every
quantity the merge tests need is an additive statistic over the interface and
over each basin within `local_analysis_radius` of it, so the sweep can proceed in
fixed-size tiles and hold one tile at a time. Memory is set by the tile, not by
the object, the interface area, or the distance between somata -- all three of
which are unbounded in real data.

Scope and blast radius
----------------------
This pass only ever converts a deferred KEEP into a MERGE. It does not look at
interfaces the worker already scored, and it cannot turn a merge back into a
split. A wrong decision here costs exactly one wrong boundary between two final
labels: merges are applied as a direct relabel, with no seed-set unioning, so
nothing propagates to a third label. Every decision is logged with the full
statistics behind it.
"""

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import ndimage

try:
    from .interface_metrics import InterfaceStats, decide_merge, format_decision
except ImportError:
    from interface_metrics import InterfaceStats, decide_merge, format_decision


class DeferredInterface:
    """One interface the worker declined to score, in whole-image coordinates."""

    __slots__ = ("label_a", "label_b", "seeds_a", "seeds_b", "bbox")

    def __init__(self, label_a: int, label_b: int,
                 seeds_a: Set[int], seeds_b: Set[int],
                 bbox: Tuple[Tuple[int, int], ...]):
        self.label_a = int(label_a)
        self.label_b = int(label_b)
        self.seeds_a = set(int(s) for s in seeds_a)
        self.seeds_b = set(int(s) for s in seeds_b)
        self.bbox = bbox

    def __repr__(self) -> str:
        return (f"DeferredInterface({self.label_a},{self.label_b},"
                f"bbox={self.bbox})")


class _Union:
    """Minimal union-find over label ids, smallest id wins as the root."""

    def __init__(self) -> None:
        self.p: Dict[int, int] = {}

    def find(self, x: int) -> int:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def _iter_tiles(bbox, shape, tile, halo):
    """
    Yield (target_slices, padded_slices, offset_of_target_within_pad).

    `target` regions tile the bbox without overlapping, so statistics summed over
    them count every voxel exactly once. `pad` extends by `halo` so the dilations
    that define the interface and its neighbourhood are complete at tile edges.
    """
    ndim = len(shape)
    starts = [range(b[0], b[1], t) for b, t in zip(bbox, tile)]

    def rec(i, acc):
        if i == ndim:
            tgt = tuple(acc)
            pad = tuple(
                slice(max(0, s.start - halo), min(shape[d], s.stop + halo))
                for d, s in enumerate(tgt)
            )
            off = tuple(s.start - p.start for s, p in zip(tgt, pad))
            yield tgt, pad, off
            return
        for s in starts[i]:
            e = min(s + tile[i], bbox[i][1])
            yield from rec(i + 1, acc + [slice(s, e)])

    yield from rec(0, [])


def _footprints(ndim: int, radius: int):
    from skimage.morphology import ball, disk, footprint_rectangle
    adj = footprint_rectangle((3,) * ndim)
    if radius > 1:
        zone = ball(radius) if ndim == 3 else disk(radius)
    else:
        zone = adj
    return adj, zone


def resolve_deferred_interfaces(
    segmentation: np.ndarray,
    intensity: np.ndarray,
    deferred: Sequence[DeferredInterface],
    stitch_label_map: Optional[Dict[int, int]] = None,
    soma_intensities: Optional[Dict[int, float]] = None,
    soma_centroids: Optional[Dict[int, np.ndarray]] = None,
    local_analysis_radius: int = 10,
    min_local_intensity_difference: float = 0.0,
    min_path_intensity_ratio: float = 1.0,
    max_interface_to_zone_ratio: float = 0.85,
    max_seed_centroid_dist: float = 0.0,
    search_margin: int = 0,
    tile_shape: Optional[Tuple[int, ...]] = None,
    max_rounds: int = 3,
    log=print,
) -> np.ndarray:
    """
    Score every deferred interface and merge the ones the tests judge to be one
    cell. Returns `segmentation`, modified in place.

    `stitch_label_map` is the stitcher's label remapping; recorded ids are
    resolved through it before use, since the labels the worker emitted may have
    been joined across seams since.

    `search_margin` widens the swept region beyond the recorded interface. It has
    to be at least the chunk overlap: the recorded box is where the boundary sat
    when the worker deferred it, and stitching can move a boundary anywhere within
    the overlap before this pass runs. Sweeping the bare recorded box then finds
    no interface at all and the pair is silently skipped. The region is widened by
    `local_analysis_radius` on top of that, because the neighbourhood the contrast
    test reads extends that far past the interface itself.
    """
    ndim = segmentation.ndim
    if tile_shape is None:
        tile_shape = (64, 256, 256)[-ndim:]
    halo = int(local_analysis_radius) + 1
    adj_fp, zone_fp = _footprints(ndim, int(local_analysis_radius))
    from skimage.morphology import binary_dilation

    soma_intensities = soma_intensities or {}
    soma_centroids = soma_centroids or {}

    def _root(lbl: int) -> int:
        if not stitch_label_map:
            return lbl
        seen = 0
        while lbl in stitch_label_map and stitch_label_map[lbl] != lbl and seen < 64:
            lbl = stitch_label_map[lbl]
            seen += 1
        return lbl

    if not deferred:
        log("  [Refine] No deferred interfaces to resolve.")
        return segmentation

    log(f"  [Refine] Resolving deferred interfaces: {len(deferred)} recorded.")

    uf = _Union()
    n_merged_total = 0

    for rnd in range(max_rounds):
        # ---- collect the pairs still worth looking at ----------------------
        pairs: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for d in deferred:
            a, b = uf.find(_root(d.label_a)), uf.find(_root(d.label_b))
            if a == b:
                continue                       # already one label
            key = (min(a, b), max(a, b))
            e = pairs.setdefault(key, {"seeds": set(), "bbox": None})
            e["seeds"] |= d.seeds_a | d.seeds_b
            e["bbox"] = d.bbox if e["bbox"] is None else tuple(
                (min(p[0], q[0]), max(p[1], q[1])) for p, q in zip(e["bbox"], d.bbox)
            )

        # distance bound: never reconsider somata further apart than the user's
        # merge distance, exactly as the in-chunk path does.
        if max_seed_centroid_dist > 0 and soma_centroids:
            drop = []
            for key, e in pairs.items():
                cs = [soma_centroids[s] for s in e["seeds"] if s in soma_centroids]
                if len(cs) >= 2:
                    dmin = min(
                        float(np.linalg.norm(np.asarray(p) - np.asarray(q)))
                        for i, p in enumerate(cs) for q in cs[i + 1:]
                    )
                    if dmin > max_seed_centroid_dist:
                        drop.append(key)
            for k in drop:
                del pairs[k]
            if drop:
                log(f"    [PROFILE|DEFERRED] round {rnd}: {len(drop)} pair(s) "
                    f"beyond max_seed_centroid_dist, not reconsidered")

        if not pairs:
            break

        # union of all bboxes -> the region the sweep has to visit
        union_bbox = None
        for e in pairs.values():
            union_bbox = e["bbox"] if union_bbox is None else tuple(
                (min(p[0], q[0]), max(p[1], q[1]))
                for p, q in zip(union_bbox, e["bbox"])
            )
        # Widen: the neighbourhood reaches `halo` past the interface, and the
        # boundary itself may have moved by up to `search_margin` since it was
        # recorded. Without this the sweep can miss the interface entirely.
        grow = int(local_analysis_radius) + int(search_margin) + 1
        union_bbox = tuple(
            (max(0, lo - grow), min(segmentation.shape[d], hi + grow))
            for d, (lo, hi) in enumerate(union_bbox)
        )

        n_tiles = 1
        for (lo, hi), t in zip(union_bbox, tile_shape):
            n_tiles *= max(1, -(-(hi - lo) // t))
        log(f"    [PROFILE|DEFERRED] round {rnd}: {len(pairs)} pair(s), "
            f"sweeping {n_tiles} tile(s) of {tile_shape}")

        # ---- one streaming sweep, all pairs at once ------------------------
        acc: Dict[Tuple[int, int], InterfaceStats] = {
            k: InterfaceStats() for k in pairs
        }
        want = {}
        for (a, b) in pairs:
            want.setdefault(a, set()).add((a, b))
            want.setdefault(b, set()).add((a, b))

        for tgt, pad, off in _iter_tiles(union_bbox, segmentation.shape,
                                         tile_shape, halo):
            lab = np.asarray(segmentation[pad])
            present = set(int(v) for v in np.unique(lab) if v > 0)
            if stitch_label_map or uf.p:
                # labels on disk are pre-merge ids; map them to current roots
                remap = {v: uf.find(_root(v)) for v in present}
                if any(k != v for k, v in remap.items()):
                    out = lab.copy()
                    for k, v in remap.items():
                        if k != v:
                            out[lab == k] = v
                    lab = out
                    present = set(remap.values())

            active = set()
            for v in present:
                active |= want.get(v, set())
            active = {k for k in active if k[0] in present and k[1] in present}
            if not active:
                continue

            inten = np.asarray(intensity[pad])
            tmask = np.zeros(lab.shape, bool)
            tmask[tuple(slice(o, o + (s.stop - s.start))
                        for o, s in zip(off, tgt))] = True

            for key in active:
                a, b = key
                ma, mb = lab == a, lab == b
                iface = binary_dilation(ma, footprint=adj_fp) & mb
                if not iface.any():
                    continue
                zone = binary_dilation(iface, footprint=zone_fp)
                acc[key].add(inten, iface & tmask,
                             (zone & ma) & tmask, (zone & mb) & tmask)

        # ---- decide ---------------------------------------------------------
        merged_this_round = 0
        for key, st in acc.items():
            a, b = key
            seeds = pairs[key]["seeds"]
            ints = [soma_intensities[s] for s in seeds if s in soma_intensities]
            soma_ref = float(np.mean(ints)) if ints else 1.0

            m = decide_merge(
                st, soma_ref,
                min_local_intensity_difference,
                min_path_intensity_ratio,
                max_interface_to_zone_ratio,
            )
            log(f"    [PROFILE|DEFERRED] pair=({a},{b}) | "
                + format_decision(m, min_path_intensity_ratio,
                                  max_interface_to_zone_ratio))
            if m["should_merge_decision"]:
                uf.union(a, b)
                merged_this_round += 1

        n_merged_total += merged_this_round
        log(f"    [PROFILE|DEFERRED] round {rnd}: merged {merged_this_round}")
        if merged_this_round == 0:
            break

    # ---- apply --------------------------------------------------------------
    if n_merged_total:
        fg_before = int(np.count_nonzero(segmentation))
        vals = np.unique(segmentation)
        remap = {int(v): uf.find(int(v)) for v in vals if v > 0}
        changed = {k: v for k, v in remap.items() if k != v}
        for k, v in changed.items():
            segmentation[segmentation == k] = v
        fg_after = int(np.count_nonzero(segmentation))
        log(f"  [PROFILE|DEFERRED|SUMMARY] merges={n_merged_total} | "
            f"labels_remapped={len(changed)} | foreground {fg_before} -> "
            f"{fg_after} "
            + ("(conserved)" if fg_after == fg_before
               else "*** FOREGROUND CHANGED -- UNEXPECTED ***"))
    else:
        log("  [PROFILE|DEFERRED|SUMMARY] merges=0 (no deferred interface "
            "judged to be one cell)")

    return segmentation
