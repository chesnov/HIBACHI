"""
Streaming label statistics for step 4's post-stitch passes.

Why this exists
---------------
Everything after the stitch currently runs on the whole label volume in RAM:
``ret = np.array(final_mask)``, then ``find_objects(segmentation)`` in the island
pass and ``np.bincount(segmentation.ravel())`` in the undersize pass. Measured
marginal cost is ~12 bytes per voxel of non-reclaimable memory, which caps step 4
around 0.6 G voxels on a 64 GB machine. A whole brain at 20x is ~4.5e12 voxels.

The observation that makes it tractable: those passes do not need the volume, they
need per-label and per-label-pair aggregates.

  * undersize merge      -> label sizes, and which neighbour each label touches most
  * global merge tests   -> where each interface is, so it can be re-read as a crop
  * intensity references -> per-label and per-interface intensity sums

Two things this module is careful about, both measured rather than assumed:

1.  **Iteration is by 3D block, not by z-slab.** A slab is cheap only while the
    cross-section is small. At a brain cross-section of 24615 x 18462 one plane is
    3.6 GB, so slabs are the wrong shape entirely. A 256^3 block is 134 MB and a
    128^3 block 17 MB, whatever the total volume.

2.  **Aggregates live in numpy arrays, not dicts.** A Python dict keyed by label
    pairs costs ~166 bytes per entry (measured on 2 M entries: 333 MB, against
    31 MB for two int64 arrays). At a million labels with ten neighbours each that
    is 1.7 GB versus 160 MB. Buffers are compacted by sort-and-reduce whenever they
    grow past ``compact_rows``.

Blocks are read with a one-voxel halo on every side, and only pairs whose first
voxel lies in the owned interior are counted, so every adjacent voxel pair in the
volume is visited exactly once -- including pairs straddling a block face.
"""

from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# Forward half of the 26-neighbourhood: enough to see every adjacent pair once.
_FORWARD_OFFSETS_26: List[Tuple[int, int, int]] = [
    (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1),
    (1, -1, -1), (1, -1, 0), (1, -1, 1),
    (1, 0, -1), (1, 0, 0), (1, 0, 1),
    (1, 1, -1), (1, 1, 0), (1, 1, 1),
]

_KEY_STRIDE = 1 << 31          # pair key = lo * stride + hi, stays inside int64
_BIG = 1 << 60


class _Aggregator:
    """
    Array-backed group-by. Rows are buffered as they arrive and periodically
    compacted by sorting on the key and reducing runs, so memory is proportional to
    the number of distinct keys rather than to the number of rows added.
    """

    def __init__(self, n_sum: int, n_box: int, compact_rows: int = 4_000_000) -> None:
        self.n_sum = n_sum
        self.n_box = n_box
        self.compact_rows = compact_rows
        self._keys: List[np.ndarray] = []
        self._sums: List[np.ndarray] = []
        self._mins: List[np.ndarray] = []
        self._maxs: List[np.ndarray] = []
        self._buffered = 0
        self.keys = np.zeros(0, np.int64)
        self.sums = np.zeros((0, n_sum), np.float64)
        self.mins = np.zeros((0, n_box), np.int64)
        self.maxs = np.zeros((0, n_box), np.int64)

    def add(self, keys, sums, mins=None, maxs=None) -> None:
        if keys.size == 0:
            return
        self._keys.append(np.ascontiguousarray(keys, np.int64))
        self._sums.append(np.ascontiguousarray(sums, np.float64).reshape(-1, self.n_sum))
        if self.n_box:
            self._mins.append(np.ascontiguousarray(mins, np.int64).reshape(-1, self.n_box))
            self._maxs.append(np.ascontiguousarray(maxs, np.int64).reshape(-1, self.n_box))
        self._buffered += int(keys.size)
        if self._buffered >= self.compact_rows:
            self.compact()

    def compact(self) -> None:
        if not self._keys:
            return
        keys = np.concatenate([self.keys] + self._keys)
        sums = np.concatenate([self.sums] + self._sums)
        if self.n_box:
            mins = np.concatenate([self.mins] + self._mins)
            maxs = np.concatenate([self.maxs] + self._maxs)
        self._keys, self._sums, self._mins, self._maxs = [], [], [], []
        self._buffered = 0

        order = np.argsort(keys, kind="stable")
        keys_sorted = keys[order]
        uniq, start = np.unique(keys_sorted, return_index=True)
        self.keys = uniq
        self.sums = np.add.reduceat(sums[order], start, axis=0)
        if self.n_box:
            self.mins = np.minimum.reduceat(mins[order], start, axis=0)
            self.maxs = np.maximum.reduceat(maxs[order], start, axis=0)

    def lookup(self, key: int) -> int:
        i = int(np.searchsorted(self.keys, np.int64(key)))
        if i < self.keys.size and int(self.keys[i]) == int(key):
            return i
        return -1


def iter_blocks(
    shape: Sequence[int],
    block_shape: Sequence[int],
) -> Iterator[Tuple[Tuple[slice, ...], Tuple[slice, ...], Tuple[int, ...]]]:
    """
    Yields ``(read_slices, owned_slices_within_read, owned_origin)``.

    ``read_slices`` is the block grown by one voxel on every side and clipped to the
    volume; ``owned_slices_within_read`` locates the block proper inside what was
    read. Owned regions tile the volume exactly once.
    """
    for z0 in range(0, shape[0], block_shape[0]):
        for y0 in range(0, shape[1], block_shape[1]):
            for x0 in range(0, shape[2], block_shape[2]):
                own_start = (z0, y0, x0)
                own_stop = tuple(
                    min(shape[k], own_start[k] + block_shape[k]) for k in range(3)
                )
                read_start = tuple(max(0, own_start[k] - 1) for k in range(3))
                read_stop = tuple(min(shape[k], own_stop[k] + 1) for k in range(3))
                read = tuple(slice(read_start[k], read_stop[k]) for k in range(3))
                inner = tuple(
                    slice(
                        own_start[k] - read_start[k],
                        own_start[k] - read_start[k] + (own_stop[k] - own_start[k]),
                    )
                    for k in range(3)
                )
                yield read, inner, own_start


class LabelStatistics:
    """
    Exact aggregates over a label volume. Arrays, not dicts; the dict view is a
    convenience for tests and small result sets.
    """

    def __init__(self, lab_agg: _Aggregator, pair_agg: _Aggregator) -> None:
        lab_agg.compact()
        pair_agg.compact()
        self._lab = lab_agg
        self._pair = pair_agg
        self.labels = lab_agg.keys
        self.label_count = lab_agg.sums[:, 0].astype(np.int64)
        self.label_intensity_sum = lab_agg.sums[:, 1]
        self.pairs = np.stack(
            [pair_agg.keys // _KEY_STRIDE, pair_agg.keys % _KEY_STRIDE], axis=1
        ) if pair_agg.keys.size else np.zeros((0, 2), np.int64)
        self.pair_contact = pair_agg.sums[:, 0].astype(np.int64)
        self.pair_intensity_sum = pair_agg.sums[:, 1]
        self.pair_bbox_min = pair_agg.mins
        self.pair_bbox_max = pair_agg.maxs

    def count_of(self, label: int) -> int:
        i = self._lab.lookup(label)
        return int(self.label_count[i]) if i >= 0 else 0

    def mean_intensity_of(self, label: int) -> float:
        i = self._lab.lookup(label)
        if i < 0 or self.label_count[i] == 0:
            return 0.0
        return float(self.label_intensity_sum[i] / self.label_count[i])

    def pair_row(self, a: int, b: int) -> int:
        lo, hi = (int(a), int(b)) if a <= b else (int(b), int(a))
        return self._pair.lookup(lo * _KEY_STRIDE + hi)

    def contact_of(self, a: int, b: int) -> int:
        r = self.pair_row(a, b)
        return int(self.pair_contact[r]) if r >= 0 else 0

    def interface_mean_intensity(self, a: int, b: int) -> float:
        r = self.pair_row(a, b)
        if r < 0 or self.pair_contact[r] == 0:
            return 0.0
        # Two intensity samples were summed for each adjacent pair of voxels.
        return float(self.pair_intensity_sum[r] / (2 * self.pair_contact[r]))

    def interface_bbox(self, a: int, b: int, pad: int = 0, shape=None):
        r = self.pair_row(a, b)
        if r < 0:
            return None
        lo, hi = self.pair_bbox_min[r], self.pair_bbox_max[r]
        out = []
        for k in range(3):
            s = int(lo[k]) - pad
            e = int(hi[k]) + 1 + pad
            if shape is not None:
                s, e = max(0, s), min(int(shape[k]), e)
            out.append(slice(max(0, s), e))
        return tuple(out)

    def neighbours_of(self, label: int) -> Dict[int, int]:
        if self.pairs.size == 0:
            return {}
        sel = (self.pairs[:, 0] == label) | (self.pairs[:, 1] == label)
        out: Dict[int, int] = {}
        for (a, b), n in zip(self.pairs[sel], self.pair_contact[sel]):
            other = int(b) if int(a) == int(label) else int(a)
            out[other] = out.get(other, 0) + int(n)
        return out

    def as_dicts(self):
        count = {int(k): int(v) for k, v in zip(self.labels, self.label_count)}
        isum = {int(k): float(v) for k, v in zip(self.labels, self.label_intensity_sum)}
        contact = {(int(a), int(b)): int(n)
                   for (a, b), n in zip(self.pairs, self.pair_contact)}
        ipsum = {(int(a), int(b)): float(n)
                 for (a, b), n in zip(self.pairs, self.pair_intensity_sum)}
        bbox = {(int(a), int(b)): [int(lo[0]), int(hi[0]), int(lo[1]), int(hi[1]),
                                   int(lo[2]), int(hi[2])]
                for (a, b), lo, hi in zip(self.pairs, self.pair_bbox_min,
                                          self.pair_bbox_max)}
        return count, isum, contact, bbox, ipsum

    def memory_bytes(self) -> int:
        return int(sum(a.nbytes for a in (
            self.labels, self.label_count, self.label_intensity_sum, self.pairs,
            self.pair_contact, self.pair_intensity_sum,
            self.pair_bbox_min, self.pair_bbox_max)))


def accumulate_label_statistics(
    labels,
    intensity=None,
    block_shape: Sequence[int] = (128, 128, 128),
    compact_rows: int = 4_000_000,
) -> LabelStatistics:
    """
    One bounded-memory sweep over ``labels`` (memmap or array).

    Peak memory is one padded block of labels plus one of intensity, plus the
    aggregates. Nothing scales with the size of the volume.
    """
    shape = tuple(int(s) for s in labels.shape)
    block_shape = tuple(max(1, int(b)) for b in block_shape)

    lab_agg = _Aggregator(n_sum=2, n_box=0, compact_rows=compact_rows)
    pair_agg = _Aggregator(n_sum=2, n_box=3, compact_rows=compact_rows)

    for read, inner, origin in iter_blocks(shape, block_shape):
        lab = np.asarray(labels[read])
        own = lab[inner]
        nz = own > 0
        if not nz.any():
            continue

        inten = np.asarray(intensity[read]) if intensity is not None else None

        # ---- per-label size and intensity ---------------------------------
        uniq, inv = np.unique(own[nz], return_inverse=True)
        inv = inv.reshape(-1)
        cnt = np.bincount(inv, minlength=uniq.size).astype(np.float64)
        if inten is not None:
            w = inten[inner][nz].astype(np.float64)
            isum = np.bincount(inv, weights=w, minlength=uniq.size)
        else:
            isum = np.zeros(uniq.size, np.float64)
        lab_agg.add(uniq.astype(np.int64), np.stack([cnt, isum], axis=1))

        # ---- per-pair adjacency ------------------------------------------
        for d in _FORWARD_OFFSETS_26:
            a_start, length, ok = [], [], True
            for k in range(3):
                a0 = inner[k].start
                ln = inner[k].stop - inner[k].start
                b0 = a0 + d[k]
                if b0 < 0:                       # no halo here: volume edge
                    a0 -= b0
                    ln += b0
                    b0 = 0
                over = (b0 + ln) - lab.shape[k]
                if over > 0:
                    ln -= over
                if ln <= 0:
                    ok = False
                    break
                a_start.append(a0)
                length.append(ln)
            if not ok:
                continue

            a_sl = tuple(slice(a_start[k], a_start[k] + length[k]) for k in range(3))
            b_sl = tuple(slice(a_start[k] + d[k], a_start[k] + d[k] + length[k])
                         for k in range(3))
            A, B = lab[a_sl], lab[b_sl]
            hit = (A > 0) & (B > 0) & (A != B)
            if not hit.any():
                continue

            a_v = A[hit].astype(np.int64)
            b_v = B[hit].astype(np.int64)
            keys = np.minimum(a_v, b_v) * _KEY_STRIDE + np.maximum(a_v, b_v)

            k_uniq, k_inv = np.unique(keys, return_inverse=True)
            k_inv = k_inv.reshape(-1)
            k_cnt = np.bincount(k_inv, minlength=k_uniq.size).astype(np.float64)
            if inten is not None:
                w = (inten[a_sl][hit].astype(np.float64)
                     + inten[b_sl][hit].astype(np.float64))
                k_isum = np.bincount(k_inv, weights=w, minlength=k_uniq.size)
            else:
                k_isum = np.zeros(k_uniq.size, np.float64)

            # Interface bounding box in GLOBAL coordinates, so the pass that runs
            # the merge tests can re-read exactly this crop and nothing more.
            coords = np.nonzero(hit)
            g = np.stack(
                [coords[k].astype(np.int64) + (a_start[k] + origin[k] - inner[k].start)
                 for k in range(3)],
                axis=1,
            )
            mins = np.full((k_uniq.size, 3), _BIG, np.int64)
            maxs = np.full((k_uniq.size, 3), -1, np.int64)
            np.minimum.at(mins, k_inv, g)
            np.maximum.at(maxs, k_inv, g)

            pair_agg.add(k_uniq, np.stack([k_cnt, k_isum], axis=1), mins, maxs)

    return LabelStatistics(lab_agg, pair_agg)
