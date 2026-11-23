import heapq

import numba
import numpy as np
from scipy import ndimage

try:
    from .elongated_somas import SEED_UM, _anchors, _prepare
except ImportError:
    from elongated_somas import SEED_UM, _anchors, _prepare

EPS = 0.05
CONF = 0.5


@numba.njit(cache=True)
def _march_f(idx, pos, cost, offs, steps, unit, src, slab, sst, h0, b0, bri, sig_a, sig_b, mem, kk):
    n = len(cost)
    lab_f = np.full((n, kk), -1, np.int64)
    dist_f = np.full((n, kk), np.inf)
    h_f = np.zeros((n, kk, 3), np.float32)
    b_f = np.zeros((n, kk), np.float32)
    lab_t = np.full((n, kk), -1, np.int64)
    dist_t = np.full((n, kk), np.inf)
    h_t = np.zeros((n, kk, 3), np.float32)
    b_t = np.zeros((n, kk), np.float32)
    done = np.zeros(n, np.int64)
    heap = [(0.0, np.int64(0), np.int64(0))]
    heap.pop()
    for i in range(len(src)):
        a = src[i]
        lab_t[a, 0] = slab[i]
        dist_t[a, 0] = 0.0
        h_t[a, 0] = h0[sst[i]]
        b_t[a, 0] = b0[sst[i]]
        heap.append((0.0, a, slab[i]))
    heapq.heapify(heap)
    sz, sy, sx = idx.shape
    while len(heap):
        dc, a, lab = heapq.heappop(heap)
        if done[a] == kk:
            continue
        j0 = -1
        for j in range(kk):
            if lab_t[a, j] == lab and dist_t[a, j] == dc:
                j0 = j
        if j0 < 0:
            continue
        seen = False
        for j in range(done[a]):
            if lab_f[a, j] == lab:
                seen = True
        if seen:
            continue
        f = done[a]
        lab_f[a, f] = lab
        dist_f[a, f] = dc
        h_f[a, f] = h_t[a, j0]
        b_f[a, f] = b_t[a, j0]
        done[a] += 1
        for k in range(offs.shape[0]):
            z = pos[a, 0] + offs[k, 0]
            y = pos[a, 1] + offs[k, 1]
            x = pos[a, 2] + offs[k, 2]
            if z < 0 or y < 0 or x < 0 or z >= sz or y >= sy or x >= sx:
                continue
            b = idx[z, y, x]
            if b < 0 or done[b] == kk:
                continue
            seen = False
            for j in range(done[b]):
                if lab_f[b, j] == lab:
                    seen = True
            if seen:
                continue
            c = unit[k, 0] * h_f[a, f, 0] + unit[k, 1] * h_f[a, f, 1] + unit[k, 2] * h_f[a, f, 2]
            ang = np.arccos(min(1.0, abs(c))) / sig_a
            db = (bri[b] - b_f[a, f]) / sig_b
            nd = dc + steps[k] * ((cost[a] + cost[b]) * 0.5 + 0.5 * ang * ang + 0.5 * db * db)
            slot = -1
            worst = 0
            for j in range(kk):
                if lab_t[b, j] == lab:
                    slot = j
                if dist_t[b, j] > dist_t[b, worst]:
                    worst = j
            if slot < 0:
                if nd >= dist_t[b, worst]:
                    continue
                slot = worst
                lab_t[b, slot] = lab
            elif nd >= dist_t[b, slot]:
                continue
            dist_t[b, slot] = nd
            g = min(1.0, steps[k] / mem)
            sg = 1.0 if c >= 0 else -1.0
            hx = h_f[a, f, 0] * (1 - g) + sg * unit[k, 0] * g
            hy = h_f[a, f, 1] * (1 - g) + sg * unit[k, 1] * g
            hz = h_f[a, f, 2] * (1 - g) + sg * unit[k, 2] * g
            hn = np.sqrt(hx * hx + hy * hy + hz * hz)
            h_t[b, slot, 0] = hx / hn
            h_t[b, slot, 1] = hy / hn
            h_t[b, slot, 2] = hz / hn
            b_t[b, slot] = b_f[a, f] + g * (bri[b] - b_f[a, f])
            heapq.heappush(heap, (nd, np.int64(b), lab))
    return lab_f, dist_f


def _vote(t, region, anchors, seedlab, sp, bri, ldir):
    pos = np.argwhere(region).astype(np.int64)
    idx = -np.ones(region.shape, np.int32)
    idx[region] = np.arange(len(pos), dtype=np.int32)
    cost = (1.0 / (EPS + t / max(float(np.percentile(t, 99)), 1e-6))).astype(np.float64)
    offs = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)], np.int64)
    steps = np.linalg.norm(offs * sp, axis=1)
    unit = offs * sp / steps[:, None]
    nodes = []
    for vol in (seedlab, np.where(seedlab > 0, 0, anchors)):
        pts = np.argwhere(vol > 0)
        lab = vol[tuple(pts.T)]
        o = np.argsort(lab, kind="stable")
        groups = np.split(o, np.flatnonzero(np.diff(lab[o])) + 1) if len(o) else []
        nodes.append([(int(lab[g[0]]), idx[tuple(pts[g].T)].astype(np.int64)) for g in groups])
    seeds, pieces = nodes
    src_nodes = [nd for _, nd in seeds] + [nd for _, nd in pieces]
    h0 = np.zeros((len(src_nodes), 3))
    b0 = np.zeros(len(src_nodes))
    for i, nd in enumerate(src_nodes):
        if len(nd) >= 3:
            xk = pos[nd] * sp
            h0[i] = np.linalg.svd(xk - xk.mean(0), full_matrices=False)[2][0]
        else:
            d = ldir[nd[0]] * sp
            h0[i] = d / max(np.linalg.norm(d), 1e-12)
        b0[i] = np.median(bri[nd])
    sig_b = max(float(np.median(np.concatenate([np.abs(bri[nd] - b0[i]) for i, (_, nd) in enumerate(seeds)]))) / 0.6745, 1e-6)
    look = -np.ones(int(seedlab.max()) + 1, np.int64)
    look[[k for k, _ in seeds]] = np.arange(len(seeds))
    ang = []
    for k in range(13):
        sh = seedlab[max(0, -offs[k, 0]):seedlab.shape[0] - max(0, offs[k, 0]), max(0, -offs[k, 1]):seedlab.shape[1] - max(0, offs[k, 1]), max(0, -offs[k, 2]):seedlab.shape[2] - max(0, offs[k, 2])]
        nb = seedlab[max(0, offs[k, 0]):seedlab.shape[0] + min(0, offs[k, 0]), max(0, offs[k, 1]):seedlab.shape[1] + min(0, offs[k, 1]), max(0, offs[k, 2]):seedlab.shape[2] + min(0, offs[k, 2])]
        q = look[sh[(sh > 0) & (sh == nb)]]
        ang.append(np.abs(h0[q] @ unit[k]))
    ang = np.concatenate(ang)
    sig_a = max(float(np.median(np.arccos(np.minimum(1.0, ang)))) / 0.6745, 1e-6) if len(ang) else 0.6

    def run(groups):
        src = np.concatenate([src_nodes[i] for g in groups for i in g])
        slab = np.concatenate([np.full(sum(len(src_nodes[i]) for i in g), j, np.int64) for j, g in enumerate(groups)])
        sst = np.concatenate([np.full(len(src_nodes[i]), i, np.int64) for g in groups for i in g])
        lab, dist = _march_f(idx, pos, cost, offs, steps, unit, src, slab, sst, h0, b0, bri, sig_a, sig_b, SEED_UM, 2)
        d0 = np.where(lab[:, 0] < 0, 0.0, dist[:, 0])
        d1 = np.where(lab[:, 1] < 0, 1.0, dist[:, 1])
        rel = np.where(lab[:, 0] < 0, 0.0, np.where(lab[:, 1] < 0, 1.0, (d1 - d0) / np.maximum(d1, 1e-12)))
        return lab[:, 0], rel

    first = [[i] for i in range(len(seeds))]
    win, rel = run(first)
    unknown = [len(seeds) + j for j in range(len(pieces)) if np.median(rel[src_nodes[len(seeds) + j]]) < CONF]
    win, rel = run(first + ([unknown] if unknown else []))
    ids = np.array([k for k, _ in seeds] + [0], np.int64)
    win = np.where((win < 0) | (win >= len(seeds)), len(seeds), win)
    return idx, ids[win], np.where(ids[win] > 0, rel, 0.0).astype(np.float32)


def separate_elongated_cells(segmentation_mask, intensity_image, seed_mask, spacing, final, confidence):
    sp = np.asarray(spacing, float)
    nxt = 1
    records = []
    for lbl, sl in enumerate(ndimage.find_objects(np.asarray(segmentation_mask)), 1):
        if sl is None:
            continue
        own = np.asarray(segmentation_mask[sl]) == lbl
        seedlab = np.where(own, np.asarray(seed_mask[sl]), 0)
        out = final[sl]
        if own.ndim != 3 or intensity_image is None or not np.any(seedlab):
            out[own] = nxt
            nxt += 1
            continue
        own, region, t, v, bri = _prepare(lbl, sl, segmentation_mask, intensity_image, sp)
        anchors = _anchors(t, v, region, sp)
        ldir = v[:, :, 0].copy()
        del v
        idx, fib, rel = _vote(t, region, anchors, seedlab, sp, bri, ldir)
        del t, anchors
        ks = np.unique(fib[fib > 0])
        new = np.zeros(int(fib.max()) + 1, np.int64)
        new[ks] = np.arange(nxt, nxt + len(ks))
        nxt += len(ks)
        lab = new[fib]
        w = lab[idx[own]]
        if np.any(w == 0):
            w = np.where(w == 0, nxt, w)
            nxt += 1
        out[own] = w
        cv = confidence[sl]
        cv[region] = np.maximum(cv[region], rel)
        base = np.array([s.start for s in sl])
        rp = np.argwhere(region)
        for k in ks:
            m = (fib == k) & (rel >= 1.0 / 255)
            g = rp[m] + base
            records.append((int(new[k]), np.ravel_multi_index(tuple(g.T), final.shape).astype(np.uint32), np.round(rel[m] * 255).astype(np.uint8), np.r_[g.min(0), g.max(0) + 1].astype(np.int32)))
    return records
