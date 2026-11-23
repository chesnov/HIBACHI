import heapq
import numba
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

SOMA_SHAPES = ("compact", "elongated")
SIGMA_UM = (0.5, 0.25, 0.25)
MAX_TILT = 0.7
MASK_W = 1.0
SIGNAL = 2.0
SEED_UM = 3.0
STRAIGHT = 0.9
ALONE_UM = 1.5
LATERAL_UM = 0.4
CROWD = 0.1
NOISE_VOX = 10
EPS = 0.05
CONF = 0.5
EDT_CHUNK = 128


def _edt(mask, sp):
    out = np.zeros(mask.shape, np.float32)
    n = mask.shape[2]
    for a in range(0, n, EDT_CHUNK):
        b = min(n, a + EDT_CHUNK)
        m = 16
        while True:
            lo, hi = max(0, a - m), min(n, b + m)
            d = ndimage.distance_transform_edt(mask[:, :, lo:hi], sampling=sp)[:, :, a - lo:b - lo]
            if (lo == 0 and hi == n) or d.max() <= m * sp[2]:
                break
            m *= 2
        out[:, :, a:b] = d
    return out


def _hessian(img, region, sp, inner, cval):
    sig = np.asarray(SIGMA_UM)
    h = np.empty((int(region.sum()), 3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            order = [0, 0, 0]
            order[i] += 1
            order[j] += 1
            v = ndimage.gaussian_filter(img, sig / sp, order=order, mode="constant", cval=cval)[inner][region]
            v = v * sig[i] * sig[j] / (sp[i] * sp[j])
            h[:, i, j] = v
            h[:, j, i] = v
    w, v = np.linalg.eigh(h)
    o = np.argsort(np.abs(w), axis=1)
    w = np.take_along_axis(w, o, 1)
    v = np.take_along_axis(v, o[:, None, :], 2)
    t = np.where((w[:, 1] < 0) & (w[:, 2] < 0), np.sqrt(np.abs(w[:, 1] * w[:, 2])), 0).astype(np.float32)
    return t, v


def _anchors(t, v, region, sp):
    dense = np.zeros(region.shape, np.float32)
    dense[region] = t
    pts = np.argwhere(region).astype(np.float32)
    d = v[:, :, 0] * np.asarray(SIGMA_UM)
    keep = (t > 0) & (np.abs(d[:, 0]) <= MAX_TILT * np.linalg.norm(d, axis=1))
    for k in (1, 2):
        step = v[:, :, k] * np.asarray(SIGMA_UM) / sp
        for sgn in (-1.0, 1.0):
            keep &= t >= ndimage.map_coordinates(dense, (pts + sgn * step).T, order=1)
    ridge = np.zeros(region.shape, bool)
    ridge[region] = keep
    line = skeletonize(ridge)
    nb = ndimage.convolve(line.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant")
    return ndimage.label(line & (nb < 4), np.ones((3, 3, 3)))[0]


def _seeds(anchors, sp):
    pts = np.argwhere(anchors > 0)
    lab = anchors[tuple(pts.T)]
    x = pts * sp
    tree = cKDTree(x)
    sizes = np.bincount(lab)
    o = np.argsort(lab, kind="stable")
    cuts = np.flatnonzero(np.diff(lab[o])) + 1
    seeds = []
    lines = []
    for idx in np.split(o, cuts):
        if len(idx) < 4:
            continue
        xk = x[idx]
        c = xk.mean(0)
        _, sv, vt = np.linalg.svd(xk - c, full_matrices=False)
        u = vt[0]
        if sv[0] ** 2 < STRAIGHT * np.sum(sv ** 2) or np.ptp((xk - c) @ u) < SEED_UM:
            continue
        lines.append(lab[idx[0]])
        crowded = 0
        for p, nb in zip(xk, tree.query_ball_point(xk, ALONE_UM)):
            nb = np.asarray(nb)
            nb = nb[(lab[nb] != lab[idx[0]]) & (sizes[lab[nb]] >= NOISE_VOX)]
            if len(nb):
                w = x[nb] - p
                if np.any(np.linalg.norm(w - np.outer(w @ u, u), axis=1) > LATERAL_UM):
                    crowded += 1
        if crowded <= CROWD * len(idx):
            seeds.append(lab[idx[0]])
    return np.asarray(seeds, int), np.asarray(lines, int)


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


def _vote(t, region, anchors, seeds, lines, sp, bri, ldir):
    pos = np.argwhere(region)
    idx = -np.ones(region.shape, np.int32)
    idx[region] = np.arange(len(pos), dtype=np.int32)
    if not len(seeds):
        return idx, np.full(len(pos), -1, np.int64), np.zeros(len(pos), np.float32)
    cost = (1.0 / (EPS + t / max(float(np.percentile(t, 99)), 1e-6))).astype(np.float64)
    offs = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)], np.int64)
    steps = np.linalg.norm(offs * sp, axis=1)
    apts = np.argwhere(anchors > 0)
    alab = anchors[tuple(apts.T)]
    anode = idx[tuple(apts.T)].astype(np.int64)
    o = np.argsort(alab, kind="stable")
    nodes = {int(alab[g[0]]): anode[g] for g in np.split(o, np.flatnonzero(np.diff(alab[o])) + 1)}

    unit = offs * sp / steps[:, None]
    top = int(anchors.max()) + 1
    h0 = np.zeros((top, 3))
    b0 = np.zeros(top)
    for k, nd in nodes.items():
        if len(nd) >= 3:
            xk = pos[nd] * sp
            h0[k] = np.linalg.svd(xk - xk.mean(0), full_matrices=False)[2][0]
        else:
            d = ldir[nd[0]] * sp
            h0[k] = d / max(np.linalg.norm(d), 1e-12)
        b0[k] = np.median(bri[nd])
    sig_b = max(float(np.median(np.concatenate([np.abs(bri[nodes[int(k)]] - b0[int(k)]) for k in seeds]))) / 0.6745, 1e-6)
    is_seed = np.zeros(top, bool)
    is_seed[np.asarray(seeds, int)] = True
    ang = []
    for k in range(13):
        sh = anchors[max(0, -offs[k, 0]):anchors.shape[0] - max(0, offs[k, 0]), max(0, -offs[k, 1]):anchors.shape[1] - max(0, offs[k, 1]), max(0, -offs[k, 2]):anchors.shape[2] - max(0, offs[k, 2])]
        nb = anchors[max(0, offs[k, 0]):anchors.shape[0] + min(0, offs[k, 0]), max(0, offs[k, 1]):anchors.shape[1] + min(0, offs[k, 1]), max(0, offs[k, 2]):anchors.shape[2] + min(0, offs[k, 2])]
        q = sh[(sh > 0) & (sh == nb)]
        q = q[is_seed[q]]
        ang.append(np.abs(h0[q] @ unit[k]))
    sig_a = max(float(np.median(np.arccos(np.minimum(1.0, np.concatenate(ang))))) / 0.6745, 1e-6)

    def run_f(groups, kk):
        src = np.concatenate([nodes[k] for g in groups for k in g])
        slab = np.concatenate([np.full(sum(len(nodes[k]) for k in g), i, np.int64) for i, g in enumerate(groups)])
        sst = np.concatenate([np.full(len(nodes[k]), k, np.int64) for g in groups for k in g])
        lab, dist = _march_f(idx, pos.astype(np.int64), cost, offs, steps, unit, src, slab, sst, h0, b0, bri, sig_a, sig_b, SEED_UM, kk)
        rel = np.where(lab[:, 0] < 0, 0.0, np.where(lab[:, 1] < 0, 1.0, (dist[:, 1] - dist[:, 0]) / np.maximum(dist[:, 1], 1e-12)))
        return lab, dist, rel

    kept = [int(k) for k in seeds]
    rest = [int(k) for k in lines if int(k) not in set(kept)]
    while True:
        lab, dist, rel = run_f([[k] for k in kept], 2)
        add = [k for k in rest if np.median(rel[nodes[k]]) < CONF]
        if not add:
            break
        kept += add
        rest = [k for k in rest if k not in set(add)]
    lab, dist, rel = run_f([[k] for k in kept], 3)
    parent = list(range(len(kept)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in sorted(range(len(kept)), key=lambda i: len(nodes[kept[i]])):
        nd = nodes[kept[i]]
        alt = lab[nd, 1]
        if not np.any(alt >= 0):
            continue
        a = int(np.bincount(alt[alt >= 0]).argmax())
        r = np.where(alt == a, np.where(lab[nd, 2] < 0, 1.0, (dist[nd, 2] - dist[nd, 1]) / np.maximum(dist[nd, 2], 1e-12)), 0.0)
        if np.median(r) >= CONF:
            parent[find(i)] = find(a)
    roots = sorted({find(i) for i in range(len(kept))})
    groups = [[kept[i] for i in range(len(kept)) if find(i) == r] for r in roots]
    lab, dist, rel = run_f(groups, 2)
    inside = {k for g in groups for k in g}
    unknown = [k for k in nodes if k not in inside and np.median(rel[nodes[k]]) < CONF]
    final = groups + ([unknown] if unknown else [])
    lab, dist, rel = run_f(final, 2)
    win = np.where((lab[:, 0] < 0) | (lab[:, 0] == len(groups)), -1, lab[:, 0])
    rel = np.where(win < 0, 0.0, rel)
    return idx, win, rel.astype(np.float32)


def _elongated_label_candidates(lbl, sl, segmentation_mask, intensity_image, params):
    box = tuple(sl)
    own = np.asarray(segmentation_mask[box]) == lbl
    coords = (np.argwhere(own) + np.array([s.start for s in box])).astype(np.int32)
    if len(coords) == 0:
        return [], {}
    score = np.zeros(len(coords), np.float32)
    if intensity_image is not None and own.ndim == 3:
        sp = np.asarray(params.spacing, float)
        pad = [int(np.ceil(4 * s / q)) for s, q in zip(SIGMA_UM, sp)]
        big = tuple(slice(max(0, b.start - p), min(m, b.stop + p)) for b, p, m in zip(box, pad, segmentation_mask.shape))
        inner = tuple(slice(b.start - g.start, b.stop - g.start) for b, g in zip(box, big))
        own_b = np.asarray(segmentation_mask[big]) == lbl
        img = np.asarray(intensity_image[big], np.float32)
        img /= max(float(np.percentile(img[own_b], 99.5)), 1e-6)
        sm = ndimage.gaussian_filter(img, np.asarray(SIGMA_UM) / sp)[inner]
        seg = np.asarray(segmentation_mask[box])
        bg = float(np.median(sm[seg == 0]))
        dt = _edt(own_b, sp)
        img += MASK_W * dt / max(float(np.percentile(dt[own_b], 99)), 1e-6)
        grow = ndimage.binary_dilation(own, iterations=2) | ((sm >= SIGNAL * bg) & (seg == 0))
        comp = ndimage.label(grow, np.ones((3, 3, 3)))[0]
        region = np.isin(comp, np.unique(comp[own]))
        t, v = _hessian(img, region, sp, inner, bg)
        bri = np.log(np.maximum(sm[region], 1e-6)).astype(np.float32)
        del img, dt, sm, grow, comp, own_b, seg
        anchors = _anchors(t, v, region, sp)
        ldir = v[:, :, 0].copy()
        del v
        seeds, lines = _seeds(anchors, sp)
        idx, win, rel = _vote(t, region, anchors, seeds, lines, sp, bri, ldir)
        fibre = win[idx[own]] + 1
        score = rel[idx[own]]
        outside = (anchors > 0) & ~own
        out = []
        for k in np.unique(fibre):
            sel = fibre == k
            out.append({"coords": coords[sel], "vol": int(sel.sum()), "score": 1.0 if k else 0.0, "rank_vals": score[sel]})
        out[0]["heat_coords"] = (np.argwhere(outside) + np.array([s.start for s in box])).astype(np.int32)
        out[0]["heat_vals"] = rel[idx[outside]]
        return out, {}
    return [{"coords": coords, "vol": len(coords), "score": 1.0, "rank_vals": score}], {}