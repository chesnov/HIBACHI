import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

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
GAP_UM = 3.0
MISS_UM = 1.0
MARGIN = 1.0
BEHIND_UM = 0.5
MEET_COS = 0.7


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
    for idx in np.split(o, cuts):
        if len(idx) < 4:
            continue
        xk = x[idx]
        c = xk.mean(0)
        _, sv, vt = np.linalg.svd(xk - c, full_matrices=False)
        u = vt[0]
        if sv[0] ** 2 < STRAIGHT * np.sum(sv ** 2) or np.ptp((xk - c) @ u) < SEED_UM:
            continue
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
    return np.asarray(seeds, int)


def _grow(anchors, seeds, sp):
    owner = np.zeros(int(anchors.max()) + 1, np.int64)
    if not len(seeds):
        return owner
    pts = np.argwhere(anchors > 0)
    lab = anchors[tuple(pts.T)]
    x = pts * sp
    pdir = np.zeros((len(owner), 3))
    o = np.argsort(lab, kind="stable")
    members = {}
    for idx in np.split(o, np.flatnonzero(np.diff(lab[o])) + 1):
        members[int(lab[idx[0]])] = idx
        if len(idx) >= 3:
            pdir[lab[idx[0]]] = np.linalg.svd(x[idx] - x[idx].mean(0), full_matrices=False)[2][0]
    ends = []
    for k in seeds:
        owner[k] = k
        xk = x[members[int(k)]]
        u = pdir[k]
        t = (xk - xk.mean(0)) @ u
        ends.append([k, xk[np.argmax(t)], u])
        ends.append([k, xk[np.argmin(t)], -u])
    tree = cKDTree(x)
    while True:
        offers = {}
        for e, (f, p, u) in enumerate(ends):
            nb = np.asarray(tree.query_ball_point(p, GAP_UM))
            if not len(nb):
                continue
            nb = nb[owner[lab[nb]] == 0]
            if not len(nb):
                continue
            g = x[nb] - p
            ahead = g @ u
            miss = np.linalg.norm(g - np.outer(ahead, u), axis=1)
            ok = (ahead > 0) & (miss <= MISS_UM)
            if not ok.any():
                continue
            q, g, ahead, miss = nb[ok], g[ok], ahead[ok], miss[ok]
            k = lab[q]
            d = pdir[k].copy()
            flat = ~np.any(d, axis=1)
            d[flat] = g[flat] / np.maximum(np.linalg.norm(g[flat], axis=1, keepdims=True), 1e-6)
            turn = np.degrees(np.arccos(np.minimum(1.0, np.abs(d @ u))))
            c = miss / 0.5 + turn / 30.0 + np.hypot(ahead, miss) / GAP_UM
            order = np.argsort(c)
            uk, first = np.unique(k[order], return_index=True)
            for kk, j in zip(uk, order[first]):
                if np.min((x[members[int(kk)]] - p) @ u) >= -BEHIND_UM:
                    offers.setdefault(int(kk), {})[e] = (float(c[j]), int(q[j]))
        best_for_end = {}
        for k, by_end in offers.items():
            ranked = sorted((c, e, q) for e, (c, q) in by_end.items())
            c1, e1, q1 = ranked[0]
            rivals = [c for c, e, q in ranked[1:] if ends[e][0] != ends[e1][0]]
            if rivals and rivals[0] - c1 < MARGIN:
                continue
            if e1 not in best_for_end or c1 < best_for_end[e1][0]:
                best_for_end[e1] = (c1, k, q1)
        if not best_for_end:
            break
        for e, (c, k, q) in best_for_end.items():
            f, p, u = ends[e]
            owner[k] = f
            xk = x[members[k]]
            far = xk[np.argmax((xk - p) @ u)]
            d = pdir[k] if np.any(pdir[k]) else (far - p) / max(np.linalg.norm(far - p), 1e-6)
            d = d if d @ u >= 0 else -d
            nu = u + d
            ends[e] = [f, far, nu / max(np.linalg.norm(nu), 1e-6)]
    blocked = {}
    for e, (f, p, u) in enumerate(ends):
        nb = np.asarray(tree.query_ball_point(p, GAP_UM))
        if not len(nb):
            continue
        nb = nb[(owner[lab[nb]] > 0) & (owner[lab[nb]] != f)]
        if not len(nb):
            continue
        g = x[nb] - p
        ahead = g @ u
        miss = np.linalg.norm(g - np.outer(ahead, u), axis=1)
        ok = (ahead > 0) & (miss <= MISS_UM)
        best = None
        for kk in np.unique(lab[nb[ok]]):
            if np.min((x[members[int(kk)]] - p) @ u) >= -GAP_UM:
                sel = ok & (lab[nb] == kk)
                c = float(np.min(miss[sel] / 0.5 + np.hypot(ahead[sel], miss[sel]) / GAP_UM))
                if best is None or c < best[0]:
                    best = (c, int(owner[kk]))
        if best:
            blocked[e] = best[1]
    root = np.arange(len(owner))
    for e, fb in blocked.items():
        fa = ends[e][0]
        for e2, fb2 in blocked.items():
            if ends[e2][0] == fb and fb2 == fa and ends[e][2] @ ends[e2][2] <= -MEET_COS:
                ra, rb = root[fa], root[fb]
                if ra != rb:
                    root[root == ra] = rb
    return root[owner]


def _assign(t, region, anchors, owner, own):
    costs = np.zeros(region.shape, np.float32)
    costs[region] = 1.0 / (EPS + t / max(float(np.percentile(t, 99)), 1e-6))
    return owner[watershed(costs, anchors, mask=region)[own]]


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
        dt = ndimage.distance_transform_edt(own_b, sampling=sp).astype(np.float32)
        img += MASK_W * dt / max(float(np.percentile(dt[own_b], 99)), 1e-6)
        grow = ndimage.binary_dilation(own, iterations=2) | ((sm >= SIGNAL * bg) & (seg == 0))
        comp = ndimage.label(grow, np.ones((3, 3, 3)))[0]
        region = np.isin(comp, np.unique(comp[own]))
        t, v = _hessian(img, region, sp, inner, bg)
        del img, dt, sm, grow, comp, own_b, seg
        anchors = _anchors(t, v, region, sp)
        del v
        seeds = _seeds(anchors, sp)
        owner = _grow(anchors, seeds, sp)
        view = np.where(np.isin(anchors, seeds), 1.0, np.where(owner[anchors] > 0, 0.7, np.where(anchors > 0, 0.35, 0.0))).astype(np.float32)
        score = view[own]
        outside = (anchors > 0) & ~own
        fibre = _assign(t, region, anchors, owner, own)
        out = []
        for k in np.unique(fibre):
            sel = fibre == k
            out.append({"coords": coords[sel], "vol": int(sel.sum()), "score": 1.0 if k else 0.0, "rank_vals": score[sel]})
        out[0]["heat_coords"] = (np.argwhere(outside) + np.array([s.start for s in box])).astype(np.int32)
        out[0]["heat_vals"] = view[outside]
        return out, {}
    return [{"coords": coords, "vol": len(coords), "score": 1.0, "rank_vals": score}], {}