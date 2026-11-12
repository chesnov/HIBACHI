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
REACH_UM = 1.0


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


def _seed_somas(t, region, anchors, seeds, own, sp):
    lab = np.where(np.isin(anchors, seeds), anchors, 0)
    if not lab.any():
        return np.zeros(int(own.sum()), np.int64)
    costs = np.zeros(region.shape, np.float32)
    costs[region] = 1.0 / (EPS + t / max(float(np.percentile(t, 99)), 1e-6))
    basin = watershed(costs, lab, mask=region)[own]
    pts = np.argwhere(lab > 0)
    dist, near = cKDTree(pts * sp).query(np.argwhere(own) * sp)
    nearest = lab[tuple(pts[near].T)]
    return np.where((dist <= REACH_UM) & (basin == nearest), nearest, 0)


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
        view = np.where(np.isin(anchors, seeds), 1.0, np.where(anchors > 0, 0.35, 0.0)).astype(np.float32)
        score = view[own]
        outside = (anchors > 0) & ~own
        fibre = _seed_somas(t, region, anchors, seeds, own, sp)
        out = []
        for k in np.unique(fibre):
            sel = fibre == k
            out.append({"coords": coords[sel], "vol": int(sel.sum()), "score": 1.0 if k else 0.0, "rank_vals": score[sel]})
        out[0]["heat_coords"] = (np.argwhere(outside) + np.array([s.start for s in box])).astype(np.int32)
        out[0]["heat_vals"] = view[outside]
        return out, {}
    return [{"coords": coords, "vol": len(coords), "score": 1.0, "rank_vals": score}], {}