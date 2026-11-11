import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

SOMA_SHAPES = ("compact", "elongated")
SIGMA_UM = (0.5, 0.25, 0.25)
MAX_TILT = 0.7
MASK_W = 1.0
GAP_UM = 3.0
END_UM = 1.0
FACE_COS = 0.87
MISS_UM = 0.5
DIP = 0.3


def _hessian(img, region, sp):
    sig = np.asarray(SIGMA_UM)
    h = np.empty((int(region.sum()), 3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            order = [0, 0, 0]
            order[i] += 1
            order[j] += 1
            v = ndimage.gaussian_filter(img, sig / sp, order=order)[region] * sig[i] * sig[j] / (sp[i] * sp[j])
            h[:, i, j] = v
            h[:, j, i] = v
    w, v = np.linalg.eigh(h)
    o = np.argsort(np.abs(w), axis=1)
    w = np.take_along_axis(w, o, 1)
    v = np.take_along_axis(v, o[:, None, :], 2)
    t = np.where((w[:, 1] < 0) & (w[:, 2] < 0), np.sqrt(w[:, 1] * w[:, 2]), 0).astype(np.float32)
    return t, v


def _centreline(t, v, region, sp):
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
    return skeletonize(ridge)[region]


def _ends(line, sp):
    nb = ndimage.convolve(line.astype(np.uint8), np.ones((3, 3, 3), np.uint8), mode="constant")
    lab, n = ndimage.label(line, np.ones((3, 3, 3)))
    pts = np.argwhere(line)
    ends = np.argwhere(line & (nb == 2))
    tree = cKDTree(pts * sp)
    dirs = []
    for e in ends:
        near = pts[tree.query_ball_point(e * sp, END_UM)]
        near = near[lab[tuple(near.T)] == lab[tuple(e)]]
        d = (e - near.mean(0)) * sp
        dirs.append(d / max(np.linalg.norm(d), 1e-6))
    return lab, n, ends, np.array(dirs).reshape(-1, 3)


def _links(lab, ends, dirs, sm, bg, sp):
    if len(ends) < 2:
        return []
    x = ends * sp
    pairs = cKDTree(x).query_pairs(GAP_UM, output_type="ndarray")
    ok = []
    for a, b in pairs:
        if lab[tuple(ends[a])] == lab[tuple(ends[b])]:
            continue
        g = x[b] - x[a]
        dist = np.linalg.norm(g)
        if dirs[a] @ g < FACE_COS * dist or dirs[b] @ -g < FACE_COS * dist:
            continue
        if np.linalg.norm(g - (g @ dirs[a]) * dirs[a]) > MISS_UM or np.linalg.norm(g - (g @ dirs[b]) * dirs[b]) > MISS_UM:
            continue
        q = ends[a] + np.linspace(0, 1, max(3, int(dist / 0.1)))[:, None] * (ends[b] - ends[a])
        lo = ndimage.map_coordinates(sm, q.T, order=1).min()
        level = min(sm[tuple(ends[a])], sm[tuple(ends[b])])
        if lo - bg >= DIP * (level - bg):
            ok.append((dist, a, b))
    best = {}
    for dist, a, b in ok:
        for e in (a, b):
            if e not in best or dist < best[e][0]:
                best[e] = (dist, a, b)
    return [(a, b) for dist, a, b in ok if best[a] == (dist, a, b) and best[b] == (dist, a, b)]


def _elongated_label_candidates(lbl, sl, segmentation_mask, intensity_image, params):
    box = tuple(sl)
    own = np.asarray(segmentation_mask[box]) == lbl
    coords = (np.argwhere(own) + np.array([s.start for s in box])).astype(np.int32)
    if len(coords) == 0:
        return [], {}
    score = np.zeros(len(coords), np.float32)
    if intensity_image is not None and own.ndim == 3:
        sp = np.asarray(params.spacing, float)
        img = np.asarray(intensity_image[box], np.float32)
        img /= max(float(np.percentile(img[own], 99.5)), 1e-6)
        sm = ndimage.gaussian_filter(img, np.asarray(SIGMA_UM) / sp)
        bg = float(np.median(sm[np.asarray(segmentation_mask[box]) == 0]))
        dt = ndimage.distance_transform_edt(own, sampling=sp).astype(np.float32)
        img += MASK_W * dt / max(float(np.percentile(dt[own], 99)), 1e-6)
        region = ndimage.binary_dilation(own, iterations=2)
        t, v = _hessian(img, region, sp)
        line = _centreline(t, v, region, sp)
        dense = np.zeros(own.shape, bool)
        dense[region] = line
        lab, n, ends, dirs = _ends(dense, sp)
        root = np.arange(n + 1)
        for a, b in _links(lab, ends, dirs, sm, bg, sp):
            ra, rb = root[lab[tuple(ends[a])]], root[lab[tuple(ends[b])]]
            root[root == ra] = rb
        fibre = root[lab[own]] * (lab[own] > 0)
        inside = own[region]
        t = t[inside]
        score = np.where(fibre > 0, np.clip(t / max(float(np.percentile(t, 99)), 1e-6), 0, 1), 0).astype(np.float32)
        out = []
        for f in np.unique(fibre):
            sel = fibre == f
            out.append({"coords": coords[sel], "vol": int(sel.sum()), "score": 1.0 if f else 0.0,
                        "rank_vals": score[sel]})
        return out, {}
    return [{"coords": coords, "vol": len(coords), "score": 1.0, "rank_vals": score}], {}