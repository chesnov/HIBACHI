import numpy as np
from scipy import ndimage
from skimage.graph import MCP_Geometric
from skimage.morphology import skeletonize

SOMA_SHAPES = ("compact", "elongated")
SIGMA_UM = (0.5, 0.25, 0.25)
MAX_TILT = 0.7
MASK_W = 1.0
SIGNAL = 2.0
EPS = 0.05
REACH_UM = 2.0
TILE = 256
HALO_UM = 10.0


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


def _flow_cost(t, region, anchors, sp):
    out = np.full(region.shape, np.inf, np.float32)
    costs = np.full(region.shape, np.inf, np.float32)
    costs[region] = 1.0 / (EPS + t / max(float(np.percentile(t, 99)), 1e-6))
    h = [int(np.ceil(HALO_UM / q)) for q in sp[1:]]
    for y0 in range(0, region.shape[1], TILE):
        for x0 in range(0, region.shape[2], TILE):
            ys = slice(max(0, y0 - h[0]), min(region.shape[1], y0 + TILE + h[0]))
            xs = slice(max(0, x0 - h[1]), min(region.shape[2], x0 + TILE + h[1]))
            a = anchors[:, ys, xs]
            if not a.any():
                continue
            cum = MCP_Geometric(costs[:, ys, xs].astype(np.float64), sampling=tuple(sp)).find_costs(np.argwhere(a))[0]
            out[:, y0:y0 + TILE, x0:x0 + TILE] = cum[:, y0 - ys.start:y0 - ys.start + TILE, x0 - xs.start:x0 - xs.start + TILE]
    return out


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
        anchors = _anchors(t, v, region, sp) > 0
        del v
        cost = _flow_cost(t, region, anchors, sp)
        score = np.exp(-cost[own] / REACH_UM).astype(np.float32)
    return [{"coords": coords, "vol": len(coords), "score": 1.0, "rank_vals": score}], {}