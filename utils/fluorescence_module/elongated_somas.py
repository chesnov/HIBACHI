import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

SOMA_SHAPES = ("compact", "elongated")
SIGMA_UM = (0.5, 0.25, 0.25)
MAX_TILT = 0.7
MASK_W = 1.0
SIGNAL = 2.0


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
        anchors = _anchors(t, v, region, sp)
        tv = np.zeros(own.shape, np.float32)
        tv[region] = t
        score = np.where(anchors[own] > 0, np.clip(tv[own] / max(float(np.percentile(tv[own], 99)), 1e-6), 0, 1), 0).astype(np.float32)
    return [{"coords": coords, "vol": len(coords), "score": 1.0, "rank_vals": score}], {}