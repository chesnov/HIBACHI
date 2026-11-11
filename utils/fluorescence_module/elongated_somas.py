import numpy as np
from scipy import ndimage

SOMA_SHAPES = ("compact", "elongated")
SIGMA_UM = (0.5, 0.25, 0.25)
MAX_TILT = 0.7
MASK_W = 1.0


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
        step /= np.abs(step).max(axis=1, keepdims=True)
        for sgn in (-1.0, 1.0):
            keep &= t >= ndimage.map_coordinates(dense, (pts + sgn * step).T, order=1)
    return keep


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
        dt = ndimage.distance_transform_edt(own, sampling=sp).astype(np.float32)
        img += MASK_W * dt / max(float(np.percentile(dt[own], 99)), 1e-6)
        region = ndimage.binary_dilation(own, iterations=2)
        t, v = _hessian(img, region, sp)
        line = _centreline(t, v, region, sp)
        inside = own[region]
        t, line = t[inside], line[inside]
        score = np.where(line, np.clip(t / max(float(np.percentile(t, 99)), 1e-6), 0, 1), 0).astype(np.float32)
    return [{"coords": coords, "vol": len(coords), "score": 1.0, "rank_vals": score}], {}