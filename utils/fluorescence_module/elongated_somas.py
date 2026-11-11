import numpy as np
from scipy import ndimage

SOMA_SHAPES = ("compact", "elongated")
SIGMA_UM = (0.5, 0.25, 0.25)


def _tubeness(img, own, sp):
    sig = np.asarray(SIGMA_UM)
    h = np.empty((int(own.sum()), 3, 3), np.float32)
    for i in range(3):
        for j in range(i, 3):
            order = [0, 0, 0]
            order[i] += 1
            order[j] += 1
            v = ndimage.gaussian_filter(img, sig / sp, order=order)[own] * sig[i] * sig[j] / (sp[i] * sp[j])
            h[:, i, j] = v
            h[:, j, i] = v
    w = np.linalg.eigvalsh(h)
    w = np.take_along_axis(w, np.argsort(np.abs(w), axis=1), 1)
    return np.where((w[:, 1] < 0) & (w[:, 2] < 0), np.sqrt(w[:, 1] * w[:, 2]), 0)


def _elongated_label_candidates(lbl, sl, segmentation_mask, intensity_image, params):
    box = tuple(sl)
    own = np.asarray(segmentation_mask[box]) == lbl
    coords = (np.argwhere(own) + np.array([s.start for s in box])).astype(np.int32)
    if len(coords) == 0:
        return [], {}
    score = np.zeros(len(coords), np.float32)
    if intensity_image is not None and own.ndim == 3:
        img = np.asarray(intensity_image[box], np.float32)
        img /= max(float(np.percentile(img[own], 99.5)), 1e-6)
        t = _tubeness(img, own, np.asarray(params.spacing, float))
        score = np.clip(t / max(float(np.percentile(t, 99)), 1e-6), 0, 1).astype(np.float32)
    return [{"coords": coords, "vol": len(coords), "score": 1.0, "rank_vals": score}], {}