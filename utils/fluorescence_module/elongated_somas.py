"""Elongated somas: fibre seeds from the image's own flow (step 3, soma_shape "elongated").

Seeds for spindle- and fibre-shaped cells. `soma_extraction.py` imports
`_elongated_label_candidates` (and `SOMA_SHAPES`) from here and calls it once per object;
seed placement, the worker pool and the compact mode stay in `soma_extraction.py`. Nothing
here imports `soma_extraction`.

The evidence comes from the raw image, not from the mask's shape (the previous, mask-based
version -- skeleton tracing plus repair passes -- is in the repository history):

1. Direction field. Per voxel, a multi-scale Hessian of the intensity (scales in um, the z
   spacing handled explicitly) gives a tubeness (how much the voxel looks like the core of a
   bright tube) and the tube's direction. Computed tile by tile, only near the object.
2. Ridges. Voxels whose tubeness is a maximum across the local direction are centreline
   points of a fibre.
3. Segments. Centreline points are linked along their direction. Where linked points branch
   (neighbours ahead or behind spread sideways, or a neighbour runs at a clear angle), the
   point is a junction point; junction points are removed and what remains falls into
   confident single-fibre segments. Short segments caught between junction zones are part
   of the zone (fibres that touch along a short stretch).
4. Junctions. In each junction zone, the segment ends that enter it are paired as a whole:
   ends on one side of the zone with ends on the other, by turning angle, sideways miss,
   gap length and brightness along the connection; an end may also stop in the zone. The
   pairing with the lowest total cost is taken (assignment, not greedy). A pair whose
   removal costs little (below `_FLOW_MARGIN`) is not trusted: both ends stop there.
5. Votes. Every mask voxel votes for the fibres whose centreline (segments and their
   junction connections) passes near it, weighted by distance and by agreement between the
   voxel's own direction and the fibre's. Each voxel's score is the winning fibre's share of
   its votes (0 if no centreline is near). A voxel goes to that fibre if the score is at
   least `_FLOW_CONFIDENT`.

Every voxel of the object's mask ends up in exactly one seed. Fibre seeds are the confident
voxels of each fibre (fibres of at least the minimum seed size). All remaining voxels of the
object -- below `_FLOW_CONFIDENT`, without votes, or of fibres too small for a seed of their
own -- form one more seed, the object's "uncertain" seed (strat_name "Uncertain", placed
after the fibres). It marks where the method is unsure, so that each uncertain part can be
looked at and given its own heuristic later. An object without any fibre seed is one seed
as a whole, whatever its size.

Units: micrometres. 3D only: a 2D object becomes one seed as a whole. Seeds contain only,
and together all, of the object's own mask voxels.
"""
import math

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

SOMA_SHAPES = ("compact", "elongated")

_FLOW_SCALES_UM = (0.25, 0.35, 0.5, 0.7)  # Hessian scales (Gaussian sigma)
_FLOW_TILE = (256, 256)                   # tile size in y, x (whole z); halo added
_FLOW_HALO_UM = 2.5                       # halo around a tile (covers the largest filter)
_FLOW_NEAR_UM = 1.0                       # the field is computed within this of the object
_FLOW_RIDGE_STEP_UM = 0.35                # a ridge point is a tubeness maximum across its direction, at +- this
_FLOW_RIDGE_REL = 0.12                    # and at least this times the object's 99th-percentile ridge tubeness
_FLOW_LINK_UM = 0.5                       # ridge points this close are linked if their directions agree
_FLOW_LINK_DEG = 25.0                     # within this, and the step between them runs along that direction
_FLOW_LINK_ALONG = 0.7                    # (at least this share of the step; steps under 0.25 um always link)
_FLOW_BRANCH_SPREAD_UM = 0.4              # linked points ahead (or behind) spread sideways more than this: junction
_FLOW_JUNCTION_GROW_UM = 0.3              # junction points are grown by this before cutting
_FLOW_ZONE_UM = 0.8                       # segment ends (and junction points) this close form one junction zone,
_FLOW_GAP_UM = 3.0                        # as do two ends this close that face each other (a gap in the fibre)
_FLOW_THIN_LAT_UM = 0.4                   # ridge clouds are collapsed to centrelines: each point moves (twice) to the
_FLOW_THIN_ALONG_UM = 0.25                # tubeness-weighted centre of the ridge points within this across / along it,
_FLOW_THIN_GRID_UM = 0.15                 # then points closer than this are merged
_FLOW_MIN_SEG_UM = 1.5                    # shorter pieces are crumbs
_FLOW_ABSORB_UM = 0.0                     # (off) a segment this short between two zones would belong to them
_FLOW_END_UM = 2.5                        # an end's direction is taken over this much of its segment
_FLOW_PAIR_MAX_DEG = 60.0                 # ends are only paired if they turn by less than this
_FLOW_COST_END = 1.8                      # cost of an end stopping in a zone (relative to the pair cost terms)
_FLOW_MARGIN = 0.5                        # a pair is trusted only if the next-best pairing costs at least this more
_FLOW_SIGMA_UM = 0.5                      # vote weight falls off with this distance
_FLOW_VOTE_MAX_UM = 1.5                   # votes come from centrelines within this
_FLOW_CONFIDENT = 0.7                     # share of the votes a fibre needs to get the voxel


def _elongated_label_candidates(lbl, sl, segmentation_mask, intensity_image, params):
    """Seed candidates for one object in elongated mode (see the module docstring).

    Fibre seeds plus the object's uncertain seed cover every voxel of the object. An object
    without any fibre seed (also every 2D object, and any object when no intensity image is
    given) becomes one seed as a whole. Each candidate's "rank_vals" holds its voxels' scores
    (0 where no fibre analysis was possible); `soma_extraction` writes them to the
    confidence heatmap.
    """
    sp = np.asarray(params.spacing, float)
    diag = {}
    box = tuple(sl)
    off = np.array([s_.start for s_ in box])
    own = np.asarray(segmentation_mask[box]) == lbl
    if not own.any():
        return [], diag
    min_vol = int(params.min_seed_vol)
    if len(sp) != 3 or intensity_image is None or own.ndim != 3:
        return _whole_object(own, off, None), diag
    field = _direction_field(own, box, intensity_image, sp)
    ridge = _ridge_points(field, own, sp)
    fibres = []
    if len(ridge["x"]) >= 8:
        segs, zones = _segments(ridge, sp)
        fibres = _resolve_junctions(segs, zones, ridge, field, sp)
    if fibres:
        labels, score = _vote(own, field, fibres, sp)
    else:
        labels, score = np.zeros(int(own.sum()), np.int32), np.zeros(int(own.sum()), np.float32)
    vox = np.argwhere(own)
    candidates = []
    seeded = np.zeros(len(vox), bool)
    for f in range(1, len(fibres) + 1):
        sel = labels == f
        n = int(sel.sum())
        if n < min_vol:
            continue
        seeded |= sel
        candidates.append(_candidate(vox[sel] + off, score[sel], "Fibre", 1.0))
    if not candidates:
        return _whole_object(own, off, score), diag
    rest = ~seeded
    if rest.any():
        candidates.append(_candidate(vox[rest] + off, score[rest], "Uncertain", 0.0))
    return candidates, diag


def _candidate(coords, score, kind, priority):
    cc = coords.astype(np.int32)
    n = len(cc)
    return {"coords": cc, "peak_coord": np.asarray(cc[n // 2]), "vol": n, "score": priority,
            "strat_name": kind, "frag_max_thick": 0.0, "family": "Fibre",
            "dt_vals": np.zeros(n, np.float32), "rank_vals": np.asarray(score, np.float32)}


def _whole_object(own, off, score):
    """The whole object as one seed, whatever its size (every mask voxel gets a soma).
    `score`: the voxels' scores if the fibre analysis ran, else None (scores 0: not analysed)."""
    n = int(own.sum())
    if n == 0:
        return []
    sc = np.zeros(n, np.float32) if score is None else np.asarray(score, np.float32)
    return [_candidate(np.argwhere(own) + off, sc, "WholeObject", 1.0)]


# --------------------------------------------------------------------------- field
def _direction_field(own, box, intensity_image, sp):
    """Tubeness T, direction d (unit, physical z,y,x) and best scale, for voxels within
    `_FLOW_NEAR_UM` of the object. Returned sparsely: sorted flat indices into the box."""
    shape = own.shape
    vals = np.asarray(intensity_image[box], np.float32)[own]
    norm = float(np.percentile(vals, 99.5)) if len(vals) else 1.0
    norm = norm if norm > 0 else 1.0
    halo = [int(math.ceil(_FLOW_HALO_UM / s_)) for s_ in sp]
    F, TT, DD, SS = [], [], [], []
    for y0 in range(0, shape[1], _FLOW_TILE[0]):
        for x0 in range(0, shape[2], _FLOW_TILE[1]):
            y1, x1 = min(shape[1], y0 + _FLOW_TILE[0]), min(shape[2], x0 + _FLOW_TILE[1])
            gy0, gy1 = max(0, y0 - halo[1]), min(shape[1], y1 + halo[1])
            gx0, gx1 = max(0, x0 - halo[2]), min(shape[2], x1 + halo[2])
            own_t = own[:, gy0:gy1, gx0:gx1]
            if not own_t.any():
                continue
            near_t = ndimage.distance_transform_edt(~own_t, sampling=sp) <= _FLOW_NEAR_UM
            core = np.zeros_like(near_t)
            core[:, y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0] = near_t[:, y0 - gy0:y1 - gy0, x0 - gx0:x1 - gx0]
            if not core.any():
                continue
            sub = (slice(box[0].start, box[0].stop), slice(box[1].start + gy0, box[1].start + gy1),
                   slice(box[2].start + gx0, box[2].start + gx1))
            img = np.asarray(intensity_image[sub], np.float32) / norm
            cz, ly, lx = np.nonzero(core)
            bestT = np.zeros(len(cz), np.float32)
            bestD = np.zeros((len(cz), 3), np.float32)
            bestS = np.zeros(len(cz), np.float32)
            for s_um in _FLOW_SCALES_UM:
                sig = s_um / sp
                H = np.empty((len(cz), 3, 3), np.float32)
                for i in range(3):
                    for j in range(i, 3):
                        order = [0, 0, 0]
                        order[i] += 1
                        order[j] += 1
                        h = ndimage.gaussian_filter(img, sig, order=order)[cz, ly, lx]
                        h = h / (sp[i] * sp[j]) * s_um ** 2
                        H[:, i, j] = h
                        H[:, j, i] = h
                w, v = np.linalg.eigh(H)
                o = np.argsort(np.abs(w), axis=1)
                ws = np.take_along_axis(w, o, 1)
                T = np.where((ws[:, 1] < 0) & (ws[:, 2] < 0), np.sqrt(np.abs(ws[:, 1] * ws[:, 2])), 0)
                d = np.take_along_axis(v, o[:, None, :], 2)[:, :, 0]
                better = T > bestT
                bestT[better] = T[better]
                bestD[better] = d[better]
                bestS[better] = s_um
                del H, w, v, o, ws
            F.append(np.ravel_multi_index((cz, ly + gy0, lx + gx0), shape))
            TT.append(bestT); DD.append(bestD); SS.append(bestS)
    if not F:
        z = np.zeros(0)
        return {"flat": z.astype(np.int64), "T": z.astype(np.float32), "d": np.zeros((0, 3), np.float32),
                "s": z.astype(np.float32), "shape": shape, "box": box, "norm": norm,
                "img": np.asarray(intensity_image[box], np.float32) / norm}
    flat = np.concatenate(F)
    o = np.argsort(flat)
    return {"flat": flat[o], "T": np.concatenate(TT)[o], "d": np.concatenate(DD)[o], "s": np.concatenate(SS)[o],
            "shape": shape, "box": box, "norm": norm, "img": np.asarray(intensity_image[box], np.float32) / norm}


def _field_dense(field):
    T = np.zeros(field["shape"], np.float32)
    T.flat[field["flat"]] = field["T"]
    return T


# --------------------------------------------------------------------------- ridges
def _ridge_points(field, own, sp):
    """Centreline points: tubeness maxima across the local direction, inside the object."""
    T = _field_dense(field)
    flat, Tv, Dv = field["flat"], field["T"], field["d"]
    inside = own.flat[flat] & (Tv > 0)
    idx = np.flatnonzero(inside)
    pts = np.array(np.unravel_index(flat[idx], field["shape"])).T.astype(float)
    d = Dv[idx] / (np.linalg.norm(Dv[idx], axis=1, keepdims=True) + 1e-9)
    # two directions across d (physical), as voxel steps
    a = np.where(np.abs(d[:, 0:1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    e2 = np.cross(d, a)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-9
    e3 = np.cross(d, e2)
    keep = np.ones(len(idx), bool)
    t0 = Tv[idx]
    for e in (e2, e3):
        for sgn in (-1.0, 1.0):
            q = pts + sgn * _FLOW_RIDGE_STEP_UM * e / sp
            tq = ndimage.map_coordinates(T, q.T, order=1, mode="nearest")
            keep &= t0 >= tq
    idx, pts, d = idx[keep], pts[keep], d[keep]
    tv = Tv[idx]
    if len(tv):
        thr = _FLOW_RIDGE_REL * float(np.percentile(tv, 99))
        ok = tv >= thr
        idx, pts, d, tv = idx[ok], pts[ok], d[ok], tv[ok]
    x, d, tv = _thin(pts * sp, d, tv)
    return {"x": x, "vox": np.round(x / sp).astype(int), "d": d, "T": tv}


def _thin(x, d, tv):
    """Collapse each fibre's ridge cloud (several voxels across, more in z) to a thin centreline."""
    if len(x) < 2:
        return x, d, tv
    for _it in range(2):
        tree = cKDTree(x)
        nb = tree.query_ball_point(x, max(_FLOW_THIN_LAT_UM, _FLOW_THIN_ALONG_UM))
        nx = x.copy()
        for i, js in enumerate(nb):
            js = np.asarray(js)
            disp = x[js] - x[i]
            al = disp @ d[i]
            lat = np.linalg.norm(disp - np.outer(al, d[i]), axis=1)
            m = (np.abs(al) <= _FLOW_THIN_ALONG_UM) & (lat <= _FLOW_THIN_LAT_UM) & (np.abs(d[js] @ d[i]) >= 0.8)
            w = tv[js[m]]
            nx[i] = (x[js[m]] * w[:, None]).sum(0) / w.sum()
        x = nx
    # merge points closer than the grid (keep the most tube-like)
    key = np.round(x / _FLOW_THIN_GRID_UM).astype(np.int64)
    o = np.lexsort((-tv, key[:, 2], key[:, 1], key[:, 0]))
    k = key[o]
    first = np.r_[True, np.any(k[1:] != k[:-1], axis=1)]
    sel = o[first]
    return x[sel], d[sel], tv[sel]


# --------------------------------------------------------------------------- segments
def _segments(ridge, sp, stats=None):
    X, D = ridge["x"], ridge["d"]
    tree = cKDTree(X)
    n = len(X)
    pairs = tree.query_pairs(_FLOW_LINK_UM, output_type="ndarray")
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        cosd = np.abs(np.sum(D[a] * D[b], axis=1))
        disp = X[b] - X[a]
        dl = np.linalg.norm(disp, axis=1)
        along = np.abs(np.sum(disp * D[a], axis=1)) / np.maximum(dl, 1e-9)
        ok = (cosd >= math.cos(math.radians(_FLOW_LINK_DEG))) & ((along >= _FLOW_LINK_ALONG) | (dl <= 0.25))
        pairs = pairs[ok]
    # branching along the chain: linked points ahead (or behind) that spread sideways
    nbrs = [[] for _ in range(n)]
    for a, b in pairs:
        nbrs[a].append(b)
        nbrs[b].append(a)
    junction = np.zeros(n, bool)
    for i in range(n):
        js = nbrs[i]
        if len(js) < 2:
            continue
        js = np.asarray(js)
        disp = X[js] - X[i]
        al = disp @ D[i]
        lat = disp - np.outer(al, D[i])
        for side in (al > 0.15, al < -0.15):
            if side.sum() >= 2:
                L_ = lat[side]
                if np.max(np.linalg.norm(L_[:, None] - L_[None], axis=2)) > _FLOW_BRANCH_SPREAD_UM:
                    junction[i] = True
                    break
    jz = junction.copy()
    if junction.any():
        for g in tree.query_ball_point(X[junction], _FLOW_JUNCTION_GROW_UM):
            jz[g] = True
    keep_e = pairs[~jz[pairs[:, 0]] & ~jz[pairs[:, 1]]] if len(pairs) else pairs
    seg_id = _components(n, keep_e, ~jz)
    segs = _seg_records(seg_id, X, D, ridge["T"])
    # zones: clusters of segment ends and junction points
    E = [(si, k) for si, s_ in enumerate(segs) for k in (0, 1)]
    EP = np.array([segs[si]["ends"][k]["p"] for si, k in E]) if E else np.zeros((0, 3))
    JP = X[junction]
    P = np.vstack([EP, JP]) if len(JP) else EP
    links = cKDTree(P).query_pairs(_FLOW_ZONE_UM, output_type="ndarray") if len(P) > 1 else np.zeros((0, 2), int)
    if len(EP) > 1:
        EU = np.array([segs[si]["ends"][k]["u"] for si, k in E])
        gp = cKDTree(EP).query_pairs(_FLOW_GAP_UM, output_type="ndarray")
        if len(gp):
            g = EP[gp[:, 1]] - EP[gp[:, 0]]
            face = (np.sum(EU[gp[:, 0]] * g, axis=1) > 0) & (np.sum(EU[gp[:, 1]] * -g, axis=1) > 0)
            diff = np.array([E[a][0] != E[b][0] for a, b in gp])
            links = np.vstack([links, gp[face & diff]]) if len(links) else gp[face & diff]
    zone_of = _components(len(P), links, np.ones(len(P), bool)) if len(P) else np.zeros(0, int)
    # a zone needs at least two segment ends
    cnt = np.bincount(zone_of[:len(EP)], minlength=zone_of.max() + 1 if len(zone_of) else 0) if len(EP) else np.zeros(0)
    for (si, k), z in zip(E, zone_of[:len(EP)]):
        segs[si]["ends"][k]["zone"] = int(z) if cnt[z] >= 2 else None
    zones = {}
    for (si, k), z in zip(E, zone_of[:len(EP)]):
        if cnt[z] >= 2:
            zones.setdefault(int(z), []).append(si)
    # short segments with both ends in zones belong to them (zones merged)
    parent = {z: z for z in zones}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    for s_ in segs:
        z0, z1 = s_["ends"][0].get("zone"), s_["ends"][1].get("zone")
        if s_["len"] < _FLOW_ABSORB_UM and z0 is not None and z1 is not None:
            s_["absorbed"] = True
            parent[find(z0)] = find(z1)
    for s_ in segs:
        for e in s_["ends"]:
            if e.get("zone") is not None:
                e["zone"] = find(e["zone"])
    zones = {find(z): True for z in zones}
    if stats is not None:
        stats.update(ridge_points=n, links=int(len(pairs)), junction_points=int(junction.sum()),
                     segments=len(segs), absorbed=sum(1 for s_ in segs if s_.get("absorbed")), zones=len(zones))
    return segs, zones


def _components(n, edges, valid):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    out = np.full(n, -1)
    roots = {}
    for i in range(n):
        if not valid[i]:
            continue
        r = find(i)
        out[i] = roots.setdefault(r, len(roots))
    return out


def _seg_records(seg_id, X, D, Tv):
    segs = []
    for s in range(seg_id.max() + 1 if len(seg_id) and seg_id.max() >= 0 else 0):
        m = np.flatnonzero(seg_id == s)
        if len(m) < 3:
            continue
        x = X[m]
        c = x.mean(0)
        u = np.linalg.svd(x - c, full_matrices=False)[2][0]
        t = (x - c) @ u
        o = np.argsort(t)
        m, t = m[o], t[o]
        length = float(t[-1] - t[0])
        if length < _FLOW_MIN_SEG_UM:
            continue
        ends = []
        for sgn, k in ((-1.0, 0), (1.0, -1)):
            sel = (t <= t[0] + _FLOW_END_UM) if sgn < 0 else (t >= t[-1] - _FLOW_END_UM)
            q = X[m[sel]]
            if len(q) >= 3:
                uu = np.linalg.svd(q - q.mean(0), full_matrices=False)[2][0]
            else:
                uu = u
            uu = uu if (uu @ u) * sgn > 0 else -uu      # pointing out of the segment
            ends.append({"p": X[m[k]], "u": uu})
        segs.append({"idx": m, "len": length, "ends": ends, "T": float(np.median(Tv[m]))})
    return segs


# --------------------------------------------------------------------------- junctions
def _resolve_junctions(segs, zones, ridge, field, sp):
    """Pair the segment ends in each zone; returns fibres as lists of point arrays."""
    live = [i for i, s in enumerate(segs) if not s.get("absorbed")]
    parent = {i: i for i in live}

    def find(a):
        while parent[a] != a:
            a = parent[a]
        return a
    connectors = []
    img = field["img"]                       # box crop, already normalised
    off = np.zeros(3)
    for z in zones:
        ends = [(i, k) for i in live for k in (0, 1) if segs[i]["ends"][k].get("zone") == z]
        if len(ends) < 2:
            continue
        U = np.array([segs[i]["ends"][k]["u"] for i, k in ends])
        axis = np.linalg.svd(U, full_matrices=False)[2][0]
        side = np.sign(U @ axis)
        A = [e for e, s_ in zip(ends, side) if s_ > 0]      # ends pointing +axis (enter from the - side)
        B = [e for e, s_ in zip(ends, side) if s_ <= 0]
        if not A or not B:
            continue
        C = np.full((len(A), len(B)), np.inf)
        for a, (ia, ka) in enumerate(A):
            for b, (ib, kb) in enumerate(B):
                if ia == ib:
                    continue
                C[a, b] = _pair_cost(segs[ia]["ends"][ka], segs[ib]["ends"][kb], segs[ia], segs[ib],
                                     img, off, sp, 1.0)
        nA, nB = len(A), len(B)
        big = 1e6
        M = np.full((nA + nB, nB + nA), big)
        M[:nA, :nB] = np.where(np.isfinite(C), C, big)
        M[:nA, nB:] = np.where(np.eye(nA) > 0, _FLOW_COST_END, big)
        M[nA:, :nB] = np.where(np.eye(nB) > 0, _FLOW_COST_END, big)
        M[nA:, nB:] = 0.0
        r, c = linear_sum_assignment(M)
        total = M[r, c].sum()
        for a, b in zip(r, c):
            if a >= nA or b >= nB or M[a, b] >= big:
                continue
            M2 = M.copy()
            M2[a, b] = big
            r2, c2 = linear_sum_assignment(M2)
            if M2[r2, c2].sum() - total < _FLOW_MARGIN:
                continue                                    # not trusted: both ends stop here
            (ia, ka), (ib, kb) = A[a], B[b]
            parent[find(ia)] = find(ib)
            p1, p2 = segs[ia]["ends"][ka]["p"], segs[ib]["ends"][kb]["p"]
            nstep = max(2, int(np.linalg.norm(p2 - p1) / 0.2))
            t = np.linspace(0, 1, nstep)[:, None]
            connectors.append((ia, p1 + (p2 - p1) * t, (p2 - p1) / (np.linalg.norm(p2 - p1) + 1e-9)))
    groups = {}
    for i in live:
        groups.setdefault(find(i), []).append(i)
    fid = {}
    fibres = []
    for g, members in groups.items():
        pts = np.vstack([ridge["x"][segs[i]["idx"]] for i in members])
        tan = np.vstack([ridge["d"][segs[i]["idx"]] for i in members])
        fibres.append([pts, tan])
        for i in members:
            fid[i] = len(fibres) - 1
    for ia, pts, u in connectors:
        f = fid[ia]
        fibres[f][0] = np.vstack([fibres[f][0], pts])
        fibres[f][1] = np.vstack([fibres[f][1], np.repeat(u[None], len(pts), 0)])
    return fibres


def _pair_cost(e1, e2, s1, s2, img, off, sp, norm):
    """Cost of ends e1 (pointing into the zone) and e2 (pointing into it from the other side)
    being one fibre: turning, sideways miss, gap and darkness along the connection."""
    u1, u2 = e1["u"], e2["u"]
    turn = math.degrees(math.acos(float(np.clip(u1 @ (-u2), -1, 1))))
    if turn > _FLOW_PAIR_MAX_DEG:
        return np.inf
    g = e2["p"] - e1["p"]
    gap = float(np.linalg.norm(g))
    miss1 = float(np.linalg.norm(g - (g @ u1) * u1))
    miss2 = float(np.linalg.norm(-g - (-g @ u2) * u2))
    if g @ u1 < -0.3:
        return np.inf                                       # e2 lies behind e1
    t = np.linspace(0, 1, max(3, int(gap / 0.15)))[:, None]
    q = (e1["p"] + g * t) / sp + off
    vals = ndimage.map_coordinates(img, q.T, order=1, mode="nearest") / norm
    ref = max(1e-6, min(_ridge_brightness(e1, img, off, sp, norm), _ridge_brightness(e2, img, off, sp, norm)))
    dark = max(0.0, 0.6 - float(np.min(vals)) / ref)
    return turn / 30.0 + (miss1 + miss2) / 1.0 + gap / 4.0 + 2.0 * dark


def _ridge_brightness(e, img, off, sp, norm):
    q = (e["p"] - np.outer(np.linspace(0.3, 2.0, 6), e["u"])) / sp + off
    return float(np.median(ndimage.map_coordinates(img, q.T, order=1, mode="nearest"))) / norm


# --------------------------------------------------------------------------- votes
def _vote(own, field, fibres, sp):
    """Per object voxel: the fibre it belongs to (1-based, 0 below `_FLOW_CONFIDENT`) and its
    score (the winning fibre's share of the votes; 0 without votes)."""
    vox = np.argwhere(own)
    x = vox * sp
    pts = np.vstack([f[0] for f in fibres])
    tan = np.vstack([f[1] for f in fibres])
    fid = np.concatenate([np.full(len(f[0]), k) for k, f in enumerate(fibres)])
    tree = cKDTree(pts)
    k = min(24, len(pts))
    dd, hh = tree.query(x, k=k, distance_upper_bound=_FLOW_VOTE_MAX_UM)
    dd, hh = dd.reshape(len(x), -1), hh.reshape(len(x), -1)
    valid = np.isfinite(dd)
    hh = np.where(valid, hh, 0)
    # the voxel's own direction (from the field)
    flat = np.ravel_multi_index(vox.T, field["shape"])
    pos = np.searchsorted(field["flat"], flat)
    pos = np.clip(pos, 0, len(field["flat"]) - 1)
    dv = field["d"][pos]
    dv = dv / (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
    agree = np.abs(np.einsum("ij,ikj->ik", dv, tan[hh]))
    w = np.exp(-np.where(valid, dd, 9.0) ** 2 / (2 * _FLOW_SIGMA_UM ** 2)) * (0.3 + 0.7 * agree ** 2)
    w = np.where(valid, w, 0.0)
    f = fid[hh]
    vi = np.repeat(np.arange(len(x)), w.shape[1])
    fi, wi = f.ravel(), w.ravel()
    keep = wi > 0
    vi, fi, wi = vi[keep], fi[keep], wi[keep]
    out = np.zeros(len(x), np.int32)
    score = np.zeros(len(x), np.float32)
    if not len(vi):
        return out, score
    o = np.lexsort((fi, vi))
    vi, fi, wi = vi[o], fi[o], wi[o]
    new = np.r_[True, (vi[1:] != vi[:-1]) | (fi[1:] != fi[:-1])]
    st = np.flatnonzero(new)
    wmax = np.maximum.reduceat(wi, st)                     # each fibre's best vote per voxel
    v_, f_ = vi[st], fi[st]
    vs = np.r_[True, v_[1:] != v_[:-1]]
    vst = np.flatnonzero(vs)
    tot = np.add.reduceat(wmax, vst)
    top = np.maximum.reduceat(wmax, vst)
    # fibre of the top vote per voxel
    rank = np.lexsort((-wmax, v_))
    first = rank[np.r_[True, v_[rank][1:] != v_[rank][:-1]]]
    share = top / np.maximum(tot, 1e-9)
    score[v_[vst]] = share
    ok = share >= _FLOW_CONFIDENT
    out[v_[vst][ok]] = f_[first][ok] + 1
    return out, score