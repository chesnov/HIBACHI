import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

try:
    from .elongated_somas import SEED_UM, SIGMA_UM, _prepare
except ImportError:
    from elongated_somas import SEED_UM, SIGMA_UM, _prepare

STEP_UM = 1.0
NODE_UM = 0.5
REACH_UM = 4.0


def _phys(vec):
    p = vec * np.asarray(SIGMA_UM)
    return p / np.maximum(np.linalg.norm(p, axis=-1, keepdims=True), 1e-12)


def _models(seedlab, idx, x, iv, hp, vdir):
    out = []
    pts = np.argwhere(seedlab > 0)
    lab = seedlab[tuple(pts.T)]
    o = np.argsort(lab, kind="stable")
    sig = np.asarray(SIGMA_UM)
    for g in np.split(o, np.flatnonzero(np.diff(lab[o])) + 1) if len(o) else []:
        nd = idx[tuple(pts[g].T)].astype(np.int64)
        xk = x[nd]
        c = xk.mean(0)
        if len(nd) >= 3:
            u = np.linalg.svd(xk - c, full_matrices=False)[2][0]
        else:
            d = vdir[nd]
            u = (d * np.sign(d @ d[0])[:, None]).mean(0)
            u /= max(np.linalg.norm(u), 1e-12)
        s = (xk - c) @ u
        if np.ptp(s) < NODE_UM:
            s = np.array([-NODE_UM / 2, NODE_UM / 2])
        h = hp[nd] / (sig[:, None] * sig[None, :])
        w, e = np.linalg.eigh(h)
        k = np.argsort(np.abs(w), 1)
        w = np.take_along_axis(w, k, 1)
        e = np.take_along_axis(e, k[:, None, :], 2)
        ok = (w[:, 1] < 0) & (w[:, 2] < 0)
        w1 = float(np.median(np.sqrt(iv[nd][ok] / -w[ok, 1]))) if ok.any() else np.nan
        w2 = float(np.median(np.sqrt(iv[nd][ok] / -w[ok, 2]))) if ok.any() else np.nan
        a1 = e[ok][:, :, 1] if ok.any() else e[:, :, 1]
        a1 = (a1 * np.sign(a1 @ a1[0])[:, None]).mean(0)
        a1 -= (a1 @ u) * u
        if np.linalg.norm(a1) < 1e-6:
            a1 = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
        a1 /= np.linalg.norm(a1)
        n = max(2, int(np.ceil(np.ptp(s) / NODE_UM)) + 1)
        ss = np.linspace(s.min(), s.max(), n)
        out.append({"k": int(lab[g[0]]), "nd": nd, "P": c + ss[:, None] * u, "U": np.tile(u, (n, 1)), "A1": np.tile(a1, (n, 1)), "B": np.full(n, float(np.median(np.log(np.maximum(iv[nd], 1e-6))))), "w1": w1, "w2": w2, "grow": [True, True], "members": [int(lab[g[0]])]})
    w1s = [f["w1"] for f in out if np.isfinite(f["w1"])]
    w2s = [f["w2"] for f in out if np.isfinite(f["w2"])]
    for f in out:
        if not np.isfinite(f["w1"]):
            f["w1"] = float(np.median(w1s)) if w1s else SIGMA_UM[1] * 2
            f["w2"] = float(np.median(w2s)) if w2s else SIGMA_UM[1] * 2
    return out


def _geometry(f, x, sa, lo=0, hi=None):
    hi = len(f["P"]) if hi is None else hi
    P, U, A1, B = f["P"][lo:hi], f["U"][lo:hi], f["A1"][lo:hi], f["B"][lo:hi]
    if len(P) < 2:
        lo, hi = max(0, lo - 1), min(len(f["P"]), hi + 1)
        P, U, A1, B = f["P"][lo:hi], f["U"][lo:hi], f["A1"][lo:hi], f["B"][lo:hi]
    seg = P[1:] - P[:-1]
    l2 = np.maximum((seg ** 2).sum(1), 1e-12)
    if len(seg) <= 8:
        cs = np.broadcast_to(np.arange(len(seg)), (len(x), len(seg)))
    else:
        nn = cKDTree(P).query(x, k=2)[1]
        cs = np.clip(np.stack([nn[:, 0] - 1, nn[:, 0], nn[:, 1] - 1, nn[:, 1]], 1), 0, len(seg) - 1)
    tt = np.clip(np.einsum("nkj,nkj->nk", x[:, None, :] - P[cs], seg[cs]) / l2[cs], 0, 1)
    near = P[cs] + tt[..., None] * seg[cs]
    c = ((x[:, None, :] - near) ** 2).sum(2).argmin(1)
    r = np.arange(len(x))
    si = cs[r, c]
    near = near[r, c][:, None, :]
    tsel = tt[r, c]
    ni = np.where(tsel < 0.5, si, si + 1)
    u = U[ni].copy()
    a1 = A1[ni].copy()
    b = B[ni].copy()
    g = x - near[:, 0]
    e = np.zeros(len(x))
    ends = ([(0, -U[0])] if lo == 0 else []) + ([(len(P) - 1, U[-1])] if hi == len(f["P"]) else [])
    for end, out in ends:
        m = (ni == end) & (((x - P[end]) @ out) > 0)
        g[m] = x[m] - P[end]
        e[m] = g[m] @ out
        g[m] -= e[m][:, None] * out[None]
        u[m] = out
    a1 -= np.einsum("ij,ij->i", a1, u)[:, None] * u
    a1 /= np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-12)
    a2 = np.cross(u, a1)
    v1 = f["w1"] ** 2 + (e * sa) ** 2
    v2 = f["w2"] ** 2 + (e * sa) ** 2
    rho = np.einsum("ij,ij->i", g, a1) ** 2 / v1 + np.einsum("ij,ij->i", g, a2) ** 2 / v2
    return g, e, u, b, v1, v2, rho


def _energy(f, ids, x, iv, dtv, inm, vdir, tube, bg, sp_, near=None, reach=0.0):
    sb, sdt, sa, rad = sp_
    lo, hi = 0, len(f["P"])
    if near is not None:
        dn = np.min(np.linalg.norm(f["P"][:, None, :] - near[None], axis=2), 1)
        k = np.flatnonzero(dn <= 2 * reach)
        lo, hi = int(k.min()), int(k.max()) + 1
    g, e, u, b, v1, v2, rho = _geometry(f, x[ids], sa, lo, hi)
    pred = np.log(bg + np.maximum(np.exp(b) - bg, 0) * np.exp(-0.5 * rho))
    en = 0.5 * rho + 0.5 * np.log(v1 * v2) + 0.5 * ((np.log(np.maximum(iv[ids], 1e-6)) - pred) / sb) ** 2
    ang = np.arccos(np.minimum(1, np.abs(np.einsum("ij,ij->i", vdir[ids], u))))
    en += np.where(tube[ids], 0.5 * (ang / sa) ** 2, 0)
    en += np.where(inm[ids], 0.5 * ((dtv[ids] - np.maximum(rad - np.linalg.norm(g, axis=1), 0)) / sdt) ** 2, 0)
    cb = np.log(np.maximum((iv[ids] - bg) / np.exp(-0.5 * rho) + bg, 1e-6))
    return en, cb


def _vote(seedlab, own, region, t, v, bri, ex, sp, p_label, p_grow, none_pct):
    pos = np.argwhere(region)
    x = pos * sp
    idx = -np.ones(region.shape, np.int32)
    idx[region] = np.arange(len(pos), dtype=np.int32)
    iv = np.exp(bri)
    dtv = ex["dt"]
    inm = own[region]
    vdir = _phys(v[:, :, 0])
    tube = t > 0
    bg = ex["bg"]
    fibs = _models(seedlab, idx, x, iv, ex["h"], vdir)
    tree = cKDTree(x)
    cnt = np.zeros(len(x), np.int32)
    who = np.full(len(x), -1, np.int64)
    for j, f in enumerate(fibs):
        P, u = f["P"], f["U"][0]
        ids = np.unique(np.concatenate(tree.query_ball_point(P, 2 * f["w1"] + NODE_UM))).astype(np.int64)
        if not len(ids):
            continue
        s0 = (x[ids] - P[0]) @ u
        s1 = (x[ids] - P[-1]) @ u
        lat = np.linalg.norm((x[ids] - P[0]) - np.outer(s0, u), axis=1)
        m = ids[(s0 >= 0) & (s1 <= 0) & (lat <= 2 * f["w1"])]
        cnt[m] += 1
        who[m] = j
    cal = np.flatnonzero(cnt == 1)
    rb, rm, ra = [], [], []
    for j, f in enumerate(fibs):
        f["R"] = float(np.median(dtv[f["nd"]]))
        ids = cal[who[cal] == j]
        if not len(ids):
            continue
        g, e, u, b, v1, v2, rho = _geometry(f, x[ids], 0.0)
        rb.append(np.log(np.maximum(iv[ids], 1e-6)) - np.log(bg + np.maximum(np.exp(b) - bg, 0) * np.exp(-0.5 * rho)))
        im = inm[ids]
        rm.append((dtv[ids] - np.maximum(f["R"] - np.linalg.norm(g, axis=1), 0))[im])
        ra.append(np.arccos(np.minimum(1, np.abs(np.einsum("ij,ij->i", vdir[ids], u))))[tube[ids]])
    sb = max(float(np.median(np.abs(np.concatenate(rb)))) / 0.6745, 1e-3) if rb else 0.35
    sdt = max(float(np.median(np.abs(np.concatenate(rm)))) / 0.6745, 1e-3) if rm else 0.25
    sa = max(float(np.median(np.concatenate(ra))) / 0.6745, 1e-3) if ra else 0.45
    ent = [None] * len(fibs)
    tot = np.zeros(len(x))

    def score(j, near=None):
        f = fibs[j]
        ids = np.unique(np.concatenate(tree.query_ball_point(f["P"] if near is None else near, REACH_UM + 2 * f["w1"] + STEP_UM))).astype(np.int64)
        if not len(ids):
            return
        en, cb = _energy(f, ids, x, iv, dtv, inm, vdir, tube, bg, (sb, sdt, sa, f["R"]), near, REACH_UM + 2 * f["w1"] + STEP_UM)
        if ent[j] is None:
            ent[j] = (ids, en, cb)
            np.add.at(tot, ids, np.exp(-en))
            return
        oi, oe, ocb = ent[j]
        keep = ~np.isin(oi, ids)
        np.add.at(tot, oi[~keep], -np.exp(-oe[~keep]))
        np.add.at(tot, ids, np.exp(-en))
        ni = np.r_[oi[keep], ids]
        o = np.argsort(ni, kind="stable")
        ent[j] = (ni[o], np.r_[oe[keep], en][o], np.r_[ocb[keep], cb][o])

    for j in range(len(fibs)):
        score(j)
    best = np.full(len(x), np.inf)
    for j in range(len(fibs)):
        if ent[j] is not None:
            np.minimum.at(best, ent[j][0], ent[j][1])
    ok = np.isfinite(best[cal])
    e0 = float(np.percentile(best[cal][ok], none_pct)) if ok.any() else 10.0
    z0 = np.exp(-e0)
    back = max(1, int(round(SEED_UM / NODE_UM)))

    def drop(j):
        oi, oe, _ = ent[j]
        np.add.at(tot, oi, -np.exp(-oe))
        ent[j] = None

    def orient(f, flip):
        if flip:
            return f["P"][::-1], -f["U"][::-1], f["A1"][::-1], f["B"][::-1]
        return f["P"], f["U"], f["A1"], f["B"]

    def ties():
        vs, js, ps = [], [], []
        for j in range(len(fibs)):
            if ent[j] is None:
                continue
            ei, ee, _ = ent[j]
            p = np.exp(-ee) / (tot[ei] + z0)
            m = p >= 0.05
            vs.append(ei[m])
            js.append(np.full(int(m.sum()), j))
            ps.append(p[m])
        if not vs:
            return []
        v, jj, pp = np.concatenate(vs), np.concatenate(js), np.concatenate(ps)
        o = np.lexsort((-pp, v))
        v, jj, pp = v[o], jj[o], pp[o]
        i1 = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
        i2 = np.minimum(i1 + 1, len(v) - 1)
        ok = (i2 > i1) & (v[i2] == v[i1])
        tie = ok & (pp[i1] + pp[i2] >= p_grow) & (pp[i1] < p_grow)
        a, b, tv = np.minimum(jj[i1], jj[i2])[tie], np.maximum(jj[i1], jj[i2])[tie], v[i1][tie]
        if not len(tv):
            return []
        key = a * len(fibs) + b
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        out, used = [], set()
        for q in np.argsort(-cnt):
            if cnt[q] < 3:
                break
            ja, jb = int(uk[q] // len(fibs)), int(uk[q] % len(fibs))
            if ja in used or jb in used:
                continue
            c = x[tv[inv == q]].mean(0)
            fa, fb = fibs[ja], fibs[jb]
            ea = 0 if np.linalg.norm(c - fa["P"][0]) <= np.linalg.norm(c - fa["P"][-1]) else 1
            eb = 0 if np.linalg.norm(c - fb["P"][0]) <= np.linalg.norm(c - fb["P"][-1]) else 1
            pa, pb = fa["P"][0 if ea == 0 else -1], fb["P"][0 if eb == 0 else -1]
            oa = -fa["U"][0] if ea == 0 else fa["U"][-1]
            ob = -fb["U"][0] if eb == 0 else fb["U"][-1]
            if np.linalg.norm(c - pa) > REACH_UM or np.linalg.norm(c - pb) > REACH_UM:
                continue
            if (pb - pa) @ oa <= 0 or (pa - pb) @ ob <= 0:
                continue
            used.update((ja, jb))
            out.append((ja, ea, jb, eb, int(cnt[q])))
        return out

    def grow():
        rounds = 0
        while rounds < 200:
            rounds += 1
            grown = {}
            for j, f in enumerate(fibs):
                if ent[j] is None:
                    continue
                ei, ee, ecb = ent[j]
                for end in (0, 1):
                    if not f["grow"][end]:
                        continue
                    P = f["P"]
                    ep = P[0] if end == 0 else P[-1]
                    out = -f["U"][0] if end == 0 else f["U"][-1]
                    c = np.asarray(tree.query_ball_point(ep, STEP_UM + 2 * f["w1"]), np.int64)
                    k = np.zeros(0, np.int64)
                    if len(c):
                        d = x[c] - ep
                        s = d @ out
                        lat = np.linalg.norm(d - s[:, None] * out, axis=1)
                        c = c[(s > 0) & (s <= STEP_UM) & (lat <= 2 * f["w1"])]
                    if len(c):
                        k = np.minimum(np.searchsorted(ei, c), len(ei) - 1)
                        hit = ei[k] == c
                        c, k = c[hit], k[hit]
                        p = np.exp(-ee[k]) / (tot[c] + z0)
                        c, k = c[p >= p_grow], k[p >= p_grow]
                    if len(c) < 3:
                        f["grow"][end] = False
                        continue
                    wgt = iv[c]
                    newp = (x[c] * wgt[:, None]).sum(0) / wgt.sum()
                    if np.linalg.norm(newp - ep) < 0.25 * STEP_UM:
                        f["grow"][end] = False
                        continue
                    chain = P if end == 1 else P[::-1]
                    nu = newp - chain[max(0, len(chain) - 1 - back)]
                    nu /= max(np.linalg.norm(nu), 1e-12)
                    if nu @ out < 0.5:
                        f["grow"][end] = False
                        continue
                    a1 = f["A1"][-1] if end == 1 else f["A1"][0]
                    a1 = a1 - (a1 @ nu) * nu
                    a1 /= max(np.linalg.norm(a1), 1e-12)
                    nb = float(np.median(ecb[k]))
                    if end == 1:
                        f["P"] = np.vstack([P, newp])
                        f["U"] = np.vstack([f["U"], nu])
                        f["A1"] = np.vstack([f["A1"], a1])
                        f["B"] = np.r_[f["B"], nb]
                    else:
                        f["P"] = np.vstack([newp, P])
                        f["U"] = np.vstack([-nu, f["U"]])
                        f["A1"] = np.vstack([a1, f["A1"]])
                        f["B"] = np.r_[nb, f["B"]]
                    grown.setdefault(j, []).append(newp)
            if not grown:
                break
            for j, pts in grown.items():
                score(j, np.array(pts))

    joins = []
    grow()
    while True:
        pairs = ties()
        if not pairs:
            break
        for ja, ea, jb, eb, n in pairs:
            fa, fb = fibs[ja], fibs[jb]
            pa, ua, aa, ba = orient(fa, ea == 0)
            pb, ub, ab, bb = orient(fb, eb == 1)
            joins.append((fa["k"], fb["k"], n))
            fa["P"], fa["U"], fa["A1"], fa["B"] = np.vstack([pa, pb]), np.vstack([ua, ub]), np.vstack([aa, ab]), np.r_[ba, bb]
            fa["w1"], fa["w2"], fa["R"] = (fa["w1"] + fb["w1"]) / 2, (fa["w2"] + fb["w2"]) / 2, (fa["R"] + fb["R"]) / 2
            fa["members"] += fb["members"]
            fa["grow"] = [True, True]
            drop(ja)
            drop(jb)
            score(ja)
        grow()
    win = np.full(len(x), -1, np.int64)
    pw = np.zeros(len(x))
    recs = []
    for j, f in enumerate(fibs):
        if ent[j] is None:
            continue
        ei, ee, _ = ent[j]
        p = np.exp(-ee) / (tot[ei] + z0)
        m = p > pw[ei]
        pw[ei[m]] = p[m]
        win[ei[m]] = f["k"]
        keep = p >= 1.0 / 255
        recs.append((f["k"], ei[keep], np.round(p[keep] * 255).astype(np.uint8)))
    win = np.where(pw >= p_label, win, 0)
    owner = {m: f["k"] for j, f in enumerate(fibs) if ent[j] is not None for m in f["members"]}
    return pos, idx, win, pw.astype(np.float32), recs, [(owner.get(ka, ka), ka, kb, n) for ka, kb, n in joins]


def separate_elongated_cells(segmentation_mask, intensity_image, seed_mask, spacing, final, confidence, p_label, p_grow, none_pct):
    sp = np.asarray(spacing, float)
    nxt = 1
    records = []
    merged = []
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
        own, region, t, v, bri, ex = _prepare(lbl, sl, segmentation_mask, intensity_image, sp)
        pos, idx, win, pw, recs, joins = _vote(seedlab, own, region, t, v, bri, ex, sp, p_label, p_grow, none_pct)
        del t, v, bri, ex
        top = max([int(win.max())] + [r[0] for r in recs])
        new = np.zeros(top + 1, np.int64)
        ks = np.unique([r[0] for r in recs] + [int(k) for k in np.unique(win[win > 0])])
        new[ks] = np.arange(nxt, nxt + len(ks))
        nxt += len(ks)
        w = new[np.maximum(win[idx[own]], 0)]
        if np.any(w == 0):
            w = np.where(w == 0, nxt, w)
            nxt += 1
        out[own] = w
        cv = confidence[sl]
        cv[region] = np.maximum(cv[region], np.where(win > 0, pw, 0))
        base = np.array([s.start for s in sl])
        for k, ei, val in recs:
            if not len(ei):
                continue
            g = pos[ei] + base
            records.append((int(new[k]), np.ravel_multi_index(tuple(g.T), final.shape).astype(np.uint32), val, np.r_[g.min(0), g.max(0) + 1].astype(np.int32)))
        for k, ka, kb, n in joins:
            merged.append((int(new[k]) if k < len(new) else 0, ka, kb, n))
    return records, merged