"""
qc_render.py -- JPG verification images, one per Monte-Carlo draw.

Each image shows the stationary partner channel, the randomised channel for that
draw, and a red segment per randomised object marking the shortest
surface-to-surface distance the algorithm actually measured. The endpoints come
from the same distance transform that produced the statistic, so the picture
verifies the computation rather than illustrating it.

WHY PIL AND NOT MATPLOTLIB
    matplotlib is not in HIBACHI's environment.yml -- it would arrive only as a
    transitive dependency of seaborn, at an unpinned version. Depending on that
    for a feature is how you end up with empty output directories on someone
    else's machine. Pillow is imported directly by turntable.py, so it is a
    dependency the project already relies on, it needs no backend selection, and
    it has no global figure state (which also makes it safe to drive from a
    worker thread).

3D NOTE
    A volume is shown as a maximum-intensity projection along Z, and the
    segments are projected with it. Projected lengths are therefore NOT the
    measured distances -- two objects far apart in Z can appear to touch. The
    measured value is printed beside each segment, and the caption says so.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Red is reserved for the measurement segments alone.
COLOUR_PARTNER = (34, 211, 238)      # cyan   -- stationary
COLOUR_RANDOM = (240, 171, 252)      # orchid -- randomised
COLOUR_OVERLAP = (137, 191, 245)     # blend  -- randomised over partner
COLOUR_DOMAIN = (148, 163, 184)      # slate  -- domain outline
COLOUR_DOMAIN_FILL = (23, 32, 60)
COLOUR_LINE = (239, 68, 68)          # red    -- measured distance
COLOUR_TEXT = (226, 232, 240)
COLOUR_DIST_TEXT = (254, 202, 202)
BACKGROUND = (11, 16, 32)

TARGET_WIDTH = 1100
HEADER_H = 44
FOOTER_H = 74

_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
    "C:\\Windows\\Fonts\\segoeui.ttf",
)


def _font(size: int):
    """A usable font at the requested size, degrading rather than failing."""
    from PIL import ImageFont
    for cand in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(cand, size)
        except Exception:
            continue
    try:                                  # Pillow >= 9.2 can size the default
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _projected(mask: np.ndarray) -> np.ndarray:
    """2D view of a 2D or 3D mask (max projection along the first axis)."""
    arr = np.asarray(mask)
    return arr if arr.ndim == 2 else arr.max(axis=0)


def _outline(binary: np.ndarray) -> np.ndarray:
    from scipy import ndimage
    return binary & ~ndimage.binary_erosion(
        binary, structure=ndimage.generate_binary_structure(2, 1))


def _yx(point: Sequence[int]) -> Tuple[float, float]:
    """Drop the Z index so 3D points plot on the projection."""
    p = tuple(point)
    return (float(p[-2]), float(p[-1]))


def _crop_box(masks: Sequence[Optional[np.ndarray]], shape: Tuple[int, int],
              margin_frac: float = 0.04) -> Tuple[slice, slice]:
    """Bounding box of everything worth showing, plus a margin.

    Without this the tissue sits in an island of background and the objects are
    too small to judge, which defeats the purpose of a QC image.
    """
    h, w = shape
    ys: List[int] = []
    xs: List[int] = []
    for m in masks:
        if m is None or not np.any(m):
            continue
        c = np.argwhere(m)
        ys += [int(c[:, 0].min()), int(c[:, 0].max())]
        xs += [int(c[:, 1].min()), int(c[:, 1].max())]
    if not ys:
        return slice(0, h), slice(0, w)
    my = int(max(4, (max(ys) - min(ys)) * margin_frac))
    mx = int(max(4, (max(xs) - min(xs)) * margin_frac))
    return (slice(max(0, min(ys) - my), min(h, max(ys) + my + 1)),
            slice(max(0, min(xs) - mx), min(w, max(xs) + mx + 1)))


def _nice_scale(span_um: float) -> float:
    """A round scale-bar length under ~30% of the field width."""
    if span_um <= 0:
        return 1.0
    base = 10.0 ** np.floor(np.log10(max(span_um * 0.2, 1e-9)))
    for mult in (5.0, 2.0, 1.0):
        if base * mult <= span_um * 0.30:
            return float(base * mult)
    return float(base)


def _text_outlined(draw, xy, text, font, fill, halo=BACKGROUND):
    """Text with a 1px dark outline, so it reads over any mask colour."""
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=halo, anchor="mm")
    draw.text((x, y), text, font=font, fill=fill, anchor="mm")


def render_draw(path: str,
                random_labels: np.ndarray,
                partner_labels: Optional[np.ndarray],
                pairs: Sequence[Dict[str, Any]],
                spacing: Sequence[float],
                domain_mask: Optional[np.ndarray] = None,
                title: str = "",
                subtitle: str = "",
                observed_labels: Optional[np.ndarray] = None,
                annotate_distances: bool = True,
                max_annotations: int = 40,
                target_width: int = TARGET_WIDTH,
                quality: int = 88) -> str:
    """Write one QC JPG. Returns the path written."""
    from PIL import Image, ImageDraw

    rnd = _projected(random_labels) > 0
    par = _projected(partner_labels) > 0 if partner_labels is not None else None
    dom = _projected(domain_mask) > 0 if domain_mask is not None else None

    sy_, sx_ = _crop_box([dom, par, rnd], rnd.shape)
    rnd = rnd[sy_, sx_]
    par = par[sy_, sx_] if par is not None else None
    dom = dom[sy_, sx_] if dom is not None else None
    oy, ox = sy_.start, sx_.start
    h, w = rnd.shape

    # ---- base image from the masks ----------------------------------------
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[:] = BACKGROUND
    if dom is not None:
        base[dom] = COLOUR_DOMAIN_FILL
        base[_outline(dom)] = COLOUR_DOMAIN
    if par is not None:
        base[par] = COLOUR_PARTNER
        # Overlap is biologically real, so it gets its own colour rather than
        # being painted over.
        base[rnd & ~par] = COLOUR_RANDOM
        base[rnd & par] = COLOUR_OVERLAP
    else:
        base[rnd] = COLOUR_RANDOM

    scale = max(1, int(round(target_width / float(max(w, 1)))))
    iw, ih = w * scale, h * scale
    # NEAREST on purpose: these are voxel masks and should look like it, rather
    # than being smoothed into something that implies more precision than exists.
    img = Image.fromarray(base).resize((iw, ih), Image.NEAREST)

    canvas = Image.new("RGB", (iw, ih + HEADER_H + FOOTER_H), BACKGROUND)
    canvas.paste(img, (0, HEADER_H))
    draw = ImageDraw.Draw(canvas)

    f_title = _font(19)
    f_small = _font(13)
    f_dist = _font(12)

    # ---- measurement segments --------------------------------------------
    finite = [q for q in pairs if q.get("p1") is not None
              and np.isfinite(q.get("distance_um", np.inf))]
    order = sorted(finite, key=lambda q: -q["distance_um"])

    def _pt(p):
        y, x = _yx(p)
        return ((x - ox) * scale + scale / 2.0,
                (y - oy) * scale + scale / 2.0 + HEADER_H)

    for k, pr in enumerate(order):
        x0, y0 = _pt(pr["p0"])
        x1, y1 = _pt(pr["p1"])
        # Dark halo under the line so red reads over cyan.
        draw.line([(x0, y0), (x1, y1)], fill=BACKGROUND, width=5)
        draw.line([(x0, y0), (x1, y1)], fill=COLOUR_LINE, width=2)
        # Endpoint dots: a touching pair has a sub-pixel segment that would
        # otherwise be invisible and look like a missing measurement.
        for cx, cy in ((x0, y0), (x1, y1)):
            draw.ellipse([cx - 2.6, cy - 2.6, cx + 2.6, cy + 2.6],
                         fill=COLOUR_LINE, outline=BACKGROUND)

        if annotate_distances and k < max_annotations:
            dy, dx = y1 - y0, x1 - x0
            length = float(np.hypot(dy, dx)) or 1.0
            # Offset perpendicular to the segment so the label never covers the
            # thing being measured.
            off = max(11.0, 0.014 * max(iw, ih))
            mx = (x0 + x1) / 2.0 - dy / length * off
            my = (y0 + y1) / 2.0 + dx / length * off
            mx = float(np.clip(mx, 18, max(19, iw - 18)))
            my = float(np.clip(my, HEADER_H + 10, HEADER_H + ih - 10))
            _text_outlined(draw, (mx, my), f"{pr['distance_um']:.2f}",
                           f_dist, COLOUR_DIST_TEXT)

    # ---- scale bar --------------------------------------------------------
    sx_um = float(spacing[-1])
    bar_um = _nice_scale(w * sx_um)
    bar_px = bar_um / sx_um * scale if sx_um > 0 else 0.0
    by = HEADER_H + ih - 22
    bx = 20
    if bar_px > 2:
        draw.line([(bx, by), (bx + bar_px, by)], fill=BACKGROUND, width=7)
        draw.line([(bx, by), (bx + bar_px, by)], fill=(255, 255, 255), width=3)
        _text_outlined(draw, (bx + bar_px / 2.0, by - 13), f"{bar_um:g} um",
                       f_small, (255, 255, 255))

    # ---- header and footer ------------------------------------------------
    if title:
        draw.text((iw / 2.0, HEADER_H / 2.0), title, font=f_title,
                  fill=(255, 255, 255), anchor="mm")

    caption = subtitle
    if np.asarray(random_labels).ndim == 3:
        caption = ((caption + "   |   ") if caption else "") + \
            "Z max-projection: drawn lengths are projected; printed values are " \
            "the true 3D distances"
    if caption:
        draw.text((iw / 2.0, HEADER_H + ih + 16), caption, font=f_small,
                  fill=(168, 179, 199), anchor="mm")

    entries = [(COLOUR_RANDOM, "randomised objects"),
               (COLOUR_LINE, "measured nearest distance (um)")]
    if par is not None:
        entries.insert(1, (COLOUR_PARTNER, "stationary partner (fixed)"))
        entries.append((COLOUR_OVERLAP, "overlap"))
    if dom is not None:
        entries.append((COLOUR_DOMAIN, "domain boundary"))

    widths = [draw.textlength(lbl, font=f_small) + 34 for _, lbl in entries]
    x = max(10.0, (iw - sum(widths)) / 2.0)
    ly = HEADER_H + ih + 46
    for (colour, label), wd in zip(entries, widths):
        draw.rectangle([x, ly - 6, x + 18, ly + 6], fill=colour)
        draw.text((x + 24, ly), label, font=f_small, fill=COLOUR_TEXT,
                  anchor="lm")
        x += wd

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    canvas.save(path, format="JPEG", quality=int(quality), optimize=True)
    return path


def render_observed(path: str,
                    observed_labels: np.ndarray,
                    partner_labels: Optional[np.ndarray],
                    pairs: Sequence[Dict[str, Any]],
                    spacing: Sequence[float],
                    domain_mask: Optional[np.ndarray] = None,
                    sample: str = "") -> str:
    """The real data, drawn identically, as the reference to compare draws to.

    Without this the draws have nothing to be judged against, and "does the
    randomisation look right" is unanswerable.
    """
    return render_draw(
        path, observed_labels, partner_labels, pairs, spacing,
        domain_mask=domain_mask,
        title=f"{sample} - OBSERVED (real data)",
        subtitle="real positions; compare the draw images against this")


def qc_paths(out_dir: str, sample: str) -> str:
    """Directory for one sample's QC images."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(sample))
    return os.path.join(out_dir, "qc_images", safe)


def estimate_qc_output(n_samples: int, n_images_each: int,
                       kb_each: int = 120) -> Tuple[int, float]:
    """(file count, estimated megabytes) so the caller can warn before writing."""
    n = int(n_samples) * (int(n_images_each) + 1)      # +1 for the observed image
    return n, n * kb_each / 1024.0


def self_test(tmp_dir: str) -> Tuple[bool, str]:
    """Render one tiny image to prove the backend works.

    Run before a batch so a rendering problem is reported up front with its
    traceback, instead of being discovered afterwards as a directory full of
    nothing.
    """
    try:
        lab = np.zeros((40, 50), dtype=np.int32)
        lab[8:14, 8:14] = 1
        par = np.zeros((40, 50), dtype=np.int32)
        par[20:30, 25:40] = 1
        dom = np.ones((40, 50), dtype=bool)
        os.makedirs(tmp_dir, exist_ok=True)
        path = os.path.join(tmp_dir, "_qc_selftest.jpg")
        render_draw(path, lab, par,
                    [{"label": 1, "distance_um": 1.5, "p0": (13, 13),
                      "p1": (20, 25)}],
                    (0.25, 0.25), domain_mask=dom, title="self test",
                    target_width=260)
        ok = os.path.isfile(path) and os.path.getsize(path) > 0
        try:
            os.remove(path)
        except OSError:
            pass
        return ok, "" if ok else "the renderer produced no file"
    except Exception as exc:
        import traceback
        return False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"