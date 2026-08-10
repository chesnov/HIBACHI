"""
qc_render.py -- JPG verification images, one per Monte-Carlo draw.

Each image shows the stationary partner channel, the randomised channel for that
draw, and a red segment per randomised object marking the shortest
surface-to-surface distance the algorithm actually measured. The endpoints come
from the same distance transform that produced the statistic, so the picture
verifies the computation rather than illustrating it.

Rendered through matplotlib's Agg canvas directly rather than pyplot. pyplot
keeps global figure state and is not safe to drive from a worker thread, which is
exactly where these are produced.

3D NOTE
    A volume is shown as a maximum-intensity projection along Z, and the
    segments are projected with it. Projected lengths are therefore NOT the
    measured distances -- two objects far apart in Z can appear to touch. The
    measured value is printed next to each segment, and the caption says so.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Colours chosen to stay distinguishable in the common forms of colour blindness
# and to leave red free for the measurement segments alone.
COLOUR_PARTNER = "#22d3ee"      # cyan  -- stationary
COLOUR_RANDOM = "#f0abfc"       # orchid -- randomised
COLOUR_DOMAIN = "#94a3b8"       # slate -- domain outline
COLOUR_LINE = "#ef4444"         # red   -- measured distance
BACKGROUND = "#0b1020"


def _projected(mask: np.ndarray) -> np.ndarray:
    """2D view of a 2D or 3D mask (max projection along the first axis)."""
    arr = np.asarray(mask)
    return arr if arr.ndim == 2 else arr.max(axis=0)


def _outline(binary: np.ndarray) -> np.ndarray:
    """Boundary of a 2D binary mask, for drawing the domain cheaply."""
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

    Without this the tissue sits in a small island of background and the objects
    are too small to judge, which defeats the purpose of a QC image.
    """
    h, w = shape
    ys, xs = [], []
    for m in masks:
        if m is None or not np.any(m):
            continue
        coords = np.argwhere(m)
        ys += [coords[:, 0].min(), coords[:, 0].max()]
        xs += [coords[:, 1].min(), coords[:, 1].max()]
    if not ys:
        return slice(0, h), slice(0, w)
    my = int(max(4, (max(ys) - min(ys)) * margin_frac))
    mx = int(max(4, (max(xs) - min(xs)) * margin_frac))
    return (slice(max(0, min(ys) - my), min(h, max(ys) + my + 1)),
            slice(max(0, min(xs) - mx), min(w, max(xs) + mx + 1)))


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
                dpi: int = 150,
                quality: int = 88) -> str:
    """Write one QC JPG. Returns the path written."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    import matplotlib.patheffects as pe

    rnd = _projected(random_labels) > 0
    par = _projected(partner_labels) > 0 if partner_labels is not None else None
    dom = _projected(domain_mask) > 0 if domain_mask is not None else None

    sy_, sx_ = _crop_box([dom, par, rnd], rnd.shape)
    rnd = rnd[sy_, sx_]
    par = par[sy_, sx_] if par is not None else None
    dom = dom[sy_, sx_] if dom is not None else None
    oy, ox = sy_.start, sx_.start
    h, w = rnd.shape

    def _hex(c: str) -> np.ndarray:
        c = c.lstrip("#")
        return np.array([int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4)],
                        dtype=np.float32)

    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[:] = _hex(BACKGROUND)
    if dom is not None:
        rgb[dom] = _hex(BACKGROUND) * 2.1
        rgb[_outline(dom)] = _hex(COLOUR_DOMAIN)
    if par is not None:
        rgb[par] = _hex(COLOUR_PARTNER)
        both = rnd & par
        rgb[rnd & ~par] = _hex(COLOUR_RANDOM)
        # Overlap is biologically real, so it is blended rather than painted over.
        rgb[both] = 0.5 * (_hex(COLOUR_RANDOM) + _hex(COLOUR_PARTNER))
    else:
        rgb[rnd] = _hex(COLOUR_RANDOM)

    aspect = h / float(w)
    fig = Figure(figsize=(7.0, 7.0 * aspect + 1.5), dpi=dpi, facecolor=BACKGROUND)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.02, 0.14, 0.96, 0.78])
    ax.imshow(np.clip(rgb, 0, 1), interpolation="nearest", origin="upper")
    ax.set_facecolor(BACKGROUND)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    halo = [pe.withStroke(linewidth=2.4, foreground="#0b1020")]
    finite = [q for q in pairs if q.get("p1") is not None
              and np.isfinite(q.get("distance_um", np.inf))]
    order = sorted(finite, key=lambda q: -q["distance_um"])

    for k, pr in enumerate(order):
        y0, x0 = _yx(pr["p0"]); y1, x1 = _yx(pr["p1"])
        y0, x0, y1, x1 = y0 - oy, x0 - ox, y1 - oy, x1 - ox
        ax.plot([x0, x1], [y0, y1], color=COLOUR_LINE, linewidth=1.5,
                solid_capstyle="round", zorder=4, path_effects=halo)
        # Endpoint dots: a touching pair has a sub-pixel segment, which would
        # otherwise be invisible and look like a missing measurement.
        ax.plot([x0, x1], [y0, y1], linestyle="none", marker="o",
                markersize=2.0, color=COLOUR_LINE, zorder=5,
                markeredgecolor="#0b1020", markeredgewidth=0.4)

        if annotate_distances and k < max_annotations:
            dy, dx = y1 - y0, x1 - x0
            length = float(np.hypot(dy, dx)) or 1.0
            # Offset perpendicular to the segment so the label never covers the
            # thing being measured.
            px, py = -dy / length, dx / length
            off = max(7.0, 0.012 * max(h, w))
            ax.annotate(f"{pr['distance_um']:.2f}",
                        xy=((x0 + x1) / 2.0 + px * off,
                            (y0 + y1) / 2.0 + py * off),
                        color="#fecaca", fontsize=5.8, zorder=6,
                        ha="center", va="center", path_effects=halo)

    sy, sx = float(spacing[-2]), float(spacing[-1])
    span = w * sx
    target = 10.0 ** np.floor(np.log10(max(span * 0.2, 1e-6)))
    for mult in (5.0, 2.0, 1.0):
        if target * mult <= span * 0.30:
            target *= mult
            break
    bar_px = target / sx
    ax.plot([w * 0.03, w * 0.03 + bar_px], [h * 0.975, h * 0.975],
            color="white", linewidth=3.0, zorder=7,
            path_effects=[pe.withStroke(linewidth=5.0, foreground="#0b1020")])
    ax.annotate(f"{target:g} µm", xy=(w * 0.03 + bar_px / 2.0, h * 0.955),
                color="white", fontsize=8, ha="center", va="bottom", zorder=7,
                path_effects=halo)

    handles = [Line2D([], [], color=COLOUR_RANDOM, lw=7,
                      label="randomised objects")]
    if par is not None:
        handles.append(Line2D([], [], color=COLOUR_PARTNER, lw=7,
                              label="stationary partner (fixed)"))
    handles.append(Line2D([], [], color=COLOUR_LINE, lw=1.8, marker="o",
                          markersize=3.5,
                          label="measured nearest distance (µm)"))
    if dom is not None:
        handles.append(Line2D([], [], color=COLOUR_DOMAIN, lw=1.8,
                              label="domain boundary"))
    # Legend below the image: inside the axes it covered the data.
    leg = fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5,
                     framealpha=0.0, bbox_to_anchor=(0.5, 0.045))
    for text in leg.get_texts():
        text.set_color("#e2e8f0")

    if title:
        fig.text(0.5, 0.965, title, color="white", fontsize=11.5,
                 ha="center", va="top")
    caption = subtitle
    if random_labels.ndim == 3:
        caption = ((caption + "   ·   ") if caption else "") + \
            "Z max-projection: drawn lengths are projected, printed values are " \
            "the true 3D distances"
    if caption:
        fig.text(0.5, 0.115, caption, color="#a8b3c7", fontsize=7.5,
                 ha="center", va="top")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    canvas.print_figure(path, facecolor=BACKGROUND, format="jpg",
                        pil_kwargs={"quality": int(quality)})
    return path


def render_observed(path: str,
                   observed_labels: np.ndarray,
                   partner_labels: Optional[np.ndarray],
                   pairs: Sequence[Dict[str, Any]],
                   spacing: Sequence[float],
                   domain_mask: Optional[np.ndarray] = None,
                   sample: str = "") -> str:
    """The real data, drawn identically, as the reference to compare draws to.

    Without this the draws have nothing to be judged against, and 'does the
    randomisation look right' is unanswerable.
    """
    return render_draw(
        path, observed_labels, partner_labels, pairs, spacing,
        domain_mask=domain_mask,
        title=f"{sample} — OBSERVED (real data)",
        subtitle="real positions; compare the draw images against this")


def qc_paths(out_dir: str, sample: str) -> str:
    """Directory for one sample's QC images."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(sample))
    return os.path.join(out_dir, "qc_images", safe)


def estimate_qc_output(n_samples: int, n_images_each: int,
                       kb_each: int = 190) -> Tuple[int, float]:
    """(file count, estimated megabytes) so the caller can warn before writing."""
    n = int(n_samples) * (int(n_images_each) + 1)      # +1 for the observed image
    return n, n * kb_each / 1024.0
