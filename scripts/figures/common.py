"""Shared style, palette and build checks for the figures.

Every figure script imports this module, calls `setup(scale)` with the scale at which the paper prints the figure
(FULL for a 180 mm figure placed at \textwidth), draws, and calls `finish(fig, name)`, which
  1. draws the canvas and fails if any two text bounding boxes overlap, any text leaves the figure, or a legend or
     in-axes text covers a data marker, or a legend covers a plotted line,
  2. fails if any visible text, as printed, is smaller than MIN_PT, or any superscript or subscript smaller than
     MIN_SCRIPT_PT (sizes are measured glyph by glyph through the mathtext parser),
  3. writes figures/<name>.pdf and figures/<name>.png (300 dpi) with fixed metadata, so reruns are byte-identical.
"""
import itertools
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.text import Text  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "figures")
MM = 1 / 25.4  # inches per mm
WIDTH_MM = 180  # full figure width
TEXTWIDTH_MM = 455.24 / 72.27 * 25.4  # \textwidth of paper/main.tex (455.24 pt = 160.0 mm)
FULL = TEXTWIDTH_MM / WIDTH_MM  # print scale of a WIDTH_MM figure placed at \textwidth (0.889)
MIN_PT = 5.0  # smallest ordinary text as printed
MIN_SCRIPT_PT = 4.0  # smallest superscript or subscript as printed
SCRIPT = 0.7  # mathtext superscript/subscript size relative to the base size
SCALE = 1.0  # print scale of the figure being built, set by setup()
DPI = 300

# Okabe-Ito colour-blind-safe palette; blue is reserved for our predictor in every figure.
OURS = "#0072B2"
OURS_LIGHT = "#56B4E9"
BLACK = "#000000"
GREY = "#7F7F7F"
LIGHT = "#BDBDBD"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"

# DataDecide corpus families
FAMILY = {
    "dolma": ("Dolma", VERMILION),
    "dclm": ("DCLM", "#44AA99"),
    "dclm_dolma_mix": ("DCLM-Dolma mix", PURPLE),
    "falcon": ("Falcon", ORANGE),
    "fineweb": ("FineWeb", "#999933"),
    "c4": ("C4", BLACK),
}
# Settings of the tokenizer comparison
SETTING = {
    "ts": ("TinyStories, small", ORANGE),
    "fw": ("FineWeb-Edu, small", GREEN),
    "ms": ("FineWeb-Edu, 100M", PURPLE),
}

FONT = {"family": "sans-serif", "sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]}
BASE_PT = 6.5
SMALL_PT = 5.5
LETTER_PT = 8.0


def setup(scale=1.0, script_ticks=False):
    """scale: printed size / authored size. Below 1, SMALL_PT is raised so that it prints at MIN_PT or more, and with
    script_ticks (tick labels such as 2^17) the tick labels are raised so that their exponents print at MIN_SCRIPT_PT
    or more. At scale 1 every size is the original one."""
    global SCALE, SMALL_PT
    SCALE = scale
    SMALL_PT = max(5.5, math.ceil(MIN_PT / scale * 10 - 1e-9) / 10)
    tick = max(6.0, math.ceil(MIN_SCRIPT_PT / scale / SCRIPT * 10 - 1e-9) / 10) if script_ticks else 6.0
    plt.rcParams.update({
        "font.family": FONT["family"], "font.sans-serif": FONT["sans-serif"],
        "font.size": BASE_PT, "axes.labelsize": BASE_PT, "axes.titlesize": BASE_PT,
        "xtick.labelsize": tick, "ytick.labelsize": tick, "legend.fontsize": 6.0,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "axes.labelpad": 2, "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.hashsalt": "fig",
        "savefig.dpi": DPI, "figure.dpi": DPI, "path.simplify": False,
        "axes.unicode_minus": True, "mathtext.default": "regular",
    })


def keep_tick_density(ax, which="y", fontsize=6.0):
    """Pin the automatic major ticks of a linear axis to those matplotlib chooses for tick labels of `fontsize` pt, so
    that raising the tick label size for print does not thin the ticks (AutoLocator spaces ticks by label size)."""
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    bb = ax.get_position()
    w, h = ax.figure.get_size_inches()
    length = (bb.height * h if which == "y" else bb.width * w) * 72
    n = int(np.clip(np.floor(length / (fontsize * (2 if which == "y" else 3))), 1, 9))
    (ax.yaxis if which == "y" else ax.xaxis).set_major_locator(MaxNLocator(nbins=n, steps=[1, 2, 2.5, 5, 10]))


def letters(fig, pairs, y=None):
    """Bold lower-case panel letters at the left edge of each axes' tight box, all on one baseline (the highest
    axes top, or y in figure coordinates)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    tops = [ax.get_position().y1 for ax, _ in pairs]
    y = max(tops) + 0.01 if y is None else y
    for ax, s in pairs:
        bb = ax.get_tightbbox(r).transformed(inv)
        fig.text(max(bb.x0, 0.002), y, s, fontsize=LETTER_PT, fontweight="bold", va="bottom", ha="left")


def _texts(fig):
    """Visible, non-empty text that is actually drawn (tick labels outside the view interval are not drawn)."""
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = (hi - lo) * 1e-9
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - eps <= tick.get_loc() <= hi + eps):
                    hidden.update({id(tick.label1), id(tick.label2)})
    out = []
    for t in fig.findobj(Text):
        if id(t) in hidden or not t.get_visible() or not t.get_text().strip():
            continue
        out.append(t)
    return out


def _glyph_sizes(t):
    """(ordinary, script) glyph sizes in pt of one text item as authored; mathtext is parsed glyph by glyph."""
    s, fs = t.get_text(), t.get_fontsize()
    if not t._preprocess_math(s)[1]:
        return [fs], []
    from matplotlib.mathtext import MathTextParser
    g = [x[1] for x in MathTextParser("path").parse(s, dpi=72, prop=t.get_fontproperties()).glyphs]
    return [x for x in g if x >= fs * 0.99], [x for x in g if x < fs * 0.99]


def check(fig, name):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = _texts(fig)
    small, small_script, printed = [], [], []
    for t in texts:
        o, sc = _glyph_sizes(t)
        printed += [x * SCALE for x in o]
        if o and min(o) * SCALE < MIN_PT - 1e-6:
            small.append((t.get_text(), round(min(o) * SCALE, 2)))
        if sc and min(sc) * SCALE < MIN_SCRIPT_PT - 1e-6:
            small_script.append((t.get_text(), round(min(sc) * SCALE, 2)))
    if small:
        raise SystemExit(f"[{name}] text below {MIN_PT} pt as printed (scale {SCALE:.3f}): {small}")
    if small_script:
        raise SystemExit(f"[{name}] superscript/subscript below {MIN_SCRIPT_PT} pt as printed: {small_script}")
    boxes = []
    for t in texts:
        bb = t.get_window_extent(r)
        if bb.width <= 0 or bb.height <= 0:
            continue
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            raise SystemExit(f"[{name}] text outside the figure: {t.get_text()!r}")
        boxes.append((t, bb))
    bad = []
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        if b1.overlaps(b2):
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 0.5 and iy > 0.5:  # more than half a pixel at 300 dpi in both directions
                bad.append((t1.get_text(), t2.get_text()))
    if bad:
        raise SystemExit(f"[{name}] overlapping text: {bad}")
    # data markers (scatter points, line markers) must not sit under a legend or in-axes text
    import numpy as np
    from matplotlib.collections import PathCollection
    from matplotlib.lines import Line2D
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None] + list(fig.legends)
    for ax in fig.axes:
        pts = []
        for ln in ax.get_lines():
            if ln.get_marker() not in (None, "None", "", " ") and ln.get_visible():
                pts.append(ln.get_transform().transform(np.column_stack(ln.get_data())))
        for c in ax.collections:
            if isinstance(c, PathCollection) and c.get_visible() and len(c.get_offsets()):
                pts.append(c.get_offset_transform().transform(c.get_offsets()))
        if not pts:
            continue
        P = np.vstack(pts)
        covers = [(t.get_text(), t.get_window_extent(r)) for t in ax.texts if t.get_visible()]
        covers += [("legend", lg.get_window_extent(r)) for lg in legends]
        for lab, bb in covers:
            inside = (P[:, 0] > bb.x0) & (P[:, 0] < bb.x1) & (P[:, 1] > bb.y0) & (P[:, 1] < bb.y1)
            if inside.any():
                raise SystemExit(f"[{name}] data marker under {lab!r}")
    # no legend (axes or figure legend) may cover a drawn line: lines are sampled every display pixel
    for ax in fig.axes:
        abox = ax.get_window_extent(r)
        for ln in ax.get_lines():
            if not ln.get_visible() or ln.get_linestyle() in ("None", "none", "", " ") or len(ln.get_xydata()) < 2:
                continue
            V = ln.get_transform().transform(ln.get_xydata())
            V = V[np.isfinite(V).all(axis=1)]
            seg = [np.linspace(a, b, max(2, int(np.hypot(*(b - a))) + 1)) for a, b in zip(V[:-1], V[1:])]
            if not seg:
                continue
            S = np.vstack(seg)
            if ln.get_clip_on():
                S = S[(S[:, 0] >= abox.x0) & (S[:, 0] <= abox.x1) & (S[:, 1] >= abox.y0) & (S[:, 1] <= abox.y1)]
            for lg in legends:
                bb = lg.get_window_extent(r)
                if ((S[:, 0] > bb.x0) & (S[:, 0] < bb.x1) & (S[:, 1] > bb.y0) & (S[:, 1] < bb.y1)).any():
                    raise SystemExit(f"[{name}] legend covers a plotted line ({ln.get_label()!r})")
            hw = ln.get_linewidth() / 2 * fig.dpi / 72  # half the stroke width
            for t in ax.texts:  # a label inside an axes that touches one of its lines is reported, not fatal
                if not t.get_visible() or not t.get_text().strip():
                    continue
                bb = t.get_window_extent(r)
                if ((S[:, 0] > bb.x0 - hw) & (S[:, 0] < bb.x1 + hw) & (S[:, 1] > bb.y0 - hw) & (S[:, 1] < bb.y1 + hw)).any():
                    print(f"[{name}] warning: text {t.get_text()!r} touches a plotted line")
    sizes = sorted({round(t.get_fontsize(), 2) for t in texts})
    words = sum(len(t.get_text().replace("$", "").split()) for t in texts)
    return dict(n_text=len(texts), min_pt=sizes[0], sizes=sizes, words=words, min_printed=round(min(printed), 2))


def finish(fig, name):
    info = check(fig, name)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), metadata={"CreationDate": None, "ModDate": None,
                                                             "Creator": "matplotlib", "Producer": "matplotlib"})
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=DPI, metadata={"Software": "matplotlib"})
    w, h = fig.get_size_inches() / MM
    print(f"{name}: {w:.0f} x {h:.0f} mm, {info['n_text']} text items, {info['words']} words, "
          f"min {info['min_pt']} pt, sizes {info['sizes']}; printed at scale {SCALE:.3f}, min {info['min_printed']} pt; "
          f"overlap and size checks passed")
    plt.close(fig)
    return info


def end_labels(ax, items, x_axes=1.03, gap_pt=0.6, fontsize=None):
    """Label line ends in a gutter right of the axes, spread vertically so no two labels overlap.
    items: (x_end, y_end, text, colour) in data coordinates; a thin leader joins each end to its label."""
    fontsize = SMALL_PT if fontsize is None else fontsize
    fig = ax.figure
    fig.canvas.draw()
    h = (1.25 * fontsize + gap_pt) * fig.dpi / 72  # text boxes are about 1.2 em tall
    items = sorted(items, key=lambda t: ax.transData.transform((t[0], t[1]))[1])
    y = [ax.transData.transform((x, yy))[1] for x, yy, _, _ in items]
    for _ in range(2000):  # push apart symmetric pairs until every gap is at least h
        moved = False
        for i in range(1, len(y)):
            d = y[i] - y[i - 1]
            if d < h - 1e-6:
                s = (h - d) / 2
                y[i - 1] -= s
                y[i] += s
                moved = True
        if not moved:
            break
    toax = ax.transAxes.inverted()
    for (x, yy, s, col), yd in zip(items, y):
        # leader and label in axes coordinates (an Annotation's extent would include its arrow)
        xa, ya = toax.transform(ax.transData.transform((x, yy)))
        yl = toax.transform((0, yd))[1]
        ax.plot([xa, x_axes - 0.008], [ya, yl], transform=ax.transAxes, color=LIGHT, lw=0.35, clip_on=False)
        ax.text(x_axes, yl, s, transform=ax.transAxes, fontsize=fontsize, color=col, va="center", ha="left")


def point_labels(ax, xs, ys, labels, colors, fontsize=None, avoid=None, avoid_lines=None):
    """Place one label beside each scatter point, trying offsets in a fixed order and keeping the first that
    overlaps no placed label, no point and stays inside the axes. Deterministic."""
    import numpy as np
    fontsize = SMALL_PT if fontsize is None else fontsize
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.column_stack([xs, ys]))
    if avoid is not None:
        pts = np.vstack([pts, ax.transData.transform(avoid)])
    rad = 2.6 * fig.dpi / 72
    abox = ax.get_window_extent(r)
    offs = [(4, 0), (-4, 0), (3, 3), (-3, 3), (3, -3), (-3, -3), (0, 5), (0, -5), (7, 6), (-7, 6), (7, -6),
            (-7, -6), (0, 9), (0, -9), (10, 0), (-10, 0), (6, 11), (-6, 11), (6, -11), (-6, -11)]
    lpts = np.zeros((0, 2))
    if avoid_lines is not None:  # data-space segments (x0, y0, x1, y1) that labels must not cross
        lpts = np.vstack([ax.transData.transform(np.column_stack([np.linspace(x0, x1, 400), np.linspace(y0, y1, 400)]))
                          for x0, y0, x1, y1 in avoid_lines])
    placed = []
    for i, (x, y, s, c) in enumerate(zip(xs, ys, labels, colors)):
        for dx, dy in offs:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
            t = ax.annotate(s, xy=(x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                            color=c, ha=ha, va=va)
            bb = t.get_window_extent(r).padded(0.6)
            ok = (bb.x0 >= abox.x0 and bb.x1 <= abox.x1 and bb.y0 >= abox.y0 and bb.y1 <= abox.y1
                  and not any(bb.overlaps(p) for p in placed)
                  and not any((bb.x0 - rad < px < bb.x1 + rad) and (bb.y0 - rad < py < bb.y1 + rad) for px, py in pts)
                  and not any(bb.x0 < px < bb.x1 and bb.y0 < py < bb.y1 for px, py in lpts))
            if ok:
                placed.append(bb)
                break
            t.remove()
        else:
            raise SystemExit(f"point_labels: no free position for {s!r}")
