"""Shared matplotlib style and render-time checks for the figures of the article.

Figures are built at the journal's text width (32 pc = 134.96 mm), so a font size in pt is the printed size. Layout
coordinates are in a 160-unit system (1 unit = 0.8435 mm). White background, thin dark axes, no grids; data lines
about 1 pt; a palette of blue (ours) plus two or three Okabe-Ito hues, each with one meaning within a figure.

save() draws the figure, then fails if any two text boxes or legend boxes overlap, any text leaves the canvas, any
text enters another panel's axes box, or any text is below MIN_PT; it writes PNG and PDF and a JSON dump of every
plotted value (lines, markers, error bars, bars, fills), which verify/check_replay.py compares with expected/.
"""
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection, PathCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
import matplotlib._mathtext as _mt  # noqa: E402

# Superscripts at 0.86 of the base size (matplotlib default 0.7), so a 7 pt '2^14' tick prints its exponent at 6.0 pt.
SUP = 0.86
_mt.SHRINK_FACTOR = SUP

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repository root
OUT = os.path.join(ROOT, "figures", "cl")
TEXTWIDTH_MM = 384 / 72.27 * 25.4   # CL \textwidth = 32 pc = 384 TeX pt
WIDTH_MM = 160.0                    # layout units (the v44 coordinate system)
MM = (TEXTWIDTH_MM / WIDTH_MM) / 25.4  # inches per layout unit

# One palette for Figs 1-5. Blue is ours everywhere; greys are context; the other hues are Okabe-Ito
# (colour-blind safe), each with one meaning within a figure.
ACC = "#1F5FBF"     # ours
ACC_M = "#5B9BE0"   # mid ours (one pilot + shape; second-tier ours)
ACC_L = "#9CC0EE"   # light ours (forecast band)
INK = "#1A1A1A"
G1 = "#4D4D4D"      # dark grey: baselines, rivals, proxies
G2 = "#8C8C8C"      # mid grey: reference lines
G3 = "#CFCFCF"      # light grey: context (bars below threshold)
VERM = "#D55E00"
ORNG = "#E69F00"
GREEN = "#009E73"
SKY = "#56B4E9"
RPUR = "#CC79A7"
PURP = "#7A4FA3"
WINE = "#882255"
LW = 1.2            # primary data line (pt)
LW2 = 0.9           # secondary data line, guides
MS_PT = 4.5         # marker diameter (pt) for data points (1.6 mm); scatter size s = MS_PT**2
MS2_PT = 3.6        # dense points (swarms, many runs)
TICK, LAB, NOTE, LET = 6.5, 7.5, 7.0, 9.0   # printed pt (v44 printed in CL at 5.06, 5.90, 5.48, 7.59)
LOG2 = 7.5                                  # 2^k tick labels; exponent prints at 0.86 x 7.5 = 6.45 pt
MIN_PT = 6.0

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": TICK, "axes.labelsize": LAB, "axes.titlesize": LAB, "xtick.labelsize": TICK, "ytick.labelsize": TICK,
    "legend.fontsize": TICK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "axes.labelpad": 3.0, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": LW,
    "lines.markersize": MS_PT, "patch.linewidth": 0.5, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def figure(h_mm, w_mm=WIDTH_MM):
    """h_mm, w_mm in layout units; w_mm < 160 makes a narrower figure (placed at w_mm/160 of \\textwidth)."""
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def set_type(tick, lab, note, log2):
    """Larger type for one figure (panel_cl19): call before `from style import *` so the script picks it up."""
    global TICK, LAB, NOTE, LOG2
    TICK, LAB, NOTE, LOG2 = tick, lab, note, log2
    plt.rcParams.update({"font.size": tick, "axes.labelsize": lab, "axes.titlesize": lab, "xtick.labelsize": tick,
                         "ytick.labelsize": tick, "legend.fontsize": tick})


BIG = (7.0, 8.0, 7.5, 8.2)   # tick, axis label, note, 2^k ticks (exponent 0.86 x 8.2 = 7.05 pt)


def axes_mm(fig, x, y, w, h, **kw):
    """Axes at (x, y) mm from the top-left corner, w x h mm."""
    W, H = fig.get_size_inches() / MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def letter(fig, s, x_mm, y_mm):
    W, H = fig.get_size_inches() / MM
    return fig.text(x_mm / W, 1 - y_mm / H, s, fontsize=LET, fontweight="bold", va="top", ha="left")


def legend(ax, **kw):
    """Shared in-axes legend style: no frame, short handles, tight spacing."""
    base = dict(frameon=False, handletextpad=0.4, borderaxespad=0.3, borderpad=0.1, labelspacing=0.35,
                handlelength=1.2, fontsize=TICK, markerscale=1.0)
    base.update(kw)
    return ax.legend(**base)


def keyrow(fig, x_mm, y_mm, items, gap_mm=3.5):
    """Shared figure-level key: items = [(kind, style, label)], kind in line/dot/ring; laid out left to right."""
    W, H = fig.get_size_inches() / MM
    k = fig.add_axes([0, 0, 1, 1], zorder=-1); k.set_xlim(0, W); k.set_ylim(H, 0); k.axis("off")
    r = fig.canvas.get_renderer()
    x = x_mm
    for kind, st, lab in items:
        if kind == "line":
            k.plot([x, x + 5.0], [y_mm, y_mm], gid="deco", **st); x += 6.2
        elif kind == "dot":
            k.scatter([x + 1], [y_mm], s=MS_PT ** 2, lw=0, gid="deco", **st); x += 3.0
        elif kind == "ring":
            k.scatter([x + 1], [y_mm], s=MS_PT ** 2, facecolor="white", lw=0.8, gid="deco", **st); x += 3.0
        t = k.text(x, y_mm, lab, va="center", ha="left", fontsize=TICK)
        bb = t.get_window_extent(r)
        x += bb.width / fig.dpi / MM + gap_mm
    return k


def _texts(fig):
    """Text actually drawn: figure texts, axes texts, titles, axis labels, legend texts, drawn tick labels."""
    out = list(fig.texts)
    for a in fig.axes:
        if not a.get_visible(): continue
        out += list(a.texts) + [a.title, a._left_title, a._right_title, a.xaxis.label, a.yaxis.label]
        leg = a.get_legend()
        if leg is not None: out += list(leg.get_texts())
        for ax_ in ((a.xaxis, a.yaxis) if a.axison else ()):
            for tk in ax_._update_ticks():
                for lab in (tk.label1, tk.label2):
                    if lab.get_visible(): out.append(lab)
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def check(fig, name, allow=()):
    """Render-time checks. allow: pairs of text strings permitted to touch (none used by default)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    texts = _texts(fig)
    boxes = [(t, t.get_window_extent(r).expanded(1.0, 1.0)) for t in texts]
    errs = []
    sizes = []
    for t, b in boxes:
        s_ = t.get_text()
        eff = round(t.get_fontsize() * (SUP if ("$" in s_ and ("^" in s_ or "_" in s_)) else 1.0), 2)
        sizes.append(eff)
        if eff < MIN_PT - 1e-9:
            errs.append(f"text below {MIN_PT} pt: {t.get_text()!r}")
        if b.x0 < fb.x0 - 0.5 or b.x1 > fb.x1 + 0.5 or b.y0 < fb.y0 - 0.5 or b.y1 > fb.y1 + 0.5:
            errs.append(f"text outside canvas: {t.get_text()!r}")
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        if (t1.get_text(), t2.get_text()) in allow or (t2.get_text(), t1.get_text()) in allow:
            continue
        if b1.overlaps(b2):
            ix = min(b1.x1, b2.x1) - max(b1.x0, b2.x0); iy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ix > 0.5 and iy > 0.5:
                errs.append(f"overlap: {t1.get_text()!r} / {t2.get_text()!r}")
    axs = [a for a in fig.axes if a.get_visible() and a.axison]
    for t, b in boxes:
        own = t.axes
        for a in axs:
            if a is own or getattr(a, "_group", None) is not None and getattr(own, "_group", None) == a._group:
                continue
            ab = a.get_window_extent(r)
            ix = min(b.x1, ab.x1) - max(b.x0, ab.x0); iy = min(b.y1, ab.y1) - max(b.y0, ab.y0)
            if ix > 0.5 and iy > 0.5:
                errs.append(f"text {t.get_text()!r} enters another panel's axes")
    # legend boxes (whole legend, handles included) against every text outside that legend and other panels' axes
    for a in fig.axes:
        leg = a.get_legend()
        if leg is None or not a.get_visible(): continue
        lb = leg.get_window_extent(r)
        mine = set(id(t) for t in leg.get_texts())
        for t, b in boxes:
            if id(t) in mine: continue
            ix = min(b.x1, lb.x1) - max(b.x0, lb.x0); iy = min(b.y1, lb.y1) - max(b.y0, lb.y0)
            if ix > 0.5 and iy > 0.5:
                errs.append(f"legend of {a.get_gid()!r} overlaps text {t.get_text()!r}")
        for o in axs:
            if o is a: continue
            ab = o.get_window_extent(r)
            ix = min(ab.x1, lb.x1) - max(ab.x0, lb.x0); iy = min(ab.y1, lb.y1) - max(ab.y0, lb.y0)
            if ix > 0.5 and iy > 0.5:
                errs.append(f"legend of {a.get_gid()!r} enters axes {o.get_gid()!r}")
    if errs:
        raise SystemExit(f"[{name}] checks failed:\n  " + "\n  ".join(errs))
    words = sum(len(t.get_text().replace("$", "").split()) for t in texts)
    return dict(n_text=len(texts), min_pt=min(sizes), median_pt=float(np.median(sizes)), words=words)


def dump_values(fig):
    """Every plotted value in data coordinates, per axes (axes are keyed by their 'gid')."""
    out = {}
    for i, a in enumerate(fig.axes):
        key = a.get_gid() or f"ax{i}"
        d = out.setdefault(key, {"lines": [], "points": [], "segments": [], "bars": [], "fills": []})
        for l in a.get_lines():
            if l.get_gid() == "deco": continue
            d["lines"].append([np.asarray(l.get_xdata(), float).tolist(), np.asarray(l.get_ydata(), float).tolist(), l.get_gid()])
        for c in a.collections:
            if c.get_gid() == "deco": continue
            if isinstance(c, PathCollection):
                d["points"].append([np.asarray(c.get_offsets(), float).tolist(), c.get_gid()])
            elif isinstance(c, LineCollection):
                d["segments"].append([[np.asarray(s, float).tolist() for s in c.get_segments()], c.get_gid()])
            elif isinstance(c, matplotlib.collections.QuadMesh):
                d.setdefault("cells", []).append([np.ma.filled(np.ma.asarray(c.get_array(), float), np.nan).tolist(),
                                                  c.get_gid()])
            elif isinstance(c, PolyCollection):
                d["fills"].append([[np.asarray(p.vertices, float).tolist() for p in c.get_paths()], c.get_gid()])
        for p in a.patches:
            if isinstance(p, Rectangle) and p.get_gid() not in (None, "deco"):
                d["bars"].append([p.get_x(), p.get_y(), p.get_width(), p.get_height(), p.get_gid()])
    return out


def save(fig, name, checks=True):
    info = check(fig, name) if checks else {}
    os.makedirs(OUT, exist_ok=True)
    W = fig.get_size_inches()[0]
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=2400 / W)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), metadata={"CreationDate": None})
    os.makedirs(os.path.join(OUT, "check"), exist_ok=True)
    with open(os.path.join(OUT, "check", f"{name}_values.json"), "w") as f:
        json.dump(dump_values(fig), f)
    Wp, Hp = fig.get_size_inches() * 25.4
    print(f"{name}: {Wp:.1f} x {Hp:.1f} mm printed, {info.get('n_text')} text items, ~{info.get('words')} words, "
          f"min {info.get('min_pt')} pt, median {info.get('median_pt')} pt printed; checks passed", flush=True)
    with open(os.path.join(OUT, "check", f"{name}_textstats.json"), "w") as f:
        json.dump(dict(info, w_mm=round(Wp, 2), h_mm=round(Hp, 1)), f)
    plt.close(fig)
