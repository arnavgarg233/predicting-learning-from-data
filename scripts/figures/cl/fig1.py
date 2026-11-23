"""Figures 1 and 2 of the article (schematics, illustrative values only): the two statistics, and the information law.

Colour encodes one thing each: purple, the evaluation text and the information curve read from it (the rate);
orange, the training corpus and the mismatch (the level); blue, the forecast and the learning mechanism.
Figure 1 is panel a, written to fig1; Figure 2 is panel b, written to fig1b.
a  evaluation text and training corpus -> information curve h(n) = 0.2 + 0.72 n^-0.55 (floor 0.2), which sets the
   rate, and the mismatch between two token distributions (hills, same parameters), which sets the level ->
   forecast learning curve L(D) = 0.24 + 0.66 D^-0.32 with band [0.24 + 0.62 D^-0.36, 0.24 + 0.70 D^-0.285],
   level 0.24, slope wedge 12-150, illustrative measured rings at 250, 1400, 6500 (L + 0.012).
b  events per context distance D/d^2 for D = 4.4, 17.6, 70.4 (log bars, d = 1..12) against events needed = 1;
   excess loss 0.95 x^-0.5 on log-log axes with points at x = 1.3 D/4.4.
Run: python3 -I scripts/figures/cl/fig1.py
"""
import math
import os
import sys

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle as Rect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as _st  # noqa: E402
_st.set_type(7.0, 8.2, 7.5, 8.2)  # panel_cl20: smallest text 7 pt (exponents 0.86 x 8.2)
from style import *  # noqa: E402,F401,F403

H_MM = 100.0
EV, EV_F = PURP, "#9474C6"      # evaluation text: line, fill
CO, CO_F = VERM, ORNG           # training corpus: line, fill
FRAME = "#A3A3A3"


def canvas(fig):
    """Full-figure axes in mm (origin top-left) for frames, glyphs, arrows and free labels."""
    W, H = fig.get_size_inches() / MM
    c = fig.add_axes([0, 0, 1, 1], zorder=-1)
    c.set_xlim(0, W); c.set_ylim(H, 0); c.axis("off")
    return c


def frame(c, x0, y0, x1, y1):
    c.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=1.6", fill=False,
                               lw=0.8, ec=FRAME, gid="deco"))


def arrow(c, a, b, color=G1):
    c.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>,head_length=2.4,head_width=1.2", lw=0.9, color=color,
                                shrinkA=0, shrinkB=0, mutation_scale=1.0, gid="deco"))


def box(c, x0, y0, x1, y1, line):
    """Plain labelled box: white fill, thin outline in the element's colour."""
    c.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.8", fc="white",
                               ec=line, lw=0.9, gid="deco"))


def plain(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.minorticks_off()


def panel_a(fig, c):
    # panel_cl51: panel a is its own figure (paper Fig. 1), no letter
    # panel_cl21: no frame around the two input boxes (box-in-box removed)
    frame(c, 47, 5, 102, 54)
    frame(c, 113, 5, 157, 54)

    # data: evaluation text and training corpus
    # panel_cl21: plain labelled boxes in place of the document pictograms
    box(c, 6.0, 12.0, 29.5, 24.0, EV)
    c.text(17.75, 18.0, "Evaluation text", ha="center", va="center", fontsize=NOTE)
    box(c, 6.0, 37.5, 29.5, 49.5, CO)
    c.text(17.75, 43.5, "Training corpus", ha="center", va="center", fontsize=NOTE)

    # information curve (sets the rate)
    ai = axes_mm(fig, 59, 11.5, 39, 12.5, xscale="log"); ai.set_gid("info"); plain(ai)
    ns = [10 ** (3 * i / 80) for i in range(81)]
    hinf = 0.2
    hv = [hinf + 0.72 * n ** -0.55 for n in ns]
    ai.fill_between(ns, hinf, hv, color=EV_F, alpha=0.35, lw=0, gid="excess_fill")
    ai.plot(ns, hv, color=EV, lw=1.5, gid="h(n)")
    ai.plot([1, 1000], [hinf, hinf], ls=(0, (3, 2)), lw=0.8, color=G1, gid="floor")
    ai.set_xlim(1, 1000); ai.set_ylim(0, 1.0)
    ai.set_ylabel("$h(n)$", labelpad=3); ai.set_xlabel("Context $n$", labelpad=2)
    ai.text(16, 0.6, r"$\propto n^{-\gamma}$", color=EV, fontsize=LAB, ha="left", va="center")
    ai.set_title("Information curve", fontsize=NOTE, pad=2.5)

    # mismatch between the two token distributions (sets the level)
    am = axes_mm(fig, 59, 35.5, 39, 11); am.set_gid("mismatch"); plain(am); am.spines["left"].set_visible(False)

    def hill(mu, s, amp, skew=0.0):
        u = np.arange(81) / 80
        z = (u - mu) / s
        return u, amp * np.exp(-0.5 * z * z) * (1 + skew * np.tanh(z))
    u, hc = hill(0.38, 0.16, 1.25, 0.15)
    _, he = hill(0.62, 0.13, 1.35, -0.1)
    am.fill_between(u, 0, hc, color=CO_F, alpha=0.8, lw=0, gid="corpus")
    am.fill_between(u, 0, he, color=EV_F, alpha=0.7, lw=0, gid="evaluation")
    am.plot(u, hc, color=CO, lw=0.9, gid="corpus_edge")
    am.plot(u, he, color=EV, lw=0.9, gid="evaluation_edge")
    am.set_xlim(0, 1); am.set_ylim(0, 2.0)
    pc, pe = u[np.argmax(hc)], u[np.argmax(he)]
    am.annotate("", xy=(pe, 1.72), xytext=(pc, 1.72),
                arrowprops=dict(arrowstyle="<|-|>,head_length=0.35,head_width=0.18", lw=0.9, color=INK))
    am.set_xlabel("Tokens", labelpad=2)
    am.set_title("Train–evaluation mismatch", fontsize=NOTE, pad=2.5)
    am.text(pc - 0.05, 1.72, "Corpus", fontsize=TICK, color=CO, ha="right", va="center")
    am.text(pe + 0.05, 1.72, "Evaluation", fontsize=TICK, color=EV, ha="left", va="center")

    # forecast learning curve
    af = axes_mm(fig, 124, 11.5, 30, 35.5, xscale="log"); af.set_gid("forecast"); plain(af)
    lev = 0.24

    def Lf(d): return lev + 0.66 * d ** -0.32
    Ds = np.array([10 ** (4 * i / 100) for i in range(101)])
    af.fill_between(Ds, lev + 0.62 * Ds ** -0.36, lev + 0.70 * Ds ** -0.285, color=ACC_L, alpha=0.75, lw=0, gid="band")
    af.plot(Ds, Lf(Ds), color=ACC, lw=1.5, gid="forecast")
    af.plot([1, 1e4], [lev, lev], ls=(0, (3, 2)), lw=1.1, color=CO, gid="level")
    da, db = 12.0, 150.0
    af.plot([da, db, db], [Lf(da), Lf(da), Lf(db)], color=EV, lw=0.9, gid="rate_wedge")
    af.text(math.sqrt(da * db), Lf(da) + 0.015, "Rate", ha="center", va="bottom", fontsize=NOTE, color=EV)
    af.text(1.3, lev + 0.02, "Level", ha="left", va="bottom", fontsize=NOTE, color=CO)
    md = np.array([2.5e2, 1.4e3, 6.5e3])
    af.scatter(md, Lf(md) + 0.012, s=MS_PT ** 2, facecolor="white", edgecolor=INK, lw=0.9, zorder=4, gid="measured")
    af.scatter([22], [0.80], s=MS_PT ** 2, facecolor="white", edgecolor=INK, lw=0.9, zorder=4, gid="deco")
    af.text(36, 0.80, "Measured after\ntraining", fontsize=TICK, va="center", ha="left", linespacing=1.1)
    af.set_xlim(1, 1e4); af.set_ylim(0, 1.0)
    af.set_xlabel("Training data", labelpad=2); af.set_ylabel("Loss", labelpad=2)
    af.set_title("Forecast learning curve", fontsize=NOTE, pad=2.5)

    # arrows: data -> read-outs -> forecast
    arrow(c, (29.5, 17.5), (46.0, 17.5), EV)
    c.plot([29.5, 32.0, 32.0], [21.5, 21.5, 38.5], color=EV, lw=0.9, solid_joinstyle="miter", gid="deco")
    arrow(c, (32.0, 38.5), (46.0, 38.5), EV)
    arrow(c, (29.5, 43.5), (46.0, 43.5), CO)
    arrow(c, (102.6, 17.5), (112.4, 17.5), EV)
    c.text(107.5, 15.8, "Sets\nrate", ha="center", va="bottom", fontsize=NOTE, color=EV, linespacing=1.05)
    arrow(c, (102.6, 41.0), (112.4, 41.0), CO)
    c.text(107.5, 39.3, "Sets\nlevel", ha="center", va="bottom", fontsize=NOTE, color=CO, linespacing=1.05)


def panel_b(fig, c, DY=-55.0):
    """panel_cl51: drawn as its own figure (fig1b, paper Fig. 2); every v44 coordinate shifted up by DY."""
    frame(c, 4, 60 + DY, 102, 98 + DY)
    frame(c, 113, 60 + DY, 157, 98 + DY)
    K = 1.0
    Ds = [(4.4, "D"), (17.6, "4D"), (70.4, "16D")]
    nd = 12
    d = np.arange(1, nd + 1)
    for r, (D, lab) in enumerate(Ds):
        ab = axes_mm(fig, 18 + r * 28.5, 66.5 + DY, 24, 21, yscale="log"); ab.set_gid(f"bars_{lab}"); ab._group = "b"
        e = D / d ** 2
        for di, ei in zip(d, e):
            ab.add_patch(Rect((di - 0.38, 0.018), 0.76, ei - 0.018, color=ACC if ei >= K else G3, lw=0,
                              gid=f"{lab}:d{di}"))
        ab.plot([0.4, nd + 0.6], [K, K], ls=(0, (3, 2)), lw=0.9, color=INK, gid="needed")
        ab.set_xlim(0.4, nd + 0.6); ab.set_ylim(0.018, 110)
        ab.set_yticks([]); ab.minorticks_off()
        if r > 0: ab.spines["left"].set_visible(False)
        ab.set_xticks([1, 4, 8, 12])
        ab.set_title(f"Data {lab}", fontsize=NOTE, pad=2.5)
        nl = int((e >= K).sum())
        ab.text(nd + 0.6, 110, f"{nl} learned", ha="right", va="top", fontsize=TICK, color=ACC)
        if r == 0:
            ab.text(4.6, K * 1.5, "Events needed", ha="left", va="bottom", fontsize=TICK)
            ab.set_ylabel("Events per distance", labelpad=3)
        if r == 1:
            ab.set_xlabel("Context distance", labelpad=2)

    ae = axes_mm(fig, 125, 66.5 + DY, 29, 21, xscale="log", yscale="log"); ae.set_gid("excess"); plain(ae)
    xs = np.array([10 ** (math.log10(30) * i / 50) for i in range(51)])
    ae.plot(xs, 0.95 * xs ** -0.5, color=ACC, lw=1.5, gid="excess")
    px = np.array([D / 4.4 * 1.3 for D, _ in Ds])
    ae.scatter(px, 0.95 * px ** -0.5, s=MS_PT ** 2, facecolor=ACC, edgecolor="white", lw=0.4, zorder=3, gid="points")
    for x, (_, lab) in zip(px, Ds):
        last = lab == "16D"  # panel_cl20: last label centred above its point so it stays inside the frame
        ae.annotate(lab, (x, 0.95 * x ** -0.5), xytext=(0 if last else 2, 3.5 if last else 2.5),
                    textcoords="offset points", fontsize=TICK, ha="center" if last else "left", va="bottom")
    ae.text(1.15, 0.2, r"$\propto D^{-\gamma/(1+\gamma)}$", color=ACC, fontsize=LAB, ha="left", va="center")
    ae.set_xlim(1, 30); ae.set_ylim(0.12, 1.2)
    ae.set_xlabel("Data", labelpad=2); ae.set_ylabel("Excess loss", labelpad=2)
    ae.set_title("Excess loss", fontsize=NOTE, pad=2.5)
    arrow(c, (102.6, 77.0 + DY), (112.4, 77.0 + DY), ACC)


def main():
    # panel_cl51: v44 panels a and b become two figures (fig1 = concept, fig1b = information law); geometry kept
    fig = figure(57.0)
    c = canvas(fig)
    panel_a(fig, c)
    save(fig, "fig1")
    fig = figure(46.0)
    c = canvas(fig)
    panel_b(fig, c)
    save(fig, "fig1b")


if __name__ == "__main__":
    main()
