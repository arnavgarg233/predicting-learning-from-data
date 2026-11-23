"""Figure 7 and Figure F.1 of the article: forecast error by corpus, and learning curves forecast from the data
against extrapolation from pilot runs (panel a, written to fig5; panels b and c, written to fig6).

a  one corpus per family (median two-pilot error at 2^25 in the family). Change: no pilot-range shading, plain titles.
b  |error| at 2^25 for every corpus, by family and method; deterministic beeswarm (horizontal offsets only); bars,
   medians. Change: no tinted family bands.
c  no-training |error| against two-pilot |error|; open markers, c >= 0.70. Change: no tinted halves.
Run: python3 -I scripts/figures/cl/fig5.py
"""
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as _st  # noqa: E402
_st.set_type(7.5, 8.0, 7.5, 8.5)  # panel_cl19: type at the corpus p75 (ticks 7.5 pt)
from style import *  # noqa: E402,F401,F403

S80 = os.path.join(ROOT, "results/curve_forecast/score80.json")
S25 = os.path.join(ROOT, "results/curve_forecast/score.json")
FAMS = [("english", "English"), ("wiki", "Wikipedia"), ("code", "Code")]
FAMC = {"english": GREEN, "wiki": RPUR, "code": VERM}
TRUST = 0.70
ARMS = [("A0", "No training", ACC), ("A2", "One pilot + shape", ACC_M), ("A3", "Two-pilot power law", G1)]


def nice_ticks(lo, hi, n=4):  # verbatim from v42
    span = hi - lo
    step = 10 ** math.floor(math.log10(span / n))
    for m_ in (1, 2, 2.5, 5, 10):
        if span / (step * m_) <= n + 0.5:
            step *= m_; break
    t0 = math.ceil(lo / step - 1e-9) * step
    return [round(t0 + i * step, 6) for i in range(int((hi - t0) / step + 1e-9) + 1)]


def load():  # verbatim from v42
    d = json.load(open(S80 if os.path.exists(S80) else S25))
    P = d["per_corpus"]
    ratio = {}
    for f in glob.glob(os.path.join(ROOT, "results/curve_forecast/**/forecast.json"), recursive=True):
        for k, v in json.load(open(f)).get("corpora", {}).items():
            st = v.get("ref_stats", {})
            if k in P and "ngram23" in st:
                ratio[k] = st["ref_full"] / st["ngram23"]
    assert set(ratio) == set(P), sorted(set(P) - set(ratio))
    return P, ratio


def log2_axis(ax, ticks):
    ax.set_xticks(ticks); ax.set_xticklabels([f"$2^{{{t}}}$" for t in ticks], fontsize=LOG2)


def panel_a(fig, P):
    keyrow(fig, 15, 3.4, [("line", dict(color=ACC, lw=LW), "No training"),
                          ("line", dict(color=ACC_M, lw=LW), "One pilot + shape"),
                          ("line", dict(color=G1, lw=LW, ls=(0, (3, 1.5))), "Two-pilot power law"),
                          ("ring", dict(edgecolor=G1), "Pilot runs"), ("dot", dict(color=INK), "Measured")])
    for j, (fam, title) in enumerate(FAMS):
        cs = sorted(c for c in P if P[c]["family"] == fam)
        e3 = np.array([abs(P[c]["err"]["25"]["A3"]) for c in cs])
        c = cs[int(np.argsort(e3)[len(e3) // 2])]
        R = P[c]
        ys = [R["pilots"]["19"], R["pilots"]["20"], R["measured"]["23"], R["measured"]["25"]]
        ys += [R["forecast"][k][a] for k in ("23", "25") for a, _, _ in ARMS]
        lo, hi = min(ys), max(ys)
        pad = (hi - lo) * 0.08
        ax = axes_mm(fig, 15 + j * 49, 13, 41, 45); ax.set_gid(f"a_{fam}")
        for a, _, col in ARMS[::-1]:
            xs, yv = [23, 25], [R["forecast"]["23"][a], R["forecast"]["25"][a]]
            if a != "A0":
                xs, yv = [20] + xs, [R["pilots"]["20"]] + yv
            ax.plot(xs, yv, color=col, lw=LW, ls=(0, (3, 1.5)) if a == "A3" else "-",
                    gid=f"{a}:{c}")
        ax.scatter([19, 20], [R["pilots"]["19"], R["pilots"]["20"]], s=MS_PT ** 2, facecolor="white", edgecolor=G1, lw=0.9,
                   zorder=4, gid=f"pilots:{c}")
        ax.scatter([23, 25], [R["measured"]["23"], R["measured"]["25"]], s=MS_PT ** 2, color=INK, lw=0, zorder=4,
                   gid=f"measured:{c}")
        ax.set_xlim(18.6, 25.4); ax.set_ylim(lo - pad, hi + pad)
        log2_axis(ax, [19, 21, 23, 25])
        ax.set_yticks([t for t in nice_ticks(lo - pad, hi + pad, 4) if lo - pad <= t <= hi + pad])
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_title(f"{title} ({c.replace('_', ' ')})", fontsize=LAB, pad=3, loc="left", color=FAMC[fam])
        if j == 0: ax.set_ylabel("Test loss (bits per byte)")
        if j == 1: ax.set_xlabel("Training data (bytes)")
        print(f"a {fam}: {c}")


def swarm(y_mm, r):
    """Horizontal offsets (mm) so that circles of radius r (mm) do not overlap; centre-out, deterministic."""
    order = np.argsort(y_mm)
    placed, off = [], np.zeros(len(y_mm))
    for i in order:
        y = y_mm[i]
        for k in range(200):
            dx = (k + 1) // 2 * (0.5 * r) * (1 if k % 2 else -1)
            if all((dx - px) ** 2 + (y - py) ** 2 >= (1.9 * r) ** 2 for px, py in placed):
                break
        off[i] = dx; placed.append((dx, y))
    return off


def panel_b(fig, P):
    letter(fig, "a", 1, 2)
    W_MM, H_MM, YL = 97.0, 44.0, (1e-4, 3.0)
    XL = (-0.6, 2.6)
    # swarm offsets are computed in the v44 geometry (W_MM x H_MM) so every plotted x stays identical to v44;
    # the axes are drawn 83 x 52 units (CL split: this panel is Fig. 6a)
    ax = axes_mm(fig, 15, 8, 83.0, 52.0, yscale="log"); ax.set_gid("b")
    r = 0.56
    for i, (fam, title) in enumerate(FAMS):
        cs = [c for c in P if P[c]["family"] == fam]
        for k, (a, _, col) in enumerate(ARMS):
            e = np.abs([P[c]["err"]["25"][a] for c in cs])
            xc = i + (k - 1) * 0.32
            y_mm = (np.log10(e) - math.log10(YL[0])) / (math.log10(YL[1]) - math.log10(YL[0])) * H_MM
            off = swarm(y_mm, r) * (XL[1] - XL[0]) / W_MM
            ax.scatter(xc + off, e, s=(0.9 * MS2_PT) ** 2, color=col, edgecolor="white", lw=0.3, zorder=3, gid=f"{fam}:{a}")
            med = np.median(e)
            ax.plot([xc - 0.13, xc + 0.13], [med, med], color=INK, lw=1.6, solid_capstyle="butt", zorder=4,
                    gid=f"median:{fam}:{a}")
        ax.text(i, -0.04, title, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=LAB,
                color=FAMC[fam])
        ax.text(i, -0.12, f"n = {len(cs)}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=TICK,
                color=G1)
    ax.set_xlim(*XL); ax.set_ylim(*YL)
    ax.set_xticks([]); ax.spines["bottom"].set_visible(True)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1]); ax.set_yticklabels(["0.001", "0.01", "0.1", "1"]); ax.minorticks_off()
    ax.set_ylabel("|Error| (bits per byte)")


def panel_c(fig, P, ratio):
    letter(fig, "b", 104, 2)
    lim = (1e-4, 8.0)
    ax = axes_mm(fig, 116, 8, 42, 42, xscale="log", yscale="log"); ax.set_gid("c")
    ax.plot(lim, lim, ls=(0, (3, 2)), lw=0.8, color=G2, gid="deco")
    for fam, _ in FAMS:
        cs = [c for c in P if P[c]["family"] == fam]
        x = np.abs([P[c]["err"]["25"]["A3"] for c in cs])
        y = np.abs([P[c]["err"]["25"]["A0"] for c in cs])
        tr = np.array([ratio[c] < TRUST for c in cs])
        ax.scatter(x[tr], y[tr], s=MS2_PT ** 2, color=FAMC[fam], edgecolor="white", lw=0.3, zorder=3, gid=f"trusted:{fam}")
        ax.scatter(x[~tr], y[~tr], s=MS2_PT ** 2 * 0.85, facecolor="white", edgecolor=FAMC[fam], lw=0.8, zorder=3,
                   gid=f"untrusted:{fam}")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    t = [1e-3, 1e-2, 1e-1, 1]
    ax.set_xticks(t); ax.set_yticks(t); ax.minorticks_off()
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1"]); ax.set_yticklabels(["0.001", "0.01", "0.1", "1"])
    ax.set_xlabel("|Error|, two-pilot power law"); ax.set_ylabel("|Error|, no training")
    ax.text(0.97, 0.03, "No training\nmore accurate", transform=ax.transAxes, ha="right", va="bottom", fontsize=TICK,
            color=G1)
    ax.text(0.03, 0.97, "Pilots more accurate", transform=ax.transAxes, ha="left", va="top", fontsize=TICK, color=G1)
    ax.scatter([0.05], [0.33], transform=ax.transAxes, s=MS2_PT ** 2 * 0.85, facecolor="white", edgecolor=G1, lw=0.8,
               gid="deco")
    ax.text(0.10, 0.33, f"c ≥ {TRUST:.2f}", transform=ax.transAxes, va="center", fontsize=TICK)


def main():
    P, ratio = load()
    # CL split of the v44 Fig. 5: v44 panel a -> fig5 (three example curves, no letter);
    # v44 panels b, c -> fig6 panels a, b. Axes gids are kept, so fig5 + fig6 value dumps equal the v44 fig5 dump.
    fig = figure(72)
    panel_a(fig, P)
    save(fig, "fig5")
    fig = figure(70)
    panel_b(fig, P)
    panel_c(fig, P, ratio)
    save(fig, "fig6")


if __name__ == "__main__":
    main()
