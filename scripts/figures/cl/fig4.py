"""Figure 5 of the article: excess loss and evidence threshold in exactly solvable sources.

a  critical TFIM chain, encodings X and Z, exact test KL against training tokens (results/runs/crit_T256_v3, every run;
   lines join seed geometric means; slopes = mean two-point slope). Pale: earlier protocol (results/runs/crit_T256_v2),
   least-squares slope over 2^17-2^20 (scripts/lab_exponents.fit). Slope guides from results/source_stats.json (h=1, X):
   gamma/(1+gamma) and gamma/(2 beta), starting 1.8x above the X mean at 2^17.
b  gated-copy sources, gamma 0.5 / 1 / 2: events needed to learn a distance (K_run) against training tokens, seed 0
   (filled, solid) and seed 1 (open, dashed), results/dial_scores/K_run_C_s{0,1}.json;
   dashed line, seed-0 mean at 2^20.
Run: python3 -I scripts/figures/cl/fig4.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as _st  # noqa: E402
_st.set_type(7.5, 8.0, 7.5, 8.5)  # panel_cl19: type at the corpus p75 (ticks 7.5 pt)
from style import *  # noqa: E402,F401,F403

sys.path.insert(0, os.path.join(ROOT, "scripts"))
from lab_exponents import fit as lab_fit  # noqa: E402

ISING = os.path.join(ROOT, "results/runs/crit_T256_v3")
LADDER = os.path.join(ROOT, "results/runs/crit_T256_v2")
LADDER_FROM = 2 ** 17
SOURCE = os.path.join(ROOT, "results/source_stats.json")
KRUN = [os.path.join(ROOT, f"results/dial_scores/K_run_C_s{s}.json") for s in (0, 1)]
ZCOL = ORNG
ENC = {"X": ("X", GREEN), "Z": ("Z", ZCOL)}
GCOL = {"0.5": "#A88BD0", "1": "#7A4FA3", "2": "#46216E"}   # one hue, light to dark with the exponent
GAMMAS = ["0.5", "1", "2"]
L2 = np.log2


def log2_axis(ax, ticks):
    ax.set_xticks(ticks); ax.set_xticklabels([f"$2^{{{t}}}$" for t in ticks], fontsize=LOG2)


def panel_a(fig):
    ax = axes_mm(fig, 15, 7, 52, 71, yscale="log"); ax.set_gid("a")
    letter(fig, "a", 1, 2)
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(ISING, "*.json")))]
    src = json.load(open(SOURCE))["h=1.0 X"]
    rule, pair = src["gamma"] / (1 + src["gamma"]), src["alpha_pred"]
    ladder = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(LADDER, "*.json")))]
    lslope = {}
    for b, (lab, col) in ENC.items():  # earlier protocol, pale
        R = [r for r in ladder if r["basis"] == b]
        P = sorted({r["P"] for r in R})
        grp = {p: np.array([r["test_kl_bits"] for r in R if r["P"] == p]) for p in P}
        win = [p for p in P if p >= LADDER_FROM]
        lslope[b] = lab_fit(win, [grp[p] for p in win])
        c = np.mean(np.log(np.concatenate([grp[p] for p in win]))) + lslope[b] * np.mean(
            np.concatenate([[np.log(p)] * len(grp[p]) for p in win]))
        ax.plot([L2(win[0]), L2(win[-1])], [np.exp(c - lslope[b] * np.log(win[0])), np.exp(c - lslope[b] * np.log(win[-1]))],
                color=col, alpha=0.35, lw=LW, gid=f"earlier_fit:{b}")
        ax.scatter([L2(r["P"]) for r in R], [r["test_kl_bits"] for r in R], s=MS2_PT ** 2, color=col, alpha=0.35, lw=0,
                   gid=f"earlier:{b}")
    slopes, ends = {}, []
    for b, (lab, col) in ENC.items():
        R = [r for r in runs if r["basis"] == b]
        P = sorted({r["P"] for r in R})
        seeds = sorted({r["seed"] for r in R})
        kl = {(r["P"], r["seed"]): r["test_kl_bits"] for r in R}
        gm = [np.exp(np.mean([np.log(kl[(p, s)]) for s in seeds])) for p in P]
        slopes[b] = -np.log(gm[-1] / gm[0]) / np.log(P[-1] / P[0])
        ax.plot(L2(P), gm, color=col, lw=1.4, gid=f"geomean:{b}")
        ax.scatter([L2(r["P"]) for r in R], [r["test_kl_bits"] for r in R], s=MS_PT ** 2, color=col, edgecolor="white",
                   lw=0.4, zorder=3, gid=f"runs:{b}")
        ends.append((gm[-1], f"{lab}  {slopes[b]:.2f}", col))
        if b == "X":
            x0, y0, x1 = P[0], 1.8 * gm[0], P[-1]
    # CL label: the pair-correlation guide is named by its formula, as in the caption; gid keeps the v44 name
    for s, col, lab, ls, gname in ((rule, ACC, f"γ/(1+γ)  {rule:.2f}", (0, (4, 2)), f"γ/(1+γ)  {rule:.2f}"),
                                   (pair, G1, f"γ/(2β)  {pair:.2f}", (0, (1, 1.5)), f"Cagnetta et al.  {pair:.2f}")):
        y1 = y0 * (x1 / x0) ** -s
        ax.plot([L2(x0), L2(x1)], [y0, y1], color=col, ls=ls, lw=1.1, gid=f"guide:{gname}")
        ends.append((y1, lab, col))
    # direct labels right of the axes, spread in log space so they never overlap
    ends.sort(key=lambda e: e[0])
    ly = np.log10([e[0] for e in ends])
    for _ in range(500):
        moved = False
        for i in range(1, len(ly)):
            if ly[i] - ly[i - 1] < 0.15:
                d = (0.15 - (ly[i] - ly[i - 1])) / 2; ly[i - 1] -= d; ly[i] += d; moved = True
        if not moved: break
    for (yv, lab, col), l in zip(ends, ly):
        ax.annotate(lab, (20.15, yv), xytext=(20.6, 10 ** l), textcoords="data", fontsize=NOTE,
                    color=G1 if col == G1 else col, va="center", ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=G2, shrinkA=0, shrinkB=1))
    ax.scatter([14.0, 14.25], [4.6e-4, 4.6e-4], s=MS2_PT ** 2, color=[GREEN, ZCOL], alpha=0.35, lw=0, gid="deco")
    ax.text(14.5, 4.6e-4, f"Earlier protocol,\nslopes {lslope['X']:.2f} and {lslope['Z']:.2f}", fontsize=NOTE,
            va="center", color=G1, linespacing=1.1)
    ax.set_xlim(13.6, 20.9); ax.set_ylim(3e-4, 1e-1)
    ax.spines["bottom"].set_bounds(13.6, 20.4)
    ax.set_yticks([1e-3, 1e-2, 1e-1]); ax.set_yticklabels(["0.001", "0.01", "0.1"])
    log2_axis(ax, list(range(14, 21)))
    ax.set_xlabel("Training data (tokens)")
    ax.set_ylabel("Excess loss, exact KL (bits per token)")
    print(f"a: slopes X {slopes['X']:.3f}, Z {slopes['Z']:.3f}; rule {rule:.3f}; pair {pair:.3f}; "
          f"earlier X {lslope['X']:.3f}, Z {lslope['Z']:.3f}")


def panel_b(fig):
    K = [json.load(open(f)) for f in KRUN]
    assert K[1]["slope"] == K[0]["slope"]
    k20 = np.mean([r["K_run"] for r in K[0]["runs"] if r["log2P"] == 20])
    letter(fig, "b", 85, 2)
    for j, g in enumerate(GAMMAS):
        col = GCOL[g]
        ax = axes_mm(fig, 100 + j * 20.5, 13, 16.0, 65, yscale="log"); ax.set_gid(f"b_gamma{g}"); ax._group = "b"
        ax.axhline(k20, ls=(0, (3, 2)), lw=0.8, color=G1, gid="deco")
        S = [{r["log2P"]: r["K_run"] for r in d["runs"] if r["g"] == g} for d in K]
        xc = sorted(set(S[0]) & set(S[1]))
        for seed, d in enumerate(K):
            rr = sorted((r for r in d["runs"] if r["g"] == g), key=lambda r: r["log2P"])
            xs, ys = [r["log2P"] for r in rr], [r["K_run"] for r in rr]
            if seed == 0:
                ax.plot(xs, ys, color=col, lw=LW, gid=f"s0:{g}")
                ax.scatter(xs, ys, s=MS_PT ** 2, color=col, edgecolor="white", lw=0.4, zorder=3, gid=f"s0pts:{g}")
            else:
                ax.plot(xs, ys, color=col, lw=LW2, ls=(0, (2.5, 1.5)), gid=f"s1:{g}")
                ax.scatter(xs, ys, s=MS_PT ** 2, facecolor="white", edgecolor=col, lw=0.9, zorder=4, gid=f"s1pts:{g}")
        ax.set_xlim(13.4, 20.6); ax.set_ylim(80, 3500)
        log2_axis(ax, [14, 20])
        ax.minorticks_off()
        ax.set_yticks([100, 300, 1000, 3000])
        if j == 0:
            ax.set_yticklabels(["100", "300", "1000", "3000"])
            ax.set_ylabel("Events needed to learn a distance")
        else:
            ax.set_yticklabels([])
        if j == 1:
            ax.set_xlabel("Training data (tokens)")
        if j == 2:
            ax.text(13.7, k20 * 0.92, f"{k20:.0f}", fontsize=TICK, va="top", ha="left", color=G1)
        ax.set_title(f"γ = {g}", fontsize=LAB, pad=3, color=col)
    keyrow(fig, 93, 3.2, [("dot", dict(color=G1), "Seed 0"), ("ring", dict(edgecolor=G1), "Seed 1"),
                           ("line", dict(color=G1, lw=0.8, ls=(0, (3, 2))), "Seed-0 mean")])
    print(f"b: seed-0 mean K_run at 2^20 {k20:.1f}")


def main():
    fig = figure(92)
    panel_a(fig)
    panel_b(fig)
    save(fig, "fig4")


if __name__ == "__main__":
    main()
