"""Forecast error against the compute spent before the forecast.

Source: results/reanalysis/compute_frontier.json only (scripts/reanalysis/compute_frontier.py). Plotted
values are read from the file. One panel per corpus family: r.m.s. error at 2^25 bytes (95% corpus-bootstrap interval)
against the median floating-point operations each method spends before forecasting, counting the reference model's
inference for the data-only forecast. Dotted vertical line: training the learner to 2^25 bytes itself.
Code and English: 80 test + 18 new corpora. Wikipedia: 55 corpora; the Qwen2.5-1.5B and 7B references were scored only
on the 47 test corpora (the 0.5B forecast on those 47: 0.387, on all 55: 0.385).

    python scripts/figures/fig_ed_compute.py   -> figures/fig_ed_compute.pdf (+ .png), copied to paper/fig_ed_compute.pdf
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SRC = os.path.join(C.ROOT, "results/reanalysis/compute_frontier.json")
# key, label, colour, marker, filled
METHODS = [("A0_1k", "Data alone, 1,000 reference targets", C.OURS, "o", True),
           ("A0_20k", "Data alone, 20,000 reference targets", C.OURS, "o", False),
           ("A0_1.5B", "Data alone, Qwen2.5-1.5B (1,000 targets)", C.SKY, "s", True),
           ("A0_7B", "Data alone, Qwen2.5-7B (1,000 targets)", C.SKY, "D", True),
           ("X_pool1", "One pilot + pooled exponent", C.ORANGE, "^", True),
           ("A3", "Two pilots, power law", C.GREY, "v", True),
           ("A3_20_23", "Pilots at $2^{20}$ and $2^{23}$, power law", C.BLACK, "v", False)]
PANELS = [("code", "Code"), ("english", "English"), ("wiki", "Wikipedia")]


def main():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    C.setup(C.FULL, script_ticks=True)
    d = json.load(open(SRC))["blocks"]
    fig = plt.figure(figsize=(C.WIDTH_MM * C.MM, 62 * C.MM))
    gs = fig.add_gridspec(1, 3, left=0.07, right=0.985, bottom=0.17, top=0.74, wspace=0.32)
    axes = []
    for j, (fam, title) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, j])
        axes.append(ax)
        blk = d[fam]
        meths = dict(blk["methods"])
        if fam == "wiki":
            for k in ("A0_1.5B", "A0_7B"):
                meths[k] = d["wiki47"]["methods"][k]
        # frontier of the plotted points (lower error at no more compute), drawn as a light step line
        pts = sorted((v["flops_median"], v["rms"]) for v in meths.values())
        fx, fy, best = [], [], float("inf")
        for x, y in pts:
            if y < best:
                fx.append(x)
                fy.append(y)
                best = y
        ax.step(fx + [blk["target_flops_median"]], fy + [fy[-1]], where="post", color=C.LIGHT, lw=0.8, zorder=1)
        for k, _, col, mk, filled in METHODS:
            if k not in meths:
                continue
            v = meths[k]
            x, y = v["flops_median"], v["rms"]
            ax.errorbar([x], [y], yerr=[[y - v["ci95"][0]], [v["ci95"][1] - y]], fmt="none", ecolor=col, elinewidth=0.6,
                        capsize=0, zorder=2)
            ax.plot([x], [y], ls="none", marker=mk, ms=4.2, mfc=col if filled else "white", mec=col, mew=0.8, zorder=3)
        ax.axvline(blk["target_flops_median"], color=C.BLACK, lw=0.6, ls=(0, (1, 1.5)), zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(2e14, 6e16)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Operations before the forecast (FLOPs)")
        if j == 0:
            ax.set_ylabel("R.m.s. error at $2^{25}$ bytes\n(bits per byte)")
        ax.set_title(f"{title} ({blk['n']} corpora)", loc="left", pad=3)
        C.keep_tick_density(ax, "y")
    hl = [Line2D([], [], ls="none", marker=mk, ms=4.2, mfc=col if filled else "white", mec=col, mew=0.8)
          for _, _, col, mk, filled in METHODS]
    ll = [lab for _, lab, _, _, _ in METHODS]
    hl.append(Line2D([], [], color=C.BLACK, lw=0.6, ls=(0, (1, 1.5))))
    ll.append("Training to $2^{25}$ bytes")
    fig.legend(hl, ll, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=1.6, fontsize=6.5,
               columnspacing=1.2)
    C.letters(fig, [(ax, s) for ax, s in zip(axes, "abc")])
    C.finish(fig, "fig_ed_compute")
    shutil.copy(os.path.join(C.OUT, "fig_ed_compute.pdf"), os.path.join(C.ROOT, "paper", "fig_ed_compute.pdf"))


if __name__ == "__main__":
    sys.exit(main())
