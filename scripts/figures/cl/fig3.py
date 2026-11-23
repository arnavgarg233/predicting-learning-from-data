"""Figure 6 and Figure E.1 of the article: the data-scaling exponent across tokenizers, and the learning-rate and
weight-decay grid of the 100M-parameter models (written to fig3b).

a  token-encoding exponent against the bytes exponent (log-log), every setting, tokenizer and floor choice (ts_v5 for
   TinyStories v5, results/bfine/final_analysis.json rate_powerlaw, results/midscale_rate_verify.json); whiskers,
   seed-bootstrap 95% intervals; dashed equality line.
b  learning-rate x weight-decay heat maps (bytes, BPE-8k) of test bits per byte at 2^25 bytes, seed 0, every cell run
   (results/midscale_vocab/*.json); shared viridis scale (as in the corpus heat maps); grey cells, not run; the best
   cell (asserted against best_cells.txt) outlined, with the seed-1 value next to a ring.
Run: python3 -I scripts/figures/cl/fig3.py
"""
import glob
import json
import os
import sys

import numpy as np
from matplotlib.patches import Rectangle as Rect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as _st  # noqa: E402
_st.set_type(7.5, 8.2, 7.5, 8.5)  # panel_cl19: type at the corpus p75 (ticks 7.5 pt); 10^-3 exponent 7.05 pt
from style import *  # noqa: E402,F401,F403

V5 = os.path.join(ROOT, "results/text_runs/v5")
V5_TOKS = ["bytes", "bpe512", "bpe2048", "bpe8192"]
V5_KS = [21, 22, 23, 24]
V5_FLOORS = [0.12, 0.25, 0.38]
V5_BOOT = 2000
FW = os.path.join(ROOT, "results/bfine/final_analysis.json")
MS = os.path.join(ROOT, "results/midscale_rate_verify.json")
VOCAB = os.path.join(ROOT, "results/midscale_vocab")
VOCAB_LOG2D = 25
BEST_CELLS = os.path.join(VOCAB, "best_cells.txt")
SET = {"ts": ("TinyStories, small", ORNG), "fw": ("FineWeb-Edu, small", RPUR), "ms": ("FineWeb-Edu, 100M", WINE)}


def ts_v5():  # verbatim from v42
    R = {}
    for f in glob.glob(f"{V5}/*.json"):
        d = json.load(open(f))
        k = int(os.path.basename(f).split("_b")[1].split("_")[0])
        if k in V5_KS and "bits_per_byte" in d:
            R.setdefault(d["tok"], {}).setdefault(k, []).append(d["bits_per_byte"])
    assert all(len(R[t][k]) == 5 for t in V5_TOKS for k in V5_KS)
    rng = np.random.default_rng(0)

    def slope(ys, fl):
        return -np.polyfit(np.array(V5_KS) * np.log(2), np.log(np.array(ys) - fl), 1)[0]
    out = []
    for fl in V5_FLOORS:
        a = {t: slope([np.mean(R[t][k]) for k in V5_KS], fl) for t in V5_TOKS}
        B = {t: [] for t in V5_TOKS}
        for _ in range(V5_BOOT):
            for t in V5_TOKS:
                B[t].append(slope([np.mean(rng.choice(R[t][k], len(R[t][k]))) for k in V5_KS], fl))
        out.append((fl, {t: (a[t], np.percentile(B[t], [2.5, 97.5])) for t in V5_TOKS}))
    return out


def vocab_runs():  # verbatim from v42
    runs = []
    for f in sorted(glob.glob(os.path.join(VOCAB, "*.json"))):
        d = json.load(open(f))
        if int(round(np.log2(d["train_bytes"]))) == VOCAB_LOG2D and d["encoding"] in ("bytes", "bpe8k"):
            runs.append(d)
    return runs


def panel_a(fig):
    TS = ts_v5()
    FWr = json.load(open(FW))["rate_powerlaw"]
    MSv = {k: v for k, v in json.load(open(MS)).items() if isinstance(v, dict) and "alpha_bytes" in v}
    pts = {"ts": [], "fw": [], "ms": []}
    ts_ci = []
    for fl, A in TS:
        for t in V5_TOKS[1:]:
            pts["ts"].append((A["bytes"][0], A[t][0]))
            ts_ci.append(list(A["bytes"][1]) + list(A[t][1]))
    for v in FWr.values():
        pts["fw"] += [(v["alpha_bytes"], a) for t, a in v["alpha"].items() if t != "bytes"]
    for v in MSv.values():
        pts["ms"].append((v["alpha_bytes"], v["alpha_gpt2"]))
    ci = [v["alpha_bytes_ci95"] + v["alpha_gpt2_ci95"] for v in MSv.values()]
    allv = np.array([p for P in pts.values() for p in P] + [c[:2] for c in ci + ts_ci] + [c[2:] for c in ci + ts_ci])
    lo, hi = allv.min() / 1.15, allv.max() * 1.15
    ax = axes_mm(fig, 15, 5, 61, 61, xscale="log", yscale="log"); ax.set_gid("a")  # panel_cl51: own figure (80 units wide)
    ax.plot([lo, hi], [lo, hi], ls=(0, (3, 2)), lw=0.8, color=G2, gid="deco", zorder=1)
    for key, P, C in (("ts", pts["ts"], ts_ci), ("ms", pts["ms"], ci)):
        col = SET[key][1]
        for (x, y), q in zip(P, C):
            ax.plot([q[0], q[1]], [y, y], color=col, lw=0.9, gid=f"ci_x:{key}", zorder=2)
            ax.plot([x, x], [q[2], q[3]], color=col, lw=0.9, gid=f"ci_y:{key}", zorder=2)
    for key in ("fw", "ts", "ms"):
        xy = np.array(pts[key])
        ax.scatter(xy[:, 0], xy[:, 1], s=MS_PT ** 2, color=SET[key][1], edgecolor="white", lw=0.4, zorder=3,
                   gid=f"pts:{key}", label=SET[key][0])
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
    ax.minorticks_off()
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels([f"{v:.1f}" for v in ticks]); ax.set_yticklabels([f"{v:.1f}" for v in ticks])
    ax.set_xlabel("Bytes exponent"); ax.set_ylabel("Token exponent")
    h, l = ax.get_legend_handles_labels()
    order = [l.index(SET[k][0]) for k in ("ts", "fw", "ms")]
    legend(ax, handles=[h[i] for i in order], labels=[l[i] for i in order], loc="upper left")


def panel_b(fig):
    runs = vocab_runs()
    best_txt = {l.split()[0]: (float(l.split()[3]), float(l.split()[5])) for l in open(BEST_CELLS) if l.strip()}
    lrs = sorted({r["lr"] for r in runs})
    wds = sorted({r["wd"] for r in runs})
    vals = [r["test_bpb"] for r in runs if r["seed"] == 0]
    vmin, vmax = min(vals), max(vals)
    cmap = matplotlib.colormaps["viridis"]
    norm = matplotlib.colors.Normalize(vmin, vmax)
    cw, ch = 12.0, 11.5   # panel_cl22: larger cells
    X0, Y0, GAP = 15, 8, 8.0  # panel_cl51: heat maps are their own appendix figure (fig3b)
    for k, (enc, title) in enumerate((("bytes", "Bytes"), ("bpe8k", "BPE-8k"))):
        ax = axes_mm(fig, X0, Y0 + k * (len(wds) * ch + GAP), len(lrs) * cw, len(wds) * ch)
        ax.set_gid(f"b_{enc}"); ax._group = "b"
        R0 = {(r["lr"], r["wd"]): r for r in runs if r["encoding"] == enc and r["seed"] == 0}
        best = min(R0.values(), key=lambda r: r["test_bpb"])
        assert (best["wd"], best["lr"]) == best_txt[enc], (enc, best["wd"], best["lr"])
        s1 = [r for r in runs if r["encoding"] == enc and r["seed"] == 1 and (r["lr"], r["wd"]) == (best["lr"], best["wd"])]
        M = np.full((len(wds), len(lrs)), np.nan)
        for i, wd in enumerate(wds):
            for j, lr in enumerate(lrs):
                if (lr, wd) in R0: M[i, j] = R0[(lr, wd)]["test_bpb"]
        qm = ax.pcolormesh(np.arange(len(lrs) + 1), np.arange(len(wds) + 1), np.ma.masked_invalid(M), cmap=cmap,
                           norm=norm, edgecolors="white", linewidth=0.8, gid=f"cells:{enc}")
        for i, wd in enumerate(wds):
            for j, lr in enumerate(lrs):
                v = M[i, j]
                if np.isnan(v):
                    ax.add_patch(Rect((j, i), 1, 1, fc="#e6e6e6", ec="white", lw=0.8, zorder=0.5, gid="deco"))
                    continue
                tc = "white" if norm(v) < 0.55 else INK
                isb = (lr, wd) == (best["lr"], best["wd"])
                ax.text(j + 0.5, i + 0.5 + (0.20 if isb else 0), f"{v:.3f}", ha="center", va="center", fontsize=8.0,
                        color=tc)
                if isb:  # panel_cl28/29: cell labels enlarged; second seed in parentheses instead of a marker
                    ax.add_patch(Rect((j + 0.04, i + 0.04), 0.92, 0.92, fill=False, ec=VERM, lw=1.4, gid="deco"))
                    ax.text(j + 0.5, i + 0.30, f"({s1[0]['test_bpb']:.3f})", ha="center", va="center", fontsize=7.5,
                            color=tc)
        ax.add_patch(Rect((0, 0), len(lrs), len(wds), fill=False, ec=G3, lw=0.4, gid="deco", zorder=0))
        ax.set_xlim(0, len(lrs)); ax.set_ylim(0, len(wds))
        for s_ in ("left", "bottom"): ax.spines[s_].set_visible(False)
        ax.set_xticks(np.arange(len(lrs)) + 0.5); ax.set_xticklabels([f"{lr * 1e3:g}" for lr in lrs] if k == 1 else [])
        ax.set_yticks(np.arange(len(wds)) + 0.5)
        ax.set_yticklabels([f"{wd:g}" for wd in wds])
        ax.tick_params(length=0)
        if k == 1: ax.set_xlabel(r"Learning rate ($10^{-3}$)", fontsize=LAB)
        ax.set_ylabel("Weight decay")
        ax.set_title(title, fontsize=LAB, pad=3, loc="left")
    cax = axes_mm(fig, X0 + len(lrs) * cw + 4, Y0 + 6, 2.6, 2 * len(wds) * ch + GAP - 12); cax.set_gid("deco_cbar")
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm, cmap), cax=cax)
    cb.outline.set_linewidth(0.5); cb.set_ticks([1.38, 1.42, 1.46, 1.49])
    cb.ax.tick_params(length=2.0, width=0.5)
    cb.set_label("Test loss (bits per byte)", fontsize=LAB)
    print("b: seed-0 cells", {e: len([r for r in runs if r["encoding"] == e and r["seed"] == 0]) for e in ("bytes", "bpe8k")},
          f"range {vmin:.4f}-{vmax:.4f}")


def main():
    # panel_cl51: v44 panel a stays in the main text (fig3); the tuning heat maps move to the appendix (fig3b)
    fig = figure(78, w_mm=80)
    panel_a(fig)
    save(fig, "fig3")
    fig = figure(122, w_mm=88)
    panel_b(fig)
    save(fig, "fig3b")


if __name__ == "__main__":
    main()
