"""Figures 3 and 4 of the article: ranking DataDecide recipes by loss, and benchmark accuracy of public corpora
against their predicted loss (panel c, written to fig2c).

a  predicted against measured 1B loss, 25 DataDecide recipes, leave-one-family-out (predict.json, level_1B|all11):
   ours_mismatch (filled blue) and base_ngram (open orange, the evaluation-blind colour of b); dashed equality line.
b  pairwise decision accuracy as bars from 0 with 95% family-bootstrap whiskers (scores[arm]["pairwise"],
   ["pairwise_95"]; unigram rivals from evalaware_rivals.json); bars coloured by group (blue, statistics that score
   the evaluation text, ours darkest; orange, statistics that do not; green, trained proxies); chance line.
c  FineWeb's 9-task score (data_choice_score/score.json, B1_fineweb_fig10_11) against predicted held-out loss
   (data_choice/forecast.json, primary forecast), direct corpus labels.
Run: python3 -I scripts/figures/cl/fig2.py
"""
import json
import os
import sys

import numpy as np
from matplotlib.patches import Rectangle as Rect

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style as _st  # noqa: E402
_st.set_type(8.0, 9.0, 8.5, 9.0)  # panel_cl24: ticks 8 pt, labels 9 pt, notes and legend 8.5 pt
from style import *  # noqa: E402,F401,F403

PREDICT = os.path.join(ROOT, "results/datadecide/predict.json")
VERIFY = os.path.join(ROOT, "results/datadecide/verify_headline.json")
TARGET = "level_1B|all11"
OURS_ARM = "ours_mismatch"
DATA_CHOICE = os.path.join(ROOT, "results/data_choice/forecast.json")
DATA_CHOICE_SCORE = os.path.join(ROOT, "results/data_choice_score/score.json")
RIVALS = os.path.join(ROOT, "results/datadecide/evalaware_rivals.json")
NAMES = {"fineweb_edu": "FineWeb-Edu", "fineweb": "FineWeb", "c4": "C4", "refinedweb": "RefinedWeb", "dolma": "Dolma",
         "slimpajama": "SlimPajama", "pile": "Pile"}
GROUPS = [
    ("Evaluation-aware", [
        ("ours_mismatch", "Bigram cross-entropy", "ours"),
        ("rival:token unigram", "Token unigram", "rival"),
        ("rival:word unigram", "Word unigram", "rival"),
        ("ours_select_ref_lm", "With reference LM*", "ours2"),
    ]),
    ("Evaluation-blind", [
        ("const_family_mean", "Family mean", "base"),
        ("base_compression", "Compression", "base"),
        ("base_ngram", "n-gram entropy", "base"),
        ("base_comp+ngram", "Compression + n-gram", "base"),
    ]),
    ("Trained proxies", [
        ("train_level_4M", "Proxy 4M", "train"),
        ("train_level_20M", "Proxy 20M", "train"),
        ("train_level_60M", "Proxy 60M", "train"),
        ("train_level_150M", "Proxy 150M", "train"),
    ]),
]


def load():  # verbatim from v42/fig2.py
    E = json.load(open(PREDICT))["E_heldout_family"]
    T = E["targets"][TARGET]
    units = E["units"]
    y = np.array([T["target"][u] for u in units])
    pred = np.array(T["predictions"][OURS_ARM])
    base_ng = np.array(T["predictions"]["base_ngram"])
    RV = json.load(open(RIVALS))["rows"]

    def score(arm, key="pairwise"):
        if arm.startswith("rival:"):
            return RV[arm[6:]][key]
        return T["scores"][arm][key]
    return y, pred, base_ng, score


def panel_a(fig, y, pred, base_ng):
    ax = axes_mm(fig, 16, 9, 52, 52); ax.set_gid("a")
    lo = min(y.min(), pred.min(), base_ng.min()) - 0.02
    hi = max(y.max(), pred.max(), base_ng.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], ls=(0, (3, 2)), lw=0.7, color=G2, gid="deco", zorder=1)
    ax.scatter(y, base_ng, s=MS_PT ** 2, facecolor="white", edgecolor=ORNG, lw=0.9, zorder=2, gid="base_ngram",
               label="n-gram entropy")
    ax.scatter(y, pred, s=MS_PT ** 2, facecolor=ACC, edgecolor="white", lw=0.4, zorder=3, gid="ours_mismatch",
               label="Train–evaluation mismatch")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    t = [v for v in (0.9, 1.0, 1.1) if lo <= v <= hi]
    ax.set_xticks(t); ax.set_yticks(t)
    ax.set_xlabel("Measured loss at 1B (bits per byte)")
    ax.set_ylabel("Predicted loss (bits per byte)")
    legend(ax, loc="upper left", reverse=True, fontsize=NOTE)
    letter(fig, "a", 1, 2)


GCOL = {"ours": ACC, "rival": "#B3CDEF", "ours2": "#6C9BDC", "base": ORNG, "train": GREEN}


def panel_b(fig, score):
    ax = axes_mm(fig, 122, 7, 36, 74); ax.set_gid("b")
    rows, groups = [], []
    yv = 0.0
    for head, items in GROUPS:
        top = yv
        for arm, lab, kind in items:
            rows.append((yv, arm, lab, kind)); yv -= 1.0
        groups.append((top, yv + 1.0, head))
        yv -= 0.7
    for yv, arm, lab, kind in rows:
        lo95, hi95 = score(arm, "pairwise_95")
        s = score(arm)
        ax.add_patch(Rect((0.0, yv - 0.29), s, 0.58, facecolor=GCOL[kind], edgecolor="none", zorder=2,
                          gid=f"bar:{arm}"))
        ax.plot([lo95, hi95], [yv, yv], color=INK, lw=0.9, solid_capstyle="butt", zorder=3, gid=f"ci:{arm}")
        for e in (lo95, hi95):
            ax.plot([e, e], [yv - 0.15, yv + 0.15], color=INK, lw=0.9, zorder=3, gid="deco")
    ax.axvline(0.5, ls=(0, (3, 2)), lw=0.8, color=INK, zorder=4, gid="deco")
    ax.set_yticks([r[0] for r in rows]); ax.set_yticklabels([r[2] for r in rows], fontsize=NOTE)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(rows[-1][0] - 0.7, rows[0][0] + 1.0)
    ax.set_xlim(0, 1); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("Pairwise decision accuracy", loc="right")  # panel_cl24: right-aligned so the 9 pt label fits
    ax.text(0.5, rows[0][0] + 0.75, "Chance", ha="center", va="bottom", fontsize=TICK, color=INK)
    # group brackets left of the row labels, labels rotated (classic grouped layout)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    xl = min(inv.transform(t.get_window_extent(r))[0][0] for t in ax.get_yticklabels())
    xb = xl - 1.4 / 36
    for (top, bot, head), col in zip(groups, (ACC, ORNG, GREEN)):
        ax.plot([xb, xb], [top + 0.4, bot - 0.4], transform=ax.get_yaxis_transform(), color=col, lw=1.0,
                clip_on=False, solid_capstyle="butt", gid="deco")
        # panel_cl19: two-line group labels so they fit the shorter rows at the larger type
        two = head.replace("Evaluation-", "Evaluation-\n").replace("Trained ", "Trained\n")
        ax.text(xb - 1.0 / 36, (top + bot) / 2, two, transform=ax.get_yaxis_transform(), rotation=90, ha="right",
                va="center", fontsize=NOTE, color=INK, linespacing=1.0, multialignment="center")
    letter(fig, "b", 74, 2)


def panel_c(fig):
    F_ = json.load(open(DATA_CHOICE))
    pb = F_["forecasts"][F_["primary"]]["predicted_bpb"]
    sc = json.load(open(DATA_CHOICE_SCORE))["sources"]["B1_fineweb_fig10_11"]["vals"]
    ax = axes_mm(fig, 16, 4, 76, 50); ax.set_gid("c")
    ks = sorted(sc, key=lambda k: pb[k])
    ax.scatter([pb[k] for k in ks], [100 * sc[k] for k in ks], s=MS_PT ** 2, facecolor=ACC, edgecolor="white", lw=0.4,
               zorder=3, gid="corpora")
    for k in ks:
        right = pb[k] < 0.95 or k == "c4"
        ax.annotate(NAMES[k], (pb[k], 100 * sc[k]), xytext=(3.5 if right else -3.5, 0), textcoords="offset points",
                    ha="left" if right else "right", va="center", fontsize=TICK)
    ax.set_xlim(0.83, 1.05); ax.set_ylim(45.8, 51.4)
    ax.set_xticks([0.85, 0.95, 1.05]); ax.set_xticklabels(["0.85", "0.95", "1.05"])
    ax.set_yticks([46, 48, 50])
    ax.set_xlabel("Predicted held-out loss (bits per byte)")
    ax.set_ylabel("Benchmark score (%)")


def main():
    y, pred, base_ng, score = load()
    # panel_cl19 split: v44 panels a, b -> fig2; v44 panel c -> fig2c (single panel, 0.6 of \textwidth, no letter).
    # Axes gids are kept, so fig2 + fig2c value dumps equal the v44 fig2 dump.
    fig = figure(94)
    panel_a(fig, y, pred, base_ng)
    panel_b(fig, score)
    save(fig, "fig2")
    fig = figure(66, 96)
    panel_c(fig)
    save(fig, "fig2c")
    no_smol = json.load(open(VERIFY))["results"][TARGET]["arms"]["ours_select_ref_minus_smollm2"]["pairwise"]
    print(f"caption *: without SmolLM2 the same procedure reaches {no_smol:.2f}")


if __name__ == "__main__":
    main()
