"""sensitivity (2026-11-05), part C: single-pass 40-corpus rates (3.3M learner) and 20 languages (25M learner) under every
stored floor choice, plus the reference loss itself as the floor (E = ref_full, capped at 0.95 min L as the stored rules).
Reads results/rate_singlepass/analysis.json and results/rate_singlepass25m/analysis.json."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results/sensitivity/singlepass_floors.json")
LN2 = np.log(2)
KS = (22, 23, 24, 25, 26, 27)


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def ci(x, y, B=10000):
    x, y = np.asarray(x), np.asarray(y)
    i = np.random.default_rng(0).integers(0, len(x), (B, len(x)))
    b = [spearman(x[j], y[j]) for j in i]
    return [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]


out = {}
for tag, p in (("3.3M_40", "results/rate_singlepass/analysis.json"), ("25M_20lang", "results/rate_singlepass25m/analysis.json")):
    d = json.load(open(os.path.join(ROOT, p)))
    pcs = d["per_corpus"]
    for r in pcs:
        L = np.array([r["L"][str(k)] for k in KS])
        E = min(r["ref_full"], 0.95 * L.min())
        r["rates"]["ref 1.0"] = float(-np.polyfit(np.array(KS) * LN2, np.log(L - E), 1)[0])
        r["ref_above_minL"] = bool(r["ref_full"] >= L.min())
    keys = list(pcs[0]["rates"])
    res = {}
    for st in ("all", "language", "english"):
        sub = [r for r in pcs if st == "all" or r["family"] == st]
        if len(sub) < 5:
            continue
        g = [r["gamma_ref"] for r in sub]
        blk = {}
        for k in keys:
            v = [r["rates"][k] for r in sub]
            blk[k] = dict(rho_gamma=spearman(g, v), ci95=ci(g, v), median_rate=float(np.median(v)),
                          rho_vs_raw=spearman(v, [r["rates"]["raw"] for r in sub]))
        res[st] = dict(n=len(sub), n_ref_above_minL=int(sum(r["ref_above_minL"] for r in sub)), floors=blk)
    out[tag] = res
    print("==", tag)
    for st, b in res.items():
        print(st, b["n"], "ref>=minL", b["n_ref_above_minL"])
        for k, v in b["floors"].items():
            print(f"   {k:10s} rho {v['rho_gamma']:+.2f} [{v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f}] median rate {v['median_rate']:.3f} rho_vs_raw {v['rho_vs_raw']:+.2f}")
json.dump(out, open(OUT, "w"), indent=1)
