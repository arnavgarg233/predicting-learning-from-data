"""Budget-window local slopes of the single-pass rate study, and the fitted free floors against the reference loss.

    python scripts/rate_singlepass/window_slopes.py DIR [--family language] [--out DIR/window_slopes.json]

For each complete corpus of the family: raw slope a = -d log L / d log D over three consecutive budgets (2^22-2^24, 2^23-2^25,
2^24-2^26, 2^25-2^27), Spearman with gamma_ref and a 95% corpus-bootstrap interval (10,000 resamples, seed 0). Also the free floor
of analyze.fit3 over all six budgets: how many corpora have a non-zero floor and how many of those lie above ref_full.
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

WINDOWS = ((22, 23, 24), (23, 24, 25), (24, 25, 26), (25, 26, 27))


def boot(x, y, B=10000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(x), (B, len(x)))
    i = i[np.array([len(np.unique(r)) >= 4 for r in i])]
    rx, ry = stats.rankdata(x[i], axis=1), stats.rankdata(y[i], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = A._rows_pearson(rx, ry)
    return float(stats.spearmanr(x, y)[0]), [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--family", default="language")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [r for r in A.load(a.dir) if r["complete"] and r["family"] == a.family]
    g = [r["gamma_ref"] for r in rows]
    res = dict(dir=a.dir, family=a.family, n=len(rows), windows={})
    for w in WINDOWS:
        s = [A.slope(w, [r["L"][k] for k in w]) for r in rows]
        rho, ci = boot(g, s)
        res["windows"][f"{w[0]}-{w[-1]}"] = dict(spearman=rho, ci=ci, median_slope=float(np.median(s)))
    fl = []
    for r in rows:
        _, E = A.fit3(A.BUDGETS, [r["L"][k] for k in A.BUDGETS])
        fl.append(dict(name=r["name"], E=E, ref_full=r["ref_full"], L27=r["L"][27]))
    nz = [f for f in fl if f["E"] > 0]
    res["floors"] = dict(nonzero=len(nz), above_ref_full=sum(f["E"] > f["ref_full"] for f in nz), per_corpus=fl)
    for k, v in res["windows"].items():
        print(f"{k}: {v['spearman']:+.2f} ({v['ci'][0]:+.2f}, {v['ci'][1]:+.2f})  median slope {v['median_slope']:.4f}")
    print(f"n {len(rows)}; free floor non-zero on {len(nz)}, above ref_full on {res['floors']['above_ref_full']}")
    json.dump(res, open(a.out or os.path.join(a.dir, f"window_slopes_{a.family}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
