"""Resample the 11 DataDecide evaluation sets: interval of the information-curve association with the rate.

    python scripts/evalset_rate_bootstrap.py [--out PATH]   # adds `evalset_bootstrap_gpt2` to results/datadecide/evalset_rate_test.json

Reads the per-set targets of results/datadecide/evalset_rate_test.json (written by scripts/evalset_rate_test.py) and the
per-set statistics of results/datadecide/stats/merged.json. For 20,000 draws (seed 0) the 11 sets are resampled with
replacement (draws with fewer than 4 distinct sets are skipped), and the Spearman correlation of GPT-2's gamma with the mean
ladder exponent is recomputed, with its differences from the formula of Cagnetta et al. (gamma / 2 beta) and from lzma
compressibility (sign flipped). Reported: 2.5th and 97.5th percentiles, rounded to two decimals.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F = os.path.join(ROOT, "results/datadecide/evalset_rate_test.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=F)
    a = ap.parse_args()
    r = json.load(open(F))
    S = json.load(open(os.path.join(ROOT, "results/datadecide/stats/merged.json")))["sources"]
    E = r["evals"]
    y = np.array(r["target_mean_alpha_large"])
    g = np.array([S["ev_" + e]["ref"]["gpt2"]["gamma"]["4-64"]["gamma"] for e in E])
    b = np.array([S["ev_" + e]["beta"]["beta"]["tok_1_200_floorcorr"] for e in E])
    lz = np.array([S["ev_" + e]["compression"]["lzma_bpb_8MB"] for e in E])
    rk = lambda x: pd.Series(x).rank().to_numpy()  # noqa: E731
    sp = lambda u, v: np.corrcoef(rk(u), rk(v))[0, 1]  # noqa: E731
    rng = np.random.default_rng(0)
    D = {"cag": [], "lzma": [], "ours": []}
    for _ in range(20000):
        i = rng.integers(0, 11, 11)
        if len(set(i)) < 4:
            continue
        o = sp(g[i], y[i])
        D["ours"].append(o)
        D["cag"].append(o - sp(g[i] / (2 * b[i]), y[i]))
        D["lzma"].append(o - (-sp(lz[i], y[i])))
    q = lambda v: [round(float(np.nanpercentile(v, 2.5)), 2), round(float(np.nanpercentile(v, 97.5)), 2)]  # noqa: E731
    r["evalset_bootstrap_gpt2"] = dict(ours=q(D["ours"]), ours_minus_cagnetta=q(D["cag"]), ours_minus_lzma=q(D["lzma"]),
                                       note="resample the 11 eval sets with replacement (20k draws, >=4 distinct)")
    json.dump(r, open(a.out, "w"), indent=1)
    print("ours", q(D["ours"]), "ours - Cagnetta", q(D["cag"]), "ours - lzma", q(D["lzma"]))


if __name__ == "__main__":
    main()
