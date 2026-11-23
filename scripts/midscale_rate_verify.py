"""Bytes and GPT-2 BPE data-scaling exponents of the 100M-parameter single-pass runs, fitted jointly.

    python scripts/midscale_rate_verify.py [--out PATH]   # -> results/midscale_rate_verify.json

Runs: results/midscale/*.json with 12 layers, no embedding factorisation, one pass and 2^25-2^29 training bytes; the
target is each cell's mean test bits per byte over its two seeds. Model: L = E + A_enc (D / 2^25)^(-alpha_enc), one floor
E shared by both encodings (free, or fixed at 0.6 or 0.8), least squares from three starting exponents. Intervals:
seed bootstrap (each cell's two seeds resampled with replacement, 2,000 draws) and a parametric residual bootstrap
(2,000 draws), from one generator with seed 0 used in the order free floor, floor 0.6, floor 0.8, free floor on
2^26-2^29. The ratio is alpha_bytes / alpha_gpt2.
"""
import argparse
import collections
import glob
import json
import os

import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "midscale_rate_verify.json"))
    a = ap.parse_args()
    cells = collections.defaultdict(dict)  # (encoding, train_bytes) -> {seed: test bpb}
    for f in sorted(glob.glob(os.path.join(R, "midscale/*.json"))):
        d = json.load(open(f))
        if d["nl"] == 12 and d["emb"] == 0 and d["passes"] == 1 and 25 <= np.log2(d["train_bytes"]) <= 29:
            cells[(d["encoding"], d["train_bytes"])][d["seed"]] = d["test_bpb"]
    # The seed bootstrap draws from each cell's seeds in the order the original run read the files (directory order);
    # fixing that order here makes the published intervals reproduce exactly on any file system.
    first = {("gpt2", 2 ** 26): 0, ("gpt2", 2 ** 29): 0}
    cells = {k: {s: v[s] for s in sorted(v, key=lambda s: s != first.get(k, 1))} for k, v in cells.items()}
    rng = np.random.default_rng(0)

    def mean_data(c, minlog=25):
        ks = sorted(k for k in c if np.log2(k[1]) >= minlog)
        return ks, np.array([np.mean(list(c[k].values())) for k in ks])

    def fit(ks, y, E=None):
        enc = np.array([k[0] == "gpt2" for k in ks])
        D = np.array([k[1] for k in ks]) / 2 ** 25
        o = 1 if E is None else 0

        def model(p):
            e = p[0] if E is None else E
            A = np.where(enc, p[o + 2], p[o])
            al = np.where(enc, p[o + 3], p[o + 1])
            return e + A * D ** (-al)
        best = None
        for a0 in (0.2, 0.35, 0.6):
            p0 = ([0.6] if E is None else []) + [1, a0, 1, a0]
            lo = ([0] if E is None else []) + [0, 0, 0, 0]
            hi = ([y.min() - 1e-6] if E is None else []) + [50, 3, 50, 3]
            if E is None:
                p0[0] = min(p0[0], y.min() - 0.05)
            r = least_squares(lambda p: model(p) - y, p0, bounds=(lo, hi))
            if best is None or r.cost < best.cost:
                best = r
        p = best.x
        return dict(E=float(p[0]) if E is None else E, alpha_bytes=float(p[o + 1]), alpha_gpt2=float(p[o + 3]),
                    ratio=float(p[o + 1] / p[o + 3]), rss=float(2 * best.cost)), model(p)

    def run(minlog, E, nb=2000):
        ks, y = mean_data(cells, minlog)
        res, fitted = fit(ks, y, E)
        out = dict(res)
        out["n_points"] = len(ks)
        assert all(len(cells[k]) == 2 for k in ks)
        bs = []
        for _ in range(nb):
            yb = np.array([np.mean(rng.choice(list(cells[k].values()), 2)) for k in ks])
            try:
                bs.append(fit(ks, yb, E)[0])
            except Exception:
                pass
        for q in ("E", "alpha_bytes", "alpha_gpt2", "ratio"):
            v = np.array([b[q] for b in bs])
            out[q + "_ci95"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out["ci_method"] = "seed bootstrap (resample 2 seeds with replacement per cell, %d draws; only 2 seeds so coarse)" % len(bs)
        dof = max(len(ks) - (5 if E is None else 4), 1)
        s = np.sqrt(res["rss"] / dof)
        pb = []
        for _ in range(nb):
            try:
                pb.append(fit(ks, fitted + rng.normal(0, s, len(ks)), E)[0])
            except Exception:
                pass
        for q in ("E", "alpha_bytes", "alpha_gpt2", "ratio"):
            v = np.array([b[q] for b in pb])
            out[q + "_ci95_param"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        out["param_resid_sigma"] = float(s)
        return out

    o = {"cell_means": {f"{k[0]}_2^{int(np.log2(k[1]))}": float(np.mean(list(v.values()))) for k, v in sorted(cells.items())},
         "free_E_all": run(25, None), "fixed_E_0.6": run(25, 0.6), "fixed_E_0.8": run(25, 0.8),
         "free_E_2^26-29": run(26, None), "note": "ratio = alpha_bytes/alpha_gpt2; bytes lacks 2^25 data"}
    json.dump(o, open(a.out, "w"), indent=1)
    for k, v in o.items():
        if isinstance(v, dict) and "ratio" in v:
            print(k, "ratio %.3f" % v["ratio"], "seed CI", np.round(v["ratio_ci95"], 3).tolist(),
                  "param CI", np.round(v["ratio_ci95_param"], 3).tolist())


if __name__ == "__main__":
    main()
