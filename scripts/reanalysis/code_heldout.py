"""Held-out scoring of the fall L(2^23)-L(2^25) for the CODE family, 3.3M learner, reusing infocurve_increment.py.
Calibration: fall_diagnosis.json rows kind=='code', set test80 (14 listed rows). Held out: forward18 + external45.
Lines (OLS on calibration, applied unchanged): own x, ngram23, L23, ngram23+x; scored against the calibration kind mean.
Version A: all corpora. Version B: drop corpora flagged r2_flag (document counts of the prepared corpora are not stored
in the stored files: prep_corpus.py writes *_docstarts.npy on the pods only, and prep logs report counts for 2 of 30 corpora).
Rule B was fixed before any number was computed.
    python scripts/reanalysis/code_heldout.py   # -> results/code_heldout/score.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infocurve_increment as II  # noqa: E402

CF = II.CF
OUT = os.path.join(CF.ROOT, "results", "code_heldout")
SPECS = {"kind_mean": [], "own_x": ["x"], "ngram23": ["ngram23"], "L23": ["L23"], "ngram23+x": ["ngram23", "x"]}


def run(cal, test):
    n = len(test)
    y = np.array([r["fall"] for r in test])
    pred = {}
    for k, cols in SPECS.items():
        if cols:
            pred[k] = II.X(test, cols) @ II.fit(cal, cols)
        else:
            pred[k] = np.full(n, np.mean([r["fall"] for r in cal]))
    E = {k: v - y for k, v in pred.items()}
    idx = np.random.default_rng(0).integers(0, n, (II.NB, n))
    R = lambda a, b: II.ratio_stats(E[a], E[b], idx)  # noqa: E731
    x = np.array([r["x"] for r in test])
    ps = {}
    for s in ("ngram23", "L23"):
        c = np.array([r[s] for r in test])
        ps[s] = dict(rho=II.partial_spearman(y, x, [c]), ci95=II.ci([II.partial_spearman(y[i], x[i], [c[i]]) for i in idx[:2000]]))
    return dict(n_cal=len(cal), n_test=n, heldout_corpora=[r["corpus"] for r in test],
                heldout_rms={k: CF.rms(v) for k, v in E.items()}, heldout_bias={k: float(v.mean()) for k, v in E.items()},
                vs_kind_mean={k: R(k, "kind_mean") for k in SPECS if k != "kind_mean"},
                own_vs_ngram23=R("own_x", "ngram23"), own_vs_L23=R("own_x", "L23"),
                combined_vs_ngram23=R("ngram23+x", "ngram23"), combined_vs_own=R("ngram23+x", "own_x"),
                partial_spearman_x_given=ps, spearman_x_vs_fall=dict(rho=II.spearman(x, y), ci95=II.ci([II.spearman(x[i], y[i]) for i in idx[:2000]])),
                lines={k: dict(cols=c, coef=[float(v) for v in II.fit(cal, c)] if c else [float(np.mean([r["fall"] for r in cal]))]) for k, c in SPECS.items()},
                rows=[dict(corpus=r["corpus"], fall=r["fall"], r2_flag=r["r2_flag"], pred={k: float(v[i]) for k, v in pred.items()}) for i, r in enumerate(test)])


def main():
    rows = [r for r in II.J("results/reanalysis/fall_diagnosis.json")["rows"] if r["kind"] == "code"]
    cal = [r for r in rows if r["set"] == "test80"]
    test = [r for r in rows if r["set"] in ("forward18", "external45")]
    res = {"A_all": run(cal, test)}
    res["B_drop_r2_flag"] = run([r for r in cal if not r["r2_flag"]], [r for r in test if not r["r2_flag"]])
    res["B_drop_r2_flag"]["dropped"] = [r["corpus"] for r in rows if r["r2_flag"]]
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(script=os.path.relpath(__file__, CF.ROOT), note="held-out split of stored outcomes, not a new forward test", **res),
              open(os.path.join(OUT, "score.json"), "w"), indent=1)
    for v, f in res.items():
        print("==", v, f["n_cal"], f["n_test"], {k: round(a, 3) for k, a in f["heldout_rms"].items()})
        for grp in ("vs_kind_mean",):
            for a, s in f[grp].items():
                print(f"  {a:10s}/kind_mean {s['ratio']:.3f} ({s['ci95'][0]:.3f}-{s['ci95'][1]:.3f}) {s['label']} {s['closer']}")
        for g in ("own_vs_ngram23", "own_vs_L23", "combined_vs_ngram23", "combined_vs_own"):
            s = f[g]; print(f"  {g:20s} {s['ratio']:.3f} ({s['ci95'][0]:.3f}-{s['ci95'][1]:.3f}) {s['label']} {s['closer']}")
        print("  pS", f["partial_spearman_x_given"], "rho", f["spearman_x_vs_fall"])


if __name__ == "__main__":
    main()
