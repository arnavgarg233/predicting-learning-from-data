"""New-corpora forward test: additions to score.json.

    python scripts/new_corpora_analysis.py   # -> results/curve_forecast/new_corpora/analysis.json

Reads score.json (the unchanged scorer's output) and forecast.json of this run. Adds, at 2^25 and 2^23, by family and overall:
r.m.s. of A0 (data alone), A3 (two-pilot power law) and X_pool1 (one pilot at 2^20 with the zero-floor exponent a = 0.14 pooled on
development); ratios A0/A3 and A0/X_pool1 with corpus-bootstrap 95% intervals
(score.boot_ratio, 20,000 draws, seed 0) and closer-counts; and the reference-competence ratio c = reference full-context bits per
byte / best n-gram cross-entropy at 2^23 for each corpus (flag c >= 0.70).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "curve_forecast"))
import score as SC  # noqa: E402

D = os.path.join(ROOT, "results/curve_forecast/new_corpora")
A_POOL = 0.14


def main():
    S = json.load(open(os.path.join(D, "score.json")))
    F = json.load(open(os.path.join(D, "forecast.json")))
    per = S["per_corpus"]
    names = list(per)
    rows = {}
    for c in names:
        p = per[c]
        st = F["corpora"][c]["ref_stats"]
        L20 = p["pilots"]["20"]
        r = dict(family=p["family"], c_ratio=st["ref_full"] / st["ngram23"], ref_full=st["ref_full"], ngram23=st["ngram23"],
                 pilots=p["pilots"], measured=p["measured"], steps=p["steps"])
        for k in ("23", "25"):
            m = p["measured"][k]
            xp = float(L20 * 2.0 ** (-A_POOL * (int(k) - 20)))
            r[f"err_{k}"] = dict(A0=p["err"][k]["A0"], A3=p["err"][k]["A3"], X_pool1=xp - m, A2=p["err"][k]["A2"], A4=p["err"][k]["A4"])
            r[f"forecast_{k}"] = dict(A0=p["forecast"][k]["A0"], A3=p["forecast"][k]["A3"], X_pool1=xp)
        rows[c] = r
    out = dict(source="score.json + forecast.json of this run", a_pool=A_POOL, per_corpus=rows, blocks={})
    rng = np.random.default_rng(0)
    groups = {"code": [c for c in names if rows[c]["family"] == "code"], "english": [c for c in names if rows[c]["family"] == "english"],
              "wiki": [c for c in names if rows[c]["family"] == "wiki"], "all": names}
    for k in ("25", "23"):
        blk = {}
        for g, cs in groups.items():
            E = {a: np.array([rows[c][f"err_{k}"][a] for c in cs]) for a in ("A0", "A3", "X_pool1")}
            b = dict(n=len(cs), rms={a: float(np.sqrt(np.mean(E[a] ** 2))) for a in E}, bias={a: float(np.mean(E[a])) for a in E})
            for y in ("A3", "X_pool1"):
                ci, share = SC.boot_ratio(E["A0"], E[y], rng)
                b[f"A0_over_{y}"] = dict(ratio=b["rms"]["A0"] / b["rms"][y], ci95=ci, share_draws_below_1=share,
                                         A0_closer=f"{int(np.sum(np.abs(E['A0']) < np.abs(E[y])))}/{len(cs)}")
            blk[g] = b
        out["blocks"][k] = blk
    json.dump(out, open(os.path.join(D, "analysis.json"), "w"), indent=1)
    for k in ("25", "23"):
        print(f"2^{k}")
        for g, b in out["blocks"][k].items():
            print(f"  {g:8s} n={b['n']:2d} A0 {b['rms']['A0']:.3f} A3 {b['rms']['A3']:.3f} Xp1 {b['rms']['X_pool1']:.3f} | "
                  f"A0/A3 {b['A0_over_A3']['ratio']:.2f} [{b['A0_over_A3']['ci95'][0]:.2f},{b['A0_over_A3']['ci95'][1]:.2f}] {b['A0_over_A3']['A0_closer']} | "
                  f"A0/Xp1 {b['A0_over_X_pool1']['ratio']:.2f} [{b['A0_over_X_pool1']['ci95'][0]:.2f},{b['A0_over_X_pool1']['ci95'][1]:.2f}] {b['A0_over_X_pool1']['A0_closer']}")
    for c, r in rows.items():
        e = r["err_25"]
        print(f"{c:16s} {r['family']:7s} c={r['c_ratio']:.2f}{' *' if r['c_ratio'] >= 0.7 else '  '} meas25={r['measured']['25']:.3f} "
              f"A0 {e['A0']:+.3f} A3 {e['A3']:+.3f} Xp1 {e['X_pool1']:+.3f} | 23: A0 {r['err_23']['A0']:+.3f} A3 {r['err_23']['A3']:+.3f}")


if __name__ == "__main__":
    main()
