"""Stage 1 of the applicability analysis: statistics of the CALIBRATION corpora only.
Reads no learner outcome of any held-out corpus.   python scripts/applicability/calib.py -> calibration.json"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

rows = C.fall_rows()
out = dict(kinds={})
for kind, sets in C.CAL.items():
    allcal = [r for r in rows if r["kind"] == kind and r["set"] in sets]
    cal = [r for r in allcal if not (kind == "english" and r["corpus"] in C.SET_ASIDE_EN)]
    x = np.array([r["x"] for r in cal]); y = np.array([r["fall"] for r in cal]); g = np.array([r["ngram23"] for r in cal])
    xa = np.array([r["x"] for r in allcal])
    e_mean = C.loo_errors([], y); e_own = C.loo_errors([x], y); e_ng = C.loo_errors([g], y); e_ngx = C.loo_errors([g, x], y)
    n = len(cal)
    out["kinds"][kind] = dict(
        n_all=len(allcal), n=n, corpora=[r["corpus"] for r in cal],
        n_docs_min=int(min(r["n_docs"] for r in allcal)), n_docs_median=float(np.median([r["n_docs"] for r in allcal])),
        c_median=float(np.median([r["c"] for r in allcal])), c_range=[float(min(r["c"] for r in allcal)), float(max(r["c"] for r in allcal))],
        share_c_ge_070=float(np.mean([r["c"] >= 0.70 for r in allcal])),
        x_median=float(np.median(xa)), x_scaled_mad=C.scaled_mad(xa),
        x_window=[float(np.median(xa) - 4 * C.scaled_mad(xa)), float(np.median(xa) + 4 * C.scaled_mad(xa))],
        loo_rms=dict(kind_mean=C.rms(e_mean), own=C.rms(e_own), ngram23=C.rms(e_ng), ngram23_own=C.rms(e_ngx)),
        loo_own_vs_mean=C.ratio(e_own, e_mean), loo_ngx_vs_ng=C.ratio(e_ngx, e_ng), loo_own_vs_ng=C.ratio(e_own, e_ng),
        spearman_x_fall=dict(rho=C.spearman(x, y), ci95=C.boot_ci(lambda i: C.spearman(x[i], y[i]), n)),
        partial_spearman_x_fall_given_ngram23=dict(rho=C.partial_spearman(y, x, g), ci95=C.boot_ci(lambda i: C.partial_spearman(y[i], x[i], g[i]), n)),
        spearman_x_ngram23=C.spearman(x, g), spearman_ngram23_fall=C.spearman(g, y))

# whole-curve forecast on the 80 test corpora (the calibration set of the c threshold)
S = C.J("results/curve_forecast/score80.json")["per_corpus"]
byc = {r["corpus"]: r for r in rows if r["set"] == "test80"}
cur = []
for c, p in S.items():
    r = byc[c]
    cur.append(dict(corpus=c, kind=r["kind"], c=r["c"], n_docs=r["n_docs"], e0=p["err"]["25"]["A0"], e3=p["err"]["25"]["A3"]))
assert len(cur) == 80
e0 = np.array([q["e0"] for q in cur]); e3 = np.array([q["e3"] for q in cur]); hi = np.array([q["c"] >= 0.70 for q in cur])
out["curve_test80"] = dict(
    n=80, n_docs_min=int(min(q["n_docs"] for q in cur)), n_c_ge_070=int(hi.sum()),
    trusted=dict(n=int((~hi).sum()), rms_A0=C.rms(e0[~hi]), rms_A3=C.rms(e3[~hi]), mae_A0=float(np.mean(np.abs(e0[~hi]))),
                 share_large_A0=float(np.mean(np.abs(e0[~hi]) > 0.25)), share_A0_closer=float(np.mean(np.abs(e0[~hi]) < np.abs(e3[~hi])))),
    flagged=dict(n=int(hi.sum()), rms_A0=C.rms(e0[hi]), rms_A3=C.rms(e3[hi]), mae_A0=float(np.mean(np.abs(e0[hi]))),
                 share_large_A0=float(np.mean(np.abs(e0[hi]) > 0.25)), share_A0_closer=float(np.mean(np.abs(e0[hi]) < np.abs(e3[hi])))))
json.dump(out, open(os.path.join(C.OUT, "calibration.json"), "w"), indent=1)
for k, v in out["kinds"].items():
    print(k, {a: v[a] for a in ("n_all", "n", "n_docs_min", "c_median", "share_c_ge_070", "x_window")})
    print("   loo rms", {a: round(b, 4) for a, b in v["loo_rms"].items()})
    for a in ("loo_own_vs_mean", "loo_ngx_vs_ng", "loo_own_vs_ng"):
        print(f"   {a:16s} {v[a]['ratio']:.3f} ({v[a]['ci95'][0]:.3f}-{v[a]['ci95'][1]:.3f}) {v[a]['label']} {v[a]['closer']}")
    print("   spearman x,fall", v["spearman_x_fall"], "partial|ngram", v["partial_spearman_x_fall_given_ngram23"],
          "rho(x,ngram)", round(v["spearman_x_ngram23"], 2), "rho(ngram,fall)", round(v["spearman_ngram23_fall"], 2))
print("curve test80", json.dumps(out["curve_test80"]))
