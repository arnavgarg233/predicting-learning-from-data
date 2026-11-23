"""Code forecast: level / fall decomposition and a frozen code-calibrated test on ext45.
Stored forecasts and learning curves only; CPU; deterministic.
    python3 scripts/code_decomp/decomp.py -> results/code_decomp/decomp.json
Error convention (all b/B, at the seed-mean measured loss):
    e25 = forecast(2^25) - L(2^25) = level - fall,  level = forecast(2^23) - L(2^23),
    fall = predicted fall (2^23 -> 2^25) - measured fall.
Calibration corpora: the 14 test80 code corpora. Held out: the 11 ext45 code corpora (primary) and the
5 forward18 code corpora (secondary). Every calibrated constant below is fitted on the 14 only.
The ext45 'frozen revision' (ii) was fixed before ext45 (forecasters.json). The code-only variants (lam_code,
shared falls, n-gram level) were specified in this analysis after the ext45 outcomes were known."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "applicability"))
import common as C  # noqa: E402

OUT = os.path.join(ROOT, "results", "code_decomp")
NB = 20000
A_POOL = 0.14
FZ = C.J("results/extbench/forecasters.json")["forecaster_ii"]["constants"]
LAM, CODE_BIAS = FZ["lam"], FZ["code_bias"]
SET = {"test80": "orig14", "external45": "ext45", "forward18": "new5"}

svp = C.J("results/reanalysis/shape_vs_pooled.json")["per_corpus"]
fr = {(r["set"], r["corpus"]): r for r in C.fall_rows()}
s80 = C.J("results/curve_forecast/score80.json")["per_corpus"]
sn = C.J("results/curve_forecast/new_corpora/score.json")["per_corpus"]
eb = C.J("results/extbench/score.json")["per_corpus"]
cm = {(r["set"], r["corpus"]): r for r in C.J("results/costmatch/rows.json")}
CMSET = {"test80": "test80", "forward18": "new18", "external45": "ext45"}

rows, checks = [], {}
for r in svp:
    if r["family"] != "code":
        continue
    s, c = r["set"], r["corpus"]
    f = fr[(s, c)]
    x = r["S23"] - r["S25"]
    q = dict(set=SET[s], corpus=c, L20=r["L20"], L23=r["L23"], L25=r["L25"], fall=r["L23"] - r["L25"], x=x,
             lev_A0=r["lev_A0"], ngram23=f["ngram23"], c=f["c"], n_docs=f["n_docs"])
    if s == "external45":
        p = eb[c]
        q["A3_23"], q["A3_25"] = p["forecast"]["23"]["A3"], p["forecast"]["25"]["A3"]
        q["P2023_25"] = p["forecast"]["25"]["P2023"]
        checks.setdefault("ext45_ii_reproduced", []).append(abs(p["forecast"]["25"]["ii"] - (r["lev_A0"] - CODE_BIAS - LAM * x)))
    else:
        p = (s80 if s == "test80" else sn)[c]
        q["A3_23"], q["A3_25"] = p["forecast"]["23"]["A3"], p["forecast"]["25"]["A3"]
        checks.setdefault("A0_reproduced", []).append(abs(p["forecast"]["25"]["A0"] - (r["lev_A0"] - x)))
    k = cm[(CMSET[s], c)]["err"]
    checks.setdefault("A3_matches_costmatch", []).append(abs(q["A3_25"] - q["L25"] - k["A3"]))
    checks.setdefault("Xpool1_matches_costmatch", []).append(abs(q["L20"] * 2 ** (-A_POOL * 5) - q["L25"] - k["X_pool1"]))
    rows.append(q)
checks = {a: float(max(v)) for a, v in checks.items()}

G = {g: [q for q in rows if q["set"] == g] for g in ("orig14", "ext45", "new5")}
cal = G["orig14"]
assert len(cal) == 14 and len(G["ext45"]) == 11 and len(G["new5"]) == 5
A = lambda g, k: np.array([q[k] for q in G[g]])
# ---- constants fitted on the 14 calibration corpora
bias14 = float(np.mean(A("orig14", "lev_A0") - A("orig14", "L23")))
checks["code_bias_reproduced"] = abs(bias14 - CODE_BIAS)
lam_code = float(A("orig14", "x") @ A("orig14", "fall") / (A("orig14", "x") @ A("orig14", "x")))
mean_fall = float(np.mean(A("orig14", "fall")))
mean_logr = float(np.mean(np.log(A("orig14", "L23") / A("orig14", "L25"))))
ng_coef = C.ols([A("orig14", "ngram23")], A("orig14", "L23"))
# pooled exponent fitted on code14 alone (L25 = L20 * 2^(-5a))
a_code = float(np.mean(np.log2(A("orig14", "L20") / A("orig14", "L25")) / 5))
const = dict(code_bias=bias14, lam_pooled80=LAM, lam_code14=lam_code, mean_fall_code14=mean_fall,
             mean_log_ratio_code14=mean_logr, ngram_level_coef_code14=[float(v) for v in ng_coef],
             pooled_exponent_paper=A_POOL, pooled_exponent_code14=a_code)


def methods(q):
    """(pred23, pred25) per method."""
    lr = q["lev_A0"] - bias14                        # calibrated reference level
    ln = float(ng_coef[0] + ng_coef[1] * q["ngram23"])  # n-gram level
    m = {
        "A0 (paper forecast, as run)": (q["lev_A0"], q["lev_A0"] - q["x"]),
        "Rev (frozen revision ii: ref level - bias, 0.69 x info fall)": (lr, lr - LAM * q["x"]),
        "RevC (ref level - bias, code-fitted lam x info fall)": (lr, lr - lam_code * q["x"]),
        "Two early pilots (2^19+2^20)": (q["A3_23"], q["A3_25"]),
        "One pilot 2^20, pooled exponent 0.14": (q["L20"] * 2 ** (-A_POOL * 3), q["L20"] * 2 ** (-A_POOL * 5)),
        "One pilot 2^20, code-pooled exponent": (q["L20"] * 2 ** (-a_code * 3), q["L20"] * 2 ** (-a_code * 5)),
        "Ref level - bias + pooled-exponent fall": (lr, lr * 2 ** (-2 * A_POOL)),
        "Ref level - bias + code-mean fall": (lr, lr - mean_fall),
        "Ref level - bias + code-mean log-ratio fall": (lr, lr * np.exp(-mean_logr)),
        "n-gram level + code-mean fall": (ln, ln - mean_fall),
        "n-gram level + code-mean log-ratio fall": (ln, ln * np.exp(-mean_logr)),
        "Oracle level + info fall (lam code)": (q["L23"], q["L23"] - lam_code * q["x"]),
        "Oracle level + code-mean fall": (q["L23"], q["L23"] - mean_fall),
    }
    return m


M = list(methods(rows[0]).keys())
for q in rows:
    q["err"] = {}
    for k, (p23, p25) in methods(q).items():
        lev = p23 - q["L23"]; e25 = p25 - q["L25"]
        q["err"][k] = dict(level=float(lev), fall=float(lev - e25), e25=float(e25))

# leave-one-out versions of the calibrated arms on the 14 (honest in-sample comparison)
for i, q in enumerate(cal):
    o = [cal[j] for j in range(14) if j != i]
    b = np.mean([t["lev_A0"] - t["L23"] for t in o])
    xo = np.array([t["x"] for t in o]); fo = np.array([t["fall"] for t in o])
    lc = xo @ fo / (xo @ xo); mf = fo.mean()
    nc = C.ols([np.array([t["ngram23"] for t in o])], np.array([t["L23"] for t in o]))
    lr = q["lev_A0"] - b; ln = nc[0] + nc[1] * q["ngram23"]
    for k, (p23, p25) in {"RevC": (lr, lr - lc * q["x"]), "Rev": (lr, lr - LAM * q["x"]),
                          "RefLev+meanfall": (lr, lr - mf), "NgLev+meanfall": (ln, ln - mf)}.items():
        q.setdefault("loo", {})[k] = dict(level=float(p23 - q["L23"]), e25=float(p25 - q["L25"]))


def med_ratio(ea, eb, seed=0):
    n = len(ea); idx = np.random.default_rng(seed).integers(0, n, (NB, n))
    br = np.median(np.abs(ea[idx]), 1) / np.maximum(np.median(np.abs(eb[idx]), 1), 1e-30)
    lo, hi = np.percentile(br, [2.5, 97.5])
    return dict(ratio=float(np.median(np.abs(ea)) / np.median(np.abs(eb))), ci95=[float(lo), float(hi)])


def rms_ci(e, seed=0):
    n = len(e); idx = np.random.default_rng(seed).integers(0, n, (NB, n))
    v = np.sqrt((e[idx] ** 2).mean(1)); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def decomp(sub, k):
    l = np.array([q["err"][k]["level"] for q in sub]); f = np.array([q["err"][k]["fall"] for q in sub])
    e = l - f
    ms = dict(level=float(np.mean(l ** 2)), fall=float(np.mean(f ** 2)), cross=float(-2 * np.mean(l * f)))
    return dict(rms_e25=C.rms(e), rms_level=C.rms(l), rms_fall=C.rms(f), mean_level=float(l.mean()), mean_fall=float(f.mean()),
                ms_terms=ms, rms_no_cancel=float(np.sqrt(ms["level"] + ms["fall"])),
                corr_level_fall=float(np.corrcoef(l, f)[0, 1]),
                n_same_sign=int(np.sum(np.sign(l) == np.sign(f))), n=len(sub))


P = "Two early pilots (2^19+2^20)"


def score(sub, keyfn=lambda q, k: q["err"][k]["e25"], ks=M):
    e2 = np.array([q["err"][P]["e25"] for q in sub])
    out = {}
    for k in ks:
        e = np.array([keyfn(q, k) for q in sub])
        out[k] = dict(rms=C.rms(e), rms_ci=rms_ci(e), median_abs=float(np.median(np.abs(e))), worst=float(np.max(np.abs(e))),
                      vs_pilots=C.ratio(e, e2), median_abs_vs_pilots=med_ratio(e, e2))
    return out


def head(sub, a, b):
    ea = np.array([q["err"][a]["e25"] for q in sub]); ebb = np.array([q["err"][b]["e25"] for q in sub])
    return dict(rms=C.ratio(ea, ebb), median_abs=med_ratio(ea, ebb))


res = dict(constants=const, checks=checks, methods=M)
C1 = lambda q: q["n_docs"] >= 75
subsets = {"orig14": cal, "ext45": G["ext45"], "ext45_C1": [q for q in G["ext45"] if C1(q)],
           "new5": G["new5"], "held16": G["ext45"] + G["new5"], "held16_C1": [q for q in G["ext45"] + G["new5"] if C1(q)]}
res["n"] = {k: len(v) for k, v in subsets.items()}
res["ext45_C1_dropped"] = [q["corpus"] for q in G["ext45"] if not C1(q)]
res["decomp"] = {g: {k: decomp(sub, k) for k in M} for g, sub in subsets.items()}
res["score"] = {g: score(sub) for g, sub in subsets.items()}
loo_keys = ["RevC", "Rev", "RefLev+meanfall", "NgLev+meanfall"]
e2c = np.array([q["err"][P]["e25"] for q in cal])
res["orig14_loo"] = {k: dict(rms=C.rms(np.array([q["loo"][k]["e25"] for q in cal])),
                             vs_pilots=C.ratio(np.array([q["loo"][k]["e25"] for q in cal]), e2c),
                             median_abs_vs_pilots=med_ratio(np.array([q["loo"][k]["e25"] for q in cal]), e2c)) for k in loo_keys}
ea = np.array([q["loo"]["RevC"]["e25"] for q in cal]); ebb = np.array([q["loo"]["RefLev+meanfall"]["e25"] for q in cal])
res["orig14_loo"]["RevC / RefLev+meanfall"] = dict(rms=C.ratio(ea, ebb), median_abs=med_ratio(ea, ebb))
ea = np.array([q["loo"]["Rev"]["e25"] for q in cal])
res["orig14_loo"]["Rev / RefLev+meanfall"] = dict(rms=C.ratio(ea, ebb), median_abs=med_ratio(ea, ebb))
# information-curve increment over calibrated level + shared shape (same level, only the fall differs)
INC = [("Rev (frozen revision ii: ref level - bias, 0.69 x info fall)", "Ref level - bias + code-mean fall"),
       ("RevC (ref level - bias, code-fitted lam x info fall)", "Ref level - bias + code-mean fall"),
       ("RevC (ref level - bias, code-fitted lam x info fall)", "Ref level - bias + code-mean log-ratio fall"),
       ("RevC (ref level - bias, code-fitted lam x info fall)", "n-gram level + code-mean fall"),
       ("Oracle level + info fall (lam code)", "Oracle level + code-mean fall"),
       ("A0 (paper forecast, as run)", "Ref level - bias + code-mean fall")]
res["increment"] = {g: {f"{a} / {b}": head(sub, a, b) for a, b in INC} for g, sub in subsets.items()}
res["per_corpus"] = [{k: v for k, v in q.items() if k != "loo"} | ({"loo": q["loo"]} if "loo" in q else {}) for q in rows]
json.dump(res, open(os.path.join(OUT, "decomp.json"), "w"), indent=1)
print(json.dumps(dict(constants=const, checks=checks, n=res["n"], dropped=res["ext45_C1_dropped"]), indent=1))
