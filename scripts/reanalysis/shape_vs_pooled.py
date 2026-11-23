"""Does each corpus's own information-curve shape forecast the fall of the learning curve better than a pooled shape,
at a matched level? Design: docs/analysis/shape_vs_pooled_2026-10-30.md. Reads stored JSON only (and FineWeb-Edu's
reference curve); no training. Deterministic.

    python scripts/referee_reanalysis/shape_vs_pooled.py      # -> results/referee_reanalysis/shape_vs_pooled.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "curve_forecast"))
import core as C  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "referee_reanalysis", "shape_vs_pooled.json")
NB = 20000
A_POOL = 0.14
LAM = 0.6924039665645343                      # scripts/extbench/forecast_ii.py (fitted on test80)
NGRAM = (-0.09618305026700236, 0.8134058678431364)
CODE_BIAS = 0.2011360393894477
FAM = {"code": "code", "english": "english", "wiki": "lang", "lang": "lang"}
DEV_FAM = dict(code=("code", "java_code"),
               english=("pg19", "pubmed", "arxiv", "cc_news", "openwebmath", "stackexchange", "tinystories", "fineweb_edu"),
               lang=("ja_wiki", "de_wiki", "zh_wiki"))
NOISE_FALL = 0.0126 * np.sqrt(1.5)
OWN = ("own_S", "own_lamS")
POOLED = ("pool_exp", "dev_fall_bpb", "dev_fall_log", "fw_info", "fw_meas")
CONTEXT = ("A3", "loo_mean")
ARMS = OWN + POOLED + CONTEXT
PLACE = ("primary", "A0", "rev", "oracle")
PRIMARY_SETS = ("test80", "forward18", "external45")

# ------------------------------------------------------------------ pooled constants (development only)
lk = json.load(open(os.path.join(RES, "curve_forecast", "locked.json")))
K = lk["constants"]
dev = json.load(open(os.path.join(RES, "curve_forecast", "development.json")))["corpora"]
devL = {c: {int(k): v for k, v in d["measured"].items()} for c, d in dev.items()}
dev_fall = {f: float(np.mean([devL[c][21] - devL[c][23] for c in cs])) for f, cs in DEV_FAM.items()}
dev_log = {f: float(np.mean([np.log(devL[c][21] / devL[c][23]) for c in cs])) for f, cs in DEV_FAM.items()}
FW = json.load(open(os.path.join(RES, "curve_forecast", "score80.json")))["fineweb_curve"]
FW = {int(k): v for k, v in FW.items()}
hr_fw = C.FN.HRef(os.path.join(RES, "bfine", "refcurve", "refcurve_fineweb.json"), "SmolLM2-135M")
st_fw = C.ref_stats(hr_fw)
nu_fw, n0_fw = C.shape_params("gamma", st_fw, K)
S_FW = {k: C.S(hr_fw, nu_fw, n0_fw, k) for k in (20, 23, 25)}

# ------------------------------------------------------------------ rows
rows = []
for r in json.load(open(os.path.join(RES, "referee_reanalysis", "nl_table.json"))):
    fam = FAM[r["family"]]
    g = {"test80": "test80", "new18": "forward18", "stage3": "stage3"}[r["group"]]
    rev = r["A0_23"] - CODE_BIAS if fam == "code" else (r["A0_23"] if fam == "english" else NGRAM[0] + NGRAM[1] * r["g"])
    rows.append(dict(corpus=r["corpus"], set=g, family=fam, L20=r["L20"], L23=r["L23"], L25=r["L25"],
                     S20=r["S20"], S23=r["S23"], S25=r["S25"], lev_A0=r["A0_23"], lev_rev=rev,
                     A3_fall=r["A3_23"] - r["A3_25"]))
eb = json.load(open(os.path.join(RES, "extbench_2026-10-28", "score.json")))["per_corpus"]
FI = {}
for p in (1, 2, 3):
    FI.update(json.load(open(os.path.join(RES, "extbench_2026-10-28", f"pod{p}", "forecast_i.json")))["corpora"])
for c, r in eb.items():
    S = FI[c]["S"]; f = r["forecast"]
    rows.append(dict(corpus=c, set="external45", family=FAM[r["family"]], L20=r["L"]["20"], L23=r["L"]["23"], L25=r["L"]["25"],
                     S20=S["20"], S23=S["23"], S25=S["25"], lev_A0=f["23"]["i"], lev_rev=f["23"]["ii"],
                     A3_fall=f["23"]["A3"] - f["25"]["A3"]))
    assert abs(FI[c]["A0"]["23"] - f["23"]["i"]) < 1e-12
for r in rows:
    r["fall"] = r["L23"] - r["L25"]
    r["fall20"] = r["L20"] - r["L25"]
    r["lev_primary"] = r["lev_rev"] if r["set"] == "external45" else r["lev_A0"]
    r["lev_oracle"] = r["L23"]
# leave-one-out mean fall within set x family (context only; uses other test corpora)
for r in rows:
    o = [q["fall"] for q in rows if q["set"] == r["set"] and q["family"] == r["family"] and q is not r]
    r["loo_fall"] = float(np.mean(o))


def fc_fall(r, arm, V):
    return dict(own_S=r["S23"] - r["S25"], own_lamS=LAM * (r["S23"] - r["S25"]),
                pool_exp=V * (1 - 2.0 ** (-2 * A_POOL)), dev_fall_bpb=dev_fall[r["family"]],
                dev_fall_log=V * (1 - np.exp(-dev_log[r["family"]])), fw_info=S_FW[23] - S_FW[25],
                fw_meas=FW[23] - FW[25], A3=r["A3_fall"], loo_mean=r["loo_fall"])[arm]


def fc_fall20(r, arm):
    return dict(own_S=r["S20"] - r["S25"], pool_exp=r["L20"] * (1 - 2.0 ** (-5 * A_POOL)),
                fw_info=S_FW[20] - S_FW[25], fw_meas=FW[20] - FW[25])[arm]


for r in rows:
    r["fall_err"], r["e25"] = {}, {}
    for p in PLACE:
        V = r[f"lev_{p}"]
        r["fall_err"][p] = {a: float(fc_fall(r, a, V) - r["fall"]) for a in ARMS}
        r["e25"][p] = {a: float(V - fc_fall(r, a, V) - r["L25"]) for a in ARMS}
    r["fall20_err"] = {a: float(fc_fall20(r, a) - r["fall20"]) for a in ("own_S", "pool_exp", "fw_info", "fw_meas")}


# ------------------------------------------------------------------ statistics
def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else float("nan")


def ratio(ea, eb_, idx, n):
    r = np.sqrt((ea[idx] ** 2).mean(1) / np.maximum((eb_[idx] ** 2).mean(1), 1e-30))
    lo, hi = np.percentile(r, [2.5, 97.5])
    lab = "own better" if hi < 1 else ("pooled better" if lo > 1 else "tie")
    return dict(ratio=rms(ea) / max(rms(eb_), 1e-30), ci95=[float(lo), float(hi)], label=lab,
                closer=f"{int(np.sum(np.abs(ea) < np.abs(eb_)))}/{n}")


def block(sub):
    n = len(sub)
    if n == 0:
        return dict(n=0)
    idx = np.random.default_rng(0).integers(0, n, (NB, n))
    falls = np.array([r["fall"] for r in sub])
    b = dict(n=n, fall_mean=float(falls.mean()), fall_sd=float(falls.std(ddof=1)) if n > 1 else float("nan"),
             fall20_mean=float(np.mean([r["fall20"] for r in sub])),
             fall20_sd=float(np.std([r["fall20"] for r in sub], ddof=1)) if n > 1 else float("nan"),
             noise_fall=float(NOISE_FALL))
    b["best_attainable_ratio_vs_mean"] = float(NOISE_FALL / b["fall_sd"]) if n > 1 else float("nan")
    own = np.array([r["S23"] - r["S25"] for r in sub])
    sp = [spearman(own[i], falls[i]) for i in idx[:2000]]
    b["spearman_ownS_vs_observed_fall"] = dict(rho=spearman(own, falls), ci95=[float(np.nanpercentile(sp, 2.5)), float(np.nanpercentile(sp, 97.5))],
                                               pearson=float(np.corrcoef(own, falls)[0, 1]) if n > 2 else float("nan"))
    b["fall_rms"], b["fall_bias"], b["e25_rms"], b["ratios"] = {}, {}, {}, {}
    b["fall_median_abs"] = {}   # added after the first run showed a few extreme corpora dominate code and English r.m.s.
    for p in PLACE:
        E = {a: np.array([r["fall_err"][p][a] for r in sub]) for a in ARMS}
        F = {a: np.array([r["e25"][p][a] for r in sub]) for a in ARMS}
        b["fall_rms"][p] = {a: rms(E[a]) for a in ARMS}
        b["fall_bias"][p] = {a: float(E[a].mean()) for a in ARMS}
        b["fall_median_abs"][p] = {a: float(np.median(np.abs(E[a]))) for a in ARMS}
        b["e25_rms"][p] = {a: rms(F[a]) for a in ARMS}
        for o in OWN:
            for q in POOLED + CONTEXT:
                b["ratios"][f"fall|{p}|{o}/{q}"] = ratio(E[o], E[q], idx, n)
                if p in ("primary", "oracle"):
                    b["ratios"][f"e25|{p}|{o}/{q}"] = ratio(F[o], F[q], idx, n)
    E20 = {a: np.array([r["fall20_err"][a] for r in sub]) for a in ("own_S", "pool_exp", "fw_info", "fw_meas")}
    b["fall20_rms"] = {a: rms(v) for a, v in E20.items()}
    b["fall20_ratios"] = {f"own_S/{q}": ratio(E20["own_S"], E20[q], idx, n) for q in ("pool_exp", "fw_info", "fw_meas")}
    return b


def blocks(sel):
    out = {f: block([r for r in sel if r["family"] == f]) for f in ("code", "english", "lang")}
    out["all"] = block(sel)
    return out


res = dict(design="docs/analysis/shape_vs_pooled_2026-10-30.md",
           constants=dict(pooled_exponent=A_POOL, lam=LAM, dev_fall_bpb_21_23=dev_fall, dev_mean_log_ratio_21_23=dev_log,
                          dev_fall_bpb_21_23_all13=float(np.mean([devL[c][21] - devL[c][23] for c in devL])),
                          fineweb_measured=FW, fineweb_info_shape=S_FW, fineweb_gamma=st_fw["gamma"],
                          noise_fall=float(NOISE_FALL)),
           counts={}, sets={}, pooled={})
for s in PRIMARY_SETS + ("stage3",):
    sel = [r for r in rows if r["set"] == s]
    res["counts"][s] = {f: sum(r["family"] == f for r in sel) for f in ("code", "english", "lang")}
    res["sets"][s] = blocks(sel)
res["pooled"]["three_sets"] = blocks([r for r in rows if r["set"] in PRIMARY_SETS])
res["pooled"]["forward18+external45"] = blocks([r for r in rows if r["set"] in ("forward18", "external45")])
res["pooled"]["all_four_sets"] = blocks(rows)


# verdicts on the primary endpoint
def verdict(b, own):
    lab = {q: b["ratios"][f"fall|primary|{own}/{q}"]["label"] for q in POOLED}
    if all(v == "own better" for v in lab.values()):
        s = "REAL positive"
    elif any(v == "pooled better" for v in lab.values()):
        s = "REAL negative"
    else:
        s = "tie"
    return dict(summary=s, per_arm=lab)


res["verdict"] = {"own_S, three sets": {f: verdict(res["pooled"]["three_sets"][f], "own_S") for f in ("code", "english", "lang", "all")},
                  "own_lamS, forward18+external45": {f: verdict(res["pooled"]["forward18+external45"][f], "own_lamS")
                                                     for f in ("code", "english", "lang", "all")}}
# Exploratory, added after seeing the results above: the own-S fall ranks language falls (Spearman > 0) but is ~3x too
# wide. Affine calibration fall = a + b [S(2^23) - S(2^25)] fitted per family on test80 (the same status as lambda),
# scored on forward18 + external45 against the pooled arms (primary level). Not part of the verdict.
expl = {}
for f in ("code", "english", "lang"):
    tr = [r for r in rows if r["set"] == "test80" and r["family"] == f]
    b1, b0 = np.polyfit([r["S23"] - r["S25"] for r in tr], [r["fall"] for r in tr], 1)
    te = [r for r in rows if r["set"] in ("forward18", "external45") and r["family"] == f]
    ea = np.array([b0 + b1 * (r["S23"] - r["S25"]) - r["fall"] for r in te])
    idx = np.random.default_rng(0).integers(0, len(te), (NB, len(te)))
    expl[f] = dict(a=float(b0), b=float(b1), n=len(te), rms=rms(ea), median_abs=float(np.median(np.abs(ea))),
                   ratios={q: ratio(ea, np.array([r["fall_err"]["primary"][q] for r in te]), idx, len(te)) for q in POOLED + CONTEXT})
res["exploratory_affine_test80_to_fwd_ext"] = expl
res["per_corpus"] = rows
json.dump(res, open(OUT, "w"), indent=1, default=float)

# ------------------------------------------------------------------ short printout
def show(name, b):
    if not b["n"]:
        return
    fr, er = b["fall_rms"]["primary"], b["e25_rms"]["primary"]
    print(f"{name:34s} n={b['n']:3d} fall {b['fall_mean']:.3f}+-{b['fall_sd']:.3f} rho {b['spearman_ownS_vs_observed_fall']['rho']:+.2f} | fall rms "
          + " ".join(f"{a} {fr[a]:.3f}" for a in ARMS) + f" | e25 ownS {er['own_S']:.3f} poolexp {er['pool_exp']:.3f} fwm {er['fw_meas']:.3f}")
    for q in POOLED:
        t = b["ratios"][f"fall|primary|own_S/{q}"]
        print(f"      own_S/{q:13s} {t['ratio']:.2f} ({t['ci95'][0]:.2f}-{t['ci95'][1]:.2f}) {t['closer']} {t['label']}")


print("dev fall", {k: round(v, 3) for k, v in dev_fall.items()}, "S_FW fall", round(S_FW[23] - S_FW[25], 3), "FW meas", round(FW[23] - FW[25], 3))
for s in res["sets"]:
    for f, b in res["sets"][s].items():
        show(f"{s}/{f}", b)
for s in res["pooled"]:
    for f, b in res["pooled"][s].items():
        show(f"POOLED {s}/{f}", b)
print(json.dumps(res["verdict"], indent=1))
print("->", os.path.relpath(OUT, ROOT))
