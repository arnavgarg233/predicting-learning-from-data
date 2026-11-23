"""Costmatch analysis (2026-11-05): one compute-matched comparison of forecast error against the compute spent before the
forecast, from stored outputs only (no training, CPU only).

    python3 scripts/costmatch/consolidate.py   -> results/costmatch/{rows,summary}.json

Sets (small learner, error at 2^25 training bytes, seed mean):
  test80  the 80 test corpora                      results/reanalysis/{compute_frontier,stronger_baselines,shape_vs_pooled}.json
  new18   18 corpora forecast before training      the same files
  ext45   third-party-list forward test (45)       results/extbench/score.json, shape_vs_pooled.json
  web30   web-language forward test (30)           results/fwd_weblang/score.json
  eng24   English forward test (24; eng20 = the 20 kept by the outlier rule fixed in advance)
                                                   results/fwd_english/score.json
open-sci-ref frozen test (0.13B, 0.4B; whole-curve per-corpus r.m.s. over the scored checkpoints):
                                                   results/opensciref_forecast/scores.json
Every FLOP figure is the stored one where a file holds it; otherwise it is computed with the same accounting
(core.learner_flops on the stored run's steps; core.ref_flops = 2 N x targets x sum(n+1) positions).
1,000-target versions of the forward-test forecasts are recomputed here from the stored per-target reference losses
(seeded subset, seed 20261025, as in the stronger-reference test); they are computed after the outcomes were known.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/curve_forecast"))
sys.path.insert(0, str(ROOT / "scripts/reanalysis"))
sys.path.insert(0, str(ROOT / "text/refcurve"))
import core as C  # noqa: E402
import refcurve_fineweb as RF  # noqa: E402
from ref_subset_cost import Curve  # noqa: E402

RES = ROOT / "results"
CF = RES / "curve_forecast"
RR = RES / "reanalysis"
OUT = RES / "costmatch"
NB = 20000
A_POOL = 0.14
LAM = 0.6924039665645343
NGRAM = (-0.09618305026700236, 0.8134058678431364)
CODE_BIAS = 0.2011360393894477
C_FLAG = 0.70
NS_SUM = sum(n + 1 for n in C.REF_NS)
# parameter counts: SmolLM2/Qwen 0.5B/1.5B/7B as in compute_frontier.py; 14B from the model cards (14.8B, 14.7B)
P14_Q3, P14_Q25 = 14.8e9, 14.7e9
SCREEN14 = 2 * (P14_Q3 + P14_Q25) * 512 * 200 * 8   # scripts/stronger_ref/screen14.py: 200 targets x 8 corpora, 512-token context
N_14B_CORPORA = 66                                   # score_half500.json n_h14 (screen amortised over these)

lk = json.load(open(CF / "locked.json"))
K = lk["constants"]
CH = lk["chosen"]
K05 = dict(K, level_coef=json.load(open(CF / "stronger_ref/score.json"))["arms"]["sub05"]["level_coef"])
SVP = json.load(open(RR / "shape_vs_pooled.json"))
DEV_FALL = SVP["constants"]["dev_fall_bpb_21_23"]
DEV_LOG = SVP["constants"]["dev_mean_log_ratio_21_23"]
FWM = {int(k): v for k, v in SVP["constants"]["fineweb_measured"].items()}
FWI = {int(k): v for k, v in SVP["constants"]["fineweb_info_shape"].items()}
FAM = {"code": "code", "english": "english", "wiki": "lang", "lang": "lang"}


# ------------------------------------------------------------------ helpers
def shared_falls(V, fam):
    """Falls 2^23 -> 2^25 of the shared shapes, exactly as shape_vs_pooled.py fc_fall."""
    return dict(SS_pool_exp=V * (1 - 2.0 ** (-2 * A_POOL)), SS_dev_fall=DEV_FALL[fam],
                SS_dev_log=V * (1 - np.exp(-DEV_LOG[fam])), SS_fw_info=FWI[23] - FWI[25], SS_fw_meas=FWM[23] - FWM[25])


def pl0(L1, k1, L2, k2, k):
    """Power law with no floor through two points (falls back to the larger pilot if not falling)."""
    if not L2 < L1:
        return float(L2)
    a = np.log2(L1 / L2) / (k2 - k1)
    return float(L2 * 2.0 ** (-a * (k - k2)))


def has_losses(rdir, model):
    """Per-target losses stored: the full file, or the released 1,000-target subset."""
    return (rdir / f"{model}_raw.npz").exists() or (rdir / f"{model}_sub1000.npz").exists()


def curves(rdir, model):
    d = json.load(open(rdir / "refcurve_fineweb.json"))[model]
    if (rdir / f"{model}_raw.npz").exists():
        raw = np.load(rdir / f"{model}_raw.npz")
        nt = len(raw["tgt"])
        idx = np.sort(np.random.default_rng(20261025).choice(nt, min(1000, nt), replace=False))
    else:  # released subset written by scripts/stronger_ref/extract_sub1000.py (same seeded 1,000 targets)
        raw = np.load(rdir / f"{model}_sub1000.npz")
        nt = int(raw["nt"]); idx = np.arange(len(raw["loss"]))
    r = RF.summarize(model, model, raw["loss"][idx] / np.log(2), raw["doc"][idx], raw["tgtbytes"][idx].astype(float),
                     raw["ctxb"][idx], list(raw["NS"]), d["n_docs_total"])
    return Curve(d["rows"]), Curve(r["rows"]), nt


def fc_from_curve(hr, ngram, coef):
    st = C.ref_stats(hr)
    st["ngram23"] = ngram
    nu, n0 = C.shape_params(CH["shape"], st, K)
    return dict(S23=C.S(hr, nu, n0, 23), S25=C.S(hr, nu, n0, 25),
                A0_23=C.A0(hr, st, nu, n0, CH["level"], CH["level_mode"], coef, 23),
                A0_25=C.A0(hr, st, nu, n0, CH["level"], CH["level_mode"], coef, 25), ref_full=st["ref_full"])


def full_and_1k(rdir, model, ngram):
    full, sub, nt = curves(rdir, model)
    return fc_from_curve(full, ngram, K["level_coef"]), fc_from_curve(sub, ngram, K05["level_coef"] if model != "SmolLM2-135M" else K["level_coef"]), nt


def ladder(d, c):
    out = {}
    for f in glob.glob(str(d / "text_runs" / f"cf_{c}" / "bytes_b*_s*.json")):
        r = json.load(open(f))
        out.setdefault(int(round(np.log2(r["train_bytes"]))), {})[r["seed"]] = r
    return out


def lflops(lad, k):
    return C.learner_flops(lad[k][0]["steps"], lad[k][0].get("eval_chunks", 0))


rows = []          # one per (set, corpus): family, measured, err{method}, flops{method}, target_flops, c_ratio
checks = {}


def add(set_, c, fam, y25, err, flops, tgt, cr, **kw):
    rows.append(dict(set=set_, corpus=c, family=fam, L25=y25, err=err, flops=flops, target_flops=tgt, c_ratio=cr, **kw))


# ------------------------------------------------------------------ test80 + new18
cfj = json.load(open(RR / "compute_frontier.json"))["per_corpus"]
sb = json.load(open(RR / "stronger_baselines.json"))["per_corpus"]
h14 = json.load(open(CF / "stronger_ref14b/score_half500.json"))["arms"]["h14"]["per_corpus"]
svp = {(r["set"], r["corpus"]): r for r in SVP["per_corpus"]}
FC80 = {**json.load(open(CF / "forecast.json"))["corpora"], **json.load(open(CF / "stage2/forecast.json"))["corpora"]}
NEWD = CF / "new_corpora"
FCN = json.load(open(NEWD / "forecast.json"))["corpora"]
s80 = json.load(open(CF / "score80.json"))["per_corpus"]
sn = json.load(open(NEWD / "score.json"))["per_corpus"]
ck_svp, ck_1k = [], []
for c, v in cfj.items():
    g = v["group"]
    fam = FAM[v["family"]]
    sset = "test80" if g == "test80" else "new18"
    r = svp[("test80" if g == "test80" else "forward18", c)]
    y25 = r["L25"]
    e, fl = dict(v["err"]), dict(v["flops"])
    tgt = fl.pop("target")
    fc = FC80[c] if g == "test80" else FCN[c]
    m = v["ref_model"]
    ref20 = fl["A0_20k"]
    ref1k = fl["A0_1k"]
    e["Z_frozen_20k"], fl["Z_frozen_20k"] = e.pop("A0_20k"), fl.pop("A0_20k")   # on these sets the frozen forecast is A0
    e["Z_frozen_1k"], fl["Z_frozen_1k"] = e.pop("A0_1k"), fl.pop("A0_1k")
    # shared shapes at the forecast's own level (A0 at 2^23, 20,000 targets as run)
    V = r["lev_A0"]
    ck_svp.append(abs(V - (FC80[c] if g == "test80" else FCN[c])["A0"]["23"]))
    for a, f in shared_falls(V, fam).items():
        e[a + "_20k"], fl[a + "_20k"] = V - f - y25, ref20
        if a[3:] in r["e25"]["A0"]:
            ck_svp.append(abs(e[a + "_20k"] - r["e25"]["A0"][a[3:]]))
    # 1,000-target level for the shared shapes (recomputed from stored per-target losses)
    rdir = (CF / c / "refcurve" if (CF / c / "refcurve").exists() else CF / "stage2" / c / "refcurve") if g == "test80" \
        else NEWD / "pod" / c / "refcurve"
    if has_losses(rdir, m):
        F_full, F_1k, _ = full_and_1k(rdir, m, fc["ref_stats"]["ngram23"])
        ck_1k.append(abs(F_full["A0_25"] - fc["A0"]["25"]))
        ck_1k.append(abs(F_1k["A0_25"] - y25 - e["Z_frozen_1k"]))
        for a, f in shared_falls(F_1k["A0_23"], fam).items():
            e[a + "_1k"], fl[a + "_1k"] = F_1k["A0_23"] - f - y25, ref1k
    if g == "test80":
        s = sb[c]
        e["A2_pilot_shape"], fl["A2_pilot_shape"] = s["err"]["A2"], s["flops"]["pilot_20"] + ref20
        e["A4_2pilot_shape"], fl["A4_2pilot_shape"] = s["err"]["A4"], s["flops"]["pilots_19_20"] + ref20
        e["X_pool23"], fl["X_pool23"] = s["err"]["X_pool23"], s["flops"]["run_23"]
        k14 = "test80:" + c
        e["Z_14B_500"] = h14[k14]["err_A0"]["25"]
        fl["Z_14B_500"] = 2 * P14_Q3 * 500 * NS_SUM + SCREEN14 / N_14B_CORPORA
    cr = fc["ref_stats"]["ref_full"] / fc["ref_stats"]["ngram23"]
    add(sset, c, fam, y25, e, fl, tgt, cr)

checks["shape_vs_pooled_level_and_e25_reproduced_max_abs"] = max(ck_svp)
checks["test80_new18_1k_recompute_max_abs (full reproduces stored A0; 1k reproduces compute_frontier A0_1k)"] = max(ck_1k) if ck_1k else None
checks["test80_new18_1k_level_available"] = int(sum("SS_pool_exp_1k" in r["err"] for r in rows))

# ------------------------------------------------------------------ ext45 (third-party-list forward test)
eb = json.load(open(RES / "extbench/score.json"))["per_corpus"]
FI = {}
for p in (1, 2, 3):
    FI.update(json.load(open(RES / f"extbench/pod{p}/forecast_i.json"))["corpora"])
ck = []
for c, v in eb.items():
    fam = FAM[v["family"]]
    y25 = v["L"]["25"]
    fl0 = v["flops"]
    r = svp[("external45", c)]
    m = v["ref_model"]
    fc = FI[c]
    e, fl = {}, {}
    e["Z_A0_20k"], fl["Z_A0_20k"] = v["err"]["i"]["e25"], fl0["i_run"]
    e["Z_frozen_20k"], fl["Z_frozen_20k"] = v["err"]["ii"]["e25"], fl0["i_run"]   # the revision needs the reference curve's shape
    e["A3"], fl["A3"] = v["err"]["A3"]["e25"], fl0["A3"]
    e["X_pool1"], fl["X_pool1"] = v["err"]["X_pool1"]["e25"], fl0["X_pool1"]
    e["A3_20_23"], fl["A3_20_23"] = v["err"]["P2023"]["e25"], fl0["P2023"]
    Vrev = r["lev_rev"]
    lev_cost = 0.0 if fam == "lang" else fl0["i_run"]   # language level is the n-gram route (CPU, not charged)
    for a, f in shared_falls(Vrev, fam).items():
        e[a + "_20k"], fl[a + "_20k"] = Vrev - f - y25, lev_cost
        if a[3:] in r["e25"]["rev"]:
            ck.append(abs(e[a + "_20k"] - r["e25"]["rev"][a[3:]]))
    ck.append(abs(Vrev - LAM * (r["S23"] - r["S25"]) - y25 - e["Z_frozen_20k"]))
    rdir = Path(ROOT / v["source_dir"]) / "pod" / c / "refcurve"
    F_full, F_1k, nt = full_and_1k(rdir, m, fc["ref_stats"]["ngram23"])
    ck.append(abs(F_full["S23"] - fc["S"]["23"]) + abs(F_full["S25"] - fc["S"]["25"]))
    if m == "SmolLM2-135M":
        ck.append(abs(F_full["A0_25"] - fc["A0"]["25"]))
    ref1k = C.ref_flops(m, 1000)
    lev1k = {"code": F_1k["A0_23"] - CODE_BIAS, "english": F_1k["A0_23"]}.get(fam, NGRAM[0] + NGRAM[1] * fc["ref_stats"]["ngram23"])
    e["Z_frozen_1k"], fl["Z_frozen_1k"] = lev1k - LAM * (F_1k["S23"] - F_1k["S25"]) - y25, ref1k
    e["Z_A0_1k"], fl["Z_A0_1k"] = F_1k["A0_25"] - y25, ref1k
    for a, f in shared_falls(lev1k, fam).items():
        e[a + "_1k"], fl[a + "_1k"] = lev1k - f - y25, (0.0 if fam == "lang" else ref1k)
    add("ext45", c, fam, y25, e, fl, fl0["target_25"], v["c_ratio"])
checks["ext45_reproduced_max_abs (shared shapes, revision, full-curve S and A0)"] = max(ck)

# ------------------------------------------------------------------ web30 (web-language forward test)
ws = json.load(open(RES / "fwd_weblang/score.json"))
cal = ws["calibration"]
ck = []
for c, v in ws["per_corpus"].items():
    d = ROOT / v["dir"]
    fc = json.load(open(d / "forecast_i.json"))["corpora"][c]
    lad = ladder(d, c)
    L = {int(k): x for k, x in v["L"].items()}
    y25 = L[25]
    V = v["level_23"]
    m = fc["ref_model"]
    ref20 = C.ref_flops(m, fc.get("ref_n_targets") or 20000)
    ref1k = C.ref_flops(m, 1000)
    e, fl = {}, {}
    e["Z_frozen_20k"], fl["Z_frozen_20k"] = v["e25_level"]["own_cal"], ref20
    e["Z_raw_shape_20k"], fl["Z_raw_shape_20k"] = v["e25_level"]["own_raw"], ref20
    e["Z_A0_20k"], fl["Z_A0_20k"] = fc["A0"]["25"] - y25, ref20
    e["SS_kind_mean"], fl["SS_kind_mean"] = v["e25_level"]["kind_mean"], 0.0
    for a, f in shared_falls(V, "lang").items():
        e[a + "_20k"], fl[a + "_20k"] = V - f - y25, 0.0
    ck.append(abs(e["SS_pool_exp_20k"] - v["e25_level"]["pool_exp"]))
    ck.append(abs(V - (NGRAM[0] + NGRAM[1] * fc["ref_stats"]["ngram23"])))
    a3 = pl0(L[19], 19, L[20], 20, 25)
    e["A3"], fl["A3"] = a3 - y25, lflops(lad, 19) + lflops(lad, 20)
    ck.append(abs(e["A3"] - v["A3_e25_own"]))
    e["X_pool1"], fl["X_pool1"] = L[20] * 2.0 ** (-A_POOL * 5) - y25, lflops(lad, 20)
    e["A3_20_23"], fl["A3_20_23"] = pl0(L[20], 20, L[23], 23, 25) - y25, lflops(lad, 20) + lflops(lad, 23)
    rd = d / "pod" / c / "refcurve"
    if not has_losses(rd, m):   # prepared on another pod; the full-curve check below confirms it is the same curve
        rd = Path(sorted(p for p in glob.glob(str(d.parent / "pod*" / "pod" / c / "refcurve" / f"{m}_*.npz"))
                         if p.endswith(("_raw.npz", "_sub1000.npz")))[0]).parent
    F_full, F_1k, nt = full_and_1k(rd, m, fc["ref_stats"]["ngram23"])
    ck.append(abs(F_full["S23"] - fc["S"]["23"]) + abs(F_full["S25"] - fc["S"]["25"]))
    e["Z_frozen_1k"], fl["Z_frozen_1k"] = V - (cal["a"] + cal["b"] * (F_1k["S23"] - F_1k["S25"])) - y25, ref1k
    e["Z_A0_1k"], fl["Z_A0_1k"] = F_1k["A0_25"] - y25, ref1k
    add("web30", c, "lang", y25, e, fl, lflops(lad, 25) if 25 in lad else float("nan"),
        fc["ref_stats"]["ref_full"] / fc["ref_stats"]["ngram23"])
checks["web30_reproduced_max_abs (pool_exp, n-gram level, A3, full-curve S)"] = max(ck)

# ------------------------------------------------------------------ eng24 (English forward test; no per-target losses stored)
es = json.load(open(RES / "fwd_english/score.json"))
ck = []
clean = set(es["primary_clean"]["corpora"])
for c, v in es["per_corpus"].items():
    d = ROOT / v["dir"]
    fc = json.load(open(d / "forecast_i.json"))["corpora"][c]
    lad = ladder(d, c)
    L = {int(k): x for k, x in v["L"].items()}
    y25 = L[25]
    V = v["level_23"]
    m = fc["ref_model"]
    ref20 = C.ref_flops(m, fc.get("ref_n_targets") or 20000)
    ck.append(abs(V - fc["A0"]["23"]))
    e, fl = {}, {}
    e["Z_frozen_20k"], fl["Z_frozen_20k"] = v["e25_level"]["own_cal"], ref20
    e["Z_raw_shape_20k"], fl["Z_raw_shape_20k"] = v["e25_level"]["own_raw"], ref20
    e["Z_A0_20k"], fl["Z_A0_20k"] = fc["A0"]["25"] - y25, ref20
    e["SS_kind_mean"], fl["SS_kind_mean"] = v["e25_level"]["kind_mean"], ref20
    for a, f in shared_falls(V, "english").items():
        e[a + "_20k"], fl[a + "_20k"] = V - f - y25, ref20
    e["A3"], fl["A3"] = pl0(L[19], 19, L[20], 20, 25) - y25, lflops(lad, 19) + lflops(lad, 20)
    ck.append(abs((V - (pl0(L[19], 19, L[20], 20, 23) - pl0(L[19], 19, L[20], 20, 25))) - y25 - v["e25_level"]["A3"]))
    e["X_pool1"], fl["X_pool1"] = L[20] * 2.0 ** (-A_POOL * 5) - y25, lflops(lad, 20)
    e["A3_20_23"], fl["A3_20_23"] = pl0(L[20], 20, L[23], 23, 25) - y25, lflops(lad, 20) + lflops(lad, 23)
    add("eng24", c, "english", y25, e, fl, lflops(lad, 25), fc["ref_stats"]["ref_full"] / fc["ref_stats"]["ngram23"],
        kept_by_rule=c in clean)
checks["eng24_reproduced_max_abs (A0 level, A3 fall)"] = max(ck)

# ------------------------------------------------------------------ constant baselines and the c-rule (all small-learner sets)
for r in rows:
    others = [q["L25"] for q in rows if q["set"] == r["set"] and q["family"] == r["family"] and q is not r]
    r["err"]["CONST_loo_mean"], r["flops"]["CONST_loo_mean"] = float(np.mean(others)) - r["L25"], 0.0
    # applicability rule (threshold fixed before the forward tests): forecast if c < 0.70, else two pilots
    for z in ("Z_frozen_1k", "Z_frozen_20k"):
        if z in r["err"]:
            use = r["c_ratio"] < C_FLAG
            r["err"]["RULE_c_" + z[2:]] = r["err"][z] if use else r["err"]["A3"]
            r["flops"]["RULE_c_" + z[2:]] = r["flops"][z] + (0.0 if use else r["flops"]["A3"])


# ------------------------------------------------------------------ statistics
def summarise(sub, rng_seed=0):
    n = len(sub)
    meths = sorted(set.intersection(*[set(r["err"]) for r in sub]))
    E = {m: np.array([r["err"][m] for r in sub]) for m in meths}
    idx = np.random.default_rng(rng_seed).integers(0, n, (NB, n))
    tgt = float(np.median([r["target_flops"] for r in sub]))
    out = {}
    for m in meths:
        b = np.sqrt((E[m][idx] ** 2).mean(1))
        F = np.array([r["flops"][m] for r in sub])
        o = dict(rmse=float(np.sqrt(np.mean(E[m] ** 2))), rmse_ci95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                 median_abs=float(np.median(np.abs(E[m]))), max_abs=float(np.max(np.abs(E[m]))), bias=float(E[m].mean()),
                 flops_median=float(np.median(F)), flops_mean=float(F.mean()),
                 flops_frac_target=float(np.median(F / np.array([r["target_flops"] for r in sub]))))
        if "A3" in E and m != "A3":
            rr = np.sqrt((E[m][idx] ** 2).mean(1) / (E["A3"][idx] ** 2).mean(1))
            o["ratio_vs_A3"] = [o["rmse"] / float(np.sqrt(np.mean(E["A3"] ** 2))), float(np.percentile(rr, 2.5)), float(np.percentile(rr, 97.5))]
            o["closer_than_A3"] = f"{int(np.sum(np.abs(E[m]) < np.abs(E['A3'])))}/{n}"
        out[m] = o
    # Pareto on mean FLOPs per corpus (what a user spends on average; equals the median except for the c-rule)
    front = [m for m in meths if not any(out[q]["flops_mean"] <= out[m]["flops_mean"] and out[q]["rmse"] < out[m]["rmse"]
                                         for q in meths if q != m)]
    front_max = [m for m in meths if not any(out[q]["flops_mean"] <= out[m]["flops_mean"] and out[q]["max_abs"] < out[m]["max_abs"]
                                             for q in meths if q != m)]
    return dict(n=n, target_flops_median=tgt, methods=out, pareto_rmse=sorted(front, key=lambda m: out[m]["flops_mean"]),
                pareto_max_abs=sorted(front_max, key=lambda m: out[m]["flops_mean"]))


blocks = {}
for s in ("test80", "new18", "ext45", "web30", "eng24"):
    sel = [r for r in rows if r["set"] == s]
    blocks[s] = {"all": summarise(sel)}
    for f in ("code", "english", "lang"):
        sub = [r for r in sel if r["family"] == f]
        if sub:
            blocks[s][f] = summarise(sub)
blocks["eng20"] = {"english": summarise([r for r in rows if r["set"] == "eng24" and r.get("kept_by_rule")])}
# pooled over every small-learner set (methods present in all of them)
pool = [r for r in rows]
blocks["pooled_5sets"] = {"all": summarise(pool)}
for f in ("code", "english", "lang"):
    blocks["pooled_5sets"][f] = summarise([r for r in pool if r["family"] == f])
blocks["forward_only(ext45+web30+eng24)"] = {f: summarise([r for r in rows if r["set"] in ("ext45", "web30", "eng24") and
                                                             (f == "all" or r["family"] == f)]) for f in ("all", "code", "english", "lang")}

# ------------------------------------------------------------------ open-sci-ref frozen test
osr = json.load(open(RES / "opensciref_forecast/scores.json"))
cmp_ = osr["compute"]
osr_out = {}
for size in ("0.13b", "0.4b"):
    v = osr["variants"]["sub1000"][size]
    pc = v["per_corpus"]["whole"]
    fs = v["final_signed"]
    cc = cmp_[size]
    ref = cmp_["reference"]["flops_per_corpus"]["c4"]
    fl = dict(F_ext=ref, F_ext_flat=ref, C=0.0, P2e=cc["P2e_flops_to_4k"], P2=cc["P2_flops_to_28k"])
    blk = {}
    for a in ("F_ext", "F_ext_flat", "C", "P2e", "P2"):
        e = np.array([x for x in pc[a].values() if x is not None and np.isfinite(x)], float)
        f_ = np.array([x for x in fs[a].values() if x is not None and np.isfinite(x)], float) if a in fs else None
        blk[a] = dict(n=len(e), whole_rmse=float(np.sqrt(np.mean(e ** 2))), whole_median=float(np.median(e)), whole_max=float(e.max()),
                      final_rmse=float(np.sqrt(np.mean(f_ ** 2))) if f_ is not None else None,
                      final_median_abs=float(np.median(np.abs(f_))) if f_ is not None else None,
                      final_max_abs=float(np.max(np.abs(f_))) if f_ is not None else None,
                      flops=fl[a], flops_frac_full_run=fl[a] / cc["full_run_flops"])
        assert abs(blk[a]["whole_rmse"] - v["arms"][f"{a}|whole"]["rms"]) < 1e-12
    blk["paired"] = {k: v["paired"][k] for k in v["paired"] if k.endswith("|whole")}
    blk["note"] = "C uses the other six corpora's full runs (2.3e20 / 7.3e20 FLOPs each); its marginal cost is 0."
    osr_out[size] = blk

OUT.mkdir(parents=True, exist_ok=True)
summary = dict(script="scripts/costmatch/consolidate.py", checks=checks, constants=dict(
    ref_flops_20k=dict(SmolLM2=C.ref_flops("SmolLM2-135M", 20000), Qwen05=C.ref_flops("Qwen2.5-0.5B", 20000)),
    ref_flops_1k=dict(SmolLM2=C.ref_flops("SmolLM2-135M", 1000), Qwen05=C.ref_flops("Qwen2.5-0.5B", 1000)),
    screen14_total=SCREEN14, screen14_per_corpus=SCREEN14 / N_14B_CORPORA, c_flag=C_FLAG),
    blocks=blocks, osr=osr_out)
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
json.dump(rows, open(OUT / "rows.json", "w"), indent=1)
print(json.dumps(checks, indent=1))
