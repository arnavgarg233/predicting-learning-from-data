"""Diagnose the code and English negatives of the within-kind fall calibration (fall = a + b x own fall).
Reads stored JSON/NPZ
only; no training. Deterministic.

    python scripts/reanalysis/fall_diagnosis.py     # -> results/reanalysis/fall_diagnosis.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "curve_forecast"))
import forecast_input_variants as V  # noqa: E402

RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "reanalysis", "fall_diagnosis.json")
NB = 20000
NOISE_FALL = 0.0126 * np.sqrt(1.5)
K_TS = json.load(open(os.path.join(RES, "curve_forecast", "locked.json")))["constants"]["K_ts"]
B_WEB = 0.35564                                   # frozen web-language slope (results/fwd_weblang/fall_line.json)
LN2 = np.log(2.0)


# ------------------------------------------------------------------ helpers
class Curve:
    """Piecewise-linear h(log n), as forecast_neff.HRef, from (context bytes, bits per byte) pairs."""

    def __init__(self, ctx, bpb):
        o = np.argsort(ctx)
        self.x = np.log(np.asarray(ctx, float)[o])
        self.y = np.asarray(bpb, float)[o]

    def h(self, n):
        return np.interp(np.log(n), self.x, self.y)


def own_S(cv, ks=(20, 23, 25)):
    g, A = V.power_fit(cv)
    nu = 1.0 / (1.0 + g)
    n0 = (A * g / K_TS) ** nu
    return {k: float(cv.h(n0 * (2.0 ** k) ** nu)) for k in ks}


def rms(e):
    return float(np.sqrt(np.mean(np.square(e))))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_ratio(ea, eb, nb=NB, rng=None):
    n = len(ea)
    rng = np.random.default_rng(0) if rng is None else rng
    idx = rng.integers(0, n, (nb, n))
    br = np.sqrt((ea[idx] ** 2).mean(1) / (eb[idx] ** 2).mean(1))
    lo, hi = np.percentile(br, [2.5, 97.5])
    return float(lo), float(hi)


def loo_line(x, y):
    """leave-one-out predictions: OLS line and mean of the other corpora."""
    n = len(x); own = np.empty(n); mean = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        b, a = np.polyfit(x[m], y[m], 1)
        own[i] = a + b * x[i]; mean[i] = y[m].mean()
    return own, mean


def compare(names, x, y, nb=NB):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    own, mean = loo_line(x, y)
    ea, eb = own - y, mean - y
    lo, hi = boot_ratio(ea, eb, nb)
    b, a = np.polyfit(x, y, 1)
    sd = float(y.std(ddof=1))
    return dict(n=n, fall_mean=float(y.mean()), fall_sd=sd, best_attainable_ratio=float(NOISE_FALL / sd),
                x_sd=float(x.std(ddof=1)), full_fit=dict(a=float(a), b=float(b)), spearman=spearman(x, y),
                pearson=float(np.corrcoef(x, y)[0, 1]), rms_own=rms(ea), rms_mean=rms(eb), ratio=rms(ea) / rms(eb),
                ci95=[lo, hi], label="positive" if hi < 1 else ("negative" if lo > 1 else "tie"),
                closer=f"{int(np.sum(np.abs(ea) < np.abs(eb)))}/{n}", corpora=list(names))


def r1_keep(y, k=4.0):
    y = np.asarray(y, float); med = np.median(y); mad = 1.4826 * np.median(np.abs(y - med))
    return np.abs(y - med) <= k * mad


def med_ci(v, nb=2000, seed=0):
    v = np.asarray(v, float); idx = np.random.default_rng(seed).integers(0, len(v), (nb, len(v)))
    m = np.median(v[idx], 1)
    return dict(median=float(np.median(v)), ci95=[float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))], n=len(v))


def med_diff_ci(a, b, nb=2000, seed=0):
    a = np.asarray(a, float); b = np.asarray(b, float); rng = np.random.default_rng(seed)
    d = np.median(a[rng.integers(0, len(a), (nb, len(a)))], 1) - np.median(b[rng.integers(0, len(b), (nb, len(b)))], 1)
    return dict(diff=float(np.median(a) - np.median(b)), ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])


# ------------------------------------------------------------------ rows
svp = json.load(open(os.path.join(RES, "reanalysis", "shape_vs_pooled.json")))["per_corpus"]
nlt = {r["corpus"]: r for r in json.load(open(os.path.join(RES, "reanalysis", "nl_table.json")))}
FI = {}
for p in (1, 2, 3):
    FI.update(json.load(open(os.path.join(RES, "extbench", f"pod{p}", "forecast_i.json")))["corpora"])
rows = []
for r in svp:
    kind = ("web" if r["set"] == "external45" else "wiki") if r["family"] == "lang" else r["family"]
    if r["set"] == "external45":
        st = FI[r["corpus"]]["ref_stats"]; rf, ng = st["ref_full"], st["ngram23"]
    else:
        rf, ng = nlt[r["corpus"]]["r"], nlt[r["corpus"]]["g"]
    rows.append(dict(corpus=r["corpus"], set=r["set"], kind=kind, L20=r["L20"], L23=r["L23"], L25=r["L25"],
                     S20=r["S20"], S23=r["S23"], S25=r["S25"], ref_full=rf, ngram23=ng))
FW = json.load(open(os.path.join(RES, "fwd_weblang", "score.json")))["per_corpus"]
FFI = {}
for f in sorted(glob.glob(os.path.join(RES, "fwd_weblang", "pod*", "forecast_i.json"))):
    for c, v in json.load(open(f))["corpora"].items():
        FFI.setdefault(c, v)                       # first dated forecast wins (pods in numeric order)
for c, v in FW.items():
    fi = FFI[c]; S = fi["S"]
    assert abs((S["23"] - S["25"]) - v["own_fall"]) < 1e-9, c
    L25 = float(np.mean(list(v["L25_seeds"].values()))) if v.get("L25_seeds") else v["L"]["25"]
    rows.append(dict(corpus=c, set="forward30", kind="web", L20=v["L"]["20"], L23=v["L"]["23"], L25=L25,
                     S20=S["20"], S23=S["23"], S25=S["25"], ref_full=fi["ref_stats"]["ref_full"],
                     ngram23=fi["ref_stats"]["ngram23"]))
    assert abs(rows[-1]["L23"] - L25 - v["fall"]) < 1e-9, c
for r in rows:
    r["fall"] = r["L23"] - r["L25"]; r["fall1"] = r["L20"] - r["L23"]
    r["x"] = r["S23"] - r["S25"]; r["x1"] = r["S20"] - r["S23"]
    r["r2_flag"] = bool(r["L25"] > r["ngram23"] or r["fall"] < 0 or r["fall1"] < 0)
KINDS = ("wiki", "web", "english", "code")
byk = {k: sorted([r for r in rows if r["kind"] == k], key=lambda r: (r["set"], r["corpus"])) for k in KINDS}
web29 = [r for r in byk["web"] if r["set"] == "external45"]
res = dict(design="docs/analysis/fall_diagnosis.md", noise_fall=float(NOISE_FALL),
           counts={k: {s: sum(r["set"] == s for r in v) for s in sorted({r["set"] for r in v})} for k, v in byk.items()})


def cmp_rows(sub, xk, yk, clean=False):
    if clean:
        keep = r1_keep([r[yk] for r in sub]); dropped = [r["corpus"] for r, k in zip(sub, keep) if not k]
        sub = [r for r, k in zip(sub, keep) if k]
    out = compare([r["corpus"] for r in sub], [r[xk] for r in sub], [r[yk] for r in sub])
    if clean:
        out["dropped_R1"] = dropped
    return out


# ------------------------------------------------------------------ Q1 capacity floor on code
q1 = dict(floors={}, compression={}, window={})
for k in KINDS:
    s = byk[k]
    L23 = np.array([r["L23"] for r in s]); L25 = np.array([r["L25"] for r in s])
    rf = np.array([r["ref_full"] for r in s]); ng = np.array([r["ngram23"] for r in s]); S25 = np.array([r["S25"] for r in s])
    q1["floors"][k] = dict(n=len(s), L23_over_ref=med_ci(L23 / rf), L25_over_ref=med_ci(L25 / rf),
                           L25_over_ngram23=med_ci(L25 / ng), gap_L25_minus_ref=med_ci(L25 - rf),
                           ref_over_ngram23_c=med_ci(rf / ng),
                           share_L25_below_ref=float(np.mean(L25 < rf)), share_L25_above_ngram23=float(np.mean(L25 > ng)),
                           share_L25_below_S25=float(np.mean(L25 < S25)),
                           above_ngram23=[r["corpus"] for r in s if r["L25"] > r["ngram23"]])
comp = {}
for k in KINDS:
    s = [r for r in byk[k] if r["fall"] > 0 and r["fall1"] > 0]
    a1 = np.array([np.log2(r["L20"] / r["L23"]) / 3 for r in s]); a2 = np.array([np.log2(r["L23"] / r["L25"]) / 2 for r in s])
    s1 = np.array([np.log2(r["S20"] / r["S23"]) / 3 for r in s]); s2 = np.array([np.log2(r["S23"] / r["S25"]) / 2 for r in s])
    f = np.array([r["fall"] / r["fall1"] for r in s]); fS = np.array([r["x"] / r["x1"] for r in s])
    comp[k] = dict(a1=a1, a2=a2, k=a2 / a1, kS=s2 / s1, t=(a2 / a1) / (s2 / s1), f=f, fS=fS, fr=f / fS)
    q1["compression"][k] = dict(n=len(s), excluded_nonpositive=[r["corpus"] for r in byk[k] if not (r["fall"] > 0 and r["fall1"] > 0)],
                                **{nm: med_ci(v) for nm, v in comp[k].items()})
q1["compression"]["code_minus_lang"] = {}
lang = {nm: np.concatenate([comp["wiki"][nm], comp["web"][nm]]) for nm in comp["code"]}
q1["compression"]["lang_pooled"] = {nm: med_ci(v) for nm, v in lang.items()}
for nm in ("k", "t", "fr", "a2"):
    q1["compression"]["code_minus_lang"][nm] = med_diff_ci(comp["code"][nm], lang[nm])
    q1["compression"]["english_minus_lang"] = q1["compression"].get("english_minus_lang", {})
    q1["compression"]["english_minus_lang"][nm] = med_diff_ci(comp["english"][nm], lang[nm])
for k in KINDS:
    q1["window"][k] = dict(fall_20_23=cmp_rows(byk[k], "x1", "fall1"), fall_20_23_clean=cmp_rows(byk[k], "x1", "fall1", True),
                           fall_23_25=cmp_rows(byk[k], "x", "fall"), fall_23_25_clean=cmp_rows(byk[k], "x", "fall", True))
q1["window"]["web29"] = dict(fall_20_23=cmp_rows(web29, "x1", "fall1"), fall_23_25=cmp_rows(web29, "x", "fall"))
# R2 sensitivity on code
c2 = [r for r in byk["code"] if not r["r2_flag"]]
q1["window"]["code_R2clean"] = dict(dropped=[r["corpus"] for r in byk["code"] if r["r2_flag"]],
                                    fall_20_23=cmp_rows(c2, "x1", "fall1"), fall_23_25=cmp_rows(c2, "x", "fall"))
# cost of the fair recovery test (25M learner, curve-forecast recipe) from logged throughput
STEP_S_25M = 0.2465 / 2                            # s per step per pod, two processes on one RTX 3090 (25M smoke)
steps = [nlt[c]["steps"] for c in nlt if nlt[c]["family"] == "code"]
st23_25 = float(np.median([s["23"] + s["25"] for s in steps])); st20 = float(np.median([s["20"] for s in steps]))
gpu_h = (st23_25 + st20) * STEP_S_25M / 3600 + 0.05            # + prep/test ~3 min
q1["recovery_cost"] = dict(median_steps_3p3M_2_23_plus_2_25=st23_25, median_steps_2_20=st20, s_per_step_25M_per_pod=STEP_S_25M,
                           gpu_h_per_corpus=gpu_h, usd_per_h_3090=0.226, usd_per_corpus=gpu_h * 0.226,
                           usd_30_code=30 * gpu_h * 0.226 * 1.15, usd_15_web_control=15 * gpu_h * 0.226 * 1.15 * 1.5,
                           note="steps from the 3.3M early-stopped runs; the 25M learner may stop earlier or later; "
                                "1.15 overhead for failed pods/idle; web languages take ~1.5x code's steps")
res["Q1"] = q1


# ------------------------------------------------------------------ Q2 reference reliability and competence on code
def find_raw(c, target_x, model):
    best = None
    for p in sorted(glob.glob(os.path.join(RES, "**", c, "refcurve", f"{model}_raw.npz"), recursive=True)):
        js = json.load(open(p.replace("_raw.npz", ".json")))
        cv = Curve([q["context_bytes_exact_mean"] for q in js["rows"]], [q["bits_per_byte"] for q in js["rows"]])
        S = own_S(cv); d = abs(S[23] - S[25] - target_x)
        if best is None or d < best[0]:
            best = (d, p)
    return best


q2 = dict(per_corpus={}, reliability={})
MODEL = dict(code="SmolLM2-135M", english="SmolLM2-135M", wiki="Qwen2.5-0.5B", web="Qwen2.5-0.5B")
for kd in ("code", "english", "wiki", "web"):            # code first: its numbers do not depend on the other kinds
    rng = np.random.default_rng(0); sds = []; xs_k = []
    for r in byk[kd]:
        fr = find_raw(r["corpus"], r["x"], MODEL[kd])
        if fr is None or fr[0] > 1e-6:
            continue
        d, p = fr
        z = np.load(p); L = z["loss"]; tb = z["tgtbytes"].astype(float); cb = z["ctxb"].astype(float); doc = z["doc"]
        full = own_S(Curve(cb.mean(0), L.sum(0) / tb.sum() / LN2))
        ud = np.unique(doc); groups = [np.flatnonzero(doc == u) for u in ud]
        xs = []
        for _ in range(200):
            pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
            S = own_S(Curve(cb[pick].mean(0), L[pick].sum(0) / tb[pick].sum() / LN2))
            xs.append(S[23] - S[25])
        sds.append(float(np.std(xs, ddof=1))); xs_k.append(r["x"]); r["boot_sd_x"] = sds[-1]
        q2["per_corpus"][r["corpus"]] = dict(kind=kd, raw=os.path.relpath(p, ROOT), stored_x=r["x"],
                                             recomputed_x=full[23] - full[25], match_abs=d, boot_sd_x=sds[-1],
                                             n_docs=int(len(ud)), n_targets=int(len(doc)))
    if not sds:  # the 20,000-target per-target loss files (*_raw.npz, about 1 GB) are not shipped; Q2 is then skipped
        q2["reliability"][kd] = dict(n_with_raw=0, n_kind=len(byk[kd]), note="per-target loss files not present")
        continue
    xc = np.array(xs_k)
    q2["reliability"][kd] = dict(n_with_raw=len(sds), n_kind=len(byk[kd]), median_boot_sd_x=float(np.median(sds)),
                                 max_boot_sd_x=float(np.max(sds)), n_sd_above_0p05=int(np.sum(np.array(sds) > 0.05)),
                                 between_corpus_sd_x=float(xc.std(ddof=1)),
                                 share_var_sampling=float(np.mean(np.square(sds)) / xc.var(ddof=1)))
# implied attenuation of the line from sampling error in x
q2["competence"] = {k: dict(c_median=q1["floors"][k]["ref_over_ngram23_c"], share_L25_below_ref=q1["floors"][k]["share_L25_below_ref"],
                            share_L25_below_S25=q1["floors"][k]["share_L25_below_S25"]) for k in KINDS}
dev = json.load(open(os.path.join(RES, "curve_forecast", "development.json")))["corpora"]
two = {}
for c, d in (("code", "mc/code"), ("java_code", "mc2/java_code")):
    two[c] = dict(measured_fall_21_23=dev[c]["measured"]["21"] - dev[c]["measured"]["23"])
    for m in ("SmolLM2-135M", "Qwen2.5-0.5B"):
        js = json.load(open(os.path.join(RES, d, "refcurve", f"{m}.json")))
        cv = Curve([q["context_bytes_exact_mean"] for q in js["rows"]], [q["bits_per_byte"] for q in js["rows"]])
        g, A = V.power_fit(cv); S = own_S(cv, (21, 23, 25))
        two[c][m] = dict(ref_full=float(cv.y[-1]), gamma=g, A=A, own_fall_21_23=S[21] - S[23], own_fall_23_25=S[23] - S[25])
q2["two_reference_dev_code"] = two
res["Q2"] = q2


# ------------------------------------------------------------------ Q3 English power (+ single-pass sets)
def power(x, sd_y, b, resid_sd, nsim=400, nb=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float); hits = 0; rat = []
    for _ in range(nsim):
        y = 0.2 + b * (x - x.mean()) + rng.normal(0, resid_sd, len(x))
        own, mean = loo_line(x, y); ea, eb = own - y, mean - y
        idx = rng.integers(0, len(x), (nb, len(x)))
        br = np.sqrt((ea[idx] ** 2).mean(1) / (eb[idx] ** 2).mean(1))
        hits += np.percentile(br, 97.5) < 1; rat.append(rms(ea) / rms(eb))
    return dict(power=hits / nsim, median_ratio=float(np.median(rat)), slope=float(b), resid_sd=float(resid_sd))


def power_block(sub):
    keep = r1_keep([r["fall"] for r in sub]); cl = [r for r, k in zip(sub, keep) if k]
    x = np.array([r["x"] for r in cl]); y = np.array([r["fall"] for r in cl]); sy = y.std(ddof=1); sx = x.std(ddof=1)
    b07 = np.sqrt(1 - 0.49) * sy / sx
    expl = B_WEB * sx
    out = dict(n_clean=len(cl), dropped_R1=[r["corpus"] for r, k in zip(sub, keep) if not k], fall_sd_clean=float(sy),
               x_sd_clean=float(sx), web_line_explained_sd=float(expl),
               web_line_expected_ratio=float(np.sqrt(max(1 - expl ** 2 / sy ** 2, 0.0))),
               ratio07=power(x, sy, b07, 0.7 * sy),
               web_line=power(x, sy, B_WEB, np.sqrt(max(sy ** 2 - expl ** 2, NOISE_FALL ** 2))))
    return out


def sp_rows(stats_path, runs_dir, fam):
    st = json.load(open(stats_path))[fam]["selected"]; out = []
    for e in st:
        s = e["stats"]; ctx, bpb = zip(*s["ref_curve"]); cv = Curve(ctx, bpb)
        g, A = V.power_fit(cv); S = own_S(cv, (23, 25))
        L = {}
        for k in (23, 25):
            p = os.path.join(runs_dir, f"rs_{e['name']}", f"bytes_b{k}_s0.json")
            if os.path.exists(p):
                L[k] = json.load(open(p))["bits_per_byte"]
        if len(L) == 2:
            out.append(dict(corpus=e["name"], fall=L[23] - L[25], x=S[23] - S[25], L23=L[23], L25=L[25],
                            gamma_refit=g, gamma_stored=s["gamma_ref"], ref_full=s["ref_full"], ngram23=s["ngram23"]))
    return out


SP = os.path.join(RES, "rate_singlepass")
SP25 = os.path.join(RES, "rate_singlepass25m")
sp = dict(english_3p3M=sp_rows(os.path.join(SP, "stats.json"), os.path.join(SP, "runs"), "english"),
          language_3p3M=sp_rows(os.path.join(SP, "stats.json"), os.path.join(SP, "runs"), "language"),
          language_25M=sp_rows(os.path.join(SP, "stats.json"), os.path.join(SP25, "runs"), "language"))
q3 = dict(english_corpora=[dict(corpus=r["corpus"], set=r["set"], L23=r["L23"], L25=r["L25"], fall=r["fall"], x=r["x"])
                           for r in byk["english"]],
          development=dict(fineweb_edu=dict(L23=dev["fineweb_edu"]["measured"]["23"], L25=dev["fineweb_edu"]["measured"]["25"],
                                            fall=dev["fineweb_edu"]["measured"]["23"] - dev["fineweb_edu"]["measured"]["25"],
                                            note="no own fall in the same pipeline; listed only")),
          power_english=power_block(byk["english"]), power_code=power_block(byk["code"]),
          power_web59=power_block(byk["web"]),
          loo_english_clean=cmp_rows(byk["english"], "x", "fall", True),
          single_pass={})
for nm, s in sp.items():
    q3["single_pass"][nm] = dict(loo=cmp_rows(s, "x", "fall"), loo_clean=cmp_rows(s, "x", "fall", True),
                                 power=power_block(s),
                                 gamma_refit_max_abs_diff=float(max(abs(r["gamma_refit"] - r["gamma_stored"]) for r in s)),
                                 rows=s)
res["Q3"] = q3


# ------------------------------------------------------------------ Q4 all languages, kind intercepts + shared slope
def q4_block(sub):
    n = len(sub); x = np.array([r["x"] for r in sub]); y = np.array([r["fall"] for r in sub])
    kw = np.array([r["kind"] == "wiki" for r in sub], float)
    X = np.column_stack([kw, 1 - kw, x])
    shared = np.empty(n); mean = np.empty(n); sep = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        beta = np.linalg.lstsq(X[m], y[m], rcond=None)[0]; shared[i] = X[i] @ beta
        same = m & (kw == kw[i]); mean[i] = y[same].mean()
        b, a = np.polyfit(x[same], y[same], 1); sep[i] = a + b * x[i]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    out = dict(n=n, fit=dict(a_wiki=float(beta[0]), a_web=float(beta[1]), b_shared=float(beta[2])))
    for nm, sel in (("all", np.ones(n, bool)), ("wiki", kw == 1), ("web", kw == 0)):
        ea, eb, es = shared[sel] - y[sel], mean[sel] - y[sel], sep[sel] - y[sel]
        lo, hi = boot_ratio(ea, eb); lo2, hi2 = boot_ratio(ea, es)
        out[nm] = dict(n=int(sel.sum()), rms_shared=rms(ea), rms_kind_mean=rms(eb), rms_separate_lines=rms(es),
                       ratio_shared_vs_mean=rms(ea) / rms(eb), ci95=[lo, hi],
                       label="positive" if hi < 1 else ("negative" if lo > 1 else "tie"),
                       closer=f"{int(np.sum(np.abs(ea) < np.abs(eb)))}/{int(sel.sum())}",
                       ratio_shared_vs_separate=rms(ea) / rms(es), ci95_vs_separate=[lo2, hi2])
    return out


langs = byk["wiki"] + byk["web"]
keep = np.concatenate([r1_keep([r["fall"] for r in byk["wiki"]]), r1_keep([r["fall"] for r in byk["web"]])])
res["Q4"] = dict(full=q4_block(langs), clean=q4_block([r for r, k in zip(langs, keep) if k]),
                 dropped_R1=[r["corpus"] for r, k in zip(langs, keep) if not k])
# Exploratory, added after the first run showed large sampling s.d. of the own fall on a few code corpora: repeat the
# within-kind leave-one-out (2^23 -> 2^25) on the R1-clean set restricted to corpora whose own fall is measured to
# better than 0.03 bits per byte (document bootstrap s.d.). Not part of any verdict rule.
expl = {}
for kd in KINDS:
    keep = r1_keep([r["fall"] for r in byk[kd]])
    cl = [r for r, k in zip(byk[kd], keep) if k and r.get("boot_sd_x") is not None]
    rel = [r for r in cl if r["boot_sd_x"] < 0.03]
    if len(rel) >= 8:
        expl[kd] = dict(n_clean_with_raw=len(cl), reliable=cmp_rows(rel, "x", "fall"),
                        dropped_unreliable=[r["corpus"] for r in cl if r["boot_sd_x"] >= 0.03])
res["exploratory_reliable_x"] = expl
# Exploratory, added after the first run: (i) the R1-clean English and code sets with own-fall outliers (4 scaled MADs in
# x) also set aside; (ii) power of a doubled English set (x resampled from the clean English x) at true ratios 0.7 and 0.8.
expl2 = {}
for kd in ("english", "code"):
    keep = r1_keep([r["fall"] for r in byk[kd]]); cl = [r for r, k in zip(byk[kd], keep) if k]
    kx = r1_keep([r["x"] for r in cl])
    expl2[kd] = dict(dropped_x=[r["corpus"] for r, k in zip(cl, kx) if not k],
                     loo=cmp_rows([r for r, k in zip(cl, kx) if k], "x", "fall"))
keep = r1_keep([r["fall"] for r in byk["english"]])
xe = np.array([r["x"] for r, k in zip(byk["english"], keep) if k]); ye = np.array([r["fall"] for r, k in zip(byk["english"], keep) if k])
xe2 = np.random.default_rng(0).choice(xe, 2 * len(xe), replace=True)
expl2["english_power_n%d" % len(xe2)] = {f"ratio{t}": power(xe2, ye.std(ddof=1), np.sqrt(1 - t * t) * ye.std(ddof=1) / xe.std(ddof=1),
                                                             t * ye.std(ddof=1)) for t in (0.7, 0.8)}
res["exploratory_outliers_power"] = expl2
res["rows"] = rows
json.dump(res, open(OUT, "w"), indent=1, default=float)


# ------------------------------------------------------------------ printout
def pc(name, c):
    print(f"  {name:22s} n={c['n']:3d} sd {c['fall_sd']:.3f} b {c['full_fit']['b']:+.3f} rho {c['spearman']:+.2f} own {c['rms_own']:.3f} "
          f"mean {c['rms_mean']:.3f} ratio {c['ratio']:.2f} ({c['ci95'][0]:.2f}-{c['ci95'][1]:.2f}) {c['closer']} {c['label']}"
          + (f" drop {c['dropped_R1']}" if c.get("dropped_R1") else ""))


print("counts", res["counts"])
print("Q1 floors")
for k in KINDS:
    f = q1["floors"][k]
    print(f"  {k:8s} L23/ref {f['L23_over_ref']['median']:.2f} L25/ref {f['L25_over_ref']['median']:.2f} L25/ng23 {f['L25_over_ngram23']['median']:.2f} "
          f"c {f['ref_over_ngram23_c']['median']:.2f} below ref {f['share_L25_below_ref']:.2f} below S25 {f['share_L25_below_S25']:.2f} above ng {f['above_ngram23']}")
print("Q1 compression (median [95%])")
for k in list(KINDS) + ["lang_pooled"]:
    c = q1["compression"][k]
    print(f"  {k:12s} " + " ".join(f"{nm} {c[nm]['median']:.2f} [{c[nm]['ci95'][0]:.2f},{c[nm]['ci95'][1]:.2f}]" for nm in ("a1", "a2", "k", "kS", "t", "f", "fS", "fr")))
for nm, v in q1["compression"]["code_minus_lang"].items():
    print(f"  code-lang {nm}: {v['diff']:+.2f} ({v['ci95'][0]:+.2f},{v['ci95'][1]:+.2f})   english-lang {q1['compression']['english_minus_lang'][nm]['diff']:+.2f} "
          f"({q1['compression']['english_minus_lang'][nm]['ci95'][0]:+.2f},{q1['compression']['english_minus_lang'][nm]['ci95'][1]:+.2f})")
print("Q1 windows")
for k, w in q1["window"].items():
    for nm, c in w.items():
        if isinstance(c, dict):
            pc(f"{k} {nm}", c)
print("Q1 cost", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in q1["recovery_cost"].items() if k != "note"})
for kd, v in q2["reliability"].items():
    print("Q2 reliability", kd, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in v.items()})
print("Q2 two-ref dev code", json.dumps(two, indent=0, default=lambda v: round(v, 3)))
print("Q3")
for nm in ("power_english", "power_code", "power_web59"):
    p = q3[nm]
    print(f"  {nm}: n {p['n_clean']} sd {p['fall_sd_clean']:.3f} xsd {p['x_sd_clean']:.3f} web-line expl {p['web_line_explained_sd']:.3f} "
          f"exp ratio {p['web_line_expected_ratio']:.2f} power@0.7 {p['ratio07']['power']:.2f} (med {p['ratio07']['median_ratio']:.2f}) "
          f"power web-line {p['web_line']['power']:.2f} (med {p['web_line']['median_ratio']:.2f}) drop {p['dropped_R1']}")
pc("english clean LOO", q3["loo_english_clean"])
for nm, s in q3["single_pass"].items():
    pc(f"SP {nm}", s["loo"]); pc(f"SP {nm} clean", s["loo_clean"])
    p = s["power"]
    print(f"     power@0.7 {p['ratio07']['power']:.2f} web-line exp ratio {p['web_line_expected_ratio']:.2f} power {p['web_line']['power']:.2f}; gamma refit diff {s['gamma_refit_max_abs_diff']:.2e}")
print("Q4")
for nm in ("full", "clean"):
    b = res["Q4"][nm]
    print(f"  {nm} n={b['n']} fit {b['fit']}")
    for k in ("all", "wiki", "web"):
        c = b[k]
        print(f"    {k:5s} n={c['n']:3d} shared {c['rms_shared']:.4f} mean {c['rms_kind_mean']:.4f} sep {c['rms_separate_lines']:.4f} ratio {c['ratio_shared_vs_mean']:.2f} "
              f"({c['ci95'][0]:.2f}-{c['ci95'][1]:.2f}) {c['closer']} {c['label']} | vs sep {c['ratio_shared_vs_separate']:.2f} ({c['ci95_vs_separate'][0]:.2f}-{c['ci95_vs_separate'][1]:.2f})")
print("  dropped", res["Q4"]["dropped_R1"])
print("Exploratory reliable-x LOO")
for kd, v in expl.items():
    pc(f"{kd} reliable", v["reliable"]); print("     dropped unreliable", v["dropped_unreliable"])
for kd in ("english", "code"):
    pc(f"{kd} R1 + x-outliers out", expl2[kd]["loo"]); print("     dropped x", expl2[kd]["dropped_x"])
print("English doubled-n power", expl2["english_power_n%d" % len(xe2)])
print("->", os.path.relpath(OUT, ROOT))
