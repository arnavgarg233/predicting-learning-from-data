"""Stage 2 of the applicability analysis: score the criterion (section 3) on the
held-out sets, with the rules of section 4. Thresholds are read from the design, not from held-out data.
    python scripts/applicability/score.py -> results/applicability/score.json
Choice made while writing this script, before running it: the eight-prediction agreement tally uses the held-out corpora
that pass C1 and C3 (the criterion applied as a whole); the tally on all held-out corpora is reported beside it."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

N_DOCS_MIN, C_MAX, LARGE_CURVE, LARGE_FALL, S5A = 75, 0.70, 0.25, 0.10, 0.5
cal = C.J(os.path.join(C.OUT, "calibration.json"))
rows = C.fall_rows()
eng = C.english_forward_rows()
by = {(r["set"], r["corpus"]): r for r in rows}
for r in eng:
    by[("forward_english", r["corpus"])] = r


def xwin(kind, x):
    lo, hi = cal["kinds"][kind]["x_window"]
    return lo <= x <= hi


# ------------------------------------------------------------------ curve endpoint rows
curve = []
for name, path in (("stage3", "results/curve_forecast/stage3/score_stage3.json"),
                   ("forward18", "results/curve_forecast/new_corpora/score.json")):
    for c, p in C.J(path)["per_corpus"].items():
        r = by[(name, c)]
        curve.append(dict(set=name, corpus=c, kind=r["kind"], c=r["c"], n_docs=r["n_docs"], e=p["err"]["25"]["A0"], e2=p["err"]["25"]["A3"]))
for c, p in C.J("results/extbench/score.json")["per_corpus"].items():
    r = by[("external45", c)]
    assert abs(p["c_ratio"] - r["c"]) < 1e-4, c
    curve.append(dict(set="external45", corpus=c, kind=r["kind"], c=r["c"], n_docs=r["n_docs"], e=p["err"]["ii"]["e25"], e2=p["err"]["A3"]["e25"]))


def a3_e25(L):
    l25 = L["25"]
    return L["20"] * (L["20"] / L["19"]) ** 5 - l25


S = C.J("results/fwd_weblang/score.json")["per_corpus"]
for c, p in S.items():
    r = by[("forward30", c)]
    assert abs(a3_e25(p["L"]) - p["A3_e25_own"]) < 1e-9, c   # A3 = power law through 2^19 and 2^20
    curve.append(dict(set="forward_web", corpus=c, kind="web", c=r["c"], n_docs=r["n_docs"], e=p["e25_level"]["own_cal"], e2=p["A3_e25_own"]))
S = C.J("results/fwd_english/score.json")["per_corpus"]
for r in eng:
    p = S[r["corpus"]]
    curve.append(dict(set="forward_english", corpus=r["corpus"], kind="english", c=r["c"], n_docs=r["n_docs"],
                      e=p["e25_level"]["own_cal"], e2=a3_e25(p["L"])))
assert len(curve) == 133, len(curve)
for q in curve:
    q["C1"] = q["n_docs"] >= N_DOCS_MIN; q["C2"] = q["c"] < C_MAX; q["trusted"] = q["C1"] and q["C2"]


def group_stats(sel, e, e2, large):
    n = int(sel.sum())
    if n == 0:
        return dict(n=0)
    a, b = e[sel], e2[sel]
    k_large = int(np.sum(np.abs(a) > large)); k_closer = int(np.sum(np.abs(a) < np.abs(b)))
    out = dict(n=n, rms=C.rms(a), rms_pilots=C.rms(b), mae=float(np.mean(np.abs(a))),
               large=dict(k=k_large, share=k_large / n, wilson95=C.wilson(k_large, n)),
               closer_than_pilots=dict(k=k_closer, share=k_closer / n, wilson95=C.wilson(k_closer, n)))
    if n >= 3:
        out["ratio_vs_pilots"] = C.ratio(a, b)
    return out


def separation(flag, e, large, seed=0):
    """flagged vs passing: difference in large-miss share (bootstrap within groups) and permutation test of mean |e|."""
    ae = np.abs(e); f = flag.astype(bool)
    if f.sum() == 0 or (~f).sum() == 0:
        return None
    rng = np.random.default_rng(seed)
    lf, lp = (ae[f] > large).astype(float), (ae[~f] > large).astype(float)
    d = lf.mean() - lp.mean()
    bd = np.array([lf[rng.integers(0, len(lf), len(lf))].mean() - lp[rng.integers(0, len(lp), len(lp))].mean() for _ in range(C.NB)])
    lo, hi = np.percentile(bd, [2.5, 97.5])
    obs = ae[f].mean() - ae[~f].mean()
    perm = np.empty(C.NB)
    for i in range(C.NB):
        pf = rng.permutation(f)
        perm[i] = ae[pf].mean() - ae[~pf].mean()
    p = float((np.sum(perm >= obs) + 1) / (C.NB + 1))
    return dict(n_flagged=int(f.sum()), n_pass=int((~f).sum()), large_share_flagged=float(lf.mean()), large_share_pass=float(lp.mean()),
                diff=float(d), diff_ci95=[float(lo), float(hi)],
                label="REAL positive" if lo > 0 else ("REAL negative" if hi < 0 else "tie"),
                mae_flagged=float(ae[f].mean()), mae_pass=float(ae[~f].mean()), perm_p_one_sided=p)


def curve_block(qs):
    e = np.array([q["e"] for q in qs]); e2 = np.array([q["e2"] for q in qs])
    out = {}
    for lab, key in (("C1_and_C2", "trusted"), ("C1", "C1"), ("C2", "C2")):
        ok = np.array([q[key] for q in qs])
        out[lab] = dict(passing=group_stats(ok, e, e2, LARGE_CURVE), flagged=group_stats(~ok, e, e2, LARGE_CURVE),
                        separation=separation(~ok, e, LARGE_CURVE))
    return out


res = dict(thresholds=dict(n_docs_min=N_DOCS_MIN, c_max=C_MAX, large_curve=LARGE_CURVE, large_fall=LARGE_FALL, s5a=S5A))
res["curve"] = dict(all=curve_block(curve))
for s in ("stage3", "forward18", "external45", "forward_web", "forward_english"):
    res["curve"][s] = curve_block([q for q in curve if q["set"] == s])
for k in ("web", "english", "code", "wiki"):
    qs = [q for q in curve if q["kind"] == k]
    res["curve"][f"kind_{k}"] = curve_block(qs)
# documents alone among corpora that pass C2 (removes the kind confound of c)
qs = [q for q in curve if q["C2"]]
res["curve"]["C1_within_C2_pass"] = curve_block(qs)["C1"]
bins = [(0, 75), (75, 200), (200, 10 ** 9)]
res["curve"]["by_docs_bin"] = {f"{a}-{b}": group_stats(np.array([a <= q["n_docs"] < b for q in curve]), np.array([q["e"] for q in curve]),
                                                       np.array([q["e2"] for q in curve]), LARGE_CURVE) for a, b in bins}
res["curve"]["rows"] = curve

# ------------------------------------------------------------------ fall endpoint
def kind_rows(kind, sets, src):
    return [r for r in src if r["kind"] == kind and r["set"] in sets]


fall = {}
for kind in ("web", "english", "code", "wiki"):
    calr = [r for r in rows if r["kind"] == kind and r["set"] in C.CAL[kind] and not (kind == "english" and r["corpus"] in C.SET_ASIDE_EN)]
    assert [r["corpus"] for r in calr] == cal["kinds"][kind]["corpora"]
    held = eng if kind == "english" else kind_rows(kind, C.HELD[kind], rows)
    y0 = np.array([r["fall"] for r in calr]); x0 = np.array([r["x"] for r in calr]); g0 = np.array([r["ngram23"] for r in calr])
    co_own = C.ols([x0], y0); co_ng = C.ols([g0], y0); co_ngx = C.ols([g0, x0], y0); mean = float(y0.mean())
    if kind == "web":
        Z = C.J("results/fwd_weblang/fall_line.json")["calibration"]; assert abs(co_own[0] - Z["a"]) < 1e-9 and abs(co_own[1] - Z["b"]) < 1e-9
    if kind == "english":
        Z = C.J("results/fwd_english/fall_line.json")["calibration"]["line"]; assert abs(co_own[0] - Z["a"]) < 1e-9 and abs(co_own[1] - Z["b"]) < 1e-9
    y = np.array([r["fall"] for r in held]); x = np.array([r["x"] for r in held]); g = np.array([r["ngram23"] for r in held]); n = len(held)
    E = dict(own=C.pred(co_own, [x], n) - y, mean=mean - y, ngram23=C.pred(co_ng, [g], n) - y, ngram23_own=C.pred(co_ngx, [g, x], n) - y)
    c1 = np.array([r["n_docs"] >= N_DOCS_MIN for r in held]); c3 = np.array([xwin(kind, r["x"]) for r in held]); ok = c1 & c3
    blk = dict(n=n, lines=dict(own=co_own.tolist(), ngram23=co_ng.tolist(), ngram23_own=co_ngx.tolist(), mean=mean),
               n_fail_C1=int((~c1).sum()), n_fail_C3=int((~c3).sum()), n_pass=int(ok.sum()),
               failed=[dict(corpus=r["corpus"], set=r["set"], n_docs=r["n_docs"], x=r["x"], C1=bool(a), C3=bool(b), own_err=float(e))
                       for r, a, b, e in zip(held, c1, c3, E["own"]) if not (a and b)])
    for lab, sel in (("all", np.ones(n, bool)), ("pass", ok)):
        blk[lab] = {f"{a} / {b}": C.ratio(E[a][sel], E[b][sel]) for a, b in
                    (("own", "mean"), ("ngram23_own", "ngram23"), ("own", "ngram23"), ("ngram23_own", "mean"))}
        blk[lab]["rms"] = {k: C.rms(v[sel]) for k, v in E.items()}
        blk[lab]["n"] = int(sel.sum())
    blk["per_corpus"] = dict(passing=group_stats(ok, E["own"], E["mean"], LARGE_FALL), flagged=group_stats(~ok, E["own"], E["mean"], LARGE_FALL))
    blk["_E"] = E; blk["_ok"] = ok
    fall[kind] = blk

# pooled per-corpus fall separation (C1 or C3 failing)
Eown = np.concatenate([fall[k]["_E"]["own"] for k in fall]); Emean = np.concatenate([fall[k]["_E"]["mean"] for k in fall])
OK = np.concatenate([fall[k]["_ok"] for k in fall])
res["fall_per_corpus_pooled"] = dict(passing=group_stats(OK, Eown, Emean, LARGE_FALL), flagged=group_stats(~OK, Eown, Emean, LARGE_FALL),
                                     separation=separation(~OK, Eown, LARGE_FALL))

# kind-level predictions (C4, C5)
PRED = dict(web=(True, False), english=(True, True), code=(False, False), wiki=(True, False))
tally = {}
for lab in ("pass", "all"):
    hits = []
    for k, (p4, p5) in PRED.items():
        r4 = fall[k][lab]["own / mean"]; r5 = fall[k][lab]["ngram23_own / ngram23"]
        o4, o5 = r4["ci95"][1] < 1, r5["ci95"][1] < 1
        hits += [dict(kind=k, rule="C4", predicted=p4, observed=o4, hit=p4 == o4, ratio=r4),
                 dict(kind=k, rule="C5", predicted=p5, observed=o5, hit=p5 == o5, ratio=r5)]
    kk = sum(h["hit"] for h in hits)
    wk = [h for h in hits if h["kind"] == "wiki"]
    tally[lab] = dict(hits=hits, k=kk, n=len(hits), wilson95=C.wilson(kk, len(hits)),
                      wiki_k=sum(h["hit"] for h in wk), wiki_n=len(wk))
res["kind_predictions"] = tally
res["fall"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in fall.items()}

# ------------------------------------------------------------------ open-sci-ref (descriptive)
fi = C.J("results/curve_forecast/opensciref/forecast_inputs.json")["corpora"]
ds = C.J("results/curve_forecast/opensciref/data_stats.json")
osr = {}
for c in fi:
    if c == "commoncorpus":
        rc = C.J("results/opensciref_forecast/refcurve/commoncorpus/SmolLM2-135M.json")
        r_full = rc["rows"][-1]["bits_per_byte"] if "bits_per_byte" in rc["rows"][-1] else None
        nd = rc["n_docs"]; nt = rc["n_targets"]
    else:
        rc = C.J(f"results/curve_forecast/opensciref/refcurve/{c}/SmolLM2-135M.json")
        r_full = fi[c]["ref_full_bpb"]; nd = rc["n_docs"]; nt = rc["n_targets"]
    osr[c] = dict(n_docs=nd, n_targets=nt, g=ds[c]["g"], r=r_full, c=(r_full / ds[c]["g"]) if r_full else None,
                  C1=nd >= N_DOCS_MIN, C2=(r_full / ds[c]["g"] < C_MAX) if r_full else None)
res["opensciref"] = osr

json.dump(res, open(os.path.join(C.OUT, "score.json"), "w"), indent=1, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o))
