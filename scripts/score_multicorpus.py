"""Score the multicorpus forecast (results/mc/forecast.json, written before training) against the 20 training runs.

    python scripts/score_multicorpus.py        # -> results/mc/score.json

Reads forecast.json (never writes it), results/text_runs/mc_<corpus>/bytes_b<k>_s<seed>.json, each corpus's reference
curves, and the calibration ladders (TinyStories, FineWeb-Edu, protein; d256/nl4 bytes). Measured quantities:
  L(k)        bits_per_byte, mean over the seeds present at 2^k (two at 2^21, one elsewhere)
  n_impl(k)   context (bytes) at which the corpus's primary reference LM reaches L(k) (SmolLM2-135M; Qwen2.5-0.5B for ja_wiki);
              undefined when L lies outside the curve's range
  eps         slope of ln n_impl on ln D over 2^21-2^23;  Delta = ln n_impl(2^23) - ln n_impl,TS(2^23);  lag = -Delta / (eps ln 2)
Intervals: bootstrap over corpora (corpora resampled with replacement, 20000 draws), percentile 95%.
"""
import glob
import json
import os
import sys
from itertools import permutations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import forecast_input_variants as V  # noqa: E402
import forecast_neff as FN  # noqa: E402

LN2 = np.log(2.0)
KS = (21, 22, 23)
CORPORA = ("code", "pg19", "pubmed", "ja_wiki", "abc")
NB = 20000
rng = np.random.default_rng(0)


def ladder(pat):
    out = {}
    for f in glob.glob(pat):
        d = json.load(open(f))
        out.setdefault(int(round(np.log2(d["train_bytes"]))), {})[d["seed"]] = d
    return out


def slope(ks, y):
    return float(np.polyfit(np.array(ks) * LN2, y, 1)[0])


def spearman(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if np.ptp(p) == 0 or np.ptp(y) == 0:
        return np.nan
    rp, ry = np.argsort(np.argsort(p)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rp, ry)[0, 1])


def spearman_p(p, y):
    """Exact one-sided p of a rank correlation at least this large, over all orderings."""
    r = spearman(p, y)
    if not np.isfinite(r):
        return np.nan
    n = len(p)
    hits = sum(spearman(p, perm) >= r - 1e-12 for perm in permutations(range(n)))
    import math
    return hits / math.factorial(n)


def errs(pred, meas, names):
    e = np.array([pred[c] - meas[c] for c in names], float)
    return e


def summarize(pred, meas, names, base_preds):
    """rms, bias, Spearman for one predictor; ratio of rms to each baseline with a bootstrap over corpora."""
    names = [c for c in names if np.isfinite(pred.get(c, np.nan)) and np.isfinite(meas.get(c, np.nan))]
    if len(names) < 2:
        return dict(n=len(names), corpora=names)
    e = errs(pred, meas, names)
    out = dict(n=len(names), corpora=names, rms=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
               mae=float(np.mean(np.abs(e))), err={c: float(x) for c, x in zip(names, e)})
    if len(names) >= 3:
        out["spearman"] = spearman([pred[c] for c in names], [meas[c] for c in names])
        out["spearman_p_exact"] = spearman_p([pred[c] for c in names], [meas[c] for c in names])
    for bn, bp in base_preds.items():
        eb = errs(bp, meas, names)
        idx = rng.integers(0, len(names), size=(NB, len(names)))
        r = np.sqrt((e[idx] ** 2).mean(1)) / np.maximum(np.sqrt((eb[idx] ** 2).mean(1)), 1e-12)
        out[f"rms_ratio_vs_{bn}"] = float(np.sqrt(np.mean(e ** 2)) / max(np.sqrt(np.mean(eb ** 2)), 1e-12))
        out[f"rms_ratio_vs_{bn}_ci95"] = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
        out[f"share_draws_better_than_{bn}"] = float(np.mean(r < 1))
    return out


def run(qwen_all=False):
    global rng
    rng = np.random.default_rng(0)
    F = json.load(open(os.path.join(ROOT, "results/mc/forecast.json")))
    fc = F["forecaster_constants"]

    # ---------------------------------------------------------------- calibration ladders (measured)
    R, _ = FN.load_ts()
    ts_L = {k: float(np.mean(R["bytes"][k])) for k in KS}
    fw = ladder(os.path.join(ROOT, "results/text_runs/bfine/bytes_b*_s*.json"))
    pr = ladder(os.path.join(ROOT, "results/text_runs/bprot/bytes_b*_s*.json"))
    fw_L = {k: float(np.mean([d["bits_per_byte"] for d in fw[k].values()])) for k in KS}
    pr_L = {k: float(np.mean([d["bits_per_byte"] for d in pr[k].values()])) for k in KS}
    const_L = {k: float(np.mean([ts_L[k], fw_L[k], pr_L[k]])) for k in KS}
    ts_href = FN.HRef(FN.TS_HREF)
    ts_n23 = ts_href.neff(ts_L[23])
    CAL = F["constants"]["calibration_measured"]

    # ---------------------------------------------------------------- measured outcomes
    meas, seeds = {}, {}
    for c in CORPORA:
        lad = ladder(os.path.join(ROOT, f"results/text_runs/mc_{c}/bytes_b*_s*.json"))
        L = {k: float(np.mean([d["bits_per_byte"] for d in lad[k].values()])) for k in KS}
        L0 = {k: lad[k][0]["bits_per_byte"] for k in KS}
        LvalA = {k: float(np.mean([d["bits_per_byte_valA"] for d in lad[k].values()])) for k in KS}
        seeds[c] = dict(s0=lad[21][0]["bits_per_byte"], s1=lad[21][1]["bits_per_byte"],
                        diff=lad[21][1]["bits_per_byte"] - lad[21][0]["bits_per_byte"])
        p = os.path.join(ROOT, f"results/mc/{c}/refcurve/refcurve_fineweb.json")
        primary = "Qwen2.5-0.5B" if (qwen_all or c == "ja_wiki") else F["corpora"][c]["stats"]["ref_primary"]
        per_ref = {}
        for m in ("SmolLM2-135M", "Qwen2.5-0.5B"):
            hr = FN.HRef(p, m)
            inside = {k: bool(hr.y.min() <= L[k] <= hr.y.max()) for k in KS}
            n = {k: hr.neff(L[k]) if inside[k] else np.nan for k in KS}
            eps = slope(KS, [np.log(n[k]) for k in KS]) if all(inside.values()) else np.nan
            delta = float(np.log(n[23]) - np.log(ts_n23)) if inside[23] else np.nan
            dhdln = {k: hr.slope_logn(np.log(n[k])) if inside[k] else np.nan for k in KS}
            g, _ = V.power_fit(hr)
            per_ref[m] = dict(curve_bpb_range=[float(hr.y.min()), float(hr.y.max())], inside=inside,
                              n_impl={k: float(n[k]) for k in KS}, eps=eps, delta=delta,
                              lag=float(-delta / (eps * LN2)) if np.isfinite(delta) and np.isfinite(eps) and eps > 0 else np.nan,
                              abs_dh_dln_n=dhdln, gamma=g)
        meas[c] = dict(L=L, L_seed0=L0, L_validA=LvalA, primary_ref=primary, ref=per_ref,
                       steps={k: [d["steps"] for d in lad[k].values()] for k in KS},
                       loss_drop_per_doubling=float(-slope(KS, [L[k] for k in KS]) * LN2),
                       ln_loss_slope=float(-slope(KS, [np.log(L[k]) for k in KS])),
                       pos_bpb_2_23=lad[23][0]["pos_bpb"], trunc_bpb_2_23=lad[23][0].get("trunc_bpb"))
        pm = per_ref[primary]
        meas[c].update(eps=pm["eps"], delta=pm["delta"], lag=pm["lag"], n_impl=pm["n_impl"])

    # ---------------------------------------------------------------- seed noise and resolution
    d = np.array([seeds[c]["diff"] for c in CORPORA])
    sd1 = float(np.sqrt(np.mean(d ** 2) / 2))
    res_eps, res_delta = {}, {}
    for c in CORPORA:
        pm = meas[c]["ref"][meas[c]["primary_ref"]]
        if not all(pm["inside"].values()):
            continue
        s = [sd1 / pm["abs_dh_dln_n"][k] for k in KS]
        s[0] /= np.sqrt(2)  # two seeds at 2^21
        # least-squares slope SE over x = k ln2 with unequal noise (unweighted fit)
        x = np.array(KS) * LN2
        w = (x - x.mean()) / ((x - x.mean()) ** 2).sum()
        res_eps[c] = float(np.sqrt(np.sum((w * np.array(s)) ** 2)))
        res_delta[c] = float(s[2])
    noise = dict(seed_pairs=seeds, sd_single_run_bpb=sd1, se_mean_of_2_bpb=sd1 / np.sqrt(2),
                 se_eps=res_eps, se_delta=res_delta,
                 note="sd_single_run = sqrt(mean(diff^2)/2) over the five 2^21 pairs; eps and Delta SEs propagate it through "
                      "the local slope |dh/dln n| of the primary reference curve")

    # ---------------------------------------------------------------- predictions
    ok_ref = [c for c in CORPORA if np.isfinite(meas[c]["delta"])]
    ok_eps = [c for c in CORPORA if np.isfinite(meas[c]["eps"])]
    P = {c: F["corpora"][c] for c in CORPORA}

    # Delta (ln context ratio to TinyStories at 2^23)
    dpred = {n: {c: P[c]["level"][n] for c in CORPORA} for n in ("A1_K2", "A1_K210", "A2_K2", "A2_K210", "A0")}
    dbase = dict(constant={c: F["fits"]["A0"] for c in CORPORA}, ignore_corpus={c: CAL["fw"]["delta"] for c in CORPORA},
                 main_forecaster={c: 0.0 for c in CORPORA})
    dmeas = {c: meas[c]["delta"] for c in CORPORA}
    level_delta = {n: summarize(v, dmeas, CORPORA, dbase) for n, v in {**dpred, **{f"base_{k}": v for k, v in dbase.items()}}.items()}
    level_delta["A3_bracket_coverage"] = {c: (bool(min(P[c]["level"]["A3_bracket"]) <= dmeas[c] <= max(P[c]["level"]["A3_bracket"]))
                                              if np.isfinite(dmeas[c]) else None) for c in CORPORA}

    # eps (context exponent)
    gam = {c: meas[c]["ref"][meas[c]["primary_ref"]]["gamma"] for c in CORPORA}
    epred = {n: {c: P[c]["exponent"].get(n, np.nan) for c in CORPORA} for n in ("B1_K2", "B1_K210", "B2_K2", "B2_K210", "B3", "B4", "B0")}
    ebase = dict(constant={c: F["fits"]["B0"] for c in CORPORA}, ignore_corpus={c: CAL["fw"]["eps"] for c in CORPORA},
                 main_forecaster={c: fc["nu"] for c in CORPORA},
                 info_nu_1_over_1pg={c: 1 / (1 + gam[c]) for c in CORPORA},
                 g_over_1pg={c: gam[c] / (1 + gam[c]) for c in CORPORA})
    emeas = {c: meas[c]["eps"] for c in CORPORA}
    exponent = {n: summarize(v, emeas, CORPORA, {k: ebase[k] for k in ("constant", "ignore_corpus", "main_forecaster")})
                for n, v in {**epred, **{f"base_{k}": v for k, v in ebase.items()}}.items()}

    # lag in doublings (where measured Delta and eps exist)
    lag = {}
    lmeas = {c: meas[c]["lag"] for c in CORPORA}
    for key in ("A2_K2/B2_K2", "A2_K210/B2_K210", "A2_K2/B0", "A2_K210/B0", "A2_K2/B1_K2", "A2_K210/B1_K210"):
        lp = {c: P[c]["lag_doublings_from_A2"].get(key, np.nan) for c in CORPORA}
        lag[key] = summarize(lp, lmeas, CORPORA, {})
    lag["A0/B0 (constant)"] = summarize({c: -F["fits"]["A0"] / (F["fits"]["B0"] * LN2) for c in CORPORA}, lmeas, CORPORA, {})
    lag["FW (ignore corpus)"] = summarize({c: -CAL["fw"]["delta"] / (CAL["fw"]["eps"] * LN2) for c in CORPORA}, lmeas, CORPORA, {})

    # bits/byte at each budget
    def via_ref(c, delta, eps, k):
        p = os.path.join(ROOT, f"results/mc/{c}/refcurve/refcurve_fineweb.json")
        hr = FN.HRef(p, meas[c]["primary_ref"])
        return float(hr.h(ts_n23 * np.exp(delta) * 2.0 ** ((k - 23) * eps)))

    prim = {c: meas[c]["primary_ref"] for c in CORPORA}
    bpred = {
        "C_main": {(c, k): P[c]["main_forecaster_bytes"][prim[c]]["main"][str(k)] for c in CORPORA for k in KS},
        "C_info_nu": {(c, k): P[c]["main_forecaster_bytes"][prim[c]]["info_nu"][str(k)] for c in CORPORA for k in KS},
        "A2_K2+B2_K2": {(c, k): via_ref(c, P[c]["level"]["A2_K2"], P[c]["exponent"]["B2_K2"], k) for c in CORPORA for k in KS},
        "A2_K210+B2_K210": {(c, k): via_ref(c, P[c]["level"]["A2_K210"], P[c]["exponent"]["B2_K210"], k) for c in CORPORA for k in KS},
        "A1_K2+B1_K2": {(c, k): via_ref(c, P[c]["level"]["A1_K2"], P[c]["exponent"]["B1_K2"], k) for c in CORPORA for k in KS},
        "A1_K210+B1_K210": {(c, k): via_ref(c, P[c]["level"]["A1_K210"], P[c]["exponent"]["B1_K210"], k) for c in CORPORA for k in KS},
        "A0+B0 via ref": {(c, k): via_ref(c, F["fits"]["A0"], F["fits"]["B0"], k) for c in CORPORA for k in KS},
        "FW Delta,eps via ref": {(c, k): via_ref(c, CAL["fw"]["delta"], CAL["fw"]["eps"], k) for c in CORPORA for k in KS},
    }
    bbase = {"constant_bpb": {(c, k): const_L[k] for c in CORPORA for k in KS},
             "ignore_corpus_FW_bpb": {(c, k): fw_L[k] for c in CORPORA for k in KS}}
    bmeas = {(c, k): meas[c]["L"][k] for c in CORPORA for k in KS}

    def score_bpb(pred, names):
        """Pooled over corpora x budgets; bootstrap resamples corpora (all three budgets travel together)."""
        cells = [(c, k) for c in names for k in KS]
        e = np.array([pred[q] - bmeas[q] for q in cells]).reshape(len(names), len(KS))
        out = dict(corpora=list(names), rms=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
                   per_budget={k: dict(rms=float(np.sqrt(np.mean(e[:, i] ** 2))), bias=float(np.mean(e[:, i]))) for i, k in enumerate(KS)},
                   per_corpus={c: [float(x) for x in e[j]] for j, c in enumerate(names)},
                   spearman_2_23=spearman([pred[(c, 23)] for c in names], [bmeas[(c, 23)] for c in names]),
                   spearman_2_23_p_exact=spearman_p([pred[(c, 23)] for c in names], [bmeas[(c, 23)] for c in names]))
        # loss drop per doubling, predicted vs measured
        dp = {c: (pred[(c, 21)] - pred[(c, 23)]) / 2 for c in names}
        dm = {c: (bmeas[(c, 21)] - bmeas[(c, 23)]) / 2 for c in names}
        ed = np.array([dp[c] - dm[c] for c in names])
        out["drop_per_doubling"] = dict(rms=float(np.sqrt(np.mean(ed ** 2))), bias=float(np.mean(ed)),
                                        spearman=spearman([dp[c] for c in names], [dm[c] for c in names]),
                                        pred={c: float(dp[c]) for c in names})
        for bn, bp in bbase.items():
            eb = np.array([bp[q] - bmeas[q] for q in cells]).reshape(len(names), len(KS))
            idx = rng.integers(0, len(names), size=(NB, len(names)))
            r = np.sqrt((e[idx] ** 2).mean((1, 2))) / np.sqrt((eb[idx] ** 2).mean((1, 2)))
            out[f"rms_ratio_vs_{bn}"] = float(out["rms"] / np.sqrt(np.mean(eb ** 2)))
            out[f"rms_ratio_vs_{bn}_ci95"] = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
            out[f"share_draws_better_than_{bn}"] = float(np.mean(r < 1))
        return out

    sets = {"all5": list(CORPORA), "competent_ref4": [c for c in CORPORA if c != "abc"]}
    level_bpb = {s: {n: score_bpb(v, names) for n, v in {**bpred, **bbase}.items()} for s, names in sets.items()}

    out = dict(
        forecast=dict(file="results/mc/forecast.json", created_utc_field=F["created_utc"],
                      note="created_utc is the producer's start time; timing.txt logs the write at 2026-10-16T03:48:34Z"),
        calibration_measured_bpb=dict(ts=ts_L, fw=fw_L, protein=pr_L, constant_mean=const_L, ts_n_impl_2_23=ts_n23),
        measured=meas, seed_noise=noise,
        level_delta=level_delta, exponent=exponent, lag_doublings=lag, level_bpb=level_bpb,
        defined=dict(delta=ok_ref, eps=ok_eps),
        predictions_bpb={n: {f"{c}|{k}": v for (c, k), v in p.items()} for n, p in {**bpred, **bbase}.items()},
        bootstrap=dict(draws=NB, unit="corpus", seed=0))
    return out


def clean(o):
    """NaN -> null so the file is strict JSON."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating, int, np.integer)):
        return None if not np.isfinite(o) else (int(o) if isinstance(o, (int, np.integer)) else float(o))
    return o


def timing():
    """Forecast SHA-256 against the pod log, and each run's training-loop start (file time minus its recorded seconds)."""
    import datetime as dt
    import hashlib
    f = os.path.join(ROOT, "results/mc/forecast.json")
    sha = hashlib.sha256(open(f, "rb").read()).hexdigest()
    logged = open(os.path.join(ROOT, "results/mc/logs/forecast.sha256")).read().split()[0]
    written = [ln.split()[-1] for ln in open(os.path.join(ROOT, "results/mc/logs/timing.txt")) if ln.startswith("forecast written")][0]
    w = dt.datetime.fromisoformat(written.replace("Z", "+00:00"))
    ts = lambda q: dt.datetime.fromtimestamp(os.path.getmtime(q), dt.timezone.utc)  # noqa: E731
    runs = {}
    for q in sorted(glob.glob(os.path.join(ROOT, "results/text_runs/mc_*/bytes_*.json"))):
        d = json.load(open(q))
        st = ts(q) - dt.timedelta(seconds=d["seconds"] + d.get("trunc_seconds", 0))
        runs[os.path.relpath(q, ROOT)] = dict(loop_start_utc=st.isoformat(timespec="seconds"), minutes_after_forecast=(st - w).total_seconds() / 60)
    enc = {os.path.basename(q): ts(q).isoformat(timespec="seconds") for q in glob.glob(os.path.join(ROOT, "results/mc/logs/tok_test_*.log"))}
    return dict(sha256=sha, sha256_logged=logged, sha256_match=sha == logged, forecast_written_utc=written,
                forecast_file_mtime_utc=ts(f).isoformat(timespec="seconds"), test_split_encoded_after=enc,
                all_runs_after_forecast=all(r["minutes_after_forecast"] > 0 for r in runs.values()),
                earliest_run_start_utc=min(r["loop_start_utc"] for r in runs.values()), runs=runs,
                note="the pod job runs the forecast, logs its SHA-256, encodes the test split, then starts training, in one sequential script "
                     "(scripts/pod_job_multicorpus.sh); job.out records '20 jobs to run, 0 finished' after the forecast")


def main():
    out = run()
    out["timing"] = timing()
    alt = run(qwen_all=True)
    out["sensitivity_qwen_reference_for_all"] = dict(
        note="measured n_impl, Delta, eps and the ref-curve bpb predictions read through Qwen2.5-0.5B for every corpus "
             "(the plan's primary is SmolLM2-135M, Qwen2.5-0.5B for ja_wiki)",
        measured={c: dict(eps=v["eps"], delta=v["delta"], lag=v["lag"]) for c, v in alt["measured"].items()},
        level_delta=alt["level_delta"], exponent=alt["exponent"], level_bpb=alt["level_bpb"])
    json.dump(clean(out), open(os.path.join(ROOT, "results/mc/score.json"), "w"), indent=1, allow_nan=False)
    print("wrote results/mc/score.json")


if __name__ == "__main__":
    main()
