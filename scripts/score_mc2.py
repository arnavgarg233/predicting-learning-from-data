"""Score the forecast extension against its training runs.

    python scripts/score_mc2.py        # -> results/mc2/score.json

Reads results/mc2/forecast.json (written before training; never written here), results/text_runs/mc2_<c>/bytes_b<k>_s<seed>.json,
and, for the pooled score, the first test (results/mc/forecast.json, results/text_runs/mc_<c>, corpora code, pg19, pubmed, ja_wiki).
Forecast: the main forecaster (C main) through each corpus's primary reference LM, as written in the forecast file.
Measured: test-partition bits per byte, mean over the seeds at 2^k (two at 2^21). Baselines as in scripts/score_multicorpus.py:
FineWeb-Edu's measured curve (the same bpb for every corpus at each budget) and a constant (mean of the TinyStories, FineWeb-Edu
and protein curves). Eligibility is read from the forecast file. Ratio intervals: bootstrap over corpora (20000 draws, seed 0,
the three budgets of a corpus travel together), percentile 95%.
"""
import datetime as dt
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import forecast_neff as FN  # noqa: E402
from score_multicorpus import clean, ladder, spearman  # noqa: E402

KS = (21, 22, 23)
NB = 20000
FIRST4 = ("code", "pg19", "pubmed", "ja_wiki")


def calibration():
    R, _ = FN.load_ts()
    ts = {k: float(np.mean(R["bytes"][k])) for k in KS}
    fw = ladder(os.path.join(ROOT, "results/text_runs/bfine/bytes_b*_s*.json"))
    pr = ladder(os.path.join(ROOT, "results/text_runs/bprot/bytes_b*_s*.json"))
    fw = {k: float(np.mean([d["bits_per_byte"] for d in fw[k].values()])) for k in KS}
    pr = {k: float(np.mean([d["bits_per_byte"] for d in pr[k].values()])) for k in KS}
    return dict(ts=ts, fw=fw, protein=pr, constant_mean={k: float(np.mean([ts[k], fw[k], pr[k]])) for k in KS})


def cells(F, run_prefix, corpora, test, ref=None, seed0_only=False):
    out = {}
    for c in corpora:
        lad = ladder(os.path.join(ROOT, f"results/text_runs/{run_prefix}_{c}/bytes_b*_s*.json"))
        if not all(k in lad for k in KS):
            out[c] = dict(missing=[k for k in KS if k not in lad])
            continue
        prim = ref or F["corpora"][c]["stats"]["ref_primary"]
        L = {k: float(np.mean([d["bits_per_byte"] for q, d in lad[k].items() if q == 0 or not seed0_only])) for k in KS}
        P = {k: float(F["corpora"][c]["main_forecaster_bytes"][prim]["main"][str(k)]) for k in KS}
        s = lad[21]
        out[c] = dict(test=test, ref_primary=prim, measured=L, forecast=P, err={k: P[k] - L[k] for k in KS},
                      seeds_2_21={str(q): s[q]["bits_per_byte"] for q in sorted(s)}, steps={k: [d["steps"] for d in lad[k].values()] for k in KS},
                      eligibility=F["corpora"][c].get("eligibility"))
    return out


def score(tab, names, cal, seed=0):
    names = [c for c in names if "measured" in tab[c]]
    if not names:
        return dict(n=0)
    rng = np.random.default_rng(seed)
    meas = np.array([[tab[c]["measured"][k] for k in KS] for c in names])
    e = np.array([[tab[c]["forecast"][k] for k in KS] for c in names]) - meas
    base = dict(fineweb_curve=np.array([[cal["fw"][k] for k in KS]] * len(names)) - meas,
                constant=np.array([[cal["constant_mean"][k] for k in KS]] * len(names)) - meas)
    out = dict(n=len(names), corpora=names, rms=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
               per_budget={k: dict(rms=float(np.sqrt(np.mean(e[:, i] ** 2))), bias=float(np.mean(e[:, i]))) for i, k in enumerate(KS)},
               per_corpus_err={c: [float(x) for x in e[j]] for j, c in enumerate(names)})
    if len(names) >= 3:
        out["spearman_2_23"] = spearman([tab[c]["forecast"][23] for c in names], [tab[c]["measured"][23] for c in names])
    idx = rng.integers(0, len(names), size=(NB, len(names)))
    for bn, eb in base.items():
        out[f"rms_{bn}"] = float(np.sqrt(np.mean(eb ** 2)))
        out[f"bias_{bn}"] = float(np.mean(eb))
        out[f"ratio_vs_{bn}"] = out["rms"] / out[f"rms_{bn}"]
        if len(names) >= 2:
            r = np.sqrt((e[idx] ** 2).mean((1, 2))) / np.sqrt((eb[idx] ** 2).mean((1, 2)))
            out[f"ratio_vs_{bn}_ci95"] = [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
            out[f"share_draws_better_than_{bn}"] = float(np.mean(r < 1))
    return out


def timing(res):
    f = os.path.join(ROOT, res, "forecast.json")
    sha = hashlib.sha256(open(f, "rb").read()).hexdigest()
    lp = os.path.join(ROOT, res, "logs/forecast.sha256")
    logged = open(lp).read().split()[0] if os.path.exists(lp) else None
    tp = os.path.join(ROOT, res, "logs/timing.txt")
    written = [ln.split()[-1] for ln in open(tp) if ln.startswith("forecast written")] if os.path.exists(tp) else []
    out = dict(sha256=sha, sha256_logged=logged, sha256_match=sha == logged, forecast_written_utc=written[0] if written else None)
    if written:
        w = dt.datetime.fromisoformat(written[0].replace("Z", "+00:00"))
        starts = []
        for q in glob.glob(os.path.join(ROOT, "results/text_runs/mc2_*/bytes_*.json")):
            d = json.load(open(q))
            starts.append(dt.datetime.fromtimestamp(os.path.getmtime(q), dt.timezone.utc) - dt.timedelta(seconds=d["seconds"] + d.get("trunc_seconds", 0)))
        if starts:
            out.update(earliest_run_loop_start_utc=min(starts).isoformat(timespec="seconds"),
                       all_runs_after_forecast=all(s > w for s in starts),
                       note="loop start = result file mtime minus recorded training seconds (mtimes as copied back from the pod)")
    return out


def main():
    F2 = json.load(open(os.path.join(ROOT, "results/mc2/forecast.json")))
    F1 = json.load(open(os.path.join(ROOT, "results/mc/forecast.json")))
    cal = calibration()
    t2 = cells(F2, "mc2", list(F2["corpora"]), "extension")
    t1 = cells(F1, "mc", FIRST4, "first")
    elig = {c: v for c, v in ((c, F2["corpora"][c].get("eligibility")) for c in F2["corpora"])}
    eligible = [c for c, v in elig.items() if v and v["eligible"]]
    ineligible = [c for c in F2["corpora"] if c not in eligible]
    pooled_tab = {**{f"first:{c}": v for c, v in t1.items()}, **{f"ext:{c}": v for c, v in t2.items()}}
    out = dict(
        forecast=dict(file="results/mc2/forecast.json", created_utc_field=F2["created_utc"], timing=timing("results/mc2")),
        eligibility=dict(rule="eligible if the primary reference LM's bits per byte at full context on ref_stats is below the "
                              "forecast's own learner loss at 2^23 (both numbers from the forecast file)",
                         per_corpus=elig, eligible=eligible, ineligible=ineligible),
        calibration_measured_bpb=cal,
        per_corpus=t2,
        primary_eligible=score(t2, eligible, cal),
        ineligible_reported_separately=score(t2, ineligible, cal),
        all_extension_corpora=score(t2, list(t2), cal),
        first_test_4_recomputed=score(t1, list(FIRST4), cal),
        pooled_first4_plus_eligible=score(pooled_tab, [f"first:{c}" for c in FIRST4] + [f"ext:{c}" for c in eligible], cal),
        first_test_per_corpus=t1,
        bootstrap=dict(draws=NB, unit="corpus", seed=0))
    sens = {}
    for name, kw in (("qwen_reference_for_all", dict(ref="Qwen2.5-0.5B")), ("smollm2_reference_for_all", dict(ref="SmolLM2-135M")),
                     ("seed0_only", dict(seed0_only=True))):
        a2, a1 = cells(F2, "mc2", list(F2["corpora"]), "extension", **kw), cells(F1, "mc", FIRST4, "first", **kw)
        pt = {**{f"first:{c}": v for c, v in a1.items()}, **{f"ext:{c}": v for c, v in a2.items()}}
        sens[name] = dict(extension_eligible=score(a2, eligible, cal),
                          pooled_first4_plus_eligible=score(pt, [f"first:{c}" for c in FIRST4] + [f"ext:{c}" for c in eligible], cal))
    out["sensitivity_not_primary"] = sens
    d = [v["seeds_2_21"]["1"] - v["seeds_2_21"]["0"] for v in t2.values() if "seeds_2_21" in v and len(v["seeds_2_21"]) == 2]
    out["seed_noise"] = dict(diffs_2_21=d, sd_single_run_bpb=float(np.sqrt(np.mean(np.square(d)) / 2)) if d else None)
    json.dump(clean(out), open(os.path.join(ROOT, "results/mc2/score.json"), "w"), indent=1, allow_nan=False)
    print("wrote results/mc2/score.json")


if __name__ == "__main__":
    main()
