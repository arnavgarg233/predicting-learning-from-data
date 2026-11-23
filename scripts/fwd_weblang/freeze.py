"""Forward test of the within-kind fall calibration on fresh FineWeb-2 web languages: freeze the calibration and the corpus
order before any new corpus is prepared. Design: docs/analysis/fwd_weblang_2026-10-31.md.

    python scripts/fwd_weblang/freeze.py      # -> results/fwd_weblang_2026-10-31/frozen.json (refuses to overwrite)

Calibration: ordinary least squares of measured fall L(2^23) - L(2^25) on own fall S_c(2^23) - S_c(2^25) over all 29 external
web languages (shape_vs_pooled.json rows, set external45, family lang), the full-kind fit of shape_within_kind.py; kind mean fall
of the same 29. Corpora: the FineWeb-2 'lang' spares of results/extbench_2026-10-28/draw.json in their seeded order, minus any
the external benchmark already ran as a replacement (pod*/pod/corpora.json), at most 30.
"""
import datetime
import glob
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "fwd_weblang_2026-10-31", "frozen.json")
EB = os.path.join(ROOT, "results", "extbench_2026-10-28")
HASHED = ["scripts/fwd_weblang/freeze.py", "scripts/fwd_weblang/fwd_forecast.py", "scripts/fwd_weblang/score.py",
          "scripts/referee_reanalysis/shape_within_kind.py", "scripts/referee_reanalysis/shape_vs_pooled.py",
          "results/referee_reanalysis/shape_vs_pooled.json", "results/referee_reanalysis/shape_within_kind.json",
          "scripts/extbench/forecast_ii.py", "scripts/extbench/draw.py", "scripts/extbench_main.py", "scripts/cf_stage2_main.py",
          "scripts/curve_forecast/pod_main.py", "scripts/curve_forecast/forecast.py", "scripts/curve_forecast/core.py",
          "scripts/prep_corpus.py", "scripts/run_text_ladder.py", "text/refcurve/refcurve_fineweb.py", "text/tok_family.py",
          "results/curve_forecast/locked.json", "results/extbench_2026-10-28/draw.json",
          "ops/run_extbench_pod.sh", "ops/pod_setup_extbench.sh", "ops/pod_guard_extbench.sh", "ops/pod_selfstop_extbench.sh"]


def sha(p):
    return hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"{OUT} exists; not overwriting")
    rows = [r for r in json.load(open(os.path.join(ROOT, "results/referee_reanalysis/shape_vs_pooled.json")))["per_corpus"]
            if r["set"] == "external45" and r["family"] == "lang"]
    assert len(rows) == 29, len(rows)
    x = np.array([r["S23"] - r["S25"] for r in rows]); y = np.array([r["fall"] for r in rows])
    b, a = np.polyfit(x, y, 1)
    m = float(y.mean())
    swk = json.load(open(os.path.join(ROOT, "results/referee_reanalysis/shape_within_kind.json")))["kinds"]["web"]
    assert abs(swk["full_fit"]["a"] - a) < 1e-12 and abs(swk["full_fit"]["b"] - b) < 1e-12 and abs(swk["full_fit"]["mean_fall"] - m) < 1e-12
    draw = json.load(open(os.path.join(EB, "draw.json")))["families"]["lang"]
    used = set()
    for f in glob.glob(os.path.join(EB, "pod*", "pod", "corpora.json")):
        c = json.load(open(f))
        used |= {t["name"] for t in c["test"]} | {t["failed"] for t in c.get("replaced", [])}
    spares = [draw["corpus_names"][s] for s in draw["spares"]]
    order = [s for s in spares if s not in used][:30]
    dropped_used = [s for s in spares if s in used]
    now = datetime.datetime.now().astimezone()
    out = dict(
        date=now.strftime("%Y-%m-%d %H:%M:%S %Z"), date_epoch=int(now.timestamp()),
        design="docs/analysis/fwd_weblang_2026-10-31.md",
        calibration=dict(rule="measured fall L(2^23)-L(2^25) = a + b * own_fall, own_fall = S_c(2^23) - S_c(2^25) (forecast (i) 'S'); "
                              "OLS over all 29 external web languages",
                         a=float(a), b=float(b), kind_mean_fall=m, n=len(rows), corpora=sorted(r["corpus"] for r in rows),
                         spearman_in_sample=swk["spearman"], loo_ratio_exploratory=[swk["loo"]["ratio"]] + swk["loo"]["ci95"]),
        forecast_per_corpus=dict(level_23="forecast (ii) level_23 = NGRAM[0] + NGRAM[1] * ngram23 (scripts/extbench/forecast_ii.py)",
                                 own_fall="S(2^23) - S(2^25) of forecast (i)", calibrated_fall="a + b * own_fall",
                                 pred_L25="level_23 - calibrated_fall", pred_L25_kind_mean="level_23 - kind_mean_fall",
                                 when="written with a date by scripts/fwd_weblang/fwd_forecast.py before the first training run of the corpus"),
        corpora=dict(source="results/extbench_2026-10-28/draw.json families.lang.spares (seeded order), minus any already run "
                            "in the external benchmark", order=order, removed_as_used_in_extbench=dropped_used,
                     failure_rule="a corpus whose preparation or reference pass fails is dropped (reason logged), never replaced",
                     cost_rule="if the projected spend would exceed $6.00 all-in, corpora are dropped from the end of the order"),
        pipeline="extbench unchanged: prep_corpus.py, Qwen2.5-0.5B reference pass with 20,000 targets, forecast.py (i), "
                 "forecast_ii.py, learner of pod_main.COMMON; runs seed 0 at 2^19, 2^20, 2^23, 2^25 and seed 1 at 2^25",
        scoring=dict(
            script="scripts/fwd_weblang/score.py",
            measured_fall="L(2^23) seed 0 minus L(2^25) mean of seeds 0 and 1",
            primary="r.m.s. error of the fall 2^23->2^25: frozen calibrated own fall against frozen kind mean fall; paired "
                    "corpus-bootstrap ratio own/mean (20,000 draws, numpy default_rng(0), 95% percentile interval); closer count",
            verdict="REAL positive if the interval lies below 1, REAL negative if above 1, tie otherwise",
            context=["two pilots A3 (2^19, 2^20, locked floor rule): fall A3(2^23) - A3(2^25)",
                     "unscaled own fall S(2^23) - S(2^25)", "pooled exponent 0.14: level_23 * (1 - 2^-0.28)"],
            secondary=["r.m.s. error at 2^25 at the forecast level: level_23 - fall_arm - L(2^25)",
                       "r.m.s. error at 2^25 at the observed L(2^23): L(2^23) - fall_arm - L(2^25)",
                       "Spearman own fall vs measured fall (bootstrap 95% interval)"],
            noise_floor="about 0.015 bits per byte per measured fall"),
        sha256={p: sha(p) for p in HASHED})
    json.dump(out, open(OUT, "w"), indent=1)
    print(out["date"], f"a={a:.4f} b={b:.4f} mean={m:.4f}", len(order), "corpora:", " ".join(order))
    print("removed (used in extbench):", dropped_used)


if __name__ == "__main__":
    main()
