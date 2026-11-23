"""Score a union of forecast stages with the locked scorer, unchanged.

    python scripts/curve_forecast/score_merged.py 80       # stage 1 + stage 2 -> results/curve_forecast/score80.json
    python scripts/curve_forecast/score_merged.py stage3   # stage 3 alone     -> results/curve_forecast/stage3/score_stage3.json
    [--out PATH]

The locked score.py reads one forecast.json, one corpus list and one tree of learner runs. This script builds that tree in
a temporary directory: each stage's forecast.json is first checked against the SHA-256 logged when it was written
(logs/forecast.sha256), the per-corpus entries of the stages are copied verbatim into one forecast.json, the reference
curves and learner runs of those corpora are copied next to it, and score.py and core.py are checked against the SHA-256 values
in locked.json. score.py then runs unchanged, and the stage SHA-256 values are added to its output as `merged_from`.
The stage-3 score covers the 16 further Wikipedia languages only (stage 1 is used for the shared header fields).
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CF = os.path.join(R, "results/curve_forecast")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["80", "stage3"])
    ap.add_argument("--out")
    a = ap.parse_args()
    s1 = CF
    s2 = os.path.join(CF, "stage2" if a.which == "80" else "stage3")
    out = a.out or (os.path.join(CF, "score80.json") if a.which == "80" else os.path.join(CF, "stage3/score_stage3.json"))
    chk = {}
    for name, d in (("stage1", s1), ("stage2" if a.which == "80" else "stage3", s2)):
        chk[name] = dict(sha256=sha(f"{d}/forecast.json"), logged=open(f"{d}/logs/forecast.sha256").read().split()[0])
        assert chk[name]["sha256"] == chk[name]["logged"], name
    F1, F2 = json.load(open(f"{s1}/forecast.json")), json.load(open(f"{s2}/forecast.json"))
    assert F1["locked_sha256"] == F2["locked_sha256"]
    assert F1["code_sha256"]["scripts/curve_forecast/core.py"] == F2["code_sha256"]["scripts/curve_forecast/core.py"]
    assert not set(F1["corpora"]) & set(F2["corpora"])
    with tempfile.TemporaryDirectory() as T:
        os.makedirs(f"{T}/scripts")
        os.makedirs(f"{T}/results/curve_forecast/logs")
        os.makedirs(f"{T}/results/text_runs")
        shutil.copytree(f"{R}/scripts/curve_forecast", f"{T}/scripts/curve_forecast",
                        ignore=shutil.ignore_patterns("__pycache__", "score_merged.py"))
        for n in ("forecast_neff.py", "forecast_input_variants.py", "forecast_multicorpus.py", "ngram_curve_ts.py"):
            shutil.copy(f"{R}/scripts/{n}", f"{T}/scripts/{n}")
        shutil.copy(f"{s1}/locked.json", f"{T}/results/curve_forecast/locked.json")
        corpora = {**F1["corpora"], **F2["corpora"]} if a.which == "80" else dict(F2["corpora"])
        M = dict(F1, corpora=corpora, merged_from=chk)
        json.dump(M, open(f"{T}/results/curve_forecast/forecast.json", "w"), indent=1, default=float)
        if a.which == "80":
            for c in F1["corpora"]:
                shutil.copytree(f"{s1}/{c}/refcurve", f"{T}/results/curve_forecast/{c}/refcurve")
                shutil.copytree(f"{R}/results/text_runs/cf_{c}", f"{T}/results/text_runs/cf_{c}")
        for c in F2["corpora"]:
            shutil.copytree(f"{s2}/{c}/refcurve", f"{T}/results/curve_forecast/{c}/refcurve")
            shutil.copytree(f"{s2}/text_runs/cf_{c}", f"{T}/results/text_runs/cf_{c}")
        for t in ("bfine", "cf_fineweb"):
            shutil.copytree(f"{R}/results/text_runs/{t}", f"{T}/results/text_runs/{t}")
        locked = json.load(open(f"{s1}/locked.json"))["code_sha256"]
        for n in ("core.py", "score.py"):
            assert sha(f"{T}/scripts/curve_forecast/{n}") == locked[f"scripts/curve_forecast/{n}"], n
        subprocess.run([sys.executable, f"{T}/scripts/curve_forecast/score.py"], check=True, stdout=subprocess.DEVNULL)
        s = json.load(open(f"{T}/results/curve_forecast/score.json"))
    s["merged_from"] = chk
    json.dump(s, open(out, "w"), indent=1, default=float)
    print("written", os.path.relpath(out, R))


if __name__ == "__main__":
    main()
