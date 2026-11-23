"""English forward test: write every corpus's frozen fall forecast with a
date, from results/fwd_english/fall_line.json and the pod's forecasts (i) and (ii), before any learner run of the corpus.
Run by the Mac guard before it releases training; refuses to overwrite.

    python scripts/codeeng/eng_forecast.py POD_OUTDIR     # reads POD_OUTDIR/forecast_i.json, forecast_ii.json -> eng_forecast.json
"""
import datetime
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    D = sys.argv[1]
    out = os.path.join(D, "eng_forecast.json")
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; not overwriting")
    Z = json.load(open(os.path.join(ROOT, "results", "fwd_english", "fall_line.json")))["calibration"]
    F = json.load(open(os.path.join(D, "forecast_i.json")))
    G = json.load(open(os.path.join(D, "forecast_ii.json")))["corpora"]
    a, b, m = Z["line"]["a"], Z["line"]["b"], Z["kind_mean"]
    c2 = Z["context_fall_rule_only"]
    rows = {}
    for c, fc in F["corpora"].items():
        x = fc["S"]["23"] - fc["S"]["25"]
        V = G[c]["level_23"]
        rows[c] = dict(level_23=V, S23=fc["S"]["23"], S25=fc["S"]["25"], own_fall=x, calibrated_fall=a + b * x, kind_mean_fall=m,
                       pred_L25=V - (a + b * x), pred_L25_kind_mean=V - m,
                       calibrated_fall_fall_rule_only=c2["line"]["a"] + c2["line"]["b"] * x,
                       own_fall_outlier=bool(abs(x - Z["median"]["x"]) > Z["k_mad"] * Z["scaled_mad"]["x"]))
    now = datetime.datetime.now().astimezone()
    json.dump(dict(date=now.strftime("%Y-%m-%d %H:%M:%S %Z"), date_epoch=int(now.timestamp()), forecast_i_date=F.get("date"),
                   forecast_i_created_utc=F.get("created_utc"), calibration=Z, corpora=rows), open(out, "w"), indent=1)
    print(len(rows), "corpus forecasts ->", out, now.strftime("%H:%M:%S %Z"))


if __name__ == "__main__":
    main()
