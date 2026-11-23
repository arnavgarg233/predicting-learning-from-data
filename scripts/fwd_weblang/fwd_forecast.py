"""Forward test (docs/analysis/fwd_weblang_2026-10-31.md): write every corpus's frozen fall forecast with a date, from the frozen
calibration and the pod's forecasts (i) and (ii), before any learner run of the corpus. Run by the Mac guard before it releases
training; refuses to overwrite.

    python scripts/fwd_weblang/fwd_forecast.py FROZEN_JSON FORECAST_I_JSON FORECAST_II_JSON OUT_JSON
"""
import datetime
import json
import os
import sys


def main():
    fz, fi, fii, out = sys.argv[1:5]
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; not overwriting")
    Z = json.load(open(fz))["calibration"]
    F = json.load(open(fi))["corpora"]
    G = json.load(open(fii))["corpora"]
    a, b, m = Z["a"], Z["b"], Z["kind_mean_fall"]
    rows = {}
    for c, fc in F.items():
        assert fc["family"] == "lang", c
        x = fc["S"]["23"] - fc["S"]["25"]
        V = G[c]["level_23"]
        cal = a + b * x
        rows[c] = dict(level_23=V, S23=fc["S"]["23"], S25=fc["S"]["25"], own_fall=x, calibrated_fall=cal, kind_mean_fall=m,
                       pred_L25=V - cal, pred_L25_kind_mean=V - m)
    now = datetime.datetime.now().astimezone()
    json.dump(dict(date=now.strftime("%Y-%m-%d %H:%M:%S %Z"), date_epoch=int(now.timestamp()), frozen=os.path.abspath(fz),
                   calibration=dict(a=a, b=b, kind_mean_fall=m), forecast_i_date=json.load(open(fi)).get("date"),
                   corpora=rows), open(out, "w"), indent=1)
    print(len(rows), "corpus forecasts ->", out, now.strftime("%H:%M:%S %Z"))


if __name__ == "__main__":
    main()
