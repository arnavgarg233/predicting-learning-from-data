"""Variance shares of the DataDecide rate at fixed model size and under per-evaluation-set loss floors.

    python scripts/datadecide_rate_variance_checks.py [--out PATH]   # -> results/datadecide/rate_variance_checks.json

Shares are one-way: the fraction of the variance of the exponents captured by the mean of each evaluation set ("eval")
or of each recipe ("data").
  inrun        in-run exponents of results/datadecide/inrun_rates.csv (checkpoints at 10-40% of one run), per model size
  ladder_floor slope of ln(L - f * min over recipes of the 1B loss on that set) on ln tokens along the 60M-1B ladder of
               final checkpoints (seed means from final_levels.csv), for f = 0, 0.5, 0.7, 0.85, 0.95
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/datadecide/rate_variance_checks.json")


def shares(df, col):
    y = df[col].to_numpy()
    tot = ((y - y.mean()) ** 2).sum()
    out = {}
    for g in ("eval", "data"):
        m = df.groupby(g)[col].transform("mean").to_numpy()
        out[g] = round(float(((m - y.mean()) ** 2).sum() / tot), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    fl = pd.read_csv(os.path.join(ROOT, "results/datadecide/final_levels.csv"))
    ir = pd.read_csv(os.path.join(ROOT, "results/datadecide/inrun_rates.csv"))
    out = {}
    for p, g in ir.groupby("params"):
        out[f"inrun_{p}"] = shares(g, "alpha")
    sizes = ["60M", "90M", "150M", "300M", "530M", "750M", "1B"]
    L = fl[fl["params"].isin(sizes)]
    mins = L[L["params"] == "1B"].groupby("eval")["mean"].min()
    fs = {}
    for f in (0.0, 0.5, 0.7, 0.85, 0.95):
        rows = []
        for (d, e), g in L.groupby(["data", "eval"]):
            g = g.set_index("params").reindex(sizes).dropna()
            if len(g) < 5:
                continue
            x = np.log(g["tokens"].to_numpy())
            y = np.log(g["mean"].to_numpy() - f * mins[e])
            rows.append(dict(data=d, eval=e, a=-np.polyfit(x, y, 1)[0]))
        fs[f] = shares(pd.DataFrame(rows), "a")
    json.dump(dict(inrun=out, ladder_floor=fs), open(a.out, "w"), indent=1)
    print("in-run:", out)
    print("ladder floors:", fs)


if __name__ == "__main__":
    main()
