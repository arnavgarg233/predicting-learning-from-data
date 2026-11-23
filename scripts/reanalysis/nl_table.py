"""Assemble one per-corpus table (inputs available before training + measured losses) for the 80 test, 16 stage-3 and
18 new corpora. Reads stored JSON only."""
import json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "curve_forecast")
OUT = os.path.join(R, "..", "reanalysis", "nl_table.json")

def load(fc, sc, group):
    F = {}
    for f in fc:
        F.update(json.load(open(os.path.join(R, f)))["corpora"])
    P = json.load(open(os.path.join(R, sc)))["per_corpus"]
    rows = []
    for k, p in P.items():
        f = F[k]; st = f["ref_stats"]
        L20 = p["pilots"]["20"]
        rows.append(dict(corpus=k, group=group, family=p["family"], ref_model=p["ref_model"],
                         r=st["ref_full"], g=st["ngram23"], gamma=st["gamma"], A=st["A"], c=st["ref_full"] / st["ngram23"],
                         S23=f["S"]["23"], S25=f["S"]["25"], S20=f["S"]["20"],
                         L19=p["pilots"]["19"], L20=L20, L23=p["measured"]["23"], L25=p["measured"]["25"],
                         A0_25=p["forecast"]["25"]["A0"], A3_25=p["forecast"]["25"]["A3"],
                         A0_23=p["forecast"]["23"]["A0"], A3_23=p["forecast"]["23"]["A3"],
                         Xpool1_25=L20 * 2.0 ** (-0.14 * 5), steps=p["steps"], flops=p["flops"]))
    return rows

rows = load(["forecast.json", "stage2/forecast.json"], "score80.json", "test80")
rows += load(["stage3/forecast.json"], "stage3/score_stage3.json", "stage3")
rows += load(["new_corpora/forecast.json"], "new_corpora/score.json", "new18")
json.dump(rows, open(OUT, "w"), indent=1)
from collections import Counter
print(Counter((r["group"], r["family"]) for r in rows))
