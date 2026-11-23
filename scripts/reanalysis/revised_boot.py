"""Paired corpus-bootstrap ratios (20,000 resamples) of the revised forecaster (revised_forecaster.py) against pilot
methods on the 34 corpora not used in any fit."""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import revised_forecaster as RF  # noqa (re-runs the table; cheap)
rows = [r for r in RF.rows if r["group"] != "test80"]
rng = np.random.default_rng(0); out = {}
comps = {"A3": lambda r: r["A3_25"], "X_pool1": lambda r: r["Xpool1_25"], "P20_23": RF.p2023, "A0": lambda r: r["A0_25"]}
for fam in ("wiki", "english", "code", None):
    s = [r for r in rows if fam is None or r["family"] == fam]
    e = np.array([RF.rev(r) - r["L25"] for r in s])
    for cn, f in comps.items():
        ec = np.array([f(r) - r["L25"] for r in s]); idx = rng.integers(0, len(s), (20000, len(s)))
        rat = np.sqrt((e[idx] ** 2).mean(1)) / np.sqrt((ec[idx] ** 2).mean(1))
        lo, hi = np.percentile(rat, [2.5, 97.5])
        out[f"{fam or 'all'}/{cn}"] = dict(n=len(s), ratio=float(np.sqrt((e**2).mean() / (ec**2).mean())), lo=float(lo), hi=float(hi), closer=int((abs(e) < abs(ec)).sum()))
for k, v in out.items(): print(k, {a: round(b, 2) if isinstance(b, float) else b for a, b in v.items()})
json.dump(out, open(os.path.join(RF.OUT, "revised_boot.json"), "w"), indent=1)
