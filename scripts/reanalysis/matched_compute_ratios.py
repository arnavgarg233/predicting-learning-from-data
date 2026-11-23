"""Corpus-bootstrap ratios of r.m.s. error at 2^25 for the cheap (1,000-target) data-only forecast against the
extrapolation baselines, on code and English (SmolLM2-referenced corpora).

    python scripts/reanalysis/matched_compute_ratios.py  -> results/reanalysis/matched_compute_ratios.json
Reads ref_subset_cost.json and stronger_baselines.json (run those first).
"""
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[2] / "results/reanalysis"
sub = json.load(open(R / "ref_subset_cost.json"))["per_corpus"]
sb = json.load(open(R / "stronger_baselines.json"))["per_corpus"]
out = {}
rng = np.random.default_rng(0)
for fam in ("code", "english"):
    cs = sorted(c for c in sub if sub[c]["family"] == fam)
    e0 = np.array([sub[c]["err_sub"] for c in cs])
    idx = rng.integers(0, len(cs), (20000, len(cs)))
    out[fam] = {}
    for arm in ("A3", "X_pool1", "F_ref", "A3_20_23", "X_pool23"):
        e = np.array([sb[c]["err"][arm] for c in cs])
        r = np.sqrt((e0[idx] ** 2).mean(1) / (e[idx] ** 2).mean(1))
        out[fam][f"A0_sub1000/{arm}"] = dict(ratio=float(np.sqrt((e0 ** 2).mean() / (e ** 2).mean())),
                                             ci95=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                                             closer=f"{int(np.sum(np.abs(e0) < np.abs(e)))}/{len(cs)}")
json.dump(out, open(R / "matched_compute_ratios.json", "w"), indent=1)
print(json.dumps(out, indent=1))
