"""Write the seeded 1,000-target subset of each Qwen2.5-0.5B per-target loss file, so score.py's sub05 arm runs without
the full 20,000-target files (322 MB).

    python3 scripts/stronger_ref/extract_sub05.py      # -> <refdir>/Qwen2.5-0.5B_sub1000.npz for each job
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, "results/curve_forecast/stronger_ref")

for j in json.load(open(os.path.join(OUT, "jobs.json")))["jobs"]:
    rdir = os.path.join(ROOT, j["refdir"])
    raw = np.load(os.path.join(rdir, "Qwen2.5-0.5B_raw.npz"))
    nt = len(raw["tgt"])
    idx = np.sort(np.random.default_rng(20261025).choice(nt, min(1000, nt), replace=False))
    np.savez_compressed(os.path.join(rdir, "Qwen2.5-0.5B_sub1000.npz"), nt=nt, idx=idx, loss=raw["loss"][idx],
                        doc=raw["doc"][idx], tgtbytes=raw["tgtbytes"][idx], ctxb=raw["ctxb"][idx], NS=raw["NS"])
