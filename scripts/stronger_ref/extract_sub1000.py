"""Write the seeded 1,000-target subset of reference-model per-target loss files (<refdir>/<model>_raw.npz, 20,000 targets,
about 5 MB each, not shipped) as <refdir>/<model>_sub1000.npz, in the format of extract_sub05.py. The reanalyses that score
the forecast on 1,000 targets (reanalysis/ref_subset_cost.py, compute_frontier.py, stronger_ref/score_half.py) read
the subset file when the full file is absent; with the full file they draw the same subset.

    python3 scripts/stronger_ref/extract_sub1000.py RAW_NPZ [RAW_NPZ ...] [--root DIR]
"""
import os
import sys

import numpy as np

args = sys.argv[1:]
root = "."
if "--root" in args:
    i = args.index("--root"); root = args[i + 1]; del args[i:i + 2]
for p in args:
    raw = np.load(p)
    nt = len(raw["tgt"])
    idx = np.sort(np.random.default_rng(20261025).choice(nt, min(1000, nt), replace=False))
    out = os.path.join(root, p[:-len("_raw.npz")] + "_sub1000.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, nt=nt, idx=idx, loss=raw["loss"][idx], doc=raw["doc"][idx], tgtbytes=raw["tgtbytes"][idx],
                        ctxb=raw["ctxb"][idx], NS=raw["NS"])
