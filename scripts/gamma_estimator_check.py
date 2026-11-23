"""Spearman of each evaluation set's gamma with its mean ladder rate, for the floor at the longest context (the paper's
estimator, tok_stats.gamma_stats over 4-64 tokens) and for a freely fitted floor (fit_paired, h_inf + a n^-g over >= 8 bytes).

    python3 scripts/gamma_estimator_check.py      # -> results/datadecide/gamma_estimator_check.json
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DD = ROOT / "results" / "datadecide"
S = json.loads((DD / "stats" / "merged.json").read_text())["sources"]
lad = pd.read_csv(DD / "ladder_rates.csv")
E = sorted(lad["eval"].unique())
y = lad.groupby("eval")["alpha_large"].mean().reindex(E).to_numpy()
out = {}
for m in ("gpt2", "pythia-160m", "SmolLM2-135M"):
    a = [S[f"ev_{e}"]["ref"][m]["gamma"]["4-64"]["gamma"] for e in E]
    b = [S[f"ev_{e}"]["ref"][m]["fit_paired"]["g"] for e in E]
    out[m] = dict(floor_longest=spearmanr(a, y)[0], free_floor=spearmanr(b, y)[0], between=spearmanr(a, b)[0])
note = ("primary '4-64' = slope of log(h(n)-h(longest)) over 4-64 tokens (tok_stats.gamma_stats); "
        "free = fit_paired h_inf + a n^-g over >=8 bytes")
(DD / "gamma_estimator_check.json").write_text(json.dumps(dict(note=note, results=out), indent=1))
print(json.dumps(out))
