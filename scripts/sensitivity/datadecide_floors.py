"""sensitivity (2026-11-05), part A: DataDecide 60M-1B ladder exponents under floor choices; variance shares by
evaluation set and recipe, and the gamma-rate Spearman over the 11 sets. Reads results/datadecide/final_levels.csv,
ladder_rates.csv and evalset_rate_test.json. Stored data only.

Floors: none; a fraction f of the lowest 1B loss on each set (as in the article); a free floor per cell (recipe x set,
3-parameter fit on the seven sizes, E in [0, 0.999 min L]); a free floor shared by all recipes on a set (one E per set,
fitted jointly with a slope and amplitude per recipe).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DD = ROOT / "results/datadecide"
OUT = ROOT / "results/sensitivity/datadecide_floors.json"
LARGE = ["60M", "90M", "150M", "300M", "530M", "750M", "1B"]

d = pd.read_csv(DD / "final_levels.csv")
d = d[d["params"].isin(LARGE)]
cells = {}
for (rec, ev), g in d.groupby(["data", "eval"]):
    g = g.set_index("params").loc[LARGE]
    cells[(rec, ev)] = (np.log(g["tokens"].to_numpy(float)), g["mean"].to_numpy(float))


def slope(x, L, E):
    return float(-np.polyfit(x, np.log(L - E), 1)[0])


def sse(x, L, E):
    p = np.polyfit(x, np.log(L - E), 1)
    return float(np.sum((np.log(L - E) - np.polyval(p, x)) ** 2))


def shares(a):
    df = pd.DataFrame([dict(data=r, eval=e, a=v) for (r, e), v in a.items()])
    y = df["a"].to_numpy(); tot = np.sum((y - y.mean()) ** 2)
    sh = {}
    for col in ("eval", "data"):
        m = df.groupby(col)["a"].transform("mean").to_numpy()
        sh[col] = float(np.sum((m - y.mean()) ** 2) / tot)
    return sh, df


ref = pd.read_csv(DD / "ladder_rates.csv").set_index(["data", "eval"])["alpha_large"]
res = {}
raw = {k: slope(x, L, 0.0) for k, (x, L) in cells.items()}
res["check_raw_vs_stored_max_abs"] = float(max(abs(raw[k] - ref.loc[k]) for k in raw if k in ref.index))
variants = {"none": raw}
min1b = {ev: min(L[-1] for (r, e), (x, L) in cells.items() if e == ev) for ev in d["eval"].unique()}
for f in (0.5, 0.7, 0.85, 0.95):
    variants[f"frac {f}"] = {k: slope(x, L, f * min1b[k[1]]) for k, (x, L) in cells.items()}
grid = np.linspace(0, 0.999, 400)
free, freeE = {}, {}
for k, (x, L) in cells.items():
    Es = grid * L.min()
    s = [sse(x, L, E) for E in Es]
    freeE[k] = float(Es[int(np.argmin(s))]); free[k] = slope(x, L, freeE[k])
variants["free per cell"] = free
shared, sharedE = {}, {}
for ev in d["eval"].unique():
    ks = [k for k in cells if k[1] == ev]
    m = min(cells[k][1].min() for k in ks)
    Es = grid * m
    s = [sum(sse(*cells[k], E) for k in ks) for E in Es]
    sharedE[ev] = float(Es[int(np.argmin(s))])
    for k in ks:
        shared[k] = slope(*cells[k], sharedE[ev])
variants["free per set"] = shared

ev_rate = json.load(open(DD / "evalset_rate_test.json"))
out = dict(n_cells=len(cells), variants={})
for name, a in variants.items():
    sh, df = shares(a)
    out["variants"][name] = dict(share_eval=sh["eval"], share_recipe=sh["data"], median=float(np.median(list(a.values()))),
                                 range=[float(min(a.values())), float(max(a.values()))],
                                 spearman_with_raw=float(pd.Series([a[k] for k in raw]).rank().corr(pd.Series([raw[k] for k in raw]).rank())),
                                 mean_by_eval={e: float(v) for e, v in df.groupby("eval")["a"].mean().items()})
out["free_per_cell_floor_at_grid_top"] = int(sum(freeE[k] >= 0.99 * cells[k][1].min() for k in cells))
out["free_per_cell_floor_zero"] = int(sum(freeE[k] == 0 for k in cells))
out["free_per_set_floor"] = sharedE
out["min1B"] = min1b
out.update(res)
OUT.write_text(json.dumps(out, indent=1))
for n, v in out["variants"].items():
    print(f"{n:15s} eval {v['share_eval']:.3f} recipe {v['share_recipe']:.3f} median {v['median']:.3f} rho_vs_raw {v['spearman_with_raw']:.2f}")
print("check", res, "free cell at top", out["free_per_cell_floor_at_grid_top"], "zero", out["free_per_cell_floor_zero"])

# gamma (GPT-2, floor at the longest context, as in the paper) against the set-mean rate under each floor
evs = ev_rate["evals"]; gam = np.array(ev_rate["models"]["gpt2"]["gamma"])
rng = np.random.default_rng(0)
for name, v in out["variants"].items():
    r = np.array([v["mean_by_eval"][e] for e in evs])
    rho = float(pd.Series(gam).rank().corr(pd.Series(r).rank()))
    bs = []
    for _ in range(4000):
        i = rng.integers(0, len(evs), len(evs))
        if len(set(i)) < 3:
            continue
        bs.append(pd.Series(gam[i]).rank().corr(pd.Series(r[i]).rank()))
    v["spearman_gamma_gpt2"] = rho
    v["spearman_gamma_gpt2_evalset_boot95"] = [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]
    print(f"{name:15s} rho(gamma, set-mean rate) {rho:+.2f} {v['spearman_gamma_gpt2_evalset_boot95']}")
out["gamma_floor_variants_stored"] = json.load(open(DD / "gamma_estimator_check.json"))["results"]
OUT.write_text(json.dumps(out, indent=1))
