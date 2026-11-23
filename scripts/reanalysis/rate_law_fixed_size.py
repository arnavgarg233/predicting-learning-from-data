"""The 114 corpora (80 test, 16 stage-3, 18 new) were all trained with one learner (4-layer byte transformer, 3.3M
parameters), one optimisation recipe and the same budgets (2^19, 2^20, 2^23, 2^25 bytes). This is a fixed-model-size
data-scaling study across diverse corpora. Does the reference curve's gamma order the measured data-scaling rates?
Rate definitions (all reported): raw log-slope 2^20->2^25, 2^23->2^25, 2^19->2^20, and log-slope above a floor
E = f * L(2^25) for f in 0.5, 0.7, 0.9. Partial correlations hold log L(2^25) (level) fixed. Corpus bootstrap CIs."""
import json, os
import numpy as np
from scipy import stats
H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "..", "..", "results", "reanalysis")
rows = json.load(open(os.path.join(OUT, "nl_table.json")))
LN2 = np.log(2)
def slope(L1, L2, dk, E=0.0): return np.log((L1 - E) / (L2 - E)) / (dk * LN2)
def rates(r):
    out = {"raw 20-25": slope(r["L20"], r["L25"], 5), "raw 23-25": slope(r["L23"], r["L25"], 2), "raw 19-20": slope(r["L19"], r["L20"], 1)}
    for f in (0.5, 0.7, 0.9):
        out[f"floor {f} 20-25"] = slope(r["L20"], r["L25"], 5, f * r["L25"])
    return out
def partial(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return float(np.corrcoef(ex, ey)[0, 1])
rng = np.random.default_rng(0)
res = {}
for setname, sub in (("all 114", rows), ("code 19", [r for r in rows if r["family"] == "code"]),
                     ("english 24", [r for r in rows if r["family"] == "english"]), ("wiki 71", [r for r in rows if r["family"] == "wiki"]),
                     ("natural 95", [r for r in rows if r["family"] != "code"])):
    g = np.array([r["gamma"] for r in sub]); lev = np.log([r["L25"] for r in sub])
    nu_drop = np.array([r["S20"] - r["S25"] for r in sub])
    R = {k: np.array([rates(r)[k] for r in sub]) for k in rates(sub[0])}
    res[setname] = {}
    for k, y in R.items():
        rho = stats.spearmanr(g, y)[0]
        bs = []
        for _ in range(4000):
            i = rng.integers(0, len(sub), len(sub)); bs.append(stats.spearmanr(g[i], y[i])[0])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        res[setname][k] = dict(spearman_gamma=float(rho), ci=[float(lo), float(hi)], partial_level=partial(g, y, lev),
                               spearman_level=float(stats.spearmanr(lev, y)[0]))
    # predicted (information-shape) drop vs measured drop
    md = np.array([r["L20"] - r["L25"] for r in sub])
    res[setname]["S-drop vs measured drop 20-25"] = dict(spearman=float(stats.spearmanr(nu_drop, md)[0]))
    print(f"== {setname} (n={len(sub)})")
    for k, v in res[setname].items(): print("  ", k, {a: (round(b, 2) if isinstance(b, float) else [round(x, 2) for x in b]) for a, b in v.items()})
json.dump(res, open(os.path.join(OUT, "rate_law_fixed_size.json"), "w"), indent=1)
