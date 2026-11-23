"""Follow-up diagnostics for rate_law_fixed_size.py: (1) only corpora whose reference is competent (c < 0.70);
(2) the law's exponent gamma/(1+gamma) against a floored exponent from a 3-parameter fit L = E + B D^-a to the four
budgets (E >= 0, E < min L), and against slopes above per-corpus floors E = k * ref_full and E = k * ngram23."""
import json, os
import numpy as np
from scipy import stats, optimize
H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "..", "..", "results", "reanalysis")
rows = json.load(open(os.path.join(OUT, "nl_table.json")))
LN2 = np.log(2)
def fit3(r):
    k = np.array([19, 20, 23, 25.0]); L = np.array([r["L19"], r["L20"], r["L23"], r["L25"]])
    best = None
    for E in np.linspace(0, 0.98 * L.min(), 99):
        y = np.log(L - E); A = np.vstack([np.ones(4), -k * LN2]).T
        c, res, *_ = np.linalg.lstsq(A, y, rcond=None); sse = float(np.sum((A @ c - y) ** 2))
        if best is None or sse < best[0]: best = (sse, E, c[1])
    return best[1], best[2]
def sl(r, E): return np.log((r["L20"] - E) / (r["L25"] - E)) / (5 * LN2)
out = {}
for setname, sel in (("all", lambda r: True), ("c<0.70", lambda r: r["c"] < 0.70), ("c<0.70 natural", lambda r: r["c"] < 0.70 and r["family"] != "code"),
                     ("code", lambda r: r["family"] == "code"), ("english", lambda r: r["family"] == "english"), ("wiki", lambda r: r["family"] == "wiki")):
    sub = [r for r in rows if sel(r)]
    law = np.array([r["gamma"] / (1 + r["gamma"]) for r in sub])
    E3 = [fit3(r) for r in sub]
    a3 = np.array([a for _, a in E3]); atb = np.mean([E < 1e-9 for E, _ in E3]); top = np.mean([E > 0.97 * min(r["L19"], r["L20"], r["L23"], r["L25"]) for (E, _), r in zip(E3, sub)])
    d = {"n": len(sub), "fit3 a": float(stats.spearmanr(law, a3)[0]), "fit3 E at 0": float(atb), "fit3 E at top": float(top)}
    for k in (0.5, 0.8):
        d[f"floor {k}*ref_full"] = float(stats.spearmanr(law, [sl(r, min(k * r["r"], 0.95 * r["L25"])) for r in sub])[0])
        d[f"floor {k}*ngram23"] = float(stats.spearmanr(law, [sl(r, min(k * r["g"], 0.95 * r["L25"])) for r in sub])[0])
    d["raw 20-25"] = float(stats.spearmanr(law, [sl(r, 0) for r in sub])[0])
    out[setname] = d
    print(setname, {a: (round(b, 2) if isinstance(b, float) else b) for a, b in d.items()})
json.dump(out, open(os.path.join(OUT, "rate_law_fixed_size_b.json"), "w"), indent=1)
