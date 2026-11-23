"""Writes results/datadecide/level_asymptote_test.json."""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datadecide_predict as P  # noqa: E402

ROOT = P.ROOT
SIZES = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M", "750M", "1B"]
NUM = {s: float(s[:-1]) * (1e6 if s.endswith("M") else 1e9) for s in SIZES}
lv = pd.read_csv(os.path.join(ROOT, "results/datadecide/final_levels.csv"))
L = lv.groupby(["data", "params"])["bpb"].mean().unstack()[SIZES]


def fit(n, l):
    best = None
    for E in np.linspace(0, l.min() - 1e-4, 2000):
        x, z = np.log(n), np.log(l - E)
        b, a = np.polyfit(x, z, 1)
        sse = np.sum((np.log(E + np.exp(a + b * x)) - np.log(l)) ** 2)
        if best is None or sse < best[0]:
            best = (sse, E, np.exp(a), -b)
    return dict(E=best[1], A=best[2], a=best[3], sse=best[0])


M = P.load_stats(os.path.join(ROOT, "results/datadecide/stats/merged.json"))
S, X = M["sources"], (M.get("xent") or {}).get("xent")
fam_of = {r: f for f, (_, rs) in P.FAMILIES.items() for r in rs}
names = sorted(fam_of)
fam = np.array([fam_of[r] for r in names])
F, _ = P.recipe_features(S, X, names)
C = json.load(open(os.path.join(ROOT, "results/datadecide/evalaware_rivals_cache.json")))
slug_of = {s_["name"]: k for k, s_ in S.items() if s_.get("kind") == "recipe"}
feat = {"eval-aware bigram (paper)": F.loc[names, "mismatch_all"].values.astype(float),
        "eval-aware word unigram": np.array([np.mean(list(C[slug_of[r]]["word"].values())) for r in names]),
        "compression (lzma 8MB)": F.loc[names, "comp_lzma_8MB"].values.astype(float),
        "n-gram entropy H2": F.loc[names, "ngram_H2"].values.astype(float)}
n = np.array([NUM[s] for s in SIZES])
fits = {r: fit(n, L.loc[r].values) for r in names}
fits7 = {r: fit(n[7:], L.loc[r].values[7:]) for r in names}
rng = np.random.default_rng(1)
u = np.unique(fam)
idx = {c: np.flatnonzero(fam == c) for c in u}
i, j = np.triu_indices(len(names), 1)
draws = []
for _ in range(2000):
    k = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
    ii, jj = np.triu_indices(len(k), 1)
    a, b = k[ii], k[jj]
    keep = a != b
    draws.append((a[keep], b[keep]))


def score(y):
    out = {}
    for name, v in feat.items():
        p = P.lofo(v[:, None], y, fam, [(0,)])[0]
        ag = lambda a, b: np.where(np.sign(p[a] - p[b]) == 0, 0.5, np.sign(p[a] - p[b]) == np.sign(y[a] - y[b]))  # noqa: E731
        boot = [ag(a, b).mean() for a, b in draws]
        out[name] = dict(pairwise=float(ag(i, j).mean()), ci95=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))])
    return out


res = dict(sizes=SIZES, fits={r: fits[r] for r in names}, fits_60M_1B={r: fits7[r] for r in names})
res["E_14sizes"] = score(np.array([fits[r]["E"] for r in names]))
res["E_60M_1B"] = score(np.array([fits7[r]["E"] for r in names]))
res["per_size"] = {s: score(L.loc[names, s].values.astype(float)) for s in SIZES}
res["E_range"] = [float(min(f["E"] for f in fits.values())), float(max(f["E"] for f in fits.values()))]
json.dump(res, open(os.path.join(ROOT, "results/datadecide/level_asymptote_test.json"), "w"), indent=1)
for t in ("E_14sizes", "E_60M_1B"):
    print(t, {k: (round(v["pairwise"], 3), [round(x, 2) for x in v["ci95"]]) for k, v in res[t].items()})
print("per size pairwise:")
for s in SIZES:
    print(" ", s, {k.split(" (")[0]: round(v["pairwise"], 2) for k, v in res["per_size"][s].items()})
print("E range", res["E_range"])
