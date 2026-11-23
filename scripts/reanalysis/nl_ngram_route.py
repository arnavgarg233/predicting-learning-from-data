"""Follow-up to nl_level_maps.py. (1) n-gram level map fitted on all 80 test corpora (every family) and on the 66
natural-language ones; (2) routing by the frozen ratio c: paper forecast A0 where c < 0.70, n-gram map where c >= 0.70;
(3) the same at 2^23; (4) paired corpus-bootstrap ratios against pilots on the 34 corpora never used in any fit."""
import json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "..", "..", "results", "reanalysis")
rows = json.load(open(os.path.join(OUT, "nl_table.json")))
rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
LN2 = np.log(2)

def fitg(tr, k):
    X = np.array([[1.0, r["g"]] for r in tr]); y = np.array([r[f"L{k}"] for r in tr])
    return np.linalg.lstsq(X, y, rcond=None)[0]
def p2023(r):
    a = np.log(r["L20"] / r["L23"]) / (3 * LN2); return r["L23"] * 2.0 ** (-a * 2)

dev_all = [r for r in rows if r["group"] == "test80"]
dev_nl = [r for r in dev_all if r["family"] != "code"]
held = [r for r in rows if r["group"] != "test80"]
res = {}
for k in (23, 25):
    cg_all, cg_nl = fitg(dev_all, k), fitg(dev_nl, k)
    res[f"coef_{k}"] = dict(all=[float(x) for x in cg_all], nl=[float(x) for x in cg_nl])
    arms = {
        "A0 (paper, frozen)": lambda r: r[f"A0_{k}"],
        "A3 two pilots": lambda r: r[f"A3_{k}"],
        "ngram map, fit all 80": lambda r: cg_all[0] + cg_all[1] * r["g"],
        "ngram map, fit NL 66": lambda r: cg_nl[0] + cg_nl[1] * r["g"],
        "route c: A0 | ngram(all80)": lambda r: r[f"A0_{k}"] if r["c"] < 0.70 else cg_all[0] + cg_all[1] * r["g"],
        "route c: A0 | ngram(NL66)": lambda r: r[f"A0_{k}"] if r["c"] < 0.70 else cg_nl[0] + cg_nl[1] * r["g"],
        "route family: A0 code | ngram(NL66)": lambda r: r[f"A0_{k}"] if r["family"] == "code" else cg_nl[0] + cg_nl[1] * r["g"],
    }
    if k == 25:
        arms["X_pool1 one pilot"] = lambda r: r["Xpool1_25"]
        arms["pilots 2^20+2^23"] = p2023
    tab = {}
    for name, f in arms.items():
        for setname, sub in (("dev80 (in-sample)", dev_all), ("held34", held)):
            e = [f(r) - r[f"L{k}"] for r in sub]
            tab.setdefault(name, {})[setname] = {fam: rms([x for x, r in zip(e, sub) if r["family"] == fam]) for fam in ("code", "english", "wiki")}
            tab[name][setname]["all"] = rms(e)
    res[f"rms_{k}"] = tab
    print(f"\n== r.m.s. at 2^{k} ==")
    for name, v in tab.items():
        print(f"  {name:38s}", " | ".join(f"{s}: " + " ".join(f"{fam[:4]} {x:.3f}" for fam, x in d.items()) for s, d in v.items()))
    # paired bootstrap on held-out natural-language corpora
    rng = np.random.default_rng(0)
    for fam in ("wiki", "english"):
        sub = [r for r in held if r["family"] == fam]
        if not sub: continue
        e_ng = np.array([cg_nl[0] + cg_nl[1] * r["g"] - r[f"L{k}"] for r in sub])
        comps = {"A3": np.array([r[f"A3_{k}"] - r[f"L{k}"] for r in sub]), "A0": np.array([r[f"A0_{k}"] - r[f"L{k}"] for r in sub])}
        if k == 25:
            comps["X_pool1"] = np.array([r["Xpool1_25"] - r["L25"] for r in sub])
            comps["P20_23"] = np.array([p2023(r) - r["L25"] for r in sub])
        for cn, ec in comps.items():
            idx = rng.integers(0, len(sub), size=(20000, len(sub)))
            ratio = np.sqrt((e_ng[idx] ** 2).mean(1)) / np.sqrt((ec[idx] ** 2).mean(1))
            lo, hi = np.percentile(ratio, [2.5, 97.5])
            pt = rms(e_ng) / rms(ec); closer = int((np.abs(e_ng) < np.abs(ec)).sum())
            res.setdefault(f"boot_{k}", {})[f"{fam}: ngram(NL66) / {cn}"] = dict(ratio=pt, lo=float(lo), hi=float(hi), closer=closer, n=len(sub))
            print(f"  2^{k} {fam} ngram/{cn}: {pt:.2f} ({lo:.2f}-{hi:.2f}) closer {closer}/{len(sub)}")
json.dump(res, open(os.path.join(OUT, "nl_ngram_route.json"), "w"), indent=1)
