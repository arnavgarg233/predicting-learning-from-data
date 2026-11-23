"""Can a data-only map, refitted on the 80 test corpora's natural-language rows, reach pilot accuracy on natural
language? Candidates listed here before any held-out error was computed; the choice among them is made by
leave-one-out on the 66 natural-language test corpora only, then scored once on the 29 natural-language corpora never
used in any fit (16 stage-3 Wikipedia + 13 new English/Wikipedia)."""
import json, os
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(H, "..", "..", "results", "reanalysis")
rows = json.load(open(os.path.join(OUT, "nl_table.json")))
NL = lambda r: r["family"] in ("english", "wiki")

def feats(name, r):
    dS = r["S25"] - r["S23"]
    return {"g": [r["g"]], "g+r": [r["g"], r["r"]], "g+r+c": [r["g"], r["r"], r["c"]],
            "g+dS": [r["g"], dS], "g+r+dS": [r["g"], r["r"], dS], "r": [r["r"]],
            "min(r,g)": [min(r["r"], r["g"])], "g+gamma": [r["g"], r["gamma"]]}[name]
CANDS = ["g", "g+r", "g+r+c", "g+dS", "g+r+dS", "r", "min(r,g)", "g+gamma"]

def fit(train, name, tgt="L25"):
    X = np.array([[1.0] + feats(name, r) for r in train]); y = np.array([r[tgt] for r in train])
    return np.linalg.lstsq(X, y, rcond=None)[0]
def pred(coef, name, r):
    return float(coef[0] + np.dot(coef[1:], feats(name, r)))
rms = lambda e: float(np.sqrt(np.mean(np.square(e))))

dev = [r for r in rows if r["group"] == "test80" and NL(r)]
held = [r for r in rows if r["group"] != "test80" and NL(r)]
res = {"n_dev": len(dev), "n_held": len(held), "loo": {}, "held": {}}
for name in CANDS:
    e = []
    for i in range(len(dev)):
        tr = dev[:i] + dev[i + 1:]
        e.append(pred(fit(tr, name), name, dev[i]) - dev[i]["L25"])
    res["loo"][name] = {fam: rms([x for x, r in zip(e, dev) if r["family"] == fam]) for fam in ("english", "wiki")}
    res["loo"][name]["all"] = rms(e)
best = min(CANDS, key=lambda n: res["loo"][n]["all"])
res["chosen_by_loo"] = best
def score(errs, sub):
    return {k: rms([e for e, r in zip(errs, sub) if f(r)]) for k, f in
            [("english", lambda r: r["family"] == "english"), ("wiki", lambda r: r["family"] == "wiki"),
             ("stage3", lambda r: r["group"] == "stage3"), ("new_wiki", lambda r: r["group"] == "new18" and r["family"] == "wiki"),
             ("new_english", lambda r: r["group"] == "new18" and r["family"] == "english"), ("all", lambda r: True)]}
for name in CANDS:
    coef = fit(dev, name)
    res["held"][name] = dict(coef=[float(c) for c in coef], **score([pred(coef, name, r) - r["L25"] for r in held], held))
for arm in ("A0_25", "A3_25", "Xpool1_25"):
    res["held"][arm] = score([r[arm] - r["L25"] for r in held], held)
    res["loo"][arm] = {fam: rms([r[arm] - r["L25"] for r in dev if r["family"] == fam]) for fam in ("english", "wiki")}
# pilots 2^20 + 2^23 (two-point power law, no floor)
def p2023(r):
    a = np.log(r["L20"] / r["L23"]) / (3 * np.log(2)); return r["L23"] * 2.0 ** (-a * 2)
res["held"]["P20_23"] = score([p2023(r) - r["L25"] for r in held], held)
json.dump(res, open(os.path.join(OUT, "nl_level_maps.json"), "w"), indent=1)
print("LOO on 66 NL test corpora (rms 2^25):")
for n, v in res["loo"].items(): print(f"  {n:10s} " + "  ".join(f"{k} {x:.3f}" for k, x in v.items()))
print("chosen:", best)
print("Held-out 29 NL corpora:")
for n, v in res["held"].items(): print(f"  {n:10s} " + "  ".join(f"{k} {x:.3f}" for k, x in v.items() if k != "coef"))
