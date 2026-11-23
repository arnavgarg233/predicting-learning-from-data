"""The n-gram-only level ('ngram|anch', shape 'gamma') was one of the six level candidates in the 13-corpus development
(results/curve_forecast/development.json, level_rms_on_common: 0.228 against 0.165 for the chosen 'both'). Here it is
fitted exactly as development fitted the chosen map (all development corpora, develop.level_rows) and applied unchanged
to every test corpus with each corpus's stored shape S. Nothing is fitted on any test corpus."""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "curve_forecast"))
import develop as DV  # noqa
C = DV.C
OUT = os.path.join(H, "..", "..", "results", "reanalysis")
dev = DV.load_dev(); names = list(dev); base = C.ts_constants(); K = DV.consts_for(dev, names, base)
coefs = {lv: C.fit_level(lv, DV.level_rows(dev, names, "gamma", K, lv, "anch")) for lv in ("ref", "ngram", "both")}
locked = json.load(open(os.path.join(H, "..", "..", "results", "curve_forecast", "locked.json")))["constants"]["level_coef"]
print("refit 'both' =", coefs["both"], " locked =", locked)
rows = json.load(open(os.path.join(OUT, "nl_table.json")))
rms = lambda e: float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")
res = dict(coef=coefs, n_dev=len(names), tab={})
def arm_ng(r, k):
    a, b = coefs["ngram"]; return a + b * r["g"] + r[f"S{k}"] - r["S23"]
for k in (23, 25):
    for name, f in (("A0 locked (both)", lambda r: r[f"A0_{k}"]), ("A0 ngram|anch dev-fitted", lambda r: arm_ng(r, k)),
                    ("A3 two pilots", lambda r: r[f"A3_{k}"])):
        for grp in ("test80", "stage3", "new18"):
            sub = [r for r in rows if r["group"] == grp]
            res["tab"].setdefault(f"2^{k}", {}).setdefault(name, {})[grp] = {
                fam: rms([f(r) - r[f"L{k}"] for r in sub if r["family"] == fam]) for fam in ("code", "english", "wiki")}
for k, t in res["tab"].items():
    print(f"== {k}")
    for n, v in t.items():
        print(f"  {n:26s}", " | ".join(f"{g}: " + " ".join(f"{fam[:4]} {x:.3f}" for fam, x in d.items()) for g, d in v.items()))
json.dump(res, open(os.path.join(OUT, "nl_dev_ngram_level.json"), "w"), indent=1)
