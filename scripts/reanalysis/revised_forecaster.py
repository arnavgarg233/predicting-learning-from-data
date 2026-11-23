"""A revised data-only forecaster, every constant fitted on the 80 test corpora (or development), scored on the 34
corpora not used in any fit (16 stage-3 Wikipedia + 18 new). Level at 2^23: natural language uses the n-gram-only
level map fitted on the 13 development corpora (nl_dev_ngram_level.json); code uses the locked A0 level minus its
mean bias on the 14 test code corpora. Drop 2^23->2^25: lambda * [S(2^23) - S(2^25)], lambda fitted on the 80 test
corpora (one pooled value). Reported: level error, drop error and 2^25 error separately; pilots for comparison."""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "reanalysis")
rows = json.load(open(os.path.join(OUT, "nl_table.json")))
ng = json.load(open(os.path.join(OUT, "nl_dev_ngram_level.json")))["coef"]["ngram"]
lam = json.load(open(os.path.join(OUT, "shrink_fit.json")))["lambda_test80"]
t80c = [r for r in rows if r["group"] == "test80" and r["family"] == "code"]
code_bias = float(np.mean([r["A0_23"] - r["L23"] for r in t80c]))
rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
LN2 = np.log(2)
def lvl(r): return r["A0_23"] - code_bias if r["family"] == "code" else ng[0] + ng[1] * r["g"]
def rev(r): return lvl(r) - lam * (r["S23"] - r["S25"])
def p2023(r):
    a = np.log(r["L20"] / r["L23"]) / (3 * LN2); return r["L23"] * 2.0 ** (-a * 2)
res = dict(ngram_coef=ng, lam=lam, code_bias=code_bias, tab={})
for grp in ("test80", "stage3", "new18"):
    for fam in ("code", "english", "wiki"):
        s = [r for r in rows if r["group"] == grp and r["family"] == fam]
        if not s: continue
        d = dict(n=len(s),
                 rev_level=rms([lvl(r) - r["L23"] for r in s]),
                 rev_drop=rms([lam * (r["S23"] - r["S25"]) - (r["L23"] - r["L25"]) for r in s]),
                 rev_25=rms([rev(r) - r["L25"] for r in s]), A0_25=rms([r["A0_25"] - r["L25"] for r in s]),
                 A3_25=rms([r["A3_25"] - r["L25"] for r in s]), Xpool1_25=rms([r["Xpool1_25"] - r["L25"] for r in s]),
                 P2023_25=rms([p2023(r) - r["L25"] for r in s]))
        res["tab"][f"{grp}/{fam}"] = d
        print(f"{grp}/{fam:8s}", " ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in d.items()))
held = [r for r in rows if r["group"] != "test80"]
for fam in ("code", "english", "wiki", None):
    s = [r for r in held if fam is None or r["family"] == fam]
    res["tab"][f"held34/{fam or 'all'}"] = dict(n=len(s), rev_25=rms([rev(r) - r["L25"] for r in s]), A0_25=rms([r["A0_25"] - r["L25"] for r in s]),
        A3_25=rms([r["A3_25"] - r["L25"] for r in s]), Xpool1_25=rms([r["Xpool1_25"] - r["L25"] for r in s]), P2023_25=rms([p2023(r) - r["L25"] for r in s]))
    print("held34", fam, res["tab"][f"held34/{fam or 'all'}"])
json.dump(res, open(os.path.join(OUT, "revised_forecaster.json"), "w"), indent=1)
