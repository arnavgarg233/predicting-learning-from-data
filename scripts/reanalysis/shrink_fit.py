"""What a drop shrink fitted on the 80 test corpora (after their errors were known, for illustration) does to 2^25 error on the 34 corpora
not used in the fit. Level untouched."""
import json, os
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "reanalysis")
rows = json.load(open(os.path.join(OUT, "nl_table.json")))
rms = lambda e: float(np.sqrt(np.mean(np.square(e))))
t80 = [r for r in rows if r["group"] == "test80"]
x = np.array([r["S23"] - r["S25"] for r in t80]); y = np.array([r["L23"] - r["L25"] for r in t80])
lam = float(x @ y / (x @ x)); res = {"lambda_test80": lam, "by_family_lambda": {}, "held": {}}
for fam in ("code", "english", "wiki"):
    s = [r for r in t80 if r["family"] == fam]; xx = np.array([r["S23"] - r["S25"] for r in s]); yy = np.array([r["L23"] - r["L25"] for r in s])
    res["by_family_lambda"][fam] = float(xx @ yy / (xx @ xx))
for grp in ("test80", "new18", "stage3"):
    for fam in ("code", "english", "wiki"):
        s = [r for r in rows if r["group"] == grp and r["family"] == fam]
        if not s: continue
        e0 = [r["A0_25"] - r["L25"] for r in s]
        e1 = [r["A0_23"] - lam * (r["S23"] - r["S25"]) - r["L25"] for r in s]
        res["held"][f"{grp}/{fam}"] = dict(n=len(s), A0=rms(e0), A0_shrunk=rms(e1))
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(OUT, "shrink_fit.json"), "w"), indent=1)
