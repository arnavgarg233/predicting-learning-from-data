"""Level error at 2^23 and drop error 2^23->2^25, reported separately, for the data forecast and pilot methods, on
the 80 test, 16 stage-3 and 18 new corpora. Plus a shape shrink lambda (drop = lambda * [S(2^25) - S(2^23)]) fitted
on the 13 development corpora only: least squares of measured L(k) - L(2^23) against S(k) - S(2^23) for every
development budget k != 23 (2^21, 2^22; 2^24-2^25 where run). Level is untouched by lambda."""
import json, os, sys
import numpy as np
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "curve_forecast"))
import develop as DV  # noqa
C = DV.C
OUT = os.path.join(H, "..", "..", "results", "reanalysis")
LN2 = np.log(2)
dev = DV.load_dev(); names = list(dev); K = DV.consts_for(dev, names, C.ts_constants())
xs, ys, per = [], [], {}
for c in names:
    d = dev[c]; nu, n0 = C.shape_params("gamma", d["st"], K)
    for k, L in d["L"].items():
        if k == 23 or k < 21: continue
        xs.append(C.S(d["hr"], nu, n0, k) - C.S(d["hr"], nu, n0, 23)); ys.append(L - d["L"][23])
        per.setdefault(c, []).append((k, xs[-1], ys[-1]))
xs, ys = np.array(xs), np.array(ys)
lam = float(np.dot(xs, ys) / np.dot(xs, xs))
# forward-only variant: pairs with k > 23 only (FineWeb-Edu, TinyStories)
fw = [(x, y) for c in per for k, x, y in per[c] if k > 23]
lam_fwd = float(sum(x * y for x, y in fw) / sum(x * x for x, _ in fw)) if fw else float("nan")
rows = json.load(open(os.path.join(OUT, "nl_table.json")))
rms = lambda e: float(np.sqrt(np.mean(np.square(e)))) if len(e) else float("nan")
mean = lambda e: float(np.mean(e)) if len(e) else float("nan")
def p2023(r):
    a = np.log(r["L20"] / r["L23"]) / (3 * LN2); return r["L23"] * 2.0 ** (-a * 2)
arms = {  # (level forecast at 2^23, forecast drop 2^23->2^25)
    "A0 data alone": lambda r: (r["A0_23"], r["A0_23"] - r["A0_25"]),
    f"A0, drop x lambda={lam:.2f} (dev)": lambda r: (r["A0_23"], lam * (r["S23"] - r["S25"])),
    "A3 two pilots": lambda r: (r["A3_23"], r["A3_23"] - r["A3_25"]),
    "X_pool1 one pilot": lambda r: (r["L20"] * 2 ** (-0.14 * 3), r["L20"] * (2 ** (-0.42) - 2 ** (-0.70))),
    "pilots 2^20+2^23": lambda r: (r["L23"], r["L23"] - p2023(r)),
}
res = dict(lambda_dev=lam, lambda_dev_forward_only=lam_fwd, n_dev_pairs=len(xs), n_fwd_pairs=len(fw), tab={})
print(f"lambda (13 dev corpora, {len(xs)} pairs) = {lam:.3f};  forward-only ({len(fw)} pairs) = {lam_fwd:.3f}")
for grp in ("test80", "new18", "stage3"):
    for fam in ("code", "english", "wiki"):
        sub = [r for r in rows if r["group"] == grp and r["family"] == fam]
        if not sub: continue
        for name, f in arms.items():
            lv = [f(r)[0] - r["L23"] for r in sub]; dr = [f(r)[1] - (r["L23"] - r["L25"]) for r in sub]
            e25 = [a - b for a, b in zip(lv, dr)]
            res["tab"].setdefault(f"{grp}/{fam} ({len(sub)})", {})[name] = dict(
                level_rms=rms(lv), level_bias=mean(lv), drop_rms=rms(dr), drop_bias=mean(dr), err25_rms=rms(e25),
                measured_drop=mean([r["L23"] - r["L25"] for r in sub]), forecast_drop=mean([f(r)[1] for r in sub]))
for g, t in res["tab"].items():
    print("==", g)
    for n, v in t.items():
        print(f"  {n:30s} level rms {v['level_rms']:.3f} bias {v['level_bias']:+.3f} | drop rms {v['drop_rms']:.3f} bias {v['drop_bias']:+.3f}"
              f" (meas {v['measured_drop']:.3f}, fc {v['forecast_drop']:.3f}) | 2^25 rms {v['err25_rms']:.3f}")
json.dump(res, open(os.path.join(OUT, "level_drop_split.json"), "w"), indent=1)
