"""Within a kind of source, does a per-kind affine calibration of the corpus's own information-shape fall forecast the
held-out fall better than the kind's mean fall from the same calibration corpora?
Design: docs/analysis/shape_within_kind_2026-10-31.md. Reads the per-corpus rows of shape_vs_pooled.json (run
scripts/referee_reanalysis/shape_vs_pooled.py first). No training. Deterministic.

    python scripts/referee_reanalysis/shape_within_kind.py    # -> results/referee_reanalysis/shape_within_kind.json
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "results", "referee_reanalysis", "shape_vs_pooled.json")
OUT = os.path.join(ROOT, "results", "referee_reanalysis", "shape_within_kind.json")
NB = 20000
NSPLIT = 200
NOISE_FALL = 0.0126 * np.sqrt(1.5)
CTX = ("A3", "pool_exp", "fw_info")

rows = json.load(open(SRC))["per_corpus"]


def kind(r):
    if r["family"] == "lang":
        return "web" if r["set"] == "external45" else "wiki"
    return r["family"]


KINDS = ("wiki", "web", "english", "code")
for r in rows:
    r["kind"] = kind(r)
    r["x"] = r["S23"] - r["S25"]


def rms(e):
    return float(np.sqrt(np.mean(np.square(e))))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def calib(cal):
    x = np.array([r["x"] for r in cal]); y = np.array([r["fall"] for r in cal])
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b), float(y.mean())


def predict(cal, te):
    """held-out predictions: own calibrated and kind mean, both from `cal`."""
    a, b, m = calib(cal)
    return [(a + b * r["x"], m) for r in te]


def score(te, preds):
    n = len(te)
    fall = np.array([r["fall"] for r in te])
    own = np.array([p[0] for p in preds]); mean = np.array([p[1] for p in preds])
    V = np.array([r["lev_primary"] for r in te]); L23 = np.array([r["L23"] for r in te]); L25 = np.array([r["L25"] for r in te])
    E = {"own_cal": own - fall, "kind_mean": mean - fall}
    for q in CTX:
        E[q] = np.array([r["fall_err"]["primary"][q] for r in te])
    E25 = {"own_cal": V - own - L25, "kind_mean": V - mean - L25}
    for q in CTX:
        E25[q] = np.array([r["e25"]["primary"][q] for r in te])
    E25o = {"own_cal": L23 - own - L25, "kind_mean": L23 - mean - L25}
    idx = np.random.default_rng(0).integers(0, n, (NB, n))
    br = np.sqrt((E["own_cal"][idx] ** 2).mean(1) / (E["kind_mean"][idx] ** 2).mean(1))
    lo, hi = np.percentile(br, [2.5, 97.5])
    sd = float(fall.std(ddof=1))
    return dict(n=n, fall_mean=float(fall.mean()), fall_sd=sd, best_attainable_ratio=float(NOISE_FALL / sd),
                fall_rms={k: rms(v) for k, v in E.items()}, fall_bias={k: float(v.mean()) for k, v in E.items()},
                e25_rms_own_level={k: rms(v) for k, v in E25.items()},
                e25_rms_observed_L23={k: rms(v) for k, v in E25o.items()},
                ratio=rms(E["own_cal"]) / rms(E["kind_mean"]), ci95=[float(lo), float(hi)],
                label="REAL positive" if hi < 1 else ("REAL negative" if lo > 1 else "tie"),
                closer=f"{int(np.sum(np.abs(E['own_cal']) < np.abs(E['kind_mean'])))}/{n}",
                held_out=[r["corpus"] for r in te])


def loo(sub):
    preds = [predict(sub[:i] + sub[i + 1:], [r])[0] for i, r in enumerate(sub)]
    return score(sub, preds)


def splits(sub):
    n = len(sub); fall = np.array([r["fall"] for r in sub]); out = []
    for s in range(NSPLIT):
        perm = np.random.default_rng(s).permutation(n)
        h = [perm[: n // 2], perm[n // 2:]]
        own = np.empty(n); mean = np.empty(n)
        for k in (0, 1):
            p = predict([sub[i] for i in h[1 - k]], [sub[i] for i in h[k]])
            own[h[k]] = [q[0] for q in p]; mean[h[k]] = [q[1] for q in p]
        out.append(rms(own - fall) / rms(mean - fall))
    out = np.array(out)
    return dict(n_splits=NSPLIT, median=float(np.median(out)), p2_5=float(np.percentile(out, 2.5)),
                p97_5=float(np.percentile(out, 97.5)), share_below_1=float(np.mean(out < 1)))


def design_a(sub):
    cal = [r for r in sub if r["set"] == "test80"]
    te = [r for r in sub if r["set"] in ("forward18", "stage3")]
    a, b, m = calib(cal)
    return dict(calibration=dict(a=a, b=b, mean_fall=m, n=len(cal)), **score(te, predict(cal, te)))


def drop_extremes(sub, k=3):
    med = np.median([r["fall"] for r in sub])
    order = sorted(sub, key=lambda r: (-abs(r["fall"] - med), r["corpus"]))
    gone = {r["corpus"] for r in order[:k]}
    return [r for r in sub if r["corpus"] not in gone], sorted(gone)


res = dict(design="docs/analysis/shape_within_kind_2026-10-31.md", source=os.path.relpath(SRC, ROOT),
           noise_fall=float(NOISE_FALL), kinds={})
for kd in KINDS:
    sub = sorted([r for r in rows if r["kind"] == kd], key=lambda r: (r["set"], r["corpus"]))
    a, b, m = calib(sub)
    x = np.array([r["x"] for r in sub]); y = np.array([r["fall"] for r in sub])
    K = dict(n=len(sub), sets={s: sum(r["set"] == s for r in sub) for s in sorted({r["set"] for r in sub})},
             full_fit=dict(a=a, b=b, mean_fall=m), spearman=spearman(x, y), pearson=float(np.corrcoef(x, y)[0, 1]),
             loo=loo(sub), split2=splits(sub))
    if kd == "wiki":
        K["A"] = design_a(sub)
    trim, gone = drop_extremes(sub)
    K["sensitivity_drop3"] = dict(dropped=gone, loo=loo(trim))
    if kd == "wiki":
        K["sensitivity_drop3"]["A"] = design_a(trim)
    K["primary"] = "A" if kd == "wiki" else "loo"
    K["verdict"] = K[K["primary"]]["label"]
    res["kinds"][kd] = K
json.dump(res, open(OUT, "w"), indent=1)


def line(name, s):
    f = s["fall_rms"]
    print(f"  {name:10s} n={s['n']:2d} sd {s['fall_sd']:.3f} best {s['best_attainable_ratio']:.2f} | own {f['own_cal']:.3f} "
          f"mean {f['kind_mean']:.3f} A3 {f['A3']:.3f} poolexp {f['pool_exp']:.3f} fwinfo {f['fw_info']:.3f} | ratio "
          f"{s['ratio']:.2f} ({s['ci95'][0]:.2f}-{s['ci95'][1]:.2f}) {s['closer']} {s['label']} | e25 own-level "
          f"{s['e25_rms_own_level']['own_cal']:.3f}/{s['e25_rms_own_level']['kind_mean']:.3f}")


for kd, K in res["kinds"].items():
    print(kd, K["sets"], "fit a=%.3f b=%.3f" % (K["full_fit"]["a"], K["full_fit"]["b"]), "rho %.2f r %.2f" % (K["spearman"], K["pearson"]))
    line("LOO", K["loo"])
    if "A" in K:
        line("A", K["A"])
    s2 = K["split2"]
    print(f"  split2 median {s2['median']:.2f} ({s2['p2_5']:.2f}-{s2['p97_5']:.2f}) below1 {s2['share_below_1']:.2f}")
    line("drop3 LOO", K["sensitivity_drop3"]["loo"])
    if "A" in K["sensitivity_drop3"]:
        line("drop3 A", K["sensitivity_drop3"]["A"])
    print("  dropped", K["sensitivity_drop3"]["dropped"], "VERDICT", K["verdict"])
print("->", os.path.relpath(OUT, ROOT))
