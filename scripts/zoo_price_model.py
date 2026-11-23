"""Budget-dependent price (2026-10-17). Fixed rule: price K(P) = 210 (P / 2^20)^(-c) events per distance, with c
the pooled within-source slope of log K on log P from the gated copies (results/dial_scores/K_run_C_s0/s1.json, both
seeds). Zoo predictions are recomputed with each source's exact information curve (exact_sources.zoo.tail_curves) and compared
with the fixed-price predictions on the same sources. Writes results/zoo/price_model.json."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exact_sources import zoo  # noqa: E402

xs, ys = [], []
for s in (0, 1):
    d = json.load(open(ROOT / f"results/dial_scores/K_run_C_s{s}.json"))
    by = {}
    for r in d["runs"]:
        by.setdefault(r["g"], []).append((r["log2P"], np.log2(r["K_run"])))
    for v in by.values():
        x, y = np.array([a for a, _ in v], float), np.array([b for _, b in v])
        xs.append(x - x.mean()); ys.append(y - y.mean())
xs, ys = np.concatenate(xs), np.concatenate(ys)
c = float(-(xs @ ys) / (xs @ xs))
A = json.load(open(ROOT / "results/zoo/analysis_partial.json"))
rows = [r for r in A["rows"] if not r["recoding"] and not r["length_control"] and r["alpha"] is not None and r["p_info"] is not None]
out = dict(c=c, rule="K(P) = 210 (P/2^20)^(-c)", sources={})
for r in rows:
    pr = json.load(open(ROOT / f"results/zoo/pred/{r['name']}.json"))
    h, absC, sd, T = np.array(pr["h_n"]), np.array(pr["absC"], float), np.array(pr["sd_window"], float), pr["T"]
    fitb = np.array(pr["budgets"][1:], float)
    tails = [zoo.tail_curves(h, absC, sd, T, [P], Bstar=zoo.B_STAR * (P / 2 ** 20) ** (-c))["info"][0] for P in fitb]
    a_bud = zoo.slope(fitb, tails)
    out["sources"][r["name"]] = dict(measured=r["alpha"], fixed_price=r["p_info"], budget_price=a_bud)
m = np.array([v["measured"] for v in out["sources"].values()])
for k in ("fixed_price", "budget_price"):
    p = np.array([v[k] for v in out["sources"].values()], float)
    ok = np.isfinite(p)
    out[f"rms_{k}"] = float(np.sqrt(np.mean((p[ok] - m[ok]) ** 2)))
    out[f"bias_{k}"] = float(np.mean(p[ok] - m[ok]))
    out[f"n_{k}"] = int(ok.sum())
json.dump(out, open(ROOT / "results/zoo/price_model.json", "w"), indent=1)
print("c =", round(c, 3))
for k, v in out["sources"].items():
    print(f"  {k:18s} measured {v['measured']:.3f}  fixed {v['fixed_price']:.3f}  budget {v['budget_price']:.3f}")
print({k: round(v, 4) for k, v in out.items() if k.startswith(("rms", "bias"))})
