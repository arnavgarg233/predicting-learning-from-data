"""Phase 2: corpus count for the co-primary contrasts, from development errors only.

    python scripts/curve_forecast/power.py [--budget_n 30]      # -> results/curve_forecast/power.json

Effect model: the leave-one-corpus-out errors of the locked arms on the development corpora (development.json,
per_corpus_errors). A simulated test draws N corpora with replacement (error pairs travel together) and computes the corpus
bootstrap 95% upper bound (97.5th percentile, as in the scorer) of rms(arm) / rms(A3). Power = share of simulations with
the bound below 1. N80 = smallest N on the grid with power >= 0.8 for both contrasts (A4/A3 and A2/A3).
"""
import argparse
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results/curve_forecast")
GRID = (6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 35, 40, 50, 60, 80)
SIMS, BOOT, SEED = 1000, 1000, 0


def power(ea, eb, n, rng):
    hits = 0
    for _ in range(SIMS):
        i = rng.integers(0, len(ea), n)
        a, b = ea[i] ** 2, eb[i] ** 2
        j = rng.integers(0, n, (BOOT, n))
        r = np.sqrt(a[j].mean(1) / b[j].mean(1))
        hits += np.percentile(r, 97.5) < 1
    return hits / SIMS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_n", type=int, default=None, help="largest N the cost cap allows (from the smoke test)")
    ap.add_argument("--budget_note", default=None)
    a = ap.parse_args()
    D = json.load(open(os.path.join(OUT, "development.json")))
    pc = D["per_corpus_errors"]
    names = [c for c in pc["A3"] if all(pc[x].get(c) is not None for x in ("A2", "A3", "A4"))]
    e = {x: np.array([pc[x][c] for c in names]) for x in ("A2", "A3", "A4")}
    out = dict(source="results/curve_forecast/development.json per_corpus_errors (leave-one-corpus-out, 2^23)", corpora=names,
               dev_rms={x: float(np.sqrt(np.mean(v ** 2))) for x, v in e.items()},
               dev_ratio=dict(A4_over_A3=float(np.sqrt(np.mean(e["A4"] ** 2) / np.mean(e["A3"] ** 2))),
                              A2_over_A3=float(np.sqrt(np.mean(e["A2"] ** 2) / np.mean(e["A3"] ** 2)))),
               method=dict(sims=SIMS, boot=BOOT, seed=SEED, criterion="97.5th percentile of the corpus-bootstrap rms ratio < 1"),
               power={})
    for x in ("A4", "A2"):
        rng = np.random.default_rng(SEED)
        out["power"][f"{x}_over_A3"] = {str(n): power(e[x], e["A3"], n, rng) for n in GRID}
    n80 = {}
    for k, v in out["power"].items():
        ok = [int(n) for n, p in v.items() if p >= 0.8]
        n80[k] = min(ok) if ok else None
    out["N80"] = n80
    need = None if None in n80.values() else max(n80.values())
    out["N80_both"] = need
    if a.budget_n is not None:
        out["N_budget"] = a.budget_n
        out["N_budget_note"] = a.budget_note
        out["N"] = a.budget_n if need is None else min(need, a.budget_n)
        out["power_at_N"] = {f"{x}_over_A3": power(e[x], e["A3"], out["N"], np.random.default_rng(SEED)) for x in ("A4", "A2")}
    json.dump(out, open(os.path.join(OUT, "power.json"), "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k not in ("power", "corpora")}, indent=1))
    for k, v in out["power"].items():
        print(k, " ".join(f"{n}:{p:.2f}" for n, p in v.items()))


if __name__ == "__main__":
    main()
