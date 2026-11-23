"""Score the external benchmark in one command.

    python scripts/extbench/score.py                                   # the benchmark: results/extbench/pod{1,2,3}
    python scripts/extbench/score.py --dirs D1 [D2 ...] --out OUT.json  # any run laid out the same way (test: the forward test)

Each directory holds forecast_i.json (or forecast.json: forecaster (i), written before training), optionally forecast_ii.json
(written before training; computed here from forecast (i) with scripts/extbench/forecast_ii.py if absent, and marked so), and
text_runs/cf_<corpus>/bytes_b<k>_s<seed>.json. A corpus is scored when its seed-0 runs at 2^19, 2^20, 2^23 and 2^25 exist.
Target at 2^25: mean over the seeds run (seed 0 and seed 1); at 2^23: seed 0.

Arms (forecasts at 2^23 and 2^25):
  i        the paper's forecaster A0
  ii       the revised n-gram-level route (forecast_ii.py)
  A3       two pilots (2^19, 2^20), power law with the locked floor rule
  X_pool1  one pilot at 2^20 with the pooled exponent 0.14
  P2023    pilots at 2^20 and 2^23 (zero-floor power law through both; its level error at 2^23 is 0 by construction)
Errors: level = F(2^23) - L(2^23); drop = [F(2^23) - F(2^25)] - [L(2^23) - L(2^25)]; e25 = F(2^25) - L(2^25).
Per family and overall: r.m.s. and bias of each; paired corpus-bootstrap ratios of r.m.s. (20,000 draws, seed 0, percentile 95%)
for i and ii against A3, X_pool1 and P2023 (and ii against i); closer counts; c-flag counts (c = reference full-context bits
per byte / best n-gram cross-entropy at 2^23 >= 0.70) and errors by flag; seed noise at 2^25; FLOPs per arm (median).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "curve_forecast"))
sys.path.insert(0, HERE)
import core as C  # noqa: E402
import forecast_ii as FII  # noqa: E402

NB = 20000
A_POOL = 0.14
C_FLAG = 0.70
ARMS = ("i", "ii", "A3", "X_pool1", "P2023")
PAIRS = (("i", "A3"), ("ii", "A3"), ("i", "X_pool1"), ("ii", "X_pool1"), ("i", "P2023"), ("ii", "P2023"), ("ii", "i"))
DEFAULT_DIRS = [os.path.join(ROOT, f"results/extbench/pod{k}") for k in (1, 2, 3)]


def boot_ratio(ea, eb, rng):
    n = len(ea)
    i = rng.integers(0, n, (NB, n))
    r = np.sqrt((ea[i] ** 2).mean(1) / np.maximum((eb[i] ** 2).mean(1), 1e-30))
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))], float(np.mean(r < 1))


def ladder(d, c):
    out = {}
    for f in glob.glob(os.path.join(d, "text_runs", f"cf_{c}", "bytes_b*_s*.json")):
        r = json.load(open(f))
        out.setdefault(int(round(np.log2(r["train_bytes"]))), {})[r["seed"]] = r
    return out


def load_dir(d, lk):
    fi = os.path.join(d, "forecast_i.json")
    F = json.load(open(fi if os.path.exists(fi) else os.path.join(d, "forecast.json")))
    fp = os.path.join(d, "forecast_ii.json")
    if os.path.exists(fp):
        G, g_src = json.load(open(fp))["corpora"], "file written before training"
    else:
        G, g_src = FII.forecast(F), "computed at scoring from forecast (i)"
    ch, K = lk["chosen"], lk["constants"]
    rows, missing = {}, {}
    for c, fc in F["corpora"].items():
        lad = ladder(d, c)
        need = [k for k in (19, 20, 23, 25) if 0 not in lad.get(k, {})]
        if need:
            missing[c] = need
            continue
        L = {k: lad[k][0]["bits_per_byte"] for k in (19, 20, 23)}
        s25 = {s: r["bits_per_byte"] for s, r in sorted(lad[25].items())}
        L[25] = float(np.mean(list(s25.values())))
        a2 = np.log2(L[20] / L[23]) / 3
        pred = {k: dict(i=fc["A0"][str(k)], ii=G[c][f"F{k}"],
                        A3=C.A3({19: L[19], 20: L[20]}, k, ch["A3_floor"], K["A3_floor_param"]),
                        X_pool1=float(L[20] * 2.0 ** (-A_POOL * (k - 20))),
                        P2023=float(L[23] if k == 23 else L[23] * 2.0 ** (-a2 * (k - 23)))) for k in (23, 25)}
        st = fc["ref_stats"]
        fl = {k: C.learner_flops(lad[k][0]["steps"], lad[k][0].get("eval_chunks", 0)) for k in (19, 20, 23, 25)}
        ref1k, ref_run = C.ref_flops(fc["ref_model"], 1000), C.ref_flops(fc["ref_model"], fc.get("ref_n_targets") or 20000)
        rows[c] = dict(family=fc["family"], ref_model=fc["ref_model"], c_ratio=st["ref_full"] / st["ngram23"],
                       flag=bool(st["ref_full"] / st["ngram23"] >= C_FLAG), L={str(k): v for k, v in L.items()},
                       L25_seeds={str(s): v for s, v in s25.items()}, forecast={str(k): pred[k] for k in pred},
                       err={a: dict(level=pred[23][a] - L[23], drop=(pred[23][a] - pred[25][a]) - (L[23] - L[25]),
                                    e25=pred[25][a] - L[25]) for a in ARMS},
                       flops=dict(i_1k=ref1k, i_run=ref_run, ii_23=(0.0 if fc["family"] not in ("code", "english") else ref1k),
                                  ii_25=ref1k, A3=fl[19] + fl[20], X_pool1=fl[20], P2023=fl[20] + fl[23], target_25=fl[25]),
                       source_dir=os.path.relpath(d, ROOT))
    return rows, missing, g_src


def block(rows, names, rng):
    if not names:
        return dict(n=0)
    E = {q: {a: np.array([rows[c]["err"][a][q] for c in names]) for a in ARMS} for q in ("level", "drop", "e25")}
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    b = dict(n=len(names), rms={q: {a: rms(E[q][a]) for a in ARMS} for q in E}, bias={q: {a: float(np.mean(E[q][a])) for a in ARMS} for q in E})
    b["ratios"] = {}
    for q in ("e25", "level", "drop"):
        for x, y in PAIRS:
            if q == "level" and y == "P2023":
                continue                    # P2023's level error is 0 by construction
            ci, share = boot_ratio(E[q][x], E[q][y], rng)
            b["ratios"][f"{q}:{x}/{y}"] = dict(ratio=rms(E[q][x]) / max(rms(E[q][y]), 1e-30), ci95=ci, share_draws_below_1=share,
                                              closer=f"{int(np.sum(np.abs(E[q][x]) < np.abs(E[q][y])))}/{len(names)}")
    fl = [c for c in names if rows[c]["flag"]]
    b["c_flag"] = dict(n_flagged=len(fl), n=len(names),
                       e25_rms_flagged={a: rms(np.array([rows[c]["err"][a]["e25"] for c in fl])) for a in ARMS} if fl else None,
                       e25_rms_unflagged={a: rms(np.array([rows[c]["err"][a]["e25"] for c in names if not rows[c]["flag"]])) for a in ARMS}
                       if len(fl) < len(names) else None)
    b["flops_median"] = {k: float(np.median([rows[c]["flops"][k] for c in names])) for k in rows[names[0]]["flops"]}
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/extbench/score.json"))
    a = ap.parse_args()
    lk = json.load(open(os.path.join(ROOT, "results/curve_forecast/locked.json")))
    rows, missing, gsrc = {}, {}, {}
    for d in a.dirs:
        if not os.path.exists(d):
            continue
        r, m, g = load_dir(d, lk)
        rows.update(r); missing.update(m); gsrc[os.path.relpath(d, ROOT)] = g
    names = list(rows)
    fams = sorted({rows[c]["family"] for c in names})
    rng = np.random.default_rng(0)
    out = dict(n_scored=len(names), missing=missing, forecast_ii_source=gsrc, families={f: block(rows, [c for c in names if rows[c]["family"] == f], rng) for f in fams})
    out["families"]["all"] = block(rows, names, rng)
    pairs = [(c, rows[c]["L25_seeds"]) for c in names if len(rows[c]["L25_seeds"]) > 1]
    if pairs:
        d = np.array([s["1"] - s["0"] for _, s in pairs])
        out["seed_noise_2_25"] = dict(n=len(pairs), sd_single_run=float(np.sqrt(np.mean(d ** 2) / 2)), mean_abs_diff=float(np.mean(np.abs(d))))
    out["per_corpus"] = rows
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print(f"scored {len(names)}; missing {len(missing)}; forecast (ii): {gsrc}")
    for f in fams + ["all"]:
        b = out["families"][f]
        if not b["n"]:
            continue
        r = b["rms"]
        print(f"{f:8s} n={b['n']:2d} | e25 " + " ".join(f"{x} {r['e25'][x]:.3f}" for x in ARMS) +
              f" | level i {r['level']['i']:.3f} ii {r['level']['ii']:.3f} A3 {r['level']['A3']:.3f} | drop i {r['drop']['i']:.3f} ii {r['drop']['ii']:.3f} A3 {r['drop']['A3']:.3f}"
              f" | flags {b['c_flag']['n_flagged']}/{b['n']}")
        for k in ("e25:i/A3", "e25:ii/A3", "e25:ii/X_pool1", "e25:ii/P2023"):
            q = b["ratios"][k]
            print(f"           {k:16s} {q['ratio']:.2f} ({q['ci95'][0]:.2f}-{q['ci95'][1]:.2f}) closer {q['closer']}")
    if "seed_noise_2_25" in out:
        print("seed noise 2^25", out["seed_noise_2_25"])
    print("->", a.out)


if __name__ == "__main__":
    main()
