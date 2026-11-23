"""Score Test 2, the English forward test of the frozen English line. Deterministic.

    python scripts/codeeng/score_t2.py     # results/fwd_english/pod_e*/ -> results/fwd_english/score.json

Each pod directory holds forecast_i.json, forecast_ii.json, eng_forecast.json (written and dated on the Mac before training was
released) and text_runs/cf_<corpus>/bytes_b<k>_s<seed>.json. A corpus is scored when seed 0 at 2^19, 2^20, 2^23, 2^25 exists.
Measured fall = L(2^23) seed 0 - mean L(2^25) over seeds. Outlier rule (fall_line.json): set aside if the fall or the own fall lies
more than 4 scaled MADs from the frozen calibration median. Arms (fall 2^23 -> 2^25): own_cal (frozen line), kind_mean (frozen),
A3 (two pilots 2^19, 2^20, locked zero floor), own_raw (unscaled own fall), fall_rule_only (frozen context line).
Primary: r.m.s. fall error own_cal / kind_mean on the clean set, paired corpus bootstrap (20,000, default_rng(0), 95%).
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "curve_forecast"))
import core as C  # noqa: E402

R = os.path.join(ROOT, "results", "fwd_english")
NB = 20000
ARMS = ("own_cal", "kind_mean", "A3", "own_raw", "fall_rule_only")


def rms(e):
    return float(np.sqrt(np.mean(np.square(e))))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def released(d):
    import datetime
    f = os.path.join(d, "pod", "logs", "timing.txt")
    for l in (open(f) if os.path.exists(f) else []):
        if l.startswith("eb training released"):
            return datetime.datetime.strptime(l.split()[-1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    return None


def block(names, rows):
    n = len(names)
    E = {a: np.array([rows[c]["fall_err"][a] for c in names]) for a in ARMS}
    idx = np.random.default_rng(0).integers(0, n, (NB, n))

    def ratio(ea, eb):
        br = np.sqrt((ea[idx] ** 2).mean(1) / np.maximum((eb[idx] ** 2).mean(1), 1e-30)); lo, hi = np.percentile(br, [2.5, 97.5])
        return dict(ratio=rms(ea) / rms(eb), ci95=[float(lo), float(hi)],
                    label="REAL positive" if hi < 1 else ("REAL negative" if lo > 1 else "tie"),
                    closer=f"{int(np.sum(np.abs(ea) < np.abs(eb)))}/{n}")
    f = np.array([rows[c]["fall"] for c in names]); x = np.array([rows[c]["own_fall"] for c in names])
    E25 = {a: np.array([rows[c]["e25_level"][a] for c in names]) for a in ARMS}
    return dict(n=n, corpora=names, fall_mean=float(f.mean()), fall_sd=float(f.std(ddof=1)), fall_rms={a: rms(E[a]) for a in ARMS},
                fall_bias={a: float(E[a].mean()) for a in ARMS}, primary=ratio(E["own_cal"], E["kind_mean"]),
                context={f"own_cal/{q}": ratio(E["own_cal"], E[q]) for q in ("A3", "own_raw", "fall_rule_only")}
                | {"A3/kind_mean": ratio(E["A3"], E["kind_mean"])},
                spearman=spearman(x, f), e25_rms_forecast_level={a: rms(E25[a]) for a in ARMS})


def main():
    Z = json.load(open(os.path.join(R, "fall_line.json")))["calibration"]
    lk = json.load(open(os.path.join(ROOT, "results", "curve_forecast", "locked.json")))
    rows, missing = {}, {}
    for d in sorted(glob.glob(os.path.join(R, "pod_e*"))):
        if not os.path.exists(os.path.join(d, "eng_forecast.json")):
            continue
        W = json.load(open(os.path.join(d, "eng_forecast.json"))); F = json.load(open(os.path.join(d, "forecast_i.json")))["corpora"]
        rel = released(d)
        for c, w in W["corpora"].items():
            lad = {}
            for f in glob.glob(os.path.join(d, "text_runs", f"cf_{c}", "bytes_b*_s*.json")):
                r = json.load(open(f)); lad.setdefault(int(round(np.log2(r["train_bytes"]))), {})[r["seed"]] = r["bits_per_byte"]
            need = [k for k in (19, 20, 23, 25) if 0 not in lad.get(k, {})]
            if need:
                missing[c] = need; continue
            L = {k: lad[k][0] for k in (19, 20, 23)}; s25 = dict(sorted(lad[25].items())); L[25] = float(np.mean(list(s25.values())))
            x = F[c]["S"]["23"] - F[c]["S"]["25"]
            assert abs(x - w["own_fall"]) < 1e-12 and abs(w["calibrated_fall"] - (Z["line"]["a"] + Z["line"]["b"] * x)) < 1e-12, c
            a3 = {k: C.A3({19: L[19], 20: L[20]}, k, lk["chosen"]["A3_floor"], lk["constants"]["A3_floor_param"]) for k in (23, 25)}
            fall = L[23] - L[25]; V = w["level_23"]
            pred = dict(own_cal=w["calibrated_fall"], kind_mean=Z["kind_mean"], A3=a3[23] - a3[25], own_raw=x,
                        fall_rule_only=w["calibrated_fall_fall_rule_only"])
            out_f = abs(fall - Z["median"]["fall"]) > Z["k_mad"] * Z["scaled_mad"]["fall"]
            rows[c] = dict(dir=os.path.relpath(d, ROOT), L={str(k): v for k, v in L.items()}, L25_seeds={str(s): v for s, v in s25.items()},
                           fall=fall, own_fall=x, level_23=V, pred_fall=pred, fall_err={a: p - fall for a, p in pred.items()},
                           e25_level={a: V - p - L[25] for a, p in pred.items()},
                           outlier_own_fall=w["own_fall_outlier"], outlier_fall=bool(out_f),
                           forecast_before_training=None if rel is None else bool(W["date_epoch"] < rel))
    names = sorted(rows)
    clean = [c for c in names if not (rows[c]["outlier_own_fall"] or rows[c]["outlier_fall"])]
    out = dict(design="docs/analysis/code25m_english.md (Test 2)", calibration=Z, n_scored=len(names), missing=missing,
               set_aside=[c for c in names if c not in clean], forecasts_before_training=sorted({str(rows[c]["forecast_before_training"]) for c in names}))
    if len(clean) >= 3:
        out["primary_clean"] = block(clean, rows); out["verdict"] = out["primary_clean"]["primary"]["label"]
    if len(names) >= 3:
        out["all_scored"] = block(names, rows)
    pairs = [rows[c]["L25_seeds"] for c in names if len(rows[c]["L25_seeds"]) > 1]
    out["seed_sd_2_25"] = float(np.sqrt(np.mean([(p["1"] - p["0"]) ** 2 for p in pairs]) / 2)) if pairs else None
    out["per_corpus"] = rows
    json.dump(out, open(os.path.join(R, "score.json"), "w"), indent=1)
    print("scored", len(names), "clean", len(clean), "set aside", out["set_aside"], "missing", len(missing))
    for nm in ("primary_clean", "all_scored"):
        if nm in out:
            b = out[nm]; p = b["primary"]
            print(f"{nm}: n {b['n']} rms {({a: round(v, 4) for a, v in b['fall_rms'].items()})}")
            print(f"   own/mean {p['ratio']:.3f} ({p['ci95'][0]:.3f}-{p['ci95'][1]:.3f}) {p['label']} closer {p['closer']} rho {b['spearman']:+.2f}")
            for k, v in b["context"].items():
                print(f"   {k} {v['ratio']:.3f} ({v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}) {v['closer']}")


if __name__ == "__main__":
    main()
