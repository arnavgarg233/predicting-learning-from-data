"""Score the forward test of the within-kind fall calibration on fresh FineWeb-2 web languages.
Deterministic.

    python scripts/fwd_weblang/score.py              # results/fwd_weblang/{pod*,mac*}/ -> score.json
    python scripts/fwd_weblang/score.py --selftest   # same code on the 29 calibration languages (in-sample; checks the arms)

Each run directory holds forecast_i.json, forecast_ii.json, fwd_forecast.json (written before training) and
text_runs/cf_<corpus>/bytes_b<k>_s<seed>.json. A corpus is scored when seed 0 at 2^19, 2^20, 2^23, 2^25 exists.
Measured fall = L(2^23) seed 0 - L(2^25) mean over seeds. Arms (fall 2^23 -> 2^25):
  own_cal    a + b * own_fall (fall_line.json)                 kind_mean  frozen kind mean fall
  A3         two pilots (2^19, 2^20), locked zero floor     own_raw    own fall S(2^23) - S(2^25), unscaled
  pool_exp   level_23 * (1 - 2^-0.28)
Primary: r.m.s. fall error own_cal / kind_mean, paired corpus bootstrap (20,000 draws, default_rng(0), 95% percentile),
closer count, verdict (REAL positive if interval < 1, REAL negative if > 1, else tie). The same bootstrap indices serve every
ratio. Secondary: r.m.s. 2^25 error at the forecast level (level_23 - fall - L25) and at the observed L(2^23); Spearman of own
fall with the measured fall.
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
import core as C  # noqa: E402

NB = 20000
A_POOL = 0.14
ARMS = ("own_cal", "kind_mean", "A3", "own_raw", "pool_exp")
R = os.path.join(ROOT, "results", "fwd_weblang")


def rms(e):
    return float(np.sqrt(np.mean(np.square(e))))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 0 and np.std(ry) > 0 else float("nan")


def ladder(d, c):
    out = {}
    for f in glob.glob(os.path.join(d, "text_runs", f"cf_{c}", "bytes_b*_s*.json")):
        r = json.load(open(f))
        out.setdefault(int(round(np.log2(r["train_bytes"]))), {})[r["seed"]] = r
    return out


def train_released_epoch(d):
    """first 'eb training released' stamp of the pod (UTC), or None."""
    import datetime
    f = os.path.join(d, "pod", "logs", "timing.txt")
    if not os.path.exists(f):
        return None
    for l in open(f):
        if l.startswith("eb training released"):
            return datetime.datetime.strptime(l.split()[-1], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    return None


def load(d, Z, lk, selftest, first=None):
    F = json.load(open(os.path.join(d, "forecast_i.json")))["corpora"]
    G = json.load(open(os.path.join(d, "forecast_ii.json")))["corpora"]
    if selftest:
        W, wdate = None, None
    elif os.path.exists(os.path.join(d, "fwd_forecast.json")):
        Wf = json.load(open(os.path.join(d, "fwd_forecast.json")))
        W, wdate = Wf["corpora"], Wf["date_epoch"]
        assert abs(Wf["calibration"]["a"] - Z["a"]) < 1e-15 and abs(Wf["calibration"]["b"] - Z["b"]) < 1e-15
    else:                      # recovery pods (slots 5-7) write no forecast of their own; the first dated one is scored
        W, wdate = None, None
    rel = train_released_epoch(d)
    rows, missing = {}, {}
    for c, fc in F.items():
        if fc["family"] != "lang":
            continue
        lad = ladder(d, c)
        need = [k for k in (19, 20, 23, 25) if 0 not in lad.get(k, {})]
        if need:
            missing[c] = need
            continue
        L = {k: lad[k][0]["bits_per_byte"] for k in (19, 20, 23)}
        s25 = {s: r["bits_per_byte"] for s, r in sorted(lad[25].items())}
        L[25] = float(np.mean(list(s25.values())))
        x = fc["S"]["23"] - fc["S"]["25"]
        V = G[c]["level_23"]
        cal = Z["a"] + Z["b"] * x
        if W is not None:      # the forecasts as written before training must be what is scored
            assert abs(W[c]["calibrated_fall"] - cal) < 1e-12 and abs(W[c]["level_23"] - V) < 1e-12, c
        rerun = None
        if first and c in first and first[c]["dir"] != d:
            # corpus re-prepared on a recovery pod: score the first dated forecast, report how far the rerun's differs
            f0 = first[c]
            rerun = dict(first_forecast_dir=os.path.relpath(f0["dir"], ROOT), first_forecast_date=f0["date"],
                         d_own_fall=x - f0["own_fall"], d_level_23=V - f0["level_23"])
            x, V, cal = f0["own_fall"], f0["level_23"], f0["calibrated_fall"]
            wdate = f0["epoch"]
        a3 = {k: C.A3({19: L[19], 20: L[20]}, k, lk["chosen"]["A3_floor"], lk["constants"]["A3_floor_param"]) for k in (23, 25)}
        fall = L[23] - L[25]
        pred = dict(own_cal=cal, kind_mean=Z["kind_mean_fall"], A3=a3[23] - a3[25], own_raw=x, pool_exp=V * (1 - 2.0 ** (-2 * A_POOL)))
        rows[c] = dict(L={str(k): v for k, v in L.items()}, L25_seeds={str(s): v for s, v in s25.items()}, fall=fall,
                       own_fall=x, level_23=V, pred_fall=pred, fall_err={a: p - fall for a, p in pred.items()},
                       e25_level={a: V - p - L[25] for a, p in pred.items()}, e25_obs23={a: L[23] - p - L[25] for a, p in pred.items()},
                       A3_e25_own=a3[25] - L[25], dir=os.path.relpath(d, ROOT),
                       forecast_before_training=(None if wdate is None or rel is None else bool(wdate < rel)), rerun_vs_first=rerun)
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:     # the same full-kind fit that fall_line.json carries, read from its source
        f = json.load(open(os.path.join(ROOT, "results", "reanalysis", "shape_within_kind.json")))["kinds"]["web"]["full_fit"]
        Z = dict(a=f["a"], b=f["b"], kind_mean_fall=f["mean_fall"])
    else:
        Z = json.load(open(os.path.join(R, "fall_line.json")))["calibration"]
    lk = json.load(open(os.path.join(ROOT, "results", "curve_forecast", "locked.json")))
    if a.selftest:
        dirs = sorted(glob.glob(os.path.join(ROOT, "results", "extbench", "pod[0-9]")))
        out_f = os.path.join(R, "score_selftest_calibration29.json")
    else:
        dirs = sorted(glob.glob(os.path.join(R, "pod[0-9]*")) + glob.glob(os.path.join(R, "mac*")))
        out_f = os.path.join(R, "score.json")
    first = {}                 # earliest dated forecast per corpus over all run directories
    if not a.selftest:
        for d in dirs:
            f = os.path.join(d, "fwd_forecast.json")
            if os.path.exists(f):
                W = json.load(open(f))
                for c, w in W["corpora"].items():
                    if c not in first or W["date_epoch"] < first[c]["epoch"]:
                        first[c] = dict(w, dir=d, date=W["date"], epoch=W["date_epoch"])
    rows, missing = {}, {}
    for d in dirs:
        if os.path.exists(os.path.join(d, "forecast_i.json")):
            r, m = load(d, Z, lk, a.selftest, first)
            for c, v in r.items():
                if c in rows:
                    raise SystemExit(f"{c} has complete runs in two directories")
                rows[c] = v
            missing.update(m)
    missing = {c: v for c, v in missing.items() if c not in rows}
    names = sorted(rows)
    n = len(names)
    E = {a_: np.array([rows[c]["fall_err"][a_] for c in names]) for a_ in ARMS}
    E25 = {a_: np.array([rows[c]["e25_level"][a_] for c in names]) for a_ in ARMS}
    E25o = {a_: np.array([rows[c]["e25_obs23"][a_] for c in names]) for a_ in ARMS}
    idx = np.random.default_rng(0).integers(0, n, (NB, n))

    def ratio(ea, eb):
        br = np.sqrt((ea[idx] ** 2).mean(1) / np.maximum((eb[idx] ** 2).mean(1), 1e-30))
        lo, hi = np.percentile(br, [2.5, 97.5])
        return dict(ratio=rms(ea) / rms(eb), ci95=[float(lo), float(hi)],
                    label="REAL positive" if hi < 1 else ("REAL negative" if lo > 1 else "tie"),
                    closer=f"{int(np.sum(np.abs(ea) < np.abs(eb)))}/{n}")

    fall = np.array([rows[c]["fall"] for c in names]); own = np.array([rows[c]["own_fall"] for c in names])
    sp = [spearman(own[i], fall[i]) for i in idx[:2000]]
    pairs = [rows[c]["L25_seeds"] for c in names if len(rows[c]["L25_seeds"]) > 1]
    dd = np.array([p["1"] - p["0"] for p in pairs]) if pairs else np.array([])
    out = dict(n_scored=n, missing=missing, mode="selftest (in-sample, calibration corpora)" if a.selftest else "forward test",
               calibration=Z, fall_mean=float(fall.mean()), fall_sd=float(fall.std(ddof=1)),
               fall_rms={a_: rms(E[a_]) for a_ in ARMS}, fall_bias={a_: float(E[a_].mean()) for a_ in ARMS},
               primary=ratio(E["own_cal"], E["kind_mean"]),
               context={f"own_cal/{q}": ratio(E["own_cal"], E[q]) for q in ("A3", "own_raw", "pool_exp")},
               e25_rms_forecast_level={a_: rms(E25[a_]) for a_ in ARMS},
               e25_rms_observed_L23={a_: rms(E25o[a_]) for a_ in ARMS},
               e25_ratio_forecast_level_own_cal_vs_kind_mean=ratio(E25["own_cal"], E25["kind_mean"]),
               A3_own_e25_rms=rms(np.array([rows[c]["A3_e25_own"] for c in names])),
               spearman_own_vs_fall=dict(rho=spearman(own, fall), ci95=[float(np.nanpercentile(sp, 2.5)), float(np.nanpercentile(sp, 97.5))]),
               seed_noise_2_25=dict(n=len(dd), sd_single_run=float(np.sqrt(np.mean(dd ** 2) / 2)) if len(dd) else None),
               forecasts_before_training=sorted({str(rows[c]["forecast_before_training"]) for c in names}),
               rerun_max_abs_diff=dict(own_fall=max([abs(rows[c]["rerun_vs_first"]["d_own_fall"]) for c in names if rows[c]["rerun_vs_first"]] or [0.0]),
                                       level_23=max([abs(rows[c]["rerun_vs_first"]["d_level_23"]) for c in names if rows[c]["rerun_vs_first"]] or [0.0]),
                                       n=sum(1 for c in names if rows[c]["rerun_vs_first"])),
               per_corpus=rows)
    out["verdict"] = out["primary"]["label"]

    def sub(sel):              # the same statistics on a subset (own bootstrap draws, default_rng(0))
        n_ = len(sel)
        if n_ < 2:
            return dict(n=n_)
        e = {a_: np.array([rows[c]["fall_err"][a_] for c in sel]) for a_ in ARMS}
        ix = np.random.default_rng(0).integers(0, n_, (NB, n_))
        def rt(ea, eb):
            br = np.sqrt((ea[ix] ** 2).mean(1) / np.maximum((eb[ix] ** 2).mean(1), 1e-30))
            lo, hi = np.percentile(br, [2.5, 97.5])
            return dict(ratio=rms(ea) / rms(eb), ci95=[float(lo), float(hi)],
                        label="REAL positive" if hi < 1 else ("REAL negative" if lo > 1 else "tie"),
                        closer=f"{int(np.sum(np.abs(ea) < np.abs(eb)))}/{n_}")
        f_ = np.array([rows[c]["fall"] for c in sel]); o_ = np.array([rows[c]["own_fall"] for c in sel])
        return dict(n=n_, corpora=sel, fall_rms={a_: rms(e[a_]) for a_ in ARMS}, primary=rt(e["own_cal"], e["kind_mean"]),
                    own_cal_vs_A3=rt(e["own_cal"], e["A3"]), spearman=spearman(o_, f_))
    first_dirs = {"pod1", "pod2"}
    out["subsets"] = {"first8 (pods 1-2, morning)": sub([c for c in names if os.path.basename(rows[c]["dir"]) in first_dirs]),
                      "new (pods 5-9)": sub([c for c in names if os.path.basename(rows[c]["dir"]) not in first_dirs])}
    json.dump(out, open(out_f, "w"), indent=1)
    p = out["primary"]
    print(f"{out['mode']}: n={n} missing={len(missing)} fall {out['fall_mean']:.3f}+-{out['fall_sd']:.3f}")
    print("fall rms " + " ".join(f"{k} {v:.4f}" for k, v in out["fall_rms"].items()))
    print(f"PRIMARY own_cal/kind_mean {p['ratio']:.3f} ({p['ci95'][0]:.3f}-{p['ci95'][1]:.3f}) closer {p['closer']} -> {p['label']}")
    for k, v in out["context"].items():
        print(f"  {k:18s} {v['ratio']:.3f} ({v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}) closer {v['closer']}")
    print("e25 at forecast level " + " ".join(f"{k} {v:.4f}" for k, v in out["e25_rms_forecast_level"].items()))
    print("e25 at observed L23   " + " ".join(f"{k} {v:.4f}" for k, v in out["e25_rms_observed_L23"].items()))
    s = out["spearman_own_vs_fall"]
    print(f"Spearman own vs fall {s['rho']:.3f} ({s['ci95'][0]:.2f}-{s['ci95'][1]:.2f}); seed noise {out['seed_noise_2_25']}; "
          f"forecast before training: {out['forecasts_before_training']}; rerun vs first forecast {out['rerun_max_abs_diff']}")
    for k, v in out.get("subsets", {}).items():
        if v.get("n", 0) >= 2:
            q = v["primary"]
            print(f"  {k}: n={v['n']} own {v['fall_rms']['own_cal']:.4f} mean {v['fall_rms']['kind_mean']:.4f} A3 {v['fall_rms']['A3']:.4f} "
                  f"ratio {q['ratio']:.3f} ({q['ci95'][0]:.3f}-{q['ci95'][1]:.3f}) closer {q['closer']} {q['label']}; Spearman {v['spearman']:.2f}")
    print("->", os.path.relpath(out_f, ROOT))


if __name__ == "__main__":
    main()
