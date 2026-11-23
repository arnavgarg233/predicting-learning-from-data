"""Phase 6: score every arm on the test corpora with the locked code.

    python scripts/curve_forecast/score.py      # -> results/curve_forecast/score.json

Reads locked.json, forecast.json (its sha256 is checked against logs/forecast.sha256 written on the pod), corpora.json, the
learner runs results/text_runs/cf_<c>/bytes_b<k>_s<seed>.json and the FineWeb-Edu runs (cf_fineweb at 2^19, 2^20; bfine at
2^23, 2^25). Pilots: seed 0 at 2^19 and 2^20. Targets: mean over the seeds run. A corpus is scored when its four seed-0 runs
exist. Co-primary: rms at 2^25, ratios A4/A3 and A2/A3, corpus bootstrap (20000 draws, seed 0, percentile 95%).
Secondary: 2^23 and A0/A1. Compute per arm in FLOPs (core.learner_flops, core.ref_flops).
"""
import datetime
import glob
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402

ROOT = C.ROOT
RES = os.path.join(ROOT, "results/curve_forecast")
ARMS = ("A0", "A1", "A2", "A3", "A4")
TARGETS = (23, 25)
NB = 20000


def ladder(pat):
    out = {}
    for f in glob.glob(pat):
        d = json.load(open(f))
        out.setdefault(int(round(np.log2(d["train_bytes"]))), {})[d["seed"]] = d
    return out


def boot_ratio(ea, eb, rng):
    n = len(ea)
    i = rng.integers(0, n, (NB, n))
    r = np.sqrt((ea[i] ** 2).mean(1) / (eb[i] ** 2).mean(1))
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))], float(np.mean(r < 1))


def main():
    lk = json.load(open(os.path.join(RES, "locked.json")))
    ch, K = lk["chosen"], lk["constants"]
    fp = os.path.join(RES, "forecast.json")
    F = json.load(open(fp))
    fsha = hashlib.sha256(open(fp, "rb").read()).hexdigest()
    lp = os.path.join(RES, "logs/forecast.sha256")
    logged = open(lp).read().split()[0] if os.path.exists(lp) else None
    fwr = {**ladder(os.path.join(ROOT, "results/text_runs/bfine/bytes_b*_s*.json")),
           **ladder(os.path.join(ROOT, "results/text_runs/cf_fineweb/bytes_b*_s*.json"))}
    fw = {k: float(np.mean([d["bits_per_byte"] for d in fwr[k].values()])) for k in (19, 20, 23, 25) if k in fwr}
    per, missing = {}, {}
    for c, fc in F["corpora"].items():
        lad = ladder(os.path.join(ROOT, f"results/text_runs/cf_{c}/bytes_b*_s*.json"))
        need = [k for k in (19, 20) + TARGETS if 0 not in lad.get(k, {})]
        if need:
            missing[c] = need
            continue
        hr = C.FN.HRef(os.path.join(RES, c, "refcurve", "refcurve_fineweb.json"), fc["ref_model"])
        assert hr.sha == fc["refcurve_sha256"], c
        nu, n0 = fc["shape"]["nu"], fc["shape"]["n0"]
        L19, L20 = lad[19][0]["bits_per_byte"], lad[20][0]["bits_per_byte"]
        meas = {k: float(np.mean([d["bits_per_byte"] for d in lad[k].values()])) for k in TARGETS}
        pred = {}
        for k in TARGETS:
            pred[k] = dict(A0=fc["A0"][str(k)], A1=C.A1(L20, 20, k, fw),
                           A2=C.hybrid(hr, nu, n0, {20: L20}, k, ch["A2_form"]),
                           A3=C.A3({19: L19, 20: L20}, k, ch["A3_floor"], K["A3_floor_param"]),
                           A4=C.hybrid(hr, nu, n0, {19: L19, 20: L20}, k, ch["A4_form"]))
        st = lambda k: lad[k][0]
        fl = lambda k: C.learner_flops(st(k)["steps"], st(k).get("eval_chunks", 0))
        ref = C.ref_flops(fc["ref_model"], fc.get("ref_n_targets") or 20000)
        flops = dict(A0=ref, A1=fl(20), A2=ref + fl(20), A3=fl(19) + fl(20), A4=ref + fl(19) + fl(20),
                     target_2_23=fl(23), target_2_25=fl(25))
        per[c] = dict(family=fc["family"], ref_model=fc["ref_model"], pilots={"19": L19, "20": L20},
                      measured={str(k): v for k, v in meas.items()}, seeds={str(k): sorted(lad[k]) for k in TARGETS},
                      forecast={str(k): pred[k] for k in TARGETS},
                      err={str(k): {a: pred[k][a] - meas[k] for a in ARMS} for k in TARGETS},
                      flops=flops, steps={str(k): st(k)["steps"] for k in (19, 20) + TARGETS})
    names = list(per)
    out = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               forecast=dict(sha256=fsha, sha256_logged=logged, match=fsha == logged, created_utc=F["created_utc"]),
               locked_sha256=hashlib.sha256(open(os.path.join(RES, "locked.json"), "rb").read()).hexdigest(),
               locked_sha256_in_forecast=F["locked_sha256"], chosen=ch, fineweb_curve=fw,
               n_scored=len(names), scored=names, missing=missing, per_corpus=per)
    if names:
        rng = np.random.default_rng(0)
        for k in TARGETS:
            E = {a: np.array([per[c]["err"][str(k)][a] for c in names]) for a in ARMS}
            blk = dict(rms={a: float(np.sqrt(np.mean(E[a] ** 2))) for a in ARMS}, bias={a: float(np.mean(E[a])) for a in ARMS},
                       median_abs={a: float(np.median(np.abs(E[a]))) for a in ARMS})
            for x, y in (("A4", "A3"), ("A2", "A3"), ("A0", "A1"), ("A2", "A1"), ("A4", "A1")):
                ci, share = boot_ratio(E[x], E[y], rng)
                blk[f"{x}_over_{y}"] = dict(ratio=blk["rms"][x] / blk["rms"][y], ci95=ci, share_draws_below_1=share)
            fam = {}
            for f in sorted({per[c]["family"] for c in names}):
                idx = [i for i, c in enumerate(names) if per[c]["family"] == f]
                fam[f] = dict(n=len(idx), rms={a: float(np.sqrt(np.mean(E[a][idx] ** 2))) for a in ARMS})
            blk["by_family"] = fam
            out[f"budget_2_{k}"] = blk
        out["co_primary_2_25"] = {r: out["budget_2_25"][r] for r in ("A4_over_A3", "A2_over_A3")}
        out["flops_median"] = {a: float(np.median([per[c]["flops"][a] for c in names])) for a in per[names[0]]["flops"]}
        seeds2 = [c for c in names if len(per[c]["seeds"]["25"]) > 1]
        if seeds2:
            sn = {c: {str(s): json.load(open(os.path.join(ROOT, f"results/text_runs/cf_{c}/bytes_b25_s{s}.json")))["bits_per_byte"]
                      for s in per[c]["seeds"]["25"]} for c in seeds2}
            d = [v["1"] - v["0"] for v in sn.values() if "0" in v and "1" in v]
            out["seed_noise_2_25"] = dict(per_corpus=sn, sd_single_run_bpb=float(np.sqrt(np.mean(np.square(d)) / 2)) if d else None)
    json.dump(out, open(os.path.join(RES, "score.json"), "w"), indent=1, default=float)
    print(json.dumps({k: out[k] for k in ("n_scored", "missing", "co_primary_2_25") if k in out}, indent=1))
    if names:
        print("rms 2^25", out["budget_2_25"]["rms"]); print("rms 2^23", out["budget_2_23"]["rms"])


if __name__ == "__main__":
    main()
