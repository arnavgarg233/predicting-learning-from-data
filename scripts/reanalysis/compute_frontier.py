"""Forecast error at 2^25 bytes against the compute spent before the forecast, counting
the reference model's inference.

    python scripts/reanalysis/compute_frontier.py   -> results/reanalysis/compute_frontier.json

Corpora: the 80 test corpora (score80.json) and the 18 new ones (new_corpora/score.json, analysis.json).
Methods and the FLOPs charged to each (core.learner_flops / core.ref_flops; 2 N per token for a reference pass, upper
estimate, every target scored at all 18 contexts):
  A0_20k   data alone, reference curve on 20,000 targets (as run)                      ref pass, 20,000 targets
  A0_1k    data alone, the seeded 1,000-target subset of the same per-target losses       ref pass, 1,000 targets
           SmolLM2 corpora: locked level map; Qwen2.5-0.5B corpora: the level map refitted on development with the
           subset curves (arm sub05 of scripts/stronger_ref/score.py). Recomputed here for the new corpora.
  X_pool1  one pilot at 2^20 and the zero-floor exponent pooled on development (a = 0.14)  pilot 2^20
  A3       two pilots (2^19, 2^20), power law with no floor                               pilots 2^19 + 2^20
  A3_20_23 pilots 2^20 and 2^23, power law with no floor                                  pilots 2^20 + 2^23
  A0_1.5B, A0_7B  Wikipedia only, 1,000 targets, 47 test corpora (stronger-reference score files)
  target   training the learner to 2^25 itself (error = seed noise, 0.007 bits per byte)
Fixed costs shared by the arms (the 13 development corpora that calibrate the level map, K and the pooled exponent; the
CPU n-gram statistic) are not charged to any arm. Intervals: 20,000 corpus-bootstrap resamples (seed 0).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/curve_forecast"))
sys.path.insert(0, str(ROOT / "scripts/reanalysis"))
sys.path.insert(0, str(ROOT / "text/refcurve"))
import core as C  # noqa: E402
import refcurve_fineweb as RF  # noqa: E402
from ref_subset_cost import Curve, a0  # noqa: E402
from stronger_baselines import pl, one_pilot  # noqa: E402

RES = ROOT / "results/curve_forecast"
NEW = RES / "new_corpora"
OUT = ROOT / "results/reanalysis"
NB = 20000
SEED_SD = 0.007
NS_SUM = sum(n + 1 for n in C.REF_NS)
PARAMS = {"SmolLM2-135M": 134.5e6, "Qwen2.5-0.5B": 494.0e6, "Qwen2.5-1.5B": 1.54e9, "Qwen2.5-7B": 7.62e9}
ref_cost = lambda m, nt: 2.0 * PARAMS[m] * nt * NS_SUM  # noqa: E731
A_POOL = 0.14


def sub_curve(rdir, model):
    d = json.load(open(rdir / "refcurve_fineweb.json"))[model]
    if (rdir / f"{model}_raw.npz").exists():
        raw = np.load(rdir / f"{model}_raw.npz")
        nt = len(raw["tgt"])
        idx = np.sort(np.random.default_rng(20261025).choice(nt, min(1000, nt), replace=False))
    else:  # released subset written by scripts/stronger_ref/extract_sub1000.py (same seeded 1,000 targets)
        raw = np.load(rdir / f"{model}_sub1000.npz")
        idx = np.arange(len(raw["loss"]))
    r = RF.summarize(model, model, raw["loss"][idx] / np.log(2), raw["doc"][idx], raw["tgtbytes"][idx].astype(float),
                     raw["ctxb"][idx], list(raw["NS"]), d["n_docs_total"])
    return Curve(d["rows"]), Curve(r["rows"])


def main():
    lk = json.load(open(RES / "locked.json"))["constants"]
    sr = json.load(open(RES / "stronger_ref/score.json"))["arms"]
    K05 = dict(lk, level_coef=sr["sub05"]["level_coef"])
    sb = json.load(open(OUT / "stronger_baselines.json"))["per_corpus"]
    sub = json.load(open(OUT / "ref_subset_cost.json"))["per_corpus"]
    s05 = {k.split(":", 1)[1]: v["err_A0"]["25"] for k, v in sr["sub05"]["per_corpus"].items() if k.startswith("test80:")}
    per = {}
    for c, v in sb.items():
        fl = v["flops"]
        m = v["ref_model"]
        e = v["err"]
        rec = dict(family=v["family"], group="test80", ref_model=m,
                   err=dict(A0_20k=e["A0"], A0_1k=sub[c]["err_sub"] if m == "SmolLM2-135M" else s05[c],
                            X_pool1=e["X_pool1"], A3=e["A3"], A3_20_23=e["A3_20_23"]),
                   flops=dict(A0_20k=fl["ref_pass"], A0_1k=ref_cost(m, 1000), X_pool1=fl["pilot_20"], A3=fl["pilots_19_20"],
                              A3_20_23=fl["pilots_20_23"], target=fl["run_25"]))
        if "A0_1.5B" in e:
            rec["err"]["A0_1.5B"], rec["flops"]["A0_1.5B"] = e["A0_1.5B"], ref_cost("Qwen2.5-1.5B", 1000)
            rec["err"]["A0_7B"], rec["flops"]["A0_7B"] = e["A0_7B"], ref_cost("Qwen2.5-7B", 1000)
        per[c] = rec
    # the 18 new corpora
    sn = json.load(open(NEW / "score.json"))["per_corpus"]
    an = json.load(open(NEW / "analysis.json"))
    assert abs(an["a_pool"] - A_POOL) < 1e-12
    Fn = json.load(open(NEW / "forecast.json"))["corpora"]
    chk = []
    for c, p in sn.items():
        m, st = p["ref_model"], Fn[c]["ref_stats"]
        full, s1k = sub_curve(NEW / "pod" / c / "refcurve", m)
        K = lk if m == "SmolLM2-135M" else K05
        y = p["measured"]["25"]
        chk.append(abs(a0(full, st["ngram23"], lk, 25) - p["forecast"]["25"]["A0"]))
        L19, L20, L23 = p["pilots"]["19"], p["pilots"]["20"], p["measured"]["23"]
        assert p["seeds"]["23"] == [0]
        xp = one_pilot(L20, 20, 25, A_POOL)
        assert abs(xp - an["per_corpus"][c]["forecast_25"]["X_pool1"]) < 1e-9
        fl = p["flops"]
        per[c] = dict(family=p["family"], group="new18", ref_model=m,
                      err=dict(A0_20k=p["err"]["25"]["A0"], A0_1k=a0(s1k, st["ngram23"], K, 25) - y, X_pool1=xp - y,
                               A3=p["err"]["25"]["A3"], A3_20_23=pl({20: L20, 23: L23}, 25, 0.0) - y),
                      flops=dict(A0_20k=fl["A0"], A0_1k=ref_cost(m, 1000), X_pool1=fl["A1"], A3=fl["A3"],
                                 A3_20_23=fl["A1"] + fl["target_2_23"], target=fl["target_2_25"]))
    # check: the sub05 recomputation reproduces the stronger-ref file on the 80 (Qwen corpora)
    chk05 = []
    for c, v in per.items():
        if v["group"] == "test80" and v["ref_model"] == "Qwen2.5-0.5B":
            rdir = RES / c / "refcurve" if (RES / c / "refcurve").exists() else RES / "stage2" / c / "refcurve"
            if not ((rdir / "Qwen2.5-0.5B_raw.npz").exists() or (rdir / "Qwen2.5-0.5B_sub1000.npz").exists()):
                continue
            fc = {**json.load(open(RES / "forecast.json"))["corpora"], **json.load(open(RES / "stage2/forecast.json"))["corpora"]}
            _, s1k = sub_curve(rdir, "Qwen2.5-0.5B")
            y = json.load(open(RES / "score80.json"))["per_corpus"][c]["measured"]["25"]
            chk05.append(abs(a0(s1k, fc[c]["ref_stats"]["ngram23"], K05, 25) - y - v["err"]["A0_1k"]))
            if len(chk05) >= 3:
                break
    METH = ["A0_20k", "A0_1k", "X_pool1", "A3", "A3_20_23", "A0_1.5B", "A0_7B"]
    rng = np.random.default_rng(0)
    blocks = {}
    sets = {"code": lambda v: v["family"] == "code", "english": lambda v: v["family"] == "english",
            "wiki": lambda v: v["family"] == "wiki", "wiki47": lambda v: v["family"] == "wiki" and v["group"] == "test80",
            "all": lambda v: True}
    for name, sel in sets.items():
        cs = sorted(c for c, v in per.items() if sel(v))
        ms = [m for m in METH if all(m in per[c]["err"] for c in cs)]
        E = {m: np.array([per[c]["err"][m] for c in cs]) for m in ms}
        idx = rng.integers(0, len(cs), (NB, len(cs)))
        b = {}
        for m in ms:
            r = np.sqrt((E[m][idx] ** 2).mean(1))
            b[m] = dict(rms=float(np.sqrt((E[m] ** 2).mean())), ci95=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                        flops_median=float(np.median([per[c]["flops"][m] for c in cs])))
        tgt = float(np.median([per[c]["flops"]["target"] for c in cs]))
        # frontier: not dominated (another method with <= median FLOPs and lower rms)
        front = [m for m in ms if not any(b[o]["flops_median"] <= b[m]["flops_median"] and b[o]["rms"] < b[m]["rms"]
                                          for o in ms if o != m)]
        ratios = {}
        for a in ("A0_20k", "A0_1k", "A0_1.5B", "A0_7B"):
            if a not in ms:
                continue
            for o in ms:
                if o.startswith("A0"):
                    continue
                r = np.sqrt((E[a][idx] ** 2).mean(1) / (E[o][idx] ** 2).mean(1))
                ratios[f"{a}/{o}"] = dict(ratio=b[a]["rms"] / b[o]["rms"],
                                          ci95=[float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))],
                                          closer=f"{int(np.sum(np.abs(E[a]) < np.abs(E[o])))}/{len(cs)}")
        blocks[name] = dict(n=len(cs), methods=b, target_flops_median=tgt, target_rms=SEED_SD, frontier=front,
                            ratios=ratios)
    res = dict(checks=dict(new18_A0_full_reproduces_stored_max=max(chk), test80_sub05_reproduced_max=max(chk05)),
               blocks=blocks, per_corpus=per)
    json.dump(res, open(OUT / "compute_frontier.json", "w"), indent=1)
    print(json.dumps(res["checks"]))
    for name, bl in blocks.items():
        print(f"\n[{name}] n={bl['n']} target {bl['target_flops_median']:.2g} FLOPs; frontier {bl['frontier']}")
        for m, v in sorted(bl["methods"].items(), key=lambda kv: kv[1]["flops_median"]):
            print(f"  {m:9s} {v['flops_median']:9.2g}  rms {v['rms']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")
        for k, v in bl["ratios"].items():
            print(f"    {k:18s} {v['ratio']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] closer {v['closer']}")


if __name__ == "__main__":
    main()
