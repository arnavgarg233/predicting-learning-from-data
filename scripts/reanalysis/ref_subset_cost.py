"""Does a reference pass 20 times cheaper keep the forecast's accuracy on the
SmolLM2-referenced corpora (code and English)? The stronger-reference test already showed it for Wikipedia (Qwen2.5-0.5B,
1,000 of 20,000 targets: r.m.s. 0.387 against 0.398).

    python scripts/reanalysis/ref_subset_cost.py   -> results/reanalysis/ref_subset_cost.json

For each code and English test corpus, the SmolLM2-135M curve is recomputed from the stored per-target losses on the
seeded 1,000-target subset used for Qwen (seed 20261025), with refcurve_fineweb.summarize. A0 is then recomputed with the
locked constants (shape 'gamma', level 'both', anchor 2^23), unchanged; nothing is refitted. Check: the full 20,000
targets must reproduce the stored A0.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/curve_forecast"))
sys.path.insert(0, str(ROOT / "text/refcurve"))
import core as C  # noqa: E402
import refcurve_fineweb as RF  # noqa: E402

RES = ROOT / "results/curve_forecast"


class Curve:
    def __init__(self, rows):
        rows = sorted(rows, key=lambda r: r["context_bytes_exact_mean"])
        self.x = np.log([r["context_bytes_exact_mean"] for r in rows])
        self.y = np.array([r["bits_per_byte"] for r in rows])

    def h(self, n):
        return np.interp(np.log(n), self.x, self.y)


def a0(hr, ngram, K, k):
    st = C.ref_stats(hr)
    st["ngram23"] = ngram
    nu = 1.0 / (1.0 + st["gamma"])
    n0 = (st["A"] * st["gamma"] / K["K_ts"]) ** nu
    return C.A0(hr, st, nu, n0, "both", "anch", K["level_coef"], k)


def main():
    lk = json.load(open(RES / "locked.json"))
    K = lk["constants"]
    s80 = json.load(open(RES / "score80.json"))["per_corpus"]
    FC = {**json.load(open(RES / "forecast.json"))["corpora"], **json.load(open(RES / "stage2/forecast.json"))["corpora"]}
    out = {}
    for c, p in sorted(s80.items()):
        if p["ref_model"] != "SmolLM2-135M":
            continue
        rdir = RES / c / "refcurve" if (RES / c / "refcurve").exists() else RES / "stage2" / c / "refcurve"
        d = json.load(open(rdir / "refcurve_fineweb.json"))["SmolLM2-135M"]
        if (rdir / "SmolLM2-135M_raw.npz").exists():
            raw = np.load(rdir / "SmolLM2-135M_raw.npz")
            nt = len(raw["tgt"])
            idx = np.sort(np.random.default_rng(20261025).choice(nt, min(1000, nt), replace=False))
        else:  # released subset written by scripts/stronger_ref/extract_sub1000.py (same seeded 1,000 targets)
            raw = np.load(rdir / "SmolLM2-135M_sub1000.npz")
            nt = int(raw["nt"]); idx = np.arange(len(raw["loss"]))
        r = RF.summarize("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M", raw["loss"][idx] / np.log(2), raw["doc"][idx],
                         raw["tgtbytes"][idx].astype(float), raw["ctxb"][idx], list(raw["NS"]), d["n_docs_total"])
        g = FC[c]["ref_stats"]["ngram23"]
        full = a0(Curve(d["rows"]), g, K, 25)
        sub = a0(Curve(r["rows"]), g, K, 25)
        y = p["measured"]["25"]
        out[c] = dict(family=p["family"], n_targets=nt, stored_A0=p["forecast"]["25"]["A0"], full_A0=full, sub_A0=sub,
                      err_full=full - y, err_sub=sub - y, err_A3=p["err"]["25"]["A3"])
    rms = lambda v: float(np.sqrt(np.mean(np.square(v))))  # noqa: E731
    summ = {}
    for f in ("code", "english"):
        cs = [c for c in out if out[c]["family"] == f]
        summ[f] = dict(n=len(cs), rms_full=rms([out[c]["err_full"] for c in cs]), rms_sub1000=rms([out[c]["err_sub"] for c in cs]),
                       rms_A3=rms([out[c]["err_A3"] for c in cs]))
    chk = max(abs(v["full_A0"] - v["stored_A0"]) for v in out.values())
    flops = dict(SmolLM2_20000=C.ref_flops("SmolLM2-135M", 20000), SmolLM2_1000=C.ref_flops("SmolLM2-135M", 1000),
                 Qwen05_20000=C.ref_flops("Qwen2.5-0.5B", 20000), Qwen05_1000=C.ref_flops("Qwen2.5-0.5B", 1000),
                 Qwen15_1000=float(2 * 1.54e9 * 1000 * sum(n + 1 for n in C.REF_NS)),
                 Qwen7_1000=float(2 * 7.62e9 * 1000 * sum(n + 1 for n in C.REF_NS)),
                 learner_params=float(12 * C.LEARNER["nl"] * C.LEARNER["d"] ** 2 + 2 * C.LEARNER["V"] * C.LEARNER["d"]))
    res = dict(check_full_reproduces_stored_max_abs=chk, summary=summ, flops=flops, per_corpus=out)
    json.dump(res, open(ROOT / "results/reanalysis/ref_subset_cost.json", "w"), indent=1)
    print("check", chk)
    print(json.dumps(summ, indent=1))
    print(json.dumps(flops, indent=1))


if __name__ == "__main__":
    main()
