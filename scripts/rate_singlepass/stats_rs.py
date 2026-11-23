"""Data statistics of one corpus for the single-pass rate study, measured before any training on it.

    python -u scripts/rate_singlepass/stats_rs.py --name C --data DIR --res RESDIR --model Qwen2.5-0.5B [--screen] [--nt 4000]

--screen  (small preparation) c = h_ref(512 tokens) / ngram23: reference pass at 512 tokens only, 1,000 targets.
full      reference curve at the 18 contexts of text/refcurve/refcurve_fineweb.py with --nt targets; gamma_ref and A from
          h(n) - h(960 bytes) = A n^-gamma over 8-64 bytes (scripts/forecast_input_variants.power_fit, the paper's estimator);
          ref_full = h at 512 tokens; ngram23 = best n-gram cross-entropy trained on 2^23 bytes (forecast_multicorpus.ng_curve,
          the paper's definition) and c = ref_full / ngram23; n-gram curves at 2^23 and 2^25 bytes, with gamma_ng the exponent of
          h_ng(n) - h_ng(n*) = A n^-gamma over n = 1 .. max(n* - 2, 2) bytes (n* the order with the lowest h_ng); lzma bits per byte
          (preset 9) of the first 2^23 bytes of the train partition.
Writes RESDIR/<C>/stats.json (or screen.json).
"""
import argparse
import json
import lzma
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
HF = {"Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B", "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M"}


def gamma_ng(h):
    h = np.asarray(h, float)
    ns = int(np.argmin(h))
    k = np.arange(1, max(ns - 1, 3))
    d = h[k] - h[ns]
    if ns < 3 or (d <= 0).any():
        return float("nan"), ns
    return float(-np.polyfit(np.log(k), np.log(d), 1)[0]), ns


def refpass(data, out, model, ns, nt, device):
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", os.path.join(ROOT, "text/refcurve/refcurve_fineweb.py"), "--data", data, "--out", out,
           "--models", HF[model], "--ns", ns, "--nt", str(nt), "--device", device]
    with open(os.path.join(out, "refcurve.log"), "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    p = os.path.join(out, "refcurve_fineweb.json")
    return p if os.path.exists(p) and model in json.load(open(p)) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--nt", type=int, default=4000)
    ap.add_argument("--screen_nt", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ng_budgets", default="23,25")
    ap.add_argument("--part", default="all", choices=["ng", "ref", "all"], help="ng: n-gram curves and lzma (CPU); ref: reference pass")
    a = ap.parse_args()
    import forecast_multicorpus as FM
    import forecast_neff as FN
    import forecast_input_variants as V
    res = os.path.join(a.res, a.name)
    os.makedirs(res, exist_ok=True)
    if a.screen:
        ng = FM.ng_curve(a.name, a.data, 23, a.res, ev=("ref_stats_bytes.npy",))
        g = float(min(v for v in ng.values() if np.isfinite(v)))
        if a.part == "ng":            # n-gram part only (CPU); the reference part is run separately on the GPU
            return
        p = refpass(a.data, os.path.join(res, "ref512"), a.model, "512", a.screen_nt, a.device)
        if p is None:
            out = dict(name=a.name, ok=False, reason="reference pass failed", ngram23=g)
        else:
            r = json.load(open(p))[a.model]["rows"][-1]
            out = dict(name=a.name, ok=True, ref512=r["bits_per_byte"], ngram23=g, c=r["bits_per_byte"] / g,
                       n_targets=json.load(open(p))[a.model]["n_targets"])
        json.dump(out, open(os.path.join(res, "screen.json"), "w"), indent=1)
        print(out, flush=True)
        return
    tok = os.path.join(a.data, "tok")
    if a.part == "ref":
        out = json.load(open(os.path.join(res, "stats_ng.json")))
        return ref_part(a, res, out, FN, V)
    out = dict(name=a.name, model=a.model)
    curves = {}
    for k in map(int, a.ng_budgets.split(",")):
        ng = FM.ng_curve(a.name, tok, k, a.res)
        h = [ng[n] for n in sorted(ng)]
        curves[k] = h
        gg, ns = gamma_ng(h)
        out[f"ngram{k}"] = float(min(v for v in h if np.isfinite(v)))
        out[f"gamma_ng{k}"], out[f"ng_nstar{k}"] = gg, ns
    raw = np.load(os.path.join(a.data, "train_bytes.npy"), mmap_mode="r")[: 2 ** 23].tobytes()
    out["lzma_bpb"] = 8 * len(lzma.compress(raw, preset=9)) / len(raw)
    out["ngram_curves"] = {str(k): v for k, v in curves.items()}
    json.dump(out, open(os.path.join(res, "stats_ng.json"), "w"), indent=1)
    if a.part == "ng":
        print({k: v for k, v in out.items() if k != "ngram_curves"}, flush=True)
        return
    ref_part(a, res, out, FN, V)


def ref_part(a, res, out, FN, V):
    p = refpass(a.data, os.path.join(res, "refcurve"), a.model, ",".join(map(str, [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])),
                a.nt, a.device)
    if p is None:
        out.update(ok=False, reason="reference pass failed")
    else:
        hr = FN.HRef(p, a.model)
        g, A = V.power_fit(hr)
        out.update(ok=True, gamma_ref=g, A_ref=A, ref_full=float(hr.y[-1]), c=float(hr.y[-1]) / out["ngram23"],
                   ref_n_targets=json.load(open(p))[a.model]["n_targets"],
                   ref_curve=[[float(np.exp(x)), float(y)] for x, y in zip(hr.x, hr.y)])
    json.dump(out, open(os.path.join(res, "stats.json"), "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("ref_curve", "ngram_curves")}, flush=True)


if __name__ == "__main__":
    main()
