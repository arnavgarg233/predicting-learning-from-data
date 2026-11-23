"""Score the frozen external forecast F_ext on open-sci-ref.

    python3 scripts/opensciref_forecast/score.py   # -> results/opensciref_forecast/scores.json

Needs calibration.json (DataDecide only) and the SmolLM2 CommonCorpus curve. Reads the stored open-sci-ref learner
evaluations (no new learner inference). numpy only, CPU, < 0.5 GB.
"""
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "opensciref"))
sys.path.insert(0, os.path.join(ROOT, "text", "refcurve"))
import forecast as FC  # noqa: E402
import refcurve_fineweb as RF  # noqa: E402

OSR = os.path.join(ROOT, "results/curve_forecast/opensciref")
RES = os.path.join(ROOT, "results/opensciref_forecast")
CORPORA = FC.CORPORA
NBOOT, SEED = 20000, 20261104
SUBSEED, NSUB = 20261025, 1000
SMOL_PARAMS = 134_515_008          # SmolLM2-135M parameter count
POS_PER_TARGET = int(sum(RF.NS))   # 1,788 token positions per target over the 18 contexts


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def curve_rows(npz, idx=None):
    z = np.load(npz)
    loss, doc, tbytes, ctxb, ns = z["loss"], z["doc"], z["tgtbytes"], z["ctxb"], z["NS"]
    if idx is not None:
        loss, doc, tbytes, ctxb = loss[idx], doc[idx], tbytes[idx], ctxb[idx]
    s = RF.summarize("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M", loss / np.log(2), doc, tbytes.astype(float), ctxb,
                     list(ns), int(doc.max()) + 1)
    return s["rows"], int(len(loss)), int(len(loss)) * int(np.sum(ns))


def ref_curves(variant):
    """variant 'sub1000' (frozen) or 'full4000' (secondary). CommonCorpus: the new 1,000-target SmolLM2 curve."""
    out = {}
    for c in CORPORA:
        if c == "commoncorpus":
            npz = os.path.join(RES, "refcurve/commoncorpus/SmolLM2-135M_raw.npz")
            rows, nt, pos = curve_rows(npz)
        else:
            npz = os.path.join(OSR, "refcurve", c, "SmolLM2-135M_raw.npz")
            idx = None
            if variant == "sub1000":
                idx = np.sort(np.random.default_rng(SUBSEED).choice(4000, NSUB, replace=False))
            rows, nt, pos = curve_rows(npz, idx)
        h = FC.HRef(rows)
        g, A = h.power_fit()
        out[c] = dict(h=h, r=float(h.y[-1]), gamma=g, A=A, n_targets=nt, positions=pos)
    return out


def repro_check():
    """summarize() on all 4,000 stored targets must give the stored curve."""
    worst = 0.0
    for c in CORPORA:
        if c == "commoncorpus":
            continue
        rows, _, _ = curve_rows(os.path.join(OSR, "refcurve", c, "SmolLM2-135M_raw.npz"))
        stored = json.load(open(os.path.join(OSR, "refcurve", c, "SmolLM2-135M.json")))["rows"]
        worst = max(worst, max(abs(a["bits_per_byte"] - b["bits_per_byte"]) for a, b in zip(rows, stored)))
    return worst


def boot(va, vb, rng):
    va, vb = np.abs(va), np.abs(vb)
    n = len(va)
    dr = rng.integers(0, n, size=(NBOOT, n))
    rms = lambda v: np.sqrt(np.mean(v ** 2, axis=-1))  # noqa: E731
    ra, rb = rms(va[dr]), rms(vb[dr])
    return dict(n=n, rms_a=float(rms(va)), rms_b=float(rms(vb)), ratio=float(rms(va) / rms(vb)),
                ratio_ci95=[float(np.quantile(ra / rb, 0.025)), float(np.quantile(ra / rb, 0.975))],
                diff=float(rms(va) - rms(vb)),
                diff_ci95=[float(np.quantile(ra - rb, 0.025)), float(np.quantile(ra - rb, 0.975))],
                a_closer=int(np.sum(va < vb)))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pairacc(p, y):
    i, j = np.triu_indices(len(p), 1)
    return float(np.mean(np.sign(p[i] - p[j]) == np.sign(y[i] - y[j])))


def score_size(size, fi, cal, REF):
    Y = FC.load_evals(size, fi)
    ad = json.load(open(os.path.join(OSR, "design_addendum.json")))
    cs = [c for c in CORPORA if c in Y]
    run = {c: fi["runs"][f"{size}/{c}"] for c in cs}
    T = {c: run[c]["main_iter"] for c in cs}
    pil = {c: tuple(ad["pilots"].get(f"{size}/{c}", FC.PILOTS)) for c in cs}
    const = {c: [t for t in FC.SCORE_ITERS if t in Y[c] and t not in pil[c]] for c in cs}
    whole = {c: const[c] + [T[c]] for c in cs}
    k = cal["sizes"][size]
    lk, a, b, Dcal = k["log10K"], k["a"], k["b"], k["D_cal_bytes"]

    def S(c, D):
        R = REF[c]
        nu = 1.0 / (1.0 + R["gamma"])
        logn = nu * (np.log(R["A"] * R["gamma"]) - lk * np.log(10.0) + np.log(D))
        return float(np.interp(logn, R["h"].x, R["h"].y))

    def Dbytes(c, t):
        return t * run[c]["tokens_per_iter"] * fi["corpora"][c]["bytes_per_token"]

    pred = {}
    for c in cs:
        ts = sorted(Y[c])
        P = {"F_ext": {t: a + b * REF[c]["r"] + S(c, Dbytes(c, t)) - S(c, Dcal) for t in ts},
             "F_ext_flat": {t: a + b * REF[c]["r"] for t in ts}}
        # constant: mean of the other corpora at the same checkpoint ('main' matched by label)
        P["C"] = {}
        for t in ts:
            if t == T[c]:
                v = [Y[q][T[q]] for q in cs if q != c]
            else:
                v = [Y[q][t] for q in cs if q != c and t in Y[q]]
            P["C"][t] = float(np.mean(v))
        for name, (t1, t2) in (("P2e", FC.PILOTS_EARLY), ("P2", pil[c])):
            if t1 in Y[c] and t2 in Y[c]:
                L1, L2 = Y[c][t1], Y[c][t2]
                e = np.log(L1 / L2) / np.log(t2 / t1) if L2 < L1 else 0.0
                P[name] = {t: float(L2 * (t / t2) ** (-e)) for t in ts}
        pred[c] = P
    arms = ["F_ext", "F_ext_flat", "C", "P2e", "P2"]

    def errs(arm, rng_name):
        out = []
        for c in cs:
            if arm not in pred[c]:
                out.append(np.nan)
                continue
            tt = {"whole": whole[c], "const": const[c], "final": [T[c]]}[rng_name]
            e = np.array([pred[c][arm][t] - Y[c][t] for t in tt])
            out.append(float(np.sqrt(np.mean(e ** 2))))
        return np.array(out)

    res = dict(size=size, corpora=cs, scored_whole={c: whole[c] for c in cs}, calibration=k | {"cells": None},
               ref={c: {q: REF[c][q] for q in ("r", "gamma", "A", "n_targets", "positions")} for c in cs},
               predicted={c: {a_: {str(t): v for t, v in pred[c][a_].items()} for a_ in pred[c]} for c in cs},
               measured={c: {str(t): v for t, v in sorted(Y[c].items())} for c in cs},
               final_signed={a_: {c: (pred[c][a_][T[c]] - Y[c][T[c]]) if a_ in pred[c] else None for c in cs} for a_ in arms},
               per_corpus={}, arms={}, paired={}, ranking={}, jackknife={})
    rng = np.random.default_rng(SEED)
    for rn in ("whole", "const", "final"):
        E = {a_: errs(a_, rn) for a_ in arms}
        res["per_corpus"][rn] = {a_: dict(zip(cs, E[a_].tolist())) for a_ in arms}
        for a_ in arms:
            v = E[a_][np.isfinite(E[a_])]
            res["arms"][f"{a_}|{rn}"] = dict(rms=float(np.sqrt(np.mean(v ** 2))), n=int(len(v)))
        for bname in ("P2e", "C", "P2"):
            for aname in ("F_ext",) + (("F_ext_flat",) if bname == "P2e" else ()):
                m = np.isfinite(E[aname]) & np.isfinite(E[bname])
                res["paired"][f"{aname}|{bname}|{rn}"] = boot(E[aname][m], E[bname][m], rng)
        if rn == "whole":
            m = np.isfinite(E["F_ext"]) & np.isfinite(E["P2e"])
            idx = np.flatnonzero(m)
            jk = {}
            for d in idx:
                keep = [i for i in idx if i != d]
                va, vb = E["F_ext"][keep], E["P2e"][keep]
                jk[cs[d]] = float(np.sqrt(np.mean(va ** 2)) / np.sqrt(np.mean(vb ** 2)))
            res["jackknife"]["F_ext/P2e whole, corpus dropped"] = jk
            keep = [i for i in idx if cs[i] != "commoncorpus"]
            res["paired"]["F_ext|P2e|whole|no_commoncorpus"] = boot(E["F_ext"][keep], E["P2e"][keep], rng)
            keepC = [i for i in range(len(cs)) if cs[i] != "commoncorpus"]
            res["paired"]["F_ext|C|whole|no_commoncorpus"] = boot(E["F_ext"][keepC], E["C"][keepC], rng)
    yf = np.array([Y[c][T[c]] for c in cs])
    for a_ in arms:
        m = np.array([a_ in pred[c] for c in cs])
        if m.sum() >= 3:
            p = np.array([pred[c][a_][T[c]] for c in np.array(cs)[m]])
            res["ranking"][a_] = dict(n=int(m.sum()), pairwise_acc=pairacc(p, yf[m]), spearman=spearman(p, yf[m]))
    cs6 = [i for i, c in enumerate(cs) if c != "commoncorpus"]
    p = np.array([pred[cs[i]]["F_ext"][T[cs[i]]] for i in cs6])
    res["ranking"]["F_ext_no_commoncorpus"] = dict(n=6, pairwise_acc=pairacc(p, yf[cs6]), spearman=spearman(p, yf[cs6]))
    return res


def compute(fi, REF):
    out = {}
    pos = {c: REF[c]["positions"] for c in CORPORA}
    ref_flops = {c: 2.0 * SMOL_PARAMS * pos[c] for c in CORPORA}
    out["reference"] = dict(model="SmolLM2-135M", params=SMOL_PARAMS, positions_per_target=POS_PER_TARGET,
                            targets_per_corpus={c: REF[c]["n_targets"] for c in CORPORA}, positions=pos,
                            flops_per_corpus=ref_flops, rule="2 x params x token positions processed")
    params = json.load(open(os.path.join(RES, "learner_params.json"))) if os.path.exists(os.path.join(RES, "learner_params.json")) else {"0.13b": 0.13e9, "0.4b": 0.4e9, "source": "nominal"}
    for size in FC.SIZES:
        N = params[size]
        tpi = np.mean([fi["runs"][f"{size}/{c}"]["tokens_per_iter"] for c in CORPORA])
        f = lambda it: 6.0 * N * it * tpi  # noqa: E731
        rf = float(np.mean(list(ref_flops.values())))
        out[size] = dict(learner_params=N, params_source=params.get("source", "config"), tokens_per_iter=float(tpi),
                         one_iteration_flops=f(1), P2e_flops_to_4k=f(4000), P2_flops_to_28k=f(28000),
                         full_run_flops=f(np.mean([fi["runs"][f"{size}/{c}"]["main_iter"] for c in CORPORA])),
                         ref_flops_mean=rf, P2e_over_ref=f(4000) / rf, P2_over_ref=f(28000) / rf,
                         iterations_matching_ref=rf / f(1))
    return out


def main():
    fi = json.load(open(os.path.join(OSR, "forecast_inputs.json")))
    cal = json.load(open(os.path.join(RES, "calibration.json")))
    out = dict(design="docs/analysis/opensciref_forecast.md", calibration_sha256=sha(os.path.join(RES, "calibration.json")),
               forecast_inputs_sha256=sha(os.path.join(OSR, "forecast_inputs.json")),
               repro_max_abs_bpb_full4000=repro_check(), variants={})
    for variant in ("sub1000", "full4000"):
        REF = ref_curves(variant)
        out["variants"][variant] = {s: score_size(s, fi, cal, REF) for s in FC.SIZES}
        if variant == "sub1000":
            out["compute"] = compute(fi, REF)
    p = os.path.join(RES, "scores.json")
    json.dump(out, open(p, "w"), indent=1)
    print("repro", out["repro_max_abs_bpb_full4000"])
    for v, d in out["variants"].items():
        for s, r in d.items():
            print(f"== {v} {s}")
            for k, x in r["arms"].items():
                print(f"   {k:20s} {x['rms']:.4f} (n={x['n']})")
            for k, x in r["paired"].items():
                print(f"   {k:34s} ratio {x['ratio']:.2f} ({x['ratio_ci95'][0]:.2f}-{x['ratio_ci95'][1]:.2f}) "
                      f"diff {x['diff']:+.4f} ({x['diff_ci95'][0]:+.4f} to {x['diff_ci95'][1]:+.4f}) closer {x['a_closer']}/{x['n']}")
            print("   ranking", {k: (round(x["pairwise_acc"], 2), round(x["spearman"], 2)) for k, x in r["ranking"].items()})
            print("   jackknife", {k: round(x, 2) for k, x in r["jackknife"]["F_ext/P2e whole, corpus dropped"].items()})
            print("   final signed F_ext", {c: round(x, 3) for c, x in r["final_signed"]["F_ext"].items()})
            print("   ref r", {c: round(x["r"], 3) for c, x in r["ref"].items()})
    print("compute", json.dumps(out["compute"], indent=0)[:1500])
    print("wrote", p, sha(p))


if __name__ == "__main__":
    main()
