"""14B fallback scoring: every Qwen arm restricted to the same 500 of the seeded 1,000 subset targets per corpus.

    python3 scripts/stronger_ref/score_half.py   # -> results/curve_forecast/stronger_ref14b/score_half500.json

The 500 positions are np.sort(default_rng(20261027).choice(1000, 500)) within each corpus's sorted 1,000-target subset (the
worker's SUB_N=500 switch). Curves are recomputed with refcurve_fineweb.summarize from per-target losses:
  h05  Qwen2.5-0.5B  (stored per-target losses, target and context bytes)
  h7   Qwen2.5-7B    (stronger_ref7b/runpod_raw/npz, 66 corpora)
  h14  14B           (stronger_ref14b/runpod_raw/npz/<name>_<TAG14>.npz), if present
The 1.5B lacks per-target losses for 7 corpora (incl. the 3 development corpora), so it is not rescored here.
Level map refitted per arm exactly as in score.py (same functions); A3 and measured losses as in score.py.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import score as S  # noqa: E402

RF, ROOT, RES = S.RF, S.ROOT, S.RES
TAG14 = S.TAG14
N_HALF, SEED_HALF = 500, 20261027


def sub05_raw(j):
    rdir = os.path.join(ROOT, j["refdir"])
    full = os.path.join(rdir, "Qwen2.5-0.5B_raw.npz")
    if os.path.exists(full):
        raw = np.load(full)
        idx = np.sort(np.random.default_rng(20261025).choice(len(raw["tgt"]), 1000, replace=False))
        return {k: raw[k][idx] for k in ("loss", "doc", "tgtbytes", "ctxb")} | {"NS": raw["NS"]}
    raw = np.load(os.path.join(rdir, "Qwen2.5-0.5B_sub1000.npz"))
    return {k: raw[k] for k in ("loss", "doc", "tgtbytes", "ctxb", "NS")}


def rows(loss, s, k, n_docs):
    r = RF.summarize("x", "x", loss[k] / np.log(2), s["doc"][k], s["tgtbytes"][k].astype(float), s["ctxb"][k], list(s["NS"]), n_docs)
    return [[x["n_tokens"], x["bits_per_byte"], x["bpb_se"], x["context_bytes_exact_mean"]] for x in r["rows"]]


def load_curves():
    jobs = {j["name"]: j for j in json.load(open(os.path.join(S.OUT, "jobs.json")))["jobs"]}
    k = np.sort(np.random.default_rng(SEED_HALF).choice(1000, N_HALF, replace=False))
    curves = {"h05": {}, "h7": {}, "h14": {}}
    n14 = 0
    for name, j in jobs.items():
        s = sub05_raw(j)
        rec7 = json.load(open(os.path.join(S.OUT7, "runpod_raw", f"rec_{name}.json")))
        z7 = np.load(os.path.join(S.OUT7, "runpod_raw", "npz", f"{name}_7b.npz"))
        assert np.array_equal(z7["doc"], s["doc"]), name
        nd = rec7["n_docs_total"]
        curves["h05"][name] = S.Curve(rows(s["loss"], s, k, nd), "h05")
        curves["h7"][name] = S.Curve(rows(z7["loss"], s, k, nd), "h7")
        p14 = os.path.join(S.OUT14, "runpod_raw", "npz", f"{name}_{TAG14}.npz")
        if os.path.exists(p14):
            z = np.load(p14)
            assert np.array_equal(z["k"], k) and np.array_equal(z["doc"], s["doc"][k]) and np.array_equal(z["tgt"], z7["tgt"][k]), name
            loss = np.zeros_like(z7["loss"])
            loss[k] = z["loss"]
            curves["h14"][name] = S.Curve(rows(loss, s, k, nd), "h14")
            n14 += 1
    return jobs, curves, n14


def main():
    S.KCONST = json.load(open(os.path.join(RES, "locked.json")))["constants"]
    jobs, curves, n14 = load_curves()
    arms = ["h05", "h7"] + (["h14"] if all(c in curves["h14"] for c in S.QWEN_DEV) else [])
    dev = S.D.load_dev()
    T = S.test_corpora()
    out = dict(n_targets=N_HALF, seed=SEED_HALF, n_h14=n14, tag14=TAG14, arms={})
    for arm in arms:
        coef, coef_ref, devst = S.fit_arm(dev, arm, curves, S.KCONST)
        per = {}
        for (grp, c), t in T.items():
            if t["ref_model"] == "Qwen2.5-0.5B":
                if c not in curves[arm] or c not in jobs:
                    continue
                hr = curves[arm][c]
            else:
                hr = S.C.FN.HRef(t["refcurve"], t["ref_model"])
            st = S.stats(hr, t["ngram23"])
            a0 = {kk: S.forecast(hr, st, coef, coef_ref, int(kk)) for kk in ("23", "25")}
            per[f"{grp}:{c}"] = dict(family=t["family"], group=grp, ref_model=t["ref_model"], ref_full=st["ref_full"],
                                     c_ratio=st["ref_full"] / t["ngram23"] if t["ngram23"] else None,
                                     A0=a0, err_A0={kk: a0[kk] - t["measured"][kk] for kk in a0},
                                     err_A3={kk: t["A3"][kk] - t["measured"][kk] for kk in a0})
        out["arms"][arm] = dict(level_coef=coef, level_coef_ref_fallback=coef_ref, dev_qwen_stats=devst, per_corpus=per)
    groups = {"code": lambda v: v["group"] == "test80" and v["family"] == "code",
              "english": lambda v: v["group"] == "test80" and v["family"] == "english",
              "wiki47": lambda v: v["group"] == "test80" and v["family"] == "wiki",
              "all80": lambda v: v["group"] == "test80", "stage3_16": lambda v: v["group"] == "stage3"}
    common = set.intersection(*[set(out["arms"][a]["per_corpus"]) for a in arms])
    summ = {}
    for kk in ("23", "25"):
        rng = np.random.default_rng(0)
        sk = {}
        for g, sel in groups.items():
            keys = sorted(x for x in common if sel(out["arms"][arms[0]]["per_corpus"][x]))
            E = {a: np.array([out["arms"][a]["per_corpus"][x]["err_A0"][kk] for x in keys]) for a in arms}
            E["A3"] = np.array([out["arms"][arms[0]]["per_corpus"][x]["err_A3"][kk] for x in keys])
            blk = dict(n=len(keys), rms={a: S.rms(e) for a, e in E.items()}, mean_err={a: float(np.mean(e)) for a, e in E.items()},
                       closer_than_A3={a: int(np.sum(np.abs(E[a]) < np.abs(E["A3"]))) for a in arms})
            blk["h7_vs_h05"] = S.paired_boot(E["h7"], E["h05"], rng)
            blk["h7_vs_A3"] = S.paired_boot(E["h7"], E["A3"], rng)
            if "h14" in E:
                blk["h14_vs_h7"] = S.paired_boot(E["h14"], E["h7"], rng)
                blk["h14_vs_A3"] = S.paired_boot(E["h14"], E["A3"], rng)
                blk["h14_vs_h05"] = S.paired_boot(E["h14"], E["h05"], rng)
            sk[g] = blk
        summ[f"budget_2_{kk}"] = sk
    out["summary"] = summ
    p = os.path.join(S.OUT14, "score_half500.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    for b, sk in summ.items():
        for g in ("wiki47", "all80"):
            x = sk[g]
            print(b, g, x["n"], {a: round(v, 4) for a, v in x["rms"].items()},
                  {c: [round(v, 3) for v in [x[c]["ratio"]] + x[c]["ratio_ci95"]] for c in x if c.endswith(("_vs_h7", "_vs_A3"))})


if __name__ == "__main__":
    main()
