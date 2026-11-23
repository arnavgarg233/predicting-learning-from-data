"""Score the data-only forecast A0 with a stronger Qwen reference.

    python3 scripts/stronger_ref/score.py      # -> results/curve_forecast/stronger_ref/score.json
    python3 scripts/stronger_ref/score.py --sub7   # adds arm sub7 -> results/curve_forecast/stronger_ref7b/score.json
    python3 scripts/stronger_ref/score.py --sub14  # adds arms sub7 and sub14 -> results/curve_forecast/stronger_ref14b/score.json

Plain numpy/scipy (no model). Arms:
  full05  stored Qwen2.5-0.5B curves (20,000 targets); must reproduce the paper's A0 with the locked level map
  sub05   Qwen2.5-0.5B curves recomputed from the stored per-target losses on the seeded 1,000-target subset
  sub15   Qwen2.5-1.5B curves on the same subset (Colab results, results.jsonl)
  sub7    Qwen2.5-7B curves on the same subset (RunPod 2026-10-26, stronger_ref7b/runpod_raw); only with --sub7.
          full05, sub05 and sub15 are unchanged by it (sub7 bootstraps use their own rng stream).
  sub14   14B curves on the same subset (RunPod 2026-10-27, stronger_ref14b/runpod_raw, record key rows_<TAG14>_fp16,
          TAG14 from the env, default 14b); only with --sub14, which also scores sub7. The other arms are unchanged by it
          (sub14 bootstraps use a third rng stream).
Only Qwen-referenced corpora change between arms; SmolLM2-135M and TinyStories-33M curves are the stored ones in every arm.
Per arm the level map (level 'both', mode 'anch', shape 'gamma') and its ref-only fallback are refitted on the 13
development corpora with develop.py's own functions. A3 and the measured losses are read from the stored score files.
"""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts/curve_forecast"))
sys.path.insert(0, os.path.join(ROOT, "text/refcurve"))
import core as C  # noqa: E402
import develop as D  # noqa: E402
import refcurve_fineweb as RF  # noqa: E402  (summarize only; torch is imported only inside its main)

RES = os.path.join(ROOT, "results/curve_forecast")
OUT = os.path.join(RES, "stronger_ref")
SUB14 = "--sub14" in sys.argv
SUB7 = "--sub7" in sys.argv or SUB14
OUT7 = os.path.join(RES, "stronger_ref7b")
OUT14 = os.path.join(RES, "stronger_ref14b")
TAG14 = os.environ.get("TAG14", "14b")
ARMS = ("full05", "sub05", "sub15") + (("sub7",) if SUB7 else ()) + (("sub14",) if SUB14 else ())
QWEN_DEV = {"ja_wiki": "results/mc/ja_wiki", "de_wiki": "results/mc2/de_wiki", "zh_wiki": "results/mc2/zh_wiki"}
NB = 20000
SHAPE, LEVEL, MODE = "gamma", "both", "anch"
THRESH = 0.70


class Curve:
    """Same interface as forecast_neff.HRef, from rows [[n_tokens, bpb, se, ctx_bytes], ...]."""

    def __init__(self, rows, tag):
        rows = sorted(rows, key=lambda r: r[3])
        self.x = np.log([r[3] for r in rows])
        self.y = np.array([r[1] for r in rows])
        self.sha = tag

    def h(self, n):
        return np.interp(np.log(n), self.x, self.y)

    def neff(self, L):
        return float(np.exp(np.interp(-L, -self.y, self.x)))


def stored_rows(rdir):
    d = json.load(open(os.path.join(rdir, "refcurve_fineweb.json")))["Qwen2.5-0.5B"]
    return [[r["n_tokens"], r["bits_per_byte"], r["bpb_se"], r["context_bytes_exact_mean"]] for r in d["rows"]], d


def subset_rows(rdir, nt_expected):
    _, d = stored_rows(rdir)
    full = os.path.join(rdir, "Qwen2.5-0.5B_raw.npz")
    if os.path.exists(full):
        raw = np.load(full)
        nt = len(raw["tgt"])
        idx = np.sort(np.random.default_rng(20261025).choice(nt, min(1000, nt), replace=False))
        raw = {k: raw[k][idx] for k in ("loss", "doc", "tgtbytes", "ctxb")} | {"NS": raw["NS"]}
    else:  # released subset written by extract_sub05.py
        raw = np.load(os.path.join(rdir, "Qwen2.5-0.5B_sub1000.npz"))
        nt = int(raw["nt"])
    assert nt == nt_expected
    r = RF.summarize("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B", raw["loss"] / np.log(2), raw["doc"],
                     raw["tgtbytes"].astype(float), raw["ctxb"], list(raw["NS"]), d["n_docs_total"])
    return [[x["n_tokens"], x["bits_per_byte"], x["bpb_se"], x["context_bytes_exact_mean"]] for x in r["rows"]]


def load_inputs():
    jobs = {j["name"]: j for j in json.load(open(os.path.join(OUT, "jobs.json")))["jobs"]}
    colab = {}
    raw = os.path.join(OUT, "colab_raw")
    for p in sorted(f for f in os.listdir(raw) if f.startswith("rec_") and f.endswith(".json")):
        r = json.load(open(os.path.join(raw, p)))
        if "rows_1p5b_fp16" in r:
            colab[r["name"]] = r
    curves = {a: {} for a in ARMS}
    for name, j in jobs.items():
        rdir = os.path.join(ROOT, j["refdir"])
        curves["full05"][name] = Curve(stored_rows(rdir)[0], "full05")
        curves["sub05"][name] = Curve(subset_rows(rdir, j["nt"]), "sub05")
        if name in colab:
            curves["sub15"][name] = Curve(colab[name]["rows_1p5b_fp16"], "sub15")
    if SUB7:
        raw7 = os.path.join(OUT7, "runpod_raw")
        for p in sorted(f for f in os.listdir(raw7) if f.startswith("rec_") and f.endswith(".json")):
            r = json.load(open(os.path.join(raw7, p)))
            if "rows_7b_fp16" in r and r["name"] in jobs:
                curves["sub7"][r["name"]] = Curve(r["rows_7b_fp16"], "sub7")
    if SUB14:
        raw14 = os.path.join(OUT14, "runpod_raw")
        for p in sorted(f for f in os.listdir(raw14) if f.startswith("rec_") and f.endswith(".json")):
            r = json.load(open(os.path.join(raw14, p)))
            if f"rows_{TAG14}_fp16" in r and r["name"] in jobs:
                curves["sub14"][r["name"]] = Curve(r[f"rows_{TAG14}_fp16"], "sub14")
    return jobs, colab, curves


def test_corpora():
    """name -> dict(family, group, ref_model, ngram23, measured{23,25}, A3{23,25}, stored_A0{23,25}, hr_stored path)."""
    out = {}
    fc = {}
    for f in ("forecast.json", "stage2/forecast.json", "stage3/forecast.json"):
        for c, v in json.load(open(os.path.join(RES, f)))["corpora"].items():
            fc.setdefault(c, []).append((f, v))
    for grp, sf in (("test80", "score80.json"), ("stage3", "stage3/score_stage3.json")):
        for c, p in json.load(open(os.path.join(RES, sf)))["per_corpus"].items():
            cand = [v for f, v in fc[c] if (f.startswith("stage3") == (grp == "stage3"))]
            assert len(cand) == 1, (c, grp)
            v = cand[0]
            base = os.path.join(RES, "stage3" if grp == "stage3" else "", "" if grp == "stage3" else "")
            dirs = [os.path.join(RES, "stage3", c)] if grp == "stage3" else [os.path.join(RES, c), os.path.join(RES, "stage2", c)]
            rdir = next(os.path.join(d, "refcurve") for d in dirs if os.path.exists(os.path.join(d, "refcurve", "refcurve_fineweb.json")))
            out[(grp, c)] = dict(name=c, group=grp, family=p["family"], ref_model=p["ref_model"], ngram23=v["ref_stats"]["ngram23"],
                                 measured={k: p["measured"][k] for k in ("23", "25")},
                                 A3={k: p["forecast"][k]["A3"] for k in ("23", "25")},
                                 stored_A0={k: p["forecast"][k]["A0"] for k in ("23", "25")},
                                 refcurve=os.path.join(rdir, "refcurve_fineweb.json"))
            del base
    return out


def stats(hr, ngram23):
    st = C.ref_stats(hr)
    st["ngram23"] = ngram23 if ngram23 is not None else float("nan")
    return st


def fit_arm(dev, arm, curves, K):
    """Level coefficients refitted on the 13 development corpora with the arm's Qwen curves."""
    dv = {}
    for c, d in dev.items():
        e = dict(d)
        if c in QWEN_DEV:
            e["hr"] = curves[arm][c]
            e["st"] = stats(e["hr"], d["st"]["ngram23"])
        dv[c] = e
    names = list(dv)
    coef = C.fit_level(LEVEL, D.level_rows(dv, names, SHAPE, K, LEVEL, MODE))
    coef_ref = C.fit_level("ref", D.level_rows(dv, names, SHAPE, K, "ref", MODE))
    return coef, coef_ref, {c: dict(ref_full=dv[c]["st"]["ref_full"], gamma=dv[c]["st"]["gamma"]) for c in QWEN_DEV}


def forecast(hr, st, coef, coef_ref, k):
    nu, n0 = C.shape_params(SHAPE, st, KCONST)
    if all(np.isfinite(C.level_X(LEVEL, st))):
        return C.A0(hr, st, nu, n0, LEVEL, MODE, coef, k)
    return C.A0(hr, st, nu, n0, "ref", MODE, coef_ref, k)


def rms(v):
    v = np.asarray(v, float)
    return float(np.sqrt(np.mean(v ** 2)))


def paired_boot(ea, eb, rng):
    n = len(ea)
    i = rng.integers(0, n, (NB, n))
    ra, rb = np.sqrt((ea[i] ** 2).mean(1)), np.sqrt((eb[i] ** 2).mean(1))
    q = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]  # noqa: E731
    return dict(ratio=rms(ea) / rms(eb), ratio_ci95=q(ra / rb), diff=rms(ea) - rms(eb), diff_ci95=q(ra - rb))


KCONST = None


def main():
    global KCONST
    lk = json.load(open(os.path.join(RES, "locked.json")))
    KCONST = lk["constants"]
    jobs, colab, curves = load_inputs()
    dev = D.load_dev()
    T = test_corpora()
    out = dict(design="docs/analysis/stronger_reference_design.md", n_colab=len(colab),
               colab_missing=sorted(set(jobs) - set(colab)), arms={})
    rng_seed = 0
    for arm in ARMS:
        need = [c for c in QWEN_DEV]
        if any(c not in curves[arm] for c in need):
            out["arms"][arm] = dict(skipped="development curves missing")
            continue
        coef, coef_ref, devst = fit_arm(dev, arm, curves, KCONST)
        per = {}
        for (grp, c), t in T.items():
            if t["ref_model"] == "Qwen2.5-0.5B":
                if c not in curves[arm] or (arm != "full05" and c not in jobs):
                    continue
                hr = curves[arm][c]
                if arm == "full05":
                    hr = C.FN.HRef(t["refcurve"], "Qwen2.5-0.5B")
            else:
                hr = C.FN.HRef(t["refcurve"], t["ref_model"])
            st = stats(hr, t["ngram23"])
            a0 = {k: forecast(hr, st, coef, coef_ref, int(k)) for k in ("23", "25")}
            per[f"{grp}:{c}"] = dict(family=t["family"], group=grp, ref_model=t["ref_model"], ref_full=st["ref_full"], gamma=st["gamma"],
                                     ngram23=t["ngram23"], c_ratio=st["ref_full"] / t["ngram23"] if t["ngram23"] else None,
                                     A0=a0, err_A0={k: a0[k] - t["measured"][k] for k in a0},
                                     err_A3={k: t["A3"][k] - t["measured"][k] for k in a0},
                                     stored_A0=t["stored_A0"])
        out["arms"][arm] = dict(level_coef=coef, level_coef_ref_fallback=coef_ref, dev_qwen_stats=devst, per_corpus=per)
    # reproduction check: full05 with refitted coefficients equals the locked coefficients and the stored forecasts
    if "per_corpus" in out["arms"].get("full05", {}):
        f = out["arms"]["full05"]
        out["reproduction"] = dict(
            coef_locked=KCONST["level_coef"], coef_refit=f["level_coef"],
            max_abs_A0_minus_stored=max(abs(v["A0"][k] - v["stored_A0"][k]) for v in f["per_corpus"].values() for k in ("23", "25")))
    # summaries over the corpora present in every scored arm
    groups = {"code": lambda v: v["group"] == "test80" and v["family"] == "code",
              "english": lambda v: v["group"] == "test80" and v["family"] == "english",
              "wiki47": lambda v: v["group"] == "test80" and v["family"] == "wiki",
              "all80": lambda v: v["group"] == "test80", "stage3_16": lambda v: v["group"] == "stage3"}
    scored = [a for a in ARMS if "per_corpus" in out["arms"][a]]
    common = set.intersection(*[set(out["arms"][a]["per_corpus"]) for a in scored])
    summ = {}
    for k in ("23", "25"):
        rng = np.random.default_rng(rng_seed)
        rng7 = np.random.default_rng(rng_seed)   # separate stream: the sub7 comparisons leave the others' draws unchanged
        rng14 = np.random.default_rng(rng_seed)  # third stream for the sub14 comparisons
        sk = {}
        for g, sel in groups.items():
            keys = sorted(x for x in common if sel(out["arms"][scored[0]]["per_corpus"][x]))
            if not keys:
                continue
            E = {a: np.array([out["arms"][a]["per_corpus"][x]["err_A0"][k] for x in keys]) for a in scored}
            E["A3"] = np.array([out["arms"][scored[0]]["per_corpus"][x]["err_A3"][k] for x in keys])
            blk = dict(n=len(keys), rms={a: rms(e) for a, e in E.items()}, mean_err={a: float(np.mean(e)) for a, e in E.items()})
            if "sub15" in E:
                blk["sub15_vs_sub05"] = paired_boot(E["sub15"], E["sub05"], rng)
                blk["sub15_vs_A3"] = paired_boot(E["sub15"], E["A3"], rng)
                blk["sub05_vs_A3"] = paired_boot(E["sub05"], E["A3"], rng)
                blk["closer_than_A3"] = {a: int(np.sum(np.abs(E[a]) < np.abs(E["A3"]))) for a in scored}
            if "sub7" in E:
                blk["sub7_vs_sub05"] = paired_boot(E["sub7"], E["sub05"], rng7)
                blk["sub7_vs_sub15"] = paired_boot(E["sub7"], E["sub15"], rng7)
                blk["sub7_vs_A3"] = paired_boot(E["sub7"], E["A3"], rng7)
            if "sub14" in E:
                blk["sub14_vs_sub7"] = paired_boot(E["sub14"], E["sub7"], rng14)
                blk["sub14_vs_A3"] = paired_boot(E["sub14"], E["A3"], rng14)
                blk["sub14_vs_sub05"] = paired_boot(E["sub14"], E["sub05"], rng14)
                blk["sub14_vs_sub15"] = paired_boot(E["sub14"], E["sub15"], rng14)
            sk[g] = blk
        summ[f"budget_2_{k}"] = sk
    out["summary"] = summ
    # competence ratio c
    comp = {}
    for a in scored:
        pc = out["arms"][a]["per_corpus"]
        for g in ("wiki47", "stage3_16"):
            keys = [x for x in common if groups[g](pc[x])]
            cs = [pc[x]["c_ratio"] for x in keys]
            comp.setdefault(a, {})[g] = dict(n_ge_070=int(np.sum(np.array(cs) >= THRESH)), median_c=float(np.median(cs)),
                                             median_ref_full=float(np.median([pc[x]["ref_full"] for x in keys])))
    out["competence"] = comp
    if "sub15" in scored:
        d = {x: dict(dref=out["arms"]["sub15"]["per_corpus"][x]["ref_full"] - out["arms"]["sub05"]["per_corpus"][x]["ref_full"],
                     c05=out["arms"]["sub05"]["per_corpus"][x]["c_ratio"], c15=out["arms"]["sub15"]["per_corpus"][x]["c_ratio"])
             for x in common if out["arms"]["sub15"]["per_corpus"][x]["ref_model"] == "Qwen2.5-0.5B"}
        out["competence_change"] = dict(n=len(d), n_ref_lower=int(sum(v["dref"] < 0 for v in d.values())),
                                        median_dref=float(np.median([v["dref"] for v in d.values()])),
                                        crossed_below_070=sorted(x for x, v in d.items() if v["c05"] >= THRESH > v["c15"]),
                                        crossed_above_070=sorted(x for x, v in d.items() if v["c05"] < THRESH <= v["c15"]),
                                        per_corpus=d)
    if "sub7" in scored:
        d = {x: dict(dref=out["arms"]["sub7"]["per_corpus"][x]["ref_full"] - out["arms"]["sub05"]["per_corpus"][x]["ref_full"],
                     c05=out["arms"]["sub05"]["per_corpus"][x]["c_ratio"], c7=out["arms"]["sub7"]["per_corpus"][x]["c_ratio"])
             for x in common if out["arms"]["sub7"]["per_corpus"][x]["ref_model"] == "Qwen2.5-0.5B"}
        out["n_sub7"] = len(curves["sub7"])
        out["competence_change_sub7"] = dict(n=len(d), n_ref_lower=int(sum(v["dref"] < 0 for v in d.values())),
                                             median_dref=float(np.median([v["dref"] for v in d.values()])),
                                             crossed_below_070=sorted(x for x, v in d.items() if v["c05"] >= THRESH > v["c7"]),
                                             crossed_above_070=sorted(x for x, v in d.items() if v["c05"] < THRESH <= v["c7"]),
                                             per_corpus=d)
    if "sub14" in scored:
        d = {x: dict(dref=out["arms"]["sub14"]["per_corpus"][x]["ref_full"] - out["arms"]["sub7"]["per_corpus"][x]["ref_full"],
                     c7=out["arms"]["sub7"]["per_corpus"][x]["c_ratio"], c14=out["arms"]["sub14"]["per_corpus"][x]["c_ratio"])
             for x in common if out["arms"]["sub14"]["per_corpus"][x]["ref_model"] == "Qwen2.5-0.5B"}
        out["n_sub14"], out["tag14"] = len(curves["sub14"]), TAG14
        out["competence_change_sub14_vs_sub7"] = dict(n=len(d), n_ref_lower=int(sum(v["dref"] < 0 for v in d.values())),
                                                      median_dref=float(np.median([v["dref"] for v in d.values()])),
                                                      crossed_below_070=sorted(x for x, v in d.items() if v["c7"] >= THRESH > v["c14"]),
                                                      crossed_above_070=sorted(x for x, v in d.items() if v["c7"] < THRESH <= v["c14"]),
                                                      per_corpus=d)
    p = os.path.join(OUT14 if SUB14 else OUT7 if SUB7 else OUT, "score.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print("sha256", hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], "colab results", len(colab), "of", len(jobs))
    print("reproduction", json.dumps(out.get("reproduction"), default=float))


if __name__ == "__main__":
    main()
