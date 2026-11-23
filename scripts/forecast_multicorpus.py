"""Predictions for the multicorpus level test (docs/plan/multicorpus_level_test_2026-10-15.md, section 3), computed from data
before any training run. Reads only training prefixes, ref_stats and reference curves of the new corpora, and the calibration
corpora (TinyStories, FineWeb-Edu, protein) with the constants the plan names. No outcome of a new corpus is read.

  python scripts/forecast_multicorpus.py --corpora code,pg19,pubmed,ja_wiki,abc --out results/mc/forecast.json

Per corpus X (dir DIR = data/mc/X/tok, which holds tok_bytes_train.npy and tok_bytes_ref.npy):
  j_half(X, D, K)   byte order where the share of realised histories falls to 0.5, K in {2, 210}, S = 2.79 (context_eventcount.j_half);
                    censored when the share is still >= 0.5 at the largest counted order (128), then j_half is 128
  A1/A2/A3/A0       level Delta_X = ln n_impl,X - ln n_impl,TS at 2^23: count unscaled / scaled by a_K / bracket / calibration mean
  B1/B2/B3/B4/B0    context exponent eps_X over 2^21-2^23: count growth / scaled by b_K / n-gram IG line / reference-LM IG line / mean
  C                 main forecaster (nu, n0 from the TinyStories bytes ladder) and info_nu bytes loss at each budget, per reference LM
Calibration fits (a_K, b_K, the IG lines) use TinyStories, FineWeb-Edu and protein only. A calibration corpus whose arrays are not
present is listed under missing_calibration and the fits that need it are null.
Writes created_utc and the SHA256 of every input file. Counts and n-gram curves are cached under results/mc/<corpus>/.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import context_eventcount as CE  # noqa: E402
import forecast_input_variants as V  # noqa: E402
import forecast_neff as FN  # noqa: E402
import ngram_curve_ts as NG  # noqa: E402

LN2 = np.log(2.0)
QWEN_PRIMARY = ("ja_wiki", "de_wiki", "zh_wiki")   # non-English text; SmolLM2-135M for English and code
KS_LIST = (2, 210)
# measured calibration values quoted in the plan (section 1 and section 3)
CAL = dict(ts=dict(eps=0.646, delta=0.0), fw=dict(eps=0.41, delta=-1.23), prot=dict(eps=0.20, delta=float(np.log(16.7 / 21.1))))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def input_files(d, valid=False):
    out = []
    for n in ("tok_bytes_train.npy", "tok_bytes_ref.npy", "train_bytes.npy", "ref_stats_bytes.npy") + (("tok_bytes_valid.npy",) if valid else ()):
        if os.path.exists(os.path.join(d, n)):
            out.append(os.path.join(d, n))
    return out


def ig(h, n1, n2):
    """Relative drop of the curve per doubling of context between n1 and n2 bytes."""
    return float((h(n1) - h(n2)) / (np.log2(n2 / n1) * h(n1)))


def ng_curve(name, d, k, res, ev=("tok_bytes_ref.npy", "ref_stats_bytes.npy")):
    f = os.path.join(res, name, f"ngram_{k}.json")
    if not os.path.exists(f):
        os.makedirs(os.path.dirname(f), exist_ok=True)
        tr = np.asarray(np.load(NG.find_array(d, ["tok_bytes_train.npy", "train_bytes.npy"]), mmap_mode="r")[: 2 ** k], np.uint64)
        va = np.asarray(np.load(NG.find_array(d, list(ev)), mmap_mode="r"), np.uint64)
        nev = min(2 ** 17, len(va) - NG.K)
        pos = np.sort(np.random.default_rng(0).choice(np.arange(NG.K, len(va)), nev, replace=False))
        json.dump(dict(D=2 ** k, n_eval=nev, rows=NG.ngram_curve(tr, va, pos, log=lambda *a: None)), open(f, "w"), indent=1)
    return {r["n"]: r["h_ng"] for r in json.load(open(f))["rows"]}


def load_refs(path, models):
    """{model key: HRef} for the models present in a merged refcurve file ({model: {rows}}) or a single-curve file."""
    if not path or not os.path.exists(path):
        return {}
    d = json.load(open(path))
    if "rows" in d:
        return {"single": FN.HRef(path)}
    return {m: FN.HRef(path, m) for m in (models or list(d)) if m in d}


def corpus_stats(name, d, budgets, res, refs, ev=("tok_bytes_ref.npy", "ref_stats_bytes.npy")):
    z = CE.count_corpus(name, d, budgets, os.path.join(res, name, "counts.npz"))
    z = np.load(z)
    jh = {f"K{K}": {k: CE.j_half(z, name, k, float(K)) for k in budgets} for K in KS_LIST}
    ks = np.array(budgets, float)
    eps_count = {K: float(np.polyfit(ks * LN2, np.log([jh[K][k][0] for k in budgets]), 1)[0]) for K in jh}
    ng = {k: ng_curve(name, d, k, res, ev) for k in budgets}
    ig_ng = {k: float((ng[k][8] - ng[k][24]) / (np.log2(3) * ng[k][8])) for k in budgets}
    out = dict(j_half={K: {str(k): dict(j_half=v[0], censored=v[1]) for k, v in jh[K].items()} for K in jh}, eps_count=eps_count,
               ig_ng={str(k): v for k, v in ig_ng.items()}, h_ng_8_24={str(k): [ng[k][8], ng[k][24]] for k in budgets},
               ig_ref={m: ig(hr.h, 8.0, 32.0) for m, hr in refs.items()},
               ref_competent={m: bool(hr.h(512.0) <= ng[budgets[-1]][24] - 0.3) for m, hr in refs.items()})
    out["_jh"] = jh
    return out


def fit_origin(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return float((x * y).sum() / (x * x).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="code,pg19,pubmed,ja_wiki,abc")
    ap.add_argument("--mc_root", default=os.path.join(ROOT, "data/mc"))
    ap.add_argument("--res", default=os.path.join(ROOT, "results/mc"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/mc/forecast.json"))
    ap.add_argument("--budgets", default="21,22,23")
    ap.add_argument("--ts_dir", default=os.path.join(ROOT, "data/text"))
    ap.add_argument("--fw_dir", default=os.environ.get("FW_TOK_DIR") or os.path.join(ROOT, "data/bfine/tok"))
    ap.add_argument("--prot_dir", default=os.path.join(ROOT, "data/bprot/tok"))
    ap.add_argument("--models", default="SmolLM2-135M,Qwen2.5-0.5B")
    a = ap.parse_args()
    budgets = [int(q) for q in a.budgets.split(",")]
    kmax = budgets[-1]
    models = a.models.split(",")
    sys.stdout.flush()

    cal_dirs = dict(ts=a.ts_dir, fw=a.fw_dir, prot=a.prot_dir)
    cal_ref = dict(ts=os.path.join(ROOT, "text/refcurve/refcurve.json"),
                   fw=os.path.join(ROOT, "results/bfine/refcurve/refcurve_fineweb.json#SmolLM2-135M"),
                   prot=os.path.join(ROOT, "results/bprot/refcurve_monotone.json#progen2-small"))
    cal, missing = {}, []
    for c, d in cal_dirs.items():
        if not any(os.path.exists(os.path.join(d, n)) for n in ("tok_bytes_train.npy", "train_bytes.npy")):
            missing.append(c)
            continue
        p, _, m = cal_ref[c].partition("#")
        refs = {m or "single": FN.HRef(p, m or None)} if os.path.exists(p) else {}
        # TinyStories has no ref_stats partition: its n-gram curve is scored on the valid split, as scripts/ngram_curve_ts.py always did
        cal[c] = corpus_stats(f"cal_{c}", d, budgets, a.res, refs, ev=("tok_bytes_ref.npy", "ref_stats_bytes.npy", "tok_bytes_valid.npy"))
        cal[c]["ref_key"] = next(iter(refs), None)
    res = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), budgets=budgets,
               constants=dict(S=CE.S_LAB, K_values=list(KS_LIST), calibration_measured=CAL),
               missing_calibration=missing, calibration={c: {k: v for k, v in s.items() if k != "_jh"} for c, s in cal.items()},
               inputs={}, corpora={})

    # fits on the calibration corpora only
    fits = {}
    if "ts" in cal:
        for K in KS_LIST:
            if "fw" in cal and "prot" in cal:
                A1 = {c: np.log(cal[c]["_jh"][f"K{K}"][kmax][0]) - np.log(cal["ts"]["_jh"][f"K{K}"][kmax][0]) for c in ("fw", "prot")}
                fits[f"a_K{K}"] = fit_origin([A1[c] for c in A1], [CAL[c]["delta"] for c in A1])
            if all(c in cal for c in ("fw", "prot")):
                B1 = [cal[c]["eps_count"][f"K{K}"] for c in ("ts", "fw", "prot")]
                fits[f"b_K{K}"] = fit_origin(B1, [CAL[c]["eps"] for c in ("ts", "fw", "prot")])
        if all(c in cal for c in ("fw", "prot")):
            eps = [CAL[c]["eps"] for c in ("ts", "fw", "prot")]
            x = [cal[c]["ig_ng"][str(kmax)] for c in ("ts", "fw", "prot")]
            fits["B3"] = dict(zip(("beta", "alpha"), map(float, np.polyfit(x, eps, 1))))
            if all(cal[c]["ig_ref"].get(cal[c]["ref_key"]) is not None for c in ("ts", "fw", "prot")):
                x = [cal[c]["ig_ref"][cal[c]["ref_key"]] for c in ("ts", "fw", "prot")]
                fits["B4"] = dict(zip(("beta", "alpha"), map(float, np.polyfit(x, eps, 1))))
            fits["B0"] = float(np.mean(eps))
            fits["A0"] = float(np.mean([CAL["fw"]["delta"], CAL["prot"]["delta"]]))
    res["fits"] = fits

    # the main forecaster's constants (TinyStories bytes ladder only)
    c_ts = FN.calibrate()[0]
    ts_info = V.ts_check()
    res["forecaster_constants"] = dict(nu=c_ts["nu"], n0=c_ts["n0"], info_gamma=ts_info["gamma"], info_A=ts_info["A"], info_K=ts_info["K"])

    for name in a.corpora.split(","):
        d = os.path.join(a.mc_root, name, "tok")
        refs = load_refs(os.path.join(a.res, name, "refcurve", "refcurve_fineweb.json"), models)
        primary = "Qwen2.5-0.5B" if name in QWEN_PRIMARY else "SmolLM2-135M"
        primary = primary if primary in refs else next(iter(refs), None)
        s = corpus_stats(name, d, budgets, a.res, refs)
        jh = s.pop("_jh")
        s["ref_primary"] = primary
        A = {}
        for K in KS_LIST:
            A1 = float(np.log(jh[f"K{K}"][kmax][0]) - np.log(cal["ts"]["_jh"][f"K{K}"][kmax][0])) if "ts" in cal else None
            A[f"A1_K{K}"] = A1
            if A1 is not None and f"a_K{K}" in fits:
                A[f"A2_K{K}"] = fits[f"a_K{K}"] * A1
        if all(A.get(f"A1_K{K}") is not None for K in KS_LIST):
            A["A3_bracket"] = sorted([A["A1_K2"], A["A1_K210"]])
        A["A0"] = fits.get("A0")
        B = {}
        for K in KS_LIST:
            B[f"B1_K{K}"] = s["eps_count"][f"K{K}"]
            if f"b_K{K}" in fits:
                B[f"B2_K{K}"] = fits[f"b_K{K}"] * s["eps_count"][f"K{K}"]
        if "B3" in fits:
            B["B3"] = fits["B3"]["alpha"] + fits["B3"]["beta"] * s["ig_ng"][str(kmax)]
        if "B4" in fits and primary in s["ig_ref"]:
            B["B4"] = fits["B4"]["alpha"] + fits["B4"]["beta"] * s["ig_ref"][primary]
        B["B0"] = fits.get("B0")
        lag = {}
        for an, av in A.items():
            for bn, bv in B.items():
                if isinstance(av, float) and isinstance(bv, float) and bv > 0 and an.startswith("A2"):
                    lag[f"{an}/{bn}"] = float(-av / (bv * LN2))
        C = {}
        for m, hr in refs.items():
            g, Acoef = V.power_fit(hr)
            C[m] = dict(
                main={str(k): FN.predict(hr, c_ts["nu"], c_ts["n0"], 0.0, 1.0, k) for k in budgets},
                info_nu={str(k): V.predict_info(hr, ts_info["gamma"], ts_info["A"], ts_info["K"], 1.0, k) for k in budgets},
                info_nu_own_gamma=dict(gamma=g, A=Acoef),
                curve_range_bpb=[float(hr.y.min()), float(hr.y.max())], curve_max_context_bytes=float(np.exp(hr.x.max())))
        res["corpora"][name] = dict(stats=s, level=A, exponent=B, lag_doublings_from_A2=lag, main_forecaster_bytes=C)
        if primary in refs:
            # eligibility (docs/plan/forecast_extension_2026-10-17.md): scored only if the primary reference LM at its full context
            # (largest context of its curve, ref_stats partition) is below the forecast's own learner loss at the largest budget
            ref_full = float(refs[primary].y[-1])
            fc_k = C[primary]["main"][str(kmax)]
            res["corpora"][name]["eligibility"] = dict(
                ref_model=primary, ref_bpb_full_context=ref_full, ref_full_context_bytes=float(np.exp(refs[primary].x[-1])),
                forecast_bpb_at_max_budget=fc_k, max_budget_log2=kmax, eligible=bool(ref_full < fc_k))
        res["inputs"][name] = {os.path.relpath(f, ROOT): sha_file(f) for f in input_files(d)}
        for m, hr in refs.items():
            res["inputs"][name][f"refcurve:{m}"] = hr.sha
    for c in cal:
        res["inputs"][f"cal_{c}"] = {f: sha_file(f) for f in input_files(cal_dirs[c], valid=True)}
    res["inputs"]["scripts"] = {n: sha_file(os.path.join(HERE, n)) for n in
                                ("forecast_multicorpus.py", "context_eventcount.py", "ngram_curve_ts.py", "forecast_input_variants.py", "forecast_neff.py")}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
