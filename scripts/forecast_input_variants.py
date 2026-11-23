"""Input variants of the B-FineWeb forecast that use fewer learned inputs (design note of 2026-10-13).

  python scripts/forecast_input_variants.py [--runs 'results/text_runs/bfine/*.json'] \
      [--out results/forecast_variants/bfine_variants.json]

Reads results/bfine/forecast.json (train-side b per tokenizer, the main point forecasts), the FineWeb-Edu reference curves
and the TinyStories calibration in scripts/forecast_neff.py. It writes a separate file and changes none of its inputs.

Variants, with the constants each takes from trained learning curves (all from TinyStories, results/text_runs/{v1,v1b,v1d}):
  main          L = h_ref(n0 (D b^c)^nu) on SmolLM2-135M, as in forecast.json                         nu, n0, c   (3)
  disjoint_ref  the same rule read on pythia-160m and gpt2, which never saw FineWeb-Edu                 nu, n0, c   (3)
  info_nu       nu = 1/(1 + gamma) and c = 1/nu - 1 = gamma, where gamma and the amplitude A come from the target's own
                information curve: h(n) - h(N_REF) = A n^-gamma fitted over WINDOW bytes of context. The one learner
                constant is K in D_eff |h'(n_eff)| = K, i.e. n_eff = (A gamma D_eff / K)^nu with D_eff = D b^c; K is
                fitted once on the TinyStories bytes ladder (2^20-2^24)                                  K           (1)
                No bytes learning curve sets nu or c; no tokenizer run sets c.
info_nu is read on each reference curve; on pythia-160m and gpt2 it uses no reference LM that saw the target corpus.
TinyStories check: K by leave-one-budget-out on the bytes ladder; every tokenizer cell is out of sample (c is not fitted).
--runs (optional): error of every variant on the FineWeb-Edu cells that have runs (mean over the seeds present).
"""
import argparse, glob, hashlib, json, os, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import forecast_neff as FN  # noqa: E402

WINDOW = (8.0, 64.0)   # bytes of context; covers the n_eff of the TinyStories bytes ladder (5-30 bytes) and the B cells
N_REF = 960.0          # bytes; inside every reference curve (TinyStories-33M reaches 985 bytes, FineWeb-Edu ~2300)
LN2 = np.log(2.0)


def power_fit(hr, window=WINDOW, n_ref=N_REF):
    """gamma, A of h(n) - h(n_ref) = A n^-gamma over 12 log-spaced contexts in the window (piecewise-linear h in log n)."""
    n = np.geomspace(window[0], window[1], 12)
    d = hr.h(n) - hr.h(n_ref)
    p = np.polyfit(np.log(n), np.log(d), 1)
    return float(-p[0]), float(np.exp(p[1]))


def info_n0(gamma, A, K):
    nu = 1.0 / (1.0 + gamma)
    return nu, (A * gamma / K) ** nu


def predict_info(hr, gamma, A, K, b, k):
    nu, n0 = info_n0(gamma, A, K)
    return FN.predict(hr, nu, n0, gamma, b, k)


def fit_K(hr, gamma, A, Lb, ks):
    """K from the bytes ladder with nu = 1/(1 + gamma) fixed: least squares in log n_eff, then K = A gamma / n0^(1/nu)."""
    nu = 1.0 / (1.0 + gamma)
    ln0 = float(np.mean([np.log(hr.neff(Lb[k])) - nu * k * LN2 for k in ks]))
    return A * gamma / np.exp(ln0) ** (1.0 / nu)


def ts_check():
    """info_nu on TinyStories: leave-one-budget-out on the bytes ladder; tokenizer cells with K from all bytes budgets."""
    hr = FN.HRef(FN.TS_HREF)
    R, B = FN.load_ts()
    L = FN.means(R)
    g, A = power_fit(hr)
    K = fit_K(hr, g, A, L["bytes"], FN.BYTES_KS)
    rb = {k: predict_info(hr, g, A, fit_K(hr, g, A, L["bytes"], [q for q in FN.BYTES_KS if q != k]), 1.0, k) - L["bytes"][k]
          for k in FN.BYTES_KS}
    cells = [dict(tok=t, k=k, err=predict_info(hr, g, A, K, B[t], k) - L[t][k])
             for t in FN.ONE_SPELLING for k in sorted(L[t]) if k in FN.RES_KS]
    e = np.array([c["err"] for c in cells])
    cal, _ = FN.calibrate()
    nu, n0 = info_n0(g, A, K)
    return dict(gamma=g, A=A, nu=nu, c=g, n0=n0, K=K,
                loo_bytes=dict(resid=rb, rms=float(np.sqrt(np.mean(np.square(list(rb.values())))))),
                tokenizers_out_of_sample=dict(n=len(cells), rms=float(np.sqrt(np.mean(e ** 2))), mean=float(e.mean()),
                                              maxabs=float(np.max(np.abs(e))), cells=cells),
                main_for_comparison=dict(nu=cal["nu"], c=cal["c"], loo_bytes_rms=cal["loo_bytes"]["rms"],
                                         loo_tokenizer_rms_20_24=cal["loo_c"]["rms_20_24"],
                                         H_jump_rms_20_24=cal["H_jump"]["rms_20_24"]))


def load_runs(globs):
    got = {}
    for f in sorted(x for g in globs for x in glob.glob(g)):
        j = json.load(open(f))
        k = int(round(np.log2(j["train_bytes"])))
        got.setdefault(f"{j['tok']}|{k}", []).append(float(j["bits_per_byte"]))
    return {key: dict(bpb=float(np.mean(v)), n_seeds=len(v)) for key, v in got.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast", default=str(ROOT / "results/bfine/forecast.json"))
    ap.add_argument("--refcurve", default=str(ROOT / "results/bfine/refcurve/refcurve_fineweb.json"))
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--out", default=str(ROOT / "results/forecast_variants/bfine_variants.json"))
    a = ap.parse_args()

    F = json.load(open(a.forecast))
    bpt = {t: v["b"] for t, v in F["B"]["bytes_per_token"].items()}
    refs = [r.split("#")[1] for r in F["header"]["reference_lms"]]
    hrs = {m: FN.HRef(a.refcurve, m) for m in refs}
    ts = ts_check()
    K = ts["K"]
    fits = {m: dict(zip(("gamma", "A"), power_fit(h))) for m, h in hrs.items()}
    for m, f in fits.items():
        f["nu"], f["n0"] = info_n0(f["gamma"], f["A"], K)
        f["c"] = f["gamma"]

    cells = {}
    for key, e in F["B"]["cells"].items():
        t, k = e["tok"], e["log2D"]
        b = bpt[t]
        row = dict(tok=t, stratum=e["stratum"], b=b, log2D=k, main=e["point"],
                   disjoint_ref=dict(zip(refs[1:], e["per_href"][1:])),
                   info_nu={m: predict_info(hrs[m], fits[m]["gamma"], fits[m]["A"], K, 1.0 if t == "bytes" else b, k)
                            for m in refs})
        cells[key] = row

    runs = load_runs(a.runs) if a.runs else {}
    score = None
    if runs:
        keys = [k for k in cells if k in runs]
        y = np.array([runs[k]["bpb"] for k in keys])
        cols = {"main (SmolLM2)": [cells[k]["main"] for k in keys]}
        for m in refs[1:]:
            cols[f"disjoint_ref ({m})"] = [cells[k]["disjoint_ref"][m] for k in keys]
        for m in refs:
            cols[f"info_nu ({m})"] = [cells[k]["info_nu"][m] for k in keys]
        score = dict(cells=keys, measured=y.tolist(), n_seeds={k: runs[k]["n_seeds"] for k in keys},
                     by_variant={n: dict(rms=float(np.sqrt(np.mean((np.array(p) - y) ** 2))),
                                         mean_err=float(np.mean(np.array(p) - y)),
                                         spearman=float(np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(y)))[0, 1])
                                         if len(keys) > 2 else None)
                                 for n, p in cols.items()})

    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    out = dict(
        inputs=dict(forecast=dict(path=a.forecast, sha256=sha(a.forecast)), refcurve=dict(path=a.refcurve, sha256=sha(a.refcurve)),
                    ts_refcurve=dict(path=FN.TS_HREF, sha256=sha(FN.TS_HREF)), runs=a.runs),
        constants_from_trained_curves=dict(main=["nu", "n0", "c"], disjoint_ref=["nu", "n0", "c"], info_nu=["K"],
                                           source="results/text_runs/{v1,v1b,v1d} (TinyStories)"),
        window_bytes=list(WINDOW), n_ref_bytes=N_REF, tinystories_check=ts, info_nu_by_reference=fits, cells=cells,
        score_on_runs=score)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=float)

    t = ts["tokenizers_out_of_sample"]
    print(f"TinyStories info_nu: gamma {ts['gamma']:.3f} -> nu {ts['nu']:.3f}, c {ts['c']:.3f} (main nu "
          f"{ts['main_for_comparison']['nu']:.3f}, c {ts['main_for_comparison']['c']:.3f}); K = {K:.4g}")
    print(f"  bytes ladder LOO rms {ts['loo_bytes']['rms']:.4f} (main {ts['main_for_comparison']['loo_bytes_rms']:.4f}); "
          f"tokenizer cells out of sample (n={t['n']}) rms {t['rms']:.4f}, mean {t['mean']:+.4f}, max {t['maxabs']:.3f} "
          f"(main LOO {ts['main_for_comparison']['loo_tokenizer_rms_20_24']:.4f}, H_jump {ts['main_for_comparison']['H_jump_rms_20_24']:.4f})")
    for m, f in fits.items():
        print(f"FineWeb-Edu {m:13s}: gamma {f['gamma']:.3f} -> nu {f['nu']:.3f}, c {f['c']:.3f}, n0 {f['n0']:.3g}")
    if score:
        print(f"score on {len(score['cells'])} FineWeb-Edu cells with runs:")
        for n, s in score["by_variant"].items():
            print(f"  {n:26s} rms {s['rms']:.4f}  mean err {s['mean_err']:+.4f}  spearman {s['spearman']:+.3f}")
    print("wrote", a.out, "sha256", sha(a.out))


if __name__ == "__main__":
    main()
