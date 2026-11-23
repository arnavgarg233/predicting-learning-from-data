"""Shared loaders and statistics for the applicability analysis.
Stored data only, CPU, deterministic."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "applicability")
NB = 20000


def J(p):
    return json.load(open(p if os.path.isabs(p) else os.path.join(ROOT, p)))


def rms(e):
    return float(np.sqrt(np.mean(np.square(e))))


def rank(v):
    return np.argsort(np.argsort(v)).astype(float)


def spearman(x, y):
    return float(np.corrcoef(rank(np.asarray(x)), rank(np.asarray(y)))[0, 1])


def partial_spearman(y, x, z):
    Z = np.column_stack([np.ones(len(y)), rank(z)])
    ry = rank(y) - Z @ np.linalg.lstsq(Z, rank(y), rcond=None)[0]
    rx = rank(x) - Z @ np.linalg.lstsq(Z, rank(x), rcond=None)[0]
    d = np.sqrt((ry @ ry) * (rx @ rx))
    return float(ry @ rx / d) if d > 0 else float("nan")


def ols(Xcols, y):
    X = np.column_stack([np.ones(len(y))] + list(Xcols))
    return np.linalg.lstsq(X, y, rcond=None)[0]


def pred(coef, Xcols, n):
    X = np.column_stack([np.ones(n)] + list(Xcols))
    return X @ coef


def loo_errors(cols, y):
    """Leave-one-corpus-out errors of the OLS line y ~ cols (cols = [] gives the kind mean)."""
    n = len(y); e = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        c = ols([v[m] for v in cols], y[m])
        e[i] = pred(c, [v[i:i + 1] for v in cols], 1)[0] - y[i]
    return e


def ratio(ea, eb, seed=0):
    n = len(ea)
    idx = np.random.default_rng(seed).integers(0, n, (NB, n))
    br = np.sqrt((ea[idx] ** 2).mean(1) / np.maximum((eb[idx] ** 2).mean(1), 1e-30))
    lo, hi = np.percentile(br, [2.5, 97.5])
    return dict(ratio=rms(ea) / rms(eb), ci95=[float(lo), float(hi)],
                label="REAL positive" if hi < 1 else ("REAL negative" if lo > 1 else "tie"),
                closer=f"{int(np.sum(np.abs(ea) < np.abs(eb)))}/{n}")


def boot_ci(fn, n, seed=0, nb=2000):
    idx = np.random.default_rng(seed).integers(0, n, (nb, n))
    v = np.array([fn(i) for i in idx]); v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def scaled_mad(v):
    v = np.asarray(v); return float(1.4826 * np.median(np.abs(v - np.median(v))))


# ---------------------------------------------------------------- corpus table
SET_ASIDE_EN = J("results/fwd_english/fall_line.json")["calibration"]["set_aside"]


def fall_rows():
    """All 189 stored rows (test80, forward18, external45, stage3, forward30) with n_docs and c."""
    d = J("results/reanalysis/fall_diagnosis.json")
    q = d["Q2"]["per_corpus"]
    out = []
    for r in d["rows"]:
        out.append(dict(corpus=r["corpus"], set=r["set"], kind=r["kind"], x=r["x"], fall=r["fall"], ngram23=r["ngram23"],
                        ref_full=r["ref_full"], c=r["ref_full"] / r["ngram23"], n_docs=q[r["corpus"]]["n_docs"],
                        boot_sd_x=r["boot_sd_x"], L23=r["L23"], r2_flag=r["r2_flag"]))
    return out


def english_forward_rows():
    """The 24 scored forward English corpora (20 clean + 4 set aside by the frozen x rule)."""
    S = J("results/fwd_english/score.json")
    out = []
    for c in S["all_scored"]["corpora"]:
        p = S["per_corpus"][c]
        d = os.path.join(ROOT, p["dir"])
        fi = J(os.path.join(d, "forecast_i.json"))["corpora"][c]["ref_stats"]
        rc = None
        for sub in ("pod", os.path.join("raw", "curve_forecast")):
            f = os.path.join(d, sub, c, "refcurve", "SmolLM2-135M.json")
            if os.path.exists(f):
                rc = J(f); break
        assert rc is not None, c
        out.append(dict(corpus=c, set="forward_english", kind="english", x=p["own_fall"], fall=p["fall"], ngram23=fi["ngram23"],
                        ref_full=fi["ref_full"], c=fi["ref_full"] / fi["ngram23"], n_docs=int(rc["n_docs"]),
                        n_docs_total=int(rc.get("n_docs_total", rc["n_docs"])), L23=p["L"]["23"],
                        clean=c in S["primary_clean"]["corpora"]))
    return out


# calibration sets of the frozen own-fall lines (the kind's own), and the matching held-out sets
CAL = dict(web={"external45"}, english={"test80", "forward18", "external45"}, code={"test80"}, wiki={"test80"})
HELD = dict(web={"forward30"}, english={"forward_english"}, code={"forward18", "external45"}, wiki={"forward18", "stage3"})
