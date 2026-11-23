"""Full-curve forecast vs pilot extrapolation (docs/plan/curve_forecast_vs_pilots_2026-10-17.md): shared model code.

Notation. A corpus c has a reference curve h_c(n) (reference-LM bits/byte with n bytes of context; forecast_neff.HRef) and
learner losses L_c(D) (test bits/byte after training on D = 2^k bytes). Budgets are passed as log2 k.

Shapes  S_c(D) = h_c(n0 D^nu):
  ts     nu, n0 from the TinyStories bytes ladder (forecast_neff.calibrate)
  pool   nu, n0 fitted on development corpora: least squares of the within-corpus-demeaned residuals L - S
  gamma  nu = 1/(1+gamma_c), n0 = (A_c gamma_c / K_TS)^nu, (gamma_c, A_c) = power fit of the corpus's own curve
         (forecast_input_variants.power_fit), K_TS from the TinyStories ladder (forecast_input_variants.ts_check)
Level (zero-training arm A0), X = reference bits/byte at full context ('ref'), best n-gram cross-entropy at 2^23 ('ngram'),
  or both ('both'); two modes:
    anch   the affine map gives the loss at the anchor 2^23 and the shape gives the change: L = alpha + beta . X + S(D) - S(2^23)
    curve  the affine map gives the offset from the curve: L = alpha + beta . X + S(D)
  alpha, beta by least squares over development corpora of the per-corpus mean over 2^21-2^23 of L - S (+ S(2^23) for anch).
Pilot hybrids (forms):
  off    level from the pilot(s): L(D) = c + S(D), c = mean over pilots of L(Dp) - S(Dp)
  neff   the pilot loss read on the curve: log n_eff = mean over pilots of log h^-1(L(Dp)) - nu log Dp; L(D) = h(n_eff D^nu)
Arms: A0 level + shape; A1 larger pilot + FineWeb-Edu measured curve (L(Dp) + FW(D) - FW(Dp)); A2 larger pilot + shape;
A3 two pilots + power law L = E + B D^-a with E by a floor rule; A4 two pilots + shape.
Floor rules for A3 (pilot-only information): zero (E = 0), const (E = e), frac (E = kappa L(D2)); then E <= 0.9 L(D2).
If the larger pilot is not below the smaller one, A3 returns L(D2).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, SCRIPTS)
import forecast_neff as FN  # noqa: E402
import forecast_input_variants as V  # noqa: E402

LN2 = np.log(2.0)
SHAPES = ("ts", "pool", "gamma")
LEVELS = ("ref", "ngram", "both")
MODES = ("anch", "curve")
K_ANCHOR = 23
FORMS = ("off", "neff")
FLOORS = ("zero", "const", "frac")
E_CAP = 0.9                       # E <= E_CAP * L(D2)
CONST_GRID = np.round(np.arange(0.0, 1.501, 0.01), 4)
FRAC_GRID = np.round(np.arange(0.0, 0.9001, 0.01), 4)
POOL_NU = np.round(np.arange(0.20, 1.2001, 0.01), 4)
POOL_LOG10N0 = np.round(np.arange(-7.0, 1.0001, 0.025), 4)
LEVEL_KS = (21, 22, 23)


# ------------------------------------------------------------------ inputs
def ref_stats(hr):
    g, A = V.power_fit(hr)
    return dict(ref_full=float(hr.y[-1]), gamma=g, A=A)


def best_ngram(path):
    if not os.path.exists(path):
        return float("nan")
    v = [r["h_ng"] for r in json.load(open(path))["rows"] if np.isfinite(r["h_ng"])]
    return float(min(v)) if v else float("nan")


def ts_constants():
    cal = FN.calibrate()[0]
    info = V.ts_check()
    return dict(nu_ts=cal["nu"], n0_ts=cal["n0"], K_ts=info["K"], gamma_ts_info=info["gamma"])


# ------------------------------------------------------------------ shapes
def shape_params(shape, st, C):
    """(nu, n0) for a corpus with ref stats st under constants C."""
    if shape == "ts":
        return C["nu_ts"], C["n0_ts"]
    if shape == "pool":
        return C["nu_pool"], C["n0_pool"]
    if shape == "gamma":
        nu = 1.0 / (1.0 + st["gamma"])
        return nu, (st["A"] * st["gamma"] / C["K_ts"]) ** nu
    raise ValueError(shape)


def S(hr, nu, n0, k):
    return float(hr.h(n0 * (2.0 ** k) ** nu))


def fit_pool(corpora):
    """corpora: list of (hr, {k: L}). Grid least squares of within-corpus-demeaned L - h(n0 D^nu)."""
    nu = POOL_NU[:, None, None]
    ln0 = (POOL_LOG10N0 * np.log(10.0))[None, :, None]
    tot = np.zeros((len(POOL_NU), len(POOL_LOG10N0)))
    for hr, L in corpora:
        ks = np.array(sorted(L), float)
        y = np.array([L[k] for k in sorted(L)])
        lnn = ln0 + nu * (ks * LN2)[None, None, :]
        r = y[None, None, :] - np.interp(lnn, hr.x, hr.y)
        r = r - r.mean(-1, keepdims=True)
        tot += (r ** 2).sum(-1)
    i, j = np.unravel_index(np.argmin(tot), tot.shape)
    return float(POOL_NU[i]), float(10.0 ** POOL_LOG10N0[j])


# ------------------------------------------------------------------ level (A0)
def level_X(level, st):
    if level == "ref":
        return [st["ref_full"]]
    if level == "ngram":
        return [st["ngram23"]]
    return [st["ref_full"], st["ngram23"]]


def fit_level(level, rows):
    """rows: list of (st, mean residual L - S over LEVEL_KS). Returns coefficient vector [alpha, beta...] (least squares)."""
    X = np.array([[1.0] + level_X(level, st) for st, _ in rows])
    y = np.array([r for _, r in rows])
    return [float(v) for v in np.linalg.lstsq(X, y, rcond=None)[0]]


def level_target(hr, nu, n0, L, mode):
    """Per-corpus regression target from measured losses L {k: bpb} at LEVEL_KS."""
    a = S(hr, nu, n0, K_ANCHOR) if mode == "anch" else 0.0
    return float(np.mean([L[k] - S(hr, nu, n0, k) + a for k in LEVEL_KS]))


def A0(hr, st, nu, n0, level, mode, coef, k):
    x = level_X(level, st)
    if not all(np.isfinite(x)):
        return float("nan")
    a = S(hr, nu, n0, K_ANCHOR) if mode == "anch" else 0.0
    return float(coef[0] + np.dot(coef[1:], x) + S(hr, nu, n0, k) - a)


# ------------------------------------------------------------------ pilot arms
def A1(Lp, kp, k, fw):
    return float(Lp + fw[k] - fw[kp])


def hybrid(hr, nu, n0, pilots, k, form):
    """pilots: {k: L}. One pilot -> A2, two pilots -> A4."""
    if form == "off":
        c = np.mean([L - S(hr, nu, n0, q) for q, L in pilots.items()])
        return float(c + S(hr, nu, n0, k))
    if form == "neff":
        ln = np.mean([np.log(hr.neff(L)) - nu * q * LN2 for q, L in pilots.items()])
        return float(hr.h(np.exp(ln + nu * k * LN2)))
    raise ValueError(form)


def floor_E(rule, par, L2):
    E = 0.0 if rule == "zero" else (par if rule == "const" else par * L2)
    return min(E, E_CAP * L2)


def A3(pilots, k, rule, par):
    (k1, L1), (k2, L2) = sorted(pilots.items())
    if not L2 < L1:
        return float(L2)
    E = floor_E(rule, par, L2)
    a = np.log((L1 - E) / (L2 - E)) / ((k2 - k1) * LN2)
    return float(E + (L2 - E) * 2.0 ** (-a * (k - k2)))


def fit_floor(rule, cases):
    """cases: list of (pilots, k, measured). Best grid parameter by rms."""
    if rule == "zero":
        return 0.0
    grid = CONST_GRID if rule == "const" else FRAC_GRID
    err = [np.sqrt(np.mean([(A3(p, k, rule, g) - m) ** 2 for p, k, m in cases])) for g in grid]
    return float(grid[int(np.argmin(err))])


# ------------------------------------------------------------------ compute
LEARNER = dict(d=256, nl=4, V=256, T=512, bs=32, eval_every=100, val_chunks=2000)
REF_NS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
REF_PARAMS = {"SmolLM2-135M": 134.5e6, "Qwen2.5-0.5B": 494.0e6}


def learner_flops(steps, eval_chunks=0):
    """Training (6N + 6 nl T d per token) plus periodic validation (2N + 2 nl T d per token) plus the final test pass."""
    p = LEARNER
    N = 12 * p["nl"] * p["d"] ** 2 + 2 * p["V"] * p["d"]
    att = p["nl"] * p["T"] * p["d"]
    train = steps * p["bs"] * p["T"] * (6 * N + 6 * att)
    val = (steps // p["eval_every"]) * p["val_chunks"] * p["T"] * (2 * N + 2 * att)
    test = eval_chunks * p["T"] * (2 * N + 2 * att)
    return float(train + val + test)


def ref_flops(model, nt=20000):
    """One forward pass per (target, n): nt * sum(n + 1) tokens at 2 * params FLOPs per token (upper estimate)."""
    return float(2 * REF_PARAMS[model] * nt * sum(n + 1 for n in REF_NS))
