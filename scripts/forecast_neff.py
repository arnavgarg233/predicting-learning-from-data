"""Effective-context forecaster (docs/plan/prospective_forecast_2026-10-12.md, sections 2 and 3A/3B).

    L(tok, D) = h_ref(n0 * (D * b^c)^nu)

h_ref(n): reference LM bits/byte with n bytes of context (refcurve.json rows). D: training bytes. b: bytes per token.
nu, n0 are fitted on the TinyStories bytes ladder 2^20-2^24 (2^26 left out: capacity-limited, 0.611 forecast vs 0.715
measured). c is fitted on the one-spelling tokenizers (BPE-512/2048/8192, wpair2/wpair3) at 2^21-2^23, the budgets all
five share. All constants come from results/text_runs/{v1,v1b,v1d} and are recomputed deterministically on each call.

Usage
  python scripts/forecast_neff.py --calibrate                       # constants, reproduction table, interval parts
  python scripts/forecast_neff.py --href fw_lm1.json fw_lm2.json fw_lm3.json --bpt bpt.json --budgets 21 22 23 \
      [--betas betas.json] [--gammas gammas.json] [--ctx 512] --out forecast.json
  python scripts/forecast_neff.py --midscale --href fw_lm1.json [...] --gpt2_bpt 4.3 [--betas betas.json] --out a.json

bpt.json: {tok: b} or {tok: {"b": b, "stratum": "S1"|"S2"|"anchor"}}. The first --href is the primary curve (point
forecast); the spread across all --href curves is the systematic width. betas.json / gammas.json: {tok: value or
{window: value}} including "bytes" (and "gpt2" for A2); Cagnetta's alpha = gamma / (2 beta), primary window tok_1_200.
"""
import argparse, glob, hashlib, json, os
import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS_HREF = os.path.join(ROOT, "text/refcurve/refcurve.json")
TS_TAGS = ("v1", "v1b", "v1d")
ONE_SPELLING = ("bpe512", "bpe2048", "bpe8192", "wpair2", "wpair3")
BYTES_KS = (20, 21, 22, 23, 24)     # 2^26 excluded: capacity limit of the d256 learner
C_KS = (21, 22, 23)                 # budgets for the c fit (2^21, 2^23 for wpair; 2^22 for BPE)
RES_KS = (20, 21, 22, 23, 24)       # budgets on which leave-one-out residuals are scored
Z80 = 1.2815516                     # two-sided 80% normal quantile
LN2 = np.log(2.0)
# Drift interval: the level rule (constant c) implies zero drift, TinyStories measured about -0.1 per doubling, so the
# interval spans [min(0, TS), max(0, TS)] widened on both sides by 80% loss-noise propagation: a cell's log2 m carries
# SIGMA_CELL_BPB / s(k) of noise from the tokenizer cell and the same from the bytes ladder, s(k) = |dL/dlog2 D| of the
# forecast bytes curve at that budget. SIGMA_CELL_BPB = the TinyStories per-run seed SD (0.0065), rounded.
SIGMA_CELL_BPB = 0.0065


# ---------------------------------------------------------------- reference curve
class HRef:
    """Piecewise-linear h(log n) through the refcurve rows (context_bytes_exact_mean, bits_per_byte)."""

    def __init__(self, path, model=None):
        """path: a refcurve JSON with 'rows', or a merged file {model: {rows: ...}} read with model=<key>."""
        d = json.load(open(path))
        if model is not None:
            d = d[model]
        rows = sorted(d["rows"], key=lambda r: r["context_bytes_exact_mean"])
        self.path = path if model is None else f"{path}#{model}"
        self.x = np.log([r["context_bytes_exact_mean"] for r in rows])
        self.y = np.array([r["bits_per_byte"] for r in rows])
        self.sha = hashlib.sha256(open(path, "rb").read()).hexdigest()

    def h(self, n):
        return np.interp(np.log(n), self.x, self.y)

    def neff(self, L):
        """Context (bytes) at which the reference model reaches loss L (clamped to the measured range)."""
        return float(np.exp(np.interp(-L, -self.y, self.x)))

    def slope_logn(self, logn):
        """|dh/dlog n| from a smoothed (midpoint-interpolated) piecewise slope."""
        s = -np.diff(self.y) / np.diff(self.x)
        xm = 0.5 * (self.x[1:] + self.x[:-1])
        return float(np.exp(np.interp(logn, xm, np.log(np.maximum(s, 1e-9)))))


def predict(hr, nu, n0, c, b, k):
    """Forecast bits/byte at D = 2^k bytes for a tokenizer with b bytes/token."""
    return float(hr.h(n0 * (2.0 ** k * b ** c) ** nu))


# ---------------------------------------------------------------- TinyStories runs
def load_ts(tags=TS_TAGS):
    """R[tok][k] = list of bits/byte over seeds; B[tok] = eval bytes/token."""
    R, B = {}, {}
    for tag in tags:
        for f in sorted(glob.glob(os.path.join(ROOT, f"results/text_runs/{tag}/*.json"))):
            j = json.load(open(f))
            k = int(round(np.log2(j["train_bytes"])))
            R.setdefault(j["tok"], {}).setdefault(k, []).append(j["bits_per_byte"])
            B[j["tok"]] = j["eval_bytes"] / j["eval_tokens"]
    return R, B


def means(R):
    return {t: {k: float(np.mean(v)) for k, v in r.items()} for t, r in R.items()}


def fit_bytes(hr, Lb, ks):
    """nu, n0 from a least-squares line of log n_eff on log D."""
    p = np.polyfit(np.array(ks) * LN2, [np.log(hr.neff(Lb[k])) for k in ks], 1)
    return float(p[0]), float(np.exp(p[1]))


def fit_c(hr, nu, n0, L, B, toks, ks):
    cells = [(t, k) for t in toks for k in ks if k in L[t]]
    f = lambda c: sum((predict(hr, nu, n0, c, B[t], k) - L[t][k]) ** 2 for t, k in cells)
    return float(minimize_scalar(f, bounds=(-1.0, 2.0), method="bounded", options=dict(xatol=1e-6)).x)


# ---------------------------------------------------------------- measured multiplier from a bytes ladder
# Interpolation rule (B2, B3, B4, A1, A2): the measured bytes ladder is read piecewise linearly in log2 D between
# measured budgets. There is no extrapolation: a budget outside the ladder, or a loss outside the ladder's loss range,
# is out of range (NaN), reported and not scored.
INTERP_RULE = ("bytes ladder read piecewise linearly in (log2 D, bits/byte) between measured budgets; no extrapolation; "
               "cells needing a budget or a loss outside the measured ladder are out of range: reported, not scored")


def ladder_bpb(ladder, k):
    """Bytes ladder bits/byte at log2 D = k (NaN outside the measured budgets)."""
    ks = np.array(sorted(ladder), float); ys = np.array([ladder[q] for q in sorted(ladder)])
    if not ks[0] - 1e-9 <= k <= ks[-1] + 1e-9:
        return float("nan")
    return float(np.interp(k, ks, ys))


def measured_log2m(ladder, L, k, censor=False):
    """log2 m such that the bytes ladder at 2^k * m equals L (the shift of a tokenizer cell onto the bytes curve).
    NaN if L is outside the ladder's loss range. With censor=True returns (value, flag): flag 'low' if L is above the
    ladder's highest loss (m below the ladder), 'high' if below its lowest loss, else None."""
    ks = np.array(sorted(ladder), float); ys = np.array([ladder[q] for q in sorted(ladder)])
    flag = "low" if L > ys.max() else ("high" if L < ys.min() else None)
    if flag is not None or not np.all(np.diff(ys) < 0):
        v = float("nan")
    else:
        v = float(np.interp(-L, -ys, ks) - k)
    return (v, flag) if censor else v


# ---------------------------------------------------------------- rival: constant-K information rule
def fit_K(hr, Lb, ks):
    """K = geometric mean over the bytes ladder of D * |h'(n_eff)|, h' per byte of context."""
    v = []
    for k in ks:
        n = hr.neff(Lb[k])
        v.append(np.log(2.0 ** k * hr.slope_logn(np.log(n)) / n))
    return float(np.exp(np.mean(v))), [float(x) for x in v]


def predict_K(hr, K, c, b, k):
    """Solve D_eff * |h'(n)| = K for n, D_eff = 2^k b^c, then read h_ref(n)."""
    De = 2.0 ** k * b ** c
    g = lambda ln: np.log(De * hr.slope_logn(ln) / np.exp(ln)) - np.log(K)
    lo, hi = hr.x[0] - 3, hr.x[-1] + 3
    if g(lo) * g(hi) > 0:
        return float(hr.h(np.exp(lo if g(lo) < 0 else hi)))
    return float(hr.h(np.exp(brentq(g, lo, hi))))


# ---------------------------------------------------------------- calibration on TinyStories
def calibrate(ts_href=TS_HREF):
    hr = HRef(ts_href)
    R, B = load_ts()
    L = means(R)
    nu, n0 = fit_bytes(hr, L["bytes"], BYTES_KS)
    c = fit_c(hr, nu, n0, L, B, ONE_SPELLING, C_KS)
    out = dict(nu=nu, n0=n0, c=c, bpt_ts={t: B[t] for t in ("bytes",) + ONE_SPELLING})
    out["neff_bytes"] = {k: hr.neff(L["bytes"][k]) for k in BYTES_KS}

    # reproduction: fit on 2^20-2^22, forecast 2^23, 2^24 (and the capacity-limited 2^26)
    nu3, n03 = fit_bytes(hr, L["bytes"], (20, 21, 22))
    out["bytes_oos_fit_20_22"] = dict(nu=nu3, n0=n03, **{f"err_2^{k}": predict(hr, nu3, n03, 0, 1, k) - L["bytes"][k]
                                                       for k in (23, 24, 26)})
    out["bytes_2^26"] = dict(forecast=predict(hr, nu, n0, 0, 1, 26), measured=L["bytes"][26])

    # leave-one-out over the bytes ladder
    rb = []
    for k in BYTES_KS:
        a, m = fit_bytes(hr, L["bytes"], [q for q in BYTES_KS if q != k])
        rb.append(predict(hr, a, m, 0, 1, k) - L["bytes"][k])
    out["loo_bytes"] = dict(resid={k: r for k, r in zip(BYTES_KS, rb)}, rms=float(np.sqrt(np.mean(np.square(rb)))))

    # leave-one-tokenizer-out for c; residuals on the fit budgets (design numbers) and on 2^20-2^24
    loo_c, rc, rc_all, rshift = {}, [], [], []
    ladder_ts = {k: L["bytes"][k] for k in BYTES_KS}
    for t in ONE_SPELLING:
        ct = fit_c(hr, nu, n0, L, B, [q for q in ONE_SPELLING if q != t], C_KS)
        loo_c[t] = ct
        for k in sorted(L[t]):
            if k not in RES_KS:
                continue
            e = predict(hr, nu, n0, ct, B[t], k) - L[t][k]
            rc_all.append(dict(tok=t, k=k, err=e))
            if k in C_KS:
                rc.append(e)
            rshift.append(ladder_bpb(ladder_ts, k + ct * np.log2(B[t])) - L[t][k])
    ea = np.array([r["err"] for r in rc_all])
    out["loo_c"] = dict(c=loo_c, rms_fit_budgets=float(np.sqrt(np.mean(np.square(rc)))),
                        maxabs_fit_budgets=float(np.max(np.abs(rc))), rms_20_24=float(np.sqrt(np.mean(ea ** 2))),
                        maxabs_20_24=float(np.max(np.abs(ea))), cells=rc_all,
                        rms_shift_on_measured_ladder=float(np.sqrt(np.nanmean(np.square(rshift)))),
                        n_shift_in_ladder_range=int(np.sum(np.isfinite(rshift))))

    # H_jump: no fitted tokenizer constant. If the learner's effective context in tokens follows the bytes law in token
    # units, n_tok = n0 (D/b)^nu, and each token spans b bytes, then n_bytes = n0 (b^(1/nu - 1) D)^nu: c = 1/nu - 1.
    cj = 1.0 / nu - 1.0
    ej = [predict(hr, nu, n0, cj, B[t], k) - L[t][k] for t in ONE_SPELLING for k in sorted(L[t]) if k in RES_KS]
    ef = [predict(hr, nu, n0, c, B[t], k) - L[t][k] for t in ONE_SPELLING for k in sorted(L[t]) if k in RES_KS]
    out["H_jump"] = dict(c=cj, n_cells=len(ej), rms_20_24=float(np.sqrt(np.mean(np.square(ej)))),
                         maxabs_20_24=float(np.max(np.abs(ej))), mean_20_24=float(np.mean(ej)),
                         fitted_c_in_sample_rms_20_24=float(np.sqrt(np.mean(np.square(ef)))),
                         fitted_c_in_sample_maxabs_20_24=float(np.max(np.abs(ef))),
                         note="secondary hypothesis, stated in the forecast before training; s_loo for its intervals = its rms on 2^20-2^24 "
                              "(no tokenizer constant is fitted, so this is out-of-sample for c)")

    # rivals c = 0 and c = 1 on the fit cells
    for cr in (0.0, 1.0):
        e = [abs(predict(hr, nu, n0, cr, B[t], k) - L[t][k]) for t in ONE_SPELLING for k in C_KS if k in L[t]]
        out[f"rival_c{int(cr)}_abs_err"] = [float(min(e)), float(max(e))]

    # seed SD (pooled over all two-seed cells 2^20-2^24, bytes and one-spelling tokenizers)
    sds = [np.var(R[t][k], ddof=1) for t in ("bytes",) + ONE_SPELLING for k in R[t] if k in RES_KS and len(R[t][k]) > 1]
    out["seed_sd"] = float(np.sqrt(np.mean(sds)))

    # constant-K information rule: fitted K and the drift of D |h'(n_eff)| per doubling
    K, logs = fit_K(hr, L["bytes"], BYTES_KS)
    out["constK"] = dict(K=K, drift_log2_per_doubling=float(np.polyfit(BYTES_KS, np.array(logs) / LN2, 1)[0]))

    # drift of log2 m (measured shift onto the bytes ladder) per doubling, 2^21 -> 2^23, per tokenizer and per seed
    dr, dr_seed = {}, []
    for t in ONE_SPELLING:
        m21, m23 = measured_log2m(ladder_ts, L[t][21], 21), measured_log2m(ladder_ts, L[t][23], 23)
        dr[t] = dict(log2m_21=m21, log2m_23=m23, per_doubling=(m23 - m21) / 2)
        for s in range(2):
            dr_seed.append((measured_log2m(ladder_ts, R[t][23][s], 23) - measured_log2m(ladder_ts, R[t][21][s], 21)) / 2)
    mu, sd = float(np.mean(dr_seed)), float(np.std(dr_seed, ddof=1))
    out["drift"] = dict(per_tok=dr, per_doubling_mean=mu, per_doubling_sd_seedwise=sd,
                        data_half_width80_per_doubling=float(Z80 * sd),
                        note="point = TinyStories mean drift x doublings; interval = [min(0, point), max(0, point)] "
                             "+- 1.2816 sqrt(sum over the two end budgets of 2 (SIGMA_CELL_BPB / s(k))^2)")
    out["drift"]["B3_interval_on_ts_curve_21_23"] = drift_interval(hr, nu, n0, mu, 21, 23)
    out["drift"]["A2_interval_on_ts_curve_26_29"] = drift_interval(hr, nu, n0, mu, 26, 29)
    out["sigma_cell_bpb_drift"] = SIGMA_CELL_BPB
    return out, hr


# ---------------------------------------------------------------- forecast
def drift_interval(hr, nu, n0, mu, k0, k1):
    """Drift of log2 m from 2^k0 to 2^k1. The model (constant b^c) predicts zero drift: point = 0. The interval contains
    0 and the TinyStories empirical drift mu (k1 - k0), widened by the propagated loss noise at both end budgets on the
    forecast bytes curve of hr."""
    s = lambda k: abs(predict(hr, nu, n0, 0, 1, k - 0.5) - predict(hr, nu, n0, 0, 1, k + 0.5))
    if min(s(k0), s(k1)) <= 0:
        raise ValueError(f"forecast bytes curve is flat at 2^{k0} or 2^{k1}: n_eff is beyond the reference curve")
    # each end's log2 m is noisy through the tokenizer cell and through the bytes ladder it is read on: 2 (sigma/s)^2
    w = Z80 * float(np.sqrt(2 * (SIGMA_CELL_BPB / s(k0)) ** 2 + 2 * (SIGMA_CELL_BPB / s(k1)) ** 2))
    p = mu * (k1 - k0)
    return dict(span_doublings=k1 - k0, ours=0.0, ts_empirical=p, noise_half_width=w,
                bpb_slope_per_doubling=[s(k0), s(k1)], interval80=[min(0.0, p) - w, max(0.0, p) + w],
                note="point 0 is the model's prediction (constant b^c); the TinyStories drift is an empirical note "
                     "that widens the interval, not a prediction")


# Secondary, explicitly secondary drift forecast: "TinyStories-transferred drift". Per-4x drifts of the five
# one-spelling TinyStories tokenizers from an earlier tokenizer check (bytes ladder 2^20-2^24, 2-seed bootstrap):
TS_AUDIT_DRIFT_4X = (-0.130, -0.128, -0.343, -0.151, -0.176)


def drift_secondary(primary):
    """Point = median of the measured drifts (scaled per doubling); 80% half-width = 1.2816 sqrt(spread SD^2 + the
    primary interval's propagated seed noise as an SD^2), both scaled to the span."""
    span = primary["span_doublings"]
    v = np.array(TS_AUDIT_DRIFT_4X) / 2 * span
    sd_noise = primary["noise_half_width"] / Z80
    w = Z80 * float(np.sqrt(np.std(v, ddof=1) ** 2 + sd_noise ** 2))
    p = float(np.median(v))
    return dict(point=p, interval80=[p - w, p + w], half_width=w, source="TS_AUDIT_DRIFT_4X",
                note="secondary: TinyStories-transferred drift; the primary forecast is 0 (constant b^c)")


def interval(point, *sig):
    s = float(np.sqrt(sum(x ** 2 for x in sig)))
    return [point - Z80 * s, point + Z80 * s], s


# ---------------------------------------------------------------- empirical TinyStories rivals (no c, no nu)
class TSInterp:
    """Rivals built from the measured TinyStories tokenizer losses alone, transferred to a new corpus with the same
    permitted inputs as the forecaster (b and the two reference curves): each TinyStories cell is read as an effective
    context on h_TS, log n_eff is interpolated in log b at that budget, and the result is read on h_new.
      ts_interp:  quadratic in log b through all TinyStories tokenizers at that budget (bytes, BPE-512/2048/8192,
                  wpair2/3 where run); outside the measured log b range, continued linearly along the end tangent.
      ts_nearest: the TinyStories tokenizer with the nearest log b (nearest compression).
    Defined only at budgets with TinyStories runs (2^20-2^24 bytes, 2^20-2^24 BPE, 2^21/2^23 wpair; 2^26 excluded)."""

    def __init__(self, ts):
        R, B = load_ts()
        L = means(R)
        self.tab = {}
        for k in range(20, 25):
            pts = sorted((float(np.log(B[t])), float(np.log(ts.neff(L[t][k]))))
                         for t in ("bytes",) + ONE_SPELLING if k in L[t])
            if len(pts) >= 3:
                self.tab[k] = np.array(pts)

    def log_neff(self, b, k, kind):
        if k not in self.tab:
            return None
        x, y = self.tab[k][:, 0], self.tab[k][:, 1]
        lb = np.log(b)
        if kind == "ts_nearest":
            return float(y[np.argmin(np.abs(x - lb))])
        p = np.polyfit(x, y, 2)
        if lb < x[0] or lb > x[-1]:
            e = x[0] if lb < x[0] else x[-1]
            return float(np.polyval(p, e) + np.polyval(np.polyder(p), e) * (lb - e))
        return float(np.polyval(p, lb))

    def predict(self, h_new, b, k, kind):
        v = self.log_neff(b, k, kind)
        return None if v is None else float(h_new.h(np.exp(v)))


# ---------------------------------------------------------------- Cagnetta rival
CAG_PRIMARY = "tok_1_200"
CAG_REGIME = ("Cagnetta et al. horizon-dominated regime: excess loss set by the correlation horizon n* ~ D^(1/(2 beta)) "
              "with conditional-entropy decay ~ n^-gamma, so alpha_D = gamma / (2 beta) per encoding (token units). "
              "Drift of log2 m per doubling = alpha_tok / alpha_bytes - 1. Primary window: beta and gamma from token "
              "lags 1-200 ('tok_1_200'); every other window present for both is a sensitivity number.")


def _win(v):
    return v if isinstance(v, dict) else {CAG_PRIMARY: float(v)}


def cagnetta_drift(betas, gammas, t, ref="bytes"):
    """Per-doubling drift of log2 m under alpha = gamma/(2 beta). gammas may be None (then gamma is assumed equal across
    encodings and cancels; flagged). Returns (primary, {window: value}, assumption) or (None, {}, reason)."""
    if not betas or t not in betas or ref not in betas:
        return None, {}, "no beta"
    bt, bb = _win(betas[t]), _win(betas[ref])
    if gammas and t in gammas and ref in gammas:
        gt, gb = _win(gammas[t]), _win(gammas[ref])
        wins = [w for w in bt if w in bb and w in gt and w in gb]
        f = {w: (gt[w] / (2 * bt[w])) / (gb[w] / (2 * bb[w])) - 1.0 for w in wins}
        how = "full alpha = gamma/(2 beta)"
    else:
        wins = [w for w in bt if w in bb]
        f = {w: bb[w] / bt[w] - 1.0 for w in wins}
        how = "gamma not supplied: assumed equal across encodings (cancels); beta ratio only"
    return f.get(CAG_PRIMARY), f, how


LEARNER = dict(d=256, nl=4, nh=4, emb=0, test_split=True)     # the calibrated learner (protocol v2, --test_split)


def forecast(href_paths, bpt, budgets, bytes_ladder=None, betas=None, ctx=512.0, ts_href=TS_HREF, gammas=None,
             bytes_budgets=None, learner=None):
    """B forecast. bpt: {tok: b or {b, stratum}}; budgets: tokenizer budgets (log2 D); bytes_budgets: the bytes
    anchor ladder (default = budgets). bytes_ladder is accepted for backward compatibility and ignored: B2 is computed
    by the scorer from the measured ladder with the rule stored in F['B2_rule']."""
    cal, ts = calibrate(ts_href)
    nu, n0, c = cal["nu"], cal["n0"], cal["c"]
    hrs = [p if isinstance(p, HRef) else HRef(p) for p in href_paths]
    hp = hrs[0]
    Kc = cal["constK"]["K"]
    shift_ctx = float(hp.h(ctx) - ts.h(ctx))
    tsi = TSInterp(ts)
    toks = {t: (v if isinstance(v, dict) else dict(b=v)) for t, v in bpt.items()}
    dr = cal["drift"]
    bytes_budgets = list(bytes_budgets or budgets)
    F = dict(model="L = h_ref(n0 (D b^c)^nu)", constants=dict(nu=nu, n0=n0, c=c), calibration=cal,
             href=[dict(path=h.path, sha256=h.sha) for h in hrs], ctx_bytes=ctx, budgets=list(budgets),
             bytes_budgets=bytes_budgets,
             interval_rule=("80% = point +- 1.2816 sqrt(s_loo^2 + s_seed^2 + s_sys^2); s_loo = leave-one-out rms "
                            "(bytes ladder for bytes, leave-one-tokenizer-out on 2^20-2^24 for tokenizers); s_seed = "
                            "pooled per-run seed SD; s_sys = SD of the point forecast across reference LMs"),
             B2_rule=dict(c=c, sigma_shift=cal["loo_c"]["rms_shift_on_measured_ladder"], sigma_seed=cal["seed_sd"],
                          z80=Z80, interpolation=INTERP_RULE,
                          formula="L_tok(D) = bytes ladder at log2 D + c log2 b; rivals c0 (log2 D), c1 (log2 D + log2 b)"),
             cagnetta_regime=CAG_REGIME, cells={}, drift={}, learner=dict(learner or LEARNER),
             expected={t: dict(bpe_dropout=float(v.get("bpe_dropout", 0.0))) for t, v in
                       ((t, v if isinstance(v, dict) else {}) for t, v in bpt.items())},
             variants=dict(primary="fitted c (c = %.4f)" % c,
                           H_jump="secondary, stated before training: c = 1/nu - 1 = %.4f, no fitted tokenizer constant; same "
                                  "interval rule with s_loo = its TinyStories rms on 2^20-2^24" % cal["H_jump"]["c"]))
    targets = [predict(hp, nu, n0, 0.0, 1.0, k) for k in TARGET_LOG2D]
    F["data_to_target"] = dict(targets_bpb=targets, defined_as_bytes_forecast_at_log2D=list(TARGET_LOG2D),
                               rule="per S1 tokenizer and target: ratio = 2^(forecast log2 D - measured log2 D), the "
                                    "measured log2 D interpolated piecewise linearly on the tokenizer's own measured "
                                    "ladder (no extrapolation; out of range reported)", pred_log2D={}, pred_log2D_H_jump={})
    for t, info in toks.items():
        b = float(info["b"]); stratum = info.get("stratum", "anchor" if t == "bytes" else "S1")
        cc = 0.0 if t == "bytes" else c
        s_loo = cal["loo_bytes"]["rms"] if t == "bytes" else cal["loo_c"]["rms_20_24"]
        F["data_to_target"]["pred_log2D"][t] = [log2D_to_reach(hp, nu, n0, cc, b, T) for T in targets]
        F["data_to_target"]["pred_log2D_H_jump"][t] = [
            log2D_to_reach(hp, nu, n0, 0.0 if t == "bytes" else cal["H_jump"]["c"], b, T) for T in targets]
        for k in (bytes_budgets if t == "bytes" else budgets):
            pts = [predict(h, nu, n0, cc, b, k) for h in hrs]
            p = pts[0]
            s_sys = float(np.std(pts, ddof=1)) if len(pts) > 1 else 0.0
            iv, sg = interval(p, s_loo, cal["seed_sd"], s_sys)
            riv = dict(c0=predict(hp, nu, n0, 0.0, b, k), c1=predict(hp, nu, n0, 1.0, b, k),
                       constK=predict_K(hp, Kc, cc, b, k), ts_shift=predict(ts, nu, n0, cc, b, k) + shift_ctx,
                       ts_interp=tsi.predict(hp, b, k, "ts_interp"), ts_nearest=tsi.predict(hp, b, k, "ts_nearest"))
            cj = 0.0 if t == "bytes" else cal["H_jump"]["c"]
            pj = [predict(h, nu, n0, cj, b, k) for h in hrs]
            sj = cal["loo_bytes"]["rms"] if t == "bytes" else cal["H_jump"]["rms_20_24"]
            ivj, sgj = interval(pj[0], sj, cal["seed_sd"], float(np.std(pj, ddof=1)) if len(pj) > 1 else 0.0)
            F["cells"][f"{t}|{k}"] = dict(tok=t, stratum=stratum, b=b, log2D=k, point=p, interval80=iv, sigma=sg,
                                         sigma_parts=dict(loo=s_loo, seed=cal["seed_sd"], sys=s_sys), per_href=pts,
                                         log2m_pred=cc * float(np.log2(b)), rivals=riv,
                                         H_jump=dict(c=cj, point=pj[0], interval80=ivj, sigma=sgj, per_href=pj,
                                                     log2m_pred=cj * float(np.log2(b))))
        if t != "bytes":
            d = drift_interval(hp, nu, n0, dr["per_doubling_mean"], min(budgets), max(budgets))
            d["secondary_ts_transferred"] = drift_secondary(d)
            span = max(budgets) - min(budgets)
            prim, sens, how = cagnetta_drift(betas, gammas, t)
            d.update(cagnetta=None if prim is None else prim * span,
                     cagnetta_sensitivity={w: v * span for w, v in sens.items()}, cagnetta_inputs=how)
            F["drift"][t] = d
    return F


# Data-to-target: bpb targets fixed in the forecast before training = the forecast bytes curve (primary h_ref) at these
# log2 D; for each tokenizer the forecast's log2 D needed to reach each target vs the measured value (interpolated on
# the tokenizer's own measured 2^21-2^23 ladder).
TARGET_LOG2D = (21.5, 22.0, 22.5)


def log2D_to_reach(hr, nu, n0, c, b, T):
    """Closed-form inverse of the forecast: log2 D at which h_ref(n0 (D b^c)^nu) = T."""
    n = hr.neff(T)
    return float(np.log2((n / n0) ** (1.0 / nu)) - c * np.log2(b))


def zipf_distance(counts):
    """Zipf heuristic (zero-training rival): least-squares slope of log frequency on log rank over all tokens with a
    nonzero count; distance = |slope + 1| (0 = exact Zipf). Counts must come from the tokenizer-fit / reference-stats
    partition, never from valid/test."""
    f = np.sort(np.asarray(counts, float))[::-1]
    f = f[f > 0]
    sl = np.polyfit(np.log(np.arange(1, len(f) + 1)), np.log(f), 1)[0]
    return float(abs(sl + 1.0)), float(sl)


A_GPT2 = (25, 26, 27, 28)     # GPT-2 budgets: their losses fall inside the bytes ladder 2^26-2^29 (m ~ 2)
A_BYTES = (26, 27, 28, 29)
A2_SPAN = (25, 27)            # drift 2^25 -> 2^27: GPT-2 at 2^28 with m ~ 2 sits at the ladder top (would need 2^30)


def midscale(href_paths, gpt2_bpt, betas=None, ts_href=TS_HREF, budgets=A_GPT2, gammas=None, bytes_budgets=A_BYTES):
    """A1 (level shift m = b^c of the GPT-2 ladder onto the bytes ladder) at the GPT-2 budgets, A2 (drift of log2 m over
    the GPT-2 budgets, model prediction 0), A3 (exploratory absolute curves)."""
    cal, ts = calibrate(ts_href)
    nu, n0, c = cal["nu"], cal["n0"], cal["c"]
    # m interval from the spread of per-cell measured c on TinyStories (cells inside the TinyStories ladder)
    cs = []
    R, B = load_ts(); L = means(R)
    lad = {k: L["bytes"][k] for k in BYTES_KS}
    for t in ONE_SPELLING:
        for k in sorted(L[t]):
            if k in RES_KS:
                cs.append(measured_log2m(lad, L[t][k], k) / np.log2(B[t]))
    cs = np.array([x for x in cs if np.isfinite(x)])     # cells outside the TinyStories ladder are dropped
    c_lo, c_hi = c - Z80 * float(np.std(cs, ddof=1)), c + Z80 * float(np.std(cs, ddof=1))
    lb = float(np.log2(gpt2_bpt))
    A1 = dict(b=gpt2_bpt, c=c, c_interval80=[c_lo, c_hi], c_cells_sd=float(np.std(cs, ddof=1)), n_cells=int(len(cs)),
              m=gpt2_bpt ** c, m_interval80=[gpt2_bpt ** c_lo, gpt2_bpt ** c_hi],
              log2m=c * lb, log2m_interval80=[c_lo * lb, c_hi * lb], budgets=list(budgets),
              bytes_budgets=list(bytes_budgets),
              rivals=dict(c0=dict(m=1.0, log2m=0.0), c1=dict(m=gpt2_bpt, log2m=lb)),
              hit_rule="all four GPT-2 budgets measured (else INCOMPLETE); measured m inside m_interval80 at >= 3 of 4 "
                       "(a cell outside the bytes ladder range counts as outside) AND rms(log2 m) of ours below both "
                       "rivals (over in-range budgets)")
    hw = Z80 * float(np.std(cs, ddof=1)) * lb
    cj = cal["H_jump"]["c"]
    A1["H_jump"] = dict(c=cj, log2m=cj * lb, log2m_interval80=[cj * lb - hw, cj * lb + hw], m=gpt2_bpt ** cj,
                        note="unconditional: c = 1/nu - 1 with the TinyStories nu; same half-width as the primary")
    A1["H_jump_conditional"] = dict(
        rule="c_A = 1/nu_A - 1 with nu_A fitted (least squares of log n_eff on log D, n_eff read on the primary "
             "reference curve stored in A['href_primary']) on A's own measured bytes ladder; log2 m = c_A log2 b; "
             "same half-width; computed by the scorer only after the bytes runs", half_width=hw, log2_b=lb)
    dr = cal["drift"]
    k0, k1 = A2_SPAN
    span = k1 - k0
    hrs = [p if isinstance(p, HRef) else HRef(p) for p in href_paths]
    prim, sens, how = cagnetta_drift(betas, gammas, "gpt2")
    A2 = dict(drift_interval(hrs[0] if hrs else ts, nu, n0, dr["per_doubling_mean"], k0, k1), span=[k0, k1],
              cagnetta=None if prim is None else prim * span,
              cagnetta_sensitivity={w: v * span for w, v in sens.items()}, cagnetta_inputs=how,
              cagnetta_regime=CAG_REGIME,
              hit_rule="drift of log2 m from GPT-2 2^25 to 2^27, both ends inside the bytes ladder (else INCOMPLETE); "
                       "score = |measured - 0| (symmetric); passes if inside interval80; Cagnetta's |measured - prediction| "
                       "reported alongside")
    A3 = {}
    if hrs:
        shift = float(hrs[0].h(2048.0) - ts.h(2048.0))
        for t, b, cc, ks in (("bytes", 1.0, 0.0, bytes_budgets), ("gpt2", gpt2_bpt, c, budgets)):
            for k in ks:
                A3[f"{t}|{k}"] = dict(point=predict(hrs[0], nu, n0, cc, b, k),
                                      per_href=[predict(h, nu, n0, cc, b, k) for h in hrs],
                                      rival_ts_shift_2048=predict(ts, nu, n0, cc, b, k) + shift)
    hp0 = hrs[0] if hrs else ts
    return dict(A1=A1, A2=A2, A3_exploratory=A3, constants=dict(nu=nu, n0=n0, c=c, c_H_jump=cj),
                href_primary=dict(path=hp0.path, sha256=hp0.sha, log_context_bytes=hp0.x.tolist(), bits_per_byte=hp0.y.tolist()),
                note="A3 is reported, not claimed: the 100M single-pass learner need not share n0, nu.")


# ---------------------------------------------------------------- CLI
def print_calibration(cal):
    print(f"nu = {cal['nu']:.4f}   n0 = {cal['n0']:.6g}   c = {cal['c']:.4f}   seed SD = {cal['seed_sd']:.4f}")
    print("n_eff bytes (2^20..2^24): " + ", ".join(f"{v:.1f}" for v in cal["neff_bytes"].values()))
    o = cal["bytes_oos_fit_20_22"]
    print(f"bytes fit 2^20-2^22 (nu {o['nu']:.3f}): err 2^23 {o['err_2^23']:+.4f}  2^24 {o['err_2^24']:+.4f}  "
          f"2^26 {o['err_2^26']:+.4f}")
    print(f"bytes 2^26 (full fit): forecast {cal['bytes_2^26']['forecast']:.3f} measured {cal['bytes_2^26']['measured']:.3f}")
    print(f"bytes ladder LOO rms {cal['loo_bytes']['rms']:.4f}")
    lc = cal["loo_c"]
    print("LOO c: " + ", ".join(f"{t} {v:.3f}" for t, v in lc["c"].items()))
    print(f"LOO tokenizer rms on 2^21-2^23 {lc['rms_fit_budgets']:.4f} (max {lc['maxabs_fit_budgets']:.3f}); "
          f"on 2^20-2^24 {lc['rms_20_24']:.4f} (max {lc['maxabs_20_24']:.3f}); shift rule on measured ladder "
          f"{lc['rms_shift_on_measured_ladder']:.4f}")
    print(f"rival c=0 abs err {cal['rival_c0_abs_err'][0]:.3f}-{cal['rival_c0_abs_err'][1]:.3f}; "
          f"c=1 {cal['rival_c1_abs_err'][0]:.3f}-{cal['rival_c1_abs_err'][1]:.3f}")
    j = cal["H_jump"]
    print(f"H_jump: c = 1/nu - 1 = {j['c']:.4f}; one-spelling cells 2^20-2^24 (n={j['n_cells']}): rms {j['rms_20_24']:.4f}, "
          f"max {j['maxabs_20_24']:.3f}, mean {j['mean_20_24']:+.4f}; fitted c in-sample rms "
          f"{j['fitted_c_in_sample_rms_20_24']:.4f}, max {j['fitted_c_in_sample_maxabs_20_24']:.3f}")
    print(f"constant-K: K = {cal['constK']['K']:.4g}, drift of log2(D|h'|) per doubling "
          f"{cal['constK']['drift_log2_per_doubling']:+.3f}")
    d = cal["drift"]
    print("drift of log2 m per 4x (2^21->2^23): " + ", ".join(f"{t} {2 * v['per_doubling']:+.3f}"
                                                              for t, v in d["per_tok"].items()))
    print(f"drift per doubling: mean {d['per_doubling_mean']:+.3f}, seedwise SD {d['per_doubling_sd_seedwise']:.3f}, "
          f"data 80% half-width {d['data_half_width80_per_doubling']:.3f}")
    di = drift_interval(HRef(TS_HREF), cal["nu"], cal["n0"], d["per_doubling_mean"], 21, 23)
    print(f"B3 drift interval on the TinyStories curve, 2^21->2^23: point {di['ours']:+.3f}, "
          f"[{di['interval80'][0]:+.3f}, {di['interval80'][1]:+.3f}] (noise half-width {di['noise_half_width']:.3f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--midscale", action="store_true")
    ap.add_argument("--href", nargs="*", default=[])
    ap.add_argument("--ts_href", default=TS_HREF)
    ap.add_argument("--bpt")
    ap.add_argument("--budgets", nargs="*", type=int, default=[21, 22, 23])
    ap.add_argument("--betas")
    ap.add_argument("--gammas")
    ap.add_argument("--ctx", type=float, default=512.0)
    ap.add_argument("--gpt2_bpt", type=float)
    ap.add_argument("--out")
    a = ap.parse_args()
    ld = lambda p: json.load(open(p)) if p else None
    if a.calibrate:
        cal, _ = calibrate(a.ts_href)
        print_calibration(cal)
        res = cal
    elif a.midscale:
        res = midscale(a.href, a.gpt2_bpt, ld(a.betas), a.ts_href, gammas=ld(a.gammas))
        print(json.dumps({k: res[k] for k in ("A1", "A2")}, indent=1))
    else:
        res = forecast(a.href, ld(a.bpt), a.budgets, None, ld(a.betas), a.ctx, a.ts_href, ld(a.gammas))
        print(f"{len(res['cells'])} cells, {len(res['drift'])} drift forecasts")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=float)
        print("wrote", a.out, hashlib.sha256(open(a.out, "rb").read()).hexdigest())
