"""Final analysis of the FineWeb-Edu tokenizer test (small d256/nl4 learner). No training; reads finished runs.
Writes results/bfine/final_analysis.json.

Inputs (read only): results/text_runs/bfine/*.json (112 runs), results/text_runs/bfine_{d384,emb64} (controls),
results/bfine/forecast.json (forecast written before training), results/bfine/forecast_eventcount.json,
results/tokenizer_level/context_eventcount.json, results/tokenizer_level/diagnosis.json (rerun with all cells).

Parts
  1. Forecast error decomposed into one corpus-wide offset, a budget part and a tokenizer part; every rival rescored
     after removing one constant and after one anchor (the measured bytes loss at 2^21, seed 0); ordering and
     per-tokenizer data-scaling slope.
  2. Rate: exponent ratios tokenizer / bytes (common-floor power law; and the shift-invariant ratio 1 + d log2 m / dk).
  3. Level drift: log2 m per tokenizer and budget on the measured bytes ladder (2^21-2^25), its slope per doubling.
  4. Event-count forecasts scored on the cells that were unseen when they were written.
Usage: python scripts/analyze_fineweb_final.py
"""
import glob, json, os
import numpy as np
from scipy import stats
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results/bfine/final_analysis.json")
KS = (21, 22, 23)
DUP = {"fch1"}                                   # byte-identical to fch0
RULES = ("ours", "H_jump", "c0", "c1", "constK", "ts_shift", "ts_interp", "ts_nearest")
NB = 10000
rng = np.random.default_rng(0)


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


# ---------------------------------------------------------------- data
def load_runs(tag):
    R = {}
    for f in sorted(glob.glob(os.path.join(ROOT, f"results/text_runs/{tag}/*.json"))):
        j = json.load(open(f))
        R.setdefault(j["tok"], {}).setdefault(int(round(np.log2(j["train_bytes"]))), {})[j["seed"]] = j["bits_per_byte"]
    return R


RUNS = load_runs("bfine")
M = {t: {k: float(np.mean(list(s.values()))) for k, s in d.items()} for t, d in RUNS.items()}
FC = json.load(open(os.path.join(ROOT, "results/bfine/forecast.json")))["B"]
CELLS = FC["cells"]
S1 = sorted({e["tok"] for e in CELLS.values() if e["stratum"] == "S1"})
NONBYTE = sorted({e["tok"] for e in CELLS.values() if e["tok"] not in DUP and e["tok"] != "bytes"})
ENC = NONBYTE + ["bytes"]
LADDER = M["bytes"]
B = {t: CELLS[f"{t}|21"]["b"] for t in ENC}
SEED_SD = float(np.sqrt(np.mean([np.var(list(s.values()), ddof=1) for t, d in RUNS.items() if t not in DUP
                                 for s in d.values() if len(s) == 2])))
CELL_SD = SEED_SD / np.sqrt(2)


def pred(rule, t, k):
    e = CELLS[f"{t}|{k}"]
    if rule == "ours":
        return e["point"]
    if rule == "H_jump":
        return e["H_jump"]["point"]
    return e["rivals"][rule]


# ---------------------------------------------------------------- 1. decomposition
def decompose(rule, toks):
    """Balanced grid toks x KS. e = pred - measured = a + b_k + u_t + w_tk."""
    E = np.array([[pred(rule, t, k) - M[t][k] for k in KS] for t in toks])
    a = E.mean(); bk = E.mean(0) - a; ut = E.mean(1) - a
    w = E - a - bk[None, :] - ut[:, None]
    n = E.size
    ms = dict(total=float(np.mean(E ** 2)), offset=float(a ** 2), budget=float(np.sum(bk ** 2) * len(toks) / n),
              tokenizer_main=float(np.sum(ut ** 2) * len(KS) / n), tokenizer_x_budget=float(np.mean(w ** 2)))
    slope = float(np.polyfit(KS, E.mean(0), 1)[0])
    return dict(n=n, rms=rms(E), offset=float(a), budget_effects={str(k): float(v) for k, v in zip(KS, bk)},
                budget_slope_err_per_doubling=slope,
                mean_square=ms, share_of_mean_square={q: v / ms["total"] for q, v in ms.items() if q != "total"},
                rms_after_offset=rms(E - a), rms_after_offset_budget=rms(E - a - bk[None, :]),
                tokenizer_main_effects={t: float(v) for t, v in zip(toks, ut)})


def anchored(rule, toks, ks_extra=()):
    """One cheap run: shift every forecast by (measured - forecast) at bytes 2^21 seed 0. Cells: toks x KS plus bytes
    at ks_extra (the anchor cell itself excluded)."""
    sh = RUNS["bytes"][21][0] - pred(rule, "bytes", 21)
    rows = {t: np.array([pred(rule, t, k) + sh - M[t][k] for k in KS if not (t == "bytes" and k == 21)]) for t in toks}
    if ks_extra:
        rows["bytes"] = np.concatenate([rows.get("bytes", []), [pred(rule, "bytes", k) + sh - M["bytes"][k] for k in ks_extra]])
    return sh, rows


def block_boot_delta(rows_a, rows_b, n=NB):
    toks = list(rows_a)
    idx = rng.integers(0, len(toks), (n, len(toks)))
    A = [rows_a[t] for t in toks]; Bv = [rows_b[t] for t in toks]
    v = np.array([rms(np.concatenate([Bv[i] for i in ii])) - rms(np.concatenate([A[i] for i in ii])) for ii in idx])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def part1():
    out = {}
    for name, toks in (("S1", S1), ("all_17", ENC)):
        dec = {r: decompose(r, toks) for r in RULES}
        const = {r: dict(rms_raw=dec[r]["rms"], rms_one_constant=dec[r]["rms_after_offset"],
                         rms_offset_and_budget=dec[r]["rms_after_offset_budget"]) for r in RULES}
        anc = {}
        rows = {}
        for r in RULES:
            sh, rw = anchored(r, toks, ks_extra=(24,) if "bytes" in toks else ())
            rows[r] = rw
            anc[r] = dict(shift=float(sh), rms=rms(np.concatenate(list(rw.values()))),
                          rms_by_budget={str(k): rms([pred(r, t, k) + sh - M[t][k] for t in toks if not (t == "bytes" and k == 21)])
                                         for k in KS})
        for r in RULES:
            if r != "ours":
                anc[r]["delta_rival_minus_ours"] = anc[r]["rms"] - anc["ours"]["rms"]
                anc[r]["delta_ci95_tokenizer_boot"] = block_boot_delta(rows["ours"], rows[r])
        out[name] = dict(n_tokenizers=len(toks), decomposition=dec, after_one_constant=const, bytes_anchor=anc)
    out["S1_relative_to_bytes"] = rel_bytes()
    out["single_tokenizer_anchor"] = tok_anchor()
    out["S2_tax_magnitude_after_S1_offset"] = s2_magnitude(out["S1"]["decomposition"]["ours"]["offset"])
    return out


def rel_bytes():
    """Split each S1 cell error into the bytes-curve error at that budget and the error in the tokenizer's shift
    relative to bytes: e = e_bytes(k) + [(pred_t - pred_bytes) - (meas_t - meas_bytes)]."""
    o = {}
    for r in RULES:
        eb = {k: pred(r, "bytes", k) - M["bytes"][k] for k in KS}
        ps = np.array([[pred(r, t, k) - pred(r, "bytes", k) for k in KS] for t in S1])
        ms_ = np.array([[M[t][k] - M["bytes"][k] for k in KS] for t in S1])
        se = ps - ms_
        o[r] = dict(bytes_err={str(k): float(v) for k, v in eb.items()}, mean_bytes_err=float(np.mean(list(eb.values()))),
                    mean_pred_shift=float(ps.mean()), mean_meas_shift=float(ms_.mean()),
                    meas_shift_by_budget={str(k): float(v) for k, v in zip(KS, ms_.mean(0))},
                    mean_shift_err=float(se.mean()), rms_shift_err=rms(se),
                    rms_shift_err_after_mean=rms(se - se.mean()))
    return o


def tok_anchor():
    """One cheap run of one S1 tokenizer at 2^21 (seed 0) fixes the level; score the other nine S1 tokenizers at
    2^21-2^23. Mean and range over the ten choices of anchor."""
    o = {}
    for r in RULES:
        v = []
        for j in S1:
            sh = RUNS[j][21][0] - pred(r, j, 21)
            v.append(rms([pred(r, t, k) + sh - M[t][k] for t in S1 if t != j for k in KS]))
        o[r] = dict(mean_rms=float(np.mean(v)), min_rms=float(min(v)), max_rms=float(max(v)))
    return o


def s2_magnitude(off):
    """B4 magnitude forecasts for the S2 tokenizers after subtracting the offset fitted on the S1 cells only."""
    rows = []
    for t in sorted({e["tok"] for e in CELLS.values() if e["stratum"] == "S2"} - DUP):
        for k in KS:
            mg = CELLS[f"{t}|{k}"].get("B4", {}).get("magnitude")
            if mg:
                rows.append((t, k, mg["point"], M[t][k], CELLS[f"{t}|{k}"]["point"]))
    e = np.array([p - y for _, _, p, y, _ in rows]); e0 = np.array([p - y for _, _, _, y, p in rows])
    return dict(n=len(rows), rms_raw=rms(e), rms_after_S1_offset=rms(e - off),
                rms_one_spelling_point_after_S1_offset=rms(e0 - off),
                per_cell={f"{t}|{k}": dict(tax_point=p, measured=y, after_offset_err=p - off - y) for t, k, p, y, _ in rows})


# ---------------------------------------------------------------- ordering and per-tokenizer slope
def spearman_within(P, Y):
    """P, Y: arrays tok x budget. Mean over budgets of the within-budget Spearman."""
    v = [stats.spearmanr(P[:, j], Y[:, j]).statistic for j in range(P.shape[1])]
    return float(np.mean(v)), [float(x) for x in v]


def perm_p(P, Y, n=NB):
    obs, _ = spearman_within(P, Y)
    null = np.array([spearman_within(P, Y[rng.permutation(len(Y))])[0] for _ in range(n)])
    return float((1 + np.sum(null >= obs)) / (1 + n))


def ordering(predictors, toks):
    Y = np.array([[M[t][k] for k in KS] for t in toks])
    out = {}
    for name, f in predictors.items():
        P = np.array([[f(t, k) for k in KS] for t in toks], float)
        if np.any(~np.isfinite(P)) or np.all(np.ptp(P, axis=0) == 0):
            continue
        m, per = spearman_within(P, Y)
        out[name] = dict(mean_rho=m, per_budget={str(k): v for k, v in zip(KS, per)}, p_block_perm=perm_p(P, Y, 2000))
    return out


def slopes(toks, rules=RULES):
    meas = {t: float(np.polyfit(KS, [M[t][k] for k in KS], 1)[0]) for t in toks}
    out = dict(measured_slope_bpb_per_doubling=meas)
    for r in rules:
        p = {t: float(np.polyfit(KS, [pred(r, t, k) for k in KS], 1)[0]) for t in toks}
        err = np.array([p[t] - meas[t] for t in toks])
        d = dict(rms_slope_err=rms(err), mean_slope_err=float(err.mean()))
        if np.ptp(list(p.values())) > 1e-9:
            d["spearman_slope"] = float(stats.spearmanr([p[t] for t in toks], [meas[t] for t in toks]).statistic)
        out[r] = d
    # rule: every tokenizer has the measured bytes slope (needs one bytes ladder)
    sb = float(np.polyfit(KS, [M["bytes"][k] for k in KS], 1)[0])
    eb = np.array([sb - meas[t] for t in toks if t != "bytes"])
    out["measured_bytes_slope_for_all"] = dict(rms_slope_err=rms(eb), mean_slope_err=float(eb.mean()))
    return out


def load_count_lm():
    d = json.load(open(os.path.join(ROOT, "results/tokenizer_level/count_lm.json")))
    C = {}
    for r in d:
        if r["corpus"] == "fw":
            C.setdefault(r["tok"], {})[r["log2D"]] = r["bpb"]
    return C


def load_ctx_ec():
    d = json.load(open(os.path.join(ROOT, "results/tokenizer_level/context_eventcount.json")))
    C = {}
    for c in d["cells"]:
        if c["corpus"] == "fw":
            C[(c["tok"], c["log2D"])] = c
    return d, C


# ---------------------------------------------------------------- 2. rate
def log2m_cell(L, k, lad):
    ks = np.array(sorted(lad), float); ys = np.array([lad[q] for q in sorted(lad)])
    if ys.min() <= L <= ys.max():
        return float(np.interp(-L, -ys, ks) - k), ""
    if L > ys.max():
        return float(ks[0] - (L - ys[0]) / ((ys[0] - ys[1]) / (ks[1] - ks[0])) - k), "ext"
    return float(ks[-1] + (ys[-1] - L) / ((ys[-2] - ys[-1]) / (ks[-1] - ks[-2])) - k), "ext"


def drift_fit(t, Mx, lad):
    v = [log2m_cell(Mx[t][k], k, lad)[0] for k in KS]
    return float(np.polyfit(KS, v, 1)[0]), v


def part3_and_rate(nsim=4000):
    """log2 m per cell on the measured bytes ladder 2^21-2^25; slope per doubling with a parametric interval (each
    cell mean, bytes ladder included, perturbed by the pooled seed SD / sqrt 2)."""
    out = {}
    sims = {t: [] for t in NONBYTE}
    for _ in range(nsim):
        Mx = {t: {k: M[t][k] + rng.normal(0, CELL_SD) for k in M[t]} for t in ENC}
        lad = Mx["bytes"]
        if not np.all(np.diff([lad[k] for k in sorted(lad)]) < 0):
            continue
        for t in NONBYTE:
            sims[t].append(drift_fit(t, Mx, lad)[0])
    for t in NONBYTE:
        s, v = drift_fit(t, M, LADDER)
        flags = [log2m_cell(M[t][k], k, LADDER)[1] for k in KS]
        bc = 0.4903 * np.log2(B[t])
        lo, hi = np.percentile(sims[t], [2.5, 97.5])
        out[t] = dict(b=B[t], log2m={str(k): x for k, x in zip(KS, v)}, flags={str(k): f for k, f in zip(KS, flags)},
                      slope_per_doubling=s, slope_ci95=[float(lo), float(hi)],
                      exponent_ratio_shift_invariant=1 + s, ratio_ci95=[float(1 + lo), float(1 + hi)],
                      log2m_bc=float(bc),
                      doublings_to_bc_at_this_slope=(float((bc - v[-1]) / s) if s > 0 and bc > v[-1] else None))
    return out


def powerlaw_fits(E_fixed=None, bytes_ks=(21, 22, 23, 24, 25), cells=None):
    """L_t(k) = E + A_t 2^(-alpha_t (k - 21)); E shared by all 17 encodings (free or fixed)."""
    cells = cells or M
    data = [(i, k, cells[t][k]) for i, t in enumerate(ENC) for k in (bytes_ks if t == "bytes" else KS)]
    n = len(ENC)

    def unpack(p):
        E = E_fixed if E_fixed is not None else p[-1]
        return E, p[:n], p[n:2 * n]

    def res(p):
        E, la, al = unpack(p)
        return np.array([E + np.exp(la[i]) * 2.0 ** (-al[i] * (k - 21)) - y for i, k, y in data])
    E0 = 0.885 if E_fixed is None else E_fixed
    p0 = np.r_[np.log([max(cells[t][21] - E0, 0.05) for t in ENC]), np.full(n, 0.2)]
    if E_fixed is None:
        p0 = np.r_[p0, E0]
    lb = np.r_[np.full(n, -10), np.full(n, 0.0)] if E_fixed is not None else np.r_[np.full(n, -10), np.full(n, 0.0), 0.0]
    ub = np.r_[np.full(n, 5), np.full(n, 5.0)] if E_fixed is not None else np.r_[np.full(n, 5), np.full(n, 5.0), 1.9]
    f = least_squares(res, p0, bounds=(lb, ub))
    E, la, al = unpack(f.x)
    return float(E), {t: float(al[i]) for i, t in enumerate(ENC)}, rms(f.fun)


def part2(nsim=200):
    out = {}
    for name, Ef, bks in (("free_floor_bytes21_25", None, (21, 22, 23, 24, 25)),
                          ("free_floor_bytes21_23", None, KS),
                          ("floor_0.885", 0.885, (21, 22, 23, 24, 25)),
                          ("floor_0.6", 0.6, (21, 22, 23, 24, 25))):
        E, al, r = powerlaw_fits(Ef, bks)
        sims = []
        for _ in range(nsim):
            Mx = {t: {k: M[t][k] + rng.normal(0, CELL_SD) for k in M[t]} for t in ENC}
            sims.append(powerlaw_fits(Ef, bks, Mx))
        ratio = {t: al[t] / al["bytes"] for t in NONBYTE}
        ci = {t: [float(np.percentile([s[1][t] / s[1]["bytes"] for s in sims], q)) for q in (2.5, 97.5)] for t in NONBYTE}
        s1r = [ratio[t] for t in S1]
        out[name] = dict(E=E, E_ci95=[float(np.percentile([s[0] for s in sims], q)) for q in (2.5, 97.5)],
                         fit_rms=r, alpha=al, alpha_bytes=al["bytes"], ratio_tok_over_bytes=ratio, ratio_ci95=ci,
                         S1_ratio_median=float(np.median(s1r)), S1_ratio_range=[float(min(s1r)), float(max(s1r))])
    return out


# ---------------------------------------------------------------- 4. event-count forecasts on unseen cells
def part4():
    ec = json.load(open(os.path.join(ROOT, "results/bfine/forecast_eventcount.json")))
    unseen = ec["unseen_cells"]["cells"]
    landed = sorted({c["cell"] for c in ec["landed_comparison"]["cells"]})
    vars_ = ("primary", "no_refit", "ctx", "K_ts")

    def score(keys, meas, name):
        rows = {}
        for v in vars_ + ("old", "c0"):
            p = []
            for key in keys:
                c = ec["cells"][key]
                if v == "old":
                    p.append(c["old_point"])
                elif v == "c0":
                    p.append(CELLS[key]["rivals"]["c0"] if key in CELLS else np.nan)
                else:
                    p.append(c[v]["point"])
            p = np.array(p); y = np.array([meas[k] for k in keys])
            d = dict(n=len(keys), rms=rms(p - y), mean_err=float(np.mean(p - y)), rms_one_constant=float(np.std(p - y)))
            if v in vars_:
                d["inside80"] = float(np.mean([ec["cells"][k][v]["interval80"][0] <= meas[k] <= ec["cells"][k][v]["interval80"][1]
                                               for k in keys]))
            if v == "old":
                d["inside80"] = float(np.mean([ec["cells"][k]["old_interval80"][0] <= meas[k] <= ec["cells"][k]["old_interval80"][1]
                                               for k in keys]))
            # within-budget ordering among non-byte cells
            rh = []
            for k in KS:
                idx = [i for i, key in enumerate(keys) if key.endswith(f"|{k}") and not key.startswith("bytes")
                       and not key.startswith("fch1")]
                if len(idx) >= 5:
                    rh.append(stats.spearmanr(p[idx], y[idx]).statistic)
            if rh and v != "c0":
                d["mean_within_budget_spearman"] = float(np.mean(rh))
            rows[v] = d
        return rows

    meas_mean = {f"{t}|{k}": M[t][k] for t in M for k in M[t]}
    keys = [k for k in unseen if not k.startswith("fch1")]
    res = dict(cells_unseen_when_written=len(keys), note="fch1 dropped (duplicate of fch0); seed means",
               unseen_cells=score(keys, meas_mean, "unseen"))
    s1_seed1 = {k: RUNS[k.split("|")[0]][21][1] for k in landed}
    res["landed_2e21_seed1"] = score(landed, s1_seed1, "seed1")

    # context event count (results/tokenizer_level/context_eventcount.json)
    d, C = load_ctx_ec()
    uf = {(u["tok"], u["log2D"]): u for u in d["unseen_forecasts"]}
    strict = [("bytes", 24, None), ("bytes", 25, None), ("bytes", 23, 1)] + [(t, 23, 1) for t in ("drop0", "drop1", "gpt2", "sbpe0", "sbpe1")]

    def ctx_score(items):
        o = {}
        for v in ("K_lab", "K_ts", "forecaster", "c0"):
            p, y = [], []
            for t, k, s in items:
                if v in ("K_lab", "K_ts"):
                    p.append(uf[(t, k)][v])
                elif v == "forecaster":
                    p.append(pred("ours", t, k))
                else:
                    p.append(pred("c0", t, k))
                y.append(M[t][k] if s is None else RUNS[t][k][s])
            p, y = np.array(p), np.array(y)
            o[v] = dict(n=len(p), rms=rms(p - y), mean_err=float(np.mean(p - y)), rms_one_constant=float(np.std(p - y)))
        return o
    res["context_eventcount"] = dict(
        strictly_unseen=dict(items=[f"{t}|2^{k}" + ("" if s is None else f" seed {s}") for t, k, s in strict],
                             scores=ctx_score(strict)),
        all_unseen_forecast_cells_seed_mean=dict(n=len(uf), scores=ctx_score([(t, k, None) for t, k in uf])),
        bytes_ladder_only=ctx_score([("bytes", 23, None), ("bytes", 24, None), ("bytes", 25, None)]))
    return res


# ---------------------------------------------------------------- main
def main():
    out = dict(script="scripts/analyze_fineweb_final.py", n_runs=sum(len(s) for d in RUNS.values() for s in d.values()),
               encodings_distinct=ENC, S1=S1, seed_sd_pooled=SEED_SD,
               measured={t: M[t] for t in ENC})
    out["part1_decomposition"] = part1()
    count = load_count_lm()
    _, CEC = load_ctx_ec()
    preds = {r: (lambda t, k, r=r: pred(r, t, k)) for r in RULES}
    preds["count_lm"] = lambda t, k: count.get("fch0" if t == "fch1" else t, {}).get(k, np.nan)
    preds["ctx_eventcount_K_lab"] = lambda t, k: CEC[(t, k)]["K_lab"] if (t, k) in CEC else np.nan
    preds["ctx_eventcount_K_ts"] = lambda t, k: CEC[(t, k)]["K_ts"] if (t, k) in CEC else np.nan
    ecf = json.load(open(os.path.join(ROOT, "results/bfine/forecast_eventcount.json")))["cells"]
    for v in ("primary", "ctx"):
        preds[f"type_eventcount_{v}"] = lambda t, k, v=v: ecf[f"{t}|{k}"][v]["point"]
    preds["minus_b"] = lambda t, k: -B[t]
    out["ordering"] = dict(S1=ordering(preds, S1), nonbyte_16=ordering(preds, NONBYTE),
                           note="mean within-budget Spearman of predicted against measured bits/byte; block "
                                "permutation over tokenizers; every c > 0 rule ranks by b, so ours = minus_b")
    out["per_tokenizer_slope"] = dict(S1=slopes(S1), all_17=slopes(ENC))
    out["drift_log2m"] = part3_and_rate()
    ords = [t for t in S1 + ["gpt2"]]
    sl = [out["drift_log2m"][t]["slope_per_doubling"] for t in ords]
    out["drift_summary"] = dict(ordinary=ords, mean_slope=float(np.mean(sl)), n_positive=int(sum(s > 0 for s in sl)),
                                p_sign=float(stats.binomtest(int(sum(s > 0 for s in sl)), len(sl), 0.5).pvalue),
                                mean_log2m={str(k): float(np.mean([out["drift_log2m"][t]["log2m"][str(k)] for t in ords])) for k in KS},
                                mean_log2m_bc=float(np.mean([out["drift_log2m"][t]["log2m_bc"] for t in ords])))
    out["rate_powerlaw"] = part2()
    out["eventcount_unseen"] = part4()
    diag = json.load(open(os.path.join(ROOT, "results/tokenizer_level/diagnosis.json")))
    out["bytes_realised_context_share"] = diag["H4"]["bytes_learned_context"]
    out["neff_bytes_learner"] = [dict(corpus=r["corpus"], log2D=r["log2D"], bpb=r["bpb"], neff=r["neff"]) for r in diag["neff_table"]]
    ctl = {}
    for tag in ("bfine_d384", "bfine_emb64"):
        R = load_runs(tag)
        ctl[tag] = {f"{t}|{k}": dict(control=float(np.mean(list(s.values()))), main=M[t][k],
                                     delta=float(np.mean(list(s.values())) - M[t][k])) for t, d in R.items() for k, s in d.items()}
    out["controls"] = ctl
    out["verdicts"] = {   # written after reading the numbers above
        "level_absolute": "REAL miss of the transferred level; 99.7% of the mean square is one corpus-wide offset, "
                          "-0.39 of the -0.49 b/B is the bytes curve (TinyStories n0, nu on a FineWeb learner); the "
                          "SmolLM2 reference tops out at 2.00 b/B, below every 2^21 cell, so the 2^21 cells were out of "
                          "the forecaster's reach (a design flaw that does not change the verdict)",
        "level_tokenizer_shift": "REAL: predicted token advantage -0.108 b/B against measured -0.016; with one bytes "
                                 "anchor ours 0.120 vs c0 0.024",
        "ordering": "REAL negative for b (S1 rho 0.15, p 0.33); positive object: count statistics order tokenizers "
                    "(type event count rho 0.60-0.65 over 16, p <= 0.01, on budgets unseen when written)",
        "rate": "Good: token data-scaling slopes predicted to 0.006 b/B per doubling rms (bytes slope too steep by "
                "0.030); exponent ratio tokens/bytes 0.92-1.07 with bytes to 2^25 under any floor, 1.08 shift-invariant; "
                "between-tokenizer rate differences not predicted",
        "drift": "REAL miss of the zero-drift point (+0.02 to +0.29 over two doublings, 10/10 S1 positive); MISREAD "
                 "as a loss to Cagnetta (1.9-2.8 predicted for S1; all 10 measured drifts are nearer 0 once the four censored 2^21 ends are read off the extended ladder); the rise is the positive object, too slow to reach b^c "
                 "in this learner's range",
    }
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    o = main()
    for nm in ("S1", "all_17"):
        p = o["part1_decomposition"][nm]
        print("==", nm)
        for r in RULES:
            d = p["decomposition"][r]; a = p["bytes_anchor"][r]
            print(f"{r:11s} raw {d['rms']:.3f} off {d['offset']:+.3f} 1const {d['rms_after_offset']:.3f} "
                  f"off+bud {d['rms_after_offset_budget']:.3f} slope_err {d['budget_slope_err_per_doubling']:+.3f} "
                  f"anchor {a['rms']:.3f} {a.get('delta_ci95_tokenizer_boot', '')} shares "
                  + " ".join(f"{q[:6]} {v:.2f}" for q, v in d["share_of_mean_square"].items()))
    for nm in ("S1", "nonbyte_16"):
        for r, d in o["ordering"][nm].items():
            print(nm, r, round(d["mean_rho"], 3), {k: round(v, 2) for k, v in d["per_budget"].items()}, round(d["p_block_perm"], 3))
    print(json.dumps(o["part1_decomposition"]["S1_relative_to_bytes"]["ours"]), json.dumps(o["part1_decomposition"]["S1_relative_to_bytes"]["c0"]))
    print(json.dumps(o["part1_decomposition"]["single_tokenizer_anchor"]))
    print(json.dumps({k: v for k, v in o["part1_decomposition"]["S2_tax_magnitude_after_S1_offset"].items() if k != "per_cell"}))
    print(json.dumps({nm: {r: d for r, d in v.items() if r != "measured_slope_bpb_per_doubling"} for nm, v in o["per_tokenizer_slope"].items()}))
    for t, d in o["drift_log2m"].items():
        print(t, {k: round(v, 2) for k, v in d["log2m"].items()}, d["flags"], round(d["slope_per_doubling"], 3),
              np.round(d["slope_ci95"], 3), "bc", round(d["log2m_bc"], 2), d["doublings_to_bc_at_this_slope"])
    print(o["drift_summary"])
    for nm, d in o["rate_powerlaw"].items():
        print(nm, "E", round(d["E"], 3), d["E_ci95"], "rms", round(d["fit_rms"], 4), "alpha_b", round(d["alpha_bytes"], 3),
              {t: (round(v, 2), np.round(d["ratio_ci95"][t], 2).tolist()) for t, v in d["ratio_tok_over_bytes"].items()})
    print(json.dumps(o["eventcount_unseen"], indent=0)[:4000])
    print(o["controls"])
