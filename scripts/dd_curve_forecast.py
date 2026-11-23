"""Whole-curve forecast on DataDecide in-run checkpoints (1B primary; 530M, 300M secondary).

    python3 scripts/dd_curve_forecast.py   # -> results/datadecide/dd_curve_forecast.json

Summary:
  level   affine map of the train-evaluation mismatch (headline `mismatch_all`) to the loss at an anchor checkpoint D*,
          fitted leave-one-family-out (6 families)
  shape   S_r(D) = mean_e h_e(n0_e (b_r D)^nu_e), nu_e = 1/(1+gamma_e), n0_e = (A_e gamma_e / K)^nu_e, h_e GPT-2's
          information curve on evaluation set e, gamma_e/A_e the power fit (4-64 tokens, floor at
          512 tokens), K one scalar per fold fitted on the training families' in-run curves
  arms    F0 data only; P2 two-pilot power law; P1 one pilot + information shape; B level + recipe-blind measured shape;
          C constant; P1b one pilot + recipe-blind measured shape; P2s two pilots + information shape; F0f F0 anchored
          at the final checkpoint
  scores  per-horizon and whole-curve r.m.s./median/max absolute error over recipes; paired family-bootstrap differences
Reads only on-disk files; CPU, < 1 GB.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import verify_datadecide_headline as VH  # noqa: E402  (load_features only; that script is not modified)

PPL = "data/datadecide/gt/ppl_results.parquet"
ST = os.path.join(ROOT, "results/datadecide/stats")
OUT = os.path.join(ROOT, "results/datadecide/dd_curve_forecast.json")
EVALS = ["wikitext_103", "pile", "m2d2_s2orc", "ice", "dolma_wiki", "dolma_stack", "dolma_reddit", "dolma_pes2o",
         "dolma_common-crawl", "dolma_books", "c4_en"]
TOKENS_PER_STEP = {"300M": 655360, "530M": 917504, "1B": 1441792}   # datadecide_targets.py
SAVE_EVERY = {"300M": 1250, "530M": 1250, "1B": 2500}               # ladder.py save_interval
SIZES = ["1B", "530M", "300M"]
LOG10K = np.round(np.arange(0.0, 16.0001, 0.01), 4)
NBOOT = 20000
SEED = 20261023
LN2 = np.log(2.0)


# ------------------------------------------------------------------------------------------------ loss curves
def curves(size, bpt):
    """Per-recipe bits/byte at each grid checkpoint (mean over complete seeds), per set and 11-set mean.
    Grid = regular save steps below the final step T, plus T (each run's last checkpoint)."""
    d = pd.read_parquet(PPL)
    d = d[d.params == size].drop(columns=[c for c in d.columns if c.startswith("__")])
    last = d.groupby(["data", "seed"])["step"].transform("max")
    T = float(d["step"].max())
    d = d[last >= 0.97 * T].copy()
    is_last = d["step"] == d.groupby(["data", "seed"])["step"].transform("max")
    d["tokens"] = d["step"] * TOKENS_PER_STEP[size]
    regular = (d["step"] % SAVE_EVERY[size] == 0) & (d["step"] < T)
    d = d[regular | is_last].copy()
    d.loc[is_last, "step"] = T
    for e in EVALS:
        d[e] = np.log(d[f"eval/{e}-validation/Perplexity"]) / LN2 / bpt[e]
    d["all11"] = d[EVALS].mean(axis=1)
    g = d.groupby(["data", "step"])
    mean = g[EVALS + ["all11", "tokens"]].mean()
    se = g["all11"].std() / np.sqrt(g["all11"].count())
    nseed = d.groupby("data")["seed"].nunique()
    steps = np.array(sorted(d["step"].unique()))
    return mean, se, steps, T, nseed


# ------------------------------------------------------------------------------------------------ information curves
class Info:
    """GPT-2 information curve of one evaluation set, h(n) in bits/byte against context bytes."""

    def __init__(self, src):
        p = src["ref"]["gpt2"]["paired"]
        rows = sorted(p["rows"], key=lambda r: r["context_bytes_exact_mean"])
        self.lx = np.log([r["context_bytes_exact_mean"] for r in rows])
        self.y = np.array([r["bits_per_byte"] for r in rows])
        b = src["bytes_per_token"]
        n = np.unique(np.round(np.geomspace(4, 64, 12))).astype(int)
        hn = np.exp(np.interp(np.log(n * b), self.lx, np.log(self.y)))
        q = np.polyfit(np.log(n * b), np.log(hn - self.y[-1]), 1)
        self.gamma, self.A = float(-q[0]), float(np.exp(q[1]))
        self.gamma_stored = float(src["ref"]["gpt2"]["gamma"]["4-64"]["gamma"])
        self.nu = 1.0 / (1.0 + self.gamma)

    def h_logn(self, logn):
        return np.interp(logn, self.lx, self.y)

    def S(self, log_bytes, log10K):
        """h(n0 (bD)^nu) for log training bytes (array, ...) and log10 K (array, k) -> (k, ...)."""
        lk = np.asarray(log10K, float).reshape(-1, *([1] * np.ndim(log_bytes))) * np.log(10.0)
        logn = self.nu * (np.log(self.A * self.gamma) - lk + log_bytes)
        return self.h_logn(logn)


# ------------------------------------------------------------------------------------------------ helpers
def nearest(steps, x):
    return float(steps[np.argmin(np.abs(np.log(steps) - np.log(x)))])


def ols(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def pairwise_acc(p, y):
    i, j = np.triu_indices(len(p), 1)
    sp, sy = np.sign(p[i] - p[j]), np.sign(y[i] - y[j])
    return float(np.mean(np.where(sp == 0, 0.5, (sp == sy).astype(float))))


# ------------------------------------------------------------------------------------------------ one size, one target
def run_target(names, fam, Y, TOK, steps, roles, M, logb, infos, sets, per_row_K=False):
    """Y[r] : (n_sets_target, n_steps) bits/byte (NaN where missing); sets: indices into infos per target row.
    Target rows are evaluated jointly: K is shared across rows (per_row_K=False, the design) or fitted per row
    (per_row_K=True, the recovery run added after scoring), level maps are per row.
    Returns predictions P[arm] (R, rows, steps) and fold info."""
    R, nrow, ns = Y.shape
    si = {s: k for k, s in enumerate(steps)}
    p1, p2, ds = si[roles["p1"]], si[roles["p2"]], si[roles["Dstar"]]
    T = si[steps[-1]]
    scored = np.array([k for k in range(ns) if steps[k] > roles["p2"]])
    logD = np.log(TOK)                                           # (R, ns) training tokens
    # shape table: Sg[r, row, k_grid, step]
    Sg = np.stack([np.stack([infos[sets[j]].S(logb[r] + logD[r], LOG10K) for j in range(nrow)]) for r in range(R)])
    # Sg: (R, rows, nK, ns)

    def fill_role(r, j, k):
        """Value at role checkpoint k, or nearest available."""
        if np.isfinite(Y[r, j, k]):
            return Y[r, j, k], k
        ok = np.flatnonzero(np.isfinite(Y[r, j]))
        kk = ok[np.argmin(np.abs(np.log(steps[ok]) - np.log(steps[k])))]
        return Y[r, j, kk], kk

    arms = ["F0", "P2", "P1", "B", "C", "P1b", "P2s", "F0f"]
    P = {a: np.full((R, nrow, ns), np.nan) for a in arms}
    folds = {}
    for g in sorted(set(fam)):
        te = [i for i in range(R) if fam[i] == g]
        tr = [i for i in range(R) if fam[i] != g]
        fold = {}
        for anchor_name, ak in (("Dstar", ds), ("final", T)):
            # K: least squares of (dL - dS) over training recipes, rows, scored checkpoints
            err = np.zeros((nrow, len(LOG10K)))
            for r in tr:
                for j in range(nrow):
                    La, ka = fill_role(r, j, ak)
                    dL = Y[r, j, scored] - La
                    dS = Sg[r, j][:, scored] - Sg[r, j][:, ka:ka + 1]
                    m = np.isfinite(dL)
                    err[j] += ((dL[m][None, :] - dS[:, m]) ** 2).sum(1)
            if per_row_K:
                kks = [int(np.argmin(err[j])) for j in range(nrow)]
            else:
                kks = [int(np.argmin(err.sum(0)))] * nrow
            kk = kks[0]
            fold[f"log10K_{anchor_name}"] = float(LOG10K[kk]) if not per_row_K else [float(LOG10K[q]) for q in kks]
            fold[f"K_at_grid_edge_{anchor_name}"] = bool(any(q in (0, len(LOG10K) - 1) for q in kks))
            coefs = []
            for j in range(nrow):
                ya = np.array([fill_role(r, j, ak)[0] for r in tr])
                w = ols(M[tr, j], ya)
                coefs.append(w.tolist())
                for r in te:
                    lev = w[0] + w[1] * M[r, j]
                    S = Sg[r, j, kks[j]]
                    if anchor_name == "Dstar":
                        P["F0"][r, j] = lev + S - S[ak]
                        # blind measured shape relative to D*
                        dl = np.nanmean(np.stack([Y[q, j] - fill_role(q, j, ak)[0] for q in tr]), 0)
                        P["B"][r, j] = lev + dl
                    else:
                        P["F0f"][r, j] = lev + S - S[ak]
            fold[f"level_coef_{anchor_name}"] = coefs
            if anchor_name == "Dstar":
                fold["K_Dstar_index"] = kks
        kks = fold["K_Dstar_index"]
        for j in range(nrow):
            Ctr = np.nanmean(np.stack([Y[q, j] for q in tr]), 0)
            for r in te:
                S = Sg[r, j, kks[j]]
                L1, k1 = fill_role(r, j, p1)
                L2, k2 = fill_role(r, j, p2)
                P["C"][r, j] = Ctr
                # power law through two pilots, no floor (core.A3 with E = 0)
                if L2 < L1:
                    a = np.log(L1 / L2) / np.log(TOK[r, k2] / TOK[r, k1])
                    P["P2"][r, j] = L2 * (TOK[r] / TOK[r, k2]) ** (-a)
                else:
                    P["P2"][r, j] = L2
                P["P1"][r, j] = L2 + S - S[k2]
                P["P2s"][r, j] = np.mean([L1 - S[k1], L2 - S[k2]]) + S
                dl = np.nanmean(np.stack([Y[q, j] - fill_role(q, j, k2)[0] for q in tr]), 0)
                P["P1b"][r, j] = L2 + dl
        fold.pop("K_Dstar_index")
        folds[g] = fold
    return P, folds, scored


def score(P, Y, fam, scored, steps, horizons, primary=("P2", "P1", "B"), rng_seed=SEED):
    """Errors over recipes. Per-set targets: per-recipe r.m.s. over rows at each step."""
    R = Y.shape[0]
    E = {a: P[a] - Y for a in P}                                   # (R, rows, ns)
    Er = {a: np.sqrt(np.nanmean(E[a] ** 2, axis=1)) if Y.shape[1] > 1 else np.abs(E[a][:, 0]) for a in E}
    Es = {a: (np.nanmean(E[a], axis=1) if Y.shape[1] > 1 else E[a][:, 0]) for a in E}   # signed (row mean)
    curve = {a: np.sqrt(np.nanmean(Er[a][:, scored] ** 2, axis=1)) for a in Er}       # (R,)
    out = {"arms": {}}
    hz = {f"step_{int(steps[k])}": k for k in horizons}
    for a in P:
        o = dict(curve_rms=float(np.sqrt(np.mean(curve[a] ** 2))), curve_median=float(np.median(curve[a])),
                 curve_max=float(np.max(curve[a])), horizons={})
        for name, k in hz.items():
            v = Er[a][:, k]
            o["horizons"][name] = dict(rms=float(np.sqrt(np.nanmean(v ** 2))), median_abs=float(np.nanmedian(v)),
                                       max_abs=float(np.nanmax(v)), mean_signed=float(np.nanmean(Es[a][:, k])))
        o["per_step_rms"] = [float(np.sqrt(np.nanmean(Er[a][:, k] ** 2))) for k in range(len(steps))]
        if Y.shape[1] == 1:
            o["pairwise_final"] = pairwise_acc(P[a][:, 0, -1], Y[:, 0, -1])
        # common-vs-recipe decomposition over scored steps: c(t) = mean over recipes of signed error
        c = np.nanmean(Es[a][:, scored], axis=0)
        rem = Es[a][:, scored] - c[None, :]
        o["decomp_common_rms"] = float(np.sqrt(np.nanmean(c ** 2)))
        o["decomp_recipe_rms"] = float(np.sqrt(np.nanmean(rem ** 2)))
        out["arms"][a] = o
    # paired family bootstrap
    fams = np.array(sorted(set(fam)))
    fam = np.array(fam)
    rng = np.random.default_rng(rng_seed)
    draws = rng.integers(0, len(fams), size=(NBOOT, len(fams)))
    idx = [np.concatenate([np.flatnonzero(fam == fams[d]) for d in row]) for row in draws]
    kT = len(steps) - 1
    comp = {}
    for b in [x for x in P if x != "F0"]:
        res = {}
        for metric in ("curve_rms", "curve_median", "final_rms", "final_median"):
            def m(a, ii):
                if metric.startswith("curve"):
                    v = curve[a][ii]
                else:
                    v = Er[a][ii, kT]
                return np.sqrt(np.mean(v ** 2)) if metric.endswith("rms") else np.median(v)
            pt = m("F0", np.arange(R)) - m(b, np.arange(R))
            bs = np.array([m("F0", ii) - m(b, ii) for ii in idx])
            res[metric] = dict(diff_F0_minus=float(pt), ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                               frac_F0_better=float(np.mean(bs < 0)))
        res["recipes_F0_better_curve"] = int(np.sum(curve["F0"] < curve[b]))
        comp[b] = res
    out["paired_vs_F0"] = comp
    out["primary_comparisons"] = list(primary)
    return out, curve, Er


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def diagnostics(P, Y, fam, scored, steps, roles, curve, Er, set_names, gammas=None):
    """Added after scoring (diagnosis only; arms and scores above are unchanged)."""
    si = {s: k for k, s in enumerate(steps)}
    kT, ds, kp2 = len(steps) - 1, si[roles["Dstar"]], si[roles["p2"]]
    R = Y.shape[0]
    fams = np.array(sorted(set(fam)))
    fama = np.array(fam)
    rng = np.random.default_rng(SEED + 1)
    draws = rng.integers(0, len(fams), size=(NBOOT, len(fams)))
    idx = [np.concatenate([np.flatnonzero(fama == fams[d]) for d in row]) for row in draws]
    # flat reference: the F0 level at D*, held constant (no shape)
    flat = np.repeat(P["F0"][:, :, ds:ds + 1], len(steps), axis=2)
    Ef = flat - Y
    Erf = np.sqrt(np.nanmean(Ef ** 2, axis=1)) if Y.shape[1] > 1 else np.abs(Ef[:, 0])
    curve = dict(curve, flat=np.sqrt(np.nanmean(Erf[:, scored] ** 2, axis=1)))
    Er = dict(Er, flat=Erf)
    out = dict(flat_curve_rms=float(np.sqrt(np.mean(curve["flat"] ** 2))),
               flat_final_rms=float(np.sqrt(np.nanmean(Erf[:, kT] ** 2))), extra_pairs={})
    for a, b in (("P1", "P2"), ("P1b", "P2"), ("P1", "P1b"), ("B", "P2"), ("F0", "flat")):
        res = {}
        for metric in ("curve_rms", "final_rms"):
            def m(x, ii):
                v = curve[x][ii] if metric == "curve_rms" else Er[x][ii, kT]
                return np.sqrt(np.mean(v ** 2))
            bs = np.array([m(a, ii) - m(b, ii) for ii in idx])
            res[metric] = dict(diff=float(m(a, np.arange(R)) - m(b, np.arange(R))),
                               ci95=[float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))])
        out["extra_pairs"][f"{a}_minus_{b}"] = res
    out["per_step_mean_signed"] = {a: [float(v) for v in np.nanmean(P[a] - Y, axis=(0, 1))] for a in ("F0", "P1", "P2", "B")}
    # mean falls per target row (mean over recipes)
    rows = []
    for j in range(Y.shape[1]):
        rows.append(dict(
            row=set_names[j],
            measured_fall_Dstar_to_T=float(np.nanmean(Y[:, j, ds] - Y[:, j, kT])),
            F0_shape_fall_Dstar_to_T=float(np.nanmean(P["F0"][:, j, ds] - P["F0"][:, j, kT])),
            measured_fall_p2_to_T=float(np.nanmean(Y[:, j, kp2] - Y[:, j, kT])),
            P1_shape_fall_p2_to_T=float(np.nanmean(P["P1"][:, j, kp2] - P["P1"][:, j, kT])),
            P2_powerlaw_fall_p2_to_T=float(np.nanmean(P["P2"][:, j, kp2] - P["P2"][:, j, kT]))))
    out["mean_falls"] = rows
    if Y.shape[1] > 1:
        mf = [r["measured_fall_p2_to_T"] for r in rows]
        out["across_sets_spearman_gamma_vs_measured_fall_p2_to_T"] = spearman(gammas, mf)
        out["across_sets_spearman"] = dict(
            P1_shape_vs_measured_fall_p2_to_T=spearman([r["P1_shape_fall_p2_to_T"] for r in rows], mf),
            F0_shape_vs_measured_fall_Dstar_to_T=spearman([r["F0_shape_fall_Dstar_to_T"] for r in rows],
                                                           [r["measured_fall_Dstar_to_T"] for r in rows]),
            P2_vs_measured_fall_p2_to_T=spearman([r["P2_powerlaw_fall_p2_to_T"] for r in rows], mf))
    return out


def main():
    bpt ={k: v["bytes_per_token"] for k, v in json.load(open(os.path.join(ROOT, "results/datadecide/eval_bpt.json"))).items()}
    F, _ = VH.load_features()
    names = list(F.index)
    lv = pd.read_csv(os.path.join(ROOT, "results/datadecide/final_levels.csv"))
    famof = lv.groupby("data")["family"].first()
    fam = [famof[n] for n in names]
    merged = json.load(open(os.path.join(ST, "merged.json")))["sources"]
    infos = [Info(merged[f"ev_{e}"]) for e in EVALS]
    for e, inf in zip(EVALS, infos):
        assert abs(inf.gamma - inf.gamma_stored) < 1e-9, (e, inf.gamma, inf.gamma_stored)
    # per-set mismatch, as in verify_datadecide_headline.load_features
    X = json.load(open(os.path.join(ST, "xent.json")))["xent"]
    slug_of = {v["name"]: k for k, v in merged.items() if k.startswith("rc_")}
    Mset = np.array([[(X[slug_of[n]][f"ev_{e}"] - X[f"ev_{e}"][f"ev_{e}"]) / merged[f"ev_{e}"]["bytes_per_token"]
                      for e in EVALS] for n in names])
    assert np.allclose(Mset.mean(1), F["mismatch_all"].values)
    logb = np.log(F["bytes_per_token"].values)

    out = dict(design="docs/analysis/dd_curve_forecast.md", ppl=PPL, recipes=names, families=fam,
               info_curves={e: dict(gamma=i.gamma, A_bytes=i.A, nu=i.nu, h_512tok=float(i.y[-1]),
                                    max_context_bytes=float(np.exp(i.lx[-1]))) for e, i in zip(EVALS, infos)},
               sizes={})
    for size in SIZES:
        mean, se, steps, T, nseed = curves(size, bpt)
        roles = dict(p1=nearest(steps[steps < T], T / 28), p2=nearest(steps[steps < T], T / 14),
                     Dstar=nearest(steps, T / 4), final=T)
        R, ns = len(names), len(steps)
        Y11 = np.full((R, 1, ns), np.nan)
        Yset = np.full((R, len(EVALS), ns), np.nan)
        TOK = np.full((R, ns), np.nan)
        for i, n in enumerate(names):
            sub = mean.loc[n].reindex(steps)
            Y11[i, 0] = sub["all11"].values
            Yset[i] = sub[EVALS].values.T
            TOK[i] = np.where(np.isfinite(sub["tokens"].values), sub["tokens"].values, steps * TOKENS_PER_STEP[size])
        horizons = [k for k in range(ns) if steps[k] in (nearest(steps, T / 8), roles["Dstar"], nearest(steps, T / 2), T)]
        res = dict(final_step=T, roles=roles, n_seeds={n: int(nseed[n]) for n in names}, steps=steps.tolist(),
                   tokens_final_mean=float(np.nanmean(TOK[:, -1])),
                   seed_se_all11_median_per_step=[float(np.nanmedian(se.xs(s, level="step"))) if s in
                                                  se.index.get_level_values("step") else None for s in steps],
                   targets={})
        jobs = [("all11", Y11, Mset.mean(1, keepdims=True), [None], False)]
        if size == "1B":
            jobs.append(("per_set", Yset, Mset, list(range(len(EVALS))), False))
            # recovery added after scoring: placement K fitted per evaluation set (training families only)
            jobs.append(("per_set_recovery_K_per_set", Yset, Mset, list(range(len(EVALS))), True))
        for tag, Y, M, sets, prk in jobs:
            if tag == "all11":
                # 11-set mean shape: average of per-set shapes; handled by a wrapper info object
                class Mean:
                    def S(self, lb, lk):
                        return np.mean([inf.S(lb, lk) for inf in infos], axis=0)
                inf_list, set_idx = [Mean()], [0]
            else:
                inf_list, set_idx = infos, sets
            P, folds, scored = run_target(names, fam, Y, TOK, steps, roles, M, logb, inf_list, set_idx, per_row_K=prk)
            sc, curve, Er = score(P, Y, fam, scored, steps, horizons)
            sc["folds"] = folds
            sc["n_scored_checkpoints"] = int(len(scored))
            sc["diagnostics_after_scoring"] = diagnostics(P, Y, fam, scored, steps, roles, curve, Er,
                                                          ["all11"] if tag == "all11" else EVALS,
                                                          [i.gamma for i in infos])
            sc["per_recipe_curve_rms"] = {a: dict(zip(names, map(float, curve[a]))) for a in curve}
            if tag == "all11":
                sc["measured"] = {n: [None if not np.isfinite(v) else float(v) for v in Y[i, 0]] for i, n in enumerate(names)}
                sc["predicted"] = {a: {n: [None if not np.isfinite(v) else float(v) for v in P[a][i, 0]]
                                       for i, n in enumerate(names)} for a in P}
            res["targets"][tag] = sc
            print(size, tag, {a: round(sc["arms"][a]["curve_rms"], 4) for a in sc["arms"]},
                  "final", {a: round(sc["arms"][a]["horizons"][f"step_{int(T)}"]["rms"], 4) for a in sc["arms"]})
        out["sizes"][size] = res
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
