"""Task-specific ranking reversals on the 10 DataDecide benchmarks at 1B.

Does evaluation-aware mismatch follow the recipe orderings that change from one benchmark to another? Scores reversal
accuracy, minority-side accuracy, flip capture, all-resolved accuracy and top-choice regret for data statistics, global
rankings and DataDecide small-model proxies; bootstrap over tasks x recipe families (4,000 draws, seed 20261104).
Writes results/task_reversals/reversals_k{2,3}.json.
"""
import itertools, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bench_mismatch_score import TASKS  # noqa: E402
from datadecide_predict import FAMILIES  # noqa: E402

OUT = os.path.join(ROOT, "results", "task_reversals")
B, SEED = 4000, 20261104
FLAGGED = ["boolq", "openbookqa", "social_iqa", "winogrande"]
SIZES = ["4M", "20M", "60M", "150M", "300M"]

fam_of = {r: f for f, (_, rs) in FAMILIES.items() for r in rs}
R = sorted(fam_of)
fams = list(FAMILIES)
fidx = np.array([fams.index(fam_of[r]) for r in R])
T = list(TASKS)
pairs = np.array(list(itertools.combinations(range(len(R)), 2)))
pi, pj = pairs[:, 0], pairs[:, 1]

# 1B outcomes: seed mean and pooled seed SD per task
o = pd.read_csv(os.path.join(ROOT, "results/bench_mismatch_score/outcomes_1B_final_by_seed.csv"))
Y, SD = np.zeros((len(T), len(R))), np.zeros(len(T))
for a, t in enumerate(T):
    g = o[o.task == TASKS[t]].groupby("data").acc
    assert (g.size().reindex(R) == 3).all()
    Y[a] = g.mean().reindex(R).values
    SD[a] = np.sqrt((g.std(ddof=1).reindex(R) ** 2).mean())

# predictors as score matrices S[task, recipe], higher score = predicted higher accuracy
fc = json.load(open(os.path.join(ROOT, "results/bench_mismatch/forecast.json")))
fw = json.load(open(os.path.join(ROOT, "results/bench_mismatch/forecast_word_rival.json")))
px = json.load(open(os.path.join(OUT, "proxy_by_recipe_task.json")))["values"]
neg = lambda v: -np.array([[v[r][t] for r in R] for t in T])
S = {"bigram_mismatch_task": neg(fc["values"]["ours_M_task"]),
     "token_unigram_ce_task": neg(fc["values"]["rival_unigram_ce"]),
     "word_unigram_ce_task": neg(fw["values"])}
S["global_bigram_task_avg"] = np.tile(S["bigram_mismatch_task"].mean(0), (len(T), 1))
S["global_mix11_mismatch"] = np.tile(-np.array([fc["values"]["secondary_M_mix11"][r] for r in R]), (len(T), 1))
Z = (Y - Y.mean(1, keepdims=True)) / Y.std(1, ddof=1, keepdims=True)
S["global_1B_leave_task_out"] = np.array([np.delete(Z, a, 0).mean(0) for a in range(len(T))])
for s in SIZES:
    for m, tag in [("primary_metric", "acc"), ("correct_prob_per_char", "cppc")]:
        S[f"proxy_{s}_{tag}"] = np.array([[px[s][TASKS[t]][r][m] for r in R] for t in T])
P = list(S)
GROUP = {"bigram_mismatch_task": "task-specific data statistic", "token_unigram_ce_task": "task-specific data statistic",
         "word_unigram_ce_task": "task-specific data statistic", "global_bigram_task_avg": "global data statistic",
         "global_mix11_mismatch": "global data statistic",
         "global_1B_leave_task_out": "global, outcome-informed comparator (uses other tasks' 1B accuracy)"}
GROUP.update({p: "task-specific trained proxy" for p in P if p.startswith("proxy")})


def run(k):
    # cells (task, pair)
    dY = Y[:, pi] - Y[:, pj]                      # T x npairs
    res = np.abs(dY) >= k * SD[:, None]
    sgn = np.sign(dY) * res
    rev = (sgn > 0).any(0) & (sgn < 0).any(0)     # reversing pairs
    npos, nneg = (sgn > 0).sum(0), (sgn < 0).sum(0)
    zsum = (dY / SD[:, None] * res).sum(0)
    maj = np.where(npos != nneg, np.sign(npos - nneg), np.sign(zsum))
    corr = {}
    for p in P:
        dS = S[p][:, pi] - S[p][:, pj]
        corr[p] = np.where(dS == 0, 0.5, (np.sign(dS) == np.sign(dY)).astype(float))
    cell_rev = res & rev[None, :]
    cell_min = cell_rev & (sgn == -maj[None, :])
    # flips: (pair, t, t') with opposite resolved signs
    fl = [(a, b, q) for q in np.where(rev)[0] for a, b in itertools.combinations(range(len(T)), 2)
          if sgn[a, q] * sgn[b, q] < 0]
    fl = np.array(fl, dtype=int).reshape(-1, 3)

    def cell_scores(wt, wf):
        """wt: (D, T) task counts, wf: (D, F) family counts -> dict of metric -> (D,) per predictor."""
        wp = wf[:, fidx[pi]] * wf[:, fidx[pj]]                  # D x npairs
        W = wt[:, :, None] * wp[:, None, :]                     # D x T x npairs
        out = {}
        for name, mask in [("reversal_accuracy", cell_rev), ("minority_side_accuracy", cell_min),
                           ("all_resolved_accuracy", res)]:
            Wm = W * mask[None]
            den = Wm.sum((1, 2))
            out[name] = {p: (Wm * corr[p][None]).sum((1, 2)) / den for p in P}
        if len(fl):
            Wf = wt[:, fl[:, 0]] * wt[:, fl[:, 1]] * wp[:, fl[:, 2]]
            out["flip_capture"] = {p: (Wf * (corr[p][fl[:, 0], fl[:, 2]] * corr[p][fl[:, 1], fl[:, 2]])[None]).sum(1)
                                   / Wf.sum(1) for p in P}
        return out

    def regret(wt, cand, tasks_mask):
        """cand: bool (R,) candidate recipes; returns per-predictor (mean regret pts, n within 2 SD) and random."""
        w = wt * tasks_mask
        r_out, n_out = {}, {}
        Yc = np.where(cand[None], Y, -np.inf)
        best = Yc.max(1)
        for p in P:
            Sc = np.where(cand[None], S[p], -np.inf)
            top = Sc == Sc.max(1, keepdims=True)
            chosen = (np.where(top, Y, 0).sum(1) / top.sum(1))
            reg = 100 * (best - chosen)
            r_out[p] = float((w * reg).sum() / w.sum())
            n_out[p] = float((w * (best - chosen <= 2 * SD + 1e-12)).sum() / w.sum() * tasks_mask.sum())
        rnd = 100 * (best - np.where(cand[None], Y, 0).sum(1) / cand.sum())
        return r_out, n_out, float((w * rnd).sum() / w.sum())

    allm, unfl = np.ones(len(T)), np.array([t not in FLAGGED for t in T], float)
    point = cell_scores(np.ones((1, len(T))), np.ones((1, len(fams))))
    point = {m: {p: float(v[0]) for p, v in d.items()} for m, d in point.items()}
    reg_all, within_all, rnd_all = regret(allm, np.ones(len(R), bool), allm)
    reg_unf, within_unf, rnd_unf = regret(allm, np.ones(len(R), bool), unfl)

    rng = np.random.default_rng(SEED)
    boot = {m: {p: [] for p in P} for m in point}
    breg = {"all10": {p: [] for p in P + ["random"]}, "unflagged6": {p: [] for p in P + ["random"]}}
    for c in range(0, B, 250):
        D = min(250, B - c)
        wt = np.stack([np.bincount(rng.integers(0, len(T), len(T)), minlength=len(T)) for _ in range(D)]).astype(float)
        wf = np.stack([np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)) for _ in range(D)]).astype(float)
        cs = cell_scores(wt, wf)
        for m in cs:
            for p in P:
                boot[m][p].append(cs[m][p])
        for d in range(D):
            cand = wf[d][fidx] > 0
            for key, msk in [("all10", allm), ("unflagged6", unfl)]:
                if (wt[d] * msk).sum() == 0:
                    continue
                ro, _, rr = regret(wt[d], cand, msk)
                for p in P:
                    breg[key][p].append(ro[p])
                breg[key]["random"].append(rr)
    boot = {m: {p: np.concatenate(v) for p, v in d.items()} for m, d in boot.items()}
    breg = {k: {p: np.array(v) for p, v in d.items()} for k, d in breg.items()}
    ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]

    uni = max(["token_unigram_ce_task", "word_unigram_ce_task"], key=lambda p: point["reversal_accuracy"][p])
    big = "bigram_mismatch_task"
    out = {"spec": "docs/analysis/task_reversals.md", "threshold_sd": k,
           "bootstrap": {"draws": B, "seed": SEED, "units": "10 tasks x 6 recipe families, resampled with replacement",
                         "cell_weight": "count(task) * count(family_i) * count(family_j)",
                         "regret_candidates": "union of recipes in drawn families"},
           "counts": {"resolved_cells": int(res.sum()), "reversing_pairs": int(rev.sum()),
                      "reversal_instances": int(cell_rev.sum()), "minority_side_instances": int(cell_min.sum()),
                      "flips": int(len(fl)), "pairs_total": int(len(pairs)),
                      "reversing_pairs_by_task": {t: int(cell_rev[a].sum()) for a, t in enumerate(T)}},
           "predictor_group": GROUP,
           "metrics": {m: {p: {"point": point[m][p], "ci95": ci(boot[m][p])} for p in P} for m in point},
           "regret_pts": {"all10": {"point": {p: reg_all[p] for p in P}, "random_expected": rnd_all,
                                    "tasks_within_2sd_of_best": within_all,
                                    "ci95": {p: ci(breg["all10"][p]) for p in P + ["random"]}},
                          "unflagged6": {"point": {p: reg_unf[p] for p in P}, "random_expected": rnd_unf,
                                         "tasks_within_2sd_of_best": within_unf,
                                         "ci95": {p: ci(breg["unflagged6"][p]) for p in P + ["random"]}}},
           "key_contrast": {"stronger_unigram": uni,
                            "bigram_minus_unigram": {m: {"point": point[m][big] - point[m][uni],
                                                         "ci95": ci(boot[m][big] - boot[m][uni])} for m in point},
                            "regret_unigram_minus_bigram_pts": {
                                key: {"point": d["point"][uni] - d["point"][big],
                                      "ci95": ci(breg[key][uni] - breg[key][big])}
                                for key, d in [("all10", {"point": reg_all}), ("unflagged6", {"point": reg_unf})]}},
           "bigram_minus_each": {m: {p: {"point": point[m][big] - point[m][p], "ci95": ci(boot[m][big] - boot[m][p])}
                                     for p in P if p != big} for m in point}}
    return out


if __name__ == "__main__":
    for k in (2, 3):
        r = run(k)
        f = os.path.join(OUT, f"reversals_k{k}.json")
        json.dump(r, open(f, "w"), indent=1)
        print("wrote", f, r["counts"])
