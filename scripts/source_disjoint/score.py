"""Source-disjoint selection test.

    python scripts/source_disjoint/score.py   -> results/source_disjoint/score.json

Tiers: loss on ICE (strict) and on the paper's four dataset-disjoint sets; benchmarks on the four crowd-written tasks
with the 17 Flan-free recipes (strict) and with all 25; all 10 tasks as reference. Pairwise accuracy on pairs resolved
by 2 pooled seed SDs, all-pairs accuracy, top-choice regret; bootstrap over recipe families x sets/tasks (4,000 draws).
"""
import itertools, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bench_mismatch_score import TASKS  # noqa: E402
from datadecide_predict import FAMILIES  # noqa: E402

OUT = os.path.join(ROOT, "results", "source_disjoint")
B, SEED = 4000, 20261106
SIZES = ["4M", "20M", "60M", "150M", "300M"]
fam_of = {r: f for f, (_, rs) in FAMILIES.items() for r in rs}
R_ALL = sorted(fam_of)
FLAN = ["Dolma1.6++", "Dolma1.7", "Dolma1.7 (no Reddit)", "Dolma1.7 (no code)", "Dolma1.7 (no math, code)",
        "DCLM-Baseline 25% / Dolma 75%", "DCLM-Baseline 50% / Dolma 50%", "DCLM-Baseline 75% / Dolma 25%"]
assert all(r in R_ALL for r in FLAN)
R_NOFLAN = [r for r in R_ALL if r not in FLAN]
CROWD = ["commonsense_qa", "openbookqa", "social_iqa", "winogrande"]
OOD4 = ["pile", "wikitext_103", "m2d2_s2orc", "ice"]
ci = lambda x: [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]  # noqa: E731


# ---------------------------------------------------------------------------------------------------- inputs
def bench_inputs():
    o = pd.read_csv(os.path.join(ROOT, "results/bench_mismatch_score/outcomes_1B_final_by_seed.csv"))
    T = list(TASKS)
    Ym, SD = {}, {}
    for t in T:
        g = o[o.task == TASKS[t]].groupby("data").acc
        Ym[t] = g.mean().reindex(R_ALL)
        SD[t] = float(np.sqrt((g.std(ddof=1).reindex(R_ALL) ** 2).mean()))
    fc = json.load(open(os.path.join(ROOT, "results/bench_mismatch/forecast.json")))["values"]
    fw = json.load(open(os.path.join(ROOT, "results/bench_mismatch/forecast_word_rival.json")))["values"]
    px = json.load(open(os.path.join(ROOT, "results/task_reversals/proxy_by_recipe_task.json")))["values"]
    S = {"bigram": {t: {r: -fc["ours_M_task"][r][t] for r in R_ALL} for t in T},
         "token unigram": {t: {r: -fc["rival_unigram_ce"][r][t] for r in R_ALL} for t in T},
         "word unigram": {t: {r: -fw[r][t] for r in R_ALL} for t in T}}
    qf = os.path.join(OUT, "qonly_stats.json")
    if os.path.exists(qf):
        Q = json.load(open(qf))
        if sorted(Q) == R_ALL:
            for nm, key in [("bigram (question only)", "M_task_qonly"), ("token unigram (question only)", "unigram_ce_qonly"),
                            ("word unigram (question only)", "word_ce_qonly")]:
                S[nm] = {t: {r: -Q[r][key][t] for r in R_ALL} for t in T}
    for s in SIZES:
        for m, tag in [("primary_metric", "acc"), ("correct_prob_per_char", "cppc")]:
            S[f"proxy {s} {tag}"] = {t: {r: px[s][TASKS[t]][r][m] for r in R_ALL} for t in T}
    return Ym, SD, S


def loss_inputs():
    lv = pd.read_csv(os.path.join(ROOT, "results/datadecide/final_levels.csv"))
    X = json.load(open(os.path.join(ROOT, "results/datadecide/stats/xent.json")))["xent"]
    C = json.load(open(os.path.join(ROOT, "results/datadecide/evalaware_rivals_cache.json")))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import datadecide_stats as dds
    slug = {dds.slug_name(s): s for s in dds.slugs("recipe")}
    sets = sorted(lv["eval"].unique())
    Ym, SD, S = {}, {}, {}
    for e in sets:
        d = lv[(lv["eval"] == e) & (lv.params == "1B")].set_index("data").reindex(R_ALL)
        Ym[e] = -d.bpb  # higher = better
        sd_bpb = d.sd * d.bpb / d["mean"]
        SD[e] = float(np.sqrt((sd_bpb ** 2).mean()))
    S["bigram"] = {e: {r: -(X[slug[r]][f"ev_{e}"] - X[f"ev_{e}"][f"ev_{e}"]) for r in R_ALL} for e in sets}
    S["token unigram"] = {e: {r: -C[slug[r]]["tok"][f"ev_{e}"] for r in R_ALL} for e in sets}
    S["word unigram"] = {e: {r: -C[slug[r]]["word"][f"ev_{e}"] for r in R_ALL} for e in sets}
    for s in SIZES:
        S[f"proxy {s} loss"] = {}
        for e in sets:
            d = lv[(lv["eval"] == e) & (lv.params == s)].set_index("data").reindex(R_ALL)
            assert d.bpb.notna().all(), (s, e)
            S[f"proxy {s} loss"][e] = (-d.bpb).to_dict()
    tokens = lv[lv["eval"] == "ice"].groupby("params").tokens.first().to_dict()
    return Ym, SD, S, tokens


# ---------------------------------------------------------------------------------------------------- scoring
def evaluate(Ym, SD, S, units, recipes, scale, preds, k=2.0):
    """units: sets/tasks; recipes: candidate list; scale: 100 for accuracy points, 1000 for milli-bits per byte."""
    R = recipes
    fams = sorted({fam_of[r] for r in R})
    fidx = np.array([fams.index(fam_of[r]) for r in R])
    pairs = np.array(list(itertools.combinations(range(len(R)), 2)))
    pi, pj = pairs[:, 0], pairs[:, 1]
    Y = np.array([Ym[u].reindex(R).values for u in units], float)
    sd = np.array([SD[u] for u in units])
    dY = Y[:, pi] - Y[:, pj]
    res = np.abs(dY) >= k * sd[:, None]
    Sm = {p: np.array([[S[p][u][r] for r in R] for u in units], float) for p in preds}
    corr = {}
    for p in preds:
        dS = Sm[p][:, pi] - Sm[p][:, pj]
        corr[p] = np.where(dS == 0, 0.5, (np.sign(dS) == np.sign(dY)).astype(float))

    def pair_acc(wt, wf, mask):
        wp = wf[:, fidx[pi]] * wf[:, fidx[pj]]
        Wc = wt[:, :, None] * wp[:, None, :] * mask[None]
        den = Wc.sum((1, 2))
        return {p: (Wc * corr[p][None]).sum((1, 2)) / den for p in preds}

    def regret(wt, cand):
        Yc = np.where(cand[None], Y, -np.inf)
        best = Yc.max(1)
        out = {}
        for p in preds:
            Sc = np.where(cand[None], Sm[p], -np.inf)
            top = Sc == Sc.max(1, keepdims=True)
            chosen = np.where(top, Y, 0).sum(1) / top.sum(1)
            out[p] = float((wt * scale * (best - chosen)).sum() / wt.sum())
        rnd = best - np.where(cand[None], Y, 0).sum(1) / cand.sum()
        out["random"] = float((wt * scale * rnd).sum() / wt.sum())
        within = {p: int(((best - (np.where(Sm[p] == np.where(cand[None], Sm[p], -np.inf).max(1, keepdims=True), Y, 0).sum(1)
                                   / (Sm[p] == np.where(cand[None], Sm[p], -np.inf).max(1, keepdims=True)).sum(1)))
                          <= k * sd + 1e-12).sum()) for p in preds}
        return out, within

    one_t, one_f = np.ones((1, len(units))), np.ones((1, len(fams)))
    point = {"resolved": {p: float(v[0]) for p, v in pair_acc(one_t, one_f, res).items()},
             "all_pairs": {p: float(v[0]) for p, v in pair_acc(one_t, one_f, np.ones_like(res)).items()}}
    reg_point, within = regret(np.ones(len(units)), np.ones(len(R), bool))
    rng = np.random.default_rng(SEED)
    boot = {m: {p: [] for p in preds} for m in point}
    breg = {p: [] for p in preds + ["random"]}
    resample_units = len(units) > 1
    for c in range(0, B, 250):
        D = min(250, B - c)
        wt = (np.stack([np.bincount(rng.integers(0, len(units), len(units)), minlength=len(units)) for _ in range(D)])
              if resample_units else np.ones((D, 1))).astype(float)
        wf = np.stack([np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)) for _ in range(D)]).astype(float)
        for m, mask in [("resolved", res), ("all_pairs", np.ones_like(res))]:
            pa = pair_acc(wt, wf, mask)
            for p in preds:
                boot[m][p].append(pa[p])
        for d in range(D):
            ro, _ = regret(wt[d], wf[d][fidx] > 0)
            for p in ro:
                breg[p].append(ro[p])
    boot = {m: {p: np.concatenate(v) for p, v in dd.items()} for m, dd in boot.items()}
    breg = {p: np.array(v) for p, v in breg.items()}
    uni = max(["token unigram", "word unigram"], key=lambda p: point["resolved"][p])
    contrasts = {}
    for a, b in ([("bigram", "token unigram"), ("bigram", "word unigram")] + [("bigram", p) for p in preds if p.startswith("proxy 4M")] +
                 ([("bigram (question only)", "token unigram (question only)"), ("bigram (question only)", "word unigram (question only)"),
                   ("bigram (question only)", "bigram")] if "bigram (question only)" in preds else [])):
        contrasts[f"{a} minus {b}"] = {
            "resolved": {"point": point["resolved"][a] - point["resolved"][b], "ci95": ci(boot["resolved"][a] - boot["resolved"][b])},
            "all_pairs": {"point": point["all_pairs"][a] - point["all_pairs"][b], "ci95": ci(boot["all_pairs"][a] - boot["all_pairs"][b])},
            "regret_b_minus_a": {"point": reg_point[b] - reg_point[a], "ci95": ci(breg[b] - breg[a])}}
    return {"units": units, "n_recipes": len(R), "families": fams, "pairs": int(len(pairs)),
            "resolved_cells": int(res.sum()), "resolved_by_unit": {u: int(res[a].sum()) for a, u in enumerate(units)},
            "pooled_seed_sd": dict(zip(units, sd.tolist())), "stronger_unigram": uni,
            "pairwise_resolved": {p: {"point": point["resolved"][p], "ci95": ci(boot["resolved"][p])} for p in preds},
            "pairwise_all": {p: {"point": point["all_pairs"][p], "ci95": ci(boot["all_pairs"][p])} for p in preds},
            "regret": {p: {"point": reg_point[p], "ci95": ci(breg[p])} for p in preds + ["random"]},
            "regret_units": "accuracy points" if scale == 100 else "milli-bits per byte",
            "units_within_2sd_of_best": within, "contrasts": contrasts,
            "bootstrap": {"draws": B, "seed": SEED, "resampled": ["recipe families"] + (["sets/tasks"] if resample_units else [])}}


def main():
    res = {"spec": "docs/analysis/source_disjoint.md", "flan_recipes": FLAN, "flan_free_recipes": R_NOFLAN}
    Ym, SD, S, tokens = loss_inputs()
    lp = ["bigram", "token unigram", "word unigram"] + [f"proxy {s} loss" for s in SIZES]
    res["loss_strict_ice"] = evaluate(Ym, SD, S, ["ice"], R_ALL, 1000, lp)
    res["loss_dataset_disjoint_4"] = evaluate(Ym, SD, S, OOD4, R_ALL, 1000, lp)
    res["loss_ice_flan_free"] = evaluate(Ym, SD, S, ["ice"], R_NOFLAN, 1000, lp)
    res["training_tokens"] = tokens
    Ym, SD, S = bench_inputs()
    bp = [p for p in S]
    res["bench_strict_crowd4_flan_free"] = evaluate(Ym, SD, S, CROWD, R_NOFLAN, 100, bp)
    res["bench_crowd4_all25"] = evaluate(Ym, SD, S, CROWD, R_ALL, 100, bp)
    res["bench_all10_all25"] = evaluate(Ym, SD, S, list(TASKS), R_ALL, 100, bp)
    res["bench_all10_flan_free"] = evaluate(Ym, SD, S, list(TASKS), R_NOFLAN, 100, bp)
    json.dump(res, open(os.path.join(OUT, "score.json"), "w"), indent=1)
    for k, v in res.items():
        if not isinstance(v, dict) or "pairwise_resolved" not in v:
            continue
        print(f"\n[{k}] recipes {v['n_recipes']} resolved {v['resolved_cells']} {v['resolved_by_unit']}")
        for p in v["pairwise_resolved"]:
            a, b, g = v["pairwise_resolved"][p], v["pairwise_all"][p], v["regret"][p]
            print(f"  {p:32s} res {a['point']:.3f} [{a['ci95'][0]:.2f},{a['ci95'][1]:.2f}]  all {b['point']:.3f}  "
                  f"regret {g['point']:.2f} [{g['ci95'][0]:.2f},{g['ci95'][1]:.2f}]")
        print(f"  random regret {v['regret']['random']['point']:.2f}")
        for c, d in v["contrasts"].items():
            print(f"  {c}: res {d['resolved']['point']:+.3f} {np.round(d['resolved']['ci95'], 3)}  "
                  f"regret gain {d['regret_b_minus_a']['point']:+.2f} {np.round(d['regret_b_minus_a']['ci95'], 2)}")


if __name__ == "__main__":
    main()
