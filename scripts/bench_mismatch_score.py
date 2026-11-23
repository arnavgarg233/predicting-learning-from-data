"""Score the frozen benchmark-mismatch forecast against DataDecide's released 1B downstream results.

Inputs (read only, SHA-256 checked):
  results/bench_mismatch/forecast.json             ours_M_task, rival_unigram_ce (token), secondary_M_mix11
  results/bench_mismatch/forecast_word_rival.json  rival_word_unigram_ce (word level)
Outcomes: allenai/DataDecide-eval-results, data/macro_avg-00000-of-00001.parquet (pinned revision), params == 1B,
final step, field metrics.primary_metric; seeds averaged.
Score: within each task, pairwise agreement over the 300 recipe pairs, dropping pairs with |acc diff| < 2 x pooled
seed SD; pooled over tasks; paired bootstrap over the six recipe families (FAMILIES), 2000 resamples.
Writes results/bench_mismatch_score/.
"""
import hashlib, itertools, json, os, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from datadecide_predict import FAMILIES  # noqa: E402

REPO, REV = "allenai/DataDecide-eval-results", "9919b5a0e61e57a85021263918fa82d6ceaee038"
FILE = "data/macro_avg-00000-of-00001.parquet"
FROZEN = {"results/bench_mismatch/forecast.json": "a4bf679efde72c41a0cd90189df8977ca8d1240118bbd751cb0efe1dfa218d78",
          "results/bench_mismatch/forecast_word_rival.json": "1c76fab97ad1b0dbcc5251a89082ffc48fea35e75b493e2a2b7faea349adcd98"}
TASKS = {"arc_challenge": "arc_challenge", "arc_easy": "arc_easy", "boolq": "boolq", "commonsense_qa": "csqa",
         "hellaswag": "hellaswag", "mmlu": "mmlu", "openbookqa": "openbookqa", "piqa": "piqa",
         "social_iqa": "socialiqa", "winogrande": "winogrande"}
# chance under the OLMES rank-classification format (BoolQ: majority class "yes" on validation is 62.2%)
CHANCE = {"arc_challenge": .25, "arc_easy": .25, "boolq": .622, "commonsense_qa": .20, "hellaswag": .25, "mmlu": .25,
          "openbookqa": .25, "piqa": .50, "social_iqa": 1 / 3, "winogrande": .50}
PRED = ["ours_M_task", "rival_token_unigram_ce", "rival_word_unigram_ce", "secondary_M_mix11"]
OUT = os.path.join(ROOT, "results", "bench_mismatch_score")
B, SEED = 2000, 20261016


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


EXTRACT = os.path.join(ROOT, "results", "bench_mismatch_score", "outcomes_extract.parquet")


def outcome_parquet(cache):
    """(path, SHA-256 of the pinned source file). Uses the pinned DataDecide file (downloaded into cache if needed); when it
    is absent and the shipped extract exists (scripts/bench_mismatch_extract.py: the final-checkpoint rows of the sizes
    and tasks used here, cut from that file), reads the extract instead, which records the source file's SHA-256."""
    full = os.path.join(cache, FILE)
    if os.path.exists(full):
        return full, sha(full)
    if os.path.exists(EXTRACT):
        return EXTRACT, pq.read_metadata(EXTRACT).metadata[b"source_sha256"].decode()
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REV, local_dir=cache)
    return p, sha(p)


def load_outcomes(cache):
    p, src_sha = outcome_parquet(cache)
    pf, parts = pq.ParquetFile(p), []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg)
        t = t.filter(pc.equal(t["params"], "1B"))
        if t.num_rows:
            parts.append(t)
    d = pa.concat_tables(parts).to_pandas()
    d = d[d.step == d.groupby(["data", "seed", "task"]).step.transform("max")]
    d = d[d.task.isin(TASKS.values())].copy()
    d["acc"] = d.metrics.map(lambda s: json.loads(s)["primary_metric"])
    return d[["data", "task", "seed", "step", "tokens", "acc"]], p, src_sha


def main(cache):
    os.makedirs(OUT, exist_ok=True)
    for f, h in FROZEN.items():
        assert sha(os.path.join(ROOT, f)) == h, f"SHA-256 mismatch {f}"
    fc = json.load(open(os.path.join(ROOT, "results/bench_mismatch/forecast.json")))
    fw = json.load(open(os.path.join(ROOT, "results/bench_mismatch/forecast_word_rival.json")))
    vals = {"ours_M_task": fc["values"]["ours_M_task"], "rival_token_unigram_ce": fc["values"]["rival_unigram_ce"],
            "rival_word_unigram_ce": fw["values"]}
    sec = fc["values"]["secondary_M_mix11"]

    out, src_path, src_sha = load_outcomes(cache)
    out.to_csv(os.path.join(OUT, "outcomes_1B_final_by_seed.csv"), index=False)
    fam_of = {r: f for f, (_, rs) in FAMILIES.items() for r in rs}
    recipes = sorted(fam_of)
    assert set(recipes) == set(out.data) and len(recipes) == 25
    fams = list(FAMILIES)
    fidx = np.array([fams.index(fam_of[r]) for r in recipes])
    pairs = np.array(list(itertools.combinations(range(25), 2)))
    pi, pj = pairs[:, 0], pairs[:, 1]

    per_task, arrays = {}, {}
    for t, dd in TASKS.items():
        g = out[out.task == dd].groupby("data").acc
        mean, sd, n = g.mean().reindex(recipes), g.std(ddof=1).reindex(recipes), g.size().reindex(recipes)
        assert (n == 3).all()
        pooled_sd = float(np.sqrt((sd ** 2).mean()))
        y = mean.values
        dy = y[pi] - y[pj]
        keep = np.abs(dy) >= 2 * pooled_sd
        corr = {}
        for p in PRED:
            x = np.array([sec[r] if p == "secondary_M_mix11" else vals[p][r][t] for r in recipes])
            dx = x[pi] - x[pj]
            # lower statistic predicts higher accuracy: agreement when sign(dx) == -sign(dy); exact predictor tie = 0.5
            corr[p] = np.where(dx == 0, 0.5, (np.sign(dx) == -np.sign(dy)).astype(float))
        arrays[t] = (keep, corr)
        spread = float(np.std(y, ddof=1))
        per_task[t] = {"datadecide_task": dd, "chance": CHANCE[t], "mean_acc": float(y.mean()),
                       "min_acc": float(y.min()), "max_acc": float(y.max()), "recipe_sd": spread,
                       "pooled_seed_sd": pooled_sd, "spread_over_seed_sd": spread / pooled_sd,
                       "threshold": 2 * pooled_sd, "pairs_kept": int(keep.sum()), "pairs_dropped": int((~keep).sum()),
                       "agreement": {p: float(corr[p][keep].mean()) if keep.any() else None for p in PRED},
                       "flag_spread_within_seed_noise": bool(spread < 2 * pooled_sd),
                       "flag_near_chance": bool(y.max() - CHANCE[t] < 0.05),
                       "acc_by_recipe": {r: {"mean": float(mean[r]), "seed_sd": float(sd[r])} for r in recipes}}

    def agree(w_pair, tasks):
        num = {p: 0.0 for p in PRED}; den = 0.0
        for t in tasks:
            keep, corr = arrays[t]
            w = w_pair * keep
            den += w.sum()
            for p in PRED:
                num[p] += (w * corr[p]).sum()
        return {p: num[p] / den for p in PRED}, den

    ones = np.ones(len(pairs))
    pooled, _ = agree(ones, TASKS)
    rng = np.random.default_rng(SEED)
    boot = {p: [] for p in PRED}
    boot_task = {t: {p: [] for p in PRED} for t in TASKS}
    for _ in range(B):
        c = np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)).astype(float)
        w = c[fidx[pi]] * c[fidx[pj]]
        a, den = agree(w, TASKS)
        for p in PRED:
            boot[p].append(a[p])
        for t in TASKS:
            at, dt = agree(w, [t])
            for p in PRED:
                boot_task[t][p].append(at[p] if dt > 0 else np.nan)
    boot = {p: np.array(v) for p, v in boot.items()}

    def ci(x):
        x = x[~np.isnan(x)]
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    rivals = ["rival_token_unigram_ce", "rival_word_unigram_ce", "secondary_M_mix11"]
    diffs = {f"ours_minus_{r}": {"point": pooled["ours_M_task"] - pooled[r], "ci95": ci(boot["ours_M_task"] - boot[r])}
             for r in rivals}
    stronger = max(rivals[:2], key=lambda r: pooled[r])
    for t in TASKS:
        bt = {p: np.array(v) for p, v in boot_task[t].items()}
        per_task[t]["diff_ci95"] = {f"ours_minus_{r}": ci(bt["ours_M_task"] - bt[r]) for r in rivals[:2]}
    flagged = [t for t in TASKS if per_task[t]["flag_spread_within_seed_noise"] or per_task[t]["flag_near_chance"]]
    diag_unflagged, _ = agree(ones, [t for t in TASKS if t not in flagged])
    tot_kept = sum(per_task[t]["pairs_kept"] for t in TASKS)
    supported = diffs[f"ours_minus_{stronger}"]["ci95"][0] > 0
    res = {"plan": "docs/plan/benchmark_mismatch_test.md",
           "frozen_inputs_sha256": FROZEN,
           "outcome_source": {"url": f"https://huggingface.co/datasets/{REPO}/blob/{REV}/{FILE}", "repo": REPO,
                              "revision": REV, "file": FILE, "sha256": src_sha,
                              "selection": "params == '1B', final step (69369) per recipe/seed/task, metrics.primary_metric, "
                                           "3 seeds (default, large aux 2, large aux 3) averaged; mmlu = DataDecide's macro "
                                           "average over the 57 subjects"},
           "rule": "lower statistic predicts higher accuracy; pairs with |acc diff| < 2 x pooled seed SD dropped "
                   "(pooled SD = sqrt(mean over recipes of the 3-seed variance), per task)",
           "bootstrap": {"unit": "recipe family (6, FAMILIES of scripts/datadecide_predict.py)", "resamples": B,
                         "seed": SEED, "pair_weight": "count(family_i) * count(family_j)"},
           "pairs_total": 300 * len(TASKS), "pairs_kept": tot_kept, "pairs_dropped": 300 * len(TASKS) - tot_kept,
           "pooled_agreement": pooled, "pooled_ci95": {p: ci(boot[p]) for p in PRED},
           "differences": diffs, "stronger_rival": stronger, "theory_reading_supported": bool(supported),
           "flagged_tasks": flagged,
           "diagnostic_only_pooled_agreement_unflagged_tasks": diag_unflagged,
           "per_task": per_task}
    json.dump(res, open(os.path.join(OUT, "score.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["pairs_kept", "pairs_dropped", "pooled_agreement", "pooled_ci95",
                                           "differences", "stronger_rival", "theory_reading_supported", "flagged_tasks",
                                           "diagnostic_only_pooled_agreement_unflagged_tasks"]}, indent=1))
    for t, v in per_task.items():
        print(t, f"mean={v['mean_acc']:.3f} chance={v['chance']:.3f} range=[{v['min_acc']:.3f},{v['max_acc']:.3f}] "
                 f"spread/seedSD={v['spread_over_seed_sd']:.2f} kept={v['pairs_kept']}",
              {p[:12]: round(a, 3) for p, a in v['agreement'].items()}, v["diff_ci95"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "datadecide", "eval_results_cache"))
