"""Training-proxy baseline on the pairs scored by bench_mismatch_score.py (run that first).

DataDecide's single-scale method: rank recipes by the final-checkpoint primary_metric of a small model (seeds averaged)
and read the 1B order from it. Scored on exactly the noise-filtered 1B task-pairs of score.json, with the same paired
family bootstrap (same seed). Adds the key "proxy_baseline" (proxy metric primary_metric, i.e. accuracy) or
"proxy_baseline_continuous" (correct_prob_per_char, the proxy metric DataDecide recommends for small scales, sec. 3.3;
total_prob_per_char, its co-recommended metric, reported alongside) to results/bench_mismatch_score/score.json.

    python scripts/bench_mismatch_proxy.py [cache] [primary|continuous]
"""
import itertools, json, os, sys
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bench_mismatch_score import FAMILIES, FILE, OUT, REPO, REV, TASKS, B, SEED, sha, outcome_parquet  # noqa: E402

SIZES = ["4M", "20M", "60M", "150M", "300M"]


def main(cache, mode="primary"):
    metrics = ["primary_metric"] if mode == "primary" else ["correct_prob_per_char", "total_prob_per_char"]
    S = json.load(open(os.path.join(OUT, "score.json")))
    p, src_sha = outcome_parquet(cache)
    assert src_sha == S["outcome_source"]["sha256"]
    pf, parts = pq.ParquetFile(p), []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg)
        t = t.filter(pc.is_in(t["params"], pa.array(SIZES)))
        if t.num_rows:
            parts.append(t)
    d = pa.concat_tables(parts).to_pandas()
    d = d[d.task.isin(TASKS.values())]
    d = d[d.step == d.groupby(["params", "data", "seed", "task"]).step.transform("max")].copy()
    for m in metrics:
        d[m] = d.metrics.map(lambda s: json.loads(s)[m])
    assert (d.groupby(["params", "data", "task"]).size() == 3).all()
    prox = d.groupby(["params", "task", "data"])[metrics].mean()
    tag = {"primary_metric": "", "correct_prob_per_char": "", "total_prob_per_char": "total_prob_"}

    fam_of = {r: f for f, (_, rs) in FAMILIES.items() for r in rs}
    recipes, fams = sorted(fam_of), list(FAMILIES)
    fidx = np.array([fams.index(fam_of[r]) for r in recipes])
    pairs = np.array(list(itertools.combinations(range(25), 2)))
    pi, pj = pairs[:, 0], pairs[:, 1]
    fc = json.load(open(os.path.join(ROOT, "results/bench_mismatch/forecast.json")))["values"]
    out1b = pd.read_csv(os.path.join(OUT, "outcomes_1B_final_by_seed.csv"))

    keep, corr = {}, {}
    names = ["ours_M_task", "rival_token_unigram_ce"] + [f"proxy_{tag[m]}{s}" for m in metrics for s in SIZES]
    col = {f"proxy_{tag[m]}{s}": (m, s) for m in metrics for s in SIZES}
    for t, dd in TASKS.items():
        y = out1b[out1b.task == dd].groupby("data").acc.mean().reindex(recipes).values
        dy = y[pi] - y[pj]
        keep[t] = np.abs(dy) >= S["per_task"][t]["threshold"]
        assert keep[t].sum() == S["per_task"][t]["pairs_kept"]
        c = {}
        for n in names:
            if n.startswith("proxy_"):
                m, sz = col[n]
                x = -prox.loc[(sz, dd), m].reindex(recipes).values  # higher proxy metric predicts higher 1B accuracy
            else:
                x = np.array([fc["ours_M_task" if n == "ours_M_task" else "rival_unigram_ce"][r][t] for r in recipes])
            dx = x[pi] - x[pj]
            c[n] = np.where(dx == 0, 0.5, (np.sign(dx) == -np.sign(dy)).astype(float))
        corr[t] = c

    def agree(w, tasks):
        den = sum((w * keep[t]).sum() for t in tasks)
        return {n: sum((w * keep[t] * corr[t][n]).sum() for t in tasks) / den for n in names}

    pooled = agree(np.ones(len(pairs)), TASKS)
    for n, ref in [("ours_M_task", "ours_M_task"), ("rival_token_unigram_ce", "rival_token_unigram_ce")]:
        assert abs(pooled[n] - S["pooled_agreement"][ref]) < 1e-12  # same pairs, same scoring as score.json
    rng = np.random.default_rng(SEED)
    boot = {n: [] for n in names}
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)).astype(float)
        a = agree(cnt[fidx[pi]] * cnt[fidx[pj]], TASKS)
        for n in names:
            boot[n].append(a[n])
    boot = {n: np.array(v) for n, v in boot.items()}
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
    res = {"method": f"DataDecide single-scale ranking: final-checkpoint {' / '.join(metrics)} of a small model, 3 seeds "
                     "(default, small aux 2, small aux 3) averaged; same pinned parquet; scored on the same 1,660 "
                     "noise-filtered 1B task-pairs (outcome: 1B primary_metric); higher proxy value predicts higher "
                     "1B accuracy",
           "pooled_agreement": {n: pooled[n] for n in names},
           "pooled_ci95": {n: ci(boot[n]) for n in names},
           "differences": {**{f"ours_minus_{n}": {"point": pooled["ours_M_task"] - pooled[n],
                                                   "ci95": ci(boot["ours_M_task"] - boot[n])} for n in col},
                           "token_rival_minus_proxy_150M": {
                               "point": pooled["rival_token_unigram_ce"] - pooled["proxy_150M"],
                               "ci95": ci(boot["rival_token_unigram_ce"] - boot["proxy_150M"])}},
           "per_task": {t: {n: float(corr[t][n][keep[t]].mean()) for n in names} for t in TASKS},
           "datadecide_reported": "Magnusson et al. 2025 (DataDecide), abstract: ranking models at a single small "
                                  "size (e.g. 150M) predicts the best models at 1B with ~80% of comparisons correct. "
                                  "Their pair set (all recipe pairs, their target metric and aggregation) differs "
                                  "from ours (1,660 noise-filtered task-pairs)."}
    if mode != "primary":
        res["metric_choice"] = ("Primary: correct_prob_per_char. DataDecide (arXiv 2504.11393, sec. 3.3): 'Here we chose "
                                "a single length normalization, *_per_char.' ... 'Using Correct Prob or Total Prob leads "
                                "to decision accuracy at least as good as any other metric for most small scales.' "
                                "Sec. 3.4: 'Using Correct Prob sees wider spreads or reduced noise for many tasks.' "
                                "total_prob_per_char (co-recommended) is reported alongside, not as the primary.")
        res.pop("datadecide_reported")
    S["proxy_baseline" if mode == "primary" else "proxy_baseline_continuous"] = res
    json.dump(S, open(os.path.join(OUT, "score.json"), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["pooled_agreement", "pooled_ci95", "differences"]}, indent=1))
    for t, v in res["per_task"].items():
        print(t, {k: round(x, 3) for k, x in v.items()})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "datadecide", "eval_results_cache"),
         sys.argv[2] if len(sys.argv) > 2 else "primary")
