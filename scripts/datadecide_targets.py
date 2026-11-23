"""Measured DataDecide targets: per-recipe loss levels and data-scaling rates, with seed uncertainty.

    python scripts/datadecide_targets.py [--gt data/datadecide/gt] [--out results/datadecide]

Input: AI2 DataDecide perplexity table (ppl_results.parquet; 25 recipes x 14 sizes x seeds x checkpoints, perplexity
on 11 fixed eval sets) and the tokenized eval sets (raw uint16 GPT-NeoX-OLMo streams, EOS 50279) for bytes per token.
Only the 2 MB perplexity table is read; the 700 MB eval_results files are not touched (tokens per step was read from
them once, see TOKENS_PER_STEP).

Loss is ln(perplexity), nats per token; bits per byte = loss / ln 2 / (bytes per token of that eval set). Every model
uses the same tokenizer, so for one eval set the two units differ by a constant.

Outputs (results/datadecide/):
  eval_bpt.json        bytes per token of each eval set
  final_levels.csv     final-checkpoint loss per recipe x eval x size: mean, seed SD, n complete seeds, tokens D
  ladder_rates.csv     exponent of ln L vs ln D along the final-checkpoint ladder (D = 100 N), two ranges, 80% CI
  inrun_rates.csv      exponent of ln L vs ln D inside one run, checkpoints at 10-40% of the run, per size
  resolvability.csv    across-recipe spread vs seed noise, per eval x size (levels) and per eval (rates)
  smallscale_rival.csv how well the recipe ordering at a small size predicts the 1B ordering (the standard rival)
  summary.json         headline numbers
"""
import argparse, json, os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT = "data/datadecide/gt"
EVALDATA = "data/datadecide/evaldata"
TOKENIZER = os.path.join(ROOT, "data/datadecide/tokenizer.json")  # allenai/gpt-neox-olmo-dolma-v1_5
EOS = 50279

# Tokens per optimizer step, measured from eval_results_{0..3}.parquet (tokens / step, constant within each size;
# equals global batch x 2048 from olmo-source/scripts/ladder.py).
TOKENS_PER_STEP = {"4M": 65536, "6M": 65536, "8M": 65536, "10M": 65536, "14M": 65536, "16M": 65536,
                   "20M": 131072, "60M": 196608, "90M": 327680, "150M": 393216, "300M": 655360,
                   "530M": 917504, "750M": 1179648, "1B": 1441792}
SIZES = ["4M", "6M", "8M", "10M", "14M", "16M", "20M", "60M", "90M", "150M", "300M", "530M", "750M", "1B"]
EVALS = ["wikitext_103", "pile", "m2d2_s2orc", "ice", "dolma_wiki", "dolma_stack", "dolma_reddit", "dolma_pes2o",
         "dolma_common-crawl", "dolma_books", "c4_en"]
COL = {e: f"eval/{e}-validation/Perplexity" for e in EVALS}

FAMILY = {  # Pre_Scale/src/families.py (ancestry clusters)
    "C4": "c4", "DCLM-Baseline": "dclm", "DCLM-Baseline (QC 10%)": "dclm", "DCLM-Baseline (QC 20%)": "dclm",
    "DCLM-Baseline (QC 7%, FW2)": "dclm", "DCLM-Baseline (QC 7%, FW3)": "dclm", "DCLM-Baseline (QC FW 10%)": "dclm",
    "DCLM-Baseline (QC FW 3%)": "dclm", "DCLM-Baseline 25% / Dolma 75%": "dclm_dolma_mix",
    "DCLM-Baseline 50% / Dolma 50%": "dclm_dolma_mix", "DCLM-Baseline 75% / Dolma 25%": "dclm_dolma_mix",
    "Dolma1.6++": "dolma", "Dolma1.7": "dolma", "Dolma1.7 (no Flan)": "dolma", "Dolma1.7 (no Reddit)": "dolma",
    "Dolma1.7 (no code)": "dolma", "Dolma1.7 (no math, code)": "dolma", "Falcon": "falcon", "Falcon+CC": "falcon",
    "Falcon+CC (QC 10%)": "falcon", "Falcon+CC (QC 20%)": "falcon", "Falcon+CC (QC Orig 10%)": "falcon",
    "Falcon+CC (QC Tulu 10%)": "falcon", "FineWeb-Edu": "fineweb", "FineWeb-Pro": "fineweb"}

# Which eval sets are drawn from a source the recipe trains on (component level, from the source lists in
# olmo-source/olmo/data/named_data_mixes.py). "approx" = same source family, different release/version.
_DOLMA17 = {"dolma_books": "yes", "dolma_pes2o": "yes", "dolma_wiki": "yes", "dolma_stack": "approx",
            "dolma_reddit": "yes", "dolma_common-crawl": "approx", "c4_en": "yes"}
IN_DIST = {
    "Dolma1.7": _DOLMA17,
    "Dolma1.7 (no Flan)": _DOLMA17,
    "Dolma1.7 (no Reddit)": {k: v for k, v in _DOLMA17.items() if k != "dolma_reddit"},
    "Dolma1.7 (no code)": {k: v for k, v in _DOLMA17.items() if k != "dolma_stack"},
    "Dolma1.7 (no math, code)": {k: v for k, v in _DOLMA17.items() if k not in ("dolma_stack", "dolma_pes2o")},
    "Dolma1.6++": {"dolma_books": "yes", "dolma_pes2o": "yes", "dolma_wiki": "yes", "dolma_stack": "approx",
                   "dolma_reddit": "yes", "dolma_common-crawl": "yes", "c4_en": "yes"},
    "C4": {"c4_en": "yes"},
    "Falcon+CC": {"dolma_common-crawl": "approx"},  # QC variants: classifier-filtered Dolma CC
    "Falcon+CC (QC 10%)": {"dolma_common-crawl": "approx"}, "Falcon+CC (QC 20%)": {"dolma_common-crawl": "approx"},
    "Falcon+CC (QC Orig 10%)": {"dolma_common-crawl": "approx"},
    "Falcon+CC (QC Tulu 10%)": {"dolma_common-crawl": "approx"},
    "DCLM-Baseline 25% / Dolma 75%": _DOLMA17, "DCLM-Baseline 50% / Dolma 50%": _DOLMA17,
    "DCLM-Baseline 75% / Dolma 25%": _DOLMA17,
}
LARGE = ["60M", "90M", "150M", "300M", "530M", "750M", "1B"]
SMALL = ["4M", "6M", "8M", "10M", "14M", "16M", "20M"]
NBOOT = 2000
Q = (0.1, 0.9)


def eval_bpt(evaldata, tokenizer):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer)
    out = {}
    for e in EVALS:
        a = np.fromfile(os.path.join(evaldata, f"{e}.npy"), dtype="<u2").astype(np.int64)
        cuts = np.flatnonzero(a == EOS)
        docs = np.split(a, cuts + 1)
        nb = nt = 0
        for d in docs:
            d = d[d != EOS]
            if len(d):
                nb += len(tok.decode(d.tolist()).encode("utf-8"))
                nt += len(d)
        out[e] = dict(bytes_per_token=nb / nt, n_tokens=int(nt), n_docs=int(len(docs)))
    return out


def load(gt):
    d = pd.read_parquet(os.path.join(gt, "ppl_results.parquet"))
    d = d.drop(columns=[c for c in d.columns if c.startswith("__")])
    d["tokens"] = d["step"] * d["params"].map(TOKENS_PER_STEP)
    for e in EVALS:
        d[e] = np.log(d[COL[e]])
    return d.drop(columns=list(COL.values()))


def finals(d):
    """Last checkpoint of each run; a run is complete if it reaches 97% of the longest run at its size."""
    last = d.sort_values("step").groupby(["data", "params", "seed"], as_index=False).last()
    smax = d.groupby("params")["step"].max()
    last["complete"] = last["step"] >= 0.97 * last["params"].map(smax)
    return last


def fit_slope(x, Y):
    """Least-squares slope of each row of Y (..., k) on x (k,)."""
    xc = x - x.mean()
    return ((Y - Y.mean(-1, keepdims=True)) * xc).sum(-1) / (xc ** 2).sum()


def ladder(lv, sizes, rng):
    """Exponent alpha = -dlnL/dlnD and bits/byte slope per doubling along the final ladder, parametric bootstrap
    over seed noise (cells with one complete seed borrow the median seed SD of their size across recipes x evals)."""
    rows = []
    sub = lv[lv.params.isin(sizes)]
    sd_pool = sub[sub.n >= 2].groupby(["params", "eval"])["sd"].median()
    sd_size = sub[sub.n >= 2].groupby("params")["sd"].median()
    for (r, e), g in sub.groupby(["data", "eval"]):
        g = g.set_index("params").loc[sizes]
        x = np.log(g["tokens"].values)
        m = g["mean"].values
        sd = np.array([s if n >= 2 else sd_pool.get((p, e), sd_size.get(p, np.nan))
                       for p, s, n in zip(sizes, g["sd"].values, g["n"].values)])
        sd = np.where(np.isnan(sd), np.nanmedian(sd_size.values), sd)
        se = sd / np.sqrt(np.maximum(g["n"].values, 1))
        draws = m + rng.standard_normal((NBOOT, len(sizes))) * se
        a = -fit_slope(x, np.log(draws))
        a0 = -fit_slope(x, np.log(m))
        rows.append(dict(data=r, eval=e, alpha=a0, alpha_lo=np.quantile(a, Q[0]), alpha_hi=np.quantile(a, Q[1]),
                         alpha_sd=a.std(), loss_at_1B=m[-1] if sizes[-1] == "1B" else np.nan))
    return pd.DataFrame(rows)


def inrun(d, size, lo=0.1, hi=0.4):
    """Exponent of ln L vs ln D over checkpoints in [lo, hi] of the run's step range, per seed covering hi."""
    s = d[d.params == size]
    smax = s["step"].max()
    rows = []
    for (r, seed), g in s.groupby(["data", "seed"]):
        if g["step"].max() < hi * smax:
            continue
        g = g[(g.step >= lo * smax) & (g.step <= hi * smax)]
        if len(g) < 4:
            continue
        x = np.log(g["tokens"].values)
        for e in EVALS:
            rows.append(dict(data=r, seed=seed, eval=e, alpha=-fit_slope(x, g[e].values), n_ckpt=len(g)))
    t = pd.DataFrame(rows)
    return t.groupby(["data", "eval"])["alpha"].agg(alpha="mean", sd="std", n="count").reset_index().assign(
        params=size)


def spearman(a, b):
    ra, rb = pd.Series(a).rank().values, pd.Series(b).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def pairwise_acc(a, b):
    i, j = np.triu_indices(len(a), 1)
    return float(np.mean(np.sign(a[i] - a[j]) == np.sign(b[i] - b[j])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=GT)
    ap.add_argument("--evaldata", default=EVALDATA)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/datadecide"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    bpt = eval_bpt(a.evaldata, TOKENIZER)
    json.dump(bpt, open(os.path.join(a.out, "eval_bpt.json"), "w"), indent=1)

    d = load(a.gt)
    f = finals(d)
    fc = f[f.complete]
    long = fc.melt(id_vars=["data", "params", "seed", "tokens"], value_vars=EVALS, var_name="eval", value_name="L")
    lv = long.groupby(["data", "params", "eval"]).agg(mean=("L", "mean"), sd=("L", "std"), n=("L", "count"),
                                                       tokens=("tokens", "mean")).reset_index()
    lv["bpb"] = lv["mean"] / np.log(2) / lv["eval"].map(lambda e: bpt[e]["bytes_per_token"])
    lv["family"] = lv["data"].map(FAMILY)
    lv["in_dist"] = [IN_DIST.get(r, {}).get(e, "no") for r, e in zip(lv["data"], lv["eval"])]
    lv.to_csv(os.path.join(a.out, "final_levels.csv"), index=False)

    lr = ladder(lv, LARGE, rng).rename(columns=lambda c: c if c in ("data", "eval") else c + "_large")
    sr = ladder(lv, SMALL, rng).rename(columns=lambda c: c if c in ("data", "eval") else c + "_small")
    rates = lr.merge(sr, on=["data", "eval"]).drop(columns=["loss_at_1B_small"])
    rates["family"] = rates["data"].map(FAMILY)
    rates["in_dist"] = [IN_DIST.get(r, {}).get(e, "no") for r, e in zip(rates["data"], rates["eval"])]
    rates.to_csv(os.path.join(a.out, "ladder_rates.csv"), index=False)

    ir = pd.concat([inrun(d, s) for s in ("530M", "750M", "1B")])
    ir.to_csv(os.path.join(a.out, "inrun_rates.csv"), index=False)

    # Resolvability: across-recipe SD of the target vs its seed noise.
    res = []
    for (p, e), g in lv.groupby(["params", "eval"]):
        res.append(dict(target="level", params=p, eval=e, spread=g["mean"].std(),
                        noise=g.loc[g.n >= 2, "sd"].median(), n_recipes=len(g)))
    for e, g in rates.groupby("eval"):
        res.append(dict(target="ladder_alpha_large", params="60M-1B", eval=e, spread=g["alpha_large"].std(),
                        noise=g["alpha_sd_large"].median(), n_recipes=len(g)))
    for (p, e), g in ir.groupby(["params", "eval"]):
        res.append(dict(target="inrun_alpha", params=p, eval=e, spread=g["alpha"].std(),
                        noise=(g["sd"] / np.sqrt(g["n"])).median(), n_recipes=len(g)))
    res = pd.DataFrame(res)
    res["snr"] = res["spread"] / res["noise"]
    res.to_csv(os.path.join(a.out, "resolvability.csv"), index=False)

    # Standard rival: ordering at a smaller size (final checkpoint) predicts the 1B ordering.
    riv = []
    piv = lv.pivot_table(index="data", columns=["eval", "params"], values="mean")
    for e in EVALS:
        tgt = piv[(e, "1B")].values
        for p in SIZES[:-1]:
            src = piv[(e, p)].values
            riv.append(dict(target="level_1B", eval=e, from_size=p, spearman=spearman(src, tgt),
                            pairwise_acc=pairwise_acc(src, tgt)))
        g = rates[rates["eval"] == e]
        riv.append(dict(target="alpha_large", eval=e, from_size="alpha_small(4M-20M)",
                        spearman=spearman(g["alpha_small"].values, g["alpha_large"].values),
                        pairwise_acc=pairwise_acc(g["alpha_small"].values, g["alpha_large"].values)))
    riv = pd.DataFrame(riv)
    riv.to_csv(os.path.join(a.out, "smallscale_rival.csv"), index=False)

    # Headline numbers.
    pile = rates[rates["eval"] == "pile"].sort_values("alpha_large")
    l1 = lv[(lv.params == "1B")]
    summ = dict(
        n_rows=int(len(d)), recipes=int(d["data"].nunique()), sizes=SIZES,
        complete_seeds_per_size={p: int(fc[fc.params == p].groupby("data")["seed"].nunique().median()) for p in SIZES},
        tokens_final={p: float(fc[fc.params == p]["tokens"].median()) for p in SIZES},
        eval_bytes_per_token={e: round(v["bytes_per_token"], 3) for e, v in bpt.items()},
        alpha_large_by_eval={e: dict(median=float(g["alpha_large"].median()), min=float(g["alpha_large"].min()),
                                     max=float(g["alpha_large"].max()), median_seed_sd=float(g["alpha_sd_large"].median()))
                             for e, g in rates.groupby("eval")},
        alpha_small_by_eval={e: float(g["alpha_small"].median()) for e, g in rates.groupby("eval")},
        pile_alpha_large_extremes=[pile.iloc[[0, -1]][["data", "alpha_large", "alpha_lo_large", "alpha_hi_large"]]
                                   .to_dict("records")],
        level_1B_bpb_range_by_eval={e: [float(g["bpb"].min()), float(g["bpb"].max()), float(g["sd"].median())]
                                    for e, g in l1.groupby("eval")},
        inrun_alpha_1B_median_by_eval={e: float(g["alpha"].median()) for e, g in ir[ir.params == "1B"].groupby("eval")},
        rival_level_1B_spearman_median_over_evals={p: float(riv[(riv.target == "level_1B") & (riv.from_size == p)]
                                                            ["spearman"].median()) for p in SIZES[:-1]},
        rival_alpha_spearman_by_eval={r["eval"]: r["spearman"] for _, r in riv[riv.target == "alpha_large"].iterrows()},
        snr_median={t: float(g["snr"].median()) for t, g in res.groupby("target")},
    )
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
