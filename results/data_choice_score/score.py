"""Score the frozen data-choice forecast against published head-to-head corpus comparisons.

    python results/data_choice_score/score.py   -> results/data_choice_score/score.json (+ printed tables)

Reads (never writes) results/data_choice/*.json; checks their sha256 against forecast_files_sha256_*.txt first.
Published values are typed in below with their source (URL, table/figure). Values marked "fig" were read off the
vector figures in the arXiv source tarballs by extract_paloma_fig.py / extract_linear_fig.py (outputs *.tsv here).
Order flips under sampling noise come from m_noise_draws.json (per-draw rerun of scripts/data_choice_m_noise.py,
m_noise_pairs.py); without it the script falls back to what m_noise.json rank counts can prove.
"""
import hashlib, itertools, json, math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DC, OUT = ROOT / "results" / "data_choice", ROOT / "results" / "data_choice_score"

# ---- 0. input checksums -------------------------------------------------------------------------------------------
for line in [l for t in sorted(DC.glob("forecast_files_sha256_*.txt")) for l in open(t)]:
    h, f = line.split()
    if Path(f).name in ("forecast_primary_mismatch.json", "forecast_sensitivity_no_pile_eval.json", "m_noise.json", "forecast_gpt2_ref.json", "forecast_smollm2_ref.json"):
        got = hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        assert got == h, f"SHA-256 mismatch {f}"
SHA_OK = True

prim = json.load(open(DC / "forecast_primary_mismatch.json"))
nop = json.load(open(DC / "forecast_sensitivity_no_pile_eval.json"))
riv = prim["rivals"]["unigram_ce"]
gpt2 = json.load(open(DC / "forecast_gpt2_ref.json"))["forecasts"]  # SECONDARY reference-LM row (Pile extrapolated)
smol = json.load(open(DC / "forecast_smollm2_ref.json"))["forecasts"]  # SECONDARY, contaminated (trained on FineWeb-Edu, DCLM)
ROWS = {  # lower = predicted better (bits/byte or unigram CE)
    "ours_all11 (PRIMARY)": prim["forecasts"]["mismatch|all11"]["predicted_bpb"],
    "ours_ood4": prim["forecasts"]["mismatch|ood4"]["predicted_bpb"],
    "ours_no_pile_10sets": nop["rows"]["mismatch|10sets"]["predicted_bpb"],
    "ours_ood3": nop["rows"]["mismatch|ood3"]["predicted_bpb"],
    "rival_unigram_all11": riv["datadecide_eval|all11"]["ce_bits_per_byte"],
    "rival_unigram_ood4": riv["datadecide_eval|ood4"]["ce_bits_per_byte"],
    "rival_unigram_benchpool": riv["benchmark_pool_mean"]["ce_bits_per_byte"],
    "secondary_gpt2ref_all11": gpt2["ref_gpt2_only|all11"]["predicted_bpb"],
    "secondary_gpt2ref_ood4": gpt2["ref_gpt2_only|ood4"]["predicted_bpb"],
    "secondary_smollm2ref_all11": smol["ref_SmolLM2-135M_only|all11"]["predicted_bpb"],
    "secondary_smollm2ref_ood4": smol["ref_SmolLM2-135M_only|ood4"]["predicted_bpb"],
}
NOISE_TAG = {"ours_all11 (PRIMARY)": "all11", "ours_ood4": "ood4", "ours_no_pile_10sets": "10sets", "ours_ood3": "ood3"}
RECIPES = {"dolma", "dclm_baseline", "refinedweb", "fineweb_edu", "c4"}  # DataDecide recipes: near in-sample
NEW = {"pile", "slimpajama", "redpajama_1t", "fineweb"}
PILE_EVAL_ROWS = {"ours_all11 (PRIMARY)", "ours_ood4", "rival_unigram_all11", "rival_unigram_ood4", "secondary_gpt2ref_all11", "secondary_gpt2ref_ood4", "secondary_smollm2ref_all11", "secondary_smollm2ref_ood4"}


def geo(*v):
    return float(np.exp(np.mean(np.log(v))))


# ---- 1. published comparisons ------------------------------------------------------------------------------
# value: lower_better=True for loss/perplexity, False for accuracy. proxy: corpus stands in for ours only approximately.
FWP = "https://arxiv.org/abs/2406.17557"
S = []
# (a) loss / perplexity
S.append(dict(id="A1_paloma_fig1", estimand="a_loss_broad", lower_better=True,
    src="Paloma, Magnusson et al. 2024, arXiv 2312.10523 (https://arxiv.org/abs/2312.10523), Fig. 1 (fig: figures/models_by_ppl_over_all_tasks.pdf), "
        "ordinary perplexity over all Paloma sources minus 3 fringe sources and code; 1B params, ~150B tokens, decontaminated, matched order",
    vals={"pile": 14.694, "dolma": 14.698, "redpajama_1t": 15.896, "refinedweb": 16.935, "c4": 20.644},
    proxy={"dolma": "Paloma baseline is Dolma v1.5-era sample, not v1.7", "pile": "full Pile, not the uncopyrighted subset"},
    tie_rel=0.005, note="Paloma eval has no Pile source; it does contain RedPajama, Falcon-RefinedWeb, Dolma and C4 sources (each favours its own baseline)."))
pal = {  # Paloma Fig. 2 macro-average perplexity at the last checkpoint (paloma_fig2_macro_by_source.tsv)
    "c4_panel": {"dolma": 15.113, "refinedweb": 15.626, "redpajama_1t": 16.096, "pile": 18.612, "c4": 13.923},
    "wikitext103": {"dolma": 14.086, "refinedweb": 14.747, "redpajama_1t": 13.108, "pile": 12.486, "c4": 24.485},
    "m2d2_s2orc": {"dolma": 14.621, "refinedweb": 17.404, "redpajama_1t": 14.967, "pile": 12.378, "c4": 20.392},
    "dolma_panel": {"dolma": 12.769, "refinedweb": 17.569, "redpajama_1t": 14.604, "pile": 13.554, "c4": 28.151},
}
S.append(dict(id="A2_paloma_fig2_ddmatched", estimand="a_loss_matched", lower_better=True,
    src="Paloma arXiv 2312.10523 Fig. 2 (fig: figures/models_by_macro_subdomains.pdf); geometric mean of macro perplexity over the 4 Paloma sources that "
        "overlap the DataDecide 11-set mix (C4, WikiText-103, M2D2 S2ORC, Dolma); 1B, ~150B tokens",
    vals={c: geo(*[pal[p][c] for p in pal]) for c in pal["c4_panel"]},
    proxy={"dolma": "Dolma v1.5-era", "pile": "full Pile"}, tie_rel=0.005,
    note="DataDecide sets missing from Fig. 2: pile, ice, dolma_books (Dolma panel is a macro over its subsets)."))
S.append(dict(id="A3_redpajama_t5_paloma", estimand="a_loss_broad", lower_better=True,
    src="RedPajama, Weber et al. NeurIPS 2024 D&B, arXiv 2411.12372 (https://arxiv.org/html/2411.12372v1), Table 5, Paloma val perplexity; 468M params, 100B tokens",
    vals={"c4": 39.5, "dolma": 38.3, "fineweb": 33.6, "refinedweb": 32.8, "redpajama_1t": 31.5},
    proxy={"dolma": "Dolma-v1.7 CC subset only", "redpajama_1t": "RPv1-CC subset only (CommonCrawl part of RedPajama-1T)"}, tie_rel=0.005))
S.append(dict(id="A3b_redpajama_t5_pileval", estimand="a_loss_narrow", lower_better=True,
    src="arXiv 2411.12372 Table 5, Pile validation perplexity; 468M, 100B tokens",
    vals={"c4": 29.5, "dolma": 21.4, "fineweb": 26.8, "refinedweb": 19.1, "redpajama_1t": 18.7},
    proxy={"dolma": "Dolma-v1.7 CC only", "redpajama_1t": "RPv1-CC only"}, tie_rel=0.005))
fw_t3 = {"fineweb": [14.499, 12.226, 23.814, 7.65, 12.0], "fineweb_edu": [18.336, 10.242, 29.864, 7.014, 12.243]}  # Dolma V1.5 cc, pes2o, reddit, stack, wiki
fw_f12 = {"fineweb": {"c4": 13.555, "wt103": 12.994, "s2orc": 16.275}, "fineweb_edu": {"c4": 16.736, "wt103": 11.411, "s2orc": 13.551}}
S.append(dict(id="A4_fineweb_paloma_ddmatched", estimand="a_loss_matched", lower_better=True,
    src="FineWeb, Penedo et al. NeurIPS 2024 D&B, arXiv 2406.17557 (" + FWP + "), App. F.4 Table 3 (Dolma V1.5 cc/pes2o/reddit/stack/wiki) + Fig. 12 "
        "(fig: assets/plots/paloma.pdf: C4, WikiText-103, M2D2 S2ORC); geometric mean over these 8 DataDecide-overlapping sets; 1.71B ablation models, 350B tokens, no decontamination",
    vals={c: geo(*fw_t3[c], *fw_f12[c].values()) for c in fw_t3}, proxy={}, tie_rel=0.005))
fw_all = {  # FineWeb Fig. 12, all 16 Paloma sources, last checkpoint (fineweb paloma.pdf extraction), minus fringe (Manosphere, Gab, 4chan)
    "fineweb": [13.555, 11.247, 13.352, 156.970, 12.560, 13.726, 11.704, 12.994, 12.736, 23.667, 16.308, 16.275, 6.496],
    "fineweb_edu": [16.736, 11.069, 13.036, 316.205, 15.251, 17.594, 14.565, 11.411, 13.133, 29.131, 18.053, 13.551, 5.917]}
S.append(dict(id="A4b_fineweb_paloma_broad", estimand="a_loss_broad", lower_better=True,
    src="arXiv 2406.17557 Fig. 12 (fig), geometric mean of macro perplexity over 13 Paloma sources (fringe excluded); 1.71B, 350B tokens",
    vals={c: geo(*fw_all[c]) for c in fw_all}, proxy={}, tie_rel=0.005))
S.append(dict(id="A5_chungkim_lambada", estimand="a_loss_narrow", lower_better=True,
    src="Chung & Kim 2026, arXiv 2602.10657 (https://arxiv.org/html/2602.10657v1), Table 7, LAMBADA perplexity; 3.36B, 60B tokens",
    vals={"fineweb_edu": 11.72, "dclm_baseline": 5.90, "c4": 9.17}, proxy={}, tie_rel=0.005))
# (b) downstream accuracy (higher better) -- different estimand
S.append(dict(id="B1_fineweb_fig10_11", estimand="b_downstream", lower_better=False,
    src="arXiv 2406.17557 Fig. 10 / Fig. 11 (fig: assets/plots/dataset_ablations*.pdf), aggregate score of 9 commonsense/knowledge tasks at 350B tokens; 1.71B ablation models",
    vals={"fineweb_edu": 0.5084, "fineweb": 0.4899, "c4": 0.4866, "refinedweb": 0.4844, "dolma": 0.4810, "slimpajama": 0.4738, "pile": 0.4637},
    proxy={"dolma": "Common Crawl part of Dolma (paper compares the CC part of 1.6 and 1.7; legend says 'Dolma')"}, tie_rel=0.002,
    note="RedPajama2 (not RedPajama-1T) also shown, excluded."))
S.append(dict(id="B2_dclm_table2", estimand="b_downstream", lower_better=False,
    src="DCLM, Li et al. NeurIPS 2024 D&B, arXiv 2406.11794 (https://arxiv.org/html/2406.11794), Table 2, Core (22 tasks, centered acc); 7B-1x (6.9B, 138B tokens)",
    vals={"c4": 34.2, "dolma": 35.0, "redpajama_1t": 35.3, "refinedweb": 36.9}, proxy={"dolma": "Dolma-V1 (not 1.7)"}, tie_rel=0.0))
S.append(dict(id="B3_dclm_table12", estimand="b_downstream", lower_better=False,
    src="arXiv 2406.11794 Table 12 (App. H), Core at default hyperparameters; 1B-1x (1.4B, 28.8B tokens)",
    vals={"c4": 23.7, "redpajama_1t": 25.7, "dclm_baseline": 30.2}, proxy={}, tie_rel=0.0))
S.append(dict(id="B4_dclm_table8", estimand="b_downstream", lower_better=False,
    src="arXiv 2406.11794 Table 8, Core; 7B at 0.14T and 0.28T tokens (DCLM-baseline 44.1/48.9 vs FineWeb-Edu 38.7/41.9)",
    vals={"dclm_baseline": 48.9, "fineweb_edu": 41.9}, proxy={}, tie_rel=0.0))
S.append(dict(id="B5_refinedweb_table4", estimand="b_downstream", lower_better=False,
    src="RefinedWeb, Penedo et al. 2023, arXiv 2306.01116 (https://arxiv.org/html/2306.01116), Table 4, small-agg zero-shot accuracy; 1B@27GT (3B@60GT same order: 59.8/59.6/57.9)",
    vals={"refinedweb": 56.2, "c4": 55.7, "pile": 53.4}, proxy={}, tie_rel=0.0))
S.append(dict(id="B6_slimpajama_dc_table5", estimand="b_downstream", lower_better=False,
    src="SlimPajama-DC, Shen et al. 2023, arXiv 2309.10818 (https://arxiv.org/html/2309.10818v3), Table 5, HF-leaderboard average; 1.3B, 330B tokens",
    vals={"slimpajama": 40.0, "redpajama_1t": 38.0}, proxy={"slimpajama": "DC-6 (all SlimPajama sources, paper's own reweighting)"}, tie_rel=0.0))
S.append(dict(id="B7_redpajama_t5_downstream", estimand="b_downstream", lower_better=False,
    src="arXiv 2411.12372 Table 5, benchmark average; 468M, 100B tokens",
    vals={"c4": 35.8, "dolma": 36.0, "fineweb": 36.5, "refinedweb": 37.9, "redpajama_1t": 35.6},
    proxy={"dolma": "Dolma-v1.7 CC only", "redpajama_1t": "RPv1-CC only"}, tie_rel=0.0))
ck7 = {"fineweb_edu": [73.65, 38.82, 46.63, 25.23, 90.80, 33.00, 75.14, 42.02, 49.37],
       "dclm_baseline": [70.20, 34.81, 48.48, 24.58, 89.70, 28.20, 75.90, 42.73, 52.82],
       "c4": [60.77, 26.79, 49.02, 23.77, 83.90, 26.40, 76.28, 42.27, 50.91]}
S.append(dict(id="B8_chungkim_table7", estimand="b_downstream", lower_better=False,
    src="arXiv 2602.10657 Table 7, mean accuracy over the 9 accuracy tasks (our average; paper reports per task); 3.36B, 60B tokens",
    vals={c: float(np.mean(v)) for c, v in ck7.items()}, proxy={}, tie_rel=0.0))

# DataDecide (in-sample fit check only)
import csv
dd = {}
names = {"C4": "c4", "Dolma1.7": "dolma", "DCLM-Baseline": "dclm_baseline", "FineWeb-Edu": "fineweb_edu", "Falcon": "refinedweb"}
for r in csv.DictReader(open(ROOT / "results" / "datadecide" / "final_levels.csv")):
    if r["params"] == "1B" and r["data"] in names:
        dd.setdefault(names[r["data"]], []).append(float(r["bpb"]))
S.append(dict(id="Z_datadecide_insample", estimand="in_sample_fit", lower_better=True,
    src="DataDecide, Magnusson et al. ICML 2025, arXiv 2504.11393; 1B, 100B tokens, all-11 bpb from results/datadecide/final_levels.csv (the forecast was FITTED on these)",
    vals={c: float(np.mean(v)) for c, v in dd.items() if len(v) == 11}, proxy={}, tie_rel=0.0))

# ---- 2. noise: pairwise flips --------------------------------------------------------------------------------
draws = None
dp = OUT / "m_noise_draws.json"
if dp.exists():
    draws = json.load(open(dp))["draws"]


RC = json.load(open(DC / "m_noise.json"))["rows"]


def flips(tag, a, b, pred):
    """Order flips of the pair over the 20 m_noise draws. Exact count from per-draw values if available; otherwise
    sound bounds (lo, hi) from the rank counts alone: with a predicted better, a flip needs rank_b <= k < rank_a, so
    flips >= max_k [N_b(rank<=k) - N_a(rank<=k)] and flips <= min(#draws rank_a > min rank_b, #draws rank_b < max rank_a)."""
    if draws is not None and tag in draws:
        da, db = np.array(draws[tag][a]), np.array(draws[tag][b])
        f = int(np.sum(np.sign(da - db) != np.sign(pred[a] - pred[b])))
        return (f, f)
    if tag not in RC:
        return None
    if pred[a] > pred[b]:
        a, b = b, a  # a = predicted better
    ca, cb = np.array(RC[tag]["rank_counts"][a]), np.array(RC[tag]["rank_counts"][b])
    R = sorted(set(np.flatnonzero(ca).tolist()) | set(np.flatnonzero(cb).tolist()))
    others = [o for o in RC[tag]["rank_counts"] if o not in (a, b)]
    if len(R) == 2 and all(RC[tag]["rank_counts"][o][r] == 0 for o in others for r in R):
        return (int(ca[R[1]]), int(ca[R[1]]))  # a and b share two ranks no one else takes: exact
    lo = int(max(0, max(np.cumsum(cb) - np.cumsum(ca))))
    ranks = np.arange(1, len(ca) + 1)
    hi = int(min(ca[ranks > ranks[cb > 0].min()].sum(), cb[ranks < ranks[ca > 0].max()].sum()))
    return (lo, hi)


# ---- 3. scoring ----------------------------------------------------------------------------------------------
def score(src, rowname):
    pred = ROWS[rowname]
    v = src["vals"]
    cs = [c for c in v if c in pred]
    out = dict(pairs=[], n=0, agree=0, ties=0)
    for a, b in itertools.combinations(sorted(cs), 2):
        obs = v[a] - v[b] if src["lower_better"] else v[b] - v[a]  # >0 means a observed worse
        if abs(v[a] - v[b]) <= src["tie_rel"] * max(abs(v[a]), abs(v[b])):
            out["ties"] += 1
            out["pairs"].append(dict(pair=f"{a}~{b}", result="tie"))
            continue
        pr = pred[a] - pred[b]
        ok = bool(np.sign(obs) == np.sign(pr))
        tag = NOISE_TAG.get(rowname)
        fl = flips(tag, a, b, pred) if tag else None
        cls = "new_only" if {a, b} <= NEW else ("both_recipes" if {a, b} <= RECIPES else "mixed")
        out["pairs"].append(dict(pair=f"{a}-{b}", agree=ok, flips_of_20=fl, unresolved=(fl is not None and fl[0] >= 5),
                                 undetermined=(fl is not None and fl[0] < 5 <= fl[1]),
                                 cls=cls, pile_eval_advantage=("pile" in (a, b) and rowname in PILE_EVAL_ROWS),
                                 proxy=[c for c in (a, b) if c in src["proxy"]]))
        out["n"] += 1
        out["agree"] += ok
    if len(cs) >= 4:
        sign = 1 if src["lower_better"] else -1
        out["spearman"] = float(spearmanr([pred[c] for c in cs], [sign * v[c] for c in cs]).correlation)
    return out


res = {}
for src in S:
    res[src["id"]] = dict(src=src["src"], estimand=src["estimand"], vals=src["vals"], proxy=src["proxy"], note=src.get("note", ""),
                          rows={r: score(src, r) for r in ROWS})


def subset(rs, pred):
    n = sum(1 for p in rs["pairs"] if "agree" in p and pred(p))
    k = sum(1 for p in rs["pairs"] if "agree" in p and pred(p) and p["agree"])
    return k, n


summary = {}
for sid, r in res.items():
    for row, rs in r["rows"].items():
        summary.setdefault(sid, {})[row] = dict(
            all=f"{rs['agree']}/{rs['n']}", spearman=None if "spearman" not in rs else round(rs["spearman"], 3), ties=rs["ties"],
            resolved=("%d/%d" % subset(rs, lambda p: p["flips_of_20"] is not None and p["flips_of_20"][1] < 5)) if row in NOISE_TAG else "-",
            unresolved=[p["pair"] + ("" if p["agree"] else "(miss)") + f" flips{p['flips_of_20']}" for p in rs["pairs"] if p.get("unresolved") or p.get("undetermined")],
            new_only="%d/%d" % subset(rs, lambda p: p["cls"] == "new_only"),
            no_recipe_pair="%d/%d" % subset(rs, lambda p: p["cls"] != "both_recipes"),
            exact_corpora="%d/%d" % subset(rs, lambda p: not p["proxy"]))
# pooled by estimand, primary + rival, pairs counted per source
pooled = {}
for est in ("a_loss_broad", "a_loss_matched", "a_loss_narrow", "b_downstream"):
    for row in ROWS:
        k = n = 0
        for sid, r in res.items():
            if r["estimand"] == est:
                k += r["rows"][row]["agree"]; n += r["rows"][row]["n"]
        pooled.setdefault(est, {})[row] = f"{k}/{n}"
json.dump(dict(inputs_sha256_ok=SHA_OK, noise_draws=draws is not None, sources=res, summary=summary, pooled=pooled),
          open(OUT / "score.json", "w"), indent=1)
for sid in res:
    print(f"\n== {sid} [{res[sid]['estimand']}]  vals=" + ", ".join(f"{c}={v:.4g}" for c, v in sorted(res[sid]['vals'].items(), key=lambda x: x[1])))
    for row, s in summary[sid].items():
        print(f"   {row:26s} pairs {s['all']:>6s} rho={s['spearman']}  ties={s['ties']} resolved={s['resolved']} new-only={s['new_only']} "
              f"not-both-recipes={s['no_recipe_pair']} exact-corpora={s['exact_corpora']}" + (f" UNRES/UNDET={s['unresolved']}" if s.get('unresolved') else ""))
    for p in res[sid]["rows"]["ours_all11 (PRIMARY)"]["pairs"]:
        if "agree" in p and not p["agree"]:
            print("      primary:", p)
print("\nPOOLED", json.dumps(pooled, indent=1))
