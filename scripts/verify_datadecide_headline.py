"""Independent recomputation of the DataDecide held-out-family headline (registry DD-level-heldout-family,
DD-rate-heldout-family).

    python scripts/verify_datadecide_headline.py      # -> results/datadecide/verify_headline.json  (~1 min, < 300 MB)

Written without importing datadecide_predict.py, datadecide_stats.py or tok_stats.py. It starts from the raw per-source
statistics (results/datadecide/stats/{rc,ev}_*_cpu.json, *_ref_<model>.json, xent.json) and the loss table
(results/datadecide/final_levels.csv). merged.json is read only at the end, to cross-check the gamma values recomputed here.

Protocol:
  unit       25 recipes in 6 corpus families; each family is held out in turn (6 folds)
  targets    level_1B: bits/byte at 1B averaged with equal weight over the 11 eval sets (all11) or the 4 sets no recipe
             trains on (ood4); rate: -slope of ln L on ln D (D = tokens) of that average over 4M-1B (14 sizes) or 60M-1B (7)
  predictor  ordinary least squares, intercept + 1 or 2 features, fitted on the 5 training families only
  selection  when an arm has several candidate feature sets, each outer fold picks the set with the lowest
             leave-one-family-out MAE inside its own 5 training families, then refits it on all 5
  arms       const (training-family mean); compression (4 variants); n-gram (H1-H3 per byte); compression x n-gram
             pairs; mismatch (one fixed feature, no selection); select_noref (all <= 2 sets of non-reference statistics);
             select_ref (the same with reference-LM features); select_ref_minus_smollm2 (SmolLM2 features removed)
  scores     MAE, pairwise decision accuracy over the 300 recipe pairs (tied predictions count one half), MAE ratio to the
             strongest baseline (lowest held-out MAE of the three baseline arms, chosen on the held-out score)
  intervals  resample the 6 families with replacement (NBOOT draws), applied to the fixed held-out predictions, ratios
             paired within each draw; 80% and 95% percentile intervals
"""
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST = os.path.join(ROOT, "results/datadecide/stats")
NBOOT = 20000
SEED = 20261016
SMALL = ["4M", "6M", "8M", "10M", "14M", "16M", "20M"]
LARGE = ["60M", "90M", "150M", "300M", "530M", "750M", "1B"]
OOD = ["pile", "wikitext_103", "m2d2_s2orc", "ice"]
MODELS = ["SmolLM2-135M", "pythia-160m", "gpt2"]


# ------------------------------------------------------------------------------------------------ features
def gamma_of(rows, bpt, lo=4, hi=64):
    """Decay exponent of the reference curve above its 512-token value, tokens lo..hi (12 geometric points), log-log
    interpolation of the measured rows. None when the window is beyond the curve or < 3 points lie above the floor."""
    cb = np.array([r["context_bytes_exact_mean"] for r in rows])
    h = np.array([r["bits_per_byte"] for r in rows])
    if hi * bpt > cb[-1]:
        return None
    n = np.unique(np.round(np.geomspace(lo, hi, 12))).astype(int)
    hn = np.exp(np.interp(np.log(n * bpt), np.log(cb), np.log(h)))
    d = hn - h[-1]
    m = d > 0
    if m.sum() < 3:
        return None
    return float(-np.polyfit(np.log(n[m]), np.log(d[m]), 1)[0])


def load_features():
    cpu = {os.path.basename(f)[:-9]: json.load(open(f)) for f in glob.glob(os.path.join(ST, "*_cpu.json"))}
    X = json.load(open(os.path.join(ST, "xent.json")))["xent"]
    evals = {k: v for k, v in cpu.items() if k.startswith("ev_")}
    rows, ref_cols, ngram_check = {}, {}, {}
    for slug, s in cpu.items():
        if not slug.startswith("rc_"):
            continue
        b = s["bytes_per_token"]
        f = {"comp_lzma_8MB": s["compression"]["lzma_bpb_8MB"], "comp_lzma_1MB": s["compression"]["lzma_bpb_1MB"],
             "comp_bz2_1MB": s["compression"]["bz2_bpb_1MB"], "comp_zlib_1MB": s["compression"]["zlib_bpb_1MB"]}
        for j in range(3):
            f[f"ngram_H{j + 1}"] = s["H_cond"][j] / b
        ngram_check[s["name"]] = dict(n_tokens=s["ngram_n"], H_cond=s["H_cond"], finite=bool(np.all(np.isfinite(s["H_cond"]))))
        f.update(bytes_per_token=b, unigram_bits=s["unigram_bits"], zipf=s["zipf_s"], beta_tok=s["beta"]["beta"]["tok_1_200"],
                 beta_byte=s["beta"]["beta"]["byte_10_1000"], log_doc_len=float(np.log(s["doc_len_tokens"]["mean"])),
                 doc_ctx_512=s["doc_context"]["512"])
        mm = {e["name"]: (X[slug][ek] - X[ek][ek]) / e["bytes_per_token"] for ek, e in evals.items()}
        f["mismatch_all"] = float(np.mean(list(mm.values())))
        f["mismatch_ood4"] = float(np.mean([mm[n] for n in OOD]))
        for m in MODELS:
            p = os.path.join(ST, f"{slug}_ref_{m}.json")
            if not os.path.exists(p):
                continue
            r = json.load(open(p))["paired"]["rows"]
            f[f"ref_h512_{m}"] = r[-1]["bits_per_byte"]
            g = gamma_of(r, b)
            if g:
                f[f"ref_gamma_{m}"] = g
                f[f"ref_cagnetta_{m}"] = g / (2 * f["beta_tok"])
        rows[s["name"]] = f
    F = pd.DataFrame(rows).T.sort_index()
    F = F[[c for c in F.columns if F[c].notna().all()]]
    return F, ngram_check


# ------------------------------------------------------------------------------------------------ targets
def targets():
    lv = pd.read_csv(os.path.join(ROOT, "results/datadecide/final_levels.csv"))
    fam = lv.groupby("data")["family"].first()
    out = {}
    for tag, evs in (("all11", None), ("ood4", OOD)):
        sub = lv if evs is None else lv[lv["eval"].isin(evs)]
        cnt = sub.groupby(["data", "params"])["eval"].nunique()
        assert cnt.nunique() == 1, "every (recipe, size) cell needs every eval set"
        L = sub.groupby(["data", "params"])["bpb"].mean()
        D = sub.groupby(["data", "params"])["tokens"].mean()
        T = {}
        for r in fam.index:
            def rate(sizes):
                x, y = np.log(D.loc[r].loc[sizes].values), np.log(L.loc[r].loc[sizes].values)
                return -float(np.polyfit(x, y, 1)[0])
            T[r] = dict(level_1B=float(L.loc[(r, "1B")]), rate_4M_1B=rate(SMALL + LARGE), rate_60M_1B=rate(LARGE))
        out[tag] = pd.DataFrame(T).T.sort_index()
    return out, fam.sort_index()


# ------------------------------------------------------------------------------------------------ held-out prediction
def ols_predict(A, y, tr, te):
    M = np.column_stack([np.ones(len(tr))] + [A[tr, j] for j in range(A.shape[1])])
    w, *_ = np.linalg.lstsq(M, y[tr], rcond=None)
    return np.column_stack([np.ones(len(te))] + [A[te, j] for j in range(A.shape[1])]) @ w


def heldout(F, y, groups, cand, units):
    """Leave-one-group-out over the given units. cand: list of tuples of column names. Returns predictions on the units and
    the set chosen for each held-out group (nested choice by inner leave-one-group-out MAE when len(cand) > 1)."""
    pred, chosen = {}, {}
    for g in sorted(set(groups[units])):
        te = units[groups[units] == g]
        tr = units[groups[units] != g]
        if len(cand) == 1:
            best = cand[0]
        else:
            inner = []
            for c in cand:
                p, _ = heldout(F, y, groups, [c], tr)
                inner.append(np.mean(np.abs(np.array([p[i] for i in tr]) - y[tr])))
            best = cand[int(np.argmin(inner))]
        A = F[list(best)].values if best else np.zeros((len(y), 0))
        for i, v in zip(te, ols_predict(A, y, tr, te)):
            pred[i] = v
        chosen[g] = list(best)
    return pred, chosen


def pairwise_acc(p, y):
    i, j = np.triu_indices(len(p), 1)
    sp, sy = np.sign(p[i] - p[j]), np.sign(y[i] - y[j])
    return float(np.mean(np.where(sp == 0, 0.5, (sp == sy).astype(float))))


def run():
    F, ngram_check = load_features()
    T, fam = targets()
    names = list(F.index)
    assert names == list(fam.index) and len(names) == 25, "recipe sets of stats and loss table differ"
    groups = fam.loc[names].values
    units = np.arange(len(names))
    cols = list(F.columns)
    ref = [c for c in cols if c.startswith("ref_")]
    nonref = [c for c in cols if c not in ref]
    comp = ["comp_lzma_8MB", "comp_lzma_1MB", "comp_bz2_1MB", "comp_zlib_1MB"]
    ngram = ["ngram_H1", "ngram_H2", "ngram_H3"]

    def upto2(pool):
        return [(a,) for a in pool] + list(itertools.combinations(pool, 2))

    rng = np.random.default_rng(SEED)
    fams = np.unique(groups)
    draws = rng.integers(0, len(fams), size=(NBOOT, len(fams)))
    boot_idx = [np.concatenate([np.flatnonzero(groups == fams[d]) for d in row]) for row in draws]
    out = dict(features=cols, ref_features=ref, ngram_baseline_inputs=ngram_check, families=dict(zip(names, groups.tolist())),
               nboot=NBOOT, seed=SEED, results={})
    jobs = [("level_1B", "all11"), ("level_1B", "ood4"), ("rate_4M_1B", "all11"), ("rate_4M_1B", "ood4"),
            ("rate_60M_1B", "all11"), ("rate_60M_1B", "ood4")]
    for tname, tag in jobs:
        y = T[tag].loc[names, tname].values
        mm = "mismatch_ood4" if tag == "ood4" else "mismatch_all"
        arms = {"const": [()], "base_compression": [(c,) for c in comp], "base_ngram": [(c,) for c in ngram],
                "base_comp+ngram": [(a, b) for a in comp for b in ngram], "ours_mismatch": [(mm,)],
                "ours_select_noref": upto2(nonref), "ours_select_ref_lm": upto2(cols),
                "ours_select_ref_minus_smollm2": upto2([c for c in cols if "SmolLM2" not in c]),
                "ours_select_ref_smollm2_only_ref": upto2(nonref + [c for c in ref if "SmolLM2" in c])}
        P, chosen = {}, {}
        for a, cand in arms.items():
            p, ch = heldout(F, y, groups, cand, units)
            P[a] = np.array([p[i] for i in units])
            chosen[a] = ch
        mae = {a: float(np.mean(np.abs(P[a] - y))) for a in P}
        best = min(("base_compression", "base_ngram", "base_comp+ngram"), key=lambda a: mae[a])
        res = {}
        for a in P:
            bm, bp, br = [], [], []
            for k in boot_idx:
                e = np.mean(np.abs(P[a][k] - y[k]))
                bm.append(e)
                bp.append(pairwise_acc(P[a][k], y[k]))
                br.append(e / np.mean(np.abs(P[best][k] - y[k])))
            q = lambda v, lo, hi: [float(np.quantile(v, lo)), float(np.quantile(v, hi))]  # noqa: E731
            res[a] = dict(mae=mae[a], pairwise=pairwise_acc(P[a], y), ratio_best_base=mae[a] / mae[best],
                          mae_80=q(bm, .1, .9), pairwise_80=q(bp, .1, .9), pairwise_95=q(bp, .025, .975),
                          ratio_80=q(br, .1, .9), ratio_95=q(br, .025, .975), chosen=chosen[a],
                          predictions=dict(zip(names, P[a].tolist())))
        out["results"][f"{tname}|{tag}"] = dict(best_baseline=best, target=dict(zip(names, y.tolist())), arms=res)
    # cross-check of the recomputed gamma against merged.json (read only here)
    mg = json.load(open(os.path.join(ST, "merged.json")))["sources"]
    diffs = []
    for slug, s in mg.items():
        if s.get("kind") == "recipe":
            for m in MODELS:
                g = ((s.get("ref", {}).get(m) or {}).get("gamma") or {}).get("4-64", {}).get("gamma")
                if g is not None and f"ref_gamma_{m}" in F.columns:
                    diffs.append(abs(g - F.loc[s["name"], f"ref_gamma_{m}"]))
    out["gamma_crosscheck_max_abs_diff_vs_merged"] = float(max(diffs)) if diffs else None
    return out


def main():
    out = run()
    json.dump(out, open(os.path.join(ROOT, "results/datadecide/verify_headline.json"), "w"), indent=1)
    print("features:", len(out["features"]), "of which reference-LM:", len(out["ref_features"]))
    print("n-gram baseline inputs all finite:", all(v["finite"] for v in out["ngram_baseline_inputs"].values()))
    print("gamma cross-check max |diff| vs merged.json:", out["gamma_crosscheck_max_abs_diff_vs_merged"])
    for t, R in out["results"].items():
        print(f"\n{t}  best baseline = {R['best_baseline']}")
        for a, s in R["arms"].items():
            print(f"  {a:32s} MAE {s['mae']:.4f}  ratio {s['ratio_best_base']:.3f} 95%[{s['ratio_95'][0]:.2f},{s['ratio_95'][1]:.2f}]"
                  f"  pairwise {s['pairwise']:.3f} 80%[{s['pairwise_80'][0]:.2f},{s['pairwise_80'][1]:.2f}]"
                  f" 95%[{s['pairwise_95'][0]:.2f},{s['pairwise_95'][1]:.2f}]")


if __name__ == "__main__":
    main()
