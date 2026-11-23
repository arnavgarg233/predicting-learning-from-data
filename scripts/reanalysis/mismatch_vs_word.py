"""Does the bigram mismatch add anything over word-unigram cross-entropy?

    python scripts/reanalysis/mismatch_vs_word.py   -> results/reanalysis/mismatch_vs_word.json

Three tests, each comparing word unigram alone, bigram mismatch alone, and both together, with the paper's own code:

A. DataDecide held-out families (Methods, "Held-out families and scoring"). Affine maps fitted by
   datadecide_predict.lofo on five families and applied to the sixth; target = 1B bits per byte averaged over the
   11 sets (and, as a check, over the 4 sets no recipe trains on). Word unigram values are the cached ones of
   datadecide_evalaware_rivals.py (same samples, same add-0.5 code). "Both" is the two-feature affine map; "nested"
   lets each fold choose {bigram}, {word} or {both} by leave-one-family-out MAE inside its training families, as the
   paper's selection does. Paired family bootstrap (2,000 resamples of the six families, applied to the fixed held-out
   predictions, as in the paper) on pairwise-accuracy and MAE differences.
B. Published rankings (Table B.1). Word unigram cross-entropy of the 11 sets under each of the nine public
   corpus samples, computed with the same function as for the recipes; maps fitted on all 25 recipes as in
   data_choice_forecast.fit_apply and applied unchanged; scored with the source list and tie rules of
   results/data_choice_score/score.py (read, not modified).
C. Benchmark-text test (Methods, "Benchmark-text test"). Per task, the frozen statistics of results/bench_mismatch
   (bigram mismatch to the task text, token and word unigram rivals). The single statistics are scored as in
   bench_mismatch_score.py (raw sign; kept pairs; family-bootstrap with pair weights). The combination is a
   leave-one-family-out affine map of accuracy on [bigram, word] per task (and the same map for each alone, so that
   the nested comparison uses the same machinery), plus a fit-free sum of within-task z-scores.
"""
import itertools
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import datadecide_predict as P  # noqa: E402
import datadecide_stats as dds  # noqa: E402

OUT = ROOT / "results/reanalysis"
OUT.mkdir(parents=True, exist_ok=True)
NB = 2000
W = re.compile(r"\w+")
q = lambda v: [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]  # noqa: E731


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


# ============================================================================ A. DataDecide held-out families
def part_a():
    C = json.load(open(ROOT / "results/datadecide/evalaware_rivals_cache.json"))
    M = P.load_stats(str(ROOT / "results/datadecide/stats/merged.json"))
    S, X = M["sources"], (M.get("xent") or {}).get("xent")
    lv = pd.read_csv(ROOT / "results/datadecide/final_levels.csv")
    fam_of = {r: f for f, (_, rs) in P.FAMILIES.items() for r in rs}
    names = sorted(fam_of)
    fam = np.array([fam_of[r] for r in names])
    F, _ = P.recipe_features(S, X, names)
    T, _ = P.family_targets(lv, np.random.default_rng(0))
    slug_of = {s_["name"]: k for k, s_ in S.items() if s_.get("kind") == "recipe"}
    ev_name = {k: s_["name"] for k, s_ in S.items() if s_.get("kind") == "eval"}  # slug -> name
    ood4 = [k for k, n in ev_name.items() if n in P.OOD4]
    assert len(ood4) == 4, (P.OOD4, ev_name)
    out = {}
    for tag, mm_col, evs in (("all11", "mismatch_all", None), ("ood4", "mismatch_ood4", ood4)):
        y = T[tag].loc[names, "level_1B"].values.astype(float)
        word = np.array([np.mean([v for e, v in C[slug_of[r]]["word"].items() if evs is None or e in evs]) for r in names])
        tok = np.array([np.mean([v for e, v in C[slug_of[r]]["tok"].items() if evs is None or e in evs]) for r in names])
        big = F.loc[names, mm_col].values.astype(float)
        Fm = np.c_[big, word, tok]
        pred = {"bigram": P.lofo(Fm, y, fam, [(0,)])[0],
                "word unigram": P.lofo(Fm, y, fam, [(1,)])[0],
                "token unigram": P.lofo(Fm, y, fam, [(2,)])[0],
                "bigram + word": P.lofo(Fm, y, fam, [(0, 1)])[0]}
        pn, chosen = P.lofo(Fm, y, fam, [(0,), (1,), (0, 1)])
        pred["nested choice"] = pn
        i, j = np.triu_indices(len(y), 1)

        def agree(p, a, b):
            dp = np.sign(p[a] - p[b])
            return np.where(dp == 0, 0.5, dp == np.sign(y[a] - y[b]))

        rng = np.random.default_rng(1)
        u = np.unique(fam)
        idx = {c: np.flatnonzero(fam == c) for c in u}
        bp = {k: [] for k in pred}
        bm = {k: [] for k in pred}
        for _ in range(NB):
            k = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
            ii, jj = np.triu_indices(len(k), 1)
            a, b = k[ii], k[jj]
            keep = a != b
            for nm, p in pred.items():
                bp[nm].append(agree(p, a[keep], b[keep]).mean())
                bm[nm].append(np.mean(np.abs(p[k] - y[k])))
        bp = {k: np.array(v) for k, v in bp.items()}
        bm = {k: np.array(v) for k, v in bm.items()}
        rows = {nm: dict(pairwise=float(agree(p, i, j).mean()), pairwise_95=q(bp[nm]),
                         mae=float(np.mean(np.abs(p - y))), mae_95=q(bm[nm]), spearman=spearman(p, y))
                for nm, p in pred.items()}
        diffs = {}
        for a_, b_ in (("bigram + word", "word unigram"), ("bigram + word", "bigram"), ("bigram", "word unigram"),
                       ("nested choice", "word unigram")):
            diffs[f"{a_} minus {b_}"] = dict(
                pairwise=rows[a_]["pairwise"] - rows[b_]["pairwise"], pairwise_95=q(bp[a_] - bp[b_]),
                mae=rows[a_]["mae"] - rows[b_]["mae"], mae_95=q(bm[a_] - bm[b_]),
                share_boot_both_better_mae=float(np.mean(bm[a_] < bm[b_])))
        # how much the two raw statistics share, and in-sample partial correlations with the target
        r = lambda a, b: float(np.corrcoef(a, b)[0, 1])  # noqa: E731

        def partial(a, b, c):
            ra = np.linalg.lstsq(np.c_[np.ones(len(c)), c], a, rcond=None)
            rb = np.linalg.lstsq(np.c_[np.ones(len(c)), c], b, rcond=None)
            return r(a - np.c_[np.ones(len(c)), c] @ ra[0], b - np.c_[np.ones(len(c)), c] @ rb[0])

        # per-fold coefficients of the two-feature map, in units of target per SD of each statistic
        coefs = {}
        for f in u:
            tr = fam != f
            A = np.c_[np.ones(tr.sum()), (big[tr] - big.mean()) / big.std(), (word[tr] - word.mean()) / word.std()]
            coefs[str(f)] = np.linalg.lstsq(A, y[tr], rcond=None)[0][1:].tolist()
        out[tag] = dict(rows=rows, differences=diffs, nested_choices=chosen,
                        feature_columns=["bigram", "word unigram", "token unigram"],
                        raw=dict(pearson_bigram_word=r(big, word), spearman_bigram_word=spearman(big, word),
                                 pearson_bigram_target=r(big, y), pearson_word_target=r(word, y),
                                 partial_bigram_given_word=partial(big, y, word),
                                 partial_word_given_bigram=partial(word, y, big)),
                        both_map_coef_per_sd_by_heldout_family=coefs)
    return out, names


# ============================================================================ B. Published rankings
def word_ce_of_sample(a, tok, ev_words, ev_bytes):
    """Exactly datadecide_evalaware_rivals.py's word statistic: add-0.5 unigram over the first 90% of the sample."""
    a = a[: int(0.9 * len(a))]
    wc = Counter()
    for t in dds.texts(a, tok):
        wc.update(W.findall(t.lower()))
    vocab = set(wc) | set().union(*[set(v) for v in ev_words.values()])
    Z = sum(wc.values()) + 0.5 * len(vocab)
    return {e: float(-sum(n * np.log2((wc.get(w, 0) + 0.5) / Z) for w, n in c.items()) / ev_bytes[e])
            for e, c in ev_words.items()}


def part_b():
    cache = OUT / "public_word_ce.json"
    Cw = json.load(open(cache)) if cache.exists() else {}
    new = ["c4", "dclm_baseline", "dolma", "fineweb", "fineweb_edu", "pile", "redpajama_1t", "refinedweb", "slimpajama"]
    if any(s not in Cw for s in new):
        tok = dds.tokenizer()
        EV = dds.slugs("eval")
        bpt = {e: json.load(open(ROOT / f"results/datadecide/stats/{e}_cpu.json"))["bytes_per_token"] for e in EV}
        ev_tok = {e: np.fromfile(dds.DD / "evaldata" / f"{e[3:]}.npy", dtype="<u2") for e in EV}
        ev_words = {}
        for e, t in ev_tok.items():
            c = Counter()
            for tx in dds.texts(t, tok):
                c.update(W.findall(tx.lower()))
            ev_words[e] = c
        ev_bytes = {e: float(len(ev_tok[e][ev_tok[e] != dds.EOS]) * bpt[e]) for e in EV}
        for s in new:
            if s in Cw:
                continue
            a = np.fromfile(ROOT / "data/data_choice/samples" / f"{s}.npy", dtype="<u2")
            Cw[s] = word_ce_of_sample(a, tok, ev_words, ev_bytes)
            json.dump(Cw, open(cache, "w"), indent=1)
            print("word CE", s, round(np.mean(list(Cw[s].values())), 4), flush=True)
    # recipe features and target, exactly as data_choice_forecast.main
    DD, DC = ROOT / "results/datadecide", ROOT / "results/data_choice"
    M = json.load(open(DD / "stats/merged.json"))
    N = json.load(open(DC / "stats/merged.json"))
    S = dict(M["sources"])
    X = {k: dict(v) for k, v in M["xent"]["xent"].items()}
    for s in sorted(N["sources"]):
        S["new_" + s] = N["sources"][s]
        X["new_" + s] = N["xent_rows"][s]
    assert sorted(N["sources"]) == new
    fam_of = {r: f for f, (_, rs) in P.FAMILIES.items() for r in rs}
    names = sorted(fam_of)
    lv = pd.read_csv(DD / "final_levels.csv")
    T, _ = P.family_targets(lv, np.random.default_rng(0))
    F, _ = P.recipe_features(S, X, names + ["HF:" + s for s in new])
    y = T["all11"].loc[names, "level_1B"].values.astype(float)
    Crec = json.load(open(DD / "evalaware_rivals_cache.json"))
    slug_of = {s_["name"]: k for k, s_ in S.items() if s_.get("kind") == "recipe"}
    word_r = np.array([np.mean(list(Crec[slug_of[r]]["word"].values())) for r in names])
    word_n = np.array([np.mean(list(Cw[s].values())) for s in new])
    big_r = F.loc[names, "mismatch_all"].values.astype(float)
    big_n = F.loc[["HF:" + s for s in new], "mismatch_all"].values.astype(float)
    Utok = json.load(open(DC / "stats/unigram_ce.json"))["ce"]
    tok_n = {s: float(np.mean(list(Utok[s].values()))) for s in new}

    def fit(Xr, Xn):
        A = np.c_[np.ones(len(y)), Xr]
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        return np.c_[np.ones(len(Xn)), Xn] @ coef, coef

    p_big, c_big = fit(big_r[:, None], big_n[:, None])
    p_word, c_word = fit(word_r[:, None], word_n[:, None])
    p_both, c_both = fit(np.c_[big_r, word_r], np.c_[big_n, word_n])
    prim = json.load(open(DC / "forecast_primary_mismatch.json"))["forecasts"]["mismatch|all11"]["predicted_bpb"]
    check = max(abs(prim[s] - p_big[k]) for k, s in enumerate(new))
    rows = {"bigram (paper)": {s: float(p_big[k]) for k, s in enumerate(new)},
            "token unigram (labelled word-overlap in ED Table 1)": tok_n,
            "word unigram": {s: float(p_word[k]) for k, s in enumerate(new)},
            "bigram + word": {s: float(p_both[k]) for k, s in enumerate(new)}}
    # the published sources and tie rules, read from score.py without running its scoring or writing anything
    src = open(ROOT / "results/data_choice_score/score.py").read()
    head = src.split("# ---- 1. published comparisons")[1].split("# ---- 2. noise")[0].split("\n", 1)[1]
    ns = dict(np=np, geo=lambda *v: float(np.exp(np.mean(np.log(v)))), ROOT=ROOT)
    exec("import csv\n" + head, ns)  # defines the list S of published comparisons
    srcs = [s_ for s_ in ns["S"] if s_["estimand"] != "in_sample_fit"]

    def score(sr, pred):
        v = sr["vals"]
        k = n = 0
        for a, b in itertools.combinations(sorted(c for c in v if c in pred), 2):
            if abs(v[a] - v[b]) <= sr["tie_rel"] * max(abs(v[a]), abs(v[b])):
                continue
            obs = v[a] - v[b] if sr["lower_better"] else v[b] - v[a]
            n += 1
            k += int(np.sign(obs) == np.sign(pred[a] - pred[b]))
        return k, n

    per, pooled = {}, {}
    for sr in srcs:
        per[sr["id"]] = {r: "%d/%d" % score(sr, pv) for r, pv in rows.items()}
        for r, pv in rows.items():
            k, n = score(sr, pv)
            a = pooled.setdefault(sr["estimand"], {}).setdefault(r, [0, 0])
            a[0] += k
            a[1] += n
    pooled = {e: {r: f"{v[0]}/{v[1]}" for r, v in d.items()} for e, d in pooled.items()}
    order = {r: sorted(new, key=pv.get) for r, pv in rows.items()}
    return dict(check_bigram_reproduces_frozen_forecast_max_abs_diff=float(check),
                word_ce_public=dict(zip(new, word_n.tolist())), coef=dict(bigram=c_big.tolist(), word=c_word.tolist(),
                                                                         both=c_both.tolist()),
                predicted=rows, predicted_order=order, per_source=per, pooled=pooled,
                note="token unigram row = results/data_choice/stats/unigram_ce.json (token-level, add-0.5); "
                     "ED Table 1 and Methods call it word overlap")


# ============================================================================ C. Benchmark-text test
def part_c():
    fc = json.load(open(ROOT / "results/bench_mismatch/forecast.json"))["values"]
    fw = json.load(open(ROOT / "results/bench_mismatch/forecast_word_rival.json"))["values"]
    out = pd.read_csv(ROOT / "results/bench_mismatch_score/outcomes_1B_final_by_seed.csv")
    TASKS = {"arc_challenge": "arc_challenge", "arc_easy": "arc_easy", "boolq": "boolq", "commonsense_qa": "csqa",
             "hellaswag": "hellaswag", "mmlu": "mmlu", "openbookqa": "openbookqa", "piqa": "piqa",
             "social_iqa": "socialiqa", "winogrande": "winogrande"}
    fam_of = {r: f for f, (_, rs) in P.FAMILIES.items() for r in rs}
    recipes = sorted(fam_of)
    fams = list(P.FAMILIES)
    fidx = np.array([fams.index(fam_of[r]) for r in recipes])
    fam = np.array([fam_of[r] for r in recipes])
    pairs = np.array(list(itertools.combinations(range(25), 2)))
    pi, pj = pairs[:, 0], pairs[:, 1]
    ARMS = ["bigram raw", "token raw", "word raw", "bigram map", "word map", "bigram + word map", "z-sum bigram + word"]
    arrays, mae = {}, {a: [] for a in ARMS if "map" in a}
    for t, dd in TASKS.items():
        g = out[out.task == dd].groupby("data").acc
        mean, sd = g.mean().reindex(recipes), g.std(ddof=1).reindex(recipes)
        pooled_sd = float(np.sqrt((sd ** 2).mean()))
        y = mean.values
        dy = y[pi] - y[pj]
        keep = np.abs(dy) >= 2 * pooled_sd
        b = np.array([fc["ours_M_task"][r][t] for r in recipes])
        tk = np.array([fc["rival_unigram_ce"][r][t] for r in recipes])
        w = np.array([fw[r][t] for r in recipes])
        Fm = np.c_[b, w]
        # higher predicted accuracy = better; raw statistics: lower = better, so negate
        s = {"bigram raw": -b, "token raw": -tk, "word raw": -w,
             "bigram map": P.lofo(Fm, y, fam, [(0,)])[0], "word map": P.lofo(Fm, y, fam, [(1,)])[0],
             "bigram + word map": P.lofo(Fm, y, fam, [(0, 1)])[0],
             "z-sum bigram + word": -((b - b.mean()) / b.std() + (w - w.mean()) / w.std())}
        for a in mae:
            mae[a].append(np.abs(s[a] - y))
        corr = {}
        for a in ARMS:
            dx = s[a][pi] - s[a][pj]
            corr[a] = np.where(dx == 0, 0.5, (np.sign(dx) == np.sign(dy)).astype(float))
        arrays[t] = (keep, corr)

    def agree(wp, tasks):
        num = {a: 0.0 for a in ARMS}
        den = 0.0
        for t in tasks:
            keep, corr = arrays[t]
            ww = wp * keep
            den += ww.sum()
            for a in ARMS:
                num[a] += (ww * corr[a]).sum()
        return {a: num[a] / den for a in ARMS}

    pooled = agree(np.ones(len(pairs)), TASKS)
    rng = np.random.default_rng(20261016)
    boot = {a: [] for a in ARMS}
    bmae = {a: [] for a in mae}
    E = {a: np.array(v) for a, v in mae.items()}  # tasks x recipes
    for _ in range(NB):
        c = np.bincount(rng.integers(0, len(fams), len(fams)), minlength=len(fams)).astype(float)
        r_ = agree(c[fidx[pi]] * c[fidx[pj]], TASKS)
        for a in ARMS:
            boot[a].append(r_[a])
        wr = c[fidx]
        for a in mae:
            bmae[a].append(float((E[a] * wr).sum() / (wr.sum() * E[a].shape[0])))
    boot = {a: np.array(v) for a, v in boot.items()}
    bmae = {a: np.array(v) for a, v in bmae.items()}
    diffs = {}
    for a_, b_ in (("bigram + word map", "word map"), ("bigram + word map", "bigram map"), ("bigram map", "word map"),
                   ("z-sum bigram + word", "word raw"), ("z-sum bigram + word", "bigram raw"), ("bigram raw", "word raw"),
                   ("bigram raw", "token raw")):
        d = dict(pairwise=pooled[a_] - pooled[b_], pairwise_95=q(boot[a_] - boot[b_]))
        if a_ in mae and b_ in mae:
            d.update(mae=float(E[a_].mean() - E[b_].mean()), mae_95=q(bmae[a_] - bmae[b_]))
        diffs[f"{a_} minus {b_}"] = d
    return dict(pooled_agreement=pooled, pooled_95={a: q(boot[a]) for a in ARMS},
                mae_accuracy={a: float(E[a].mean()) for a in mae}, mae_95={a: q(bmae[a]) for a in mae},
                differences=diffs,
                note="raw rows reproduce bench_mismatch_score.json; map rows are leave-one-family-out affine maps of "
                     "each task's 1B accuracy on the statistic(s), fitted per task")


def main():
    A, names = part_a()
    B = part_b()
    Cc = part_c()
    res = dict(datadecide=A, published_rankings=B, benchmark_text=Cc)
    json.dump(res, open(OUT / "mismatch_vs_word.json", "w"), indent=1)
    for tag in A:
        print(f"\n[A {tag}]")
        for k, v in A[tag]["rows"].items():
            print(f"  {k:16s} pairwise {v['pairwise']:.3f} {np.round(v['pairwise_95'], 3)}  MAE {v['mae']:.4f} "
                  f"{np.round(v['mae_95'], 4)}  rho {v['spearman']:.3f}")
        for k, v in A[tag]["differences"].items():
            print(f"  {k:34s} dpair {v['pairwise']:+.3f} {np.round(v['pairwise_95'], 3)}  dMAE {v['mae']:+.4f} "
                  f"{np.round(v['mae_95'], 4)}")
        print("  raw", {k: round(v, 3) for k, v in A[tag]["raw"].items()})
        print("  nested", A[tag]["nested_choices"])
    print("\n[B] check", B["check_bigram_reproduces_frozen_forecast_max_abs_diff"])
    print(json.dumps(B["pooled"], indent=1))
    print(json.dumps(B["predicted_order"], indent=0))
    print("\n[C]", json.dumps({k: Cc[k] for k in ("pooled_agreement", "pooled_95", "mae_accuracy", "differences")}, indent=1))


if __name__ == "__main__":
    main()
