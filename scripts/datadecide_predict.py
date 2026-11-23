"""Predict DataDecide rates and levels from data statistics, and score the predictions against the measured targets.

    python scripts/datadecide_predict.py          # the one command; rerun unchanged when stats/merged.json lands
    python scripts/datadecide_predict.py [--stats results/datadecide/stats/merged.json] [--targets results/datadecide]
                                         [--out results/datadecide/predict.json] [--tasks ABCDE]
    python scripts/datadecide_predict.py --standin oracle|random --out /tmp/x.json    # pipeline test on stand-in stats

Inputs: merged.json from scripts/datadecide_stats.py; ladder_rates.csv and final_levels.csv from
scripts/datadecide_targets.py. If merged.json does not exist yet, the same structure is assembled in memory from the
per-source files in results/datadecide/stats/ (*_cpu.json, *_ref_*.json, xent.json), so partial statistics run too.

Tasks (unit = what is resampled in the bootstrap):
  A eval_rate      unit eval set with >= 3 in-distribution recipes; target = median ladder exponent alpha (60M-1B) over
                   those recipes
  B cell_level     unit (recipe, eval set) in distribution, clustered by eval set; target = bits/byte at 1B
  C xdomain_level  per eval set, unit recipe (clustered by family); target = bits/byte at 1B on that set
  D xdomain_rate   per eval set, unit recipe; target = ladder exponent on that set
  E heldout_family unit recipe (25), held out by corpus family (FAMILIES, 6 folds); targets = bits/byte at 1B averaged
                   over the 11 eval sets (and over the 4 sets no recipe trains on), and the exponent of that average
                   over the 4M-1B ladder (and 60M-1B). Every predictor is an affine map with constants shared by all
                   recipes (no per-corpus intercepts), fitted on the training families only. Baselines: compression
                   alone, n-gram conditional entropy alone, the best compression + n-gram pair (each variant chosen
                   inside the training families). Ours: the eval-set mismatch M (theory term, needs xent) and a <= 2
                   feature model chosen inside the training families from all statistics, with and without
                   reference-LM features (REF_LM marks them). Scores: MAE, Spearman, pairwise decision accuracy,
                   MAE ratio to the strongest baseline, bootstrap over families. Small-model training rows for scale.
Predictors:
  info_rule        alpha = s(n*) / (1 + gamma): s = -dln h/dln n of the source's reference curve at the effective
                   context n*, nu = 1/(1 + gamma) (the information rule). n* from the frozen TinyStories forecaster
                   constants (0 fitted constants) or with n0 refitted (1 constant, leave one unit out)
  forecaster       L(D) = h(n0 (D b^c)^nu), D training bytes; frozen TinyStories constants (0) or n0 refitted (1)
  ..+mismatch      out of distribution: L = h_e(n_eff) + lam * M(r, e), M = bigram cross-entropy of e under r minus
                   under e itself (bits/byte); alpha shrinks by h / (h + lam M); lam fitted (1 constant)
  rivals           single statistics (compression, bigram entropy, reference-model bits/byte at full context,
                   unigram entropy, Zipf, bytes/token, Cagnetta gamma/(2 beta), bigram cross-entropy), each mapped by a
                   leave-one-out affine fit (2 constants); small-model ordering (training rival) from the targets
Scores: Spearman and pairwise ordering accuracy of the (leave-one-out) predictions, rms error, and 80% bootstrap
intervals over units (cluster bootstrap where clusters are given). Predictions with fitted constants are always
leave-one-unit-out (leave-one-cluster-out where clusters are given).
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = dict(nu=0.6461, n0=6.679e-4, c=0.4903)  # (fit on the TinyStories bytes ladder, 2^20-2^24)
warnings.filterwarnings("ignore", category=RuntimeWarning)  # constant bootstrap resamples
PRIMARY = os.environ.get("DD_PRIMARY", "SmolLM2-135M")
GAMMA_WIN = "4-64"
LARGE = ["60M", "90M", "150M", "300M", "530M", "750M", "1B"]
NBOOT = 2000
COMPONENT = {  # eval set -> substrings of the training-file paths of its source (for the mixture weight w)
    "dolma_wiki": ["v1_6-decontaminated/wiki"], "dolma_books": ["v1_6-decontaminated/books"],
    "dolma_pes2o": ["v1_6-decontaminated/pes2o"], "dolma_reddit": ["preprocessed/reddit/"],
    "dolma_stack": ["starcoder/"], "c4_en": ["preprocessed/c4/", "v1_6-decontaminated/c4"],
    "dolma_common-crawl": ["/cc_en_"]}


# ------------------------------------------------------------------------------------------------- reference curves
class Curve:
    """h(n bytes): piecewise linear in log n through the paired rows; beyond the last row, the fitted power law
    h_inf + a n^-g shifted to meet the last row (flat if no fit)."""

    def __init__(self, ref):
        rows = ref["paired"]["rows"]
        self.x = np.log([r["context_bytes_exact_mean"] for r in rows])
        self.y = np.array([r["bits_per_byte"] for r in rows])
        self.fit = ref.get("fit_paired")
        self.gamma = (ref.get("gamma") or {}).get(GAMMA_WIN, {}).get("gamma")

    def h(self, n):
        n = np.asarray(n, float)
        y = np.interp(np.log(n), self.x, self.y)
        if self.fit:
            nl = np.exp(self.x[-1])
            tail = self.y[-1] + self.fit["a"] * (n ** -self.fit["g"] - nl ** -self.fit["g"])
            y = np.where(np.log(n) > self.x[-1], tail, y)
        return y

    def s(self, n, eps=0.05):
        """-d ln h / d ln n by a central difference."""
        return -(np.log(self.h(n * np.exp(eps))) - np.log(self.h(n * np.exp(-eps)))) / (2 * eps)


def curve(src, model=PRIMARY):
    r = src.get("ref", {}).get(model)
    return Curve(r) if r and r.get("paired") else None


# ------------------------------------------------------------------------------------------------- scoring
def spearman(p, y):
    if len(p) < 3 or np.std(p) == 0 or np.std(y) == 0:
        return np.nan
    return float(pd.Series(p).rank().corr(pd.Series(y).rank()))


def pairwise(p, y):
    i, j = np.triu_indices(len(p), 1)
    if not len(i):
        return np.nan
    dp = np.sign(p[i] - p[j])
    return float(np.mean(np.where(dp == 0, 0.5, dp == np.sign(y[i] - y[j]))))  # tied predictions count one half


def score(p, y, clusters=None, rng=None):
    p, y = np.asarray(p, float), np.asarray(y, float)
    ok = np.isfinite(p) & np.isfinite(y)
    p, y = p[ok], y[ok]
    cl = np.arange(len(p)) if clusters is None else np.asarray(clusters)[ok]
    out = dict(n=int(len(p)), spearman=spearman(p, y), pairwise=pairwise(p, y), rms=float(np.sqrt(np.mean((p - y) ** 2))))
    if len(p) >= 4 and rng is not None:
        u = np.unique(cl)
        idx = {c: np.flatnonzero(cl == c) for c in u}
        bs, br = [], []
        for _ in range(NBOOT):
            k = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
            bs.append(spearman(p[k], y[k]))
            br.append(np.sqrt(np.mean((p[k] - y[k]) ** 2)))
        out["spearman_80"] = [float(np.nanquantile(bs, 0.1)), float(np.nanquantile(bs, 0.9))]
        out["rms_80"] = [float(np.quantile(br, 0.1)), float(np.quantile(br, 0.9))]
    return out


def loo_groups(clusters, n):
    cl = np.arange(n) if clusters is None else np.asarray(clusters)
    return [(np.flatnonzero(cl != c), np.flatnonzero(cl == c)) for c in np.unique(cl)]


def loo_affine(x, y, clusters=None):
    """Leave-one-unit(cluster)-out affine map of a statistic onto the target (2 constants)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    p = np.full(len(x), np.nan)
    for tr, te in loo_groups(clusters, len(x)):
        ok = np.isfinite(x[tr]) & np.isfinite(y[tr])
        if ok.sum() >= 2 and np.std(x[tr][ok]) > 0:
            b, a = np.polyfit(x[tr][ok], y[tr][ok], 1)
            p[te] = a + b * x[te]
    return p


def loo_scalar(fn, y, clusters=None, bounds=(-12.0, 12.0)):
    """Leave-one-out fit of one constant theta (fn(theta) -> predictions for all units)."""
    y = np.asarray(y, float)
    p = np.full(len(y), np.nan)
    for tr, te in loo_groups(clusters, len(y)):
        def loss(t):
            q = fn(t)[tr]
            ok = np.isfinite(q) & np.isfinite(y[tr])
            return np.mean((q[ok] - y[tr][ok]) ** 2) if ok.any() else 1e9
        t = minimize_scalar(loss, bounds=bounds, method="bounded").x
        p[te] = fn(t)[te]
    return p


# ------------------------------------------------------------------------------------------------- the rules
def neff(D_bytes, b, ln_n0=np.log(TS["n0"]), nu=TS["nu"], c=TS["c"]):
    return np.exp(ln_n0) * (D_bytes * b ** c) ** nu


def forecast_curve(cv, D_tok, b_train, ln_n0=np.log(TS["n0"]), nu=TS["nu"]):
    return cv.h(neff(np.asarray(D_tok) * b_train, b_train, ln_n0, nu))


def ladder_alpha(L, D_tok):
    x = np.log(D_tok)
    xc = x - x.mean()
    return float(-((np.log(L) - np.log(L).mean()) * xc).sum() / (xc ** 2).sum())


def info_alpha(cv, D_mid_tok, b_train, ln_n0=np.log(TS["n0"])):
    g = cv.gamma
    if g is None:
        return np.nan
    n = neff(D_mid_tok * b_train, b_train, ln_n0, 1.0 / (1.0 + g))
    return float(cv.s(n) / (1.0 + g))


# ------------------------------------------------------------------------------------------------- tasks
def stat_table(S):
    rows = {}
    for slug, s in S.items():
        if "bytes_per_token" not in s:
            continue
        cv = curve(s)
        g = cv.gamma if cv else None
        bt = s["beta"]["beta"]["tok_1_200"] if s.get("beta") else None
        rows[slug] = dict(name=s["name"], kind=s["kind"], bpt=s["bytes_per_token"], lzma=s["compression"]["lzma_bpb_1MB"],
                          bigram_bpb=s["H_cond"][1] / s["bytes_per_token"], unigram=s["unigram_bits"], zipf=s["zipf_s"],
                          ref_bpb_max=float(cv.y[-1]) if cv else np.nan, gamma=g if g is not None else np.nan,
                          cagnetta=g / (2 * bt) if g and bt else np.nan)
    return pd.DataFrame(rows).T


def mixture_weights(recipe_paths):
    rp = json.load(open(recipe_paths))["recipes"]
    w = {}
    for r, v in rp.items():
        tot = sum(s for _, s in v["files"])
        for e, keys in COMPONENT.items():
            w[(r, e)] = sum(s for p, s in v["files"] if any(k in p for k in keys)) / tot
    return w


def load_stats(path):
    """merged.json if it exists; otherwise the same structure assembled from the per-source files beside it (the
    merge step of scripts/datadecide_stats.py, kept in memory so nothing is written into the stats directory)."""
    if os.path.exists(path):
        return json.load(open(path))
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import datadecide_stats as dds
    from tok_stats import gamma_stats  # on sys.path via datadecide_stats
    d = os.path.dirname(path)
    out = {}
    for s in dds.slugs():
        rec = dict(kind="eval" if s.startswith("ev_") else "recipe", name=dds.slug_name(s), ref={})
        f = os.path.join(d, f"{s}_cpu.json")
        if os.path.exists(f):
            rec.update(json.load(open(f)))
        for mname in dds.MODELS:
            short = mname.split("/")[-1]
            g = os.path.join(d, f"{s}_ref_{short}.json")
            if not os.path.exists(g):
                continue
            r = json.load(open(g))
            e = dict(paired=r["paired"], docpos=r["docpos"])
            if r["paired"] and "bytes_per_token" in rec:
                rows = r["paired"]["rows"]
                e["gamma"] = {f"{w[0]}-{w[1]}": gamma_stats({short: r["paired"]}, rec["bytes_per_token"],
                                                             rec["beta"]["beta"]["tok_1_200"], w)[short]
                              for w in dds.GAMMA_WINDOWS if w[1] * rec["bytes_per_token"] <= rows[-1]["context_bytes_exact_mean"]}
                nb = np.array([x["context_bytes_exact_mean"] for x in rows])
                h = np.array([x["bits_per_byte"] for x in rows])
                e["fit_paired"] = dds.powerlaw_fit(nb[nb >= 8], h[nb >= 8])
            rec["ref"][short] = e
        out[s] = rec
    xf = os.path.join(d, "xent.json")
    print(f"[partial] {path} not found; assembled from per-source files in {d}")
    return dict(sources=out, xent=json.load(open(xf)) if os.path.exists(xf) else None, partial=True)


def run(stats, targets, recipe_paths, rng, tasks="ABCDE"):
    M = load_stats(stats)
    S, X = M["sources"], (M.get("xent") or {}).get("xent")
    T = stat_table(S)
    ev = {r["name"]: k for k, r in T.iterrows() if r["kind"] == "eval"}
    rc = {r["name"]: k for k, r in T.iterrows() if r["kind"] == "recipe"}
    rates = pd.read_csv(os.path.join(targets, "ladder_rates.csv"))
    lv = pd.read_csv(os.path.join(targets, "final_levels.csv"))
    D = lv.groupby("params")["tokens"].median()
    D_large = D.loc[LARGE].values
    D_mid = float(np.exp(np.log(D_large).mean()))
    W = mixture_weights(recipe_paths)
    res = {}
    RIVALS = ["lzma", "bigram_bpb", "ref_bpb_max", "unigram", "zipf", "bpt", "cagnetta"]

    # ---- A: rate of each in-distribution eval set
    ind = rates[rates.in_dist != "no"]
    A = ind.groupby("eval").agg(y=("alpha_large", "median"), n=("data", "count"), small=("alpha_small", "median"))
    A = A[(A.n >= 3) & A.index.isin(list(ev))]
    if len(A) >= 3 and "A" in tasks:
        st = T.loc[[ev[e] for e in A.index]]
        cvs = [curve(S[ev[e]]) for e in A.index]
        b_tr = st["bpt"].values.astype(float)  # in-distribution: the learner reads this source's text
        y = A["y"].values
        P = {"info_rule": np.array([info_alpha(c, D_mid, b) if c else np.nan for c, b in zip(cvs, b_tr)]),
             "forecaster": np.array([ladder_alpha(forecast_curve(c, D_large, b), D_large) if c else np.nan
                                     for c, b in zip(cvs, b_tr)])}
        P["info_rule_n0"] = loo_scalar(lambda t: np.array([info_alpha(c, D_mid, b, t) if c else np.nan
                                                           for c, b in zip(cvs, b_tr)]), y)
        P["forecaster_n0"] = loo_scalar(lambda t: np.array([ladder_alpha(forecast_curve(c, D_large, b, t), D_large)
                                                            if c else np.nan for c, b in zip(cvs, b_tr)]), y)
        for k in RIVALS:
            P[f"rival_{k}"] = loo_affine(st[k].values.astype(float), y)
        P["rival_cagnetta_raw"] = st["cagnetta"].values.astype(float)
        P["rival_small_models_4M-20M"] = loo_affine(A["small"].values, y)
        res["A_eval_rate"] = dict(units=list(A.index), target=y.tolist(),
                                  constants={"info_rule": 0, "forecaster": 0, "info_rule_n0": 1, "forecaster_n0": 1,
                                             "rival_*": 2, "rival_cagnetta_raw": 0},
                                  predictions={k: v.tolist() for k, v in P.items()},
                                  scores={k: score(v, y, rng=rng) for k, v in P.items()})

    # ---- B: level of each in-distribution (recipe, eval) cell at 1B
    B = lv[(lv.params == "1B") & (lv.in_dist != "no") & lv["eval"].isin(list(ev))].reset_index(drop=True)
    if len(B) >= 4 and "B" in tasks:
        cvs = [curve(S[ev[e]]) for e in B["eval"]]
        b_e = np.array([float(T.loc[ev[e], "bpt"]) for e in B["eval"]])
        w = np.array([W.get((r, e), np.nan) for r, e in zip(B["data"], B["eval"])])
        y, cl = B["bpb"].values, B["eval"].values
        D1 = float(D.loc["1B"])
        fc = lambda t, frac: np.array([forecast_curve(c, [D1 * f], b, t)[0] if c and f > 0 else np.nan  # noqa: E731
                                       for c, b, f in zip(cvs, b_e, frac)])
        P = {"forecaster": fc(np.log(TS["n0"]), np.ones(len(B))),
             "forecaster_n0": loo_scalar(lambda t: fc(t, np.ones(len(B))), y, cl),
             "forecaster_w_n0": loo_scalar(lambda t: fc(t, w), y, cl)}
        for k in ["lzma", "bigram_bpb", "ref_bpb_max", "unigram", "bpt"]:
            P[f"rival_{k}"] = loo_affine(np.array([float(T.loc[ev[e], k]) for e in B["eval"]]), y, cl)
        for s in ("60M", "150M"):
            small = lv[lv.params == s].set_index(["data", "eval"])["bpb"]
            P[f"rival_small_models_{s}"] = loo_affine(np.array([small.get((r, e), np.nan) for r, e in zip(B["data"], B["eval"])]), y, cl)
        res["B_cell_level"] = dict(units=[f"{r} | {e}" for r, e in zip(B["data"], B["eval"])], clusters="eval set",
                                   mixture_weight=w.tolist(), target=y.tolist(),
                                   predictions={k: v.tolist() for k, v in P.items()},
                                   scores={k: score(v, y, cl, rng) for k, v in P.items()})

    # ---- C and D: recipes on each eval set (cross-domain)
    if X and len(rc) >= 4 and ("C" in tasks or "D" in tasks):
        C, Dd = {}, {}
        for e, es in ev.items():
            sub = lv[(lv.params == "1B") & (lv["eval"] == e) & lv["data"].isin(list(rc))].set_index("data")
            rr = rates[(rates["eval"] == e) & rates["data"].isin(list(rc))].set_index("data")
            names = [r for r in sub.index if r in rr.index and rc[r] in X and es in X[rc[r]]]
            if len(names) < 4:
                continue
            fam = rr.loc[names, "family"].values
            b_e = float(T.loc[es, "bpt"])
            self_x = X.get(es, {}).get(es, np.nan)
            Mm = np.array([(X[rc[r]][es] - self_x) / b_e for r in names])  # bits/byte
            xe = np.array([X[rc[r]][es] / b_e for r in names])
            cv = curve(S[es])
            b_r = np.array([float(T.loc[rc[r], "bpt"]) for r in names])
            yC, yD = sub.loc[names, "bpb"].values, rr.loc[names, "alpha_large"].values
            h1 = np.array([forecast_curve(cv, [float(D.loc["1B"])], b)[0] if cv else np.nan for b in b_r])
            PC = {"forecaster+mismatch": loo_scalar(lambda t: h1 + np.exp(t) * Mm, yC, fam, (-12, 3)),
                  "rival_ngram_xent": loo_affine(xe, yC, fam)}
            for k in ["lzma", "ref_bpb_max", "bpt"]:
                PC[f"rival_recipe_{k}"] = loo_affine(np.array([float(T.loc[rc[r], k]) for r in names]), yC, fam)
            for s in ("60M", "150M"):
                small = lv[(lv.params == s) & (lv["eval"] == e)].set_index("data")["bpb"]
                PC[f"rival_small_models_{s}"] = loo_affine(small.loc[names].values, yC, fam)
            C[e] = dict(units=names, target=yC.tolist(), predictions={k: v.tolist() for k, v in PC.items()},
                        scores={k: score(v, yC, fam, rng) for k, v in PC.items()})
            # rate: alpha(r, e) = alpha_e * h / (h + lam M), alpha_e from the in-distribution rule for e
            a_e = info_alpha(cv, D_mid, b_e) if cv else np.nan
            hm = np.array([forecast_curve(cv, [D_mid], b)[0] if cv else np.nan for b in b_r])
            PD = {"info_rule+mismatch": loo_scalar(lambda t: a_e * hm / (hm + np.exp(t) * Mm), yD, fam, (-12, 3)),
                  "rival_ngram_xent": loo_affine(xe, yD, fam),
                  "rival_recipe_cagnetta": loo_affine(np.array([float(T.loc[rc[r], "cagnetta"]) for r in names]), yD, fam),
                  "rival_recipe_lzma": loo_affine(np.array([float(T.loc[rc[r], "lzma"]) for r in names]), yD, fam),
                  "rival_small_models_4M-20M": loo_affine(rr.loc[names, "alpha_small"].values, yD, fam)}
            Dd[e] = dict(units=names, target=yD.tolist(), predictions={k: v.tolist() for k, v in PD.items()},
                         scores={k: score(v, yD, fam, rng) for k, v in PD.items()})
        for key, R in (("C_xdomain_level", C), ("D_xdomain_rate", Dd)):
            if R:
                preds = sorted({k for v in R.values() for k in v["scores"]})
                summ = {k: dict(median_spearman=float(np.nanmedian([v["scores"][k]["spearman"] for v in R.values()])),
                                median_pairwise=float(np.nanmedian([v["scores"][k]["pairwise"] for v in R.values()])))
                        for k in preds}
                res[key] = dict(by_eval=R, summary=summ, clusters="recipe family")
    if "E" in tasks:
        res["E_heldout_family"] = heldout_family(S, X, lv, rng)
    return res


# ------------------------------------------------------------------------------------------------- task E
# Corpus families, the unit that is held out and resampled. A family shares one text pool, so a held-out family is
# text the predictor has not seen in any form. Grouping as DataDecide builds the recipes; rationale per line.
FAMILIES = {
    "dolma": ("the Dolma source mix and its one-component ablations",
              ["Dolma1.6++", "Dolma1.7", "Dolma1.7 (no Flan)", "Dolma1.7 (no Reddit)", "Dolma1.7 (no code)",
               "Dolma1.7 (no math, code)"]),
    "dclm": ("the DCLM-Baseline pool and its quality-classifier subsets",
             ["DCLM-Baseline", "DCLM-Baseline (QC 10%)", "DCLM-Baseline (QC 20%)", "DCLM-Baseline (QC 7%, FW2)",
              "DCLM-Baseline (QC 7%, FW3)", "DCLM-Baseline (QC FW 10%)", "DCLM-Baseline (QC FW 3%)"]),
    "dclm_dolma_mix": ("file-count mixes of the DCLM and Dolma pools (they share text with both, see the plan)",
                       ["DCLM-Baseline 25% / Dolma 75%", "DCLM-Baseline 50% / Dolma 50%", "DCLM-Baseline 75% / Dolma 25%"]),
    "falcon": ("RefinedWeb alone or with Dolma CC, and quality-filtered subsets of the latter",
               ["Falcon", "Falcon+CC", "Falcon+CC (QC 10%)", "Falcon+CC (QC 20%)", "Falcon+CC (QC Orig 10%)",
                "Falcon+CC (QC Tulu 10%)"]),
    "fineweb": ("FineWeb filtered for educational value (Edu) or refined by a model (Pro)", ["FineWeb-Edu", "FineWeb-Pro"]),
    "c4": ("C4 alone", ["C4"]),
}
OOD4 = ["pile", "wikitext_103", "m2d2_s2orc", "ice"]  # eval sets no recipe trains on
SMALL = ["4M", "6M", "8M", "10M", "14M", "16M", "20M"]
COMP = ["comp_lzma_8MB", "comp_lzma_1MB", "comp_bz2_1MB", "comp_zlib_1MB"]  # bits/byte, equal text size per recipe
NGRAM = ["ngram_H1", "ngram_H2", "ngram_H3"]  # plug-in conditional entropy H_k / bytes per token, bits/byte


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return np.nan
        d = d[k]
    return np.nan if d is None else d


def recipe_features(S, X, names):
    """Statistics of each recipe from a sample of its training stream (the mismatch also reads the fixed eval-set
    text), so every column exists before training on that recipe. Columns in the returned list use a pretrained
    reference LM."""
    rec = {s["name"]: k for k, s in S.items() if s.get("kind") == "recipe"}
    ev = {s["name"]: k for k, s in S.items() if s.get("kind") == "eval"}
    rows, ref_cols = {}, set()
    for r in names:
        k = rec.get(r)
        s = S.get(k, {})
        f = {}
        if "bytes_per_token" in s:
            b = s["bytes_per_token"]
            f.update(comp_lzma_8MB=_get(s, "compression", "lzma_bpb_8MB"), comp_lzma_1MB=_get(s, "compression", "lzma_bpb_1MB"),
                     comp_bz2_1MB=_get(s, "compression", "bz2_bpb_1MB"), comp_zlib_1MB=_get(s, "compression", "zlib_bpb_1MB"))
            for j in range(3):
                f[f"ngram_H{j + 1}"] = s["H_cond"][j] / b if len(s.get("H_cond", [])) > j else np.nan
            f.update(bytes_per_token=b, unigram_bits=_get(s, "unigram_bits"), zipf=_get(s, "zipf_s"),
                     beta_tok=_get(s, "beta", "beta", "tok_1_200"), beta_byte=_get(s, "beta", "beta", "byte_10_1000"),
                     log_doc_len=np.log(_get(s, "doc_len_tokens", "mean")), doc_ctx_512=_get(s, "doc_context", "512"))
        if X and k in X and all(e in X and e in X[e] and e in X[k] for e in ev.values()):
            mm = {n: (X[k][e] - X[e][e]) / S[e]["bytes_per_token"] for n, e in ev.items()}  # bits/byte
            f["mismatch_all"] = float(np.mean(list(mm.values())))
            f["mismatch_ood4"] = float(np.mean([mm[n] for n in OOD4 if n in mm]))
        for m, e in s.get("ref", {}).items():
            if not e.get("paired"):
                continue
            cv = Curve(e)
            f[f"ref_h512_{m}"] = float(cv.y[-1])
            ref_cols.add(f"ref_h512_{m}")
            if cv.gamma:
                f[f"ref_gamma_{m}"] = cv.gamma
                f[f"ref_cagnetta_{m}"] = cv.gamma / (2 * f.get("beta_tok", np.nan))
                ref_cols |= {f"ref_gamma_{m}", f"ref_cagnetta_{m}"}
        rows[r] = f
    return pd.DataFrame(rows).T.reindex(names).astype(float), sorted(ref_cols)


def family_targets(lv, rng, ndraw=200):
    """Per recipe: bits/byte at 1B averaged over the eval sets (all 11, and the 4 out-of-distribution sets), and the
    exponent of that average along the final-checkpoint ladder (4M-1B and 60M-1B). Seed SD of each target from a
    parametric bootstrap of the seed means (sd/sqrt(n) per cell; cells with one seed borrow that cell's median sd)."""
    lv = lv.copy()
    lv["sd_bpb"] = lv["sd"] * lv["bpb"] / lv["mean"]
    lv["sd_bpb"] = lv["sd_bpb"].fillna(lv.groupby(["data", "eval"])["sd_bpb"].transform("median"))
    lv["se"] = lv["sd_bpb"] / np.sqrt(lv["n"])
    sizes = SMALL + LARGE
    out, noise = {}, {}
    for tag, evs in (("all11", None), ("ood4", OOD4)):
        sub = lv if evs is None else lv[lv["eval"].isin(evs)]
        n_e = sub["eval"].nunique()
        piv = sub.pivot_table(index=["data", "params"], columns="eval", values="bpb")
        se = sub.pivot_table(index=["data", "params"], columns="eval", values="se")
        assert piv.notna().all().all(), "every (recipe, size) needs every eval set"
        D = sub.groupby(["data", "params"])["tokens"].mean()
        names = sorted(sub["data"].unique())

        def targets(P):
            L = P.mean(axis=1)
            res = {}
            for r in names:
                Lr = L.loc[r].reindex(sizes).values
                Dr = D.loc[r].reindex(sizes).values
                res[r] = dict(level_1B=Lr[-1], rate_4M_1B=ladder_alpha(Lr, Dr),
                              rate_60M_1B=ladder_alpha(Lr[len(SMALL):], Dr[len(SMALL):]),
                              rate_small=ladder_alpha(Lr[:len(SMALL)], Dr[:len(SMALL)]),
                              **{f"level_{s}": L.loc[(r, s)] for s in ("4M", "20M", "60M", "150M")})
            return pd.DataFrame(res).T

        T = targets(piv)
        draws = [targets(piv + rng.standard_normal(piv.shape) * se.values) for _ in range(ndraw)]
        sd = pd.concat(draws).groupby(level=0).std()
        out[tag], noise[tag] = T, sd
        out[tag].attrs["n_eval"] = n_e
    return out, noise


def _fit_predict(F, y, tr, te, cols):
    A = np.c_[np.ones(len(tr)), F[np.ix_(tr, cols)]]
    coef = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    return np.c_[np.ones(len(te)), F[np.ix_(te, cols)]] @ coef


def lofo(F, y, fam, cands, idx=None):
    """Leave-one-family-out predictions on the units idx. With several candidate feature sets, each fold picks the set
    with the lowest leave-one-family-out MAE inside its own training families (nested), then fits it on all of them.
    Constants are shared by every recipe; nothing is fitted on the held-out family."""
    idx = np.arange(len(y)) if idx is None else np.asarray(idx)
    p, chosen = np.full(len(y), np.nan), {}
    for f in np.unique(fam[idx]):
        te, tr = idx[fam[idx] == f], idx[fam[idx] != f]
        c = cands[0]
        if len(cands) > 1:
            inner = [np.mean(np.abs(lofo(F, y, fam, [c], tr)[0][tr] - y[tr])) for c in cands]
            c = cands[int(np.argmin(inner))]
        p[te] = _fit_predict(F, y, tr, te, list(c))
        chosen[str(f)] = list(c)
    return p, chosen


def _metrics(p, y):
    return dict(mae=float(np.mean(np.abs(p - y))), rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                spearman=spearman(p, y), pairwise=pairwise(p, y))


def heldout_family(S, X, lv, rng):
    fam_of = {r: f for f, (_, rs) in FAMILIES.items() for r in rs}
    names = sorted(fam_of)
    assert set(names) == set(lv["data"].unique()), "FAMILIES must list the 25 recipes exactly"
    fam = np.array([fam_of[r] for r in names])
    F, ref_cols = recipe_features(S, X, names)
    have = [c for c in F.columns if F[c].notna().all()]
    missing = {c: [r for r in names if not np.isfinite(F.loc[r, c])] for c in F.columns if c not in have}
    lack_cpu = [r for r in names if not np.isfinite(F.loc[r].get("comp_lzma_1MB", np.nan))]
    nonref = [c for c in have if c not in ref_cols]
    withref = have
    col = {c: i for i, c in enumerate(F.columns)}
    Fm = F.values
    sets = lambda pool: [(col[a],) for a in pool] + [(col[a], col[b]) for i, a in enumerate(pool) for b in pool[i + 1:]]  # noqa: E731
    comp, ngram = [c for c in COMP if c in have], [c for c in NGRAM if c in have]
    T, noise = family_targets(lv, rng)
    out = dict(families={f: dict(rationale=d, recipes=rs) for f, (d, rs) in FAMILIES.items()}, units=names,
               features=dict(used_no_ref_lm=nonref, used_ref_lm=[c for c in have if c in ref_cols],
                             unavailable=missing, recipes_without_cpu_stats=lack_cpu,
                             mismatch_needs="xent.json (bigram cross-entropy of each eval set under each recipe)"),
               targets={})
    for tag in ("all11", "ood4"):
        for tname in ("level_1B", "rate_4M_1B", "rate_60M_1B"):
            y = T[tag].loc[names, tname].values.astype(float)
            P, chosen = {}, {}
            P["const_family_mean"], chosen["const_family_mean"] = lofo(Fm, y, fam, [()])
            if comp:
                P["base_compression"], chosen["base_compression"] = lofo(Fm, y, fam, [(col[c],) for c in comp])
            if ngram:
                P["base_ngram"], chosen["base_ngram"] = lofo(Fm, y, fam, [(col[c],) for c in ngram])
            if comp and ngram:
                P["base_comp+ngram"], chosen["base_comp+ngram"] = lofo(Fm, y, fam, [(col[a], col[b]) for a in comp for b in ngram])
            mm = "mismatch_ood4" if tag == "ood4" else "mismatch_all"
            if mm in have:
                P["ours_mismatch"], chosen["ours_mismatch"] = lofo(Fm, y, fam, [(col[mm],)])
            if nonref:
                P["ours_select_noref"], chosen["ours_select_noref"] = lofo(Fm, y, fam, sets(nonref))
            if len(withref) > len(nonref):
                P["ours_select_ref_lm"], chosen["ours_select_ref_lm"] = lofo(Fm, y, fam, sets(withref))
            # training rival (costs training runs; for scale, not a data statistic)
            small = ["level_4M", "level_20M", "level_60M", "level_150M"] if tname == "level_1B" else ["rate_small"]
            for s in small:
                x = T[tag].loc[names, s].values.astype(float)[:, None]
                P[f"train_{s}"] = lofo(x, y, fam, [(0,)])[0]
            P = {k: np.asarray(v, float) for k, v in P.items()}
            chosen = {k: {f: [F.columns[i] for i in c] for f, c in v.items()} for k, v in chosen.items()}
            rows = {k: _metrics(v, y) for k, v in P.items()}
            for s in small:  # the raw small-model order, no fit (DataDecide's decision accuracy)
                x = T[tag].loc[names, s].values.astype(float)
                rows[f"train_{s}"]["pairwise_raw"] = pairwise(x, y)
                rows[f"train_{s}"]["spearman_raw"] = spearman(x, y)
            base = [k for k in P if k.startswith("base_")]
            best = min(base, key=lambda k: rows[k]["mae"]) if base else None
            # bootstrap over families, on the fixed held-out predictions
            fams = np.unique(fam)
            ix = {f: np.flatnonzero(fam == f) for f in fams}
            bs = {k: dict(mae=[], pairwise=[], spearman=[], ratio_best_base=[], ratio_const=[]) for k in P}
            for _ in range(NBOOT):
                k_ = np.concatenate([ix[f] for f in rng.choice(fams, len(fams))])
                e = {k: np.mean(np.abs(v[k_] - y[k_])) for k, v in P.items()}
                for k, v in P.items():
                    bs[k]["mae"].append(e[k])
                    bs[k]["pairwise"].append(pairwise(v[k_], y[k_]))
                    bs[k]["spearman"].append(spearman(v[k_], y[k_]))
                    bs[k]["ratio_best_base"].append(e[k] / e[best] if best else np.nan)
                    bs[k]["ratio_const"].append(e[k] / e["const_family_mean"])
            for k in P:
                rows[k]["ratio_best_base"] = rows[k]["mae"] / rows[best]["mae"] if best else np.nan
                rows[k]["ratio_const"] = rows[k]["mae"] / rows["const_family_mean"]["mae"]
                for m, v in bs[k].items():
                    rows[k][m + "_80"] = [float(np.nanquantile(v, 0.1)), float(np.nanquantile(v, 0.9))]
                    rows[k][m + "_95"] = [float(np.nanquantile(v, 0.025)), float(np.nanquantile(v, 0.975))]
            out["targets"][f"{tname}|{tag}"] = dict(
                target=dict(zip(names, y.tolist())), family=dict(zip(names, fam.tolist())),
                seed_sd_median=float(noise[tag].loc[names, tname].median()), target_sd_across_recipes=float(np.std(y)),
                best_baseline=best, predictions={k: v.tolist() for k, v in P.items()}, chosen=chosen, scores=rows)
    return out


def print_heldout(E):
    f = E["features"]
    print("\nE_heldout_family  (leave one corpus family out, 6 families, 25 recipes; intervals: bootstrap over families)")
    print("  features, no reference LM:", ", ".join(f["used_no_ref_lm"]) or "-")
    print("  features, reference LM:   ", ", ".join(f["used_ref_lm"]) or "- (no recipe reference curves yet)")
    if f["recipes_without_cpu_stats"]:
        print("  recipes without CPU statistics:", ", ".join(f["recipes_without_cpu_stats"]))
    for c, rs in f["unavailable"].items():
        print(f"  unavailable {c}: missing for {len(rs)} recipes")
    for t, R in E["targets"].items():
        print(f"\n  {t}: target SD across recipes {R['target_sd_across_recipes']:.4f}, median seed SD "
              f"{R['seed_sd_median']:.4f}; strongest data-statistic baseline = {R['best_baseline']}")
        for k, s in R["scores"].items():
            extra = f" pairwise_raw={s['pairwise_raw']:.3f}" if "pairwise_raw" in s else ""
            print(f"    {k:22s} MAE={s['mae']:.4f} [{s['mae_80'][0]:.4f},{s['mae_80'][1]:.4f}]  "
                  f"vs best base={s['ratio_best_base']:.2f} [{s['ratio_best_base_95'][0]:.2f},{s['ratio_best_base_95'][1]:.2f}]95  "
                  f"rho={s['spearman']:.2f} pairwise={s['pairwise']:.3f} [{s['pairwise_80'][0]:.2f},{s['pairwise_80'][1]:.2f}]{extra}")


# ------------------------------------------------------------------------------------------------- stand-in statistics
def standin(kind, targets, recipe_paths, path, seed=0):
    """Plausible fake merged.json. 'oracle': each eval set's curve amplitude is solved so that the information rule
    returns its measured rate exactly (positive control: info_rule must score Spearman 1 and rms ~0 on task A). 'random': independent statistics."""
    rng = np.random.default_rng(seed)
    rates = pd.read_csv(os.path.join(targets, "ladder_rates.csv"))
    tgt = rates[rates.in_dist != "no"].groupby("eval")["alpha_large"].median()
    allr = rates.groupby("eval")["alpha_large"].median()
    rp = json.load(open(recipe_paths))["recipes"]
    ns = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512])
    src = {}

    lv = pd.read_csv(os.path.join(targets, "final_levels.csv"))
    D_mid = float(np.exp(np.log(lv.groupby("params")["tokens"].median().loc[LARGE].values).mean()))

    def make(name, kind_, g, alpha=None):
        bpt = rng.uniform(3.2, 4.7)
        hinf, a = rng.uniform(0.6, 1.0), rng.uniform(1.5, 2.5)
        if alpha is not None:  # amplitude chosen so that the information rule returns exactly this alpha
            n = neff(D_mid * bpt, bpt, np.log(TS["n0"]), 1.0 / (1.0 + g))
            u = alpha * (1 + g) * hinf / (g - alpha * (1 + g))
            a = u * n ** g
        nb = ns * bpt
        h = hinf + a * nb ** -g
        rows = [dict(n_tokens=int(n), bits_per_byte=float(v), context_bytes_exact_mean=float(x)) for n, v, x in zip(ns, h, nb)]
        return dict(kind=kind_, name=name, bytes_per_token=bpt, compression=dict(lzma_bpb_1MB=rng.uniform(1.6, 2.7)),
                    H_cond=[rng.uniform(9, 11), rng.uniform(4, 6)], unigram_bits=rng.uniform(9.8, 11), zipf_s=rng.uniform(0.9, 1.2),
                    beta=dict(beta=dict(tok_1_200=rng.uniform(0.3, 0.8))),
                    ref={PRIMARY: dict(paired=dict(rows=rows), fit_paired=dict(h_inf=hinf, a=a, g=g),
                                       gamma={GAMMA_WIN: dict(gamma=g)})})

    for e in allr.index:
        g = rng.uniform(0.3, 0.9)
        src[f"ev_{e}"] = make(e, "eval", g, tgt.get(e, allr[e]) if kind == "oracle" else None)
    for r in rp:
        src["rc_" + r] = make(r, "recipe", rng.uniform(0.2, 0.9))
    keys = list(src)
    X = {k: {t: (8.5 if k == t else rng.uniform(9, 13)) for t in keys} for k in keys}
    json.dump(dict(sources=src, xent=dict(xent=X)), open(path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", default=os.path.join(ROOT, "results/datadecide/stats/merged.json"))
    ap.add_argument("--targets", default=os.path.join(ROOT, "results/datadecide"))
    ap.add_argument("--recipe_paths", default=os.path.join(ROOT, "data/datadecide/recipe_paths.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/datadecide/predict.json"))
    ap.add_argument("--standin", choices=["oracle", "random"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", default="ABCDE")
    a = ap.parse_args()
    if a.standin:
        a.stats = a.out.replace(".json", f"_standin_{a.standin}_stats.json")
        standin(a.standin, a.targets, a.recipe_paths, a.stats, a.seed)
    res = run(a.stats, a.targets, a.recipe_paths, np.random.default_rng(a.seed), a.tasks)
    res["inputs"] = dict(stats=a.stats, stats_partial=not os.path.exists(a.stats), targets=a.targets, standin=a.standin,
                         primary_ref=PRIMARY, gamma_window=GAMMA_WIN, ts_constants=TS)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    for task, r in res.items():
        if task == "inputs":
            continue
        if task == "E_heldout_family":
            print_heldout(r)
            continue
        print(f"\n{task}")
        rows = r["scores"] if "scores" in r else r["summary"]
        for k, s in rows.items():
            print(f"  {k:30s} " + " ".join(f"{m}={v:.3f}" if isinstance(v, float) else f"{m}={v}"
                                           for m, v in s.items() if m in ("n", "spearman", "pairwise", "rms",
                                                                          "median_spearman", "median_pairwise")))


if __name__ == "__main__":
    main()
