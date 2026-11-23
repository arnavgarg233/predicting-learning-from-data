"""Evaluation-aware rivals to the bigram statistic on DataDecide (2026-10-17).

The paper's statistic is the bigram cross-entropy of the 11 evaluation sets under each recipe's sample; its reference
term is the same for every recipe, so rankings use the cross-entropy alone. The fair rivals also see the evaluation text:
token-unigram and word-unigram cross-entropy of each evaluation set under each recipe's sample (add-0.5 smoothing),
fitted on the same first 90% of each recipe sample as the bigram and scored on the whole evaluation set, in bits per
byte, averaged over the 11 sets. Each statistic goes through the same leave-one-family-out affine map as the bigram row
(task E protocol, target 1B bits per byte over all 11 sets), scored by pairwise accuracy over the 300 recipe pairs, with
a paired family bootstrap (2,000 draws) on each difference from the bigram.
  python scripts/datadecide_evalaware_rivals.py  ->  results/datadecide/evalaware_rivals.json"""
import json, os, re, sys
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import datadecide_predict as P  # noqa: E402
import datadecide_stats as dds  # noqa: E402

ROOT = P.ROOT
OUT = os.path.join(ROOT, "results/datadecide/evalaware_rivals.json")
W = re.compile(r"\w+")
EV = dds.slugs("eval")
bpt = {e: json.load(open(os.path.join(ROOT, f"results/datadecide/stats/{e}_cpu.json")))["bytes_per_token"] for e in EV}
tok = dds.tokenizer()


def first90(a):
    return a[: int(0.9 * len(a))]


def word_counts(a):
    c = Counter()
    for t in dds.texts(a, tok):
        c.update(W.findall(t.lower()))
    return c


cache = OUT.replace(".json", "_cache.json")
C = json.load(open(cache)) if os.path.exists(cache) else {}
ev_tok = {e: np.fromfile(dds.DD / "evaldata" / f"{e[3:]}.npy", dtype="<u2") for e in EV}
ev_words = {e: word_counts(ev_tok[e]) for e in EV}
ev_bytes = {e: float(len(ev_tok[e][ev_tok[e] != dds.EOS]) * bpt[e]) for e in EV}
for s in dds.slugs("recipe"):
    if s in C:
        continue
    a = first90(dds.load(s))
    u = np.bincount(a[a != dds.EOS], minlength=dds.V).astype(float) + 0.5
    lp = np.log2(u / u.sum())
    tok_ce = {e: float(-lp[t[t != dds.EOS]].mean() / bpt[e]) for e, t in ev_tok.items()}
    wc = word_counts(a)
    vocab = set(wc) | set().union(*[set(v) for v in ev_words.values()])
    Z = sum(wc.values()) + 0.5 * len(vocab)
    word_ce = {}
    for e, c in ev_words.items():
        bits = -sum(n * np.log2((wc.get(w, 0) + 0.5) / Z) for w, n in c.items())
        word_ce[e] = float(bits / ev_bytes[e])
    C[s] = dict(tok=tok_ce, word=word_ce)
    json.dump(C, open(cache, "w"))
    print(s, round(np.mean(list(tok_ce.values())), 4), round(np.mean(list(word_ce.values())), 4), flush=True)

M = P.load_stats(os.path.join(ROOT, "results/datadecide/stats/merged.json"))
S, X = M["sources"], (M.get("xent") or {}).get("xent")
import pandas as pd  # noqa: E402
lv = pd.read_csv(os.path.join(ROOT, "results/datadecide/final_levels.csv"))
fam_of = {r: f for f, (_, rs) in P.FAMILIES.items() for r in rs}
names = sorted(fam_of)
fam = np.array([fam_of[r] for r in names])
F, _ = P.recipe_features(S, X, names)
T, _ = P.family_targets(lv, np.random.default_rng(0))
y = T["all11"].loc[names, "level_1B"].values.astype(float)
slug_of = {s_["name"]: k for k, s_ in S.items() if s_.get("kind") == "recipe"}
feat = {"bigram (paper)": F.loc[names, "mismatch_all"].values.astype(float),
        "token unigram": np.array([np.mean(list(C[slug_of[r]]["tok"].values())) for r in names]),
        "word unigram": np.array([np.mean(list(C[slug_of[r]]["word"].values())) for r in names])}
pred = {k: P.lofo(v[:, None], y, fam, [(0,)])[0] for k, v in feat.items()}
i, j = np.triu_indices(len(y), 1)


def agree(p, a, b):
    dp = np.sign(p[a] - p[b])
    return np.where(dp == 0, 0.5, dp == np.sign(y[a] - y[b]))


rng = np.random.default_rng(1)
u = np.unique(fam)
idx = {c: np.flatnonzero(fam == c) for c in u}
boot = {k: [] for k in pred}
for _ in range(2000):
    k = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
    ii, jj = np.triu_indices(len(k), 1)
    a, b = k[ii], k[jj]
    keep = a != b
    for name, p in pred.items():
        boot[name].append(agree(p, a[keep], b[keep]).mean())
q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
out = dict(n_recipes=len(y), protocol="task E leave-one-family-out affine, target level_1B all11, pairwise over 300 pairs",
           rows={})
for name, p in pred.items():
    out["rows"][name] = dict(pairwise=float(agree(p, i, j).mean()), pairwise_95=q(boot[name]),
                             mae=float(np.mean(np.abs(p - y))),
                             spearman=float(pd.Series(p).rank().corr(pd.Series(y).rank())))
    if name != "bigram (paper)":
        d = np.array(boot["bigram (paper)"]) - np.array(boot[name])
        out["rows"][name]["bigram_minus_this"] = dict(point=out["rows"]["bigram (paper)"]["pairwise"] - out["rows"][name]["pairwise"], ci95=q(d))
out["raw_feature_spearman_with_target"] = {k: float(pd.Series(v).rank().corr(pd.Series(y).rank())) for k, v in feat.items()}
json.dump(out, open(OUT, "w"), indent=1)
print(json.dumps(out, indent=1))
