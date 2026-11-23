"""Readout for the high-power tokenizer exponent test (tag v5: 4 tokenizers x log2 bytes 21-24 x seeds 0-4, held-out test split).
Per tokenizer and floor (0.12, 0.25, 0.38, plus a free floor profiled over the 4 budgets): exponent of (bpb - floor)
vs training bytes, seed bootstrap (resample seeds within each budget), bytes/BPE exponent ratio with 95% CI, and the
ratio against Cagnetta's prediction beta_BPE / beta_bytes (token window 0.88/1.28 ~ 0.70; byte window from text/beta_results.json).
Usage: python scripts/score_v5.py [--tags v5] [--ks 21,22,23,24] [--metric bits_per_byte] [--boot 2000]
Pooled older tags work too, e.g. --tags v1,v1b (2 seeds: v1 has 2^22, 2^24; v1b has 2^21, 2^23)."""
import argparse, glob, json, math, os
import numpy as np

TOKS = ["bytes", "bpe512", "bpe2048", "bpe8192"]
ap = argparse.ArgumentParser()
ap.add_argument("--tags", default="v5")
ap.add_argument("--ks", default="21,22,23,24")
ap.add_argument("--metric", default="bits_per_byte")
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()
KS = list(map(int, args.ks.split(",")))
rng = np.random.default_rng(0)


def load():
    R = {}
    for tag in args.tags.split(","):
        for f in glob.glob(f"results/text_runs/{tag}/*.json"):
            d = json.load(open(f))
            k = int(os.path.basename(f).split("_b")[1].split("_")[0])
            if k in KS and args.metric in d:
                R.setdefault(d["tok"], {}).setdefault(k, []).append(d[args.metric])
    return R


def slope(ys, fl):
    return -np.polyfit(np.array(KS) * math.log(2), np.log(np.array(ys) - fl), 1)[0]


def free_slope(ys):
    """Profile the floor on a grid in [0, min(y)): pick the floor with the smallest log-log residual."""
    x, ys = np.array(KS) * math.log(2), np.array(ys)
    fl = np.linspace(0, ys.min() * 0.999, 400)[:, None]
    ly = np.log(ys[None] - fl)
    xc = x - x.mean()
    b = (ly * xc).sum(1) / (xc ** 2).sum()
    sse = (((ly - ly.mean(1, keepdims=True)) - b[:, None] * xc) ** 2).sum(1)
    return -b[sse.argmin()]


def ci(a):
    return np.percentile(a, [2.5, 97.5])


R = load()
missing = [(t, k) for t in TOKS for k in KS if k not in R.get(t, {})]
assert not missing, f"missing cells: {missing}"
beta = json.load(open("text/beta_results.json")) if os.path.exists("text/beta_results.json") else {}
pred = {}
for t in TOKS[1:]:
    pred[t] = {"token-window": beta[t]["train"]["beta"]["tok_1_200"] / beta["bytes"]["train"]["beta"]["tok_1_200"],
               "byte-window": beta[t]["train"]["beta"]["byte_10_1000"] / beta["bytes"]["train"]["beta"]["byte_10_1000"]} if beta else {}

print(f"tags {args.tags}  metric {args.metric}  budgets {KS}  seeds per cell: "
      + " ".join(f"{t}:{min(len(R[t][k]) for k in KS)}-{max(len(R[t][k]) for k in KS)}" for t in TOKS))
for t in TOKS:
    print(f"  {t:8s} bpb " + " ".join(f"2^{k} {np.mean(R[t][k]):.4f}" for k in KS))
floors = [("0.12", lambda y: slope(y, 0.12)), ("0.25", lambda y: slope(y, 0.25)),
          ("0.38", lambda y: slope(y, 0.38)), ("free", free_slope)]
print("Cagnetta ratio beta_BPE/beta_bytes (predicted alpha_bytes/alpha_BPE): token-window 0.70 (nominal), measured "
      + "; ".join(f"{t[3:]}: tok {pred[t]['token-window']:.2f}, byte {pred[t]['byte-window']:.2f}" for t in TOKS[1:]) if beta else "")
for name, fn in floors:
    a = {t: fn([np.mean(R[t][k]) for k in KS]) for t in TOKS}
    B = {t: [] for t in TOKS}
    for _ in range(args.boot):
        for t in TOKS:
            B[t].append(fn([np.mean(rng.choice(R[t][k], len(R[t][k]))) for k in KS]))
    B = {t: np.array(B[t]) for t in TOKS}
    print(f"== floor {name}")
    for t in TOKS:
        lo, hi = ci(B[t])
        print(f"  alpha {t:8s} {a[t]:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    for t in TOKS[1:]:
        r = B["bytes"] / B[t]
        lo, hi = ci(r)
        line = f"  bytes/{t[3:]:5s} ratio {a['bytes']/a[t]:.2f}  95% CI [{lo:.2f}, {hi:.2f}]"
        for label, p in [("0.70", 0.70)] + [(f"{k} {v:.2f}", v) for k, v in pred[t].items()]:
            line += f" | vs {label}: {'excluded' if not lo <= p <= hi else 'inside'}"
        print(line)
