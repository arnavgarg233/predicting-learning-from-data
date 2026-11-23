"""Data exponents of the exact excess loss (test KL, floor exactly 0) in the Ising lab.

Fits log(excess) = c - alpha log(P) per (tag, basis) under four budget rules, with a seed
bootstrap (resample seeds within budget) for 95% CIs and the Z/X ratio. CPU, numpy only.
Usage: python scripts/lab_exponents.py
"""
import glob, json, collections
import numpy as np

RUNS = "results/runs"
B = 4000
rng = np.random.default_rng(0)
LOG2 = lambda p: int(round(np.log2(p)))

# (name, run dirs in priority order: first dir that has a (basis, budget) wins), late-window start (log2 P)
TAGS = [
    ("abs T64", ["crit_T64_v2", "crit_T64"], 16),
    ("abs T256", ["crit_T256_v2"], 17),
    ("RoPE T64", ["crit_T64_rope"], 14),
    ("RoPE T256", ["crit_T256_rope"], 15),
    ("abs T64 h=1.3", ["near_h13_T64"], 14),
]


def load(dirs):
    d = {}
    for t in dirs:
        for f in sorted(glob.glob(f"{RUNS}/{t}/*.json")):
            r = json.load(open(f))
            key = (r["basis"], r["P"])
            if any(k == key and src != t for k, (src, _) in d.items() if k == key):
                continue
            d.setdefault(key, (t, {}))[1][r["seed"]] = r["test_kl_bits"]
    out = collections.defaultdict(dict)
    for (b, p), (_, s) in d.items():
        out[b][p] = np.array([s[k] for k in sorted(s)])
    return out


def fit(ps, groups):
    x = np.concatenate([[np.log(p)] * len(g) for p, g in zip(ps, groups)])
    y = np.log(np.concatenate(groups))
    return -np.polyfit(x, y, 1)[0]


def boot(ps, groups):
    a = fit(ps, groups)
    bs = [fit(ps, [g[rng.integers(0, len(g), len(g))] for g in groups]) for _ in range(B)]
    return a, np.array(bs)


def escape(X):
    """Plateau rule: scan X local slopes from the smallest budget until one is >=0.5 (sharp drop); the last
    flat interval (slope <0.15) before that ends the plateau; fit starts one budget past its end.
    No flat interval: start at the smallest budget."""
    ps = sorted(X)
    m = [X[p].mean() for p in ps]
    s = [-np.log(m[i + 1] / m[i]) / np.log(ps[i + 1] / ps[i]) for i in range(len(ps) - 1)]
    last = None
    for i, v in enumerate(s):
        if v >= 0.5:
            break
        if v < 0.15:
            last = i
    return ps[0] if last is None else ps[last + 2]


def rules(X, w):
    e = escape(X)
    return {
        "all": lambda p: True,
        ">=2^14": lambda p: p >= 2 ** 14,
        f">=escape": lambda p, e=e: p >= e,
        "late window": lambda p: p >= 2 ** w,
    }, e


def main():
    src = json.load(open("results/source_stats.json"))
    rows = []
    for name, dirs, w in TAGS:
        D = load(dirs)
        if "X" not in D:
            continue
        rl, e = rules(D["X"], w)
        for rn, ok in rl.items():
            res = {}
            for b in "XZ":
                ps = [p for p in sorted(D[b]) if ok(p)]
                a, bs = boot(ps, [D[b][p] for p in ps])
                res[b] = (a, bs, ps)
            ratio = res["Z"][1] / res["X"][1]
            rows.append((name, rn, LOG2(res["X"][2][0]), len(res["X"][2]),
                         [len(D["X"][p]) for p in res["X"][2]], res, ratio, res["Z"][0] / res["X"][0], LOG2(e)))
    ci = lambda v: f"{np.percentile(v,2.5):.2f}-{np.percentile(v,97.5):.2f}"
    print("tag | rule | P_min | nbud | seeds | alphaX [CI] | alphaZ [CI] | Z/X [CI]")
    for name, rn, p0, nb, sd, res, ratio, r0, e in rows:
        print(f"{name} | {rn} (esc 2^{e}) | 2^{p0} | {nb} | {min(sd)}-{max(sd)} | "
              f"{res['X'][0]:.2f} [{ci(res['X'][1])}] | {res['Z'][0]:.2f} [{ci(res['Z'][1])}] | "
              f"{r0:.2f} [{ci(ratio)}]")
    for h in ("h=1.0 X", "h=1.0 Z"):
        g = src[h]["gamma"]
        print(f"source {h}: gamma={g:.3f}  gamma/(1+gamma)={g/(1+g):.3f}")
    # last-4-budget slope for X (item 3: recovery beyond plateau)
    print("last-4-budget alpha (X, Z):")
    for name, dirs, w in TAGS:
        D = load(dirs)
        out = []
        for b in "XZ":
            ps = sorted(D[b])[-4:]
            out.append(f"{fit(ps,[D[b][p] for p in ps]):.2f}")
        print(f"  {name}: 2^{LOG2(ps[0])}-2^{LOG2(ps[-1])} X {out[0]} Z {out[1]}")


if __name__ == "__main__":
    main()
