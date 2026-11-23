"""Analysis of the single-pass rate study: does the single-pass data-scaling exponent follow the information curve?

    python3 scripts/rate_singlepass/analyze.py                 # results/rate_singlepass/{stats.json,runs/} -> analysis.json
    python3 scripts/rate_singlepass/analyze.py --selftest      # synthetic curves with a planted law, and the reproduction check

Inputs: stats.json (the Mac copy written before training: per corpus gamma_ref, gamma_ng25, gamma_ng23, lzma_bpb, ref_full,
ngram23, c) and runs/rs_<corpus>/bytes_b<k>_s<s>.json (test bits per byte). L(k) is the mean over the seeds present.
A corpus enters when it has every budget 2^22..2^27.

Rate exponent per corpus, a = -d log(L - E) / d log D (least squares over the budgets used), for the floors E:
  raw            E = 0, all six budgets (and 'raw 24-27', the four largest)
  frac f         E = f * min_k L(k), f = 0.5, 0.7, 0.85, 0.95 (the DataDecide floor sensitivity of the paper)
  ref k, ng k    E = k * ref_full and E = k * ngram23, k = 0.5, 0.8 (scripts/reanalysis/rate_law_fixed_size_b.py), E <= 0.95 min L
  fit3           E free in [0, 0.98 min L], least squares of log(L - E) on a grid (as rate_law_fixed_size_b.fit3)
Reported per rate and per set (all, language, english):
  Spearman of the rate with gamma_ref, gamma_ng25, gamma_ng23 and lzma bits per byte, 95% corpus-bootstrap intervals
  (10,000 resamples) and one-sided permutation p (20,000; sign of the prediction: + for gamma, - for lzma);
  partial Spearman holding the loss level log L(2^27) fixed (rank residuals), with bootstrap intervals;
  the law's absolute prediction gamma_ref/(1+gamma_ref) against the measured exponent: bias, r.m.s. and mean absolute error,
  median ratio measured/law.
Seed noise: |L(s0) - L(s1)| at 2^26 and 2^27 and the exponent's spread when s0-only and s1-only top budgets are used.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "rate_singlepass")
LN2 = np.log(2)
BUDGETS = (22, 23, 24, 25, 26, 27)
PRED = {"gamma_ref": 1, "gamma_ng25": 1, "gamma_ng23": 1, "lzma_bpb": -1}


def slope(ks, L, E=0.0):
    ks, L = np.asarray(ks, float), np.asarray(L, float)
    return float(-np.polyfit(ks * LN2, np.log(L - E), 1)[0])


def fit3(ks, L):
    ks, L = np.asarray(ks, float), np.asarray(L, float)
    best = None
    for E in np.linspace(0, 0.98 * L.min(), 99):
        y = np.log(L - E)
        A = np.vstack([np.ones(len(ks)), -ks * LN2]).T
        c = np.linalg.lstsq(A, y, rcond=None)[0]
        sse = float(np.sum((A @ c - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, E, c[1])
    return float(best[2]), float(best[1])


def rates(r, ks=BUDGETS):
    L = np.array([r["L"][k] for k in ks])
    m = L.min()
    out = {"raw": slope(ks, L)}
    top = [k for k in ks if k >= ks[-1] - 3]
    if len(ks) > 4:
        out[f"raw {top[0]}-{top[-1]}"] = slope(top, [r["L"][k] for k in top])
    for f in (0.5, 0.7, 0.85, 0.95):
        out[f"frac {f}"] = slope(ks, L, f * m)
    for k in (0.5, 0.8):
        out[f"ref {k}"] = slope(ks, L, min(k * r["ref_full"], 0.95 * m))
        out[f"ng {k}"] = slope(ks, L, min(k * r["ngram23"], 0.95 * m))
    out["fit3"], r["fit3_E"] = fit3(ks, L)
    return out


def _rows_pearson(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def _rows_resid(a, z):
    """Residuals of each row of a on the matching row of z (least-squares line)."""
    zc = z - z.mean(1, keepdims=True)
    ac = a - a.mean(1, keepdims=True)
    return ac - zc * ((ac * zc).sum(1) / (zc * zc).sum(1))[:, None]


def rank_partial(x, y, z):
    rx, ry, rz = (stats.rankdata(v)[None] for v in (x, y, z))
    return float(_rows_pearson(_rows_resid(rx, rz), _rows_resid(ry, rz))[0])


def assoc(x, y, z, sign, rng, B=10000, P=20000):
    x, y, z = map(lambda v: np.asarray(v, float), (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    n = len(x)
    if n < 5:
        return dict(n=n)
    rho = float(stats.spearmanr(x, y)[0])
    part = rank_partial(x, y, z)
    i = rng.integers(0, n, (B, n))
    i = i[np.array([len(np.unique(r)) >= 4 for r in i])]
    rx, ry, rz = (stats.rankdata(v[i], axis=1) for v in (x, y, z))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = _rows_pearson(rx, ry)
        bp = _rows_pearson(_rows_resid(rx, rz), _rows_resid(ry, rz))
    r0x, r0y = stats.rankdata(x), stats.rankdata(y)
    perm = _rows_pearson(np.tile(r0x, (P, 1)), rng.permuted(np.tile(r0y, (P, 1)), axis=1))
    p = float((np.sum(sign * perm >= sign * rho - 1e-12) + 1) / (P + 1))
    return dict(n=n, spearman=rho, ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], p_one_sided=p,
                partial_level=part, partial_ci=[float(np.nanpercentile(bp, 2.5)), float(np.nanpercentile(bp, 97.5))])


def law_error(g, a):
    g, a = np.asarray(g, float), np.asarray(a, float)
    law = g / (1 + g)
    d = a - law
    return dict(n=len(a), law_mean=float(law.mean()), measured_mean=float(a.mean()), bias=float(d.mean()),
                rms=float(np.sqrt(np.mean(d ** 2))), mae=float(np.mean(np.abs(d))), median_ratio=float(np.median(a / law)))


def load(out_dir, budgets=BUDGETS):
    S = json.load(open(os.path.join(out_dir, "stats.json")))
    rows = []
    for fam in ("language", "english"):
        for c in S[fam]["selected"]:
            st = c["stats"]
            Ls = {}
            for f in glob.glob(os.path.join(out_dir, "runs", f"rs_{c['name']}", "bytes_b*_s*.json")):
                d = json.load(open(f))
                k = int(os.path.basename(f).split("_b")[1].split("_")[0])
                Ls.setdefault(k, []).append(d["bits_per_byte"])
            row = dict(name=c["name"], family=fam, Ls=Ls, L={k: float(np.mean(v)) for k, v in Ls.items()},
                       **{k: st.get(k, float("nan")) for k in ("gamma_ref", "gamma_ng25", "gamma_ng23", "lzma_bpb", "ref_full", "ngram23", "c", "A_ref")})
            row["complete"] = all(k in row["L"] for k in budgets)
            rows.append(row)
    return rows


def analyse(rows, budgets=BUDGETS, rng_seed=0, B=10000, P=20000):
    rng = np.random.default_rng(rng_seed)
    rows = [r for r in rows if r["complete"]]
    for r in rows:
        r["rates"] = rates(r, budgets)
    res = dict(n_corpora=len(rows), budgets=list(budgets), per_corpus=[], sets={})
    for r in rows:
        res["per_corpus"].append({k: r[k] for k in ("name", "family", "gamma_ref", "gamma_ng25", "gamma_ng23", "lzma_bpb", "ref_full",
                                                     "ngram23", "c", "fit3_E")} | dict(L={str(k): v for k, v in r["L"].items()}, rates=r["rates"]))
    rate_names = list(rows[0]["rates"]) if rows else []
    for setname, sel in (("all", lambda r: True), ("language", lambda r: r["family"] == "language"),
                         ("english", lambda r: r["family"] == "english")):
        sub = [r for r in rows if sel(r)]
        if len(sub) < 5:
            continue
        lev = np.log([r["L"][budgets[-1]] for r in sub])
        d = {}
        for rn in rate_names:
            a = np.array([r["rates"][rn] for r in sub])
            d[rn] = {s: assoc([r[s] for r in sub], a, lev, sign, rng, B, P) for s, sign in PRED.items()}
            d[rn]["law"] = law_error([r["gamma_ref"] for r in sub], a)
            d[rn]["spearman_level"] = float(stats.spearmanr(lev, a)[0])
        d["gamma_ref vs lzma_bpb"] = float(stats.spearmanr([r["gamma_ref"] for r in sub], [r["lzma_bpb"] for r in sub])[0])
        d["gamma_ref vs gamma_ng25"] = float(stats.spearmanr([r["gamma_ref"] for r in sub], [r["gamma_ng25"] for r in sub], nan_policy="omit")[0])
        d["gamma_ref vs level"] = float(stats.spearmanr([r["gamma_ref"] for r in sub], lev)[0])
        res["sets"][setname] = d
    # seed noise
    noise = {}
    for k in budgets:
        dd = [abs(r["Ls"][k][0] - r["Ls"][k][1]) for r in rows if len(r["Ls"].get(k, [])) >= 2]
        if dd:
            noise[str(k)] = dict(n=len(dd), median_abs_diff=float(np.median(dd)), max_abs_diff=float(np.max(dd)))
    spread = []
    for r in rows:
        if all(len(r["Ls"].get(k, [])) >= 2 for k in budgets[-2:]):
            e = []
            for s in (0, 1):
                L = {k: (r["Ls"][k][s] if k in budgets[-2:] else r["L"][k]) for k in budgets}
                e.append(slope(budgets, [L[k] for k in budgets]))
            spread.append(abs(e[0] - e[1]))
    if spread:
        noise["raw_exponent_seed_spread"] = dict(n=len(spread), median=float(np.median(spread)), max=float(np.max(spread)))
    res["seed_noise"] = noise
    return res


def summary(res, keys=("raw", "raw 24-27", "frac 0.5", "frac 0.7", "ref 0.8", "fit3")):
    print(f"n = {res['n_corpora']} corpora; budgets {res['budgets']}")
    for sn, d in res["sets"].items():
        print(f"== {sn}  (gamma_ref vs lzma {d['gamma_ref vs lzma_bpb']:+.2f}, vs gamma_ng25 {d['gamma_ref vs gamma_ng25']:+.2f}, vs level {d['gamma_ref vs level']:+.2f})")
        for rn in keys:
            if rn not in d:
                continue
            x = d[rn]
            g, z, ng, law = x["gamma_ref"], x["lzma_bpb"], x["gamma_ng25"], x["law"]
            f = lambda q: (f"{q['spearman']:+.2f} [{q['ci'][0]:+.2f},{q['ci'][1]:+.2f}] p={q['p_one_sided']:.3f} partial {q['partial_level']:+.2f}"
                           if "spearman" in q else "n<5")
            print(f"  {rn:10s} gamma {f(g)} | lzma {f(z)} | ng25 {f(ng)} | law bias {law['bias']:+.3f} rms {law['rms']:.3f} ratio {law['median_ratio']:.2f}")
    print("seed noise", json.dumps(res["seed_noise"]))


def selftest():
    rng = np.random.default_rng(1)
    # 1. planted law: L(D) = E + B D^-a with a = gamma/(1+gamma) + noise; gamma ~ U(0.3, 1.2); lzma anti-correlated with a
    rows = []
    for i in range(40):
        g = rng.uniform(0.3, 1.2)
        a = g / (1 + g) + rng.normal(0, 0.03)
        E = rng.uniform(0.6, 1.2)
        Bc = rng.uniform(0.5, 2.0) * (2.0 ** 22) ** a
        Ls = {k: [E + Bc * (2.0 ** k) ** -a + rng.normal(0, 0.003) for _ in range(2 if k >= 26 else 1)] for k in BUDGETS}
        rows.append(dict(name=f"s{i}", family="language" if i % 2 else "english", Ls=Ls, L={k: float(np.mean(v)) for k, v in Ls.items()},
                         gamma_ref=g, gamma_ng25=g + rng.normal(0, 0.3), gamma_ng23=g + rng.normal(0, 0.5), lzma_bpb=3 - 2 * a + rng.normal(0, 0.1),
                         ref_full=E * 0.9, ngram23=E * 1.6, c=0.56, A_ref=1.0, complete=True))
    res = analyse(rows, B=1000, P=2000)
    summary(res)
    fit = res["sets"]["all"]["fit3"]
    assert fit["gamma_ref"]["spearman"] > 0.6, fit["gamma_ref"]
    assert fit["lzma_bpb"]["spearman"] < -0.4, fit["lzma_bpb"]
    assert abs(fit["law"]["bias"]) < 0.08, fit["law"]
    # null: shuffled gamma gives an interval covering 0
    for r, g in zip(rows, rng.permutation([r["gamma_ref"] for r in rows])):
        r["gamma_ref"] = g
    null = analyse(rows, B=1000, P=2000)["sets"]["all"]["raw"]["gamma_ref"]
    assert null["ci"][0] < 0 < null["ci"][1] or abs(null["spearman"]) < 0.35, null
    print("synthetic: planted law recovered; shuffled null", round(null["spearman"], 2), [round(x, 2) for x in null["ci"]])
    # 2. reproduction on the 114 fixed-size corpora: slopes 2^20 -> 2^25 must give rate_law_fixed_size_b.json's numbers
    T = json.load(open(os.path.join(ROOT, "results/reanalysis/nl_table.json")))
    ref = json.load(open(os.path.join(ROOT, "results/reanalysis/rate_law_fixed_size_b.json")))["c<0.70 natural"]
    sub = [r for r in T if r["c"] < 0.70 and r["family"] != "code"]
    law = [r["gamma"] / (1 + r["gamma"]) for r in sub]
    raw = [slope([20, 25], [r["L20"], r["L25"]]) for r in sub]
    f8 = [slope([20, 25], [r["L20"], r["L25"]], min(0.8 * r["r"], 0.95 * r["L25"])) for r in sub]
    a, b = stats.spearmanr(law, raw)[0], stats.spearmanr(law, f8)[0]
    print(f"reproduction (51 corpora): raw {a:+.4f} vs {ref['raw 20-25']:+.4f}; floor 0.8*ref {b:+.4f} vs {ref['floor 0.8*ref_full']:+.4f}")
    assert abs(a - ref["raw 20-25"]) < 1e-9 and abs(b - ref["floor 0.8*ref_full"]) < 1e-9
    # 3. the full pipeline on those corpora's four budgets (19, 20, 23, 25), as a dry run of analyse()
    rows = [dict(name=r["corpus"], family="language" if r["family"] == "wiki" else "english",
                 Ls={19: [r["L19"]], 20: [r["L20"]], 23: [r["L23"]], 25: [r["L25"]]}, L={19: r["L19"], 20: r["L20"], 23: r["L23"], 25: r["L25"]},
                 gamma_ref=r["gamma"], gamma_ng25=float("nan"), gamma_ng23=float("nan"), lzma_bpb=-r["gamma"], ref_full=r["r"], ngram23=r["g"],
                 c=r["c"], A_ref=r["A"], complete=True) for r in sub]
    res = analyse(rows, budgets=(19, 20, 23, 25), B=500, P=500)
    print("dry run on the 51 fixed-size corpora (14-144 passes; budgets 19, 20, 23, 25):")
    summary(res, keys=("raw", "frac 0.5", "ref 0.8", "fit3"))
    print("SELFTEST OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dir", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    rows = load(a.dir)
    miss = [r["name"] for r in rows if not r["complete"]]
    res = analyse(rows)
    res["incomplete"] = miss
    json.dump(res, open(os.path.join(a.dir, "analysis.json"), "w"), indent=1, default=float)
    summary(res)
    if miss:
        print("incomplete (left out):", miss)


if __name__ == "__main__":
    main()
