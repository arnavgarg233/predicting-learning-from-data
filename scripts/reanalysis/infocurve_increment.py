"""Does the own information curve predict the fall
L(2^23) - L(2^25) beyond simpler statistics on the two forward sets?
Deterministic, CPU, stored data only.

    python scripts/reanalysis/infocurve_increment.py   # -> results/infocurve_increment/increment.json

Calibration (fit) corpora and held-out corpora are those of the frozen own-curve lines (see compression_fall.py):
  web      fit: 29 external web languages        held out: 30 forward web languages
  english  fit: 25 clean of the paper's 29        held out: 20 clean forward English corpora
Every line is OLS on the fit corpora, applied unchanged to the held-out corpora.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compression_fall as CF  # noqa: E402  (loaders for ngram23/h0/gain; asserts the frozen lines)

ROOT = CF.ROOT
OUT = os.path.join(ROOT, "results", "infocurve_increment")
NB = 20000
SIMPLE = ["kind_mean", "ngram23", "h0", "gain", "ref", "L20", "L23", "early", "A3"]
DATA_ONLY = {"kind_mean", "ngram23", "h0", "gain", "ref"}
POOL_EXP = 1 - 2 ** (-0.28)


def J(p):
    return CF.J(p)


def rows_by(kind, sets):
    return {r["corpus"]: r for r in J("results/reanalysis/fall_diagnosis.json")["rows"] if r["kind"] == kind
            and r["set"] in sets}


A3_SVP = {(r["set"], r["corpus"]): r["A3_fall"] for r in J("results/reanalysis/shape_vs_pooled.json")["per_corpus"]}


def add_cal(rows, fd, sets):
    out = []
    for r in rows:
        f = fd[r["corpus"]]
        assert abs(f["fall"] - r["fall"]) < 1e-12 and abs(f["ngram23"] - r["ngram23"]) < 1e-12
        a3 = [A3_SVP[(s, r["corpus"])] for s in sets if (s, r["corpus"]) in A3_SVP]
        assert len(a3) == 1, r["corpus"]
        out.append(dict(r, ref=f["ref_full"], L20=f["L20"], L23=f["L23"], early=f["L20"] - f["L23"], A3=a3[0]))
    return out


def add_test(rows, score, fwd_dirs_key="dir"):
    out = []
    for r in rows:
        p = score["per_corpus"][r["corpus"]]
        fi = J(os.path.join(ROOT, p[fwd_dirs_key], "forecast_i.json"))["corpora"][r["corpus"]]
        assert abs(fi["ref_stats"]["ngram23"] - r["ngram23"]) < 1e-12
        out.append(dict(r, ref=fi["ref_stats"]["ref_full"], L20=p["L"]["20"], L23=p["L"]["23"], early=p["L"]["20"] - p["L"]["23"],
                        A3=p["pred_fall"]["A3"], own_raw=p["pred_fall"]["own_raw"], level_23=p["level_23"]))
    return out


def load():
    fams = {}
    cal, test, frozen, srms = CF.web()
    cal = add_cal(cal, rows_by("web", {"external45"}), ["external45"])
    fwd = rows_by("web", {"forward30"})
    test = add_test(test, J("results/fwd_weblang/score.json"))
    for r in test:  # cross-check against the fall_diagnosis copy of the forward rows
        f = fwd[r["corpus"]]
        assert abs(f["fall"] - r["fall"]) < 1e-12 and abs(f["L20"] - r["L20"]) < 1e-12 and abs(f["ref_full"] - r["ref"]) < 1e-5  # reruns differ at 1e-7
    fams["web"] = (cal, test, frozen, srms)
    cal, test, frozen, srms = CF.english()
    cal = add_cal(cal, rows_by("english", {"test80", "forward18", "external45"}), ["test80", "forward18", "external45"])
    test = add_test(test, J("results/fwd_english/score.json"))
    fams["english"] = (cal, test, frozen, srms)
    return fams


def X(rows, cols):
    return np.column_stack([np.ones(len(rows))] + [[r[c] for r in rows] for c in cols if c != "kind_mean"])


def fit(rows, cols):
    return np.linalg.lstsq(X(rows, cols), np.array([r["fall"] for r in rows]), rcond=None)[0]


def loo_rms(rows, cols):
    e = []
    for i in range(len(rows)):
        tr = rows[:i] + rows[i + 1:]
        e.append(float((X([rows[i]], cols) @ fit(tr, cols))[0]) - rows[i]["fall"])
    return CF.rms(e)


def rank(v):
    return np.argsort(np.argsort(v)).astype(float)


def resid(y, Z):
    return y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]


def pcorr(y, x, Z):
    ry, rx = resid(y, Z), resid(x, Z)
    d = np.sqrt((ry @ ry) * (rx @ rx))
    return float(ry @ rx / d) if d > 0 else float("nan")


def partial_spearman(y, x, covs):
    Z = np.column_stack([np.ones(len(y))] + [rank(c) for c in covs])
    return pcorr(rank(y), rank(x), Z)


def spearman(x, y):
    return CF.spearman(np.asarray(x), np.asarray(y))


def ratio_stats(ea, eb, idx):
    br = np.sqrt((ea[idx] ** 2).mean(1) / np.maximum((eb[idx] ** 2).mean(1), 1e-30))
    lo, hi = np.percentile(br, [2.5, 97.5])
    r = CF.rms(ea) / CF.rms(eb)
    return dict(ratio=float(r), ci95=[float(lo), float(hi)],
                label="REAL positive" if hi < 1 else ("REAL negative" if lo > 1 else "tie"),
                closer=f"{int(np.sum(np.abs(ea) < np.abs(eb)))}/{len(ea)}", bar_met=bool(r <= 0.80 and hi < 1))


def ci(vals):
    v = np.asarray(vals)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def family(name, cal, test, frozen, srms):
    n = len(test)
    y = np.array([r["fall"] for r in test])
    cal_loo = {s: loo_rms(cal, [s]) for s in SIMPLE}
    cal_loo_own = loo_rms(cal, ["own"])
    best = min(SIMPLE, key=lambda s: cal_loo[s])
    pairs = [list(p) for p in itertools.combinations([s for s in SIMPLE if s != "kind_mean"], 2)]
    pair_loo = {"+".join(p): loo_rms(cal, p) for p in pairs}
    best_pair = min(pair_loo, key=pair_loo.get).split("+")
    specs = {s: [s] for s in SIMPLE}
    specs["own"] = ["own"]
    for s in SIMPLE:
        if s != "kind_mean":
            specs[f"{s}+own"] = [s, "own"]
    specs["pair"] = best_pair
    specs["pair+own"] = best_pair + ["own"]
    coefs, pred = {}, {}
    for k, cols in specs.items():
        c = fit(cal, cols)
        coefs[k] = dict(cols=cols, a=float(c[0]), b=[float(v) for v in c[1:]])
        pred[k] = X(test, cols) @ c
    assert np.allclose(pred["own"], [r["own_cal_frozen"] for r in test], atol=1e-12)
    assert abs(CF.rms(pred["own"] - y) - srms["own_cal"]) < 1e-9 and abs(CF.rms(pred["kind_mean"] - y) - srms["kind_mean"]) < 1e-9
    pred["ctx_pool_exp_raw"] = np.array([r["level_23"] for r in test]) * POOL_EXP  # at the forecast level, as score.json
    pred["ctx_A3_raw"] = np.array([r["A3"] for r in test])
    pred["ctx_own_raw"] = np.array([r["own_raw"] for r in test])
    assert abs(CF.rms(pred["ctx_A3_raw"] - y) - srms["A3"]) < 1e-9
    assert "pool_exp" not in srms or abs(CF.rms(pred["ctx_pool_exp_raw"] - y) - srms["pool_exp"]) < 1e-9
    E = {k: v - y for k, v in pred.items()}
    idx = np.random.default_rng(0).integers(0, n, (NB, n))
    R = lambda a, b: ratio_stats(E[a], E[b], idx)  # noqa: E731
    base = best if best != "kind_mean" else None
    inc = {}
    for s in SIMPLE:
        if s != "kind_mean":
            inc[f"{s}+own / {s}"] = R(f"{s}+own", s)
    inc["pair+own / pair"] = R("pair+own", "pair")
    head = {f"own / {s}": R("own", s) for s in SIMPLE}
    head["own / pair"] = R("own", "pair")
    comb = {"ngram23+own / ngram23": R("ngram23+own", "ngram23"), "ngram23+own / own": R("ngram23+own", "own"),
            "ngram23+own / kind_mean": R("ngram23+own", "kind_mean")}
    # residual signal on the held-out corpora
    own = np.array([r["own"] for r in test])
    ps, rs = {}, {}
    for s in [x for x in SIMPLE if x != "kind_mean"] + ["pair"]:
        cols = specs[s]
        covs = [np.array([r[c] for r in test]) for c in cols]
        val = partial_spearman(y, own, covs)
        bs = [partial_spearman(y[i], own[i], [c[i] for c in covs]) for i in idx[:2000]]
        ps[s] = dict(rho=val, ci95=ci(bs))
        res = y - pred[s]
        rs[s] = dict(rho=spearman(own, res), ci95=ci([spearman(own[i], res[i]) for i in idx[:2000]]))
    ps_none = dict(rho=spearman(own, y), ci95=ci([spearman(own[i], y[i]) for i in idx[:2000]]))
    return dict(
        n_fit=len(cal), n_heldout=n, fit_corpora=[r["corpus"] for r in cal], heldout_corpora=[r["corpus"] for r in test],
        fit_loo_rms=dict(cal_loo, own=cal_loo_own), strongest_simple=best, strongest_simple_is_data_only=best in DATA_ONLY,
        best_pair=best_pair, best_pair_loo_rms=pair_loo["+".join(best_pair)],
        fit_loo_increment={s: loo_rms(cal, [s, "own"]) / cal_loo[s] for s in SIMPLE if s != "kind_mean"},
        lines=coefs, heldout_rms={k: CF.rms(v) for k, v in E.items()}, heldout_bias={k: float(v.mean()) for k, v in E.items()},
        primary_increment=inc[f"{best}+own / {best}"] if base else head["own / kind_mean"],
        increment=inc, head_to_head=head, combined=comb,
        partial_spearman_own_given=ps, spearman_own_vs_residual_of=rs, spearman_own_vs_fall=ps_none,
        heldout_rows=[dict(r, pred={k: float(v[i]) for k, v in pred.items()}) for i, r in enumerate(test)],
        _E=E, _y=y, _own=own, _test=test, _specs=specs)


def pooled(F):
    """Both held-out groups together, each corpus predicted by its own kind's lines; draws stratified by kind."""
    rng = np.random.default_rng(0)
    nw, ne = F["web"]["n_heldout"], F["english"]["n_heldout"]
    idx = np.hstack([rng.integers(0, nw, (NB, nw)), nw + rng.integers(0, ne, (NB, ne))])
    best = {k: F[k]["strongest_simple"] for k in F}

    def cat(key_fn):
        return np.concatenate([F[k]["_E"][key_fn(k)] for k in ("web", "english")])

    E = {"own": cat(lambda k: "own"), "best": cat(lambda k: best[k]), "best+own": cat(lambda k: f"{best[k]}+own"),
         "pair": cat(lambda k: "pair"), "pair+own": cat(lambda k: "pair+own"), "kind_mean": cat(lambda k: "kind_mean")}
    for s in SIMPLE:
        E[s] = cat(lambda k: s)
        if s != "kind_mean":
            E[f"{s}+own"] = cat(lambda k: f"{s}+own")
    R = lambda a, b: ratio_stats(E[a], E[b], idx)  # noqa: E731
    out = dict(n=nw + ne, strongest_simple_per_kind=best, heldout_rms={k: CF.rms(v) for k, v in E.items()},
               primary_increment=R("best+own", "best"), pair_increment=R("pair+own", "pair"), own_vs_best=R("own", "best"),
               own_vs_kind_mean=R("own", "kind_mean"),
               increment={f"{s}+own / {s}": R(f"{s}+own", s) for s in SIMPLE if s != "kind_mean"},
               combined={"ngram23+own / ngram23": R("ngram23+own", "ngram23"), "ngram23+own / own": R("ngram23+own", "own")})
    # pooled partial Spearman: within-kind ranks scaled to [0,1], kind indicator as covariate
    def wr(v, n0):
        return np.concatenate([rank(v[:n0]) / (n0 - 1), rank(v[n0:]) / (len(v) - n0 - 1)])

    def pps(y, own, base, sel):
        yy, oo, bb = y[sel], own[sel], base[sel]
        n0 = int(np.sum(sel < nw))
        kd = np.r_[np.zeros(n0), np.ones(len(sel) - n0)]
        Z = np.column_stack([np.ones(len(sel)), kd, wr(bb, n0)])
        return pcorr(wr(yy, n0), wr(oo, n0), Z)

    y = np.concatenate([F[k]["_y"] for k in ("web", "english")])
    own = np.concatenate([F[k]["_own"] for k in ("web", "english")])
    ps = {}
    for lab, keyf in (("best", lambda k: best[k]), ("ngram23", lambda k: "ngram23")):
        base = np.concatenate([np.array([r[keyf(k)] for r in F[k]["_test"]]) if keyf(k) != "kind_mean" else np.zeros(F[k]["n_heldout"])
                               for k in ("web", "english")])
        allidx = np.arange(nw + ne)
        bs = [pps(y, own, base, np.sort(i)) for i in idx[:2000]]
        ps[lab] = dict(rho=pps(y, own, base, allidx), ci95=ci(bs))
    out["partial_spearman_own_given"] = ps
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    fams = load()
    F = {k: family(k, *v) for k, v in fams.items()}
    P = pooled(F)
    clean = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for k, v in F.items()}
    out = dict(design="docs/analysis/infocurve_increment.md", script=os.path.relpath(__file__, ROOT),
               simple_predictors=SIMPLE, data_only=sorted(DATA_ONLY), families=clean, pooled=P,
               unavailable=["lzma (learner bytes not stored locally)", "n-gram fall (n-gram statistics only at 2^23 bytes)"])
    json.dump(out, open(os.path.join(OUT, "increment.json"), "w"), indent=1)
    for k, f in clean.items():
        print(f"== {k}: fit {f['n_fit']} held-out {f['n_heldout']} strongest simple {f['strongest_simple']} pair {f['best_pair']}")
        print("  fit LOO", {a: round(b, 4) for a, b in f["fit_loo_rms"].items()})
        print("  fit LOO increment", {a: round(b, 3) for a, b in f["fit_loo_increment"].items()})
        print("  held-out rms", {a: round(b, 4) for a, b in f["heldout_rms"].items()})
        for grp in ("increment", "head_to_head", "combined"):
            for a, v in f[grp].items():
                print(f"  {grp:12s} {a:24s} {v['ratio']:.3f} ({v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}) {v['label']} {v['closer']}")
        print("  partial spearman", {a: (round(v["rho"], 2), [round(c, 2) for c in v["ci95"]]) for a, v in f["partial_spearman_own_given"].items()})
        print("  resid spearman", {a: (round(v["rho"], 2), [round(c, 2) for c in v["ci95"]]) for a, v in f["spearman_own_vs_residual_of"].items()})
        print("  own~fall", f["spearman_own_vs_fall"])
    print("== pooled", P["strongest_simple_per_kind"])
    print("  rms", {a: round(b, 4) for a, b in P["heldout_rms"].items()})
    for a in ("primary_increment", "pair_increment", "own_vs_best", "own_vs_kind_mean"):
        v = P[a]; print(f"  {a:20s} {v['ratio']:.3f} ({v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}) {v['label']} {v['closer']}")
    for grp in ("increment", "combined"):
        for a, v in P[grp].items():
            print(f"  {grp:12s} {a:24s} {v['ratio']:.3f} ({v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}) {v['label']} {v['closer']}")
    print("  partial spearman", P["partial_spearman_own_given"])


if __name__ == "__main__":
    main()
