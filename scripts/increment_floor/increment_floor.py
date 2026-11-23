"""increment_floor (2026-11-05): the curve's increment under each floor convention.
Recompute the infocurve_increment headline numbers under the three information-floor conventions of sensitivity:
  h960 (paper), zero (no floor), free (fitted floor). Stored outputs only, CPU, no training.
Specified after every outcome it uses was known (the variants are those of sensitivity, chosen after the forward tests).

x (the own fall S(2^23) - S(2^25)) per convention comes from scripts/sensitivity/sens.py chain(), which refits the whole
chain (K, level map) under that floor. Everything else (falls, n-gram, learner losses, fit and held-out corpora, the OLS
lines, the bootstrap and the partial Spearman) is scripts/reanalysis/infocurve_increment.py unchanged; only the
"own" column is replaced, and the own line is the OLS refit on the calibration corpora (for h960 that equals the frozen line).

    python3 scripts/increment_floor/increment_floor.py
        -> results/increment_floor/score.json
"""
import copy
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reanalysis"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "sensitivity"))
import infocurve_increment as II  # noqa: E402
import sens as S  # noqa: E402
import load as LD  # noqa: E402

OUT = os.path.join(ROOT, "results", "increment_floor", "score.json")
SETS = {"web": ({"external45"}, {"weblang30"}, "lang"),
        "english": ({"test80", "forward18", "external45"}, {"english24"}, "english")}


def x_table(rows, ch):
    return {(r["corpus"], r["set"], r["family"]): ch["per"][r["corpus"], r["set"]]["S"][23] - ch["per"][r["corpus"], r["set"]]["S"][25]
            for r in rows}


def lookup(xt, corpus, sets, fam):
    hit = [v for (c, s, f), v in xt.items() if c == corpus and s in sets and f == fam]
    assert len(hit) == 1, (corpus, sets, fam, len(hit))
    return hit[0]


def with_x(fams, xt):
    out = {}
    for k, (cal, test, frozen, srms) in fams.items():
        cs, ts, fam = SETS[k]
        cal, test, srms = copy.deepcopy(cal), copy.deepcopy(test), dict(srms)
        for r in cal:
            r["own"] = lookup(xt, r["corpus"], cs, fam)
        for r in test:
            r["own"] = lookup(xt, r["corpus"], ts, fam)
        c = II.fit(cal, ["own"])
        p = II.X(test, ["own"]) @ c
        for r, v in zip(test, p):
            r["own_cal_frozen"] = float(v)  # the own line refitted on calibration under this convention
        srms["own_cal"] = II.CF.rms(p - np.array([r["fall"] for r in test]))
        out[k] = (cal, test, frozen, srms)
    return out


def pooled_partial_L23(F):
    """Pooled partial Spearman given L23, the same construction (and bootstrap draws) as infocurve_increment.pooled()."""
    rng = np.random.default_rng(0)
    nw, ne = F["web"]["n_heldout"], F["english"]["n_heldout"]
    idx = np.hstack([rng.integers(0, nw, (II.NB, nw)), nw + rng.integers(0, ne, (II.NB, ne))])
    y = np.concatenate([F[k]["_y"] for k in ("web", "english")])
    own = np.concatenate([F[k]["_own"] for k in ("web", "english")])
    base = np.concatenate([np.array([r["L23"] for r in F[k]["_test"]]) for k in ("web", "english")])

    def wr(v, n0):
        return np.concatenate([II.rank(v[:n0]) / (n0 - 1), II.rank(v[n0:]) / (len(v) - n0 - 1)])

    def pps(sel):
        n0 = int(np.sum(sel < nw))
        Z = np.column_stack([np.ones(len(sel)), np.r_[np.zeros(n0), np.ones(len(sel) - n0)], wr(base[sel], n0)])
        return II.pcorr(wr(y[sel], n0), wr(own[sel], n0), Z)

    return dict(rho=pps(np.arange(nw + ne)), ci95=II.ci([pps(np.sort(i)) for i in idx[:2000]]))


def headline(F, P):
    def R(v):
        return dict(ratio=v["ratio"], ci95=v["ci95"], label=v["label"], closer=v["closer"])
    out = {}
    for k in ("web", "english"):
        f = F[k]
        out[k] = dict(
            strongest_simple=f["strongest_simple"],
            base_plus_own_vs_base=R(f["primary_increment"]),
            own_plus_ngram_vs_ngram=R(f["increment"]["ngram23+own / ngram23"]),
            partial_spearman_given_ngram23=f["partial_spearman_own_given"]["ngram23"],
            partial_spearman_given_L23=f["partial_spearman_own_given"]["L23"],
            combined=dict(rms=f["heldout_rms"]["ngram23+own"], rms_ngram23=f["heldout_rms"]["ngram23"], rms_own=f["heldout_rms"]["own"],
                          coef=f["lines"]["ngram23+own"], vs_ngram23=R(f["combined"]["ngram23+own / ngram23"]),
                          vs_own=R(f["combined"]["ngram23+own / own"]), vs_kind_mean=R(f["combined"]["ngram23+own / kind_mean"])),
            own_line=dict(coef=f["lines"]["own"], rms=f["heldout_rms"]["own"], vs_kind_mean=R(f["head_to_head"]["own / kind_mean"])),
            spearman_own_vs_fall=f["spearman_own_vs_fall"], fit_loo_rms_own=f["fit_loo_rms"]["own"])
    out["pooled"] = dict(
        strongest_simple=P["strongest_simple_per_kind"],
        base_plus_own_vs_base=R(P["primary_increment"]),
        own_plus_ngram_vs_ngram=R(P["increment"]["ngram23+own / ngram23"]),
        partial_spearman_given_ngram23=P["partial_spearman_own_given"]["ngram23"],
        partial_spearman_given_strongest=P["partial_spearman_own_given"]["best"],
        partial_spearman_given_L23=P["partial_spearman_given_L23"],
        combined=dict(rms=P["heldout_rms"]["ngram23+own"], vs_ngram23=R(P["combined"]["ngram23+own / ngram23"]),
                      vs_own=R(P["combined"]["ngram23+own / own"])))
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fams = II.load()
    rows = LD.load()
    chains = {v: S.chain(rows, v) for v in S.VARIANTS}
    res, checks = {}, {}
    # paper convention exactly as infocurve_increment (stored x, frozen own line)
    F0 = {k: II.family(k, *v) for k, v in fams.items()}
    P0 = II.pooled(F0)
    P0["partial_spearman_given_L23"] = pooled_partial_L23(F0)
    res["h960_stored"] = headline(F0, P0)
    ref = json.load(open(os.path.join(ROOT, "results/infocurve_increment/increment.json")))
    checks["h960_stored_reproduces_increment_json"] = float(max(
        abs(ref["families"][k]["increment"]["ngram23+own / ngram23"]["ratio"] - F0[k]["increment"]["ngram23+own / ngram23"]["ratio"])
        for k in F0))
    for v in S.VARIANTS:
        xt = x_table(rows, chains[v])
        fv = with_x(fams, xt)
        if v == "h960":
            checks["h960_chain_x_vs_stored_max_abs"] = float(max(abs(a["own"] - b["own"]) for k in fams for part in (0, 1)
                                                                for a, b in zip(fv[k][part], fams[k][part])))
        F = {k: II.family(k, *fv[k]) for k in fv}
        P = II.pooled(F)
        P["partial_spearman_given_L23"] = pooled_partial_L23(F)
        res[v] = headline(F, P)
        res[v]["x_spread"] = {k: dict(cal_sd=float(np.std([r["own"] for r in fv[k][0]])),
                                      heldout_sd=float(np.std([r["own"] for r in fv[k][1]])),
                                      spearman_x_vs_h960_heldout=II.spearman([r["own"] for r in fv[k][1]], [r["own"] for r in fams[k][1]]))
                              for k in fv}
        print(v, "done", flush=True)
    out = dict(analysis="increment_floor", plan="infocurve_increment under the three floor conventions of sensitivity",
               doc="docs/analysis/increment_floor.md", script=os.path.relpath(__file__, ROOT),
               conventions=dict(h960="paper: h(n) - h(960) = A n^-gamma, 8-64 bytes (Methods fixed 2026-10-22)",
                                zero="no floor: h(n) = A n^-gamma, 8-64 bytes", free="fitted floor E over 8-960 bytes"),
               timing="h960 fixed in Methods 2026-10-22, before the English forward test was frozen 2026-11-01; zero and free "
                      "variants chosen 2026-11-05, after all outcomes were known.",
               checks=checks, results=res)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(json.dumps(checks))
    for v in ("h960_stored", "h960", "zero", "free"):
        for k in ("web", "english", "pooled"):
            h = res[v][k]
            f = lambda d: f"{d['ratio']:.3f} ({d['ci95'][0]:.3f}-{d['ci95'][1]:.3f}) {d['label']}"  # noqa: E731
            g = lambda d: f"{d['rho']:.2f} ({d['ci95'][0]:.2f}-{d['ci95'][1]:.2f})"  # noqa: E731
            print(f"{v:12s} {k:8s} base+own {f(h['base_plus_own_vs_base'])} | own+ng/ng {f(h['own_plus_ngram_vs_ngram'])} | "
                  f"ps|ng {g(h['partial_spearman_given_ngram23'])} ps|L23 {g(h['partial_spearman_given_L23'])} | "
                  f"comb rms {h['combined']['rms']:.4f} vs own {f(h['combined']['vs_own'])}")


if __name__ == "__main__":
    main()
