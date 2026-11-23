"""sensitivity (2026-11-05), part B: sensitivity of the curve-forecast estimates to floor choices and to the stored
protocol settings. Stored outputs only, CPU, no training. Written after all the outcomes it uses were known.

    python3 scripts/sensitivity/sens.py   -> results/sensitivity/sens.json

Floor of the information curve (sets gamma, hence the shape S and the own fall):
  h960  paper: h(n) - h(960) = A n^-gamma over 8-64 bytes
  zero  no floor: h(n) = A n^-gamma over 8-64 bytes
  free  floor fitted: h(n) - E = A n^-gamma over 8-960 bytes, E on a grid in [0, 0.999 h(960)]
For each, the whole frozen chain is refitted the way it was built: K on the TinyStories bytes ladder, the level map on the
13 development corpora, the revision's shape scale and code bias on the 80 test corpora; then every score is recomputed.
Floor of the two-pilot power law (paper: none): reference E = k_r r and n-gram E = k_g g with k fitted on the development
emulation (stored in stronger_baselines.json), plus a fixed n-gram floor E = 0.5 g; E <= 0.9 L(2^20) as in the paper.
Learner rate (local exponent 2^23 -> 2^25, ladder slope 2^19-2^25): none, fitted (3-parameter fit on the four budgets),
reference loss r, and n-gram 0.5 g.
Protocol: seed 0 against seed 1 at 2^25; evaluation on the early-stopping split (valA) against the held-out test split;
1,000 against 20,000 reference targets.
"""
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import load as LD  # noqa: E402

C = LD.C
V = C.V
ROOT = LD.ROOT
sys.path.insert(0, os.path.join(ROOT, "scripts", "curve_forecast"))
import develop as DEV  # noqa: E402

OUT = os.path.join(ROOT, "results/sensitivity/sens.json")
NB = 20000
LN2 = np.log(2.0)
NGRAM = (-0.09618305026700236, 0.8134058678431364)       # development fit of the 2^23 level to g (shape-free)
LAM0, CODE_BIAS0 = 0.6924039665645343, 0.2011360393894477
EN_SET_ASIDE = ("patents", "billsum", "pile_dm_math", "pile_ubuntu_irc")
ORIG_POWER_FIT = V.power_fit


# ------------------------------------------------------------------ information-curve floor variants
def pf_zero(hr, window=V.WINDOW, n_ref=V.N_REF):
    n = np.geomspace(window[0], window[1], 12)
    p = np.polyfit(np.log(n), np.log(hr.h(n)), 1)
    return float(-p[0]), float(np.exp(p[1]))


def pf_free(hr, window=V.WINDOW, n_ref=V.N_REF):
    n = np.geomspace(8.0, 960.0, 16)
    h = hr.h(n)
    best = None
    for E in np.linspace(0.0, 0.999, 300) * float(hr.h(960.0)):
        y = np.log(h - E)
        p = np.polyfit(np.log(n), y, 1)
        s = float(np.sum((y - np.polyval(p, np.log(n))) ** 2))
        if best is None or s < best[0]:
            best = (s, float(-p[0]), float(np.exp(p[1])), float(E))
    PF_FREE_E[id(hr)] = best[3]
    return best[1], best[2]


PF_FREE_E = {}
VARIANTS = {"h960": ORIG_POWER_FIT, "zero": pf_zero, "free": pf_free}


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.mean(x ** 2)))


def boot_ratio(ea, eb, seed=0):
    ea, eb = np.asarray(ea, float), np.asarray(eb, float)
    n = len(ea)
    i = np.random.default_rng(seed).integers(0, n, (NB, n))
    r = np.sqrt((ea[i] ** 2).mean(1) / np.maximum((eb[i] ** 2).mean(1), 1e-30))
    lo, hi = np.percentile(r, [2.5, 97.5])
    return dict(ratio=rms(ea) / rms(eb), ci95=[float(lo), float(hi)], n=n)


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    rx, ry = np.argsort(np.argsort(x[ok])), np.argsort(np.argsort(y[ok]))
    return float(np.corrcoef(rx, ry)[0, 1]), int(ok.sum())


def spearman_ci(x, y, seed=0, B=4000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = spearman(x, y)[0]
    i = np.random.default_rng(seed).integers(0, len(x), (B, len(x)))
    bs = [spearman(x[j], y[j])[0] for j in i]
    return dict(rho=rho, ci95=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))], n=int(len(x)))


# ------------------------------------------------------------------ the frozen chain under one variant
def chain(rows, variant, hrs=None):
    """Refit K, level map, revision constants under the information-floor variant; return per-row forecasts."""
    V.power_fit = VARIANTS[variant]
    try:
        ts = V.ts_check()
        dev = DEV.load_dev()
        K = dict(K_ts=ts["K"])
        names = list(dev)
        coef = C.fit_level("both", DEV.level_rows(dev, names, "gamma", K, "both", "anch"))
        coef_ref = C.fit_level("ref", DEV.level_rows(dev, names, "gamma", K, "ref", "anch"))
        out = {}
        for r in rows:
            hr = hrs[r["corpus"], r["set"]] if hrs is not None else r["hr"]
            st = C.ref_stats(hr)
            st["ngram23"] = r["g"]
            nu, n0 = C.shape_params("gamma", st, K)
            S = {k: C.S(hr, nu, n0, k) for k in (19, 20, 23, 25)}
            if np.isfinite(r["g"]):
                a0 = {k: C.A0(hr, st, nu, n0, "both", "anch", coef, k) for k in (23, 25)}
            else:
                a0 = {k: C.A0(hr, st, nu, n0, "ref", "anch", coef_ref, k) for k in (23, 25)}
            out[r["corpus"], r["set"]] = dict(gamma=st["gamma"], A=st["A"], r=st["ref_full"], S=S, A0=a0,
                                              E_free=PF_FREE_E.get(id(hr)))
    finally:
        V.power_fit = ORIG_POWER_FIT
    return dict(K=ts["K"], gamma_ts=ts["gamma"], coef=coef, per=out)


def fit_revision(rows, F, L):
    """Shape scale and code bias, refitted on the 80 test corpora the way the revision was built."""
    t80 = [r for r in rows if r["set"] == "test80"]
    code = [r for r in t80 if r["family"] == "code"]
    bias = float(np.mean([F[r["corpus"], r["set"]]["A0"][23] - L(r)[23] for r in code]))
    s = np.array([F[r["corpus"], r["set"]]["S"][23] - F[r["corpus"], r["set"]]["S"][25] for r in t80])
    f = np.array([L(r)[23] - L(r)[25] for r in t80])
    lam_origin = float(np.sum(s * f) / np.sum(s * s))
    return dict(code_bias=bias, lam_ls_origin=lam_origin)


def revision(r, Fr, lam, bias):
    if r["family"] == "code":
        lv = Fr["A0"][23] - bias
    elif r["family"] == "english":
        lv = Fr["A0"][23]
    else:
        lv = NGRAM[0] + NGRAM[1] * r["g"]
    drop = lam * (Fr["S"][23] - Fr["S"][25])
    return {23: lv, 25: lv - drop}


# ------------------------------------------------------------------ two-pilot power law with floors
SB = json.load(open(os.path.join(ROOT, "results/reanalysis/stronger_baselines.json")))["constants"]


def a3(r, L, floor, rr):
    pil = {19: L[19], 20: L[20]}
    if floor == "none":
        return {k: C.A3(pil, k, "zero", 0.0) for k in (23, 25)}
    E = {"ref_dev": SB["k_r"] * rr, "ngram_dev": SB["k_g"] * r["g"], "ngram_0.5": 0.5 * r["g"]}[floor]
    if not np.isfinite(E):
        return {k: float("nan") for k in (23, 25)}
    return {k: C.A3(pil, k, "const", E) for k in (23, 25)}


A3_FLOORS = ("none", "ref_dev", "ngram_dev", "ngram_0.5")


# ------------------------------------------------------------------ scores
def scores(rows, F, L, lam, bias, tag=""):
    """Every headline number that the variant can move. L(r) gives the losses used as truth and pilots."""
    res = {}
    key = lambda r: (r["corpus"], r["set"])  # noqa: E731
    rows = [r for r in rows if key(r) in F]
    # Table 1: test80, data alone (A0) against two pilots, at 2^25
    t80 = [r for r in rows if r["set"] == "test80"]
    for fam in ("code", "english", "lang", "all"):
        sub = [r for r in t80 if fam == "all" or r["family"] == fam]
        if len(sub) < 3:
            continue
        e0 = [F[key(r)]["A0"][25] - L(r)[25] for r in sub]
        blk = dict(n=len(sub), rms_A0=rms(e0))
        for fl in A3_FLOORS:
            e3 = [a3(r, L(r), fl, F[key(r)]["r"])[25] - L(r)[25] for r in sub]
            if not np.all(np.isfinite(e3)):
                continue
            blk[f"rms_A3_{fl}"] = rms(e3)
            blk[f"A0_over_A3_{fl}"] = boot_ratio(e0, e3)
        res[f"table1_test80_{fam}"] = blk
    # forward18 code
    f18 = [r for r in rows if r["set"] == "forward18" and r["family"] == "code"]
    e0 = [F[key(r)]["A0"][25] - L(r)[25] for r in f18]
    e3 = [a3(r, L(r), "none", 0)[25] - L(r)[25] for r in f18]
    if len(f18) >= 3:
        res["forward18_code"] = dict(rms_A0=rms(e0), rms_A3=rms(e3), ratio=boot_ratio(e0, e3))
    # external45: revision against two pilots, by family, and the level / fall split on languages
    e45 = [r for r in rows if r["set"] == "external45"]
    for fam in ("code", "english", "lang", "all"):
        sub = [r for r in e45 if fam == "all" or r["family"] == fam]
        if len(sub) < 3:
            continue
        ev = [revision(r, F[key(r)], lam, bias)[25] - L(r)[25] for r in sub]
        blk = dict(n=len(sub), rms_rev=rms(ev), max_abs_rev=float(np.max(np.abs(ev))))
        for fl in A3_FLOORS:
            e3 = [a3(r, L(r), fl, F[key(r)]["r"])[25] - L(r)[25] for r in sub]
            if not np.all(np.isfinite(e3)):
                continue
            blk[f"rms_A3_{fl}"] = rms(e3)
            blk[f"rev_over_A3_{fl}"] = boot_ratio(ev, e3)
        if fam in ("lang", "all", "code"):
            fr = [(revision(r, F[key(r)], lam, bias)[23] - revision(r, F[key(r)], lam, bias)[25]) - (L(r)[23] - L(r)[25]) for r in sub]
            f3 = [(a3(r, L(r), "none", 0)[23] - a3(r, L(r), "none", 0)[25]) - (L(r)[23] - L(r)[25]) for r in sub]
            blk["fall_rms_rev"], blk["fall_rms_A3"] = rms(fr), rms(f3)
            blk["fall_ratio_rev_over_A3"] = boot_ratio(fr, f3)
        res[f"external45_{fam}"] = blk
    # own fall orders the falls of the 84 web and Wikipedia languages
    langs = [r for r in rows if r["family"] == "lang" and r["set"] in ("test80", "forward18", "external45")]
    own = [F[key(r)]["S"][23] - F[key(r)]["S"][25] for r in langs]
    fall = [L(r)[23] - L(r)[25] for r in langs]
    res["spearman_own_fall_84_languages"] = spearman_ci(own, fall)
    res["own_fall_spread_over_measured"] = float(np.std(own) / np.std(fall))
    # web-language fall line: calibrate on the 29 external languages, test on the 30 forward languages
    cal = [r for r in e45 if r["family"] == "lang"]
    wl = [r for r in rows if r["set"] == "weblang30"]
    if len(cal) < 5 or len(wl) < 5:
        return res
    x = np.array([F[key(r)]["S"][23] - F[key(r)]["S"][25] for r in cal])
    y = np.array([L(r)[23] - L(r)[25] for r in cal])
    b, a = np.polyfit(x, y, 1)
    km = float(y.mean())
    wl = [r for r in rows if r["set"] == "weblang30"]
    xo = np.array([F[key(r)]["S"][23] - F[key(r)]["S"][25] for r in wl])
    yo = np.array([L(r)[23] - L(r)[25] for r in wl])
    res["weblang30_fall_line"] = dict(a=float(a), b=float(b), kind_mean=km, rms_line=rms(a + b * xo - yo), rms_mean=rms(km - yo),
                                      ratio=boot_ratio(a + b * xo - yo, km - yo), spearman_own=spearman(xo, yo)[0])
    # English fall line: calibrate on the 25 clean English corpora, test on the 20 clean forward English corpora
    en_cal0 = [r for r in rows if r["family"] == "english" and r["set"] in ("test80", "forward18", "external45") and r["corpus"] not in EN_SET_ASIDE]
    en_cal = en_cal0
    if len([r for r in rows if r["set"] == "english24"]) < 5:
        return res
    x = np.array([F[key(r)]["S"][23] - F[key(r)]["S"][25] for r in en_cal])
    y = np.array([L(r)[23] - L(r)[25] for r in en_cal])
    b, a = np.polyfit(x, y, 1)
    km = float(y.mean())
    en = [r for r in rows if r["set"] == "english24" and r["corpus"] in EN_PRIMARY]
    xo = np.array([F[key(r)]["S"][23] - F[key(r)]["S"][25] for r in en])
    yo = np.array([L(r)[23] - L(r)[25] for r in en])
    res["english20_fall_line"] = dict(n_cal=len(en_cal), a=float(a), b=float(b), kind_mean=km, rms_line=rms(a + b * xo - yo),
                                      rms_mean=rms(km - yo), ratio=boot_ratio(a + b * xo - yo, km - yo), spearman_own=spearman(xo, yo)[0])
    return res


EN_PRIMARY = tuple(json.load(open(os.path.join(ROOT, "results/fwd_english/score.json")))["primary_clean"]["corpora"])


# ------------------------------------------------------------------ learner rates under floors
def free_floor(L, ks=(19, 20, 23, 25)):
    x = np.array(ks, float) * LN2
    y0 = np.array([L[k] for k in ks])
    best = None
    for E in np.linspace(0.0, 0.999, 500) * y0.min():
        y = np.log(y0 - E)
        p = np.polyfit(x, y, 1)
        s = float(np.sum((y - np.polyval(p, x)) ** 2))
        if best is None or s < best[0]:
            best = (s, float(E))
    return best[1]


def rates(rows, F):
    key = lambda r: (r["corpus"], r["set"])  # noqa: E731
    out = []
    for r in rows:
        L = r["L"]
        fl = dict(none=0.0, fitted=free_floor(L), reference=F[key(r)]["r"], ngram_half=0.5 * r["g"] if np.isfinite(r["g"]) else np.nan)
        row = dict(corpus=r["corpus"], set=r["set"], family=r["family"], gamma=F[key(r)]["gamma"], floors=fl, local={}, ladder={})
        for f, E in fl.items():
            if not np.isfinite(E) or E >= L[25]:
                row["local"][f] = row["ladder"][f] = float("nan")
                continue
            row["local"][f] = float(np.log2((L[23] - E) / (L[25] - E)) / 2.0)
            ks = np.array([19, 20, 23, 25], float)
            row["ladder"][f] = float(-np.polyfit(ks * LN2, np.log([L[int(k)] - E for k in ks]), 1)[0])
        out.append(row)
    return out


def rate_summary(rr, gam_by_variant):
    res = {}
    sets = ("test80", "forward18", "external45", "stage3")
    for fam in ("code", "english", "lang", "all"):
        sub = [r for r in rr if r["set"] in sets and (fam == "all" or r["family"] == fam)]
        blk = dict(n=len(sub))
        for kind in ("local", "ladder"):
            for f in ("none", "fitted", "reference", "ngram_half"):
                v = np.array([r[kind][f] for r in sub])
                ok = np.isfinite(v)
                d = dict(n_defined=int(ok.sum()), median=float(np.median(v[ok])) if ok.any() else None,
                         iqr=[float(np.percentile(v[ok], 25)), float(np.percentile(v[ok], 75))] if ok.any() else None)
                if ok.sum() > 3:
                    d["spearman_vs_none"] = spearman(v, [r[kind]["none"] for r in sub])[0]
                    for gv, g in gam_by_variant.items():
                        d[f"spearman_gamma_{gv}"] = spearman_ci([g[r["corpus"], r["set"]] for r in sub], v)
                blk[f"{kind}|{f}"] = d
        blk["n_reference_ge_L25"] = int(sum(1 for r in sub if not np.isfinite(r["local"]["reference"])))
        blk["n_fitted_floor_zero"] = int(sum(1 for r in sub if r["floors"]["fitted"] == 0.0))
        res[fam] = blk
    return res


# ------------------------------------------------------------------ protocol: 1,000 reference targets
def sub1000_hr(r):
    d = os.path.dirname(r["refpath"])
    raw = os.path.join(d, f"{r['ref_model']}_raw.npz")
    sub1000 = os.path.join(d, f"{r['ref_model']}_sub1000.npz")
    if not os.path.exists(raw) and not os.path.exists(sub1000):
        return None, None
    z = np.load(raw if os.path.exists(raw) else sub1000)
    Lb = z["loss"] / np.log(2.0)
    tb = z["tgtbytes"].astype(float)
    nt = len(tb)

    class H:
        pass

    def mk(ix):
        h = H()
        bpb = Lb[ix].sum(0) / tb[ix].sum()
        cb = z["ctxb"][ix].mean(0)
        o = np.argsort(cb)
        h.x, h.y = np.log(cb[o]), bpb[o]
        h.h = lambda n, h=h: np.interp(np.log(n), h.x, h.y)
        return h
    if not os.path.exists(raw):  # released subset (scripts/stronger_ref/extract_sub1000.py): same seeded 1,000 targets,
        return mk(np.arange(nt)), float("nan")  # but no full curve to check against
    idx = np.sort(np.random.default_rng(20261025).choice(nt, min(1000, nt), replace=False))
    full, sub = mk(np.arange(nt)), mk(idx)
    chk = float(np.max(np.abs(full.y - r["hr"].y))) if len(full.y) == len(r["hr"].y) else float("nan")
    return sub, chk


# ------------------------------------------------------------------ main
def main():
    rows = LD.load()
    key = lambda r: (r["corpus"], r["set"])  # noqa: E731
    Lmean = lambda r: r["L"]  # noqa: E731
    out = dict(n_rows=len(rows))

    chains = {v: chain(rows, v) for v in VARIANTS}
    # reproduction checks (paper variant)
    P = chains["h960"]["per"]
    lk = json.load(open(os.path.join(LD.CF, "locked.json")))["constants"]
    chk = dict(K=abs(chains["h960"]["K"] - lk["K_ts"]), coef=float(np.max(np.abs(np.array(chains["h960"]["coef"]) - lk["level_coef"]))))
    chk["A0_25_stored"] = float(max(abs(P[key(r)]["A0"][25] - r["stored"]["A0_25"]) for r in rows if "A0_25" in r["stored"]))
    chk["S_fall_stored"] = float(max(abs((P[key(r)]["S"][23] - P[key(r)]["S"][25]) - (r["stored"]["own_fall"] if "own_fall" in r["stored"]
                                    else r["stored"]["S23"] - r["stored"]["S25"])) for r in rows if "own_fall" in r["stored"] or "S23" in r["stored"]))
    chk["A3_25_stored"] = float(max(abs(a3(r, r["L"], "none", 0)[25] - r["stored"]["A3_25"]) for r in rows if "A3_25" in r["stored"]))
    rv = fit_revision(rows, P, Lmean)
    chk["code_bias"] = abs(rv["code_bias"] - CODE_BIAS0)
    chk["lam_ls_origin_vs_stored"] = rv["lam_ls_origin"] - LAM0
    chk["rev_25_stored"] = float(max(abs(revision(r, P[key(r)], LAM0, CODE_BIAS0)[25] - r["stored"]["rev_25"]) for r in rows if "rev_25" in r["stored"]))
    out["checks"] = chk
    print("checks", json.dumps(chk))

    # (a) information-curve floor variants: whole chain refitted
    out["info_floor"] = {}
    for v, ch in chains.items():
        rv = fit_revision(rows, ch["per"], Lmean)
        lam = LAM0 if v == "h960" else LAM0 * rv["lam_ls_origin"] / (LAM0 + chk["lam_ls_origin_vs_stored"])
        bias = CODE_BIAS0 if v == "h960" else rv["code_bias"]
        g = np.array([ch["per"][key(r)]["gamma"] for r in rows])
        out["info_floor"][v] = dict(K=ch["K"], gamma_ts=ch["gamma_ts"], level_coef=ch["coef"], lam=lam, code_bias=bias,
                                    gamma_median=float(np.median(g)), gamma_iqr=[float(np.percentile(g, 25)), float(np.percentile(g, 75))],
                                    spearman_gamma_vs_h960=spearman(g, [P[key(r)]["gamma"] for r in rows])[0],
                                    scores=scores(rows, ch["per"], Lmean, lam, bias))
        if v == "free":
            E = [ch["per"][key(r)]["E_free"] for r in rows]
            out["info_floor"][v]["free_E_over_h960_median"] = float(np.median([e / P[key(r)]["r"] for e, r in zip(E, rows)]))
            out["info_floor"][v]["n_free_E_zero"] = int(sum(e == 0 for e in E))
        print(v, "done")

    # (b) learner rates under floors
    rr = rates(rows, P)
    gam = {v: {key(r): chains[v]["per"][key(r)]["gamma"] for r in rows} for v in VARIANTS}
    out["learner_rates"] = rate_summary(rr, gam)
    out["learner_rates_per_corpus"] = rr

    # (c) protocol: seeds at 2^25
    two = [r for r in rows if "0" in r["L25_seeds"] and "1" in r["L25_seeds"]]
    d = np.array([r["L25_seeds"]["1"] - r["L25_seeds"]["0"] for r in two])
    out["seed"] = dict(n=len(two), sd_single_run=float(np.sqrt(np.mean(d ** 2) / 2)), max_abs_diff=float(np.max(np.abs(d))))
    for s in ("0", "1"):
        def Ls(r, s=s):
            L = dict(r["L"])
            if s in r["L25_seeds"] and len(r["L25_seeds"]) > 1:
                L[25] = r["L25_seeds"][s]
            return L
        out["seed"][f"s{s}_only"] = scores(rows, P, Ls, LAM0, CODE_BIAS0)
    print("seeds done")

    # (d) protocol: evaluation on the early-stopping split (valA) instead of the held-out test split
    def runs_of(r):
        return LD._runs(r["rundirs"])
    va, miss = {}, 0
    for r in rows:
        R = runs_of(r)
        try:
            v = {k: float(np.mean([R[(k, s)]["bits_per_byte_valA"] for s in (0, 1) if (k, s) in R and "bits_per_byte_valA" in R[(k, s)]]))
                 for k in (19, 20, 23, 25)}
            t = {k: float(np.mean([R[(k, s)]["bits_per_byte"] for s in (0, 1) if (k, s) in R])) for k in (19, 20, 23, 25)}
            # pilots are seed 0 only in the paper
            v[19], v[20] = R[(19, 0)]["bits_per_byte_valA"], R[(20, 0)]["bits_per_byte_valA"]
            if not all(np.isfinite(list(v.values()))):
                raise KeyError
            va[key(r)] = dict(valA=v, test=t)
        except (KeyError, TypeError):
            miss += 1
    have = [r for r in rows if key(r) in va]
    dv = {k: np.array([va[key(r)]["valA"][k] - va[key(r)]["test"][k] for r in have]) for k in (19, 20, 23, 25)}
    dfall = np.array([(va[key(r)]["valA"][23] - va[key(r)]["valA"][25]) - (va[key(r)]["test"][23] - va[key(r)]["test"][25]) for r in have])
    out["eval_split"] = dict(n=len(have), missing=miss,
                             valA_minus_test={k: dict(mean=float(v.mean()), sd=float(v.std()), max_abs=float(np.max(np.abs(v)))) for k, v in dv.items()},
                             fall_valA_minus_test=dict(mean=float(dfall.mean()), sd=float(dfall.std()), max_abs=float(np.max(np.abs(dfall)))))
    Pv = {key(r): P[key(r)] for r in have}
    out["eval_split"]["scores_valA"] = scores(have, Pv, lambda r: va[key(r)]["valA"], LAM0, CODE_BIAS0)
    out["eval_split"]["scores_test_same_rows"] = scores(have, Pv, lambda r: va[key(r)]["test"], LAM0, CODE_BIAS0)
    print("valA", len(have), miss)

    # (e) protocol: 1,000 against 20,000 reference targets (curve re-read from the stored per-target losses)
    hsub, chks = {}, []
    for r in rows:
        h, c = sub1000_hr(r)
        if h is not None:
            hsub[key(r)] = h
            chks.append(c)
    rows1k = [r for r in rows if key(r) in hsub]
    out["ref1000"] = dict(n=len(rows1k), full_reproduces_stored_max_abs=float(np.nanmax(chks)))
    ch1 = chain(rows1k, "h960", hrs=hsub)
    P1k = ch1["per"]
    out["ref1000"]["gamma_diff"] = dict(median_abs=float(np.median([abs(P1k[key(r)]["gamma"] - P[key(r)]["gamma"]) for r in rows1k])),
                                        max_abs=float(np.max([abs(P1k[key(r)]["gamma"] - P[key(r)]["gamma"]) for r in rows1k])))
    out["ref1000"]["own_fall_diff"] = dict(median_abs=float(np.median([abs((P1k[key(r)]["S"][23] - P1k[key(r)]["S"][25]) - (P[key(r)]["S"][23] - P[key(r)]["S"][25])) for r in rows1k])))
    out["ref1000"]["A0_25_diff"] = dict(median_abs=float(np.median([abs(P1k[key(r)]["A0"][25] - P[key(r)]["A0"][25]) for r in rows1k])))
    out["ref1000"]["scores_1000"] = scores(rows1k, P1k, Lmean, LAM0, CODE_BIAS0)
    out["ref1000"]["scores_20000_same_rows"] = scores(rows1k, {key(r): P[key(r)] for r in rows1k}, Lmean, LAM0, CODE_BIAS0)
    if True:
        out["ref1000"]["missing"] = sorted(f"{r['set']}/{r['corpus']}" for r in rows if key(r) not in hsub)
    print("ref1000", len(rows1k))

    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
