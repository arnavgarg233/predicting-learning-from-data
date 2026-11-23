"""Which data statistic predicts the learner's exponent across the exact-source zoo (landed runs only).

Reads results/zoo/analysis_landed.json (scripts/analyze_zoo.py --content_check results/zoo/content_check.json) and the
data-only prediction files results/zoo/pred/<name>.json, whose information curves h(n) are computed from the exact
sources (scripts/make_zoo.py). Nothing here is refitted from the runs except the linear maps of the leave-one-group-out
test, which are fitted on the training groups only.

Measured exponents (fit budgets 2^14..2^20, seed means, the same OLS as analyze_zoo.py):
  alpha       -dlog(excess KL)/dlog P over all positions (analyze_zoo's measured alpha)
  alpha_tail  copy positions for gated sources, all positions otherwise (analyze_zoo's tail series)
  nu_match    matched-context exponent: n_m(P) solves h(n_m) - h(T-1) = late-position excess (t >= T/2), interpolated
              in log n; nu_match = dlog n_m / dlog P over the fit budgets where n_m >= 1 is defined (at least 3 of 4).
              Undefined for gated_step, whose curve is a single step (n_m is pinned at 15-16). This is the quantity reported as the context exponent on the corpora.
  nu_learned  gated only: dlog(learned distances) / dlog P, distances counted as in analyze_zoo (KL below half its
              unlearned value)
Data statistics (exact source, no run read):
  gamma       Delta(n) power fit over n = 4..64 (stats.gamma_fit); info_formula = gamma/(1+gamma), nu_formula = 1/(1+gamma)
  gain_total  h(0) - h(T-1), bits; gain_rel = gain_total / h(0)
  gain_d8     (h(8) - h(16)) / h(8): relative gain of one context doubling at 8 -> 16 (the corpus statistic)
  gain_learn  (h(1) - h(n*)) / (h(1) log2 n*): mean relative gain per doubling over the learnable range 1..n*, with n* the
              information-rule horizon at 2^20 tokens
  p_info      information rule (K = 210 events = 111.5 bits per distance, window T) exponent; p_info_nu = slope of its
              predicted horizon
  beta        pair-correlation exponent (exact for gated power sources, fit otherwise); cag_formula = gamma/(2 beta);
              p_cag = Cagnetta rule exponent
Usage: python scripts/analyze_zoo_partial.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
FIT = [16384, 65536, 262144, 1048576]


def nn(v):
    return None if v is None or not np.isfinite(v) else float(v)


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.any(~np.isfinite(y)) or np.any(y <= 0):
        return np.nan
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def matched_context(h, excess):
    """n with h(n) - h(T-1) = excess, log-linear interpolation on the monotone envelope; nan outside the curve."""
    d = np.minimum.accumulate(np.asarray(h) - h[-1])
    if not (d[-2] < excess <= d[1]):  # needs at least one token of context and must not be below the curve's tail
        return np.nan
    n = np.arange(len(d))
    i = int(np.argmax(d < excess))  # first n with d(n) < excess; d(i-1) >= excess
    lo, hi = d[i - 1], d[i]
    if lo == hi:
        return float(i)
    f = (lo - excess) / (lo - hi)
    return float(np.exp(np.log(max(n[i - 1], 1)) + f * (np.log(n[i]) - np.log(max(n[i - 1], 1)))))


def stats_for(pr):
    h = np.array(pr["h_n"])
    st, ab = pr["stats"], pr["at_budgets"]
    T = pr["T"]
    g = st["gamma_fit"]
    bud = pr["budgets"]
    fi = [bud.index(P) for P in FIT]
    hor = [ab["info_horizon"][i] for i in fi]
    nstar = int(hor[-1])
    gl = (h[1] - h[nstar]) / (h[1] * np.log2(nstar)) if nstar >= 2 else np.nan
    beta = st["beta"]
    return dict(gamma=g, gamma_exact=st["gamma"], info_formula=g / (1 + g), nu_formula=1 / (1 + g),
                gain_total=float(h[0] - h[-1]), gain_rel=float((h[0] - h[-1]) / h[0]),
                gain_d8=float((h[8] - h[16]) / h[8]), gain_learn=nn(gl), n_star_2e20=nstar,
                p_info=pr["alpha"]["info"], p_info_nu=nn(slope(FIT, hor)), p_soft=pr["alpha"]["soft"],
                beta=beta, cag_formula=nn(g / (2 * beta)) if beta else None, p_cag=pr["alpha"]["cag"], T=T)


def measured_for(pr, ps):
    h = np.array(pr["h_n"])
    B = ps["budgets"]
    fi = [B.index(P) for P in FIT if P in B]
    me = ps["measured"]
    late = np.array(me["late"]["mean"])
    nm = np.array([matched_context(h, late[i]) for i in fi])
    out = dict(alpha=me["total"]["alpha"], alpha_ci=me["total"]["alpha_ci"], alpha_seeds=me["total"]["per_seed_alpha"],
               alpha_copy=me.get("copy", {}).get("alpha"), alpha_late=me["late"]["alpha"],
               n_match=[nn(v) for v in nm], seeds=len(me["total"]["seeds"]))
    ok = np.isfinite(nm)  # budgets where the excess lies on the curve (at small budgets it can exceed h(1) - h(T-1))
    out["nu_match"] = nn(slope(np.array([B[i] for i in fi])[ok], nm[ok])) if ok.sum() >= 3 and pr["shape"] != "step" else None
    out["nu_match_budgets"] = int(ok.sum())
    out["alpha_tail"] = out["alpha_copy"] if pr["family"] == "gated" else out["alpha"]
    if "horizons" in ps:
        L = {n: np.mean([r["learned"] for r in ps["horizons"] if r["nwin"] == n]) for n in sorted({r["nwin"] for r in ps["horizons"]})}
        E = {n: np.mean([r["eff_horizon"] for r in ps["horizons"] if r["nwin"] == n]) for n in L}
        Pn = [n * pr["T"] for n in L if n * pr["T"] in FIT]
        out["learned"] = {str(n * pr["T"]): float(v) for n, v in L.items()}
        out["nu_learned"] = nn(slope(Pn, [L[P // pr["T"]] for P in Pn]))
        out["nu_eff_horizon"] = nn(slope(Pn, [E[P // pr["T"]] for P in Pn]))
    return out


def lin_pred(xtr, ytr, xte):
    b, a = np.polyfit(xtr, ytr, 1)
    return a + b * np.asarray(xte)


def score(rows, stat, target, fitted):
    ok = [r for r in rows if r[stat] is not None and r[target] is not None]
    x, y = np.array([r[stat] for r in ok], float), np.array([r[target] for r in ok], float)
    out = dict(n=len(ok), spearman=spearman(x, y))
    if not fitted:  # a parameter-free prediction of the exponent itself
        e = x - y
        out.update(rms=float(np.sqrt(np.mean(e ** 2))), bias=float(np.mean(e)),
                   worst=sorted(((r["name"], round(float(v), 3)) for r, v in zip(ok, e)), key=lambda t: -abs(t[1]))[:4])
    return out


def logo(rows, stats, target, group):
    """Leave one group out: linear map stat -> target fitted on the other groups, rms on the held-out group; the constant
    baseline predicts the mean of the other groups. Parameter-free rules are also scored raw (no fit)."""
    groups = sorted({r[group] for r in rows})
    res = {}
    for s in stats + ["constant"]:
        errs, raw, per = [], [], {}
        for gname in groups:
            tr = [r for r in rows if r[group] != gname and r[target] is not None and (s == "constant" or r[s] is not None)]
            te = [r for r in rows if r[group] == gname and r[target] is not None and (s == "constant" or r[s] is not None)]
            if len(te) == 0 or len(tr) < 3:
                continue
            ytr, yte = np.array([r[target] for r in tr]), np.array([r[target] for r in te])
            if s == "constant":
                p = np.full(len(te), ytr.mean())
            else:
                xtr, xte = np.array([r[s] for r in tr]), np.array([r[s] for r in te])
                p = lin_pred(xtr, ytr, xte) if np.ptp(xtr) > 0 else np.full(len(te), ytr.mean())
                raw += list(xte - yte)
            errs += list(p - yte)
            per[gname] = float(np.sqrt(np.mean((p - yte) ** 2)))
        if errs:
            res[s] = dict(n=len(errs), rms=float(np.sqrt(np.mean(np.square(errs)))), per_group=per)
            if s in ("info_formula", "p_info", "p_soft", "nu_formula", "p_info_nu", "cag_formula", "p_cag") and raw:
                res[s]["raw_rms_no_fit"] = float(np.sqrt(np.mean(np.square(raw))))
    return res


def main():
    A = json.loads((ROOT / "results/zoo/analysis_landed.json").read_text())
    preds = {p.stem: json.loads(p.read_text()) for p in sorted((ROOT / "results/zoo/pred").glob("*.json"))}
    rows = []
    for name, ps in A["per_source"].items():
        pr = preds[name]
        r = dict(name=name, family=pr["family"], shape=pr["shape"], recoding=pr["recoding_of"] is not None,
                 length_control=pr["length_control_of"], **stats_for(pr), **measured_for(pr, ps))
        r["group"] = "hmm" if r["family"] == "hmm" else ("gated_power" if r["shape"] == "power" else "gated_other")
        rows.append(r)
    direct = [r for r in rows if not r["recoding"] and not r["length_control"]]
    stats = ["gamma", "info_formula", "gain_total", "gain_d8", "gain_learn", "p_info", "p_soft", "beta", "cag_formula", "p_cag"]
    rule_like = {"info_formula", "p_info", "p_soft", "cag_formula", "p_cag"}
    cross = {}
    for target in ("alpha", "alpha_tail", "nu_match"):
        st = stats if target != "nu_match" else ["gamma", "nu_formula", "gain_total", "gain_d8", "gain_learn", "p_info_nu", "beta"]
        rl = rule_like if target != "nu_match" else {"nu_formula", "p_info_nu"}
        cross[target] = dict(
            scores={s: score(direct, s, target, s not in rl) for s in st},
            leave_one_family_out=logo(direct, st, target, "family"),
            leave_one_group_out=logo(direct, st, target, "group"))
    # does gain per doubling explain what the information rule leaves over?
    add = {}
    for target, rule in (("alpha", "p_info"), ("alpha", "info_formula"), ("nu_match", "p_info_nu"), ("nu_match", "nu_formula")):
        ok = [r for r in direct if r[target] is not None and r[rule] is not None]
        res = [r[target] - r[rule] for r in ok]
        add[f"{target}_minus_{rule}"] = {g: dict(n=len(ok), spearman=spearman([r[g] for r in ok if r[g] is not None],
                                                                              [e for r, e in zip(ok, res) if r[g] is not None]))
                                         for g in ("gain_d8", "gain_learn", "gain_total")}
    # gated power sources only (the dial-like family, exact gamma)
    gp = [r for r in direct if r["group"] == "gated_power"]
    gpow = {t: {s: score(gp, s, t, s not in rule_like) for s in ("gamma", "info_formula", "p_info", "gain_d8", "gain_learn")}
            for t in ("alpha", "alpha_tail")}
    gpow["nu_learned"] = {s: score(gp, s, "nu_learned", s not in {"nu_formula", "p_info_nu"})
                          for s in ("gamma", "nu_formula", "p_info_nu", "gain_d8", "gain_learn")}
    gpow["nu_match"] = {s: score(gp, s, "nu_match", s not in {"nu_formula", "p_info_nu"})
                        for s in ("gamma", "nu_formula", "p_info_nu", "gain_d8", "gain_learn")}
    # is the gain statistic more than a family label or a re-expression of gamma?
    rng = np.random.default_rng(0)
    extra = {}
    for target in ("alpha", "nu_match"):
        ok = [r for r in direct if r[target] is not None and r["name"] != "gated_step"]
        y = [r[target] for r in ok]
        fam = np.array([r["family"] for r in ok])
        e = {"sd_measured": float(np.std(y, ddof=1)), "n": len(ok), "excluded": ["gated_step (step curve: no exponent)"]}
        for g in ("gain_d8", "gain_learn", "gain_total", "info_formula" if target == "alpha" else "nu_formula"):
            x = np.array([r[g] for r in ok], float)
            r0 = spearman(x, y)
            cnt = 0
            for _ in range(10000):
                xp = x.copy()
                for f in set(fam):
                    i = np.where(fam == f)[0]
                    xp[i] = xp[rng.permutation(i)]
                cnt += spearman(xp, y) >= r0
            within = {f: spearman(x[fam == f], np.array(y)[fam == f]) for f in sorted(set(fam))}
            Z = np.column_stack([np.ones(len(ok)), rankdata([r["gamma"] for r in ok])])
            rx, ry = rankdata(x), rankdata(y)
            ex, ey = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0], ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
            e[g] = dict(spearman=r0, within_family=within, perm_p_within_family=float((cnt + 1) / 10001),
                        partial_given_gamma=None if g.endswith("formula") else float(np.corrcoef(ex, ey)[0, 1]))
        extra[target] = e
    # sources with nearly the same gamma in different families: does the gain order their exponents?
    pairs = []
    for a_, b_ in (("hmm_pl_a", "gated_p1.5"), ("hmm_pl_b", "gated_p2"), ("hmm_two", "gated_p1.5"), ("hmm_exp", "gated_p3")):
        ra, rb = next(r for r in rows if r["name"] == a_), next(r for r in rows if r["name"] == b_)
        pairs.append({k: [ra[k], rb[k]] for k in ("name", "gamma", "gain_learn", "gain_d8", "gain_total", "alpha", "nu_match")})
    extra["matched_gamma_pairs"] = pairs
    jobs = [l.split()[:3] for l in (ROOT / "results/zoo/jobs.txt").read_text().splitlines()]
    have = {(n, w, s) for n in A["per_source"] for w, s in
            ((f.stem[3:].split("_s")[0], f.stem.split("_s")[-1]) for f in (ROOT / "results/runs" / f"zoo_{n}").glob("S_n*_s*.json"))}
    missing = {}
    for n, w, s in jobs:
        if (n, w, s) not in have:
            missing.setdefault(n, []).append(f"n{w}_s{s}")
    out = dict(note="landed runs only (134 of 230); data content verified per run (results/zoo/content_check.json)",
               fit_budgets=FIT, rows=rows, cross_direct_T256=cross, gain_vs_rule_residual=add, gated_power=gpow, gain_tests=extra,
               length_controls={r["name"]: {k: r[k] for k in ("alpha", "alpha_tail", "nu_match", "nu_learned", "p_info",
                                                               "info_formula", "p_info_nu", "nu_formula")}
                                for r in rows if r["length_control"]},
               missing_runs=missing)
    p = ROOT / "results/zoo/analysis_partial.json"
    p.write_text(json.dumps(out, indent=1))
    f = lambda v: "   - " if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:6.3f}"
    print(f"{'source':15s} {'alpha':>6s} {'a_tail':>6s} {'nu_m':>6s} {'nu_lrn':>6s} | {'gamma':>6s} {'g/1+g':>6s} {'p_info':>6s} "
          f"{'p_nu':>6s} {'gain':>6s} {'g_d8%':>6s} {'g_ln%':>6s} {'beta':>6s} {'p_cag':>6s}")
    for r in rows:
        print(f"{r['name']:15s} {f(r['alpha'])} {f(r['alpha_tail'])} {f(r['nu_match'])} {f(r.get('nu_learned'))} | {f(r['gamma'])} "
              f"{f(r['info_formula'])} {f(r['p_info'])} {f(r['p_info_nu'])} {f(r['gain_total'])} {f(100 * r['gain_d8'])} "
              f"{f(None if r['gain_learn'] is None else 100 * r['gain_learn'])} {f(r['beta'])} {f(r['p_cag'])}")
    print(json.dumps(dict(gain_tests=extra), indent=0, default=str))


if __name__ == "__main__":
    main()
