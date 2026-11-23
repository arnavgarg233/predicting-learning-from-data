"""When does the information law apply? A data-only diagnostic, computed from each source's exact structure.

The law (paper, "An information law for the learning rate"): a learner that reaches every context distance and learns
each distance from its own evidence has data exponent gamma/(1+gamma); one price (210 events per distance) places the
curve. It holds on Ising X and the 13-source zoo and misses Ising Z (faster) and the parity recoding gated_p1_D (slower).

Diagnostic (brief of 2026-10-23): compare, at each distance k, the pairwise mutual information I_k = I(x_t; x_{t-k})
(conditioned on the visible marker channel where a source has one, as h(n) is) with the information gain
dh_k = h(k-1) - h(k) = I(x_t; x_{t-k} | x_{t-k+1..t-1}). Primary statistic: R = sum_K I_k / sum_K dh_k over the
distances K that the law itself says become learned inside the fitted budget range (price 210 events = zoo.B_STAR
bits). Nothing here reads a measured learning exponent; those are joined only in the scoring block at the end.

Exact where the source allows it:
  hmm     I_k from the exact pair correlation (zero mean).
  ising   X: exact mean 2/pi and connected <X_0 X_r>; Z: exact string correlation (zero mean). dh from the zoo curve.
  gated   given all markers the tokens are a zero-field tree Ising model (the copy forest), and every encoding here
          (copy C, diff D, parity Cst, ramp R) writes each token as a parity of a set of x's. E[prod_S x] is then a
          product of edge correlations over the edges cut an odd number of times, or 0 if some tree holds an odd number
          of S; I_k is exact per marker draw and Monte Carlo over marker draws only. Check: I_1 = dh_1 by definition.

Outputs results/law_applicability.json. CPU only, small memory.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exact_sources import gated, zoo  # noqa: E402

OUT = ROOT / "results/law_applicability.json"
NWIN = 256  # marker draws per gated source
NPERM = 200_000


# ------------------------------------------------------------------ binary pair information
def pair_mi(mua, mub, muab):
    """I(a; b) in bits for a, b in {+-1} with E[a], E[b], E[ab] (arrays)."""
    mua, mub, muab = (np.asarray(v, float) for v in (mua, mub, muab))
    H = np.zeros(np.broadcast(mua, mub, muab).shape)
    for a in (1, -1):
        for b in (1, -1):
            p = np.clip((1 + a * mua + b * mub + a * b * muab) / 4, 0, 1)
            H -= np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)
    return zoo.hbin((1 + mua) / 2) + zoo.hbin((1 + mub) / 2) - H


# ------------------------------------------------------------------ gated sources: exact tree-parity moments
def token_sets(enc, mw, T, offs=None):
    """Each token as a bitmask of the x indices whose product it is (given the markers mw)."""
    S = [0] * T
    for t in range(T):
        if enc == "C":
            S[t] = 1 << t
        elif enc == "D":
            S[t] = (1 << t) | (1 << (t - 1)) if t else 1
        else:  # Cst / R: z_t = x_t x_s z_s prod_j z_{t-j} on a copy event (exact_sources.gated.to_parity[_multi])
            k = int(mw[t])
            if k == 0 or t == 0:
                S[t] = 1 << t
            else:
                s = t - k
                v = (1 << t) ^ (1 << s) ^ S[s]
                for j in np.atleast_1d(offs[k]):
                    if j > 0:
                        v ^= S[t - int(j)]
                S[t] = v
    return S


def gated_pair_info(q, sigma, eps, T, enc, rng, nwin=NWIN, offs=None):
    """Mean over marker draws and positions t >= k of I(token_t; token_{t-k} | markers), k = 1..T-1."""
    r = 1 - 2 * eps
    tot = np.zeros(T)
    cnt = np.zeros(T)
    for _ in range(nwin):
        mw = gated.sample_markers(q, T, 1, rng)[0]
        pm, root, neg = [0] * T, [0] * T, 0
        for u in range(T):
            if mw[u] > 0:
                p = u - int(mw[u])
                pm[u], root[u] = pm[p] | (1 << u), root[p]
                if sigma[mw[u]] < 0:
                    neg |= 1 << u
            else:
                root[u] = u
        S = token_sets(enc, mw, T, offs)
        TM, TR = [0] * T, [0] * T
        for t in range(T):
            a, b, v = 0, 0, S[t]
            while v:
                low = v & -v
                i = low.bit_length() - 1
                a ^= pm[i]
                b ^= 1 << root[i]
                v ^= low
            TM[t], TR[t] = a, b

        def mom(F, R):
            if R:
                return 0.0
            return r ** F.bit_count() * (-1.0 if (F & neg).bit_count() & 1 else 1.0)

        mu = np.array([mom(TM[t], TR[t]) for t in range(T)])
        for k in range(1, T):
            t = np.arange(k, T)
            muab = np.array([mom(TM[i] ^ TM[i - k], TR[i] ^ TR[i - k]) for i in t])
            tot[k] += pair_mi(mu[t], mu[t - k], muab).sum()
            cnt[k] += len(t)
    out = np.full(T, np.nan)
    out[1:] = tot[1:] / cnt[1:]
    return out


# ------------------------------------------------------------------ fitted range and the statistic
def law_range(h, T, budgets, Bstar=zoo.B_STAR):
    """Distances that the law marks learned at the largest fit budget and not at the smallest (zoo.tail_curves rule)."""
    n = np.arange(1, T)
    delta = np.clip(h[:-1] - h[1:], 0, None)
    N = lambda P: P * delta * (T - n) / T / Bstar
    return n[(N(max(budgets)) >= 1) & (N(min(budgets)) < 1)]


def diagnostic(h, I, T, budgets):
    dh = np.clip(h[:-1] - h[1:], 0, None)  # dh[k-1] = h(k-1) - h(k)
    Ik = I[1:]
    K = law_range(h, T, budgets)
    ratio = lambda ks: float(np.nansum(Ik[ks - 1]) / np.sum(dh[ks - 1])) if len(ks) else None
    k2 = np.arange(2, min(64, T - 1) + 1)
    per = Ik[K - 1] / np.where(dh[K - 1] > 0, dh[K - 1], np.nan) if len(K) else np.array([])
    return dict(
        K=[int(K.min()), int(K.max())] if len(K) else None, nK=int(len(K)),
        R=ratio(K), log10R=float(np.log10(ratio(K))) if len(K) and ratio(K) > 0 else None,
        R_k2_64=ratio(k2),
        median_log10_ratio_K=float(np.nanmedian(np.log10(np.clip(per, 1e-300, None)))) if len(per) else None,
        check_I1_vs_dh1=[float(Ik[0]), float(dh[0])],
        I_k=[float(v) for v in Ik[:64]], dh_k=[float(v) for v in dh[:64]])


# ------------------------------------------------------------------ sources
def load_pred(name):
    return json.load(open(ROOT / f"results/zoo/pred/{name}.json"))


ZOO_FIT = [2 ** e for e in range(14, 21)]  # analyze_zoo regresses 2^14..2^20
ISING_FIT = [2 ** 17, 2 ** 20]  # Fig. 4a, longer protocol (results/runs/crit_T256_v3)
RAMP_FIT = [2 ** e for e in range(16, 21)]  # results/analysis_ramp.json


def compute_all():
    rows = {}
    zoo_names = ["gated_geo", "gated_mix", "gated_p0.25", "gated_p0.5", "gated_p0.75", "gated_p1", "gated_p1.5",
                 "gated_p2", "gated_p3", "hmm_exp", "hmm_pl_a", "hmm_pl_b", "hmm_two", "gated_p1_D",
                 "gated_p0.5_T64", "gated_p1_T64", "gated_step"]
    for nm in zoo_names:
        p = load_pred(nm)
        T, h = p["T"], np.array(p["h_n"])
        if p["family"] == "gated":
            src = p["source"]
            q, sigma = np.array(p["extra"]["q"]), np.array(p["extra"]["sigma"])
            enc = "D" if src["enc"] == "D" else "C"
            I = gated_pair_info(q, sigma, src["eps"], T, enc, np.random.default_rng([src["seed"], 2310]))
            how = f"exact tree-parity moments given markers, {NWIN} marker draws"
        else:
            C = np.array([np.nan if v is None else v for v in p["absC"]])
            I = np.full(T, np.nan)
            I[1:] = pair_mi(0.0, 0.0, C[1:])
            how = "exact pair correlation"
        rows[nm] = dict(group="zoo", family=p["family"], how=how, fit_budgets=ZOO_FIT,
                        **diagnostic(h, I, T, ZOO_FIT))
        print(nm, rows[nm]["K"], rows[nm]["R"], rows[nm]["check_I1_vs_dh1"], flush=True)
    for b, nm in (("X", "ising_X_h1"), ("Z", "ising_Z_h1")):
        p = load_pred(nm)
        T, h = p["T"], np.array(p["h_n"])
        C = np.array([np.nan if v is None else v for v in p["absC"]])
        mu = 2 / np.pi if b == "X" else 0.0  # <X> = G(0) at h = 1; Z has a fair global sign
        I = np.full(T, np.nan)
        I[1:] = pair_mi(mu, mu, mu * mu + C[1:])
        for tag, fb in (("", ISING_FIT), ("_zoo_budgets", ZOO_FIT)):
            rows[f"ising_{b}{tag}"] = dict(group="ising" if not tag else "ising_sensitivity", family="ising",
                                           how="exact mean and correlation (free fermions)", fit_budgets=fb,
                                           **diagnostic(h, I, T, fb))
        print(nm, rows[f"ising_{b}"]["K"], rows[f"ising_{b}"]["R"], rows[f"ising_{b}"]["check_I1_vs_dh1"], flush=True)
    # built but never trained (results/zoo/logs/ising_exit): diagnostic only, for a later prospective test
    from exact_sources import tfim
    for nm in ("ising_X_h0.8", "ising_X_h1.1", "ising_Z_h0.909", "ising_Z_h1.25"):
        p = load_pred(nm)
        T, h = p["T"], np.array(p["h_n"])
        C = np.array([np.nan if v is None else v for v in p["absC"]])
        mu = float(tfim.g_coeffs(p["source"]["h"], 0)[0]) if p["source"]["basis"] == "X" else 0.0
        I = np.full(T, np.nan)
        I[1:] = pair_mi(mu, mu, mu * mu + C[1:])
        rows[nm] = dict(group="untrained", family="ising", how="exact mean and correlation (free fermions)",
                        fit_budgets=ZOO_FIT, zoo_predicted_alpha=p["alpha"]["info"], **diagnostic(h, I, T, ZOO_FIT))
        print(nm, rows[nm]["K"], rows[nm]["R"], rows[nm]["check_I1_vs_dh1"], flush=True)
    # ramp arms (gamma 0.5): identical information curve, so dh = the exact direct curve of the copy source
    d = np.load(ROOT / "data/gated_g0.5_ramp2_T256.npz")
    q, sigma, eps, T = d["q"], d["sigma"], float(d["eps"]), int(d["T"])
    h = 1 - (1 - zoo.hbin(eps)) * np.cumsum(q)[:T]
    for arm, offs in (("C", None), ("Cst", d["Cst_offset"]), ("R", d["R_offset"])):
        I = gated_pair_info(q, sigma, eps, T, arm, np.random.default_rng([int(d["seed"]), 2311]), nwin=128, offs=offs)
        rows[f"ramp_{arm}"] = dict(group="ramp", family="gated", fit_budgets=RAMP_FIT,
                                   how="exact tree-parity moments given markers, 128 marker draws; dh = exact direct curve",
                                   **diagnostic(h, I, T, RAMP_FIT))
        print("ramp", arm, rows[f"ramp_{arm}"]["K"], rows[f"ramp_{arm}"]["R"], flush=True)
    return rows


# ------------------------------------------------------------------ measured exponents (joined after the diagnostic)
def measured():
    """(measured, predicted, file) per source. Zoo: results/zoo/price_model.json (the ED Fig. 3 numbers) and
    analysis_partial.json for the recoding and the length controls; Ising: results/runs/crit_T256_v3 (Fig. 4a) with
    the paper's predictions; ramp: results/analysis_ramp.json, predicted = the copy arm's exponent (same curve)."""
    M = {}
    pm = json.load(open(ROOT / "results/zoo/price_model.json"))["sources"]
    for k, v in pm.items():
        M[k] = dict(measured=v["measured"], predicted=v["fixed_price"], file="results/zoo/price_model.json")
    for r in json.load(open(ROOT / "results/zoo/analysis_partial.json"))["rows"]:
        if r["name"] in ("gated_p1_D", "gated_p0.5_T64", "gated_p1_T64"):
            M[r["name"]] = dict(measured=r["alpha"], predicted=r["p_info"], file="results/zoo/analysis_partial.json")
    runs = [json.load(open(f)) for f in sorted((ROOT / "results/runs/crit_T256_v3").glob("*.json"))]
    g = json.load(open(ROOT / "results/source_stats.json"))
    gX, gZ = g["h=1.0 X"]["gamma"], g["h=1.0 Z"]["gamma"]
    for b in "XZ":
        R = [r for r in runs if r["basis"] == b]
        P = sorted({r["P"] for r in R})
        S = sorted({r["seed"] for r in R})
        kl = {(r["P"], r["seed"]): r["test_kl_bits"] for r in R}
        gm = [np.exp(np.mean([np.log(kl[(p, s)]) for s in S])) for p in P]
        a = float(-np.log(gm[-1] / gm[0]) / np.log(P[-1] / P[0]))
        pred = gX / (1 + gX) if b == "X" else 0.5 * (gX / (1 + gX) + gZ / (1 + gZ))  # Z: midpoint of 0.47-0.50
        M[f"ising_{b}"] = dict(measured=a, predicted=float(pred), file="results/runs/crit_T256_v3 + results/source_stats.json")
    ra = json.load(open(ROOT / "results/analysis_ramp.json"))["arms"]
    for arm in ("C", "Cst", "R"):
        M[f"ramp_{arm}"] = dict(measured=ra[arm]["exponents_2seed_mean"]["loss_exp"],
                                predicted=ra["C"]["exponents_2seed_mean"]["loss_exp"],
                                file="results/analysis_ramp.json (loss_exp; prediction = copy arm, identical curve)")
    return M


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def score(rows, M, names, key="log10R", tol=0.25):
    x = np.array([rows[n][key] for n in names], float)
    miss = np.array([M[n]["measured"] - M[n]["predicted"] for n in names])
    rho = spearman(x, miss)
    rng = np.random.default_rng(20261023)
    perm = np.array([spearman(x, rng.permutation(miss)) for _ in range(NPERM // 10)])
    cls = np.where(miss > tol, "faster", np.where(miss < -tol, "slower", "holds"))
    holds = x[cls == "holds"]
    sep = {n: dict(cls=str(c), value=float(v),
                   rank_among_holds=("above all" if v > holds.max() else "below all" if v < holds.min() else "inside"))
           for n, c, v in zip(names, cls, x) if c != "holds"}
    return dict(statistic=key, n=len(names), spearman=rho, perm_p_two_sided=float(np.mean(np.abs(perm) >= abs(rho) - 1e-12)),
                perm_p_one_sided=float(np.mean(perm >= rho - 1e-12)), tolerance=tol,
                holds_range=[float(holds.min()), float(holds.max())], failures=sep,
                table=[dict(name=n, value=float(v), miss=float(m), cls=str(c)) for n, v, m, c in zip(names, x, miss, cls)])


# ------------------------------------------------------------------ follow-up: how deep is the synergy at distance k?
# I_k(j) = I(x_t; x_{t-k} | x_{t-1..t-j}); j = 0 is the pair information, j = k-1 is dh_k. Exact for Ising X (free
# fermions), the gated sources (tree-parity moments given markers, Monte Carlo over marker draws) and the HMMs (forward).
def _cmi_from_joint(P):
    """P[a, c..., b]: I(a; b | c) in bits."""
    def H(p):
        p = p[p > 0]
        return float(-(p * np.log2(p)).sum())
    n = P.ndim
    return H(P.sum(axis=n - 1).ravel()) + H(P.sum(axis=0).ravel()) - H(P.sum(axis=(0, n - 1)).ravel()) - H(P.ravel())


def depth_ising_x(k):
    import itertools
    from exact_sources import tfim
    Gam = tfim.window_gamma(tfim.g_coeffs(1.0, k + 1), k + 1)
    out = []
    for j in range(k):
        sites = [0] + list(range(k - j, k)) + [k]
        idx = np.concatenate([[2 * s, 2 * s + 1] for s in sites])
        Gs = Gam[np.ix_(idx, idx)]
        P = np.empty([2] * len(sites))
        for c in itertools.product([0, 1], repeat=len(sites)):
            P[c] = tfim.window_prob(Gs, 1.0 - 2.0 * np.array(c))
        out.append(_cmi_from_joint(P))
    return out


def depth_gated(q, sigma, eps, T, enc, k, rng, nwin=32, tpos=range(160, 256, 3)):
    r = 1 - 2 * eps
    acc = np.zeros(k)
    cnt = 0
    for _ in range(nwin):
        mw = gated.sample_markers(q, T, 1, rng)[0]
        pm, root, neg = [0] * T, [0] * T, 0
        for u in range(T):
            if mw[u] > 0:
                p = u - int(mw[u])
                pm[u], root[u] = pm[p] | (1 << u), root[p]
                if sigma[mw[u]] < 0:
                    neg |= 1 << u
            else:
                root[u] = u
        S = token_sets(enc, mw, T)
        TM, TR = [], []
        for t in range(T):
            a, b, v = 0, 0, S[t]
            while v:
                low = v & -v
                i = low.bit_length() - 1
                a ^= pm[i]
                b ^= 1 << root[i]
                v ^= low
            TM.append(a)
            TR.append(b)
        for t in tpos:
            for j in range(k):
                toks = [t - k] + list(range(t - j, t)) + [t]
                n = len(toks)
                F, R = [0], [0]
                for i in toks:  # subset index bit order: position i in toks -> bit i
                    F = F + [f ^ TM[i] for f in F]
                    R = R + [x ^ TR[i] for x in R]
                mu = np.array([0.0 if rr else r ** f.bit_count() * (-1.0 if (f & neg).bit_count() & 1 else 1.0)
                               for f, rr in zip(F, R)])
                # p(v) = 2^-n sum_S mu_S prod_{i in S} v_i, v_i = +-1 (bit 0 -> +1)
                Hd = np.array([[1.0]])
                for _ in range(n):
                    Hd = np.block([[Hd, Hd], [Hd, -Hd]])
                p = (Hd @ mu) / 2 ** n  # index of p: bit i set <-> v_i = -1, subset index likewise
                P = p.reshape([2] * n)  # C order: axis 0 = highest bit = last token; reverse to toks order
                P = np.transpose(P, list(range(n))[::-1])
                acc[j] += _cmi_from_joint(np.clip(P, 0, None))
            cnt += 1
    return (acc / cnt).tolist()


def depth_hmm(src, k):
    """Exact joint of (x_{t-k}, x_{t-j..t}) by the forward algorithm in the stationary law."""
    J = src.J
    nS = 2 ** J

    def A(nstep):  # per-channel flip over nstep steps, as a full transition matrix
        rho = (1 - (1 - 2 * src.r) ** nstep) / 2
        M = np.ones((nS, nS))
        for jj in range(J):
            same = src.S[:, jj][:, None] == src.S[:, jj][None, :]
            M *= np.where(same, 1 - rho[jj], rho[jj])
        return M

    em = lambda x: (1 + x * src.mu) / 2
    out = []
    for j in range(k):
        n = j + 2
        P = np.empty([2] * n)
        import itertools
        for c in itertools.product([0, 1], repeat=n):
            v = 1.0 - 2.0 * np.array(c)
            b = np.full(nS, 1.0 / nS) * em(v[0])
            b = b @ A(k - j) if j > 0 else b @ A(k)
            for i in range(1, n):
                b = b * em(v[i])
                if i < n - 1:
                    b = b @ A(1)
            P[c] = b.sum()
        out.append(_cmi_from_joint(P))
    return out


def depth_profiles(rows):
    D = {}
    for k in (6, 10):
        D[f"ising_X_k{k}"] = depth_ising_x(k)
        for nm in ("gated_p1", "gated_p1_D"):
            p = load_pred(nm)
            enc = "D" if p["source"]["enc"] == "D" else "C"
            D[f"{nm}_k{k}"] = depth_gated(np.array(p["extra"]["q"]), np.array(p["extra"]["sigma"]), p["source"]["eps"],
                                          p["T"], enc, k, np.random.default_rng([12, 2312, k]))
        s = load_pred("hmm_two")["source"]
        D[f"hmm_two_k{k}"] = depth_hmm(zoo.Switching(1 / (2 * np.asarray(s["tau"], float)), s["w"], s["eps"]), k)
        for key in [x for x in D if x.endswith(f"_k{k}")]:
            v = D[key]
            print(key, " ".join(f"{a:.2e}" for a in v), flush=True)
    return D


def main():
    rows = compute_all()  # blind to the exponents
    M = measured()
    main16 = ["gated_geo", "gated_mix", "gated_p0.25", "gated_p0.5", "gated_p0.75", "gated_p1", "gated_p1.5", "gated_p2",
              "gated_p3", "hmm_exp", "hmm_pl_a", "hmm_pl_b", "hmm_two", "gated_p1_D", "ising_X", "ising_Z"]
    plus = main16 + ["gated_p0.5_T64", "gated_p1_T64", "ramp_C", "ramp_Cst", "ramp_R"]
    res = dict(diagnostic=rows, exponents=M,
               score_main=score(rows, M, main16), score_main_median_ratio=score(rows, M, main16, "median_log10_ratio_K"),
               score_with_supplementary=score(rows, M, plus),
               depth_profiles=depth_profiles(rows))
    OUT.write_text(json.dumps(res, indent=1))
    for k in ("score_main", "score_main_median_ratio", "score_with_supplementary"):
        s = res[k]
        print(k, "rho", round(s["spearman"], 3), "p2", s["perm_p_two_sided"], "holds", s["holds_range"], s["failures"])


if __name__ == "__main__":
    main()
