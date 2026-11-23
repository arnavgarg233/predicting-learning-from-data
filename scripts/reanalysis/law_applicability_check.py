"""Is there a quantity, computable from the source alone, that says when the information law
applies? The law matches Ising X and the gated-copy sources and misses Ising Z (faster: 0.83 against 0.47-0.50) and the
parity recoding gated_p1_D (slower: 0.25 against 0.58).

    python scripts/reanalysis/law_applicability_check.py -> results/reanalysis/law_applicability_check.json

Three candidates, each tied to one stated reason the law could fail, chosen before this script was run. Which sources
pass and fail is known from the paper, so this is a check of named candidates, not a blind test. Two of the three were
first computed on 2026-10-23 (scripts/law_applicability.py); their exact values are read from its output,
and the depth profiles are recomputed here with that script's functions and extended to Ising Z.

C1 pair-information gap. R = sum_K I(x_t; x_{t-k}) / sum_K [h(k-1) - h(k)] over the distances K the law marks as learned
   inside the fitted budgets. R >> 1: distant tokens carry far more pairwise information than they add to the
   prediction (redundant, shareable across distances: faster?). R << 1: the information is invisible pairwise (slower?).
C2 low-order recoverability. rho_k = I(x_t; x_{t-k} | x_{t-j..t-1}) / dh_k with j = floor((k-1)/2): the share of
   distance k's information visible when only the nearer half of the intermediate tokens is known, at k = 6 and 10.
C3 information-curve cost. Any functional of h(n) and the event price (the law's own per-distance cost), represented by
   the law's prediction itself. X and Z, and copy and parity, have the same information curve up to a one-symbol shift.
Separation means: every source where the law holds lies in one interval of the candidate, and each failing source lies
outside it. Ising X and Z are exact (free fermions); gated sources exact per marker draw (32 draws); HMM exact.
"""
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("lawapp", ROOT / "scripts/law_applicability.py")
LA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LA)
from exact_sources import tfim, zoo  # noqa: E402

OUT = ROOT / "results/reanalysis"


def depth_ising_z(k):
    """I(z_t; z_{t-k} | z_{t-j..t-1}), j = 0..k-1, critical chain in the Z encoding. z_{t-k} is a uniform random sign
    independent of the dual X window w_1..w_k, and z_{t-i} = z_{t-k} prod_{m <= k-i} w_m (Methods)."""
    Gam = tfim.window_gamma(tfim.g_coeffs(1.0, k), k)
    pw = {}
    for c in itertools.product([0, 1], repeat=k):
        w = 1.0 - 2.0 * np.array(c)
        pw[c] = tfim.window_prob(Gam, w)
    out = []
    for j in range(k):
        n = j + 2  # tokens: z_{t-k}, z_{t-j}, ..., z_t
        P = np.zeros([2] * n)
        for c, p in pw.items():
            w = 1.0 - 2.0 * np.array(c)
            cp = np.cumprod(w)  # cp[i-1] = prod_{m <= i} w_m, relative sign of z_{t-k+i}
            rel = [cp[k - i - 1] for i in range(j, -1, -1)]  # z_{t-j}, ..., z_t relative to z_{t-k}
            for z0 in (1.0, -1.0):
                idx = tuple(int(v < 0) for v in [z0] + [z0 * r for r in rel])
                P[idx] += 0.5 * p
        out.append(LA._cmi_from_joint(P))
    return out


def main():
    prior = json.load(open(ROOT / "results/law_applicability.json"))
    diag, M = prior["diagnostic"], prior["exponents"]
    # ---------------- C2: depth profiles
    D = {}
    for k in (6, 10):
        D[f"ising_X_k{k}"] = LA.depth_ising_x(k)
        D[f"ising_Z_k{k}"] = depth_ising_z(k)
        for nm in ("gated_p1", "gated_p1_D"):
            p = LA.load_pred(nm)
            enc = "D" if p["source"]["enc"] == "D" else "C"
            D[f"{nm}_k{k}"] = LA.depth_gated(np.array(p["extra"]["q"]), np.array(p["extra"]["sigma"]), p["source"]["eps"],
                                             p["T"], enc, k, np.random.default_rng([12, 2312, k]))
        for nm in ("hmm_two", "hmm_exp"):
            s = LA.load_pred(nm)["source"]
            D[f"{nm}_k{k}"] = LA.depth_hmm(zoo.Switching(1 / (2 * np.asarray(s["tau"], float)), s["w"], s["eps"]), k)
    # self-checks: Z's information gain at k equals X's at k-1 (curves shifted by one symbol); reproduce stored profiles
    check = {f"Z_dh_k{k}_vs_X_dh_k{k - 1}": [D[f"ising_Z_k{k}"][-1], LA.depth_ising_x(k - 1)[-1]] for k in (6, 10)}
    check["max_abs_diff_vs_profiles"] = max(
        abs(a - b) for key, v in prior["depth_profiles"].items() for a, b in zip(v, D[key]))
    rho = {}
    for key, v in D.items():
        k = len(v)
        j = (k - 1) // 2
        rho[key] = dict(j=j, I_half=v[j], dh=v[-1], rho=v[j] / v[-1], I_pair=v[0], pair_over_dh=v[0] / v[-1])
    # ---------------- assemble per source
    cls = {"ising_X": "holds", "ising_Z": "faster", "gated_p1": "holds", "gated_p1_D": "slower", "hmm_two": "holds",
           "hmm_exp": "holds"}
    rows = {}
    for s, c in cls.items():
        rows[s] = dict(cls=c, measured=M[s]["measured"], predicted=M[s]["predicted"],
                       miss=M[s]["measured"] - M[s]["predicted"], R=diag[s]["R"],
                       rho_k6=rho[f"{s}_k6"]["rho"], rho_k10=rho[f"{s}_k10"]["rho"])
    # C1 over all 16 main sources (stored)
    main16 = [r["name"] for r in prior["score_main"]["table"]]
    c1 = {n: dict(R=diag[n]["R"], cls=next(r["cls"] for r in prior["score_main"]["table"] if r["name"] == n)) for n in main16}

    def separates(vals):
        hold = [v for n, v in vals.items() if cls_all(n) == "holds"]
        lo, hi = min(hold), max(hold)
        return {n: ("inside" if lo <= v <= hi else ("above" if v > hi else "below"))
                for n, v in vals.items() if cls_all(n) != "holds"}, [lo, hi]

    def cls_all(n):
        return c1[n]["cls"] if n in c1 else cls[n]

    sep = dict(C1_R_main16=separates({n: v["R"] for n, v in c1.items()}),
               C2_rho_k6=separates({n: rows[n]["rho_k6"] for n in rows}),
               C2_rho_k10=separates({n: rows[n]["rho_k10"] for n in rows}),
               C3_predicted_exponent_pairs=dict(ising_X=M["ising_X"]["predicted"], ising_Z=M["ising_Z"]["predicted"],
                                                gated_p1=M["gated_p1"]["predicted"],
                                                gated_p1_D=M["gated_p1_D"]["predicted"]))
    res = dict(rows=rows, separation=sep, depth_profiles=D, rho=rho, checks=check, C1_main16=c1)
    json.dump(res, open(OUT / "law_applicability_check.json", "w"), indent=1)
    print(json.dumps(check, indent=1))
    print(f"{'source':12s} {'class':7s} {'meas':>6s} {'pred':>6s} {'R':>9s} {'rho6':>9s} {'rho10':>9s}")
    for s, r in rows.items():
        print(f"{s:12s} {r['cls']:7s} {r['measured']:6.3f} {r['predicted']:6.3f} {r['R']:9.3g} {r['rho_k6']:9.3g} "
              f"{r['rho_k10']:9.3g}")
    for k, v in D.items():
        print(k, " ".join(f"{a:.2e}" for a in v))
    print(json.dumps(sep, indent=1))


if __name__ == "__main__":
    main()
