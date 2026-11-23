"""Per-run event cost K_run of the gamma dial.

For each run, distance k counts as learned when f_k >= 1/2 (test events >= 50, k <= kmax). P(learned) is a logistic in
ln N_k, N_k = exact marker-k events in the training windows, with one slope shared by all runs of an encoding and one
threshold per run; K_run is the event count where P(learned) = 1/2. --slope fixes the shared slope (e.g. to the seed-0
value when scoring a new seed).
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

H1 = 1 + 0.1 * np.log2(0.1) + 0.9 * np.log2(0.9)
ap = argparse.ArgumentParser()
ap.add_argument("--enc", default="C")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--kmax", type=int, default=140)
ap.add_argument("--nwmin", type=int, default=64)
ap.add_argument("--slope", type=float, default=None)
ap.add_argument("--tag", default="gated_g{g}_T256_rope")
ap.add_argument("--data", default="data/gated_g{g}_T256.npz")
ap.add_argument("--out", default=None)
a = ap.parse_args()

runs, rows = [], []
for g in ("0.5", "1", "2"):
    D = np.load(a.data.format(g=g)); q = D["q"]; mtr = D[f"{a.enc}_train_c"]
    for p in sorted(Path("results/runs", a.tag.format(g=g)).glob(f"{a.enc}_n*_s{a.seed}.json")):
        r = json.loads(p.read_text()); nw = r["nwin"]
        if nw < a.nwmin:
            continue
        s, c = np.zeros(256), np.zeros(256)
        s[:len(r["kl_sum_by_cond"])] = r["kl_sum_by_cond"]; c[:len(r["count_by_cond"])] = r["count_by_cond"]
        N = np.bincount(mtr[:nw].ravel(), minlength=256)[:256].astype(float)
        j = len(runs); runs.append(dict(g=g, nw=nw, P=nw * 256, q=q, steps=r["steps"]))
        for k in range(1, a.kmax + 1):
            if c[k] >= 50:
                rows.append((j, np.log(max(N[k], 0.5)), float(1 - s[k] / c[k] / H1 >= 0.5)))
A = np.array(rows); j = A[:, 0].astype(int); lN, y = A[:, 1], A[:, 2]


def nll(th):
    sl = a.slope if a.slope is not None else np.exp(th[0])
    z = sl * (lN - th[1:][j])
    return float(np.sum(np.logaddexp(0, -z) * y + np.logaddexp(0, z) * (1 - y)))


r = minimize(nll, np.r_[np.log(2), np.full(len(runs), 6.0)], method="L-BFGS-B", bounds=[(-3, 3)] + [(0, 14)] * len(runs))
sl = a.slope if a.slope is not None else float(np.exp(r.x[0]))
print(f"{a.enc} seed {a.seed}: shared slope {sl:.2f}{' (fixed)' if a.slope is not None else ''}")
out = []
for jj, ru in enumerate(runs):
    K = float(np.exp(r.x[1 + jj])); T = 256; k = np.arange(1, T)
    N = ru["P"] * ru["q"][1:T] * (T - k) / T
    nstar = int(k[np.argmax(N < K)] - 1) if np.any(N < K) else 255
    ok = [v for v in np.linspace(r.x[1 + jj] - 3, r.x[1 + jj] + 3, 301)
          if nll(np.r_[r.x[:1 + jj], v, r.x[2 + jj:]]) - r.fun <= 1.92]
    lo, hi = float(np.exp(min(ok))), float(np.exp(max(ok)))
    nl, nr = int(y[j == jj].sum()), int((j == jj).sum())
    print(f"  g{ru['g']} 2^{int(np.log2(ru['P']))}: K_run {K:6.0f} [{lo:.0f}, {hi:.0f}]  implied n* {nstar}  learned {nl}/{nr}")
    out.append(dict(g=ru["g"], log2P=int(np.log2(ru["P"])), K_run=K, ci95=[lo, hi], implied_nstar=nstar, learned=nl, tested=nr))
if a.out:
    json.dump(dict(enc=a.enc, seed=a.seed, kmax=a.kmax, slope=sl, runs=out), open(a.out, "w"), indent=1)
