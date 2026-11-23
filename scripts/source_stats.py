"""Exact source statistics for the TFIM X and Z measurement bases (infinite chain).

Delta(n) = h(n) - h(N) is measured as the mean KL between the next-token law given N
context tokens and given only the last n, with both laws exact. ||C(r)||_op for a binary
alphabet is |connected correlator| / 2.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact_sources import tfim

NS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128]


def delta_curve(h, N, nsamp, rng):
    """Delta(n) for X-basis data at field h, n in NS, with standard errors."""
    G = tfim.g_coeffs(h, N + 1)
    x, m = tfim.sample_x(tfim.window_gamma(G, N + 1), nsamp, rng, batch=64)
    m_full = m[:, -1]
    out = {}
    for n in NS:
        m_n = tfim.conditional_mean_last(tfim.window_gamma(G, n + 1), x[:, N - n:N])
        kl = tfim.kl_bits(m_full, m_n)
        out[n] = (float(kl.mean()), float(kl.std() / np.sqrt(len(kl))))
    h_full = float(tfim.binary_entropy_bits(m_full).mean())
    return out, h_full


def corr_curve(h, basis, rs):
    if basis == "X":
        G = tfim.g_coeffs(h, max(rs))
        c = np.array([tfim.x_connected(G, r) for r in rs])
    else:
        G = tfim.g_coeffs(1.0 / h, max(rs))
        c = np.array([tfim.string_corr(G, r) for r in rs])
    return np.abs(c) / 2


def fit_power(xs, ys):
    b, a = np.polyfit(np.log(xs), np.log(ys), 1)
    return -b, np.exp(a)


def fit_gamma(ns, d, N):
    """Fit Delta(n) = a (n^-g - N^-g) by grid search on g."""
    best = None
    for g in np.linspace(0.3, 2.0, 1701):
        f = ns ** -g - N ** -g
        a = np.dot(f, d) / np.dot(f, f)
        res = np.sum((np.log(d) - np.log(a * f)) ** 2)
        if best is None or res < best[0]:
            best = (res, g, a)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, nargs="+", default=[1.0, 1.5])
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--nsamp", type=int, default=4000)
    ap.add_argument("--out", default="results/source_stats.json")
    args = ap.parse_args()
    rng = np.random.default_rng(12345)
    rs = np.unique(np.round(np.logspace(0, np.log10(1024), 40)).astype(int))
    report = {}
    for h in args.h:
        for basis in ("X", "Z"):
            t0 = time.time()
            hx = h if basis == "X" else 1.0 / h
            d, h_full = delta_curve(hx, args.N, args.nsamp, rng)
            shift = 0 if basis == "X" else 1  # Z context n carries n-1 domain walls
            ns = np.array([n + shift for n in NS])
            dm = np.array([d[n][0] for n in NS])
            de = np.array([d[n][1] for n in NS])
            C = corr_curve(h, basis, rs)
            sel = (rs >= 16) & (rs <= 256)
            beta, camp = fit_power(rs[sel], C[sel])
            fitn = (ns >= 4) & (ns <= 64)
            gamma, gamp = fit_gamma(ns[fitn].astype(float), dm[fitn], args.N + shift)
            key = f"h={h} {basis}"
            report[key] = dict(
                h=h, basis=basis, n=ns.tolist(), delta_bits=dm.tolist(), delta_se=de.tolist(),
                h_at_N_bits=h_full, r=rs.tolist(), C_op=C.tolist(), beta=beta, C_amp=camp,
                gamma=gamma, gamma_amp=gamp, alpha_pred=gamma / (2 * beta), seconds=time.time() - t0,
            )
            print(f"{key}: beta={beta:.3f} gamma={gamma:.3f} alpha=gamma/2beta={gamma/(2*beta):.3f} "
                  f"h(N)={h_full:.4f} bits  ({time.time()-t0:.0f}s)")
            for n, a, b in zip(ns, dm, de):
                print(f"    n={n:4d}  h(n)-h(N) = {a:.3e} +- {b:.1e} bits")
            for r, c in zip(rs[::6], C[::6]):
                print(f"    r={r:5d}  ||C||_op = {c:.3e}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
