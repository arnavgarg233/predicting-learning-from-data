"""Paired X/Z datasets with exact next-token conditional means.

X data are X-basis windows of the chain at field hx. Z data are the Z-basis windows of the
chain at field 1/hx, built from the same samples by Kramers-Wannier duality, so each Z
window is an exact recoding of the matching X window.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact_sources import tfim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hx", type=float, default=1.0)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--ntrain", type=int, default=8192)
    ap.add_argument("--nval", type=int, default=512)
    ap.add_argument("--ntest", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.ntrain + args.nval + args.ntest
    Gam = tfim.window_gamma(tfim.g_coeffs(args.hx, args.T), args.T)
    w, mw = tfim.sample_x(Gam, n, rng, batch=128)
    z = tfim.x_to_z(w[:, :-1], rng)
    mz = np.zeros_like(z)
    mz[:, 1:] = z[:, :-1] * mw[:, :-1]
    splits = np.cumsum([0, args.ntrain, args.nval, args.ntest])
    data = {}
    for name, a, b in zip(("train", "val", "test"), splits[:-1], splits[1:]):
        data[f"X_{name}"] = (w[a:b] < 0).astype(np.int8)
        data[f"X_{name}_m"] = mw[a:b].astype(np.float64)
        data[f"Z_{name}"] = (z[a:b] < 0).astype(np.int8)
        data[f"Z_{name}_m"] = mz[a:b].astype(np.float64)
    np.savez_compressed(args.out, hx=args.hx, hz=1.0 / args.hx, T=args.T, **data)
    print("wrote", args.out, {k: v.shape for k, v in data.items() if not k.endswith("_m")})


if __name__ == "__main__":
    main()
