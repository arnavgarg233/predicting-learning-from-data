"""Gated-copy datasets (exact_sources/gated.py) in the ladder npz format: bases C (copy) and D (diff recoding).

Keys per basis b and split s: b_s (int8 tokens, 1 = value -1), b_s_m (exact conditional mean),
b_s_c (uint8 marker = copy distance, 0 = fair coin; the model sees it at the predicted position).
Both bases share the same windows and markers, so D is an exact recoding of C.
--cap_k1 K1 adds two more exact recodings of the same windows (exact_sources/gated.py to_parity): Cst, one extra parity bit at
every distance k >= 2, and B, the same extra bit only at k >= K1 (a copy below). The bit offsets come from their own
generator (--parity_seed), so the C and D arrays are unchanged.
--ramp_k0 K0 --ramp_dmax DMAX adds R, whose number of extra parity bits grows with distance,
d(k) = clip(floor(log2(k / K0)) + 1, 0, DMAX), with offsets from a third generator; it also adds Cst if --cap_k1 is not set.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact_sources import gated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--Q", type=float, default=0.5, help="total copy probability (before window truncation)")
    ap.add_argument("--eps", type=float, default=0.1, help="copy flip probability")
    ap.add_argument("--ntrain", type=int, default=4096)
    ap.add_argument("--nval", type=int, default=512)
    ap.add_argument("--ntest", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="set q_k = 0 for LO <= k <= HI after normalising (other q_k unchanged)")
    ap.add_argument("--cap_k1", type=int, default=None, help="add bases Cst and B (B: extra parity bit only at k >= K1)")
    ap.add_argument("--parity_seed", type=int, default=125, help="seed of the parity-bit offsets (own generator)")
    ap.add_argument("--ramp_k0", type=int, default=None, help="add base R (extra bits growing with log2 distance from K0)")
    ap.add_argument("--ramp_dmax", type=int, default=4)
    ap.add_argument("--qshape", default="power", choices=["power", "geo", "step", "mix"],
                    help="memory shape of q (gated.make_q_shape); power is the original make_q path")
    ap.add_argument("--rho", type=float, default=0.9, help="geo: q_k ~ rho^k")
    ap.add_argument("--lag", type=int, default=16, help="step: the one copy distance")
    ap.add_argument("--band", type=int, nargs=2, default=[48, 63], help="mix: band of distances")
    ap.add_argument("--Qband", type=float, default=0.05, help="mix: total copy probability in the band")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    N = a.T - 1
    extra = {}
    if a.qshape == "power":
        q = gated.make_q(a.gamma, N, a.Q)
    else:
        q = gated.make_q_shape(a.qshape, N, a.Q, gamma=a.gamma, rho=a.rho, lag=a.lag, band=tuple(a.band), Qband=a.Qband)
        extra.update(qshape=a.qshape, rho=a.rho, lag=a.lag, band=np.array(a.band), Qband=a.Qband)
    if a.zero:
        q[a.zero[0]:a.zero[1] + 1] = 0
    sigma = gated.make_sigma(N, rng)
    n = a.ntrain + a.nval + a.ntest
    x, m, mx = gated.sample(q, sigma, a.eps, a.T, n, rng)
    y, my = gated.to_diff(x, mx)
    bases = [("C", x, mx), ("D", y, my)]
    if a.cap_k1:
        offs = gated.parity_offsets(N, np.random.default_rng(a.parity_seed))
        offs_b = np.where(np.arange(N + 1) >= a.cap_k1, offs, 0)
        for b, o in (("Cst", offs), ("B", offs_b)):
            z, mz = gated.to_parity(x, m, mx, o)
            assert np.array_equal(gated.from_parity(z, m, o), x), f"{b} does not decode to the source"
            bases.append((b, z, mz))
            extra[f"{b}_offset"] = o
        extra.update(cap_k1=a.cap_k1, parity_seed=a.parity_seed)
    if a.ramp_k0:
        if not a.cap_k1:
            offs = gated.parity_offsets(N, np.random.default_rng(a.parity_seed))
            z, mz = gated.to_parity(x, m, mx, offs)
            assert np.array_equal(gated.from_parity(z, m, offs), x), "Cst does not decode to the source"
            bases.append(("Cst", z, mz))
            extra.update(Cst_offset=offs, parity_seed=a.parity_seed)
        depth = gated.ramp_depth(N, a.ramp_k0, a.ramp_dmax)
        roffs = gated.ramp_offsets(N, depth, np.random.default_rng([a.parity_seed, 2]))
        z, mz = gated.to_parity_multi(x, m, mx, roffs)
        assert np.array_equal(gated.from_parity_multi(z, m, roffs), x), "R does not decode to the source"
        bases.append(("R", z, mz))
        extra.update(R_offset=roffs, R_depth=depth, ramp_k0=a.ramp_k0, ramp_dmax=a.ramp_dmax)
    splits = np.cumsum([0, a.ntrain, a.nval, a.ntest])
    data = {}
    for name, lo, hi in zip(("train", "val", "test"), splits[:-1], splits[1:]):
        for b, v, mv in bases:
            data[f"{b}_{name}"] = (v[lo:hi] < 0).astype(np.int8)
            data[f"{b}_{name}_m"] = mv[lo:hi]
            data[f"{b}_{name}_c"] = m[lo:hi].astype(np.uint8)
    floor = gated.true_entropy_by_pos(q, a.eps, a.T)
    np.savez_compressed(a.out, T=a.T, gamma=a.gamma, Q=a.Q, eps=a.eps, q=q, sigma=sigma, seed=a.seed,
                        floor_by_pos=floor, **extra, **data)
    emp = np.mean(gated.hbin((1 - mx[splits[2]:]) / 2))
    print(f"wrote {a.out}: gamma={a.gamma} T={a.T} Q={a.Q} eps={a.eps}; exact floor {floor.mean():.5f} bits/token, "
          f"test-set mean of exact conditional entropies {emp:.5f}")


if __name__ == "__main__":
    main()
