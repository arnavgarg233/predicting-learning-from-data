"""Exactness of k-spin block-token distributions (exact_sources/blocks.py) for TFIM X and Z windows.

Reference: exact string probabilities tfim.window_prob (Pfaffian/determinant), independent of the
sequential conditioning used by blocks.py. Run: python tests/test_blocks.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact_sources import blocks, tfim

T = 8


def pm(bits):
    return 1 - 2 * np.array(bits, dtype=float)


def test_block_probs_x_and_z():
    rng = np.random.default_rng(0)
    Gam = tfim.window_gamma(tfim.g_coeffs(1.0, T), T)
    w, mw = tfim.sample_x(Gam, 40, rng)
    z = tfim.x_to_z(w[:, :-1], rng)
    worst = {}
    for k in (1, 2, 4):
        Px = blocks.block_probs_x(Gam, w, k, batch=16)
        Pz = blocks.block_probs_z(Gam, w, z, k, batch=16)
        ex = ez = ec = 0.0
        for i in range(len(w)):
            for j in range(T // k):
                t0 = j * k
                for b, bits in enumerate(itertools.product((0, 1), repeat=k)):
                    xs = np.concatenate([w[i, :t0], pm(bits)])
                    den = tfim.window_prob(tfim.window_gamma(tfim.g_coeffs(1.0, t0), t0), w[i, :t0]) if t0 else 1.0
                    G = tfim.window_gamma(tfim.g_coeffs(1.0, t0 + k), t0 + k)
                    ex = max(ex, abs(tfim.window_prob(G, xs) / den - Px[i, j, b]))
                    zs = np.concatenate([z[i, :t0], pm(bits)])
                    pz = lambda s: 0.5 * (tfim.window_prob(tfim.window_gamma(tfim.g_coeffs(1.0, len(s) - 1), len(s) - 1),
                                                           s[1:] * s[:-1]) if len(s) > 1 else 1.0)
                    ez = max(ez, abs(pz(zs) / (pz(z[i, :t0]) if t0 else 1.0) - Pz[i, j, b]))
        # consistency with the stored single-spin conditionals along the realized path
        bx = blocks.pack((w < 0).astype(np.int8), k)
        lp = np.log(np.take_along_axis(Px, bx[..., None].astype(int), -1)[..., 0]).sum(1)
        ec = np.max(np.abs(lp - np.log((1 + w * mw) / 2).sum(1)))
        worst[k] = (ex, ez, ec)
        print(f"k={k}: block probs vs exact string probabilities X {ex:.1e}, Z {ez:.1e}; "
              f"realized-path log-lik vs sampler {ec:.1e}; rows sum to 1: "
              f"{np.allclose(Px.sum(-1), 1) and np.allclose(Pz.sum(-1), 1)}")
        assert ex < 1e-10 and ez < 1e-10 and ec < 1e-10
        assert np.allclose(Px.sum(-1), 1) and np.allclose(Pz.sum(-1), 1)


if __name__ == "__main__":
    test_block_probs_x_and_z()
    print("all block-token checks passed")
