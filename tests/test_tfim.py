"""Check the free-fermion sampler against exact diagonalization of periodic chains."""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact_sources import tfim


def ed_ground_state(L, h):
    dim = 1 << L
    s = np.arange(dim)
    diag = np.zeros(dim)
    for i in range(L):
        j = (i + 1) % L
        diag -= (1 - 2 * ((s >> i) & 1)) * (1 - 2 * ((s >> j) & 1))
    rows, cols, vals = [s], [s], [diag]
    for i in range(L):
        rows.append(s)
        cols.append(s ^ (1 << i))
        vals.append(np.full(dim, -h))
    H = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim)).tocsr()
    _, v = eigsh(H, k=1, which="SA", tol=0)
    return v[:, 0]


def basis_probs(psi, L, basis):
    a = psi.reshape((2,) * L)
    if basis == "X":
        Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        for ax in range(L):
            a = np.moveaxis(np.tensordot(Hd, a, axes=([1], [ax])), 0, ax)
    # reshape index order: axis 0 is the highest bit, i.e. site L-1
    p = np.abs(a) ** 2
    return np.transpose(p, axes=range(L - 1, -1, -1))  # p[x_0, x_1, ...]


def marginal(p, n):
    return p.sum(axis=tuple(range(n, p.ndim)))


def to_pm(bits):
    return 1 - 2 * np.array(bits, dtype=float)  # bit 0 -> +1


def main():
    rng = np.random.default_rng(0)
    L, n = 12, 6
    worst = {}
    for h in (1.0, 1.5, 0.7):
        psi = ed_ground_state(L, h)
        pX = marginal(basis_probs(psi, L, "X"), n)
        pZ = marginal(basis_probs(psi, L, "Z"), n + 1)
        Gam = tfim.window_gamma(tfim.g_coeffs(h, n, L=L), n)
        err = max(abs(tfim.window_prob(Gam, to_pm(b)) - pX[b]) for b in itertools.product((0, 1), repeat=n))
        worst[f"X window, h={h}"] = err

        Gd = tfim.window_gamma(tfim.g_coeffs(1 / h, n, L=L), n)
        zerr = 0.0
        for b in itertools.product((0, 1), repeat=n + 1):
            z = to_pm(b)
            w = z[:-1] * z[1:]
            zerr = max(zerr, abs(0.5 * tfim.window_prob(Gd, w) - pZ[b]))
        worst[f"Z window via duality, h={h}"] = zerr

        x, m = tfim.sample_x(Gam, 200, rng)
        logp_chain = np.sum(np.log((1 + x * m) / 2), axis=1)
        logp_exact = np.log([tfim.window_prob(Gam, xi) for xi in x])
        worst[f"sampler chain rule, h={h}"] = np.max(np.abs(logp_chain - logp_exact))

    for k, v in worst.items():
        print(f"{k:34s} max abs error {v:.2e}")

    # empirical frequencies vs exact (infinite chain, h=1, T=4)
    Gam = tfim.window_gamma(tfim.g_coeffs(1.0, 4), 4)
    x, _ = tfim.sample_x(Gam, 200000, rng)
    keys, counts = np.unique((x < 0).astype(int) @ (1 << np.arange(4)), return_counts=True)
    exact = np.array([tfim.window_prob(Gam, to_pm([(k >> i) & 1 for i in range(4)])) for k in keys])
    z = (counts / len(x) - exact) / np.sqrt(exact * (1 - exact) / len(x))
    print(f"sampler frequencies, infinite chain: max |z| = {np.max(np.abs(z)):.2f} over {len(keys)} strings")
    print(f"<X> at h=1, infinite chain: {tfim.g_coeffs(1.0, 0)[0]:.8f} (2/pi = {2/np.pi:.8f})")


def check_one_shot_conditioning():
    rng = np.random.default_rng(1)
    for h in (1.0, 1.5):
        T = 40
        Gam = tfim.window_gamma(tfim.g_coeffs(h, T), T)
        x, m = tfim.sample_x(Gam, 50, rng)
        m1 = tfim.conditional_mean_last(Gam, x[:, :-1])
        print(f"one-shot vs sequential conditional, h={h}: max abs diff {np.max(np.abs(m1 - m[:, -1])):.2e}")


if __name__ == "__main__":
    main()
    check_one_shot_conditioning()
