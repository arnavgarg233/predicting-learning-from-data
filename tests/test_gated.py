"""Exactness checks for the gated-copy source (exact_sources/gated.py) by brute-force enumeration.

Run: python tests/test_gated.py   (also collected by pytest)
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as _chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact_sources import gated

T, GAMMA, Q, EPS = 5, 1.0, 0.6, 0.15
N = T - 1


def setup():
    rng = np.random.default_rng(3)
    q = gated.make_q(GAMMA, N, Q)
    sigma = gated.make_sigma(N, rng)
    return q, sigma


def enumerate_joint(q, sigma):
    """All (observed markers, x) with their probability, from the generative definition: a full marker
    k in {0..N}; k = 0 or k > t gives a fair coin and observed marker 0."""
    pk = np.concatenate([[1 - q[1:].sum()], q[1:]])
    table = {}
    for full in itertools.product(range(N + 1), repeat=T):
        pm = np.prod([pk[k] for k in full])
        obs = tuple(k if 0 < k <= t else 0 for t, k in enumerate(full))
        for xs in itertools.product((1, -1), repeat=T):
            p = pm
            for t, k in enumerate(obs):
                if k == 0:
                    p *= 0.5
                else:
                    p *= (1 - EPS) if xs[t] == sigma[k] * xs[t - k] else EPS
            table[(obs, xs)] = table.get((obs, xs), 0.0) + p
    return table


def brute_cond(table, t, n, recode=None):
    """Brute-force E[v_t | v_{t-n..t-1}, m_0..m_t] for every (obs, x) key; v = x, or y if recode."""
    num, den = {}, {}
    for (obs, xs), p in table.items():
        v = xs if recode is None else recode(xs)
        key = (obs[: t + 1], v[max(t - n, 0):t])
        num[key] = num.get(key, 0.0) + p * v[t]
        den[key] = den.get(key, 0.0) + p
    return {k: num[k] / den[k] for k in num}


def diff(xs):
    return (xs[0],) + tuple(xs[t] * xs[t - 1] for t in range(1, len(xs)))


def test_full_context_conditionals_and_recoding():
    q, sigma = setup()
    table = enumerate_joint(q, sigma)
    assert abs(sum(table.values()) - 1) < 1e-12
    keys = list(table)
    m = np.array([k[0] for k in keys])
    x = np.array([k[1] for k in keys], dtype=float)
    mx = gated.cond_mean_x(x, m, sigma, EPS)
    y, my = gated.to_diff(x, mx)
    assert np.array_equal(gated.from_diff(y), x)
    err_x = err_y = 0.0
    for t in range(T):
        bx = brute_cond(table, t, t)
        by = brute_cond(table, t, t, recode=diff)
        for i, (obs, xs) in enumerate(keys):
            err_x = max(err_x, abs(bx[(obs[: t + 1], xs[:t])] - mx[i, t]))
            err_y = max(err_y, abs(by[(obs[: t + 1], diff(xs)[:t])] - my[i, t]))
    print(f"full-context conditional mean, C encoding: max abs error {err_x:.1e}")
    print(f"full-context conditional mean, D encoding: max abs error {err_y:.1e}")
    assert err_x < 1e-12 and err_y < 1e-12
    # same information: per-token exact conditional entropies coincide position by position
    assert np.allclose(np.abs(mx), np.abs(my))
    # floor: exact expected conditional entropy per position
    probs = np.array([table[k] for k in keys])
    H = (probs[:, None] * gated.hbin((1 - mx) / 2)).sum(0)
    err_h = np.max(np.abs(H - gated.true_entropy_by_pos(q, EPS, T)))
    print(f"floor per position vs closed form: max abs error {err_h:.1e}")
    assert err_h < 1e-12


def test_truncated_context_bp():
    q, sigma = setup()
    table = enumerate_joint(q, sigma)
    worst = 0.0
    for t in range(1, T):
        for n in range(0, t + 1):
            b = brute_cond(table, t, n)
            for (obs, xs) in table:
                v = gated.ctx_mean(np.array(xs, float), np.array(obs), t, n, sigma, EPS)
                worst = max(worst, abs(v - b[(obs[: t + 1], xs[max(t - n, 0):t])]))
    print(f"truncated-context conditional (belief propagation) vs brute force: max abs error {worst:.1e}")
    assert worst < 1e-12


def test_correlators():
    q, sigma = setup()
    table = enumerate_joint(q, sigma)
    M2 = gated.two_point(q, sigma, EPS, T)
    keys = list(table)
    probs = np.array([table[k] for k in keys])
    x = np.array([k[1] for k in keys], dtype=float)
    E2 = np.einsum("k,ka,kb->ab", probs, x, x)
    e2 = np.max(np.abs(E2 - M2))
    y = x.copy()
    y[:, 1:] = x[:, 1:] * x[:, :-1]
    c4 = gated.four_point_diff_corr(q, sigma, EPS, T)
    e4 = 0.0
    for n, v in c4.items():
        t = np.arange(1, T - n)
        yy = np.einsum("k,kt->t", probs, y[:, t] * y[:, t + n])
        ey = np.einsum("k,kt->t", probs, y[:, t]) * np.einsum("k,kt->t", probs, y[:, t + n])
        e4 = max(e4, abs(np.mean(yy - ey) - v))
    print(f"two-point (C) and connected y-pair (D) correlators vs brute force: {e2:.1e}, {e4:.1e}")
    assert e2 < 1e-12 and e4 < 1e-12


def chi2_sf(x, dof):
    return float(_chi2.sf(x, dof))


def test_sampler_frequencies():
    q, sigma = setup()
    table = enumerate_joint(q, sigma)
    rng = np.random.default_rng(5)
    n = 400000
    x, m, mx = gated.sample(q, sigma, EPS, T, n, rng)
    keys = [tuple(r) for r in np.concatenate([m, x], 1).astype(int)]
    uniq, counts = np.unique(np.array(keys), axis=0, return_counts=True)
    exact = np.array([table[(tuple(u[:T]), tuple(u[T:]))] for u in uniq])
    e = exact * n
    big = e > 20  # Pearson chi-square over cells with enough expected counts
    chi2 = np.sum((counts[big] - e[big]) ** 2 / e[big])
    p = chi2_sf(chi2, big.sum())
    print(f"sampler vs exact joint: chi2/dof = {chi2 / big.sum():.3f} over {big.sum()} cells, p = {p:.3f}; "
          f"sampled mass {exact.sum():.4f}")
    assert p > 1e-3 and exact.sum() > 0.999
    assert np.allclose(mx, gated.cond_mean_x(x, m, sigma, EPS))


if __name__ == "__main__":
    test_full_context_conditionals_and_recoding()
    test_truncated_context_bp()
    test_correlators()
    test_sampler_frequencies()
    print("all gated-source checks passed")
