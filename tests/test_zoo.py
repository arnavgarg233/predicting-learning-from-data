"""Exactness checks for the zoo's switching source (exact_sources/zoo.py) by brute-force enumeration of hidden paths,
and for the gated q shapes.

Run: python tests/test_zoo.py   (also collected by pytest)
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact_sources import gated, zoo

T = 5
SRC = zoo.Switching(r=[0.3, 0.05], w=[0.5, 0.3], eps=0.1)


def joint():
    """p(x_0..x_{T-1}) for all 2^T strings, summing over every hidden path and switch from the generative definition."""
    J = SRC.J
    states = list(itertools.product([1.0, -1.0], repeat=J))
    p = {}
    for xs in itertools.product([1.0, -1.0], repeat=T):
        tot = 0.0
        for path in itertools.product(states, repeat=T):
            pr = 0.5 ** J
            for t in range(T):
                if t:
                    for j in range(J):
                        pr *= SRC.r[j] if path[t][j] != path[t - 1][j] else 1 - SRC.r[j]
                e = (1 - SRC.w.sum()) * 0.5
                for j in range(J):
                    e += SRC.w[j] * ((1 - SRC.eps) if path[t][j] == xs[t] else SRC.eps)
                pr *= e
            tot += pr
        p[xs] = tot
    return p


P = joint()


def marg(prefix):
    return sum(v for k, v in P.items() if k[:len(prefix)] == prefix)


def test_normalised():
    assert abs(sum(P.values()) - 1) < 1e-12


def test_cond_means_exact():
    xs = np.array(list(P.keys()))
    m = SRC.cond_means(xs)
    for i, x in enumerate(map(tuple, xs)):
        for t in range(T):
            pp, pm = marg(x[:t] + (1.0,)), marg(x[:t] + (-1.0,))
            assert abs(m[i, t] - (pp - pm) / (pp + pm)) < 1e-12


def test_mean_after_context_only():
    """E[x_{T-1} | last n tokens] with the earlier tokens marginalised (stationary source)."""
    xs = np.array(list(P.keys()))
    for n in range(0, T):
        mn = SRC.mean_after(xs[:, T - 1 - n:T - 1])
        for i, x in enumerate(map(tuple, xs)):
            ctx = x[T - 1 - n:T - 1]
            num = sum(v * k[T - 1] for k, v in P.items() if k[T - 1 - n:T - 1] == ctx)
            den = sum(v for k, v in P.items() if k[T - 1 - n:T - 1] == ctx)
            assert abs(mn[i] - num / den) < 1e-12


def test_pair_corr():
    c = SRC.pair_corr(T)
    for n in range(1, T):
        ex = sum(v * k[0] * k[n] for k, v in P.items())
        assert abs(c[n] - ex) < 1e-12


def test_q_shapes():
    N, Q = 63, 0.5
    assert np.array_equal(gated.make_q_shape("power", N, Q, gamma=0.7), gated.make_q(0.7, N, Q))
    for sh in ("geo", "step", "mix"):
        q = gated.make_q_shape(sh, N, Q, gamma=2.0, lag=16, band=(48, 63), Qband=0.05)
        assert q[0] == 0 and abs(q.sum() - Q) < 1e-12 and np.all(q >= 0)
    assert np.count_nonzero(gated.make_q_shape("step", N, Q, lag=16)) == 1


def test_tail_power_law_limit():
    """Far from the window edge the information rule's tail exponent is gamma/(1+gamma) (Hutter 2021)."""
    Tl, g = 1 << 15, 1.0
    q = gated.make_q(g, Tl - 1, 0.5)
    h = 1 - zoo.H_EPS01 * np.cumsum(q)[:Tl]
    P = 2.0 ** np.linspace(16, 22, 25)
    tc = zoo.tail_curves(h, np.full(Tl, np.nan), np.full(Tl, np.nan), Tl, P)
    assert abs(zoo.slope(P, tc["info"]) - g / (1 + g)) < 0.03


def test_gated_indirect_matches_reference_bp():
    """Fast BP (zoo.gated_indirect) equals exact_sources.gated.ctx_mean (itself checked against brute force)."""
    rng = np.random.default_rng(1)
    Tg, eps = 48, 0.1
    q = gated.make_q(1.0, Tg - 1, 0.5)
    sigma = gated.make_sigma(Tg - 1, rng)
    x, m, _ = gated.sample(q, sigma, eps, Tg, 4, rng)
    ns = [0, 1, 2, 4, 8]
    ind = zoo.gated_indirect(x, m, sigma, eps, ns)
    for w in range(4):
        for i, n in enumerate(ns):
            g = sum(1 - zoo.hbin((1 + gated.ctx_mean(x[w], m[w], t, n, sigma, eps)) / 2) for t in range(n, Tg) if m[w, t] > n)
            assert abs(ind[w, i] - g / (Tg - n)) < 1e-12
    xw, mw = np.ones(3), np.array([0, 1, 2])  # m_t = 2, m_{t-1} = 1, one token of context: |mean| = (1 - 2 eps)^2
    assert abs(gated.ctx_mean(xw, mw, 2, 1, np.ones(4), eps) - 0.64) < 1e-12


def test_unknown_correlations_stay_unknown():
    Tl = 16
    h = np.linspace(1, 0.8, Tl)
    absC = np.full(Tl, 0.1)
    absC[10] = np.nan
    tc = zoo.tail_curves(h, absC, np.full(Tl, 0.5), Tl, [1e3, 1e4])
    assert np.all(np.isnan(tc["cag"])) and np.all(np.isfinite(tc["info"]))


def test_compress_null_calibration():
    rng = np.random.default_rng(0)
    tok = rng.integers(0, 2, (64, 256)).astype(np.int8)
    side = rng.integers(0, 256, (64, 256)).astype(np.uint8)
    cal, raw, null = zoo.compress_bits(tok, side, np.random.default_rng(1))
    assert raw > 1.3 and abs(cal - 1) < 0.1  # the codec overhead found earlier is removed by the null stream


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("ok", name)
