"""Gated-copy source with a visible marker.

Each window starts a fresh sequence (like a document). At position t an i.i.d. marker picks a
distance k with P(k) = q_k = c k^-(1+gamma) for k = 1..N, or no copy with P = 1 - sum q. If k <= t the
token copies the token k steps back through a binary symmetric channel with flip probability eps and a
fixed sign sigma_k:  x_t = sigma_k x_{t-k} w.p. 1 - eps, else -sigma_k x_{t-k}. Otherwise (no copy, or
k > t) x_t is a fair coin and the observed marker is 0. The marker of position t is shown to the
learner together with the past tokens, so

    E[x_t | x_<t, markers] = (1 - 2 eps) sigma_k x_{t-k}   (k = m_t >= 1), else 0,

and the per-distance information is exactly I_k = q_k (1 - h(eps)) bits: the classes {sigma} satisfy
A1-A3 of the theorem, so H_n - H_inf ~ n^-gamma and the information rule gives gamma/(1+gamma).

Two encodings of the same stream (same markers):
  C ("copy"): the tokens x. Pair correlation C(n) ~ q_n, beta = 1 + gamma.
  D ("diff"): y_0 = x_0, y_t = x_t x_{t-1}. A causal bijection (x_t = prod_{s<=t} y_s). The copy at
              distance k becomes a parity of the k-1 tokens y_{t-k+1..t-1}; C(n) ~ q_n^2, beta = 2 + 2 gamma.
Optional capped-parity recodings (to_parity): Cst adds one parity bit at every distance k >= 2, B only at k >= k1.
Values are +-1; stored tokens are (value < 0), as in make_data.py.
"""
import numpy as np


def hbin(e):
    e = np.clip(e, 1e-300, 1)
    f = np.clip(1 - e, 1e-300, 1)
    return -(e * np.log2(e) + f * np.log2(f))


def make_q(gamma, N, Q):
    """q_k for k = 1..N (index 0 unused, = 0), normalised so that sum q = Q."""
    q = np.zeros(N + 1)
    q[1:] = np.arange(1, N + 1, dtype=float) ** -(1.0 + gamma)
    return q * (Q / q.sum())


def make_q_shape(shape, N, Q, gamma=1.0, rho=0.9, lag=16, band=(48, 63), Qband=0.05):
    """Other memory shapes for the exact-source zoo; sum q = Q.
    power: make_q (unchanged); geo: q_k ~ rho^k; step: all copies at one distance `lag`;
    mix: power law with total Q - Qband plus a uniform band of distances band[0]..band[1] with total Qband."""
    if shape == "power":
        return make_q(gamma, N, Q)
    k = np.arange(1, N + 1, dtype=float)
    q = np.zeros(N + 1)
    if shape == "mix":
        q = make_q(gamma, N, Q - Qband)
        b = (k >= band[0]) & (k <= band[1])
        q[1:] += Qband * b / b.sum()
        return q
    if shape == "geo":
        w = rho ** k
    elif shape == "step":
        w = (k == lag).astype(float)
    else:
        raise ValueError(shape)
    q[1:] = w * (Q / w.sum())
    return q


def make_sigma(N, rng):
    s = np.ones(N + 1)
    s[1:] = np.where(rng.random(N) < 0.5, 1.0, -1.0)
    return s


def sample_markers(q, T, n, rng):
    """Observed markers (n, T): 0 = fair coin (no copy, or a copy that would reach before the window)."""
    probs = np.concatenate([[1 - q[1:].sum()], q[1:]])
    m = rng.choice(len(q), size=(n, T), p=probs)
    t = np.arange(T)[None, :]
    return np.where(m > t, 0, m)


def cond_mean_x(x, m, sigma, eps):
    """Exact E[x_t | x_<t, m_<=t] for every position."""
    n, T = x.shape
    t = np.arange(T)[None, :]
    par = np.clip(t - m, 0, None)
    xp = np.take_along_axis(x, par, axis=1)
    return np.where(m > 0, (1 - 2 * eps) * sigma[m] * xp, 0.0)


def sample(q, sigma, eps, T, n, rng):
    """x (n, T) in +-1, markers m (n, T), exact conditional mean mx (n, T)."""
    m = sample_markers(q, T, n, rng)
    x = np.empty((n, T))
    flip = rng.random((n, T)) < eps
    coin = np.where(rng.random((n, T)) < 0.5, 1.0, -1.0)
    rows = np.arange(n)
    for t in range(T):
        k = m[:, t]
        par = x[rows, np.clip(t - k, 0, None)] if t > 0 else np.zeros(n)
        cp = sigma[k] * par * np.where(flip[:, t], -1.0, 1.0)
        x[:, t] = np.where(k > 0, cp, coin[:, t])
    return x, m, cond_mean_x(x, m, sigma, eps)


def to_diff(x, mx):
    """Recode to y_t = x_t x_{t-1} (y_0 = x_0) with exact conditional means E[y_t | y_<t, m]."""
    y = x.copy()
    y[:, 1:] = x[:, 1:] * x[:, :-1]
    my = mx.copy()
    my[:, 1:] = mx[:, 1:] * x[:, :-1]
    return y, my


def from_diff(y):
    return np.cumprod(y, axis=1)


def parity_offsets(N, rng):
    """One extra-bit offset j_k per distance, uniform in 1..k-1 (k >= 2); j_1 = 0 (no room). Index 0 unused."""
    j = np.zeros(N + 1, dtype=np.int64)
    for k in range(2, N + 1):
        j[k] = rng.integers(1, k)
    return j


def to_parity(x, m, mx, offs):
    """Lossless recoding with an extra parity bit at chosen distances.

    offs[k] = 0: a copy event at distance k stays a copy in the new tokens, z_t = (x_t x_s) z_s with s = t - k.
    offs[k] = j in 1..k-1: z_t = (x_t x_s) z_s z_{t-j}, a parity of two visible tokens. Fair coins are unchanged.
    x_t x_s = sigma_k times the flip noise, so E[z_t | z_<t, m] = (1 - 2 eps) sigma_k z_s z_{t-j} = mx_t x_t z_t: every
    conditional mean keeps its magnitude, and the decode x_t = z_t z_s x_s z_{t-j} is causal given the markers."""
    n, T = x.shape
    z = x.copy()
    rows = np.arange(n)
    for t in range(1, T):
        k = m[:, t]
        s = np.clip(t - k, 0, None)
        j = offs[k]
        extra = np.where(j > 0, z[rows, np.clip(t - j, 0, None)], 1.0)
        z[:, t] = np.where(k > 0, x[:, t] * x[rows, s] * z[rows, s] * extra, x[:, t])
    return z, mx * x * z


def ramp_depth(N, k0, dmax):
    """Extra parity bits per distance: d(k) = clip(floor(log2(k / k0)) + 1, 0, dmax) for k >= k0, else 0."""
    k = np.arange(N + 1)
    d = np.where(k >= k0, np.floor(np.log2(np.maximum(k, 1) / k0)) + 1, 0)
    return np.clip(d, 0, dmax).astype(np.int64)


def ramp_offsets(N, depth, rng):
    """(N+1, max depth) offsets: d(k) distinct offsets uniform in 1..k-1 for distance k, 0 = unused."""
    offs = np.zeros((N + 1, max(int(depth.max()), 1)), dtype=np.int64)
    for k in range(2, N + 1):
        if depth[k]:
            assert depth[k] <= k - 1, f"no room for {depth[k]} bits at k = {k}"
            offs[k, : depth[k]] = np.sort(rng.choice(k - 1, size=depth[k], replace=False) + 1)
    return offs


def _extra(z, rows, t, offs_k):
    """Product of z[t - j] over the nonzero offsets in offs_k (n, ncol); 1 where unused."""
    e = np.ones(len(rows))
    for c in range(offs_k.shape[1]):
        j = offs_k[:, c]
        e = e * np.where(j > 0, z[rows, np.clip(t - j, 0, None)], 1.0)
    return e


def to_parity_multi(x, m, mx, offs):
    """to_parity with several extra bits per distance (offs from ramp_offsets): z_t = (x_t x_s) z_s prod_j z_{t-j}."""
    n, T = x.shape
    z = x.copy()
    rows = np.arange(n)
    for t in range(1, T):
        k = m[:, t]
        s = np.clip(t - k, 0, None)
        z[:, t] = np.where(k > 0, x[:, t] * x[rows, s] * z[rows, s] * _extra(z, rows, t, offs[k]), x[:, t])
    return z, mx * x * z


def from_parity_multi(z, m, offs):
    n, T = z.shape
    x = z.copy()
    rows = np.arange(n)
    for t in range(1, T):
        k = m[:, t]
        s = np.clip(t - k, 0, None)
        x[:, t] = np.where(k > 0, z[:, t] * z[rows, s] * x[rows, s] * _extra(z, rows, t, offs[k]), z[:, t])
    return x


def from_parity(z, m, offs):
    """Inverse of to_parity (needs only the new tokens and the markers)."""
    n, T = z.shape
    x = z.copy()
    rows = np.arange(n)
    for t in range(1, T):
        k = m[:, t]
        s = np.clip(t - k, 0, None)
        j = offs[k]
        extra = np.where(j > 0, z[rows, np.clip(t - j, 0, None)], 1.0)
        x[:, t] = np.where(k > 0, z[:, t] * z[rows, s] * x[rows, s] * extra, z[:, t])
    return x


def true_entropy_by_pos(q, eps, T):
    """Exact E[H(x_t | x_<t, m)] per position, bits: the floor of both encodings."""
    cq = np.cumsum(q)[:T]  # cq[t] = sum_{k<=t} q_k
    return 1 - (1 - hbin(eps)) * cq


def two_point(q, sigma, eps, T):
    """Exact E[x_a x_b] for the windowed process (markers marginalised), T x T."""
    a = (1 - 2 * eps) * q * sigma
    M = np.eye(T)
    for d in range(1, T):
        k = np.arange(1, min(d, len(q) - 1) + 1)
        M[:d, d] = M[:d, d - k] @ a[k]
        M[d, :d] = M[:d, d]
    return M


def four_point_diff_corr(q, sigma, eps, T):
    """Exact connected <y_t y_{t+n}>, averaged over t, for the D encoding of the windowed process.

    Uses E[x_d | x_<d] = sum_k (1-2eps) q_k sigma_k x_{d-k} (markers marginalised), filled by the
    largest index. Memory T^4 doubles: keep T <= ~48."""
    a = (1 - 2 * eps) * q * sigma
    M2 = two_point(q, sigma, eps, T)
    M = np.zeros((T, T, T, T))
    for d in range(T):
        if d:
            k = np.arange(1, min(d, len(q) - 1) + 1)
            S = np.tensordot(M[:d, :d, :d][..., d - k], a[k], axes=([3], [0]))  # a,b,c < d, d once
            M[:d, :d, :d, d] = S
            M[:d, :d, d, :d] = S
            M[:d, d, :d, :d] = S
            M[d, :d, :d, :d] = S
            P = M2[:d, :d]  # d twice
            for (i, j) in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
                idx = [slice(None, d)] * 4
                idx[i] = d
                idx[j] = d
                M[tuple(idx)] = P
            v = M2[:d, d]  # d three times
            M[:d, d, d, d] = v
            M[d, :d, d, d] = v
            M[d, d, :d, d] = v
            M[d, d, d, :d] = v
        M[d, d, d, d] = 1.0
    # y_t = x_{t-1} x_t (t >= 1); E[y_t] = M2[t-1, t]
    out = {}
    for n in range(1, T - 1):
        t = np.arange(1, T - n)
        yy = M[t - 1, t, t + n - 1, t + n]
        ey = M2[t - 1, t] * M2[t + n - 1, t + n]
        out[n] = float(np.mean(yy - ey))
    return out


def _bp_llr(root, excl, par, kids, sgn, obs, xv, J):
    """LLR of x_root from the tree component, ignoring the branch through `excl`.
    par: parent index or -1; kids: children lists (already restricted to the past);
    sgn: edge sign of node u to its parent; obs: observed mask; xv: values; J = 2 atanh(1-2eps)."""
    L = 0.0
    nbrs = [(c, sgn[c]) for c in kids[root] if c != excl]
    if par[root] >= 0 and par[root] != excl:
        nbrs.append((par[root], sgn[root]))
    for v, s in nbrs:
        if obs[v]:
            L += s * xv[v] * J
        else:
            Lv = _bp_llr(v, root, par, kids, sgn, obs, xv, J)
            if Lv != 0.0:
                L += s * 2 * np.arctanh(np.tanh(J / 2) * np.tanh(Lv / 2))
    return L


def ctx_mean(xw, mw, t, n, sigma, eps):
    """Exact E[x_t | x_{t-n..t-1}, m_0..m_t] for one window (values xw, observed markers mw)."""
    k = mw[t]
    if k == 0:
        return 0.0
    if k <= n:
        return (1 - 2 * eps) * sigma[k] * xw[t - k]
    J = 2 * np.arctanh(1 - 2 * eps)
    par = np.where(mw[:t] > 0, np.arange(t) - mw[:t], -1)
    sgn = np.where(mw[:t] > 0, sigma[mw[:t]], 0.0)
    kids = [[] for _ in range(t)]
    for u in range(t):
        if par[u] >= 0:
            kids[par[u]].append(u)
    obs = np.zeros(t, bool)
    obs[max(t - n, 0):t] = True
    L = _bp_llr(t - k, -1, par, kids, sgn, obs, xw, J)
    return (1 - 2 * eps) * sigma[k] * np.tanh(L / 2)


def cond_entropy_ctx(x, m, sigma, eps, ns, tmin):
    """Exact H(x_t | x_{t-n..t-1}, m_0..m_t) in bits, averaged over the given windows and positions
    t >= tmin, for each n in ns (n <= tmin). Belief propagation on the copy forest (a zero-field tree
    Ising model once the markers are known); exact per window, Monte Carlo only over windows."""
    W, T = x.shape
    out = np.zeros(len(ns))
    for w in range(W):
        for t in range(tmin, T):
            for i, n in enumerate(ns):
                out[i] += hbin((1 - ctx_mean(x[w], m[w], t, n, sigma, eps)) / 2)
    return out / (W * (T - tmin))
