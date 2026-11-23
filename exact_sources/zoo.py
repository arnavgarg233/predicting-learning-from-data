"""Exact-source zoo: a hidden-Markov switching source, information curves,
pair correlations, and the data-only predictions of how much and how fast a network learns.

Conventions: tokens are +-1 values (stored as value < 0, as everywhere in exact_sources); h[n] = H(x_t | n previous tokens) in
bits for n = 0..T-1; absC[n] = |connected <x_t x_{t+n}>| for n = 1..T-1 (index 0 unused).
"""
import lzma

import numpy as np

H_EPS01 = 1 + 0.1 * np.log2(0.1) + 0.9 * np.log2(0.9)  # 1 - h(0.1): bits per gated-copy event at eps = 0.1
K_EVENTS = 210.0  # events per distance at the plateau
B_STAR = K_EVENTS * H_EPS01  # the same price in bits of training information per distance (~111.5)
SOFT_SLOPE = 2.79  # logistic slope of P(learned) in ln(events), shared by all dial runs (dial_cost_fit.py)


def hbin(p):
    p = np.clip(p, 1e-300, 1)
    q = np.clip(1 - p, 1e-300, 1)
    return -(p * np.log2(p) + q * np.log2(q))


def kl_mean_bits(m_p, m_q):
    """KL(p || q) in bits for binary laws given by conditional means E[x] in [-1, 1]."""
    p = np.clip((1 + m_p) / 2, 1e-15, 1 - 1e-15)
    q = np.clip((1 + m_q) / 2, 1e-15, 1 - 1e-15)
    return p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q))


# ------------------------------------------------------------------ multi-timescale switching source (hidden Markov)
class Switching:
    """J hidden telegraph bits s_j in +-1, each flipped with probability r_j per step (stationary start: uniform).
    At every step a hidden switch picks channel j with probability w_j, or a fair coin with probability 1 - sum w;
    the token is s_j flipped with probability eps. The hidden state has 2^J values, so every conditional law
    p(x_t | any stretch of past tokens) follows exactly from the forward algorithm."""

    def __init__(self, r, w, eps):
        self.r, self.w, self.eps = np.asarray(r, float), np.asarray(w, float), float(eps)
        self.J = len(self.r)
        bits = (np.arange(2 ** self.J)[:, None] >> np.arange(self.J)[None, :]) & 1
        self.S = 1.0 - 2.0 * bits  # (2^J, J): state index i has s_j = -1 where bit j of i is set
        self.mu = (1 - 2 * self.eps) * self.S @ self.w  # E[x_t | state]

    def sample(self, n, T, rng):
        J = self.J
        s = np.where(rng.random((n, J)) < 0.5, 1.0, -1.0)
        cw = np.cumsum(self.w)
        x = np.empty((n, T))
        rows = np.arange(n)
        for t in range(T):
            if t:
                s = np.where(rng.random((n, J)) < self.r, -s, s)
            g = np.searchsorted(cw, rng.random(n), side="right")  # g == J: fair coin
            flip = np.where(rng.random(n) < self.eps, -1.0, 1.0)
            coin = np.where(rng.random(n) < 0.5, 1.0, -1.0)
            x[:, t] = np.where(g < J, s[rows, np.minimum(g, J - 1)] * flip, coin)
        return x

    def _advance(self, b):
        """Belief over the state at t-1 -> prior at t (independent flips per channel)."""
        B, J = b.shape[0], self.J
        b = b.reshape((B,) + (2,) * J)
        for j in range(J):
            ax = J - j  # C order: the last axis is bit 0
            b = (1 - self.r[j]) * b + self.r[j] * np.flip(b, axis=ax)
        return b.reshape(B, -1)

    def cond_means(self, x):
        """Exact E[x_t | x_0..x_{t-1}] for every position of every window (window starts in the stationary law)."""
        B, T = x.shape
        b = np.full((B, 2 ** self.J), 2.0 ** -self.J)
        m = np.empty((B, T))
        for t in range(T):
            if t:
                b = self._advance(b)
            m[:, t] = b @ self.mu
            b = b * (1 + x[:, t:t + 1] * self.mu[None, :]) / 2
            b /= b.sum(1, keepdims=True)
        return m

    def mean_after(self, ctx):
        """E[x_next | the context tokens ctx (B, n)] with the filter started in the stationary law n steps back."""
        B, n = ctx.shape
        b = np.full((B, 2 ** self.J), 2.0 ** -self.J)
        for t in range(n):
            if t:
                b = self._advance(b)
            b = b * (1 + ctx[:, t:t + 1] * self.mu[None, :]) / 2
            b /= b.sum(1, keepdims=True)
        if n:
            b = self._advance(b)
        return b @ self.mu

    def pair_corr(self, T):
        """Exact <x_t x_{t+n}> = (1 - 2 eps)^2 sum_j w_j^2 (1 - 2 r_j)^n (the channels and switches are independent)."""
        n = np.arange(T)
        c = (1 - 2 * self.eps) ** 2 * ((self.w ** 2)[None, :] * (1 - 2 * self.r[None, :]) ** n[:, None]).sum(1)
        c[0] = np.nan
        return c


# ------------------------------------------------------------------ information curves from exact conditionals
NGRID = (0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 224)


def grid_for(T):
    g = sorted({n for n in NGRID if n < T - 1} | {T - 2})
    return np.array(g)


def curve_from_kl(ns, Delta, h_ref, T):
    """h[n] for n = 0..T-1 from Delta(n) = h(n) - h(T-1) on a grid. Each Delta(n) is a Monte Carlo mean over windows of
    an exact per-window quantity (sampled, not exact). Delta is forced non-increasing and interpolated log-log between
    grid points (linearly to 0 at n = T-1); interp_check() measures the interpolation error."""
    ns = np.asarray(ns, float)
    D = np.minimum.accumulate(np.maximum(np.asarray(Delta, float), 0.0))
    ns = np.append(ns, T - 1)
    D = np.append(D, 0.0)
    n_all = np.arange(T, dtype=float)
    out = np.interp(n_all, ns, D)  # linear fallback
    pos = D > 0
    if pos.any():
        xs, ys = np.log1p(ns[pos]), np.log(D[pos])
        inside = n_all <= ns[pos].max()
        out[inside] = np.exp(np.interp(np.log1p(n_all[inside]), xs, ys))
    return h_ref + out


def interp_check(ns, Delta, T):
    """Leave-one-grid-point-out: interpolate each interior point from the others; max and median relative error."""
    ns, D = np.asarray(ns), np.minimum.accumulate(np.maximum(np.asarray(Delta, float), 0.0))
    errs = []
    for i in range(1, len(ns) - 1):
        if D[i] <= 1e-6:
            continue
        k = np.arange(len(ns)) != i
        c = curve_from_kl(ns[k], D[k], 0.0, T)
        errs.append(abs(c[ns[i]] - D[i]) / D[i])
    return dict(max_rel=float(max(errs)) if errs else None, median_rel=float(np.median(errs)) if errs else None, points=len(errs))


def boot_curves(ns, rows, h_ref_rows, T, B, rng, base=None):
    """Bootstrap the information curve over statistics windows (rows: windows x grid, per-window Delta or indirect
    information; resampling whole windows keeps the correlation across n). base(n) (exact, e.g. the gated direct
    curve) is added when given: then h = base - curve_from_kl(indirect)."""
    out = []
    W = rows.shape[0]
    for _ in range(B):
        i = rng.integers(0, W, W)
        D = rows[i].mean(0)
        if base is None:
            out.append(curve_from_kl(ns, D, float(h_ref_rows[i].mean()), T))
        else:
            out.append(base - indirect_curve(ns, D, T))
    return np.array(out)


def indirect_curve(ns, ind, T):
    """Indirect information (bits per position) on all n from grid values; linear in n, 0 at n = T - 1."""
    return np.interp(np.arange(T), np.append(ns, T - 1), np.append(np.maximum(ind, 0.0), 0.0))


def _gated_msg(u, frm, s, t, lo, par, kids, sgn, xv, J):
    if lo <= u < t:
        return s * xv[u] * J
    L = _gated_llr(u, frm, t, lo, par, kids, sgn, xv, J)
    return 0.0 if L == 0.0 else s * 2 * np.arctanh(np.tanh(J / 2) * np.tanh(L / 2))


def _gated_llr(v, excl, t, lo, par, kids, sgn, xv, J):
    """LLR of the unobserved token v from the copy forest restricted to nodes < t, observed nodes lo..t-1, ignoring the
    branch through excl (same recursion as exact_sources.gated._bp_llr, with the child lists built once per window)."""
    L = 0.0
    p = par[v]
    if p >= 0 and p != excl:
        L += _gated_msg(p, v, sgn[v], t, lo, par, kids, sgn, xv, J)
    for c in kids[v]:
        if c >= t:
            break
        if c != excl:
            L += _gated_msg(c, v, sgn[c], t, lo, par, kids, sgn, xv, J)
    return L


def gated_indirect(x, m, sigma, eps, ns):
    """Per-window information (bits per position, mean over positions t >= n) that n tokens of context give about x_t
    through chains of copies when the marker points beyond the context (m_t > n). Exact per window (belief propagation
    on the copy forest, a zero-field tree Ising model once the markers are known); the full context-conditioned curve is
    h(n) = h_direct(n) - E[indirect(n)]. Returns (W, len(ns))."""
    J = 2 * np.arctanh(1 - 2 * eps)
    W, T = x.shape
    out = np.zeros((W, len(ns)))
    for w in range(W):
        mw, xw = m[w].astype(np.int64), x[w]
        par = np.where(mw > 0, np.arange(T) - mw, -1)
        sgn = np.where(mw > 0, sigma[mw], 0.0)
        kids = [[] for _ in range(T)]
        for u in range(T):
            if par[u] >= 0:
                kids[par[u]].append(u)
        for i, n in enumerate(ns):
            g = 0.0
            for t in range(n, T):
                k = mw[t]
                if k > n:
                    L = _gated_llr(t - k, -1, t, t - n, par, kids, sgn, xw, J)
                    if L != 0.0:
                        mu = (1 - 2 * eps) * np.tanh(L / 2)
                        g += 1 - hbin((1 + mu) / 2)
            out[w, i] = g / (T - n)
    return out


def window_corr_sd(x, mean):
    """Per-window estimates of the connected <(x_t - mu_t)(x_{t+n} - mu_{t+n})> for every lag (mean: scalar or the exact
    per-position means, shape (T,)); returns their SD across windows, (T,)."""
    xc = x - np.broadcast_to(np.asarray(mean, float), (x.shape[1],))[None, :]
    W, T = x.shape
    sd = np.full(T, np.nan)
    for n in range(1, T):
        sd[n] = (xc[:, :-n] * xc[:, n:]).mean(1).std()
    return sd


# ------------------------------------------------------------------ estimators (calibrated on the gated sources)
def fit_gamma(h, lo=4, hi=64):
    """gamma from Delta(n) = h(n) - h(T-1) ~ a (n^-g - N^-g), N = T - 1, n in [lo, hi] (as scripts/source_stats.py).
    Returns gamma, rms of the log residual (power-law quality)."""
    T = len(h)
    N = T - 1
    hi = min(hi, N - 1)
    n = np.arange(lo, hi + 1, dtype=float)
    d = h[lo:hi + 1] - h[N]
    ok = d > 0
    if ok.sum() < 4:
        return np.nan, np.nan
    n, d = n[ok], d[ok]
    best = (np.inf, np.nan)
    for g in np.linspace(0.05, 5.0, 991):
        f = n ** -g - N ** -g
        la = np.mean(np.log(d) - np.log(f))
        res = np.sqrt(np.mean((np.log(d) - la - np.log(f)) ** 2))
        if res < best[0]:
            best = (res, g)
    return float(best[1]), float(best[0])


def fit_beta(absC, lo=4, hi=64):
    """beta from |C(n)| ~ n^-beta, least squares in log over n in [lo, hi]; returns beta, rms of the log residual."""
    hi = min(hi, len(absC) - 1)
    n = np.arange(lo, hi + 1)
    c = absC[lo:hi + 1]
    ok = np.isfinite(c) & (c > 1e-300)
    if ok.sum() < 4:
        return np.nan, np.nan
    b, a = np.polyfit(np.log(n[ok]), np.log(c[ok]), 1)
    res = np.sqrt(np.mean((np.log(c[ok]) - (a + b * np.log(n[ok]))) ** 2))
    return float(-b), float(res)


def _lzma_bits(tok, side):
    both = np.stack([side.ravel().astype(np.uint8), tok.ravel().astype(np.uint8)], 1).tobytes()
    lb = len(lzma.compress(both, preset=9))
    ls = len(lzma.compress(side.ravel().astype(np.uint8).tobytes(), preset=9))
    return 8.0 * (lb - ls) / tok.size


def compress_bits(tok, side, rng):
    """A heuristic compressibility, the same serialization for every source: lzma bits per token of (side byte, token
    byte) pairs minus lzma bits of the side bytes alone (side = 0 where there is no side input). The codec's overhead is
    removed by a null calibration: the same estimate on fair coins with the same side stream should be 1 bit, so the
    result is estimate - null + 1. Returns (calibrated, raw, null)."""
    raw = _lzma_bits(tok, side)
    null = _lzma_bits(rng.integers(0, 2, tok.shape).astype(np.int8), side)
    return raw - null + 1.0, raw, null


# ------------------------------------------------------------------ data-only predictions
def tail_curves(h, absC, sdC, T, P, Bstar=B_STAR, s_soft=SOFT_SLOPE, zC=1.0, tmin=0):
    """Predicted excess loss (bits per token, averaged over positions t >= tmin) of a learner that has learned a set of
    context distances, for every budget in P (tokens). Distance n carries delta(n) = h(n-1) - h(n) bits per position
    with t >= n, and the training set holds P delta(n) (T - n)/T bits of it.
      info:  learned when P delta(n) (T-n)/T >= Bstar (one price per distance; gated copy: K = 210 events)
      soft:  learned fraction N^s / (1 + N^s), N = that information / Bstar (the dial's logistic edge)
      cag:   learned when the pair-correlation SNR |C(n)| sqrt(P/T) / sd_window(n) >= zC (Cagnetta et al.)
    Returns dict of arrays over P: tails and horizons (largest learned distance)."""
    n = np.arange(1, T)
    delta = np.clip(h[:-1] - h[1:], 0, None)
    w_all = (T - n) / T
    w = (T - np.maximum(n, tmin)) / (T - tmin)
    P = np.asarray(P, float)[:, None]
    N = P * delta[None, :] * w_all[None, :] / Bstar
    lrn = N >= 1
    f = np.where(N > 0, N ** s_soft / (1 + N ** s_soft), 0.0)
    cag_ok = bool(np.all(np.isfinite(absC[1:])) and np.all(np.isfinite(sdC[1:])))  # unknown correlations stay unknown
    C = np.nan_to_num(absC[1:], nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.where(sdC[1:] > 0, C[None, :] * np.sqrt(P / T) / sdC[None, 1:], 0.0)
    lc = snr >= zC
    dw = (delta * w)[None, :]
    hor = lambda L: np.where(L.any(1), np.where(L, n[None, :], 0).max(1), 0)
    nan = np.full(len(P), np.nan)
    return dict(info=(dw * ~lrn).sum(1), soft=(dw * (1 - f)).sum(1), cag=(dw * ~lc).sum(1) if cag_ok else nan,
                info_horizon=hor(lrn), cag_horizon=hor(lc) if cag_ok else nan, info_learned=lrn.sum(1),
                cag_learned=lc.sum(1) if cag_ok else nan)


def slope(P, y):
    """-d log y / d log P by least squares; nan if any y <= 0."""
    y = np.asarray(y, float)
    if np.any(~np.isfinite(y)) or np.any(y <= 0):
        return np.nan
    return float(-np.polyfit(np.log(P), np.log(y), 1)[0])
