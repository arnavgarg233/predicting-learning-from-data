"""Exact measurement statistics of the transverse-field Ising ground state.

H = -sum_i Z_i Z_{i+1} - h sum_i X_i.  After Jordan-Wigner, X_i = i g_{2i} g_{2i+1}
is local in the Majoranas, so the X-basis distribution of any window is fixed by the
window's Majorana covariance Gamma[2i, 2j+1] = G(j - i).

Z-basis strings come from Kramers-Wannier duality: the domain walls z_i z_{i+1} of the
chain at field h have the X-basis distribution of the chain at field 1/h, and the
global sign is an independent fair coin.
"""
import numpy as np


def g_coeffs(h, rmax, L=None, nk=1 << 20):
    """G(r) for r = -rmax..rmax. L=None is the infinite chain; otherwise the periodic
    chain of length L in its ground-state (antiperiodic fermion) sector."""
    if L is None and h == 1.0:
        r = np.arange(-rmax, rmax + 1)
        return -2 / (np.pi * (2 * r - 1))
    n = L if L is not None else nk
    k = 2 * np.pi * (np.arange(n) + 0.5) / n
    phase = (h - np.exp(-1j * k)) / np.abs(h - np.exp(-1j * k))
    r = np.arange(-rmax, rmax + 1)
    if L is not None:
        return np.real(np.exp(1j * np.outer(r, k)) @ phase) / n
    out = np.empty(len(r))
    for i, ri in enumerate(r):
        out[i] = np.mean(np.real(np.exp(1j * ri * k) * phase))
    return out


def window_gamma(G, T):
    """2T x 2T Majorana covariance of a T-site window; G indexed as G[r + rmax]."""
    rmax = (len(G) - 1) // 2
    i = np.arange(T)
    D = G[(i[None, :] - i[:, None]) + rmax]
    Gam = np.zeros((2 * T, 2 * T))
    Gam[0::2, 1::2] = D
    Gam[1::2, 0::2] = -D.T
    return Gam


def window_prob(Gam, x):
    """Exact probability of an X-basis string x in {+1,-1}^T."""
    T = len(x)
    S = np.zeros((2 * T, 2 * T))
    S[0::2, 1::2] = np.diag(x)
    S[1::2, 0::2] = -np.diag(x)
    return np.sqrt(max(np.linalg.det(np.eye(2 * T) - Gam @ S), 0.0)) / 2 ** T


def sample_x(Gam, n, rng, batch=256):
    """Draw n independent X-basis windows exactly.

    Returns x (n, T) in {+1,-1} and m (n, T), the exact conditional mean
    E[x_t | x_<t], so p(x_t = +1 | x_<t) = (1 + m_t) / 2.
    """
    T = Gam.shape[0] // 2
    xs, ms = [], []
    for start in range(0, n, batch):
        B = min(batch, n - start)
        A = np.broadcast_to(Gam, (B, 2 * T, 2 * T)).copy()
        x = np.empty((B, T))
        m = np.empty((B, T))
        for t in range(T):
            mt = A[:, 0, 1]
            s = np.where(rng.random(B) < (1 + mt) / 2, 1.0, -1.0)
            m[:, t], x[:, t] = mt, s
            if t == T - 1:
                break
            ca = A[:, 2:, 0]
            cb = A[:, 2:, 1]
            coef = (s / (1 + s * mt))[:, None, None]
            A = A[:, 2:, 2:] - coef * (ca[:, :, None] * cb[:, None, :] - cb[:, :, None] * ca[:, None, :])
        xs.append(x)
        ms.append(m)
    return np.concatenate(xs), np.concatenate(ms)


def sample_x_torch(Gam, n, rng, batch=256, threads=8):
    """Same as sample_x, using multithreaded torch float64 for the covariance updates."""
    import torch

    torch.set_num_threads(threads)
    T = Gam.shape[0] // 2
    G0 = torch.as_tensor(Gam, dtype=torch.float64)
    xs, ms = [], []
    for start in range(0, n, batch):
        B = min(batch, n - start)
        A = G0.expand(B, 2 * T, 2 * T).clone()
        x = np.empty((B, T))
        m = np.empty((B, T))
        for t in range(T):
            mt = A[:, 0, 1].numpy().copy()
            s = np.where(rng.random(B) < (1 + mt) / 2, 1.0, -1.0)
            m[:, t], x[:, t] = mt, s
            if t == T - 1:
                break
            ca = A[:, 2:, 0].unsqueeze(2)
            cb = A[:, 2:, 1].unsqueeze(2)
            coef = torch.as_tensor(s / (1 + s * mt)).view(B, 1, 1)
            A = A[:, 2:, 2:] - coef * (ca @ cb.transpose(1, 2) - cb @ ca.transpose(1, 2))
        xs.append(x)
        ms.append(m)
    return np.concatenate(xs), np.concatenate(ms)


def x_to_z(w, rng):
    """Z-basis windows of length T+1 from dual X-basis windows of length T."""
    z0 = np.where(rng.random(len(w)) < 0.5, 1.0, -1.0)
    return np.concatenate([z0[:, None], z0[:, None] * np.cumprod(w, axis=1)], axis=1)


def sample(h, basis, T, n, rng, nk=1 << 20):
    """n independent length-T windows of the infinite chain in basis 'X' or 'Z',
    with exact conditional means. For Z the first token's mean is 0."""
    if basis == "X":
        G = g_coeffs(h, T, nk=nk)
        return sample_x(window_gamma(G, T), n, rng)
    G = g_coeffs(1.0 / h, T - 1, nk=nk)
    w, mw = sample_x(window_gamma(G, T - 1), n, rng)
    z = x_to_z(w, rng)
    m = np.zeros_like(z)
    m[:, 1:] = z[:, :-1] * mw
    return z, m


def conditional_mean_last(Gam, ctx):
    """E[x_target | context] where Gam is the (n+1)-site window covariance, the target is
    the last site and ctx (B, n) holds the X values of the first n sites."""
    B, n = ctx.shape
    if n == 0:
        return np.full(B, Gam[0, 1])
    c, t = slice(0, 2 * n), slice(2 * n, 2 * n + 2)
    S = np.zeros((B, 2 * n, 2 * n))
    i = np.arange(n)
    S[:, 2 * i, 2 * i + 1] = ctx
    S[:, 2 * i + 1, 2 * i] = -ctx
    M = Gam[c, c][None] + S
    y = np.linalg.solve(M, np.broadcast_to(Gam[c, t], (B, 2 * n, 2)))
    return Gam[t, t][0, 1] - np.einsum("i,bi->b", Gam[t, c][0], y[:, :, 1])


def kl_bits(m_p, m_q):
    p = np.clip((1 + m_p) / 2, 1e-15, 1 - 1e-15)
    q = np.clip((1 + m_q) / 2, 1e-15, 1 - 1e-15)
    return (p * np.log2(p / q) + (1 - p) * np.log2((1 - p) / (1 - q)))


def binary_entropy_bits(m):
    p = np.clip((1 + m) / 2, 1e-300, 1)
    q = np.clip((1 - m) / 2, 1e-300, 1)
    return -(p * np.log2(p) + q * np.log2(q))


def x_connected(G, r):
    """Exact connected <X_0 X_r> on the infinite chain (Wick)."""
    rmax = (len(G) - 1) // 2
    return -G[r + rmax] * G[-r + rmax]


def string_corr(G, r):
    """<X_0 ... X_{r-1}> = Toeplitz determinant det[G(j - i)]."""
    rmax = (len(G) - 1) // 2
    i = np.arange(r)
    return np.linalg.det(G[(i[None, :] - i[:, None]) + rmax])
