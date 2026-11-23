"""Exact next-block distributions for k-spin block tokens of TFIM X and Z windows.

A block token packs k consecutive spins (first spin = most significant bit, bit 1 = value -1, as in
make_data.py). p(block | past blocks) is the product of exact single-spin conditionals along each of
the 2^k completions, obtained by branching the free-fermion conditioning of tfim.sample_x.
"""
import numpy as np


def _cond(A, s, mt):
    """Condition the Majorana covariance (B, 2R, 2R) of the remaining sites on the next spin s (B,)."""
    if A.shape[1] <= 2:
        return A[:, 2:, 2:]
    ca = A[:, 2:, 0]
    cb = A[:, 2:, 1]
    coef = (s / (1 + s * mt))[:, None, None]
    return A[:, 2:, 2:] - coef * (ca[:, :, None] * cb[:, None, :] - cb[:, :, None] * ca[:, None, :])


def pack(bits, k):
    """(n, T) bits -> (n, T//k) block token ids, first spin most significant."""
    n, T = bits.shape
    w = 1 << np.arange(k - 1, -1, -1)
    return (bits.reshape(n, T // k, k).astype(np.int64) @ w).astype(np.int16)


def block_probs_x(Gam, x, k, batch=128):
    """x (n, T) in +-1 drawn from the window covariance Gam. Returns (n, T//k, 2^k)."""
    n, T = x.shape
    out = np.empty((n, T // k, 1 << k))
    for s0 in range(0, n, batch):
        xb = x[s0:s0 + batch]
        B = len(xb)
        A = np.broadcast_to(Gam, (B,) + Gam.shape).copy()
        for j, t0 in enumerate(range(0, T, k)):
            states, probs = [A], [np.ones(B)]
            for i in range(k):
                ns, npr = [], []
                for S, pr in zip(states, probs):
                    mt = S[:, 0, 1]
                    for v in (1.0, -1.0):  # bit 0 (+1) first
                        sv = np.full(B, v)
                        npr.append(pr * (1 + v * mt) / 2)
                        ns.append(_cond(S, sv, mt) if i < k - 1 else None)
                states, probs = ns, npr
            out[s0:s0 + B, j] = np.stack(probs, axis=1)
            for t in range(t0, t0 + k):
                A = _cond(A, xb[:, t], A[:, 0, 1])
    return out


def block_probs_z(Gam_w, w, z, k, batch=128):
    """Z windows z (n, T) with z_t = z_{t-1} w_{t-1} and z_0 a fair coin; Gam_w is the covariance of the
    dual X chain over at least T-1 sites. Returns (n, T//k, 2^k)."""
    n, T = z.shape
    G = Gam_w[: 2 * (T - 1), : 2 * (T - 1)]
    out = np.empty((n, T // k, 1 << k))
    for s0 in range(0, n, batch):
        wb, zb = w[s0:s0 + batch, : T - 1], z[s0:s0 + batch]
        B = len(zb)
        A = np.broadcast_to(G, (B,) + G.shape).copy()
        for j, t0 in enumerate(range(0, T, k)):
            zprev0 = zb[:, t0 - 1] if t0 > 0 else np.zeros(B)
            states, probs, zprev = [A], [np.ones(B)], [zprev0]
            for i in range(k):
                t = t0 + i
                ns, npr, nz = [], [], []
                for S, pr, zp in zip(states, probs, zprev):
                    for v in (1.0, -1.0):
                        if t == 0:
                            npr.append(pr * 0.5)
                            ns.append(S)
                        else:
                            mt = S[:, 0, 1]
                            wv = zp * v
                            npr.append(pr * (1 + wv * mt) / 2)
                            ns.append(_cond(S, wv, mt) if i < k - 1 else None)
                        nz.append(np.full(B, v))
                states, probs, zprev = ns, npr, nz
            out[s0:s0 + B, j] = np.stack(probs, axis=1)
            for t in range(max(t0, 1), t0 + k):
                A = _cond(A, wb[:, t - 1], A[:, 0, 1])
    return out
