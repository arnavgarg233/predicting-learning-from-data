"""Random Hierarchy Model (Cagnetta et al.) as an exactly solvable sequence source.

v symbols per level, m production rules per symbol, branching s, depth L: a window is the s^L leaves of one tree whose
root is uniform over the v symbols and whose every node picks one of its m rules uniformly. At each level the v*m rules
are distinct s-tuples drawn without replacement from the v^s possible ones (unambiguous grammar), independently per level.

Exact inference: the tree is fixed, so p(x_t | x_lo..x_{t-1}) follows from one upward (inside) pass in which leaves
outside [lo, t] are unobserved (message 1), leaves in [lo, t-1] are observed, and leaf t carries a query axis.
Only the O(2L) nodes cut by the boundaries lo and t need recomputing; whole observed subtrees reuse full messages.
"""
import numpy as np


class RHM:
    def __init__(self, v, m, s, L, seed):
        assert v * m <= v ** s, "need v*m <= v^s distinct rules"
        self.v, self.m, self.s, self.L = v, m, s, L
        self.T = s ** L
        rng = np.random.default_rng(seed)
        self.R = {}  # level k (1..L): (v, m, s) child symbols of each rule of a level-k parent
        for k in range(1, L + 1):
            idx = rng.choice(v ** s, size=v * m, replace=False)
            digits = (idx[:, None] // v ** np.arange(s - 1, -1, -1)[None, :]) % v
            self.R[k] = digits.reshape(v, m, s)
        # transition matrices M[k][c][a, b] = P(child c = b | parent = a)
        self.M = {}
        for k in range(1, L + 1):
            Mk = np.zeros((s, v, v))
            for c in range(s):
                for a in range(v):
                    np.add.at(Mk[c, a], self.R[k][a, :, c], 1.0 / m)
            self.M[k] = Mk

    # ------------------------------------------------------------ sampling
    def sample(self, n, rng):
        z = rng.integers(0, self.v, (n, 1))
        for k in range(self.L, 0, -1):
            r = rng.integers(0, self.m, z.shape)
            z = self.R[k][z, r].reshape(n, -1)  # (n, nodes, s) -> (n, nodes*s)
        return z.astype(np.int8)

    def entropy_rate(self):
        """Exact H(window)/T in bits: log2 v for the root plus log2 m per internal node (rules are distinct, so the
        tree is a function of the leaves)."""
        n_int = sum(self.s ** k for k in range(self.L))  # internal nodes: levels 1..L have s^(L-k) nodes
        return (np.log2(self.v) + n_int * np.log2(self.m)) / self.T

    # ------------------------------------------------------------ exact inference
    def full_messages(self, x):
        """F[k]: (W, s^(L-k), v) inside messages with every leaf observed, each normalised to max 1."""
        W = x.shape[0]
        F = {0: np.eye(self.v)[x.astype(np.int64)]}
        for k in range(1, self.L + 1):
            ch = F[k - 1].reshape(W, -1, self.s, self.v)
            prod = np.ones((W, ch.shape[1], self.v, self.m))
            for c in range(self.s):
                prod *= ch[:, :, c, :][:, :, self.R[k][:, :, c]]
            msg = prod.mean(-1)
            F[k] = msg / msg.max(-1, keepdims=True)
        return F

    def _msg(self, F, k, j, lo, t):
        """Inside message of node (k, j) with leaves lo..t-1 observed and leaf t queried: (W|1, Q|1, v) or None (=1)."""
        w = self.s ** k
        a, b = j * w, (j + 1) * w
        if b <= lo or a > t:
            return None
        if a >= lo and b <= t:
            return F[k][:, j, None, :]
        if k == 0:  # leaf t
            return np.eye(self.v)[None]
        prod = None
        for c in range(self.s):
            ch = self._msg(F, k - 1, j * self.s + c, lo, t)
            if ch is None:
                continue
            g = ch[..., self.R[k][:, :, c]]  # (W, Q, v, m)
            prod = g if prod is None else prod * g
        msg = prod.mean(-1)
        return msg / msg.max(axis=(-2, -1), keepdims=True)

    def cond(self, F, lo, t):
        """p(x_t = . | x_lo..x_{t-1}) for every window: (W, v)."""
        root = self._msg(F, self.L, 0, lo, t)
        joint = root.sum(-1)  # uniform root prior
        if joint.shape[0] == 1:
            joint = np.broadcast_to(joint, (F[0].shape[0], self.v))
        return joint / joint.sum(-1, keepdims=True)

    def prefix_probs(self, x):
        """Exact next-token distributions p(x_t | x_0..x_{t-1}) for every position: (W, T, v) float32."""
        F = self.full_messages(x)
        out = np.empty((x.shape[0], self.T, self.v), np.float32)
        for t in range(self.T):
            out[:, t] = self.cond(F, 0, t)
        return out

    # ------------------------------------------------------------ exact pair correlations
    def pair_corr(self):
        """C(n) = mean over i of ||P(x_i, x_{i+n}) - P(x_i) P(x_{i+n})||_F (and the top singular value), n = 1..T-1;
        index 0 unused (nan)."""
        s, L, v, T = self.s, self.L, self.v, self.T
        digit = lambda i, k: (i // s ** k) % s
        # node marginals pi[k][j] (v,)
        pi = {L: np.full((1, v), 1.0 / v)}
        for k in range(L, 0, -1):
            pk = pi[k]
            pi[k - 1] = np.stack([pk @ self.M[k][c] for c in range(s)], 1).reshape(-1, v)

        def down(i, K):  # P(leaf i | node at level K-1 on its path): (v, v)
            P = np.eye(v)
            for k in range(K - 1, 0, -1):
                P = P @ self.M[k][digit(i, k - 1)]
            return P
        fro = np.zeros(T)
        op = np.zeros(T)
        cnt = np.zeros(T)
        for i in range(T):
            for j in range(i + 1, T):
                K = next(k for k in range(1, L + 1) if i // s ** k == j // s ** k)
                ci, cj = digit(i, K - 1), digit(j, K - 1)
                z = pi[K][i // s ** K]
                A = down(i, K)[self.R[K][:, :, ci]]  # (v, m, v)
                B = down(j, K)[self.R[K][:, :, cj]]
                J = np.einsum("z,zra,zrb->ab", z / self.m, A, B)
                C = J - np.outer(J.sum(1), J.sum(0))
                n = j - i
                fro[n] += np.sqrt((C ** 2).sum())
                op[n] += np.linalg.svd(C, compute_uv=False)[0]
                cnt[n] += 1
        fro[1:] /= cnt[1:]
        op[1:] /= cnt[1:]
        fro[0] = op[0] = np.nan
        return fro, op


def entropy_bits(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(-1)


def info_curve(src, x, nmax=None):
    """Per-window conditional entropies E_t(n) = H(p(x_t | x_{max(0,t-n)}..x_{t-1})) (Rao-Blackwellised: the entropy of
    the exact conditional, unbiased for H(x_t | n tokens of context, position t)) for n = 0..nmax and every t.
    Returns H with shape (W, nmax+1, T); n > t is the full prefix."""
    T = src.T
    nmax = T - 1 if nmax is None else nmax
    F = src.full_messages(x)
    W = x.shape[0]
    H = np.empty((W, nmax + 1, T))
    for t in range(T):
        for n in range(0, nmax + 1):
            lo = max(0, t - n)
            if n > t:
                H[:, n, t] = H[:, t, t]
                continue
            H[:, n, t] = entropy_bits(src.cond(F, lo, t))
    return H


def deltas_from_H(H):
    """delta(n), n = 1..T-1: mean over windows and over positions t >= n of E_t(n-1) - E_t(n) (the information distance
    n adds where it exists, the zoo's convention for a stationary source); h(n) = h(0) - cumsum(delta).
    Returns h (T,), per-window delta (W, T-1)."""
    W, N1, T = H.shape
    d = np.zeros((W, T - 1))
    for n in range(1, N1):
        d[:, n - 1] = (H[:, n - 1, n:] - H[:, n, n:]).mean(-1)
    h0 = H[:, 0, :].mean()
    h = h0 - np.concatenate([[0.0], np.cumsum(d.mean(0))])
    return h, d
