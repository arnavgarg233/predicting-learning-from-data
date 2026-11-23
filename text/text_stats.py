"""Operator norm of the lag-n token covariance C(n) = P(x_i, x_{i+n}) - P(x_i)P(x_{i+n}) for several
tokenizations of TinyStories, with a bootstrap over contiguous text chunks.

  python text_stats.py measure NAME [B]   NAME in bytes|bpe512|bpe2048|bpe8192, B bootstrap replicates
  python text_stats.py report             merge partial results, write beta_results.json and the figure
"""
import json, sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds

ROOT = Path(__file__).resolve().parent
D = ROOT.parent / "data" / "text"
NAMES = ["bytes", "bpe512", "bpe2048", "bpe8192"]
FIT_TOK = np.unique(np.round(np.geomspace(1, 200, 15))).astype(int)
LAGS = np.unique(np.concatenate([np.round(np.geomspace(1, 1024, 25)), FIT_TOK])).astype(int)
BYTE_WIN = (10, 1000)
K = 20


def vocab(name):
    return 256 if name == "bytes" else int(name[3:])


def chunk_joints(tok, n, V):
    edges = np.linspace(0, len(tok), K + 1).astype(int)
    Js = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a, b = tok[lo:hi - n].astype(np.int32), tok[lo + n:hi].astype(np.int32)
        Js.append(sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(V, V)).tocsr())
    return Js


class Cov:
    """Weighted sum of per-chunk joint counts, applied implicitly as J/N - outer(pa, pb)."""

    def __init__(self, Js, V):
        self.V, self.K = V, len(Js)
        self.cat = sp.hstack(Js, format="csr")
        self.row = np.stack([np.asarray(J.sum(1)).ravel() for J in Js])
        self.col = np.stack([np.asarray(J.sum(0)).ravel() for J in Js])

    def sigma(self, w, v0=None):
        V, K, cat = self.V, self.K, self.cat
        N = w @ self.row.sum(1)
        pa, pb = w @ self.row / N, w @ self.col / N
        mv = lambda v: cat @ np.outer(w, np.ravel(v)).ravel() / N - pa * (pb @ np.ravel(v))
        rmv = lambda u: w @ (cat.T @ np.ravel(u)).reshape(K, V) / N - pb * (pa @ np.ravel(u))
        op = LinearOperator((V, V), matvec=mv, rmatvec=rmv, dtype=float)
        _, s, vh = svds(op, k=1, tol=1e-5, v0=v0, return_singular_vectors="vh")
        return s[0], vh[0]


def beta(x, y):
    return -np.polyfit(np.log(x), np.log(y), 1)[0]


def fits(lags, bpt, sig, floor):
    sc = np.sqrt(np.maximum(sig ** 2 - floor ** 2, 1e-30))
    tm = np.isin(lags, FIT_TOK)
    bm = (lags * bpt >= BYTE_WIN[0]) & (lags * bpt <= BYTE_WIN[1])
    return dict(tok_1_200=beta(lags[tm], sig[tm]), byte_10_1000=beta(lags[bm], sig[bm]),
                tok_1_200_floorcorr=beta(lags[tm], sc[tm]), byte_10_1000_floorcorr=beta(lags[bm], sc[bm]))


def sweep(tok, V, W):
    """sigma for every weight vector in W (rows) at every lag; row 0 of W should be all ones."""
    out, v0 = np.zeros((len(W), len(LAGS))), None
    for i, n in enumerate(LAGS):
        cov = Cov(chunk_joints(tok, int(n), V), V)
        out[0, i], v0 = cov.sigma(W[0])
        for b in range(1, len(W)):
            out[b, i] = cov.sigma(W[b], v0=v0)[0]
        print(f"  lag {n} sigma {out[0, i]:.4g}", flush=True)
    return out


def measure(name, B):
    V, t0 = vocab(name), time.time()
    nbytes = {s: len(np.load(D / f"tok_bytes_{s}.npy", mmap_mode="r")) for s in ("train", "valid")}
    res = dict(name=name, V=V, lags=LAGS.tolist(), fit_tok=FIT_TOK.tolist(), byte_win=BYTE_WIN, K=K, B=B)
    rng = np.random.default_rng(0)
    W = np.vstack([np.ones(K), rng.multinomial(K, np.ones(K) / K, size=B)]).astype(float)
    for split in ("train", "valid"):
        tok = np.load(D / f"tok_{name}_{split}.npy")
        bpt = nbytes[split] / len(tok)
        sh = tok.copy()
        np.random.default_rng(1).shuffle(sh)
        floor = Cov(chunk_joints(sh, 1, V), V).sigma(np.ones(K))[0]
        del sh
        sig = sweep(tok, V, W if split == "train" else W[:1])
        r = dict(n_tokens=len(tok), n_bytes=nbytes[split], bytes_per_token=bpt, floor=floor, sigma=sig[0].tolist(),
                 beta=fits(LAGS, bpt, sig[0], floor))
        if split == "train" and B > 0:
            bb = [fits(LAGS, bpt, s, floor) for s in sig[1:]]
            r["sigma_boot"] = sig[1:].tolist()
            r["beta_err"] = {k: float(np.std([x[k] for x in bb], ddof=1)) for k in bb[0]}
        res[split] = r
        print(name, split, r["beta"], r.get("beta_err"), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(ROOT / f"partial_{name}.json", "w"))


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = {n: json.load(open(ROOT / f"partial_{n}.json")) for n in NAMES}
    json.dump(res, open(ROOT / "beta_results.json", "w"), indent=1)
    cols = dict(bytes="#444444", bpe512="#0072B2", bpe2048="#009E73", bpe8192="#D55E00")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    lags = np.array(LAGS)
    for n, r in res.items():
        t = r["train"]
        x, y = lags * t["bytes_per_token"], np.array(t["sigma"])
        lab = f"{'raw bytes' if n == 'bytes' else 'BPE'} V={r['V']}  beta={t['beta']['tok_1_200']:.2f} / {t['beta']['byte_10_1000']:.2f}"
        ax.plot(x, y, "o-", ms=3, lw=1.3, color=cols[n], label=lab)
        ax.axhline(t["floor"], color=cols[n], ls=":", lw=0.8)
    ax.axvspan(*BYTE_WIN, color="0.9", zorder=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("lag (bytes) = lag in tokens x mean bytes/token")
    ax.set_ylabel(r"$\|C(n)\|_{op}$")
    ax.set_title("TinyStories (150 MB train slice); legend: beta fit over tokens 1-200 / bytes 10-1000\n"
                 "dotted: shuffled-stream noise floor", fontsize=8)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(ROOT / "beta_by_tokenizer.png", dpi=200)


if __name__ == "__main__":
    if sys.argv[1] == "measure":
        measure(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 30)
    else:
        report()
