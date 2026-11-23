"""Per-tokenizer statistics for the forecaster, for a directory written by text/tok_family.py.

  python text/tok_stats.py --dir data/bfine/tok --data data/bfine --names bpe0,bpe1      writes DIR/stats/{name}.json
  python text/tok_stats.py --dir data/bfine/tok --merge                                  writes DIR/stats.json

Per tokenizer:
  Only train (counts of what the learner sees) and ref_stats are read; valid and test are never read.
  bytes_per_token   train prefix (the tokens the learner sees), ref_stats
  spelling          word-level phase / spelling multiplicity on the first --n_words ref_stats bytes (as text/spelling.py)
  rare_spelling     M_k(D), Mt_k(D), Mtb_k(D) for D = 2^ds bytes of train (as text/rare_spelling.py); for drop* also the same
                    with the spellings counted in dropout sample s0 instead of the canonical train stream (M_drop_s0)
  beta              pair-correlation exponent of the operator norm of the lag-n token covariance, fitted over token lags 1-200
                    and byte lags 10-1000 (as text/text_stats.py, no bootstrap), on the ref_stats stream
  gamma             conditional-entropy exponent in token units from the reference LMs (--refcurve, refcurve_fineweb.json): for each
                    reference model, Delta(n) = h(n) - h(n_ref) with n_ref the longest context, fitted as Delta ~ n^-gamma over the
                    declared window n = GAMMA_WINDOW tokens of THIS tokenizer, i.e. bytes n * bytes_per_token (log-log interpolation of
                    the model's bits/byte curve in context bytes); alpha = gamma / (2 beta) with the token-window beta (Cagnetta et al.)
"""
GAMMA_WINDOW = (4, 64)  # tokens
import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rare_spelling as rs  # noqa: E402
import spelling as sp  # noqa: E402
import text_stats as ts  # noqa: E402


def spelling_stats(t, L, raw, N):
    n = int(np.searchsorted(np.cumsum(L[t[: 2 * N]].astype(np.int64)), N + 64)) + 1
    t = t[:n].astype(np.int64)
    bs = np.concatenate([[0], np.cumsum(L[t])])
    assert bs[-1] > N, "valid is shorter than n_words"
    st = np.append(np.flatnonzero(sp.word_starts(raw[:N])), N)
    a, e = st[:-2], st[1:-1]
    i0 = np.searchsorted(bs, a, side="right") - 1
    i1 = np.searchsorted(bs, e, side="left")
    off = bs[i0] - a
    ph, spl = defaultdict(Counter), defaultdict(Counter)
    rb, tl = raw.tobytes(), t.tolist()
    for x, y, o, p, q in zip(a.tolist(), e.tolist(), off.tolist(), i0.tolist(), i1.tolist()):
        w = rb[x:y]
        ph[w][o] += 1
        spl[w][(o, tuple(tl[p:q]))] += 1
    r = sp.summarize({w: (sum(c.values()), sp.mult(c), sp.mult(spl[w])) for w, c in ph.items()})
    r["tokens_per_word_freq"] = float(np.mean(i1 - i0))
    return r


def rare_stats(tv, L, vraw, traw, head, ds, N, V):
    """head: train token stream whose spellings are counted (canonical, or a dropout sample)."""
    hv, _ = rs.word_hashes(tv, L, vraw, N)
    uv, inv = np.unique(hv, return_inverse=True)
    bv = np.concatenate([[0], np.cumsum(L[tv])])
    ntv = int(np.searchsorted(bv, N, side="right")) - 1
    tvv, lv = tv[:ntv], L[tv[:ntv]]
    cum = np.cumsum(L[head])
    Ns = {k: int(np.searchsorted(cum, 2 ** k)) + 1 for k in ds}
    assert all(n <= len(head) for n in Ns.values()), "train head too short for the largest D"
    nbs = {k: int(cum[Ns[k] - 1]) for k in ds}
    ht, et = rs.word_hashes(head[: Ns[max(ds)]], L, traw, nbs[max(ds)])
    out = {}
    for k in ds:
        m = int(np.searchsorted(et, nbs[k] - 1, side="right"))
        u, c = np.unique(ht[:m], return_counts=True)
        pos = np.minimum(np.searchsorted(u, uv), len(u) - 1)
        cu = np.where(u[pos] == uv, c[pos], 0)[inv]
        tc = np.bincount(head[: Ns[k]], minlength=V)[tvv]
        r = {f"M{q}": v for q, v in rs.mass(cu).items()}
        r.update({f"Mt{q}": float(np.mean(tc < q)) for q in rs.KS})
        r.update({f"Mtb{q}": float(np.sum(lv[tc < q]) / lv.sum()) for q in rs.KS})
        r.update(n_word_occ_valid=len(hv), n_word_occ_train=m, n_spellings_train=len(u), train_bytes=nbs[k])
        out[str(k)] = r
    return out


def beta_stats(tok, V, n_bytes):
    bpt = n_bytes / len(tok)
    sh = tok.copy()
    np.random.default_rng(1).shuffle(sh)
    floor = ts.Cov(ts.chunk_joints(sh, 1, V), V).sigma(np.ones(ts.K))[0]
    del sh
    sig, v0 = [], None
    for n in ts.LAGS:
        s, v0 = ts.Cov(ts.chunk_joints(tok, int(n), V), V).sigma(np.ones(ts.K), v0=v0)
        sig.append(float(s))
    sig = np.array(sig)
    return dict(n_tokens=int(len(tok)), n_bytes=int(n_bytes), bytes_per_token=bpt, floor=float(floor), lags=ts.LAGS.tolist(),
                sigma=sig.tolist(), beta=ts.fits(ts.LAGS, bpt, sig, floor))


def gamma_stats(refjson, bpt, beta_tok, window=GAMMA_WINDOW):
    out = {}
    for short, r in refjson.items():
        x = np.array([row["context_bytes_exact_mean"] for row in r["rows"]])
        y = np.array([row["bits_per_byte"] for row in r["rows"]])
        h = lambda b: np.exp(np.interp(np.log(b), np.log(x), np.log(y)))  # noqa: E731
        n = np.unique(np.round(np.geomspace(window[0], window[1], 12))).astype(int)
        delta = h(n * bpt) - y[-1]
        ok = delta > 0
        g = float(-np.polyfit(np.log(n[ok]), np.log(delta[ok]), 1)[0]) if ok.sum() >= 3 else None
        out[short] = dict(gamma=g, window_tokens=list(window), window_bytes=[window[0] * bpt, window[1] * bpt], n_ref_bytes=float(x[-1]),
                          alpha=None if g is None or not beta_tok else g / (2 * beta_tok))
    return out


def cache_key(a):
    """Everything a cached stats file depends on besides the tokenizer: manifest, reference curve, parameters."""
    sh = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest() if f and Path(f).exists() else None  # noqa: E731
    return dict(manifest=sh(Path(a.dir) / "manifest.json"), refcurve=sh(a.refcurve), n_words=a.n_words, ds=a.ds, no_beta=a.no_beta,
                data=[sh(Path(a.data) / f) for f in ("ref_stats_bytes.npy", "train_bytes.npy")])


def one(name, a):
    d, t0 = Path(a.dir), time.time()
    meta = json.loads((d / "tokenization_meta.json").read_text())
    L = np.load(d / f"tok_{name}_L.npy").astype(np.int64)
    V = len(L)
    out = dict(name=name, V=V, bytes_per_token={s: meta[f"{name}_{s}"]["bytes_per_token"] for s in ("train", "ref")},
               manifest_sha256=hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest(), cache_key=cache_key(a))
    data = Path(a.data)
    N = min(a.n_words, len(np.load(data / "ref_stats_bytes.npy", mmap_mode="r")) - 4096)
    ds = [int(x) for x in a.ds.split(",")]
    vraw = np.load(data / "ref_stats_bytes.npy", mmap_mode="r")[:N].astype(np.uint8)
    traw = np.load(data / "train_bytes.npy", mmap_mode="r")[: 2 ** max(ds) + 4096].astype(np.uint8)
    tv_full = np.load(d / f"tok_{name}_ref.npy", mmap_mode="r")
    tv = tv_full
    n = int(np.searchsorted(np.cumsum(L[np.asarray(tv[: 2 * N])]), N + 64)) + 1
    tv = np.asarray(tv[:n]).astype(np.int64)
    out["token_counts_ref_stats"] = np.bincount(np.asarray(tv_full).astype(np.int64), minlength=V).tolist()  # Zipf rival input
    out["spelling"] = spelling_stats(np.asarray(tv_full), L, vraw, N)
    tr = np.load(d / f"tok_{name}_train.npy", mmap_mode="r")
    head = np.asarray(tr[: int(2 ** max(ds) / out["bytes_per_token"]["train"] * 1.5) + 2000]).astype(np.int64)
    out["rare_spelling"] = rare_stats(tv, L, vraw, traw, head, ds, N, V)
    drop = sorted(d.glob(f"tok_{name}_drop*_s0_train.npy"))
    if drop:
        h0 = np.load(drop[0]).astype(np.int64)
        out["rare_spelling_drop_s0"] = rare_stats(tv, L, vraw, traw, h0, ds, N, V)
    print(name, "spelling/rare done", f"{time.time() - t0:.0f}s", flush=True)
    if not a.no_beta:
        out["beta"] = beta_stats(np.asarray(tv_full), V, meta[f"{name}_ref"]["n_bytes"])
        print(name, "beta", json.dumps(out["beta"]["beta"]), flush=True)
        if a.refcurve:
            gm = gamma_stats(json.loads(Path(a.refcurve).read_text()), out["bytes_per_token"]["ref"], out["beta"]["beta"]["tok_1_200"])
            out["gamma_by_model"] = gm
            prim = gm.get("SmolLM2-135M") or next(iter(gm.values()))
            # forecaster format: {beta window: gamma}; gamma is one token-unit exponent (primary reference LM), used with every beta window
            out["gamma"] = {w: prim["gamma"] for w in out["beta"]["beta"]} if prim["gamma"] is not None else None
    out["seconds"] = time.time() - t0
    (d / "stats").mkdir(exist_ok=True)
    (d / "stats" / f"{name}.json").write_text(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--data", default=str(HERE.parent / "data" / "bfine"), help="directory with train_bytes.npy and valid_bytes.npy")
    ap.add_argument("--names", default=None)
    ap.add_argument("--n_words", type=int, default=4_000_000)
    ap.add_argument("--ds", default="21,22,23", help="log2 train bytes D for the rare-spelling mass")
    ap.add_argument("--no_beta", action="store_true")
    ap.add_argument("--refcurve", default=None, help="refcurve_fineweb.json; adds gamma and alpha per reference model")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    d = Path(a.dir)
    man = json.loads((d / "manifest.json").read_text())
    if a.merge:
        res = {n: json.loads((d / "stats" / f"{n}.json").read_text()) for n in man["tokenizers"]}
        stale = [n for n, r in res.items() if r["manifest_sha256"] != hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest()]
        assert not stale, f"stats for a different family: {stale}"
        keys = {json.dumps(r["cache_key"], sort_keys=True) for r in res.values()}
        assert len(keys) == 1, "stats computed under different reference curves or parameters"
        for n, r in res.items():
            r["spec"] = man["tokenizers"][n]["spec"]
        (d / "stats.json").write_text(json.dumps(res, indent=1))
        print("merged", len(res))
        return
    msha = hashlib.sha256((d / "manifest.json").read_bytes()).hexdigest()
    for n in (a.names.split(",") if a.names else list(man["tokenizers"])):
        f = d / "stats" / f"{n}.json"
        if f.exists() and json.loads(f.read_text()).get("cache_key") == cache_key(a):
            continue  # cached for this exact family, reference curve, data and parameters
        one(n, a)


if __name__ == "__main__":
    main()
