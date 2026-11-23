"""Count-based byte information curve h_ng(n) (interpolated absolute discounting, D = n1/(n1 + 2 n2) per order), n = 0..24.
Feasibility check on TinyStories (design note of 2026-10-13, section 4): with no arguments it trains on
the first 2^25 bytes of the TinyStories train stream and scores 2^17 valid positions, as before. Run from the repo root:
  python scripts/ngram_curve_ts.py
For any corpus (docs/plan/multicorpus_level_test_2026-10-15.md, 2a):
  python scripts/ngram_curve_ts.py --train DIR --D 2^23 --eval ref_stats [--out curve.json]
DIR holds tok_bytes_train.npy (a tok_family bytes build, which also has tok_bytes_ref.npy = ref_stats) or {train,ref_stats}_bytes.npy.
--eval ref_stats scores 2^17 ref_stats positions (valid and test are never read); --eval valid keeps the TinyStories behaviour.
"""
import argparse
import json
import os
import time
import numpy as np

K = 24
M = np.uint64(1000003)


def parse_bytes(s):
    return int(2 ** int(s[2:])) if str(s).startswith("2^") else int(s)


def find_array(d, names):
    for n in names:
        if os.path.exists(os.path.join(d, n)):
            return os.path.join(d, n)
    raise FileNotFoundError(f"none of {names} in {d}")


def ngram_curve(tr, va, pos, log=print):
    """tr, va: uint64 byte arrays; pos: eval positions in va (each >= K). Returns a list of dicts (n, h_ng bits/byte, seen, D)."""
    NTR, NEV = len(tr), len(pos)
    t0 = time.time()
    ctx_tr = np.zeros(NTR, np.uint64)
    ctx_ev = np.zeros(NEV, np.uint64)
    p = np.full(NEV, 1.0 / 256)
    xe = va[pos]
    rows = []
    with np.errstate(over="ignore"):
        for k in range(0, K + 1):
            if k > 0:
                ctx_tr[k:] = ctx_tr[k:] * M + tr[: NTR - k] + np.uint64(k * 7919 + 1)
                ctx_ev = ctx_ev * M + va[pos - k] + np.uint64(k * 7919 + 1)
            c = ctx_tr[k:]
            key = c * M + tr[k:] + np.uint64(1)
            order = np.argsort(key, kind="stable")
            ks, cs = key[order], c[order]
            first = np.r_[True, ks[1:] != ks[:-1]]
            uk = ks[first]
            ck = np.diff(np.r_[np.flatnonzero(first), len(ks)])
            uctx_of_key = cs[first]
            uc, cc = np.unique(c, return_counts=True)
            uc2, n1p = np.unique(uctx_of_key, return_counts=True)  # same set as uc
            n1, n2 = np.sum(ck == 1), np.sum(ck == 2)
            # n1 = 0 (typical at order 0 on code/ABC: every byte value seen >= 3 times) made D = 0/0 or 0, so unseen bytes
            # got p = 0 (h NaN or inf). Count at least one singleton; D is unchanged whenever n1 > 0. Fixed 2026-10-16.
            Dk = max(n1, 1) / (max(n1, 1) + 2 * n2)
            ke = ctx_ev * M + xe + np.uint64(1)
            i = np.clip(np.searchsorted(uk, ke), 0, len(uk) - 1)
            cw = np.where(uk[i] == ke, ck[i], 0)
            j = np.clip(np.searchsorted(uc, ctx_ev), 0, len(uc) - 1)
            hit = uc[j] == ctx_ev
            cctx = np.where(hit, cc[j], 0)
            nn = np.where(hit, n1p[j], 0)
            pk = np.where(cctx > 0, (np.maximum(cw - Dk, 0) + Dk * nn * p) / np.maximum(cctx, 1), p)
            p = pk
            h = float(-np.mean(np.log2(p)))
            rows.append(dict(n=k, h_ng=h, seen=float(np.mean(hit)), D=float(Dk)))
            log(f"n={k:2d} bytes  h_ng={h:.4f} bits/byte  context seen={np.mean(hit):.3f}  D={Dk:.3f}  {time.time() - t0:.0f}s")
            del order, ks, cs, first, key
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=None, help="corpus directory (default: data/text, TinyStories)")
    ap.add_argument("--D", type=parse_bytes, default=2 ** 25, help="training prefix in bytes")
    ap.add_argument("--eval", default="valid", choices=["valid", "ref_stats"])
    ap.add_argument("--nev", type=parse_bytes, default=2 ** 17)
    ap.add_argument("--out", default=None, help="write the curve as JSON")
    a = ap.parse_args()
    d = a.train or "data/text"
    tr_f = find_array(d, ["tok_bytes_train.npy", "train_bytes.npy"])
    ev_f = find_array(d, ["tok_bytes_valid.npy", "valid_bytes.npy"] if a.eval == "valid" else ["tok_bytes_ref.npy", "ref_stats_bytes.npy"])
    tr = np.load(tr_f, mmap_mode="r")
    assert len(tr) >= a.D, f"{tr_f} has {len(tr)} bytes, need {a.D}"
    tr = np.asarray(tr[: a.D], np.uint64)
    va = np.asarray(np.load(ev_f, mmap_mode="r"), np.uint64)
    nev = min(a.nev, len(va) - K)
    rng = np.random.default_rng(0)
    pos = np.sort(rng.choice(np.arange(K, len(va)), nev, replace=False))
    rows = ngram_curve(tr, va, pos)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(dict(train=tr_f, eval=ev_f, D=a.D, n_eval=nev, rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
