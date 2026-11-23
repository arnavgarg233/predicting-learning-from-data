"""RETROSPECTIVE rare-spelling mass M_k(D) and its relation to the measured tax:  python text/rare_spelling.py
M_k(D)  = fraction of word occurrences in the first ~4 MB of valid (word rule and spelling of text/spelling.py) whose
          spelling occurs < k times among word occurrences in the first D bytes of train (prefix rule of scripts/train_text.py).
Mt_k(D) = fraction of those valid tokens whose token type occurs < k times in the same train prefix (Mtb: byte-weighted).
Spellings are compared through a 64-bit polynomial hash of (offset, ids of overlapping tokens).
Stage 2 relates the excess over the compression rule to M_k (one constant through the origin), see text/RARE_SPELLING.md."""
import glob, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spelling import word_starts, byte_lengths, TOKS, N as NV
ROOT = HERE.parent
DD = ROOT / "data" / "text"
KS = (1, 3, 10, 30)
DS = [20, 21, 22, 23, 24]
REF = ["bytes", "bpe512", "bpe2048", "bpe8192"]
P = np.uint64(0x9E3779B97F4A7C15 | 1)
PINV = np.uint64(pow(int(P), -1, 2 ** 64))
np.seterr(over="ignore")


def powers(base, n):
    p = np.empty(n, np.uint64); p[0] = 1; p[1:] = base
    return np.cumprod(p, dtype=np.uint64)


def word_hashes(t, L, raw, nb):
    """spelling hash for each complete word of raw[:nb]; returns (hash, word_end)."""
    bs = np.concatenate([[0], np.cumsum(L[t])]); assert bs[-1] >= nb
    st = np.append(np.flatnonzero(word_starts(raw[:nb])), nb)
    a, e = st[:-2], st[1:-1]
    i0 = np.searchsorted(bs, a, side="right") - 1
    i1 = np.searchsorted(bs, e, side="left")
    S = np.concatenate([np.zeros(1, np.uint64), np.cumsum((t.astype(np.uint64) + np.uint64(1)) * powers(P, len(t)), dtype=np.uint64)])
    h = (S[i1] - S[i0]) * powers(PINV, len(t))[i0]
    off = (bs[i0] - a).astype(np.uint64)
    h = (h ^ (off * np.uint64(0xC2B2AE3D27D4EB4F))) * np.uint64(0x165667B19E3779F9) + (i1 - i0).astype(np.uint64)
    return h, e


def mass(counts):  # counts: train count of each valid item
    return {k: float(np.mean(counts < k)) for k in KS}


def run(tok, vraw, traw):
    meta = json.load(open(DD / "tokenization_meta.json")); V = meta[f"{tok}_valid"]["V"]
    bpt = meta[f"{tok}_train"]["bytes_per_token"]
    L = byte_lengths(tok, V)
    tv = np.load(DD / f"tok_{tok}_valid.npy", mmap_mode="r")
    n = int(np.searchsorted(np.cumsum(L[np.asarray(tv[: 2 * NV])]), NV + 64)) + 1
    tv = np.asarray(tv[:n]).astype(np.int64)
    hv, _ = word_hashes(tv, L, vraw, NV)
    uv, inv = np.unique(hv, return_inverse=True)
    bv = np.concatenate([[0], np.cumsum(L[tv])]); ntv = int(np.searchsorted(bv, NV, side="right")) - 1  # tokens ending <= NV
    tvv, lv = tv[:ntv], L[tv[:ntv]]
    tr = np.load(DD / f"tok_{tok}_train.npy", mmap_mode="r")
    head = np.asarray(tr[: int(2 ** max(DS) / bpt * 1.5) + 2000]).astype(np.int64)
    cum = np.cumsum(L[head])
    Ns = {k: int(np.searchsorted(cum, 2 ** k)) + 1 for k in DS}
    nbs = {k: int(cum[Ns[k] - 1]) for k in DS}
    nmax = Ns[max(DS)]
    ht, et = word_hashes(head[:nmax], L, traw, nbs[max(DS)])
    out = {}
    for k in DS:
        m = int(np.searchsorted(et, nbs[k] - 1, side="right"))  # words ending before the last prefix byte
        u, c = np.unique(ht[:m], return_counts=True)
        pos = np.minimum(np.searchsorted(u, uv), len(u) - 1)
        cu = np.where(u[pos] == uv, c[pos], 0)[inv]
        tc = np.bincount(head[: Ns[k]], minlength=V)[tvv]
        r = {f"M{q}": v for q, v in mass(cu).items()}
        r.update({f"Mt{q}": float(np.mean(tc < q)) for q in KS})
        r.update({f"Mtb{q}": float(np.sum(lv[tc < q]) / lv.sum()) for q in KS})
        r.update(n_word_occ_valid=len(hv), n_word_occ_train=m, n_spellings_train=len(u), train_bytes=nbs[k])
        out[str(k)] = r
    return out


def measured():
    R = {}
    for d in ("v1", "v1b", "v1c", "v1d", "v1e"):
        for f in sorted(glob.glob(str(ROOT / f"results/text_runs/{d}/*.json"))):
            j = json.load(open(f)); k = int(os.path.basename(f).split("_b")[1].split("_")[0])
            R.setdefault(j["tok"], {}).setdefault(k, []).append(j["bits_per_byte"])
    return {t: {k: float(np.mean(v)) for k, v in R[t].items()} for t in R}


def analyse(M):
    meta = json.load(open(DD / "tokenization_meta.json"))
    BPT = {t: meta[f"{t}_valid"]["bytes_per_token"] for t in TOKS}
    Lm = measured(); X = np.log([BPT[r] for r in REF]); ks = sorted(Lm["bytes"])
    def curve(t, mode):
        x = np.log(BPT[t]); o = {}
        for k in ks:
            y = np.array([Lm[r][k] for r in REF])
            if x <= X.max(): o[k] = float(np.interp(x, X, y))
            elif mode == "clamp": o[k] = float(y[-1])
            else:
                b, a = np.polyfit(X[1:], y[1:], 1); o[k] = float(a + b * x)
        return o
    def tax(c, k, y):  # data multiplier, as in scripts/score_sbpe.py
        x = np.array(ks, float); cc = np.array([c[q] for q in ks])
        ke = x[0] + (y - cc[0]) * (x[1] - x[0]) / (cc[1] - cc[0]) if y > cc[0] else float(np.interp(-y, -cc, x))
        return 2 ** (k - ke)
    cells = []
    for t in TOKS:
        for k in (20, 21, 22, 23, 24):
            if k not in Lm.get(t, {}): continue
            modes = ("clamp", "extrap") if t == "sbpe" else ("ref",)
            for mode in modes:
                c = curve(t, "clamp" if mode == "clamp" else "extrap")
                m = dict(M[t][str(k)]); m.update({f"dM{q}": m[f"M{q}"] - M["bytes"][str(k)][f"M{q}"] for q in KS})  # beyond word-type rarity
                cells.append(dict(tok=t, k=k, mode=mode, bpb=Lm[t][k], ref=c[k], excess=Lm[t][k] - c[k],
                                  tax=tax(c, k, Lm[t][k]) if t not in REF else None, **m))
    fitset = [c for c in cells if c["tok"] not in REF]
    fits = {}
    for mode in ("clamp", "extrap"):
        cs = [c for c in fitset if c["mode"] in ("ref", mode)]
        for var in [f"{p}{q}" for p in ("M", "dM", "Mt", "Mtb") for q in KS]:
            for target in ("excess", "log2tax"):
                y = np.array([c["excess"] if target == "excess" else np.log2(c["tax"]) for c in cs])
                x = np.array([c[var] for c in cs]); tk = [c["tok"] for c in cs]
                cf = float(x @ y / (x @ x)); rms = float(np.sqrt(np.mean((y - cf * x) ** 2)))
                loo = {}
                for h in sorted(set(tk)):
                    tr = np.array([a != h for a in tk]); ch = float(x[tr] @ y[tr] / (x[tr] @ x[tr]))
                    loo[h] = dict(c=ch, pred=[float(ch * v) for v in x[~tr]], actual=[float(v) for v in y[~tr]])
                lr = float(np.sqrt(np.mean([(p - a) ** 2 for v in loo.values() for p, a in zip(v["pred"], v["actual"])])))
                fits[f"{mode}|{target}|{var}"] = dict(c=cf, rms=rms, rms_zero=float(np.sqrt(np.mean(y ** 2))), loo_rms=lr, loo=loo)
    return cells, fits


if __name__ == "__main__":
    pj = HERE / "rare_spelling.json"
    prev = json.load(open(pj)) if pj.exists() else {}
    M = prev.get("M", {})
    vraw = np.load(DD / "tok_bytes_valid.npy", mmap_mode="r")[:NV].astype(np.uint8)
    traw = np.load(DD / "tok_bytes_train.npy", mmap_mode="r")[: 2 ** max(DS) + 4096].astype(np.uint8)
    for t in sys.argv[1:] or TOKS:
        if t in M and not sys.argv[1:]: continue
        M[t] = run(t, vraw, traw); print(t, json.dumps({k: round(v["M3"], 4) for k, v in M[t].items()}), flush=True)
        json.dump(dict(label="RETROSPECTIVE", M=M), open(pj, "w"), indent=1)
    if all(t in M for t in TOKS):
        cells, fits = analyse(M)
        json.dump(dict(label="RETROSPECTIVE", M=M, cells=cells, fits=fits), open(pj, "w"), indent=1)
