"""Spelling multiplicity per tokenizer:  python text/spelling.py
On the first ~4 MB of the valid byte stream, split into words (rule of text/make_wpair.py). For each word occurrence,
spelling = (offset of the first overlapping token's start relative to the word start (<= 0),
            tuple of ids of all tokens overlapping the word's bytes).
Per word type: phase multiplicity = exp(entropy of the offset distribution), spelling multiplicity = exp(entropy of the
spelling distribution). Averages are weighted by word-occurrence count ("freq") and also unweighted over types ("type");
each is reported over all word types and over types with >= 20 occurrences."""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
D = HERE.parent / "data" / "text"
N = 4_000_000
TOKS = ["bytes", "bpe512", "bpe2048", "bpe8192", "pair2", "pair3", "wpair2", "wpair3", "sbpe"]


def is_alnum(b):
    return ((b >= 48) & (b <= 57)) | ((b >= 65) & (b <= 90)) | ((b >= 97) & (b <= 122)) | (b >= 128)


def word_starts(x):  # rule of text/make_wpair.py, stream start is a word start
    prev = np.empty_like(x)
    prev[1:] = x[:-1]
    prev[0] = 0
    a, pa = is_alnum(x), is_alnum(prev)
    s = (x == 32) | ~a | (a & ~pa & (prev != 32))
    s[0] = True
    return s


def byte_lengths(tok, V):
    if tok == "bytes":
        return np.ones(V, np.int64)
    if tok in ("pair2", "pair3"):
        return np.full(V, int(tok[-1]), np.int64)
    if tok in ("wpair2", "wpair3"):
        m = int(tok[-1])
        key = np.load(D / f"{tok}_chunks.npy").astype(np.int64)
        return sum(((key // 257 ** j) % 257 > 0).astype(np.int64) for j in range(m))
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(D / f"{tok}.json"))
    return np.array([len(t.id_to_token(i)) for i in range(V)], np.int64)


def mult(counts):
    n = sum(counts.values())
    return math.exp(-sum(c / n * math.log(c / n) for c in counts.values()))


def summarize(per_type):  # per_type: word -> (n, phase_mult, spell_mult)
    out = {}
    for name, minocc in (("all", 1), ("ge20", 20)):
        rows = [v for v in per_type.values() if v[0] >= minocc]
        w = sum(r[0] for r in rows)
        out[name] = dict(n_types=len(rows), n_occurrences=int(w),
                         phase_freq=sum(r[0] * r[1] for r in rows) / w, spelling_freq=sum(r[0] * r[2] for r in rows) / w,
                         phase_type=sum(r[1] for r in rows) / len(rows), spelling_type=sum(r[2] for r in rows) / len(rows))
    return out


def run(tok, raw):
    meta = json.load(open(D / "tokenization_meta.json"))
    V = meta[f"{tok}_valid"]["V"]
    L = byte_lengths(tok, V)
    t = np.load(D / f"tok_{tok}_valid.npy", mmap_mode="r")
    n = int(np.searchsorted(np.cumsum(L[np.asarray(t[: 2 * N])].astype(np.int64)), N + 64)) + 1
    t = np.asarray(t[:n]).astype(np.int64)
    bs = np.concatenate([[0], np.cumsum(L[t])])  # token i covers bytes [bs[i], bs[i+1])
    assert bs[-1] > N
    st = np.flatnonzero(word_starts(raw[:N]))
    st = np.append(st, N)  # last word may be cut; dropped below
    a, e = st[:-2], st[1:-1]  # skip the final (possibly truncated) word
    i0 = np.searchsorted(bs, a, side="right") - 1
    i1 = np.searchsorted(bs, e, side="left")
    off = bs[i0] - a
    ph, sp = defaultdict(Counter), defaultdict(Counter)
    rb = raw.tobytes()
    tl = t.tolist()
    for x, y, o, p, q in zip(a.tolist(), e.tolist(), off.tolist(), i0.tolist(), i1.tolist()):
        w = rb[x:y]
        ph[w][o] += 1
        sp[w][(o, tuple(tl[p:q]))] += 1
    per_type = {w: (sum(c.values()), mult(c), mult(sp[w])) for w, c in ph.items()}
    r = summarize(per_type)
    r["V"] = V
    r["tokens_per_word_freq"] = float(np.mean(i1 - i0))
    return r


if __name__ == "__main__":
    raw = np.load(D / "tok_bytes_valid.npy", mmap_mode="r")[:N].astype(np.uint8)
    toks = sys.argv[1:] or TOKS
    res = {}
    for tok in toks:
        res[tok] = run(tok, raw)
        print(tok, json.dumps(res[tok]["all"]), json.dumps(res[tok]["ge20"]), flush=True)
    path = HERE / "spelling.json"
    old = json.load(open(path)) if path.exists() else {}
    old.update(res)
    json.dump(old, open(path, "w"), indent=1)
