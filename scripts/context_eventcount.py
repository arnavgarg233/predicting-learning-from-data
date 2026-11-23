"""Event-count rule on contexts: the lab's K applied to n-gram histories (docs/theory/context_eventcount_2026-10-14.md).

In the gated-copy lab a dependency is learned with probability sigma(N) = 1 / (1 + (K/N)^S), N = its training events,
K ~ 210, S = 2.79 (scripts/analyze_capped_parity.py, fit_cost). On text the event of a context c of k units is a
training position whose k-unit history is c. Order k is realised at budget D on the share

    r_k(D) = sum_positions l(target) sigma(N_D(history_k)) / sum_positions l(target)

of target bytes, with N_D counted in the training prefix of D bytes. The learner backs off to the longest realised order.
Since a history's count can only fall when it is extended, the share of targets realised up to order k is r_k, and

    L(D) = mean over eval contexts j of [ g(0) - sum_{k=1..j} r_k(D) (g(k-1) - g(k)) ],   g(k) = h(k b),

where b is bytes per unit, h(n) is the byte-context curve (plug-in byte n-gram entropy for n <= 3 bytes, the reference
LM curve h_ref from its first row on, linear in log n between), and g(0) is the unigram entropy of the encoding per byte.
Contexts j run over the evaluation window (512 bytes: 1..T units, uniform, as scripts/train_text.py scores) or, for
the ~100M cells, are fixed at the ~1790-byte scoring context. No constant is fitted except, as a variant, K on TinyStories (slope kept at 2.79).

Usage
  python scripts/context_eventcount.py --count            # n-gram count histograms (slow, cached in results/tokenizer_level)
  python scripts/context_eventcount.py                    # predictions and scores -> results/tokenizer_level/context_eventcount.json
FW_TOK_DIR: the unpacked data/bfine/tok of data/bfine_colab_2026-10-13.tgz.
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from forecast_neff import HRef  # noqa: E402

OUT = os.path.join(ROOT, "results/tokenizer_level")
CACHE = os.path.join(OUT, "context_counts.npz")
FW_TOK = os.environ.get("FW_TOK_DIR", "")
TS_DIR = os.path.join(ROOT, "data/text")
K_LAB, S_LAB = 210.0, 2.79
P_HASH = np.uint64(0x9E3779B97F4A7C15)
TS_TOKS = ("bpe512", "bpe2048", "bpe8192", "wpair2", "wpair3")
FW_TOKS = ("bpe0", "bpe1", "bpe2", "bpe3", "uni0", "uni1", "uni2", "wp0", "wch0", "wch1", "fch0", "drop0", "drop1",
           "sbpe0", "sbpe1", "gpt2")      # fch1 is byte-identical to fch0 (diagnosis note), counted once
S1 = ("bpe0", "bpe1", "bpe2", "bpe3", "uni0", "uni1", "uni2", "wp0", "gpt2")
BYTE_ORDERS = list(range(1, 17)) + [20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 511]
TOK_ORDERS = list(range(1, 9)) + [10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
PLUGIN_N = 4                              # plug-in byte entropies h(0..PLUGIN_N-1)
CTX = 512                                 # small-learner window (bytes)
MID_CTX = 1790                            # ~100M learner scoring context (eval_context_bytes mean)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- data
def ts_lengths(tok, V):
    if tok == "bytes":
        return np.ones(V, np.int64)
    if tok in ("wpair2", "wpair3"):
        m = int(tok[-1])
        key = np.load(os.path.join(TS_DIR, f"{tok}_chunks.npy")).astype(np.int64)
        return sum(((key // 257 ** j) % 257 > 0).astype(np.int64) for j in range(m))
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(os.path.join(TS_DIR, f"{tok}.json"))
    return np.array([len(t.id_to_token(i)) for i in range(V)], np.int64)


def prefix_ids(x, L, D):
    """Token prefix covering D bytes, the cut scripts/train_text.py makes (uint16 ids)."""
    if len(L) == 256 and np.all(L == 1):
        assert len(x) >= D, "array shorter than the budget"
        return np.asarray(x[:D]).astype(np.uint16)
    bpt = float(L[np.asarray(x[: 2 ** 20])].mean())
    head = np.asarray(x[: int(D / bpt * 1.3) + 4096])
    cum = np.cumsum(L[head].astype(np.int32), dtype=np.int64)
    N = int(np.searchsorted(cum, D)) + 1
    assert N <= len(head) and cum[N - 1] >= D, "array shorter than the budget"
    return head[:N].astype(np.uint16)


def short_keys(key, chunk=1 << 20):
    """32-bit sort keys (high half of a multiplicative remix of the 64-bit hash), built in chunks to save memory.
    With at most ~2^24 distinct histories, collisions merge a few thousand pairs of rare histories: negligible at K ~ 200."""
    a = np.empty(len(key), np.uint32)
    for i in range(0, len(key), chunk):
        a[i:i + chunk] = (key[i:i + chunk] * np.uint64(0xD6E8FEB86659FD93)) >> np.uint64(32)
    return a


def encodings(corpus):
    """Yield (enc, [id arrays: one, or two dropout samples], L, budgets)."""
    if corpus == "ts":
        meta = json.load(open(os.path.join(TS_DIR, "tokenization_meta.json")))
        for t in ("bytes",) + TS_TOKS:
            V = meta[f"{t}_train"]["V"]
            yield t, [np.load(os.path.join(TS_DIR, f"tok_{t}_train.npy"), mmap_mode="r")], ts_lengths(t, V), \
                ((21, 23) if t.startswith("wpair") else range(20, 25))   # the budgets with runs
    else:
        for t in ("bytes",) + FW_TOKS:
            L = np.load(os.path.join(FW_TOK, f"tok_{t}_L.npy")).astype(np.int64)
            if t.startswith("drop"):   # the learner cycles through 8 resamplings; two are pooled (memory limit)
                xs = [np.load(f, mmap_mode="r") for f in sorted(glob.glob(os.path.join(FW_TOK, f"tok_{t}_drop*_s*_train.npy")))[:2]]
            elif t == "gpt2":
                xs = [gpt2_long()]
            else:
                xs = [np.load(os.path.join(FW_TOK, f"tok_{t}_train.npy"), mmap_mode="r")]
            yield t, xs, L, (range(21, 25) if t in ("bytes", "gpt2") else range(21, 24))


def gpt2_long():
    """GPT-2 ids of the first 2^24 bytes of the FineWeb-Edu bytes train array (cut after a paragraph break). The stored
    gpt2 array covers 2^23 bytes; the ~100M cells are scaled from the largest counted prefix. Encoded with
    text/tok_family.py's TikTok and checked against the stored prefix."""
    cache = os.path.join(OUT, "fw_gpt2_train_2e24.npy")
    if not os.path.exists(cache):
        sys.path.insert(0, os.path.join(ROOT, "text"))
        from tok_family import TikTok, prefix
        x = np.load(os.path.join(FW_TOK, "tok_bytes_train.npy"), mmap_mode="r")
        raw = prefix(np.asarray(x[: 2 ** 24 + 2 ** 20]).astype(np.uint8).tobytes(), 2 ** 24 + 4096)
        ids = TikTok().encode(raw)
        ref = np.load(os.path.join(FW_TOK, "tok_gpt2_train.npy"), mmap_mode="r")
        n = len(ref) - 64
        assert np.array_equal(ids[:n], np.asarray(ref[:n])), "GPT-2 re-encoding differs from the stored prefix"
        np.save(cache, ids.astype(np.uint16))
    return np.load(cache, mmap_mode="r")


# ---------------------------------------------------------------- counting (RSS kept under ~400 MB at 2^24 bytes)
def run_hist(a, w=None, chunk=1 << 20):
    """Sorted keys a (and weights w aligned with a): {run length N: summed weight of the run's positions}."""
    acc = {}

    def add(lens, ws):
        if len(lens):
            u, inv = np.unique(lens, return_inverse=True)
            for q, v in zip(u, np.bincount(inv, weights=ws)):
                acc[int(q)] = acc.get(int(q), 0.0) + float(v)

    cn, cw, n = 0, 0.0, len(a)
    for i in range(0, n, chunk):
        j = min(i + chunk, n)
        seg = a[i:j]
        bnd = np.concatenate(([0], np.flatnonzero(seg[1:] != seg[:-1]) + 1, [j - i]))
        lens = np.diff(bnd)
        ws = lens.astype(float) if w is None else np.add.reduceat(np.asarray(w[i:j], float), bnd[:-1])
        if i > 0 and seg[0] == a[i - 1]:
            lens[0] += cn; ws[0] += cw
        elif cn:
            add(np.array([cn]), np.array([cw]))
        add(lens[:-1], ws[:-1])
        cn, cw = int(lens[-1]), float(ws[-1])
    if cn:
        add(np.array([cn]), np.array([cw]))
    N = np.array(sorted(acc), float)
    return N, np.array([acc[int(q)] for q in N])


def count_encoding(xs, L, D, orders):
    """Per order k: byte-weighted histogram of k-unit history counts in the D-byte prefix
    (dropout samples pooled, counts and weights divided by the number of samples)."""
    seqs = [prefix_ids(x, L, D) for x in xs]
    S = len(seqs)
    is_bytes = len(L) == 256 and np.all(L == 1)
    out = {}
    keys = [np.zeros(len(s), np.uint64) for s in seqs]
    for k in range(1, max(orders) + 1):
        for s, key in zip(seqs, keys):   # key[i] <- polynomial hash of s[i-k:i], in place
            np.multiply(key[k:], P_HASH, out=key[k:])
            np.add(key[k:], s[:-k], out=key[k:], casting="unsafe")
        if k not in orders:
            continue
        if is_bytes:
            a = short_keys(keys[0][k:]); a.sort()
            v, w = run_hist(a)
            del a
        else:
            kk = np.concatenate([short_keys(key[k:]) for key in keys])
            ww = np.concatenate([L[s[k:]].astype(np.float32) for s in seqs])
            o = np.argsort(kk, kind="stable")
            v, w = run_hist(kk[o], ww[o])
            del kk, ww, o
            v, w = v / S, w / S
        out[k] = (v, w)
    del keys
    c = sum(np.bincount(s, minlength=len(L)) for s in seqs).astype(float); p = c[c > 0] / c.sum()
    bpt = float(c @ L / c.sum())
    return out, float(-(p * np.log2(p)).sum() / bpt), bpt


def plugin_bytes(x):
    """Plug-in conditional byte entropy h(n), n = 0..PLUGIN_N-1, on the byte array x."""
    x = np.asarray(x).astype(np.uint16)
    T = float(len(x) - PLUGIN_N)
    key = np.zeros(len(x), np.uint64)
    h = []
    for n in range(PLUGIN_N):
        if n > 0:
            np.multiply(key[n:], P_HASH, out=key[n:]); np.add(key[n:], x[:-n], out=key[n:], casting="unsafe")
        tot = 0.0
        for joint in (True, False):
            if joint:
                j = key[PLUGIN_N:].copy(); j *= np.uint64(257); np.add(j, x[PLUGIN_N:], out=j, casting="unsafe")
                a = short_keys(j); del j
            else:
                a = short_keys(key[PLUGIN_N:])
            a.sort()
            N, wt = run_hist(a)
            tot += (-1 if joint else 1) * float((wt * np.log2(N)).sum())
            del a
        h.append(tot / T)
    return h


def run_count():
    store, meta = {}, {}
    for corpus in ("ts", "fw"):
        for enc, xs, L, ks in encodings(corpus):
            orders = [k for k in (BYTE_ORDERS if enc == "bytes" else TOK_ORDERS)
                      if k < (CTX if enc == "bytes" else 2 * CTX)]
            for k in ks:
                t0 = time.time()
                try:
                    res, g0, b = count_encoding(xs, L, 2 ** k, orders)
                except AssertionError as e:
                    log(corpus, enc, k, "skip:", e); continue
                for o, (v, w) in res.items():
                    store[f"{corpus}|{enc}|{k}|{o}|N"] = v
                    store[f"{corpus}|{enc}|{k}|{o}|w"] = w
                meta[f"{corpus}|{enc}|{k}"] = dict(g0=g0, b=b, orders=sorted(res), samples=len(xs))
                log(corpus, enc, k, f"b={b:.3f} g0={g0:.3f} r1={rshare(res[1], K_LAB):.3f} r4={rshare(res[4], K_LAB):.3f}"
                    f" {time.time() - t0:.0f}s")
        xb = np.load(os.path.join(TS_DIR if corpus == "ts" else FW_TOK, "tok_bytes_train.npy"), mmap_mode="r")[: 2 ** 24]
        meta[f"{corpus}|plugin"] = plugin_bytes(xb)
        log(corpus, "plug-in h(0..3)", meta[f"{corpus}|plugin"])
        np.savez_compressed(CACHE, **store)
        json.dump(meta, open(CACHE.replace(".npz", "_meta.json"), "w"), indent=1)


# ---------------------------------------------------------------- the rule
def sigma(N, K, S=S_LAB):
    N = np.asarray(N, float)
    return 1.0 / (1.0 + (K / np.maximum(N, 1e-12)) ** S)


def rshare(vw, K, scale=1.0):
    v, w = vw
    return float((w * sigma(v * scale, K)).sum() / w.sum())


class Curve:
    """Byte-context curve h(n): plug-in byte entropies at n = 0..3, the reference LM curve from its first row on,
    linear in log n in between, constant past the last row."""

    def __init__(self, plugin, href):
        self.p, self.hr = np.array(plugin), href
        self.nr, self.hr0 = float(np.exp(href.x[0])), float(href.y[0])

    def h(self, n):
        n = np.atleast_1d(np.asarray(n, float))
        m = len(self.p) - 1
        out = np.interp(n, np.arange(m + 1), self.p)
        mid = (n > m) & (n < self.nr)
        out[mid] = np.interp(np.log(n[mid]), [np.log(m), np.log(self.nr)], [self.p[m], self.hr0])
        hi = n >= self.nr
        out[hi] = self.hr.h(n[hi])
        return out


class Model:
    def __init__(self, cache=CACHE):
        z = np.load(cache)
        self.H = {}
        for key in z.files:
            c, e, k, o, f = key.split("|")
            self.H.setdefault((c, e, int(k)), {}).setdefault(int(o), {})[f] = z[key]
        self.meta = json.load(open(cache.replace(".npz", "_meta.json")))
        fw = os.path.join(ROOT, "results/bfine/refcurve/refcurve_fineweb.json")
        self.refs = {"ts": {"TinyStories-33M": HRef(os.path.join(ROOT, "text/refcurve/refcurve.json"))},
                     "fw": {m: HRef(fw, m) for m in ("SmolLM2-135M", "pythia-160m", "gpt2")}}
        self.curves = {c: {m: Curve(self.meta[f"{c}|plugin"], hr) for m, hr in self.refs[c].items()} for c in self.refs}
        self.primary = {"ts": "TinyStories-33M", "fw": "SmolLM2-135M"}

    def has(self, c, e, k):
        return (c, e, k) in self.H

    def r_units(self, c, e, k, K, J):
        """r_1..r_J: realised share of order j, j in units. Budgets past the counted prefixes use the largest counted
        prefix with every count scaled by D / D0 (the expected count in the larger prefix)."""
        k0 = max(q for (cc, ee, q) in self.H if cc == c and ee == e and q <= k) if not self.has(c, e, k) else k
        scale = 2.0 ** (k - k0)
        H = self.H[(c, e, k0)]
        o = np.array(sorted(H), float)
        r = np.array([rshare((H[int(q)]["N"], H[int(q)]["w"]), K, scale) for q in o])
        j = np.arange(1, J + 1, dtype=float)
        return np.interp(np.log(j), np.log(o), r), k0

    def loss_curve(self, c, e, k, K, J, ref=None):
        """L(j), j = 0..J units of available context."""
        r, k0 = self.r_units(c, e, k, K, J)
        m = self.meta[f"{c}|{e}|{k0}"]
        b, g0 = m["b"], m["g0"]
        cur = self.curves[c][ref or self.primary[c]]
        g = np.concatenate([[g0], cur.h(np.arange(1, J + 1) * b)])
        return np.concatenate([[g0], g0 - np.cumsum(r * (g[:-1] - g[1:]))]), b, r

    def predict(self, c, e, k, K, mode="small", ref=None):
        b = self.meta[f"{c}|{e}|{min(k, max(q for (cc, ee, q) in self.H if cc == c and ee == e))}"]["b"]
        if mode == "small":   # 512-byte windows: targets see 1..T units of context, each scored once
            T = int(round(CTX / b))
            Lj = self.loss_curve(c, e, k, K, T, ref)[0]
            return float(Lj[1:].mean())
        J = int(round(MID_CTX / b))
        return float(self.loss_curve(c, e, k, K, J, ref)[0][-1])

    def realised_share(self, c, e, k, K, lo=11.0, hi=362.0):
        """Share of the byte-curve gain between lo and hi bytes of context that the rule realises (bytes encoding;
        lo, hi = geometric centres of the pos_bpb bins 8-15 and 256-511)."""
        J = int(hi)
        r, _ = self.r_units(c, e, k, K, J)
        cur = self.curves[c][self.primary[c]]
        g = cur.h(np.arange(0, J + 1).astype(float))
        d = (g[:-1] - g[1:]) * r
        n = np.arange(1, J + 1)
        sel = n > lo
        return float(d[sel].sum() / (cur.h(lo)[0] - cur.h(hi)[0]))


# ---------------------------------------------------------------- measured cells
def load_cells():
    cells = {}
    for tag in ("v1", "v1b", "v1d"):
        for f in sorted(glob.glob(os.path.join(ROOT, f"results/text_runs/{tag}/*.json"))):
            j = json.load(open(f)); k = int(round(np.log2(j["train_bytes"])))
            if j["tok"] in ("bytes",) + TS_TOKS and 20 <= k <= 24:
                c = cells.setdefault(("ts", j["tok"], k), dict(L=[], pos=[]))
                c["L"].append(j["bits_per_byte"])
                if j.get("pos_bpb"):
                    c["pos"].append(j["pos_bpb"])
    for f in sorted(glob.glob(os.path.join(ROOT, "results/text_runs/bfine/*.json"))):
        j = json.load(open(f)); k = int(round(np.log2(j["train_bytes"])))
        if j["tok"] == "fch1":
            continue
        c = cells.setdefault(("fw", j["tok"], k), dict(L=[], pos=[], files=[]))
        c["L"].append(j["bits_per_byte"]); c["files"].append(os.path.basename(f))
        if j.get("pos_bpb"):
            c["pos"].append(j["pos_bpb"])
    for f in sorted(glob.glob(os.path.join(ROOT, "results/midscale/*.json"))):
        j = json.load(open(f))
        if not isinstance(j, dict) or "encoding" not in j or (j["passes"], j["nl"], j["emb"]) != (1, 12, 0):
            continue
        k = int(round(np.log2(j["train_bytes"])))
        c = cells.setdefault(("mid", j["encoding"], k), dict(L=[], pos=[], files=[], b=j["bytes_per_token"]))
        c["L"].append(j["test_bpb"]); c["files"].append(os.path.basename(f))
    for c in cells.values():
        c["Lm"] = float(np.mean(c["L"])); c["n"] = len(c["L"])
    return cells


SEEN_AT_BUILD = ("FineWeb small 2^21 and 2^22 (both seeds), FineWeb ~100M 2^25-2^29, all TinyStories cells; FineWeb 2^23 "
                 "cells landed during the build and were not opened until the first scoring run")


def status(c, e, k):
    if c == "ts":
        return "calibration"
    if c == "mid":
        return "seen"
    return "seen" if k <= 22 else "landed_during_build"   # 2^23+: landed while the rule was built, or later


def rms(x):
    x = np.asarray([v for v in x if np.isfinite(v)])
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else float("nan")


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation) if len(a) > 2 else float("nan")


def ladder_shift(ladder, L, k):
    """log2 m: shift of loss L at 2^k onto a (k -> loss) ladder, linear in log2 D, end slopes outside; flag 'ext'."""
    ks = np.array(sorted(ladder), float); ys = np.array([ladder[q] for q in sorted(ladder)])
    if len(ks) < 2:
        return float("nan"), "out"
    if ys.min() <= L <= ys.max() and np.all(np.diff(ys) < 0):
        return float(np.interp(-L, -ys, ks) - k), ""
    if L > ys.max():
        return float(ks[0] - (L - ys[0]) / ((ys[0] - ys[1]) / (ks[1] - ks[0])) - k), "ext"
    return float(ks[-1] + (ys[-1] - L) / ((ys[-2] - ys[-1]) / (ks[-1] - ks[-2])) - k), "ext"


def analyse():
    M = Model()
    cells = load_cells()
    fc = json.load(open(os.path.join(ROOT, "results/bfine/forecast.json")))
    cal = fc["calibration"]; nu, n0, cfit = cal["nu"], cal["n0"], cal["c"]
    from forecast_neff import predict as fpredict
    ts_ref = M.refs["ts"]["TinyStories-33M"]; fw_ref = M.refs["fw"]["SmolLM2-135M"]

    def forecaster(c, e, k, b, cc):
        hr = ts_ref if c == "ts" else fw_ref
        return fpredict(hr, nu, n0, cc, b, k)

    def ours(c, e, k, K, ref=None):
        if c == "mid":
            return M.predict("fw", e, k, K, "mid", ref)
        if not any(cc == c and ee == e and q <= k for (cc, ee, q) in M.H):
            return float("nan")
        return M.predict(c, e, k, K, "small", ref)   # past the counted prefixes: counts scaled by D / D0

    # one constant from TinyStories: K with the slope held at 2.79
    ts_cells = [key for key in cells if key[0] == "ts"]
    grid = np.exp(np.linspace(np.log(1.0), np.log(1e4), 121))
    ts_rms = [rms([ours(*key, K) - cells[key]["Lm"] for key in ts_cells]) for K in grid]
    K_TS = float(grid[int(np.argmin(ts_rms))])
    Ks = {"K_lab": K_LAB, "K_ts": K_TS}

    rows = []
    for key in sorted(cells):
        c, e, k = key
        cd = cells[key]
        b = cd.get("b") or M.meta[f"{'fw' if c == 'mid' else c}|{e}|{min(k, 24 if e in ('bytes', 'gpt2') or c == 'ts' else 23)}"]["b"]
        row = dict(corpus=c, tok=e, log2D=k, measured=cd["Lm"], n_seeds=cd["n"], status=status(c, e, k), b=b,
                   forecaster=forecaster(c, e, k, b, cfit), c0=forecaster(c, e, k, b, 0.0))
        for name, K in Ks.items():
            row[name] = ours(c, e, k, K)
        if c != "ts":
            row["K_lab_per_ref"] = {m: ours(c, e, k, K_LAB, m) for m in M.refs["fw"]}
        rows.append(row)

    # measured and predicted multipliers (shift onto the bytes ladder of the same corpus and learner)
    for c in ("ts", "fw", "mid"):
        lad = {r["log2D"]: r["measured"] for r in rows if r["corpus"] == c and r["tok"] == "bytes"}
        for name in Ks:
            pl = {r["log2D"]: r[name] for r in rows if r["corpus"] == c and r["tok"] == "bytes"}
            if c != "ts":   # predicted bytes ladder over every counted budget (2^21-2^25; 2^25-2^29 for ~100M)
                pl = {q: ours(c, "bytes", q, Ks[name]) for q in (range(21, 26) if c == "fw" else range(25, 30))}
            for r in rows:
                if r["corpus"] == c and r["tok"] != "bytes":
                    r[f"log2m_{name}"] = ladder_shift(pl, r[name], r["log2D"])[0]
        for r in rows:
            if r["corpus"] == c and r["tok"] != "bytes":
                r["log2m_measured"], r["log2m_flag"] = ladder_shift(lad, r["measured"], r["log2D"])
                r["log2m_bc"] = cfit * np.log2(r["b"])

    # realised share of the context gain, bytes encoding, against the learner's pos_bpb share
    shares = []
    for key in sorted(cells):
        c, e, k = key
        if e != "bytes" or c == "mid":
            continue
        s = dict(corpus=c, log2D=k, **{f"rule_{n}": M.realised_share(c, "bytes", k, K) for n, K in Ks.items()})
        if cells[key]["pos"]:
            p = np.mean(cells[key]["pos"], axis=0)
            cur = M.curves[c][M.primary[c]]
            s["learner"] = float((p[3] - p[8]) / (cur.h(np.sqrt(8 * 15))[0] - cur.h(np.sqrt(256 * 511))[0]))
            # predicted pos_bpb by context bin [2^i, 2^(i+1)) against measured
            for n, K in Ks.items():
                Lj = M.loss_curve(c, "bytes", k, K, CTX - 1)[0]
                s[f"pos_pred_{n}"] = [float(Lj[1 << i:1 << (i + 1)].mean()) for i in range(9)]   # bin i: context 2^i..2^(i+1)-1
            s["pos_measured"] = [float(v) for v in p[:9]]
        shares.append(s)

    # scores
    def group(pred, sel):
        rr = [r for r in rows if sel(r) and np.isfinite(r.get(pred, np.nan))]
        return dict(n=len(rr), rms=rms([r[pred] - r["measured"] for r in rr]),
                    mean=float(np.mean([r[pred] - r["measured"] for r in rr])) if rr else float("nan"))

    groups = {
        "ts_all": lambda r: r["corpus"] == "ts",
        "ts_tok": lambda r: r["corpus"] == "ts" and r["tok"] != "bytes",
        "fw_seen_S1": lambda r: r["corpus"] == "fw" and r["status"] == "seen" and r["tok"] in S1,
        "fw_seen_all_tok": lambda r: r["corpus"] == "fw" and r["status"] == "seen" and r["tok"] != "bytes",
        "fw_seen_bytes": lambda r: r["corpus"] == "fw" and r["status"] == "seen" and r["tok"] == "bytes",
        "fw_2e23_landed_during_build": lambda r: r["corpus"] == "fw" and r["status"] == "landed_during_build",
        "mid_all": lambda r: r["corpus"] == "mid",
    }
    preds = ("K_lab", "K_ts", "forecaster", "c0")
    scores = {g: {p: group(p, f) for p in preds} for g, f in groups.items()}
    mscores = {}
    for g, f in groups.items():
        rr = [r for r in rows if f(r) and r["tok"] != "bytes" and np.isfinite(r.get("log2m_measured", np.nan))]
        if not rr:
            continue
        d = dict(n=len(rr))
        for p in ("K_lab", "K_ts"):
            d[p] = rms([r[f"log2m_{p}"] - r["log2m_measured"] for r in rr])
        d["b^c"] = rms([r["log2m_bc"] - r["log2m_measured"] for r in rr])
        d["m=1"] = rms([r["log2m_measured"] for r in rr])
        d["mean_measured"] = float(np.mean([r["log2m_measured"] for r in rr]))
        for p in ("K_lab", "K_ts"):
            d[f"mean_{p}"] = float(np.mean([r[f"log2m_{p}"] for r in rr]))
            d[f"spearman_loss_{p}"] = float(np.mean([spearman([r[p] for r in rr if r["log2D"] == q],
                                                               [r["measured"] for r in rr if r["log2D"] == q])
                                                      for q in sorted({r["log2D"] for r in rr})]))
        d["spearman_loss_b"] = float(np.mean([spearman([-np.log(r["b"]) for r in rr if r["log2D"] == q],
                                                       [r["measured"] for r in rr if r["log2D"] == q])
                                              for q in sorted({r["log2D"] for r in rr})]))
        mscores[g] = d

    out = dict(constants=dict(K_lab=K_LAB, S=S_LAB, K_ts=K_TS, ts_K_scan=dict(K=grid.tolist(), rms=ts_rms),
                              forecaster=dict(nu=nu, n0=n0, c=cfit), plugin={c: M.meta[f"{c}|plugin"] for c in ("ts", "fw")}),
               seen_at_build=SEEN_AT_BUILD, scores=scores, multiplier_scores=mscores, realised_share=shares,
               cells=rows,
               unseen_forecasts=[dict(tok=t, log2D=k, **{n: ours("fw", t, k, K) for n, K in Ks.items()})
                                 for t in ("bytes",) + FW_TOKS for k in ((23, 24, 25) if t == "bytes" else (23,))])
    json.dump(out, open(os.path.join(OUT, "context_eventcount.json"), "w"), indent=1, default=float)
    return out


def report(o):
    print(f"K_lab {K_LAB}  K_ts {o['constants']['K_ts']:.1f}   plug-in h(0..3): {o['constants']['plugin']}")
    print("\n== loss rms (b/B): group n | K_lab | K_ts | forecaster | c0 ==")
    for g, d in o["scores"].items():
        print(f"{g:18s} {d['K_lab']['n']:3d} | " + " | ".join(f"{d[p]['rms']:.3f} ({d[p]['mean']:+.3f})" for p in d))
    print("\n== log2 m rms: group n | K_lab | K_ts | b^c | m=1 | mean meas | mean K_lab | mean K_ts | rho K_lab, K_ts, b ==")
    for g, d in o["multiplier_scores"].items():
        print(f"{g:18s} {d['n']:3d} | {d['K_lab']:.2f} | {d['K_ts']:.2f} | {d['b^c']:.2f} | {d['m=1']:.2f} | "
              f"{d['mean_measured']:+.2f} | {d['mean_K_lab']:+.2f} | {d['mean_K_ts']:+.2f} | "
              f"{d['spearman_loss_K_lab']:+.2f} {d['spearman_loss_K_ts']:+.2f} {d['spearman_loss_b']:+.2f}")
    print("\n== realised share of context gain (11 -> 362 bytes), bytes ==")
    for s in o["realised_share"]:
        print(s["corpus"], s["log2D"], f"rule K_lab {s['rule_K_lab']:.3f}  K_ts {s['rule_K_ts']:.3f}  learner {s.get('learner', float('nan')):.3f}")
    print("\n== cells ==")
    for r in o["cells"]:
        print(f"{r['corpus']:3s} {r['tok']:8s} 2^{r['log2D']} {r['status']:11s} meas {r['measured']:.3f}  K_lab {r['K_lab']:.3f}  "
              f"K_ts {r['K_ts']:.3f}  fc {r['forecaster']:.3f}  c0 {r['c0']:.3f}  "
              + (f"log2m meas {r['log2m_measured']:+.2f}{r['log2m_flag']} K_ts {r['log2m_K_ts']:+.2f} bc {r['log2m_bc']:+.2f}"
                 if "log2m_measured" in r else ""))


MC_ORDERS = list(range(1, 17)) + [20, 24, 32, 48, 64, 96, 128]   # byte orders for new corpora (multicorpus plan, 2a.3)


def find_train(d):
    for n in ("tok_bytes_train.npy", "train_bytes.npy"):
        if os.path.exists(os.path.join(d, n)):
            return os.path.join(d, n)
    raise FileNotFoundError(f"no tok_bytes_train.npy or train_bytes.npy in {d}")


def count_corpus(corpus, bytes_dir, budgets, out=None, orders=MC_ORDERS):
    """Byte-history counts only, for one corpus (residues for protein): the same store layout as run_count
    ('{corpus}|bytes|{k}|{order}|N' and '|w', meta '{corpus}|bytes|{k}' and '{corpus}|plugin'). Cached at out
    (default results/mc/<corpus>/counts.npz); the cache is reused when it holds every budget and the train file is unchanged."""
    out = out or os.path.join(ROOT, "results/mc", corpus, "counts.npz")
    f = find_train(bytes_dir)
    x = np.load(f, mmap_mode="r")
    stamp = dict(file=os.path.abspath(f), n=int(len(x)), head=hashlib.sha256(np.asarray(x[:65536]).tobytes()).hexdigest(),
                 budgets=list(budgets), orders=list(orders))
    mp = out.replace(".npz", "_meta.json")
    if os.path.exists(out) and os.path.exists(mp) and json.load(open(mp)).get("stamp") == stamp:
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    L = np.ones(256, np.int64)
    store, meta = {}, {}
    for k in budgets:
        t0 = time.time()
        res, g0, b = count_encoding([x], L, 2 ** k, [o for o in orders if o < CTX])
        for o, (v, w) in res.items():
            store[f"{corpus}|bytes|{k}|{o}|N"] = v
            store[f"{corpus}|bytes|{k}|{o}|w"] = w
        meta[f"{corpus}|bytes|{k}"] = dict(g0=g0, b=b, orders=sorted(res), samples=1)
        log(corpus, k, f"g0={g0:.3f} r1={rshare(res[1], K_LAB):.3f} {time.time() - t0:.0f}s")
    meta[f"{corpus}|plugin"] = plugin_bytes(np.asarray(x[: 2 ** 24]))
    meta["stamp"] = stamp
    np.savez_compressed(out, **store)
    json.dump(meta, open(mp, "w"), indent=1)
    return out


def j_half(counts_npz, corpus, k, K, S=S_LAB):
    """Byte order at which r_j = mean sigma(N_hist) falls to 0.5 (log-linear interpolation, as level_offset_predictor.count_context).
    Returns (j_half, censored): if r is still >= 0.5 at the largest counted order, j_half is that order and censored is True."""
    z = counts_npz if hasattr(counts_npz, "files") else np.load(counts_npz)
    orders = sorted({int(q.split("|")[3]) for q in z.files if q.startswith(f"{corpus}|bytes|{k}|")})
    r = np.array([float((z[f"{corpus}|bytes|{k}|{o}|w"] * sigma(z[f"{corpus}|bytes|{k}|{o}|N"], K, S)).sum()
                        / z[f"{corpus}|bytes|{k}|{o}|w"].sum()) for o in orders])
    lo = np.log(orders)
    if r[0] < 0.5:
        return float(orders[0]), False
    if r[-1] >= 0.5:
        return float(orders[-1]), True
    return float(np.exp(np.interp(-0.5, -r, lo))), False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", action="store_true")
    ap.add_argument("--corpus", default=None, help="count byte histories of one corpus only (needs --bytes_dir); default: the ts and fw tokenizer counts")
    ap.add_argument("--bytes_dir", default=None, help="directory with tok_bytes_train.npy or train_bytes.npy")
    ap.add_argument("--budgets", default="21,22,23", help="log2 byte budgets (with --corpus)")
    ap.add_argument("--out", default=None, help="counts .npz (with --corpus; default results/mc/<corpus>/counts.npz)")
    a = ap.parse_args()
    if a.corpus:
        assert a.count and a.bytes_dir, "--corpus needs --count and --bytes_dir"
        print(count_corpus(a.corpus, a.bytes_dir, [int(q) for q in a.budgets.split(",")], a.out))
    elif a.count:
        assert FW_TOK and os.path.exists(os.path.join(FW_TOK, "manifest.json")), "set FW_TOK_DIR"
        os.makedirs(OUT, exist_ok=True)
        run_count()
    else:
        report(analyse())
