"""Seeded tokenizer family for the randomized prospective test (docs/plan/prospective_forecast_2026-10-12.md, B-FineWeb).

  python text/tok_family.py --seed_file results/bfine/seed.txt --data data/bfine --out data/bfine/tok

Input: an integer seed (taken later from a public randomness beacon; --seed N or --seed_file, decimal or 0x-hex) and a
byte partitions: {tokenizer_fit,ref_stats,train,valid,test}_bytes.npy (uint8, scripts/prep_fineweb.py). Tokenizers are fit on tokenizer_fit only;
statistics use ref_stats (and the train prefix for training-side counts). Accept/redraw decisions and the family are frozen on
tokenizer_fit, train, ref_stats, valid and the adversarial sample ONLY; test is never read until `--encode_test`, run after the forecast is
made (it encodes and checks test; a failure invalidates the family, it never triggers a redraw).
The seed fixes every random draw (vocabulary sizes, chunk lengths, dropout rates, merge counts); a given seed always
gives the same specs. BPE, superword and chunk tokenizers retrain identically. Unigram and WordPiece training is not
bit-reproducible (multi-threaded), and BPE-dropout resamplings use the tokenizers library's internal RNG, so the saved
files (sha256 in the manifest) are the fixed artifact.

Strata (18 tokenizers: 10 in S1, 6 in S2, 2 anchors):
  S1 one spelling per word:  bpe0-3   byte-level BPE (GPT-2 regex pre-tokenization), V log-uniform 384-16384
                             uni0-2   byte-level unigram LM (SentencePiece-style), V log-uniform 512-16384
                             wp0      byte-level WordPiece, V log-uniform 512-16384
                             wch0-1   word-aligned chunks of L bytes, L uniform {2..6}
  S2 several spellings:      fch0-1   fixed chunks of L bytes, L uniform {2..5}
                             drop0-1  BPE-dropout p uniform 0.05-0.3 on a random-V BPE (train arrays are resegmented, eval canonical)
                             sbpe0-1  superword BPE: BPE at V1 = 8192 plus n2 greedy second-stage merges, n2 log-uniform 512-8192
  anchors:                   bytes, gpt2 (tiktoken)

Segmentation: BPE, unigram, WordPiece, superword and tiktoken tokenizers see the text as pieces that end after every
"\\n\\n" (so no token crosses a paragraph break); chunk tokenizers cut the byte stream directly. Every tokenizer is
checked to be lossless on the training prefix, valid and test: the byte strings of the emitted ids, concatenated, equal the input.

Output (all under --out):
  manifest.json            seed, args, corpus hashes, per tokenizer: spec, V, bytes/token (train prefix, ref_stats, valid), sha256 of every array, table and tokenizer file
  tokenization_meta.json   {name}_{train,ref,valid}: V, n_tokens, n_bytes, bytes_per_token (the format scripts/train_text.py reads)
  tok_{name}_{train,ref,valid}.npy  uint16 ids; tok_{name}_test.npy and test_manifest.json only after --encode_test (train = first enc_bytes of train, cut at a paragraph break)
  tok_{name}_L.npy         int64 byte length of every id
  tok_{name}_drop{p}_s{i}_train.npy  BPE-dropout resamplings of the same train prefix (drop*)
  {name}.json              HF tokenizers file (BPE, unigram, WordPiece, superword);  {name}_chunks.npy  chunk keys by id (chunks)
Train with:  python scripts/train_text.py --tok NAME --tok_dir OUT --test_split [--bpe_dropout P --dropout_dir OUT for drop*]
"""
import argparse
import hashlib
import heapq
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("RAYON_NUM_THREADS", "8")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spelling import word_starts  # noqa: E402  (word rule of text/make_wpair.py)

SPLITS = ("train", "valid", "test")
SEP = b"\n\n"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


U2B = {v: k for k, v in bytes_to_unicode().items()}
WP_PREFIX = "\u0001"  # WordPiece continuation marker: not a byte-level character, so it cannot clash with text


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes() if isinstance(x, np.ndarray) else x).hexdigest()


def parse_seed(s):
    s = s.strip()
    return int(s, 16) if re.fullmatch(r"[0-9a-fA-F]+", s) and re.search(r"[a-fA-F]", s) else int(s, 0)


def pieces(raw):
    """Text pieces that each end after '\\n\\n' (the last one may not)."""
    return [p for p in re.split(r"(?<=\n\n)", raw.decode("utf-8")) if p]


def prefix(raw, nbytes):
    """First >= nbytes of raw, cut after a paragraph break (all of raw if it is shorter)."""
    if len(raw) <= nbytes:
        return raw[: raw.rfind(SEP) + 2] if raw.rfind(SEP) >= 0 else raw
    e = raw.find(SEP, max(nbytes - 2, 0))
    return raw if e < 0 else raw[: e + 2]


# ------------------------------------------------------------------ the sampler

# template index -> (name, family, stratum); parameters are drawn from default_rng([seed, template index, attempt]).
# Precommitted reject-and-redraw rule: a draw is rejected, and the same template redrawn with attempt + 1 (at most MAX_ATTEMPTS),
# when (a) the tokenizer is not lossless on the fit/train/ref/valid/test arrays or on the adversarial byte sample, (b) an HF
# tokenizer has a normalizer, or its vocabulary is below 80% of the requested size, (c) an S1 draw has frequency-weighted
# spelling multiplicity above SPELL_MAX on ref_stats (it would not be one spelling per word). Every rejection is in the manifest.
TEMPLATES = ([(f"bpe{i}", "bpe", "S1") for i in range(4)] + [(f"uni{i}", "unigram", "S1") for i in range(3)] + [("wp0", "wordpiece", "S1")]
             + [(f"wch{i}", "chunk", "S1") for i in range(2)] + [(f"fch{i}", "chunk", "S2") for i in range(2)]
             + [(f"drop{i}", "bpe_dropout", "S2") for i in range(2)] + [(f"sbpe{i}", "superword", "S2") for i in range(2)]
             + [("bytes", "bytes", "anchor"), ("gpt2", "tiktoken", "anchor")])
MAX_ATTEMPTS = 5
SPELL_MAX = 1.5  # TinyStories one-spelling BPEs measure 1.12-1.16, wpair 1.0; pair2 6.8, pair3 42, sbpe 153
ADV = ("".join(chr(c) for c in range(0x80, 0x3000, 5) if not 0xD800 <= c < 0xE000) + "\x00\x01\x1f\x7f\t\r\n\r\n\n\n\n" + "9" * 3000 + "0123456789" * 200
       + "\U0001F600\U0001F9EC\U0001F1FA\U0001F1F8\U0001F468\u200d\U0001F469\u200d\U0001F467\u200b\u202e\ufeff\u0301e\u0301" + " " * 500 + "a" * 5000
       + "\U0010FFFF\uFFFD\n\n").encode("utf-8")


class Reject(Exception):
    pass


def draw_spec(seed, ti, attempt, a):
    name, fam, stratum = TEMPLATES[ti]
    rng = np.random.default_rng([seed, ti, attempt])

    def logu(lo, hi):
        return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))

    sp = dict(name=name, family=fam, stratum=stratum, attempt=attempt)
    if fam == "bpe":
        sp["V"] = logu(a.v_lo, a.v_hi)
    elif fam in ("unigram", "wordpiece"):
        sp["V"] = logu(a.uni_lo, a.v_hi)
    elif fam == "chunk":
        sp.update(L=int(rng.integers(2, 7 if stratum == "S1" else 6)), word_aligned=stratum == "S1")
    elif fam == "bpe_dropout":
        sp.update(V=logu(a.v_lo, a.v_hi), p=round(float(rng.uniform(0.05, 0.3)), 3))
    elif fam == "superword":
        sp.update(V1=a.sbpe_v1, n2=logu(a.n2_lo, a.n2_hi))
    return sp


def sample_specs(seed, a):
    """First-attempt draws (what the dry run prints); redraws happen in main only after a rejection."""
    return [draw_spec(seed, ti, 0, a) for ti in range(len(TEMPLATES))]


# ------------------------------------------------------------------ tokenizers

class Tok:
    """table: bytes of every id. encode(raw) -> uint16 ids."""
    table = None

    def lengths(self):
        return np.array([len(t) for t in self.table], np.int64)


class ByteTok(Tok):
    def __init__(self):
        self.table = [bytes([i]) for i in range(256)]

    def encode(self, raw):
        return np.frombuffer(raw, np.uint8).astype(np.uint16)


class TikTok(Tok):
    def __init__(self):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.table = [self.enc.decode_single_token_bytes(i) for i in range(self.enc.n_vocab)]

    def encode(self, raw):
        out = []
        ps = pieces(raw)
        for i in range(0, len(ps), 20000):
            out += [np.asarray(t, np.uint16) for t in self.enc.encode_ordinary_batch(ps[i:i + 20000], num_threads=8)]
        return np.concatenate(out)


class HFTok(Tok):
    def __init__(self, path, wordpiece=False):
        from tokenizers import Tokenizer
        self.tok, self.path = Tokenizer.from_file(str(path)), str(path)
        self.wordpiece = wordpiece
        self.table = []
        for i in range(self.tok.get_vocab_size()):
            s = self.tok.id_to_token(i)
            if wordpiece and s.startswith(WP_PREFIX):
                s = s[1:]
            self.table.append(bytes(U2B[c] for c in s) if all(c in U2B for c in s) else b"")  # [UNK] -> b"", caught by the lossless check

    def encode(self, raw, dropout=None):
        if dropout is not None:
            self.tok.model.dropout = dropout
        out = []
        ps = pieces(raw)
        for i in range(0, len(ps), 20000):
            out += [np.asarray(e.ids, np.uint16) for e in self.tok.encode_batch(ps[i:i + 20000])]
        if dropout is not None:
            self.tok.model.dropout = None
        return np.concatenate(out)

    def decode_ok(self, ids, raw, n=300_000):
        ids = ids[:n]
        want = b"".join(self.table[i] for i in ids.tolist())
        return self.tok.decode(ids.tolist(), skip_special_tokens=False).encode("utf-8") == want == raw[:len(want)]


def train_bpe(ps, V, path):
    from tokenizers.implementations import ByteLevelBPETokenizer
    t = ByteLevelBPETokenizer()
    t.train_from_iterator(ps, vocab_size=V, min_frequency=2, special_tokens=[], show_progress=False)
    t.save(str(path))


def train_unigram(ps, V, path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    t = Tokenizer(models.Unigram())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.UnigramTrainer(vocab_size=V, special_tokens=[], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                 unk_token=None, show_progress=False, max_piece_length=16)
    t.train_from_iterator(ps, tr)
    t.save(str(path))


def train_wordpiece(ps, V, path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    tr = trainers.WordPieceTrainer(vocab_size=V, special_tokens=["[UNK]"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                   continuing_subword_prefix=WP_PREFIX, show_progress=False)
    t.train_from_iterator(ps, tr)
    spec = json.loads(t.to_str())
    vocab = spec["model"]["vocab"]
    for c in pre_tokenizers.ByteLevel.alphabet():  # continuation form of every byte, so no word can fall back to [UNK]
        vocab.setdefault(WP_PREFIX + c, len(vocab))
    spec["model"]["max_input_chars_per_word"] = 10 ** 7
    Path(path).write_text(json.dumps(spec))


def learn_superword(t, strs, n_merges):
    """Greedy pair merges on the stage-1 id stream t (algorithm of text/make_sbpe.py). Returns [(a, b)], strings of new ids appended to strs.
    Pairs where either side contains a newline are never merged, so no superword crosses a line."""
    V1 = len(strs)
    V = V1 + n_merges
    forbid = np.zeros(V, bool)
    forbid[:V1] = ["Ċ" in s for s in strs]
    t = np.asarray(t).astype(np.int32)
    cnt = np.zeros(V * V, np.int32)
    k = t[:-1].astype(np.int64) * V + t[1:]
    ok = ~(forbid[t[:-1]] | forbid[t[1:]])
    u, c = np.unique(k[ok], return_counts=True)
    cnt[u] = c
    heap = [-(int(ci) << 30 | int(ui)) for ui, ci in zip(u, c) if ci >= 2]
    heapq.heapify(heap)
    del u, c, k, ok
    merges, seen, nxt = [], set(strs), V1
    while nxt < V:
        while heap:
            e = -heapq.heappop(heap)
            key, ci = e & ((1 << 30) - 1), e >> 30
            if cnt[key] == ci:
                break
            if cnt[key] >= 2:
                heapq.heappush(heap, -(int(cnt[key]) << 30 | key))
        else:
            break  # no pair occurs twice any more
        a, b = divmod(key, V)
        s = strs[a] + strs[b]
        if s in seen:  # same string reachable through another merge path: not a new vocabulary entry
            cnt[key] = 0
            continue
        P = np.flatnonzero((t[:-1] == a) & (t[1:] == b))
        if a == b and len(P) > 1:  # non-overlapping, left to right
            new_run = np.concatenate([[True], np.diff(P) != 1])
            rs = np.maximum.accumulate(np.where(new_run, np.arange(len(P)), 0))
            P = P[(np.arange(len(P)) - rs) % 2 == 0]
        old_idx = np.unique(np.concatenate([P - 1, P, P + 1]))
        old_idx = old_idx[(old_idx >= 0) & (old_idx < len(t) - 1)]
        ok_k = old_idx[~(forbid[t[old_idx]] | forbid[t[old_idx + 1]])]
        ko = t[ok_k].astype(np.int64) * V + t[ok_k + 1]
        keep = np.ones(len(t), bool)
        keep[P + 1] = False
        t[P] = nxt
        t = t[keep]
        ni = np.arange(len(P))
        new_idx = np.unique(np.concatenate([P - ni - 1, P - ni]))
        new_idx = new_idx[(new_idx >= 0) & (new_idx < len(t) - 1)]
        new_idx = new_idx[~(forbid[t[new_idx]] | forbid[t[new_idx + 1]])]
        nk = t[new_idx].astype(np.int64) * V + t[new_idx + 1]
        uo, co = np.unique(ko, return_counts=True)
        cnt[uo] -= co.astype(np.int32)
        un, cn = np.unique(nk, return_counts=True)
        cnt[un] += cn.astype(np.int32)
        for kk in un:
            if cnt[kk] >= 2:
                heapq.heappush(heap, -(int(cnt[kk]) << 30 | int(kk)))
        merges.append((a, b))
        strs.append(s)
        seen.add(s)
        nxt += 1
        if nxt % 1024 == 0:
            log(f"  superword merges {nxt - V1}/{n_merges}, tokens left {len(t)}")
    return merges


def write_superword(stage1_path, strs1, merges, n2, path):
    spec = json.loads(Path(stage1_path).read_text())
    V1 = len(strs1)
    strs = list(strs1)
    for a, b in merges[:n2]:
        strs.append(strs[a] + strs[b])
    assert len(set(strs)) == len(strs)
    spec["model"]["vocab"] = {s: i for i, s in enumerate(strs)}
    m1 = spec["model"]["merges"]
    fmt = (lambda x, y: [x, y]) if isinstance(m1[0], list) else (lambda x, y: f"{x} {y}")
    spec["model"]["merges"] = m1 + [fmt(strs[a], strs[b]) for a, b in merges[:n2]]
    spec["pre_tokenizer"] = dict(spec["pre_tokenizer"], use_regex=False)
    Path(path).write_text(json.dumps(spec))


class ChunkTok(Tok):
    """Fixed or word-aligned chunks of M bytes. Vocabulary = the 256 single bytes plus the most frequent multi-byte chunks of
    the training text (at most chunk_vocab in all). A chunk outside the vocabulary is split in halves until its parts are in it."""

    def __init__(self, M, word_aligned, keys=None):
        self.M, self.wa = M, word_aligned
        self.keys = keys
        if keys is not None:
            self._index()

    def chunk_keys(self, b):
        M, n = self.M, len(b)
        if self.wa:
            s = word_starts(b)
            idx = np.arange(n)
            pos = idx - np.maximum.accumulate(np.where(s, idx, 0))
            c = np.flatnonzero(pos % M == 0)
        else:
            c, pos = np.arange(0, n, M), None
        key = np.zeros(len(c), np.int64)
        for j in range(M):
            i = np.minimum(c + j, n - 1)
            ok = (c + j < n) if pos is None else ((c + j < n) & (pos[i] == pos[c] + j))
            key = key * 257 + np.where(ok, b[i].astype(np.int64) + 1, 0)
        return key

    def key_bytes(self, key):
        d = [(int(key) // 257 ** (self.M - 1 - j)) % 257 for j in range(self.M)]
        return bytes(x - 1 for x in d if x > 0)

    def key_of(self, bs):
        return sum((x + 1) * 257 ** (self.M - 1 - j) for j, x in enumerate(bs))

    def fit(self, raw, vocab_cap):
        b = np.frombuffer(raw, np.uint8)
        u, c = np.unique(self.chunk_keys(b), return_counts=True)
        single = np.array([self.key_of([v]) for v in range(256)], np.int64)
        multi = ~np.isin(u, single)
        u, c = u[multi], c[multi]
        top = np.lexsort((u, -c))[: max(vocab_cap - 256, 0)]
        self.keys = np.concatenate([single, u[top]])
        self._index()

    def _index(self):
        self.table = [self.key_bytes(k) for k in self.keys]
        self.order = np.argsort(self.keys)
        self.sorted_keys = self.keys[self.order]
        self.id_of = {int(k): i for i, k in enumerate(self.keys)}

    def lookup(self, uniq):
        pos = np.minimum(np.searchsorted(self.sorted_keys, uniq), len(self.sorted_keys) - 1)
        hit = self.sorted_keys[pos] == uniq
        return hit, self.order[pos]

    def split(self, bs):
        k = self.key_of(bs)
        if k in self.id_of:
            return [self.id_of[k]]
        h = len(bs) // 2
        return self.split(bs[:h]) + self.split(bs[h:])

    def encode(self, raw):
        b = np.frombuffer(raw, np.uint8)
        uniq, inv = np.unique(self.chunk_keys(b), return_inverse=True)
        hit, first = self.lookup(uniq)
        first = first.astype(np.int64)
        n_ids = np.ones(len(uniq), np.int64)
        rest = {}
        for j in np.flatnonzero(~hit):
            ids = self.split(self.key_bytes(uniq[j]))
            first[j], n_ids[j] = ids[0], len(ids)
            for q in range(1, len(ids)):
                rest.setdefault(q, {})[j] = ids[q]
        c = n_ids[inv]
        starts = np.cumsum(c) - c
        out = np.empty(int(c.sum()), np.uint16)
        out[starts] = first[inv]
        for q, d in rest.items():
            r = np.zeros(len(uniq), np.int64)
            r[list(d)] = list(d.values())
            m = c > q
            out[starts[m] + q] = r[inv[m]]
        self.fallback_frac = float(np.mean(~hit[inv]))
        return out


# ------------------------------------------------------------------ checks and bookkeeping

def reconstruct(ids, table_L, table_off, flat):
    ids = ids.astype(np.int64)
    lens = table_L[ids]
    total = int(lens.sum())
    start = np.repeat(table_off[ids] - (np.cumsum(lens) - lens), lens) + np.arange(total)
    return flat[start]


class Checker:
    def __init__(self, table):
        self.L = np.array([len(t) for t in table], np.int64)
        self.off = np.cumsum(self.L) - self.L
        self.flat = np.frombuffer(b"".join(table), np.uint8)

    def lossless(self, ids, raw):
        if int(self.L[ids.astype(np.int64)].sum()) != len(raw):
            return False
        return bool(np.array_equal(reconstruct(ids, self.L, self.off, self.flat), np.frombuffer(raw, np.uint8)))


PARTS = ("tokenizer_fit", "ref_stats", "train", "valid")  # test is read only by encode_test()


def load_corpus(a):
    raw = {}
    for s in PARTS:
        p = Path(a.data) / f"{s}_bytes.npy"
        raw[s] = np.load(p, mmap_mode="r").tobytes()
    return raw


def corpus_info(a, raw):
    """The encoded inputs and their hashes (also what --verify compares against)."""
    fit = raw["tokenizer_fit"] if a.fit_bytes is None else prefix(raw["tokenizer_fit"], a.fit_bytes)
    data = {"train": prefix(raw["train"], a.enc_bytes), "ref": raw["ref_stats"], "valid": raw["valid"]}
    assert len(raw["train"]) >= a.enc_bytes, f"train has {len(raw['train'])} bytes, need {a.enc_bytes}"
    return fit, data


def corpus_hashes(fit, data):
    return {"tokenizer_fit": sha(fit), **{k: sha(v) for k, v in data.items()}}


def check_hf(t):
    """No normalization; every id's bytes agree with HF's own decode wherever a lone token is valid UTF-8."""
    if t.tok.normalizer is not None:
        raise Reject("normalizer present")
    bad = 0
    for i, b in enumerate(t.table):
        if t.wordpiece or not b:
            continue
        try:
            u = b.decode("utf-8")
        except UnicodeDecodeError:
            continue
        bad += t.tok.decode([i], skip_special_tokens=False) != u
    if bad:
        raise Reject(f"{bad} ids whose table bytes differ from HF decode")


def build(sp, a, state, fit, ps_fit, data, out, ref_np):
    """Train and encode one tokenizer; raises Reject on any failed check. Returns (info, meta rows)."""
    name, fam, t0 = sp["name"], sp["family"], time.time()
    info = dict(spec=sp, files={})
    if fam == "bytes":
        tok = ByteTok()
    elif fam == "tiktoken":
        tok = TikTok()
    elif fam in ("bpe", "unigram", "wordpiece", "bpe_dropout"):
        path = out / f"{name}.json"
        {"bpe": train_bpe, "bpe_dropout": train_bpe, "unigram": train_unigram, "wordpiece": train_wordpiece}[fam](ps_fit(), sp["V"], path)
        tok = HFTok(path, wordpiece=fam == "wordpiece")
        check_hf(tok)
        if len(tok.table) < 0.8 * sp["V"]:
            raise Reject(f"vocabulary {len(tok.table)} below 80% of requested {sp['V']}")
        info["files"]["tokenizer"] = path.name
    elif fam == "superword":
        if "stage1" not in state:
            p1 = out / f"sbpe_stage1_V{sp['V1']}.json"
            train_bpe(ps_fit(), sp["V1"], p1)
            t1 = HFTok(p1)
            strs1 = [t1.tok.id_to_token(i) for i in range(t1.tok.get_vocab_size())]
            sub = prefix(fit, a.sbpe_bytes)
            log("  superword: stage-1 V", len(strs1), "learning", a.n2_hi, "merges on", len(sub), "bytes")
            merges = learn_superword(t1.encode(sub), list(strs1), a.n2_hi)
            state["stage1"] = (p1, strs1, merges)
        p1, strs1, merges = state["stage1"]
        path = out / f"{name}.json"
        write_superword(p1, strs1, merges, sp["n2"], path)
        sp["n2_effective"] = min(sp["n2"], len(merges))
        tok = HFTok(path)
        check_hf(tok)
        info["files"]["tokenizer"] = path.name
    elif fam == "chunk":
        tok = ChunkTok(sp["L"], sp["word_aligned"])
        tok.fit(fit, a.chunk_vocab)
        np.save(out / f"{name}_chunks.npy", tok.keys)
        info["files"]["chunks"] = f"{name}_chunks.npy"
    else:
        raise ValueError(fam)
    V = len(tok.table)
    sp["V_actual"] = V
    chk = Checker(tok.table)
    adv = tok.encode(ADV)
    if not chk.lossless(adv, ADV):
        raise Reject("not lossless on the adversarial byte sample")
    info.update(V=V, bytes_per_token={}, n_tokens={}, sha256={}, lossless={"adversarial": True})
    rows, arrays = {}, {}
    splits = [("fit", fit), ("train", data["train"]), ("ref", data["ref"]), ("valid", data["valid"])]
    if fam == "bytes":  # the bytes ladder reaches 2^25, so the anchor's train array is longer
        splits[1] = ("train", data["train_long"])
    for s, raw in splits:
        ids = tok.encode(raw)
        assert ids.max() < V
        ok = chk.lossless(ids, raw)
        if isinstance(tok, HFTok) and not tok.wordpiece and s == "ref":
            ok = ok and tok.decode_ok(ids, raw)
        if not ok:
            raise Reject(f"not lossless on {s}")
        arrays[s] = ids
        r = dict(V=V, n_tokens=int(len(ids)), n_bytes=len(raw), bytes_per_token=len(raw) / len(ids))
        if s != "fit":
            info["bytes_per_token"][s], info["n_tokens"][s], info["sha256"][s], info["lossless"][s] = r["bytes_per_token"], r["n_tokens"], sha(ids), True
        if s == "fit":
            info["bytes_per_token_fit"] = r["bytes_per_token"]
        elif s != "test":
            rows[f"{name}_{s}"] = r
    # one spelling per word, measured on ref_stats
    from tok_stats import spelling_stats
    nw = min(a.spell_words, len(ref_np) - 4096)
    sps = spelling_stats(arrays["ref"], chk.L, ref_np, nw)
    info["spelling_freq_all"] = sps["all"]["spelling_freq"]
    info["spelling_freq_ge20"] = sps["ge20"]["spelling_freq"]
    if sp["stratum"] == "S1" and sps["all"]["spelling_freq"] > SPELL_MAX:
        raise Reject(f"S1 draw has spelling multiplicity {sps['all']['spelling_freq']:.3f} > {SPELL_MAX}")
    # commit files only after every check passed
    np.save(out / f"tok_{name}_L.npy", chk.L)
    info["L_sha256"] = sha(chk.L)
    for s, ids in arrays.items():
        if s != "fit":
            np.save(out / f"tok_{name}_{s}.npy", ids)
    if fam == "chunk":
        tok.encode(data["train"])
        info["fallback_chunk_frac_train"] = tok.fallback_frac
    if fam == "bpe_dropout":
        info["dropout_samples"] = {}
        for i in range(a.n_drop):
            ids = tok.encode(data["train"], dropout=sp["p"])
            if not chk.lossless(ids, data["train"]):
                raise Reject(f"dropout sample {i} is not lossless")
            f = f"tok_{name}_drop{sp['p']:g}_s{i}_train.npy"
            np.save(out / f, ids)
            info["dropout_samples"][f] = dict(n_tokens=int(len(ids)), bytes_per_token=len(data["train"]) / len(ids), sha256=sha(ids))
        info["train_flags"] = f"--bpe_dropout {sp['p']:g} --dropout_dir {out}"
    for k, f in info["files"].items():
        info["files"][k] = dict(name=f, sha256=sha((out / f).read_bytes()))
    info["seconds"] = time.time() - t0
    return info, rows


def artifacts_ok(out, m, with_test=False):
    """Every artifact the manifest lists exists and matches its recorded SHA256 (arrays, L tables, tokenizer files, chunk tables, dropout samples)."""
    for n, info in m["tokenizers"].items():
        for s_ in ("train", "ref", "valid"):
            f = out / f"tok_{n}_{s_}.npy"
            if not f.exists() or sha(np.load(f)) != info["sha256"][s_]:
                return False
        f = out / f"tok_{n}_L.npy"
        if not f.exists() or sha(np.load(f)) != info["L_sha256"]:
            return False
        for v in info["files"].values():
            if not (out / v["name"]).exists() or sha((out / v["name"]).read_bytes()) != v["sha256"]:
                return False
        for f, d in info.get("dropout_samples", {}).items():
            if not (out / f).exists() or sha(np.load(out / f)) != d["sha256"]:
                return False
    if with_test:
        tm = json.loads((out / "test_manifest.json").read_text())
        for n in m["tokenizers"]:
            f = out / f"tok_{n}_test.npy"
            if not f.exists() or sha(np.load(f)) != tm["tokenizers"][n]["sha256"]:
                return False
    return True


def load_tok(sp, out, info):
    fam = sp["family"]
    if fam == "bytes":
        return ByteTok()
    if fam == "tiktoken":
        return TikTok()
    if fam == "chunk":
        return ChunkTok(sp["L"], sp["word_aligned"], np.load(out / info["files"]["chunks"]["name"]))
    return HFTok(out / info["files"]["tokenizer"]["name"], wordpiece=fam == "wordpiece")


def encode_test(a, out):
    """After the forecast is made: encode and check test with the family. A failure invalidates the family (never a redraw)."""
    m = json.loads((out / "manifest.json").read_text())
    assert m.get("complete") and artifacts_ok(out, m), "family is not the frozen, verified one"
    test = np.load(Path(a.data) / "test_bytes.npy", mmap_mode="r").tobytes()
    tm = dict(test_sha256=sha(test), test_bytes=len(test), manifest_seed=m["seed"], tokenizers={})
    for n, info in m["tokenizers"].items():
        tok = load_tok(info["spec"], out, info)
        chk = Checker(tok.table)
        if not np.array_equal(chk.L, np.load(out / f"tok_{n}_L.npy")):
            raise SystemExit(f"FAMILY INVALID: {n} byte table differs from the frozen one")
        ids = tok.encode(test)
        if not chk.lossless(ids, test):
            raise SystemExit(f"FAMILY INVALID: {n} is not lossless on test (the family is NOT redrawn; the experiment is invalid)")
        np.save(out / f"tok_{n}_test.npy", ids)
        tm["tokenizers"][n] = dict(bytes_per_token=len(test) / len(ids), n_tokens=int(len(ids)), sha256=sha(ids), lossless=True)
        log("test", n, f"bytes/token {len(test) / len(ids):.3f} lossless OK")
    (out / "test_manifest.json").write_text(json.dumps(tm, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=parse_seed)
    ap.add_argument("--seed_file")
    ap.add_argument("--data", default=str(HERE.parent / "data" / "bfine"),
                    help="directory with {tokenizer_fit,ref_stats,train,valid,test}_bytes.npy")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fit_bytes", type=int, default=None, help="declared size of the tokenizer-fit text (default: the whole tokenizer_fit partition)")
    ap.add_argument("--enc_bytes", type=int, default=int(2 ** 23 * 1.05), help="bytes of train that are encoded for the learner (>= largest budget)")
    ap.add_argument("--enc_bytes_bytes", type=int, default=int(2 ** 25 * 1.02), help="same for the bytes anchor (its ladder reaches 2^25)")
    ap.add_argument("--sbpe_bytes", type=int, default=2 ** 23, help="bytes of stage-1 stream used to learn second-stage merges")
    ap.add_argument("--n_drop", type=int, default=8, help="BPE-dropout resamplings of the train prefix per drop* tokenizer")
    ap.add_argument("--spell_words", type=int, default=500_000, help="ref_stats bytes used for the one-spelling check")
    ap.add_argument("--chunk_vocab", type=int, default=16384)
    ap.add_argument("--v_lo", type=int, default=384)
    ap.add_argument("--uni_lo", type=int, default=512)
    ap.add_argument("--v_hi", type=int, default=16384)
    ap.add_argument("--sbpe_v1", type=int, default=8192)
    ap.add_argument("--n2_lo", type=int, default=512)
    ap.add_argument("--n2_hi", type=int, default=8192)
    ap.add_argument("--dry_run", action="store_true", help="print the first-attempt specs and stop")
    ap.add_argument("--verify", action="store_true",
                    help="exit 0 only if --out holds a complete family built from this seed, these arguments and these exact input arrays")
    ap.add_argument("--with_test", action="store_true", help="with --verify: also require the test arrays written by --encode_test")
    ap.add_argument("--encode_test", action="store_true", help="after the forecast is made: encode and check test with the family")
    ap.add_argument("--only", default=None, help="comma-separated names (debugging)")
    ap.add_argument("--names", default=None,
                    help="build only these templates as a complete family (e.g. bytes: tok_bytes_*.npy, tokenization_meta.json, manifest.json for "
                         "scripts/train_text.py --tok bytes); the bytes array then holds enc_bytes (enc_bytes_bytes is set to it)")
    a = ap.parse_args()
    templates = [t for t in TEMPLATES if a.names is None or t[0] in a.names.split(",")]
    if a.names is not None:
        assert len(templates) == len(a.names.split(",")), f"unknown name in --names {a.names}"
        a.enc_bytes_bytes = a.enc_bytes
    seed = a.seed if a.seed is not None else parse_seed(Path(a.seed_file).read_text())
    if a.dry_run:
        print(json.dumps(sample_specs(seed, a), indent=1))
        return
    out = Path(a.out)
    if a.encode_test:
        encode_test(a, out)
        return
    raw = load_corpus(a)
    fit, data = corpus_info(a, raw)
    if len(raw["train"]) < a.enc_bytes_bytes:
        raise SystemExit(f"train partition has {len(raw['train'])} bytes but the bytes ladder needs {a.enc_bytes_bytes} (2^25 plus slack): rerun prep_fineweb.py with a larger --train_bytes")
    data["train_long"] = prefix(raw["train"], a.enc_bytes_bytes)
    hashes = corpus_hashes(fit, {k: v for k, v in data.items() if k != "train_long"})
    hashes["train_long"] = sha(data["train_long"])
    keep = {k: v for k, v in vars(a).items() if k not in ("verify", "dry_run", "only", "out", "data", "seed_file", "with_test", "encode_test", "names")}
    if a.verify:
        try:
            m = json.loads((out / "manifest.json").read_text())
            ok = (m["seed"] == str(seed) and m["corpus_sha256"] == hashes and m["params"] == json.loads(json.dumps(keep))
                  and m.get("complete") is True and len(m["tokenizers"]) == len(templates)
                  and artifacts_ok(out, m, a.with_test))
        except Exception as e:  # noqa: BLE001
            print("verify failed:", e)
            ok = False
        print("family verified" if ok else "family stale or incomplete")
        sys.exit(0 if ok else 1)
    out.mkdir(parents=True, exist_ok=True)
    log("bytes", {k: len(v) for k, v in data.items()}, "tokenizer-fit bytes", len(fit))
    manifest = dict(seed=str(seed), params=keep, complete=False, corpus_sha256=hashes, corpus=hashes,
                    corpus_bytes={"tokenizer_fit": len(fit), **{k: len(v) for k, v in data.items()}},
                    rule=dict(max_attempts=MAX_ATTEMPTS, spell_max=SPELL_MAX, text="see comment above TEMPLATES in text/tok_family.py"),
                    specs={}, rejections=[], tokenizers={})
    meta, ps_cache, state = {}, [], {}
    ref_np = np.frombuffer(data["ref"], np.uint8)

    def ps_fit():
        if not ps_cache:
            ps_cache.append(pieces(fit))
        return ps_cache[0]

    for ti, (name, fam, stratum) in enumerate(TEMPLATES):
        if a.names is not None and name not in a.names.split(","):
            continue
        if a.only and name not in a.only.split(","):
            continue
        for attempt in range(MAX_ATTEMPTS):
            sp = draw_spec(seed, ti, attempt, a)
            log("==", name, {k: v for k, v in sp.items() if k not in ("name", "stratum")})
            try:
                info, rows = build(sp, a, state, fit, ps_fit, data, out, ref_np)
                break
            except Reject as e:
                log("   REJECTED:", e)
                manifest["rejections"].append(dict(name=name, attempt=attempt, spec=sp, reason=str(e)))
        else:
            raise SystemExit(f"{name}: {MAX_ATTEMPTS} draws rejected")
        meta.update(rows)
        manifest["tokenizers"][name] = info
        manifest["specs"][name] = sp
        bp = info["bytes_per_token"]
        log(f"   V={info['V']} bytes/token train {bp['train']:.3f} ref {bp['ref']:.3f}  spelling {info['spelling_freq_all']:.3f}  lossless OK ({info['seconds']:.0f}s)")
        (out / "tokenization_meta.json").write_text(json.dumps(meta, indent=1))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    manifest["complete"] = not a.only
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    log("done", len(manifest["tokenizers"]), "tokenizers;", len(manifest["rejections"]), "rejections")


if __name__ == "__main__":
    main()
