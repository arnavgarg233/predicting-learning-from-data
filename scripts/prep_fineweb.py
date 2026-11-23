"""Stream a FineWeb-Edu sample and write byte and GPT-2 BPE arrays for scripts/train_midscale.py. Run on the pod.

Documents are assigned to partitions by a seeded hash of the stable document id (not by stream order):
tokenizer_fit, ref_stats, valid, test, train. Each partition fills to its declared byte quota; documents that arrive
after their partition is full are skipped. Exact duplicates (hash of the document bytes) and near duplicates (a later
document sharing >= 50% of its sampled 5-word shingles with one earlier kept document) are dropped across all
partitions, so no kept document has a duplicate in another partition. Counts go to manifest.json.
Training documents are written in a seeded shuffled order, independent of tokenizer; a byte budget is a prefix of it.
Every document is followed by "\\n\\n"; the byte and token arrays hold the same text, so bits per byte is exact.

Files in --out, for each partition P: P_bytes.npy (uint8), P_gpt2.npy (uint16), P_docstarts.npy (int64 byte offset of
each document's first byte, used by text/refcurve/refcurve_fineweb.py to keep contexts inside one document),
P_docids.txt (document ids in array order). meta.json holds bytes, tokens and bytes_per_token per partition;
manifest.json holds the quotas, sizes, id-list SHA256s and the duplicate audit.
"""
import argparse
import hashlib
import json
import os
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import tiktoken

SEP = b"\n\n"
PARTS = ["tokenizer_fit", "ref_stats", "valid", "test", "train"]
SALT = "k125-midscale-v1"  # fixed partition salt, an arbitrary label; keep it to reproduce the published partitions
M1, M2, M3, M4, M5 = (np.uint64(m) for m in (0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5, 0xFF51AFD7ED558CCD))


def parse_bytes(s):
    return int(2 ** int(s[2:])) if s.startswith("2^") else int(s)


def hashes(doc_id):
    d = hashlib.sha256(f"{SALT}:{doc_id}".encode()).digest()
    return int.from_bytes(d[:8], "big") / 2 ** 64, int.from_bytes(d[8:16], "big")  # partition draw, shuffle key


def shingles(raw):
    """Hashes of 5-word shingles, keeping about 1 in 8 (by hash value)."""
    w = raw.split()
    if len(w) < 5:
        return set()
    c = np.fromiter((zlib.crc32(x) for x in w), dtype=np.uint64, count=len(w))
    n = len(c) - 4
    h = c[:n] * M1 ^ c[1:n + 1] * M2 ^ c[2:n + 2] * M3 ^ c[3:n + 3] * M4 ^ c[4:n + 4] * M5
    h ^= h >> np.uint64(29)
    return set(h[(h >> np.uint64(7)) % np.uint64(8) == 0].tolist())


def write_splits(docs, out, quota, log=print):
    """docs: iterator of (doc_id, text). quota: bytes per partition (dict over PARTS). Writes arrays, meta.json and
    manifest.json to out and returns the manifest."""
    enc = tiktoken.get_encoding("gpt2")
    out.mkdir(parents=True, exist_ok=True)
    edges = np.cumsum([quota[p] for p in PARTS]) / sum(quota.values())
    kept = {p: [] for p in PARTS}  # (shuffle key, doc id, raw bytes)
    nb = {p: 0 for p in PARTS}
    seen_exact, index, docpart, shingle_sets = {}, {}, [], []
    audit = Counter()
    n_stream = 0
    for doc_id, text in docs:
        n_stream += 1
        u, key = hashes(doc_id)
        p = PARTS[min(int(np.searchsorted(edges, u, side="right")), len(PARTS) - 1)]
        if nb[p] >= quota[p]:
            audit["skipped_partition_full"] += 1
            continue
        raw = text.encode("utf-8") + SEP
        ex = hashlib.blake2b(raw, digest_size=16).digest()
        if ex in seen_exact:
            audit[f"exact_dropped:{p}|kept_in:{docpart[seen_exact[ex]]}"] += 1
            continue
        sh = shingles(raw)
        if len(sh) >= 10:  # inverted index shingle -> ALL earlier kept docs; overlap counted per candidate doc
            m = Counter(i for s in sh for i in index.get(s, ())).most_common(1)
            if m and m[0][1] >= 0.5 * len(sh):
                audit[f"near_dropped:{p}|kept_in:{docpart[m[0][0]]}"] += 1
                continue
        seen_exact[ex] = len(docpart)
        for s in sh:
            index.setdefault(s, []).append(len(docpart))
        shingle_sets.append(sh)
        docpart.append(p)
        kept[p].append((key, doc_id, raw))
        nb[p] += len(raw)
        if n_stream % 20000 == 0:
            log(f"{n_stream} docs, bytes {nb}")
        if all(nb[q] >= quota[q] for q in PARTS):
            break
    # cross-partition near-duplicate audit over the final kept set: for each kept doc, its largest shingle overlap with a doc of ANOTHER partition
    cross = Counter()
    for i, sh in enumerate(shingle_sets):
        if len(sh) < 10:
            continue
        c = Counter(j for s in sh for j in index[s] if docpart[j] != docpart[i])
        f = c.most_common(1)[0][1] / len(sh) if c else 0.0
        cross["docs_checked"] += 1
        cross["overlap_ge_0.5"] += f >= 0.5
        cross["overlap_ge_0.3"] += f >= 0.3
    meta, man = {}, {}
    exact_sets = {}
    for p in PARTS:
        docs_p = sorted(kept[p])  # seeded shuffled order (the hash key does not depend on any tokenizer)
        raws = [r for _, _, r in docs_p]
        ids = [i for _, i, _ in docs_p]
        toks = enc.encode_ordinary_batch([r.decode("utf-8") for r in raws], num_threads=8)
        xb = np.frombuffer(b"".join(raws), dtype=np.uint8)
        xt = np.concatenate([np.array(t, dtype=np.uint16) for t in toks])
        np.save(out / f"{p}_bytes.npy", xb)
        np.save(out / f"{p}_gpt2.npy", xt)
        np.save(out / f"{p}_docstarts.npy", np.concatenate([[0], np.cumsum([len(r) for r in raws])[:-1]]).astype(np.int64))
        (out / f"{p}_docids.txt").write_text("\n".join(map(str, ids)) + "\n")
        exact_sets[p] = {hashlib.blake2b(r, digest_size=16).digest() for r in raws}
        meta[p] = dict(bytes=len(xb), tokens=len(xt), bytes_per_token=len(xb) / len(xt))
        # held-out (test) compression statistics stay out of meta.json and manifest.json: they go to test_meta.json (read only after scoring)
        man[p] = dict(quota_bytes=quota[p], n_docs=len(ids), docids_sha256=hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest(),
                      **({} if p == "test" else meta[p]))
    overlap = sum(len(exact_sets[a] & exact_sets[b]) for i, a in enumerate(PARTS) for b in PARTS[i + 1:])
    manifest = dict(salt=SALT, partition="seeded hash of document id", stream_docs_read=n_stream, splits=man,
                    dedup=dict(rule="exact: blake2b of document bytes; near: >=50% of sampled 5-word shingles (1/8 by hash, "
                                    "docs with >=10 sampled shingles) shared with one earlier kept doc; later doc dropped",
                               counts=dict(audit), cross_partition_exact_overlap=overlap, cross_partition_near_overlap=dict(cross)))
    (out / "meta.json").write_text(json.dumps({p: v for p, v in meta.items() if p != "test"}, indent=1))
    (out / "test_meta.json").write_text(json.dumps({"test": meta["test"]}, indent=1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(json.dumps({p: v for p, v in meta.items() if p != "test"}))
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "midscale"))
    ap.add_argument("--train_bytes", type=parse_bytes, default="2^29")
    ap.add_argument("--valid_bytes", type=parse_bytes, default="2^22")
    ap.add_argument("--test_bytes", type=parse_bytes, default="2^22")
    ap.add_argument("--tokenizer_fit_bytes", type=parse_bytes, default="2^24")
    ap.add_argument("--ref_stats_bytes", type=parse_bytes, default="2^24")
    ap.add_argument("--name", default="sample-10BT", help="HuggingFaceFW/fineweb-edu config")
    args = ap.parse_args()
    quota = dict(tokenizer_fit=args.tokenizer_fit_bytes, ref_stats=args.ref_stats_bytes, valid=args.valid_bytes,
                 test=args.test_bytes, train=int(args.train_bytes * 1.01) + 4096)
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name=args.name, split="train", streaming=True)
    write_splits(((r["id"], r["text"]) for r in ds), Path(args.out), quota)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)  # the streaming dataset can hang at interpreter exit (seen on Colab 2026-10-13); every file is written by now
