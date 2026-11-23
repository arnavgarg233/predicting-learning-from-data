"""Held-out evaluation samples and training samples of the 7 open-sci-ref-0.01 corpora (huggingface.co only).

    python make_samples.py OUTDIR            # -> OUTDIR/<corpus>/{heldout.jsonl, cross.jsonl, train_sample.bin, meta.json}

Per corpus (seed 20261024):
  heldout.jsonl     whole documents, NFC-normalised, each cut to at most 32,768 UTF-8 bytes, until >= 1,500,000 bytes
  cross.jsonl       the leading documents of heldout.jsonl until >= 500,000 bytes (the 7x7 cross matrix)
  train_sample.bin  2^23 bytes: other documents of the same corpus (same treatment), joined with b"\\n"; the n-gram
                    statistics are trained on it. Disjoint from heldout by construction (different files or disjoint
                    documents of one shuffled pool).
Sources: held-out split where the corpus has one (C4 validation, Pile validation, SlimPajama-6B validation), otherwise
random corpus files (FineWeb-Edu, DCLM-baseline, HPLT 2.0 English, CommonCorpus). The training sample comes from the
training distribution: a disjoint C4 validation shard, disjoint Pile validation documents, the SlimPajama-6B test split
(iid with train), or a different random file of the corpus.
"""
import gzip
import hashlib
import io
import json
import os
import sys
import unicodedata

import numpy as np

SEED = 20261024
HELD, CROSS, TRAIN = 1_500_000, 500_000, 2 ** 23
DOC_CAP = 32768


def clean(t):
    t = unicodedata.normalize("NFC", t)
    b = t.encode("utf-8")
    if len(b) > DOC_CAP:
        t = unicodedata.normalize("NFC", b[:DOC_CAP].decode("utf-8", "ignore"))
    return t


def parquet_docs(fs, path, need, rng, col="text"):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(fs.open(path, "rb", block_size=8 << 20))
    order = rng.permutation(pf.num_row_groups)
    docs, nb, used = [], 0, []
    for g in order:
        tab = pf.read_row_group(int(g), columns=[col])
        for t in tab.column(col).to_pylist():
            if t and t.strip():
                docs.append(t)
                nb += min(len(t.encode("utf-8")), DOC_CAP)
        used.append(int(g))
        if nb >= need:
            break
    return docs, dict(file=path, row_groups=used, n_row_groups=pf.num_row_groups)


def jsonl_docs(fs, path, need, comp):
    f = fs.open(path, "rb", block_size=8 << 20)
    if comp == "gz":
        s = io.TextIOWrapper(gzip.GzipFile(fileobj=f), encoding="utf-8")
    else:
        import zstandard
        s = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8")
    docs, nb = [], 0
    for line in s:
        t = json.loads(line)["text"]
        if t and t.strip():
            docs.append(t)
            nb += min(len(t.encode("utf-8")), DOC_CAP)
        if nb >= need:
            break
    return docs, dict(file=path, first_docs=len(docs))


def take(docs, budget):
    out, nb = [], 0
    for t in docs:
        c = clean(t)
        if not c.strip():
            continue
        out.append(c)
        nb += len(c.encode("utf-8"))
        if nb >= budget:
            break
    assert nb >= budget, f"pool too small: {nb} < {budget}"
    return out


def build(name, fs, api):
    rng = np.random.default_rng([SEED, sum(map(ord, name))])
    D = "datasets/"
    if name == "c4":
        h, hm = jsonl_docs(fs, D + "allenai/c4/en/c4-validation.00000-of-00008.json.gz", 3 * HELD, "gz")
        t, tm = jsonl_docs(fs, D + "allenai/c4/en/c4-validation.00001-of-00008.json.gz", 2 * TRAIN, "gz")
    elif name == "pile":
        pool, pm = jsonl_docs(fs, D + "mit-han-lab/pile-val-backup/val.jsonl.zst", 3 * (HELD + TRAIN), "zst")
        pool = [pool[i] for i in rng.permutation(len(pool))]
        k = int(len(pool) * 3 * HELD / (3 * (HELD + TRAIN)))
        h, t, hm, tm = pool[:k], pool[k:], dict(pm, part="first share of shuffled pool"), dict(pm, part="rest of pool")
    elif name == "slimpajama":
        h, hm = parquet_docs(fs, D + "DKYoon/SlimPajama-6B/data/validation-00000-of-00001-4fb685c22a3f91ef.parquet", 3 * HELD, rng)
        t, tm = parquet_docs(fs, D + "DKYoon/SlimPajama-6B/data/test-00000-of-00001-9f769cf7ce219017.parquet", 2 * TRAIN, rng)
    else:
        repo, pre = dict(fineweb_edu=("HuggingFaceFW/fineweb-edu", "data/CC-MAIN-"),
                         dclm=("mlfoundations/dclm-baseline-1.0-parquet", "filtered/"),
                         hplt=("HPLT/HPLT2.0_cleaned", "eng_Latn"),
                         commoncorpus=("PleIAs/common_corpus", "common_corpus_"))[name]
        files = sorted(s.rfilename for s in api.dataset_info(repo).siblings
                       if s.rfilename.startswith(pre) and s.rfilename.endswith(".parquet"))
        fh, ft = rng.choice(len(files), 2, replace=False)
        h, hm = parquet_docs(fs, D + repo + "/" + files[fh], 3 * HELD, rng)
        t, tm = parquet_docs(fs, D + repo + "/" + files[ft], 2 * TRAIN, rng)
        hm["n_candidate_files"] = tm["n_candidate_files"] = len(files)
    h = [h[i] for i in rng.permutation(len(h))]
    t = [t[i] for i in rng.permutation(len(t))]
    held = take(h, HELD)
    cross = take(held, CROSS)
    tr = b"\n".join(c.encode("utf-8") for c in take(t, TRAIN + 4096))[:TRAIN]
    return held, cross, tr, dict(heldout_source=hm, train_source=tm)


def main():
    from huggingface_hub import HfApi, HfFileSystem
    out = sys.argv[1]
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else ["c4", "pile", "slimpajama", "fineweb_edu", "dclm", "hplt", "commoncorpus"]
    fs, api = HfFileSystem(), HfApi()
    for name in names:
        d = os.path.join(out, name)
        if os.path.exists(os.path.join(d, "meta.json")):
            continue
        os.makedirs(d, exist_ok=True)
        held, cross, tr, meta = build(name, fs, api)
        for fn, docs in (("heldout.jsonl", held), ("cross.jsonl", cross)):
            with open(os.path.join(d, fn), "w") as f:
                for c in docs:
                    f.write(json.dumps({"text": c}) + "\n")
        open(os.path.join(d, "train_sample.bin"), "wb").write(tr)
        meta.update(seed=SEED, doc_cap_bytes=DOC_CAP, heldout_docs=len(held),
                    heldout_bytes=sum(len(c.encode()) for c in held), cross_docs=len(cross),
                    cross_bytes=sum(len(c.encode()) for c in cross), train_bytes=len(tr),
                    sha256={fn: hashlib.sha256(open(os.path.join(d, fn), "rb").read()).hexdigest()
                            for fn in ("heldout.jsonl", "cross.jsonl", "train_sample.bin")})
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        print(name, meta["heldout_docs"], meta["heldout_bytes"], meta["cross_docs"], meta["train_bytes"],
              meta["heldout_source"].get("file"), meta["train_source"].get("file"), flush=True)


if __name__ == "__main__":
    main()
