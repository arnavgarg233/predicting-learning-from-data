"""Corpus sources and preparation for the single-pass rate study (results/rate_singlepass/draw.json).

    python -u scripts/rate_singlepass/prep_rs.py --corpus NAME --out DIR [--train_bytes ...] [--screen] [--encode_test]
    python -u scripts/rate_singlepass/prep_rs.py --pile_extract       # Pile shards -> data/rs/pile/<subset>/<shard>.jsonl

Sources (one generator per draw.json entry):
  fineweb2  HuggingFaceFW/fineweb-2, config = subset, streaming train split through datasets' seeded 1000-document shuffle
  pile      monology/pile-uncopyrighted train shards 00-05, rows of one meta.pile_set_name extracted by --pile_extract (at most
            PILE_CAP raw bytes per subset and shard), read in shard order through a seeded 1000-row shuffle buffer
  dolma     allenai/dolma v1.7 files of one source (draw.json order), streamed with curl | gzip -dc through the same buffer
Every source is then packed in stream order into ~8 KB documents (scripts/prep_corpus.pack: documents of 8 KB or more stay
alone), so every corpus has documents long enough for the reference pass. Splitting, deduplication and the train order are
scripts/prep_fineweb.write_splits unchanged; the bytes arrays for the learner come from text/tok_family.py, as for the earlier
corpora. --screen uses small quotas (train 2^23*1.05, ref_stats 2^21, the rest 2^16) and skips the tokenizer step.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import prep_corpus as P  # noqa: E402
from prep_fineweb import write_splits, parse_bytes  # noqa: E402

DRAW = ROOT / "results/rate_singlepass/draw.json"
PILE_DIR = ROOT / "data/rs/pile"
PILE_URL = os.environ.get("RS_PILE_URL", "https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/{:02d}.jsonl.zst")
PILE_SHARDS = range(int(os.environ.get("RS_PILE_SHARDS", "6")))
PILE_CAP = int(os.environ.get("RS_PILE_CAP", "60000000"))
PILE_DIR = Path(os.environ.get("RS_PILE_DIR", str(PILE_DIR)))
UA = "curl/8.4"


def entries():
    d = json.load(open(os.environ.get("RS_DRAW", DRAW)))["order"]
    return {c["name"]: c for c in d["languages"] + d["english"]}


def slug(s):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_").replace("__", "_")


def lines_of(cmd):
    p = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1 << 20)
    try:
        yield from p.stdout
    finally:
        p.kill()
        p.wait()


def docs_fw2(subset):
    for r in P.stream("HuggingFaceFW/fineweb-2", subset):
        yield f"fw2:{r['id']}", r["text"]


def docs_pile(subset):
    d = PILE_DIR / slug(subset)
    assert (PILE_DIR / "DONE").exists(), "Pile extraction has not finished"

    def rows():
        for f in sorted(d.glob("*.jsonl")):
            with open(f, "rb") as fh:
                for i, line in enumerate(fh):
                    yield f"pile:{f.stem}:{i}", json.loads(line)["text"]
    yield from P._buffer_shuffle(rows())


def docs_dolma(files):
    def rows():
        for u in files:
            for i, line in enumerate(lines_of(f"curl -sL --retry 5 -A {UA} '{u}' | gzip -dc")):
                r = json.loads(line)
                t = r.get("text") or ""
                if t.strip():
                    yield f"dolma:{r.get('id') or u.split('/')[-1] + ':' + str(i)}", t
    yield from P._buffer_shuffle(rows())


def docs(c):
    g = {"fineweb2": lambda: docs_fw2(c["subset"]), "pile": lambda: docs_pile(c["subset"]),
         "dolma": lambda: docs_dolma(c["files"])}[c["source"]]()
    return P.pack(g, c["name"])


def pile_shard(i, names):
    """Rows of each Pile subset in `names` from shard i, at most PILE_CAP raw bytes per subset (stops reading when all are full)."""
    done = PILE_DIR / f"shard{i:02d}.DONE"
    if done.exists():
        return
    want = {n.encode(): slug(n) for n in names}
    for s in want.values():
        (PILE_DIR / s).mkdir(parents=True, exist_ok=True)
    fh = {s: open(PILE_DIR / s / f"{i:02d}.jsonl", "wb") for s in want.values()}
    nb = {s: 0 for s in want.values()}
    key = b'"pile_set_name": "'
    for line in lines_of(f"curl -sL --retry 5 -A {UA} '{PILE_URL.format(i)}' | zstd -dc"):
        j = line.rfind(key)
        if j < 0:
            continue
        s = want.get(line[j + len(key): line.index(b'"', j + len(key))])
        if s is None or nb[s] >= PILE_CAP:
            continue
        fh[s].write(line)
        nb[s] += len(line)
        if all(v >= PILE_CAP for v in nb.values()):
            break
    for f in fh.values():
        f.close()
    done.write_text(json.dumps(nb))
    print(f"shard {i:02d}", {s: round(v / 1e6, 1) for s, v in nb.items()}, flush=True)


def pile_extract():
    import concurrent.futures as cf
    names = [c["subset"] for c in entries().values() if c["source"] == "pile"]
    PILE_DIR.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(len(PILE_SHARDS)) as ex:
        list(ex.map(lambda i: pile_shard(i, names), PILE_SHARDS))
    tot = {slug(n): sum(f.stat().st_size for f in (PILE_DIR / slug(n)).glob("*.jsonl")) for n in names}
    (PILE_DIR / "DONE").write_text(json.dumps(tot))
    print("pile extraction done", {k: round(v / 1e6) for k, v in tot.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--out")
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--train_bytes", type=parse_bytes, default=int(2 ** 27 * 1.03))
    ap.add_argument("--enc_bytes", type=parse_bytes, default=int(2 ** 27 * 1.02))
    ap.add_argument("--encode_test", action="store_true")
    ap.add_argument("--pile_extract", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every quota (local smoke tests only)")
    a = ap.parse_args()
    if a.pile_extract:
        return pile_extract()
    c = entries()[a.corpus]
    out = Path(a.out)
    if a.screen:
        quota = dict(tokenizer_fit=2 ** 16, ref_stats=2 ** 21, valid=2 ** 16, test=2 ** 16, train=int(2 ** 23 * 1.05))
    else:
        quota = dict(tokenizer_fit=2 ** 20, ref_stats=2 ** 23, valid=2 ** 22, test=2 ** 22, train=a.train_bytes)
    quota = {k: int(v * a.scale) for k, v in quota.items()}
    a.enc_bytes = int(a.enc_bytes * a.scale)
    write_splits(docs(c), out, quota)
    import numpy as np
    for p in ("train", "valid", "test", "ref_stats"):
        got = len(np.load(out / f"{p}_bytes.npy", mmap_mode="r"))
        if got < 0.98 * quota[p]:
            sys.exit(f"{a.corpus}: partition {p} has {got} of {quota[p]} bytes (stream exhausted)")
    for f in out.glob("*_gpt2.npy"):      # GPT-2 token arrays are not used here
        f.unlink()
    if not a.screen:
        tok = [sys.executable, str(ROOT / "text/tok_family.py"), "--data", str(out), "--out", str(out / "tok"), "--names", "bytes",
               "--seed", "0", "--enc_bytes", str(a.enc_bytes)]
        subprocess.run(tok, check=True)
        subprocess.run(tok + ["--verify"], check=True)
        if a.encode_test:
            subprocess.run(tok + ["--encode_test"], check=True)
    print("prepared", a.corpus, out, flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
