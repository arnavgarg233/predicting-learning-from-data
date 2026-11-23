"""Seeded samples of public pretraining corpora, in the same form as the DataDecide recipe samples.

    python scripts/data_choice_sample.py SLUG [SLUG ...]     -> data/data_choice/samples/<slug>.npy + <slug>_provenance.json
    python scripts/data_choice_sample.py list

Each sample copies the design of scripts/datadecide_stats.py `sample` for the DataDecide recipes: N_RUNS = 128 runs of
RUN_TOKENS = 131072 tokens, each from a file drawn with probability proportional to its size, starting at a random point
of a random document, keeping only the whole documents inside the window (each followed by EOS 50279), in the DataDecide
tokenizer (allenai/gpt-neox-olmo-dolma-v1_5). Sample size therefore matches the recipe samples (about 16M tokens, 65-75 MB
of text), which matters for the size-dependent statistics (unigram entropy, the bigram cross-entropy behind the mismatch).

Random access: parquet corpora read one random row group per run (row groups hold 1000 rows) and start at a random row
in it; compressed JSON-lines corpora (.json.gz, .jsonl.zst), which cannot be entered in the middle, are streamed from the
start of each drawn file and each run starts after a uniform random gap of 0-1999 documents. Every source is pinned to a
dataset revision (HF commit sha), and the seed is derived from the slug, so a rerun reproduces the same tokens.
"""
import gzip, hashlib, io, json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "data_choice" / "samples"
TOK = ROOT / "data" / "datadecide" / "tokenizer.json"
EOS, V = 50279, 50280
RUN_TOKENS, N_RUNS, GAP = 131072, 128, 2000
UA = {"User-Agent": "curl/8.7.1"}  # olmo-data.org refuses the default Python agent
MAX_CHARS = RUN_TOKENS * 12  # longer documents cannot fit in one window, whatever their tokenization

CORPORA = {  # slug: (HF dataset id, file prefix, file suffix, format, text field, note)
    "fineweb": ("HuggingFaceFW/fineweb", "data/", ".parquet", "parquet", "text", "all CC dumps"),
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "data/", ".parquet", "parquet", "text", "all CC dumps (int_score >= 3)"),
    "c4": ("allenai/c4", "en/c4-train", ".json.gz", "jsonl.gz", "text", "c4/en train"),
    "dclm_baseline": ("mlfoundations/dclm-baseline-1.0", "global-shard", ".jsonl.zst", "jsonl.zst", "text", "DCLM-baseline 1.0"),
    "refinedweb": ("tiiuae/falcon-refinedweb", "data/", ".parquet", "parquet", "content", "RefinedWeb public extract"),
    "pile": ("monology/pile-uncopyrighted", "train/", ".jsonl.zst", "jsonl.zst", "text", "the Pile minus copyrighted subsets"),
    "dolma": ("allenai/dolma", "urls/v1_7.txt", "", "url_list_jsonl.gz", "text", "Dolma v1.7 (files on olmo-data.org, list on HF)"),
    "slimpajama": ("cerebras-archive/SlimPajama-627B", "data/train", ".parquet", "parquet", "text", "SlimPajama-627B train"),
    "redpajama_1t": ("liang2kl/RedPajama-Data-1T-Sample-Backup", "data/train", ".parquet", "parquet", "text",
                     "RedPajama-Data-1T-Sample (1B-token random sample of RedPajama-1T; official repo no longer served)"),
}


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOK))


def files_of(slug):
    """(revision sha, [(path or url, size)]) for the corpus, sorted by path."""
    from huggingface_hub import HfApi, hf_hub_download
    rid, pre, suf, fmt, _, _ = CORPORA[slug]
    info = HfApi().dataset_info(rid, files_metadata=True)
    if fmt == "url_list_jsonl.gz":
        urls = open(hf_hub_download(rid, pre, repo_type="dataset", revision=info.sha)).read().split()

        def head(u):
            for k in range(5):
                try:
                    with urllib.request.urlopen(urllib.request.Request(u, method="HEAD", headers=UA), timeout=60) as r:
                        return int(r.headers["Content-Length"])
                except Exception:
                    time.sleep(2 ** k)
            raise RuntimeError(u)
        with ThreadPoolExecutor(16) as ex:
            sizes = list(ex.map(head, urls))
        return info.sha, sorted(zip(urls, sizes))
    return info.sha, sorted((s.rfilename, s.size) for s in info.siblings
                            if s.rfilename.startswith(pre) and s.rfilename.endswith(suf))


def lengths(tok, docs):
    """Token counts (documents too long to ever fit get RUN_TOKENS + 1 without tokenizing) and the token lists."""
    short = [i for i, d in enumerate(docs) if len(d) <= MAX_CHARS]
    enc = tok.encode_batch([docs[i] for i in short], add_special_tokens=False)
    ids = [None] * len(docs)
    for i, e in zip(short, enc):
        ids[i] = e.ids
    return [len(x) if x is not None else RUN_TOKENS + 1 for x in ids], ids


def window(tok, docs, u01):
    """One run: the window of RUN_TOKENS tokens starts at a uniform point of docs[0] (+ its EOS); keep the whole
    documents after it that fit (as datadecide_stats.sample keeps r[cut[0]+1:cut[-1]+1]). Returns token ids or None
    if more documents are needed."""
    n, ids = lengths(tok, docs)
    used = (n[0] + 1) - int(u01 * (n[0] + 1))
    keep = []
    for L, x in zip(n[1:], ids[1:]):
        if used + L + 1 > RUN_TOKENS:
            return keep
        keep.append(x)
        used += L + 1
    return None


def run_from_iter(tok, it, u01, batch=64):
    docs = []
    while True:
        more = [d for _, d in zip(range(batch), it)]
        docs += more
        r = window(tok, docs, u01)
        if r is not None or not more:
            return r or []
        batch *= 2


def hf_url(rid, sha, path):
    return f"https://huggingface.co/datasets/{rid}/resolve/{sha}/{path}"


def stream_lines(url, fmt):
    for k in range(6):
        try:
            resp = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300)
            break
        except Exception as ex:
            print("retry", url, ex, flush=True)
            time.sleep(2 ** k)
    if fmt.endswith("gz"):
        f = gzip.GzipFile(fileobj=resp)
    else:
        import zstandard
        f = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(resp)
    for line in io.TextIOWrapper(f, encoding="utf-8"):
        yield line


def sample(slug):
    out = OUT / f"{slug}.npy"
    if out.exists():
        return
    OUT.mkdir(parents=True, exist_ok=True)
    rid, _, _, fmt, field, note = CORPORA[slug]
    sha, files = files_of(slug)
    size = np.array([s for _, s in files], float)
    rng = np.random.default_rng(int(hashlib.sha256(slug.encode()).hexdigest()[:8], 16))
    pick = rng.choice(len(files), N_RUNS, p=size / size.sum())
    u = rng.random(N_RUNS)
    tok = tokenizer()
    t0 = time.time()
    if fmt == "parquet":
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem
        fs = HfFileSystem()
        seeds = rng.integers(0, 2 ** 31, N_RUNS)

        def one(j):
            path = files[pick[j]][0]
            r = np.random.default_rng(int(seeds[j]))
            for k in range(6):
                try:
                    with fs.open(f"datasets/{rid}@{sha}/{path}", "rb", block_size=2 ** 22) as f:
                        pf = pq.ParquetFile(f)
                        g = int(r.integers(pf.num_row_groups))
                        col = pf.read_row_group(g, columns=[field]).column(0).to_pylist()
                        row = int(r.integers(len(col)))
                        docs = [d for d in col[row:] if d]
                        gg = g + 1
                        while gg < pf.num_row_groups and sum(map(len, docs)) < 2 * RUN_TOKENS * 6:
                            docs += [d for d in pf.read_row_group(gg, columns=[field]).column(0).to_pylist() if d]
                            gg += 1
                    break
                except Exception as ex:
                    print("retry", path, ex, flush=True)
                    time.sleep(2 ** k)
            ids = run_from_iter(tok, iter(docs), u[j])
            return dict(file=path, row_group=g, row=row), ids
        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one, range(N_RUNS)))
    else:
        by_file = {}
        for j in range(N_RUNS):
            by_file.setdefault(int(pick[j]), []).append(j)
        gaps = rng.integers(0, GAP, N_RUNS)

        def per_file(i):
            path = files[i][0]
            url = path if path.startswith("http") else hf_url(rid, sha, path)
            it = (json.loads(line)[field] for line in stream_lines(url, fmt))
            outs = []
            for j in by_file[i]:
                for _ in range(int(gaps[j])):
                    next(it, None)
                ids = run_from_iter(tok, it, u[j])  # consumes the run's documents (and a few beyond the window)
                outs.append((j, dict(file=path, doc_skip_before=int(gaps[j])), ids))
            return outs
        with ThreadPoolExecutor(8) as ex:
            flat = [x for o in ex.map(per_file, list(by_file)) for x in o]
        res = [(m, ids) for _, m, ids in sorted(flat, key=lambda t: t[0])]
    parts, prov = [], []
    for meta, ids in res:
        meta.update(n_docs=len(ids), n_tokens=int(sum(len(x) + 1 for x in ids)))
        prov.append(meta)
        for x in ids:
            parts.append(np.asarray(x, np.int64))
            parts.append(np.array([EOS]))
    a = np.concatenate(parts).astype("<u2")
    assert int(np.concatenate(parts).max()) < V
    tmp = out.with_suffix(".tmp")
    a.tofile(tmp)
    os.replace(tmp, out)
    meta = dict(slug=slug, dataset=rid, revision=sha, format=fmt, text_field=field, note=note, n_files=len(files),
                total_bytes_listed=int(size.sum()), n_runs=N_RUNS, run_tokens=RUN_TOKENS, n_tokens=int(len(a)),
                n_docs=int((a == EOS).sum()), sha256_npy=hashlib.sha256(a.tobytes()).hexdigest(),
                seconds=time.time() - t0, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runs=prov)
    (OUT / f"{slug}_provenance.json").write_text(json.dumps(meta, indent=1))
    print(slug, len(a), "tokens", meta["n_docs"], "docs", f"{meta['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "list":
        print("\n".join(CORPORA))
    else:
        for s in sys.argv[1:]:
            sample(s)
