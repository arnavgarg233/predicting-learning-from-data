"""prep_corpus.py with a lower-memory Wikipedia stream (the same documents in the same order; checked by the stored sha256).

    python -u prep_lowmem.py --corpus wiki_de ... (same arguments as scripts/prep_corpus.py)

Streaming wikimedia/wikipedia straight from the Hub reached about 10 GB of memory on Colab (12.9 GB in all). Here the same
parquet files of the same config are first downloaded to local disk and read with the generic parquet builder, the files
in the Hub's own (sorted) order, with the same .shuffle(seed=0, buffer_size=1000). Every corpus prepared this way is accepted only if its reference partition matches the stored sha256.
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
sys.path.insert(0, os.path.join(HERE, "scripts"))
import prep_corpus as P  # noqa: E402


def stream(path, name=None, **kw):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    if path != "wikimedia/wikipedia":
        return P._orig_stream(path, name, **kw)
    cache = os.environ.get("LOWMEM_CACHE", "/content/pq_cache")
    root = snapshot_download(path, repo_type="dataset", allow_patterns=[f"{name}/*.parquet"], local_dir=os.path.join(cache, name))
    files = sorted(glob.glob(os.path.join(root, name, "*.parquet")))
    ds = load_dataset("parquet", data_files={"train": files}, split="train", streaming=True)
    return ds.shuffle(seed=P.STREAM_SEED, buffer_size=1000)


P._orig_stream = P.stream
P.stream = stream

if __name__ == "__main__":
    P.main()
    sys.stdout.flush()
    os._exit(0)
