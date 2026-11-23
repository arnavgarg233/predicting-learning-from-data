"""Cut the rows the benchmark-text scoring reads from DataDecide's released downstream results, for offline replay.

    python scripts/bench_mismatch_extract.py [CACHE]   # needs the network once (or the file already in CACHE)
      -> results/bench_mismatch_score/outcomes_extract.parquet

Source: allenai/DataDecide-eval-results, data/macro_avg-00000-of-00001.parquet at the revision pinned in
scripts/bench_mismatch_score.py (139 MB; its SHA-256 is checked against the value recorded in
results/bench_mismatch_score/score.json). Kept: model sizes 1B (the targets) and 4M, 20M, 60M, 150M, 300M (the trained
proxies), the ten OLMES tasks of the test, and for each (size, recipe, seed, task) only the final checkpoint, which is
the only step either scoring script uses. Columns and values are copied unchanged; the source SHA-256 is stored in the
file's metadata (source_sha256), and bench_mismatch_score.py / bench_mismatch_proxy.py read the extract when the full
file is not in their cache.
"""
import json
import os
import sys

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bench_mismatch_score import EXTRACT, FILE, REPO, REV, TASKS, sha  # noqa: E402
from bench_mismatch_proxy import SIZES  # noqa: E402


def main(cache):
    full = os.path.join(cache, FILE)
    if not os.path.exists(full):
        from huggingface_hub import hf_hub_download
        full = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REV, local_dir=cache)
    src = sha(full)
    want = json.load(open(os.path.join(ROOT, "results/bench_mismatch_score/score.json")))["outcome_source"]["sha256"]
    assert src == want, "not the pinned source file"
    pf, parts = pq.ParquetFile(full), []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg)
        t = t.filter(pc.and_(pc.is_in(t["params"], pa.array(["1B"] + SIZES)), pc.is_in(t["task"], pa.array(list(TASKS.values())))))
        if t.num_rows:
            parts.append(t)
    t = pa.concat_tables(parts)
    d = t.select(["params", "data", "seed", "task", "step"]).to_pandas()
    last = d.groupby(["params", "data", "seed", "task"]).step.transform("max")
    t = t.filter(pa.array((d.step == last).to_numpy()))
    meta = dict(t.schema.metadata or {})
    meta.update({b"source_sha256": src.encode(), b"source": f"{REPO}@{REV}/{FILE}".encode(),
                 b"selection": b"params in 1B,4M,20M,60M,150M,300M; the 10 test tasks; final step per params/data/seed/task"})
    pq.write_table(t.replace_schema_metadata(meta), EXTRACT, compression="zstd")
    print(f"{t.num_rows} rows -> {os.path.relpath(EXTRACT, ROOT)} ({os.path.getsize(EXTRACT):,} bytes)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "datadecide", "eval_results_cache"))
