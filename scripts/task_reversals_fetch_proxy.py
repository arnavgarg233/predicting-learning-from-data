"""Stream DataDecide small-model final-checkpoint metrics (4M-300M) from the pinned parquet without writing it to disk.

Writes results/task_reversals/proxy_by_recipe_task.json: mean over 3 seeds of primary_metric and
correct_prob_per_char at the final step, per size, task and recipe.
"""
import json, os, sys, time
import pandas as pd
import pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from bench_mismatch_score import FILE, REPO, REV, TASKS  # noqa: E402

SIZES = ["4M", "20M", "60M", "150M", "300M"]
OUT = os.path.join(ROOT, "results", "task_reversals", "proxy_by_recipe_task.json")

fs = HfFileSystem()
path = f"datasets/{REPO}@{REV}/{FILE}"
t0 = time.time()
with fs.open(path, "rb", block_size=8 * 2**20, cache_type="readahead") as f:
    pf = pq.ParquetFile(f)
    print("row groups", pf.metadata.num_row_groups, "rows", pf.metadata.num_rows, pf.schema_arrow.names, flush=True)
    # One row group of 235k rows: stream it in record batches, keep only the five sizes and ten tasks,
    # and parse the two needed metrics per kept row so the large JSON column is never held whole.
    cols = ["params", "data", "seed", "task", "step", "metrics"]
    keep_sizes, keep_tasks = pa.array(SIZES), pa.array(list(TASKS.values()))
    rows = []
    for i, b in enumerate(pf.iter_batches(batch_size=20000, columns=cols)):
        b = b.filter(pc.and_(pc.is_in(b["params"], keep_sizes), pc.is_in(b["task"], keep_tasks)))
        for r in b.to_pylist():
            m = json.loads(r.pop("metrics"))
            r["primary_metric"], r["correct_prob_per_char"] = m["primary_metric"], m["correct_prob_per_char"]
            rows.append(r)
        print("batch", i, "kept", len(rows), round(time.time() - t0), "s", flush=True)
d = pd.DataFrame(rows)
d = d[d.step == d.groupby(["params", "data", "seed", "task"]).step.transform("max")].copy()
n = d.groupby(["params", "data", "task"]).size()
assert (n == 3).all(), n[n != 3]
g = d.groupby(["params", "task", "data"])[["primary_metric", "correct_prob_per_char"]].mean()
res = {"source": {"repo": REPO, "revision": REV, "file": FILE, "read": "HfFileSystem range reads, not saved"},
       "selection": "final step per params/data/seed/task, 3 seeds averaged",
       "values": {}}
for (s, task, data), row in g.iterrows():
    res["values"].setdefault(s, {}).setdefault(task, {})[data] = {k: float(v) for k, v in row.items()}
json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT, "rows", len(d), round(time.time() - t0), "s")
