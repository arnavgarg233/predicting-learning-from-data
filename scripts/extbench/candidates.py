"""Per-pod candidate files from draw.json: results/extbench/candidates_p{1,2,3}.json ({list, pools}).
Each pod's drawn corpora keep their family's seeded order and are interleaved by family (sorted by (rank + 0.5) / n within the
family), so cutting the tail of a pod's list drops the last corpora of every family in step. Pools are the pod's spares."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
D = os.path.join(ROOT, "results/extbench")
draw = json.load(open(os.path.join(D, "draw.json")))
for p in ("1", "2", "3"):
    items = []
    for fam, f in draw["families"].items():
        names = f["pods"][p]["drawn"]
        items += [((i + 0.5) / len(names), fam, n, f["ref_model"]) for i, n in enumerate(names)]
    items.sort(key=lambda x: (x[0], x[1]))
    lst = [dict(name=n, family=fam, ref_model=ref, position=i + 1) for i, (_, fam, n, ref) in enumerate(items)]
    pools = {fam: [dict(name=n, family=fam, ref_model=f["ref_model"]) for n in f["pods"][p]["spares"]] for fam, f in draw["families"].items()}
    json.dump(dict(pod=int(p), list=lst, pools=pools, written=draw["written_pt"] + " (draw); before any corpus was prepared"),
              open(os.path.join(D, f"candidates_p{p}.json"), "w"), indent=1)
    print(p, [c["name"] for c in lst])
