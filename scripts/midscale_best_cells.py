"""Best cell of the tuned 100M-parameter level grid at 2^25 bytes, per encoding.

    python scripts/midscale_best_cells.py [--out PATH]   # -> results/midscale_vocab/best_cells.txt

For bytes and the 8,192-symbol BPE: among the seed-0, eight-pass runs at 2^25 bytes in results/midscale_vocab, the run
with the lowest validation bits per byte; reports its weight decay and learning rate and whether either lies at an
edge of the values tried. The second seed was then run at that cell (train_midscale.py with --seed 1).
"""
import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V = os.path.join(ROOT, "results/midscale_vocab")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(V, "best_cells.txt"))
    a = ap.parse_args()
    lines = []
    for enc in ("bytes", "bpe8k"):
        r = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(V, f"{enc}_b25_*.json")))]
        r = [x for x in r if x["seed"] == 0 and x["passes"] == 8]
        b = min(r, key=lambda x: x["valid_bpb"])
        wds, lrs = sorted({x["wd"] for x in r}), sorted({x["lr"] for x in r})
        edge = [n for n, v, g in (("wd", b["wd"], wds), ("lr", b["lr"], lrs)) if v in (g[0], g[-1])]
        lines.append(f"{enc} best wd {b['wd']} lr {b['lr']} edge {','.join(edge) or 'interior'}")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
