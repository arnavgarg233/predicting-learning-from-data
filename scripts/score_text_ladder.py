"""Score a tokenizer ladder: bits/byte table and exponents of (bpb - floor) vs training bytes.
Usage: python scripts/score_text_ladder.py v1 v2_emb64"""
import json, glob, sys, os
import numpy as np

TOKS = ["bytes", "bpe512", "bpe2048", "bpe8192"]
FIT = [22, 24, 26]
FLOORS = [0.12, 0.25, 0.38, 0.42]


def load(tag):
    R = {}
    for f in glob.glob(f"results/text_runs/{tag}/*.json"):
        d = json.load(open(f))
        k = int(os.path.basename(f).split("_b")[1].split("_")[0])
        R.setdefault(d["tok"], {}).setdefault(k, []).append(d)
    return R


def slope(bpb, floor):
    return -np.polyfit(np.array(FIT) * np.log(2), np.log(np.asarray(bpb) - floor), 1)[0]


rng = np.random.default_rng(0)
out = {}
for tag in sys.argv[1:]:
    R = load(tag)
    print(f"== {tag}")
    print("tok      " + "  ".join(f"2^{k}" for k in [20, 22, 24, 26]) + "   steps@2^26  decays@2^26")
    for t in TOKS:
        row = [np.mean([d["bits_per_byte"] for d in R[t][k]]) for k in [20, 22, 24, 26]]
        st = [d["steps"] for d in R[t][26]]; dc = [d.get("decays") for d in R[t][26]]
        print(f"{t:8s} " + " ".join(f"{v:.3f}" for v in row) + f"   {st}  {dc}")
    out[tag] = {}
    for fl in FLOORS:
        line = []
        for t in TOKS:
            per = [[d["bits_per_byte"] for d in R[t][k]] for k in FIT]
            a = slope([np.mean(p) for p in per], fl)
            boots = [slope([np.mean(rng.choice(p, len(p))) for p in per], fl) for _ in range(2000)]
            se = float(np.std(boots))
            out[tag].setdefault(str(fl), {})[t] = (a, se)
            line.append(f"{t} {a:.3f}+-{se:.3f}")
        print(f"floor {fl:.2f}: " + "  ".join(line))
json.dump(out, open("results/text_runs/exponents.json", "w"), indent=1)
