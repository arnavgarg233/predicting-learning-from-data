"""SmolLM2-135M reference curve on the open-sci-ref CommonCorpus held-out sample (1,000 targets, CPU).
Calls scripts/opensciref/refcurve.py `run` unchanged, with SmolLM2-135M instead of Qwen2.5-0.5B. Documents whose
SmolLM2 token byte lengths do not add up to the UTF-8 length (2 of 76 long documents) are dropped first, because
refcurve.py asserts that every document's token bytes add up.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "opensciref"))
import refcurve  # noqa: E402

RF = refcurve.RF
NAME = "HuggingFaceTB/SmolLM2-135M"
SAMPLES = os.path.join(ROOT, "results/curve_forecast/opensciref/samples")
OUT = os.path.join(ROOT, "results/opensciref_forecast/refcurve/commoncorpus")

if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(NAME)
    u2b = {v: k for k, v in RF.bytes_to_unicode().items()}
    tb = np.zeros(len(tok), np.int64)
    for t, i in tok.get_vocab().items():
        tb[i] = len(t) if all(c in u2b for c in t) else len(t.encode("utf-8"))
    docs = [json.loads(l)["text"] for l in open(os.path.join(SAMPLES, "commoncorpus", "heldout.jsonl"))]
    docs = [d for d in docs if len(d.encode("utf-8")) >= 4 * max(RF.NS)]
    keep = [d for d in docs if int(tb[tok(d, add_special_tokens=False)["input_ids"]].sum()) == len(d.encode("utf-8"))]
    print("long docs", len(docs), "kept", len(keep), flush=True)
    r = refcurve.run(NAME, keep, OUT, 1000, 50, 0, torch, torch.device("cpu"))
    json.dump(dict(long_docs=len(docs), kept=len(keep)), open(os.path.join(OUT, "doc_filter.json"), "w"))
    print([(x["n_tokens"], round(x["bits_per_byte"], 3)) for x in r["rows"]], flush=True)
