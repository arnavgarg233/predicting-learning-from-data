"""Data statistics of the 7 open-sci-ref corpora from their samples (no learner involved).

    python3 scripts/opensciref/data_stats.py SAMPLES TOKENIZER_JSON OUT.json

Per corpus:
  ngram       byte n-gram information curve h_ng(n), n = 0..24 context bytes (scripts/ngram_curve_ts.ngram_curve, the
              paper's estimator: interpolated absolute discounting), trained on train_sample.bin (2^23 bytes) and scored at
              2^17 random positions (seed 0) of the held-out text (documents joined with b"\\n").
              bigram = h_ng(1) (one byte of context); g = min over n (the paper's best n-gram cross-entropy at 2^23 bytes).
  bytes_per_token   UTF-8 bytes per GPT-NeoX-20B token over the held-out documents.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import ngram_curve_ts as NG  # noqa: E402

CORPORA = ["c4", "pile", "slimpajama", "fineweb_edu", "dclm", "hplt", "commoncorpus"]


def main():
    samples, tokjson, out = sys.argv[1:4]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokjson)
    res = {}
    for c in CORPORA:
        d = os.path.join(samples, c)
        docs = [json.loads(l)["text"] for l in open(os.path.join(d, "heldout.jsonl"))]
        va = np.frombuffer(b"\n".join(t.encode("utf-8") for t in docs), np.uint8).astype(np.uint64)
        tr = np.frombuffer(open(os.path.join(d, "train_sample.bin"), "rb").read(), np.uint8).astype(np.uint64)
        assert len(tr) == 2 ** 23
        nev = min(2 ** 17, len(va) - NG.K)
        pos = np.sort(np.random.default_rng(0).choice(np.arange(NG.K, len(va)), nev, replace=False))
        rows = NG.ngram_curve(tr, va, pos, log=lambda *a: None)
        enc = tok.encode_batch(docs, add_special_tokens=False)
        ntok = sum(len(e.ids) for e in enc)
        nbytes = sum(len(t.encode("utf-8")) for t in docs)
        h = {r["n"]: r["h_ng"] for r in rows}
        res[c] = dict(bigram=h[1], g=min(h.values()), g_order=int(min(h, key=h.get)), ngram_rows=rows,
                      bytes_per_token=nbytes / ntok, heldout_tokens=ntok, heldout_bytes=nbytes, n_eval_positions=nev)
        print(c, round(h[1], 4), round(res[c]["g"], 4), res[c]["g_order"], round(nbytes / ntok, 4), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
