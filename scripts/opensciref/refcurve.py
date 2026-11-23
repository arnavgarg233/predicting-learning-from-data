"""Reference information curve h(n) of each corpus's held-out sample (text/refcurve/refcurve_fineweb.py protocol).

    python refcurve.py SAMPLES OUT [--nt 10000] [--device cuda]
      -> OUT/<corpus>/<model short>.json (+ _raw.npz)

Reference model: SmolLM2-135M for the English corpora, Qwen2.5-0.5B for CommonCorpus. Documents of heldout.jsonl with at
least 2,048 UTF-8 bytes; targets drawn uniformly from positions with >= 512 in-document predecessor tokens, identical for
every context n (paired); each n gets its own forward pass on exactly the n preceding tokens. Weights in float32.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "text", "refcurve"))
import refcurve_fineweb as RF  # noqa: E402

REF = {"commoncorpus": "Qwen/Qwen2.5-0.5B"}
DEFAULT_REF = "HuggingFaceTB/SmolLM2-135M"


def run(name, docs, out, nt, chunk, seed, torch, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    short = name.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
    u2b = {v: k for k, v in RF.bytes_to_unicode().items()}
    V = max(len(tok), m.get_output_embeddings().weight.shape[0])
    tb = np.zeros(V, np.int32)
    for t, i in tok.get_vocab().items():
        tb[i] = len(t) if all(c in u2b for c in t) else len(t.encode("utf-8"))
    ids = [tok(d, add_special_tokens=False)["input_ids"] for d in docs]
    for d, x in zip(docs, ids):
        assert int(tb[x].sum()) == len(d.encode("utf-8")), f"{name}: token byte lengths do not add up"
    ns = RF.NS
    nmax = max(ns)
    ok_pos = np.array([max(len(x) - nmax, 0) for x in ids])
    rng = np.random.default_rng(seed)
    nt = min(nt, int(ok_pos.sum()))
    flat = np.sort(rng.choice(int(ok_pos.sum()), nt, replace=False))
    cs = np.cumsum(ok_pos)
    doc = np.searchsorted(cs, flat, side="right")
    pos = flat - (cs[doc] - ok_pos[doc]) + nmax
    tgt = np.array([ids[d][p] for d, p in zip(doc, pos)])
    ctxb = np.array([[tb[ids[d][p - n:p]].sum() for n in ns] for d, p in zip(doc, pos)])
    loss = np.zeros((nt, len(ns)), np.float32)
    t0 = time.time()
    for c in range(0, nt, chunk):
        sl = slice(c, min(c + chunk, nt))
        for k, n in enumerate(ns):
            x = torch.tensor([ids[d][p - n:p] for d, p in zip(doc[sl], pos[sl])], device=dev)
            with torch.no_grad():
                h = m.base_model(x).last_hidden_state[:, -1]
                lp = torch.log_softmax(m.get_output_embeddings()(h).float(), -1)
            loss[sl, k] = -lp[torch.arange(len(x), device=dev), torch.as_tensor(tgt[sl], device=dev)].cpu().numpy()
        print(short, min(c + chunk, nt), "/", nt, f"{time.time() - t0:.0f}s", flush=True)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, f"{short}_raw.npz"), loss=loss, doc=doc, pos=pos, tgt=tgt, tgtbytes=tb[tgt], ctxb=ctxb, NS=ns)
    res = RF.summarize(short, name, loss / np.log(2), doc, tb[tgt].astype(float), ctxb, ns, len(docs))
    res.update(dtype="float32", device=str(dev), seconds=round(time.time() - t0, 1), n_long_docs=len(docs))
    json.dump(res, open(os.path.join(out, f"{short}.json"), "w"), indent=1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples")
    ap.add_argument("out")
    ap.add_argument("--nt", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--corpora", default="c4,pile,slimpajama,fineweb_edu,dclm,hplt,commoncorpus")
    a = ap.parse_args()
    import torch
    dev = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    for c in a.corpora.split(","):
        name = REF.get(c, DEFAULT_REF)
        out = os.path.join(a.out, c)
        if os.path.exists(os.path.join(out, name.split("/")[-1] + ".json")):
            continue
        docs = [json.loads(l)["text"] for l in open(os.path.join(a.samples, c, "heldout.jsonl"))]
        docs = [d for d in docs if len(d.encode("utf-8")) >= 4 * max(RF.NS)]
        r = run(name, docs, out, a.nt, a.chunk, a.seed, torch, dev)
        print(c, name, r["n_targets"], [(x["n_tokens"], round(x["bits_per_byte"], 3)) for x in r["rows"]], flush=True)


if __name__ == "__main__":
    main()
