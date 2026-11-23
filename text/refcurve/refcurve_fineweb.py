"""FineWeb-Edu version of refcurve_eval.py: reference-LM cross-entropy (bits per byte) against context length.

  python text/refcurve/refcurve_fineweb.py --data data/bfine --out results/bfine/refcurve [--device cuda]

For each reference LM and each n in NS the input is exactly the n previous tokens of the same document (own forward pass
per n, absolute positions start at 0, no BOS). Targets: the same number of positions per model, drawn uniformly from
all ref_stats positions (valid and test are never read) with >= max(NS) in-document predecessors (so every n is available and the targets are
identical across n; paired). Documents come from ref_stats_bytes.npy cut at ref_stats_docstarts.npy (scripts/prep_fineweb.py).
Bits per byte at n = sum of target NLL (bits) / sum of UTF-8 bytes of the target tokens; SE by bootstrap over documents.
Context length is reported in bytes: exact mean of the summed byte lengths of the n context tokens.

Only documents that are already NFC-normalized are used (Pythia's tokenizer applies NFC, which would change the byte
count); the same documents for every model, so the curves stay paired.
Reference models (all small, byte-level BPE, so token byte lengths are exact; Pythia's raw-space added tokens count their
UTF-8 bytes):
  HuggingFaceTB/SmolLM2-135M   likely in distribution: its pretraining mixture is mostly FineWeb-Edu (SmolLM2 paper), with DCLM
                               and code; so it has very probably seen documents like these.
  EleutherAI/pythia-160m       never saw FineWeb-Edu: trained on the Pile, released 2023, before FineWeb-Edu (2024).
  openai-community/gpt2        never saw FineWeb-Edu: WebText, 2019.
The spread between the in-distribution model and the two others sets the systematic width of the absolute forecasts.
Output per model: {short}_raw.npz (per-target losses) and {short}.json (the table); refcurve_fineweb.json merges them.
"""
import argparse
import json
import time
import unicodedata
from pathlib import Path

import numpy as np

MODELS = ["HuggingFaceTB/SmolLM2-135M", "EleutherAI/pythia-160m", "openai-community/gpt2"]
NS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def load_docs(data, min_bytes):
    raw = np.load(Path(data) / "ref_stats_bytes.npy", mmap_mode="r")
    st = np.load(Path(data) / "ref_stats_docstarts.npy")
    en = np.append(st[1:], len(raw))
    docs, n_long = [], 0
    for s, e in zip(st.tolist(), en.tolist()):
        if e - s >= min_bytes:
            n_long += 1
            d = bytes(raw[s:e]).rstrip(b"\n").decode("utf-8")
            if unicodedata.is_normalized("NFC", d):
                docs.append(d)
    print(f"{n_long - len(docs)} of {n_long} long documents dropped as not NFC-normalized", flush=True)
    return docs


def run_model(name, docs, a, torch, dev, key):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    short = name.split("/")[-1]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(name)
    # torch < 2.1.1 (RunPod torch-v21 image) fails transformers' SDPA check; eager attention computes the same thing
    old = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:3]) < (2, 1, 1)
    m = AutoModelForCausalLM.from_pretrained(name, **({"attn_implementation": "eager"} if old else {})).to(dev).eval()
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    V = max(len(tok), m.get_output_embeddings().weight.shape[0])
    tb = np.zeros(V, np.int32)
    for t, i in tok.get_vocab().items():
        tb[i] = len(t) if all(c in u2b for c in t) else len(t.encode("utf-8"))  # byte-level: one character per byte
    ids = [tok(d, add_special_tokens=False)["input_ids"] for d in docs]
    for d, x in zip(docs, ids):
        assert int(tb[x].sum()) == len(d.encode("utf-8")), f"{name}: token byte lengths do not add up"
    ns = [int(v) for v in a.ns.split(",")]
    nmax = max(ns)
    ok_pos = np.array([max(len(x) - nmax, 0) for x in ids])
    assert ok_pos.sum() > 0, "no document is long enough for the largest context"
    rng = np.random.default_rng(a.seed)
    nt = min(a.nt, int(ok_pos.sum()))
    flat = np.sort(rng.choice(int(ok_pos.sum()), nt, replace=False))
    cs = np.cumsum(ok_pos)
    doc = np.searchsorted(cs, flat, side="right")
    pos = flat - (cs[doc] - ok_pos[doc]) + nmax
    tgt = np.array([ids[d][p] for d, p in zip(doc, pos)])
    ctxb = np.array([[tb[ids[d][p - n:p]].sum() for n in ns] for d, p in zip(doc, pos)])
    loss = np.zeros((nt, len(ns)), np.float32)
    t0 = time.time()
    for c in range(0, nt, a.chunk):
        sl = slice(c, min(c + a.chunk, nt))
        for k, n in enumerate(ns):
            x = torch.tensor([ids[d][p - n:p] for d, p in zip(doc[sl], pos[sl])], device=dev)
            with torch.no_grad():
                h = m.base_model(x).last_hidden_state[:, -1]
                lp = torch.log_softmax(m.get_output_embeddings()(h).float(), -1)
            loss[sl, k] = -lp[torch.arange(len(x), device=dev), torch.as_tensor(tgt[sl], device=dev)].cpu().numpy()
        print(short, min(c + a.chunk, nt), "/", nt, f"{time.time() - t0:.0f}s", flush=True)
    np.savez(out / f"{short}_raw.npz", loss=loss, doc=doc, pos=pos, tgt=tgt, tgtbytes=tb[tgt], ctxb=ctxb, NS=ns)
    res = summarize(short, name, loss / np.log(2), doc, tb[tgt].astype(float), ctxb, ns, len(docs))
    res["key"] = key
    (out / f"{short}.json").write_text(json.dumps(res, indent=1))
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return res


def summarize(short, name, L, doc, tb, cb, ns, n_docs_total, B=1000):
    ud, inv = np.unique(doc, return_inverse=True)
    nd = len(ud)
    S_tb = np.bincount(inv, weights=tb, minlength=nd)
    bi = np.random.default_rng(1).integers(0, nd, (B, nd))
    rows = []
    for k, n in enumerate(ns):
        Sk = np.bincount(inv, weights=L[:, k], minlength=nd)
        bpb = Sk.sum() / S_tb.sum()
        se = (Sk[bi].sum(1) / S_tb[bi].sum(1)).std()
        rows.append(dict(n_tokens=int(n), bits_per_token=float(L[:, k].mean()), bits_per_byte=float(bpb), bpb_se=float(se),
                         context_bytes_exact_mean=float(cb[:, k].mean())))
    return dict(model=name, short=short, n_targets=int(len(doc)), n_docs=int(nd), n_docs_total=int(n_docs_total),
                mean_bytes_per_target_token=float(tb.mean()), rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bfine")
    ap.add_argument("--out", default="results/bfine/refcurve")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--ns", default=",".join(map(str, NS)))
    ap.add_argument("--nt", type=int, default=20000, help="targets per model")
    ap.add_argument("--chunk", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    import hashlib
    h = lambda f: hashlib.sha256((Path(a.data) / f).read_bytes()).hexdigest()  # noqa: E731
    corpus = [h("ref_stats_bytes.npy"), h("ref_stats_docstarts.npy")]
    key = lambda name: dict(model=name, corpus=corpus, ns=a.ns, nt=a.nt, seed=a.seed, docs="nfc_only")  # noqa: E731
    docs = load_docs(a.data, 4 * max(int(x) for x in a.ns.split(",")))
    print(len(docs), "documents long enough", flush=True)
    res = {}
    for name in a.models.split(","):
        short = name.split("/")[-1]
        f = Path(a.out) / f"{short}.json"
        cached = json.loads(f.read_text()) if f.exists() else None
        res[short] = cached if cached and cached.get("key") == key(name) else run_model(name, docs, a, torch, dev, key(name))
    (Path(a.out) / "refcurve_fineweb.json").write_text(json.dumps(res, indent=1))
    for short, r in res.items():
        print(short, [(x["n_tokens"], round(x["bits_per_byte"], 3)) for x in r["rows"]])


if __name__ == "__main__":
    main()
