"""Bits per byte of open-sci-ref-0.01 checkpoints on fixed evaluation samples (Colab GPU/CPU or Mac).

    python eval_ckpt.py --jobs jobs.txt --samples DIR --out results.jsonl [--device cuda] [--dtype float32]
      jobs.txt   lines 'repo revision sample_name'  (revision = iter_00XXXXX or main)
      DIR        <sample_name>.jsonl, one {"text": ...} per line, NFC-normalised (make_samples.py)
    Appends one JSON line per finished job to results.jsonl and skips jobs already there (resumable).

Evaluation follows Megatron's training layout: documents tokenised with the GPT-NeoX-20B tokenizer, each followed by
the end-of-document token (id 0), concatenated and cut into non-overlapping 4096-token sequences. The loss counts every
target that is not the end-of-document token; bits per byte = summed target NLL in bits / summed UTF-8 bytes of those
targets (byte-level BPE, so token byte lengths are exact; checked against each document's byte count).

The network is re-implemented here in plain PyTorch from the released safetensors and config.json (pre-norm decoder,
RMSNorm, RMSNorm on per-head queries and keys, biased projections, SwiGLU, rotate-half RoPE, tied embeddings), so no
repository code is executed. Its correctness is checked by matching the final checkpoint's loss to the validation loss
in the training log.
"""
import argparse
import json
import os
import time

import numpy as np

EOD = 0
SEQ = 4096


def bytes_to_unicode():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def token_bytes(tok, V):
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    tb = np.zeros(V, np.int64)
    for t, i in tok.get_vocab().items():
        tb[i] = len(t) if all(c in u2b for c in t) else len(t.encode("utf-8"))
    return tb


def stream(tok, texts, tb):
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    out = []
    for t, x in zip(texts, ids):
        assert int(tb[np.asarray(x, np.int64)].sum()) == len(t.encode("utf-8")), "token byte lengths do not add up"
        out.extend(x)
        out.append(EOD)
    a = np.asarray(out, np.int64)
    n = (len(a) // SEQ) * SEQ
    return a[:n].reshape(-1, SEQ)


def load_model(repo, rev, torch, dev, dt, cache):
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import torch.nn.functional as F
    cfg = json.load(open(hf_hub_download(repo, "config.json", revision=rev, cache_dir=cache)))
    sd = load_file(hf_hub_download(repo, "model.safetensors", revision=rev, cache_dir=cache))
    sd = {k: v.to(dev, dt) for k, v in sd.items()}
    H, D, eps, L = cfg["num_attention_heads"], cfg["head_dim"], cfg["rms_norm_eps"], cfg["num_hidden_layers"]
    assert cfg["num_key_value_heads"] == H and cfg["tie_word_embeddings"] and cfg["qk_layernorm"]
    assert cfg.get("rope_scaling") is None and cfg["hidden_act"] == "silu"
    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, D, 2, dtype=torch.float32, device=dev) / D))

    def rms(x, w):
        xf = x.float()
        return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)

    def lin(x, p):
        return F.linear(x, sd[p + ".weight"], sd.get(p + ".bias"))

    def rot(x):
        x1, x2 = x[..., : D // 2], x[..., D // 2:]
        return torch.cat((-x2, x1), -1)

    def forward(x):
        B, T = x.shape
        f = torch.outer(torch.arange(T, device=dev, dtype=torch.float32), inv)
        e = torch.cat((f, f), -1)
        cos, sin = e.cos().to(dt)[None, None], e.sin().to(dt)[None, None]
        h = sd["model.embed_tokens.weight"][x]
        for i in range(L):
            p = f"model.layers.{i}."
            a = rms(h, sd[p + "input_layernorm.weight"])
            q, k, v = (lin(a, p + f"self_attn.{n}_proj").view(B, T, H, D).transpose(1, 2) for n in "qkv")
            q = rms(q, sd[p + "self_attn.q_layernorm.weight"])
            k = rms(k, sd[p + "self_attn.k_layernorm.weight"])
            q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            h = h + lin(o.transpose(1, 2).reshape(B, T, H * D), p + "self_attn.o_proj")
            m = rms(h, sd[p + "post_attention_layernorm.weight"])
            h = h + lin(F.silu(lin(m, p + "mlp.gate_proj")) * lin(m, p + "mlp.up_proj"), p + "mlp.down_proj")
        return rms(h, sd["model.norm.weight"])

    forward.unembed = sd["model.embed_tokens.weight"]

    return forward


def evaluate(model, X, tb, torch, dev, bs):
    nll_bits, nbytes, ntok = 0.0, 0, 0
    for i in range(0, len(X), bs):
        x = torch.as_tensor(X[i:i + bs], device=dev)
        with torch.no_grad():
            h = model(x)[:, :-1]
            tgt = x[:, 1:]
            nll = torch.empty(tgt.shape, device=dev)
            for c in range(0, tgt.shape[1], 512):   # logits in chunks of positions (memory)
                lp = torch.log_softmax(torch.nn.functional.linear(h[:, c:c + 512], model.unembed).float(), -1)
                nll[:, c:c + 512] = -lp.gather(-1, tgt[:, c:c + 512, None])[..., 0]
        keep = (tgt != EOD)
        nll_bits += float((nll * keep).sum().item()) / np.log(2)
        t = X[i:i + bs, 1:]
        nbytes += int(tb[t[t != EOD]].sum())
        ntok += int(keep.sum().item())
    return nll_bits / nbytes, nll_bits / ntok * np.log(2), nbytes, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--cache", default=None, help="HF cache dir; each checkpoint is deleted after use when set")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer
    dev = torch.device(a.device if a.device != "cuda" or torch.cuda.is_available() else "cpu")
    dt = getattr(torch, a.dtype)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["repo"], r["revision"], r["sample"]))
    jobs = [l.split() for l in open(a.jobs) if l.strip()]
    tok, tb, Xs = None, None, {}
    for repo, rev, sname in jobs:
        if (repo, rev, sname) in done:
            continue
        t0 = time.time()
        if tok is None:
            tok = AutoTokenizer.from_pretrained(repo, revision=rev)
            tb = token_bytes(tok, 50304)
        if sname not in Xs:
            texts = [json.loads(l)["text"] for l in open(os.path.join(a.samples, sname + ".jsonl"))]
            X = stream(tok, texts, tb)
            Xs[sname] = X[:a.max_seqs] if a.max_seqs else X
        m = load_model(repo, rev, torch, dev, dt, a.cache)
        t1 = time.time()
        bpb, npt, nbytes, ntok = evaluate(m, Xs[sname], tb, torch, dev, a.bs)
        r = dict(repo=repo, revision=rev, sample=sname, bpb=bpb, nats_per_token=npt, n_bytes=nbytes, n_tokens=ntok,
                 n_seqs=int(len(Xs[sname])), dtype=a.dtype, device=str(dev), load_s=round(t1 - t0, 1),
                 eval_s=round(time.time() - t1, 1))
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        if a.cache:
            import shutil
            for d in os.listdir(a.cache):
                if d.startswith("models--") or d == "blobs":   # huggingface_hub 1.x keeps blobs at the cache root
                    shutil.rmtree(os.path.join(a.cache, d), ignore_errors=True)


if __name__ == "__main__":
    main()
