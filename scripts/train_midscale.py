"""Single-pass RoPE GPT on FineWeb-Edu arrays from prep_fineweb.py, raw bytes or GPT-2 BPE, at one training-byte budget.

Training text: the budget is a prefix of the seeded document order in train_{bytes,gpt2}.npy, the same documents for both
encodings (bytes: exactly the first B bytes; BPE: the longest token prefix of at most B bytes). Windows are
non-overlapping, each used once per pass in random order (--passes 2 repeats the same data for the convergence check).
Cosine LR with a 5% warmup, no early stopping.

Evaluation: the valid and test slices are cut into target spans of about --span bytes on GPT-2 token boundaries, the
same spans for both encodings. Each span is scored inside a sliding window of the model's training length (2048 bytes,
or round(2048 / bytes per token) tokens), with the context-only positions masked out, so context is never reset
inside a document and the total scored bytes are identical. The context length in bytes is reported.
Metric: bits per byte = NLL in bits / bytes of the scored tokens.
"""
import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from train_text import GPT

ROOT = Path(__file__).resolve().parent.parent
WARMUP_FRAC = 0.05


def parse_bytes(s):
    return int(2 ** int(s[2:])) if s.startswith("2^") else int(s)


def byte_lengths(enc, data=None):
    if enc == "bytes":
        return np.ones(256, dtype=np.int64)
    if enc == "bpe8k":
        return np.load(Path(data) / "bpe8k" / "token_bytes.npy")
    g = tiktoken.get_encoding("gpt2")
    return np.array([len(g.decode_single_token_bytes(i)) for i in range(g.n_vocab)], dtype=np.int64)


def _byte_unicode():
    """GPT-2 byte -> printable unicode map used by byte-level BPE vocabularies."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b), cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def ensure_bpe8k(data, vocab=8192):
    """Byte-level BPE (HF tokenizers) trained once on the tokenizer_fit bytes; every split is encoded document by document
    and cached under data/bpe8k/ with the exact UTF-8 byte length of each token id (token_bytes.npy)."""
    d = Path(data) / "bpe8k"
    if (d / "meta.json").exists():
        return d
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    d.mkdir(parents=True, exist_ok=True)

    def docs(name):
        x = np.load(Path(data) / f"{name}_bytes.npy")
        st = np.append(np.load(Path(data) / f"{name}_docstarts.npy"), len(x))
        return x, [x[a:b].tobytes().decode("utf-8") for a, b in zip(st[:-1], st[1:])]

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(docs("tokenizer_fit")[1], trainers.BpeTrainer(
        vocab_size=vocab, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=[], show_progress=False))
    tok.save(str(d / "tokenizer.json"))
    u2b = {c: b for b, c in _byte_unicode().items()}
    v = tok.get_vocab()
    assert sorted(v.values()) == list(range(len(v))), "token ids are not contiguous"
    L = np.zeros(len(v), dtype=np.int64)
    for t, i in v.items():
        L[i] = len(bytes(u2b[c] for c in t))
    np.save(d / "token_bytes.npy", L)
    meta = {}
    for name in ("train", "valid", "test"):
        x, dl = docs(name)
        parts = []
        for i in range(0, len(dl), 2000):
            parts += [np.asarray(e.ids, dtype=np.uint16) for e in tok.encode_batch(dl[i:i + 2000])]
        xt = np.concatenate(parts)
        assert int(L[xt].sum()) == len(x), f"{name}: token byte lengths sum to {int(L[xt].sum())}, text has {len(x)} bytes"
        n = min(2000, len(xt))  # the tokens spell the text: spot check the head
        head = b"".join(bytes(u2b[c] for c in t) for t in (tok.id_to_token(int(i)) for i in xt[:n]))
        assert head == x[:len(head)].tobytes()
        np.save(d / f"{name}_bpe8k.npy", xt)
        meta[name] = dict(bytes=len(x), tokens=len(xt), bytes_per_token=len(x) / len(xt))
    meta["vocab"] = len(v)
    tmp = d / "meta.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    os.replace(tmp, d / "meta.json")
    return d


def token_prefix(x, L, B, chunk=1 << 24):
    """Largest N with sum(L[x[:N]]) <= B."""
    total, n = 0, 0
    while n < len(x):
        c = np.cumsum(L[np.asarray(x[n:n + chunk])]) + total
        if c[-1] > B:
            return n + int(np.searchsorted(c, B, side="right"))
        total, n = int(c[-1]), n + chunk
    return len(x)


def eval_spans(xg, Lg, enc, span, T, Tg, aux=None):
    """Target spans on GPT-2 token boundaries (identical for all encodings). Returns window starts a, number of target
    tokens, context bytes, target bytes for encoding enc whose window length is T (Tg = the GPT-2 window length).
    aux = (x8, L8, T8) aligns the evaluation range across encodings: the first start and the last end move to byte offsets
    that are token boundaries in both GPT-2 and bpe8k, so every encoding scores the same total bytes; a bpe8k token
    belongs to the span containing its first byte."""
    cb = np.concatenate([[0], np.cumsum(Lg[xg])])
    cb8 = None
    lb = 2048
    if aux is not None:
        x8, L8, T8 = aux
        cb8 = np.concatenate([[0], np.cumsum(L8[x8])])
        assert cb8[-1] == cb[-1], "bpe8k and GPT-2 arrays cover different bytes"
        lb = max(2049, int(cb[Tg + 1]), int(cb8[T8 + 1]))
    i = max(int(np.searchsorted(cb, lb)), Tg + 1)
    si, sj = [], []
    while True:
        j = int(np.searchsorted(cb, cb[i] + span))
        if j >= len(cb):
            break
        si.append(i), sj.append(j)
        i = j
    si, sj = np.array(si), np.array(sj)
    bs, be = cb[si], cb[sj]
    if aux is not None:
        common = np.intersect1d(cb, cb8)
        bs[0] = common[np.searchsorted(common, bs[0])]
        be[-1] = common[np.searchsorted(common, be[-1], side="right") - 1]
        keep = be > bs
        bs, be = bs[keep], be[keep]
    cbe = {"bytes": None, "gpt2": cb, "bpe8k": cb8}[enc]
    if enc == "bytes":
        ie, je = bs, be
    else:
        ie, je = np.searchsorted(cbe, bs), np.searchsorted(cbe, be)
        assert (je > ie).all()
    a = je - T - 1
    assert a.min() >= 0
    return dict(a=a, n_t=je - ie, ctx=bs - a if enc == "bytes" else cbe[ie] - cbe[a], nbytes=int((be - bs).sum()) if enc == "bytes" else int((cbe[je] - cbe[ie]).sum()))


def evaluate(model, x, sp, T, L, dev, max_spans=None, bs=16):
    idx = np.arange(len(sp["a"]))
    if max_spans and len(idx) > max_spans:
        idx = idx[np.linspace(0, len(idx) - 1, max_spans).astype(int)]
    Lt = torch.as_tensor(L, device=dev)
    model.eval()
    nll, nbytes = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xt = torch.as_tensor(np.stack([x[s:s + T + 1] for s in sp["a"][b]]).astype(np.int64), device=dev)
            nt = torch.as_tensor(sp["n_t"][b], device=dev)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(xt[:, :-1])
            ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), xt[:, 1:].reshape(-1), reduction="none").view(len(b), T)
            mask = torch.arange(T, device=dev)[None] >= (T - nt)[:, None]  # target positions only
            nll += (ce * mask).sum().item()
            nbytes += int((Lt[xt[:, 1:]] * mask).sum().item())
    model.train()
    return nll / math.log(2) / nbytes


def ctx_summary(sp):
    c = sp["ctx"]
    return dict(spans=len(c), target_bytes=sp["nbytes"], min=int(c.min()), p5=float(np.percentile(c, 5)),
                mean=float(c.mean()), p95=float(np.percentile(c, 95)), max=int(c.max()))


def lr_at(step, steps, warmup, lr, floor=0.1):
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(steps - warmup, 1)
    return lr * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enc", required=True, choices=["bytes", "gpt2", "bpe8k"])
    ap.add_argument("--train_bytes", type=parse_bytes, required=True, help="e.g. 2^26")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=832)
    ap.add_argument("--nl", type=int, default=12, help="24 = the 2x capacity variant")
    ap.add_argument("--head_dim", type=int, default=64)
    ap.add_argument("--emb", type=int, default=0, help="factorized tied embedding size (total-parameter control); 0 = untied")
    ap.add_argument("--passes", type=int, default=1, help="2 = the convergence check on the same data")
    ap.add_argument("--ctx_bytes", type=int, default=2048)
    ap.add_argument("--span", type=int, default=256, help="target span of the sliding-window evaluation, bytes")
    ap.add_argument("--bs", type=int, default=32, help="windows per step")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--wd", type=float, default=0.1, help="AdamW weight decay")
    ap.add_argument("--best_ckpt", action="store_true", help="keep the best curve-subset validation state, final valid/test use it")
    ap.add_argument("--align_eval", action="store_true", help="evaluation range shared with bpe8k (implied by --enc bpe8k)")
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--curve_spans", type=int, default=256)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="run --pilot_steps steps, print tokens/s and MFU, write nothing else")
    ap.add_argument("--pilot_steps", type=int, default=40)
    ap.add_argument("--peak_tflops", type=float, default=989.0, help="dense bf16 peak of the GPU (H100 SXM 989, A100 312)")
    ap.add_argument("--data", default=str(ROOT / "data" / "midscale"))
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "midscale"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = Path(args.data)
    k = round(math.log2(args.train_bytes))
    fingerprint = hashlib.sha256((data / "manifest.json").read_bytes()).hexdigest()
    config = dict(enc=args.enc, train_bytes=args.train_bytes, seed=args.seed, d=args.d, nl=args.nl, head_dim=args.head_dim,
                  emb=args.emb, passes=args.passes, ctx_bytes=args.ctx_bytes, span=args.span, bs=args.bs, lr=args.lr,
                  warmup_frac=WARMUP_FRAC, data=fingerprint)
    # added after the first runs: left out at their defaults so earlier config hashes stay valid
    args.align_eval = args.align_eval or args.enc == "bpe8k"
    if args.wd != 0.1:
        config["wd"] = args.wd
    if args.best_ckpt:
        config["best_ckpt"] = True
    if args.align_eval:
        config["align_eval"] = True
    if args.enc == "bpe8k":
        config["bpe8k_vocab"] = 8192
    chash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]
    tag = f"{args.enc}_b{k}_s{args.seed}_d{args.d}x{args.nl}_e{args.emb}_p{args.passes}"
    tag += (f"_wd{args.wd:g}" if args.wd != 0.1 else "") + ("_best" if args.best_ckpt else "") + f"_{chash}"
    out = Path(args.out_dir) / f"{tag}.json"
    if out.exists() and not args.pilot:
        print("exists", out)
        return
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    meta = json.loads((data / "meta.json").read_text())
    Tg = round(args.ctx_bytes / meta["train"]["bytes_per_token"])
    if args.enc == "bpe8k" or args.align_eval:
        d8 = ensure_bpe8k(data)
        meta8 = json.loads((d8 / "meta.json").read_text())
        T8 = round(args.ctx_bytes / meta8["train"]["bytes_per_token"])
    bpt = {"bytes": 1.0, "gpt2": meta["train"]["bytes_per_token"], "bpe8k": meta8["train"]["bytes_per_token"] if args.enc == "bpe8k" else 0}[args.enc]
    T = {"bytes": args.ctx_bytes, "gpt2": Tg, "bpe8k": T8 if args.enc == "bpe8k" else 0}[args.enc]
    L = byte_lengths(args.enc, data)
    V = len(L)

    xall = np.load(d8 / "train_bpe8k.npy", mmap_mode="r") if args.enc == "bpe8k" else np.load(data / f"train_{args.enc}.npy", mmap_mode="r")
    N = args.train_bytes + 1 if args.enc == "bytes" else token_prefix(xall, L, args.train_bytes)
    assert N <= len(xall), f"train array has {len(xall)} items, need {N}"
    xtr = np.asarray(xall[:N])
    nwin = (N - 1) // T
    per_pass = nwin // args.bs
    steps = per_pass * args.passes
    warmup = max(math.ceil(WARMUP_FRAC * steps), 1)
    order = np.concatenate([rng.permutation(nwin)[: per_pass * args.bs] for _ in range(args.passes)])
    n_docs = int(np.searchsorted(np.load(data / "train_docstarts.npy"), args.train_bytes, side="left"))
    ids = (data / "train_docids.txt").read_text().split("\n")[:n_docs]
    ids_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()

    evals = {}
    for name in ("valid", "test"):
        xg = np.load(data / f"{name}_gpt2.npy")
        x8 = np.load(d8 / f"{name}_bpe8k.npy") if args.align_eval else None
        x = {"gpt2": xg, "bytes": None, "bpe8k": x8}[args.enc]
        if x is None:
            x = np.load(data / f"{name}_bytes.npy")
        if args.enc == "bpe8k":  # token byte lengths must tile the split exactly
            assert int(L[x].sum()) == len(np.load(data / f"{name}_bytes.npy", mmap_mode="r")), name
        aux = (x8, byte_lengths("bpe8k", data), T8) if args.align_eval else None
        evals[name] = (x, eval_spans(xg, byte_lengths("gpt2"), args.enc, args.span, T, Tg, aux))

    model = GPT(V, T, args.d, args.nl, args.d // args.head_dim, args.emb).to(dev)
    n_total = sum(p.numel() for p in model.parameters())
    n_body = sum(p.numel() for p in model.blocks.parameters()) + sum(p.numel() for p in model.ln.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)
    fwd = torch.compile(model) if args.compile else model

    # training FLOPs per token: 6 * (body + output matmuls) + 12 * layers * d * T for attention scores and values
    flops_tok = 6 * (n_body + V * (args.emb or args.d) + (2 * args.d * args.emb)) + 12 * args.nl * args.d * T
    print(f"{tag} T={T} bpt={bpt:.3f} steps={steps} warmup={warmup} docs={n_docs} params body={n_body/1e6:.1f}M total={n_total/1e6:.1f}M", flush=True)

    curve, bytes_seen, t0 = [], 0, time.time()
    best = None  # (curve valid bpb, step, CPU copy of the model state)
    n_run = args.pilot_steps if args.pilot else steps
    for step in range(n_run):
        idx = order[step * args.bs:(step + 1) * args.bs]
        xb = np.stack([xtr[s * T:s * T + T + 1] for s in idx]).astype(np.int64)
        bytes_seen += int(L[xb[:, 1:]].sum())
        xt = torch.as_tensor(xb, device=dev)
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, warmup, args.lr)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            logits = fwd(xt[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, V), xt[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if args.pilot and step == 9:  # skip compile and warm-up steps
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t_pilot, tok0 = time.time(), step + 1
        if not args.pilot and ((step + 1) % args.eval_every == 0 or step + 1 == steps):
            curve.append(dict(step=step + 1, bytes=bytes_seen, train_loss=loss.item(),
                              valid_bpb=evaluate(model, *evals["valid"], T, L, dev, args.curve_spans)))
            print(curve[-1], flush=True)
            if args.best_ckpt and (best is None or curve[-1]["valid_bpb"] < best[0]):
                best = (curve[-1]["valid_bpb"], step + 1, {n: v.detach().cpu().clone() for n, v in model.state_dict().items()})

    if dev.type == "cuda":
        torch.cuda.synchronize()
    if args.pilot:
        dt = time.time() - t_pilot
        tps = (n_run - tok0) * args.bs * T / dt
        print(f"PILOT {tag}: {tps:.0f} tokens/s, {tps * bpt:.0f} bytes/s, MFU {tps * flops_tok / (args.peak_tflops * 1e12):.3f}, "
              f"est. full run {steps * args.bs * T * bpt / (tps * bpt) / 3600:.2f} h")
        return
    secs = time.time() - t0
    if best is not None:
        model.load_state_dict(best[2])
    res = dict(config=config, config_hash=chash, encoding=args.enc, train_bytes=args.train_bytes, seed=args.seed, d=args.d,
               nl=args.nl, emb=args.emb, passes=args.passes, T=T, bytes_per_token=bpt, params_nonembedding=n_body,
               params_total=n_total, steps=steps, bs=args.bs, lr=args.lr, warmup=warmup, tokens_seen=steps * args.bs * T,
               bytes_seen=bytes_seen, train_bytes_in_prefix=N - 1 if args.enc == "bytes" else int(L[xtr].sum()),
               train_docs=n_docs, train_docids_sha256=ids_sha, data_fingerprint=fingerprint,
               valid_bpb=evaluate(model, *evals["valid"], T, L, dev),
               test_bpb=evaluate(model, *evals["test"], T, L, dev),
               eval_context_bytes={n: ctx_summary(evals[n][1]) for n in evals}, curve=curve,
               wd=args.wd, best_ckpt=args.best_ckpt, best_step=best[1] if best else None,
               best_curve_valid=best[0] if best else None, align_eval=args.align_eval,
               seconds=secs, tokens_per_s=steps * args.bs * T / secs, compile=args.compile)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, out)
    print(f"{tag} valid_bpb={res['valid_bpb']:.4f} test_bpb={res['test_bpb']:.4f} ({secs:.0f}s)")


if __name__ == "__main__":
    main()
