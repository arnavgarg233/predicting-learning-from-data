"""Train a small RoPE GPT on TinyStories under one tokenizer and training-byte budget.

Metric: bits per byte = total NLL (bits) / UTF-8 bytes of the predicted (target) tokens.
Byte length of each token id is the length of its ByteLevel token string (one character = one byte).
The only special token, <|endoftext|> (id 0), counts 13 bytes because the byte stream spells it out.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path(__file__).resolve().parent.parent / "data" / "text"
ROOT = Path(__file__).resolve().parent.parent
BUILTIN_TOKS = ["bytes", "bpe512", "bpe2048", "bpe8192", "pair2", "pair3", "wpair2", "wpair3", "sbpe"]


class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.nh = nh

    def forward(self, x, rope):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.nh, d // self.nh).permute(2, 0, 3, 1, 4)
        q, k = rotate(q, *rope), rotate(k, *rope)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


def rotate(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class GPT(nn.Module):
    def __init__(self, V, T, d=256, nl=4, nh=4, emb=0):
        super().__init__()
        self.emb = emb
        if emb:
            # factorized, tied embedding: V x emb shared by input and output, projected to d
            self.tok = nn.Embedding(V, emb)
            self.up = nn.Linear(emb, d, bias=False)
            self.down = nn.Linear(d, emb, bias=False)
        else:
            self.tok = nn.Embedding(V, d)
        hd = d // nh
        ang = torch.outer(torch.arange(T).float(), 1.0 / 10000 ** (torch.arange(0, hd, 2).float() / hd))
        self.register_buffer("cos", ang.cos()[None, None], persistent=False)
        self.register_buffer("sin", ang.sin()[None, None], persistent=False)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        if not emb:
            self.head = nn.Linear(d, V)

    def forward(self, tokens, last=None):  # last: per-row position whose logits are wanted (saves the head matmul elsewhere)
        T = tokens.shape[1]
        h = self.up(self.tok(tokens)) if self.emb else self.tok(tokens)
        rope = (self.cos[:, :, :T], self.sin[:, :, :T])
        for blk in self.blocks:
            h = blk(h, rope)
        h = self.ln(h)
        if last is not None:
            h = h[torch.arange(len(h), device=h.device), last]
        if self.emb:
            return self.down(h) @ self.tok.weight.T
        return self.head(h)


def byte_lengths(tok, V, data=DATA):
    if data != DATA:  # generic tokenizer family from text/tok_family.py: byte length of every id is stored with the arrays
        L = np.load(data / f"tok_{tok}_L.npy").astype(np.int64)
        assert len(L) == V
        return L
    if tok == "bytes":
        return np.ones(V, dtype=np.int64)
    if tok == "pair2":
        return np.full(V, 2, dtype=np.int64)
    if tok == "pair3":
        return np.full(V, 3, dtype=np.int64)
    if tok in ("wpair2", "wpair3"):  # base-257 chunk keys (text/make_wpair.py); length = number of nonzero digits
        m = int(tok[-1])
        key = np.load(DATA / f"{tok}_chunks.npy").astype(np.int64)
        return sum(((key // 257 ** j) % 257 > 0).astype(np.int64) for j in range(m))
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(str(DATA / f"{tok}.json"))
    return np.array([len(t.id_to_token(i)) for i in range(V)], dtype=np.int64)


def chunks(x, T, stride, n=None):
    starts = np.arange(0, len(x) - T - 1, stride)
    if n is not None and len(starts) > n:
        starts = starts[np.linspace(0, len(starts) - 1, n).astype(int)]
    return starts


def byte_windows(x, L, ctx_bytes, nmax):
    """Non-overlapping eval windows (start, n_targets) whose target bytes are as close to ctx_bytes as the token boundaries allow,
    so every tokenizer is scored on the same byte context. Window = x[s : s + n + 1] (inputs x[s:s+n], targets x[s+1:s+n+1])."""
    cb = np.concatenate([[0], np.cumsum(L[x].astype(np.int64))])
    wins, s = [], 0
    while s + 2 <= len(x) - 1:
        e = int(np.searchsorted(cb, cb[s + 1] + ctx_bytes))
        e = min(e, len(x))
        if e - 1 > s + 1 and abs(cb[e - 1] - cb[s + 1] - ctx_bytes) < abs(cb[e] - cb[s + 1] - ctx_bytes):
            e -= 1
        if cb[e] - cb[s + 1] < 0.5 * ctx_bytes:  # truncated tail of the stream
            break
        n = max(min(e - s - 1, nmax), 1)
        if s + n + 1 > len(x):
            break
        wins.append((s, n))
        s += n
    return np.array(wins, dtype=np.int64), cb


def evaluate_win(model, x, wins, L, device, nmax, bs=64, pos=None):
    """Like evaluate for variable-length windows (padded to nmax; the model is causal so padding after n does not change earlier logits)."""
    model.eval()
    nll, nbytes, ntok = 0.0, 0, 0
    ar = np.arange(nmax + 1)
    with torch.no_grad():
        for i in range(0, len(wins), bs):
            w = wins[i:i + bs]
            xn = x[np.minimum(w[:, :1] + ar[None], len(x) - 1)].astype(np.int64)
            mask = ar[None, :nmax] < w[:, 1:2]
            xb = torch.as_tensor(xn, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(xb[:, :-1])
            tok_nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1), reduction="none")
            tok_nll = tok_nll.reshape(len(w), nmax).cpu().numpy() * mask
            nll += float(tok_nll.sum())
            nbytes += int((L[xn[:, 1:]] * mask).sum())
            ntok += int(mask.sum())
            if pos is not None:
                b = np.floor(np.log2(np.cumsum(L[xn[:, :-1]], axis=1))).astype(int)
                pos["nll"] += np.bincount(b.ravel(), (tok_nll / math.log(2)).ravel(), minlength=12)[:12]
                pos["bytes"] += np.bincount(b.ravel(), (L[xn[:, 1:]] * mask).ravel(), minlength=12)[:12]
                # per window position (target index 0..nmax-1; for bytes this is the byte position), output only (2026-10-17)
                pos.setdefault("idx_nll", np.zeros(nmax))
                pos.setdefault("idx_bytes", np.zeros(nmax))
                pos["idx_nll"] += (tok_nll / math.log(2)).sum(0)
                pos["idx_bytes"] += (L[xn[:, 1:]] * mask).sum(0)
    model.train()
    return nll / math.log(2), nbytes, ntok


TRUNC_BYTES = (8, 16, 32, 64, 128, 256, 512)


def trunc_eval(model, x, L, cuts, M, nmax, device, bs=64):
    """Bits per byte of the next token with the context truncated to ~c bytes (cut at the token boundary nearest c bytes before the
    target), for each c in cuts. The same M target positions for every c: those with at least max(cuts) + 64 bytes of history."""
    cb = np.concatenate([[0], np.cumsum(L[x].astype(np.int64))])
    ok = np.flatnonzero(cb[:-1] >= max(cuts) + 64)
    ok = ok[ok < len(x)]
    tg = ok[np.linspace(0, len(ok) - 1, min(M, len(ok))).astype(int)]
    tb = L[x[tg]].sum()
    out = {}
    model.eval()
    with torch.no_grad():
        for c in cuts:
            j = np.searchsorted(cb, cb[tg] - c)  # first j with cb[j] >= cb[i] - c
            j = np.where(np.abs(cb[np.maximum(j - 1, 0)] - (cb[tg] - c)) < np.abs(cb[j] - (cb[tg] - c)), np.maximum(j - 1, 0), j)
            j = np.clip(j, tg - nmax, tg - 1)  # at least one preceding token; the target itself is never an input
            assert (j < tg).all()
            n = tg - j  # context tokens (>= 1 since cb[tg] - cb[j] ~ c > 0)
            nll = 0.0
            for i in range(0, len(tg), bs):
                nn = n[i:i + bs]
                ar = np.arange(int(nn.max()) + 1)
                idx = np.minimum(j[i:i + bs, None] + ar[None], len(x) - 1)
                xb = torch.as_tensor(x[idx].astype(np.int64), device=device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model(xb[:, :-1], last=torch.as_tensor(nn - 1, device=device))
                last = logits.float()  # prediction after the full context
                nll += F.cross_entropy(last, torch.as_tensor(x[tg[i:i + bs]].astype(np.int64), device=device), reduction="sum").item()
            out[str(c)] = nll / math.log(2) / tb
    model.train()
    return out, len(tg)


def evaluate(model, x, starts, T, L, device, bs=64, pos=None):
    if starts.ndim == 2:
        return evaluate_win(model, x, starts, L, device, int(starts[:, 1].max()), bs, pos)
    """Return (total NLL in bits, total target bytes, total target tokens) over the chunks.
    If pos is a dict, also accumulate NLL and target bytes by context length in bytes (log2 bins)."""
    model.eval()
    nll, nbytes = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(starts), bs):
            xb = torch.as_tensor(np.stack([x[s:s + T + 1] for s in starts[i:i + bs]]).astype(np.int64), device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(xb[:, :-1])
            tok_nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), xb[:, 1:].reshape(-1), reduction="none")
            nll += tok_nll.sum().item()
            xn = xb.cpu().numpy()
            nbytes += int(L[xn[:, 1:]].sum())
            if pos is not None:
                b = np.floor(np.log2(np.cumsum(L[xn[:, :-1]], axis=1))).astype(int).ravel()
                pos["nll"] += np.bincount(b, tok_nll.cpu().numpy() / math.log(2), minlength=12)[:12]
                pos["bytes"] += np.bincount(b, L[xn[:, 1:]].ravel(), minlength=12)[:12]
    model.train()
    return nll / math.log(2), nbytes, len(starts) * T


def provenance(args):
    """Fingerprint of everything a result depends on (tok_dir runs): trainer code, effective arguments, the tokenizer's recorded
    artifact and corpus checksums, the test-array record if present."""
    sha = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731
    d = Path(args.tok_dir)
    man = json.loads((d / "manifest.json").read_text())
    tm = d / "test_manifest.json"
    p = dict(trainer_sha256=sha(Path(__file__).read_bytes()),
             args={k: v for k, v in vars(args).items() if k not in ("out_dir", "tag", "fingerprint_only")},
             manifest_sha256=sha((d / "manifest.json").read_bytes()), tokenizer=man["tokenizers"][args.tok], corpus_sha256=man["corpus_sha256"], manifest_seed=man["seed"],
             test_manifest_sha256=sha(tm.read_bytes()) if tm.exists() else None)
    p["fingerprint"] = sha(json.dumps(p, sort_keys=True, default=str).encode())
    return p


def parse_bytes(s):
    return int(2 ** int(s[2:])) if s.startswith("2^") else int(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True)
    ap.add_argument("--tok_dir", default=None,
                    help="directory written by text/tok_family.py (manifest.json, tokenization_meta.json, tok_{tok}_{train,valid,test}.npy, "
                         "tok_{tok}_L.npy); --tok is then any name in its manifest, and --test_split scores on the separate test array")
    ap.add_argument("--train_bytes", type=parse_bytes, required=True, help="power of two, e.g. 2^20 or 1048576")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--nl", type=int, default=4)
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--ctx_bytes", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--val_chunks", type=int, default=2000)
    ap.add_argument("--final_chunks", type=int, default=0, help="0 = full valid split (non-overlapping chunks)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--emb", type=int, default=0, help="factorized tied embedding size (0 = untied full)")
    ap.add_argument("--max_decays", type=int, default=3)
    ap.add_argument("--test_split", action="store_true",
                    help="early stopping / lr decay on the first half of the valid BYTES, final bpb on the second half")
    ap.add_argument("--bpe_dropout", type=float, default=0.0,
                    help="train on BPE-dropout re-segmentations (data/text/tok_{tok}_drop{p}_s{i}_train.npy from text/make_dropout.py, "
                         "one sample per pass over the byte budget, cycled); eval stays canonical. 0 = off")
    ap.add_argument("--trunc_targets", type=int, default=1500, help="targets for the truncated-context evaluation (tok_dir runs)")
    ap.add_argument("--fingerprint_only", action="store_true", help="print the provenance fingerprint of this run and exit (tok_dir runs)")
    ap.add_argument("--dropout_dir", default=str(DATA))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    if args.tok_dir is None and args.tok not in BUILTIN_TOKS:
        ap.error(f"argument --tok: invalid choice: {args.tok!r} (choose from {BUILTIN_TOKS}, or pass --tok_dir)")
    data = Path(args.tok_dir) if args.tok_dir else DATA
    prov = provenance(args) if args.tok_dir else None
    if args.fingerprint_only:
        print(prov["fingerprint"] if prov else "")
        return

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    meta = json.loads((data / "tokenization_meta.json").read_text())
    V = meta[f"{args.tok}_train"]["V"]
    bpt = meta[f"{args.tok}_train"]["bytes_per_token"]
    T = round(args.ctx_bytes / bpt)
    L = byte_lengths(args.tok, V, data)

    tr = np.load(data / f"tok_{args.tok}_train.npy", mmap_mode="r")
    va = np.load(data / f"tok_{args.tok}_valid.npy", mmap_mode="r")
    head = np.asarray(tr[: int(args.train_bytes / bpt * 1.5) + T + 2])
    cum = np.cumsum(L[head])
    N = int(np.searchsorted(cum, args.train_bytes)) + 1
    assert N <= len(head) and cum[N - 1] >= args.train_bytes
    xtr = torch.as_tensor(head[:N].astype(np.int64))
    nbytes_train = int(cum[N - 1])
    va = np.asarray(va)
    if args.test_split and args.tok_dir:  # separate test array (FineWeb-Edu): early stopping on valid, final bpb on test
        vt = np.asarray(np.load(data / f"tok_{args.tok}_test.npy", mmap_mode="r"))
        cut_byte = int(L[va].sum())
    elif args.test_split:
        # Cut at the first token boundary at or after the byte midpoint, so the cut is at (nearly) the same byte
        # offset for every tokenizer: it differs between tokenizers by less than one token's bytes (<= max L).
        cum_va = np.cumsum(L[va])
        cut = int(np.searchsorted(cum_va, cum_va[-1] / 2)) + 1
        cut_byte = int(cum_va[cut - 1])
        va, vt = va[:cut], va[cut:]

    dev = torch.device(args.device)
    xtrs, T_tr = None, T
    if args.bpe_dropout > 0:  # every sample covers the same first train_bytes (to the next token boundary) as the canonical run
        xtrs, nb_s = [], []
        for i in range(8):
            f = Path(args.dropout_dir) / f"tok_{args.tok}_drop{args.bpe_dropout:g}_s{i}_train.npy"
            if not f.exists():
                break
            h = np.load(f)
            c = np.cumsum(L[h])
            assert c[-1] >= args.train_bytes, f"{f.name} covers only {c[-1]} bytes"
            n_i = int(np.searchsorted(c, args.train_bytes)) + 1
            xtrs.append(torch.as_tensor(h[:n_i].astype(np.int32)).to(dev))
            nb_s.append(int(c[n_i - 1]))
        assert xtrs, "no dropout samples found"
        nbytes_train = int(np.mean(nb_s))
        # training windows span ctx_bytes as in every other run (dropout spellings are shorter); eval stays canonical T
        T_tr = round(args.ctx_bytes * np.mean([len(x) for x in xtrs]) / nbytes_train)
        spe = max(int(np.mean([len(x) for x in xtrs])) // (args.bs * T_tr), 1)  # steps per pass; one dropout sample per pass
    nmax = 2 * T + 8  # most targets in one byte-matched eval window (tok_dir runs)
    model = GPT(V, max(T, T_tr, nmax + 1 if args.tok_dir else 0), args.d, args.nl, args.nh, args.emb).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    xtr = xtr.to(dev)

    def mk(x, n):  # eval windows: fixed T tokens (TinyStories tokenizers) or ctx_bytes bytes (tok_dir family)
        if not args.tok_dir:
            return chunks(x, T, T, n)
        w, _ = byte_windows(x, L, args.ctx_bytes, nmax)
        return w[np.linspace(0, len(w) - 1, n).astype(int)] if n is not None and len(w) > n else w

    val_starts = mk(va, args.val_chunks)
    ar = torch.arange(T + 1, device=dev)
    ar_tr = torch.arange(T_tr + 1, device=dev)

    best, best_state, bad, step, decays, t0 = float("inf"), None, 0, 0, 0, time.time()
    while step < args.max_steps:
        for _ in range(args.eval_every):
            if xtrs is None:
                s = torch.as_tensor(rng.integers(0, max(N - T - 1, 1), args.bs), device=dev)
                xb = xtr[s[:, None] + ar[None]]
            else:
                xs = xtrs[(step // spe) % len(xtrs)]
                s = torch.as_tensor(rng.integers(0, max(len(xs) - T_tr - 1, 1), args.bs), device=dev)
                xb = xs[s[:, None] + ar_tr[None]].long()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(xb[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, V), xb[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step >= args.max_steps:
                break
        nll, nb, _ = evaluate(model, va, val_starts, T, L, dev)
        val = nll / nb
        if val < best * (1 - 1e-3):
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= args.patience:
            if decays >= args.max_decays:
                break
            model.load_state_dict(best_state)
            for g in opt.param_groups:
                g["lr"] /= 3
            decays, bad = decays + 1, 0
    model.load_state_dict(best_state)
    if args.test_split:
        starts_a = mk(va, args.final_chunks or None)
        nll_a, nb_a, _ = evaluate(model, va, starts_a, T, L, dev)
        va = vt
    starts = mk(va, args.final_chunks or None)
    pos = {"nll": np.zeros(12), "bytes": np.zeros(12)}
    nll, nb, nt = evaluate(model, va, starts, T, L, dev, pos=pos)
    res = dict(tok=args.tok, V=V, T=T, train_bytes=args.train_bytes, train_bytes_exact=nbytes_train, train_tokens=N,
               seed=args.seed, d=args.d, nl=args.nl, nh=args.nh, steps=step, decays=decays,
               best_val_bpb=best, bits_per_byte=nll / nb, bits_per_token=nll / nt, eval_chunks=len(starts),
               eval_bytes=nb, eval_tokens=nt, seconds=time.time() - t0, emb=args.emb,
               bpe_dropout=args.bpe_dropout, dropout_samples=0 if xtrs is None else len(xtrs), T_train=T_tr,
               pos_bin_log2_ctx_bytes=list(range(12)), pos_bpb=(pos["nll"] / np.maximum(pos["bytes"], 1)).tolist(),
               pos_bytes=pos["bytes"].tolist())
    if "idx_nll" in pos:  # test bits/byte at each window position, best checkpoint (position 0 = first target of the window)
        res.update(test_bpb_by_position=(pos["idx_nll"] / np.maximum(pos["idx_bytes"], 1)).tolist(),
                   test_bytes_by_position=pos["idx_bytes"].tolist())
    if prov:
        res["provenance"] = prov
        res["args"] = {k: v for k, v in vars(args).items() if k != "fingerprint_only"}
    if args.tok_dir:  # learned-context curve: bits/byte with the context truncated to c bytes (same targets for every c)
        t_tr = time.time()
        res["trunc_bpb"], res["trunc_targets"] = trunc_eval(model, va, L, TRUNC_BYTES, args.trunc_targets, nmax, dev)
        res["trunc_seconds"] = time.time() - t_tr
    if args.tok_dir:  # byte context actually seen: eval windows (bytes of targets) and mean train window
        cbw = np.array([L[va[s + 1:s + 1 + n]].sum() for s, n in starts])
        res.update(eval_ctx_bytes=dict(mean=float(cbw.mean()), sd=float(cbw.std()), min=int(cbw.min()), max=int(cbw.max())),
                   train_ctx_bytes_mean=float(T_tr * nbytes_train / (N if xtrs is None else np.mean([len(x) for x in xtrs]))),
                   eval_tokens_per_window=float(starts[:, 1].mean()))
    if args.test_split:
        res.update(test_split=True, bits_per_byte_valA=nll_a / nb_a, eval_bytes_valA=nb_a, test_cut_byte=cut_byte)
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "text_runs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    k = round(math.log2(args.train_bytes))
    (out_dir / f"{args.tok}_b{k}_s{args.seed}.json").write_text(json.dumps(res))
    print(f"{args.tok} b{k} s{args.seed} T={T} steps={step} bpb={nll/nb:.4f} bpt={nll/nt:.4f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
