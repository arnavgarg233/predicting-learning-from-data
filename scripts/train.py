"""Train a small causal transformer (or, with --arch, an LSTM, diagonal SSM or fixed-context MLP) on one basis and budget; report exact excess loss.

Excess loss is the test-set mean of KL(true next-token law || model) in bits per token,
computed from the exact conditional means stored with the data.

Generic stored bases (any name with a "<basis>_train" key, e.g. the gated-copy C/D or block-token
X2/Z4 data): "<basis>_<split>_m" holds binary conditional means, or "<basis>_<split>_p" holds exact
next-token distributions over a vocabulary V; an optional "<basis>_<split>_c" holds a side input
(the gated-copy marker) that the model sees at the position it predicts; "<basis>_spt" is the number
of spins per token, used to report excess per spin.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.nh = nh

    def forward(self, x, rope=None):
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.nh, d // self.nh).permute(2, 0, 3, 1, 4)
        if rope is not None:
            q, k = rotate(q, *rope), rotate(k, *rope)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, d))
        return x + self.mlp(self.ln2(x))


def rotate(x, cos, sin):
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class GPT(nn.Module):
    def __init__(self, T, d=128, nl=4, nh=4, pos="abs", V=2, ncond=0):
        super().__init__()
        self.tok = nn.Embedding(V + 1, d)  # index V = start token
        self.use_rope = pos == "rope"
        if self.use_rope:
            hd = d // nh
            ang = torch.outer(torch.arange(T).float(), 1.0 / 10000 ** (torch.arange(0, hd, 2).float() / hd))
            self.register_buffer("cos", ang.cos()[None, None], persistent=False)
            self.register_buffer("sin", ang.sin()[None, None], persistent=False)
        else:
            self.pos = nn.Parameter(torch.zeros(1, T, d))
            nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.V = V
        if ncond:  # created last so the binary model's initialisation is unchanged
            self.cond = nn.Embedding(ncond, d)

    def forward(self, tokens, cond=None):
        inp = torch.cat([torch.full_like(tokens[:, :1], self.V), tokens[:, :-1]], dim=1)
        T = inp.shape[1]
        h = self.tok(inp)
        if cond is not None:
            h = h + self.cond(cond)
        rope = None
        if self.use_rope:
            rope = (self.cos[:, :, :T], self.sin[:, :, :T])
        else:
            h = h + self.pos[:, :T]
        for blk in self.blocks:
            h = blk(h, rope)
        return self.head(self.ln(h))


def shifted(tokens, V):
    """Inputs at each position: the previous token (start token V at position 0), as in GPT."""
    return torch.cat([torch.full_like(tokens[:, :1], V), tokens[:, :-1]], dim=1)


class LSTMLM(nn.Module):
    """Stacked LSTM; the side input (gated-copy marker) is added to the input at the predicted position."""
    def __init__(self, d=256, nl=2, V=2, ncond=0):
        super().__init__()
        self.tok = nn.Embedding(V + 1, d)
        self.rnn = nn.LSTM(d, d, nl, batch_first=True)
        self.head = nn.Linear(d, V)
        self.V = V
        if ncond:
            self.cond = nn.Embedding(ncond, d)

    def forward(self, tokens, cond=None):
        h = self.tok(shifted(tokens, self.V))
        if cond is not None:
            h = h + self.cond(cond)
        return self.head(self.rnn(h)[0])


class S4DReal(nn.Module):
    """Diagonal linear recurrence x_t = exp(dt A) x_{t-1} + B u_t, y_t = C x_t + D u_t, per channel, with
    real negative A (S4D-Real init A_n = -(n+1)), zero-order-hold discretisation; applied as a causal FFT
    convolution with the kernel K_l = sum_n C_n Bbar_n exp(dt A_n l)."""
    def __init__(self, d, N=16, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        self.log_dt = nn.Parameter(torch.rand(d) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.log_A = nn.Parameter(torch.log(torch.arange(1, N + 1).float()).repeat(d, 1))
        self.C = nn.Parameter(torch.randn(d, N) / math.sqrt(N))
        self.D = nn.Parameter(torch.randn(d))

    def forward(self, u):  # (B, T, d)
        T = u.shape[1]
        dtA = -torch.exp(self.log_A) * torch.exp(self.log_dt)[:, None]  # (d, N)
        Bbar = torch.expm1(dtA) / -torch.exp(self.log_A)  # ZOH with B = 1
        K = torch.einsum("hn,hnl->hl", self.C * Bbar, torch.exp(dtA[..., None] * torch.arange(T, device=u.device)))
        uf = torch.fft.rfft(u.transpose(1, 2), n=2 * T)
        y = torch.fft.irfft(uf * torch.fft.rfft(K, n=2 * T), n=2 * T)[..., :T].transpose(1, 2)
        return y + u * self.D


class SSMBlock(nn.Module):
    def __init__(self, d, N=16):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ssm, self.out = S4DReal(d, N), nn.Linear(d, d)
        self.w1, self.w3, self.w2 = nn.Linear(d, 4 * d), nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.out(F.gelu(self.ssm(self.ln1(x))))
        h = self.ln2(x)
        return x + self.w2(F.gelu(self.w1(h)) * self.w3(h))  # gated MLP


class SSMLM(nn.Module):
    def __init__(self, d=128, nl=4, V=2, ncond=0, N=16):
        super().__init__()
        self.tok = nn.Embedding(V + 1, d)
        self.blocks = nn.ModuleList([SSMBlock(d, N) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.V = V
        if ncond:
            self.cond = nn.Embedding(ncond, d)

    def forward(self, tokens, cond=None):
        h = self.tok(shifted(tokens, self.V))
        if cond is not None:
            h = h + self.cond(cond)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.ln(h))


class MLPLM(nn.Module):
    """Fixed context: the previous ctx symbols, one-hot (symbol V = before the start), through nl hidden layers;
    the side input enters the first hidden layer (one-hot, i.e. an embedding added to it)."""
    def __init__(self, d=512, nl=3, V=2, ncond=0, ctx=64):
        super().__init__()
        self.inp = nn.Linear(ctx * (V + 1), d)
        layers = [nn.GELU()]
        for _ in range(nl - 1):
            layers += [nn.Linear(d, d), nn.GELU()]
        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(d, V)
        self.V, self.ctx = V, ctx
        if ncond:
            self.cond = nn.Embedding(ncond, d)

    def forward(self, tokens, cond=None):
        B, T = tokens.shape
        seq = torch.cat([torch.full((B, self.ctx), self.V, dtype=tokens.dtype, device=tokens.device), tokens[:, :-1]], 1)
        win = seq.unfold(1, self.ctx, 1)  # (B, T, ctx): tokens t-ctx .. t-1
        h = self.inp(F.one_hot(win, self.V + 1).float().reshape(B, T, -1))
        if cond is not None:
            h = h + self.cond(cond)
        return self.head(self.hidden(h))


ARCH_SIZE = dict(transformer=(128, 4), lstm=(128, 2), ssm=(128, 4), mlp=(512, 3))


def true_probs(m):
    """Exact next-token distribution: binary conditional means (n, T) or distributions (n, T, V)."""
    if m.ndim == 3:
        return m
    p1 = (1 - m) / 2  # probability of token 1 (value -1)
    return np.stack([1 - p1, p1], axis=-1)


def evaluate(model, x, m, device, bs=512, c=None):
    model.eval()
    kl, nll = [], []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.as_tensor(x[i:i + bs], dtype=torch.long, device=device)
            cb = None if c is None else torch.as_tensor(c[i:i + bs], dtype=torch.long, device=device)
            logq = F.log_softmax(model(xb, cb).float().cpu().double(), dim=-1).numpy()
            p = true_probs(m[i:i + bs])
            with np.errstate(divide="ignore", invalid="ignore"):
                kl.append(np.sum(np.where(p > 0, p * (np.log(p) - logq), 0.0), axis=-1))
            nll.append(-np.take_along_axis(logq, x[i:i + bs, :, None].astype(int), -1)[..., 0])
    model.train()
    kl, nll = np.concatenate(kl), np.concatenate(nll)
    return kl.mean() / math.log(2), nll.mean() / math.log(2), kl.mean(axis=0) / math.log(2)


def load_split(D, basis, split):
    """X or Z data, or B<b>: the X data recoded by a running product restarted every b tokens."""
    if basis in ("X", "Z"):
        return D[f"{basis}_{split}"], D[f"{basis}_{split}_m"]
    if f"{basis}_{split}" in D.files:  # any other stored basis: binary means (_m) or distributions (_p)
        for suf in ("_p", "_m"):
            if f"{basis}_{split}{suf}" in D.files:
                return D[f"{basis}_{split}"], D[f"{basis}_{split}{suf}"]
        return D[f"{basis}_{split}"], None
    blk = int(basis[1:])
    w = 1.0 - 2.0 * D[f"X_{split}"]
    mw = D[f"X_{split}_m"]
    z, mz = w.copy(), mw.copy()
    for t in range(1, w.shape[1]):
        if t % blk:
            z[:, t] = z[:, t - 1] * w[:, t]
            mz[:, t] = z[:, t - 1] * mw[:, t]
    return (z < 0).astype(np.int8), mz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--basis", required=True)
    ap.add_argument("--nwin", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arch", choices=list(ARCH_SIZE), default="transformer")
    ap.add_argument("--d", type=int, default=None, help="width (default per arch: transformer 128, lstm 128 (plan: 256, reduced for CPU time), ssm 128, mlp 512)")
    ap.add_argument("--nl", type=int, default=None, help="layers (default per arch: 4, 2, 4, 3)")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (default: torch's own)")
    ap.add_argument("--resume", action="store_true", help="continue from <out>.abort.pt if present (memory watchdog)")
    ap.add_argument("--pos", choices=["abs", "rope"], default="abs")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=60000)
    ap.add_argument("--max_decays", type=int, default=3)
    ap.add_argument("--sched", choices=["plateau", "cosine"], default="plateau",
                    help="plateau: protocol v2 (patience, lr/3 decays, early stop); cosine: fixed max_steps, 2%% warmup + cosine, no early stop")
    ap.add_argument("--clip", type=float, default=0.0, help="gradient-norm clip (0 = none)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--ncond", type=int, default=None,
                    help="fixed side-input capacity (default: max marker + 1, as before)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    args.d = ARCH_SIZE[args.arch][0] if args.d is None else args.d
    args.nl = ARCH_SIZE[args.arch][1] if args.nl is None else args.nl
    if args.threads:
        torch.set_num_threads(args.threads)
    stop_file = Path(str(args.out) + ".stop")  # memory watchdog: when it appears, save a checkpoint and exit (code 3)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    D = np.load(args.data)
    b = args.basis
    xtr = load_split(D, b, "train")[0][: args.nwin]
    if len(xtr) < args.nwin:
        raise SystemExit(f"{args.data} has only {len(xtr)} training windows, --nwin {args.nwin} requested")
    xva, mva = load_split(D, b, "val")
    xte, mte = load_split(D, b, "test")
    cond = {s: D[f"{b}_{s}_c"] if f"{b}_{s}_c" in D.files else None for s in ("train", "val", "test")}
    V = mte.shape[-1] if mte.ndim == 3 else 2
    ncond = int(max(cond[s].max() for s in cond)) + 1 if cond["train"] is not None else 0
    if args.ncond is not None and ncond:
        if ncond > args.ncond:
            raise SystemExit(f"marker {ncond - 1} exceeds --ncond {args.ncond}")
        ncond = args.ncond
    spt = int(D[f"{b}_spt"]) if f"{b}_spt" in D.files else 1
    T = xtr.shape[1]
    dev = torch.device(args.device)
    if args.arch == "transformer":
        model = GPT(T, args.d, args.nl, pos=args.pos, V=V, ncond=ncond).to(dev)
    else:
        model = dict(lstm=LSTMLM, ssm=SSMLM, mlp=MLPLM)[args.arch](args.d, args.nl, V=V, ncond=ncond).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    xtr_t = torch.as_tensor(xtr, dtype=torch.long, device=dev)
    ctr_t = None if not ncond else torch.as_tensor(cond["train"][: args.nwin], dtype=torch.long, device=dev)

    best, best_state, bad, step, decays, t0 = float("inf"), None, 0, 0, 0, time.time()
    ckpt, t_prev = Path(str(args.out) + ".abort.pt"), 0.0
    if args.resume and ckpt.exists():  # continue a run stopped by the memory watchdog exactly where it stopped
        S = torch.load(ckpt, weights_only=False)
        model.load_state_dict(S["model"]); opt.load_state_dict(S["opt"]); rng.bit_generator.state = S["rng"]
        best, best_state, bad, step, decays, t_prev = S["best"], S["best_state"], S["bad"], S["step"], S["decays"], S["seconds"]
        stop_file.unlink(missing_ok=True)
        print(f"resumed from {ckpt} at step {step}", flush=True)
    while step < args.max_steps:
        for _ in range(args.eval_every):
            idx = torch.as_tensor(rng.integers(0, len(xtr), min(args.bs, len(xtr))), device=dev)
            xb = xtr_t[idx]
            cb = None if ctr_t is None else ctr_t[idx]
            loss = F.cross_entropy(model(xb, cb).reshape(-1, V), xb.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            if args.sched == "cosine":
                w = max(1, args.max_steps // 50)
                f = (step + 1) / w if step < w else 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, args.max_steps - w)))
                for g in opt.param_groups:
                    g["lr"] = args.lr * f
            opt.step()
            step += 1
        val_kl, _, _ = evaluate(model, xva, mva, dev, c=cond["val"])
        if val_kl < best * (1 - 1e-3):
            best, bad = val_kl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if args.sched == "plateau" and bad >= args.patience:
            if decays >= args.max_decays:
                break
            model.load_state_dict(best_state)
            for g in opt.param_groups:
                g["lr"] /= 3
            decays, bad = decays + 1, 0
        if stop_file.exists():
            torch.save(dict(best_state=best_state, model=model.state_dict(), opt=opt.state_dict(), step=step, bad=bad,
                            decays=decays, best=best, rng=rng.bit_generator.state, seconds=t_prev + time.time() - t0,
                            args=vars(args)), ckpt)
            print(f"memory watchdog: saved {args.out}.abort.pt at step {step}, exiting", flush=True)
            raise SystemExit(3)
    model.load_state_dict(best_state)
    kl, nll, kl_pos = evaluate(model, xte, mte, dev, c=cond["test"])
    if mte.ndim == 2:
        floor = float(np.mean(-np.log2((1 + mte) / 2 * (xte == 0) + (1 - mte) / 2 * (xte == 1))))
    else:
        floor = float(np.mean(-np.log2(np.take_along_axis(mte, xte[..., None].astype(int), -1))))
    res = dict(basis=b, nwin=args.nwin, P=int(args.nwin * T), seed=args.seed, d=args.d, nl=args.nl,
               steps=step, decays=decays, protocol="v2", pos=args.pos, best_val_kl=best, test_kl_bits=kl, test_nll_bits=nll, test_true_nll_bits=floor,
               kl_by_pos=kl_pos.tolist(), seconds=t_prev + time.time() - t0, hx=float(D["hx"]) if "hx" in D.files else None)
    if args.arch != "transformer":
        res.update(arch=args.arch, n_params=sum(p.numel() for p in model.parameters()))
    if V != 2 or spt != 1 or ncond:
        res.update(V=V, spins_per_token=spt, P_spins=int(args.nwin * T * spt), test_kl_bits_per_spin=kl / spt,
                   data=str(args.data))
    if ncond:  # exact excess grouped by the side input (gated copy: by copy distance; 0 = fair coin)
        model.eval()
        with torch.no_grad():
            byc, cnt = np.zeros(ncond), np.zeros(ncond)
            for i in range(0, len(xte), 512):
                xb = torch.as_tensor(xte[i:i + 512], dtype=torch.long, device=dev)
                cb = torch.as_tensor(cond["test"][i:i + 512], dtype=torch.long, device=dev)
                logq = F.log_softmax(model(xb, cb).float().cpu().double(), dim=-1).numpy()
                p = true_probs(mte[i:i + 512])
                with np.errstate(divide="ignore", invalid="ignore"):
                    k = np.sum(np.where(p > 0, p * (np.log(p) - logq), 0.0), axis=-1) / math.log(2)
                np.add.at(byc, cond["test"][i:i + 512].ravel(), k.ravel())
                np.add.at(cnt, cond["test"][i:i + 512].ravel(), 1)
        res.update(kl_sum_by_cond=byc.tolist(), count_by_cond=cnt.tolist())
    import hashlib
    import os
    with open(args.data, "rb") as f:
        res.update(data_sha256=hashlib.sha256(f.read()).hexdigest(), config=vars(args), ncond=ncond, T=T)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.out) + ".tmp")
    tmp.write_text(json.dumps(res))
    os.replace(tmp, args.out)
    print(f"{b} nwin={args.nwin} seed={args.seed} steps={step} excess={kl:.4e} bits/token "
          f"nll={nll:.4f} true={floor:.4f} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
