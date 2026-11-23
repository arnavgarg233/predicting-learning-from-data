"""Single-pass training of the 3.3M byte-level learner (scripts/train_text.py's model, optimizer and evaluation, unchanged)
for the single-pass rate study.

    python -u scripts/rate_singlepass/train_onepass.py --tok_dir data/rs/full/<c>/tok --log2_bytes 27 --seed 0 --out results/text_runs/rs_<c>

The first D = 2^k bytes of the train array are cut into D/ctx non-overlapping windows (inputs x[s:s+ctx], targets x[s+1:s+ctx+1]),
and each window is used exactly once, in a seeded random order, in batches of bs: D/(ctx*bs) steps, no repeated data.
As in train_text.py: AdamW (lr 1e-3, weight decay 0.1), bf16 autocast on CUDA, validation every eval_every steps on val_chunks
byte windows, the best validation state is kept, and after `patience` evaluations without a 0.1% gain the best state is reloaded
and the learning rate divided by 3 (at most max_decays times); training continues with the next unseen windows. The final
validation is at the last step. Test bits per byte of the best state on the full test array (byte-matched windows).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train_text as TT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok_dir", required=True)
    ap.add_argument("--log2_bytes", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--nl", type=int, default=4)
    ap.add_argument("--nh", type=int, default=4)
    ap.add_argument("--ctx_bytes", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--val_chunks", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_decays", type=int, default=3)
    ap.add_argument("--test_chunks", type=int, default=0, help="0 = the full test array")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = Path(a.tok_dir)
    meta = json.loads((data / "tokenization_meta.json").read_text())
    V = meta["bytes_train"]["V"]
    T = a.ctx_bytes                                     # bytes: one token per byte
    L = TT.byte_lengths("bytes", V, data)
    assert (L == 1).all()
    D = 2 ** a.log2_bytes
    tr = np.load(data / "tok_bytes_train.npy", mmap_mode="r")
    assert len(tr) >= D + 1, f"train has {len(tr)} bytes, need {D + 1}"
    dev = torch.device(a.device)
    xtr = torch.as_tensor(np.asarray(tr[: D + 1]).astype(np.uint8)).to(dev)
    nwin = D // T
    order = torch.as_tensor(rng.permutation(nwin) * T, device=dev)
    steps_total = nwin // a.bs
    va = np.asarray(np.load(data / "tok_bytes_valid.npy", mmap_mode="r"))
    vt = np.asarray(np.load(data / "tok_bytes_test.npy", mmap_mode="r"))
    nmax = 2 * T + 8

    def mk(x, n):
        w, _ = TT.byte_windows(x, L, T, nmax)
        return w[np.linspace(0, len(w) - 1, n).astype(int)] if n and len(w) > n else w

    val_starts = mk(va, a.val_chunks)
    model = TT.GPT(V, nmax + 1, a.d, a.nl, a.nh, 0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    ar = torch.arange(T + 1, device=dev)
    best, best_state, bad, decays, curve = float("inf"), None, 0, 0, []
    t0 = time.time()
    step = 0
    while step < steps_total:
        n = min(a.eval_every, steps_total - step)
        for _ in range(n):
            s = order[step * a.bs:(step + 1) * a.bs]
            xb = xtr[s[:, None] + ar[None]].long()
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logits = model(xb[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, V), xb[:, 1:].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
        nll, nb, _ = TT.evaluate(model, va, val_starts, T, L, dev)
        val = nll / nb
        curve.append([step, val, opt.param_groups[0]["lr"]])
        if val < best * (1 - 1e-3):
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= a.patience and decays < a.max_decays and step < steps_total:
            model.load_state_dict(best_state)
            for g in opt.param_groups:
                g["lr"] /= 3
            decays, bad = decays + 1, 0
    train_seconds = time.time() - t0
    model.load_state_dict(best_state)
    starts = mk(vt, a.test_chunks)
    nll, nb, nt = TT.evaluate(model, vt, starts, T, L, dev)
    res = dict(tok="bytes", one_pass=True, train_bytes=D, windows=nwin, windows_used=steps_total * a.bs, steps=step, seed=a.seed,
               d=a.d, nl=a.nl, nh=a.nh, ctx_bytes=T, lr=a.lr, bs=a.bs, decays=decays, best_val_bpb=best, bits_per_byte=nll / nb,
               eval_bytes=nb, eval_chunks=len(starts), val_curve=curve, train_seconds=train_seconds, seconds=time.time() - t0,
               tok_dir=str(data), args=vars(a))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"bytes_b{a.log2_bytes}_s{a.seed}.json"
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.rename(f)
    print(f"b{a.log2_bytes} s{a.seed} steps={step} bpb={nll / nb:.4f} best_val={best:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
