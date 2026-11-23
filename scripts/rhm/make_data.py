"""RHM data and data-only predictions.

For each setting: train/val/test windows (one tree of s^L = 256 leaves per window) with the exact next-token laws of the
val and test windows (basis "S", read by scripts/train.py through its "_p" convention), and, from a separate statistics
split, the information curve h(n), the pair correlations C(n), gamma, beta and both predictions. The prediction file is
written with its PT timestamp before any training on the setting.
Usage: python scripts/rhm/make_data.py --name v8m2 [--tiny]
"""
import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from exact_sources import zoo  # noqa: E402
from rhm import RHM, deltas_from_H, info_curve  # noqa: E402

SETTINGS = {  # name: (v, m, s, L, grammar seed)
    "v8m2": (8, 2, 2, 8, 101),
    "v4m2": (4, 2, 2, 8, 105),
    "v8m4": (8, 4, 2, 8, 102),
    "v16m8": (16, 8, 2, 8, 104),
}
SIZES = dict(ntrain=4096, nval=512, ntest=2048, nstats=1024)
NWINS = [16, 64, 256, 1024, 4096]  # 2^12 .. 2^20 tokens at T = 256 (2.4 decades), the zoo's budgets
FIT_NWINS = NWINS  # primary fit range: all five budgets; secondary: 2^14 .. 2^20 (the zoo's fit range)
DATA = ROOT / "data/rhm"
OUT = ROOT / "results/rhm"


def now_pt():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-8))).strftime("%Y-%m-%d %H:%M:%S PST")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tiny", action="store_true", help="small splits, for the smoke test")
    ap.add_argument("--boot", type=int, default=200)
    a = ap.parse_args()
    v, m, s, L, seed = SETTINGS[a.name]
    sz = dict(ntrain=4096, nval=64, ntest=128, nstats=64) if a.tiny else SIZES
    tag = a.name + ("_tiny" if a.tiny else "")
    src = RHM(v, m, s, L, seed)
    T = src.T
    rng = np.random.default_rng(seed + 1000)
    t0 = time.time()
    x = {k: src.sample(sz["n" + k], rng) for k in ("train", "val", "test", "stats")}
    d = dict(S_train=x["train"], S_val=x["val"], S_test=x["test"])
    for k in ("val", "test"):
        d[f"S_{k}_p"] = src.prefix_probs(x[k])
    t_probs = time.time() - t0
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{tag}.npz"
    np.savez(path, **d)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    # statistics split: information curve and correlations
    t1 = time.time()
    H = info_curve(src, x["stats"])
    h, drows = deltas_from_H(H)
    t_info = time.time() - t1
    fro, op = src.pair_corr()
    gamma, g_res = zoo.fit_gamma(h)
    beta, b_res = zoo.fit_beta(fro)
    beta_op, _ = zoo.fit_beta(op)
    beta_16_128, _ = zoo.fit_beta(fro, 16, 128)
    rb = np.random.default_rng(7)
    W = drows.shape[0]
    h0w = H[:, 0, :].mean(1)
    gboot = []
    for _ in range(a.boot):
        i = rb.integers(0, W, W)
        hb = h0w[i].mean() - np.concatenate([[0.0], np.cumsum(drows[i].mean(0))])
        gboot.append(zoo.fit_gamma(hb)[0])
    gboot = np.array(gboot)
    budgets = np.array(NWINS) * T
    fitb = np.array(FIT_NWINS) * T
    tb = zoo.tail_curves(h, fro, np.ones_like(fro), T, budgets)
    fi = np.isin(budgets, fitb)
    law_fin = zoo.slope(fitb, tb["info"][fi])
    law_fin_14 = zoo.slope(budgets[1:], tb["info"][1:])
    law_fin_boot = []
    for _ in range(min(a.boot, 100)):
        i = rb.integers(0, W, W)
        hb = h0w[i].mean() - np.concatenate([[0.0], np.cumsum(drows[i].mean(0))])
        law_fin_boot.append(zoo.slope(fitb, zoo.tail_curves(hb, fro, np.ones_like(fro), T, fitb)["info"]))
    pct = lambda z: [float(np.nanpercentile(z, 2.5)), float(np.nanpercentile(z, 97.5))]
    pred = dict(
        name=a.name, tiny=a.tiny, written_pt=now_pt(), v=v, m=m, s=s, L=L, grammar_seed=seed, T=T, sizes=sz,
        data=str(path.relative_to(ROOT)), data_sha256=sha, entropy_rate_exact=src.entropy_rate(),
        h0=float(h[0]), h_full_mc=float(h[-1]), prefix_entropy_mc=float(H[:, -1, :].mean()),
        prefix_entropy_mc_se=float(H[:, -1, :].mean(1).std() / np.sqrt(H.shape[0])), h_n=h.tolist(), corr_fro=[None if not np.isfinite(c) else float(c) for c in fro],
        corr_op=[None if not np.isfinite(c) else float(c) for c in op],
        gamma=gamma, gamma_fit_rms=g_res, gamma_ci=pct(gboot), beta=beta, beta_fit_rms=b_res, beta_op=beta_op,
        beta_16_128=beta_16_128,
        pred_law_closed=gamma / (1 + gamma), pred_law_closed_ci=pct(gboot / (1 + gboot)),
        pred_law_finite_210=law_fin, pred_law_finite_210_ci=pct(law_fin_boot), pred_law_finite_210_fit14=law_fin_14,
        pred_cagnetta=gamma / (2 * beta), pred_cagnetta_ci=pct(gboot / (2 * beta)), pred_cagnetta_op=gamma / (2 * beta_op),
        law_learned_distances_at_budgets=tb["info_learned"].tolist(), law_tail_at_budgets=tb["info"].tolist(),
        budgets=budgets.tolist(), fit_budgets=fitb.tolist(),
        definitions=dict(
            h="h(n) = h(0) - sum_{k<=n} delta(k); delta(k) = mean over stats windows and positions t >= k of "
              "H(x_t | k-1 tokens) - H(x_t | k tokens), each the entropy of the exact conditional (position known)",
            gamma="zoo.fit_gamma: Delta(n) = h(n) - h(T-1) ~ a (n^-g - N^-g), n = 4..64",
            beta="zoo.fit_beta on C(n) = mean_i ||P(x_i, x_i+n) - P(x_i)P(x_i+n)||_F (exact), n = 4..64",
            law_finite="zoo.tail_curves info rule, price 210 events = 111.5 bits per distance, slope over fit budgets"),
        seconds=dict(probs=t_probs, info_curve=t_info))
    (OUT / "pred").mkdir(parents=True, exist_ok=True)
    pf = OUT / "pred" / f"{tag}.json"
    if pf.exists() and not a.tiny:
        raise SystemExit(f"{pf} exists; refusing to overwrite a written prediction")
    pf.write_text(json.dumps(pred, indent=1))
    print(f"{tag}: gamma {gamma:.3f} {pct(gboot)} beta {beta:.3f} (op {beta_op:.3f}) law {gamma/(1+gamma):.3f} "
          f"law_fin {law_fin:.3f} cag {gamma/(2*beta):.3f} rate {src.entropy_rate():.4f} h(T-1) {h[-1]:.4f} "
          f"probs {t_probs:.0f}s info {t_info:.0f}s -> {pf}")


if __name__ == "__main__":
    main()
