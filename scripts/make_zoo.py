"""Exact-source zoo: 21 binary sources in windows of T = 256 tokens plus two
T = 64 length controls. Each has exact next-token laws (what the trained network is scored against), an information
curve h(n), pair correlations C(n), entropy rate, compressibility, and data-only predictions of how much and how fast a
network learns it.

What is exact and what is sampled (stored per source as `curve_kind`):
  gated  the per-distance (direct) curve is exact in closed form; the information that reaches x_t through chains of
         copies when its marker points beyond the context is exact per window (belief propagation) and averaged over
         the statistics windows. h(n) = direct - indirect is the full context-conditioned curve, with a window bootstrap.
         Pair correlations: exact (two-point recursion); for the D recoding exact only to lag 46 (four-point recursion
         at T = 48), so the pair rule is marked unavailable for D.
  ising  exact conditional laws (Gaussian fermion algebra) averaged over the statistics windows; pair correlations exact.
  hmm    exact conditional laws (forward algorithm) averaged over the statistics windows; pair correlations exact.
Sampled curves are measured on a grid of context lengths and interpolated (leave-one-point-out error stored), with a
bootstrap over windows propagated into every predicted exponent.

Splits: train / val / test are what scripts/train.py reads. Every statistic that feeds a prediction comes from a separate
statistics split (its own random stream, never used in training or scoring) or from the training windows (the pair-SNR
spread and compressibility, which a data-only forecaster may use). The test split is never read here except to count
markers for the unseen-marker report.

Every source carries a marker channel S_*_c (zeros where there is none) and is trained with a fixed marker capacity
(--ncond NCOND) and the same tokens per update (T = 64 controls use batch 256 windows), so the architecture and
optimizer exposure are identical across sources (`train_args`).

Outputs: <out_data>/<name>.npz and <out_pred>/<name>.json, written atomically, with a config hash and the data file's
SHA-256 (scripts/colab_zoo.sh and scripts/analyze_zoo.py refuse runs whose checksums do not match).
Usage: python scripts/make_zoo.py [--tiny] [--only gated,hmm | name,...] [--list]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exact_sources import gated, tfim, zoo  # noqa: E402

VERSION = "zoo-v2"
FULL = dict(T=256, ntrain=4096, nval=512, ntest=4096, nstats=1024, nwins=[16, 64, 256, 1024, 4096], bs=64, ncond=256)
TINY = dict(T=32, ntrain=64, nval=16, ntest=64, nstats=64, nwins=[4, 8, 16, 32, 64], bs=64, ncond=32)
NBOOT = 200


def registry(tiny=False):
    c = TINY if tiny else FULL
    S = []
    for g, seed in ((0.25, 21), (0.5, 11), (0.75, 22), (1.0, 12), (1.5, 23), (2.0, 13), (3.0, 24)):
        S.append(dict(name=f"gated_p{g:g}", family="gated", shape="power", gamma=g, Q=0.5, eps=0.1, seed=seed, enc="C"))
    S.append(dict(name="gated_geo", family="gated", shape="geo", rho=0.9, Q=0.5, eps=0.1, seed=25, enc="C"))
    S.append(dict(name="gated_step", family="gated", shape="step", lag=8 if tiny else 16, Q=0.004, eps=0.1, seed=26, enc="C"))
    S.append(dict(name="gated_mix", family="gated", shape="mix", gamma=2.0, band=(12, 15) if tiny else (48, 63), Qband=0.05,
                  Q=0.5, eps=0.1, seed=27, enc="C"))
    S.append(dict(name="gated_p1_D", family="gated", shape="power", gamma=1.0, Q=0.5, eps=0.1, seed=12, enc="D",
                  recoding_of="gated_p1"))
    for hx, seed in ((1.0, 31), (1.1, 32), (0.8, 33)):
        S.append(dict(name=f"ising_X_h{hx:g}", family="ising", basis="X", h=hx, sample_h=hx, seed=seed, shape="power" if hx == 1 else "exp"))
        S.append(dict(name=f"ising_Z_h{1 / hx:.3g}", family="ising", basis="Z", h=1 / hx, sample_h=hx, seed=seed,
                      shape="power" if hx == 1 else ("ordered" if hx > 1 else "exp"), recoding_of=f"ising_X_h{hx:g}"))
    S.append(dict(name="hmm_exp", family="hmm", tau=[64], w=[0.4], eps=0.1, seed=41, shape="exp"))
    S.append(dict(name="hmm_two", family="hmm", tau=[4, 96], w=[0.6, 0.35], eps=0.0, seed=42, shape="two-scale"))
    tau6 = [float(v) for v in 2.6 ** np.arange(6) * 2]
    for nm, a, seed in (("hmm_pl_a", 0.25, 43), ("hmm_pl_b", 0.75, 44)):
        w = np.array(tau6) ** -a
        S.append(dict(name=nm, family="hmm", tau=tau6, w=[float(v) for v in 0.95 * w / w.sum()], eps=0.0, seed=seed, shape="power"))
    # length control: the same gated sources in windows 4x shorter, the same q_k for k < T/4, the same token budgets,
    # split sizes in tokens and tokens per optimizer update (batch 4x more windows)
    for g, seed in ((0.5, 11), (1.0, 12)):
        f = 4
        Qs = float(gated.make_q(g, c["T"] - 1, 0.5)[1:c["T"] // f].sum())
        S.append(dict(name=f"gated_p{g:g}_T{c['T'] // f}", family="gated", shape="power", gamma=g, Q=Qs, eps=0.1, seed=seed,
                      enc="C", length_control_of=f"gated_p{g:g}", T=c["T"] // f, ntrain=c["ntrain"] * f, nval=c["nval"] * f,
                      ntest=c["ntest"] * f, nstats=c["nstats"] * f, nwins=[n * f for n in c["nwins"]], bs=c["bs"] * f))
    for s in S:
        for k in ("T", "ntrain", "nval", "ntest", "nstats", "nwins", "bs", "ncond"):
            s.setdefault(k, c[k])
        s["V"] = 2
        s.setdefault("recoding_of", None)
        s.setdefault("length_control_of", None)
        s["train_args"] = f"--pos rope --ncond {s['ncond']} --bs {s['bs']}"
        s["config_hash"] = hashlib.sha256(json.dumps({**s, "version": VERSION}, sort_keys=True).encode()).hexdigest()[:16]
    return S


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def stats_rng(s):
    return np.random.default_rng([s["seed"], 101])  # its own stream: the training/test draws are untouched


# ------------------------------------------------------------------ builders
def gated_windows(s):
    """Same draw order as scripts/make_gated.py (rng -> sigma -> sample), so power sources with the dial's seeds are
    bit-identical to data/gated_g{0.5,1,2}_T256.npz (checked by scripts/check_gated_regen.py)."""
    rng = np.random.default_rng(s["seed"])
    N = s["T"] - 1
    q = gated.make_q_shape(s["shape"], N, s["Q"], gamma=s.get("gamma", 1.0), rho=s.get("rho", 0.9), lag=s.get("lag", 16),
                           band=tuple(s.get("band", (48, 63))), Qband=s.get("Qband", 0.05))
    sigma = gated.make_sigma(N, rng)
    x, m, mx = gated.sample(q, sigma, s["eps"], s["T"], s["ntrain"] + s["nval"] + s["ntest"], rng)
    return x, m, mx, q, sigma


def build_gated(s, a):
    T, eps = s["T"], s["eps"]
    x, mk, mx, q, sigma = gated_windows(s)
    xs, ms, mxs = gated.sample(q, sigma, eps, T, s["nstats"], stats_rng(s))
    hdir = 1 - (1 - zoo.hbin(eps)) * np.cumsum(q)[:T]  # exact direct curve: 1 - (1 - h(eps)) sum_{k<=n} q_k
    ns = zoo.grid_for(T)
    rows = zoo.gated_indirect(xs, ms, sigma, eps, ns)  # (windows, grid), exact per window
    hC = hdir - zoo.indirect_curve(ns, rows.mean(0), T)
    bootC = zoo.boot_curves(ns, rows, None, T, NBOOT, np.random.default_rng([s["seed"], 102]), base=hdir)
    ind = dict(ns=ns.tolist(), indirect=rows.mean(0).tolist(), se=(rows.std(0) / np.sqrt(len(rows))).tolist(), windows=len(rows))
    M2 = gated.two_point(q, sigma, eps, T)
    extra = dict(q=q.tolist(), sigma=sigma.tolist(), indirect_grid=ind)
    rate = float(np.mean(zoo.hbin((1 + mxs) / 2)))
    if s["enc"] == "D":
        x, mx = gated.to_diff(x, mx)
        # H(y_t | n previous y, markers) = H(x_t | n + 1 previous x, markers) exactly (the source is symmetric under a
        # global flip, and y_t is flip-invariant): h_D(n) = h_C(n + 1)
        shift = lambda v: np.append(v[..., 1:], v[..., -1:], axis=-1)
        h, hd, boot = shift(hC), shift(hdir), shift(bootC)
        Tc = min(T, 48)
        absC = np.full(T, np.nan)
        for n, v in gated.four_point_diff_corr(q, sigma, eps, Tc).items():
            absC[n] = abs(v)
        mean = np.zeros(T)
        mean[1:] = np.diagonal(M2, 1)  # exact E[y_t] = E[x_{t-1} x_t]
        ys, mys = gated.to_diff(xs, mxs)
        rate = float(np.mean(zoo.hbin((1 + mys) / 2)))
        kind = "direct exact + indirect sampled (BP per window); pair correlations exact to lag 46 only: pair rule unavailable"
    else:
        h, hd, boot = hC, hdir, bootC
        absC = np.full(T, np.nan)
        for n in range(1, T):
            absC[n] = abs(np.mean(np.diagonal(M2, n)))
        mean = 0.0
        kind = "direct exact + indirect sampled (BP per window); pair correlations exact"
    return dict(x=x, m=mx, c=mk.astype(np.uint8), h=h, h_direct=hd, boot=boot, absC=absC, mean=mean, rate=rate,
                curve_kind=kind, extra=extra)


def kl_rows(cond_last, xs, mfull, ns):
    """(windows, grid) per-window KL(p(x | T-1 tokens) || p(x | last n tokens)), both exact."""
    T = xs.shape[1]
    return np.stack([zoo.kl_mean_bits(mfull, cond_last(xs[:, T - 1 - n:T - 1], n)) for n in ns], 1)


def ising_pair(s, a):
    """X windows at field sample_h and their Kramers-Wannier Z windows (field 1/sample_h), as scripts/make_data.py."""
    T, hx = s["T"], s["sample_h"]
    G = tfim.g_coeffs(hx, T, nk=a.nk)
    Gam = tfim.window_gamma(G, T)

    def draw(n, rng):
        try:
            import torch  # noqa: F401
            return tfim.sample_x_torch(Gam, n, rng, batch=128, threads=a.threads)
        except ImportError:
            return tfim.sample_x(Gam, n, rng, batch=128)

    rng = np.random.default_rng(s["seed"])
    w, mw = draw(s["ntrain"] + s["nval"] + s["ntest"], rng)
    z = tfim.x_to_z(w[:, :-1], rng)
    mz = np.zeros_like(z)
    mz[:, 1:] = z[:, :-1] * mw[:, :-1]
    rs = stats_rng(s)
    ws, mws = draw(s["nstats"], rs)
    zs = tfim.x_to_z(ws[:, :-1], rs)
    mzs = np.zeros_like(zs)
    mzs[:, 1:] = zs[:, :-1] * mws[:, :-1]
    ns = zoo.grid_for(T)

    def cond_last(ctx, n):
        GamN = tfim.window_gamma(G, n + 1)
        return np.concatenate([tfim.conditional_mean_last(GamN, ctx[i:i + 128]) for i in range(0, len(ctx), 128)])

    rows = kl_rows(cond_last, ws, mws[:, T - 1], ns)
    href = zoo.hbin((1 + mws[:, T - 1]) / 2)
    hX = zoo.curve_from_kl(ns, rows.mean(0), float(href.mean()), T)
    bootX = zoo.boot_curves(ns, rows, href, T, NBOOT, np.random.default_rng([s["seed"], 102]))
    rsC = np.arange(1, T)
    cX = np.full(T, np.nan)
    cX[1:] = np.abs([tfim.x_connected(G, r) for r in rsC])
    cZ = np.full(T, np.nan)
    cZ[1:] = np.abs([tfim.string_corr(G, r) for r in rsC])
    grid = dict(ns=ns.tolist(), Delta=rows.mean(0).tolist(), se=(rows.std(0) / np.sqrt(len(rows))).tolist(),
                windows=len(rows), interp=zoo.interp_check(ns, rows.mean(0), T))
    kind = "sampled: exact conditional laws averaged over statistics windows; pair correlations exact"
    X = dict(x=w, m=mw, c=None, h=hX, boot=bootX, absC=cX, mean=float(G[(len(G) - 1) // 2]),
             rate=float(np.mean(zoo.hbin((1 + mws) / 2))), curve_kind=kind, extra=dict(kl_grid=grid))
    shift = lambda v: np.concatenate([np.ones(v.shape[:-1] + (1,)), v[..., :-1]], -1)
    # z_{t-n..t-1} <-> (z_{t-1}, w_{t-n+1..t-1}): h_Z(n) = h_X(n - 1), h_Z(0) = 1 (exact recoding)
    Z = dict(x=z, m=mz, c=None, h=shift(hX), boot=shift(bootX), absC=cZ, mean=0.0, h_gamma=hX,
             rate=float(np.mean(zoo.hbin((1 + mzs) / 2))), curve_kind=kind + "; h_Z(n) = h_X(n-1) of the dual windows",
             extra=dict(kl_grid=grid))
    return X, Z


def build_hmm(s, a):
    T = s["T"]
    src = zoo.Switching(1 / (2 * np.asarray(s["tau"], float)), s["w"], s["eps"])
    x = src.sample(s["ntrain"] + s["nval"] + s["ntest"], T, np.random.default_rng(s["seed"]))
    m = src.cond_means(x)
    xs = src.sample(s["nstats"], T, stats_rng(s))
    ms = src.cond_means(xs)
    ns = zoo.grid_for(T)
    rows = kl_rows(lambda ctx, n: src.mean_after(ctx), xs, ms[:, T - 1], ns)
    href = zoo.hbin((1 + ms[:, T - 1]) / 2)
    h = zoo.curve_from_kl(ns, rows.mean(0), float(href.mean()), T)
    boot = zoo.boot_curves(ns, rows, href, T, NBOOT, np.random.default_rng([s["seed"], 102]))
    grid = dict(ns=ns.tolist(), Delta=rows.mean(0).tolist(), se=(rows.std(0) / np.sqrt(len(rows))).tolist(),
                windows=len(rows), interp=zoo.interp_check(ns, rows.mean(0), T))
    return dict(x=x, m=m, c=None, h=h, boot=boot, absC=np.abs(src.pair_corr(T)), mean=0.0,
                rate=float(np.mean(zoo.hbin((1 + ms) / 2))),
                curve_kind="sampled: exact conditional laws (forward algorithm) averaged over statistics windows; pair correlations exact",
                extra=dict(kl_grid=grid, flip_prob=(1 / (2 * np.asarray(s["tau"]))).tolist()))


# ------------------------------------------------------------------ statistics, predictions, files
def fnum(v):
    return None if v is None or not np.isfinite(v) else float(v)


def finish(s, d, a):
    T, nw = s["T"], s["nwins"]
    ntr, nv = s["ntrain"], s["nval"]
    x, m, h, absC = d["x"], d["m"], d["h"], d["absC"]
    c = d["c"] if d["c"] is not None else np.zeros(x.shape, np.uint8)
    assert c.max() < s["ncond"], "marker capacity too small"
    tok = (x < 0).astype(np.int8)
    data = {}
    for split, lo, hi in (("train", 0, ntr), ("val", ntr, ntr + nv), ("test", ntr + nv, None)):
        data[f"S_{split}"] = tok[lo:hi]
        data[f"S_{split}_m"] = m[lo:hi].astype(np.float64)
        data[f"S_{split}_c"] = c[lo:hi]
    out = Path(a.out_data) / f"{s['name']}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, T=T, name=s["name"], config_hash=s["config_hash"], h_n=h, absC=absC, **data)
    os.replace(tmp, out)

    sd = zoo.window_corr_sd(x[:ntr], d["mean"])
    budgets = np.array(nw) * T
    fitb = budgets[1:]  # the measured budgets of the fit range (2^14 .. 2^20), the same points analyze_zoo.py regresses
    P = 2.0 ** np.linspace(np.log2(budgets[0]) - 0.5, np.log2(budgets[-1]) + 0.5, 81)  # dense: for plots only
    tc = zoo.tail_curves(h, absC, sd, T, P)
    tb, tbl = zoo.tail_curves(h, absC, sd, T, budgets), zoo.tail_curves(h, absC, sd, T, budgets, tmin=T // 2)
    fi = np.isin(budgets, fitb)
    alpha = {k: fnum(zoo.slope(fitb, tb[k][fi])) for k in ("info", "soft", "cag")}
    alpha.update({f"late_{k}": fnum(zoo.slope(fitb, tbl[k][fi])) for k in ("info", "soft", "cag")})
    # uncertainty of the predictions from the sampled part of the curve (window bootstrap)
    bs = {k: [] for k in ("info", "soft", "cag")}
    for hb in d["boot"]:
        tt = zoo.tail_curves(hb, absC, sd, T, fitb)
        for k in bs:
            bs[k].append(zoo.slope(fitb, tt[k]))
    alpha_boot = {k: dict(sd=fnum(np.nanstd(v)) if np.isfinite(v).sum() > 10 else None,
                          ci=[fnum(np.nanpercentile(v, 2.5)), fnum(np.nanpercentile(v, 97.5))] if np.isfinite(v).sum() > 10 else None,
                          undefined_fraction=float(np.mean(~np.isfinite(v)))) for k, v in bs.items()}
    if "h_direct" in d:  # the same rule on the direct-only curve: size of the approximation the full curve removes
        td = zoo.tail_curves(d["h_direct"], absC, sd, T, fitb)
        alpha["info_direct_curve"] = fnum(zoo.slope(fitb, td["info"]))
    gam_fit, gam_rms = zoo.fit_gamma(d.get("h_gamma", h))  # a recoding keeps the gamma of the curve it recodes
    beta_fit, beta_rms = zoo.fit_beta(absC)
    gam = s["gamma"] if s["family"] == "gated" and s["shape"] == "power" else (gam_fit if s["shape"] == "power" else np.nan)
    if s["family"] == "gated" and s["shape"] == "power":
        beta = (1 + gam) if s["enc"] == "C" else (2 + 2 * gam)
    else:
        beta = beta_fit if s["shape"] == "power" else np.nan
    alpha["info_formula"] = fnum(gam / (1 + gam)) if np.isfinite(gam) else None
    alpha["cag_formula"] = fnum(gam / (2 * beta)) if np.isfinite(gam) and np.isfinite(beta) and beta > 0 else None
    comp, comp_raw, comp_null = zoo.compress_bits(tok[:ntr], c[:ntr], np.random.default_rng([s["seed"], 103]))
    pred = dict(
        name=s["name"], version=VERSION, config_hash=s["config_hash"], data_sha256=sha256(out), family=s["family"],
        shape=s["shape"], recoding_of=s["recoding_of"], length_control_of=s["length_control_of"],
        side_input=d["c"] is not None, curve_kind=d["curve_kind"],
        source={k: v for k, v in s.items() if k not in ("name", "family", "shape", "recoding_of", "length_control_of")},
        T=T, V=2, budgets=budgets.tolist(), fit_budgets=fitb.tolist(),
        h_n=h.tolist(), h_sd=np.std(d["boot"], 0).tolist(), absC=[fnum(v) for v in absC], sd_window=[fnum(v) for v in sd],
        stats=dict(h0=float(h[0]), h1=float(h[1]), h_last=float(h[-1]), predictable_bits=float(h[0] - h[-1]),
                   entropy_rate_window=d["rate"], gamma=fnum(gam), gamma_fit=fnum(gam_fit), gamma_fit_rms=fnum(gam_rms),
                   beta=fnum(beta), beta_fit=fnum(beta_fit), beta_fit_rms=fnum(beta_rms),
                   compress_bits=comp, compress_raw=comp_raw, compress_null=comp_null),
        constants=dict(B_star_bits=zoo.B_STAR, K_events=zoo.K_EVENTS, soft_slope=zoo.SOFT_SLOPE, zC=1.0),
        dense=dict(P=P.tolist(), **{k: [fnum(v) for v in tc[k]] for k in tc}),
        at_budgets={**{k: [fnum(v) for v in tb[k]] for k in tb}, **{f"late_{k}": [fnum(v) for v in tbl[k]] for k in tbl}},
        alpha=alpha, alpha_boot=alpha_boot, extra=d["extra"])
    if d["c"] is not None:  # explicit events: exact counts per distance in the first nwin training windows
        mtr, mte = c[:ntr].astype(np.int64), c[ntr + nv:].astype(np.int64)
        Nk = [np.bincount(mtr[:n].ravel(), minlength=T)[:T] for n in nw]
        cte = np.bincount(mte.ravel(), minlength=T)[:T]
        pred["events"] = dict(N_k=[v.tolist() for v in Nk], test_count=cte.tolist(),
                              learned_pred=[[k for k in range(1, T) if row[k] >= zoo.K_EVENTS] for row in Nk],
                              unseen_marker_mass=[float(cte[1:][row[1:] == 0].sum() / mte.size) for row in Nk])
    jp = Path(a.out_pred) / f"{s['name']}.json"
    jp.parent.mkdir(parents=True, exist_ok=True)
    tmp = jp.with_suffix(".tmp")
    tmp.write_text(json.dumps(pred))
    os.replace(tmp, jp)
    al = alpha
    print(f"{s['name']:16s} h0={h[0]:.3f} h_last={h[-1]:.3f} gamma={gam:.3g} beta={beta:.3g} alpha info={al['info']} "
          f"(sd {alpha_boot['info']['sd']}) cag={al['cag']} formula={al['info_formula']}", flush=True)


def is_current(s, a):
    jp, dp = Path(a.out_pred) / f"{s['name']}.json", Path(a.out_data) / f"{s['name']}.npz"
    if not (jp.exists() and dp.exists()):
        return False
    try:
        p = json.loads(jp.read_text())
        return p.get("config_hash") == s["config_hash"] and p.get("data_sha256") == sha256(dp)
    except (ValueError, OSError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="toy sizes (T = 32) for a CPU code test; own output folders")
    ap.add_argument("--only", default="", help="comma list of families or source names")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out_data", default=None, help="default data/zoo (tiny: data/zoo_tiny)")
    ap.add_argument("--out_pred", default=None, help="default results/zoo/pred (tiny: results/zoo_tiny/pred)")
    ap.add_argument("--nk", type=int, default=None, help="momentum grid of the off-critical Ising G(r) (default 2^20, tiny 2^14)")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--skip_existing", action="store_true", help="skip sources whose files match the config and data checksums")
    a = ap.parse_args()
    sub = "zoo_tiny" if a.tiny else "zoo"
    a.out_data = a.out_data or str(ROOT / "data" / sub)
    a.out_pred = a.out_pred or str(ROOT / "results" / sub / "pred")
    a.nk = a.nk or (1 << 14 if a.tiny else 1 << 20)
    full_reg = registry(a.tiny)
    want = set(filter(None, a.only.split(",")))
    reg = [s for s in full_reg if not want or s["name"] in want or s["family"] in want]
    if a.list:
        for s in reg:
            print(s["name"], s["family"], s["shape"], s.get("recoding_of") or "", s["train_args"])
        return
    done = set()
    for s in reg:
        if s["name"] in done or (a.skip_existing and is_current(s, a)):
            continue
        t0 = time.time()
        if s["family"] == "gated":
            finish(s, build_gated(s, a), a)
        elif s["family"] == "hmm":
            finish(s, build_hmm(s, a), a)
        else:
            sx = next(r for r in full_reg if r["family"] == "ising" and r["basis"] == "X" and r["sample_h"] == s["sample_h"])
            sz = next(r for r in full_reg if r["family"] == "ising" and r["basis"] == "Z" and r["sample_h"] == s["sample_h"])
            X, Z = ising_pair(sx, a)
            for ss, dd in ((sx, X), (sz, Z)):
                if not want or ss["name"] in want or "ising" in want:
                    finish(ss, dd, a)
                    done.add(ss["name"])
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
