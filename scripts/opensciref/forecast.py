"""Learning-curve forecast from the data on open-sci-ref-0.01 checkpoints, against pilot extrapolation.

    python3 scripts/opensciref/forecast.py inputs   # -> results/curve_forecast/opensciref/forecast_inputs.json
    python3 scripts/opensciref/forecast.py score    # -> .../scores.json (needs evals_<size>.jsonl)

`inputs` is run before any learner checkpoint beyond the step-0 smoke is evaluated. It records the data statistics,
reference curves, run constants and the full design below; nothing in it depends on a learner checkpoint.

Design:
  learner bits/byte L_c(t) of corpus c's checkpoint at iteration t on c's held-out sample (eval_ckpt.py)
  D = t * tokens_per_iter * bytes_per_token_c  (training bytes)
  shape  S_c(D) = h_c(n0 D^nu), nu = 1/(1+gamma_c), n0 = (A_c gamma_c / K)^nu; h_c the reference model's information
         curve on c's held-out text, (gamma_c, A_c) the paper's power fit h(n) - h(960) = A n^-gamma over 8-64 bytes;
         K one constant per fold: grid least squares of (L(t) - L(t*)) - (S(t) - S(t*)) over the other corpora at
         FIT_ITERS (constant-LR checkpoints), t* = ANCHOR
  F0_x   L = a + b x_c + S(D) - S(D*)   (a, b) least squares on the other corpora's L(t*); x = bigram (primary), ref
         bits/byte r (secondary), best n-gram g (additional)
  Ff_x   final checkpoint only: a + b x_c, (a, b) fitted on the other corpora's final losses
  P2     power law through 26k and 28k with no floor (primary baseline); P2e through 2k and 4k (secondary)
  P1     L(26k) + S(D) - S(D_26k)
  scored ranges: constant LR = SCORE_ITERS; final = main (reported separately)
"""
import datetime
import hashlib
import json
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.environ.get("OSR_RES", os.path.join(ROOT, "results/curve_forecast/opensciref"))
CORPORA = ["c4", "pile", "slimpajama", "fineweb_edu", "dclm", "hplt", "commoncorpus"]
RUN = {"c4": "c4-300B-4096-warmup25000", "pile": "pile-300B-4096", "slimpajama": "slimpajama-300B-4096",
       "fineweb_edu": "fineweb-edu-1.4t-300B-4096", "dclm": "dclm-300B-4096", "hplt": "hplt-2.0-300B-4096",
       "commoncorpus": "commoncorpus-300B-4096"}
REFMODEL = {c: ("Qwen2.5-0.5B" if c == "commoncorpus" else "SmolLM2-135M") for c in CORPORA}
SIZES = ["0.13b", "0.4b"]
EVAL_ITERS = [2000, 4000, 26000, 28000, 30000, 36000, 42000, 50000, 58000, 66000]   # plus main
PILOTS = (26000, 28000)
PILOTS_EARLY = (2000, 4000)
PILOT1 = 26000
ANCHOR = 42000
FIT_ITERS = [26000, 28000, 30000, 36000, 42000, 50000, 58000]
SCORE_ITERS = [30000, 36000, 42000, 50000, 58000]
DECAY_ITERS = [66000]
LOG10K = np.round(np.arange(0.0, 20.0001, 0.01), 4)
WINDOW, N_REF = (8.0, 64.0), 960.0
LEVELS = ["bigram", "ref", "g"]
NBOOT, SEED = 20000, 20261024
DUPES = []


def repo(size, c):
    return f"open-sci/open-sci-ref-v0.01-{size}-{RUN[c]}"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class HRef:
    def __init__(self, rows):
        rows = sorted(rows, key=lambda r: r["context_bytes_exact_mean"])
        self.x = np.log([r["context_bytes_exact_mean"] for r in rows])
        self.y = np.array([r["bits_per_byte"] for r in rows])

    def h(self, n):
        return np.interp(np.log(n), self.x, self.y)

    def power_fit(self):
        n = np.geomspace(WINDOW[0], WINDOW[1], 12)
        p = np.polyfit(np.log(n), np.log(self.h(n) - self.h(N_REF)), 1)
        return float(-p[0]), float(np.exp(p[1]))


# ------------------------------------------------------------------------------------------------ inputs
def inputs():
    samples = sys.argv[2] if len(sys.argv) > 2 else os.path.join(RES, "samples")
    ds = json.load(open(os.path.join(RES, "data_stats.json")))
    logs = json.load(open(os.path.join(RES, "step0/log_summary.json")))["runs"]
    out = dict(created_pt=datetime.datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(timespec="seconds"),
               note="Written before any learner checkpoint beyond the step-0 smoke (C4 and FineWeb-Edu 0.13b main and "
                    "iter_0010000 on smoke samples) was evaluated.",
               design=dict(eval_iters=EVAL_ITERS, plus="main", pilots=PILOTS, pilots_early=PILOTS_EARLY, pilot1=PILOT1,
                           anchor=ANCHOR, fit_iters_K=FIT_ITERS, score_iters_constant_lr=SCORE_ITERS,
                           decay_iters_descriptive=DECAY_ITERS, final="main", log10K_grid=[0.0, 20.0, 0.01],
                           power_fit_window_bytes=WINDOW, power_fit_ref_bytes=N_REF, levels=LEVELS,
                           primary_level="bigram", secondary_level="ref", arms_primary=["F0_bigram", "Ff_bigram", "P2"],
                           bootstrap=dict(n=NBOOT, seed=SEED, unit="corpus"), sizes=SIZES, primary_size="0.13b"),
               code_sha256={os.path.relpath(p, ROOT): sha(p) for p in
                            [os.path.join(ROOT, "scripts/opensciref", f) for f in
                             ("make_samples.py", "data_stats.py", "refcurve.py", "eval_ckpt.py", "forecast.py")] +
                            [os.path.join(ROOT, "scripts/ngram_curve_ts.py")]},
               corpora={}, runs={})
    for c in CORPORA:
        rc = json.load(open(os.path.join(RES, "refcurve", c, REFMODEL[c] + ".json")))
        hr = HRef(rc["rows"])
        g, A = hr.power_fit()
        meta = json.load(open(os.path.join(samples, c, "meta.json")))
        r_full = float(hr.y[-1])
        out["corpora"][c] = dict(
            ref_model=REFMODEL[c], ref_full_bpb=r_full, gamma=g, A=A, nu=1.0 / (1.0 + g),
            ref_rows=[dict(n_tokens=x["n_tokens"], context_bytes=x["context_bytes_exact_mean"], bpb=x["bits_per_byte"],
                           se=x["bpb_se"]) for x in rc["rows"]], ref_n_targets=rc["n_targets"], ref_n_docs=rc["n_docs"],
            bigram=ds[c]["bigram"], g=ds[c]["g"], g_order=ds[c]["g_order"], c_ratio=r_full / ds[c]["g"],
            bytes_per_token=ds[c]["bytes_per_token"], sample_meta=meta)
        for s in SIZES:
            L = logs[f"{s}-{RUN[c]}"]
            iters = EVAL_ITERS if not (s == "0.13b" and c == "commoncorpus") else [i for i in EVAL_ITERS if i >= 24000]
            out["runs"][f"{s}/{c}"] = dict(repo=repo(s, c), tokens_per_iter=L["tokens_per_iter"], main_iter=L["train_iters"],
                                          warmup_iters=L["warmup_iters"], decay_start_iter=L["train_iters"] - L["wsd_decay_iters"],
                                          eval_iters=iters, train_over_heldout_log=L["train_over_heldout"],
                                          D_eff_tokens_diag=L["train_iters"] * L["tokens_per_iter"] * L["train_over_heldout"])
    p = os.path.join(RES, "forecast_inputs.json")
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p, "sha256", sha(p), out["created_pt"])
    for c, v in out["corpora"].items():
        print(c, {k: round(v[k], 4) for k in ("bigram", "g", "ref_full_bpb", "c_ratio", "gamma", "A", "bytes_per_token")})


# ------------------------------------------------------------------------------------------------ scoring
def pairwise_acc(p, y):
    i, j = np.triu_indices(len(p), 1)
    return float(np.mean(np.sign(p[i] - p[j]) == np.sign(y[i] - y[j])))


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def load_evals(size, fi):
    # T4 results plus the Mac and Windows CPU workers' files (0.4b); a job evaluated twice must agree to 1e-4 bpb
    files = [os.path.join(RES, f"evals_{size}{x}.jsonl") for x in ("", "_mac", "_win")]
    Y, src = {}, {}
    for fname, line in ((os.path.basename(f), l) for f in files if os.path.exists(f) for l in open(f)):
        r = json.loads(line)
        if not r["sample"].endswith("/heldout"):
            continue
        c = r["sample"].split("/")[0]
        if r["repo"] != fi["runs"][f"{size}/{c}"]["repo"]:
            continue
        t = fi["runs"][f"{size}/{c}"]["main_iter"] if r["revision"] == "main" else int(r["revision"][5:])
        if t in Y.get(c, {}):
            assert abs(Y[c][t] - r["bpb"]) < 1e-4, (c, t, Y[c][t], r["bpb"])
            DUPES.append(dict(corpus=c, iter=t, bpb=[Y[c][t], r["bpb"]], files=[src[(c, t)], fname]))
            continue
        Y.setdefault(c, {})[t] = r["bpb"]
        src[(c, t)] = fname
    return Y


def score_size(size, fi):
    Y = load_evals(size, fi)
    cs = [c for c in CORPORA if c in Y]
    C = fi["corpora"]
    run = {c: fi["runs"][f"{size}/{c}"] for c in cs}
    T = {c: run[c]["main_iter"] for c in cs}
    hr = {c: HRef([dict(context_bytes_exact_mean=x["context_bytes"], bits_per_byte=x["bpb"]) for x in C[c]["ref_rows"]]) for c in cs}
    ad = json.load(open(os.path.join(RES, "design_addendum.json")))      # missing branches (written before scoring)
    pil = {c: tuple(ad["pilots"].get(f"{size}/{c}", PILOTS)) for c in cs}
    p1 = {c: ad["pilot1"].get(f"{size}/{c}", PILOT1) for c in cs}
    Ya = {}
    for c in cs:
        if ANCHOR in Y[c]:
            Ya[c] = Y[c][ANCHOR]
        else:
            t1, t2 = ad["anchor_interp"][f"{size}/{c}"]
            w = np.log(ANCHOR / t1) / np.log(t2 / t1)
            Ya[c] = float((1 - w) * Y[c][t1] + w * Y[c][t2])
    scored = {c: [t for t in SCORE_ITERS if t in Y[c] and t not in pil[c]] for c in cs}

    def Dbytes(c, t):
        return np.asarray(t, float) * run[c]["tokens_per_iter"] * C[c]["bytes_per_token"]

    def S(c, t, lk):
        g, A = C[c]["gamma"], C[c]["A"]
        nu = 1.0 / (1.0 + g)
        logn = nu * (np.log(A * g) - lk * np.log(10.0) + np.log(Dbytes(c, t)))
        return np.interp(logn, hr[c].x, hr[c].y)

    def fitK(train):
        err = np.zeros(len(LOG10K))
        for q in train:
            ts = [t for t in FIT_ITERS if t in Y[q]]
            dL = np.array([Y[q][t] - Ya[q] for t in ts])
            Sg = np.stack([S(q, ts, lk) for lk in LOG10K])            # (nK, nt)
            Sa = np.array([S(q, ANCHOR, lk) for lk in LOG10K])
            err += ((dL[None, :] - (Sg - Sa[:, None])) ** 2).sum(1)
        i = int(np.argmin(err))
        return float(LOG10K[i]), i in (0, len(LOG10K) - 1)

    def ols(x, y):
        X = np.column_stack([np.ones(len(x)), x])
        return np.linalg.lstsq(X, y, rcond=None)[0]

    xs = {lv: {c: (C[c]["ref_full_bpb"] if lv == "ref" else C[c][lv]) for c in cs} for lv in LEVELS}
    pred, folds = {}, {}
    for c in cs:
        tr = [q for q in cs if q != c]
        lk, edge = fitK(tr)
        ts = sorted(Y[c])
        P = {}
        for lv in LEVELS:
            w = ols([xs[lv][q] for q in tr], [Ya[q] for q in tr])
            lev = w[0] + w[1] * xs[lv][c]
            P[f"F0_{lv}"] = {t: float(lev + S(c, t, lk) - S(c, ANCHOR, lk)) for t in ts}
            P[f"flat_{lv}"] = {t: float(lev) for t in ts}
            wf = ols([xs[lv][q] for q in tr], [Y[q][T[q]] for q in tr])
            P[f"Ff_{lv}"] = {T[c]: float(wf[0] + wf[1] * xs[lv][c])}
            folds.setdefault(c, {})[f"level_{lv}"] = dict(anchor=w.tolist(), final=wf.tolist())
        folds[c].update(log10K=lk, K_at_grid_edge=edge)
        for name, (t1, t2) in (("P2", pil[c]), ("P2e", PILOTS_EARLY)):
            if t1 in Y[c] and t2 in Y[c]:
                L1, L2 = Y[c][t1], Y[c][t2]
                a = np.log(L1 / L2) / np.log(t2 / t1) if L2 < L1 else 0.0
                P[name] = {t: float(L2 * (t / t2) ** (-a)) for t in ts}
                folds[c][f"{name}_exponent"] = float(a)
        P["P1"] = {t: float(Y[c][p1[c]] + S(c, t, lk) - S(c, p1[c], lk)) for t in ts}
        pred[c] = P
    arms = sorted({a for c in cs for a in pred[c]})

    def curve_err(a, c):
        if a not in pred[c] or a.startswith("Ff_"):
            return np.nan
        return float(np.sqrt(np.mean([(pred[c][a][t] - Y[c][t]) ** 2 for t in scored[c]])))

    def final_err(a, c):
        if a not in pred[c]:
            return np.nan
        return float(pred[c][a][T[c]] - Y[c][T[c]])

    per = {c: {a: dict(curve_rms=curve_err(a, c), final_signed=final_err(a, c)) for a in arms} for c in cs}
    out = dict(size=size, corpora=cs, measured={c: {str(t): v for t, v in sorted(Y[c].items())} for c in cs},
               predicted={c: {a: {str(t): v for t, v in pred[c][a].items()} for a in pred[c]} for c in cs},
               folds=folds, per_corpus=per, arms={}, scored_iters=scored, pilots=pil, pilot1=p1, anchor_value=Ya)
    rng = np.random.default_rng(SEED)
    for a in arms:
        ce = np.array([per[c][a]["curve_rms"] for c in cs])
        fe = np.array([per[c][a]["final_signed"] for c in cs])
        o = {}
        for metric, v in (("curve", ce), ("final", np.abs(fe))):
            m = np.isfinite(v)
            o[f"{metric}_rms"] = float(np.sqrt(np.mean(v[m] ** 2))) if m.any() else None
            o[f"{metric}_n"] = int(m.sum())
        m = np.isfinite(fe)
        if m.sum() >= 3:
            pf = np.array([pred[c][a][T[c]] for c in np.array(cs)[m]])
            yf = np.array([Y[c][T[c]] for c in np.array(cs)[m]])
            o["final_pairwise_acc"] = pairwise_acc(pf, yf)
            o["final_spearman"] = spearman(pf, yf)
            o["final_mean_signed"] = float(np.mean(fe[m]))
        out["arms"][a] = o
    # paired corpus bootstrap: arm vs baselines
    comp = {}
    draws = rng.integers(0, len(cs), size=(NBOOT, len(cs)))
    for a in arms:
        for b in ("P2", "P2e", "P1"):
            if a == b:
                continue
            for metric in ("curve", "final"):
                va = np.array([per[c][a]["curve_rms" if metric == "curve" else "final_signed"] for c in cs])
                vb = np.array([per[c][b]["curve_rms" if metric == "curve" else "final_signed"] for c in cs])
                m = np.isfinite(va) & np.isfinite(vb)
                if m.sum() < 3:
                    continue
                va, vb = np.abs(va[m]), np.abs(vb[m])
                dr = rng.integers(0, len(va), size=(NBOOT, len(va)))
                rms = lambda v: np.sqrt(np.mean(v ** 2, axis=-1))  # noqa: E731
                diff = rms(va[dr]) - rms(vb[dr])
                ratio = rms(va[dr]) / np.maximum(rms(vb[dr]), 1e-12)
                comp[f"{a}|{b}|{metric}"] = dict(
                    n=int(m.sum()), rms_a=float(rms(va)), rms_b=float(rms(vb)), diff=float(rms(va) - rms(vb)),
                    diff_ci95=[float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
                    ratio=float(rms(va) / rms(vb)), ratio_ci95=[float(np.quantile(ratio, 0.025)), float(np.quantile(ratio, 0.975))],
                    frac_a_better=float(np.mean(diff < 0)), corpora_a_closer=int(np.sum(va < vb)))
    out["paired"] = comp
    del draws
    return out


def score():
    fi = json.load(open(os.path.join(RES, "forecast_inputs.json")))
    res = dict(forecast_inputs_sha256=sha(os.path.join(RES, "forecast_inputs.json")), sizes={})
    for size in (sys.argv[2].split(",") if len(sys.argv) > 2 else SIZES):
        if os.path.exists(os.path.join(RES, f"evals_{size}.jsonl")):
            res["sizes"][size] = score_size(size, fi)
            s = res["sizes"][size]
            print(size, s["corpora"])
            for a, o in s["arms"].items():
                print(f"  {a:12s} curve {o['curve_rms']}  final {o['final_rms']}  acc {o.get('final_pairwise_acc')}")
    res["duplicate_evaluations"] = DUPES
    json.dump(res, open(os.path.join(RES, f"scores_{'_'.join(res['sizes'])}.json"), "w"), indent=1)


def jobs():
    """Job lists for eval_ckpt.py from forecast_inputs.json -> RES/jobs/{0.13b,0.4b,cross_0.13b,cross_0.4b}.txt"""
    fi = json.load(open(os.path.join(RES, "forecast_inputs.json")))
    d = os.path.join(RES, "jobs")
    os.makedirs(d, exist_ok=True)
    for s in SIZES:
        with open(os.path.join(d, f"{s}.txt"), "w") as f:
            for c in CORPORA:
                r = fi["runs"][f"{s}/{c}"]
                for t in r["eval_iters"]:
                    f.write(f"{r['repo']} iter_{t:07d} {c}/heldout\n")
                f.write(f"{r['repo']} main {c}/heldout\n")
        with open(os.path.join(d, f"cross_{s}.txt"), "w") as f:
            for c in CORPORA:
                for e in CORPORA:
                    f.write(f"{fi['runs'][f'{s}/{c}']['repo']} main {e}/cross\n")


if __name__ == "__main__":
    {"inputs": inputs, "score": score, "jobs": jobs}[sys.argv[1]]()
