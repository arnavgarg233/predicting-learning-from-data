"""Score the RHM test: measured data exponents against the information law
and the formula of Cagnetta et al.; the evidence price at learning; per-level learning diagnostics.
Writes results/rhm/analysis.json and analysis.txt."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from exact_sources import zoo  # noqa: E402
from rhm import RHM  # noqa: E402
from make_data import SETTINGS  # noqa: E402

OUT = ROOT / "results/rhm"
RNG = np.random.default_rng(0)
BITS_PER_EVENT = zoo.H_EPS01


def slope(P, y):
    return zoo.slope(np.asarray(P, float), np.asarray(y, float))


def tail_K(delta, T):
    """tail(K) = sum_{n > K} delta(n) (T - n)/T for K = 0..T-1 (the law's unlearned excess with horizon K)."""
    n = np.arange(1, T)
    w = delta * (T - n) / T
    return np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])  # index K -> sum over n >= K+1


def horizon(tail, E):
    """Largest K with tail(K) >= E, interpolated log-linearly between integers."""
    if E >= tail[0]:
        return 0.0
    i = int(np.argmax(tail < E))  # first K with tail < E
    a, b = np.log(tail[i - 1]), np.log(max(tail[i], 1e-300))
    return float(i - 1 + (a - np.log(E)) / (a - b)) if a > b else float(i)


def fit_price(P, E, h, T):
    """Constant price B (bits per distance) and free level c: log E ~ log c + log tail_info(P; B); grid search."""
    best = (np.inf, None, None)
    for lb in np.linspace(np.log(5), np.log(5e6), 400):
        tl = zoo.tail_curves(h, np.ones(T), np.ones(T), T, P, Bstar=np.exp(lb))["info"]
        if np.any(tl <= 0):
            continue
        r = np.log(E) - np.log(tl)
        c = r.mean()
        sse = float(((r - c) ** 2).sum())
        if sse < best[0]:
            best = (sse, float(np.exp(lb)), float(np.exp(c)))
    return dict(B_bits=best[1], events=None if best[1] is None else best[1] / BITS_PER_EVENT, level=best[2],
                rms_log=None if best[1] is None else float(np.sqrt(best[0] / len(P))))


def attainable_info(name, pr, src, v, s, L):
    """Attainable excess per position on the test windows (exact model against the exact position-only model). Computed from
    the data file when present (data/rhm/<name>.npz, written by make_data.py, 12-44 MB, not shipped); otherwise
    read from attainable_info.json, written from the same files by scripts/rhm/extract_attainable.py."""
    path = ROOT / pr["data"]
    if not path.exists():
        return np.array(json.loads((OUT / "attainable_info.json").read_text())[name], dtype=np.float64)
    p = np.load(path)["S_test_p"].astype(np.float64)
    pi = {L: np.full((1, v), 1.0 / v)}
    for k in range(L, 0, -1):
        pi[k - 1] = np.stack([pi[k] @ src.M[k][c] for c in range(s)], 1).reshape(-1, v)
    marg = pi[0]  # (T, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, p * (np.log2(p) - np.log2(marg[None])), 0).sum(-1).mean(0)  # attainable, per position


def main():
    res = {}
    lines = []
    for name in SETTINGS:
        pf = OUT / f"pred/{name}.json"
        if not pf.exists():
            continue
        pr = json.loads(pf.read_text())
        T = pr["T"]
        runs = {}
        for f in sorted((OUT / f"runs/{name}").glob("S_n*_s*.json")):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            runs[(d["nwin"], d["seed"])] = d
        nws = sorted({k[0] for k in runs})
        seeds = sorted({k[1] for k in runs})
        full = [n for n in nws if all((n, s) in runs for s in seeds)]
        if len(full) < 3:
            continue
        P = np.array(full) * T
        E = np.array([[runs[(n, s)]["test_kl_bits"] for n in full] for s in seeds])  # seeds x budgets
        Em = E.mean(0)
        out = dict(budgets=P.tolist(), seeds=seeds, excess=E.tolist(), excess_mean=Em.tolist(),
                   steps=[[runs[(n, s)]["steps"] for n in full] for s in seeds],
                   hit_cap=[[runs[(n, s)]["steps"] >= 12000 for n in full] for s in seeds],
                   true_nll=[[runs[(n, s)]["test_true_nll_bits"] for n in full] for s in seeds],
                   entropy_rate_exact=pr["entropy_rate_exact"])
        for tag, sel in (("all", np.ones(len(P), bool)), ("from14", P >= 2 ** 14)):
            a = slope(P[sel], Em[sel])
            boots = []
            for _ in range(4000):
                idx = RNG.integers(0, len(seeds), len(P))
                boots.append(slope(P[sel], E[idx, np.arange(len(P))][sel]))
            boots = np.array(boots)
            out[f"alpha_{tag}"] = a
            out[f"alpha_{tag}_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
            out[f"alpha_{tag}_per_seed"] = [slope(P[sel], row[sel]) for row in E]
        preds = dict(law_closed=pr["pred_law_closed"], law_finite_210=pr["pred_law_finite_210"],
                     cagnetta=pr["pred_cagnetta"])
        out["pred"] = preds
        out["pred_ci"] = dict(law_closed=pr["pred_law_closed_ci"], law_finite_210=pr["pred_law_finite_210_ci"],
                              cagnetta=pr["pred_cagnetta_ci"])
        out["pred_law_finite_210_from14"] = pr["pred_law_finite_210_fit14"]
        out["error"] = {k: v - out["alpha_all"] for k, v in preds.items()}
        out["error_from14"] = {k: v - out["alpha_from14"] for k, v in preds.items()}
        out["error_ci"] = {k: [v - out["alpha_all_ci"][1], v - out["alpha_all_ci"][0]] for k, v in preds.items()}
        out["gamma"], out["beta"] = pr["gamma"], pr["beta"]
        # price at learning: horizon from the measured excess, evidence at that distance
        h = np.array(pr["h_n"])
        delta = np.clip(h[:-1] - h[1:], 0, None)
        tl = tail_K(delta, T)
        hor, ev = [], []
        for Pi, Ei in zip(P, Em):
            K = horizon(tl, Ei)
            hor.append(K)
            k = int(round(K))
            if 1 <= k <= T - 1:
                ev.append(float(Pi * delta[k - 1] * (T - k) / T / BITS_PER_EVENT))
            else:
                ev.append(None)
        out["horizon"] = hor
        out["events_at_horizon"] = ev
        out["law_tail_210_at_budgets"] = zoo.tail_curves(h, np.ones(T), np.ones(T), T, P)["info"].tolist()
        out["const_price_fit"] = fit_price(P, Em, h, T)
        hp = np.array(hor) > 0
        out["horizon_exponent"] = -slope(P[hp], np.array(hor)[hp]) if hp.sum() >= 3 else None
        # per-level learning: positions by the number of trailing zero bits l (t = j 2^l, j odd): the nearest shared
        # ancestor with x_{t-1} is at level l+1
        v, m, s, L, gseed = SETTINGS[name]
        src = RHM(v, m, s, L, gseed)
        I_t = attainable_info(name, pr, src, v, s, L)
        lev = np.array([0 if t == 0 else (t & -t).bit_length() - 1 for t in range(T)])  # trailing zeros; t=0 separate
        lev[0] = L
        out["marginal_baseline_excess"] = float(I_t.mean())  # excess of the exact position-only (unigram) model
        out["level_info_bits"] = {int(l): float(I_t[lev == l].mean()) for l in range(L)}
        frac = {}
        for (n, sd), r in runs.items():
            kp = np.array(r["kl_by_pos"])
            frac[f"{n}_{sd}"] = {int(l): float(1 - kp[lev == l].mean() / I_t[lev == l].mean()) for l in range(L)}
        out["level_learned_fraction"] = frac
        res[name] = out
        lines.append(f"{name}: gamma {pr['gamma']:.3f} beta {pr['beta']:.3f} | law {preds['law_closed']:.3f} "
                     f"law210 {preds['law_finite_210']:.3f} cag {preds['cagnetta']:.3f} | measured {out['alpha_all']:.3f} "
                     f"({out['alpha_all_ci'][0]:.3f}-{out['alpha_all_ci'][1]:.3f}) per-seed "
                     f"{[round(x, 3) for x in out['alpha_all_per_seed']]} from14 {out['alpha_from14']:.3f} | "
                     f"err law {out['error']['law_closed']:+.3f} law210 {out['error']['law_finite_210']:+.3f} "
                     f"cag {out['error']['cagnetta']:+.3f}")
        lines.append(f"   excess mean {[f'{x:.4f}' for x in Em]} budgets {[int(np.log2(x)) for x in P]}; "
                     f"true nll {np.mean(out['true_nll']):.4f} vs rate {pr['entropy_rate_exact']:.4f}; steps {out['steps']}")
        lines.append(f"   horizon {[round(x, 1) for x in hor]} events@horizon {[None if e is None else round(e) for e in ev]}; "
                     f"const price {out['const_price_fit']['events']:.0f} events (rms log {out['const_price_fit']['rms_log']:.3f})")
        lines.append(f"   position-only baseline excess {I_t.mean():.4f}; from14 errors " + " ".join(
            f"{k} {v_:+.3f}" for k, v_ in out["error_from14"].items()))
        lines.append("   level info bits " + " ".join(f"l{k}:{v_:.3f}" for k, v_ in out["level_info_bits"].items()))
        for n in full:
            fr = np.mean([[frac[f'{n}_{sd}'][l] for l in range(L)] for sd in seeds], 0)
            lines.append(f"   learned fraction by level at 2^{int(np.log2(n * T))}: " + " ".join(f"{x:.2f}" for x in fr))
    if res:
        summ = {}
        for k in ("law_closed", "law_finite_210", "cagnetta"):
            e = np.array([r["error"][k] for r in res.values()])
            summ[k] = dict(rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), n=len(e))
        res["_summary"] = summ
        summ14 = {}
        for k in ("law_closed", "law_finite_210", "cagnetta"):
            e = np.array([r["error_from14"][k] for n_, r in res.items() if not n_.startswith("_")])
            summ14[k] = dict(rms=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), n=len(e))
        res["_summary_from14"] = summ14
        lines.append("r.m.s. error, fit 2^14-2^20: " + ", ".join(f"{k} {v['rms']:.3f} (bias {v['bias']:+.3f})"
                                                              for k, v in summ14.items()))
        lines.append("r.m.s. error over settings: " + ", ".join(f"{k} {v['rms']:.3f} (bias {v['bias']:+.3f})"
                                                              for k, v in res["_summary"].items()))
    (OUT / "analysis.json").write_text(json.dumps(res, indent=1))
    (OUT / "analysis.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
