"""External calibration of the frozen open-sci-ref forecast F_ext.

    python3 scripts/opensciref_forecast/calibrate.py   # -> results/opensciref_forecast/calibration.json

Reads only DataDecide files (perplexity table, evaluation-set statistics and SmolLM2-135M curves). No open-sci-ref
learner number is read. Size matching: open-sci-ref 0.13b <- DataDecide 150M, 0.4b <- 530M.
"""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "opensciref"))
from forecast import HRef, LOG10K  # noqa: E402

PPL = os.path.join(ROOT, "data/datadecide/gt/ppl_results.parquet")  # DataDecide perplexity table, not shipped (data/manifest.json)
ST = os.path.join(ROOT, "results/datadecide/stats")
OUT = os.path.join(ROOT, "results/opensciref_forecast/calibration.json")
CELLS = [("C4", "rc_c4", "c4_en")] + [("Dolma1.7", "rc_dolma1_7", e) for e in
                                      ("dolma_wiki", "dolma_stack", "dolma_reddit", "dolma_pes2o",
                                       "dolma_common-crawl", "dolma_books")]
SIZE = {"0.13b": "150M", "0.4b": "530M"}
TOKENS_PER_STEP = {"150M": 393216, "530M": 917504}   # scripts/datadecide_targets.py
SAVE_EVERY = 1250
LN2 = np.log(2.0)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def href(e):
    rows = json.load(open(os.path.join(ST, f"ev_{e}_ref_SmolLM2-135M.json")))["paired"]["rows"]
    return HRef(rows)


def dd_curve(d, size, recipe, e, bpt):
    """Mean over complete seeds of bits/byte on set e; each seed's last step is mapped to T (dd_curve_forecast)."""
    x = d[(d.params == size) & (d.data == recipe)].copy()
    T = float(d[d.params == size]["step"].max())
    last = x.groupby("seed")["step"].transform("max")
    x = x[last >= 0.97 * T].copy()
    is_last = x["step"] == x.groupby("seed")["step"].transform("max")
    x = x[((x["step"] % SAVE_EVERY == 0) & (x["step"] < T)) | is_last].copy()
    x.loc[is_last, "step"] = T
    x["bpb"] = np.log(x[f"eval/{e}-validation/Perplexity"]) / LN2 / bpt
    g = x.groupby("step")["bpb"].mean()
    return g.index.values.astype(float), g.values, T, int(x["seed"].nunique())


def main():
    d = pd.read_parquet(PPL)
    src = json.load(open(os.path.join(ST, "merged.json")))["sources"]
    out = dict(cells=[c[0] + "|" + c[2] for c in CELLS], size_match=SIZE, sizes={},
               inputs_sha256={os.path.relpath(p, ROOT) if p.startswith(ROOT) else p: sha(p) for p in
                              [PPL, os.path.join(ST, "merged.json")] +
                              [os.path.join(ST, f"ev_{e}_ref_SmolLM2-135M.json") for _, _, e in CELLS]})
    H = {e: href(e) for _, _, e in CELLS}
    for osr, size in SIZE.items():
        tps = TOKENS_PER_STEP[size]
        rows, err = [], np.zeros(len(LOG10K))
        for recipe, rc, e in CELLS:
            bpt_e = src[f"ev_{e}"]["bytes_per_token"]
            b_r = src[rc]["bytes_per_token"]
            steps, y, T, nseed = dd_curve(d, size, recipe, e, bpt_e)
            r = float(H[e].y[-1])
            gam, A = H[e].power_fit()
            logD = np.log(steps * tps * b_r)
            m = steps >= T / 4
            nu = 1.0 / (1.0 + gam)
            logn = nu * (np.log(A * gam) - LOG10K[:, None] * np.log(10.0) + logD[None, m])
            Sg = np.interp(logn, H[e].x, H[e].y)                       # (nK, n_steps)
            dS = Sg - Sg[:, -1:]
            dL = y[m] - y[m][-1]
            err += ((dL[None, :] - dS) ** 2).sum(1)
            rows.append(dict(recipe=recipe, eval=e, r=r, gamma=gam, A=A, L_final=float(y[-1]), T_step=T, n_seeds=nseed,
                             n_K_steps=int(m.sum()), bpt_eval=bpt_e, bpt_recipe=b_r))
        x = np.array([q["r"] for q in rows])
        yL = np.array([q["L_final"] for q in rows])
        a, b = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), yL, rcond=None)[0]
        k = int(np.argmin(err))
        T = rows[0]["T_step"]
        bbar = float(np.mean([q["bpt_recipe"] for q in rows]))
        res = yL - (a + b * x)
        out["sizes"][osr] = dict(dd_size=size, a=float(a), b=float(b), map_resid_rms=float(np.sqrt(np.mean(res ** 2))),
                                 map_pearson=float(np.corrcoef(x, yL)[0, 1]), log10K=float(LOG10K[k]),
                                 K_at_grid_edge=bool(k in (0, len(LOG10K) - 1)), T_step=T, tokens_per_step=tps,
                                 D_cal_bytes=float(T * tps * bbar), mean_recipe_bpt=bbar, cells=rows)
        print(osr, size, f"a={a:.3f} b={b:.3f} resid={out['sizes'][osr]['map_resid_rms']:.3f} log10K={LOG10K[k]:.2f}",
              f"D_cal={T * tps * bbar:.3e}")
        for q in rows:
            print(f"   {q['recipe']:9s} {q['eval']:20s} r={q['r']:.3f} L={q['L_final']:.3f} gamma={q['gamma']:.3f} seeds={q['n_seeds']}")
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote", OUT, sha(OUT))


if __name__ == "__main__":
    main()
