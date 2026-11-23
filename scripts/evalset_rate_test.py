"""Does gamma of each evaluation text's information curve order DataDecide's 60M-1B exponents across evaluation sets?
Writes results/datadecide/evalset_rate_test.json."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DD = ROOT / "results" / "datadecide"
S = json.loads((DD / "stats" / "merged.json").read_text())["sources"]
lad = pd.read_csv(DD / "ladder_rates.csv")
inrun = pd.read_csv(DD / "inrun_rates.csv")
EVALS = sorted(lad["eval"].unique())
rng = np.random.default_rng(0)


def rank(x):
    return pd.Series(x).rank().to_numpy()


def spearman(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def perm_p(x, y, sign, n=200_000):
    r0 = sign * spearman(x, y)
    rx, ry = rank(x), rank(y)
    cnt = 0
    for _ in range(n // 1000):
        P = np.array([rng.permutation(ry) for _ in range(1000)])
        r = np.array([np.corrcoef(rx, p)[0, 1] for p in P]) * sign
        cnt += int((r >= r0 - 1e-12).sum())
    return (cnt + 1) / (n + 1)


def stat(model, key):
    out = []
    for e in EVALS:
        s = S[f"ev_{e}"]
        if key == "gamma":
            out.append(s["ref"][model]["gamma"]["4-64"]["gamma"])
        elif key == "lzma":
            out.append(s["compression"]["lzma_bpb_8MB"])
        elif key == "beta":
            out.append(s["beta"]["beta"]["tok_1_200_floorcorr"])
    return np.array(out, float)


y = lad.groupby("eval")["alpha_large"].mean().reindex(EVALS).to_numpy()
fams = sorted(lad["family"].unique())
boot_y = []
for _ in range(2000):
    pick = rng.choice(fams, size=len(fams), replace=True)
    sub = pd.concat([lad[lad["family"] == f] for f in pick])
    boot_y.append(sub.groupby("eval")["alpha_large"].mean().reindex(EVALS).to_numpy())
boot_y = np.array(boot_y)

res = dict(evals=EVALS, target_mean_alpha_large=y.tolist(), models={})
for model in ("gpt2", "pythia-160m", "SmolLM2-135M"):
    g = stat(model, "gamma")
    preds = {"ours_gamma": (g, +1), "rival_lzma": (stat(model, "lzma"), -1),
             "rival_cagnetta_gamma_over_2beta": (g / (2 * stat(model, "beta")), +1)}
    M = {"gamma": g.tolist(), "info_rule_gamma_over_1pgamma": (g / (1 + g)).tolist()}
    for k, (x, sgn) in preds.items():
        b = np.array([sgn * spearman(x, yy) for yy in boot_y])
        M[k] = dict(signed_spearman=sgn * spearman(x, y), p_one_sided=perm_p(x, y, sgn),
                    boot95=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    for k in ("rival_lzma", "rival_cagnetta_gamma_over_2beta"):
        d = np.array([preds["ours_gamma"][1] * spearman(preds["ours_gamma"][0], yy) - preds[k][1] * spearman(preds[k][0], yy)
                      for yy in boot_y])
        M[f"diff_vs_{k}"] = dict(point=M["ours_gamma"]["signed_spearman"] - M[k]["signed_spearman"],
                                 boot95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    # secondary: within-run exponents at fixed size
    sec = {}
    for size in ("530M", "1B"):
        ir = inrun[inrun["params"] == size].groupby("eval")["alpha"].mean().reindex(EVALS).to_numpy()
        gr = g / (1 + g)
        sec[size] = dict(mean_inrun_alpha=ir.tolist(), spearman_gamma=spearman(g, ir),
                         ratio_measured_over_rule_median=float(np.nanmedian(ir / gr)))
    M["secondary_inrun"] = sec
    res["models"][model] = M
p = DD / "evalset_rate_test.json"
p.write_text(json.dumps(res, indent=1))
for model, M in res["models"].items():
    print(model, {k: (round(v["signed_spearman"], 3), round(v["p_one_sided"], 4), [round(x, 2) for x in v["boot95"]])
                  for k, v in M.items() if isinstance(v, dict) and "signed_spearman" in v})
    print("   diffs", {k: (round(v["point"], 3), [round(x, 2) for x in v["boot95"]]) for k, v in M.items() if k.startswith("diff")})
    print("   inrun", {s: (round(v["spearman_gamma"], 3), round(v["ratio_measured_over_rule_median"], 3)) for s, v in M["secondary_inrun"].items()})
