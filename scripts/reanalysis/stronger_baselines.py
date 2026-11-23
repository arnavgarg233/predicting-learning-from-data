"""Stronger and compute-matched extrapolation baselines for the 80-corpus forecast.

    python scripts/reanalysis/stronger_baselines.py   -> results/reanalysis/stronger_baselines.json

Only existing runs are used. Every constant is fitted on the 13 development corpora (scripts/curve_forecast/develop.py
load_dev), with the development emulation the paper used to choose the power law's floor: pilots at 2^21 and 2^22
(seed 0), target 2^23 (seed mean). Nothing is fitted on the 80 test corpora. Test protocol as score80.json: pilots
2^19 and 2^20 (seed 0), target 2^25 (seed mean); the compute-matched arms also use the seed-0 run at 2^23, which every
test corpus has.

Baselines (defined before any test error was computed):
  A3        power law through two pilots, no floor (the paper's comparison)
  F_ref     power law through two pilots with floor E = k_r * r, r = reference model's full-context bits per byte
  F_ngram   power law through two pilots with floor E = k_g * g, g = best n-gram cross-entropy at 2^23 bytes
  F_pool    L = E + B D^-a with the exponent a pooled from development and E, B through the two pilots
  F_lit     the same form with the literature data exponent a = 0.28 (Chinchilla); added after F_pool's development fit
            ran to its grid edge (the floored exponent is not identified by two-octave development ladders)
  X_pool1   one pilot (2^20) and the zero-floor exponent pooled from development: L = L(2^20) (D/2^20)^-a
  LOG       L linear in log D through the two pilots
Compute-matched (pilots up to 2^23, whose total cost is below the reference pass of the forecast):
  A3_20_23  power law, no floor, through 2^20 and 2^23
  X_pool23  one pilot at 2^23 and the pooled zero-floor exponent
  F_pool_20_23  pooled floored exponent, E and B through 2^20 and 2^23
Reported against A0 (data alone, Qwen2.5-0.5B / SmolLM2-135M; and the 1.5B and 7B Wikipedia references from the
stronger-reference score files), A2 and A4 from score80.json. Floors are capped at 0.9 L(larger pilot) as in core.A3.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/curve_forecast"))
import core as C  # noqa: E402
import develop as DV  # noqa: E402

RES = ROOT / "results/curve_forecast"
OUT = ROOT / "results/reanalysis"
NB = 20000
LN2 = np.log(2.0)
CAP = 0.9
A_LIT = 0.28  # data exponent of Hoffmann et al. 2022 (Chinchilla), L = E + A/N^a + B/D^0.28; added because F_pool's
#               development fit is not identified (its rms keeps falling as a grows; it stops at the grid edge, a = 2)


def pl(pilots, k, E):
    """Power law E + B D^-a through two pilots {k: L}; returns L at 2^k. Falls back to the larger pilot if not falling."""
    (k1, L1), (k2, L2) = sorted(pilots.items())
    E = min(E, CAP * L2)
    if not L2 < L1:
        return float(L2)
    a = np.log((L1 - E) / (L2 - E)) / ((k2 - k1) * LN2)
    return float(E + (L2 - E) * 2.0 ** (-a * (k - k2)))


def pl_fixed_a(pilots, k, a):
    """L = E + B D^-a with a fixed: E and B through two pilots (E capped at 0.9 L2; then B from the larger pilot)."""
    (k1, L1), (k2, L2) = sorted(pilots.items())
    x1, x2 = 2.0 ** (-a * k1), 2.0 ** (-a * k2)
    B = (L1 - L2) / (x1 - x2)
    E = L2 - B * x2
    if E > CAP * L2 or B < 0:
        E = min(max(E, 0.0), CAP * L2) if B >= 0 else L2
        B = (L2 - E) / x2
    return float(E + B * 2.0 ** (-a * k))


def one_pilot(L, kp, k, a):
    return float(L * 2.0 ** (-a * (k - kp)))


def loglin(pilots, k):
    (k1, L1), (k2, L2) = sorted(pilots.items())
    return float(L2 + (L2 - L1) / (k2 - k1) * (k - k2))


def rms(e):
    e = np.asarray(e, float)
    return float(np.sqrt(np.mean(e ** 2)))


def fit_dev():
    dev = DV.load_dev()
    cases = []
    for c, d in dev.items():
        if not all(k in d["R"] and 0 in d["R"][k] for k in (21, 22)) or 23 not in d["L"]:
            continue
        cases.append(dict(c=c, p={21: d["R"][21][0], 22: d["R"][22][0]}, t=d["L"][23], r=d["st"]["ref_full"],
                          g=d["st"]["ngram23"]))
    grid = np.round(np.arange(0.0, 1.5001, 0.01), 4)
    err = lambda f: rms([f(x) - x["t"] for x in cases])  # noqa: E731
    errg = lambda f: rms([f(x) - x["t"] for x in cases if np.isfinite(x["g"])])  # noqa: E731
    k_r = float(grid[np.argmin([err(lambda x, g=g: pl(x["p"], 23, g * x["r"])) for g in grid])])
    k_g = float(grid[np.argmin([errg(lambda x, g=g: pl(x["p"], 23, g * x["g"])) for g in grid])])
    agrid = np.round(np.arange(0.02, 2.0001, 0.01), 4)
    a_f = float(agrid[np.argmin([err(lambda x, a=a: pl_fixed_a(x["p"], 23, a)) for a in agrid])])
    # zero-floor exponent pooled over development: least squares in log loss over every dev corpus's 2^21-2^23 runs
    slopes = []
    for x in cases:
        slopes.append(np.log(x["p"][21] / x["t"]) / (2 * LN2))
    a0_med = float(np.median(slopes))
    a0_ls = float(agrid[np.argmin([err(lambda x, a=a: one_pilot(x["p"][22], 22, 23, a)) for a in agrid])])
    dev_rms = dict(A3=err(lambda x: pl(x["p"], 23, 0.0)), F_ref=err(lambda x: pl(x["p"], 23, k_r * x["r"])),
                   F_ngram=errg(lambda x: pl(x["p"], 23, k_g * x["g"])), F_pool=err(lambda x: pl_fixed_a(x["p"], 23, a_f)),
                   X_pool1=err(lambda x: one_pilot(x["p"][22], 22, 23, a0_ls)), LOG=err(lambda x: loglin(x["p"], 23)))
    # far-geometry check on the two development corpora with long ladders (reported only)
    far = {}
    fw = DV.runs(str(ROOT / "results/text_runs/cf_fineweb/bytes_b*_s*.json"))
    fwe = dev["fineweb_edu"]
    if 19 in fw and 20 in fw and 25 in fwe["L"]:
        p = {19: fw[19][0], 20: fw[20][0]}
        far["fineweb_edu_19_20_to_25"] = dict(
            measured=fwe["L"][25], A3=pl(p, 25, 0.0), F_ref=pl(p, 25, k_r * fwe["st"]["ref_full"]),
            F_ngram=pl(p, 25, k_g * fwe["st"]["ngram23"]), F_pool=pl_fixed_a(p, 25, a_f),
            X_pool1=one_pilot(p[20], 20, 25, a0_ls), LOG=loglin(p, 25))
    return dict(k_r=k_r, k_g=k_g, a_floored=a_f, a_zero_ls=a0_ls, a_zero_median=a0_med, n_dev=len(cases),
                dev_rms_2_23=dev_rms, far_check=far, dev_corpora=[x["c"] for x in cases])


def main():
    K = fit_dev()
    s80 = json.load(open(RES / "score80.json"))["per_corpus"]
    F1 = json.load(open(RES / "forecast.json"))["corpora"]
    F2 = json.load(open(RES / "stage2/forecast.json"))["corpora"]
    FC = {**F1, **F2}
    sr = {}
    for tag, path in (("A0_1.5B", RES / "stronger_ref/score.json"),
                      ("A0_7B", RES / "stronger_ref7b/score.json")):
        arm = "sub15" if tag == "A0_1.5B" else "sub7"
        pc = json.load(open(path))["arms"][arm]["per_corpus"]
        sr[tag] = {k.split(":", 1)[1]: v["err_A0"]["25"] for k, v in pc.items() if k.startswith("test80:")}
    names = sorted(s80)
    per = {}
    for c in names:
        p = s80[c]
        st = FC[c]["ref_stats"]
        L19, L20 = p["pilots"]["19"], p["pilots"]["20"]
        L23 = p["measured"]["23"]
        assert p["seeds"]["23"] == [0]
        y = p["measured"]["25"]
        two = {19: L19, 20: L20}
        far = {20: L20, 23: L23}
        g = st.get("ngram23", np.nan)
        pred = dict(
            A0=p["forecast"]["25"]["A0"], A2=p["forecast"]["25"]["A2"], A4=p["forecast"]["25"]["A4"],
            A3=pl(two, 25, 0.0),
            F_ref=pl(two, 25, K["k_r"] * st["ref_full"]),
            F_ngram=pl(two, 25, K["k_g"] * g) if np.isfinite(g) else np.nan,
            F_pool=pl_fixed_a(two, 25, K["a_floored"]),
            F_lit=pl_fixed_a(two, 25, A_LIT),
            X_pool1=one_pilot(L20, 20, 25, K["a_zero_ls"]),
            LOG=loglin(two, 25),
            A3_20_23=pl(far, 25, 0.0),
            X_pool23=one_pilot(L23, 23, 25, K["a_zero_ls"]),
            F_pool_20_23=pl_fixed_a(far, 25, K["a_floored"]))
        err = {a: v - y for a, v in pred.items()}
        for tag in sr:
            if c in sr[tag]:
                err[tag] = sr[tag][c]
        assert abs(err["A3"] - p["err"]["25"]["A3"]) < 1e-9, c
        fl = p["flops"]
        flops = dict(ref_pass=fl["A0"], pilots_19_20=fl["A3"], pilot_20=fl["A1"], run_23=fl["target_2_23"],
                     run_25=fl["target_2_25"])
        flops["pilots_20_23"] = flops["pilot_20"] + flops["run_23"]
        per[c] = dict(family=p["family"], ref_model=p["ref_model"], measured_25=y, err=err, flops=flops)
    arms = list(per[names[0]]["err"])
    fams = {"code": [c for c in names if per[c]["family"] == "code"],
            "english": [c for c in names if per[c]["family"] == "english"],
            "wiki": [c for c in names if per[c]["family"] == "wiki"], "all": names}
    rng = np.random.default_rng(0)
    table, ratios = {}, {}
    for f, cs in fams.items():
        table[f] = dict(n=len(cs))
        for a in arms:
            e = [per[c]["err"].get(a, np.nan) for c in cs]
            e = np.array(e, float)
            table[f][a] = dict(rms=rms(e[np.isfinite(e)]), n=int(np.isfinite(e).sum()), bias=float(np.nanmean(e)))
        ratios[f] = {}
        idx = rng.integers(0, len(cs), (NB, len(cs)))
        for num in arms:
            for den in ("A0", "A3", "A0_7B"):
                if num == den:
                    continue
                en = np.array([per[c]["err"].get(num, np.nan) for c in cs], float)
                ed = np.array([per[c]["err"].get(den, np.nan) for c in cs], float)
                ok = np.isfinite(en) & np.isfinite(ed)
                if ok.sum() < 3:
                    continue
                en, ed = en[ok], ed[ok]
                ii = idx if ok.all() else np.random.default_rng(1).integers(0, ok.sum(), (NB, ok.sum()))
                r = np.sqrt((en[ii] ** 2).mean(1) / (ed[ii] ** 2).mean(1))
                ratios[f][f"{num}/{den}"] = dict(ratio=rms(en) / rms(ed), ci95=[float(np.percentile(r, 2.5)),
                                                                               float(np.percentile(r, 97.5))],
                                                 closer_count=f"{int(np.sum(np.abs(en) < np.abs(ed)))}/{int(ok.sum())}")
    comp = {}
    for f, cs in fams.items():
        comp[f] = {k: float(np.median([per[c]["flops"][k] for c in cs])) for k in per[names[0]]["flops"]}
    for ref in ("SmolLM2-135M", "Qwen2.5-0.5B"):
        cs = [c for c in names if per[c]["ref_model"] == ref]
        comp[f"by_ref:{ref}"] = {k: float(np.median([per[c]["flops"][k] for c in cs])) for k in per[names[0]]["flops"]}
        comp[f"by_ref:{ref}"]["n"] = len(cs)
    comp["share_ref_pass_above_pilots_20_23"] = float(np.mean([per[c]["flops"]["ref_pass"] > per[c]["flops"]["pilots_20_23"]
                                                                for c in names]))
    comp["share_ref_pass_above_run_25"] = float(np.mean([per[c]["flops"]["ref_pass"] > per[c]["flops"]["run_25"] for c in names]))
    res = dict(constants=K, rms_2_25=table, ratios_2_25=ratios, flops_median=comp, per_corpus=per)
    json.dump(res, open(OUT / "stronger_baselines.json", "w"), indent=1, default=float)
    print(json.dumps(K, indent=1))
    print(f"{'arm':14s}" + "".join(f"{f:>10s}" for f in fams))
    for a in arms:
        print(f"{a:14s}" + "".join(f"{table[f][a]['rms']:10.3f}" for f in fams))
    for f in fams:
        print(f"\n[{f}]")
        for k, v in ratios[f].items():
            if k.endswith("/A0") or k.endswith("/A3"):
                print(f"  {k:18s} {v['ratio']:.2f} [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}] closer {v['closer_count']}")
    print(json.dumps(comp, indent=1))


if __name__ == "__main__":
    main()
