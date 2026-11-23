"""Does the Qwen2.5-1.5B information curve give the pilot-anchored forecasts (A2, A4) a better SHAPE than the 0.5B curve?

    python3 scripts/stronger_ref/shape_test.py   # -> results/curve_forecast/stronger_ref/shape_test.json
    python3 scripts/stronger_ref/shape_test.py --sub7   # adds the Qwen2.5-7B arm -> stronger_ref7b/shape_test.json
    python3 scripts/stronger_ref/shape_test.py --sub14  # adds the 7B and 14B arms -> stronger_ref14b/shape_test.json

Plain numpy, no model. Re-scores A2 (one pilot read on the curve, locked form 'neff') and A4 (two pilots + shape, locked
form 'off') on every Qwen-referenced test corpus (Wikipedia-47 and stage-3-16) with three curves: full05 (stored 0.5B,
20,000 targets; must reproduce the stored A2/A4), sub05 (0.5B on the 1,000-target subset) and sub15 (1.5B on the same
subset). Nothing is refitted: the locked shape 'gamma' takes (gamma, A) from each curve's own power fit
(forecast_input_variants.power_fit, as in forecast.py) and the locked K_ts; pilots, measured losses and A3 are read from
the stored score files. Bootstrap ratios use scripts/stronger_ref/score.py's paired_boot (20,000 draws, rng seed 0 per budget).
The diagnosis compares the curves' local slopes with the learner: the learner's loss read back onto each curve gives an
effective context n_eff(D) = h^-1(L(D)); the shape is right when log n_eff grows as nu log D.
"""
import hashlib
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sr_score", os.path.join(HERE, "score.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)
C, RES, OUT = S.C, S.RES, S.OUT
ARMS = S.ARMS
LN2 = np.log(2.0)
GROUPS = {"wiki47": ("test80", "score80.json"), "stage3_16": ("stage3", "stage3/score_stage3.json")}


def pilots_and_stored():
    out = {}
    for g, (grp, sf) in GROUPS.items():
        for c, p in json.load(open(os.path.join(RES, sf)))["per_corpus"].items():
            if p["ref_model"] != "Qwen2.5-0.5B":
                continue
            out[f"{grp}:{c}"] = dict(name=c, group=g, family=p["family"], L19=p["pilots"]["19"], L20=p["pilots"]["20"],
                                     measured=p["measured"], stored={k: p["forecast"][k] for k in ("23", "25")})
    return out


def arm_forecasts(hr, t, ch, K):
    st = C.ref_stats(hr)
    nu, n0 = C.shape_params(ch["shape"], st, K)
    f = {}
    for k in (23, 25):
        f[str(k)] = dict(A2=C.hybrid(hr, nu, n0, {20: t["L20"]}, k, ch["A2_form"]),
                         A4=C.hybrid(hr, nu, n0, {19: t["L19"], 20: t["L20"]}, k, ch["A4_form"]))
    # diagnostics on the shape
    L = {19: t["L19"], 20: t["L20"], 23: t["measured"]["23"], 25: t["measured"]["25"]}
    hmin, hmax = float(hr.y.min()), float(hr.y.max())
    lneff = {k: float(np.log(hr.neff(v))) for k, v in L.items()}
    inside = {str(k): bool(hmin < v < hmax) for k, v in L.items()}
    below = {str(k): bool(v <= hmin) for k, v in L.items()}
    # local slope of h per doubling of context, at the shape's effective contexts for 2^20 and 2^25 and at the pilot's n_eff
    sl = lambda n: float(-np.interp(np.log(n), 0.5 * (hr.x[1:] + hr.x[:-1]), np.diff(hr.y) / np.diff(hr.x)) * LN2)  # noqa: E731
    diag = dict(gamma=st["gamma"], A=st["A"], nu=nu, n0=n0, ref_full=st["ref_full"], h_min=hmin,
                L_inside_curve=inside, L_below_curve_min=below,
                n_eff_bytes={str(k): float(np.exp(v)) for k, v in lneff.items()},
                # implied exponent of the learner on this curve, d log n_eff / d log D, 2^20 -> 2^25 (only if both inside)
                implied_nu_20_25=(lneff[25] - lneff[20]) / (5 * LN2) if inside["20"] and inside["25"] else None,
                shape_ctx_bytes={str(k): float(n0 * 2.0 ** (k * nu)) for k in (19, 20, 23, 25)},
                h_slope_per_doubling_at_shape_ctx={str(k): sl(n0 * 2.0 ** (k * nu)) for k in (20, 25)},
                h_slope_per_doubling_at_pilot_neff=sl(np.exp(lneff[20])),
                drop_20_25=dict(measured=L[20] - L[25], A2=L[20] - f["25"]["A2"],
                                S=C.S(hr, nu, n0, 20) - C.S(hr, nu, n0, 25)),
                drop_19_20=dict(measured=L[19] - L[20], S=C.S(hr, nu, n0, 19) - C.S(hr, nu, n0, 20)))
    return f, diag


def main():
    lk = json.load(open(os.path.join(RES, "locked.json")))
    ch, K = lk["chosen"], lk["constants"]
    jobs, colab, curves = S.load_inputs()
    T = S.test_corpora()
    P = pilots_and_stored()
    per = {}
    for key, t in P.items():
        grp, c = key.split(":", 1)
        tc = T[(grp, c)]
        assert tc["ref_model"] == "Qwen2.5-0.5B" and c in jobs, key
        row = dict(name=c, group=t["group"], measured=t["measured"], pilots={"19": t["L19"], "20": t["L20"]},
                   A3={k: tc["A3"][k] for k in ("23", "25")}, stored_A2={k: t["stored"][k]["A2"] for k in ("23", "25")},
                   stored_A4={k: t["stored"][k]["A4"] for k in ("23", "25")}, arms={})
        for arm in ARMS:
            if c not in curves[arm]:
                continue
            hr = C.FN.HRef(tc["refcurve"], "Qwen2.5-0.5B") if arm == "full05" else curves[arm][c]
            f, dg = arm_forecasts(hr, t, ch, K)
            row["arms"][arm] = dict(forecast=f, err={k: {a: f[k][a] - t["measured"][k] for a in ("A2", "A4")} for k in f}, diag=dg)
        row["err_A3"] = {k: tc["A3"][k] - t["measured"][k] for k in ("23", "25")}
        per[key] = row
    keys_all = [x for x in per if all(a in per[x]["arms"] for a in ARMS)]
    assert len(keys_all) == len(per) == 63, (len(keys_all), len(per))
    repro = max(abs(per[x]["arms"]["full05"]["forecast"][k][a] - per[x][f"stored_{a}"][k])
                for x in per for k in ("23", "25") for a in ("A2", "A4"))

    summ = {}
    for k in ("23", "25"):
        rng = np.random.default_rng(0)
        rng7 = np.random.default_rng(0)   # separate stream for the sub7 comparisons (others unchanged)
        rng14 = np.random.default_rng(0)  # third stream for the sub14 comparisons
        sk = {}
        for g in GROUPS:
            ks = sorted(x for x in keys_all if per[x]["group"] == g)
            E3 = np.array([per[x]["err_A3"][k] for x in ks])
            blk = dict(n=len(ks), rms_A3=S.rms(E3), mean_err_A3=float(E3.mean()))
            for a in ("A2", "A4"):
                E = {arm: np.array([per[x]["arms"][arm]["err"][k][a] for x in ks]) for arm in ARMS}
                blk[a] = dict(rms={arm: S.rms(e) for arm, e in E.items()}, mean_err={arm: float(e.mean()) for arm, e in E.items()},
                              sub15_vs_sub05=S.paired_boot(E["sub15"], E["sub05"], rng),
                              sub15_vs_A3=S.paired_boot(E["sub15"], E3, rng),
                              n_closer_than_A3={arm: int(np.sum(np.abs(e) < np.abs(E3))) for arm, e in E.items()})
                if "sub7" in E:
                    blk[a].update(sub7_vs_sub05=S.paired_boot(E["sub7"], E["sub05"], rng7),
                                  sub7_vs_sub15=S.paired_boot(E["sub7"], E["sub15"], rng7),
                                  sub7_vs_A3=S.paired_boot(E["sub7"], E3, rng7))
                if "sub14" in E:
                    blk[a].update(sub14_vs_sub7=S.paired_boot(E["sub14"], E["sub7"], rng14),
                                  sub14_vs_A3=S.paired_boot(E["sub14"], E3, rng14))
            sk[g] = blk
        summ[f"budget_2_{k}"] = sk

    # shape diagnosis, per group and arm (medians over corpora)
    diag = {}
    for g in GROUPS:
        ks = [x for x in keys_all if per[x]["group"] == g]
        dg = {}
        for arm in ARMS:
            D = [per[x]["arms"][arm]["diag"] for x in ks]
            med = lambda f: float(np.median([f(d) for d in D]))  # noqa: E731
            imp = [(d["implied_nu_20_25"], d["nu"]) for d in D if d["implied_nu_20_25"] is not None]
            dg[arm] = dict(
                median_gamma=med(lambda d: d["gamma"]), median_nu=med(lambda d: d["nu"]),
                median_ref_full=med(lambda d: d["ref_full"]),
                n_pilot20_below_curve=int(sum(d["L_below_curve_min"]["20"] for d in D)),
                n_L25_below_curve=int(sum(d["L_below_curve_min"]["25"] for d in D)),
                n_L25_outside_curve=int(sum(not d["L_inside_curve"]["25"] for d in D)),
                n_implied_nu=len(imp), median_implied_nu_20_25=float(np.median([i for i, _ in imp])) if imp else None,
                median_implied_over_nu=float(np.median([i / n for i, n in imp])) if imp else None,
                median_shape_ctx_bytes={k: med(lambda d, k=k: d["shape_ctx_bytes"][k]) for k in ("20", "25")},
                median_pilot_neff_bytes=med(lambda d: d["n_eff_bytes"]["20"]),
                median_h_slope_per_doubling_at_shape_ctx={k: med(lambda d, k=k: d["h_slope_per_doubling_at_shape_ctx"][k]) for k in ("20", "25")},
                median_h_slope_per_doubling_at_pilot_neff=med(lambda d: d["h_slope_per_doubling_at_pilot_neff"]),
                median_drop_20_25=dict(measured=med(lambda d: d["drop_20_25"]["measured"]), A2=med(lambda d: d["drop_20_25"]["A2"]),
                                       S_used_by_A4=med(lambda d: d["drop_20_25"]["S"])),
                median_drop_19_20=dict(measured=med(lambda d: d["drop_19_20"]["measured"]), S_used_by_A4=med(lambda d: d["drop_19_20"]["S"])),
                median_learner_slope_per_doubling_20_25=med(lambda d: d["drop_20_25"]["measured"] / 5))
        # range split at 2^25 (descriptive: selects on the measured loss). A2 ('neff') cannot fall below the curve's minimum,
        # so where the learner ends below it the error has the lower bound h_min - L(2^25) whatever the shape.
        e2 = lambda xs, arm: [per[x]["arms"][arm]["err"]["25"]["A2"] for x in xs]  # noqa: E731
        e3 = lambda xs: [per[x]["err_A3"]["25"] for x in xs]  # noqa: E731
        rs = lambda v: S.rms(v) if len(v) else None  # noqa: E731
        for arm in ARMS:
            ins = [x for x in ks if per[x]["arms"][arm]["diag"]["L_inside_curve"]["25"]]
            out_ = [x for x in ks if x not in ins]
            dg[arm]["range_split_2_25"] = dict(
                n_inside=len(ins), rms_A2_inside=rs(e2(ins, arm)), rms_A3_inside=rs(e3(ins)),
                n_below=len(out_), rms_A2_below=rs(e2(out_, arm)), rms_A3_below=rs(e3(out_)),
                rms_floor_lower_bound_all=S.rms([max(0.0, per[x]["arms"][arm]["diag"]["h_min"] - per[x]["measured"]["25"]) for x in ks]))
        com = [x for x in ks if all(per[x]["arms"][a]["diag"]["L_inside_curve"]["25"] for a in ("sub05", "sub15"))]
        dg["inside_both_sub05_sub15_2_25"] = dict(
            n=len(com), rms_A3=rs(e3(com)),
            **{arm: dict(rms_A2=rs(e2(com, arm)), mean_err_A2=float(np.mean(e2(com, arm))) if com else None,
                         median_implied_over_nu=float(np.median([per[x]["arms"][arm]["diag"]["implied_nu_20_25"] / per[x]["arms"][arm]["diag"]["nu"]
                                                                 for x in com])) if com else None) for arm in ("sub05", "sub15")})
        diag[g] = dg

    out = dict(design="CPU re-score of A2/A4 with the 0.5B and 1.5B Qwen curves; no refit (see module docstring)",
               chosen=ch, K_ts=K["K_ts"], n_corpora=len(keys_all),
               reproduction=dict(max_abs_full05_minus_stored_A2_A4=repro),
               summary=summ, diagnosis=diag, per_corpus=per)
    p = os.path.join(S.OUT14 if S.SUB14 else S.OUT7 if S.SUB7 else OUT, "shape_test.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print("sha256", hashlib.sha256(open(p, "rb").read()).hexdigest()[:16], "reproduction", repro)
    for k in ("23", "25"):
        for g in GROUPS:
            b = summ[f"budget_2_{k}"][g]
            print(k, g, "A3 %.3f" % b["rms_A3"], *("%s %s" % (a, {m: round(v, 3) for m, v in b[a]["rms"].items()}) for a in ("A2", "A4")))


if __name__ == "__main__":
    main()
