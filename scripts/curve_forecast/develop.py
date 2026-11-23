"""Phase 1: development on existing data only, then the lock.

    python scripts/curve_forecast/develop.py      # -> results/curve_forecast/development.json, locked.json (+ sha256 printed)

Development corpora: the 11 measured corpora of results/mc (code, pg19, pubmed, ja_wiki) and results/mc2 (7), TinyStories
(text/refcurve/refcurve.json, TinyStories-33M) and FineWeb-Edu (bfine, SmolLM2-135M). Primary reference LM: Qwen2.5-0.5B for
non-English Wikipedia, SmolLM2-135M otherwise. Emulation of the test protocol: pilots at 2^21 and 2^22 (seed 0), target 2^23
(mean over seeds); the one-pilot arms use the larger pilot. Every learned constant is refitted leaving the scored corpus out.
Choices (each by leave-one-corpus-out rms at 2^23):
  shape + A2 form + A4 form   jointly, by the pooled rms of the A2 and A4 errors (the co-primary information arms)
  level (A0)                  given the chosen shape, on the corpora where every level statistic exists
  A3 floor rule               by A3's own rms (the strongest pilot-only baseline)
Secondary, reported only: far extrapolation on FineWeb-Edu (pilots 2^21, 2^22 -> 2^25) and TinyStories (2^20, 2^21 -> 2^24).
"""
import datetime
import glob
import hashlib
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402

ROOT = C.ROOT
OUT = os.path.join(ROOT, "results/curve_forecast")
FIRST4 = ("code", "pg19", "pubmed", "ja_wiki")
EXT7 = ("arxiv", "cc_news", "de_wiki", "java_code", "openwebmath", "stackexchange", "zh_wiki")
QWEN = ("ja_wiki", "de_wiki", "zh_wiki")
P1, P2, KT = 21, 22, 23


def runs(pat):
    out = {}
    for f in glob.glob(pat):
        d = json.load(open(f))
        out.setdefault(int(round(np.log2(d["train_bytes"]))), {})[d["seed"]] = d["bits_per_byte"]
    return out


def load_dev():
    dev = {}
    for c in FIRST4 + EXT7:
        res = "mc" if c in FIRST4 else "mc2"
        m = "Qwen2.5-0.5B" if c in QWEN else "SmolLM2-135M"
        hr = C.FN.HRef(os.path.join(ROOT, f"results/{res}/{c}/refcurve/refcurve_fineweb.json"), m)
        dev[c] = dict(hr=hr, ref_model=m, R=runs(os.path.join(ROOT, f"results/text_runs/{res}_{c}/bytes_b*_s*.json")),
                      ngram_file=f"results/{res}/{c}/ngram_23.json")
    dev["tinystories"] = dict(hr=C.FN.HRef(C.FN.TS_HREF), ref_model="TinyStories-33M",
                              R={k: v for t in C.FN.TS_TAGS for k, v in runs(os.path.join(ROOT, f"results/text_runs/{t}/bytes_b*_s*.json")).items()},
                              ngram_file="results/mc2/cal_ts/ngram_23.json")
    dev["fineweb_edu"] = dict(hr=C.FN.HRef(os.path.join(ROOT, "results/bfine/refcurve/refcurve_fineweb.json"), "SmolLM2-135M"),
                              ref_model="SmolLM2-135M", R=runs(os.path.join(ROOT, "results/text_runs/bfine/bytes_b*_s*.json")),
                              ngram_file="results/mc2/cal_fw/ngram_23.json")
    for c, d in dev.items():
        d["st"] = C.ref_stats(d["hr"])
        d["st"]["ngram23"] = C.best_ngram(os.path.join(ROOT, d["ngram_file"]))
        d["R"] = {k: v for k, v in d["R"].items() if k <= 25}          # TinyStories 2^26 is capacity-limited (forecast_neff)
        d["L"] = {k: float(np.mean(list(v.values()))) for k, v in d["R"].items()}
        d["pil"] = lambda k, d=d: d["R"][k][0]                           # pilots: seed 0
    return dev


def consts_for(dev, train, base):
    """Learned constants from the training corpora only."""
    K = dict(base)
    K["nu_pool"], K["n0_pool"] = C.fit_pool([(dev[c]["hr"], dev[c]["L"]) for c in train])
    return K


def loco(dev, names, base, p1=P1, p2=P2, kt=KT, fw=None):
    """Leave-one-corpus-out errors of every candidate at kt for the corpora in names. Returns {candidate: {corpus: err}}."""
    E = {}
    put = lambda key, c, v: E.setdefault(key, {}).__setitem__(c, float(v))
    for c in names:
        tr = [q for q in dev if q != c]
        K = consts_for(dev, tr, base)
        d = dev[c]
        meas = d["L"][kt]
        pil = {p1: d["pil"](p1), p2: d["pil"](p2)}
        for sh in C.SHAPES:
            nu, n0 = C.shape_params(sh, d["st"], K)
            for f in C.FORMS:
                put(f"A2|{sh}|{f}", c, C.hybrid(d["hr"], nu, n0, {p2: pil[p2]}, kt, f) - meas)
                put(f"A4|{sh}|{f}", c, C.hybrid(d["hr"], nu, n0, pil, kt, f) - meas)
            for lv, md in itertools.product(C.LEVELS, C.MODES):
                coef = C.fit_level(lv, level_rows(dev, tr, sh, K, lv, md))
                put(f"A0|{sh}|{lv}|{md}", c, C.A0(d["hr"], d["st"], nu, n0, lv, md, coef, kt) - meas)
        for rule in C.FLOORS:
            # floor parameters are always fitted on the standard emulation (2^21, 2^22 -> 2^23) of the other corpora
            cases = [({P1: dev[q]["pil"](P1), P2: dev[q]["pil"](P2)}, KT, dev[q]["L"][KT]) for q in tr]
            put(f"A3|{rule}", c, C.A3(pil, kt, rule, C.fit_floor(rule, cases)) - meas)
        if fw is not None and c != "fineweb_edu":
            put("A1|fineweb", c, C.A1(pil[p2], p2, kt, fw) - meas)
    return E


def level_rows(dev, names, sh, K, lv, md):
    rows = []
    for q in names:
        if all(np.isfinite(C.level_X(lv, dev[q]["st"]))):
            nq, mq = C.shape_params(sh, dev[q]["st"], K)
            rows.append((dev[q]["st"], C.level_target(dev[q]["hr"], nq, mq, dev[q]["L"], md)))
    return rows


def rms(d, names=None):
    v = [d[c] for c in (names or d) if c in d and np.isfinite(d[c])]
    return float(np.sqrt(np.mean(np.square(v)))) if v else float("nan")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = load_dev()
    names = list(dev)
    base = C.ts_constants()
    fw = dev["fineweb_edu"]["L"]
    E = loco(dev, names, base, fw=fw)
    score = {k: rms(v) for k, v in E.items()}

    # shape + forms: pooled rms of A2 and A4
    joint = {}
    for sh, f2, f4 in itertools.product(C.SHAPES, C.FORMS, C.FORMS):
        e = [E[f"A2|{sh}|{f2}"][c] for c in names] + [E[f"A4|{sh}|{f4}"][c] for c in names]
        joint[f"{sh}|A2:{f2}|A4:{f4}"] = float(np.sqrt(np.mean(np.square(e))))
    best = min(joint, key=joint.get)
    shape, f2, f4 = best.split("|")[0], best.split("|")[1][3:], best.split("|")[2][3:]
    # level: corpora where every statistic is finite
    common = [c for c in names if all(np.isfinite(C.level_X("both", dev[c]["st"])))]
    lvl = {f"{lv}|{md}": rms(E[f"A0|{shape}|{lv}|{md}"], common) for lv, md in itertools.product(C.LEVELS, C.MODES)}
    level, mode = min(lvl, key=lvl.get).split("|")
    flo = {r: score[f"A3|{r}"] for r in C.FLOORS}
    floor = min(flo, key=flo.get)
    chosen = dict(shape=shape, A2_form=f2, A4_form=f4, level=level, level_mode=mode, A3_floor=floor)
    arms = dict(A0=f"A0|{shape}|{level}|{mode}", A1="A1|fineweb", A2=f"A2|{shape}|{f2}", A3=f"A3|{floor}", A4=f"A4|{shape}|{f4}")
    per = {a: E[k] for a, k in arms.items()}

    # constants fitted on all development corpora
    K = consts_for(dev, names, base)
    coef = C.fit_level(level, level_rows(dev, names, shape, K, level, mode))
    cases = [({P1: dev[q]["pil"](P1), P2: dev[q]["pil"](P2)}, KT, dev[q]["L"][KT]) for q in names]
    fpar = C.fit_floor(floor, cases)
    # fallback level (ref only) for a test corpus whose n-gram statistic is not finite
    coef_ref = C.fit_level("ref", level_rows(dev, names, shape, K, "ref", mode))

    # secondary: far extrapolation with the chosen arms (constants refitted leaving the corpus out)
    far = {}
    for c, p1, p2, kt in (("fineweb_edu", 21, 22, 25), ("tinystories", 20, 21, 24)):
        e = loco(dev, [c], base, p1, p2, kt, fw=None)
        far[c] = dict(pilots=[p1, p2], target=kt, err={a: e[k][c] for a, k in arms.items() if k in e})

    dev_out = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        design="docs/plan/curve_forecast_vs_pilots_2026-10-17.md",
        corpora={c: dict(ref_model=dev[c]["ref_model"], ref_full=dev[c]["st"]["ref_full"], gamma=dev[c]["st"]["gamma"],
                         ngram23=dev[c]["st"]["ngram23"], measured={str(k): v for k, v in sorted(dev[c]["L"].items())},
                         pilots_seed0={str(k): dev[c]["pil"](k) for k in (P1, P2)}) for c in names},
        emulation=dict(pilots=[P1, P2], one_pilot=P2, target=KT, pilot_seed=0, target_value="mean over seeds"),
        loco_rms_all_candidates=score, joint_shape_forms_rms=joint,
        level_rms_on_common=dict(corpora=common, rms=lvl), floor_rms=flo,
        chosen=chosen, chosen_arm_keys=arms,
        chosen_arms_loco_rms={a: rms(v) for a, v in per.items()},
        chosen_arms_loco_rms_without_fineweb={a: rms(v, [c for c in names if c != "fineweb_edu"]) for a, v in per.items()},
        ratios=dict(A4_over_A3=rms(per["A4"]) / rms(per["A3"]), A2_over_A3=rms(per["A2"]) / rms(per["A3"]),
                    A0_over_A1=rms(per["A0"], [c for c in names if c != "fineweb_edu"]) / rms(per["A1"])),
        per_corpus_errors=per, far_extrapolation_secondary=far, all_errors=E)
    json.dump(C_clean(dev_out), open(os.path.join(OUT, "development.json"), "w"), indent=1)

    files = sorted(glob.glob(os.path.join(HERE, "*.py"))) + [os.path.join(C.SCRIPTS, n) for n in ("forecast_neff.py", "forecast_input_variants.py", "forecast_multicorpus.py",
                                                                                  "ngram_curve_ts.py", "prep_corpus.py", "run_text_ladder.py", "train_text.py")]
    locked = dict(
        created_utc=dev_out["created_utc"], design="docs/plan/curve_forecast_vs_pilots_2026-10-17.md", chosen=chosen,
        constants=dict(nu_ts=K["nu_ts"], n0_ts=K["n0_ts"], K_ts=K["K_ts"], nu_pool=K["nu_pool"], n0_pool=K["n0_pool"],
                       level_coef=coef, level_coef_ref_fallback=coef_ref, A3_floor_param=fpar, E_cap=C.E_CAP),
        fineweb_edu_measured_bpb={str(k): v for k, v in sorted(fw.items())},
        rules=dict(primary_ref="Qwen2.5-0.5B for non-English text, SmolLM2-135M for English and code",
                   pilots="seed 0 at 2^19 and 2^20; one-pilot arms use 2^20",
                   targets="2^23 and 2^25, mean over the seeds run",
                   A1_shape="FineWeb-Edu measured bpb, mean over seeds, at the pilot and target budgets (2^19, 2^20 run on the pod)",
                   level_fallback="if the chosen level statistic is not finite for a corpus, the ref-only level map is used"),
        code_sha256={os.path.relpath(f, ROOT): sha(f) for f in files})
    lp = os.path.join(OUT, "locked.json")
    json.dump(C_clean(locked), open(lp, "w"), indent=1)
    print(json.dumps(dict(chosen=chosen, chosen_arms_loco_rms=dev_out["chosen_arms_loco_rms"], ratios=dev_out["ratios"]), indent=1))
    print("sha256", sha(lp), os.path.relpath(lp, ROOT))
    for f, h in locked["code_sha256"].items():
        print("sha256", h, f)


def C_clean(o):
    if isinstance(o, dict):
        return {str(k): C_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [C_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    return o


if __name__ == "__main__":
    main()
