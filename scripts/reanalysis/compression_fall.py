"""Does a compression statistic, calibrated the same way, forecast the fall
L(2^23) - L(2^25) as well as the own information-curve fall? Deterministic, CPU, stored data only.

    python scripts/reanalysis/compression_fall.py   # -> results/reanalysis/compression_fall.json

Done after the two forward outcomes were known. The own-curve lines were frozen before the forward corpora were prepared
(results/fwd_weblang/fall_line.json, results/fwd_english/fall_line.json); the compression lines are fitted here,
on the same calibration corpora only, and applied unchanged to the same test corpora.

The prepared training bytes are not stored locally (they were prepared on the pods and not pulled), so lzma on the learner's bytes
cannot be computed. Statistics used, all stored per corpus and computed on the learner's 2^23 training bytes before training:
  ngram23  best-order n-gram cross-entropy at 2^23 bytes (forecast_i.json ref_stats.ngram23 = min over orders of ngram_23.json)
  h0       order-0 (byte-frequency) cross-entropy at 2^23 bytes (ngram_23.json, row n = 0)
  gain     h0 - ngram23, the bits per byte that context removes for the n-gram compressor
Families and sets (the frozen own-curve line's own):
  web      calibration: the 29 external web languages (shape_vs_pooled.json, external45, lang); test: the 30 forward web languages
  english  calibration: the 25 clean of 29 English corpora (fall_diagnosis.json, frozen set_aside removed); test: the 20 clean
           forward English corpora (fwd_english score.json primary_clean)
Per statistic s: OLS fall = a + b s on the calibration corpora, frozen, applied to the test corpora. Two-variable context line:
fall = a + b own_fall + c ngram23. Paired corpus bootstrap on the test corpora (20,000 draws, numpy default_rng(0), 95%
percentile interval), the same draws for every ratio.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results")
NB = 20000
STATS = ("ngram23", "h0", "gain")


def J(p):
    return json.load(open(os.path.join(ROOT, p) if not os.path.isabs(p) else p))


def rms(e):
    return float(np.sqrt(np.mean(np.square(e))))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


NG_FILES = sorted(glob.glob(os.path.join(RES, "**", "ngram_23.json"), recursive=True))


def ngram_rows(name, ngram23):
    """ngram_23.json of `name` whose best order reproduces the stored ngram23 (several pulled copies exist)."""
    for f in NG_FILES:
        if os.path.basename(os.path.dirname(f)) != name:
            continue
        rows = J(f)["rows"]
        if abs(min(r["h_ng"] for r in rows) - ngram23) < 1e-9:
            h = {r["n"]: r["h_ng"] for r in rows}
            return dict(ngram23=ngram23, h0=h[0], gain=h[0] - ngram23, src=os.path.relpath(f, ROOT))
    raise SystemExit(f"no ngram_23.json matching {name} {ngram23}")


def forecast_ngram23(dirs, name):
    for d in dirs:
        c = J(os.path.join(d, "forecast_i.json"))["corpora"]
        if name in c:
            return c[name]["ref_stats"]["ngram23"]
    raise SystemExit(f"no forecast_i for {name}")


def web():
    eb = sorted(glob.glob(os.path.join(RES, "extbench", "pod*")))
    cal = []
    for r in J("results/reanalysis/shape_vs_pooled.json")["per_corpus"]:
        if r["set"] == "external45" and r["family"] == "lang":
            cal.append(dict(corpus=r["corpus"], fall=r["fall"], own=r["S23"] - r["S25"],
                            **ngram_rows(r["corpus"], forecast_ngram23(eb, r["corpus"]))))
    S = J("results/fwd_weblang/score.json"); Z = J("results/fwd_weblang/fall_line.json")["calibration"]
    test = []
    for c, r in sorted(S["per_corpus"].items()):
        test.append(dict(corpus=c, fall=r["fall"], own=r["own_fall"], own_cal_frozen=r["pred_fall"]["own_cal"], L23=r["L"]["23"],
                         **ngram_rows(c, forecast_ngram23([os.path.join(ROOT, r["dir"])], c))))
    return cal, test, dict(a=Z["a"], b=Z["b"], kind_mean=Z["kind_mean_fall"]), S["fall_rms"]


def english():
    Z = J("results/fwd_english/fall_line.json")["calibration"]
    cal = []
    for r in J("results/reanalysis/fall_diagnosis.json")["rows"]:
        if r["kind"] == "english" and r["corpus"] not in Z["set_aside"]:
            cal.append(dict(corpus=r["corpus"], fall=r["fall"], own=r["x"], **ngram_rows(r["corpus"], r["ngram23"])))
    S = J("results/fwd_english/score.json"); P = S["primary_clean"]
    test = []
    for c in P["corpora"]:
        r = S["per_corpus"][c]
        test.append(dict(corpus=c, fall=r["fall"], own=r["own_fall"], own_cal_frozen=r["pred_fall"]["own_cal"], L23=r["L"]["23"],
                         **ngram_rows(c, forecast_ngram23([os.path.join(ROOT, r["dir"])], c))))
    return cal, test, dict(a=Z["line"]["a"], b=Z["line"]["b"], kind_mean=Z["kind_mean"]), P["fall_rms"]


def fit(cal, cols):
    X = np.column_stack([np.ones(len(cal))] + [[r[c] for r in cal] for c in cols])
    y = np.array([r["fall"] for r in cal])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def apply(coef, rows, cols):
    X = np.column_stack([np.ones(len(rows))] + [[r[c] for r in rows] for c in cols])
    return X @ coef


def family(name, cal, test, frozen, stored_rms):
    n = len(test)
    y = np.array([r["fall"] for r in test])
    own = np.array([r["own"] for r in test])
    # refit of the own line on the same calibration corpora must reproduce the frozen line
    co = fit(cal, ["own"])
    assert abs(co[0] - frozen["a"]) < 1e-9 and abs(co[1] - frozen["b"]) < 1e-9, (name, co, frozen)
    assert abs(np.mean([r["fall"] for r in cal]) - frozen["kind_mean"]) < 1e-9
    pred = dict(own_line=frozen["a"] + frozen["b"] * own, kind_mean=np.full(n, frozen["kind_mean"]))
    assert np.allclose(pred["own_line"], [r["own_cal_frozen"] for r in test], atol=1e-12)
    lines = dict(own_line=dict(cols=["own"], a=frozen["a"], b=[frozen["b"]], when="frozen before the forward corpora"))
    for s in STATS:
        c = fit(cal, [s]); pred[f"{s}_line"] = apply(c, test, [s])
        lines[f"{s}_line"] = dict(cols=[s], a=float(c[0]), b=[float(c[1])], when="fitted 2026-11-02 on calibration corpora only")
    c = fit(cal, ["own", "ngram23"]); pred["own+ngram23_line"] = apply(c, test, ["own", "ngram23"])
    lines["own+ngram23_line"] = dict(cols=["own", "ngram23"], a=float(c[0]), b=[float(c[1]), float(c[2])],
                                     when="fitted 2026-11-02 on calibration corpora only")
    E = {k: v - y for k, v in pred.items()}
    assert abs(rms(E["own_line"]) - stored_rms["own_cal"]) < 1e-9 and abs(rms(E["kind_mean"]) - stored_rms["kind_mean"]) < 1e-9
    idx = np.random.default_rng(0).integers(0, n, (NB, n))

    def ratio(a, b):
        ea, eb = E[a], E[b]
        br = np.sqrt((ea[idx] ** 2).mean(1) / np.maximum((eb[idx] ** 2).mean(1), 1e-30)); lo, hi = np.percentile(br, [2.5, 97.5])
        return dict(ratio=rms(ea) / rms(eb), ci95=[float(lo), float(hi)],
                    label="below 1" if hi < 1 else ("above 1" if lo > 1 else "tie"),
                    closer=f"{int(np.sum(np.abs(ea) < np.abs(eb)))}/{n}")
    comps = [("own_line", "kind_mean")] + [("own_line", f"{s}_line") for s in STATS] + \
            [(f"{s}_line", "kind_mean") for s in STATS] + [("own+ngram23_line", "own_line"), ("own+ngram23_line", "kind_mean")]
    calx = {s: np.array([r[s] for r in cal]) for s in ("own",) + STATS}; caly = np.array([r["fall"] for r in cal])
    return dict(
        n_calibration=len(cal), n_test=n, calibration_corpora=[r["corpus"] for r in cal], test_corpora=[r["corpus"] for r in test],
        lines=lines, fall_rms={k: rms(v) for k, v in E.items()}, fall_bias={k: float(v.mean()) for k, v in E.items()},
        ratios={f"{a}/{b}": ratio(a, b) for a, b in comps},
        spearman_test={s: spearman(np.array([r[s] for r in test]), y) for s in ("own",) + STATS},
        spearman_calibration={s: spearman(calx[s], caly) for s in calx},
        spearman_own_vs_ngram23_test=spearman(own, np.array([r["ngram23"] for r in test])),
        spearman_context_test=dict(ngram23_vs_L23=spearman(np.array([r["ngram23"] for r in test]), np.array([r["L23"] for r in test])),
                                   L23_vs_fall=spearman(np.array([r["L23"] for r in test]), y)),
        per_corpus=[dict(r, pred={k: float(v[i]) for k, v in pred.items()}) for i, r in enumerate(test)],
        calibration_rows=cal)


def main():
    out = dict(
        design=__doc__,
        timing="Comparison made 2026-11-02 after both forward outcomes were known. Own-curve lines were frozen before the forward "
               "corpora were prepared; compression lines are fitted now on the calibration corpora only and never see a test fall.",
        text_on_mac=False,
        text_note="Prepared training bytes (data/cf/<corpus>/tok/tok_bytes_train.npy) exist only on the deleted pods; no FineWeb-2, "
                  "Common Pile or Pile of Law text for these corpora is cached on the Mac. lzma could not be computed on the learner's "
                  "bytes; stored n-gram statistics of the same bytes are used instead.",
        families={})
    for name, f in (("web", web), ("english", english)):
        out["families"][name] = family(name, *f())
    json.dump(out, open(os.path.join(RES, "reanalysis", "compression_fall.json"), "w"), indent=1)
    for name, F in out["families"].items():
        print(f"{name}: calib {F['n_calibration']} test {F['n_test']}")
        print("  rms", {k: round(v, 4) for k, v in F["fall_rms"].items()})
        for k, v in F["ratios"].items():
            print(f"  {k:28s} {v['ratio']:.3f} ({v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}) {v['label']} closer {v['closer']}")
        print("  spearman test", {k: round(v, 2) for k, v in F["spearman_test"].items()},
              "calib", {k: round(v, 2) for k, v in F["spearman_calibration"].items()},
              "own~ngram23", round(F["spearman_own_vs_ngram23_test"], 2), {k: round(v, 2) for k, v in F["spearman_context_test"].items()})
        print("  lines", {k: (round(v["a"], 4), [round(b, 4) for b in v["b"]]) for k, v in F["lines"].items()})


if __name__ == "__main__":
    main()
