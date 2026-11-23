"""Score Test 1, code at 25M with a web-language control. Deterministic.

    python scripts/codeeng/score_t1.py      # results/code25m/pod_*/raw/ -> results/code25m/score.json

Per corpus: 25M losses at 2^23 (seed 0) and 2^25 (seeds 0, 1) from raw/text_runs/c25_<name>/; measured fall = L23 - mean L25.
Own fall x: stored (results/reanalysis/fall_diagnosis.json rows), or, when the pod's bytes differ from the 3.3M bytes,
the pod's forecast.py on the new bytes. 3.3M falls and reference statistics from the same rows. A corpus is scored when all
three runs exist. Per kind (code, web): R1-clean (4 scaled MADs of the 25M fall) leave-one-out line vs kind mean with a paired
bootstrap (20,000, default_rng(0)), 200 seeded 2-fold splits, Spearman; the full set as context; capacity statistics.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results", "code25m")
NB = 20000


def rms(e):
    return float(np.sqrt(np.mean(np.square(e))))


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def r1_keep(y, k=4.0):
    y = np.asarray(y, float); med = np.median(y); mad = 1.4826 * np.median(np.abs(y - med))
    return np.abs(y - med) <= k * mad


def loo(x, y):
    n = len(x); own = np.empty(n); mean = np.empty(n)
    for i in range(n):
        m = np.arange(n) != i
        b, a = np.polyfit(x[m], y[m], 1); own[i] = a + b * x[i]; mean[i] = y[m].mean()
    return own, mean


def compare(names, x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    own, mean = loo(x, y); ea, eb = own - y, mean - y
    idx = np.random.default_rng(0).integers(0, n, (NB, n))
    br = np.sqrt((ea[idx] ** 2).mean(1) / (eb[idx] ** 2).mean(1)); lo, hi = np.percentile(br, [2.5, 97.5])
    splits = []
    for s in range(200):
        p = np.random.default_rng(s).permutation(n); h = [p[: n // 2], p[n // 2:]]
        po, pm = np.empty(n), np.empty(n)
        for i in (0, 1):
            c, t = h[i], h[1 - i]
            b, a = np.polyfit(x[c], y[c], 1); po[t] = a + b * x[t]; pm[t] = y[c].mean()
        splits.append(rms(po - y) / rms(pm - y))
    b, a = np.polyfit(x, y, 1)
    return dict(n=n, corpora=list(names), fall_mean=float(y.mean()), fall_sd=float(y.std(ddof=1)), full_fit=dict(a=float(a), b=float(b)),
                spearman=spearman(x, y), rms_own=rms(ea), rms_mean=rms(eb), ratio=rms(ea) / rms(eb), ci95=[float(lo), float(hi)],
                label="REAL positive" if hi < 1 else ("REAL negative" if lo > 1 else "tie"),
                closer=f"{int(np.sum(np.abs(ea) < np.abs(eb)))}/{n}",
                splits=dict(median=float(np.median(splits)), p2_5=float(np.percentile(splits, 2.5)), p97_5=float(np.percentile(splits, 97.5)),
                            share_below_1=float(np.mean(np.array(splits) < 1))))


def main():
    cand = json.load(open(os.path.join(R, "candidates.json")))["list"]
    D = {r["corpus"]: r for r in json.load(open(os.path.join(ROOT, "results", "reanalysis", "fall_diagnosis.json")))["rows"]}
    rows, missing = [], {}
    for c in cand:
        n = c["name"]; got = {}
        for d in sorted(glob.glob(os.path.join(R, "pod_*", "raw"))):
            for k, s in ((23, 0), (25, 0), (25, 1)):
                f = os.path.join(d, "text_runs", f"c25_{n}", f"bytes_b{k}_s{s}.json")
                if os.path.exists(f):
                    got[(k, s)] = (json.load(open(f)), d)
        if len(got) < 3:
            missing[n] = sorted(f"{k}s{s}" for k, s in got)
            continue
        d = got[(23, 0)][1]
        by = json.load(open(os.path.join(d, "c25_bytes", f"{n}.json")))
        same = bool(by.get("train_same") and by.get("ref_same"))
        r0 = D[n]; x = r0["x"]; x_src = "stored"
        if not same:
            fc = json.load(open(os.path.join(d, "curve_forecast", "forecast.json")))["corpora"][n]
            x = fc["S"]["23"] - fc["S"]["25"]; x_src = "recomputed on the pod's bytes"
        L23 = got[(23, 0)][0]["bits_per_byte"]; L25s = [got[(25, s)][0]["bits_per_byte"] for s in (0, 1)]
        L25 = float(np.mean(L25s))
        rows.append(dict(corpus=n, kind="code" if c["family"] == "code" else "web", bytes_same=same, x=x, x_source=x_src,
                         L23=L23, L25_seeds=L25s, L25=L25, fall=L23 - L25,
                         steps={f"{k}s{s}": got[(k, s)][0]["steps"] for k, s in got}, d=got[(25, 0)][0]["d"], nl=got[(25, 0)][0]["nl"],
                         L23_3p3M=r0["L23"], L25_3p3M=r0["L25"], fall_3p3M=r0["fall"], x_stored=r0["x"],
                         ref_full=r0["ref_full"], ngram23=r0["ngram23"]))
    out = dict(design="docs/analysis/code25m_english.md", n_scored=len(rows), missing=missing, kinds={})
    for kd in ("code", "web"):
        s = [r for r in rows if r["kind"] == kd]
        if len(s) < 4:
            out["kinds"][kd] = dict(n=len(s)); continue
        y = np.array([r["fall"] for r in s]); x = np.array([r["x"] for r in s]); keep = r1_keep(y)
        cl = [r for r, k in zip(s, keep) if k]
        L25 = np.array([r["L25"] for r in s]); L25o = np.array([r["L25_3p3M"] for r in s])
        rf = np.array([r["ref_full"] for r in s]); ng = np.array([r["ngram23"] for r in s])
        f3 = np.array([r["fall_3p3M"] for r in s])
        out["kinds"][kd] = dict(
            n=len(s), dropped_R1=[r["corpus"] for r, k in zip(s, keep) if not k],
            bytes_differ=[r["corpus"] for r in s if not r["bytes_same"]],
            primary_R1_clean=compare([r["corpus"] for r in cl], [r["x"] for r in cl], [r["fall"] for r in cl]),
            full=compare([r["corpus"] for r in s], x, y),
            same_corpora_3p3M_R1_clean=compare([r["corpus"] for r in s if r1_keep(f3)[s.index(r)]],
                                               [r["x_stored"] for r, k in zip(s, r1_keep(f3)) if k], f3[r1_keep(f3)]),
            capacity=dict(L25_over_ref_25M=float(np.median(L25 / rf)), L25_over_ref_3p3M=float(np.median(L25o / rf)),
                          L25_over_ngram23_25M=float(np.median(L25 / ng)), L25_over_ngram23_3p3M=float(np.median(L25o / ng)),
                          share_L25_below_ref_25M=float(np.mean(L25 < rf)), share_L25_below_ref_3p3M=float(np.mean(L25o < rf)),
                          realised_fall_over_x_25M=float(np.median(y / x)), realised_fall_over_x_3p3M=float(np.median(f3 / np.array([r["x_stored"] for r in s]))),
                          median_fall_25M=float(np.median(y)), median_fall_3p3M=float(np.median(f3)),
                          L25_25M_minus_3p3M_median=float(np.median(L25 - L25o))),
            seed_sd_2_25=float(np.sqrt(np.mean([(r["L25_seeds"][1] - r["L25_seeds"][0]) ** 2 for r in s]) / 2)))
    w = out["kinds"].get("web", {}).get("primary_R1_clean")
    if w:
        out["control_holds"] = bool(w["ratio"] < 0.95 and w["spearman"] > 0.3)
    out["rows"] = rows
    json.dump(out, open(os.path.join(R, "score.json"), "w"), indent=1)
    print("scored", len(rows), "missing", len(missing))
    for kd, v in out["kinds"].items():
        if "primary_R1_clean" not in v:
            print(kd, v); continue
        for nm in ("primary_R1_clean", "full", "same_corpora_3p3M_R1_clean"):
            c = v[nm]
            print(f"{kd:4s} {nm:28s} n={c['n']:2d} ratio {c['ratio']:.3f} ({c['ci95'][0]:.3f}-{c['ci95'][1]:.3f}) {c['label']} closer {c['closer']} "
                  f"rho {c['spearman']:+.2f} b {c['full_fit']['b']:+.3f} splits {c['splits']['median']:.2f} [{c['splits']['p2_5']:.2f},{c['splits']['p97_5']:.2f}] <1 {c['splits']['share_below_1']:.2f}")
        print("     dropped", v["dropped_R1"], "bytes differ", v["bytes_differ"], "seed sd", round(v["seed_sd_2_25"], 4))
        print("     capacity", {k: round(q, 3) for k, q in v["capacity"].items()})
    print("control holds:", out.get("control_holds"))


if __name__ == "__main__":
    main()
