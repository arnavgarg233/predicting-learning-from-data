"""sensitivity (2026-11-05): one loader for every stored curve-forecast corpus (no training, stored JSON only).
Row: corpus, set, family (code / english / lang), ref_model, hr (reference information curve), g (best n-gram
cross-entropy at 2^23, if stored), L {19, 20, 23, 25} (as scored: 2^25 is the mean over the seeds run), L25_seeds,
stored values for reproduction checks (A0, S, own_fall).
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts", "curve_forecast"))
import core as C  # noqa: E402

RES = os.path.join(ROOT, "results")
CF = os.path.join(RES, "curve_forecast")
FAM = {"code": "code", "english": "english", "wiki": "lang", "lang": "lang"}


def _hr(path, model):
    return C.FN.HRef(path, model)


def _first(*paths):
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(paths[0])


def _seeds(dirs, k=25):
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, f"bytes_b{k}_s*.json")):
            j = json.load(open(f))
            out[str(j["seed"])] = j["bits_per_byte"]
    return out


def _runs(dirs):
    """{(k, seed): run json} for the first dir that has runs."""
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "bytes_b*_s*.json")):
            j = json.load(open(f))
            out.setdefault((int(round(np.log2(j["train_bytes"]))), j["seed"]), j)
    return out


def load():
    rows = []
    s80 = json.load(open(os.path.join(CF, "score80.json")))["per_corpus"]
    FC = {**json.load(open(os.path.join(CF, "forecast.json")))["corpora"],
          **json.load(open(os.path.join(CF, "stage2/forecast.json")))["corpora"]}
    for c, p in s80.items():
        rp = _first(os.path.join(CF, c, "refcurve/refcurve_fineweb.json"), os.path.join(CF, "stage2", c, "refcurve/refcurve_fineweb.json"))
        rdirs = [os.path.join(RES, "text_runs", f"cf_{c}"), os.path.join(CF, "stage2/text_runs", f"cf_{c}")]
        rows.append(dict(corpus=c, set="test80", family=FAM[p["family"]], ref_model=p["ref_model"], refpath=rp,
                         g=FC[c]["ref_stats"]["ngram23"], L={int(k): v for k, v in {**p["pilots"], **p["measured"]}.items()},
                         L25_seeds=_seeds(rdirs), rundirs=rdirs, stored=dict(A0_23=p["forecast"]["23"]["A0"], A0_25=p["forecast"]["25"]["A0"],
                                                                     A3_25=p["forecast"]["25"]["A3"])))
    nl = json.load(open(os.path.join(RES, "reanalysis/nl_table.json")))
    for r in nl:
        if r["group"] == "test80":
            continue
        if r["group"] == "new18":
            base = os.path.join(CF, "new_corpora")
            rp = os.path.join(base, "pod", r["corpus"], "refcurve/refcurve_fineweb.json")
            rdirs = [os.path.join(base, "text_runs", f"cf_{r['corpus']}")]
            st = "forward18"
        else:
            rp = os.path.join(CF, "stage3", r["corpus"], "refcurve/refcurve_fineweb.json")
            rdirs = [os.path.join(CF, "stage3/text_runs", f"cf_{r['corpus']}")]
            st = "stage3"
        rows.append(dict(corpus=r["corpus"], set=st, family=FAM[r["family"]], ref_model=r["ref_model"], refpath=rp, g=r["g"],
                         L={19: r["L19"], 20: r["L20"], 23: r["L23"], 25: r["L25"]}, L25_seeds=_seeds(rdirs), rundirs=rdirs,
                         stored=dict(A0_23=r["A0_23"], A0_25=r["A0_25"], A3_25=r["A3_25"], S23=r["S23"], S25=r["S25"])))
    eb = json.load(open(os.path.join(RES, "extbench/score.json")))["per_corpus"]
    FI = {}
    for p in (1, 2, 3):
        FI.update(json.load(open(os.path.join(RES, "extbench", f"pod{p}", "forecast_i.json")))["corpora"])
    for c, r in eb.items():
        sd = os.path.join(ROOT, r["source_dir"])
        rp = os.path.join(sd, "pod", c, "refcurve/refcurve_fineweb.json")
        rdirs = [os.path.join(sd, "text_runs", f"cf_{c}")]
        rows.append(dict(corpus=c, set="external45", family=FAM[r["family"]], ref_model=r["ref_model"], refpath=rp,
                         g=FI[c]["ref_stats"]["ngram23"], L={int(k): v for k, v in r["L"].items()}, L25_seeds=r["L25_seeds"],
                         rundirs=rdirs, stored=dict(A0_23=FI[c]["A0"]["23"], A0_25=FI[c]["A0"]["25"], S23=FI[c]["S"]["23"],
                                                    S25=FI[c]["S"]["25"], rev_25=r["forecast"]["25"]["ii"], rev_23=r["forecast"]["23"]["ii"],
                                                    A3_25=r["forecast"]["25"]["A3"])))
    wl = json.load(open(os.path.join(RES, "fwd_weblang/score.json")))["per_corpus"]
    for c, r in wl.items():
        cands = glob.glob(os.path.join(RES, "fwd_weblang", "pod*", "pod", c, "refcurve/refcurve_fineweb.json")) + \
            glob.glob(os.path.join(RES, "fwd_weblang", "pod*", "raw/curve_forecast", c, "refcurve/refcurve_fineweb.json"))
        rdirs = sorted({os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(p)))) for p in cands})
        rdirs = [os.path.join(d, "text_runs", f"cf_{c}") for d in rdirs] + \
            [os.path.join(d, "raw/text_runs", f"cf_{c}") for d in glob.glob(os.path.join(RES, "fwd_weblang", "pod*"))]
        rows.append(dict(corpus=c, set="weblang30", family="lang", ref_model="Qwen2.5-0.5B", refpath=cands[0], g=float("nan"),
                         L={int(k): v for k, v in r["L"].items()}, L25_seeds=r["L25_seeds"], rundirs=rdirs,
                         stored=dict(own_fall=r["own_fall"], level_23=r["level_23"])))
    en = json.load(open(os.path.join(RES, "fwd_english/score.json")))["per_corpus"]
    for c, r in en.items():
        d = os.path.join(ROOT, r["dir"])
        rp = _first(os.path.join(d, "pod", c, "refcurve/refcurve_fineweb.json"), os.path.join(d, "raw/curve_forecast", c, "refcurve/refcurve_fineweb.json"))
        rows.append(dict(corpus=c, set="english24", family="english", ref_model="SmolLM2-135M", refpath=rp, g=float("nan"),
                         L={int(k): v for k, v in r["L"].items()}, L25_seeds=r["L25_seeds"],
                         rundirs=[os.path.join(d, "text_runs", f"cf_{c}"), os.path.join(d, "raw/text_runs", f"cf_{c}")],
                         stored=dict(own_fall=r["own_fall"], level_23=r["level_23"])))
    for r in rows:
        r["hr"] = _hr(r["refpath"], r["ref_model"])
    return rows


if __name__ == "__main__":
    import collections
    rows = load()
    print(len(rows), collections.Counter((r["set"], r["family"]) for r in rows))
    print("with 2 seeds at 2^25:", collections.Counter(r["set"] for r in rows if len(r["L25_seeds"]) > 1))
