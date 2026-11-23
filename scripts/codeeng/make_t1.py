"""Test 1 (code at 25M) corpus list and pod lists. Run once before any pod.

    python scripts/codeeng/make_t1.py     # -> results/code25m/candidates.json, candidates_<slot>.json

Code: the 30 code corpora with own falls, in the order of results/reanalysis/shape_vs_pooled.json (test80 14,
forward18 5, external45 11). Control: the first 15 of the 30 forward web languages (results/fwd_weblang/fall_line.json
corpora.order). For each corpus the sha256 of tok_bytes_train.npy and tok_bytes_ref.npy recorded by the forecast of its 3.3M run
(the bytes the 3.3M learner and the own fall used). Pod lists: code in three contiguous blocks of 10 (c1, c2, c3), web
alternately (w1 positions 1, 3, ..., 15; w2 positions 2, 4, ..., 14).
"""
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "code25m")


def inputs(files):
    out = {}
    for f in files:
        for c, v in json.load(open(f))["corpora"].items():
            t = [h for p, h in v["inputs"].items() if p.endswith("tok_bytes_train.npy")][0]
            r = [h for p, h in v["inputs"].items() if p.endswith("tok_bytes_ref.npy")][0]
            out.setdefault(c, (t, r, os.path.relpath(f, ROOT), v["ref_model"]))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    svp = json.load(open(os.path.join(R, "reanalysis", "shape_vs_pooled.json")))["per_corpus"]
    code = [r["corpus"] for r in svp if r["family"] == "code"]
    web = json.load(open(os.path.join(R, "fwd_weblang", "fall_line.json")))["corpora"]["order"][:15]
    ins = inputs([os.path.join(R, "curve_forecast", "forecast.json"),
                  os.path.join(R, "curve_forecast", "stage2", "forecast.json"),
                  os.path.join(R, "curve_forecast", "new_corpora", "forecast.json")]
                 + sorted(glob.glob(os.path.join(R, "extbench", "pod*", "forecast_i.json")))
                 + sorted(glob.glob(os.path.join(R, "fwd_weblang", "pod*", "forecast_i.json"))))
    lst = []
    for i, c in enumerate(code + web):
        t, r, src, model = ins[c]
        lst.append(dict(position=i + 1, name=c, family="code" if c.startswith("code_") else "lang", ref_model=model,
                        train_sha256=t, ref_sha256=r, sha_source=src))
    cd, wb = lst[:30], lst[30:]
    slots = dict(c1=cd[0:10], c2=cd[10:20], c3=cd[20:30], w1=wb[0::2], w2=wb[1::2])
    json.dump(dict(design="docs/analysis/code25m_english.md", list=lst, slots={k: [c["name"] for c in v] for k, v in slots.items()}),
              open(os.path.join(OUT, "candidates.json"), "w"), indent=1)
    for k, v in slots.items():
        json.dump(dict(list=v), open(os.path.join(OUT, f"candidates_{k}.json"), "w"), indent=1)
        print(k, [c["name"] for c in v])


if __name__ == "__main__":
    main()
