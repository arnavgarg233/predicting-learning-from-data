"""External benchmark (docs/plan/referee_fixes_cost_2026-10-28.md, section 3): forecaster (ii), the revised n-gram-level route
of section 1b, applied to the forecast (i) file of one pod. No learner run is read.

    python scripts/extbench/forecast_ii.py FORECAST_I_JSON OUT_JSON

Per corpus (inputs all come from forecast (i), which is computed from the data and the reference curve before training):
  level at 2^23:
    code      A0(2^23) of forecast (i) minus CODE_BIAS (the paper's mean level bias on the 14 test code corpora)
    english   A0(2^23) of forecast (i), unchanged (English keeps the reference level; chosen from test80 alone, where the
              n-gram level was worse on English: leave-one-out 0.117 against 0.094)
    other     NGRAM[0] + NGRAM[1] * g, g = best n-gram cross-entropy at 2^23 (the development fit, nl_dev_ngram_level.json)
  drop 2^23 -> 2^25: LAM * (S(2^23) - S(2^25)), S the information shape of forecast (i)
  forecast at 2^25 = level - drop.
Constants as in results/referee_reanalysis/revised_forecaster.json (copied here as numbers so this file is self-contained).
"""
import datetime
import json
import sys

NGRAM = (-0.09618305026700236, 0.8134058678431364)
LAM = 0.6924039665645343
CODE_BIAS = 0.2011360393894477


def level(fc):
    if fc["family"] == "code":
        return fc["A0"]["23"] - CODE_BIAS, "code: A0(2^23) - code_bias"
    if fc["family"] == "english":
        return fc["A0"]["23"], "english: A0(2^23)"
    return NGRAM[0] + NGRAM[1] * fc["ref_stats"]["ngram23"], "n-gram level"


def forecast(F):
    out = {}
    for name, fc in F["corpora"].items():
        lv, how = level(fc)
        drop = LAM * (fc["S"]["23"] - fc["S"]["25"])
        out[name] = dict(family=fc["family"], level_rule=how, level_23=lv, drop_23_25=drop, F23=lv, F25=lv - drop,
                         c_ratio=fc["ref_stats"]["ref_full"] / fc["ref_stats"]["ngram23"])
    return out


def main():
    F = json.load(open(sys.argv[1]))
    out = dict(created_pt=datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"), forecaster="(ii) revised n-gram-level route",
               constants=dict(ngram=NGRAM, lam=LAM, code_bias=CODE_BIAS), source_forecast_i_created_utc=F.get("created_utc"),
               corpora=forecast(F))
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(len(out["corpora"]), "corpora ->", sys.argv[2])


if __name__ == "__main__":
    main()
