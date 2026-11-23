"""Key table for sensitivity (2026-11-05): each headline number, its paper interval, and its value under every
variant; flags a value outside the paper's own 95% interval. Reads sens.json, datadecide_floors.json, singlepass_floors.json,
results/code_fwd3m/numerics_*.json. Writes key_table.json and prints a markdown table."""
import json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = os.path.join(ROOT, "results/sensitivity")
S = json.load(open(os.path.join(R, "sens.json")))
DD = json.load(open(os.path.join(R, "datadecide_floors.json")))
SP = json.load(open(os.path.join(R, "singlepass_floors.json")))

H = {  # name: (getter on a scores dict, paper value key)
    "Table 1 code, data alone / two pilots": lambda s, f="none": s["table1_test80_code"][f"A0_over_A3_{f}"],
    "Table 1 English": lambda s, f="none": s["table1_test80_english"][f"A0_over_A3_{f}"],
    "Table 1 Wikipedia": lambda s, f="none": s["table1_test80_lang"][f"A0_over_A3_{f}"],
    "Table 1 all 80": lambda s, f="none": s["table1_test80_all"][f"A0_over_A3_{f}"],
    "Forward 45, revision / two pilots, all": lambda s, f="none": s["external45_all"][f"rev_over_A3_{f}"],
    "Forward 45, code": lambda s, f="none": s["external45_code"][f"rev_over_A3_{f}"],
    "Forward 45, web languages": lambda s, f="none": s["external45_lang"][f"rev_over_A3_{f}"],
    "Forward 45, web-language fall vs two pilots": lambda s, f="none": s["external45_lang"]["fall_ratio_rev_over_A3"],
    "Own fall orders 84 language falls (Spearman)": lambda s, f="none": dict(ratio=s["spearman_own_fall_84_languages"]["rho"], ci95=s["spearman_own_fall_84_languages"]["ci95"]),
    "Web-language fall line / kind mean (30)": lambda s, f="none": s["weblang30_fall_line"]["ratio"],
    "English fall line / kind mean (20)": lambda s, f="none": s["english20_fall_line"]["ratio"],
}
paper = S["info_floor"]["h960"]["scores"]
cols = [("info floor: none", S["info_floor"]["zero"]["scores"], "none"),
        ("info floor: fitted", S["info_floor"]["free"]["scores"], "none"),
        ("pilots floor: reference", paper, "ref_dev"),
        ("pilots floor: 0.5 n-gram", paper, "ngram_0.5"),
        ("seed 0 only", S["seed"]["s0_only"], "none"),
        ("seed 1 only", S["seed"]["s1_only"], "none"),
        ("early-stopping split", S["eval_split"]["scores_valA"], "none"),
        ("1,000 targets*", S["ref1000"]["scores_1000"], "none")]
rows = []
for name, g in H.items():
    p = g(paper)
    row = dict(headline=name, paper=p["ratio"], ci95=p["ci95"], variants={})
    uses_floor = "f=" in name or name.startswith(("Table 1", "Forward 45, revision", "Forward 45, code", "Forward 45, web languages"))
    for cn, sc, fl in cols:
        if fl != "none" and not uses_floor:
            continue
        try:
            v = g(sc, fl)["ratio"]
        except (KeyError, TypeError):
            try:
                v = g(sc)["ratio"] if fl == "none" else None
            except KeyError:
                v = None
            if fl != "none":
                v = None
        if v is not None:
            row["variants"][cn] = dict(value=v, outside=bool(v < p["ci95"][0] or v > p["ci95"][1]))
    if "Web-language fall line" in name or "Own fall" in name:   # 1,000-target rows differ; use the same-row comparison
        a, b = g(S["ref1000"]["scores_1000"])["ratio"], g(S["ref1000"]["scores_20000_same_rows"])["ratio"]
        row["variants"]["1,000 targets*"] = dict(value=a, same_rows_20000=b, outside=bool(abs(a - b) > (p["ci95"][1] - p["ci95"][0]) / 2))
    rows.append(row)
# rate rows
dd = DD["variants"]
rate_rows = [
    dict(headline="DataDecide: evaluation-set share of rate variance", paper=dd["none"]["share_eval"],
         variants={k: dict(value=v["share_eval"]) for k, v in dd.items()}),
    dict(headline="DataDecide: Spearman(gamma, set-mean rate), 11 sets", paper=dd["none"]["spearman_gamma_gpt2"], ci95=dd["none"]["spearman_gamma_gpt2_evalset_boot95"],
         variants={k: dict(value=v["spearman_gamma_gpt2"]) for k, v in dd.items()}),
    dict(headline="Single pass 40: Spearman(gamma, rate)", paper=SP["3.3M_40"]["all"]["floors"]["raw"]["rho_gamma"], ci95=SP["3.3M_40"]["all"]["floors"]["raw"]["ci95"],
         variants={k: dict(value=v["rho_gamma"], ci95=v["ci95"]) for k, v in SP["3.3M_40"]["all"]["floors"].items()}),
]
lr = S["learner_rates"]["all"]
rate_rows.append(dict(headline="Curve-forecast learner, 159 corpora: median local exponent 2^23->2^25",
                      paper=lr["local|none"]["median"], variants={f: dict(value=lr[f"local|{f}"]["median"], n=lr[f"local|{f}"]["n_defined"],
                                                                          rank_agreement_with_none=lr[f"local|{f}"].get("spearman_vs_none"))
                                                                   for f in ("none", "fitted", "reference", "ngram_half")}))
rate_rows.append(dict(headline="Curve-forecast learner, 159 corpora: Spearman(gamma, local exponent)",
                      paper=lr["local|none"]["spearman_gamma_h960"]["rho"], ci95=lr["local|none"]["spearman_gamma_h960"]["ci95"],
                      variants={f"{f} / gamma {gv}": dict(value=lr[f"local|{f}"][f"spearman_gamma_{gv}"]["rho"])
                                for f in ("none", "fitted", "reference", "ngram_half") for gv in ("h960", "zero", "free")}))
nl = json.load(open(os.path.join(ROOT, "results/code_fwd3m/numerics_learner.json")))
nr = json.load(open(os.path.join(ROOT, "results/code_fwd3m/numerics_ref.json")))
num = dict(learner_fp16_minus_bf16={k: v["diff"] for k, v in nl["runs"].items()},
           fall_diff={c: nl["runs"][f"{c}_b23_s0"]["diff"] - nl["runs"][f"{c}_b25_s0"]["diff"] for c in ("code_xslt", "code_coffeescript")},
           ref_x_diff=nr["x_fp16"] - nr["x_stored"], ref_full_diff=nr["ref_full_fp16"] - nr["ref_full_stored"])
out = dict(forecast_headlines=rows, rate_headlines=rate_rows, numerics=num, seed=dict(n=S["seed"]["n"], sd_single_run=S["seed"]["sd_single_run"]),
           eval_split={k: v for k, v in S["eval_split"].items() if not k.startswith("scores")},
           ref1000=dict(n=S["ref1000"]["n"], gamma_diff=S["ref1000"]["gamma_diff"], own_fall_diff=S["ref1000"]["own_fall_diff"], A0_25_diff=S["ref1000"]["A0_25_diff"]))
json.dump(out, open(os.path.join(R, "key_table.json"), "w"), indent=1)
hdr = "| headline | paper (95% CI) | " + " | ".join(c for c, _, _ in cols) + " |"
print(hdr); print("|" + "---|" * (2 + len(cols)))
for r in rows:
    cells = []
    for cn, _, _ in cols:
        v = r["variants"].get(cn)
        cells.append("" if v is None else (f"**{v['value']:.2f}**" if v["outside"] else f"{v['value']:.2f}"))
    print(f"| {r['headline']} | {r['paper']:.2f} ({r['ci95'][0]:.2f}-{r['ci95'][1]:.2f}) | " + " | ".join(cells) + " |")
print(json.dumps(num, indent=1))
