"""Compare the numbers quoted in the article, and a few stored results it does not quote, with the values in the shipped
(and, under reproduce.sh, regenerated) result files. Each check carries its expected value; the Table 2 rows are in
verify/table_rows.txt.

    python verify/paper_checks.py [--root DIR] [--list]   # DIR: a regenerated tree (verify/regenerate.py --work DIR)

Each check reads one value from a result file, rounds it to the number of decimals printed in the paper and compares it
with the printed value. Exit status 1 if any check fails.
"""
import json
import math
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROWS = os.path.join(ROOT, "verify", "table_rows.txt")
R = "results/"
FWD = R + "fwd_weblang/score.json"
FWDZ = R + "fwd_weblang/fall_line.json"
ENG = R + "fwd_english/score.json"
ENGZ = R + "fwd_english/fall_line.json"
C25 = R + "code25m/score.json"
EB = R + "extbench/score.json"
CF = R + "reanalysis/compression_fall.json"
SVP = R + "reanalysis/shape_vs_pooled.json"
SWK = R + "reanalysis/shape_within_kind.json"
S15 = R + "curve_forecast/stronger_ref/score.json"
S7 = R + "curve_forecast/stronger_ref7b/score.json"
S14 = R + "curve_forecast/stronger_ref14b/score_half500.json"
NC = R + "curve_forecast/new_corpora/analysis.json"
SB = R + "reanalysis/stronger_baselines.json"
RSC = R + "reanalysis/ref_subset_cost.json"
CFR = R + "reanalysis/compute_frontier.json"
RS = R + "rate_singlepass/analysis.json"
RS25 = R + "rate_singlepass25m/analysis.json"
RHM = R + "rhm/analysis.json"
MW = R + "reanalysis/mismatch_vs_word.json"

INC = R + "infocurve_increment/increment.json"
IFL = R + "increment_floor/score.json"
APP = R + "applicability/score.json"
TR = R + "task_reversals/reversals_k2.json"
OSR = R + "opensciref_forecast/scores.json"
OSR24 = R + "curve_forecast/opensciref/forecast_inputs.json"
CM = R + "costmatch/summary.json"
PS = R + "sensitivity/key_table.json"
DDF = R + "sensitivity/datadecide_floors.json"
T3 = ["pooled", "three_sets"]
B25 = ["summary", "budget_2_25"]

# (id, where in the paper, file, key path, value as printed)
CHECKS = [
    # Results, "Whole learning curves": forward test on third-party lists (Methods, "Forward test on third-party lists")
    ("fwd51.lang_unrevised", "Methods: unrevised forecast on web languages 0.62", EB, ["families", "lang", "rms", "e25", "i"], "0.62"),
    ("fwd51.lang_revised", "Methods: revision cuts the error to 0.19", EB, ["families", "lang", "rms", "e25", "ii"], "0.19"),
    ("fwd51.lang_pilots", "Methods: two pilot runs 0.16", EB, ["families", "lang", "rms", "e25", "A3"], "0.16"),
    ("fwd51.all_revised", "Methods: 0.97 for the revision over all corpora", EB, ["families", "all", "rms", "e25", "ii"], "0.97"),
    ("fwd51.all_pilots", "Methods: against 0.81 for two pilots", EB, ["families", "all", "rms", "e25", "A3"], "0.81"),
    ("fwd51.code_revised", "Methods: 1.75 on code", EB, ["families", "code", "rms", "e25", "ii"], "1.75"),
    ("fwd51.code_pilots", "Methods: against 0.85 on code", EB, ["families", "code", "rms", "e25", "A3"], "0.85"),
    ("fwd51.ratio_all", "Methods: revision/two pilots 1.20", EB, ["families", "all", "ratios", "e25:ii/A3", "ratio"], "1.20"),
    ("fwd51.ratio_all_lo", "Methods: (0.64-", EB, ["families", "all", "ratios", "e25:ii/A3", "ci95", 0], "0.64"),
    ("fwd51.ratio_all_hi", "Methods: -2.69)", EB, ["families", "all", "ratios", "e25:ii/A3", "ci95", 1], "2.69"),
    ("fwd51.ratio_code", "Methods: 2.06 on code", EB, ["families", "code", "ratios", "e25:ii/A3", "ratio"], "2.06"),
    ("fwd51.ratio_code_ci", "Methods: (0.66-3.63)", EB, ["families", "code", "ratios", "e25:ii/A3", "ci95", 1], "3.63"),
    ("fwd51.ratio_eng", "Methods: 0.62 on English", EB, ["families", "english", "ratios", "e25:ii/A3", "ratio"], "0.62"),
    ("fwd51.ratio_eng_lo", "Methods: (0.47-", EB, ["families", "english", "ratios", "e25:ii/A3", "ci95", 0], "0.47"),
    ("fwd51.ratio_lang", "Methods: 1.18 on languages", EB, ["families", "lang", "ratios", "e25:ii/A3", "ratio"], "1.18"),
    ("fwd51.ratio_lang_hi", "Methods: (0.68-2.12)", EB, ["families", "lang", "ratios", "e25:ii/A3", "ci95", 1], "2.12"),
    ("fwd51.ratio_pool", "Methods: against one pilot with the pooled exponent 1.55", EB, ["families", "all", "ratios", "e25:ii/X_pool1", "ratio"], "1.55"),
    ("fwd51.ratio_p2023", "Methods: against pilots at 2^20 and 2^23 2.35", EB, ["families", "all", "ratios", "e25:ii/P2023", "ratio"], "2.35"),
    ("fwd51.ratio_p2023_ci", "Methods: (1.31-4.53)", EB, ["families", "all", "ratios", "e25:ii/P2023", "ci95", 1], "4.53"),
    ("fwd51.unrevised_ratio", "Methods: four times two pilots (3.98", EB, ["families", "lang", "ratios", "e25:i/A3", "ratio"], "3.98"),
    ("fwd51.unrevised_ratio_ci", "Methods: 2.59-6.77)", EB, ["families", "lang", "ratios", "e25:i/A3", "ci95", 1], "6.77"),
    ("fwd51.lang_level", "Methods: level 0.16 on languages", EB, ["families", "lang", "rms", "level", "ii"], "0.16"),
    ("fwd51.lang_fall", "Methods: fall 0.054 on languages", EB, ["families", "lang", "rms", "drop", "ii"], "0.054"),
    ("fwd51.lang_level_A3", "Methods: against 0.14", EB, ["families", "lang", "rms", "level", "A3"], "0.14"),
    ("fwd51.lang_fall_A3", "Methods: and 0.048", EB, ["families", "lang", "rms", "drop", "A3"], "0.048"),
    ("fwd51.lang_fall_ratio", "Methods: fall ratio 1.13", EB, ["families", "lang", "ratios", "drop:ii/A3", "ratio"], "1.13"),
    ("fwd51.code_level", "Methods: 1.51 on code (level)", EB, ["families", "code", "rms", "level", "ii"], "1.51"),
    ("fwd51.code_fall", "Methods: 0.47 on code (fall)", EB, ["families", "code", "rms", "drop", "ii"], "0.47"),
    ("fwd51.all_level", "Methods: 0.77 over all corpora (level)", EB, ["families", "all", "rms", "level", "ii"], "0.77"),
    ("fwd51.all_fall", "Methods: 0.49 over all corpora (fall)", EB, ["families", "all", "rms", "drop", "ii"], "0.49"),
    ("fwd51.seed_sd", "Methods: single-run s.d. 0.013", EB, ["seed_noise_2_25", "sd_single_run"], "0.013"),
    # forward web-language test of the frozen line
    ("fwdweb.line_a", "Methods: fall = 0.065 + ...", FWDZ, ["calibration", "a"], "0.065"),
    ("fwdweb.line_b", "Methods: ... + 0.356 x own fall", FWDZ, ["calibration", "b"], "0.356"),
    ("fwdweb.mean_fall", "Methods: mean fall of 0.164", FWDZ, ["calibration", "kind_mean_fall"], "0.164"),
    ("fwdweb.loo", "Methods: leave-one-out ratio 0.68", FWD, ["calibration", "loo_ratio_exploratory", 0], "0.68"),
    ("fwdweb.loo_ci", "Methods: (0.50-0.92)", FWD, ["calibration", "loo_ratio_exploratory", 2], "0.92"),
    ("fwdweb.rms_line", "Methods: r.m.s. error 0.032", FWD, ["fall_rms", "own_cal"], "0.032"),
    ("fwdweb.rms_mean", "Methods: against 0.043", FWD, ["fall_rms", "kind_mean"], "0.043"),
    ("fwdweb.ratio", "Results/Methods: ratio 0.75", FWD, ["primary", "ratio"], "0.75"),
    ("fwdweb.ratio_lo", "Results/Methods: (ratio 0.75, 0.54-", FWD, ["primary", "ci95", 0], "0.54"),
    ("fwdweb.ratio_hi", "Results/Methods: -0.98", FWD, ["primary", "ci95", 1], "0.98"),
    ("fwdweb.pilots", "Methods: two pilot runs (0.063)", FWD, ["fall_rms", "A3"], "0.063"),
    ("fwdweb.first8", "Methods: after 8 corpora (ratio 0.63", FWD, ["subsets", "first8 (pods 1-2, morning)", "primary", "ratio"], "0.63"),
    ("fwdweb.first8_ci", "Methods: 0.37-1.02)", FWD, ["subsets", "first8 (pods 1-2, morning)", "primary", "ci95", 1], "1.02"),
    ("fwdweb.new22", "Methods: the other 22 (0.78", FWD, ["subsets", "new (pods 5-9)", "primary", "ratio"], "0.78"),
    ("fwdweb.new22_ci", "Methods: 0.52-1.06)", FWD, ["subsets", "new (pods 5-9)", "primary", "ci95", 1], "1.06"),
    ("fwdweb.spearman", "Methods: rank correlation 0.66", FWD, ["spearman_own_vs_fall", "rho"], "0.66"),
    ("fwdweb.spearman_ci", "Methods: (0.35-0.86)", FWD, ["spearman_own_vs_fall", "ci95", 1], "0.86"),
    ("fwdweb.vs_pilots", "Methods: ratio against two pilot runs 0.51", FWD, ["context", "own_cal/A3", "ratio"], "0.51"),
    ("fwdweb.vs_pilots_ci", "Methods: (0.42-0.66)", FWD, ["context", "own_cal/A3", "ci95", 1], "0.66"),
    ("fwdweb.seed_sd", "Methods: single-run s.d. 0.011", FWD, ["seed_noise_2_25", "sd_single_run"], "0.011"),
    ("fwdweb.e25_line", "Appendix: 0.135", FWD, ["e25_rms_forecast_level", "own_cal"], "0.135"),
    ("fwdweb.e25_mean", "Appendix: against 0.132", FWD, ["e25_rms_forecast_level", "kind_mean"], "0.132"),
    # English forward test and code at 25M
    ("eng.ratio", "Results/Methods: 0.84", ENG, ["primary_clean", "primary", "ratio"], "0.84"),
    ("eng.ratio_lo", "Results/Methods: (0.75-", ENG, ["primary_clean", "primary", "ci95", 0], "0.75"),
    ("eng.ratio_hi", "Results/Methods: -0.92)", ENG, ["primary_clean", "primary", "ci95", 1], "0.92"),
    ("eng.all24", "Methods: all 24, 0.85", ENG, ["all_scored", "primary", "ratio"], "0.85"),
    ("eng.all24_ci", "Methods: (0.72-0.96)", ENG, ["all_scored", "primary", "ci95", 1], "0.96"),
    ("eng.spearman", "Methods: rank correlation 0.65", ENG, ["primary_clean", "spearman"], "0.65"),
    ("eng.vs_pilots", "Methods: against two pilot runs 1.03", ENG, ["primary_clean", "context", "own_cal/A3", "ratio"], "1.03"),
    ("eng.vs_pilots_ci", "Methods: (0.69-1.57)", ENG, ["primary_clean", "context", "own_cal/A3", "ci95", 1], "1.57"),
    ("eng.own_raw", "Methods: unscaled own fall 0.054", ENG, ["primary_clean", "fall_rms", "own_raw"], "0.054"),
    ("eng.line", "Methods: against 0.072 for the line", ENG, ["primary_clean", "fall_rms", "own_cal"], "0.072"),
    ("code25.capacity", "Methods: 1.83 times the reference model's", C25, ["kinds", "code", "capacity", "L25_over_ref_25M"], "1.83"),
    ("code25.capacity_small", "Methods: (1.92 for the small learner)", C25, ["kinds", "code", "capacity", "L25_over_ref_3p3M"], "1.92"),
    ("code25.ratio", "Methods: ties the mean fall (1.03", C25, ["kinds", "code", "primary_R1_clean", "ratio"], "1.03"),
    ("code25.ratio_ci", "Methods: 0.97-1.11", C25, ["kinds", "code", "primary_R1_clean", "ci95", 1], "1.11"),
    ("code25.spearman", "Methods: rank correlation 0.13", C25, ["kinds", "code", "primary_R1_clean", "spearman"], "0.13"),
    ("code25.web_ratio", "Methods: web control 0.92", C25, ["kinds", "web", "primary_R1_clean", "ratio"], "0.92"),
    ("code25.web_ci", "Methods: (0.69-1.27", C25, ["kinds", "web", "primary_R1_clean", "ci95", 1], "1.27"),
    ("code25.web_spearman", "Methods: rank correlation 0.49", C25, ["kinds", "web", "primary_R1_clean", "spearman"], "0.49"),
    # compression and n-gram lines for the fall (Methods, English forward test paragraph)
    ("cfall.eng_ngram", "Methods: 0.085 (n-gram line, English)", CF, ["families", "english", "fall_rms", "ngram23_line"], "0.085"),
    ("cfall.eng_h0", "Methods: 0.086 (order-0 line, English)", CF, ["families", "english", "fall_rms", "h0_line"], "0.086"),
    ("cfall.eng_ratio", "Methods: own line against n-gram line 0.85", CF, ["families", "english", "ratios", "own_line/ngram23_line", "ratio"], "0.85"),
    ("cfall.eng_ratio_lo", "Methods: (0.75-", CF, ["families", "english", "ratios", "own_line/ngram23_line", "ci95", 0], "0.75"),
    ("cfall.eng_ratio_hi", "Methods: -0.94)", CF, ["families", "english", "ratios", "own_line/ngram23_line", "ci95", 1], "0.94"),
    ("cfall.web_ngram", "Methods: 0.022 (n-gram line, web)", CF, ["families", "web", "fall_rms", "ngram23_line"], "0.022"),
    ("cfall.web_h0", "Methods: 0.026 (order-0 line, web)", CF, ["families", "web", "fall_rms", "h0_line"], "0.026"),
    ("cfall.web_ratio", "Methods: 1.43", CF, ["families", "web", "ratios", "own_line/ngram23_line", "ratio"], "1.43"),
    ("cfall.web_ratio_ci", "Methods: (1.00-2.08)", CF, ["families", "web", "ratios", "own_line/ngram23_line", "ci95", 1], "2.08"),
    ("cfall.web_level", "Methods: Spearman 0.89 with the loss at 2^23", CF, ["families", "web", "spearman_context_test", "L23_vs_fall"], "0.89"),
    ("cfall.web_both", "Methods: adding the own fall leaves 0.022", CF, ["families", "web", "fall_rms", "own+ngram23_line"], "0.022"),
    # the appendix (fall from 2^23 to 2^25 at a matched level)
] + [
    (f"supp2.{fam}.{col}", f"Appendix table, {fam}, {col}", SVP, T3 + [fam] + key, val)
    for fam, vals in (("code", ("0.28", "0.37", "0.26", "0.41", "0.29", "0.28")),
                      ("english", ("0.57", "0.52", "0.47", "0.48", "0.57", "0.53")),
                      ("lang", ("0.052", "0.119", "0.127", "0.114", "0.055", "0.060")),
                      ("all", ("0.29", "0.30", "0.26", "0.30", "0.29", "0.28")))
    for col, key, val in zip(("sd", "own", "pooled_exp", "dev_fall", "fineweb", "two_pilots"),
                             (["fall_sd"], ["fall_rms", "primary", "own_S"], ["fall_rms", "primary", "pool_exp"],
                              ["fall_rms", "primary", "dev_fall_log"], ["fall_rms", "primary", "fw_info"],
                              ["fall_rms", "primary", "A3"]), vals)
] + [
    ("supp2.n_all", "Appendix: 143 corpora", SVP, T3 + ["all", "n"], "143"),
    # the appendix, within-kind calibration
    ("swk.web", "Appendix: web languages (0.68", SWK, ["kinds", "web", "loo", "ratio"], "0.68"),
    ("swk.web_ci", "Appendix: 0.50-0.92)", SWK, ["kinds", "web", "loo", "ci95", 0], "0.50"),
    ("swk.wiki", "Appendix: Wikipedia (0.90", SWK, ["kinds", "wiki", "loo", "ratio"], "0.90"),
    ("swk.wiki_ci", "Appendix: 0.80-1.00)", SWK, ["kinds", "wiki", "loo", "ci95", 1], "1.00"),
    ("swk.english", "Appendix: English (1.18)", SWK, ["kinds", "english", "loo", "ratio"], "1.18"),
    ("swk.code", "Appendix: code (1.03)", SWK, ["kinds", "code", "loo", "ratio"], "1.03"),
    ("swk.web_spearman", "Appendix: 0.80 on the external web languages", SWK, ["kinds", "web", "spearman"], "0.80"),
    ("swk.wiki_A", "Appendix: ties the mean on the 24 other languages (0.98", SWK, ["kinds", "wiki", "A", "ratio"], "0.98"),
    ("swk.wiki_A_ci", "Appendix: 0.77-1.15)", SWK, ["kinds", "wiki", "A", "ci95", 1], "1.15"),
    # Table 1 rows with larger references and the appendix
    ("t1.15b.code", "Table 1, Qwen2.5-1.5B row, code", S15, B25 + ["code", "rms", "sub15"], "0.22"),
    ("t1.15b.english", "Table 1, Qwen2.5-1.5B row, English", S15, B25 + ["english", "rms", "sub15"], "0.10"),
    ("t1.15b.wiki", "Table 1, Qwen2.5-1.5B row, Wikipedia", S15, B25 + ["wiki47", "rms", "sub15"], "0.31"),
    ("t1.15b.all", "Table 1, Qwen2.5-1.5B row, all", S15, B25 + ["all80", "rms", "sub15"], "0.26"),
    ("t1.7b.code", "Table 1, Qwen2.5-7B row, code", S7, B25 + ["code", "rms", "sub7"], "0.24"),
    ("t1.7b.english", "Table 1, Qwen2.5-7B row, English", S7, B25 + ["english", "rms", "sub7"], "0.10"),
    ("t1.7b.wiki", "Table 1, Qwen2.5-7B row, Wikipedia", S7, B25 + ["wiki47", "rms", "sub7"], "0.20"),
    ("t1.7b.all", "Table 1, Qwen2.5-7B row, all", S7, B25 + ["all80", "rms", "sub7"], "0.19"),
    ("t1.14b.code", "Table 1, Qwen3-14B row, code", S14, B25 + ["code", "rms", "h14"], "0.29"),
    ("t1.14b.english", "Table 1, Qwen3-14B row, English", S14, B25 + ["english", "rms", "h14"], "0.09"),
    ("t1.14b.wiki", "Table 1, Qwen3-14B row, Wikipedia", S14, B25 + ["wiki47", "rms", "h14"], "0.15"),
    ("t1.14b.all", "Table 1, Qwen3-14B row, all", S14, B25 + ["all80", "rms", "h14"], "0.17"),
    ("sn6.15b_wiki_from", "Appendix: falls from 0.39 (0.5B, same 1,000 targets)", S15, B25 + ["wiki47", "rms", "sub05"], "0.39"),
    ("sn6.15b_wiki_to", "Appendix: to 0.31", S15, B25 + ["wiki47", "rms", "sub15"], "0.31"),
    ("sn6.15b_wiki_ratio", "Appendix: ratio 0.81", S15, B25 + ["wiki47", "sub15_vs_sub05", "ratio"], "0.81"),
    ("sn6.15b_wiki_ratio_ci", "Appendix: 0.75-0.86", S15, B25 + ["wiki47", "sub15_vs_sub05", "ratio_ci95", 1], "0.86"),
    ("sn6.15b_further", "Appendix: 16 further languages 0.86", S15, B25 + ["stage3_16", "sub15_vs_sub05", "ratio"], "0.86"),
    ("sn6.15b_all", "Appendix: 0.262 over all 80", S15, B25 + ["all80", "rms", "sub15"], "0.262"),
    ("sn6.15b_all_A3", "Appendix: against 0.269", S15, B25 + ["all80", "rms", "A3"], "0.269"),
    ("sn6.15b_all_ratio", "Appendix: ratio 0.98", S15, B25 + ["all80", "sub15_vs_A3", "ratio"], "0.98"),
    ("sn6.15b_all_ratio_ci", "Appendix: 0.64-1.69", S15, B25 + ["all80", "sub15_vs_A3", "ratio_ci95", 1], "1.69"),
    ("sn6.7b_wiki_ratio", "Appendix: ratio to the 1.5B 0.63", S7, B25 + ["wiki47", "sub7_vs_sub15", "ratio"], "0.63"),
    ("sn6.7b_wiki_ratio_ci", "Appendix: 0.57-0.69", S7, B25 + ["wiki47", "sub7_vs_sub15", "ratio_ci95", 0], "0.57"),
    ("sn6.7b_further", "Appendix: 0.25 on the 16 further languages", S7, B25 + ["stage3_16", "rms", "sub7"], "0.25"),
    ("sn6.7b_further_ratio", "Appendix: (0.62, 0.51-0.69)", S7, B25 + ["stage3_16", "sub7_vs_sub15", "ratio"], "0.62"),
    ("sn6.7b_all_ratio", "Appendix: 0.19 against 0.27 (ratio 0.70", S7, B25 + ["all80", "sub7_vs_A3", "ratio"], "0.70"),
    ("sn6.7b_all_ratio_ci", "Appendix: 0.47-1.18)", S7, B25 + ["all80", "sub7_vs_A3", "ratio_ci95", 1], "1.18"),
    ("sn6.7b_wiki_vs_A3", "Appendix: ratio 1.94", S7, B25 + ["wiki47", "sub7_vs_A3", "ratio"], "1.94"),
    ("sn6.7b_wiki_vs_A3_ci", "Appendix: 1.37-2.72", S7, B25 + ["wiki47", "sub7_vs_A3", "ratio_ci95", 1], "2.72"),
    ("sn6.14b_wiki_7b", "Appendix: from 0.19 with the 7B (500 targets)", S14, B25 + ["wiki47", "rms", "h7"], "0.19"),
    ("sn6.14b_wiki_ratio", "Appendix: ratio 0.78", S14, B25 + ["wiki47", "h14_vs_h7", "ratio"], "0.78"),
    ("sn6.14b_wiki_ratio_ci", "Appendix: 0.65-0.995", S14, B25 + ["wiki47", "h14_vs_h7", "ratio_ci95", 1], "0.995"),
    ("sn6.14b_further_7b", "Appendix: further languages from 0.26", S14, B25 + ["stage3_16", "rms", "h7"], "0.26"),
    ("sn6.14b_further", "Appendix: to 0.16", S14, B25 + ["stage3_16", "rms", "h14"], "0.16"),
    ("sn6.14b_further_ratio", "Appendix: (0.61, 0.48-0.94)", S14, B25 + ["stage3_16", "h14_vs_h7", "ratio"], "0.61"),
    ("sn6.14b_further_A3", "Appendix: extrapolation (0.12", S14, B25 + ["stage3_16", "rms", "A3"], "0.12"),
    ("sn6.14b_further_vs_A3", "Appendix: ratio 1.31, 0.69-3.22", S14, B25 + ["stage3_16", "h14_vs_A3", "ratio_ci95", 1], "3.22"),
    ("sn6.14b_all_ratio", "Appendix: 0.17 against 0.27 (ratio 0.63", S14, B25 + ["all80", "h14_vs_A3", "ratio"], "0.63"),
    ("sn6.14b_all_ratio_ci", "Appendix: 0.45-0.995)", S14, B25 + ["all80", "h14_vs_A3", "ratio_ci95", 1], "0.995"),
    ("sn6.14b_wiki_vs_A3", "Appendix: ratio 1.42", S14, B25 + ["wiki47", "h14_vs_A3", "ratio"], "1.42"),
    ("sn6.14b_wiki_vs_A3_ci", "Appendix: 1.09-1.93", S14, B25 + ["wiki47", "h14_vs_A3", "ratio_ci95", 1], "1.93"),
    ("sn6.14b_code_7b", "Appendix: code 0.25 to 0.29", S14, B25 + ["code", "rms", "h7"], "0.25"),
    ("sn6.14b_eng_7b", "Appendix: English 0.10 to 0.09", S14, B25 + ["english", "rms", "h7"], "0.10"),
    # Methods, "Forecasts made before training" (18 corpora)
    ("nc18.code_A0", "Methods: 0.078 on the five code corpora", NC, ["blocks", "25", "code", "rms", "A0"], "0.078"),
    ("nc18.code_A3", "Methods: against 0.217 for two pilots", NC, ["blocks", "25", "code", "rms", "A3"], "0.217"),
    ("nc18.code_ratio", "Methods: ratio 0.36", NC, ["blocks", "25", "code", "A0_over_A3", "ratio"], "0.36"),
    ("nc18.code_ratio_ci", "Methods: 0.11-1.07", NC, ["blocks", "25", "code", "A0_over_A3", "ci95", 1], "1.07"),
    ("nc18.code_pool", "Methods: 0.340 for one pilot with the pooled exponent", NC, ["blocks", "25", "code", "rms", "X_pool1"], "0.340"),
    ("nc18.code_pool_ratio", "Methods: (0.23, 0.08-0.44", NC, ["blocks", "25", "code", "A0_over_X_pool1", "ratio"], "0.23"),
    ("nc18.eng_A0", "Methods: On English it is 0.077", NC, ["blocks", "25", "english", "rms", "A0"], "0.077"),
    ("nc18.eng_A3", "Methods: against 0.067", NC, ["blocks", "25", "english", "rms", "A3"], "0.067"),
    ("nc18.eng_pool", "Methods: and 0.228", NC, ["blocks", "25", "english", "rms", "X_pool1"], "0.228"),
    ("nc18.wiki_A0", "Methods: on Wikipedia 0.427", NC, ["blocks", "25", "wiki", "rms", "A0"], "0.427"),
    ("nc18.wiki_A3", "Methods: against 0.170", NC, ["blocks", "25", "wiki", "rms", "A3"], "0.170"),
    ("nc18.wiki_pool", "Methods: and 0.089", NC, ["blocks", "25", "wiki", "rms", "X_pool1"], "0.089"),
    ("nc18.code_23", "Methods: too high at 2^23 (r.m.s. 0.32", NC, ["blocks", "23", "code", "rms", "A0"], "0.32"),
    ("nc18.code_23_A3", "Methods: against 0.22 for two pilots)", NC, ["blocks", "23", "code", "rms", "A3"], "0.22"),
    # Methods, forecast compute; the appendix and Table 1 pooled-exponent row
    ("cost.sub1000_code", "Methods: 1,000 targets, code 0.21", RSC, ["summary", "code", "rms_sub1000"], "0.21"),
    ("cost.sub1000_eng", "Methods: English 0.10", RSC, ["summary", "english", "rms_sub1000"], "0.10"),
    ("cost.p2023_code", "Methods: pilots at 2^20 and 2^23, 0.17 on code", SB, ["rms_2_25", "code", "A3_20_23", "rms"], "0.17"),
    ("t1.pool.code", "Table 1, one pilot with pooled exponent, code", SB, ["rms_2_25", "code", "X_pool1", "rms"], "0.52"),
    ("t1.pool.english", "Table 1, one pilot with pooled exponent, English", SB, ["rms_2_25", "english", "X_pool1", "rms"], "0.19"),
    ("t1.pool.wiki", "Table 1, one pilot with pooled exponent, Wikipedia", SB, ["rms_2_25", "wiki", "X_pool1", "rms"], "0.08"),
    ("t1.pool.all", "Table 1, one pilot with pooled exponent, all", SB, ["rms_2_25", "all", "X_pool1", "rms"], "0.25"),
    ("sn8.floor_ref", "Appendix: floor proportional to the reference (0.62)", SB, ["rms_2_25", "code", "F_ref", "rms"], "0.62"),
    ("sn8.chinchilla", "Appendix: Chinchilla's exponent (0.87)", SB, ["rms_2_25", "code", "F_lit", "rms"], "0.87"),
    ("sn8.log", "Appendix: a line in log D (1.07)", SB, ["rms_2_25", "code", "LOG", "rms"], "1.07"),
    ("sn8.pool_vs_A3", "Appendix: ratio 0.91", SB, ["ratios_2_25", "all", "X_pool1/A3", "ratio"], "0.91"),
    ("sn8.pool_vs_A3_ci", "Appendix: (0.81-1.12)", SB, ["ratios_2_25", "all", "X_pool1/A3", "ci95", 1], "1.12"),
    ("sn8.A2_all", "Appendix: one pilot onto the information curve 0.31", SB, ["rms_2_25", "all", "A2", "rms"], "0.31"),
    ("sn8.A3_all", "Appendix: against 0.27", SB, ["rms_2_25", "all", "A3", "rms"], "0.27"),
    # the appendix, single-pass rates
    ("sp.raw_all", "Appendix: Spearman with gamma -0.25", RS, ["sets", "all", "raw", "gamma_ref", "spearman"], "-0.25"),
    ("sp.raw_all_lo", "Appendix: (-0.53 to", RS, ["sets", "all", "raw", "gamma_ref", "ci", 0], "-0.53"),
    ("sp.raw_all_hi", "Appendix: 0.08)", RS, ["sets", "all", "raw", "gamma_ref", "ci", 1], "0.08"),
    ("sp.raw_lang", "Appendix: +0.03 on the 20 languages", RS, ["sets", "language", "raw", "gamma_ref", "spearman"], "0.03"),
    ("sp.raw_eng", "Appendix: -0.51 on the 20 English domains", RS, ["sets", "english", "raw", "gamma_ref", "spearman"], "-0.51"),
    ("sp.fit3", "Appendix: above a floor fitted per corpus +0.33", RS, ["sets", "all", "fit3", "gamma_ref", "spearman"], "0.33"),
    ("sp.fit3_ci", "Appendix: (0.03 to 0.59)", RS, ["sets", "all", "fit3", "gamma_ref", "ci", 1], "0.59"),
    ("sp.fit3_25m", "Appendix: with the larger learner +0.05", RS25, ["sets", "all", "fit3", "gamma_ref", "spearman"], "0.05"),
    ("sp.raw_25m", "Appendix: raw +0.05", RS25, ["sets", "all", "raw", "gamma_ref", "spearman"], "0.05"),
    ("sp.raw_25m_ci", "Appendix: (-0.43 to 0.51)", RS25, ["sets", "all", "raw", "gamma_ref", "ci", 1], "0.51"),
    ("sp.lzma_all", "Appendix: lzma -0.57 over all 40", RS, ["sets", "all", "raw", "lzma_bpb", "spearman"], "-0.57"),
    ("sp.lzma_eng", "Appendix: -0.80 on English", RS, ["sets", "english", "raw", "lzma_bpb", "spearman"], "-0.80"),
    ("sp.lzma_25m", "Appendix: +0.03 on the languages (larger learner)", RS25, ["sets", "all", "raw", "lzma_bpb", "spearman"], "0.03"),
    # Results and the appendix / Table 3, Random Hierarchy Model
    ("rhm.rms_law", "Results: r.m.s. error 0.049", RHM, ["_summary", "law_closed", "rms"], "0.049"),
    ("rhm.rms_cag", "Results: against 0.135", RHM, ["_summary", "cagnetta", "rms"], "0.135"),
    ("rhm.rms_law14", "Appendix: 0.123 for the law (2^14-2^20)", RHM, ["_summary_from14", "law_closed", "rms"], "0.123"),
    ("rhm.rms_cag14", "Appendix: 0.147 for the formula", RHM, ["_summary_from14", "cagnetta", "rms"], "0.147"),
    ("rhm.rms_210", "Appendix: single price r.m.s. 0.256", RHM, ["_summary", "law_finite_210", "rms"], "0.256"),
] + [
    (f"supp3.{g}.{col}", f"Appendix table, {g}, {col}", RHM, [g] + key, val)
    for g, vals in (("v8m2", ("0.88", "1.05", "0.47", "0.42", "0.47", "0.46", "0.48", "0.54")),
                    ("v4m2", ("0.83", "1.09", "0.45", "0.38", "0.51", "0.50", "0.53", "0.61")),
                    ("v8m4", ("0.52", "2.10", "0.34", "0.12", "0.26", "0.25", "0.28", "0.25")),
                    ("v16m8", ("0.34", "2.92", "0.25", "0.06", "0.24", "0.23", "0.24", "0.10")))
    for col, key, val in zip(("gamma", "beta", "law", "cagnetta", "measured", "measured_lo", "measured_hi", "measured_14_20"),
                             (["gamma"], ["beta"], ["pred", "law_closed"], ["pred", "cagnetta"], ["alpha_all"],
                              ["alpha_all_ci", 0], ["alpha_all_ci", 1], ["alpha_from14"]), vals)
] + [
    # the appendix, mismatch against word overlap
    ("sn1.bigram_minus_word", "Appendix: (-0.06,", MW, ["datadecide", "all11", "differences", "bigram minus word unigram", "pairwise"], "-0.06"),
    ("sn1.bigram_minus_word_ci", "Appendix: -0.13 to 0.09)", MW, ["datadecide", "all11", "differences", "bigram minus word unigram", "pairwise_95", 1], "0.09"),
    ("sn1.pearson", "Appendix: Pearson 0.76", MW, ["datadecide", "all11", "raw", "pearson_bigram_word"], "0.76"),
    ("sn1.both", "Appendix: one map with both reaches 73%", MW, ["datadecide", "all11", "rows", "bigram + word", "pairwise"], "0.73"),
    ("sn1.both_minus_word", "Appendix: both minus word -0.07", MW, ["datadecide", "all11", "differences", "bigram + word minus word unigram", "pairwise"], "-0.07"),
    ("sn1.both_minus_word_lo", "Appendix: (-0.20 to", MW, ["datadecide", "all11", "differences", "bigram + word minus word unigram", "pairwise_95", 0], "-0.20"),
    ("sn1.ood_word", "Appendix: word overlap 52% of pairs", MW, ["datadecide", "ood4", "rows", "word unigram", "pairwise"], "0.52"),
    ("sn1.ood_word_rho", "Appendix: Spearman -0.06", MW, ["datadecide", "ood4", "rows", "word unigram", "spearman"], "-0.06"),
    ("sn1.ood_bigram", "Appendix: the mismatch orders 67%", MW, ["datadecide", "ood4", "rows", "bigram", "pairwise"], "0.67"),
    ("sn1.ood_bigram_rho", "Appendix: Spearman 0.47", MW, ["datadecide", "ood4", "rows", "bigram", "spearman"], "0.47"),
    ("sn1.ood_mae_diff", "Appendix: mean absolute error difference -0.025", MW, ["datadecide", "ood4", "differences", "bigram minus word unigram", "mae"], "-0.025"),
    ("sn1.ood_mae_diff_ci", "Appendix: (-0.062 to -0.001)", MW, ["datadecide", "ood4", "differences", "bigram minus word unigram", "mae_95", 1], "-0.001"),
]


# ---------------------------------------------------------------- further checks
def ci(*k):
    return list(k)

def _osr(size):
    return ["variants", "sub1000", size]

def _cm(s, f, m):
    return ["blocks", s, f, "methods", m]

def _ps(head, var=None):
    def f(d):
        h = next(x for x in d["forecast_headlines"] if x["headline"] == head)
        return h["paper"] if var is None else h["variants"][var]["value"]
    return f

def _bias_range(kind, which):
    def f(d):
        b = d["families"][kind]["heldout_bias"]
        v = [abs(b[k]) for k in ("kind_mean", "ngram23", "h0", "gain", "ref", "own")]
        return min(v) if which == "min" else max(v)
    return f

FDG = R + "reanalysis/fall_diagnosis.json"

def _refdocs(c):
    def f(d):
        base = os.path.join(ROOT, d["per_corpus"][c]["dir"])
        for sub in ("pod", os.path.join("raw", "curve_forecast")):
            p = os.path.join(base, sub, c, "refcurve", "SmolLM2-135M.json")
            if os.path.exists(p):
                return int(json.load(open(p))["n_docs"])
        raise KeyError(c)
    return f

def _maxabs(m):
    return lambda d: max(abs(v["err"][m]) for v in d["per_corpus"].values())

E, W, POOL = ["families", "english"], ["families", "web"], ["pooled"]
MORE = [
    # Results, public rankings: swaps between benchmarks (Methods, Benchmark-text test; appendix table)
    ("tr.pairs", "Results: the 145 recipe pairs that swap", TR, ["counts", "reversing_pairs"], "145"),
    ("tr.cases", "Methods: 899 for 145 pairs", TR, ["counts", "reversal_instances"], "899"),
    ("tr.flips", "Methods: the 1,245 swaps", TR, ["counts", "flips"], "1,245"),
    ("tr.bigram", "Results: orders 91% of the cases", TR, ["metrics", "reversal_accuracy", "bigram_mismatch_task", "point"], "91%"),
    ("tr.global_lo", "Results: single ranking 53--60%", TR, lambda d: 100 * d["metrics"]["reversal_accuracy"]["global_mix11_mismatch"]["point"], "53"),
    ("tr.global_hi", "Results: -60%", TR, ["metrics", "reversal_accuracy", "global_1B_leave_task_out", "point"], "60%"),
    ("tr.flip_bigram", "Methods: captures 0.89 of the swaps", TR, ["metrics", "flip_capture", "bigram_mismatch_task", "point"], "0.89"),
    ("tr.flip_word", "Methods: word-unigram 0.88", TR, ["metrics", "flip_capture", "word_unigram_ce_task", "point"], "0.88"),
    ("tr.flip_token", "Methods: token-unigram 0.78", TR, ["metrics", "flip_capture", "token_unigram_ce_task", "point"], "0.78"),
    ("tr.regret", "Methods: regret 1.7 points", TR, ["regret_pts", "all10", "point", "bigram_mismatch_task"], "1.7"),
    ("tr.regret_lo", "Methods: (0.4-", TR, ["regret_pts", "all10", "ci95", "bigram_mismatch_task", 0], "0.4"),
    ("tr.regret_hi", "Methods: -2.3)", TR, ["regret_pts", "all10", "ci95", "bigram_mismatch_task", 1], "2.3"),
    ("tr.regret_token", "Methods: 1.9 token unigram", TR, ["regret_pts", "all10", "point", "token_unigram_ce_task"], "1.9"),
    ("tr.regret_word", "Methods: 2.0 word unigram", TR, ["regret_pts", "all10", "point", "word_unigram_ce_task"], "2.0"),
    ("tr.regret_150", "Methods: 1.5 for the 150M proxy", TR, ["regret_pts", "all10", "point", "proxy_150M_cppc"], "1.5"),
    ("tr.regret_300", "Methods: 0.8 for the 300M proxy", TR, ["regret_pts", "all10", "point", "proxy_300M_acc"], "0.8"),
    ("tr.regret_random", "Methods: 4.7 for a random choice", TR, ["regret_pts", "all10", "random_expected"], "4.7"),
    ("tr.edge", "Methods: bigram edge 0.4 points", TR, lambda d: 100 * d["key_contrast"]["bigram_minus_unigram"]["reversal_accuracy"]["point"], "0.4"),
    ("tr.edge_lo", "Methods: (-10.2", TR, lambda d: 100 * d["key_contrast"]["bigram_minus_unigram"]["reversal_accuracy"]["ci95"][0], "-10.2"),
    ("tr.edge_hi", "Methods: to 8.0)", TR, lambda d: 100 * d["key_contrast"]["bigram_minus_unigram"]["reversal_accuracy"]["ci95"][1], "8.0"),
    # Results, rate section and the appendix: floors on the DataDecide ladders
    ("ddf.cells", "Appendix: 275 recipe-set ladders", DDF, ["n_cells"], "275"),
    ("ddf.zero", "Appendix: zero on 268", DDF, ["free_per_cell_floor_zero"], "268"),
    ("ddf.cell_eval", "Appendix: per-ladder floors give 35%", DDF, ["variants", "free per cell", "share_eval"], "35%"),
    ("ddf.cell_recipe", "Appendix: and 13%", DDF, ["variants", "free per cell", "share_recipe"], "13%"),
    ("ddf.set_eval", "Appendix: shared fitted floor leaves 73%", DDF, ["variants", "free per set", "share_eval"], "73%"),
    ("ddf.set_recipe", "Appendix: and 17%", DDF, ["variants", "free per set", "share_recipe"], "17%"),
    # Results, whole curves: compute and the later-pilot row of Table 1
    ("cm.code_frac", "Results: forecast costs 4-15% (code 0.06)", CM, _cm("test80", "code", "Z_frozen_1k") + ["flops_frac_target"], "0.06"),
    ("cm.eng_frac", "Results: forecast costs 4--15%", CM, lambda d: 100 * d["blocks"]["test80"]["english"]["methods"]["Z_frozen_1k"]["flops_frac_target"], "4"),
    ("cm.wiki_frac", "Results: -15%", CM, _cm("test80", "lang", "Z_frozen_1k") + ["flops_frac_target"], "15%"),
    ("cm.pil_eng_frac", "Results: 36--56%", CM, lambda d: 100 * d["blocks"]["test80"]["english"]["methods"]["A3"]["flops_frac_target"], "36"),
    ("cm.pil_code_frac", "Results: -56% for the two pilots", CM, _cm("test80", "code", "A3") + ["flops_frac_target"], "56%"),
    ("cm.ext_pool", "Methods: pooled exponent does as well (0.18)", CM, _cm("ext45", "lang", "SS_pool_exp_1k") + ["rmse"], "0.18"),
    ("cm.web_mean", "ED Table 3 caption: mean fall reaches 0.13", CM, _cm("web30", "lang", "SS_kind_mean") + ["rmse"], "0.13"),
    ("cm.wiki_pool1", "ED Table 3 caption: one pilot 0.08", CM, _cm("test80", "lang", "X_pool1") + ["rmse"], "0.08"),
    ("cm.wiki_pool1_frac", "ED Table 3 caption: at 0.19 of a run", CM, _cm("test80", "lang", "X_pool1") + ["flops_frac_target"], "0.19"),
    ("t1.late.code", "Table 1, later pilots, code", SB, ["rms_2_25", "code", "A3_20_23", "rms"], "0.17"),
    ("t1.late.english", "Table 1, later pilots, English", SB, ["rms_2_25", "english", "A3_20_23", "rms"], "0.04"),
    ("t1.late.wiki", "Table 1, later pilots, Wikipedia", SB, ["rms_2_25", "wiki", "A3_20_23", "rms"], "0.06"),
    ("t1.late.all", "Table 1, later pilots, all", SB, ["rms_2_25", "all", "A3_20_23", "rms"], "0.09"),
    ("t1.late.max", "Table 1, later pilots, largest", SB, _maxabs("A3_20_23"), "0.30"),
    ("t1.pool.max", "Table 1, one pilot pooled exponent, largest", SB, _maxabs("X_pool1"), "1.27"),
    # Results and Methods, scope checks
    ("app.n", "Methods: 133 held-out corpora", APP, lambda d: d["curve"]["all"]["C1"]["separation"]["n_flagged"] + d["curve"]["all"]["C1"]["separation"]["n_pass"], "133"),
    ("app.flag_n", "Methods: the 9 corpora", APP, ["curve", "all", "C1", "separation", "n_flagged"], "9"),
    ("app.pass_n", "Methods: the other 124", APP, ["curve", "all", "C1", "separation", "n_pass"], "124"),
    ("app.flag_k", "Methods: miss in 6 cases", APP, ["curve", "all", "C1", "flagged", "large", "k"], "6"),
    ("app.pass_k", "Methods: 28 of the other 124", APP, ["curve", "all", "C1", "passing", "large", "k"], "28"),
    ("app.flag_share", "Methods: 67%", APP, ["curve", "all", "C1", "flagged", "large", "share"], "67%"),
    ("app.flag_lo", "Methods: Wilson 35-", APP, ["curve", "all", "C1", "flagged", "large", "wilson95", 0], "35%"),
    ("app.flag_hi", "Methods: -88%", APP, ["curve", "all", "C1", "flagged", "large", "wilson95", 1], "88%"),
    ("app.pass_share", "Methods: 23%", APP, ["curve", "all", "C1", "passing", "large", "share"], "23%"),
    ("app.pass_lo", "Methods: 16-", APP, ["curve", "all", "C1", "passing", "large", "wilson95", 0], "16%"),
    ("app.pass_hi", "Methods: -31%", APP, ["curve", "all", "C1", "passing", "large", "wilson95", 1], "31%"),
    ("app.diff", "Methods: a difference of 0.44", APP, ["curve", "all", "C1", "separation", "diff"], "0.44"),
    ("app.diff_lo", "Methods: (0.10-", APP, ["curve", "all", "C1", "separation", "diff_ci95", 0], "0.10"),
    ("app.diff_hi", "Methods: -0.74)", APP, ["curve", "all", "C1", "separation", "diff_ci95", 1], "0.74"),
    ("app.pil_flag", "Methods: two pilots miss them too (1.72", APP, ["curve", "all", "C1", "flagged", "rms_pilots"], "1.72"),
    ("app.pil_pass", "Methods: against 0.25)", APP, ["curve", "all", "C1", "passing", "rms_pilots"], "0.25"),
    ("app.fall_flag_k", "Methods: miss in 5 of 10", APP, ["fall_per_corpus_pooled", "flagged", "large", "k"], "5"),
    ("app.fall_flag_n", "Methods: of 10", APP, ["fall_per_corpus_pooled", "flagged", "n"], "10"),
    ("app.fall_pass_k", "Methods: against 10 of 84", APP, ["fall_per_corpus_pooled", "passing", "large", "k"], "10"),
    ("app.fall_pass_n", "Methods: of 84", APP, ["fall_per_corpus_pooled", "passing", "n"], "84"),
    ("app.fall_diff", "Methods: difference 0.38", APP, ["fall_per_corpus_pooled", "separation", "diff"], "0.38"),
    ("app.fall_diff_lo", "Methods: (0.07-", APP, ["fall_per_corpus_pooled", "separation", "diff_ci95", 0], "0.07"),
    ("app.fall_diff_hi", "Methods: -0.69)", APP, ["fall_per_corpus_pooled", "separation", "diff_ci95", 1], "0.69"),
    ("app.c_flag", "Methods (reference competence): 27% above the threshold", APP, ["curve", "all", "C2", "separation", "large_share_flagged"], "27%"),
    ("app.c_pass", "Methods: against 24% below", APP, ["curve", "all", "C2", "separation", "large_share_pass"], "24%"),
    ("app.wiki_comb", "Results: 0.34 on held-out Wikipedia", APP, ["fall", "wiki", "all", "ngram23_own / mean", "ratio"], "0.34"),
    ("app.wiki_comb_lo", "Results: (0.21-", APP, ["fall", "wiki", "all", "ngram23_own / mean", "ci95", 0], "0.21"),
    ("app.wiki_comb_hi", "Results: -0.51)", APP, ["fall", "wiki", "all", "ngram23_own / mean", "ci95", 1], "0.51"),
    ("app.wiki_n", "Results: 24 held-out Wikipedia languages", APP, ["fall", "wiki", "all", "n"], "24"),
    ("app.code_own", "Methods: own line 1.01", APP, ["fall", "code", "all", "own / mean", "ratio"], "1.01"),
    ("app.code_own_lo", "Methods: 0.99-", APP, ["fall", "code", "all", "own / mean", "ci95", 0], "0.99"),
    ("app.code_own_hi", "Methods: -1.02", APP, ["fall", "code", "all", "own / mean", "ci95", 1], "1.02"),
    ("app.code_comb", "Methods: combined line 0.90", APP, ["fall", "code", "all", "ngram23_own / mean", "ratio"], "0.90"),
    ("app.code_comb_lo", "Methods: 0.70-", APP, ["fall", "code", "all", "ngram23_own / mean", "ci95", 0], "0.70"),
    ("app.code_comb_hi", "Methods: -2.28", APP, ["fall", "code", "all", "ngram23_own / mean", "ci95", 1], "2.28"),
    ("app.code_n", "Methods: the 16 held-out code corpora", APP, ["fall", "code", "all", "n"], "16"),
    ("app.cc_docs", "Appendix: CommonCorpus 66 documents", APP, ["opensciref", "commoncorpus", "n_docs"], "66"),
    ("app.cc_c", "Appendix: its ratio c (0.45)", APP, ["opensciref", "commoncorpus", "c"], "0.45"),
    # Results, how far the loss falls
    ("inc.web_comb", "Results: adding the curve changes nothing (0.98", INC, W + ["combined", "ngram23+own / ngram23", "ratio"], "0.98"),
    ("inc.web_comb_lo", "Results: 0.86-", INC, W + ["combined", "ngram23+own / ngram23", "ci95", 0], "0.86"),
    ("inc.web_comb_hi", "Results: -1.12)", INC, W + ["combined", "ngram23+own / ngram23", "ci95", 1], "1.12"),
    ("inc.eng_comb", "Results: curve lowers the n-gram line's error (0.85", INC, E + ["combined", "ngram23+own / ngram23", "ratio"], "0.85"),
    ("inc.eng_comb_lo", "Results: 0.76-", INC, E + ["combined", "ngram23+own / ngram23", "ci95", 0], "0.76"),
    ("inc.eng_comb_hi", "Results: -0.94)", INC, E + ["combined", "ngram23+own / ngram23", "ci95", 1], "0.94"),
    ("inc.eng_prho", "Results: partial Spearman 0.68", INC, E + ["partial_spearman_own_given", "ngram23", "rho"], "0.68"),
    ("inc.eng_prho_lo", "Results: (0.32-", INC, E + ["partial_spearman_own_given", "ngram23", "ci95", 0], "0.32"),
    ("inc.eng_prho_hi", "Results: -0.86)", INC, E + ["partial_spearman_own_given", "ngram23", "ci95", 1], "0.86"),
    ("inc.web_vs_mean", "Results: combined line 0.51 on web languages", INC, W + ["combined", "ngram23+own / kind_mean", "ratio"], "0.51"),
    ("inc.eng_vs_mean", "Results: 0.84 on English", INC, E + ["combined", "ngram23+own / kind_mean", "ratio"], "0.84"),
    ("inc.web_prho_L23", "Methods: partial Spearman -0.03 given L23", INC, W + ["partial_spearman_own_given", "L23", "rho"], "-0.03"),
    ("inc.bias_lo", "Methods: too small by 0.04-", INC, _bias_range("english", "min"), "0.04"),
    ("inc.bias_hi", "Methods: -0.06", INC, _bias_range("english", "max"), "0.06"),
    ("inc.n_cal_eng", "Methods: 25 earlier English corpora", INC, E + ["n_fit"], "25"),
    ("inc.n_cal_web", "Methods: 29 web languages", INC, W + ["n_fit"], "29"),
    # floor conventions (Results, English paragraph; Methods, finite-budget target)
    ("ifl.zero_prho", "Results/Methods: 0.56 with no floor", IFL, ["results", "zero", "english", "partial_spearman_given_ngram23", "rho"], "0.56"),
    ("ifl.zero_prho_lo", "Methods: (0.12-", IFL, ["results", "zero", "english", "partial_spearman_given_ngram23", "ci95", 0], "0.12"),
    ("ifl.zero_prho_hi", "Methods: -0.82)", IFL, ["results", "zero", "english", "partial_spearman_given_ngram23", "ci95", 1], "0.82"),
    ("ifl.free_prho", "Results/Methods: 0.63 with a fitted floor", IFL, ["results", "free", "english", "partial_spearman_given_ngram23", "rho"], "0.63"),
    ("ifl.free_prho_lo", "Methods: (0.22-", IFL, ["results", "free", "english", "partial_spearman_given_ngram23", "ci95", 0], "0.22"),
    ("ifl.free_prho_hi", "Methods: -0.85)", IFL, ["results", "free", "english", "partial_spearman_given_ngram23", "ci95", 1], "0.85"),
    ("ifl.zero_inc", "Results/Methods: gain becomes 0.92", IFL, ["results", "zero", "english", "own_plus_ngram_vs_ngram", "ratio"], "0.92"),
    ("ifl.zero_inc_lo", "Methods: (0.79-", IFL, ["results", "zero", "english", "own_plus_ngram_vs_ngram", "ci95", 0], "0.79"),
    ("ifl.zero_inc_hi", "Methods: -1.09)", IFL, ["results", "zero", "english", "own_plus_ngram_vs_ngram", "ci95", 1], "1.09"),
    ("ifl.free_inc", "Results/Methods: and 0.98", IFL, ["results", "free", "english", "own_plus_ngram_vs_ngram", "ratio"], "0.98"),
    ("ifl.free_inc_lo", "Methods: (0.93-", IFL, ["results", "free", "english", "own_plus_ngram_vs_ngram", "ci95", 0], "0.93"),
    ("ifl.free_inc_hi", "Methods: -1.05)", IFL, ["results", "free", "english", "own_plus_ngram_vs_ngram", "ci95", 1], "1.05"),
    ("ifl.loo_h960", "Methods: leave-one-out 0.050", IFL, ["results", "h960", "english", "fit_loo_rms_own"], "0.050"),
    ("ifl.loo_zero", "Methods: against 0.053", IFL, ["results", "zero", "english", "fit_loo_rms_own"], "0.053"),
    ("ifl.loo_free", "Methods: and 0.055", IFL, ["results", "free", "english", "fit_loo_rms_own"], "0.055"),
    ("ps.code_paper", "Methods: code advantage 0.37 (h960)", PS, _ps("Table 1 code, data alone / two pilots"), "0.37"),
    ("ps.code_none", "Methods: 0.35 (no floor)", PS, _ps("Table 1 code, data alone / two pilots", "info floor: none"), "0.35"),
    ("ps.code_fit", "Methods: within 0.35-0.37 (fitted, 0.36)", PS, _ps("Table 1 code, data alone / two pilots", "info floor: fitted"), "0.36"),
    ("ps.web_paper", "Methods: web line 0.75", PS, _ps("Web-language fall line / kind mean (30)"), "0.75"),
    ("ps.web_none", "Methods: web line 0.73", PS, _ps("Web-language fall line / kind mean (30)", "info floor: none"), "0.73"),
    ("ps.web_fit", "Methods: web line within 0.73-0.75 (0.74)", PS, _ps("Web-language fall line / kind mean (30)", "info floor: fitted"), "0.74"),
    ("ps.eng_none", "Methods: English fall line 0.89", PS, _ps("English fall line / kind mean (20)", "info floor: none"), "0.89"),
    ("ps.eng_fit", "Methods: and 0.98", PS, _ps("English fall line / kind mean (20)", "info floor: fitted"), "0.98"),
    ("ps.t1eng_none", "Methods: Table 1 English 1.32", PS, _ps("Table 1 English", "info floor: none"), "1.32"),
    ("ps.t1eng_fit", "Methods: and 1.16", PS, _ps("Table 1 English", "info floor: fitted"), "1.16"),
    ("ps.spear84_fit", "Methods/Appendix: Spearman 0.33", PS, _ps("Own fall orders 84 language falls (Spearman)", "info floor: fitted"), "0.33"),
    ("ps.seed_sd", "Methods: single-run s.d. 0.012", PS, ["seed", "sd_single_run"], "0.012"),
    ("ps.dgamma", "Methods: moves gamma by a median of 0.017", PS, ["ref1000", "gamma_diff", "median_abs"], "0.017"),
    ("svp.spear84", "Appendix: Spearman 0.57", SVP, T3 + ["lang", "spearman_ownS_vs_observed_fall", "rho"], "0.57"),
    ("svp.spear84_lo", "Appendix: (0.40-", SVP, T3 + ["lang", "spearman_ownS_vs_observed_fall", "ci95", 0], "0.40"),
    ("svp.spear84_hi", "Appendix: -0.72)", SVP, T3 + ["lang", "spearman_ownS_vs_observed_fall", "ci95", 1], "0.72"),
    # Results and the appendix, open-sci-ref with every constant fixed on DataDecide
    ("osr.f13", "Results/Appendix: 0.096 at 0.13B", OSR, _osr("0.13b") + ["arms", "F_ext|whole", "rms"], "0.096"),
    ("osr.f4", "Results/Appendix: 0.10 at 0.4B", OSR, _osr("0.4b") + ["arms", "F_ext|whole", "rms"], "0.10"),
    ("osr.r13", "Results: error ratio 0.22", OSR, _osr("0.13b") + ["paired", "F_ext|P2e|whole", "ratio"], "0.22"),
    ("osr.r13_lo", "Results: 0.10-", OSR, _osr("0.13b") + ["paired", "F_ext|P2e|whole", "ratio_ci95", 0], "0.10"),
    ("osr.r13_hi", "Results: -0.32", OSR, _osr("0.13b") + ["paired", "F_ext|P2e|whole", "ratio_ci95", 1], "0.32"),
    ("osr.r4", "Results: 0.57", OSR, _osr("0.4b") + ["paired", "F_ext|P2e|whole", "ratio"], "0.57"),
    ("osr.r4_lo", "Results: 0.20-", OSR, _osr("0.4b") + ["paired", "F_ext|P2e|whole", "ratio_ci95", 0], "0.20"),
    ("osr.r4_hi", "Results: -0.91", OSR, _osr("0.4b") + ["paired", "F_ext|P2e|whole", "ratio_ci95", 1], "0.91"),
    ("osr.closer13", "Appendix: closer on 6 of 6", OSR, _osr("0.13b") + ["paired", "F_ext|P2e|whole", "a_closer"], "6"),
    ("osr.closer4", "Appendix: 6 of 7", OSR, _osr("0.4b") + ["paired", "F_ext|P2e|whole", "a_closer"], "6"),
    ("osr.pairs18", "Appendix: 18 of the 21 pairs", OSR, lambda d: 21 * d["variants"]["sub1000"]["0.13b"]["ranking"]["F_ext"]["pairwise_acc"], "18"),
    ("osr.pairs12", "Appendix: 12 of 15 without CommonCorpus", OSR, lambda d: 15 * d["variants"]["sub1000"]["0.13b"]["ranking"]["F_ext_no_commoncorpus"]["pairwise_acc"], "12"),
    ("osr.sp13", "Appendix: Spearman 0.82", OSR, _osr("0.13b") + ["ranking", "F_ext", "spearman"], "0.82"),
    ("osr.sp4", "Appendix: and 0.79", OSR, _osr("0.4b") + ["ranking", "F_ext", "spearman"], "0.79"),
    ("osr.k13", "Appendix: log10 K = 7.14", OSR, _osr("0.13b") + ["calibration", "log10K"], "7.14"),
    ("osr.k4", "Appendix: and 7.49", OSR, _osr("0.4b") + ["calibration", "log10K"], "7.49"),
    ("osr.p2e13", "Appendix: early checkpoints err by 0.24", OSR, _osr("0.13b") + ["arms", "P2e|whole", "rms"], "0.24"),
    ("osr.p2e4", "Appendix: and 0.18", OSR, _osr("0.4b") + ["arms", "P2e|whole", "rms"], "0.18"),
    ("osr.c13", "Appendix: mean of six errs by 0.13", OSR, _osr("0.13b") + ["arms", "C|whole", "rms"], "0.13"),
    ("osr.c4", "Appendix: and 0.12", OSR, _osr("0.4b") + ["arms", "C|whole", "rms"], "0.12"),
    ("osr.rc13", "Appendix: ratio 0.73", OSR, _osr("0.13b") + ["paired", "F_ext|C|whole", "ratio"], "0.73"),
    ("osr.rc13_lo", "Appendix: 0.36-", OSR, _osr("0.13b") + ["paired", "F_ext|C|whole", "ratio_ci95", 0], "0.36"),
    ("osr.rc13_hi", "Appendix: -1.09", OSR, _osr("0.13b") + ["paired", "F_ext|C|whole", "ratio_ci95", 1], "1.09"),
    ("osr.rc4", "Appendix: 0.85", OSR, _osr("0.4b") + ["paired", "F_ext|C|whole", "ratio"], "0.85"),
    ("osr.rc4_lo", "Appendix: 0.39-", OSR, _osr("0.4b") + ["paired", "F_ext|C|whole", "ratio_ci95", 0], "0.39"),
    ("osr.rc4_hi", "Appendix: -1.28", OSR, _osr("0.4b") + ["paired", "F_ext|C|whole", "ratio_ci95", 1], "1.28"),
    ("osr.p2_13", "Appendix: late checkpoints 0.021", OSR, _osr("0.13b") + ["arms", "P2|whole", "rms"], "0.021"),
    ("osr.p2_4", "Appendix: and 0.017", OSR, _osr("0.4b") + ["arms", "P2|whole", "rms"], "0.017"),
    ("osr.flat13", "Appendix: level held flat 0.14", OSR, _osr("0.13b") + ["arms", "F_ext_flat|whole", "rms"], "0.14"),
    ("osr.flat4", "Appendix: and 0.12", OSR, _osr("0.4b") + ["arms", "F_ext_flat|whole", "rms"], "0.12"),
    ("osr.five13", "Results: late checkpoints about five times (4.7)", OSR, _osr("0.13b") + ["paired", "F_ext|P2|whole", "ratio"], "5"),
    ("osr.five4", "Results: about five times (5.9)", OSR, _osr("0.4b") + ["paired", "F_ext|P2|whole", "ratio"], "6"),
    ("osr.jk13_lo", "Appendix: between 0.17", OSR, lambda d: min(d["variants"]["sub1000"]["0.13b"]["jackknife"]["F_ext/P2e whole, corpus dropped"].values()), "0.17"),
    ("osr.jk13_hi", "Appendix: and 0.24", OSR, lambda d: max(d["variants"]["sub1000"]["0.13b"]["jackknife"]["F_ext/P2e whole, corpus dropped"].values()), "0.24"),
    ("osr.jk4_lo", "Appendix: between 0.32", OSR, lambda d: min(d["variants"]["sub1000"]["0.4b"]["jackknife"]["F_ext/P2e whole, corpus dropped"].values()), "0.32"),
    ("osr.jk4_hi", "Appendix: and 0.62", OSR, lambda d: max(d["variants"]["sub1000"]["0.4b"]["jackknife"]["F_ext/P2e whole, corpus dropped"].values()), "0.62"),
    ("osr.cc13", "Appendix: CommonCorpus 0.27 too high", OSR, _osr("0.13b") + ["final_signed", "F_ext", "commoncorpus"], "0.27"),
    ("osr.cc4", "Appendix: and 0.28", OSR, _osr("0.4b") + ["final_signed", "F_ext", "commoncorpus"], "0.28"),
    ("osr.cc_smol", "Appendix: SmolLM2 reads it at 1.505", OSR, _osr("0.13b") + ["ref", "commoncorpus", "r"], "1.505"),
    ("osr.cc_qwen", "Appendix: against 1.240 for Qwen2.5-0.5B", OSR24, ["corpora", "commoncorpus", "ref_full_bpb"], "1.240"),
    ("osr.ref_flops", "Appendix: 4.8e14 operations", OSR, lambda d: d["compute"]["reference"]["flops_per_corpus"]["c4"] / 1e14, "4.8"),
    ("osr.ratio13", "Results/Appendix: a 26,000th (under a 25,000th)", OSR, lambda d: d["compute"]["0.13b"]["P2e_over_ref"] / 1000, "26"),
    ("osr.ratio4", "Appendix: an 82,000th", OSR, lambda d: d["compute"]["0.4b"]["P2e_over_ref"] / 1000, "82"),
    # further numbers: Wikipedia ratio, later pilots, swap interval, combined lines, open-sci-ref, floors, document counts
    ("wiki_ratio", "Results: Wikipedia error ratio 3.79", CM, _cm("test80", "lang", "Z_frozen_1k") + ["ratio_vs_A3", 0], "3.79"),
    ("wiki_ratio_lo", "Results: 2.90-", CM, _cm("test80", "lang", "Z_frozen_1k") + ["ratio_vs_A3", 1], "2.90"),
    ("wiki_ratio_hi", "Results: -5.07", CM, _cm("test80", "lang", "Z_frozen_1k") + ["ratio_vs_A3", 2], "5.07"),
    ("late_frac_lo", "Results: later pilots at 64-", CM, lambda d: 100 * min(d["blocks"]["test80"][f]["methods"]["A3_20_23"]["flops_frac_target"] for f in ("code", "english", "lang")), "64"),
    ("late_frac_hi", "Results: -79% of the run's cost", CM, lambda d: max(d["blocks"]["test80"][f]["methods"]["A3_20_23"]["flops_frac_target"] for f in ("code", "english", "lang")), "79%"),
    ("tr_lo", "Results: swaps 91% (0.64-", TR, ["metrics", "reversal_accuracy", "bigram_mismatch_task", "ci95", 0], "0.64"),
    ("tr_hi", "Results: -0.97)", TR, ["metrics", "reversal_accuracy", "bigram_mismatch_task", "ci95", 1], "0.97"),
    ("web_vs_mean_lo", "Results: combined web 0.51 (0.33-", INC, W + ["combined", "ngram23+own / kind_mean", "ci95", 0], "0.33"),
    ("web_vs_mean_hi", "Results: -0.70)", INC, W + ["combined", "ngram23+own / kind_mean", "ci95", 1], "0.70"),
    ("eng_vs_mean_lo", "Results: combined English 0.84 (0.75-", INC, E + ["combined", "ngram23+own / kind_mean", "ci95", 0], "0.75"),
    ("eng_vs_mean_hi", "Results: -0.92)", INC, E + ["combined", "ngram23+own / kind_mean", "ci95", 1], "0.92"),
    ("eng24_inc", "Methods: all 24 English, combined/n-gram 0.89", APP, ["fall", "english", "all", "ngram23_own / ngram23", "ratio"], "0.89"),
    ("eng24_inc_lo", "Methods: (0.75-", APP, ["fall", "english", "all", "ngram23_own / ngram23", "ci95", 0], "0.75"),
    ("eng24_inc_hi", "Methods: -1.01)", APP, ["fall", "english", "all", "ngram23_own / ngram23", "ci95", 1], "1.01"),
    ("eng24_n", "Methods: all 24 completed English corpora", APP, ["fall", "english", "all", "n"], "24"),
    ("osr_six", "Appendix: on the other six the forecast errs by 0.053", OSR, _osr("0.13b") + ["paired", "F_ext|P2e|whole", "rms_a"], "0.053"),
    ("osr_six_n", "Appendix: the other six corpora", OSR, _osr("0.13b") + ["paired", "F_ext|P2e|whole", "n"], "6"),
    ("osr_p2e_pairs13", "Appendix: early checkpoints 13 of the 15 pairs", OSR, lambda d: 15 * d["variants"]["sub1000"]["0.13b"]["ranking"]["P2e"]["pairwise_acc"], "13"),
    ("osr_p2e_pairs4", "Appendix: all 21 at 0.4B", OSR, lambda d: 21 * d["variants"]["sub1000"]["0.4b"]["ranking"]["P2e"]["pairwise_acc"], "21"),
    ("osr_cc_first", "Appendix: CommonCorpus has no 0.13B checkpoint before 24,000", OSR, lambda d: float(min(int(k) for k in d["variants"]["sub1000"]["0.13b"]["measured"]["commoncorpus"]) >= 24000), "1"),
    ("sp_nchoice", "Appendix: one of twelve choices", PS, lambda d: len(next(h for h in d["rate_headlines"] if h["headline"].startswith("Single pass 40"))["variants"]), "12"),
    ("sp_min", "Appendix: which give -0.38", PS, lambda d: min(v["value"] for v in next(h for h in d["rate_headlines"] if h["headline"].startswith("Single pass 40"))["variants"].values()), "-0.38"),
    ("sp_max", "Appendix: to +0.33", PS, lambda d: max(v["value"] for v in next(h for h in d["rate_headlines"] if h["headline"].startswith("Single pass 40"))["variants"].values()), "0.33"),
    ("flag_nesc", "Appendix: NesC (6 documents)", FDG, ["Q2", "per_corpus", "code_nesc", "n_docs"], "6"),
    ("flag_irc", "Appendix: Ubuntu IRC (17)", FDG, ["Q2", "per_corpus", "pile_ubuntu_irc", "n_docs"], "17"),
    ("flag_mma", "Appendix: Mathematica (25)", FDG, ["Q2", "per_corpus", "code_mathematica", "n_docs"], "25"),
    ("flag_min", "Appendix: Minangkabau Wikipedia (44)", FDG, ["Q2", "per_corpus", "wiki_min", "n_docs"], "44"),
    ("flag_state", "Appendix: state code (1)", ENG, _refdocs("pol_state_code"), "1"),
    ("flag_cfr", "Appendix: CFR (3)", ENG, _refdocs("pol_cfr"), "3"),
    ("flag_fmshrc", "Appendix: FMSHRC (16)", ENG, _refdocs("pol_fmshrc"), "16"),
    ("flag_hansard", "Appendix: UK Hansard (28)", ENG, _refdocs("cp_uk_hansard"), "28"),
    ("flag_hear", "Appendix: congressional hearings (43)", ENG, _refdocs("pol_congressional_hearings"), "43"),
]

# ---------------------------------------------------------------- further checks
SD = R + "source_disjoint/score.json"
CD = R + "code_decomp/decomp.json"
BS = ["bench_strict_crowd4_flan_free"]
REV = "Rev (frozen revision ii: ref level - bias, 0.69 x info fall)"
A0 = "A0 (paper forecast, as run)"
P2 = "Two early pilots (2^19+2^20)"
RLM = "Ref level - bias + code-mean fall"
ORC = "Oracle level + info fall (lam code) / Oracle level + code-mean fall"

def _share_dropped(d):
    pc = [c for c in d["per_corpus"] if c["set"] == "ext45"]
    t = sum(c["err"][REV]["e25"] ** 2 for c in pc)
    return sum(c["err"][REV]["e25"] ** 2 for c in pc if c["corpus"] in d["ext45_C1_dropped"]) / t

def _cancel(d):
    m = d["decomp"]["orig14"][A0]["ms_terms"]
    return -m["cross"] / (m["level"] + m["fall"])

def _maxpr(keys):
    return lambda d: max(d["bench_strict_crowd4_flan_free"]["pairwise_resolved"][k]["point"] for k in keys)

MORE2 = [
    # Results, benchmark text no recipe contains (Appendix; appendix table)
    ("bs_bigram", "Results/SN1: orders 92%", SD, BS + ["pairwise_resolved", "bigram", "point"], "92%"),
    ("bs_bigram_lo", "SN1: (82--", SD, lambda d: 100 * d["bench_strict_crowd4_flan_free"]["pairwise_resolved"]["bigram"]["ci95"][0], "82"),
    ("bs_token", "SN1: token 91%", SD, BS + ["pairwise_resolved", "token unigram", "point"], "91%"),
    ("bs_word", "SN1: word 86%", SD, BS + ["pairwise_resolved", "word unigram", "point"], "86%"),
    ("bs_p4m", "Results: 4M proxy by accuracy 48%", SD, BS + ["pairwise_resolved", "proxy 4M acc", "point"], "48%"),
    ("bs_n17", "Results: 17 candidate recipes", SD, BS + ["n_recipes"], "17"),
    ("bs_pairs", "SN1: 136 pairs", SD, BS + ["pairs"], "136"),
    ("bs_cells", "SN1: 210 pair-task cells", SD, BS + ["resolved_cells"], "210"),
    ("bs_fam", "SN1: five families", SD, lambda d: len(d["bench_strict_crowd4_flan_free"]["families"]), "5"),
    ("bs_tok_d", "SN1: edge over token 0.01", SD, BS + ["contrasts", "bigram minus token unigram", "resolved", "point"], "0.01"),
    ("bs_tok_lo", "SN1: (-0.07", SD, BS + ["contrasts", "bigram minus token unigram", "resolved", "ci95", 0], "-0.07"),
    ("bs_tok_hi", "SN1: to 0.13)", SD, BS + ["contrasts", "bigram minus token unigram", "resolved", "ci95", 1], "0.13"),
    ("bs_word_d", "SN1: over word 0.07", SD, BS + ["contrasts", "bigram minus word unigram", "resolved", "point"], "0.07"),
    ("bs_word_hi", "SN1: (0.00 to 0.31)", SD, BS + ["contrasts", "bigram minus word unigram", "resolved", "ci95", 1], "0.31"),
    ("bs_4m_d", "SN1: over 4M accuracy proxy 0.45", SD, BS + ["contrasts", "bigram minus proxy 4M acc", "resolved", "point"], "0.45"),
    ("bs_4m_lo", "SN1: (0.16", SD, BS + ["contrasts", "bigram minus proxy 4M acc", "resolved", "ci95", 0], "0.16"),
    ("bs_4m_hi", "SN1: to 0.69)", SD, BS + ["contrasts", "bigram minus proxy 4M acc", "resolved", "ci95", 1], "0.69"),
    ("bs_reg_b", "SN1: regret 1.3", SD, BS + ["regret", "bigram", "point"], "1.3"),
    ("bs_reg_t", "SN1: 1.8 token", SD, BS + ["regret", "token unigram", "point"], "1.8"),
    ("bs_reg_w", "SN1: 2.3 word", SD, BS + ["regret", "word unigram", "point"], "2.3"),
    ("bs_reg_4m", "SN1: 5.1 for the 4M accuracy proxy", SD, BS + ["regret", "proxy 4M acc", "point"], "5.1"),
    ("bs_reg_150", "SN1: 0.7 best 150M", SD, BS + ["regret", "proxy 150M cppc", "point"], "0.7"),
    ("bs_reg_300", "SN1: 0.9 best 300M", SD, BS + ["regret", "proxy 300M acc", "point"], "0.9"),
    ("bs_reg_rand", "SN1: 3.2 random", SD, BS + ["regret", "random", "point"], "3.2"),
    ("bs_best150", "SN1/Results: best 150M proxy 87%", SD, _maxpr(["proxy 150M acc", "proxy 150M cppc"]), "87%"),
    ("bs_best300", "SN1/Results: best 300M proxy 90%", SD, _maxpr(["proxy 300M acc", "proxy 300M cppc"]), "90%"),
    ("bs_q_b", "SN1: questions only, bigram 96%", SD, BS + ["pairwise_resolved", "bigram (question only)", "point"], "96%"),
    ("bs_q_t", "SN1: questions only, token 96%", SD, BS + ["pairwise_resolved", "token unigram (question only)", "point"], "96%"),
    ("bs25_b", "SN1: with Flan, bigram 84%", SD, ["bench_crowd4_all25", "pairwise_resolved", "bigram", "point"], "84%"),
    ("bs25_t", "SN1: with Flan, token 88%", SD, ["bench_crowd4_all25", "pairwise_resolved", "token unigram", "point"], "88%"),
    ("bs25_cells", "SN1: 479 resolved cells", SD, ["bench_crowd4_all25", "resolved_cells"], "479"),
    ("p4m_flops", "SN1: one 4M proxy seed 9e15 operations", SD, lambda d: 6 * 4e6 * d["training_tokens"]["4M"] / 1e15, "9"),
    # Appendix, dataset-disjoint loss sets
    ("ood_token", "SN1: token-unigram 68%", MW, ["datadecide", "ood4", "rows", "token unigram", "pairwise"], "68%"),
    ("ood_token_lo", "SN1: (0.49--", MW, ["datadecide", "ood4", "rows", "token unigram", "pairwise_95", 0], "0.49"),
    ("ood_token_hi", "SN1: --0.98", MW, ["datadecide", "ood4", "rows", "token unigram", "pairwise_95", 1], "0.98"),
    ("ood_token_rho", "SN1: Spearman 0.43", MW, ["datadecide", "ood4", "rows", "token unigram", "spearman"], "0.43"),
    ("ood_bigram_lo", "SN1: 67% (45--", MW, lambda d: 100 * d["datadecide"]["ood4"]["rows"]["bigram"]["pairwise_95"][0], "45"),
    ("ood_bigram_hi", "SN1: --92%)", MW, ["datadecide", "ood4", "rows", "bigram", "pairwise_95", 1], "92%"),
    ("d4_4m", "SN1: 4M proxy 83%", SD, ["loss_dataset_disjoint_4", "pairwise_resolved", "proxy 4M loss", "point"], "83%"),
    ("d4_b", "SN1: 73% bigram", SD, ["loss_dataset_disjoint_4", "pairwise_resolved", "bigram", "point"], "73%"),
    ("d4_t", "SN1: 73% token", SD, ["loss_dataset_disjoint_4", "pairwise_resolved", "token unigram", "point"], "73%"),
    ("d4_w", "SN1: 63% word", SD, ["loss_dataset_disjoint_4", "pairwise_resolved", "word unigram", "point"], "63%"),
    ("d4_cells", "SN1: 1,071 cells", SD, ["loss_dataset_disjoint_4", "resolved_cells"], "1,071"),
    ("d4_diff", "SN1: bigram minus proxy -0.10", SD, ["loss_dataset_disjoint_4", "contrasts", "bigram minus proxy 4M loss", "resolved", "point"], "-0.10"),
    ("d4_lo", "SN1: -0.19", SD, ["loss_dataset_disjoint_4", "contrasts", "bigram minus proxy 4M loss", "resolved", "ci95", 0], "-0.19"),
    ("d4_hi", "SN1: to -0.01", SD, ["loss_dataset_disjoint_4", "contrasts", "bigram minus proxy 4M loss", "resolved", "ci95", 1], "-0.01"),
    ("ice_b", "SN1: ICE bigram 82%", SD, ["loss_strict_ice", "pairwise_resolved", "bigram", "point"], "82%"),
    ("ice_t", "SN1: ICE token 78%", SD, ["loss_strict_ice", "pairwise_resolved", "token unigram", "point"], "78%"),
    ("ice_4m", "SN1: ICE 4M 84%", SD, ["loss_strict_ice", "pairwise_resolved", "proxy 4M loss", "point"], "84%"),
    ("ice_w", "SN1: ICE word 67%", SD, ["loss_strict_ice", "pairwise_resolved", "word unigram", "point"], "67%"),
    ("ice_cells", "SN1: 235 resolved pairs", SD, ["loss_strict_ice", "resolved_cells"], "235"),
    # Results and Appendix, code level and fall
    ("c_rev", "SN8: corrected level 0.21 (in-sample)", CD, ["decomp", "orig14", REV, "rms_e25"], "0.21"),
    ("c_rlm", "SN8: level + mean code fall 0.21", CD, ["decomp", "orig14", RLM, "rms_e25"], "0.21"),
    ("c_p2", "Results/SN8: two early pilots 0.60", CD, ["decomp", "orig14", P2, "rms_e25"], "0.60"),
    ("c_p2_lev", "SN8: pilots' level 0.39 too high", CD, ["decomp", "orig14", P2, "mean_level"], "0.39"),
    ("c_p2_fall", "SN8: pilots' fall error 0.14", CD, ["decomp", "orig14", P2, "rms_fall"], "0.14"),
    ("c_a0", "Results/SN8: uncorrected 0.22", CD, ["decomp", "orig14", A0, "rms_e25"], "0.22"),
    ("c_nocancel", "SN8: 0.39 without cancellation", CD, ["decomp", "orig14", A0, "rms_no_cancel"], "0.39"),
    ("c_cancel", "SN8: cancellation removes 68%", CD, _cancel, "68%"),
    ("c_samesign", "SN8: share a sign on 13 of the 14", CD, ["decomp", "orig14", A0, "n_same_sign"], "13"),
    ("c_lev_bias", "SN8: level 0.20 too high", CD, ["decomp", "orig14", A0, "mean_level"], "0.20"),
    ("c_loo_rev", "Results/SN8: leave-one-out 0.23", CD, ["orig14_loo", "Rev", "rms"], "0.23"),
    ("c_loo_rlm", "SN8: LOO 0.23 (mean fall)", CD, ["orig14_loo", "RefLev+meanfall", "rms"], "0.23"),
    ("c_loo_r", "Results/SN8: ratio 0.37", CD, ["orig14_loo", "Rev", "vs_pilots", "ratio"], "0.37"),
    ("c_loo_lo", "SN8: (0.14--", CD, ["orig14_loo", "Rev", "vs_pilots", "ci95", 0], "0.14"),
    ("c_loo_hi", "SN8: --0.75)", CD, ["orig14_loo", "Rev", "vs_pilots", "ci95", 1], "0.75"),
    ("c_loo_r2", "SN8: 0.38", CD, ["orig14_loo", "RefLev+meanfall", "vs_pilots", "ratio"], "0.38"),
    ("c_loo_lo2", "SN8: (0.22--", CD, ["orig14_loo", "RefLev+meanfall", "vs_pilots", "ci95", 0], "0.22"),
    ("c_loo_hi2", "SN8: --0.66)", CD, ["orig14_loo", "RefLev+meanfall", "vs_pilots", "ci95", 1], "0.66"),
    ("c_ext_ratio", "Results/SN8: tie with pilots 2.06", CD, ["score", "ext45", REV, "vs_pilots", "ratio"], "2.06"),
    ("c_ext_rms", "SN8: revision errs 1.75", CD, ["score", "ext45", REV, "rms"], "1.75"),
    ("c_ext_p2", "SN8: against 0.85", CD, ["score", "ext45", P2, "rms"], "0.85"),
    ("c_share", "Results/SN8: 93% of the squared error", CD, _share_dropped, "93%"),
    ("c_nesc", "SN8: NesC (6 documents)", CD, lambda d: next(c["n_docs"] for c in d["per_corpus"] if c["corpus"] == "code_nesc"), "6"),
    ("c_mma", "SN8: Mathematica (25)", CD, lambda d: next(c["n_docs"] for c in d["per_corpus"] if c["corpus"] == "code_mathematica"), "25"),
    ("c_above5", "SN8: learner above 5 bits per byte at 2^23 on both", CD, lambda d: float(min(c["L23"] for c in d["per_corpus"] if c["corpus"] in d["ext45_C1_dropped"]) > 5), "1"),
    ("c_c1", "SN8: other nine 0.75", CD, ["score", "ext45_C1", REV, "vs_pilots", "ratio"], "0.75"),
    ("c_c1_lo", "SN8: (0.43--", CD, ["score", "ext45_C1", REV, "vs_pilots", "ci95", 0], "0.43"),
    ("c_c1_hi", "SN8: --2.09)", CD, ["score", "ext45_C1", REV, "vs_pilots", "ci95", 1], "2.09"),
    ("c_c1_n", "SN8: the other nine", CD, ["n", "ext45_C1"], "9"),
    ("c_orc_ext", "SN8: curve fall worse 1.12", CD, ["increment", "ext45", ORC, "rms", "ratio"], "1.12"),
    ("c_orc_ext_lo", "SN8: (1.02--", CD, ["increment", "ext45", ORC, "rms", "ci95", 0], "1.02"),
    ("c_orc_ext_hi", "SN8: --1.22)", CD, ["increment", "ext45", ORC, "rms", "ci95", 1], "1.22"),
    ("c_orc_14", "SN8: ties on the 14 (1.10", CD, ["increment", "orig14", ORC, "rms", "ratio"], "1.10"),
    ("c_orc_14_lo", "SN8: 0.62--", CD, ["increment", "orig14", ORC, "rms", "ci95", 0], "0.62"),
    ("c_orc_14_hi", "SN8: --1.30)", CD, ["increment", "orig14", ORC, "rms", "ci95", 1], "1.30"),
    ("c_rev_rlm_14", "Results: constant fall matches the curve on the 14 (Rev/level+mean fall rounds to 1.00)", CD, ["increment", "orig14", REV + " / " + RLM, "rms", "ratio"], "1.00"),
]
MORE = MORE + MORE2

MORE3 = [
    # strict benchmark test, both 4M scorings and top-choice regret; code tie on the third-party lists
    ("bs_p4m_cppc", "Results: 4M proxy by answer probability 74%", SD, BS + ["pairwise_resolved", "proxy 4M cppc", "point"], "74%"),
    ("bs_nrec", "Results: 17 recipes that leave out the Flan Collection", SD, BS + ["n_recipes"], "17"),
    ("bs_ntask", "Abstract/Results: four benchmarks", SD, lambda d: len(d["bench_strict_crowd4_flan_free"]["units"]), "4"),
    ("bs_reg_b", "Results: statistic's regret 1.3", SD, BS + ["regret", "bigram", "point"], "1.3"),
    ("bs_reg_150", "Results: 150M proxy regret 0.7", SD, BS + ["regret", "proxy 150M cppc", "point"], "0.7"),
    ("bs_reg_300", "Results: 300M proxy regret 0.9", SD, BS + ["regret", "proxy 300M acc", "point"], "0.9"),
    ("c_ext_lo", "Results/Methods: (0.66--3.63)", EB, ["families", "code", "ratios", "e25:ii/A3", "ci95", 0], "0.66"),
    ("c_ext_closer", "Results: closer on 6 of them", CD, lambda d: int(d["score"]["ext45"][REV]["vs_pilots"]["closer"].split("/")[0]), "6"),
    ("c_ext_n", "Results: 11 programming languages", CD, lambda d: int(d["score"]["ext45"][REV]["vs_pilots"]["closer"].split("/")[1]), "11"),
]
MORE = MORE + MORE3

# appendixc and 3: each printed row is rebuilt from the result file and must appear verbatim
def _f(x, nd=2):
    return str(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))

# Table 2 (tab:benchmark): ten tasks, swaps, uncontained tasks
BENCH = [("Mismatch to benchmark text", "bigram_mismatch_task", "bigram"),
         ("Token-unigram cross-entropy", "token_unigram_ce_task", "token unigram"),
         ("Word-unigram cross-entropy", "word_unigram_ce_task", "word unigram"),
         ("Proxy 4M, accuracy", "proxy_4M_acc", "proxy 4M acc"),
         ("Proxy 4M, answer probability", "proxy_4M_cppc", "proxy 4M cppc"),
         ("Proxy 150M, answer probability", "proxy_150M_cppc", "proxy 150M cppc"),
         ("Proxy 300M, accuracy", "proxy_300M_acc", "proxy 300M acc")]


def table_rows(root):
    out = []
    m = json.load(open(os.path.join(root, TR)))["metrics"]
    sd = json.load(open(os.path.join(root, SD)))["bench_strict_crowd4_flan_free"]
    for name, k, ks in BENCH:
        out.append((f"bench.{k}", "main.tex", f"{name} & {_f(m['all_resolved_accuracy'][k]['point'])} & "
                    f"{_f(m['reversal_accuracy'][k]['point'])} & {_f(sd['pairwise_resolved'][ks]['point'])} \\\\"))
    return out


def get(d, keys):
    if callable(keys):
        return keys(d)
    for k in keys:
        d = d[k]
    return d


def half_up(x, nd):
    return float(Decimal(repr(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def judge(value, printed):
    pct = printed.endswith("%")
    p_s = printed.rstrip("%").replace(",", "")
    nd = len(p_s.split(".")[1]) if "." in p_s else 0
    p = float(p_s)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or math.isnan(value):
        return "FAIL", value
    v = 100 * value if pct else value
    if abs(round(v, nd) - p) < 1e-9 or abs(half_up(v, nd) - p) < 1e-9:
        return "ok", value
    return "FAIL", value


def main():
    root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else ROOT
    cache, fails, n = {}, [], 0
    for cid, where, f, keys, printed in CHECKS + MORE:
        n += 1
        if f not in cache:
            cache[f] = json.load(open(os.path.join(root, f)))
        try:
            st, v = judge(get(cache[f], keys), printed)
        except (KeyError, IndexError, TypeError, StopIteration) as e:
            st, v = "FAIL", f"missing key {e!r}"
        if "--list" in sys.argv or not st.startswith("ok"):
            print(f"{st:34s} {cid:28s} paper {printed:>7s}  file {v if isinstance(v, str) else f'{v:.6g}':>12s}  {where}")
        if not st.startswith("ok"):
            fails.append(cid)
    for cid, tf, row in table_rows(root):
        n += 1
        ok = row in open(ROWS).read()
        if "--list" in sys.argv or not ok:
            print(f"{'ok' if ok else 'FAIL (row differs)':34s} {cid:28s} {tf}: {row}")
        if not ok:
            fails.append(cid)
    print(f"paper_checks: {n} numbers and table rows compared with the result files, {n - len(fails)} agree, {len(fails)} differ")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
