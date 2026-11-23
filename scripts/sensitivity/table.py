"""Print the key sensitivity table from sens.json (sensitivity 2026-11-05)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = json.load(open(os.path.join(ROOT, "results/sensitivity/sens.json")))


def f(b):
    return f"{b['ratio']:.2f} ({b['ci95'][0]:.2f}-{b['ci95'][1]:.2f})"


def line(name, sc):
    out = [name]
    t = sc.get("table1_test80_code"); out.append(f"T1 code A0/A3 {f(t['A0_over_A3_none'])} [{t['rms_A0']:.3f} v {t['rms_A3_none']:.3f}]" if t else "")
    t = sc.get("table1_test80_english"); out.append(f"T1 eng {f(t['A0_over_A3_none'])}" if t else "")
    t = sc.get("table1_test80_lang"); out.append(f"T1 wiki {f(t['A0_over_A3_none'])}" if t else "")
    t = sc.get("table1_test80_all"); out.append(f"T1 all {f(t['A0_over_A3_none'])} [{t['rms_A0']:.3f} v {t['rms_A3_none']:.3f}]" if t else "")
    t = sc.get("forward18_code"); out.append(f"F18 code {f(t['ratio'])}" if t else "")
    for fam in ("all", "code", "english", "lang"):
        t = sc.get(f"external45_{fam}")
        if t:
            out.append(f"E45 {fam} rev/A3 {f(t['rev_over_A3_none'])}" + (f" fall {f(t['fall_ratio_rev_over_A3'])}" if 'fall_ratio_rev_over_A3' in t else ""))
    t = sc.get("spearman_own_fall_84_languages"); out.append(f"rho own fall 84 {t['rho']:.2f} ({t['ci95'][0]:.2f}-{t['ci95'][1]:.2f}) n{t['n']}; spread {sc.get('own_fall_spread_over_measured', float('nan')):.2f}" if t else "")
    t = sc.get("weblang30_fall_line"); out.append(f"W30 line/mean {f(t['ratio'])} [{t['rms_line']:.3f} v {t['rms_mean']:.3f}]" if t else "")
    t = sc.get("english20_fall_line"); out.append(f"E20 line/mean {f(t['ratio'])} [{t['rms_line']:.3f} v {t['rms_mean']:.3f}]" if t else "")
    print("\n  ".join(x for x in out if x))


print("checks", S["checks"])
for v, d in S["info_floor"].items():
    print(f"\n### info floor {v}: K {d['K']:.3g} gamma_ts {d['gamma_ts']:.3f} lam {d['lam']:.3f} bias {d['code_bias']:.3f} gamma median {d['gamma_median']:.3f} IQR {d['gamma_iqr']} rho_vs_h960 {d['spearman_gamma_vs_h960']:.2f}",
          {k: d[k] for k in d if k.startswith(("free_E", "n_free"))})
    line(v, d["scores"])
print("\n### A3 floors (paper chain)")
sc = S["info_floor"]["h960"]["scores"]
for k in ("table1_test80_code", "table1_test80_english", "table1_test80_lang", "table1_test80_all", "external45_all", "external45_code", "external45_english", "external45_lang"):
    t = sc[k]
    print(k, {x: (f(t[x]) if isinstance(t[x], dict) else round(t[x], 3)) for x in t if x.startswith(("rms_A3", "A0_over", "rev_over"))})
print("\n### seeds", {k: v for k, v in S["seed"].items() if not k.endswith("only")})
for s in ("s0_only", "s1_only"):
    line(s, S["seed"][s])
print("\n### eval split", json.dumps({k: v for k, v in S["eval_split"].items() if not k.startswith("scores")}))
line("valA", S["eval_split"]["scores_valA"]); line("test same rows", S["eval_split"]["scores_test_same_rows"])
print("\n### ref1000", json.dumps({k: v for k, v in S["ref1000"].items() if not k.startswith("scores")})[:900])
line("1000", S["ref1000"]["scores_1000"]); line("20000 same rows", S["ref1000"]["scores_20000_same_rows"])
