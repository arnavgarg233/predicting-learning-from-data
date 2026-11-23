"""Tables F.1 and F.2 of the article: every forecast test corpus, its source, the reference ratio c and the errors at
2^25 bytes, and the 16 further Wikipedia languages of the reference check.

Same data and row values as scripts/figures/supp_table_corpora.py (results/curve_forecast/score80.json,
results/curve_forecast/stage3/score_stage3.json, ratios c from the forecast files written before training), laid out
as in the article. Writes paper/supp_table1_cl.tex.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import common as C  # noqa: E402
import supp_table_corpora as T  # noqa: E402

OUT = os.path.join(C.ROOT, "paper/supp_table1_cl.tex")

CAP1 = (r"English test corpora of the forecast test, with their Hugging Face sources: the ratio $c$ of "
        r"Section~\ref{sec:results80}, computed from the data alone, the measured test loss at $2^{25}$ training bytes, "
        r"and the absolute errors there (bits per byte) of the forecast with no training ($L_0$), with one pilot ($L_2$) "
        r"and of the two-pilot power law ($L_3$). Reference model SmolLM2-135M.")
CAP2 = (r"Code and Wikipedia corpora of the forecast test, and the further Wikipedia languages of the reference check. "
        r"Columns as in Table~\ref{tab:corpora}. Reference model SmolLM2-135M for code and Qwen2.5-0.5B for Wikipedia; "
        r"languages are given by their codes. On the further languages, forecasts with $c \geq 0.70$ missed by two and a "
        r"half times as much as those below the threshold (one-sided permutation $p = 0.006$).")


def cells(R, c):
    e = R["err"]["25"]
    return f"{c:.2f} & {R['measured']['25']:.3f} & {abs(e['A0']):.3f} & {abs(e['A2']):.3f} & {abs(e['A3']):.3f}"


def two_columns(items):
    half = (len(items) + 1) // 2
    left, right = items[:half], items[half:]
    rows = []
    for i, a in enumerate(left):
        b = right[i] if i < len(right) else " & & & & & "
        rows.append(f"{a} & {b} \\\\")
    return rows


def main():
    P = json.load(open(os.path.join(C.ROOT, "results/curve_forecast/score80.json")))["per_corpus"]
    Q = json.load(open(os.path.join(C.ROOT, "results/curve_forecast/stage3/score_stage3.json")))["per_corpus"]
    c = T.ratios()
    fam = lambda S, f: sorted(k for k in S if S[k]["family"] == f)  # noqa: E731

    eng = [f"{T.ENGLISH[k][0]} & {T.ENGLISH[k][1]} & {cells(P[k], c[k])} \\\\" for k in fam(P, "english")]
    t1 = "\n".join([
        r"\begin{table}[tbp]", r"\centering\footnotesize", f"\\caption{{{CAP1}}}", r"\label{tab:corpora}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.3cm}>{\raggedright\arraybackslash}p{4.9cm}ccccc@{}}",
        r"\toprule", r"Corpus & Source & $c$ & Loss & $|L_0|$ & $|L_2|$ & $|L_3|$ \\", r"\midrule"]
        + eng + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])

    code = [f"{T.CODE_NAME[k[5:]]} & {cells(P[k], c[k])}" for k in fam(P, "code")]
    wiki = [f"{k[5:]} & {cells(P[k], c[k])}" for k in fam(P, "wiki")]
    more = [f"{k[5:]} & {cells(Q[k], c[k])}" for k in fam(Q, "wiki")]
    head = r"Corpus & $c$ & Loss & $|L_0|$ & $|L_2|$ & $|L_3|$ & Corpus & $c$ & Loss & $|L_0|$ & $|L_2|$ & $|L_3|$ \\"
    t2 = "\n".join([
        r"\begin{table}[tbp]", r"\centering\footnotesize", f"\\caption{{{CAP2}}}", r"\label{tab:corpora2}",
        r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}lccccc@{\hspace{14pt}}lccccc@{}}", r"\toprule", head,
        r"\midrule", f"\\multicolumn{{12}}{{@{{}}l}}{{\\textit{{Source code}} ({len(code)}; {T.CODE})}} \\\\"]
        + two_columns(code)
        + [r"\midrule", f"\\multicolumn{{12}}{{@{{}}l}}{{\\textit{{Wikipedia}} ({len(wiki)}; {T.WIKI})}} \\\\"]
        + two_columns(wiki)
        + [r"\midrule",
           f"\\multicolumn{{12}}{{@{{}}l}}{{\\textit{{Further Wikipedia languages, reference-check test}} ({len(more)}; same dump)}} \\\\"]
        + two_columns(more) + [r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(t1 + "\n\n" + t2 + "\n")
    print(OUT, len(eng) + len(code) + len(wiki) + len(more), "rows")


if __name__ == "__main__":
    sys.exit(main())
