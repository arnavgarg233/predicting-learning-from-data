"""Supplementary Table 1: every forecast test corpus, its source, the reference ratio c and the errors at 2^25 bytes.

Sources: results/curve_forecast/score80.json (80 test corpora), results/curve_forecast/stage3/score_stage3.json (16
further Wikipedia languages), ratios c from the forecast files written before training. Writes paper/supp_table1.tex.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = os.path.join(C.ROOT, "paper/supp_table1.tex")
WIKI = "wikimedia/wikipedia, 2023-11-01 dump"
CODE = "codeparrot/github-code-clean"
ENGLISH = {
    "amazon_polarity": ("Amazon reviews", "fancyzhx/amazon\\_polarity"),
    "billsum": ("US Congressional bills", "FiscalNote/billsum"),
    "booksum": ("Book chapters", "kmfoda/booksum"),
    "caselaw": ("US case law", "common-pile/caselaw\\_access\\_project"),
    "cnn_dailymail": ("News (CNN, Daily Mail)", "abisee/cnn\\_dailymail"),
    "dbpedia": ("DBpedia abstracts", "fancyzhx/dbpedia\\_14"),
    "ecthr": ("ECtHR case facts", "coastalcph/lex\\_glue (ecthr\\_a)"),
    "edgar": ("10-K financial filings", "eloukas/edgar-corpus"),
    "govreport": ("US government reports", "ccdv/govreport-summarization"),
    "hacker_news": ("Hacker News posts", "OpenPipe/hacker-news"),
    "hh_dialogue": ("Assistant dialogues", "Anthropic/hh-rlhf"),
    "multi_news": ("News articles", "alexfabbri/multi\\_news"),
    "patents": ("Patents", "ccdv/patent-classification"),
    "poetry": ("Poetry (Project Gutenberg)", "biglam/gutenberg-poetry-corpus"),
    "recipes": ("Recipes", "corbt/all-recipes"),
    "sci_abstracts": ("Scientific abstracts", "allenai/scirepeval"),
    "scotus": ("US Supreme Court opinions", "coastalcph/lex\\_glue (scotus)"),
    "xsum": ("BBC news articles", "EdinburghNLP/xsum"),
    "yelp": ("Yelp reviews", "Yelp/yelp\\_review\\_full"),
}
CODE_NAME = {"c": "C", "cpp": "C++", "csharp": "C\\#", "css": "CSS", "go": "Go", "html": "HTML", "javascript": "JavaScript",
             "markdown": "Markdown", "perl": "Perl", "php": "PHP", "ruby": "Ruby", "scala": "Scala", "sql": "SQL",
             "typescript": "TypeScript"}
REF = {"SmolLM2-135M": "S", "Qwen2.5-0.5B": "Q"}


def ratios():
    r = {}
    for f in glob.glob(os.path.join(C.ROOT, "results/curve_forecast/**/forecast.json"), recursive=True):
        for k, v in json.load(open(f)).get("corpora", {}).items():
            st = v.get("ref_stats", {})
            if "ngram23" in st:
                r[k] = st["ref_full"] / st["ngram23"]
    return r


def row(name, label, source, R, c):
    e = R["err"]["25"]
    return (f"{label} & {source} & {REF[R['ref_model']]} & {c:.2f} & {R['measured']['25']:.3f} & "
            f"{abs(e['A0']):.3f} & {abs(e['A2']):.3f} & {abs(e['A3']):.3f} \\\\")


def main():
    P = json.load(open(os.path.join(C.ROOT, "results/curve_forecast/score80.json")))["per_corpus"]
    Q = json.load(open(os.path.join(C.ROOT, "results/curve_forecast/stage3/score_stage3.json")))["per_corpus"]
    c = ratios()
    assert set(P) | set(Q) <= set(c)
    lines = []
    for title, fam, S in [("Source code", "code", P), ("English text", "english", P), ("Wikipedia", "wiki", P),
                          ("Further Wikipedia languages (threshold test)", "wiki", Q)]:
        cs = sorted(k for k in S if S[k]["family"] == fam)
        src = {"code": f"; {CODE}", "wiki": f"; {WIKI}"}.get(fam, "")
        lines.append(f"\\midrule\n\\multicolumn{{8}}{{l}}{{\\textit{{{title}}} ({len(cs)}{src})}} \\\\")
        for k in cs:
            if fam == "code":
                label, src = CODE_NAME[k[5:]], ""
            elif fam == "english":
                label, src = ENGLISH[k]
            else:
                label, src = k[5:], ""
            lines.append(row(k, label, src, S[k], c[k]))
    body = "\n".join(lines)
    tex = r"""{\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{longtable}{>{\raggedright\arraybackslash}p{4.3cm}>{\raggedright\arraybackslash}p{5.3cm}cccccc}
\caption{\textbf{Forecast test corpora.} Every corpus of the curve-forecast test, with its Hugging Face source, the
reference model (S, SmolLM2-135M; Q, Qwen2.5-0.5B), the ratio $c$ of the reference model's bits per byte to the
corpus's $n$-gram cross-entropy (computed before training), the measured test loss at $2^{25}$ training bytes and the
absolute errors there (bits per byte) of the forecast with no training ($L_0$), with one pilot ($L_2$) and of the
two-pilot power law ($L_3$). Wikipedia languages are given by their codes.}
\label{tab:s1}\\
\toprule
Corpus & Source & Ref. & $c$ & Loss & $|L_0|$ & $|L_2|$ & $|L_3|$ \\
\endfirsthead
\toprule
Corpus & Source & Ref. & $c$ & Loss & $|L_0|$ & $|L_2|$ & $|L_3|$ \\
\endhead
""" + body + "\n\\bottomrule\n\\end{longtable}}\n"
    open(OUT, "w").write(tex)
    print(OUT, len(P) + len(Q), "rows")


if __name__ == "__main__":
    sys.exit(main())
