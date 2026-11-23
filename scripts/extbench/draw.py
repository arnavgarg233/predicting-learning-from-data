"""External benchmark: seeded draw of 60 corpora from third-party lists (docs/plan/referee_fixes_cost_2026-10-28.md, section 3).
Run once, before any corpus is prepared.

    python scripts/extbench/draw.py     # -> results/extbench_2026-10-28/draw.json

Frames (third-party lists, downloaded 2026-10-28 into ops/state/extbench/):
  code     bigcode/commitpack language directories (HF tree API, commitpack_sizes.json), Linguist type 'programming'
           (github-linguist languages.yml, names and aliases), at least 2^30 bytes of CommitPack jsonl. The 2^30 floor comes from
           the forward test: languages with 0.47-0.82 GB of jsonl (r, elixir, racket) ran out of unique files before the
           ~2^25.7 bytes a corpus needs, and 1.28 GB (scheme) was enough.
  english  the 17 subsets of monology/pile-uncopyrighted (pile_set_name values of train/00.jsonl.zst), then the 15 sources of
           Dolma v1.7 (allenai/dolma card, urls/v1_7.txt) as top-up.
  lang     HuggingFaceFW/fineweb-2 configs (datasets-server size endpoint, HF tree API for the configs it misses), not
           '_removed', at least 2^26 bytes of train parquet.
Exclusions (everything the paper uses): code languages of any corpus or candidate list in the paper (and JSX, which Linguist
files under JavaScript); English sources whose whole documents can occur in a corpus the paper uses, or that are code or not
English; every language of a Wikipedia edition the paper uses or listed (a macrolanguage code excludes all its members).
Draw: seed 20261028 (the date). Each family's eligible names are sorted, then permuted with numpy.random.default_rng(seed + j),
j = 0 code, 1 english, 2 lang. The first n of each order are drawn (15 code, 15 english, 30 lang; all of a frame if smaller).
Replacement rule: the drawn corpora are dealt to 3 pods in order (position i of the family order goes to pod i mod 3);
the remaining names of each family's order (the spares) are dealt the same way, continuing the count. A corpus whose preparation
or reference curve fails is replaced by its pod's next unused spare of the same family, else dropped.
"""
import datetime
import json
import os
import re

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ST = os.path.join(ROOT, "ops/state/extbench")
OUT = os.path.join(ROOT, "results/extbench_2026-10-28/draw.json")
SEED = 20261028
N = dict(code=15, english=15, lang=30)
NPOD = 3

# ---------------------------------------------------------------- code
USED_CODE = {  # every code language in a corpus or candidate list of the paper (scripts/prep_corpus.py)
    "python", "java", "c", "c++", "c#", "javascript", "php", "html",
    "go", "rust", "ruby", "typescript", "shell", "sql", "scala", "haskell", "lua", "perl", "julia", "tex", "powershell",
    "fortran", "css", "markdown",
    "kotlin", "swift", "dart", "r", "ocaml", "elixir", "clojure", "erlang", "scheme", "groovy", "racket"}


def code_frame():
    L = yaml.safe_load(open(os.path.join(ST, "linguist_languages.yml")))
    typ = {}
    for name, v in L.items():
        for k in {name} | set(v.get("aliases", [])):
            typ.setdefault(k.lower().replace(" ", "-"), v["type"])
    rows = json.load(open(os.path.join(ST, "commitpack_sizes.json")))
    elig, excl = [], {}
    for lang, size, _ in rows:
        if not size or size < 2 ** 30:
            continue
        t = typ.get(lang)
        if lang in USED_CODE:
            excl[lang] = "used in the paper"
        elif t is None:
            excl[lang] = "no Linguist entry (JSX is filed under JavaScript)"
        elif t != "programming":
            excl[lang] = f"Linguist type {t}"
        else:
            elig.append(lang)
    return sorted(elig), excl


# ---------------------------------------------------------------- english
PILE = ["Pile-CC", "PubMed Central", "ArXiv", "Github", "FreeLaw", "StackExchange", "Wikipedia (en)", "USPTO Backgrounds",
        "PubMed Abstracts", "Gutenberg (PG-19)", "DM Mathematics", "HackerNews", "EuroParl", "Ubuntu IRC", "PhilPapers",
        "NIH ExPorter", "Enron Emails"]
PILE_EXCL = {
    "Pile-CC": "Common Crawl pages (C4, FineWeb-Edu, CC-News and OpenWebMath are Common Crawl extractions)",
    "ArXiv": "arxiv used", "Github": "code", "StackExchange": "stackexchange used",
    "FreeLaw": "CourtListener opinions include the court opinions of scotus and caselaw",
    "Wikipedia (en)": "English Wikipedia (dbpedia abstracts are its lead paragraphs)",
    "USPTO Backgrounds": "US patent grants (patents corpus)", "PubMed Abstracts": "pubmed used",
    "Gutenberg (PG-19)": "pg19 and poetry used", "HackerNews": "hacker_news used", "EuroParl": "multilingual, not English",
    "Enron Emails": "enron (forward-test candidate)"}
DOLMA = ["Dolma CC", "Refined Web", "StarCoder", "C4", "Reddit", "peS2o", "arXiv (RedPajama)", "StackExchange (RedPajama)", "Flan",
         "CC News", "OpenWebMath", "Algebraic Stack", "Project Gutenberg", "MegaWika", "Wikipedia & Wikibooks"]
DOLMA_EXCL = {
    "Dolma CC": "Common Crawl", "Refined Web": "Common Crawl", "StarCoder": "code", "C4": "c4 used",
    "peS2o": "Semantic Scholar papers (sci_abstracts are S2 abstracts)", "arXiv (RedPajama)": "arxiv used",
    "StackExchange (RedPajama)": "stackexchange used",
    "Flan": "built from datasets the paper uses (xsum, cnn_dailymail, ag_news, squad, yelp, dbpedia)",
    "CC News": "cc_news used", "OpenWebMath": "openwebmath used", "Algebraic Stack": "code",
    "Project Gutenberg": "pg19 used", "MegaWika": "web pages cited from Wikipedia, can occur in Common Crawl extractions",
    "Wikipedia & Wikibooks": "contains Simple English Wikipedia (wiki_simple)"}
EN_SLUG = {"PubMed Central": "pile_pmc", "DM Mathematics": "pile_dm_math", "Ubuntu IRC": "pile_ubuntu_irc",
           "PhilPapers": "pile_philpapers", "NIH ExPorter": "pile_nih_exporter", "Reddit": "dolma_reddit"}


def english_frame():
    pile = sorted(s for s in PILE if s not in PILE_EXCL)
    dolma = sorted(s for s in DOLMA if s not in DOLMA_EXCL)
    return pile, dolma, {**PILE_EXCL, **{f"Dolma {k}" if not k.startswith("Dolma") else k: v for k, v in DOLMA_EXCL.items()}}


# ---------------------------------------------------------------- languages
USED_WIKI = sorted({
    "fr", "ru", "es", "ar", "ko", "it", "pt", "pl", "tr", "vi", "fa", "nl", "sv", "uk", "he", "hi", "cs", "el", "fi", "th",
    "id", "ro", "hu", "da", "no", "bg", "ca", "sr", "hr", "lt", "sk", "sl", "et", "lv", "ms", "bn", "eu", "az", "be", "ka", "hy",
    "kk", "mk", "is", "cy", "ga", "sw", "af", "sq", "mr", "te", "ta", "ur",
    "ceb", "arz", "zh", "ce", "sh", "uz", "min", "gl", "nn", "la", "my", "mg", "bs", "br", "de", "ja", "war", "tt", "eo", "azb",
    "lld", "ast", "tg", "oc", "ml", "nds",
    "simple", "en", "tl", "zh-yue", "fy", "lb", "als", "km", "jv", "an"})
# Wikipedia code -> ISO 639-3 codes as FineWeb-2 names them (macrolanguages expanded to their members in FineWeb-2)
WIKI_ISO = {
    "fr": ["fra"], "ru": ["rus"], "es": ["spa"], "ar": ["arb", "ars", "ary", "arz", "acm", "apc", "ajp", "aeb", "acq", "afb", "ayl",
    "apd", "abv", "shu", "ssh", "acw", "ayp", "acx", "adf", "aao", "auz", "avl", "ayh", "ayn", "pga"], "ko": ["kor"], "it": ["ita"],
    "pt": ["por"], "pl": ["pol"], "tr": ["tur"], "vi": ["vie"], "fa": ["fas", "pes", "prs"], "nl": ["nld"], "sv": ["swe"],
    "uk": ["ukr"], "he": ["heb"], "hi": ["hin"], "cs": ["ces"], "el": ["ell"], "fi": ["fin"], "th": ["tha"], "id": ["ind"],
    "ro": ["ron"], "hu": ["hun"], "da": ["dan"], "no": ["nor", "nob", "nno"], "bg": ["bul"], "ca": ["cat"], "sr": ["srp"],
    "hr": ["hrv"], "lt": ["lit"], "sk": ["slk"], "sl": ["slv"], "et": ["est", "ekk", "vro"], "lv": ["lav", "lvs", "ltg"],
    "ms": ["msa", "zsm", "zlm", "max", "mfa", "meo", "btj", "bve", "kxd", "jax", "xmm", "min", "mui", "zmi", "mqg", "pse", "msi",
    "lce", "liw", "kvr", "hji", "dup", "coa", "bjn", "urk", "vkk", "vkt", "tmw", "lcf", "mfb", "orn", "ors", "xmm"],
    "bn": ["ben"], "eu": ["eus"], "az": ["aze", "azj", "azb"], "be": ["bel"], "ka": ["kat"], "hy": ["hye"], "kk": ["kaz"],
    "mk": ["mkd"], "is": ["isl"], "cy": ["cym"], "ga": ["gle"], "sw": ["swa", "swh", "swc"], "af": ["afr"],
    "sq": ["sqi", "als", "aln", "aae", "aat"], "mr": ["mar"], "te": ["tel"], "ta": ["tam"], "ur": ["urd"], "ceb": ["ceb"],
    "arz": ["arz"], "zh": ["zho", "cmn", "yue", "wuu", "nan", "hak", "gan", "hsn", "cdo", "cjy", "cpx", "czh", "czo", "mnp", "lzh", "cnp", "csp"],
    "ce": ["che"], "sh": ["hbs", "hrv", "bos", "srp", "cnr"], "uz": ["uzb", "uzn", "uzs"], "min": ["min"], "gl": ["glg"],
    "nn": ["nno"], "la": ["lat"], "my": ["mya"], "mg": ["mlg", "plt", "xmv", "bhr", "msh", "bmm", "tdx", "txy", "bzc", "skg", "tkg", "xmw"],
    "bs": ["bos"], "br": ["bre"], "de": ["deu"], "ja": ["jpn"], "war": ["war"], "tt": ["tat"], "eo": ["epo"], "azb": ["azb"],
    "lld": ["lld"], "ast": ["ast"], "tg": ["tgk"], "oc": ["oci"], "ml": ["mal"], "nds": ["nds"],
    "simple": ["eng"], "en": ["eng"], "tl": ["tgl", "fil"], "zh-yue": ["yue"], "fy": ["fry"], "lb": ["ltz"], "als": ["gsw"],
    "km": ["khm"], "jv": ["jav"], "an": ["arg"]}


def lang_frame():
    d = json.load(open(os.path.join(ST, "fw2_size_datasets_server.json")))
    size = {c["config"]: c["num_bytes_parquet_files"] for c in d["size"]["configs"]}
    for k, v in json.load(open(os.path.join(ST, "fw2_size_tree_failed.json"))).items():
        size.setdefault(k, v)
    missing = sorted(k for k, v in size.items() if v is None)
    assert set(USED_WIKI) <= set(WIKI_ISO), set(USED_WIKI) - set(WIKI_ISO)
    used_iso = {i: w for w in USED_WIKI for i in WIKI_ISO[w]}
    elig, excl = [], {}
    for cfg, s in sorted(size.items()):
        if cfg.endswith("_removed") or not s or s < 2 ** 26:
            continue
        iso = cfg.split("_")[0]
        if iso in used_iso:
            excl[cfg] = f"Wikipedia '{used_iso[iso]}' used or listed in the paper"
        else:
            elig.append(cfg)
    return elig, excl, missing, len(size)


def deal(order, n):
    drawn, spares = order[:n], order[n:]
    pods = {p: dict(drawn=[], spares=[]) for p in range(1, NPOD + 1)}
    for i, x in enumerate(drawn):
        pods[i % NPOD + 1]["drawn"].append(x)
    for i, x in enumerate(spares, start=len(drawn)):
        pods[i % NPOD + 1]["spares"].append(x)
    return drawn, spares, pods


def main():
    assert not os.path.exists(OUT), "draw.json exists; the draw is made once"
    code, code_ex = code_frame()
    pile, dolma, en_ex = english_frame()
    lang, lang_ex, lang_missing, n_cfg = lang_frame()
    out = dict(written_pt=datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"), seed=SEED,
               rule=__doc__, n_target=N, families={})
    fr = dict(code=(code, code_ex, "code_{}", "SmolLM2-135M"),
              english=(pile + dolma, en_ex, None, "SmolLM2-135M"),
              lang=(lang, lang_ex, "fw2_{}", "Qwen2.5-0.5B"))
    for j, fam in enumerate(("code", "english", "lang")):
        elig, ex, fmt, ref = fr[fam]
        if fam == "english":   # Pile subsets first (seeded order), Dolma sources after them (seeded order): Dolma only tops up
            o1 = [pile[i] for i in np.random.default_rng(SEED + j).permutation(len(pile))]
            o2 = [dolma[i] for i in np.random.default_rng(SEED + j + 10).permutation(len(dolma))]
            order = o1 + o2
        else:
            order = [elig[i] for i in np.random.default_rng(SEED + j).permutation(len(elig))]
        drawn, spares, pods = deal(order, N[fam])
        name = (lambda x: EN_SLUG[x]) if fam == "english" else (lambda x, f=fmt: f.format(x.replace("+", "p").replace("#", "sharp").replace("-", "_").lower()))
        out["families"][fam] = dict(frame_size=len(elig), eligible_sorted=elig, excluded=ex, seeded_order=order,
                                    corpus_names={x: name(x) for x in order}, drawn=drawn, spares=spares, ref_model=ref,
                                    pods={p: {k: [name(x) for x in v] for k, v in d.items()} for p, d in pods.items()})
    out["families"]["lang"]["configs_without_size"] = lang_missing
    out["families"]["lang"]["configs_listed"] = n_cfg
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    for fam, d in out["families"].items():
        print(fam, "frame", d["frame_size"], "drawn", len(d["drawn"]), d["drawn"])


if __name__ == "__main__":
    main()
