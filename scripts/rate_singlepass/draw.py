"""Seeded draw of the corpora for the single-pass rate study.
Run once on the Mac before any corpus is prepared:

    python3 scripts/rate_singlepass/draw.py      # -> results/rate_singlepass/draw.json

Frames (saved beside the output in frames/):
  languages  FineWeb-2 (HuggingFaceFW/fineweb-2), filtered train subsets (not *_removed, not und_*) with at least 2^28 UTF-8
             bytes in fineweb2-language-distribution.csv (github.com/huggingface/fineweb-2, fetched 2026-10-28).
  English    the natural-language English subsets of the uncopyrighted Pile (monology/pile-uncopyrighted), then the Dolma v1.7
             sources (allenai/dolma urls/v1_7.txt) that are natural-language English and do not repeat a Pile subset.
Nothing is fitted on these corpora, so earlier use of a language or domain does not exclude it here.
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "rate_singlepass")
SEED = 20261028                       # the date of the draw

PILE = ["Pile-CC", "PubMed Central", "ArXiv", "FreeLaw", "StackExchange", "USPTO Backgrounds", "PubMed Abstracts",
        "Gutenberg (PG-19)", "Wikipedia (en)", "Ubuntu IRC", "HackerNews", "PhilPapers", "NIH ExPorter", "Enron Emails"]
PILE_EXCLUDED = {"Github": "source code", "DM Mathematics": "synthetic mathematics questions, not natural language",
                 "EuroParl": "multilingual, not English"}
# Dolma v1.7 directory -> short name. Excluded: redpajama-arxiv, redpajama-stackexchange, books and wiki repeat Pile subsets
# (ArXiv, StackExchange, Gutenberg, Wikipedia); starcoder and proof_pile_2-algebraic_stack are code; cc_en_middle/tail and
# cc_news_middle/tail are lower-quality tiers of cc_en_head and cc_news_head (one entry per source).
DOLMA = {"reddit": "reddit", "pes2o": "pes2o", "tulu_flan": "tulu_flan", "wikiref_megawika": "megawika",
         "falcon-refinedweb-filtered": "refinedweb", "c4-filtered": "c4", "cc_en_head": "cc_en_head",
         "cc_news_head": "cc_news", "proof_pile_2-open_web_math": "open_web_math"}
N_PER_FAMILY = 20
BUDGETS = [22, 23, 24, 25, 26, 27]
SEED_BUDGETS = [26, 27]


def slug(s):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_").replace("__", "_")


def main():
    fr = os.path.join(OUT, "frames")
    rows = [r for r in csv.DictReader(open(os.path.join(fr, "fw2_dist.csv")))
            if r["split"] == "train" and not r["subset"].endswith("_removed") and not r["subset"].startswith("und_")]
    lang_frame = sorted(r["subset"] for r in rows if int(r["utf8_bytes"]) >= 2 ** 28)
    size = {r["subset"]: int(r["utf8_bytes"]) for r in rows}
    urls = [u.strip() for u in open(os.path.join(fr, "dolma17.txt")) if u.strip()]
    files = {d: sorted(u for u in urls if u.split("/")[-2] == d) for d in DOLMA}

    rng = np.random.default_rng(SEED)
    lang_order = [lang_frame[i] for i in rng.permutation(len(lang_frame))]
    pile_order = [sorted(PILE)[i] for i in rng.permutation(len(PILE))]
    dkeys = sorted(DOLMA)
    dolma_order = [dkeys[i] for i in rng.permutation(len(dkeys))]
    dolma_file = {d: files[d][int(rng.integers(len(files[d])))] for d in dkeys}   # one seeded file per source (more in order if short)
    dolma_file_order = {d: [files[d][(files[d].index(dolma_file[d]) + j) % len(files[d])] for j in range(len(files[d]))][:4]
                        for d in dkeys}

    languages = [dict(name="fw2_" + s.lower(), source="fineweb2", subset=s, utf8_bytes=size[s], family="language",
                      ref_model="Qwen2.5-0.5B") for s in lang_order]
    english = ([dict(name="pile_" + slug(s), source="pile", subset=s, family="english", ref_model="SmolLM2-135M") for s in pile_order]
               + [dict(name="dolma_" + DOLMA[d], source="dolma", subset=d, files=dolma_file_order[d], family="english",
                       ref_model="SmolLM2-135M") for d in dolma_order])
    out = dict(
        seed=SEED, written="2026-10-28 (before any corpus was prepared)",
        frames=dict(
            languages=dict(source="HuggingFaceFW/fineweb-2, frames/fw2_dist.csv (github.com/huggingface/fineweb-2/blob/main/"
                                  "fineweb2-language-distribution.csv)", rule="filtered train subsets with utf8_bytes >= 2^28",
                           n=len(lang_frame), members=lang_frame),
            english=dict(pile=dict(source="monology/pile-uncopyrighted, train/00-05.jsonl.zst, meta.pile_set_name", members=sorted(PILE),
                                   excluded=PILE_EXCLUDED),
                         dolma=dict(source="allenai/dolma urls/v1_7.txt (frames/dolma17.txt)", members=dkeys,
                                    excluded="redpajama-arxiv, redpajama-stackexchange, books, wiki (repeat a Pile subset); starcoder, "
                                             "proof_pile_2-algebraic_stack (code); cc_en_middle/tail, cc_news_middle/tail (lower tiers "
                                             "of the same sources)"))),
        order=dict(languages=languages, english=english,
                   note="English order is the seeded Pile order followed by the seeded Dolma order (the Pile first, Dolma as the "
                        "top-up, as in the plan's frame)."),
        rules=dict(
            reference="Qwen2.5-0.5B for languages, SmolLM2-135M for English (the protocol of the earlier corpora).",
            screening="Each candidate, in its family's seeded order, is prepared with small quotas (train 2^23*1.05, ref_stats 2^21) and "
                      "screened: c = h_ref(512 tokens) / ngram23, with h_ref from 1,000 targets of the reference model at 512 tokens of "
                      "context and ngram23 the best n-gram cross-entropy trained on 2^23 bytes (the paper's definitions). A candidate "
                      "is eligible when c < 0.70. A candidate whose screening fails (stream, preparation or the reference's byte check) is "
                      "ineligible.",
            selection=f"The first {N_PER_FAMILY} eligible candidates of each family in seeded order are the study corpora; later "
                      "eligible candidates are spares in the same order.",
            replacement="A study corpus that fails full preparation, its statistics or its reference pass is replaced by the next "
                        "eligible spare of its family in seeded order (screening continues down the order if the spares run out); "
                        "if the family's order is exhausted the slot is dropped.",
            training_order="Interleaved: language 1, English 1, language 2, English 2, ... (positions in each family's final list).",
            cost_cut="If the projected all-in spend exceeds the $3.50 cap, corpora are dropped from the end of the interleaved order "
                     "in language/English pairs, never below 30 corpora in total; if 30 cannot be afforded, the run stops and is "
                     "reported.",
            statistics_before_training="gamma (reference curve and n-gram curves), ngram23, c and lzma bits per byte are written "
                                       "to stats.json and copied to the Mac with its local date before any training run starts."),
        design=dict(learner="4-layer byte-level transformer, d=256, 4 heads, ctx 512 bytes, AdamW lr 1e-3 wd 0.1, batch 32 (3.3M)",
                    one_pass="each budget D trains on the first D bytes of the train partition, cut into D/512 non-overlapping "
                             "windows, each used exactly once in a seeded random order (D/16384 steps)",
                    budgets_log2=BUDGETS, seeds={str(k): ([0, 1] if k in SEED_BUDGETS else [0]) for k in BUDGETS}),
    )
    json.dump(out, open(os.path.join(OUT, "draw.json"), "w"), indent=1)
    print(len(lang_frame), "languages in frame;", len(english), "English sources")
    print("languages:", [c["subset"] for c in languages[:30]])
    print("english:", [c["name"] for c in english])


if __name__ == "__main__":
    main()
