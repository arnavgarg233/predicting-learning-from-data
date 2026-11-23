"""Question-only benchmark text and 13-gram overlap check.

    python scripts/source_disjoint/qonly_stats.py   -> results/source_disjoint/qonly_stats.json

For the 10 OLMES tasks, rebuilds the text at the revisions pinned in data/bench_mismatch/texts_meta.json, keeping only
the question or context (no answer options). Checks that the full text rebuilt the same way reproduces the stored
.npy byte for byte. Per recipe sample: bigram mismatch (datadecide_stats.Bigram, first 90%), token-unigram CE (add 0.5,
bits/byte) and word-unigram CE (add 0.5, bits/word, the stored vocabulary size) of the question-only text; and the
fraction of 13-token windows of the full and question-only text found verbatim in the sample. No accuracy is read.
"""
import json, re, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import datadecide_stats as dds  # noqa: E402
from bench_mismatch_forecast import TASKS, TXT  # noqa: E402

OUT = ROOT / "results/source_disjoint"
W = re.compile(r"\w+")
QONLY = {
    "arc_challenge": lambda r: r["question"], "arc_easy": lambda r: r["question"], "boolq": lambda r: r["question"],
    "commonsense_qa": lambda r: r["question"], "hellaswag": lambda r: r["ctx"], "mmlu": lambda r: r["question"],
    "openbookqa": lambda r: r["question_stem"], "piqa": lambda r: r["goal"],
    "social_iqa": lambda r: r["context"] + " " + r["question"], "winogrande": lambda r: r["sentence"]}
K = 13


def ngram_hashes(a):
    a64 = a.astype(np.uint64)
    n = len(a) - K + 1
    h = np.zeros(n, np.uint64)
    bad = np.zeros(n, bool)
    P = np.uint64(1000003)
    with np.errstate(over="ignore"):
        for k in range(K):
            h = h * P + a64[k:k + n]
            bad |= a[k:k + n] == dds.EOS
    return h[~bad]


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    meta = json.load(open(TXT / "texts_meta.json"))
    tok = dds.tokenizer()
    full, qo, qbytes, qwords = {}, {}, {}, {}
    for k, (rid, _, f, _, fn) in TASKS.items():
        src = hf_hub_download(rid, f, repo_type="dataset", revision=meta[k]["revision"])
        rows = pq.read_table(src).to_pylist()
        enc = lambda texts: np.array([t for x in tok.encode_batch(texts, add_special_tokens=False) for t in x.ids + [dds.EOS]], "<u2")  # noqa: E731
        a_full = enc([fn(r) for r in rows])
        stored = np.fromfile(TXT / f"{k}.npy", dtype="<u2")
        assert np.array_equal(a_full, stored), k
        q = [QONLY[k](r) for r in rows]
        qo[k] = enc(q)
        full[k] = stored
        qbytes[k] = sum(len(t.encode("utf-8")) for t in q)
        c = Counter()
        for t in q:
            c.update(W.findall(t.lower()))
        qwords[k] = c
        print(k, len(rows), "q tokens", int((qo[k] != dds.EOS).sum()), flush=True)
    fw = json.load(open(ROOT / "results/bench_mismatch/forecast_word_rival.json"))
    Vw, Nw = fw["vocabulary_size"], fw["recipe_words"]
    TH = {k: ngram_hashes(v) for k, v in full.items()}
    QH = {k: ngram_hashes(v) for k, v in qo.items()}
    f = OUT / "qonly_stats.json"
    S = json.load(open(f)) if f.exists() else {}
    for s in dds.slugs("recipe"):
        name = dds.slug_name(s)
        if name in S:
            continue
        t0 = time.time()
        a = dds.load(s)
        bg = dds.Bigram(a[: int(0.9 * len(a))])
        t_bigram = time.time() - t0
        u = np.bincount(a[a != dds.EOS], minlength=dds.V).astype(float) + 0.5
        lp = np.log2(u / u.sum())
        # check: bigram on the stored full text reproduces forecast.json's M_task
        m_full = {k: bg.bits(v)[0] for k, v in full.items()}
        t1 = time.time()
        wc = Counter()
        for t in dds.texts(a, tok):
            wc.update(W.findall(t.lower()))
        t_word = time.time() - t1
        assert sum(wc.values()) == Nw[name], (name, sum(wc.values()), Nw[name])
        den = Nw[name] + 0.5 * Vw
        sh = np.unique(ngram_hashes(a))
        S[name] = dict(
            M_task_qonly={k: bg.bits(v)[0] for k, v in qo.items()},
            M_task_full_recomputed=m_full,
            unigram_ce_qonly={k: float(-lp[v[v != dds.EOS]].sum() / qbytes[k]) for k, v in qo.items()},
            word_ce_qonly={k: float(sum(n * -np.log2((wc.get(w, 0) + 0.5) / den) for w, n in c.items()) / sum(c.values()))
                           for k, c in qwords.items()},
            overlap13_full={k: float(np.isin(h, sh).mean()) for k, h in TH.items()},
            overlap13_qonly={k: float(np.isin(h, sh).mean()) for k, h in QH.items()},
            seconds=dict(bigram_build=t_bigram, word_count=t_word, total=time.time() - t0))
        json.dump(S, open(f, "w"), indent=1)
        print(name, {k: round(v, 3) for k, v in S[name]["seconds"].items()}, flush=True)
        del a, bg, wc, sh


if __name__ == "__main__":
    main()
