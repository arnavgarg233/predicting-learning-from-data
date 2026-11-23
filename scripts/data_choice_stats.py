"""Statistics of the public-corpus samples (scripts/data_choice_sample.py), computed with the DataDecide code unchanged.

    python scripts/data_choice_stats.py cpu SLUG [SLUG ...]                    -> results/data_choice/stats/<slug>_cpu.json
    python scripts/data_choice_stats.py ref --model HF_ID --device mps SLUG ... -> results/data_choice/stats/<slug>_ref_<short>.json
    python scripts/data_choice_stats.py xent                                    -> results/data_choice/stats/xent_rows.json
    python scripts/data_choice_stats.py unigram                                 -> results/data_choice/stats/unigram_ce.json
    python scripts/data_choice_stats.py merge                                   -> results/data_choice/stats/merged.json
    python scripts/data_choice_stats.py cpu_perm SLUG [SLUG ...]               -> results/data_choice/stats_perm/<slug>_p<k>_cpu.json

cpu_perm (sensitivity, not an input of the forecast): compression, n-gram and beta statistics read only the start of the
sample (1-8 MB, 200k tokens, 4M tokens), i.e. the first one to thirty of its 128 runs. cpu_perm recomputes the cpu
statistics on PERMS reorderings of the runs (seeds 1..PERMS) to measure how much the forecast moves with that choice.

cpu and ref call scripts/datadecide_stats.py (cpu, ref) with only the sample and output folders changed, so every
statistic uses the same code, sizes and seeds as for the 25 DataDecide recipes. One difference: ref skips the
in-document position curve (docpos), which no predictor reads. It still reads documents until 800 of them have at least
1024 tokens, as before, because the paired targets are drawn from that document pool and a smaller pool (fewer
documents, more clustered targets) roughly doubles the standard error of the curve.

xent: the row of the bigram cross-entropy matrix for each new corpus (bigram trained on the first 90% of the corpus
sample, scored on each of the 11 DataDecide eval sets, bits/token), the same quantity as stats/xent.json. As a check
of the code path it also recomputes each eval set's own entry X[e][e] and compares it to the stored value.

unigram (a rival, not part of our predictor): unigram cross-entropy of each eval set under the corpus's token
frequencies (add-0.5 smoothing, as the bigram's back-off), bits/byte, for the new corpora; lower means more word overlap.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import datadecide_stats as dds  # noqa: E402

SAMPLES = ROOT / "data" / "data_choice" / "samples"
STATS = ROOT / "results" / "data_choice" / "stats"
EVALDATA = ROOT / "data" / "datadecide" / "evaldata"
DD_STATS = ROOT / "results" / "datadecide" / "stats"
dds.SAMPLES, dds.STATS = SAMPLES, STATS
dds.slug_name = lambda s: s
os.environ.setdefault("DD_NO_DOCPOS", "1")  # no docpos curve (not used by the predictor); same document pool as before


def new_slugs():
    return sorted(p.stem for p in SAMPLES.glob("*.npy"))


def evals():
    return {e: np.fromfile(EVALDATA / f"{e}.npy", dtype="<u2") for e in dds.EVALS}


def xent():
    X0 = json.load(open(DD_STATS / "xent.json"))["xent"]
    E = evals()
    check = {}
    for e, a in E.items():  # X[e][e]: bigram on the first 90% of the eval set, scored on its last 10%
        s = int(0.9 * len(a))
        check[e] = dict(recomputed=dds.Bigram(a[:s]).bits(a[s:])[0], stored=X0[f"ev_{e}"][f"ev_{e}"])
    rows = {}
    for slug in new_slugs():
        a = dds.load(slug)
        bg = dds.Bigram(a[:int(0.9 * len(a))])
        rows[slug] = {f"ev_{e}": bg.bits(t)[0] for e, t in E.items()}
        print("xent", slug, flush=True)
    dds.dump(dict(unit="bits per token (bigram, abs. discount 0.75)", model_on_first_90pct=True, rows=rows,
                  self_check=check), STATS / "xent_rows.json")
    print("self-check max |diff|", max(abs(v["recomputed"] - v["stored"]) for v in check.values()))


BENCH = {  # benchmark-style text pool: name -> (HF id, parquet file, function row -> text)
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet",
                      lambda r: r["question"] + " " + " ".join(r["choices"]["text"])),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet",
                 lambda r: r["question"] + " " + " ".join(r["choices"]["text"])),
    "hellaswag": ("Rowan/hellaswag", "data/validation-00000-of-00001.parquet", lambda r: r["ctx"] + " " + " ".join(r["endings"])),
    "mmlu": ("cais/mmlu", "all/test-00000-of-00001.parquet", lambda r: r["question"] + " " + " ".join(r["choices"])),
    "openbookqa": ("allenai/openbookqa", "main/test-00000-of-00001.parquet",
                   lambda r: r["question_stem"] + " " + " ".join(r["choices"]["text"])),
    "winogrande": ("allenai/winogrande", "winogrande_xl/validation-00000-of-00001.parquet",
                   lambda r: r["sentence"] + " " + r["option1"] + " " + r["option2"]),
    "commonsense_qa": ("tau/commonsense_qa", "data/validation-00000-of-00001.parquet",
                       lambda r: r["question"] + " " + " ".join(r["choices"]["text"])),
    "boolq": ("google/boolq", "data/validation-00000-of-00001.parquet", lambda r: r["passage"] + " " + r["question"]),
    "sciq": ("allenai/sciq", "data/test-00000-of-00001.parquet",
             lambda r: " ".join([r["support"], r["question"], r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]])),
}


def bench_tokens():
    """Token streams of the benchmark pool (DataDecide tokenizer), cached under data/data_choice/bench/."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    d = ROOT / "data" / "data_choice" / "bench"
    d.mkdir(parents=True, exist_ok=True)
    tok, out, meta = dds.tokenizer(), {}, {}
    for name, (rid, f, fn) in BENCH.items():
        p = d / f"{name}.npy"
        sha = HfApi().dataset_info(rid).sha
        if not p.exists():
            rows = pq.read_table(hf_hub_download(rid, f, repo_type="dataset", revision=sha)).to_pylist()
            ids = [x.ids + [dds.EOS] for x in tok.encode_batch([fn(r) for r in rows], add_special_tokens=False)]
            np.array([t for x in ids for t in x], "<u2").tofile(p)
        a = np.fromfile(p, dtype="<u2")
        txt = dds.texts(a, tok)
        out[name] = (a, sum(len(t.encode("utf-8")) for t in txt) / int((a != dds.EOS).sum()))
        meta[name] = dict(dataset=rid, file=f, revision=sha, n_tokens=int((a != dds.EOS).sum()))
    return out, meta


def unigram():
    E = evals()
    bpt = {e: json.load(open(DD_STATS / f"ev_{e}_cpu.json"))["bytes_per_token"] for e in E}
    B, bmeta = bench_tokens()
    out, pool = {}, {}
    for slug in new_slugs():
        a = dds.load(slug)
        u = np.bincount(a[a != dds.EOS], minlength=dds.V).astype(float) + 0.5
        lp = np.log2(u / u.sum())
        out[slug] = {e: float(-lp[t[t != dds.EOS]].mean() / bpt[e]) for e, t in E.items()}
        pool[slug] = {b: float(-lp[t[t != dds.EOS]].mean() / bb) for b, (t, bb) in B.items()}
    dds.dump(dict(unit="bits per byte of eval text", smoothing="add 0.5", ce=out, bench_ce=pool, bench=bmeta,
                  pools={"benchmark_pool_mean": {s: float(np.mean(list(v.values()))) for s, v in pool.items()}}),
             STATS / "unigram_ce.json")


def merge():
    """Same record per source as datadecide_stats.merge (gamma windows, power-law fit of the paired curve), plus the
    xent rows keyed like stats/xent.json."""
    from tok_stats import gamma_stats
    out = {}
    for s in new_slugs():
        rec = json.loads((STATS / f"{s}_cpu.json").read_text())
        rec.update(kind="recipe", name="HF:" + s)
        rec["ref"] = {}
        for mname in dds.MODELS:
            short = mname.split("/")[-1]
            g = STATS / f"{s}_ref_{short}.json"
            if not g.exists():
                continue
            r = json.loads(g.read_text())
            e = dict(paired=r["paired"], docpos=None)
            rows = r["paired"]["rows"]
            e["gamma"] = {f"{w[0]}-{w[1]}": gamma_stats({short: r["paired"]}, rec["bytes_per_token"],
                                                         rec["beta"]["beta"]["tok_1_200"], w)[short]
                          for w in dds.GAMMA_WINDOWS if w[1] * rec["bytes_per_token"] <= rows[-1]["context_bytes_exact_mean"]}
            nb = np.array([x["context_bytes_exact_mean"] for x in rows])
            h = np.array([x["bits_per_byte"] for x in rows])
            e["fit_paired"] = dds.powerlaw_fit(nb[nb >= 8], h[nb >= 8])
            rec["ref"][short] = e
        out[s] = rec
    xr = json.loads((STATS / "xent_rows.json").read_text())
    dds.dump(dict(sources=out, xent_rows=xr["rows"], xent_self_check=xr["self_check"], models=dds.MODELS),
             STATS / "merged.json")


PERMS = 4


def cpu_perm(slug):
    prov = json.loads((SAMPLES / f"{slug}_provenance.json").read_text())
    a = np.fromfile(SAMPLES / f"{slug}.npy", dtype="<u2")
    cuts = np.cumsum([r["n_tokens"] for r in prov["runs"]])[:-1]
    runs = np.split(a, cuts)
    assert sum(map(len, runs)) == len(a)
    pdir, sdir = SAMPLES.parent / "samples_perm", STATS.parent / "stats_perm"
    pdir.mkdir(exist_ok=True)
    dds.SAMPLES, dds.STATS = pdir, sdir
    for k in range(1, PERMS + 1):
        name = f"{slug}_p{k}"
        if (sdir / f"{name}_cpu.json").exists():
            continue
        order = np.random.default_rng(k).permutation(len(runs))
        np.concatenate([runs[i] for i in order]).astype("<u2").tofile(pdir / f"{name}.npy")
        dds.cpu(name)
        (pdir / f"{name}.npy").unlink()
    dds.SAMPLES, dds.STATS = SAMPLES, STATS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cpu", "ref", "xent", "unigram", "merge", "cpu_perm"])
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--model", default=dds.MODELS[0])
    ap.add_argument("--device", default="mps")
    a = ap.parse_intermixed_args()
    if a.cmd == "cpu":
        for s in a.slugs:
            dds.cpu(s)
    elif a.cmd == "ref":
        dds.ref(a.model, a.slugs, a.device)
    elif a.cmd == "xent":
        xent()
    elif a.cmd == "cpu_perm":
        for s in a.slugs:
            cpu_perm(s)
    elif a.cmd == "unigram":
        unigram()
    else:
        merge()


if __name__ == "__main__":
    main()
