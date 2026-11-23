"""Blind forecast for docs/plan/benchmark_mismatch_test_2026-10-16.md: rank the 25 DataDecide recipes per OLMES task
from data statistics of the task text alone. No downstream accuracy is read anywhere.

    python scripts/bench_mismatch_forecast.py texts     -> data/bench_mismatch/<task>.npy (+ texts_meta.json)
    python scripts/bench_mismatch_forecast.py stats     -> results/bench_mismatch/stats.json (one recipe at a time, resumable)
    python scripts/bench_mismatch_forecast.py write     -> results/bench_mismatch/forecast.json and forecast.sha256

Task text: question plus all answer options, one document per instance (EOS after each), in the DataDecide tokenizer.
Splits follow the OLMES task configs (allenai/olmes oe_eval/configs/tasks.py): test for ARC-C, ARC-E, MMLU, OBQA;
validation for BoolQ, CSQA, HellaSwag, PIQA, SIQA, WinoGrande. The full split is used (OLMES caps some tasks at 1000
sampled instances; the full split is a superset of that sample).

Ours: M_task = bigram cross-entropy (bits/token) of the task text under the recipe's bigram model, the Bigram class of
datadecide_stats.py trained on the first 90% of the recipe sample, as in its xent command. Rival: unigram cross-entropy
(bits/byte) of the task text under the recipe's token frequencies, add-0.5 smoothing, the code of data_choice_stats.py
unigram. Secondary: M against the 11-set eval mix = mean over the 11 eval sets of xent[recipe][eval] - xent[eval][eval]
divided by the eval set's bytes/token, from results/datadecide/stats/xent.json (the mismatch_all feature of
datadecide_predict.py, bits/byte). Lower predicts better.
"""
import hashlib, json, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import datadecide_stats as dds  # noqa: E402

TXT = ROOT / "data" / "bench_mismatch"
OUT = ROOT / "results" / "bench_mismatch"

# name -> (HF id, revision ref for the file list, parquet path, split, row -> text)
TASKS = {
    "arc_challenge": ("allenai/ai2_arc", "main", "ARC-Challenge/test-00000-of-00001.parquet", "test",
                      lambda r: r["question"] + " " + " ".join(r["choices"]["text"])),
    "arc_easy": ("allenai/ai2_arc", "main", "ARC-Easy/test-00000-of-00001.parquet", "test",
                 lambda r: r["question"] + " " + " ".join(r["choices"]["text"])),
    "boolq": ("google/boolq", "main", "data/validation-00000-of-00001.parquet", "validation",
              lambda r: r["passage"] + " " + r["question"] + " yes no"),
    "commonsense_qa": ("tau/commonsense_qa", "main", "data/validation-00000-of-00001.parquet", "validation",
                       lambda r: r["question"] + " " + " ".join(r["choices"]["text"])),
    "hellaswag": ("Rowan/hellaswag", "main", "data/validation-00000-of-00001.parquet", "validation",
                  lambda r: r["ctx"] + " " + " ".join(r["endings"])),
    "mmlu": ("cais/mmlu", "main", "all/test-00000-of-00001.parquet", "test",
             lambda r: r["question"] + " " + " ".join(r["choices"])),
    "openbookqa": ("allenai/openbookqa", "main", "main/test-00000-of-00001.parquet", "test",
                   lambda r: r["question_stem"] + " " + " ".join(r["choices"]["text"])),
    "piqa": ("ybisk/piqa", "refs/convert/parquet", "plain_text/validation/0000.parquet", "validation",
             lambda r: r["goal"] + " " + r["sol1"] + " " + r["sol2"]),
    "social_iqa": ("allenai/social_i_qa", "refs/convert/parquet", "default/validation/0000.parquet", "validation",
                   lambda r: r["context"] + " " + r["question"] + " " + " ".join([r["answerA"], r["answerB"], r["answerC"]])),
    "winogrande": ("allenai/winogrande", "main", "winogrande_xl/validation-00000-of-00001.parquet", "validation",
                   lambda r: r["sentence"] + " " + r["option1"] + " " + r["option2"]),
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def texts():
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download
    TXT.mkdir(parents=True, exist_ok=True)
    tok, meta, api = dds.tokenizer(), {}, HfApi()
    for name, (rid, ref, f, split, fn) in TASKS.items():
        rev = api.dataset_info(rid, revision=ref).sha
        src = hf_hub_download(rid, f, repo_type="dataset", revision=rev)
        rows = pq.read_table(src).to_pylist()
        ids = [x.ids + [dds.EOS] for x in tok.encode_batch([fn(r) for r in rows], add_special_tokens=False)]
        a = np.array([t for x in ids for t in x], "<u2")
        a.tofile(TXT / f"{name}.npy")
        nb = sum(len(t.encode("utf-8")) for t in dds.texts(a, tok))
        meta[name] = dict(dataset=rid, revision=rev, file=f, split=split, n_instances=len(rows),
                          n_tokens=int((a != dds.EOS).sum()), n_bytes=nb, sha256_parquet=sha(src),
                          sha256_npy=sha(TXT / f"{name}.npy"))
        print(name, meta[name]["n_instances"], meta[name]["n_tokens"], flush=True)
    dds.dump(meta, TXT / "texts_meta.json")


def stats():
    meta = json.load(open(TXT / "texts_meta.json"))
    T = {k: np.fromfile(TXT / f"{k}.npy", dtype="<u2") for k in TASKS}
    bpt = {k: meta[k]["n_bytes"] / meta[k]["n_tokens"] for k in TASKS}
    E = {e: np.fromfile(dds.DD / "evaldata" / f"{e}.npy", dtype="<u2") for e in dds.EVALS}
    X0 = json.load(open(dds.STATS / "xent.json"))["xent"]
    f = OUT / "stats.json"
    S = json.load(open(f)) if f.exists() else {}
    for slug in dds.slugs("recipe"):
        if slug in S:
            continue
        a = dds.load(slug)
        bg = dds.Bigram(a[:int(0.9 * len(a))])  # as in datadecide_stats.xent
        u = np.bincount(a[a != dds.EOS], minlength=dds.V).astype(float) + 0.5  # as in data_choice_stats.unigram
        lp = np.log2(u / u.sum())
        # check: the same sample and estimator reproduce the stored xent row (recipe -> each eval set)
        chk = max(abs(bg.bits(t)[0] - X0[slug][f"ev_{e}"]) for e, t in E.items())
        S[slug] = dict(M_task={k: bg.bits(t)[0] for k, t in T.items()},
                       unigram_ce={k: float(-lp[t[t != dds.EOS]].mean() / bpt[k]) for k, t in T.items()},
                       n_tokens=int((a != dds.EOS).sum()), sha256_sample=sha(dds.SAMPLES / f"{slug}.npy"),
                       xent_row_max_abs_diff_vs_stored=float(chk))
        dds.dump(S, f)
        print(slug, f"check {chk:.2e}", flush=True)
        del a, bg


def write():
    meta = json.load(open(TXT / "texts_meta.json"))
    S = json.load(open(OUT / "stats.json"))
    xf = dds.STATS / "xent.json"
    X0 = json.load(open(xf))["xent"]
    rc = dds.slugs("recipe")
    assert sorted(S) == sorted(rc)
    name = {s: dds.slug_name(s) for s in rc}
    bpt = {e: json.load(open(dds.STATS / f"ev_{e}_cpu.json"))["bytes_per_token"] for e in dds.EVALS}
    sec = {s: float(np.mean([(X0[s][f"ev_{e}"] - X0[f"ev_{e}"][f"ev_{e}"]) / bpt[e] for e in dds.EVALS])) for s in rc}
    order = lambda v: [name[s] for s in sorted(rc, key=lambda s: (v[s], s))]  # first = predicted best
    pred = {}
    for k in TASKS:
        m = {s: S[s]["M_task"][k] for s in rc}
        u = {s: S[s]["unigram_ce"][k] for s in rc}
        pred[k] = dict(ours_M_task=order(m), rival_unigram_ce=order(u), secondary_M_mix11=order(sec))
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in
              [TXT / f"{k}.npy" for k in TASKS] + [TXT / "texts_meta.json", OUT / "stats.json", xf,
               dds.DD / "recipe_paths.json", dds.DD / "tokenizer.json"] +
              [dds.DD / "evaldata" / f"{e}.npy" for e in dds.EVALS] +
              [dds.STATS / f"ev_{e}_cpu.json" for e in dds.EVALS] +
              [dds.SAMPLES / f"{s}.npy" for s in rc] +
              [Path(__file__), ROOT / "scripts" / "datadecide_stats.py", ROOT / "scripts" / "data_choice_stats.py"]}
    created = subprocess.run(["date", "+%Y-%m-%dT%H:%M:%S%z"], capture_output=True, text=True).stdout.strip()
    out = dict(
        created=created, plan="docs/plan/benchmark_mismatch_test_2026-10-16.md", blind=True,
        note="No DataDecide downstream or benchmark accuracy was opened or used. Orders list recipe names from predicted "
             "best (lowest statistic) to worst.",
        direction="lower statistic predicts higher accuracy, for all three predictors",
        predictors=dict(
            ours_M_task="bigram cross-entropy of the task text under the recipe sample's bigram model (datadecide_stats.Bigram, "
                        "abs. discount 0.75, trained on the first 90% of the sample, as in its xent command), bits/token",
            rival_unigram_ce="unigram cross-entropy of the task text under the recipe sample's token frequencies (DataDecide "
                             "tokenizer, add-0.5 smoothing), bits/byte; the code of data_choice_stats.unigram, the unigram_ce "
                             "rival of forecast_primary_mismatch.json",
            secondary_M_mix11="mismatch to the 11 DataDecide eval sets: mean over eval sets e of (xent[recipe][e] - xent[e][e]) / "
                              "bytes_per_token[e] (results/datadecide/stats/xent.json), the mismatch_all statistic of "
                              "datadecide_predict.py, bits/byte; same for every task"),
        tasks={k: meta[k] for k in TASKS},
        task_text="question plus all answer options per instance (BoolQ: passage, question, 'yes no'; HellaSwag: ctx and "
                  "the four endings; WinoGrande: sentence with blank and both options; PIQA: goal and both solutions; "
                  "Social IQa: context, question and three answers); full OLMES split",
        recipes={s: name[s] for s in rc},
        predicted_order=pred,
        values=dict(ours_M_task={name[s]: S[s]["M_task"] for s in rc},
                    rival_unigram_ce={name[s]: S[s]["unigram_ce"] for s in rc},
                    secondary_M_mix11={name[s]: sec[s] for s in rc}),
        sample_checks={name[s]: dict(n_tokens=S[s]["n_tokens"],
                                     xent_row_max_abs_diff_vs_stored=S[s]["xent_row_max_abs_diff_vs_stored"]) for s in rc},
        input_sha256=inputs)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "forecast.json"
    p.write_text(json.dumps(out, indent=1))
    (OUT / "forecast.sha256").write_text(f"{sha(p)}  results/bench_mismatch/forecast.json\n")
    print(created, sha(p))


if __name__ == "__main__":
    {"texts": texts, "stats": stats, "write": write}[sys.argv[1]]()
