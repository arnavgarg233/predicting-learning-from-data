"""Word-level unigram rival for docs/plan/benchmark_mismatch_test_2026-10-16.md (after Chung & Kim), blind.

    python scripts/bench_mismatch_word_rival.py   -> results/bench_mismatch/forecast_word_rival.json (+ .sha256)

Words: decoded text, lowercased, re.findall(r"\\w+"). For each recipe sample, a unigram word model with add-0.5 smoothing
(as data_choice_stats.unigram), over one fixed vocabulary for every recipe: the union of the words of all 25 recipe
samples and all 10 task texts (the analogue of the token code's fixed vocabulary V). Statistic: cross-entropy of each
task's text in bits per word, lower predicts better. Task texts and recipe samples are those of bench_mismatch_forecast.py.
"""
import json, re, subprocess, sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import datadecide_stats as dds  # noqa: E402
from bench_mismatch_forecast import OUT, TASKS, TXT, sha  # noqa: E402

W = re.compile(r"\w+")


def words(a, tok):
    c = Counter()
    for t in dds.texts(a, tok):
        c.update(W.findall(t.lower()))
    return c


def main():
    tok = dds.tokenizer()
    task = {k: words(np.fromfile(TXT / f"{k}.npy", dtype="<u2"), tok) for k in TASKS}
    vocab = set().union(*task.values())
    rc = dds.slugs("recipe")
    N, hit = {}, {}
    for s in rc:
        c = words(dds.load(s), tok)
        N[s] = sum(c.values())
        hit[s] = {k: {w: c.get(w, 0) for w in tc} for k, tc in task.items()}
        vocab.update(c)
        print(s, N[s], len(c), len(vocab), flush=True)
        del c
    Vw = len(vocab)
    ce = {}
    for s in rc:
        den = N[s] + 0.5 * Vw
        ce[s] = {k: float(sum(n * -np.log2((hit[s][k][w] + 0.5) / den) for w, n in tc.items()) / sum(tc.values()))
                 for k, tc in task.items()}
    name = {s: dds.slug_name(s) for s in rc}
    created = subprocess.run(["date", "+%Y-%m-%dT%H:%M:%S%z"], capture_output=True, text=True).stdout.strip()
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in
              [TXT / f"{k}.npy" for k in TASKS] + [TXT / "texts_meta.json", dds.DD / "tokenizer.json"] +
              [dds.SAMPLES / f"{s}.npy" for s in rc] +
              [Path(__file__), ROOT / "scripts" / "bench_mismatch_forecast.py", ROOT / "scripts" / "datadecide_stats.py"]}
    out = dict(
        created=created, plan="docs/plan/benchmark_mismatch_test_2026-10-16.md", blind=True,
        note="Second rival, word level. No DataDecide downstream or benchmark accuracy was opened or used. forecast.json "
             "(token-level rival) is unchanged. Orders run from predicted best (lowest cross-entropy) to worst.",
        predictor="rival_word_unigram_ce: cross-entropy of the task text's words under the recipe sample's word unigram "
                  "model, bits/word; lowercase, regex \\w+; add-0.5 smoothing (the smoothing of data_choice_stats.unigram) "
                  "over a fixed vocabulary = union of words in all 25 recipe samples and all 10 task texts",
        smoothing="add 0.5", vocabulary_size=Vw, direction="lower predicts higher accuracy",
        task_words={k: dict(n_words=int(sum(tc.values())), n_types=len(tc)) for k, tc in task.items()},
        recipe_words={name[s]: int(N[s]) for s in rc}, recipes={s: name[s] for s in rc},
        predicted_order={k: [name[s] for s in sorted(rc, key=lambda s: (ce[s][k], s))] for k in TASKS},
        values={name[s]: ce[s] for s in rc}, input_sha256=inputs)
    p = OUT / "forecast_word_rival.json"
    p.write_text(json.dumps(out, indent=1))
    (OUT / "forecast_word_rival.sha256").write_text(f"{sha(p)}  results/bench_mismatch/forecast_word_rival.json\n")
    print(created, sha(p), "V =", Vw)


if __name__ == "__main__":
    main()
