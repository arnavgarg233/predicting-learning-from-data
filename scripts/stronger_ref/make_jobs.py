"""Job list for the Qwen2.5-1.5B reference passes.

For every corpus whose protocol reference is Qwen2.5-0.5B (3 development, 47 Wikipedia test, 16 stage-3 corpora): the
preparation command, the stored reference-partition sha256 pair, the stored number of targets, the seeded 1,000-target
subset and a checksum of the subset's stored target tokens (to verify the target selection is reproduced exactly).
Plain numpy; no model.
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "results/curve_forecast")
OUT = os.path.join(RES, "stronger_ref")
SUB_SEED, N_SUB = 20261025, 1000
TRAIN_Q, ENC = 35300000, 35232154


def subset(nt):
    return np.sort(np.random.default_rng(SUB_SEED).choice(nt, min(N_SUB, nt), replace=False))


def refdir(name):
    for d in (os.path.join(RES, name), os.path.join(RES, "stage2", name), os.path.join(RES, "stage3", name)):
        if os.path.exists(os.path.join(d, "refcurve", "Qwen2.5-0.5B_raw.npz")):
            return os.path.join(d, "refcurve")
    raise FileNotFoundError(name)


def job(name, group, rdir, prep_args):
    key = json.load(open(os.path.join(rdir, "refcurve_fineweb.json")))["Qwen2.5-0.5B"]["key"]
    raw = np.load(os.path.join(rdir, "Qwen2.5-0.5B_raw.npz"))
    nt = len(raw["tgt"])
    idx = subset(nt)
    tg = raw["tgt"][idx].astype(np.int64)
    return dict(name=name, group=group, prep_corpus=prep_args["corpus"], prep_extra=prep_args["extra"],
                corpus_sha=key["corpus"], nt=nt, n_sub=len(idx),
                tgt_sub_sha=hashlib.sha256(tg.tobytes()).hexdigest()[:16], refdir=os.path.relpath(rdir, ROOT))


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for name, res in (("ja_wiki", "mc"), ("de_wiki", "mc2"), ("zh_wiki", "mc2")):
        jobs.append(job(name, "dev", os.path.join(ROOT, f"results/{res}/{name}/refcurve"), dict(corpus=name, extra="")))
    s3 = json.load(open(os.path.join(RES, "stage3/score_stage3.json")))["per_corpus"]
    for name in s3:
        jobs.append(job(name, "stage3", os.path.join(RES, "stage3", name, "refcurve"),
                        dict(corpus=name, extra=f"--train_bytes {TRAIN_Q} --enc_bytes {ENC}")))
    s80 = json.load(open(os.path.join(RES, "score80.json")))["per_corpus"]
    for name, p in s80.items():
        if p["ref_model"] == "Qwen2.5-0.5B":
            jobs.append(job(name, "test80", refdir(name), dict(corpus=name, extra=f"--train_bytes {TRAIN_Q} --enc_bytes {ENC}")))
    json.dump(dict(subset_seed=SUB_SEED, n_sub=N_SUB, jobs=jobs), open(os.path.join(OUT, "jobs.json"), "w"), indent=1)
    print(len(jobs), {g: sum(j["group"] == g for j in jobs) for g in ("dev", "stage3", "test80")})


if __name__ == "__main__":
    main()
