"""Pod orchestrator of the 25M code test, run from /workspace/k125.

    python -u scripts/codeeng/c25_main.py

Reads results/curve_forecast/c25_candidates.json ({list: [{name, family, ref_model, train_sha256, ref_sha256}]}).
1. Prepares each corpus with the unchanged pod_main.prep (prep_corpus.py and the bytes tokenizer check, no n-gram curve),
   C25_PREP_PAR at a time, in list order. Writes results/c25_bytes/<name>.json with the sha256 of tok_bytes_train.npy and
   tok_bytes_ref.npy and whether they equal the values stored by the 3.3M runs. A corpus whose bytes differ gets the reference
   pass of its kind's model on the new bytes (pod_main.refcurve) and, after all preparations, forecast.py over those corpora
   (results/curve_forecast/forecast.json), so its own fall is computed on the bytes the learner sees.
2. Encodes the test split and trains the 25.5M learner (d 512, 8 layers, 8 heads, lr 5e-4; otherwise pod_main.COMMON) at
   2^25 seed 0, 2^23 seed 0 and 2^25 seed 1 per corpus in list order (first prepared corpus first), C25_PAR processes, tag c25_<name>. Before each run it
   reads results/EB_LIMIT (number of corpora, in list order, that will be trained), set by the pod's tail cut.
results/EB_DONE at the end, results/EB_FAILED on an exception (the names the self-stop and pull.py expect).
"""
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import threading
import time

sys.path.insert(0, "scripts/curve_forecast")
sys.path.insert(0, "scripts")
import pod_main as PM  # noqa: E402

RES, LOG, PY = PM.RES, PM.LOG, PM.PY
CAND = f"{RES}/c25_candidates.json"
PAR, PREP_PAR = int(os.environ.get("C25_PAR", "2")), int(os.environ.get("C25_PREP_PAR", "3"))
RUNS = ((25, 0), (23, 0), (25, 1))
C25 = ("--test_split --d 512 --nl 8 --nh 8 --ctx_bytes 512 --lr 5e-4 --bs 32 --eval_every 100 --val_chunks 2000 "
       "--patience 10 --max_steps 60000 --max_decays 3 --device cuda")


def limit():
    try:
        return int(open("results/EB_LIMIT").read().strip())
    except (OSError, ValueError):
        return 10 ** 6


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def train(tag, tok_dir, k, seed):
    f = f"results/text_runs/{tag}/bytes_b{k}_s{seed}.json"
    if not os.path.exists(f):
        PM.sh(f"{PY} -u scripts/run_text_ladder.py --toks bytes --seeds {seed} --log2_bytes {k} --tag {tag} "
              f"--extra \"--tok_dir {tok_dir} {C25}\"", f"{LOG}/train_{tag}_b{k}_s{seed}.log")
    return os.path.exists(f)


def main():
    PM.stamp(f"c25 main start PAR={PAR} PREP_PAR={PREP_PAR}")
    os.makedirs("results/c25_bytes", exist_ok=True)
    lst = json.load(open(CAND))["list"]
    psem, rsem = threading.Semaphore(PREP_PAR), threading.Semaphore(1)
    mism = []

    def unit(c):
        n = c["name"]; d = f"data/cf/{n}"
        with psem:
            good, sec = PM.prep(n, ng=False)
        rec = dict(name=n, prep_ok=good, prep_seconds=sec)
        if good:
            rec["train_sha256"] = sha(f"{d}/tok/tok_bytes_train.npy"); rec["ref_sha256"] = sha(f"{d}/tok/tok_bytes_ref.npy")
            rec["train_same"] = rec["train_sha256"] == c["train_sha256"]; rec["ref_same"] = rec["ref_sha256"] == c["ref_sha256"]
            rc = PM.sh(f"[ -f {d}/tok/tok_bytes_test.npy ] || {PY} text/tok_family.py --data {d} --out {d}/tok --names bytes "
                       f"--seed 0 --enc_bytes {PM.ENC} --encode_test", f"{LOG}/tok_test_{n}.log")
            rec["test_encoded"] = rc == 0 and os.path.exists(f"{d}/tok/tok_bytes_test.npy")
            if not (rec["train_same"] and rec["ref_same"]):
                with rsem:
                    ok, s2 = PM.refcurve(n, c["ref_model"])
                rec["refcurve_new_bytes"] = dict(ok=ok, seconds=s2)
                if ok:
                    mism.append(c)
        json.dump(rec, open(f"results/c25_bytes/{n}.json", "w"), indent=1)
        PM.stamp(f"prep {n} ok={good} same={rec.get('train_same')}/{rec.get('ref_same')} {sec:.0f}s")
        return good and rec["test_encoded"]

    pex = cf.ThreadPoolExecutor(PREP_PAR + 1)
    fut = [pex.submit(unit, c) for c in lst]

    def run(a):
        j, c, k, s = a
        if j >= limit() or not fut[j].result():
            return None
        t0 = time.time()
        r = train(f"c25_{c['name']}", f"data/cf/{c['name']}/tok", k, s)
        PM.stamp(f"train c25_{c['name']} b{k} s{s} ok={r} {time.time() - t0:.0f}s")
        return r

    def own_fall_new_bytes():
        cf.wait(fut)
        if mism:
            json.dump(dict(N=len(mism), test=mism, written_utc=PM.now()), open(f"{RES}/corpora.json", "w"), indent=1)
            rc = PM.sh(f"{PY} -u scripts/curve_forecast/forecast.py", f"{LOG}/forecast.log")
            PM.stamp(f"c25 forecast.py on {len(mism)} re-prepared corpora rc={rc}")
    th = threading.Thread(target=own_fall_new_bytes); th.start()
    jobs = [(j, c, k, s) for j, c in enumerate(lst) for k, s in RUNS]
    json.dump([(j, c["name"], k, s) for j, c, k, s in jobs], open(f"{RES}/jobs.json", "w"))
    # a worker takes the first job in list order whose corpus is prepared, so a slow preparation does not idle the GPU
    pending, lock, res = list(jobs), threading.Lock(), []

    def worker():
        while True:
            with lock:
                if not pending:
                    return
                a = next((q for q in pending if fut[q[0]].done()), None)
                if a is not None:
                    pending.remove(a)
            if a is None:
                time.sleep(10)
                continue
            r = run(a)
            with lock:
                res.append(r)
    ws = [threading.Thread(target=worker) for _ in range(PAR)]
    [w.start() for w in ws]
    [w.join() for w in ws]
    th.join()
    PM.stamp(f"c25 training done {sum(1 for r in res if r)}/{sum(1 for r in res if r is not None)} (limit {limit()})")
    open("results/EB_DONE", "w").write(PM.now())


if __name__ == "__main__":
    os.makedirs(LOG, exist_ok=True)
    try:
        main()
    except BaseException as e:  # noqa: BLE001
        open("results/EB_FAILED", "w").write(f"{PM.now()} {e!r}\n")
        raise
