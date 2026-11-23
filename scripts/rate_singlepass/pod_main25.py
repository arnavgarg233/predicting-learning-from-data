"""Pod orchestrator of the 25M rerun of the single-pass rate study (languages only), run from /workspace/k125.

    RS25_CORPORA=i:name,i:name,... RS25_TRAIN_ARGS="--d 512 --nl 8 --nh 8 --lr 0.001" python -u scripts/rate_singlepass/pod_main25.py

i is the corpus's position in the seeded order of the 20 languages (results/rate_singlepass/stats.json). A prep thread
prepares the corpora in the given order (prep_rs.py --encode_test, the 3.3M run's preparation unchanged), at most RS25_AHEAD corpora
ahead of training, and checks the data against the stored statistics: lzma bits per byte (preset 9) of the first 2^23 train bytes
must equal stats.json's lzma_bpb (results/rs25/data_check.json; a mismatch is recorded, not fatal). Training runs RS25_PAR processes
of train_onepass.py over the jobs 2^27 s0, 2^27 s1, 2^26 s0, 2^26 s1, 2^25 .. 2^22 s0 per corpus, in corpus order (one queue, so
no slot idles at a corpus boundary); a corpus's data are deleted after its last run. Before each run
it reads results/RS25_LIMIT (if present: corpora with i >= LIMIT are skipped), which the Mac guard may lower. Outputs:
results/rs25/runs/rs_<name>/bytes_b<k>_s<s>.json, results/rs25/logs/, results/RS25_DONE at the end, RS25_FAILED on an exception.
"""
import concurrent.futures as cf
import datetime
import json
import lzma
import os
import shutil
import subprocess
import sys
import threading

import numpy as np

RES = "results/rs25"
LOG = f"{RES}/logs"
PY = sys.executable
CORPORA = [(int(x.split(":")[0]), x.split(":")[1]) for x in os.environ["RS25_CORPORA"].split(",")]
TRAIN_ARGS = os.environ["RS25_TRAIN_ARGS"].split()
PAR = int(os.environ.get("RS25_PAR", "2"))
AHEAD = int(os.environ.get("RS25_AHEAD", "2"))
JOBS = [(27, 0), (27, 1), (26, 0), (26, 1), (25, 0), (24, 0), (23, 0), (22, 0)]
STATS = json.load(open("results/rate_singlepass/stats.json"))
LZ = {c["name"]: c["stats"]["lzma_bpb"] for c in STATS["language"]["selected"]}
lock = threading.Lock()
ready = {n: threading.Event() for _, n in CORPORA}
prep_ok = {}
trained = threading.Semaphore(AHEAD)


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp(msg):
    with lock:
        with open(f"{LOG}/timing.txt", "a") as f:
            f.write(f"{msg} {now()}\n")
    print(msg, now(), flush=True)


def limit():
    try:
        return int(open("results/RS25_LIMIT").read().strip())
    except (OSError, ValueError):
        return 10 ** 6


def prep_all():
    for i, n in CORPORA:
        trained.acquire()
        d = f"data/rs/full/{n}"
        if i >= limit():
            prep_ok[n] = False
            ready[n].set()
            continue
        rc = 0
        if not os.path.exists(f"{d}/PREP_OK"):
            shutil.rmtree(d, ignore_errors=True)
            with open(f"{LOG}/prep_{n}.log", "w") as f:
                rc = subprocess.run([PY, "-u", "scripts/rate_singlepass/prep_rs.py", "--corpus", n, "--out", d, "--encode_test"],
                                    stdout=f, stderr=subprocess.STDOUT, timeout=4 * 3600).returncode
            if rc == 0:
                open(f"{d}/PREP_OK", "w").write(now())
        if rc == 0:
            raw = np.load(f"{d}/train_bytes.npy", mmap_mode="r")[: 2 ** 23].tobytes()
            lz = 8 * len(lzma.compress(raw, preset=9)) / len(raw)
            chk = dict(name=n, lzma_bpb=lz, stored=LZ[n], identical=abs(lz - LZ[n]) < 1e-12)
            with lock:
                f = f"{RES}/data_check.json"
                dc = json.load(open(f)) if os.path.exists(f) else {}
                dc[n] = chk
                json.dump(dc, open(f, "w"), indent=1)
        prep_ok[n] = rc == 0
        stamp(f"prep {n} rc={rc}" + (f" lzma {lz:.6f} stored {LZ[n]:.6f}" if rc == 0 else ""))
        ready[n].set()


def run(n, k, s):
    f = f"{RES}/runs/rs_{n}/bytes_b{k}_s{s}.json"
    if not os.path.exists(f):
        with open(f"{LOG}/train_{n}_b{k}_s{s}.log", "w") as fh:
            subprocess.run([PY, "-u", "scripts/rate_singlepass/train_onepass.py", "--tok_dir", f"data/rs/full/{n}/tok", "--log2_bytes",
                            str(k), "--seed", str(s), "--out", f"{RES}/runs/rs_{n}", "--device", "cuda"] + TRAIN_ARGS,
                           stdout=fh, stderr=subprocess.STDOUT)
    return os.path.exists(f)


def main():
    os.makedirs(LOG, exist_ok=True)
    stamp(f"rs25 main start PAR={PAR} AHEAD={AHEAD} args={' '.join(TRAIN_ARGS)} corpora={[n for _, n in CORPORA]}")
    threading.Thread(target=prep_all, daemon=True).start()
    left = {n: len(JOBS) for _, n in CORPORA}

    def job(a):
        i, n, k, s = a
        ready[n].wait()
        ok = None
        if i < limit() and prep_ok.get(n):
            ok = run(n, k, s)
            stamp(f"train {n} b{k} s{s} ok={ok}")
        with lock:
            left[n] -= 1
            last = left[n] == 0
        if last:
            shutil.rmtree(f"data/rs/full/{n}", ignore_errors=True)
            if ok is None:
                stamp(f"skip {n} (index {i}, limit {limit()}, prep ok {prep_ok.get(n)})")
            trained.release()
        return ok

    with cf.ThreadPoolExecutor(PAR) as ex:
        res = list(ex.map(job, [(i, n, k, s) for i, n in CORPORA for k, s in JOBS]))
    stamp(f"training done {sum(1 for r in res if r)}/{sum(1 for r in res if r is not None)} runs ok (limit {limit()})")
    open("results/RS25_DONE", "w").write(now())


if __name__ == "__main__":
    os.makedirs(LOG, exist_ok=True)
    try:
        main()
    except BaseException as e:  # noqa: BLE001
        open("results/RS25_FAILED", "w").write(f"{now()} {e!r}\n")
        raise
