"""Pod orchestrator of the single-pass rate study, run from /workspace/k125 (rules in results/rate_singlepass/draw.json).

    python -u scripts/rate_singlepass/pod_main.py

1. Extracts the Pile subsets (prep_rs.py --pile_extract) while the FineWeb-2 candidates are screened.
2. Per family, in seeded order: screening preparation and c (stats_rs.py --screen); the first 20 eligible (c < 0.70) are
   prepared in full (prep_rs.py, test encoded) and measured (stats_rs.py: n-gram curves and lzma on the CPU, then the
   reference curve on the GPU). A failure is replaced by the next eligible candidate of the family in seeded order.
3. Writes results/rate_singlepass/stats.json (per corpus statistics, screening table, failures) and results/RS_STATS_DONE,
   then waits for results/RS_TRAIN, which the Mac guard writes after it has copied stats.json with its local date.
4. Trains in interleaved order (language 1, English 1, ...), per corpus 2^27 s0, 2^27 s1, 2^26 s0, 2^26 s1, 2^25 .. 2^22 s0,
   RS_PAR processes. Before each run it reads results/RS_LIMIT (if present: the number of corpora, in interleaved order, that
   are trained), which the Mac guard lowers if the cost projection requires it. results/RS_DONE at the end, RS_FAILED on an
   exception.
"""
import concurrent.futures as cf
import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time

RES = "results/rate_singlepass"
LOG = f"{RES}/logs"
PY = sys.executable
DRAW = os.environ.get("RS_DRAW", "results/rate_singlepass/draw.json")
N_FAM = int(os.environ.get("RS_N_FAM", "20"))
SCALE = os.environ.get("RS_SCALE", "1.0")                     # local smoke tests only
NG_BUDGETS = os.environ.get("RS_NG_BUDGETS", "23,25")
DEVICE = os.environ.get("RS_DEVICE", "cuda")
SCREEN_NT = os.environ.get("RS_SCREEN_NT", "1000")
PAR = int(os.environ.get("RS_PAR", "4"))
NCPU = os.cpu_count() or 8
SCREEN_PAR = int(os.environ.get("RS_SCREEN_PAR", "8"))
FULL_PAR = int(os.environ.get("RS_FULL_PAR", str(max(2, min(5, NCPU // 4)))))
GPU_PAR = int(os.environ.get("RS_GPU_PAR", "1"))   # two Qwen2.5-0.5B passes at once ran out of GPU memory (2026-10-28)
NT = int(os.environ.get("RS_NT", "4000"))
JOBS = [tuple(map(int, j.split(":"))) for j in os.environ.get("RS_JOBS", "27:0,27:1,26:0,26:1,25:0,24:0,23:0,22:0").split(",")]
EXTRA_TRAIN = os.environ.get("RS_EXTRA_TRAIN", "").split()       # local smoke tests only
cpu_screen, cpu_full, gpu = threading.Semaphore(SCREEN_PAR), threading.Semaphore(FULL_PAR), threading.Semaphore(GPU_PAR)
pile_ready = threading.Event()
lock = threading.Lock()


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp(msg):
    with lock:
        with open(f"{LOG}/timing.txt", "a") as f:
            f.write(f"{msg} {now()}\n")
    print(msg, now(), flush=True)


def sh(cmd, log, timeout=None):
    with open(log, "w") as f:
        try:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout}s\n")
            return 124


def screen(c):
    n = c["name"]
    f = f"{RES}/screen/{n}/screen.json"
    if os.path.exists(f):
        return json.load(open(f))
    if c["source"] == "pile":
        pile_ready.wait()
    d = f"data/rs/screen/{n}"
    t0 = time.time()
    with cpu_screen:
        rc = sh([PY, "-u", "scripts/rate_singlepass/prep_rs.py", "--corpus", n, "--out", d, "--screen", "--scale", SCALE],
                f"{LOG}/screenprep_{n}.log", 3600)
        if rc == 0:                   # ngram23 on the CPU first (cached), so the GPU slot holds only the reference pass
            rc = sh([PY, "-u", "scripts/rate_singlepass/stats_rs.py", "--name", n, "--data", d, "--res", f"{RES}/screen", "--model",
                     c["ref_model"], "--screen", "--part", "ng"], f"{LOG}/screenng_{n}.log", 3600)
    if rc != 0:
        r = dict(name=n, ok=False, reason=f"screening preparation failed (rc {rc})")
    else:
        with gpu:
            rc = sh([PY, "-u", "scripts/rate_singlepass/stats_rs.py", "--name", n, "--data", d, "--res", f"{RES}/screen", "--model",
                     c["ref_model"], "--screen", "--screen_nt", SCREEN_NT, "--device", DEVICE], f"{LOG}/screenstats_{n}.log", 3600)
        r = json.load(open(f)) if os.path.exists(f) else dict(name=n, ok=False, reason=f"screening statistics failed (rc {rc})")
    r["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(f), exist_ok=True)
    json.dump(r, open(f, "w"), indent=1)
    shutil.rmtree(d, ignore_errors=True)
    stamp(f"screen {n} ok={r.get('ok')} c={r.get('c')} {r['seconds']:.0f}s")
    return r


def full(c):
    n = c["name"]
    f = f"{RES}/stats/{n}/stats.json"
    if os.path.exists(f) and json.load(open(f)).get("ok"):
        return json.load(open(f))
    d = f"data/rs/full/{n}"
    t0 = time.time()
    with cpu_full:
        rc = 0
        if not os.path.exists(f"{d}/PREP_OK"):
            shutil.rmtree(d, ignore_errors=True)
            rc = sh([PY, "-u", "scripts/rate_singlepass/prep_rs.py", "--corpus", n, "--out", d, "--encode_test", "--scale", SCALE],
                    f"{LOG}/prep_{n}.log", 4 * 3600)
            if rc == 0:
                open(f"{d}/PREP_OK", "w").write(now())
        stamp(f"prep {n} rc={rc} {time.time() - t0:.0f}s")
        if rc != 0:
            return dict(name=n, ok=False, reason=f"full preparation failed (rc {rc})")
        rc = sh([PY, "-u", "scripts/rate_singlepass/stats_rs.py", "--name", n, "--data", d, "--res", f"{RES}/stats", "--model", c["ref_model"],
                 "--part", "ng", "--ng_budgets", NG_BUDGETS], f"{LOG}/stats_ng_{n}.log", 3 * 3600)
    if rc != 0:
        return dict(name=n, ok=False, reason=f"n-gram statistics failed (rc {rc})")
    with gpu:
        rc = sh([PY, "-u", "scripts/rate_singlepass/stats_rs.py", "--name", n, "--data", d, "--res", f"{RES}/stats", "--model", c["ref_model"],
                 "--part", "ref", "--nt", str(NT), "--device", DEVICE], f"{LOG}/stats_ref_{n}.log", 3 * 3600)
    r = json.load(open(f)) if os.path.exists(f) else dict(name=n, ok=False, reason=f"reference statistics failed (rc {rc})")
    stamp(f"full {n} ok={r.get('ok')} gamma={r.get('gamma_ref')} c={r.get('c')} {time.time() - t0:.0f}s")
    return r


def family(fam, order, ex, out):
    """Screen in seeded order, keep N_FAM corpora prepared and measured; replacements in seeded order."""
    idx = {c["name"]: i for i, c in enumerate(order)}
    sfut, ffut = {}, {}
    nxt, resolved = 0, 0
    spares, ok, failed, screened = [], [], [], []
    while True:
        active = len(ffut) + len(ok)
        need = N_FAM - active
        while need - len(spares) > sum(1 for k in sfut if k >= resolved) - 0 and nxt < len(order) and len([k for k in sfut if k >= resolved]) < 6:
            sfut[nxt] = ex.submit(screen, order[nxt])
            nxt += 1
        while resolved < nxt and sfut[resolved].done():
            r = sfut[resolved].result()
            c = order[resolved]
            resolved += 1
            elig = bool(r.get("ok")) and r.get("c", 9) < 0.70
            screened.append(dict(position=resolved, name=c["name"], ok=r.get("ok"), c=r.get("c"), ref512=r.get("ref512"),
                                 ngram23=r.get("ngram23"), reason=r.get("reason"), eligible=elig))
            if elig:
                spares.append(c)
        while spares and len(ffut) + len(ok) < N_FAM:
            c = spares.pop(0)
            ffut[c["name"]] = (c, ex.submit(full, c))
        for n in [k for k, (c, fu) in ffut.items() if fu.done()]:
            c, fu = ffut.pop(n)
            r = fu.result()
            if r.get("ok"):
                ok.append(dict(c, stats=r))
            else:
                failed.append(dict(name=n, stage="full", reason=r.get("reason")))
        if len(ok) >= N_FAM or (not ffut and not spares and resolved == nxt == len(order)):
            break
        time.sleep(5)
    ok.sort(key=lambda c: idx[c["name"]])
    out[fam] = dict(selected=ok[:N_FAM], unused_eligible=[c["name"] for c in spares], screened=screened, failed=failed,
                    exhausted=len(ok) < N_FAM)
    stamp(f"family {fam} done: {len(ok)} corpora, {len(screened)} screened, {len(failed)} failed")


def limit():
    try:
        return int(open("results/RS_LIMIT").read().strip())
    except (OSError, ValueError):
        return 10 ** 6


def run_train(name, k, s):
    f = f"results/text_runs/rs_{name}/bytes_b{k}_s{s}.json"
    if not os.path.exists(f):
        sh([PY, "-u", "scripts/rate_singlepass/train_onepass.py", "--tok_dir", f"data/rs/full/{name}/tok", "--log2_bytes", str(k),
            "--seed", str(s), "--out", f"results/text_runs/rs_{name}", "--device", DEVICE] + EXTRA_TRAIN, f"{LOG}/train_{name}_b{k}_s{s}.log")
    return os.path.exists(f)


def main():
    os.makedirs(LOG, exist_ok=True)
    stamp(f"rs main start PAR={PAR} SCREEN_PAR={SCREEN_PAR} FULL_PAR={FULL_PAR} GPU_PAR={GPU_PAR} NT={NT} ncpu={NCPU}")
    D = json.load(open(DRAW))["order"]
    if not os.path.exists(f"{RES}/stats.json"):
        def pile():
            rc = sh([PY, "-u", "scripts/rate_singlepass/prep_rs.py", "--pile_extract"], f"{LOG}/pile_extract.log", 4 * 3600)
            stamp(f"pile extraction rc={rc}")
            pile_ready.set()
        threading.Thread(target=pile, daemon=True).start()
        out = {}
        with cf.ThreadPoolExecutor(64) as ex:
            ts = [threading.Thread(target=family, args=(fam, D[key], ex, out)) for fam, key in (("language", "languages"), ("english", "english"))]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
        json.dump(dict(written_utc=now(), nt=NT, **out), open(f"{RES}/stats.json", "w"), indent=1)
        stamp("stats.json written")
    open("results/RS_STATS_DONE", "w").write(now())
    while not os.path.exists("results/RS_TRAIN"):
        time.sleep(15)
    stamp("training released")
    S = json.load(open(f"{RES}/stats.json"))
    L, E = [c["name"] for c in S["language"]["selected"]], [c["name"] for c in S["english"]["selected"]]
    inter = [x for pair in zip(L, E) for x in pair] + L[len(E):] + E[len(L):]
    json.dump(inter, open(f"{RES}/train_order.json", "w"))
    jobs = [(j, n, k, s) for j, n in enumerate(inter) for k, s in JOBS]

    def run(a):
        j, n, k, s = a
        if j >= limit():
            return None
        t0 = time.time()
        r = run_train(n, k, s)
        stamp(f"train {n} b{k} s{s} ok={r} {time.time() - t0:.0f}s")
        return r

    with cf.ThreadPoolExecutor(PAR) as ex:
        res = list(ex.map(run, jobs))
    stamp(f"training done {sum(1 for r in res if r)}/{sum(1 for r in res if r is not None)} (limit {limit()})")
    open("results/RS_DONE", "w").write(now())


if __name__ == "__main__":
    os.makedirs(LOG, exist_ok=True)
    try:
        main()
    except BaseException as e:  # noqa: BLE001
        open("results/RS_FAILED", "w").write(f"{now()} {e!r}\n")
        raise
