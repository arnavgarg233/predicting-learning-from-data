"""Pod orchestrator for the curve-forecast test (docs/plan/curve_forecast_vs_pilots_2026-10-17.md), run from /workspace/k125.

    python -u scripts/curve_forecast/pod_main.py smoke     # then waits for results/CF_GO and runs main in the same process
    python -u scripts/curve_forecast/pod_main.py main

smoke  environment record; FineWeb-Edu pilots (cf_fineweb, 2^19 and 2^20, seeds 0 and 1: A1's corpus-blind shape) while the
       first listed test corpus is prepared (prep_corpus.py, bytes tok dir, n-gram curve at 2^23) and its reference curve is
       run; per-stage seconds go to logs/smoke.json. No learner run of a test corpus.
main   reads results/curve_forecast/candidates.json ({N, list: [{name, family, ref_model}]}). Prepares the first N; a corpus
       whose preparation or reference curve fails is replaced by the next unused candidate of the same family (list order).
       Writes corpora.json, runs forecast.py and logs its sha256 before any test-corpus training, encodes the test splits,
       then trains (seed 0 at 2^25, 2^23, 2^20, 2^19 per corpus in list order; seed 1 at 2^25 for the first 5) with
       CF_PAR processes. results/CF_DONE at the end, results/CF_FAILED on an exception.
"""
import concurrent.futures as cf
import datetime
import json
import os
import subprocess
import sys
import time

RES = "results/curve_forecast"
LOG = f"{RES}/logs"
PY = sys.executable
ENC = 35232154                  # int(2^25 * 1.05) bytes of train encoded for the learner
TRAIN_Q = 35300000              # train partition quota (>= ENC)
COMMON = ("--test_split --d 256 --nl 4 --nh 4 --ctx_bytes 512 --lr 1e-3 --bs 32 --eval_every 100 --val_chunks 2000 "
          "--patience 10 --max_steps 60000 --max_decays 3 --device cuda")
HF = {"Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B", "SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M"}
PAR = int(os.environ.get("CF_PAR", "4"))
PREP_PAR, REF_PAR = int(os.environ.get("CF_PREP_PAR", "4")), int(os.environ.get("CF_REF_PAR", "3"))


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp(msg):
    with open(f"{LOG}/timing.txt", "a") as f:
        f.write(f"{msg} {now()}\n")
    print(msg, now(), flush=True)


def sh(cmd, log, timeout=None):
    with open(log, "w") as f:
        try:
            return subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\nTIMEOUT after {timeout}s\n")
            return 124


def prep(c, ng=True):
    d = f"data/cf/{c}"
    if os.path.exists(f"{d}/PREP_OK"):
        return True, 0.0
    t0 = time.time()
    rc = sh(f"{PY} -u scripts/prep_corpus.py --corpus {c} --out {d} --train_bytes {TRAIN_Q} --enc_bytes {ENC}", f"{LOG}/prep_{c}.log", 3600)
    if rc == 0:
        rc = sh(f"{PY} text/tok_family.py --data {d} --out {d}/tok --names bytes --seed 0 --enc_bytes {ENC} --verify", f"{LOG}/tok_verify_{c}.log")
    if rc == 0 and ng:
        rc = sh(f"{PY} -c \"import sys; sys.path.insert(0, 'scripts'); import forecast_multicorpus as FM; FM.ng_curve('{c}', '{d}/tok', 23, '{RES}')\"",
                f"{LOG}/ngram_{c}.log", 3600)
    sec = time.time() - t0
    os.makedirs(d, exist_ok=True)
    open(f"{d}/{'PREP_OK' if rc == 0 else 'PREP_FAIL'}", "w").write(f"{sec:.0f}\n")
    return rc == 0, sec


def refcurve(c, model):
    out = f"{RES}/{c}/refcurve"
    f = f"{out}/refcurve_fineweb.json"
    t0 = time.time()
    if not (os.path.exists(f) and model in json.load(open(f))):
        os.makedirs(out, exist_ok=True)
        sh(f"{PY} -u text/refcurve/refcurve_fineweb.py --data data/cf/{c} --out {out} --models {HF[model]} --device cuda", f"{LOG}/refcurve_{c}.log", 5400)
    ok = os.path.exists(f) and model in json.load(open(f))
    return ok, time.time() - t0


def train_cmd(tag, tok_dir, k, seed):
    return (f"{PY} -u scripts/run_text_ladder.py --toks bytes --seeds {seed} --log2_bytes {k} --tag {tag} "
            f"--extra \"--tok_dir {tok_dir} {COMMON}\"")


def run_train(tag, tok_dir, k, seed):
    f = f"results/text_runs/{tag}/bytes_b{k}_s{seed}.json"
    if not os.path.exists(f):
        sh(train_cmd(tag, tok_dir, k, seed), f"{LOG}/train_{tag}_b{k}_s{seed}.log")
    return os.path.exists(f)


def smoke():
    stamp("smoke start")
    sh(f"{PY} -c \"import torch, transformers, datasets; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
       f"torch.cuda.get_device_name(0)); print('transformers', transformers.__version__, 'datasets', datasets.__version__)\"; nproc; free -g; df -h /workspace",
       f"{LOG}/smoke_env.txt")
    cand = json.load(open(f"{RES}/candidates.json"))["list"][0]
    out = dict(first_candidate=cand["name"])
    with cf.ThreadPoolExecutor(5) as ex:
        t0 = time.time()
        fw = {(k, s): ex.submit(run_train, "cf_fineweb", "data/bfine/tok", k, s) for k in (19, 20) for s in (0, 1)}
        pf = ex.submit(prep, cand["name"])
        ok, out["prep_seconds"] = pf.result()
        out["prep_ok"] = ok
        if ok:
            ok, out["refcurve_seconds"] = refcurve(cand["name"], cand["ref_model"])
            out["refcurve_ok"] = ok
        out["fineweb_runs_ok"] = {f"{k}_{s}": f.result() for (k, s), f in fw.items()}
        out["wall_seconds"] = time.time() - t0
    runs = {}
    for k in (19, 20):
        for s in (0, 1):
            p = f"results/text_runs/cf_fineweb/bytes_b{k}_s{s}.json"
            if os.path.exists(p):
                d = json.load(open(p))
                runs[f"{k}_{s}"] = dict(steps=d["steps"], seconds=d["seconds"], sec_per_step=d["seconds"] / d["steps"], bpb=d["bits_per_byte"])
    out["fineweb_runs"] = runs
    json.dump(out, open(f"{LOG}/smoke.json", "w"), indent=1)
    stamp("smoke done")
    open("results/CF_SMOKE_DONE", "w").write(now())
    for _ in range(180):                      # wait up to 60 min for the go file (N and the list are decided locally)
        if os.path.exists("results/CF_GO"):
            return main()
        time.sleep(20)
    stamp("no go file within 60 min; exiting")


def main():
    stamp("main start")
    C = json.load(open(f"{RES}/candidates.json"))
    N, lst = C["N"], C["list"]
    slots = {i: lst[i] for i in range(N)}
    used = set(range(N))
    replaced, failed = [], []
    ref_ok = {}

    def nxt(fam):
        for j, c in enumerate(lst):
            if j not in used and c["family"] == fam:
                used.add(j)
                return c
        return None

    with cf.ThreadPoolExecutor(PREP_PAR) as pex, cf.ThreadPoolExecutor(REF_PAR) as rex:
        fut = {pex.submit(prep, slots[i]["name"]): ("prep", i, slots[i]) for i in slots}
        while fut:
            done, _ = cf.wait(list(fut), return_when=cf.FIRST_COMPLETED)
            for f in done:
                stage, i, c = fut.pop(f)
                ok, sec = f.result()
                stamp(f"{stage} {c['name']} ok={ok} {sec:.0f}s")
                if ok and stage == "prep":
                    fut[rex.submit(refcurve, c["name"], c["ref_model"])] = ("ref", i, c)
                elif ok:
                    ref_ok[i] = c
                else:
                    failed.append(dict(name=c["name"], stage=stage))
                    r = nxt(c["family"])
                    if r is None:
                        slots.pop(i)
                        continue
                    replaced.append(dict(slot=i, failed=c["name"], replacement=r["name"]))
                    slots[i] = r
                    fut[pex.submit(prep, r["name"])] = ("prep", i, r)
    test = [ref_ok[i] for i in sorted(ref_ok)]
    json.dump(dict(N=N, test=test, replaced=replaced, failed=failed, written_utc=now()), open(f"{RES}/corpora.json", "w"), indent=1)
    stamp("refcurves done")
    rc = sh(f"{PY} -u scripts/curve_forecast/forecast.py", f"{LOG}/forecast.log")
    if rc != 0:
        raise SystemExit("forecast failed")
    stamp("forecast written")
    sh(f"sha256sum {RES}/forecast.json > {LOG}/forecast.sha256; ls -l --time-style=full-iso {RES}/forecast.json >> {LOG}/forecast.sha256", f"{LOG}/sha.log")
    print(open(f"{LOG}/forecast.sha256").read(), flush=True)
    with cf.ThreadPoolExecutor(PREP_PAR) as ex:
        list(ex.map(lambda c: sh(f"[ -f data/cf/{c['name']}/tok/tok_bytes_test.npy ] || {PY} text/tok_family.py --data data/cf/{c['name']} "
                                 f"--out data/cf/{c['name']}/tok --names bytes --seed 0 --enc_bytes {ENC} --encode_test", f"{LOG}/tok_test_{c['name']}.log"), test))
    stamp("test splits encoded")
    jobs = []
    for j, c in enumerate(test):
        for k in (25, 23, 20, 19):
            jobs.append((f"cf_{c['name']}", f"data/cf/{c['name']}/tok", k, 0))
        if j < 5:
            jobs.append((f"cf_{c['name']}", f"data/cf/{c['name']}/tok", 25, 1))
    json.dump(jobs, open(f"{RES}/jobs.json", "w"))
    with cf.ThreadPoolExecutor(PAR) as ex:
        res = list(ex.map(lambda a: run_train(*a), jobs))
    stamp(f"training done {sum(res)}/{len(res)}")
    open("results/CF_DONE", "w").write(now())


def resume():
    """Restart on a fresh pod: reuse corpora.json, the reference curves and forecast.json from the first run (never rewritten);
    prepare data only for corpora with unfinished runs, then train the missing runs."""
    stamp("resume start")
    assert os.path.exists(f"{RES}/forecast.json"), "no forecast.json: resume needs the original forecast"
    test = json.load(open(f"{RES}/corpora.json"))["test"]
    jobs = []
    for j, c in enumerate(test):
        for k in (25, 23, 20, 19):
            jobs.append((f"cf_{c['name']}", f"data/cf/{c['name']}/tok", k, 0))
        if j < 5:
            jobs.append((f"cf_{c['name']}", f"data/cf/{c['name']}/tok", 25, 1))
    todo = [a for a in jobs if not os.path.exists(f"results/text_runs/{a[0]}/bytes_b{a[2]}_s{a[3]}.json")]
    need = [c for c in test if any(a[0] == f"cf_{c['name']}" for a in todo)]
    stamp(f"resume: {len(todo)} of {len(jobs)} runs left, {len(need)} corpora to prepare")
    with cf.ThreadPoolExecutor(PREP_PAR) as ex:
        res = list(ex.map(lambda c: prep(c["name"], ng=False), need))
    stamp(f"resume prep ok {sum(r[0] for r in res)}/{len(res)}")
    with cf.ThreadPoolExecutor(PREP_PAR) as ex:
        list(ex.map(lambda c: sh(f"[ -f data/cf/{c['name']}/tok/tok_bytes_test.npy ] || {PY} text/tok_family.py --data data/cf/{c['name']} "
                                 f"--out data/cf/{c['name']}/tok --names bytes --seed 0 --enc_bytes {ENC} --encode_test", f"{LOG}/tok_test_{c['name']}.log"), need))
    stamp("resume test splits encoded")
    with cf.ThreadPoolExecutor(PAR) as ex:
        res = list(ex.map(lambda a: run_train(*a), todo))
    stamp(f"training done {sum(res)}/{len(res)} (resume)")
    open("results/CF_DONE", "w").write(now())


if __name__ == "__main__":
    os.makedirs(LOG, exist_ok=True)
    try:
        {"smoke": smoke, "resume": resume}.get(sys.argv[1], main)()
    except BaseException as e:  # noqa: BLE001
        open("results/CF_FAILED", "w").write(f"{now()} {e!r}\n")
        raise
