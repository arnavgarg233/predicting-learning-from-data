"""Stage-2 pod orchestrator of the curve-forecast test (docs/plan/curve_forecast_vs_pilots_2026-10-17.md, "Decisive design: two
stages, N = 80": candidates 31-80), run from /workspace/k125 on its own pod. Reuses scripts/curve_forecast/pod_main.py's steps
(prep commands, reference curves, training commands) and the locked forecast.py unchanged; differs from pod_main.main only in
the candidate file, a longer preparation timeout (code shards are downloaded), no seed-1 runs, and the exclusion of names
stage 1 used.

    python -u scripts/cf_stage2_main.py smoke    # timing, then waits for results/S2_GO and runs main in the same process
    python -u scripts/cf_stage2_main.py main

smoke  environment; first two documents of every stage-2 source (logs/source_check.json); learner timing on FineWeb-Edu bytes at
       2^25 with S2_PAR processes, 300 and 1200 steps (tags s2smoke_300, s2smoke_1200; no test-corpus learner run);
       preparation and reference curve of the first wiki and the first English candidate, each curve run alone; logs/smoke2.json.
main   results/curve_forecast/stage2_candidates.json ({N, list, pools}: positions 26-80 after the stage-2 list fix); extracts
       the first 20 all-all code shards (shared cache) while preparing every corpus and its reference curve; failures are
       replaced in list order from the pools (code and wiki -> next unused wiki_ext; English -> english_ext, else wiki_ext),
       in rounds until none is pending; then waits for results/S2_EXCLUDE (names stage 1 used), whose slots are replaced
       the same way; writes corpora.json (with source_family and each code language's all-all reach), runs forecast.py and logs its sha256 before any test-corpus training, encodes the test splits,
       then trains seed 0 at 2^25, 2^23, 2^20, 2^19 per corpus in list order with S2_PAR processes. results/S2_DONE.
results/S2_GO may hold JSON overrides {"PAR": , "PREP_PAR": , "REF_PAR": }.
"""
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "scripts/curve_forecast")
import pod_main as PM  # noqa: E402

RES, LOG, PY = PM.RES, PM.LOG, PM.PY
CAND = os.environ.get("S2_CAND", f"{RES}/stage2_candidates.json")
CFG = dict(PAR=int(os.environ.get("S2_PAR", "4")), PREP_PAR=int(os.environ.get("S2_PREP_PAR", "4")),
           REF_PAR=int(os.environ.get("S2_REF_PAR", "1")))
PREP_TIMEOUT = 4 * 3600


def prep(c):
    """pod_main.prep with a longer timeout for prep_corpus.py (same commands)."""
    d = f"data/cf/{c}"
    if os.path.exists(f"{d}/PREP_OK"):
        return True, 0.0
    t0 = time.time()
    rc = PM.sh(f"{PY} -u scripts/prep_corpus.py --corpus {c} --out {d} --train_bytes {PM.TRAIN_Q} --enc_bytes {PM.ENC}",
               f"{LOG}/prep_{c}.log", PREP_TIMEOUT)
    if rc == 0:
        rc = PM.sh(f"{PY} text/tok_family.py --data {d} --out {d}/tok --names bytes --seed 0 --enc_bytes {PM.ENC} --verify",
                   f"{LOG}/tok_verify_{c}.log")
    if rc == 0:
        rc = PM.sh(f"{PY} -c \"import sys; sys.path.insert(0, 'scripts'); import forecast_multicorpus as FM; "
                   f"FM.ng_curve('{c}', '{d}/tok', 23, '{RES}')\"", f"{LOG}/ngram_{c}.log", 3600)
    sec = time.time() - t0
    os.makedirs(d, exist_ok=True)
    open(f"{d}/{'PREP_OK' if rc == 0 else 'PREP_FAIL'}", "w").write(f"{sec:.0f}\n")
    return rc == 0, sec


def source_check(names):
    code = ("import sys, itertools, json; sys.path.insert(0, 'scripts'); import prep_corpus as P; "
            "d = list(itertools.islice(P.SOURCES['{n}'](), 2)); print(json.dumps([len(t) for _, t in d])); sys.stdout.flush(); "
            "import os; os._exit(0)")
    def one(n):
        t0 = time.time()
        try:
            r = subprocess.run([PY, "-c", code.format(n=n)], capture_output=True, text=True, timeout=1200)
            return n, dict(rc=r.returncode, out=r.stdout.strip()[-200:], err=r.stderr.strip()[-400:], sec=round(time.time() - t0))
        except subprocess.TimeoutExpired:
            return n, dict(rc=124, sec=round(time.time() - t0))
    with cf.ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, names))


def timing(steps, P):
    tag = f"s2smoke_{steps}"
    cmd = PM.train_cmd(tag, "data/bfine/tok", 25, 0).replace("--max_steps 60000", f"--max_steps {steps}")
    t0 = time.time()
    with cf.ThreadPoolExecutor(P) as ex:
        list(ex.map(lambda s: PM.sh(cmd.replace("--seeds 0", f"--seeds {s}"), f"{LOG}/{tag}_s{s}.log"), range(P)))
    wall = time.time() - t0
    runs = [json.load(open(f"results/text_runs/{tag}/bytes_b25_s{s}.json")) for s in range(P)
            if os.path.exists(f"results/text_runs/{tag}/bytes_b25_s{s}.json")]
    return dict(steps=steps, P=P, wall=wall, n_ok=len(runs), run_seconds=[r["seconds"] for r in runs])


def smoke():
    PM.stamp("s2 smoke start")
    PM.sh(f"{PY} -c \"import torch, transformers, datasets; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
          f"torch.cuda.get_device_name(0)); print('transformers', transformers.__version__, 'datasets', datasets.__version__)\"; "
          f"nproc; free -g; df -h /workspace", f"{LOG}/smoke_env.txt")
    gpu = subprocess.Popen(["bash", "-c", f"nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 15 > {LOG}/gpu.csv"])
    C = json.load(open(CAND))
    lst = C["list"]
    out = dict(cfg=CFG)
    with cf.ThreadPoolExecutor(3) as ex:
        sc = ex.submit(source_check, [c["name"] for c in lst + C["pools"]["wiki_ext"] if c["family"] != "code"])
        sys.path.insert(0, "scripts")
        import prep_corpus as P
        t0 = time.time()
        has_code = any(c["family"] == "code" for c in lst)  # stage 3 (wiki only) skips the code-shard timing
        sh0 = ex.submit(P.ghc_extract, 0) if has_code else None
        first = [next(c for c in lst if c["family"] == "wiki")] + [c for c in lst if c["family"] == "english"][:1]
        pf = {c["name"]: ex.submit(prep, c["name"]) for c in first}
        out["timing"] = [timing(1200, CFG["PAR"]), timing(300, CFG["PAR"])]
        a, b = out["timing"]
        out["sec_per_step_per_process"] = (a["wall"] - b["wall"]) / (a["steps"] - b["steps"])
        out["sec_per_step_throughput"] = out["sec_per_step_per_process"] / CFG["PAR"]
        out["prep"] = {n: f.result() for n, f in pf.items()}
        out["refcurve"] = {}
        for c in first:
            if out["prep"][c["name"]][0]:
                out["refcurve"][c["name"]] = (c["ref_model"],) + PM.refcurve(c["name"], c["ref_model"])
        if sh0 is not None:
            sh0.result()
            out["ghc_shard0_seconds"] = time.time() - t0
        out["source_check"] = sc.result()
    gpu.terminate()
    json.dump(out, open(f"{LOG}/smoke2.json", "w"), indent=1)
    PM.stamp("s2 smoke done")
    open("results/S2_SMOKE_DONE", "w").write(PM.now())
    for _ in range(360):                      # wait up to 2 h for the go file
        if os.path.exists("results/S2_GO"):
            txt = open("results/S2_GO").read().strip()
            CFG.update(json.loads(txt) if txt.startswith("{") else {})
            return main()
        time.sleep(20)
    PM.stamp("no go file within 2 h; exiting")


def code_reach():
    """Bytes of each all-all language within the first GHC_MAX_SHARDS shards (UTF-8 bytes of the code column)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    sys.path.insert(0, "scripts")
    import prep_corpus as P
    out = {}
    for lang, v in P.GHC_FULL.items():
        n, rows = 0, 0
        for i in range(P.GHC_MAX_SHARDS):
            f = P.GHC_CACHE / f"{i:05d}" / f"{v}.parquet"
            if f.exists():
                t = pq.read_table(f, columns=["code"])
                n += int(pc.sum(pc.binary_length(t["code"])).as_py() or 0)
                rows += t.num_rows
        out["code_" + lang.lower()] = dict(language=v, shards=P.GHC_MAX_SHARDS, files=rows, bytes=n)
    return out


def main():
    PM.stamp(f"s2 main start {CFG}")
    C = json.load(open(CAND))
    N, lst, pools = C["N"], C["list"], C["pools"]
    used = {c["name"] for c in lst}
    slots = {c["position"]: dict(c) for c in lst}
    replaced, failed, ok = [], [], {}
    sys.path.insert(0, "scripts")
    import prep_corpus as P
    import threading
    psem, rsem = threading.Semaphore(CFG["PREP_PAR"]), threading.Semaphore(CFG["REF_PAR"])

    def unit(c):
        with psem:
            good, sec = prep(c["name"])
        PM.stamp(f"prep {c['name']} ok={good} {sec:.0f}s")
        if not good:
            return False, "prep"
        with rsem:
            good, sec = PM.refcurve(c["name"], c["ref_model"])
        PM.stamp(f"ref {c['name']} ok={good} {sec:.0f}s")
        return good, "ref"

    def take(pool):
        for c in pools.get(pool, []):
            if c["name"] not in used:
                used.add(c["name"])
                return dict(c)
        return None

    def replace(pos, c, why):
        failed.append(dict(position=pos, name=c["name"], stage=why))
        r = None
        if c["family"] == "english":
            r = take("english_ext")
        if r is None:
            r = take("wiki_ext")
        if r is None:
            slots.pop(pos)
            replaced.append(dict(position=pos, failed=c["name"], replacement=None))
            return None
        r["position"] = pos
        r["replaces"] = c["name"]
        replaced.append(dict(position=pos, failed=c["name"], replacement=r["name"], reason=why))
        slots[pos] = r
        return pos

    ex = cf.ThreadPoolExecutor(CFG["PREP_PAR"] + CFG["REF_PAR"] + 2)
    shx = cf.ThreadPoolExecutor(4)
    shard_fut = [shx.submit(P.ghc_extract, i) for i in range(P.GHC_MAX_SHARDS)]   # shared cache for the code languages
    todo = sorted(slots)
    excl_done = False
    while True:
        fut = {pos: ex.submit(unit, slots[pos]) for pos in todo}
        bad = []
        for pos in sorted(fut):
            good, stage = fut[pos].result()
            if good:
                ok[pos] = slots[pos]
            else:
                bad.append((pos, stage))
        todo = [q for q in (replace(pos, slots[pos], stage) for pos, stage in sorted(bad)) if q is not None]
        if todo:
            continue
        if not excl_done:
            PM.stamp("s2 refcurves done; waiting for S2_EXCLUDE")
            while not os.path.exists("results/S2_EXCLUDE"):
                time.sleep(30)
            excl = set(json.load(open("results/S2_EXCLUDE")))
            used |= excl
            excl_done = True
            hit = sorted(pos for pos in ok if ok[pos]["name"] in excl)
            for pos in hit:
                ok.pop(pos)
            todo = [q for q in (replace(pos, slots[pos], "used by stage 1") for pos in hit) if q is not None]
            if todo:
                continue
        break
    for f in shard_fut:
        f.result()
    reach = code_reach()
    test = [ok[pos] for pos in sorted(ok)]
    for c in test:
        c["source_family"] = dict(wiki="wikipedia", code="code", english="english-text").get(c["family"], "other")
    rec = dict(N=N, stage=2, test=test, replaced=replaced, failed=failed, code_all_all_reach=reach,
               code_outcome={c["name"]: ("used" if c["name"] in {t["name"] for t in test} else "fell back to Wikipedia")
                             for c in lst if c["family"] == "code"},
               excluded_used_by_stage1=sorted(excl), written_utc=PM.now())
    json.dump(rec, open(f"{RES}/corpora.json", "w"), indent=1)
    PM.stamp(f"s2 corpora.json written: {len(test)} corpora")
    rc = PM.sh(f"{PY} -u scripts/curve_forecast/forecast.py", f"{LOG}/forecast.log")
    if rc != 0:
        raise SystemExit("forecast failed")
    PM.stamp("s2 forecast written")
    PM.sh(f"sha256sum {RES}/forecast.json > {LOG}/forecast.sha256; ls -l --time-style=full-iso {RES}/forecast.json >> {LOG}/forecast.sha256",
          f"{LOG}/sha.log")
    print(open(f"{LOG}/forecast.sha256").read(), flush=True)
    open("results/S2_FORECAST_DONE", "w").write(PM.now())
    while not os.path.exists("results/S2_TRAIN"):    # training starts only after the forecast hash is recorded locally
        time.sleep(15)
    with cf.ThreadPoolExecutor(CFG["PREP_PAR"]) as tex:
        list(tex.map(lambda c: PM.sh(f"[ -f data/cf/{c['name']}/tok/tok_bytes_test.npy ] || {PY} text/tok_family.py --data data/cf/{c['name']} "
                                     f"--out data/cf/{c['name']}/tok --names bytes --seed 0 --enc_bytes {PM.ENC} --encode_test",
                                     f"{LOG}/tok_test_{c['name']}.log"), test))
    PM.stamp("s2 test splits encoded")
    jobs = [(f"cf_{c['name']}", f"data/cf/{c['name']}/tok", k, 0) for c in test for k in (25, 23, 20, 19)]
    json.dump(jobs, open(f"{RES}/jobs.json", "w"))
    with cf.ThreadPoolExecutor(CFG["PAR"]) as tex:
        res = list(tex.map(lambda a: PM.run_train(*a), jobs))
    PM.stamp(f"s2 training done {sum(res)}/{len(res)}")
    open("results/S2_DONE", "w").write(PM.now())


if __name__ == "__main__":
    os.makedirs(LOG, exist_ok=True)
    try:
        smoke() if sys.argv[1] == "smoke" else main()
    except BaseException as e:  # noqa: BLE001
        open("results/S2_FAILED", "w").write(f"{PM.now()} {e!r}\n")
        raise
