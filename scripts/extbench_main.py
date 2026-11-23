"""Pod orchestrator of the external benchmark (docs/plan/referee_fixes_cost_2026-10-28.md section 3; draw in
results/extbench_2026-10-28/draw.json), run from /workspace/k125 on each of the three pods. A copy of new_corpora_main.py with
this run's candidate file, flag names, the pod's own spares, and a second seed at 2^25.

    python -u scripts/extbench_main.py

1. Prepares every corpus of results/curve_forecast/eb_candidates.json (EB_PREP_PAR at a time) and runs its reference curve
   (one at a time on the GPU). A corpus that fails preparation or its curve is replaced by the pod's next unused spare of its
   family (eb_candidates.json "pools"), else dropped.
2. Writes corpora.json and runs the unchanged forecast.py for all corpora (results/curve_forecast/forecast.json), then
   results/EB_FORECAST_DONE.
3. Waits for results/EB_TRAIN, which the Mac guard writes after it has pulled forecast.json, recorded it with the local date and
   written forecaster (ii) from it.
4. Encodes the test splits and trains, per corpus in list order, seed 0 at 2^25, 2^23, 2^20, 2^19 and seed 1 at 2^25, with
   EB_PAR processes. Before each run it reads results/EB_LIMIT (if present: the number of corpora, in list order, that will be
   trained) so the Mac can cut the tail of the list if the cost projection requires it. results/EB_DONE at the end,
   results/EB_FAILED on an exception.
"""
import concurrent.futures as cf
import json
import os
import sys
import threading
import time

sys.path.insert(0, "scripts/curve_forecast")
sys.path.insert(0, "scripts")
import pod_main as PM  # noqa: E402
import cf_stage2_main as S2  # noqa: E402

RES, LOG, PY = PM.RES, PM.LOG, PM.PY
CAND = f"{RES}/eb_candidates.json"
PAR, PREP_PAR = int(os.environ.get("EB_PAR", "4")), int(os.environ.get("EB_PREP_PAR", "4"))
RUNS = ((25, 0), (23, 0), (20, 0), (19, 0), (25, 1))
S2.PREP_TIMEOUT = 5400       # 90 min per preparation (the forward test needed 1-4 min; a stalled stream is replaced instead of waited on)


def limit():
    try:
        return int(open("results/EB_LIMIT").read().strip())
    except (OSError, ValueError):
        return 10 ** 6


def main():
    PM.stamp(f"eb main start PAR={PAR} PREP_PAR={PREP_PAR}")
    C = json.load(open(CAND))
    lst, pools = C["list"], C["pools"]
    used = {c["name"] for c in lst}
    slots = {c["position"]: dict(c) for c in lst}
    replaced, failed, ok = [], [], {}
    psem, rsem = threading.Semaphore(PREP_PAR), threading.Semaphore(1)

    def unit(c):
        with psem:
            good, sec = S2.prep(c["name"])
        PM.stamp(f"prep {c['name']} ok={good} {sec:.0f}s")
        if not good:
            return False, "prep"
        with rsem:
            good, sec = PM.refcurve(c["name"], c["ref_model"])
        PM.stamp(f"ref {c['name']} ok={good} {sec:.0f}s")
        return good, "ref"

    def replace(pos, c, why):
        failed.append(dict(position=pos, name=c["name"], stage=why))
        r = next((dict(x) for x in pools.get(c["family"], []) if x["name"] not in used), None)
        if r is None:
            slots.pop(pos)
            replaced.append(dict(position=pos, failed=c["name"], replacement=None, reason=why))
            return None
        used.add(r["name"])
        r["position"], r["replaces"] = pos, c["name"]
        replaced.append(dict(position=pos, failed=c["name"], replacement=r["name"], reason=why))
        slots[pos] = r
        return pos

    ex = cf.ThreadPoolExecutor(PREP_PAR + 2)
    todo = sorted(slots)
    while todo:
        fut = {pos: ex.submit(unit, slots[pos]) for pos in todo}
        bad = []
        for pos in sorted(fut):
            good, stage = fut[pos].result()
            if good:
                ok[pos] = slots[pos]
            else:
                bad.append((pos, stage))
        todo = [q for q in (replace(pos, slots[pos], stage) for pos, stage in sorted(bad)) if q is not None]
    test = [ok[pos] for pos in sorted(ok)]
    json.dump(dict(N=len(test), test=test, replaced=replaced, failed=failed, written_utc=PM.now()), open(f"{RES}/corpora.json", "w"), indent=1)
    PM.stamp(f"eb corpora.json written: {len(test)} corpora")
    rc = PM.sh(f"{PY} -u scripts/curve_forecast/forecast.py", f"{LOG}/forecast.log")
    if rc != 0:
        raise SystemExit("forecast failed")
    PM.stamp("eb forecast written")
    open("results/EB_FORECAST_DONE", "w").write(PM.now())
    while not os.path.exists("results/EB_TRAIN"):    # released by the Mac after it has both forecasts with their date
        time.sleep(15)
    PM.stamp("eb training released")
    with cf.ThreadPoolExecutor(PREP_PAR) as tex:
        list(tex.map(lambda c: PM.sh(f"[ -f data/cf/{c['name']}/tok/tok_bytes_test.npy ] || {PY} text/tok_family.py --data data/cf/{c['name']} "
                                     f"--out data/cf/{c['name']}/tok --names bytes --seed 0 --enc_bytes {PM.ENC} --encode_test",
                                     f"{LOG}/tok_test_{c['name']}.log"), test))
    PM.stamp("eb test splits encoded")
    jobs = [(j, f"cf_{c['name']}", f"data/cf/{c['name']}/tok", k, s) for j, c in enumerate(test) for k, s in RUNS]
    json.dump(jobs, open(f"{RES}/jobs.json", "w"))

    def run(a):
        j, tag, d, k, s = a
        if j >= limit():
            return None
        t0 = time.time()
        r = PM.run_train(tag, d, k, s)
        PM.stamp(f"train {tag} b{k} s{s} ok={r} {time.time() - t0:.0f}s")
        return r

    with cf.ThreadPoolExecutor(PAR) as tex:
        res = list(tex.map(run, jobs))
    PM.stamp(f"eb training done {sum(1 for r in res if r)}/{sum(1 for r in res if r is not None)} (limit {limit()})")
    open("results/EB_DONE", "w").write(PM.now())


if __name__ == "__main__":
    os.makedirs(LOG, exist_ok=True)
    try:
        main()
    except BaseException as e:  # noqa: BLE001
        open("results/EB_FAILED", "w").write(f"{PM.now()} {e!r}\n")
        raise
