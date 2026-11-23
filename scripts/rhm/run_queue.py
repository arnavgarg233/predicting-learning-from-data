"""Training queue for the RHM test with a memory watchdog.

Jobs "name nwin seed" run PAR at a time on MPS through scripts/rhm/train_rhm.py. A job starts only once its setting's
prediction file exists (predictions are written before training). Every 15 s the watchdog reads memory_pressure; below
--kill_free % free it touches each running job's stop file (train.py saves a checkpoint and exits with code 3) and
no job starts again until free memory is back above --start_free %; stopped jobs resume from their checkpoints.
Finished runs (valid JSON) are skipped, so rerunning resumes the queue. Only processes this script starts are stopped.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/rhm"
PY = sys.executable


def free_pct():
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    return int(re.search(r"free percentage: (\d+)%", out).group(1))


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S %Z')} {msg}"
    print(line, flush=True)
    with open(OUT / "queue.log", "a") as f:
        f.write(line + "\n")


def done(out):
    try:
        return "test_kl_bits" in json.loads(Path(out).read_text())
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="+", default=["v8m2", "v4m2", "v8m4", "v16m8"])
    ap.add_argument("--nwins", nargs="+", type=int, default=[4096, 1024, 256, 64, 16])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--par", type=int, default=1)
    ap.add_argument("--kill_free", type=int, default=18)
    ap.add_argument("--start_free", type=int, default=25)
    ap.add_argument("--extra", default="--pos abs --bs 64 --max_steps 12000 --threads 2")
    a = ap.parse_args()
    jobs = [(nm, n, s) for nm in a.names for s in a.seeds for n in a.nwins]
    env = dict(os.environ, PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.2", PYTORCH_MPS_LOW_WATERMARK_RATIO="0.15")
    pending = [j for j in jobs if not done(OUT / f"runs/{j[0]}/S_n{j[1]}_s{j[2]}.json")]
    log(f"queue start: {len(pending)} of {len(jobs)} jobs pending")
    running = {}  # pid -> (job, Popen, out, resume)
    resume = set()
    paused = False
    while pending or running:
        fp = free_pct()
        if fp < a.kill_free and running:
            log(f"free memory {fp}% < {a.kill_free}%: stopping {len(running)} job(s) with checkpoints")
            for pid, (job, p, out) in running.items():
                Path(str(out) + ".stop").touch()
            paused = True
        if paused and fp >= a.start_free:
            paused = False
        for pid in list(running):
            job, p, out = running[pid]
            rc = p.poll()
            if rc is None:
                continue
            del running[pid]
            if rc == 0 and done(out):
                log(f"done {job} ")
            elif rc == 3:
                log(f"stopped {job} (watchdog); will resume")
                resume.add(job)
                pending.insert(0, job)
            else:
                log(f"FAILED {job} rc={rc}")
        if not paused and fp >= a.start_free:
            for job in list(pending):
                if len(running) >= a.par:
                    break
                nm, n, s = job
                pred = OUT / f"pred/{nm}.json"
                if not pred.exists():
                    continue
                pending.remove(job)
                out = OUT / f"runs/{nm}/S_n{n}_s{s}.json"
                out.parent.mkdir(parents=True, exist_ok=True)
                cmd = [PY, "-u", str(ROOT / "scripts/rhm/train_rhm.py"), "--data", str(ROOT / f"data/rhm/{nm}.npz"),
                       "--basis", "S", "--nwin", str(n), "--seed", str(s), "--device", "mps", "--out", str(out)] + a.extra.split()
                if job in resume or Path(str(out) + ".abort.pt").exists():
                    cmd.append("--resume")
                lf = open(OUT / f"runs/{nm}/train_n{n}_s{s}.log", "a")
                p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
                running[p.pid] = (job, p, out)
                log(f"start {job} pid {p.pid} free {fp}%{' (resume)' if job in resume else ''}")
                time.sleep(5)
        if not running and pending and not paused and not (OUT / f"pred/{pending[0][0]}.json").exists():
            nm = pending[0][0]
            log(f"make_data {nm} (serial, nothing else of ours running)")
            with open(OUT / "make_data.log", "a") as lf:
                rc = subprocess.call([PY, "-u", str(ROOT / "scripts/rhm/make_data.py"), "--name", nm], stdout=lf,
                                     stderr=subprocess.STDOUT, cwd=str(ROOT))
            if rc != 0:
                log(f"make_data {nm} FAILED rc={rc}; dropping its jobs")
                pending = [j for j in pending if j[0] != nm]
            continue
        time.sleep(15)
    log("queue finished")
    (OUT / "QUEUE_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S %Z") + "\n")


if __name__ == "__main__":
    sys.exit(main())
