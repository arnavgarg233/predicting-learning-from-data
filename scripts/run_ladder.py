"""Run the data-budget ladder sequentially, skipping runs that already have results."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nwins", type=int, nargs="+", default=[2 ** k for k in range(4, 15)])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--bases", nargs="+", default=["X", "Z"])
    ap.add_argument("--extra", default="")
    args = ap.parse_args()
    for seed in args.seeds:
        for nwin in args.nwins:
            for b in args.bases:
                out = ROOT / "results" / "runs" / args.tag / f"{b}_n{nwin}_s{seed}.json"
                if out.exists():
                    continue
                cmd = [sys.executable, "-u", str(ROOT / "scripts" / "train.py"), "--data", args.data,
                       "--basis", b, "--nwin", str(nwin), "--seed", str(seed), "--out", str(out)]
                cmd += args.extra.split()
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
